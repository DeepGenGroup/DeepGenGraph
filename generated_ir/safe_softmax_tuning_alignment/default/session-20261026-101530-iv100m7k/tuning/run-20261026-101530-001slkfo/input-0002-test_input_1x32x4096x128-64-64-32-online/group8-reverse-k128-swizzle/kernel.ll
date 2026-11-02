; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/group8-reverse-k128-swizzle/input.mlir"

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
  %42 = lshr i64 %35, 2, !dbg !16
  %43 = shl nuw nsw i64 %35, 2, !dbg !16
  %44 = mul nsw i64 %42, -16, !dbg !16
  %45 = add nsw i64 %43, %44, !dbg !16
  %46 = and i32 %34, 1, !dbg !16
  %index.compact114 = zext nneg i32 %46 to i64, !dbg !16
  %47 = or disjoint i64 %45, %index.compact114, !dbg !16
  %48 = add nuw nsw i64 %47, 1, !dbg !16
  %49 = sdiv i64 %48, 2, !dbg !16
  %50 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %51 = getelementptr i8, ptr addrspace(1) %50, i64 64, !dbg !11
  %52 = shl i64 %15, 7, !dbg !15
  %53 = and i64 %52, 3968, !dbg !15
  %54 = shl nsw i64 %18, 12, !dbg !15
  %55 = or disjoint i64 %53, %54, !dbg !15
  %56 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %57 = getelementptr i8, ptr addrspace(1) %56, i64 128, !dbg !11
  %58 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %59 = getelementptr i8, ptr addrspace(1) %58, i64 192, !dbg !11
  br label %60, !dbg !7

60:                                               ; preds = %421, %4
  %61 = phi i64 [ %733, %421 ], [ 0, %4 ], !dbg !7
  %62 = phi <4 x float> [ %616, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <4 x float> [ %624, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi <8 x float> [ %729, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <1 x float> [ %580, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <1 x float> [ %483, %421 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %67 = phi <4 x float> [ %656, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi <4 x float> [ %664, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = phi <8 x float> [ %730, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %70 = phi <4 x float> [ %696, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %71 = phi <4 x float> [ %704, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %72 = phi <8 x float> [ %731, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %73 = phi <4 x float> [ %720, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %74 = phi <4 x float> [ %728, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %75 = phi <8 x float> [ %732, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %76 = icmp slt i64 %61, %40, !dbg !7
  br i1 %76, label %77, label %734, !dbg !7

77:                                               ; preds = %60
  %78 = shl nsw i64 %61, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local137 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local138 = and i64 %address.local137, 1536, !dbg !11
  %address.local139 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local138, !dbg !11
  %address.local140 = ptrtoint ptr addrspace(3) %address.local139 to i64, !dbg !11
  %address.local141 = trunc i64 %address.local140 to i32, !dbg !11
  %79 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !11
  %80 = zext i32 %79 to i64, !dbg !11
  %81 = inttoptr i64 %80 to ptr addrspace(3), !dbg !11
  %address.local142 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local143 = and i64 %address.local142, 8064, !dbg !11
  %address.local144 = lshr i64 %address.local, 3, !dbg !11
  %address.local145 = and i64 %address.local144, 24, !dbg !11
  %address.local146 = or disjoint i64 %address.local143, %address.local145, !dbg !11
  %address.local147 = getelementptr [2 x i8], ptr addrspace(1) %41, i64 %address.local146, !dbg !11
  %82 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %83 = ptrtoint ptr addrspace(1) %address.local147 to i64, !dbg !11
  %84 = sub i64 %83, %82, !dbg !11
  %dma.buffer.offset679 = trunc i64 %84 to i32, !dbg !11
  %85 = trunc i64 %82 to i32, !dbg !11
  %86 = insertelement <4 x i32> poison, i32 %85, i64 0, !dbg !11
  %87 = lshr i64 %82, 32, !dbg !11
  %88 = trunc i64 %87 to i32, !dbg !11
  %89 = insertelement <4 x i32> %86, i32 %88, i64 1, !dbg !11
  %90 = insertelement <4 x i32> %89, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc680 = insertelement <4 x i32> %90, i32 131072, i64 3, !dbg !11
  %91 = ptrtoint ptr addrspace(3) %81 to i32, !dbg !11
  %dma.buffer.lds681 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %91), !dbg !11
  %92 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds681, i32 %dma.buffer.offset679, <4 x i32> %dma.buffer.rsrc680) #10, !dbg !11
  %93 = lshr i64 %61, 13, !dbg !15
  %94 = add nuw nsw i64 %93, %15, !dbg !15
  %95 = lshr i64 %61, 6, !dbg !15
  %96 = and i64 %95, 127, !dbg !15
  %97 = shl i64 %94, 19, !dbg !15
  %98 = and i64 %97, 16252928, !dbg !15
  %99 = shl i64 %95, 19, !dbg !15
  %100 = shl i64 %61, 13, !dbg !15
  %101 = sub i64 %100, %99, !dbg !15
  %102 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %98, !dbg !15
  %103 = getelementptr [2 x i8], ptr addrspace(1) %102, i64 %96, !dbg !15
  %104 = getelementptr [2 x i8], ptr addrspace(1) %103, i64 %101, !dbg !15
  %address.local148 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local138, !dbg !15
  %address.local149 = ptrtoint ptr addrspace(3) %address.local148 to i64, !dbg !15
  %address.local150 = trunc i64 %address.local149 to i32, !dbg !15
  %105 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !15
  %106 = zext i32 %105 to i64, !dbg !15
  %107 = inttoptr i64 %106 to ptr addrspace(3), !dbg !15
  %address.local151 = lshr i64 %address.local, 3, !dbg !15
  %address.local152 = and i64 %address.local151, 24, !dbg !15
  %address.local153 = and i64 %address.local, 59, !dbg !15
  %address.local154 = lshr i64 %address.local, 2, !dbg !15
  %address.local155 = and i64 %address.local154, 4, !dbg !15
  %address.local156 = add nuw nsw i64 %address.local155, %address.local, !dbg !15
  %address.local157 = and i64 %address.local156, 4, !dbg !15
  %address.local158 = or disjoint i64 %address.local153, %address.local157, !dbg !15
  %address.local159 = shl nuw nsw i64 %address.local158, 7, !dbg !15
  %address.local160 = or disjoint i64 %address.local152, %address.local159, !dbg !15
  %address.local161 = getelementptr [2 x i8], ptr addrspace(1) %104, i64 %address.local160, !dbg !15
  %108 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %109 = ptrtoint ptr addrspace(1) %address.local161 to i64, !dbg !15
  %110 = sub i64 %109, %108, !dbg !15
  %dma.buffer.offset676 = trunc i64 %110 to i32, !dbg !15
  %111 = trunc i64 %108 to i32, !dbg !15
  %112 = insertelement <4 x i32> poison, i32 %111, i64 0, !dbg !15
  %113 = lshr i64 %108, 32, !dbg !15
  %114 = trunc i64 %113 to i32, !dbg !15
  %115 = insertelement <4 x i32> %112, i32 %114, i64 1, !dbg !15
  %116 = insertelement <4 x i32> %115, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc677 = insertelement <4 x i32> %116, i32 131072, i64 3, !dbg !15
  %117 = ptrtoint ptr addrspace(3) %107 to i32, !dbg !15
  %dma.buffer.lds678 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %117), !dbg !15
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds678, i32 %dma.buffer.offset676, <4 x i32> %dma.buffer.rsrc677) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local162 = lshr i32 %address.thread, 2, !dbg !12
  %address.local163 = and i32 %address.local162, 4, !dbg !12
  %address.local164 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local165 = and i32 %address.local164, 384, !dbg !12
  %address.local166 = or disjoint i32 %address.local163, %address.local165, !dbg !12
  %address.local167 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local168 = and i32 %address.local167, 120, !dbg !12
  %address.local169 = or disjoint i32 %address.local166, %address.local168, !dbg !12
  %address.local170 = shl nuw nsw i32 %address.thread, 4, !dbg !12
  %address.local171 = and i32 %address.local170, 512, !dbg !12
  %address.local172 = or disjoint i32 %address.local169, %address.local171, !dbg !12
  %address.local173 = zext nneg i32 %address.local172 to i64, !dbg !12
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local173, !dbg !12
  %119 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !12
  %120 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %121 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !12
  %122 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local177 = mul nuw nsw i64 %address.local, 160, !dbg !16
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local177, !dbg !16
  %address.local179 = lshr i64 %address.local, 2, !dbg !16
  %address.local180 = mul nsw i64 %address.local179, -1280, !dbg !16
  %address.local181 = getelementptr i8, ptr addrspace(3) %address.local178, i64 %address.local180, !dbg !16
  %address.local182 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local183 = and i64 %address.local182, 24, !dbg !16
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) %address.local181, i64 %address.local183, !dbg !16
  %address.local185 = shl i64 %address.local, 62, !dbg !16
  %address.local186 = ashr exact i64 %address.local185, 57, !dbg !16
  %address.local187 = and i64 %address.local186, -64, !dbg !16
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) %address.local184, i64 %address.local187, !dbg !16
  %address.local189 = lshr i64 %address.local, 2, !dbg !12
  %address.local190 = and i64 %address.local189, 12, !dbg !12
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) %address.local188, i64 %address.local190, !dbg !16
  %address.local192 = and i64 %address.local, 63, !dbg !12
  %address.local193 = lshr i64 %address.local192, 5, !dbg !12
  %address.local194 = mul nuw nsw i64 %address.local193, 504, !dbg !12
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) %address.local191, i64 %address.local194, !dbg !16
  %123 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !16
  %lds.const.gep622 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !16
  %124 = load <4 x half>, ptr addrspace(3) %lds.const.gep622, align 8, !dbg !16
  %125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %123, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local206 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local207 = and i32 %address.local206, 24, !dbg !16
  %address.local208 = shl nuw nsw i32 %address.thread, 8, !dbg !16
  %address.local209 = and i32 %address.local208, 768, !dbg !16
  %address.local210 = or disjoint i32 %address.local207, %address.local209, !dbg !16
  %address.local211 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local212 = and i32 %address.local211, 96, !dbg !16
  %address.local213 = or disjoint i32 %address.local210, %address.local212, !dbg !16
  %address.local214 = zext nneg i32 %address.local213 to i64, !dbg !16
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %49) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local215 = lshr i32 %address.thread, 1, !dbg !16
  %address.local216 = and i32 %address.local215, 1, !dbg !16
  %address.local217 = zext nneg i32 %address.local216 to i64, !dbg !16
  %address.local218 = add nsw i64 %address.base, %address.local217, !dbg !16
  %address.local219 = shl i64 %address.local218, 6, !dbg !16
  %address.local220 = sub i64 %address.local214, %address.local219, !dbg !16
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local220, !dbg !16
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local190, !dbg !16
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local194, !dbg !16
  %address.local224 = getelementptr i8, ptr addrspace(3) %address.local223, i64 64, !dbg !16
  %126 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !16
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %124, <4 x float> %125) #9, !dbg !16
  %lds.const.gep623 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !16
  %128 = load <4 x half>, ptr addrspace(3) %lds.const.gep623, align 8, !dbg !16
  %lds.const.gep624 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !16
  %129 = load <4 x half>, ptr addrspace(3) %lds.const.gep624, align 8, !dbg !16
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %126, <4 x float> zeroinitializer) #9, !dbg !16
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %128, <4 x float> %130) #9, !dbg !16
  %lds.const.gep625 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !16
  %132 = load <4 x half>, ptr addrspace(3) %lds.const.gep625, align 8, !dbg !16
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !16
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %132, <4 x float> %133) #9, !dbg !16
  %lds.const.gep626 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !16
  %135 = load <4 x half>, ptr addrspace(3) %lds.const.gep626, align 8, !dbg !16
  %lds.const.gep627 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !16
  %136 = load <4 x half>, ptr addrspace(3) %lds.const.gep627, align 8, !dbg !16
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %135, <4 x float> zeroinitializer) #9, !dbg !16
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %136, <4 x float> %137) #9, !dbg !16
  %139 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !11
  %140 = zext i32 %139 to i64, !dbg !11
  %141 = inttoptr i64 %140 to ptr addrspace(3), !dbg !11
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %51, i64 %address.local146, !dbg !11
  %142 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %143 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !11
  %144 = sub i64 %143, %142, !dbg !11
  %dma.buffer.offset673 = trunc i64 %144 to i32, !dbg !11
  %145 = trunc i64 %142 to i32, !dbg !11
  %146 = insertelement <4 x i32> poison, i32 %145, i64 0, !dbg !11
  %147 = lshr i64 %142, 32, !dbg !11
  %148 = trunc i64 %147 to i32, !dbg !11
  %149 = insertelement <4 x i32> %146, i32 %148, i64 1, !dbg !11
  %150 = insertelement <4 x i32> %149, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc674 = insertelement <4 x i32> %150, i32 131072, i64 3, !dbg !11
  %151 = ptrtoint ptr addrspace(3) %141 to i32, !dbg !11
  %dma.buffer.lds675 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %151), !dbg !11
  %152 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds675, i32 %dma.buffer.offset673, <4 x i32> %dma.buffer.rsrc674) #10, !dbg !11
  %153 = add nuw i64 %61, 2048, !dbg !15
  %154 = icmp slt i64 %153, 0, !dbg !15
  %155 = sub nuw i64 -2049, %61, !dbg !15
  %156 = select i1 %154, i64 %155, i64 %153, !dbg !15
  %157 = sdiv i64 %156, 8192, !dbg !15
  %.lobit19 = ashr i64 %153, 63, !dbg !15
  %158 = xor i64 %157, %.lobit19, !dbg !15
  %159 = add nsw i64 %158, %15, !dbg !15
  %160 = add nsw i64 %95, %55, !dbg !15
  %161 = add nsw i64 %160, 32, !dbg !15
  %162 = icmp slt i64 %160, -32, !dbg !15
  %163 = sub nsw i64 -33, %160, !dbg !15
  %164 = select i1 %162, i64 %163, i64 %161, !dbg !15
  %165 = sdiv i64 %164, 128, !dbg !15
  %166 = sext i1 %162 to i64, !dbg !15
  %167 = xor i64 %165, %166, !dbg !15
  %168 = shl nsw i64 %167, 7, !dbg !15
  %169 = sub i64 %55, %168, !dbg !15
  %170 = shl i64 %159, 19, !dbg !15
  %171 = and i64 %170, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %172 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %171, !dbg !15
  %173 = getelementptr [2 x i8], ptr addrspace(1) %172, i64 %95, !dbg !15
  %174 = getelementptr [2 x i8], ptr addrspace(1) %173, i64 %169, !dbg !15
  %175 = getelementptr i8, ptr addrspace(1) %174, i64 64, !dbg !15
  %176 = getelementptr [2 x i8], ptr addrspace(1) %175, i64 %101, !dbg !15
  %177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !15
  %178 = zext i32 %177 to i64, !dbg !15
  %179 = inttoptr i64 %178 to ptr addrspace(3), !dbg !15
  %address.local366 = getelementptr [2 x i8], ptr addrspace(1) %176, i64 %address.local160, !dbg !15
  %180 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %181 = ptrtoint ptr addrspace(1) %address.local366 to i64, !dbg !15
  %182 = sub i64 %181, %180, !dbg !15
  %dma.buffer.offset670 = trunc i64 %182 to i32, !dbg !15
  %183 = trunc i64 %180 to i32, !dbg !15
  %184 = insertelement <4 x i32> poison, i32 %183, i64 0, !dbg !15
  %185 = lshr i64 %180, 32, !dbg !15
  %186 = trunc i64 %185 to i32, !dbg !15
  %187 = insertelement <4 x i32> %184, i32 %186, i64 1, !dbg !15
  %188 = insertelement <4 x i32> %187, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc671 = insertelement <4 x i32> %188, i32 131072, i64 3, !dbg !15
  %189 = ptrtoint ptr addrspace(3) %179 to i32, !dbg !15
  %dma.buffer.lds672 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %189), !dbg !15
  %190 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds672, i32 %dma.buffer.offset670, <4 x i32> %dma.buffer.rsrc671) #10, !dbg !15
  %191 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !12
  %lds.const.gep628 = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !12
  %192 = load <4 x half>, ptr addrspace(3) %lds.const.gep628, align 8, !dbg !12
  %193 = fmul <4 x half> %191, splat (half 0xH3015), !dbg !12
  %194 = fmul <4 x half> %192, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %195 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !16
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %195, <4 x float> %127) #9, !dbg !16
  %lds.const.gep629 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !16
  %197 = load <4 x half>, ptr addrspace(3) %lds.const.gep629, align 8, !dbg !16
  %198 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !16
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %197, <4 x float> %196) #9, !dbg !16
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %198, <4 x float> %131) #9, !dbg !16
  %lds.const.gep630 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !16
  %201 = load <4 x half>, ptr addrspace(3) %lds.const.gep630, align 8, !dbg !16
  %lds.const.gep631 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !16
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep631, align 8, !dbg !16
  %203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %201, <4 x float> %200) #9, !dbg !16
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %202, <4 x float> %134) #9, !dbg !16
  %lds.const.gep632 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !16
  %205 = load <4 x half>, ptr addrspace(3) %lds.const.gep632, align 8, !dbg !16
  %lds.const.gep633 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !16
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep633, align 8, !dbg !16
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %205, <4 x float> %204) #9, !dbg !16
  %lds.const.gep634 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !16
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep634, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %193, <4 x half> %206, <4 x float> %138) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %194, <4 x half> %208, <4 x float> %209) #9, !dbg !16
  %211 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !11
  %212 = zext i32 %211 to i64, !dbg !11
  %213 = inttoptr i64 %212 to ptr addrspace(3), !dbg !11
  %address.local367 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local146, !dbg !11
  %214 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %215 = ptrtoint ptr addrspace(1) %address.local367 to i64, !dbg !11
  %216 = sub i64 %215, %214, !dbg !11
  %dma.buffer.offset667 = trunc i64 %216 to i32, !dbg !11
  %217 = trunc i64 %214 to i32, !dbg !11
  %218 = insertelement <4 x i32> poison, i32 %217, i64 0, !dbg !11
  %219 = lshr i64 %214, 32, !dbg !11
  %220 = trunc i64 %219 to i32, !dbg !11
  %221 = insertelement <4 x i32> %218, i32 %220, i64 1, !dbg !11
  %222 = insertelement <4 x i32> %221, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc668 = insertelement <4 x i32> %222, i32 131072, i64 3, !dbg !11
  %223 = ptrtoint ptr addrspace(3) %213 to i32, !dbg !11
  %dma.buffer.lds669 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %223), !dbg !11
  %224 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds669, i32 %dma.buffer.offset667, <4 x i32> %dma.buffer.rsrc668) #10, !dbg !11
  %225 = add nuw i64 %61, 4096, !dbg !15
  %226 = icmp slt i64 %225, 0, !dbg !15
  %227 = sub nuw i64 -4097, %61, !dbg !15
  %228 = select i1 %226, i64 %227, i64 %225, !dbg !15
  %229 = sdiv i64 %228, 8192, !dbg !15
  %.lobit20 = ashr i64 %225, 63, !dbg !15
  %230 = xor i64 %229, %.lobit20, !dbg !15
  %231 = add nsw i64 %230, %15, !dbg !15
  %232 = add nsw i64 %160, 64, !dbg !15
  %233 = icmp slt i64 %160, -64, !dbg !15
  %234 = sub nsw i64 -65, %160, !dbg !15
  %235 = select i1 %233, i64 %234, i64 %232, !dbg !15
  %236 = sdiv i64 %235, 128, !dbg !15
  %237 = sext i1 %233 to i64, !dbg !15
  %238 = xor i64 %236, %237, !dbg !15
  %239 = shl nsw i64 %238, 7, !dbg !15
  %240 = sub i64 %55, %239, !dbg !15
  %241 = shl i64 %231, 19, !dbg !15
  %242 = and i64 %241, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %243 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %242, !dbg !15
  %244 = getelementptr [2 x i8], ptr addrspace(1) %243, i64 %95, !dbg !15
  %245 = getelementptr [2 x i8], ptr addrspace(1) %244, i64 %240, !dbg !15
  %246 = getelementptr i8, ptr addrspace(1) %245, i64 128, !dbg !15
  %247 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %101, !dbg !15
  %248 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !15
  %249 = zext i32 %248 to i64, !dbg !15
  %250 = inttoptr i64 %249 to ptr addrspace(3), !dbg !15
  %address.local368 = getelementptr [2 x i8], ptr addrspace(1) %247, i64 %address.local160, !dbg !15
  %251 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %252 = ptrtoint ptr addrspace(1) %address.local368 to i64, !dbg !15
  %253 = sub i64 %252, %251, !dbg !15
  %dma.buffer.offset664 = trunc i64 %253 to i32, !dbg !15
  %254 = trunc i64 %251 to i32, !dbg !15
  %255 = insertelement <4 x i32> poison, i32 %254, i64 0, !dbg !15
  %256 = lshr i64 %251, 32, !dbg !15
  %257 = trunc i64 %256 to i32, !dbg !15
  %258 = insertelement <4 x i32> %255, i32 %257, i64 1, !dbg !15
  %259 = insertelement <4 x i32> %258, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc665 = insertelement <4 x i32> %259, i32 131072, i64 3, !dbg !15
  %260 = ptrtoint ptr addrspace(3) %250 to i32, !dbg !15
  %dma.buffer.lds666 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %260), !dbg !15
  %261 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds666, i32 %dma.buffer.offset664, <4 x i32> %dma.buffer.rsrc665) #10, !dbg !15
  %262 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !12
  %lds.const.gep635 = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !12
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep635, align 8, !dbg !12
  %264 = fmul <4 x half> %262, splat (half 0xH3015), !dbg !12
  %265 = fmul <4 x half> %263, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %266 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !16
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %266, <4 x float> %199) #9, !dbg !16
  %lds.const.gep636 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !16
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep636, align 8, !dbg !16
  %269 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !16
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %265, <4 x half> %268, <4 x float> %267) #9, !dbg !16
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %269, <4 x float> %203) #9, !dbg !16
  %lds.const.gep637 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !16
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep637, align 8, !dbg !16
  %lds.const.gep638 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep638, align 8, !dbg !16
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %265, <4 x half> %272, <4 x float> %271) #9, !dbg !16
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %273, <4 x float> %207) #9, !dbg !16
  %lds.const.gep639 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !16
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep639, align 8, !dbg !16
  %lds.const.gep640 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !16
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep640, align 8, !dbg !16
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %265, <4 x half> %276, <4 x float> %275) #9, !dbg !16
  %lds.const.gep641 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !16
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep641, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %264, <4 x half> %277, <4 x float> %210) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %265, <4 x half> %279, <4 x float> %280) #9, !dbg !16
  %282 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !11
  %283 = zext i32 %282 to i64, !dbg !11
  %284 = inttoptr i64 %283 to ptr addrspace(3), !dbg !11
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %59, i64 %address.local146, !dbg !11
  %285 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %286 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !11
  %287 = sub i64 %286, %285, !dbg !11
  %dma.buffer.offset661 = trunc i64 %287 to i32, !dbg !11
  %288 = trunc i64 %285 to i32, !dbg !11
  %289 = insertelement <4 x i32> poison, i32 %288, i64 0, !dbg !11
  %290 = lshr i64 %285, 32, !dbg !11
  %291 = trunc i64 %290 to i32, !dbg !11
  %292 = insertelement <4 x i32> %289, i32 %291, i64 1, !dbg !11
  %293 = insertelement <4 x i32> %292, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc662 = insertelement <4 x i32> %293, i32 131072, i64 3, !dbg !11
  %294 = ptrtoint ptr addrspace(3) %284 to i32, !dbg !11
  %dma.buffer.lds663 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %294), !dbg !11
  %295 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds663, i32 %dma.buffer.offset661, <4 x i32> %dma.buffer.rsrc662) #10, !dbg !11
  %296 = add nuw i64 %61, 6144, !dbg !15
  %297 = icmp slt i64 %296, 0, !dbg !15
  %298 = sub nuw i64 -6145, %61, !dbg !15
  %299 = select i1 %297, i64 %298, i64 %296, !dbg !15
  %300 = sdiv i64 %299, 8192, !dbg !15
  %.lobit21 = ashr i64 %296, 63, !dbg !15
  %301 = xor i64 %300, %.lobit21, !dbg !15
  %302 = add nsw i64 %301, %15, !dbg !15
  %303 = add nsw i64 %160, 96, !dbg !15
  %304 = icmp slt i64 %160, -96, !dbg !15
  %305 = sub nsw i64 -97, %160, !dbg !15
  %306 = select i1 %304, i64 %305, i64 %303, !dbg !15
  %307 = sdiv i64 %306, 128, !dbg !15
  %308 = sext i1 %304 to i64, !dbg !15
  %309 = xor i64 %307, %308, !dbg !15
  %310 = shl nsw i64 %309, 7, !dbg !15
  %311 = sub i64 %55, %310, !dbg !15
  %312 = shl i64 %302, 19, !dbg !15
  %313 = and i64 %312, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %314 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %313, !dbg !15
  %315 = getelementptr [2 x i8], ptr addrspace(1) %314, i64 %95, !dbg !15
  %316 = getelementptr [2 x i8], ptr addrspace(1) %315, i64 %311, !dbg !15
  %317 = getelementptr i8, ptr addrspace(1) %316, i64 192, !dbg !15
  %318 = getelementptr [2 x i8], ptr addrspace(1) %317, i64 %101, !dbg !15
  %319 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local150), !dbg !15
  %320 = zext i32 %319 to i64, !dbg !15
  %321 = inttoptr i64 %320 to ptr addrspace(3), !dbg !15
  %address.local370 = getelementptr [2 x i8], ptr addrspace(1) %318, i64 %address.local160, !dbg !15
  %322 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %323 = ptrtoint ptr addrspace(1) %address.local370 to i64, !dbg !15
  %324 = sub i64 %323, %322, !dbg !15
  %dma.buffer.offset658 = trunc i64 %324 to i32, !dbg !15
  %325 = trunc i64 %322 to i32, !dbg !15
  %326 = insertelement <4 x i32> poison, i32 %325, i64 0, !dbg !15
  %327 = lshr i64 %322, 32, !dbg !15
  %328 = trunc i64 %327 to i32, !dbg !15
  %329 = insertelement <4 x i32> %326, i32 %328, i64 1, !dbg !15
  %330 = insertelement <4 x i32> %329, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc659 = insertelement <4 x i32> %330, i32 131072, i64 3, !dbg !15
  %331 = ptrtoint ptr addrspace(3) %321 to i32, !dbg !15
  %dma.buffer.lds660 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %331), !dbg !15
  %332 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds660, i32 %dma.buffer.offset658, <4 x i32> %dma.buffer.rsrc659) #10, !dbg !15
  %333 = load <4 x half>, ptr addrspace(3) %address.local174, align 8, !dbg !12
  %lds.const.gep642 = getelementptr i8, ptr addrspace(3) %address.local174, i64 2048, !dbg !12
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep642, align 8, !dbg !12
  %335 = fmul <4 x half> %333, splat (half 0xH3015), !dbg !12
  %336 = fmul <4 x half> %334, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %337 = add nuw nsw i64 %95, %15, !dbg !17
  %338 = shl i64 %337, 19, !dbg !17
  %339 = and i64 %338, 16252928, !dbg !17
  %340 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %339, !dbg !17
  %341 = getelementptr [2 x i8], ptr addrspace(1) %340, i64 %101, !dbg !17
  %342 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local141), !dbg !17
  %343 = zext i32 %342 to i64, !dbg !17
  %344 = inttoptr i64 %343 to ptr addrspace(3), !dbg !17
  %address.local371 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local372 = and i64 %address.local371, 8064, !dbg !17
  %address.local373 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local374 = and i64 %address.local373, 24, !dbg !17
  %address.local375 = or disjoint i64 %address.local372, %address.local374, !dbg !17
  %address.local376 = getelementptr [2 x i8], ptr addrspace(1) %341, i64 %address.local375, !dbg !17
  %345 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %346 = ptrtoint ptr addrspace(1) %address.local376 to i64, !dbg !17
  %347 = sub i64 %346, %345, !dbg !17
  %dma.buffer.offset655 = trunc i64 %347 to i32, !dbg !17
  %348 = trunc i64 %345 to i32, !dbg !17
  %349 = insertelement <4 x i32> poison, i32 %348, i64 0, !dbg !17
  %350 = lshr i64 %345, 32, !dbg !17
  %351 = trunc i64 %350 to i32, !dbg !17
  %352 = insertelement <4 x i32> %349, i32 %351, i64 1, !dbg !17
  %353 = insertelement <4 x i32> %352, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc656 = insertelement <4 x i32> %353, i32 131072, i64 3, !dbg !17
  %354 = ptrtoint ptr addrspace(3) %344 to i32, !dbg !17
  %dma.buffer.lds657 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %354), !dbg !17
  %355 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds657, i32 %dma.buffer.offset655, <4 x i32> %dma.buffer.rsrc656) #10, !dbg !17
  %356 = load <4 x half>, ptr addrspace(3) %address.local195, align 8, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %356, <4 x float> %270) #9, !dbg !16
  %lds.const.gep643 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2048, !dbg !16
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep643, align 8, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !16
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %358, <4 x float> %357) #9, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %359, <4 x float> %274) #9, !dbg !16
  %lds.const.gep644 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !16
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep644, align 8, !dbg !16
  %lds.const.gep645 = getelementptr i8, ptr addrspace(3) %address.local195, i64 128, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep645, align 8, !dbg !16
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %362, <4 x float> %361) #9, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %363, <4 x float> %278) #9, !dbg !16
  %lds.const.gep646 = getelementptr i8, ptr addrspace(3) %address.local195, i64 2176, !dbg !16
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep646, align 8, !dbg !16
  %lds.const.gep647 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !16
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep647, align 8, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %366, <4 x float> %365) #9, !dbg !16
  %lds.const.gep648 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !16
  %369 = load <4 x half>, ptr addrspace(3) %lds.const.gep648, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %367, <4 x float> %281) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %371 = or disjoint i64 %78, 63, !dbg !13
  %372 = icmp uge i64 %19, %371, !dbg !13
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %369, <4 x float> %370) #9, !dbg !16
  %374 = and i1 %39, %372, !dbg !13
  br i1 %374, label %375, label %376, !dbg !13

375:                                              ; preds = %77
  br label %419, !dbg !13

376:                                              ; preds = %77
  %377 = add i64 %19, %35, !dbg !13
  %378 = shl nuw nsw i64 %36, 4, !dbg !13
  %379 = add i64 %377, %378, !dbg !13
  %380 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %380 to i64, !dbg !13
  %381 = sub i64 %379, %index.compact, !dbg !13
  %382 = add i64 %381, 1, !dbg !22
  %383 = and i64 %35, 48, !dbg !13
  %384 = or disjoint i64 %78, %383, !dbg !13
  %385 = lshr i64 %382, 32, !dbg !23
  %compare.high = trunc i64 %385 to i32, !dbg !23
  %386 = lshr i64 %384, 32, !dbg !23
  %compare.high562 = trunc i64 %386 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high562, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high562, !dbg !23
  %compare.low.base = trunc i64 %382 to i32, !dbg !23
  %compare.low.base563 = trunc i64 %384 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base563, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %387 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low564 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result565 = select i1 %compare.high.equal, i1 %compare.low564, i1 %compare.high.order, !dbg !23
  %388 = select i1 %compare.result565, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base566 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local567 = or i32 %compare.local.base566, 2, !dbg !23
  %compare.low568 = icmp ugt i32 %compare.low.base, %compare.local567, !dbg !23
  %compare.result569 = select i1 %compare.high.equal, i1 %compare.low568, i1 %compare.high.order, !dbg !23
  %389 = select i1 %compare.result569, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base570 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local571 = or i32 %compare.local.base570, 3, !dbg !23
  %compare.low572 = icmp ugt i32 %compare.low.base, %compare.local571, !dbg !23
  %compare.result573 = select i1 %compare.high.equal, i1 %compare.low572, i1 %compare.high.order, !dbg !23
  %390 = select i1 %compare.result573, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base574 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local575 = or i32 %compare.local.base574, 4, !dbg !23
  %compare.low576 = icmp ugt i32 %compare.low.base, %compare.local575, !dbg !23
  %compare.result577 = select i1 %compare.high.equal, i1 %compare.low576, i1 %compare.high.order, !dbg !23
  %391 = select i1 %compare.result577, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base578 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local579 = or i32 %compare.local.base578, 5, !dbg !23
  %compare.low580 = icmp ugt i32 %compare.low.base, %compare.local579, !dbg !23
  %compare.result581 = select i1 %compare.high.equal, i1 %compare.low580, i1 %compare.high.order, !dbg !23
  %392 = select i1 %compare.result581, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base582 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local583 = or i32 %compare.local.base582, 6, !dbg !23
  %compare.low584 = icmp ugt i32 %compare.low.base, %compare.local583, !dbg !23
  %compare.result585 = select i1 %compare.high.equal, i1 %compare.low584, i1 %compare.high.order, !dbg !23
  %393 = select i1 %compare.result585, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base586 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local587 = or i32 %compare.local.base586, 7, !dbg !23
  %compare.low588 = icmp ugt i32 %compare.low.base, %compare.local587, !dbg !23
  %compare.result589 = select i1 %compare.high.equal, i1 %compare.low588, i1 %compare.high.order, !dbg !23
  %394 = select i1 %compare.result589, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base590 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local591 = or i32 %compare.local.base590, 8, !dbg !23
  %compare.low592 = icmp ugt i32 %compare.low.base, %compare.local591, !dbg !23
  %compare.result593 = select i1 %compare.high.equal, i1 %compare.low592, i1 %compare.high.order, !dbg !23
  %395 = select i1 %compare.result593, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base594 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local595 = or i32 %compare.local.base594, 9, !dbg !23
  %compare.low596 = icmp ugt i32 %compare.low.base, %compare.local595, !dbg !23
  %compare.result597 = select i1 %compare.high.equal, i1 %compare.low596, i1 %compare.high.order, !dbg !23
  %396 = select i1 %compare.result597, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base598 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local599 = or i32 %compare.local.base598, 10, !dbg !23
  %compare.low600 = icmp ugt i32 %compare.low.base, %compare.local599, !dbg !23
  %compare.result601 = select i1 %compare.high.equal, i1 %compare.low600, i1 %compare.high.order, !dbg !23
  %397 = select i1 %compare.result601, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base602 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local603 = or i32 %compare.local.base602, 11, !dbg !23
  %compare.low604 = icmp ugt i32 %compare.low.base, %compare.local603, !dbg !23
  %compare.result605 = select i1 %compare.high.equal, i1 %compare.low604, i1 %compare.high.order, !dbg !23
  %398 = select i1 %compare.result605, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base606 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local607 = or i32 %compare.local.base606, 12, !dbg !23
  %compare.low608 = icmp ugt i32 %compare.low.base, %compare.local607, !dbg !23
  %compare.result609 = select i1 %compare.high.equal, i1 %compare.low608, i1 %compare.high.order, !dbg !23
  %399 = select i1 %compare.result609, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base610 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local611 = or i32 %compare.local.base610, 13, !dbg !23
  %compare.low612 = icmp ugt i32 %compare.low.base, %compare.local611, !dbg !23
  %compare.result613 = select i1 %compare.high.equal, i1 %compare.low612, i1 %compare.high.order, !dbg !23
  %400 = select i1 %compare.result613, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base614 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local615 = or i32 %compare.local.base614, 14, !dbg !23
  %compare.low616 = icmp ugt i32 %compare.low.base, %compare.local615, !dbg !23
  %compare.result617 = select i1 %compare.high.equal, i1 %compare.low616, i1 %compare.high.order, !dbg !23
  %401 = select i1 %compare.result617, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base618 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base563, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local619 = or i32 %compare.local.base618, 15, !dbg !23
  %compare.low620 = icmp ugt i32 %compare.low.base, %compare.local619, !dbg !23
  %compare.result621 = select i1 %compare.high.equal, i1 %compare.low620, i1 %compare.high.order, !dbg !23
  %402 = select i1 %compare.result621, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %403 = insertelement <16 x float> poison, float %387, i64 0, !dbg !13
  %404 = insertelement <16 x float> %403, float %388, i64 1, !dbg !13
  %405 = insertelement <16 x float> %404, float %389, i64 2, !dbg !13
  %406 = insertelement <16 x float> %405, float %390, i64 3, !dbg !13
  %407 = insertelement <16 x float> %406, float %391, i64 4, !dbg !13
  %408 = insertelement <16 x float> %407, float %392, i64 5, !dbg !13
  %409 = insertelement <16 x float> %408, float %393, i64 6, !dbg !13
  %410 = insertelement <16 x float> %409, float %394, i64 7, !dbg !13
  %411 = insertelement <16 x float> %410, float %395, i64 8, !dbg !13
  %412 = insertelement <16 x float> %411, float %396, i64 9, !dbg !13
  %413 = insertelement <16 x float> %412, float %397, i64 10, !dbg !13
  %414 = insertelement <16 x float> %413, float %398, i64 11, !dbg !13
  %415 = insertelement <16 x float> %414, float %399, i64 12, !dbg !13
  %416 = insertelement <16 x float> %415, float %400, i64 13, !dbg !13
  %417 = insertelement <16 x float> %416, float %401, i64 14, !dbg !13
  %418 = insertelement <16 x float> %417, float %402, i64 15, !dbg !13
  br label %419, !dbg !13

419:                                              ; preds = %375, %376
  %420 = phi <16 x float> [ %418, %376 ], [ zeroinitializer, %375 ], !dbg !13
  br label %421, !dbg !13

421:                                              ; preds = %419
  %422 = shufflevector <16 x float> %420, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %423 = fadd <4 x float> %360, %422, !dbg !25
  %424 = extractelement <4 x float> %423, i64 0, !dbg !25
  %425 = extractelement <4 x float> %423, i64 1, !dbg !25
  %426 = extractelement <4 x float> %423, i64 2, !dbg !25
  %427 = extractelement <4 x float> %423, i64 3, !dbg !25
  %428 = shufflevector <16 x float> %420, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %429 = fadd <4 x float> %364, %428, !dbg !25
  %430 = extractelement <4 x float> %429, i64 0, !dbg !25
  %431 = extractelement <4 x float> %429, i64 1, !dbg !25
  %432 = extractelement <4 x float> %429, i64 2, !dbg !25
  %433 = extractelement <4 x float> %429, i64 3, !dbg !25
  %434 = shufflevector <16 x float> %420, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %435 = fadd <4 x float> %368, %434, !dbg !25
  %436 = extractelement <4 x float> %435, i64 0, !dbg !25
  %437 = extractelement <4 x float> %435, i64 1, !dbg !25
  %438 = extractelement <4 x float> %435, i64 2, !dbg !25
  %439 = extractelement <4 x float> %435, i64 3, !dbg !25
  %440 = shufflevector <16 x float> %420, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %441 = fadd <4 x float> %373, %440, !dbg !25
  %442 = extractelement <4 x float> %441, i64 0, !dbg !25
  %443 = extractelement <4 x float> %441, i64 1, !dbg !25
  %444 = extractelement <4 x float> %441, i64 2, !dbg !25
  %445 = extractelement <4 x float> %441, i64 3, !dbg !25
  %446 = call float @llvm.maxnum.f32(float %424, float 0xFFF0000000000000), !dbg !26
  %447 = call float @llvm.maxnum.f32(float %446, float %425), !dbg !26
  %448 = call float @llvm.maxnum.f32(float %447, float %426), !dbg !26
  %449 = call float @llvm.maxnum.f32(float %448, float %427), !dbg !26
  %450 = call float @llvm.maxnum.f32(float %449, float %430), !dbg !26
  %451 = call float @llvm.maxnum.f32(float %450, float %431), !dbg !26
  %452 = call float @llvm.maxnum.f32(float %451, float %432), !dbg !26
  %453 = call float @llvm.maxnum.f32(float %452, float %433), !dbg !26
  %454 = call float @llvm.maxnum.f32(float %453, float %436), !dbg !26
  %455 = call float @llvm.maxnum.f32(float %454, float %437), !dbg !26
  %456 = call float @llvm.maxnum.f32(float %455, float %438), !dbg !26
  %457 = call float @llvm.maxnum.f32(float %456, float %439), !dbg !26
  %458 = call float @llvm.maxnum.f32(float %457, float %442), !dbg !26
  %459 = call float @llvm.maxnum.f32(float %458, float %443), !dbg !26
  %460 = call float @llvm.maxnum.f32(float %459, float %444), !dbg !26
  %461 = call float @llvm.maxnum.f32(float %460, float %445), !dbg !26
  %462 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %463 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %462), !dbg !26
  %464 = add nuw nsw i32 %463, 64, !dbg !26
  %465 = xor i32 %463, 16, !dbg !26
  %466 = and i32 %464, 192, !dbg !26
  %467 = icmp samesign ult i32 %465, %466, !dbg !26
  %468 = select i1 %467, i32 %465, i32 %463, !dbg !26
  %469 = shl nuw nsw i32 %468, 2, !dbg !26
  %470 = bitcast float %461 to i32, !dbg !26
  %471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %469, i32 %470), !dbg !26
  %472 = bitcast i32 %471 to float, !dbg !26
  %473 = call float @llvm.maxnum.f32(float %461, float %472), !dbg !26
  %474 = xor i32 %463, 32, !dbg !26
  %475 = icmp samesign ult i32 %474, %466, !dbg !26
  %476 = select i1 %475, i32 %474, i32 %463, !dbg !26
  %477 = shl nuw nsw i32 %476, 2, !dbg !26
  %478 = bitcast float %473 to i32, !dbg !26
  %479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %478), !dbg !26
  %480 = bitcast i32 %479 to float, !dbg !26
  %481 = call float @llvm.maxnum.f32(float %473, float %480), !dbg !26
  %482 = insertelement <1 x float> poison, float %481, i64 0, !dbg !26
  %483 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %482, <1 x float> %66), !dbg !26
  %484 = shufflevector <1 x float> %483, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %485 = fsub <4 x float> %423, %484, !dbg !27
  %486 = fsub <4 x float> %429, %484, !dbg !27
  %487 = fsub <4 x float> %435, %484, !dbg !27
  %488 = fsub <4 x float> %441, %484, !dbg !27
  %489 = fsub <1 x float> %66, %483, !dbg !28
  %490 = extractelement <1 x float> %489, i64 0, !dbg !29
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !29
  %492 = insertelement <4 x float> poison, float %491, i64 0, !dbg !34
  %493 = shufflevector <4 x float> %492, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %494 = fmul <4 x float> %74, %493, !dbg !34
  %495 = fmul <4 x float> %73, %493, !dbg !34
  %496 = fmul <4 x float> %71, %493, !dbg !34
  %497 = fmul <4 x float> %70, %493, !dbg !34
  %498 = fmul <4 x float> %68, %493, !dbg !34
  %499 = fmul <4 x float> %67, %493, !dbg !34
  %500 = fmul <4 x float> %63, %493, !dbg !34
  %501 = fmul <4 x float> %62, %493, !dbg !34
  %502 = extractelement <1 x float> %65, i64 0, !dbg !30
  %503 = fmul float %502, %491, !dbg !30
  %504 = extractelement <4 x float> %485, i64 0, !dbg !31
  %505 = call float @llvm.amdgcn.exp2.f32(float %504), !dbg !31
  %506 = insertelement <4 x float> poison, float %505, i64 0, !dbg !33
  %507 = extractelement <4 x float> %485, i64 1, !dbg !31
  %508 = call float @llvm.amdgcn.exp2.f32(float %507), !dbg !31
  %509 = insertelement <4 x float> %506, float %508, i64 1, !dbg !33
  %510 = fadd float %505, %508, !dbg !32
  %511 = extractelement <4 x float> %485, i64 2, !dbg !31
  %512 = call float @llvm.amdgcn.exp2.f32(float %511), !dbg !31
  %513 = insertelement <4 x float> %509, float %512, i64 2, !dbg !33
  %514 = fadd float %510, %512, !dbg !32
  %515 = extractelement <4 x float> %485, i64 3, !dbg !31
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !31
  %517 = insertelement <4 x float> %513, float %516, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %517) #9, !dbg !33, !frisk.register_lifetimes !1
  %518 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %518) #9, !dbg !32, !frisk.register_lifetimes !1
  %519 = fadd float %514, %516, !dbg !32
  %520 = extractelement <4 x float> %486, i64 0, !dbg !31
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !31
  %522 = insertelement <4 x float> poison, float %521, i64 0, !dbg !33
  %523 = fadd float %519, %521, !dbg !32
  %524 = extractelement <4 x float> %486, i64 1, !dbg !31
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !31
  %526 = insertelement <4 x float> %522, float %525, i64 1, !dbg !33
  %527 = fadd float %523, %525, !dbg !32
  %528 = extractelement <4 x float> %486, i64 2, !dbg !31
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !31
  %530 = insertelement <4 x float> %526, float %529, i64 2, !dbg !33
  %531 = fadd float %527, %529, !dbg !32
  %532 = extractelement <4 x float> %486, i64 3, !dbg !31
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !31
  %534 = insertelement <4 x float> %530, float %533, i64 3, !dbg !33
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %534) #9, !dbg !33, !frisk.register_lifetimes !1
  %535 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !33
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %535) #9, !dbg !32, !frisk.register_lifetimes !1
  %536 = fadd float %531, %533, !dbg !32
  %537 = extractelement <4 x float> %487, i64 0, !dbg !31
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !31
  %539 = insertelement <4 x float> poison, float %538, i64 0, !dbg !33
  %540 = fadd float %536, %538, !dbg !32
  %541 = extractelement <4 x float> %487, i64 1, !dbg !31
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !31
  %543 = insertelement <4 x float> %539, float %542, i64 1, !dbg !33
  %544 = fadd float %540, %542, !dbg !32
  %545 = extractelement <4 x float> %487, i64 2, !dbg !31
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !31
  %547 = insertelement <4 x float> %543, float %546, i64 2, !dbg !33
  %548 = fadd float %544, %546, !dbg !32
  %549 = extractelement <4 x float> %487, i64 3, !dbg !31
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !31
  %551 = insertelement <4 x float> %547, float %550, i64 3, !dbg !33
  %math.result133 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %551) #9, !dbg !33, !frisk.register_lifetimes !1
  %552 = fptrunc <4 x float> %math.result133 to <4 x half>, !dbg !33
  %math.packed134 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %552) #9, !dbg !32, !frisk.register_lifetimes !1
  %553 = fadd float %548, %550, !dbg !32
  %554 = extractelement <4 x float> %488, i64 0, !dbg !31
  %555 = call float @llvm.amdgcn.exp2.f32(float %554), !dbg !31
  %556 = insertelement <4 x float> poison, float %555, i64 0, !dbg !33
  %557 = fadd float %553, %555, !dbg !32
  %558 = extractelement <4 x float> %488, i64 1, !dbg !31
  %559 = call float @llvm.amdgcn.exp2.f32(float %558), !dbg !31
  %560 = insertelement <4 x float> %556, float %559, i64 1, !dbg !33
  %561 = fadd float %557, %559, !dbg !32
  %562 = extractelement <4 x float> %488, i64 2, !dbg !31
  %563 = call float @llvm.amdgcn.exp2.f32(float %562), !dbg !31
  %564 = insertelement <4 x float> %560, float %563, i64 2, !dbg !33
  %565 = fadd float %561, %563, !dbg !32
  %566 = extractelement <4 x float> %488, i64 3, !dbg !31
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !31
  %568 = insertelement <4 x float> %564, float %567, i64 3, !dbg !33
  %math.result135 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %568) #9, !dbg !33, !frisk.register_lifetimes !1
  %569 = fptrunc <4 x float> %math.result135 to <4 x half>, !dbg !33
  %math.packed136 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %569) #9, !dbg !32, !frisk.register_lifetimes !1
  %570 = fadd float %565, %567, !dbg !32
  %571 = bitcast float %570 to i32, !dbg !32
  %572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %469, i32 %571), !dbg !32
  %573 = bitcast i32 %572 to float, !dbg !32
  %574 = fadd float %570, %573, !dbg !32
  %575 = bitcast float %574 to i32, !dbg !32
  %576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %477, i32 %575), !dbg !32
  %577 = bitcast i32 %576 to float, !dbg !32
  %578 = fadd float %574, %577, !dbg !32
  %579 = fadd float %578, %503, !dbg !32
  %580 = insertelement <1 x float> poison, float %579, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread377 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local378 = zext nneg i32 %address.thread377 to i64, !dbg !12
  %address.local379 = and i64 %address.local378, 60, !dbg !14
  %address.local380 = add nuw nsw i64 %address.local379, %address.local378, !dbg !14
  %address.local381 = shl nuw nsw i64 %address.local380, 3, !dbg !14
  %address.local382 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !14
  %address.local383 = and i64 %address.local378, 63, !dbg !12
  %address.local384 = lshr i64 %address.local383, 4, !dbg !14
  %address.local385 = mul nuw nsw i64 %address.local384, 384, !dbg !14
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) %address.local382, i64 %address.local385, !dbg !14
  %address.local387 = lshr i64 %address.local378, 2, !dbg !16
  %address.local388 = mul nsw i64 %address.local387, -32, !dbg !14
  %address.local389 = getelementptr [2 x i8], ptr addrspace(3) %address.local386, i64 %address.local388, !dbg !14
  %581 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !14
  %582 = bitcast <4 x i32> %581 to <8 x half>, !dbg !14
  %address.local390 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !14
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) %address.local390, i64 %address.local385, !dbg !14
  %address.local392 = getelementptr [2 x i8], ptr addrspace(3) %address.local391, i64 %address.local388, !dbg !14
  %address.local393 = getelementptr i8, ptr addrspace(3) %address.local392, i64 256, !dbg !14
  %583 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !14
  %584 = bitcast <4 x i32> %583 to <8 x half>, !dbg !14
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !14
  %address.local395 = getelementptr [2 x i8], ptr addrspace(3) %address.local394, i64 %address.local385, !dbg !14
  %address.local396 = getelementptr [2 x i8], ptr addrspace(3) %address.local395, i64 %address.local388, !dbg !14
  %address.local397 = getelementptr i8, ptr addrspace(3) %address.local396, i64 512, !dbg !14
  %585 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !14
  %586 = bitcast <4 x i32> %585 to <8 x half>, !dbg !14
  %address.local398 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local381, !dbg !14
  %address.local399 = getelementptr [2 x i8], ptr addrspace(3) %address.local398, i64 %address.local385, !dbg !14
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) %address.local399, i64 %address.local388, !dbg !14
  %address.local401 = getelementptr i8, ptr addrspace(3) %address.local400, i64 768, !dbg !14
  %587 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !14
  %588 = bitcast <4 x i32> %587 to <8 x half>, !dbg !14
  %589 = add nuw nsw i64 %95, %15, !dbg !17
  %590 = shl i64 %589, 19, !dbg !17
  %591 = and i64 %590, 16252928, !dbg !17
  %592 = add i64 %591, %101, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %593 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %592, !dbg !17
  %594 = getelementptr i8, ptr addrspace(1) %593, i64 64, !dbg !17
  %address.local402 = shl nuw nsw i64 %address.local378, 3, !dbg !11
  %address.local403 = and i64 %address.local402, 1536, !dbg !11
  %address.local404 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local403, !dbg !11
  %address.local405 = ptrtoint ptr addrspace(3) %address.local404 to i64, !dbg !11
  %address.local406 = trunc i64 %address.local405 to i32, !dbg !11
  %595 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !17
  %596 = zext i32 %595 to i64, !dbg !17
  %597 = inttoptr i64 %596 to ptr addrspace(3), !dbg !17
  %address.local407 = shl nuw nsw i64 %address.local378, 5, !dbg !17
  %address.local408 = and i64 %address.local407, 8064, !dbg !17
  %address.local409 = shl nuw nsw i64 %address.local378, 3, !dbg !17
  %address.local410 = and i64 %address.local409, 24, !dbg !17
  %address.local411 = or disjoint i64 %address.local408, %address.local410, !dbg !17
  %address.local412 = getelementptr [2 x i8], ptr addrspace(1) %594, i64 %address.local411, !dbg !17
  %598 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %599 = ptrtoint ptr addrspace(1) %address.local412 to i64, !dbg !17
  %600 = sub i64 %599, %598, !dbg !17
  %dma.buffer.offset652 = trunc i64 %600 to i32, !dbg !17
  %601 = trunc i64 %598 to i32, !dbg !17
  %602 = insertelement <4 x i32> poison, i32 %601, i64 0, !dbg !17
  %603 = lshr i64 %598, 32, !dbg !17
  %604 = trunc i64 %603 to i32, !dbg !17
  %605 = insertelement <4 x i32> %602, i32 %604, i64 1, !dbg !17
  %606 = insertelement <4 x i32> %605, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc653 = insertelement <4 x i32> %606, i32 131072, i64 3, !dbg !17
  %607 = ptrtoint ptr addrspace(3) %597 to i32, !dbg !17
  %dma.buffer.lds654 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %607), !dbg !17
  %608 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds654, i32 %dma.buffer.offset652, <4 x i32> %dma.buffer.rsrc653) #10, !dbg !17
  %609 = shufflevector <8 x half> %582, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %609, <4 x float> %501) #9, !dbg !14
  %611 = shufflevector <8 x half> %584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %611, <4 x float> %610) #9, !dbg !14
  %613 = shufflevector <8 x half> %586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %613, <4 x float> %612) #9, !dbg !14
  %615 = shufflevector <8 x half> %588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %615, <4 x float> %614) #9, !dbg !14
  %617 = shufflevector <8 x half> %582, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %617, <4 x float> %500) #9, !dbg !14
  %619 = shufflevector <8 x half> %584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %619, <4 x float> %618) #9, !dbg !14
  %621 = shufflevector <8 x half> %586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %621, <4 x float> %620) #9, !dbg !14
  %623 = shufflevector <8 x half> %588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %623, <4 x float> %622) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %625 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !14
  %626 = bitcast <4 x i32> %625 to <8 x half>, !dbg !14
  %627 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !14
  %628 = bitcast <4 x i32> %627 to <8 x half>, !dbg !14
  %629 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !14
  %630 = bitcast <4 x i32> %629 to <8 x half>, !dbg !14
  %631 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !14
  %632 = bitcast <4 x i32> %631 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %633 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %592, !dbg !17
  %634 = getelementptr i8, ptr addrspace(1) %633, i64 128, !dbg !17
  %635 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !17
  %636 = zext i32 %635 to i64, !dbg !17
  %637 = inttoptr i64 %636 to ptr addrspace(3), !dbg !17
  %address.local413 = getelementptr [2 x i8], ptr addrspace(1) %634, i64 %address.local411, !dbg !17
  %638 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %639 = ptrtoint ptr addrspace(1) %address.local413 to i64, !dbg !17
  %640 = sub i64 %639, %638, !dbg !17
  %dma.buffer.offset649 = trunc i64 %640 to i32, !dbg !17
  %641 = trunc i64 %638 to i32, !dbg !17
  %642 = insertelement <4 x i32> poison, i32 %641, i64 0, !dbg !17
  %643 = lshr i64 %638, 32, !dbg !17
  %644 = trunc i64 %643 to i32, !dbg !17
  %645 = insertelement <4 x i32> %642, i32 %644, i64 1, !dbg !17
  %646 = insertelement <4 x i32> %645, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc650 = insertelement <4 x i32> %646, i32 131072, i64 3, !dbg !17
  %647 = ptrtoint ptr addrspace(3) %637 to i32, !dbg !17
  %dma.buffer.lds651 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %647), !dbg !17
  %648 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds651, i32 %dma.buffer.offset649, <4 x i32> %dma.buffer.rsrc650) #10, !dbg !17
  %649 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %649, <4 x float> %499) #9, !dbg !14
  %651 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %651, <4 x float> %650) #9, !dbg !14
  %653 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %653, <4 x float> %652) #9, !dbg !14
  %655 = shufflevector <8 x half> %632, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %655, <4 x float> %654) #9, !dbg !14
  %657 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %657, <4 x float> %498) #9, !dbg !14
  %659 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %659, <4 x float> %658) #9, !dbg !14
  %661 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %661, <4 x float> %660) #9, !dbg !14
  %663 = shufflevector <8 x half> %632, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %663, <4 x float> %662) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %665 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !14
  %666 = bitcast <4 x i32> %665 to <8 x half>, !dbg !14
  %667 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !14
  %668 = bitcast <4 x i32> %667 to <8 x half>, !dbg !14
  %669 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !14
  %670 = bitcast <4 x i32> %669 to <8 x half>, !dbg !14
  %671 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !14
  %672 = bitcast <4 x i32> %671 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %673 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %592, !dbg !17
  %674 = getelementptr i8, ptr addrspace(1) %673, i64 192, !dbg !17
  %675 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local406), !dbg !17
  %676 = zext i32 %675 to i64, !dbg !17
  %677 = inttoptr i64 %676 to ptr addrspace(3), !dbg !17
  %address.local414 = getelementptr [2 x i8], ptr addrspace(1) %674, i64 %address.local411, !dbg !17
  %678 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %679 = ptrtoint ptr addrspace(1) %address.local414 to i64, !dbg !17
  %680 = sub i64 %679, %678, !dbg !17
  %dma.buffer.offset = trunc i64 %680 to i32, !dbg !17
  %681 = trunc i64 %678 to i32, !dbg !17
  %682 = insertelement <4 x i32> poison, i32 %681, i64 0, !dbg !17
  %683 = lshr i64 %678, 32, !dbg !17
  %684 = trunc i64 %683 to i32, !dbg !17
  %685 = insertelement <4 x i32> %682, i32 %684, i64 1, !dbg !17
  %686 = insertelement <4 x i32> %685, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %686, i32 131072, i64 3, !dbg !17
  %687 = ptrtoint ptr addrspace(3) %677 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %687), !dbg !17
  %688 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %689 = shufflevector <8 x half> %666, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %689, <4 x float> %497) #9, !dbg !14
  %691 = shufflevector <8 x half> %668, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %691, <4 x float> %690) #9, !dbg !14
  %693 = shufflevector <8 x half> %670, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %693, <4 x float> %692) #9, !dbg !14
  %695 = shufflevector <8 x half> %672, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %695, <4 x float> %694) #9, !dbg !14
  %697 = shufflevector <8 x half> %666, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %697, <4 x float> %496) #9, !dbg !14
  %699 = shufflevector <8 x half> %668, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %699, <4 x float> %698) #9, !dbg !14
  %701 = shufflevector <8 x half> %670, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %702 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %701, <4 x float> %700) #9, !dbg !14
  %703 = shufflevector <8 x half> %672, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %704 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %703, <4 x float> %702) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %705 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local389, i32 0) #0, !dbg !14
  %706 = bitcast <4 x i32> %705 to <8 x half>, !dbg !14
  %707 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !14
  %708 = bitcast <4 x i32> %707 to <8 x half>, !dbg !14
  %709 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #0, !dbg !14
  %710 = bitcast <4 x i32> %709 to <8 x half>, !dbg !14
  %711 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local401, i32 0) #0, !dbg !14
  %712 = bitcast <4 x i32> %711 to <8 x half>, !dbg !14
  %713 = shufflevector <8 x half> %706, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %713, <4 x float> %495) #9, !dbg !14
  %715 = shufflevector <8 x half> %708, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %715, <4 x float> %714) #9, !dbg !14
  %717 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %717, <4 x float> %716) #9, !dbg !14
  %719 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %719, <4 x float> %718) #9, !dbg !14
  %721 = shufflevector <8 x half> %706, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %721, <4 x float> %494) #9, !dbg !14
  %723 = shufflevector <8 x half> %708, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %723, <4 x float> %722) #9, !dbg !14
  %725 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed134, <4 x half> %725, <4 x float> %724) #9, !dbg !14
  %727 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed136, <4 x half> %727, <4 x float> %726) #9, !dbg !14
  %729 = shufflevector <4 x float> %616, <4 x float> %624, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %730 = shufflevector <4 x float> %656, <4 x float> %664, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %731 = shufflevector <4 x float> %696, <4 x float> %704, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %732 = shufflevector <4 x float> %720, <4 x float> %728, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %733 = add nuw nsw i64 %61, 1, !dbg !7
  br label %60, !dbg !7

734:                                              ; preds = %60
  %735 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %735, 0, !dbg !7
  %736 = and i32 %34, 32, !dbg !7
  %.not18 = icmp eq i32 %736, 0, !dbg !7
  %737 = extractelement <8 x float> %75, i64 0, !dbg !7
  %738 = extractelement <8 x float> %75, i64 1, !dbg !7
  %739 = extractelement <8 x float> %75, i64 2, !dbg !7
  %740 = extractelement <8 x float> %75, i64 3, !dbg !7
  %741 = select i1 %.not, float %738, float %737, !dbg !7
  %742 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %743 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %742), !dbg !7
  %744 = add nuw nsw i32 %743, 64, !dbg !7
  %745 = xor i32 %743, 16, !dbg !7
  %746 = and i32 %744, 192, !dbg !7
  %747 = icmp samesign ult i32 %745, %746, !dbg !7
  %748 = select i1 %747, i32 %745, i32 %743, !dbg !7
  %749 = shl nuw nsw i32 %748, 2, !dbg !7
  %750 = bitcast float %741 to i32, !dbg !7
  %751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %750), !dbg !7
  %752 = bitcast i32 %751 to float, !dbg !7
  %753 = select i1 %.not, float %737, float %752, !dbg !7
  %754 = select i1 %.not, float %752, float %738, !dbg !7
  %755 = select i1 %.not, float %740, float %739, !dbg !7
  %756 = bitcast float %755 to i32, !dbg !7
  %757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %756), !dbg !7
  %758 = bitcast i32 %757 to float, !dbg !7
  %759 = select i1 %.not, float %739, float %758, !dbg !7
  %760 = select i1 %.not, float %758, float %740, !dbg !7
  %761 = select i1 %.not18, float %759, float %753, !dbg !7
  %762 = xor i32 %743, 32, !dbg !7
  %763 = icmp samesign ult i32 %762, %746, !dbg !7
  %764 = select i1 %763, i32 %762, i32 %743, !dbg !7
  %765 = shl nuw nsw i32 %764, 2, !dbg !7
  %766 = bitcast float %761 to i32, !dbg !7
  %767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %766), !dbg !7
  %768 = bitcast i32 %767 to float, !dbg !7
  %769 = select i1 %.not18, float %753, float %768, !dbg !7
  %770 = select i1 %.not18, float %768, float %759, !dbg !7
  %771 = select i1 %.not18, float %760, float %754, !dbg !7
  %772 = bitcast float %771 to i32, !dbg !7
  %773 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %772), !dbg !7
  %774 = bitcast i32 %773 to float, !dbg !7
  %775 = select i1 %.not18, float %754, float %774, !dbg !7
  %776 = select i1 %.not18, float %774, float %760, !dbg !7
  %777 = extractelement <8 x float> %72, i64 0, !dbg !7
  %778 = extractelement <8 x float> %72, i64 1, !dbg !7
  %779 = extractelement <8 x float> %72, i64 2, !dbg !7
  %780 = extractelement <8 x float> %72, i64 3, !dbg !7
  %781 = select i1 %.not, float %778, float %777, !dbg !7
  %782 = bitcast float %781 to i32, !dbg !7
  %783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %782), !dbg !7
  %784 = bitcast i32 %783 to float, !dbg !7
  %785 = select i1 %.not, float %777, float %784, !dbg !7
  %786 = select i1 %.not, float %784, float %778, !dbg !7
  %787 = select i1 %.not, float %780, float %779, !dbg !7
  %788 = bitcast float %787 to i32, !dbg !7
  %789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %788), !dbg !7
  %790 = bitcast i32 %789 to float, !dbg !7
  %791 = select i1 %.not, float %779, float %790, !dbg !7
  %792 = select i1 %.not, float %790, float %780, !dbg !7
  %793 = select i1 %.not18, float %791, float %785, !dbg !7
  %794 = bitcast float %793 to i32, !dbg !7
  %795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %794), !dbg !7
  %796 = bitcast i32 %795 to float, !dbg !7
  %797 = select i1 %.not18, float %785, float %796, !dbg !7
  %798 = select i1 %.not18, float %796, float %791, !dbg !7
  %799 = select i1 %.not18, float %792, float %786, !dbg !7
  %800 = bitcast float %799 to i32, !dbg !7
  %801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %800), !dbg !7
  %802 = bitcast i32 %801 to float, !dbg !7
  %803 = select i1 %.not18, float %786, float %802, !dbg !7
  %804 = select i1 %.not18, float %802, float %792, !dbg !7
  %805 = extractelement <8 x float> %69, i64 0, !dbg !7
  %806 = extractelement <8 x float> %69, i64 1, !dbg !7
  %807 = extractelement <8 x float> %69, i64 2, !dbg !7
  %808 = extractelement <8 x float> %69, i64 3, !dbg !7
  %809 = select i1 %.not, float %806, float %805, !dbg !7
  %810 = bitcast float %809 to i32, !dbg !7
  %811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %810), !dbg !7
  %812 = bitcast i32 %811 to float, !dbg !7
  %813 = select i1 %.not, float %805, float %812, !dbg !7
  %814 = select i1 %.not, float %812, float %806, !dbg !7
  %815 = select i1 %.not, float %808, float %807, !dbg !7
  %816 = bitcast float %815 to i32, !dbg !7
  %817 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %816), !dbg !7
  %818 = bitcast i32 %817 to float, !dbg !7
  %819 = select i1 %.not, float %807, float %818, !dbg !7
  %820 = select i1 %.not, float %818, float %808, !dbg !7
  %821 = select i1 %.not18, float %819, float %813, !dbg !7
  %822 = bitcast float %821 to i32, !dbg !7
  %823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %822), !dbg !7
  %824 = bitcast i32 %823 to float, !dbg !7
  %825 = select i1 %.not18, float %813, float %824, !dbg !7
  %826 = select i1 %.not18, float %824, float %819, !dbg !7
  %827 = select i1 %.not18, float %820, float %814, !dbg !7
  %828 = bitcast float %827 to i32, !dbg !7
  %829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %828), !dbg !7
  %830 = bitcast i32 %829 to float, !dbg !7
  %831 = select i1 %.not18, float %814, float %830, !dbg !7
  %832 = select i1 %.not18, float %830, float %820, !dbg !7
  %833 = extractelement <8 x float> %64, i64 0, !dbg !7
  %834 = extractelement <8 x float> %64, i64 1, !dbg !7
  %835 = extractelement <8 x float> %64, i64 2, !dbg !7
  %836 = extractelement <8 x float> %64, i64 3, !dbg !7
  %837 = select i1 %.not, float %834, float %833, !dbg !7
  %838 = bitcast float %837 to i32, !dbg !7
  %839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %838), !dbg !7
  %840 = bitcast i32 %839 to float, !dbg !7
  %841 = select i1 %.not, float %833, float %840, !dbg !7
  %842 = select i1 %.not, float %840, float %834, !dbg !7
  %843 = select i1 %.not, float %836, float %835, !dbg !7
  %844 = bitcast float %843 to i32, !dbg !7
  %845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %844), !dbg !7
  %846 = bitcast i32 %845 to float, !dbg !7
  %847 = select i1 %.not, float %835, float %846, !dbg !7
  %848 = select i1 %.not, float %846, float %836, !dbg !7
  %849 = select i1 %.not18, float %847, float %841, !dbg !7
  %850 = bitcast float %849 to i32, !dbg !7
  %851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %850), !dbg !7
  %852 = select i1 %.not18, float %848, float %842, !dbg !7
  %853 = bitcast float %852 to i32, !dbg !7
  %854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %853), !dbg !7
  %855 = shufflevector <1 x float> %65, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %856 = extractelement <8 x float> %75, i64 4, !dbg !7
  %857 = extractelement <8 x float> %75, i64 5, !dbg !7
  %858 = extractelement <8 x float> %75, i64 6, !dbg !7
  %859 = extractelement <8 x float> %75, i64 7, !dbg !7
  %860 = select i1 %.not, float %857, float %856, !dbg !7
  %861 = bitcast float %860 to i32, !dbg !7
  %862 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %861), !dbg !7
  %863 = bitcast i32 %862 to float, !dbg !7
  %864 = select i1 %.not, float %856, float %863, !dbg !7
  %865 = select i1 %.not, float %863, float %857, !dbg !7
  %866 = select i1 %.not, float %859, float %858, !dbg !7
  %867 = bitcast float %866 to i32, !dbg !7
  %868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %867), !dbg !7
  %869 = bitcast i32 %868 to float, !dbg !7
  %870 = select i1 %.not, float %858, float %869, !dbg !7
  %871 = select i1 %.not, float %869, float %859, !dbg !7
  %872 = select i1 %.not18, float %870, float %864, !dbg !7
  %873 = bitcast float %872 to i32, !dbg !7
  %874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %873), !dbg !7
  %875 = bitcast i32 %874 to float, !dbg !7
  %876 = select i1 %.not18, float %864, float %875, !dbg !7
  %877 = select i1 %.not18, float %875, float %870, !dbg !7
  %878 = select i1 %.not18, float %871, float %865, !dbg !7
  %879 = bitcast float %878 to i32, !dbg !7
  %880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %879), !dbg !7
  %881 = bitcast i32 %880 to float, !dbg !7
  %882 = select i1 %.not18, float %865, float %881, !dbg !7
  %883 = select i1 %.not18, float %881, float %871, !dbg !7
  %884 = extractelement <8 x float> %72, i64 4, !dbg !7
  %885 = extractelement <8 x float> %72, i64 5, !dbg !7
  %886 = extractelement <8 x float> %72, i64 6, !dbg !7
  %887 = extractelement <8 x float> %72, i64 7, !dbg !7
  %888 = select i1 %.not, float %885, float %884, !dbg !7
  %889 = bitcast float %888 to i32, !dbg !7
  %890 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %889), !dbg !7
  %891 = bitcast i32 %890 to float, !dbg !7
  %892 = select i1 %.not, float %884, float %891, !dbg !7
  %893 = select i1 %.not, float %891, float %885, !dbg !7
  %894 = select i1 %.not, float %887, float %886, !dbg !7
  %895 = bitcast float %894 to i32, !dbg !7
  %896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %895), !dbg !7
  %897 = bitcast i32 %896 to float, !dbg !7
  %898 = select i1 %.not, float %886, float %897, !dbg !7
  %899 = select i1 %.not, float %897, float %887, !dbg !7
  %900 = select i1 %.not18, float %898, float %892, !dbg !7
  %901 = bitcast float %900 to i32, !dbg !7
  %902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %901), !dbg !7
  %903 = bitcast i32 %902 to float, !dbg !7
  %904 = select i1 %.not18, float %892, float %903, !dbg !7
  %905 = select i1 %.not18, float %903, float %898, !dbg !7
  %906 = select i1 %.not18, float %899, float %893, !dbg !7
  %907 = bitcast float %906 to i32, !dbg !7
  %908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %907), !dbg !7
  %909 = bitcast i32 %908 to float, !dbg !7
  %910 = select i1 %.not18, float %893, float %909, !dbg !7
  %911 = select i1 %.not18, float %909, float %899, !dbg !7
  %912 = extractelement <8 x float> %69, i64 4, !dbg !7
  %913 = extractelement <8 x float> %69, i64 5, !dbg !7
  %914 = extractelement <8 x float> %69, i64 6, !dbg !7
  %915 = extractelement <8 x float> %69, i64 7, !dbg !7
  %916 = select i1 %.not, float %913, float %912, !dbg !7
  %917 = bitcast float %916 to i32, !dbg !7
  %918 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %917), !dbg !7
  %919 = bitcast i32 %918 to float, !dbg !7
  %920 = select i1 %.not, float %912, float %919, !dbg !7
  %921 = select i1 %.not, float %919, float %913, !dbg !7
  %922 = select i1 %.not, float %915, float %914, !dbg !7
  %923 = bitcast float %922 to i32, !dbg !7
  %924 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %923), !dbg !7
  %925 = bitcast i32 %924 to float, !dbg !7
  %926 = select i1 %.not, float %914, float %925, !dbg !7
  %927 = select i1 %.not, float %925, float %915, !dbg !7
  %928 = select i1 %.not18, float %926, float %920, !dbg !7
  %929 = bitcast float %928 to i32, !dbg !7
  %930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %929), !dbg !7
  %931 = bitcast i32 %930 to float, !dbg !7
  %932 = select i1 %.not18, float %920, float %931, !dbg !7
  %933 = select i1 %.not18, float %931, float %926, !dbg !7
  %934 = select i1 %.not18, float %927, float %921, !dbg !7
  %935 = bitcast float %934 to i32, !dbg !7
  %936 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %935), !dbg !7
  %937 = bitcast i32 %936 to float, !dbg !7
  %938 = select i1 %.not18, float %921, float %937, !dbg !7
  %939 = select i1 %.not18, float %937, float %927, !dbg !7
  %940 = extractelement <8 x float> %64, i64 4, !dbg !7
  %941 = extractelement <8 x float> %64, i64 5, !dbg !7
  %942 = extractelement <8 x float> %64, i64 6, !dbg !7
  %943 = extractelement <8 x float> %64, i64 7, !dbg !7
  %944 = select i1 %.not, float %941, float %940, !dbg !7
  %945 = bitcast float %944 to i32, !dbg !7
  %946 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %945), !dbg !7
  %947 = bitcast i32 %946 to float, !dbg !7
  %948 = select i1 %.not, float %940, float %947, !dbg !7
  %949 = select i1 %.not, float %947, float %941, !dbg !7
  %950 = select i1 %.not, float %943, float %942, !dbg !7
  %951 = bitcast float %950 to i32, !dbg !7
  %952 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %951), !dbg !7
  %953 = bitcast i32 %952 to float, !dbg !7
  %954 = select i1 %.not, float %942, float %953, !dbg !7
  %955 = select i1 %.not, float %953, float %943, !dbg !7
  %956 = select i1 %.not18, float %954, float %948, !dbg !7
  %957 = bitcast float %956 to i32, !dbg !7
  %958 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %957), !dbg !7
  %959 = select i1 %.not18, float %955, float %949, !dbg !7
  %960 = bitcast float %959 to i32, !dbg !7
  %961 = call i32 @llvm.amdgcn.ds.bpermute(i32 %765, i32 %960), !dbg !7
  br i1 true, label %962, label %996, !dbg !20

962:                                              ; preds = %734
  %963 = bitcast i32 %958 to float, !dbg !7
  %964 = select i1 %.not18, float %948, float %963, !dbg !7
  %965 = insertelement <4 x float> poison, float %964, i64 0, !dbg !18
  %966 = bitcast i32 %961 to float, !dbg !7
  %967 = select i1 %.not18, float %949, float %966, !dbg !7
  %968 = insertelement <4 x float> %965, float %967, i64 1, !dbg !18
  %969 = select i1 %.not18, float %963, float %954, !dbg !7
  %970 = insertelement <4 x float> %968, float %969, i64 2, !dbg !18
  %971 = select i1 %.not18, float %966, float %955, !dbg !7
  %972 = insertelement <4 x float> %970, float %971, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %855, !dbg !18
  %normalized = fmul <4 x float> %972, %row.reciprocal, !dbg !18
  %973 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %974 = extractelement <4 x half> %973, i64 3, !dbg !19
  %975 = extractelement <4 x half> %973, i64 2, !dbg !19
  %976 = extractelement <4 x half> %973, i64 1, !dbg !19
  %977 = extractelement <4 x half> %973, i64 0, !dbg !19
  %978 = bitcast i32 %851 to float, !dbg !7
  %979 = select i1 %.not18, float %841, float %978, !dbg !7
  %980 = insertelement <4 x float> poison, float %979, i64 0, !dbg !18
  %981 = bitcast i32 %854 to float, !dbg !7
  %982 = select i1 %.not18, float %842, float %981, !dbg !7
  %983 = insertelement <4 x float> %980, float %982, i64 1, !dbg !18
  %984 = select i1 %.not18, float %978, float %847, !dbg !7
  %985 = insertelement <4 x float> %983, float %984, i64 2, !dbg !18
  %986 = select i1 %.not18, float %981, float %848, !dbg !7
  %987 = insertelement <4 x float> %985, float %986, i64 3, !dbg !18
  %normalized38 = fmul <4 x float> %987, %row.reciprocal, !dbg !18
  %988 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !19
  %989 = extractelement <4 x half> %988, i64 3, !dbg !19
  %990 = extractelement <4 x half> %988, i64 2, !dbg !19
  %991 = extractelement <4 x half> %988, i64 1, !dbg !19
  %992 = extractelement <4 x half> %988, i64 0, !dbg !19
  %993 = add nsw i64 %15, %25, !dbg !20
  %994 = shl i64 %993, 19, !dbg !20
  %995 = and i64 %994, 16252928, !dbg !20
  %address.thread415 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local416 = zext nneg i32 %address.thread415 to i64, !dbg !12
  %address.local417 = add i64 %28, %address.local416, !dbg !20
  %address.local418 = lshr i64 %address.local416, 6, !dbg !12
  %address.local419 = shl nuw nsw i64 %address.local418, 4, !dbg !20
  %address.local420 = add i64 %address.local417, %address.local419, !dbg !20
  %address.local421 = and i64 %address.local416, 240, !dbg !20
  %address.local422 = sub i64 %address.local420, %address.local421, !dbg !20
  %address.local423 = add i64 %address.local422, %29, !dbg !20
  %address.local424 = shl nuw nsw i64 %address.local423, 7, !dbg !20
  %address.local425 = add nuw nsw i64 %995, %address.local424, !dbg !20
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local427 = lshr i64 %address.local416, 2, !dbg !12
  %address.local428 = and i64 %address.local427, 12, !dbg !12
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local426, i64 %address.local428, !dbg !20
  store half %992, ptr addrspace(1) %address.local429, align 2, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local428, !dbg !20
  %address.local432 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local431, i64 2, !dbg !20
  store half %991, ptr addrspace(1) %address.local432, align 2, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local428, !dbg !20
  %address.local435 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local434, i64 4, !dbg !20
  store half %990, ptr addrspace(1) %address.local435, align 2, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local428, !dbg !20
  %address.local438 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local437, i64 6, !dbg !20
  store half %989, ptr addrspace(1) %address.local438, align 2, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local428, !dbg !20
  %address.local441 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local440, i64 32, !dbg !20
  store half %977, ptr addrspace(1) %address.local441, align 2, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local428, !dbg !20
  %address.local444 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local443, i64 34, !dbg !20
  store half %976, ptr addrspace(1) %address.local444, align 2, !dbg !20
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local428, !dbg !20
  %address.local447 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local446, i64 36, !dbg !20
  store half %975, ptr addrspace(1) %address.local447, align 2, !dbg !20
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local425, !dbg !20
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local428, !dbg !20
  %address.local450 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local449, i64 38, !dbg !20
  store half %974, ptr addrspace(1) %address.local450, align 2, !dbg !20
  br label %996, !dbg !20

996:                                              ; preds = %962, %734
  br i1 true, label %997, label %1017, !dbg !20

997:                                              ; preds = %996
  %998 = insertelement <4 x float> poison, float %932, i64 0, !dbg !18
  %999 = insertelement <4 x float> %998, float %938, i64 1, !dbg !18
  %1000 = insertelement <4 x float> %999, float %933, i64 2, !dbg !18
  %1001 = insertelement <4 x float> %1000, float %939, i64 3, !dbg !18
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %855, !dbg !18
  %normalized40 = fmul <4 x float> %1001, %row.reciprocal39, !dbg !18
  %1002 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !19
  %1003 = extractelement <4 x half> %1002, i64 3, !dbg !19
  %1004 = extractelement <4 x half> %1002, i64 2, !dbg !19
  %1005 = extractelement <4 x half> %1002, i64 1, !dbg !19
  %1006 = extractelement <4 x half> %1002, i64 0, !dbg !19
  %1007 = insertelement <4 x float> poison, float %825, i64 0, !dbg !18
  %1008 = insertelement <4 x float> %1007, float %831, i64 1, !dbg !18
  %1009 = insertelement <4 x float> %1008, float %826, i64 2, !dbg !18
  %1010 = insertelement <4 x float> %1009, float %832, i64 3, !dbg !18
  %normalized41 = fmul <4 x float> %1010, %row.reciprocal39, !dbg !18
  %1011 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !19
  %1012 = extractelement <4 x half> %1011, i64 3, !dbg !19
  %1013 = extractelement <4 x half> %1011, i64 2, !dbg !19
  %1014 = extractelement <4 x half> %1011, i64 1, !dbg !19
  %1015 = extractelement <4 x half> %1011, i64 0, !dbg !19
  %1016 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread451 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local452 = zext nneg i32 %address.thread451 to i64, !dbg !12
  %address.local453 = add i64 %28, %address.local452, !dbg !20
  %address.local454 = lshr i64 %address.local452, 6, !dbg !12
  %address.local455 = shl nuw nsw i64 %address.local454, 4, !dbg !20
  %address.local456 = add i64 %address.local453, %address.local455, !dbg !20
  %address.local457 = and i64 %address.local452, 240, !dbg !20
  %address.local458 = sub i64 %address.local456, %address.local457, !dbg !20
  %address.local459 = add i64 %address.local458, %29, !dbg !20
  %address.local460 = shl nuw nsw i64 %address.local459, 7, !dbg !20
  %address.local461 = add nuw nsw i64 %1016, %address.local460, !dbg !20
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local463 = lshr i64 %address.local452, 2, !dbg !12
  %address.local464 = and i64 %address.local463, 12, !dbg !12
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local464, !dbg !20
  %address.local466 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local465, i64 64, !dbg !20
  store half %1015, ptr addrspace(1) %address.local466, align 2, !dbg !20
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local464, !dbg !20
  %address.local469 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local468, i64 66, !dbg !20
  store half %1014, ptr addrspace(1) %address.local469, align 2, !dbg !20
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local470, i64 %address.local464, !dbg !20
  %address.local472 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local471, i64 68, !dbg !20
  store half %1013, ptr addrspace(1) %address.local472, align 2, !dbg !20
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local464, !dbg !20
  %address.local475 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local474, i64 70, !dbg !20
  store half %1012, ptr addrspace(1) %address.local475, align 2, !dbg !20
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local476, i64 %address.local464, !dbg !20
  %address.local478 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local477, i64 96, !dbg !20
  store half %1006, ptr addrspace(1) %address.local478, align 2, !dbg !20
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local464, !dbg !20
  %address.local481 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local480, i64 98, !dbg !20
  store half %1005, ptr addrspace(1) %address.local481, align 2, !dbg !20
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local482, i64 %address.local464, !dbg !20
  %address.local484 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local483, i64 100, !dbg !20
  store half %1004, ptr addrspace(1) %address.local484, align 2, !dbg !20
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !20
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local464, !dbg !20
  %address.local487 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local486, i64 102, !dbg !20
  store half %1003, ptr addrspace(1) %address.local487, align 2, !dbg !20
  br label %1017, !dbg !20

1017:                                             ; preds = %997, %996
  br i1 true, label %1018, label %1038, !dbg !20

1018:                                             ; preds = %1017
  %1019 = insertelement <4 x float> poison, float %904, i64 0, !dbg !18
  %1020 = insertelement <4 x float> %1019, float %910, i64 1, !dbg !18
  %1021 = insertelement <4 x float> %1020, float %905, i64 2, !dbg !18
  %1022 = insertelement <4 x float> %1021, float %911, i64 3, !dbg !18
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %855, !dbg !18
  %normalized43 = fmul <4 x float> %1022, %row.reciprocal42, !dbg !18
  %1023 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !19
  %1024 = extractelement <4 x half> %1023, i64 3, !dbg !19
  %1025 = extractelement <4 x half> %1023, i64 2, !dbg !19
  %1026 = extractelement <4 x half> %1023, i64 1, !dbg !19
  %1027 = extractelement <4 x half> %1023, i64 0, !dbg !19
  %1028 = insertelement <4 x float> poison, float %797, i64 0, !dbg !18
  %1029 = insertelement <4 x float> %1028, float %803, i64 1, !dbg !18
  %1030 = insertelement <4 x float> %1029, float %798, i64 2, !dbg !18
  %1031 = insertelement <4 x float> %1030, float %804, i64 3, !dbg !18
  %normalized44 = fmul <4 x float> %1031, %row.reciprocal42, !dbg !18
  %1032 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !19
  %1033 = extractelement <4 x half> %1032, i64 3, !dbg !19
  %1034 = extractelement <4 x half> %1032, i64 2, !dbg !19
  %1035 = extractelement <4 x half> %1032, i64 1, !dbg !19
  %1036 = extractelement <4 x half> %1032, i64 0, !dbg !19
  %1037 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread488 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local489 = zext nneg i32 %address.thread488 to i64, !dbg !12
  %address.local490 = add i64 %28, %address.local489, !dbg !20
  %address.local491 = lshr i64 %address.local489, 6, !dbg !12
  %address.local492 = shl nuw nsw i64 %address.local491, 4, !dbg !20
  %address.local493 = add i64 %address.local490, %address.local492, !dbg !20
  %address.local494 = and i64 %address.local489, 240, !dbg !20
  %address.local495 = sub i64 %address.local493, %address.local494, !dbg !20
  %address.local496 = add i64 %address.local495, %29, !dbg !20
  %address.local497 = shl nuw nsw i64 %address.local496, 7, !dbg !20
  %address.local498 = add nuw nsw i64 %1037, %address.local497, !dbg !20
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local500 = lshr i64 %address.local489, 2, !dbg !12
  %address.local501 = and i64 %address.local500, 12, !dbg !12
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local501, !dbg !20
  %address.local503 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local502, i64 128, !dbg !20
  store half %1036, ptr addrspace(1) %address.local503, align 2, !dbg !20
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local505 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local504, i64 %address.local501, !dbg !20
  %address.local506 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local505, i64 130, !dbg !20
  store half %1035, ptr addrspace(1) %address.local506, align 2, !dbg !20
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local508 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local507, i64 %address.local501, !dbg !20
  %address.local509 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local508, i64 132, !dbg !20
  store half %1034, ptr addrspace(1) %address.local509, align 2, !dbg !20
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local510, i64 %address.local501, !dbg !20
  %address.local512 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local511, i64 134, !dbg !20
  store half %1033, ptr addrspace(1) %address.local512, align 2, !dbg !20
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local514 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local501, !dbg !20
  %address.local515 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local514, i64 160, !dbg !20
  store half %1027, ptr addrspace(1) %address.local515, align 2, !dbg !20
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local501, !dbg !20
  %address.local518 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local517, i64 162, !dbg !20
  store half %1026, ptr addrspace(1) %address.local518, align 2, !dbg !20
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local501, !dbg !20
  %address.local521 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local520, i64 164, !dbg !20
  store half %1025, ptr addrspace(1) %address.local521, align 2, !dbg !20
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local498, !dbg !20
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local501, !dbg !20
  %address.local524 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local523, i64 166, !dbg !20
  store half %1024, ptr addrspace(1) %address.local524, align 2, !dbg !20
  br label %1038, !dbg !20

1038:                                             ; preds = %1018, %1017
  br i1 true, label %1039, label %1059, !dbg !20

1039:                                             ; preds = %1038
  %1040 = insertelement <4 x float> poison, float %876, i64 0, !dbg !18
  %1041 = insertelement <4 x float> %1040, float %882, i64 1, !dbg !18
  %1042 = insertelement <4 x float> %1041, float %877, i64 2, !dbg !18
  %1043 = insertelement <4 x float> %1042, float %883, i64 3, !dbg !18
  %row.reciprocal45 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %855, !dbg !18
  %normalized46 = fmul <4 x float> %1043, %row.reciprocal45, !dbg !18
  %1044 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !19
  %1045 = extractelement <4 x half> %1044, i64 3, !dbg !19
  %1046 = extractelement <4 x half> %1044, i64 2, !dbg !19
  %1047 = extractelement <4 x half> %1044, i64 1, !dbg !19
  %1048 = extractelement <4 x half> %1044, i64 0, !dbg !19
  %1049 = insertelement <4 x float> poison, float %769, i64 0, !dbg !18
  %1050 = insertelement <4 x float> %1049, float %775, i64 1, !dbg !18
  %1051 = insertelement <4 x float> %1050, float %770, i64 2, !dbg !18
  %1052 = insertelement <4 x float> %1051, float %776, i64 3, !dbg !18
  %normalized47 = fmul <4 x float> %1052, %row.reciprocal45, !dbg !18
  %1053 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !19
  %1054 = extractelement <4 x half> %1053, i64 3, !dbg !19
  %1055 = extractelement <4 x half> %1053, i64 2, !dbg !19
  %1056 = extractelement <4 x half> %1053, i64 1, !dbg !19
  %1057 = extractelement <4 x half> %1053, i64 0, !dbg !19
  %1058 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread525 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local526 = zext nneg i32 %address.thread525 to i64, !dbg !12
  %address.local527 = add i64 %28, %address.local526, !dbg !20
  %address.local528 = lshr i64 %address.local526, 6, !dbg !12
  %address.local529 = shl nuw nsw i64 %address.local528, 4, !dbg !20
  %address.local530 = add i64 %address.local527, %address.local529, !dbg !20
  %address.local531 = and i64 %address.local526, 240, !dbg !20
  %address.local532 = sub i64 %address.local530, %address.local531, !dbg !20
  %address.local533 = add i64 %address.local532, %29, !dbg !20
  %address.local534 = shl nuw nsw i64 %address.local533, 7, !dbg !20
  %address.local535 = add nuw nsw i64 %1058, %address.local534, !dbg !20
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local537 = lshr i64 %address.local526, 2, !dbg !12
  %address.local538 = and i64 %address.local537, 12, !dbg !12
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local538, !dbg !20
  %address.local540 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local539, i64 192, !dbg !20
  store half %1057, ptr addrspace(1) %address.local540, align 2, !dbg !20
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local538, !dbg !20
  %address.local543 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local542, i64 194, !dbg !20
  store half %1056, ptr addrspace(1) %address.local543, align 2, !dbg !20
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local538, !dbg !20
  %address.local546 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local545, i64 196, !dbg !20
  store half %1055, ptr addrspace(1) %address.local546, align 2, !dbg !20
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local538, !dbg !20
  %address.local549 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local548, i64 198, !dbg !20
  store half %1054, ptr addrspace(1) %address.local549, align 2, !dbg !20
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local538, !dbg !20
  %address.local552 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local551, i64 224, !dbg !20
  store half %1048, ptr addrspace(1) %address.local552, align 2, !dbg !20
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local538, !dbg !20
  %address.local555 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local554, i64 226, !dbg !20
  store half %1047, ptr addrspace(1) %address.local555, align 2, !dbg !20
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local538, !dbg !20
  %address.local558 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local557, i64 228, !dbg !20
  store half %1046, ptr addrspace(1) %address.local558, align 2, !dbg !20
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local535, !dbg !20
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local538, !dbg !20
  %address.local561 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local560, i64 230, !dbg !20
  store half %1045, ptr addrspace(1) %address.local561, align 2, !dbg !20
  br label %1059, !dbg !20

1059:                                             ; preds = %1039, %1038
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/group8-reverse-k128-swizzle")
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
