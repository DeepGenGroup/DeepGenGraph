; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/online-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/online-baseline/input.mlir"

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
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 63, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 2047, !dbg !8
  %18 = add nsw i64 %14, -64, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 64, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 6, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !38
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local138 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local139 = and i64 %address.local138, 1536, !dbg !8
  %address.local140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local139, !dbg !8
  %address.local141 = ptrtoint ptr addrspace(3) %address.local140 to i64, !dbg !8
  %address.local142 = trunc i64 %address.local141 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local143 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local144 = and i64 %address.local143, 8064, !dbg !8
  %address.local145 = lshr i64 %address.local, 3, !dbg !8
  %address.local146 = and i64 %address.local145, 24, !dbg !8
  %address.local147 = or disjoint i64 %address.local144, %address.local146, !dbg !8
  %address.local148 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local147, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local148 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset718 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc719 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds720 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds720, i32 %dma.buffer.offset718, <4 x i32> %dma.buffer.rsrc719) #10, !dbg !8
  %48 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %49 = sext i32 %48 to i64, !dbg !5
  %50 = shl i64 %15, 6, !dbg !10
  %51 = lshr i64 %32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %52 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %53 = getelementptr i8, ptr addrspace(1) %52, i64 64, !dbg !8
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local139, !dbg !8
  %address.local150 = ptrtoint ptr addrspace(3) %address.local149 to i64, !dbg !8
  %address.local151 = trunc i64 %address.local150 to i32, !dbg !8
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local151), !dbg !8
  %55 = zext i32 %54 to i64, !dbg !8
  %56 = inttoptr i64 %55 to ptr addrspace(3), !dbg !8
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local147, !dbg !8
  %57 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %58 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !8
  %59 = sub i64 %58, %57, !dbg !8
  %dma.buffer.offset715 = trunc i64 %59 to i32, !dbg !8
  %60 = trunc i64 %57 to i32, !dbg !8
  %61 = insertelement <4 x i32> poison, i32 %60, i64 0, !dbg !8
  %62 = lshr i64 %57, 32, !dbg !8
  %63 = trunc i64 %62 to i32, !dbg !8
  %64 = insertelement <4 x i32> %61, i32 %63, i64 1, !dbg !8
  %65 = insertelement <4 x i32> %64, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc716 = insertelement <4 x i32> %65, i32 131072, i64 3, !dbg !8
  %66 = ptrtoint ptr addrspace(3) %56 to i32, !dbg !8
  %dma.buffer.lds717 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !8
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds717, i32 %dma.buffer.offset715, <4 x i32> %dma.buffer.rsrc716) #10, !dbg !8
  %address.local153 = lshr i32 %address.thread, 2, !dbg !11
  %address.local154 = and i32 %address.local153, 4, !dbg !11
  %address.local155 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local156 = and i32 %address.local155, 384, !dbg !11
  %address.local157 = or disjoint i32 %address.local154, %address.local156, !dbg !11
  %address.local158 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local159 = and i32 %address.local158, 120, !dbg !11
  %address.local160 = or disjoint i32 %address.local157, %address.local159, !dbg !11
  %address.local161 = shl nuw nsw i32 %address.thread, 4, !dbg !11
  %address.local162 = and i32 %address.local161, 512, !dbg !11
  %address.local163 = or disjoint i32 %address.local160, %address.local162, !dbg !11
  %address.local164 = zext nneg i32 %address.local163 to i64, !dbg !11
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local164, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local165, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local165, i64 2048, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !11
  %71 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %72 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %73 = getelementptr i8, ptr addrspace(1) %72, i64 128, !dbg !8
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local142), !dbg !8
  %75 = zext i32 %74 to i64, !dbg !8
  %76 = inttoptr i64 %75 to ptr addrspace(3), !dbg !8
  %address.local180 = getelementptr [2 x i8], ptr addrspace(1) %73, i64 %address.local147, !dbg !8
  %77 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %78 = ptrtoint ptr addrspace(1) %address.local180 to i64, !dbg !8
  %79 = sub i64 %78, %77, !dbg !8
  %dma.buffer.offset712 = trunc i64 %79 to i32, !dbg !8
  %80 = trunc i64 %77 to i32, !dbg !8
  %81 = insertelement <4 x i32> poison, i32 %80, i64 0, !dbg !8
  %82 = lshr i64 %77, 32, !dbg !8
  %83 = trunc i64 %82 to i32, !dbg !8
  %84 = insertelement <4 x i32> %81, i32 %83, i64 1, !dbg !8
  %85 = insertelement <4 x i32> %84, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc713 = insertelement <4 x i32> %85, i32 131072, i64 3, !dbg !8
  %86 = ptrtoint ptr addrspace(3) %76 to i32, !dbg !8
  %dma.buffer.lds714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !8
  %87 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds714, i32 %dma.buffer.offset712, <4 x i32> %dma.buffer.rsrc713) #10, !dbg !8
  %lds.const.gep655 = getelementptr i8, ptr addrspace(3) %address.local165, i64 4096, !dbg !11
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep655, align 8, !dbg !11
  %lds.const.gep656 = getelementptr i8, ptr addrspace(3) %address.local165, i64 6144, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep656, align 8, !dbg !11
  %90 = fmul <4 x half> %88, splat (half 0xH3015), !dbg !11
  %91 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %92 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %93 = getelementptr i8, ptr addrspace(1) %92, i64 192, !dbg !8
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local151), !dbg !8
  %95 = zext i32 %94 to i64, !dbg !8
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !8
  %address.local183 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local147, !dbg !8
  %97 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %98 = ptrtoint ptr addrspace(1) %address.local183 to i64, !dbg !8
  %99 = sub i64 %98, %97, !dbg !8
  %dma.buffer.offset709 = trunc i64 %99 to i32, !dbg !8
  %100 = trunc i64 %97 to i32, !dbg !8
  %101 = insertelement <4 x i32> poison, i32 %100, i64 0, !dbg !8
  %102 = lshr i64 %97, 32, !dbg !8
  %103 = trunc i64 %102 to i32, !dbg !8
  %104 = insertelement <4 x i32> %101, i32 %103, i64 1, !dbg !8
  %105 = insertelement <4 x i32> %104, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc710 = insertelement <4 x i32> %105, i32 131072, i64 3, !dbg !8
  %106 = ptrtoint ptr addrspace(3) %96 to i32, !dbg !8
  %dma.buffer.lds711 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !8
  %107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds711, i32 %dma.buffer.offset709, <4 x i32> %dma.buffer.rsrc710) #10, !dbg !8
  %108 = load <4 x half>, ptr addrspace(3) %address.local165, align 8, !dbg !11
  %lds.const.gep657 = getelementptr i8, ptr addrspace(3) %address.local165, i64 2048, !dbg !11
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep657, align 8, !dbg !11
  %110 = fmul <4 x half> %108, splat (half 0xH3015), !dbg !11
  %111 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep658 = getelementptr i8, ptr addrspace(3) %address.local165, i64 4096, !dbg !11
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep658, align 8, !dbg !11
  %lds.const.gep659 = getelementptr i8, ptr addrspace(3) %address.local165, i64 6144, !dbg !11
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep659, align 8, !dbg !11
  %114 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !11
  %115 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %116 = or disjoint i64 %50, 63, !dbg !13
  %117 = add i64 %50, 64, !dbg !13
  %118 = icmp uge i64 %117, %116, !dbg !13
  %119 = sub nsw i64 64, %14, !dbg !7
  %120 = shl i64 %16, 7, !dbg !16
  %121 = and i64 %120, 3968, !dbg !16
  %122 = shl nsw i64 %49, 12, !dbg !16
  %123 = or disjoint i64 %121, %122, !dbg !16
  %124 = lshr i64 %32, 2, !dbg !15
  %125 = shl nuw nsw i64 %32, 2, !dbg !15
  %126 = mul nsw i64 %124, -16, !dbg !15
  %127 = add nsw i64 %125, %126, !dbg !15
  %128 = and i32 %31, 1, !dbg !15
  %index.compact114 = zext nneg i32 %128 to i64, !dbg !15
  %129 = or disjoint i64 %127, %index.compact114, !dbg !15
  %130 = add nuw nsw i64 %129, 1, !dbg !15
  %131 = sdiv i64 %130, 2, !dbg !15
  br label %132, !dbg !7

132:                                              ; preds = %421, %4
  %133 = phi i64 [ %733, %421 ], [ 0, %4 ], !dbg !7
  %134 = phi <4 x float> [ %616, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %135 = phi <4 x float> [ %624, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %136 = phi <8 x float> [ %729, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %137 = phi <1 x float> [ %580, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %138 = phi <1 x float> [ %483, %421 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %139 = phi <4 x float> [ %656, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %140 = phi <4 x float> [ %664, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %141 = phi <8 x float> [ %730, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %142 = phi <4 x float> [ %696, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %143 = phi <4 x float> [ %704, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %144 = phi <8 x float> [ %731, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %145 = phi <4 x float> [ %720, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %146 = phi <4 x float> [ %728, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %147 = phi <8 x float> [ %732, %421 ], [ zeroinitializer, %4 ], !dbg !7
  %148 = icmp slt i64 %133, %119, !dbg !7
  br i1 %148, label %149, label %734, !dbg !7

149:                                              ; preds = %132
  %150 = shl nsw i64 %133, 6, !dbg !7
  %151 = lshr i64 %133, 13, !dbg !16
  %152 = add nuw nsw i64 %151, %16, !dbg !16
  %153 = lshr i64 %133, 6, !dbg !16
  %154 = and i64 %153, 127, !dbg !16
  %155 = shl i64 %152, 19, !dbg !16
  %156 = and i64 %155, 16252928, !dbg !16
  %157 = shl i64 %153, 19, !dbg !16
  %158 = shl i64 %133, 13, !dbg !16
  %159 = sub i64 %158, %157, !dbg !16
  %160 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %156, !dbg !16
  %161 = getelementptr [2 x i8], ptr addrspace(1) %160, i64 %154, !dbg !16
  %162 = getelementptr [2 x i8], ptr addrspace(1) %161, i64 %159, !dbg !16
  %address.thread184 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !16, !range !38, !frisk.register_lifetimes !1
  %address.local185 = zext nneg i32 %address.thread184 to i64, !dbg !8
  %address.local186 = shl nuw nsw i64 %address.local185, 3, !dbg !8
  %address.local187 = and i64 %address.local186, 1536, !dbg !8
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !8
  %address.local189 = ptrtoint ptr addrspace(3) %address.local188 to i64, !dbg !8
  %address.local190 = trunc i64 %address.local189 to i32, !dbg !8
  %163 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !16
  %164 = zext i32 %163 to i64, !dbg !16
  %165 = inttoptr i64 %164 to ptr addrspace(3), !dbg !16
  %address.local191 = lshr i64 %address.local185, 3, !dbg !16
  %address.local192 = and i64 %address.local191, 24, !dbg !16
  %address.local193 = and i64 %address.local185, 59, !dbg !16
  %address.local194 = lshr i64 %address.local185, 2, !dbg !16
  %address.local195 = and i64 %address.local194, 4, !dbg !16
  %address.local196 = add nuw nsw i64 %address.local195, %address.local185, !dbg !16
  %address.local197 = and i64 %address.local196, 4, !dbg !16
  %address.local198 = or disjoint i64 %address.local193, %address.local197, !dbg !16
  %address.local199 = shl nuw nsw i64 %address.local198, 7, !dbg !16
  %address.local200 = or disjoint i64 %address.local192, %address.local199, !dbg !16
  %address.local201 = getelementptr [2 x i8], ptr addrspace(1) %162, i64 %address.local200, !dbg !16
  %166 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %167 = ptrtoint ptr addrspace(1) %address.local201 to i64, !dbg !16
  %168 = sub i64 %167, %166, !dbg !16
  %dma.buffer.offset706 = trunc i64 %168 to i32, !dbg !16
  %169 = trunc i64 %166 to i32, !dbg !16
  %170 = insertelement <4 x i32> poison, i32 %169, i64 0, !dbg !16
  %171 = lshr i64 %166, 32, !dbg !16
  %172 = trunc i64 %171 to i32, !dbg !16
  %173 = insertelement <4 x i32> %170, i32 %172, i64 1, !dbg !16
  %174 = insertelement <4 x i32> %173, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc707 = insertelement <4 x i32> %174, i32 131072, i64 3, !dbg !16
  %175 = ptrtoint ptr addrspace(3) %165 to i32, !dbg !16
  %dma.buffer.lds708 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %175), !dbg !16
  %176 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds708, i32 %dma.buffer.offset706, <4 x i32> %dma.buffer.rsrc707) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %177 = add nuw i64 %133, 2048, !dbg !16
  %178 = icmp slt i64 %177, 0, !dbg !16
  %179 = sub nuw i64 -2049, %133, !dbg !16
  %180 = select i1 %178, i64 %179, i64 %177, !dbg !16
  %181 = sdiv i64 %180, 8192, !dbg !16
  %.lobit19 = ashr i64 %177, 63, !dbg !16
  %182 = xor i64 %181, %.lobit19, !dbg !16
  %183 = add nsw i64 %182, %16, !dbg !16
  %184 = add nsw i64 %153, %123, !dbg !16
  %185 = add nsw i64 %184, 32, !dbg !16
  %186 = icmp slt i64 %184, -32, !dbg !16
  %187 = sub nsw i64 -33, %184, !dbg !16
  %188 = select i1 %186, i64 %187, i64 %185, !dbg !16
  %189 = sdiv i64 %188, 128, !dbg !16
  %190 = sext i1 %186 to i64, !dbg !16
  %191 = xor i64 %189, %190, !dbg !16
  %192 = shl nsw i64 %191, 7, !dbg !16
  %193 = sub i64 %123, %192, !dbg !16
  %194 = shl i64 %183, 19, !dbg !16
  %195 = and i64 %194, 16252928, !dbg !16
  %196 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %195, !dbg !16
  %197 = getelementptr [2 x i8], ptr addrspace(1) %196, i64 %153, !dbg !16
  %198 = getelementptr [2 x i8], ptr addrspace(1) %197, i64 %193, !dbg !16
  %199 = getelementptr i8, ptr addrspace(1) %198, i64 64, !dbg !16
  %200 = getelementptr [2 x i8], ptr addrspace(1) %199, i64 %159, !dbg !16
  %address.local202 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local187, !dbg !8
  %address.local203 = ptrtoint ptr addrspace(3) %address.local202 to i64, !dbg !8
  %address.local204 = trunc i64 %address.local203 to i32, !dbg !8
  %201 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local204), !dbg !16
  %202 = zext i32 %201 to i64, !dbg !16
  %203 = inttoptr i64 %202 to ptr addrspace(3), !dbg !16
  %address.local205 = getelementptr [2 x i8], ptr addrspace(1) %200, i64 %address.local200, !dbg !16
  %204 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %205 = ptrtoint ptr addrspace(1) %address.local205 to i64, !dbg !16
  %206 = sub i64 %205, %204, !dbg !16
  %dma.buffer.offset703 = trunc i64 %206 to i32, !dbg !16
  %207 = trunc i64 %204 to i32, !dbg !16
  %208 = insertelement <4 x i32> poison, i32 %207, i64 0, !dbg !16
  %209 = lshr i64 %204, 32, !dbg !16
  %210 = trunc i64 %209 to i32, !dbg !16
  %211 = insertelement <4 x i32> %208, i32 %210, i64 1, !dbg !16
  %212 = insertelement <4 x i32> %211, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc704 = insertelement <4 x i32> %212, i32 131072, i64 3, !dbg !16
  %213 = ptrtoint ptr addrspace(3) %203 to i32, !dbg !16
  %dma.buffer.lds705 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %213), !dbg !16
  %214 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds705, i32 %dma.buffer.offset703, <4 x i32> %dma.buffer.rsrc704) #10, !dbg !16
  %address.local206 = mul nuw nsw i64 %address.local185, 160, !dbg !15
  %address.local207 = lshr i64 %address.local185, 2, !dbg !15
  %address.local208 = mul nsw i64 %address.local207, -640, !dbg !15
  %address.local209 = add nsw i64 %address.local206, %address.local208, !dbg !15
  %address.local210 = shl nuw nsw i64 %address.local185, 1, !dbg !15
  %address.local211 = and i64 %address.local210, 24, !dbg !15
  %address.local212 = or disjoint i64 %address.local209, %address.local211, !dbg !15
  %address.local213 = shl i64 %address.local185, 62, !dbg !15
  %address.local214 = ashr exact i64 %address.local213, 57, !dbg !15
  %address.local215 = and i64 %address.local214, -64, !dbg !15
  %address.local216 = add nsw i64 %address.local212, %address.local215, !dbg !15
  %address.local217 = lshr i64 %address.local185, 2, !dbg !11
  %address.local218 = and i64 %address.local217, 12, !dbg !11
  %address.local219 = add nsw i64 %address.local216, %address.local218, !dbg !15
  %address.local220 = and i64 %address.local185, 63, !dbg !11
  %address.local221 = lshr i64 %address.local220, 5, !dbg !11
  %address.local222 = mul nuw nsw i64 %address.local221, 504, !dbg !11
  %address.local223 = add i64 %address.local219, %address.local222, !dbg !15
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local223, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !15
  %lds.const.gep660 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !15
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep660, align 8, !dbg !15
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !15
  %address.local235 = shl nuw nsw i32 %address.thread184, 1, !dbg !15
  %address.local236 = and i32 %address.local235, 24, !dbg !15
  %address.local237 = shl nuw nsw i32 %address.thread184, 8, !dbg !15
  %address.local238 = and i32 %address.local237, 768, !dbg !15
  %address.local239 = or disjoint i32 %address.local236, %address.local238, !dbg !15
  %address.local240 = shl nuw nsw i32 %address.thread184, 5, !dbg !15
  %address.local241 = and i32 %address.local240, 96, !dbg !15
  %address.local242 = or disjoint i32 %address.local239, %address.local241, !dbg !15
  %address.local243 = zext nneg i32 %address.local242 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %131) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local244 = lshr i32 %address.thread184, 1, !dbg !15
  %address.local245 = and i32 %address.local244, 1, !dbg !15
  %address.local246 = zext nneg i32 %address.local245 to i64, !dbg !15
  %address.local247 = add nsw i64 %address.base, %address.local246, !dbg !15
  %address.local248 = shl i64 %address.local247, 6, !dbg !15
  %address.local249 = sub i64 %address.local243, %address.local248, !dbg !15
  %address.local250 = add i64 %address.local249, %address.local218, !dbg !15
  %address.local251 = add i64 %address.local250, %address.local222, !dbg !15
  %address.local252 = add i64 %address.local251, 32, !dbg !15
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local252, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %address.local253, align 8, !dbg !15
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %216, <4 x float> %217) #9, !dbg !15
  %lds.const.gep661 = getelementptr i8, ptr addrspace(3) %address.local253, i64 2048, !dbg !15
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep661, align 8, !dbg !15
  %lds.const.gep662 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !15
  %221 = load <4 x half>, ptr addrspace(3) %lds.const.gep662, align 8, !dbg !15
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %218, <4 x float> zeroinitializer) #9, !dbg !15
  %lds.const.gep663 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !15
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep663, align 8, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %220, <4 x float> %222) #9, !dbg !15
  %lds.const.gep664 = getelementptr i8, ptr addrspace(3) %address.local253, i64 128, !dbg !15
  %225 = load <4 x half>, ptr addrspace(3) %lds.const.gep664, align 8, !dbg !15
  %lds.const.gep665 = getelementptr i8, ptr addrspace(3) %address.local253, i64 2176, !dbg !15
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep665, align 8, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %221, <4 x float> zeroinitializer) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %223, <4 x float> %227) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %225, <4 x float> zeroinitializer) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %226, <4 x float> %229) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %231 = add nuw i64 %133, 4096, !dbg !16
  %232 = icmp slt i64 %231, 0, !dbg !16
  %233 = sub nuw i64 -4097, %133, !dbg !16
  %234 = select i1 %232, i64 %233, i64 %231, !dbg !16
  %235 = sdiv i64 %234, 8192, !dbg !16
  %.lobit20 = ashr i64 %231, 63, !dbg !16
  %236 = xor i64 %235, %.lobit20, !dbg !16
  %237 = add nsw i64 %236, %16, !dbg !16
  %238 = add nsw i64 %184, 64, !dbg !16
  %239 = icmp slt i64 %184, -64, !dbg !16
  %240 = sub nsw i64 -65, %184, !dbg !16
  %241 = select i1 %239, i64 %240, i64 %238, !dbg !16
  %242 = sdiv i64 %241, 128, !dbg !16
  %243 = sext i1 %239 to i64, !dbg !16
  %244 = xor i64 %242, %243, !dbg !16
  %245 = shl nsw i64 %244, 7, !dbg !16
  %246 = sub i64 %123, %245, !dbg !16
  %247 = shl i64 %237, 19, !dbg !16
  %248 = and i64 %247, 16252928, !dbg !16
  %249 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %248, !dbg !16
  %250 = getelementptr [2 x i8], ptr addrspace(1) %249, i64 %153, !dbg !16
  %251 = getelementptr [2 x i8], ptr addrspace(1) %250, i64 %246, !dbg !16
  %252 = getelementptr i8, ptr addrspace(1) %251, i64 128, !dbg !16
  %253 = getelementptr [2 x i8], ptr addrspace(1) %252, i64 %159, !dbg !16
  %254 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !16
  %255 = zext i32 %254 to i64, !dbg !16
  %256 = inttoptr i64 %255 to ptr addrspace(3), !dbg !16
  %address.local394 = getelementptr [2 x i8], ptr addrspace(1) %253, i64 %address.local200, !dbg !16
  %257 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %258 = ptrtoint ptr addrspace(1) %address.local394 to i64, !dbg !16
  %259 = sub i64 %258, %257, !dbg !16
  %dma.buffer.offset700 = trunc i64 %259 to i32, !dbg !16
  %260 = trunc i64 %257 to i32, !dbg !16
  %261 = insertelement <4 x i32> poison, i32 %260, i64 0, !dbg !16
  %262 = lshr i64 %257, 32, !dbg !16
  %263 = trunc i64 %262 to i32, !dbg !16
  %264 = insertelement <4 x i32> %261, i32 %263, i64 1, !dbg !16
  %265 = insertelement <4 x i32> %264, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc701 = insertelement <4 x i32> %265, i32 131072, i64 3, !dbg !16
  %266 = ptrtoint ptr addrspace(3) %256 to i32, !dbg !16
  %dma.buffer.lds702 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %266), !dbg !16
  %267 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds702, i32 %dma.buffer.offset700, <4 x i32> %dma.buffer.rsrc701) #10, !dbg !16
  %lds.const.gep666 = getelementptr i8, ptr addrspace(3) %address.local224, i64 4096, !dbg !15
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep666, align 8, !dbg !15
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %268, <4 x float> %219) #9, !dbg !15
  %lds.const.gep667 = getelementptr i8, ptr addrspace(3) %address.local224, i64 6144, !dbg !15
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep667, align 8, !dbg !15
  %lds.const.gep668 = getelementptr i8, ptr addrspace(3) %address.local253, i64 4096, !dbg !15
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep668, align 8, !dbg !15
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %270, <4 x float> %269) #9, !dbg !15
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %271, <4 x float> %224) #9, !dbg !15
  %lds.const.gep669 = getelementptr i8, ptr addrspace(3) %address.local253, i64 6144, !dbg !15
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep669, align 8, !dbg !15
  %lds.const.gep670 = getelementptr i8, ptr addrspace(3) %address.local224, i64 4224, !dbg !15
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep670, align 8, !dbg !15
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %274, <4 x float> %273) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %275, <4 x float> %228) #9, !dbg !15
  %lds.const.gep671 = getelementptr i8, ptr addrspace(3) %address.local224, i64 6272, !dbg !15
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep671, align 8, !dbg !15
  %lds.const.gep672 = getelementptr i8, ptr addrspace(3) %address.local253, i64 4224, !dbg !15
  %279 = load <4 x half>, ptr addrspace(3) %lds.const.gep672, align 8, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %278, <4 x float> %277) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %279, <4 x float> %230) #9, !dbg !15
  %lds.const.gep673 = getelementptr i8, ptr addrspace(3) %address.local253, i64 6272, !dbg !15
  %282 = load <4 x half>, ptr addrspace(3) %lds.const.gep673, align 8, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %282, <4 x float> %281) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %284 = add nuw i64 %133, 6144, !dbg !16
  %285 = icmp slt i64 %284, 0, !dbg !16
  %286 = sub nuw i64 -6145, %133, !dbg !16
  %287 = select i1 %285, i64 %286, i64 %284, !dbg !16
  %288 = sdiv i64 %287, 8192, !dbg !16
  %.lobit21 = ashr i64 %284, 63, !dbg !16
  %289 = xor i64 %288, %.lobit21, !dbg !16
  %290 = add nsw i64 %289, %16, !dbg !16
  %291 = add nsw i64 %184, 96, !dbg !16
  %292 = icmp slt i64 %184, -96, !dbg !16
  %293 = sub nsw i64 -97, %184, !dbg !16
  %294 = select i1 %292, i64 %293, i64 %291, !dbg !16
  %295 = sdiv i64 %294, 128, !dbg !16
  %296 = sext i1 %292 to i64, !dbg !16
  %297 = xor i64 %295, %296, !dbg !16
  %298 = shl nsw i64 %297, 7, !dbg !16
  %299 = sub i64 %123, %298, !dbg !16
  %300 = shl i64 %290, 19, !dbg !16
  %301 = and i64 %300, 16252928, !dbg !16
  %302 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %301, !dbg !16
  %303 = getelementptr [2 x i8], ptr addrspace(1) %302, i64 %153, !dbg !16
  %304 = getelementptr [2 x i8], ptr addrspace(1) %303, i64 %299, !dbg !16
  %305 = getelementptr i8, ptr addrspace(1) %304, i64 192, !dbg !16
  %306 = getelementptr [2 x i8], ptr addrspace(1) %305, i64 %159, !dbg !16
  %307 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local204), !dbg !16
  %308 = zext i32 %307 to i64, !dbg !16
  %309 = inttoptr i64 %308 to ptr addrspace(3), !dbg !16
  %address.local403 = getelementptr [2 x i8], ptr addrspace(1) %306, i64 %address.local200, !dbg !16
  %310 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %311 = ptrtoint ptr addrspace(1) %address.local403 to i64, !dbg !16
  %312 = sub i64 %311, %310, !dbg !16
  %dma.buffer.offset697 = trunc i64 %312 to i32, !dbg !16
  %313 = trunc i64 %310 to i32, !dbg !16
  %314 = insertelement <4 x i32> poison, i32 %313, i64 0, !dbg !16
  %315 = lshr i64 %310, 32, !dbg !16
  %316 = trunc i64 %315 to i32, !dbg !16
  %317 = insertelement <4 x i32> %314, i32 %316, i64 1, !dbg !16
  %318 = insertelement <4 x i32> %317, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc698 = insertelement <4 x i32> %318, i32 131072, i64 3, !dbg !16
  %319 = ptrtoint ptr addrspace(3) %309 to i32, !dbg !16
  %dma.buffer.lds699 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %319), !dbg !16
  %320 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds699, i32 %dma.buffer.offset697, <4 x i32> %dma.buffer.rsrc698) #10, !dbg !16
  %321 = load <4 x half>, ptr addrspace(3) %address.local224, align 8, !dbg !15
  %322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %321, <4 x float> %272) #9, !dbg !15
  %lds.const.gep674 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2048, !dbg !15
  %323 = load <4 x half>, ptr addrspace(3) %lds.const.gep674, align 8, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %address.local253, align 8, !dbg !15
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %323, <4 x float> %322) #9, !dbg !15
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %324, <4 x float> %276) #9, !dbg !15
  %lds.const.gep675 = getelementptr i8, ptr addrspace(3) %address.local253, i64 2048, !dbg !15
  %327 = load <4 x half>, ptr addrspace(3) %lds.const.gep675, align 8, !dbg !15
  %lds.const.gep676 = getelementptr i8, ptr addrspace(3) %address.local224, i64 128, !dbg !15
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep676, align 8, !dbg !15
  %329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %327, <4 x float> %326) #9, !dbg !15
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %328, <4 x float> %280) #9, !dbg !15
  %lds.const.gep677 = getelementptr i8, ptr addrspace(3) %address.local224, i64 2176, !dbg !15
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep677, align 8, !dbg !15
  %lds.const.gep678 = getelementptr i8, ptr addrspace(3) %address.local253, i64 128, !dbg !15
  %332 = load <4 x half>, ptr addrspace(3) %lds.const.gep678, align 8, !dbg !15
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %331, <4 x float> %330) #9, !dbg !15
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %332, <4 x float> %283) #9, !dbg !15
  %lds.const.gep679 = getelementptr i8, ptr addrspace(3) %address.local253, i64 2176, !dbg !15
  %335 = load <4 x half>, ptr addrspace(3) %lds.const.gep679, align 8, !dbg !15
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %335, <4 x float> %334) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %337 = add nuw nsw i64 %153, %16, !dbg !17
  %338 = shl i64 %337, 19, !dbg !17
  %339 = and i64 %338, 16252928, !dbg !17
  %340 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %339, !dbg !17
  %341 = getelementptr [2 x i8], ptr addrspace(1) %340, i64 %159, !dbg !17
  %342 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local190), !dbg !17
  %343 = zext i32 %342 to i64, !dbg !17
  %344 = inttoptr i64 %343 to ptr addrspace(3), !dbg !17
  %address.local404 = shl nuw nsw i64 %address.local185, 5, !dbg !17
  %address.local405 = and i64 %address.local404, 8064, !dbg !17
  %address.local406 = shl nuw nsw i64 %address.local185, 3, !dbg !17
  %address.local407 = and i64 %address.local406, 24, !dbg !17
  %address.local408 = or disjoint i64 %address.local405, %address.local407, !dbg !17
  %address.local409 = getelementptr [2 x i8], ptr addrspace(1) %341, i64 %address.local408, !dbg !17
  %345 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %346 = ptrtoint ptr addrspace(1) %address.local409 to i64, !dbg !17
  %347 = sub i64 %346, %345, !dbg !17
  %dma.buffer.offset694 = trunc i64 %347 to i32, !dbg !17
  %348 = trunc i64 %345 to i32, !dbg !17
  %349 = insertelement <4 x i32> poison, i32 %348, i64 0, !dbg !17
  %350 = lshr i64 %345, 32, !dbg !17
  %351 = trunc i64 %350 to i32, !dbg !17
  %352 = insertelement <4 x i32> %349, i32 %351, i64 1, !dbg !17
  %353 = insertelement <4 x i32> %352, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc695 = insertelement <4 x i32> %353, i32 131072, i64 3, !dbg !17
  %354 = ptrtoint ptr addrspace(3) %344 to i32, !dbg !17
  %dma.buffer.lds696 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %354), !dbg !17
  %355 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds696, i32 %dma.buffer.offset694, <4 x i32> %dma.buffer.rsrc695) #10, !dbg !17
  %lds.const.gep680 = getelementptr i8, ptr addrspace(3) %address.local224, i64 4096, !dbg !15
  %356 = load <4 x half>, ptr addrspace(3) %lds.const.gep680, align 8, !dbg !15
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %356, <4 x float> %325) #9, !dbg !15
  %lds.const.gep681 = getelementptr i8, ptr addrspace(3) %address.local224, i64 6144, !dbg !15
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep681, align 8, !dbg !15
  %lds.const.gep682 = getelementptr i8, ptr addrspace(3) %address.local253, i64 4096, !dbg !15
  %359 = load <4 x half>, ptr addrspace(3) %lds.const.gep682, align 8, !dbg !15
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %358, <4 x float> %357) #9, !dbg !15
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %359, <4 x float> %329) #9, !dbg !15
  %lds.const.gep683 = getelementptr i8, ptr addrspace(3) %address.local253, i64 6144, !dbg !15
  %362 = load <4 x half>, ptr addrspace(3) %lds.const.gep683, align 8, !dbg !15
  %lds.const.gep684 = getelementptr i8, ptr addrspace(3) %address.local224, i64 4224, !dbg !15
  %363 = load <4 x half>, ptr addrspace(3) %lds.const.gep684, align 8, !dbg !15
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %362, <4 x float> %361) #9, !dbg !15
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %363, <4 x float> %333) #9, !dbg !15
  %lds.const.gep685 = getelementptr i8, ptr addrspace(3) %address.local224, i64 6272, !dbg !15
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep685, align 8, !dbg !15
  %lds.const.gep686 = getelementptr i8, ptr addrspace(3) %address.local253, i64 4224, !dbg !15
  %367 = load <4 x half>, ptr addrspace(3) %lds.const.gep686, align 8, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %366, <4 x float> %365) #9, !dbg !15
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %367, <4 x float> %336) #9, !dbg !15
  %lds.const.gep687 = getelementptr i8, ptr addrspace(3) %address.local253, i64 6272, !dbg !15
  %370 = load <4 x half>, ptr addrspace(3) %lds.const.gep687, align 8, !dbg !15
  %371 = or disjoint i64 %150, 63, !dbg !13
  %372 = icmp uge i64 %50, %371, !dbg !13
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %370, <4 x float> %369) #9, !dbg !15
  %374 = and i1 %118, %372, !dbg !13
  br i1 %374, label %375, label %376, !dbg !13

375:                                              ; preds = %149
  br label %419, !dbg !13

376:                                              ; preds = %149
  %377 = add i64 %50, %32, !dbg !13
  %378 = shl nuw nsw i64 %51, 4, !dbg !13
  %379 = add i64 %377, %378, !dbg !13
  %380 = and i32 %31, 240, !dbg !13
  %index.compact = zext nneg i32 %380 to i64, !dbg !13
  %381 = sub i64 %379, %index.compact, !dbg !13
  %382 = add i64 %381, 1, !dbg !22
  %383 = and i64 %32, 48, !dbg !13
  %384 = or disjoint i64 %150, %383, !dbg !13
  %385 = lshr i64 %382, 32, !dbg !23
  %compare.high = trunc i64 %385 to i32, !dbg !23
  %386 = lshr i64 %384, 32, !dbg !23
  %compare.high595 = trunc i64 %386 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high595, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high595, !dbg !23
  %compare.low.base = trunc i64 %382 to i32, !dbg !23
  %compare.low.base596 = trunc i64 %384 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base596, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %387 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low597 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result598 = select i1 %compare.high.equal, i1 %compare.low597, i1 %compare.high.order, !dbg !23
  %388 = select i1 %compare.result598, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base599 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local600 = or i32 %compare.local.base599, 2, !dbg !23
  %compare.low601 = icmp ugt i32 %compare.low.base, %compare.local600, !dbg !23
  %compare.result602 = select i1 %compare.high.equal, i1 %compare.low601, i1 %compare.high.order, !dbg !23
  %389 = select i1 %compare.result602, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base603 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local604 = or i32 %compare.local.base603, 3, !dbg !23
  %compare.low605 = icmp ugt i32 %compare.low.base, %compare.local604, !dbg !23
  %compare.result606 = select i1 %compare.high.equal, i1 %compare.low605, i1 %compare.high.order, !dbg !23
  %390 = select i1 %compare.result606, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base607 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local608 = or i32 %compare.local.base607, 4, !dbg !23
  %compare.low609 = icmp ugt i32 %compare.low.base, %compare.local608, !dbg !23
  %compare.result610 = select i1 %compare.high.equal, i1 %compare.low609, i1 %compare.high.order, !dbg !23
  %391 = select i1 %compare.result610, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base611 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local612 = or i32 %compare.local.base611, 5, !dbg !23
  %compare.low613 = icmp ugt i32 %compare.low.base, %compare.local612, !dbg !23
  %compare.result614 = select i1 %compare.high.equal, i1 %compare.low613, i1 %compare.high.order, !dbg !23
  %392 = select i1 %compare.result614, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base615 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local616 = or i32 %compare.local.base615, 6, !dbg !23
  %compare.low617 = icmp ugt i32 %compare.low.base, %compare.local616, !dbg !23
  %compare.result618 = select i1 %compare.high.equal, i1 %compare.low617, i1 %compare.high.order, !dbg !23
  %393 = select i1 %compare.result618, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base619 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local620 = or i32 %compare.local.base619, 7, !dbg !23
  %compare.low621 = icmp ugt i32 %compare.low.base, %compare.local620, !dbg !23
  %compare.result622 = select i1 %compare.high.equal, i1 %compare.low621, i1 %compare.high.order, !dbg !23
  %394 = select i1 %compare.result622, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base623 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local624 = or i32 %compare.local.base623, 8, !dbg !23
  %compare.low625 = icmp ugt i32 %compare.low.base, %compare.local624, !dbg !23
  %compare.result626 = select i1 %compare.high.equal, i1 %compare.low625, i1 %compare.high.order, !dbg !23
  %395 = select i1 %compare.result626, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base627 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local628 = or i32 %compare.local.base627, 9, !dbg !23
  %compare.low629 = icmp ugt i32 %compare.low.base, %compare.local628, !dbg !23
  %compare.result630 = select i1 %compare.high.equal, i1 %compare.low629, i1 %compare.high.order, !dbg !23
  %396 = select i1 %compare.result630, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base631 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local632 = or i32 %compare.local.base631, 10, !dbg !23
  %compare.low633 = icmp ugt i32 %compare.low.base, %compare.local632, !dbg !23
  %compare.result634 = select i1 %compare.high.equal, i1 %compare.low633, i1 %compare.high.order, !dbg !23
  %397 = select i1 %compare.result634, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base635 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local636 = or i32 %compare.local.base635, 11, !dbg !23
  %compare.low637 = icmp ugt i32 %compare.low.base, %compare.local636, !dbg !23
  %compare.result638 = select i1 %compare.high.equal, i1 %compare.low637, i1 %compare.high.order, !dbg !23
  %398 = select i1 %compare.result638, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base639 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local640 = or i32 %compare.local.base639, 12, !dbg !23
  %compare.low641 = icmp ugt i32 %compare.low.base, %compare.local640, !dbg !23
  %compare.result642 = select i1 %compare.high.equal, i1 %compare.low641, i1 %compare.high.order, !dbg !23
  %399 = select i1 %compare.result642, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base643 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local644 = or i32 %compare.local.base643, 13, !dbg !23
  %compare.low645 = icmp ugt i32 %compare.low.base, %compare.local644, !dbg !23
  %compare.result646 = select i1 %compare.high.equal, i1 %compare.low645, i1 %compare.high.order, !dbg !23
  %400 = select i1 %compare.result646, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base647 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local648 = or i32 %compare.local.base647, 14, !dbg !23
  %compare.low649 = icmp ugt i32 %compare.low.base, %compare.local648, !dbg !23
  %compare.result650 = select i1 %compare.high.equal, i1 %compare.low649, i1 %compare.high.order, !dbg !23
  %401 = select i1 %compare.result650, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base651 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base596, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local652 = or i32 %compare.local.base651, 15, !dbg !23
  %compare.low653 = icmp ugt i32 %compare.low.base, %compare.local652, !dbg !23
  %compare.result654 = select i1 %compare.high.equal, i1 %compare.low653, i1 %compare.high.order, !dbg !23
  %402 = select i1 %compare.result654, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
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
  %483 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %482, <1 x float> %138), !dbg !26
  %484 = shufflevector <1 x float> %483, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %485 = fsub <4 x float> %423, %484, !dbg !27
  %486 = fsub <4 x float> %429, %484, !dbg !27
  %487 = fsub <4 x float> %435, %484, !dbg !27
  %488 = fsub <4 x float> %441, %484, !dbg !27
  %489 = fsub <1 x float> %138, %483, !dbg !28
  %490 = extractelement <1 x float> %489, i64 0, !dbg !29
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !29
  %492 = insertelement <4 x float> poison, float %491, i64 0, !dbg !34
  %493 = shufflevector <4 x float> %492, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %494 = fmul <4 x float> %146, %493, !dbg !34
  %495 = fmul <4 x float> %145, %493, !dbg !34
  %496 = fmul <4 x float> %143, %493, !dbg !34
  %497 = fmul <4 x float> %142, %493, !dbg !34
  %498 = fmul <4 x float> %140, %493, !dbg !34
  %499 = fmul <4 x float> %139, %493, !dbg !34
  %500 = fmul <4 x float> %135, %493, !dbg !34
  %501 = fmul <4 x float> %134, %493, !dbg !34
  %502 = extractelement <1 x float> %137, i64 0, !dbg !30
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
  %math.result132 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %534) #9, !dbg !33, !frisk.register_lifetimes !1
  %535 = fptrunc <4 x float> %math.result132 to <4 x half>, !dbg !33
  %math.packed133 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %535) #9, !dbg !32, !frisk.register_lifetimes !1
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
  %math.result134 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %551) #9, !dbg !33, !frisk.register_lifetimes !1
  %552 = fptrunc <4 x float> %math.result134 to <4 x half>, !dbg !33
  %math.packed135 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %552) #9, !dbg !32, !frisk.register_lifetimes !1
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
  %math.result136 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %568) #9, !dbg !33, !frisk.register_lifetimes !1
  %569 = fptrunc <4 x float> %math.result136 to <4 x half>, !dbg !33
  %math.packed137 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %569) #9, !dbg !32, !frisk.register_lifetimes !1
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
  %address.thread410 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local411 = zext nneg i32 %address.thread410 to i64, !dbg !8
  %address.local412 = and i64 %address.local411, 60, !dbg !14
  %address.local413 = add nuw nsw i64 %address.local412, %address.local411, !dbg !14
  %address.local414 = shl nuw nsw i64 %address.local413, 3, !dbg !14
  %address.local415 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !14
  %address.local416 = and i64 %address.local411, 63, !dbg !11
  %address.local417 = lshr i64 %address.local416, 4, !dbg !14
  %address.local418 = mul nuw nsw i64 %address.local417, 384, !dbg !14
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) %address.local415, i64 %address.local418, !dbg !14
  %address.local420 = lshr i64 %address.local411, 2, !dbg !15
  %address.local421 = mul nsw i64 %address.local420, -32, !dbg !14
  %address.local422 = getelementptr [2 x i8], ptr addrspace(3) %address.local419, i64 %address.local421, !dbg !14
  %581 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local422, i32 0) #0, !dbg !14
  %582 = bitcast <4 x i32> %581 to <8 x half>, !dbg !14
  %address.local423 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !14
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) %address.local423, i64 %address.local418, !dbg !14
  %address.local425 = getelementptr [2 x i8], ptr addrspace(3) %address.local424, i64 %address.local421, !dbg !14
  %address.local426 = getelementptr i8, ptr addrspace(3) %address.local425, i64 256, !dbg !14
  %583 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local426, i32 0) #0, !dbg !14
  %584 = bitcast <4 x i32> %583 to <8 x half>, !dbg !14
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !14
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) %address.local427, i64 %address.local418, !dbg !14
  %address.local429 = getelementptr [2 x i8], ptr addrspace(3) %address.local428, i64 %address.local421, !dbg !14
  %address.local430 = getelementptr i8, ptr addrspace(3) %address.local429, i64 512, !dbg !14
  %585 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local430, i32 0) #0, !dbg !14
  %586 = bitcast <4 x i32> %585 to <8 x half>, !dbg !14
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local414, !dbg !14
  %address.local432 = getelementptr [2 x i8], ptr addrspace(3) %address.local431, i64 %address.local418, !dbg !14
  %address.local433 = getelementptr [2 x i8], ptr addrspace(3) %address.local432, i64 %address.local421, !dbg !14
  %address.local434 = getelementptr i8, ptr addrspace(3) %address.local433, i64 768, !dbg !14
  %587 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local434, i32 0) #0, !dbg !14
  %588 = bitcast <4 x i32> %587 to <8 x half>, !dbg !14
  %589 = add nuw nsw i64 %153, %16, !dbg !17
  %590 = shl i64 %589, 19, !dbg !17
  %591 = and i64 %590, 16252928, !dbg !17
  %592 = add i64 %591, %159, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %593 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %592, !dbg !17
  %594 = getelementptr i8, ptr addrspace(1) %593, i64 64, !dbg !17
  %address.local435 = shl nuw nsw i64 %address.local411, 3, !dbg !8
  %address.local436 = and i64 %address.local435, 1536, !dbg !8
  %address.local437 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local436, !dbg !8
  %address.local438 = ptrtoint ptr addrspace(3) %address.local437 to i64, !dbg !8
  %address.local439 = trunc i64 %address.local438 to i32, !dbg !8
  %595 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %596 = zext i32 %595 to i64, !dbg !17
  %597 = inttoptr i64 %596 to ptr addrspace(3), !dbg !17
  %address.local440 = shl nuw nsw i64 %address.local411, 5, !dbg !17
  %address.local441 = and i64 %address.local440, 8064, !dbg !17
  %address.local442 = shl nuw nsw i64 %address.local411, 3, !dbg !17
  %address.local443 = and i64 %address.local442, 24, !dbg !17
  %address.local444 = or disjoint i64 %address.local441, %address.local443, !dbg !17
  %address.local445 = getelementptr [2 x i8], ptr addrspace(1) %594, i64 %address.local444, !dbg !17
  %598 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %599 = ptrtoint ptr addrspace(1) %address.local445 to i64, !dbg !17
  %600 = sub i64 %599, %598, !dbg !17
  %dma.buffer.offset691 = trunc i64 %600 to i32, !dbg !17
  %601 = trunc i64 %598 to i32, !dbg !17
  %602 = insertelement <4 x i32> poison, i32 %601, i64 0, !dbg !17
  %603 = lshr i64 %598, 32, !dbg !17
  %604 = trunc i64 %603 to i32, !dbg !17
  %605 = insertelement <4 x i32> %602, i32 %604, i64 1, !dbg !17
  %606 = insertelement <4 x i32> %605, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc692 = insertelement <4 x i32> %606, i32 131072, i64 3, !dbg !17
  %607 = ptrtoint ptr addrspace(3) %597 to i32, !dbg !17
  %dma.buffer.lds693 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %607), !dbg !17
  %608 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds693, i32 %dma.buffer.offset691, <4 x i32> %dma.buffer.rsrc692) #10, !dbg !17
  %609 = shufflevector <8 x half> %582, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %609, <4 x float> %501) #9, !dbg !14
  %611 = shufflevector <8 x half> %584, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %612 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %611, <4 x float> %610) #9, !dbg !14
  %613 = shufflevector <8 x half> %586, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %613, <4 x float> %612) #9, !dbg !14
  %615 = shufflevector <8 x half> %588, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %615, <4 x float> %614) #9, !dbg !14
  %617 = shufflevector <8 x half> %582, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %617, <4 x float> %500) #9, !dbg !14
  %619 = shufflevector <8 x half> %584, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %619, <4 x float> %618) #9, !dbg !14
  %621 = shufflevector <8 x half> %586, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %621, <4 x float> %620) #9, !dbg !14
  %623 = shufflevector <8 x half> %588, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %623, <4 x float> %622) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %625 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local422, i32 0) #0, !dbg !14
  %626 = bitcast <4 x i32> %625 to <8 x half>, !dbg !14
  %627 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local426, i32 0) #0, !dbg !14
  %628 = bitcast <4 x i32> %627 to <8 x half>, !dbg !14
  %629 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local430, i32 0) #0, !dbg !14
  %630 = bitcast <4 x i32> %629 to <8 x half>, !dbg !14
  %631 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local434, i32 0) #0, !dbg !14
  %632 = bitcast <4 x i32> %631 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %633 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %592, !dbg !17
  %634 = getelementptr i8, ptr addrspace(1) %633, i64 128, !dbg !17
  %635 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %636 = zext i32 %635 to i64, !dbg !17
  %637 = inttoptr i64 %636 to ptr addrspace(3), !dbg !17
  %address.local446 = getelementptr [2 x i8], ptr addrspace(1) %634, i64 %address.local444, !dbg !17
  %638 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %639 = ptrtoint ptr addrspace(1) %address.local446 to i64, !dbg !17
  %640 = sub i64 %639, %638, !dbg !17
  %dma.buffer.offset688 = trunc i64 %640 to i32, !dbg !17
  %641 = trunc i64 %638 to i32, !dbg !17
  %642 = insertelement <4 x i32> poison, i32 %641, i64 0, !dbg !17
  %643 = lshr i64 %638, 32, !dbg !17
  %644 = trunc i64 %643 to i32, !dbg !17
  %645 = insertelement <4 x i32> %642, i32 %644, i64 1, !dbg !17
  %646 = insertelement <4 x i32> %645, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc689 = insertelement <4 x i32> %646, i32 131072, i64 3, !dbg !17
  %647 = ptrtoint ptr addrspace(3) %637 to i32, !dbg !17
  %dma.buffer.lds690 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %647), !dbg !17
  %648 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds690, i32 %dma.buffer.offset688, <4 x i32> %dma.buffer.rsrc689) #10, !dbg !17
  %649 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %649, <4 x float> %499) #9, !dbg !14
  %651 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %651, <4 x float> %650) #9, !dbg !14
  %653 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %653, <4 x float> %652) #9, !dbg !14
  %655 = shufflevector <8 x half> %632, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %655, <4 x float> %654) #9, !dbg !14
  %657 = shufflevector <8 x half> %626, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %657, <4 x float> %498) #9, !dbg !14
  %659 = shufflevector <8 x half> %628, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %659, <4 x float> %658) #9, !dbg !14
  %661 = shufflevector <8 x half> %630, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %662 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %661, <4 x float> %660) #9, !dbg !14
  %663 = shufflevector <8 x half> %632, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %663, <4 x float> %662) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %665 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local422, i32 0) #0, !dbg !14
  %666 = bitcast <4 x i32> %665 to <8 x half>, !dbg !14
  %667 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local426, i32 0) #0, !dbg !14
  %668 = bitcast <4 x i32> %667 to <8 x half>, !dbg !14
  %669 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local430, i32 0) #0, !dbg !14
  %670 = bitcast <4 x i32> %669 to <8 x half>, !dbg !14
  %671 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local434, i32 0) #0, !dbg !14
  %672 = bitcast <4 x i32> %671 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %673 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %592, !dbg !17
  %674 = getelementptr i8, ptr addrspace(1) %673, i64 192, !dbg !17
  %675 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local439), !dbg !17
  %676 = zext i32 %675 to i64, !dbg !17
  %677 = inttoptr i64 %676 to ptr addrspace(3), !dbg !17
  %address.local447 = getelementptr [2 x i8], ptr addrspace(1) %674, i64 %address.local444, !dbg !17
  %678 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %679 = ptrtoint ptr addrspace(1) %address.local447 to i64, !dbg !17
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
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %691, <4 x float> %690) #9, !dbg !14
  %693 = shufflevector <8 x half> %670, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %693, <4 x float> %692) #9, !dbg !14
  %695 = shufflevector <8 x half> %672, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %695, <4 x float> %694) #9, !dbg !14
  %697 = shufflevector <8 x half> %666, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %697, <4 x float> %496) #9, !dbg !14
  %699 = shufflevector <8 x half> %668, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %699, <4 x float> %698) #9, !dbg !14
  %701 = shufflevector <8 x half> %670, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %702 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %701, <4 x float> %700) #9, !dbg !14
  %703 = shufflevector <8 x half> %672, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %704 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %703, <4 x float> %702) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %705 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local422, i32 0) #0, !dbg !14
  %706 = bitcast <4 x i32> %705 to <8 x half>, !dbg !14
  %707 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local426, i32 0) #0, !dbg !14
  %708 = bitcast <4 x i32> %707 to <8 x half>, !dbg !14
  %709 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local430, i32 0) #0, !dbg !14
  %710 = bitcast <4 x i32> %709 to <8 x half>, !dbg !14
  %711 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local434, i32 0) #0, !dbg !14
  %712 = bitcast <4 x i32> %711 to <8 x half>, !dbg !14
  %713 = shufflevector <8 x half> %706, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %714 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %713, <4 x float> %495) #9, !dbg !14
  %715 = shufflevector <8 x half> %708, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %715, <4 x float> %714) #9, !dbg !14
  %717 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %717, <4 x float> %716) #9, !dbg !14
  %719 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %720 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %719, <4 x float> %718) #9, !dbg !14
  %721 = shufflevector <8 x half> %706, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %722 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %721, <4 x float> %494) #9, !dbg !14
  %723 = shufflevector <8 x half> %708, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %724 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed133, <4 x half> %723, <4 x float> %722) #9, !dbg !14
  %725 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed135, <4 x half> %725, <4 x float> %724) #9, !dbg !14
  %727 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %728 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed137, <4 x half> %727, <4 x float> %726) #9, !dbg !14
  %729 = shufflevector <4 x float> %616, <4 x float> %624, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %730 = shufflevector <4 x float> %656, <4 x float> %664, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %731 = shufflevector <4 x float> %696, <4 x float> %704, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %732 = shufflevector <4 x float> %720, <4 x float> %728, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %733 = add nuw nsw i64 %133, 1, !dbg !7
  br label %132, !dbg !7

734:                                              ; preds = %132
  %735 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %735, 0, !dbg !7
  %736 = and i32 %31, 32, !dbg !7
  %.not18 = icmp eq i32 %736, 0, !dbg !7
  %737 = extractelement <8 x float> %147, i64 0, !dbg !7
  %738 = extractelement <8 x float> %147, i64 1, !dbg !7
  %739 = extractelement <8 x float> %147, i64 2, !dbg !7
  %740 = extractelement <8 x float> %147, i64 3, !dbg !7
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
  %777 = extractelement <8 x float> %144, i64 0, !dbg !7
  %778 = extractelement <8 x float> %144, i64 1, !dbg !7
  %779 = extractelement <8 x float> %144, i64 2, !dbg !7
  %780 = extractelement <8 x float> %144, i64 3, !dbg !7
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
  %805 = extractelement <8 x float> %141, i64 0, !dbg !7
  %806 = extractelement <8 x float> %141, i64 1, !dbg !7
  %807 = extractelement <8 x float> %141, i64 2, !dbg !7
  %808 = extractelement <8 x float> %141, i64 3, !dbg !7
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
  %833 = extractelement <8 x float> %136, i64 0, !dbg !7
  %834 = extractelement <8 x float> %136, i64 1, !dbg !7
  %835 = extractelement <8 x float> %136, i64 2, !dbg !7
  %836 = extractelement <8 x float> %136, i64 3, !dbg !7
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
  %855 = shufflevector <1 x float> %137, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %856 = extractelement <8 x float> %147, i64 4, !dbg !7
  %857 = extractelement <8 x float> %147, i64 5, !dbg !7
  %858 = extractelement <8 x float> %147, i64 6, !dbg !7
  %859 = extractelement <8 x float> %147, i64 7, !dbg !7
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
  %884 = extractelement <8 x float> %144, i64 4, !dbg !7
  %885 = extractelement <8 x float> %144, i64 5, !dbg !7
  %886 = extractelement <8 x float> %144, i64 6, !dbg !7
  %887 = extractelement <8 x float> %144, i64 7, !dbg !7
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
  %912 = extractelement <8 x float> %141, i64 4, !dbg !7
  %913 = extractelement <8 x float> %141, i64 5, !dbg !7
  %914 = extractelement <8 x float> %141, i64 6, !dbg !7
  %915 = extractelement <8 x float> %141, i64 7, !dbg !7
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
  %940 = extractelement <8 x float> %136, i64 4, !dbg !7
  %941 = extractelement <8 x float> %136, i64 5, !dbg !7
  %942 = extractelement <8 x float> %136, i64 6, !dbg !7
  %943 = extractelement <8 x float> %136, i64 7, !dbg !7
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
  %993 = add nsw i64 %16, %22, !dbg !20
  %994 = shl i64 %993, 19, !dbg !20
  %995 = and i64 %994, 16252928, !dbg !20
  %address.thread448 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local449 = zext nneg i32 %address.thread448 to i64, !dbg !8
  %address.local450 = add i64 %25, %address.local449, !dbg !20
  %address.local451 = lshr i64 %address.local449, 6, !dbg !11
  %address.local452 = shl nuw nsw i64 %address.local451, 4, !dbg !20
  %address.local453 = add i64 %address.local450, %address.local452, !dbg !20
  %address.local454 = and i64 %address.local449, 240, !dbg !20
  %address.local455 = sub i64 %address.local453, %address.local454, !dbg !20
  %address.local456 = add i64 %address.local455, %26, !dbg !20
  %address.local457 = shl nuw nsw i64 %address.local456, 7, !dbg !20
  %address.local458 = add nuw nsw i64 %995, %address.local457, !dbg !20
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local460 = lshr i64 %address.local449, 2, !dbg !11
  %address.local461 = and i64 %address.local460, 12, !dbg !11
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local459, i64 %address.local461, !dbg !20
  store half %992, ptr addrspace(1) %address.local462, align 2, !dbg !20
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local461, !dbg !20
  %address.local465 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local464, i64 2, !dbg !20
  store half %991, ptr addrspace(1) %address.local465, align 2, !dbg !20
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local461, !dbg !20
  %address.local468 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local467, i64 4, !dbg !20
  store half %990, ptr addrspace(1) %address.local468, align 2, !dbg !20
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local461, !dbg !20
  %address.local471 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local470, i64 6, !dbg !20
  store half %989, ptr addrspace(1) %address.local471, align 2, !dbg !20
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local461, !dbg !20
  %address.local474 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local473, i64 32, !dbg !20
  store half %977, ptr addrspace(1) %address.local474, align 2, !dbg !20
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local461, !dbg !20
  %address.local477 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local476, i64 34, !dbg !20
  store half %976, ptr addrspace(1) %address.local477, align 2, !dbg !20
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local461, !dbg !20
  %address.local480 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local479, i64 36, !dbg !20
  store half %975, ptr addrspace(1) %address.local480, align 2, !dbg !20
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local458, !dbg !20
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local481, i64 %address.local461, !dbg !20
  %address.local483 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local482, i64 38, !dbg !20
  store half %974, ptr addrspace(1) %address.local483, align 2, !dbg !20
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
  %1016 = shl nuw nsw i64 %24, 19, !dbg !20
  %address.thread484 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local485 = zext nneg i32 %address.thread484 to i64, !dbg !8
  %address.local486 = add i64 %25, %address.local485, !dbg !20
  %address.local487 = lshr i64 %address.local485, 6, !dbg !11
  %address.local488 = shl nuw nsw i64 %address.local487, 4, !dbg !20
  %address.local489 = add i64 %address.local486, %address.local488, !dbg !20
  %address.local490 = and i64 %address.local485, 240, !dbg !20
  %address.local491 = sub i64 %address.local489, %address.local490, !dbg !20
  %address.local492 = add i64 %address.local491, %26, !dbg !20
  %address.local493 = shl nuw nsw i64 %address.local492, 7, !dbg !20
  %address.local494 = add nuw nsw i64 %1016, %address.local493, !dbg !20
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local496 = lshr i64 %address.local485, 2, !dbg !11
  %address.local497 = and i64 %address.local496, 12, !dbg !11
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local497, !dbg !20
  %address.local499 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local498, i64 64, !dbg !20
  store half %1015, ptr addrspace(1) %address.local499, align 2, !dbg !20
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local500, i64 %address.local497, !dbg !20
  %address.local502 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local501, i64 66, !dbg !20
  store half %1014, ptr addrspace(1) %address.local502, align 2, !dbg !20
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local503, i64 %address.local497, !dbg !20
  %address.local505 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local504, i64 68, !dbg !20
  store half %1013, ptr addrspace(1) %address.local505, align 2, !dbg !20
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local506, i64 %address.local497, !dbg !20
  %address.local508 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local507, i64 70, !dbg !20
  store half %1012, ptr addrspace(1) %address.local508, align 2, !dbg !20
  %address.local509 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local509, i64 %address.local497, !dbg !20
  %address.local511 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local510, i64 96, !dbg !20
  store half %1006, ptr addrspace(1) %address.local511, align 2, !dbg !20
  %address.local512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local512, i64 %address.local497, !dbg !20
  %address.local514 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local513, i64 98, !dbg !20
  store half %1005, ptr addrspace(1) %address.local514, align 2, !dbg !20
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local515, i64 %address.local497, !dbg !20
  %address.local517 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local516, i64 100, !dbg !20
  store half %1004, ptr addrspace(1) %address.local517, align 2, !dbg !20
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local494, !dbg !20
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local518, i64 %address.local497, !dbg !20
  %address.local520 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local519, i64 102, !dbg !20
  store half %1003, ptr addrspace(1) %address.local520, align 2, !dbg !20
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
  %1037 = shl nuw nsw i64 %24, 19, !dbg !20
  %address.thread521 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local522 = zext nneg i32 %address.thread521 to i64, !dbg !8
  %address.local523 = add i64 %25, %address.local522, !dbg !20
  %address.local524 = lshr i64 %address.local522, 6, !dbg !11
  %address.local525 = shl nuw nsw i64 %address.local524, 4, !dbg !20
  %address.local526 = add i64 %address.local523, %address.local525, !dbg !20
  %address.local527 = and i64 %address.local522, 240, !dbg !20
  %address.local528 = sub i64 %address.local526, %address.local527, !dbg !20
  %address.local529 = add i64 %address.local528, %26, !dbg !20
  %address.local530 = shl nuw nsw i64 %address.local529, 7, !dbg !20
  %address.local531 = add nuw nsw i64 %1037, %address.local530, !dbg !20
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local533 = lshr i64 %address.local522, 2, !dbg !11
  %address.local534 = and i64 %address.local533, 12, !dbg !11
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local532, i64 %address.local534, !dbg !20
  %address.local536 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local535, i64 128, !dbg !20
  store half %1036, ptr addrspace(1) %address.local536, align 2, !dbg !20
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local537, i64 %address.local534, !dbg !20
  %address.local539 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local538, i64 130, !dbg !20
  store half %1035, ptr addrspace(1) %address.local539, align 2, !dbg !20
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local534, !dbg !20
  %address.local542 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local541, i64 132, !dbg !20
  store half %1034, ptr addrspace(1) %address.local542, align 2, !dbg !20
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local543, i64 %address.local534, !dbg !20
  %address.local545 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local544, i64 134, !dbg !20
  store half %1033, ptr addrspace(1) %address.local545, align 2, !dbg !20
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local534, !dbg !20
  %address.local548 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local547, i64 160, !dbg !20
  store half %1027, ptr addrspace(1) %address.local548, align 2, !dbg !20
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local534, !dbg !20
  %address.local551 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local550, i64 162, !dbg !20
  store half %1026, ptr addrspace(1) %address.local551, align 2, !dbg !20
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local534, !dbg !20
  %address.local554 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local553, i64 164, !dbg !20
  store half %1025, ptr addrspace(1) %address.local554, align 2, !dbg !20
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local531, !dbg !20
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local534, !dbg !20
  %address.local557 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local556, i64 166, !dbg !20
  store half %1024, ptr addrspace(1) %address.local557, align 2, !dbg !20
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
  %1058 = shl nuw nsw i64 %24, 19, !dbg !20
  %address.thread558 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local559 = zext nneg i32 %address.thread558 to i64, !dbg !8
  %address.local560 = add i64 %25, %address.local559, !dbg !20
  %address.local561 = lshr i64 %address.local559, 6, !dbg !11
  %address.local562 = shl nuw nsw i64 %address.local561, 4, !dbg !20
  %address.local563 = add i64 %address.local560, %address.local562, !dbg !20
  %address.local564 = and i64 %address.local559, 240, !dbg !20
  %address.local565 = sub i64 %address.local563, %address.local564, !dbg !20
  %address.local566 = add i64 %address.local565, %26, !dbg !20
  %address.local567 = shl nuw nsw i64 %address.local566, 7, !dbg !20
  %address.local568 = add nuw nsw i64 %1058, %address.local567, !dbg !20
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local570 = lshr i64 %address.local559, 2, !dbg !11
  %address.local571 = and i64 %address.local570, 12, !dbg !11
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local571, !dbg !20
  %address.local573 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local572, i64 192, !dbg !20
  store half %1057, ptr addrspace(1) %address.local573, align 2, !dbg !20
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local574, i64 %address.local571, !dbg !20
  %address.local576 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local575, i64 194, !dbg !20
  store half %1056, ptr addrspace(1) %address.local576, align 2, !dbg !20
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local571, !dbg !20
  %address.local579 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local578, i64 196, !dbg !20
  store half %1055, ptr addrspace(1) %address.local579, align 2, !dbg !20
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local580, i64 %address.local571, !dbg !20
  %address.local582 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local581, i64 198, !dbg !20
  store half %1054, ptr addrspace(1) %address.local582, align 2, !dbg !20
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local571, !dbg !20
  %address.local585 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local584, i64 224, !dbg !20
  store half %1048, ptr addrspace(1) %address.local585, align 2, !dbg !20
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local586, i64 %address.local571, !dbg !20
  %address.local588 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local587, i64 226, !dbg !20
  store half %1047, ptr addrspace(1) %address.local588, align 2, !dbg !20
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local571, !dbg !20
  %address.local591 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local590, i64 228, !dbg !20
  store half %1046, ptr addrspace(1) %address.local591, align 2, !dbg !20
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local571, !dbg !20
  %address.local594 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local593, i64 230, !dbg !20
  store half %1045, ptr addrspace(1) %address.local594, align 2, !dbg !20
  br label %1059, !dbg !20

1059:                                             ; preds = %1039, %1038
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!37}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0002-test_input_1x32x4096x128-64-64-32-online/online-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 77, column: 19, scope: !4)
!14 = !DILocation(line: 102, column: 23, scope: !4)
!15 = !DILocation(line: 73, column: 26, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
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
