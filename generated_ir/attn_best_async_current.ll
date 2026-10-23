; ModuleID = '3rd/deepgengraph/test/test_input_sd16.mlir'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !10
  %.lobit = ashr i32 %5, 31, !dbg !12
  %10 = xor i32 %5, %.lobit, !dbg !12
  %11 = sdiv i32 %10, 32, !dbg !12
  %.lobit1 = ashr i32 %5, 31, !dbg !12
  %12 = xor i32 %11, %.lobit1, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !12
  %14 = add nsw i64 %8, %13, !dbg !12
  %15 = and i64 %14, 31, !dbg !12
  %16 = shl nsw i64 %6, 7, !dbg !12
  %17 = mul nsw i64 %13, -4096, !dbg !12
  %18 = add nsw i64 %16, %17, !dbg !12
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !33
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = or disjoint i64 %9, 127, !dbg !14
  %23 = add nsw i64 %9, 128, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  br label %25, !dbg !16

25:                                               ; preds = %767, %4
  %26 = phi i64 [ %1466, %767 ], [ 0, %4 ], !dbg !16
  %27 = phi [2 x <8 x float>] [ %1453, %767 ], [ zeroinitializer, %4 ], !dbg !16
  %28 = phi [2 x <1 x float>] [ %1075, %767 ], [ zeroinitializer, %4 ], !dbg !16
  %29 = phi [2 x <8 x float>] [ %1457, %767 ], [ zeroinitializer, %4 ], !dbg !16
  %30 = phi [2 x <8 x float>] [ %1461, %767 ], [ zeroinitializer, %4 ], !dbg !16
  %31 = phi [2 x <8 x float>] [ %1465, %767 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %26, %6, !dbg !16
  br i1 %.not, label %1467, label %32, !dbg !16

32:                                               ; preds = %25
  %33 = shl nsw i64 %26, 7, !dbg !16
  %34 = lshr i64 %26, 5, !dbg !17
  %35 = shl i64 %34, 12, !dbg !17
  %36 = sub i64 %33, %35, !dbg !17
  br label %37, !dbg !22

37:                                               ; preds = %41, %32
  %38 = phi i64 [ %233, %41 ], [ 0, %32 ], !dbg !22
  %39 = phi [2 x <32 x float>] [ %232, %41 ], [ zeroinitializer, %32 ], !dbg !22
  %40 = icmp samesign ult i64 %38, 4, !dbg !22
  br i1 %40, label %41, label %234, !dbg !22

41:                                               ; preds = %37
  %42 = shl nuw nsw i64 %38, 5, !dbg !22
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %43 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx71 = shl nsw i64 %18, 8, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 %.idx71, !dbg !12
  %45 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %42, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local134 = and i64 %address.local133, 1536, !dbg !12
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !12
  %address.local136 = ptrtoint ptr addrspace(3) %address.local135 to i64, !dbg !12
  %address.local137 = trunc i64 %address.local136 to i32, !dbg !12
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !12
  %47 = zext i32 %46 to i64, !dbg !12
  %48 = inttoptr i64 %47 to ptr addrspace(3), !dbg !12
  %address.local138 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local139 = and i64 %address.local138, 16256, !dbg !12
  %address.local140 = lshr i64 %address.local, 4, !dbg !12
  %address.local141 = and i64 %address.local140, 8, !dbg !12
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !12
  %address.local143 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local142, !dbg !12
  %49 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %50 = ptrtoint ptr addrspace(1) %address.local143 to i64, !dbg !12
  %51 = sub i64 %50, %49, !dbg !12
  %dma.buffer.offset623 = trunc i64 %51 to i32, !dbg !12
  %52 = trunc i64 %49 to i32, !dbg !12
  %53 = insertelement <4 x i32> poison, i32 %52, i64 0, !dbg !12
  %54 = lshr i64 %49, 32, !dbg !12
  %55 = trunc i64 %54 to i32, !dbg !12
  %56 = insertelement <4 x i32> %53, i32 %55, i64 1, !dbg !12
  %57 = insertelement <4 x i32> %56, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc624 = insertelement <4 x i32> %57, i32 131072, i64 3, !dbg !12
  %58 = ptrtoint ptr addrspace(3) %48 to i32, !dbg !12
  %dma.buffer.lds625 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !12
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !12
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !12
  %address.local146 = and i64 %address.local145, 3584, !dbg !12
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !12
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !12
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !12
  %60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !12
  %61 = zext i32 %60 to i64, !dbg !12
  %62 = inttoptr i64 %61 to ptr addrspace(3), !dbg !12
  %address.local150 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local151 = and i64 %address.local150, 16256, !dbg !12
  %address.local152 = lshr i64 %address.local144, 4, !dbg !12
  %address.local153 = and i64 %address.local152, 24, !dbg !12
  %address.local154 = or disjoint i64 %address.local151, %address.local153, !dbg !12
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local154, !dbg !12
  %63 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %64 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !12
  %65 = sub i64 %64, %63, !dbg !12
  %dma.buffer.offset620 = trunc i64 %65 to i32, !dbg !12
  %66 = trunc i64 %63 to i32, !dbg !12
  %67 = insertelement <4 x i32> poison, i32 %66, i64 0, !dbg !12
  %68 = lshr i64 %63, 32, !dbg !12
  %69 = trunc i64 %68 to i32, !dbg !12
  %70 = insertelement <4 x i32> %67, i32 %69, i64 1, !dbg !12
  %71 = insertelement <4 x i32> %70, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc621 = insertelement <4 x i32> %71, i32 131072, i64 3, !dbg !12
  %72 = ptrtoint ptr addrspace(3) %62 to i32, !dbg !12
  %dma.buffer.lds622 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72), !dbg !12
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc621) #10, !dbg !12
  %74 = shl nuw nsw i64 %38, 10, !dbg !17
  %75 = add nuw i64 %26, %74, !dbg !17
  %.lobit72 = ashr i64 %75, 63, !dbg !17
  %76 = xor i64 %75, %.lobit72, !dbg !17
  %77 = sdiv i64 %76, 4096, !dbg !17
  %.lobit73 = ashr i64 %75, 63, !dbg !17
  %78 = xor i64 %77, %.lobit73, !dbg !17
  %79 = add nsw i64 %78, %8, !dbg !17
  %80 = add nuw nsw i64 %42, %34, !dbg !17
  %81 = and i64 %80, 127, !dbg !17
  %82 = shl i64 %79, 19, !dbg !17
  %83 = and i64 %82, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %84 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %83, !dbg !17
  %85 = getelementptr [2 x i8], ptr addrspace(1) %84, i64 %81, !dbg !17
  %.idx74 = shl i64 %36, 8, !dbg !17
  %86 = getelementptr i8, ptr addrspace(1) %85, i64 %.idx74, !dbg !17
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local134, !dbg !17
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !17
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !17
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !17
  %88 = zext i32 %87 to i64, !dbg !17
  %89 = inttoptr i64 %88 to ptr addrspace(3), !dbg !17
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %address.local142, !dbg !17
  %90 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %91 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !17
  %92 = sub i64 %91, %90, !dbg !17
  %dma.buffer.offset617 = trunc i64 %92 to i32, !dbg !17
  %93 = trunc i64 %90 to i32, !dbg !17
  %94 = insertelement <4 x i32> poison, i32 %93, i64 0, !dbg !17
  %95 = lshr i64 %90, 32, !dbg !17
  %96 = trunc i64 %95 to i32, !dbg !17
  %97 = insertelement <4 x i32> %94, i32 %96, i64 1, !dbg !17
  %98 = insertelement <4 x i32> %97, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc618 = insertelement <4 x i32> %98, i32 131072, i64 3, !dbg !17
  %99 = ptrtoint ptr addrspace(3) %89 to i32, !dbg !17
  %dma.buffer.lds619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %99), !dbg !17
  %100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !17
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local146, !dbg !17
  %address.local161 = ptrtoint ptr addrspace(3) %address.local160 to i64, !dbg !17
  %address.local162 = trunc i64 %address.local161 to i32, !dbg !17
  %101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local162), !dbg !17
  %102 = zext i32 %101 to i64, !dbg !17
  %103 = inttoptr i64 %102 to ptr addrspace(3), !dbg !17
  %address.local163 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %address.local154, !dbg !17
  %104 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %105 = ptrtoint ptr addrspace(1) %address.local163 to i64, !dbg !17
  %106 = sub i64 %105, %104, !dbg !17
  %dma.buffer.offset = trunc i64 %106 to i32, !dbg !17
  %107 = trunc i64 %104 to i32, !dbg !17
  %108 = insertelement <4 x i32> poison, i32 %107, i64 0, !dbg !17
  %109 = lshr i64 %104, 32, !dbg !17
  %110 = trunc i64 %109 to i32, !dbg !17
  %111 = insertelement <4 x i32> %108, i32 %110, i64 1, !dbg !17
  %112 = insertelement <4 x i32> %111, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %112, i32 131072, i64 3, !dbg !17
  %113 = ptrtoint ptr addrspace(3) %103 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113), !dbg !17
  %114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %address.local164 = lshr i32 %address.thread, 2, !dbg !13
  %address.local165 = and i32 %address.local164, 4, !dbg !13
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local167 = and i32 %address.local166, 384, !dbg !13
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !13
  %address.local169 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local170 = and i32 %address.local169, 120, !dbg !13
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !13
  %address.local172 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local173 = and i32 %address.local172, 1024, !dbg !13
  %address.local174 = or disjoint i32 %address.local171, %address.local173, !dbg !13
  %address.local175 = zext nneg i32 %address.local174 to i64, !dbg !13
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %115 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !13
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 4096, !dbg !13
  %116 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !13
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local179, i64 1024, !dbg !13
  %117 = load <4 x half>, ptr addrspace(3) %address.local180, align 2, !dbg !13
  %118 = fmul <4 x half> %115, splat (half 0xH3015), !dbg !13
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !13
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 5120, !dbg !13
  %119 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !13
  %120 = fmul <4 x half> %116, splat (half 0xH3015), !dbg !13
  %121 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !13
  %122 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local183 = lshr i32 %address.thread, 2, !dbg !8
  %address.local184 = and i32 %address.local183, 4, !dbg !8
  %address.local185 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local186 = and i32 %address.local185, 24, !dbg !8
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !8
  %address.local188 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local189 = and i32 %address.local188, 1024, !dbg !8
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !8
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local192 = and i32 %address.local191, 384, !dbg !8
  %address.local193 = or disjoint i32 %address.local190, %address.local192, !dbg !8
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !8
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !8
  %123 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !8
  %124 = extractvalue [2 x <32 x float>] %39, 0, !dbg !30
  %125 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %126 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %123, <4 x float> %125) #11, !dbg !8
  %address.local196 = lshr i32 %address.thread, 2, !dbg !8
  %address.local197 = and i32 %address.local196, 4, !dbg !8
  %address.local198 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local199 = and i32 %address.local198, 24, !dbg !8
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !8
  %address.local201 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local202 = and i32 %address.local201, 1024, !dbg !8
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !8
  %address.local204 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local205 = and i32 %address.local204, 384, !dbg !8
  %address.local206 = or disjoint i32 %address.local203, %address.local205, !dbg !8
  %address.local207 = zext nneg i32 %address.local206 to i64, !dbg !8
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local208, i64 4096, !dbg !8
  %127 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !8
  %128 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %129 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %127, <4 x float> %126) #11, !dbg !8
  %130 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %131 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %132 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !30
  %133 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !30
  %134 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !30
  %135 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !30
  %136 = extractvalue [2 x <32 x float>] %39, 1, !dbg !30
  %137 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %138 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %139 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !30
  %140 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !30
  %141 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !30
  %142 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !30
  %143 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !30
  %144 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !30
  %address.local210 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local212 = lshr i64 %address.local, 2, !dbg !8
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !8
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !8
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local216 = and i64 %address.local215, 24, !dbg !8
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !8
  %address.local218 = lshr i64 %address.local, 2, !dbg !13
  %address.local219 = and i64 %address.local218, 12, !dbg !13
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !8
  %address.local221 = and i64 %address.local, 63, !dbg !13
  %address.local222 = lshr i64 %address.local221, 5, !dbg !13
  %address.local223 = mul nuw nsw i64 %address.local222, 1016, !dbg !13
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !8
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !8
  %145 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !8
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local219, !dbg !8
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !8
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !8
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !8
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !8
  %146 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %147 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %145, <4 x float> %128) #11, !dbg !8
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local213, !dbg !8
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local216, !dbg !8
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local219, !dbg !8
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local223, !dbg !8
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 128, !dbg !8
  %148 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !8
  %149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %146, <4 x float> %147) #11, !dbg !8
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local219, !dbg !8
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local223, !dbg !8
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local213, !dbg !8
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local216, !dbg !8
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 4224, !dbg !8
  %150 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !8
  %151 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %148, <4 x float> %130) #11, !dbg !8
  %152 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %150, <4 x float> %151) #11, !dbg !8
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local213, !dbg !8
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local216, !dbg !8
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local219, !dbg !8
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local223, !dbg !8
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local248, i64 192, !dbg !8
  %153 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !8
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local210, !dbg !8
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local219, !dbg !8
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local223, !dbg !8
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) %address.local252, i64 %address.local213, !dbg !8
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local253, i64 %address.local216, !dbg !8
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 4288, !dbg !8
  %154 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !8
  %155 = shufflevector <4 x float> %152, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %153, <4 x float> %131) #11, !dbg !8
  %157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %154, <4 x float> %156) #11, !dbg !8
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local194, !dbg !8
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local256, i64 1024, !dbg !8
  %158 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !8
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 5120, !dbg !8
  %159 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !8
  %160 = shufflevector <4 x float> %157, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %161 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %158, <4 x float> %132) #11, !dbg !8
  %162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %159, <4 x float> %161) #11, !dbg !8
  %address.local260 = add nuw nsw i64 %address.local210, 512, !dbg !8
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !8
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !8
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !8
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !8
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !8
  %163 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !8
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !8
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local219, !dbg !8
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !8
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !8
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !8
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !8
  %164 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !8
  %165 = shufflevector <4 x float> %162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %163, <4 x float> %133) #11, !dbg !8
  %167 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %164, <4 x float> %166) #11, !dbg !8
  %address.local273 = add nuw nsw i64 %address.local210, 512, !dbg !8
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !8
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local213, !dbg !8
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local216, !dbg !8
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local219, !dbg !8
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local223, !dbg !8
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local278, i64 128, !dbg !8
  %168 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !8
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local273, !dbg !8
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local219, !dbg !8
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local223, !dbg !8
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local213, !dbg !8
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local216, !dbg !8
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 4224, !dbg !8
  %169 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !8
  %170 = shufflevector <4 x float> %167, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %171 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %168, <4 x float> %134) #11, !dbg !8
  %172 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %169, <4 x float> %171) #11, !dbg !8
  %address.local286 = add nuw nsw i64 %address.local210, 512, !dbg !8
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local286, !dbg !8
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local213, !dbg !8
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local216, !dbg !8
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local219, !dbg !8
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local223, !dbg !8
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 192, !dbg !8
  %173 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !8
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local286, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local219, !dbg !8
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %address.local294, i64 %address.local223, !dbg !8
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local213, !dbg !8
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local216, !dbg !8
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local297, i64 4288, !dbg !8
  %174 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !8
  %175 = shufflevector <4 x float> %172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %176 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %173, <4 x float> %135) #11, !dbg !8
  %177 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %174, <4 x float> %176) #11, !dbg !8
  %178 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !8
  %179 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !8
  %180 = shufflevector <4 x float> %177, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %181 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %178, <4 x float> %137) #11, !dbg !8
  %182 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %179, <4 x float> %181) #11, !dbg !8
  %183 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !8
  %184 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %185 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %183, <4 x float> %138) #11, !dbg !8
  %186 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %184, <4 x float> %185) #11, !dbg !8
  %187 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !8
  %188 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !8
  %189 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %187, <4 x float> %139) #11, !dbg !8
  %190 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %188, <4 x float> %189) #11, !dbg !8
  %191 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !8
  %192 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !8
  %193 = shufflevector <4 x float> %190, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %194 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %191, <4 x float> %140) #11, !dbg !8
  %195 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %192, <4 x float> %194) #11, !dbg !8
  %196 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !8
  %197 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !8
  %198 = shufflevector <4 x float> %195, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %199 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %196, <4 x float> %141) #11, !dbg !8
  %200 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %197, <4 x float> %199) #11, !dbg !8
  %201 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !8
  %202 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !8
  %203 = shufflevector <4 x float> %200, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %204 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %201, <4 x float> %142) #11, !dbg !8
  %205 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %202, <4 x float> %204) #11, !dbg !8
  %206 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !8
  %207 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !8
  %208 = shufflevector <4 x float> %205, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %209 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %206, <4 x float> %143) #11, !dbg !8
  %210 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %207, <4 x float> %209) #11, !dbg !8
  %211 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !8
  %212 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !8
  %213 = shufflevector <4 x float> %210, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %211, <4 x float> %144) #11, !dbg !8
  %215 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %212, <4 x float> %214) #11, !dbg !8
  %216 = shufflevector <4 x float> %215, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !30
  %217 = shufflevector <4 x float> %129, <4 x float> %149, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %218 = shufflevector <32 x float> %217, <32 x float> %155, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %219 = shufflevector <32 x float> %218, <32 x float> %160, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %220 = shufflevector <32 x float> %219, <32 x float> %165, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %221 = shufflevector <32 x float> %220, <32 x float> %170, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %222 = shufflevector <32 x float> %221, <32 x float> %175, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %223 = shufflevector <32 x float> %222, <32 x float> %180, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %224 = insertvalue [2 x <32 x float>] poison, <32 x float> %223, 0, !dbg !22
  %225 = shufflevector <4 x float> %182, <4 x float> %186, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %226 = shufflevector <32 x float> %225, <32 x float> %193, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %227 = shufflevector <32 x float> %226, <32 x float> %198, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %228 = shufflevector <32 x float> %227, <32 x float> %203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %229 = shufflevector <32 x float> %228, <32 x float> %208, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %230 = shufflevector <32 x float> %229, <32 x float> %213, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %231 = shufflevector <32 x float> %230, <32 x float> %216, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %232 = insertvalue [2 x <32 x float>] %224, <32 x float> %231, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %233 = add nuw nsw i64 %38, 1, !dbg !22
  br label %37, !dbg !22

234:                                              ; preds = %37
  %235 = add nsw i64 %34, %8, !dbg !18
  %236 = shl i64 %235, 19, !dbg !18
  %237 = and i64 %236, 16252928, !dbg !18
  %238 = shl nsw i64 %36, 7, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %239 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %237, !dbg !18
  %240 = getelementptr [2 x i8], ptr addrspace(1) %239, i64 %238, !dbg !18
  %address.thread299 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !33, !frisk.register_lifetimes !1
  %address.local300 = zext nneg i32 %address.thread299 to i64, !dbg !13
  %address.local301 = shl nuw nsw i64 %address.local300, 3, !dbg !12
  %address.local302 = and i64 %address.local301, 1536, !dbg !12
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !12
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !12
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !12
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !18
  %242 = zext i32 %241 to i64, !dbg !18
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !18
  %address.local306 = shl nuw nsw i64 %address.local300, 5, !dbg !18
  %address.local307 = and i64 %address.local306, 8064, !dbg !18
  %address.local308 = shl nuw nsw i64 %address.local300, 3, !dbg !18
  %address.local309 = and i64 %address.local308, 24, !dbg !18
  %address.local310 = or disjoint i64 %address.local307, %address.local309, !dbg !18
  %address.local311 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local310, !dbg !18
  %244 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %245 = ptrtoint ptr addrspace(1) %address.local311 to i64, !dbg !18
  %246 = sub i64 %245, %244, !dbg !18
  %dma.buffer.offset647 = trunc i64 %246 to i32, !dbg !18
  %247 = trunc i64 %244 to i32, !dbg !18
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !18
  %249 = lshr i64 %244, 32, !dbg !18
  %250 = trunc i64 %249 to i32, !dbg !18
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !18
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc648 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !18
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !18
  %dma.buffer.lds649 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !18
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds649, i32 %dma.buffer.offset647, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !18
  %address.local312 = or disjoint i64 %address.local300, 256, !dbg !12
  %address.local313 = shl nuw nsw i64 %address.local312, 3, !dbg !12
  %address.local314 = and i64 %address.local313, 3584, !dbg !12
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !12
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !12
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !12
  %255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !18
  %256 = zext i32 %255 to i64, !dbg !18
  %257 = inttoptr i64 %256 to ptr addrspace(3), !dbg !18
  %address.local318 = shl nuw nsw i64 %address.local312, 5, !dbg !18
  %address.local319 = and i64 %address.local318, 16256, !dbg !18
  %address.local320 = shl nuw nsw i64 %address.local300, 3, !dbg !18
  %address.local321 = and i64 %address.local320, 24, !dbg !18
  %address.local322 = or disjoint i64 %address.local319, %address.local321, !dbg !18
  %address.local323 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local322, !dbg !18
  %258 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %259 = ptrtoint ptr addrspace(1) %address.local323 to i64, !dbg !18
  %260 = sub i64 %259, %258, !dbg !18
  %dma.buffer.offset644 = trunc i64 %260 to i32, !dbg !18
  %261 = trunc i64 %258 to i32, !dbg !18
  %262 = insertelement <4 x i32> poison, i32 %261, i64 0, !dbg !18
  %263 = lshr i64 %258, 32, !dbg !18
  %264 = trunc i64 %263 to i32, !dbg !18
  %265 = insertelement <4 x i32> %262, i32 %264, i64 1, !dbg !18
  %266 = insertelement <4 x i32> %265, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc645 = insertelement <4 x i32> %266, i32 131072, i64 3, !dbg !18
  %267 = ptrtoint ptr addrspace(3) %257 to i32, !dbg !18
  %dma.buffer.lds646 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %267), !dbg !18
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds646, i32 %dma.buffer.offset644, <4 x i32> %dma.buffer.rsrc645) #10, !dbg !18
  %269 = or disjoint i64 %33, 127, !dbg !14
  %270 = icmp uge i64 %9, %269, !dbg !14
  %271 = and i1 %24, %270, !dbg !14
  br i1 %271, label %272, label %467, !dbg !14

272:                                              ; preds = %234
  %273 = extractvalue [2 x <32 x float>] %39, 0, !dbg !14
  %274 = extractelement <32 x float> %273, i64 0, !dbg !14
  %275 = fadd float %274, 0x0000000000000000, !dbg !14
  %276 = extractelement <32 x float> %273, i64 1, !dbg !14
  %277 = fadd float %276, 0x0000000000000000, !dbg !14
  %278 = extractelement <32 x float> %273, i64 2, !dbg !14
  %279 = fadd float %278, 0x0000000000000000, !dbg !14
  %280 = extractelement <32 x float> %273, i64 3, !dbg !14
  %281 = fadd float %280, 0x0000000000000000, !dbg !14
  %282 = extractelement <32 x float> %273, i64 4, !dbg !14
  %283 = fadd float %282, 0x0000000000000000, !dbg !14
  %284 = extractelement <32 x float> %273, i64 5, !dbg !14
  %285 = fadd float %284, 0x0000000000000000, !dbg !14
  %286 = extractelement <32 x float> %273, i64 6, !dbg !14
  %287 = fadd float %286, 0x0000000000000000, !dbg !14
  %288 = extractelement <32 x float> %273, i64 7, !dbg !14
  %289 = fadd float %288, 0x0000000000000000, !dbg !14
  %290 = extractelement <32 x float> %273, i64 8, !dbg !14
  %291 = fadd float %290, 0x0000000000000000, !dbg !14
  %292 = extractelement <32 x float> %273, i64 9, !dbg !14
  %293 = fadd float %292, 0x0000000000000000, !dbg !14
  %294 = extractelement <32 x float> %273, i64 10, !dbg !14
  %295 = fadd float %294, 0x0000000000000000, !dbg !14
  %296 = extractelement <32 x float> %273, i64 11, !dbg !14
  %297 = fadd float %296, 0x0000000000000000, !dbg !14
  %298 = extractelement <32 x float> %273, i64 12, !dbg !14
  %299 = fadd float %298, 0x0000000000000000, !dbg !14
  %300 = extractelement <32 x float> %273, i64 13, !dbg !14
  %301 = fadd float %300, 0x0000000000000000, !dbg !14
  %302 = extractelement <32 x float> %273, i64 14, !dbg !14
  %303 = fadd float %302, 0x0000000000000000, !dbg !14
  %304 = extractelement <32 x float> %273, i64 15, !dbg !14
  %305 = fadd float %304, 0x0000000000000000, !dbg !14
  %306 = extractelement <32 x float> %273, i64 16, !dbg !14
  %307 = fadd float %306, 0x0000000000000000, !dbg !14
  %308 = extractelement <32 x float> %273, i64 17, !dbg !14
  %309 = fadd float %308, 0x0000000000000000, !dbg !14
  %310 = extractelement <32 x float> %273, i64 18, !dbg !14
  %311 = fadd float %310, 0x0000000000000000, !dbg !14
  %312 = extractelement <32 x float> %273, i64 19, !dbg !14
  %313 = fadd float %312, 0x0000000000000000, !dbg !14
  %314 = extractelement <32 x float> %273, i64 20, !dbg !14
  %315 = fadd float %314, 0x0000000000000000, !dbg !14
  %316 = extractelement <32 x float> %273, i64 21, !dbg !14
  %317 = fadd float %316, 0x0000000000000000, !dbg !14
  %318 = extractelement <32 x float> %273, i64 22, !dbg !14
  %319 = fadd float %318, 0x0000000000000000, !dbg !14
  %320 = extractelement <32 x float> %273, i64 23, !dbg !14
  %321 = fadd float %320, 0x0000000000000000, !dbg !14
  %322 = extractelement <32 x float> %273, i64 24, !dbg !14
  %323 = fadd float %322, 0x0000000000000000, !dbg !14
  %324 = extractelement <32 x float> %273, i64 25, !dbg !14
  %325 = fadd float %324, 0x0000000000000000, !dbg !14
  %326 = extractelement <32 x float> %273, i64 26, !dbg !14
  %327 = fadd float %326, 0x0000000000000000, !dbg !14
  %328 = extractelement <32 x float> %273, i64 27, !dbg !14
  %329 = fadd float %328, 0x0000000000000000, !dbg !14
  %330 = extractelement <32 x float> %273, i64 28, !dbg !14
  %331 = fadd float %330, 0x0000000000000000, !dbg !14
  %332 = extractelement <32 x float> %273, i64 29, !dbg !14
  %333 = fadd float %332, 0x0000000000000000, !dbg !14
  %334 = extractelement <32 x float> %273, i64 30, !dbg !14
  %335 = fadd float %334, 0x0000000000000000, !dbg !14
  %336 = extractelement <32 x float> %273, i64 31, !dbg !14
  %337 = fadd float %336, 0x0000000000000000, !dbg !14
  %338 = extractvalue [2 x <32 x float>] %39, 1, !dbg !14
  %339 = extractelement <32 x float> %338, i64 0, !dbg !14
  %340 = fadd float %339, 0x0000000000000000, !dbg !14
  %341 = extractelement <32 x float> %338, i64 1, !dbg !14
  %342 = fadd float %341, 0x0000000000000000, !dbg !14
  %343 = extractelement <32 x float> %338, i64 2, !dbg !14
  %344 = fadd float %343, 0x0000000000000000, !dbg !14
  %345 = extractelement <32 x float> %338, i64 3, !dbg !14
  %346 = fadd float %345, 0x0000000000000000, !dbg !14
  %347 = extractelement <32 x float> %338, i64 4, !dbg !14
  %348 = fadd float %347, 0x0000000000000000, !dbg !14
  %349 = extractelement <32 x float> %338, i64 5, !dbg !14
  %350 = fadd float %349, 0x0000000000000000, !dbg !14
  %351 = extractelement <32 x float> %338, i64 6, !dbg !14
  %352 = fadd float %351, 0x0000000000000000, !dbg !14
  %353 = extractelement <32 x float> %338, i64 7, !dbg !14
  %354 = fadd float %353, 0x0000000000000000, !dbg !14
  %355 = extractelement <32 x float> %338, i64 8, !dbg !14
  %356 = fadd float %355, 0x0000000000000000, !dbg !14
  %357 = extractelement <32 x float> %338, i64 9, !dbg !14
  %358 = fadd float %357, 0x0000000000000000, !dbg !14
  %359 = extractelement <32 x float> %338, i64 10, !dbg !14
  %360 = fadd float %359, 0x0000000000000000, !dbg !14
  %361 = extractelement <32 x float> %338, i64 11, !dbg !14
  %362 = fadd float %361, 0x0000000000000000, !dbg !14
  %363 = extractelement <32 x float> %338, i64 12, !dbg !14
  %364 = fadd float %363, 0x0000000000000000, !dbg !14
  %365 = extractelement <32 x float> %338, i64 13, !dbg !14
  %366 = fadd float %365, 0x0000000000000000, !dbg !14
  %367 = extractelement <32 x float> %338, i64 14, !dbg !14
  %368 = fadd float %367, 0x0000000000000000, !dbg !14
  %369 = extractelement <32 x float> %338, i64 15, !dbg !14
  %370 = fadd float %369, 0x0000000000000000, !dbg !14
  %371 = extractelement <32 x float> %338, i64 16, !dbg !14
  %372 = fadd float %371, 0x0000000000000000, !dbg !14
  %373 = extractelement <32 x float> %338, i64 17, !dbg !14
  %374 = fadd float %373, 0x0000000000000000, !dbg !14
  %375 = extractelement <32 x float> %338, i64 18, !dbg !14
  %376 = fadd float %375, 0x0000000000000000, !dbg !14
  %377 = extractelement <32 x float> %338, i64 19, !dbg !14
  %378 = fadd float %377, 0x0000000000000000, !dbg !14
  %379 = extractelement <32 x float> %338, i64 20, !dbg !14
  %380 = fadd float %379, 0x0000000000000000, !dbg !14
  %381 = extractelement <32 x float> %338, i64 21, !dbg !14
  %382 = fadd float %381, 0x0000000000000000, !dbg !14
  %383 = extractelement <32 x float> %338, i64 22, !dbg !14
  %384 = fadd float %383, 0x0000000000000000, !dbg !14
  %385 = extractelement <32 x float> %338, i64 23, !dbg !14
  %386 = fadd float %385, 0x0000000000000000, !dbg !14
  %387 = extractelement <32 x float> %338, i64 24, !dbg !14
  %388 = fadd float %387, 0x0000000000000000, !dbg !14
  %389 = extractelement <32 x float> %338, i64 25, !dbg !14
  %390 = fadd float %389, 0x0000000000000000, !dbg !14
  %391 = extractelement <32 x float> %338, i64 26, !dbg !14
  %392 = fadd float %391, 0x0000000000000000, !dbg !14
  %393 = extractelement <32 x float> %338, i64 27, !dbg !14
  %394 = fadd float %393, 0x0000000000000000, !dbg !14
  %395 = extractelement <32 x float> %338, i64 28, !dbg !14
  %396 = fadd float %395, 0x0000000000000000, !dbg !14
  %397 = extractelement <32 x float> %338, i64 29, !dbg !14
  %398 = fadd float %397, 0x0000000000000000, !dbg !14
  %399 = extractelement <32 x float> %338, i64 30, !dbg !14
  %400 = fadd float %399, 0x0000000000000000, !dbg !14
  %401 = extractelement <32 x float> %338, i64 31, !dbg !14
  %402 = fadd float %401, 0x0000000000000000, !dbg !14
  %403 = insertelement <32 x float> poison, float %275, i64 0, !dbg !14
  %404 = insertelement <32 x float> %403, float %277, i64 1, !dbg !14
  %405 = insertelement <32 x float> %404, float %279, i64 2, !dbg !14
  %406 = insertelement <32 x float> %405, float %281, i64 3, !dbg !14
  %407 = insertelement <32 x float> %406, float %283, i64 4, !dbg !14
  %408 = insertelement <32 x float> %407, float %285, i64 5, !dbg !14
  %409 = insertelement <32 x float> %408, float %287, i64 6, !dbg !14
  %410 = insertelement <32 x float> %409, float %289, i64 7, !dbg !14
  %411 = insertelement <32 x float> %410, float %291, i64 8, !dbg !14
  %412 = insertelement <32 x float> %411, float %293, i64 9, !dbg !14
  %413 = insertelement <32 x float> %412, float %295, i64 10, !dbg !14
  %414 = insertelement <32 x float> %413, float %297, i64 11, !dbg !14
  %415 = insertelement <32 x float> %414, float %299, i64 12, !dbg !14
  %416 = insertelement <32 x float> %415, float %301, i64 13, !dbg !14
  %417 = insertelement <32 x float> %416, float %303, i64 14, !dbg !14
  %418 = insertelement <32 x float> %417, float %305, i64 15, !dbg !14
  %419 = insertelement <32 x float> %418, float %307, i64 16, !dbg !14
  %420 = insertelement <32 x float> %419, float %309, i64 17, !dbg !14
  %421 = insertelement <32 x float> %420, float %311, i64 18, !dbg !14
  %422 = insertelement <32 x float> %421, float %313, i64 19, !dbg !14
  %423 = insertelement <32 x float> %422, float %315, i64 20, !dbg !14
  %424 = insertelement <32 x float> %423, float %317, i64 21, !dbg !14
  %425 = insertelement <32 x float> %424, float %319, i64 22, !dbg !14
  %426 = insertelement <32 x float> %425, float %321, i64 23, !dbg !14
  %427 = insertelement <32 x float> %426, float %323, i64 24, !dbg !14
  %428 = insertelement <32 x float> %427, float %325, i64 25, !dbg !14
  %429 = insertelement <32 x float> %428, float %327, i64 26, !dbg !14
  %430 = insertelement <32 x float> %429, float %329, i64 27, !dbg !14
  %431 = insertelement <32 x float> %430, float %331, i64 28, !dbg !14
  %432 = insertelement <32 x float> %431, float %333, i64 29, !dbg !14
  %433 = insertelement <32 x float> %432, float %335, i64 30, !dbg !14
  %434 = insertelement <32 x float> %433, float %337, i64 31, !dbg !14
  %435 = insertelement <32 x float> poison, float %340, i64 0, !dbg !14
  %436 = insertelement <32 x float> %435, float %342, i64 1, !dbg !14
  %437 = insertelement <32 x float> %436, float %344, i64 2, !dbg !14
  %438 = insertelement <32 x float> %437, float %346, i64 3, !dbg !14
  %439 = insertelement <32 x float> %438, float %348, i64 4, !dbg !14
  %440 = insertelement <32 x float> %439, float %350, i64 5, !dbg !14
  %441 = insertelement <32 x float> %440, float %352, i64 6, !dbg !14
  %442 = insertelement <32 x float> %441, float %354, i64 7, !dbg !14
  %443 = insertelement <32 x float> %442, float %356, i64 8, !dbg !14
  %444 = insertelement <32 x float> %443, float %358, i64 9, !dbg !14
  %445 = insertelement <32 x float> %444, float %360, i64 10, !dbg !14
  %446 = insertelement <32 x float> %445, float %362, i64 11, !dbg !14
  %447 = insertelement <32 x float> %446, float %364, i64 12, !dbg !14
  %448 = insertelement <32 x float> %447, float %366, i64 13, !dbg !14
  %449 = insertelement <32 x float> %448, float %368, i64 14, !dbg !14
  %450 = insertelement <32 x float> %449, float %370, i64 15, !dbg !14
  %451 = insertelement <32 x float> %450, float %372, i64 16, !dbg !14
  %452 = insertelement <32 x float> %451, float %374, i64 17, !dbg !14
  %453 = insertelement <32 x float> %452, float %376, i64 18, !dbg !14
  %454 = insertelement <32 x float> %453, float %378, i64 19, !dbg !14
  %455 = insertelement <32 x float> %454, float %380, i64 20, !dbg !14
  %456 = insertelement <32 x float> %455, float %382, i64 21, !dbg !14
  %457 = insertelement <32 x float> %456, float %384, i64 22, !dbg !14
  %458 = insertelement <32 x float> %457, float %386, i64 23, !dbg !14
  %459 = insertelement <32 x float> %458, float %388, i64 24, !dbg !14
  %460 = insertelement <32 x float> %459, float %390, i64 25, !dbg !14
  %461 = insertelement <32 x float> %460, float %392, i64 26, !dbg !14
  %462 = insertelement <32 x float> %461, float %394, i64 27, !dbg !14
  %463 = insertelement <32 x float> %462, float %396, i64 28, !dbg !14
  %464 = insertelement <32 x float> %463, float %398, i64 29, !dbg !14
  %465 = insertelement <32 x float> %464, float %400, i64 30, !dbg !14
  %466 = insertelement <32 x float> %465, float %402, i64 31, !dbg !14
  br label %766, !dbg !14

467:                                              ; preds = %234
  %468 = add nsw i64 %9, %20, !dbg !14
  %469 = shl nuw nsw i64 %21, 4, !dbg !14
  %470 = add nsw i64 %468, %469, !dbg !14
  %471 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %471 to i64, !dbg !14
  %472 = sub nsw i64 %470, %index.compact, !dbg !14
  %473 = and i64 %20, 48, !dbg !14
  %474 = or disjoint i64 %33, %473, !dbg !14
  %475 = add nsw i64 %472, 1, !dbg !23
  %.not4 = icmp ugt i64 %475, %474, !dbg !24
  %476 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %477 = extractvalue [2 x <32 x float>] %39, 0, !dbg !14
  %478 = extractelement <32 x float> %477, i64 0, !dbg !14
  %479 = fadd float %478, %476, !dbg !14
  %480 = or disjoint i64 %474, 1, !dbg !14
  %.not5 = icmp ugt i64 %475, %480, !dbg !24
  %481 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %482 = extractelement <32 x float> %477, i64 1, !dbg !14
  %483 = fadd float %482, %481, !dbg !14
  %484 = or disjoint i64 %474, 2, !dbg !14
  %.not6 = icmp ugt i64 %475, %484, !dbg !24
  %485 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %486 = extractelement <32 x float> %477, i64 2, !dbg !14
  %487 = fadd float %486, %485, !dbg !14
  %488 = or disjoint i64 %474, 3, !dbg !14
  %.not7 = icmp ugt i64 %475, %488, !dbg !24
  %489 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %490 = extractelement <32 x float> %477, i64 3, !dbg !14
  %491 = fadd float %490, %489, !dbg !14
  %492 = or disjoint i64 %474, 4, !dbg !14
  %.not8 = icmp ugt i64 %475, %492, !dbg !24
  %493 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %494 = extractelement <32 x float> %477, i64 4, !dbg !14
  %495 = fadd float %494, %493, !dbg !14
  %496 = or disjoint i64 %474, 5, !dbg !14
  %.not9 = icmp ugt i64 %475, %496, !dbg !24
  %497 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %498 = extractelement <32 x float> %477, i64 5, !dbg !14
  %499 = fadd float %498, %497, !dbg !14
  %500 = or disjoint i64 %474, 6, !dbg !14
  %.not10 = icmp ugt i64 %475, %500, !dbg !24
  %501 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %502 = extractelement <32 x float> %477, i64 6, !dbg !14
  %503 = fadd float %502, %501, !dbg !14
  %504 = or disjoint i64 %474, 7, !dbg !14
  %.not11 = icmp ugt i64 %475, %504, !dbg !24
  %505 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %506 = extractelement <32 x float> %477, i64 7, !dbg !14
  %507 = fadd float %506, %505, !dbg !14
  %508 = or disjoint i64 %474, 8, !dbg !14
  %.not12 = icmp ugt i64 %475, %508, !dbg !24
  %509 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %510 = extractelement <32 x float> %477, i64 8, !dbg !14
  %511 = fadd float %510, %509, !dbg !14
  %512 = or disjoint i64 %474, 9, !dbg !14
  %.not13 = icmp ugt i64 %475, %512, !dbg !24
  %513 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %514 = extractelement <32 x float> %477, i64 9, !dbg !14
  %515 = fadd float %514, %513, !dbg !14
  %516 = or disjoint i64 %474, 10, !dbg !14
  %.not14 = icmp ugt i64 %475, %516, !dbg !24
  %517 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %518 = extractelement <32 x float> %477, i64 10, !dbg !14
  %519 = fadd float %518, %517, !dbg !14
  %520 = or disjoint i64 %474, 11, !dbg !14
  %.not15 = icmp ugt i64 %475, %520, !dbg !24
  %521 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %522 = extractelement <32 x float> %477, i64 11, !dbg !14
  %523 = fadd float %522, %521, !dbg !14
  %524 = or disjoint i64 %474, 12, !dbg !14
  %.not16 = icmp ugt i64 %475, %524, !dbg !24
  %525 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %526 = extractelement <32 x float> %477, i64 12, !dbg !14
  %527 = fadd float %526, %525, !dbg !14
  %528 = or disjoint i64 %474, 13, !dbg !14
  %.not17 = icmp ugt i64 %475, %528, !dbg !24
  %529 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %530 = extractelement <32 x float> %477, i64 13, !dbg !14
  %531 = fadd float %530, %529, !dbg !14
  %532 = or disjoint i64 %474, 14, !dbg !14
  %.not18 = icmp ugt i64 %475, %532, !dbg !24
  %533 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %534 = extractelement <32 x float> %477, i64 14, !dbg !14
  %535 = fadd float %534, %533, !dbg !14
  %536 = or disjoint i64 %474, 15, !dbg !14
  %.not19 = icmp ugt i64 %475, %536, !dbg !24
  %537 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %538 = extractelement <32 x float> %477, i64 15, !dbg !14
  %539 = fadd float %538, %537, !dbg !14
  %540 = or disjoint i64 %474, 64, !dbg !14
  %.not20 = icmp ugt i64 %475, %540, !dbg !24
  %541 = select i1 %.not20, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %542 = extractelement <32 x float> %477, i64 16, !dbg !14
  %543 = fadd float %542, %541, !dbg !14
  %544 = or disjoint i64 %474, 65, !dbg !14
  %.not21 = icmp ugt i64 %475, %544, !dbg !24
  %545 = select i1 %.not21, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %546 = extractelement <32 x float> %477, i64 17, !dbg !14
  %547 = fadd float %546, %545, !dbg !14
  %548 = or disjoint i64 %474, 66, !dbg !14
  %.not22 = icmp ugt i64 %475, %548, !dbg !24
  %549 = select i1 %.not22, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %550 = extractelement <32 x float> %477, i64 18, !dbg !14
  %551 = fadd float %550, %549, !dbg !14
  %552 = or disjoint i64 %474, 67, !dbg !14
  %.not23 = icmp ugt i64 %475, %552, !dbg !24
  %553 = select i1 %.not23, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %554 = extractelement <32 x float> %477, i64 19, !dbg !14
  %555 = fadd float %554, %553, !dbg !14
  %556 = or disjoint i64 %474, 68, !dbg !14
  %.not24 = icmp ugt i64 %475, %556, !dbg !24
  %557 = select i1 %.not24, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %558 = extractelement <32 x float> %477, i64 20, !dbg !14
  %559 = fadd float %558, %557, !dbg !14
  %560 = or disjoint i64 %474, 69, !dbg !14
  %.not25 = icmp ugt i64 %475, %560, !dbg !24
  %561 = select i1 %.not25, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %562 = extractelement <32 x float> %477, i64 21, !dbg !14
  %563 = fadd float %562, %561, !dbg !14
  %564 = or disjoint i64 %474, 70, !dbg !14
  %.not26 = icmp ugt i64 %475, %564, !dbg !24
  %565 = select i1 %.not26, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %566 = extractelement <32 x float> %477, i64 22, !dbg !14
  %567 = fadd float %566, %565, !dbg !14
  %568 = or disjoint i64 %474, 71, !dbg !14
  %.not27 = icmp ugt i64 %475, %568, !dbg !24
  %569 = select i1 %.not27, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %570 = extractelement <32 x float> %477, i64 23, !dbg !14
  %571 = fadd float %570, %569, !dbg !14
  %572 = or disjoint i64 %474, 72, !dbg !14
  %.not28 = icmp ugt i64 %475, %572, !dbg !24
  %573 = select i1 %.not28, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %574 = extractelement <32 x float> %477, i64 24, !dbg !14
  %575 = fadd float %574, %573, !dbg !14
  %576 = or disjoint i64 %474, 73, !dbg !14
  %.not29 = icmp ugt i64 %475, %576, !dbg !24
  %577 = select i1 %.not29, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %578 = extractelement <32 x float> %477, i64 25, !dbg !14
  %579 = fadd float %578, %577, !dbg !14
  %580 = or disjoint i64 %474, 74, !dbg !14
  %.not30 = icmp ugt i64 %475, %580, !dbg !24
  %581 = select i1 %.not30, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %582 = extractelement <32 x float> %477, i64 26, !dbg !14
  %583 = fadd float %582, %581, !dbg !14
  %584 = or disjoint i64 %474, 75, !dbg !14
  %.not31 = icmp ugt i64 %475, %584, !dbg !24
  %585 = select i1 %.not31, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %586 = extractelement <32 x float> %477, i64 27, !dbg !14
  %587 = fadd float %586, %585, !dbg !14
  %588 = or disjoint i64 %474, 76, !dbg !14
  %.not32 = icmp ugt i64 %475, %588, !dbg !24
  %589 = select i1 %.not32, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %590 = extractelement <32 x float> %477, i64 28, !dbg !14
  %591 = fadd float %590, %589, !dbg !14
  %592 = or disjoint i64 %474, 77, !dbg !14
  %.not33 = icmp ugt i64 %475, %592, !dbg !24
  %593 = select i1 %.not33, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %594 = extractelement <32 x float> %477, i64 29, !dbg !14
  %595 = fadd float %594, %593, !dbg !14
  %596 = or disjoint i64 %474, 78, !dbg !14
  %.not34 = icmp ugt i64 %475, %596, !dbg !24
  %597 = select i1 %.not34, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %598 = extractelement <32 x float> %477, i64 30, !dbg !14
  %599 = fadd float %598, %597, !dbg !14
  %600 = or disjoint i64 %474, 79, !dbg !14
  %.not35 = icmp ugt i64 %475, %600, !dbg !24
  %601 = select i1 %.not35, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %602 = extractelement <32 x float> %477, i64 31, !dbg !14
  %603 = fadd float %602, %601, !dbg !14
  %604 = add nsw i64 %472, 65, !dbg !23
  %.not36 = icmp ugt i64 %604, %474, !dbg !24
  %605 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %606 = extractvalue [2 x <32 x float>] %39, 1, !dbg !14
  %607 = extractelement <32 x float> %606, i64 0, !dbg !14
  %608 = fadd float %607, %605, !dbg !14
  %.not37 = icmp ugt i64 %604, %480, !dbg !24
  %609 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %610 = extractelement <32 x float> %606, i64 1, !dbg !14
  %611 = fadd float %610, %609, !dbg !14
  %.not38 = icmp ugt i64 %604, %484, !dbg !24
  %612 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %613 = extractelement <32 x float> %606, i64 2, !dbg !14
  %614 = fadd float %613, %612, !dbg !14
  %.not39 = icmp ugt i64 %604, %488, !dbg !24
  %615 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %616 = extractelement <32 x float> %606, i64 3, !dbg !14
  %617 = fadd float %616, %615, !dbg !14
  %.not40 = icmp ugt i64 %604, %492, !dbg !24
  %618 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %619 = extractelement <32 x float> %606, i64 4, !dbg !14
  %620 = fadd float %619, %618, !dbg !14
  %.not41 = icmp ugt i64 %604, %496, !dbg !24
  %621 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %622 = extractelement <32 x float> %606, i64 5, !dbg !14
  %623 = fadd float %622, %621, !dbg !14
  %.not42 = icmp ugt i64 %604, %500, !dbg !24
  %624 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %625 = extractelement <32 x float> %606, i64 6, !dbg !14
  %626 = fadd float %625, %624, !dbg !14
  %.not43 = icmp ugt i64 %604, %504, !dbg !24
  %627 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %628 = extractelement <32 x float> %606, i64 7, !dbg !14
  %629 = fadd float %628, %627, !dbg !14
  %.not44 = icmp ugt i64 %604, %508, !dbg !24
  %630 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %631 = extractelement <32 x float> %606, i64 8, !dbg !14
  %632 = fadd float %631, %630, !dbg !14
  %.not45 = icmp ugt i64 %604, %512, !dbg !24
  %633 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %634 = extractelement <32 x float> %606, i64 9, !dbg !14
  %635 = fadd float %634, %633, !dbg !14
  %.not46 = icmp ugt i64 %604, %516, !dbg !24
  %636 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %637 = extractelement <32 x float> %606, i64 10, !dbg !14
  %638 = fadd float %637, %636, !dbg !14
  %.not47 = icmp ugt i64 %604, %520, !dbg !24
  %639 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %640 = extractelement <32 x float> %606, i64 11, !dbg !14
  %641 = fadd float %640, %639, !dbg !14
  %.not48 = icmp ugt i64 %604, %524, !dbg !24
  %642 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %643 = extractelement <32 x float> %606, i64 12, !dbg !14
  %644 = fadd float %643, %642, !dbg !14
  %.not49 = icmp ugt i64 %604, %528, !dbg !24
  %645 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %646 = extractelement <32 x float> %606, i64 13, !dbg !14
  %647 = fadd float %646, %645, !dbg !14
  %.not50 = icmp ugt i64 %604, %532, !dbg !24
  %648 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %649 = extractelement <32 x float> %606, i64 14, !dbg !14
  %650 = fadd float %649, %648, !dbg !14
  %.not51 = icmp ugt i64 %604, %536, !dbg !24
  %651 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %652 = extractelement <32 x float> %606, i64 15, !dbg !14
  %653 = fadd float %652, %651, !dbg !14
  %.not52 = icmp ugt i64 %604, %540, !dbg !24
  %654 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %655 = extractelement <32 x float> %606, i64 16, !dbg !14
  %656 = fadd float %655, %654, !dbg !14
  %.not53 = icmp ugt i64 %604, %544, !dbg !24
  %657 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %658 = extractelement <32 x float> %606, i64 17, !dbg !14
  %659 = fadd float %658, %657, !dbg !14
  %.not54 = icmp ugt i64 %604, %548, !dbg !24
  %660 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %661 = extractelement <32 x float> %606, i64 18, !dbg !14
  %662 = fadd float %661, %660, !dbg !14
  %.not55 = icmp ugt i64 %604, %552, !dbg !24
  %663 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %664 = extractelement <32 x float> %606, i64 19, !dbg !14
  %665 = fadd float %664, %663, !dbg !14
  %.not56 = icmp ugt i64 %604, %556, !dbg !24
  %666 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %667 = extractelement <32 x float> %606, i64 20, !dbg !14
  %668 = fadd float %667, %666, !dbg !14
  %.not57 = icmp ugt i64 %604, %560, !dbg !24
  %669 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %670 = extractelement <32 x float> %606, i64 21, !dbg !14
  %671 = fadd float %670, %669, !dbg !14
  %.not58 = icmp ugt i64 %604, %564, !dbg !24
  %672 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %673 = extractelement <32 x float> %606, i64 22, !dbg !14
  %674 = fadd float %673, %672, !dbg !14
  %.not59 = icmp ugt i64 %604, %568, !dbg !24
  %675 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %676 = extractelement <32 x float> %606, i64 23, !dbg !14
  %677 = fadd float %676, %675, !dbg !14
  %.not60 = icmp ugt i64 %604, %572, !dbg !24
  %678 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %679 = extractelement <32 x float> %606, i64 24, !dbg !14
  %680 = fadd float %679, %678, !dbg !14
  %.not61 = icmp ugt i64 %604, %576, !dbg !24
  %681 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %682 = extractelement <32 x float> %606, i64 25, !dbg !14
  %683 = fadd float %682, %681, !dbg !14
  %.not62 = icmp ugt i64 %604, %580, !dbg !24
  %684 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %685 = extractelement <32 x float> %606, i64 26, !dbg !14
  %686 = fadd float %685, %684, !dbg !14
  %.not63 = icmp ugt i64 %604, %584, !dbg !24
  %687 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %688 = extractelement <32 x float> %606, i64 27, !dbg !14
  %689 = fadd float %688, %687, !dbg !14
  %.not64 = icmp ugt i64 %604, %588, !dbg !24
  %690 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %691 = extractelement <32 x float> %606, i64 28, !dbg !14
  %692 = fadd float %691, %690, !dbg !14
  %.not65 = icmp ugt i64 %604, %592, !dbg !24
  %693 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %694 = extractelement <32 x float> %606, i64 29, !dbg !14
  %695 = fadd float %694, %693, !dbg !14
  %.not66 = icmp ugt i64 %604, %596, !dbg !24
  %696 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %697 = extractelement <32 x float> %606, i64 30, !dbg !14
  %698 = fadd float %697, %696, !dbg !14
  %.not67 = icmp ugt i64 %604, %600, !dbg !24
  %699 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %700 = extractelement <32 x float> %606, i64 31, !dbg !14
  %701 = fadd float %700, %699, !dbg !14
  %702 = insertelement <32 x float> poison, float %479, i64 0, !dbg !14
  %703 = insertelement <32 x float> %702, float %483, i64 1, !dbg !14
  %704 = insertelement <32 x float> %703, float %487, i64 2, !dbg !14
  %705 = insertelement <32 x float> %704, float %491, i64 3, !dbg !14
  %706 = insertelement <32 x float> %705, float %495, i64 4, !dbg !14
  %707 = insertelement <32 x float> %706, float %499, i64 5, !dbg !14
  %708 = insertelement <32 x float> %707, float %503, i64 6, !dbg !14
  %709 = insertelement <32 x float> %708, float %507, i64 7, !dbg !14
  %710 = insertelement <32 x float> %709, float %511, i64 8, !dbg !14
  %711 = insertelement <32 x float> %710, float %515, i64 9, !dbg !14
  %712 = insertelement <32 x float> %711, float %519, i64 10, !dbg !14
  %713 = insertelement <32 x float> %712, float %523, i64 11, !dbg !14
  %714 = insertelement <32 x float> %713, float %527, i64 12, !dbg !14
  %715 = insertelement <32 x float> %714, float %531, i64 13, !dbg !14
  %716 = insertelement <32 x float> %715, float %535, i64 14, !dbg !14
  %717 = insertelement <32 x float> %716, float %539, i64 15, !dbg !14
  %718 = insertelement <32 x float> %717, float %543, i64 16, !dbg !14
  %719 = insertelement <32 x float> %718, float %547, i64 17, !dbg !14
  %720 = insertelement <32 x float> %719, float %551, i64 18, !dbg !14
  %721 = insertelement <32 x float> %720, float %555, i64 19, !dbg !14
  %722 = insertelement <32 x float> %721, float %559, i64 20, !dbg !14
  %723 = insertelement <32 x float> %722, float %563, i64 21, !dbg !14
  %724 = insertelement <32 x float> %723, float %567, i64 22, !dbg !14
  %725 = insertelement <32 x float> %724, float %571, i64 23, !dbg !14
  %726 = insertelement <32 x float> %725, float %575, i64 24, !dbg !14
  %727 = insertelement <32 x float> %726, float %579, i64 25, !dbg !14
  %728 = insertelement <32 x float> %727, float %583, i64 26, !dbg !14
  %729 = insertelement <32 x float> %728, float %587, i64 27, !dbg !14
  %730 = insertelement <32 x float> %729, float %591, i64 28, !dbg !14
  %731 = insertelement <32 x float> %730, float %595, i64 29, !dbg !14
  %732 = insertelement <32 x float> %731, float %599, i64 30, !dbg !14
  %733 = insertelement <32 x float> %732, float %603, i64 31, !dbg !14
  %734 = insertelement <32 x float> poison, float %608, i64 0, !dbg !14
  %735 = insertelement <32 x float> %734, float %611, i64 1, !dbg !14
  %736 = insertelement <32 x float> %735, float %614, i64 2, !dbg !14
  %737 = insertelement <32 x float> %736, float %617, i64 3, !dbg !14
  %738 = insertelement <32 x float> %737, float %620, i64 4, !dbg !14
  %739 = insertelement <32 x float> %738, float %623, i64 5, !dbg !14
  %740 = insertelement <32 x float> %739, float %626, i64 6, !dbg !14
  %741 = insertelement <32 x float> %740, float %629, i64 7, !dbg !14
  %742 = insertelement <32 x float> %741, float %632, i64 8, !dbg !14
  %743 = insertelement <32 x float> %742, float %635, i64 9, !dbg !14
  %744 = insertelement <32 x float> %743, float %638, i64 10, !dbg !14
  %745 = insertelement <32 x float> %744, float %641, i64 11, !dbg !14
  %746 = insertelement <32 x float> %745, float %644, i64 12, !dbg !14
  %747 = insertelement <32 x float> %746, float %647, i64 13, !dbg !14
  %748 = insertelement <32 x float> %747, float %650, i64 14, !dbg !14
  %749 = insertelement <32 x float> %748, float %653, i64 15, !dbg !14
  %750 = insertelement <32 x float> %749, float %656, i64 16, !dbg !14
  %751 = insertelement <32 x float> %750, float %659, i64 17, !dbg !14
  %752 = insertelement <32 x float> %751, float %662, i64 18, !dbg !14
  %753 = insertelement <32 x float> %752, float %665, i64 19, !dbg !14
  %754 = insertelement <32 x float> %753, float %668, i64 20, !dbg !14
  %755 = insertelement <32 x float> %754, float %671, i64 21, !dbg !14
  %756 = insertelement <32 x float> %755, float %674, i64 22, !dbg !14
  %757 = insertelement <32 x float> %756, float %677, i64 23, !dbg !14
  %758 = insertelement <32 x float> %757, float %680, i64 24, !dbg !14
  %759 = insertelement <32 x float> %758, float %683, i64 25, !dbg !14
  %760 = insertelement <32 x float> %759, float %686, i64 26, !dbg !14
  %761 = insertelement <32 x float> %760, float %689, i64 27, !dbg !14
  %762 = insertelement <32 x float> %761, float %692, i64 28, !dbg !14
  %763 = insertelement <32 x float> %762, float %695, i64 29, !dbg !14
  %764 = insertelement <32 x float> %763, float %698, i64 30, !dbg !14
  %765 = insertelement <32 x float> %764, float %701, i64 31, !dbg !14
  br label %766, !dbg !14

766:                                              ; preds = %272, %467
  %.pn70 = phi <32 x float> [ %733, %467 ], [ %434, %272 ]
  %.pn68 = phi <32 x float> [ %765, %467 ], [ %466, %272 ]
  br label %767, !dbg !14

767:                                              ; preds = %766
  %768 = extractelement <32 x float> %.pn70, i64 0, !dbg !26
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !26
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !26
  %771 = extractelement <32 x float> %.pn70, i64 1, !dbg !26
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !26
  %773 = fadd float %769, %772, !dbg !27
  %774 = insertelement <4 x float> %770, float %772, i64 1, !dbg !26
  %775 = extractelement <32 x float> %.pn70, i64 2, !dbg !26
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !26
  %777 = fadd float %773, %776, !dbg !27
  %778 = insertelement <4 x float> %774, float %776, i64 2, !dbg !26
  %779 = extractelement <32 x float> %.pn70, i64 3, !dbg !26
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !26
  %781 = fadd float %777, %780, !dbg !27
  %782 = insertelement <4 x float> %778, float %780, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %782) #9, !dbg !28, !frisk.register_lifetimes !1
  %783 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !28
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %783) #9, !dbg !26, !frisk.register_lifetimes !1
  %784 = extractelement <32 x float> %.pn70, i64 4, !dbg !26
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !26
  %786 = fadd float %781, %785, !dbg !27
  %787 = insertelement <4 x float> poison, float %785, i64 0, !dbg !26
  %788 = extractelement <32 x float> %.pn70, i64 5, !dbg !26
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !26
  %790 = fadd float %786, %789, !dbg !27
  %791 = insertelement <4 x float> %787, float %789, i64 1, !dbg !26
  %792 = extractelement <32 x float> %.pn70, i64 6, !dbg !26
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !26
  %794 = fadd float %790, %793, !dbg !27
  %795 = insertelement <4 x float> %791, float %793, i64 2, !dbg !26
  %796 = extractelement <32 x float> %.pn70, i64 7, !dbg !26
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !26
  %798 = fadd float %794, %797, !dbg !27
  %799 = insertelement <4 x float> %795, float %797, i64 3, !dbg !26
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %799) #9, !dbg !28, !frisk.register_lifetimes !1
  %800 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !28
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %800) #9, !dbg !26, !frisk.register_lifetimes !1
  %801 = extractelement <32 x float> %.pn70, i64 8, !dbg !26
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !26
  %803 = fadd float %798, %802, !dbg !27
  %804 = insertelement <4 x float> poison, float %802, i64 0, !dbg !26
  %805 = extractelement <32 x float> %.pn70, i64 9, !dbg !26
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !26
  %807 = fadd float %803, %806, !dbg !27
  %808 = insertelement <4 x float> %804, float %806, i64 1, !dbg !26
  %809 = extractelement <32 x float> %.pn70, i64 10, !dbg !26
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !26
  %811 = fadd float %807, %810, !dbg !27
  %812 = insertelement <4 x float> %808, float %810, i64 2, !dbg !26
  %813 = extractelement <32 x float> %.pn70, i64 11, !dbg !26
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !26
  %815 = fadd float %811, %814, !dbg !27
  %816 = insertelement <4 x float> %812, float %814, i64 3, !dbg !26
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %816) #9, !dbg !28, !frisk.register_lifetimes !1
  %817 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !28
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %817) #9, !dbg !26, !frisk.register_lifetimes !1
  %818 = extractelement <32 x float> %.pn70, i64 12, !dbg !26
  %819 = call float @llvm.amdgcn.exp2.f32(float %818), !dbg !26
  %820 = fadd float %815, %819, !dbg !27
  %821 = insertelement <4 x float> poison, float %819, i64 0, !dbg !26
  %822 = extractelement <32 x float> %.pn70, i64 13, !dbg !26
  %823 = call float @llvm.amdgcn.exp2.f32(float %822), !dbg !26
  %824 = fadd float %820, %823, !dbg !27
  %825 = insertelement <4 x float> %821, float %823, i64 1, !dbg !26
  %826 = extractelement <32 x float> %.pn70, i64 14, !dbg !26
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !26
  %828 = fadd float %824, %827, !dbg !27
  %829 = insertelement <4 x float> %825, float %827, i64 2, !dbg !26
  %830 = extractelement <32 x float> %.pn70, i64 15, !dbg !26
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !26
  %832 = fadd float %828, %831, !dbg !27
  %833 = insertelement <4 x float> %829, float %831, i64 3, !dbg !26
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %833) #9, !dbg !28, !frisk.register_lifetimes !1
  %834 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !28
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %834) #9, !dbg !26, !frisk.register_lifetimes !1
  %835 = extractelement <32 x float> %.pn70, i64 16, !dbg !26
  %836 = call float @llvm.amdgcn.exp2.f32(float %835), !dbg !26
  %837 = fadd float %832, %836, !dbg !27
  %838 = insertelement <4 x float> poison, float %836, i64 0, !dbg !26
  %839 = extractelement <32 x float> %.pn70, i64 17, !dbg !26
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !26
  %841 = fadd float %837, %840, !dbg !27
  %842 = insertelement <4 x float> %838, float %840, i64 1, !dbg !26
  %843 = extractelement <32 x float> %.pn70, i64 18, !dbg !26
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !26
  %845 = fadd float %841, %844, !dbg !27
  %846 = insertelement <4 x float> %842, float %844, i64 2, !dbg !26
  %847 = extractelement <32 x float> %.pn70, i64 19, !dbg !26
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !26
  %849 = fadd float %845, %848, !dbg !27
  %850 = insertelement <4 x float> %846, float %848, i64 3, !dbg !26
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %850) #9, !dbg !28, !frisk.register_lifetimes !1
  %851 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !28
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %851) #9, !dbg !26, !frisk.register_lifetimes !1
  %852 = extractelement <32 x float> %.pn70, i64 20, !dbg !26
  %853 = call float @llvm.amdgcn.exp2.f32(float %852), !dbg !26
  %854 = fadd float %849, %853, !dbg !27
  %855 = insertelement <4 x float> poison, float %853, i64 0, !dbg !26
  %856 = extractelement <32 x float> %.pn70, i64 21, !dbg !26
  %857 = call float @llvm.amdgcn.exp2.f32(float %856), !dbg !26
  %858 = fadd float %854, %857, !dbg !27
  %859 = insertelement <4 x float> %855, float %857, i64 1, !dbg !26
  %860 = extractelement <32 x float> %.pn70, i64 22, !dbg !26
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !26
  %862 = fadd float %858, %861, !dbg !27
  %863 = insertelement <4 x float> %859, float %861, i64 2, !dbg !26
  %864 = extractelement <32 x float> %.pn70, i64 23, !dbg !26
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !26
  %866 = fadd float %862, %865, !dbg !27
  %867 = insertelement <4 x float> %863, float %865, i64 3, !dbg !26
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %867) #9, !dbg !28, !frisk.register_lifetimes !1
  %868 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !28
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %868) #9, !dbg !26, !frisk.register_lifetimes !1
  %869 = extractelement <32 x float> %.pn70, i64 24, !dbg !26
  %870 = call float @llvm.amdgcn.exp2.f32(float %869), !dbg !26
  %871 = fadd float %866, %870, !dbg !27
  %872 = insertelement <4 x float> poison, float %870, i64 0, !dbg !26
  %873 = extractelement <32 x float> %.pn70, i64 25, !dbg !26
  %874 = call float @llvm.amdgcn.exp2.f32(float %873), !dbg !26
  %875 = fadd float %871, %874, !dbg !27
  %876 = insertelement <4 x float> %872, float %874, i64 1, !dbg !26
  %877 = extractelement <32 x float> %.pn70, i64 26, !dbg !26
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !26
  %879 = fadd float %875, %878, !dbg !27
  %880 = insertelement <4 x float> %876, float %878, i64 2, !dbg !26
  %881 = extractelement <32 x float> %.pn70, i64 27, !dbg !26
  %882 = call float @llvm.amdgcn.exp2.f32(float %881), !dbg !26
  %883 = fadd float %879, %882, !dbg !27
  %884 = insertelement <4 x float> %880, float %882, i64 3, !dbg !26
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %884) #9, !dbg !28, !frisk.register_lifetimes !1
  %885 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !28
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %885) #9, !dbg !26, !frisk.register_lifetimes !1
  %886 = extractelement <32 x float> %.pn70, i64 28, !dbg !26
  %887 = call float @llvm.amdgcn.exp2.f32(float %886), !dbg !26
  %888 = fadd float %883, %887, !dbg !27
  %889 = insertelement <4 x float> poison, float %887, i64 0, !dbg !26
  %890 = extractelement <32 x float> %.pn70, i64 29, !dbg !26
  %891 = call float @llvm.amdgcn.exp2.f32(float %890), !dbg !26
  %892 = fadd float %888, %891, !dbg !27
  %893 = insertelement <4 x float> %889, float %891, i64 1, !dbg !26
  %894 = extractelement <32 x float> %.pn70, i64 30, !dbg !26
  %895 = call float @llvm.amdgcn.exp2.f32(float %894), !dbg !26
  %896 = fadd float %892, %895, !dbg !27
  %897 = insertelement <4 x float> %893, float %895, i64 2, !dbg !26
  %898 = extractelement <32 x float> %.pn70, i64 31, !dbg !26
  %899 = call float @llvm.amdgcn.exp2.f32(float %898), !dbg !26
  %900 = fadd float %896, %899, !dbg !27
  %901 = insertelement <4 x float> %897, float %899, i64 3, !dbg !26
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %901) #9, !dbg !28, !frisk.register_lifetimes !1
  %902 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !28
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %902) #9, !dbg !26, !frisk.register_lifetimes !1
  %903 = extractelement <32 x float> %.pn68, i64 0, !dbg !26
  %904 = call float @llvm.amdgcn.exp2.f32(float %903), !dbg !26
  %905 = insertelement <4 x float> poison, float %904, i64 0, !dbg !26
  %906 = extractelement <32 x float> %.pn68, i64 1, !dbg !26
  %907 = call float @llvm.amdgcn.exp2.f32(float %906), !dbg !26
  %908 = fadd float %904, %907, !dbg !27
  %909 = insertelement <4 x float> %905, float %907, i64 1, !dbg !26
  %910 = extractelement <32 x float> %.pn68, i64 2, !dbg !26
  %911 = call float @llvm.amdgcn.exp2.f32(float %910), !dbg !26
  %912 = fadd float %908, %911, !dbg !27
  %913 = insertelement <4 x float> %909, float %911, i64 2, !dbg !26
  %914 = extractelement <32 x float> %.pn68, i64 3, !dbg !26
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !26
  %916 = fadd float %912, %915, !dbg !27
  %917 = insertelement <4 x float> %913, float %915, i64 3, !dbg !26
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %917) #9, !dbg !28, !frisk.register_lifetimes !1
  %918 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !28
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %918) #9, !dbg !26, !frisk.register_lifetimes !1
  %919 = extractelement <32 x float> %.pn68, i64 4, !dbg !26
  %920 = call float @llvm.amdgcn.exp2.f32(float %919), !dbg !26
  %921 = fadd float %916, %920, !dbg !27
  %922 = insertelement <4 x float> poison, float %920, i64 0, !dbg !26
  %923 = extractelement <32 x float> %.pn68, i64 5, !dbg !26
  %924 = call float @llvm.amdgcn.exp2.f32(float %923), !dbg !26
  %925 = fadd float %921, %924, !dbg !27
  %926 = insertelement <4 x float> %922, float %924, i64 1, !dbg !26
  %927 = extractelement <32 x float> %.pn68, i64 6, !dbg !26
  %928 = call float @llvm.amdgcn.exp2.f32(float %927), !dbg !26
  %929 = fadd float %925, %928, !dbg !27
  %930 = insertelement <4 x float> %926, float %928, i64 2, !dbg !26
  %931 = extractelement <32 x float> %.pn68, i64 7, !dbg !26
  %932 = call float @llvm.amdgcn.exp2.f32(float %931), !dbg !26
  %933 = fadd float %929, %932, !dbg !27
  %934 = insertelement <4 x float> %930, float %932, i64 3, !dbg !26
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %934) #9, !dbg !28, !frisk.register_lifetimes !1
  %935 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !28
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %935) #9, !dbg !26, !frisk.register_lifetimes !1
  %936 = extractelement <32 x float> %.pn68, i64 8, !dbg !26
  %937 = call float @llvm.amdgcn.exp2.f32(float %936), !dbg !26
  %938 = fadd float %933, %937, !dbg !27
  %939 = insertelement <4 x float> poison, float %937, i64 0, !dbg !26
  %940 = extractelement <32 x float> %.pn68, i64 9, !dbg !26
  %941 = call float @llvm.amdgcn.exp2.f32(float %940), !dbg !26
  %942 = fadd float %938, %941, !dbg !27
  %943 = insertelement <4 x float> %939, float %941, i64 1, !dbg !26
  %944 = extractelement <32 x float> %.pn68, i64 10, !dbg !26
  %945 = call float @llvm.amdgcn.exp2.f32(float %944), !dbg !26
  %946 = fadd float %942, %945, !dbg !27
  %947 = insertelement <4 x float> %943, float %945, i64 2, !dbg !26
  %948 = extractelement <32 x float> %.pn68, i64 11, !dbg !26
  %949 = call float @llvm.amdgcn.exp2.f32(float %948), !dbg !26
  %950 = fadd float %946, %949, !dbg !27
  %951 = insertelement <4 x float> %947, float %949, i64 3, !dbg !26
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %951) #9, !dbg !28, !frisk.register_lifetimes !1
  %952 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !28
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %952) #9, !dbg !26, !frisk.register_lifetimes !1
  %953 = extractelement <32 x float> %.pn68, i64 12, !dbg !26
  %954 = call float @llvm.amdgcn.exp2.f32(float %953), !dbg !26
  %955 = fadd float %950, %954, !dbg !27
  %956 = insertelement <4 x float> poison, float %954, i64 0, !dbg !26
  %957 = extractelement <32 x float> %.pn68, i64 13, !dbg !26
  %958 = call float @llvm.amdgcn.exp2.f32(float %957), !dbg !26
  %959 = fadd float %955, %958, !dbg !27
  %960 = insertelement <4 x float> %956, float %958, i64 1, !dbg !26
  %961 = extractelement <32 x float> %.pn68, i64 14, !dbg !26
  %962 = call float @llvm.amdgcn.exp2.f32(float %961), !dbg !26
  %963 = fadd float %959, %962, !dbg !27
  %964 = insertelement <4 x float> %960, float %962, i64 2, !dbg !26
  %965 = extractelement <32 x float> %.pn68, i64 15, !dbg !26
  %966 = call float @llvm.amdgcn.exp2.f32(float %965), !dbg !26
  %967 = fadd float %963, %966, !dbg !27
  %968 = insertelement <4 x float> %964, float %966, i64 3, !dbg !26
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %968) #9, !dbg !28, !frisk.register_lifetimes !1
  %969 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !28
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %969) #9, !dbg !26, !frisk.register_lifetimes !1
  %970 = extractelement <32 x float> %.pn68, i64 16, !dbg !26
  %971 = call float @llvm.amdgcn.exp2.f32(float %970), !dbg !26
  %972 = fadd float %967, %971, !dbg !27
  %973 = insertelement <4 x float> poison, float %971, i64 0, !dbg !26
  %974 = extractelement <32 x float> %.pn68, i64 17, !dbg !26
  %975 = call float @llvm.amdgcn.exp2.f32(float %974), !dbg !26
  %976 = fadd float %972, %975, !dbg !27
  %977 = insertelement <4 x float> %973, float %975, i64 1, !dbg !26
  %978 = extractelement <32 x float> %.pn68, i64 18, !dbg !26
  %979 = call float @llvm.amdgcn.exp2.f32(float %978), !dbg !26
  %980 = fadd float %976, %979, !dbg !27
  %981 = insertelement <4 x float> %977, float %979, i64 2, !dbg !26
  %982 = extractelement <32 x float> %.pn68, i64 19, !dbg !26
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !26
  %984 = fadd float %980, %983, !dbg !27
  %985 = insertelement <4 x float> %981, float %983, i64 3, !dbg !26
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %985) #9, !dbg !28, !frisk.register_lifetimes !1
  %986 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !28
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %986) #9, !dbg !26, !frisk.register_lifetimes !1
  %987 = extractelement <32 x float> %.pn68, i64 20, !dbg !26
  %988 = call float @llvm.amdgcn.exp2.f32(float %987), !dbg !26
  %989 = fadd float %984, %988, !dbg !27
  %990 = insertelement <4 x float> poison, float %988, i64 0, !dbg !26
  %991 = extractelement <32 x float> %.pn68, i64 21, !dbg !26
  %992 = call float @llvm.amdgcn.exp2.f32(float %991), !dbg !26
  %993 = fadd float %989, %992, !dbg !27
  %994 = insertelement <4 x float> %990, float %992, i64 1, !dbg !26
  %995 = extractelement <32 x float> %.pn68, i64 22, !dbg !26
  %996 = call float @llvm.amdgcn.exp2.f32(float %995), !dbg !26
  %997 = fadd float %993, %996, !dbg !27
  %998 = insertelement <4 x float> %994, float %996, i64 2, !dbg !26
  %999 = extractelement <32 x float> %.pn68, i64 23, !dbg !26
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999), !dbg !26
  %1001 = fadd float %997, %1000, !dbg !27
  %1002 = insertelement <4 x float> %998, float %1000, i64 3, !dbg !26
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1002) #9, !dbg !28, !frisk.register_lifetimes !1
  %1003 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !28
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1003) #9, !dbg !26, !frisk.register_lifetimes !1
  %1004 = extractelement <32 x float> %.pn68, i64 24, !dbg !26
  %1005 = call float @llvm.amdgcn.exp2.f32(float %1004), !dbg !26
  %1006 = fadd float %1001, %1005, !dbg !27
  %1007 = insertelement <4 x float> poison, float %1005, i64 0, !dbg !26
  %1008 = extractelement <32 x float> %.pn68, i64 25, !dbg !26
  %1009 = call float @llvm.amdgcn.exp2.f32(float %1008), !dbg !26
  %1010 = fadd float %1006, %1009, !dbg !27
  %1011 = insertelement <4 x float> %1007, float %1009, i64 1, !dbg !26
  %1012 = extractelement <32 x float> %.pn68, i64 26, !dbg !26
  %1013 = call float @llvm.amdgcn.exp2.f32(float %1012), !dbg !26
  %1014 = fadd float %1010, %1013, !dbg !27
  %1015 = insertelement <4 x float> %1011, float %1013, i64 2, !dbg !26
  %1016 = extractelement <32 x float> %.pn68, i64 27, !dbg !26
  %1017 = call float @llvm.amdgcn.exp2.f32(float %1016), !dbg !26
  %1018 = fadd float %1014, %1017, !dbg !27
  %1019 = insertelement <4 x float> %1015, float %1017, i64 3, !dbg !26
  %math.result129 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1019) #9, !dbg !28, !frisk.register_lifetimes !1
  %1020 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !28
  %math.packed130 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1020) #9, !dbg !26, !frisk.register_lifetimes !1
  %1021 = extractelement <32 x float> %.pn68, i64 28, !dbg !26
  %1022 = call float @llvm.amdgcn.exp2.f32(float %1021), !dbg !26
  %1023 = fadd float %1018, %1022, !dbg !27
  %1024 = insertelement <4 x float> poison, float %1022, i64 0, !dbg !26
  %1025 = extractelement <32 x float> %.pn68, i64 29, !dbg !26
  %1026 = call float @llvm.amdgcn.exp2.f32(float %1025), !dbg !26
  %1027 = fadd float %1023, %1026, !dbg !27
  %1028 = insertelement <4 x float> %1024, float %1026, i64 1, !dbg !26
  %1029 = extractelement <32 x float> %.pn68, i64 30, !dbg !26
  %1030 = call float @llvm.amdgcn.exp2.f32(float %1029), !dbg !26
  %1031 = fadd float %1027, %1030, !dbg !27
  %1032 = insertelement <4 x float> %1028, float %1030, i64 2, !dbg !26
  %1033 = extractelement <32 x float> %.pn68, i64 31, !dbg !26
  %1034 = call float @llvm.amdgcn.exp2.f32(float %1033), !dbg !26
  %1035 = fadd float %1031, %1034, !dbg !27
  %1036 = insertelement <4 x float> %1032, float %1034, i64 3, !dbg !26
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1036) #9, !dbg !28, !frisk.register_lifetimes !1
  %1037 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !28
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1037) #9, !dbg !27, !frisk.register_lifetimes !1
  %1038 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %1039 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1038), !dbg !27
  %1040 = add nuw nsw i32 %1039, 64, !dbg !27
  %1041 = xor i32 %1039, 16, !dbg !27
  %1042 = and i32 %1040, 192, !dbg !27
  %1043 = icmp samesign ult i32 %1041, %1042, !dbg !27
  %1044 = select i1 %1043, i32 %1041, i32 %1039, !dbg !27
  %1045 = shl nuw nsw i32 %1044, 2, !dbg !27
  %1046 = bitcast float %900 to i32, !dbg !27
  %1047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1045, i32 %1046), !dbg !27
  %1048 = bitcast i32 %1047 to float, !dbg !27
  %1049 = fadd float %900, %1048, !dbg !27
  %1050 = xor i32 %1039, 32, !dbg !27
  %1051 = icmp samesign ult i32 %1050, %1042, !dbg !27
  %1052 = select i1 %1051, i32 %1050, i32 %1039, !dbg !27
  %1053 = shl nuw nsw i32 %1052, 2, !dbg !27
  %1054 = bitcast float %1049 to i32, !dbg !27
  %1055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1054), !dbg !27
  %1056 = bitcast i32 %1055 to float, !dbg !27
  %1057 = fadd float %1049, %1056, !dbg !27
  %1058 = extractvalue [2 x <1 x float>] %28, 0, !dbg !27
  %1059 = extractelement <1 x float> %1058, i64 0, !dbg !27
  %1060 = fadd float %1057, %1059, !dbg !27
  %1061 = insertelement <1 x float> poison, float %1060, i64 0, !dbg !16
  %1062 = insertvalue [2 x <1 x float>] poison, <1 x float> %1061, 0, !dbg !16
  %1063 = bitcast float %1035 to i32, !dbg !27
  %1064 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1045, i32 %1063), !dbg !27
  %1065 = bitcast i32 %1064 to float, !dbg !27
  %1066 = fadd float %1035, %1065, !dbg !27
  %1067 = bitcast float %1066 to i32, !dbg !27
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1053, i32 %1067), !dbg !27
  %1069 = bitcast i32 %1068 to float, !dbg !27
  %1070 = fadd float %1066, %1069, !dbg !27
  %1071 = extractvalue [2 x <1 x float>] %28, 1, !dbg !27
  %1072 = extractelement <1 x float> %1071, i64 0, !dbg !27
  %1073 = fadd float %1070, %1072, !dbg !27
  %1074 = insertelement <1 x float> poison, float %1073, i64 0, !dbg !16
  %1075 = insertvalue [2 x <1 x float>] %1062, <1 x float> %1074, 1, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread324 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !33, !frisk.register_lifetimes !1
  %address.local325 = shl nuw nsw i32 %address.thread324, 3, !dbg !15
  %address.local326 = and i32 %address.local325, 120, !dbg !15
  %address.local327 = shl nuw nsw i32 %address.thread324, 5, !dbg !15
  %address.local328 = and i32 %address.local327, 1536, !dbg !15
  %address.local329 = or disjoint i32 %address.local326, %address.local328, !dbg !15
  %address.local330 = zext nneg i32 %address.local329 to i64, !dbg !15
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !15
  %1076 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !15
  %1077 = bitcast <4 x i32> %1076 to <8 x half>, !dbg !15
  %address.local332 = zext nneg i32 %address.thread324 to i64, !dbg !13
  %address.local333 = and i64 %address.local332, 60, !dbg !15
  %address.local334 = add nuw nsw i64 %address.local333, %address.local332, !dbg !15
  %address.local335 = shl nuw nsw i64 %address.local334, 3, !dbg !15
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !15
  %address.local337 = and i64 %address.local332, 63, !dbg !13
  %address.local338 = lshr i64 %address.local337, 4, !dbg !15
  %address.local339 = mul nuw nsw i64 %address.local338, 384, !dbg !15
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) %address.local336, i64 %address.local339, !dbg !15
  %address.local341 = lshr i64 %address.local332, 2, !dbg !8
  %address.local342 = mul nsw i64 %address.local341, -32, !dbg !15
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) %address.local340, i64 %address.local342, !dbg !15
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 256, !dbg !15
  %1078 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !15
  %1079 = bitcast <4 x i32> %1078 to <8 x half>, !dbg !15
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !15
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local339, !dbg !15
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) %address.local346, i64 %address.local342, !dbg !15
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local347, i64 512, !dbg !15
  %1080 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %1081 = bitcast <4 x i32> %1080 to <8 x half>, !dbg !15
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !15
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) %address.local349, i64 %address.local339, !dbg !15
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local342, !dbg !15
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local351, i64 768, !dbg !15
  %1082 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %1083 = bitcast <4 x i32> %1082 to <8 x half>, !dbg !15
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !15
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local353, i64 4096, !dbg !15
  %1084 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !15
  %1085 = bitcast <4 x i32> %1084 to <8 x half>, !dbg !15
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !15
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local339, !dbg !15
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local342, !dbg !15
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local357, i64 4352, !dbg !15
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !15
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !15
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !15
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local339, !dbg !15
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %address.local360, i64 %address.local342, !dbg !15
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !15
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !15
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !15
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !15
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) %address.local363, i64 %address.local339, !dbg !15
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) %address.local364, i64 %address.local342, !dbg !15
  %address.local366 = getelementptr i8, ptr addrspace(3) %address.local365, i64 4864, !dbg !15
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !15
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !15
  %1092 = add nsw i64 %34, %8, !dbg !18
  %1093 = shl i64 %1092, 19, !dbg !18
  %1094 = and i64 %1093, 16252928, !dbg !18
  %1095 = add i64 %1094, %238, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1096 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1095, !dbg !18
  %1097 = getelementptr i8, ptr addrspace(1) %1096, i64 64, !dbg !18
  %address.local367 = shl nuw nsw i64 %address.local332, 3, !dbg !12
  %address.local368 = and i64 %address.local367, 1536, !dbg !12
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !12
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !12
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !12
  %1098 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !18
  %1099 = zext i32 %1098 to i64, !dbg !18
  %1100 = inttoptr i64 %1099 to ptr addrspace(3), !dbg !18
  %address.local372 = shl nuw nsw i64 %address.local332, 5, !dbg !18
  %address.local373 = and i64 %address.local372, 8064, !dbg !18
  %address.local374 = shl nuw nsw i64 %address.local332, 3, !dbg !18
  %address.local375 = and i64 %address.local374, 24, !dbg !18
  %address.local376 = or disjoint i64 %address.local373, %address.local375, !dbg !18
  %address.local377 = getelementptr [2 x i8], ptr addrspace(1) %1097, i64 %address.local376, !dbg !18
  %1101 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1102 = ptrtoint ptr addrspace(1) %address.local377 to i64, !dbg !18
  %1103 = sub i64 %1102, %1101, !dbg !18
  %dma.buffer.offset641 = trunc i64 %1103 to i32, !dbg !18
  %1104 = trunc i64 %1101 to i32, !dbg !18
  %1105 = insertelement <4 x i32> poison, i32 %1104, i64 0, !dbg !18
  %1106 = lshr i64 %1101, 32, !dbg !18
  %1107 = trunc i64 %1106 to i32, !dbg !18
  %1108 = insertelement <4 x i32> %1105, i32 %1107, i64 1, !dbg !18
  %1109 = insertelement <4 x i32> %1108, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc642 = insertelement <4 x i32> %1109, i32 131072, i64 3, !dbg !18
  %1110 = ptrtoint ptr addrspace(3) %1100 to i32, !dbg !18
  %dma.buffer.lds643 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1110), !dbg !18
  %1111 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset641, <4 x i32> %dma.buffer.rsrc642) #10, !dbg !18
  %address.local378 = or disjoint i64 %address.local332, 256, !dbg !12
  %address.local379 = shl nuw nsw i64 %address.local378, 3, !dbg !12
  %address.local380 = and i64 %address.local379, 3584, !dbg !12
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local380, !dbg !12
  %address.local382 = ptrtoint ptr addrspace(3) %address.local381 to i64, !dbg !12
  %address.local383 = trunc i64 %address.local382 to i32, !dbg !12
  %1112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !18
  %1113 = zext i32 %1112 to i64, !dbg !18
  %1114 = inttoptr i64 %1113 to ptr addrspace(3), !dbg !18
  %address.local384 = shl nuw nsw i64 %address.local378, 5, !dbg !18
  %address.local385 = and i64 %address.local384, 16256, !dbg !18
  %address.local386 = shl nuw nsw i64 %address.local332, 3, !dbg !18
  %address.local387 = and i64 %address.local386, 24, !dbg !18
  %address.local388 = or disjoint i64 %address.local385, %address.local387, !dbg !18
  %address.local389 = getelementptr [2 x i8], ptr addrspace(1) %1097, i64 %address.local388, !dbg !18
  %1115 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1116 = ptrtoint ptr addrspace(1) %address.local389 to i64, !dbg !18
  %1117 = sub i64 %1116, %1115, !dbg !18
  %dma.buffer.offset638 = trunc i64 %1117 to i32, !dbg !18
  %1118 = trunc i64 %1115 to i32, !dbg !18
  %1119 = insertelement <4 x i32> poison, i32 %1118, i64 0, !dbg !18
  %1120 = lshr i64 %1115, 32, !dbg !18
  %1121 = trunc i64 %1120 to i32, !dbg !18
  %1122 = insertelement <4 x i32> %1119, i32 %1121, i64 1, !dbg !18
  %1123 = insertelement <4 x i32> %1122, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc639 = insertelement <4 x i32> %1123, i32 131072, i64 3, !dbg !18
  %1124 = ptrtoint ptr addrspace(3) %1114 to i32, !dbg !18
  %dma.buffer.lds640 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1124), !dbg !18
  %1125 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset638, <4 x i32> %dma.buffer.rsrc639) #10, !dbg !18
  %1126 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1127 = extractvalue [2 x <8 x float>] %27, 0, !dbg !29
  %1128 = shufflevector <8 x float> %1127, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1129 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1126, <4 x float> %1128) #11, !dbg !15
  %1130 = shufflevector <8 x half> %1079, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1131 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1130, <4 x float> %1129) #11, !dbg !15
  %1132 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1133 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1132, <4 x float> %1131) #11, !dbg !15
  %1134 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1135 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1134, <4 x float> %1133) #11, !dbg !15
  %1136 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1137 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1136, <4 x float> %1135) #11, !dbg !15
  %1138 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1139 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1138, <4 x float> %1137) #11, !dbg !15
  %1140 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1141 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1140, <4 x float> %1139) #11, !dbg !15
  %1142 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1143 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1142, <4 x float> %1141) #11, !dbg !15
  %1144 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1145 = shufflevector <8 x float> %1127, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1146 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1144, <4 x float> %1145) #11, !dbg !15
  %1147 = shufflevector <8 x half> %1079, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1148 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1147, <4 x float> %1146) #11, !dbg !15
  %1149 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1150 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1149, <4 x float> %1148) #11, !dbg !15
  %1151 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1152 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1151, <4 x float> %1150) #11, !dbg !15
  %1153 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1153, <4 x float> %1152) #11, !dbg !15
  %1155 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1155, <4 x float> %1154) #11, !dbg !15
  %1157 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1158 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1157, <4 x float> %1156) #11, !dbg !15
  %1159 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1160 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1159, <4 x float> %1158) #11, !dbg !15
  %1161 = extractvalue [2 x <8 x float>] %27, 1, !dbg !29
  %1162 = shufflevector <8 x float> %1161, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1163 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1126, <4 x float> %1162) #11, !dbg !15
  %1164 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1130, <4 x float> %1163) #11, !dbg !15
  %1165 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1132, <4 x float> %1164) #11, !dbg !15
  %1166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1134, <4 x float> %1165) #11, !dbg !15
  %1167 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1136, <4 x float> %1166) #11, !dbg !15
  %1168 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1138, <4 x float> %1167) #11, !dbg !15
  %1169 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1140, <4 x float> %1168) #11, !dbg !15
  %1170 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1142, <4 x float> %1169) #11, !dbg !15
  %1171 = shufflevector <8 x float> %1161, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1172 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1144, <4 x float> %1171) #11, !dbg !15
  %1173 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1147, <4 x float> %1172) #11, !dbg !15
  %1174 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1149, <4 x float> %1173) #11, !dbg !15
  %1175 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1151, <4 x float> %1174) #11, !dbg !15
  %1176 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1153, <4 x float> %1175) #11, !dbg !15
  %1177 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1155, <4 x float> %1176) #11, !dbg !15
  %1178 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1157, <4 x float> %1177) #11, !dbg !15
  %1179 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1159, <4 x float> %1178) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1180 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !15
  %1181 = bitcast <4 x i32> %1180 to <8 x half>, !dbg !15
  %1182 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !15
  %1183 = bitcast <4 x i32> %1182 to <8 x half>, !dbg !15
  %1184 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %1185 = bitcast <4 x i32> %1184 to <8 x half>, !dbg !15
  %1186 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %1187 = bitcast <4 x i32> %1186 to <8 x half>, !dbg !15
  %1188 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !15
  %1189 = bitcast <4 x i32> %1188 to <8 x half>, !dbg !15
  %1190 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !15
  %1191 = bitcast <4 x i32> %1190 to <8 x half>, !dbg !15
  %1192 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !15
  %1193 = bitcast <4 x i32> %1192 to <8 x half>, !dbg !15
  %1194 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !15
  %1195 = bitcast <4 x i32> %1194 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1196 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1095, !dbg !18
  %1197 = getelementptr i8, ptr addrspace(1) %1196, i64 128, !dbg !18
  %1198 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !18
  %1199 = zext i32 %1198 to i64, !dbg !18
  %1200 = inttoptr i64 %1199 to ptr addrspace(3), !dbg !18
  %address.local390 = getelementptr [2 x i8], ptr addrspace(1) %1197, i64 %address.local376, !dbg !18
  %1201 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1202 = ptrtoint ptr addrspace(1) %address.local390 to i64, !dbg !18
  %1203 = sub i64 %1202, %1201, !dbg !18
  %dma.buffer.offset635 = trunc i64 %1203 to i32, !dbg !18
  %1204 = trunc i64 %1201 to i32, !dbg !18
  %1205 = insertelement <4 x i32> poison, i32 %1204, i64 0, !dbg !18
  %1206 = lshr i64 %1201, 32, !dbg !18
  %1207 = trunc i64 %1206 to i32, !dbg !18
  %1208 = insertelement <4 x i32> %1205, i32 %1207, i64 1, !dbg !18
  %1209 = insertelement <4 x i32> %1208, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc636 = insertelement <4 x i32> %1209, i32 131072, i64 3, !dbg !18
  %1210 = ptrtoint ptr addrspace(3) %1200 to i32, !dbg !18
  %dma.buffer.lds637 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1210), !dbg !18
  %1211 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds637, i32 %dma.buffer.offset635, <4 x i32> %dma.buffer.rsrc636) #10, !dbg !18
  %1212 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !18
  %1213 = zext i32 %1212 to i64, !dbg !18
  %1214 = inttoptr i64 %1213 to ptr addrspace(3), !dbg !18
  %address.local391 = getelementptr [2 x i8], ptr addrspace(1) %1197, i64 %address.local388, !dbg !18
  %1215 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1216 = ptrtoint ptr addrspace(1) %address.local391 to i64, !dbg !18
  %1217 = sub i64 %1216, %1215, !dbg !18
  %dma.buffer.offset632 = trunc i64 %1217 to i32, !dbg !18
  %1218 = trunc i64 %1215 to i32, !dbg !18
  %1219 = insertelement <4 x i32> poison, i32 %1218, i64 0, !dbg !18
  %1220 = lshr i64 %1215, 32, !dbg !18
  %1221 = trunc i64 %1220 to i32, !dbg !18
  %1222 = insertelement <4 x i32> %1219, i32 %1221, i64 1, !dbg !18
  %1223 = insertelement <4 x i32> %1222, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc633 = insertelement <4 x i32> %1223, i32 131072, i64 3, !dbg !18
  %1224 = ptrtoint ptr addrspace(3) %1214 to i32, !dbg !18
  %dma.buffer.lds634 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1224), !dbg !18
  %1225 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds634, i32 %dma.buffer.offset632, <4 x i32> %dma.buffer.rsrc633) #10, !dbg !18
  %1226 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1227 = extractvalue [2 x <8 x float>] %29, 0, !dbg !29
  %1228 = shufflevector <8 x float> %1227, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1229 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1226, <4 x float> %1228) #11, !dbg !15
  %1230 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1231 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1230, <4 x float> %1229) #11, !dbg !15
  %1232 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1233 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1232, <4 x float> %1231) #11, !dbg !15
  %1234 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1235 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1234, <4 x float> %1233) #11, !dbg !15
  %1236 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1236, <4 x float> %1235) #11, !dbg !15
  %1238 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1239 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1238, <4 x float> %1237) #11, !dbg !15
  %1240 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1241 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1240, <4 x float> %1239) #11, !dbg !15
  %1242 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1243 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1242, <4 x float> %1241) #11, !dbg !15
  %1244 = shufflevector <8 x half> %1181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1245 = shufflevector <8 x float> %1227, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1244, <4 x float> %1245) #11, !dbg !15
  %1247 = shufflevector <8 x half> %1183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1248 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1247, <4 x float> %1246) #11, !dbg !15
  %1249 = shufflevector <8 x half> %1185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1249, <4 x float> %1248) #11, !dbg !15
  %1251 = shufflevector <8 x half> %1187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1252 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1251, <4 x float> %1250) #11, !dbg !15
  %1253 = shufflevector <8 x half> %1189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1254 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1253, <4 x float> %1252) #11, !dbg !15
  %1255 = shufflevector <8 x half> %1191, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1256 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1255, <4 x float> %1254) #11, !dbg !15
  %1257 = shufflevector <8 x half> %1193, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1258 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1257, <4 x float> %1256) #11, !dbg !15
  %1259 = shufflevector <8 x half> %1195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1260 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1259, <4 x float> %1258) #11, !dbg !15
  %1261 = extractvalue [2 x <8 x float>] %29, 1, !dbg !29
  %1262 = shufflevector <8 x float> %1261, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1263 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1226, <4 x float> %1262) #11, !dbg !15
  %1264 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1230, <4 x float> %1263) #11, !dbg !15
  %1265 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1232, <4 x float> %1264) #11, !dbg !15
  %1266 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1234, <4 x float> %1265) #11, !dbg !15
  %1267 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1236, <4 x float> %1266) #11, !dbg !15
  %1268 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1238, <4 x float> %1267) #11, !dbg !15
  %1269 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1240, <4 x float> %1268) #11, !dbg !15
  %1270 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1242, <4 x float> %1269) #11, !dbg !15
  %1271 = shufflevector <8 x float> %1261, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1272 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1244, <4 x float> %1271) #11, !dbg !15
  %1273 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1247, <4 x float> %1272) #11, !dbg !15
  %1274 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1249, <4 x float> %1273) #11, !dbg !15
  %1275 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1251, <4 x float> %1274) #11, !dbg !15
  %1276 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1253, <4 x float> %1275) #11, !dbg !15
  %1277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1255, <4 x float> %1276) #11, !dbg !15
  %1278 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1257, <4 x float> %1277) #11, !dbg !15
  %1279 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1259, <4 x float> %1278) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1280 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !15
  %1281 = bitcast <4 x i32> %1280 to <8 x half>, !dbg !15
  %1282 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !15
  %1283 = bitcast <4 x i32> %1282 to <8 x half>, !dbg !15
  %1284 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %1285 = bitcast <4 x i32> %1284 to <8 x half>, !dbg !15
  %1286 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %1287 = bitcast <4 x i32> %1286 to <8 x half>, !dbg !15
  %1288 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !15
  %1289 = bitcast <4 x i32> %1288 to <8 x half>, !dbg !15
  %1290 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !15
  %1291 = bitcast <4 x i32> %1290 to <8 x half>, !dbg !15
  %1292 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !15
  %1293 = bitcast <4 x i32> %1292 to <8 x half>, !dbg !15
  %1294 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !15
  %1295 = bitcast <4 x i32> %1294 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1296 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1095, !dbg !18
  %1297 = getelementptr i8, ptr addrspace(1) %1296, i64 192, !dbg !18
  %1298 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !18
  %1299 = zext i32 %1298 to i64, !dbg !18
  %1300 = inttoptr i64 %1299 to ptr addrspace(3), !dbg !18
  %address.local392 = getelementptr [2 x i8], ptr addrspace(1) %1297, i64 %address.local376, !dbg !18
  %1301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1302 = ptrtoint ptr addrspace(1) %address.local392 to i64, !dbg !18
  %1303 = sub i64 %1302, %1301, !dbg !18
  %dma.buffer.offset629 = trunc i64 %1303 to i32, !dbg !18
  %1304 = trunc i64 %1301 to i32, !dbg !18
  %1305 = insertelement <4 x i32> poison, i32 %1304, i64 0, !dbg !18
  %1306 = lshr i64 %1301, 32, !dbg !18
  %1307 = trunc i64 %1306 to i32, !dbg !18
  %1308 = insertelement <4 x i32> %1305, i32 %1307, i64 1, !dbg !18
  %1309 = insertelement <4 x i32> %1308, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc630 = insertelement <4 x i32> %1309, i32 131072, i64 3, !dbg !18
  %1310 = ptrtoint ptr addrspace(3) %1300 to i32, !dbg !18
  %dma.buffer.lds631 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1310), !dbg !18
  %1311 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds631, i32 %dma.buffer.offset629, <4 x i32> %dma.buffer.rsrc630) #10, !dbg !18
  %1312 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !18
  %1313 = zext i32 %1312 to i64, !dbg !18
  %1314 = inttoptr i64 %1313 to ptr addrspace(3), !dbg !18
  %address.local393 = getelementptr [2 x i8], ptr addrspace(1) %1297, i64 %address.local388, !dbg !18
  %1315 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1316 = ptrtoint ptr addrspace(1) %address.local393 to i64, !dbg !18
  %1317 = sub i64 %1316, %1315, !dbg !18
  %dma.buffer.offset626 = trunc i64 %1317 to i32, !dbg !18
  %1318 = trunc i64 %1315 to i32, !dbg !18
  %1319 = insertelement <4 x i32> poison, i32 %1318, i64 0, !dbg !18
  %1320 = lshr i64 %1315, 32, !dbg !18
  %1321 = trunc i64 %1320 to i32, !dbg !18
  %1322 = insertelement <4 x i32> %1319, i32 %1321, i64 1, !dbg !18
  %1323 = insertelement <4 x i32> %1322, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc627 = insertelement <4 x i32> %1323, i32 131072, i64 3, !dbg !18
  %1324 = ptrtoint ptr addrspace(3) %1314 to i32, !dbg !18
  %dma.buffer.lds628 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1324), !dbg !18
  %1325 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds628, i32 %dma.buffer.offset626, <4 x i32> %dma.buffer.rsrc627) #10, !dbg !18
  %1326 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1327 = extractvalue [2 x <8 x float>] %30, 0, !dbg !29
  %1328 = shufflevector <8 x float> %1327, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1329 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1326, <4 x float> %1328) #11, !dbg !15
  %1330 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1331 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1330, <4 x float> %1329) #11, !dbg !15
  %1332 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1333 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1332, <4 x float> %1331) #11, !dbg !15
  %1334 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1335 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1334, <4 x float> %1333) #11, !dbg !15
  %1336 = shufflevector <8 x half> %1289, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1337 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1336, <4 x float> %1335) #11, !dbg !15
  %1338 = shufflevector <8 x half> %1291, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1339 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1338, <4 x float> %1337) #11, !dbg !15
  %1340 = shufflevector <8 x half> %1293, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1341 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1340, <4 x float> %1339) #11, !dbg !15
  %1342 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1343 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1342, <4 x float> %1341) #11, !dbg !15
  %1344 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1345 = shufflevector <8 x float> %1327, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1346 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1344, <4 x float> %1345) #11, !dbg !15
  %1347 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1348 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1347, <4 x float> %1346) #11, !dbg !15
  %1349 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1350 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1349, <4 x float> %1348) #11, !dbg !15
  %1351 = shufflevector <8 x half> %1287, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1352 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1351, <4 x float> %1350) #11, !dbg !15
  %1353 = shufflevector <8 x half> %1289, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1354 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1353, <4 x float> %1352) #11, !dbg !15
  %1355 = shufflevector <8 x half> %1291, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1356 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1355, <4 x float> %1354) #11, !dbg !15
  %1357 = shufflevector <8 x half> %1293, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1358 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1357, <4 x float> %1356) #11, !dbg !15
  %1359 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1360 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1359, <4 x float> %1358) #11, !dbg !15
  %1361 = extractvalue [2 x <8 x float>] %30, 1, !dbg !29
  %1362 = shufflevector <8 x float> %1361, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1363 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1326, <4 x float> %1362) #11, !dbg !15
  %1364 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1330, <4 x float> %1363) #11, !dbg !15
  %1365 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1332, <4 x float> %1364) #11, !dbg !15
  %1366 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1334, <4 x float> %1365) #11, !dbg !15
  %1367 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1336, <4 x float> %1366) #11, !dbg !15
  %1368 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1338, <4 x float> %1367) #11, !dbg !15
  %1369 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1340, <4 x float> %1368) #11, !dbg !15
  %1370 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1342, <4 x float> %1369) #11, !dbg !15
  %1371 = shufflevector <8 x float> %1361, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1372 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1344, <4 x float> %1371) #11, !dbg !15
  %1373 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1347, <4 x float> %1372) #11, !dbg !15
  %1374 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1349, <4 x float> %1373) #11, !dbg !15
  %1375 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1351, <4 x float> %1374) #11, !dbg !15
  %1376 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1353, <4 x float> %1375) #11, !dbg !15
  %1377 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1355, <4 x float> %1376) #11, !dbg !15
  %1378 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1357, <4 x float> %1377) #11, !dbg !15
  %1379 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1359, <4 x float> %1378) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1380 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !15
  %1381 = bitcast <4 x i32> %1380 to <8 x half>, !dbg !15
  %1382 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !15
  %1383 = bitcast <4 x i32> %1382 to <8 x half>, !dbg !15
  %1384 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !15
  %1385 = bitcast <4 x i32> %1384 to <8 x half>, !dbg !15
  %1386 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !15
  %1387 = bitcast <4 x i32> %1386 to <8 x half>, !dbg !15
  %1388 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !15
  %1389 = bitcast <4 x i32> %1388 to <8 x half>, !dbg !15
  %1390 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !15
  %1391 = bitcast <4 x i32> %1390 to <8 x half>, !dbg !15
  %1392 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !15
  %1393 = bitcast <4 x i32> %1392 to <8 x half>, !dbg !15
  %1394 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !15
  %1395 = bitcast <4 x i32> %1394 to <8 x half>, !dbg !15
  %1396 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1397 = extractvalue [2 x <8 x float>] %31, 0, !dbg !29
  %1398 = shufflevector <8 x float> %1397, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1399 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1396, <4 x float> %1398) #11, !dbg !15
  %1400 = shufflevector <8 x half> %1383, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1401 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1400, <4 x float> %1399) #11, !dbg !15
  %1402 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1403 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1402, <4 x float> %1401) #11, !dbg !15
  %1404 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1405 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1404, <4 x float> %1403) #11, !dbg !15
  %1406 = shufflevector <8 x half> %1389, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1407 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1406, <4 x float> %1405) #11, !dbg !15
  %1408 = shufflevector <8 x half> %1391, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1409 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1408, <4 x float> %1407) #11, !dbg !15
  %1410 = shufflevector <8 x half> %1393, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1411 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1410, <4 x float> %1409) #11, !dbg !15
  %1412 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1413 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1412, <4 x float> %1411) #11, !dbg !15
  %1414 = shufflevector <8 x half> %1381, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1415 = shufflevector <8 x float> %1397, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1416 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1414, <4 x float> %1415) #11, !dbg !15
  %1417 = shufflevector <8 x half> %1383, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1418 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1417, <4 x float> %1416) #11, !dbg !15
  %1419 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1420 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1419, <4 x float> %1418) #11, !dbg !15
  %1421 = shufflevector <8 x half> %1387, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1422 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1421, <4 x float> %1420) #11, !dbg !15
  %1423 = shufflevector <8 x half> %1389, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1424 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1423, <4 x float> %1422) #11, !dbg !15
  %1425 = shufflevector <8 x half> %1391, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1426 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1425, <4 x float> %1424) #11, !dbg !15
  %1427 = shufflevector <8 x half> %1393, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1428 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1427, <4 x float> %1426) #11, !dbg !15
  %1429 = shufflevector <8 x half> %1395, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1430 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1429, <4 x float> %1428) #11, !dbg !15
  %1431 = extractvalue [2 x <8 x float>] %31, 1, !dbg !29
  %1432 = shufflevector <8 x float> %1431, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !29
  %1433 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1396, <4 x float> %1432) #11, !dbg !15
  %1434 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1400, <4 x float> %1433) #11, !dbg !15
  %1435 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1402, <4 x float> %1434) #11, !dbg !15
  %1436 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1404, <4 x float> %1435) #11, !dbg !15
  %1437 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1406, <4 x float> %1436) #11, !dbg !15
  %1438 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1408, <4 x float> %1437) #11, !dbg !15
  %1439 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1410, <4 x float> %1438) #11, !dbg !15
  %1440 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1412, <4 x float> %1439) #11, !dbg !15
  %1441 = shufflevector <8 x float> %1431, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !29
  %1442 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1414, <4 x float> %1441) #11, !dbg !15
  %1443 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1417, <4 x float> %1442) #11, !dbg !15
  %1444 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1419, <4 x float> %1443) #11, !dbg !15
  %1445 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1421, <4 x float> %1444) #11, !dbg !15
  %1446 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1423, <4 x float> %1445) #11, !dbg !15
  %1447 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1425, <4 x float> %1446) #11, !dbg !15
  %1448 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1427, <4 x float> %1447) #11, !dbg !15
  %1449 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1429, <4 x float> %1448) #11, !dbg !15
  %1450 = shufflevector <4 x float> %1143, <4 x float> %1160, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1451 = insertvalue [2 x <8 x float>] poison, <8 x float> %1450, 0, !dbg !16
  %1452 = shufflevector <4 x float> %1170, <4 x float> %1179, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1453 = insertvalue [2 x <8 x float>] %1451, <8 x float> %1452, 1, !dbg !16
  %1454 = shufflevector <4 x float> %1243, <4 x float> %1260, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1455 = insertvalue [2 x <8 x float>] poison, <8 x float> %1454, 0, !dbg !16
  %1456 = shufflevector <4 x float> %1270, <4 x float> %1279, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1457 = insertvalue [2 x <8 x float>] %1455, <8 x float> %1456, 1, !dbg !16
  %1458 = shufflevector <4 x float> %1343, <4 x float> %1360, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1459 = insertvalue [2 x <8 x float>] poison, <8 x float> %1458, 0, !dbg !16
  %1460 = shufflevector <4 x float> %1370, <4 x float> %1379, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1461 = insertvalue [2 x <8 x float>] %1459, <8 x float> %1460, 1, !dbg !16
  %1462 = shufflevector <4 x float> %1413, <4 x float> %1430, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1463 = insertvalue [2 x <8 x float>] poison, <8 x float> %1462, 0, !dbg !16
  %1464 = shufflevector <4 x float> %1440, <4 x float> %1449, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1465 = insertvalue [2 x <8 x float>] %1463, <8 x float> %1464, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1466 = add nuw nsw i64 %26, 1, !dbg !16
  br label %25, !dbg !16

1467:                                             ; preds = %25
  %1468 = and i32 %19, 16, !dbg !16
  %.not2 = icmp eq i32 %1468, 0, !dbg !16
  %1469 = and i32 %19, 32, !dbg !16
  %.not3 = icmp eq i32 %1469, 0, !dbg !16
  %1470 = extractvalue [2 x <8 x float>] %31, 0, !dbg !16
  %1471 = extractelement <8 x float> %1470, i64 0, !dbg !16
  %1472 = extractelement <8 x float> %1470, i64 1, !dbg !16
  %1473 = extractelement <8 x float> %1470, i64 2, !dbg !16
  %1474 = extractelement <8 x float> %1470, i64 3, !dbg !16
  %1475 = select i1 %.not2, float %1472, float %1471, !dbg !16
  %1476 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1477 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1476), !dbg !16
  %1478 = add nuw nsw i32 %1477, 64, !dbg !16
  %1479 = xor i32 %1477, 16, !dbg !16
  %1480 = and i32 %1478, 192, !dbg !16
  %1481 = icmp samesign ult i32 %1479, %1480, !dbg !16
  %1482 = select i1 %1481, i32 %1479, i32 %1477, !dbg !16
  %1483 = shl nuw nsw i32 %1482, 2, !dbg !16
  %1484 = bitcast float %1475 to i32, !dbg !16
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1484), !dbg !16
  %1486 = bitcast i32 %1485 to float, !dbg !16
  %1487 = select i1 %.not2, float %1471, float %1486, !dbg !16
  %1488 = select i1 %.not2, float %1486, float %1472, !dbg !16
  %1489 = select i1 %.not2, float %1474, float %1473, !dbg !16
  %1490 = bitcast float %1489 to i32, !dbg !16
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1490), !dbg !16
  %1492 = bitcast i32 %1491 to float, !dbg !16
  %1493 = select i1 %.not2, float %1473, float %1492, !dbg !16
  %1494 = select i1 %.not2, float %1492, float %1474, !dbg !16
  %1495 = select i1 %.not3, float %1493, float %1487, !dbg !16
  %1496 = xor i32 %1477, 32, !dbg !16
  %1497 = icmp samesign ult i32 %1496, %1480, !dbg !16
  %1498 = select i1 %1497, i32 %1496, i32 %1477, !dbg !16
  %1499 = shl nuw nsw i32 %1498, 2, !dbg !16
  %1500 = bitcast float %1495 to i32, !dbg !16
  %1501 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1500), !dbg !16
  %1502 = bitcast i32 %1501 to float, !dbg !16
  %1503 = select i1 %.not3, float %1487, float %1502, !dbg !16
  %1504 = select i1 %.not3, float %1502, float %1493, !dbg !16
  %1505 = select i1 %.not3, float %1494, float %1488, !dbg !16
  %1506 = bitcast float %1505 to i32, !dbg !16
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1506), !dbg !16
  %1508 = bitcast i32 %1507 to float, !dbg !16
  %1509 = select i1 %.not3, float %1488, float %1508, !dbg !16
  %1510 = select i1 %.not3, float %1508, float %1494, !dbg !16
  %1511 = extractvalue [2 x <8 x float>] %30, 0, !dbg !16
  %1512 = extractelement <8 x float> %1511, i64 0, !dbg !16
  %1513 = extractelement <8 x float> %1511, i64 1, !dbg !16
  %1514 = extractelement <8 x float> %1511, i64 2, !dbg !16
  %1515 = extractelement <8 x float> %1511, i64 3, !dbg !16
  %1516 = select i1 %.not2, float %1513, float %1512, !dbg !16
  %1517 = bitcast float %1516 to i32, !dbg !16
  %1518 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1517), !dbg !16
  %1519 = bitcast i32 %1518 to float, !dbg !16
  %1520 = select i1 %.not2, float %1512, float %1519, !dbg !16
  %1521 = select i1 %.not2, float %1519, float %1513, !dbg !16
  %1522 = select i1 %.not2, float %1515, float %1514, !dbg !16
  %1523 = bitcast float %1522 to i32, !dbg !16
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1523), !dbg !16
  %1525 = bitcast i32 %1524 to float, !dbg !16
  %1526 = select i1 %.not2, float %1514, float %1525, !dbg !16
  %1527 = select i1 %.not2, float %1525, float %1515, !dbg !16
  %1528 = select i1 %.not3, float %1526, float %1520, !dbg !16
  %1529 = bitcast float %1528 to i32, !dbg !16
  %1530 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1529), !dbg !16
  %1531 = bitcast i32 %1530 to float, !dbg !16
  %1532 = select i1 %.not3, float %1520, float %1531, !dbg !16
  %1533 = select i1 %.not3, float %1531, float %1526, !dbg !16
  %1534 = select i1 %.not3, float %1527, float %1521, !dbg !16
  %1535 = bitcast float %1534 to i32, !dbg !16
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1535), !dbg !16
  %1537 = bitcast i32 %1536 to float, !dbg !16
  %1538 = select i1 %.not3, float %1521, float %1537, !dbg !16
  %1539 = select i1 %.not3, float %1537, float %1527, !dbg !16
  %1540 = extractvalue [2 x <8 x float>] %29, 0, !dbg !16
  %1541 = extractelement <8 x float> %1540, i64 0, !dbg !16
  %1542 = extractelement <8 x float> %1540, i64 1, !dbg !16
  %1543 = extractelement <8 x float> %1540, i64 2, !dbg !16
  %1544 = extractelement <8 x float> %1540, i64 3, !dbg !16
  %1545 = select i1 %.not2, float %1542, float %1541, !dbg !16
  %1546 = bitcast float %1545 to i32, !dbg !16
  %1547 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1546), !dbg !16
  %1548 = bitcast i32 %1547 to float, !dbg !16
  %1549 = select i1 %.not2, float %1541, float %1548, !dbg !16
  %1550 = select i1 %.not2, float %1548, float %1542, !dbg !16
  %1551 = select i1 %.not2, float %1544, float %1543, !dbg !16
  %1552 = bitcast float %1551 to i32, !dbg !16
  %1553 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1552), !dbg !16
  %1554 = bitcast i32 %1553 to float, !dbg !16
  %1555 = select i1 %.not2, float %1543, float %1554, !dbg !16
  %1556 = select i1 %.not2, float %1554, float %1544, !dbg !16
  %1557 = select i1 %.not3, float %1555, float %1549, !dbg !16
  %1558 = bitcast float %1557 to i32, !dbg !16
  %1559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1558), !dbg !16
  %1560 = bitcast i32 %1559 to float, !dbg !16
  %1561 = select i1 %.not3, float %1549, float %1560, !dbg !16
  %1562 = select i1 %.not3, float %1560, float %1555, !dbg !16
  %1563 = select i1 %.not3, float %1556, float %1550, !dbg !16
  %1564 = bitcast float %1563 to i32, !dbg !16
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1564), !dbg !16
  %1566 = bitcast i32 %1565 to float, !dbg !16
  %1567 = select i1 %.not3, float %1550, float %1566, !dbg !16
  %1568 = select i1 %.not3, float %1566, float %1556, !dbg !16
  %1569 = extractvalue [2 x <8 x float>] %27, 0, !dbg !16
  %1570 = extractelement <8 x float> %1569, i64 0, !dbg !16
  %1571 = extractelement <8 x float> %1569, i64 1, !dbg !16
  %1572 = extractelement <8 x float> %1569, i64 2, !dbg !16
  %1573 = extractelement <8 x float> %1569, i64 3, !dbg !16
  %1574 = select i1 %.not2, float %1571, float %1570, !dbg !16
  %1575 = bitcast float %1574 to i32, !dbg !16
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1575), !dbg !16
  %1577 = bitcast i32 %1576 to float, !dbg !16
  %1578 = select i1 %.not2, float %1570, float %1577, !dbg !16
  %1579 = select i1 %.not2, float %1577, float %1571, !dbg !16
  %1580 = select i1 %.not2, float %1573, float %1572, !dbg !16
  %1581 = bitcast float %1580 to i32, !dbg !16
  %1582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1581), !dbg !16
  %1583 = bitcast i32 %1582 to float, !dbg !16
  %1584 = select i1 %.not2, float %1572, float %1583, !dbg !16
  %1585 = select i1 %.not2, float %1583, float %1573, !dbg !16
  %1586 = select i1 %.not3, float %1584, float %1578, !dbg !16
  %1587 = bitcast float %1586 to i32, !dbg !16
  %1588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1587), !dbg !16
  %1589 = select i1 %.not3, float %1585, float %1579, !dbg !16
  %1590 = bitcast float %1589 to i32, !dbg !16
  %1591 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1590), !dbg !16
  %1592 = extractvalue [2 x <1 x float>] %28, 0, !dbg !19
  %1593 = shufflevector <1 x float> %1592, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1594 = extractelement <8 x float> %1470, i64 4, !dbg !16
  %1595 = extractelement <8 x float> %1470, i64 5, !dbg !16
  %1596 = extractelement <8 x float> %1470, i64 6, !dbg !16
  %1597 = extractelement <8 x float> %1470, i64 7, !dbg !16
  %1598 = select i1 %.not2, float %1595, float %1594, !dbg !16
  %1599 = bitcast float %1598 to i32, !dbg !16
  %1600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1599), !dbg !16
  %1601 = bitcast i32 %1600 to float, !dbg !16
  %1602 = select i1 %.not2, float %1594, float %1601, !dbg !16
  %1603 = select i1 %.not2, float %1601, float %1595, !dbg !16
  %1604 = select i1 %.not2, float %1597, float %1596, !dbg !16
  %1605 = bitcast float %1604 to i32, !dbg !16
  %1606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1605), !dbg !16
  %1607 = bitcast i32 %1606 to float, !dbg !16
  %1608 = select i1 %.not2, float %1596, float %1607, !dbg !16
  %1609 = select i1 %.not2, float %1607, float %1597, !dbg !16
  %1610 = select i1 %.not3, float %1608, float %1602, !dbg !16
  %1611 = bitcast float %1610 to i32, !dbg !16
  %1612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1611), !dbg !16
  %1613 = bitcast i32 %1612 to float, !dbg !16
  %1614 = select i1 %.not3, float %1602, float %1613, !dbg !16
  %1615 = select i1 %.not3, float %1613, float %1608, !dbg !16
  %1616 = select i1 %.not3, float %1609, float %1603, !dbg !16
  %1617 = bitcast float %1616 to i32, !dbg !16
  %1618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1617), !dbg !16
  %1619 = bitcast i32 %1618 to float, !dbg !16
  %1620 = select i1 %.not3, float %1603, float %1619, !dbg !16
  %1621 = select i1 %.not3, float %1619, float %1609, !dbg !16
  %1622 = extractelement <8 x float> %1511, i64 4, !dbg !16
  %1623 = extractelement <8 x float> %1511, i64 5, !dbg !16
  %1624 = extractelement <8 x float> %1511, i64 6, !dbg !16
  %1625 = extractelement <8 x float> %1511, i64 7, !dbg !16
  %1626 = select i1 %.not2, float %1623, float %1622, !dbg !16
  %1627 = bitcast float %1626 to i32, !dbg !16
  %1628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1627), !dbg !16
  %1629 = bitcast i32 %1628 to float, !dbg !16
  %1630 = select i1 %.not2, float %1622, float %1629, !dbg !16
  %1631 = select i1 %.not2, float %1629, float %1623, !dbg !16
  %1632 = select i1 %.not2, float %1625, float %1624, !dbg !16
  %1633 = bitcast float %1632 to i32, !dbg !16
  %1634 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1633), !dbg !16
  %1635 = bitcast i32 %1634 to float, !dbg !16
  %1636 = select i1 %.not2, float %1624, float %1635, !dbg !16
  %1637 = select i1 %.not2, float %1635, float %1625, !dbg !16
  %1638 = select i1 %.not3, float %1636, float %1630, !dbg !16
  %1639 = bitcast float %1638 to i32, !dbg !16
  %1640 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1639), !dbg !16
  %1641 = bitcast i32 %1640 to float, !dbg !16
  %1642 = select i1 %.not3, float %1630, float %1641, !dbg !16
  %1643 = select i1 %.not3, float %1641, float %1636, !dbg !16
  %1644 = select i1 %.not3, float %1637, float %1631, !dbg !16
  %1645 = bitcast float %1644 to i32, !dbg !16
  %1646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1645), !dbg !16
  %1647 = bitcast i32 %1646 to float, !dbg !16
  %1648 = select i1 %.not3, float %1631, float %1647, !dbg !16
  %1649 = select i1 %.not3, float %1647, float %1637, !dbg !16
  %1650 = extractelement <8 x float> %1540, i64 4, !dbg !16
  %1651 = extractelement <8 x float> %1540, i64 5, !dbg !16
  %1652 = extractelement <8 x float> %1540, i64 6, !dbg !16
  %1653 = extractelement <8 x float> %1540, i64 7, !dbg !16
  %1654 = select i1 %.not2, float %1651, float %1650, !dbg !16
  %1655 = bitcast float %1654 to i32, !dbg !16
  %1656 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1655), !dbg !16
  %1657 = bitcast i32 %1656 to float, !dbg !16
  %1658 = select i1 %.not2, float %1650, float %1657, !dbg !16
  %1659 = select i1 %.not2, float %1657, float %1651, !dbg !16
  %1660 = select i1 %.not2, float %1653, float %1652, !dbg !16
  %1661 = bitcast float %1660 to i32, !dbg !16
  %1662 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1661), !dbg !16
  %1663 = bitcast i32 %1662 to float, !dbg !16
  %1664 = select i1 %.not2, float %1652, float %1663, !dbg !16
  %1665 = select i1 %.not2, float %1663, float %1653, !dbg !16
  %1666 = select i1 %.not3, float %1664, float %1658, !dbg !16
  %1667 = bitcast float %1666 to i32, !dbg !16
  %1668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1667), !dbg !16
  %1669 = bitcast i32 %1668 to float, !dbg !16
  %1670 = select i1 %.not3, float %1658, float %1669, !dbg !16
  %1671 = select i1 %.not3, float %1669, float %1664, !dbg !16
  %1672 = select i1 %.not3, float %1665, float %1659, !dbg !16
  %1673 = bitcast float %1672 to i32, !dbg !16
  %1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1673), !dbg !16
  %1675 = bitcast i32 %1674 to float, !dbg !16
  %1676 = select i1 %.not3, float %1659, float %1675, !dbg !16
  %1677 = select i1 %.not3, float %1675, float %1665, !dbg !16
  %1678 = extractelement <8 x float> %1569, i64 4, !dbg !16
  %1679 = extractelement <8 x float> %1569, i64 5, !dbg !16
  %1680 = extractelement <8 x float> %1569, i64 6, !dbg !16
  %1681 = extractelement <8 x float> %1569, i64 7, !dbg !16
  %1682 = select i1 %.not2, float %1679, float %1678, !dbg !16
  %1683 = bitcast float %1682 to i32, !dbg !16
  %1684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1683), !dbg !16
  %1685 = bitcast i32 %1684 to float, !dbg !16
  %1686 = select i1 %.not2, float %1678, float %1685, !dbg !16
  %1687 = select i1 %.not2, float %1685, float %1679, !dbg !16
  %1688 = select i1 %.not2, float %1681, float %1680, !dbg !16
  %1689 = bitcast float %1688 to i32, !dbg !16
  %1690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1689), !dbg !16
  %1691 = bitcast i32 %1690 to float, !dbg !16
  %1692 = select i1 %.not2, float %1680, float %1691, !dbg !16
  %1693 = select i1 %.not2, float %1691, float %1681, !dbg !16
  %1694 = select i1 %.not3, float %1692, float %1686, !dbg !16
  %1695 = bitcast float %1694 to i32, !dbg !16
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1695), !dbg !16
  %1697 = select i1 %.not3, float %1693, float %1687, !dbg !16
  %1698 = bitcast float %1697 to i32, !dbg !16
  %1699 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1698), !dbg !16
  %1700 = extractvalue [2 x <8 x float>] %31, 1, !dbg !16
  %1701 = extractelement <8 x float> %1700, i64 0, !dbg !16
  %1702 = extractelement <8 x float> %1700, i64 1, !dbg !16
  %1703 = extractelement <8 x float> %1700, i64 2, !dbg !16
  %1704 = extractelement <8 x float> %1700, i64 3, !dbg !16
  %1705 = select i1 %.not2, float %1702, float %1701, !dbg !16
  %1706 = bitcast float %1705 to i32, !dbg !16
  %1707 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1706), !dbg !16
  %1708 = bitcast i32 %1707 to float, !dbg !16
  %1709 = select i1 %.not2, float %1701, float %1708, !dbg !16
  %1710 = select i1 %.not2, float %1708, float %1702, !dbg !16
  %1711 = select i1 %.not2, float %1704, float %1703, !dbg !16
  %1712 = bitcast float %1711 to i32, !dbg !16
  %1713 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1712), !dbg !16
  %1714 = bitcast i32 %1713 to float, !dbg !16
  %1715 = select i1 %.not2, float %1703, float %1714, !dbg !16
  %1716 = select i1 %.not2, float %1714, float %1704, !dbg !16
  %1717 = select i1 %.not3, float %1715, float %1709, !dbg !16
  %1718 = bitcast float %1717 to i32, !dbg !16
  %1719 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1718), !dbg !16
  %1720 = bitcast i32 %1719 to float, !dbg !16
  %1721 = select i1 %.not3, float %1709, float %1720, !dbg !16
  %1722 = select i1 %.not3, float %1720, float %1715, !dbg !16
  %1723 = select i1 %.not3, float %1716, float %1710, !dbg !16
  %1724 = bitcast float %1723 to i32, !dbg !16
  %1725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1724), !dbg !16
  %1726 = bitcast i32 %1725 to float, !dbg !16
  %1727 = select i1 %.not3, float %1710, float %1726, !dbg !16
  %1728 = select i1 %.not3, float %1726, float %1716, !dbg !16
  %1729 = extractvalue [2 x <8 x float>] %30, 1, !dbg !16
  %1730 = extractelement <8 x float> %1729, i64 0, !dbg !16
  %1731 = extractelement <8 x float> %1729, i64 1, !dbg !16
  %1732 = extractelement <8 x float> %1729, i64 2, !dbg !16
  %1733 = extractelement <8 x float> %1729, i64 3, !dbg !16
  %1734 = select i1 %.not2, float %1731, float %1730, !dbg !16
  %1735 = bitcast float %1734 to i32, !dbg !16
  %1736 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1735), !dbg !16
  %1737 = bitcast i32 %1736 to float, !dbg !16
  %1738 = select i1 %.not2, float %1730, float %1737, !dbg !16
  %1739 = select i1 %.not2, float %1737, float %1731, !dbg !16
  %1740 = select i1 %.not2, float %1733, float %1732, !dbg !16
  %1741 = bitcast float %1740 to i32, !dbg !16
  %1742 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1741), !dbg !16
  %1743 = bitcast i32 %1742 to float, !dbg !16
  %1744 = select i1 %.not2, float %1732, float %1743, !dbg !16
  %1745 = select i1 %.not2, float %1743, float %1733, !dbg !16
  %1746 = select i1 %.not3, float %1744, float %1738, !dbg !16
  %1747 = bitcast float %1746 to i32, !dbg !16
  %1748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1747), !dbg !16
  %1749 = bitcast i32 %1748 to float, !dbg !16
  %1750 = select i1 %.not3, float %1738, float %1749, !dbg !16
  %1751 = select i1 %.not3, float %1749, float %1744, !dbg !16
  %1752 = select i1 %.not3, float %1745, float %1739, !dbg !16
  %1753 = bitcast float %1752 to i32, !dbg !16
  %1754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1753), !dbg !16
  %1755 = bitcast i32 %1754 to float, !dbg !16
  %1756 = select i1 %.not3, float %1739, float %1755, !dbg !16
  %1757 = select i1 %.not3, float %1755, float %1745, !dbg !16
  %1758 = extractvalue [2 x <8 x float>] %29, 1, !dbg !16
  %1759 = extractelement <8 x float> %1758, i64 0, !dbg !16
  %1760 = extractelement <8 x float> %1758, i64 1, !dbg !16
  %1761 = extractelement <8 x float> %1758, i64 2, !dbg !16
  %1762 = extractelement <8 x float> %1758, i64 3, !dbg !16
  %1763 = select i1 %.not2, float %1760, float %1759, !dbg !16
  %1764 = bitcast float %1763 to i32, !dbg !16
  %1765 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1764), !dbg !16
  %1766 = bitcast i32 %1765 to float, !dbg !16
  %1767 = select i1 %.not2, float %1759, float %1766, !dbg !16
  %1768 = select i1 %.not2, float %1766, float %1760, !dbg !16
  %1769 = select i1 %.not2, float %1762, float %1761, !dbg !16
  %1770 = bitcast float %1769 to i32, !dbg !16
  %1771 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1770), !dbg !16
  %1772 = bitcast i32 %1771 to float, !dbg !16
  %1773 = select i1 %.not2, float %1761, float %1772, !dbg !16
  %1774 = select i1 %.not2, float %1772, float %1762, !dbg !16
  %1775 = select i1 %.not3, float %1773, float %1767, !dbg !16
  %1776 = bitcast float %1775 to i32, !dbg !16
  %1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1776), !dbg !16
  %1778 = bitcast i32 %1777 to float, !dbg !16
  %1779 = select i1 %.not3, float %1767, float %1778, !dbg !16
  %1780 = select i1 %.not3, float %1778, float %1773, !dbg !16
  %1781 = select i1 %.not3, float %1774, float %1768, !dbg !16
  %1782 = bitcast float %1781 to i32, !dbg !16
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1782), !dbg !16
  %1784 = bitcast i32 %1783 to float, !dbg !16
  %1785 = select i1 %.not3, float %1768, float %1784, !dbg !16
  %1786 = select i1 %.not3, float %1784, float %1774, !dbg !16
  %1787 = extractvalue [2 x <8 x float>] %27, 1, !dbg !16
  %1788 = extractelement <8 x float> %1787, i64 0, !dbg !16
  %1789 = extractelement <8 x float> %1787, i64 1, !dbg !16
  %1790 = extractelement <8 x float> %1787, i64 2, !dbg !16
  %1791 = extractelement <8 x float> %1787, i64 3, !dbg !16
  %1792 = select i1 %.not2, float %1789, float %1788, !dbg !16
  %1793 = bitcast float %1792 to i32, !dbg !16
  %1794 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1793), !dbg !16
  %1795 = bitcast i32 %1794 to float, !dbg !16
  %1796 = select i1 %.not2, float %1788, float %1795, !dbg !16
  %1797 = select i1 %.not2, float %1795, float %1789, !dbg !16
  %1798 = select i1 %.not2, float %1791, float %1790, !dbg !16
  %1799 = bitcast float %1798 to i32, !dbg !16
  %1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1799), !dbg !16
  %1801 = bitcast i32 %1800 to float, !dbg !16
  %1802 = select i1 %.not2, float %1790, float %1801, !dbg !16
  %1803 = select i1 %.not2, float %1801, float %1791, !dbg !16
  %1804 = select i1 %.not3, float %1802, float %1796, !dbg !16
  %1805 = bitcast float %1804 to i32, !dbg !16
  %1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1805), !dbg !16
  %1807 = select i1 %.not3, float %1803, float %1797, !dbg !16
  %1808 = bitcast float %1807 to i32, !dbg !16
  %1809 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1808), !dbg !16
  %1810 = extractvalue [2 x <1 x float>] %28, 1, !dbg !19
  %1811 = shufflevector <1 x float> %1810, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1812 = extractelement <8 x float> %1700, i64 4, !dbg !16
  %1813 = extractelement <8 x float> %1700, i64 5, !dbg !16
  %1814 = extractelement <8 x float> %1700, i64 6, !dbg !16
  %1815 = extractelement <8 x float> %1700, i64 7, !dbg !16
  %1816 = select i1 %.not2, float %1813, float %1812, !dbg !16
  %1817 = bitcast float %1816 to i32, !dbg !16
  %1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1817), !dbg !16
  %1819 = bitcast i32 %1818 to float, !dbg !16
  %1820 = select i1 %.not2, float %1812, float %1819, !dbg !16
  %1821 = select i1 %.not2, float %1819, float %1813, !dbg !16
  %1822 = select i1 %.not2, float %1815, float %1814, !dbg !16
  %1823 = bitcast float %1822 to i32, !dbg !16
  %1824 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1823), !dbg !16
  %1825 = bitcast i32 %1824 to float, !dbg !16
  %1826 = select i1 %.not2, float %1814, float %1825, !dbg !16
  %1827 = select i1 %.not2, float %1825, float %1815, !dbg !16
  %1828 = select i1 %.not3, float %1826, float %1820, !dbg !16
  %1829 = bitcast float %1828 to i32, !dbg !16
  %1830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1829), !dbg !16
  %1831 = bitcast i32 %1830 to float, !dbg !16
  %1832 = select i1 %.not3, float %1820, float %1831, !dbg !16
  %1833 = select i1 %.not3, float %1831, float %1826, !dbg !16
  %1834 = select i1 %.not3, float %1827, float %1821, !dbg !16
  %1835 = bitcast float %1834 to i32, !dbg !16
  %1836 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1835), !dbg !16
  %1837 = bitcast i32 %1836 to float, !dbg !16
  %1838 = select i1 %.not3, float %1821, float %1837, !dbg !16
  %1839 = select i1 %.not3, float %1837, float %1827, !dbg !16
  %1840 = extractelement <8 x float> %1729, i64 4, !dbg !16
  %1841 = extractelement <8 x float> %1729, i64 5, !dbg !16
  %1842 = extractelement <8 x float> %1729, i64 6, !dbg !16
  %1843 = extractelement <8 x float> %1729, i64 7, !dbg !16
  %1844 = select i1 %.not2, float %1841, float %1840, !dbg !16
  %1845 = bitcast float %1844 to i32, !dbg !16
  %1846 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1845), !dbg !16
  %1847 = bitcast i32 %1846 to float, !dbg !16
  %1848 = select i1 %.not2, float %1840, float %1847, !dbg !16
  %1849 = select i1 %.not2, float %1847, float %1841, !dbg !16
  %1850 = select i1 %.not2, float %1843, float %1842, !dbg !16
  %1851 = bitcast float %1850 to i32, !dbg !16
  %1852 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1851), !dbg !16
  %1853 = bitcast i32 %1852 to float, !dbg !16
  %1854 = select i1 %.not2, float %1842, float %1853, !dbg !16
  %1855 = select i1 %.not2, float %1853, float %1843, !dbg !16
  %1856 = select i1 %.not3, float %1854, float %1848, !dbg !16
  %1857 = bitcast float %1856 to i32, !dbg !16
  %1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1857), !dbg !16
  %1859 = bitcast i32 %1858 to float, !dbg !16
  %1860 = select i1 %.not3, float %1848, float %1859, !dbg !16
  %1861 = select i1 %.not3, float %1859, float %1854, !dbg !16
  %1862 = select i1 %.not3, float %1855, float %1849, !dbg !16
  %1863 = bitcast float %1862 to i32, !dbg !16
  %1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1863), !dbg !16
  %1865 = bitcast i32 %1864 to float, !dbg !16
  %1866 = select i1 %.not3, float %1849, float %1865, !dbg !16
  %1867 = select i1 %.not3, float %1865, float %1855, !dbg !16
  %1868 = extractelement <8 x float> %1758, i64 4, !dbg !16
  %1869 = extractelement <8 x float> %1758, i64 5, !dbg !16
  %1870 = extractelement <8 x float> %1758, i64 6, !dbg !16
  %1871 = extractelement <8 x float> %1758, i64 7, !dbg !16
  %1872 = select i1 %.not2, float %1869, float %1868, !dbg !16
  %1873 = bitcast float %1872 to i32, !dbg !16
  %1874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1873), !dbg !16
  %1875 = bitcast i32 %1874 to float, !dbg !16
  %1876 = select i1 %.not2, float %1868, float %1875, !dbg !16
  %1877 = select i1 %.not2, float %1875, float %1869, !dbg !16
  %1878 = select i1 %.not2, float %1871, float %1870, !dbg !16
  %1879 = bitcast float %1878 to i32, !dbg !16
  %1880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1879), !dbg !16
  %1881 = bitcast i32 %1880 to float, !dbg !16
  %1882 = select i1 %.not2, float %1870, float %1881, !dbg !16
  %1883 = select i1 %.not2, float %1881, float %1871, !dbg !16
  %1884 = select i1 %.not3, float %1882, float %1876, !dbg !16
  %1885 = bitcast float %1884 to i32, !dbg !16
  %1886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1885), !dbg !16
  %1887 = bitcast i32 %1886 to float, !dbg !16
  %1888 = select i1 %.not3, float %1876, float %1887, !dbg !16
  %1889 = select i1 %.not3, float %1887, float %1882, !dbg !16
  %1890 = select i1 %.not3, float %1883, float %1877, !dbg !16
  %1891 = bitcast float %1890 to i32, !dbg !16
  %1892 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1891), !dbg !16
  %1893 = bitcast i32 %1892 to float, !dbg !16
  %1894 = select i1 %.not3, float %1877, float %1893, !dbg !16
  %1895 = select i1 %.not3, float %1893, float %1883, !dbg !16
  %1896 = extractelement <8 x float> %1787, i64 4, !dbg !16
  %1897 = extractelement <8 x float> %1787, i64 5, !dbg !16
  %1898 = extractelement <8 x float> %1787, i64 6, !dbg !16
  %1899 = extractelement <8 x float> %1787, i64 7, !dbg !16
  %1900 = select i1 %.not2, float %1897, float %1896, !dbg !16
  %1901 = bitcast float %1900 to i32, !dbg !16
  %1902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1901), !dbg !16
  %1903 = bitcast i32 %1902 to float, !dbg !16
  %1904 = select i1 %.not2, float %1896, float %1903, !dbg !16
  %1905 = select i1 %.not2, float %1903, float %1897, !dbg !16
  %1906 = select i1 %.not2, float %1899, float %1898, !dbg !16
  %1907 = bitcast float %1906 to i32, !dbg !16
  %1908 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1483, i32 %1907), !dbg !16
  %1909 = bitcast i32 %1908 to float, !dbg !16
  %1910 = select i1 %.not2, float %1898, float %1909, !dbg !16
  %1911 = select i1 %.not2, float %1909, float %1899, !dbg !16
  %1912 = select i1 %.not3, float %1910, float %1904, !dbg !16
  %1913 = bitcast float %1912 to i32, !dbg !16
  %1914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1913), !dbg !16
  %1915 = select i1 %.not3, float %1911, float %1905, !dbg !16
  %1916 = bitcast float %1915 to i32, !dbg !16
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1499, i32 %1916), !dbg !16
  br i1 true, label %1918, label %1982, !dbg !21

1918:                                             ; preds = %1467
  %1919 = bitcast i32 %1914 to float, !dbg !16
  %1920 = select i1 %.not3, float %1904, float %1919, !dbg !16
  %1921 = insertelement <4 x float> poison, float %1920, i64 0, !dbg !19
  %1922 = bitcast i32 %1917 to float, !dbg !16
  %1923 = select i1 %.not3, float %1905, float %1922, !dbg !16
  %1924 = insertelement <4 x float> %1921, float %1923, i64 1, !dbg !19
  %1925 = select i1 %.not3, float %1919, float %1910, !dbg !16
  %1926 = insertelement <4 x float> %1924, float %1925, i64 2, !dbg !19
  %1927 = select i1 %.not3, float %1922, float %1911, !dbg !16
  %1928 = insertelement <4 x float> %1926, float %1927, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1811, !dbg !19
  %normalized = fmul <4 x float> %1928, %row.reciprocal, !dbg !19
  %1929 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1930 = extractelement <4 x half> %1929, i64 3, !dbg !20
  %1931 = extractelement <4 x half> %1929, i64 2, !dbg !20
  %1932 = extractelement <4 x half> %1929, i64 1, !dbg !20
  %1933 = extractelement <4 x half> %1929, i64 0, !dbg !20
  %1934 = bitcast i32 %1806 to float, !dbg !16
  %1935 = select i1 %.not3, float %1796, float %1934, !dbg !16
  %1936 = insertelement <4 x float> poison, float %1935, i64 0, !dbg !19
  %1937 = bitcast i32 %1809 to float, !dbg !16
  %1938 = select i1 %.not3, float %1797, float %1937, !dbg !16
  %1939 = insertelement <4 x float> %1936, float %1938, i64 1, !dbg !19
  %1940 = select i1 %.not3, float %1934, float %1802, !dbg !16
  %1941 = insertelement <4 x float> %1939, float %1940, i64 2, !dbg !19
  %1942 = select i1 %.not3, float %1937, float %1803, !dbg !16
  %1943 = insertelement <4 x float> %1941, float %1942, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1943, %row.reciprocal, !dbg !19
  %1944 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1945 = extractelement <4 x half> %1944, i64 3, !dbg !20
  %1946 = extractelement <4 x half> %1944, i64 2, !dbg !20
  %1947 = extractelement <4 x half> %1944, i64 1, !dbg !20
  %1948 = extractelement <4 x half> %1944, i64 0, !dbg !20
  %1949 = bitcast i32 %1696 to float, !dbg !16
  %1950 = select i1 %.not3, float %1686, float %1949, !dbg !16
  %1951 = insertelement <4 x float> poison, float %1950, i64 0, !dbg !19
  %1952 = bitcast i32 %1699 to float, !dbg !16
  %1953 = select i1 %.not3, float %1687, float %1952, !dbg !16
  %1954 = insertelement <4 x float> %1951, float %1953, i64 1, !dbg !19
  %1955 = select i1 %.not3, float %1949, float %1692, !dbg !16
  %1956 = insertelement <4 x float> %1954, float %1955, i64 2, !dbg !19
  %1957 = select i1 %.not3, float %1952, float %1693, !dbg !16
  %1958 = insertelement <4 x float> %1956, float %1957, i64 3, !dbg !19
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1593, !dbg !19
  %normalized77 = fmul <4 x float> %1958, %row.reciprocal76, !dbg !19
  %1959 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1960 = extractelement <4 x half> %1959, i64 3, !dbg !20
  %1961 = extractelement <4 x half> %1959, i64 2, !dbg !20
  %1962 = extractelement <4 x half> %1959, i64 1, !dbg !20
  %1963 = extractelement <4 x half> %1959, i64 0, !dbg !20
  %1964 = bitcast i32 %1588 to float, !dbg !16
  %1965 = select i1 %.not3, float %1578, float %1964, !dbg !16
  %1966 = insertelement <4 x float> poison, float %1965, i64 0, !dbg !19
  %1967 = bitcast i32 %1591 to float, !dbg !16
  %1968 = select i1 %.not3, float %1579, float %1967, !dbg !16
  %1969 = insertelement <4 x float> %1966, float %1968, i64 1, !dbg !19
  %1970 = select i1 %.not3, float %1964, float %1584, !dbg !16
  %1971 = insertelement <4 x float> %1969, float %1970, i64 2, !dbg !19
  %1972 = select i1 %.not3, float %1967, float %1585, !dbg !16
  %1973 = insertelement <4 x float> %1971, float %1972, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1973, %row.reciprocal76, !dbg !19
  %1974 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1975 = extractelement <4 x half> %1974, i64 3, !dbg !20
  %1976 = extractelement <4 x half> %1974, i64 2, !dbg !20
  %1977 = extractelement <4 x half> %1974, i64 1, !dbg !20
  %1978 = extractelement <4 x half> %1974, i64 0, !dbg !20
  %1979 = add nsw i64 %8, %13, !dbg !21
  %1980 = shl nsw i64 %1979, 19, !dbg !21
  %1981 = and i64 %1980, 16252928, !dbg !21
  %address.thread394 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local395 = zext nneg i32 %address.thread394 to i64, !dbg !13
  %address.local396 = add nsw i64 %16, %address.local395, !dbg !21
  %address.local397 = lshr i64 %address.local395, 6, !dbg !13
  %address.local398 = shl nuw nsw i64 %address.local397, 4, !dbg !21
  %address.local399 = add nsw i64 %address.local396, %address.local398, !dbg !21
  %address.local400 = and i64 %address.local395, 240, !dbg !21
  %address.local401 = sub nsw i64 %address.local399, %address.local400, !dbg !21
  %address.local402 = add nsw i64 %address.local401, %17, !dbg !21
  %address.local403 = shl nuw nsw i64 %address.local402, 7, !dbg !21
  %address.local404 = add nuw nsw i64 %1981, %address.local403, !dbg !21
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local406 = lshr i64 %address.local395, 2, !dbg !13
  %address.local407 = and i64 %address.local406, 12, !dbg !13
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local407, !dbg !21
  store half %1978, ptr addrspace(1) %address.local408, align 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local410 = or disjoint i64 %address.local407, 1, !dbg !21
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !21
  store half %1977, ptr addrspace(1) %address.local411, align 2, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local413 = or disjoint i64 %address.local407, 2, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local413, !dbg !21
  store half %1976, ptr addrspace(1) %address.local414, align 2, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local416 = or disjoint i64 %address.local407, 3, !dbg !21
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local416, !dbg !21
  store half %1975, ptr addrspace(1) %address.local417, align 2, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local419 = or disjoint i64 %address.local407, 16, !dbg !21
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local419, !dbg !21
  store half %1963, ptr addrspace(1) %address.local420, align 2, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local422 = or disjoint i64 %address.local407, 17, !dbg !21
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local422, !dbg !21
  store half %1962, ptr addrspace(1) %address.local423, align 2, !dbg !21
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local425 = or disjoint i64 %address.local407, 18, !dbg !21
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local425, !dbg !21
  store half %1961, ptr addrspace(1) %address.local426, align 2, !dbg !21
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !21
  %address.local428 = or disjoint i64 %address.local407, 19, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local428, !dbg !21
  store half %1960, ptr addrspace(1) %address.local429, align 2, !dbg !21
  %address.local430 = shl nsw i64 %address.local402, 7, !dbg !21
  %address.local431 = add i64 %address.local430, 8192, !dbg !21
  %address.local432 = add nuw nsw i64 %1981, %address.local431, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local407, !dbg !21
  store half %1948, ptr addrspace(1) %address.local434, align 2, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local410, !dbg !21
  store half %1947, ptr addrspace(1) %address.local436, align 2, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local413, !dbg !21
  store half %1946, ptr addrspace(1) %address.local438, align 2, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local416, !dbg !21
  store half %1945, ptr addrspace(1) %address.local440, align 2, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local419, !dbg !21
  store half %1933, ptr addrspace(1) %address.local442, align 2, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local422, !dbg !21
  store half %1932, ptr addrspace(1) %address.local444, align 2, !dbg !21
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local425, !dbg !21
  store half %1931, ptr addrspace(1) %address.local446, align 2, !dbg !21
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !21
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local447, i64 %address.local428, !dbg !21
  store half %1930, ptr addrspace(1) %address.local448, align 2, !dbg !21
  br label %1982, !dbg !21

1982:                                             ; preds = %1918, %1467
  br i1 true, label %1983, label %2021, !dbg !21

1983:                                             ; preds = %1982
  %1984 = insertelement <4 x float> poison, float %1888, i64 0, !dbg !19
  %1985 = insertelement <4 x float> %1984, float %1894, i64 1, !dbg !19
  %1986 = insertelement <4 x float> %1985, float %1889, i64 2, !dbg !19
  %1987 = insertelement <4 x float> %1986, float %1895, i64 3, !dbg !19
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1811, !dbg !19
  %normalized80 = fmul <4 x float> %1987, %row.reciprocal79, !dbg !19
  %1988 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %1989 = extractelement <4 x half> %1988, i64 3, !dbg !20
  %1990 = extractelement <4 x half> %1988, i64 2, !dbg !20
  %1991 = extractelement <4 x half> %1988, i64 1, !dbg !20
  %1992 = extractelement <4 x half> %1988, i64 0, !dbg !20
  %1993 = insertelement <4 x float> poison, float %1779, i64 0, !dbg !19
  %1994 = insertelement <4 x float> %1993, float %1785, i64 1, !dbg !19
  %1995 = insertelement <4 x float> %1994, float %1780, i64 2, !dbg !19
  %1996 = insertelement <4 x float> %1995, float %1786, i64 3, !dbg !19
  %normalized81 = fmul <4 x float> %1996, %row.reciprocal79, !dbg !19
  %1997 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1998 = extractelement <4 x half> %1997, i64 3, !dbg !20
  %1999 = extractelement <4 x half> %1997, i64 2, !dbg !20
  %2000 = extractelement <4 x half> %1997, i64 1, !dbg !20
  %2001 = extractelement <4 x half> %1997, i64 0, !dbg !20
  %2002 = insertelement <4 x float> poison, float %1670, i64 0, !dbg !19
  %2003 = insertelement <4 x float> %2002, float %1676, i64 1, !dbg !19
  %2004 = insertelement <4 x float> %2003, float %1671, i64 2, !dbg !19
  %2005 = insertelement <4 x float> %2004, float %1677, i64 3, !dbg !19
  %row.reciprocal82 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1593, !dbg !19
  %normalized83 = fmul <4 x float> %2005, %row.reciprocal82, !dbg !19
  %2006 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %2007 = extractelement <4 x half> %2006, i64 3, !dbg !20
  %2008 = extractelement <4 x half> %2006, i64 2, !dbg !20
  %2009 = extractelement <4 x half> %2006, i64 1, !dbg !20
  %2010 = extractelement <4 x half> %2006, i64 0, !dbg !20
  %2011 = insertelement <4 x float> poison, float %1561, i64 0, !dbg !19
  %2012 = insertelement <4 x float> %2011, float %1567, i64 1, !dbg !19
  %2013 = insertelement <4 x float> %2012, float %1562, i64 2, !dbg !19
  %2014 = insertelement <4 x float> %2013, float %1568, i64 3, !dbg !19
  %normalized84 = fmul <4 x float> %2014, %row.reciprocal82, !dbg !19
  %2015 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !20
  %2016 = extractelement <4 x half> %2015, i64 3, !dbg !20
  %2017 = extractelement <4 x half> %2015, i64 2, !dbg !20
  %2018 = extractelement <4 x half> %2015, i64 1, !dbg !20
  %2019 = extractelement <4 x half> %2015, i64 0, !dbg !20
  %2020 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread449 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local450 = zext nneg i32 %address.thread449 to i64, !dbg !13
  %address.local451 = add nsw i64 %16, %address.local450, !dbg !21
  %address.local452 = lshr i64 %address.local450, 6, !dbg !13
  %address.local453 = shl nuw nsw i64 %address.local452, 4, !dbg !21
  %address.local454 = add nsw i64 %address.local451, %address.local453, !dbg !21
  %address.local455 = and i64 %address.local450, 240, !dbg !21
  %address.local456 = sub nsw i64 %address.local454, %address.local455, !dbg !21
  %address.local457 = add nsw i64 %address.local456, %17, !dbg !21
  %address.local458 = shl nuw nsw i64 %address.local457, 7, !dbg !21
  %address.local459 = add nuw nsw i64 %2020, %address.local458, !dbg !21
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local461 = lshr i64 %address.local450, 2, !dbg !13
  %address.local462 = and i64 %address.local461, 12, !dbg !13
  %address.local463 = or disjoint i64 %address.local462, 32, !dbg !21
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local463, !dbg !21
  store half %2019, ptr addrspace(1) %address.local464, align 2, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local466 = or disjoint i64 %address.local462, 33, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local466, !dbg !21
  store half %2018, ptr addrspace(1) %address.local467, align 2, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local469 = or disjoint i64 %address.local462, 34, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local469, !dbg !21
  store half %2017, ptr addrspace(1) %address.local470, align 2, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local472 = or disjoint i64 %address.local462, 35, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local472, !dbg !21
  store half %2016, ptr addrspace(1) %address.local473, align 2, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local475 = or disjoint i64 %address.local462, 48, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local474, i64 %address.local475, !dbg !21
  store half %2010, ptr addrspace(1) %address.local476, align 2, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local478 = or disjoint i64 %address.local462, 49, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local478, !dbg !21
  store half %2009, ptr addrspace(1) %address.local479, align 2, !dbg !21
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local481 = or disjoint i64 %address.local462, 50, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local480, i64 %address.local481, !dbg !21
  store half %2008, ptr addrspace(1) %address.local482, align 2, !dbg !21
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !21
  %address.local484 = or i64 %address.local461, 51, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local483, i64 %address.local484, !dbg !21
  store half %2007, ptr addrspace(1) %address.local485, align 2, !dbg !21
  %address.local486 = shl nsw i64 %address.local457, 7, !dbg !21
  %address.local487 = add i64 %address.local486, 8192, !dbg !21
  %address.local488 = add nuw nsw i64 %2020, %address.local487, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local463, !dbg !21
  store half %2001, ptr addrspace(1) %address.local490, align 2, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local466, !dbg !21
  store half %2000, ptr addrspace(1) %address.local492, align 2, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local469, !dbg !21
  store half %1999, ptr addrspace(1) %address.local494, align 2, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local472, !dbg !21
  store half %1998, ptr addrspace(1) %address.local496, align 2, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local475, !dbg !21
  store half %1992, ptr addrspace(1) %address.local498, align 2, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local478, !dbg !21
  store half %1991, ptr addrspace(1) %address.local500, align 2, !dbg !21
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local481, !dbg !21
  store half %1990, ptr addrspace(1) %address.local502, align 2, !dbg !21
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !21
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local503, i64 %address.local484, !dbg !21
  store half %1989, ptr addrspace(1) %address.local504, align 2, !dbg !21
  br label %2021, !dbg !21

2021:                                             ; preds = %1983, %1982
  br i1 true, label %2022, label %2060, !dbg !21

2022:                                             ; preds = %2021
  %2023 = insertelement <4 x float> poison, float %1860, i64 0, !dbg !19
  %2024 = insertelement <4 x float> %2023, float %1866, i64 1, !dbg !19
  %2025 = insertelement <4 x float> %2024, float %1861, i64 2, !dbg !19
  %2026 = insertelement <4 x float> %2025, float %1867, i64 3, !dbg !19
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1811, !dbg !19
  %normalized86 = fmul <4 x float> %2026, %row.reciprocal85, !dbg !19
  %2027 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %2028 = extractelement <4 x half> %2027, i64 3, !dbg !20
  %2029 = extractelement <4 x half> %2027, i64 2, !dbg !20
  %2030 = extractelement <4 x half> %2027, i64 1, !dbg !20
  %2031 = extractelement <4 x half> %2027, i64 0, !dbg !20
  %2032 = insertelement <4 x float> poison, float %1750, i64 0, !dbg !19
  %2033 = insertelement <4 x float> %2032, float %1756, i64 1, !dbg !19
  %2034 = insertelement <4 x float> %2033, float %1751, i64 2, !dbg !19
  %2035 = insertelement <4 x float> %2034, float %1757, i64 3, !dbg !19
  %normalized87 = fmul <4 x float> %2035, %row.reciprocal85, !dbg !19
  %2036 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !20
  %2037 = extractelement <4 x half> %2036, i64 3, !dbg !20
  %2038 = extractelement <4 x half> %2036, i64 2, !dbg !20
  %2039 = extractelement <4 x half> %2036, i64 1, !dbg !20
  %2040 = extractelement <4 x half> %2036, i64 0, !dbg !20
  %2041 = insertelement <4 x float> poison, float %1642, i64 0, !dbg !19
  %2042 = insertelement <4 x float> %2041, float %1648, i64 1, !dbg !19
  %2043 = insertelement <4 x float> %2042, float %1643, i64 2, !dbg !19
  %2044 = insertelement <4 x float> %2043, float %1649, i64 3, !dbg !19
  %row.reciprocal88 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1593, !dbg !19
  %normalized89 = fmul <4 x float> %2044, %row.reciprocal88, !dbg !19
  %2045 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %2046 = extractelement <4 x half> %2045, i64 3, !dbg !20
  %2047 = extractelement <4 x half> %2045, i64 2, !dbg !20
  %2048 = extractelement <4 x half> %2045, i64 1, !dbg !20
  %2049 = extractelement <4 x half> %2045, i64 0, !dbg !20
  %2050 = insertelement <4 x float> poison, float %1532, i64 0, !dbg !19
  %2051 = insertelement <4 x float> %2050, float %1538, i64 1, !dbg !19
  %2052 = insertelement <4 x float> %2051, float %1533, i64 2, !dbg !19
  %2053 = insertelement <4 x float> %2052, float %1539, i64 3, !dbg !19
  %normalized90 = fmul <4 x float> %2053, %row.reciprocal88, !dbg !19
  %2054 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !20
  %2055 = extractelement <4 x half> %2054, i64 3, !dbg !20
  %2056 = extractelement <4 x half> %2054, i64 2, !dbg !20
  %2057 = extractelement <4 x half> %2054, i64 1, !dbg !20
  %2058 = extractelement <4 x half> %2054, i64 0, !dbg !20
  %2059 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread505 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local506 = zext nneg i32 %address.thread505 to i64, !dbg !13
  %address.local507 = add nsw i64 %16, %address.local506, !dbg !21
  %address.local508 = lshr i64 %address.local506, 6, !dbg !13
  %address.local509 = shl nuw nsw i64 %address.local508, 4, !dbg !21
  %address.local510 = add nsw i64 %address.local507, %address.local509, !dbg !21
  %address.local511 = and i64 %address.local506, 240, !dbg !21
  %address.local512 = sub nsw i64 %address.local510, %address.local511, !dbg !21
  %address.local513 = add nsw i64 %address.local512, %17, !dbg !21
  %address.local514 = shl nuw nsw i64 %address.local513, 7, !dbg !21
  %address.local515 = add nuw nsw i64 %2059, %address.local514, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local517 = lshr i64 %address.local506, 2, !dbg !13
  %address.local518 = and i64 %address.local517, 12, !dbg !13
  %address.local519 = or disjoint i64 %address.local518, 64, !dbg !21
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local519, !dbg !21
  store half %2058, ptr addrspace(1) %address.local520, align 2, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local522 = or disjoint i64 %address.local518, 65, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local522, !dbg !21
  store half %2057, ptr addrspace(1) %address.local523, align 2, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local525 = or disjoint i64 %address.local518, 66, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local524, i64 %address.local525, !dbg !21
  store half %2056, ptr addrspace(1) %address.local526, align 2, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local528 = or disjoint i64 %address.local518, 67, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local528, !dbg !21
  store half %2055, ptr addrspace(1) %address.local529, align 2, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local531 = or disjoint i64 %address.local518, 80, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local530, i64 %address.local531, !dbg !21
  store half %2049, ptr addrspace(1) %address.local532, align 2, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local534 = or disjoint i64 %address.local518, 81, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !21
  store half %2048, ptr addrspace(1) %address.local535, align 2, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local537 = or disjoint i64 %address.local518, 82, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local537, !dbg !21
  store half %2047, ptr addrspace(1) %address.local538, align 2, !dbg !21
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !21
  %address.local540 = or disjoint i64 %address.local518, 83, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local540, !dbg !21
  store half %2046, ptr addrspace(1) %address.local541, align 2, !dbg !21
  %address.local542 = shl nsw i64 %address.local513, 7, !dbg !21
  %address.local543 = add i64 %address.local542, 8192, !dbg !21
  %address.local544 = add nuw nsw i64 %2059, %address.local543, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local519, !dbg !21
  store half %2040, ptr addrspace(1) %address.local546, align 2, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local522, !dbg !21
  store half %2039, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local525, !dbg !21
  store half %2038, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local528, !dbg !21
  store half %2037, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local531, !dbg !21
  store half %2031, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local534, !dbg !21
  store half %2030, ptr addrspace(1) %address.local556, align 2, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local537, !dbg !21
  store half %2029, ptr addrspace(1) %address.local558, align 2, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local540, !dbg !21
  store half %2028, ptr addrspace(1) %address.local560, align 2, !dbg !21
  br label %2060, !dbg !21

2060:                                             ; preds = %2022, %2021
  br i1 true, label %2061, label %2099, !dbg !21

2061:                                             ; preds = %2060
  %2062 = insertelement <4 x float> poison, float %1832, i64 0, !dbg !19
  %2063 = insertelement <4 x float> %2062, float %1838, i64 1, !dbg !19
  %2064 = insertelement <4 x float> %2063, float %1833, i64 2, !dbg !19
  %2065 = insertelement <4 x float> %2064, float %1839, i64 3, !dbg !19
  %row.reciprocal91 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1811, !dbg !19
  %normalized92 = fmul <4 x float> %2065, %row.reciprocal91, !dbg !19
  %2066 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !20
  %2067 = extractelement <4 x half> %2066, i64 3, !dbg !20
  %2068 = extractelement <4 x half> %2066, i64 2, !dbg !20
  %2069 = extractelement <4 x half> %2066, i64 1, !dbg !20
  %2070 = extractelement <4 x half> %2066, i64 0, !dbg !20
  %2071 = insertelement <4 x float> poison, float %1721, i64 0, !dbg !19
  %2072 = insertelement <4 x float> %2071, float %1727, i64 1, !dbg !19
  %2073 = insertelement <4 x float> %2072, float %1722, i64 2, !dbg !19
  %2074 = insertelement <4 x float> %2073, float %1728, i64 3, !dbg !19
  %normalized93 = fmul <4 x float> %2074, %row.reciprocal91, !dbg !19
  %2075 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !20
  %2076 = extractelement <4 x half> %2075, i64 3, !dbg !20
  %2077 = extractelement <4 x half> %2075, i64 2, !dbg !20
  %2078 = extractelement <4 x half> %2075, i64 1, !dbg !20
  %2079 = extractelement <4 x half> %2075, i64 0, !dbg !20
  %2080 = insertelement <4 x float> poison, float %1614, i64 0, !dbg !19
  %2081 = insertelement <4 x float> %2080, float %1620, i64 1, !dbg !19
  %2082 = insertelement <4 x float> %2081, float %1615, i64 2, !dbg !19
  %2083 = insertelement <4 x float> %2082, float %1621, i64 3, !dbg !19
  %row.reciprocal94 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1593, !dbg !19
  %normalized95 = fmul <4 x float> %2083, %row.reciprocal94, !dbg !19
  %2084 = fptrunc <4 x float> %normalized95 to <4 x half>, !dbg !20
  %2085 = extractelement <4 x half> %2084, i64 3, !dbg !20
  %2086 = extractelement <4 x half> %2084, i64 2, !dbg !20
  %2087 = extractelement <4 x half> %2084, i64 1, !dbg !20
  %2088 = extractelement <4 x half> %2084, i64 0, !dbg !20
  %2089 = insertelement <4 x float> poison, float %1503, i64 0, !dbg !19
  %2090 = insertelement <4 x float> %2089, float %1509, i64 1, !dbg !19
  %2091 = insertelement <4 x float> %2090, float %1504, i64 2, !dbg !19
  %2092 = insertelement <4 x float> %2091, float %1510, i64 3, !dbg !19
  %normalized96 = fmul <4 x float> %2092, %row.reciprocal94, !dbg !19
  %2093 = fptrunc <4 x float> %normalized96 to <4 x half>, !dbg !20
  %2094 = extractelement <4 x half> %2093, i64 3, !dbg !20
  %2095 = extractelement <4 x half> %2093, i64 2, !dbg !20
  %2096 = extractelement <4 x half> %2093, i64 1, !dbg !20
  %2097 = extractelement <4 x half> %2093, i64 0, !dbg !20
  %2098 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread561 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local562 = zext nneg i32 %address.thread561 to i64, !dbg !13
  %address.local563 = add nsw i64 %16, %address.local562, !dbg !21
  %address.local564 = lshr i64 %address.local562, 6, !dbg !13
  %address.local565 = shl nuw nsw i64 %address.local564, 4, !dbg !21
  %address.local566 = add nsw i64 %address.local563, %address.local565, !dbg !21
  %address.local567 = and i64 %address.local562, 240, !dbg !21
  %address.local568 = sub nsw i64 %address.local566, %address.local567, !dbg !21
  %address.local569 = add nsw i64 %address.local568, %17, !dbg !21
  %address.local570 = shl nuw nsw i64 %address.local569, 7, !dbg !21
  %address.local571 = add nuw nsw i64 %2098, %address.local570, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local573 = lshr i64 %address.local562, 2, !dbg !13
  %address.local574 = and i64 %address.local573, 12, !dbg !13
  %address.local575 = or disjoint i64 %address.local574, 96, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local575, !dbg !21
  store half %2097, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local578 = or disjoint i64 %address.local574, 97, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local578, !dbg !21
  store half %2096, ptr addrspace(1) %address.local579, align 2, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local581 = or disjoint i64 %address.local574, 98, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local580, i64 %address.local581, !dbg !21
  store half %2095, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local584 = or disjoint i64 %address.local574, 99, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local584, !dbg !21
  store half %2094, ptr addrspace(1) %address.local585, align 2, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local587 = or disjoint i64 %address.local574, 112, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local586, i64 %address.local587, !dbg !21
  store half %2088, ptr addrspace(1) %address.local588, align 2, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local590 = or disjoint i64 %address.local574, 113, !dbg !21
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local590, !dbg !21
  store half %2087, ptr addrspace(1) %address.local591, align 2, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local593 = or disjoint i64 %address.local574, 114, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local593, !dbg !21
  store half %2086, ptr addrspace(1) %address.local594, align 2, !dbg !21
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !21
  %address.local596 = or i64 %address.local573, 115, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local596, !dbg !21
  store half %2085, ptr addrspace(1) %address.local597, align 2, !dbg !21
  %address.local598 = shl nsw i64 %address.local569, 7, !dbg !21
  %address.local599 = add i64 %address.local598, 8192, !dbg !21
  %address.local600 = add nuw nsw i64 %2098, %address.local599, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local575, !dbg !21
  store half %2079, ptr addrspace(1) %address.local602, align 2, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local578, !dbg !21
  store half %2078, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local581, !dbg !21
  store half %2077, ptr addrspace(1) %address.local606, align 2, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local584, !dbg !21
  store half %2076, ptr addrspace(1) %address.local608, align 2, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local587, !dbg !21
  store half %2070, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local590, !dbg !21
  store half %2069, ptr addrspace(1) %address.local612, align 2, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local593, !dbg !21
  store half %2068, ptr addrspace(1) %address.local614, align 2, !dbg !21
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local596, !dbg !21
  store half %2067, ptr addrspace(1) %address.local616, align 2, !dbg !21
  br label %2099, !dbg !21

2099:                                             ; preds = %2061, %2060
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
!llvm.module.flags = !{!32}

!0 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!10 = !DILocation(line: 46, column: 17, scope: !4)
!12 = !DILocation(line: 66, column: 22, scope: !4)
!13 = !DILocation(line: 68, column: 24, scope: !4)
!14 = !DILocation(line: 73, column: 19, scope: !4)
!15 = !DILocation(line: 93, column: 23, scope: !4)
!16 = !DILocation(line: 56, column: 24, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 92, column: 20, scope: !4)
!19 = !DILocation(line: 97, column: 23, scope: !4)
!20 = !DILocation(line: 98, column: 19, scope: !4)
!21 = !DILocation(line: 99, column: 9, scope: !4)
!22 = !DILocation(line: 59, column: 17, scope: !4)
!23 = !DILocation(line: 75, column: 24, scope: !4)
!24 = !DILocation(line: 76, column: 23, scope: !4)
!25 = !DILocation(line: 77, column: 26, scope: !4)
!26 = !DILocation(line: 85, column: 22, scope: !4)
!27 = !DILocation(line: 86, column: 20, scope: !4)
!28 = !DILocation(line: 87, column: 16, scope: !4)
!29 = !DILocation(line: 94, column: 20, scope: !4)
!30 = !DILocation(line: 70, column: 23, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
