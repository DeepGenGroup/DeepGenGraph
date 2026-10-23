; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/attn_best_async_current.ll'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !32 !reqd_work_group_size !55 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !33
  %6 = sext i32 %5 to i64, !dbg !33
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !33
  %8 = sext i32 %7 to i64, !dbg !33
  %9 = shl nsw i64 %6, 7, !dbg !35
  %.lobit = ashr i32 %5, 31, !dbg !36
  %10 = xor i32 %5, %.lobit, !dbg !36
  %11 = sdiv i32 %10, 32, !dbg !36
  %.lobit1 = ashr i32 %5, 31, !dbg !36
  %12 = xor i32 %11, %.lobit1, !dbg !36
  %13 = sext i32 %12 to i64, !dbg !36
  %14 = add nsw i64 %8, %13, !dbg !36
  %15 = and i64 %14, 31, !dbg !36
  %16 = shl nsw i64 %6, 7, !dbg !36
  %17 = mul nsw i64 %13, -4096, !dbg !36
  %18 = add nsw i64 %16, %17, !dbg !36
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !37, !range !57
  %20 = zext nneg i32 %19 to i64, !dbg !37
  %21 = lshr i64 %20, 6, !dbg !37
  %22 = or disjoint i64 %9, 127, !dbg !38
  %23 = add nsw i64 %9, 128, !dbg !38
  %24 = icmp uge i64 %23, %22, !dbg !38
  br label %25, !dbg !40

25:                                               ; preds = %768, %4
  %26 = phi i64 [ %1467, %768 ], [ 0, %4 ], !dbg !40
  %27 = phi [2 x <8 x float>] [ %1454, %768 ], [ zeroinitializer, %4 ], !dbg !40
  %28 = phi [2 x <1 x float>] [ %1076, %768 ], [ zeroinitializer, %4 ], !dbg !40
  %29 = phi [2 x <8 x float>] [ %1458, %768 ], [ zeroinitializer, %4 ], !dbg !40
  %30 = phi [2 x <8 x float>] [ %1462, %768 ], [ zeroinitializer, %4 ], !dbg !40
  %31 = phi [2 x <8 x float>] [ %1466, %768 ], [ zeroinitializer, %4 ], !dbg !40
  %.not = icmp sgt i64 %26, %6, !dbg !40
  br i1 %.not, label %1468, label %32, !dbg !40

32:                                               ; preds = %25
  %33 = shl nsw i64 %26, 7, !dbg !40
  %34 = lshr i64 %26, 5, !dbg !41
  %35 = shl i64 %34, 12, !dbg !41
  %36 = sub i64 %33, %35, !dbg !41
  br label %37, !dbg !46

37:                                               ; preds = %41, %32
  %38 = phi i64 [ %234, %41 ], [ 0, %32 ], !dbg !46
  %39 = phi [2 x <32 x float>] [ %233, %41 ], [ zeroinitializer, %32 ], !dbg !46
  %40 = icmp samesign ult i64 %38, 4, !dbg !46
  br i1 %40, label %41, label %235, !dbg !46

41:                                               ; preds = %37
  %42 = shl nuw nsw i64 %38, 5, !dbg !46
  %.idx = shl nuw nsw i64 %15, 20, !dbg !36
  %43 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !36
  %.idx71 = shl nsw i64 %18, 8, !dbg !36
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 %.idx71, !dbg !36
  %45 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %42, !dbg !36
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !36, !range !57, !frisk.register_lifetimes !29
  %address.local = zext nneg i32 %address.thread to i64, !dbg !37
  %address.local133 = shl nuw nsw i64 %address.local, 3, !dbg !36
  %address.local134 = and i64 %address.local133, 1536, !dbg !36
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !36
  %address.local136 = ptrtoint ptr addrspace(3) %address.local135 to i64, !dbg !36
  %address.local137 = trunc i64 %address.local136 to i32, !dbg !36
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !36
  %47 = zext i32 %46 to i64, !dbg !36
  %48 = inttoptr i64 %47 to ptr addrspace(3), !dbg !36
  %address.local138 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local139 = and i64 %address.local138, 16256, !dbg !36
  %address.local140 = lshr i64 %address.local, 4, !dbg !36
  %address.local141 = and i64 %address.local140, 8, !dbg !36
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !36
  %address.local143 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local142, !dbg !36
  %49 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %50 = ptrtoint ptr addrspace(1) %address.local143 to i64, !dbg !36
  %51 = sub i64 %50, %49, !dbg !36
  %dma.buffer.offset623 = trunc i64 %51 to i32, !dbg !36
  %52 = trunc i64 %49 to i32, !dbg !36
  %53 = insertelement <4 x i32> poison, i32 %52, i64 0, !dbg !36
  %54 = lshr i64 %49, 32, !dbg !36
  %55 = trunc i64 %54 to i32, !dbg !36
  %56 = insertelement <4 x i32> %53, i32 %55, i64 1, !dbg !36
  %57 = insertelement <4 x i32> %56, i32 33554432, i64 2, !dbg !36
  %dma.buffer.rsrc624 = insertelement <4 x i32> %57, i32 131072, i64 3, !dbg !36
  %58 = ptrtoint ptr addrspace(3) %48 to i32, !dbg !36
  %dma.buffer.lds625 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !36
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !36
  %address.local144 = or disjoint i64 %address.local, 256, !dbg !36
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !36
  %address.local146 = and i64 %address.local145, 3584, !dbg !36
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !36
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !36
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !36
  %60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !36
  %61 = zext i32 %60 to i64, !dbg !36
  %62 = inttoptr i64 %61 to ptr addrspace(3), !dbg !36
  %address.local150 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local151 = and i64 %address.local150, 16256, !dbg !36
  %address.local152 = lshr i64 %address.local144, 4, !dbg !36
  %address.local153 = and i64 %address.local152, 24, !dbg !36
  %address.local154 = or disjoint i64 %address.local151, %address.local153, !dbg !36
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local154, !dbg !36
  %63 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %64 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !36
  %65 = sub i64 %64, %63, !dbg !36
  %dma.buffer.offset620 = trunc i64 %65 to i32, !dbg !36
  %66 = trunc i64 %63 to i32, !dbg !36
  %67 = insertelement <4 x i32> poison, i32 %66, i64 0, !dbg !36
  %68 = lshr i64 %63, 32, !dbg !36
  %69 = trunc i64 %68 to i32, !dbg !36
  %70 = insertelement <4 x i32> %67, i32 %69, i64 1, !dbg !36
  %71 = insertelement <4 x i32> %70, i32 33554432, i64 2, !dbg !36
  %dma.buffer.rsrc621 = insertelement <4 x i32> %71, i32 131072, i64 3, !dbg !36
  %72 = ptrtoint ptr addrspace(3) %62 to i32, !dbg !36
  %dma.buffer.lds622 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72), !dbg !36
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc621) #10, !dbg !36
  %74 = shl nuw nsw i64 %38, 10, !dbg !41
  %75 = add nuw i64 %26, %74, !dbg !41
  %.lobit72 = ashr i64 %75, 63, !dbg !41
  %76 = xor i64 %75, %.lobit72, !dbg !41
  %77 = sdiv i64 %76, 4096, !dbg !41
  %.lobit73 = ashr i64 %75, 63, !dbg !41
  %78 = xor i64 %77, %.lobit73, !dbg !41
  %79 = add nsw i64 %78, %8, !dbg !41
  %80 = add nuw nsw i64 %42, %34, !dbg !41
  %81 = and i64 %80, 127, !dbg !41
  %82 = shl i64 %79, 19, !dbg !41
  %83 = and i64 %82, 16252928, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %84 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %83, !dbg !41
  %85 = getelementptr [2 x i8], ptr addrspace(1) %84, i64 %81, !dbg !41
  %.idx74 = shl i64 %36, 8, !dbg !41
  %86 = getelementptr i8, ptr addrspace(1) %85, i64 %.idx74, !dbg !41
  %87 = getelementptr inbounds nuw i8, ptr addrspace(3) @shm_0, i64 8192, !dbg !34
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local134, !dbg !41
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !41
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !41
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !41
  %89 = zext i32 %88 to i64, !dbg !41
  %90 = inttoptr i64 %89 to ptr addrspace(3), !dbg !41
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %address.local142, !dbg !41
  %91 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !41
  %92 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !41
  %93 = sub i64 %92, %91, !dbg !41
  %dma.buffer.offset617 = trunc i64 %93 to i32, !dbg !41
  %94 = trunc i64 %91 to i32, !dbg !41
  %95 = insertelement <4 x i32> poison, i32 %94, i64 0, !dbg !41
  %96 = lshr i64 %91, 32, !dbg !41
  %97 = trunc i64 %96 to i32, !dbg !41
  %98 = insertelement <4 x i32> %95, i32 %97, i64 1, !dbg !41
  %99 = insertelement <4 x i32> %98, i32 33554432, i64 2, !dbg !41
  %dma.buffer.rsrc618 = insertelement <4 x i32> %99, i32 131072, i64 3, !dbg !41
  %100 = ptrtoint ptr addrspace(3) %90 to i32, !dbg !41
  %dma.buffer.lds619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %100), !dbg !41
  %101 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !41
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local146, !dbg !41
  %address.local161 = ptrtoint ptr addrspace(3) %address.local160 to i64, !dbg !41
  %address.local162 = trunc i64 %address.local161 to i32, !dbg !41
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local162), !dbg !41
  %103 = zext i32 %102 to i64, !dbg !41
  %104 = inttoptr i64 %103 to ptr addrspace(3), !dbg !41
  %address.local163 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %address.local154, !dbg !41
  %105 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !41
  %106 = ptrtoint ptr addrspace(1) %address.local163 to i64, !dbg !41
  %107 = sub i64 %106, %105, !dbg !41
  %dma.buffer.offset = trunc i64 %107 to i32, !dbg !41
  %108 = trunc i64 %105 to i32, !dbg !41
  %109 = insertelement <4 x i32> poison, i32 %108, i64 0, !dbg !41
  %110 = lshr i64 %105, 32, !dbg !41
  %111 = trunc i64 %110 to i32, !dbg !41
  %112 = insertelement <4 x i32> %109, i32 %111, i64 1, !dbg !41
  %113 = insertelement <4 x i32> %112, i32 33554432, i64 2, !dbg !41
  %dma.buffer.rsrc = insertelement <4 x i32> %113, i32 131072, i64 3, !dbg !41
  %114 = ptrtoint ptr addrspace(3) %104 to i32, !dbg !41
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %114), !dbg !41
  %115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !41
  %address.local164 = lshr i32 %address.thread, 2, !dbg !37
  %address.local165 = and i32 %address.local164, 4, !dbg !37
  %address.local166 = shl nuw nsw i32 %address.thread, 1, !dbg !37
  %address.local167 = and i32 %address.local166, 384, !dbg !37
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !37
  %address.local169 = shl nuw nsw i32 %address.thread, 3, !dbg !37
  %address.local170 = and i32 %address.local169, 120, !dbg !37
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !37
  %address.local172 = shl nuw nsw i32 %address.thread, 5, !dbg !37
  %address.local173 = and i32 %address.local172, 1024, !dbg !37
  %address.local174 = or disjoint i32 %address.local171, %address.local173, !dbg !37
  %address.local175 = zext nneg i32 %address.local174 to i64, !dbg !37
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %116 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !37
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 4096, !dbg !37
  %117 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !37
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local180 = getelementptr i8, ptr addrspace(3) %address.local179, i64 1024, !dbg !37
  %118 = load <4 x half>, ptr addrspace(3) %address.local180, align 2, !dbg !37
  %119 = fmul <4 x half> %116, splat (half 0xH3015), !dbg !37
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 5120, !dbg !37
  %120 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !37
  %121 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !37
  %122 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !37
  %123 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %address.local183 = lshr i32 %address.thread, 2, !dbg !34
  %address.local184 = and i32 %address.local183, 4, !dbg !34
  %address.local185 = shl nuw nsw i32 %address.thread, 1, !dbg !34
  %address.local186 = and i32 %address.local185, 24, !dbg !34
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !34
  %address.local188 = shl nuw nsw i32 %address.thread, 5, !dbg !34
  %address.local189 = and i32 %address.local188, 1024, !dbg !34
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !34
  %address.local191 = shl nuw nsw i32 %address.thread, 7, !dbg !34
  %address.local192 = and i32 %address.local191, 384, !dbg !34
  %address.local193 = or disjoint i32 %address.local190, %address.local192, !dbg !34
  %address.local194 = zext nneg i32 %address.local193 to i64, !dbg !34
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local194, !dbg !34
  %124 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !34
  %125 = extractvalue [2 x <32 x float>] %39, 0, !dbg !54
  %126 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %127 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %124, <4 x float> %126) #9, !dbg !34
  %address.local196 = lshr i32 %address.thread, 2, !dbg !34
  %address.local197 = and i32 %address.local196, 4, !dbg !34
  %address.local198 = shl nuw nsw i32 %address.thread, 1, !dbg !34
  %address.local199 = and i32 %address.local198, 24, !dbg !34
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !34
  %address.local201 = shl nuw nsw i32 %address.thread, 5, !dbg !34
  %address.local202 = and i32 %address.local201, 1024, !dbg !34
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !34
  %address.local204 = shl nuw nsw i32 %address.thread, 7, !dbg !34
  %address.local205 = and i32 %address.local204, 384, !dbg !34
  %address.local206 = or disjoint i32 %address.local203, %address.local205, !dbg !34
  %address.local207 = zext nneg i32 %address.local206 to i64, !dbg !34
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local207, !dbg !34
  %address.local209 = getelementptr i8, ptr addrspace(3) %address.local208, i64 4096, !dbg !34
  %128 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !34
  %129 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %130 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %128, <4 x float> %127) #9, !dbg !34
  %131 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !54
  %132 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !54
  %133 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !54
  %134 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !54
  %135 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !54
  %136 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !54
  %137 = extractvalue [2 x <32 x float>] %39, 1, !dbg !54
  %138 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %139 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %140 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !54
  %141 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !54
  %142 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !54
  %143 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !54
  %144 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !54
  %145 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !54
  %address.local210 = shl nuw nsw i64 %address.local, 7, !dbg !34
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local212 = lshr i64 %address.local, 2, !dbg !34
  %address.local213 = mul nsw i64 %address.local212, -512, !dbg !34
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !34
  %address.local215 = shl nuw nsw i64 %address.local, 1, !dbg !34
  %address.local216 = and i64 %address.local215, 24, !dbg !34
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !34
  %address.local218 = lshr i64 %address.local, 2, !dbg !37
  %address.local219 = and i64 %address.local218, 12, !dbg !37
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local219, !dbg !34
  %address.local221 = and i64 %address.local, 63, !dbg !37
  %address.local222 = lshr i64 %address.local221, 5, !dbg !37
  %address.local223 = mul nuw nsw i64 %address.local222, 1016, !dbg !37
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local223, !dbg !34
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 64, !dbg !34
  %146 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !34
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local219, !dbg !34
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local223, !dbg !34
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local213, !dbg !34
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local216, !dbg !34
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4160, !dbg !34
  %147 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !34
  %148 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %146, <4 x float> %129) #9, !dbg !34
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local213, !dbg !34
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local216, !dbg !34
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local219, !dbg !34
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local223, !dbg !34
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 128, !dbg !34
  %149 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !34
  %150 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %147, <4 x float> %148) #9, !dbg !34
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local219, !dbg !34
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local223, !dbg !34
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local213, !dbg !34
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local216, !dbg !34
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 4224, !dbg !34
  %151 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !34
  %152 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %149, <4 x float> %131) #9, !dbg !34
  %153 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %151, <4 x float> %152) #9, !dbg !34
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local213, !dbg !34
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local216, !dbg !34
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local219, !dbg !34
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local223, !dbg !34
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local248, i64 192, !dbg !34
  %154 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !34
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local219, !dbg !34
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local223, !dbg !34
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) %address.local252, i64 %address.local213, !dbg !34
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local253, i64 %address.local216, !dbg !34
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 4288, !dbg !34
  %155 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !34
  %156 = shufflevector <4 x float> %153, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %157 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %154, <4 x float> %132) #9, !dbg !34
  %158 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %155, <4 x float> %157) #9, !dbg !34
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local194, !dbg !34
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local256, i64 1024, !dbg !34
  %159 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !34
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local207, !dbg !34
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 5120, !dbg !34
  %160 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !34
  %161 = shufflevector <4 x float> %158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %162 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %159, <4 x float> %133) #9, !dbg !34
  %163 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %160, <4 x float> %162) #9, !dbg !34
  %address.local260 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local260, !dbg !34
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !34
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !34
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !34
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !34
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !34
  %164 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !34
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local260, !dbg !34
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local219, !dbg !34
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !34
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !34
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !34
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !34
  %165 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !34
  %166 = shufflevector <4 x float> %163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %167 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %164, <4 x float> %134) #9, !dbg !34
  %168 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %165, <4 x float> %167) #9, !dbg !34
  %address.local273 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local273, !dbg !34
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local213, !dbg !34
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local216, !dbg !34
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local219, !dbg !34
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local223, !dbg !34
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local278, i64 128, !dbg !34
  %169 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !34
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local273, !dbg !34
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local219, !dbg !34
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local223, !dbg !34
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local213, !dbg !34
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local216, !dbg !34
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 4224, !dbg !34
  %170 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !34
  %171 = shufflevector <4 x float> %168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %172 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %169, <4 x float> %135) #9, !dbg !34
  %173 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %170, <4 x float> %172) #9, !dbg !34
  %address.local286 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local286, !dbg !34
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local213, !dbg !34
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local216, !dbg !34
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local219, !dbg !34
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local223, !dbg !34
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 192, !dbg !34
  %174 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !34
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local286, !dbg !34
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local219, !dbg !34
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %address.local294, i64 %address.local223, !dbg !34
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local213, !dbg !34
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local216, !dbg !34
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local297, i64 4288, !dbg !34
  %175 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !34
  %176 = shufflevector <4 x float> %173, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %177 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %174, <4 x float> %136) #9, !dbg !34
  %178 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %175, <4 x float> %177) #9, !dbg !34
  %179 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !34
  %180 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !34
  %181 = shufflevector <4 x float> %178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %182 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %179, <4 x float> %138) #9, !dbg !34
  %183 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %180, <4 x float> %182) #9, !dbg !34
  %184 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !34
  %185 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !34
  %186 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %184, <4 x float> %139) #9, !dbg !34
  %187 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %185, <4 x float> %186) #9, !dbg !34
  %188 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !34
  %189 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !34
  %190 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %188, <4 x float> %140) #9, !dbg !34
  %191 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %189, <4 x float> %190) #9, !dbg !34
  %192 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !34
  %193 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !34
  %194 = shufflevector <4 x float> %191, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %195 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %192, <4 x float> %141) #9, !dbg !34
  %196 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %193, <4 x float> %195) #9, !dbg !34
  %197 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !34
  %198 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !34
  %199 = shufflevector <4 x float> %196, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %200 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %197, <4 x float> %142) #9, !dbg !34
  %201 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %198, <4 x float> %200) #9, !dbg !34
  %202 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !34
  %203 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !34
  %204 = shufflevector <4 x float> %201, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %205 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %202, <4 x float> %143) #9, !dbg !34
  %206 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %203, <4 x float> %205) #9, !dbg !34
  %207 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !34
  %208 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !34
  %209 = shufflevector <4 x float> %206, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %210 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %207, <4 x float> %144) #9, !dbg !34
  %211 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %208, <4 x float> %210) #9, !dbg !34
  %212 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !34
  %213 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !34
  %214 = shufflevector <4 x float> %211, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %215 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %212, <4 x float> %145) #9, !dbg !34
  %216 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %213, <4 x float> %215) #9, !dbg !34
  %217 = shufflevector <4 x float> %216, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %218 = shufflevector <4 x float> %130, <4 x float> %150, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %219 = shufflevector <32 x float> %218, <32 x float> %156, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %220 = shufflevector <32 x float> %219, <32 x float> %161, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %221 = shufflevector <32 x float> %220, <32 x float> %166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %222 = shufflevector <32 x float> %221, <32 x float> %171, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %223 = shufflevector <32 x float> %222, <32 x float> %176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %224 = shufflevector <32 x float> %223, <32 x float> %181, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !46
  %225 = insertvalue [2 x <32 x float>] poison, <32 x float> %224, 0, !dbg !46
  %226 = shufflevector <4 x float> %183, <4 x float> %187, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %227 = shufflevector <32 x float> %226, <32 x float> %194, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %228 = shufflevector <32 x float> %227, <32 x float> %199, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %229 = shufflevector <32 x float> %228, <32 x float> %204, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %230 = shufflevector <32 x float> %229, <32 x float> %209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %231 = shufflevector <32 x float> %230, <32 x float> %214, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %232 = shufflevector <32 x float> %231, <32 x float> %217, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !46
  %233 = insertvalue [2 x <32 x float>] %225, <32 x float> %232, 1, !dbg !46
  fence syncscope("workgroup") release, !dbg !46
  call void @llvm.amdgcn.s.barrier(), !dbg !46
  fence syncscope("workgroup") acquire, !dbg !46
  %234 = add nuw nsw i64 %38, 1, !dbg !46
  br label %37, !dbg !46

235:                                              ; preds = %37
  %236 = add nsw i64 %34, %8, !dbg !42
  %237 = shl i64 %236, 19, !dbg !42
  %238 = and i64 %237, 16252928, !dbg !42
  %239 = shl nsw i64 %36, 7, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %240 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %238, !dbg !42
  %241 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %239, !dbg !42
  %address.thread299 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !42, !range !57, !frisk.register_lifetimes !29
  %address.local300 = zext nneg i32 %address.thread299 to i64, !dbg !37
  %address.local301 = shl nuw nsw i64 %address.local300, 3, !dbg !36
  %address.local302 = and i64 %address.local301, 1536, !dbg !36
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !36
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !36
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !36
  %242 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !42
  %243 = zext i32 %242 to i64, !dbg !42
  %244 = inttoptr i64 %243 to ptr addrspace(3), !dbg !42
  %address.local306 = shl nuw nsw i64 %address.local300, 5, !dbg !42
  %address.local307 = and i64 %address.local306, 8064, !dbg !42
  %address.local308 = shl nuw nsw i64 %address.local300, 3, !dbg !42
  %address.local309 = and i64 %address.local308, 24, !dbg !42
  %address.local310 = or disjoint i64 %address.local307, %address.local309, !dbg !42
  %address.local311 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %address.local310, !dbg !42
  %245 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %246 = ptrtoint ptr addrspace(1) %address.local311 to i64, !dbg !42
  %247 = sub i64 %246, %245, !dbg !42
  %dma.buffer.offset647 = trunc i64 %247 to i32, !dbg !42
  %248 = trunc i64 %245 to i32, !dbg !42
  %249 = insertelement <4 x i32> poison, i32 %248, i64 0, !dbg !42
  %250 = lshr i64 %245, 32, !dbg !42
  %251 = trunc i64 %250 to i32, !dbg !42
  %252 = insertelement <4 x i32> %249, i32 %251, i64 1, !dbg !42
  %253 = insertelement <4 x i32> %252, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc648 = insertelement <4 x i32> %253, i32 131072, i64 3, !dbg !42
  %254 = ptrtoint ptr addrspace(3) %244 to i32, !dbg !42
  %dma.buffer.lds649 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %254), !dbg !42
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds649, i32 %dma.buffer.offset647, <4 x i32> %dma.buffer.rsrc648) #10, !dbg !42
  %address.local312 = or disjoint i64 %address.local300, 256, !dbg !36
  %address.local313 = shl nuw nsw i64 %address.local312, 3, !dbg !36
  %address.local314 = and i64 %address.local313, 3584, !dbg !36
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !36
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !36
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !36
  %256 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !42
  %257 = zext i32 %256 to i64, !dbg !42
  %258 = inttoptr i64 %257 to ptr addrspace(3), !dbg !42
  %address.local318 = shl nuw nsw i64 %address.local312, 5, !dbg !42
  %address.local319 = and i64 %address.local318, 16256, !dbg !42
  %address.local320 = shl nuw nsw i64 %address.local300, 3, !dbg !42
  %address.local321 = and i64 %address.local320, 24, !dbg !42
  %address.local322 = or disjoint i64 %address.local319, %address.local321, !dbg !42
  %address.local323 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %address.local322, !dbg !42
  %259 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %260 = ptrtoint ptr addrspace(1) %address.local323 to i64, !dbg !42
  %261 = sub i64 %260, %259, !dbg !42
  %dma.buffer.offset644 = trunc i64 %261 to i32, !dbg !42
  %262 = trunc i64 %259 to i32, !dbg !42
  %263 = insertelement <4 x i32> poison, i32 %262, i64 0, !dbg !42
  %264 = lshr i64 %259, 32, !dbg !42
  %265 = trunc i64 %264 to i32, !dbg !42
  %266 = insertelement <4 x i32> %263, i32 %265, i64 1, !dbg !42
  %267 = insertelement <4 x i32> %266, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc645 = insertelement <4 x i32> %267, i32 131072, i64 3, !dbg !42
  %268 = ptrtoint ptr addrspace(3) %258 to i32, !dbg !42
  %dma.buffer.lds646 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %268), !dbg !42
  %269 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds646, i32 %dma.buffer.offset644, <4 x i32> %dma.buffer.rsrc645) #10, !dbg !42
  %270 = or disjoint i64 %33, 127, !dbg !38
  %271 = icmp uge i64 %9, %270, !dbg !38
  %272 = and i1 %24, %271, !dbg !38
  br i1 %272, label %273, label %468, !dbg !38

273:                                              ; preds = %235
  %274 = extractvalue [2 x <32 x float>] %39, 0, !dbg !38
  %275 = extractelement <32 x float> %274, i64 0, !dbg !38
  %276 = fadd float %275, 0x0000000000000000, !dbg !38
  %277 = extractelement <32 x float> %274, i64 1, !dbg !38
  %278 = fadd float %277, 0x0000000000000000, !dbg !38
  %279 = extractelement <32 x float> %274, i64 2, !dbg !38
  %280 = fadd float %279, 0x0000000000000000, !dbg !38
  %281 = extractelement <32 x float> %274, i64 3, !dbg !38
  %282 = fadd float %281, 0x0000000000000000, !dbg !38
  %283 = extractelement <32 x float> %274, i64 4, !dbg !38
  %284 = fadd float %283, 0x0000000000000000, !dbg !38
  %285 = extractelement <32 x float> %274, i64 5, !dbg !38
  %286 = fadd float %285, 0x0000000000000000, !dbg !38
  %287 = extractelement <32 x float> %274, i64 6, !dbg !38
  %288 = fadd float %287, 0x0000000000000000, !dbg !38
  %289 = extractelement <32 x float> %274, i64 7, !dbg !38
  %290 = fadd float %289, 0x0000000000000000, !dbg !38
  %291 = extractelement <32 x float> %274, i64 8, !dbg !38
  %292 = fadd float %291, 0x0000000000000000, !dbg !38
  %293 = extractelement <32 x float> %274, i64 9, !dbg !38
  %294 = fadd float %293, 0x0000000000000000, !dbg !38
  %295 = extractelement <32 x float> %274, i64 10, !dbg !38
  %296 = fadd float %295, 0x0000000000000000, !dbg !38
  %297 = extractelement <32 x float> %274, i64 11, !dbg !38
  %298 = fadd float %297, 0x0000000000000000, !dbg !38
  %299 = extractelement <32 x float> %274, i64 12, !dbg !38
  %300 = fadd float %299, 0x0000000000000000, !dbg !38
  %301 = extractelement <32 x float> %274, i64 13, !dbg !38
  %302 = fadd float %301, 0x0000000000000000, !dbg !38
  %303 = extractelement <32 x float> %274, i64 14, !dbg !38
  %304 = fadd float %303, 0x0000000000000000, !dbg !38
  %305 = extractelement <32 x float> %274, i64 15, !dbg !38
  %306 = fadd float %305, 0x0000000000000000, !dbg !38
  %307 = extractelement <32 x float> %274, i64 16, !dbg !38
  %308 = fadd float %307, 0x0000000000000000, !dbg !38
  %309 = extractelement <32 x float> %274, i64 17, !dbg !38
  %310 = fadd float %309, 0x0000000000000000, !dbg !38
  %311 = extractelement <32 x float> %274, i64 18, !dbg !38
  %312 = fadd float %311, 0x0000000000000000, !dbg !38
  %313 = extractelement <32 x float> %274, i64 19, !dbg !38
  %314 = fadd float %313, 0x0000000000000000, !dbg !38
  %315 = extractelement <32 x float> %274, i64 20, !dbg !38
  %316 = fadd float %315, 0x0000000000000000, !dbg !38
  %317 = extractelement <32 x float> %274, i64 21, !dbg !38
  %318 = fadd float %317, 0x0000000000000000, !dbg !38
  %319 = extractelement <32 x float> %274, i64 22, !dbg !38
  %320 = fadd float %319, 0x0000000000000000, !dbg !38
  %321 = extractelement <32 x float> %274, i64 23, !dbg !38
  %322 = fadd float %321, 0x0000000000000000, !dbg !38
  %323 = extractelement <32 x float> %274, i64 24, !dbg !38
  %324 = fadd float %323, 0x0000000000000000, !dbg !38
  %325 = extractelement <32 x float> %274, i64 25, !dbg !38
  %326 = fadd float %325, 0x0000000000000000, !dbg !38
  %327 = extractelement <32 x float> %274, i64 26, !dbg !38
  %328 = fadd float %327, 0x0000000000000000, !dbg !38
  %329 = extractelement <32 x float> %274, i64 27, !dbg !38
  %330 = fadd float %329, 0x0000000000000000, !dbg !38
  %331 = extractelement <32 x float> %274, i64 28, !dbg !38
  %332 = fadd float %331, 0x0000000000000000, !dbg !38
  %333 = extractelement <32 x float> %274, i64 29, !dbg !38
  %334 = fadd float %333, 0x0000000000000000, !dbg !38
  %335 = extractelement <32 x float> %274, i64 30, !dbg !38
  %336 = fadd float %335, 0x0000000000000000, !dbg !38
  %337 = extractelement <32 x float> %274, i64 31, !dbg !38
  %338 = fadd float %337, 0x0000000000000000, !dbg !38
  %339 = extractvalue [2 x <32 x float>] %39, 1, !dbg !38
  %340 = extractelement <32 x float> %339, i64 0, !dbg !38
  %341 = fadd float %340, 0x0000000000000000, !dbg !38
  %342 = extractelement <32 x float> %339, i64 1, !dbg !38
  %343 = fadd float %342, 0x0000000000000000, !dbg !38
  %344 = extractelement <32 x float> %339, i64 2, !dbg !38
  %345 = fadd float %344, 0x0000000000000000, !dbg !38
  %346 = extractelement <32 x float> %339, i64 3, !dbg !38
  %347 = fadd float %346, 0x0000000000000000, !dbg !38
  %348 = extractelement <32 x float> %339, i64 4, !dbg !38
  %349 = fadd float %348, 0x0000000000000000, !dbg !38
  %350 = extractelement <32 x float> %339, i64 5, !dbg !38
  %351 = fadd float %350, 0x0000000000000000, !dbg !38
  %352 = extractelement <32 x float> %339, i64 6, !dbg !38
  %353 = fadd float %352, 0x0000000000000000, !dbg !38
  %354 = extractelement <32 x float> %339, i64 7, !dbg !38
  %355 = fadd float %354, 0x0000000000000000, !dbg !38
  %356 = extractelement <32 x float> %339, i64 8, !dbg !38
  %357 = fadd float %356, 0x0000000000000000, !dbg !38
  %358 = extractelement <32 x float> %339, i64 9, !dbg !38
  %359 = fadd float %358, 0x0000000000000000, !dbg !38
  %360 = extractelement <32 x float> %339, i64 10, !dbg !38
  %361 = fadd float %360, 0x0000000000000000, !dbg !38
  %362 = extractelement <32 x float> %339, i64 11, !dbg !38
  %363 = fadd float %362, 0x0000000000000000, !dbg !38
  %364 = extractelement <32 x float> %339, i64 12, !dbg !38
  %365 = fadd float %364, 0x0000000000000000, !dbg !38
  %366 = extractelement <32 x float> %339, i64 13, !dbg !38
  %367 = fadd float %366, 0x0000000000000000, !dbg !38
  %368 = extractelement <32 x float> %339, i64 14, !dbg !38
  %369 = fadd float %368, 0x0000000000000000, !dbg !38
  %370 = extractelement <32 x float> %339, i64 15, !dbg !38
  %371 = fadd float %370, 0x0000000000000000, !dbg !38
  %372 = extractelement <32 x float> %339, i64 16, !dbg !38
  %373 = fadd float %372, 0x0000000000000000, !dbg !38
  %374 = extractelement <32 x float> %339, i64 17, !dbg !38
  %375 = fadd float %374, 0x0000000000000000, !dbg !38
  %376 = extractelement <32 x float> %339, i64 18, !dbg !38
  %377 = fadd float %376, 0x0000000000000000, !dbg !38
  %378 = extractelement <32 x float> %339, i64 19, !dbg !38
  %379 = fadd float %378, 0x0000000000000000, !dbg !38
  %380 = extractelement <32 x float> %339, i64 20, !dbg !38
  %381 = fadd float %380, 0x0000000000000000, !dbg !38
  %382 = extractelement <32 x float> %339, i64 21, !dbg !38
  %383 = fadd float %382, 0x0000000000000000, !dbg !38
  %384 = extractelement <32 x float> %339, i64 22, !dbg !38
  %385 = fadd float %384, 0x0000000000000000, !dbg !38
  %386 = extractelement <32 x float> %339, i64 23, !dbg !38
  %387 = fadd float %386, 0x0000000000000000, !dbg !38
  %388 = extractelement <32 x float> %339, i64 24, !dbg !38
  %389 = fadd float %388, 0x0000000000000000, !dbg !38
  %390 = extractelement <32 x float> %339, i64 25, !dbg !38
  %391 = fadd float %390, 0x0000000000000000, !dbg !38
  %392 = extractelement <32 x float> %339, i64 26, !dbg !38
  %393 = fadd float %392, 0x0000000000000000, !dbg !38
  %394 = extractelement <32 x float> %339, i64 27, !dbg !38
  %395 = fadd float %394, 0x0000000000000000, !dbg !38
  %396 = extractelement <32 x float> %339, i64 28, !dbg !38
  %397 = fadd float %396, 0x0000000000000000, !dbg !38
  %398 = extractelement <32 x float> %339, i64 29, !dbg !38
  %399 = fadd float %398, 0x0000000000000000, !dbg !38
  %400 = extractelement <32 x float> %339, i64 30, !dbg !38
  %401 = fadd float %400, 0x0000000000000000, !dbg !38
  %402 = extractelement <32 x float> %339, i64 31, !dbg !38
  %403 = fadd float %402, 0x0000000000000000, !dbg !38
  %404 = insertelement <32 x float> poison, float %276, i64 0, !dbg !38
  %405 = insertelement <32 x float> %404, float %278, i64 1, !dbg !38
  %406 = insertelement <32 x float> %405, float %280, i64 2, !dbg !38
  %407 = insertelement <32 x float> %406, float %282, i64 3, !dbg !38
  %408 = insertelement <32 x float> %407, float %284, i64 4, !dbg !38
  %409 = insertelement <32 x float> %408, float %286, i64 5, !dbg !38
  %410 = insertelement <32 x float> %409, float %288, i64 6, !dbg !38
  %411 = insertelement <32 x float> %410, float %290, i64 7, !dbg !38
  %412 = insertelement <32 x float> %411, float %292, i64 8, !dbg !38
  %413 = insertelement <32 x float> %412, float %294, i64 9, !dbg !38
  %414 = insertelement <32 x float> %413, float %296, i64 10, !dbg !38
  %415 = insertelement <32 x float> %414, float %298, i64 11, !dbg !38
  %416 = insertelement <32 x float> %415, float %300, i64 12, !dbg !38
  %417 = insertelement <32 x float> %416, float %302, i64 13, !dbg !38
  %418 = insertelement <32 x float> %417, float %304, i64 14, !dbg !38
  %419 = insertelement <32 x float> %418, float %306, i64 15, !dbg !38
  %420 = insertelement <32 x float> %419, float %308, i64 16, !dbg !38
  %421 = insertelement <32 x float> %420, float %310, i64 17, !dbg !38
  %422 = insertelement <32 x float> %421, float %312, i64 18, !dbg !38
  %423 = insertelement <32 x float> %422, float %314, i64 19, !dbg !38
  %424 = insertelement <32 x float> %423, float %316, i64 20, !dbg !38
  %425 = insertelement <32 x float> %424, float %318, i64 21, !dbg !38
  %426 = insertelement <32 x float> %425, float %320, i64 22, !dbg !38
  %427 = insertelement <32 x float> %426, float %322, i64 23, !dbg !38
  %428 = insertelement <32 x float> %427, float %324, i64 24, !dbg !38
  %429 = insertelement <32 x float> %428, float %326, i64 25, !dbg !38
  %430 = insertelement <32 x float> %429, float %328, i64 26, !dbg !38
  %431 = insertelement <32 x float> %430, float %330, i64 27, !dbg !38
  %432 = insertelement <32 x float> %431, float %332, i64 28, !dbg !38
  %433 = insertelement <32 x float> %432, float %334, i64 29, !dbg !38
  %434 = insertelement <32 x float> %433, float %336, i64 30, !dbg !38
  %435 = insertelement <32 x float> %434, float %338, i64 31, !dbg !38
  %436 = insertelement <32 x float> poison, float %341, i64 0, !dbg !38
  %437 = insertelement <32 x float> %436, float %343, i64 1, !dbg !38
  %438 = insertelement <32 x float> %437, float %345, i64 2, !dbg !38
  %439 = insertelement <32 x float> %438, float %347, i64 3, !dbg !38
  %440 = insertelement <32 x float> %439, float %349, i64 4, !dbg !38
  %441 = insertelement <32 x float> %440, float %351, i64 5, !dbg !38
  %442 = insertelement <32 x float> %441, float %353, i64 6, !dbg !38
  %443 = insertelement <32 x float> %442, float %355, i64 7, !dbg !38
  %444 = insertelement <32 x float> %443, float %357, i64 8, !dbg !38
  %445 = insertelement <32 x float> %444, float %359, i64 9, !dbg !38
  %446 = insertelement <32 x float> %445, float %361, i64 10, !dbg !38
  %447 = insertelement <32 x float> %446, float %363, i64 11, !dbg !38
  %448 = insertelement <32 x float> %447, float %365, i64 12, !dbg !38
  %449 = insertelement <32 x float> %448, float %367, i64 13, !dbg !38
  %450 = insertelement <32 x float> %449, float %369, i64 14, !dbg !38
  %451 = insertelement <32 x float> %450, float %371, i64 15, !dbg !38
  %452 = insertelement <32 x float> %451, float %373, i64 16, !dbg !38
  %453 = insertelement <32 x float> %452, float %375, i64 17, !dbg !38
  %454 = insertelement <32 x float> %453, float %377, i64 18, !dbg !38
  %455 = insertelement <32 x float> %454, float %379, i64 19, !dbg !38
  %456 = insertelement <32 x float> %455, float %381, i64 20, !dbg !38
  %457 = insertelement <32 x float> %456, float %383, i64 21, !dbg !38
  %458 = insertelement <32 x float> %457, float %385, i64 22, !dbg !38
  %459 = insertelement <32 x float> %458, float %387, i64 23, !dbg !38
  %460 = insertelement <32 x float> %459, float %389, i64 24, !dbg !38
  %461 = insertelement <32 x float> %460, float %391, i64 25, !dbg !38
  %462 = insertelement <32 x float> %461, float %393, i64 26, !dbg !38
  %463 = insertelement <32 x float> %462, float %395, i64 27, !dbg !38
  %464 = insertelement <32 x float> %463, float %397, i64 28, !dbg !38
  %465 = insertelement <32 x float> %464, float %399, i64 29, !dbg !38
  %466 = insertelement <32 x float> %465, float %401, i64 30, !dbg !38
  %467 = insertelement <32 x float> %466, float %403, i64 31, !dbg !38
  br label %767, !dbg !38

468:                                              ; preds = %235
  %469 = add nsw i64 %9, %20, !dbg !38
  %470 = shl nuw nsw i64 %21, 4, !dbg !38
  %471 = add nsw i64 %469, %470, !dbg !38
  %472 = and i32 %19, 240, !dbg !38
  %index.compact = zext nneg i32 %472 to i64, !dbg !38
  %473 = sub nsw i64 %471, %index.compact, !dbg !38
  %474 = and i64 %20, 48, !dbg !38
  %475 = or disjoint i64 %33, %474, !dbg !38
  %476 = add nsw i64 %473, 1, !dbg !47
  %.not4 = icmp ugt i64 %476, %475, !dbg !48
  %477 = select i1 %.not4, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %478 = extractvalue [2 x <32 x float>] %39, 0, !dbg !38
  %479 = extractelement <32 x float> %478, i64 0, !dbg !38
  %480 = fadd float %479, %477, !dbg !38
  %481 = or disjoint i64 %475, 1, !dbg !38
  %.not5 = icmp ugt i64 %476, %481, !dbg !48
  %482 = select i1 %.not5, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %483 = extractelement <32 x float> %478, i64 1, !dbg !38
  %484 = fadd float %483, %482, !dbg !38
  %485 = or disjoint i64 %475, 2, !dbg !38
  %.not6 = icmp ugt i64 %476, %485, !dbg !48
  %486 = select i1 %.not6, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %487 = extractelement <32 x float> %478, i64 2, !dbg !38
  %488 = fadd float %487, %486, !dbg !38
  %489 = or disjoint i64 %475, 3, !dbg !38
  %.not7 = icmp ugt i64 %476, %489, !dbg !48
  %490 = select i1 %.not7, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %491 = extractelement <32 x float> %478, i64 3, !dbg !38
  %492 = fadd float %491, %490, !dbg !38
  %493 = or disjoint i64 %475, 4, !dbg !38
  %.not8 = icmp ugt i64 %476, %493, !dbg !48
  %494 = select i1 %.not8, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %495 = extractelement <32 x float> %478, i64 4, !dbg !38
  %496 = fadd float %495, %494, !dbg !38
  %497 = or disjoint i64 %475, 5, !dbg !38
  %.not9 = icmp ugt i64 %476, %497, !dbg !48
  %498 = select i1 %.not9, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %499 = extractelement <32 x float> %478, i64 5, !dbg !38
  %500 = fadd float %499, %498, !dbg !38
  %501 = or disjoint i64 %475, 6, !dbg !38
  %.not10 = icmp ugt i64 %476, %501, !dbg !48
  %502 = select i1 %.not10, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %503 = extractelement <32 x float> %478, i64 6, !dbg !38
  %504 = fadd float %503, %502, !dbg !38
  %505 = or disjoint i64 %475, 7, !dbg !38
  %.not11 = icmp ugt i64 %476, %505, !dbg !48
  %506 = select i1 %.not11, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %507 = extractelement <32 x float> %478, i64 7, !dbg !38
  %508 = fadd float %507, %506, !dbg !38
  %509 = or disjoint i64 %475, 8, !dbg !38
  %.not12 = icmp ugt i64 %476, %509, !dbg !48
  %510 = select i1 %.not12, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %511 = extractelement <32 x float> %478, i64 8, !dbg !38
  %512 = fadd float %511, %510, !dbg !38
  %513 = or disjoint i64 %475, 9, !dbg !38
  %.not13 = icmp ugt i64 %476, %513, !dbg !48
  %514 = select i1 %.not13, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %515 = extractelement <32 x float> %478, i64 9, !dbg !38
  %516 = fadd float %515, %514, !dbg !38
  %517 = or disjoint i64 %475, 10, !dbg !38
  %.not14 = icmp ugt i64 %476, %517, !dbg !48
  %518 = select i1 %.not14, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %519 = extractelement <32 x float> %478, i64 10, !dbg !38
  %520 = fadd float %519, %518, !dbg !38
  %521 = or disjoint i64 %475, 11, !dbg !38
  %.not15 = icmp ugt i64 %476, %521, !dbg !48
  %522 = select i1 %.not15, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %523 = extractelement <32 x float> %478, i64 11, !dbg !38
  %524 = fadd float %523, %522, !dbg !38
  %525 = or disjoint i64 %475, 12, !dbg !38
  %.not16 = icmp ugt i64 %476, %525, !dbg !48
  %526 = select i1 %.not16, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %527 = extractelement <32 x float> %478, i64 12, !dbg !38
  %528 = fadd float %527, %526, !dbg !38
  %529 = or disjoint i64 %475, 13, !dbg !38
  %.not17 = icmp ugt i64 %476, %529, !dbg !48
  %530 = select i1 %.not17, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %531 = extractelement <32 x float> %478, i64 13, !dbg !38
  %532 = fadd float %531, %530, !dbg !38
  %533 = or disjoint i64 %475, 14, !dbg !38
  %.not18 = icmp ugt i64 %476, %533, !dbg !48
  %534 = select i1 %.not18, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %535 = extractelement <32 x float> %478, i64 14, !dbg !38
  %536 = fadd float %535, %534, !dbg !38
  %537 = or disjoint i64 %475, 15, !dbg !38
  %.not19 = icmp ugt i64 %476, %537, !dbg !48
  %538 = select i1 %.not19, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %539 = extractelement <32 x float> %478, i64 15, !dbg !38
  %540 = fadd float %539, %538, !dbg !38
  %541 = or disjoint i64 %475, 64, !dbg !38
  %.not20 = icmp ugt i64 %476, %541, !dbg !48
  %542 = select i1 %.not20, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %543 = extractelement <32 x float> %478, i64 16, !dbg !38
  %544 = fadd float %543, %542, !dbg !38
  %545 = or disjoint i64 %475, 65, !dbg !38
  %.not21 = icmp ugt i64 %476, %545, !dbg !48
  %546 = select i1 %.not21, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %547 = extractelement <32 x float> %478, i64 17, !dbg !38
  %548 = fadd float %547, %546, !dbg !38
  %549 = or disjoint i64 %475, 66, !dbg !38
  %.not22 = icmp ugt i64 %476, %549, !dbg !48
  %550 = select i1 %.not22, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %551 = extractelement <32 x float> %478, i64 18, !dbg !38
  %552 = fadd float %551, %550, !dbg !38
  %553 = or disjoint i64 %475, 67, !dbg !38
  %.not23 = icmp ugt i64 %476, %553, !dbg !48
  %554 = select i1 %.not23, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %555 = extractelement <32 x float> %478, i64 19, !dbg !38
  %556 = fadd float %555, %554, !dbg !38
  %557 = or disjoint i64 %475, 68, !dbg !38
  %.not24 = icmp ugt i64 %476, %557, !dbg !48
  %558 = select i1 %.not24, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %559 = extractelement <32 x float> %478, i64 20, !dbg !38
  %560 = fadd float %559, %558, !dbg !38
  %561 = or disjoint i64 %475, 69, !dbg !38
  %.not25 = icmp ugt i64 %476, %561, !dbg !48
  %562 = select i1 %.not25, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %563 = extractelement <32 x float> %478, i64 21, !dbg !38
  %564 = fadd float %563, %562, !dbg !38
  %565 = or disjoint i64 %475, 70, !dbg !38
  %.not26 = icmp ugt i64 %476, %565, !dbg !48
  %566 = select i1 %.not26, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %567 = extractelement <32 x float> %478, i64 22, !dbg !38
  %568 = fadd float %567, %566, !dbg !38
  %569 = or disjoint i64 %475, 71, !dbg !38
  %.not27 = icmp ugt i64 %476, %569, !dbg !48
  %570 = select i1 %.not27, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %571 = extractelement <32 x float> %478, i64 23, !dbg !38
  %572 = fadd float %571, %570, !dbg !38
  %573 = or disjoint i64 %475, 72, !dbg !38
  %.not28 = icmp ugt i64 %476, %573, !dbg !48
  %574 = select i1 %.not28, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %575 = extractelement <32 x float> %478, i64 24, !dbg !38
  %576 = fadd float %575, %574, !dbg !38
  %577 = or disjoint i64 %475, 73, !dbg !38
  %.not29 = icmp ugt i64 %476, %577, !dbg !48
  %578 = select i1 %.not29, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %579 = extractelement <32 x float> %478, i64 25, !dbg !38
  %580 = fadd float %579, %578, !dbg !38
  %581 = or disjoint i64 %475, 74, !dbg !38
  %.not30 = icmp ugt i64 %476, %581, !dbg !48
  %582 = select i1 %.not30, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %583 = extractelement <32 x float> %478, i64 26, !dbg !38
  %584 = fadd float %583, %582, !dbg !38
  %585 = or disjoint i64 %475, 75, !dbg !38
  %.not31 = icmp ugt i64 %476, %585, !dbg !48
  %586 = select i1 %.not31, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %587 = extractelement <32 x float> %478, i64 27, !dbg !38
  %588 = fadd float %587, %586, !dbg !38
  %589 = or disjoint i64 %475, 76, !dbg !38
  %.not32 = icmp ugt i64 %476, %589, !dbg !48
  %590 = select i1 %.not32, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %591 = extractelement <32 x float> %478, i64 28, !dbg !38
  %592 = fadd float %591, %590, !dbg !38
  %593 = or disjoint i64 %475, 77, !dbg !38
  %.not33 = icmp ugt i64 %476, %593, !dbg !48
  %594 = select i1 %.not33, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %595 = extractelement <32 x float> %478, i64 29, !dbg !38
  %596 = fadd float %595, %594, !dbg !38
  %597 = or disjoint i64 %475, 78, !dbg !38
  %.not34 = icmp ugt i64 %476, %597, !dbg !48
  %598 = select i1 %.not34, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %599 = extractelement <32 x float> %478, i64 30, !dbg !38
  %600 = fadd float %599, %598, !dbg !38
  %601 = or disjoint i64 %475, 79, !dbg !38
  %.not35 = icmp ugt i64 %476, %601, !dbg !48
  %602 = select i1 %.not35, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %603 = extractelement <32 x float> %478, i64 31, !dbg !38
  %604 = fadd float %603, %602, !dbg !38
  %605 = add nsw i64 %473, 65, !dbg !47
  %.not36 = icmp ugt i64 %605, %475, !dbg !48
  %606 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %607 = extractvalue [2 x <32 x float>] %39, 1, !dbg !38
  %608 = extractelement <32 x float> %607, i64 0, !dbg !38
  %609 = fadd float %608, %606, !dbg !38
  %.not37 = icmp ugt i64 %605, %481, !dbg !48
  %610 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %611 = extractelement <32 x float> %607, i64 1, !dbg !38
  %612 = fadd float %611, %610, !dbg !38
  %.not38 = icmp ugt i64 %605, %485, !dbg !48
  %613 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %614 = extractelement <32 x float> %607, i64 2, !dbg !38
  %615 = fadd float %614, %613, !dbg !38
  %.not39 = icmp ugt i64 %605, %489, !dbg !48
  %616 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %617 = extractelement <32 x float> %607, i64 3, !dbg !38
  %618 = fadd float %617, %616, !dbg !38
  %.not40 = icmp ugt i64 %605, %493, !dbg !48
  %619 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %620 = extractelement <32 x float> %607, i64 4, !dbg !38
  %621 = fadd float %620, %619, !dbg !38
  %.not41 = icmp ugt i64 %605, %497, !dbg !48
  %622 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %623 = extractelement <32 x float> %607, i64 5, !dbg !38
  %624 = fadd float %623, %622, !dbg !38
  %.not42 = icmp ugt i64 %605, %501, !dbg !48
  %625 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %626 = extractelement <32 x float> %607, i64 6, !dbg !38
  %627 = fadd float %626, %625, !dbg !38
  %.not43 = icmp ugt i64 %605, %505, !dbg !48
  %628 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %629 = extractelement <32 x float> %607, i64 7, !dbg !38
  %630 = fadd float %629, %628, !dbg !38
  %.not44 = icmp ugt i64 %605, %509, !dbg !48
  %631 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %632 = extractelement <32 x float> %607, i64 8, !dbg !38
  %633 = fadd float %632, %631, !dbg !38
  %.not45 = icmp ugt i64 %605, %513, !dbg !48
  %634 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %635 = extractelement <32 x float> %607, i64 9, !dbg !38
  %636 = fadd float %635, %634, !dbg !38
  %.not46 = icmp ugt i64 %605, %517, !dbg !48
  %637 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %638 = extractelement <32 x float> %607, i64 10, !dbg !38
  %639 = fadd float %638, %637, !dbg !38
  %.not47 = icmp ugt i64 %605, %521, !dbg !48
  %640 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %641 = extractelement <32 x float> %607, i64 11, !dbg !38
  %642 = fadd float %641, %640, !dbg !38
  %.not48 = icmp ugt i64 %605, %525, !dbg !48
  %643 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %644 = extractelement <32 x float> %607, i64 12, !dbg !38
  %645 = fadd float %644, %643, !dbg !38
  %.not49 = icmp ugt i64 %605, %529, !dbg !48
  %646 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %647 = extractelement <32 x float> %607, i64 13, !dbg !38
  %648 = fadd float %647, %646, !dbg !38
  %.not50 = icmp ugt i64 %605, %533, !dbg !48
  %649 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %650 = extractelement <32 x float> %607, i64 14, !dbg !38
  %651 = fadd float %650, %649, !dbg !38
  %.not51 = icmp ugt i64 %605, %537, !dbg !48
  %652 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %653 = extractelement <32 x float> %607, i64 15, !dbg !38
  %654 = fadd float %653, %652, !dbg !38
  %.not52 = icmp ugt i64 %605, %541, !dbg !48
  %655 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %656 = extractelement <32 x float> %607, i64 16, !dbg !38
  %657 = fadd float %656, %655, !dbg !38
  %.not53 = icmp ugt i64 %605, %545, !dbg !48
  %658 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %659 = extractelement <32 x float> %607, i64 17, !dbg !38
  %660 = fadd float %659, %658, !dbg !38
  %.not54 = icmp ugt i64 %605, %549, !dbg !48
  %661 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %662 = extractelement <32 x float> %607, i64 18, !dbg !38
  %663 = fadd float %662, %661, !dbg !38
  %.not55 = icmp ugt i64 %605, %553, !dbg !48
  %664 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %665 = extractelement <32 x float> %607, i64 19, !dbg !38
  %666 = fadd float %665, %664, !dbg !38
  %.not56 = icmp ugt i64 %605, %557, !dbg !48
  %667 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %668 = extractelement <32 x float> %607, i64 20, !dbg !38
  %669 = fadd float %668, %667, !dbg !38
  %.not57 = icmp ugt i64 %605, %561, !dbg !48
  %670 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %671 = extractelement <32 x float> %607, i64 21, !dbg !38
  %672 = fadd float %671, %670, !dbg !38
  %.not58 = icmp ugt i64 %605, %565, !dbg !48
  %673 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %674 = extractelement <32 x float> %607, i64 22, !dbg !38
  %675 = fadd float %674, %673, !dbg !38
  %.not59 = icmp ugt i64 %605, %569, !dbg !48
  %676 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %677 = extractelement <32 x float> %607, i64 23, !dbg !38
  %678 = fadd float %677, %676, !dbg !38
  %.not60 = icmp ugt i64 %605, %573, !dbg !48
  %679 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %680 = extractelement <32 x float> %607, i64 24, !dbg !38
  %681 = fadd float %680, %679, !dbg !38
  %.not61 = icmp ugt i64 %605, %577, !dbg !48
  %682 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %683 = extractelement <32 x float> %607, i64 25, !dbg !38
  %684 = fadd float %683, %682, !dbg !38
  %.not62 = icmp ugt i64 %605, %581, !dbg !48
  %685 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %686 = extractelement <32 x float> %607, i64 26, !dbg !38
  %687 = fadd float %686, %685, !dbg !38
  %.not63 = icmp ugt i64 %605, %585, !dbg !48
  %688 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %689 = extractelement <32 x float> %607, i64 27, !dbg !38
  %690 = fadd float %689, %688, !dbg !38
  %.not64 = icmp ugt i64 %605, %589, !dbg !48
  %691 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %692 = extractelement <32 x float> %607, i64 28, !dbg !38
  %693 = fadd float %692, %691, !dbg !38
  %.not65 = icmp ugt i64 %605, %593, !dbg !48
  %694 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %695 = extractelement <32 x float> %607, i64 29, !dbg !38
  %696 = fadd float %695, %694, !dbg !38
  %.not66 = icmp ugt i64 %605, %597, !dbg !48
  %697 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %698 = extractelement <32 x float> %607, i64 30, !dbg !38
  %699 = fadd float %698, %697, !dbg !38
  %.not67 = icmp ugt i64 %605, %601, !dbg !48
  %700 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !49
  %701 = extractelement <32 x float> %607, i64 31, !dbg !38
  %702 = fadd float %701, %700, !dbg !38
  %703 = insertelement <32 x float> poison, float %480, i64 0, !dbg !38
  %704 = insertelement <32 x float> %703, float %484, i64 1, !dbg !38
  %705 = insertelement <32 x float> %704, float %488, i64 2, !dbg !38
  %706 = insertelement <32 x float> %705, float %492, i64 3, !dbg !38
  %707 = insertelement <32 x float> %706, float %496, i64 4, !dbg !38
  %708 = insertelement <32 x float> %707, float %500, i64 5, !dbg !38
  %709 = insertelement <32 x float> %708, float %504, i64 6, !dbg !38
  %710 = insertelement <32 x float> %709, float %508, i64 7, !dbg !38
  %711 = insertelement <32 x float> %710, float %512, i64 8, !dbg !38
  %712 = insertelement <32 x float> %711, float %516, i64 9, !dbg !38
  %713 = insertelement <32 x float> %712, float %520, i64 10, !dbg !38
  %714 = insertelement <32 x float> %713, float %524, i64 11, !dbg !38
  %715 = insertelement <32 x float> %714, float %528, i64 12, !dbg !38
  %716 = insertelement <32 x float> %715, float %532, i64 13, !dbg !38
  %717 = insertelement <32 x float> %716, float %536, i64 14, !dbg !38
  %718 = insertelement <32 x float> %717, float %540, i64 15, !dbg !38
  %719 = insertelement <32 x float> %718, float %544, i64 16, !dbg !38
  %720 = insertelement <32 x float> %719, float %548, i64 17, !dbg !38
  %721 = insertelement <32 x float> %720, float %552, i64 18, !dbg !38
  %722 = insertelement <32 x float> %721, float %556, i64 19, !dbg !38
  %723 = insertelement <32 x float> %722, float %560, i64 20, !dbg !38
  %724 = insertelement <32 x float> %723, float %564, i64 21, !dbg !38
  %725 = insertelement <32 x float> %724, float %568, i64 22, !dbg !38
  %726 = insertelement <32 x float> %725, float %572, i64 23, !dbg !38
  %727 = insertelement <32 x float> %726, float %576, i64 24, !dbg !38
  %728 = insertelement <32 x float> %727, float %580, i64 25, !dbg !38
  %729 = insertelement <32 x float> %728, float %584, i64 26, !dbg !38
  %730 = insertelement <32 x float> %729, float %588, i64 27, !dbg !38
  %731 = insertelement <32 x float> %730, float %592, i64 28, !dbg !38
  %732 = insertelement <32 x float> %731, float %596, i64 29, !dbg !38
  %733 = insertelement <32 x float> %732, float %600, i64 30, !dbg !38
  %734 = insertelement <32 x float> %733, float %604, i64 31, !dbg !38
  %735 = insertelement <32 x float> poison, float %609, i64 0, !dbg !38
  %736 = insertelement <32 x float> %735, float %612, i64 1, !dbg !38
  %737 = insertelement <32 x float> %736, float %615, i64 2, !dbg !38
  %738 = insertelement <32 x float> %737, float %618, i64 3, !dbg !38
  %739 = insertelement <32 x float> %738, float %621, i64 4, !dbg !38
  %740 = insertelement <32 x float> %739, float %624, i64 5, !dbg !38
  %741 = insertelement <32 x float> %740, float %627, i64 6, !dbg !38
  %742 = insertelement <32 x float> %741, float %630, i64 7, !dbg !38
  %743 = insertelement <32 x float> %742, float %633, i64 8, !dbg !38
  %744 = insertelement <32 x float> %743, float %636, i64 9, !dbg !38
  %745 = insertelement <32 x float> %744, float %639, i64 10, !dbg !38
  %746 = insertelement <32 x float> %745, float %642, i64 11, !dbg !38
  %747 = insertelement <32 x float> %746, float %645, i64 12, !dbg !38
  %748 = insertelement <32 x float> %747, float %648, i64 13, !dbg !38
  %749 = insertelement <32 x float> %748, float %651, i64 14, !dbg !38
  %750 = insertelement <32 x float> %749, float %654, i64 15, !dbg !38
  %751 = insertelement <32 x float> %750, float %657, i64 16, !dbg !38
  %752 = insertelement <32 x float> %751, float %660, i64 17, !dbg !38
  %753 = insertelement <32 x float> %752, float %663, i64 18, !dbg !38
  %754 = insertelement <32 x float> %753, float %666, i64 19, !dbg !38
  %755 = insertelement <32 x float> %754, float %669, i64 20, !dbg !38
  %756 = insertelement <32 x float> %755, float %672, i64 21, !dbg !38
  %757 = insertelement <32 x float> %756, float %675, i64 22, !dbg !38
  %758 = insertelement <32 x float> %757, float %678, i64 23, !dbg !38
  %759 = insertelement <32 x float> %758, float %681, i64 24, !dbg !38
  %760 = insertelement <32 x float> %759, float %684, i64 25, !dbg !38
  %761 = insertelement <32 x float> %760, float %687, i64 26, !dbg !38
  %762 = insertelement <32 x float> %761, float %690, i64 27, !dbg !38
  %763 = insertelement <32 x float> %762, float %693, i64 28, !dbg !38
  %764 = insertelement <32 x float> %763, float %696, i64 29, !dbg !38
  %765 = insertelement <32 x float> %764, float %699, i64 30, !dbg !38
  %766 = insertelement <32 x float> %765, float %702, i64 31, !dbg !38
  br label %767, !dbg !38

767:                                              ; preds = %468, %273
  %.pn70 = phi <32 x float> [ %734, %468 ], [ %435, %273 ]
  %.pn68 = phi <32 x float> [ %766, %468 ], [ %467, %273 ]
  br label %768, !dbg !38

768:                                              ; preds = %767
  %769 = extractelement <32 x float> %.pn70, i64 0, !dbg !50
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !50
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !50
  %772 = extractelement <32 x float> %.pn70, i64 1, !dbg !50
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !50
  %774 = fadd float %770, %773, !dbg !51
  %775 = insertelement <4 x float> %771, float %773, i64 1, !dbg !50
  %776 = extractelement <32 x float> %.pn70, i64 2, !dbg !50
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !50
  %778 = fadd float %774, %777, !dbg !51
  %779 = insertelement <4 x float> %775, float %777, i64 2, !dbg !50
  %780 = extractelement <32 x float> %.pn70, i64 3, !dbg !50
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !50
  %782 = fadd float %778, %781, !dbg !51
  %783 = insertelement <4 x float> %779, float %781, i64 3, !dbg !50
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #9, !dbg !52, !frisk.register_lifetimes !29
  %784 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !52
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #9, !dbg !50, !frisk.register_lifetimes !29
  %785 = extractelement <32 x float> %.pn70, i64 4, !dbg !50
  %786 = call float @llvm.amdgcn.exp2.f32(float %785), !dbg !50
  %787 = fadd float %782, %786, !dbg !51
  %788 = insertelement <4 x float> poison, float %786, i64 0, !dbg !50
  %789 = extractelement <32 x float> %.pn70, i64 5, !dbg !50
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !50
  %791 = fadd float %787, %790, !dbg !51
  %792 = insertelement <4 x float> %788, float %790, i64 1, !dbg !50
  %793 = extractelement <32 x float> %.pn70, i64 6, !dbg !50
  %794 = call float @llvm.amdgcn.exp2.f32(float %793), !dbg !50
  %795 = fadd float %791, %794, !dbg !51
  %796 = insertelement <4 x float> %792, float %794, i64 2, !dbg !50
  %797 = extractelement <32 x float> %.pn70, i64 7, !dbg !50
  %798 = call float @llvm.amdgcn.exp2.f32(float %797), !dbg !50
  %799 = fadd float %795, %798, !dbg !51
  %800 = insertelement <4 x float> %796, float %798, i64 3, !dbg !50
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #9, !dbg !52, !frisk.register_lifetimes !29
  %801 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !52
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #9, !dbg !50, !frisk.register_lifetimes !29
  %802 = extractelement <32 x float> %.pn70, i64 8, !dbg !50
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !50
  %804 = fadd float %799, %803, !dbg !51
  %805 = insertelement <4 x float> poison, float %803, i64 0, !dbg !50
  %806 = extractelement <32 x float> %.pn70, i64 9, !dbg !50
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !50
  %808 = fadd float %804, %807, !dbg !51
  %809 = insertelement <4 x float> %805, float %807, i64 1, !dbg !50
  %810 = extractelement <32 x float> %.pn70, i64 10, !dbg !50
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !50
  %812 = fadd float %808, %811, !dbg !51
  %813 = insertelement <4 x float> %809, float %811, i64 2, !dbg !50
  %814 = extractelement <32 x float> %.pn70, i64 11, !dbg !50
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !50
  %816 = fadd float %812, %815, !dbg !51
  %817 = insertelement <4 x float> %813, float %815, i64 3, !dbg !50
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !52, !frisk.register_lifetimes !29
  %818 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !52
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !50, !frisk.register_lifetimes !29
  %819 = extractelement <32 x float> %.pn70, i64 12, !dbg !50
  %820 = call float @llvm.amdgcn.exp2.f32(float %819), !dbg !50
  %821 = fadd float %816, %820, !dbg !51
  %822 = insertelement <4 x float> poison, float %820, i64 0, !dbg !50
  %823 = extractelement <32 x float> %.pn70, i64 13, !dbg !50
  %824 = call float @llvm.amdgcn.exp2.f32(float %823), !dbg !50
  %825 = fadd float %821, %824, !dbg !51
  %826 = insertelement <4 x float> %822, float %824, i64 1, !dbg !50
  %827 = extractelement <32 x float> %.pn70, i64 14, !dbg !50
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !50
  %829 = fadd float %825, %828, !dbg !51
  %830 = insertelement <4 x float> %826, float %828, i64 2, !dbg !50
  %831 = extractelement <32 x float> %.pn70, i64 15, !dbg !50
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !50
  %833 = fadd float %829, %832, !dbg !51
  %834 = insertelement <4 x float> %830, float %832, i64 3, !dbg !50
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !52, !frisk.register_lifetimes !29
  %835 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !52
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !50, !frisk.register_lifetimes !29
  %836 = extractelement <32 x float> %.pn70, i64 16, !dbg !50
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !50
  %838 = fadd float %833, %837, !dbg !51
  %839 = insertelement <4 x float> poison, float %837, i64 0, !dbg !50
  %840 = extractelement <32 x float> %.pn70, i64 17, !dbg !50
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !50
  %842 = fadd float %838, %841, !dbg !51
  %843 = insertelement <4 x float> %839, float %841, i64 1, !dbg !50
  %844 = extractelement <32 x float> %.pn70, i64 18, !dbg !50
  %845 = call float @llvm.amdgcn.exp2.f32(float %844), !dbg !50
  %846 = fadd float %842, %845, !dbg !51
  %847 = insertelement <4 x float> %843, float %845, i64 2, !dbg !50
  %848 = extractelement <32 x float> %.pn70, i64 19, !dbg !50
  %849 = call float @llvm.amdgcn.exp2.f32(float %848), !dbg !50
  %850 = fadd float %846, %849, !dbg !51
  %851 = insertelement <4 x float> %847, float %849, i64 3, !dbg !50
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #9, !dbg !52, !frisk.register_lifetimes !29
  %852 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !52
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #9, !dbg !50, !frisk.register_lifetimes !29
  %853 = extractelement <32 x float> %.pn70, i64 20, !dbg !50
  %854 = call float @llvm.amdgcn.exp2.f32(float %853), !dbg !50
  %855 = fadd float %850, %854, !dbg !51
  %856 = insertelement <4 x float> poison, float %854, i64 0, !dbg !50
  %857 = extractelement <32 x float> %.pn70, i64 21, !dbg !50
  %858 = call float @llvm.amdgcn.exp2.f32(float %857), !dbg !50
  %859 = fadd float %855, %858, !dbg !51
  %860 = insertelement <4 x float> %856, float %858, i64 1, !dbg !50
  %861 = extractelement <32 x float> %.pn70, i64 22, !dbg !50
  %862 = call float @llvm.amdgcn.exp2.f32(float %861), !dbg !50
  %863 = fadd float %859, %862, !dbg !51
  %864 = insertelement <4 x float> %860, float %862, i64 2, !dbg !50
  %865 = extractelement <32 x float> %.pn70, i64 23, !dbg !50
  %866 = call float @llvm.amdgcn.exp2.f32(float %865), !dbg !50
  %867 = fadd float %863, %866, !dbg !51
  %868 = insertelement <4 x float> %864, float %866, i64 3, !dbg !50
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %868) #9, !dbg !52, !frisk.register_lifetimes !29
  %869 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !52
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %869) #9, !dbg !50, !frisk.register_lifetimes !29
  %870 = extractelement <32 x float> %.pn70, i64 24, !dbg !50
  %871 = call float @llvm.amdgcn.exp2.f32(float %870), !dbg !50
  %872 = fadd float %867, %871, !dbg !51
  %873 = insertelement <4 x float> poison, float %871, i64 0, !dbg !50
  %874 = extractelement <32 x float> %.pn70, i64 25, !dbg !50
  %875 = call float @llvm.amdgcn.exp2.f32(float %874), !dbg !50
  %876 = fadd float %872, %875, !dbg !51
  %877 = insertelement <4 x float> %873, float %875, i64 1, !dbg !50
  %878 = extractelement <32 x float> %.pn70, i64 26, !dbg !50
  %879 = call float @llvm.amdgcn.exp2.f32(float %878), !dbg !50
  %880 = fadd float %876, %879, !dbg !51
  %881 = insertelement <4 x float> %877, float %879, i64 2, !dbg !50
  %882 = extractelement <32 x float> %.pn70, i64 27, !dbg !50
  %883 = call float @llvm.amdgcn.exp2.f32(float %882), !dbg !50
  %884 = fadd float %880, %883, !dbg !51
  %885 = insertelement <4 x float> %881, float %883, i64 3, !dbg !50
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %885) #9, !dbg !52, !frisk.register_lifetimes !29
  %886 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !52
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %886) #9, !dbg !50, !frisk.register_lifetimes !29
  %887 = extractelement <32 x float> %.pn70, i64 28, !dbg !50
  %888 = call float @llvm.amdgcn.exp2.f32(float %887), !dbg !50
  %889 = fadd float %884, %888, !dbg !51
  %890 = insertelement <4 x float> poison, float %888, i64 0, !dbg !50
  %891 = extractelement <32 x float> %.pn70, i64 29, !dbg !50
  %892 = call float @llvm.amdgcn.exp2.f32(float %891), !dbg !50
  %893 = fadd float %889, %892, !dbg !51
  %894 = insertelement <4 x float> %890, float %892, i64 1, !dbg !50
  %895 = extractelement <32 x float> %.pn70, i64 30, !dbg !50
  %896 = call float @llvm.amdgcn.exp2.f32(float %895), !dbg !50
  %897 = fadd float %893, %896, !dbg !51
  %898 = insertelement <4 x float> %894, float %896, i64 2, !dbg !50
  %899 = extractelement <32 x float> %.pn70, i64 31, !dbg !50
  %900 = call float @llvm.amdgcn.exp2.f32(float %899), !dbg !50
  %901 = fadd float %897, %900, !dbg !51
  %902 = insertelement <4 x float> %898, float %900, i64 3, !dbg !50
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %902) #9, !dbg !52, !frisk.register_lifetimes !29
  %903 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !52
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %903) #9, !dbg !50, !frisk.register_lifetimes !29
  %904 = extractelement <32 x float> %.pn68, i64 0, !dbg !50
  %905 = call float @llvm.amdgcn.exp2.f32(float %904), !dbg !50
  %906 = insertelement <4 x float> poison, float %905, i64 0, !dbg !50
  %907 = extractelement <32 x float> %.pn68, i64 1, !dbg !50
  %908 = call float @llvm.amdgcn.exp2.f32(float %907), !dbg !50
  %909 = fadd float %905, %908, !dbg !51
  %910 = insertelement <4 x float> %906, float %908, i64 1, !dbg !50
  %911 = extractelement <32 x float> %.pn68, i64 2, !dbg !50
  %912 = call float @llvm.amdgcn.exp2.f32(float %911), !dbg !50
  %913 = fadd float %909, %912, !dbg !51
  %914 = insertelement <4 x float> %910, float %912, i64 2, !dbg !50
  %915 = extractelement <32 x float> %.pn68, i64 3, !dbg !50
  %916 = call float @llvm.amdgcn.exp2.f32(float %915), !dbg !50
  %917 = fadd float %913, %916, !dbg !51
  %918 = insertelement <4 x float> %914, float %916, i64 3, !dbg !50
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %918) #9, !dbg !52, !frisk.register_lifetimes !29
  %919 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !52
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %919) #9, !dbg !50, !frisk.register_lifetimes !29
  %920 = extractelement <32 x float> %.pn68, i64 4, !dbg !50
  %921 = call float @llvm.amdgcn.exp2.f32(float %920), !dbg !50
  %922 = fadd float %917, %921, !dbg !51
  %923 = insertelement <4 x float> poison, float %921, i64 0, !dbg !50
  %924 = extractelement <32 x float> %.pn68, i64 5, !dbg !50
  %925 = call float @llvm.amdgcn.exp2.f32(float %924), !dbg !50
  %926 = fadd float %922, %925, !dbg !51
  %927 = insertelement <4 x float> %923, float %925, i64 1, !dbg !50
  %928 = extractelement <32 x float> %.pn68, i64 6, !dbg !50
  %929 = call float @llvm.amdgcn.exp2.f32(float %928), !dbg !50
  %930 = fadd float %926, %929, !dbg !51
  %931 = insertelement <4 x float> %927, float %929, i64 2, !dbg !50
  %932 = extractelement <32 x float> %.pn68, i64 7, !dbg !50
  %933 = call float @llvm.amdgcn.exp2.f32(float %932), !dbg !50
  %934 = fadd float %930, %933, !dbg !51
  %935 = insertelement <4 x float> %931, float %933, i64 3, !dbg !50
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %935) #9, !dbg !52, !frisk.register_lifetimes !29
  %936 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !52
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %936) #9, !dbg !50, !frisk.register_lifetimes !29
  %937 = extractelement <32 x float> %.pn68, i64 8, !dbg !50
  %938 = call float @llvm.amdgcn.exp2.f32(float %937), !dbg !50
  %939 = fadd float %934, %938, !dbg !51
  %940 = insertelement <4 x float> poison, float %938, i64 0, !dbg !50
  %941 = extractelement <32 x float> %.pn68, i64 9, !dbg !50
  %942 = call float @llvm.amdgcn.exp2.f32(float %941), !dbg !50
  %943 = fadd float %939, %942, !dbg !51
  %944 = insertelement <4 x float> %940, float %942, i64 1, !dbg !50
  %945 = extractelement <32 x float> %.pn68, i64 10, !dbg !50
  %946 = call float @llvm.amdgcn.exp2.f32(float %945), !dbg !50
  %947 = fadd float %943, %946, !dbg !51
  %948 = insertelement <4 x float> %944, float %946, i64 2, !dbg !50
  %949 = extractelement <32 x float> %.pn68, i64 11, !dbg !50
  %950 = call float @llvm.amdgcn.exp2.f32(float %949), !dbg !50
  %951 = fadd float %947, %950, !dbg !51
  %952 = insertelement <4 x float> %948, float %950, i64 3, !dbg !50
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %952) #9, !dbg !52, !frisk.register_lifetimes !29
  %953 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !52
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %953) #9, !dbg !50, !frisk.register_lifetimes !29
  %954 = extractelement <32 x float> %.pn68, i64 12, !dbg !50
  %955 = call float @llvm.amdgcn.exp2.f32(float %954), !dbg !50
  %956 = fadd float %951, %955, !dbg !51
  %957 = insertelement <4 x float> poison, float %955, i64 0, !dbg !50
  %958 = extractelement <32 x float> %.pn68, i64 13, !dbg !50
  %959 = call float @llvm.amdgcn.exp2.f32(float %958), !dbg !50
  %960 = fadd float %956, %959, !dbg !51
  %961 = insertelement <4 x float> %957, float %959, i64 1, !dbg !50
  %962 = extractelement <32 x float> %.pn68, i64 14, !dbg !50
  %963 = call float @llvm.amdgcn.exp2.f32(float %962), !dbg !50
  %964 = fadd float %960, %963, !dbg !51
  %965 = insertelement <4 x float> %961, float %963, i64 2, !dbg !50
  %966 = extractelement <32 x float> %.pn68, i64 15, !dbg !50
  %967 = call float @llvm.amdgcn.exp2.f32(float %966), !dbg !50
  %968 = fadd float %964, %967, !dbg !51
  %969 = insertelement <4 x float> %965, float %967, i64 3, !dbg !50
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %969) #9, !dbg !52, !frisk.register_lifetimes !29
  %970 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !52
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %970) #9, !dbg !50, !frisk.register_lifetimes !29
  %971 = extractelement <32 x float> %.pn68, i64 16, !dbg !50
  %972 = call float @llvm.amdgcn.exp2.f32(float %971), !dbg !50
  %973 = fadd float %968, %972, !dbg !51
  %974 = insertelement <4 x float> poison, float %972, i64 0, !dbg !50
  %975 = extractelement <32 x float> %.pn68, i64 17, !dbg !50
  %976 = call float @llvm.amdgcn.exp2.f32(float %975), !dbg !50
  %977 = fadd float %973, %976, !dbg !51
  %978 = insertelement <4 x float> %974, float %976, i64 1, !dbg !50
  %979 = extractelement <32 x float> %.pn68, i64 18, !dbg !50
  %980 = call float @llvm.amdgcn.exp2.f32(float %979), !dbg !50
  %981 = fadd float %977, %980, !dbg !51
  %982 = insertelement <4 x float> %978, float %980, i64 2, !dbg !50
  %983 = extractelement <32 x float> %.pn68, i64 19, !dbg !50
  %984 = call float @llvm.amdgcn.exp2.f32(float %983), !dbg !50
  %985 = fadd float %981, %984, !dbg !51
  %986 = insertelement <4 x float> %982, float %984, i64 3, !dbg !50
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %986) #9, !dbg !52, !frisk.register_lifetimes !29
  %987 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !52
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %987) #9, !dbg !50, !frisk.register_lifetimes !29
  %988 = extractelement <32 x float> %.pn68, i64 20, !dbg !50
  %989 = call float @llvm.amdgcn.exp2.f32(float %988), !dbg !50
  %990 = fadd float %985, %989, !dbg !51
  %991 = insertelement <4 x float> poison, float %989, i64 0, !dbg !50
  %992 = extractelement <32 x float> %.pn68, i64 21, !dbg !50
  %993 = call float @llvm.amdgcn.exp2.f32(float %992), !dbg !50
  %994 = fadd float %990, %993, !dbg !51
  %995 = insertelement <4 x float> %991, float %993, i64 1, !dbg !50
  %996 = extractelement <32 x float> %.pn68, i64 22, !dbg !50
  %997 = call float @llvm.amdgcn.exp2.f32(float %996), !dbg !50
  %998 = fadd float %994, %997, !dbg !51
  %999 = insertelement <4 x float> %995, float %997, i64 2, !dbg !50
  %1000 = extractelement <32 x float> %.pn68, i64 23, !dbg !50
  %1001 = call float @llvm.amdgcn.exp2.f32(float %1000), !dbg !50
  %1002 = fadd float %998, %1001, !dbg !51
  %1003 = insertelement <4 x float> %999, float %1001, i64 3, !dbg !50
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1003) #9, !dbg !52, !frisk.register_lifetimes !29
  %1004 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !52
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1004) #9, !dbg !50, !frisk.register_lifetimes !29
  %1005 = extractelement <32 x float> %.pn68, i64 24, !dbg !50
  %1006 = call float @llvm.amdgcn.exp2.f32(float %1005), !dbg !50
  %1007 = fadd float %1002, %1006, !dbg !51
  %1008 = insertelement <4 x float> poison, float %1006, i64 0, !dbg !50
  %1009 = extractelement <32 x float> %.pn68, i64 25, !dbg !50
  %1010 = call float @llvm.amdgcn.exp2.f32(float %1009), !dbg !50
  %1011 = fadd float %1007, %1010, !dbg !51
  %1012 = insertelement <4 x float> %1008, float %1010, i64 1, !dbg !50
  %1013 = extractelement <32 x float> %.pn68, i64 26, !dbg !50
  %1014 = call float @llvm.amdgcn.exp2.f32(float %1013), !dbg !50
  %1015 = fadd float %1011, %1014, !dbg !51
  %1016 = insertelement <4 x float> %1012, float %1014, i64 2, !dbg !50
  %1017 = extractelement <32 x float> %.pn68, i64 27, !dbg !50
  %1018 = call float @llvm.amdgcn.exp2.f32(float %1017), !dbg !50
  %1019 = fadd float %1015, %1018, !dbg !51
  %1020 = insertelement <4 x float> %1016, float %1018, i64 3, !dbg !50
  %math.result129 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1020) #9, !dbg !52, !frisk.register_lifetimes !29
  %1021 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !52
  %math.packed130 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1021) #9, !dbg !50, !frisk.register_lifetimes !29
  %1022 = extractelement <32 x float> %.pn68, i64 28, !dbg !50
  %1023 = call float @llvm.amdgcn.exp2.f32(float %1022), !dbg !50
  %1024 = fadd float %1019, %1023, !dbg !51
  %1025 = insertelement <4 x float> poison, float %1023, i64 0, !dbg !50
  %1026 = extractelement <32 x float> %.pn68, i64 29, !dbg !50
  %1027 = call float @llvm.amdgcn.exp2.f32(float %1026), !dbg !50
  %1028 = fadd float %1024, %1027, !dbg !51
  %1029 = insertelement <4 x float> %1025, float %1027, i64 1, !dbg !50
  %1030 = extractelement <32 x float> %.pn68, i64 30, !dbg !50
  %1031 = call float @llvm.amdgcn.exp2.f32(float %1030), !dbg !50
  %1032 = fadd float %1028, %1031, !dbg !51
  %1033 = insertelement <4 x float> %1029, float %1031, i64 2, !dbg !50
  %1034 = extractelement <32 x float> %.pn68, i64 31, !dbg !50
  %1035 = call float @llvm.amdgcn.exp2.f32(float %1034), !dbg !50
  %1036 = fadd float %1032, %1035, !dbg !51
  %1037 = insertelement <4 x float> %1033, float %1035, i64 3, !dbg !50
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1037) #9, !dbg !52, !frisk.register_lifetimes !29
  %1038 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !52
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1038) #9, !dbg !51, !frisk.register_lifetimes !29
  %1039 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !51
  %1040 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1039), !dbg !51
  %1041 = add nuw nsw i32 %1040, 64, !dbg !51
  %1042 = xor i32 %1040, 16, !dbg !51
  %1043 = and i32 %1041, 192, !dbg !51
  %1044 = icmp samesign ult i32 %1042, %1043, !dbg !51
  %1045 = select i1 %1044, i32 %1042, i32 %1040, !dbg !51
  %1046 = shl nuw nsw i32 %1045, 2, !dbg !51
  %1047 = bitcast float %901 to i32, !dbg !51
  %1048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1047), !dbg !51
  %1049 = bitcast i32 %1048 to float, !dbg !51
  %1050 = fadd float %901, %1049, !dbg !51
  %1051 = xor i32 %1040, 32, !dbg !51
  %1052 = icmp samesign ult i32 %1051, %1043, !dbg !51
  %1053 = select i1 %1052, i32 %1051, i32 %1040, !dbg !51
  %1054 = shl nuw nsw i32 %1053, 2, !dbg !51
  %1055 = bitcast float %1050 to i32, !dbg !51
  %1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1054, i32 %1055), !dbg !51
  %1057 = bitcast i32 %1056 to float, !dbg !51
  %1058 = fadd float %1050, %1057, !dbg !51
  %1059 = extractvalue [2 x <1 x float>] %28, 0, !dbg !51
  %1060 = extractelement <1 x float> %1059, i64 0, !dbg !51
  %1061 = fadd float %1058, %1060, !dbg !51
  %1062 = insertelement <1 x float> poison, float %1061, i64 0, !dbg !40
  %1063 = insertvalue [2 x <1 x float>] poison, <1 x float> %1062, 0, !dbg !40
  %1064 = bitcast float %1036 to i32, !dbg !51
  %1065 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1046, i32 %1064), !dbg !51
  %1066 = bitcast i32 %1065 to float, !dbg !51
  %1067 = fadd float %1036, %1066, !dbg !51
  %1068 = bitcast float %1067 to i32, !dbg !51
  %1069 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1054, i32 %1068), !dbg !51
  %1070 = bitcast i32 %1069 to float, !dbg !51
  %1071 = fadd float %1067, %1070, !dbg !51
  %1072 = extractvalue [2 x <1 x float>] %28, 1, !dbg !51
  %1073 = extractelement <1 x float> %1072, i64 0, !dbg !51
  %1074 = fadd float %1071, %1073, !dbg !51
  %1075 = insertelement <1 x float> poison, float %1074, i64 0, !dbg !40
  %1076 = insertvalue [2 x <1 x float>] %1063, <1 x float> %1075, 1, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %address.thread324 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !39, !range !57, !frisk.register_lifetimes !29
  %address.local325 = shl nuw nsw i32 %address.thread324, 3, !dbg !39
  %address.local326 = and i32 %address.local325, 120, !dbg !39
  %address.local327 = shl nuw nsw i32 %address.thread324, 5, !dbg !39
  %address.local328 = and i32 %address.local327, 1536, !dbg !39
  %address.local329 = or disjoint i32 %address.local326, %address.local328, !dbg !39
  %address.local330 = zext nneg i32 %address.local329 to i64, !dbg !39
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !39
  %1077 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1078 = bitcast <4 x i32> %1077 to <8 x half>, !dbg !39
  %address.local332 = zext nneg i32 %address.thread324 to i64, !dbg !37
  %address.local333 = and i64 %address.local332, 60, !dbg !39
  %address.local334 = add nuw nsw i64 %address.local333, %address.local332, !dbg !39
  %address.local335 = shl nuw nsw i64 %address.local334, 3, !dbg !39
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local337 = and i64 %address.local332, 63, !dbg !37
  %address.local338 = lshr i64 %address.local337, 4, !dbg !39
  %address.local339 = mul nuw nsw i64 %address.local338, 384, !dbg !39
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) %address.local336, i64 %address.local339, !dbg !39
  %address.local341 = lshr i64 %address.local332, 2, !dbg !34
  %address.local342 = mul nsw i64 %address.local341, -32, !dbg !39
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) %address.local340, i64 %address.local342, !dbg !39
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local343, i64 256, !dbg !39
  %1079 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1080 = bitcast <4 x i32> %1079 to <8 x half>, !dbg !39
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local339, !dbg !39
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) %address.local346, i64 %address.local342, !dbg !39
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local347, i64 512, !dbg !39
  %1081 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1082 = bitcast <4 x i32> %1081 to <8 x half>, !dbg !39
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) %address.local349, i64 %address.local339, !dbg !39
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local342, !dbg !39
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local351, i64 768, !dbg !39
  %1083 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1084 = bitcast <4 x i32> %1083 to <8 x half>, !dbg !39
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !39
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local353, i64 4096, !dbg !39
  %1085 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1086 = bitcast <4 x i32> %1085 to <8 x half>, !dbg !39
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local339, !dbg !39
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local342, !dbg !39
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local357, i64 4352, !dbg !39
  %1087 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1088 = bitcast <4 x i32> %1087 to <8 x half>, !dbg !39
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local339, !dbg !39
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %address.local360, i64 %address.local342, !dbg !39
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !39
  %1089 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1090 = bitcast <4 x i32> %1089 to <8 x half>, !dbg !39
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) %address.local363, i64 %address.local339, !dbg !39
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) %address.local364, i64 %address.local342, !dbg !39
  %address.local366 = getelementptr i8, ptr addrspace(3) %address.local365, i64 4864, !dbg !39
  %1091 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1092 = bitcast <4 x i32> %1091 to <8 x half>, !dbg !39
  %1093 = add nsw i64 %34, %8, !dbg !42
  %1094 = shl i64 %1093, 19, !dbg !42
  %1095 = and i64 %1094, 16252928, !dbg !42
  %1096 = add i64 %1095, %239, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1097 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1096, !dbg !42
  %1098 = getelementptr i8, ptr addrspace(1) %1097, i64 64, !dbg !42
  %address.local367 = shl nuw nsw i64 %address.local332, 3, !dbg !36
  %address.local368 = and i64 %address.local367, 1536, !dbg !36
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !36
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !36
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !36
  %1099 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !42
  %1100 = zext i32 %1099 to i64, !dbg !42
  %1101 = inttoptr i64 %1100 to ptr addrspace(3), !dbg !42
  %address.local372 = shl nuw nsw i64 %address.local332, 5, !dbg !42
  %address.local373 = and i64 %address.local372, 8064, !dbg !42
  %address.local374 = shl nuw nsw i64 %address.local332, 3, !dbg !42
  %address.local375 = and i64 %address.local374, 24, !dbg !42
  %address.local376 = or disjoint i64 %address.local373, %address.local375, !dbg !42
  %address.local377 = getelementptr [2 x i8], ptr addrspace(1) %1098, i64 %address.local376, !dbg !42
  %1102 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1103 = ptrtoint ptr addrspace(1) %address.local377 to i64, !dbg !42
  %1104 = sub i64 %1103, %1102, !dbg !42
  %dma.buffer.offset641 = trunc i64 %1104 to i32, !dbg !42
  %1105 = trunc i64 %1102 to i32, !dbg !42
  %1106 = insertelement <4 x i32> poison, i32 %1105, i64 0, !dbg !42
  %1107 = lshr i64 %1102, 32, !dbg !42
  %1108 = trunc i64 %1107 to i32, !dbg !42
  %1109 = insertelement <4 x i32> %1106, i32 %1108, i64 1, !dbg !42
  %1110 = insertelement <4 x i32> %1109, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc642 = insertelement <4 x i32> %1110, i32 131072, i64 3, !dbg !42
  %1111 = ptrtoint ptr addrspace(3) %1101 to i32, !dbg !42
  %dma.buffer.lds643 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1111), !dbg !42
  %1112 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset641, <4 x i32> %dma.buffer.rsrc642) #10, !dbg !42
  %address.local378 = or disjoint i64 %address.local332, 256, !dbg !36
  %address.local379 = shl nuw nsw i64 %address.local378, 3, !dbg !36
  %address.local380 = and i64 %address.local379, 3584, !dbg !36
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local380, !dbg !36
  %address.local382 = ptrtoint ptr addrspace(3) %address.local381 to i64, !dbg !36
  %address.local383 = trunc i64 %address.local382 to i32, !dbg !36
  %1113 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !42
  %1114 = zext i32 %1113 to i64, !dbg !42
  %1115 = inttoptr i64 %1114 to ptr addrspace(3), !dbg !42
  %address.local384 = shl nuw nsw i64 %address.local378, 5, !dbg !42
  %address.local385 = and i64 %address.local384, 16256, !dbg !42
  %address.local386 = shl nuw nsw i64 %address.local332, 3, !dbg !42
  %address.local387 = and i64 %address.local386, 24, !dbg !42
  %address.local388 = or disjoint i64 %address.local385, %address.local387, !dbg !42
  %address.local389 = getelementptr [2 x i8], ptr addrspace(1) %1098, i64 %address.local388, !dbg !42
  %1116 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1117 = ptrtoint ptr addrspace(1) %address.local389 to i64, !dbg !42
  %1118 = sub i64 %1117, %1116, !dbg !42
  %dma.buffer.offset638 = trunc i64 %1118 to i32, !dbg !42
  %1119 = trunc i64 %1116 to i32, !dbg !42
  %1120 = insertelement <4 x i32> poison, i32 %1119, i64 0, !dbg !42
  %1121 = lshr i64 %1116, 32, !dbg !42
  %1122 = trunc i64 %1121 to i32, !dbg !42
  %1123 = insertelement <4 x i32> %1120, i32 %1122, i64 1, !dbg !42
  %1124 = insertelement <4 x i32> %1123, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc639 = insertelement <4 x i32> %1124, i32 131072, i64 3, !dbg !42
  %1125 = ptrtoint ptr addrspace(3) %1115 to i32, !dbg !42
  %dma.buffer.lds640 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1125), !dbg !42
  %1126 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset638, <4 x i32> %dma.buffer.rsrc639) #10, !dbg !42
  %1127 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1128 = extractvalue [2 x <8 x float>] %27, 0, !dbg !53
  %1129 = shufflevector <8 x float> %1128, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1130 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1127, <4 x float> %1129) #9, !dbg !39
  %1131 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1132 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1131, <4 x float> %1130) #9, !dbg !39
  %1133 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1134 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1133, <4 x float> %1132) #9, !dbg !39
  %1135 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1136 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1135, <4 x float> %1134) #9, !dbg !39
  %1137 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1138 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1137, <4 x float> %1136) #9, !dbg !39
  %1139 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1140 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1139, <4 x float> %1138) #9, !dbg !39
  %1141 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1142 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1141, <4 x float> %1140) #9, !dbg !39
  %1143 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1144 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1143, <4 x float> %1142) #9, !dbg !39
  %1145 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1146 = shufflevector <8 x float> %1128, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1147 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1145, <4 x float> %1146) #9, !dbg !39
  %1148 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1149 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1148, <4 x float> %1147) #9, !dbg !39
  %1150 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1151 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1150, <4 x float> %1149) #9, !dbg !39
  %1152 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1153 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1152, <4 x float> %1151) #9, !dbg !39
  %1154 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1155 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1154, <4 x float> %1153) #9, !dbg !39
  %1156 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1157 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1156, <4 x float> %1155) #9, !dbg !39
  %1158 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1159 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1158, <4 x float> %1157) #9, !dbg !39
  %1160 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1161 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1160, <4 x float> %1159) #9, !dbg !39
  %1162 = extractvalue [2 x <8 x float>] %27, 1, !dbg !53
  %1163 = shufflevector <8 x float> %1162, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1164 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1127, <4 x float> %1163) #9, !dbg !39
  %1165 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1131, <4 x float> %1164) #9, !dbg !39
  %1166 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1133, <4 x float> %1165) #9, !dbg !39
  %1167 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1135, <4 x float> %1166) #9, !dbg !39
  %1168 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1137, <4 x float> %1167) #9, !dbg !39
  %1169 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1139, <4 x float> %1168) #9, !dbg !39
  %1170 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1141, <4 x float> %1169) #9, !dbg !39
  %1171 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1143, <4 x float> %1170) #9, !dbg !39
  %1172 = shufflevector <8 x float> %1162, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1173 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1145, <4 x float> %1172) #9, !dbg !39
  %1174 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1148, <4 x float> %1173) #9, !dbg !39
  %1175 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1150, <4 x float> %1174) #9, !dbg !39
  %1176 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1152, <4 x float> %1175) #9, !dbg !39
  %1177 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1154, <4 x float> %1176) #9, !dbg !39
  %1178 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1156, <4 x float> %1177) #9, !dbg !39
  %1179 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1158, <4 x float> %1178) #9, !dbg !39
  %1180 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1160, <4 x float> %1179) #9, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1182 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !39
  %1183 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1184 = bitcast <4 x i32> %1183 to <8 x half>, !dbg !39
  %1185 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1186 = bitcast <4 x i32> %1185 to <8 x half>, !dbg !39
  %1187 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1188 = bitcast <4 x i32> %1187 to <8 x half>, !dbg !39
  %1189 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1190 = bitcast <4 x i32> %1189 to <8 x half>, !dbg !39
  %1191 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1192 = bitcast <4 x i32> %1191 to <8 x half>, !dbg !39
  %1193 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1194 = bitcast <4 x i32> %1193 to <8 x half>, !dbg !39
  %1195 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1196 = bitcast <4 x i32> %1195 to <8 x half>, !dbg !39
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1197 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1096, !dbg !42
  %1198 = getelementptr i8, ptr addrspace(1) %1197, i64 128, !dbg !42
  %1199 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !42
  %1200 = zext i32 %1199 to i64, !dbg !42
  %1201 = inttoptr i64 %1200 to ptr addrspace(3), !dbg !42
  %address.local390 = getelementptr [2 x i8], ptr addrspace(1) %1198, i64 %address.local376, !dbg !42
  %1202 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1203 = ptrtoint ptr addrspace(1) %address.local390 to i64, !dbg !42
  %1204 = sub i64 %1203, %1202, !dbg !42
  %dma.buffer.offset635 = trunc i64 %1204 to i32, !dbg !42
  %1205 = trunc i64 %1202 to i32, !dbg !42
  %1206 = insertelement <4 x i32> poison, i32 %1205, i64 0, !dbg !42
  %1207 = lshr i64 %1202, 32, !dbg !42
  %1208 = trunc i64 %1207 to i32, !dbg !42
  %1209 = insertelement <4 x i32> %1206, i32 %1208, i64 1, !dbg !42
  %1210 = insertelement <4 x i32> %1209, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc636 = insertelement <4 x i32> %1210, i32 131072, i64 3, !dbg !42
  %1211 = ptrtoint ptr addrspace(3) %1201 to i32, !dbg !42
  %dma.buffer.lds637 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1211), !dbg !42
  %1212 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds637, i32 %dma.buffer.offset635, <4 x i32> %dma.buffer.rsrc636) #10, !dbg !42
  %1213 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !42
  %1214 = zext i32 %1213 to i64, !dbg !42
  %1215 = inttoptr i64 %1214 to ptr addrspace(3), !dbg !42
  %address.local391 = getelementptr [2 x i8], ptr addrspace(1) %1198, i64 %address.local388, !dbg !42
  %1216 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1217 = ptrtoint ptr addrspace(1) %address.local391 to i64, !dbg !42
  %1218 = sub i64 %1217, %1216, !dbg !42
  %dma.buffer.offset632 = trunc i64 %1218 to i32, !dbg !42
  %1219 = trunc i64 %1216 to i32, !dbg !42
  %1220 = insertelement <4 x i32> poison, i32 %1219, i64 0, !dbg !42
  %1221 = lshr i64 %1216, 32, !dbg !42
  %1222 = trunc i64 %1221 to i32, !dbg !42
  %1223 = insertelement <4 x i32> %1220, i32 %1222, i64 1, !dbg !42
  %1224 = insertelement <4 x i32> %1223, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc633 = insertelement <4 x i32> %1224, i32 131072, i64 3, !dbg !42
  %1225 = ptrtoint ptr addrspace(3) %1215 to i32, !dbg !42
  %dma.buffer.lds634 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1225), !dbg !42
  %1226 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds634, i32 %dma.buffer.offset632, <4 x i32> %dma.buffer.rsrc633) #10, !dbg !42
  %1227 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1228 = extractvalue [2 x <8 x float>] %29, 0, !dbg !53
  %1229 = shufflevector <8 x float> %1228, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1230 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1227, <4 x float> %1229) #9, !dbg !39
  %1231 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1232 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1231, <4 x float> %1230) #9, !dbg !39
  %1233 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1234 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1233, <4 x float> %1232) #9, !dbg !39
  %1235 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1236 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1235, <4 x float> %1234) #9, !dbg !39
  %1237 = shufflevector <8 x half> %1190, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1238 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1237, <4 x float> %1236) #9, !dbg !39
  %1239 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1240 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1239, <4 x float> %1238) #9, !dbg !39
  %1241 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1242 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1241, <4 x float> %1240) #9, !dbg !39
  %1243 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1244 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1243, <4 x float> %1242) #9, !dbg !39
  %1245 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1246 = shufflevector <8 x float> %1228, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1247 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1245, <4 x float> %1246) #9, !dbg !39
  %1248 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1249 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1248, <4 x float> %1247) #9, !dbg !39
  %1250 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1251 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1250, <4 x float> %1249) #9, !dbg !39
  %1252 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1253 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1252, <4 x float> %1251) #9, !dbg !39
  %1254 = shufflevector <8 x half> %1190, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1255 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1254, <4 x float> %1253) #9, !dbg !39
  %1256 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1257 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1256, <4 x float> %1255) #9, !dbg !39
  %1258 = shufflevector <8 x half> %1194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1259 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1258, <4 x float> %1257) #9, !dbg !39
  %1260 = shufflevector <8 x half> %1196, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1261 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1260, <4 x float> %1259) #9, !dbg !39
  %1262 = extractvalue [2 x <8 x float>] %29, 1, !dbg !53
  %1263 = shufflevector <8 x float> %1262, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1264 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1227, <4 x float> %1263) #9, !dbg !39
  %1265 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1231, <4 x float> %1264) #9, !dbg !39
  %1266 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1233, <4 x float> %1265) #9, !dbg !39
  %1267 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1235, <4 x float> %1266) #9, !dbg !39
  %1268 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1237, <4 x float> %1267) #9, !dbg !39
  %1269 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1239, <4 x float> %1268) #9, !dbg !39
  %1270 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1241, <4 x float> %1269) #9, !dbg !39
  %1271 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1243, <4 x float> %1270) #9, !dbg !39
  %1272 = shufflevector <8 x float> %1262, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1273 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1245, <4 x float> %1272) #9, !dbg !39
  %1274 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1248, <4 x float> %1273) #9, !dbg !39
  %1275 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1250, <4 x float> %1274) #9, !dbg !39
  %1276 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1252, <4 x float> %1275) #9, !dbg !39
  %1277 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1254, <4 x float> %1276) #9, !dbg !39
  %1278 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1256, <4 x float> %1277) #9, !dbg !39
  %1279 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1258, <4 x float> %1278) #9, !dbg !39
  %1280 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1260, <4 x float> %1279) #9, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1281 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1282 = bitcast <4 x i32> %1281 to <8 x half>, !dbg !39
  %1283 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1284 = bitcast <4 x i32> %1283 to <8 x half>, !dbg !39
  %1285 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1286 = bitcast <4 x i32> %1285 to <8 x half>, !dbg !39
  %1287 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1288 = bitcast <4 x i32> %1287 to <8 x half>, !dbg !39
  %1289 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1290 = bitcast <4 x i32> %1289 to <8 x half>, !dbg !39
  %1291 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1292 = bitcast <4 x i32> %1291 to <8 x half>, !dbg !39
  %1293 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1294 = bitcast <4 x i32> %1293 to <8 x half>, !dbg !39
  %1295 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1296 = bitcast <4 x i32> %1295 to <8 x half>, !dbg !39
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1297 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1096, !dbg !42
  %1298 = getelementptr i8, ptr addrspace(1) %1297, i64 192, !dbg !42
  %1299 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !42
  %1300 = zext i32 %1299 to i64, !dbg !42
  %1301 = inttoptr i64 %1300 to ptr addrspace(3), !dbg !42
  %address.local392 = getelementptr [2 x i8], ptr addrspace(1) %1298, i64 %address.local376, !dbg !42
  %1302 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1303 = ptrtoint ptr addrspace(1) %address.local392 to i64, !dbg !42
  %1304 = sub i64 %1303, %1302, !dbg !42
  %dma.buffer.offset629 = trunc i64 %1304 to i32, !dbg !42
  %1305 = trunc i64 %1302 to i32, !dbg !42
  %1306 = insertelement <4 x i32> poison, i32 %1305, i64 0, !dbg !42
  %1307 = lshr i64 %1302, 32, !dbg !42
  %1308 = trunc i64 %1307 to i32, !dbg !42
  %1309 = insertelement <4 x i32> %1306, i32 %1308, i64 1, !dbg !42
  %1310 = insertelement <4 x i32> %1309, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc630 = insertelement <4 x i32> %1310, i32 131072, i64 3, !dbg !42
  %1311 = ptrtoint ptr addrspace(3) %1301 to i32, !dbg !42
  %dma.buffer.lds631 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1311), !dbg !42
  %1312 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds631, i32 %dma.buffer.offset629, <4 x i32> %dma.buffer.rsrc630) #10, !dbg !42
  %1313 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !42
  %1314 = zext i32 %1313 to i64, !dbg !42
  %1315 = inttoptr i64 %1314 to ptr addrspace(3), !dbg !42
  %address.local393 = getelementptr [2 x i8], ptr addrspace(1) %1298, i64 %address.local388, !dbg !42
  %1316 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1317 = ptrtoint ptr addrspace(1) %address.local393 to i64, !dbg !42
  %1318 = sub i64 %1317, %1316, !dbg !42
  %dma.buffer.offset626 = trunc i64 %1318 to i32, !dbg !42
  %1319 = trunc i64 %1316 to i32, !dbg !42
  %1320 = insertelement <4 x i32> poison, i32 %1319, i64 0, !dbg !42
  %1321 = lshr i64 %1316, 32, !dbg !42
  %1322 = trunc i64 %1321 to i32, !dbg !42
  %1323 = insertelement <4 x i32> %1320, i32 %1322, i64 1, !dbg !42
  %1324 = insertelement <4 x i32> %1323, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc627 = insertelement <4 x i32> %1324, i32 131072, i64 3, !dbg !42
  %1325 = ptrtoint ptr addrspace(3) %1315 to i32, !dbg !42
  %dma.buffer.lds628 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1325), !dbg !42
  %1326 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds628, i32 %dma.buffer.offset626, <4 x i32> %dma.buffer.rsrc627) #10, !dbg !42
  %1327 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1328 = extractvalue [2 x <8 x float>] %30, 0, !dbg !53
  %1329 = shufflevector <8 x float> %1328, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1330 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1327, <4 x float> %1329) #9, !dbg !39
  %1331 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1332 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1331, <4 x float> %1330) #9, !dbg !39
  %1333 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1334 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1333, <4 x float> %1332) #9, !dbg !39
  %1335 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1336 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1335, <4 x float> %1334) #9, !dbg !39
  %1337 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1338 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1337, <4 x float> %1336) #9, !dbg !39
  %1339 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1340 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1339, <4 x float> %1338) #9, !dbg !39
  %1341 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1342 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1341, <4 x float> %1340) #9, !dbg !39
  %1343 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1344 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1343, <4 x float> %1342) #9, !dbg !39
  %1345 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1346 = shufflevector <8 x float> %1328, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1347 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1345, <4 x float> %1346) #9, !dbg !39
  %1348 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1349 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1348, <4 x float> %1347) #9, !dbg !39
  %1350 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1351 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1350, <4 x float> %1349) #9, !dbg !39
  %1352 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1353 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1352, <4 x float> %1351) #9, !dbg !39
  %1354 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1355 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1354, <4 x float> %1353) #9, !dbg !39
  %1356 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1357 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1356, <4 x float> %1355) #9, !dbg !39
  %1358 = shufflevector <8 x half> %1294, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1359 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1358, <4 x float> %1357) #9, !dbg !39
  %1360 = shufflevector <8 x half> %1296, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1361 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1360, <4 x float> %1359) #9, !dbg !39
  %1362 = extractvalue [2 x <8 x float>] %30, 1, !dbg !53
  %1363 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1364 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1327, <4 x float> %1363) #9, !dbg !39
  %1365 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1331, <4 x float> %1364) #9, !dbg !39
  %1366 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1333, <4 x float> %1365) #9, !dbg !39
  %1367 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1335, <4 x float> %1366) #9, !dbg !39
  %1368 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1337, <4 x float> %1367) #9, !dbg !39
  %1369 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1339, <4 x float> %1368) #9, !dbg !39
  %1370 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1341, <4 x float> %1369) #9, !dbg !39
  %1371 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1343, <4 x float> %1370) #9, !dbg !39
  %1372 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1373 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1345, <4 x float> %1372) #9, !dbg !39
  %1374 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1348, <4 x float> %1373) #9, !dbg !39
  %1375 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1350, <4 x float> %1374) #9, !dbg !39
  %1376 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1352, <4 x float> %1375) #9, !dbg !39
  %1377 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1354, <4 x float> %1376) #9, !dbg !39
  %1378 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1356, <4 x float> %1377) #9, !dbg !39
  %1379 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1358, <4 x float> %1378) #9, !dbg !39
  %1380 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1360, <4 x float> %1379) #9, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !39
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !39
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !39
  %1387 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1388 = bitcast <4 x i32> %1387 to <8 x half>, !dbg !39
  %1389 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1390 = bitcast <4 x i32> %1389 to <8 x half>, !dbg !39
  %1391 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1392 = bitcast <4 x i32> %1391 to <8 x half>, !dbg !39
  %1393 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1394 = bitcast <4 x i32> %1393 to <8 x half>, !dbg !39
  %1395 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1396 = bitcast <4 x i32> %1395 to <8 x half>, !dbg !39
  %1397 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1398 = extractvalue [2 x <8 x float>] %31, 0, !dbg !53
  %1399 = shufflevector <8 x float> %1398, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1400 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1397, <4 x float> %1399) #9, !dbg !39
  %1401 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1402 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1401, <4 x float> %1400) #9, !dbg !39
  %1403 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1404 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1403, <4 x float> %1402) #9, !dbg !39
  %1405 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1406 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1405, <4 x float> %1404) #9, !dbg !39
  %1407 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1408 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1407, <4 x float> %1406) #9, !dbg !39
  %1409 = shufflevector <8 x half> %1392, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1410 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1409, <4 x float> %1408) #9, !dbg !39
  %1411 = shufflevector <8 x half> %1394, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1412 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1411, <4 x float> %1410) #9, !dbg !39
  %1413 = shufflevector <8 x half> %1396, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1414 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1413, <4 x float> %1412) #9, !dbg !39
  %1415 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1416 = shufflevector <8 x float> %1398, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1417 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1415, <4 x float> %1416) #9, !dbg !39
  %1418 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1419 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1418, <4 x float> %1417) #9, !dbg !39
  %1420 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1421 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1420, <4 x float> %1419) #9, !dbg !39
  %1422 = shufflevector <8 x half> %1388, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1423 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1422, <4 x float> %1421) #9, !dbg !39
  %1424 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1425 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1424, <4 x float> %1423) #9, !dbg !39
  %1426 = shufflevector <8 x half> %1392, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1427 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1426, <4 x float> %1425) #9, !dbg !39
  %1428 = shufflevector <8 x half> %1394, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1429 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1428, <4 x float> %1427) #9, !dbg !39
  %1430 = shufflevector <8 x half> %1396, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1431 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1430, <4 x float> %1429) #9, !dbg !39
  %1432 = extractvalue [2 x <8 x float>] %31, 1, !dbg !53
  %1433 = shufflevector <8 x float> %1432, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1434 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1397, <4 x float> %1433) #9, !dbg !39
  %1435 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1401, <4 x float> %1434) #9, !dbg !39
  %1436 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1403, <4 x float> %1435) #9, !dbg !39
  %1437 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1405, <4 x float> %1436) #9, !dbg !39
  %1438 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1407, <4 x float> %1437) #9, !dbg !39
  %1439 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1409, <4 x float> %1438) #9, !dbg !39
  %1440 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1411, <4 x float> %1439) #9, !dbg !39
  %1441 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1413, <4 x float> %1440) #9, !dbg !39
  %1442 = shufflevector <8 x float> %1432, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1443 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1415, <4 x float> %1442) #9, !dbg !39
  %1444 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1418, <4 x float> %1443) #9, !dbg !39
  %1445 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1420, <4 x float> %1444) #9, !dbg !39
  %1446 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1422, <4 x float> %1445) #9, !dbg !39
  %1447 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1424, <4 x float> %1446) #9, !dbg !39
  %1448 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1426, <4 x float> %1447) #9, !dbg !39
  %1449 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1428, <4 x float> %1448) #9, !dbg !39
  %1450 = call <4 x float> asm "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1430, <4 x float> %1449) #9, !dbg !39
  %1451 = shufflevector <4 x float> %1144, <4 x float> %1161, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1452 = insertvalue [2 x <8 x float>] poison, <8 x float> %1451, 0, !dbg !40
  %1453 = shufflevector <4 x float> %1171, <4 x float> %1180, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1454 = insertvalue [2 x <8 x float>] %1452, <8 x float> %1453, 1, !dbg !40
  %1455 = shufflevector <4 x float> %1244, <4 x float> %1261, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1456 = insertvalue [2 x <8 x float>] poison, <8 x float> %1455, 0, !dbg !40
  %1457 = shufflevector <4 x float> %1271, <4 x float> %1280, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1458 = insertvalue [2 x <8 x float>] %1456, <8 x float> %1457, 1, !dbg !40
  %1459 = shufflevector <4 x float> %1344, <4 x float> %1361, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1460 = insertvalue [2 x <8 x float>] poison, <8 x float> %1459, 0, !dbg !40
  %1461 = shufflevector <4 x float> %1371, <4 x float> %1380, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1462 = insertvalue [2 x <8 x float>] %1460, <8 x float> %1461, 1, !dbg !40
  %1463 = shufflevector <4 x float> %1414, <4 x float> %1431, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1464 = insertvalue [2 x <8 x float>] poison, <8 x float> %1463, 0, !dbg !40
  %1465 = shufflevector <4 x float> %1441, <4 x float> %1450, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1466 = insertvalue [2 x <8 x float>] %1464, <8 x float> %1465, 1, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %1467 = add nuw nsw i64 %26, 1, !dbg !40
  br label %25, !dbg !40

1468:                                             ; preds = %25
  %1469 = and i32 %19, 16, !dbg !40
  %.not2 = icmp eq i32 %1469, 0, !dbg !40
  %1470 = and i32 %19, 32, !dbg !40
  %.not3 = icmp eq i32 %1470, 0, !dbg !40
  %1471 = extractvalue [2 x <8 x float>] %31, 0, !dbg !40
  %1472 = extractelement <8 x float> %1471, i64 0, !dbg !40
  %1473 = extractelement <8 x float> %1471, i64 1, !dbg !40
  %1474 = extractelement <8 x float> %1471, i64 2, !dbg !40
  %1475 = extractelement <8 x float> %1471, i64 3, !dbg !40
  %1476 = select i1 %.not2, float %1473, float %1472, !dbg !40
  %1477 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !40
  %1478 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1477), !dbg !40
  %1479 = add nuw nsw i32 %1478, 64, !dbg !40
  %1480 = xor i32 %1478, 16, !dbg !40
  %1481 = and i32 %1479, 192, !dbg !40
  %1482 = icmp samesign ult i32 %1480, %1481, !dbg !40
  %1483 = select i1 %1482, i32 %1480, i32 %1478, !dbg !40
  %1484 = shl nuw nsw i32 %1483, 2, !dbg !40
  %1485 = bitcast float %1476 to i32, !dbg !40
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1485), !dbg !40
  %1487 = bitcast i32 %1486 to float, !dbg !40
  %1488 = select i1 %.not2, float %1472, float %1487, !dbg !40
  %1489 = select i1 %.not2, float %1487, float %1473, !dbg !40
  %1490 = select i1 %.not2, float %1475, float %1474, !dbg !40
  %1491 = bitcast float %1490 to i32, !dbg !40
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1491), !dbg !40
  %1493 = bitcast i32 %1492 to float, !dbg !40
  %1494 = select i1 %.not2, float %1474, float %1493, !dbg !40
  %1495 = select i1 %.not2, float %1493, float %1475, !dbg !40
  %1496 = select i1 %.not3, float %1494, float %1488, !dbg !40
  %1497 = xor i32 %1478, 32, !dbg !40
  %1498 = icmp samesign ult i32 %1497, %1481, !dbg !40
  %1499 = select i1 %1498, i32 %1497, i32 %1478, !dbg !40
  %1500 = shl nuw nsw i32 %1499, 2, !dbg !40
  %1501 = bitcast float %1496 to i32, !dbg !40
  %1502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1501), !dbg !40
  %1503 = bitcast i32 %1502 to float, !dbg !40
  %1504 = select i1 %.not3, float %1488, float %1503, !dbg !40
  %1505 = select i1 %.not3, float %1503, float %1494, !dbg !40
  %1506 = select i1 %.not3, float %1495, float %1489, !dbg !40
  %1507 = bitcast float %1506 to i32, !dbg !40
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1507), !dbg !40
  %1509 = bitcast i32 %1508 to float, !dbg !40
  %1510 = select i1 %.not3, float %1489, float %1509, !dbg !40
  %1511 = select i1 %.not3, float %1509, float %1495, !dbg !40
  %1512 = extractvalue [2 x <8 x float>] %30, 0, !dbg !40
  %1513 = extractelement <8 x float> %1512, i64 0, !dbg !40
  %1514 = extractelement <8 x float> %1512, i64 1, !dbg !40
  %1515 = extractelement <8 x float> %1512, i64 2, !dbg !40
  %1516 = extractelement <8 x float> %1512, i64 3, !dbg !40
  %1517 = select i1 %.not2, float %1514, float %1513, !dbg !40
  %1518 = bitcast float %1517 to i32, !dbg !40
  %1519 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1518), !dbg !40
  %1520 = bitcast i32 %1519 to float, !dbg !40
  %1521 = select i1 %.not2, float %1513, float %1520, !dbg !40
  %1522 = select i1 %.not2, float %1520, float %1514, !dbg !40
  %1523 = select i1 %.not2, float %1516, float %1515, !dbg !40
  %1524 = bitcast float %1523 to i32, !dbg !40
  %1525 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1524), !dbg !40
  %1526 = bitcast i32 %1525 to float, !dbg !40
  %1527 = select i1 %.not2, float %1515, float %1526, !dbg !40
  %1528 = select i1 %.not2, float %1526, float %1516, !dbg !40
  %1529 = select i1 %.not3, float %1527, float %1521, !dbg !40
  %1530 = bitcast float %1529 to i32, !dbg !40
  %1531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1530), !dbg !40
  %1532 = bitcast i32 %1531 to float, !dbg !40
  %1533 = select i1 %.not3, float %1521, float %1532, !dbg !40
  %1534 = select i1 %.not3, float %1532, float %1527, !dbg !40
  %1535 = select i1 %.not3, float %1528, float %1522, !dbg !40
  %1536 = bitcast float %1535 to i32, !dbg !40
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1536), !dbg !40
  %1538 = bitcast i32 %1537 to float, !dbg !40
  %1539 = select i1 %.not3, float %1522, float %1538, !dbg !40
  %1540 = select i1 %.not3, float %1538, float %1528, !dbg !40
  %1541 = extractvalue [2 x <8 x float>] %29, 0, !dbg !40
  %1542 = extractelement <8 x float> %1541, i64 0, !dbg !40
  %1543 = extractelement <8 x float> %1541, i64 1, !dbg !40
  %1544 = extractelement <8 x float> %1541, i64 2, !dbg !40
  %1545 = extractelement <8 x float> %1541, i64 3, !dbg !40
  %1546 = select i1 %.not2, float %1543, float %1542, !dbg !40
  %1547 = bitcast float %1546 to i32, !dbg !40
  %1548 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1547), !dbg !40
  %1549 = bitcast i32 %1548 to float, !dbg !40
  %1550 = select i1 %.not2, float %1542, float %1549, !dbg !40
  %1551 = select i1 %.not2, float %1549, float %1543, !dbg !40
  %1552 = select i1 %.not2, float %1545, float %1544, !dbg !40
  %1553 = bitcast float %1552 to i32, !dbg !40
  %1554 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1553), !dbg !40
  %1555 = bitcast i32 %1554 to float, !dbg !40
  %1556 = select i1 %.not2, float %1544, float %1555, !dbg !40
  %1557 = select i1 %.not2, float %1555, float %1545, !dbg !40
  %1558 = select i1 %.not3, float %1556, float %1550, !dbg !40
  %1559 = bitcast float %1558 to i32, !dbg !40
  %1560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1559), !dbg !40
  %1561 = bitcast i32 %1560 to float, !dbg !40
  %1562 = select i1 %.not3, float %1550, float %1561, !dbg !40
  %1563 = select i1 %.not3, float %1561, float %1556, !dbg !40
  %1564 = select i1 %.not3, float %1557, float %1551, !dbg !40
  %1565 = bitcast float %1564 to i32, !dbg !40
  %1566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1565), !dbg !40
  %1567 = bitcast i32 %1566 to float, !dbg !40
  %1568 = select i1 %.not3, float %1551, float %1567, !dbg !40
  %1569 = select i1 %.not3, float %1567, float %1557, !dbg !40
  %1570 = extractvalue [2 x <8 x float>] %27, 0, !dbg !40
  %1571 = extractelement <8 x float> %1570, i64 0, !dbg !40
  %1572 = extractelement <8 x float> %1570, i64 1, !dbg !40
  %1573 = extractelement <8 x float> %1570, i64 2, !dbg !40
  %1574 = extractelement <8 x float> %1570, i64 3, !dbg !40
  %1575 = select i1 %.not2, float %1572, float %1571, !dbg !40
  %1576 = bitcast float %1575 to i32, !dbg !40
  %1577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1576), !dbg !40
  %1578 = bitcast i32 %1577 to float, !dbg !40
  %1579 = select i1 %.not2, float %1571, float %1578, !dbg !40
  %1580 = select i1 %.not2, float %1578, float %1572, !dbg !40
  %1581 = select i1 %.not2, float %1574, float %1573, !dbg !40
  %1582 = bitcast float %1581 to i32, !dbg !40
  %1583 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1582), !dbg !40
  %1584 = bitcast i32 %1583 to float, !dbg !40
  %1585 = select i1 %.not2, float %1573, float %1584, !dbg !40
  %1586 = select i1 %.not2, float %1584, float %1574, !dbg !40
  %1587 = select i1 %.not3, float %1585, float %1579, !dbg !40
  %1588 = bitcast float %1587 to i32, !dbg !40
  %1589 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1588), !dbg !40
  %1590 = select i1 %.not3, float %1586, float %1580, !dbg !40
  %1591 = bitcast float %1590 to i32, !dbg !40
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1591), !dbg !40
  %1593 = extractvalue [2 x <1 x float>] %28, 0, !dbg !43
  %1594 = shufflevector <1 x float> %1593, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %1595 = extractelement <8 x float> %1471, i64 4, !dbg !40
  %1596 = extractelement <8 x float> %1471, i64 5, !dbg !40
  %1597 = extractelement <8 x float> %1471, i64 6, !dbg !40
  %1598 = extractelement <8 x float> %1471, i64 7, !dbg !40
  %1599 = select i1 %.not2, float %1596, float %1595, !dbg !40
  %1600 = bitcast float %1599 to i32, !dbg !40
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1600), !dbg !40
  %1602 = bitcast i32 %1601 to float, !dbg !40
  %1603 = select i1 %.not2, float %1595, float %1602, !dbg !40
  %1604 = select i1 %.not2, float %1602, float %1596, !dbg !40
  %1605 = select i1 %.not2, float %1598, float %1597, !dbg !40
  %1606 = bitcast float %1605 to i32, !dbg !40
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1606), !dbg !40
  %1608 = bitcast i32 %1607 to float, !dbg !40
  %1609 = select i1 %.not2, float %1597, float %1608, !dbg !40
  %1610 = select i1 %.not2, float %1608, float %1598, !dbg !40
  %1611 = select i1 %.not3, float %1609, float %1603, !dbg !40
  %1612 = bitcast float %1611 to i32, !dbg !40
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1612), !dbg !40
  %1614 = bitcast i32 %1613 to float, !dbg !40
  %1615 = select i1 %.not3, float %1603, float %1614, !dbg !40
  %1616 = select i1 %.not3, float %1614, float %1609, !dbg !40
  %1617 = select i1 %.not3, float %1610, float %1604, !dbg !40
  %1618 = bitcast float %1617 to i32, !dbg !40
  %1619 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1618), !dbg !40
  %1620 = bitcast i32 %1619 to float, !dbg !40
  %1621 = select i1 %.not3, float %1604, float %1620, !dbg !40
  %1622 = select i1 %.not3, float %1620, float %1610, !dbg !40
  %1623 = extractelement <8 x float> %1512, i64 4, !dbg !40
  %1624 = extractelement <8 x float> %1512, i64 5, !dbg !40
  %1625 = extractelement <8 x float> %1512, i64 6, !dbg !40
  %1626 = extractelement <8 x float> %1512, i64 7, !dbg !40
  %1627 = select i1 %.not2, float %1624, float %1623, !dbg !40
  %1628 = bitcast float %1627 to i32, !dbg !40
  %1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1628), !dbg !40
  %1630 = bitcast i32 %1629 to float, !dbg !40
  %1631 = select i1 %.not2, float %1623, float %1630, !dbg !40
  %1632 = select i1 %.not2, float %1630, float %1624, !dbg !40
  %1633 = select i1 %.not2, float %1626, float %1625, !dbg !40
  %1634 = bitcast float %1633 to i32, !dbg !40
  %1635 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1634), !dbg !40
  %1636 = bitcast i32 %1635 to float, !dbg !40
  %1637 = select i1 %.not2, float %1625, float %1636, !dbg !40
  %1638 = select i1 %.not2, float %1636, float %1626, !dbg !40
  %1639 = select i1 %.not3, float %1637, float %1631, !dbg !40
  %1640 = bitcast float %1639 to i32, !dbg !40
  %1641 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1640), !dbg !40
  %1642 = bitcast i32 %1641 to float, !dbg !40
  %1643 = select i1 %.not3, float %1631, float %1642, !dbg !40
  %1644 = select i1 %.not3, float %1642, float %1637, !dbg !40
  %1645 = select i1 %.not3, float %1638, float %1632, !dbg !40
  %1646 = bitcast float %1645 to i32, !dbg !40
  %1647 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1646), !dbg !40
  %1648 = bitcast i32 %1647 to float, !dbg !40
  %1649 = select i1 %.not3, float %1632, float %1648, !dbg !40
  %1650 = select i1 %.not3, float %1648, float %1638, !dbg !40
  %1651 = extractelement <8 x float> %1541, i64 4, !dbg !40
  %1652 = extractelement <8 x float> %1541, i64 5, !dbg !40
  %1653 = extractelement <8 x float> %1541, i64 6, !dbg !40
  %1654 = extractelement <8 x float> %1541, i64 7, !dbg !40
  %1655 = select i1 %.not2, float %1652, float %1651, !dbg !40
  %1656 = bitcast float %1655 to i32, !dbg !40
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1656), !dbg !40
  %1658 = bitcast i32 %1657 to float, !dbg !40
  %1659 = select i1 %.not2, float %1651, float %1658, !dbg !40
  %1660 = select i1 %.not2, float %1658, float %1652, !dbg !40
  %1661 = select i1 %.not2, float %1654, float %1653, !dbg !40
  %1662 = bitcast float %1661 to i32, !dbg !40
  %1663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1662), !dbg !40
  %1664 = bitcast i32 %1663 to float, !dbg !40
  %1665 = select i1 %.not2, float %1653, float %1664, !dbg !40
  %1666 = select i1 %.not2, float %1664, float %1654, !dbg !40
  %1667 = select i1 %.not3, float %1665, float %1659, !dbg !40
  %1668 = bitcast float %1667 to i32, !dbg !40
  %1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1668), !dbg !40
  %1670 = bitcast i32 %1669 to float, !dbg !40
  %1671 = select i1 %.not3, float %1659, float %1670, !dbg !40
  %1672 = select i1 %.not3, float %1670, float %1665, !dbg !40
  %1673 = select i1 %.not3, float %1666, float %1660, !dbg !40
  %1674 = bitcast float %1673 to i32, !dbg !40
  %1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1674), !dbg !40
  %1676 = bitcast i32 %1675 to float, !dbg !40
  %1677 = select i1 %.not3, float %1660, float %1676, !dbg !40
  %1678 = select i1 %.not3, float %1676, float %1666, !dbg !40
  %1679 = extractelement <8 x float> %1570, i64 4, !dbg !40
  %1680 = extractelement <8 x float> %1570, i64 5, !dbg !40
  %1681 = extractelement <8 x float> %1570, i64 6, !dbg !40
  %1682 = extractelement <8 x float> %1570, i64 7, !dbg !40
  %1683 = select i1 %.not2, float %1680, float %1679, !dbg !40
  %1684 = bitcast float %1683 to i32, !dbg !40
  %1685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1684), !dbg !40
  %1686 = bitcast i32 %1685 to float, !dbg !40
  %1687 = select i1 %.not2, float %1679, float %1686, !dbg !40
  %1688 = select i1 %.not2, float %1686, float %1680, !dbg !40
  %1689 = select i1 %.not2, float %1682, float %1681, !dbg !40
  %1690 = bitcast float %1689 to i32, !dbg !40
  %1691 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1690), !dbg !40
  %1692 = bitcast i32 %1691 to float, !dbg !40
  %1693 = select i1 %.not2, float %1681, float %1692, !dbg !40
  %1694 = select i1 %.not2, float %1692, float %1682, !dbg !40
  %1695 = select i1 %.not3, float %1693, float %1687, !dbg !40
  %1696 = bitcast float %1695 to i32, !dbg !40
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1696), !dbg !40
  %1698 = select i1 %.not3, float %1694, float %1688, !dbg !40
  %1699 = bitcast float %1698 to i32, !dbg !40
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1699), !dbg !40
  %1701 = extractvalue [2 x <8 x float>] %31, 1, !dbg !40
  %1702 = extractelement <8 x float> %1701, i64 0, !dbg !40
  %1703 = extractelement <8 x float> %1701, i64 1, !dbg !40
  %1704 = extractelement <8 x float> %1701, i64 2, !dbg !40
  %1705 = extractelement <8 x float> %1701, i64 3, !dbg !40
  %1706 = select i1 %.not2, float %1703, float %1702, !dbg !40
  %1707 = bitcast float %1706 to i32, !dbg !40
  %1708 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1707), !dbg !40
  %1709 = bitcast i32 %1708 to float, !dbg !40
  %1710 = select i1 %.not2, float %1702, float %1709, !dbg !40
  %1711 = select i1 %.not2, float %1709, float %1703, !dbg !40
  %1712 = select i1 %.not2, float %1705, float %1704, !dbg !40
  %1713 = bitcast float %1712 to i32, !dbg !40
  %1714 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1713), !dbg !40
  %1715 = bitcast i32 %1714 to float, !dbg !40
  %1716 = select i1 %.not2, float %1704, float %1715, !dbg !40
  %1717 = select i1 %.not2, float %1715, float %1705, !dbg !40
  %1718 = select i1 %.not3, float %1716, float %1710, !dbg !40
  %1719 = bitcast float %1718 to i32, !dbg !40
  %1720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1719), !dbg !40
  %1721 = bitcast i32 %1720 to float, !dbg !40
  %1722 = select i1 %.not3, float %1710, float %1721, !dbg !40
  %1723 = select i1 %.not3, float %1721, float %1716, !dbg !40
  %1724 = select i1 %.not3, float %1717, float %1711, !dbg !40
  %1725 = bitcast float %1724 to i32, !dbg !40
  %1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1725), !dbg !40
  %1727 = bitcast i32 %1726 to float, !dbg !40
  %1728 = select i1 %.not3, float %1711, float %1727, !dbg !40
  %1729 = select i1 %.not3, float %1727, float %1717, !dbg !40
  %1730 = extractvalue [2 x <8 x float>] %30, 1, !dbg !40
  %1731 = extractelement <8 x float> %1730, i64 0, !dbg !40
  %1732 = extractelement <8 x float> %1730, i64 1, !dbg !40
  %1733 = extractelement <8 x float> %1730, i64 2, !dbg !40
  %1734 = extractelement <8 x float> %1730, i64 3, !dbg !40
  %1735 = select i1 %.not2, float %1732, float %1731, !dbg !40
  %1736 = bitcast float %1735 to i32, !dbg !40
  %1737 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1736), !dbg !40
  %1738 = bitcast i32 %1737 to float, !dbg !40
  %1739 = select i1 %.not2, float %1731, float %1738, !dbg !40
  %1740 = select i1 %.not2, float %1738, float %1732, !dbg !40
  %1741 = select i1 %.not2, float %1734, float %1733, !dbg !40
  %1742 = bitcast float %1741 to i32, !dbg !40
  %1743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1742), !dbg !40
  %1744 = bitcast i32 %1743 to float, !dbg !40
  %1745 = select i1 %.not2, float %1733, float %1744, !dbg !40
  %1746 = select i1 %.not2, float %1744, float %1734, !dbg !40
  %1747 = select i1 %.not3, float %1745, float %1739, !dbg !40
  %1748 = bitcast float %1747 to i32, !dbg !40
  %1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1748), !dbg !40
  %1750 = bitcast i32 %1749 to float, !dbg !40
  %1751 = select i1 %.not3, float %1739, float %1750, !dbg !40
  %1752 = select i1 %.not3, float %1750, float %1745, !dbg !40
  %1753 = select i1 %.not3, float %1746, float %1740, !dbg !40
  %1754 = bitcast float %1753 to i32, !dbg !40
  %1755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1754), !dbg !40
  %1756 = bitcast i32 %1755 to float, !dbg !40
  %1757 = select i1 %.not3, float %1740, float %1756, !dbg !40
  %1758 = select i1 %.not3, float %1756, float %1746, !dbg !40
  %1759 = extractvalue [2 x <8 x float>] %29, 1, !dbg !40
  %1760 = extractelement <8 x float> %1759, i64 0, !dbg !40
  %1761 = extractelement <8 x float> %1759, i64 1, !dbg !40
  %1762 = extractelement <8 x float> %1759, i64 2, !dbg !40
  %1763 = extractelement <8 x float> %1759, i64 3, !dbg !40
  %1764 = select i1 %.not2, float %1761, float %1760, !dbg !40
  %1765 = bitcast float %1764 to i32, !dbg !40
  %1766 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1765), !dbg !40
  %1767 = bitcast i32 %1766 to float, !dbg !40
  %1768 = select i1 %.not2, float %1760, float %1767, !dbg !40
  %1769 = select i1 %.not2, float %1767, float %1761, !dbg !40
  %1770 = select i1 %.not2, float %1763, float %1762, !dbg !40
  %1771 = bitcast float %1770 to i32, !dbg !40
  %1772 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1771), !dbg !40
  %1773 = bitcast i32 %1772 to float, !dbg !40
  %1774 = select i1 %.not2, float %1762, float %1773, !dbg !40
  %1775 = select i1 %.not2, float %1773, float %1763, !dbg !40
  %1776 = select i1 %.not3, float %1774, float %1768, !dbg !40
  %1777 = bitcast float %1776 to i32, !dbg !40
  %1778 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1777), !dbg !40
  %1779 = bitcast i32 %1778 to float, !dbg !40
  %1780 = select i1 %.not3, float %1768, float %1779, !dbg !40
  %1781 = select i1 %.not3, float %1779, float %1774, !dbg !40
  %1782 = select i1 %.not3, float %1775, float %1769, !dbg !40
  %1783 = bitcast float %1782 to i32, !dbg !40
  %1784 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1783), !dbg !40
  %1785 = bitcast i32 %1784 to float, !dbg !40
  %1786 = select i1 %.not3, float %1769, float %1785, !dbg !40
  %1787 = select i1 %.not3, float %1785, float %1775, !dbg !40
  %1788 = extractvalue [2 x <8 x float>] %27, 1, !dbg !40
  %1789 = extractelement <8 x float> %1788, i64 0, !dbg !40
  %1790 = extractelement <8 x float> %1788, i64 1, !dbg !40
  %1791 = extractelement <8 x float> %1788, i64 2, !dbg !40
  %1792 = extractelement <8 x float> %1788, i64 3, !dbg !40
  %1793 = select i1 %.not2, float %1790, float %1789, !dbg !40
  %1794 = bitcast float %1793 to i32, !dbg !40
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1794), !dbg !40
  %1796 = bitcast i32 %1795 to float, !dbg !40
  %1797 = select i1 %.not2, float %1789, float %1796, !dbg !40
  %1798 = select i1 %.not2, float %1796, float %1790, !dbg !40
  %1799 = select i1 %.not2, float %1792, float %1791, !dbg !40
  %1800 = bitcast float %1799 to i32, !dbg !40
  %1801 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1800), !dbg !40
  %1802 = bitcast i32 %1801 to float, !dbg !40
  %1803 = select i1 %.not2, float %1791, float %1802, !dbg !40
  %1804 = select i1 %.not2, float %1802, float %1792, !dbg !40
  %1805 = select i1 %.not3, float %1803, float %1797, !dbg !40
  %1806 = bitcast float %1805 to i32, !dbg !40
  %1807 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1806), !dbg !40
  %1808 = select i1 %.not3, float %1804, float %1798, !dbg !40
  %1809 = bitcast float %1808 to i32, !dbg !40
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1809), !dbg !40
  %1811 = extractvalue [2 x <1 x float>] %28, 1, !dbg !43
  %1812 = shufflevector <1 x float> %1811, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %1813 = extractelement <8 x float> %1701, i64 4, !dbg !40
  %1814 = extractelement <8 x float> %1701, i64 5, !dbg !40
  %1815 = extractelement <8 x float> %1701, i64 6, !dbg !40
  %1816 = extractelement <8 x float> %1701, i64 7, !dbg !40
  %1817 = select i1 %.not2, float %1814, float %1813, !dbg !40
  %1818 = bitcast float %1817 to i32, !dbg !40
  %1819 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1818), !dbg !40
  %1820 = bitcast i32 %1819 to float, !dbg !40
  %1821 = select i1 %.not2, float %1813, float %1820, !dbg !40
  %1822 = select i1 %.not2, float %1820, float %1814, !dbg !40
  %1823 = select i1 %.not2, float %1816, float %1815, !dbg !40
  %1824 = bitcast float %1823 to i32, !dbg !40
  %1825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1824), !dbg !40
  %1826 = bitcast i32 %1825 to float, !dbg !40
  %1827 = select i1 %.not2, float %1815, float %1826, !dbg !40
  %1828 = select i1 %.not2, float %1826, float %1816, !dbg !40
  %1829 = select i1 %.not3, float %1827, float %1821, !dbg !40
  %1830 = bitcast float %1829 to i32, !dbg !40
  %1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1830), !dbg !40
  %1832 = bitcast i32 %1831 to float, !dbg !40
  %1833 = select i1 %.not3, float %1821, float %1832, !dbg !40
  %1834 = select i1 %.not3, float %1832, float %1827, !dbg !40
  %1835 = select i1 %.not3, float %1828, float %1822, !dbg !40
  %1836 = bitcast float %1835 to i32, !dbg !40
  %1837 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1836), !dbg !40
  %1838 = bitcast i32 %1837 to float, !dbg !40
  %1839 = select i1 %.not3, float %1822, float %1838, !dbg !40
  %1840 = select i1 %.not3, float %1838, float %1828, !dbg !40
  %1841 = extractelement <8 x float> %1730, i64 4, !dbg !40
  %1842 = extractelement <8 x float> %1730, i64 5, !dbg !40
  %1843 = extractelement <8 x float> %1730, i64 6, !dbg !40
  %1844 = extractelement <8 x float> %1730, i64 7, !dbg !40
  %1845 = select i1 %.not2, float %1842, float %1841, !dbg !40
  %1846 = bitcast float %1845 to i32, !dbg !40
  %1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1846), !dbg !40
  %1848 = bitcast i32 %1847 to float, !dbg !40
  %1849 = select i1 %.not2, float %1841, float %1848, !dbg !40
  %1850 = select i1 %.not2, float %1848, float %1842, !dbg !40
  %1851 = select i1 %.not2, float %1844, float %1843, !dbg !40
  %1852 = bitcast float %1851 to i32, !dbg !40
  %1853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1852), !dbg !40
  %1854 = bitcast i32 %1853 to float, !dbg !40
  %1855 = select i1 %.not2, float %1843, float %1854, !dbg !40
  %1856 = select i1 %.not2, float %1854, float %1844, !dbg !40
  %1857 = select i1 %.not3, float %1855, float %1849, !dbg !40
  %1858 = bitcast float %1857 to i32, !dbg !40
  %1859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1858), !dbg !40
  %1860 = bitcast i32 %1859 to float, !dbg !40
  %1861 = select i1 %.not3, float %1849, float %1860, !dbg !40
  %1862 = select i1 %.not3, float %1860, float %1855, !dbg !40
  %1863 = select i1 %.not3, float %1856, float %1850, !dbg !40
  %1864 = bitcast float %1863 to i32, !dbg !40
  %1865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1864), !dbg !40
  %1866 = bitcast i32 %1865 to float, !dbg !40
  %1867 = select i1 %.not3, float %1850, float %1866, !dbg !40
  %1868 = select i1 %.not3, float %1866, float %1856, !dbg !40
  %1869 = extractelement <8 x float> %1759, i64 4, !dbg !40
  %1870 = extractelement <8 x float> %1759, i64 5, !dbg !40
  %1871 = extractelement <8 x float> %1759, i64 6, !dbg !40
  %1872 = extractelement <8 x float> %1759, i64 7, !dbg !40
  %1873 = select i1 %.not2, float %1870, float %1869, !dbg !40
  %1874 = bitcast float %1873 to i32, !dbg !40
  %1875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1874), !dbg !40
  %1876 = bitcast i32 %1875 to float, !dbg !40
  %1877 = select i1 %.not2, float %1869, float %1876, !dbg !40
  %1878 = select i1 %.not2, float %1876, float %1870, !dbg !40
  %1879 = select i1 %.not2, float %1872, float %1871, !dbg !40
  %1880 = bitcast float %1879 to i32, !dbg !40
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1880), !dbg !40
  %1882 = bitcast i32 %1881 to float, !dbg !40
  %1883 = select i1 %.not2, float %1871, float %1882, !dbg !40
  %1884 = select i1 %.not2, float %1882, float %1872, !dbg !40
  %1885 = select i1 %.not3, float %1883, float %1877, !dbg !40
  %1886 = bitcast float %1885 to i32, !dbg !40
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1886), !dbg !40
  %1888 = bitcast i32 %1887 to float, !dbg !40
  %1889 = select i1 %.not3, float %1877, float %1888, !dbg !40
  %1890 = select i1 %.not3, float %1888, float %1883, !dbg !40
  %1891 = select i1 %.not3, float %1884, float %1878, !dbg !40
  %1892 = bitcast float %1891 to i32, !dbg !40
  %1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1892), !dbg !40
  %1894 = bitcast i32 %1893 to float, !dbg !40
  %1895 = select i1 %.not3, float %1878, float %1894, !dbg !40
  %1896 = select i1 %.not3, float %1894, float %1884, !dbg !40
  %1897 = extractelement <8 x float> %1788, i64 4, !dbg !40
  %1898 = extractelement <8 x float> %1788, i64 5, !dbg !40
  %1899 = extractelement <8 x float> %1788, i64 6, !dbg !40
  %1900 = extractelement <8 x float> %1788, i64 7, !dbg !40
  %1901 = select i1 %.not2, float %1898, float %1897, !dbg !40
  %1902 = bitcast float %1901 to i32, !dbg !40
  %1903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1902), !dbg !40
  %1904 = bitcast i32 %1903 to float, !dbg !40
  %1905 = select i1 %.not2, float %1897, float %1904, !dbg !40
  %1906 = select i1 %.not2, float %1904, float %1898, !dbg !40
  %1907 = select i1 %.not2, float %1900, float %1899, !dbg !40
  %1908 = bitcast float %1907 to i32, !dbg !40
  %1909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1484, i32 %1908), !dbg !40
  %1910 = bitcast i32 %1909 to float, !dbg !40
  %1911 = select i1 %.not2, float %1899, float %1910, !dbg !40
  %1912 = select i1 %.not2, float %1910, float %1900, !dbg !40
  %1913 = select i1 %.not3, float %1911, float %1905, !dbg !40
  %1914 = bitcast float %1913 to i32, !dbg !40
  %1915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1914), !dbg !40
  %1916 = select i1 %.not3, float %1912, float %1906, !dbg !40
  %1917 = bitcast float %1916 to i32, !dbg !40
  %1918 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1500, i32 %1917), !dbg !40
  br i1 true, label %1919, label %1983, !dbg !45

1919:                                             ; preds = %1468
  %1920 = bitcast i32 %1915 to float, !dbg !40
  %1921 = select i1 %.not3, float %1905, float %1920, !dbg !40
  %1922 = insertelement <4 x float> poison, float %1921, i64 0, !dbg !43
  %1923 = bitcast i32 %1918 to float, !dbg !40
  %1924 = select i1 %.not3, float %1906, float %1923, !dbg !40
  %1925 = insertelement <4 x float> %1922, float %1924, i64 1, !dbg !43
  %1926 = select i1 %.not3, float %1920, float %1911, !dbg !40
  %1927 = insertelement <4 x float> %1925, float %1926, i64 2, !dbg !43
  %1928 = select i1 %.not3, float %1923, float %1912, !dbg !40
  %1929 = insertelement <4 x float> %1927, float %1928, i64 3, !dbg !43
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1812, !dbg !43
  %normalized = fmul <4 x float> %1929, %row.reciprocal, !dbg !43
  %1930 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !44
  %1931 = extractelement <4 x half> %1930, i64 3, !dbg !44
  %1932 = extractelement <4 x half> %1930, i64 2, !dbg !44
  %1933 = extractelement <4 x half> %1930, i64 1, !dbg !44
  %1934 = extractelement <4 x half> %1930, i64 0, !dbg !44
  %1935 = bitcast i32 %1807 to float, !dbg !40
  %1936 = select i1 %.not3, float %1797, float %1935, !dbg !40
  %1937 = insertelement <4 x float> poison, float %1936, i64 0, !dbg !43
  %1938 = bitcast i32 %1810 to float, !dbg !40
  %1939 = select i1 %.not3, float %1798, float %1938, !dbg !40
  %1940 = insertelement <4 x float> %1937, float %1939, i64 1, !dbg !43
  %1941 = select i1 %.not3, float %1935, float %1803, !dbg !40
  %1942 = insertelement <4 x float> %1940, float %1941, i64 2, !dbg !43
  %1943 = select i1 %.not3, float %1938, float %1804, !dbg !40
  %1944 = insertelement <4 x float> %1942, float %1943, i64 3, !dbg !43
  %normalized75 = fmul <4 x float> %1944, %row.reciprocal, !dbg !43
  %1945 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !44
  %1946 = extractelement <4 x half> %1945, i64 3, !dbg !44
  %1947 = extractelement <4 x half> %1945, i64 2, !dbg !44
  %1948 = extractelement <4 x half> %1945, i64 1, !dbg !44
  %1949 = extractelement <4 x half> %1945, i64 0, !dbg !44
  %1950 = bitcast i32 %1697 to float, !dbg !40
  %1951 = select i1 %.not3, float %1687, float %1950, !dbg !40
  %1952 = insertelement <4 x float> poison, float %1951, i64 0, !dbg !43
  %1953 = bitcast i32 %1700 to float, !dbg !40
  %1954 = select i1 %.not3, float %1688, float %1953, !dbg !40
  %1955 = insertelement <4 x float> %1952, float %1954, i64 1, !dbg !43
  %1956 = select i1 %.not3, float %1950, float %1693, !dbg !40
  %1957 = insertelement <4 x float> %1955, float %1956, i64 2, !dbg !43
  %1958 = select i1 %.not3, float %1953, float %1694, !dbg !40
  %1959 = insertelement <4 x float> %1957, float %1958, i64 3, !dbg !43
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1594, !dbg !43
  %normalized77 = fmul <4 x float> %1959, %row.reciprocal76, !dbg !43
  %1960 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !44
  %1961 = extractelement <4 x half> %1960, i64 3, !dbg !44
  %1962 = extractelement <4 x half> %1960, i64 2, !dbg !44
  %1963 = extractelement <4 x half> %1960, i64 1, !dbg !44
  %1964 = extractelement <4 x half> %1960, i64 0, !dbg !44
  %1965 = bitcast i32 %1589 to float, !dbg !40
  %1966 = select i1 %.not3, float %1579, float %1965, !dbg !40
  %1967 = insertelement <4 x float> poison, float %1966, i64 0, !dbg !43
  %1968 = bitcast i32 %1592 to float, !dbg !40
  %1969 = select i1 %.not3, float %1580, float %1968, !dbg !40
  %1970 = insertelement <4 x float> %1967, float %1969, i64 1, !dbg !43
  %1971 = select i1 %.not3, float %1965, float %1585, !dbg !40
  %1972 = insertelement <4 x float> %1970, float %1971, i64 2, !dbg !43
  %1973 = select i1 %.not3, float %1968, float %1586, !dbg !40
  %1974 = insertelement <4 x float> %1972, float %1973, i64 3, !dbg !43
  %normalized78 = fmul <4 x float> %1974, %row.reciprocal76, !dbg !43
  %1975 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !44
  %1976 = extractelement <4 x half> %1975, i64 3, !dbg !44
  %1977 = extractelement <4 x half> %1975, i64 2, !dbg !44
  %1978 = extractelement <4 x half> %1975, i64 1, !dbg !44
  %1979 = extractelement <4 x half> %1975, i64 0, !dbg !44
  %1980 = add nsw i64 %8, %13, !dbg !45
  %1981 = shl nsw i64 %1980, 19, !dbg !45
  %1982 = and i64 %1981, 16252928, !dbg !45
  %address.thread394 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !45, !range !57, !frisk.register_lifetimes !29
  %address.local395 = zext nneg i32 %address.thread394 to i64, !dbg !37
  %address.local396 = add nsw i64 %16, %address.local395, !dbg !45
  %address.local397 = lshr i64 %address.local395, 6, !dbg !37
  %address.local398 = shl nuw nsw i64 %address.local397, 4, !dbg !45
  %address.local399 = add nsw i64 %address.local396, %address.local398, !dbg !45
  %address.local400 = and i64 %address.local395, 240, !dbg !45
  %address.local401 = sub nsw i64 %address.local399, %address.local400, !dbg !45
  %address.local402 = add nsw i64 %address.local401, %17, !dbg !45
  %address.local403 = shl nuw nsw i64 %address.local402, 7, !dbg !45
  %address.local404 = add nuw nsw i64 %1982, %address.local403, !dbg !45
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local406 = lshr i64 %address.local395, 2, !dbg !37
  %address.local407 = and i64 %address.local406, 12, !dbg !37
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local407, !dbg !45
  store half %1979, ptr addrspace(1) %address.local408, align 2, !dbg !45
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local410 = or disjoint i64 %address.local407, 1, !dbg !45
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !45
  store half %1978, ptr addrspace(1) %address.local411, align 2, !dbg !45
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local413 = or disjoint i64 %address.local407, 2, !dbg !45
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local413, !dbg !45
  store half %1977, ptr addrspace(1) %address.local414, align 2, !dbg !45
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local416 = or disjoint i64 %address.local407, 3, !dbg !45
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local416, !dbg !45
  store half %1976, ptr addrspace(1) %address.local417, align 2, !dbg !45
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local419 = or disjoint i64 %address.local407, 16, !dbg !45
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local419, !dbg !45
  store half %1964, ptr addrspace(1) %address.local420, align 2, !dbg !45
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local422 = or disjoint i64 %address.local407, 17, !dbg !45
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local422, !dbg !45
  store half %1963, ptr addrspace(1) %address.local423, align 2, !dbg !45
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local425 = or disjoint i64 %address.local407, 18, !dbg !45
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local425, !dbg !45
  store half %1962, ptr addrspace(1) %address.local426, align 2, !dbg !45
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local428 = or disjoint i64 %address.local407, 19, !dbg !45
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local428, !dbg !45
  store half %1961, ptr addrspace(1) %address.local429, align 2, !dbg !45
  %address.local430 = shl nsw i64 %address.local402, 7, !dbg !45
  %address.local431 = add i64 %address.local430, 8192, !dbg !45
  %address.local432 = add nuw nsw i64 %1982, %address.local431, !dbg !45
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local407, !dbg !45
  store half %1949, ptr addrspace(1) %address.local434, align 2, !dbg !45
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local410, !dbg !45
  store half %1948, ptr addrspace(1) %address.local436, align 2, !dbg !45
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local413, !dbg !45
  store half %1947, ptr addrspace(1) %address.local438, align 2, !dbg !45
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local416, !dbg !45
  store half %1946, ptr addrspace(1) %address.local440, align 2, !dbg !45
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local419, !dbg !45
  store half %1934, ptr addrspace(1) %address.local442, align 2, !dbg !45
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local422, !dbg !45
  store half %1933, ptr addrspace(1) %address.local444, align 2, !dbg !45
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local425, !dbg !45
  store half %1932, ptr addrspace(1) %address.local446, align 2, !dbg !45
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local447, i64 %address.local428, !dbg !45
  store half %1931, ptr addrspace(1) %address.local448, align 2, !dbg !45
  br label %1983, !dbg !45

1983:                                             ; preds = %1919, %1468
  br i1 true, label %1984, label %2022, !dbg !45

1984:                                             ; preds = %1983
  %1985 = insertelement <4 x float> poison, float %1889, i64 0, !dbg !43
  %1986 = insertelement <4 x float> %1985, float %1895, i64 1, !dbg !43
  %1987 = insertelement <4 x float> %1986, float %1890, i64 2, !dbg !43
  %1988 = insertelement <4 x float> %1987, float %1896, i64 3, !dbg !43
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1812, !dbg !43
  %normalized80 = fmul <4 x float> %1988, %row.reciprocal79, !dbg !43
  %1989 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !44
  %1990 = extractelement <4 x half> %1989, i64 3, !dbg !44
  %1991 = extractelement <4 x half> %1989, i64 2, !dbg !44
  %1992 = extractelement <4 x half> %1989, i64 1, !dbg !44
  %1993 = extractelement <4 x half> %1989, i64 0, !dbg !44
  %1994 = insertelement <4 x float> poison, float %1780, i64 0, !dbg !43
  %1995 = insertelement <4 x float> %1994, float %1786, i64 1, !dbg !43
  %1996 = insertelement <4 x float> %1995, float %1781, i64 2, !dbg !43
  %1997 = insertelement <4 x float> %1996, float %1787, i64 3, !dbg !43
  %normalized81 = fmul <4 x float> %1997, %row.reciprocal79, !dbg !43
  %1998 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !44
  %1999 = extractelement <4 x half> %1998, i64 3, !dbg !44
  %2000 = extractelement <4 x half> %1998, i64 2, !dbg !44
  %2001 = extractelement <4 x half> %1998, i64 1, !dbg !44
  %2002 = extractelement <4 x half> %1998, i64 0, !dbg !44
  %2003 = insertelement <4 x float> poison, float %1671, i64 0, !dbg !43
  %2004 = insertelement <4 x float> %2003, float %1677, i64 1, !dbg !43
  %2005 = insertelement <4 x float> %2004, float %1672, i64 2, !dbg !43
  %2006 = insertelement <4 x float> %2005, float %1678, i64 3, !dbg !43
  %row.reciprocal82 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1594, !dbg !43
  %normalized83 = fmul <4 x float> %2006, %row.reciprocal82, !dbg !43
  %2007 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !44
  %2008 = extractelement <4 x half> %2007, i64 3, !dbg !44
  %2009 = extractelement <4 x half> %2007, i64 2, !dbg !44
  %2010 = extractelement <4 x half> %2007, i64 1, !dbg !44
  %2011 = extractelement <4 x half> %2007, i64 0, !dbg !44
  %2012 = insertelement <4 x float> poison, float %1562, i64 0, !dbg !43
  %2013 = insertelement <4 x float> %2012, float %1568, i64 1, !dbg !43
  %2014 = insertelement <4 x float> %2013, float %1563, i64 2, !dbg !43
  %2015 = insertelement <4 x float> %2014, float %1569, i64 3, !dbg !43
  %normalized84 = fmul <4 x float> %2015, %row.reciprocal82, !dbg !43
  %2016 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !44
  %2017 = extractelement <4 x half> %2016, i64 3, !dbg !44
  %2018 = extractelement <4 x half> %2016, i64 2, !dbg !44
  %2019 = extractelement <4 x half> %2016, i64 1, !dbg !44
  %2020 = extractelement <4 x half> %2016, i64 0, !dbg !44
  %2021 = shl nuw nsw i64 %15, 19, !dbg !45
  %address.thread449 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !45, !range !57, !frisk.register_lifetimes !29
  %address.local450 = zext nneg i32 %address.thread449 to i64, !dbg !37
  %address.local451 = add nsw i64 %16, %address.local450, !dbg !45
  %address.local452 = lshr i64 %address.local450, 6, !dbg !37
  %address.local453 = shl nuw nsw i64 %address.local452, 4, !dbg !45
  %address.local454 = add nsw i64 %address.local451, %address.local453, !dbg !45
  %address.local455 = and i64 %address.local450, 240, !dbg !45
  %address.local456 = sub nsw i64 %address.local454, %address.local455, !dbg !45
  %address.local457 = add nsw i64 %address.local456, %17, !dbg !45
  %address.local458 = shl nuw nsw i64 %address.local457, 7, !dbg !45
  %address.local459 = add nuw nsw i64 %2021, %address.local458, !dbg !45
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local461 = lshr i64 %address.local450, 2, !dbg !37
  %address.local462 = and i64 %address.local461, 12, !dbg !37
  %address.local463 = or disjoint i64 %address.local462, 32, !dbg !45
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local463, !dbg !45
  store half %2020, ptr addrspace(1) %address.local464, align 2, !dbg !45
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local466 = or disjoint i64 %address.local462, 33, !dbg !45
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local466, !dbg !45
  store half %2019, ptr addrspace(1) %address.local467, align 2, !dbg !45
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local469 = or disjoint i64 %address.local462, 34, !dbg !45
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local469, !dbg !45
  store half %2018, ptr addrspace(1) %address.local470, align 2, !dbg !45
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local472 = or disjoint i64 %address.local462, 35, !dbg !45
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local472, !dbg !45
  store half %2017, ptr addrspace(1) %address.local473, align 2, !dbg !45
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local475 = or disjoint i64 %address.local462, 48, !dbg !45
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local474, i64 %address.local475, !dbg !45
  store half %2011, ptr addrspace(1) %address.local476, align 2, !dbg !45
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local478 = or disjoint i64 %address.local462, 49, !dbg !45
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local478, !dbg !45
  store half %2010, ptr addrspace(1) %address.local479, align 2, !dbg !45
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local481 = or disjoint i64 %address.local462, 50, !dbg !45
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local480, i64 %address.local481, !dbg !45
  store half %2009, ptr addrspace(1) %address.local482, align 2, !dbg !45
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local484 = or i64 %address.local461, 51, !dbg !45
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local483, i64 %address.local484, !dbg !45
  store half %2008, ptr addrspace(1) %address.local485, align 2, !dbg !45
  %address.local486 = shl nsw i64 %address.local457, 7, !dbg !45
  %address.local487 = add i64 %address.local486, 8192, !dbg !45
  %address.local488 = add nuw nsw i64 %2021, %address.local487, !dbg !45
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local463, !dbg !45
  store half %2002, ptr addrspace(1) %address.local490, align 2, !dbg !45
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local466, !dbg !45
  store half %2001, ptr addrspace(1) %address.local492, align 2, !dbg !45
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local469, !dbg !45
  store half %2000, ptr addrspace(1) %address.local494, align 2, !dbg !45
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local472, !dbg !45
  store half %1999, ptr addrspace(1) %address.local496, align 2, !dbg !45
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local475, !dbg !45
  store half %1993, ptr addrspace(1) %address.local498, align 2, !dbg !45
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local478, !dbg !45
  store half %1992, ptr addrspace(1) %address.local500, align 2, !dbg !45
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local481, !dbg !45
  store half %1991, ptr addrspace(1) %address.local502, align 2, !dbg !45
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local503, i64 %address.local484, !dbg !45
  store half %1990, ptr addrspace(1) %address.local504, align 2, !dbg !45
  br label %2022, !dbg !45

2022:                                             ; preds = %1984, %1983
  br i1 true, label %2023, label %2061, !dbg !45

2023:                                             ; preds = %2022
  %2024 = insertelement <4 x float> poison, float %1861, i64 0, !dbg !43
  %2025 = insertelement <4 x float> %2024, float %1867, i64 1, !dbg !43
  %2026 = insertelement <4 x float> %2025, float %1862, i64 2, !dbg !43
  %2027 = insertelement <4 x float> %2026, float %1868, i64 3, !dbg !43
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1812, !dbg !43
  %normalized86 = fmul <4 x float> %2027, %row.reciprocal85, !dbg !43
  %2028 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !44
  %2029 = extractelement <4 x half> %2028, i64 3, !dbg !44
  %2030 = extractelement <4 x half> %2028, i64 2, !dbg !44
  %2031 = extractelement <4 x half> %2028, i64 1, !dbg !44
  %2032 = extractelement <4 x half> %2028, i64 0, !dbg !44
  %2033 = insertelement <4 x float> poison, float %1751, i64 0, !dbg !43
  %2034 = insertelement <4 x float> %2033, float %1757, i64 1, !dbg !43
  %2035 = insertelement <4 x float> %2034, float %1752, i64 2, !dbg !43
  %2036 = insertelement <4 x float> %2035, float %1758, i64 3, !dbg !43
  %normalized87 = fmul <4 x float> %2036, %row.reciprocal85, !dbg !43
  %2037 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !44
  %2038 = extractelement <4 x half> %2037, i64 3, !dbg !44
  %2039 = extractelement <4 x half> %2037, i64 2, !dbg !44
  %2040 = extractelement <4 x half> %2037, i64 1, !dbg !44
  %2041 = extractelement <4 x half> %2037, i64 0, !dbg !44
  %2042 = insertelement <4 x float> poison, float %1643, i64 0, !dbg !43
  %2043 = insertelement <4 x float> %2042, float %1649, i64 1, !dbg !43
  %2044 = insertelement <4 x float> %2043, float %1644, i64 2, !dbg !43
  %2045 = insertelement <4 x float> %2044, float %1650, i64 3, !dbg !43
  %row.reciprocal88 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1594, !dbg !43
  %normalized89 = fmul <4 x float> %2045, %row.reciprocal88, !dbg !43
  %2046 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !44
  %2047 = extractelement <4 x half> %2046, i64 3, !dbg !44
  %2048 = extractelement <4 x half> %2046, i64 2, !dbg !44
  %2049 = extractelement <4 x half> %2046, i64 1, !dbg !44
  %2050 = extractelement <4 x half> %2046, i64 0, !dbg !44
  %2051 = insertelement <4 x float> poison, float %1533, i64 0, !dbg !43
  %2052 = insertelement <4 x float> %2051, float %1539, i64 1, !dbg !43
  %2053 = insertelement <4 x float> %2052, float %1534, i64 2, !dbg !43
  %2054 = insertelement <4 x float> %2053, float %1540, i64 3, !dbg !43
  %normalized90 = fmul <4 x float> %2054, %row.reciprocal88, !dbg !43
  %2055 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !44
  %2056 = extractelement <4 x half> %2055, i64 3, !dbg !44
  %2057 = extractelement <4 x half> %2055, i64 2, !dbg !44
  %2058 = extractelement <4 x half> %2055, i64 1, !dbg !44
  %2059 = extractelement <4 x half> %2055, i64 0, !dbg !44
  %2060 = shl nuw nsw i64 %15, 19, !dbg !45
  %address.thread505 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !45, !range !57, !frisk.register_lifetimes !29
  %address.local506 = zext nneg i32 %address.thread505 to i64, !dbg !37
  %address.local507 = add nsw i64 %16, %address.local506, !dbg !45
  %address.local508 = lshr i64 %address.local506, 6, !dbg !37
  %address.local509 = shl nuw nsw i64 %address.local508, 4, !dbg !45
  %address.local510 = add nsw i64 %address.local507, %address.local509, !dbg !45
  %address.local511 = and i64 %address.local506, 240, !dbg !45
  %address.local512 = sub nsw i64 %address.local510, %address.local511, !dbg !45
  %address.local513 = add nsw i64 %address.local512, %17, !dbg !45
  %address.local514 = shl nuw nsw i64 %address.local513, 7, !dbg !45
  %address.local515 = add nuw nsw i64 %2060, %address.local514, !dbg !45
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local517 = lshr i64 %address.local506, 2, !dbg !37
  %address.local518 = and i64 %address.local517, 12, !dbg !37
  %address.local519 = or disjoint i64 %address.local518, 64, !dbg !45
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local519, !dbg !45
  store half %2059, ptr addrspace(1) %address.local520, align 2, !dbg !45
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local522 = or disjoint i64 %address.local518, 65, !dbg !45
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local522, !dbg !45
  store half %2058, ptr addrspace(1) %address.local523, align 2, !dbg !45
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local525 = or disjoint i64 %address.local518, 66, !dbg !45
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local524, i64 %address.local525, !dbg !45
  store half %2057, ptr addrspace(1) %address.local526, align 2, !dbg !45
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local528 = or disjoint i64 %address.local518, 67, !dbg !45
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local528, !dbg !45
  store half %2056, ptr addrspace(1) %address.local529, align 2, !dbg !45
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local531 = or disjoint i64 %address.local518, 80, !dbg !45
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local530, i64 %address.local531, !dbg !45
  store half %2050, ptr addrspace(1) %address.local532, align 2, !dbg !45
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local534 = or disjoint i64 %address.local518, 81, !dbg !45
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !45
  store half %2049, ptr addrspace(1) %address.local535, align 2, !dbg !45
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local537 = or disjoint i64 %address.local518, 82, !dbg !45
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local537, !dbg !45
  store half %2048, ptr addrspace(1) %address.local538, align 2, !dbg !45
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local540 = or disjoint i64 %address.local518, 83, !dbg !45
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local540, !dbg !45
  store half %2047, ptr addrspace(1) %address.local541, align 2, !dbg !45
  %address.local542 = shl nsw i64 %address.local513, 7, !dbg !45
  %address.local543 = add i64 %address.local542, 8192, !dbg !45
  %address.local544 = add nuw nsw i64 %2060, %address.local543, !dbg !45
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local519, !dbg !45
  store half %2041, ptr addrspace(1) %address.local546, align 2, !dbg !45
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local522, !dbg !45
  store half %2040, ptr addrspace(1) %address.local548, align 2, !dbg !45
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local525, !dbg !45
  store half %2039, ptr addrspace(1) %address.local550, align 2, !dbg !45
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local528, !dbg !45
  store half %2038, ptr addrspace(1) %address.local552, align 2, !dbg !45
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local531, !dbg !45
  store half %2032, ptr addrspace(1) %address.local554, align 2, !dbg !45
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local534, !dbg !45
  store half %2031, ptr addrspace(1) %address.local556, align 2, !dbg !45
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local537, !dbg !45
  store half %2030, ptr addrspace(1) %address.local558, align 2, !dbg !45
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local540, !dbg !45
  store half %2029, ptr addrspace(1) %address.local560, align 2, !dbg !45
  br label %2061, !dbg !45

2061:                                             ; preds = %2023, %2022
  br i1 true, label %2062, label %2100, !dbg !45

2062:                                             ; preds = %2061
  %2063 = insertelement <4 x float> poison, float %1833, i64 0, !dbg !43
  %2064 = insertelement <4 x float> %2063, float %1839, i64 1, !dbg !43
  %2065 = insertelement <4 x float> %2064, float %1834, i64 2, !dbg !43
  %2066 = insertelement <4 x float> %2065, float %1840, i64 3, !dbg !43
  %row.reciprocal91 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1812, !dbg !43
  %normalized92 = fmul <4 x float> %2066, %row.reciprocal91, !dbg !43
  %2067 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !44
  %2068 = extractelement <4 x half> %2067, i64 3, !dbg !44
  %2069 = extractelement <4 x half> %2067, i64 2, !dbg !44
  %2070 = extractelement <4 x half> %2067, i64 1, !dbg !44
  %2071 = extractelement <4 x half> %2067, i64 0, !dbg !44
  %2072 = insertelement <4 x float> poison, float %1722, i64 0, !dbg !43
  %2073 = insertelement <4 x float> %2072, float %1728, i64 1, !dbg !43
  %2074 = insertelement <4 x float> %2073, float %1723, i64 2, !dbg !43
  %2075 = insertelement <4 x float> %2074, float %1729, i64 3, !dbg !43
  %normalized93 = fmul <4 x float> %2075, %row.reciprocal91, !dbg !43
  %2076 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !44
  %2077 = extractelement <4 x half> %2076, i64 3, !dbg !44
  %2078 = extractelement <4 x half> %2076, i64 2, !dbg !44
  %2079 = extractelement <4 x half> %2076, i64 1, !dbg !44
  %2080 = extractelement <4 x half> %2076, i64 0, !dbg !44
  %2081 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !43
  %2082 = insertelement <4 x float> %2081, float %1621, i64 1, !dbg !43
  %2083 = insertelement <4 x float> %2082, float %1616, i64 2, !dbg !43
  %2084 = insertelement <4 x float> %2083, float %1622, i64 3, !dbg !43
  %row.reciprocal94 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1594, !dbg !43
  %normalized95 = fmul <4 x float> %2084, %row.reciprocal94, !dbg !43
  %2085 = fptrunc <4 x float> %normalized95 to <4 x half>, !dbg !44
  %2086 = extractelement <4 x half> %2085, i64 3, !dbg !44
  %2087 = extractelement <4 x half> %2085, i64 2, !dbg !44
  %2088 = extractelement <4 x half> %2085, i64 1, !dbg !44
  %2089 = extractelement <4 x half> %2085, i64 0, !dbg !44
  %2090 = insertelement <4 x float> poison, float %1504, i64 0, !dbg !43
  %2091 = insertelement <4 x float> %2090, float %1510, i64 1, !dbg !43
  %2092 = insertelement <4 x float> %2091, float %1505, i64 2, !dbg !43
  %2093 = insertelement <4 x float> %2092, float %1511, i64 3, !dbg !43
  %normalized96 = fmul <4 x float> %2093, %row.reciprocal94, !dbg !43
  %2094 = fptrunc <4 x float> %normalized96 to <4 x half>, !dbg !44
  %2095 = extractelement <4 x half> %2094, i64 3, !dbg !44
  %2096 = extractelement <4 x half> %2094, i64 2, !dbg !44
  %2097 = extractelement <4 x half> %2094, i64 1, !dbg !44
  %2098 = extractelement <4 x half> %2094, i64 0, !dbg !44
  %2099 = shl nuw nsw i64 %15, 19, !dbg !45
  %address.thread561 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !45, !range !57, !frisk.register_lifetimes !29
  %address.local562 = zext nneg i32 %address.thread561 to i64, !dbg !37
  %address.local563 = add nsw i64 %16, %address.local562, !dbg !45
  %address.local564 = lshr i64 %address.local562, 6, !dbg !37
  %address.local565 = shl nuw nsw i64 %address.local564, 4, !dbg !45
  %address.local566 = add nsw i64 %address.local563, %address.local565, !dbg !45
  %address.local567 = and i64 %address.local562, 240, !dbg !45
  %address.local568 = sub nsw i64 %address.local566, %address.local567, !dbg !45
  %address.local569 = add nsw i64 %address.local568, %17, !dbg !45
  %address.local570 = shl nuw nsw i64 %address.local569, 7, !dbg !45
  %address.local571 = add nuw nsw i64 %2099, %address.local570, !dbg !45
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local573 = lshr i64 %address.local562, 2, !dbg !37
  %address.local574 = and i64 %address.local573, 12, !dbg !37
  %address.local575 = or disjoint i64 %address.local574, 96, !dbg !45
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local575, !dbg !45
  store half %2098, ptr addrspace(1) %address.local576, align 2, !dbg !45
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local578 = or disjoint i64 %address.local574, 97, !dbg !45
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local578, !dbg !45
  store half %2097, ptr addrspace(1) %address.local579, align 2, !dbg !45
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local581 = or disjoint i64 %address.local574, 98, !dbg !45
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local580, i64 %address.local581, !dbg !45
  store half %2096, ptr addrspace(1) %address.local582, align 2, !dbg !45
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local584 = or disjoint i64 %address.local574, 99, !dbg !45
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local584, !dbg !45
  store half %2095, ptr addrspace(1) %address.local585, align 2, !dbg !45
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local587 = or disjoint i64 %address.local574, 112, !dbg !45
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local586, i64 %address.local587, !dbg !45
  store half %2089, ptr addrspace(1) %address.local588, align 2, !dbg !45
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local590 = or disjoint i64 %address.local574, 113, !dbg !45
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local590, !dbg !45
  store half %2088, ptr addrspace(1) %address.local591, align 2, !dbg !45
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local593 = or disjoint i64 %address.local574, 114, !dbg !45
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local593, !dbg !45
  store half %2087, ptr addrspace(1) %address.local594, align 2, !dbg !45
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local596 = or i64 %address.local573, 115, !dbg !45
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local596, !dbg !45
  store half %2086, ptr addrspace(1) %address.local597, align 2, !dbg !45
  %address.local598 = shl nsw i64 %address.local569, 7, !dbg !45
  %address.local599 = add i64 %address.local598, 8192, !dbg !45
  %address.local600 = add nuw nsw i64 %2099, %address.local599, !dbg !45
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local575, !dbg !45
  store half %2080, ptr addrspace(1) %address.local602, align 2, !dbg !45
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local578, !dbg !45
  store half %2079, ptr addrspace(1) %address.local604, align 2, !dbg !45
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local581, !dbg !45
  store half %2078, ptr addrspace(1) %address.local606, align 2, !dbg !45
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local584, !dbg !45
  store half %2077, ptr addrspace(1) %address.local608, align 2, !dbg !45
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local587, !dbg !45
  store half %2071, ptr addrspace(1) %address.local610, align 2, !dbg !45
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local590, !dbg !45
  store half %2070, ptr addrspace(1) %address.local612, align 2, !dbg !45
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local593, !dbg !45
  store half %2069, ptr addrspace(1) %address.local614, align 2, !dbg !45
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local596, !dbg !45
  store half %2068, ptr addrspace(1) %address.local616, align 2, !dbg !45
  br label %2100, !dbg !45

2100:                                             ; preds = %2062, %2061
  ret void, !dbg !33
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

!llvm.dbg.cu = !{!31}
!llvm.module.flags = !{!56}

!28 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!29 = !{}
!30 = !DISubroutineType(types: !29)
!31 = distinct !DICompileUnit(language: DW_LANG_C, file: !28, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!32 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !28, file: !28, line: 22, type: !30, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !31)
!33 = !DILocation(line: 22, column: 3, scope: !32)
!34 = !DILocation(line: 69, column: 26, scope: !32)
!35 = !DILocation(line: 46, column: 17, scope: !32)
!36 = !DILocation(line: 66, column: 22, scope: !32)
!37 = !DILocation(line: 68, column: 24, scope: !32)
!38 = !DILocation(line: 73, column: 19, scope: !32)
!39 = !DILocation(line: 93, column: 23, scope: !32)
!40 = !DILocation(line: 56, column: 24, scope: !32)
!41 = !DILocation(line: 67, column: 22, scope: !32)
!42 = !DILocation(line: 92, column: 20, scope: !32)
!43 = !DILocation(line: 97, column: 23, scope: !32)
!44 = !DILocation(line: 98, column: 19, scope: !32)
!45 = !DILocation(line: 99, column: 9, scope: !32)
!46 = !DILocation(line: 59, column: 17, scope: !32)
!47 = !DILocation(line: 75, column: 24, scope: !32)
!48 = !DILocation(line: 76, column: 23, scope: !32)
!49 = !DILocation(line: 77, column: 26, scope: !32)
!50 = !DILocation(line: 85, column: 22, scope: !32)
!51 = !DILocation(line: 86, column: 20, scope: !32)
!52 = !DILocation(line: 87, column: 16, scope: !32)
!53 = !DILocation(line: 94, column: 20, scope: !32)
!54 = !DILocation(line: 70, column: 23, scope: !32)
!55 = !{i32 256, i32 1, i32 1}
!56 = !{i32 2, !"Debug Info Version", i32 3}
!57 = !{i32 0, i32 256}
