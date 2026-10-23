; ModuleID = 'temp.ll'
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

25:                                               ; preds = %741, %4
  %26 = phi i64 [ %1440, %741 ], [ 0, %4 ], !dbg !40
  %27 = phi [2 x <8 x float>] [ %1427, %741 ], [ zeroinitializer, %4 ], !dbg !40
  %28 = phi [2 x <1 x float>] [ %1049, %741 ], [ zeroinitializer, %4 ], !dbg !40
  %29 = phi [2 x <8 x float>] [ %1431, %741 ], [ zeroinitializer, %4 ], !dbg !40
  %30 = phi [2 x <8 x float>] [ %1435, %741 ], [ zeroinitializer, %4 ], !dbg !40
  %31 = phi [2 x <8 x float>] [ %1439, %741 ], [ zeroinitializer, %4 ], !dbg !40
  %.not = icmp sgt i64 %26, %6, !dbg !40
  br i1 %.not, label %1441, label %32, !dbg !40

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
  %119 = fmul <4 x half> %116, splat (half 1.275630e-01), !dbg !37
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !37
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 5120, !dbg !37
  %120 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !37
  %121 = fmul <4 x half> %117, splat (half 1.275630e-01), !dbg !37
  %122 = fmul <4 x half> %118, splat (half 1.275630e-01), !dbg !37
  %123 = fmul <4 x half> %120, splat (half 1.275630e-01), !dbg !37
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
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
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
  %127 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %124, <4 x float> %126) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !54
  %129 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %127, <4 x float> %128) #9, !dbg !34
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
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %146, <4 x float> %129) #9, !dbg !34
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local213, !dbg !34
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local216, !dbg !34
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local219, !dbg !34
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local223, !dbg !34
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 128, !dbg !34
  %149 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !34
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %147, <4 x float> %148) #9, !dbg !34
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local219, !dbg !34
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local223, !dbg !34
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local213, !dbg !34
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local216, !dbg !34
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 4224, !dbg !34
  %151 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !34
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %149, <4 x float> %131) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local213, !dbg !34
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) %address.local245, i64 %address.local216, !dbg !34
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local219, !dbg !34
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local223, !dbg !34
  %address.local249 = getelementptr i8, ptr addrspace(3) %address.local248, i64 192, !dbg !34
  %153 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %151, <4 x float> %152) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local210, !dbg !34
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local219, !dbg !34
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) %address.local251, i64 %address.local223, !dbg !34
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) %address.local252, i64 %address.local213, !dbg !34
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) %address.local253, i64 %address.local216, !dbg !34
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 4288, !dbg !34
  %155 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !34
  %156 = shufflevector <4 x float> %154, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %153, <4 x float> %132) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local256 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local194, !dbg !34
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local256, i64 1024, !dbg !34
  %158 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %155, <4 x float> %157) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local207, !dbg !34
  %address.local259 = getelementptr i8, ptr addrspace(3) %address.local258, i64 5120, !dbg !34
  %160 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !34
  %161 = shufflevector <4 x float> %159, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %158, <4 x float> %133) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local260 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local260, !dbg !34
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local213, !dbg !34
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) %address.local262, i64 %address.local216, !dbg !34
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local219, !dbg !34
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local223, !dbg !34
  %address.local266 = getelementptr i8, ptr addrspace(3) %address.local265, i64 64, !dbg !34
  %163 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %160, <4 x float> %162) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local260, !dbg !34
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local219, !dbg !34
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) %address.local268, i64 %address.local223, !dbg !34
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local213, !dbg !34
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local216, !dbg !34
  %address.local272 = getelementptr i8, ptr addrspace(3) %address.local271, i64 4160, !dbg !34
  %165 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !34
  %166 = shufflevector <4 x float> %164, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %163, <4 x float> %134) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local273 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local273, !dbg !34
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local213, !dbg !34
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local275, i64 %address.local216, !dbg !34
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local219, !dbg !34
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local223, !dbg !34
  %address.local279 = getelementptr i8, ptr addrspace(3) %address.local278, i64 128, !dbg !34
  %168 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %165, <4 x float> %167) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local273, !dbg !34
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local219, !dbg !34
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) %address.local281, i64 %address.local223, !dbg !34
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local213, !dbg !34
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local216, !dbg !34
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 4224, !dbg !34
  %170 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !34
  %171 = shufflevector <4 x float> %169, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %168, <4 x float> %135) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local286 = add nuw nsw i64 %address.local210, 512, !dbg !34
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local286, !dbg !34
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local213, !dbg !34
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local216, !dbg !34
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local219, !dbg !34
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local223, !dbg !34
  %address.local292 = getelementptr i8, ptr addrspace(3) %address.local291, i64 192, !dbg !34
  %173 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %170, <4 x float> %172) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %87, i64 %address.local286, !dbg !34
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local219, !dbg !34
  %address.local295 = getelementptr [2 x i8], ptr addrspace(3) %address.local294, i64 %address.local223, !dbg !34
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) %address.local295, i64 %address.local213, !dbg !34
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local216, !dbg !34
  %address.local298 = getelementptr i8, ptr addrspace(3) %address.local297, i64 4288, !dbg !34
  %175 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !34
  %176 = shufflevector <4 x float> %174, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %173, <4 x float> %136) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %178 = load <4 x half>, ptr addrspace(3) %address.local195, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %175, <4 x float> %177) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %180 = load <4 x half>, ptr addrspace(3) %address.local209, align 2, !dbg !34
  %181 = shufflevector <4 x float> %179, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %178, <4 x float> %138) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %183 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %180, <4 x float> %182) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %185 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !34
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %183, <4 x float> %139) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %187 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %185, <4 x float> %186) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %189 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !34
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %187, <4 x float> %140) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %191 = load <4 x half>, ptr addrspace(3) %address.local249, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %189, <4 x float> %190) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %193 = load <4 x half>, ptr addrspace(3) %address.local255, align 2, !dbg !34
  %194 = shufflevector <4 x float> %192, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %191, <4 x float> %141) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %196 = load <4 x half>, ptr addrspace(3) %address.local257, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %193, <4 x float> %195) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %198 = load <4 x half>, ptr addrspace(3) %address.local259, align 2, !dbg !34
  %199 = shufflevector <4 x float> %197, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %196, <4 x float> %142) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %201 = load <4 x half>, ptr addrspace(3) %address.local266, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %198, <4 x float> %200) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %203 = load <4 x half>, ptr addrspace(3) %address.local272, align 2, !dbg !34
  %204 = shufflevector <4 x float> %202, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %201, <4 x float> %143) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %206 = load <4 x half>, ptr addrspace(3) %address.local279, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %203, <4 x float> %205) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %208 = load <4 x half>, ptr addrspace(3) %address.local285, align 2, !dbg !34
  %209 = shufflevector <4 x float> %207, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %206, <4 x float> %144) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %211 = load <4 x half>, ptr addrspace(3) %address.local292, align 2, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %208, <4 x float> %210) #9, !dbg !34
  call void @llvm.amdgcn.sched.barrier(i32 0), !dbg !34
  %213 = load <4 x half>, ptr addrspace(3) %address.local298, align 2, !dbg !34
  %214 = shufflevector <4 x float> %212, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %211, <4 x float> %145) #9, !dbg !34
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %213, <4 x float> %215) #9, !dbg !34
  %217 = shufflevector <4 x float> %216, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !54
  %218 = shufflevector <4 x float> %130, <4 x float> %150, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %219 = shufflevector <32 x float> %218, <32 x float> %156, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %220 = shufflevector <32 x float> %219, <32 x float> %161, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %221 = shufflevector <32 x float> %220, <32 x float> %166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %222 = shufflevector <32 x float> %221, <32 x float> %171, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %223 = shufflevector <32 x float> %222, <32 x float> %176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
  %224 = shufflevector <32 x float> %223, <32 x float> %181, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !46
  %225 = insertvalue [2 x <32 x float>] poison, <32 x float> %224, 0, !dbg !46
  %226 = shufflevector <4 x float> %184, <4 x float> %188, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !46
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
  %276 = fadd float %275, 0.000000e+00, !dbg !38
  %277 = extractelement <32 x float> %274, i64 1, !dbg !38
  %278 = fadd float %277, 0.000000e+00, !dbg !38
  %279 = extractelement <32 x float> %274, i64 2, !dbg !38
  %280 = fadd float %279, 0.000000e+00, !dbg !38
  %281 = extractelement <32 x float> %274, i64 3, !dbg !38
  %282 = fadd float %281, 0.000000e+00, !dbg !38
  %283 = extractelement <32 x float> %274, i64 4, !dbg !38
  %284 = fadd float %283, 0.000000e+00, !dbg !38
  %285 = extractelement <32 x float> %274, i64 5, !dbg !38
  %286 = fadd float %285, 0.000000e+00, !dbg !38
  %287 = extractelement <32 x float> %274, i64 6, !dbg !38
  %288 = fadd float %287, 0.000000e+00, !dbg !38
  %289 = extractelement <32 x float> %274, i64 7, !dbg !38
  %290 = fadd float %289, 0.000000e+00, !dbg !38
  %291 = extractelement <32 x float> %274, i64 8, !dbg !38
  %292 = fadd float %291, 0.000000e+00, !dbg !38
  %293 = extractelement <32 x float> %274, i64 9, !dbg !38
  %294 = fadd float %293, 0.000000e+00, !dbg !38
  %295 = extractelement <32 x float> %274, i64 10, !dbg !38
  %296 = fadd float %295, 0.000000e+00, !dbg !38
  %297 = extractelement <32 x float> %274, i64 11, !dbg !38
  %298 = fadd float %297, 0.000000e+00, !dbg !38
  %299 = extractelement <32 x float> %274, i64 12, !dbg !38
  %300 = fadd float %299, 0.000000e+00, !dbg !38
  %301 = extractelement <32 x float> %274, i64 13, !dbg !38
  %302 = fadd float %301, 0.000000e+00, !dbg !38
  %303 = extractelement <32 x float> %274, i64 14, !dbg !38
  %304 = fadd float %303, 0.000000e+00, !dbg !38
  %305 = extractelement <32 x float> %274, i64 15, !dbg !38
  %306 = fadd float %305, 0.000000e+00, !dbg !38
  %307 = extractelement <32 x float> %274, i64 16, !dbg !38
  %308 = fadd float %307, 0.000000e+00, !dbg !38
  %309 = extractelement <32 x float> %274, i64 17, !dbg !38
  %310 = fadd float %309, 0.000000e+00, !dbg !38
  %311 = extractelement <32 x float> %274, i64 18, !dbg !38
  %312 = fadd float %311, 0.000000e+00, !dbg !38
  %313 = extractelement <32 x float> %274, i64 19, !dbg !38
  %314 = fadd float %313, 0.000000e+00, !dbg !38
  %315 = extractelement <32 x float> %274, i64 20, !dbg !38
  %316 = fadd float %315, 0.000000e+00, !dbg !38
  %317 = extractelement <32 x float> %274, i64 21, !dbg !38
  %318 = fadd float %317, 0.000000e+00, !dbg !38
  %319 = extractelement <32 x float> %274, i64 22, !dbg !38
  %320 = fadd float %319, 0.000000e+00, !dbg !38
  %321 = extractelement <32 x float> %274, i64 23, !dbg !38
  %322 = fadd float %321, 0.000000e+00, !dbg !38
  %323 = extractelement <32 x float> %274, i64 24, !dbg !38
  %324 = fadd float %323, 0.000000e+00, !dbg !38
  %325 = extractelement <32 x float> %274, i64 25, !dbg !38
  %326 = fadd float %325, 0.000000e+00, !dbg !38
  %327 = extractelement <32 x float> %274, i64 26, !dbg !38
  %328 = fadd float %327, 0.000000e+00, !dbg !38
  %329 = extractelement <32 x float> %274, i64 27, !dbg !38
  %330 = fadd float %329, 0.000000e+00, !dbg !38
  %331 = extractelement <32 x float> %274, i64 28, !dbg !38
  %332 = fadd float %331, 0.000000e+00, !dbg !38
  %333 = extractelement <32 x float> %274, i64 29, !dbg !38
  %334 = fadd float %333, 0.000000e+00, !dbg !38
  %335 = extractelement <32 x float> %274, i64 30, !dbg !38
  %336 = fadd float %335, 0.000000e+00, !dbg !38
  %337 = extractelement <32 x float> %274, i64 31, !dbg !38
  %338 = fadd float %337, 0.000000e+00, !dbg !38
  %339 = extractvalue [2 x <32 x float>] %39, 1, !dbg !38
  %340 = extractelement <32 x float> %339, i64 0, !dbg !38
  %341 = fadd float %340, 0.000000e+00, !dbg !38
  %342 = extractelement <32 x float> %339, i64 1, !dbg !38
  %343 = fadd float %342, 0.000000e+00, !dbg !38
  %344 = extractelement <32 x float> %339, i64 2, !dbg !38
  %345 = fadd float %344, 0.000000e+00, !dbg !38
  %346 = extractelement <32 x float> %339, i64 3, !dbg !38
  %347 = fadd float %346, 0.000000e+00, !dbg !38
  %348 = extractelement <32 x float> %339, i64 4, !dbg !38
  %349 = fadd float %348, 0.000000e+00, !dbg !38
  %350 = extractelement <32 x float> %339, i64 5, !dbg !38
  %351 = fadd float %350, 0.000000e+00, !dbg !38
  %352 = extractelement <32 x float> %339, i64 6, !dbg !38
  %353 = fadd float %352, 0.000000e+00, !dbg !38
  %354 = extractelement <32 x float> %339, i64 7, !dbg !38
  %355 = fadd float %354, 0.000000e+00, !dbg !38
  %356 = extractelement <32 x float> %339, i64 8, !dbg !38
  %357 = fadd float %356, 0.000000e+00, !dbg !38
  %358 = extractelement <32 x float> %339, i64 9, !dbg !38
  %359 = fadd float %358, 0.000000e+00, !dbg !38
  %360 = extractelement <32 x float> %339, i64 10, !dbg !38
  %361 = fadd float %360, 0.000000e+00, !dbg !38
  %362 = extractelement <32 x float> %339, i64 11, !dbg !38
  %363 = fadd float %362, 0.000000e+00, !dbg !38
  %364 = extractelement <32 x float> %339, i64 12, !dbg !38
  %365 = fadd float %364, 0.000000e+00, !dbg !38
  %366 = extractelement <32 x float> %339, i64 13, !dbg !38
  %367 = fadd float %366, 0.000000e+00, !dbg !38
  %368 = extractelement <32 x float> %339, i64 14, !dbg !38
  %369 = fadd float %368, 0.000000e+00, !dbg !38
  %370 = extractelement <32 x float> %339, i64 15, !dbg !38
  %371 = fadd float %370, 0.000000e+00, !dbg !38
  %372 = extractelement <32 x float> %339, i64 16, !dbg !38
  %373 = fadd float %372, 0.000000e+00, !dbg !38
  %374 = extractelement <32 x float> %339, i64 17, !dbg !38
  %375 = fadd float %374, 0.000000e+00, !dbg !38
  %376 = extractelement <32 x float> %339, i64 18, !dbg !38
  %377 = fadd float %376, 0.000000e+00, !dbg !38
  %378 = extractelement <32 x float> %339, i64 19, !dbg !38
  %379 = fadd float %378, 0.000000e+00, !dbg !38
  %380 = extractelement <32 x float> %339, i64 20, !dbg !38
  %381 = fadd float %380, 0.000000e+00, !dbg !38
  %382 = extractelement <32 x float> %339, i64 21, !dbg !38
  %383 = fadd float %382, 0.000000e+00, !dbg !38
  %384 = extractelement <32 x float> %339, i64 22, !dbg !38
  %385 = fadd float %384, 0.000000e+00, !dbg !38
  %386 = extractelement <32 x float> %339, i64 23, !dbg !38
  %387 = fadd float %386, 0.000000e+00, !dbg !38
  %388 = extractelement <32 x float> %339, i64 24, !dbg !38
  %389 = fadd float %388, 0.000000e+00, !dbg !38
  %390 = extractelement <32 x float> %339, i64 25, !dbg !38
  %391 = fadd float %390, 0.000000e+00, !dbg !38
  %392 = extractelement <32 x float> %339, i64 26, !dbg !38
  %393 = fadd float %392, 0.000000e+00, !dbg !38
  %394 = extractelement <32 x float> %339, i64 27, !dbg !38
  %395 = fadd float %394, 0.000000e+00, !dbg !38
  %396 = extractelement <32 x float> %339, i64 28, !dbg !38
  %397 = fadd float %396, 0.000000e+00, !dbg !38
  %398 = extractelement <32 x float> %339, i64 29, !dbg !38
  %399 = fadd float %398, 0.000000e+00, !dbg !38
  %400 = extractelement <32 x float> %339, i64 30, !dbg !38
  %401 = fadd float %400, 0.000000e+00, !dbg !38
  %402 = extractelement <32 x float> %339, i64 31, !dbg !38
  %403 = fadd float %402, 0.000000e+00, !dbg !38
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
  br label %740, !dbg !38

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
  %477 = lshr i64 %476, 32, !dbg !48
  %compare.high = trunc i64 %477 to i32, !dbg !48
  %478 = lshr i64 %475, 32, !dbg !48
  %compare.high1 = trunc i64 %478 to i32, !dbg !48
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high1, !dbg !48
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high1, !dbg !48
  %compare.low.base = trunc i64 %476 to i32, !dbg !48
  %compare.low.base2 = trunc i64 %475 to i32, !dbg !48
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base2, !dbg !48
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !48
  %479 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !49
  %480 = extractvalue [2 x <32 x float>] %39, 0, !dbg !38
  %481 = extractelement <32 x float> %480, i64 0, !dbg !38
  %482 = fadd float %481, %479, !dbg !38
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 1) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local = or i32 %compare.local.base, 1, !dbg !48
  %compare.low3 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !48
  %compare.result4 = select i1 %compare.high.equal, i1 %compare.low3, i1 %compare.high.order, !dbg !48
  %483 = select i1 %compare.result4, float 0.000000e+00, float -inf, !dbg !49
  %484 = extractelement <32 x float> %480, i64 1, !dbg !38
  %485 = fadd float %484, %483, !dbg !38
  %compare.local.base5 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 2) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local6 = or i32 %compare.local.base5, 2, !dbg !48
  %compare.low7 = icmp ugt i32 %compare.low.base, %compare.local6, !dbg !48
  %compare.result8 = select i1 %compare.high.equal, i1 %compare.low7, i1 %compare.high.order, !dbg !48
  %486 = select i1 %compare.result8, float 0.000000e+00, float -inf, !dbg !49
  %487 = extractelement <32 x float> %480, i64 2, !dbg !38
  %488 = fadd float %487, %486, !dbg !38
  %compare.local.base9 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 3) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local10 = or i32 %compare.local.base9, 3, !dbg !48
  %compare.low11 = icmp ugt i32 %compare.low.base, %compare.local10, !dbg !48
  %compare.result12 = select i1 %compare.high.equal, i1 %compare.low11, i1 %compare.high.order, !dbg !48
  %489 = select i1 %compare.result12, float 0.000000e+00, float -inf, !dbg !49
  %490 = extractelement <32 x float> %480, i64 3, !dbg !38
  %491 = fadd float %490, %489, !dbg !38
  %compare.local.base13 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 4) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local14 = or i32 %compare.local.base13, 4, !dbg !48
  %compare.low15 = icmp ugt i32 %compare.low.base, %compare.local14, !dbg !48
  %compare.result16 = select i1 %compare.high.equal, i1 %compare.low15, i1 %compare.high.order, !dbg !48
  %492 = select i1 %compare.result16, float 0.000000e+00, float -inf, !dbg !49
  %493 = extractelement <32 x float> %480, i64 4, !dbg !38
  %494 = fadd float %493, %492, !dbg !38
  %compare.local.base17 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 5) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local18 = or i32 %compare.local.base17, 5, !dbg !48
  %compare.low19 = icmp ugt i32 %compare.low.base, %compare.local18, !dbg !48
  %compare.result20 = select i1 %compare.high.equal, i1 %compare.low19, i1 %compare.high.order, !dbg !48
  %495 = select i1 %compare.result20, float 0.000000e+00, float -inf, !dbg !49
  %496 = extractelement <32 x float> %480, i64 5, !dbg !38
  %497 = fadd float %496, %495, !dbg !38
  %compare.local.base21 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 6) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local22 = or i32 %compare.local.base21, 6, !dbg !48
  %compare.low23 = icmp ugt i32 %compare.low.base, %compare.local22, !dbg !48
  %compare.result24 = select i1 %compare.high.equal, i1 %compare.low23, i1 %compare.high.order, !dbg !48
  %498 = select i1 %compare.result24, float 0.000000e+00, float -inf, !dbg !49
  %499 = extractelement <32 x float> %480, i64 6, !dbg !38
  %500 = fadd float %499, %498, !dbg !38
  %compare.local.base25 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 7) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local26 = or i32 %compare.local.base25, 7, !dbg !48
  %compare.low27 = icmp ugt i32 %compare.low.base, %compare.local26, !dbg !48
  %compare.result28 = select i1 %compare.high.equal, i1 %compare.low27, i1 %compare.high.order, !dbg !48
  %501 = select i1 %compare.result28, float 0.000000e+00, float -inf, !dbg !49
  %502 = extractelement <32 x float> %480, i64 7, !dbg !38
  %503 = fadd float %502, %501, !dbg !38
  %compare.local.base29 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 8) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local30 = or i32 %compare.local.base29, 8, !dbg !48
  %compare.low31 = icmp ugt i32 %compare.low.base, %compare.local30, !dbg !48
  %compare.result32 = select i1 %compare.high.equal, i1 %compare.low31, i1 %compare.high.order, !dbg !48
  %504 = select i1 %compare.result32, float 0.000000e+00, float -inf, !dbg !49
  %505 = extractelement <32 x float> %480, i64 8, !dbg !38
  %506 = fadd float %505, %504, !dbg !38
  %compare.local.base33 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 9) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local34 = or i32 %compare.local.base33, 9, !dbg !48
  %compare.low35 = icmp ugt i32 %compare.low.base, %compare.local34, !dbg !48
  %compare.result36 = select i1 %compare.high.equal, i1 %compare.low35, i1 %compare.high.order, !dbg !48
  %507 = select i1 %compare.result36, float 0.000000e+00, float -inf, !dbg !49
  %508 = extractelement <32 x float> %480, i64 9, !dbg !38
  %509 = fadd float %508, %507, !dbg !38
  %compare.local.base37 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 10) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local38 = or i32 %compare.local.base37, 10, !dbg !48
  %compare.low39 = icmp ugt i32 %compare.low.base, %compare.local38, !dbg !48
  %compare.result40 = select i1 %compare.high.equal, i1 %compare.low39, i1 %compare.high.order, !dbg !48
  %510 = select i1 %compare.result40, float 0.000000e+00, float -inf, !dbg !49
  %511 = extractelement <32 x float> %480, i64 10, !dbg !38
  %512 = fadd float %511, %510, !dbg !38
  %compare.local.base41 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 11) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local42 = or i32 %compare.local.base41, 11, !dbg !48
  %compare.low43 = icmp ugt i32 %compare.low.base, %compare.local42, !dbg !48
  %compare.result44 = select i1 %compare.high.equal, i1 %compare.low43, i1 %compare.high.order, !dbg !48
  %513 = select i1 %compare.result44, float 0.000000e+00, float -inf, !dbg !49
  %514 = extractelement <32 x float> %480, i64 11, !dbg !38
  %515 = fadd float %514, %513, !dbg !38
  %compare.local.base45 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 12) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local46 = or i32 %compare.local.base45, 12, !dbg !48
  %compare.low47 = icmp ugt i32 %compare.low.base, %compare.local46, !dbg !48
  %compare.result48 = select i1 %compare.high.equal, i1 %compare.low47, i1 %compare.high.order, !dbg !48
  %516 = select i1 %compare.result48, float 0.000000e+00, float -inf, !dbg !49
  %517 = extractelement <32 x float> %480, i64 12, !dbg !38
  %518 = fadd float %517, %516, !dbg !38
  %compare.local.base49 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 13) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local50 = or i32 %compare.local.base49, 13, !dbg !48
  %compare.low51 = icmp ugt i32 %compare.low.base, %compare.local50, !dbg !48
  %compare.result52 = select i1 %compare.high.equal, i1 %compare.low51, i1 %compare.high.order, !dbg !48
  %519 = select i1 %compare.result52, float 0.000000e+00, float -inf, !dbg !49
  %520 = extractelement <32 x float> %480, i64 13, !dbg !38
  %521 = fadd float %520, %519, !dbg !38
  %compare.local.base53 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 14) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local54 = or i32 %compare.local.base53, 14, !dbg !48
  %compare.low55 = icmp ugt i32 %compare.low.base, %compare.local54, !dbg !48
  %compare.result56 = select i1 %compare.high.equal, i1 %compare.low55, i1 %compare.high.order, !dbg !48
  %522 = select i1 %compare.result56, float 0.000000e+00, float -inf, !dbg !49
  %523 = extractelement <32 x float> %480, i64 14, !dbg !38
  %524 = fadd float %523, %522, !dbg !38
  %compare.local.base57 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 15) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local58 = or i32 %compare.local.base57, 15, !dbg !48
  %compare.low59 = icmp ugt i32 %compare.low.base, %compare.local58, !dbg !48
  %compare.result60 = select i1 %compare.high.equal, i1 %compare.low59, i1 %compare.high.order, !dbg !48
  %525 = select i1 %compare.result60, float 0.000000e+00, float -inf, !dbg !49
  %526 = extractelement <32 x float> %480, i64 15, !dbg !38
  %527 = fadd float %526, %525, !dbg !38
  %compare.local.base61 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 16) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local62 = or i32 %compare.local.base61, 64, !dbg !48
  %compare.low63 = icmp ugt i32 %compare.low.base, %compare.local62, !dbg !48
  %compare.result64 = select i1 %compare.high.equal, i1 %compare.low63, i1 %compare.high.order, !dbg !48
  %528 = select i1 %compare.result64, float 0.000000e+00, float -inf, !dbg !49
  %529 = extractelement <32 x float> %480, i64 16, !dbg !38
  %530 = fadd float %529, %528, !dbg !38
  %compare.local.base65 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 17) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local66 = or i32 %compare.local.base65, 65, !dbg !48
  %compare.low67 = icmp ugt i32 %compare.low.base, %compare.local66, !dbg !48
  %compare.result68 = select i1 %compare.high.equal, i1 %compare.low67, i1 %compare.high.order, !dbg !48
  %531 = select i1 %compare.result68, float 0.000000e+00, float -inf, !dbg !49
  %532 = extractelement <32 x float> %480, i64 17, !dbg !38
  %533 = fadd float %532, %531, !dbg !38
  %compare.local.base69 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 18) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local70 = or i32 %compare.local.base69, 66, !dbg !48
  %compare.low71 = icmp ugt i32 %compare.low.base, %compare.local70, !dbg !48
  %compare.result72 = select i1 %compare.high.equal, i1 %compare.low71, i1 %compare.high.order, !dbg !48
  %534 = select i1 %compare.result72, float 0.000000e+00, float -inf, !dbg !49
  %535 = extractelement <32 x float> %480, i64 18, !dbg !38
  %536 = fadd float %535, %534, !dbg !38
  %compare.local.base73 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 19) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local74 = or i32 %compare.local.base73, 67, !dbg !48
  %compare.low75 = icmp ugt i32 %compare.low.base, %compare.local74, !dbg !48
  %compare.result76 = select i1 %compare.high.equal, i1 %compare.low75, i1 %compare.high.order, !dbg !48
  %537 = select i1 %compare.result76, float 0.000000e+00, float -inf, !dbg !49
  %538 = extractelement <32 x float> %480, i64 19, !dbg !38
  %539 = fadd float %538, %537, !dbg !38
  %compare.local.base77 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 20) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local78 = or i32 %compare.local.base77, 68, !dbg !48
  %compare.low79 = icmp ugt i32 %compare.low.base, %compare.local78, !dbg !48
  %compare.result80 = select i1 %compare.high.equal, i1 %compare.low79, i1 %compare.high.order, !dbg !48
  %540 = select i1 %compare.result80, float 0.000000e+00, float -inf, !dbg !49
  %541 = extractelement <32 x float> %480, i64 20, !dbg !38
  %542 = fadd float %541, %540, !dbg !38
  %compare.local.base81 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 21) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local82 = or i32 %compare.local.base81, 69, !dbg !48
  %compare.low83 = icmp ugt i32 %compare.low.base, %compare.local82, !dbg !48
  %compare.result84 = select i1 %compare.high.equal, i1 %compare.low83, i1 %compare.high.order, !dbg !48
  %543 = select i1 %compare.result84, float 0.000000e+00, float -inf, !dbg !49
  %544 = extractelement <32 x float> %480, i64 21, !dbg !38
  %545 = fadd float %544, %543, !dbg !38
  %compare.local.base85 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 22) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local86 = or i32 %compare.local.base85, 70, !dbg !48
  %compare.low87 = icmp ugt i32 %compare.low.base, %compare.local86, !dbg !48
  %compare.result88 = select i1 %compare.high.equal, i1 %compare.low87, i1 %compare.high.order, !dbg !48
  %546 = select i1 %compare.result88, float 0.000000e+00, float -inf, !dbg !49
  %547 = extractelement <32 x float> %480, i64 22, !dbg !38
  %548 = fadd float %547, %546, !dbg !38
  %compare.local.base89 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 23) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local90 = or i32 %compare.local.base89, 71, !dbg !48
  %compare.low91 = icmp ugt i32 %compare.low.base, %compare.local90, !dbg !48
  %compare.result92 = select i1 %compare.high.equal, i1 %compare.low91, i1 %compare.high.order, !dbg !48
  %549 = select i1 %compare.result92, float 0.000000e+00, float -inf, !dbg !49
  %550 = extractelement <32 x float> %480, i64 23, !dbg !38
  %551 = fadd float %550, %549, !dbg !38
  %compare.local.base93 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 24) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local94 = or i32 %compare.local.base93, 72, !dbg !48
  %compare.low95 = icmp ugt i32 %compare.low.base, %compare.local94, !dbg !48
  %compare.result96 = select i1 %compare.high.equal, i1 %compare.low95, i1 %compare.high.order, !dbg !48
  %552 = select i1 %compare.result96, float 0.000000e+00, float -inf, !dbg !49
  %553 = extractelement <32 x float> %480, i64 24, !dbg !38
  %554 = fadd float %553, %552, !dbg !38
  %compare.local.base97 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 25) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local98 = or i32 %compare.local.base97, 73, !dbg !48
  %compare.low99 = icmp ugt i32 %compare.low.base, %compare.local98, !dbg !48
  %compare.result100 = select i1 %compare.high.equal, i1 %compare.low99, i1 %compare.high.order, !dbg !48
  %555 = select i1 %compare.result100, float 0.000000e+00, float -inf, !dbg !49
  %556 = extractelement <32 x float> %480, i64 25, !dbg !38
  %557 = fadd float %556, %555, !dbg !38
  %compare.local.base101 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 26) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local102 = or i32 %compare.local.base101, 74, !dbg !48
  %compare.low103 = icmp ugt i32 %compare.low.base, %compare.local102, !dbg !48
  %compare.result104 = select i1 %compare.high.equal, i1 %compare.low103, i1 %compare.high.order, !dbg !48
  %558 = select i1 %compare.result104, float 0.000000e+00, float -inf, !dbg !49
  %559 = extractelement <32 x float> %480, i64 26, !dbg !38
  %560 = fadd float %559, %558, !dbg !38
  %compare.local.base105 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 27) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local106 = or i32 %compare.local.base105, 75, !dbg !48
  %compare.low107 = icmp ugt i32 %compare.low.base, %compare.local106, !dbg !48
  %compare.result108 = select i1 %compare.high.equal, i1 %compare.low107, i1 %compare.high.order, !dbg !48
  %561 = select i1 %compare.result108, float 0.000000e+00, float -inf, !dbg !49
  %562 = extractelement <32 x float> %480, i64 27, !dbg !38
  %563 = fadd float %562, %561, !dbg !38
  %compare.local.base109 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 28) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local110 = or i32 %compare.local.base109, 76, !dbg !48
  %compare.low111 = icmp ugt i32 %compare.low.base, %compare.local110, !dbg !48
  %compare.result112 = select i1 %compare.high.equal, i1 %compare.low111, i1 %compare.high.order, !dbg !48
  %564 = select i1 %compare.result112, float 0.000000e+00, float -inf, !dbg !49
  %565 = extractelement <32 x float> %480, i64 28, !dbg !38
  %566 = fadd float %565, %564, !dbg !38
  %compare.local.base113 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 29) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local114 = or i32 %compare.local.base113, 77, !dbg !48
  %compare.low115 = icmp ugt i32 %compare.low.base, %compare.local114, !dbg !48
  %compare.result116 = select i1 %compare.high.equal, i1 %compare.low115, i1 %compare.high.order, !dbg !48
  %567 = select i1 %compare.result116, float 0.000000e+00, float -inf, !dbg !49
  %568 = extractelement <32 x float> %480, i64 29, !dbg !38
  %569 = fadd float %568, %567, !dbg !38
  %compare.local.base117 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 30) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local118 = or i32 %compare.local.base117, 78, !dbg !48
  %compare.low119 = icmp ugt i32 %compare.low.base, %compare.local118, !dbg !48
  %compare.result120 = select i1 %compare.high.equal, i1 %compare.low119, i1 %compare.high.order, !dbg !48
  %570 = select i1 %compare.result120, float 0.000000e+00, float -inf, !dbg !49
  %571 = extractelement <32 x float> %480, i64 30, !dbg !38
  %572 = fadd float %571, %570, !dbg !38
  %compare.local.base121 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base2, i32 31) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local122 = or i32 %compare.local.base121, 79, !dbg !48
  %compare.low123 = icmp ugt i32 %compare.low.base, %compare.local122, !dbg !48
  %compare.result124 = select i1 %compare.high.equal, i1 %compare.low123, i1 %compare.high.order, !dbg !48
  %573 = select i1 %compare.result124, float 0.000000e+00, float -inf, !dbg !49
  %574 = extractelement <32 x float> %480, i64 31, !dbg !38
  %575 = fadd float %574, %573, !dbg !38
  %576 = add nsw i64 %473, 65, !dbg !47
  %577 = lshr i64 %576, 32, !dbg !48
  %compare.high125 = trunc i64 %577 to i32, !dbg !48
  %578 = lshr i64 %475, 32, !dbg !48
  %compare.high126 = trunc i64 %578 to i32, !dbg !48
  %compare.high.equal127 = icmp eq i32 %compare.high125, %compare.high126, !dbg !48
  %compare.high.order128 = icmp ugt i32 %compare.high125, %compare.high126, !dbg !48
  %compare.low.base129 = trunc i64 %576 to i32, !dbg !48
  %compare.low.base130 = trunc i64 %475 to i32, !dbg !48
  %compare.low131 = icmp ugt i32 %compare.low.base129, %compare.low.base130, !dbg !48
  %compare.result132 = select i1 %compare.high.equal127, i1 %compare.low131, i1 %compare.high.order128, !dbg !48
  %579 = select i1 %compare.result132, float 0.000000e+00, float -inf, !dbg !49
  %580 = extractvalue [2 x <32 x float>] %39, 1, !dbg !38
  %581 = extractelement <32 x float> %580, i64 0, !dbg !38
  %582 = fadd float %581, %579, !dbg !38
  %compare.local.base133 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 32) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local134 = or i32 %compare.local.base133, 1, !dbg !48
  %compare.low135 = icmp ugt i32 %compare.low.base129, %compare.local134, !dbg !48
  %compare.result136 = select i1 %compare.high.equal127, i1 %compare.low135, i1 %compare.high.order128, !dbg !48
  %583 = select i1 %compare.result136, float 0.000000e+00, float -inf, !dbg !49
  %584 = extractelement <32 x float> %580, i64 1, !dbg !38
  %585 = fadd float %584, %583, !dbg !38
  %compare.local.base137 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 33) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local138 = or i32 %compare.local.base137, 2, !dbg !48
  %compare.low139 = icmp ugt i32 %compare.low.base129, %compare.local138, !dbg !48
  %compare.result140 = select i1 %compare.high.equal127, i1 %compare.low139, i1 %compare.high.order128, !dbg !48
  %586 = select i1 %compare.result140, float 0.000000e+00, float -inf, !dbg !49
  %587 = extractelement <32 x float> %580, i64 2, !dbg !38
  %588 = fadd float %587, %586, !dbg !38
  %compare.local.base141 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 34) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local142 = or i32 %compare.local.base141, 3, !dbg !48
  %compare.low143 = icmp ugt i32 %compare.low.base129, %compare.local142, !dbg !48
  %compare.result144 = select i1 %compare.high.equal127, i1 %compare.low143, i1 %compare.high.order128, !dbg !48
  %589 = select i1 %compare.result144, float 0.000000e+00, float -inf, !dbg !49
  %590 = extractelement <32 x float> %580, i64 3, !dbg !38
  %591 = fadd float %590, %589, !dbg !38
  %compare.local.base145 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 35) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local146 = or i32 %compare.local.base145, 4, !dbg !48
  %compare.low147 = icmp ugt i32 %compare.low.base129, %compare.local146, !dbg !48
  %compare.result148 = select i1 %compare.high.equal127, i1 %compare.low147, i1 %compare.high.order128, !dbg !48
  %592 = select i1 %compare.result148, float 0.000000e+00, float -inf, !dbg !49
  %593 = extractelement <32 x float> %580, i64 4, !dbg !38
  %594 = fadd float %593, %592, !dbg !38
  %compare.local.base149 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 36) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local150 = or i32 %compare.local.base149, 5, !dbg !48
  %compare.low151 = icmp ugt i32 %compare.low.base129, %compare.local150, !dbg !48
  %compare.result152 = select i1 %compare.high.equal127, i1 %compare.low151, i1 %compare.high.order128, !dbg !48
  %595 = select i1 %compare.result152, float 0.000000e+00, float -inf, !dbg !49
  %596 = extractelement <32 x float> %580, i64 5, !dbg !38
  %597 = fadd float %596, %595, !dbg !38
  %compare.local.base153 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 37) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local154 = or i32 %compare.local.base153, 6, !dbg !48
  %compare.low155 = icmp ugt i32 %compare.low.base129, %compare.local154, !dbg !48
  %compare.result156 = select i1 %compare.high.equal127, i1 %compare.low155, i1 %compare.high.order128, !dbg !48
  %598 = select i1 %compare.result156, float 0.000000e+00, float -inf, !dbg !49
  %599 = extractelement <32 x float> %580, i64 6, !dbg !38
  %600 = fadd float %599, %598, !dbg !38
  %compare.local.base157 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 38) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local158 = or i32 %compare.local.base157, 7, !dbg !48
  %compare.low159 = icmp ugt i32 %compare.low.base129, %compare.local158, !dbg !48
  %compare.result160 = select i1 %compare.high.equal127, i1 %compare.low159, i1 %compare.high.order128, !dbg !48
  %601 = select i1 %compare.result160, float 0.000000e+00, float -inf, !dbg !49
  %602 = extractelement <32 x float> %580, i64 7, !dbg !38
  %603 = fadd float %602, %601, !dbg !38
  %compare.local.base161 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 39) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local162 = or i32 %compare.local.base161, 8, !dbg !48
  %compare.low163 = icmp ugt i32 %compare.low.base129, %compare.local162, !dbg !48
  %compare.result164 = select i1 %compare.high.equal127, i1 %compare.low163, i1 %compare.high.order128, !dbg !48
  %604 = select i1 %compare.result164, float 0.000000e+00, float -inf, !dbg !49
  %605 = extractelement <32 x float> %580, i64 8, !dbg !38
  %606 = fadd float %605, %604, !dbg !38
  %compare.local.base165 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 40) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local166 = or i32 %compare.local.base165, 9, !dbg !48
  %compare.low167 = icmp ugt i32 %compare.low.base129, %compare.local166, !dbg !48
  %compare.result168 = select i1 %compare.high.equal127, i1 %compare.low167, i1 %compare.high.order128, !dbg !48
  %607 = select i1 %compare.result168, float 0.000000e+00, float -inf, !dbg !49
  %608 = extractelement <32 x float> %580, i64 9, !dbg !38
  %609 = fadd float %608, %607, !dbg !38
  %compare.local.base169 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 41) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local170 = or i32 %compare.local.base169, 10, !dbg !48
  %compare.low171 = icmp ugt i32 %compare.low.base129, %compare.local170, !dbg !48
  %compare.result172 = select i1 %compare.high.equal127, i1 %compare.low171, i1 %compare.high.order128, !dbg !48
  %610 = select i1 %compare.result172, float 0.000000e+00, float -inf, !dbg !49
  %611 = extractelement <32 x float> %580, i64 10, !dbg !38
  %612 = fadd float %611, %610, !dbg !38
  %compare.local.base173 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 42) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local174 = or i32 %compare.local.base173, 11, !dbg !48
  %compare.low175 = icmp ugt i32 %compare.low.base129, %compare.local174, !dbg !48
  %compare.result176 = select i1 %compare.high.equal127, i1 %compare.low175, i1 %compare.high.order128, !dbg !48
  %613 = select i1 %compare.result176, float 0.000000e+00, float -inf, !dbg !49
  %614 = extractelement <32 x float> %580, i64 11, !dbg !38
  %615 = fadd float %614, %613, !dbg !38
  %compare.local.base177 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 43) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local178 = or i32 %compare.local.base177, 12, !dbg !48
  %compare.low179 = icmp ugt i32 %compare.low.base129, %compare.local178, !dbg !48
  %compare.result180 = select i1 %compare.high.equal127, i1 %compare.low179, i1 %compare.high.order128, !dbg !48
  %616 = select i1 %compare.result180, float 0.000000e+00, float -inf, !dbg !49
  %617 = extractelement <32 x float> %580, i64 12, !dbg !38
  %618 = fadd float %617, %616, !dbg !38
  %compare.local.base181 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 44) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local182 = or i32 %compare.local.base181, 13, !dbg !48
  %compare.low183 = icmp ugt i32 %compare.low.base129, %compare.local182, !dbg !48
  %compare.result184 = select i1 %compare.high.equal127, i1 %compare.low183, i1 %compare.high.order128, !dbg !48
  %619 = select i1 %compare.result184, float 0.000000e+00, float -inf, !dbg !49
  %620 = extractelement <32 x float> %580, i64 13, !dbg !38
  %621 = fadd float %620, %619, !dbg !38
  %compare.local.base185 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 45) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local186 = or i32 %compare.local.base185, 14, !dbg !48
  %compare.low187 = icmp ugt i32 %compare.low.base129, %compare.local186, !dbg !48
  %compare.result188 = select i1 %compare.high.equal127, i1 %compare.low187, i1 %compare.high.order128, !dbg !48
  %622 = select i1 %compare.result188, float 0.000000e+00, float -inf, !dbg !49
  %623 = extractelement <32 x float> %580, i64 14, !dbg !38
  %624 = fadd float %623, %622, !dbg !38
  %compare.local.base189 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 46) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local190 = or i32 %compare.local.base189, 15, !dbg !48
  %compare.low191 = icmp ugt i32 %compare.low.base129, %compare.local190, !dbg !48
  %compare.result192 = select i1 %compare.high.equal127, i1 %compare.low191, i1 %compare.high.order128, !dbg !48
  %625 = select i1 %compare.result192, float 0.000000e+00, float -inf, !dbg !49
  %626 = extractelement <32 x float> %580, i64 15, !dbg !38
  %627 = fadd float %626, %625, !dbg !38
  %compare.local.base193 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 47) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local194 = or i32 %compare.local.base193, 64, !dbg !48
  %compare.low195 = icmp ugt i32 %compare.low.base129, %compare.local194, !dbg !48
  %compare.result196 = select i1 %compare.high.equal127, i1 %compare.low195, i1 %compare.high.order128, !dbg !48
  %628 = select i1 %compare.result196, float 0.000000e+00, float -inf, !dbg !49
  %629 = extractelement <32 x float> %580, i64 16, !dbg !38
  %630 = fadd float %629, %628, !dbg !38
  %compare.local.base197 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 48) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local198 = or i32 %compare.local.base197, 65, !dbg !48
  %compare.low199 = icmp ugt i32 %compare.low.base129, %compare.local198, !dbg !48
  %compare.result200 = select i1 %compare.high.equal127, i1 %compare.low199, i1 %compare.high.order128, !dbg !48
  %631 = select i1 %compare.result200, float 0.000000e+00, float -inf, !dbg !49
  %632 = extractelement <32 x float> %580, i64 17, !dbg !38
  %633 = fadd float %632, %631, !dbg !38
  %compare.local.base201 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 49) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local202 = or i32 %compare.local.base201, 66, !dbg !48
  %compare.low203 = icmp ugt i32 %compare.low.base129, %compare.local202, !dbg !48
  %compare.result204 = select i1 %compare.high.equal127, i1 %compare.low203, i1 %compare.high.order128, !dbg !48
  %634 = select i1 %compare.result204, float 0.000000e+00, float -inf, !dbg !49
  %635 = extractelement <32 x float> %580, i64 18, !dbg !38
  %636 = fadd float %635, %634, !dbg !38
  %compare.local.base205 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 50) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local206 = or i32 %compare.local.base205, 67, !dbg !48
  %compare.low207 = icmp ugt i32 %compare.low.base129, %compare.local206, !dbg !48
  %compare.result208 = select i1 %compare.high.equal127, i1 %compare.low207, i1 %compare.high.order128, !dbg !48
  %637 = select i1 %compare.result208, float 0.000000e+00, float -inf, !dbg !49
  %638 = extractelement <32 x float> %580, i64 19, !dbg !38
  %639 = fadd float %638, %637, !dbg !38
  %compare.local.base209 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 51) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local210 = or i32 %compare.local.base209, 68, !dbg !48
  %compare.low211 = icmp ugt i32 %compare.low.base129, %compare.local210, !dbg !48
  %compare.result212 = select i1 %compare.high.equal127, i1 %compare.low211, i1 %compare.high.order128, !dbg !48
  %640 = select i1 %compare.result212, float 0.000000e+00, float -inf, !dbg !49
  %641 = extractelement <32 x float> %580, i64 20, !dbg !38
  %642 = fadd float %641, %640, !dbg !38
  %compare.local.base213 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 52) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local214 = or i32 %compare.local.base213, 69, !dbg !48
  %compare.low215 = icmp ugt i32 %compare.low.base129, %compare.local214, !dbg !48
  %compare.result216 = select i1 %compare.high.equal127, i1 %compare.low215, i1 %compare.high.order128, !dbg !48
  %643 = select i1 %compare.result216, float 0.000000e+00, float -inf, !dbg !49
  %644 = extractelement <32 x float> %580, i64 21, !dbg !38
  %645 = fadd float %644, %643, !dbg !38
  %compare.local.base217 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 53) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local218 = or i32 %compare.local.base217, 70, !dbg !48
  %compare.low219 = icmp ugt i32 %compare.low.base129, %compare.local218, !dbg !48
  %compare.result220 = select i1 %compare.high.equal127, i1 %compare.low219, i1 %compare.high.order128, !dbg !48
  %646 = select i1 %compare.result220, float 0.000000e+00, float -inf, !dbg !49
  %647 = extractelement <32 x float> %580, i64 22, !dbg !38
  %648 = fadd float %647, %646, !dbg !38
  %compare.local.base221 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 54) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local222 = or i32 %compare.local.base221, 71, !dbg !48
  %compare.low223 = icmp ugt i32 %compare.low.base129, %compare.local222, !dbg !48
  %compare.result224 = select i1 %compare.high.equal127, i1 %compare.low223, i1 %compare.high.order128, !dbg !48
  %649 = select i1 %compare.result224, float 0.000000e+00, float -inf, !dbg !49
  %650 = extractelement <32 x float> %580, i64 23, !dbg !38
  %651 = fadd float %650, %649, !dbg !38
  %compare.local.base225 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 55) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local226 = or i32 %compare.local.base225, 72, !dbg !48
  %compare.low227 = icmp ugt i32 %compare.low.base129, %compare.local226, !dbg !48
  %compare.result228 = select i1 %compare.high.equal127, i1 %compare.low227, i1 %compare.high.order128, !dbg !48
  %652 = select i1 %compare.result228, float 0.000000e+00, float -inf, !dbg !49
  %653 = extractelement <32 x float> %580, i64 24, !dbg !38
  %654 = fadd float %653, %652, !dbg !38
  %compare.local.base229 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 56) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local230 = or i32 %compare.local.base229, 73, !dbg !48
  %compare.low231 = icmp ugt i32 %compare.low.base129, %compare.local230, !dbg !48
  %compare.result232 = select i1 %compare.high.equal127, i1 %compare.low231, i1 %compare.high.order128, !dbg !48
  %655 = select i1 %compare.result232, float 0.000000e+00, float -inf, !dbg !49
  %656 = extractelement <32 x float> %580, i64 25, !dbg !38
  %657 = fadd float %656, %655, !dbg !38
  %compare.local.base233 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 57) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local234 = or i32 %compare.local.base233, 74, !dbg !48
  %compare.low235 = icmp ugt i32 %compare.low.base129, %compare.local234, !dbg !48
  %compare.result236 = select i1 %compare.high.equal127, i1 %compare.low235, i1 %compare.high.order128, !dbg !48
  %658 = select i1 %compare.result236, float 0.000000e+00, float -inf, !dbg !49
  %659 = extractelement <32 x float> %580, i64 26, !dbg !38
  %660 = fadd float %659, %658, !dbg !38
  %compare.local.base237 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 58) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local238 = or i32 %compare.local.base237, 75, !dbg !48
  %compare.low239 = icmp ugt i32 %compare.low.base129, %compare.local238, !dbg !48
  %compare.result240 = select i1 %compare.high.equal127, i1 %compare.low239, i1 %compare.high.order128, !dbg !48
  %661 = select i1 %compare.result240, float 0.000000e+00, float -inf, !dbg !49
  %662 = extractelement <32 x float> %580, i64 27, !dbg !38
  %663 = fadd float %662, %661, !dbg !38
  %compare.local.base241 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 59) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local242 = or i32 %compare.local.base241, 76, !dbg !48
  %compare.low243 = icmp ugt i32 %compare.low.base129, %compare.local242, !dbg !48
  %compare.result244 = select i1 %compare.high.equal127, i1 %compare.low243, i1 %compare.high.order128, !dbg !48
  %664 = select i1 %compare.result244, float 0.000000e+00, float -inf, !dbg !49
  %665 = extractelement <32 x float> %580, i64 28, !dbg !38
  %666 = fadd float %665, %664, !dbg !38
  %compare.local.base245 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 60) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local246 = or i32 %compare.local.base245, 77, !dbg !48
  %compare.low247 = icmp ugt i32 %compare.low.base129, %compare.local246, !dbg !48
  %compare.result248 = select i1 %compare.high.equal127, i1 %compare.low247, i1 %compare.high.order128, !dbg !48
  %667 = select i1 %compare.result248, float 0.000000e+00, float -inf, !dbg !49
  %668 = extractelement <32 x float> %580, i64 29, !dbg !38
  %669 = fadd float %668, %667, !dbg !38
  %compare.local.base249 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 61) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local250 = or i32 %compare.local.base249, 78, !dbg !48
  %compare.low251 = icmp ugt i32 %compare.low.base129, %compare.local250, !dbg !48
  %compare.result252 = select i1 %compare.high.equal127, i1 %compare.low251, i1 %compare.high.order128, !dbg !48
  %670 = select i1 %compare.result252, float 0.000000e+00, float -inf, !dbg !49
  %671 = extractelement <32 x float> %580, i64 30, !dbg !38
  %672 = fadd float %671, %670, !dbg !38
  %compare.local.base253 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base130, i32 62) #9, !dbg !48, !frisk.register_lifetimes !29
  %compare.local254 = or i32 %compare.local.base253, 79, !dbg !48
  %compare.low255 = icmp ugt i32 %compare.low.base129, %compare.local254, !dbg !48
  %compare.result256 = select i1 %compare.high.equal127, i1 %compare.low255, i1 %compare.high.order128, !dbg !48
  %673 = select i1 %compare.result256, float 0.000000e+00, float -inf, !dbg !49
  %674 = extractelement <32 x float> %580, i64 31, !dbg !38
  %675 = fadd float %674, %673, !dbg !38
  %676 = insertelement <32 x float> poison, float %482, i64 0, !dbg !38
  %677 = insertelement <32 x float> %676, float %485, i64 1, !dbg !38
  %678 = insertelement <32 x float> %677, float %488, i64 2, !dbg !38
  %679 = insertelement <32 x float> %678, float %491, i64 3, !dbg !38
  %680 = insertelement <32 x float> %679, float %494, i64 4, !dbg !38
  %681 = insertelement <32 x float> %680, float %497, i64 5, !dbg !38
  %682 = insertelement <32 x float> %681, float %500, i64 6, !dbg !38
  %683 = insertelement <32 x float> %682, float %503, i64 7, !dbg !38
  %684 = insertelement <32 x float> %683, float %506, i64 8, !dbg !38
  %685 = insertelement <32 x float> %684, float %509, i64 9, !dbg !38
  %686 = insertelement <32 x float> %685, float %512, i64 10, !dbg !38
  %687 = insertelement <32 x float> %686, float %515, i64 11, !dbg !38
  %688 = insertelement <32 x float> %687, float %518, i64 12, !dbg !38
  %689 = insertelement <32 x float> %688, float %521, i64 13, !dbg !38
  %690 = insertelement <32 x float> %689, float %524, i64 14, !dbg !38
  %691 = insertelement <32 x float> %690, float %527, i64 15, !dbg !38
  %692 = insertelement <32 x float> %691, float %530, i64 16, !dbg !38
  %693 = insertelement <32 x float> %692, float %533, i64 17, !dbg !38
  %694 = insertelement <32 x float> %693, float %536, i64 18, !dbg !38
  %695 = insertelement <32 x float> %694, float %539, i64 19, !dbg !38
  %696 = insertelement <32 x float> %695, float %542, i64 20, !dbg !38
  %697 = insertelement <32 x float> %696, float %545, i64 21, !dbg !38
  %698 = insertelement <32 x float> %697, float %548, i64 22, !dbg !38
  %699 = insertelement <32 x float> %698, float %551, i64 23, !dbg !38
  %700 = insertelement <32 x float> %699, float %554, i64 24, !dbg !38
  %701 = insertelement <32 x float> %700, float %557, i64 25, !dbg !38
  %702 = insertelement <32 x float> %701, float %560, i64 26, !dbg !38
  %703 = insertelement <32 x float> %702, float %563, i64 27, !dbg !38
  %704 = insertelement <32 x float> %703, float %566, i64 28, !dbg !38
  %705 = insertelement <32 x float> %704, float %569, i64 29, !dbg !38
  %706 = insertelement <32 x float> %705, float %572, i64 30, !dbg !38
  %707 = insertelement <32 x float> %706, float %575, i64 31, !dbg !38
  %708 = insertelement <32 x float> poison, float %582, i64 0, !dbg !38
  %709 = insertelement <32 x float> %708, float %585, i64 1, !dbg !38
  %710 = insertelement <32 x float> %709, float %588, i64 2, !dbg !38
  %711 = insertelement <32 x float> %710, float %591, i64 3, !dbg !38
  %712 = insertelement <32 x float> %711, float %594, i64 4, !dbg !38
  %713 = insertelement <32 x float> %712, float %597, i64 5, !dbg !38
  %714 = insertelement <32 x float> %713, float %600, i64 6, !dbg !38
  %715 = insertelement <32 x float> %714, float %603, i64 7, !dbg !38
  %716 = insertelement <32 x float> %715, float %606, i64 8, !dbg !38
  %717 = insertelement <32 x float> %716, float %609, i64 9, !dbg !38
  %718 = insertelement <32 x float> %717, float %612, i64 10, !dbg !38
  %719 = insertelement <32 x float> %718, float %615, i64 11, !dbg !38
  %720 = insertelement <32 x float> %719, float %618, i64 12, !dbg !38
  %721 = insertelement <32 x float> %720, float %621, i64 13, !dbg !38
  %722 = insertelement <32 x float> %721, float %624, i64 14, !dbg !38
  %723 = insertelement <32 x float> %722, float %627, i64 15, !dbg !38
  %724 = insertelement <32 x float> %723, float %630, i64 16, !dbg !38
  %725 = insertelement <32 x float> %724, float %633, i64 17, !dbg !38
  %726 = insertelement <32 x float> %725, float %636, i64 18, !dbg !38
  %727 = insertelement <32 x float> %726, float %639, i64 19, !dbg !38
  %728 = insertelement <32 x float> %727, float %642, i64 20, !dbg !38
  %729 = insertelement <32 x float> %728, float %645, i64 21, !dbg !38
  %730 = insertelement <32 x float> %729, float %648, i64 22, !dbg !38
  %731 = insertelement <32 x float> %730, float %651, i64 23, !dbg !38
  %732 = insertelement <32 x float> %731, float %654, i64 24, !dbg !38
  %733 = insertelement <32 x float> %732, float %657, i64 25, !dbg !38
  %734 = insertelement <32 x float> %733, float %660, i64 26, !dbg !38
  %735 = insertelement <32 x float> %734, float %663, i64 27, !dbg !38
  %736 = insertelement <32 x float> %735, float %666, i64 28, !dbg !38
  %737 = insertelement <32 x float> %736, float %669, i64 29, !dbg !38
  %738 = insertelement <32 x float> %737, float %672, i64 30, !dbg !38
  %739 = insertelement <32 x float> %738, float %675, i64 31, !dbg !38
  br label %740, !dbg !38

740:                                              ; preds = %468, %273
  %.pn70 = phi <32 x float> [ %707, %468 ], [ %435, %273 ]
  %.pn68 = phi <32 x float> [ %739, %468 ], [ %467, %273 ]
  br label %741, !dbg !38

741:                                              ; preds = %740
  %742 = extractelement <32 x float> %.pn70, i64 0, !dbg !50
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !50
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !50
  %745 = extractelement <32 x float> %.pn70, i64 1, !dbg !50
  %746 = call float @llvm.amdgcn.exp2.f32(float %745), !dbg !50
  %747 = fadd float %743, %746, !dbg !51
  %748 = insertelement <4 x float> %744, float %746, i64 1, !dbg !50
  %749 = extractelement <32 x float> %.pn70, i64 2, !dbg !50
  %750 = call float @llvm.amdgcn.exp2.f32(float %749), !dbg !50
  %751 = fadd float %747, %750, !dbg !51
  %752 = insertelement <4 x float> %748, float %750, i64 2, !dbg !50
  %753 = extractelement <32 x float> %.pn70, i64 3, !dbg !50
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !50
  %755 = fadd float %751, %754, !dbg !51
  %756 = insertelement <4 x float> %752, float %754, i64 3, !dbg !50
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #9, !dbg !52, !frisk.register_lifetimes !29
  %757 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !52
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #9, !dbg !50, !frisk.register_lifetimes !29
  %758 = extractelement <32 x float> %.pn70, i64 4, !dbg !50
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !50
  %760 = fadd float %755, %759, !dbg !51
  %761 = insertelement <4 x float> poison, float %759, i64 0, !dbg !50
  %762 = extractelement <32 x float> %.pn70, i64 5, !dbg !50
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !50
  %764 = fadd float %760, %763, !dbg !51
  %765 = insertelement <4 x float> %761, float %763, i64 1, !dbg !50
  %766 = extractelement <32 x float> %.pn70, i64 6, !dbg !50
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !50
  %768 = fadd float %764, %767, !dbg !51
  %769 = insertelement <4 x float> %765, float %767, i64 2, !dbg !50
  %770 = extractelement <32 x float> %.pn70, i64 7, !dbg !50
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !50
  %772 = fadd float %768, %771, !dbg !51
  %773 = insertelement <4 x float> %769, float %771, i64 3, !dbg !50
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %773) #9, !dbg !52, !frisk.register_lifetimes !29
  %774 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !52
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %774) #9, !dbg !50, !frisk.register_lifetimes !29
  %775 = extractelement <32 x float> %.pn70, i64 8, !dbg !50
  %776 = call float @llvm.amdgcn.exp2.f32(float %775), !dbg !50
  %777 = fadd float %772, %776, !dbg !51
  %778 = insertelement <4 x float> poison, float %776, i64 0, !dbg !50
  %779 = extractelement <32 x float> %.pn70, i64 9, !dbg !50
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !50
  %781 = fadd float %777, %780, !dbg !51
  %782 = insertelement <4 x float> %778, float %780, i64 1, !dbg !50
  %783 = extractelement <32 x float> %.pn70, i64 10, !dbg !50
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !50
  %785 = fadd float %781, %784, !dbg !51
  %786 = insertelement <4 x float> %782, float %784, i64 2, !dbg !50
  %787 = extractelement <32 x float> %.pn70, i64 11, !dbg !50
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !50
  %789 = fadd float %785, %788, !dbg !51
  %790 = insertelement <4 x float> %786, float %788, i64 3, !dbg !50
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %790) #9, !dbg !52, !frisk.register_lifetimes !29
  %791 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !52
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %791) #9, !dbg !50, !frisk.register_lifetimes !29
  %792 = extractelement <32 x float> %.pn70, i64 12, !dbg !50
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !50
  %794 = fadd float %789, %793, !dbg !51
  %795 = insertelement <4 x float> poison, float %793, i64 0, !dbg !50
  %796 = extractelement <32 x float> %.pn70, i64 13, !dbg !50
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !50
  %798 = fadd float %794, %797, !dbg !51
  %799 = insertelement <4 x float> %795, float %797, i64 1, !dbg !50
  %800 = extractelement <32 x float> %.pn70, i64 14, !dbg !50
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !50
  %802 = fadd float %798, %801, !dbg !51
  %803 = insertelement <4 x float> %799, float %801, i64 2, !dbg !50
  %804 = extractelement <32 x float> %.pn70, i64 15, !dbg !50
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !50
  %806 = fadd float %802, %805, !dbg !51
  %807 = insertelement <4 x float> %803, float %805, i64 3, !dbg !50
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %807) #9, !dbg !52, !frisk.register_lifetimes !29
  %808 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !52
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %808) #9, !dbg !50, !frisk.register_lifetimes !29
  %809 = extractelement <32 x float> %.pn70, i64 16, !dbg !50
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !50
  %811 = fadd float %806, %810, !dbg !51
  %812 = insertelement <4 x float> poison, float %810, i64 0, !dbg !50
  %813 = extractelement <32 x float> %.pn70, i64 17, !dbg !50
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !50
  %815 = fadd float %811, %814, !dbg !51
  %816 = insertelement <4 x float> %812, float %814, i64 1, !dbg !50
  %817 = extractelement <32 x float> %.pn70, i64 18, !dbg !50
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !50
  %819 = fadd float %815, %818, !dbg !51
  %820 = insertelement <4 x float> %816, float %818, i64 2, !dbg !50
  %821 = extractelement <32 x float> %.pn70, i64 19, !dbg !50
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !50
  %823 = fadd float %819, %822, !dbg !51
  %824 = insertelement <4 x float> %820, float %822, i64 3, !dbg !50
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %824) #9, !dbg !52, !frisk.register_lifetimes !29
  %825 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !52
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %825) #9, !dbg !50, !frisk.register_lifetimes !29
  %826 = extractelement <32 x float> %.pn70, i64 20, !dbg !50
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !50
  %828 = fadd float %823, %827, !dbg !51
  %829 = insertelement <4 x float> poison, float %827, i64 0, !dbg !50
  %830 = extractelement <32 x float> %.pn70, i64 21, !dbg !50
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !50
  %832 = fadd float %828, %831, !dbg !51
  %833 = insertelement <4 x float> %829, float %831, i64 1, !dbg !50
  %834 = extractelement <32 x float> %.pn70, i64 22, !dbg !50
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !50
  %836 = fadd float %832, %835, !dbg !51
  %837 = insertelement <4 x float> %833, float %835, i64 2, !dbg !50
  %838 = extractelement <32 x float> %.pn70, i64 23, !dbg !50
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !50
  %840 = fadd float %836, %839, !dbg !51
  %841 = insertelement <4 x float> %837, float %839, i64 3, !dbg !50
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %841) #9, !dbg !52, !frisk.register_lifetimes !29
  %842 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !52
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %842) #9, !dbg !50, !frisk.register_lifetimes !29
  %843 = extractelement <32 x float> %.pn70, i64 24, !dbg !50
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !50
  %845 = fadd float %840, %844, !dbg !51
  %846 = insertelement <4 x float> poison, float %844, i64 0, !dbg !50
  %847 = extractelement <32 x float> %.pn70, i64 25, !dbg !50
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !50
  %849 = fadd float %845, %848, !dbg !51
  %850 = insertelement <4 x float> %846, float %848, i64 1, !dbg !50
  %851 = extractelement <32 x float> %.pn70, i64 26, !dbg !50
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !50
  %853 = fadd float %849, %852, !dbg !51
  %854 = insertelement <4 x float> %850, float %852, i64 2, !dbg !50
  %855 = extractelement <32 x float> %.pn70, i64 27, !dbg !50
  %856 = call float @llvm.amdgcn.exp2.f32(float %855), !dbg !50
  %857 = fadd float %853, %856, !dbg !51
  %858 = insertelement <4 x float> %854, float %856, i64 3, !dbg !50
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %858) #9, !dbg !52, !frisk.register_lifetimes !29
  %859 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !52
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %859) #9, !dbg !50, !frisk.register_lifetimes !29
  %860 = extractelement <32 x float> %.pn70, i64 28, !dbg !50
  %861 = call float @llvm.amdgcn.exp2.f32(float %860), !dbg !50
  %862 = fadd float %857, %861, !dbg !51
  %863 = insertelement <4 x float> poison, float %861, i64 0, !dbg !50
  %864 = extractelement <32 x float> %.pn70, i64 29, !dbg !50
  %865 = call float @llvm.amdgcn.exp2.f32(float %864), !dbg !50
  %866 = fadd float %862, %865, !dbg !51
  %867 = insertelement <4 x float> %863, float %865, i64 1, !dbg !50
  %868 = extractelement <32 x float> %.pn70, i64 30, !dbg !50
  %869 = call float @llvm.amdgcn.exp2.f32(float %868), !dbg !50
  %870 = fadd float %866, %869, !dbg !51
  %871 = insertelement <4 x float> %867, float %869, i64 2, !dbg !50
  %872 = extractelement <32 x float> %.pn70, i64 31, !dbg !50
  %873 = call float @llvm.amdgcn.exp2.f32(float %872), !dbg !50
  %874 = fadd float %870, %873, !dbg !51
  %875 = insertelement <4 x float> %871, float %873, i64 3, !dbg !50
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %875) #9, !dbg !52, !frisk.register_lifetimes !29
  %876 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !52
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %876) #9, !dbg !50, !frisk.register_lifetimes !29
  %877 = extractelement <32 x float> %.pn68, i64 0, !dbg !50
  %878 = call float @llvm.amdgcn.exp2.f32(float %877), !dbg !50
  %879 = insertelement <4 x float> poison, float %878, i64 0, !dbg !50
  %880 = extractelement <32 x float> %.pn68, i64 1, !dbg !50
  %881 = call float @llvm.amdgcn.exp2.f32(float %880), !dbg !50
  %882 = fadd float %878, %881, !dbg !51
  %883 = insertelement <4 x float> %879, float %881, i64 1, !dbg !50
  %884 = extractelement <32 x float> %.pn68, i64 2, !dbg !50
  %885 = call float @llvm.amdgcn.exp2.f32(float %884), !dbg !50
  %886 = fadd float %882, %885, !dbg !51
  %887 = insertelement <4 x float> %883, float %885, i64 2, !dbg !50
  %888 = extractelement <32 x float> %.pn68, i64 3, !dbg !50
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !50
  %890 = fadd float %886, %889, !dbg !51
  %891 = insertelement <4 x float> %887, float %889, i64 3, !dbg !50
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %891) #9, !dbg !52, !frisk.register_lifetimes !29
  %892 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !52
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %892) #9, !dbg !50, !frisk.register_lifetimes !29
  %893 = extractelement <32 x float> %.pn68, i64 4, !dbg !50
  %894 = call float @llvm.amdgcn.exp2.f32(float %893), !dbg !50
  %895 = fadd float %890, %894, !dbg !51
  %896 = insertelement <4 x float> poison, float %894, i64 0, !dbg !50
  %897 = extractelement <32 x float> %.pn68, i64 5, !dbg !50
  %898 = call float @llvm.amdgcn.exp2.f32(float %897), !dbg !50
  %899 = fadd float %895, %898, !dbg !51
  %900 = insertelement <4 x float> %896, float %898, i64 1, !dbg !50
  %901 = extractelement <32 x float> %.pn68, i64 6, !dbg !50
  %902 = call float @llvm.amdgcn.exp2.f32(float %901), !dbg !50
  %903 = fadd float %899, %902, !dbg !51
  %904 = insertelement <4 x float> %900, float %902, i64 2, !dbg !50
  %905 = extractelement <32 x float> %.pn68, i64 7, !dbg !50
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !50
  %907 = fadd float %903, %906, !dbg !51
  %908 = insertelement <4 x float> %904, float %906, i64 3, !dbg !50
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %908) #9, !dbg !52, !frisk.register_lifetimes !29
  %909 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !52
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %909) #9, !dbg !50, !frisk.register_lifetimes !29
  %910 = extractelement <32 x float> %.pn68, i64 8, !dbg !50
  %911 = call float @llvm.amdgcn.exp2.f32(float %910), !dbg !50
  %912 = fadd float %907, %911, !dbg !51
  %913 = insertelement <4 x float> poison, float %911, i64 0, !dbg !50
  %914 = extractelement <32 x float> %.pn68, i64 9, !dbg !50
  %915 = call float @llvm.amdgcn.exp2.f32(float %914), !dbg !50
  %916 = fadd float %912, %915, !dbg !51
  %917 = insertelement <4 x float> %913, float %915, i64 1, !dbg !50
  %918 = extractelement <32 x float> %.pn68, i64 10, !dbg !50
  %919 = call float @llvm.amdgcn.exp2.f32(float %918), !dbg !50
  %920 = fadd float %916, %919, !dbg !51
  %921 = insertelement <4 x float> %917, float %919, i64 2, !dbg !50
  %922 = extractelement <32 x float> %.pn68, i64 11, !dbg !50
  %923 = call float @llvm.amdgcn.exp2.f32(float %922), !dbg !50
  %924 = fadd float %920, %923, !dbg !51
  %925 = insertelement <4 x float> %921, float %923, i64 3, !dbg !50
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %925) #9, !dbg !52, !frisk.register_lifetimes !29
  %926 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !52
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %926) #9, !dbg !50, !frisk.register_lifetimes !29
  %927 = extractelement <32 x float> %.pn68, i64 12, !dbg !50
  %928 = call float @llvm.amdgcn.exp2.f32(float %927), !dbg !50
  %929 = fadd float %924, %928, !dbg !51
  %930 = insertelement <4 x float> poison, float %928, i64 0, !dbg !50
  %931 = extractelement <32 x float> %.pn68, i64 13, !dbg !50
  %932 = call float @llvm.amdgcn.exp2.f32(float %931), !dbg !50
  %933 = fadd float %929, %932, !dbg !51
  %934 = insertelement <4 x float> %930, float %932, i64 1, !dbg !50
  %935 = extractelement <32 x float> %.pn68, i64 14, !dbg !50
  %936 = call float @llvm.amdgcn.exp2.f32(float %935), !dbg !50
  %937 = fadd float %933, %936, !dbg !51
  %938 = insertelement <4 x float> %934, float %936, i64 2, !dbg !50
  %939 = extractelement <32 x float> %.pn68, i64 15, !dbg !50
  %940 = call float @llvm.amdgcn.exp2.f32(float %939), !dbg !50
  %941 = fadd float %937, %940, !dbg !51
  %942 = insertelement <4 x float> %938, float %940, i64 3, !dbg !50
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %942) #9, !dbg !52, !frisk.register_lifetimes !29
  %943 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !52
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %943) #9, !dbg !50, !frisk.register_lifetimes !29
  %944 = extractelement <32 x float> %.pn68, i64 16, !dbg !50
  %945 = call float @llvm.amdgcn.exp2.f32(float %944), !dbg !50
  %946 = fadd float %941, %945, !dbg !51
  %947 = insertelement <4 x float> poison, float %945, i64 0, !dbg !50
  %948 = extractelement <32 x float> %.pn68, i64 17, !dbg !50
  %949 = call float @llvm.amdgcn.exp2.f32(float %948), !dbg !50
  %950 = fadd float %946, %949, !dbg !51
  %951 = insertelement <4 x float> %947, float %949, i64 1, !dbg !50
  %952 = extractelement <32 x float> %.pn68, i64 18, !dbg !50
  %953 = call float @llvm.amdgcn.exp2.f32(float %952), !dbg !50
  %954 = fadd float %950, %953, !dbg !51
  %955 = insertelement <4 x float> %951, float %953, i64 2, !dbg !50
  %956 = extractelement <32 x float> %.pn68, i64 19, !dbg !50
  %957 = call float @llvm.amdgcn.exp2.f32(float %956), !dbg !50
  %958 = fadd float %954, %957, !dbg !51
  %959 = insertelement <4 x float> %955, float %957, i64 3, !dbg !50
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %959) #9, !dbg !52, !frisk.register_lifetimes !29
  %960 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !52
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %960) #9, !dbg !50, !frisk.register_lifetimes !29
  %961 = extractelement <32 x float> %.pn68, i64 20, !dbg !50
  %962 = call float @llvm.amdgcn.exp2.f32(float %961), !dbg !50
  %963 = fadd float %958, %962, !dbg !51
  %964 = insertelement <4 x float> poison, float %962, i64 0, !dbg !50
  %965 = extractelement <32 x float> %.pn68, i64 21, !dbg !50
  %966 = call float @llvm.amdgcn.exp2.f32(float %965), !dbg !50
  %967 = fadd float %963, %966, !dbg !51
  %968 = insertelement <4 x float> %964, float %966, i64 1, !dbg !50
  %969 = extractelement <32 x float> %.pn68, i64 22, !dbg !50
  %970 = call float @llvm.amdgcn.exp2.f32(float %969), !dbg !50
  %971 = fadd float %967, %970, !dbg !51
  %972 = insertelement <4 x float> %968, float %970, i64 2, !dbg !50
  %973 = extractelement <32 x float> %.pn68, i64 23, !dbg !50
  %974 = call float @llvm.amdgcn.exp2.f32(float %973), !dbg !50
  %975 = fadd float %971, %974, !dbg !51
  %976 = insertelement <4 x float> %972, float %974, i64 3, !dbg !50
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %976) #9, !dbg !52, !frisk.register_lifetimes !29
  %977 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !52
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %977) #9, !dbg !50, !frisk.register_lifetimes !29
  %978 = extractelement <32 x float> %.pn68, i64 24, !dbg !50
  %979 = call float @llvm.amdgcn.exp2.f32(float %978), !dbg !50
  %980 = fadd float %975, %979, !dbg !51
  %981 = insertelement <4 x float> poison, float %979, i64 0, !dbg !50
  %982 = extractelement <32 x float> %.pn68, i64 25, !dbg !50
  %983 = call float @llvm.amdgcn.exp2.f32(float %982), !dbg !50
  %984 = fadd float %980, %983, !dbg !51
  %985 = insertelement <4 x float> %981, float %983, i64 1, !dbg !50
  %986 = extractelement <32 x float> %.pn68, i64 26, !dbg !50
  %987 = call float @llvm.amdgcn.exp2.f32(float %986), !dbg !50
  %988 = fadd float %984, %987, !dbg !51
  %989 = insertelement <4 x float> %985, float %987, i64 2, !dbg !50
  %990 = extractelement <32 x float> %.pn68, i64 27, !dbg !50
  %991 = call float @llvm.amdgcn.exp2.f32(float %990), !dbg !50
  %992 = fadd float %988, %991, !dbg !51
  %993 = insertelement <4 x float> %989, float %991, i64 3, !dbg !50
  %math.result129 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %993) #9, !dbg !52, !frisk.register_lifetimes !29
  %994 = fptrunc <4 x float> %math.result129 to <4 x half>, !dbg !52
  %math.packed130 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %994) #9, !dbg !50, !frisk.register_lifetimes !29
  %995 = extractelement <32 x float> %.pn68, i64 28, !dbg !50
  %996 = call float @llvm.amdgcn.exp2.f32(float %995), !dbg !50
  %997 = fadd float %992, %996, !dbg !51
  %998 = insertelement <4 x float> poison, float %996, i64 0, !dbg !50
  %999 = extractelement <32 x float> %.pn68, i64 29, !dbg !50
  %1000 = call float @llvm.amdgcn.exp2.f32(float %999), !dbg !50
  %1001 = fadd float %997, %1000, !dbg !51
  %1002 = insertelement <4 x float> %998, float %1000, i64 1, !dbg !50
  %1003 = extractelement <32 x float> %.pn68, i64 30, !dbg !50
  %1004 = call float @llvm.amdgcn.exp2.f32(float %1003), !dbg !50
  %1005 = fadd float %1001, %1004, !dbg !51
  %1006 = insertelement <4 x float> %1002, float %1004, i64 2, !dbg !50
  %1007 = extractelement <32 x float> %.pn68, i64 31, !dbg !50
  %1008 = call float @llvm.amdgcn.exp2.f32(float %1007), !dbg !50
  %1009 = fadd float %1005, %1008, !dbg !51
  %1010 = insertelement <4 x float> %1006, float %1008, i64 3, !dbg !50
  %math.result131 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1010) #9, !dbg !52, !frisk.register_lifetimes !29
  %1011 = fptrunc <4 x float> %math.result131 to <4 x half>, !dbg !52
  %math.packed132 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1011) #9, !dbg !51, !frisk.register_lifetimes !29
  %1012 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !51
  %1013 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1012), !dbg !51
  %1014 = add nuw nsw i32 %1013, 64, !dbg !51
  %1015 = xor i32 %1013, 16, !dbg !51
  %1016 = and i32 %1014, 192, !dbg !51
  %1017 = icmp samesign ult i32 %1015, %1016, !dbg !51
  %1018 = select i1 %1017, i32 %1015, i32 %1013, !dbg !51
  %1019 = shl nuw nsw i32 %1018, 2, !dbg !51
  %1020 = bitcast float %874 to i32, !dbg !51
  %1021 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1019, i32 %1020), !dbg !51
  %1022 = bitcast i32 %1021 to float, !dbg !51
  %1023 = fadd float %874, %1022, !dbg !51
  %1024 = xor i32 %1013, 32, !dbg !51
  %1025 = icmp samesign ult i32 %1024, %1016, !dbg !51
  %1026 = select i1 %1025, i32 %1024, i32 %1013, !dbg !51
  %1027 = shl nuw nsw i32 %1026, 2, !dbg !51
  %1028 = bitcast float %1023 to i32, !dbg !51
  %1029 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1027, i32 %1028), !dbg !51
  %1030 = bitcast i32 %1029 to float, !dbg !51
  %1031 = fadd float %1023, %1030, !dbg !51
  %1032 = extractvalue [2 x <1 x float>] %28, 0, !dbg !51
  %1033 = extractelement <1 x float> %1032, i64 0, !dbg !51
  %1034 = fadd float %1031, %1033, !dbg !51
  %1035 = insertelement <1 x float> poison, float %1034, i64 0, !dbg !40
  %1036 = insertvalue [2 x <1 x float>] poison, <1 x float> %1035, 0, !dbg !40
  %1037 = bitcast float %1009 to i32, !dbg !51
  %1038 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1019, i32 %1037), !dbg !51
  %1039 = bitcast i32 %1038 to float, !dbg !51
  %1040 = fadd float %1009, %1039, !dbg !51
  %1041 = bitcast float %1040 to i32, !dbg !51
  %1042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1027, i32 %1041), !dbg !51
  %1043 = bitcast i32 %1042 to float, !dbg !51
  %1044 = fadd float %1040, %1043, !dbg !51
  %1045 = extractvalue [2 x <1 x float>] %28, 1, !dbg !51
  %1046 = extractelement <1 x float> %1045, i64 0, !dbg !51
  %1047 = fadd float %1044, %1046, !dbg !51
  %1048 = insertelement <1 x float> poison, float %1047, i64 0, !dbg !40
  %1049 = insertvalue [2 x <1 x float>] %1036, <1 x float> %1048, 1, !dbg !40
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
  %1050 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1051 = bitcast <4 x i32> %1050 to <8 x half>, !dbg !39
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
  %1052 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1053 = bitcast <4 x i32> %1052 to <8 x half>, !dbg !39
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local339, !dbg !39
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) %address.local346, i64 %address.local342, !dbg !39
  %address.local348 = getelementptr i8, ptr addrspace(3) %address.local347, i64 512, !dbg !39
  %1054 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1055 = bitcast <4 x i32> %1054 to <8 x half>, !dbg !39
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) %address.local349, i64 %address.local339, !dbg !39
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local342, !dbg !39
  %address.local352 = getelementptr i8, ptr addrspace(3) %address.local351, i64 768, !dbg !39
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !39
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !39
  %address.local354 = getelementptr i8, ptr addrspace(3) %address.local353, i64 4096, !dbg !39
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !39
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local339, !dbg !39
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local342, !dbg !39
  %address.local358 = getelementptr i8, ptr addrspace(3) %address.local357, i64 4352, !dbg !39
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !39
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local339, !dbg !39
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %address.local360, i64 %address.local342, !dbg !39
  %address.local362 = getelementptr i8, ptr addrspace(3) %address.local361, i64 4608, !dbg !39
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !39
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local335, !dbg !39
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) %address.local363, i64 %address.local339, !dbg !39
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) %address.local364, i64 %address.local342, !dbg !39
  %address.local366 = getelementptr i8, ptr addrspace(3) %address.local365, i64 4864, !dbg !39
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !39
  %1066 = add nsw i64 %34, %8, !dbg !42
  %1067 = shl i64 %1066, 19, !dbg !42
  %1068 = and i64 %1067, 16252928, !dbg !42
  %1069 = add i64 %1068, %239, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1070 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1069, !dbg !42
  %1071 = getelementptr i8, ptr addrspace(1) %1070, i64 64, !dbg !42
  %address.local367 = shl nuw nsw i64 %address.local332, 3, !dbg !36
  %address.local368 = and i64 %address.local367, 1536, !dbg !36
  %address.local369 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local368, !dbg !36
  %address.local370 = ptrtoint ptr addrspace(3) %address.local369 to i64, !dbg !36
  %address.local371 = trunc i64 %address.local370 to i32, !dbg !36
  %1072 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !42
  %1073 = zext i32 %1072 to i64, !dbg !42
  %1074 = inttoptr i64 %1073 to ptr addrspace(3), !dbg !42
  %address.local372 = shl nuw nsw i64 %address.local332, 5, !dbg !42
  %address.local373 = and i64 %address.local372, 8064, !dbg !42
  %address.local374 = shl nuw nsw i64 %address.local332, 3, !dbg !42
  %address.local375 = and i64 %address.local374, 24, !dbg !42
  %address.local376 = or disjoint i64 %address.local373, %address.local375, !dbg !42
  %address.local377 = getelementptr [2 x i8], ptr addrspace(1) %1071, i64 %address.local376, !dbg !42
  %1075 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1076 = ptrtoint ptr addrspace(1) %address.local377 to i64, !dbg !42
  %1077 = sub i64 %1076, %1075, !dbg !42
  %dma.buffer.offset641 = trunc i64 %1077 to i32, !dbg !42
  %1078 = trunc i64 %1075 to i32, !dbg !42
  %1079 = insertelement <4 x i32> poison, i32 %1078, i64 0, !dbg !42
  %1080 = lshr i64 %1075, 32, !dbg !42
  %1081 = trunc i64 %1080 to i32, !dbg !42
  %1082 = insertelement <4 x i32> %1079, i32 %1081, i64 1, !dbg !42
  %1083 = insertelement <4 x i32> %1082, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc642 = insertelement <4 x i32> %1083, i32 131072, i64 3, !dbg !42
  %1084 = ptrtoint ptr addrspace(3) %1074 to i32, !dbg !42
  %dma.buffer.lds643 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1084), !dbg !42
  %1085 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds643, i32 %dma.buffer.offset641, <4 x i32> %dma.buffer.rsrc642) #10, !dbg !42
  %address.local378 = or disjoint i64 %address.local332, 256, !dbg !36
  %address.local379 = shl nuw nsw i64 %address.local378, 3, !dbg !36
  %address.local380 = and i64 %address.local379, 3584, !dbg !36
  %address.local381 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local380, !dbg !36
  %address.local382 = ptrtoint ptr addrspace(3) %address.local381 to i64, !dbg !36
  %address.local383 = trunc i64 %address.local382 to i32, !dbg !36
  %1086 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !42
  %1087 = zext i32 %1086 to i64, !dbg !42
  %1088 = inttoptr i64 %1087 to ptr addrspace(3), !dbg !42
  %address.local384 = shl nuw nsw i64 %address.local378, 5, !dbg !42
  %address.local385 = and i64 %address.local384, 16256, !dbg !42
  %address.local386 = shl nuw nsw i64 %address.local332, 3, !dbg !42
  %address.local387 = and i64 %address.local386, 24, !dbg !42
  %address.local388 = or disjoint i64 %address.local385, %address.local387, !dbg !42
  %address.local389 = getelementptr [2 x i8], ptr addrspace(1) %1071, i64 %address.local388, !dbg !42
  %1089 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1090 = ptrtoint ptr addrspace(1) %address.local389 to i64, !dbg !42
  %1091 = sub i64 %1090, %1089, !dbg !42
  %dma.buffer.offset638 = trunc i64 %1091 to i32, !dbg !42
  %1092 = trunc i64 %1089 to i32, !dbg !42
  %1093 = insertelement <4 x i32> poison, i32 %1092, i64 0, !dbg !42
  %1094 = lshr i64 %1089, 32, !dbg !42
  %1095 = trunc i64 %1094 to i32, !dbg !42
  %1096 = insertelement <4 x i32> %1093, i32 %1095, i64 1, !dbg !42
  %1097 = insertelement <4 x i32> %1096, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc639 = insertelement <4 x i32> %1097, i32 131072, i64 3, !dbg !42
  %1098 = ptrtoint ptr addrspace(3) %1088 to i32, !dbg !42
  %dma.buffer.lds640 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1098), !dbg !42
  %1099 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds640, i32 %dma.buffer.offset638, <4 x i32> %dma.buffer.rsrc639) #10, !dbg !42
  %1100 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1101 = extractvalue [2 x <8 x float>] %27, 0, !dbg !53
  %1102 = shufflevector <8 x float> %1101, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1100, <4 x float> %1102) #9, !dbg !39
  %1104 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1104, <4 x float> %1103) #9, !dbg !39
  %1106 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1106, <4 x float> %1105) #9, !dbg !39
  %1108 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1108, <4 x float> %1107) #9, !dbg !39
  %1110 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1110, <4 x float> %1109) #9, !dbg !39
  %1112 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1112, <4 x float> %1111) #9, !dbg !39
  %1114 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1114, <4 x float> %1113) #9, !dbg !39
  %1116 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1116, <4 x float> %1115) #9, !dbg !39
  %1118 = shufflevector <8 x half> %1051, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1119 = shufflevector <8 x float> %1101, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1118, <4 x float> %1119) #9, !dbg !39
  %1121 = shufflevector <8 x half> %1053, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1121, <4 x float> %1120) #9, !dbg !39
  %1123 = shufflevector <8 x half> %1055, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1123, <4 x float> %1122) #9, !dbg !39
  %1125 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1125, <4 x float> %1124) #9, !dbg !39
  %1127 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1127, <4 x float> %1126) #9, !dbg !39
  %1129 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1129, <4 x float> %1128) #9, !dbg !39
  %1131 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1131, <4 x float> %1130) #9, !dbg !39
  %1133 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1133, <4 x float> %1132) #9, !dbg !39
  %1135 = extractvalue [2 x <8 x float>] %27, 1, !dbg !53
  %1136 = shufflevector <8 x float> %1135, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1100, <4 x float> %1136) #9, !dbg !39
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1104, <4 x float> %1137) #9, !dbg !39
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1106, <4 x float> %1138) #9, !dbg !39
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1108, <4 x float> %1139) #9, !dbg !39
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1110, <4 x float> %1140) #9, !dbg !39
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1112, <4 x float> %1141) #9, !dbg !39
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1114, <4 x float> %1142) #9, !dbg !39
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1116, <4 x float> %1143) #9, !dbg !39
  %1145 = shufflevector <8 x float> %1135, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1118, <4 x float> %1145) #9, !dbg !39
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1121, <4 x float> %1146) #9, !dbg !39
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1123, <4 x float> %1147) #9, !dbg !39
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1125, <4 x float> %1148) #9, !dbg !39
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1127, <4 x float> %1149) #9, !dbg !39
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1129, <4 x float> %1150) #9, !dbg !39
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1131, <4 x float> %1151) #9, !dbg !39
  %1153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1133, <4 x float> %1152) #9, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1154 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1155 = bitcast <4 x i32> %1154 to <8 x half>, !dbg !39
  %1156 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1157 = bitcast <4 x i32> %1156 to <8 x half>, !dbg !39
  %1158 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1159 = bitcast <4 x i32> %1158 to <8 x half>, !dbg !39
  %1160 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1161 = bitcast <4 x i32> %1160 to <8 x half>, !dbg !39
  %1162 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1163 = bitcast <4 x i32> %1162 to <8 x half>, !dbg !39
  %1164 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1165 = bitcast <4 x i32> %1164 to <8 x half>, !dbg !39
  %1166 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1167 = bitcast <4 x i32> %1166 to <8 x half>, !dbg !39
  %1168 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1169 = bitcast <4 x i32> %1168 to <8 x half>, !dbg !39
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1170 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1069, !dbg !42
  %1171 = getelementptr i8, ptr addrspace(1) %1170, i64 128, !dbg !42
  %1172 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !42
  %1173 = zext i32 %1172 to i64, !dbg !42
  %1174 = inttoptr i64 %1173 to ptr addrspace(3), !dbg !42
  %address.local390 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local376, !dbg !42
  %1175 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1176 = ptrtoint ptr addrspace(1) %address.local390 to i64, !dbg !42
  %1177 = sub i64 %1176, %1175, !dbg !42
  %dma.buffer.offset635 = trunc i64 %1177 to i32, !dbg !42
  %1178 = trunc i64 %1175 to i32, !dbg !42
  %1179 = insertelement <4 x i32> poison, i32 %1178, i64 0, !dbg !42
  %1180 = lshr i64 %1175, 32, !dbg !42
  %1181 = trunc i64 %1180 to i32, !dbg !42
  %1182 = insertelement <4 x i32> %1179, i32 %1181, i64 1, !dbg !42
  %1183 = insertelement <4 x i32> %1182, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc636 = insertelement <4 x i32> %1183, i32 131072, i64 3, !dbg !42
  %1184 = ptrtoint ptr addrspace(3) %1174 to i32, !dbg !42
  %dma.buffer.lds637 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1184), !dbg !42
  %1185 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds637, i32 %dma.buffer.offset635, <4 x i32> %dma.buffer.rsrc636) #10, !dbg !42
  %1186 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !42
  %1187 = zext i32 %1186 to i64, !dbg !42
  %1188 = inttoptr i64 %1187 to ptr addrspace(3), !dbg !42
  %address.local391 = getelementptr [2 x i8], ptr addrspace(1) %1171, i64 %address.local388, !dbg !42
  %1189 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1190 = ptrtoint ptr addrspace(1) %address.local391 to i64, !dbg !42
  %1191 = sub i64 %1190, %1189, !dbg !42
  %dma.buffer.offset632 = trunc i64 %1191 to i32, !dbg !42
  %1192 = trunc i64 %1189 to i32, !dbg !42
  %1193 = insertelement <4 x i32> poison, i32 %1192, i64 0, !dbg !42
  %1194 = lshr i64 %1189, 32, !dbg !42
  %1195 = trunc i64 %1194 to i32, !dbg !42
  %1196 = insertelement <4 x i32> %1193, i32 %1195, i64 1, !dbg !42
  %1197 = insertelement <4 x i32> %1196, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc633 = insertelement <4 x i32> %1197, i32 131072, i64 3, !dbg !42
  %1198 = ptrtoint ptr addrspace(3) %1188 to i32, !dbg !42
  %dma.buffer.lds634 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1198), !dbg !42
  %1199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds634, i32 %dma.buffer.offset632, <4 x i32> %dma.buffer.rsrc633) #10, !dbg !42
  %1200 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1201 = extractvalue [2 x <8 x float>] %29, 0, !dbg !53
  %1202 = shufflevector <8 x float> %1201, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1200, <4 x float> %1202) #9, !dbg !39
  %1204 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1204, <4 x float> %1203) #9, !dbg !39
  %1206 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1206, <4 x float> %1205) #9, !dbg !39
  %1208 = shufflevector <8 x half> %1161, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1208, <4 x float> %1207) #9, !dbg !39
  %1210 = shufflevector <8 x half> %1163, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1210, <4 x float> %1209) #9, !dbg !39
  %1212 = shufflevector <8 x half> %1165, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1212, <4 x float> %1211) #9, !dbg !39
  %1214 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1214, <4 x float> %1213) #9, !dbg !39
  %1216 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1216, <4 x float> %1215) #9, !dbg !39
  %1218 = shufflevector <8 x half> %1155, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1219 = shufflevector <8 x float> %1201, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1218, <4 x float> %1219) #9, !dbg !39
  %1221 = shufflevector <8 x half> %1157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1221, <4 x float> %1220) #9, !dbg !39
  %1223 = shufflevector <8 x half> %1159, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1223, <4 x float> %1222) #9, !dbg !39
  %1225 = shufflevector <8 x half> %1161, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1225, <4 x float> %1224) #9, !dbg !39
  %1227 = shufflevector <8 x half> %1163, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1227, <4 x float> %1226) #9, !dbg !39
  %1229 = shufflevector <8 x half> %1165, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1229, <4 x float> %1228) #9, !dbg !39
  %1231 = shufflevector <8 x half> %1167, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1231, <4 x float> %1230) #9, !dbg !39
  %1233 = shufflevector <8 x half> %1169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1233, <4 x float> %1232) #9, !dbg !39
  %1235 = extractvalue [2 x <8 x float>] %29, 1, !dbg !53
  %1236 = shufflevector <8 x float> %1235, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1200, <4 x float> %1236) #9, !dbg !39
  %1238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1204, <4 x float> %1237) #9, !dbg !39
  %1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1206, <4 x float> %1238) #9, !dbg !39
  %1240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1208, <4 x float> %1239) #9, !dbg !39
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1210, <4 x float> %1240) #9, !dbg !39
  %1242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1212, <4 x float> %1241) #9, !dbg !39
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1214, <4 x float> %1242) #9, !dbg !39
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1216, <4 x float> %1243) #9, !dbg !39
  %1245 = shufflevector <8 x float> %1235, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1218, <4 x float> %1245) #9, !dbg !39
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1221, <4 x float> %1246) #9, !dbg !39
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1223, <4 x float> %1247) #9, !dbg !39
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1225, <4 x float> %1248) #9, !dbg !39
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1227, <4 x float> %1249) #9, !dbg !39
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1229, <4 x float> %1250) #9, !dbg !39
  %1252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1231, <4 x float> %1251) #9, !dbg !39
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1233, <4 x float> %1252) #9, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1254 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1255 = bitcast <4 x i32> %1254 to <8 x half>, !dbg !39
  %1256 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !39
  %1258 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1259 = bitcast <4 x i32> %1258 to <8 x half>, !dbg !39
  %1260 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1261 = bitcast <4 x i32> %1260 to <8 x half>, !dbg !39
  %1262 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1263 = bitcast <4 x i32> %1262 to <8 x half>, !dbg !39
  %1264 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1265 = bitcast <4 x i32> %1264 to <8 x half>, !dbg !39
  %1266 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1267 = bitcast <4 x i32> %1266 to <8 x half>, !dbg !39
  %1268 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1269 = bitcast <4 x i32> %1268 to <8 x half>, !dbg !39
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1270 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %1069, !dbg !42
  %1271 = getelementptr i8, ptr addrspace(1) %1270, i64 192, !dbg !42
  %1272 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local371), !dbg !42
  %1273 = zext i32 %1272 to i64, !dbg !42
  %1274 = inttoptr i64 %1273 to ptr addrspace(3), !dbg !42
  %address.local392 = getelementptr [2 x i8], ptr addrspace(1) %1271, i64 %address.local376, !dbg !42
  %1275 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1276 = ptrtoint ptr addrspace(1) %address.local392 to i64, !dbg !42
  %1277 = sub i64 %1276, %1275, !dbg !42
  %dma.buffer.offset629 = trunc i64 %1277 to i32, !dbg !42
  %1278 = trunc i64 %1275 to i32, !dbg !42
  %1279 = insertelement <4 x i32> poison, i32 %1278, i64 0, !dbg !42
  %1280 = lshr i64 %1275, 32, !dbg !42
  %1281 = trunc i64 %1280 to i32, !dbg !42
  %1282 = insertelement <4 x i32> %1279, i32 %1281, i64 1, !dbg !42
  %1283 = insertelement <4 x i32> %1282, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc630 = insertelement <4 x i32> %1283, i32 131072, i64 3, !dbg !42
  %1284 = ptrtoint ptr addrspace(3) %1274 to i32, !dbg !42
  %dma.buffer.lds631 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1284), !dbg !42
  %1285 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds631, i32 %dma.buffer.offset629, <4 x i32> %dma.buffer.rsrc630) #10, !dbg !42
  %1286 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local383), !dbg !42
  %1287 = zext i32 %1286 to i64, !dbg !42
  %1288 = inttoptr i64 %1287 to ptr addrspace(3), !dbg !42
  %address.local393 = getelementptr [2 x i8], ptr addrspace(1) %1271, i64 %address.local388, !dbg !42
  %1289 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !42
  %1290 = ptrtoint ptr addrspace(1) %address.local393 to i64, !dbg !42
  %1291 = sub i64 %1290, %1289, !dbg !42
  %dma.buffer.offset626 = trunc i64 %1291 to i32, !dbg !42
  %1292 = trunc i64 %1289 to i32, !dbg !42
  %1293 = insertelement <4 x i32> poison, i32 %1292, i64 0, !dbg !42
  %1294 = lshr i64 %1289, 32, !dbg !42
  %1295 = trunc i64 %1294 to i32, !dbg !42
  %1296 = insertelement <4 x i32> %1293, i32 %1295, i64 1, !dbg !42
  %1297 = insertelement <4 x i32> %1296, i32 33554432, i64 2, !dbg !42
  %dma.buffer.rsrc627 = insertelement <4 x i32> %1297, i32 131072, i64 3, !dbg !42
  %1298 = ptrtoint ptr addrspace(3) %1288 to i32, !dbg !42
  %dma.buffer.lds628 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1298), !dbg !42
  %1299 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds628, i32 %dma.buffer.offset626, <4 x i32> %dma.buffer.rsrc627) #10, !dbg !42
  %1300 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1301 = extractvalue [2 x <8 x float>] %30, 0, !dbg !53
  %1302 = shufflevector <8 x float> %1301, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1300, <4 x float> %1302) #9, !dbg !39
  %1304 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1304, <4 x float> %1303) #9, !dbg !39
  %1306 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1306, <4 x float> %1305) #9, !dbg !39
  %1308 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1308, <4 x float> %1307) #9, !dbg !39
  %1310 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1310, <4 x float> %1309) #9, !dbg !39
  %1312 = shufflevector <8 x half> %1265, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1312, <4 x float> %1311) #9, !dbg !39
  %1314 = shufflevector <8 x half> %1267, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1314, <4 x float> %1313) #9, !dbg !39
  %1316 = shufflevector <8 x half> %1269, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1316, <4 x float> %1315) #9, !dbg !39
  %1318 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1319 = shufflevector <8 x float> %1301, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1318, <4 x float> %1319) #9, !dbg !39
  %1321 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1321, <4 x float> %1320) #9, !dbg !39
  %1323 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1323, <4 x float> %1322) #9, !dbg !39
  %1325 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1325, <4 x float> %1324) #9, !dbg !39
  %1327 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1327, <4 x float> %1326) #9, !dbg !39
  %1329 = shufflevector <8 x half> %1265, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1329, <4 x float> %1328) #9, !dbg !39
  %1331 = shufflevector <8 x half> %1267, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1331, <4 x float> %1330) #9, !dbg !39
  %1333 = shufflevector <8 x half> %1269, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1333, <4 x float> %1332) #9, !dbg !39
  %1335 = extractvalue [2 x <8 x float>] %30, 1, !dbg !53
  %1336 = shufflevector <8 x float> %1335, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1300, <4 x float> %1336) #9, !dbg !39
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1304, <4 x float> %1337) #9, !dbg !39
  %1339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1306, <4 x float> %1338) #9, !dbg !39
  %1340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1308, <4 x float> %1339) #9, !dbg !39
  %1341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1310, <4 x float> %1340) #9, !dbg !39
  %1342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1312, <4 x float> %1341) #9, !dbg !39
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1314, <4 x float> %1342) #9, !dbg !39
  %1344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1316, <4 x float> %1343) #9, !dbg !39
  %1345 = shufflevector <8 x float> %1335, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1318, <4 x float> %1345) #9, !dbg !39
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1321, <4 x float> %1346) #9, !dbg !39
  %1348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1323, <4 x float> %1347) #9, !dbg !39
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1325, <4 x float> %1348) #9, !dbg !39
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1327, <4 x float> %1349) #9, !dbg !39
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1329, <4 x float> %1350) #9, !dbg !39
  %1352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1331, <4 x float> %1351) #9, !dbg !39
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1333, <4 x float> %1352) #9, !dbg !39
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !42
  fence syncscope("workgroup") release, !dbg !42
  call void @llvm.amdgcn.s.barrier(), !dbg !42
  fence syncscope("workgroup") acquire, !dbg !42
  %1354 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local331, i32 0) #0, !dbg !39
  %1355 = bitcast <4 x i32> %1354 to <8 x half>, !dbg !39
  %1356 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local344, i32 0) #0, !dbg !39
  %1357 = bitcast <4 x i32> %1356 to <8 x half>, !dbg !39
  %1358 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local348, i32 0) #0, !dbg !39
  %1359 = bitcast <4 x i32> %1358 to <8 x half>, !dbg !39
  %1360 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local352, i32 0) #0, !dbg !39
  %1361 = bitcast <4 x i32> %1360 to <8 x half>, !dbg !39
  %1362 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local354, i32 0) #0, !dbg !39
  %1363 = bitcast <4 x i32> %1362 to <8 x half>, !dbg !39
  %1364 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local358, i32 0) #0, !dbg !39
  %1365 = bitcast <4 x i32> %1364 to <8 x half>, !dbg !39
  %1366 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !39
  %1367 = bitcast <4 x i32> %1366 to <8 x half>, !dbg !39
  %1368 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !39
  %1369 = bitcast <4 x i32> %1368 to <8 x half>, !dbg !39
  %1370 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1371 = extractvalue [2 x <8 x float>] %31, 0, !dbg !53
  %1372 = shufflevector <8 x float> %1371, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1370, <4 x float> %1372) #9, !dbg !39
  %1374 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1374, <4 x float> %1373) #9, !dbg !39
  %1376 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1376, <4 x float> %1375) #9, !dbg !39
  %1378 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1378, <4 x float> %1377) #9, !dbg !39
  %1380 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1380, <4 x float> %1379) #9, !dbg !39
  %1382 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1383 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1382, <4 x float> %1381) #9, !dbg !39
  %1384 = shufflevector <8 x half> %1367, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1384, <4 x float> %1383) #9, !dbg !39
  %1386 = shufflevector <8 x half> %1369, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %1387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1386, <4 x float> %1385) #9, !dbg !39
  %1388 = shufflevector <8 x half> %1355, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1389 = shufflevector <8 x float> %1371, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1388, <4 x float> %1389) #9, !dbg !39
  %1391 = shufflevector <8 x half> %1357, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %1391, <4 x float> %1390) #9, !dbg !39
  %1393 = shufflevector <8 x half> %1359, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %1393, <4 x float> %1392) #9, !dbg !39
  %1395 = shufflevector <8 x half> %1361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %1395, <4 x float> %1394) #9, !dbg !39
  %1397 = shufflevector <8 x half> %1363, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %1397, <4 x float> %1396) #9, !dbg !39
  %1399 = shufflevector <8 x half> %1365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %1399, <4 x float> %1398) #9, !dbg !39
  %1401 = shufflevector <8 x half> %1367, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %1401, <4 x float> %1400) #9, !dbg !39
  %1403 = shufflevector <8 x half> %1369, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %1404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %1403, <4 x float> %1402) #9, !dbg !39
  %1405 = extractvalue [2 x <8 x float>] %31, 1, !dbg !53
  %1406 = shufflevector <8 x float> %1405, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !53
  %1407 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1370, <4 x float> %1406) #9, !dbg !39
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1374, <4 x float> %1407) #9, !dbg !39
  %1409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1376, <4 x float> %1408) #9, !dbg !39
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1378, <4 x float> %1409) #9, !dbg !39
  %1411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1380, <4 x float> %1410) #9, !dbg !39
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1382, <4 x float> %1411) #9, !dbg !39
  %1413 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1384, <4 x float> %1412) #9, !dbg !39
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1386, <4 x float> %1413) #9, !dbg !39
  %1415 = shufflevector <8 x float> %1405, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !53
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %1388, <4 x float> %1415) #9, !dbg !39
  %1417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %1391, <4 x float> %1416) #9, !dbg !39
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %1393, <4 x float> %1417) #9, !dbg !39
  %1419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %1395, <4 x float> %1418) #9, !dbg !39
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %1397, <4 x float> %1419) #9, !dbg !39
  %1421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %1399, <4 x float> %1420) #9, !dbg !39
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed130, <4 x half> %1401, <4 x float> %1421) #9, !dbg !39
  %1423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed132, <4 x half> %1403, <4 x float> %1422) #9, !dbg !39
  %1424 = shufflevector <4 x float> %1117, <4 x float> %1134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1425 = insertvalue [2 x <8 x float>] poison, <8 x float> %1424, 0, !dbg !40
  %1426 = shufflevector <4 x float> %1144, <4 x float> %1153, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1427 = insertvalue [2 x <8 x float>] %1425, <8 x float> %1426, 1, !dbg !40
  %1428 = shufflevector <4 x float> %1217, <4 x float> %1234, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1429 = insertvalue [2 x <8 x float>] poison, <8 x float> %1428, 0, !dbg !40
  %1430 = shufflevector <4 x float> %1244, <4 x float> %1253, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1431 = insertvalue [2 x <8 x float>] %1429, <8 x float> %1430, 1, !dbg !40
  %1432 = shufflevector <4 x float> %1317, <4 x float> %1334, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1433 = insertvalue [2 x <8 x float>] poison, <8 x float> %1432, 0, !dbg !40
  %1434 = shufflevector <4 x float> %1344, <4 x float> %1353, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1435 = insertvalue [2 x <8 x float>] %1433, <8 x float> %1434, 1, !dbg !40
  %1436 = shufflevector <4 x float> %1387, <4 x float> %1404, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1437 = insertvalue [2 x <8 x float>] poison, <8 x float> %1436, 0, !dbg !40
  %1438 = shufflevector <4 x float> %1414, <4 x float> %1423, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !40
  %1439 = insertvalue [2 x <8 x float>] %1437, <8 x float> %1438, 1, !dbg !40
  fence syncscope("workgroup") release, !dbg !40
  call void @llvm.amdgcn.s.barrier(), !dbg !40
  fence syncscope("workgroup") acquire, !dbg !40
  %1440 = add nuw nsw i64 %26, 1, !dbg !40
  br label %25, !dbg !40

1441:                                             ; preds = %25
  %1442 = and i32 %19, 16, !dbg !40
  %.not2 = icmp eq i32 %1442, 0, !dbg !40
  %1443 = and i32 %19, 32, !dbg !40
  %.not3 = icmp eq i32 %1443, 0, !dbg !40
  %1444 = extractvalue [2 x <8 x float>] %31, 0, !dbg !40
  %1445 = extractelement <8 x float> %1444, i64 0, !dbg !40
  %1446 = extractelement <8 x float> %1444, i64 1, !dbg !40
  %1447 = extractelement <8 x float> %1444, i64 2, !dbg !40
  %1448 = extractelement <8 x float> %1444, i64 3, !dbg !40
  %1449 = select i1 %.not2, float %1446, float %1445, !dbg !40
  %1450 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !40
  %1451 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1450), !dbg !40
  %1452 = add nuw nsw i32 %1451, 64, !dbg !40
  %1453 = xor i32 %1451, 16, !dbg !40
  %1454 = and i32 %1452, 192, !dbg !40
  %1455 = icmp samesign ult i32 %1453, %1454, !dbg !40
  %1456 = select i1 %1455, i32 %1453, i32 %1451, !dbg !40
  %1457 = shl nuw nsw i32 %1456, 2, !dbg !40
  %1458 = bitcast float %1449 to i32, !dbg !40
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1458), !dbg !40
  %1460 = bitcast i32 %1459 to float, !dbg !40
  %1461 = select i1 %.not2, float %1445, float %1460, !dbg !40
  %1462 = select i1 %.not2, float %1460, float %1446, !dbg !40
  %1463 = select i1 %.not2, float %1448, float %1447, !dbg !40
  %1464 = bitcast float %1463 to i32, !dbg !40
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1464), !dbg !40
  %1466 = bitcast i32 %1465 to float, !dbg !40
  %1467 = select i1 %.not2, float %1447, float %1466, !dbg !40
  %1468 = select i1 %.not2, float %1466, float %1448, !dbg !40
  %1469 = select i1 %.not3, float %1467, float %1461, !dbg !40
  %1470 = xor i32 %1451, 32, !dbg !40
  %1471 = icmp samesign ult i32 %1470, %1454, !dbg !40
  %1472 = select i1 %1471, i32 %1470, i32 %1451, !dbg !40
  %1473 = shl nuw nsw i32 %1472, 2, !dbg !40
  %1474 = bitcast float %1469 to i32, !dbg !40
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1474), !dbg !40
  %1476 = bitcast i32 %1475 to float, !dbg !40
  %1477 = select i1 %.not3, float %1461, float %1476, !dbg !40
  %1478 = select i1 %.not3, float %1476, float %1467, !dbg !40
  %1479 = select i1 %.not3, float %1468, float %1462, !dbg !40
  %1480 = bitcast float %1479 to i32, !dbg !40
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1480), !dbg !40
  %1482 = bitcast i32 %1481 to float, !dbg !40
  %1483 = select i1 %.not3, float %1462, float %1482, !dbg !40
  %1484 = select i1 %.not3, float %1482, float %1468, !dbg !40
  %1485 = extractvalue [2 x <8 x float>] %30, 0, !dbg !40
  %1486 = extractelement <8 x float> %1485, i64 0, !dbg !40
  %1487 = extractelement <8 x float> %1485, i64 1, !dbg !40
  %1488 = extractelement <8 x float> %1485, i64 2, !dbg !40
  %1489 = extractelement <8 x float> %1485, i64 3, !dbg !40
  %1490 = select i1 %.not2, float %1487, float %1486, !dbg !40
  %1491 = bitcast float %1490 to i32, !dbg !40
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1491), !dbg !40
  %1493 = bitcast i32 %1492 to float, !dbg !40
  %1494 = select i1 %.not2, float %1486, float %1493, !dbg !40
  %1495 = select i1 %.not2, float %1493, float %1487, !dbg !40
  %1496 = select i1 %.not2, float %1489, float %1488, !dbg !40
  %1497 = bitcast float %1496 to i32, !dbg !40
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1497), !dbg !40
  %1499 = bitcast i32 %1498 to float, !dbg !40
  %1500 = select i1 %.not2, float %1488, float %1499, !dbg !40
  %1501 = select i1 %.not2, float %1499, float %1489, !dbg !40
  %1502 = select i1 %.not3, float %1500, float %1494, !dbg !40
  %1503 = bitcast float %1502 to i32, !dbg !40
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1503), !dbg !40
  %1505 = bitcast i32 %1504 to float, !dbg !40
  %1506 = select i1 %.not3, float %1494, float %1505, !dbg !40
  %1507 = select i1 %.not3, float %1505, float %1500, !dbg !40
  %1508 = select i1 %.not3, float %1501, float %1495, !dbg !40
  %1509 = bitcast float %1508 to i32, !dbg !40
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1509), !dbg !40
  %1511 = bitcast i32 %1510 to float, !dbg !40
  %1512 = select i1 %.not3, float %1495, float %1511, !dbg !40
  %1513 = select i1 %.not3, float %1511, float %1501, !dbg !40
  %1514 = extractvalue [2 x <8 x float>] %29, 0, !dbg !40
  %1515 = extractelement <8 x float> %1514, i64 0, !dbg !40
  %1516 = extractelement <8 x float> %1514, i64 1, !dbg !40
  %1517 = extractelement <8 x float> %1514, i64 2, !dbg !40
  %1518 = extractelement <8 x float> %1514, i64 3, !dbg !40
  %1519 = select i1 %.not2, float %1516, float %1515, !dbg !40
  %1520 = bitcast float %1519 to i32, !dbg !40
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1520), !dbg !40
  %1522 = bitcast i32 %1521 to float, !dbg !40
  %1523 = select i1 %.not2, float %1515, float %1522, !dbg !40
  %1524 = select i1 %.not2, float %1522, float %1516, !dbg !40
  %1525 = select i1 %.not2, float %1518, float %1517, !dbg !40
  %1526 = bitcast float %1525 to i32, !dbg !40
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1526), !dbg !40
  %1528 = bitcast i32 %1527 to float, !dbg !40
  %1529 = select i1 %.not2, float %1517, float %1528, !dbg !40
  %1530 = select i1 %.not2, float %1528, float %1518, !dbg !40
  %1531 = select i1 %.not3, float %1529, float %1523, !dbg !40
  %1532 = bitcast float %1531 to i32, !dbg !40
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1532), !dbg !40
  %1534 = bitcast i32 %1533 to float, !dbg !40
  %1535 = select i1 %.not3, float %1523, float %1534, !dbg !40
  %1536 = select i1 %.not3, float %1534, float %1529, !dbg !40
  %1537 = select i1 %.not3, float %1530, float %1524, !dbg !40
  %1538 = bitcast float %1537 to i32, !dbg !40
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1538), !dbg !40
  %1540 = bitcast i32 %1539 to float, !dbg !40
  %1541 = select i1 %.not3, float %1524, float %1540, !dbg !40
  %1542 = select i1 %.not3, float %1540, float %1530, !dbg !40
  %1543 = extractvalue [2 x <8 x float>] %27, 0, !dbg !40
  %1544 = extractelement <8 x float> %1543, i64 0, !dbg !40
  %1545 = extractelement <8 x float> %1543, i64 1, !dbg !40
  %1546 = extractelement <8 x float> %1543, i64 2, !dbg !40
  %1547 = extractelement <8 x float> %1543, i64 3, !dbg !40
  %1548 = select i1 %.not2, float %1545, float %1544, !dbg !40
  %1549 = bitcast float %1548 to i32, !dbg !40
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1549), !dbg !40
  %1551 = bitcast i32 %1550 to float, !dbg !40
  %1552 = select i1 %.not2, float %1544, float %1551, !dbg !40
  %1553 = select i1 %.not2, float %1551, float %1545, !dbg !40
  %1554 = select i1 %.not2, float %1547, float %1546, !dbg !40
  %1555 = bitcast float %1554 to i32, !dbg !40
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1555), !dbg !40
  %1557 = bitcast i32 %1556 to float, !dbg !40
  %1558 = select i1 %.not2, float %1546, float %1557, !dbg !40
  %1559 = select i1 %.not2, float %1557, float %1547, !dbg !40
  %1560 = select i1 %.not3, float %1558, float %1552, !dbg !40
  %1561 = bitcast float %1560 to i32, !dbg !40
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1561), !dbg !40
  %1563 = select i1 %.not3, float %1559, float %1553, !dbg !40
  %1564 = bitcast float %1563 to i32, !dbg !40
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1564), !dbg !40
  %1566 = extractvalue [2 x <1 x float>] %28, 0, !dbg !43
  %1567 = shufflevector <1 x float> %1566, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %1568 = extractelement <8 x float> %1444, i64 4, !dbg !40
  %1569 = extractelement <8 x float> %1444, i64 5, !dbg !40
  %1570 = extractelement <8 x float> %1444, i64 6, !dbg !40
  %1571 = extractelement <8 x float> %1444, i64 7, !dbg !40
  %1572 = select i1 %.not2, float %1569, float %1568, !dbg !40
  %1573 = bitcast float %1572 to i32, !dbg !40
  %1574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1573), !dbg !40
  %1575 = bitcast i32 %1574 to float, !dbg !40
  %1576 = select i1 %.not2, float %1568, float %1575, !dbg !40
  %1577 = select i1 %.not2, float %1575, float %1569, !dbg !40
  %1578 = select i1 %.not2, float %1571, float %1570, !dbg !40
  %1579 = bitcast float %1578 to i32, !dbg !40
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1579), !dbg !40
  %1581 = bitcast i32 %1580 to float, !dbg !40
  %1582 = select i1 %.not2, float %1570, float %1581, !dbg !40
  %1583 = select i1 %.not2, float %1581, float %1571, !dbg !40
  %1584 = select i1 %.not3, float %1582, float %1576, !dbg !40
  %1585 = bitcast float %1584 to i32, !dbg !40
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1585), !dbg !40
  %1587 = bitcast i32 %1586 to float, !dbg !40
  %1588 = select i1 %.not3, float %1576, float %1587, !dbg !40
  %1589 = select i1 %.not3, float %1587, float %1582, !dbg !40
  %1590 = select i1 %.not3, float %1583, float %1577, !dbg !40
  %1591 = bitcast float %1590 to i32, !dbg !40
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1591), !dbg !40
  %1593 = bitcast i32 %1592 to float, !dbg !40
  %1594 = select i1 %.not3, float %1577, float %1593, !dbg !40
  %1595 = select i1 %.not3, float %1593, float %1583, !dbg !40
  %1596 = extractelement <8 x float> %1485, i64 4, !dbg !40
  %1597 = extractelement <8 x float> %1485, i64 5, !dbg !40
  %1598 = extractelement <8 x float> %1485, i64 6, !dbg !40
  %1599 = extractelement <8 x float> %1485, i64 7, !dbg !40
  %1600 = select i1 %.not2, float %1597, float %1596, !dbg !40
  %1601 = bitcast float %1600 to i32, !dbg !40
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1601), !dbg !40
  %1603 = bitcast i32 %1602 to float, !dbg !40
  %1604 = select i1 %.not2, float %1596, float %1603, !dbg !40
  %1605 = select i1 %.not2, float %1603, float %1597, !dbg !40
  %1606 = select i1 %.not2, float %1599, float %1598, !dbg !40
  %1607 = bitcast float %1606 to i32, !dbg !40
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1607), !dbg !40
  %1609 = bitcast i32 %1608 to float, !dbg !40
  %1610 = select i1 %.not2, float %1598, float %1609, !dbg !40
  %1611 = select i1 %.not2, float %1609, float %1599, !dbg !40
  %1612 = select i1 %.not3, float %1610, float %1604, !dbg !40
  %1613 = bitcast float %1612 to i32, !dbg !40
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1613), !dbg !40
  %1615 = bitcast i32 %1614 to float, !dbg !40
  %1616 = select i1 %.not3, float %1604, float %1615, !dbg !40
  %1617 = select i1 %.not3, float %1615, float %1610, !dbg !40
  %1618 = select i1 %.not3, float %1611, float %1605, !dbg !40
  %1619 = bitcast float %1618 to i32, !dbg !40
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1619), !dbg !40
  %1621 = bitcast i32 %1620 to float, !dbg !40
  %1622 = select i1 %.not3, float %1605, float %1621, !dbg !40
  %1623 = select i1 %.not3, float %1621, float %1611, !dbg !40
  %1624 = extractelement <8 x float> %1514, i64 4, !dbg !40
  %1625 = extractelement <8 x float> %1514, i64 5, !dbg !40
  %1626 = extractelement <8 x float> %1514, i64 6, !dbg !40
  %1627 = extractelement <8 x float> %1514, i64 7, !dbg !40
  %1628 = select i1 %.not2, float %1625, float %1624, !dbg !40
  %1629 = bitcast float %1628 to i32, !dbg !40
  %1630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1629), !dbg !40
  %1631 = bitcast i32 %1630 to float, !dbg !40
  %1632 = select i1 %.not2, float %1624, float %1631, !dbg !40
  %1633 = select i1 %.not2, float %1631, float %1625, !dbg !40
  %1634 = select i1 %.not2, float %1627, float %1626, !dbg !40
  %1635 = bitcast float %1634 to i32, !dbg !40
  %1636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1635), !dbg !40
  %1637 = bitcast i32 %1636 to float, !dbg !40
  %1638 = select i1 %.not2, float %1626, float %1637, !dbg !40
  %1639 = select i1 %.not2, float %1637, float %1627, !dbg !40
  %1640 = select i1 %.not3, float %1638, float %1632, !dbg !40
  %1641 = bitcast float %1640 to i32, !dbg !40
  %1642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1641), !dbg !40
  %1643 = bitcast i32 %1642 to float, !dbg !40
  %1644 = select i1 %.not3, float %1632, float %1643, !dbg !40
  %1645 = select i1 %.not3, float %1643, float %1638, !dbg !40
  %1646 = select i1 %.not3, float %1639, float %1633, !dbg !40
  %1647 = bitcast float %1646 to i32, !dbg !40
  %1648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1647), !dbg !40
  %1649 = bitcast i32 %1648 to float, !dbg !40
  %1650 = select i1 %.not3, float %1633, float %1649, !dbg !40
  %1651 = select i1 %.not3, float %1649, float %1639, !dbg !40
  %1652 = extractelement <8 x float> %1543, i64 4, !dbg !40
  %1653 = extractelement <8 x float> %1543, i64 5, !dbg !40
  %1654 = extractelement <8 x float> %1543, i64 6, !dbg !40
  %1655 = extractelement <8 x float> %1543, i64 7, !dbg !40
  %1656 = select i1 %.not2, float %1653, float %1652, !dbg !40
  %1657 = bitcast float %1656 to i32, !dbg !40
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1657), !dbg !40
  %1659 = bitcast i32 %1658 to float, !dbg !40
  %1660 = select i1 %.not2, float %1652, float %1659, !dbg !40
  %1661 = select i1 %.not2, float %1659, float %1653, !dbg !40
  %1662 = select i1 %.not2, float %1655, float %1654, !dbg !40
  %1663 = bitcast float %1662 to i32, !dbg !40
  %1664 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1663), !dbg !40
  %1665 = bitcast i32 %1664 to float, !dbg !40
  %1666 = select i1 %.not2, float %1654, float %1665, !dbg !40
  %1667 = select i1 %.not2, float %1665, float %1655, !dbg !40
  %1668 = select i1 %.not3, float %1666, float %1660, !dbg !40
  %1669 = bitcast float %1668 to i32, !dbg !40
  %1670 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1669), !dbg !40
  %1671 = select i1 %.not3, float %1667, float %1661, !dbg !40
  %1672 = bitcast float %1671 to i32, !dbg !40
  %1673 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1672), !dbg !40
  %1674 = extractvalue [2 x <8 x float>] %31, 1, !dbg !40
  %1675 = extractelement <8 x float> %1674, i64 0, !dbg !40
  %1676 = extractelement <8 x float> %1674, i64 1, !dbg !40
  %1677 = extractelement <8 x float> %1674, i64 2, !dbg !40
  %1678 = extractelement <8 x float> %1674, i64 3, !dbg !40
  %1679 = select i1 %.not2, float %1676, float %1675, !dbg !40
  %1680 = bitcast float %1679 to i32, !dbg !40
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1680), !dbg !40
  %1682 = bitcast i32 %1681 to float, !dbg !40
  %1683 = select i1 %.not2, float %1675, float %1682, !dbg !40
  %1684 = select i1 %.not2, float %1682, float %1676, !dbg !40
  %1685 = select i1 %.not2, float %1678, float %1677, !dbg !40
  %1686 = bitcast float %1685 to i32, !dbg !40
  %1687 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1686), !dbg !40
  %1688 = bitcast i32 %1687 to float, !dbg !40
  %1689 = select i1 %.not2, float %1677, float %1688, !dbg !40
  %1690 = select i1 %.not2, float %1688, float %1678, !dbg !40
  %1691 = select i1 %.not3, float %1689, float %1683, !dbg !40
  %1692 = bitcast float %1691 to i32, !dbg !40
  %1693 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1692), !dbg !40
  %1694 = bitcast i32 %1693 to float, !dbg !40
  %1695 = select i1 %.not3, float %1683, float %1694, !dbg !40
  %1696 = select i1 %.not3, float %1694, float %1689, !dbg !40
  %1697 = select i1 %.not3, float %1690, float %1684, !dbg !40
  %1698 = bitcast float %1697 to i32, !dbg !40
  %1699 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1698), !dbg !40
  %1700 = bitcast i32 %1699 to float, !dbg !40
  %1701 = select i1 %.not3, float %1684, float %1700, !dbg !40
  %1702 = select i1 %.not3, float %1700, float %1690, !dbg !40
  %1703 = extractvalue [2 x <8 x float>] %30, 1, !dbg !40
  %1704 = extractelement <8 x float> %1703, i64 0, !dbg !40
  %1705 = extractelement <8 x float> %1703, i64 1, !dbg !40
  %1706 = extractelement <8 x float> %1703, i64 2, !dbg !40
  %1707 = extractelement <8 x float> %1703, i64 3, !dbg !40
  %1708 = select i1 %.not2, float %1705, float %1704, !dbg !40
  %1709 = bitcast float %1708 to i32, !dbg !40
  %1710 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1709), !dbg !40
  %1711 = bitcast i32 %1710 to float, !dbg !40
  %1712 = select i1 %.not2, float %1704, float %1711, !dbg !40
  %1713 = select i1 %.not2, float %1711, float %1705, !dbg !40
  %1714 = select i1 %.not2, float %1707, float %1706, !dbg !40
  %1715 = bitcast float %1714 to i32, !dbg !40
  %1716 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1715), !dbg !40
  %1717 = bitcast i32 %1716 to float, !dbg !40
  %1718 = select i1 %.not2, float %1706, float %1717, !dbg !40
  %1719 = select i1 %.not2, float %1717, float %1707, !dbg !40
  %1720 = select i1 %.not3, float %1718, float %1712, !dbg !40
  %1721 = bitcast float %1720 to i32, !dbg !40
  %1722 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1721), !dbg !40
  %1723 = bitcast i32 %1722 to float, !dbg !40
  %1724 = select i1 %.not3, float %1712, float %1723, !dbg !40
  %1725 = select i1 %.not3, float %1723, float %1718, !dbg !40
  %1726 = select i1 %.not3, float %1719, float %1713, !dbg !40
  %1727 = bitcast float %1726 to i32, !dbg !40
  %1728 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1727), !dbg !40
  %1729 = bitcast i32 %1728 to float, !dbg !40
  %1730 = select i1 %.not3, float %1713, float %1729, !dbg !40
  %1731 = select i1 %.not3, float %1729, float %1719, !dbg !40
  %1732 = extractvalue [2 x <8 x float>] %29, 1, !dbg !40
  %1733 = extractelement <8 x float> %1732, i64 0, !dbg !40
  %1734 = extractelement <8 x float> %1732, i64 1, !dbg !40
  %1735 = extractelement <8 x float> %1732, i64 2, !dbg !40
  %1736 = extractelement <8 x float> %1732, i64 3, !dbg !40
  %1737 = select i1 %.not2, float %1734, float %1733, !dbg !40
  %1738 = bitcast float %1737 to i32, !dbg !40
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1738), !dbg !40
  %1740 = bitcast i32 %1739 to float, !dbg !40
  %1741 = select i1 %.not2, float %1733, float %1740, !dbg !40
  %1742 = select i1 %.not2, float %1740, float %1734, !dbg !40
  %1743 = select i1 %.not2, float %1736, float %1735, !dbg !40
  %1744 = bitcast float %1743 to i32, !dbg !40
  %1745 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1744), !dbg !40
  %1746 = bitcast i32 %1745 to float, !dbg !40
  %1747 = select i1 %.not2, float %1735, float %1746, !dbg !40
  %1748 = select i1 %.not2, float %1746, float %1736, !dbg !40
  %1749 = select i1 %.not3, float %1747, float %1741, !dbg !40
  %1750 = bitcast float %1749 to i32, !dbg !40
  %1751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1750), !dbg !40
  %1752 = bitcast i32 %1751 to float, !dbg !40
  %1753 = select i1 %.not3, float %1741, float %1752, !dbg !40
  %1754 = select i1 %.not3, float %1752, float %1747, !dbg !40
  %1755 = select i1 %.not3, float %1748, float %1742, !dbg !40
  %1756 = bitcast float %1755 to i32, !dbg !40
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1756), !dbg !40
  %1758 = bitcast i32 %1757 to float, !dbg !40
  %1759 = select i1 %.not3, float %1742, float %1758, !dbg !40
  %1760 = select i1 %.not3, float %1758, float %1748, !dbg !40
  %1761 = extractvalue [2 x <8 x float>] %27, 1, !dbg !40
  %1762 = extractelement <8 x float> %1761, i64 0, !dbg !40
  %1763 = extractelement <8 x float> %1761, i64 1, !dbg !40
  %1764 = extractelement <8 x float> %1761, i64 2, !dbg !40
  %1765 = extractelement <8 x float> %1761, i64 3, !dbg !40
  %1766 = select i1 %.not2, float %1763, float %1762, !dbg !40
  %1767 = bitcast float %1766 to i32, !dbg !40
  %1768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1767), !dbg !40
  %1769 = bitcast i32 %1768 to float, !dbg !40
  %1770 = select i1 %.not2, float %1762, float %1769, !dbg !40
  %1771 = select i1 %.not2, float %1769, float %1763, !dbg !40
  %1772 = select i1 %.not2, float %1765, float %1764, !dbg !40
  %1773 = bitcast float %1772 to i32, !dbg !40
  %1774 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1773), !dbg !40
  %1775 = bitcast i32 %1774 to float, !dbg !40
  %1776 = select i1 %.not2, float %1764, float %1775, !dbg !40
  %1777 = select i1 %.not2, float %1775, float %1765, !dbg !40
  %1778 = select i1 %.not3, float %1776, float %1770, !dbg !40
  %1779 = bitcast float %1778 to i32, !dbg !40
  %1780 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1779), !dbg !40
  %1781 = select i1 %.not3, float %1777, float %1771, !dbg !40
  %1782 = bitcast float %1781 to i32, !dbg !40
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1782), !dbg !40
  %1784 = extractvalue [2 x <1 x float>] %28, 1, !dbg !43
  %1785 = shufflevector <1 x float> %1784, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %1786 = extractelement <8 x float> %1674, i64 4, !dbg !40
  %1787 = extractelement <8 x float> %1674, i64 5, !dbg !40
  %1788 = extractelement <8 x float> %1674, i64 6, !dbg !40
  %1789 = extractelement <8 x float> %1674, i64 7, !dbg !40
  %1790 = select i1 %.not2, float %1787, float %1786, !dbg !40
  %1791 = bitcast float %1790 to i32, !dbg !40
  %1792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1791), !dbg !40
  %1793 = bitcast i32 %1792 to float, !dbg !40
  %1794 = select i1 %.not2, float %1786, float %1793, !dbg !40
  %1795 = select i1 %.not2, float %1793, float %1787, !dbg !40
  %1796 = select i1 %.not2, float %1789, float %1788, !dbg !40
  %1797 = bitcast float %1796 to i32, !dbg !40
  %1798 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1797), !dbg !40
  %1799 = bitcast i32 %1798 to float, !dbg !40
  %1800 = select i1 %.not2, float %1788, float %1799, !dbg !40
  %1801 = select i1 %.not2, float %1799, float %1789, !dbg !40
  %1802 = select i1 %.not3, float %1800, float %1794, !dbg !40
  %1803 = bitcast float %1802 to i32, !dbg !40
  %1804 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1803), !dbg !40
  %1805 = bitcast i32 %1804 to float, !dbg !40
  %1806 = select i1 %.not3, float %1794, float %1805, !dbg !40
  %1807 = select i1 %.not3, float %1805, float %1800, !dbg !40
  %1808 = select i1 %.not3, float %1801, float %1795, !dbg !40
  %1809 = bitcast float %1808 to i32, !dbg !40
  %1810 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1809), !dbg !40
  %1811 = bitcast i32 %1810 to float, !dbg !40
  %1812 = select i1 %.not3, float %1795, float %1811, !dbg !40
  %1813 = select i1 %.not3, float %1811, float %1801, !dbg !40
  %1814 = extractelement <8 x float> %1703, i64 4, !dbg !40
  %1815 = extractelement <8 x float> %1703, i64 5, !dbg !40
  %1816 = extractelement <8 x float> %1703, i64 6, !dbg !40
  %1817 = extractelement <8 x float> %1703, i64 7, !dbg !40
  %1818 = select i1 %.not2, float %1815, float %1814, !dbg !40
  %1819 = bitcast float %1818 to i32, !dbg !40
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1819), !dbg !40
  %1821 = bitcast i32 %1820 to float, !dbg !40
  %1822 = select i1 %.not2, float %1814, float %1821, !dbg !40
  %1823 = select i1 %.not2, float %1821, float %1815, !dbg !40
  %1824 = select i1 %.not2, float %1817, float %1816, !dbg !40
  %1825 = bitcast float %1824 to i32, !dbg !40
  %1826 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1825), !dbg !40
  %1827 = bitcast i32 %1826 to float, !dbg !40
  %1828 = select i1 %.not2, float %1816, float %1827, !dbg !40
  %1829 = select i1 %.not2, float %1827, float %1817, !dbg !40
  %1830 = select i1 %.not3, float %1828, float %1822, !dbg !40
  %1831 = bitcast float %1830 to i32, !dbg !40
  %1832 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1831), !dbg !40
  %1833 = bitcast i32 %1832 to float, !dbg !40
  %1834 = select i1 %.not3, float %1822, float %1833, !dbg !40
  %1835 = select i1 %.not3, float %1833, float %1828, !dbg !40
  %1836 = select i1 %.not3, float %1829, float %1823, !dbg !40
  %1837 = bitcast float %1836 to i32, !dbg !40
  %1838 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1837), !dbg !40
  %1839 = bitcast i32 %1838 to float, !dbg !40
  %1840 = select i1 %.not3, float %1823, float %1839, !dbg !40
  %1841 = select i1 %.not3, float %1839, float %1829, !dbg !40
  %1842 = extractelement <8 x float> %1732, i64 4, !dbg !40
  %1843 = extractelement <8 x float> %1732, i64 5, !dbg !40
  %1844 = extractelement <8 x float> %1732, i64 6, !dbg !40
  %1845 = extractelement <8 x float> %1732, i64 7, !dbg !40
  %1846 = select i1 %.not2, float %1843, float %1842, !dbg !40
  %1847 = bitcast float %1846 to i32, !dbg !40
  %1848 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1847), !dbg !40
  %1849 = bitcast i32 %1848 to float, !dbg !40
  %1850 = select i1 %.not2, float %1842, float %1849, !dbg !40
  %1851 = select i1 %.not2, float %1849, float %1843, !dbg !40
  %1852 = select i1 %.not2, float %1845, float %1844, !dbg !40
  %1853 = bitcast float %1852 to i32, !dbg !40
  %1854 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1853), !dbg !40
  %1855 = bitcast i32 %1854 to float, !dbg !40
  %1856 = select i1 %.not2, float %1844, float %1855, !dbg !40
  %1857 = select i1 %.not2, float %1855, float %1845, !dbg !40
  %1858 = select i1 %.not3, float %1856, float %1850, !dbg !40
  %1859 = bitcast float %1858 to i32, !dbg !40
  %1860 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1859), !dbg !40
  %1861 = bitcast i32 %1860 to float, !dbg !40
  %1862 = select i1 %.not3, float %1850, float %1861, !dbg !40
  %1863 = select i1 %.not3, float %1861, float %1856, !dbg !40
  %1864 = select i1 %.not3, float %1857, float %1851, !dbg !40
  %1865 = bitcast float %1864 to i32, !dbg !40
  %1866 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1865), !dbg !40
  %1867 = bitcast i32 %1866 to float, !dbg !40
  %1868 = select i1 %.not3, float %1851, float %1867, !dbg !40
  %1869 = select i1 %.not3, float %1867, float %1857, !dbg !40
  %1870 = extractelement <8 x float> %1761, i64 4, !dbg !40
  %1871 = extractelement <8 x float> %1761, i64 5, !dbg !40
  %1872 = extractelement <8 x float> %1761, i64 6, !dbg !40
  %1873 = extractelement <8 x float> %1761, i64 7, !dbg !40
  %1874 = select i1 %.not2, float %1871, float %1870, !dbg !40
  %1875 = bitcast float %1874 to i32, !dbg !40
  %1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1875), !dbg !40
  %1877 = bitcast i32 %1876 to float, !dbg !40
  %1878 = select i1 %.not2, float %1870, float %1877, !dbg !40
  %1879 = select i1 %.not2, float %1877, float %1871, !dbg !40
  %1880 = select i1 %.not2, float %1873, float %1872, !dbg !40
  %1881 = bitcast float %1880 to i32, !dbg !40
  %1882 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1457, i32 %1881), !dbg !40
  %1883 = bitcast i32 %1882 to float, !dbg !40
  %1884 = select i1 %.not2, float %1872, float %1883, !dbg !40
  %1885 = select i1 %.not2, float %1883, float %1873, !dbg !40
  %1886 = select i1 %.not3, float %1884, float %1878, !dbg !40
  %1887 = bitcast float %1886 to i32, !dbg !40
  %1888 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1887), !dbg !40
  %1889 = select i1 %.not3, float %1885, float %1879, !dbg !40
  %1890 = bitcast float %1889 to i32, !dbg !40
  %1891 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1473, i32 %1890), !dbg !40
  br i1 true, label %1892, label %1956, !dbg !45

1892:                                             ; preds = %1441
  %1893 = bitcast i32 %1888 to float, !dbg !40
  %1894 = select i1 %.not3, float %1878, float %1893, !dbg !40
  %1895 = insertelement <4 x float> poison, float %1894, i64 0, !dbg !43
  %1896 = bitcast i32 %1891 to float, !dbg !40
  %1897 = select i1 %.not3, float %1879, float %1896, !dbg !40
  %1898 = insertelement <4 x float> %1895, float %1897, i64 1, !dbg !43
  %1899 = select i1 %.not3, float %1893, float %1884, !dbg !40
  %1900 = insertelement <4 x float> %1898, float %1899, i64 2, !dbg !43
  %1901 = select i1 %.not3, float %1896, float %1885, !dbg !40
  %1902 = insertelement <4 x float> %1900, float %1901, i64 3, !dbg !43
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1785, !dbg !43
  %normalized = fmul <4 x float> %1902, %row.reciprocal, !dbg !43
  %1903 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !44
  %1904 = extractelement <4 x half> %1903, i64 3, !dbg !44
  %1905 = extractelement <4 x half> %1903, i64 2, !dbg !44
  %1906 = extractelement <4 x half> %1903, i64 1, !dbg !44
  %1907 = extractelement <4 x half> %1903, i64 0, !dbg !44
  %1908 = bitcast i32 %1780 to float, !dbg !40
  %1909 = select i1 %.not3, float %1770, float %1908, !dbg !40
  %1910 = insertelement <4 x float> poison, float %1909, i64 0, !dbg !43
  %1911 = bitcast i32 %1783 to float, !dbg !40
  %1912 = select i1 %.not3, float %1771, float %1911, !dbg !40
  %1913 = insertelement <4 x float> %1910, float %1912, i64 1, !dbg !43
  %1914 = select i1 %.not3, float %1908, float %1776, !dbg !40
  %1915 = insertelement <4 x float> %1913, float %1914, i64 2, !dbg !43
  %1916 = select i1 %.not3, float %1911, float %1777, !dbg !40
  %1917 = insertelement <4 x float> %1915, float %1916, i64 3, !dbg !43
  %normalized75 = fmul <4 x float> %1917, %row.reciprocal, !dbg !43
  %1918 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !44
  %1919 = extractelement <4 x half> %1918, i64 3, !dbg !44
  %1920 = extractelement <4 x half> %1918, i64 2, !dbg !44
  %1921 = extractelement <4 x half> %1918, i64 1, !dbg !44
  %1922 = extractelement <4 x half> %1918, i64 0, !dbg !44
  %1923 = bitcast i32 %1670 to float, !dbg !40
  %1924 = select i1 %.not3, float %1660, float %1923, !dbg !40
  %1925 = insertelement <4 x float> poison, float %1924, i64 0, !dbg !43
  %1926 = bitcast i32 %1673 to float, !dbg !40
  %1927 = select i1 %.not3, float %1661, float %1926, !dbg !40
  %1928 = insertelement <4 x float> %1925, float %1927, i64 1, !dbg !43
  %1929 = select i1 %.not3, float %1923, float %1666, !dbg !40
  %1930 = insertelement <4 x float> %1928, float %1929, i64 2, !dbg !43
  %1931 = select i1 %.not3, float %1926, float %1667, !dbg !40
  %1932 = insertelement <4 x float> %1930, float %1931, i64 3, !dbg !43
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1567, !dbg !43
  %normalized77 = fmul <4 x float> %1932, %row.reciprocal76, !dbg !43
  %1933 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !44
  %1934 = extractelement <4 x half> %1933, i64 3, !dbg !44
  %1935 = extractelement <4 x half> %1933, i64 2, !dbg !44
  %1936 = extractelement <4 x half> %1933, i64 1, !dbg !44
  %1937 = extractelement <4 x half> %1933, i64 0, !dbg !44
  %1938 = bitcast i32 %1562 to float, !dbg !40
  %1939 = select i1 %.not3, float %1552, float %1938, !dbg !40
  %1940 = insertelement <4 x float> poison, float %1939, i64 0, !dbg !43
  %1941 = bitcast i32 %1565 to float, !dbg !40
  %1942 = select i1 %.not3, float %1553, float %1941, !dbg !40
  %1943 = insertelement <4 x float> %1940, float %1942, i64 1, !dbg !43
  %1944 = select i1 %.not3, float %1938, float %1558, !dbg !40
  %1945 = insertelement <4 x float> %1943, float %1944, i64 2, !dbg !43
  %1946 = select i1 %.not3, float %1941, float %1559, !dbg !40
  %1947 = insertelement <4 x float> %1945, float %1946, i64 3, !dbg !43
  %normalized78 = fmul <4 x float> %1947, %row.reciprocal76, !dbg !43
  %1948 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !44
  %1949 = extractelement <4 x half> %1948, i64 3, !dbg !44
  %1950 = extractelement <4 x half> %1948, i64 2, !dbg !44
  %1951 = extractelement <4 x half> %1948, i64 1, !dbg !44
  %1952 = extractelement <4 x half> %1948, i64 0, !dbg !44
  %1953 = add nsw i64 %8, %13, !dbg !45
  %1954 = shl nsw i64 %1953, 19, !dbg !45
  %1955 = and i64 %1954, 16252928, !dbg !45
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
  %address.local404 = add nuw nsw i64 %1955, %address.local403, !dbg !45
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local406 = lshr i64 %address.local395, 2, !dbg !37
  %address.local407 = and i64 %address.local406, 12, !dbg !37
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local405, i64 %address.local407, !dbg !45
  store half %1952, ptr addrspace(1) %address.local408, align 2, !dbg !45
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local410 = or disjoint i64 %address.local407, 1, !dbg !45
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !45
  store half %1951, ptr addrspace(1) %address.local411, align 2, !dbg !45
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local413 = or disjoint i64 %address.local407, 2, !dbg !45
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local413, !dbg !45
  store half %1950, ptr addrspace(1) %address.local414, align 2, !dbg !45
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local416 = or disjoint i64 %address.local407, 3, !dbg !45
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local416, !dbg !45
  store half %1949, ptr addrspace(1) %address.local417, align 2, !dbg !45
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local419 = or disjoint i64 %address.local407, 16, !dbg !45
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local419, !dbg !45
  store half %1937, ptr addrspace(1) %address.local420, align 2, !dbg !45
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local422 = or disjoint i64 %address.local407, 17, !dbg !45
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local422, !dbg !45
  store half %1936, ptr addrspace(1) %address.local423, align 2, !dbg !45
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local425 = or disjoint i64 %address.local407, 18, !dbg !45
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local425, !dbg !45
  store half %1935, ptr addrspace(1) %address.local426, align 2, !dbg !45
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local404, !dbg !45
  %address.local428 = or disjoint i64 %address.local407, 19, !dbg !45
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local427, i64 %address.local428, !dbg !45
  store half %1934, ptr addrspace(1) %address.local429, align 2, !dbg !45
  %address.local430 = shl nsw i64 %address.local402, 7, !dbg !45
  %address.local431 = add i64 %address.local430, 8192, !dbg !45
  %address.local432 = add nuw nsw i64 %1955, %address.local431, !dbg !45
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local433, i64 %address.local407, !dbg !45
  store half %1922, ptr addrspace(1) %address.local434, align 2, !dbg !45
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local410, !dbg !45
  store half %1921, ptr addrspace(1) %address.local436, align 2, !dbg !45
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local413, !dbg !45
  store half %1920, ptr addrspace(1) %address.local438, align 2, !dbg !45
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local416, !dbg !45
  store half %1919, ptr addrspace(1) %address.local440, align 2, !dbg !45
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local419, !dbg !45
  store half %1907, ptr addrspace(1) %address.local442, align 2, !dbg !45
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local422, !dbg !45
  store half %1906, ptr addrspace(1) %address.local444, align 2, !dbg !45
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local425, !dbg !45
  store half %1905, ptr addrspace(1) %address.local446, align 2, !dbg !45
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local432, !dbg !45
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local447, i64 %address.local428, !dbg !45
  store half %1904, ptr addrspace(1) %address.local448, align 2, !dbg !45
  br label %1956, !dbg !45

1956:                                             ; preds = %1892, %1441
  br i1 true, label %1957, label %1995, !dbg !45

1957:                                             ; preds = %1956
  %1958 = insertelement <4 x float> poison, float %1862, i64 0, !dbg !43
  %1959 = insertelement <4 x float> %1958, float %1868, i64 1, !dbg !43
  %1960 = insertelement <4 x float> %1959, float %1863, i64 2, !dbg !43
  %1961 = insertelement <4 x float> %1960, float %1869, i64 3, !dbg !43
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1785, !dbg !43
  %normalized80 = fmul <4 x float> %1961, %row.reciprocal79, !dbg !43
  %1962 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !44
  %1963 = extractelement <4 x half> %1962, i64 3, !dbg !44
  %1964 = extractelement <4 x half> %1962, i64 2, !dbg !44
  %1965 = extractelement <4 x half> %1962, i64 1, !dbg !44
  %1966 = extractelement <4 x half> %1962, i64 0, !dbg !44
  %1967 = insertelement <4 x float> poison, float %1753, i64 0, !dbg !43
  %1968 = insertelement <4 x float> %1967, float %1759, i64 1, !dbg !43
  %1969 = insertelement <4 x float> %1968, float %1754, i64 2, !dbg !43
  %1970 = insertelement <4 x float> %1969, float %1760, i64 3, !dbg !43
  %normalized81 = fmul <4 x float> %1970, %row.reciprocal79, !dbg !43
  %1971 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !44
  %1972 = extractelement <4 x half> %1971, i64 3, !dbg !44
  %1973 = extractelement <4 x half> %1971, i64 2, !dbg !44
  %1974 = extractelement <4 x half> %1971, i64 1, !dbg !44
  %1975 = extractelement <4 x half> %1971, i64 0, !dbg !44
  %1976 = insertelement <4 x float> poison, float %1644, i64 0, !dbg !43
  %1977 = insertelement <4 x float> %1976, float %1650, i64 1, !dbg !43
  %1978 = insertelement <4 x float> %1977, float %1645, i64 2, !dbg !43
  %1979 = insertelement <4 x float> %1978, float %1651, i64 3, !dbg !43
  %row.reciprocal82 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1567, !dbg !43
  %normalized83 = fmul <4 x float> %1979, %row.reciprocal82, !dbg !43
  %1980 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !44
  %1981 = extractelement <4 x half> %1980, i64 3, !dbg !44
  %1982 = extractelement <4 x half> %1980, i64 2, !dbg !44
  %1983 = extractelement <4 x half> %1980, i64 1, !dbg !44
  %1984 = extractelement <4 x half> %1980, i64 0, !dbg !44
  %1985 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !43
  %1986 = insertelement <4 x float> %1985, float %1541, i64 1, !dbg !43
  %1987 = insertelement <4 x float> %1986, float %1536, i64 2, !dbg !43
  %1988 = insertelement <4 x float> %1987, float %1542, i64 3, !dbg !43
  %normalized84 = fmul <4 x float> %1988, %row.reciprocal82, !dbg !43
  %1989 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !44
  %1990 = extractelement <4 x half> %1989, i64 3, !dbg !44
  %1991 = extractelement <4 x half> %1989, i64 2, !dbg !44
  %1992 = extractelement <4 x half> %1989, i64 1, !dbg !44
  %1993 = extractelement <4 x half> %1989, i64 0, !dbg !44
  %1994 = shl nuw nsw i64 %15, 19, !dbg !45
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
  %address.local459 = add nuw nsw i64 %1994, %address.local458, !dbg !45
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local461 = lshr i64 %address.local450, 2, !dbg !37
  %address.local462 = and i64 %address.local461, 12, !dbg !37
  %address.local463 = or disjoint i64 %address.local462, 32, !dbg !45
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local463, !dbg !45
  store half %1993, ptr addrspace(1) %address.local464, align 2, !dbg !45
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local466 = or disjoint i64 %address.local462, 33, !dbg !45
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local466, !dbg !45
  store half %1992, ptr addrspace(1) %address.local467, align 2, !dbg !45
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local469 = or disjoint i64 %address.local462, 34, !dbg !45
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local469, !dbg !45
  store half %1991, ptr addrspace(1) %address.local470, align 2, !dbg !45
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local472 = or disjoint i64 %address.local462, 35, !dbg !45
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local472, !dbg !45
  store half %1990, ptr addrspace(1) %address.local473, align 2, !dbg !45
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local475 = or disjoint i64 %address.local462, 48, !dbg !45
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local474, i64 %address.local475, !dbg !45
  store half %1984, ptr addrspace(1) %address.local476, align 2, !dbg !45
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local478 = or disjoint i64 %address.local462, 49, !dbg !45
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local478, !dbg !45
  store half %1983, ptr addrspace(1) %address.local479, align 2, !dbg !45
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local481 = or disjoint i64 %address.local462, 50, !dbg !45
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local480, i64 %address.local481, !dbg !45
  store half %1982, ptr addrspace(1) %address.local482, align 2, !dbg !45
  %address.local483 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local459, !dbg !45
  %address.local484 = or i64 %address.local461, 51, !dbg !45
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local483, i64 %address.local484, !dbg !45
  store half %1981, ptr addrspace(1) %address.local485, align 2, !dbg !45
  %address.local486 = shl nsw i64 %address.local457, 7, !dbg !45
  %address.local487 = add i64 %address.local486, 8192, !dbg !45
  %address.local488 = add nuw nsw i64 %1994, %address.local487, !dbg !45
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local489, i64 %address.local463, !dbg !45
  store half %1975, ptr addrspace(1) %address.local490, align 2, !dbg !45
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local466, !dbg !45
  store half %1974, ptr addrspace(1) %address.local492, align 2, !dbg !45
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local469, !dbg !45
  store half %1973, ptr addrspace(1) %address.local494, align 2, !dbg !45
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local472, !dbg !45
  store half %1972, ptr addrspace(1) %address.local496, align 2, !dbg !45
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local475, !dbg !45
  store half %1966, ptr addrspace(1) %address.local498, align 2, !dbg !45
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local478, !dbg !45
  store half %1965, ptr addrspace(1) %address.local500, align 2, !dbg !45
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local481, !dbg !45
  store half %1964, ptr addrspace(1) %address.local502, align 2, !dbg !45
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local488, !dbg !45
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local503, i64 %address.local484, !dbg !45
  store half %1963, ptr addrspace(1) %address.local504, align 2, !dbg !45
  br label %1995, !dbg !45

1995:                                             ; preds = %1957, %1956
  br i1 true, label %1996, label %2034, !dbg !45

1996:                                             ; preds = %1995
  %1997 = insertelement <4 x float> poison, float %1834, i64 0, !dbg !43
  %1998 = insertelement <4 x float> %1997, float %1840, i64 1, !dbg !43
  %1999 = insertelement <4 x float> %1998, float %1835, i64 2, !dbg !43
  %2000 = insertelement <4 x float> %1999, float %1841, i64 3, !dbg !43
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1785, !dbg !43
  %normalized86 = fmul <4 x float> %2000, %row.reciprocal85, !dbg !43
  %2001 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !44
  %2002 = extractelement <4 x half> %2001, i64 3, !dbg !44
  %2003 = extractelement <4 x half> %2001, i64 2, !dbg !44
  %2004 = extractelement <4 x half> %2001, i64 1, !dbg !44
  %2005 = extractelement <4 x half> %2001, i64 0, !dbg !44
  %2006 = insertelement <4 x float> poison, float %1724, i64 0, !dbg !43
  %2007 = insertelement <4 x float> %2006, float %1730, i64 1, !dbg !43
  %2008 = insertelement <4 x float> %2007, float %1725, i64 2, !dbg !43
  %2009 = insertelement <4 x float> %2008, float %1731, i64 3, !dbg !43
  %normalized87 = fmul <4 x float> %2009, %row.reciprocal85, !dbg !43
  %2010 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !44
  %2011 = extractelement <4 x half> %2010, i64 3, !dbg !44
  %2012 = extractelement <4 x half> %2010, i64 2, !dbg !44
  %2013 = extractelement <4 x half> %2010, i64 1, !dbg !44
  %2014 = extractelement <4 x half> %2010, i64 0, !dbg !44
  %2015 = insertelement <4 x float> poison, float %1616, i64 0, !dbg !43
  %2016 = insertelement <4 x float> %2015, float %1622, i64 1, !dbg !43
  %2017 = insertelement <4 x float> %2016, float %1617, i64 2, !dbg !43
  %2018 = insertelement <4 x float> %2017, float %1623, i64 3, !dbg !43
  %row.reciprocal88 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1567, !dbg !43
  %normalized89 = fmul <4 x float> %2018, %row.reciprocal88, !dbg !43
  %2019 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !44
  %2020 = extractelement <4 x half> %2019, i64 3, !dbg !44
  %2021 = extractelement <4 x half> %2019, i64 2, !dbg !44
  %2022 = extractelement <4 x half> %2019, i64 1, !dbg !44
  %2023 = extractelement <4 x half> %2019, i64 0, !dbg !44
  %2024 = insertelement <4 x float> poison, float %1506, i64 0, !dbg !43
  %2025 = insertelement <4 x float> %2024, float %1512, i64 1, !dbg !43
  %2026 = insertelement <4 x float> %2025, float %1507, i64 2, !dbg !43
  %2027 = insertelement <4 x float> %2026, float %1513, i64 3, !dbg !43
  %normalized90 = fmul <4 x float> %2027, %row.reciprocal88, !dbg !43
  %2028 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !44
  %2029 = extractelement <4 x half> %2028, i64 3, !dbg !44
  %2030 = extractelement <4 x half> %2028, i64 2, !dbg !44
  %2031 = extractelement <4 x half> %2028, i64 1, !dbg !44
  %2032 = extractelement <4 x half> %2028, i64 0, !dbg !44
  %2033 = shl nuw nsw i64 %15, 19, !dbg !45
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
  %address.local515 = add nuw nsw i64 %2033, %address.local514, !dbg !45
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local517 = lshr i64 %address.local506, 2, !dbg !37
  %address.local518 = and i64 %address.local517, 12, !dbg !37
  %address.local519 = or disjoint i64 %address.local518, 64, !dbg !45
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local519, !dbg !45
  store half %2032, ptr addrspace(1) %address.local520, align 2, !dbg !45
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local522 = or disjoint i64 %address.local518, 65, !dbg !45
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local522, !dbg !45
  store half %2031, ptr addrspace(1) %address.local523, align 2, !dbg !45
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local525 = or disjoint i64 %address.local518, 66, !dbg !45
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local524, i64 %address.local525, !dbg !45
  store half %2030, ptr addrspace(1) %address.local526, align 2, !dbg !45
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local528 = or disjoint i64 %address.local518, 67, !dbg !45
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local528, !dbg !45
  store half %2029, ptr addrspace(1) %address.local529, align 2, !dbg !45
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local531 = or disjoint i64 %address.local518, 80, !dbg !45
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local530, i64 %address.local531, !dbg !45
  store half %2023, ptr addrspace(1) %address.local532, align 2, !dbg !45
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local534 = or disjoint i64 %address.local518, 81, !dbg !45
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !45
  store half %2022, ptr addrspace(1) %address.local535, align 2, !dbg !45
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local537 = or disjoint i64 %address.local518, 82, !dbg !45
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local537, !dbg !45
  store half %2021, ptr addrspace(1) %address.local538, align 2, !dbg !45
  %address.local539 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local515, !dbg !45
  %address.local540 = or disjoint i64 %address.local518, 83, !dbg !45
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local539, i64 %address.local540, !dbg !45
  store half %2020, ptr addrspace(1) %address.local541, align 2, !dbg !45
  %address.local542 = shl nsw i64 %address.local513, 7, !dbg !45
  %address.local543 = add i64 %address.local542, 8192, !dbg !45
  %address.local544 = add nuw nsw i64 %2033, %address.local543, !dbg !45
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local545, i64 %address.local519, !dbg !45
  store half %2014, ptr addrspace(1) %address.local546, align 2, !dbg !45
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local522, !dbg !45
  store half %2013, ptr addrspace(1) %address.local548, align 2, !dbg !45
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local525, !dbg !45
  store half %2012, ptr addrspace(1) %address.local550, align 2, !dbg !45
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local528, !dbg !45
  store half %2011, ptr addrspace(1) %address.local552, align 2, !dbg !45
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local531, !dbg !45
  store half %2005, ptr addrspace(1) %address.local554, align 2, !dbg !45
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local534, !dbg !45
  store half %2004, ptr addrspace(1) %address.local556, align 2, !dbg !45
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local537, !dbg !45
  store half %2003, ptr addrspace(1) %address.local558, align 2, !dbg !45
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local544, !dbg !45
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local540, !dbg !45
  store half %2002, ptr addrspace(1) %address.local560, align 2, !dbg !45
  br label %2034, !dbg !45

2034:                                             ; preds = %1996, %1995
  br i1 true, label %2035, label %2073, !dbg !45

2035:                                             ; preds = %2034
  %2036 = insertelement <4 x float> poison, float %1806, i64 0, !dbg !43
  %2037 = insertelement <4 x float> %2036, float %1812, i64 1, !dbg !43
  %2038 = insertelement <4 x float> %2037, float %1807, i64 2, !dbg !43
  %2039 = insertelement <4 x float> %2038, float %1813, i64 3, !dbg !43
  %row.reciprocal91 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1785, !dbg !43
  %normalized92 = fmul <4 x float> %2039, %row.reciprocal91, !dbg !43
  %2040 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !44
  %2041 = extractelement <4 x half> %2040, i64 3, !dbg !44
  %2042 = extractelement <4 x half> %2040, i64 2, !dbg !44
  %2043 = extractelement <4 x half> %2040, i64 1, !dbg !44
  %2044 = extractelement <4 x half> %2040, i64 0, !dbg !44
  %2045 = insertelement <4 x float> poison, float %1695, i64 0, !dbg !43
  %2046 = insertelement <4 x float> %2045, float %1701, i64 1, !dbg !43
  %2047 = insertelement <4 x float> %2046, float %1696, i64 2, !dbg !43
  %2048 = insertelement <4 x float> %2047, float %1702, i64 3, !dbg !43
  %normalized93 = fmul <4 x float> %2048, %row.reciprocal91, !dbg !43
  %2049 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !44
  %2050 = extractelement <4 x half> %2049, i64 3, !dbg !44
  %2051 = extractelement <4 x half> %2049, i64 2, !dbg !44
  %2052 = extractelement <4 x half> %2049, i64 1, !dbg !44
  %2053 = extractelement <4 x half> %2049, i64 0, !dbg !44
  %2054 = insertelement <4 x float> poison, float %1588, i64 0, !dbg !43
  %2055 = insertelement <4 x float> %2054, float %1594, i64 1, !dbg !43
  %2056 = insertelement <4 x float> %2055, float %1589, i64 2, !dbg !43
  %2057 = insertelement <4 x float> %2056, float %1595, i64 3, !dbg !43
  %row.reciprocal94 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1567, !dbg !43
  %normalized95 = fmul <4 x float> %2057, %row.reciprocal94, !dbg !43
  %2058 = fptrunc <4 x float> %normalized95 to <4 x half>, !dbg !44
  %2059 = extractelement <4 x half> %2058, i64 3, !dbg !44
  %2060 = extractelement <4 x half> %2058, i64 2, !dbg !44
  %2061 = extractelement <4 x half> %2058, i64 1, !dbg !44
  %2062 = extractelement <4 x half> %2058, i64 0, !dbg !44
  %2063 = insertelement <4 x float> poison, float %1477, i64 0, !dbg !43
  %2064 = insertelement <4 x float> %2063, float %1483, i64 1, !dbg !43
  %2065 = insertelement <4 x float> %2064, float %1478, i64 2, !dbg !43
  %2066 = insertelement <4 x float> %2065, float %1484, i64 3, !dbg !43
  %normalized96 = fmul <4 x float> %2066, %row.reciprocal94, !dbg !43
  %2067 = fptrunc <4 x float> %normalized96 to <4 x half>, !dbg !44
  %2068 = extractelement <4 x half> %2067, i64 3, !dbg !44
  %2069 = extractelement <4 x half> %2067, i64 2, !dbg !44
  %2070 = extractelement <4 x half> %2067, i64 1, !dbg !44
  %2071 = extractelement <4 x half> %2067, i64 0, !dbg !44
  %2072 = shl nuw nsw i64 %15, 19, !dbg !45
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
  %address.local571 = add nuw nsw i64 %2072, %address.local570, !dbg !45
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local573 = lshr i64 %address.local562, 2, !dbg !37
  %address.local574 = and i64 %address.local573, 12, !dbg !37
  %address.local575 = or disjoint i64 %address.local574, 96, !dbg !45
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local575, !dbg !45
  store half %2071, ptr addrspace(1) %address.local576, align 2, !dbg !45
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local578 = or disjoint i64 %address.local574, 97, !dbg !45
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local578, !dbg !45
  store half %2070, ptr addrspace(1) %address.local579, align 2, !dbg !45
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local581 = or disjoint i64 %address.local574, 98, !dbg !45
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local580, i64 %address.local581, !dbg !45
  store half %2069, ptr addrspace(1) %address.local582, align 2, !dbg !45
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local584 = or disjoint i64 %address.local574, 99, !dbg !45
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local584, !dbg !45
  store half %2068, ptr addrspace(1) %address.local585, align 2, !dbg !45
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local587 = or disjoint i64 %address.local574, 112, !dbg !45
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local586, i64 %address.local587, !dbg !45
  store half %2062, ptr addrspace(1) %address.local588, align 2, !dbg !45
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local590 = or disjoint i64 %address.local574, 113, !dbg !45
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local590, !dbg !45
  store half %2061, ptr addrspace(1) %address.local591, align 2, !dbg !45
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local593 = or disjoint i64 %address.local574, 114, !dbg !45
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local593, !dbg !45
  store half %2060, ptr addrspace(1) %address.local594, align 2, !dbg !45
  %address.local595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local571, !dbg !45
  %address.local596 = or i64 %address.local573, 115, !dbg !45
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local595, i64 %address.local596, !dbg !45
  store half %2059, ptr addrspace(1) %address.local597, align 2, !dbg !45
  %address.local598 = shl nsw i64 %address.local569, 7, !dbg !45
  %address.local599 = add i64 %address.local598, 8192, !dbg !45
  %address.local600 = add nuw nsw i64 %2072, %address.local599, !dbg !45
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local601, i64 %address.local575, !dbg !45
  store half %2053, ptr addrspace(1) %address.local602, align 2, !dbg !45
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local578, !dbg !45
  store half %2052, ptr addrspace(1) %address.local604, align 2, !dbg !45
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local581, !dbg !45
  store half %2051, ptr addrspace(1) %address.local606, align 2, !dbg !45
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local584, !dbg !45
  store half %2050, ptr addrspace(1) %address.local608, align 2, !dbg !45
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local587, !dbg !45
  store half %2044, ptr addrspace(1) %address.local610, align 2, !dbg !45
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local590, !dbg !45
  store half %2043, ptr addrspace(1) %address.local612, align 2, !dbg !45
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local593, !dbg !45
  store half %2042, ptr addrspace(1) %address.local614, align 2, !dbg !45
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local600, !dbg !45
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local596, !dbg !45
  store half %2041, ptr addrspace(1) %address.local616, align 2, !dbg !45
  br label %2073, !dbg !45

2073:                                             ; preds = %2035, %2034
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

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="6" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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
