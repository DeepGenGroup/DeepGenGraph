; ModuleID = '3rd/deepgengraph/test/test_input_sd16.mlir'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !32 {
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
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !34
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = or disjoint i64 %9, 127, !dbg !14
  %23 = add nsw i64 %9, 128, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  br label %25, !dbg !16

25:                                               ; preds = %419, %4
  %26 = phi i64 [ %1170, %419 ], [ 0, %4 ], !dbg !16
  %27 = phi [2 x <8 x float>] [ %1157, %419 ], [ zeroinitializer, %4 ], !dbg !16
  %28 = phi [2 x <1 x float>] [ %779, %419 ], [ zeroinitializer, %4 ], !dbg !16
  %29 = phi [2 x <8 x float>] [ %1161, %419 ], [ zeroinitializer, %4 ], !dbg !16
  %30 = phi [2 x <8 x float>] [ %1165, %419 ], [ zeroinitializer, %4 ], !dbg !16
  %31 = phi [2 x <8 x float>] [ %1169, %419 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %26, %6, !dbg !16
  br i1 %.not, label %1171, label %32, !dbg !16

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
  %.idx68 = shl nsw i64 %18, 8, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 %.idx68, !dbg !12
  %45 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %42, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !34, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local130 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local131 = and i64 %address.local130, 1536, !dbg !12
  %address.local132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local131, !dbg !12
  %address.local133 = ptrtoint ptr addrspace(3) %address.local132 to i64, !dbg !12
  %address.local134 = trunc i64 %address.local133 to i32, !dbg !12
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local134), !dbg !12
  %47 = zext i32 %46 to i64, !dbg !12
  %48 = inttoptr i64 %47 to ptr addrspace(3), !dbg !12
  %address.local135 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local136 = and i64 %address.local135, 16256, !dbg !12
  %address.local137 = lshr i64 %address.local, 4, !dbg !12
  %address.local138 = and i64 %address.local137, 8, !dbg !12
  %address.local139 = or disjoint i64 %address.local136, %address.local138, !dbg !12
  %address.local140 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local139, !dbg !12
  %49 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %50 = ptrtoint ptr addrspace(1) %address.local140 to i64, !dbg !12
  %51 = sub i64 %50, %49, !dbg !12
  %dma.buffer.offset876 = trunc i64 %51 to i32, !dbg !12
  %52 = trunc i64 %49 to i32, !dbg !12
  %53 = insertelement <4 x i32> poison, i32 %52, i64 0, !dbg !12
  %54 = lshr i64 %49, 32, !dbg !12
  %55 = trunc i64 %54 to i32, !dbg !12
  %56 = insertelement <4 x i32> %53, i32 %55, i64 1, !dbg !12
  %57 = insertelement <4 x i32> %56, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc877 = insertelement <4 x i32> %57, i32 131072, i64 3, !dbg !12
  %58 = ptrtoint ptr addrspace(3) %48 to i32, !dbg !12
  %dma.buffer.lds878 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !12
  %59 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds878, i32 %dma.buffer.offset876, <4 x i32> %dma.buffer.rsrc877) #10, !dbg !12
  %address.local141 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local142 = shl nuw nsw i64 %address.local141, 3, !dbg !12
  %address.local143 = and i64 %address.local142, 3584, !dbg !12
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !12
  %address.local145 = ptrtoint ptr addrspace(3) %address.local144 to i64, !dbg !12
  %address.local146 = trunc i64 %address.local145 to i32, !dbg !12
  %60 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local146), !dbg !12
  %61 = zext i32 %60 to i64, !dbg !12
  %62 = inttoptr i64 %61 to ptr addrspace(3), !dbg !12
  %address.local147 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local148 = and i64 %address.local147, 16256, !dbg !12
  %address.local149 = lshr i64 %address.local141, 4, !dbg !12
  %address.local150 = and i64 %address.local149, 24, !dbg !12
  %address.local151 = or disjoint i64 %address.local148, %address.local150, !dbg !12
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %address.local151, !dbg !12
  %63 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %64 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !12
  %65 = sub i64 %64, %63, !dbg !12
  %dma.buffer.offset873 = trunc i64 %65 to i32, !dbg !12
  %66 = trunc i64 %63 to i32, !dbg !12
  %67 = insertelement <4 x i32> poison, i32 %66, i64 0, !dbg !12
  %68 = lshr i64 %63, 32, !dbg !12
  %69 = trunc i64 %68 to i32, !dbg !12
  %70 = insertelement <4 x i32> %67, i32 %69, i64 1, !dbg !12
  %71 = insertelement <4 x i32> %70, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc874 = insertelement <4 x i32> %71, i32 131072, i64 3, !dbg !12
  %72 = ptrtoint ptr addrspace(3) %62 to i32, !dbg !12
  %dma.buffer.lds875 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72), !dbg !12
  %73 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds875, i32 %dma.buffer.offset873, <4 x i32> %dma.buffer.rsrc874) #10, !dbg !12
  %74 = shl nuw nsw i64 %38, 10, !dbg !17
  %75 = add nuw i64 %26, %74, !dbg !17
  %.lobit69 = ashr i64 %75, 63, !dbg !17
  %76 = xor i64 %75, %.lobit69, !dbg !17
  %77 = sdiv i64 %76, 4096, !dbg !17
  %.lobit70 = ashr i64 %75, 63, !dbg !17
  %78 = xor i64 %77, %.lobit70, !dbg !17
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
  %.idx71 = shl i64 %36, 8, !dbg !17
  %86 = getelementptr i8, ptr addrspace(1) %85, i64 %.idx71, !dbg !17
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local131, !dbg !17
  %address.local154 = ptrtoint ptr addrspace(3) %address.local153 to i64, !dbg !17
  %address.local155 = trunc i64 %address.local154 to i32, !dbg !17
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !17
  %88 = zext i32 %87 to i64, !dbg !17
  %89 = inttoptr i64 %88 to ptr addrspace(3), !dbg !17
  %address.local156 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %address.local139, !dbg !17
  %90 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %91 = ptrtoint ptr addrspace(1) %address.local156 to i64, !dbg !17
  %92 = sub i64 %91, %90, !dbg !17
  %dma.buffer.offset870 = trunc i64 %92 to i32, !dbg !17
  %93 = trunc i64 %90 to i32, !dbg !17
  %94 = insertelement <4 x i32> poison, i32 %93, i64 0, !dbg !17
  %95 = lshr i64 %90, 32, !dbg !17
  %96 = trunc i64 %95 to i32, !dbg !17
  %97 = insertelement <4 x i32> %94, i32 %96, i64 1, !dbg !17
  %98 = insertelement <4 x i32> %97, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc871 = insertelement <4 x i32> %98, i32 131072, i64 3, !dbg !17
  %99 = ptrtoint ptr addrspace(3) %89 to i32, !dbg !17
  %dma.buffer.lds872 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %99), !dbg !17
  %100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds872, i32 %dma.buffer.offset870, <4 x i32> %dma.buffer.rsrc871) #10, !dbg !17
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local143, !dbg !17
  %address.local158 = ptrtoint ptr addrspace(3) %address.local157 to i64, !dbg !17
  %address.local159 = trunc i64 %address.local158 to i32, !dbg !17
  %101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local159), !dbg !17
  %102 = zext i32 %101 to i64, !dbg !17
  %103 = inttoptr i64 %102 to ptr addrspace(3), !dbg !17
  %address.local160 = getelementptr [2 x i8], ptr addrspace(1) %86, i64 %address.local151, !dbg !17
  %104 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %105 = ptrtoint ptr addrspace(1) %address.local160 to i64, !dbg !17
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
  %address.local161 = lshr i32 %address.thread, 2, !dbg !13
  %address.local162 = and i32 %address.local161, 4, !dbg !13
  %address.local163 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local164 = and i32 %address.local163, 384, !dbg !13
  %address.local165 = or disjoint i32 %address.local162, %address.local164, !dbg !13
  %address.local166 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local167 = and i32 %address.local166, 120, !dbg !13
  %address.local168 = or disjoint i32 %address.local165, %address.local167, !dbg !13
  %address.local169 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local170 = and i32 %address.local169, 1024, !dbg !13
  %address.local171 = or disjoint i32 %address.local168, %address.local170, !dbg !13
  %address.local172 = zext nneg i32 %address.local171 to i64, !dbg !13
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %115 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !13
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local175 = getelementptr i8, ptr addrspace(3) %address.local174, i64 4096, !dbg !13
  %116 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !13
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local177 = getelementptr i8, ptr addrspace(3) %address.local176, i64 1024, !dbg !13
  %117 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !13
  %118 = fmul <4 x half> %115, splat (half 0xH3015), !dbg !13
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local179 = getelementptr i8, ptr addrspace(3) %address.local178, i64 5120, !dbg !13
  %119 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !13
  %120 = fmul <4 x half> %116, splat (half 0xH3015), !dbg !13
  %121 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !13
  %122 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local180 = lshr i32 %address.thread, 2, !dbg !8
  %address.local181 = and i32 %address.local180, 4, !dbg !8
  %address.local182 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local183 = and i32 %address.local182, 24, !dbg !8
  %address.local184 = or disjoint i32 %address.local181, %address.local183, !dbg !8
  %address.local185 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local186 = and i32 %address.local185, 1024, !dbg !8
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !8
  %address.local188 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local189 = and i32 %address.local188, 384, !dbg !8
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !8
  %address.local191 = zext nneg i32 %address.local190 to i64, !dbg !8
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !8
  %123 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !8
  %124 = extractvalue [2 x <32 x float>] %39, 0, !dbg !31
  %125 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %123, <4 x float> %125) #9, !dbg !8
  %address.local193 = lshr i32 %address.thread, 2, !dbg !8
  %address.local194 = and i32 %address.local193, 4, !dbg !8
  %address.local195 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local196 = and i32 %address.local195, 24, !dbg !8
  %address.local197 = or disjoint i32 %address.local194, %address.local196, !dbg !8
  %address.local198 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local199 = and i32 %address.local198, 1024, !dbg !8
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !8
  %address.local201 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local202 = and i32 %address.local201, 384, !dbg !8
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !8
  %address.local204 = zext nneg i32 %address.local203 to i64, !dbg !8
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local204, !dbg !8
  %address.local206 = getelementptr i8, ptr addrspace(3) %address.local205, i64 4096, !dbg !8
  %127 = load <4 x half>, ptr addrspace(3) %address.local206, align 2, !dbg !8
  %128 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %127, <4 x float> %126) #9, !dbg !8
  %130 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %131 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %132 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %133 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %134 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %135 = shufflevector <32 x float> %124, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %136 = extractvalue [2 x <32 x float>] %39, 1, !dbg !31
  %137 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %138 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %139 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %140 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %141 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %142 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %143 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %144 = shufflevector <32 x float> %136, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local207 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local209 = lshr i64 %address.local, 2, !dbg !8
  %address.local210 = mul nsw i64 %address.local209, -512, !dbg !8
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local210, !dbg !8
  %address.local212 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local213 = and i64 %address.local212, 24, !dbg !8
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !8
  %address.local215 = lshr i64 %address.local, 2, !dbg !13
  %address.local216 = and i64 %address.local215, 12, !dbg !13
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !8
  %address.local218 = and i64 %address.local, 63, !dbg !13
  %address.local219 = lshr i64 %address.local218, 5, !dbg !13
  %address.local220 = mul nuw nsw i64 %address.local219, 1016, !dbg !13
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local220, !dbg !8
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 64, !dbg !8
  %145 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !8
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local216, !dbg !8
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local220, !dbg !8
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local210, !dbg !8
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local213, !dbg !8
  %address.local228 = getelementptr i8, ptr addrspace(3) %address.local227, i64 4160, !dbg !8
  %146 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !8
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %145, <4 x float> %128) #9, !dbg !8
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local210, !dbg !8
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local213, !dbg !8
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local216, !dbg !8
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local220, !dbg !8
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local233, i64 128, !dbg !8
  %148 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !8
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %146, <4 x float> %147) #9, !dbg !8
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local216, !dbg !8
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local220, !dbg !8
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local210, !dbg !8
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local213, !dbg !8
  %address.local240 = getelementptr i8, ptr addrspace(3) %address.local239, i64 4224, !dbg !8
  %150 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !8
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %148, <4 x float> %130) #9, !dbg !8
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %150, <4 x float> %151) #9, !dbg !8
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local210, !dbg !8
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local213, !dbg !8
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local216, !dbg !8
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local220, !dbg !8
  %address.local246 = getelementptr i8, ptr addrspace(3) %address.local245, i64 192, !dbg !8
  %153 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !8
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !8
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local216, !dbg !8
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local220, !dbg !8
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local210, !dbg !8
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local213, !dbg !8
  %address.local252 = getelementptr i8, ptr addrspace(3) %address.local251, i64 4288, !dbg !8
  %154 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !8
  %155 = shufflevector <4 x float> %152, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %153, <4 x float> %131) #9, !dbg !8
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %154, <4 x float> %156) #9, !dbg !8
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !8
  %address.local254 = getelementptr i8, ptr addrspace(3) %address.local253, i64 1024, !dbg !8
  %158 = load <4 x half>, ptr addrspace(3) %address.local254, align 2, !dbg !8
  %159 = shufflevector <4 x float> %157, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %158, <4 x float> %132) #9, !dbg !8
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local204, !dbg !8
  %address.local256 = getelementptr i8, ptr addrspace(3) %address.local255, i64 5120, !dbg !8
  %161 = load <4 x half>, ptr addrspace(3) %address.local256, align 2, !dbg !8
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %161, <4 x float> %160) #9, !dbg !8
  %address.local257 = add nuw nsw i64 %address.local207, 512, !dbg !8
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !8
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local210, !dbg !8
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local213, !dbg !8
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local216, !dbg !8
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local220, !dbg !8
  %address.local263 = getelementptr i8, ptr addrspace(3) %address.local262, i64 64, !dbg !8
  %163 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !8
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local216, !dbg !8
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local220, !dbg !8
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local210, !dbg !8
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local213, !dbg !8
  %address.local269 = getelementptr i8, ptr addrspace(3) %address.local268, i64 4160, !dbg !8
  %164 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !8
  %165 = shufflevector <4 x float> %162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %163, <4 x float> %133) #9, !dbg !8
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %164, <4 x float> %166) #9, !dbg !8
  %address.local270 = add nuw nsw i64 %address.local207, 512, !dbg !8
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !8
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local210, !dbg !8
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local213, !dbg !8
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local216, !dbg !8
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local220, !dbg !8
  %address.local276 = getelementptr i8, ptr addrspace(3) %address.local275, i64 128, !dbg !8
  %168 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !8
  %169 = shufflevector <4 x float> %167, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %168, <4 x float> %134) #9, !dbg !8
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !8
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local216, !dbg !8
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local220, !dbg !8
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local210, !dbg !8
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local213, !dbg !8
  %address.local282 = getelementptr i8, ptr addrspace(3) %address.local281, i64 4224, !dbg !8
  %171 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !8
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %171, <4 x float> %170) #9, !dbg !8
  %address.local283 = add nuw nsw i64 %address.local207, 512, !dbg !8
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local283, !dbg !8
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local210, !dbg !8
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local213, !dbg !8
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local216, !dbg !8
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local220, !dbg !8
  %address.local289 = getelementptr i8, ptr addrspace(3) %address.local288, i64 192, !dbg !8
  %173 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !8
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local283, !dbg !8
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local216, !dbg !8
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local220, !dbg !8
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local210, !dbg !8
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local213, !dbg !8
  %address.local295 = getelementptr i8, ptr addrspace(3) %address.local294, i64 4288, !dbg !8
  %174 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !8
  %175 = shufflevector <4 x float> %172, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %118, <4 x half> %173, <4 x float> %135) #9, !dbg !8
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %120, <4 x half> %174, <4 x float> %176) #9, !dbg !8
  %178 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !8
  %179 = shufflevector <4 x float> %177, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %178, <4 x float> %137) #9, !dbg !8
  %181 = load <4 x half>, ptr addrspace(3) %address.local206, align 2, !dbg !8
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %181, <4 x float> %180) #9, !dbg !8
  %183 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !8
  %184 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !8
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %183, <4 x float> %138) #9, !dbg !8
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %184, <4 x float> %185) #9, !dbg !8
  %187 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !8
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %187, <4 x float> %139) #9, !dbg !8
  %189 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !8
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %189, <4 x float> %188) #9, !dbg !8
  %191 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !8
  %192 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !8
  %193 = shufflevector <4 x float> %190, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %191, <4 x float> %140) #9, !dbg !8
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %192, <4 x float> %194) #9, !dbg !8
  %196 = load <4 x half>, ptr addrspace(3) %address.local254, align 2, !dbg !8
  %197 = shufflevector <4 x float> %195, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %198 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %196, <4 x float> %141) #9, !dbg !8
  %199 = load <4 x half>, ptr addrspace(3) %address.local256, align 2, !dbg !8
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %199, <4 x float> %198) #9, !dbg !8
  %201 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !8
  %202 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !8
  %203 = shufflevector <4 x float> %200, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %201, <4 x float> %142) #9, !dbg !8
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %202, <4 x float> %204) #9, !dbg !8
  %206 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !8
  %207 = shufflevector <4 x float> %205, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %206, <4 x float> %143) #9, !dbg !8
  %209 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !8
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %209, <4 x float> %208) #9, !dbg !8
  %211 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !8
  %212 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !8
  %213 = shufflevector <4 x float> %210, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %211, <4 x float> %144) #9, !dbg !8
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %212, <4 x float> %214) #9, !dbg !8
  %216 = shufflevector <4 x float> %215, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %217 = shufflevector <4 x float> %129, <4 x float> %149, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %218 = shufflevector <32 x float> %217, <32 x float> %155, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %219 = shufflevector <32 x float> %218, <32 x float> %159, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %220 = shufflevector <32 x float> %219, <32 x float> %165, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %221 = shufflevector <32 x float> %220, <32 x float> %169, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %222 = shufflevector <32 x float> %221, <32 x float> %175, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %223 = shufflevector <32 x float> %222, <32 x float> %179, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %224 = insertvalue [2 x <32 x float>] poison, <32 x float> %223, 0, !dbg !22
  %225 = shufflevector <4 x float> %182, <4 x float> %186, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %226 = shufflevector <32 x float> %225, <32 x float> %193, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %227 = shufflevector <32 x float> %226, <32 x float> %197, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %228 = shufflevector <32 x float> %227, <32 x float> %203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %229 = shufflevector <32 x float> %228, <32 x float> %207, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
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
  %address.thread296 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local297 = zext nneg i32 %address.thread296 to i64, !dbg !13
  %address.local298 = shl nuw nsw i64 %address.local297, 3, !dbg !12
  %address.local299 = and i64 %address.local298, 1536, !dbg !12
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local299, !dbg !12
  %address.local301 = ptrtoint ptr addrspace(3) %address.local300 to i64, !dbg !12
  %address.local302 = trunc i64 %address.local301 to i32, !dbg !12
  %241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local302), !dbg !18
  %242 = zext i32 %241 to i64, !dbg !18
  %243 = inttoptr i64 %242 to ptr addrspace(3), !dbg !18
  %address.local303 = shl nuw nsw i64 %address.local297, 5, !dbg !18
  %address.local304 = and i64 %address.local303, 8064, !dbg !18
  %address.local305 = shl nuw nsw i64 %address.local297, 3, !dbg !18
  %address.local306 = and i64 %address.local305, 24, !dbg !18
  %address.local307 = or disjoint i64 %address.local304, %address.local306, !dbg !18
  %address.local308 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local307, !dbg !18
  %244 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %245 = ptrtoint ptr addrspace(1) %address.local308 to i64, !dbg !18
  %246 = sub i64 %245, %244, !dbg !18
  %dma.buffer.offset900 = trunc i64 %246 to i32, !dbg !18
  %247 = trunc i64 %244 to i32, !dbg !18
  %248 = insertelement <4 x i32> poison, i32 %247, i64 0, !dbg !18
  %249 = lshr i64 %244, 32, !dbg !18
  %250 = trunc i64 %249 to i32, !dbg !18
  %251 = insertelement <4 x i32> %248, i32 %250, i64 1, !dbg !18
  %252 = insertelement <4 x i32> %251, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc901 = insertelement <4 x i32> %252, i32 131072, i64 3, !dbg !18
  %253 = ptrtoint ptr addrspace(3) %243 to i32, !dbg !18
  %dma.buffer.lds902 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !18
  %254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds902, i32 %dma.buffer.offset900, <4 x i32> %dma.buffer.rsrc901) #10, !dbg !18
  %address.local309 = or disjoint i64 %address.local297, 256, !dbg !12
  %address.local310 = shl nuw nsw i64 %address.local309, 3, !dbg !12
  %address.local311 = and i64 %address.local310, 3584, !dbg !12
  %address.local312 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local311, !dbg !12
  %address.local313 = ptrtoint ptr addrspace(3) %address.local312 to i64, !dbg !12
  %address.local314 = trunc i64 %address.local313 to i32, !dbg !12
  %255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local314), !dbg !18
  %256 = zext i32 %255 to i64, !dbg !18
  %257 = inttoptr i64 %256 to ptr addrspace(3), !dbg !18
  %address.local315 = shl nuw nsw i64 %address.local309, 5, !dbg !18
  %address.local316 = and i64 %address.local315, 16256, !dbg !18
  %address.local317 = shl nuw nsw i64 %address.local297, 3, !dbg !18
  %address.local318 = and i64 %address.local317, 24, !dbg !18
  %address.local319 = or disjoint i64 %address.local316, %address.local318, !dbg !18
  %address.local320 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local319, !dbg !18
  %258 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %259 = ptrtoint ptr addrspace(1) %address.local320 to i64, !dbg !18
  %260 = sub i64 %259, %258, !dbg !18
  %dma.buffer.offset897 = trunc i64 %260 to i32, !dbg !18
  %261 = trunc i64 %258 to i32, !dbg !18
  %262 = insertelement <4 x i32> poison, i32 %261, i64 0, !dbg !18
  %263 = lshr i64 %258, 32, !dbg !18
  %264 = trunc i64 %263 to i32, !dbg !18
  %265 = insertelement <4 x i32> %262, i32 %264, i64 1, !dbg !18
  %266 = insertelement <4 x i32> %265, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc898 = insertelement <4 x i32> %266, i32 131072, i64 3, !dbg !18
  %267 = ptrtoint ptr addrspace(3) %257 to i32, !dbg !18
  %dma.buffer.lds899 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %267), !dbg !18
  %268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds899, i32 %dma.buffer.offset897, <4 x i32> %dma.buffer.rsrc898) #10, !dbg !18
  %269 = or disjoint i64 %33, 127, !dbg !14
  %270 = icmp uge i64 %9, %269, !dbg !14
  %271 = and i1 %24, %270, !dbg !14
  br i1 %271, label %272, label %273, !dbg !14

272:                                              ; preds = %234
  br label %417, !dbg !14

273:                                              ; preds = %234
  %274 = add nsw i64 %9, %20, !dbg !14
  %275 = shl nuw nsw i64 %21, 4, !dbg !14
  %276 = add nsw i64 %274, %275, !dbg !14
  %277 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %277 to i64, !dbg !14
  %278 = sub nsw i64 %276, %index.compact, !dbg !14
  %279 = and i64 %20, 48, !dbg !14
  %280 = or disjoint i64 %33, %279, !dbg !14
  %281 = add nsw i64 %278, 1, !dbg !23
  %282 = lshr i64 %281, 32, !dbg !24
  %compare.high = trunc i64 %282 to i32, !dbg !24
  %283 = lshr i64 %280, 32, !dbg !24
  %compare.high614 = trunc i64 %283 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high614, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high614, !dbg !24
  %compare.low.base = trunc i64 %281 to i32, !dbg !24
  %compare.low.base615 = trunc i64 %280 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base615, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %284 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low616 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result617 = select i1 %compare.high.equal, i1 %compare.low616, i1 %compare.high.order, !dbg !24
  %285 = select i1 %compare.result617, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base618 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local619 = or i32 %compare.local.base618, 2, !dbg !24
  %compare.low620 = icmp ugt i32 %compare.low.base, %compare.local619, !dbg !24
  %compare.result621 = select i1 %compare.high.equal, i1 %compare.low620, i1 %compare.high.order, !dbg !24
  %286 = select i1 %compare.result621, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base622 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local623 = or i32 %compare.local.base622, 3, !dbg !24
  %compare.low624 = icmp ugt i32 %compare.low.base, %compare.local623, !dbg !24
  %compare.result625 = select i1 %compare.high.equal, i1 %compare.low624, i1 %compare.high.order, !dbg !24
  %287 = select i1 %compare.result625, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base626 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local627 = or i32 %compare.local.base626, 4, !dbg !24
  %compare.low628 = icmp ugt i32 %compare.low.base, %compare.local627, !dbg !24
  %compare.result629 = select i1 %compare.high.equal, i1 %compare.low628, i1 %compare.high.order, !dbg !24
  %288 = select i1 %compare.result629, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base630 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local631 = or i32 %compare.local.base630, 5, !dbg !24
  %compare.low632 = icmp ugt i32 %compare.low.base, %compare.local631, !dbg !24
  %compare.result633 = select i1 %compare.high.equal, i1 %compare.low632, i1 %compare.high.order, !dbg !24
  %289 = select i1 %compare.result633, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base634 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local635 = or i32 %compare.local.base634, 6, !dbg !24
  %compare.low636 = icmp ugt i32 %compare.low.base, %compare.local635, !dbg !24
  %compare.result637 = select i1 %compare.high.equal, i1 %compare.low636, i1 %compare.high.order, !dbg !24
  %290 = select i1 %compare.result637, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base638 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local639 = or i32 %compare.local.base638, 7, !dbg !24
  %compare.low640 = icmp ugt i32 %compare.low.base, %compare.local639, !dbg !24
  %compare.result641 = select i1 %compare.high.equal, i1 %compare.low640, i1 %compare.high.order, !dbg !24
  %291 = select i1 %compare.result641, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base642 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local643 = or i32 %compare.local.base642, 8, !dbg !24
  %compare.low644 = icmp ugt i32 %compare.low.base, %compare.local643, !dbg !24
  %compare.result645 = select i1 %compare.high.equal, i1 %compare.low644, i1 %compare.high.order, !dbg !24
  %292 = select i1 %compare.result645, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base646 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local647 = or i32 %compare.local.base646, 9, !dbg !24
  %compare.low648 = icmp ugt i32 %compare.low.base, %compare.local647, !dbg !24
  %compare.result649 = select i1 %compare.high.equal, i1 %compare.low648, i1 %compare.high.order, !dbg !24
  %293 = select i1 %compare.result649, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base650 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local651 = or i32 %compare.local.base650, 10, !dbg !24
  %compare.low652 = icmp ugt i32 %compare.low.base, %compare.local651, !dbg !24
  %compare.result653 = select i1 %compare.high.equal, i1 %compare.low652, i1 %compare.high.order, !dbg !24
  %294 = select i1 %compare.result653, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base654 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local655 = or i32 %compare.local.base654, 11, !dbg !24
  %compare.low656 = icmp ugt i32 %compare.low.base, %compare.local655, !dbg !24
  %compare.result657 = select i1 %compare.high.equal, i1 %compare.low656, i1 %compare.high.order, !dbg !24
  %295 = select i1 %compare.result657, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base658 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local659 = or i32 %compare.local.base658, 12, !dbg !24
  %compare.low660 = icmp ugt i32 %compare.low.base, %compare.local659, !dbg !24
  %compare.result661 = select i1 %compare.high.equal, i1 %compare.low660, i1 %compare.high.order, !dbg !24
  %296 = select i1 %compare.result661, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base662 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local663 = or i32 %compare.local.base662, 13, !dbg !24
  %compare.low664 = icmp ugt i32 %compare.low.base, %compare.local663, !dbg !24
  %compare.result665 = select i1 %compare.high.equal, i1 %compare.low664, i1 %compare.high.order, !dbg !24
  %297 = select i1 %compare.result665, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base666 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local667 = or i32 %compare.local.base666, 14, !dbg !24
  %compare.low668 = icmp ugt i32 %compare.low.base, %compare.local667, !dbg !24
  %compare.result669 = select i1 %compare.high.equal, i1 %compare.low668, i1 %compare.high.order, !dbg !24
  %298 = select i1 %compare.result669, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base670 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local671 = or i32 %compare.local.base670, 15, !dbg !24
  %compare.low672 = icmp ugt i32 %compare.low.base, %compare.local671, !dbg !24
  %compare.result673 = select i1 %compare.high.equal, i1 %compare.low672, i1 %compare.high.order, !dbg !24
  %299 = select i1 %compare.result673, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base674 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local675 = or i32 %compare.local.base674, 64, !dbg !24
  %compare.low676 = icmp ugt i32 %compare.low.base, %compare.local675, !dbg !24
  %compare.result677 = select i1 %compare.high.equal, i1 %compare.low676, i1 %compare.high.order, !dbg !24
  %300 = select i1 %compare.result677, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base678 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local679 = or i32 %compare.local.base678, 65, !dbg !24
  %compare.low680 = icmp ugt i32 %compare.low.base, %compare.local679, !dbg !24
  %compare.result681 = select i1 %compare.high.equal, i1 %compare.low680, i1 %compare.high.order, !dbg !24
  %301 = select i1 %compare.result681, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base682 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local683 = or i32 %compare.local.base682, 66, !dbg !24
  %compare.low684 = icmp ugt i32 %compare.low.base, %compare.local683, !dbg !24
  %compare.result685 = select i1 %compare.high.equal, i1 %compare.low684, i1 %compare.high.order, !dbg !24
  %302 = select i1 %compare.result685, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base686 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local687 = or i32 %compare.local.base686, 67, !dbg !24
  %compare.low688 = icmp ugt i32 %compare.low.base, %compare.local687, !dbg !24
  %compare.result689 = select i1 %compare.high.equal, i1 %compare.low688, i1 %compare.high.order, !dbg !24
  %303 = select i1 %compare.result689, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base690 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local691 = or i32 %compare.local.base690, 68, !dbg !24
  %compare.low692 = icmp ugt i32 %compare.low.base, %compare.local691, !dbg !24
  %compare.result693 = select i1 %compare.high.equal, i1 %compare.low692, i1 %compare.high.order, !dbg !24
  %304 = select i1 %compare.result693, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base694 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local695 = or i32 %compare.local.base694, 69, !dbg !24
  %compare.low696 = icmp ugt i32 %compare.low.base, %compare.local695, !dbg !24
  %compare.result697 = select i1 %compare.high.equal, i1 %compare.low696, i1 %compare.high.order, !dbg !24
  %305 = select i1 %compare.result697, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base698 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local699 = or i32 %compare.local.base698, 70, !dbg !24
  %compare.low700 = icmp ugt i32 %compare.low.base, %compare.local699, !dbg !24
  %compare.result701 = select i1 %compare.high.equal, i1 %compare.low700, i1 %compare.high.order, !dbg !24
  %306 = select i1 %compare.result701, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base702 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local703 = or i32 %compare.local.base702, 71, !dbg !24
  %compare.low704 = icmp ugt i32 %compare.low.base, %compare.local703, !dbg !24
  %compare.result705 = select i1 %compare.high.equal, i1 %compare.low704, i1 %compare.high.order, !dbg !24
  %307 = select i1 %compare.result705, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base706 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local707 = or i32 %compare.local.base706, 72, !dbg !24
  %compare.low708 = icmp ugt i32 %compare.low.base, %compare.local707, !dbg !24
  %compare.result709 = select i1 %compare.high.equal, i1 %compare.low708, i1 %compare.high.order, !dbg !24
  %308 = select i1 %compare.result709, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base710 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local711 = or i32 %compare.local.base710, 73, !dbg !24
  %compare.low712 = icmp ugt i32 %compare.low.base, %compare.local711, !dbg !24
  %compare.result713 = select i1 %compare.high.equal, i1 %compare.low712, i1 %compare.high.order, !dbg !24
  %309 = select i1 %compare.result713, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base714 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local715 = or i32 %compare.local.base714, 74, !dbg !24
  %compare.low716 = icmp ugt i32 %compare.low.base, %compare.local715, !dbg !24
  %compare.result717 = select i1 %compare.high.equal, i1 %compare.low716, i1 %compare.high.order, !dbg !24
  %310 = select i1 %compare.result717, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base718 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local719 = or i32 %compare.local.base718, 75, !dbg !24
  %compare.low720 = icmp ugt i32 %compare.low.base, %compare.local719, !dbg !24
  %compare.result721 = select i1 %compare.high.equal, i1 %compare.low720, i1 %compare.high.order, !dbg !24
  %311 = select i1 %compare.result721, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base722 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local723 = or i32 %compare.local.base722, 76, !dbg !24
  %compare.low724 = icmp ugt i32 %compare.low.base, %compare.local723, !dbg !24
  %compare.result725 = select i1 %compare.high.equal, i1 %compare.low724, i1 %compare.high.order, !dbg !24
  %312 = select i1 %compare.result725, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base726 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local727 = or i32 %compare.local.base726, 77, !dbg !24
  %compare.low728 = icmp ugt i32 %compare.low.base, %compare.local727, !dbg !24
  %compare.result729 = select i1 %compare.high.equal, i1 %compare.low728, i1 %compare.high.order, !dbg !24
  %313 = select i1 %compare.result729, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base730 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local731 = or i32 %compare.local.base730, 78, !dbg !24
  %compare.low732 = icmp ugt i32 %compare.low.base, %compare.local731, !dbg !24
  %compare.result733 = select i1 %compare.high.equal, i1 %compare.low732, i1 %compare.high.order, !dbg !24
  %314 = select i1 %compare.result733, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base734 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local735 = or i32 %compare.local.base734, 79, !dbg !24
  %compare.low736 = icmp ugt i32 %compare.low.base, %compare.local735, !dbg !24
  %compare.result737 = select i1 %compare.high.equal, i1 %compare.low736, i1 %compare.high.order, !dbg !24
  %315 = select i1 %compare.result737, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %316 = add nsw i64 %278, 65, !dbg !23
  %317 = lshr i64 %316, 32, !dbg !24
  %compare.high738 = trunc i64 %317 to i32, !dbg !24
  %318 = lshr i64 %280, 32, !dbg !24
  %compare.high739 = trunc i64 %318 to i32, !dbg !24
  %compare.high.equal740 = icmp eq i32 %compare.high738, %compare.high739, !dbg !24
  %compare.high.order741 = icmp ugt i32 %compare.high738, %compare.high739, !dbg !24
  %compare.low.base742 = trunc i64 %316 to i32, !dbg !24
  %compare.low.base743 = trunc i64 %280 to i32, !dbg !24
  %compare.low744 = icmp ugt i32 %compare.low.base742, %compare.low.base743, !dbg !24
  %compare.result745 = select i1 %compare.high.equal740, i1 %compare.low744, i1 %compare.high.order741, !dbg !24
  %319 = select i1 %compare.result745, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base746 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local747 = or i32 %compare.local.base746, 1, !dbg !24
  %compare.low748 = icmp ugt i32 %compare.low.base742, %compare.local747, !dbg !24
  %compare.result749 = select i1 %compare.high.equal740, i1 %compare.low748, i1 %compare.high.order741, !dbg !24
  %320 = select i1 %compare.result749, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base750 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local751 = or i32 %compare.local.base750, 2, !dbg !24
  %compare.low752 = icmp ugt i32 %compare.low.base742, %compare.local751, !dbg !24
  %compare.result753 = select i1 %compare.high.equal740, i1 %compare.low752, i1 %compare.high.order741, !dbg !24
  %321 = select i1 %compare.result753, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base754 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local755 = or i32 %compare.local.base754, 3, !dbg !24
  %compare.low756 = icmp ugt i32 %compare.low.base742, %compare.local755, !dbg !24
  %compare.result757 = select i1 %compare.high.equal740, i1 %compare.low756, i1 %compare.high.order741, !dbg !24
  %322 = select i1 %compare.result757, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base758 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local759 = or i32 %compare.local.base758, 4, !dbg !24
  %compare.low760 = icmp ugt i32 %compare.low.base742, %compare.local759, !dbg !24
  %compare.result761 = select i1 %compare.high.equal740, i1 %compare.low760, i1 %compare.high.order741, !dbg !24
  %323 = select i1 %compare.result761, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base762 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local763 = or i32 %compare.local.base762, 5, !dbg !24
  %compare.low764 = icmp ugt i32 %compare.low.base742, %compare.local763, !dbg !24
  %compare.result765 = select i1 %compare.high.equal740, i1 %compare.low764, i1 %compare.high.order741, !dbg !24
  %324 = select i1 %compare.result765, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base766 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local767 = or i32 %compare.local.base766, 6, !dbg !24
  %compare.low768 = icmp ugt i32 %compare.low.base742, %compare.local767, !dbg !24
  %compare.result769 = select i1 %compare.high.equal740, i1 %compare.low768, i1 %compare.high.order741, !dbg !24
  %325 = select i1 %compare.result769, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base770 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local771 = or i32 %compare.local.base770, 7, !dbg !24
  %compare.low772 = icmp ugt i32 %compare.low.base742, %compare.local771, !dbg !24
  %compare.result773 = select i1 %compare.high.equal740, i1 %compare.low772, i1 %compare.high.order741, !dbg !24
  %326 = select i1 %compare.result773, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base774 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local775 = or i32 %compare.local.base774, 8, !dbg !24
  %compare.low776 = icmp ugt i32 %compare.low.base742, %compare.local775, !dbg !24
  %compare.result777 = select i1 %compare.high.equal740, i1 %compare.low776, i1 %compare.high.order741, !dbg !24
  %327 = select i1 %compare.result777, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base778 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local779 = or i32 %compare.local.base778, 9, !dbg !24
  %compare.low780 = icmp ugt i32 %compare.low.base742, %compare.local779, !dbg !24
  %compare.result781 = select i1 %compare.high.equal740, i1 %compare.low780, i1 %compare.high.order741, !dbg !24
  %328 = select i1 %compare.result781, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base782 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local783 = or i32 %compare.local.base782, 10, !dbg !24
  %compare.low784 = icmp ugt i32 %compare.low.base742, %compare.local783, !dbg !24
  %compare.result785 = select i1 %compare.high.equal740, i1 %compare.low784, i1 %compare.high.order741, !dbg !24
  %329 = select i1 %compare.result785, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base786 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local787 = or i32 %compare.local.base786, 11, !dbg !24
  %compare.low788 = icmp ugt i32 %compare.low.base742, %compare.local787, !dbg !24
  %compare.result789 = select i1 %compare.high.equal740, i1 %compare.low788, i1 %compare.high.order741, !dbg !24
  %330 = select i1 %compare.result789, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base790 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local791 = or i32 %compare.local.base790, 12, !dbg !24
  %compare.low792 = icmp ugt i32 %compare.low.base742, %compare.local791, !dbg !24
  %compare.result793 = select i1 %compare.high.equal740, i1 %compare.low792, i1 %compare.high.order741, !dbg !24
  %331 = select i1 %compare.result793, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base794 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local795 = or i32 %compare.local.base794, 13, !dbg !24
  %compare.low796 = icmp ugt i32 %compare.low.base742, %compare.local795, !dbg !24
  %compare.result797 = select i1 %compare.high.equal740, i1 %compare.low796, i1 %compare.high.order741, !dbg !24
  %332 = select i1 %compare.result797, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base798 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local799 = or i32 %compare.local.base798, 14, !dbg !24
  %compare.low800 = icmp ugt i32 %compare.low.base742, %compare.local799, !dbg !24
  %compare.result801 = select i1 %compare.high.equal740, i1 %compare.low800, i1 %compare.high.order741, !dbg !24
  %333 = select i1 %compare.result801, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base802 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local803 = or i32 %compare.local.base802, 15, !dbg !24
  %compare.low804 = icmp ugt i32 %compare.low.base742, %compare.local803, !dbg !24
  %compare.result805 = select i1 %compare.high.equal740, i1 %compare.low804, i1 %compare.high.order741, !dbg !24
  %334 = select i1 %compare.result805, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base806 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local807 = or i32 %compare.local.base806, 64, !dbg !24
  %compare.low808 = icmp ugt i32 %compare.low.base742, %compare.local807, !dbg !24
  %compare.result809 = select i1 %compare.high.equal740, i1 %compare.low808, i1 %compare.high.order741, !dbg !24
  %335 = select i1 %compare.result809, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base810 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local811 = or i32 %compare.local.base810, 65, !dbg !24
  %compare.low812 = icmp ugt i32 %compare.low.base742, %compare.local811, !dbg !24
  %compare.result813 = select i1 %compare.high.equal740, i1 %compare.low812, i1 %compare.high.order741, !dbg !24
  %336 = select i1 %compare.result813, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base814 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local815 = or i32 %compare.local.base814, 66, !dbg !24
  %compare.low816 = icmp ugt i32 %compare.low.base742, %compare.local815, !dbg !24
  %compare.result817 = select i1 %compare.high.equal740, i1 %compare.low816, i1 %compare.high.order741, !dbg !24
  %337 = select i1 %compare.result817, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base818 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local819 = or i32 %compare.local.base818, 67, !dbg !24
  %compare.low820 = icmp ugt i32 %compare.low.base742, %compare.local819, !dbg !24
  %compare.result821 = select i1 %compare.high.equal740, i1 %compare.low820, i1 %compare.high.order741, !dbg !24
  %338 = select i1 %compare.result821, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base822 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local823 = or i32 %compare.local.base822, 68, !dbg !24
  %compare.low824 = icmp ugt i32 %compare.low.base742, %compare.local823, !dbg !24
  %compare.result825 = select i1 %compare.high.equal740, i1 %compare.low824, i1 %compare.high.order741, !dbg !24
  %339 = select i1 %compare.result825, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base826 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local827 = or i32 %compare.local.base826, 69, !dbg !24
  %compare.low828 = icmp ugt i32 %compare.low.base742, %compare.local827, !dbg !24
  %compare.result829 = select i1 %compare.high.equal740, i1 %compare.low828, i1 %compare.high.order741, !dbg !24
  %340 = select i1 %compare.result829, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base830 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local831 = or i32 %compare.local.base830, 70, !dbg !24
  %compare.low832 = icmp ugt i32 %compare.low.base742, %compare.local831, !dbg !24
  %compare.result833 = select i1 %compare.high.equal740, i1 %compare.low832, i1 %compare.high.order741, !dbg !24
  %341 = select i1 %compare.result833, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base834 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local835 = or i32 %compare.local.base834, 71, !dbg !24
  %compare.low836 = icmp ugt i32 %compare.low.base742, %compare.local835, !dbg !24
  %compare.result837 = select i1 %compare.high.equal740, i1 %compare.low836, i1 %compare.high.order741, !dbg !24
  %342 = select i1 %compare.result837, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base838 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local839 = or i32 %compare.local.base838, 72, !dbg !24
  %compare.low840 = icmp ugt i32 %compare.low.base742, %compare.local839, !dbg !24
  %compare.result841 = select i1 %compare.high.equal740, i1 %compare.low840, i1 %compare.high.order741, !dbg !24
  %343 = select i1 %compare.result841, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base842 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local843 = or i32 %compare.local.base842, 73, !dbg !24
  %compare.low844 = icmp ugt i32 %compare.low.base742, %compare.local843, !dbg !24
  %compare.result845 = select i1 %compare.high.equal740, i1 %compare.low844, i1 %compare.high.order741, !dbg !24
  %344 = select i1 %compare.result845, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base846 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local847 = or i32 %compare.local.base846, 74, !dbg !24
  %compare.low848 = icmp ugt i32 %compare.low.base742, %compare.local847, !dbg !24
  %compare.result849 = select i1 %compare.high.equal740, i1 %compare.low848, i1 %compare.high.order741, !dbg !24
  %345 = select i1 %compare.result849, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base850 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local851 = or i32 %compare.local.base850, 75, !dbg !24
  %compare.low852 = icmp ugt i32 %compare.low.base742, %compare.local851, !dbg !24
  %compare.result853 = select i1 %compare.high.equal740, i1 %compare.low852, i1 %compare.high.order741, !dbg !24
  %346 = select i1 %compare.result853, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base854 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local855 = or i32 %compare.local.base854, 76, !dbg !24
  %compare.low856 = icmp ugt i32 %compare.low.base742, %compare.local855, !dbg !24
  %compare.result857 = select i1 %compare.high.equal740, i1 %compare.low856, i1 %compare.high.order741, !dbg !24
  %347 = select i1 %compare.result857, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base858 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local859 = or i32 %compare.local.base858, 77, !dbg !24
  %compare.low860 = icmp ugt i32 %compare.low.base742, %compare.local859, !dbg !24
  %compare.result861 = select i1 %compare.high.equal740, i1 %compare.low860, i1 %compare.high.order741, !dbg !24
  %348 = select i1 %compare.result861, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base862 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local863 = or i32 %compare.local.base862, 78, !dbg !24
  %compare.low864 = icmp ugt i32 %compare.low.base742, %compare.local863, !dbg !24
  %compare.result865 = select i1 %compare.high.equal740, i1 %compare.low864, i1 %compare.high.order741, !dbg !24
  %349 = select i1 %compare.result865, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base866 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local867 = or i32 %compare.local.base866, 79, !dbg !24
  %compare.low868 = icmp ugt i32 %compare.low.base742, %compare.local867, !dbg !24
  %compare.result869 = select i1 %compare.high.equal740, i1 %compare.low868, i1 %compare.high.order741, !dbg !24
  %350 = select i1 %compare.result869, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %351 = insertelement <32 x float> poison, float %284, i64 0, !dbg !14
  %352 = insertelement <32 x float> %351, float %285, i64 1, !dbg !14
  %353 = insertelement <32 x float> %352, float %286, i64 2, !dbg !14
  %354 = insertelement <32 x float> %353, float %287, i64 3, !dbg !14
  %355 = insertelement <32 x float> %354, float %288, i64 4, !dbg !14
  %356 = insertelement <32 x float> %355, float %289, i64 5, !dbg !14
  %357 = insertelement <32 x float> %356, float %290, i64 6, !dbg !14
  %358 = insertelement <32 x float> %357, float %291, i64 7, !dbg !14
  %359 = insertelement <32 x float> %358, float %292, i64 8, !dbg !14
  %360 = insertelement <32 x float> %359, float %293, i64 9, !dbg !14
  %361 = insertelement <32 x float> %360, float %294, i64 10, !dbg !14
  %362 = insertelement <32 x float> %361, float %295, i64 11, !dbg !14
  %363 = insertelement <32 x float> %362, float %296, i64 12, !dbg !14
  %364 = insertelement <32 x float> %363, float %297, i64 13, !dbg !14
  %365 = insertelement <32 x float> %364, float %298, i64 14, !dbg !14
  %366 = insertelement <32 x float> %365, float %299, i64 15, !dbg !14
  %367 = insertelement <32 x float> %366, float %300, i64 16, !dbg !14
  %368 = insertelement <32 x float> %367, float %301, i64 17, !dbg !14
  %369 = insertelement <32 x float> %368, float %302, i64 18, !dbg !14
  %370 = insertelement <32 x float> %369, float %303, i64 19, !dbg !14
  %371 = insertelement <32 x float> %370, float %304, i64 20, !dbg !14
  %372 = insertelement <32 x float> %371, float %305, i64 21, !dbg !14
  %373 = insertelement <32 x float> %372, float %306, i64 22, !dbg !14
  %374 = insertelement <32 x float> %373, float %307, i64 23, !dbg !14
  %375 = insertelement <32 x float> %374, float %308, i64 24, !dbg !14
  %376 = insertelement <32 x float> %375, float %309, i64 25, !dbg !14
  %377 = insertelement <32 x float> %376, float %310, i64 26, !dbg !14
  %378 = insertelement <32 x float> %377, float %311, i64 27, !dbg !14
  %379 = insertelement <32 x float> %378, float %312, i64 28, !dbg !14
  %380 = insertelement <32 x float> %379, float %313, i64 29, !dbg !14
  %381 = insertelement <32 x float> %380, float %314, i64 30, !dbg !14
  %382 = insertelement <32 x float> %381, float %315, i64 31, !dbg !14
  %383 = insertvalue [2 x <32 x float>] poison, <32 x float> %382, 0, !dbg !14
  %384 = insertelement <32 x float> poison, float %319, i64 0, !dbg !14
  %385 = insertelement <32 x float> %384, float %320, i64 1, !dbg !14
  %386 = insertelement <32 x float> %385, float %321, i64 2, !dbg !14
  %387 = insertelement <32 x float> %386, float %322, i64 3, !dbg !14
  %388 = insertelement <32 x float> %387, float %323, i64 4, !dbg !14
  %389 = insertelement <32 x float> %388, float %324, i64 5, !dbg !14
  %390 = insertelement <32 x float> %389, float %325, i64 6, !dbg !14
  %391 = insertelement <32 x float> %390, float %326, i64 7, !dbg !14
  %392 = insertelement <32 x float> %391, float %327, i64 8, !dbg !14
  %393 = insertelement <32 x float> %392, float %328, i64 9, !dbg !14
  %394 = insertelement <32 x float> %393, float %329, i64 10, !dbg !14
  %395 = insertelement <32 x float> %394, float %330, i64 11, !dbg !14
  %396 = insertelement <32 x float> %395, float %331, i64 12, !dbg !14
  %397 = insertelement <32 x float> %396, float %332, i64 13, !dbg !14
  %398 = insertelement <32 x float> %397, float %333, i64 14, !dbg !14
  %399 = insertelement <32 x float> %398, float %334, i64 15, !dbg !14
  %400 = insertelement <32 x float> %399, float %335, i64 16, !dbg !14
  %401 = insertelement <32 x float> %400, float %336, i64 17, !dbg !14
  %402 = insertelement <32 x float> %401, float %337, i64 18, !dbg !14
  %403 = insertelement <32 x float> %402, float %338, i64 19, !dbg !14
  %404 = insertelement <32 x float> %403, float %339, i64 20, !dbg !14
  %405 = insertelement <32 x float> %404, float %340, i64 21, !dbg !14
  %406 = insertelement <32 x float> %405, float %341, i64 22, !dbg !14
  %407 = insertelement <32 x float> %406, float %342, i64 23, !dbg !14
  %408 = insertelement <32 x float> %407, float %343, i64 24, !dbg !14
  %409 = insertelement <32 x float> %408, float %344, i64 25, !dbg !14
  %410 = insertelement <32 x float> %409, float %345, i64 26, !dbg !14
  %411 = insertelement <32 x float> %410, float %346, i64 27, !dbg !14
  %412 = insertelement <32 x float> %411, float %347, i64 28, !dbg !14
  %413 = insertelement <32 x float> %412, float %348, i64 29, !dbg !14
  %414 = insertelement <32 x float> %413, float %349, i64 30, !dbg !14
  %415 = insertelement <32 x float> %414, float %350, i64 31, !dbg !14
  %416 = insertvalue [2 x <32 x float>] %383, <32 x float> %415, 1, !dbg !14
  br label %417, !dbg !14

417:                                              ; preds = %272, %273
  %418 = phi [2 x <32 x float>] [ %416, %273 ], [ zeroinitializer, %272 ], !dbg !14
  br label %419, !dbg !14

419:                                              ; preds = %417
  %420 = extractvalue [2 x <32 x float>] %39, 0, !dbg !26
  %421 = extractvalue [2 x <32 x float>] %418, 0, !dbg !26
  %422 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %423 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %424 = fadd <4 x float> %422, %423, !dbg !26
  %425 = extractelement <4 x float> %424, i64 0, !dbg !27
  %426 = call float @llvm.amdgcn.exp2.f32(float %425), !dbg !27
  %427 = insertelement <4 x float> poison, float %426, i64 0, !dbg !29
  %428 = extractelement <4 x float> %424, i64 1, !dbg !27
  %429 = call float @llvm.amdgcn.exp2.f32(float %428), !dbg !27
  %430 = insertelement <4 x float> %427, float %429, i64 1, !dbg !29
  %431 = fadd float %426, %429, !dbg !28
  %432 = extractelement <4 x float> %424, i64 2, !dbg !27
  %433 = call float @llvm.amdgcn.exp2.f32(float %432), !dbg !27
  %434 = insertelement <4 x float> %430, float %433, i64 2, !dbg !29
  %435 = fadd float %431, %433, !dbg !28
  %436 = extractelement <4 x float> %424, i64 3, !dbg !27
  %437 = call float @llvm.amdgcn.exp2.f32(float %436), !dbg !27
  %438 = insertelement <4 x float> %434, float %437, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %438) #9, !dbg !29, !frisk.register_lifetimes !1
  %439 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %439) #9, !dbg !28, !frisk.register_lifetimes !1
  %440 = fadd float %435, %437, !dbg !28
  %441 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %442 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %443 = fadd <4 x float> %441, %442, !dbg !26
  %444 = extractelement <4 x float> %443, i64 0, !dbg !27
  %445 = call float @llvm.amdgcn.exp2.f32(float %444), !dbg !27
  %446 = insertelement <4 x float> poison, float %445, i64 0, !dbg !29
  %447 = fadd float %440, %445, !dbg !28
  %448 = extractelement <4 x float> %443, i64 1, !dbg !27
  %449 = call float @llvm.amdgcn.exp2.f32(float %448), !dbg !27
  %450 = insertelement <4 x float> %446, float %449, i64 1, !dbg !29
  %451 = fadd float %447, %449, !dbg !28
  %452 = extractelement <4 x float> %443, i64 2, !dbg !27
  %453 = call float @llvm.amdgcn.exp2.f32(float %452), !dbg !27
  %454 = insertelement <4 x float> %450, float %453, i64 2, !dbg !29
  %455 = fadd float %451, %453, !dbg !28
  %456 = extractelement <4 x float> %443, i64 3, !dbg !27
  %457 = call float @llvm.amdgcn.exp2.f32(float %456), !dbg !27
  %458 = insertelement <4 x float> %454, float %457, i64 3, !dbg !29
  %math.result100 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %458) #9, !dbg !29, !frisk.register_lifetimes !1
  %459 = fptrunc <4 x float> %math.result100 to <4 x half>, !dbg !29
  %math.packed101 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %459) #9, !dbg !28, !frisk.register_lifetimes !1
  %460 = fadd float %455, %457, !dbg !28
  %461 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %462 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %463 = fadd <4 x float> %461, %462, !dbg !26
  %464 = extractelement <4 x float> %463, i64 0, !dbg !27
  %465 = call float @llvm.amdgcn.exp2.f32(float %464), !dbg !27
  %466 = insertelement <4 x float> poison, float %465, i64 0, !dbg !29
  %467 = fadd float %460, %465, !dbg !28
  %468 = extractelement <4 x float> %463, i64 1, !dbg !27
  %469 = call float @llvm.amdgcn.exp2.f32(float %468), !dbg !27
  %470 = insertelement <4 x float> %466, float %469, i64 1, !dbg !29
  %471 = fadd float %467, %469, !dbg !28
  %472 = extractelement <4 x float> %463, i64 2, !dbg !27
  %473 = call float @llvm.amdgcn.exp2.f32(float %472), !dbg !27
  %474 = insertelement <4 x float> %470, float %473, i64 2, !dbg !29
  %475 = fadd float %471, %473, !dbg !28
  %476 = extractelement <4 x float> %463, i64 3, !dbg !27
  %477 = call float @llvm.amdgcn.exp2.f32(float %476), !dbg !27
  %478 = insertelement <4 x float> %474, float %477, i64 3, !dbg !29
  %math.result102 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %478) #9, !dbg !29, !frisk.register_lifetimes !1
  %479 = fptrunc <4 x float> %math.result102 to <4 x half>, !dbg !29
  %math.packed103 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %479) #9, !dbg !28, !frisk.register_lifetimes !1
  %480 = fadd float %475, %477, !dbg !28
  %481 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %482 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %483 = fadd <4 x float> %481, %482, !dbg !26
  %484 = extractelement <4 x float> %483, i64 0, !dbg !27
  %485 = call float @llvm.amdgcn.exp2.f32(float %484), !dbg !27
  %486 = insertelement <4 x float> poison, float %485, i64 0, !dbg !29
  %487 = fadd float %480, %485, !dbg !28
  %488 = extractelement <4 x float> %483, i64 1, !dbg !27
  %489 = call float @llvm.amdgcn.exp2.f32(float %488), !dbg !27
  %490 = insertelement <4 x float> %486, float %489, i64 1, !dbg !29
  %491 = fadd float %487, %489, !dbg !28
  %492 = extractelement <4 x float> %483, i64 2, !dbg !27
  %493 = call float @llvm.amdgcn.exp2.f32(float %492), !dbg !27
  %494 = insertelement <4 x float> %490, float %493, i64 2, !dbg !29
  %495 = fadd float %491, %493, !dbg !28
  %496 = extractelement <4 x float> %483, i64 3, !dbg !27
  %497 = call float @llvm.amdgcn.exp2.f32(float %496), !dbg !27
  %498 = insertelement <4 x float> %494, float %497, i64 3, !dbg !29
  %math.result104 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %498) #9, !dbg !29, !frisk.register_lifetimes !1
  %499 = fptrunc <4 x float> %math.result104 to <4 x half>, !dbg !29
  %math.packed105 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %499) #9, !dbg !28, !frisk.register_lifetimes !1
  %500 = fadd float %495, %497, !dbg !28
  %501 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %502 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %503 = fadd <4 x float> %501, %502, !dbg !26
  %504 = extractelement <4 x float> %503, i64 0, !dbg !27
  %505 = call float @llvm.amdgcn.exp2.f32(float %504), !dbg !27
  %506 = insertelement <4 x float> poison, float %505, i64 0, !dbg !29
  %507 = fadd float %500, %505, !dbg !28
  %508 = extractelement <4 x float> %503, i64 1, !dbg !27
  %509 = call float @llvm.amdgcn.exp2.f32(float %508), !dbg !27
  %510 = insertelement <4 x float> %506, float %509, i64 1, !dbg !29
  %511 = fadd float %507, %509, !dbg !28
  %512 = extractelement <4 x float> %503, i64 2, !dbg !27
  %513 = call float @llvm.amdgcn.exp2.f32(float %512), !dbg !27
  %514 = insertelement <4 x float> %510, float %513, i64 2, !dbg !29
  %515 = fadd float %511, %513, !dbg !28
  %516 = extractelement <4 x float> %503, i64 3, !dbg !27
  %517 = call float @llvm.amdgcn.exp2.f32(float %516), !dbg !27
  %518 = insertelement <4 x float> %514, float %517, i64 3, !dbg !29
  %math.result106 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %518) #9, !dbg !29, !frisk.register_lifetimes !1
  %519 = fptrunc <4 x float> %math.result106 to <4 x half>, !dbg !29
  %math.packed107 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %519) #9, !dbg !28, !frisk.register_lifetimes !1
  %520 = fadd float %515, %517, !dbg !28
  %521 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %522 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %523 = fadd <4 x float> %521, %522, !dbg !26
  %524 = extractelement <4 x float> %523, i64 0, !dbg !27
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !27
  %526 = insertelement <4 x float> poison, float %525, i64 0, !dbg !29
  %527 = fadd float %520, %525, !dbg !28
  %528 = extractelement <4 x float> %523, i64 1, !dbg !27
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !27
  %530 = insertelement <4 x float> %526, float %529, i64 1, !dbg !29
  %531 = fadd float %527, %529, !dbg !28
  %532 = extractelement <4 x float> %523, i64 2, !dbg !27
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !27
  %534 = insertelement <4 x float> %530, float %533, i64 2, !dbg !29
  %535 = fadd float %531, %533, !dbg !28
  %536 = extractelement <4 x float> %523, i64 3, !dbg !27
  %537 = call float @llvm.amdgcn.exp2.f32(float %536), !dbg !27
  %538 = insertelement <4 x float> %534, float %537, i64 3, !dbg !29
  %math.result108 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %538) #9, !dbg !29, !frisk.register_lifetimes !1
  %539 = fptrunc <4 x float> %math.result108 to <4 x half>, !dbg !29
  %math.packed109 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %539) #9, !dbg !28, !frisk.register_lifetimes !1
  %540 = fadd float %535, %537, !dbg !28
  %541 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %542 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %543 = fadd <4 x float> %541, %542, !dbg !26
  %544 = extractelement <4 x float> %543, i64 0, !dbg !27
  %545 = call float @llvm.amdgcn.exp2.f32(float %544), !dbg !27
  %546 = insertelement <4 x float> poison, float %545, i64 0, !dbg !29
  %547 = fadd float %540, %545, !dbg !28
  %548 = extractelement <4 x float> %543, i64 1, !dbg !27
  %549 = call float @llvm.amdgcn.exp2.f32(float %548), !dbg !27
  %550 = insertelement <4 x float> %546, float %549, i64 1, !dbg !29
  %551 = fadd float %547, %549, !dbg !28
  %552 = extractelement <4 x float> %543, i64 2, !dbg !27
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !27
  %554 = insertelement <4 x float> %550, float %553, i64 2, !dbg !29
  %555 = fadd float %551, %553, !dbg !28
  %556 = extractelement <4 x float> %543, i64 3, !dbg !27
  %557 = call float @llvm.amdgcn.exp2.f32(float %556), !dbg !27
  %558 = insertelement <4 x float> %554, float %557, i64 3, !dbg !29
  %math.result110 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %558) #9, !dbg !29, !frisk.register_lifetimes !1
  %559 = fptrunc <4 x float> %math.result110 to <4 x half>, !dbg !29
  %math.packed111 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %559) #9, !dbg !28, !frisk.register_lifetimes !1
  %560 = fadd float %555, %557, !dbg !28
  %561 = shufflevector <32 x float> %420, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %562 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %563 = fadd <4 x float> %561, %562, !dbg !26
  %564 = extractelement <4 x float> %563, i64 0, !dbg !27
  %565 = call float @llvm.amdgcn.exp2.f32(float %564), !dbg !27
  %566 = insertelement <4 x float> poison, float %565, i64 0, !dbg !29
  %567 = fadd float %560, %565, !dbg !28
  %568 = extractelement <4 x float> %563, i64 1, !dbg !27
  %569 = call float @llvm.amdgcn.exp2.f32(float %568), !dbg !27
  %570 = insertelement <4 x float> %566, float %569, i64 1, !dbg !29
  %571 = fadd float %567, %569, !dbg !28
  %572 = extractelement <4 x float> %563, i64 2, !dbg !27
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !27
  %574 = insertelement <4 x float> %570, float %573, i64 2, !dbg !29
  %575 = fadd float %571, %573, !dbg !28
  %576 = extractelement <4 x float> %563, i64 3, !dbg !27
  %577 = call float @llvm.amdgcn.exp2.f32(float %576), !dbg !27
  %578 = insertelement <4 x float> %574, float %577, i64 3, !dbg !29
  %math.result112 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %578) #9, !dbg !29, !frisk.register_lifetimes !1
  %579 = fptrunc <4 x float> %math.result112 to <4 x half>, !dbg !29
  %math.packed113 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %579) #9, !dbg !28, !frisk.register_lifetimes !1
  %580 = fadd float %575, %577, !dbg !28
  %581 = extractvalue [2 x <32 x float>] %39, 1, !dbg !26
  %582 = extractvalue [2 x <32 x float>] %418, 1, !dbg !26
  %583 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %584 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %585 = fadd <4 x float> %583, %584, !dbg !26
  %586 = extractelement <4 x float> %585, i64 0, !dbg !27
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !27
  %588 = insertelement <4 x float> poison, float %587, i64 0, !dbg !29
  %589 = extractelement <4 x float> %585, i64 1, !dbg !27
  %590 = call float @llvm.amdgcn.exp2.f32(float %589), !dbg !27
  %591 = insertelement <4 x float> %588, float %590, i64 1, !dbg !29
  %592 = fadd float %587, %590, !dbg !28
  %593 = extractelement <4 x float> %585, i64 2, !dbg !27
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !27
  %595 = insertelement <4 x float> %591, float %594, i64 2, !dbg !29
  %596 = fadd float %592, %594, !dbg !28
  %597 = extractelement <4 x float> %585, i64 3, !dbg !27
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !27
  %599 = insertelement <4 x float> %595, float %598, i64 3, !dbg !29
  %math.result114 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %599) #9, !dbg !29, !frisk.register_lifetimes !1
  %600 = fptrunc <4 x float> %math.result114 to <4 x half>, !dbg !29
  %math.packed115 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %600) #9, !dbg !28, !frisk.register_lifetimes !1
  %601 = fadd float %596, %598, !dbg !28
  %602 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %603 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %604 = fadd <4 x float> %602, %603, !dbg !26
  %605 = extractelement <4 x float> %604, i64 0, !dbg !27
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !27
  %607 = insertelement <4 x float> poison, float %606, i64 0, !dbg !29
  %608 = fadd float %601, %606, !dbg !28
  %609 = extractelement <4 x float> %604, i64 1, !dbg !27
  %610 = call float @llvm.amdgcn.exp2.f32(float %609), !dbg !27
  %611 = insertelement <4 x float> %607, float %610, i64 1, !dbg !29
  %612 = fadd float %608, %610, !dbg !28
  %613 = extractelement <4 x float> %604, i64 2, !dbg !27
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !27
  %615 = insertelement <4 x float> %611, float %614, i64 2, !dbg !29
  %616 = fadd float %612, %614, !dbg !28
  %617 = extractelement <4 x float> %604, i64 3, !dbg !27
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !27
  %619 = insertelement <4 x float> %615, float %618, i64 3, !dbg !29
  %math.result116 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %619) #9, !dbg !29, !frisk.register_lifetimes !1
  %620 = fptrunc <4 x float> %math.result116 to <4 x half>, !dbg !29
  %math.packed117 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %620) #9, !dbg !28, !frisk.register_lifetimes !1
  %621 = fadd float %616, %618, !dbg !28
  %622 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %623 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %624 = fadd <4 x float> %622, %623, !dbg !26
  %625 = extractelement <4 x float> %624, i64 0, !dbg !27
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !27
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !29
  %628 = fadd float %621, %626, !dbg !28
  %629 = extractelement <4 x float> %624, i64 1, !dbg !27
  %630 = call float @llvm.amdgcn.exp2.f32(float %629), !dbg !27
  %631 = insertelement <4 x float> %627, float %630, i64 1, !dbg !29
  %632 = fadd float %628, %630, !dbg !28
  %633 = extractelement <4 x float> %624, i64 2, !dbg !27
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !27
  %635 = insertelement <4 x float> %631, float %634, i64 2, !dbg !29
  %636 = fadd float %632, %634, !dbg !28
  %637 = extractelement <4 x float> %624, i64 3, !dbg !27
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !27
  %639 = insertelement <4 x float> %635, float %638, i64 3, !dbg !29
  %math.result118 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %639) #9, !dbg !29, !frisk.register_lifetimes !1
  %640 = fptrunc <4 x float> %math.result118 to <4 x half>, !dbg !29
  %math.packed119 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %640) #9, !dbg !28, !frisk.register_lifetimes !1
  %641 = fadd float %636, %638, !dbg !28
  %642 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %643 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %644 = fadd <4 x float> %642, %643, !dbg !26
  %645 = extractelement <4 x float> %644, i64 0, !dbg !27
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !27
  %647 = insertelement <4 x float> poison, float %646, i64 0, !dbg !29
  %648 = fadd float %641, %646, !dbg !28
  %649 = extractelement <4 x float> %644, i64 1, !dbg !27
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !27
  %651 = insertelement <4 x float> %647, float %650, i64 1, !dbg !29
  %652 = fadd float %648, %650, !dbg !28
  %653 = extractelement <4 x float> %644, i64 2, !dbg !27
  %654 = call float @llvm.amdgcn.exp2.f32(float %653), !dbg !27
  %655 = insertelement <4 x float> %651, float %654, i64 2, !dbg !29
  %656 = fadd float %652, %654, !dbg !28
  %657 = extractelement <4 x float> %644, i64 3, !dbg !27
  %658 = call float @llvm.amdgcn.exp2.f32(float %657), !dbg !27
  %659 = insertelement <4 x float> %655, float %658, i64 3, !dbg !29
  %math.result120 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %659) #9, !dbg !29, !frisk.register_lifetimes !1
  %660 = fptrunc <4 x float> %math.result120 to <4 x half>, !dbg !29
  %math.packed121 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %660) #9, !dbg !28, !frisk.register_lifetimes !1
  %661 = fadd float %656, %658, !dbg !28
  %662 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %663 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %664 = fadd <4 x float> %662, %663, !dbg !26
  %665 = extractelement <4 x float> %664, i64 0, !dbg !27
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !27
  %667 = insertelement <4 x float> poison, float %666, i64 0, !dbg !29
  %668 = fadd float %661, %666, !dbg !28
  %669 = extractelement <4 x float> %664, i64 1, !dbg !27
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !27
  %671 = insertelement <4 x float> %667, float %670, i64 1, !dbg !29
  %672 = fadd float %668, %670, !dbg !28
  %673 = extractelement <4 x float> %664, i64 2, !dbg !27
  %674 = call float @llvm.amdgcn.exp2.f32(float %673), !dbg !27
  %675 = insertelement <4 x float> %671, float %674, i64 2, !dbg !29
  %676 = fadd float %672, %674, !dbg !28
  %677 = extractelement <4 x float> %664, i64 3, !dbg !27
  %678 = call float @llvm.amdgcn.exp2.f32(float %677), !dbg !27
  %679 = insertelement <4 x float> %675, float %678, i64 3, !dbg !29
  %math.result122 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %679) #9, !dbg !29, !frisk.register_lifetimes !1
  %680 = fptrunc <4 x float> %math.result122 to <4 x half>, !dbg !29
  %math.packed123 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %680) #9, !dbg !28, !frisk.register_lifetimes !1
  %681 = fadd float %676, %678, !dbg !28
  %682 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %683 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %684 = fadd <4 x float> %682, %683, !dbg !26
  %685 = extractelement <4 x float> %684, i64 0, !dbg !27
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !27
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !29
  %688 = fadd float %681, %686, !dbg !28
  %689 = extractelement <4 x float> %684, i64 1, !dbg !27
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !27
  %691 = insertelement <4 x float> %687, float %690, i64 1, !dbg !29
  %692 = fadd float %688, %690, !dbg !28
  %693 = extractelement <4 x float> %684, i64 2, !dbg !27
  %694 = call float @llvm.amdgcn.exp2.f32(float %693), !dbg !27
  %695 = insertelement <4 x float> %691, float %694, i64 2, !dbg !29
  %696 = fadd float %692, %694, !dbg !28
  %697 = extractelement <4 x float> %684, i64 3, !dbg !27
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !27
  %699 = insertelement <4 x float> %695, float %698, i64 3, !dbg !29
  %math.result124 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %699) #9, !dbg !29, !frisk.register_lifetimes !1
  %700 = fptrunc <4 x float> %math.result124 to <4 x half>, !dbg !29
  %math.packed125 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %700) #9, !dbg !28, !frisk.register_lifetimes !1
  %701 = fadd float %696, %698, !dbg !28
  %702 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %703 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %704 = fadd <4 x float> %702, %703, !dbg !26
  %705 = extractelement <4 x float> %704, i64 0, !dbg !27
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !27
  %707 = insertelement <4 x float> poison, float %706, i64 0, !dbg !29
  %708 = fadd float %701, %706, !dbg !28
  %709 = extractelement <4 x float> %704, i64 1, !dbg !27
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !27
  %711 = insertelement <4 x float> %707, float %710, i64 1, !dbg !29
  %712 = fadd float %708, %710, !dbg !28
  %713 = extractelement <4 x float> %704, i64 2, !dbg !27
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !27
  %715 = insertelement <4 x float> %711, float %714, i64 2, !dbg !29
  %716 = fadd float %712, %714, !dbg !28
  %717 = extractelement <4 x float> %704, i64 3, !dbg !27
  %718 = call float @llvm.amdgcn.exp2.f32(float %717), !dbg !27
  %719 = insertelement <4 x float> %715, float %718, i64 3, !dbg !29
  %math.result126 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %719) #9, !dbg !29, !frisk.register_lifetimes !1
  %720 = fptrunc <4 x float> %math.result126 to <4 x half>, !dbg !29
  %math.packed127 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %720) #9, !dbg !28, !frisk.register_lifetimes !1
  %721 = fadd float %716, %718, !dbg !28
  %722 = shufflevector <32 x float> %581, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %723 = shufflevector <32 x float> %582, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %724 = fadd <4 x float> %722, %723, !dbg !26
  %725 = extractelement <4 x float> %724, i64 0, !dbg !27
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !27
  %727 = insertelement <4 x float> poison, float %726, i64 0, !dbg !29
  %728 = fadd float %721, %726, !dbg !28
  %729 = extractelement <4 x float> %724, i64 1, !dbg !27
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !27
  %731 = insertelement <4 x float> %727, float %730, i64 1, !dbg !29
  %732 = fadd float %728, %730, !dbg !28
  %733 = extractelement <4 x float> %724, i64 2, !dbg !27
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !27
  %735 = insertelement <4 x float> %731, float %734, i64 2, !dbg !29
  %736 = fadd float %732, %734, !dbg !28
  %737 = extractelement <4 x float> %724, i64 3, !dbg !27
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !27
  %739 = insertelement <4 x float> %735, float %738, i64 3, !dbg !29
  %math.result128 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %739) #9, !dbg !29, !frisk.register_lifetimes !1
  %740 = fptrunc <4 x float> %math.result128 to <4 x half>, !dbg !29
  %math.packed129 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %740) #9, !dbg !28, !frisk.register_lifetimes !1
  %741 = fadd float %736, %738, !dbg !28
  %742 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %743 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %742), !dbg !28
  %744 = add nuw nsw i32 %743, 64, !dbg !28
  %745 = xor i32 %743, 16, !dbg !28
  %746 = and i32 %744, 192, !dbg !28
  %747 = icmp samesign ult i32 %745, %746, !dbg !28
  %748 = select i1 %747, i32 %745, i32 %743, !dbg !28
  %749 = shl nuw nsw i32 %748, 2, !dbg !28
  %750 = bitcast float %580 to i32, !dbg !28
  %751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %750), !dbg !28
  %752 = bitcast i32 %751 to float, !dbg !28
  %753 = fadd float %580, %752, !dbg !28
  %754 = xor i32 %743, 32, !dbg !28
  %755 = icmp samesign ult i32 %754, %746, !dbg !28
  %756 = select i1 %755, i32 %754, i32 %743, !dbg !28
  %757 = shl nuw nsw i32 %756, 2, !dbg !28
  %758 = bitcast float %753 to i32, !dbg !28
  %759 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %758), !dbg !28
  %760 = bitcast i32 %759 to float, !dbg !28
  %761 = fadd float %753, %760, !dbg !28
  %762 = extractvalue [2 x <1 x float>] %28, 0, !dbg !28
  %763 = extractelement <1 x float> %762, i64 0, !dbg !28
  %764 = fadd float %761, %763, !dbg !28
  %765 = insertelement <1 x float> poison, float %764, i64 0, !dbg !16
  %766 = insertvalue [2 x <1 x float>] poison, <1 x float> %765, 0, !dbg !16
  %767 = bitcast float %741 to i32, !dbg !28
  %768 = call i32 @llvm.amdgcn.ds.bpermute(i32 %749, i32 %767), !dbg !28
  %769 = bitcast i32 %768 to float, !dbg !28
  %770 = fadd float %741, %769, !dbg !28
  %771 = bitcast float %770 to i32, !dbg !28
  %772 = call i32 @llvm.amdgcn.ds.bpermute(i32 %757, i32 %771), !dbg !28
  %773 = bitcast i32 %772 to float, !dbg !28
  %774 = fadd float %770, %773, !dbg !28
  %775 = extractvalue [2 x <1 x float>] %28, 1, !dbg !28
  %776 = extractelement <1 x float> %775, i64 0, !dbg !28
  %777 = fadd float %774, %776, !dbg !28
  %778 = insertelement <1 x float> poison, float %777, i64 0, !dbg !16
  %779 = insertvalue [2 x <1 x float>] %766, <1 x float> %778, 1, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread321 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !34, !frisk.register_lifetimes !1
  %address.local322 = shl nuw nsw i32 %address.thread321, 3, !dbg !15
  %address.local323 = and i32 %address.local322, 120, !dbg !15
  %address.local324 = shl nuw nsw i32 %address.thread321, 5, !dbg !15
  %address.local325 = and i32 %address.local324, 1536, !dbg !15
  %address.local326 = or disjoint i32 %address.local323, %address.local325, !dbg !15
  %address.local327 = zext nneg i32 %address.local326 to i64, !dbg !15
  %address.local328 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !15
  %780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !15
  %address.local329 = zext nneg i32 %address.thread321 to i64, !dbg !13
  %address.local330 = and i64 %address.local329, 60, !dbg !15
  %address.local331 = add nuw nsw i64 %address.local330, %address.local329, !dbg !15
  %address.local332 = shl nuw nsw i64 %address.local331, 3, !dbg !15
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local334 = and i64 %address.local329, 63, !dbg !13
  %address.local335 = lshr i64 %address.local334, 4, !dbg !15
  %address.local336 = mul nuw nsw i64 %address.local335, 384, !dbg !15
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) %address.local333, i64 %address.local336, !dbg !15
  %address.local338 = lshr i64 %address.local329, 2, !dbg !8
  %address.local339 = mul nsw i64 %address.local338, -32, !dbg !15
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) %address.local337, i64 %address.local339, !dbg !15
  %address.local341 = getelementptr i8, ptr addrspace(3) %address.local340, i64 256, !dbg !15
  %782 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !15
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) %address.local342, i64 %address.local336, !dbg !15
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) %address.local343, i64 %address.local339, !dbg !15
  %address.local345 = getelementptr i8, ptr addrspace(3) %address.local344, i64 512, !dbg !15
  %784 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !15
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) %address.local346, i64 %address.local336, !dbg !15
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) %address.local347, i64 %address.local339, !dbg !15
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local348, i64 768, !dbg !15
  %786 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %787 = bitcast <4 x i32> %786 to <8 x half>, !dbg !15
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !15
  %address.local351 = getelementptr i8, ptr addrspace(3) %address.local350, i64 4096, !dbg !15
  %788 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %789 = bitcast <4 x i32> %788 to <8 x half>, !dbg !15
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) %address.local352, i64 %address.local336, !dbg !15
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) %address.local353, i64 %address.local339, !dbg !15
  %address.local355 = getelementptr i8, ptr addrspace(3) %address.local354, i64 4352, !dbg !15
  %790 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %791 = bitcast <4 x i32> %790 to <8 x half>, !dbg !15
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local336, !dbg !15
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) %address.local357, i64 %address.local339, !dbg !15
  %address.local359 = getelementptr i8, ptr addrspace(3) %address.local358, i64 4608, !dbg !15
  %792 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %793 = bitcast <4 x i32> %792 to <8 x half>, !dbg !15
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %address.local360, i64 %address.local336, !dbg !15
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) %address.local361, i64 %address.local339, !dbg !15
  %address.local363 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4864, !dbg !15
  %794 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %795 = bitcast <4 x i32> %794 to <8 x half>, !dbg !15
  %796 = add nsw i64 %34, %8, !dbg !18
  %797 = shl i64 %796, 19, !dbg !18
  %798 = and i64 %797, 16252928, !dbg !18
  %799 = add i64 %798, %238, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %800 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %799, !dbg !18
  %801 = getelementptr i8, ptr addrspace(1) %800, i64 64, !dbg !18
  %address.local364 = shl nuw nsw i64 %address.local329, 3, !dbg !12
  %address.local365 = and i64 %address.local364, 1536, !dbg !12
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local365, !dbg !12
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !12
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !12
  %802 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !18
  %803 = zext i32 %802 to i64, !dbg !18
  %804 = inttoptr i64 %803 to ptr addrspace(3), !dbg !18
  %address.local369 = shl nuw nsw i64 %address.local329, 5, !dbg !18
  %address.local370 = and i64 %address.local369, 8064, !dbg !18
  %address.local371 = shl nuw nsw i64 %address.local329, 3, !dbg !18
  %address.local372 = and i64 %address.local371, 24, !dbg !18
  %address.local373 = or disjoint i64 %address.local370, %address.local372, !dbg !18
  %address.local374 = getelementptr [2 x i8], ptr addrspace(1) %801, i64 %address.local373, !dbg !18
  %805 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %806 = ptrtoint ptr addrspace(1) %address.local374 to i64, !dbg !18
  %807 = sub i64 %806, %805, !dbg !18
  %dma.buffer.offset894 = trunc i64 %807 to i32, !dbg !18
  %808 = trunc i64 %805 to i32, !dbg !18
  %809 = insertelement <4 x i32> poison, i32 %808, i64 0, !dbg !18
  %810 = lshr i64 %805, 32, !dbg !18
  %811 = trunc i64 %810 to i32, !dbg !18
  %812 = insertelement <4 x i32> %809, i32 %811, i64 1, !dbg !18
  %813 = insertelement <4 x i32> %812, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc895 = insertelement <4 x i32> %813, i32 131072, i64 3, !dbg !18
  %814 = ptrtoint ptr addrspace(3) %804 to i32, !dbg !18
  %dma.buffer.lds896 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %814), !dbg !18
  %815 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds896, i32 %dma.buffer.offset894, <4 x i32> %dma.buffer.rsrc895) #10, !dbg !18
  %address.local375 = or disjoint i64 %address.local329, 256, !dbg !12
  %address.local376 = shl nuw nsw i64 %address.local375, 3, !dbg !12
  %address.local377 = and i64 %address.local376, 3584, !dbg !12
  %address.local378 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local377, !dbg !12
  %address.local379 = ptrtoint ptr addrspace(3) %address.local378 to i64, !dbg !12
  %address.local380 = trunc i64 %address.local379 to i32, !dbg !12
  %816 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !18
  %817 = zext i32 %816 to i64, !dbg !18
  %818 = inttoptr i64 %817 to ptr addrspace(3), !dbg !18
  %address.local381 = shl nuw nsw i64 %address.local375, 5, !dbg !18
  %address.local382 = and i64 %address.local381, 16256, !dbg !18
  %address.local383 = shl nuw nsw i64 %address.local329, 3, !dbg !18
  %address.local384 = and i64 %address.local383, 24, !dbg !18
  %address.local385 = or disjoint i64 %address.local382, %address.local384, !dbg !18
  %address.local386 = getelementptr [2 x i8], ptr addrspace(1) %801, i64 %address.local385, !dbg !18
  %819 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %820 = ptrtoint ptr addrspace(1) %address.local386 to i64, !dbg !18
  %821 = sub i64 %820, %819, !dbg !18
  %dma.buffer.offset891 = trunc i64 %821 to i32, !dbg !18
  %822 = trunc i64 %819 to i32, !dbg !18
  %823 = insertelement <4 x i32> poison, i32 %822, i64 0, !dbg !18
  %824 = lshr i64 %819, 32, !dbg !18
  %825 = trunc i64 %824 to i32, !dbg !18
  %826 = insertelement <4 x i32> %823, i32 %825, i64 1, !dbg !18
  %827 = insertelement <4 x i32> %826, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc892 = insertelement <4 x i32> %827, i32 131072, i64 3, !dbg !18
  %828 = ptrtoint ptr addrspace(3) %818 to i32, !dbg !18
  %dma.buffer.lds893 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %828), !dbg !18
  %829 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds893, i32 %dma.buffer.offset891, <4 x i32> %dma.buffer.rsrc892) #10, !dbg !18
  %830 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %831 = extractvalue [2 x <8 x float>] %27, 0, !dbg !30
  %832 = shufflevector <8 x float> %831, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %830, <4 x float> %832) #9, !dbg !15
  %834 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %834, <4 x float> %833) #9, !dbg !15
  %836 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %836, <4 x float> %835) #9, !dbg !15
  %838 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %838, <4 x float> %837) #9, !dbg !15
  %840 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %840, <4 x float> %839) #9, !dbg !15
  %842 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %842, <4 x float> %841) #9, !dbg !15
  %844 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %844, <4 x float> %843) #9, !dbg !15
  %846 = shufflevector <8 x half> %795, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %846, <4 x float> %845) #9, !dbg !15
  %848 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %849 = shufflevector <8 x float> %831, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %848, <4 x float> %849) #9, !dbg !15
  %851 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %851, <4 x float> %850) #9, !dbg !15
  %853 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %853, <4 x float> %852) #9, !dbg !15
  %855 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %856 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %855, <4 x float> %854) #9, !dbg !15
  %857 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %858 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %857, <4 x float> %856) #9, !dbg !15
  %859 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %859, <4 x float> %858) #9, !dbg !15
  %861 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %862 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %861, <4 x float> %860) #9, !dbg !15
  %863 = shufflevector <8 x half> %795, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %864 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %863, <4 x float> %862) #9, !dbg !15
  %865 = extractvalue [2 x <8 x float>] %27, 1, !dbg !30
  %866 = shufflevector <8 x float> %865, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %830, <4 x float> %866) #9, !dbg !15
  %868 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %834, <4 x float> %867) #9, !dbg !15
  %869 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %836, <4 x float> %868) #9, !dbg !15
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %838, <4 x float> %869) #9, !dbg !15
  %871 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %840, <4 x float> %870) #9, !dbg !15
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %842, <4 x float> %871) #9, !dbg !15
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %844, <4 x float> %872) #9, !dbg !15
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %846, <4 x float> %873) #9, !dbg !15
  %875 = shufflevector <8 x float> %865, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %876 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %848, <4 x float> %875) #9, !dbg !15
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %851, <4 x float> %876) #9, !dbg !15
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %853, <4 x float> %877) #9, !dbg !15
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %855, <4 x float> %878) #9, !dbg !15
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %857, <4 x float> %879) #9, !dbg !15
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %859, <4 x float> %880) #9, !dbg !15
  %882 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %861, <4 x float> %881) #9, !dbg !15
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %863, <4 x float> %882) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !15
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !15
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !15
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !15
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !15
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !15
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !15
  %898 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %899 = bitcast <4 x i32> %898 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %900 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %799, !dbg !18
  %901 = getelementptr i8, ptr addrspace(1) %900, i64 128, !dbg !18
  %902 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !18
  %903 = zext i32 %902 to i64, !dbg !18
  %904 = inttoptr i64 %903 to ptr addrspace(3), !dbg !18
  %address.local387 = getelementptr [2 x i8], ptr addrspace(1) %901, i64 %address.local373, !dbg !18
  %905 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %906 = ptrtoint ptr addrspace(1) %address.local387 to i64, !dbg !18
  %907 = sub i64 %906, %905, !dbg !18
  %dma.buffer.offset888 = trunc i64 %907 to i32, !dbg !18
  %908 = trunc i64 %905 to i32, !dbg !18
  %909 = insertelement <4 x i32> poison, i32 %908, i64 0, !dbg !18
  %910 = lshr i64 %905, 32, !dbg !18
  %911 = trunc i64 %910 to i32, !dbg !18
  %912 = insertelement <4 x i32> %909, i32 %911, i64 1, !dbg !18
  %913 = insertelement <4 x i32> %912, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc889 = insertelement <4 x i32> %913, i32 131072, i64 3, !dbg !18
  %914 = ptrtoint ptr addrspace(3) %904 to i32, !dbg !18
  %dma.buffer.lds890 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %914), !dbg !18
  %915 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds890, i32 %dma.buffer.offset888, <4 x i32> %dma.buffer.rsrc889) #10, !dbg !18
  %916 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !18
  %917 = zext i32 %916 to i64, !dbg !18
  %918 = inttoptr i64 %917 to ptr addrspace(3), !dbg !18
  %address.local388 = getelementptr [2 x i8], ptr addrspace(1) %901, i64 %address.local385, !dbg !18
  %919 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %920 = ptrtoint ptr addrspace(1) %address.local388 to i64, !dbg !18
  %921 = sub i64 %920, %919, !dbg !18
  %dma.buffer.offset885 = trunc i64 %921 to i32, !dbg !18
  %922 = trunc i64 %919 to i32, !dbg !18
  %923 = insertelement <4 x i32> poison, i32 %922, i64 0, !dbg !18
  %924 = lshr i64 %919, 32, !dbg !18
  %925 = trunc i64 %924 to i32, !dbg !18
  %926 = insertelement <4 x i32> %923, i32 %925, i64 1, !dbg !18
  %927 = insertelement <4 x i32> %926, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc886 = insertelement <4 x i32> %927, i32 131072, i64 3, !dbg !18
  %928 = ptrtoint ptr addrspace(3) %918 to i32, !dbg !18
  %dma.buffer.lds887 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %928), !dbg !18
  %929 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds887, i32 %dma.buffer.offset885, <4 x i32> %dma.buffer.rsrc886) #10, !dbg !18
  %930 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %931 = extractvalue [2 x <8 x float>] %29, 0, !dbg !30
  %932 = shufflevector <8 x float> %931, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %930, <4 x float> %932) #9, !dbg !15
  %934 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %934, <4 x float> %933) #9, !dbg !15
  %936 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %936, <4 x float> %935) #9, !dbg !15
  %938 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %938, <4 x float> %937) #9, !dbg !15
  %940 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %940, <4 x float> %939) #9, !dbg !15
  %942 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %942, <4 x float> %941) #9, !dbg !15
  %944 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %944, <4 x float> %943) #9, !dbg !15
  %946 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %946, <4 x float> %945) #9, !dbg !15
  %948 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %949 = shufflevector <8 x float> %931, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %948, <4 x float> %949) #9, !dbg !15
  %951 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %951, <4 x float> %950) #9, !dbg !15
  %953 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %953, <4 x float> %952) #9, !dbg !15
  %955 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %955, <4 x float> %954) #9, !dbg !15
  %957 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %957, <4 x float> %956) #9, !dbg !15
  %959 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %959, <4 x float> %958) #9, !dbg !15
  %961 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %961, <4 x float> %960) #9, !dbg !15
  %963 = shufflevector <8 x half> %899, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %963, <4 x float> %962) #9, !dbg !15
  %965 = extractvalue [2 x <8 x float>] %29, 1, !dbg !30
  %966 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %930, <4 x float> %966) #9, !dbg !15
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %934, <4 x float> %967) #9, !dbg !15
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %936, <4 x float> %968) #9, !dbg !15
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %938, <4 x float> %969) #9, !dbg !15
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %940, <4 x float> %970) #9, !dbg !15
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %942, <4 x float> %971) #9, !dbg !15
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %944, <4 x float> %972) #9, !dbg !15
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %946, <4 x float> %973) #9, !dbg !15
  %975 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %948, <4 x float> %975) #9, !dbg !15
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %951, <4 x float> %976) #9, !dbg !15
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %953, <4 x float> %977) #9, !dbg !15
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %955, <4 x float> %978) #9, !dbg !15
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %957, <4 x float> %979) #9, !dbg !15
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %959, <4 x float> %980) #9, !dbg !15
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %961, <4 x float> %981) #9, !dbg !15
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %963, <4 x float> %982) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %984 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !15
  %986 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %987 = bitcast <4 x i32> %986 to <8 x half>, !dbg !15
  %988 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %989 = bitcast <4 x i32> %988 to <8 x half>, !dbg !15
  %990 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %991 = bitcast <4 x i32> %990 to <8 x half>, !dbg !15
  %992 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %993 = bitcast <4 x i32> %992 to <8 x half>, !dbg !15
  %994 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %995 = bitcast <4 x i32> %994 to <8 x half>, !dbg !15
  %996 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %997 = bitcast <4 x i32> %996 to <8 x half>, !dbg !15
  %998 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %999 = bitcast <4 x i32> %998 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1000 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %799, !dbg !18
  %1001 = getelementptr i8, ptr addrspace(1) %1000, i64 192, !dbg !18
  %1002 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !18
  %1003 = zext i32 %1002 to i64, !dbg !18
  %1004 = inttoptr i64 %1003 to ptr addrspace(3), !dbg !18
  %address.local389 = getelementptr [2 x i8], ptr addrspace(1) %1001, i64 %address.local373, !dbg !18
  %1005 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1006 = ptrtoint ptr addrspace(1) %address.local389 to i64, !dbg !18
  %1007 = sub i64 %1006, %1005, !dbg !18
  %dma.buffer.offset882 = trunc i64 %1007 to i32, !dbg !18
  %1008 = trunc i64 %1005 to i32, !dbg !18
  %1009 = insertelement <4 x i32> poison, i32 %1008, i64 0, !dbg !18
  %1010 = lshr i64 %1005, 32, !dbg !18
  %1011 = trunc i64 %1010 to i32, !dbg !18
  %1012 = insertelement <4 x i32> %1009, i32 %1011, i64 1, !dbg !18
  %1013 = insertelement <4 x i32> %1012, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc883 = insertelement <4 x i32> %1013, i32 131072, i64 3, !dbg !18
  %1014 = ptrtoint ptr addrspace(3) %1004 to i32, !dbg !18
  %dma.buffer.lds884 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1014), !dbg !18
  %1015 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds884, i32 %dma.buffer.offset882, <4 x i32> %dma.buffer.rsrc883) #10, !dbg !18
  %1016 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !18
  %1017 = zext i32 %1016 to i64, !dbg !18
  %1018 = inttoptr i64 %1017 to ptr addrspace(3), !dbg !18
  %address.local390 = getelementptr [2 x i8], ptr addrspace(1) %1001, i64 %address.local385, !dbg !18
  %1019 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1020 = ptrtoint ptr addrspace(1) %address.local390 to i64, !dbg !18
  %1021 = sub i64 %1020, %1019, !dbg !18
  %dma.buffer.offset879 = trunc i64 %1021 to i32, !dbg !18
  %1022 = trunc i64 %1019 to i32, !dbg !18
  %1023 = insertelement <4 x i32> poison, i32 %1022, i64 0, !dbg !18
  %1024 = lshr i64 %1019, 32, !dbg !18
  %1025 = trunc i64 %1024 to i32, !dbg !18
  %1026 = insertelement <4 x i32> %1023, i32 %1025, i64 1, !dbg !18
  %1027 = insertelement <4 x i32> %1026, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc880 = insertelement <4 x i32> %1027, i32 131072, i64 3, !dbg !18
  %1028 = ptrtoint ptr addrspace(3) %1018 to i32, !dbg !18
  %dma.buffer.lds881 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1028), !dbg !18
  %1029 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds881, i32 %dma.buffer.offset879, <4 x i32> %dma.buffer.rsrc880) #10, !dbg !18
  %1030 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1031 = extractvalue [2 x <8 x float>] %30, 0, !dbg !30
  %1032 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1032) #9, !dbg !15
  %1034 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1034, <4 x float> %1033) #9, !dbg !15
  %1036 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1036, <4 x float> %1035) #9, !dbg !15
  %1038 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1038, <4 x float> %1037) #9, !dbg !15
  %1040 = shufflevector <8 x half> %993, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1040, <4 x float> %1039) #9, !dbg !15
  %1042 = shufflevector <8 x half> %995, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1042, <4 x float> %1041) #9, !dbg !15
  %1044 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1044, <4 x float> %1043) #9, !dbg !15
  %1046 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1046, <4 x float> %1045) #9, !dbg !15
  %1048 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1049 = shufflevector <8 x float> %1031, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1048, <4 x float> %1049) #9, !dbg !15
  %1051 = shufflevector <8 x half> %987, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1051, <4 x float> %1050) #9, !dbg !15
  %1053 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1053, <4 x float> %1052) #9, !dbg !15
  %1055 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1055, <4 x float> %1054) #9, !dbg !15
  %1057 = shufflevector <8 x half> %993, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1057, <4 x float> %1056) #9, !dbg !15
  %1059 = shufflevector <8 x half> %995, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1059, <4 x float> %1058) #9, !dbg !15
  %1061 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1061, <4 x float> %1060) #9, !dbg !15
  %1063 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1063, <4 x float> %1062) #9, !dbg !15
  %1065 = extractvalue [2 x <8 x float>] %30, 1, !dbg !30
  %1066 = shufflevector <8 x float> %1065, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1030, <4 x float> %1066) #9, !dbg !15
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1034, <4 x float> %1067) #9, !dbg !15
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1036, <4 x float> %1068) #9, !dbg !15
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1038, <4 x float> %1069) #9, !dbg !15
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1040, <4 x float> %1070) #9, !dbg !15
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1042, <4 x float> %1071) #9, !dbg !15
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1044, <4 x float> %1072) #9, !dbg !15
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1046, <4 x float> %1073) #9, !dbg !15
  %1075 = shufflevector <8 x float> %1065, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1076 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1048, <4 x float> %1075) #9, !dbg !15
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1051, <4 x float> %1076) #9, !dbg !15
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1053, <4 x float> %1077) #9, !dbg !15
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1055, <4 x float> %1078) #9, !dbg !15
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1057, <4 x float> %1079) #9, !dbg !15
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1059, <4 x float> %1080) #9, !dbg !15
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1061, <4 x float> %1081) #9, !dbg !15
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1063, <4 x float> %1082) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1084 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %1085 = bitcast <4 x i32> %1084 to <8 x half>, !dbg !15
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !15
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !15
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !15
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !15
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !15
  %1096 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %1097 = bitcast <4 x i32> %1096 to <8 x half>, !dbg !15
  %1098 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %1099 = bitcast <4 x i32> %1098 to <8 x half>, !dbg !15
  %1100 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1101 = extractvalue [2 x <8 x float>] %31, 0, !dbg !30
  %1102 = shufflevector <8 x float> %1101, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1100, <4 x float> %1102) #9, !dbg !15
  %1104 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1105 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1104, <4 x float> %1103) #9, !dbg !15
  %1106 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1107 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1106, <4 x float> %1105) #9, !dbg !15
  %1108 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1109 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1108, <4 x float> %1107) #9, !dbg !15
  %1110 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1110, <4 x float> %1109) #9, !dbg !15
  %1112 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1112, <4 x float> %1111) #9, !dbg !15
  %1114 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1114, <4 x float> %1113) #9, !dbg !15
  %1116 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1116, <4 x float> %1115) #9, !dbg !15
  %1118 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1119 = shufflevector <8 x float> %1101, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1118, <4 x float> %1119) #9, !dbg !15
  %1121 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1121, <4 x float> %1120) #9, !dbg !15
  %1123 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1123, <4 x float> %1122) #9, !dbg !15
  %1125 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1125, <4 x float> %1124) #9, !dbg !15
  %1127 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1127, <4 x float> %1126) #9, !dbg !15
  %1129 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1129, <4 x float> %1128) #9, !dbg !15
  %1131 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1131, <4 x float> %1130) #9, !dbg !15
  %1133 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1133, <4 x float> %1132) #9, !dbg !15
  %1135 = extractvalue [2 x <8 x float>] %31, 1, !dbg !30
  %1136 = shufflevector <8 x float> %1135, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1100, <4 x float> %1136) #9, !dbg !15
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1104, <4 x float> %1137) #9, !dbg !15
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1106, <4 x float> %1138) #9, !dbg !15
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1108, <4 x float> %1139) #9, !dbg !15
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1110, <4 x float> %1140) #9, !dbg !15
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1112, <4 x float> %1141) #9, !dbg !15
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1114, <4 x float> %1142) #9, !dbg !15
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1116, <4 x float> %1143) #9, !dbg !15
  %1145 = shufflevector <8 x float> %1135, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1118, <4 x float> %1145) #9, !dbg !15
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1121, <4 x float> %1146) #9, !dbg !15
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1123, <4 x float> %1147) #9, !dbg !15
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1125, <4 x float> %1148) #9, !dbg !15
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1127, <4 x float> %1149) #9, !dbg !15
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1129, <4 x float> %1150) #9, !dbg !15
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1131, <4 x float> %1151) #9, !dbg !15
  %1153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1133, <4 x float> %1152) #9, !dbg !15
  %1154 = shufflevector <4 x float> %847, <4 x float> %864, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1155 = insertvalue [2 x <8 x float>] poison, <8 x float> %1154, 0, !dbg !16
  %1156 = shufflevector <4 x float> %874, <4 x float> %883, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1157 = insertvalue [2 x <8 x float>] %1155, <8 x float> %1156, 1, !dbg !16
  %1158 = shufflevector <4 x float> %947, <4 x float> %964, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1159 = insertvalue [2 x <8 x float>] poison, <8 x float> %1158, 0, !dbg !16
  %1160 = shufflevector <4 x float> %974, <4 x float> %983, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1161 = insertvalue [2 x <8 x float>] %1159, <8 x float> %1160, 1, !dbg !16
  %1162 = shufflevector <4 x float> %1047, <4 x float> %1064, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1163 = insertvalue [2 x <8 x float>] poison, <8 x float> %1162, 0, !dbg !16
  %1164 = shufflevector <4 x float> %1074, <4 x float> %1083, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1165 = insertvalue [2 x <8 x float>] %1163, <8 x float> %1164, 1, !dbg !16
  %1166 = shufflevector <4 x float> %1117, <4 x float> %1134, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1167 = insertvalue [2 x <8 x float>] poison, <8 x float> %1166, 0, !dbg !16
  %1168 = shufflevector <4 x float> %1144, <4 x float> %1153, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1169 = insertvalue [2 x <8 x float>] %1167, <8 x float> %1168, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1170 = add nuw nsw i64 %26, 1, !dbg !16
  br label %25, !dbg !16

1171:                                             ; preds = %25
  %1172 = and i32 %19, 16, !dbg !16
  %.not2 = icmp eq i32 %1172, 0, !dbg !16
  %1173 = and i32 %19, 32, !dbg !16
  %.not3 = icmp eq i32 %1173, 0, !dbg !16
  %1174 = extractvalue [2 x <8 x float>] %31, 0, !dbg !16
  %1175 = extractelement <8 x float> %1174, i64 0, !dbg !16
  %1176 = extractelement <8 x float> %1174, i64 1, !dbg !16
  %1177 = extractelement <8 x float> %1174, i64 2, !dbg !16
  %1178 = extractelement <8 x float> %1174, i64 3, !dbg !16
  %1179 = select i1 %.not2, float %1176, float %1175, !dbg !16
  %1180 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1181 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1180), !dbg !16
  %1182 = add nuw nsw i32 %1181, 64, !dbg !16
  %1183 = xor i32 %1181, 16, !dbg !16
  %1184 = and i32 %1182, 192, !dbg !16
  %1185 = icmp samesign ult i32 %1183, %1184, !dbg !16
  %1186 = select i1 %1185, i32 %1183, i32 %1181, !dbg !16
  %1187 = shl nuw nsw i32 %1186, 2, !dbg !16
  %1188 = bitcast float %1179 to i32, !dbg !16
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1188), !dbg !16
  %1190 = bitcast i32 %1189 to float, !dbg !16
  %1191 = select i1 %.not2, float %1175, float %1190, !dbg !16
  %1192 = select i1 %.not2, float %1190, float %1176, !dbg !16
  %1193 = select i1 %.not2, float %1178, float %1177, !dbg !16
  %1194 = bitcast float %1193 to i32, !dbg !16
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1194), !dbg !16
  %1196 = bitcast i32 %1195 to float, !dbg !16
  %1197 = select i1 %.not2, float %1177, float %1196, !dbg !16
  %1198 = select i1 %.not2, float %1196, float %1178, !dbg !16
  %1199 = select i1 %.not3, float %1197, float %1191, !dbg !16
  %1200 = xor i32 %1181, 32, !dbg !16
  %1201 = icmp samesign ult i32 %1200, %1184, !dbg !16
  %1202 = select i1 %1201, i32 %1200, i32 %1181, !dbg !16
  %1203 = shl nuw nsw i32 %1202, 2, !dbg !16
  %1204 = bitcast float %1199 to i32, !dbg !16
  %1205 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1204), !dbg !16
  %1206 = bitcast i32 %1205 to float, !dbg !16
  %1207 = select i1 %.not3, float %1191, float %1206, !dbg !16
  %1208 = select i1 %.not3, float %1206, float %1197, !dbg !16
  %1209 = select i1 %.not3, float %1198, float %1192, !dbg !16
  %1210 = bitcast float %1209 to i32, !dbg !16
  %1211 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1210), !dbg !16
  %1212 = bitcast i32 %1211 to float, !dbg !16
  %1213 = select i1 %.not3, float %1192, float %1212, !dbg !16
  %1214 = select i1 %.not3, float %1212, float %1198, !dbg !16
  %1215 = extractvalue [2 x <8 x float>] %30, 0, !dbg !16
  %1216 = extractelement <8 x float> %1215, i64 0, !dbg !16
  %1217 = extractelement <8 x float> %1215, i64 1, !dbg !16
  %1218 = extractelement <8 x float> %1215, i64 2, !dbg !16
  %1219 = extractelement <8 x float> %1215, i64 3, !dbg !16
  %1220 = select i1 %.not2, float %1217, float %1216, !dbg !16
  %1221 = bitcast float %1220 to i32, !dbg !16
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1221), !dbg !16
  %1223 = bitcast i32 %1222 to float, !dbg !16
  %1224 = select i1 %.not2, float %1216, float %1223, !dbg !16
  %1225 = select i1 %.not2, float %1223, float %1217, !dbg !16
  %1226 = select i1 %.not2, float %1219, float %1218, !dbg !16
  %1227 = bitcast float %1226 to i32, !dbg !16
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1227), !dbg !16
  %1229 = bitcast i32 %1228 to float, !dbg !16
  %1230 = select i1 %.not2, float %1218, float %1229, !dbg !16
  %1231 = select i1 %.not2, float %1229, float %1219, !dbg !16
  %1232 = select i1 %.not3, float %1230, float %1224, !dbg !16
  %1233 = bitcast float %1232 to i32, !dbg !16
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1233), !dbg !16
  %1235 = bitcast i32 %1234 to float, !dbg !16
  %1236 = select i1 %.not3, float %1224, float %1235, !dbg !16
  %1237 = select i1 %.not3, float %1235, float %1230, !dbg !16
  %1238 = select i1 %.not3, float %1231, float %1225, !dbg !16
  %1239 = bitcast float %1238 to i32, !dbg !16
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1239), !dbg !16
  %1241 = bitcast i32 %1240 to float, !dbg !16
  %1242 = select i1 %.not3, float %1225, float %1241, !dbg !16
  %1243 = select i1 %.not3, float %1241, float %1231, !dbg !16
  %1244 = extractvalue [2 x <8 x float>] %29, 0, !dbg !16
  %1245 = extractelement <8 x float> %1244, i64 0, !dbg !16
  %1246 = extractelement <8 x float> %1244, i64 1, !dbg !16
  %1247 = extractelement <8 x float> %1244, i64 2, !dbg !16
  %1248 = extractelement <8 x float> %1244, i64 3, !dbg !16
  %1249 = select i1 %.not2, float %1246, float %1245, !dbg !16
  %1250 = bitcast float %1249 to i32, !dbg !16
  %1251 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1250), !dbg !16
  %1252 = bitcast i32 %1251 to float, !dbg !16
  %1253 = select i1 %.not2, float %1245, float %1252, !dbg !16
  %1254 = select i1 %.not2, float %1252, float %1246, !dbg !16
  %1255 = select i1 %.not2, float %1248, float %1247, !dbg !16
  %1256 = bitcast float %1255 to i32, !dbg !16
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1256), !dbg !16
  %1258 = bitcast i32 %1257 to float, !dbg !16
  %1259 = select i1 %.not2, float %1247, float %1258, !dbg !16
  %1260 = select i1 %.not2, float %1258, float %1248, !dbg !16
  %1261 = select i1 %.not3, float %1259, float %1253, !dbg !16
  %1262 = bitcast float %1261 to i32, !dbg !16
  %1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1262), !dbg !16
  %1264 = bitcast i32 %1263 to float, !dbg !16
  %1265 = select i1 %.not3, float %1253, float %1264, !dbg !16
  %1266 = select i1 %.not3, float %1264, float %1259, !dbg !16
  %1267 = select i1 %.not3, float %1260, float %1254, !dbg !16
  %1268 = bitcast float %1267 to i32, !dbg !16
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1268), !dbg !16
  %1270 = bitcast i32 %1269 to float, !dbg !16
  %1271 = select i1 %.not3, float %1254, float %1270, !dbg !16
  %1272 = select i1 %.not3, float %1270, float %1260, !dbg !16
  %1273 = extractvalue [2 x <8 x float>] %27, 0, !dbg !16
  %1274 = extractelement <8 x float> %1273, i64 0, !dbg !16
  %1275 = extractelement <8 x float> %1273, i64 1, !dbg !16
  %1276 = extractelement <8 x float> %1273, i64 2, !dbg !16
  %1277 = extractelement <8 x float> %1273, i64 3, !dbg !16
  %1278 = select i1 %.not2, float %1275, float %1274, !dbg !16
  %1279 = bitcast float %1278 to i32, !dbg !16
  %1280 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1279), !dbg !16
  %1281 = bitcast i32 %1280 to float, !dbg !16
  %1282 = select i1 %.not2, float %1274, float %1281, !dbg !16
  %1283 = select i1 %.not2, float %1281, float %1275, !dbg !16
  %1284 = select i1 %.not2, float %1277, float %1276, !dbg !16
  %1285 = bitcast float %1284 to i32, !dbg !16
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1285), !dbg !16
  %1287 = bitcast i32 %1286 to float, !dbg !16
  %1288 = select i1 %.not2, float %1276, float %1287, !dbg !16
  %1289 = select i1 %.not2, float %1287, float %1277, !dbg !16
  %1290 = select i1 %.not3, float %1288, float %1282, !dbg !16
  %1291 = bitcast float %1290 to i32, !dbg !16
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1291), !dbg !16
  %1293 = select i1 %.not3, float %1289, float %1283, !dbg !16
  %1294 = bitcast float %1293 to i32, !dbg !16
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1294), !dbg !16
  %1296 = extractvalue [2 x <1 x float>] %28, 0, !dbg !19
  %1297 = shufflevector <1 x float> %1296, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1298 = extractelement <8 x float> %1174, i64 4, !dbg !16
  %1299 = extractelement <8 x float> %1174, i64 5, !dbg !16
  %1300 = extractelement <8 x float> %1174, i64 6, !dbg !16
  %1301 = extractelement <8 x float> %1174, i64 7, !dbg !16
  %1302 = select i1 %.not2, float %1299, float %1298, !dbg !16
  %1303 = bitcast float %1302 to i32, !dbg !16
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1303), !dbg !16
  %1305 = bitcast i32 %1304 to float, !dbg !16
  %1306 = select i1 %.not2, float %1298, float %1305, !dbg !16
  %1307 = select i1 %.not2, float %1305, float %1299, !dbg !16
  %1308 = select i1 %.not2, float %1301, float %1300, !dbg !16
  %1309 = bitcast float %1308 to i32, !dbg !16
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1309), !dbg !16
  %1311 = bitcast i32 %1310 to float, !dbg !16
  %1312 = select i1 %.not2, float %1300, float %1311, !dbg !16
  %1313 = select i1 %.not2, float %1311, float %1301, !dbg !16
  %1314 = select i1 %.not3, float %1312, float %1306, !dbg !16
  %1315 = bitcast float %1314 to i32, !dbg !16
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1315), !dbg !16
  %1317 = bitcast i32 %1316 to float, !dbg !16
  %1318 = select i1 %.not3, float %1306, float %1317, !dbg !16
  %1319 = select i1 %.not3, float %1317, float %1312, !dbg !16
  %1320 = select i1 %.not3, float %1313, float %1307, !dbg !16
  %1321 = bitcast float %1320 to i32, !dbg !16
  %1322 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1321), !dbg !16
  %1323 = bitcast i32 %1322 to float, !dbg !16
  %1324 = select i1 %.not3, float %1307, float %1323, !dbg !16
  %1325 = select i1 %.not3, float %1323, float %1313, !dbg !16
  %1326 = extractelement <8 x float> %1215, i64 4, !dbg !16
  %1327 = extractelement <8 x float> %1215, i64 5, !dbg !16
  %1328 = extractelement <8 x float> %1215, i64 6, !dbg !16
  %1329 = extractelement <8 x float> %1215, i64 7, !dbg !16
  %1330 = select i1 %.not2, float %1327, float %1326, !dbg !16
  %1331 = bitcast float %1330 to i32, !dbg !16
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1331), !dbg !16
  %1333 = bitcast i32 %1332 to float, !dbg !16
  %1334 = select i1 %.not2, float %1326, float %1333, !dbg !16
  %1335 = select i1 %.not2, float %1333, float %1327, !dbg !16
  %1336 = select i1 %.not2, float %1329, float %1328, !dbg !16
  %1337 = bitcast float %1336 to i32, !dbg !16
  %1338 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1337), !dbg !16
  %1339 = bitcast i32 %1338 to float, !dbg !16
  %1340 = select i1 %.not2, float %1328, float %1339, !dbg !16
  %1341 = select i1 %.not2, float %1339, float %1329, !dbg !16
  %1342 = select i1 %.not3, float %1340, float %1334, !dbg !16
  %1343 = bitcast float %1342 to i32, !dbg !16
  %1344 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1343), !dbg !16
  %1345 = bitcast i32 %1344 to float, !dbg !16
  %1346 = select i1 %.not3, float %1334, float %1345, !dbg !16
  %1347 = select i1 %.not3, float %1345, float %1340, !dbg !16
  %1348 = select i1 %.not3, float %1341, float %1335, !dbg !16
  %1349 = bitcast float %1348 to i32, !dbg !16
  %1350 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1349), !dbg !16
  %1351 = bitcast i32 %1350 to float, !dbg !16
  %1352 = select i1 %.not3, float %1335, float %1351, !dbg !16
  %1353 = select i1 %.not3, float %1351, float %1341, !dbg !16
  %1354 = extractelement <8 x float> %1244, i64 4, !dbg !16
  %1355 = extractelement <8 x float> %1244, i64 5, !dbg !16
  %1356 = extractelement <8 x float> %1244, i64 6, !dbg !16
  %1357 = extractelement <8 x float> %1244, i64 7, !dbg !16
  %1358 = select i1 %.not2, float %1355, float %1354, !dbg !16
  %1359 = bitcast float %1358 to i32, !dbg !16
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1359), !dbg !16
  %1361 = bitcast i32 %1360 to float, !dbg !16
  %1362 = select i1 %.not2, float %1354, float %1361, !dbg !16
  %1363 = select i1 %.not2, float %1361, float %1355, !dbg !16
  %1364 = select i1 %.not2, float %1357, float %1356, !dbg !16
  %1365 = bitcast float %1364 to i32, !dbg !16
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1365), !dbg !16
  %1367 = bitcast i32 %1366 to float, !dbg !16
  %1368 = select i1 %.not2, float %1356, float %1367, !dbg !16
  %1369 = select i1 %.not2, float %1367, float %1357, !dbg !16
  %1370 = select i1 %.not3, float %1368, float %1362, !dbg !16
  %1371 = bitcast float %1370 to i32, !dbg !16
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1371), !dbg !16
  %1373 = bitcast i32 %1372 to float, !dbg !16
  %1374 = select i1 %.not3, float %1362, float %1373, !dbg !16
  %1375 = select i1 %.not3, float %1373, float %1368, !dbg !16
  %1376 = select i1 %.not3, float %1369, float %1363, !dbg !16
  %1377 = bitcast float %1376 to i32, !dbg !16
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1377), !dbg !16
  %1379 = bitcast i32 %1378 to float, !dbg !16
  %1380 = select i1 %.not3, float %1363, float %1379, !dbg !16
  %1381 = select i1 %.not3, float %1379, float %1369, !dbg !16
  %1382 = extractelement <8 x float> %1273, i64 4, !dbg !16
  %1383 = extractelement <8 x float> %1273, i64 5, !dbg !16
  %1384 = extractelement <8 x float> %1273, i64 6, !dbg !16
  %1385 = extractelement <8 x float> %1273, i64 7, !dbg !16
  %1386 = select i1 %.not2, float %1383, float %1382, !dbg !16
  %1387 = bitcast float %1386 to i32, !dbg !16
  %1388 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1387), !dbg !16
  %1389 = bitcast i32 %1388 to float, !dbg !16
  %1390 = select i1 %.not2, float %1382, float %1389, !dbg !16
  %1391 = select i1 %.not2, float %1389, float %1383, !dbg !16
  %1392 = select i1 %.not2, float %1385, float %1384, !dbg !16
  %1393 = bitcast float %1392 to i32, !dbg !16
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1393), !dbg !16
  %1395 = bitcast i32 %1394 to float, !dbg !16
  %1396 = select i1 %.not2, float %1384, float %1395, !dbg !16
  %1397 = select i1 %.not2, float %1395, float %1385, !dbg !16
  %1398 = select i1 %.not3, float %1396, float %1390, !dbg !16
  %1399 = bitcast float %1398 to i32, !dbg !16
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1399), !dbg !16
  %1401 = select i1 %.not3, float %1397, float %1391, !dbg !16
  %1402 = bitcast float %1401 to i32, !dbg !16
  %1403 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1402), !dbg !16
  %1404 = extractvalue [2 x <8 x float>] %31, 1, !dbg !16
  %1405 = extractelement <8 x float> %1404, i64 0, !dbg !16
  %1406 = extractelement <8 x float> %1404, i64 1, !dbg !16
  %1407 = extractelement <8 x float> %1404, i64 2, !dbg !16
  %1408 = extractelement <8 x float> %1404, i64 3, !dbg !16
  %1409 = select i1 %.not2, float %1406, float %1405, !dbg !16
  %1410 = bitcast float %1409 to i32, !dbg !16
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1410), !dbg !16
  %1412 = bitcast i32 %1411 to float, !dbg !16
  %1413 = select i1 %.not2, float %1405, float %1412, !dbg !16
  %1414 = select i1 %.not2, float %1412, float %1406, !dbg !16
  %1415 = select i1 %.not2, float %1408, float %1407, !dbg !16
  %1416 = bitcast float %1415 to i32, !dbg !16
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1416), !dbg !16
  %1418 = bitcast i32 %1417 to float, !dbg !16
  %1419 = select i1 %.not2, float %1407, float %1418, !dbg !16
  %1420 = select i1 %.not2, float %1418, float %1408, !dbg !16
  %1421 = select i1 %.not3, float %1419, float %1413, !dbg !16
  %1422 = bitcast float %1421 to i32, !dbg !16
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1422), !dbg !16
  %1424 = bitcast i32 %1423 to float, !dbg !16
  %1425 = select i1 %.not3, float %1413, float %1424, !dbg !16
  %1426 = select i1 %.not3, float %1424, float %1419, !dbg !16
  %1427 = select i1 %.not3, float %1420, float %1414, !dbg !16
  %1428 = bitcast float %1427 to i32, !dbg !16
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1428), !dbg !16
  %1430 = bitcast i32 %1429 to float, !dbg !16
  %1431 = select i1 %.not3, float %1414, float %1430, !dbg !16
  %1432 = select i1 %.not3, float %1430, float %1420, !dbg !16
  %1433 = extractvalue [2 x <8 x float>] %30, 1, !dbg !16
  %1434 = extractelement <8 x float> %1433, i64 0, !dbg !16
  %1435 = extractelement <8 x float> %1433, i64 1, !dbg !16
  %1436 = extractelement <8 x float> %1433, i64 2, !dbg !16
  %1437 = extractelement <8 x float> %1433, i64 3, !dbg !16
  %1438 = select i1 %.not2, float %1435, float %1434, !dbg !16
  %1439 = bitcast float %1438 to i32, !dbg !16
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1439), !dbg !16
  %1441 = bitcast i32 %1440 to float, !dbg !16
  %1442 = select i1 %.not2, float %1434, float %1441, !dbg !16
  %1443 = select i1 %.not2, float %1441, float %1435, !dbg !16
  %1444 = select i1 %.not2, float %1437, float %1436, !dbg !16
  %1445 = bitcast float %1444 to i32, !dbg !16
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1445), !dbg !16
  %1447 = bitcast i32 %1446 to float, !dbg !16
  %1448 = select i1 %.not2, float %1436, float %1447, !dbg !16
  %1449 = select i1 %.not2, float %1447, float %1437, !dbg !16
  %1450 = select i1 %.not3, float %1448, float %1442, !dbg !16
  %1451 = bitcast float %1450 to i32, !dbg !16
  %1452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1451), !dbg !16
  %1453 = bitcast i32 %1452 to float, !dbg !16
  %1454 = select i1 %.not3, float %1442, float %1453, !dbg !16
  %1455 = select i1 %.not3, float %1453, float %1448, !dbg !16
  %1456 = select i1 %.not3, float %1449, float %1443, !dbg !16
  %1457 = bitcast float %1456 to i32, !dbg !16
  %1458 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1457), !dbg !16
  %1459 = bitcast i32 %1458 to float, !dbg !16
  %1460 = select i1 %.not3, float %1443, float %1459, !dbg !16
  %1461 = select i1 %.not3, float %1459, float %1449, !dbg !16
  %1462 = extractvalue [2 x <8 x float>] %29, 1, !dbg !16
  %1463 = extractelement <8 x float> %1462, i64 0, !dbg !16
  %1464 = extractelement <8 x float> %1462, i64 1, !dbg !16
  %1465 = extractelement <8 x float> %1462, i64 2, !dbg !16
  %1466 = extractelement <8 x float> %1462, i64 3, !dbg !16
  %1467 = select i1 %.not2, float %1464, float %1463, !dbg !16
  %1468 = bitcast float %1467 to i32, !dbg !16
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1468), !dbg !16
  %1470 = bitcast i32 %1469 to float, !dbg !16
  %1471 = select i1 %.not2, float %1463, float %1470, !dbg !16
  %1472 = select i1 %.not2, float %1470, float %1464, !dbg !16
  %1473 = select i1 %.not2, float %1466, float %1465, !dbg !16
  %1474 = bitcast float %1473 to i32, !dbg !16
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1474), !dbg !16
  %1476 = bitcast i32 %1475 to float, !dbg !16
  %1477 = select i1 %.not2, float %1465, float %1476, !dbg !16
  %1478 = select i1 %.not2, float %1476, float %1466, !dbg !16
  %1479 = select i1 %.not3, float %1477, float %1471, !dbg !16
  %1480 = bitcast float %1479 to i32, !dbg !16
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1480), !dbg !16
  %1482 = bitcast i32 %1481 to float, !dbg !16
  %1483 = select i1 %.not3, float %1471, float %1482, !dbg !16
  %1484 = select i1 %.not3, float %1482, float %1477, !dbg !16
  %1485 = select i1 %.not3, float %1478, float %1472, !dbg !16
  %1486 = bitcast float %1485 to i32, !dbg !16
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1486), !dbg !16
  %1488 = bitcast i32 %1487 to float, !dbg !16
  %1489 = select i1 %.not3, float %1472, float %1488, !dbg !16
  %1490 = select i1 %.not3, float %1488, float %1478, !dbg !16
  %1491 = extractvalue [2 x <8 x float>] %27, 1, !dbg !16
  %1492 = extractelement <8 x float> %1491, i64 0, !dbg !16
  %1493 = extractelement <8 x float> %1491, i64 1, !dbg !16
  %1494 = extractelement <8 x float> %1491, i64 2, !dbg !16
  %1495 = extractelement <8 x float> %1491, i64 3, !dbg !16
  %1496 = select i1 %.not2, float %1493, float %1492, !dbg !16
  %1497 = bitcast float %1496 to i32, !dbg !16
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1497), !dbg !16
  %1499 = bitcast i32 %1498 to float, !dbg !16
  %1500 = select i1 %.not2, float %1492, float %1499, !dbg !16
  %1501 = select i1 %.not2, float %1499, float %1493, !dbg !16
  %1502 = select i1 %.not2, float %1495, float %1494, !dbg !16
  %1503 = bitcast float %1502 to i32, !dbg !16
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1503), !dbg !16
  %1505 = bitcast i32 %1504 to float, !dbg !16
  %1506 = select i1 %.not2, float %1494, float %1505, !dbg !16
  %1507 = select i1 %.not2, float %1505, float %1495, !dbg !16
  %1508 = select i1 %.not3, float %1506, float %1500, !dbg !16
  %1509 = bitcast float %1508 to i32, !dbg !16
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1509), !dbg !16
  %1511 = select i1 %.not3, float %1507, float %1501, !dbg !16
  %1512 = bitcast float %1511 to i32, !dbg !16
  %1513 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1512), !dbg !16
  %1514 = extractvalue [2 x <1 x float>] %28, 1, !dbg !19
  %1515 = shufflevector <1 x float> %1514, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1516 = extractelement <8 x float> %1404, i64 4, !dbg !16
  %1517 = extractelement <8 x float> %1404, i64 5, !dbg !16
  %1518 = extractelement <8 x float> %1404, i64 6, !dbg !16
  %1519 = extractelement <8 x float> %1404, i64 7, !dbg !16
  %1520 = select i1 %.not2, float %1517, float %1516, !dbg !16
  %1521 = bitcast float %1520 to i32, !dbg !16
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1521), !dbg !16
  %1523 = bitcast i32 %1522 to float, !dbg !16
  %1524 = select i1 %.not2, float %1516, float %1523, !dbg !16
  %1525 = select i1 %.not2, float %1523, float %1517, !dbg !16
  %1526 = select i1 %.not2, float %1519, float %1518, !dbg !16
  %1527 = bitcast float %1526 to i32, !dbg !16
  %1528 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1527), !dbg !16
  %1529 = bitcast i32 %1528 to float, !dbg !16
  %1530 = select i1 %.not2, float %1518, float %1529, !dbg !16
  %1531 = select i1 %.not2, float %1529, float %1519, !dbg !16
  %1532 = select i1 %.not3, float %1530, float %1524, !dbg !16
  %1533 = bitcast float %1532 to i32, !dbg !16
  %1534 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1533), !dbg !16
  %1535 = bitcast i32 %1534 to float, !dbg !16
  %1536 = select i1 %.not3, float %1524, float %1535, !dbg !16
  %1537 = select i1 %.not3, float %1535, float %1530, !dbg !16
  %1538 = select i1 %.not3, float %1531, float %1525, !dbg !16
  %1539 = bitcast float %1538 to i32, !dbg !16
  %1540 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1539), !dbg !16
  %1541 = bitcast i32 %1540 to float, !dbg !16
  %1542 = select i1 %.not3, float %1525, float %1541, !dbg !16
  %1543 = select i1 %.not3, float %1541, float %1531, !dbg !16
  %1544 = extractelement <8 x float> %1433, i64 4, !dbg !16
  %1545 = extractelement <8 x float> %1433, i64 5, !dbg !16
  %1546 = extractelement <8 x float> %1433, i64 6, !dbg !16
  %1547 = extractelement <8 x float> %1433, i64 7, !dbg !16
  %1548 = select i1 %.not2, float %1545, float %1544, !dbg !16
  %1549 = bitcast float %1548 to i32, !dbg !16
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1549), !dbg !16
  %1551 = bitcast i32 %1550 to float, !dbg !16
  %1552 = select i1 %.not2, float %1544, float %1551, !dbg !16
  %1553 = select i1 %.not2, float %1551, float %1545, !dbg !16
  %1554 = select i1 %.not2, float %1547, float %1546, !dbg !16
  %1555 = bitcast float %1554 to i32, !dbg !16
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1555), !dbg !16
  %1557 = bitcast i32 %1556 to float, !dbg !16
  %1558 = select i1 %.not2, float %1546, float %1557, !dbg !16
  %1559 = select i1 %.not2, float %1557, float %1547, !dbg !16
  %1560 = select i1 %.not3, float %1558, float %1552, !dbg !16
  %1561 = bitcast float %1560 to i32, !dbg !16
  %1562 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1561), !dbg !16
  %1563 = bitcast i32 %1562 to float, !dbg !16
  %1564 = select i1 %.not3, float %1552, float %1563, !dbg !16
  %1565 = select i1 %.not3, float %1563, float %1558, !dbg !16
  %1566 = select i1 %.not3, float %1559, float %1553, !dbg !16
  %1567 = bitcast float %1566 to i32, !dbg !16
  %1568 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1567), !dbg !16
  %1569 = bitcast i32 %1568 to float, !dbg !16
  %1570 = select i1 %.not3, float %1553, float %1569, !dbg !16
  %1571 = select i1 %.not3, float %1569, float %1559, !dbg !16
  %1572 = extractelement <8 x float> %1462, i64 4, !dbg !16
  %1573 = extractelement <8 x float> %1462, i64 5, !dbg !16
  %1574 = extractelement <8 x float> %1462, i64 6, !dbg !16
  %1575 = extractelement <8 x float> %1462, i64 7, !dbg !16
  %1576 = select i1 %.not2, float %1573, float %1572, !dbg !16
  %1577 = bitcast float %1576 to i32, !dbg !16
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1577), !dbg !16
  %1579 = bitcast i32 %1578 to float, !dbg !16
  %1580 = select i1 %.not2, float %1572, float %1579, !dbg !16
  %1581 = select i1 %.not2, float %1579, float %1573, !dbg !16
  %1582 = select i1 %.not2, float %1575, float %1574, !dbg !16
  %1583 = bitcast float %1582 to i32, !dbg !16
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1583), !dbg !16
  %1585 = bitcast i32 %1584 to float, !dbg !16
  %1586 = select i1 %.not2, float %1574, float %1585, !dbg !16
  %1587 = select i1 %.not2, float %1585, float %1575, !dbg !16
  %1588 = select i1 %.not3, float %1586, float %1580, !dbg !16
  %1589 = bitcast float %1588 to i32, !dbg !16
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1589), !dbg !16
  %1591 = bitcast i32 %1590 to float, !dbg !16
  %1592 = select i1 %.not3, float %1580, float %1591, !dbg !16
  %1593 = select i1 %.not3, float %1591, float %1586, !dbg !16
  %1594 = select i1 %.not3, float %1587, float %1581, !dbg !16
  %1595 = bitcast float %1594 to i32, !dbg !16
  %1596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1595), !dbg !16
  %1597 = bitcast i32 %1596 to float, !dbg !16
  %1598 = select i1 %.not3, float %1581, float %1597, !dbg !16
  %1599 = select i1 %.not3, float %1597, float %1587, !dbg !16
  %1600 = extractelement <8 x float> %1491, i64 4, !dbg !16
  %1601 = extractelement <8 x float> %1491, i64 5, !dbg !16
  %1602 = extractelement <8 x float> %1491, i64 6, !dbg !16
  %1603 = extractelement <8 x float> %1491, i64 7, !dbg !16
  %1604 = select i1 %.not2, float %1601, float %1600, !dbg !16
  %1605 = bitcast float %1604 to i32, !dbg !16
  %1606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1605), !dbg !16
  %1607 = bitcast i32 %1606 to float, !dbg !16
  %1608 = select i1 %.not2, float %1600, float %1607, !dbg !16
  %1609 = select i1 %.not2, float %1607, float %1601, !dbg !16
  %1610 = select i1 %.not2, float %1603, float %1602, !dbg !16
  %1611 = bitcast float %1610 to i32, !dbg !16
  %1612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1187, i32 %1611), !dbg !16
  %1613 = bitcast i32 %1612 to float, !dbg !16
  %1614 = select i1 %.not2, float %1602, float %1613, !dbg !16
  %1615 = select i1 %.not2, float %1613, float %1603, !dbg !16
  %1616 = select i1 %.not3, float %1614, float %1608, !dbg !16
  %1617 = bitcast float %1616 to i32, !dbg !16
  %1618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1617), !dbg !16
  %1619 = select i1 %.not3, float %1615, float %1609, !dbg !16
  %1620 = bitcast float %1619 to i32, !dbg !16
  %1621 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1203, i32 %1620), !dbg !16
  br i1 true, label %1622, label %1686, !dbg !21

1622:                                             ; preds = %1171
  %1623 = bitcast i32 %1618 to float, !dbg !16
  %1624 = select i1 %.not3, float %1608, float %1623, !dbg !16
  %1625 = insertelement <4 x float> poison, float %1624, i64 0, !dbg !19
  %1626 = bitcast i32 %1621 to float, !dbg !16
  %1627 = select i1 %.not3, float %1609, float %1626, !dbg !16
  %1628 = insertelement <4 x float> %1625, float %1627, i64 1, !dbg !19
  %1629 = select i1 %.not3, float %1623, float %1614, !dbg !16
  %1630 = insertelement <4 x float> %1628, float %1629, i64 2, !dbg !19
  %1631 = select i1 %.not3, float %1626, float %1615, !dbg !16
  %1632 = insertelement <4 x float> %1630, float %1631, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1515, !dbg !19
  %normalized = fmul <4 x float> %1632, %row.reciprocal, !dbg !19
  %1633 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !20
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !20
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !20
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !20
  %1638 = bitcast i32 %1510 to float, !dbg !16
  %1639 = select i1 %.not3, float %1500, float %1638, !dbg !16
  %1640 = insertelement <4 x float> poison, float %1639, i64 0, !dbg !19
  %1641 = bitcast i32 %1513 to float, !dbg !16
  %1642 = select i1 %.not3, float %1501, float %1641, !dbg !16
  %1643 = insertelement <4 x float> %1640, float %1642, i64 1, !dbg !19
  %1644 = select i1 %.not3, float %1638, float %1506, !dbg !16
  %1645 = insertelement <4 x float> %1643, float %1644, i64 2, !dbg !19
  %1646 = select i1 %.not3, float %1641, float %1507, !dbg !16
  %1647 = insertelement <4 x float> %1645, float %1646, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1647, %row.reciprocal, !dbg !19
  %1648 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1649 = extractelement <4 x half> %1648, i64 3, !dbg !20
  %1650 = extractelement <4 x half> %1648, i64 2, !dbg !20
  %1651 = extractelement <4 x half> %1648, i64 1, !dbg !20
  %1652 = extractelement <4 x half> %1648, i64 0, !dbg !20
  %1653 = bitcast i32 %1400 to float, !dbg !16
  %1654 = select i1 %.not3, float %1390, float %1653, !dbg !16
  %1655 = insertelement <4 x float> poison, float %1654, i64 0, !dbg !19
  %1656 = bitcast i32 %1403 to float, !dbg !16
  %1657 = select i1 %.not3, float %1391, float %1656, !dbg !16
  %1658 = insertelement <4 x float> %1655, float %1657, i64 1, !dbg !19
  %1659 = select i1 %.not3, float %1653, float %1396, !dbg !16
  %1660 = insertelement <4 x float> %1658, float %1659, i64 2, !dbg !19
  %1661 = select i1 %.not3, float %1656, float %1397, !dbg !16
  %1662 = insertelement <4 x float> %1660, float %1661, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1297, !dbg !19
  %normalized74 = fmul <4 x float> %1662, %row.reciprocal73, !dbg !19
  %1663 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1664 = extractelement <4 x half> %1663, i64 3, !dbg !20
  %1665 = extractelement <4 x half> %1663, i64 2, !dbg !20
  %1666 = extractelement <4 x half> %1663, i64 1, !dbg !20
  %1667 = extractelement <4 x half> %1663, i64 0, !dbg !20
  %1668 = bitcast i32 %1292 to float, !dbg !16
  %1669 = select i1 %.not3, float %1282, float %1668, !dbg !16
  %1670 = insertelement <4 x float> poison, float %1669, i64 0, !dbg !19
  %1671 = bitcast i32 %1295 to float, !dbg !16
  %1672 = select i1 %.not3, float %1283, float %1671, !dbg !16
  %1673 = insertelement <4 x float> %1670, float %1672, i64 1, !dbg !19
  %1674 = select i1 %.not3, float %1668, float %1288, !dbg !16
  %1675 = insertelement <4 x float> %1673, float %1674, i64 2, !dbg !19
  %1676 = select i1 %.not3, float %1671, float %1289, !dbg !16
  %1677 = insertelement <4 x float> %1675, float %1676, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1677, %row.reciprocal73, !dbg !19
  %1678 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1679 = extractelement <4 x half> %1678, i64 3, !dbg !20
  %1680 = extractelement <4 x half> %1678, i64 2, !dbg !20
  %1681 = extractelement <4 x half> %1678, i64 1, !dbg !20
  %1682 = extractelement <4 x half> %1678, i64 0, !dbg !20
  %1683 = add nsw i64 %8, %13, !dbg !21
  %1684 = shl nsw i64 %1683, 19, !dbg !21
  %1685 = and i64 %1684, 16252928, !dbg !21
  %address.thread391 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local392 = zext nneg i32 %address.thread391 to i64, !dbg !13
  %address.local393 = add nsw i64 %16, %address.local392, !dbg !21
  %address.local394 = lshr i64 %address.local392, 6, !dbg !13
  %address.local395 = shl nuw nsw i64 %address.local394, 4, !dbg !21
  %address.local396 = add nsw i64 %address.local393, %address.local395, !dbg !21
  %address.local397 = and i64 %address.local392, 240, !dbg !21
  %address.local398 = sub nsw i64 %address.local396, %address.local397, !dbg !21
  %address.local399 = add nsw i64 %address.local398, %17, !dbg !21
  %address.local400 = shl nuw nsw i64 %address.local399, 7, !dbg !21
  %address.local401 = add nuw nsw i64 %1685, %address.local400, !dbg !21
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local403 = lshr i64 %address.local392, 2, !dbg !13
  %address.local404 = and i64 %address.local403, 12, !dbg !13
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local402, i64 %address.local404, !dbg !21
  store half %1682, ptr addrspace(1) %address.local405, align 2, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local407 = or disjoint i64 %address.local404, 1, !dbg !21
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local407, !dbg !21
  store half %1681, ptr addrspace(1) %address.local408, align 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local410 = or disjoint i64 %address.local404, 2, !dbg !21
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !21
  store half %1680, ptr addrspace(1) %address.local411, align 2, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local413 = or disjoint i64 %address.local404, 3, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local413, !dbg !21
  store half %1679, ptr addrspace(1) %address.local414, align 2, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local416 = or disjoint i64 %address.local404, 16, !dbg !21
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local416, !dbg !21
  store half %1667, ptr addrspace(1) %address.local417, align 2, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local419 = or disjoint i64 %address.local404, 17, !dbg !21
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local419, !dbg !21
  store half %1666, ptr addrspace(1) %address.local420, align 2, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local422 = or disjoint i64 %address.local404, 18, !dbg !21
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local422, !dbg !21
  store half %1665, ptr addrspace(1) %address.local423, align 2, !dbg !21
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !21
  %address.local425 = or disjoint i64 %address.local404, 19, !dbg !21
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local425, !dbg !21
  store half %1664, ptr addrspace(1) %address.local426, align 2, !dbg !21
  %address.local427 = shl nsw i64 %address.local399, 7, !dbg !21
  %address.local428 = add i64 %address.local427, 8192, !dbg !21
  %address.local429 = add nuw nsw i64 %1685, %address.local428, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local404, !dbg !21
  store half %1652, ptr addrspace(1) %address.local431, align 2, !dbg !21
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local407, !dbg !21
  store half %1651, ptr addrspace(1) %address.local433, align 2, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local410, !dbg !21
  store half %1650, ptr addrspace(1) %address.local435, align 2, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local413, !dbg !21
  store half %1649, ptr addrspace(1) %address.local437, align 2, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local416, !dbg !21
  store half %1637, ptr addrspace(1) %address.local439, align 2, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local419, !dbg !21
  store half %1636, ptr addrspace(1) %address.local441, align 2, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local422, !dbg !21
  store half %1635, ptr addrspace(1) %address.local443, align 2, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !21
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local444, i64 %address.local425, !dbg !21
  store half %1634, ptr addrspace(1) %address.local445, align 2, !dbg !21
  br label %1686, !dbg !21

1686:                                             ; preds = %1622, %1171
  br i1 true, label %1687, label %1725, !dbg !21

1687:                                             ; preds = %1686
  %1688 = insertelement <4 x float> poison, float %1592, i64 0, !dbg !19
  %1689 = insertelement <4 x float> %1688, float %1598, i64 1, !dbg !19
  %1690 = insertelement <4 x float> %1689, float %1593, i64 2, !dbg !19
  %1691 = insertelement <4 x float> %1690, float %1599, i64 3, !dbg !19
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1515, !dbg !19
  %normalized77 = fmul <4 x float> %1691, %row.reciprocal76, !dbg !19
  %1692 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1693 = extractelement <4 x half> %1692, i64 3, !dbg !20
  %1694 = extractelement <4 x half> %1692, i64 2, !dbg !20
  %1695 = extractelement <4 x half> %1692, i64 1, !dbg !20
  %1696 = extractelement <4 x half> %1692, i64 0, !dbg !20
  %1697 = insertelement <4 x float> poison, float %1483, i64 0, !dbg !19
  %1698 = insertelement <4 x float> %1697, float %1489, i64 1, !dbg !19
  %1699 = insertelement <4 x float> %1698, float %1484, i64 2, !dbg !19
  %1700 = insertelement <4 x float> %1699, float %1490, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1700, %row.reciprocal76, !dbg !19
  %1701 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1702 = extractelement <4 x half> %1701, i64 3, !dbg !20
  %1703 = extractelement <4 x half> %1701, i64 2, !dbg !20
  %1704 = extractelement <4 x half> %1701, i64 1, !dbg !20
  %1705 = extractelement <4 x half> %1701, i64 0, !dbg !20
  %1706 = insertelement <4 x float> poison, float %1374, i64 0, !dbg !19
  %1707 = insertelement <4 x float> %1706, float %1380, i64 1, !dbg !19
  %1708 = insertelement <4 x float> %1707, float %1375, i64 2, !dbg !19
  %1709 = insertelement <4 x float> %1708, float %1381, i64 3, !dbg !19
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1297, !dbg !19
  %normalized80 = fmul <4 x float> %1709, %row.reciprocal79, !dbg !19
  %1710 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %1711 = extractelement <4 x half> %1710, i64 3, !dbg !20
  %1712 = extractelement <4 x half> %1710, i64 2, !dbg !20
  %1713 = extractelement <4 x half> %1710, i64 1, !dbg !20
  %1714 = extractelement <4 x half> %1710, i64 0, !dbg !20
  %1715 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !19
  %1716 = insertelement <4 x float> %1715, float %1271, i64 1, !dbg !19
  %1717 = insertelement <4 x float> %1716, float %1266, i64 2, !dbg !19
  %1718 = insertelement <4 x float> %1717, float %1272, i64 3, !dbg !19
  %normalized81 = fmul <4 x float> %1718, %row.reciprocal79, !dbg !19
  %1719 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1720 = extractelement <4 x half> %1719, i64 3, !dbg !20
  %1721 = extractelement <4 x half> %1719, i64 2, !dbg !20
  %1722 = extractelement <4 x half> %1719, i64 1, !dbg !20
  %1723 = extractelement <4 x half> %1719, i64 0, !dbg !20
  %1724 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread446 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local447 = zext nneg i32 %address.thread446 to i64, !dbg !13
  %address.local448 = add nsw i64 %16, %address.local447, !dbg !21
  %address.local449 = lshr i64 %address.local447, 6, !dbg !13
  %address.local450 = shl nuw nsw i64 %address.local449, 4, !dbg !21
  %address.local451 = add nsw i64 %address.local448, %address.local450, !dbg !21
  %address.local452 = and i64 %address.local447, 240, !dbg !21
  %address.local453 = sub nsw i64 %address.local451, %address.local452, !dbg !21
  %address.local454 = add nsw i64 %address.local453, %17, !dbg !21
  %address.local455 = shl nuw nsw i64 %address.local454, 7, !dbg !21
  %address.local456 = add nuw nsw i64 %1724, %address.local455, !dbg !21
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local458 = lshr i64 %address.local447, 2, !dbg !13
  %address.local459 = and i64 %address.local458, 12, !dbg !13
  %address.local460 = or disjoint i64 %address.local459, 32, !dbg !21
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local457, i64 %address.local460, !dbg !21
  store half %1723, ptr addrspace(1) %address.local461, align 2, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local463 = or disjoint i64 %address.local459, 33, !dbg !21
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local463, !dbg !21
  store half %1722, ptr addrspace(1) %address.local464, align 2, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local466 = or disjoint i64 %address.local459, 34, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local466, !dbg !21
  store half %1721, ptr addrspace(1) %address.local467, align 2, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local469 = or disjoint i64 %address.local459, 35, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local469, !dbg !21
  store half %1720, ptr addrspace(1) %address.local470, align 2, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local472 = or disjoint i64 %address.local459, 48, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local472, !dbg !21
  store half %1714, ptr addrspace(1) %address.local473, align 2, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local475 = or disjoint i64 %address.local459, 49, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local474, i64 %address.local475, !dbg !21
  store half %1713, ptr addrspace(1) %address.local476, align 2, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local478 = or disjoint i64 %address.local459, 50, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local478, !dbg !21
  store half %1712, ptr addrspace(1) %address.local479, align 2, !dbg !21
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !21
  %address.local481 = or i64 %address.local458, 51, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local480, i64 %address.local481, !dbg !21
  store half %1711, ptr addrspace(1) %address.local482, align 2, !dbg !21
  %address.local483 = shl nsw i64 %address.local454, 7, !dbg !21
  %address.local484 = add i64 %address.local483, 8192, !dbg !21
  %address.local485 = add nuw nsw i64 %1724, %address.local484, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local486, i64 %address.local460, !dbg !21
  store half %1705, ptr addrspace(1) %address.local487, align 2, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local488, i64 %address.local463, !dbg !21
  store half %1704, ptr addrspace(1) %address.local489, align 2, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local466, !dbg !21
  store half %1703, ptr addrspace(1) %address.local491, align 2, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local469, !dbg !21
  store half %1702, ptr addrspace(1) %address.local493, align 2, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local472, !dbg !21
  store half %1696, ptr addrspace(1) %address.local495, align 2, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local475, !dbg !21
  store half %1695, ptr addrspace(1) %address.local497, align 2, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local478, !dbg !21
  store half %1694, ptr addrspace(1) %address.local499, align 2, !dbg !21
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !21
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local500, i64 %address.local481, !dbg !21
  store half %1693, ptr addrspace(1) %address.local501, align 2, !dbg !21
  br label %1725, !dbg !21

1725:                                             ; preds = %1687, %1686
  br i1 true, label %1726, label %1764, !dbg !21

1726:                                             ; preds = %1725
  %1727 = insertelement <4 x float> poison, float %1564, i64 0, !dbg !19
  %1728 = insertelement <4 x float> %1727, float %1570, i64 1, !dbg !19
  %1729 = insertelement <4 x float> %1728, float %1565, i64 2, !dbg !19
  %1730 = insertelement <4 x float> %1729, float %1571, i64 3, !dbg !19
  %row.reciprocal82 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1515, !dbg !19
  %normalized83 = fmul <4 x float> %1730, %row.reciprocal82, !dbg !19
  %1731 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %1732 = extractelement <4 x half> %1731, i64 3, !dbg !20
  %1733 = extractelement <4 x half> %1731, i64 2, !dbg !20
  %1734 = extractelement <4 x half> %1731, i64 1, !dbg !20
  %1735 = extractelement <4 x half> %1731, i64 0, !dbg !20
  %1736 = insertelement <4 x float> poison, float %1454, i64 0, !dbg !19
  %1737 = insertelement <4 x float> %1736, float %1460, i64 1, !dbg !19
  %1738 = insertelement <4 x float> %1737, float %1455, i64 2, !dbg !19
  %1739 = insertelement <4 x float> %1738, float %1461, i64 3, !dbg !19
  %normalized84 = fmul <4 x float> %1739, %row.reciprocal82, !dbg !19
  %1740 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !20
  %1741 = extractelement <4 x half> %1740, i64 3, !dbg !20
  %1742 = extractelement <4 x half> %1740, i64 2, !dbg !20
  %1743 = extractelement <4 x half> %1740, i64 1, !dbg !20
  %1744 = extractelement <4 x half> %1740, i64 0, !dbg !20
  %1745 = insertelement <4 x float> poison, float %1346, i64 0, !dbg !19
  %1746 = insertelement <4 x float> %1745, float %1352, i64 1, !dbg !19
  %1747 = insertelement <4 x float> %1746, float %1347, i64 2, !dbg !19
  %1748 = insertelement <4 x float> %1747, float %1353, i64 3, !dbg !19
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1297, !dbg !19
  %normalized86 = fmul <4 x float> %1748, %row.reciprocal85, !dbg !19
  %1749 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %1750 = extractelement <4 x half> %1749, i64 3, !dbg !20
  %1751 = extractelement <4 x half> %1749, i64 2, !dbg !20
  %1752 = extractelement <4 x half> %1749, i64 1, !dbg !20
  %1753 = extractelement <4 x half> %1749, i64 0, !dbg !20
  %1754 = insertelement <4 x float> poison, float %1236, i64 0, !dbg !19
  %1755 = insertelement <4 x float> %1754, float %1242, i64 1, !dbg !19
  %1756 = insertelement <4 x float> %1755, float %1237, i64 2, !dbg !19
  %1757 = insertelement <4 x float> %1756, float %1243, i64 3, !dbg !19
  %normalized87 = fmul <4 x float> %1757, %row.reciprocal85, !dbg !19
  %1758 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !20
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !20
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !20
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !20
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !20
  %1763 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread502 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local503 = zext nneg i32 %address.thread502 to i64, !dbg !13
  %address.local504 = add nsw i64 %16, %address.local503, !dbg !21
  %address.local505 = lshr i64 %address.local503, 6, !dbg !13
  %address.local506 = shl nuw nsw i64 %address.local505, 4, !dbg !21
  %address.local507 = add nsw i64 %address.local504, %address.local506, !dbg !21
  %address.local508 = and i64 %address.local503, 240, !dbg !21
  %address.local509 = sub nsw i64 %address.local507, %address.local508, !dbg !21
  %address.local510 = add nsw i64 %address.local509, %17, !dbg !21
  %address.local511 = shl nuw nsw i64 %address.local510, 7, !dbg !21
  %address.local512 = add nuw nsw i64 %1763, %address.local511, !dbg !21
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local514 = lshr i64 %address.local503, 2, !dbg !13
  %address.local515 = and i64 %address.local514, 12, !dbg !13
  %address.local516 = or disjoint i64 %address.local515, 64, !dbg !21
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local516, !dbg !21
  store half %1762, ptr addrspace(1) %address.local517, align 2, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local519 = or disjoint i64 %address.local515, 65, !dbg !21
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local518, i64 %address.local519, !dbg !21
  store half %1761, ptr addrspace(1) %address.local520, align 2, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local522 = or disjoint i64 %address.local515, 66, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local522, !dbg !21
  store half %1760, ptr addrspace(1) %address.local523, align 2, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local525 = or disjoint i64 %address.local515, 67, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local524, i64 %address.local525, !dbg !21
  store half %1759, ptr addrspace(1) %address.local526, align 2, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local528 = or disjoint i64 %address.local515, 80, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local528, !dbg !21
  store half %1753, ptr addrspace(1) %address.local529, align 2, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local531 = or disjoint i64 %address.local515, 81, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local530, i64 %address.local531, !dbg !21
  store half %1752, ptr addrspace(1) %address.local532, align 2, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local534 = or disjoint i64 %address.local515, 82, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !21
  store half %1751, ptr addrspace(1) %address.local535, align 2, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !21
  %address.local537 = or disjoint i64 %address.local515, 83, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local537, !dbg !21
  store half %1750, ptr addrspace(1) %address.local538, align 2, !dbg !21
  %address.local539 = shl nsw i64 %address.local510, 7, !dbg !21
  %address.local540 = add i64 %address.local539, 8192, !dbg !21
  %address.local541 = add nuw nsw i64 %1763, %address.local540, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local542, i64 %address.local516, !dbg !21
  store half %1744, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local519, !dbg !21
  store half %1743, ptr addrspace(1) %address.local545, align 2, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local522, !dbg !21
  store half %1742, ptr addrspace(1) %address.local547, align 2, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local525, !dbg !21
  store half %1741, ptr addrspace(1) %address.local549, align 2, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local528, !dbg !21
  store half %1735, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local531, !dbg !21
  store half %1734, ptr addrspace(1) %address.local553, align 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local534, !dbg !21
  store half %1733, ptr addrspace(1) %address.local555, align 2, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local537, !dbg !21
  store half %1732, ptr addrspace(1) %address.local557, align 2, !dbg !21
  br label %1764, !dbg !21

1764:                                             ; preds = %1726, %1725
  br i1 true, label %1765, label %1803, !dbg !21

1765:                                             ; preds = %1764
  %1766 = insertelement <4 x float> poison, float %1536, i64 0, !dbg !19
  %1767 = insertelement <4 x float> %1766, float %1542, i64 1, !dbg !19
  %1768 = insertelement <4 x float> %1767, float %1537, i64 2, !dbg !19
  %1769 = insertelement <4 x float> %1768, float %1543, i64 3, !dbg !19
  %row.reciprocal88 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1515, !dbg !19
  %normalized89 = fmul <4 x float> %1769, %row.reciprocal88, !dbg !19
  %1770 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %1771 = extractelement <4 x half> %1770, i64 3, !dbg !20
  %1772 = extractelement <4 x half> %1770, i64 2, !dbg !20
  %1773 = extractelement <4 x half> %1770, i64 1, !dbg !20
  %1774 = extractelement <4 x half> %1770, i64 0, !dbg !20
  %1775 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !19
  %1776 = insertelement <4 x float> %1775, float %1431, i64 1, !dbg !19
  %1777 = insertelement <4 x float> %1776, float %1426, i64 2, !dbg !19
  %1778 = insertelement <4 x float> %1777, float %1432, i64 3, !dbg !19
  %normalized90 = fmul <4 x float> %1778, %row.reciprocal88, !dbg !19
  %1779 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !20
  %1780 = extractelement <4 x half> %1779, i64 3, !dbg !20
  %1781 = extractelement <4 x half> %1779, i64 2, !dbg !20
  %1782 = extractelement <4 x half> %1779, i64 1, !dbg !20
  %1783 = extractelement <4 x half> %1779, i64 0, !dbg !20
  %1784 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !19
  %1785 = insertelement <4 x float> %1784, float %1324, i64 1, !dbg !19
  %1786 = insertelement <4 x float> %1785, float %1319, i64 2, !dbg !19
  %1787 = insertelement <4 x float> %1786, float %1325, i64 3, !dbg !19
  %row.reciprocal91 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1297, !dbg !19
  %normalized92 = fmul <4 x float> %1787, %row.reciprocal91, !dbg !19
  %1788 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !20
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !20
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !20
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !20
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !20
  %1793 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !19
  %1794 = insertelement <4 x float> %1793, float %1213, i64 1, !dbg !19
  %1795 = insertelement <4 x float> %1794, float %1208, i64 2, !dbg !19
  %1796 = insertelement <4 x float> %1795, float %1214, i64 3, !dbg !19
  %normalized93 = fmul <4 x float> %1796, %row.reciprocal91, !dbg !19
  %1797 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !20
  %1798 = extractelement <4 x half> %1797, i64 3, !dbg !20
  %1799 = extractelement <4 x half> %1797, i64 2, !dbg !20
  %1800 = extractelement <4 x half> %1797, i64 1, !dbg !20
  %1801 = extractelement <4 x half> %1797, i64 0, !dbg !20
  %1802 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread558 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local559 = zext nneg i32 %address.thread558 to i64, !dbg !13
  %address.local560 = add nsw i64 %16, %address.local559, !dbg !21
  %address.local561 = lshr i64 %address.local559, 6, !dbg !13
  %address.local562 = shl nuw nsw i64 %address.local561, 4, !dbg !21
  %address.local563 = add nsw i64 %address.local560, %address.local562, !dbg !21
  %address.local564 = and i64 %address.local559, 240, !dbg !21
  %address.local565 = sub nsw i64 %address.local563, %address.local564, !dbg !21
  %address.local566 = add nsw i64 %address.local565, %17, !dbg !21
  %address.local567 = shl nuw nsw i64 %address.local566, 7, !dbg !21
  %address.local568 = add nuw nsw i64 %1802, %address.local567, !dbg !21
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local570 = lshr i64 %address.local559, 2, !dbg !13
  %address.local571 = and i64 %address.local570, 12, !dbg !13
  %address.local572 = or disjoint i64 %address.local571, 96, !dbg !21
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local572, !dbg !21
  store half %1801, ptr addrspace(1) %address.local573, align 2, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local575 = or disjoint i64 %address.local571, 97, !dbg !21
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local574, i64 %address.local575, !dbg !21
  store half %1800, ptr addrspace(1) %address.local576, align 2, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local578 = or disjoint i64 %address.local571, 98, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local578, !dbg !21
  store half %1799, ptr addrspace(1) %address.local579, align 2, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local581 = or disjoint i64 %address.local571, 99, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local580, i64 %address.local581, !dbg !21
  store half %1798, ptr addrspace(1) %address.local582, align 2, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local584 = or disjoint i64 %address.local571, 112, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local584, !dbg !21
  store half %1792, ptr addrspace(1) %address.local585, align 2, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local587 = or disjoint i64 %address.local571, 113, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local586, i64 %address.local587, !dbg !21
  store half %1791, ptr addrspace(1) %address.local588, align 2, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local590 = or disjoint i64 %address.local571, 114, !dbg !21
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local590, !dbg !21
  store half %1790, ptr addrspace(1) %address.local591, align 2, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !21
  %address.local593 = or i64 %address.local570, 115, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local593, !dbg !21
  store half %1789, ptr addrspace(1) %address.local594, align 2, !dbg !21
  %address.local595 = shl nsw i64 %address.local566, 7, !dbg !21
  %address.local596 = add i64 %address.local595, 8192, !dbg !21
  %address.local597 = add nuw nsw i64 %1802, %address.local596, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local572, !dbg !21
  store half %1783, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local575, !dbg !21
  store half %1782, ptr addrspace(1) %address.local601, align 2, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local602, i64 %address.local578, !dbg !21
  store half %1781, ptr addrspace(1) %address.local603, align 2, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local581, !dbg !21
  store half %1780, ptr addrspace(1) %address.local605, align 2, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local584, !dbg !21
  store half %1774, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local587, !dbg !21
  store half %1773, ptr addrspace(1) %address.local609, align 2, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local590, !dbg !21
  store half %1772, ptr addrspace(1) %address.local611, align 2, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local612, i64 %address.local593, !dbg !21
  store half %1771, ptr addrspace(1) %address.local613, align 2, !dbg !21
  br label %1803, !dbg !21

1803:                                             ; preds = %1765, %1764
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

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
!26 = !DILocation(line: 84, column: 21, scope: !4)
!27 = !DILocation(line: 85, column: 22, scope: !4)
!28 = !DILocation(line: 86, column: 20, scope: !4)
!29 = !DILocation(line: 87, column: 16, scope: !4)
!30 = !DILocation(line: 94, column: 20, scope: !4)
!31 = !DILocation(line: 70, column: 23, scope: !4)
!32 = !{i32 256, i32 1, i32 1}
!33 = !{i32 2, !"Debug Info Version", i32 3}
!34 = !{i32 0, i32 256}
