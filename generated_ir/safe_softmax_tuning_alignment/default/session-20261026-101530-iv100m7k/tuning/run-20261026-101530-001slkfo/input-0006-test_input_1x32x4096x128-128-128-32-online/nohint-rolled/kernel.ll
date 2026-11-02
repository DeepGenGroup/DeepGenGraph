; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/nohint-rolled/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/nohint-rolled/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 7, !dbg !10
  %.lobit = ashr i32 %7, 31, !dbg !12
  %10 = xor i32 %7, %.lobit, !dbg !12
  %11 = sdiv i32 %10, 32, !dbg !12
  %.lobit1 = ashr i32 %7, 31, !dbg !12
  %12 = xor i32 %11, %.lobit1, !dbg !12
  %13 = sext i32 %12 to i64, !dbg !12
  %14 = add nsw i64 %6, %13, !dbg !12
  %15 = and i64 %14, 31, !dbg !12
  %16 = shl nsw i64 %8, 7, !dbg !12
  %17 = mul nsw i64 %13, -4096, !dbg !12
  %18 = add nsw i64 %16, %17, !dbg !12
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !39
  %20 = zext nneg i32 %19 to i64, !dbg !13
  %21 = lshr i64 %20, 6, !dbg !13
  %22 = or disjoint i64 %9, 127, !dbg !14
  %23 = add nsw i64 %9, 128, !dbg !14
  %24 = icmp uge i64 %23, %22, !dbg !14
  br label %25, !dbg !8

25:                                               ; preds = %420, %4
  %26 = phi i64 [ %1369, %420 ], [ 0, %4 ], !dbg !8
  %27 = phi [2 x <8 x float>] [ %1354, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %28 = phi [2 x <1 x float>] [ %960, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %29 = phi [2 x <1 x float>] [ %1356, %420 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %30 = phi [2 x <8 x float>] [ %1360, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %31 = phi [2 x <8 x float>] [ %1364, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %32 = phi [2 x <8 x float>] [ %1368, %420 ], [ zeroinitializer, %4 ], !dbg !8
  %.not = icmp sgt i64 %26, %8, !dbg !8
  br i1 %.not, label %1370, label %33, !dbg !8

33:                                               ; preds = %25
  %34 = shl nsw i64 %26, 7, !dbg !8
  %35 = lshr i64 %26, 5, !dbg !16
  %36 = shl i64 %35, 12, !dbg !16
  %37 = sub i64 %34, %36, !dbg !16
  br label %38, !dbg !21

38:                                               ; preds = %42, %33
  %39 = phi i64 [ %234, %42 ], [ 0, %33 ], !dbg !21
  %40 = phi [2 x <32 x float>] [ %233, %42 ], [ zeroinitializer, %33 ], !dbg !21
  %41 = icmp samesign ult i64 %39, 4, !dbg !21
  br i1 %41, label %42, label %235, !dbg !21

42:                                               ; preds = %38
  %43 = shl nuw nsw i64 %39, 5, !dbg !21
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %44 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx68 = shl nsw i64 %18, 8, !dbg !12
  %45 = getelementptr i8, ptr addrspace(1) %44, i64 %.idx68, !dbg !12
  %46 = getelementptr [2 x i8], ptr addrspace(1) %45, i64 %43, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local130 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local131 = and i64 %address.local130, 1536, !dbg !12
  %address.local132 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local131, !dbg !12
  %address.local133 = ptrtoint ptr addrspace(3) %address.local132 to i64, !dbg !12
  %address.local134 = trunc i64 %address.local133 to i32, !dbg !12
  %47 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local134), !dbg !12
  %48 = zext i32 %47 to i64, !dbg !12
  %49 = inttoptr i64 %48 to ptr addrspace(3), !dbg !12
  %address.local135 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local136 = and i64 %address.local135, 16256, !dbg !12
  %address.local137 = lshr i64 %address.local, 4, !dbg !12
  %address.local138 = and i64 %address.local137, 8, !dbg !12
  %address.local139 = or disjoint i64 %address.local136, %address.local138, !dbg !12
  %address.local140 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local139, !dbg !12
  %50 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %51 = ptrtoint ptr addrspace(1) %address.local140 to i64, !dbg !12
  %52 = sub i64 %51, %50, !dbg !12
  %dma.buffer.offset876 = trunc i64 %52 to i32, !dbg !12
  %53 = trunc i64 %50 to i32, !dbg !12
  %54 = insertelement <4 x i32> poison, i32 %53, i64 0, !dbg !12
  %55 = lshr i64 %50, 32, !dbg !12
  %56 = trunc i64 %55 to i32, !dbg !12
  %57 = insertelement <4 x i32> %54, i32 %56, i64 1, !dbg !12
  %58 = insertelement <4 x i32> %57, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc877 = insertelement <4 x i32> %58, i32 131072, i64 3, !dbg !12
  %59 = ptrtoint ptr addrspace(3) %49 to i32, !dbg !12
  %dma.buffer.lds878 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %59), !dbg !12
  %60 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds878, i32 %dma.buffer.offset876, <4 x i32> %dma.buffer.rsrc877) #10, !dbg !12
  %address.local141 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local142 = shl nuw nsw i64 %address.local141, 3, !dbg !12
  %address.local143 = and i64 %address.local142, 3584, !dbg !12
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !12
  %address.local145 = ptrtoint ptr addrspace(3) %address.local144 to i64, !dbg !12
  %address.local146 = trunc i64 %address.local145 to i32, !dbg !12
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local146), !dbg !12
  %62 = zext i32 %61 to i64, !dbg !12
  %63 = inttoptr i64 %62 to ptr addrspace(3), !dbg !12
  %address.local147 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local148 = and i64 %address.local147, 16256, !dbg !12
  %address.local149 = lshr i64 %address.local141, 4, !dbg !12
  %address.local150 = and i64 %address.local149, 24, !dbg !12
  %address.local151 = or disjoint i64 %address.local148, %address.local150, !dbg !12
  %address.local152 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local151, !dbg !12
  %64 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %65 = ptrtoint ptr addrspace(1) %address.local152 to i64, !dbg !12
  %66 = sub i64 %65, %64, !dbg !12
  %dma.buffer.offset873 = trunc i64 %66 to i32, !dbg !12
  %67 = trunc i64 %64 to i32, !dbg !12
  %68 = insertelement <4 x i32> poison, i32 %67, i64 0, !dbg !12
  %69 = lshr i64 %64, 32, !dbg !12
  %70 = trunc i64 %69 to i32, !dbg !12
  %71 = insertelement <4 x i32> %68, i32 %70, i64 1, !dbg !12
  %72 = insertelement <4 x i32> %71, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc874 = insertelement <4 x i32> %72, i32 131072, i64 3, !dbg !12
  %73 = ptrtoint ptr addrspace(3) %63 to i32, !dbg !12
  %dma.buffer.lds875 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %73), !dbg !12
  %74 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds875, i32 %dma.buffer.offset873, <4 x i32> %dma.buffer.rsrc874) #10, !dbg !12
  %75 = shl nuw nsw i64 %39, 10, !dbg !16
  %76 = add nuw i64 %26, %75, !dbg !16
  %.lobit69 = ashr i64 %76, 63, !dbg !16
  %77 = xor i64 %76, %.lobit69, !dbg !16
  %78 = sdiv i64 %77, 4096, !dbg !16
  %.lobit70 = ashr i64 %76, 63, !dbg !16
  %79 = xor i64 %78, %.lobit70, !dbg !16
  %80 = add nsw i64 %79, %6, !dbg !16
  %81 = add nuw nsw i64 %43, %35, !dbg !16
  %82 = and i64 %81, 127, !dbg !16
  %83 = shl i64 %80, 19, !dbg !16
  %84 = and i64 %83, 16252928, !dbg !16
  %85 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %84, !dbg !16
  %86 = getelementptr [2 x i8], ptr addrspace(1) %85, i64 %82, !dbg !16
  %.idx71 = shl i64 %37, 8, !dbg !16
  %87 = getelementptr i8, ptr addrspace(1) %86, i64 %.idx71, !dbg !16
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local131, !dbg !16
  %address.local154 = ptrtoint ptr addrspace(3) %address.local153 to i64, !dbg !16
  %address.local155 = trunc i64 %address.local154 to i32, !dbg !16
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local155), !dbg !16
  %89 = zext i32 %88 to i64, !dbg !16
  %90 = inttoptr i64 %89 to ptr addrspace(3), !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %address.local139, !dbg !16
  %91 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %92 = ptrtoint ptr addrspace(1) %address.local156 to i64, !dbg !16
  %93 = sub i64 %92, %91, !dbg !16
  %dma.buffer.offset870 = trunc i64 %93 to i32, !dbg !16
  %94 = trunc i64 %91 to i32, !dbg !16
  %95 = insertelement <4 x i32> poison, i32 %94, i64 0, !dbg !16
  %96 = lshr i64 %91, 32, !dbg !16
  %97 = trunc i64 %96 to i32, !dbg !16
  %98 = insertelement <4 x i32> %95, i32 %97, i64 1, !dbg !16
  %99 = insertelement <4 x i32> %98, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc871 = insertelement <4 x i32> %99, i32 131072, i64 3, !dbg !16
  %100 = ptrtoint ptr addrspace(3) %90 to i32, !dbg !16
  %dma.buffer.lds872 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %100), !dbg !16
  %101 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds872, i32 %dma.buffer.offset870, <4 x i32> %dma.buffer.rsrc871) #10, !dbg !16
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local143, !dbg !16
  %address.local158 = ptrtoint ptr addrspace(3) %address.local157 to i64, !dbg !16
  %address.local159 = trunc i64 %address.local158 to i32, !dbg !16
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local159), !dbg !16
  %103 = zext i32 %102 to i64, !dbg !16
  %104 = inttoptr i64 %103 to ptr addrspace(3), !dbg !16
  %address.local160 = getelementptr [2 x i8], ptr addrspace(1) %87, i64 %address.local151, !dbg !16
  %105 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %106 = ptrtoint ptr addrspace(1) %address.local160 to i64, !dbg !16
  %107 = sub i64 %106, %105, !dbg !16
  %dma.buffer.offset = trunc i64 %107 to i32, !dbg !16
  %108 = trunc i64 %105 to i32, !dbg !16
  %109 = insertelement <4 x i32> poison, i32 %108, i64 0, !dbg !16
  %110 = lshr i64 %105, 32, !dbg !16
  %111 = trunc i64 %110 to i32, !dbg !16
  %112 = insertelement <4 x i32> %109, i32 %111, i64 1, !dbg !16
  %113 = insertelement <4 x i32> %112, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc = insertelement <4 x i32> %113, i32 131072, i64 3, !dbg !16
  %114 = ptrtoint ptr addrspace(3) %104 to i32, !dbg !16
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %114), !dbg !16
  %115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
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
  %116 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !13
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local175 = getelementptr i8, ptr addrspace(3) %address.local174, i64 4096, !dbg !13
  %117 = load <4 x half>, ptr addrspace(3) %address.local175, align 2, !dbg !13
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local177 = getelementptr i8, ptr addrspace(3) %address.local176, i64 1024, !dbg !13
  %118 = load <4 x half>, ptr addrspace(3) %address.local177, align 2, !dbg !13
  %119 = fmul <4 x half> %116, splat (half 0xH3015), !dbg !13
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local172, !dbg !13
  %address.local179 = getelementptr i8, ptr addrspace(3) %address.local178, i64 5120, !dbg !13
  %120 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !13
  %121 = fmul <4 x half> %117, splat (half 0xH3015), !dbg !13
  %122 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !13
  %123 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local180 = lshr i32 %address.thread, 2, !dbg !7
  %address.local181 = and i32 %address.local180, 4, !dbg !7
  %address.local182 = shl nuw nsw i32 %address.thread, 1, !dbg !7
  %address.local183 = and i32 %address.local182, 24, !dbg !7
  %address.local184 = or disjoint i32 %address.local181, %address.local183, !dbg !7
  %address.local185 = shl nuw nsw i32 %address.thread, 5, !dbg !7
  %address.local186 = and i32 %address.local185, 1024, !dbg !7
  %address.local187 = or disjoint i32 %address.local184, %address.local186, !dbg !7
  %address.local188 = shl nuw nsw i32 %address.thread, 7, !dbg !7
  %address.local189 = and i32 %address.local188, 384, !dbg !7
  %address.local190 = or disjoint i32 %address.local187, %address.local189, !dbg !7
  %address.local191 = zext nneg i32 %address.local190 to i64, !dbg !7
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !7
  %124 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !7
  %125 = extractvalue [2 x <32 x float>] %40, 0, !dbg !36
  %126 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %address.local193 = lshr i32 %address.thread, 2, !dbg !7
  %address.local194 = and i32 %address.local193, 4, !dbg !7
  %address.local195 = shl nuw nsw i32 %address.thread, 1, !dbg !7
  %address.local196 = and i32 %address.local195, 24, !dbg !7
  %address.local197 = or disjoint i32 %address.local194, %address.local196, !dbg !7
  %address.local198 = shl nuw nsw i32 %address.thread, 5, !dbg !7
  %address.local199 = and i32 %address.local198, 1024, !dbg !7
  %address.local200 = or disjoint i32 %address.local197, %address.local199, !dbg !7
  %address.local201 = shl nuw nsw i32 %address.thread, 7, !dbg !7
  %address.local202 = and i32 %address.local201, 384, !dbg !7
  %address.local203 = or disjoint i32 %address.local200, %address.local202, !dbg !7
  %address.local204 = zext nneg i32 %address.local203 to i64, !dbg !7
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local204, !dbg !7
  %address.local206 = getelementptr i8, ptr addrspace(3) %address.local205, i64 4096, !dbg !7
  %127 = load <4 x half>, ptr addrspace(3) %address.local206, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %124, <4 x float> %126) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %129 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %127, <4 x float> %128) #9, !dbg !7
  %131 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %132 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %133 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !36
  %134 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !36
  %135 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !36
  %136 = shufflevector <32 x float> %125, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !36
  %137 = extractvalue [2 x <32 x float>] %40, 1, !dbg !36
  %138 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %139 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %140 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %141 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %142 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !36
  %143 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !36
  %144 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !36
  %145 = shufflevector <32 x float> %137, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !36
  %address.local207 = shl nuw nsw i64 %address.local, 7, !dbg !7
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !7
  %address.local209 = lshr i64 %address.local, 2, !dbg !7
  %address.local210 = mul nsw i64 %address.local209, -512, !dbg !7
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local210, !dbg !7
  %address.local212 = shl nuw nsw i64 %address.local, 1, !dbg !7
  %address.local213 = and i64 %address.local212, 24, !dbg !7
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local211, i64 %address.local213, !dbg !7
  %address.local215 = lshr i64 %address.local, 2, !dbg !13
  %address.local216 = and i64 %address.local215, 12, !dbg !13
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local216, !dbg !7
  %address.local218 = and i64 %address.local, 63, !dbg !13
  %address.local219 = lshr i64 %address.local218, 5, !dbg !13
  %address.local220 = mul nuw nsw i64 %address.local219, 1016, !dbg !13
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local220, !dbg !7
  %address.local222 = getelementptr i8, ptr addrspace(3) %address.local221, i64 64, !dbg !7
  %146 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !7
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !7
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local216, !dbg !7
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local220, !dbg !7
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local210, !dbg !7
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local213, !dbg !7
  %address.local228 = getelementptr i8, ptr addrspace(3) %address.local227, i64 4160, !dbg !7
  %147 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !7
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %146, <4 x float> %129) #9, !dbg !7
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !7
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local210, !dbg !7
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local213, !dbg !7
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local216, !dbg !7
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local220, !dbg !7
  %address.local234 = getelementptr i8, ptr addrspace(3) %address.local233, i64 128, !dbg !7
  %149 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !7
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %147, <4 x float> %148) #9, !dbg !7
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !7
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local216, !dbg !7
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local220, !dbg !7
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local210, !dbg !7
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local213, !dbg !7
  %address.local240 = getelementptr i8, ptr addrspace(3) %address.local239, i64 4224, !dbg !7
  %151 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !7
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %149, <4 x float> %131) #9, !dbg !7
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %151, <4 x float> %152) #9, !dbg !7
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !7
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local210, !dbg !7
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local213, !dbg !7
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local216, !dbg !7
  %address.local245 = getelementptr [2 x i8], ptr addrspace(3) %address.local244, i64 %address.local220, !dbg !7
  %address.local246 = getelementptr i8, ptr addrspace(3) %address.local245, i64 192, !dbg !7
  %154 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !7
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local207, !dbg !7
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local216, !dbg !7
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local220, !dbg !7
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local210, !dbg !7
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) %address.local250, i64 %address.local213, !dbg !7
  %address.local252 = getelementptr i8, ptr addrspace(3) %address.local251, i64 4288, !dbg !7
  %155 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !7
  %156 = shufflevector <4 x float> %153, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %154, <4 x float> %132) #9, !dbg !7
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local191, !dbg !7
  %address.local254 = getelementptr i8, ptr addrspace(3) %address.local253, i64 1024, !dbg !7
  %158 = load <4 x half>, ptr addrspace(3) %address.local254, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %155, <4 x float> %157) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %160 = shufflevector <4 x float> %159, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %address.local255 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local204, !dbg !7
  %address.local256 = getelementptr i8, ptr addrspace(3) %address.local255, i64 5120, !dbg !7
  %161 = load <4 x half>, ptr addrspace(3) %address.local256, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %158, <4 x float> %133) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %161, <4 x float> %162) #9, !dbg !7
  %address.local257 = add nuw nsw i64 %address.local207, 512, !dbg !7
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !7
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local210, !dbg !7
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local213, !dbg !7
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local216, !dbg !7
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) %address.local261, i64 %address.local220, !dbg !7
  %address.local263 = getelementptr i8, ptr addrspace(3) %address.local262, i64 64, !dbg !7
  %164 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !7
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !7
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local216, !dbg !7
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local220, !dbg !7
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local210, !dbg !7
  %address.local268 = getelementptr [2 x i8], ptr addrspace(3) %address.local267, i64 %address.local213, !dbg !7
  %address.local269 = getelementptr i8, ptr addrspace(3) %address.local268, i64 4160, !dbg !7
  %165 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !7
  %166 = shufflevector <4 x float> %163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %164, <4 x float> %134) #9, !dbg !7
  %address.local270 = add nuw nsw i64 %address.local207, 512, !dbg !7
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !7
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local210, !dbg !7
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local213, !dbg !7
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local216, !dbg !7
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) %address.local274, i64 %address.local220, !dbg !7
  %address.local276 = getelementptr i8, ptr addrspace(3) %address.local275, i64 128, !dbg !7
  %168 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %165, <4 x float> %167) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %170 = shufflevector <4 x float> %169, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local270, !dbg !7
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local216, !dbg !7
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local220, !dbg !7
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local210, !dbg !7
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) %address.local280, i64 %address.local213, !dbg !7
  %address.local282 = getelementptr i8, ptr addrspace(3) %address.local281, i64 4224, !dbg !7
  %171 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %168, <4 x float> %135) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %171, <4 x float> %172) #9, !dbg !7
  %address.local283 = add nuw nsw i64 %address.local207, 512, !dbg !7
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local283, !dbg !7
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local210, !dbg !7
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local213, !dbg !7
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local216, !dbg !7
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) %address.local287, i64 %address.local220, !dbg !7
  %address.local289 = getelementptr i8, ptr addrspace(3) %address.local288, i64 192, !dbg !7
  %174 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !7
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local283, !dbg !7
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local216, !dbg !7
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local220, !dbg !7
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local210, !dbg !7
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local213, !dbg !7
  %address.local295 = getelementptr i8, ptr addrspace(3) %address.local294, i64 4288, !dbg !7
  %175 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !7
  %176 = shufflevector <4 x float> %173, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %119, <4 x half> %174, <4 x float> %136) #9, !dbg !7
  %178 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %175, <4 x float> %177) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %180 = shufflevector <4 x float> %179, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %181 = load <4 x half>, ptr addrspace(3) %address.local206, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %178, <4 x float> %138) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %181, <4 x float> %182) #9, !dbg !7
  %184 = load <4 x half>, ptr addrspace(3) %address.local222, align 2, !dbg !7
  %185 = load <4 x half>, ptr addrspace(3) %address.local228, align 2, !dbg !7
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %184, <4 x float> %139) #9, !dbg !7
  %187 = load <4 x half>, ptr addrspace(3) %address.local234, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %185, <4 x float> %186) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %189 = load <4 x half>, ptr addrspace(3) %address.local240, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %187, <4 x float> %140) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %189, <4 x float> %190) #9, !dbg !7
  %192 = load <4 x half>, ptr addrspace(3) %address.local246, align 2, !dbg !7
  %193 = load <4 x half>, ptr addrspace(3) %address.local252, align 2, !dbg !7
  %194 = shufflevector <4 x float> %191, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %192, <4 x float> %141) #9, !dbg !7
  %196 = load <4 x half>, ptr addrspace(3) %address.local254, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %193, <4 x float> %195) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %198 = shufflevector <4 x float> %197, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %199 = load <4 x half>, ptr addrspace(3) %address.local256, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %196, <4 x float> %142) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %199, <4 x float> %200) #9, !dbg !7
  %202 = load <4 x half>, ptr addrspace(3) %address.local263, align 2, !dbg !7
  %203 = load <4 x half>, ptr addrspace(3) %address.local269, align 2, !dbg !7
  %204 = shufflevector <4 x float> %201, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %202, <4 x float> %143) #9, !dbg !7
  %206 = load <4 x half>, ptr addrspace(3) %address.local276, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %203, <4 x float> %205) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %208 = shufflevector <4 x float> %207, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %209 = load <4 x half>, ptr addrspace(3) %address.local282, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %206, <4 x float> %144) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %209, <4 x float> %210) #9, !dbg !7
  %212 = load <4 x half>, ptr addrspace(3) %address.local289, align 2, !dbg !7
  %213 = load <4 x half>, ptr addrspace(3) %address.local295, align 2, !dbg !7
  %214 = shufflevector <4 x float> %211, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %212, <4 x float> %145) #9, !dbg !7
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %213, <4 x float> %215) #9, !dbg !7
  %217 = shufflevector <4 x float> %216, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %218 = shufflevector <4 x float> %130, <4 x float> %150, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %219 = shufflevector <32 x float> %218, <32 x float> %156, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %220 = shufflevector <32 x float> %219, <32 x float> %160, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %221 = shufflevector <32 x float> %220, <32 x float> %166, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %222 = shufflevector <32 x float> %221, <32 x float> %170, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %223 = shufflevector <32 x float> %222, <32 x float> %176, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %224 = shufflevector <32 x float> %223, <32 x float> %180, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !21
  %225 = insertvalue [2 x <32 x float>] poison, <32 x float> %224, 0, !dbg !21
  %226 = shufflevector <4 x float> %183, <4 x float> %188, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %227 = shufflevector <32 x float> %226, <32 x float> %194, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %228 = shufflevector <32 x float> %227, <32 x float> %198, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %229 = shufflevector <32 x float> %228, <32 x float> %204, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %230 = shufflevector <32 x float> %229, <32 x float> %208, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %231 = shufflevector <32 x float> %230, <32 x float> %214, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %232 = shufflevector <32 x float> %231, <32 x float> %217, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !21
  %233 = insertvalue [2 x <32 x float>] %225, <32 x float> %232, 1, !dbg !21
  fence syncscope("workgroup") release, !dbg !21
  call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %234 = add nuw nsw i64 %39, 1, !dbg !21
  br label %38, !dbg !21

235:                                              ; preds = %38
  %236 = add nsw i64 %35, %6, !dbg !17
  %237 = shl i64 %236, 19, !dbg !17
  %238 = and i64 %237, 16252928, !dbg !17
  %239 = shl nsw i64 %37, 7, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %240 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %238, !dbg !17
  %241 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %239, !dbg !17
  %address.thread296 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !17, !range !39, !frisk.register_lifetimes !1
  %address.local297 = zext nneg i32 %address.thread296 to i64, !dbg !13
  %address.local298 = shl nuw nsw i64 %address.local297, 3, !dbg !12
  %address.local299 = and i64 %address.local298, 1536, !dbg !12
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local299, !dbg !12
  %address.local301 = ptrtoint ptr addrspace(3) %address.local300 to i64, !dbg !12
  %address.local302 = trunc i64 %address.local301 to i32, !dbg !12
  %242 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local302), !dbg !17
  %243 = zext i32 %242 to i64, !dbg !17
  %244 = inttoptr i64 %243 to ptr addrspace(3), !dbg !17
  %address.local303 = shl nuw nsw i64 %address.local297, 5, !dbg !17
  %address.local304 = and i64 %address.local303, 8064, !dbg !17
  %address.local305 = shl nuw nsw i64 %address.local297, 3, !dbg !17
  %address.local306 = and i64 %address.local305, 24, !dbg !17
  %address.local307 = or disjoint i64 %address.local304, %address.local306, !dbg !17
  %address.local308 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %address.local307, !dbg !17
  %245 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %246 = ptrtoint ptr addrspace(1) %address.local308 to i64, !dbg !17
  %247 = sub i64 %246, %245, !dbg !17
  %dma.buffer.offset900 = trunc i64 %247 to i32, !dbg !17
  %248 = trunc i64 %245 to i32, !dbg !17
  %249 = insertelement <4 x i32> poison, i32 %248, i64 0, !dbg !17
  %250 = lshr i64 %245, 32, !dbg !17
  %251 = trunc i64 %250 to i32, !dbg !17
  %252 = insertelement <4 x i32> %249, i32 %251, i64 1, !dbg !17
  %253 = insertelement <4 x i32> %252, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc901 = insertelement <4 x i32> %253, i32 131072, i64 3, !dbg !17
  %254 = ptrtoint ptr addrspace(3) %244 to i32, !dbg !17
  %dma.buffer.lds902 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %254), !dbg !17
  %255 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds902, i32 %dma.buffer.offset900, <4 x i32> %dma.buffer.rsrc901) #10, !dbg !17
  %address.local309 = or disjoint i64 %address.local297, 256, !dbg !12
  %address.local310 = shl nuw nsw i64 %address.local309, 3, !dbg !12
  %address.local311 = and i64 %address.local310, 3584, !dbg !12
  %address.local312 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local311, !dbg !12
  %address.local313 = ptrtoint ptr addrspace(3) %address.local312 to i64, !dbg !12
  %address.local314 = trunc i64 %address.local313 to i32, !dbg !12
  %256 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local314), !dbg !17
  %257 = zext i32 %256 to i64, !dbg !17
  %258 = inttoptr i64 %257 to ptr addrspace(3), !dbg !17
  %address.local315 = shl nuw nsw i64 %address.local309, 5, !dbg !17
  %address.local316 = and i64 %address.local315, 16256, !dbg !17
  %address.local317 = shl nuw nsw i64 %address.local297, 3, !dbg !17
  %address.local318 = and i64 %address.local317, 24, !dbg !17
  %address.local319 = or disjoint i64 %address.local316, %address.local318, !dbg !17
  %address.local320 = getelementptr [2 x i8], ptr addrspace(1) %241, i64 %address.local319, !dbg !17
  %259 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %260 = ptrtoint ptr addrspace(1) %address.local320 to i64, !dbg !17
  %261 = sub i64 %260, %259, !dbg !17
  %dma.buffer.offset897 = trunc i64 %261 to i32, !dbg !17
  %262 = trunc i64 %259 to i32, !dbg !17
  %263 = insertelement <4 x i32> poison, i32 %262, i64 0, !dbg !17
  %264 = lshr i64 %259, 32, !dbg !17
  %265 = trunc i64 %264 to i32, !dbg !17
  %266 = insertelement <4 x i32> %263, i32 %265, i64 1, !dbg !17
  %267 = insertelement <4 x i32> %266, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc898 = insertelement <4 x i32> %267, i32 131072, i64 3, !dbg !17
  %268 = ptrtoint ptr addrspace(3) %258 to i32, !dbg !17
  %dma.buffer.lds899 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %268), !dbg !17
  %269 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds899, i32 %dma.buffer.offset897, <4 x i32> %dma.buffer.rsrc898) #10, !dbg !17
  %270 = or disjoint i64 %34, 127, !dbg !14
  %271 = icmp uge i64 %9, %270, !dbg !14
  %272 = and i1 %24, %271, !dbg !14
  br i1 %272, label %273, label %274, !dbg !14

273:                                              ; preds = %235
  br label %418, !dbg !14

274:                                              ; preds = %235
  %275 = add nsw i64 %9, %20, !dbg !14
  %276 = shl nuw nsw i64 %21, 4, !dbg !14
  %277 = add nsw i64 %275, %276, !dbg !14
  %278 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %278 to i64, !dbg !14
  %279 = sub nsw i64 %277, %index.compact, !dbg !14
  %280 = and i64 %20, 48, !dbg !14
  %281 = or disjoint i64 %34, %280, !dbg !14
  %282 = add nsw i64 %279, 1, !dbg !22
  %283 = lshr i64 %282, 32, !dbg !23
  %compare.high = trunc i64 %283 to i32, !dbg !23
  %284 = lshr i64 %281, 32, !dbg !23
  %compare.high614 = trunc i64 %284 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high614, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high614, !dbg !23
  %compare.low.base = trunc i64 %282 to i32, !dbg !23
  %compare.low.base615 = trunc i64 %281 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base615, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %285 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low616 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result617 = select i1 %compare.high.equal, i1 %compare.low616, i1 %compare.high.order, !dbg !23
  %286 = select i1 %compare.result617, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base618 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local619 = or i32 %compare.local.base618, 2, !dbg !23
  %compare.low620 = icmp ugt i32 %compare.low.base, %compare.local619, !dbg !23
  %compare.result621 = select i1 %compare.high.equal, i1 %compare.low620, i1 %compare.high.order, !dbg !23
  %287 = select i1 %compare.result621, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base622 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local623 = or i32 %compare.local.base622, 3, !dbg !23
  %compare.low624 = icmp ugt i32 %compare.low.base, %compare.local623, !dbg !23
  %compare.result625 = select i1 %compare.high.equal, i1 %compare.low624, i1 %compare.high.order, !dbg !23
  %288 = select i1 %compare.result625, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base626 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local627 = or i32 %compare.local.base626, 4, !dbg !23
  %compare.low628 = icmp ugt i32 %compare.low.base, %compare.local627, !dbg !23
  %compare.result629 = select i1 %compare.high.equal, i1 %compare.low628, i1 %compare.high.order, !dbg !23
  %289 = select i1 %compare.result629, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base630 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local631 = or i32 %compare.local.base630, 5, !dbg !23
  %compare.low632 = icmp ugt i32 %compare.low.base, %compare.local631, !dbg !23
  %compare.result633 = select i1 %compare.high.equal, i1 %compare.low632, i1 %compare.high.order, !dbg !23
  %290 = select i1 %compare.result633, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base634 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local635 = or i32 %compare.local.base634, 6, !dbg !23
  %compare.low636 = icmp ugt i32 %compare.low.base, %compare.local635, !dbg !23
  %compare.result637 = select i1 %compare.high.equal, i1 %compare.low636, i1 %compare.high.order, !dbg !23
  %291 = select i1 %compare.result637, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base638 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local639 = or i32 %compare.local.base638, 7, !dbg !23
  %compare.low640 = icmp ugt i32 %compare.low.base, %compare.local639, !dbg !23
  %compare.result641 = select i1 %compare.high.equal, i1 %compare.low640, i1 %compare.high.order, !dbg !23
  %292 = select i1 %compare.result641, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base642 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local643 = or i32 %compare.local.base642, 8, !dbg !23
  %compare.low644 = icmp ugt i32 %compare.low.base, %compare.local643, !dbg !23
  %compare.result645 = select i1 %compare.high.equal, i1 %compare.low644, i1 %compare.high.order, !dbg !23
  %293 = select i1 %compare.result645, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base646 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local647 = or i32 %compare.local.base646, 9, !dbg !23
  %compare.low648 = icmp ugt i32 %compare.low.base, %compare.local647, !dbg !23
  %compare.result649 = select i1 %compare.high.equal, i1 %compare.low648, i1 %compare.high.order, !dbg !23
  %294 = select i1 %compare.result649, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base650 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local651 = or i32 %compare.local.base650, 10, !dbg !23
  %compare.low652 = icmp ugt i32 %compare.low.base, %compare.local651, !dbg !23
  %compare.result653 = select i1 %compare.high.equal, i1 %compare.low652, i1 %compare.high.order, !dbg !23
  %295 = select i1 %compare.result653, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base654 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local655 = or i32 %compare.local.base654, 11, !dbg !23
  %compare.low656 = icmp ugt i32 %compare.low.base, %compare.local655, !dbg !23
  %compare.result657 = select i1 %compare.high.equal, i1 %compare.low656, i1 %compare.high.order, !dbg !23
  %296 = select i1 %compare.result657, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base658 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local659 = or i32 %compare.local.base658, 12, !dbg !23
  %compare.low660 = icmp ugt i32 %compare.low.base, %compare.local659, !dbg !23
  %compare.result661 = select i1 %compare.high.equal, i1 %compare.low660, i1 %compare.high.order, !dbg !23
  %297 = select i1 %compare.result661, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base662 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local663 = or i32 %compare.local.base662, 13, !dbg !23
  %compare.low664 = icmp ugt i32 %compare.low.base, %compare.local663, !dbg !23
  %compare.result665 = select i1 %compare.high.equal, i1 %compare.low664, i1 %compare.high.order, !dbg !23
  %298 = select i1 %compare.result665, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base666 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local667 = or i32 %compare.local.base666, 14, !dbg !23
  %compare.low668 = icmp ugt i32 %compare.low.base, %compare.local667, !dbg !23
  %compare.result669 = select i1 %compare.high.equal, i1 %compare.low668, i1 %compare.high.order, !dbg !23
  %299 = select i1 %compare.result669, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base670 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local671 = or i32 %compare.local.base670, 15, !dbg !23
  %compare.low672 = icmp ugt i32 %compare.low.base, %compare.local671, !dbg !23
  %compare.result673 = select i1 %compare.high.equal, i1 %compare.low672, i1 %compare.high.order, !dbg !23
  %300 = select i1 %compare.result673, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base674 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local675 = or i32 %compare.local.base674, 64, !dbg !23
  %compare.low676 = icmp ugt i32 %compare.low.base, %compare.local675, !dbg !23
  %compare.result677 = select i1 %compare.high.equal, i1 %compare.low676, i1 %compare.high.order, !dbg !23
  %301 = select i1 %compare.result677, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base678 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local679 = or i32 %compare.local.base678, 65, !dbg !23
  %compare.low680 = icmp ugt i32 %compare.low.base, %compare.local679, !dbg !23
  %compare.result681 = select i1 %compare.high.equal, i1 %compare.low680, i1 %compare.high.order, !dbg !23
  %302 = select i1 %compare.result681, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base682 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local683 = or i32 %compare.local.base682, 66, !dbg !23
  %compare.low684 = icmp ugt i32 %compare.low.base, %compare.local683, !dbg !23
  %compare.result685 = select i1 %compare.high.equal, i1 %compare.low684, i1 %compare.high.order, !dbg !23
  %303 = select i1 %compare.result685, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base686 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local687 = or i32 %compare.local.base686, 67, !dbg !23
  %compare.low688 = icmp ugt i32 %compare.low.base, %compare.local687, !dbg !23
  %compare.result689 = select i1 %compare.high.equal, i1 %compare.low688, i1 %compare.high.order, !dbg !23
  %304 = select i1 %compare.result689, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base690 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local691 = or i32 %compare.local.base690, 68, !dbg !23
  %compare.low692 = icmp ugt i32 %compare.low.base, %compare.local691, !dbg !23
  %compare.result693 = select i1 %compare.high.equal, i1 %compare.low692, i1 %compare.high.order, !dbg !23
  %305 = select i1 %compare.result693, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base694 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local695 = or i32 %compare.local.base694, 69, !dbg !23
  %compare.low696 = icmp ugt i32 %compare.low.base, %compare.local695, !dbg !23
  %compare.result697 = select i1 %compare.high.equal, i1 %compare.low696, i1 %compare.high.order, !dbg !23
  %306 = select i1 %compare.result697, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base698 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local699 = or i32 %compare.local.base698, 70, !dbg !23
  %compare.low700 = icmp ugt i32 %compare.low.base, %compare.local699, !dbg !23
  %compare.result701 = select i1 %compare.high.equal, i1 %compare.low700, i1 %compare.high.order, !dbg !23
  %307 = select i1 %compare.result701, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base702 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local703 = or i32 %compare.local.base702, 71, !dbg !23
  %compare.low704 = icmp ugt i32 %compare.low.base, %compare.local703, !dbg !23
  %compare.result705 = select i1 %compare.high.equal, i1 %compare.low704, i1 %compare.high.order, !dbg !23
  %308 = select i1 %compare.result705, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base706 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local707 = or i32 %compare.local.base706, 72, !dbg !23
  %compare.low708 = icmp ugt i32 %compare.low.base, %compare.local707, !dbg !23
  %compare.result709 = select i1 %compare.high.equal, i1 %compare.low708, i1 %compare.high.order, !dbg !23
  %309 = select i1 %compare.result709, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base710 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local711 = or i32 %compare.local.base710, 73, !dbg !23
  %compare.low712 = icmp ugt i32 %compare.low.base, %compare.local711, !dbg !23
  %compare.result713 = select i1 %compare.high.equal, i1 %compare.low712, i1 %compare.high.order, !dbg !23
  %310 = select i1 %compare.result713, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base714 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local715 = or i32 %compare.local.base714, 74, !dbg !23
  %compare.low716 = icmp ugt i32 %compare.low.base, %compare.local715, !dbg !23
  %compare.result717 = select i1 %compare.high.equal, i1 %compare.low716, i1 %compare.high.order, !dbg !23
  %311 = select i1 %compare.result717, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base718 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local719 = or i32 %compare.local.base718, 75, !dbg !23
  %compare.low720 = icmp ugt i32 %compare.low.base, %compare.local719, !dbg !23
  %compare.result721 = select i1 %compare.high.equal, i1 %compare.low720, i1 %compare.high.order, !dbg !23
  %312 = select i1 %compare.result721, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base722 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local723 = or i32 %compare.local.base722, 76, !dbg !23
  %compare.low724 = icmp ugt i32 %compare.low.base, %compare.local723, !dbg !23
  %compare.result725 = select i1 %compare.high.equal, i1 %compare.low724, i1 %compare.high.order, !dbg !23
  %313 = select i1 %compare.result725, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base726 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local727 = or i32 %compare.local.base726, 77, !dbg !23
  %compare.low728 = icmp ugt i32 %compare.low.base, %compare.local727, !dbg !23
  %compare.result729 = select i1 %compare.high.equal, i1 %compare.low728, i1 %compare.high.order, !dbg !23
  %314 = select i1 %compare.result729, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base730 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local731 = or i32 %compare.local.base730, 78, !dbg !23
  %compare.low732 = icmp ugt i32 %compare.low.base, %compare.local731, !dbg !23
  %compare.result733 = select i1 %compare.high.equal, i1 %compare.low732, i1 %compare.high.order, !dbg !23
  %315 = select i1 %compare.result733, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base734 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base615, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local735 = or i32 %compare.local.base734, 79, !dbg !23
  %compare.low736 = icmp ugt i32 %compare.low.base, %compare.local735, !dbg !23
  %compare.result737 = select i1 %compare.high.equal, i1 %compare.low736, i1 %compare.high.order, !dbg !23
  %316 = select i1 %compare.result737, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %317 = add nsw i64 %279, 65, !dbg !22
  %318 = lshr i64 %317, 32, !dbg !23
  %compare.high738 = trunc i64 %318 to i32, !dbg !23
  %319 = lshr i64 %281, 32, !dbg !23
  %compare.high739 = trunc i64 %319 to i32, !dbg !23
  %compare.high.equal740 = icmp eq i32 %compare.high738, %compare.high739, !dbg !23
  %compare.high.order741 = icmp ugt i32 %compare.high738, %compare.high739, !dbg !23
  %compare.low.base742 = trunc i64 %317 to i32, !dbg !23
  %compare.low.base743 = trunc i64 %281 to i32, !dbg !23
  %compare.low744 = icmp ugt i32 %compare.low.base742, %compare.low.base743, !dbg !23
  %compare.result745 = select i1 %compare.high.equal740, i1 %compare.low744, i1 %compare.high.order741, !dbg !23
  %320 = select i1 %compare.result745, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base746 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 32) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local747 = or i32 %compare.local.base746, 1, !dbg !23
  %compare.low748 = icmp ugt i32 %compare.low.base742, %compare.local747, !dbg !23
  %compare.result749 = select i1 %compare.high.equal740, i1 %compare.low748, i1 %compare.high.order741, !dbg !23
  %321 = select i1 %compare.result749, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base750 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 33) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local751 = or i32 %compare.local.base750, 2, !dbg !23
  %compare.low752 = icmp ugt i32 %compare.low.base742, %compare.local751, !dbg !23
  %compare.result753 = select i1 %compare.high.equal740, i1 %compare.low752, i1 %compare.high.order741, !dbg !23
  %322 = select i1 %compare.result753, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base754 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 34) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local755 = or i32 %compare.local.base754, 3, !dbg !23
  %compare.low756 = icmp ugt i32 %compare.low.base742, %compare.local755, !dbg !23
  %compare.result757 = select i1 %compare.high.equal740, i1 %compare.low756, i1 %compare.high.order741, !dbg !23
  %323 = select i1 %compare.result757, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base758 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 35) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local759 = or i32 %compare.local.base758, 4, !dbg !23
  %compare.low760 = icmp ugt i32 %compare.low.base742, %compare.local759, !dbg !23
  %compare.result761 = select i1 %compare.high.equal740, i1 %compare.low760, i1 %compare.high.order741, !dbg !23
  %324 = select i1 %compare.result761, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base762 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 36) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local763 = or i32 %compare.local.base762, 5, !dbg !23
  %compare.low764 = icmp ugt i32 %compare.low.base742, %compare.local763, !dbg !23
  %compare.result765 = select i1 %compare.high.equal740, i1 %compare.low764, i1 %compare.high.order741, !dbg !23
  %325 = select i1 %compare.result765, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base766 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 37) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local767 = or i32 %compare.local.base766, 6, !dbg !23
  %compare.low768 = icmp ugt i32 %compare.low.base742, %compare.local767, !dbg !23
  %compare.result769 = select i1 %compare.high.equal740, i1 %compare.low768, i1 %compare.high.order741, !dbg !23
  %326 = select i1 %compare.result769, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base770 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 38) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local771 = or i32 %compare.local.base770, 7, !dbg !23
  %compare.low772 = icmp ugt i32 %compare.low.base742, %compare.local771, !dbg !23
  %compare.result773 = select i1 %compare.high.equal740, i1 %compare.low772, i1 %compare.high.order741, !dbg !23
  %327 = select i1 %compare.result773, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base774 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 39) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local775 = or i32 %compare.local.base774, 8, !dbg !23
  %compare.low776 = icmp ugt i32 %compare.low.base742, %compare.local775, !dbg !23
  %compare.result777 = select i1 %compare.high.equal740, i1 %compare.low776, i1 %compare.high.order741, !dbg !23
  %328 = select i1 %compare.result777, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base778 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 40) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local779 = or i32 %compare.local.base778, 9, !dbg !23
  %compare.low780 = icmp ugt i32 %compare.low.base742, %compare.local779, !dbg !23
  %compare.result781 = select i1 %compare.high.equal740, i1 %compare.low780, i1 %compare.high.order741, !dbg !23
  %329 = select i1 %compare.result781, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base782 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 41) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local783 = or i32 %compare.local.base782, 10, !dbg !23
  %compare.low784 = icmp ugt i32 %compare.low.base742, %compare.local783, !dbg !23
  %compare.result785 = select i1 %compare.high.equal740, i1 %compare.low784, i1 %compare.high.order741, !dbg !23
  %330 = select i1 %compare.result785, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base786 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 42) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local787 = or i32 %compare.local.base786, 11, !dbg !23
  %compare.low788 = icmp ugt i32 %compare.low.base742, %compare.local787, !dbg !23
  %compare.result789 = select i1 %compare.high.equal740, i1 %compare.low788, i1 %compare.high.order741, !dbg !23
  %331 = select i1 %compare.result789, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base790 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 43) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local791 = or i32 %compare.local.base790, 12, !dbg !23
  %compare.low792 = icmp ugt i32 %compare.low.base742, %compare.local791, !dbg !23
  %compare.result793 = select i1 %compare.high.equal740, i1 %compare.low792, i1 %compare.high.order741, !dbg !23
  %332 = select i1 %compare.result793, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base794 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 44) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local795 = or i32 %compare.local.base794, 13, !dbg !23
  %compare.low796 = icmp ugt i32 %compare.low.base742, %compare.local795, !dbg !23
  %compare.result797 = select i1 %compare.high.equal740, i1 %compare.low796, i1 %compare.high.order741, !dbg !23
  %333 = select i1 %compare.result797, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base798 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 45) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local799 = or i32 %compare.local.base798, 14, !dbg !23
  %compare.low800 = icmp ugt i32 %compare.low.base742, %compare.local799, !dbg !23
  %compare.result801 = select i1 %compare.high.equal740, i1 %compare.low800, i1 %compare.high.order741, !dbg !23
  %334 = select i1 %compare.result801, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base802 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 46) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local803 = or i32 %compare.local.base802, 15, !dbg !23
  %compare.low804 = icmp ugt i32 %compare.low.base742, %compare.local803, !dbg !23
  %compare.result805 = select i1 %compare.high.equal740, i1 %compare.low804, i1 %compare.high.order741, !dbg !23
  %335 = select i1 %compare.result805, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base806 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 47) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local807 = or i32 %compare.local.base806, 64, !dbg !23
  %compare.low808 = icmp ugt i32 %compare.low.base742, %compare.local807, !dbg !23
  %compare.result809 = select i1 %compare.high.equal740, i1 %compare.low808, i1 %compare.high.order741, !dbg !23
  %336 = select i1 %compare.result809, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base810 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 48) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local811 = or i32 %compare.local.base810, 65, !dbg !23
  %compare.low812 = icmp ugt i32 %compare.low.base742, %compare.local811, !dbg !23
  %compare.result813 = select i1 %compare.high.equal740, i1 %compare.low812, i1 %compare.high.order741, !dbg !23
  %337 = select i1 %compare.result813, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base814 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 49) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local815 = or i32 %compare.local.base814, 66, !dbg !23
  %compare.low816 = icmp ugt i32 %compare.low.base742, %compare.local815, !dbg !23
  %compare.result817 = select i1 %compare.high.equal740, i1 %compare.low816, i1 %compare.high.order741, !dbg !23
  %338 = select i1 %compare.result817, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base818 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 50) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local819 = or i32 %compare.local.base818, 67, !dbg !23
  %compare.low820 = icmp ugt i32 %compare.low.base742, %compare.local819, !dbg !23
  %compare.result821 = select i1 %compare.high.equal740, i1 %compare.low820, i1 %compare.high.order741, !dbg !23
  %339 = select i1 %compare.result821, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base822 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 51) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local823 = or i32 %compare.local.base822, 68, !dbg !23
  %compare.low824 = icmp ugt i32 %compare.low.base742, %compare.local823, !dbg !23
  %compare.result825 = select i1 %compare.high.equal740, i1 %compare.low824, i1 %compare.high.order741, !dbg !23
  %340 = select i1 %compare.result825, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base826 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 52) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local827 = or i32 %compare.local.base826, 69, !dbg !23
  %compare.low828 = icmp ugt i32 %compare.low.base742, %compare.local827, !dbg !23
  %compare.result829 = select i1 %compare.high.equal740, i1 %compare.low828, i1 %compare.high.order741, !dbg !23
  %341 = select i1 %compare.result829, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base830 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 53) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local831 = or i32 %compare.local.base830, 70, !dbg !23
  %compare.low832 = icmp ugt i32 %compare.low.base742, %compare.local831, !dbg !23
  %compare.result833 = select i1 %compare.high.equal740, i1 %compare.low832, i1 %compare.high.order741, !dbg !23
  %342 = select i1 %compare.result833, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base834 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 54) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local835 = or i32 %compare.local.base834, 71, !dbg !23
  %compare.low836 = icmp ugt i32 %compare.low.base742, %compare.local835, !dbg !23
  %compare.result837 = select i1 %compare.high.equal740, i1 %compare.low836, i1 %compare.high.order741, !dbg !23
  %343 = select i1 %compare.result837, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base838 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 55) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local839 = or i32 %compare.local.base838, 72, !dbg !23
  %compare.low840 = icmp ugt i32 %compare.low.base742, %compare.local839, !dbg !23
  %compare.result841 = select i1 %compare.high.equal740, i1 %compare.low840, i1 %compare.high.order741, !dbg !23
  %344 = select i1 %compare.result841, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base842 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 56) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local843 = or i32 %compare.local.base842, 73, !dbg !23
  %compare.low844 = icmp ugt i32 %compare.low.base742, %compare.local843, !dbg !23
  %compare.result845 = select i1 %compare.high.equal740, i1 %compare.low844, i1 %compare.high.order741, !dbg !23
  %345 = select i1 %compare.result845, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base846 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 57) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local847 = or i32 %compare.local.base846, 74, !dbg !23
  %compare.low848 = icmp ugt i32 %compare.low.base742, %compare.local847, !dbg !23
  %compare.result849 = select i1 %compare.high.equal740, i1 %compare.low848, i1 %compare.high.order741, !dbg !23
  %346 = select i1 %compare.result849, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base850 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 58) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local851 = or i32 %compare.local.base850, 75, !dbg !23
  %compare.low852 = icmp ugt i32 %compare.low.base742, %compare.local851, !dbg !23
  %compare.result853 = select i1 %compare.high.equal740, i1 %compare.low852, i1 %compare.high.order741, !dbg !23
  %347 = select i1 %compare.result853, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base854 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 59) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local855 = or i32 %compare.local.base854, 76, !dbg !23
  %compare.low856 = icmp ugt i32 %compare.low.base742, %compare.local855, !dbg !23
  %compare.result857 = select i1 %compare.high.equal740, i1 %compare.low856, i1 %compare.high.order741, !dbg !23
  %348 = select i1 %compare.result857, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base858 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 60) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local859 = or i32 %compare.local.base858, 77, !dbg !23
  %compare.low860 = icmp ugt i32 %compare.low.base742, %compare.local859, !dbg !23
  %compare.result861 = select i1 %compare.high.equal740, i1 %compare.low860, i1 %compare.high.order741, !dbg !23
  %349 = select i1 %compare.result861, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base862 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 61) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local863 = or i32 %compare.local.base862, 78, !dbg !23
  %compare.low864 = icmp ugt i32 %compare.low.base742, %compare.local863, !dbg !23
  %compare.result865 = select i1 %compare.high.equal740, i1 %compare.low864, i1 %compare.high.order741, !dbg !23
  %350 = select i1 %compare.result865, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base866 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base743, i32 62) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local867 = or i32 %compare.local.base866, 79, !dbg !23
  %compare.low868 = icmp ugt i32 %compare.low.base742, %compare.local867, !dbg !23
  %compare.result869 = select i1 %compare.high.equal740, i1 %compare.low868, i1 %compare.high.order741, !dbg !23
  %351 = select i1 %compare.result869, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %352 = insertelement <32 x float> poison, float %285, i64 0, !dbg !14
  %353 = insertelement <32 x float> %352, float %286, i64 1, !dbg !14
  %354 = insertelement <32 x float> %353, float %287, i64 2, !dbg !14
  %355 = insertelement <32 x float> %354, float %288, i64 3, !dbg !14
  %356 = insertelement <32 x float> %355, float %289, i64 4, !dbg !14
  %357 = insertelement <32 x float> %356, float %290, i64 5, !dbg !14
  %358 = insertelement <32 x float> %357, float %291, i64 6, !dbg !14
  %359 = insertelement <32 x float> %358, float %292, i64 7, !dbg !14
  %360 = insertelement <32 x float> %359, float %293, i64 8, !dbg !14
  %361 = insertelement <32 x float> %360, float %294, i64 9, !dbg !14
  %362 = insertelement <32 x float> %361, float %295, i64 10, !dbg !14
  %363 = insertelement <32 x float> %362, float %296, i64 11, !dbg !14
  %364 = insertelement <32 x float> %363, float %297, i64 12, !dbg !14
  %365 = insertelement <32 x float> %364, float %298, i64 13, !dbg !14
  %366 = insertelement <32 x float> %365, float %299, i64 14, !dbg !14
  %367 = insertelement <32 x float> %366, float %300, i64 15, !dbg !14
  %368 = insertelement <32 x float> %367, float %301, i64 16, !dbg !14
  %369 = insertelement <32 x float> %368, float %302, i64 17, !dbg !14
  %370 = insertelement <32 x float> %369, float %303, i64 18, !dbg !14
  %371 = insertelement <32 x float> %370, float %304, i64 19, !dbg !14
  %372 = insertelement <32 x float> %371, float %305, i64 20, !dbg !14
  %373 = insertelement <32 x float> %372, float %306, i64 21, !dbg !14
  %374 = insertelement <32 x float> %373, float %307, i64 22, !dbg !14
  %375 = insertelement <32 x float> %374, float %308, i64 23, !dbg !14
  %376 = insertelement <32 x float> %375, float %309, i64 24, !dbg !14
  %377 = insertelement <32 x float> %376, float %310, i64 25, !dbg !14
  %378 = insertelement <32 x float> %377, float %311, i64 26, !dbg !14
  %379 = insertelement <32 x float> %378, float %312, i64 27, !dbg !14
  %380 = insertelement <32 x float> %379, float %313, i64 28, !dbg !14
  %381 = insertelement <32 x float> %380, float %314, i64 29, !dbg !14
  %382 = insertelement <32 x float> %381, float %315, i64 30, !dbg !14
  %383 = insertelement <32 x float> %382, float %316, i64 31, !dbg !14
  %384 = insertvalue [2 x <32 x float>] poison, <32 x float> %383, 0, !dbg !14
  %385 = insertelement <32 x float> poison, float %320, i64 0, !dbg !14
  %386 = insertelement <32 x float> %385, float %321, i64 1, !dbg !14
  %387 = insertelement <32 x float> %386, float %322, i64 2, !dbg !14
  %388 = insertelement <32 x float> %387, float %323, i64 3, !dbg !14
  %389 = insertelement <32 x float> %388, float %324, i64 4, !dbg !14
  %390 = insertelement <32 x float> %389, float %325, i64 5, !dbg !14
  %391 = insertelement <32 x float> %390, float %326, i64 6, !dbg !14
  %392 = insertelement <32 x float> %391, float %327, i64 7, !dbg !14
  %393 = insertelement <32 x float> %392, float %328, i64 8, !dbg !14
  %394 = insertelement <32 x float> %393, float %329, i64 9, !dbg !14
  %395 = insertelement <32 x float> %394, float %330, i64 10, !dbg !14
  %396 = insertelement <32 x float> %395, float %331, i64 11, !dbg !14
  %397 = insertelement <32 x float> %396, float %332, i64 12, !dbg !14
  %398 = insertelement <32 x float> %397, float %333, i64 13, !dbg !14
  %399 = insertelement <32 x float> %398, float %334, i64 14, !dbg !14
  %400 = insertelement <32 x float> %399, float %335, i64 15, !dbg !14
  %401 = insertelement <32 x float> %400, float %336, i64 16, !dbg !14
  %402 = insertelement <32 x float> %401, float %337, i64 17, !dbg !14
  %403 = insertelement <32 x float> %402, float %338, i64 18, !dbg !14
  %404 = insertelement <32 x float> %403, float %339, i64 19, !dbg !14
  %405 = insertelement <32 x float> %404, float %340, i64 20, !dbg !14
  %406 = insertelement <32 x float> %405, float %341, i64 21, !dbg !14
  %407 = insertelement <32 x float> %406, float %342, i64 22, !dbg !14
  %408 = insertelement <32 x float> %407, float %343, i64 23, !dbg !14
  %409 = insertelement <32 x float> %408, float %344, i64 24, !dbg !14
  %410 = insertelement <32 x float> %409, float %345, i64 25, !dbg !14
  %411 = insertelement <32 x float> %410, float %346, i64 26, !dbg !14
  %412 = insertelement <32 x float> %411, float %347, i64 27, !dbg !14
  %413 = insertelement <32 x float> %412, float %348, i64 28, !dbg !14
  %414 = insertelement <32 x float> %413, float %349, i64 29, !dbg !14
  %415 = insertelement <32 x float> %414, float %350, i64 30, !dbg !14
  %416 = insertelement <32 x float> %415, float %351, i64 31, !dbg !14
  %417 = insertvalue [2 x <32 x float>] %384, <32 x float> %416, 1, !dbg !14
  br label %418, !dbg !14

418:                                              ; preds = %273, %274
  %419 = phi [2 x <32 x float>] [ %417, %274 ], [ zeroinitializer, %273 ], !dbg !14
  br label %420, !dbg !14

420:                                              ; preds = %418
  %421 = extractvalue [2 x <32 x float>] %40, 0, !dbg !25
  %422 = extractvalue [2 x <32 x float>] %419, 0, !dbg !25
  %423 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %424 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %425 = fadd <4 x float> %423, %424, !dbg !25
  %426 = extractelement <4 x float> %425, i64 0, !dbg !25
  %427 = extractelement <4 x float> %425, i64 1, !dbg !25
  %428 = extractelement <4 x float> %425, i64 2, !dbg !25
  %429 = extractelement <4 x float> %425, i64 3, !dbg !25
  %430 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %431 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %432 = fadd <4 x float> %430, %431, !dbg !25
  %433 = extractelement <4 x float> %432, i64 0, !dbg !25
  %434 = extractelement <4 x float> %432, i64 1, !dbg !25
  %435 = extractelement <4 x float> %432, i64 2, !dbg !25
  %436 = extractelement <4 x float> %432, i64 3, !dbg !25
  %437 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %438 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %439 = fadd <4 x float> %437, %438, !dbg !25
  %440 = extractelement <4 x float> %439, i64 0, !dbg !25
  %441 = extractelement <4 x float> %439, i64 1, !dbg !25
  %442 = extractelement <4 x float> %439, i64 2, !dbg !25
  %443 = extractelement <4 x float> %439, i64 3, !dbg !25
  %444 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %445 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %446 = fadd <4 x float> %444, %445, !dbg !25
  %447 = extractelement <4 x float> %446, i64 0, !dbg !25
  %448 = extractelement <4 x float> %446, i64 1, !dbg !25
  %449 = extractelement <4 x float> %446, i64 2, !dbg !25
  %450 = extractelement <4 x float> %446, i64 3, !dbg !25
  %451 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %452 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %453 = fadd <4 x float> %451, %452, !dbg !25
  %454 = extractelement <4 x float> %453, i64 0, !dbg !25
  %455 = extractelement <4 x float> %453, i64 1, !dbg !25
  %456 = extractelement <4 x float> %453, i64 2, !dbg !25
  %457 = extractelement <4 x float> %453, i64 3, !dbg !25
  %458 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %459 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %460 = fadd <4 x float> %458, %459, !dbg !25
  %461 = extractelement <4 x float> %460, i64 0, !dbg !25
  %462 = extractelement <4 x float> %460, i64 1, !dbg !25
  %463 = extractelement <4 x float> %460, i64 2, !dbg !25
  %464 = extractelement <4 x float> %460, i64 3, !dbg !25
  %465 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %466 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %467 = fadd <4 x float> %465, %466, !dbg !25
  %468 = extractelement <4 x float> %467, i64 0, !dbg !25
  %469 = extractelement <4 x float> %467, i64 1, !dbg !25
  %470 = extractelement <4 x float> %467, i64 2, !dbg !25
  %471 = extractelement <4 x float> %467, i64 3, !dbg !25
  %472 = shufflevector <32 x float> %421, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %473 = shufflevector <32 x float> %422, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %474 = fadd <4 x float> %472, %473, !dbg !25
  %475 = extractelement <4 x float> %474, i64 0, !dbg !25
  %476 = extractelement <4 x float> %474, i64 1, !dbg !25
  %477 = extractelement <4 x float> %474, i64 2, !dbg !25
  %478 = extractelement <4 x float> %474, i64 3, !dbg !25
  %479 = extractvalue [2 x <32 x float>] %40, 1, !dbg !25
  %480 = extractvalue [2 x <32 x float>] %419, 1, !dbg !25
  %481 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %482 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %483 = fadd <4 x float> %481, %482, !dbg !25
  %484 = extractelement <4 x float> %483, i64 0, !dbg !25
  %485 = extractelement <4 x float> %483, i64 1, !dbg !25
  %486 = extractelement <4 x float> %483, i64 2, !dbg !25
  %487 = extractelement <4 x float> %483, i64 3, !dbg !25
  %488 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %489 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %490 = fadd <4 x float> %488, %489, !dbg !25
  %491 = extractelement <4 x float> %490, i64 0, !dbg !25
  %492 = extractelement <4 x float> %490, i64 1, !dbg !25
  %493 = extractelement <4 x float> %490, i64 2, !dbg !25
  %494 = extractelement <4 x float> %490, i64 3, !dbg !25
  %495 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %496 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %497 = fadd <4 x float> %495, %496, !dbg !25
  %498 = extractelement <4 x float> %497, i64 0, !dbg !25
  %499 = extractelement <4 x float> %497, i64 1, !dbg !25
  %500 = extractelement <4 x float> %497, i64 2, !dbg !25
  %501 = extractelement <4 x float> %497, i64 3, !dbg !25
  %502 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %503 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %504 = fadd <4 x float> %502, %503, !dbg !25
  %505 = extractelement <4 x float> %504, i64 0, !dbg !25
  %506 = extractelement <4 x float> %504, i64 1, !dbg !25
  %507 = extractelement <4 x float> %504, i64 2, !dbg !25
  %508 = extractelement <4 x float> %504, i64 3, !dbg !25
  %509 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %510 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %511 = fadd <4 x float> %509, %510, !dbg !25
  %512 = extractelement <4 x float> %511, i64 0, !dbg !25
  %513 = extractelement <4 x float> %511, i64 1, !dbg !25
  %514 = extractelement <4 x float> %511, i64 2, !dbg !25
  %515 = extractelement <4 x float> %511, i64 3, !dbg !25
  %516 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %517 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %518 = fadd <4 x float> %516, %517, !dbg !25
  %519 = extractelement <4 x float> %518, i64 0, !dbg !25
  %520 = extractelement <4 x float> %518, i64 1, !dbg !25
  %521 = extractelement <4 x float> %518, i64 2, !dbg !25
  %522 = extractelement <4 x float> %518, i64 3, !dbg !25
  %523 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %524 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %525 = fadd <4 x float> %523, %524, !dbg !25
  %526 = extractelement <4 x float> %525, i64 0, !dbg !25
  %527 = extractelement <4 x float> %525, i64 1, !dbg !25
  %528 = extractelement <4 x float> %525, i64 2, !dbg !25
  %529 = extractelement <4 x float> %525, i64 3, !dbg !25
  %530 = shufflevector <32 x float> %479, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %531 = shufflevector <32 x float> %480, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %532 = fadd <4 x float> %530, %531, !dbg !25
  %533 = extractelement <4 x float> %532, i64 0, !dbg !25
  %534 = extractelement <4 x float> %532, i64 1, !dbg !25
  %535 = extractelement <4 x float> %532, i64 2, !dbg !25
  %536 = extractelement <4 x float> %532, i64 3, !dbg !25
  %537 = call float @llvm.maxnum.f32(float %426, float 0xFFF0000000000000), !dbg !26
  %538 = call float @llvm.maxnum.f32(float %537, float %427), !dbg !26
  %539 = call float @llvm.maxnum.f32(float %538, float %428), !dbg !26
  %540 = call float @llvm.maxnum.f32(float %539, float %429), !dbg !26
  %541 = call float @llvm.maxnum.f32(float %540, float %433), !dbg !26
  %542 = call float @llvm.maxnum.f32(float %541, float %434), !dbg !26
  %543 = call float @llvm.maxnum.f32(float %542, float %435), !dbg !26
  %544 = call float @llvm.maxnum.f32(float %543, float %436), !dbg !26
  %545 = call float @llvm.maxnum.f32(float %544, float %440), !dbg !26
  %546 = call float @llvm.maxnum.f32(float %545, float %441), !dbg !26
  %547 = call float @llvm.maxnum.f32(float %546, float %442), !dbg !26
  %548 = call float @llvm.maxnum.f32(float %547, float %443), !dbg !26
  %549 = call float @llvm.maxnum.f32(float %548, float %447), !dbg !26
  %550 = call float @llvm.maxnum.f32(float %549, float %448), !dbg !26
  %551 = call float @llvm.maxnum.f32(float %550, float %449), !dbg !26
  %552 = call float @llvm.maxnum.f32(float %551, float %450), !dbg !26
  %553 = call float @llvm.maxnum.f32(float %552, float %454), !dbg !26
  %554 = call float @llvm.maxnum.f32(float %553, float %455), !dbg !26
  %555 = call float @llvm.maxnum.f32(float %554, float %456), !dbg !26
  %556 = call float @llvm.maxnum.f32(float %555, float %457), !dbg !26
  %557 = call float @llvm.maxnum.f32(float %556, float %461), !dbg !26
  %558 = call float @llvm.maxnum.f32(float %557, float %462), !dbg !26
  %559 = call float @llvm.maxnum.f32(float %558, float %463), !dbg !26
  %560 = call float @llvm.maxnum.f32(float %559, float %464), !dbg !26
  %561 = call float @llvm.maxnum.f32(float %560, float %468), !dbg !26
  %562 = call float @llvm.maxnum.f32(float %561, float %469), !dbg !26
  %563 = call float @llvm.maxnum.f32(float %562, float %470), !dbg !26
  %564 = call float @llvm.maxnum.f32(float %563, float %471), !dbg !26
  %565 = call float @llvm.maxnum.f32(float %564, float %475), !dbg !26
  %566 = call float @llvm.maxnum.f32(float %565, float %476), !dbg !26
  %567 = call float @llvm.maxnum.f32(float %566, float %477), !dbg !26
  %568 = call float @llvm.maxnum.f32(float %567, float %478), !dbg !26
  %569 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %570 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %569), !dbg !26
  %571 = add nuw nsw i32 %570, 64, !dbg !26
  %572 = xor i32 %570, 16, !dbg !26
  %573 = and i32 %571, 192, !dbg !26
  %574 = icmp samesign ult i32 %572, %573, !dbg !26
  %575 = select i1 %574, i32 %572, i32 %570, !dbg !26
  %576 = shl nuw nsw i32 %575, 2, !dbg !26
  %577 = bitcast float %568 to i32, !dbg !26
  %578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %576, i32 %577), !dbg !26
  %579 = bitcast i32 %578 to float, !dbg !26
  %580 = call float @llvm.maxnum.f32(float %568, float %579), !dbg !26
  %581 = xor i32 %570, 32, !dbg !26
  %582 = icmp samesign ult i32 %581, %573, !dbg !26
  %583 = select i1 %582, i32 %581, i32 %570, !dbg !26
  %584 = shl nuw nsw i32 %583, 2, !dbg !26
  %585 = bitcast float %580 to i32, !dbg !26
  %586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %584, i32 %585), !dbg !26
  %587 = bitcast i32 %586 to float, !dbg !26
  %588 = call float @llvm.maxnum.f32(float %580, float %587), !dbg !26
  %589 = extractvalue [2 x <1 x float>] %29, 0, !dbg !26
  %590 = insertelement <1 x float> poison, float %588, i64 0, !dbg !26
  %591 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %590, <1 x float> %589), !dbg !26
  %592 = call float @llvm.maxnum.f32(float %484, float 0xFFF0000000000000), !dbg !26
  %593 = call float @llvm.maxnum.f32(float %592, float %485), !dbg !26
  %594 = call float @llvm.maxnum.f32(float %593, float %486), !dbg !26
  %595 = call float @llvm.maxnum.f32(float %594, float %487), !dbg !26
  %596 = call float @llvm.maxnum.f32(float %595, float %491), !dbg !26
  %597 = call float @llvm.maxnum.f32(float %596, float %492), !dbg !26
  %598 = call float @llvm.maxnum.f32(float %597, float %493), !dbg !26
  %599 = call float @llvm.maxnum.f32(float %598, float %494), !dbg !26
  %600 = call float @llvm.maxnum.f32(float %599, float %498), !dbg !26
  %601 = call float @llvm.maxnum.f32(float %600, float %499), !dbg !26
  %602 = call float @llvm.maxnum.f32(float %601, float %500), !dbg !26
  %603 = call float @llvm.maxnum.f32(float %602, float %501), !dbg !26
  %604 = call float @llvm.maxnum.f32(float %603, float %505), !dbg !26
  %605 = call float @llvm.maxnum.f32(float %604, float %506), !dbg !26
  %606 = call float @llvm.maxnum.f32(float %605, float %507), !dbg !26
  %607 = call float @llvm.maxnum.f32(float %606, float %508), !dbg !26
  %608 = call float @llvm.maxnum.f32(float %607, float %512), !dbg !26
  %609 = call float @llvm.maxnum.f32(float %608, float %513), !dbg !26
  %610 = call float @llvm.maxnum.f32(float %609, float %514), !dbg !26
  %611 = call float @llvm.maxnum.f32(float %610, float %515), !dbg !26
  %612 = call float @llvm.maxnum.f32(float %611, float %519), !dbg !26
  %613 = call float @llvm.maxnum.f32(float %612, float %520), !dbg !26
  %614 = call float @llvm.maxnum.f32(float %613, float %521), !dbg !26
  %615 = call float @llvm.maxnum.f32(float %614, float %522), !dbg !26
  %616 = call float @llvm.maxnum.f32(float %615, float %526), !dbg !26
  %617 = call float @llvm.maxnum.f32(float %616, float %527), !dbg !26
  %618 = call float @llvm.maxnum.f32(float %617, float %528), !dbg !26
  %619 = call float @llvm.maxnum.f32(float %618, float %529), !dbg !26
  %620 = call float @llvm.maxnum.f32(float %619, float %533), !dbg !26
  %621 = call float @llvm.maxnum.f32(float %620, float %534), !dbg !26
  %622 = call float @llvm.maxnum.f32(float %621, float %535), !dbg !26
  %623 = call float @llvm.maxnum.f32(float %622, float %536), !dbg !26
  %624 = bitcast float %623 to i32, !dbg !26
  %625 = call i32 @llvm.amdgcn.ds.bpermute(i32 %576, i32 %624), !dbg !26
  %626 = bitcast i32 %625 to float, !dbg !26
  %627 = call float @llvm.maxnum.f32(float %623, float %626), !dbg !26
  %628 = bitcast float %627 to i32, !dbg !26
  %629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %584, i32 %628), !dbg !26
  %630 = bitcast i32 %629 to float, !dbg !26
  %631 = call float @llvm.maxnum.f32(float %627, float %630), !dbg !26
  %632 = extractvalue [2 x <1 x float>] %29, 1, !dbg !26
  %633 = insertelement <1 x float> poison, float %631, i64 0, !dbg !26
  %634 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %633, <1 x float> %632), !dbg !26
  %635 = shufflevector <1 x float> %591, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %636 = fsub <4 x float> %425, %635, !dbg !27
  %637 = fsub <4 x float> %432, %635, !dbg !27
  %638 = fsub <4 x float> %439, %635, !dbg !27
  %639 = fsub <4 x float> %446, %635, !dbg !27
  %640 = fsub <4 x float> %453, %635, !dbg !27
  %641 = fsub <4 x float> %460, %635, !dbg !27
  %642 = fsub <4 x float> %467, %635, !dbg !27
  %643 = fsub <4 x float> %474, %635, !dbg !27
  %644 = shufflevector <1 x float> %634, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %645 = fsub <4 x float> %483, %644, !dbg !27
  %646 = fsub <4 x float> %490, %644, !dbg !27
  %647 = fsub <4 x float> %497, %644, !dbg !27
  %648 = fsub <4 x float> %504, %644, !dbg !27
  %649 = fsub <4 x float> %511, %644, !dbg !27
  %650 = fsub <4 x float> %518, %644, !dbg !27
  %651 = fsub <4 x float> %525, %644, !dbg !27
  %652 = fsub <4 x float> %532, %644, !dbg !27
  %653 = fsub <1 x float> %589, %591, !dbg !28
  %654 = extractelement <1 x float> %653, i64 0, !dbg !29
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !29
  %656 = insertelement <4 x float> poison, float %655, i64 0, !dbg !34
  %657 = shufflevector <4 x float> %656, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %658 = extractvalue [2 x <1 x float>] %28, 0, !dbg !30
  %659 = extractelement <1 x float> %658, i64 0, !dbg !30
  %660 = fmul float %659, %655, !dbg !30
  %661 = fsub <1 x float> %632, %634, !dbg !28
  %662 = extractelement <1 x float> %661, i64 0, !dbg !29
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !29
  %664 = insertelement <4 x float> poison, float %663, i64 0, !dbg !34
  %665 = shufflevector <4 x float> %664, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %666 = extractvalue [2 x <1 x float>] %28, 1, !dbg !30
  %667 = extractelement <1 x float> %666, i64 0, !dbg !30
  %668 = fmul float %667, %663, !dbg !30
  %669 = extractelement <4 x float> %636, i64 0, !dbg !31
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !31
  %671 = insertelement <4 x float> poison, float %670, i64 0, !dbg !33
  %672 = extractelement <4 x float> %636, i64 1, !dbg !31
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !31
  %674 = insertelement <4 x float> %671, float %673, i64 1, !dbg !33
  %675 = fadd float %670, %673, !dbg !32
  %676 = extractelement <4 x float> %636, i64 2, !dbg !31
  %677 = call float @llvm.amdgcn.exp2.f32(float %676), !dbg !31
  %678 = insertelement <4 x float> %674, float %677, i64 2, !dbg !33
  %679 = fadd float %675, %677, !dbg !32
  %680 = extractelement <4 x float> %636, i64 3, !dbg !31
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !31
  %682 = insertelement <4 x float> %678, float %681, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %682) #9, !dbg !33, !frisk.register_lifetimes !1
  %683 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %683) #9, !dbg !32, !frisk.register_lifetimes !1
  %684 = fadd float %679, %681, !dbg !32
  %685 = extractelement <4 x float> %637, i64 0, !dbg !31
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !31
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !33
  %688 = fadd float %684, %686, !dbg !32
  %689 = extractelement <4 x float> %637, i64 1, !dbg !31
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !31
  %691 = insertelement <4 x float> %687, float %690, i64 1, !dbg !33
  %692 = fadd float %688, %690, !dbg !32
  %693 = extractelement <4 x float> %637, i64 2, !dbg !31
  %694 = call float @llvm.amdgcn.exp2.f32(float %693), !dbg !31
  %695 = insertelement <4 x float> %691, float %694, i64 2, !dbg !33
  %696 = fadd float %692, %694, !dbg !32
  %697 = extractelement <4 x float> %637, i64 3, !dbg !31
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !31
  %699 = insertelement <4 x float> %695, float %698, i64 3, !dbg !33
  %math.result100 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %699) #9, !dbg !33, !frisk.register_lifetimes !1
  %700 = fptrunc <4 x float> %math.result100 to <4 x half>, !dbg !33
  %math.packed101 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %700) #9, !dbg !32, !frisk.register_lifetimes !1
  %701 = fadd float %696, %698, !dbg !32
  %702 = extractelement <4 x float> %638, i64 0, !dbg !31
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !31
  %704 = insertelement <4 x float> poison, float %703, i64 0, !dbg !33
  %705 = fadd float %701, %703, !dbg !32
  %706 = extractelement <4 x float> %638, i64 1, !dbg !31
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !31
  %708 = insertelement <4 x float> %704, float %707, i64 1, !dbg !33
  %709 = fadd float %705, %707, !dbg !32
  %710 = extractelement <4 x float> %638, i64 2, !dbg !31
  %711 = call float @llvm.amdgcn.exp2.f32(float %710), !dbg !31
  %712 = insertelement <4 x float> %708, float %711, i64 2, !dbg !33
  %713 = fadd float %709, %711, !dbg !32
  %714 = extractelement <4 x float> %638, i64 3, !dbg !31
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !31
  %716 = insertelement <4 x float> %712, float %715, i64 3, !dbg !33
  %math.result102 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %716) #9, !dbg !33, !frisk.register_lifetimes !1
  %717 = fptrunc <4 x float> %math.result102 to <4 x half>, !dbg !33
  %math.packed103 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %717) #9, !dbg !32, !frisk.register_lifetimes !1
  %718 = fadd float %713, %715, !dbg !32
  %719 = extractelement <4 x float> %639, i64 0, !dbg !31
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !31
  %721 = insertelement <4 x float> poison, float %720, i64 0, !dbg !33
  %722 = fadd float %718, %720, !dbg !32
  %723 = extractelement <4 x float> %639, i64 1, !dbg !31
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !31
  %725 = insertelement <4 x float> %721, float %724, i64 1, !dbg !33
  %726 = fadd float %722, %724, !dbg !32
  %727 = extractelement <4 x float> %639, i64 2, !dbg !31
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !31
  %729 = insertelement <4 x float> %725, float %728, i64 2, !dbg !33
  %730 = fadd float %726, %728, !dbg !32
  %731 = extractelement <4 x float> %639, i64 3, !dbg !31
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !31
  %733 = insertelement <4 x float> %729, float %732, i64 3, !dbg !33
  %math.result104 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %733) #9, !dbg !33, !frisk.register_lifetimes !1
  %734 = fptrunc <4 x float> %math.result104 to <4 x half>, !dbg !33
  %math.packed105 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %734) #9, !dbg !32, !frisk.register_lifetimes !1
  %735 = fadd float %730, %732, !dbg !32
  %736 = extractelement <4 x float> %640, i64 0, !dbg !31
  %737 = call float @llvm.amdgcn.exp2.f32(float %736), !dbg !31
  %738 = insertelement <4 x float> poison, float %737, i64 0, !dbg !33
  %739 = fadd float %735, %737, !dbg !32
  %740 = extractelement <4 x float> %640, i64 1, !dbg !31
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !31
  %742 = insertelement <4 x float> %738, float %741, i64 1, !dbg !33
  %743 = fadd float %739, %741, !dbg !32
  %744 = extractelement <4 x float> %640, i64 2, !dbg !31
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !31
  %746 = insertelement <4 x float> %742, float %745, i64 2, !dbg !33
  %747 = fadd float %743, %745, !dbg !32
  %748 = extractelement <4 x float> %640, i64 3, !dbg !31
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !31
  %750 = insertelement <4 x float> %746, float %749, i64 3, !dbg !33
  %math.result106 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %750) #9, !dbg !33, !frisk.register_lifetimes !1
  %751 = fptrunc <4 x float> %math.result106 to <4 x half>, !dbg !33
  %math.packed107 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %751) #9, !dbg !32, !frisk.register_lifetimes !1
  %752 = fadd float %747, %749, !dbg !32
  %753 = extractelement <4 x float> %641, i64 0, !dbg !31
  %754 = call float @llvm.amdgcn.exp2.f32(float %753), !dbg !31
  %755 = insertelement <4 x float> poison, float %754, i64 0, !dbg !33
  %756 = fadd float %752, %754, !dbg !32
  %757 = extractelement <4 x float> %641, i64 1, !dbg !31
  %758 = call float @llvm.amdgcn.exp2.f32(float %757), !dbg !31
  %759 = insertelement <4 x float> %755, float %758, i64 1, !dbg !33
  %760 = fadd float %756, %758, !dbg !32
  %761 = extractelement <4 x float> %641, i64 2, !dbg !31
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !31
  %763 = insertelement <4 x float> %759, float %762, i64 2, !dbg !33
  %764 = fadd float %760, %762, !dbg !32
  %765 = extractelement <4 x float> %641, i64 3, !dbg !31
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !31
  %767 = insertelement <4 x float> %763, float %766, i64 3, !dbg !33
  %math.result108 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %767) #9, !dbg !33, !frisk.register_lifetimes !1
  %768 = fptrunc <4 x float> %math.result108 to <4 x half>, !dbg !33
  %math.packed109 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %768) #9, !dbg !32, !frisk.register_lifetimes !1
  %769 = fadd float %764, %766, !dbg !32
  %770 = extractelement <4 x float> %642, i64 0, !dbg !31
  %771 = call float @llvm.amdgcn.exp2.f32(float %770), !dbg !31
  %772 = insertelement <4 x float> poison, float %771, i64 0, !dbg !33
  %773 = fadd float %769, %771, !dbg !32
  %774 = extractelement <4 x float> %642, i64 1, !dbg !31
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !31
  %776 = insertelement <4 x float> %772, float %775, i64 1, !dbg !33
  %777 = fadd float %773, %775, !dbg !32
  %778 = extractelement <4 x float> %642, i64 2, !dbg !31
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !31
  %780 = insertelement <4 x float> %776, float %779, i64 2, !dbg !33
  %781 = fadd float %777, %779, !dbg !32
  %782 = extractelement <4 x float> %642, i64 3, !dbg !31
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !31
  %784 = insertelement <4 x float> %780, float %783, i64 3, !dbg !33
  %math.result110 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %784) #9, !dbg !33, !frisk.register_lifetimes !1
  %785 = fptrunc <4 x float> %math.result110 to <4 x half>, !dbg !33
  %math.packed111 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %785) #9, !dbg !32, !frisk.register_lifetimes !1
  %786 = fadd float %781, %783, !dbg !32
  %787 = extractelement <4 x float> %643, i64 0, !dbg !31
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !31
  %789 = insertelement <4 x float> poison, float %788, i64 0, !dbg !33
  %790 = fadd float %786, %788, !dbg !32
  %791 = extractelement <4 x float> %643, i64 1, !dbg !31
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !31
  %793 = insertelement <4 x float> %789, float %792, i64 1, !dbg !33
  %794 = fadd float %790, %792, !dbg !32
  %795 = extractelement <4 x float> %643, i64 2, !dbg !31
  %796 = call float @llvm.amdgcn.exp2.f32(float %795), !dbg !31
  %797 = insertelement <4 x float> %793, float %796, i64 2, !dbg !33
  %798 = fadd float %794, %796, !dbg !32
  %799 = extractelement <4 x float> %643, i64 3, !dbg !31
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !31
  %801 = insertelement <4 x float> %797, float %800, i64 3, !dbg !33
  %math.result112 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %801) #9, !dbg !33, !frisk.register_lifetimes !1
  %802 = fptrunc <4 x float> %math.result112 to <4 x half>, !dbg !33
  %math.packed113 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %802) #9, !dbg !32, !frisk.register_lifetimes !1
  %803 = fadd float %798, %800, !dbg !32
  %804 = extractelement <4 x float> %645, i64 0, !dbg !31
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !31
  %806 = insertelement <4 x float> poison, float %805, i64 0, !dbg !33
  %807 = extractelement <4 x float> %645, i64 1, !dbg !31
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !31
  %809 = insertelement <4 x float> %806, float %808, i64 1, !dbg !33
  %810 = fadd float %805, %808, !dbg !32
  %811 = extractelement <4 x float> %645, i64 2, !dbg !31
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !31
  %813 = insertelement <4 x float> %809, float %812, i64 2, !dbg !33
  %814 = fadd float %810, %812, !dbg !32
  %815 = extractelement <4 x float> %645, i64 3, !dbg !31
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !31
  %817 = insertelement <4 x float> %813, float %816, i64 3, !dbg !33
  %math.result114 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !33, !frisk.register_lifetimes !1
  %818 = fptrunc <4 x float> %math.result114 to <4 x half>, !dbg !33
  %math.packed115 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !32, !frisk.register_lifetimes !1
  %819 = fadd float %814, %816, !dbg !32
  %820 = extractelement <4 x float> %646, i64 0, !dbg !31
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !31
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !33
  %823 = fadd float %819, %821, !dbg !32
  %824 = extractelement <4 x float> %646, i64 1, !dbg !31
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !31
  %826 = insertelement <4 x float> %822, float %825, i64 1, !dbg !33
  %827 = fadd float %823, %825, !dbg !32
  %828 = extractelement <4 x float> %646, i64 2, !dbg !31
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !31
  %830 = insertelement <4 x float> %826, float %829, i64 2, !dbg !33
  %831 = fadd float %827, %829, !dbg !32
  %832 = extractelement <4 x float> %646, i64 3, !dbg !31
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !31
  %834 = insertelement <4 x float> %830, float %833, i64 3, !dbg !33
  %math.result116 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !33, !frisk.register_lifetimes !1
  %835 = fptrunc <4 x float> %math.result116 to <4 x half>, !dbg !33
  %math.packed117 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !32, !frisk.register_lifetimes !1
  %836 = fadd float %831, %833, !dbg !32
  %837 = extractelement <4 x float> %647, i64 0, !dbg !31
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !31
  %839 = insertelement <4 x float> poison, float %838, i64 0, !dbg !33
  %840 = fadd float %836, %838, !dbg !32
  %841 = extractelement <4 x float> %647, i64 1, !dbg !31
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !31
  %843 = insertelement <4 x float> %839, float %842, i64 1, !dbg !33
  %844 = fadd float %840, %842, !dbg !32
  %845 = extractelement <4 x float> %647, i64 2, !dbg !31
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !31
  %847 = insertelement <4 x float> %843, float %846, i64 2, !dbg !33
  %848 = fadd float %844, %846, !dbg !32
  %849 = extractelement <4 x float> %647, i64 3, !dbg !31
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !31
  %851 = insertelement <4 x float> %847, float %850, i64 3, !dbg !33
  %math.result118 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #9, !dbg !33, !frisk.register_lifetimes !1
  %852 = fptrunc <4 x float> %math.result118 to <4 x half>, !dbg !33
  %math.packed119 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #9, !dbg !32, !frisk.register_lifetimes !1
  %853 = fadd float %848, %850, !dbg !32
  %854 = extractelement <4 x float> %648, i64 0, !dbg !31
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !31
  %856 = insertelement <4 x float> poison, float %855, i64 0, !dbg !33
  %857 = fadd float %853, %855, !dbg !32
  %858 = extractelement <4 x float> %648, i64 1, !dbg !31
  %859 = call float @llvm.amdgcn.exp2.f32(float %858), !dbg !31
  %860 = insertelement <4 x float> %856, float %859, i64 1, !dbg !33
  %861 = fadd float %857, %859, !dbg !32
  %862 = extractelement <4 x float> %648, i64 2, !dbg !31
  %863 = call float @llvm.amdgcn.exp2.f32(float %862), !dbg !31
  %864 = insertelement <4 x float> %860, float %863, i64 2, !dbg !33
  %865 = fadd float %861, %863, !dbg !32
  %866 = extractelement <4 x float> %648, i64 3, !dbg !31
  %867 = call float @llvm.amdgcn.exp2.f32(float %866), !dbg !31
  %868 = insertelement <4 x float> %864, float %867, i64 3, !dbg !33
  %math.result120 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %868) #9, !dbg !33, !frisk.register_lifetimes !1
  %869 = fptrunc <4 x float> %math.result120 to <4 x half>, !dbg !33
  %math.packed121 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %869) #9, !dbg !32, !frisk.register_lifetimes !1
  %870 = fadd float %865, %867, !dbg !32
  %871 = extractelement <4 x float> %649, i64 0, !dbg !31
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !31
  %873 = insertelement <4 x float> poison, float %872, i64 0, !dbg !33
  %874 = fadd float %870, %872, !dbg !32
  %875 = extractelement <4 x float> %649, i64 1, !dbg !31
  %876 = call float @llvm.amdgcn.exp2.f32(float %875), !dbg !31
  %877 = insertelement <4 x float> %873, float %876, i64 1, !dbg !33
  %878 = fadd float %874, %876, !dbg !32
  %879 = extractelement <4 x float> %649, i64 2, !dbg !31
  %880 = call float @llvm.amdgcn.exp2.f32(float %879), !dbg !31
  %881 = insertelement <4 x float> %877, float %880, i64 2, !dbg !33
  %882 = fadd float %878, %880, !dbg !32
  %883 = extractelement <4 x float> %649, i64 3, !dbg !31
  %884 = call float @llvm.amdgcn.exp2.f32(float %883), !dbg !31
  %885 = insertelement <4 x float> %881, float %884, i64 3, !dbg !33
  %math.result122 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %885) #9, !dbg !33, !frisk.register_lifetimes !1
  %886 = fptrunc <4 x float> %math.result122 to <4 x half>, !dbg !33
  %math.packed123 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %886) #9, !dbg !32, !frisk.register_lifetimes !1
  %887 = fadd float %882, %884, !dbg !32
  %888 = extractelement <4 x float> %650, i64 0, !dbg !31
  %889 = call float @llvm.amdgcn.exp2.f32(float %888), !dbg !31
  %890 = insertelement <4 x float> poison, float %889, i64 0, !dbg !33
  %891 = fadd float %887, %889, !dbg !32
  %892 = extractelement <4 x float> %650, i64 1, !dbg !31
  %893 = call float @llvm.amdgcn.exp2.f32(float %892), !dbg !31
  %894 = insertelement <4 x float> %890, float %893, i64 1, !dbg !33
  %895 = fadd float %891, %893, !dbg !32
  %896 = extractelement <4 x float> %650, i64 2, !dbg !31
  %897 = call float @llvm.amdgcn.exp2.f32(float %896), !dbg !31
  %898 = insertelement <4 x float> %894, float %897, i64 2, !dbg !33
  %899 = fadd float %895, %897, !dbg !32
  %900 = extractelement <4 x float> %650, i64 3, !dbg !31
  %901 = call float @llvm.amdgcn.exp2.f32(float %900), !dbg !31
  %902 = insertelement <4 x float> %898, float %901, i64 3, !dbg !33
  %math.result124 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %902) #9, !dbg !33, !frisk.register_lifetimes !1
  %903 = fptrunc <4 x float> %math.result124 to <4 x half>, !dbg !33
  %math.packed125 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %903) #9, !dbg !32, !frisk.register_lifetimes !1
  %904 = fadd float %899, %901, !dbg !32
  %905 = extractelement <4 x float> %651, i64 0, !dbg !31
  %906 = call float @llvm.amdgcn.exp2.f32(float %905), !dbg !31
  %907 = insertelement <4 x float> poison, float %906, i64 0, !dbg !33
  %908 = fadd float %904, %906, !dbg !32
  %909 = extractelement <4 x float> %651, i64 1, !dbg !31
  %910 = call float @llvm.amdgcn.exp2.f32(float %909), !dbg !31
  %911 = insertelement <4 x float> %907, float %910, i64 1, !dbg !33
  %912 = fadd float %908, %910, !dbg !32
  %913 = extractelement <4 x float> %651, i64 2, !dbg !31
  %914 = call float @llvm.amdgcn.exp2.f32(float %913), !dbg !31
  %915 = insertelement <4 x float> %911, float %914, i64 2, !dbg !33
  %916 = fadd float %912, %914, !dbg !32
  %917 = extractelement <4 x float> %651, i64 3, !dbg !31
  %918 = call float @llvm.amdgcn.exp2.f32(float %917), !dbg !31
  %919 = insertelement <4 x float> %915, float %918, i64 3, !dbg !33
  %math.result126 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %919) #9, !dbg !33, !frisk.register_lifetimes !1
  %920 = fptrunc <4 x float> %math.result126 to <4 x half>, !dbg !33
  %math.packed127 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %920) #9, !dbg !32, !frisk.register_lifetimes !1
  %921 = fadd float %916, %918, !dbg !32
  %922 = extractelement <4 x float> %652, i64 0, !dbg !31
  %923 = call float @llvm.amdgcn.exp2.f32(float %922), !dbg !31
  %924 = insertelement <4 x float> poison, float %923, i64 0, !dbg !33
  %925 = fadd float %921, %923, !dbg !32
  %926 = extractelement <4 x float> %652, i64 1, !dbg !31
  %927 = call float @llvm.amdgcn.exp2.f32(float %926), !dbg !31
  %928 = insertelement <4 x float> %924, float %927, i64 1, !dbg !33
  %929 = fadd float %925, %927, !dbg !32
  %930 = extractelement <4 x float> %652, i64 2, !dbg !31
  %931 = call float @llvm.amdgcn.exp2.f32(float %930), !dbg !31
  %932 = insertelement <4 x float> %928, float %931, i64 2, !dbg !33
  %933 = fadd float %929, %931, !dbg !32
  %934 = extractelement <4 x float> %652, i64 3, !dbg !31
  %935 = call float @llvm.amdgcn.exp2.f32(float %934), !dbg !31
  %936 = insertelement <4 x float> %932, float %935, i64 3, !dbg !33
  %math.result128 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %936) #9, !dbg !33, !frisk.register_lifetimes !1
  %937 = fptrunc <4 x float> %math.result128 to <4 x half>, !dbg !33
  %math.packed129 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %937) #9, !dbg !32, !frisk.register_lifetimes !1
  %938 = fadd float %933, %935, !dbg !32
  %939 = bitcast float %803 to i32, !dbg !32
  %940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %576, i32 %939), !dbg !32
  %941 = bitcast i32 %940 to float, !dbg !32
  %942 = fadd float %803, %941, !dbg !32
  %943 = bitcast float %942 to i32, !dbg !32
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %584, i32 %943), !dbg !32
  %945 = bitcast i32 %944 to float, !dbg !32
  %946 = fadd float %942, %945, !dbg !32
  %947 = fadd float %946, %660, !dbg !32
  %948 = insertelement <1 x float> poison, float %947, i64 0, !dbg !8
  %949 = insertvalue [2 x <1 x float>] poison, <1 x float> %948, 0, !dbg !8
  %950 = bitcast float %938 to i32, !dbg !32
  %951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %576, i32 %950), !dbg !32
  %952 = bitcast i32 %951 to float, !dbg !32
  %953 = fadd float %938, %952, !dbg !32
  %954 = bitcast float %953 to i32, !dbg !32
  %955 = call i32 @llvm.amdgcn.ds.bpermute(i32 %584, i32 %954), !dbg !32
  %956 = bitcast i32 %955 to float, !dbg !32
  %957 = fadd float %953, %956, !dbg !32
  %958 = fadd float %957, %668, !dbg !32
  %959 = insertelement <1 x float> poison, float %958, i64 0, !dbg !8
  %960 = insertvalue [2 x <1 x float>] %949, <1 x float> %959, 1, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread321 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local322 = shl nuw nsw i32 %address.thread321, 3, !dbg !15
  %address.local323 = and i32 %address.local322, 120, !dbg !15
  %address.local324 = shl nuw nsw i32 %address.thread321, 5, !dbg !15
  %address.local325 = and i32 %address.local324, 1536, !dbg !15
  %address.local326 = or disjoint i32 %address.local323, %address.local325, !dbg !15
  %address.local327 = zext nneg i32 %address.local326 to i64, !dbg !15
  %address.local328 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !15
  %961 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %962 = bitcast <4 x i32> %961 to <8 x half>, !dbg !15
  %address.local329 = zext nneg i32 %address.thread321 to i64, !dbg !13
  %address.local330 = and i64 %address.local329, 60, !dbg !15
  %address.local331 = add nuw nsw i64 %address.local330, %address.local329, !dbg !15
  %address.local332 = shl nuw nsw i64 %address.local331, 3, !dbg !15
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local334 = and i64 %address.local329, 63, !dbg !13
  %address.local335 = lshr i64 %address.local334, 4, !dbg !15
  %address.local336 = mul nuw nsw i64 %address.local335, 384, !dbg !15
  %address.local337 = getelementptr [2 x i8], ptr addrspace(3) %address.local333, i64 %address.local336, !dbg !15
  %address.local338 = lshr i64 %address.local329, 2, !dbg !7
  %address.local339 = mul nsw i64 %address.local338, -32, !dbg !15
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) %address.local337, i64 %address.local339, !dbg !15
  %address.local341 = getelementptr i8, ptr addrspace(3) %address.local340, i64 256, !dbg !15
  %963 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %964 = bitcast <4 x i32> %963 to <8 x half>, !dbg !15
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) %address.local342, i64 %address.local336, !dbg !15
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) %address.local343, i64 %address.local339, !dbg !15
  %address.local345 = getelementptr i8, ptr addrspace(3) %address.local344, i64 512, !dbg !15
  %965 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %966 = bitcast <4 x i32> %965 to <8 x half>, !dbg !15
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) %address.local346, i64 %address.local336, !dbg !15
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) %address.local347, i64 %address.local339, !dbg !15
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local348, i64 768, !dbg !15
  %967 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %968 = bitcast <4 x i32> %967 to <8 x half>, !dbg !15
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local327, !dbg !15
  %address.local351 = getelementptr i8, ptr addrspace(3) %address.local350, i64 4096, !dbg !15
  %969 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %970 = bitcast <4 x i32> %969 to <8 x half>, !dbg !15
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) %address.local352, i64 %address.local336, !dbg !15
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) %address.local353, i64 %address.local339, !dbg !15
  %address.local355 = getelementptr i8, ptr addrspace(3) %address.local354, i64 4352, !dbg !15
  %971 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %972 = bitcast <4 x i32> %971 to <8 x half>, !dbg !15
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) %address.local356, i64 %address.local336, !dbg !15
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) %address.local357, i64 %address.local339, !dbg !15
  %address.local359 = getelementptr i8, ptr addrspace(3) %address.local358, i64 4608, !dbg !15
  %973 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %974 = bitcast <4 x i32> %973 to <8 x half>, !dbg !15
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local332, !dbg !15
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) %address.local360, i64 %address.local336, !dbg !15
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) %address.local361, i64 %address.local339, !dbg !15
  %address.local363 = getelementptr i8, ptr addrspace(3) %address.local362, i64 4864, !dbg !15
  %975 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %976 = bitcast <4 x i32> %975 to <8 x half>, !dbg !15
  %977 = add nsw i64 %35, %6, !dbg !17
  %978 = shl i64 %977, 19, !dbg !17
  %979 = and i64 %978, 16252928, !dbg !17
  %980 = add i64 %979, %239, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %981 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %980, !dbg !17
  %982 = getelementptr i8, ptr addrspace(1) %981, i64 64, !dbg !17
  %address.local364 = shl nuw nsw i64 %address.local329, 3, !dbg !12
  %address.local365 = and i64 %address.local364, 1536, !dbg !12
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local365, !dbg !12
  %address.local367 = ptrtoint ptr addrspace(3) %address.local366 to i64, !dbg !12
  %address.local368 = trunc i64 %address.local367 to i32, !dbg !12
  %983 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !17
  %984 = zext i32 %983 to i64, !dbg !17
  %985 = inttoptr i64 %984 to ptr addrspace(3), !dbg !17
  %address.local369 = shl nuw nsw i64 %address.local329, 5, !dbg !17
  %address.local370 = and i64 %address.local369, 8064, !dbg !17
  %address.local371 = shl nuw nsw i64 %address.local329, 3, !dbg !17
  %address.local372 = and i64 %address.local371, 24, !dbg !17
  %address.local373 = or disjoint i64 %address.local370, %address.local372, !dbg !17
  %address.local374 = getelementptr [2 x i8], ptr addrspace(1) %982, i64 %address.local373, !dbg !17
  %986 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %987 = ptrtoint ptr addrspace(1) %address.local374 to i64, !dbg !17
  %988 = sub i64 %987, %986, !dbg !17
  %dma.buffer.offset894 = trunc i64 %988 to i32, !dbg !17
  %989 = trunc i64 %986 to i32, !dbg !17
  %990 = insertelement <4 x i32> poison, i32 %989, i64 0, !dbg !17
  %991 = lshr i64 %986, 32, !dbg !17
  %992 = trunc i64 %991 to i32, !dbg !17
  %993 = insertelement <4 x i32> %990, i32 %992, i64 1, !dbg !17
  %994 = insertelement <4 x i32> %993, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc895 = insertelement <4 x i32> %994, i32 131072, i64 3, !dbg !17
  %995 = ptrtoint ptr addrspace(3) %985 to i32, !dbg !17
  %dma.buffer.lds896 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %995), !dbg !17
  %996 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds896, i32 %dma.buffer.offset894, <4 x i32> %dma.buffer.rsrc895) #10, !dbg !17
  %address.local375 = or disjoint i64 %address.local329, 256, !dbg !12
  %address.local376 = shl nuw nsw i64 %address.local375, 3, !dbg !12
  %address.local377 = and i64 %address.local376, 3584, !dbg !12
  %address.local378 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local377, !dbg !12
  %address.local379 = ptrtoint ptr addrspace(3) %address.local378 to i64, !dbg !12
  %address.local380 = trunc i64 %address.local379 to i32, !dbg !12
  %997 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !17
  %998 = zext i32 %997 to i64, !dbg !17
  %999 = inttoptr i64 %998 to ptr addrspace(3), !dbg !17
  %address.local381 = shl nuw nsw i64 %address.local375, 5, !dbg !17
  %address.local382 = and i64 %address.local381, 16256, !dbg !17
  %address.local383 = shl nuw nsw i64 %address.local329, 3, !dbg !17
  %address.local384 = and i64 %address.local383, 24, !dbg !17
  %address.local385 = or disjoint i64 %address.local382, %address.local384, !dbg !17
  %address.local386 = getelementptr [2 x i8], ptr addrspace(1) %982, i64 %address.local385, !dbg !17
  %1000 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1001 = ptrtoint ptr addrspace(1) %address.local386 to i64, !dbg !17
  %1002 = sub i64 %1001, %1000, !dbg !17
  %dma.buffer.offset891 = trunc i64 %1002 to i32, !dbg !17
  %1003 = trunc i64 %1000 to i32, !dbg !17
  %1004 = insertelement <4 x i32> poison, i32 %1003, i64 0, !dbg !17
  %1005 = lshr i64 %1000, 32, !dbg !17
  %1006 = trunc i64 %1005 to i32, !dbg !17
  %1007 = insertelement <4 x i32> %1004, i32 %1006, i64 1, !dbg !17
  %1008 = insertelement <4 x i32> %1007, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc892 = insertelement <4 x i32> %1008, i32 131072, i64 3, !dbg !17
  %1009 = ptrtoint ptr addrspace(3) %999 to i32, !dbg !17
  %dma.buffer.lds893 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1009), !dbg !17
  %1010 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds893, i32 %dma.buffer.offset891, <4 x i32> %dma.buffer.rsrc892) #10, !dbg !17
  %1011 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1012 = extractvalue [2 x <8 x float>] %27, 0, !dbg !34
  %1013 = shufflevector <8 x float> %1012, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1014 = fmul <4 x float> %1013, %657, !dbg !34
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1011, <4 x float> %1014) #9, !dbg !15
  %1016 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1016, <4 x float> %1015) #9, !dbg !15
  %1018 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1018, <4 x float> %1017) #9, !dbg !15
  %1020 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1020, <4 x float> %1019) #9, !dbg !15
  %1022 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1022, <4 x float> %1021) #9, !dbg !15
  %1024 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1024, <4 x float> %1023) #9, !dbg !15
  %1026 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1026, <4 x float> %1025) #9, !dbg !15
  %1028 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1028, <4 x float> %1027) #9, !dbg !15
  %1030 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1031 = shufflevector <8 x float> %1012, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1032 = fmul <4 x float> %1031, %657, !dbg !34
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1030, <4 x float> %1032) #9, !dbg !15
  %1034 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1034, <4 x float> %1033) #9, !dbg !15
  %1036 = shufflevector <8 x half> %966, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1036, <4 x float> %1035) #9, !dbg !15
  %1038 = shufflevector <8 x half> %968, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1038, <4 x float> %1037) #9, !dbg !15
  %1040 = shufflevector <8 x half> %970, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1040, <4 x float> %1039) #9, !dbg !15
  %1042 = shufflevector <8 x half> %972, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1042, <4 x float> %1041) #9, !dbg !15
  %1044 = shufflevector <8 x half> %974, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1044, <4 x float> %1043) #9, !dbg !15
  %1046 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1046, <4 x float> %1045) #9, !dbg !15
  %1048 = extractvalue [2 x <8 x float>] %27, 1, !dbg !34
  %1049 = shufflevector <8 x float> %1048, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1050 = fmul <4 x float> %1049, %665, !dbg !34
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1011, <4 x float> %1050) #9, !dbg !15
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1016, <4 x float> %1051) #9, !dbg !15
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1018, <4 x float> %1052) #9, !dbg !15
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1020, <4 x float> %1053) #9, !dbg !15
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1022, <4 x float> %1054) #9, !dbg !15
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1024, <4 x float> %1055) #9, !dbg !15
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1026, <4 x float> %1056) #9, !dbg !15
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1028, <4 x float> %1057) #9, !dbg !15
  %1059 = shufflevector <8 x float> %1048, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1060 = fmul <4 x float> %1059, %665, !dbg !34
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1030, <4 x float> %1060) #9, !dbg !15
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1034, <4 x float> %1061) #9, !dbg !15
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1036, <4 x float> %1062) #9, !dbg !15
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1038, <4 x float> %1063) #9, !dbg !15
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1040, <4 x float> %1064) #9, !dbg !15
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1042, <4 x float> %1065) #9, !dbg !15
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1044, <4 x float> %1066) #9, !dbg !15
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1046, <4 x float> %1067) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1069 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %1070 = bitcast <4 x i32> %1069 to <8 x half>, !dbg !15
  %1071 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %1072 = bitcast <4 x i32> %1071 to <8 x half>, !dbg !15
  %1073 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %1074 = bitcast <4 x i32> %1073 to <8 x half>, !dbg !15
  %1075 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %1076 = bitcast <4 x i32> %1075 to <8 x half>, !dbg !15
  %1077 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %1078 = bitcast <4 x i32> %1077 to <8 x half>, !dbg !15
  %1079 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %1080 = bitcast <4 x i32> %1079 to <8 x half>, !dbg !15
  %1081 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %1082 = bitcast <4 x i32> %1081 to <8 x half>, !dbg !15
  %1083 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %1084 = bitcast <4 x i32> %1083 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1085 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %980, !dbg !17
  %1086 = getelementptr i8, ptr addrspace(1) %1085, i64 128, !dbg !17
  %1087 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !17
  %1088 = zext i32 %1087 to i64, !dbg !17
  %1089 = inttoptr i64 %1088 to ptr addrspace(3), !dbg !17
  %address.local387 = getelementptr [2 x i8], ptr addrspace(1) %1086, i64 %address.local373, !dbg !17
  %1090 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1091 = ptrtoint ptr addrspace(1) %address.local387 to i64, !dbg !17
  %1092 = sub i64 %1091, %1090, !dbg !17
  %dma.buffer.offset888 = trunc i64 %1092 to i32, !dbg !17
  %1093 = trunc i64 %1090 to i32, !dbg !17
  %1094 = insertelement <4 x i32> poison, i32 %1093, i64 0, !dbg !17
  %1095 = lshr i64 %1090, 32, !dbg !17
  %1096 = trunc i64 %1095 to i32, !dbg !17
  %1097 = insertelement <4 x i32> %1094, i32 %1096, i64 1, !dbg !17
  %1098 = insertelement <4 x i32> %1097, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc889 = insertelement <4 x i32> %1098, i32 131072, i64 3, !dbg !17
  %1099 = ptrtoint ptr addrspace(3) %1089 to i32, !dbg !17
  %dma.buffer.lds890 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1099), !dbg !17
  %1100 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds890, i32 %dma.buffer.offset888, <4 x i32> %dma.buffer.rsrc889) #10, !dbg !17
  %1101 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !17
  %1102 = zext i32 %1101 to i64, !dbg !17
  %1103 = inttoptr i64 %1102 to ptr addrspace(3), !dbg !17
  %address.local388 = getelementptr [2 x i8], ptr addrspace(1) %1086, i64 %address.local385, !dbg !17
  %1104 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1105 = ptrtoint ptr addrspace(1) %address.local388 to i64, !dbg !17
  %1106 = sub i64 %1105, %1104, !dbg !17
  %dma.buffer.offset885 = trunc i64 %1106 to i32, !dbg !17
  %1107 = trunc i64 %1104 to i32, !dbg !17
  %1108 = insertelement <4 x i32> poison, i32 %1107, i64 0, !dbg !17
  %1109 = lshr i64 %1104, 32, !dbg !17
  %1110 = trunc i64 %1109 to i32, !dbg !17
  %1111 = insertelement <4 x i32> %1108, i32 %1110, i64 1, !dbg !17
  %1112 = insertelement <4 x i32> %1111, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc886 = insertelement <4 x i32> %1112, i32 131072, i64 3, !dbg !17
  %1113 = ptrtoint ptr addrspace(3) %1103 to i32, !dbg !17
  %dma.buffer.lds887 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1113), !dbg !17
  %1114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds887, i32 %dma.buffer.offset885, <4 x i32> %dma.buffer.rsrc886) #10, !dbg !17
  %1115 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1116 = extractvalue [2 x <8 x float>] %30, 0, !dbg !34
  %1117 = shufflevector <8 x float> %1116, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1118 = fmul <4 x float> %1117, %657, !dbg !34
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1115, <4 x float> %1118) #9, !dbg !15
  %1120 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1120, <4 x float> %1119) #9, !dbg !15
  %1122 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1122, <4 x float> %1121) #9, !dbg !15
  %1124 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1124, <4 x float> %1123) #9, !dbg !15
  %1126 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1126, <4 x float> %1125) #9, !dbg !15
  %1128 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1128, <4 x float> %1127) #9, !dbg !15
  %1130 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1130, <4 x float> %1129) #9, !dbg !15
  %1132 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1132, <4 x float> %1131) #9, !dbg !15
  %1134 = shufflevector <8 x half> %1070, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1135 = shufflevector <8 x float> %1116, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1136 = fmul <4 x float> %1135, %657, !dbg !34
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1134, <4 x float> %1136) #9, !dbg !15
  %1138 = shufflevector <8 x half> %1072, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1138, <4 x float> %1137) #9, !dbg !15
  %1140 = shufflevector <8 x half> %1074, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1140, <4 x float> %1139) #9, !dbg !15
  %1142 = shufflevector <8 x half> %1076, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1142, <4 x float> %1141) #9, !dbg !15
  %1144 = shufflevector <8 x half> %1078, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1144, <4 x float> %1143) #9, !dbg !15
  %1146 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1146, <4 x float> %1145) #9, !dbg !15
  %1148 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1148, <4 x float> %1147) #9, !dbg !15
  %1150 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1150, <4 x float> %1149) #9, !dbg !15
  %1152 = extractvalue [2 x <8 x float>] %30, 1, !dbg !34
  %1153 = shufflevector <8 x float> %1152, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1154 = fmul <4 x float> %1153, %665, !dbg !34
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1115, <4 x float> %1154) #9, !dbg !15
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1120, <4 x float> %1155) #9, !dbg !15
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1122, <4 x float> %1156) #9, !dbg !15
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1124, <4 x float> %1157) #9, !dbg !15
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1126, <4 x float> %1158) #9, !dbg !15
  %1160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1128, <4 x float> %1159) #9, !dbg !15
  %1161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1130, <4 x float> %1160) #9, !dbg !15
  %1162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1132, <4 x float> %1161) #9, !dbg !15
  %1163 = shufflevector <8 x float> %1152, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1164 = fmul <4 x float> %1163, %665, !dbg !34
  %1165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1134, <4 x float> %1164) #9, !dbg !15
  %1166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1138, <4 x float> %1165) #9, !dbg !15
  %1167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1140, <4 x float> %1166) #9, !dbg !15
  %1168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1142, <4 x float> %1167) #9, !dbg !15
  %1169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1144, <4 x float> %1168) #9, !dbg !15
  %1170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1146, <4 x float> %1169) #9, !dbg !15
  %1171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1148, <4 x float> %1170) #9, !dbg !15
  %1172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1150, <4 x float> %1171) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !15
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !15
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !15
  %1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !15
  %1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %1182 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !15
  %1183 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %1184 = bitcast <4 x i32> %1183 to <8 x half>, !dbg !15
  %1185 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %1186 = bitcast <4 x i32> %1185 to <8 x half>, !dbg !15
  %1187 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %1188 = bitcast <4 x i32> %1187 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1189 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %980, !dbg !17
  %1190 = getelementptr i8, ptr addrspace(1) %1189, i64 192, !dbg !17
  %1191 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local368), !dbg !17
  %1192 = zext i32 %1191 to i64, !dbg !17
  %1193 = inttoptr i64 %1192 to ptr addrspace(3), !dbg !17
  %address.local389 = getelementptr [2 x i8], ptr addrspace(1) %1190, i64 %address.local373, !dbg !17
  %1194 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1195 = ptrtoint ptr addrspace(1) %address.local389 to i64, !dbg !17
  %1196 = sub i64 %1195, %1194, !dbg !17
  %dma.buffer.offset882 = trunc i64 %1196 to i32, !dbg !17
  %1197 = trunc i64 %1194 to i32, !dbg !17
  %1198 = insertelement <4 x i32> poison, i32 %1197, i64 0, !dbg !17
  %1199 = lshr i64 %1194, 32, !dbg !17
  %1200 = trunc i64 %1199 to i32, !dbg !17
  %1201 = insertelement <4 x i32> %1198, i32 %1200, i64 1, !dbg !17
  %1202 = insertelement <4 x i32> %1201, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc883 = insertelement <4 x i32> %1202, i32 131072, i64 3, !dbg !17
  %1203 = ptrtoint ptr addrspace(3) %1193 to i32, !dbg !17
  %dma.buffer.lds884 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1203), !dbg !17
  %1204 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds884, i32 %dma.buffer.offset882, <4 x i32> %dma.buffer.rsrc883) #10, !dbg !17
  %1205 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local380), !dbg !17
  %1206 = zext i32 %1205 to i64, !dbg !17
  %1207 = inttoptr i64 %1206 to ptr addrspace(3), !dbg !17
  %address.local390 = getelementptr [2 x i8], ptr addrspace(1) %1190, i64 %address.local385, !dbg !17
  %1208 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1209 = ptrtoint ptr addrspace(1) %address.local390 to i64, !dbg !17
  %1210 = sub i64 %1209, %1208, !dbg !17
  %dma.buffer.offset879 = trunc i64 %1210 to i32, !dbg !17
  %1211 = trunc i64 %1208 to i32, !dbg !17
  %1212 = insertelement <4 x i32> poison, i32 %1211, i64 0, !dbg !17
  %1213 = lshr i64 %1208, 32, !dbg !17
  %1214 = trunc i64 %1213 to i32, !dbg !17
  %1215 = insertelement <4 x i32> %1212, i32 %1214, i64 1, !dbg !17
  %1216 = insertelement <4 x i32> %1215, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc880 = insertelement <4 x i32> %1216, i32 131072, i64 3, !dbg !17
  %1217 = ptrtoint ptr addrspace(3) %1207 to i32, !dbg !17
  %dma.buffer.lds881 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1217), !dbg !17
  %1218 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds881, i32 %dma.buffer.offset879, <4 x i32> %dma.buffer.rsrc880) #10, !dbg !17
  %1219 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1220 = extractvalue [2 x <8 x float>] %31, 0, !dbg !34
  %1221 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1222 = fmul <4 x float> %1221, %657, !dbg !34
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1219, <4 x float> %1222) #9, !dbg !15
  %1224 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1224, <4 x float> %1223) #9, !dbg !15
  %1226 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1226, <4 x float> %1225) #9, !dbg !15
  %1228 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1228, <4 x float> %1227) #9, !dbg !15
  %1230 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1230, <4 x float> %1229) #9, !dbg !15
  %1232 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1232, <4 x float> %1231) #9, !dbg !15
  %1234 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1234, <4 x float> %1233) #9, !dbg !15
  %1236 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1236, <4 x float> %1235) #9, !dbg !15
  %1238 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1239 = shufflevector <8 x float> %1220, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1240 = fmul <4 x float> %1239, %657, !dbg !34
  %1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1238, <4 x float> %1240) #9, !dbg !15
  %1242 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1242, <4 x float> %1241) #9, !dbg !15
  %1244 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1244, <4 x float> %1243) #9, !dbg !15
  %1246 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1246, <4 x float> %1245) #9, !dbg !15
  %1248 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1248, <4 x float> %1247) #9, !dbg !15
  %1250 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1250, <4 x float> %1249) #9, !dbg !15
  %1252 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1252, <4 x float> %1251) #9, !dbg !15
  %1254 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1254, <4 x float> %1253) #9, !dbg !15
  %1256 = extractvalue [2 x <8 x float>] %31, 1, !dbg !34
  %1257 = shufflevector <8 x float> %1256, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1258 = fmul <4 x float> %1257, %665, !dbg !34
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1219, <4 x float> %1258) #9, !dbg !15
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1224, <4 x float> %1259) #9, !dbg !15
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1226, <4 x float> %1260) #9, !dbg !15
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1228, <4 x float> %1261) #9, !dbg !15
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1230, <4 x float> %1262) #9, !dbg !15
  %1264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1232, <4 x float> %1263) #9, !dbg !15
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1234, <4 x float> %1264) #9, !dbg !15
  %1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1236, <4 x float> %1265) #9, !dbg !15
  %1267 = shufflevector <8 x float> %1256, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1268 = fmul <4 x float> %1267, %665, !dbg !34
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1238, <4 x float> %1268) #9, !dbg !15
  %1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1242, <4 x float> %1269) #9, !dbg !15
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1244, <4 x float> %1270) #9, !dbg !15
  %1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1246, <4 x float> %1271) #9, !dbg !15
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1248, <4 x float> %1272) #9, !dbg !15
  %1274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1250, <4 x float> %1273) #9, !dbg !15
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1252, <4 x float> %1274) #9, !dbg !15
  %1276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1254, <4 x float> %1275) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1277 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local328, i32 0) #0, !dbg !15
  %1278 = bitcast <4 x i32> %1277 to <8 x half>, !dbg !15
  %1279 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local341, i32 0) #0, !dbg !15
  %1280 = bitcast <4 x i32> %1279 to <8 x half>, !dbg !15
  %1281 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local345, i32 0) #0, !dbg !15
  %1282 = bitcast <4 x i32> %1281 to <8 x half>, !dbg !15
  %1283 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !15
  %1284 = bitcast <4 x i32> %1283 to <8 x half>, !dbg !15
  %1285 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local351, i32 0) #0, !dbg !15
  %1286 = bitcast <4 x i32> %1285 to <8 x half>, !dbg !15
  %1287 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local355, i32 0) #0, !dbg !15
  %1288 = bitcast <4 x i32> %1287 to <8 x half>, !dbg !15
  %1289 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !15
  %1290 = bitcast <4 x i32> %1289 to <8 x half>, !dbg !15
  %1291 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !15
  %1292 = bitcast <4 x i32> %1291 to <8 x half>, !dbg !15
  %1293 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1294 = extractvalue [2 x <8 x float>] %32, 0, !dbg !34
  %1295 = shufflevector <8 x float> %1294, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1296 = fmul <4 x float> %1295, %657, !dbg !34
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1293, <4 x float> %1296) #9, !dbg !15
  %1298 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1298, <4 x float> %1297) #9, !dbg !15
  %1300 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1300, <4 x float> %1299) #9, !dbg !15
  %1302 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1302, <4 x float> %1301) #9, !dbg !15
  %1304 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1304, <4 x float> %1303) #9, !dbg !15
  %1306 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1306, <4 x float> %1305) #9, !dbg !15
  %1308 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1308, <4 x float> %1307) #9, !dbg !15
  %1310 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1311 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1310, <4 x float> %1309) #9, !dbg !15
  %1312 = shufflevector <8 x half> %1278, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1313 = shufflevector <8 x float> %1294, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1314 = fmul <4 x float> %1313, %657, !dbg !34
  %1315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1312, <4 x float> %1314) #9, !dbg !15
  %1316 = shufflevector <8 x half> %1280, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed101, <4 x half> %1316, <4 x float> %1315) #9, !dbg !15
  %1318 = shufflevector <8 x half> %1282, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed103, <4 x half> %1318, <4 x float> %1317) #9, !dbg !15
  %1320 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed105, <4 x half> %1320, <4 x float> %1319) #9, !dbg !15
  %1322 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed107, <4 x half> %1322, <4 x float> %1321) #9, !dbg !15
  %1324 = shufflevector <8 x half> %1288, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed109, <4 x half> %1324, <4 x float> %1323) #9, !dbg !15
  %1326 = shufflevector <8 x half> %1290, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1327 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed111, <4 x half> %1326, <4 x float> %1325) #9, !dbg !15
  %1328 = shufflevector <8 x half> %1292, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed113, <4 x half> %1328, <4 x float> %1327) #9, !dbg !15
  %1330 = extractvalue [2 x <8 x float>] %32, 1, !dbg !34
  %1331 = shufflevector <8 x float> %1330, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1332 = fmul <4 x float> %1331, %665, !dbg !34
  %1333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1293, <4 x float> %1332) #9, !dbg !15
  %1334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1298, <4 x float> %1333) #9, !dbg !15
  %1335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1300, <4 x float> %1334) #9, !dbg !15
  %1336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1302, <4 x float> %1335) #9, !dbg !15
  %1337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1304, <4 x float> %1336) #9, !dbg !15
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1306, <4 x float> %1337) #9, !dbg !15
  %1339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1308, <4 x float> %1338) #9, !dbg !15
  %1340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1310, <4 x float> %1339) #9, !dbg !15
  %1341 = shufflevector <8 x float> %1330, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1342 = fmul <4 x float> %1341, %665, !dbg !34
  %1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed115, <4 x half> %1312, <4 x float> %1342) #9, !dbg !15
  %1344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed117, <4 x half> %1316, <4 x float> %1343) #9, !dbg !15
  %1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed119, <4 x half> %1318, <4 x float> %1344) #9, !dbg !15
  %1346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed121, <4 x half> %1320, <4 x float> %1345) #9, !dbg !15
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed123, <4 x half> %1322, <4 x float> %1346) #9, !dbg !15
  %1348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed125, <4 x half> %1324, <4 x float> %1347) #9, !dbg !15
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed127, <4 x half> %1326, <4 x float> %1348) #9, !dbg !15
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed129, <4 x half> %1328, <4 x float> %1349) #9, !dbg !15
  %1351 = shufflevector <4 x float> %1029, <4 x float> %1047, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1352 = insertvalue [2 x <8 x float>] poison, <8 x float> %1351, 0, !dbg !8
  %1353 = shufflevector <4 x float> %1058, <4 x float> %1068, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1354 = insertvalue [2 x <8 x float>] %1352, <8 x float> %1353, 1, !dbg !8
  %1355 = insertvalue [2 x <1 x float>] poison, <1 x float> %591, 0, !dbg !8
  %1356 = insertvalue [2 x <1 x float>] %1355, <1 x float> %634, 1, !dbg !8
  %1357 = shufflevector <4 x float> %1133, <4 x float> %1151, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1358 = insertvalue [2 x <8 x float>] poison, <8 x float> %1357, 0, !dbg !8
  %1359 = shufflevector <4 x float> %1162, <4 x float> %1172, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1360 = insertvalue [2 x <8 x float>] %1358, <8 x float> %1359, 1, !dbg !8
  %1361 = shufflevector <4 x float> %1237, <4 x float> %1255, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1362 = insertvalue [2 x <8 x float>] poison, <8 x float> %1361, 0, !dbg !8
  %1363 = shufflevector <4 x float> %1266, <4 x float> %1276, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1364 = insertvalue [2 x <8 x float>] %1362, <8 x float> %1363, 1, !dbg !8
  %1365 = shufflevector <4 x float> %1311, <4 x float> %1329, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1366 = insertvalue [2 x <8 x float>] poison, <8 x float> %1365, 0, !dbg !8
  %1367 = shufflevector <4 x float> %1340, <4 x float> %1350, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %1368 = insertvalue [2 x <8 x float>] %1366, <8 x float> %1367, 1, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %1369 = add nuw nsw i64 %26, 1, !dbg !8
  br label %25, !dbg !8

1370:                                             ; preds = %25
  %1371 = and i32 %19, 16, !dbg !8
  %.not2 = icmp eq i32 %1371, 0, !dbg !8
  %1372 = and i32 %19, 32, !dbg !8
  %.not3 = icmp eq i32 %1372, 0, !dbg !8
  %1373 = extractvalue [2 x <8 x float>] %32, 0, !dbg !8
  %1374 = extractelement <8 x float> %1373, i64 0, !dbg !8
  %1375 = extractelement <8 x float> %1373, i64 1, !dbg !8
  %1376 = extractelement <8 x float> %1373, i64 2, !dbg !8
  %1377 = extractelement <8 x float> %1373, i64 3, !dbg !8
  %1378 = select i1 %.not2, float %1375, float %1374, !dbg !8
  %1379 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %1380 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1379), !dbg !8
  %1381 = add nuw nsw i32 %1380, 64, !dbg !8
  %1382 = xor i32 %1380, 16, !dbg !8
  %1383 = and i32 %1381, 192, !dbg !8
  %1384 = icmp samesign ult i32 %1382, %1383, !dbg !8
  %1385 = select i1 %1384, i32 %1382, i32 %1380, !dbg !8
  %1386 = shl nuw nsw i32 %1385, 2, !dbg !8
  %1387 = bitcast float %1378 to i32, !dbg !8
  %1388 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1387), !dbg !8
  %1389 = bitcast i32 %1388 to float, !dbg !8
  %1390 = select i1 %.not2, float %1374, float %1389, !dbg !8
  %1391 = select i1 %.not2, float %1389, float %1375, !dbg !8
  %1392 = select i1 %.not2, float %1377, float %1376, !dbg !8
  %1393 = bitcast float %1392 to i32, !dbg !8
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1393), !dbg !8
  %1395 = bitcast i32 %1394 to float, !dbg !8
  %1396 = select i1 %.not2, float %1376, float %1395, !dbg !8
  %1397 = select i1 %.not2, float %1395, float %1377, !dbg !8
  %1398 = select i1 %.not3, float %1396, float %1390, !dbg !8
  %1399 = xor i32 %1380, 32, !dbg !8
  %1400 = icmp samesign ult i32 %1399, %1383, !dbg !8
  %1401 = select i1 %1400, i32 %1399, i32 %1380, !dbg !8
  %1402 = shl nuw nsw i32 %1401, 2, !dbg !8
  %1403 = bitcast float %1398 to i32, !dbg !8
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1403), !dbg !8
  %1405 = bitcast i32 %1404 to float, !dbg !8
  %1406 = select i1 %.not3, float %1390, float %1405, !dbg !8
  %1407 = select i1 %.not3, float %1405, float %1396, !dbg !8
  %1408 = select i1 %.not3, float %1397, float %1391, !dbg !8
  %1409 = bitcast float %1408 to i32, !dbg !8
  %1410 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1409), !dbg !8
  %1411 = bitcast i32 %1410 to float, !dbg !8
  %1412 = select i1 %.not3, float %1391, float %1411, !dbg !8
  %1413 = select i1 %.not3, float %1411, float %1397, !dbg !8
  %1414 = extractvalue [2 x <8 x float>] %31, 0, !dbg !8
  %1415 = extractelement <8 x float> %1414, i64 0, !dbg !8
  %1416 = extractelement <8 x float> %1414, i64 1, !dbg !8
  %1417 = extractelement <8 x float> %1414, i64 2, !dbg !8
  %1418 = extractelement <8 x float> %1414, i64 3, !dbg !8
  %1419 = select i1 %.not2, float %1416, float %1415, !dbg !8
  %1420 = bitcast float %1419 to i32, !dbg !8
  %1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1420), !dbg !8
  %1422 = bitcast i32 %1421 to float, !dbg !8
  %1423 = select i1 %.not2, float %1415, float %1422, !dbg !8
  %1424 = select i1 %.not2, float %1422, float %1416, !dbg !8
  %1425 = select i1 %.not2, float %1418, float %1417, !dbg !8
  %1426 = bitcast float %1425 to i32, !dbg !8
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1426), !dbg !8
  %1428 = bitcast i32 %1427 to float, !dbg !8
  %1429 = select i1 %.not2, float %1417, float %1428, !dbg !8
  %1430 = select i1 %.not2, float %1428, float %1418, !dbg !8
  %1431 = select i1 %.not3, float %1429, float %1423, !dbg !8
  %1432 = bitcast float %1431 to i32, !dbg !8
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1432), !dbg !8
  %1434 = bitcast i32 %1433 to float, !dbg !8
  %1435 = select i1 %.not3, float %1423, float %1434, !dbg !8
  %1436 = select i1 %.not3, float %1434, float %1429, !dbg !8
  %1437 = select i1 %.not3, float %1430, float %1424, !dbg !8
  %1438 = bitcast float %1437 to i32, !dbg !8
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1438), !dbg !8
  %1440 = bitcast i32 %1439 to float, !dbg !8
  %1441 = select i1 %.not3, float %1424, float %1440, !dbg !8
  %1442 = select i1 %.not3, float %1440, float %1430, !dbg !8
  %1443 = extractvalue [2 x <8 x float>] %30, 0, !dbg !8
  %1444 = extractelement <8 x float> %1443, i64 0, !dbg !8
  %1445 = extractelement <8 x float> %1443, i64 1, !dbg !8
  %1446 = extractelement <8 x float> %1443, i64 2, !dbg !8
  %1447 = extractelement <8 x float> %1443, i64 3, !dbg !8
  %1448 = select i1 %.not2, float %1445, float %1444, !dbg !8
  %1449 = bitcast float %1448 to i32, !dbg !8
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1449), !dbg !8
  %1451 = bitcast i32 %1450 to float, !dbg !8
  %1452 = select i1 %.not2, float %1444, float %1451, !dbg !8
  %1453 = select i1 %.not2, float %1451, float %1445, !dbg !8
  %1454 = select i1 %.not2, float %1447, float %1446, !dbg !8
  %1455 = bitcast float %1454 to i32, !dbg !8
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1455), !dbg !8
  %1457 = bitcast i32 %1456 to float, !dbg !8
  %1458 = select i1 %.not2, float %1446, float %1457, !dbg !8
  %1459 = select i1 %.not2, float %1457, float %1447, !dbg !8
  %1460 = select i1 %.not3, float %1458, float %1452, !dbg !8
  %1461 = bitcast float %1460 to i32, !dbg !8
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1461), !dbg !8
  %1463 = bitcast i32 %1462 to float, !dbg !8
  %1464 = select i1 %.not3, float %1452, float %1463, !dbg !8
  %1465 = select i1 %.not3, float %1463, float %1458, !dbg !8
  %1466 = select i1 %.not3, float %1459, float %1453, !dbg !8
  %1467 = bitcast float %1466 to i32, !dbg !8
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1467), !dbg !8
  %1469 = bitcast i32 %1468 to float, !dbg !8
  %1470 = select i1 %.not3, float %1453, float %1469, !dbg !8
  %1471 = select i1 %.not3, float %1469, float %1459, !dbg !8
  %1472 = extractvalue [2 x <8 x float>] %27, 0, !dbg !8
  %1473 = extractelement <8 x float> %1472, i64 0, !dbg !8
  %1474 = extractelement <8 x float> %1472, i64 1, !dbg !8
  %1475 = extractelement <8 x float> %1472, i64 2, !dbg !8
  %1476 = extractelement <8 x float> %1472, i64 3, !dbg !8
  %1477 = select i1 %.not2, float %1474, float %1473, !dbg !8
  %1478 = bitcast float %1477 to i32, !dbg !8
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1478), !dbg !8
  %1480 = bitcast i32 %1479 to float, !dbg !8
  %1481 = select i1 %.not2, float %1473, float %1480, !dbg !8
  %1482 = select i1 %.not2, float %1480, float %1474, !dbg !8
  %1483 = select i1 %.not2, float %1476, float %1475, !dbg !8
  %1484 = bitcast float %1483 to i32, !dbg !8
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1484), !dbg !8
  %1486 = bitcast i32 %1485 to float, !dbg !8
  %1487 = select i1 %.not2, float %1475, float %1486, !dbg !8
  %1488 = select i1 %.not2, float %1486, float %1476, !dbg !8
  %1489 = select i1 %.not3, float %1487, float %1481, !dbg !8
  %1490 = bitcast float %1489 to i32, !dbg !8
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1490), !dbg !8
  %1492 = select i1 %.not3, float %1488, float %1482, !dbg !8
  %1493 = bitcast float %1492 to i32, !dbg !8
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1493), !dbg !8
  %1495 = extractvalue [2 x <1 x float>] %28, 0, !dbg !18
  %1496 = shufflevector <1 x float> %1495, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1497 = extractelement <8 x float> %1373, i64 4, !dbg !8
  %1498 = extractelement <8 x float> %1373, i64 5, !dbg !8
  %1499 = extractelement <8 x float> %1373, i64 6, !dbg !8
  %1500 = extractelement <8 x float> %1373, i64 7, !dbg !8
  %1501 = select i1 %.not2, float %1498, float %1497, !dbg !8
  %1502 = bitcast float %1501 to i32, !dbg !8
  %1503 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1502), !dbg !8
  %1504 = bitcast i32 %1503 to float, !dbg !8
  %1505 = select i1 %.not2, float %1497, float %1504, !dbg !8
  %1506 = select i1 %.not2, float %1504, float %1498, !dbg !8
  %1507 = select i1 %.not2, float %1500, float %1499, !dbg !8
  %1508 = bitcast float %1507 to i32, !dbg !8
  %1509 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1508), !dbg !8
  %1510 = bitcast i32 %1509 to float, !dbg !8
  %1511 = select i1 %.not2, float %1499, float %1510, !dbg !8
  %1512 = select i1 %.not2, float %1510, float %1500, !dbg !8
  %1513 = select i1 %.not3, float %1511, float %1505, !dbg !8
  %1514 = bitcast float %1513 to i32, !dbg !8
  %1515 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1514), !dbg !8
  %1516 = bitcast i32 %1515 to float, !dbg !8
  %1517 = select i1 %.not3, float %1505, float %1516, !dbg !8
  %1518 = select i1 %.not3, float %1516, float %1511, !dbg !8
  %1519 = select i1 %.not3, float %1512, float %1506, !dbg !8
  %1520 = bitcast float %1519 to i32, !dbg !8
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1520), !dbg !8
  %1522 = bitcast i32 %1521 to float, !dbg !8
  %1523 = select i1 %.not3, float %1506, float %1522, !dbg !8
  %1524 = select i1 %.not3, float %1522, float %1512, !dbg !8
  %1525 = extractelement <8 x float> %1414, i64 4, !dbg !8
  %1526 = extractelement <8 x float> %1414, i64 5, !dbg !8
  %1527 = extractelement <8 x float> %1414, i64 6, !dbg !8
  %1528 = extractelement <8 x float> %1414, i64 7, !dbg !8
  %1529 = select i1 %.not2, float %1526, float %1525, !dbg !8
  %1530 = bitcast float %1529 to i32, !dbg !8
  %1531 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1530), !dbg !8
  %1532 = bitcast i32 %1531 to float, !dbg !8
  %1533 = select i1 %.not2, float %1525, float %1532, !dbg !8
  %1534 = select i1 %.not2, float %1532, float %1526, !dbg !8
  %1535 = select i1 %.not2, float %1528, float %1527, !dbg !8
  %1536 = bitcast float %1535 to i32, !dbg !8
  %1537 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1536), !dbg !8
  %1538 = bitcast i32 %1537 to float, !dbg !8
  %1539 = select i1 %.not2, float %1527, float %1538, !dbg !8
  %1540 = select i1 %.not2, float %1538, float %1528, !dbg !8
  %1541 = select i1 %.not3, float %1539, float %1533, !dbg !8
  %1542 = bitcast float %1541 to i32, !dbg !8
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1542), !dbg !8
  %1544 = bitcast i32 %1543 to float, !dbg !8
  %1545 = select i1 %.not3, float %1533, float %1544, !dbg !8
  %1546 = select i1 %.not3, float %1544, float %1539, !dbg !8
  %1547 = select i1 %.not3, float %1540, float %1534, !dbg !8
  %1548 = bitcast float %1547 to i32, !dbg !8
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1548), !dbg !8
  %1550 = bitcast i32 %1549 to float, !dbg !8
  %1551 = select i1 %.not3, float %1534, float %1550, !dbg !8
  %1552 = select i1 %.not3, float %1550, float %1540, !dbg !8
  %1553 = extractelement <8 x float> %1443, i64 4, !dbg !8
  %1554 = extractelement <8 x float> %1443, i64 5, !dbg !8
  %1555 = extractelement <8 x float> %1443, i64 6, !dbg !8
  %1556 = extractelement <8 x float> %1443, i64 7, !dbg !8
  %1557 = select i1 %.not2, float %1554, float %1553, !dbg !8
  %1558 = bitcast float %1557 to i32, !dbg !8
  %1559 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1558), !dbg !8
  %1560 = bitcast i32 %1559 to float, !dbg !8
  %1561 = select i1 %.not2, float %1553, float %1560, !dbg !8
  %1562 = select i1 %.not2, float %1560, float %1554, !dbg !8
  %1563 = select i1 %.not2, float %1556, float %1555, !dbg !8
  %1564 = bitcast float %1563 to i32, !dbg !8
  %1565 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1564), !dbg !8
  %1566 = bitcast i32 %1565 to float, !dbg !8
  %1567 = select i1 %.not2, float %1555, float %1566, !dbg !8
  %1568 = select i1 %.not2, float %1566, float %1556, !dbg !8
  %1569 = select i1 %.not3, float %1567, float %1561, !dbg !8
  %1570 = bitcast float %1569 to i32, !dbg !8
  %1571 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1570), !dbg !8
  %1572 = bitcast i32 %1571 to float, !dbg !8
  %1573 = select i1 %.not3, float %1561, float %1572, !dbg !8
  %1574 = select i1 %.not3, float %1572, float %1567, !dbg !8
  %1575 = select i1 %.not3, float %1568, float %1562, !dbg !8
  %1576 = bitcast float %1575 to i32, !dbg !8
  %1577 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1576), !dbg !8
  %1578 = bitcast i32 %1577 to float, !dbg !8
  %1579 = select i1 %.not3, float %1562, float %1578, !dbg !8
  %1580 = select i1 %.not3, float %1578, float %1568, !dbg !8
  %1581 = extractelement <8 x float> %1472, i64 4, !dbg !8
  %1582 = extractelement <8 x float> %1472, i64 5, !dbg !8
  %1583 = extractelement <8 x float> %1472, i64 6, !dbg !8
  %1584 = extractelement <8 x float> %1472, i64 7, !dbg !8
  %1585 = select i1 %.not2, float %1582, float %1581, !dbg !8
  %1586 = bitcast float %1585 to i32, !dbg !8
  %1587 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1586), !dbg !8
  %1588 = bitcast i32 %1587 to float, !dbg !8
  %1589 = select i1 %.not2, float %1581, float %1588, !dbg !8
  %1590 = select i1 %.not2, float %1588, float %1582, !dbg !8
  %1591 = select i1 %.not2, float %1584, float %1583, !dbg !8
  %1592 = bitcast float %1591 to i32, !dbg !8
  %1593 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1592), !dbg !8
  %1594 = bitcast i32 %1593 to float, !dbg !8
  %1595 = select i1 %.not2, float %1583, float %1594, !dbg !8
  %1596 = select i1 %.not2, float %1594, float %1584, !dbg !8
  %1597 = select i1 %.not3, float %1595, float %1589, !dbg !8
  %1598 = bitcast float %1597 to i32, !dbg !8
  %1599 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1598), !dbg !8
  %1600 = select i1 %.not3, float %1596, float %1590, !dbg !8
  %1601 = bitcast float %1600 to i32, !dbg !8
  %1602 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1601), !dbg !8
  %1603 = extractvalue [2 x <8 x float>] %32, 1, !dbg !8
  %1604 = extractelement <8 x float> %1603, i64 0, !dbg !8
  %1605 = extractelement <8 x float> %1603, i64 1, !dbg !8
  %1606 = extractelement <8 x float> %1603, i64 2, !dbg !8
  %1607 = extractelement <8 x float> %1603, i64 3, !dbg !8
  %1608 = select i1 %.not2, float %1605, float %1604, !dbg !8
  %1609 = bitcast float %1608 to i32, !dbg !8
  %1610 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1609), !dbg !8
  %1611 = bitcast i32 %1610 to float, !dbg !8
  %1612 = select i1 %.not2, float %1604, float %1611, !dbg !8
  %1613 = select i1 %.not2, float %1611, float %1605, !dbg !8
  %1614 = select i1 %.not2, float %1607, float %1606, !dbg !8
  %1615 = bitcast float %1614 to i32, !dbg !8
  %1616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1615), !dbg !8
  %1617 = bitcast i32 %1616 to float, !dbg !8
  %1618 = select i1 %.not2, float %1606, float %1617, !dbg !8
  %1619 = select i1 %.not2, float %1617, float %1607, !dbg !8
  %1620 = select i1 %.not3, float %1618, float %1612, !dbg !8
  %1621 = bitcast float %1620 to i32, !dbg !8
  %1622 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1621), !dbg !8
  %1623 = bitcast i32 %1622 to float, !dbg !8
  %1624 = select i1 %.not3, float %1612, float %1623, !dbg !8
  %1625 = select i1 %.not3, float %1623, float %1618, !dbg !8
  %1626 = select i1 %.not3, float %1619, float %1613, !dbg !8
  %1627 = bitcast float %1626 to i32, !dbg !8
  %1628 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1627), !dbg !8
  %1629 = bitcast i32 %1628 to float, !dbg !8
  %1630 = select i1 %.not3, float %1613, float %1629, !dbg !8
  %1631 = select i1 %.not3, float %1629, float %1619, !dbg !8
  %1632 = extractvalue [2 x <8 x float>] %31, 1, !dbg !8
  %1633 = extractelement <8 x float> %1632, i64 0, !dbg !8
  %1634 = extractelement <8 x float> %1632, i64 1, !dbg !8
  %1635 = extractelement <8 x float> %1632, i64 2, !dbg !8
  %1636 = extractelement <8 x float> %1632, i64 3, !dbg !8
  %1637 = select i1 %.not2, float %1634, float %1633, !dbg !8
  %1638 = bitcast float %1637 to i32, !dbg !8
  %1639 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1638), !dbg !8
  %1640 = bitcast i32 %1639 to float, !dbg !8
  %1641 = select i1 %.not2, float %1633, float %1640, !dbg !8
  %1642 = select i1 %.not2, float %1640, float %1634, !dbg !8
  %1643 = select i1 %.not2, float %1636, float %1635, !dbg !8
  %1644 = bitcast float %1643 to i32, !dbg !8
  %1645 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1644), !dbg !8
  %1646 = bitcast i32 %1645 to float, !dbg !8
  %1647 = select i1 %.not2, float %1635, float %1646, !dbg !8
  %1648 = select i1 %.not2, float %1646, float %1636, !dbg !8
  %1649 = select i1 %.not3, float %1647, float %1641, !dbg !8
  %1650 = bitcast float %1649 to i32, !dbg !8
  %1651 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1650), !dbg !8
  %1652 = bitcast i32 %1651 to float, !dbg !8
  %1653 = select i1 %.not3, float %1641, float %1652, !dbg !8
  %1654 = select i1 %.not3, float %1652, float %1647, !dbg !8
  %1655 = select i1 %.not3, float %1648, float %1642, !dbg !8
  %1656 = bitcast float %1655 to i32, !dbg !8
  %1657 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1656), !dbg !8
  %1658 = bitcast i32 %1657 to float, !dbg !8
  %1659 = select i1 %.not3, float %1642, float %1658, !dbg !8
  %1660 = select i1 %.not3, float %1658, float %1648, !dbg !8
  %1661 = extractvalue [2 x <8 x float>] %30, 1, !dbg !8
  %1662 = extractelement <8 x float> %1661, i64 0, !dbg !8
  %1663 = extractelement <8 x float> %1661, i64 1, !dbg !8
  %1664 = extractelement <8 x float> %1661, i64 2, !dbg !8
  %1665 = extractelement <8 x float> %1661, i64 3, !dbg !8
  %1666 = select i1 %.not2, float %1663, float %1662, !dbg !8
  %1667 = bitcast float %1666 to i32, !dbg !8
  %1668 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1667), !dbg !8
  %1669 = bitcast i32 %1668 to float, !dbg !8
  %1670 = select i1 %.not2, float %1662, float %1669, !dbg !8
  %1671 = select i1 %.not2, float %1669, float %1663, !dbg !8
  %1672 = select i1 %.not2, float %1665, float %1664, !dbg !8
  %1673 = bitcast float %1672 to i32, !dbg !8
  %1674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1673), !dbg !8
  %1675 = bitcast i32 %1674 to float, !dbg !8
  %1676 = select i1 %.not2, float %1664, float %1675, !dbg !8
  %1677 = select i1 %.not2, float %1675, float %1665, !dbg !8
  %1678 = select i1 %.not3, float %1676, float %1670, !dbg !8
  %1679 = bitcast float %1678 to i32, !dbg !8
  %1680 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1679), !dbg !8
  %1681 = bitcast i32 %1680 to float, !dbg !8
  %1682 = select i1 %.not3, float %1670, float %1681, !dbg !8
  %1683 = select i1 %.not3, float %1681, float %1676, !dbg !8
  %1684 = select i1 %.not3, float %1677, float %1671, !dbg !8
  %1685 = bitcast float %1684 to i32, !dbg !8
  %1686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1685), !dbg !8
  %1687 = bitcast i32 %1686 to float, !dbg !8
  %1688 = select i1 %.not3, float %1671, float %1687, !dbg !8
  %1689 = select i1 %.not3, float %1687, float %1677, !dbg !8
  %1690 = extractvalue [2 x <8 x float>] %27, 1, !dbg !8
  %1691 = extractelement <8 x float> %1690, i64 0, !dbg !8
  %1692 = extractelement <8 x float> %1690, i64 1, !dbg !8
  %1693 = extractelement <8 x float> %1690, i64 2, !dbg !8
  %1694 = extractelement <8 x float> %1690, i64 3, !dbg !8
  %1695 = select i1 %.not2, float %1692, float %1691, !dbg !8
  %1696 = bitcast float %1695 to i32, !dbg !8
  %1697 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1696), !dbg !8
  %1698 = bitcast i32 %1697 to float, !dbg !8
  %1699 = select i1 %.not2, float %1691, float %1698, !dbg !8
  %1700 = select i1 %.not2, float %1698, float %1692, !dbg !8
  %1701 = select i1 %.not2, float %1694, float %1693, !dbg !8
  %1702 = bitcast float %1701 to i32, !dbg !8
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1702), !dbg !8
  %1704 = bitcast i32 %1703 to float, !dbg !8
  %1705 = select i1 %.not2, float %1693, float %1704, !dbg !8
  %1706 = select i1 %.not2, float %1704, float %1694, !dbg !8
  %1707 = select i1 %.not3, float %1705, float %1699, !dbg !8
  %1708 = bitcast float %1707 to i32, !dbg !8
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1708), !dbg !8
  %1710 = select i1 %.not3, float %1706, float %1700, !dbg !8
  %1711 = bitcast float %1710 to i32, !dbg !8
  %1712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1711), !dbg !8
  %1713 = extractvalue [2 x <1 x float>] %28, 1, !dbg !18
  %1714 = shufflevector <1 x float> %1713, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1715 = extractelement <8 x float> %1603, i64 4, !dbg !8
  %1716 = extractelement <8 x float> %1603, i64 5, !dbg !8
  %1717 = extractelement <8 x float> %1603, i64 6, !dbg !8
  %1718 = extractelement <8 x float> %1603, i64 7, !dbg !8
  %1719 = select i1 %.not2, float %1716, float %1715, !dbg !8
  %1720 = bitcast float %1719 to i32, !dbg !8
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1720), !dbg !8
  %1722 = bitcast i32 %1721 to float, !dbg !8
  %1723 = select i1 %.not2, float %1715, float %1722, !dbg !8
  %1724 = select i1 %.not2, float %1722, float %1716, !dbg !8
  %1725 = select i1 %.not2, float %1718, float %1717, !dbg !8
  %1726 = bitcast float %1725 to i32, !dbg !8
  %1727 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1726), !dbg !8
  %1728 = bitcast i32 %1727 to float, !dbg !8
  %1729 = select i1 %.not2, float %1717, float %1728, !dbg !8
  %1730 = select i1 %.not2, float %1728, float %1718, !dbg !8
  %1731 = select i1 %.not3, float %1729, float %1723, !dbg !8
  %1732 = bitcast float %1731 to i32, !dbg !8
  %1733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1732), !dbg !8
  %1734 = bitcast i32 %1733 to float, !dbg !8
  %1735 = select i1 %.not3, float %1723, float %1734, !dbg !8
  %1736 = select i1 %.not3, float %1734, float %1729, !dbg !8
  %1737 = select i1 %.not3, float %1730, float %1724, !dbg !8
  %1738 = bitcast float %1737 to i32, !dbg !8
  %1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1738), !dbg !8
  %1740 = bitcast i32 %1739 to float, !dbg !8
  %1741 = select i1 %.not3, float %1724, float %1740, !dbg !8
  %1742 = select i1 %.not3, float %1740, float %1730, !dbg !8
  %1743 = extractelement <8 x float> %1632, i64 4, !dbg !8
  %1744 = extractelement <8 x float> %1632, i64 5, !dbg !8
  %1745 = extractelement <8 x float> %1632, i64 6, !dbg !8
  %1746 = extractelement <8 x float> %1632, i64 7, !dbg !8
  %1747 = select i1 %.not2, float %1744, float %1743, !dbg !8
  %1748 = bitcast float %1747 to i32, !dbg !8
  %1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1748), !dbg !8
  %1750 = bitcast i32 %1749 to float, !dbg !8
  %1751 = select i1 %.not2, float %1743, float %1750, !dbg !8
  %1752 = select i1 %.not2, float %1750, float %1744, !dbg !8
  %1753 = select i1 %.not2, float %1746, float %1745, !dbg !8
  %1754 = bitcast float %1753 to i32, !dbg !8
  %1755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1754), !dbg !8
  %1756 = bitcast i32 %1755 to float, !dbg !8
  %1757 = select i1 %.not2, float %1745, float %1756, !dbg !8
  %1758 = select i1 %.not2, float %1756, float %1746, !dbg !8
  %1759 = select i1 %.not3, float %1757, float %1751, !dbg !8
  %1760 = bitcast float %1759 to i32, !dbg !8
  %1761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1760), !dbg !8
  %1762 = bitcast i32 %1761 to float, !dbg !8
  %1763 = select i1 %.not3, float %1751, float %1762, !dbg !8
  %1764 = select i1 %.not3, float %1762, float %1757, !dbg !8
  %1765 = select i1 %.not3, float %1758, float %1752, !dbg !8
  %1766 = bitcast float %1765 to i32, !dbg !8
  %1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1766), !dbg !8
  %1768 = bitcast i32 %1767 to float, !dbg !8
  %1769 = select i1 %.not3, float %1752, float %1768, !dbg !8
  %1770 = select i1 %.not3, float %1768, float %1758, !dbg !8
  %1771 = extractelement <8 x float> %1661, i64 4, !dbg !8
  %1772 = extractelement <8 x float> %1661, i64 5, !dbg !8
  %1773 = extractelement <8 x float> %1661, i64 6, !dbg !8
  %1774 = extractelement <8 x float> %1661, i64 7, !dbg !8
  %1775 = select i1 %.not2, float %1772, float %1771, !dbg !8
  %1776 = bitcast float %1775 to i32, !dbg !8
  %1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1776), !dbg !8
  %1778 = bitcast i32 %1777 to float, !dbg !8
  %1779 = select i1 %.not2, float %1771, float %1778, !dbg !8
  %1780 = select i1 %.not2, float %1778, float %1772, !dbg !8
  %1781 = select i1 %.not2, float %1774, float %1773, !dbg !8
  %1782 = bitcast float %1781 to i32, !dbg !8
  %1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1782), !dbg !8
  %1784 = bitcast i32 %1783 to float, !dbg !8
  %1785 = select i1 %.not2, float %1773, float %1784, !dbg !8
  %1786 = select i1 %.not2, float %1784, float %1774, !dbg !8
  %1787 = select i1 %.not3, float %1785, float %1779, !dbg !8
  %1788 = bitcast float %1787 to i32, !dbg !8
  %1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1788), !dbg !8
  %1790 = bitcast i32 %1789 to float, !dbg !8
  %1791 = select i1 %.not3, float %1779, float %1790, !dbg !8
  %1792 = select i1 %.not3, float %1790, float %1785, !dbg !8
  %1793 = select i1 %.not3, float %1786, float %1780, !dbg !8
  %1794 = bitcast float %1793 to i32, !dbg !8
  %1795 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1794), !dbg !8
  %1796 = bitcast i32 %1795 to float, !dbg !8
  %1797 = select i1 %.not3, float %1780, float %1796, !dbg !8
  %1798 = select i1 %.not3, float %1796, float %1786, !dbg !8
  %1799 = extractelement <8 x float> %1690, i64 4, !dbg !8
  %1800 = extractelement <8 x float> %1690, i64 5, !dbg !8
  %1801 = extractelement <8 x float> %1690, i64 6, !dbg !8
  %1802 = extractelement <8 x float> %1690, i64 7, !dbg !8
  %1803 = select i1 %.not2, float %1800, float %1799, !dbg !8
  %1804 = bitcast float %1803 to i32, !dbg !8
  %1805 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1804), !dbg !8
  %1806 = bitcast i32 %1805 to float, !dbg !8
  %1807 = select i1 %.not2, float %1799, float %1806, !dbg !8
  %1808 = select i1 %.not2, float %1806, float %1800, !dbg !8
  %1809 = select i1 %.not2, float %1802, float %1801, !dbg !8
  %1810 = bitcast float %1809 to i32, !dbg !8
  %1811 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1386, i32 %1810), !dbg !8
  %1812 = bitcast i32 %1811 to float, !dbg !8
  %1813 = select i1 %.not2, float %1801, float %1812, !dbg !8
  %1814 = select i1 %.not2, float %1812, float %1802, !dbg !8
  %1815 = select i1 %.not3, float %1813, float %1807, !dbg !8
  %1816 = bitcast float %1815 to i32, !dbg !8
  %1817 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1816), !dbg !8
  %1818 = select i1 %.not3, float %1814, float %1808, !dbg !8
  %1819 = bitcast float %1818 to i32, !dbg !8
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1402, i32 %1819), !dbg !8
  br i1 true, label %1821, label %1885, !dbg !20

1821:                                             ; preds = %1370
  %1822 = bitcast i32 %1817 to float, !dbg !8
  %1823 = select i1 %.not3, float %1807, float %1822, !dbg !8
  %1824 = insertelement <4 x float> poison, float %1823, i64 0, !dbg !18
  %1825 = bitcast i32 %1820 to float, !dbg !8
  %1826 = select i1 %.not3, float %1808, float %1825, !dbg !8
  %1827 = insertelement <4 x float> %1824, float %1826, i64 1, !dbg !18
  %1828 = select i1 %.not3, float %1822, float %1813, !dbg !8
  %1829 = insertelement <4 x float> %1827, float %1828, i64 2, !dbg !18
  %1830 = select i1 %.not3, float %1825, float %1814, !dbg !8
  %1831 = insertelement <4 x float> %1829, float %1830, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1714, !dbg !18
  %normalized = fmul <4 x float> %1831, %row.reciprocal, !dbg !18
  %1832 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1833 = extractelement <4 x half> %1832, i64 3, !dbg !19
  %1834 = extractelement <4 x half> %1832, i64 2, !dbg !19
  %1835 = extractelement <4 x half> %1832, i64 1, !dbg !19
  %1836 = extractelement <4 x half> %1832, i64 0, !dbg !19
  %1837 = bitcast i32 %1709 to float, !dbg !8
  %1838 = select i1 %.not3, float %1699, float %1837, !dbg !8
  %1839 = insertelement <4 x float> poison, float %1838, i64 0, !dbg !18
  %1840 = bitcast i32 %1712 to float, !dbg !8
  %1841 = select i1 %.not3, float %1700, float %1840, !dbg !8
  %1842 = insertelement <4 x float> %1839, float %1841, i64 1, !dbg !18
  %1843 = select i1 %.not3, float %1837, float %1705, !dbg !8
  %1844 = insertelement <4 x float> %1842, float %1843, i64 2, !dbg !18
  %1845 = select i1 %.not3, float %1840, float %1706, !dbg !8
  %1846 = insertelement <4 x float> %1844, float %1845, i64 3, !dbg !18
  %normalized72 = fmul <4 x float> %1846, %row.reciprocal, !dbg !18
  %1847 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !19
  %1848 = extractelement <4 x half> %1847, i64 3, !dbg !19
  %1849 = extractelement <4 x half> %1847, i64 2, !dbg !19
  %1850 = extractelement <4 x half> %1847, i64 1, !dbg !19
  %1851 = extractelement <4 x half> %1847, i64 0, !dbg !19
  %1852 = bitcast i32 %1599 to float, !dbg !8
  %1853 = select i1 %.not3, float %1589, float %1852, !dbg !8
  %1854 = insertelement <4 x float> poison, float %1853, i64 0, !dbg !18
  %1855 = bitcast i32 %1602 to float, !dbg !8
  %1856 = select i1 %.not3, float %1590, float %1855, !dbg !8
  %1857 = insertelement <4 x float> %1854, float %1856, i64 1, !dbg !18
  %1858 = select i1 %.not3, float %1852, float %1595, !dbg !8
  %1859 = insertelement <4 x float> %1857, float %1858, i64 2, !dbg !18
  %1860 = select i1 %.not3, float %1855, float %1596, !dbg !8
  %1861 = insertelement <4 x float> %1859, float %1860, i64 3, !dbg !18
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1496, !dbg !18
  %normalized74 = fmul <4 x float> %1861, %row.reciprocal73, !dbg !18
  %1862 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !19
  %1863 = extractelement <4 x half> %1862, i64 3, !dbg !19
  %1864 = extractelement <4 x half> %1862, i64 2, !dbg !19
  %1865 = extractelement <4 x half> %1862, i64 1, !dbg !19
  %1866 = extractelement <4 x half> %1862, i64 0, !dbg !19
  %1867 = bitcast i32 %1491 to float, !dbg !8
  %1868 = select i1 %.not3, float %1481, float %1867, !dbg !8
  %1869 = insertelement <4 x float> poison, float %1868, i64 0, !dbg !18
  %1870 = bitcast i32 %1494 to float, !dbg !8
  %1871 = select i1 %.not3, float %1482, float %1870, !dbg !8
  %1872 = insertelement <4 x float> %1869, float %1871, i64 1, !dbg !18
  %1873 = select i1 %.not3, float %1867, float %1487, !dbg !8
  %1874 = insertelement <4 x float> %1872, float %1873, i64 2, !dbg !18
  %1875 = select i1 %.not3, float %1870, float %1488, !dbg !8
  %1876 = insertelement <4 x float> %1874, float %1875, i64 3, !dbg !18
  %normalized75 = fmul <4 x float> %1876, %row.reciprocal73, !dbg !18
  %1877 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !19
  %1878 = extractelement <4 x half> %1877, i64 3, !dbg !19
  %1879 = extractelement <4 x half> %1877, i64 2, !dbg !19
  %1880 = extractelement <4 x half> %1877, i64 1, !dbg !19
  %1881 = extractelement <4 x half> %1877, i64 0, !dbg !19
  %1882 = add nsw i64 %6, %13, !dbg !20
  %1883 = shl nsw i64 %1882, 19, !dbg !20
  %1884 = and i64 %1883, 16252928, !dbg !20
  %address.thread391 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local392 = zext nneg i32 %address.thread391 to i64, !dbg !13
  %address.local393 = add nsw i64 %16, %address.local392, !dbg !20
  %address.local394 = lshr i64 %address.local392, 6, !dbg !13
  %address.local395 = shl nuw nsw i64 %address.local394, 4, !dbg !20
  %address.local396 = add nsw i64 %address.local393, %address.local395, !dbg !20
  %address.local397 = and i64 %address.local392, 240, !dbg !20
  %address.local398 = sub nsw i64 %address.local396, %address.local397, !dbg !20
  %address.local399 = add nsw i64 %address.local398, %17, !dbg !20
  %address.local400 = shl nuw nsw i64 %address.local399, 7, !dbg !20
  %address.local401 = add nuw nsw i64 %1884, %address.local400, !dbg !20
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local403 = lshr i64 %address.local392, 2, !dbg !13
  %address.local404 = and i64 %address.local403, 12, !dbg !13
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local402, i64 %address.local404, !dbg !20
  store half %1881, ptr addrspace(1) %address.local405, align 2, !dbg !20
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local407 = or disjoint i64 %address.local404, 1, !dbg !20
  %address.local408 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local406, i64 %address.local407, !dbg !20
  store half %1880, ptr addrspace(1) %address.local408, align 2, !dbg !20
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local410 = or disjoint i64 %address.local404, 2, !dbg !20
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local409, i64 %address.local410, !dbg !20
  store half %1879, ptr addrspace(1) %address.local411, align 2, !dbg !20
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local413 = or disjoint i64 %address.local404, 3, !dbg !20
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local412, i64 %address.local413, !dbg !20
  store half %1878, ptr addrspace(1) %address.local414, align 2, !dbg !20
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local416 = or disjoint i64 %address.local404, 16, !dbg !20
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local416, !dbg !20
  store half %1866, ptr addrspace(1) %address.local417, align 2, !dbg !20
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local419 = or disjoint i64 %address.local404, 17, !dbg !20
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local419, !dbg !20
  store half %1865, ptr addrspace(1) %address.local420, align 2, !dbg !20
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local422 = or disjoint i64 %address.local404, 18, !dbg !20
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local421, i64 %address.local422, !dbg !20
  store half %1864, ptr addrspace(1) %address.local423, align 2, !dbg !20
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local401, !dbg !20
  %address.local425 = or disjoint i64 %address.local404, 19, !dbg !20
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local424, i64 %address.local425, !dbg !20
  store half %1863, ptr addrspace(1) %address.local426, align 2, !dbg !20
  %address.local427 = shl nsw i64 %address.local399, 7, !dbg !20
  %address.local428 = add i64 %address.local427, 8192, !dbg !20
  %address.local429 = add nuw nsw i64 %1884, %address.local428, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local404, !dbg !20
  store half %1851, ptr addrspace(1) %address.local431, align 2, !dbg !20
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local407, !dbg !20
  store half %1850, ptr addrspace(1) %address.local433, align 2, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local410, !dbg !20
  store half %1849, ptr addrspace(1) %address.local435, align 2, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local413, !dbg !20
  store half %1848, ptr addrspace(1) %address.local437, align 2, !dbg !20
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local416, !dbg !20
  store half %1836, ptr addrspace(1) %address.local439, align 2, !dbg !20
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local419, !dbg !20
  store half %1835, ptr addrspace(1) %address.local441, align 2, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local422, !dbg !20
  store half %1834, ptr addrspace(1) %address.local443, align 2, !dbg !20
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local429, !dbg !20
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local444, i64 %address.local425, !dbg !20
  store half %1833, ptr addrspace(1) %address.local445, align 2, !dbg !20
  br label %1885, !dbg !20

1885:                                             ; preds = %1821, %1370
  br i1 true, label %1886, label %1924, !dbg !20

1886:                                             ; preds = %1885
  %1887 = insertelement <4 x float> poison, float %1791, i64 0, !dbg !18
  %1888 = insertelement <4 x float> %1887, float %1797, i64 1, !dbg !18
  %1889 = insertelement <4 x float> %1888, float %1792, i64 2, !dbg !18
  %1890 = insertelement <4 x float> %1889, float %1798, i64 3, !dbg !18
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1714, !dbg !18
  %normalized77 = fmul <4 x float> %1890, %row.reciprocal76, !dbg !18
  %1891 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !19
  %1892 = extractelement <4 x half> %1891, i64 3, !dbg !19
  %1893 = extractelement <4 x half> %1891, i64 2, !dbg !19
  %1894 = extractelement <4 x half> %1891, i64 1, !dbg !19
  %1895 = extractelement <4 x half> %1891, i64 0, !dbg !19
  %1896 = insertelement <4 x float> poison, float %1682, i64 0, !dbg !18
  %1897 = insertelement <4 x float> %1896, float %1688, i64 1, !dbg !18
  %1898 = insertelement <4 x float> %1897, float %1683, i64 2, !dbg !18
  %1899 = insertelement <4 x float> %1898, float %1689, i64 3, !dbg !18
  %normalized78 = fmul <4 x float> %1899, %row.reciprocal76, !dbg !18
  %1900 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !19
  %1901 = extractelement <4 x half> %1900, i64 3, !dbg !19
  %1902 = extractelement <4 x half> %1900, i64 2, !dbg !19
  %1903 = extractelement <4 x half> %1900, i64 1, !dbg !19
  %1904 = extractelement <4 x half> %1900, i64 0, !dbg !19
  %1905 = insertelement <4 x float> poison, float %1573, i64 0, !dbg !18
  %1906 = insertelement <4 x float> %1905, float %1579, i64 1, !dbg !18
  %1907 = insertelement <4 x float> %1906, float %1574, i64 2, !dbg !18
  %1908 = insertelement <4 x float> %1907, float %1580, i64 3, !dbg !18
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1496, !dbg !18
  %normalized80 = fmul <4 x float> %1908, %row.reciprocal79, !dbg !18
  %1909 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !19
  %1910 = extractelement <4 x half> %1909, i64 3, !dbg !19
  %1911 = extractelement <4 x half> %1909, i64 2, !dbg !19
  %1912 = extractelement <4 x half> %1909, i64 1, !dbg !19
  %1913 = extractelement <4 x half> %1909, i64 0, !dbg !19
  %1914 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !18
  %1915 = insertelement <4 x float> %1914, float %1470, i64 1, !dbg !18
  %1916 = insertelement <4 x float> %1915, float %1465, i64 2, !dbg !18
  %1917 = insertelement <4 x float> %1916, float %1471, i64 3, !dbg !18
  %normalized81 = fmul <4 x float> %1917, %row.reciprocal79, !dbg !18
  %1918 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !19
  %1919 = extractelement <4 x half> %1918, i64 3, !dbg !19
  %1920 = extractelement <4 x half> %1918, i64 2, !dbg !19
  %1921 = extractelement <4 x half> %1918, i64 1, !dbg !19
  %1922 = extractelement <4 x half> %1918, i64 0, !dbg !19
  %1923 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread446 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local447 = zext nneg i32 %address.thread446 to i64, !dbg !13
  %address.local448 = add nsw i64 %16, %address.local447, !dbg !20
  %address.local449 = lshr i64 %address.local447, 6, !dbg !13
  %address.local450 = shl nuw nsw i64 %address.local449, 4, !dbg !20
  %address.local451 = add nsw i64 %address.local448, %address.local450, !dbg !20
  %address.local452 = and i64 %address.local447, 240, !dbg !20
  %address.local453 = sub nsw i64 %address.local451, %address.local452, !dbg !20
  %address.local454 = add nsw i64 %address.local453, %17, !dbg !20
  %address.local455 = shl nuw nsw i64 %address.local454, 7, !dbg !20
  %address.local456 = add nuw nsw i64 %1923, %address.local455, !dbg !20
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local458 = lshr i64 %address.local447, 2, !dbg !13
  %address.local459 = and i64 %address.local458, 12, !dbg !13
  %address.local460 = or disjoint i64 %address.local459, 32, !dbg !20
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local457, i64 %address.local460, !dbg !20
  store half %1922, ptr addrspace(1) %address.local461, align 2, !dbg !20
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local463 = or disjoint i64 %address.local459, 33, !dbg !20
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local463, !dbg !20
  store half %1921, ptr addrspace(1) %address.local464, align 2, !dbg !20
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local466 = or disjoint i64 %address.local459, 34, !dbg !20
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local466, !dbg !20
  store half %1920, ptr addrspace(1) %address.local467, align 2, !dbg !20
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local469 = or disjoint i64 %address.local459, 35, !dbg !20
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local469, !dbg !20
  store half %1919, ptr addrspace(1) %address.local470, align 2, !dbg !20
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local472 = or disjoint i64 %address.local459, 48, !dbg !20
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local472, !dbg !20
  store half %1913, ptr addrspace(1) %address.local473, align 2, !dbg !20
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local475 = or disjoint i64 %address.local459, 49, !dbg !20
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local474, i64 %address.local475, !dbg !20
  store half %1912, ptr addrspace(1) %address.local476, align 2, !dbg !20
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local478 = or disjoint i64 %address.local459, 50, !dbg !20
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local477, i64 %address.local478, !dbg !20
  store half %1911, ptr addrspace(1) %address.local479, align 2, !dbg !20
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local456, !dbg !20
  %address.local481 = or i64 %address.local458, 51, !dbg !20
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local480, i64 %address.local481, !dbg !20
  store half %1910, ptr addrspace(1) %address.local482, align 2, !dbg !20
  %address.local483 = shl nsw i64 %address.local454, 7, !dbg !20
  %address.local484 = add i64 %address.local483, 8192, !dbg !20
  %address.local485 = add nuw nsw i64 %1923, %address.local484, !dbg !20
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local486, i64 %address.local460, !dbg !20
  store half %1904, ptr addrspace(1) %address.local487, align 2, !dbg !20
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local488, i64 %address.local463, !dbg !20
  store half %1903, ptr addrspace(1) %address.local489, align 2, !dbg !20
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local466, !dbg !20
  store half %1902, ptr addrspace(1) %address.local491, align 2, !dbg !20
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local469, !dbg !20
  store half %1901, ptr addrspace(1) %address.local493, align 2, !dbg !20
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local472, !dbg !20
  store half %1895, ptr addrspace(1) %address.local495, align 2, !dbg !20
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local475, !dbg !20
  store half %1894, ptr addrspace(1) %address.local497, align 2, !dbg !20
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local478, !dbg !20
  store half %1893, ptr addrspace(1) %address.local499, align 2, !dbg !20
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local485, !dbg !20
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local500, i64 %address.local481, !dbg !20
  store half %1892, ptr addrspace(1) %address.local501, align 2, !dbg !20
  br label %1924, !dbg !20

1924:                                             ; preds = %1886, %1885
  br i1 true, label %1925, label %1963, !dbg !20

1925:                                             ; preds = %1924
  %1926 = insertelement <4 x float> poison, float %1763, i64 0, !dbg !18
  %1927 = insertelement <4 x float> %1926, float %1769, i64 1, !dbg !18
  %1928 = insertelement <4 x float> %1927, float %1764, i64 2, !dbg !18
  %1929 = insertelement <4 x float> %1928, float %1770, i64 3, !dbg !18
  %row.reciprocal82 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1714, !dbg !18
  %normalized83 = fmul <4 x float> %1929, %row.reciprocal82, !dbg !18
  %1930 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !19
  %1931 = extractelement <4 x half> %1930, i64 3, !dbg !19
  %1932 = extractelement <4 x half> %1930, i64 2, !dbg !19
  %1933 = extractelement <4 x half> %1930, i64 1, !dbg !19
  %1934 = extractelement <4 x half> %1930, i64 0, !dbg !19
  %1935 = insertelement <4 x float> poison, float %1653, i64 0, !dbg !18
  %1936 = insertelement <4 x float> %1935, float %1659, i64 1, !dbg !18
  %1937 = insertelement <4 x float> %1936, float %1654, i64 2, !dbg !18
  %1938 = insertelement <4 x float> %1937, float %1660, i64 3, !dbg !18
  %normalized84 = fmul <4 x float> %1938, %row.reciprocal82, !dbg !18
  %1939 = fptrunc <4 x float> %normalized84 to <4 x half>, !dbg !19
  %1940 = extractelement <4 x half> %1939, i64 3, !dbg !19
  %1941 = extractelement <4 x half> %1939, i64 2, !dbg !19
  %1942 = extractelement <4 x half> %1939, i64 1, !dbg !19
  %1943 = extractelement <4 x half> %1939, i64 0, !dbg !19
  %1944 = insertelement <4 x float> poison, float %1545, i64 0, !dbg !18
  %1945 = insertelement <4 x float> %1944, float %1551, i64 1, !dbg !18
  %1946 = insertelement <4 x float> %1945, float %1546, i64 2, !dbg !18
  %1947 = insertelement <4 x float> %1946, float %1552, i64 3, !dbg !18
  %row.reciprocal85 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1496, !dbg !18
  %normalized86 = fmul <4 x float> %1947, %row.reciprocal85, !dbg !18
  %1948 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !19
  %1949 = extractelement <4 x half> %1948, i64 3, !dbg !19
  %1950 = extractelement <4 x half> %1948, i64 2, !dbg !19
  %1951 = extractelement <4 x half> %1948, i64 1, !dbg !19
  %1952 = extractelement <4 x half> %1948, i64 0, !dbg !19
  %1953 = insertelement <4 x float> poison, float %1435, i64 0, !dbg !18
  %1954 = insertelement <4 x float> %1953, float %1441, i64 1, !dbg !18
  %1955 = insertelement <4 x float> %1954, float %1436, i64 2, !dbg !18
  %1956 = insertelement <4 x float> %1955, float %1442, i64 3, !dbg !18
  %normalized87 = fmul <4 x float> %1956, %row.reciprocal85, !dbg !18
  %1957 = fptrunc <4 x float> %normalized87 to <4 x half>, !dbg !19
  %1958 = extractelement <4 x half> %1957, i64 3, !dbg !19
  %1959 = extractelement <4 x half> %1957, i64 2, !dbg !19
  %1960 = extractelement <4 x half> %1957, i64 1, !dbg !19
  %1961 = extractelement <4 x half> %1957, i64 0, !dbg !19
  %1962 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread502 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local503 = zext nneg i32 %address.thread502 to i64, !dbg !13
  %address.local504 = add nsw i64 %16, %address.local503, !dbg !20
  %address.local505 = lshr i64 %address.local503, 6, !dbg !13
  %address.local506 = shl nuw nsw i64 %address.local505, 4, !dbg !20
  %address.local507 = add nsw i64 %address.local504, %address.local506, !dbg !20
  %address.local508 = and i64 %address.local503, 240, !dbg !20
  %address.local509 = sub nsw i64 %address.local507, %address.local508, !dbg !20
  %address.local510 = add nsw i64 %address.local509, %17, !dbg !20
  %address.local511 = shl nuw nsw i64 %address.local510, 7, !dbg !20
  %address.local512 = add nuw nsw i64 %1962, %address.local511, !dbg !20
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local514 = lshr i64 %address.local503, 2, !dbg !13
  %address.local515 = and i64 %address.local514, 12, !dbg !13
  %address.local516 = or disjoint i64 %address.local515, 64, !dbg !20
  %address.local517 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local513, i64 %address.local516, !dbg !20
  store half %1961, ptr addrspace(1) %address.local517, align 2, !dbg !20
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local519 = or disjoint i64 %address.local515, 65, !dbg !20
  %address.local520 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local518, i64 %address.local519, !dbg !20
  store half %1960, ptr addrspace(1) %address.local520, align 2, !dbg !20
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local522 = or disjoint i64 %address.local515, 66, !dbg !20
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local521, i64 %address.local522, !dbg !20
  store half %1959, ptr addrspace(1) %address.local523, align 2, !dbg !20
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local525 = or disjoint i64 %address.local515, 67, !dbg !20
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local524, i64 %address.local525, !dbg !20
  store half %1958, ptr addrspace(1) %address.local526, align 2, !dbg !20
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local528 = or disjoint i64 %address.local515, 80, !dbg !20
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local527, i64 %address.local528, !dbg !20
  store half %1952, ptr addrspace(1) %address.local529, align 2, !dbg !20
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local531 = or disjoint i64 %address.local515, 81, !dbg !20
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local530, i64 %address.local531, !dbg !20
  store half %1951, ptr addrspace(1) %address.local532, align 2, !dbg !20
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local534 = or disjoint i64 %address.local515, 82, !dbg !20
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local533, i64 %address.local534, !dbg !20
  store half %1950, ptr addrspace(1) %address.local535, align 2, !dbg !20
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local512, !dbg !20
  %address.local537 = or disjoint i64 %address.local515, 83, !dbg !20
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local536, i64 %address.local537, !dbg !20
  store half %1949, ptr addrspace(1) %address.local538, align 2, !dbg !20
  %address.local539 = shl nsw i64 %address.local510, 7, !dbg !20
  %address.local540 = add i64 %address.local539, 8192, !dbg !20
  %address.local541 = add nuw nsw i64 %1962, %address.local540, !dbg !20
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local542, i64 %address.local516, !dbg !20
  store half %1943, ptr addrspace(1) %address.local543, align 2, !dbg !20
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local519, !dbg !20
  store half %1942, ptr addrspace(1) %address.local545, align 2, !dbg !20
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local522, !dbg !20
  store half %1941, ptr addrspace(1) %address.local547, align 2, !dbg !20
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local525, !dbg !20
  store half %1940, ptr addrspace(1) %address.local549, align 2, !dbg !20
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local528, !dbg !20
  store half %1934, ptr addrspace(1) %address.local551, align 2, !dbg !20
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local531, !dbg !20
  store half %1933, ptr addrspace(1) %address.local553, align 2, !dbg !20
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local534, !dbg !20
  store half %1932, ptr addrspace(1) %address.local555, align 2, !dbg !20
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local541, !dbg !20
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local556, i64 %address.local537, !dbg !20
  store half %1931, ptr addrspace(1) %address.local557, align 2, !dbg !20
  br label %1963, !dbg !20

1963:                                             ; preds = %1925, %1924
  br i1 true, label %1964, label %2002, !dbg !20

1964:                                             ; preds = %1963
  %1965 = insertelement <4 x float> poison, float %1735, i64 0, !dbg !18
  %1966 = insertelement <4 x float> %1965, float %1741, i64 1, !dbg !18
  %1967 = insertelement <4 x float> %1966, float %1736, i64 2, !dbg !18
  %1968 = insertelement <4 x float> %1967, float %1742, i64 3, !dbg !18
  %row.reciprocal88 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1714, !dbg !18
  %normalized89 = fmul <4 x float> %1968, %row.reciprocal88, !dbg !18
  %1969 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !19
  %1970 = extractelement <4 x half> %1969, i64 3, !dbg !19
  %1971 = extractelement <4 x half> %1969, i64 2, !dbg !19
  %1972 = extractelement <4 x half> %1969, i64 1, !dbg !19
  %1973 = extractelement <4 x half> %1969, i64 0, !dbg !19
  %1974 = insertelement <4 x float> poison, float %1624, i64 0, !dbg !18
  %1975 = insertelement <4 x float> %1974, float %1630, i64 1, !dbg !18
  %1976 = insertelement <4 x float> %1975, float %1625, i64 2, !dbg !18
  %1977 = insertelement <4 x float> %1976, float %1631, i64 3, !dbg !18
  %normalized90 = fmul <4 x float> %1977, %row.reciprocal88, !dbg !18
  %1978 = fptrunc <4 x float> %normalized90 to <4 x half>, !dbg !19
  %1979 = extractelement <4 x half> %1978, i64 3, !dbg !19
  %1980 = extractelement <4 x half> %1978, i64 2, !dbg !19
  %1981 = extractelement <4 x half> %1978, i64 1, !dbg !19
  %1982 = extractelement <4 x half> %1978, i64 0, !dbg !19
  %1983 = insertelement <4 x float> poison, float %1517, i64 0, !dbg !18
  %1984 = insertelement <4 x float> %1983, float %1523, i64 1, !dbg !18
  %1985 = insertelement <4 x float> %1984, float %1518, i64 2, !dbg !18
  %1986 = insertelement <4 x float> %1985, float %1524, i64 3, !dbg !18
  %row.reciprocal91 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1496, !dbg !18
  %normalized92 = fmul <4 x float> %1986, %row.reciprocal91, !dbg !18
  %1987 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !19
  %1988 = extractelement <4 x half> %1987, i64 3, !dbg !19
  %1989 = extractelement <4 x half> %1987, i64 2, !dbg !19
  %1990 = extractelement <4 x half> %1987, i64 1, !dbg !19
  %1991 = extractelement <4 x half> %1987, i64 0, !dbg !19
  %1992 = insertelement <4 x float> poison, float %1406, i64 0, !dbg !18
  %1993 = insertelement <4 x float> %1992, float %1412, i64 1, !dbg !18
  %1994 = insertelement <4 x float> %1993, float %1407, i64 2, !dbg !18
  %1995 = insertelement <4 x float> %1994, float %1413, i64 3, !dbg !18
  %normalized93 = fmul <4 x float> %1995, %row.reciprocal91, !dbg !18
  %1996 = fptrunc <4 x float> %normalized93 to <4 x half>, !dbg !19
  %1997 = extractelement <4 x half> %1996, i64 3, !dbg !19
  %1998 = extractelement <4 x half> %1996, i64 2, !dbg !19
  %1999 = extractelement <4 x half> %1996, i64 1, !dbg !19
  %2000 = extractelement <4 x half> %1996, i64 0, !dbg !19
  %2001 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread558 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local559 = zext nneg i32 %address.thread558 to i64, !dbg !13
  %address.local560 = add nsw i64 %16, %address.local559, !dbg !20
  %address.local561 = lshr i64 %address.local559, 6, !dbg !13
  %address.local562 = shl nuw nsw i64 %address.local561, 4, !dbg !20
  %address.local563 = add nsw i64 %address.local560, %address.local562, !dbg !20
  %address.local564 = and i64 %address.local559, 240, !dbg !20
  %address.local565 = sub nsw i64 %address.local563, %address.local564, !dbg !20
  %address.local566 = add nsw i64 %address.local565, %17, !dbg !20
  %address.local567 = shl nuw nsw i64 %address.local566, 7, !dbg !20
  %address.local568 = add nuw nsw i64 %2001, %address.local567, !dbg !20
  %address.local569 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local570 = lshr i64 %address.local559, 2, !dbg !13
  %address.local571 = and i64 %address.local570, 12, !dbg !13
  %address.local572 = or disjoint i64 %address.local571, 96, !dbg !20
  %address.local573 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local569, i64 %address.local572, !dbg !20
  store half %2000, ptr addrspace(1) %address.local573, align 2, !dbg !20
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local575 = or disjoint i64 %address.local571, 97, !dbg !20
  %address.local576 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local574, i64 %address.local575, !dbg !20
  store half %1999, ptr addrspace(1) %address.local576, align 2, !dbg !20
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local578 = or disjoint i64 %address.local571, 98, !dbg !20
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local577, i64 %address.local578, !dbg !20
  store half %1998, ptr addrspace(1) %address.local579, align 2, !dbg !20
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local581 = or disjoint i64 %address.local571, 99, !dbg !20
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local580, i64 %address.local581, !dbg !20
  store half %1997, ptr addrspace(1) %address.local582, align 2, !dbg !20
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local584 = or disjoint i64 %address.local571, 112, !dbg !20
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local583, i64 %address.local584, !dbg !20
  store half %1991, ptr addrspace(1) %address.local585, align 2, !dbg !20
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local587 = or disjoint i64 %address.local571, 113, !dbg !20
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local586, i64 %address.local587, !dbg !20
  store half %1990, ptr addrspace(1) %address.local588, align 2, !dbg !20
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local590 = or disjoint i64 %address.local571, 114, !dbg !20
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local589, i64 %address.local590, !dbg !20
  store half %1989, ptr addrspace(1) %address.local591, align 2, !dbg !20
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local568, !dbg !20
  %address.local593 = or i64 %address.local570, 115, !dbg !20
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local592, i64 %address.local593, !dbg !20
  store half %1988, ptr addrspace(1) %address.local594, align 2, !dbg !20
  %address.local595 = shl nsw i64 %address.local566, 7, !dbg !20
  %address.local596 = add i64 %address.local595, 8192, !dbg !20
  %address.local597 = add nuw nsw i64 %2001, %address.local596, !dbg !20
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local572, !dbg !20
  store half %1982, ptr addrspace(1) %address.local599, align 2, !dbg !20
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local575, !dbg !20
  store half %1981, ptr addrspace(1) %address.local601, align 2, !dbg !20
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local602, i64 %address.local578, !dbg !20
  store half %1980, ptr addrspace(1) %address.local603, align 2, !dbg !20
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local581, !dbg !20
  store half %1979, ptr addrspace(1) %address.local605, align 2, !dbg !20
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local584, !dbg !20
  store half %1973, ptr addrspace(1) %address.local607, align 2, !dbg !20
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local587, !dbg !20
  store half %1972, ptr addrspace(1) %address.local609, align 2, !dbg !20
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local590, !dbg !20
  store half %1971, ptr addrspace(1) %address.local611, align 2, !dbg !20
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local597, !dbg !20
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local612, i64 %address.local593, !dbg !20
  store half %1970, ptr addrspace(1) %address.local613, align 2, !dbg !20
  br label %2002, !dbg !20

2002:                                             ; preds = %1964, %1963
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0006-test_input_1x32x4096x128-128-128-32-online/nohint-rolled")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 73, column: 26, scope: !4)
!8 = !DILocation(line: 60, column: 24, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!12 = !DILocation(line: 70, column: 22, scope: !4)
!13 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 71, column: 22, scope: !4)
!17 = !DILocation(line: 101, column: 20, scope: !4)
!18 = !DILocation(line: 107, column: 23, scope: !4)
!19 = !DILocation(line: 108, column: 19, scope: !4)
!20 = !DILocation(line: 109, column: 9, scope: !4)
!21 = !DILocation(line: 63, column: 17, scope: !4)
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
!36 = !DILocation(line: 74, column: 23, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
