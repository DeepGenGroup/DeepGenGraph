; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/nohint-rolled/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/nohint-rolled/input.mlir"

@shm_0 = addrspace(3) global [12288 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 6, !dbg !10
  %10 = icmp slt i32 %7, 0, !dbg !12
  %11 = xor i64 %8, -1, !dbg !12
  %12 = select i1 %10, i64 %11, i64 %8, !dbg !12
  %13 = sdiv i64 %12, 64, !dbg !12
  %.lobit = ashr i32 %7, 31, !dbg !12
  %14 = sext i32 %.lobit to i64, !dbg !12
  %15 = xor i64 %13, %14, !dbg !12
  %16 = add nsw i64 %15, %6, !dbg !12
  %17 = and i64 %16, 31, !dbg !12
  %18 = shl nsw i64 %8, 6, !dbg !12
  %19 = mul nsw i64 %15, -4096, !dbg !12
  %20 = add nsw i64 %18, %19, !dbg !12
  %21 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !39
  %22 = zext nneg i32 %21 to i64, !dbg !13
  %23 = lshr i64 %22, 6, !dbg !13
  %24 = or disjoint i64 %9, 63, !dbg !14
  %25 = add nsw i64 %9, 64, !dbg !14
  %26 = icmp uge i64 %25, %24, !dbg !14
  %27 = icmp slt i32 %7, 0, !dbg !7
  %28 = select i1 %27, i64 %11, i64 %8, !dbg !7
  %29 = sdiv i64 %28, 2, !dbg !7
  %30 = sub nsw i64 0, %29, !dbg !7
  %31 = add nsw i64 %29, 1, !dbg !7
  %32 = select i1 %27, i64 %30, i64 %31, !dbg !7
  br label %33, !dbg !7

33:                                               ; preds = %293, %4
  %34 = phi i64 [ %862, %293 ], [ 0, %4 ], !dbg !7
  %35 = phi <4 x float> [ %637, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %36 = phi <4 x float> [ %653, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %37 = phi <8 x float> [ %858, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %38 = phi <1 x float> [ %571, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %39 = phi <1 x float> [ %402, %293 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %40 = phi <4 x float> [ %715, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %41 = phi <4 x float> [ %731, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %42 = phi <8 x float> [ %859, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %43 = phi <4 x float> [ %793, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %44 = phi <4 x float> [ %809, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %45 = phi <8 x float> [ %860, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %46 = phi <4 x float> [ %841, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %47 = phi <4 x float> [ %857, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %48 = phi <8 x float> [ %861, %293 ], [ zeroinitializer, %4 ], !dbg !7
  %49 = icmp slt i64 %34, %32, !dbg !7
  br i1 %49, label %50, label %863, !dbg !7

50:                                               ; preds = %33
  %51 = shl nsw i64 %34, 7, !dbg !7
  %52 = lshr i64 %34, 5, !dbg !16
  %53 = shl i64 %52, 12, !dbg !16
  %54 = sub i64 %51, %53, !dbg !16
  br label %55, !dbg !21

55:                                               ; preds = %60, %50
  %56 = phi i64 [ %176, %60 ], [ 0, %50 ], !dbg !21
  %57 = phi <4 x float> [ %167, %60 ], [ zeroinitializer, %50 ], !dbg !21
  %58 = phi <32 x float> [ %175, %60 ], [ zeroinitializer, %50 ], !dbg !21
  %59 = icmp samesign ult i64 %56, 4, !dbg !21
  br i1 %59, label %60, label %177, !dbg !21

60:                                               ; preds = %55
  %61 = shl nuw nsw i64 %56, 5, !dbg !21
  %.idx = shl nuw nsw i64 %17, 20, !dbg !12
  %62 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx34 = shl i64 %20, 8, !dbg !12
  %63 = getelementptr i8, ptr addrspace(1) %62, i64 %.idx34, !dbg !12
  %64 = getelementptr [2 x i8], ptr addrspace(1) %63, i64 %61, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local68 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local69 = and i64 %address.local68, 1536, !dbg !12
  %address.local70 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local69, !dbg !12
  %address.local71 = ptrtoint ptr addrspace(3) %address.local70 to i64, !dbg !12
  %address.local72 = trunc i64 %address.local71 to i32, !dbg !12
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local72), !dbg !12
  %66 = zext i32 %65 to i64, !dbg !12
  %67 = inttoptr i64 %66 to ptr addrspace(3), !dbg !12
  %address.local73 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local74 = and i64 %address.local73, 8064, !dbg !12
  %address.local75 = getelementptr [2 x i8], ptr addrspace(1) %64, i64 %address.local74, !dbg !12
  %address.local76 = lshr i64 %address.local, 3, !dbg !12
  %address.local77 = and i64 %address.local76, 24, !dbg !12
  %address.local78 = getelementptr [2 x i8], ptr addrspace(1) %address.local75, i64 %address.local77, !dbg !12
  %68 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %69 = ptrtoint ptr addrspace(1) %address.local78 to i64, !dbg !12
  %70 = sub i64 %69, %68, !dbg !12
  %dma.buffer.offset601 = trunc i64 %70 to i32, !dbg !12
  %71 = trunc i64 %68 to i32, !dbg !12
  %72 = insertelement <4 x i32> poison, i32 %71, i64 0, !dbg !12
  %73 = lshr i64 %68, 32, !dbg !12
  %74 = trunc i64 %73 to i32, !dbg !12
  %75 = insertelement <4 x i32> %72, i32 %74, i64 1, !dbg !12
  %76 = insertelement <4 x i32> %75, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc602 = insertelement <4 x i32> %76, i32 131072, i64 3, !dbg !12
  %77 = ptrtoint ptr addrspace(3) %67 to i32, !dbg !12
  %dma.buffer.lds603 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %77), !dbg !12
  %78 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds603, i32 %dma.buffer.offset601, <4 x i32> %dma.buffer.rsrc602) #10, !dbg !12
  %79 = shl nuw nsw i64 %56, 10, !dbg !16
  %80 = add nuw i64 %34, %79, !dbg !16
  %.lobit35 = ashr i64 %80, 63, !dbg !16
  %81 = xor i64 %80, %.lobit35, !dbg !16
  %82 = sdiv i64 %81, 4096, !dbg !16
  %.lobit36 = ashr i64 %80, 63, !dbg !16
  %83 = xor i64 %82, %.lobit36, !dbg !16
  %84 = add nsw i64 %83, %6, !dbg !16
  %85 = add nuw nsw i64 %61, %52, !dbg !16
  %86 = and i64 %85, 127, !dbg !16
  %87 = shl i64 %84, 19, !dbg !16
  %88 = and i64 %87, 16252928, !dbg !16
  %89 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %88, !dbg !16
  %90 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %86, !dbg !16
  %.idx37 = shl i64 %54, 8, !dbg !16
  %91 = getelementptr i8, ptr addrspace(1) %90, i64 %.idx37, !dbg !16
  %address.local79 = and i64 %address.local68, 1536, !dbg !16
  %address.local80 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local79, !dbg !16
  %address.local81 = ptrtoint ptr addrspace(3) %address.local80 to i64, !dbg !16
  %address.local82 = trunc i64 %address.local81 to i32, !dbg !16
  %92 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local82), !dbg !16
  %93 = zext i32 %92 to i64, !dbg !16
  %94 = inttoptr i64 %93 to ptr addrspace(3), !dbg !16
  %address.local83 = lshr i64 %address.local, 4, !dbg !16
  %address.local84 = and i64 %address.local83, 8, !dbg !16
  %address.local85 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %address.local84, !dbg !16
  %address.local86 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local87 = and i64 %address.local86, 16256, !dbg !16
  %address.local88 = getelementptr [2 x i8], ptr addrspace(1) %address.local85, i64 %address.local87, !dbg !16
  %95 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %96 = ptrtoint ptr addrspace(1) %address.local88 to i64, !dbg !16
  %97 = sub i64 %96, %95, !dbg !16
  %dma.buffer.offset598 = trunc i64 %97 to i32, !dbg !16
  %98 = trunc i64 %95 to i32, !dbg !16
  %99 = insertelement <4 x i32> poison, i32 %98, i64 0, !dbg !16
  %100 = lshr i64 %95, 32, !dbg !16
  %101 = trunc i64 %100 to i32, !dbg !16
  %102 = insertelement <4 x i32> %99, i32 %101, i64 1, !dbg !16
  %103 = insertelement <4 x i32> %102, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc599 = insertelement <4 x i32> %103, i32 131072, i64 3, !dbg !16
  %104 = ptrtoint ptr addrspace(3) %94 to i32, !dbg !16
  %dma.buffer.lds600 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %104), !dbg !16
  %105 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds600, i32 %dma.buffer.offset598, <4 x i32> %dma.buffer.rsrc599) #10, !dbg !16
  %address.local89 = or disjoint i64 %address.local, 256, !dbg !16
  %address.local90 = shl nuw nsw i64 %address.local89, 3, !dbg !16
  %address.local91 = and i64 %address.local90, 3584, !dbg !16
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local91, !dbg !16
  %address.local93 = ptrtoint ptr addrspace(3) %address.local92 to i64, !dbg !16
  %address.local94 = trunc i64 %address.local93 to i32, !dbg !16
  %106 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !16
  %107 = zext i32 %106 to i64, !dbg !16
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !16
  %address.local95 = lshr i64 %address.local89, 4, !dbg !16
  %address.local96 = and i64 %address.local95, 24, !dbg !16
  %address.local97 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %address.local96, !dbg !16
  %address.local98 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local99 = and i64 %address.local98, 16256, !dbg !16
  %address.local100 = getelementptr [2 x i8], ptr addrspace(1) %address.local97, i64 %address.local99, !dbg !16
  %109 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %110 = ptrtoint ptr addrspace(1) %address.local100 to i64, !dbg !16
  %111 = sub i64 %110, %109, !dbg !16
  %dma.buffer.offset = trunc i64 %111 to i32, !dbg !16
  %112 = trunc i64 %109 to i32, !dbg !16
  %113 = insertelement <4 x i32> poison, i32 %112, i64 0, !dbg !16
  %114 = lshr i64 %109, 32, !dbg !16
  %115 = trunc i64 %114 to i32, !dbg !16
  %116 = insertelement <4 x i32> %113, i32 %115, i64 1, !dbg !16
  %117 = insertelement <4 x i32> %116, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc = insertelement <4 x i32> %117, i32 131072, i64 3, !dbg !16
  %118 = ptrtoint ptr addrspace(3) %108 to i32, !dbg !16
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !16
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local101 = lshr i32 %address.thread, 2, !dbg !13
  %address.local102 = and i32 %address.local101, 4, !dbg !13
  %address.local103 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local104 = and i32 %address.local103, 384, !dbg !13
  %address.local105 = or disjoint i32 %address.local102, %address.local104, !dbg !13
  %address.local106 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local107 = and i32 %address.local106, 120, !dbg !13
  %address.local108 = or disjoint i32 %address.local105, %address.local107, !dbg !13
  %address.local109 = shl nuw nsw i32 %address.thread, 4, !dbg !13
  %address.local110 = and i32 %address.local109, 512, !dbg !13
  %address.local111 = or disjoint i32 %address.local108, %address.local110, !dbg !13
  %address.local112 = zext nneg i32 %address.local111 to i64, !dbg !13
  %address.local113 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local112, !dbg !13
  %120 = load <4 x half>, ptr addrspace(3) %address.local113, align 2, !dbg !13
  %address.local114 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local112, !dbg !13
  %address.local115 = getelementptr i8, ptr addrspace(3) %address.local114, i64 2048, !dbg !13
  %121 = load <4 x half>, ptr addrspace(3) %address.local115, align 2, !dbg !13
  %122 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !13
  %123 = fmul <4 x half> %121, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local116 = lshr i32 %address.thread, 2, !dbg !8
  %address.local117 = and i32 %address.local116, 4, !dbg !8
  %address.local118 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local119 = and i32 %address.local118, 24, !dbg !8
  %address.local120 = or disjoint i32 %address.local117, %address.local119, !dbg !8
  %address.local121 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local122 = and i32 %address.local121, 1024, !dbg !8
  %address.local123 = or disjoint i32 %address.local120, %address.local122, !dbg !8
  %address.local124 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local125 = and i32 %address.local124, 384, !dbg !8
  %address.local126 = or disjoint i32 %address.local123, %address.local125, !dbg !8
  %address.local127 = zext nneg i32 %address.local126 to i64, !dbg !8
  %address.local128 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local127, !dbg !8
  %124 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !8
  %125 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %address.local129 = lshr i32 %address.thread, 2, !dbg !8
  %address.local130 = and i32 %address.local129, 4, !dbg !8
  %address.local131 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local132 = and i32 %address.local131, 24, !dbg !8
  %address.local133 = or disjoint i32 %address.local130, %address.local132, !dbg !8
  %address.local134 = shl nuw nsw i32 %address.thread, 5, !dbg !8
  %address.local135 = and i32 %address.local134, 1024, !dbg !8
  %address.local136 = or disjoint i32 %address.local133, %address.local135, !dbg !8
  %address.local137 = shl nuw nsw i32 %address.thread, 7, !dbg !8
  %address.local138 = and i32 %address.local137, 384, !dbg !8
  %address.local139 = or disjoint i32 %address.local136, %address.local138, !dbg !8
  %address.local140 = zext nneg i32 %address.local139 to i64, !dbg !8
  %address.local141 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local140, !dbg !8
  %address.local142 = getelementptr i8, ptr addrspace(3) %address.local141, i64 4096, !dbg !8
  %126 = load <4 x half>, ptr addrspace(3) %address.local142, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %124, <4 x float> %125) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %128 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %126, <4 x float> %127) #9, !dbg !8
  %130 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %131 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %132 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !36
  %133 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !36
  %134 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !36
  %address.local143 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local145 = lshr i64 %address.local, 2, !dbg !8
  %address.local146 = mul nsw i64 %address.local145, -512, !dbg !8
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) %address.local144, i64 %address.local146, !dbg !8
  %address.local148 = shl nuw nsw i64 %address.local, 1, !dbg !8
  %address.local149 = and i64 %address.local148, 24, !dbg !8
  %address.local150 = getelementptr [2 x i8], ptr addrspace(3) %address.local147, i64 %address.local149, !dbg !8
  %address.local151 = lshr i64 %address.local, 2, !dbg !13
  %address.local152 = and i64 %address.local151, 12, !dbg !13
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) %address.local150, i64 %address.local152, !dbg !8
  %address.local154 = and i64 %address.local, 63, !dbg !13
  %address.local155 = lshr i64 %address.local154, 5, !dbg !13
  %address.local156 = mul nuw nsw i64 %address.local155, 1016, !dbg !8
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) %address.local153, i64 %address.local156, !dbg !8
  %address.local158 = getelementptr i8, ptr addrspace(3) %address.local157, i64 64, !dbg !8
  %135 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !8
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local152, !dbg !8
  %address.local161 = getelementptr [2 x i8], ptr addrspace(3) %address.local160, i64 %address.local156, !dbg !8
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) %address.local161, i64 %address.local146, !dbg !8
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) %address.local162, i64 %address.local149, !dbg !8
  %address.local164 = getelementptr i8, ptr addrspace(3) %address.local163, i64 4160, !dbg !8
  %136 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !8
  %137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %135, <4 x float> %128) #9, !dbg !8
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) %address.local165, i64 %address.local146, !dbg !8
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) %address.local166, i64 %address.local149, !dbg !8
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) %address.local167, i64 %address.local152, !dbg !8
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) %address.local168, i64 %address.local156, !dbg !8
  %address.local170 = getelementptr i8, ptr addrspace(3) %address.local169, i64 128, !dbg !8
  %138 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !8
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %136, <4 x float> %137) #9, !dbg !8
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) %address.local171, i64 %address.local152, !dbg !8
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) %address.local172, i64 %address.local156, !dbg !8
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) %address.local173, i64 %address.local146, !dbg !8
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) %address.local174, i64 %address.local149, !dbg !8
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 4224, !dbg !8
  %140 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !8
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %138, <4 x float> %130) #9, !dbg !8
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %140, <4 x float> %141) #9, !dbg !8
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) %address.local177, i64 %address.local146, !dbg !8
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) %address.local178, i64 %address.local149, !dbg !8
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) %address.local179, i64 %address.local152, !dbg !8
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) %address.local180, i64 %address.local156, !dbg !8
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 192, !dbg !8
  %143 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !8
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local143, !dbg !8
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) %address.local183, i64 %address.local152, !dbg !8
  %address.local185 = getelementptr [2 x i8], ptr addrspace(3) %address.local184, i64 %address.local156, !dbg !8
  %address.local186 = getelementptr [2 x i8], ptr addrspace(3) %address.local185, i64 %address.local146, !dbg !8
  %address.local187 = getelementptr [2 x i8], ptr addrspace(3) %address.local186, i64 %address.local149, !dbg !8
  %address.local188 = getelementptr i8, ptr addrspace(3) %address.local187, i64 4288, !dbg !8
  %144 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !8
  %145 = shufflevector <4 x float> %142, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %143, <4 x float> %131) #9, !dbg !8
  %address.local189 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local127, !dbg !8
  %address.local190 = getelementptr i8, ptr addrspace(3) %address.local189, i64 1024, !dbg !8
  %147 = load <4 x half>, ptr addrspace(3) %address.local190, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %144, <4 x float> %146) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %149 = shufflevector <4 x float> %148, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local140, !dbg !8
  %address.local192 = getelementptr i8, ptr addrspace(3) %address.local191, i64 5120, !dbg !8
  %150 = load <4 x half>, ptr addrspace(3) %address.local192, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %147, <4 x float> %132) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %150, <4 x float> %151) #9, !dbg !8
  %address.local193 = add nuw nsw i64 %address.local143, 512, !dbg !8
  %address.local194 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local193, !dbg !8
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) %address.local194, i64 %address.local146, !dbg !8
  %address.local196 = getelementptr [2 x i8], ptr addrspace(3) %address.local195, i64 %address.local149, !dbg !8
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) %address.local196, i64 %address.local152, !dbg !8
  %address.local198 = getelementptr [2 x i8], ptr addrspace(3) %address.local197, i64 %address.local156, !dbg !8
  %address.local199 = getelementptr i8, ptr addrspace(3) %address.local198, i64 64, !dbg !8
  %153 = load <4 x half>, ptr addrspace(3) %address.local199, align 2, !dbg !8
  %address.local200 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local193, !dbg !8
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) %address.local200, i64 %address.local152, !dbg !8
  %address.local202 = getelementptr [2 x i8], ptr addrspace(3) %address.local201, i64 %address.local156, !dbg !8
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) %address.local202, i64 %address.local146, !dbg !8
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) %address.local203, i64 %address.local149, !dbg !8
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local204, i64 4160, !dbg !8
  %154 = load <4 x half>, ptr addrspace(3) %address.local205, align 2, !dbg !8
  %155 = shufflevector <4 x float> %152, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %153, <4 x float> %133) #9, !dbg !8
  %address.local206 = add nuw nsw i64 %address.local143, 512, !dbg !8
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local206, !dbg !8
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local146, !dbg !8
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local149, !dbg !8
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local209, i64 %address.local152, !dbg !8
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local156, !dbg !8
  %address.local212 = getelementptr i8, ptr addrspace(3) %address.local211, i64 128, !dbg !8
  %157 = load <4 x half>, ptr addrspace(3) %address.local212, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %154, <4 x float> %156) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %159 = shufflevector <4 x float> %158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local206, !dbg !8
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local152, !dbg !8
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local156, !dbg !8
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local215, i64 %address.local146, !dbg !8
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local149, !dbg !8
  %address.local218 = getelementptr i8, ptr addrspace(3) %address.local217, i64 4224, !dbg !8
  %160 = load <4 x half>, ptr addrspace(3) %address.local218, align 2, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %157, <4 x float> %134) #9, !dbg !8
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !8
  %162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %160, <4 x float> %161) #9, !dbg !8
  %address.local219 = add nuw nsw i64 %address.local143, 512, !dbg !8
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !8
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local146, !dbg !8
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local149, !dbg !8
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local152, !dbg !8
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local156, !dbg !8
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 192, !dbg !8
  %163 = load <4 x half>, ptr addrspace(3) %address.local225, align 2, !dbg !8
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local219, !dbg !8
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local152, !dbg !8
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local156, !dbg !8
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local146, !dbg !8
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local149, !dbg !8
  %address.local231 = getelementptr i8, ptr addrspace(3) %address.local230, i64 4288, !dbg !8
  %164 = load <4 x half>, ptr addrspace(3) %address.local231, align 2, !dbg !8
  %165 = shufflevector <4 x float> %162, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %122, <4 x half> %163, <4 x float> %57) #9, !dbg !8
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %164, <4 x float> %166) #9, !dbg !8
  %168 = shufflevector <4 x float> %167, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %169 = shufflevector <4 x float> %129, <4 x float> %139, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %170 = shufflevector <32 x float> %169, <32 x float> %145, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %171 = shufflevector <32 x float> %170, <32 x float> %149, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %172 = shufflevector <32 x float> %171, <32 x float> %155, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %173 = shufflevector <32 x float> %172, <32 x float> %159, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %174 = shufflevector <32 x float> %173, <32 x float> %165, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %175 = shufflevector <32 x float> %174, <32 x float> %168, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !21
  fence syncscope("workgroup") release, !dbg !21
  call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %176 = add nuw nsw i64 %56, 1, !dbg !21
  br label %55, !dbg !21

177:                                              ; preds = %55
  %178 = add nsw i64 %52, %6, !dbg !17
  %179 = shl i64 %178, 19, !dbg !17
  %180 = and i64 %179, 16252928, !dbg !17
  %181 = shl nsw i64 %54, 7, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %182 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %180, !dbg !17
  %183 = getelementptr [2 x i8], ptr addrspace(1) %182, i64 %181, !dbg !17
  %address.thread232 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !17, !range !39, !frisk.register_lifetimes !1
  %address.local233 = zext nneg i32 %address.thread232 to i64, !dbg !13
  %address.local234 = shl nuw nsw i64 %address.local233, 3, !dbg !12
  %address.local235 = and i64 %address.local234, 1536, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local235, !dbg !16
  %address.local237 = ptrtoint ptr addrspace(3) %address.local236 to i64, !dbg !16
  %address.local238 = trunc i64 %address.local237 to i32, !dbg !16
  %184 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local238), !dbg !17
  %185 = zext i32 %184 to i64, !dbg !17
  %186 = inttoptr i64 %185 to ptr addrspace(3), !dbg !17
  %address.local239 = shl nuw nsw i64 %address.local233, 5, !dbg !17
  %address.local240 = and i64 %address.local239, 8064, !dbg !17
  %address.local241 = shl nuw nsw i64 %address.local233, 3, !dbg !17
  %address.local242 = and i64 %address.local241, 24, !dbg !17
  %address.local243 = or disjoint i64 %address.local240, %address.local242, !dbg !17
  %address.local244 = getelementptr [2 x i8], ptr addrspace(1) %183, i64 %address.local243, !dbg !17
  %187 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %188 = ptrtoint ptr addrspace(1) %address.local244 to i64, !dbg !17
  %189 = sub i64 %188, %187, !dbg !17
  %dma.buffer.offset625 = trunc i64 %189 to i32, !dbg !17
  %190 = trunc i64 %187 to i32, !dbg !17
  %191 = insertelement <4 x i32> poison, i32 %190, i64 0, !dbg !17
  %192 = lshr i64 %187, 32, !dbg !17
  %193 = trunc i64 %192 to i32, !dbg !17
  %194 = insertelement <4 x i32> %191, i32 %193, i64 1, !dbg !17
  %195 = insertelement <4 x i32> %194, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc626 = insertelement <4 x i32> %195, i32 131072, i64 3, !dbg !17
  %196 = ptrtoint ptr addrspace(3) %186 to i32, !dbg !17
  %dma.buffer.lds627 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %196), !dbg !17
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds627, i32 %dma.buffer.offset625, <4 x i32> %dma.buffer.rsrc626) #10, !dbg !17
  %address.local245 = or disjoint i64 %address.local233, 256, !dbg !16
  %address.local246 = shl nuw nsw i64 %address.local245, 3, !dbg !16
  %address.local247 = and i64 %address.local246, 3584, !dbg !16
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local247, !dbg !16
  %address.local249 = ptrtoint ptr addrspace(3) %address.local248 to i64, !dbg !16
  %address.local250 = trunc i64 %address.local249 to i32, !dbg !16
  %198 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local250), !dbg !17
  %199 = zext i32 %198 to i64, !dbg !17
  %200 = inttoptr i64 %199 to ptr addrspace(3), !dbg !17
  %address.local251 = shl nuw nsw i64 %address.local245, 5, !dbg !17
  %address.local252 = and i64 %address.local251, 16256, !dbg !17
  %address.local253 = shl nuw nsw i64 %address.local233, 3, !dbg !17
  %address.local254 = and i64 %address.local253, 24, !dbg !17
  %address.local255 = or disjoint i64 %address.local252, %address.local254, !dbg !17
  %address.local256 = getelementptr [2 x i8], ptr addrspace(1) %183, i64 %address.local255, !dbg !17
  %201 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %202 = ptrtoint ptr addrspace(1) %address.local256 to i64, !dbg !17
  %203 = sub i64 %202, %201, !dbg !17
  %dma.buffer.offset622 = trunc i64 %203 to i32, !dbg !17
  %204 = trunc i64 %201 to i32, !dbg !17
  %205 = insertelement <4 x i32> poison, i32 %204, i64 0, !dbg !17
  %206 = lshr i64 %201, 32, !dbg !17
  %207 = trunc i64 %206 to i32, !dbg !17
  %208 = insertelement <4 x i32> %205, i32 %207, i64 1, !dbg !17
  %209 = insertelement <4 x i32> %208, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc623 = insertelement <4 x i32> %209, i32 131072, i64 3, !dbg !17
  %210 = ptrtoint ptr addrspace(3) %200 to i32, !dbg !17
  %dma.buffer.lds624 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %210), !dbg !17
  %211 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds624, i32 %dma.buffer.offset622, <4 x i32> %dma.buffer.rsrc623) #10, !dbg !17
  %212 = or disjoint i64 %51, 127, !dbg !14
  %213 = icmp uge i64 %9, %212, !dbg !14
  %214 = and i1 %26, %213, !dbg !14
  br i1 %214, label %215, label %216, !dbg !14

215:                                              ; preds = %177
  br label %291, !dbg !14

216:                                              ; preds = %177
  %217 = add nsw i64 %9, %22, !dbg !14
  %218 = shl nuw nsw i64 %23, 4, !dbg !14
  %219 = add nsw i64 %217, %218, !dbg !14
  %220 = and i32 %21, 240, !dbg !14
  %index.compact = zext nneg i32 %220 to i64, !dbg !14
  %221 = sub nsw i64 %219, %index.compact, !dbg !14
  %222 = add nsw i64 %221, 1, !dbg !22
  %223 = and i64 %22, 48, !dbg !14
  %224 = or disjoint i64 %51, %223, !dbg !14
  %225 = lshr i64 %222, 32, !dbg !23
  %compare.high = trunc i64 %225 to i32, !dbg !23
  %226 = lshr i64 %224, 32, !dbg !23
  %compare.high474 = trunc i64 %226 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high474, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high474, !dbg !23
  %compare.low.base = trunc i64 %222 to i32, !dbg !23
  %compare.low.base475 = trunc i64 %224 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base475, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %227 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low476 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result477 = select i1 %compare.high.equal, i1 %compare.low476, i1 %compare.high.order, !dbg !23
  %228 = select i1 %compare.result477, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base478 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local479 = or i32 %compare.local.base478, 2, !dbg !23
  %compare.low480 = icmp ugt i32 %compare.low.base, %compare.local479, !dbg !23
  %compare.result481 = select i1 %compare.high.equal, i1 %compare.low480, i1 %compare.high.order, !dbg !23
  %229 = select i1 %compare.result481, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base482 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local483 = or i32 %compare.local.base482, 3, !dbg !23
  %compare.low484 = icmp ugt i32 %compare.low.base, %compare.local483, !dbg !23
  %compare.result485 = select i1 %compare.high.equal, i1 %compare.low484, i1 %compare.high.order, !dbg !23
  %230 = select i1 %compare.result485, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base486 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local487 = or i32 %compare.local.base486, 4, !dbg !23
  %compare.low488 = icmp ugt i32 %compare.low.base, %compare.local487, !dbg !23
  %compare.result489 = select i1 %compare.high.equal, i1 %compare.low488, i1 %compare.high.order, !dbg !23
  %231 = select i1 %compare.result489, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base490 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local491 = or i32 %compare.local.base490, 5, !dbg !23
  %compare.low492 = icmp ugt i32 %compare.low.base, %compare.local491, !dbg !23
  %compare.result493 = select i1 %compare.high.equal, i1 %compare.low492, i1 %compare.high.order, !dbg !23
  %232 = select i1 %compare.result493, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base494 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local495 = or i32 %compare.local.base494, 6, !dbg !23
  %compare.low496 = icmp ugt i32 %compare.low.base, %compare.local495, !dbg !23
  %compare.result497 = select i1 %compare.high.equal, i1 %compare.low496, i1 %compare.high.order, !dbg !23
  %233 = select i1 %compare.result497, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base498 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local499 = or i32 %compare.local.base498, 7, !dbg !23
  %compare.low500 = icmp ugt i32 %compare.low.base, %compare.local499, !dbg !23
  %compare.result501 = select i1 %compare.high.equal, i1 %compare.low500, i1 %compare.high.order, !dbg !23
  %234 = select i1 %compare.result501, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base502 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local503 = or i32 %compare.local.base502, 8, !dbg !23
  %compare.low504 = icmp ugt i32 %compare.low.base, %compare.local503, !dbg !23
  %compare.result505 = select i1 %compare.high.equal, i1 %compare.low504, i1 %compare.high.order, !dbg !23
  %235 = select i1 %compare.result505, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base506 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local507 = or i32 %compare.local.base506, 9, !dbg !23
  %compare.low508 = icmp ugt i32 %compare.low.base, %compare.local507, !dbg !23
  %compare.result509 = select i1 %compare.high.equal, i1 %compare.low508, i1 %compare.high.order, !dbg !23
  %236 = select i1 %compare.result509, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base510 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local511 = or i32 %compare.local.base510, 10, !dbg !23
  %compare.low512 = icmp ugt i32 %compare.low.base, %compare.local511, !dbg !23
  %compare.result513 = select i1 %compare.high.equal, i1 %compare.low512, i1 %compare.high.order, !dbg !23
  %237 = select i1 %compare.result513, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base514 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local515 = or i32 %compare.local.base514, 11, !dbg !23
  %compare.low516 = icmp ugt i32 %compare.low.base, %compare.local515, !dbg !23
  %compare.result517 = select i1 %compare.high.equal, i1 %compare.low516, i1 %compare.high.order, !dbg !23
  %238 = select i1 %compare.result517, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base518 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local519 = or i32 %compare.local.base518, 12, !dbg !23
  %compare.low520 = icmp ugt i32 %compare.low.base, %compare.local519, !dbg !23
  %compare.result521 = select i1 %compare.high.equal, i1 %compare.low520, i1 %compare.high.order, !dbg !23
  %239 = select i1 %compare.result521, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base522 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local523 = or i32 %compare.local.base522, 13, !dbg !23
  %compare.low524 = icmp ugt i32 %compare.low.base, %compare.local523, !dbg !23
  %compare.result525 = select i1 %compare.high.equal, i1 %compare.low524, i1 %compare.high.order, !dbg !23
  %240 = select i1 %compare.result525, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base526 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local527 = or i32 %compare.local.base526, 14, !dbg !23
  %compare.low528 = icmp ugt i32 %compare.low.base, %compare.local527, !dbg !23
  %compare.result529 = select i1 %compare.high.equal, i1 %compare.low528, i1 %compare.high.order, !dbg !23
  %241 = select i1 %compare.result529, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base530 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local531 = or i32 %compare.local.base530, 15, !dbg !23
  %compare.low532 = icmp ugt i32 %compare.low.base, %compare.local531, !dbg !23
  %compare.result533 = select i1 %compare.high.equal, i1 %compare.low532, i1 %compare.high.order, !dbg !23
  %242 = select i1 %compare.result533, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base534 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local535 = or i32 %compare.local.base534, 64, !dbg !23
  %compare.low536 = icmp ugt i32 %compare.low.base, %compare.local535, !dbg !23
  %compare.result537 = select i1 %compare.high.equal, i1 %compare.low536, i1 %compare.high.order, !dbg !23
  %243 = select i1 %compare.result537, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base538 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local539 = or i32 %compare.local.base538, 65, !dbg !23
  %compare.low540 = icmp ugt i32 %compare.low.base, %compare.local539, !dbg !23
  %compare.result541 = select i1 %compare.high.equal, i1 %compare.low540, i1 %compare.high.order, !dbg !23
  %244 = select i1 %compare.result541, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base542 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local543 = or i32 %compare.local.base542, 66, !dbg !23
  %compare.low544 = icmp ugt i32 %compare.low.base, %compare.local543, !dbg !23
  %compare.result545 = select i1 %compare.high.equal, i1 %compare.low544, i1 %compare.high.order, !dbg !23
  %245 = select i1 %compare.result545, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base546 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local547 = or i32 %compare.local.base546, 67, !dbg !23
  %compare.low548 = icmp ugt i32 %compare.low.base, %compare.local547, !dbg !23
  %compare.result549 = select i1 %compare.high.equal, i1 %compare.low548, i1 %compare.high.order, !dbg !23
  %246 = select i1 %compare.result549, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base550 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local551 = or i32 %compare.local.base550, 68, !dbg !23
  %compare.low552 = icmp ugt i32 %compare.low.base, %compare.local551, !dbg !23
  %compare.result553 = select i1 %compare.high.equal, i1 %compare.low552, i1 %compare.high.order, !dbg !23
  %247 = select i1 %compare.result553, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base554 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local555 = or i32 %compare.local.base554, 69, !dbg !23
  %compare.low556 = icmp ugt i32 %compare.low.base, %compare.local555, !dbg !23
  %compare.result557 = select i1 %compare.high.equal, i1 %compare.low556, i1 %compare.high.order, !dbg !23
  %248 = select i1 %compare.result557, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base558 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local559 = or i32 %compare.local.base558, 70, !dbg !23
  %compare.low560 = icmp ugt i32 %compare.low.base, %compare.local559, !dbg !23
  %compare.result561 = select i1 %compare.high.equal, i1 %compare.low560, i1 %compare.high.order, !dbg !23
  %249 = select i1 %compare.result561, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base562 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local563 = or i32 %compare.local.base562, 71, !dbg !23
  %compare.low564 = icmp ugt i32 %compare.low.base, %compare.local563, !dbg !23
  %compare.result565 = select i1 %compare.high.equal, i1 %compare.low564, i1 %compare.high.order, !dbg !23
  %250 = select i1 %compare.result565, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base566 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local567 = or i32 %compare.local.base566, 72, !dbg !23
  %compare.low568 = icmp ugt i32 %compare.low.base, %compare.local567, !dbg !23
  %compare.result569 = select i1 %compare.high.equal, i1 %compare.low568, i1 %compare.high.order, !dbg !23
  %251 = select i1 %compare.result569, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base570 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local571 = or i32 %compare.local.base570, 73, !dbg !23
  %compare.low572 = icmp ugt i32 %compare.low.base, %compare.local571, !dbg !23
  %compare.result573 = select i1 %compare.high.equal, i1 %compare.low572, i1 %compare.high.order, !dbg !23
  %252 = select i1 %compare.result573, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base574 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local575 = or i32 %compare.local.base574, 74, !dbg !23
  %compare.low576 = icmp ugt i32 %compare.low.base, %compare.local575, !dbg !23
  %compare.result577 = select i1 %compare.high.equal, i1 %compare.low576, i1 %compare.high.order, !dbg !23
  %253 = select i1 %compare.result577, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base578 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local579 = or i32 %compare.local.base578, 75, !dbg !23
  %compare.low580 = icmp ugt i32 %compare.low.base, %compare.local579, !dbg !23
  %compare.result581 = select i1 %compare.high.equal, i1 %compare.low580, i1 %compare.high.order, !dbg !23
  %254 = select i1 %compare.result581, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base582 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local583 = or i32 %compare.local.base582, 76, !dbg !23
  %compare.low584 = icmp ugt i32 %compare.low.base, %compare.local583, !dbg !23
  %compare.result585 = select i1 %compare.high.equal, i1 %compare.low584, i1 %compare.high.order, !dbg !23
  %255 = select i1 %compare.result585, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base586 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local587 = or i32 %compare.local.base586, 77, !dbg !23
  %compare.low588 = icmp ugt i32 %compare.low.base, %compare.local587, !dbg !23
  %compare.result589 = select i1 %compare.high.equal, i1 %compare.low588, i1 %compare.high.order, !dbg !23
  %256 = select i1 %compare.result589, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base590 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local591 = or i32 %compare.local.base590, 78, !dbg !23
  %compare.low592 = icmp ugt i32 %compare.low.base, %compare.local591, !dbg !23
  %compare.result593 = select i1 %compare.high.equal, i1 %compare.low592, i1 %compare.high.order, !dbg !23
  %257 = select i1 %compare.result593, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base594 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base475, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local595 = or i32 %compare.local.base594, 79, !dbg !23
  %compare.low596 = icmp ugt i32 %compare.low.base, %compare.local595, !dbg !23
  %compare.result597 = select i1 %compare.high.equal, i1 %compare.low596, i1 %compare.high.order, !dbg !23
  %258 = select i1 %compare.result597, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %259 = insertelement <32 x float> poison, float %227, i64 0, !dbg !14
  %260 = insertelement <32 x float> %259, float %228, i64 1, !dbg !14
  %261 = insertelement <32 x float> %260, float %229, i64 2, !dbg !14
  %262 = insertelement <32 x float> %261, float %230, i64 3, !dbg !14
  %263 = insertelement <32 x float> %262, float %231, i64 4, !dbg !14
  %264 = insertelement <32 x float> %263, float %232, i64 5, !dbg !14
  %265 = insertelement <32 x float> %264, float %233, i64 6, !dbg !14
  %266 = insertelement <32 x float> %265, float %234, i64 7, !dbg !14
  %267 = insertelement <32 x float> %266, float %235, i64 8, !dbg !14
  %268 = insertelement <32 x float> %267, float %236, i64 9, !dbg !14
  %269 = insertelement <32 x float> %268, float %237, i64 10, !dbg !14
  %270 = insertelement <32 x float> %269, float %238, i64 11, !dbg !14
  %271 = insertelement <32 x float> %270, float %239, i64 12, !dbg !14
  %272 = insertelement <32 x float> %271, float %240, i64 13, !dbg !14
  %273 = insertelement <32 x float> %272, float %241, i64 14, !dbg !14
  %274 = insertelement <32 x float> %273, float %242, i64 15, !dbg !14
  %275 = insertelement <32 x float> %274, float %243, i64 16, !dbg !14
  %276 = insertelement <32 x float> %275, float %244, i64 17, !dbg !14
  %277 = insertelement <32 x float> %276, float %245, i64 18, !dbg !14
  %278 = insertelement <32 x float> %277, float %246, i64 19, !dbg !14
  %279 = insertelement <32 x float> %278, float %247, i64 20, !dbg !14
  %280 = insertelement <32 x float> %279, float %248, i64 21, !dbg !14
  %281 = insertelement <32 x float> %280, float %249, i64 22, !dbg !14
  %282 = insertelement <32 x float> %281, float %250, i64 23, !dbg !14
  %283 = insertelement <32 x float> %282, float %251, i64 24, !dbg !14
  %284 = insertelement <32 x float> %283, float %252, i64 25, !dbg !14
  %285 = insertelement <32 x float> %284, float %253, i64 26, !dbg !14
  %286 = insertelement <32 x float> %285, float %254, i64 27, !dbg !14
  %287 = insertelement <32 x float> %286, float %255, i64 28, !dbg !14
  %288 = insertelement <32 x float> %287, float %256, i64 29, !dbg !14
  %289 = insertelement <32 x float> %288, float %257, i64 30, !dbg !14
  %290 = insertelement <32 x float> %289, float %258, i64 31, !dbg !14
  br label %291, !dbg !14

291:                                              ; preds = %215, %216
  %292 = phi <32 x float> [ %290, %216 ], [ zeroinitializer, %215 ], !dbg !14
  br label %293, !dbg !14

293:                                              ; preds = %291
  %294 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %295 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %296 = fadd <4 x float> %294, %295, !dbg !25
  %297 = extractelement <4 x float> %296, i64 0, !dbg !25
  %298 = extractelement <4 x float> %296, i64 1, !dbg !25
  %299 = extractelement <4 x float> %296, i64 2, !dbg !25
  %300 = extractelement <4 x float> %296, i64 3, !dbg !25
  %301 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %302 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %303 = fadd <4 x float> %301, %302, !dbg !25
  %304 = extractelement <4 x float> %303, i64 0, !dbg !25
  %305 = extractelement <4 x float> %303, i64 1, !dbg !25
  %306 = extractelement <4 x float> %303, i64 2, !dbg !25
  %307 = extractelement <4 x float> %303, i64 3, !dbg !25
  %308 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %309 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %310 = fadd <4 x float> %308, %309, !dbg !25
  %311 = extractelement <4 x float> %310, i64 0, !dbg !25
  %312 = extractelement <4 x float> %310, i64 1, !dbg !25
  %313 = extractelement <4 x float> %310, i64 2, !dbg !25
  %314 = extractelement <4 x float> %310, i64 3, !dbg !25
  %315 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %316 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %317 = fadd <4 x float> %315, %316, !dbg !25
  %318 = extractelement <4 x float> %317, i64 0, !dbg !25
  %319 = extractelement <4 x float> %317, i64 1, !dbg !25
  %320 = extractelement <4 x float> %317, i64 2, !dbg !25
  %321 = extractelement <4 x float> %317, i64 3, !dbg !25
  %322 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %323 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %324 = fadd <4 x float> %322, %323, !dbg !25
  %325 = extractelement <4 x float> %324, i64 0, !dbg !25
  %326 = extractelement <4 x float> %324, i64 1, !dbg !25
  %327 = extractelement <4 x float> %324, i64 2, !dbg !25
  %328 = extractelement <4 x float> %324, i64 3, !dbg !25
  %329 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %330 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %331 = fadd <4 x float> %329, %330, !dbg !25
  %332 = extractelement <4 x float> %331, i64 0, !dbg !25
  %333 = extractelement <4 x float> %331, i64 1, !dbg !25
  %334 = extractelement <4 x float> %331, i64 2, !dbg !25
  %335 = extractelement <4 x float> %331, i64 3, !dbg !25
  %336 = shufflevector <32 x float> %58, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %337 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %338 = fadd <4 x float> %336, %337, !dbg !25
  %339 = extractelement <4 x float> %338, i64 0, !dbg !25
  %340 = extractelement <4 x float> %338, i64 1, !dbg !25
  %341 = extractelement <4 x float> %338, i64 2, !dbg !25
  %342 = extractelement <4 x float> %338, i64 3, !dbg !25
  %343 = shufflevector <32 x float> %292, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %344 = fadd <4 x float> %57, %343, !dbg !25
  %345 = extractelement <4 x float> %344, i64 0, !dbg !25
  %346 = extractelement <4 x float> %344, i64 1, !dbg !25
  %347 = extractelement <4 x float> %344, i64 2, !dbg !25
  %348 = extractelement <4 x float> %344, i64 3, !dbg !25
  %349 = call float @llvm.maxnum.f32(float %297, float 0xFFF0000000000000), !dbg !26
  %350 = call float @llvm.maxnum.f32(float %349, float %298), !dbg !26
  %351 = call float @llvm.maxnum.f32(float %350, float %299), !dbg !26
  %352 = call float @llvm.maxnum.f32(float %351, float %300), !dbg !26
  %353 = call float @llvm.maxnum.f32(float %352, float %304), !dbg !26
  %354 = call float @llvm.maxnum.f32(float %353, float %305), !dbg !26
  %355 = call float @llvm.maxnum.f32(float %354, float %306), !dbg !26
  %356 = call float @llvm.maxnum.f32(float %355, float %307), !dbg !26
  %357 = call float @llvm.maxnum.f32(float %356, float %311), !dbg !26
  %358 = call float @llvm.maxnum.f32(float %357, float %312), !dbg !26
  %359 = call float @llvm.maxnum.f32(float %358, float %313), !dbg !26
  %360 = call float @llvm.maxnum.f32(float %359, float %314), !dbg !26
  %361 = call float @llvm.maxnum.f32(float %360, float %318), !dbg !26
  %362 = call float @llvm.maxnum.f32(float %361, float %319), !dbg !26
  %363 = call float @llvm.maxnum.f32(float %362, float %320), !dbg !26
  %364 = call float @llvm.maxnum.f32(float %363, float %321), !dbg !26
  %365 = call float @llvm.maxnum.f32(float %364, float %325), !dbg !26
  %366 = call float @llvm.maxnum.f32(float %365, float %326), !dbg !26
  %367 = call float @llvm.maxnum.f32(float %366, float %327), !dbg !26
  %368 = call float @llvm.maxnum.f32(float %367, float %328), !dbg !26
  %369 = call float @llvm.maxnum.f32(float %368, float %332), !dbg !26
  %370 = call float @llvm.maxnum.f32(float %369, float %333), !dbg !26
  %371 = call float @llvm.maxnum.f32(float %370, float %334), !dbg !26
  %372 = call float @llvm.maxnum.f32(float %371, float %335), !dbg !26
  %373 = call float @llvm.maxnum.f32(float %372, float %339), !dbg !26
  %374 = call float @llvm.maxnum.f32(float %373, float %340), !dbg !26
  %375 = call float @llvm.maxnum.f32(float %374, float %341), !dbg !26
  %376 = call float @llvm.maxnum.f32(float %375, float %342), !dbg !26
  %377 = call float @llvm.maxnum.f32(float %376, float %345), !dbg !26
  %378 = call float @llvm.maxnum.f32(float %377, float %346), !dbg !26
  %379 = call float @llvm.maxnum.f32(float %378, float %347), !dbg !26
  %380 = call float @llvm.maxnum.f32(float %379, float %348), !dbg !26
  %381 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %382 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %381), !dbg !26
  %383 = add nuw nsw i32 %382, 64, !dbg !26
  %384 = xor i32 %382, 16, !dbg !26
  %385 = and i32 %383, 192, !dbg !26
  %386 = icmp samesign ult i32 %384, %385, !dbg !26
  %387 = select i1 %386, i32 %384, i32 %382, !dbg !26
  %388 = shl nuw nsw i32 %387, 2, !dbg !26
  %389 = bitcast float %380 to i32, !dbg !26
  %390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %388, i32 %389), !dbg !26
  %391 = bitcast i32 %390 to float, !dbg !26
  %392 = call float @llvm.maxnum.f32(float %380, float %391), !dbg !26
  %393 = xor i32 %382, 32, !dbg !26
  %394 = icmp samesign ult i32 %393, %385, !dbg !26
  %395 = select i1 %394, i32 %393, i32 %382, !dbg !26
  %396 = shl nuw nsw i32 %395, 2, !dbg !26
  %397 = bitcast float %392 to i32, !dbg !26
  %398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %396, i32 %397), !dbg !26
  %399 = bitcast i32 %398 to float, !dbg !26
  %400 = call float @llvm.maxnum.f32(float %392, float %399), !dbg !26
  %401 = insertelement <1 x float> poison, float %400, i64 0, !dbg !26
  %402 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %401, <1 x float> %39), !dbg !26
  %403 = shufflevector <1 x float> %402, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %404 = fsub <4 x float> %296, %403, !dbg !27
  %405 = fsub <4 x float> %303, %403, !dbg !27
  %406 = fsub <4 x float> %310, %403, !dbg !27
  %407 = fsub <4 x float> %317, %403, !dbg !27
  %408 = fsub <4 x float> %324, %403, !dbg !27
  %409 = fsub <4 x float> %331, %403, !dbg !27
  %410 = fsub <4 x float> %338, %403, !dbg !27
  %411 = fsub <4 x float> %344, %403, !dbg !27
  %412 = fsub <1 x float> %39, %402, !dbg !28
  %413 = extractelement <1 x float> %412, i64 0, !dbg !29
  %414 = call float @llvm.amdgcn.exp2.f32(float %413), !dbg !29
  %415 = insertelement <4 x float> poison, float %414, i64 0, !dbg !34
  %416 = shufflevector <4 x float> %415, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %417 = fmul <4 x float> %47, %416, !dbg !34
  %418 = fmul <4 x float> %46, %416, !dbg !34
  %419 = fmul <4 x float> %44, %416, !dbg !34
  %420 = fmul <4 x float> %43, %416, !dbg !34
  %421 = fmul <4 x float> %41, %416, !dbg !34
  %422 = fmul <4 x float> %40, %416, !dbg !34
  %423 = fmul <4 x float> %36, %416, !dbg !34
  %424 = fmul <4 x float> %35, %416, !dbg !34
  %425 = extractelement <1 x float> %38, i64 0, !dbg !30
  %426 = fmul float %425, %414, !dbg !30
  %427 = extractelement <4 x float> %404, i64 0, !dbg !31
  %428 = call float @llvm.amdgcn.exp2.f32(float %427), !dbg !31
  %429 = insertelement <4 x float> poison, float %428, i64 0, !dbg !33
  %430 = extractelement <4 x float> %404, i64 1, !dbg !31
  %431 = call float @llvm.amdgcn.exp2.f32(float %430), !dbg !31
  %432 = insertelement <4 x float> %429, float %431, i64 1, !dbg !33
  %433 = fadd float %428, %431, !dbg !32
  %434 = extractelement <4 x float> %404, i64 2, !dbg !31
  %435 = call float @llvm.amdgcn.exp2.f32(float %434), !dbg !31
  %436 = insertelement <4 x float> %432, float %435, i64 2, !dbg !33
  %437 = fadd float %433, %435, !dbg !32
  %438 = extractelement <4 x float> %404, i64 3, !dbg !31
  %439 = call float @llvm.amdgcn.exp2.f32(float %438), !dbg !31
  %440 = insertelement <4 x float> %436, float %439, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %440) #9, !dbg !33, !frisk.register_lifetimes !1
  %441 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %441) #9, !dbg !32, !frisk.register_lifetimes !1
  %442 = fadd float %437, %439, !dbg !32
  %443 = extractelement <4 x float> %405, i64 0, !dbg !31
  %444 = call float @llvm.amdgcn.exp2.f32(float %443), !dbg !31
  %445 = insertelement <4 x float> poison, float %444, i64 0, !dbg !33
  %446 = fadd float %442, %444, !dbg !32
  %447 = extractelement <4 x float> %405, i64 1, !dbg !31
  %448 = call float @llvm.amdgcn.exp2.f32(float %447), !dbg !31
  %449 = insertelement <4 x float> %445, float %448, i64 1, !dbg !33
  %450 = fadd float %446, %448, !dbg !32
  %451 = extractelement <4 x float> %405, i64 2, !dbg !31
  %452 = call float @llvm.amdgcn.exp2.f32(float %451), !dbg !31
  %453 = insertelement <4 x float> %449, float %452, i64 2, !dbg !33
  %454 = fadd float %450, %452, !dbg !32
  %455 = extractelement <4 x float> %405, i64 3, !dbg !31
  %456 = call float @llvm.amdgcn.exp2.f32(float %455), !dbg !31
  %457 = insertelement <4 x float> %453, float %456, i64 3, !dbg !33
  %math.result54 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %457) #9, !dbg !33, !frisk.register_lifetimes !1
  %458 = fptrunc <4 x float> %math.result54 to <4 x half>, !dbg !33
  %math.packed55 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %458) #9, !dbg !32, !frisk.register_lifetimes !1
  %459 = fadd float %454, %456, !dbg !32
  %460 = extractelement <4 x float> %406, i64 0, !dbg !31
  %461 = call float @llvm.amdgcn.exp2.f32(float %460), !dbg !31
  %462 = insertelement <4 x float> poison, float %461, i64 0, !dbg !33
  %463 = fadd float %459, %461, !dbg !32
  %464 = extractelement <4 x float> %406, i64 1, !dbg !31
  %465 = call float @llvm.amdgcn.exp2.f32(float %464), !dbg !31
  %466 = insertelement <4 x float> %462, float %465, i64 1, !dbg !33
  %467 = fadd float %463, %465, !dbg !32
  %468 = extractelement <4 x float> %406, i64 2, !dbg !31
  %469 = call float @llvm.amdgcn.exp2.f32(float %468), !dbg !31
  %470 = insertelement <4 x float> %466, float %469, i64 2, !dbg !33
  %471 = fadd float %467, %469, !dbg !32
  %472 = extractelement <4 x float> %406, i64 3, !dbg !31
  %473 = call float @llvm.amdgcn.exp2.f32(float %472), !dbg !31
  %474 = insertelement <4 x float> %470, float %473, i64 3, !dbg !33
  %math.result56 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %474) #9, !dbg !33, !frisk.register_lifetimes !1
  %475 = fptrunc <4 x float> %math.result56 to <4 x half>, !dbg !33
  %math.packed57 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %475) #9, !dbg !32, !frisk.register_lifetimes !1
  %476 = fadd float %471, %473, !dbg !32
  %477 = extractelement <4 x float> %407, i64 0, !dbg !31
  %478 = call float @llvm.amdgcn.exp2.f32(float %477), !dbg !31
  %479 = insertelement <4 x float> poison, float %478, i64 0, !dbg !33
  %480 = fadd float %476, %478, !dbg !32
  %481 = extractelement <4 x float> %407, i64 1, !dbg !31
  %482 = call float @llvm.amdgcn.exp2.f32(float %481), !dbg !31
  %483 = insertelement <4 x float> %479, float %482, i64 1, !dbg !33
  %484 = fadd float %480, %482, !dbg !32
  %485 = extractelement <4 x float> %407, i64 2, !dbg !31
  %486 = call float @llvm.amdgcn.exp2.f32(float %485), !dbg !31
  %487 = insertelement <4 x float> %483, float %486, i64 2, !dbg !33
  %488 = fadd float %484, %486, !dbg !32
  %489 = extractelement <4 x float> %407, i64 3, !dbg !31
  %490 = call float @llvm.amdgcn.exp2.f32(float %489), !dbg !31
  %491 = insertelement <4 x float> %487, float %490, i64 3, !dbg !33
  %math.result58 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %491) #9, !dbg !33, !frisk.register_lifetimes !1
  %492 = fptrunc <4 x float> %math.result58 to <4 x half>, !dbg !33
  %math.packed59 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %492) #9, !dbg !32, !frisk.register_lifetimes !1
  %493 = fadd float %488, %490, !dbg !32
  %494 = extractelement <4 x float> %408, i64 0, !dbg !31
  %495 = call float @llvm.amdgcn.exp2.f32(float %494), !dbg !31
  %496 = insertelement <4 x float> poison, float %495, i64 0, !dbg !33
  %497 = fadd float %493, %495, !dbg !32
  %498 = extractelement <4 x float> %408, i64 1, !dbg !31
  %499 = call float @llvm.amdgcn.exp2.f32(float %498), !dbg !31
  %500 = insertelement <4 x float> %496, float %499, i64 1, !dbg !33
  %501 = fadd float %497, %499, !dbg !32
  %502 = extractelement <4 x float> %408, i64 2, !dbg !31
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !31
  %504 = insertelement <4 x float> %500, float %503, i64 2, !dbg !33
  %505 = fadd float %501, %503, !dbg !32
  %506 = extractelement <4 x float> %408, i64 3, !dbg !31
  %507 = call float @llvm.amdgcn.exp2.f32(float %506), !dbg !31
  %508 = insertelement <4 x float> %504, float %507, i64 3, !dbg !33
  %math.result60 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %508) #9, !dbg !33, !frisk.register_lifetimes !1
  %509 = fptrunc <4 x float> %math.result60 to <4 x half>, !dbg !33
  %math.packed61 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %509) #9, !dbg !32, !frisk.register_lifetimes !1
  %510 = fadd float %505, %507, !dbg !32
  %511 = extractelement <4 x float> %409, i64 0, !dbg !31
  %512 = call float @llvm.amdgcn.exp2.f32(float %511), !dbg !31
  %513 = insertelement <4 x float> poison, float %512, i64 0, !dbg !33
  %514 = fadd float %510, %512, !dbg !32
  %515 = extractelement <4 x float> %409, i64 1, !dbg !31
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !31
  %517 = insertelement <4 x float> %513, float %516, i64 1, !dbg !33
  %518 = fadd float %514, %516, !dbg !32
  %519 = extractelement <4 x float> %409, i64 2, !dbg !31
  %520 = call float @llvm.amdgcn.exp2.f32(float %519), !dbg !31
  %521 = insertelement <4 x float> %517, float %520, i64 2, !dbg !33
  %522 = fadd float %518, %520, !dbg !32
  %523 = extractelement <4 x float> %409, i64 3, !dbg !31
  %524 = call float @llvm.amdgcn.exp2.f32(float %523), !dbg !31
  %525 = insertelement <4 x float> %521, float %524, i64 3, !dbg !33
  %math.result62 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %525) #9, !dbg !33, !frisk.register_lifetimes !1
  %526 = fptrunc <4 x float> %math.result62 to <4 x half>, !dbg !33
  %math.packed63 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %526) #9, !dbg !32, !frisk.register_lifetimes !1
  %527 = fadd float %522, %524, !dbg !32
  %528 = extractelement <4 x float> %410, i64 0, !dbg !31
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !31
  %530 = insertelement <4 x float> poison, float %529, i64 0, !dbg !33
  %531 = fadd float %527, %529, !dbg !32
  %532 = extractelement <4 x float> %410, i64 1, !dbg !31
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !31
  %534 = insertelement <4 x float> %530, float %533, i64 1, !dbg !33
  %535 = fadd float %531, %533, !dbg !32
  %536 = extractelement <4 x float> %410, i64 2, !dbg !31
  %537 = call float @llvm.amdgcn.exp2.f32(float %536), !dbg !31
  %538 = insertelement <4 x float> %534, float %537, i64 2, !dbg !33
  %539 = fadd float %535, %537, !dbg !32
  %540 = extractelement <4 x float> %410, i64 3, !dbg !31
  %541 = call float @llvm.amdgcn.exp2.f32(float %540), !dbg !31
  %542 = insertelement <4 x float> %538, float %541, i64 3, !dbg !33
  %math.result64 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #9, !dbg !33, !frisk.register_lifetimes !1
  %543 = fptrunc <4 x float> %math.result64 to <4 x half>, !dbg !33
  %math.packed65 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #9, !dbg !32, !frisk.register_lifetimes !1
  %544 = fadd float %539, %541, !dbg !32
  %545 = extractelement <4 x float> %411, i64 0, !dbg !31
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !31
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !33
  %548 = fadd float %544, %546, !dbg !32
  %549 = extractelement <4 x float> %411, i64 1, !dbg !31
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !31
  %551 = insertelement <4 x float> %547, float %550, i64 1, !dbg !33
  %552 = fadd float %548, %550, !dbg !32
  %553 = extractelement <4 x float> %411, i64 2, !dbg !31
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !31
  %555 = insertelement <4 x float> %551, float %554, i64 2, !dbg !33
  %556 = fadd float %552, %554, !dbg !32
  %557 = extractelement <4 x float> %411, i64 3, !dbg !31
  %558 = call float @llvm.amdgcn.exp2.f32(float %557), !dbg !31
  %559 = insertelement <4 x float> %555, float %558, i64 3, !dbg !33
  %math.result66 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %559) #9, !dbg !33, !frisk.register_lifetimes !1
  %560 = fptrunc <4 x float> %math.result66 to <4 x half>, !dbg !33
  %math.packed67 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %560) #9, !dbg !32, !frisk.register_lifetimes !1
  %561 = fadd float %556, %558, !dbg !32
  %562 = bitcast float %561 to i32, !dbg !32
  %563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %388, i32 %562), !dbg !32
  %564 = bitcast i32 %563 to float, !dbg !32
  %565 = fadd float %561, %564, !dbg !32
  %566 = bitcast float %565 to i32, !dbg !32
  %567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %396, i32 %566), !dbg !32
  %568 = bitcast i32 %567 to float, !dbg !32
  %569 = fadd float %565, %568, !dbg !32
  %570 = fadd float %569, %426, !dbg !32
  %571 = insertelement <1 x float> poison, float %570, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread257 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local258 = shl nuw nsw i32 %address.thread257, 3, !dbg !15
  %address.local259 = and i32 %address.local258, 120, !dbg !15
  %address.local260 = shl nuw nsw i32 %address.thread257, 5, !dbg !15
  %address.local261 = and i32 %address.local260, 1536, !dbg !15
  %address.local262 = or disjoint i32 %address.local259, %address.local261, !dbg !15
  %address.local263 = zext nneg i32 %address.local262 to i64, !dbg !15
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local263, !dbg !15
  %572 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local264, i32 0) #0, !dbg !15
  %573 = bitcast <4 x i32> %572 to <8 x half>, !dbg !15
  %address.local265 = zext nneg i32 %address.thread257 to i64, !dbg !13
  %address.local266 = and i64 %address.local265, 60, !dbg !15
  %address.local267 = add nuw nsw i64 %address.local266, %address.local265, !dbg !15
  %address.local268 = shl nuw nsw i64 %address.local267, 3, !dbg !15
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !15
  %address.local270 = and i64 %address.local265, 63, !dbg !13
  %address.local271 = lshr i64 %address.local270, 4, !dbg !15
  %address.local272 = mul nuw nsw i64 %address.local271, 384, !dbg !15
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local269, i64 %address.local272, !dbg !15
  %address.local274 = lshr i64 %address.local265, 2, !dbg !8
  %address.local275 = mul nsw i64 %address.local274, -32, !dbg !15
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local275, !dbg !15
  %address.local277 = getelementptr i8, ptr addrspace(3) %address.local276, i64 256, !dbg !15
  %574 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local277, i32 0) #0, !dbg !15
  %575 = bitcast <4 x i32> %574 to <8 x half>, !dbg !15
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !15
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local272, !dbg !15
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local275, !dbg !15
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 512, !dbg !15
  %576 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local281, i32 0) #0, !dbg !15
  %577 = bitcast <4 x i32> %576 to <8 x half>, !dbg !15
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !15
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) %address.local282, i64 %address.local272, !dbg !15
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local275, !dbg !15
  %address.local285 = getelementptr i8, ptr addrspace(3) %address.local284, i64 768, !dbg !15
  %578 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local285, i32 0) #0, !dbg !15
  %579 = bitcast <4 x i32> %578 to <8 x half>, !dbg !15
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local263, !dbg !15
  %address.local287 = getelementptr i8, ptr addrspace(3) %address.local286, i64 4096, !dbg !15
  %580 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local287, i32 0) #0, !dbg !15
  %581 = bitcast <4 x i32> %580 to <8 x half>, !dbg !15
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !15
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) %address.local288, i64 %address.local272, !dbg !15
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local275, !dbg !15
  %address.local291 = getelementptr i8, ptr addrspace(3) %address.local290, i64 4352, !dbg !15
  %582 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local291, i32 0) #0, !dbg !15
  %583 = bitcast <4 x i32> %582 to <8 x half>, !dbg !15
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !15
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local272, !dbg !15
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) %address.local293, i64 %address.local275, !dbg !15
  %address.local295 = getelementptr i8, ptr addrspace(3) %address.local294, i64 4608, !dbg !15
  %584 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local295, i32 0) #0, !dbg !15
  %585 = bitcast <4 x i32> %584 to <8 x half>, !dbg !15
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !15
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) %address.local296, i64 %address.local272, !dbg !15
  %address.local298 = getelementptr [2 x i8], ptr addrspace(3) %address.local297, i64 %address.local275, !dbg !15
  %address.local299 = getelementptr i8, ptr addrspace(3) %address.local298, i64 4864, !dbg !15
  %586 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local299, i32 0) #0, !dbg !15
  %587 = bitcast <4 x i32> %586 to <8 x half>, !dbg !15
  %588 = add nsw i64 %52, %6, !dbg !17
  %589 = shl i64 %588, 19, !dbg !17
  %590 = and i64 %589, 16252928, !dbg !17
  %591 = add i64 %590, %181, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %592 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %591, !dbg !17
  %593 = getelementptr i8, ptr addrspace(1) %592, i64 64, !dbg !17
  %address.local300 = shl nuw nsw i64 %address.local265, 3, !dbg !12
  %address.local301 = and i64 %address.local300, 1536, !dbg !16
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local301, !dbg !16
  %address.local303 = ptrtoint ptr addrspace(3) %address.local302 to i64, !dbg !16
  %address.local304 = trunc i64 %address.local303 to i32, !dbg !16
  %594 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !17
  %595 = zext i32 %594 to i64, !dbg !17
  %596 = inttoptr i64 %595 to ptr addrspace(3), !dbg !17
  %address.local305 = shl nuw nsw i64 %address.local265, 5, !dbg !17
  %address.local306 = and i64 %address.local305, 8064, !dbg !17
  %address.local307 = shl nuw nsw i64 %address.local265, 3, !dbg !17
  %address.local308 = and i64 %address.local307, 24, !dbg !17
  %address.local309 = or disjoint i64 %address.local306, %address.local308, !dbg !17
  %address.local310 = getelementptr [2 x i8], ptr addrspace(1) %593, i64 %address.local309, !dbg !17
  %597 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %598 = ptrtoint ptr addrspace(1) %address.local310 to i64, !dbg !17
  %599 = sub i64 %598, %597, !dbg !17
  %dma.buffer.offset619 = trunc i64 %599 to i32, !dbg !17
  %600 = trunc i64 %597 to i32, !dbg !17
  %601 = insertelement <4 x i32> poison, i32 %600, i64 0, !dbg !17
  %602 = lshr i64 %597, 32, !dbg !17
  %603 = trunc i64 %602 to i32, !dbg !17
  %604 = insertelement <4 x i32> %601, i32 %603, i64 1, !dbg !17
  %605 = insertelement <4 x i32> %604, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc620 = insertelement <4 x i32> %605, i32 131072, i64 3, !dbg !17
  %606 = ptrtoint ptr addrspace(3) %596 to i32, !dbg !17
  %dma.buffer.lds621 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %606), !dbg !17
  %607 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds621, i32 %dma.buffer.offset619, <4 x i32> %dma.buffer.rsrc620) #10, !dbg !17
  %address.local311 = or disjoint i64 %address.local265, 256, !dbg !16
  %address.local312 = shl nuw nsw i64 %address.local311, 3, !dbg !16
  %address.local313 = and i64 %address.local312, 3584, !dbg !16
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local313, !dbg !16
  %address.local315 = ptrtoint ptr addrspace(3) %address.local314 to i64, !dbg !16
  %address.local316 = trunc i64 %address.local315 to i32, !dbg !16
  %608 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local316), !dbg !17
  %609 = zext i32 %608 to i64, !dbg !17
  %610 = inttoptr i64 %609 to ptr addrspace(3), !dbg !17
  %address.local317 = shl nuw nsw i64 %address.local311, 5, !dbg !17
  %address.local318 = and i64 %address.local317, 16256, !dbg !17
  %address.local319 = shl nuw nsw i64 %address.local265, 3, !dbg !17
  %address.local320 = and i64 %address.local319, 24, !dbg !17
  %address.local321 = or disjoint i64 %address.local318, %address.local320, !dbg !17
  %address.local322 = getelementptr [2 x i8], ptr addrspace(1) %593, i64 %address.local321, !dbg !17
  %611 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %612 = ptrtoint ptr addrspace(1) %address.local322 to i64, !dbg !17
  %613 = sub i64 %612, %611, !dbg !17
  %dma.buffer.offset616 = trunc i64 %613 to i32, !dbg !17
  %614 = trunc i64 %611 to i32, !dbg !17
  %615 = insertelement <4 x i32> poison, i32 %614, i64 0, !dbg !17
  %616 = lshr i64 %611, 32, !dbg !17
  %617 = trunc i64 %616 to i32, !dbg !17
  %618 = insertelement <4 x i32> %615, i32 %617, i64 1, !dbg !17
  %619 = insertelement <4 x i32> %618, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc617 = insertelement <4 x i32> %619, i32 131072, i64 3, !dbg !17
  %620 = ptrtoint ptr addrspace(3) %610 to i32, !dbg !17
  %dma.buffer.lds618 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %620), !dbg !17
  %621 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds618, i32 %dma.buffer.offset616, <4 x i32> %dma.buffer.rsrc617) #10, !dbg !17
  %622 = shufflevector <8 x half> %573, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %623 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %622, <4 x float> %424) #9, !dbg !15
  %624 = shufflevector <8 x half> %575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %624, <4 x float> %623) #9, !dbg !15
  %626 = shufflevector <8 x half> %577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %626, <4 x float> %625) #9, !dbg !15
  %628 = shufflevector <8 x half> %579, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %628, <4 x float> %627) #9, !dbg !15
  %630 = shufflevector <8 x half> %581, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %630, <4 x float> %629) #9, !dbg !15
  %632 = shufflevector <8 x half> %583, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %632, <4 x float> %631) #9, !dbg !15
  %634 = shufflevector <8 x half> %585, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %635 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %634, <4 x float> %633) #9, !dbg !15
  %636 = shufflevector <8 x half> %587, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %636, <4 x float> %635) #9, !dbg !15
  %638 = shufflevector <8 x half> %573, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %638, <4 x float> %423) #9, !dbg !15
  %640 = shufflevector <8 x half> %575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %640, <4 x float> %639) #9, !dbg !15
  %642 = shufflevector <8 x half> %577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %642, <4 x float> %641) #9, !dbg !15
  %644 = shufflevector <8 x half> %579, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %644, <4 x float> %643) #9, !dbg !15
  %646 = shufflevector <8 x half> %581, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %647 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %646, <4 x float> %645) #9, !dbg !15
  %648 = shufflevector <8 x half> %583, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %648, <4 x float> %647) #9, !dbg !15
  %650 = shufflevector <8 x half> %585, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %651 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %650, <4 x float> %649) #9, !dbg !15
  %652 = shufflevector <8 x half> %587, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %652, <4 x float> %651) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %654 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local264, i32 0) #0, !dbg !15
  %655 = bitcast <4 x i32> %654 to <8 x half>, !dbg !15
  %656 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local277, i32 0) #0, !dbg !15
  %657 = bitcast <4 x i32> %656 to <8 x half>, !dbg !15
  %658 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local281, i32 0) #0, !dbg !15
  %659 = bitcast <4 x i32> %658 to <8 x half>, !dbg !15
  %660 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local285, i32 0) #0, !dbg !15
  %661 = bitcast <4 x i32> %660 to <8 x half>, !dbg !15
  %662 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local287, i32 0) #0, !dbg !15
  %663 = bitcast <4 x i32> %662 to <8 x half>, !dbg !15
  %664 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local291, i32 0) #0, !dbg !15
  %665 = bitcast <4 x i32> %664 to <8 x half>, !dbg !15
  %666 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local295, i32 0) #0, !dbg !15
  %667 = bitcast <4 x i32> %666 to <8 x half>, !dbg !15
  %668 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local299, i32 0) #0, !dbg !15
  %669 = bitcast <4 x i32> %668 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %670 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %591, !dbg !17
  %671 = getelementptr i8, ptr addrspace(1) %670, i64 128, !dbg !17
  %672 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !17
  %673 = zext i32 %672 to i64, !dbg !17
  %674 = inttoptr i64 %673 to ptr addrspace(3), !dbg !17
  %address.local323 = getelementptr [2 x i8], ptr addrspace(1) %671, i64 %address.local309, !dbg !17
  %675 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %676 = ptrtoint ptr addrspace(1) %address.local323 to i64, !dbg !17
  %677 = sub i64 %676, %675, !dbg !17
  %dma.buffer.offset613 = trunc i64 %677 to i32, !dbg !17
  %678 = trunc i64 %675 to i32, !dbg !17
  %679 = insertelement <4 x i32> poison, i32 %678, i64 0, !dbg !17
  %680 = lshr i64 %675, 32, !dbg !17
  %681 = trunc i64 %680 to i32, !dbg !17
  %682 = insertelement <4 x i32> %679, i32 %681, i64 1, !dbg !17
  %683 = insertelement <4 x i32> %682, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc614 = insertelement <4 x i32> %683, i32 131072, i64 3, !dbg !17
  %684 = ptrtoint ptr addrspace(3) %674 to i32, !dbg !17
  %dma.buffer.lds615 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %684), !dbg !17
  %685 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds615, i32 %dma.buffer.offset613, <4 x i32> %dma.buffer.rsrc614) #10, !dbg !17
  %686 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local316), !dbg !17
  %687 = zext i32 %686 to i64, !dbg !17
  %688 = inttoptr i64 %687 to ptr addrspace(3), !dbg !17
  %address.local324 = getelementptr [2 x i8], ptr addrspace(1) %671, i64 %address.local321, !dbg !17
  %689 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %690 = ptrtoint ptr addrspace(1) %address.local324 to i64, !dbg !17
  %691 = sub i64 %690, %689, !dbg !17
  %dma.buffer.offset610 = trunc i64 %691 to i32, !dbg !17
  %692 = trunc i64 %689 to i32, !dbg !17
  %693 = insertelement <4 x i32> poison, i32 %692, i64 0, !dbg !17
  %694 = lshr i64 %689, 32, !dbg !17
  %695 = trunc i64 %694 to i32, !dbg !17
  %696 = insertelement <4 x i32> %693, i32 %695, i64 1, !dbg !17
  %697 = insertelement <4 x i32> %696, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc611 = insertelement <4 x i32> %697, i32 131072, i64 3, !dbg !17
  %698 = ptrtoint ptr addrspace(3) %688 to i32, !dbg !17
  %dma.buffer.lds612 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %698), !dbg !17
  %699 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds612, i32 %dma.buffer.offset610, <4 x i32> %dma.buffer.rsrc611) #10, !dbg !17
  %700 = shufflevector <8 x half> %655, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %700, <4 x float> %422) #9, !dbg !15
  %702 = shufflevector <8 x half> %657, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %702, <4 x float> %701) #9, !dbg !15
  %704 = shufflevector <8 x half> %659, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %705 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %704, <4 x float> %703) #9, !dbg !15
  %706 = shufflevector <8 x half> %661, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %707 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %706, <4 x float> %705) #9, !dbg !15
  %708 = shufflevector <8 x half> %663, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %708, <4 x float> %707) #9, !dbg !15
  %710 = shufflevector <8 x half> %665, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %710, <4 x float> %709) #9, !dbg !15
  %712 = shufflevector <8 x half> %667, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %713 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %712, <4 x float> %711) #9, !dbg !15
  %714 = shufflevector <8 x half> %669, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %715 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %714, <4 x float> %713) #9, !dbg !15
  %716 = shufflevector <8 x half> %655, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %716, <4 x float> %421) #9, !dbg !15
  %718 = shufflevector <8 x half> %657, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %719 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %718, <4 x float> %717) #9, !dbg !15
  %720 = shufflevector <8 x half> %659, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %721 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %720, <4 x float> %719) #9, !dbg !15
  %722 = shufflevector <8 x half> %661, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %722, <4 x float> %721) #9, !dbg !15
  %724 = shufflevector <8 x half> %663, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %725 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %724, <4 x float> %723) #9, !dbg !15
  %726 = shufflevector <8 x half> %665, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %726, <4 x float> %725) #9, !dbg !15
  %728 = shufflevector <8 x half> %667, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %728, <4 x float> %727) #9, !dbg !15
  %730 = shufflevector <8 x half> %669, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %730, <4 x float> %729) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %732 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local264, i32 0) #0, !dbg !15
  %733 = bitcast <4 x i32> %732 to <8 x half>, !dbg !15
  %734 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local277, i32 0) #0, !dbg !15
  %735 = bitcast <4 x i32> %734 to <8 x half>, !dbg !15
  %736 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local281, i32 0) #0, !dbg !15
  %737 = bitcast <4 x i32> %736 to <8 x half>, !dbg !15
  %738 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local285, i32 0) #0, !dbg !15
  %739 = bitcast <4 x i32> %738 to <8 x half>, !dbg !15
  %740 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local287, i32 0) #0, !dbg !15
  %741 = bitcast <4 x i32> %740 to <8 x half>, !dbg !15
  %742 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local291, i32 0) #0, !dbg !15
  %743 = bitcast <4 x i32> %742 to <8 x half>, !dbg !15
  %744 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local295, i32 0) #0, !dbg !15
  %745 = bitcast <4 x i32> %744 to <8 x half>, !dbg !15
  %746 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local299, i32 0) #0, !dbg !15
  %747 = bitcast <4 x i32> %746 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %748 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %591, !dbg !17
  %749 = getelementptr i8, ptr addrspace(1) %748, i64 192, !dbg !17
  %750 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !17
  %751 = zext i32 %750 to i64, !dbg !17
  %752 = inttoptr i64 %751 to ptr addrspace(3), !dbg !17
  %address.local325 = getelementptr [2 x i8], ptr addrspace(1) %749, i64 %address.local309, !dbg !17
  %753 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %754 = ptrtoint ptr addrspace(1) %address.local325 to i64, !dbg !17
  %755 = sub i64 %754, %753, !dbg !17
  %dma.buffer.offset607 = trunc i64 %755 to i32, !dbg !17
  %756 = trunc i64 %753 to i32, !dbg !17
  %757 = insertelement <4 x i32> poison, i32 %756, i64 0, !dbg !17
  %758 = lshr i64 %753, 32, !dbg !17
  %759 = trunc i64 %758 to i32, !dbg !17
  %760 = insertelement <4 x i32> %757, i32 %759, i64 1, !dbg !17
  %761 = insertelement <4 x i32> %760, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc608 = insertelement <4 x i32> %761, i32 131072, i64 3, !dbg !17
  %762 = ptrtoint ptr addrspace(3) %752 to i32, !dbg !17
  %dma.buffer.lds609 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %762), !dbg !17
  %763 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds609, i32 %dma.buffer.offset607, <4 x i32> %dma.buffer.rsrc608) #10, !dbg !17
  %764 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local316), !dbg !17
  %765 = zext i32 %764 to i64, !dbg !17
  %766 = inttoptr i64 %765 to ptr addrspace(3), !dbg !17
  %address.local326 = getelementptr [2 x i8], ptr addrspace(1) %749, i64 %address.local321, !dbg !17
  %767 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %768 = ptrtoint ptr addrspace(1) %address.local326 to i64, !dbg !17
  %769 = sub i64 %768, %767, !dbg !17
  %dma.buffer.offset604 = trunc i64 %769 to i32, !dbg !17
  %770 = trunc i64 %767 to i32, !dbg !17
  %771 = insertelement <4 x i32> poison, i32 %770, i64 0, !dbg !17
  %772 = lshr i64 %767, 32, !dbg !17
  %773 = trunc i64 %772 to i32, !dbg !17
  %774 = insertelement <4 x i32> %771, i32 %773, i64 1, !dbg !17
  %775 = insertelement <4 x i32> %774, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc605 = insertelement <4 x i32> %775, i32 131072, i64 3, !dbg !17
  %776 = ptrtoint ptr addrspace(3) %766 to i32, !dbg !17
  %dma.buffer.lds606 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %776), !dbg !17
  %777 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds606, i32 %dma.buffer.offset604, <4 x i32> %dma.buffer.rsrc605) #10, !dbg !17
  %778 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %778, <4 x float> %420) #9, !dbg !15
  %780 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %780, <4 x float> %779) #9, !dbg !15
  %782 = shufflevector <8 x half> %737, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %782, <4 x float> %781) #9, !dbg !15
  %784 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %784, <4 x float> %783) #9, !dbg !15
  %786 = shufflevector <8 x half> %741, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %786, <4 x float> %785) #9, !dbg !15
  %788 = shufflevector <8 x half> %743, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %788, <4 x float> %787) #9, !dbg !15
  %790 = shufflevector <8 x half> %745, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %790, <4 x float> %789) #9, !dbg !15
  %792 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %792, <4 x float> %791) #9, !dbg !15
  %794 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %794, <4 x float> %419) #9, !dbg !15
  %796 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %796, <4 x float> %795) #9, !dbg !15
  %798 = shufflevector <8 x half> %737, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %798, <4 x float> %797) #9, !dbg !15
  %800 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %800, <4 x float> %799) #9, !dbg !15
  %802 = shufflevector <8 x half> %741, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %802, <4 x float> %801) #9, !dbg !15
  %804 = shufflevector <8 x half> %743, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %805 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %804, <4 x float> %803) #9, !dbg !15
  %806 = shufflevector <8 x half> %745, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %807 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %806, <4 x float> %805) #9, !dbg !15
  %808 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %808, <4 x float> %807) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %810 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local264, i32 0) #0, !dbg !15
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !15
  %812 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local277, i32 0) #0, !dbg !15
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !15
  %814 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local281, i32 0) #0, !dbg !15
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !15
  %816 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local285, i32 0) #0, !dbg !15
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !15
  %818 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local287, i32 0) #0, !dbg !15
  %819 = bitcast <4 x i32> %818 to <8 x half>, !dbg !15
  %820 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local291, i32 0) #0, !dbg !15
  %821 = bitcast <4 x i32> %820 to <8 x half>, !dbg !15
  %822 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local295, i32 0) #0, !dbg !15
  %823 = bitcast <4 x i32> %822 to <8 x half>, !dbg !15
  %824 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local299, i32 0) #0, !dbg !15
  %825 = bitcast <4 x i32> %824 to <8 x half>, !dbg !15
  %826 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %826, <4 x float> %418) #9, !dbg !15
  %828 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %828, <4 x float> %827) #9, !dbg !15
  %830 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %830, <4 x float> %829) #9, !dbg !15
  %832 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %832, <4 x float> %831) #9, !dbg !15
  %834 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %834, <4 x float> %833) #9, !dbg !15
  %836 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %836, <4 x float> %835) #9, !dbg !15
  %838 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %838, <4 x float> %837) #9, !dbg !15
  %840 = shufflevector <8 x half> %825, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %840, <4 x float> %839) #9, !dbg !15
  %842 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %842, <4 x float> %417) #9, !dbg !15
  %844 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed55, <4 x half> %844, <4 x float> %843) #9, !dbg !15
  %846 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed57, <4 x half> %846, <4 x float> %845) #9, !dbg !15
  %848 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed59, <4 x half> %848, <4 x float> %847) #9, !dbg !15
  %850 = shufflevector <8 x half> %819, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %850, <4 x float> %849) #9, !dbg !15
  %852 = shufflevector <8 x half> %821, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %852, <4 x float> %851) #9, !dbg !15
  %854 = shufflevector <8 x half> %823, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %854, <4 x float> %853) #9, !dbg !15
  %856 = shufflevector <8 x half> %825, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %856, <4 x float> %855) #9, !dbg !15
  %858 = shufflevector <4 x float> %637, <4 x float> %653, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %859 = shufflevector <4 x float> %715, <4 x float> %731, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %860 = shufflevector <4 x float> %793, <4 x float> %809, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %861 = shufflevector <4 x float> %841, <4 x float> %857, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %862 = add nuw nsw i64 %34, 1, !dbg !7
  br label %33, !dbg !7

863:                                              ; preds = %33
  %864 = and i32 %21, 16, !dbg !7
  %.not = icmp eq i32 %864, 0, !dbg !7
  %865 = and i32 %21, 32, !dbg !7
  %.not1 = icmp eq i32 %865, 0, !dbg !7
  %866 = extractelement <8 x float> %48, i64 0, !dbg !7
  %867 = extractelement <8 x float> %48, i64 1, !dbg !7
  %868 = extractelement <8 x float> %48, i64 2, !dbg !7
  %869 = extractelement <8 x float> %48, i64 3, !dbg !7
  %870 = select i1 %.not, float %867, float %866, !dbg !7
  %871 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %872 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %871), !dbg !7
  %873 = add nuw nsw i32 %872, 64, !dbg !7
  %874 = xor i32 %872, 16, !dbg !7
  %875 = and i32 %873, 192, !dbg !7
  %876 = icmp samesign ult i32 %874, %875, !dbg !7
  %877 = select i1 %876, i32 %874, i32 %872, !dbg !7
  %878 = shl nuw nsw i32 %877, 2, !dbg !7
  %879 = bitcast float %870 to i32, !dbg !7
  %880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %879), !dbg !7
  %881 = bitcast i32 %880 to float, !dbg !7
  %882 = select i1 %.not, float %866, float %881, !dbg !7
  %883 = select i1 %.not, float %881, float %867, !dbg !7
  %884 = select i1 %.not, float %869, float %868, !dbg !7
  %885 = bitcast float %884 to i32, !dbg !7
  %886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %885), !dbg !7
  %887 = bitcast i32 %886 to float, !dbg !7
  %888 = select i1 %.not, float %868, float %887, !dbg !7
  %889 = select i1 %.not, float %887, float %869, !dbg !7
  %890 = select i1 %.not1, float %888, float %882, !dbg !7
  %891 = xor i32 %872, 32, !dbg !7
  %892 = icmp samesign ult i32 %891, %875, !dbg !7
  %893 = select i1 %892, i32 %891, i32 %872, !dbg !7
  %894 = shl nuw nsw i32 %893, 2, !dbg !7
  %895 = bitcast float %890 to i32, !dbg !7
  %896 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %895), !dbg !7
  %897 = bitcast i32 %896 to float, !dbg !7
  %898 = select i1 %.not1, float %882, float %897, !dbg !7
  %899 = select i1 %.not1, float %897, float %888, !dbg !7
  %900 = select i1 %.not1, float %889, float %883, !dbg !7
  %901 = bitcast float %900 to i32, !dbg !7
  %902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %901), !dbg !7
  %903 = bitcast i32 %902 to float, !dbg !7
  %904 = select i1 %.not1, float %883, float %903, !dbg !7
  %905 = select i1 %.not1, float %903, float %889, !dbg !7
  %906 = extractelement <8 x float> %45, i64 0, !dbg !7
  %907 = extractelement <8 x float> %45, i64 1, !dbg !7
  %908 = extractelement <8 x float> %45, i64 2, !dbg !7
  %909 = extractelement <8 x float> %45, i64 3, !dbg !7
  %910 = select i1 %.not, float %907, float %906, !dbg !7
  %911 = bitcast float %910 to i32, !dbg !7
  %912 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %911), !dbg !7
  %913 = bitcast i32 %912 to float, !dbg !7
  %914 = select i1 %.not, float %906, float %913, !dbg !7
  %915 = select i1 %.not, float %913, float %907, !dbg !7
  %916 = select i1 %.not, float %909, float %908, !dbg !7
  %917 = bitcast float %916 to i32, !dbg !7
  %918 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %917), !dbg !7
  %919 = bitcast i32 %918 to float, !dbg !7
  %920 = select i1 %.not, float %908, float %919, !dbg !7
  %921 = select i1 %.not, float %919, float %909, !dbg !7
  %922 = select i1 %.not1, float %920, float %914, !dbg !7
  %923 = bitcast float %922 to i32, !dbg !7
  %924 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %923), !dbg !7
  %925 = bitcast i32 %924 to float, !dbg !7
  %926 = select i1 %.not1, float %914, float %925, !dbg !7
  %927 = select i1 %.not1, float %925, float %920, !dbg !7
  %928 = select i1 %.not1, float %921, float %915, !dbg !7
  %929 = bitcast float %928 to i32, !dbg !7
  %930 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %929), !dbg !7
  %931 = bitcast i32 %930 to float, !dbg !7
  %932 = select i1 %.not1, float %915, float %931, !dbg !7
  %933 = select i1 %.not1, float %931, float %921, !dbg !7
  %934 = extractelement <8 x float> %42, i64 0, !dbg !7
  %935 = extractelement <8 x float> %42, i64 1, !dbg !7
  %936 = extractelement <8 x float> %42, i64 2, !dbg !7
  %937 = extractelement <8 x float> %42, i64 3, !dbg !7
  %938 = select i1 %.not, float %935, float %934, !dbg !7
  %939 = bitcast float %938 to i32, !dbg !7
  %940 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %939), !dbg !7
  %941 = bitcast i32 %940 to float, !dbg !7
  %942 = select i1 %.not, float %934, float %941, !dbg !7
  %943 = select i1 %.not, float %941, float %935, !dbg !7
  %944 = select i1 %.not, float %937, float %936, !dbg !7
  %945 = bitcast float %944 to i32, !dbg !7
  %946 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %945), !dbg !7
  %947 = bitcast i32 %946 to float, !dbg !7
  %948 = select i1 %.not, float %936, float %947, !dbg !7
  %949 = select i1 %.not, float %947, float %937, !dbg !7
  %950 = select i1 %.not1, float %948, float %942, !dbg !7
  %951 = bitcast float %950 to i32, !dbg !7
  %952 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %951), !dbg !7
  %953 = bitcast i32 %952 to float, !dbg !7
  %954 = select i1 %.not1, float %942, float %953, !dbg !7
  %955 = select i1 %.not1, float %953, float %948, !dbg !7
  %956 = select i1 %.not1, float %949, float %943, !dbg !7
  %957 = bitcast float %956 to i32, !dbg !7
  %958 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %957), !dbg !7
  %959 = bitcast i32 %958 to float, !dbg !7
  %960 = select i1 %.not1, float %943, float %959, !dbg !7
  %961 = select i1 %.not1, float %959, float %949, !dbg !7
  %962 = extractelement <8 x float> %37, i64 0, !dbg !7
  %963 = extractelement <8 x float> %37, i64 1, !dbg !7
  %964 = extractelement <8 x float> %37, i64 2, !dbg !7
  %965 = extractelement <8 x float> %37, i64 3, !dbg !7
  %966 = select i1 %.not, float %963, float %962, !dbg !7
  %967 = bitcast float %966 to i32, !dbg !7
  %968 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %967), !dbg !7
  %969 = bitcast i32 %968 to float, !dbg !7
  %970 = select i1 %.not, float %962, float %969, !dbg !7
  %971 = select i1 %.not, float %969, float %963, !dbg !7
  %972 = select i1 %.not, float %965, float %964, !dbg !7
  %973 = bitcast float %972 to i32, !dbg !7
  %974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %973), !dbg !7
  %975 = bitcast i32 %974 to float, !dbg !7
  %976 = select i1 %.not, float %964, float %975, !dbg !7
  %977 = select i1 %.not, float %975, float %965, !dbg !7
  %978 = select i1 %.not1, float %976, float %970, !dbg !7
  %979 = bitcast float %978 to i32, !dbg !7
  %980 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %979), !dbg !7
  %981 = select i1 %.not1, float %977, float %971, !dbg !7
  %982 = bitcast float %981 to i32, !dbg !7
  %983 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %982), !dbg !7
  %984 = shufflevector <1 x float> %38, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %985 = extractelement <8 x float> %48, i64 4, !dbg !7
  %986 = extractelement <8 x float> %48, i64 5, !dbg !7
  %987 = extractelement <8 x float> %48, i64 6, !dbg !7
  %988 = extractelement <8 x float> %48, i64 7, !dbg !7
  %989 = select i1 %.not, float %986, float %985, !dbg !7
  %990 = bitcast float %989 to i32, !dbg !7
  %991 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %990), !dbg !7
  %992 = bitcast i32 %991 to float, !dbg !7
  %993 = select i1 %.not, float %985, float %992, !dbg !7
  %994 = select i1 %.not, float %992, float %986, !dbg !7
  %995 = select i1 %.not, float %988, float %987, !dbg !7
  %996 = bitcast float %995 to i32, !dbg !7
  %997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %996), !dbg !7
  %998 = bitcast i32 %997 to float, !dbg !7
  %999 = select i1 %.not, float %987, float %998, !dbg !7
  %1000 = select i1 %.not, float %998, float %988, !dbg !7
  %1001 = select i1 %.not1, float %999, float %993, !dbg !7
  %1002 = bitcast float %1001 to i32, !dbg !7
  %1003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1002), !dbg !7
  %1004 = bitcast i32 %1003 to float, !dbg !7
  %1005 = select i1 %.not1, float %993, float %1004, !dbg !7
  %1006 = select i1 %.not1, float %1004, float %999, !dbg !7
  %1007 = select i1 %.not1, float %1000, float %994, !dbg !7
  %1008 = bitcast float %1007 to i32, !dbg !7
  %1009 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1008), !dbg !7
  %1010 = bitcast i32 %1009 to float, !dbg !7
  %1011 = select i1 %.not1, float %994, float %1010, !dbg !7
  %1012 = select i1 %.not1, float %1010, float %1000, !dbg !7
  %1013 = extractelement <8 x float> %45, i64 4, !dbg !7
  %1014 = extractelement <8 x float> %45, i64 5, !dbg !7
  %1015 = extractelement <8 x float> %45, i64 6, !dbg !7
  %1016 = extractelement <8 x float> %45, i64 7, !dbg !7
  %1017 = select i1 %.not, float %1014, float %1013, !dbg !7
  %1018 = bitcast float %1017 to i32, !dbg !7
  %1019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %1018), !dbg !7
  %1020 = bitcast i32 %1019 to float, !dbg !7
  %1021 = select i1 %.not, float %1013, float %1020, !dbg !7
  %1022 = select i1 %.not, float %1020, float %1014, !dbg !7
  %1023 = select i1 %.not, float %1016, float %1015, !dbg !7
  %1024 = bitcast float %1023 to i32, !dbg !7
  %1025 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %1024), !dbg !7
  %1026 = bitcast i32 %1025 to float, !dbg !7
  %1027 = select i1 %.not, float %1015, float %1026, !dbg !7
  %1028 = select i1 %.not, float %1026, float %1016, !dbg !7
  %1029 = select i1 %.not1, float %1027, float %1021, !dbg !7
  %1030 = bitcast float %1029 to i32, !dbg !7
  %1031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1030), !dbg !7
  %1032 = bitcast i32 %1031 to float, !dbg !7
  %1033 = select i1 %.not1, float %1021, float %1032, !dbg !7
  %1034 = select i1 %.not1, float %1032, float %1027, !dbg !7
  %1035 = select i1 %.not1, float %1028, float %1022, !dbg !7
  %1036 = bitcast float %1035 to i32, !dbg !7
  %1037 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1036), !dbg !7
  %1038 = bitcast i32 %1037 to float, !dbg !7
  %1039 = select i1 %.not1, float %1022, float %1038, !dbg !7
  %1040 = select i1 %.not1, float %1038, float %1028, !dbg !7
  %1041 = extractelement <8 x float> %42, i64 4, !dbg !7
  %1042 = extractelement <8 x float> %42, i64 5, !dbg !7
  %1043 = extractelement <8 x float> %42, i64 6, !dbg !7
  %1044 = extractelement <8 x float> %42, i64 7, !dbg !7
  %1045 = select i1 %.not, float %1042, float %1041, !dbg !7
  %1046 = bitcast float %1045 to i32, !dbg !7
  %1047 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %1046), !dbg !7
  %1048 = bitcast i32 %1047 to float, !dbg !7
  %1049 = select i1 %.not, float %1041, float %1048, !dbg !7
  %1050 = select i1 %.not, float %1048, float %1042, !dbg !7
  %1051 = select i1 %.not, float %1044, float %1043, !dbg !7
  %1052 = bitcast float %1051 to i32, !dbg !7
  %1053 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %1052), !dbg !7
  %1054 = bitcast i32 %1053 to float, !dbg !7
  %1055 = select i1 %.not, float %1043, float %1054, !dbg !7
  %1056 = select i1 %.not, float %1054, float %1044, !dbg !7
  %1057 = select i1 %.not1, float %1055, float %1049, !dbg !7
  %1058 = bitcast float %1057 to i32, !dbg !7
  %1059 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1058), !dbg !7
  %1060 = bitcast i32 %1059 to float, !dbg !7
  %1061 = select i1 %.not1, float %1049, float %1060, !dbg !7
  %1062 = select i1 %.not1, float %1060, float %1055, !dbg !7
  %1063 = select i1 %.not1, float %1056, float %1050, !dbg !7
  %1064 = bitcast float %1063 to i32, !dbg !7
  %1065 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1064), !dbg !7
  %1066 = bitcast i32 %1065 to float, !dbg !7
  %1067 = select i1 %.not1, float %1050, float %1066, !dbg !7
  %1068 = select i1 %.not1, float %1066, float %1056, !dbg !7
  %1069 = extractelement <8 x float> %37, i64 4, !dbg !7
  %1070 = extractelement <8 x float> %37, i64 5, !dbg !7
  %1071 = extractelement <8 x float> %37, i64 6, !dbg !7
  %1072 = extractelement <8 x float> %37, i64 7, !dbg !7
  %1073 = select i1 %.not, float %1070, float %1069, !dbg !7
  %1074 = bitcast float %1073 to i32, !dbg !7
  %1075 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %1074), !dbg !7
  %1076 = bitcast i32 %1075 to float, !dbg !7
  %1077 = select i1 %.not, float %1069, float %1076, !dbg !7
  %1078 = select i1 %.not, float %1076, float %1070, !dbg !7
  %1079 = select i1 %.not, float %1072, float %1071, !dbg !7
  %1080 = bitcast float %1079 to i32, !dbg !7
  %1081 = call i32 @llvm.amdgcn.ds.bpermute(i32 %878, i32 %1080), !dbg !7
  %1082 = bitcast i32 %1081 to float, !dbg !7
  %1083 = select i1 %.not, float %1071, float %1082, !dbg !7
  %1084 = select i1 %.not, float %1082, float %1072, !dbg !7
  %1085 = select i1 %.not1, float %1083, float %1077, !dbg !7
  %1086 = bitcast float %1085 to i32, !dbg !7
  %1087 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1086), !dbg !7
  %1088 = select i1 %.not1, float %1084, float %1078, !dbg !7
  %1089 = bitcast float %1088 to i32, !dbg !7
  %1090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %894, i32 %1089), !dbg !7
  br i1 true, label %1091, label %1125, !dbg !20

1091:                                             ; preds = %863
  %1092 = bitcast i32 %1087 to float, !dbg !7
  %1093 = select i1 %.not1, float %1077, float %1092, !dbg !7
  %1094 = insertelement <4 x float> poison, float %1093, i64 0, !dbg !18
  %1095 = bitcast i32 %1090 to float, !dbg !7
  %1096 = select i1 %.not1, float %1078, float %1095, !dbg !7
  %1097 = insertelement <4 x float> %1094, float %1096, i64 1, !dbg !18
  %1098 = select i1 %.not1, float %1092, float %1083, !dbg !7
  %1099 = insertelement <4 x float> %1097, float %1098, i64 2, !dbg !18
  %1100 = select i1 %.not1, float %1095, float %1084, !dbg !7
  %1101 = insertelement <4 x float> %1099, float %1100, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %984, !dbg !18
  %normalized = fmul <4 x float> %1101, %row.reciprocal, !dbg !18
  %1102 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1103 = extractelement <4 x half> %1102, i64 3, !dbg !19
  %1104 = extractelement <4 x half> %1102, i64 2, !dbg !19
  %1105 = extractelement <4 x half> %1102, i64 1, !dbg !19
  %1106 = extractelement <4 x half> %1102, i64 0, !dbg !19
  %1107 = bitcast i32 %980 to float, !dbg !7
  %1108 = select i1 %.not1, float %970, float %1107, !dbg !7
  %1109 = insertelement <4 x float> poison, float %1108, i64 0, !dbg !18
  %1110 = bitcast i32 %983 to float, !dbg !7
  %1111 = select i1 %.not1, float %971, float %1110, !dbg !7
  %1112 = insertelement <4 x float> %1109, float %1111, i64 1, !dbg !18
  %1113 = select i1 %.not1, float %1107, float %976, !dbg !7
  %1114 = insertelement <4 x float> %1112, float %1113, i64 2, !dbg !18
  %1115 = select i1 %.not1, float %1110, float %977, !dbg !7
  %1116 = insertelement <4 x float> %1114, float %1115, i64 3, !dbg !18
  %normalized38 = fmul <4 x float> %1116, %row.reciprocal, !dbg !18
  %1117 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !19
  %1118 = extractelement <4 x half> %1117, i64 3, !dbg !19
  %1119 = extractelement <4 x half> %1117, i64 2, !dbg !19
  %1120 = extractelement <4 x half> %1117, i64 1, !dbg !19
  %1121 = extractelement <4 x half> %1117, i64 0, !dbg !19
  %1122 = add nsw i64 %15, %6, !dbg !20
  %1123 = shl nsw i64 %1122, 19, !dbg !20
  %1124 = and i64 %1123, 16252928, !dbg !20
  %address.thread327 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local328 = zext nneg i32 %address.thread327 to i64, !dbg !13
  %address.local329 = add nsw i64 %18, %address.local328, !dbg !20
  %address.local330 = lshr i64 %address.local328, 6, !dbg !13
  %address.local331 = shl nuw nsw i64 %address.local330, 4, !dbg !20
  %address.local332 = add nsw i64 %address.local329, %address.local331, !dbg !20
  %address.local333 = and i64 %address.local328, 240, !dbg !20
  %address.local334 = sub nsw i64 %address.local332, %address.local333, !dbg !20
  %address.local335 = add nsw i64 %address.local334, %19, !dbg !20
  %address.local336 = shl nuw nsw i64 %address.local335, 7, !dbg !20
  %address.local337 = add nuw nsw i64 %1124, %address.local336, !dbg !20
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local339 = lshr i64 %address.local328, 2, !dbg !13
  %address.local340 = and i64 %address.local339, 12, !dbg !13
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local340, !dbg !20
  store half %1121, ptr addrspace(1) %address.local341, align 2, !dbg !20
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local342, i64 %address.local340, !dbg !20
  %address.local344 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local343, i64 2, !dbg !20
  store half %1120, ptr addrspace(1) %address.local344, align 2, !dbg !20
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local345, i64 %address.local340, !dbg !20
  %address.local347 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local346, i64 4, !dbg !20
  store half %1119, ptr addrspace(1) %address.local347, align 2, !dbg !20
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local348, i64 %address.local340, !dbg !20
  %address.local350 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local349, i64 6, !dbg !20
  store half %1118, ptr addrspace(1) %address.local350, align 2, !dbg !20
  %address.local351 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local352 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local351, i64 %address.local340, !dbg !20
  %address.local353 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local352, i64 32, !dbg !20
  store half %1106, ptr addrspace(1) %address.local353, align 2, !dbg !20
  %address.local354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local354, i64 %address.local340, !dbg !20
  %address.local356 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local355, i64 34, !dbg !20
  store half %1105, ptr addrspace(1) %address.local356, align 2, !dbg !20
  %address.local357 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local358 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local357, i64 %address.local340, !dbg !20
  %address.local359 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local358, i64 36, !dbg !20
  store half %1104, ptr addrspace(1) %address.local359, align 2, !dbg !20
  %address.local360 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local337, !dbg !20
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local360, i64 %address.local340, !dbg !20
  %address.local362 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local361, i64 38, !dbg !20
  store half %1103, ptr addrspace(1) %address.local362, align 2, !dbg !20
  br label %1125, !dbg !20

1125:                                             ; preds = %1091, %863
  br i1 true, label %1126, label %1146, !dbg !20

1126:                                             ; preds = %1125
  %1127 = insertelement <4 x float> poison, float %1061, i64 0, !dbg !18
  %1128 = insertelement <4 x float> %1127, float %1067, i64 1, !dbg !18
  %1129 = insertelement <4 x float> %1128, float %1062, i64 2, !dbg !18
  %1130 = insertelement <4 x float> %1129, float %1068, i64 3, !dbg !18
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %984, !dbg !18
  %normalized40 = fmul <4 x float> %1130, %row.reciprocal39, !dbg !18
  %1131 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !19
  %1132 = extractelement <4 x half> %1131, i64 3, !dbg !19
  %1133 = extractelement <4 x half> %1131, i64 2, !dbg !19
  %1134 = extractelement <4 x half> %1131, i64 1, !dbg !19
  %1135 = extractelement <4 x half> %1131, i64 0, !dbg !19
  %1136 = insertelement <4 x float> poison, float %954, i64 0, !dbg !18
  %1137 = insertelement <4 x float> %1136, float %960, i64 1, !dbg !18
  %1138 = insertelement <4 x float> %1137, float %955, i64 2, !dbg !18
  %1139 = insertelement <4 x float> %1138, float %961, i64 3, !dbg !18
  %normalized41 = fmul <4 x float> %1139, %row.reciprocal39, !dbg !18
  %1140 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !19
  %1141 = extractelement <4 x half> %1140, i64 3, !dbg !19
  %1142 = extractelement <4 x half> %1140, i64 2, !dbg !19
  %1143 = extractelement <4 x half> %1140, i64 1, !dbg !19
  %1144 = extractelement <4 x half> %1140, i64 0, !dbg !19
  %1145 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread363 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local364 = zext nneg i32 %address.thread363 to i64, !dbg !13
  %address.local365 = add nsw i64 %18, %address.local364, !dbg !20
  %address.local366 = lshr i64 %address.local364, 6, !dbg !13
  %address.local367 = shl nuw nsw i64 %address.local366, 4, !dbg !20
  %address.local368 = add nsw i64 %address.local365, %address.local367, !dbg !20
  %address.local369 = and i64 %address.local364, 240, !dbg !20
  %address.local370 = sub nsw i64 %address.local368, %address.local369, !dbg !20
  %address.local371 = add nsw i64 %address.local370, %19, !dbg !20
  %address.local372 = shl nuw nsw i64 %address.local371, 7, !dbg !20
  %address.local373 = add nuw nsw i64 %1145, %address.local372, !dbg !20
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local375 = lshr i64 %address.local364, 2, !dbg !13
  %address.local376 = and i64 %address.local375, 12, !dbg !13
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local374, i64 %address.local376, !dbg !20
  %address.local378 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local377, i64 64, !dbg !20
  store half %1144, ptr addrspace(1) %address.local378, align 2, !dbg !20
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local376, !dbg !20
  %address.local381 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local380, i64 66, !dbg !20
  store half %1143, ptr addrspace(1) %address.local381, align 2, !dbg !20
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local376, !dbg !20
  %address.local384 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local383, i64 68, !dbg !20
  store half %1142, ptr addrspace(1) %address.local384, align 2, !dbg !20
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local385, i64 %address.local376, !dbg !20
  %address.local387 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local386, i64 70, !dbg !20
  store half %1141, ptr addrspace(1) %address.local387, align 2, !dbg !20
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local388, i64 %address.local376, !dbg !20
  %address.local390 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local389, i64 96, !dbg !20
  store half %1135, ptr addrspace(1) %address.local390, align 2, !dbg !20
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local391, i64 %address.local376, !dbg !20
  %address.local393 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local392, i64 98, !dbg !20
  store half %1134, ptr addrspace(1) %address.local393, align 2, !dbg !20
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local376, !dbg !20
  %address.local396 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local395, i64 100, !dbg !20
  store half %1133, ptr addrspace(1) %address.local396, align 2, !dbg !20
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local373, !dbg !20
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local397, i64 %address.local376, !dbg !20
  %address.local399 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local398, i64 102, !dbg !20
  store half %1132, ptr addrspace(1) %address.local399, align 2, !dbg !20
  br label %1146, !dbg !20

1146:                                             ; preds = %1126, %1125
  br i1 true, label %1147, label %1167, !dbg !20

1147:                                             ; preds = %1146
  %1148 = insertelement <4 x float> poison, float %1033, i64 0, !dbg !18
  %1149 = insertelement <4 x float> %1148, float %1039, i64 1, !dbg !18
  %1150 = insertelement <4 x float> %1149, float %1034, i64 2, !dbg !18
  %1151 = insertelement <4 x float> %1150, float %1040, i64 3, !dbg !18
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %984, !dbg !18
  %normalized43 = fmul <4 x float> %1151, %row.reciprocal42, !dbg !18
  %1152 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !19
  %1153 = extractelement <4 x half> %1152, i64 3, !dbg !19
  %1154 = extractelement <4 x half> %1152, i64 2, !dbg !19
  %1155 = extractelement <4 x half> %1152, i64 1, !dbg !19
  %1156 = extractelement <4 x half> %1152, i64 0, !dbg !19
  %1157 = insertelement <4 x float> poison, float %926, i64 0, !dbg !18
  %1158 = insertelement <4 x float> %1157, float %932, i64 1, !dbg !18
  %1159 = insertelement <4 x float> %1158, float %927, i64 2, !dbg !18
  %1160 = insertelement <4 x float> %1159, float %933, i64 3, !dbg !18
  %normalized44 = fmul <4 x float> %1160, %row.reciprocal42, !dbg !18
  %1161 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !19
  %1162 = extractelement <4 x half> %1161, i64 3, !dbg !19
  %1163 = extractelement <4 x half> %1161, i64 2, !dbg !19
  %1164 = extractelement <4 x half> %1161, i64 1, !dbg !19
  %1165 = extractelement <4 x half> %1161, i64 0, !dbg !19
  %1166 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread400 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local401 = zext nneg i32 %address.thread400 to i64, !dbg !13
  %address.local402 = add nsw i64 %18, %address.local401, !dbg !20
  %address.local403 = lshr i64 %address.local401, 6, !dbg !13
  %address.local404 = shl nuw nsw i64 %address.local403, 4, !dbg !20
  %address.local405 = add nsw i64 %address.local402, %address.local404, !dbg !20
  %address.local406 = and i64 %address.local401, 240, !dbg !20
  %address.local407 = sub nsw i64 %address.local405, %address.local406, !dbg !20
  %address.local408 = add nsw i64 %address.local407, %19, !dbg !20
  %address.local409 = shl nuw nsw i64 %address.local408, 7, !dbg !20
  %address.local410 = add nuw nsw i64 %1166, %address.local409, !dbg !20
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local412 = lshr i64 %address.local401, 2, !dbg !13
  %address.local413 = and i64 %address.local412, 12, !dbg !13
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local411, i64 %address.local413, !dbg !20
  %address.local415 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local414, i64 128, !dbg !20
  store half %1165, ptr addrspace(1) %address.local415, align 2, !dbg !20
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local413, !dbg !20
  %address.local418 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local417, i64 130, !dbg !20
  store half %1164, ptr addrspace(1) %address.local418, align 2, !dbg !20
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local413, !dbg !20
  %address.local421 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local420, i64 132, !dbg !20
  store half %1163, ptr addrspace(1) %address.local421, align 2, !dbg !20
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local422, i64 %address.local413, !dbg !20
  %address.local424 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local423, i64 134, !dbg !20
  store half %1162, ptr addrspace(1) %address.local424, align 2, !dbg !20
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local425, i64 %address.local413, !dbg !20
  %address.local427 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local426, i64 160, !dbg !20
  store half %1156, ptr addrspace(1) %address.local427, align 2, !dbg !20
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local413, !dbg !20
  %address.local430 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local429, i64 162, !dbg !20
  store half %1155, ptr addrspace(1) %address.local430, align 2, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local431, i64 %address.local413, !dbg !20
  %address.local433 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local432, i64 164, !dbg !20
  store half %1154, ptr addrspace(1) %address.local433, align 2, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local410, !dbg !20
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local413, !dbg !20
  %address.local436 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local435, i64 166, !dbg !20
  store half %1153, ptr addrspace(1) %address.local436, align 2, !dbg !20
  br label %1167, !dbg !20

1167:                                             ; preds = %1147, %1146
  br i1 true, label %1168, label %1188, !dbg !20

1168:                                             ; preds = %1167
  %1169 = insertelement <4 x float> poison, float %1005, i64 0, !dbg !18
  %1170 = insertelement <4 x float> %1169, float %1011, i64 1, !dbg !18
  %1171 = insertelement <4 x float> %1170, float %1006, i64 2, !dbg !18
  %1172 = insertelement <4 x float> %1171, float %1012, i64 3, !dbg !18
  %row.reciprocal45 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %984, !dbg !18
  %normalized46 = fmul <4 x float> %1172, %row.reciprocal45, !dbg !18
  %1173 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !19
  %1174 = extractelement <4 x half> %1173, i64 3, !dbg !19
  %1175 = extractelement <4 x half> %1173, i64 2, !dbg !19
  %1176 = extractelement <4 x half> %1173, i64 1, !dbg !19
  %1177 = extractelement <4 x half> %1173, i64 0, !dbg !19
  %1178 = insertelement <4 x float> poison, float %898, i64 0, !dbg !18
  %1179 = insertelement <4 x float> %1178, float %904, i64 1, !dbg !18
  %1180 = insertelement <4 x float> %1179, float %899, i64 2, !dbg !18
  %1181 = insertelement <4 x float> %1180, float %905, i64 3, !dbg !18
  %normalized47 = fmul <4 x float> %1181, %row.reciprocal45, !dbg !18
  %1182 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !19
  %1183 = extractelement <4 x half> %1182, i64 3, !dbg !19
  %1184 = extractelement <4 x half> %1182, i64 2, !dbg !19
  %1185 = extractelement <4 x half> %1182, i64 1, !dbg !19
  %1186 = extractelement <4 x half> %1182, i64 0, !dbg !19
  %1187 = shl nuw nsw i64 %17, 19, !dbg !20
  %address.thread437 = call i32 asm sideeffect "", "=v,0"(i32 %21) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local438 = zext nneg i32 %address.thread437 to i64, !dbg !13
  %address.local439 = add nsw i64 %18, %address.local438, !dbg !20
  %address.local440 = lshr i64 %address.local438, 6, !dbg !13
  %address.local441 = shl nuw nsw i64 %address.local440, 4, !dbg !20
  %address.local442 = add nsw i64 %address.local439, %address.local441, !dbg !20
  %address.local443 = and i64 %address.local438, 240, !dbg !20
  %address.local444 = sub nsw i64 %address.local442, %address.local443, !dbg !20
  %address.local445 = add nsw i64 %address.local444, %19, !dbg !20
  %address.local446 = shl nuw nsw i64 %address.local445, 7, !dbg !20
  %address.local447 = add nuw nsw i64 %1187, %address.local446, !dbg !20
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local449 = lshr i64 %address.local438, 2, !dbg !13
  %address.local450 = and i64 %address.local449, 12, !dbg !13
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local450, !dbg !20
  %address.local452 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local451, i64 192, !dbg !20
  store half %1186, ptr addrspace(1) %address.local452, align 2, !dbg !20
  %address.local453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local453, i64 %address.local450, !dbg !20
  %address.local455 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local454, i64 194, !dbg !20
  store half %1185, ptr addrspace(1) %address.local455, align 2, !dbg !20
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local456, i64 %address.local450, !dbg !20
  %address.local458 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local457, i64 196, !dbg !20
  store half %1184, ptr addrspace(1) %address.local458, align 2, !dbg !20
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local459, i64 %address.local450, !dbg !20
  %address.local461 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local460, i64 198, !dbg !20
  store half %1183, ptr addrspace(1) %address.local461, align 2, !dbg !20
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local450, !dbg !20
  %address.local464 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local463, i64 224, !dbg !20
  store half %1177, ptr addrspace(1) %address.local464, align 2, !dbg !20
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local465, i64 %address.local450, !dbg !20
  %address.local467 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local466, i64 226, !dbg !20
  store half %1176, ptr addrspace(1) %address.local467, align 2, !dbg !20
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local468, i64 %address.local450, !dbg !20
  %address.local470 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local469, i64 228, !dbg !20
  store half %1175, ptr addrspace(1) %address.local470, align 2, !dbg !20
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local447, !dbg !20
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local471, i64 %address.local450, !dbg !20
  %address.local473 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local472, i64 230, !dbg !20
  store half %1174, ptr addrspace(1) %address.local473, align 2, !dbg !20
  br label %1188, !dbg !20

1188:                                             ; preds = %1168, %1167
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/nohint-rolled")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 73, column: 26, scope: !4)
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
