; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/nohint-rolled/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/nohint-rolled/input.mlir"

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
  %25 = shl nsw i64 %8, 1, !dbg !8
  %26 = add nsw i64 %25, 2, !dbg !8
  br label %27, !dbg !8

27:                                               ; preds = %275, %4
  %28 = phi i64 [ %822, %275 ], [ 0, %4 ], !dbg !8
  %29 = phi [2 x <8 x float>] [ %807, %275 ], [ zeroinitializer, %4 ], !dbg !8
  %30 = phi [2 x <1 x float>] [ %583, %275 ], [ zeroinitializer, %4 ], !dbg !8
  %31 = phi [2 x <1 x float>] [ %809, %275 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !8
  %32 = phi [2 x <8 x float>] [ %813, %275 ], [ zeroinitializer, %4 ], !dbg !8
  %33 = phi [2 x <8 x float>] [ %817, %275 ], [ zeroinitializer, %4 ], !dbg !8
  %34 = phi [2 x <8 x float>] [ %821, %275 ], [ zeroinitializer, %4 ], !dbg !8
  %35 = icmp slt i64 %28, %26, !dbg !8
  br i1 %35, label %36, label %823, !dbg !8

36:                                               ; preds = %27
  %37 = shl nsw i64 %28, 6, !dbg !8
  %38 = lshr i64 %28, 6, !dbg !16
  %39 = shl i64 %38, 12, !dbg !16
  %40 = sub i64 %37, %39, !dbg !16
  br label %41, !dbg !21

41:                                               ; preds = %45, %36
  %42 = phi i64 [ %167, %45 ], [ 0, %36 ], !dbg !21
  %43 = phi [2 x <16 x float>] [ %166, %45 ], [ zeroinitializer, %36 ], !dbg !21
  %44 = icmp samesign ult i64 %42, 4, !dbg !21
  br i1 %44, label %45, label %168, !dbg !21

45:                                               ; preds = %41
  %46 = shl nuw nsw i64 %42, 5, !dbg !21
  %.idx = shl nuw nsw i64 %15, 20, !dbg !12
  %47 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !12
  %.idx35 = shl nsw i64 %18, 8, !dbg !12
  %48 = getelementptr i8, ptr addrspace(1) %47, i64 %.idx35, !dbg !12
  %49 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %46, !dbg !12
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local77 = shl nuw nsw i64 %address.local, 3, !dbg !12
  %address.local78 = and i64 %address.local77, 1536, !dbg !12
  %address.local79 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local78, !dbg !12
  %address.local80 = ptrtoint ptr addrspace(3) %address.local79 to i64, !dbg !12
  %address.local81 = trunc i64 %address.local80 to i32, !dbg !12
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local81), !dbg !12
  %51 = zext i32 %50 to i64, !dbg !12
  %52 = inttoptr i64 %51 to ptr addrspace(3), !dbg !12
  %address.local82 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local83 = and i64 %address.local82, 16256, !dbg !12
  %address.local84 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local83, !dbg !12
  %address.local85 = lshr i64 %address.local, 4, !dbg !12
  %address.local86 = and i64 %address.local85, 8, !dbg !12
  %address.local87 = getelementptr [2 x i8], ptr addrspace(1) %address.local84, i64 %address.local86, !dbg !12
  %53 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %54 = ptrtoint ptr addrspace(1) %address.local87 to i64, !dbg !12
  %55 = sub i64 %54, %53, !dbg !12
  %dma.buffer.offset591 = trunc i64 %55 to i32, !dbg !12
  %56 = trunc i64 %53 to i32, !dbg !12
  %57 = insertelement <4 x i32> poison, i32 %56, i64 0, !dbg !12
  %58 = lshr i64 %53, 32, !dbg !12
  %59 = trunc i64 %58 to i32, !dbg !12
  %60 = insertelement <4 x i32> %57, i32 %59, i64 1, !dbg !12
  %61 = insertelement <4 x i32> %60, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc592 = insertelement <4 x i32> %61, i32 131072, i64 3, !dbg !12
  %62 = ptrtoint ptr addrspace(3) %52 to i32, !dbg !12
  %dma.buffer.lds593 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %62), !dbg !12
  %63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds593, i32 %dma.buffer.offset591, <4 x i32> %dma.buffer.rsrc592) #10, !dbg !12
  %address.local88 = or disjoint i64 %address.local, 256, !dbg !12
  %address.local89 = shl nuw nsw i64 %address.local88, 3, !dbg !12
  %address.local90 = and i64 %address.local89, 3584, !dbg !12
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local90, !dbg !12
  %address.local92 = ptrtoint ptr addrspace(3) %address.local91 to i64, !dbg !12
  %address.local93 = trunc i64 %address.local92 to i32, !dbg !12
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local93), !dbg !12
  %65 = zext i32 %64 to i64, !dbg !12
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !12
  %address.local94 = shl nuw nsw i64 %address.local, 7, !dbg !12
  %address.local95 = and i64 %address.local94, 16256, !dbg !12
  %address.local96 = getelementptr [2 x i8], ptr addrspace(1) %49, i64 %address.local95, !dbg !12
  %address.local97 = lshr i64 %address.local88, 4, !dbg !12
  %address.local98 = and i64 %address.local97, 24, !dbg !12
  %address.local99 = getelementptr [2 x i8], ptr addrspace(1) %address.local96, i64 %address.local98, !dbg !12
  %67 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !12
  %68 = ptrtoint ptr addrspace(1) %address.local99 to i64, !dbg !12
  %69 = sub i64 %68, %67, !dbg !12
  %dma.buffer.offset588 = trunc i64 %69 to i32, !dbg !12
  %70 = trunc i64 %67 to i32, !dbg !12
  %71 = insertelement <4 x i32> poison, i32 %70, i64 0, !dbg !12
  %72 = lshr i64 %67, 32, !dbg !12
  %73 = trunc i64 %72 to i32, !dbg !12
  %74 = insertelement <4 x i32> %71, i32 %73, i64 1, !dbg !12
  %75 = insertelement <4 x i32> %74, i32 33554432, i64 2, !dbg !12
  %dma.buffer.rsrc589 = insertelement <4 x i32> %75, i32 131072, i64 3, !dbg !12
  %76 = ptrtoint ptr addrspace(3) %66 to i32, !dbg !12
  %dma.buffer.lds590 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %76), !dbg !12
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds590, i32 %dma.buffer.offset588, <4 x i32> %dma.buffer.rsrc589) #10, !dbg !12
  %78 = shl nuw nsw i64 %42, 11, !dbg !16
  %79 = add nuw i64 %28, %78, !dbg !16
  %.lobit36 = ashr i64 %79, 63, !dbg !16
  %80 = xor i64 %79, %.lobit36, !dbg !16
  %81 = sdiv i64 %80, 8192, !dbg !16
  %.lobit37 = ashr i64 %79, 63, !dbg !16
  %82 = xor i64 %81, %.lobit37, !dbg !16
  %83 = add nsw i64 %82, %6, !dbg !16
  %84 = add nuw nsw i64 %46, %38, !dbg !16
  %85 = and i64 %84, 127, !dbg !16
  %86 = shl i64 %83, 19, !dbg !16
  %87 = and i64 %86, 16252928, !dbg !16
  %88 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %87, !dbg !16
  %89 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %85, !dbg !16
  %.idx38 = shl i64 %40, 8, !dbg !16
  %90 = getelementptr i8, ptr addrspace(1) %89, i64 %.idx38, !dbg !16
  %address.local100 = and i64 %address.local77, 1536, !dbg !16
  %address.local101 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local100, !dbg !16
  %address.local102 = ptrtoint ptr addrspace(3) %address.local101 to i64, !dbg !16
  %address.local103 = trunc i64 %address.local102 to i32, !dbg !16
  %91 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local103), !dbg !16
  %92 = zext i32 %91 to i64, !dbg !16
  %93 = inttoptr i64 %92 to ptr addrspace(3), !dbg !16
  %address.local104 = lshr i64 %address.local, 3, !dbg !16
  %address.local105 = and i64 %address.local104, 24, !dbg !16
  %address.local106 = getelementptr [2 x i8], ptr addrspace(1) %90, i64 %address.local105, !dbg !16
  %address.local107 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local108 = and i64 %address.local107, 8064, !dbg !16
  %address.local109 = getelementptr [2 x i8], ptr addrspace(1) %address.local106, i64 %address.local108, !dbg !16
  %94 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %95 = ptrtoint ptr addrspace(1) %address.local109 to i64, !dbg !16
  %96 = sub i64 %95, %94, !dbg !16
  %dma.buffer.offset = trunc i64 %96 to i32, !dbg !16
  %97 = trunc i64 %94 to i32, !dbg !16
  %98 = insertelement <4 x i32> poison, i32 %97, i64 0, !dbg !16
  %99 = lshr i64 %94, 32, !dbg !16
  %100 = trunc i64 %99 to i32, !dbg !16
  %101 = insertelement <4 x i32> %98, i32 %100, i64 1, !dbg !16
  %102 = insertelement <4 x i32> %101, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc = insertelement <4 x i32> %102, i32 131072, i64 3, !dbg !16
  %103 = ptrtoint ptr addrspace(3) %93 to i32, !dbg !16
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103), !dbg !16
  %104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local110 = lshr i32 %address.thread, 2, !dbg !13
  %address.local111 = and i32 %address.local110, 4, !dbg !13
  %address.local112 = shl nuw nsw i32 %address.thread, 1, !dbg !13
  %address.local113 = and i32 %address.local112, 384, !dbg !13
  %address.local114 = or disjoint i32 %address.local111, %address.local113, !dbg !13
  %address.local115 = shl nuw nsw i32 %address.thread, 3, !dbg !13
  %address.local116 = and i32 %address.local115, 120, !dbg !13
  %address.local117 = or disjoint i32 %address.local114, %address.local116, !dbg !13
  %address.local118 = shl nuw nsw i32 %address.thread, 5, !dbg !13
  %address.local119 = and i32 %address.local118, 1024, !dbg !13
  %address.local120 = or disjoint i32 %address.local117, %address.local119, !dbg !13
  %address.local121 = zext nneg i32 %address.local120 to i64, !dbg !13
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local121, !dbg !13
  %105 = load <4 x half>, ptr addrspace(3) %address.local122, align 2, !dbg !13
  %address.local123 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local121, !dbg !13
  %address.local124 = getelementptr i8, ptr addrspace(3) %address.local123, i64 4096, !dbg !13
  %106 = load <4 x half>, ptr addrspace(3) %address.local124, align 2, !dbg !13
  %address.local125 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local121, !dbg !13
  %address.local126 = getelementptr i8, ptr addrspace(3) %address.local125, i64 1024, !dbg !13
  %107 = load <4 x half>, ptr addrspace(3) %address.local126, align 2, !dbg !13
  %108 = fmul <4 x half> %105, splat (half 0xH3015), !dbg !13
  %address.local127 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local121, !dbg !13
  %address.local128 = getelementptr i8, ptr addrspace(3) %address.local127, i64 5120, !dbg !13
  %109 = load <4 x half>, ptr addrspace(3) %address.local128, align 2, !dbg !13
  %110 = fmul <4 x half> %106, splat (half 0xH3015), !dbg !13
  %111 = fmul <4 x half> %107, splat (half 0xH3015), !dbg !13
  %112 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local129 = shl nuw nsw i64 %address.local, 7, !dbg !7
  %address.local130 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local131 = lshr i64 %address.local, 2, !dbg !7
  %address.local132 = mul nsw i64 %address.local131, -512, !dbg !7
  %address.local133 = getelementptr [2 x i8], ptr addrspace(3) %address.local130, i64 %address.local132, !dbg !7
  %address.local134 = shl nuw nsw i64 %address.local, 1, !dbg !7
  %address.local135 = and i64 %address.local134, 24, !dbg !7
  %address.local136 = getelementptr [2 x i8], ptr addrspace(3) %address.local133, i64 %address.local135, !dbg !7
  %address.local137 = lshr i64 %address.local, 2, !dbg !13
  %address.local138 = and i64 %address.local137, 12, !dbg !13
  %address.local139 = getelementptr [2 x i8], ptr addrspace(3) %address.local136, i64 %address.local138, !dbg !7
  %address.local140 = and i64 %address.local, 63, !dbg !13
  %address.local141 = lshr i64 %address.local140, 5, !dbg !13
  %address.local142 = mul nuw nsw i64 %address.local141, 504, !dbg !7
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) %address.local139, i64 %address.local142, !dbg !7
  %113 = load <4 x half>, ptr addrspace(3) %address.local143, align 2, !dbg !7
  %114 = extractvalue [2 x <16 x float>] %43, 0, !dbg !36
  %115 = shufflevector <16 x float> %114, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local145 = getelementptr [2 x i8], ptr addrspace(3) %address.local144, i64 %address.local138, !dbg !7
  %address.local146 = getelementptr [2 x i8], ptr addrspace(3) %address.local145, i64 %address.local142, !dbg !7
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) %address.local146, i64 %address.local132, !dbg !7
  %address.local148 = getelementptr [2 x i8], ptr addrspace(3) %address.local147, i64 %address.local135, !dbg !7
  %address.local149 = getelementptr i8, ptr addrspace(3) %address.local148, i64 2048, !dbg !7
  %116 = load <4 x half>, ptr addrspace(3) %address.local149, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %113, <4 x float> %115) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %118 = shufflevector <16 x float> %114, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %116, <4 x float> %117) #9, !dbg !7
  %120 = shufflevector <16 x float> %114, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %121 = shufflevector <16 x float> %114, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %122 = extractvalue [2 x <16 x float>] %43, 1, !dbg !36
  %123 = shufflevector <16 x float> %122, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !36
  %124 = shufflevector <16 x float> %122, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !36
  %125 = shufflevector <16 x float> %122, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !36
  %126 = shufflevector <16 x float> %122, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !36
  %address.local150 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) %address.local150, i64 %address.local132, !dbg !7
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) %address.local151, i64 %address.local135, !dbg !7
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) %address.local152, i64 %address.local138, !dbg !7
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) %address.local153, i64 %address.local142, !dbg !7
  %address.local155 = getelementptr i8, ptr addrspace(3) %address.local154, i64 64, !dbg !7
  %127 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !7
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) %address.local156, i64 %address.local138, !dbg !7
  %address.local158 = getelementptr [2 x i8], ptr addrspace(3) %address.local157, i64 %address.local142, !dbg !7
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) %address.local158, i64 %address.local132, !dbg !7
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local135, !dbg !7
  %address.local161 = getelementptr i8, ptr addrspace(3) %address.local160, i64 2112, !dbg !7
  %128 = load <4 x half>, ptr addrspace(3) %address.local161, align 2, !dbg !7
  %129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %127, <4 x float> %118) #9, !dbg !7
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) %address.local162, i64 %address.local132, !dbg !7
  %address.local164 = getelementptr [2 x i8], ptr addrspace(3) %address.local163, i64 %address.local135, !dbg !7
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) %address.local164, i64 %address.local138, !dbg !7
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) %address.local165, i64 %address.local142, !dbg !7
  %address.local167 = getelementptr i8, ptr addrspace(3) %address.local166, i64 128, !dbg !7
  %130 = load <4 x half>, ptr addrspace(3) %address.local167, align 2, !dbg !7
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %128, <4 x float> %129) #9, !dbg !7
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) %address.local168, i64 %address.local138, !dbg !7
  %address.local170 = getelementptr [2 x i8], ptr addrspace(3) %address.local169, i64 %address.local142, !dbg !7
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) %address.local170, i64 %address.local132, !dbg !7
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) %address.local171, i64 %address.local135, !dbg !7
  %address.local173 = getelementptr i8, ptr addrspace(3) %address.local172, i64 2176, !dbg !7
  %132 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !7
  %133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %130, <4 x float> %120) #9, !dbg !7
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %132, <4 x float> %133) #9, !dbg !7
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) %address.local174, i64 %address.local132, !dbg !7
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) %address.local175, i64 %address.local135, !dbg !7
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) %address.local176, i64 %address.local138, !dbg !7
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) %address.local177, i64 %address.local142, !dbg !7
  %address.local179 = getelementptr i8, ptr addrspace(3) %address.local178, i64 192, !dbg !7
  %135 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !7
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local129, !dbg !7
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) %address.local180, i64 %address.local138, !dbg !7
  %address.local182 = getelementptr [2 x i8], ptr addrspace(3) %address.local181, i64 %address.local142, !dbg !7
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) %address.local182, i64 %address.local132, !dbg !7
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) %address.local183, i64 %address.local135, !dbg !7
  %address.local185 = getelementptr i8, ptr addrspace(3) %address.local184, i64 2240, !dbg !7
  %136 = load <4 x half>, ptr addrspace(3) %address.local185, align 2, !dbg !7
  %137 = shufflevector <4 x float> %134, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %108, <4 x half> %135, <4 x float> %121) #9, !dbg !7
  %139 = load <4 x half>, ptr addrspace(3) %address.local143, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %136, <4 x float> %138) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !36
  %141 = shufflevector <4 x float> %140, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %142 = load <4 x half>, ptr addrspace(3) %address.local149, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %139, <4 x float> %123) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %142, <4 x float> %143) #9, !dbg !7
  %145 = load <4 x half>, ptr addrspace(3) %address.local155, align 2, !dbg !7
  %146 = load <4 x half>, ptr addrspace(3) %address.local161, align 2, !dbg !7
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %145, <4 x float> %124) #9, !dbg !7
  %148 = load <4 x half>, ptr addrspace(3) %address.local167, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %146, <4 x float> %147) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %150 = load <4 x half>, ptr addrspace(3) %address.local173, align 2, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %148, <4 x float> %125) #9, !dbg !7
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %150, <4 x float> %151) #9, !dbg !7
  %153 = load <4 x half>, ptr addrspace(3) %address.local179, align 2, !dbg !7
  %154 = load <4 x half>, ptr addrspace(3) %address.local185, align 2, !dbg !7
  %155 = shufflevector <4 x float> %152, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %153, <4 x float> %126) #9, !dbg !7
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %112, <4 x half> %154, <4 x float> %156) #9, !dbg !7
  %158 = shufflevector <4 x float> %157, <4 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !36
  %159 = shufflevector <4 x float> %119, <4 x float> %131, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %160 = shufflevector <16 x float> %159, <16 x float> %137, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %161 = shufflevector <16 x float> %160, <16 x float> %141, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !21
  %162 = insertvalue [2 x <16 x float>] poison, <16 x float> %161, 0, !dbg !21
  %163 = shufflevector <4 x float> %144, <4 x float> %149, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %164 = shufflevector <16 x float> %163, <16 x float> %155, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 16, i32 17, i32 18, i32 19, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !21
  %165 = shufflevector <16 x float> %164, <16 x float> %158, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 16, i32 17, i32 18, i32 19>, !dbg !21
  %166 = insertvalue [2 x <16 x float>] %162, <16 x float> %165, 1, !dbg !21
  fence syncscope("workgroup") release, !dbg !21
  call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %167 = add nuw nsw i64 %42, 1, !dbg !21
  br label %41, !dbg !21

168:                                              ; preds = %41
  %169 = add nsw i64 %38, %6, !dbg !17
  %170 = shl i64 %169, 19, !dbg !17
  %171 = and i64 %170, 16252928, !dbg !17
  %172 = shl nsw i64 %40, 7, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %173 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %171, !dbg !17
  %174 = getelementptr [2 x i8], ptr addrspace(1) %173, i64 %172, !dbg !17
  %address.thread186 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !17, !range !39, !frisk.register_lifetimes !1
  %address.local187 = zext nneg i32 %address.thread186 to i64, !dbg !13
  %address.local188 = shl nuw nsw i64 %address.local187, 3, !dbg !12
  %address.local189 = and i64 %address.local188, 1536, !dbg !16
  %address.local190 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local189, !dbg !17
  %address.local191 = ptrtoint ptr addrspace(3) %address.local190 to i64, !dbg !17
  %address.local192 = trunc i64 %address.local191 to i32, !dbg !17
  %175 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local192), !dbg !17
  %176 = zext i32 %175 to i64, !dbg !17
  %177 = inttoptr i64 %176 to ptr addrspace(3), !dbg !17
  %address.local193 = shl nuw nsw i64 %address.local187, 5, !dbg !17
  %address.local194 = and i64 %address.local193, 8064, !dbg !17
  %address.local195 = shl nuw nsw i64 %address.local187, 3, !dbg !17
  %address.local196 = and i64 %address.local195, 24, !dbg !17
  %address.local197 = or disjoint i64 %address.local194, %address.local196, !dbg !17
  %address.local198 = getelementptr [2 x i8], ptr addrspace(1) %174, i64 %address.local197, !dbg !17
  %178 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %179 = ptrtoint ptr addrspace(1) %address.local198 to i64, !dbg !17
  %180 = sub i64 %179, %178, !dbg !17
  %dma.buffer.offset603 = trunc i64 %180 to i32, !dbg !17
  %181 = trunc i64 %178 to i32, !dbg !17
  %182 = insertelement <4 x i32> poison, i32 %181, i64 0, !dbg !17
  %183 = lshr i64 %178, 32, !dbg !17
  %184 = trunc i64 %183 to i32, !dbg !17
  %185 = insertelement <4 x i32> %182, i32 %184, i64 1, !dbg !17
  %186 = insertelement <4 x i32> %185, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc604 = insertelement <4 x i32> %186, i32 131072, i64 3, !dbg !17
  %187 = ptrtoint ptr addrspace(3) %177 to i32, !dbg !17
  %dma.buffer.lds605 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %187), !dbg !17
  %188 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds605, i32 %dma.buffer.offset603, <4 x i32> %dma.buffer.rsrc604) #10, !dbg !17
  %189 = or disjoint i64 %37, 63, !dbg !14
  %190 = icmp uge i64 %9, %189, !dbg !14
  %191 = and i1 %24, %190, !dbg !14
  br i1 %191, label %192, label %193, !dbg !14

192:                                              ; preds = %168
  br label %273, !dbg !14

193:                                              ; preds = %168
  %194 = add nsw i64 %9, %20, !dbg !14
  %195 = shl nuw nsw i64 %21, 4, !dbg !14
  %196 = add nsw i64 %194, %195, !dbg !14
  %197 = and i32 %19, 240, !dbg !14
  %index.compact = zext nneg i32 %197 to i64, !dbg !14
  %198 = sub nsw i64 %196, %index.compact, !dbg !14
  %199 = and i64 %20, 48, !dbg !14
  %200 = or disjoint i64 %37, %199, !dbg !14
  %201 = add nsw i64 %198, 1, !dbg !22
  %202 = lshr i64 %201, 32, !dbg !23
  %compare.high = trunc i64 %202 to i32, !dbg !23
  %203 = lshr i64 %200, 32, !dbg !23
  %compare.high460 = trunc i64 %203 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high460, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high460, !dbg !23
  %compare.low.base = trunc i64 %201 to i32, !dbg !23
  %compare.low.base461 = trunc i64 %200 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base461, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %204 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low462 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result463 = select i1 %compare.high.equal, i1 %compare.low462, i1 %compare.high.order, !dbg !23
  %205 = select i1 %compare.result463, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base464 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local465 = or i32 %compare.local.base464, 2, !dbg !23
  %compare.low466 = icmp ugt i32 %compare.low.base, %compare.local465, !dbg !23
  %compare.result467 = select i1 %compare.high.equal, i1 %compare.low466, i1 %compare.high.order, !dbg !23
  %206 = select i1 %compare.result467, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base468 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local469 = or i32 %compare.local.base468, 3, !dbg !23
  %compare.low470 = icmp ugt i32 %compare.low.base, %compare.local469, !dbg !23
  %compare.result471 = select i1 %compare.high.equal, i1 %compare.low470, i1 %compare.high.order, !dbg !23
  %207 = select i1 %compare.result471, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base472 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local473 = or i32 %compare.local.base472, 4, !dbg !23
  %compare.low474 = icmp ugt i32 %compare.low.base, %compare.local473, !dbg !23
  %compare.result475 = select i1 %compare.high.equal, i1 %compare.low474, i1 %compare.high.order, !dbg !23
  %208 = select i1 %compare.result475, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base476 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local477 = or i32 %compare.local.base476, 5, !dbg !23
  %compare.low478 = icmp ugt i32 %compare.low.base, %compare.local477, !dbg !23
  %compare.result479 = select i1 %compare.high.equal, i1 %compare.low478, i1 %compare.high.order, !dbg !23
  %209 = select i1 %compare.result479, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base480 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local481 = or i32 %compare.local.base480, 6, !dbg !23
  %compare.low482 = icmp ugt i32 %compare.low.base, %compare.local481, !dbg !23
  %compare.result483 = select i1 %compare.high.equal, i1 %compare.low482, i1 %compare.high.order, !dbg !23
  %210 = select i1 %compare.result483, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base484 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local485 = or i32 %compare.local.base484, 7, !dbg !23
  %compare.low486 = icmp ugt i32 %compare.low.base, %compare.local485, !dbg !23
  %compare.result487 = select i1 %compare.high.equal, i1 %compare.low486, i1 %compare.high.order, !dbg !23
  %211 = select i1 %compare.result487, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base488 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local489 = or i32 %compare.local.base488, 8, !dbg !23
  %compare.low490 = icmp ugt i32 %compare.low.base, %compare.local489, !dbg !23
  %compare.result491 = select i1 %compare.high.equal, i1 %compare.low490, i1 %compare.high.order, !dbg !23
  %212 = select i1 %compare.result491, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base492 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local493 = or i32 %compare.local.base492, 9, !dbg !23
  %compare.low494 = icmp ugt i32 %compare.low.base, %compare.local493, !dbg !23
  %compare.result495 = select i1 %compare.high.equal, i1 %compare.low494, i1 %compare.high.order, !dbg !23
  %213 = select i1 %compare.result495, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base496 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local497 = or i32 %compare.local.base496, 10, !dbg !23
  %compare.low498 = icmp ugt i32 %compare.low.base, %compare.local497, !dbg !23
  %compare.result499 = select i1 %compare.high.equal, i1 %compare.low498, i1 %compare.high.order, !dbg !23
  %214 = select i1 %compare.result499, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base500 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local501 = or i32 %compare.local.base500, 11, !dbg !23
  %compare.low502 = icmp ugt i32 %compare.low.base, %compare.local501, !dbg !23
  %compare.result503 = select i1 %compare.high.equal, i1 %compare.low502, i1 %compare.high.order, !dbg !23
  %215 = select i1 %compare.result503, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base504 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local505 = or i32 %compare.local.base504, 12, !dbg !23
  %compare.low506 = icmp ugt i32 %compare.low.base, %compare.local505, !dbg !23
  %compare.result507 = select i1 %compare.high.equal, i1 %compare.low506, i1 %compare.high.order, !dbg !23
  %216 = select i1 %compare.result507, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base508 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local509 = or i32 %compare.local.base508, 13, !dbg !23
  %compare.low510 = icmp ugt i32 %compare.low.base, %compare.local509, !dbg !23
  %compare.result511 = select i1 %compare.high.equal, i1 %compare.low510, i1 %compare.high.order, !dbg !23
  %217 = select i1 %compare.result511, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base512 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local513 = or i32 %compare.local.base512, 14, !dbg !23
  %compare.low514 = icmp ugt i32 %compare.low.base, %compare.local513, !dbg !23
  %compare.result515 = select i1 %compare.high.equal, i1 %compare.low514, i1 %compare.high.order, !dbg !23
  %218 = select i1 %compare.result515, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base516 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base461, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local517 = or i32 %compare.local.base516, 15, !dbg !23
  %compare.low518 = icmp ugt i32 %compare.low.base, %compare.local517, !dbg !23
  %compare.result519 = select i1 %compare.high.equal, i1 %compare.low518, i1 %compare.high.order, !dbg !23
  %219 = select i1 %compare.result519, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %220 = add nsw i64 %198, 65, !dbg !22
  %221 = lshr i64 %220, 32, !dbg !23
  %compare.high520 = trunc i64 %221 to i32, !dbg !23
  %222 = lshr i64 %200, 32, !dbg !23
  %compare.high521 = trunc i64 %222 to i32, !dbg !23
  %compare.high.equal522 = icmp eq i32 %compare.high520, %compare.high521, !dbg !23
  %compare.high.order523 = icmp ugt i32 %compare.high520, %compare.high521, !dbg !23
  %compare.low.base524 = trunc i64 %220 to i32, !dbg !23
  %compare.low.base525 = trunc i64 %200 to i32, !dbg !23
  %compare.low526 = icmp ugt i32 %compare.low.base524, %compare.low.base525, !dbg !23
  %compare.result527 = select i1 %compare.high.equal522, i1 %compare.low526, i1 %compare.high.order523, !dbg !23
  %223 = select i1 %compare.result527, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base528 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local529 = or i32 %compare.local.base528, 1, !dbg !23
  %compare.low530 = icmp ugt i32 %compare.low.base524, %compare.local529, !dbg !23
  %compare.result531 = select i1 %compare.high.equal522, i1 %compare.low530, i1 %compare.high.order523, !dbg !23
  %224 = select i1 %compare.result531, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base532 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local533 = or i32 %compare.local.base532, 2, !dbg !23
  %compare.low534 = icmp ugt i32 %compare.low.base524, %compare.local533, !dbg !23
  %compare.result535 = select i1 %compare.high.equal522, i1 %compare.low534, i1 %compare.high.order523, !dbg !23
  %225 = select i1 %compare.result535, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base536 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local537 = or i32 %compare.local.base536, 3, !dbg !23
  %compare.low538 = icmp ugt i32 %compare.low.base524, %compare.local537, !dbg !23
  %compare.result539 = select i1 %compare.high.equal522, i1 %compare.low538, i1 %compare.high.order523, !dbg !23
  %226 = select i1 %compare.result539, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base540 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local541 = or i32 %compare.local.base540, 4, !dbg !23
  %compare.low542 = icmp ugt i32 %compare.low.base524, %compare.local541, !dbg !23
  %compare.result543 = select i1 %compare.high.equal522, i1 %compare.low542, i1 %compare.high.order523, !dbg !23
  %227 = select i1 %compare.result543, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base544 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local545 = or i32 %compare.local.base544, 5, !dbg !23
  %compare.low546 = icmp ugt i32 %compare.low.base524, %compare.local545, !dbg !23
  %compare.result547 = select i1 %compare.high.equal522, i1 %compare.low546, i1 %compare.high.order523, !dbg !23
  %228 = select i1 %compare.result547, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base548 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local549 = or i32 %compare.local.base548, 6, !dbg !23
  %compare.low550 = icmp ugt i32 %compare.low.base524, %compare.local549, !dbg !23
  %compare.result551 = select i1 %compare.high.equal522, i1 %compare.low550, i1 %compare.high.order523, !dbg !23
  %229 = select i1 %compare.result551, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base552 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local553 = or i32 %compare.local.base552, 7, !dbg !23
  %compare.low554 = icmp ugt i32 %compare.low.base524, %compare.local553, !dbg !23
  %compare.result555 = select i1 %compare.high.equal522, i1 %compare.low554, i1 %compare.high.order523, !dbg !23
  %230 = select i1 %compare.result555, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base556 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local557 = or i32 %compare.local.base556, 8, !dbg !23
  %compare.low558 = icmp ugt i32 %compare.low.base524, %compare.local557, !dbg !23
  %compare.result559 = select i1 %compare.high.equal522, i1 %compare.low558, i1 %compare.high.order523, !dbg !23
  %231 = select i1 %compare.result559, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base560 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local561 = or i32 %compare.local.base560, 9, !dbg !23
  %compare.low562 = icmp ugt i32 %compare.low.base524, %compare.local561, !dbg !23
  %compare.result563 = select i1 %compare.high.equal522, i1 %compare.low562, i1 %compare.high.order523, !dbg !23
  %232 = select i1 %compare.result563, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base564 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local565 = or i32 %compare.local.base564, 10, !dbg !23
  %compare.low566 = icmp ugt i32 %compare.low.base524, %compare.local565, !dbg !23
  %compare.result567 = select i1 %compare.high.equal522, i1 %compare.low566, i1 %compare.high.order523, !dbg !23
  %233 = select i1 %compare.result567, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base568 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local569 = or i32 %compare.local.base568, 11, !dbg !23
  %compare.low570 = icmp ugt i32 %compare.low.base524, %compare.local569, !dbg !23
  %compare.result571 = select i1 %compare.high.equal522, i1 %compare.low570, i1 %compare.high.order523, !dbg !23
  %234 = select i1 %compare.result571, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base572 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local573 = or i32 %compare.local.base572, 12, !dbg !23
  %compare.low574 = icmp ugt i32 %compare.low.base524, %compare.local573, !dbg !23
  %compare.result575 = select i1 %compare.high.equal522, i1 %compare.low574, i1 %compare.high.order523, !dbg !23
  %235 = select i1 %compare.result575, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base576 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local577 = or i32 %compare.local.base576, 13, !dbg !23
  %compare.low578 = icmp ugt i32 %compare.low.base524, %compare.local577, !dbg !23
  %compare.result579 = select i1 %compare.high.equal522, i1 %compare.low578, i1 %compare.high.order523, !dbg !23
  %236 = select i1 %compare.result579, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base580 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local581 = or i32 %compare.local.base580, 14, !dbg !23
  %compare.low582 = icmp ugt i32 %compare.low.base524, %compare.local581, !dbg !23
  %compare.result583 = select i1 %compare.high.equal522, i1 %compare.low582, i1 %compare.high.order523, !dbg !23
  %237 = select i1 %compare.result583, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base584 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base525, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local585 = or i32 %compare.local.base584, 15, !dbg !23
  %compare.low586 = icmp ugt i32 %compare.low.base524, %compare.local585, !dbg !23
  %compare.result587 = select i1 %compare.high.equal522, i1 %compare.low586, i1 %compare.high.order523, !dbg !23
  %238 = select i1 %compare.result587, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %239 = insertelement <16 x float> poison, float %204, i64 0, !dbg !14
  %240 = insertelement <16 x float> %239, float %205, i64 1, !dbg !14
  %241 = insertelement <16 x float> %240, float %206, i64 2, !dbg !14
  %242 = insertelement <16 x float> %241, float %207, i64 3, !dbg !14
  %243 = insertelement <16 x float> %242, float %208, i64 4, !dbg !14
  %244 = insertelement <16 x float> %243, float %209, i64 5, !dbg !14
  %245 = insertelement <16 x float> %244, float %210, i64 6, !dbg !14
  %246 = insertelement <16 x float> %245, float %211, i64 7, !dbg !14
  %247 = insertelement <16 x float> %246, float %212, i64 8, !dbg !14
  %248 = insertelement <16 x float> %247, float %213, i64 9, !dbg !14
  %249 = insertelement <16 x float> %248, float %214, i64 10, !dbg !14
  %250 = insertelement <16 x float> %249, float %215, i64 11, !dbg !14
  %251 = insertelement <16 x float> %250, float %216, i64 12, !dbg !14
  %252 = insertelement <16 x float> %251, float %217, i64 13, !dbg !14
  %253 = insertelement <16 x float> %252, float %218, i64 14, !dbg !14
  %254 = insertelement <16 x float> %253, float %219, i64 15, !dbg !14
  %255 = insertvalue [2 x <16 x float>] poison, <16 x float> %254, 0, !dbg !14
  %256 = insertelement <16 x float> poison, float %223, i64 0, !dbg !14
  %257 = insertelement <16 x float> %256, float %224, i64 1, !dbg !14
  %258 = insertelement <16 x float> %257, float %225, i64 2, !dbg !14
  %259 = insertelement <16 x float> %258, float %226, i64 3, !dbg !14
  %260 = insertelement <16 x float> %259, float %227, i64 4, !dbg !14
  %261 = insertelement <16 x float> %260, float %228, i64 5, !dbg !14
  %262 = insertelement <16 x float> %261, float %229, i64 6, !dbg !14
  %263 = insertelement <16 x float> %262, float %230, i64 7, !dbg !14
  %264 = insertelement <16 x float> %263, float %231, i64 8, !dbg !14
  %265 = insertelement <16 x float> %264, float %232, i64 9, !dbg !14
  %266 = insertelement <16 x float> %265, float %233, i64 10, !dbg !14
  %267 = insertelement <16 x float> %266, float %234, i64 11, !dbg !14
  %268 = insertelement <16 x float> %267, float %235, i64 12, !dbg !14
  %269 = insertelement <16 x float> %268, float %236, i64 13, !dbg !14
  %270 = insertelement <16 x float> %269, float %237, i64 14, !dbg !14
  %271 = insertelement <16 x float> %270, float %238, i64 15, !dbg !14
  %272 = insertvalue [2 x <16 x float>] %255, <16 x float> %271, 1, !dbg !14
  br label %273, !dbg !14

273:                                              ; preds = %192, %193
  %274 = phi [2 x <16 x float>] [ %272, %193 ], [ zeroinitializer, %192 ], !dbg !14
  br label %275, !dbg !14

275:                                              ; preds = %273
  %276 = extractvalue [2 x <16 x float>] %43, 0, !dbg !25
  %277 = extractvalue [2 x <16 x float>] %274, 0, !dbg !25
  %278 = shufflevector <16 x float> %276, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %279 = shufflevector <16 x float> %277, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %280 = fadd <4 x float> %278, %279, !dbg !25
  %281 = extractelement <4 x float> %280, i64 0, !dbg !25
  %282 = extractelement <4 x float> %280, i64 1, !dbg !25
  %283 = extractelement <4 x float> %280, i64 2, !dbg !25
  %284 = extractelement <4 x float> %280, i64 3, !dbg !25
  %285 = shufflevector <16 x float> %276, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %286 = shufflevector <16 x float> %277, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %287 = fadd <4 x float> %285, %286, !dbg !25
  %288 = extractelement <4 x float> %287, i64 0, !dbg !25
  %289 = extractelement <4 x float> %287, i64 1, !dbg !25
  %290 = extractelement <4 x float> %287, i64 2, !dbg !25
  %291 = extractelement <4 x float> %287, i64 3, !dbg !25
  %292 = shufflevector <16 x float> %276, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %293 = shufflevector <16 x float> %277, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %294 = fadd <4 x float> %292, %293, !dbg !25
  %295 = extractelement <4 x float> %294, i64 0, !dbg !25
  %296 = extractelement <4 x float> %294, i64 1, !dbg !25
  %297 = extractelement <4 x float> %294, i64 2, !dbg !25
  %298 = extractelement <4 x float> %294, i64 3, !dbg !25
  %299 = shufflevector <16 x float> %276, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %300 = shufflevector <16 x float> %277, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %301 = fadd <4 x float> %299, %300, !dbg !25
  %302 = extractelement <4 x float> %301, i64 0, !dbg !25
  %303 = extractelement <4 x float> %301, i64 1, !dbg !25
  %304 = extractelement <4 x float> %301, i64 2, !dbg !25
  %305 = extractelement <4 x float> %301, i64 3, !dbg !25
  %306 = extractvalue [2 x <16 x float>] %43, 1, !dbg !25
  %307 = extractvalue [2 x <16 x float>] %274, 1, !dbg !25
  %308 = shufflevector <16 x float> %306, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %309 = shufflevector <16 x float> %307, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %310 = fadd <4 x float> %308, %309, !dbg !25
  %311 = extractelement <4 x float> %310, i64 0, !dbg !25
  %312 = extractelement <4 x float> %310, i64 1, !dbg !25
  %313 = extractelement <4 x float> %310, i64 2, !dbg !25
  %314 = extractelement <4 x float> %310, i64 3, !dbg !25
  %315 = shufflevector <16 x float> %306, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %316 = shufflevector <16 x float> %307, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %317 = fadd <4 x float> %315, %316, !dbg !25
  %318 = extractelement <4 x float> %317, i64 0, !dbg !25
  %319 = extractelement <4 x float> %317, i64 1, !dbg !25
  %320 = extractelement <4 x float> %317, i64 2, !dbg !25
  %321 = extractelement <4 x float> %317, i64 3, !dbg !25
  %322 = shufflevector <16 x float> %306, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %323 = shufflevector <16 x float> %307, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %324 = fadd <4 x float> %322, %323, !dbg !25
  %325 = extractelement <4 x float> %324, i64 0, !dbg !25
  %326 = extractelement <4 x float> %324, i64 1, !dbg !25
  %327 = extractelement <4 x float> %324, i64 2, !dbg !25
  %328 = extractelement <4 x float> %324, i64 3, !dbg !25
  %329 = shufflevector <16 x float> %306, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %330 = shufflevector <16 x float> %307, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %331 = fadd <4 x float> %329, %330, !dbg !25
  %332 = extractelement <4 x float> %331, i64 0, !dbg !25
  %333 = extractelement <4 x float> %331, i64 1, !dbg !25
  %334 = extractelement <4 x float> %331, i64 2, !dbg !25
  %335 = extractelement <4 x float> %331, i64 3, !dbg !25
  %336 = call float @llvm.maxnum.f32(float %281, float 0xFFF0000000000000), !dbg !26
  %337 = call float @llvm.maxnum.f32(float %336, float %282), !dbg !26
  %338 = call float @llvm.maxnum.f32(float %337, float %283), !dbg !26
  %339 = call float @llvm.maxnum.f32(float %338, float %284), !dbg !26
  %340 = call float @llvm.maxnum.f32(float %339, float %288), !dbg !26
  %341 = call float @llvm.maxnum.f32(float %340, float %289), !dbg !26
  %342 = call float @llvm.maxnum.f32(float %341, float %290), !dbg !26
  %343 = call float @llvm.maxnum.f32(float %342, float %291), !dbg !26
  %344 = call float @llvm.maxnum.f32(float %343, float %295), !dbg !26
  %345 = call float @llvm.maxnum.f32(float %344, float %296), !dbg !26
  %346 = call float @llvm.maxnum.f32(float %345, float %297), !dbg !26
  %347 = call float @llvm.maxnum.f32(float %346, float %298), !dbg !26
  %348 = call float @llvm.maxnum.f32(float %347, float %302), !dbg !26
  %349 = call float @llvm.maxnum.f32(float %348, float %303), !dbg !26
  %350 = call float @llvm.maxnum.f32(float %349, float %304), !dbg !26
  %351 = call float @llvm.maxnum.f32(float %350, float %305), !dbg !26
  %352 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %353 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %352), !dbg !26
  %354 = add nuw nsw i32 %353, 64, !dbg !26
  %355 = xor i32 %353, 16, !dbg !26
  %356 = and i32 %354, 192, !dbg !26
  %357 = icmp samesign ult i32 %355, %356, !dbg !26
  %358 = select i1 %357, i32 %355, i32 %353, !dbg !26
  %359 = shl nuw nsw i32 %358, 2, !dbg !26
  %360 = bitcast float %351 to i32, !dbg !26
  %361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %359, i32 %360), !dbg !26
  %362 = bitcast i32 %361 to float, !dbg !26
  %363 = call float @llvm.maxnum.f32(float %351, float %362), !dbg !26
  %364 = xor i32 %353, 32, !dbg !26
  %365 = icmp samesign ult i32 %364, %356, !dbg !26
  %366 = select i1 %365, i32 %364, i32 %353, !dbg !26
  %367 = shl nuw nsw i32 %366, 2, !dbg !26
  %368 = bitcast float %363 to i32, !dbg !26
  %369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %367, i32 %368), !dbg !26
  %370 = bitcast i32 %369 to float, !dbg !26
  %371 = call float @llvm.maxnum.f32(float %363, float %370), !dbg !26
  %372 = extractvalue [2 x <1 x float>] %31, 0, !dbg !26
  %373 = insertelement <1 x float> poison, float %371, i64 0, !dbg !26
  %374 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %373, <1 x float> %372), !dbg !26
  %375 = call float @llvm.maxnum.f32(float %311, float 0xFFF0000000000000), !dbg !26
  %376 = call float @llvm.maxnum.f32(float %375, float %312), !dbg !26
  %377 = call float @llvm.maxnum.f32(float %376, float %313), !dbg !26
  %378 = call float @llvm.maxnum.f32(float %377, float %314), !dbg !26
  %379 = call float @llvm.maxnum.f32(float %378, float %318), !dbg !26
  %380 = call float @llvm.maxnum.f32(float %379, float %319), !dbg !26
  %381 = call float @llvm.maxnum.f32(float %380, float %320), !dbg !26
  %382 = call float @llvm.maxnum.f32(float %381, float %321), !dbg !26
  %383 = call float @llvm.maxnum.f32(float %382, float %325), !dbg !26
  %384 = call float @llvm.maxnum.f32(float %383, float %326), !dbg !26
  %385 = call float @llvm.maxnum.f32(float %384, float %327), !dbg !26
  %386 = call float @llvm.maxnum.f32(float %385, float %328), !dbg !26
  %387 = call float @llvm.maxnum.f32(float %386, float %332), !dbg !26
  %388 = call float @llvm.maxnum.f32(float %387, float %333), !dbg !26
  %389 = call float @llvm.maxnum.f32(float %388, float %334), !dbg !26
  %390 = call float @llvm.maxnum.f32(float %389, float %335), !dbg !26
  %391 = bitcast float %390 to i32, !dbg !26
  %392 = call i32 @llvm.amdgcn.ds.bpermute(i32 %359, i32 %391), !dbg !26
  %393 = bitcast i32 %392 to float, !dbg !26
  %394 = call float @llvm.maxnum.f32(float %390, float %393), !dbg !26
  %395 = bitcast float %394 to i32, !dbg !26
  %396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %367, i32 %395), !dbg !26
  %397 = bitcast i32 %396 to float, !dbg !26
  %398 = call float @llvm.maxnum.f32(float %394, float %397), !dbg !26
  %399 = extractvalue [2 x <1 x float>] %31, 1, !dbg !26
  %400 = insertelement <1 x float> poison, float %398, i64 0, !dbg !26
  %401 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %400, <1 x float> %399), !dbg !26
  %402 = shufflevector <1 x float> %374, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %403 = fsub <4 x float> %280, %402, !dbg !27
  %404 = fsub <4 x float> %287, %402, !dbg !27
  %405 = fsub <4 x float> %294, %402, !dbg !27
  %406 = fsub <4 x float> %301, %402, !dbg !27
  %407 = shufflevector <1 x float> %401, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %408 = fsub <4 x float> %310, %407, !dbg !27
  %409 = fsub <4 x float> %317, %407, !dbg !27
  %410 = fsub <4 x float> %324, %407, !dbg !27
  %411 = fsub <4 x float> %331, %407, !dbg !27
  %412 = fsub <1 x float> %372, %374, !dbg !28
  %413 = extractelement <1 x float> %412, i64 0, !dbg !29
  %414 = call float @llvm.amdgcn.exp2.f32(float %413), !dbg !29
  %415 = insertelement <4 x float> poison, float %414, i64 0, !dbg !34
  %416 = shufflevector <4 x float> %415, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %417 = extractvalue [2 x <1 x float>] %30, 0, !dbg !30
  %418 = extractelement <1 x float> %417, i64 0, !dbg !30
  %419 = fmul float %418, %414, !dbg !30
  %420 = fsub <1 x float> %399, %401, !dbg !28
  %421 = extractelement <1 x float> %420, i64 0, !dbg !29
  %422 = call float @llvm.amdgcn.exp2.f32(float %421), !dbg !29
  %423 = insertelement <4 x float> poison, float %422, i64 0, !dbg !34
  %424 = shufflevector <4 x float> %423, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %425 = extractvalue [2 x <1 x float>] %30, 1, !dbg !30
  %426 = extractelement <1 x float> %425, i64 0, !dbg !30
  %427 = fmul float %426, %422, !dbg !30
  %428 = extractelement <4 x float> %403, i64 0, !dbg !31
  %429 = call float @llvm.amdgcn.exp2.f32(float %428), !dbg !31
  %430 = insertelement <4 x float> poison, float %429, i64 0, !dbg !33
  %431 = extractelement <4 x float> %403, i64 1, !dbg !31
  %432 = call float @llvm.amdgcn.exp2.f32(float %431), !dbg !31
  %433 = insertelement <4 x float> %430, float %432, i64 1, !dbg !33
  %434 = fadd float %429, %432, !dbg !32
  %435 = extractelement <4 x float> %403, i64 2, !dbg !31
  %436 = call float @llvm.amdgcn.exp2.f32(float %435), !dbg !31
  %437 = insertelement <4 x float> %433, float %436, i64 2, !dbg !33
  %438 = fadd float %434, %436, !dbg !32
  %439 = extractelement <4 x float> %403, i64 3, !dbg !31
  %440 = call float @llvm.amdgcn.exp2.f32(float %439), !dbg !31
  %441 = insertelement <4 x float> %437, float %440, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %441) #9, !dbg !33, !frisk.register_lifetimes !1
  %442 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %442) #9, !dbg !32, !frisk.register_lifetimes !1
  %443 = fadd float %438, %440, !dbg !32
  %444 = extractelement <4 x float> %404, i64 0, !dbg !31
  %445 = call float @llvm.amdgcn.exp2.f32(float %444), !dbg !31
  %446 = insertelement <4 x float> poison, float %445, i64 0, !dbg !33
  %447 = fadd float %443, %445, !dbg !32
  %448 = extractelement <4 x float> %404, i64 1, !dbg !31
  %449 = call float @llvm.amdgcn.exp2.f32(float %448), !dbg !31
  %450 = insertelement <4 x float> %446, float %449, i64 1, !dbg !33
  %451 = fadd float %447, %449, !dbg !32
  %452 = extractelement <4 x float> %404, i64 2, !dbg !31
  %453 = call float @llvm.amdgcn.exp2.f32(float %452), !dbg !31
  %454 = insertelement <4 x float> %450, float %453, i64 2, !dbg !33
  %455 = fadd float %451, %453, !dbg !32
  %456 = extractelement <4 x float> %404, i64 3, !dbg !31
  %457 = call float @llvm.amdgcn.exp2.f32(float %456), !dbg !31
  %458 = insertelement <4 x float> %454, float %457, i64 3, !dbg !33
  %math.result63 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %458) #9, !dbg !33, !frisk.register_lifetimes !1
  %459 = fptrunc <4 x float> %math.result63 to <4 x half>, !dbg !33
  %math.packed64 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %459) #9, !dbg !32, !frisk.register_lifetimes !1
  %460 = fadd float %455, %457, !dbg !32
  %461 = extractelement <4 x float> %405, i64 0, !dbg !31
  %462 = call float @llvm.amdgcn.exp2.f32(float %461), !dbg !31
  %463 = insertelement <4 x float> poison, float %462, i64 0, !dbg !33
  %464 = fadd float %460, %462, !dbg !32
  %465 = extractelement <4 x float> %405, i64 1, !dbg !31
  %466 = call float @llvm.amdgcn.exp2.f32(float %465), !dbg !31
  %467 = insertelement <4 x float> %463, float %466, i64 1, !dbg !33
  %468 = fadd float %464, %466, !dbg !32
  %469 = extractelement <4 x float> %405, i64 2, !dbg !31
  %470 = call float @llvm.amdgcn.exp2.f32(float %469), !dbg !31
  %471 = insertelement <4 x float> %467, float %470, i64 2, !dbg !33
  %472 = fadd float %468, %470, !dbg !32
  %473 = extractelement <4 x float> %405, i64 3, !dbg !31
  %474 = call float @llvm.amdgcn.exp2.f32(float %473), !dbg !31
  %475 = insertelement <4 x float> %471, float %474, i64 3, !dbg !33
  %math.result65 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %475) #9, !dbg !33, !frisk.register_lifetimes !1
  %476 = fptrunc <4 x float> %math.result65 to <4 x half>, !dbg !33
  %math.packed66 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %476) #9, !dbg !32, !frisk.register_lifetimes !1
  %477 = fadd float %472, %474, !dbg !32
  %478 = extractelement <4 x float> %406, i64 0, !dbg !31
  %479 = call float @llvm.amdgcn.exp2.f32(float %478), !dbg !31
  %480 = insertelement <4 x float> poison, float %479, i64 0, !dbg !33
  %481 = fadd float %477, %479, !dbg !32
  %482 = extractelement <4 x float> %406, i64 1, !dbg !31
  %483 = call float @llvm.amdgcn.exp2.f32(float %482), !dbg !31
  %484 = insertelement <4 x float> %480, float %483, i64 1, !dbg !33
  %485 = fadd float %481, %483, !dbg !32
  %486 = extractelement <4 x float> %406, i64 2, !dbg !31
  %487 = call float @llvm.amdgcn.exp2.f32(float %486), !dbg !31
  %488 = insertelement <4 x float> %484, float %487, i64 2, !dbg !33
  %489 = fadd float %485, %487, !dbg !32
  %490 = extractelement <4 x float> %406, i64 3, !dbg !31
  %491 = call float @llvm.amdgcn.exp2.f32(float %490), !dbg !31
  %492 = insertelement <4 x float> %488, float %491, i64 3, !dbg !33
  %math.result67 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %492) #9, !dbg !33, !frisk.register_lifetimes !1
  %493 = fptrunc <4 x float> %math.result67 to <4 x half>, !dbg !33
  %math.packed68 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %493) #9, !dbg !32, !frisk.register_lifetimes !1
  %494 = fadd float %489, %491, !dbg !32
  %495 = extractelement <4 x float> %408, i64 0, !dbg !31
  %496 = call float @llvm.amdgcn.exp2.f32(float %495), !dbg !31
  %497 = insertelement <4 x float> poison, float %496, i64 0, !dbg !33
  %498 = extractelement <4 x float> %408, i64 1, !dbg !31
  %499 = call float @llvm.amdgcn.exp2.f32(float %498), !dbg !31
  %500 = insertelement <4 x float> %497, float %499, i64 1, !dbg !33
  %501 = fadd float %496, %499, !dbg !32
  %502 = extractelement <4 x float> %408, i64 2, !dbg !31
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !31
  %504 = insertelement <4 x float> %500, float %503, i64 2, !dbg !33
  %505 = fadd float %501, %503, !dbg !32
  %506 = extractelement <4 x float> %408, i64 3, !dbg !31
  %507 = call float @llvm.amdgcn.exp2.f32(float %506), !dbg !31
  %508 = insertelement <4 x float> %504, float %507, i64 3, !dbg !33
  %math.result69 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %508) #9, !dbg !33, !frisk.register_lifetimes !1
  %509 = fptrunc <4 x float> %math.result69 to <4 x half>, !dbg !33
  %math.packed70 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %509) #9, !dbg !32, !frisk.register_lifetimes !1
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
  %math.result71 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %525) #9, !dbg !33, !frisk.register_lifetimes !1
  %526 = fptrunc <4 x float> %math.result71 to <4 x half>, !dbg !33
  %math.packed72 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %526) #9, !dbg !32, !frisk.register_lifetimes !1
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
  %math.result73 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #9, !dbg !33, !frisk.register_lifetimes !1
  %543 = fptrunc <4 x float> %math.result73 to <4 x half>, !dbg !33
  %math.packed74 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #9, !dbg !32, !frisk.register_lifetimes !1
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
  %math.result75 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %559) #9, !dbg !33, !frisk.register_lifetimes !1
  %560 = fptrunc <4 x float> %math.result75 to <4 x half>, !dbg !33
  %math.packed76 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %560) #9, !dbg !32, !frisk.register_lifetimes !1
  %561 = fadd float %556, %558, !dbg !32
  %562 = bitcast float %494 to i32, !dbg !32
  %563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %359, i32 %562), !dbg !32
  %564 = bitcast i32 %563 to float, !dbg !32
  %565 = fadd float %494, %564, !dbg !32
  %566 = bitcast float %565 to i32, !dbg !32
  %567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %367, i32 %566), !dbg !32
  %568 = bitcast i32 %567 to float, !dbg !32
  %569 = fadd float %565, %568, !dbg !32
  %570 = fadd float %569, %419, !dbg !32
  %571 = insertelement <1 x float> poison, float %570, i64 0, !dbg !8
  %572 = insertvalue [2 x <1 x float>] poison, <1 x float> %571, 0, !dbg !8
  %573 = bitcast float %561 to i32, !dbg !32
  %574 = call i32 @llvm.amdgcn.ds.bpermute(i32 %359, i32 %573), !dbg !32
  %575 = bitcast i32 %574 to float, !dbg !32
  %576 = fadd float %561, %575, !dbg !32
  %577 = bitcast float %576 to i32, !dbg !32
  %578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %367, i32 %577), !dbg !32
  %579 = bitcast i32 %578 to float, !dbg !32
  %580 = fadd float %576, %579, !dbg !32
  %581 = fadd float %580, %427, !dbg !32
  %582 = insertelement <1 x float> poison, float %581, i64 0, !dbg !8
  %583 = insertvalue [2 x <1 x float>] %572, <1 x float> %582, 1, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread199 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local200 = zext nneg i32 %address.thread199 to i64, !dbg !13
  %address.local201 = and i64 %address.local200, 60, !dbg !15
  %address.local202 = add nuw nsw i64 %address.local201, %address.local200, !dbg !15
  %address.local203 = shl nuw nsw i64 %address.local202, 3, !dbg !15
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local203, !dbg !15
  %address.local205 = and i64 %address.local200, 63, !dbg !13
  %address.local206 = lshr i64 %address.local205, 4, !dbg !15
  %address.local207 = mul nuw nsw i64 %address.local206, 384, !dbg !15
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) %address.local204, i64 %address.local207, !dbg !15
  %address.local209 = lshr i64 %address.local200, 2, !dbg !7
  %address.local210 = mul nsw i64 %address.local209, -32, !dbg !15
  %address.local211 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local210, !dbg !15
  %584 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local211, i32 0) #0, !dbg !15
  %585 = bitcast <4 x i32> %584 to <8 x half>, !dbg !15
  %address.local212 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local203, !dbg !15
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local212, i64 %address.local207, !dbg !15
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local210, !dbg !15
  %address.local215 = getelementptr i8, ptr addrspace(3) %address.local214, i64 256, !dbg !15
  %586 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !15
  %587 = bitcast <4 x i32> %586 to <8 x half>, !dbg !15
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local203, !dbg !15
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local207, !dbg !15
  %address.local218 = getelementptr [2 x i8], ptr addrspace(3) %address.local217, i64 %address.local210, !dbg !15
  %address.local219 = getelementptr i8, ptr addrspace(3) %address.local218, i64 512, !dbg !15
  %588 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !15
  %589 = bitcast <4 x i32> %588 to <8 x half>, !dbg !15
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local203, !dbg !15
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local207, !dbg !15
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local210, !dbg !15
  %address.local223 = getelementptr i8, ptr addrspace(3) %address.local222, i64 768, !dbg !15
  %590 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !15
  %591 = bitcast <4 x i32> %590 to <8 x half>, !dbg !15
  %592 = add nsw i64 %38, %6, !dbg !17
  %593 = shl i64 %592, 19, !dbg !17
  %594 = and i64 %593, 16252928, !dbg !17
  %595 = add i64 %594, %172, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %596 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %595, !dbg !17
  %597 = getelementptr i8, ptr addrspace(1) %596, i64 64, !dbg !17
  %address.local224 = shl nuw nsw i64 %address.local200, 3, !dbg !12
  %address.local225 = and i64 %address.local224, 1536, !dbg !16
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local225, !dbg !17
  %address.local227 = ptrtoint ptr addrspace(3) %address.local226 to i64, !dbg !17
  %address.local228 = trunc i64 %address.local227 to i32, !dbg !17
  %598 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local228), !dbg !17
  %599 = zext i32 %598 to i64, !dbg !17
  %600 = inttoptr i64 %599 to ptr addrspace(3), !dbg !17
  %address.local229 = shl nuw nsw i64 %address.local200, 5, !dbg !17
  %address.local230 = and i64 %address.local229, 8064, !dbg !17
  %address.local231 = shl nuw nsw i64 %address.local200, 3, !dbg !17
  %address.local232 = and i64 %address.local231, 24, !dbg !17
  %address.local233 = or disjoint i64 %address.local230, %address.local232, !dbg !17
  %address.local234 = getelementptr [2 x i8], ptr addrspace(1) %597, i64 %address.local233, !dbg !17
  %601 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %602 = ptrtoint ptr addrspace(1) %address.local234 to i64, !dbg !17
  %603 = sub i64 %602, %601, !dbg !17
  %dma.buffer.offset600 = trunc i64 %603 to i32, !dbg !17
  %604 = trunc i64 %601 to i32, !dbg !17
  %605 = insertelement <4 x i32> poison, i32 %604, i64 0, !dbg !17
  %606 = lshr i64 %601, 32, !dbg !17
  %607 = trunc i64 %606 to i32, !dbg !17
  %608 = insertelement <4 x i32> %605, i32 %607, i64 1, !dbg !17
  %609 = insertelement <4 x i32> %608, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc601 = insertelement <4 x i32> %609, i32 131072, i64 3, !dbg !17
  %610 = ptrtoint ptr addrspace(3) %600 to i32, !dbg !17
  %dma.buffer.lds602 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %610), !dbg !17
  %611 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds602, i32 %dma.buffer.offset600, <4 x i32> %dma.buffer.rsrc601) #10, !dbg !17
  %612 = shufflevector <8 x half> %585, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %613 = extractvalue [2 x <8 x float>] %29, 0, !dbg !34
  %614 = shufflevector <8 x float> %613, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %615 = fmul <4 x float> %614, %416, !dbg !34
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %612, <4 x float> %615) #9, !dbg !15
  %617 = shufflevector <8 x half> %587, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %617, <4 x float> %616) #9, !dbg !15
  %619 = shufflevector <8 x half> %589, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %619, <4 x float> %618) #9, !dbg !15
  %621 = shufflevector <8 x half> %591, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %621, <4 x float> %620) #9, !dbg !15
  %623 = shufflevector <8 x half> %585, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %624 = shufflevector <8 x float> %613, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %625 = fmul <4 x float> %624, %416, !dbg !34
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %623, <4 x float> %625) #9, !dbg !15
  %627 = shufflevector <8 x half> %587, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %627, <4 x float> %626) #9, !dbg !15
  %629 = shufflevector <8 x half> %589, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %630 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %629, <4 x float> %628) #9, !dbg !15
  %631 = shufflevector <8 x half> %591, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %631, <4 x float> %630) #9, !dbg !15
  %633 = extractvalue [2 x <8 x float>] %29, 1, !dbg !34
  %634 = shufflevector <8 x float> %633, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %635 = fmul <4 x float> %634, %424, !dbg !34
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %612, <4 x float> %635) #9, !dbg !15
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %617, <4 x float> %636) #9, !dbg !15
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %619, <4 x float> %637) #9, !dbg !15
  %639 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %621, <4 x float> %638) #9, !dbg !15
  %640 = shufflevector <8 x float> %633, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %641 = fmul <4 x float> %640, %424, !dbg !34
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %623, <4 x float> %641) #9, !dbg !15
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %627, <4 x float> %642) #9, !dbg !15
  %644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %629, <4 x float> %643) #9, !dbg !15
  %645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %631, <4 x float> %644) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %646 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local211, i32 0) #0, !dbg !15
  %647 = bitcast <4 x i32> %646 to <8 x half>, !dbg !15
  %648 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !15
  %649 = bitcast <4 x i32> %648 to <8 x half>, !dbg !15
  %650 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !15
  %651 = bitcast <4 x i32> %650 to <8 x half>, !dbg !15
  %652 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !15
  %653 = bitcast <4 x i32> %652 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %654 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %595, !dbg !17
  %655 = getelementptr i8, ptr addrspace(1) %654, i64 128, !dbg !17
  %656 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local228), !dbg !17
  %657 = zext i32 %656 to i64, !dbg !17
  %658 = inttoptr i64 %657 to ptr addrspace(3), !dbg !17
  %address.local235 = getelementptr [2 x i8], ptr addrspace(1) %655, i64 %address.local233, !dbg !17
  %659 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %660 = ptrtoint ptr addrspace(1) %address.local235 to i64, !dbg !17
  %661 = sub i64 %660, %659, !dbg !17
  %dma.buffer.offset597 = trunc i64 %661 to i32, !dbg !17
  %662 = trunc i64 %659 to i32, !dbg !17
  %663 = insertelement <4 x i32> poison, i32 %662, i64 0, !dbg !17
  %664 = lshr i64 %659, 32, !dbg !17
  %665 = trunc i64 %664 to i32, !dbg !17
  %666 = insertelement <4 x i32> %663, i32 %665, i64 1, !dbg !17
  %667 = insertelement <4 x i32> %666, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc598 = insertelement <4 x i32> %667, i32 131072, i64 3, !dbg !17
  %668 = ptrtoint ptr addrspace(3) %658 to i32, !dbg !17
  %dma.buffer.lds599 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %668), !dbg !17
  %669 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds599, i32 %dma.buffer.offset597, <4 x i32> %dma.buffer.rsrc598) #10, !dbg !17
  %670 = shufflevector <8 x half> %647, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %671 = extractvalue [2 x <8 x float>] %32, 0, !dbg !34
  %672 = shufflevector <8 x float> %671, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %673 = fmul <4 x float> %672, %416, !dbg !34
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %670, <4 x float> %673) #9, !dbg !15
  %675 = shufflevector <8 x half> %649, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %675, <4 x float> %674) #9, !dbg !15
  %677 = shufflevector <8 x half> %651, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %677, <4 x float> %676) #9, !dbg !15
  %679 = shufflevector <8 x half> %653, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %680 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %679, <4 x float> %678) #9, !dbg !15
  %681 = shufflevector <8 x half> %647, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %682 = shufflevector <8 x float> %671, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %683 = fmul <4 x float> %682, %416, !dbg !34
  %684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %681, <4 x float> %683) #9, !dbg !15
  %685 = shufflevector <8 x half> %649, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %686 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %685, <4 x float> %684) #9, !dbg !15
  %687 = shufflevector <8 x half> %651, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %687, <4 x float> %686) #9, !dbg !15
  %689 = shufflevector <8 x half> %653, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %690 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %689, <4 x float> %688) #9, !dbg !15
  %691 = extractvalue [2 x <8 x float>] %32, 1, !dbg !34
  %692 = shufflevector <8 x float> %691, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %693 = fmul <4 x float> %692, %424, !dbg !34
  %694 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %670, <4 x float> %693) #9, !dbg !15
  %695 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %675, <4 x float> %694) #9, !dbg !15
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %677, <4 x float> %695) #9, !dbg !15
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %679, <4 x float> %696) #9, !dbg !15
  %698 = shufflevector <8 x float> %691, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %699 = fmul <4 x float> %698, %424, !dbg !34
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %681, <4 x float> %699) #9, !dbg !15
  %701 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %685, <4 x float> %700) #9, !dbg !15
  %702 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %687, <4 x float> %701) #9, !dbg !15
  %703 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %689, <4 x float> %702) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %704 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local211, i32 0) #0, !dbg !15
  %705 = bitcast <4 x i32> %704 to <8 x half>, !dbg !15
  %706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !15
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !15
  %708 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !15
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !15
  %710 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !15
  %711 = bitcast <4 x i32> %710 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %712 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %595, !dbg !17
  %713 = getelementptr i8, ptr addrspace(1) %712, i64 192, !dbg !17
  %714 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local228), !dbg !17
  %715 = zext i32 %714 to i64, !dbg !17
  %716 = inttoptr i64 %715 to ptr addrspace(3), !dbg !17
  %address.local236 = getelementptr [2 x i8], ptr addrspace(1) %713, i64 %address.local233, !dbg !17
  %717 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %718 = ptrtoint ptr addrspace(1) %address.local236 to i64, !dbg !17
  %719 = sub i64 %718, %717, !dbg !17
  %dma.buffer.offset594 = trunc i64 %719 to i32, !dbg !17
  %720 = trunc i64 %717 to i32, !dbg !17
  %721 = insertelement <4 x i32> poison, i32 %720, i64 0, !dbg !17
  %722 = lshr i64 %717, 32, !dbg !17
  %723 = trunc i64 %722 to i32, !dbg !17
  %724 = insertelement <4 x i32> %721, i32 %723, i64 1, !dbg !17
  %725 = insertelement <4 x i32> %724, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc595 = insertelement <4 x i32> %725, i32 131072, i64 3, !dbg !17
  %726 = ptrtoint ptr addrspace(3) %716 to i32, !dbg !17
  %dma.buffer.lds596 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %726), !dbg !17
  %727 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds596, i32 %dma.buffer.offset594, <4 x i32> %dma.buffer.rsrc595) #10, !dbg !17
  %728 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %729 = extractvalue [2 x <8 x float>] %33, 0, !dbg !34
  %730 = shufflevector <8 x float> %729, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %731 = fmul <4 x float> %730, %416, !dbg !34
  %732 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %728, <4 x float> %731) #9, !dbg !15
  %733 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %733, <4 x float> %732) #9, !dbg !15
  %735 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %735, <4 x float> %734) #9, !dbg !15
  %737 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %737, <4 x float> %736) #9, !dbg !15
  %739 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %740 = shufflevector <8 x float> %729, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %741 = fmul <4 x float> %740, %416, !dbg !34
  %742 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %739, <4 x float> %741) #9, !dbg !15
  %743 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %743, <4 x float> %742) #9, !dbg !15
  %745 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %745, <4 x float> %744) #9, !dbg !15
  %747 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %747, <4 x float> %746) #9, !dbg !15
  %749 = extractvalue [2 x <8 x float>] %33, 1, !dbg !34
  %750 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %751 = fmul <4 x float> %750, %424, !dbg !34
  %752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %728, <4 x float> %751) #9, !dbg !15
  %753 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %733, <4 x float> %752) #9, !dbg !15
  %754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %735, <4 x float> %753) #9, !dbg !15
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %737, <4 x float> %754) #9, !dbg !15
  %756 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %757 = fmul <4 x float> %756, %424, !dbg !34
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %739, <4 x float> %757) #9, !dbg !15
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %743, <4 x float> %758) #9, !dbg !15
  %760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %745, <4 x float> %759) #9, !dbg !15
  %761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %747, <4 x float> %760) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %762 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local211, i32 0) #0, !dbg !15
  %763 = bitcast <4 x i32> %762 to <8 x half>, !dbg !15
  %764 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local215, i32 0) #0, !dbg !15
  %765 = bitcast <4 x i32> %764 to <8 x half>, !dbg !15
  %766 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local219, i32 0) #0, !dbg !15
  %767 = bitcast <4 x i32> %766 to <8 x half>, !dbg !15
  %768 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local223, i32 0) #0, !dbg !15
  %769 = bitcast <4 x i32> %768 to <8 x half>, !dbg !15
  %770 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %771 = extractvalue [2 x <8 x float>] %34, 0, !dbg !34
  %772 = shufflevector <8 x float> %771, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %773 = fmul <4 x float> %772, %416, !dbg !34
  %774 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %770, <4 x float> %773) #9, !dbg !15
  %775 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %775, <4 x float> %774) #9, !dbg !15
  %777 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %777, <4 x float> %776) #9, !dbg !15
  %779 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %780 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %779, <4 x float> %778) #9, !dbg !15
  %781 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %782 = shufflevector <8 x float> %771, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %783 = fmul <4 x float> %782, %416, !dbg !34
  %784 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %781, <4 x float> %783) #9, !dbg !15
  %785 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %786 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed64, <4 x half> %785, <4 x float> %784) #9, !dbg !15
  %787 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %788 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed66, <4 x half> %787, <4 x float> %786) #9, !dbg !15
  %789 = shufflevector <8 x half> %769, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %790 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed68, <4 x half> %789, <4 x float> %788) #9, !dbg !15
  %791 = extractvalue [2 x <8 x float>] %34, 1, !dbg !34
  %792 = shufflevector <8 x float> %791, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %793 = fmul <4 x float> %792, %424, !dbg !34
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %770, <4 x float> %793) #9, !dbg !15
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %775, <4 x float> %794) #9, !dbg !15
  %796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %777, <4 x float> %795) #9, !dbg !15
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %779, <4 x float> %796) #9, !dbg !15
  %798 = shufflevector <8 x float> %791, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %799 = fmul <4 x float> %798, %424, !dbg !34
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed70, <4 x half> %781, <4 x float> %799) #9, !dbg !15
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed72, <4 x half> %785, <4 x float> %800) #9, !dbg !15
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed74, <4 x half> %787, <4 x float> %801) #9, !dbg !15
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed76, <4 x half> %789, <4 x float> %802) #9, !dbg !15
  %804 = shufflevector <4 x float> %622, <4 x float> %632, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %805 = insertvalue [2 x <8 x float>] poison, <8 x float> %804, 0, !dbg !8
  %806 = shufflevector <4 x float> %639, <4 x float> %645, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %807 = insertvalue [2 x <8 x float>] %805, <8 x float> %806, 1, !dbg !8
  %808 = insertvalue [2 x <1 x float>] poison, <1 x float> %374, 0, !dbg !8
  %809 = insertvalue [2 x <1 x float>] %808, <1 x float> %401, 1, !dbg !8
  %810 = shufflevector <4 x float> %680, <4 x float> %690, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %811 = insertvalue [2 x <8 x float>] poison, <8 x float> %810, 0, !dbg !8
  %812 = shufflevector <4 x float> %697, <4 x float> %703, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %813 = insertvalue [2 x <8 x float>] %811, <8 x float> %812, 1, !dbg !8
  %814 = shufflevector <4 x float> %738, <4 x float> %748, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %815 = insertvalue [2 x <8 x float>] poison, <8 x float> %814, 0, !dbg !8
  %816 = shufflevector <4 x float> %755, <4 x float> %761, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %817 = insertvalue [2 x <8 x float>] %815, <8 x float> %816, 1, !dbg !8
  %818 = shufflevector <4 x float> %780, <4 x float> %790, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %819 = insertvalue [2 x <8 x float>] poison, <8 x float> %818, 0, !dbg !8
  %820 = shufflevector <4 x float> %797, <4 x float> %803, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %821 = insertvalue [2 x <8 x float>] %819, <8 x float> %820, 1, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %822 = add nuw nsw i64 %28, 1, !dbg !8
  br label %27, !dbg !8

823:                                              ; preds = %27
  %824 = and i32 %19, 16, !dbg !8
  %.not = icmp eq i32 %824, 0, !dbg !8
  %825 = and i32 %19, 32, !dbg !8
  %.not2 = icmp eq i32 %825, 0, !dbg !8
  %826 = extractvalue [2 x <8 x float>] %34, 0, !dbg !8
  %827 = extractelement <8 x float> %826, i64 0, !dbg !8
  %828 = extractelement <8 x float> %826, i64 1, !dbg !8
  %829 = extractelement <8 x float> %826, i64 2, !dbg !8
  %830 = extractelement <8 x float> %826, i64 3, !dbg !8
  %831 = select i1 %.not, float %828, float %827, !dbg !8
  %832 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %833 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %832), !dbg !8
  %834 = add nuw nsw i32 %833, 64, !dbg !8
  %835 = xor i32 %833, 16, !dbg !8
  %836 = and i32 %834, 192, !dbg !8
  %837 = icmp samesign ult i32 %835, %836, !dbg !8
  %838 = select i1 %837, i32 %835, i32 %833, !dbg !8
  %839 = shl nuw nsw i32 %838, 2, !dbg !8
  %840 = bitcast float %831 to i32, !dbg !8
  %841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %840), !dbg !8
  %842 = bitcast i32 %841 to float, !dbg !8
  %843 = select i1 %.not, float %827, float %842, !dbg !8
  %844 = select i1 %.not, float %842, float %828, !dbg !8
  %845 = select i1 %.not, float %830, float %829, !dbg !8
  %846 = bitcast float %845 to i32, !dbg !8
  %847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %846), !dbg !8
  %848 = bitcast i32 %847 to float, !dbg !8
  %849 = select i1 %.not, float %829, float %848, !dbg !8
  %850 = select i1 %.not, float %848, float %830, !dbg !8
  %851 = select i1 %.not2, float %849, float %843, !dbg !8
  %852 = xor i32 %833, 32, !dbg !8
  %853 = icmp samesign ult i32 %852, %836, !dbg !8
  %854 = select i1 %853, i32 %852, i32 %833, !dbg !8
  %855 = shl nuw nsw i32 %854, 2, !dbg !8
  %856 = bitcast float %851 to i32, !dbg !8
  %857 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %856), !dbg !8
  %858 = bitcast i32 %857 to float, !dbg !8
  %859 = select i1 %.not2, float %843, float %858, !dbg !8
  %860 = select i1 %.not2, float %858, float %849, !dbg !8
  %861 = select i1 %.not2, float %850, float %844, !dbg !8
  %862 = bitcast float %861 to i32, !dbg !8
  %863 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %862), !dbg !8
  %864 = bitcast i32 %863 to float, !dbg !8
  %865 = select i1 %.not2, float %844, float %864, !dbg !8
  %866 = select i1 %.not2, float %864, float %850, !dbg !8
  %867 = extractvalue [2 x <8 x float>] %33, 0, !dbg !8
  %868 = extractelement <8 x float> %867, i64 0, !dbg !8
  %869 = extractelement <8 x float> %867, i64 1, !dbg !8
  %870 = extractelement <8 x float> %867, i64 2, !dbg !8
  %871 = extractelement <8 x float> %867, i64 3, !dbg !8
  %872 = select i1 %.not, float %869, float %868, !dbg !8
  %873 = bitcast float %872 to i32, !dbg !8
  %874 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %873), !dbg !8
  %875 = bitcast i32 %874 to float, !dbg !8
  %876 = select i1 %.not, float %868, float %875, !dbg !8
  %877 = select i1 %.not, float %875, float %869, !dbg !8
  %878 = select i1 %.not, float %871, float %870, !dbg !8
  %879 = bitcast float %878 to i32, !dbg !8
  %880 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %879), !dbg !8
  %881 = bitcast i32 %880 to float, !dbg !8
  %882 = select i1 %.not, float %870, float %881, !dbg !8
  %883 = select i1 %.not, float %881, float %871, !dbg !8
  %884 = select i1 %.not2, float %882, float %876, !dbg !8
  %885 = bitcast float %884 to i32, !dbg !8
  %886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %885), !dbg !8
  %887 = bitcast i32 %886 to float, !dbg !8
  %888 = select i1 %.not2, float %876, float %887, !dbg !8
  %889 = select i1 %.not2, float %887, float %882, !dbg !8
  %890 = select i1 %.not2, float %883, float %877, !dbg !8
  %891 = bitcast float %890 to i32, !dbg !8
  %892 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %891), !dbg !8
  %893 = bitcast i32 %892 to float, !dbg !8
  %894 = select i1 %.not2, float %877, float %893, !dbg !8
  %895 = select i1 %.not2, float %893, float %883, !dbg !8
  %896 = extractvalue [2 x <8 x float>] %32, 0, !dbg !8
  %897 = extractelement <8 x float> %896, i64 0, !dbg !8
  %898 = extractelement <8 x float> %896, i64 1, !dbg !8
  %899 = extractelement <8 x float> %896, i64 2, !dbg !8
  %900 = extractelement <8 x float> %896, i64 3, !dbg !8
  %901 = select i1 %.not, float %898, float %897, !dbg !8
  %902 = bitcast float %901 to i32, !dbg !8
  %903 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %902), !dbg !8
  %904 = bitcast i32 %903 to float, !dbg !8
  %905 = select i1 %.not, float %897, float %904, !dbg !8
  %906 = select i1 %.not, float %904, float %898, !dbg !8
  %907 = select i1 %.not, float %900, float %899, !dbg !8
  %908 = bitcast float %907 to i32, !dbg !8
  %909 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %908), !dbg !8
  %910 = bitcast i32 %909 to float, !dbg !8
  %911 = select i1 %.not, float %899, float %910, !dbg !8
  %912 = select i1 %.not, float %910, float %900, !dbg !8
  %913 = select i1 %.not2, float %911, float %905, !dbg !8
  %914 = bitcast float %913 to i32, !dbg !8
  %915 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %914), !dbg !8
  %916 = bitcast i32 %915 to float, !dbg !8
  %917 = select i1 %.not2, float %905, float %916, !dbg !8
  %918 = select i1 %.not2, float %916, float %911, !dbg !8
  %919 = select i1 %.not2, float %912, float %906, !dbg !8
  %920 = bitcast float %919 to i32, !dbg !8
  %921 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %920), !dbg !8
  %922 = bitcast i32 %921 to float, !dbg !8
  %923 = select i1 %.not2, float %906, float %922, !dbg !8
  %924 = select i1 %.not2, float %922, float %912, !dbg !8
  %925 = extractvalue [2 x <8 x float>] %29, 0, !dbg !8
  %926 = extractelement <8 x float> %925, i64 0, !dbg !8
  %927 = extractelement <8 x float> %925, i64 1, !dbg !8
  %928 = extractelement <8 x float> %925, i64 2, !dbg !8
  %929 = extractelement <8 x float> %925, i64 3, !dbg !8
  %930 = select i1 %.not, float %927, float %926, !dbg !8
  %931 = bitcast float %930 to i32, !dbg !8
  %932 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %931), !dbg !8
  %933 = bitcast i32 %932 to float, !dbg !8
  %934 = select i1 %.not, float %926, float %933, !dbg !8
  %935 = select i1 %.not, float %933, float %927, !dbg !8
  %936 = select i1 %.not, float %929, float %928, !dbg !8
  %937 = bitcast float %936 to i32, !dbg !8
  %938 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %937), !dbg !8
  %939 = bitcast i32 %938 to float, !dbg !8
  %940 = select i1 %.not, float %928, float %939, !dbg !8
  %941 = select i1 %.not, float %939, float %929, !dbg !8
  %942 = select i1 %.not2, float %940, float %934, !dbg !8
  %943 = bitcast float %942 to i32, !dbg !8
  %944 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %943), !dbg !8
  %945 = select i1 %.not2, float %941, float %935, !dbg !8
  %946 = bitcast float %945 to i32, !dbg !8
  %947 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %946), !dbg !8
  %948 = extractvalue [2 x <1 x float>] %30, 0, !dbg !18
  %949 = shufflevector <1 x float> %948, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %950 = extractelement <8 x float> %826, i64 4, !dbg !8
  %951 = extractelement <8 x float> %826, i64 5, !dbg !8
  %952 = extractelement <8 x float> %826, i64 6, !dbg !8
  %953 = extractelement <8 x float> %826, i64 7, !dbg !8
  %954 = select i1 %.not, float %951, float %950, !dbg !8
  %955 = bitcast float %954 to i32, !dbg !8
  %956 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %955), !dbg !8
  %957 = bitcast i32 %956 to float, !dbg !8
  %958 = select i1 %.not, float %950, float %957, !dbg !8
  %959 = select i1 %.not, float %957, float %951, !dbg !8
  %960 = select i1 %.not, float %953, float %952, !dbg !8
  %961 = bitcast float %960 to i32, !dbg !8
  %962 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %961), !dbg !8
  %963 = bitcast i32 %962 to float, !dbg !8
  %964 = select i1 %.not, float %952, float %963, !dbg !8
  %965 = select i1 %.not, float %963, float %953, !dbg !8
  %966 = select i1 %.not2, float %964, float %958, !dbg !8
  %967 = bitcast float %966 to i32, !dbg !8
  %968 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %967), !dbg !8
  %969 = bitcast i32 %968 to float, !dbg !8
  %970 = select i1 %.not2, float %958, float %969, !dbg !8
  %971 = select i1 %.not2, float %969, float %964, !dbg !8
  %972 = select i1 %.not2, float %965, float %959, !dbg !8
  %973 = bitcast float %972 to i32, !dbg !8
  %974 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %973), !dbg !8
  %975 = bitcast i32 %974 to float, !dbg !8
  %976 = select i1 %.not2, float %959, float %975, !dbg !8
  %977 = select i1 %.not2, float %975, float %965, !dbg !8
  %978 = extractelement <8 x float> %867, i64 4, !dbg !8
  %979 = extractelement <8 x float> %867, i64 5, !dbg !8
  %980 = extractelement <8 x float> %867, i64 6, !dbg !8
  %981 = extractelement <8 x float> %867, i64 7, !dbg !8
  %982 = select i1 %.not, float %979, float %978, !dbg !8
  %983 = bitcast float %982 to i32, !dbg !8
  %984 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %983), !dbg !8
  %985 = bitcast i32 %984 to float, !dbg !8
  %986 = select i1 %.not, float %978, float %985, !dbg !8
  %987 = select i1 %.not, float %985, float %979, !dbg !8
  %988 = select i1 %.not, float %981, float %980, !dbg !8
  %989 = bitcast float %988 to i32, !dbg !8
  %990 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %989), !dbg !8
  %991 = bitcast i32 %990 to float, !dbg !8
  %992 = select i1 %.not, float %980, float %991, !dbg !8
  %993 = select i1 %.not, float %991, float %981, !dbg !8
  %994 = select i1 %.not2, float %992, float %986, !dbg !8
  %995 = bitcast float %994 to i32, !dbg !8
  %996 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %995), !dbg !8
  %997 = bitcast i32 %996 to float, !dbg !8
  %998 = select i1 %.not2, float %986, float %997, !dbg !8
  %999 = select i1 %.not2, float %997, float %992, !dbg !8
  %1000 = select i1 %.not2, float %993, float %987, !dbg !8
  %1001 = bitcast float %1000 to i32, !dbg !8
  %1002 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1001), !dbg !8
  %1003 = bitcast i32 %1002 to float, !dbg !8
  %1004 = select i1 %.not2, float %987, float %1003, !dbg !8
  %1005 = select i1 %.not2, float %1003, float %993, !dbg !8
  %1006 = extractelement <8 x float> %896, i64 4, !dbg !8
  %1007 = extractelement <8 x float> %896, i64 5, !dbg !8
  %1008 = extractelement <8 x float> %896, i64 6, !dbg !8
  %1009 = extractelement <8 x float> %896, i64 7, !dbg !8
  %1010 = select i1 %.not, float %1007, float %1006, !dbg !8
  %1011 = bitcast float %1010 to i32, !dbg !8
  %1012 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1011), !dbg !8
  %1013 = bitcast i32 %1012 to float, !dbg !8
  %1014 = select i1 %.not, float %1006, float %1013, !dbg !8
  %1015 = select i1 %.not, float %1013, float %1007, !dbg !8
  %1016 = select i1 %.not, float %1009, float %1008, !dbg !8
  %1017 = bitcast float %1016 to i32, !dbg !8
  %1018 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1017), !dbg !8
  %1019 = bitcast i32 %1018 to float, !dbg !8
  %1020 = select i1 %.not, float %1008, float %1019, !dbg !8
  %1021 = select i1 %.not, float %1019, float %1009, !dbg !8
  %1022 = select i1 %.not2, float %1020, float %1014, !dbg !8
  %1023 = bitcast float %1022 to i32, !dbg !8
  %1024 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1023), !dbg !8
  %1025 = bitcast i32 %1024 to float, !dbg !8
  %1026 = select i1 %.not2, float %1014, float %1025, !dbg !8
  %1027 = select i1 %.not2, float %1025, float %1020, !dbg !8
  %1028 = select i1 %.not2, float %1021, float %1015, !dbg !8
  %1029 = bitcast float %1028 to i32, !dbg !8
  %1030 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1029), !dbg !8
  %1031 = bitcast i32 %1030 to float, !dbg !8
  %1032 = select i1 %.not2, float %1015, float %1031, !dbg !8
  %1033 = select i1 %.not2, float %1031, float %1021, !dbg !8
  %1034 = extractelement <8 x float> %925, i64 4, !dbg !8
  %1035 = extractelement <8 x float> %925, i64 5, !dbg !8
  %1036 = extractelement <8 x float> %925, i64 6, !dbg !8
  %1037 = extractelement <8 x float> %925, i64 7, !dbg !8
  %1038 = select i1 %.not, float %1035, float %1034, !dbg !8
  %1039 = bitcast float %1038 to i32, !dbg !8
  %1040 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1039), !dbg !8
  %1041 = bitcast i32 %1040 to float, !dbg !8
  %1042 = select i1 %.not, float %1034, float %1041, !dbg !8
  %1043 = select i1 %.not, float %1041, float %1035, !dbg !8
  %1044 = select i1 %.not, float %1037, float %1036, !dbg !8
  %1045 = bitcast float %1044 to i32, !dbg !8
  %1046 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1045), !dbg !8
  %1047 = bitcast i32 %1046 to float, !dbg !8
  %1048 = select i1 %.not, float %1036, float %1047, !dbg !8
  %1049 = select i1 %.not, float %1047, float %1037, !dbg !8
  %1050 = select i1 %.not2, float %1048, float %1042, !dbg !8
  %1051 = bitcast float %1050 to i32, !dbg !8
  %1052 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1051), !dbg !8
  %1053 = select i1 %.not2, float %1049, float %1043, !dbg !8
  %1054 = bitcast float %1053 to i32, !dbg !8
  %1055 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1054), !dbg !8
  %1056 = extractvalue [2 x <8 x float>] %34, 1, !dbg !8
  %1057 = extractelement <8 x float> %1056, i64 0, !dbg !8
  %1058 = extractelement <8 x float> %1056, i64 1, !dbg !8
  %1059 = extractelement <8 x float> %1056, i64 2, !dbg !8
  %1060 = extractelement <8 x float> %1056, i64 3, !dbg !8
  %1061 = select i1 %.not, float %1058, float %1057, !dbg !8
  %1062 = bitcast float %1061 to i32, !dbg !8
  %1063 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1062), !dbg !8
  %1064 = bitcast i32 %1063 to float, !dbg !8
  %1065 = select i1 %.not, float %1057, float %1064, !dbg !8
  %1066 = select i1 %.not, float %1064, float %1058, !dbg !8
  %1067 = select i1 %.not, float %1060, float %1059, !dbg !8
  %1068 = bitcast float %1067 to i32, !dbg !8
  %1069 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1068), !dbg !8
  %1070 = bitcast i32 %1069 to float, !dbg !8
  %1071 = select i1 %.not, float %1059, float %1070, !dbg !8
  %1072 = select i1 %.not, float %1070, float %1060, !dbg !8
  %1073 = select i1 %.not2, float %1071, float %1065, !dbg !8
  %1074 = bitcast float %1073 to i32, !dbg !8
  %1075 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1074), !dbg !8
  %1076 = bitcast i32 %1075 to float, !dbg !8
  %1077 = select i1 %.not2, float %1065, float %1076, !dbg !8
  %1078 = select i1 %.not2, float %1076, float %1071, !dbg !8
  %1079 = select i1 %.not2, float %1072, float %1066, !dbg !8
  %1080 = bitcast float %1079 to i32, !dbg !8
  %1081 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1080), !dbg !8
  %1082 = bitcast i32 %1081 to float, !dbg !8
  %1083 = select i1 %.not2, float %1066, float %1082, !dbg !8
  %1084 = select i1 %.not2, float %1082, float %1072, !dbg !8
  %1085 = extractvalue [2 x <8 x float>] %33, 1, !dbg !8
  %1086 = extractelement <8 x float> %1085, i64 0, !dbg !8
  %1087 = extractelement <8 x float> %1085, i64 1, !dbg !8
  %1088 = extractelement <8 x float> %1085, i64 2, !dbg !8
  %1089 = extractelement <8 x float> %1085, i64 3, !dbg !8
  %1090 = select i1 %.not, float %1087, float %1086, !dbg !8
  %1091 = bitcast float %1090 to i32, !dbg !8
  %1092 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1091), !dbg !8
  %1093 = bitcast i32 %1092 to float, !dbg !8
  %1094 = select i1 %.not, float %1086, float %1093, !dbg !8
  %1095 = select i1 %.not, float %1093, float %1087, !dbg !8
  %1096 = select i1 %.not, float %1089, float %1088, !dbg !8
  %1097 = bitcast float %1096 to i32, !dbg !8
  %1098 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1097), !dbg !8
  %1099 = bitcast i32 %1098 to float, !dbg !8
  %1100 = select i1 %.not, float %1088, float %1099, !dbg !8
  %1101 = select i1 %.not, float %1099, float %1089, !dbg !8
  %1102 = select i1 %.not2, float %1100, float %1094, !dbg !8
  %1103 = bitcast float %1102 to i32, !dbg !8
  %1104 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1103), !dbg !8
  %1105 = bitcast i32 %1104 to float, !dbg !8
  %1106 = select i1 %.not2, float %1094, float %1105, !dbg !8
  %1107 = select i1 %.not2, float %1105, float %1100, !dbg !8
  %1108 = select i1 %.not2, float %1101, float %1095, !dbg !8
  %1109 = bitcast float %1108 to i32, !dbg !8
  %1110 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1109), !dbg !8
  %1111 = bitcast i32 %1110 to float, !dbg !8
  %1112 = select i1 %.not2, float %1095, float %1111, !dbg !8
  %1113 = select i1 %.not2, float %1111, float %1101, !dbg !8
  %1114 = extractvalue [2 x <8 x float>] %32, 1, !dbg !8
  %1115 = extractelement <8 x float> %1114, i64 0, !dbg !8
  %1116 = extractelement <8 x float> %1114, i64 1, !dbg !8
  %1117 = extractelement <8 x float> %1114, i64 2, !dbg !8
  %1118 = extractelement <8 x float> %1114, i64 3, !dbg !8
  %1119 = select i1 %.not, float %1116, float %1115, !dbg !8
  %1120 = bitcast float %1119 to i32, !dbg !8
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1120), !dbg !8
  %1122 = bitcast i32 %1121 to float, !dbg !8
  %1123 = select i1 %.not, float %1115, float %1122, !dbg !8
  %1124 = select i1 %.not, float %1122, float %1116, !dbg !8
  %1125 = select i1 %.not, float %1118, float %1117, !dbg !8
  %1126 = bitcast float %1125 to i32, !dbg !8
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1126), !dbg !8
  %1128 = bitcast i32 %1127 to float, !dbg !8
  %1129 = select i1 %.not, float %1117, float %1128, !dbg !8
  %1130 = select i1 %.not, float %1128, float %1118, !dbg !8
  %1131 = select i1 %.not2, float %1129, float %1123, !dbg !8
  %1132 = bitcast float %1131 to i32, !dbg !8
  %1133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1132), !dbg !8
  %1134 = bitcast i32 %1133 to float, !dbg !8
  %1135 = select i1 %.not2, float %1123, float %1134, !dbg !8
  %1136 = select i1 %.not2, float %1134, float %1129, !dbg !8
  %1137 = select i1 %.not2, float %1130, float %1124, !dbg !8
  %1138 = bitcast float %1137 to i32, !dbg !8
  %1139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1138), !dbg !8
  %1140 = bitcast i32 %1139 to float, !dbg !8
  %1141 = select i1 %.not2, float %1124, float %1140, !dbg !8
  %1142 = select i1 %.not2, float %1140, float %1130, !dbg !8
  %1143 = extractvalue [2 x <8 x float>] %29, 1, !dbg !8
  %1144 = extractelement <8 x float> %1143, i64 0, !dbg !8
  %1145 = extractelement <8 x float> %1143, i64 1, !dbg !8
  %1146 = extractelement <8 x float> %1143, i64 2, !dbg !8
  %1147 = extractelement <8 x float> %1143, i64 3, !dbg !8
  %1148 = select i1 %.not, float %1145, float %1144, !dbg !8
  %1149 = bitcast float %1148 to i32, !dbg !8
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1149), !dbg !8
  %1151 = bitcast i32 %1150 to float, !dbg !8
  %1152 = select i1 %.not, float %1144, float %1151, !dbg !8
  %1153 = select i1 %.not, float %1151, float %1145, !dbg !8
  %1154 = select i1 %.not, float %1147, float %1146, !dbg !8
  %1155 = bitcast float %1154 to i32, !dbg !8
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1155), !dbg !8
  %1157 = bitcast i32 %1156 to float, !dbg !8
  %1158 = select i1 %.not, float %1146, float %1157, !dbg !8
  %1159 = select i1 %.not, float %1157, float %1147, !dbg !8
  %1160 = select i1 %.not2, float %1158, float %1152, !dbg !8
  %1161 = bitcast float %1160 to i32, !dbg !8
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1161), !dbg !8
  %1163 = select i1 %.not2, float %1159, float %1153, !dbg !8
  %1164 = bitcast float %1163 to i32, !dbg !8
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1164), !dbg !8
  %1166 = extractvalue [2 x <1 x float>] %30, 1, !dbg !18
  %1167 = shufflevector <1 x float> %1166, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1168 = extractelement <8 x float> %1056, i64 4, !dbg !8
  %1169 = extractelement <8 x float> %1056, i64 5, !dbg !8
  %1170 = extractelement <8 x float> %1056, i64 6, !dbg !8
  %1171 = extractelement <8 x float> %1056, i64 7, !dbg !8
  %1172 = select i1 %.not, float %1169, float %1168, !dbg !8
  %1173 = bitcast float %1172 to i32, !dbg !8
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1173), !dbg !8
  %1175 = bitcast i32 %1174 to float, !dbg !8
  %1176 = select i1 %.not, float %1168, float %1175, !dbg !8
  %1177 = select i1 %.not, float %1175, float %1169, !dbg !8
  %1178 = select i1 %.not, float %1171, float %1170, !dbg !8
  %1179 = bitcast float %1178 to i32, !dbg !8
  %1180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1179), !dbg !8
  %1181 = bitcast i32 %1180 to float, !dbg !8
  %1182 = select i1 %.not, float %1170, float %1181, !dbg !8
  %1183 = select i1 %.not, float %1181, float %1171, !dbg !8
  %1184 = select i1 %.not2, float %1182, float %1176, !dbg !8
  %1185 = bitcast float %1184 to i32, !dbg !8
  %1186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1185), !dbg !8
  %1187 = bitcast i32 %1186 to float, !dbg !8
  %1188 = select i1 %.not2, float %1176, float %1187, !dbg !8
  %1189 = select i1 %.not2, float %1187, float %1182, !dbg !8
  %1190 = select i1 %.not2, float %1183, float %1177, !dbg !8
  %1191 = bitcast float %1190 to i32, !dbg !8
  %1192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1191), !dbg !8
  %1193 = bitcast i32 %1192 to float, !dbg !8
  %1194 = select i1 %.not2, float %1177, float %1193, !dbg !8
  %1195 = select i1 %.not2, float %1193, float %1183, !dbg !8
  %1196 = extractelement <8 x float> %1085, i64 4, !dbg !8
  %1197 = extractelement <8 x float> %1085, i64 5, !dbg !8
  %1198 = extractelement <8 x float> %1085, i64 6, !dbg !8
  %1199 = extractelement <8 x float> %1085, i64 7, !dbg !8
  %1200 = select i1 %.not, float %1197, float %1196, !dbg !8
  %1201 = bitcast float %1200 to i32, !dbg !8
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1201), !dbg !8
  %1203 = bitcast i32 %1202 to float, !dbg !8
  %1204 = select i1 %.not, float %1196, float %1203, !dbg !8
  %1205 = select i1 %.not, float %1203, float %1197, !dbg !8
  %1206 = select i1 %.not, float %1199, float %1198, !dbg !8
  %1207 = bitcast float %1206 to i32, !dbg !8
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1207), !dbg !8
  %1209 = bitcast i32 %1208 to float, !dbg !8
  %1210 = select i1 %.not, float %1198, float %1209, !dbg !8
  %1211 = select i1 %.not, float %1209, float %1199, !dbg !8
  %1212 = select i1 %.not2, float %1210, float %1204, !dbg !8
  %1213 = bitcast float %1212 to i32, !dbg !8
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1213), !dbg !8
  %1215 = bitcast i32 %1214 to float, !dbg !8
  %1216 = select i1 %.not2, float %1204, float %1215, !dbg !8
  %1217 = select i1 %.not2, float %1215, float %1210, !dbg !8
  %1218 = select i1 %.not2, float %1211, float %1205, !dbg !8
  %1219 = bitcast float %1218 to i32, !dbg !8
  %1220 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1219), !dbg !8
  %1221 = bitcast i32 %1220 to float, !dbg !8
  %1222 = select i1 %.not2, float %1205, float %1221, !dbg !8
  %1223 = select i1 %.not2, float %1221, float %1211, !dbg !8
  %1224 = extractelement <8 x float> %1114, i64 4, !dbg !8
  %1225 = extractelement <8 x float> %1114, i64 5, !dbg !8
  %1226 = extractelement <8 x float> %1114, i64 6, !dbg !8
  %1227 = extractelement <8 x float> %1114, i64 7, !dbg !8
  %1228 = select i1 %.not, float %1225, float %1224, !dbg !8
  %1229 = bitcast float %1228 to i32, !dbg !8
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1229), !dbg !8
  %1231 = bitcast i32 %1230 to float, !dbg !8
  %1232 = select i1 %.not, float %1224, float %1231, !dbg !8
  %1233 = select i1 %.not, float %1231, float %1225, !dbg !8
  %1234 = select i1 %.not, float %1227, float %1226, !dbg !8
  %1235 = bitcast float %1234 to i32, !dbg !8
  %1236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1235), !dbg !8
  %1237 = bitcast i32 %1236 to float, !dbg !8
  %1238 = select i1 %.not, float %1226, float %1237, !dbg !8
  %1239 = select i1 %.not, float %1237, float %1227, !dbg !8
  %1240 = select i1 %.not2, float %1238, float %1232, !dbg !8
  %1241 = bitcast float %1240 to i32, !dbg !8
  %1242 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1241), !dbg !8
  %1243 = bitcast i32 %1242 to float, !dbg !8
  %1244 = select i1 %.not2, float %1232, float %1243, !dbg !8
  %1245 = select i1 %.not2, float %1243, float %1238, !dbg !8
  %1246 = select i1 %.not2, float %1239, float %1233, !dbg !8
  %1247 = bitcast float %1246 to i32, !dbg !8
  %1248 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1247), !dbg !8
  %1249 = bitcast i32 %1248 to float, !dbg !8
  %1250 = select i1 %.not2, float %1233, float %1249, !dbg !8
  %1251 = select i1 %.not2, float %1249, float %1239, !dbg !8
  %1252 = extractelement <8 x float> %1143, i64 4, !dbg !8
  %1253 = extractelement <8 x float> %1143, i64 5, !dbg !8
  %1254 = extractelement <8 x float> %1143, i64 6, !dbg !8
  %1255 = extractelement <8 x float> %1143, i64 7, !dbg !8
  %1256 = select i1 %.not, float %1253, float %1252, !dbg !8
  %1257 = bitcast float %1256 to i32, !dbg !8
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1257), !dbg !8
  %1259 = bitcast i32 %1258 to float, !dbg !8
  %1260 = select i1 %.not, float %1252, float %1259, !dbg !8
  %1261 = select i1 %.not, float %1259, float %1253, !dbg !8
  %1262 = select i1 %.not, float %1255, float %1254, !dbg !8
  %1263 = bitcast float %1262 to i32, !dbg !8
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %839, i32 %1263), !dbg !8
  %1265 = bitcast i32 %1264 to float, !dbg !8
  %1266 = select i1 %.not, float %1254, float %1265, !dbg !8
  %1267 = select i1 %.not, float %1265, float %1255, !dbg !8
  %1268 = select i1 %.not2, float %1266, float %1260, !dbg !8
  %1269 = bitcast float %1268 to i32, !dbg !8
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1269), !dbg !8
  %1271 = select i1 %.not2, float %1267, float %1261, !dbg !8
  %1272 = bitcast float %1271 to i32, !dbg !8
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %855, i32 %1272), !dbg !8
  br i1 true, label %1274, label %1338, !dbg !20

1274:                                             ; preds = %823
  %1275 = bitcast i32 %1270 to float, !dbg !8
  %1276 = select i1 %.not2, float %1260, float %1275, !dbg !8
  %1277 = insertelement <4 x float> poison, float %1276, i64 0, !dbg !18
  %1278 = bitcast i32 %1273 to float, !dbg !8
  %1279 = select i1 %.not2, float %1261, float %1278, !dbg !8
  %1280 = insertelement <4 x float> %1277, float %1279, i64 1, !dbg !18
  %1281 = select i1 %.not2, float %1275, float %1266, !dbg !8
  %1282 = insertelement <4 x float> %1280, float %1281, i64 2, !dbg !18
  %1283 = select i1 %.not2, float %1278, float %1267, !dbg !8
  %1284 = insertelement <4 x float> %1282, float %1283, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1167, !dbg !18
  %normalized = fmul <4 x float> %1284, %row.reciprocal, !dbg !18
  %1285 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1286 = extractelement <4 x half> %1285, i64 3, !dbg !19
  %1287 = extractelement <4 x half> %1285, i64 2, !dbg !19
  %1288 = extractelement <4 x half> %1285, i64 1, !dbg !19
  %1289 = extractelement <4 x half> %1285, i64 0, !dbg !19
  %1290 = bitcast i32 %1162 to float, !dbg !8
  %1291 = select i1 %.not2, float %1152, float %1290, !dbg !8
  %1292 = insertelement <4 x float> poison, float %1291, i64 0, !dbg !18
  %1293 = bitcast i32 %1165 to float, !dbg !8
  %1294 = select i1 %.not2, float %1153, float %1293, !dbg !8
  %1295 = insertelement <4 x float> %1292, float %1294, i64 1, !dbg !18
  %1296 = select i1 %.not2, float %1290, float %1158, !dbg !8
  %1297 = insertelement <4 x float> %1295, float %1296, i64 2, !dbg !18
  %1298 = select i1 %.not2, float %1293, float %1159, !dbg !8
  %1299 = insertelement <4 x float> %1297, float %1298, i64 3, !dbg !18
  %normalized39 = fmul <4 x float> %1299, %row.reciprocal, !dbg !18
  %1300 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !19
  %1301 = extractelement <4 x half> %1300, i64 3, !dbg !19
  %1302 = extractelement <4 x half> %1300, i64 2, !dbg !19
  %1303 = extractelement <4 x half> %1300, i64 1, !dbg !19
  %1304 = extractelement <4 x half> %1300, i64 0, !dbg !19
  %1305 = bitcast i32 %1052 to float, !dbg !8
  %1306 = select i1 %.not2, float %1042, float %1305, !dbg !8
  %1307 = insertelement <4 x float> poison, float %1306, i64 0, !dbg !18
  %1308 = bitcast i32 %1055 to float, !dbg !8
  %1309 = select i1 %.not2, float %1043, float %1308, !dbg !8
  %1310 = insertelement <4 x float> %1307, float %1309, i64 1, !dbg !18
  %1311 = select i1 %.not2, float %1305, float %1048, !dbg !8
  %1312 = insertelement <4 x float> %1310, float %1311, i64 2, !dbg !18
  %1313 = select i1 %.not2, float %1308, float %1049, !dbg !8
  %1314 = insertelement <4 x float> %1312, float %1313, i64 3, !dbg !18
  %row.reciprocal40 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %949, !dbg !18
  %normalized41 = fmul <4 x float> %1314, %row.reciprocal40, !dbg !18
  %1315 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !19
  %1316 = extractelement <4 x half> %1315, i64 3, !dbg !19
  %1317 = extractelement <4 x half> %1315, i64 2, !dbg !19
  %1318 = extractelement <4 x half> %1315, i64 1, !dbg !19
  %1319 = extractelement <4 x half> %1315, i64 0, !dbg !19
  %1320 = bitcast i32 %944 to float, !dbg !8
  %1321 = select i1 %.not2, float %934, float %1320, !dbg !8
  %1322 = insertelement <4 x float> poison, float %1321, i64 0, !dbg !18
  %1323 = bitcast i32 %947 to float, !dbg !8
  %1324 = select i1 %.not2, float %935, float %1323, !dbg !8
  %1325 = insertelement <4 x float> %1322, float %1324, i64 1, !dbg !18
  %1326 = select i1 %.not2, float %1320, float %940, !dbg !8
  %1327 = insertelement <4 x float> %1325, float %1326, i64 2, !dbg !18
  %1328 = select i1 %.not2, float %1323, float %941, !dbg !8
  %1329 = insertelement <4 x float> %1327, float %1328, i64 3, !dbg !18
  %normalized42 = fmul <4 x float> %1329, %row.reciprocal40, !dbg !18
  %1330 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !19
  %1331 = extractelement <4 x half> %1330, i64 3, !dbg !19
  %1332 = extractelement <4 x half> %1330, i64 2, !dbg !19
  %1333 = extractelement <4 x half> %1330, i64 1, !dbg !19
  %1334 = extractelement <4 x half> %1330, i64 0, !dbg !19
  %1335 = add nsw i64 %6, %13, !dbg !20
  %1336 = shl nsw i64 %1335, 19, !dbg !20
  %1337 = and i64 %1336, 16252928, !dbg !20
  %address.thread237 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local238 = zext nneg i32 %address.thread237 to i64, !dbg !13
  %address.local239 = add nsw i64 %16, %address.local238, !dbg !20
  %address.local240 = lshr i64 %address.local238, 6, !dbg !13
  %address.local241 = shl nuw nsw i64 %address.local240, 4, !dbg !20
  %address.local242 = add nsw i64 %address.local239, %address.local241, !dbg !20
  %address.local243 = and i64 %address.local238, 240, !dbg !20
  %address.local244 = sub nsw i64 %address.local242, %address.local243, !dbg !20
  %address.local245 = add nsw i64 %address.local244, %17, !dbg !20
  %address.local246 = shl nuw nsw i64 %address.local245, 7, !dbg !20
  %address.local247 = add nuw nsw i64 %1337, %address.local246, !dbg !20
  %address.local248 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local249 = lshr i64 %address.local238, 2, !dbg !13
  %address.local250 = and i64 %address.local249, 12, !dbg !13
  %address.local251 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local248, i64 %address.local250, !dbg !20
  store half %1334, ptr addrspace(1) %address.local251, align 2, !dbg !20
  %address.local252 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local253 = or disjoint i64 %address.local250, 1, !dbg !20
  %address.local254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local252, i64 %address.local253, !dbg !20
  store half %1333, ptr addrspace(1) %address.local254, align 2, !dbg !20
  %address.local255 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local256 = or disjoint i64 %address.local250, 2, !dbg !20
  %address.local257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local255, i64 %address.local256, !dbg !20
  store half %1332, ptr addrspace(1) %address.local257, align 2, !dbg !20
  %address.local258 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local259 = or disjoint i64 %address.local250, 3, !dbg !20
  %address.local260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local258, i64 %address.local259, !dbg !20
  store half %1331, ptr addrspace(1) %address.local260, align 2, !dbg !20
  %address.local261 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local262 = or disjoint i64 %address.local250, 16, !dbg !20
  %address.local263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local261, i64 %address.local262, !dbg !20
  store half %1319, ptr addrspace(1) %address.local263, align 2, !dbg !20
  %address.local264 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local265 = or disjoint i64 %address.local250, 17, !dbg !20
  %address.local266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local264, i64 %address.local265, !dbg !20
  store half %1318, ptr addrspace(1) %address.local266, align 2, !dbg !20
  %address.local267 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local268 = or disjoint i64 %address.local250, 18, !dbg !20
  %address.local269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local267, i64 %address.local268, !dbg !20
  store half %1317, ptr addrspace(1) %address.local269, align 2, !dbg !20
  %address.local270 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local247, !dbg !20
  %address.local271 = or disjoint i64 %address.local250, 19, !dbg !20
  %address.local272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local270, i64 %address.local271, !dbg !20
  store half %1316, ptr addrspace(1) %address.local272, align 2, !dbg !20
  %address.local273 = shl nsw i64 %address.local245, 7, !dbg !20
  %address.local274 = add i64 %address.local273, 8192, !dbg !20
  %address.local275 = add nuw nsw i64 %1337, %address.local274, !dbg !20
  %address.local276 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local277 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local276, i64 %address.local250, !dbg !20
  store half %1304, ptr addrspace(1) %address.local277, align 2, !dbg !20
  %address.local278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local278, i64 %address.local253, !dbg !20
  store half %1303, ptr addrspace(1) %address.local279, align 2, !dbg !20
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local280, i64 %address.local256, !dbg !20
  store half %1302, ptr addrspace(1) %address.local281, align 2, !dbg !20
  %address.local282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local282, i64 %address.local259, !dbg !20
  store half %1301, ptr addrspace(1) %address.local283, align 2, !dbg !20
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local284, i64 %address.local262, !dbg !20
  store half %1289, ptr addrspace(1) %address.local285, align 2, !dbg !20
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local286, i64 %address.local265, !dbg !20
  store half %1288, ptr addrspace(1) %address.local287, align 2, !dbg !20
  %address.local288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local288, i64 %address.local268, !dbg !20
  store half %1287, ptr addrspace(1) %address.local289, align 2, !dbg !20
  %address.local290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local275, !dbg !20
  %address.local291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local290, i64 %address.local271, !dbg !20
  store half %1286, ptr addrspace(1) %address.local291, align 2, !dbg !20
  br label %1338, !dbg !20

1338:                                             ; preds = %1274, %823
  br i1 true, label %1339, label %1377, !dbg !20

1339:                                             ; preds = %1338
  %1340 = insertelement <4 x float> poison, float %1244, i64 0, !dbg !18
  %1341 = insertelement <4 x float> %1340, float %1250, i64 1, !dbg !18
  %1342 = insertelement <4 x float> %1341, float %1245, i64 2, !dbg !18
  %1343 = insertelement <4 x float> %1342, float %1251, i64 3, !dbg !18
  %row.reciprocal43 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1167, !dbg !18
  %normalized44 = fmul <4 x float> %1343, %row.reciprocal43, !dbg !18
  %1344 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !19
  %1345 = extractelement <4 x half> %1344, i64 3, !dbg !19
  %1346 = extractelement <4 x half> %1344, i64 2, !dbg !19
  %1347 = extractelement <4 x half> %1344, i64 1, !dbg !19
  %1348 = extractelement <4 x half> %1344, i64 0, !dbg !19
  %1349 = insertelement <4 x float> poison, float %1135, i64 0, !dbg !18
  %1350 = insertelement <4 x float> %1349, float %1141, i64 1, !dbg !18
  %1351 = insertelement <4 x float> %1350, float %1136, i64 2, !dbg !18
  %1352 = insertelement <4 x float> %1351, float %1142, i64 3, !dbg !18
  %normalized45 = fmul <4 x float> %1352, %row.reciprocal43, !dbg !18
  %1353 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !19
  %1354 = extractelement <4 x half> %1353, i64 3, !dbg !19
  %1355 = extractelement <4 x half> %1353, i64 2, !dbg !19
  %1356 = extractelement <4 x half> %1353, i64 1, !dbg !19
  %1357 = extractelement <4 x half> %1353, i64 0, !dbg !19
  %1358 = insertelement <4 x float> poison, float %1026, i64 0, !dbg !18
  %1359 = insertelement <4 x float> %1358, float %1032, i64 1, !dbg !18
  %1360 = insertelement <4 x float> %1359, float %1027, i64 2, !dbg !18
  %1361 = insertelement <4 x float> %1360, float %1033, i64 3, !dbg !18
  %row.reciprocal46 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %949, !dbg !18
  %normalized47 = fmul <4 x float> %1361, %row.reciprocal46, !dbg !18
  %1362 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !19
  %1363 = extractelement <4 x half> %1362, i64 3, !dbg !19
  %1364 = extractelement <4 x half> %1362, i64 2, !dbg !19
  %1365 = extractelement <4 x half> %1362, i64 1, !dbg !19
  %1366 = extractelement <4 x half> %1362, i64 0, !dbg !19
  %1367 = insertelement <4 x float> poison, float %917, i64 0, !dbg !18
  %1368 = insertelement <4 x float> %1367, float %923, i64 1, !dbg !18
  %1369 = insertelement <4 x float> %1368, float %918, i64 2, !dbg !18
  %1370 = insertelement <4 x float> %1369, float %924, i64 3, !dbg !18
  %normalized48 = fmul <4 x float> %1370, %row.reciprocal46, !dbg !18
  %1371 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !19
  %1372 = extractelement <4 x half> %1371, i64 3, !dbg !19
  %1373 = extractelement <4 x half> %1371, i64 2, !dbg !19
  %1374 = extractelement <4 x half> %1371, i64 1, !dbg !19
  %1375 = extractelement <4 x half> %1371, i64 0, !dbg !19
  %1376 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread292 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local293 = zext nneg i32 %address.thread292 to i64, !dbg !13
  %address.local294 = add nsw i64 %16, %address.local293, !dbg !20
  %address.local295 = lshr i64 %address.local293, 6, !dbg !13
  %address.local296 = shl nuw nsw i64 %address.local295, 4, !dbg !20
  %address.local297 = add nsw i64 %address.local294, %address.local296, !dbg !20
  %address.local298 = and i64 %address.local293, 240, !dbg !20
  %address.local299 = sub nsw i64 %address.local297, %address.local298, !dbg !20
  %address.local300 = add nsw i64 %address.local299, %17, !dbg !20
  %address.local301 = shl nuw nsw i64 %address.local300, 7, !dbg !20
  %address.local302 = add nuw nsw i64 %1376, %address.local301, !dbg !20
  %address.local303 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local304 = lshr i64 %address.local293, 2, !dbg !13
  %address.local305 = and i64 %address.local304, 12, !dbg !13
  %address.local306 = or disjoint i64 %address.local305, 32, !dbg !20
  %address.local307 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local303, i64 %address.local306, !dbg !20
  store half %1375, ptr addrspace(1) %address.local307, align 2, !dbg !20
  %address.local308 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local309 = or disjoint i64 %address.local305, 33, !dbg !20
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local308, i64 %address.local309, !dbg !20
  store half %1374, ptr addrspace(1) %address.local310, align 2, !dbg !20
  %address.local311 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local312 = or disjoint i64 %address.local305, 34, !dbg !20
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local311, i64 %address.local312, !dbg !20
  store half %1373, ptr addrspace(1) %address.local313, align 2, !dbg !20
  %address.local314 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local315 = or disjoint i64 %address.local305, 35, !dbg !20
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local314, i64 %address.local315, !dbg !20
  store half %1372, ptr addrspace(1) %address.local316, align 2, !dbg !20
  %address.local317 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local318 = or disjoint i64 %address.local305, 48, !dbg !20
  %address.local319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local317, i64 %address.local318, !dbg !20
  store half %1366, ptr addrspace(1) %address.local319, align 2, !dbg !20
  %address.local320 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local321 = or disjoint i64 %address.local305, 49, !dbg !20
  %address.local322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local320, i64 %address.local321, !dbg !20
  store half %1365, ptr addrspace(1) %address.local322, align 2, !dbg !20
  %address.local323 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local324 = or disjoint i64 %address.local305, 50, !dbg !20
  %address.local325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local323, i64 %address.local324, !dbg !20
  store half %1364, ptr addrspace(1) %address.local325, align 2, !dbg !20
  %address.local326 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local302, !dbg !20
  %address.local327 = or i64 %address.local304, 51, !dbg !20
  %address.local328 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local326, i64 %address.local327, !dbg !20
  store half %1363, ptr addrspace(1) %address.local328, align 2, !dbg !20
  %address.local329 = shl nsw i64 %address.local300, 7, !dbg !20
  %address.local330 = add i64 %address.local329, 8192, !dbg !20
  %address.local331 = add nuw nsw i64 %1376, %address.local330, !dbg !20
  %address.local332 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local333 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local332, i64 %address.local306, !dbg !20
  store half %1357, ptr addrspace(1) %address.local333, align 2, !dbg !20
  %address.local334 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local334, i64 %address.local309, !dbg !20
  store half %1356, ptr addrspace(1) %address.local335, align 2, !dbg !20
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local336, i64 %address.local312, !dbg !20
  store half %1355, ptr addrspace(1) %address.local337, align 2, !dbg !20
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local315, !dbg !20
  store half %1354, ptr addrspace(1) %address.local339, align 2, !dbg !20
  %address.local340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local340, i64 %address.local318, !dbg !20
  store half %1348, ptr addrspace(1) %address.local341, align 2, !dbg !20
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local342, i64 %address.local321, !dbg !20
  store half %1347, ptr addrspace(1) %address.local343, align 2, !dbg !20
  %address.local344 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local344, i64 %address.local324, !dbg !20
  store half %1346, ptr addrspace(1) %address.local345, align 2, !dbg !20
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local331, !dbg !20
  %address.local347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local346, i64 %address.local327, !dbg !20
  store half %1345, ptr addrspace(1) %address.local347, align 2, !dbg !20
  br label %1377, !dbg !20

1377:                                             ; preds = %1339, %1338
  br i1 true, label %1378, label %1416, !dbg !20

1378:                                             ; preds = %1377
  %1379 = insertelement <4 x float> poison, float %1216, i64 0, !dbg !18
  %1380 = insertelement <4 x float> %1379, float %1222, i64 1, !dbg !18
  %1381 = insertelement <4 x float> %1380, float %1217, i64 2, !dbg !18
  %1382 = insertelement <4 x float> %1381, float %1223, i64 3, !dbg !18
  %row.reciprocal49 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1167, !dbg !18
  %normalized50 = fmul <4 x float> %1382, %row.reciprocal49, !dbg !18
  %1383 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !19
  %1384 = extractelement <4 x half> %1383, i64 3, !dbg !19
  %1385 = extractelement <4 x half> %1383, i64 2, !dbg !19
  %1386 = extractelement <4 x half> %1383, i64 1, !dbg !19
  %1387 = extractelement <4 x half> %1383, i64 0, !dbg !19
  %1388 = insertelement <4 x float> poison, float %1106, i64 0, !dbg !18
  %1389 = insertelement <4 x float> %1388, float %1112, i64 1, !dbg !18
  %1390 = insertelement <4 x float> %1389, float %1107, i64 2, !dbg !18
  %1391 = insertelement <4 x float> %1390, float %1113, i64 3, !dbg !18
  %normalized51 = fmul <4 x float> %1391, %row.reciprocal49, !dbg !18
  %1392 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !19
  %1393 = extractelement <4 x half> %1392, i64 3, !dbg !19
  %1394 = extractelement <4 x half> %1392, i64 2, !dbg !19
  %1395 = extractelement <4 x half> %1392, i64 1, !dbg !19
  %1396 = extractelement <4 x half> %1392, i64 0, !dbg !19
  %1397 = insertelement <4 x float> poison, float %998, i64 0, !dbg !18
  %1398 = insertelement <4 x float> %1397, float %1004, i64 1, !dbg !18
  %1399 = insertelement <4 x float> %1398, float %999, i64 2, !dbg !18
  %1400 = insertelement <4 x float> %1399, float %1005, i64 3, !dbg !18
  %row.reciprocal52 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %949, !dbg !18
  %normalized53 = fmul <4 x float> %1400, %row.reciprocal52, !dbg !18
  %1401 = fptrunc <4 x float> %normalized53 to <4 x half>, !dbg !19
  %1402 = extractelement <4 x half> %1401, i64 3, !dbg !19
  %1403 = extractelement <4 x half> %1401, i64 2, !dbg !19
  %1404 = extractelement <4 x half> %1401, i64 1, !dbg !19
  %1405 = extractelement <4 x half> %1401, i64 0, !dbg !19
  %1406 = insertelement <4 x float> poison, float %888, i64 0, !dbg !18
  %1407 = insertelement <4 x float> %1406, float %894, i64 1, !dbg !18
  %1408 = insertelement <4 x float> %1407, float %889, i64 2, !dbg !18
  %1409 = insertelement <4 x float> %1408, float %895, i64 3, !dbg !18
  %normalized54 = fmul <4 x float> %1409, %row.reciprocal52, !dbg !18
  %1410 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !19
  %1411 = extractelement <4 x half> %1410, i64 3, !dbg !19
  %1412 = extractelement <4 x half> %1410, i64 2, !dbg !19
  %1413 = extractelement <4 x half> %1410, i64 1, !dbg !19
  %1414 = extractelement <4 x half> %1410, i64 0, !dbg !19
  %1415 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread348 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local349 = zext nneg i32 %address.thread348 to i64, !dbg !13
  %address.local350 = add nsw i64 %16, %address.local349, !dbg !20
  %address.local351 = lshr i64 %address.local349, 6, !dbg !13
  %address.local352 = shl nuw nsw i64 %address.local351, 4, !dbg !20
  %address.local353 = add nsw i64 %address.local350, %address.local352, !dbg !20
  %address.local354 = and i64 %address.local349, 240, !dbg !20
  %address.local355 = sub nsw i64 %address.local353, %address.local354, !dbg !20
  %address.local356 = add nsw i64 %address.local355, %17, !dbg !20
  %address.local357 = shl nuw nsw i64 %address.local356, 7, !dbg !20
  %address.local358 = add nuw nsw i64 %1415, %address.local357, !dbg !20
  %address.local359 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local360 = lshr i64 %address.local349, 2, !dbg !13
  %address.local361 = and i64 %address.local360, 12, !dbg !13
  %address.local362 = or disjoint i64 %address.local361, 64, !dbg !20
  %address.local363 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local359, i64 %address.local362, !dbg !20
  store half %1414, ptr addrspace(1) %address.local363, align 2, !dbg !20
  %address.local364 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local365 = or disjoint i64 %address.local361, 65, !dbg !20
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local364, i64 %address.local365, !dbg !20
  store half %1413, ptr addrspace(1) %address.local366, align 2, !dbg !20
  %address.local367 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local368 = or disjoint i64 %address.local361, 66, !dbg !20
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local367, i64 %address.local368, !dbg !20
  store half %1412, ptr addrspace(1) %address.local369, align 2, !dbg !20
  %address.local370 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local371 = or disjoint i64 %address.local361, 67, !dbg !20
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local370, i64 %address.local371, !dbg !20
  store half %1411, ptr addrspace(1) %address.local372, align 2, !dbg !20
  %address.local373 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local374 = or disjoint i64 %address.local361, 80, !dbg !20
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local373, i64 %address.local374, !dbg !20
  store half %1405, ptr addrspace(1) %address.local375, align 2, !dbg !20
  %address.local376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local377 = or disjoint i64 %address.local361, 81, !dbg !20
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local376, i64 %address.local377, !dbg !20
  store half %1404, ptr addrspace(1) %address.local378, align 2, !dbg !20
  %address.local379 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local380 = or disjoint i64 %address.local361, 82, !dbg !20
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local379, i64 %address.local380, !dbg !20
  store half %1403, ptr addrspace(1) %address.local381, align 2, !dbg !20
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local358, !dbg !20
  %address.local383 = or disjoint i64 %address.local361, 83, !dbg !20
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local383, !dbg !20
  store half %1402, ptr addrspace(1) %address.local384, align 2, !dbg !20
  %address.local385 = shl nsw i64 %address.local356, 7, !dbg !20
  %address.local386 = add i64 %address.local385, 8192, !dbg !20
  %address.local387 = add nuw nsw i64 %1415, %address.local386, !dbg !20
  %address.local388 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local388, i64 %address.local362, !dbg !20
  store half %1396, ptr addrspace(1) %address.local389, align 2, !dbg !20
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local390, i64 %address.local365, !dbg !20
  store half %1395, ptr addrspace(1) %address.local391, align 2, !dbg !20
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local368, !dbg !20
  store half %1394, ptr addrspace(1) %address.local393, align 2, !dbg !20
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local371, !dbg !20
  store half %1393, ptr addrspace(1) %address.local395, align 2, !dbg !20
  %address.local396 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local396, i64 %address.local374, !dbg !20
  store half %1387, ptr addrspace(1) %address.local397, align 2, !dbg !20
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local377, !dbg !20
  store half %1386, ptr addrspace(1) %address.local399, align 2, !dbg !20
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local380, !dbg !20
  store half %1385, ptr addrspace(1) %address.local401, align 2, !dbg !20
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local387, !dbg !20
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local402, i64 %address.local383, !dbg !20
  store half %1384, ptr addrspace(1) %address.local403, align 2, !dbg !20
  br label %1416, !dbg !20

1416:                                             ; preds = %1378, %1377
  br i1 true, label %1417, label %1455, !dbg !20

1417:                                             ; preds = %1416
  %1418 = insertelement <4 x float> poison, float %1188, i64 0, !dbg !18
  %1419 = insertelement <4 x float> %1418, float %1194, i64 1, !dbg !18
  %1420 = insertelement <4 x float> %1419, float %1189, i64 2, !dbg !18
  %1421 = insertelement <4 x float> %1420, float %1195, i64 3, !dbg !18
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1167, !dbg !18
  %normalized56 = fmul <4 x float> %1421, %row.reciprocal55, !dbg !18
  %1422 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !19
  %1423 = extractelement <4 x half> %1422, i64 3, !dbg !19
  %1424 = extractelement <4 x half> %1422, i64 2, !dbg !19
  %1425 = extractelement <4 x half> %1422, i64 1, !dbg !19
  %1426 = extractelement <4 x half> %1422, i64 0, !dbg !19
  %1427 = insertelement <4 x float> poison, float %1077, i64 0, !dbg !18
  %1428 = insertelement <4 x float> %1427, float %1083, i64 1, !dbg !18
  %1429 = insertelement <4 x float> %1428, float %1078, i64 2, !dbg !18
  %1430 = insertelement <4 x float> %1429, float %1084, i64 3, !dbg !18
  %normalized57 = fmul <4 x float> %1430, %row.reciprocal55, !dbg !18
  %1431 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !19
  %1432 = extractelement <4 x half> %1431, i64 3, !dbg !19
  %1433 = extractelement <4 x half> %1431, i64 2, !dbg !19
  %1434 = extractelement <4 x half> %1431, i64 1, !dbg !19
  %1435 = extractelement <4 x half> %1431, i64 0, !dbg !19
  %1436 = insertelement <4 x float> poison, float %970, i64 0, !dbg !18
  %1437 = insertelement <4 x float> %1436, float %976, i64 1, !dbg !18
  %1438 = insertelement <4 x float> %1437, float %971, i64 2, !dbg !18
  %1439 = insertelement <4 x float> %1438, float %977, i64 3, !dbg !18
  %row.reciprocal58 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %949, !dbg !18
  %normalized59 = fmul <4 x float> %1439, %row.reciprocal58, !dbg !18
  %1440 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !19
  %1441 = extractelement <4 x half> %1440, i64 3, !dbg !19
  %1442 = extractelement <4 x half> %1440, i64 2, !dbg !19
  %1443 = extractelement <4 x half> %1440, i64 1, !dbg !19
  %1444 = extractelement <4 x half> %1440, i64 0, !dbg !19
  %1445 = insertelement <4 x float> poison, float %859, i64 0, !dbg !18
  %1446 = insertelement <4 x float> %1445, float %865, i64 1, !dbg !18
  %1447 = insertelement <4 x float> %1446, float %860, i64 2, !dbg !18
  %1448 = insertelement <4 x float> %1447, float %866, i64 3, !dbg !18
  %normalized60 = fmul <4 x float> %1448, %row.reciprocal58, !dbg !18
  %1449 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !19
  %1450 = extractelement <4 x half> %1449, i64 3, !dbg !19
  %1451 = extractelement <4 x half> %1449, i64 2, !dbg !19
  %1452 = extractelement <4 x half> %1449, i64 1, !dbg !19
  %1453 = extractelement <4 x half> %1449, i64 0, !dbg !19
  %1454 = shl nuw nsw i64 %15, 19, !dbg !20
  %address.thread404 = call i32 asm sideeffect "", "=v,0"(i32 %19) #9, !dbg !20, !range !39, !frisk.register_lifetimes !1
  %address.local405 = zext nneg i32 %address.thread404 to i64, !dbg !13
  %address.local406 = add nsw i64 %16, %address.local405, !dbg !20
  %address.local407 = lshr i64 %address.local405, 6, !dbg !13
  %address.local408 = shl nuw nsw i64 %address.local407, 4, !dbg !20
  %address.local409 = add nsw i64 %address.local406, %address.local408, !dbg !20
  %address.local410 = and i64 %address.local405, 240, !dbg !20
  %address.local411 = sub nsw i64 %address.local409, %address.local410, !dbg !20
  %address.local412 = add nsw i64 %address.local411, %17, !dbg !20
  %address.local413 = shl nuw nsw i64 %address.local412, 7, !dbg !20
  %address.local414 = add nuw nsw i64 %1454, %address.local413, !dbg !20
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local416 = lshr i64 %address.local405, 2, !dbg !13
  %address.local417 = and i64 %address.local416, 12, !dbg !13
  %address.local418 = or disjoint i64 %address.local417, 96, !dbg !20
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local415, i64 %address.local418, !dbg !20
  store half %1453, ptr addrspace(1) %address.local419, align 2, !dbg !20
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local421 = or disjoint i64 %address.local417, 97, !dbg !20
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local420, i64 %address.local421, !dbg !20
  store half %1452, ptr addrspace(1) %address.local422, align 2, !dbg !20
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local424 = or disjoint i64 %address.local417, 98, !dbg !20
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local423, i64 %address.local424, !dbg !20
  store half %1451, ptr addrspace(1) %address.local425, align 2, !dbg !20
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local427 = or disjoint i64 %address.local417, 99, !dbg !20
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local426, i64 %address.local427, !dbg !20
  store half %1450, ptr addrspace(1) %address.local428, align 2, !dbg !20
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local430 = or disjoint i64 %address.local417, 112, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local430, !dbg !20
  store half %1444, ptr addrspace(1) %address.local431, align 2, !dbg !20
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local433 = or disjoint i64 %address.local417, 113, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local433, !dbg !20
  store half %1443, ptr addrspace(1) %address.local434, align 2, !dbg !20
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local436 = or disjoint i64 %address.local417, 114, !dbg !20
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local436, !dbg !20
  store half %1442, ptr addrspace(1) %address.local437, align 2, !dbg !20
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local414, !dbg !20
  %address.local439 = or i64 %address.local416, 115, !dbg !20
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local439, !dbg !20
  store half %1441, ptr addrspace(1) %address.local440, align 2, !dbg !20
  %address.local441 = shl nsw i64 %address.local412, 7, !dbg !20
  %address.local442 = add i64 %address.local441, 8192, !dbg !20
  %address.local443 = add nuw nsw i64 %1454, %address.local442, !dbg !20
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local444, i64 %address.local418, !dbg !20
  store half %1435, ptr addrspace(1) %address.local445, align 2, !dbg !20
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local446, i64 %address.local421, !dbg !20
  store half %1434, ptr addrspace(1) %address.local447, align 2, !dbg !20
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local424, !dbg !20
  store half %1433, ptr addrspace(1) %address.local449, align 2, !dbg !20
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local450, i64 %address.local427, !dbg !20
  store half %1432, ptr addrspace(1) %address.local451, align 2, !dbg !20
  %address.local452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local452, i64 %address.local430, !dbg !20
  store half %1426, ptr addrspace(1) %address.local453, align 2, !dbg !20
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local454, i64 %address.local433, !dbg !20
  store half %1425, ptr addrspace(1) %address.local455, align 2, !dbg !20
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local456, i64 %address.local436, !dbg !20
  store half %1424, ptr addrspace(1) %address.local457, align 2, !dbg !20
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local443, !dbg !20
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local458, i64 %address.local439, !dbg !20
  store half %1423, ptr addrspace(1) %address.local459, align 2, !dbg !20
  br label %1455, !dbg !20

1455:                                             ; preds = %1417, %1416
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/nohint-rolled")
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
