; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/group8-reverse/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/group8-reverse/input.mlir"

@shm_0 = addrspace(3) global [12288 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !36 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 5, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423480, !dbg !5
  %13 = and i64 %6, 7, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = lshr i64 %10, 3, !dbg !5
  %16 = sub nsw i64 31, %14, !dbg !5
  %17 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = shl i64 %16, 7, !dbg !9
  %20 = icmp ugt i64 %10, 1023, !dbg !11
  %21 = add nsw i64 %14, -32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %16, !dbg !11
  %23 = sdiv i64 %22, 32, !dbg !11
  %24 = sext i1 %20 to i64, !dbg !11
  %25 = xor i64 %23, %24, !dbg !11
  %26 = add nsw i64 %15, %25, !dbg !11
  %27 = and i64 %26, 31, !dbg !11
  %28 = shl nsw i64 %16, 7, !dbg !11
  %29 = mul nsw i64 %25, -4096, !dbg !11
  %30 = add i64 %28, %29, !dbg !11
  %31 = shl nuw nsw i64 %27, 19, !dbg !11
  %32 = shl nsw i64 %30, 7, !dbg !11
  %33 = add i64 %31, %32, !dbg !11
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !38
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = or disjoint i64 %19, 127, !dbg !13
  %38 = add i64 %19, 128, !dbg !13
  %39 = icmp uge i64 %38, %37, !dbg !13
  %40 = shl nsw i64 %16, 1, !dbg !7
  %41 = add nsw i64 %40, 2, !dbg !7
  %42 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %43 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %44 = getelementptr i8, ptr addrspace(1) %43, i64 64, !dbg !11
  %45 = shl i64 %15, 7, !dbg !15
  %46 = and i64 %45, 3968, !dbg !15
  %47 = shl nsw i64 %18, 12, !dbg !15
  %48 = or disjoint i64 %46, %47, !dbg !15
  %49 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %50 = getelementptr i8, ptr addrspace(1) %49, i64 128, !dbg !11
  %51 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %52 = getelementptr i8, ptr addrspace(1) %51, i64 192, !dbg !11
  br label %53, !dbg !7

53:                                               ; preds = %579, %4
  %54 = phi i64 [ %1116, %579 ], [ 0, %4 ], !dbg !7
  %55 = phi [2 x <8 x float>] [ %1101, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %56 = phi [2 x <1 x float>] [ %877, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %57 = phi [2 x <1 x float>] [ %1103, %579 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %58 = phi [2 x <8 x float>] [ %1107, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %59 = phi [2 x <8 x float>] [ %1111, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi [2 x <8 x float>] [ %1115, %579 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = icmp slt i64 %54, %41, !dbg !7
  br i1 %61, label %62, label %1117, !dbg !7

62:                                               ; preds = %53
  %63 = shl nsw i64 %54, 6, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local74 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local75 = and i64 %address.local74, 1536, !dbg !11
  %address.local76 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local75, !dbg !11
  %address.local77 = ptrtoint ptr addrspace(3) %address.local76 to i64, !dbg !11
  %address.local78 = trunc i64 %address.local77 to i32, !dbg !11
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local78), !dbg !11
  %65 = zext i32 %64 to i64, !dbg !11
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !11
  %address.local79 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local80 = and i64 %address.local79, 16256, !dbg !11
  %address.local81 = lshr i64 %address.local, 4, !dbg !11
  %address.local82 = and i64 %address.local81, 8, !dbg !11
  %address.local83 = or disjoint i64 %address.local80, %address.local82, !dbg !11
  %address.local84 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local83, !dbg !11
  %67 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %68 = ptrtoint ptr addrspace(1) %address.local84 to i64, !dbg !11
  %69 = sub i64 %68, %67, !dbg !11
  %dma.buffer.offset632 = trunc i64 %69 to i32, !dbg !11
  %70 = trunc i64 %67 to i32, !dbg !11
  %71 = insertelement <4 x i32> poison, i32 %70, i64 0, !dbg !11
  %72 = lshr i64 %67, 32, !dbg !11
  %73 = trunc i64 %72 to i32, !dbg !11
  %74 = insertelement <4 x i32> %71, i32 %73, i64 1, !dbg !11
  %75 = insertelement <4 x i32> %74, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc633 = insertelement <4 x i32> %75, i32 131072, i64 3, !dbg !11
  %76 = ptrtoint ptr addrspace(3) %66 to i32, !dbg !11
  %dma.buffer.lds634 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %76), !dbg !11
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds634, i32 %dma.buffer.offset632, <4 x i32> %dma.buffer.rsrc633) #10, !dbg !11
  %address.local85 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local86 = shl nuw nsw i64 %address.local85, 3, !dbg !11
  %address.local87 = and i64 %address.local86, 3584, !dbg !11
  %address.local88 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local87, !dbg !11
  %address.local89 = ptrtoint ptr addrspace(3) %address.local88 to i64, !dbg !11
  %address.local90 = trunc i64 %address.local89 to i32, !dbg !11
  %78 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local90), !dbg !11
  %79 = zext i32 %78 to i64, !dbg !11
  %80 = inttoptr i64 %79 to ptr addrspace(3), !dbg !11
  %address.local91 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local92 = and i64 %address.local91, 16256, !dbg !11
  %address.local93 = lshr i64 %address.local85, 4, !dbg !11
  %address.local94 = and i64 %address.local93, 24, !dbg !11
  %address.local95 = or disjoint i64 %address.local92, %address.local94, !dbg !11
  %address.local96 = getelementptr [2 x i8], ptr addrspace(1) %42, i64 %address.local95, !dbg !11
  %81 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %82 = ptrtoint ptr addrspace(1) %address.local96 to i64, !dbg !11
  %83 = sub i64 %82, %81, !dbg !11
  %dma.buffer.offset629 = trunc i64 %83 to i32, !dbg !11
  %84 = trunc i64 %81 to i32, !dbg !11
  %85 = insertelement <4 x i32> poison, i32 %84, i64 0, !dbg !11
  %86 = lshr i64 %81, 32, !dbg !11
  %87 = trunc i64 %86 to i32, !dbg !11
  %88 = insertelement <4 x i32> %85, i32 %87, i64 1, !dbg !11
  %89 = insertelement <4 x i32> %88, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc630 = insertelement <4 x i32> %89, i32 131072, i64 3, !dbg !11
  %90 = ptrtoint ptr addrspace(3) %80 to i32, !dbg !11
  %dma.buffer.lds631 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %90), !dbg !11
  %91 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds631, i32 %dma.buffer.offset629, <4 x i32> %dma.buffer.rsrc630) #10, !dbg !11
  %92 = lshr i64 %54, 13, !dbg !15
  %93 = add nuw nsw i64 %92, %15, !dbg !15
  %94 = lshr i64 %54, 6, !dbg !15
  %95 = and i64 %94, 127, !dbg !15
  %96 = shl i64 %93, 19, !dbg !15
  %97 = and i64 %96, 16252928, !dbg !15
  %98 = shl i64 %94, 19, !dbg !15
  %99 = shl i64 %54, 13, !dbg !15
  %100 = sub i64 %99, %98, !dbg !15
  %101 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %97, !dbg !15
  %102 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %95, !dbg !15
  %103 = getelementptr [2 x i8], ptr addrspace(1) %102, i64 %100, !dbg !15
  %address.local97 = and i64 %address.local74, 1536, !dbg !15
  %address.local98 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local97, !dbg !15
  %address.local99 = ptrtoint ptr addrspace(3) %address.local98 to i64, !dbg !15
  %address.local100 = trunc i64 %address.local99 to i32, !dbg !15
  %104 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local100), !dbg !15
  %105 = zext i32 %104 to i64, !dbg !15
  %106 = inttoptr i64 %105 to ptr addrspace(3), !dbg !15
  %address.local101 = lshr i64 %address.local, 3, !dbg !15
  %address.local102 = and i64 %address.local101, 24, !dbg !15
  %address.local103 = shl nuw nsw i64 %address.local, 7, !dbg !15
  %address.local104 = and i64 %address.local103, 8064, !dbg !15
  %address.local105 = or disjoint i64 %address.local102, %address.local104, !dbg !15
  %address.local106 = getelementptr [2 x i8], ptr addrspace(1) %103, i64 %address.local105, !dbg !15
  %107 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %108 = ptrtoint ptr addrspace(1) %address.local106 to i64, !dbg !15
  %109 = sub i64 %108, %107, !dbg !15
  %dma.buffer.offset626 = trunc i64 %109 to i32, !dbg !15
  %110 = trunc i64 %107 to i32, !dbg !15
  %111 = insertelement <4 x i32> poison, i32 %110, i64 0, !dbg !15
  %112 = lshr i64 %107, 32, !dbg !15
  %113 = trunc i64 %112 to i32, !dbg !15
  %114 = insertelement <4 x i32> %111, i32 %113, i64 1, !dbg !15
  %115 = insertelement <4 x i32> %114, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc627 = insertelement <4 x i32> %115, i32 131072, i64 3, !dbg !15
  %116 = ptrtoint ptr addrspace(3) %106 to i32, !dbg !15
  %dma.buffer.lds628 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %116), !dbg !15
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds628, i32 %dma.buffer.offset626, <4 x i32> %dma.buffer.rsrc627) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local107 = lshr i32 %address.thread, 2, !dbg !12
  %address.local108 = and i32 %address.local107, 4, !dbg !12
  %address.local109 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local110 = and i32 %address.local109, 384, !dbg !12
  %address.local111 = or disjoint i32 %address.local108, %address.local110, !dbg !12
  %address.local112 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local113 = and i32 %address.local112, 120, !dbg !12
  %address.local114 = or disjoint i32 %address.local111, %address.local113, !dbg !12
  %address.local115 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local116 = and i32 %address.local115, 1024, !dbg !12
  %address.local117 = or disjoint i32 %address.local114, %address.local116, !dbg !12
  %address.local118 = zext nneg i32 %address.local117 to i64, !dbg !12
  %address.local119 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local118, !dbg !12
  %118 = load <4 x half>, ptr addrspace(3) %address.local119, align 2, !dbg !12
  %address.local120 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local118, !dbg !12
  %address.local121 = getelementptr i8, ptr addrspace(3) %address.local120, i64 4096, !dbg !12
  %119 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local118, !dbg !12
  %address.local123 = getelementptr i8, ptr addrspace(3) %address.local122, i64 1024, !dbg !12
  %120 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %121 = fmul <4 x half> %118, splat (half 0xH3015), !dbg !12
  %address.local124 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local118, !dbg !12
  %address.local125 = getelementptr i8, ptr addrspace(3) %address.local124, i64 5120, !dbg !12
  %122 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %123 = fmul <4 x half> %119, splat (half 0xH3015), !dbg !12
  %124 = fmul <4 x half> %120, splat (half 0xH3015), !dbg !12
  %125 = fmul <4 x half> %122, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local126 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local127 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local128 = lshr i64 %address.local, 2, !dbg !16
  %address.local129 = mul nsw i64 %address.local128, -512, !dbg !16
  %address.local130 = getelementptr [2 x i8], ptr addrspace(3) %address.local127, i64 %address.local129, !dbg !16
  %address.local131 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local132 = and i64 %address.local131, 24, !dbg !16
  %address.local133 = getelementptr [2 x i8], ptr addrspace(3) %address.local130, i64 %address.local132, !dbg !16
  %address.local134 = lshr i64 %address.local, 2, !dbg !12
  %address.local135 = and i64 %address.local134, 12, !dbg !12
  %address.local136 = getelementptr [2 x i8], ptr addrspace(3) %address.local133, i64 %address.local135, !dbg !16
  %address.local137 = and i64 %address.local, 63, !dbg !12
  %address.local138 = lshr i64 %address.local137, 5, !dbg !12
  %address.local139 = mul nuw nsw i64 %address.local138, 504, !dbg !16
  %address.local140 = getelementptr [2 x i8], ptr addrspace(3) %address.local136, i64 %address.local139, !dbg !16
  %126 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %126, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local141 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local142 = getelementptr [2 x i8], ptr addrspace(3) %address.local141, i64 %address.local135, !dbg !16
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) %address.local142, i64 %address.local139, !dbg !16
  %address.local144 = getelementptr [2 x i8], ptr addrspace(3) %address.local143, i64 %address.local129, !dbg !16
  %address.local145 = getelementptr [2 x i8], ptr addrspace(3) %address.local144, i64 %address.local132, !dbg !16
  %address.local146 = getelementptr i8, ptr addrspace(3) %address.local145, i64 2048, !dbg !16
  %128 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local148 = getelementptr [2 x i8], ptr addrspace(3) %address.local147, i64 %address.local129, !dbg !16
  %address.local149 = getelementptr [2 x i8], ptr addrspace(3) %address.local148, i64 %address.local132, !dbg !16
  %address.local150 = getelementptr [2 x i8], ptr addrspace(3) %address.local149, i64 %address.local135, !dbg !16
  %address.local151 = getelementptr [2 x i8], ptr addrspace(3) %address.local150, i64 %address.local139, !dbg !16
  %address.local152 = getelementptr i8, ptr addrspace(3) %address.local151, i64 64, !dbg !16
  %129 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %128, <4 x float> %127) #9, !dbg !16
  %131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %129, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local154 = getelementptr [2 x i8], ptr addrspace(3) %address.local153, i64 %address.local135, !dbg !16
  %address.local155 = getelementptr [2 x i8], ptr addrspace(3) %address.local154, i64 %address.local139, !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) %address.local155, i64 %address.local129, !dbg !16
  %address.local157 = getelementptr [2 x i8], ptr addrspace(3) %address.local156, i64 %address.local132, !dbg !16
  %address.local158 = getelementptr i8, ptr addrspace(3) %address.local157, i64 2112, !dbg !16
  %132 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local160 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local129, !dbg !16
  %address.local161 = getelementptr [2 x i8], ptr addrspace(3) %address.local160, i64 %address.local132, !dbg !16
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) %address.local161, i64 %address.local135, !dbg !16
  %address.local163 = getelementptr [2 x i8], ptr addrspace(3) %address.local162, i64 %address.local139, !dbg !16
  %address.local164 = getelementptr i8, ptr addrspace(3) %address.local163, i64 128, !dbg !16
  %133 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %132, <4 x float> %131) #9, !dbg !16
  %135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local166 = getelementptr [2 x i8], ptr addrspace(3) %address.local165, i64 %address.local135, !dbg !16
  %address.local167 = getelementptr [2 x i8], ptr addrspace(3) %address.local166, i64 %address.local139, !dbg !16
  %address.local168 = getelementptr [2 x i8], ptr addrspace(3) %address.local167, i64 %address.local129, !dbg !16
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) %address.local168, i64 %address.local132, !dbg !16
  %address.local170 = getelementptr i8, ptr addrspace(3) %address.local169, i64 2176, !dbg !16
  %136 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) %address.local171, i64 %address.local129, !dbg !16
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) %address.local172, i64 %address.local132, !dbg !16
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) %address.local173, i64 %address.local135, !dbg !16
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) %address.local174, i64 %address.local139, !dbg !16
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 192, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %136, <4 x float> %135) #9, !dbg !16
  %139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %121, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local126, !dbg !16
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) %address.local177, i64 %address.local135, !dbg !16
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) %address.local178, i64 %address.local139, !dbg !16
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) %address.local179, i64 %address.local129, !dbg !16
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) %address.local180, i64 %address.local132, !dbg !16
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 2240, !dbg !16
  %140 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %141 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %140, <4 x float> %139) #9, !dbg !16
  %143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %141, <4 x float> zeroinitializer) #9, !dbg !16
  %144 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %145 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %144, <4 x float> %143) #9, !dbg !16
  %147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %145, <4 x float> zeroinitializer) #9, !dbg !16
  %148 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %149 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %148, <4 x float> %147) #9, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %152 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %153 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %152, <4 x float> %151) #9, !dbg !16
  %155 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %124, <4 x half> %153, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %125, <4 x half> %155, <4 x float> %156) #9, !dbg !16
  %158 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local78), !dbg !11
  %159 = zext i32 %158 to i64, !dbg !11
  %160 = inttoptr i64 %159 to ptr addrspace(3), !dbg !11
  %address.local183 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local83, !dbg !11
  %161 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %162 = ptrtoint ptr addrspace(1) %address.local183 to i64, !dbg !11
  %163 = sub i64 %162, %161, !dbg !11
  %dma.buffer.offset623 = trunc i64 %163 to i32, !dbg !11
  %164 = trunc i64 %161 to i32, !dbg !11
  %165 = insertelement <4 x i32> poison, i32 %164, i64 0, !dbg !11
  %166 = lshr i64 %161, 32, !dbg !11
  %167 = trunc i64 %166 to i32, !dbg !11
  %168 = insertelement <4 x i32> %165, i32 %167, i64 1, !dbg !11
  %169 = insertelement <4 x i32> %168, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc624 = insertelement <4 x i32> %169, i32 131072, i64 3, !dbg !11
  %170 = ptrtoint ptr addrspace(3) %160 to i32, !dbg !11
  %dma.buffer.lds625 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %170), !dbg !11
  %171 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds625, i32 %dma.buffer.offset623, <4 x i32> %dma.buffer.rsrc624) #10, !dbg !11
  %172 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local90), !dbg !11
  %173 = zext i32 %172 to i64, !dbg !11
  %174 = inttoptr i64 %173 to ptr addrspace(3), !dbg !11
  %address.local184 = getelementptr [2 x i8], ptr addrspace(1) %44, i64 %address.local95, !dbg !11
  %175 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %176 = ptrtoint ptr addrspace(1) %address.local184 to i64, !dbg !11
  %177 = sub i64 %176, %175, !dbg !11
  %dma.buffer.offset620 = trunc i64 %177 to i32, !dbg !11
  %178 = trunc i64 %175 to i32, !dbg !11
  %179 = insertelement <4 x i32> poison, i32 %178, i64 0, !dbg !11
  %180 = lshr i64 %175, 32, !dbg !11
  %181 = trunc i64 %180 to i32, !dbg !11
  %182 = insertelement <4 x i32> %179, i32 %181, i64 1, !dbg !11
  %183 = insertelement <4 x i32> %182, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc621 = insertelement <4 x i32> %183, i32 131072, i64 3, !dbg !11
  %184 = ptrtoint ptr addrspace(3) %174 to i32, !dbg !11
  %dma.buffer.lds622 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %184), !dbg !11
  %185 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds622, i32 %dma.buffer.offset620, <4 x i32> %dma.buffer.rsrc621) #10, !dbg !11
  %186 = add nuw i64 %54, 2048, !dbg !15
  %187 = icmp slt i64 %186, 0, !dbg !15
  %188 = sub nuw i64 -2049, %54, !dbg !15
  %189 = select i1 %187, i64 %188, i64 %186, !dbg !15
  %190 = sdiv i64 %189, 8192, !dbg !15
  %.lobit = ashr i64 %186, 63, !dbg !15
  %191 = xor i64 %190, %.lobit, !dbg !15
  %192 = add nsw i64 %191, %15, !dbg !15
  %193 = add nsw i64 %94, %48, !dbg !15
  %194 = add nsw i64 %193, 32, !dbg !15
  %195 = icmp slt i64 %193, -32, !dbg !15
  %196 = sub nsw i64 -33, %193, !dbg !15
  %197 = select i1 %195, i64 %196, i64 %194, !dbg !15
  %198 = sdiv i64 %197, 128, !dbg !15
  %199 = sext i1 %195 to i64, !dbg !15
  %200 = xor i64 %198, %199, !dbg !15
  %201 = shl nsw i64 %200, 7, !dbg !15
  %202 = sub i64 %48, %201, !dbg !15
  %203 = shl i64 %192, 19, !dbg !15
  %204 = and i64 %203, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %205 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %204, !dbg !15
  %206 = getelementptr [2 x i8], ptr addrspace(1) %205, i64 %94, !dbg !15
  %207 = getelementptr [2 x i8], ptr addrspace(1) %206, i64 %202, !dbg !15
  %208 = getelementptr i8, ptr addrspace(1) %207, i64 64, !dbg !15
  %209 = getelementptr [2 x i8], ptr addrspace(1) %208, i64 %100, !dbg !15
  %210 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local100), !dbg !15
  %211 = zext i32 %210 to i64, !dbg !15
  %212 = inttoptr i64 %211 to ptr addrspace(3), !dbg !15
  %address.local185 = getelementptr [2 x i8], ptr addrspace(1) %209, i64 %address.local105, !dbg !15
  %213 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %214 = ptrtoint ptr addrspace(1) %address.local185 to i64, !dbg !15
  %215 = sub i64 %214, %213, !dbg !15
  %dma.buffer.offset617 = trunc i64 %215 to i32, !dbg !15
  %216 = trunc i64 %213 to i32, !dbg !15
  %217 = insertelement <4 x i32> poison, i32 %216, i64 0, !dbg !15
  %218 = lshr i64 %213, 32, !dbg !15
  %219 = trunc i64 %218 to i32, !dbg !15
  %220 = insertelement <4 x i32> %217, i32 %219, i64 1, !dbg !15
  %221 = insertelement <4 x i32> %220, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc618 = insertelement <4 x i32> %221, i32 131072, i64 3, !dbg !15
  %222 = ptrtoint ptr addrspace(3) %212 to i32, !dbg !15
  %dma.buffer.lds619 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %222), !dbg !15
  %223 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds619, i32 %dma.buffer.offset617, <4 x i32> %dma.buffer.rsrc618) #10, !dbg !15
  %224 = load <4 x half>, ptr addrspace(3) %address.local119, align 2, !dbg !12
  %225 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %226 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %227 = fmul <4 x half> %224, splat (half 0xH3015), !dbg !12
  %228 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %229 = fmul <4 x half> %225, splat (half 0xH3015), !dbg !12
  %230 = fmul <4 x half> %226, splat (half 0xH3015), !dbg !12
  %231 = fmul <4 x half> %228, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %232 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %232, <4 x float> %130) #9, !dbg !16
  %234 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %235 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %234, <4 x float> %233) #9, !dbg !16
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %235, <4 x float> %134) #9, !dbg !16
  %238 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %238, <4 x float> %237) #9, !dbg !16
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %239, <4 x float> %138) #9, !dbg !16
  %242 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %242, <4 x float> %241) #9, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %243, <4 x float> %142) #9, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %229, <4 x half> %246, <4 x float> %245) #9, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %247, <4 x float> %146) #9, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %251 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %250, <4 x float> %249) #9, !dbg !16
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %251, <4 x float> %150) #9, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %255 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %254, <4 x float> %253) #9, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %255, <4 x float> %154) #9, !dbg !16
  %258 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %259 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %258, <4 x float> %257) #9, !dbg !16
  %261 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %230, <4 x half> %259, <4 x float> %157) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %261, <4 x float> %262) #9, !dbg !16
  %264 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local78), !dbg !11
  %265 = zext i32 %264 to i64, !dbg !11
  %266 = inttoptr i64 %265 to ptr addrspace(3), !dbg !11
  %address.local186 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local83, !dbg !11
  %267 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %268 = ptrtoint ptr addrspace(1) %address.local186 to i64, !dbg !11
  %269 = sub i64 %268, %267, !dbg !11
  %dma.buffer.offset614 = trunc i64 %269 to i32, !dbg !11
  %270 = trunc i64 %267 to i32, !dbg !11
  %271 = insertelement <4 x i32> poison, i32 %270, i64 0, !dbg !11
  %272 = lshr i64 %267, 32, !dbg !11
  %273 = trunc i64 %272 to i32, !dbg !11
  %274 = insertelement <4 x i32> %271, i32 %273, i64 1, !dbg !11
  %275 = insertelement <4 x i32> %274, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc615 = insertelement <4 x i32> %275, i32 131072, i64 3, !dbg !11
  %276 = ptrtoint ptr addrspace(3) %266 to i32, !dbg !11
  %dma.buffer.lds616 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %276), !dbg !11
  %277 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds616, i32 %dma.buffer.offset614, <4 x i32> %dma.buffer.rsrc615) #10, !dbg !11
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local90), !dbg !11
  %279 = zext i32 %278 to i64, !dbg !11
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !11
  %address.local187 = getelementptr [2 x i8], ptr addrspace(1) %50, i64 %address.local95, !dbg !11
  %281 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %282 = ptrtoint ptr addrspace(1) %address.local187 to i64, !dbg !11
  %283 = sub i64 %282, %281, !dbg !11
  %dma.buffer.offset611 = trunc i64 %283 to i32, !dbg !11
  %284 = trunc i64 %281 to i32, !dbg !11
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !11
  %286 = lshr i64 %281, 32, !dbg !11
  %287 = trunc i64 %286 to i32, !dbg !11
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !11
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc612 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !11
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !11
  %dma.buffer.lds613 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !11
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds613, i32 %dma.buffer.offset611, <4 x i32> %dma.buffer.rsrc612) #10, !dbg !11
  %292 = add nuw i64 %54, 4096, !dbg !15
  %293 = icmp slt i64 %292, 0, !dbg !15
  %294 = sub nuw i64 -4097, %54, !dbg !15
  %295 = select i1 %293, i64 %294, i64 %292, !dbg !15
  %296 = sdiv i64 %295, 8192, !dbg !15
  %.lobit2 = ashr i64 %292, 63, !dbg !15
  %297 = xor i64 %296, %.lobit2, !dbg !15
  %298 = add nsw i64 %297, %15, !dbg !15
  %299 = add nsw i64 %193, 64, !dbg !15
  %300 = icmp slt i64 %193, -64, !dbg !15
  %301 = sub nsw i64 -65, %193, !dbg !15
  %302 = select i1 %300, i64 %301, i64 %299, !dbg !15
  %303 = sdiv i64 %302, 128, !dbg !15
  %304 = sext i1 %300 to i64, !dbg !15
  %305 = xor i64 %303, %304, !dbg !15
  %306 = shl nsw i64 %305, 7, !dbg !15
  %307 = sub i64 %48, %306, !dbg !15
  %308 = shl i64 %298, 19, !dbg !15
  %309 = and i64 %308, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %310 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %309, !dbg !15
  %311 = getelementptr [2 x i8], ptr addrspace(1) %310, i64 %94, !dbg !15
  %312 = getelementptr [2 x i8], ptr addrspace(1) %311, i64 %307, !dbg !15
  %313 = getelementptr i8, ptr addrspace(1) %312, i64 128, !dbg !15
  %314 = getelementptr [2 x i8], ptr addrspace(1) %313, i64 %100, !dbg !15
  %315 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local100), !dbg !15
  %316 = zext i32 %315 to i64, !dbg !15
  %317 = inttoptr i64 %316 to ptr addrspace(3), !dbg !15
  %address.local188 = getelementptr [2 x i8], ptr addrspace(1) %314, i64 %address.local105, !dbg !15
  %318 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %319 = ptrtoint ptr addrspace(1) %address.local188 to i64, !dbg !15
  %320 = sub i64 %319, %318, !dbg !15
  %dma.buffer.offset608 = trunc i64 %320 to i32, !dbg !15
  %321 = trunc i64 %318 to i32, !dbg !15
  %322 = insertelement <4 x i32> poison, i32 %321, i64 0, !dbg !15
  %323 = lshr i64 %318, 32, !dbg !15
  %324 = trunc i64 %323 to i32, !dbg !15
  %325 = insertelement <4 x i32> %322, i32 %324, i64 1, !dbg !15
  %326 = insertelement <4 x i32> %325, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc609 = insertelement <4 x i32> %326, i32 131072, i64 3, !dbg !15
  %327 = ptrtoint ptr addrspace(3) %317 to i32, !dbg !15
  %dma.buffer.lds610 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %327), !dbg !15
  %328 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds610, i32 %dma.buffer.offset608, <4 x i32> %dma.buffer.rsrc609) #10, !dbg !15
  %329 = load <4 x half>, ptr addrspace(3) %address.local119, align 2, !dbg !12
  %330 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %331 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %332 = fmul <4 x half> %329, splat (half 0xH3015), !dbg !12
  %333 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %334 = fmul <4 x half> %330, splat (half 0xH3015), !dbg !12
  %335 = fmul <4 x half> %331, splat (half 0xH3015), !dbg !12
  %336 = fmul <4 x half> %333, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %337 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %332, <4 x half> %337, <4 x float> %236) #9, !dbg !16
  %339 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %340 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %339, <4 x float> %338) #9, !dbg !16
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %332, <4 x half> %340, <4 x float> %240) #9, !dbg !16
  %343 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %344 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %343, <4 x float> %342) #9, !dbg !16
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %332, <4 x half> %344, <4 x float> %244) #9, !dbg !16
  %347 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %347, <4 x float> %346) #9, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %332, <4 x half> %348, <4 x float> %248) #9, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %334, <4 x half> %351, <4 x float> %350) #9, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %352, <4 x float> %252) #9, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %356 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %355, <4 x float> %354) #9, !dbg !16
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %356, <4 x float> %256) #9, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %359, <4 x float> %358) #9, !dbg !16
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %360, <4 x float> %260) #9, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %364 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %363, <4 x float> %362) #9, !dbg !16
  %366 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %364, <4 x float> %263) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %366, <4 x float> %367) #9, !dbg !16
  %369 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local78), !dbg !11
  %370 = zext i32 %369 to i64, !dbg !11
  %371 = inttoptr i64 %370 to ptr addrspace(3), !dbg !11
  %address.local189 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local83, !dbg !11
  %372 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %373 = ptrtoint ptr addrspace(1) %address.local189 to i64, !dbg !11
  %374 = sub i64 %373, %372, !dbg !11
  %dma.buffer.offset605 = trunc i64 %374 to i32, !dbg !11
  %375 = trunc i64 %372 to i32, !dbg !11
  %376 = insertelement <4 x i32> poison, i32 %375, i64 0, !dbg !11
  %377 = lshr i64 %372, 32, !dbg !11
  %378 = trunc i64 %377 to i32, !dbg !11
  %379 = insertelement <4 x i32> %376, i32 %378, i64 1, !dbg !11
  %380 = insertelement <4 x i32> %379, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc606 = insertelement <4 x i32> %380, i32 131072, i64 3, !dbg !11
  %381 = ptrtoint ptr addrspace(3) %371 to i32, !dbg !11
  %dma.buffer.lds607 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %381), !dbg !11
  %382 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds607, i32 %dma.buffer.offset605, <4 x i32> %dma.buffer.rsrc606) #10, !dbg !11
  %383 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local90), !dbg !11
  %384 = zext i32 %383 to i64, !dbg !11
  %385 = inttoptr i64 %384 to ptr addrspace(3), !dbg !11
  %address.local190 = getelementptr [2 x i8], ptr addrspace(1) %52, i64 %address.local95, !dbg !11
  %386 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %387 = ptrtoint ptr addrspace(1) %address.local190 to i64, !dbg !11
  %388 = sub i64 %387, %386, !dbg !11
  %dma.buffer.offset602 = trunc i64 %388 to i32, !dbg !11
  %389 = trunc i64 %386 to i32, !dbg !11
  %390 = insertelement <4 x i32> poison, i32 %389, i64 0, !dbg !11
  %391 = lshr i64 %386, 32, !dbg !11
  %392 = trunc i64 %391 to i32, !dbg !11
  %393 = insertelement <4 x i32> %390, i32 %392, i64 1, !dbg !11
  %394 = insertelement <4 x i32> %393, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc603 = insertelement <4 x i32> %394, i32 131072, i64 3, !dbg !11
  %395 = ptrtoint ptr addrspace(3) %385 to i32, !dbg !11
  %dma.buffer.lds604 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %395), !dbg !11
  %396 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds604, i32 %dma.buffer.offset602, <4 x i32> %dma.buffer.rsrc603) #10, !dbg !11
  %397 = add nuw i64 %54, 6144, !dbg !15
  %398 = icmp slt i64 %397, 0, !dbg !15
  %399 = sub nuw i64 -6145, %54, !dbg !15
  %400 = select i1 %398, i64 %399, i64 %397, !dbg !15
  %401 = sdiv i64 %400, 8192, !dbg !15
  %.lobit3 = ashr i64 %397, 63, !dbg !15
  %402 = xor i64 %401, %.lobit3, !dbg !15
  %403 = add nsw i64 %402, %15, !dbg !15
  %404 = add nsw i64 %193, 96, !dbg !15
  %405 = icmp slt i64 %193, -96, !dbg !15
  %406 = sub nsw i64 -97, %193, !dbg !15
  %407 = select i1 %405, i64 %406, i64 %404, !dbg !15
  %408 = sdiv i64 %407, 128, !dbg !15
  %409 = sext i1 %405 to i64, !dbg !15
  %410 = xor i64 %408, %409, !dbg !15
  %411 = shl nsw i64 %410, 7, !dbg !15
  %412 = sub i64 %48, %411, !dbg !15
  %413 = shl i64 %403, 19, !dbg !15
  %414 = and i64 %413, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %415 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %414, !dbg !15
  %416 = getelementptr [2 x i8], ptr addrspace(1) %415, i64 %94, !dbg !15
  %417 = getelementptr [2 x i8], ptr addrspace(1) %416, i64 %412, !dbg !15
  %418 = getelementptr i8, ptr addrspace(1) %417, i64 192, !dbg !15
  %419 = getelementptr [2 x i8], ptr addrspace(1) %418, i64 %100, !dbg !15
  %420 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local100), !dbg !15
  %421 = zext i32 %420 to i64, !dbg !15
  %422 = inttoptr i64 %421 to ptr addrspace(3), !dbg !15
  %address.local191 = getelementptr [2 x i8], ptr addrspace(1) %419, i64 %address.local105, !dbg !15
  %423 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %424 = ptrtoint ptr addrspace(1) %address.local191 to i64, !dbg !15
  %425 = sub i64 %424, %423, !dbg !15
  %dma.buffer.offset599 = trunc i64 %425 to i32, !dbg !15
  %426 = trunc i64 %423 to i32, !dbg !15
  %427 = insertelement <4 x i32> poison, i32 %426, i64 0, !dbg !15
  %428 = lshr i64 %423, 32, !dbg !15
  %429 = trunc i64 %428 to i32, !dbg !15
  %430 = insertelement <4 x i32> %427, i32 %429, i64 1, !dbg !15
  %431 = insertelement <4 x i32> %430, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc600 = insertelement <4 x i32> %431, i32 131072, i64 3, !dbg !15
  %432 = ptrtoint ptr addrspace(3) %422 to i32, !dbg !15
  %dma.buffer.lds601 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %432), !dbg !15
  %433 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds601, i32 %dma.buffer.offset599, <4 x i32> %dma.buffer.rsrc600) #10, !dbg !15
  %434 = load <4 x half>, ptr addrspace(3) %address.local119, align 2, !dbg !12
  %435 = load <4 x half>, ptr addrspace(3) %address.local121, align 2, !dbg !12
  %436 = load <4 x half>, ptr addrspace(3) %address.local123, align 2, !dbg !12
  %437 = fmul <4 x half> %434, splat (half 0xH3015), !dbg !12
  %438 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %439 = fmul <4 x half> %435, splat (half 0xH3015), !dbg !12
  %440 = fmul <4 x half> %436, splat (half 0xH3015), !dbg !12
  %441 = fmul <4 x half> %438, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %442 = add nuw nsw i64 %94, %15, !dbg !17
  %443 = shl i64 %442, 19, !dbg !17
  %444 = and i64 %443, 16252928, !dbg !17
  %445 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %444, !dbg !17
  %446 = getelementptr [2 x i8], ptr addrspace(1) %445, i64 %100, !dbg !17
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local97, !dbg !17
  %address.local193 = ptrtoint ptr addrspace(3) %address.local192 to i64, !dbg !17
  %address.local194 = trunc i64 %address.local193 to i32, !dbg !17
  %447 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local194), !dbg !17
  %448 = zext i32 %447 to i64, !dbg !17
  %449 = inttoptr i64 %448 to ptr addrspace(3), !dbg !17
  %address.local195 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local196 = and i64 %address.local195, 8064, !dbg !17
  %address.local197 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local198 = and i64 %address.local197, 24, !dbg !17
  %address.local199 = or disjoint i64 %address.local196, %address.local198, !dbg !17
  %address.local200 = getelementptr [2 x i8], ptr addrspace(1) %446, i64 %address.local199, !dbg !17
  %450 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %451 = ptrtoint ptr addrspace(1) %address.local200 to i64, !dbg !17
  %452 = sub i64 %451, %450, !dbg !17
  %dma.buffer.offset596 = trunc i64 %452 to i32, !dbg !17
  %453 = trunc i64 %450 to i32, !dbg !17
  %454 = insertelement <4 x i32> poison, i32 %453, i64 0, !dbg !17
  %455 = lshr i64 %450, 32, !dbg !17
  %456 = trunc i64 %455 to i32, !dbg !17
  %457 = insertelement <4 x i32> %454, i32 %456, i64 1, !dbg !17
  %458 = insertelement <4 x i32> %457, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc597 = insertelement <4 x i32> %458, i32 131072, i64 3, !dbg !17
  %459 = ptrtoint ptr addrspace(3) %449 to i32, !dbg !17
  %dma.buffer.lds598 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %459), !dbg !17
  %460 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds598, i32 %dma.buffer.offset596, <4 x i32> %dma.buffer.rsrc597) #10, !dbg !17
  %461 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %461, <4 x float> %341) #9, !dbg !16
  %463 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %464 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %463, <4 x float> %462) #9, !dbg !16
  %466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %464, <4 x float> %345) #9, !dbg !16
  %467 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %468 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %467, <4 x float> %466) #9, !dbg !16
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %468, <4 x float> %349) #9, !dbg !16
  %471 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %472 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %471, <4 x float> %470) #9, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %472, <4 x float> %353) #9, !dbg !16
  %475 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %476 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %439, <4 x half> %475, <4 x float> %474) #9, !dbg !16
  %478 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %476, <4 x float> %357) #9, !dbg !16
  %479 = load <4 x half>, ptr addrspace(3) %address.local146, align 2, !dbg !16
  %480 = load <4 x half>, ptr addrspace(3) %address.local152, align 2, !dbg !16
  %481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %479, <4 x float> %478) #9, !dbg !16
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %480, <4 x float> %361) #9, !dbg !16
  %483 = load <4 x half>, ptr addrspace(3) %address.local158, align 2, !dbg !16
  %484 = load <4 x half>, ptr addrspace(3) %address.local164, align 2, !dbg !16
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %483, <4 x float> %482) #9, !dbg !16
  %486 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %484, <4 x float> %365) #9, !dbg !16
  %487 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %488 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %489 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %487, <4 x float> %486) #9, !dbg !16
  %490 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %440, <4 x half> %488, <4 x float> %368) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %492 = or disjoint i64 %63, 63, !dbg !13
  %493 = icmp uge i64 %19, %492, !dbg !13
  %494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %441, <4 x half> %490, <4 x float> %491) #9, !dbg !16
  %495 = and i1 %39, %493, !dbg !13
  br i1 %495, label %496, label %497, !dbg !13

496:                                              ; preds = %62
  br label %577, !dbg !13

497:                                              ; preds = %62
  %498 = add i64 %19, %35, !dbg !13
  %499 = shl nuw nsw i64 %36, 4, !dbg !13
  %500 = add i64 %498, %499, !dbg !13
  %501 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %501 to i64, !dbg !13
  %502 = sub i64 %500, %index.compact, !dbg !13
  %503 = and i64 %35, 48, !dbg !13
  %504 = or disjoint i64 %63, %503, !dbg !13
  %505 = add i64 %502, 1, !dbg !22
  %506 = lshr i64 %505, 32, !dbg !23
  %compare.high = trunc i64 %506 to i32, !dbg !23
  %507 = lshr i64 %504, 32, !dbg !23
  %compare.high462 = trunc i64 %507 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high462, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high462, !dbg !23
  %compare.low.base = trunc i64 %505 to i32, !dbg !23
  %compare.low.base463 = trunc i64 %504 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base463, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %508 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low464 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result465 = select i1 %compare.high.equal, i1 %compare.low464, i1 %compare.high.order, !dbg !23
  %509 = select i1 %compare.result465, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base466 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local467 = or i32 %compare.local.base466, 2, !dbg !23
  %compare.low468 = icmp ugt i32 %compare.low.base, %compare.local467, !dbg !23
  %compare.result469 = select i1 %compare.high.equal, i1 %compare.low468, i1 %compare.high.order, !dbg !23
  %510 = select i1 %compare.result469, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base470 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local471 = or i32 %compare.local.base470, 3, !dbg !23
  %compare.low472 = icmp ugt i32 %compare.low.base, %compare.local471, !dbg !23
  %compare.result473 = select i1 %compare.high.equal, i1 %compare.low472, i1 %compare.high.order, !dbg !23
  %511 = select i1 %compare.result473, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base474 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local475 = or i32 %compare.local.base474, 4, !dbg !23
  %compare.low476 = icmp ugt i32 %compare.low.base, %compare.local475, !dbg !23
  %compare.result477 = select i1 %compare.high.equal, i1 %compare.low476, i1 %compare.high.order, !dbg !23
  %512 = select i1 %compare.result477, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base478 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local479 = or i32 %compare.local.base478, 5, !dbg !23
  %compare.low480 = icmp ugt i32 %compare.low.base, %compare.local479, !dbg !23
  %compare.result481 = select i1 %compare.high.equal, i1 %compare.low480, i1 %compare.high.order, !dbg !23
  %513 = select i1 %compare.result481, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base482 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local483 = or i32 %compare.local.base482, 6, !dbg !23
  %compare.low484 = icmp ugt i32 %compare.low.base, %compare.local483, !dbg !23
  %compare.result485 = select i1 %compare.high.equal, i1 %compare.low484, i1 %compare.high.order, !dbg !23
  %514 = select i1 %compare.result485, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base486 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local487 = or i32 %compare.local.base486, 7, !dbg !23
  %compare.low488 = icmp ugt i32 %compare.low.base, %compare.local487, !dbg !23
  %compare.result489 = select i1 %compare.high.equal, i1 %compare.low488, i1 %compare.high.order, !dbg !23
  %515 = select i1 %compare.result489, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base490 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local491 = or i32 %compare.local.base490, 8, !dbg !23
  %compare.low492 = icmp ugt i32 %compare.low.base, %compare.local491, !dbg !23
  %compare.result493 = select i1 %compare.high.equal, i1 %compare.low492, i1 %compare.high.order, !dbg !23
  %516 = select i1 %compare.result493, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base494 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local495 = or i32 %compare.local.base494, 9, !dbg !23
  %compare.low496 = icmp ugt i32 %compare.low.base, %compare.local495, !dbg !23
  %compare.result497 = select i1 %compare.high.equal, i1 %compare.low496, i1 %compare.high.order, !dbg !23
  %517 = select i1 %compare.result497, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base498 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local499 = or i32 %compare.local.base498, 10, !dbg !23
  %compare.low500 = icmp ugt i32 %compare.low.base, %compare.local499, !dbg !23
  %compare.result501 = select i1 %compare.high.equal, i1 %compare.low500, i1 %compare.high.order, !dbg !23
  %518 = select i1 %compare.result501, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base502 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local503 = or i32 %compare.local.base502, 11, !dbg !23
  %compare.low504 = icmp ugt i32 %compare.low.base, %compare.local503, !dbg !23
  %compare.result505 = select i1 %compare.high.equal, i1 %compare.low504, i1 %compare.high.order, !dbg !23
  %519 = select i1 %compare.result505, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base506 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local507 = or i32 %compare.local.base506, 12, !dbg !23
  %compare.low508 = icmp ugt i32 %compare.low.base, %compare.local507, !dbg !23
  %compare.result509 = select i1 %compare.high.equal, i1 %compare.low508, i1 %compare.high.order, !dbg !23
  %520 = select i1 %compare.result509, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base510 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local511 = or i32 %compare.local.base510, 13, !dbg !23
  %compare.low512 = icmp ugt i32 %compare.low.base, %compare.local511, !dbg !23
  %compare.result513 = select i1 %compare.high.equal, i1 %compare.low512, i1 %compare.high.order, !dbg !23
  %521 = select i1 %compare.result513, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base514 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local515 = or i32 %compare.local.base514, 14, !dbg !23
  %compare.low516 = icmp ugt i32 %compare.low.base, %compare.local515, !dbg !23
  %compare.result517 = select i1 %compare.high.equal, i1 %compare.low516, i1 %compare.high.order, !dbg !23
  %522 = select i1 %compare.result517, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base518 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base463, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local519 = or i32 %compare.local.base518, 15, !dbg !23
  %compare.low520 = icmp ugt i32 %compare.low.base, %compare.local519, !dbg !23
  %compare.result521 = select i1 %compare.high.equal, i1 %compare.low520, i1 %compare.high.order, !dbg !23
  %523 = select i1 %compare.result521, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %524 = add i64 %502, 65, !dbg !22
  %525 = lshr i64 %524, 32, !dbg !23
  %compare.high522 = trunc i64 %525 to i32, !dbg !23
  %526 = lshr i64 %504, 32, !dbg !23
  %compare.high523 = trunc i64 %526 to i32, !dbg !23
  %compare.high.equal524 = icmp eq i32 %compare.high522, %compare.high523, !dbg !23
  %compare.high.order525 = icmp ugt i32 %compare.high522, %compare.high523, !dbg !23
  %compare.low.base526 = trunc i64 %524 to i32, !dbg !23
  %compare.low.base527 = trunc i64 %504 to i32, !dbg !23
  %compare.low528 = icmp ugt i32 %compare.low.base526, %compare.low.base527, !dbg !23
  %compare.result529 = select i1 %compare.high.equal524, i1 %compare.low528, i1 %compare.high.order525, !dbg !23
  %527 = select i1 %compare.result529, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base530 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local531 = or i32 %compare.local.base530, 1, !dbg !23
  %compare.low532 = icmp ugt i32 %compare.low.base526, %compare.local531, !dbg !23
  %compare.result533 = select i1 %compare.high.equal524, i1 %compare.low532, i1 %compare.high.order525, !dbg !23
  %528 = select i1 %compare.result533, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base534 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local535 = or i32 %compare.local.base534, 2, !dbg !23
  %compare.low536 = icmp ugt i32 %compare.low.base526, %compare.local535, !dbg !23
  %compare.result537 = select i1 %compare.high.equal524, i1 %compare.low536, i1 %compare.high.order525, !dbg !23
  %529 = select i1 %compare.result537, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base538 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local539 = or i32 %compare.local.base538, 3, !dbg !23
  %compare.low540 = icmp ugt i32 %compare.low.base526, %compare.local539, !dbg !23
  %compare.result541 = select i1 %compare.high.equal524, i1 %compare.low540, i1 %compare.high.order525, !dbg !23
  %530 = select i1 %compare.result541, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base542 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local543 = or i32 %compare.local.base542, 4, !dbg !23
  %compare.low544 = icmp ugt i32 %compare.low.base526, %compare.local543, !dbg !23
  %compare.result545 = select i1 %compare.high.equal524, i1 %compare.low544, i1 %compare.high.order525, !dbg !23
  %531 = select i1 %compare.result545, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base546 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local547 = or i32 %compare.local.base546, 5, !dbg !23
  %compare.low548 = icmp ugt i32 %compare.low.base526, %compare.local547, !dbg !23
  %compare.result549 = select i1 %compare.high.equal524, i1 %compare.low548, i1 %compare.high.order525, !dbg !23
  %532 = select i1 %compare.result549, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base550 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local551 = or i32 %compare.local.base550, 6, !dbg !23
  %compare.low552 = icmp ugt i32 %compare.low.base526, %compare.local551, !dbg !23
  %compare.result553 = select i1 %compare.high.equal524, i1 %compare.low552, i1 %compare.high.order525, !dbg !23
  %533 = select i1 %compare.result553, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base554 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local555 = or i32 %compare.local.base554, 7, !dbg !23
  %compare.low556 = icmp ugt i32 %compare.low.base526, %compare.local555, !dbg !23
  %compare.result557 = select i1 %compare.high.equal524, i1 %compare.low556, i1 %compare.high.order525, !dbg !23
  %534 = select i1 %compare.result557, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base558 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local559 = or i32 %compare.local.base558, 8, !dbg !23
  %compare.low560 = icmp ugt i32 %compare.low.base526, %compare.local559, !dbg !23
  %compare.result561 = select i1 %compare.high.equal524, i1 %compare.low560, i1 %compare.high.order525, !dbg !23
  %535 = select i1 %compare.result561, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base562 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local563 = or i32 %compare.local.base562, 9, !dbg !23
  %compare.low564 = icmp ugt i32 %compare.low.base526, %compare.local563, !dbg !23
  %compare.result565 = select i1 %compare.high.equal524, i1 %compare.low564, i1 %compare.high.order525, !dbg !23
  %536 = select i1 %compare.result565, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base566 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local567 = or i32 %compare.local.base566, 10, !dbg !23
  %compare.low568 = icmp ugt i32 %compare.low.base526, %compare.local567, !dbg !23
  %compare.result569 = select i1 %compare.high.equal524, i1 %compare.low568, i1 %compare.high.order525, !dbg !23
  %537 = select i1 %compare.result569, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base570 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local571 = or i32 %compare.local.base570, 11, !dbg !23
  %compare.low572 = icmp ugt i32 %compare.low.base526, %compare.local571, !dbg !23
  %compare.result573 = select i1 %compare.high.equal524, i1 %compare.low572, i1 %compare.high.order525, !dbg !23
  %538 = select i1 %compare.result573, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base574 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local575 = or i32 %compare.local.base574, 12, !dbg !23
  %compare.low576 = icmp ugt i32 %compare.low.base526, %compare.local575, !dbg !23
  %compare.result577 = select i1 %compare.high.equal524, i1 %compare.low576, i1 %compare.high.order525, !dbg !23
  %539 = select i1 %compare.result577, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base578 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local579 = or i32 %compare.local.base578, 13, !dbg !23
  %compare.low580 = icmp ugt i32 %compare.low.base526, %compare.local579, !dbg !23
  %compare.result581 = select i1 %compare.high.equal524, i1 %compare.low580, i1 %compare.high.order525, !dbg !23
  %540 = select i1 %compare.result581, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base582 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local583 = or i32 %compare.local.base582, 14, !dbg !23
  %compare.low584 = icmp ugt i32 %compare.low.base526, %compare.local583, !dbg !23
  %compare.result585 = select i1 %compare.high.equal524, i1 %compare.low584, i1 %compare.high.order525, !dbg !23
  %541 = select i1 %compare.result585, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base586 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base527, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local587 = or i32 %compare.local.base586, 15, !dbg !23
  %compare.low588 = icmp ugt i32 %compare.low.base526, %compare.local587, !dbg !23
  %compare.result589 = select i1 %compare.high.equal524, i1 %compare.low588, i1 %compare.high.order525, !dbg !23
  %542 = select i1 %compare.result589, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %543 = insertelement <16 x float> poison, float %508, i64 0, !dbg !13
  %544 = insertelement <16 x float> %543, float %509, i64 1, !dbg !13
  %545 = insertelement <16 x float> %544, float %510, i64 2, !dbg !13
  %546 = insertelement <16 x float> %545, float %511, i64 3, !dbg !13
  %547 = insertelement <16 x float> %546, float %512, i64 4, !dbg !13
  %548 = insertelement <16 x float> %547, float %513, i64 5, !dbg !13
  %549 = insertelement <16 x float> %548, float %514, i64 6, !dbg !13
  %550 = insertelement <16 x float> %549, float %515, i64 7, !dbg !13
  %551 = insertelement <16 x float> %550, float %516, i64 8, !dbg !13
  %552 = insertelement <16 x float> %551, float %517, i64 9, !dbg !13
  %553 = insertelement <16 x float> %552, float %518, i64 10, !dbg !13
  %554 = insertelement <16 x float> %553, float %519, i64 11, !dbg !13
  %555 = insertelement <16 x float> %554, float %520, i64 12, !dbg !13
  %556 = insertelement <16 x float> %555, float %521, i64 13, !dbg !13
  %557 = insertelement <16 x float> %556, float %522, i64 14, !dbg !13
  %558 = insertelement <16 x float> %557, float %523, i64 15, !dbg !13
  %559 = insertvalue [2 x <16 x float>] poison, <16 x float> %558, 0, !dbg !13
  %560 = insertelement <16 x float> poison, float %527, i64 0, !dbg !13
  %561 = insertelement <16 x float> %560, float %528, i64 1, !dbg !13
  %562 = insertelement <16 x float> %561, float %529, i64 2, !dbg !13
  %563 = insertelement <16 x float> %562, float %530, i64 3, !dbg !13
  %564 = insertelement <16 x float> %563, float %531, i64 4, !dbg !13
  %565 = insertelement <16 x float> %564, float %532, i64 5, !dbg !13
  %566 = insertelement <16 x float> %565, float %533, i64 6, !dbg !13
  %567 = insertelement <16 x float> %566, float %534, i64 7, !dbg !13
  %568 = insertelement <16 x float> %567, float %535, i64 8, !dbg !13
  %569 = insertelement <16 x float> %568, float %536, i64 9, !dbg !13
  %570 = insertelement <16 x float> %569, float %537, i64 10, !dbg !13
  %571 = insertelement <16 x float> %570, float %538, i64 11, !dbg !13
  %572 = insertelement <16 x float> %571, float %539, i64 12, !dbg !13
  %573 = insertelement <16 x float> %572, float %540, i64 13, !dbg !13
  %574 = insertelement <16 x float> %573, float %541, i64 14, !dbg !13
  %575 = insertelement <16 x float> %574, float %542, i64 15, !dbg !13
  %576 = insertvalue [2 x <16 x float>] %559, <16 x float> %575, 1, !dbg !13
  br label %577, !dbg !13

577:                                              ; preds = %496, %497
  %578 = phi [2 x <16 x float>] [ %576, %497 ], [ zeroinitializer, %496 ], !dbg !13
  br label %579, !dbg !13

579:                                              ; preds = %577
  %580 = extractvalue [2 x <16 x float>] %578, 0, !dbg !25
  %581 = shufflevector <16 x float> %580, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %582 = fadd <4 x float> %465, %581, !dbg !25
  %583 = extractelement <4 x float> %582, i64 0, !dbg !25
  %584 = extractelement <4 x float> %582, i64 1, !dbg !25
  %585 = extractelement <4 x float> %582, i64 2, !dbg !25
  %586 = extractelement <4 x float> %582, i64 3, !dbg !25
  %587 = shufflevector <16 x float> %580, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %588 = fadd <4 x float> %469, %587, !dbg !25
  %589 = extractelement <4 x float> %588, i64 0, !dbg !25
  %590 = extractelement <4 x float> %588, i64 1, !dbg !25
  %591 = extractelement <4 x float> %588, i64 2, !dbg !25
  %592 = extractelement <4 x float> %588, i64 3, !dbg !25
  %593 = shufflevector <16 x float> %580, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %594 = fadd <4 x float> %473, %593, !dbg !25
  %595 = extractelement <4 x float> %594, i64 0, !dbg !25
  %596 = extractelement <4 x float> %594, i64 1, !dbg !25
  %597 = extractelement <4 x float> %594, i64 2, !dbg !25
  %598 = extractelement <4 x float> %594, i64 3, !dbg !25
  %599 = shufflevector <16 x float> %580, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %600 = fadd <4 x float> %477, %599, !dbg !25
  %601 = extractelement <4 x float> %600, i64 0, !dbg !25
  %602 = extractelement <4 x float> %600, i64 1, !dbg !25
  %603 = extractelement <4 x float> %600, i64 2, !dbg !25
  %604 = extractelement <4 x float> %600, i64 3, !dbg !25
  %605 = extractvalue [2 x <16 x float>] %578, 1, !dbg !25
  %606 = shufflevector <16 x float> %605, <16 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %607 = fadd <4 x float> %481, %606, !dbg !25
  %608 = extractelement <4 x float> %607, i64 0, !dbg !25
  %609 = extractelement <4 x float> %607, i64 1, !dbg !25
  %610 = extractelement <4 x float> %607, i64 2, !dbg !25
  %611 = extractelement <4 x float> %607, i64 3, !dbg !25
  %612 = shufflevector <16 x float> %605, <16 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %613 = fadd <4 x float> %485, %612, !dbg !25
  %614 = extractelement <4 x float> %613, i64 0, !dbg !25
  %615 = extractelement <4 x float> %613, i64 1, !dbg !25
  %616 = extractelement <4 x float> %613, i64 2, !dbg !25
  %617 = extractelement <4 x float> %613, i64 3, !dbg !25
  %618 = shufflevector <16 x float> %605, <16 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %619 = fadd <4 x float> %489, %618, !dbg !25
  %620 = extractelement <4 x float> %619, i64 0, !dbg !25
  %621 = extractelement <4 x float> %619, i64 1, !dbg !25
  %622 = extractelement <4 x float> %619, i64 2, !dbg !25
  %623 = extractelement <4 x float> %619, i64 3, !dbg !25
  %624 = shufflevector <16 x float> %605, <16 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %625 = fadd <4 x float> %494, %624, !dbg !25
  %626 = extractelement <4 x float> %625, i64 0, !dbg !25
  %627 = extractelement <4 x float> %625, i64 1, !dbg !25
  %628 = extractelement <4 x float> %625, i64 2, !dbg !25
  %629 = extractelement <4 x float> %625, i64 3, !dbg !25
  %630 = call float @llvm.maxnum.f32(float %583, float 0xFFF0000000000000), !dbg !26
  %631 = call float @llvm.maxnum.f32(float %630, float %584), !dbg !26
  %632 = call float @llvm.maxnum.f32(float %631, float %585), !dbg !26
  %633 = call float @llvm.maxnum.f32(float %632, float %586), !dbg !26
  %634 = call float @llvm.maxnum.f32(float %633, float %589), !dbg !26
  %635 = call float @llvm.maxnum.f32(float %634, float %590), !dbg !26
  %636 = call float @llvm.maxnum.f32(float %635, float %591), !dbg !26
  %637 = call float @llvm.maxnum.f32(float %636, float %592), !dbg !26
  %638 = call float @llvm.maxnum.f32(float %637, float %595), !dbg !26
  %639 = call float @llvm.maxnum.f32(float %638, float %596), !dbg !26
  %640 = call float @llvm.maxnum.f32(float %639, float %597), !dbg !26
  %641 = call float @llvm.maxnum.f32(float %640, float %598), !dbg !26
  %642 = call float @llvm.maxnum.f32(float %641, float %601), !dbg !26
  %643 = call float @llvm.maxnum.f32(float %642, float %602), !dbg !26
  %644 = call float @llvm.maxnum.f32(float %643, float %603), !dbg !26
  %645 = call float @llvm.maxnum.f32(float %644, float %604), !dbg !26
  %646 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %647 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %646), !dbg !26
  %648 = add nuw nsw i32 %647, 64, !dbg !26
  %649 = xor i32 %647, 16, !dbg !26
  %650 = and i32 %648, 192, !dbg !26
  %651 = icmp samesign ult i32 %649, %650, !dbg !26
  %652 = select i1 %651, i32 %649, i32 %647, !dbg !26
  %653 = shl nuw nsw i32 %652, 2, !dbg !26
  %654 = bitcast float %645 to i32, !dbg !26
  %655 = call i32 @llvm.amdgcn.ds.bpermute(i32 %653, i32 %654), !dbg !26
  %656 = bitcast i32 %655 to float, !dbg !26
  %657 = call float @llvm.maxnum.f32(float %645, float %656), !dbg !26
  %658 = xor i32 %647, 32, !dbg !26
  %659 = icmp samesign ult i32 %658, %650, !dbg !26
  %660 = select i1 %659, i32 %658, i32 %647, !dbg !26
  %661 = shl nuw nsw i32 %660, 2, !dbg !26
  %662 = bitcast float %657 to i32, !dbg !26
  %663 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %662), !dbg !26
  %664 = bitcast i32 %663 to float, !dbg !26
  %665 = call float @llvm.maxnum.f32(float %657, float %664), !dbg !26
  %666 = extractvalue [2 x <1 x float>] %57, 0, !dbg !26
  %667 = insertelement <1 x float> poison, float %665, i64 0, !dbg !26
  %668 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %667, <1 x float> %666), !dbg !26
  %669 = call float @llvm.maxnum.f32(float %608, float 0xFFF0000000000000), !dbg !26
  %670 = call float @llvm.maxnum.f32(float %669, float %609), !dbg !26
  %671 = call float @llvm.maxnum.f32(float %670, float %610), !dbg !26
  %672 = call float @llvm.maxnum.f32(float %671, float %611), !dbg !26
  %673 = call float @llvm.maxnum.f32(float %672, float %614), !dbg !26
  %674 = call float @llvm.maxnum.f32(float %673, float %615), !dbg !26
  %675 = call float @llvm.maxnum.f32(float %674, float %616), !dbg !26
  %676 = call float @llvm.maxnum.f32(float %675, float %617), !dbg !26
  %677 = call float @llvm.maxnum.f32(float %676, float %620), !dbg !26
  %678 = call float @llvm.maxnum.f32(float %677, float %621), !dbg !26
  %679 = call float @llvm.maxnum.f32(float %678, float %622), !dbg !26
  %680 = call float @llvm.maxnum.f32(float %679, float %623), !dbg !26
  %681 = call float @llvm.maxnum.f32(float %680, float %626), !dbg !26
  %682 = call float @llvm.maxnum.f32(float %681, float %627), !dbg !26
  %683 = call float @llvm.maxnum.f32(float %682, float %628), !dbg !26
  %684 = call float @llvm.maxnum.f32(float %683, float %629), !dbg !26
  %685 = bitcast float %684 to i32, !dbg !26
  %686 = call i32 @llvm.amdgcn.ds.bpermute(i32 %653, i32 %685), !dbg !26
  %687 = bitcast i32 %686 to float, !dbg !26
  %688 = call float @llvm.maxnum.f32(float %684, float %687), !dbg !26
  %689 = bitcast float %688 to i32, !dbg !26
  %690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %689), !dbg !26
  %691 = bitcast i32 %690 to float, !dbg !26
  %692 = call float @llvm.maxnum.f32(float %688, float %691), !dbg !26
  %693 = extractvalue [2 x <1 x float>] %57, 1, !dbg !26
  %694 = insertelement <1 x float> poison, float %692, i64 0, !dbg !26
  %695 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %694, <1 x float> %693), !dbg !26
  %696 = shufflevector <1 x float> %668, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %697 = fsub <4 x float> %582, %696, !dbg !27
  %698 = fsub <4 x float> %588, %696, !dbg !27
  %699 = fsub <4 x float> %594, %696, !dbg !27
  %700 = fsub <4 x float> %600, %696, !dbg !27
  %701 = shufflevector <1 x float> %695, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %702 = fsub <4 x float> %607, %701, !dbg !27
  %703 = fsub <4 x float> %613, %701, !dbg !27
  %704 = fsub <4 x float> %619, %701, !dbg !27
  %705 = fsub <4 x float> %625, %701, !dbg !27
  %706 = fsub <1 x float> %666, %668, !dbg !28
  %707 = extractelement <1 x float> %706, i64 0, !dbg !29
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !29
  %709 = insertelement <4 x float> poison, float %708, i64 0, !dbg !34
  %710 = shufflevector <4 x float> %709, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %711 = extractvalue [2 x <1 x float>] %56, 0, !dbg !30
  %712 = extractelement <1 x float> %711, i64 0, !dbg !30
  %713 = fmul float %712, %708, !dbg !30
  %714 = fsub <1 x float> %693, %695, !dbg !28
  %715 = extractelement <1 x float> %714, i64 0, !dbg !29
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !29
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !34
  %718 = shufflevector <4 x float> %717, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %719 = extractvalue [2 x <1 x float>] %56, 1, !dbg !30
  %720 = extractelement <1 x float> %719, i64 0, !dbg !30
  %721 = fmul float %720, %716, !dbg !30
  %722 = extractelement <4 x float> %697, i64 0, !dbg !31
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !31
  %724 = insertelement <4 x float> poison, float %723, i64 0, !dbg !33
  %725 = extractelement <4 x float> %697, i64 1, !dbg !31
  %726 = call float @llvm.amdgcn.exp2.f32(float %725), !dbg !31
  %727 = insertelement <4 x float> %724, float %726, i64 1, !dbg !33
  %728 = fadd float %723, %726, !dbg !32
  %729 = extractelement <4 x float> %697, i64 2, !dbg !31
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !31
  %731 = insertelement <4 x float> %727, float %730, i64 2, !dbg !33
  %732 = fadd float %728, %730, !dbg !32
  %733 = extractelement <4 x float> %697, i64 3, !dbg !31
  %734 = call float @llvm.amdgcn.exp2.f32(float %733), !dbg !31
  %735 = insertelement <4 x float> %731, float %734, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %735) #9, !dbg !33, !frisk.register_lifetimes !1
  %736 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %736) #9, !dbg !32, !frisk.register_lifetimes !1
  %737 = fadd float %732, %734, !dbg !32
  %738 = extractelement <4 x float> %698, i64 0, !dbg !31
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !31
  %740 = insertelement <4 x float> poison, float %739, i64 0, !dbg !33
  %741 = fadd float %737, %739, !dbg !32
  %742 = extractelement <4 x float> %698, i64 1, !dbg !31
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !31
  %744 = insertelement <4 x float> %740, float %743, i64 1, !dbg !33
  %745 = fadd float %741, %743, !dbg !32
  %746 = extractelement <4 x float> %698, i64 2, !dbg !31
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !31
  %748 = insertelement <4 x float> %744, float %747, i64 2, !dbg !33
  %749 = fadd float %745, %747, !dbg !32
  %750 = extractelement <4 x float> %698, i64 3, !dbg !31
  %751 = call float @llvm.amdgcn.exp2.f32(float %750), !dbg !31
  %752 = insertelement <4 x float> %748, float %751, i64 3, !dbg !33
  %math.result60 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %752) #9, !dbg !33, !frisk.register_lifetimes !1
  %753 = fptrunc <4 x float> %math.result60 to <4 x half>, !dbg !33
  %math.packed61 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %753) #9, !dbg !32, !frisk.register_lifetimes !1
  %754 = fadd float %749, %751, !dbg !32
  %755 = extractelement <4 x float> %699, i64 0, !dbg !31
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !31
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !33
  %758 = fadd float %754, %756, !dbg !32
  %759 = extractelement <4 x float> %699, i64 1, !dbg !31
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !31
  %761 = insertelement <4 x float> %757, float %760, i64 1, !dbg !33
  %762 = fadd float %758, %760, !dbg !32
  %763 = extractelement <4 x float> %699, i64 2, !dbg !31
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !31
  %765 = insertelement <4 x float> %761, float %764, i64 2, !dbg !33
  %766 = fadd float %762, %764, !dbg !32
  %767 = extractelement <4 x float> %699, i64 3, !dbg !31
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !31
  %769 = insertelement <4 x float> %765, float %768, i64 3, !dbg !33
  %math.result62 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %769) #9, !dbg !33, !frisk.register_lifetimes !1
  %770 = fptrunc <4 x float> %math.result62 to <4 x half>, !dbg !33
  %math.packed63 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %770) #9, !dbg !32, !frisk.register_lifetimes !1
  %771 = fadd float %766, %768, !dbg !32
  %772 = extractelement <4 x float> %700, i64 0, !dbg !31
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !31
  %774 = insertelement <4 x float> poison, float %773, i64 0, !dbg !33
  %775 = fadd float %771, %773, !dbg !32
  %776 = extractelement <4 x float> %700, i64 1, !dbg !31
  %777 = call float @llvm.amdgcn.exp2.f32(float %776), !dbg !31
  %778 = insertelement <4 x float> %774, float %777, i64 1, !dbg !33
  %779 = fadd float %775, %777, !dbg !32
  %780 = extractelement <4 x float> %700, i64 2, !dbg !31
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !31
  %782 = insertelement <4 x float> %778, float %781, i64 2, !dbg !33
  %783 = fadd float %779, %781, !dbg !32
  %784 = extractelement <4 x float> %700, i64 3, !dbg !31
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !31
  %786 = insertelement <4 x float> %782, float %785, i64 3, !dbg !33
  %math.result64 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %786) #9, !dbg !33, !frisk.register_lifetimes !1
  %787 = fptrunc <4 x float> %math.result64 to <4 x half>, !dbg !33
  %math.packed65 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %787) #9, !dbg !32, !frisk.register_lifetimes !1
  %788 = fadd float %783, %785, !dbg !32
  %789 = extractelement <4 x float> %702, i64 0, !dbg !31
  %790 = call float @llvm.amdgcn.exp2.f32(float %789), !dbg !31
  %791 = insertelement <4 x float> poison, float %790, i64 0, !dbg !33
  %792 = extractelement <4 x float> %702, i64 1, !dbg !31
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !31
  %794 = insertelement <4 x float> %791, float %793, i64 1, !dbg !33
  %795 = fadd float %790, %793, !dbg !32
  %796 = extractelement <4 x float> %702, i64 2, !dbg !31
  %797 = call float @llvm.amdgcn.exp2.f32(float %796), !dbg !31
  %798 = insertelement <4 x float> %794, float %797, i64 2, !dbg !33
  %799 = fadd float %795, %797, !dbg !32
  %800 = extractelement <4 x float> %702, i64 3, !dbg !31
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !31
  %802 = insertelement <4 x float> %798, float %801, i64 3, !dbg !33
  %math.result66 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %802) #9, !dbg !33, !frisk.register_lifetimes !1
  %803 = fptrunc <4 x float> %math.result66 to <4 x half>, !dbg !33
  %math.packed67 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %803) #9, !dbg !32, !frisk.register_lifetimes !1
  %804 = fadd float %799, %801, !dbg !32
  %805 = extractelement <4 x float> %703, i64 0, !dbg !31
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !31
  %807 = insertelement <4 x float> poison, float %806, i64 0, !dbg !33
  %808 = fadd float %804, %806, !dbg !32
  %809 = extractelement <4 x float> %703, i64 1, !dbg !31
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !31
  %811 = insertelement <4 x float> %807, float %810, i64 1, !dbg !33
  %812 = fadd float %808, %810, !dbg !32
  %813 = extractelement <4 x float> %703, i64 2, !dbg !31
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !31
  %815 = insertelement <4 x float> %811, float %814, i64 2, !dbg !33
  %816 = fadd float %812, %814, !dbg !32
  %817 = extractelement <4 x float> %703, i64 3, !dbg !31
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !31
  %819 = insertelement <4 x float> %815, float %818, i64 3, !dbg !33
  %math.result68 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %819) #9, !dbg !33, !frisk.register_lifetimes !1
  %820 = fptrunc <4 x float> %math.result68 to <4 x half>, !dbg !33
  %math.packed69 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %820) #9, !dbg !32, !frisk.register_lifetimes !1
  %821 = fadd float %816, %818, !dbg !32
  %822 = extractelement <4 x float> %704, i64 0, !dbg !31
  %823 = call float @llvm.amdgcn.exp2.f32(float %822), !dbg !31
  %824 = insertelement <4 x float> poison, float %823, i64 0, !dbg !33
  %825 = fadd float %821, %823, !dbg !32
  %826 = extractelement <4 x float> %704, i64 1, !dbg !31
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !31
  %828 = insertelement <4 x float> %824, float %827, i64 1, !dbg !33
  %829 = fadd float %825, %827, !dbg !32
  %830 = extractelement <4 x float> %704, i64 2, !dbg !31
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !31
  %832 = insertelement <4 x float> %828, float %831, i64 2, !dbg !33
  %833 = fadd float %829, %831, !dbg !32
  %834 = extractelement <4 x float> %704, i64 3, !dbg !31
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !31
  %836 = insertelement <4 x float> %832, float %835, i64 3, !dbg !33
  %math.result70 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %836) #9, !dbg !33, !frisk.register_lifetimes !1
  %837 = fptrunc <4 x float> %math.result70 to <4 x half>, !dbg !33
  %math.packed71 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %837) #9, !dbg !32, !frisk.register_lifetimes !1
  %838 = fadd float %833, %835, !dbg !32
  %839 = extractelement <4 x float> %705, i64 0, !dbg !31
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !31
  %841 = insertelement <4 x float> poison, float %840, i64 0, !dbg !33
  %842 = fadd float %838, %840, !dbg !32
  %843 = extractelement <4 x float> %705, i64 1, !dbg !31
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !31
  %845 = insertelement <4 x float> %841, float %844, i64 1, !dbg !33
  %846 = fadd float %842, %844, !dbg !32
  %847 = extractelement <4 x float> %705, i64 2, !dbg !31
  %848 = call float @llvm.amdgcn.exp2.f32(float %847), !dbg !31
  %849 = insertelement <4 x float> %845, float %848, i64 2, !dbg !33
  %850 = fadd float %846, %848, !dbg !32
  %851 = extractelement <4 x float> %705, i64 3, !dbg !31
  %852 = call float @llvm.amdgcn.exp2.f32(float %851), !dbg !31
  %853 = insertelement <4 x float> %849, float %852, i64 3, !dbg !33
  %math.result72 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %853) #9, !dbg !33, !frisk.register_lifetimes !1
  %854 = fptrunc <4 x float> %math.result72 to <4 x half>, !dbg !33
  %math.packed73 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %854) #9, !dbg !32, !frisk.register_lifetimes !1
  %855 = fadd float %850, %852, !dbg !32
  %856 = bitcast float %788 to i32, !dbg !32
  %857 = call i32 @llvm.amdgcn.ds.bpermute(i32 %653, i32 %856), !dbg !32
  %858 = bitcast i32 %857 to float, !dbg !32
  %859 = fadd float %788, %858, !dbg !32
  %860 = bitcast float %859 to i32, !dbg !32
  %861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %860), !dbg !32
  %862 = bitcast i32 %861 to float, !dbg !32
  %863 = fadd float %859, %862, !dbg !32
  %864 = fadd float %863, %713, !dbg !32
  %865 = insertelement <1 x float> poison, float %864, i64 0, !dbg !7
  %866 = insertvalue [2 x <1 x float>] poison, <1 x float> %865, 0, !dbg !7
  %867 = bitcast float %855 to i32, !dbg !32
  %868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %653, i32 %867), !dbg !32
  %869 = bitcast i32 %868 to float, !dbg !32
  %870 = fadd float %855, %869, !dbg !32
  %871 = bitcast float %870 to i32, !dbg !32
  %872 = call i32 @llvm.amdgcn.ds.bpermute(i32 %661, i32 %871), !dbg !32
  %873 = bitcast i32 %872 to float, !dbg !32
  %874 = fadd float %870, %873, !dbg !32
  %875 = fadd float %874, %721, !dbg !32
  %876 = insertelement <1 x float> poison, float %875, i64 0, !dbg !7
  %877 = insertvalue [2 x <1 x float>] %866, <1 x float> %876, 1, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread201 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local202 = zext nneg i32 %address.thread201 to i64, !dbg !12
  %address.local203 = and i64 %address.local202, 60, !dbg !14
  %address.local204 = add nuw nsw i64 %address.local203, %address.local202, !dbg !14
  %address.local205 = shl nuw nsw i64 %address.local204, 3, !dbg !14
  %address.local206 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local205, !dbg !14
  %address.local207 = and i64 %address.local202, 63, !dbg !12
  %address.local208 = lshr i64 %address.local207, 4, !dbg !14
  %address.local209 = mul nuw nsw i64 %address.local208, 384, !dbg !14
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local206, i64 %address.local209, !dbg !14
  %address.local211 = lshr i64 %address.local202, 2, !dbg !16
  %address.local212 = mul nsw i64 %address.local211, -32, !dbg !14
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212, !dbg !14
  %878 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local213, i32 0) #0, !dbg !14
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !14
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local205, !dbg !14
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local209, !dbg !14
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local215, i64 %address.local212, !dbg !14
  %address.local217 = getelementptr i8, ptr addrspace(3) %address.local216, i64 256, !dbg !14
  %880 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local217, i32 0) #0, !dbg !14
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !14
  %address.local218 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local205, !dbg !14
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) %address.local218, i64 %address.local209, !dbg !14
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local219, i64 %address.local212, !dbg !14
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 512, !dbg !14
  %882 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local221, i32 0) #0, !dbg !14
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !14
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local205, !dbg !14
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local209, !dbg !14
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local212, !dbg !14
  %address.local225 = getelementptr i8, ptr addrspace(3) %address.local224, i64 768, !dbg !14
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local225, i32 0) #0, !dbg !14
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !14
  %886 = add nuw nsw i64 %94, %15, !dbg !17
  %887 = shl i64 %886, 19, !dbg !17
  %888 = and i64 %887, 16252928, !dbg !17
  %889 = add i64 %888, %100, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %890 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %889, !dbg !17
  %891 = getelementptr i8, ptr addrspace(1) %890, i64 64, !dbg !17
  %address.local226 = shl nuw nsw i64 %address.local202, 3, !dbg !11
  %address.local227 = and i64 %address.local226, 1536, !dbg !15
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local227, !dbg !17
  %address.local229 = ptrtoint ptr addrspace(3) %address.local228 to i64, !dbg !17
  %address.local230 = trunc i64 %address.local229 to i32, !dbg !17
  %892 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local230), !dbg !17
  %893 = zext i32 %892 to i64, !dbg !17
  %894 = inttoptr i64 %893 to ptr addrspace(3), !dbg !17
  %address.local231 = shl nuw nsw i64 %address.local202, 5, !dbg !17
  %address.local232 = and i64 %address.local231, 8064, !dbg !17
  %address.local233 = shl nuw nsw i64 %address.local202, 3, !dbg !17
  %address.local234 = and i64 %address.local233, 24, !dbg !17
  %address.local235 = or disjoint i64 %address.local232, %address.local234, !dbg !17
  %address.local236 = getelementptr [2 x i8], ptr addrspace(1) %891, i64 %address.local235, !dbg !17
  %895 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %896 = ptrtoint ptr addrspace(1) %address.local236 to i64, !dbg !17
  %897 = sub i64 %896, %895, !dbg !17
  %dma.buffer.offset593 = trunc i64 %897 to i32, !dbg !17
  %898 = trunc i64 %895 to i32, !dbg !17
  %899 = insertelement <4 x i32> poison, i32 %898, i64 0, !dbg !17
  %900 = lshr i64 %895, 32, !dbg !17
  %901 = trunc i64 %900 to i32, !dbg !17
  %902 = insertelement <4 x i32> %899, i32 %901, i64 1, !dbg !17
  %903 = insertelement <4 x i32> %902, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc594 = insertelement <4 x i32> %903, i32 131072, i64 3, !dbg !17
  %904 = ptrtoint ptr addrspace(3) %894 to i32, !dbg !17
  %dma.buffer.lds595 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %904), !dbg !17
  %905 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds595, i32 %dma.buffer.offset593, <4 x i32> %dma.buffer.rsrc594) #10, !dbg !17
  %906 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %907 = extractvalue [2 x <8 x float>] %55, 0, !dbg !34
  %908 = shufflevector <8 x float> %907, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %909 = fmul <4 x float> %908, %710, !dbg !34
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %906, <4 x float> %909) #9, !dbg !14
  %911 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %911, <4 x float> %910) #9, !dbg !14
  %913 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %913, <4 x float> %912) #9, !dbg !14
  %915 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %915, <4 x float> %914) #9, !dbg !14
  %917 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %918 = shufflevector <8 x float> %907, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %919 = fmul <4 x float> %918, %710, !dbg !34
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %917, <4 x float> %919) #9, !dbg !14
  %921 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %921, <4 x float> %920) #9, !dbg !14
  %923 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %923, <4 x float> %922) #9, !dbg !14
  %925 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %925, <4 x float> %924) #9, !dbg !14
  %927 = extractvalue [2 x <8 x float>] %55, 1, !dbg !34
  %928 = shufflevector <8 x float> %927, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %929 = fmul <4 x float> %928, %718, !dbg !34
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %906, <4 x float> %929) #9, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %911, <4 x float> %930) #9, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %913, <4 x float> %931) #9, !dbg !14
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %915, <4 x float> %932) #9, !dbg !14
  %934 = shufflevector <8 x float> %927, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %935 = fmul <4 x float> %934, %718, !dbg !34
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %917, <4 x float> %935) #9, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %921, <4 x float> %936) #9, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %923, <4 x float> %937) #9, !dbg !14
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %925, <4 x float> %938) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local213, i32 0) #0, !dbg !14
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !14
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local217, i32 0) #0, !dbg !14
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !14
  %944 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local221, i32 0) #0, !dbg !14
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !14
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local225, i32 0) #0, !dbg !14
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %948 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %889, !dbg !17
  %949 = getelementptr i8, ptr addrspace(1) %948, i64 128, !dbg !17
  %950 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local230), !dbg !17
  %951 = zext i32 %950 to i64, !dbg !17
  %952 = inttoptr i64 %951 to ptr addrspace(3), !dbg !17
  %address.local237 = getelementptr [2 x i8], ptr addrspace(1) %949, i64 %address.local235, !dbg !17
  %953 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %954 = ptrtoint ptr addrspace(1) %address.local237 to i64, !dbg !17
  %955 = sub i64 %954, %953, !dbg !17
  %dma.buffer.offset590 = trunc i64 %955 to i32, !dbg !17
  %956 = trunc i64 %953 to i32, !dbg !17
  %957 = insertelement <4 x i32> poison, i32 %956, i64 0, !dbg !17
  %958 = lshr i64 %953, 32, !dbg !17
  %959 = trunc i64 %958 to i32, !dbg !17
  %960 = insertelement <4 x i32> %957, i32 %959, i64 1, !dbg !17
  %961 = insertelement <4 x i32> %960, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc591 = insertelement <4 x i32> %961, i32 131072, i64 3, !dbg !17
  %962 = ptrtoint ptr addrspace(3) %952 to i32, !dbg !17
  %dma.buffer.lds592 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %962), !dbg !17
  %963 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds592, i32 %dma.buffer.offset590, <4 x i32> %dma.buffer.rsrc591) #10, !dbg !17
  %964 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %965 = extractvalue [2 x <8 x float>] %58, 0, !dbg !34
  %966 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %967 = fmul <4 x float> %966, %710, !dbg !34
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %964, <4 x float> %967) #9, !dbg !14
  %969 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %969, <4 x float> %968) #9, !dbg !14
  %971 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %971, <4 x float> %970) #9, !dbg !14
  %973 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %973, <4 x float> %972) #9, !dbg !14
  %975 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %976 = shufflevector <8 x float> %965, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %977 = fmul <4 x float> %976, %710, !dbg !34
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %975, <4 x float> %977) #9, !dbg !14
  %979 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %979, <4 x float> %978) #9, !dbg !14
  %981 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %981, <4 x float> %980) #9, !dbg !14
  %983 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %983, <4 x float> %982) #9, !dbg !14
  %985 = extractvalue [2 x <8 x float>] %58, 1, !dbg !34
  %986 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %987 = fmul <4 x float> %986, %718, !dbg !34
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %964, <4 x float> %987) #9, !dbg !14
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %969, <4 x float> %988) #9, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %971, <4 x float> %989) #9, !dbg !14
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %973, <4 x float> %990) #9, !dbg !14
  %992 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %993 = fmul <4 x float> %992, %718, !dbg !34
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %975, <4 x float> %993) #9, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %979, <4 x float> %994) #9, !dbg !14
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %981, <4 x float> %995) #9, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %983, <4 x float> %996) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %998 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local213, i32 0) #0, !dbg !14
  %999 = bitcast <4 x i32> %998 to <8 x half>, !dbg !14
  %1000 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local217, i32 0) #0, !dbg !14
  %1001 = bitcast <4 x i32> %1000 to <8 x half>, !dbg !14
  %1002 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local221, i32 0) #0, !dbg !14
  %1003 = bitcast <4 x i32> %1002 to <8 x half>, !dbg !14
  %1004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local225, i32 0) #0, !dbg !14
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1006 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %889, !dbg !17
  %1007 = getelementptr i8, ptr addrspace(1) %1006, i64 192, !dbg !17
  %1008 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local230), !dbg !17
  %1009 = zext i32 %1008 to i64, !dbg !17
  %1010 = inttoptr i64 %1009 to ptr addrspace(3), !dbg !17
  %address.local238 = getelementptr [2 x i8], ptr addrspace(1) %1007, i64 %address.local235, !dbg !17
  %1011 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1012 = ptrtoint ptr addrspace(1) %address.local238 to i64, !dbg !17
  %1013 = sub i64 %1012, %1011, !dbg !17
  %dma.buffer.offset = trunc i64 %1013 to i32, !dbg !17
  %1014 = trunc i64 %1011 to i32, !dbg !17
  %1015 = insertelement <4 x i32> poison, i32 %1014, i64 0, !dbg !17
  %1016 = lshr i64 %1011, 32, !dbg !17
  %1017 = trunc i64 %1016 to i32, !dbg !17
  %1018 = insertelement <4 x i32> %1015, i32 %1017, i64 1, !dbg !17
  %1019 = insertelement <4 x i32> %1018, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1019, i32 131072, i64 3, !dbg !17
  %1020 = ptrtoint ptr addrspace(3) %1010 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1020), !dbg !17
  %1021 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1022 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1023 = extractvalue [2 x <8 x float>] %59, 0, !dbg !34
  %1024 = shufflevector <8 x float> %1023, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1025 = fmul <4 x float> %1024, %710, !dbg !34
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1022, <4 x float> %1025) #9, !dbg !14
  %1027 = shufflevector <8 x half> %1001, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %1027, <4 x float> %1026) #9, !dbg !14
  %1029 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1029, <4 x float> %1028) #9, !dbg !14
  %1031 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1031, <4 x float> %1030) #9, !dbg !14
  %1033 = shufflevector <8 x half> %999, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1034 = shufflevector <8 x float> %1023, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1035 = fmul <4 x float> %1034, %710, !dbg !34
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1033, <4 x float> %1035) #9, !dbg !14
  %1037 = shufflevector <8 x half> %1001, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %1037, <4 x float> %1036) #9, !dbg !14
  %1039 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1039, <4 x float> %1038) #9, !dbg !14
  %1041 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1041, <4 x float> %1040) #9, !dbg !14
  %1043 = extractvalue [2 x <8 x float>] %59, 1, !dbg !34
  %1044 = shufflevector <8 x float> %1043, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1045 = fmul <4 x float> %1044, %718, !dbg !34
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1022, <4 x float> %1045) #9, !dbg !14
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1027, <4 x float> %1046) #9, !dbg !14
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1029, <4 x float> %1047) #9, !dbg !14
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1031, <4 x float> %1048) #9, !dbg !14
  %1050 = shufflevector <8 x float> %1043, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1051 = fmul <4 x float> %1050, %718, !dbg !34
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1033, <4 x float> %1051) #9, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1037, <4 x float> %1052) #9, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1039, <4 x float> %1053) #9, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1041, <4 x float> %1054) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1056 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local213, i32 0) #0, !dbg !14
  %1057 = bitcast <4 x i32> %1056 to <8 x half>, !dbg !14
  %1058 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local217, i32 0) #0, !dbg !14
  %1059 = bitcast <4 x i32> %1058 to <8 x half>, !dbg !14
  %1060 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local221, i32 0) #0, !dbg !14
  %1061 = bitcast <4 x i32> %1060 to <8 x half>, !dbg !14
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local225, i32 0) #0, !dbg !14
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !14
  %1064 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1065 = extractvalue [2 x <8 x float>] %60, 0, !dbg !34
  %1066 = shufflevector <8 x float> %1065, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1067 = fmul <4 x float> %1066, %710, !dbg !34
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1067) #9, !dbg !14
  %1069 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1070 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %1069, <4 x float> %1068) #9, !dbg !14
  %1071 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1072 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1071, <4 x float> %1070) #9, !dbg !14
  %1073 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1073, <4 x float> %1072) #9, !dbg !14
  %1075 = shufflevector <8 x half> %1057, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1076 = shufflevector <8 x float> %1065, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1077 = fmul <4 x float> %1076, %710, !dbg !34
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1075, <4 x float> %1077) #9, !dbg !14
  %1079 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1080 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed61, <4 x half> %1079, <4 x float> %1078) #9, !dbg !14
  %1081 = shufflevector <8 x half> %1061, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1082 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed63, <4 x half> %1081, <4 x float> %1080) #9, !dbg !14
  %1083 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1084 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed65, <4 x half> %1083, <4 x float> %1082) #9, !dbg !14
  %1085 = extractvalue [2 x <8 x float>] %60, 1, !dbg !34
  %1086 = shufflevector <8 x float> %1085, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !34
  %1087 = fmul <4 x float> %1086, %718, !dbg !34
  %1088 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1064, <4 x float> %1087) #9, !dbg !14
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1069, <4 x float> %1088) #9, !dbg !14
  %1090 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1071, <4 x float> %1089) #9, !dbg !14
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1073, <4 x float> %1090) #9, !dbg !14
  %1092 = shufflevector <8 x float> %1085, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !34
  %1093 = fmul <4 x float> %1092, %718, !dbg !34
  %1094 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1075, <4 x float> %1093) #9, !dbg !14
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1079, <4 x float> %1094) #9, !dbg !14
  %1096 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1081, <4 x float> %1095) #9, !dbg !14
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1083, <4 x float> %1096) #9, !dbg !14
  %1098 = shufflevector <4 x float> %916, <4 x float> %926, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1099 = insertvalue [2 x <8 x float>] poison, <8 x float> %1098, 0, !dbg !7
  %1100 = shufflevector <4 x float> %933, <4 x float> %939, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1101 = insertvalue [2 x <8 x float>] %1099, <8 x float> %1100, 1, !dbg !7
  %1102 = insertvalue [2 x <1 x float>] poison, <1 x float> %668, 0, !dbg !7
  %1103 = insertvalue [2 x <1 x float>] %1102, <1 x float> %695, 1, !dbg !7
  %1104 = shufflevector <4 x float> %974, <4 x float> %984, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1105 = insertvalue [2 x <8 x float>] poison, <8 x float> %1104, 0, !dbg !7
  %1106 = shufflevector <4 x float> %991, <4 x float> %997, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1107 = insertvalue [2 x <8 x float>] %1105, <8 x float> %1106, 1, !dbg !7
  %1108 = shufflevector <4 x float> %1032, <4 x float> %1042, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1109 = insertvalue [2 x <8 x float>] poison, <8 x float> %1108, 0, !dbg !7
  %1110 = shufflevector <4 x float> %1049, <4 x float> %1055, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1111 = insertvalue [2 x <8 x float>] %1109, <8 x float> %1110, 1, !dbg !7
  %1112 = shufflevector <4 x float> %1074, <4 x float> %1084, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1113 = insertvalue [2 x <8 x float>] poison, <8 x float> %1112, 0, !dbg !7
  %1114 = shufflevector <4 x float> %1091, <4 x float> %1097, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1115 = insertvalue [2 x <8 x float>] %1113, <8 x float> %1114, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1116 = add nuw nsw i64 %54, 1, !dbg !7
  br label %53, !dbg !7

1117:                                             ; preds = %53
  %1118 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %1118, 0, !dbg !7
  %1119 = and i32 %34, 32, !dbg !7
  %.not1 = icmp eq i32 %1119, 0, !dbg !7
  %1120 = extractvalue [2 x <8 x float>] %60, 0, !dbg !7
  %1121 = extractelement <8 x float> %1120, i64 0, !dbg !7
  %1122 = extractelement <8 x float> %1120, i64 1, !dbg !7
  %1123 = extractelement <8 x float> %1120, i64 2, !dbg !7
  %1124 = extractelement <8 x float> %1120, i64 3, !dbg !7
  %1125 = select i1 %.not, float %1122, float %1121, !dbg !7
  %1126 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1127 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1126), !dbg !7
  %1128 = add nuw nsw i32 %1127, 64, !dbg !7
  %1129 = xor i32 %1127, 16, !dbg !7
  %1130 = and i32 %1128, 192, !dbg !7
  %1131 = icmp samesign ult i32 %1129, %1130, !dbg !7
  %1132 = select i1 %1131, i32 %1129, i32 %1127, !dbg !7
  %1133 = shl nuw nsw i32 %1132, 2, !dbg !7
  %1134 = bitcast float %1125 to i32, !dbg !7
  %1135 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1134), !dbg !7
  %1136 = bitcast i32 %1135 to float, !dbg !7
  %1137 = select i1 %.not, float %1121, float %1136, !dbg !7
  %1138 = select i1 %.not, float %1136, float %1122, !dbg !7
  %1139 = select i1 %.not, float %1124, float %1123, !dbg !7
  %1140 = bitcast float %1139 to i32, !dbg !7
  %1141 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1140), !dbg !7
  %1142 = bitcast i32 %1141 to float, !dbg !7
  %1143 = select i1 %.not, float %1123, float %1142, !dbg !7
  %1144 = select i1 %.not, float %1142, float %1124, !dbg !7
  %1145 = select i1 %.not1, float %1143, float %1137, !dbg !7
  %1146 = xor i32 %1127, 32, !dbg !7
  %1147 = icmp samesign ult i32 %1146, %1130, !dbg !7
  %1148 = select i1 %1147, i32 %1146, i32 %1127, !dbg !7
  %1149 = shl nuw nsw i32 %1148, 2, !dbg !7
  %1150 = bitcast float %1145 to i32, !dbg !7
  %1151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1150), !dbg !7
  %1152 = bitcast i32 %1151 to float, !dbg !7
  %1153 = select i1 %.not1, float %1137, float %1152, !dbg !7
  %1154 = select i1 %.not1, float %1152, float %1143, !dbg !7
  %1155 = select i1 %.not1, float %1144, float %1138, !dbg !7
  %1156 = bitcast float %1155 to i32, !dbg !7
  %1157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1156), !dbg !7
  %1158 = bitcast i32 %1157 to float, !dbg !7
  %1159 = select i1 %.not1, float %1138, float %1158, !dbg !7
  %1160 = select i1 %.not1, float %1158, float %1144, !dbg !7
  %1161 = extractvalue [2 x <8 x float>] %59, 0, !dbg !7
  %1162 = extractelement <8 x float> %1161, i64 0, !dbg !7
  %1163 = extractelement <8 x float> %1161, i64 1, !dbg !7
  %1164 = extractelement <8 x float> %1161, i64 2, !dbg !7
  %1165 = extractelement <8 x float> %1161, i64 3, !dbg !7
  %1166 = select i1 %.not, float %1163, float %1162, !dbg !7
  %1167 = bitcast float %1166 to i32, !dbg !7
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1167), !dbg !7
  %1169 = bitcast i32 %1168 to float, !dbg !7
  %1170 = select i1 %.not, float %1162, float %1169, !dbg !7
  %1171 = select i1 %.not, float %1169, float %1163, !dbg !7
  %1172 = select i1 %.not, float %1165, float %1164, !dbg !7
  %1173 = bitcast float %1172 to i32, !dbg !7
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1173), !dbg !7
  %1175 = bitcast i32 %1174 to float, !dbg !7
  %1176 = select i1 %.not, float %1164, float %1175, !dbg !7
  %1177 = select i1 %.not, float %1175, float %1165, !dbg !7
  %1178 = select i1 %.not1, float %1176, float %1170, !dbg !7
  %1179 = bitcast float %1178 to i32, !dbg !7
  %1180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1179), !dbg !7
  %1181 = bitcast i32 %1180 to float, !dbg !7
  %1182 = select i1 %.not1, float %1170, float %1181, !dbg !7
  %1183 = select i1 %.not1, float %1181, float %1176, !dbg !7
  %1184 = select i1 %.not1, float %1177, float %1171, !dbg !7
  %1185 = bitcast float %1184 to i32, !dbg !7
  %1186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1185), !dbg !7
  %1187 = bitcast i32 %1186 to float, !dbg !7
  %1188 = select i1 %.not1, float %1171, float %1187, !dbg !7
  %1189 = select i1 %.not1, float %1187, float %1177, !dbg !7
  %1190 = extractvalue [2 x <8 x float>] %58, 0, !dbg !7
  %1191 = extractelement <8 x float> %1190, i64 0, !dbg !7
  %1192 = extractelement <8 x float> %1190, i64 1, !dbg !7
  %1193 = extractelement <8 x float> %1190, i64 2, !dbg !7
  %1194 = extractelement <8 x float> %1190, i64 3, !dbg !7
  %1195 = select i1 %.not, float %1192, float %1191, !dbg !7
  %1196 = bitcast float %1195 to i32, !dbg !7
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1196), !dbg !7
  %1198 = bitcast i32 %1197 to float, !dbg !7
  %1199 = select i1 %.not, float %1191, float %1198, !dbg !7
  %1200 = select i1 %.not, float %1198, float %1192, !dbg !7
  %1201 = select i1 %.not, float %1194, float %1193, !dbg !7
  %1202 = bitcast float %1201 to i32, !dbg !7
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1202), !dbg !7
  %1204 = bitcast i32 %1203 to float, !dbg !7
  %1205 = select i1 %.not, float %1193, float %1204, !dbg !7
  %1206 = select i1 %.not, float %1204, float %1194, !dbg !7
  %1207 = select i1 %.not1, float %1205, float %1199, !dbg !7
  %1208 = bitcast float %1207 to i32, !dbg !7
  %1209 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1208), !dbg !7
  %1210 = bitcast i32 %1209 to float, !dbg !7
  %1211 = select i1 %.not1, float %1199, float %1210, !dbg !7
  %1212 = select i1 %.not1, float %1210, float %1205, !dbg !7
  %1213 = select i1 %.not1, float %1206, float %1200, !dbg !7
  %1214 = bitcast float %1213 to i32, !dbg !7
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1214), !dbg !7
  %1216 = bitcast i32 %1215 to float, !dbg !7
  %1217 = select i1 %.not1, float %1200, float %1216, !dbg !7
  %1218 = select i1 %.not1, float %1216, float %1206, !dbg !7
  %1219 = extractvalue [2 x <8 x float>] %55, 0, !dbg !7
  %1220 = extractelement <8 x float> %1219, i64 0, !dbg !7
  %1221 = extractelement <8 x float> %1219, i64 1, !dbg !7
  %1222 = extractelement <8 x float> %1219, i64 2, !dbg !7
  %1223 = extractelement <8 x float> %1219, i64 3, !dbg !7
  %1224 = select i1 %.not, float %1221, float %1220, !dbg !7
  %1225 = bitcast float %1224 to i32, !dbg !7
  %1226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1225), !dbg !7
  %1227 = bitcast i32 %1226 to float, !dbg !7
  %1228 = select i1 %.not, float %1220, float %1227, !dbg !7
  %1229 = select i1 %.not, float %1227, float %1221, !dbg !7
  %1230 = select i1 %.not, float %1223, float %1222, !dbg !7
  %1231 = bitcast float %1230 to i32, !dbg !7
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1231), !dbg !7
  %1233 = bitcast i32 %1232 to float, !dbg !7
  %1234 = select i1 %.not, float %1222, float %1233, !dbg !7
  %1235 = select i1 %.not, float %1233, float %1223, !dbg !7
  %1236 = select i1 %.not1, float %1234, float %1228, !dbg !7
  %1237 = bitcast float %1236 to i32, !dbg !7
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1237), !dbg !7
  %1239 = select i1 %.not1, float %1235, float %1229, !dbg !7
  %1240 = bitcast float %1239 to i32, !dbg !7
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1240), !dbg !7
  %1242 = extractvalue [2 x <1 x float>] %56, 0, !dbg !18
  %1243 = shufflevector <1 x float> %1242, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1244 = extractelement <8 x float> %1120, i64 4, !dbg !7
  %1245 = extractelement <8 x float> %1120, i64 5, !dbg !7
  %1246 = extractelement <8 x float> %1120, i64 6, !dbg !7
  %1247 = extractelement <8 x float> %1120, i64 7, !dbg !7
  %1248 = select i1 %.not, float %1245, float %1244, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1249), !dbg !7
  %1251 = bitcast i32 %1250 to float, !dbg !7
  %1252 = select i1 %.not, float %1244, float %1251, !dbg !7
  %1253 = select i1 %.not, float %1251, float %1245, !dbg !7
  %1254 = select i1 %.not, float %1247, float %1246, !dbg !7
  %1255 = bitcast float %1254 to i32, !dbg !7
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1255), !dbg !7
  %1257 = bitcast i32 %1256 to float, !dbg !7
  %1258 = select i1 %.not, float %1246, float %1257, !dbg !7
  %1259 = select i1 %.not, float %1257, float %1247, !dbg !7
  %1260 = select i1 %.not1, float %1258, float %1252, !dbg !7
  %1261 = bitcast float %1260 to i32, !dbg !7
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1261), !dbg !7
  %1263 = bitcast i32 %1262 to float, !dbg !7
  %1264 = select i1 %.not1, float %1252, float %1263, !dbg !7
  %1265 = select i1 %.not1, float %1263, float %1258, !dbg !7
  %1266 = select i1 %.not1, float %1259, float %1253, !dbg !7
  %1267 = bitcast float %1266 to i32, !dbg !7
  %1268 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1267), !dbg !7
  %1269 = bitcast i32 %1268 to float, !dbg !7
  %1270 = select i1 %.not1, float %1253, float %1269, !dbg !7
  %1271 = select i1 %.not1, float %1269, float %1259, !dbg !7
  %1272 = extractelement <8 x float> %1161, i64 4, !dbg !7
  %1273 = extractelement <8 x float> %1161, i64 5, !dbg !7
  %1274 = extractelement <8 x float> %1161, i64 6, !dbg !7
  %1275 = extractelement <8 x float> %1161, i64 7, !dbg !7
  %1276 = select i1 %.not, float %1273, float %1272, !dbg !7
  %1277 = bitcast float %1276 to i32, !dbg !7
  %1278 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1277), !dbg !7
  %1279 = bitcast i32 %1278 to float, !dbg !7
  %1280 = select i1 %.not, float %1272, float %1279, !dbg !7
  %1281 = select i1 %.not, float %1279, float %1273, !dbg !7
  %1282 = select i1 %.not, float %1275, float %1274, !dbg !7
  %1283 = bitcast float %1282 to i32, !dbg !7
  %1284 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1283), !dbg !7
  %1285 = bitcast i32 %1284 to float, !dbg !7
  %1286 = select i1 %.not, float %1274, float %1285, !dbg !7
  %1287 = select i1 %.not, float %1285, float %1275, !dbg !7
  %1288 = select i1 %.not1, float %1286, float %1280, !dbg !7
  %1289 = bitcast float %1288 to i32, !dbg !7
  %1290 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1289), !dbg !7
  %1291 = bitcast i32 %1290 to float, !dbg !7
  %1292 = select i1 %.not1, float %1280, float %1291, !dbg !7
  %1293 = select i1 %.not1, float %1291, float %1286, !dbg !7
  %1294 = select i1 %.not1, float %1287, float %1281, !dbg !7
  %1295 = bitcast float %1294 to i32, !dbg !7
  %1296 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1295), !dbg !7
  %1297 = bitcast i32 %1296 to float, !dbg !7
  %1298 = select i1 %.not1, float %1281, float %1297, !dbg !7
  %1299 = select i1 %.not1, float %1297, float %1287, !dbg !7
  %1300 = extractelement <8 x float> %1190, i64 4, !dbg !7
  %1301 = extractelement <8 x float> %1190, i64 5, !dbg !7
  %1302 = extractelement <8 x float> %1190, i64 6, !dbg !7
  %1303 = extractelement <8 x float> %1190, i64 7, !dbg !7
  %1304 = select i1 %.not, float %1301, float %1300, !dbg !7
  %1305 = bitcast float %1304 to i32, !dbg !7
  %1306 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1305), !dbg !7
  %1307 = bitcast i32 %1306 to float, !dbg !7
  %1308 = select i1 %.not, float %1300, float %1307, !dbg !7
  %1309 = select i1 %.not, float %1307, float %1301, !dbg !7
  %1310 = select i1 %.not, float %1303, float %1302, !dbg !7
  %1311 = bitcast float %1310 to i32, !dbg !7
  %1312 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1311), !dbg !7
  %1313 = bitcast i32 %1312 to float, !dbg !7
  %1314 = select i1 %.not, float %1302, float %1313, !dbg !7
  %1315 = select i1 %.not, float %1313, float %1303, !dbg !7
  %1316 = select i1 %.not1, float %1314, float %1308, !dbg !7
  %1317 = bitcast float %1316 to i32, !dbg !7
  %1318 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1317), !dbg !7
  %1319 = bitcast i32 %1318 to float, !dbg !7
  %1320 = select i1 %.not1, float %1308, float %1319, !dbg !7
  %1321 = select i1 %.not1, float %1319, float %1314, !dbg !7
  %1322 = select i1 %.not1, float %1315, float %1309, !dbg !7
  %1323 = bitcast float %1322 to i32, !dbg !7
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1323), !dbg !7
  %1325 = bitcast i32 %1324 to float, !dbg !7
  %1326 = select i1 %.not1, float %1309, float %1325, !dbg !7
  %1327 = select i1 %.not1, float %1325, float %1315, !dbg !7
  %1328 = extractelement <8 x float> %1219, i64 4, !dbg !7
  %1329 = extractelement <8 x float> %1219, i64 5, !dbg !7
  %1330 = extractelement <8 x float> %1219, i64 6, !dbg !7
  %1331 = extractelement <8 x float> %1219, i64 7, !dbg !7
  %1332 = select i1 %.not, float %1329, float %1328, !dbg !7
  %1333 = bitcast float %1332 to i32, !dbg !7
  %1334 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1333), !dbg !7
  %1335 = bitcast i32 %1334 to float, !dbg !7
  %1336 = select i1 %.not, float %1328, float %1335, !dbg !7
  %1337 = select i1 %.not, float %1335, float %1329, !dbg !7
  %1338 = select i1 %.not, float %1331, float %1330, !dbg !7
  %1339 = bitcast float %1338 to i32, !dbg !7
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1339), !dbg !7
  %1341 = bitcast i32 %1340 to float, !dbg !7
  %1342 = select i1 %.not, float %1330, float %1341, !dbg !7
  %1343 = select i1 %.not, float %1341, float %1331, !dbg !7
  %1344 = select i1 %.not1, float %1342, float %1336, !dbg !7
  %1345 = bitcast float %1344 to i32, !dbg !7
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1345), !dbg !7
  %1347 = select i1 %.not1, float %1343, float %1337, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1348), !dbg !7
  %1350 = extractvalue [2 x <8 x float>] %60, 1, !dbg !7
  %1351 = extractelement <8 x float> %1350, i64 0, !dbg !7
  %1352 = extractelement <8 x float> %1350, i64 1, !dbg !7
  %1353 = extractelement <8 x float> %1350, i64 2, !dbg !7
  %1354 = extractelement <8 x float> %1350, i64 3, !dbg !7
  %1355 = select i1 %.not, float %1352, float %1351, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1356), !dbg !7
  %1358 = bitcast i32 %1357 to float, !dbg !7
  %1359 = select i1 %.not, float %1351, float %1358, !dbg !7
  %1360 = select i1 %.not, float %1358, float %1352, !dbg !7
  %1361 = select i1 %.not, float %1354, float %1353, !dbg !7
  %1362 = bitcast float %1361 to i32, !dbg !7
  %1363 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1362), !dbg !7
  %1364 = bitcast i32 %1363 to float, !dbg !7
  %1365 = select i1 %.not, float %1353, float %1364, !dbg !7
  %1366 = select i1 %.not, float %1364, float %1354, !dbg !7
  %1367 = select i1 %.not1, float %1365, float %1359, !dbg !7
  %1368 = bitcast float %1367 to i32, !dbg !7
  %1369 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1368), !dbg !7
  %1370 = bitcast i32 %1369 to float, !dbg !7
  %1371 = select i1 %.not1, float %1359, float %1370, !dbg !7
  %1372 = select i1 %.not1, float %1370, float %1365, !dbg !7
  %1373 = select i1 %.not1, float %1366, float %1360, !dbg !7
  %1374 = bitcast float %1373 to i32, !dbg !7
  %1375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1374), !dbg !7
  %1376 = bitcast i32 %1375 to float, !dbg !7
  %1377 = select i1 %.not1, float %1360, float %1376, !dbg !7
  %1378 = select i1 %.not1, float %1376, float %1366, !dbg !7
  %1379 = extractvalue [2 x <8 x float>] %59, 1, !dbg !7
  %1380 = extractelement <8 x float> %1379, i64 0, !dbg !7
  %1381 = extractelement <8 x float> %1379, i64 1, !dbg !7
  %1382 = extractelement <8 x float> %1379, i64 2, !dbg !7
  %1383 = extractelement <8 x float> %1379, i64 3, !dbg !7
  %1384 = select i1 %.not, float %1381, float %1380, !dbg !7
  %1385 = bitcast float %1384 to i32, !dbg !7
  %1386 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1385), !dbg !7
  %1387 = bitcast i32 %1386 to float, !dbg !7
  %1388 = select i1 %.not, float %1380, float %1387, !dbg !7
  %1389 = select i1 %.not, float %1387, float %1381, !dbg !7
  %1390 = select i1 %.not, float %1383, float %1382, !dbg !7
  %1391 = bitcast float %1390 to i32, !dbg !7
  %1392 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1391), !dbg !7
  %1393 = bitcast i32 %1392 to float, !dbg !7
  %1394 = select i1 %.not, float %1382, float %1393, !dbg !7
  %1395 = select i1 %.not, float %1393, float %1383, !dbg !7
  %1396 = select i1 %.not1, float %1394, float %1388, !dbg !7
  %1397 = bitcast float %1396 to i32, !dbg !7
  %1398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1397), !dbg !7
  %1399 = bitcast i32 %1398 to float, !dbg !7
  %1400 = select i1 %.not1, float %1388, float %1399, !dbg !7
  %1401 = select i1 %.not1, float %1399, float %1394, !dbg !7
  %1402 = select i1 %.not1, float %1395, float %1389, !dbg !7
  %1403 = bitcast float %1402 to i32, !dbg !7
  %1404 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1403), !dbg !7
  %1405 = bitcast i32 %1404 to float, !dbg !7
  %1406 = select i1 %.not1, float %1389, float %1405, !dbg !7
  %1407 = select i1 %.not1, float %1405, float %1395, !dbg !7
  %1408 = extractvalue [2 x <8 x float>] %58, 1, !dbg !7
  %1409 = extractelement <8 x float> %1408, i64 0, !dbg !7
  %1410 = extractelement <8 x float> %1408, i64 1, !dbg !7
  %1411 = extractelement <8 x float> %1408, i64 2, !dbg !7
  %1412 = extractelement <8 x float> %1408, i64 3, !dbg !7
  %1413 = select i1 %.not, float %1410, float %1409, !dbg !7
  %1414 = bitcast float %1413 to i32, !dbg !7
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1414), !dbg !7
  %1416 = bitcast i32 %1415 to float, !dbg !7
  %1417 = select i1 %.not, float %1409, float %1416, !dbg !7
  %1418 = select i1 %.not, float %1416, float %1410, !dbg !7
  %1419 = select i1 %.not, float %1412, float %1411, !dbg !7
  %1420 = bitcast float %1419 to i32, !dbg !7
  %1421 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1420), !dbg !7
  %1422 = bitcast i32 %1421 to float, !dbg !7
  %1423 = select i1 %.not, float %1411, float %1422, !dbg !7
  %1424 = select i1 %.not, float %1422, float %1412, !dbg !7
  %1425 = select i1 %.not1, float %1423, float %1417, !dbg !7
  %1426 = bitcast float %1425 to i32, !dbg !7
  %1427 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1426), !dbg !7
  %1428 = bitcast i32 %1427 to float, !dbg !7
  %1429 = select i1 %.not1, float %1417, float %1428, !dbg !7
  %1430 = select i1 %.not1, float %1428, float %1423, !dbg !7
  %1431 = select i1 %.not1, float %1424, float %1418, !dbg !7
  %1432 = bitcast float %1431 to i32, !dbg !7
  %1433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1432), !dbg !7
  %1434 = bitcast i32 %1433 to float, !dbg !7
  %1435 = select i1 %.not1, float %1418, float %1434, !dbg !7
  %1436 = select i1 %.not1, float %1434, float %1424, !dbg !7
  %1437 = extractvalue [2 x <8 x float>] %55, 1, !dbg !7
  %1438 = extractelement <8 x float> %1437, i64 0, !dbg !7
  %1439 = extractelement <8 x float> %1437, i64 1, !dbg !7
  %1440 = extractelement <8 x float> %1437, i64 2, !dbg !7
  %1441 = extractelement <8 x float> %1437, i64 3, !dbg !7
  %1442 = select i1 %.not, float %1439, float %1438, !dbg !7
  %1443 = bitcast float %1442 to i32, !dbg !7
  %1444 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1443), !dbg !7
  %1445 = bitcast i32 %1444 to float, !dbg !7
  %1446 = select i1 %.not, float %1438, float %1445, !dbg !7
  %1447 = select i1 %.not, float %1445, float %1439, !dbg !7
  %1448 = select i1 %.not, float %1441, float %1440, !dbg !7
  %1449 = bitcast float %1448 to i32, !dbg !7
  %1450 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1449), !dbg !7
  %1451 = bitcast i32 %1450 to float, !dbg !7
  %1452 = select i1 %.not, float %1440, float %1451, !dbg !7
  %1453 = select i1 %.not, float %1451, float %1441, !dbg !7
  %1454 = select i1 %.not1, float %1452, float %1446, !dbg !7
  %1455 = bitcast float %1454 to i32, !dbg !7
  %1456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1455), !dbg !7
  %1457 = select i1 %.not1, float %1453, float %1447, !dbg !7
  %1458 = bitcast float %1457 to i32, !dbg !7
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1458), !dbg !7
  %1460 = extractvalue [2 x <1 x float>] %56, 1, !dbg !18
  %1461 = shufflevector <1 x float> %1460, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1462 = extractelement <8 x float> %1350, i64 4, !dbg !7
  %1463 = extractelement <8 x float> %1350, i64 5, !dbg !7
  %1464 = extractelement <8 x float> %1350, i64 6, !dbg !7
  %1465 = extractelement <8 x float> %1350, i64 7, !dbg !7
  %1466 = select i1 %.not, float %1463, float %1462, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1467), !dbg !7
  %1469 = bitcast i32 %1468 to float, !dbg !7
  %1470 = select i1 %.not, float %1462, float %1469, !dbg !7
  %1471 = select i1 %.not, float %1469, float %1463, !dbg !7
  %1472 = select i1 %.not, float %1465, float %1464, !dbg !7
  %1473 = bitcast float %1472 to i32, !dbg !7
  %1474 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1473), !dbg !7
  %1475 = bitcast i32 %1474 to float, !dbg !7
  %1476 = select i1 %.not, float %1464, float %1475, !dbg !7
  %1477 = select i1 %.not, float %1475, float %1465, !dbg !7
  %1478 = select i1 %.not1, float %1476, float %1470, !dbg !7
  %1479 = bitcast float %1478 to i32, !dbg !7
  %1480 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1479), !dbg !7
  %1481 = bitcast i32 %1480 to float, !dbg !7
  %1482 = select i1 %.not1, float %1470, float %1481, !dbg !7
  %1483 = select i1 %.not1, float %1481, float %1476, !dbg !7
  %1484 = select i1 %.not1, float %1477, float %1471, !dbg !7
  %1485 = bitcast float %1484 to i32, !dbg !7
  %1486 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1485), !dbg !7
  %1487 = bitcast i32 %1486 to float, !dbg !7
  %1488 = select i1 %.not1, float %1471, float %1487, !dbg !7
  %1489 = select i1 %.not1, float %1487, float %1477, !dbg !7
  %1490 = extractelement <8 x float> %1379, i64 4, !dbg !7
  %1491 = extractelement <8 x float> %1379, i64 5, !dbg !7
  %1492 = extractelement <8 x float> %1379, i64 6, !dbg !7
  %1493 = extractelement <8 x float> %1379, i64 7, !dbg !7
  %1494 = select i1 %.not, float %1491, float %1490, !dbg !7
  %1495 = bitcast float %1494 to i32, !dbg !7
  %1496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1495), !dbg !7
  %1497 = bitcast i32 %1496 to float, !dbg !7
  %1498 = select i1 %.not, float %1490, float %1497, !dbg !7
  %1499 = select i1 %.not, float %1497, float %1491, !dbg !7
  %1500 = select i1 %.not, float %1493, float %1492, !dbg !7
  %1501 = bitcast float %1500 to i32, !dbg !7
  %1502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1501), !dbg !7
  %1503 = bitcast i32 %1502 to float, !dbg !7
  %1504 = select i1 %.not, float %1492, float %1503, !dbg !7
  %1505 = select i1 %.not, float %1503, float %1493, !dbg !7
  %1506 = select i1 %.not1, float %1504, float %1498, !dbg !7
  %1507 = bitcast float %1506 to i32, !dbg !7
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1507), !dbg !7
  %1509 = bitcast i32 %1508 to float, !dbg !7
  %1510 = select i1 %.not1, float %1498, float %1509, !dbg !7
  %1511 = select i1 %.not1, float %1509, float %1504, !dbg !7
  %1512 = select i1 %.not1, float %1505, float %1499, !dbg !7
  %1513 = bitcast float %1512 to i32, !dbg !7
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1513), !dbg !7
  %1515 = bitcast i32 %1514 to float, !dbg !7
  %1516 = select i1 %.not1, float %1499, float %1515, !dbg !7
  %1517 = select i1 %.not1, float %1515, float %1505, !dbg !7
  %1518 = extractelement <8 x float> %1408, i64 4, !dbg !7
  %1519 = extractelement <8 x float> %1408, i64 5, !dbg !7
  %1520 = extractelement <8 x float> %1408, i64 6, !dbg !7
  %1521 = extractelement <8 x float> %1408, i64 7, !dbg !7
  %1522 = select i1 %.not, float %1519, float %1518, !dbg !7
  %1523 = bitcast float %1522 to i32, !dbg !7
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1523), !dbg !7
  %1525 = bitcast i32 %1524 to float, !dbg !7
  %1526 = select i1 %.not, float %1518, float %1525, !dbg !7
  %1527 = select i1 %.not, float %1525, float %1519, !dbg !7
  %1528 = select i1 %.not, float %1521, float %1520, !dbg !7
  %1529 = bitcast float %1528 to i32, !dbg !7
  %1530 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1529), !dbg !7
  %1531 = bitcast i32 %1530 to float, !dbg !7
  %1532 = select i1 %.not, float %1520, float %1531, !dbg !7
  %1533 = select i1 %.not, float %1531, float %1521, !dbg !7
  %1534 = select i1 %.not1, float %1532, float %1526, !dbg !7
  %1535 = bitcast float %1534 to i32, !dbg !7
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1535), !dbg !7
  %1537 = bitcast i32 %1536 to float, !dbg !7
  %1538 = select i1 %.not1, float %1526, float %1537, !dbg !7
  %1539 = select i1 %.not1, float %1537, float %1532, !dbg !7
  %1540 = select i1 %.not1, float %1533, float %1527, !dbg !7
  %1541 = bitcast float %1540 to i32, !dbg !7
  %1542 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1541), !dbg !7
  %1543 = bitcast i32 %1542 to float, !dbg !7
  %1544 = select i1 %.not1, float %1527, float %1543, !dbg !7
  %1545 = select i1 %.not1, float %1543, float %1533, !dbg !7
  %1546 = extractelement <8 x float> %1437, i64 4, !dbg !7
  %1547 = extractelement <8 x float> %1437, i64 5, !dbg !7
  %1548 = extractelement <8 x float> %1437, i64 6, !dbg !7
  %1549 = extractelement <8 x float> %1437, i64 7, !dbg !7
  %1550 = select i1 %.not, float %1547, float %1546, !dbg !7
  %1551 = bitcast float %1550 to i32, !dbg !7
  %1552 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1551), !dbg !7
  %1553 = bitcast i32 %1552 to float, !dbg !7
  %1554 = select i1 %.not, float %1546, float %1553, !dbg !7
  %1555 = select i1 %.not, float %1553, float %1547, !dbg !7
  %1556 = select i1 %.not, float %1549, float %1548, !dbg !7
  %1557 = bitcast float %1556 to i32, !dbg !7
  %1558 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1133, i32 %1557), !dbg !7
  %1559 = bitcast i32 %1558 to float, !dbg !7
  %1560 = select i1 %.not, float %1548, float %1559, !dbg !7
  %1561 = select i1 %.not, float %1559, float %1549, !dbg !7
  %1562 = select i1 %.not1, float %1560, float %1554, !dbg !7
  %1563 = bitcast float %1562 to i32, !dbg !7
  %1564 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1563), !dbg !7
  %1565 = select i1 %.not1, float %1561, float %1555, !dbg !7
  %1566 = bitcast float %1565 to i32, !dbg !7
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1149, i32 %1566), !dbg !7
  br i1 true, label %1568, label %1632, !dbg !20

1568:                                             ; preds = %1117
  %1569 = bitcast i32 %1564 to float, !dbg !7
  %1570 = select i1 %.not1, float %1554, float %1569, !dbg !7
  %1571 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !18
  %1572 = bitcast i32 %1567 to float, !dbg !7
  %1573 = select i1 %.not1, float %1555, float %1572, !dbg !7
  %1574 = insertelement <4 x float> %1571, float %1573, i64 1, !dbg !18
  %1575 = select i1 %.not1, float %1569, float %1560, !dbg !7
  %1576 = insertelement <4 x float> %1574, float %1575, i64 2, !dbg !18
  %1577 = select i1 %.not1, float %1572, float %1561, !dbg !7
  %1578 = insertelement <4 x float> %1576, float %1577, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1461, !dbg !18
  %normalized = fmul <4 x float> %1578, %row.reciprocal, !dbg !18
  %1579 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !19
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !19
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !19
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !19
  %1584 = bitcast i32 %1456 to float, !dbg !7
  %1585 = select i1 %.not1, float %1446, float %1584, !dbg !7
  %1586 = insertelement <4 x float> poison, float %1585, i64 0, !dbg !18
  %1587 = bitcast i32 %1459 to float, !dbg !7
  %1588 = select i1 %.not1, float %1447, float %1587, !dbg !7
  %1589 = insertelement <4 x float> %1586, float %1588, i64 1, !dbg !18
  %1590 = select i1 %.not1, float %1584, float %1452, !dbg !7
  %1591 = insertelement <4 x float> %1589, float %1590, i64 2, !dbg !18
  %1592 = select i1 %.not1, float %1587, float %1453, !dbg !7
  %1593 = insertelement <4 x float> %1591, float %1592, i64 3, !dbg !18
  %normalized36 = fmul <4 x float> %1593, %row.reciprocal, !dbg !18
  %1594 = fptrunc <4 x float> %normalized36 to <4 x half>, !dbg !19
  %1595 = extractelement <4 x half> %1594, i64 3, !dbg !19
  %1596 = extractelement <4 x half> %1594, i64 2, !dbg !19
  %1597 = extractelement <4 x half> %1594, i64 1, !dbg !19
  %1598 = extractelement <4 x half> %1594, i64 0, !dbg !19
  %1599 = bitcast i32 %1346 to float, !dbg !7
  %1600 = select i1 %.not1, float %1336, float %1599, !dbg !7
  %1601 = insertelement <4 x float> poison, float %1600, i64 0, !dbg !18
  %1602 = bitcast i32 %1349 to float, !dbg !7
  %1603 = select i1 %.not1, float %1337, float %1602, !dbg !7
  %1604 = insertelement <4 x float> %1601, float %1603, i64 1, !dbg !18
  %1605 = select i1 %.not1, float %1599, float %1342, !dbg !7
  %1606 = insertelement <4 x float> %1604, float %1605, i64 2, !dbg !18
  %1607 = select i1 %.not1, float %1602, float %1343, !dbg !7
  %1608 = insertelement <4 x float> %1606, float %1607, i64 3, !dbg !18
  %row.reciprocal37 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1243, !dbg !18
  %normalized38 = fmul <4 x float> %1608, %row.reciprocal37, !dbg !18
  %1609 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !19
  %1610 = extractelement <4 x half> %1609, i64 3, !dbg !19
  %1611 = extractelement <4 x half> %1609, i64 2, !dbg !19
  %1612 = extractelement <4 x half> %1609, i64 1, !dbg !19
  %1613 = extractelement <4 x half> %1609, i64 0, !dbg !19
  %1614 = bitcast i32 %1238 to float, !dbg !7
  %1615 = select i1 %.not1, float %1228, float %1614, !dbg !7
  %1616 = insertelement <4 x float> poison, float %1615, i64 0, !dbg !18
  %1617 = bitcast i32 %1241 to float, !dbg !7
  %1618 = select i1 %.not1, float %1229, float %1617, !dbg !7
  %1619 = insertelement <4 x float> %1616, float %1618, i64 1, !dbg !18
  %1620 = select i1 %.not1, float %1614, float %1234, !dbg !7
  %1621 = insertelement <4 x float> %1619, float %1620, i64 2, !dbg !18
  %1622 = select i1 %.not1, float %1617, float %1235, !dbg !7
  %1623 = insertelement <4 x float> %1621, float %1622, i64 3, !dbg !18
  %normalized39 = fmul <4 x float> %1623, %row.reciprocal37, !dbg !18
  %1624 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !19
  %1625 = extractelement <4 x half> %1624, i64 3, !dbg !19
  %1626 = extractelement <4 x half> %1624, i64 2, !dbg !19
  %1627 = extractelement <4 x half> %1624, i64 1, !dbg !19
  %1628 = extractelement <4 x half> %1624, i64 0, !dbg !19
  %1629 = add nsw i64 %15, %25, !dbg !20
  %1630 = shl i64 %1629, 19, !dbg !20
  %1631 = and i64 %1630, 16252928, !dbg !20
  %address.thread239 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local240 = zext nneg i32 %address.thread239 to i64, !dbg !12
  %address.local241 = add i64 %28, %address.local240, !dbg !20
  %address.local242 = lshr i64 %address.local240, 6, !dbg !12
  %address.local243 = shl nuw nsw i64 %address.local242, 4, !dbg !20
  %address.local244 = add i64 %address.local241, %address.local243, !dbg !20
  %address.local245 = and i64 %address.local240, 240, !dbg !20
  %address.local246 = sub i64 %address.local244, %address.local245, !dbg !20
  %address.local247 = add i64 %address.local246, %29, !dbg !20
  %address.local248 = shl nuw nsw i64 %address.local247, 7, !dbg !20
  %address.local249 = add nuw nsw i64 %1631, %address.local248, !dbg !20
  %address.local250 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local251 = lshr i64 %address.local240, 2, !dbg !12
  %address.local252 = and i64 %address.local251, 12, !dbg !12
  %address.local253 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local250, i64 %address.local252, !dbg !20
  store half %1628, ptr addrspace(1) %address.local253, align 2, !dbg !20
  %address.local254 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local255 = or disjoint i64 %address.local252, 1, !dbg !20
  %address.local256 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local254, i64 %address.local255, !dbg !20
  store half %1627, ptr addrspace(1) %address.local256, align 2, !dbg !20
  %address.local257 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local258 = or disjoint i64 %address.local252, 2, !dbg !20
  %address.local259 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local257, i64 %address.local258, !dbg !20
  store half %1626, ptr addrspace(1) %address.local259, align 2, !dbg !20
  %address.local260 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local261 = or disjoint i64 %address.local252, 3, !dbg !20
  %address.local262 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local260, i64 %address.local261, !dbg !20
  store half %1625, ptr addrspace(1) %address.local262, align 2, !dbg !20
  %address.local263 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local264 = or disjoint i64 %address.local252, 16, !dbg !20
  %address.local265 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local263, i64 %address.local264, !dbg !20
  store half %1613, ptr addrspace(1) %address.local265, align 2, !dbg !20
  %address.local266 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local267 = or disjoint i64 %address.local252, 17, !dbg !20
  %address.local268 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local266, i64 %address.local267, !dbg !20
  store half %1612, ptr addrspace(1) %address.local268, align 2, !dbg !20
  %address.local269 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local270 = or disjoint i64 %address.local252, 18, !dbg !20
  %address.local271 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local269, i64 %address.local270, !dbg !20
  store half %1611, ptr addrspace(1) %address.local271, align 2, !dbg !20
  %address.local272 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local249, !dbg !20
  %address.local273 = or disjoint i64 %address.local252, 19, !dbg !20
  %address.local274 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local272, i64 %address.local273, !dbg !20
  store half %1610, ptr addrspace(1) %address.local274, align 2, !dbg !20
  %address.local275 = shl i64 %address.local247, 7, !dbg !20
  %address.local276 = add i64 %address.local275, 8192, !dbg !20
  %address.local277 = add nuw nsw i64 %1631, %address.local276, !dbg !20
  %address.local278 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local279 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local278, i64 %address.local252, !dbg !20
  store half %1598, ptr addrspace(1) %address.local279, align 2, !dbg !20
  %address.local280 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local281 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local280, i64 %address.local255, !dbg !20
  store half %1597, ptr addrspace(1) %address.local281, align 2, !dbg !20
  %address.local282 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local283 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local282, i64 %address.local258, !dbg !20
  store half %1596, ptr addrspace(1) %address.local283, align 2, !dbg !20
  %address.local284 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local285 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local284, i64 %address.local261, !dbg !20
  store half %1595, ptr addrspace(1) %address.local285, align 2, !dbg !20
  %address.local286 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local287 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local286, i64 %address.local264, !dbg !20
  store half %1583, ptr addrspace(1) %address.local287, align 2, !dbg !20
  %address.local288 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local289 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local288, i64 %address.local267, !dbg !20
  store half %1582, ptr addrspace(1) %address.local289, align 2, !dbg !20
  %address.local290 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local291 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local290, i64 %address.local270, !dbg !20
  store half %1581, ptr addrspace(1) %address.local291, align 2, !dbg !20
  %address.local292 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local277, !dbg !20
  %address.local293 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local292, i64 %address.local273, !dbg !20
  store half %1580, ptr addrspace(1) %address.local293, align 2, !dbg !20
  br label %1632, !dbg !20

1632:                                             ; preds = %1568, %1117
  br i1 true, label %1633, label %1671, !dbg !20

1633:                                             ; preds = %1632
  %1634 = insertelement <4 x float> poison, float %1538, i64 0, !dbg !18
  %1635 = insertelement <4 x float> %1634, float %1544, i64 1, !dbg !18
  %1636 = insertelement <4 x float> %1635, float %1539, i64 2, !dbg !18
  %1637 = insertelement <4 x float> %1636, float %1545, i64 3, !dbg !18
  %row.reciprocal40 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1461, !dbg !18
  %normalized41 = fmul <4 x float> %1637, %row.reciprocal40, !dbg !18
  %1638 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !19
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !19
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !19
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !19
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !19
  %1643 = insertelement <4 x float> poison, float %1429, i64 0, !dbg !18
  %1644 = insertelement <4 x float> %1643, float %1435, i64 1, !dbg !18
  %1645 = insertelement <4 x float> %1644, float %1430, i64 2, !dbg !18
  %1646 = insertelement <4 x float> %1645, float %1436, i64 3, !dbg !18
  %normalized42 = fmul <4 x float> %1646, %row.reciprocal40, !dbg !18
  %1647 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !19
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !19
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !19
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !19
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !19
  %1652 = insertelement <4 x float> poison, float %1320, i64 0, !dbg !18
  %1653 = insertelement <4 x float> %1652, float %1326, i64 1, !dbg !18
  %1654 = insertelement <4 x float> %1653, float %1321, i64 2, !dbg !18
  %1655 = insertelement <4 x float> %1654, float %1327, i64 3, !dbg !18
  %row.reciprocal43 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1243, !dbg !18
  %normalized44 = fmul <4 x float> %1655, %row.reciprocal43, !dbg !18
  %1656 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !19
  %1657 = extractelement <4 x half> %1656, i64 3, !dbg !19
  %1658 = extractelement <4 x half> %1656, i64 2, !dbg !19
  %1659 = extractelement <4 x half> %1656, i64 1, !dbg !19
  %1660 = extractelement <4 x half> %1656, i64 0, !dbg !19
  %1661 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !18
  %1662 = insertelement <4 x float> %1661, float %1217, i64 1, !dbg !18
  %1663 = insertelement <4 x float> %1662, float %1212, i64 2, !dbg !18
  %1664 = insertelement <4 x float> %1663, float %1218, i64 3, !dbg !18
  %normalized45 = fmul <4 x float> %1664, %row.reciprocal43, !dbg !18
  %1665 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !19
  %1666 = extractelement <4 x half> %1665, i64 3, !dbg !19
  %1667 = extractelement <4 x half> %1665, i64 2, !dbg !19
  %1668 = extractelement <4 x half> %1665, i64 1, !dbg !19
  %1669 = extractelement <4 x half> %1665, i64 0, !dbg !19
  %1670 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread294 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local295 = zext nneg i32 %address.thread294 to i64, !dbg !12
  %address.local296 = add i64 %28, %address.local295, !dbg !20
  %address.local297 = lshr i64 %address.local295, 6, !dbg !12
  %address.local298 = shl nuw nsw i64 %address.local297, 4, !dbg !20
  %address.local299 = add i64 %address.local296, %address.local298, !dbg !20
  %address.local300 = and i64 %address.local295, 240, !dbg !20
  %address.local301 = sub i64 %address.local299, %address.local300, !dbg !20
  %address.local302 = add i64 %address.local301, %29, !dbg !20
  %address.local303 = shl nuw nsw i64 %address.local302, 7, !dbg !20
  %address.local304 = add nuw nsw i64 %1670, %address.local303, !dbg !20
  %address.local305 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local306 = lshr i64 %address.local295, 2, !dbg !12
  %address.local307 = and i64 %address.local306, 12, !dbg !12
  %address.local308 = or disjoint i64 %address.local307, 32, !dbg !20
  %address.local309 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local305, i64 %address.local308, !dbg !20
  store half %1669, ptr addrspace(1) %address.local309, align 2, !dbg !20
  %address.local310 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local311 = or disjoint i64 %address.local307, 33, !dbg !20
  %address.local312 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local310, i64 %address.local311, !dbg !20
  store half %1668, ptr addrspace(1) %address.local312, align 2, !dbg !20
  %address.local313 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local314 = or disjoint i64 %address.local307, 34, !dbg !20
  %address.local315 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local313, i64 %address.local314, !dbg !20
  store half %1667, ptr addrspace(1) %address.local315, align 2, !dbg !20
  %address.local316 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local317 = or disjoint i64 %address.local307, 35, !dbg !20
  %address.local318 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local316, i64 %address.local317, !dbg !20
  store half %1666, ptr addrspace(1) %address.local318, align 2, !dbg !20
  %address.local319 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local320 = or disjoint i64 %address.local307, 48, !dbg !20
  %address.local321 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local319, i64 %address.local320, !dbg !20
  store half %1660, ptr addrspace(1) %address.local321, align 2, !dbg !20
  %address.local322 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local323 = or disjoint i64 %address.local307, 49, !dbg !20
  %address.local324 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local322, i64 %address.local323, !dbg !20
  store half %1659, ptr addrspace(1) %address.local324, align 2, !dbg !20
  %address.local325 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local326 = or disjoint i64 %address.local307, 50, !dbg !20
  %address.local327 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local325, i64 %address.local326, !dbg !20
  store half %1658, ptr addrspace(1) %address.local327, align 2, !dbg !20
  %address.local328 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local304, !dbg !20
  %address.local329 = or i64 %address.local306, 51, !dbg !20
  %address.local330 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local328, i64 %address.local329, !dbg !20
  store half %1657, ptr addrspace(1) %address.local330, align 2, !dbg !20
  %address.local331 = shl i64 %address.local302, 7, !dbg !20
  %address.local332 = add i64 %address.local331, 8192, !dbg !20
  %address.local333 = add nuw nsw i64 %1670, %address.local332, !dbg !20
  %address.local334 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local335 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local334, i64 %address.local308, !dbg !20
  store half %1651, ptr addrspace(1) %address.local335, align 2, !dbg !20
  %address.local336 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local337 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local336, i64 %address.local311, !dbg !20
  store half %1650, ptr addrspace(1) %address.local337, align 2, !dbg !20
  %address.local338 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local339 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local338, i64 %address.local314, !dbg !20
  store half %1649, ptr addrspace(1) %address.local339, align 2, !dbg !20
  %address.local340 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local341 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local340, i64 %address.local317, !dbg !20
  store half %1648, ptr addrspace(1) %address.local341, align 2, !dbg !20
  %address.local342 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local343 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local342, i64 %address.local320, !dbg !20
  store half %1642, ptr addrspace(1) %address.local343, align 2, !dbg !20
  %address.local344 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local345 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local344, i64 %address.local323, !dbg !20
  store half %1641, ptr addrspace(1) %address.local345, align 2, !dbg !20
  %address.local346 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local347 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local346, i64 %address.local326, !dbg !20
  store half %1640, ptr addrspace(1) %address.local347, align 2, !dbg !20
  %address.local348 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local333, !dbg !20
  %address.local349 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local348, i64 %address.local329, !dbg !20
  store half %1639, ptr addrspace(1) %address.local349, align 2, !dbg !20
  br label %1671, !dbg !20

1671:                                             ; preds = %1633, %1632
  br i1 true, label %1672, label %1710, !dbg !20

1672:                                             ; preds = %1671
  %1673 = insertelement <4 x float> poison, float %1510, i64 0, !dbg !18
  %1674 = insertelement <4 x float> %1673, float %1516, i64 1, !dbg !18
  %1675 = insertelement <4 x float> %1674, float %1511, i64 2, !dbg !18
  %1676 = insertelement <4 x float> %1675, float %1517, i64 3, !dbg !18
  %row.reciprocal46 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1461, !dbg !18
  %normalized47 = fmul <4 x float> %1676, %row.reciprocal46, !dbg !18
  %1677 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !19
  %1678 = extractelement <4 x half> %1677, i64 3, !dbg !19
  %1679 = extractelement <4 x half> %1677, i64 2, !dbg !19
  %1680 = extractelement <4 x half> %1677, i64 1, !dbg !19
  %1681 = extractelement <4 x half> %1677, i64 0, !dbg !19
  %1682 = insertelement <4 x float> poison, float %1400, i64 0, !dbg !18
  %1683 = insertelement <4 x float> %1682, float %1406, i64 1, !dbg !18
  %1684 = insertelement <4 x float> %1683, float %1401, i64 2, !dbg !18
  %1685 = insertelement <4 x float> %1684, float %1407, i64 3, !dbg !18
  %normalized48 = fmul <4 x float> %1685, %row.reciprocal46, !dbg !18
  %1686 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !19
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !19
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !19
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !19
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !19
  %1691 = insertelement <4 x float> poison, float %1292, i64 0, !dbg !18
  %1692 = insertelement <4 x float> %1691, float %1298, i64 1, !dbg !18
  %1693 = insertelement <4 x float> %1692, float %1293, i64 2, !dbg !18
  %1694 = insertelement <4 x float> %1693, float %1299, i64 3, !dbg !18
  %row.reciprocal49 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1243, !dbg !18
  %normalized50 = fmul <4 x float> %1694, %row.reciprocal49, !dbg !18
  %1695 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !19
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !19
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !19
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !19
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !19
  %1700 = insertelement <4 x float> poison, float %1182, i64 0, !dbg !18
  %1701 = insertelement <4 x float> %1700, float %1188, i64 1, !dbg !18
  %1702 = insertelement <4 x float> %1701, float %1183, i64 2, !dbg !18
  %1703 = insertelement <4 x float> %1702, float %1189, i64 3, !dbg !18
  %normalized51 = fmul <4 x float> %1703, %row.reciprocal49, !dbg !18
  %1704 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !19
  %1705 = extractelement <4 x half> %1704, i64 3, !dbg !19
  %1706 = extractelement <4 x half> %1704, i64 2, !dbg !19
  %1707 = extractelement <4 x half> %1704, i64 1, !dbg !19
  %1708 = extractelement <4 x half> %1704, i64 0, !dbg !19
  %1709 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread350 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local351 = zext nneg i32 %address.thread350 to i64, !dbg !12
  %address.local352 = add i64 %28, %address.local351, !dbg !20
  %address.local353 = lshr i64 %address.local351, 6, !dbg !12
  %address.local354 = shl nuw nsw i64 %address.local353, 4, !dbg !20
  %address.local355 = add i64 %address.local352, %address.local354, !dbg !20
  %address.local356 = and i64 %address.local351, 240, !dbg !20
  %address.local357 = sub i64 %address.local355, %address.local356, !dbg !20
  %address.local358 = add i64 %address.local357, %29, !dbg !20
  %address.local359 = shl nuw nsw i64 %address.local358, 7, !dbg !20
  %address.local360 = add nuw nsw i64 %1709, %address.local359, !dbg !20
  %address.local361 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local362 = lshr i64 %address.local351, 2, !dbg !12
  %address.local363 = and i64 %address.local362, 12, !dbg !12
  %address.local364 = or disjoint i64 %address.local363, 64, !dbg !20
  %address.local365 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local361, i64 %address.local364, !dbg !20
  store half %1708, ptr addrspace(1) %address.local365, align 2, !dbg !20
  %address.local366 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local367 = or disjoint i64 %address.local363, 65, !dbg !20
  %address.local368 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local366, i64 %address.local367, !dbg !20
  store half %1707, ptr addrspace(1) %address.local368, align 2, !dbg !20
  %address.local369 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local370 = or disjoint i64 %address.local363, 66, !dbg !20
  %address.local371 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local369, i64 %address.local370, !dbg !20
  store half %1706, ptr addrspace(1) %address.local371, align 2, !dbg !20
  %address.local372 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local373 = or disjoint i64 %address.local363, 67, !dbg !20
  %address.local374 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local372, i64 %address.local373, !dbg !20
  store half %1705, ptr addrspace(1) %address.local374, align 2, !dbg !20
  %address.local375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local376 = or disjoint i64 %address.local363, 80, !dbg !20
  %address.local377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local375, i64 %address.local376, !dbg !20
  store half %1699, ptr addrspace(1) %address.local377, align 2, !dbg !20
  %address.local378 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local379 = or disjoint i64 %address.local363, 81, !dbg !20
  %address.local380 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local378, i64 %address.local379, !dbg !20
  store half %1698, ptr addrspace(1) %address.local380, align 2, !dbg !20
  %address.local381 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local382 = or disjoint i64 %address.local363, 82, !dbg !20
  %address.local383 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local381, i64 %address.local382, !dbg !20
  store half %1697, ptr addrspace(1) %address.local383, align 2, !dbg !20
  %address.local384 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local360, !dbg !20
  %address.local385 = or disjoint i64 %address.local363, 83, !dbg !20
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local384, i64 %address.local385, !dbg !20
  store half %1696, ptr addrspace(1) %address.local386, align 2, !dbg !20
  %address.local387 = shl i64 %address.local358, 7, !dbg !20
  %address.local388 = add i64 %address.local387, 8192, !dbg !20
  %address.local389 = add nuw nsw i64 %1709, %address.local388, !dbg !20
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local391 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local390, i64 %address.local364, !dbg !20
  store half %1690, ptr addrspace(1) %address.local391, align 2, !dbg !20
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local367, !dbg !20
  store half %1689, ptr addrspace(1) %address.local393, align 2, !dbg !20
  %address.local394 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local394, i64 %address.local370, !dbg !20
  store half %1688, ptr addrspace(1) %address.local395, align 2, !dbg !20
  %address.local396 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local397 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local396, i64 %address.local373, !dbg !20
  store half %1687, ptr addrspace(1) %address.local397, align 2, !dbg !20
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local376, !dbg !20
  store half %1681, ptr addrspace(1) %address.local399, align 2, !dbg !20
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local379, !dbg !20
  store half %1680, ptr addrspace(1) %address.local401, align 2, !dbg !20
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local402, i64 %address.local382, !dbg !20
  store half %1679, ptr addrspace(1) %address.local403, align 2, !dbg !20
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local389, !dbg !20
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local385, !dbg !20
  store half %1678, ptr addrspace(1) %address.local405, align 2, !dbg !20
  br label %1710, !dbg !20

1710:                                             ; preds = %1672, %1671
  br i1 true, label %1711, label %1749, !dbg !20

1711:                                             ; preds = %1710
  %1712 = insertelement <4 x float> poison, float %1482, i64 0, !dbg !18
  %1713 = insertelement <4 x float> %1712, float %1488, i64 1, !dbg !18
  %1714 = insertelement <4 x float> %1713, float %1483, i64 2, !dbg !18
  %1715 = insertelement <4 x float> %1714, float %1489, i64 3, !dbg !18
  %row.reciprocal52 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1461, !dbg !18
  %normalized53 = fmul <4 x float> %1715, %row.reciprocal52, !dbg !18
  %1716 = fptrunc <4 x float> %normalized53 to <4 x half>, !dbg !19
  %1717 = extractelement <4 x half> %1716, i64 3, !dbg !19
  %1718 = extractelement <4 x half> %1716, i64 2, !dbg !19
  %1719 = extractelement <4 x half> %1716, i64 1, !dbg !19
  %1720 = extractelement <4 x half> %1716, i64 0, !dbg !19
  %1721 = insertelement <4 x float> poison, float %1371, i64 0, !dbg !18
  %1722 = insertelement <4 x float> %1721, float %1377, i64 1, !dbg !18
  %1723 = insertelement <4 x float> %1722, float %1372, i64 2, !dbg !18
  %1724 = insertelement <4 x float> %1723, float %1378, i64 3, !dbg !18
  %normalized54 = fmul <4 x float> %1724, %row.reciprocal52, !dbg !18
  %1725 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !19
  %1726 = extractelement <4 x half> %1725, i64 3, !dbg !19
  %1727 = extractelement <4 x half> %1725, i64 2, !dbg !19
  %1728 = extractelement <4 x half> %1725, i64 1, !dbg !19
  %1729 = extractelement <4 x half> %1725, i64 0, !dbg !19
  %1730 = insertelement <4 x float> poison, float %1264, i64 0, !dbg !18
  %1731 = insertelement <4 x float> %1730, float %1270, i64 1, !dbg !18
  %1732 = insertelement <4 x float> %1731, float %1265, i64 2, !dbg !18
  %1733 = insertelement <4 x float> %1732, float %1271, i64 3, !dbg !18
  %row.reciprocal55 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1243, !dbg !18
  %normalized56 = fmul <4 x float> %1733, %row.reciprocal55, !dbg !18
  %1734 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !19
  %1735 = extractelement <4 x half> %1734, i64 3, !dbg !19
  %1736 = extractelement <4 x half> %1734, i64 2, !dbg !19
  %1737 = extractelement <4 x half> %1734, i64 1, !dbg !19
  %1738 = extractelement <4 x half> %1734, i64 0, !dbg !19
  %1739 = insertelement <4 x float> poison, float %1153, i64 0, !dbg !18
  %1740 = insertelement <4 x float> %1739, float %1159, i64 1, !dbg !18
  %1741 = insertelement <4 x float> %1740, float %1154, i64 2, !dbg !18
  %1742 = insertelement <4 x float> %1741, float %1160, i64 3, !dbg !18
  %normalized57 = fmul <4 x float> %1742, %row.reciprocal55, !dbg !18
  %1743 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !19
  %1744 = extractelement <4 x half> %1743, i64 3, !dbg !19
  %1745 = extractelement <4 x half> %1743, i64 2, !dbg !19
  %1746 = extractelement <4 x half> %1743, i64 1, !dbg !19
  %1747 = extractelement <4 x half> %1743, i64 0, !dbg !19
  %1748 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread406 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local407 = zext nneg i32 %address.thread406 to i64, !dbg !12
  %address.local408 = add i64 %28, %address.local407, !dbg !20
  %address.local409 = lshr i64 %address.local407, 6, !dbg !12
  %address.local410 = shl nuw nsw i64 %address.local409, 4, !dbg !20
  %address.local411 = add i64 %address.local408, %address.local410, !dbg !20
  %address.local412 = and i64 %address.local407, 240, !dbg !20
  %address.local413 = sub i64 %address.local411, %address.local412, !dbg !20
  %address.local414 = add i64 %address.local413, %29, !dbg !20
  %address.local415 = shl nuw nsw i64 %address.local414, 7, !dbg !20
  %address.local416 = add nuw nsw i64 %1748, %address.local415, !dbg !20
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local418 = lshr i64 %address.local407, 2, !dbg !12
  %address.local419 = and i64 %address.local418, 12, !dbg !12
  %address.local420 = or disjoint i64 %address.local419, 96, !dbg !20
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local417, i64 %address.local420, !dbg !20
  store half %1747, ptr addrspace(1) %address.local421, align 2, !dbg !20
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local423 = or disjoint i64 %address.local419, 97, !dbg !20
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local422, i64 %address.local423, !dbg !20
  store half %1746, ptr addrspace(1) %address.local424, align 2, !dbg !20
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local426 = or disjoint i64 %address.local419, 98, !dbg !20
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local425, i64 %address.local426, !dbg !20
  store half %1745, ptr addrspace(1) %address.local427, align 2, !dbg !20
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local429 = or disjoint i64 %address.local419, 99, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local429, !dbg !20
  store half %1744, ptr addrspace(1) %address.local430, align 2, !dbg !20
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local432 = or disjoint i64 %address.local419, 112, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local431, i64 %address.local432, !dbg !20
  store half %1738, ptr addrspace(1) %address.local433, align 2, !dbg !20
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local435 = or disjoint i64 %address.local419, 113, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local435, !dbg !20
  store half %1737, ptr addrspace(1) %address.local436, align 2, !dbg !20
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local438 = or disjoint i64 %address.local419, 114, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local438, !dbg !20
  store half %1736, ptr addrspace(1) %address.local439, align 2, !dbg !20
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local416, !dbg !20
  %address.local441 = or i64 %address.local418, 115, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local441, !dbg !20
  store half %1735, ptr addrspace(1) %address.local442, align 2, !dbg !20
  %address.local443 = shl i64 %address.local414, 7, !dbg !20
  %address.local444 = add i64 %address.local443, 8192, !dbg !20
  %address.local445 = add nuw nsw i64 %1748, %address.local444, !dbg !20
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local446, i64 %address.local420, !dbg !20
  store half %1729, ptr addrspace(1) %address.local447, align 2, !dbg !20
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local448, i64 %address.local423, !dbg !20
  store half %1728, ptr addrspace(1) %address.local449, align 2, !dbg !20
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local451 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local450, i64 %address.local426, !dbg !20
  store half %1727, ptr addrspace(1) %address.local451, align 2, !dbg !20
  %address.local452 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local453 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local452, i64 %address.local429, !dbg !20
  store half %1726, ptr addrspace(1) %address.local453, align 2, !dbg !20
  %address.local454 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local454, i64 %address.local432, !dbg !20
  store half %1720, ptr addrspace(1) %address.local455, align 2, !dbg !20
  %address.local456 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local457 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local456, i64 %address.local435, !dbg !20
  store half %1719, ptr addrspace(1) %address.local457, align 2, !dbg !20
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local458, i64 %address.local438, !dbg !20
  store half %1718, ptr addrspace(1) %address.local459, align 2, !dbg !20
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local445, !dbg !20
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local441, !dbg !20
  store half %1717, ptr addrspace(1) %address.local461, align 2, !dbg !20
  br label %1749, !dbg !20

1749:                                             ; preds = %1711, %1710
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0005-test_input_1x32x4096x128-128-64-32-online/group8-reverse")
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
