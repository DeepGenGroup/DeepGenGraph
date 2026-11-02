; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/group8-reverse/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/group8-reverse/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

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
  %40 = icmp ugt i64 %10, 2047, !dbg !7
  %41 = select i1 %40, i64 %21, i64 %16, !dbg !7
  %42 = sdiv i64 %41, 2, !dbg !7
  %43 = sub nsw i64 0, %42, !dbg !7
  %44 = add nsw i64 %42, 1, !dbg !7
  %45 = select i1 %40, i64 %43, i64 %44, !dbg !7
  %46 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %47 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %48 = getelementptr i8, ptr addrspace(1) %47, i64 64, !dbg !11
  %49 = shl i64 %15, 7, !dbg !15
  %50 = and i64 %49, 3968, !dbg !15
  %51 = shl nsw i64 %18, 12, !dbg !15
  %52 = or disjoint i64 %50, %51, !dbg !15
  %53 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %54 = getelementptr i8, ptr addrspace(1) %53, i64 128, !dbg !11
  %55 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 192, !dbg !11
  br label %57, !dbg !7

57:                                               ; preds = %584, %4
  %58 = phi i64 [ %1146, %584 ], [ 0, %4 ], !dbg !7
  %59 = phi <4 x float> [ %921, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi <4 x float> [ %937, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi <8 x float> [ %1142, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %62 = phi <1 x float> [ %855, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi <1 x float> [ %686, %584 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %64 = phi <4 x float> [ %999, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = phi <4 x float> [ %1015, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %66 = phi <8 x float> [ %1143, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %67 = phi <4 x float> [ %1077, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %68 = phi <4 x float> [ %1093, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %69 = phi <8 x float> [ %1144, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %70 = phi <4 x float> [ %1125, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %71 = phi <4 x float> [ %1141, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %72 = phi <8 x float> [ %1145, %584 ], [ zeroinitializer, %4 ], !dbg !7
  %73 = icmp slt i64 %58, %45, !dbg !7
  br i1 %73, label %74, label %1147, !dbg !7

74:                                               ; preds = %57
  %75 = shl nsw i64 %58, 7, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !38, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local80 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local81 = and i64 %address.local80, 1536, !dbg !11
  %address.local82 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local81, !dbg !11
  %address.local83 = ptrtoint ptr addrspace(3) %address.local82 to i64, !dbg !11
  %address.local84 = trunc i64 %address.local83 to i32, !dbg !11
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local84), !dbg !11
  %77 = zext i32 %76 to i64, !dbg !11
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !11
  %address.local85 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local86 = and i64 %address.local85, 8064, !dbg !11
  %address.local87 = lshr i64 %address.local, 3, !dbg !11
  %address.local88 = and i64 %address.local87, 24, !dbg !11
  %address.local89 = or disjoint i64 %address.local86, %address.local88, !dbg !11
  %address.local90 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local89, !dbg !11
  %79 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %80 = ptrtoint ptr addrspace(1) %address.local90 to i64, !dbg !11
  %81 = sub i64 %80, %79, !dbg !11
  %dma.buffer.offset696 = trunc i64 %81 to i32, !dbg !11
  %82 = trunc i64 %79 to i32, !dbg !11
  %83 = insertelement <4 x i32> poison, i32 %82, i64 0, !dbg !11
  %84 = lshr i64 %79, 32, !dbg !11
  %85 = trunc i64 %84 to i32, !dbg !11
  %86 = insertelement <4 x i32> %83, i32 %85, i64 1, !dbg !11
  %87 = insertelement <4 x i32> %86, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc697 = insertelement <4 x i32> %87, i32 131072, i64 3, !dbg !11
  %88 = ptrtoint ptr addrspace(3) %78 to i32, !dbg !11
  %dma.buffer.lds698 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88), !dbg !11
  %89 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds698, i32 %dma.buffer.offset696, <4 x i32> %dma.buffer.rsrc697) #10, !dbg !11
  %90 = lshr i64 %58, 12, !dbg !15
  %91 = add nuw nsw i64 %90, %15, !dbg !15
  %92 = lshr i64 %58, 5, !dbg !15
  %93 = and i64 %92, 127, !dbg !15
  %94 = shl i64 %91, 19, !dbg !15
  %95 = and i64 %94, 16252928, !dbg !15
  %96 = shl i64 %92, 19, !dbg !15
  %97 = shl i64 %58, 14, !dbg !15
  %98 = sub i64 %97, %96, !dbg !15
  %99 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %95, !dbg !15
  %100 = getelementptr [2 x i8], ptr addrspace(1) %99, i64 %93, !dbg !15
  %101 = getelementptr [2 x i8], ptr addrspace(1) %100, i64 %98, !dbg !15
  %address.local91 = and i64 %address.local80, 1536, !dbg !15
  %address.local92 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local91, !dbg !15
  %address.local93 = ptrtoint ptr addrspace(3) %address.local92 to i64, !dbg !15
  %address.local94 = trunc i64 %address.local93 to i32, !dbg !15
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !15
  %103 = zext i32 %102 to i64, !dbg !15
  %104 = inttoptr i64 %103 to ptr addrspace(3), !dbg !15
  %address.local95 = lshr i64 %address.local, 4, !dbg !15
  %address.local96 = and i64 %address.local95, 8, !dbg !15
  %address.local97 = shl nuw nsw i64 %address.local, 7, !dbg !15
  %address.local98 = and i64 %address.local97, 16256, !dbg !15
  %address.local99 = or disjoint i64 %address.local96, %address.local98, !dbg !15
  %address.local100 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %address.local99, !dbg !15
  %105 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %106 = ptrtoint ptr addrspace(1) %address.local100 to i64, !dbg !15
  %107 = sub i64 %106, %105, !dbg !15
  %dma.buffer.offset693 = trunc i64 %107 to i32, !dbg !15
  %108 = trunc i64 %105 to i32, !dbg !15
  %109 = insertelement <4 x i32> poison, i32 %108, i64 0, !dbg !15
  %110 = lshr i64 %105, 32, !dbg !15
  %111 = trunc i64 %110 to i32, !dbg !15
  %112 = insertelement <4 x i32> %109, i32 %111, i64 1, !dbg !15
  %113 = insertelement <4 x i32> %112, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc694 = insertelement <4 x i32> %113, i32 131072, i64 3, !dbg !15
  %114 = ptrtoint ptr addrspace(3) %104 to i32, !dbg !15
  %dma.buffer.lds695 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %114), !dbg !15
  %115 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds695, i32 %dma.buffer.offset693, <4 x i32> %dma.buffer.rsrc694) #10, !dbg !15
  %address.local101 = or disjoint i64 %address.local, 256, !dbg !15
  %address.local102 = shl nuw nsw i64 %address.local101, 3, !dbg !15
  %address.local103 = and i64 %address.local102, 3584, !dbg !15
  %address.local104 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local103, !dbg !15
  %address.local105 = ptrtoint ptr addrspace(3) %address.local104 to i64, !dbg !15
  %address.local106 = trunc i64 %address.local105 to i32, !dbg !15
  %116 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local106), !dbg !15
  %117 = zext i32 %116 to i64, !dbg !15
  %118 = inttoptr i64 %117 to ptr addrspace(3), !dbg !15
  %address.local107 = lshr i64 %address.local101, 4, !dbg !15
  %address.local108 = and i64 %address.local107, 24, !dbg !15
  %address.local109 = shl nuw nsw i64 %address.local, 7, !dbg !15
  %address.local110 = and i64 %address.local109, 16256, !dbg !15
  %address.local111 = or disjoint i64 %address.local108, %address.local110, !dbg !15
  %address.local112 = getelementptr [2 x i8], ptr addrspace(1) %101, i64 %address.local111, !dbg !15
  %119 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %120 = ptrtoint ptr addrspace(1) %address.local112 to i64, !dbg !15
  %121 = sub i64 %120, %119, !dbg !15
  %dma.buffer.offset690 = trunc i64 %121 to i32, !dbg !15
  %122 = trunc i64 %119 to i32, !dbg !15
  %123 = insertelement <4 x i32> poison, i32 %122, i64 0, !dbg !15
  %124 = lshr i64 %119, 32, !dbg !15
  %125 = trunc i64 %124 to i32, !dbg !15
  %126 = insertelement <4 x i32> %123, i32 %125, i64 1, !dbg !15
  %127 = insertelement <4 x i32> %126, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc691 = insertelement <4 x i32> %127, i32 131072, i64 3, !dbg !15
  %128 = ptrtoint ptr addrspace(3) %118 to i32, !dbg !15
  %dma.buffer.lds692 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %128), !dbg !15
  %129 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds692, i32 %dma.buffer.offset690, <4 x i32> %dma.buffer.rsrc691) #10, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local113 = lshr i32 %address.thread, 2, !dbg !12
  %address.local114 = and i32 %address.local113, 4, !dbg !12
  %address.local115 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local116 = and i32 %address.local115, 384, !dbg !12
  %address.local117 = or disjoint i32 %address.local114, %address.local116, !dbg !12
  %address.local118 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local119 = and i32 %address.local118, 120, !dbg !12
  %address.local120 = or disjoint i32 %address.local117, %address.local119, !dbg !12
  %address.local121 = shl nuw nsw i32 %address.thread, 4, !dbg !12
  %address.local122 = and i32 %address.local121, 512, !dbg !12
  %address.local123 = or disjoint i32 %address.local120, %address.local122, !dbg !12
  %address.local124 = zext nneg i32 %address.local123 to i64, !dbg !12
  %address.local125 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local124, !dbg !12
  %130 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %address.local126 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local124, !dbg !12
  %address.local127 = getelementptr i8, ptr addrspace(3) %address.local126, i64 2048, !dbg !12
  %131 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %132 = fmul <4 x half> %130, splat (half 0xH3015), !dbg !12
  %133 = fmul <4 x half> %131, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local128 = lshr i32 %address.thread, 2, !dbg !16
  %address.local129 = and i32 %address.local128, 4, !dbg !16
  %address.local130 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local131 = and i32 %address.local130, 24, !dbg !16
  %address.local132 = or disjoint i32 %address.local129, %address.local131, !dbg !16
  %address.local133 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local134 = and i32 %address.local133, 1024, !dbg !16
  %address.local135 = or disjoint i32 %address.local132, %address.local134, !dbg !16
  %address.local136 = shl nuw nsw i32 %address.thread, 7, !dbg !16
  %address.local137 = and i32 %address.local136, 384, !dbg !16
  %address.local138 = or disjoint i32 %address.local135, %address.local137, !dbg !16
  %address.local139 = zext nneg i32 %address.local138 to i64, !dbg !16
  %address.local140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local139, !dbg !16
  %134 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %address.local141 = lshr i32 %address.thread, 2, !dbg !16
  %address.local142 = and i32 %address.local141, 4, !dbg !16
  %address.local143 = shl nuw nsw i32 %address.thread, 1, !dbg !16
  %address.local144 = and i32 %address.local143, 24, !dbg !16
  %address.local145 = or disjoint i32 %address.local142, %address.local144, !dbg !16
  %address.local146 = shl nuw nsw i32 %address.thread, 5, !dbg !16
  %address.local147 = and i32 %address.local146, 1024, !dbg !16
  %address.local148 = or disjoint i32 %address.local145, %address.local147, !dbg !16
  %address.local149 = shl nuw nsw i32 %address.thread, 7, !dbg !16
  %address.local150 = and i32 %address.local149, 384, !dbg !16
  %address.local151 = or disjoint i32 %address.local148, %address.local150, !dbg !16
  %address.local152 = zext nneg i32 %address.local151 to i64, !dbg !16
  %address.local153 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local152, !dbg !16
  %address.local154 = getelementptr i8, ptr addrspace(3) %address.local153, i64 4096, !dbg !16
  %135 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %134, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local155 = shl nuw nsw i64 %address.local, 7, !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !16
  %address.local157 = lshr i64 %address.local, 2, !dbg !16
  %address.local158 = mul nsw i64 %address.local157, -512, !dbg !16
  %address.local159 = getelementptr [2 x i8], ptr addrspace(3) %address.local156, i64 %address.local158, !dbg !16
  %address.local160 = shl nuw nsw i64 %address.local, 1, !dbg !16
  %address.local161 = and i64 %address.local160, 24, !dbg !16
  %address.local162 = getelementptr [2 x i8], ptr addrspace(3) %address.local159, i64 %address.local161, !dbg !16
  %address.local163 = lshr i64 %address.local, 2, !dbg !12
  %address.local164 = and i64 %address.local163, 12, !dbg !12
  %address.local165 = getelementptr [2 x i8], ptr addrspace(3) %address.local162, i64 %address.local164, !dbg !16
  %address.local166 = and i64 %address.local, 32, !dbg !16
  %address.local167 = icmp eq i64 %address.local166, 0, !dbg !16
  %address.local168 = select i1 %address.local167, i64 0, i64 1016, !dbg !16
  %address.local169 = getelementptr [2 x i8], ptr addrspace(3) %address.local165, i64 %address.local168, !dbg !16
  %address.local170 = getelementptr i8, ptr addrspace(3) %address.local169, i64 64, !dbg !16
  %137 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %135, <4 x float> %136) #9, !dbg !16
  %address.local171 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !16
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) %address.local171, i64 %address.local164, !dbg !16
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) %address.local172, i64 %address.local168, !dbg !16
  %address.local174 = getelementptr [2 x i8], ptr addrspace(3) %address.local173, i64 %address.local158, !dbg !16
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) %address.local174, i64 %address.local161, !dbg !16
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 4160, !dbg !16
  %139 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !16
  %address.local178 = getelementptr [2 x i8], ptr addrspace(3) %address.local177, i64 %address.local158, !dbg !16
  %address.local179 = getelementptr [2 x i8], ptr addrspace(3) %address.local178, i64 %address.local161, !dbg !16
  %address.local180 = getelementptr [2 x i8], ptr addrspace(3) %address.local179, i64 %address.local164, !dbg !16
  %address.local181 = getelementptr [2 x i8], ptr addrspace(3) %address.local180, i64 %address.local168, !dbg !16
  %address.local182 = getelementptr i8, ptr addrspace(3) %address.local181, i64 128, !dbg !16
  %140 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !16
  %142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %139, <4 x float> %141) #9, !dbg !16
  %address.local183 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !16
  %address.local184 = getelementptr [2 x i8], ptr addrspace(3) %address.local183, i64 %address.local164, !dbg !16
  %address.local185 = getelementptr [2 x i8], ptr addrspace(3) %address.local184, i64 %address.local168, !dbg !16
  %address.local186 = getelementptr [2 x i8], ptr addrspace(3) %address.local185, i64 %address.local158, !dbg !16
  %address.local187 = getelementptr [2 x i8], ptr addrspace(3) %address.local186, i64 %address.local161, !dbg !16
  %address.local188 = getelementptr i8, ptr addrspace(3) %address.local187, i64 4224, !dbg !16
  %143 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !16
  %144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %140, <4 x float> zeroinitializer) #9, !dbg !16
  %145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %143, <4 x float> %144) #9, !dbg !16
  %address.local189 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !16
  %address.local190 = getelementptr [2 x i8], ptr addrspace(3) %address.local189, i64 %address.local158, !dbg !16
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) %address.local190, i64 %address.local161, !dbg !16
  %address.local192 = getelementptr [2 x i8], ptr addrspace(3) %address.local191, i64 %address.local164, !dbg !16
  %address.local193 = getelementptr [2 x i8], ptr addrspace(3) %address.local192, i64 %address.local168, !dbg !16
  %address.local194 = getelementptr i8, ptr addrspace(3) %address.local193, i64 192, !dbg !16
  %146 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !16
  %address.local195 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local155, !dbg !16
  %address.local196 = getelementptr [2 x i8], ptr addrspace(3) %address.local195, i64 %address.local164, !dbg !16
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) %address.local196, i64 %address.local168, !dbg !16
  %address.local198 = getelementptr [2 x i8], ptr addrspace(3) %address.local197, i64 %address.local158, !dbg !16
  %address.local199 = getelementptr [2 x i8], ptr addrspace(3) %address.local198, i64 %address.local161, !dbg !16
  %address.local200 = getelementptr i8, ptr addrspace(3) %address.local199, i64 4288, !dbg !16
  %147 = load <4 x half>, ptr addrspace(3) %address.local200, align 2, !dbg !16
  %148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local139, !dbg !16
  %address.local202 = getelementptr i8, ptr addrspace(3) %address.local201, i64 1024, !dbg !16
  %149 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %147, <4 x float> %148) #9, !dbg !16
  %address.local203 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local152, !dbg !16
  %address.local204 = getelementptr i8, ptr addrspace(3) %address.local203, i64 5120, !dbg !16
  %151 = load <4 x half>, ptr addrspace(3) %address.local204, align 2, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %151, <4 x float> %152) #9, !dbg !16
  %address.local205 = add nuw nsw i64 %address.local155, 512, !dbg !16
  %address.local206 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local205, !dbg !16
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) %address.local206, i64 %address.local158, !dbg !16
  %address.local208 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local161, !dbg !16
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) %address.local208, i64 %address.local164, !dbg !16
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local209, i64 %address.local168, !dbg !16
  %address.local211 = getelementptr i8, ptr addrspace(3) %address.local210, i64 64, !dbg !16
  %154 = load <4 x half>, ptr addrspace(3) %address.local211, align 2, !dbg !16
  %address.local212 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local205, !dbg !16
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local212, i64 %address.local164, !dbg !16
  %address.local214 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local168, !dbg !16
  %address.local215 = getelementptr [2 x i8], ptr addrspace(3) %address.local214, i64 %address.local158, !dbg !16
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local215, i64 %address.local161, !dbg !16
  %address.local217 = getelementptr i8, ptr addrspace(3) %address.local216, i64 4160, !dbg !16
  %155 = load <4 x half>, ptr addrspace(3) %address.local217, align 2, !dbg !16
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local218 = add nuw nsw i64 %address.local155, 512, !dbg !16
  %address.local219 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local218, !dbg !16
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local219, i64 %address.local158, !dbg !16
  %address.local221 = getelementptr [2 x i8], ptr addrspace(3) %address.local220, i64 %address.local161, !dbg !16
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) %address.local221, i64 %address.local164, !dbg !16
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local168, !dbg !16
  %address.local224 = getelementptr i8, ptr addrspace(3) %address.local223, i64 128, !dbg !16
  %157 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %155, <4 x float> %156) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local218, !dbg !16
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local164, !dbg !16
  %address.local227 = getelementptr [2 x i8], ptr addrspace(3) %address.local226, i64 %address.local168, !dbg !16
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) %address.local227, i64 %address.local158, !dbg !16
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local161, !dbg !16
  %address.local230 = getelementptr i8, ptr addrspace(3) %address.local229, i64 4224, !dbg !16
  %159 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %157, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %159, <4 x float> %160) #9, !dbg !16
  %address.local231 = add nuw nsw i64 %address.local155, 512, !dbg !16
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local231, !dbg !16
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) %address.local232, i64 %address.local158, !dbg !16
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) %address.local233, i64 %address.local161, !dbg !16
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local164, !dbg !16
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local168, !dbg !16
  %address.local237 = getelementptr i8, ptr addrspace(3) %address.local236, i64 192, !dbg !16
  %162 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local231, !dbg !16
  %address.local239 = getelementptr [2 x i8], ptr addrspace(3) %address.local238, i64 %address.local164, !dbg !16
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) %address.local239, i64 %address.local168, !dbg !16
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local158, !dbg !16
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local161, !dbg !16
  %address.local243 = getelementptr i8, ptr addrspace(3) %address.local242, i64 4288, !dbg !16
  %163 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !16
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %132, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !16
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %133, <4 x half> %163, <4 x float> %164) #9, !dbg !16
  %166 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local84), !dbg !11
  %167 = zext i32 %166 to i64, !dbg !11
  %168 = inttoptr i64 %167 to ptr addrspace(3), !dbg !11
  %address.local244 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local89, !dbg !11
  %169 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %170 = ptrtoint ptr addrspace(1) %address.local244 to i64, !dbg !11
  %171 = sub i64 %170, %169, !dbg !11
  %dma.buffer.offset687 = trunc i64 %171 to i32, !dbg !11
  %172 = trunc i64 %169 to i32, !dbg !11
  %173 = insertelement <4 x i32> poison, i32 %172, i64 0, !dbg !11
  %174 = lshr i64 %169, 32, !dbg !11
  %175 = trunc i64 %174 to i32, !dbg !11
  %176 = insertelement <4 x i32> %173, i32 %175, i64 1, !dbg !11
  %177 = insertelement <4 x i32> %176, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc688 = insertelement <4 x i32> %177, i32 131072, i64 3, !dbg !11
  %178 = ptrtoint ptr addrspace(3) %168 to i32, !dbg !11
  %dma.buffer.lds689 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %178), !dbg !11
  %179 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds689, i32 %dma.buffer.offset687, <4 x i32> %dma.buffer.rsrc688) #10, !dbg !11
  %180 = add nuw i64 %58, 1024, !dbg !15
  %181 = icmp slt i64 %180, 0, !dbg !15
  %182 = sub nuw i64 -1025, %58, !dbg !15
  %183 = select i1 %181, i64 %182, i64 %180, !dbg !15
  %184 = sdiv i64 %183, 4096, !dbg !15
  %.lobit = ashr i64 %180, 63, !dbg !15
  %185 = xor i64 %184, %.lobit, !dbg !15
  %186 = add nsw i64 %185, %15, !dbg !15
  %187 = add nsw i64 %92, %52, !dbg !15
  %188 = add nsw i64 %187, 32, !dbg !15
  %189 = icmp slt i64 %187, -32, !dbg !15
  %190 = sub nsw i64 -33, %187, !dbg !15
  %191 = select i1 %189, i64 %190, i64 %188, !dbg !15
  %192 = sdiv i64 %191, 128, !dbg !15
  %193 = sext i1 %189 to i64, !dbg !15
  %194 = xor i64 %192, %193, !dbg !15
  %195 = shl nsw i64 %194, 7, !dbg !15
  %196 = sub i64 %52, %195, !dbg !15
  %197 = shl i64 %186, 19, !dbg !15
  %198 = and i64 %197, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %199 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %198, !dbg !15
  %200 = getelementptr [2 x i8], ptr addrspace(1) %199, i64 %92, !dbg !15
  %201 = getelementptr [2 x i8], ptr addrspace(1) %200, i64 %196, !dbg !15
  %202 = getelementptr i8, ptr addrspace(1) %201, i64 64, !dbg !15
  %203 = getelementptr [2 x i8], ptr addrspace(1) %202, i64 %98, !dbg !15
  %204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !15
  %205 = zext i32 %204 to i64, !dbg !15
  %206 = inttoptr i64 %205 to ptr addrspace(3), !dbg !15
  %address.local245 = getelementptr [2 x i8], ptr addrspace(1) %203, i64 %address.local99, !dbg !15
  %207 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %208 = ptrtoint ptr addrspace(1) %address.local245 to i64, !dbg !15
  %209 = sub i64 %208, %207, !dbg !15
  %dma.buffer.offset684 = trunc i64 %209 to i32, !dbg !15
  %210 = trunc i64 %207 to i32, !dbg !15
  %211 = insertelement <4 x i32> poison, i32 %210, i64 0, !dbg !15
  %212 = lshr i64 %207, 32, !dbg !15
  %213 = trunc i64 %212 to i32, !dbg !15
  %214 = insertelement <4 x i32> %211, i32 %213, i64 1, !dbg !15
  %215 = insertelement <4 x i32> %214, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc685 = insertelement <4 x i32> %215, i32 131072, i64 3, !dbg !15
  %216 = ptrtoint ptr addrspace(3) %206 to i32, !dbg !15
  %dma.buffer.lds686 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %216), !dbg !15
  %217 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds686, i32 %dma.buffer.offset684, <4 x i32> %dma.buffer.rsrc685) #10, !dbg !15
  %218 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local106), !dbg !15
  %219 = zext i32 %218 to i64, !dbg !15
  %220 = inttoptr i64 %219 to ptr addrspace(3), !dbg !15
  %address.local246 = getelementptr [2 x i8], ptr addrspace(1) %203, i64 %address.local111, !dbg !15
  %221 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %222 = ptrtoint ptr addrspace(1) %address.local246 to i64, !dbg !15
  %223 = sub i64 %222, %221, !dbg !15
  %dma.buffer.offset681 = trunc i64 %223 to i32, !dbg !15
  %224 = trunc i64 %221 to i32, !dbg !15
  %225 = insertelement <4 x i32> poison, i32 %224, i64 0, !dbg !15
  %226 = lshr i64 %221, 32, !dbg !15
  %227 = trunc i64 %226 to i32, !dbg !15
  %228 = insertelement <4 x i32> %225, i32 %227, i64 1, !dbg !15
  %229 = insertelement <4 x i32> %228, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc682 = insertelement <4 x i32> %229, i32 131072, i64 3, !dbg !15
  %230 = ptrtoint ptr addrspace(3) %220 to i32, !dbg !15
  %dma.buffer.lds683 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %230), !dbg !15
  %231 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds683, i32 %dma.buffer.offset681, <4 x i32> %dma.buffer.rsrc682) #10, !dbg !15
  %232 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %233 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %234 = fmul <4 x half> %232, splat (half 0xH3015), !dbg !12
  %235 = fmul <4 x half> %233, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %236 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %236, <4 x float> %138) #9, !dbg !16
  %238 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %239 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %238, <4 x float> %237) #9, !dbg !16
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %239, <4 x float> %142) #9, !dbg !16
  %242 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %243 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %242, <4 x float> %241) #9, !dbg !16
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %243, <4 x float> %145) #9, !dbg !16
  %246 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !16
  %247 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !16
  %248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %246, <4 x float> %245) #9, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %247, <4 x float> %150) #9, !dbg !16
  %250 = load <4 x half>, ptr addrspace(3) %address.local200, align 2, !dbg !16
  %251 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %250, <4 x float> %249) #9, !dbg !16
  %253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %251, <4 x float> %153) #9, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %address.local204, align 2, !dbg !16
  %255 = load <4 x half>, ptr addrspace(3) %address.local211, align 2, !dbg !16
  %256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %254, <4 x float> %253) #9, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %255, <4 x float> %158) #9, !dbg !16
  %258 = load <4 x half>, ptr addrspace(3) %address.local217, align 2, !dbg !16
  %259 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %258, <4 x float> %257) #9, !dbg !16
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %259, <4 x float> %161) #9, !dbg !16
  %262 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !16
  %263 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %262, <4 x float> %261) #9, !dbg !16
  %265 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %234, <4 x half> %263, <4 x float> %165) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %265, <4 x float> %266) #9, !dbg !16
  %268 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local84), !dbg !11
  %269 = zext i32 %268 to i64, !dbg !11
  %270 = inttoptr i64 %269 to ptr addrspace(3), !dbg !11
  %address.local247 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local89, !dbg !11
  %271 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %272 = ptrtoint ptr addrspace(1) %address.local247 to i64, !dbg !11
  %273 = sub i64 %272, %271, !dbg !11
  %dma.buffer.offset678 = trunc i64 %273 to i32, !dbg !11
  %274 = trunc i64 %271 to i32, !dbg !11
  %275 = insertelement <4 x i32> poison, i32 %274, i64 0, !dbg !11
  %276 = lshr i64 %271, 32, !dbg !11
  %277 = trunc i64 %276 to i32, !dbg !11
  %278 = insertelement <4 x i32> %275, i32 %277, i64 1, !dbg !11
  %279 = insertelement <4 x i32> %278, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc679 = insertelement <4 x i32> %279, i32 131072, i64 3, !dbg !11
  %280 = ptrtoint ptr addrspace(3) %270 to i32, !dbg !11
  %dma.buffer.lds680 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %280), !dbg !11
  %281 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds680, i32 %dma.buffer.offset678, <4 x i32> %dma.buffer.rsrc679) #10, !dbg !11
  %282 = add nuw i64 %58, 2048, !dbg !15
  %283 = icmp slt i64 %282, 0, !dbg !15
  %284 = sub nuw i64 -2049, %58, !dbg !15
  %285 = select i1 %283, i64 %284, i64 %282, !dbg !15
  %286 = sdiv i64 %285, 4096, !dbg !15
  %.lobit2 = ashr i64 %282, 63, !dbg !15
  %287 = xor i64 %286, %.lobit2, !dbg !15
  %288 = add nsw i64 %287, %15, !dbg !15
  %289 = add nsw i64 %187, 64, !dbg !15
  %290 = icmp slt i64 %187, -64, !dbg !15
  %291 = sub nsw i64 -65, %187, !dbg !15
  %292 = select i1 %290, i64 %291, i64 %289, !dbg !15
  %293 = sdiv i64 %292, 128, !dbg !15
  %294 = sext i1 %290 to i64, !dbg !15
  %295 = xor i64 %293, %294, !dbg !15
  %296 = shl nsw i64 %295, 7, !dbg !15
  %297 = sub i64 %52, %296, !dbg !15
  %298 = shl i64 %288, 19, !dbg !15
  %299 = and i64 %298, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %300 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %299, !dbg !15
  %301 = getelementptr [2 x i8], ptr addrspace(1) %300, i64 %92, !dbg !15
  %302 = getelementptr [2 x i8], ptr addrspace(1) %301, i64 %297, !dbg !15
  %303 = getelementptr i8, ptr addrspace(1) %302, i64 128, !dbg !15
  %304 = getelementptr [2 x i8], ptr addrspace(1) %303, i64 %98, !dbg !15
  %305 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !15
  %306 = zext i32 %305 to i64, !dbg !15
  %307 = inttoptr i64 %306 to ptr addrspace(3), !dbg !15
  %address.local248 = getelementptr [2 x i8], ptr addrspace(1) %304, i64 %address.local99, !dbg !15
  %308 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %309 = ptrtoint ptr addrspace(1) %address.local248 to i64, !dbg !15
  %310 = sub i64 %309, %308, !dbg !15
  %dma.buffer.offset675 = trunc i64 %310 to i32, !dbg !15
  %311 = trunc i64 %308 to i32, !dbg !15
  %312 = insertelement <4 x i32> poison, i32 %311, i64 0, !dbg !15
  %313 = lshr i64 %308, 32, !dbg !15
  %314 = trunc i64 %313 to i32, !dbg !15
  %315 = insertelement <4 x i32> %312, i32 %314, i64 1, !dbg !15
  %316 = insertelement <4 x i32> %315, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc676 = insertelement <4 x i32> %316, i32 131072, i64 3, !dbg !15
  %317 = ptrtoint ptr addrspace(3) %307 to i32, !dbg !15
  %dma.buffer.lds677 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %317), !dbg !15
  %318 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds677, i32 %dma.buffer.offset675, <4 x i32> %dma.buffer.rsrc676) #10, !dbg !15
  %319 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local106), !dbg !15
  %320 = zext i32 %319 to i64, !dbg !15
  %321 = inttoptr i64 %320 to ptr addrspace(3), !dbg !15
  %address.local249 = getelementptr [2 x i8], ptr addrspace(1) %304, i64 %address.local111, !dbg !15
  %322 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %323 = ptrtoint ptr addrspace(1) %address.local249 to i64, !dbg !15
  %324 = sub i64 %323, %322, !dbg !15
  %dma.buffer.offset672 = trunc i64 %324 to i32, !dbg !15
  %325 = trunc i64 %322 to i32, !dbg !15
  %326 = insertelement <4 x i32> poison, i32 %325, i64 0, !dbg !15
  %327 = lshr i64 %322, 32, !dbg !15
  %328 = trunc i64 %327 to i32, !dbg !15
  %329 = insertelement <4 x i32> %326, i32 %328, i64 1, !dbg !15
  %330 = insertelement <4 x i32> %329, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc673 = insertelement <4 x i32> %330, i32 131072, i64 3, !dbg !15
  %331 = ptrtoint ptr addrspace(3) %321 to i32, !dbg !15
  %dma.buffer.lds674 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %331), !dbg !15
  %332 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds674, i32 %dma.buffer.offset672, <4 x i32> %dma.buffer.rsrc673) #10, !dbg !15
  %333 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %334 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %335 = fmul <4 x half> %333, splat (half 0xH3015), !dbg !12
  %336 = fmul <4 x half> %334, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %337 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %337, <4 x float> %240) #9, !dbg !16
  %339 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %340 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %339, <4 x float> %338) #9, !dbg !16
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %340, <4 x float> %244) #9, !dbg !16
  %343 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %344 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %343, <4 x float> %342) #9, !dbg !16
  %346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %344, <4 x float> %248) #9, !dbg !16
  %347 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !16
  %348 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %347, <4 x float> %346) #9, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %348, <4 x float> %252) #9, !dbg !16
  %351 = load <4 x half>, ptr addrspace(3) %address.local200, align 2, !dbg !16
  %352 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !16
  %353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %351, <4 x float> %350) #9, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %352, <4 x float> %256) #9, !dbg !16
  %355 = load <4 x half>, ptr addrspace(3) %address.local204, align 2, !dbg !16
  %356 = load <4 x half>, ptr addrspace(3) %address.local211, align 2, !dbg !16
  %357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %355, <4 x float> %354) #9, !dbg !16
  %358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %356, <4 x float> %260) #9, !dbg !16
  %359 = load <4 x half>, ptr addrspace(3) %address.local217, align 2, !dbg !16
  %360 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !16
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %359, <4 x float> %358) #9, !dbg !16
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %360, <4 x float> %264) #9, !dbg !16
  %363 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !16
  %364 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %363, <4 x float> %362) #9, !dbg !16
  %366 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %335, <4 x half> %364, <4 x float> %267) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %336, <4 x half> %366, <4 x float> %367) #9, !dbg !16
  %369 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local84), !dbg !11
  %370 = zext i32 %369 to i64, !dbg !11
  %371 = inttoptr i64 %370 to ptr addrspace(3), !dbg !11
  %address.local250 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local89, !dbg !11
  %372 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %373 = ptrtoint ptr addrspace(1) %address.local250 to i64, !dbg !11
  %374 = sub i64 %373, %372, !dbg !11
  %dma.buffer.offset669 = trunc i64 %374 to i32, !dbg !11
  %375 = trunc i64 %372 to i32, !dbg !11
  %376 = insertelement <4 x i32> poison, i32 %375, i64 0, !dbg !11
  %377 = lshr i64 %372, 32, !dbg !11
  %378 = trunc i64 %377 to i32, !dbg !11
  %379 = insertelement <4 x i32> %376, i32 %378, i64 1, !dbg !11
  %380 = insertelement <4 x i32> %379, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc670 = insertelement <4 x i32> %380, i32 131072, i64 3, !dbg !11
  %381 = ptrtoint ptr addrspace(3) %371 to i32, !dbg !11
  %dma.buffer.lds671 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %381), !dbg !11
  %382 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds671, i32 %dma.buffer.offset669, <4 x i32> %dma.buffer.rsrc670) #10, !dbg !11
  %383 = add nuw i64 %58, 3072, !dbg !15
  %384 = icmp slt i64 %383, 0, !dbg !15
  %385 = sub nuw i64 -3073, %58, !dbg !15
  %386 = select i1 %384, i64 %385, i64 %383, !dbg !15
  %387 = sdiv i64 %386, 4096, !dbg !15
  %.lobit3 = ashr i64 %383, 63, !dbg !15
  %388 = xor i64 %387, %.lobit3, !dbg !15
  %389 = add nsw i64 %388, %15, !dbg !15
  %390 = add nsw i64 %187, 96, !dbg !15
  %391 = icmp slt i64 %187, -96, !dbg !15
  %392 = sub nsw i64 -97, %187, !dbg !15
  %393 = select i1 %391, i64 %392, i64 %390, !dbg !15
  %394 = sdiv i64 %393, 128, !dbg !15
  %395 = sext i1 %391 to i64, !dbg !15
  %396 = xor i64 %394, %395, !dbg !15
  %397 = shl nsw i64 %396, 7, !dbg !15
  %398 = sub i64 %52, %397, !dbg !15
  %399 = shl i64 %389, 19, !dbg !15
  %400 = and i64 %399, 16252928, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %401 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %400, !dbg !15
  %402 = getelementptr [2 x i8], ptr addrspace(1) %401, i64 %92, !dbg !15
  %403 = getelementptr [2 x i8], ptr addrspace(1) %402, i64 %398, !dbg !15
  %404 = getelementptr i8, ptr addrspace(1) %403, i64 192, !dbg !15
  %405 = getelementptr [2 x i8], ptr addrspace(1) %404, i64 %98, !dbg !15
  %406 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local94), !dbg !15
  %407 = zext i32 %406 to i64, !dbg !15
  %408 = inttoptr i64 %407 to ptr addrspace(3), !dbg !15
  %address.local251 = getelementptr [2 x i8], ptr addrspace(1) %405, i64 %address.local99, !dbg !15
  %409 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %410 = ptrtoint ptr addrspace(1) %address.local251 to i64, !dbg !15
  %411 = sub i64 %410, %409, !dbg !15
  %dma.buffer.offset666 = trunc i64 %411 to i32, !dbg !15
  %412 = trunc i64 %409 to i32, !dbg !15
  %413 = insertelement <4 x i32> poison, i32 %412, i64 0, !dbg !15
  %414 = lshr i64 %409, 32, !dbg !15
  %415 = trunc i64 %414 to i32, !dbg !15
  %416 = insertelement <4 x i32> %413, i32 %415, i64 1, !dbg !15
  %417 = insertelement <4 x i32> %416, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc667 = insertelement <4 x i32> %417, i32 131072, i64 3, !dbg !15
  %418 = ptrtoint ptr addrspace(3) %408 to i32, !dbg !15
  %dma.buffer.lds668 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %418), !dbg !15
  %419 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds668, i32 %dma.buffer.offset666, <4 x i32> %dma.buffer.rsrc667) #10, !dbg !15
  %420 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local106), !dbg !15
  %421 = zext i32 %420 to i64, !dbg !15
  %422 = inttoptr i64 %421 to ptr addrspace(3), !dbg !15
  %address.local252 = getelementptr [2 x i8], ptr addrspace(1) %405, i64 %address.local111, !dbg !15
  %423 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !15
  %424 = ptrtoint ptr addrspace(1) %address.local252 to i64, !dbg !15
  %425 = sub i64 %424, %423, !dbg !15
  %dma.buffer.offset663 = trunc i64 %425 to i32, !dbg !15
  %426 = trunc i64 %423 to i32, !dbg !15
  %427 = insertelement <4 x i32> poison, i32 %426, i64 0, !dbg !15
  %428 = lshr i64 %423, 32, !dbg !15
  %429 = trunc i64 %428 to i32, !dbg !15
  %430 = insertelement <4 x i32> %427, i32 %429, i64 1, !dbg !15
  %431 = insertelement <4 x i32> %430, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc664 = insertelement <4 x i32> %431, i32 131072, i64 3, !dbg !15
  %432 = ptrtoint ptr addrspace(3) %422 to i32, !dbg !15
  %dma.buffer.lds665 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %432), !dbg !15
  %433 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds665, i32 %dma.buffer.offset663, <4 x i32> %dma.buffer.rsrc664) #10, !dbg !15
  %434 = load <4 x half>, ptr addrspace(3) %address.local125, align 2, !dbg !12
  %435 = load <4 x half>, ptr addrspace(3) %address.local127, align 2, !dbg !12
  %436 = fmul <4 x half> %434, splat (half 0xH3015), !dbg !12
  %437 = fmul <4 x half> %435, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %438 = add nuw nsw i64 %92, %15, !dbg !17
  %439 = shl i64 %438, 19, !dbg !17
  %440 = and i64 %439, 16252928, !dbg !17
  %441 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %440, !dbg !17
  %442 = getelementptr [2 x i8], ptr addrspace(1) %441, i64 %98, !dbg !17
  %address.local253 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local91, !dbg !17
  %address.local254 = ptrtoint ptr addrspace(3) %address.local253 to i64, !dbg !17
  %address.local255 = trunc i64 %address.local254 to i32, !dbg !17
  %443 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local255), !dbg !17
  %444 = zext i32 %443 to i64, !dbg !17
  %445 = inttoptr i64 %444 to ptr addrspace(3), !dbg !17
  %address.local256 = shl nuw nsw i64 %address.local, 5, !dbg !17
  %address.local257 = and i64 %address.local256, 8064, !dbg !17
  %address.local258 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local259 = and i64 %address.local258, 24, !dbg !17
  %address.local260 = or disjoint i64 %address.local257, %address.local259, !dbg !17
  %address.local261 = getelementptr [2 x i8], ptr addrspace(1) %442, i64 %address.local260, !dbg !17
  %446 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %447 = ptrtoint ptr addrspace(1) %address.local261 to i64, !dbg !17
  %448 = sub i64 %447, %446, !dbg !17
  %dma.buffer.offset660 = trunc i64 %448 to i32, !dbg !17
  %449 = trunc i64 %446 to i32, !dbg !17
  %450 = insertelement <4 x i32> poison, i32 %449, i64 0, !dbg !17
  %451 = lshr i64 %446, 32, !dbg !17
  %452 = trunc i64 %451 to i32, !dbg !17
  %453 = insertelement <4 x i32> %450, i32 %452, i64 1, !dbg !17
  %454 = insertelement <4 x i32> %453, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc661 = insertelement <4 x i32> %454, i32 131072, i64 3, !dbg !17
  %455 = ptrtoint ptr addrspace(3) %445 to i32, !dbg !17
  %dma.buffer.lds662 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %455), !dbg !17
  %456 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds662, i32 %dma.buffer.offset660, <4 x i32> %dma.buffer.rsrc661) #10, !dbg !17
  %address.local262 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local103, !dbg !17
  %address.local263 = ptrtoint ptr addrspace(3) %address.local262 to i64, !dbg !17
  %address.local264 = trunc i64 %address.local263 to i32, !dbg !17
  %457 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local264), !dbg !17
  %458 = zext i32 %457 to i64, !dbg !17
  %459 = inttoptr i64 %458 to ptr addrspace(3), !dbg !17
  %address.local265 = shl nuw nsw i64 %address.local101, 5, !dbg !17
  %address.local266 = and i64 %address.local265, 16256, !dbg !17
  %address.local267 = shl nuw nsw i64 %address.local, 3, !dbg !17
  %address.local268 = and i64 %address.local267, 24, !dbg !17
  %address.local269 = or disjoint i64 %address.local266, %address.local268, !dbg !17
  %address.local270 = getelementptr [2 x i8], ptr addrspace(1) %442, i64 %address.local269, !dbg !17
  %460 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %461 = ptrtoint ptr addrspace(1) %address.local270 to i64, !dbg !17
  %462 = sub i64 %461, %460, !dbg !17
  %dma.buffer.offset657 = trunc i64 %462 to i32, !dbg !17
  %463 = trunc i64 %460 to i32, !dbg !17
  %464 = insertelement <4 x i32> poison, i32 %463, i64 0, !dbg !17
  %465 = lshr i64 %460, 32, !dbg !17
  %466 = trunc i64 %465 to i32, !dbg !17
  %467 = insertelement <4 x i32> %464, i32 %466, i64 1, !dbg !17
  %468 = insertelement <4 x i32> %467, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc658 = insertelement <4 x i32> %468, i32 131072, i64 3, !dbg !17
  %469 = ptrtoint ptr addrspace(3) %459 to i32, !dbg !17
  %dma.buffer.lds659 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %469), !dbg !17
  %470 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds659, i32 %dma.buffer.offset657, <4 x i32> %dma.buffer.rsrc658) #10, !dbg !17
  %471 = load <4 x half>, ptr addrspace(3) %address.local140, align 2, !dbg !16
  %472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %471, <4 x float> %341) #9, !dbg !16
  %473 = load <4 x half>, ptr addrspace(3) %address.local154, align 2, !dbg !16
  %474 = load <4 x half>, ptr addrspace(3) %address.local170, align 2, !dbg !16
  %475 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %473, <4 x float> %472) #9, !dbg !16
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %474, <4 x float> %345) #9, !dbg !16
  %477 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !16
  %478 = load <4 x half>, ptr addrspace(3) %address.local182, align 2, !dbg !16
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %477, <4 x float> %476) #9, !dbg !16
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %478, <4 x float> %349) #9, !dbg !16
  %481 = load <4 x half>, ptr addrspace(3) %address.local188, align 2, !dbg !16
  %482 = load <4 x half>, ptr addrspace(3) %address.local194, align 2, !dbg !16
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %481, <4 x float> %480) #9, !dbg !16
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %482, <4 x float> %353) #9, !dbg !16
  %485 = load <4 x half>, ptr addrspace(3) %address.local200, align 2, !dbg !16
  %486 = load <4 x half>, ptr addrspace(3) %address.local202, align 2, !dbg !16
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %485, <4 x float> %484) #9, !dbg !16
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %486, <4 x float> %357) #9, !dbg !16
  %489 = load <4 x half>, ptr addrspace(3) %address.local204, align 2, !dbg !16
  %490 = load <4 x half>, ptr addrspace(3) %address.local211, align 2, !dbg !16
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %489, <4 x float> %488) #9, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %490, <4 x float> %361) #9, !dbg !16
  %493 = load <4 x half>, ptr addrspace(3) %address.local217, align 2, !dbg !16
  %494 = load <4 x half>, ptr addrspace(3) %address.local224, align 2, !dbg !16
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %493, <4 x float> %492) #9, !dbg !16
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %494, <4 x float> %365) #9, !dbg !16
  %497 = load <4 x half>, ptr addrspace(3) %address.local230, align 2, !dbg !16
  %498 = load <4 x half>, ptr addrspace(3) %address.local237, align 2, !dbg !16
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %497, <4 x float> %496) #9, !dbg !16
  %500 = load <4 x half>, ptr addrspace(3) %address.local243, align 2, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %501 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %436, <4 x half> %498, <4 x float> %368) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %502 = or disjoint i64 %75, 127, !dbg !13
  %503 = icmp uge i64 %19, %502, !dbg !13
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %437, <4 x half> %500, <4 x float> %501) #9, !dbg !16
  %505 = and i1 %39, %503, !dbg !13
  br i1 %505, label %506, label %507, !dbg !13

506:                                              ; preds = %74
  br label %582, !dbg !13

507:                                              ; preds = %74
  %508 = add i64 %19, %35, !dbg !13
  %509 = shl nuw nsw i64 %36, 4, !dbg !13
  %510 = add i64 %508, %509, !dbg !13
  %511 = and i32 %34, 240, !dbg !13
  %index.compact = zext nneg i32 %511 to i64, !dbg !13
  %512 = sub i64 %510, %index.compact, !dbg !13
  %513 = add i64 %512, 1, !dbg !22
  %514 = and i64 %35, 48, !dbg !13
  %515 = or disjoint i64 %75, %514, !dbg !13
  %516 = lshr i64 %513, 32, !dbg !23
  %compare.high = trunc i64 %516 to i32, !dbg !23
  %517 = lshr i64 %515, 32, !dbg !23
  %compare.high518 = trunc i64 %517 to i32, !dbg !23
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high518, !dbg !23
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high518, !dbg !23
  %compare.low.base = trunc i64 %513 to i32, !dbg !23
  %compare.low.base519 = trunc i64 %515 to i32, !dbg !23
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base519, !dbg !23
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !23
  %518 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 1) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !23
  %compare.low520 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !23
  %compare.result521 = select i1 %compare.high.equal, i1 %compare.low520, i1 %compare.high.order, !dbg !23
  %519 = select i1 %compare.result521, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base522 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 2) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local523 = or i32 %compare.local.base522, 2, !dbg !23
  %compare.low524 = icmp ugt i32 %compare.low.base, %compare.local523, !dbg !23
  %compare.result525 = select i1 %compare.high.equal, i1 %compare.low524, i1 %compare.high.order, !dbg !23
  %520 = select i1 %compare.result525, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base526 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 3) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local527 = or i32 %compare.local.base526, 3, !dbg !23
  %compare.low528 = icmp ugt i32 %compare.low.base, %compare.local527, !dbg !23
  %compare.result529 = select i1 %compare.high.equal, i1 %compare.low528, i1 %compare.high.order, !dbg !23
  %521 = select i1 %compare.result529, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base530 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 4) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local531 = or i32 %compare.local.base530, 4, !dbg !23
  %compare.low532 = icmp ugt i32 %compare.low.base, %compare.local531, !dbg !23
  %compare.result533 = select i1 %compare.high.equal, i1 %compare.low532, i1 %compare.high.order, !dbg !23
  %522 = select i1 %compare.result533, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base534 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 5) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local535 = or i32 %compare.local.base534, 5, !dbg !23
  %compare.low536 = icmp ugt i32 %compare.low.base, %compare.local535, !dbg !23
  %compare.result537 = select i1 %compare.high.equal, i1 %compare.low536, i1 %compare.high.order, !dbg !23
  %523 = select i1 %compare.result537, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base538 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 6) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local539 = or i32 %compare.local.base538, 6, !dbg !23
  %compare.low540 = icmp ugt i32 %compare.low.base, %compare.local539, !dbg !23
  %compare.result541 = select i1 %compare.high.equal, i1 %compare.low540, i1 %compare.high.order, !dbg !23
  %524 = select i1 %compare.result541, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base542 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 7) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local543 = or i32 %compare.local.base542, 7, !dbg !23
  %compare.low544 = icmp ugt i32 %compare.low.base, %compare.local543, !dbg !23
  %compare.result545 = select i1 %compare.high.equal, i1 %compare.low544, i1 %compare.high.order, !dbg !23
  %525 = select i1 %compare.result545, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base546 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 8) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local547 = or i32 %compare.local.base546, 8, !dbg !23
  %compare.low548 = icmp ugt i32 %compare.low.base, %compare.local547, !dbg !23
  %compare.result549 = select i1 %compare.high.equal, i1 %compare.low548, i1 %compare.high.order, !dbg !23
  %526 = select i1 %compare.result549, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base550 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 9) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local551 = or i32 %compare.local.base550, 9, !dbg !23
  %compare.low552 = icmp ugt i32 %compare.low.base, %compare.local551, !dbg !23
  %compare.result553 = select i1 %compare.high.equal, i1 %compare.low552, i1 %compare.high.order, !dbg !23
  %527 = select i1 %compare.result553, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base554 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 10) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local555 = or i32 %compare.local.base554, 10, !dbg !23
  %compare.low556 = icmp ugt i32 %compare.low.base, %compare.local555, !dbg !23
  %compare.result557 = select i1 %compare.high.equal, i1 %compare.low556, i1 %compare.high.order, !dbg !23
  %528 = select i1 %compare.result557, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base558 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 11) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local559 = or i32 %compare.local.base558, 11, !dbg !23
  %compare.low560 = icmp ugt i32 %compare.low.base, %compare.local559, !dbg !23
  %compare.result561 = select i1 %compare.high.equal, i1 %compare.low560, i1 %compare.high.order, !dbg !23
  %529 = select i1 %compare.result561, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base562 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 12) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local563 = or i32 %compare.local.base562, 12, !dbg !23
  %compare.low564 = icmp ugt i32 %compare.low.base, %compare.local563, !dbg !23
  %compare.result565 = select i1 %compare.high.equal, i1 %compare.low564, i1 %compare.high.order, !dbg !23
  %530 = select i1 %compare.result565, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base566 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 13) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local567 = or i32 %compare.local.base566, 13, !dbg !23
  %compare.low568 = icmp ugt i32 %compare.low.base, %compare.local567, !dbg !23
  %compare.result569 = select i1 %compare.high.equal, i1 %compare.low568, i1 %compare.high.order, !dbg !23
  %531 = select i1 %compare.result569, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base570 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 14) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local571 = or i32 %compare.local.base570, 14, !dbg !23
  %compare.low572 = icmp ugt i32 %compare.low.base, %compare.local571, !dbg !23
  %compare.result573 = select i1 %compare.high.equal, i1 %compare.low572, i1 %compare.high.order, !dbg !23
  %532 = select i1 %compare.result573, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base574 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 15) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local575 = or i32 %compare.local.base574, 15, !dbg !23
  %compare.low576 = icmp ugt i32 %compare.low.base, %compare.local575, !dbg !23
  %compare.result577 = select i1 %compare.high.equal, i1 %compare.low576, i1 %compare.high.order, !dbg !23
  %533 = select i1 %compare.result577, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base578 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 16) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local579 = or i32 %compare.local.base578, 64, !dbg !23
  %compare.low580 = icmp ugt i32 %compare.low.base, %compare.local579, !dbg !23
  %compare.result581 = select i1 %compare.high.equal, i1 %compare.low580, i1 %compare.high.order, !dbg !23
  %534 = select i1 %compare.result581, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base582 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 17) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local583 = or i32 %compare.local.base582, 65, !dbg !23
  %compare.low584 = icmp ugt i32 %compare.low.base, %compare.local583, !dbg !23
  %compare.result585 = select i1 %compare.high.equal, i1 %compare.low584, i1 %compare.high.order, !dbg !23
  %535 = select i1 %compare.result585, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base586 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 18) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local587 = or i32 %compare.local.base586, 66, !dbg !23
  %compare.low588 = icmp ugt i32 %compare.low.base, %compare.local587, !dbg !23
  %compare.result589 = select i1 %compare.high.equal, i1 %compare.low588, i1 %compare.high.order, !dbg !23
  %536 = select i1 %compare.result589, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base590 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 19) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local591 = or i32 %compare.local.base590, 67, !dbg !23
  %compare.low592 = icmp ugt i32 %compare.low.base, %compare.local591, !dbg !23
  %compare.result593 = select i1 %compare.high.equal, i1 %compare.low592, i1 %compare.high.order, !dbg !23
  %537 = select i1 %compare.result593, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base594 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 20) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local595 = or i32 %compare.local.base594, 68, !dbg !23
  %compare.low596 = icmp ugt i32 %compare.low.base, %compare.local595, !dbg !23
  %compare.result597 = select i1 %compare.high.equal, i1 %compare.low596, i1 %compare.high.order, !dbg !23
  %538 = select i1 %compare.result597, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base598 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 21) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local599 = or i32 %compare.local.base598, 69, !dbg !23
  %compare.low600 = icmp ugt i32 %compare.low.base, %compare.local599, !dbg !23
  %compare.result601 = select i1 %compare.high.equal, i1 %compare.low600, i1 %compare.high.order, !dbg !23
  %539 = select i1 %compare.result601, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base602 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 22) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local603 = or i32 %compare.local.base602, 70, !dbg !23
  %compare.low604 = icmp ugt i32 %compare.low.base, %compare.local603, !dbg !23
  %compare.result605 = select i1 %compare.high.equal, i1 %compare.low604, i1 %compare.high.order, !dbg !23
  %540 = select i1 %compare.result605, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base606 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 23) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local607 = or i32 %compare.local.base606, 71, !dbg !23
  %compare.low608 = icmp ugt i32 %compare.low.base, %compare.local607, !dbg !23
  %compare.result609 = select i1 %compare.high.equal, i1 %compare.low608, i1 %compare.high.order, !dbg !23
  %541 = select i1 %compare.result609, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base610 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 24) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local611 = or i32 %compare.local.base610, 72, !dbg !23
  %compare.low612 = icmp ugt i32 %compare.low.base, %compare.local611, !dbg !23
  %compare.result613 = select i1 %compare.high.equal, i1 %compare.low612, i1 %compare.high.order, !dbg !23
  %542 = select i1 %compare.result613, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base614 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 25) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local615 = or i32 %compare.local.base614, 73, !dbg !23
  %compare.low616 = icmp ugt i32 %compare.low.base, %compare.local615, !dbg !23
  %compare.result617 = select i1 %compare.high.equal, i1 %compare.low616, i1 %compare.high.order, !dbg !23
  %543 = select i1 %compare.result617, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base618 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 26) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local619 = or i32 %compare.local.base618, 74, !dbg !23
  %compare.low620 = icmp ugt i32 %compare.low.base, %compare.local619, !dbg !23
  %compare.result621 = select i1 %compare.high.equal, i1 %compare.low620, i1 %compare.high.order, !dbg !23
  %544 = select i1 %compare.result621, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base622 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 27) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local623 = or i32 %compare.local.base622, 75, !dbg !23
  %compare.low624 = icmp ugt i32 %compare.low.base, %compare.local623, !dbg !23
  %compare.result625 = select i1 %compare.high.equal, i1 %compare.low624, i1 %compare.high.order, !dbg !23
  %545 = select i1 %compare.result625, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base626 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 28) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local627 = or i32 %compare.local.base626, 76, !dbg !23
  %compare.low628 = icmp ugt i32 %compare.low.base, %compare.local627, !dbg !23
  %compare.result629 = select i1 %compare.high.equal, i1 %compare.low628, i1 %compare.high.order, !dbg !23
  %546 = select i1 %compare.result629, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base630 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 29) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local631 = or i32 %compare.local.base630, 77, !dbg !23
  %compare.low632 = icmp ugt i32 %compare.low.base, %compare.local631, !dbg !23
  %compare.result633 = select i1 %compare.high.equal, i1 %compare.low632, i1 %compare.high.order, !dbg !23
  %547 = select i1 %compare.result633, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base634 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 30) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local635 = or i32 %compare.local.base634, 78, !dbg !23
  %compare.low636 = icmp ugt i32 %compare.low.base, %compare.local635, !dbg !23
  %compare.result637 = select i1 %compare.high.equal, i1 %compare.low636, i1 %compare.high.order, !dbg !23
  %548 = select i1 %compare.result637, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %compare.local.base638 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base519, i32 31) #9, !dbg !23, !frisk.register_lifetimes !1
  %compare.local639 = or i32 %compare.local.base638, 79, !dbg !23
  %compare.low640 = icmp ugt i32 %compare.low.base, %compare.local639, !dbg !23
  %compare.result641 = select i1 %compare.high.equal, i1 %compare.low640, i1 %compare.high.order, !dbg !23
  %549 = select i1 %compare.result641, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !24
  %550 = insertelement <32 x float> poison, float %518, i64 0, !dbg !13
  %551 = insertelement <32 x float> %550, float %519, i64 1, !dbg !13
  %552 = insertelement <32 x float> %551, float %520, i64 2, !dbg !13
  %553 = insertelement <32 x float> %552, float %521, i64 3, !dbg !13
  %554 = insertelement <32 x float> %553, float %522, i64 4, !dbg !13
  %555 = insertelement <32 x float> %554, float %523, i64 5, !dbg !13
  %556 = insertelement <32 x float> %555, float %524, i64 6, !dbg !13
  %557 = insertelement <32 x float> %556, float %525, i64 7, !dbg !13
  %558 = insertelement <32 x float> %557, float %526, i64 8, !dbg !13
  %559 = insertelement <32 x float> %558, float %527, i64 9, !dbg !13
  %560 = insertelement <32 x float> %559, float %528, i64 10, !dbg !13
  %561 = insertelement <32 x float> %560, float %529, i64 11, !dbg !13
  %562 = insertelement <32 x float> %561, float %530, i64 12, !dbg !13
  %563 = insertelement <32 x float> %562, float %531, i64 13, !dbg !13
  %564 = insertelement <32 x float> %563, float %532, i64 14, !dbg !13
  %565 = insertelement <32 x float> %564, float %533, i64 15, !dbg !13
  %566 = insertelement <32 x float> %565, float %534, i64 16, !dbg !13
  %567 = insertelement <32 x float> %566, float %535, i64 17, !dbg !13
  %568 = insertelement <32 x float> %567, float %536, i64 18, !dbg !13
  %569 = insertelement <32 x float> %568, float %537, i64 19, !dbg !13
  %570 = insertelement <32 x float> %569, float %538, i64 20, !dbg !13
  %571 = insertelement <32 x float> %570, float %539, i64 21, !dbg !13
  %572 = insertelement <32 x float> %571, float %540, i64 22, !dbg !13
  %573 = insertelement <32 x float> %572, float %541, i64 23, !dbg !13
  %574 = insertelement <32 x float> %573, float %542, i64 24, !dbg !13
  %575 = insertelement <32 x float> %574, float %543, i64 25, !dbg !13
  %576 = insertelement <32 x float> %575, float %544, i64 26, !dbg !13
  %577 = insertelement <32 x float> %576, float %545, i64 27, !dbg !13
  %578 = insertelement <32 x float> %577, float %546, i64 28, !dbg !13
  %579 = insertelement <32 x float> %578, float %547, i64 29, !dbg !13
  %580 = insertelement <32 x float> %579, float %548, i64 30, !dbg !13
  %581 = insertelement <32 x float> %580, float %549, i64 31, !dbg !13
  br label %582, !dbg !13

582:                                              ; preds = %506, %507
  %583 = phi <32 x float> [ %581, %507 ], [ zeroinitializer, %506 ], !dbg !13
  br label %584, !dbg !13

584:                                              ; preds = %582
  %585 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %586 = fadd <4 x float> %475, %585, !dbg !25
  %587 = extractelement <4 x float> %586, i64 0, !dbg !25
  %588 = extractelement <4 x float> %586, i64 1, !dbg !25
  %589 = extractelement <4 x float> %586, i64 2, !dbg !25
  %590 = extractelement <4 x float> %586, i64 3, !dbg !25
  %591 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %592 = fadd <4 x float> %479, %591, !dbg !25
  %593 = extractelement <4 x float> %592, i64 0, !dbg !25
  %594 = extractelement <4 x float> %592, i64 1, !dbg !25
  %595 = extractelement <4 x float> %592, i64 2, !dbg !25
  %596 = extractelement <4 x float> %592, i64 3, !dbg !25
  %597 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !25
  %598 = fadd <4 x float> %483, %597, !dbg !25
  %599 = extractelement <4 x float> %598, i64 0, !dbg !25
  %600 = extractelement <4 x float> %598, i64 1, !dbg !25
  %601 = extractelement <4 x float> %598, i64 2, !dbg !25
  %602 = extractelement <4 x float> %598, i64 3, !dbg !25
  %603 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !25
  %604 = fadd <4 x float> %487, %603, !dbg !25
  %605 = extractelement <4 x float> %604, i64 0, !dbg !25
  %606 = extractelement <4 x float> %604, i64 1, !dbg !25
  %607 = extractelement <4 x float> %604, i64 2, !dbg !25
  %608 = extractelement <4 x float> %604, i64 3, !dbg !25
  %609 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !25
  %610 = fadd <4 x float> %491, %609, !dbg !25
  %611 = extractelement <4 x float> %610, i64 0, !dbg !25
  %612 = extractelement <4 x float> %610, i64 1, !dbg !25
  %613 = extractelement <4 x float> %610, i64 2, !dbg !25
  %614 = extractelement <4 x float> %610, i64 3, !dbg !25
  %615 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !25
  %616 = fadd <4 x float> %495, %615, !dbg !25
  %617 = extractelement <4 x float> %616, i64 0, !dbg !25
  %618 = extractelement <4 x float> %616, i64 1, !dbg !25
  %619 = extractelement <4 x float> %616, i64 2, !dbg !25
  %620 = extractelement <4 x float> %616, i64 3, !dbg !25
  %621 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !25
  %622 = fadd <4 x float> %499, %621, !dbg !25
  %623 = extractelement <4 x float> %622, i64 0, !dbg !25
  %624 = extractelement <4 x float> %622, i64 1, !dbg !25
  %625 = extractelement <4 x float> %622, i64 2, !dbg !25
  %626 = extractelement <4 x float> %622, i64 3, !dbg !25
  %627 = shufflevector <32 x float> %583, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !25
  %628 = fadd <4 x float> %504, %627, !dbg !25
  %629 = extractelement <4 x float> %628, i64 0, !dbg !25
  %630 = extractelement <4 x float> %628, i64 1, !dbg !25
  %631 = extractelement <4 x float> %628, i64 2, !dbg !25
  %632 = extractelement <4 x float> %628, i64 3, !dbg !25
  %633 = call float @llvm.maxnum.f32(float %587, float 0xFFF0000000000000), !dbg !26
  %634 = call float @llvm.maxnum.f32(float %633, float %588), !dbg !26
  %635 = call float @llvm.maxnum.f32(float %634, float %589), !dbg !26
  %636 = call float @llvm.maxnum.f32(float %635, float %590), !dbg !26
  %637 = call float @llvm.maxnum.f32(float %636, float %593), !dbg !26
  %638 = call float @llvm.maxnum.f32(float %637, float %594), !dbg !26
  %639 = call float @llvm.maxnum.f32(float %638, float %595), !dbg !26
  %640 = call float @llvm.maxnum.f32(float %639, float %596), !dbg !26
  %641 = call float @llvm.maxnum.f32(float %640, float %599), !dbg !26
  %642 = call float @llvm.maxnum.f32(float %641, float %600), !dbg !26
  %643 = call float @llvm.maxnum.f32(float %642, float %601), !dbg !26
  %644 = call float @llvm.maxnum.f32(float %643, float %602), !dbg !26
  %645 = call float @llvm.maxnum.f32(float %644, float %605), !dbg !26
  %646 = call float @llvm.maxnum.f32(float %645, float %606), !dbg !26
  %647 = call float @llvm.maxnum.f32(float %646, float %607), !dbg !26
  %648 = call float @llvm.maxnum.f32(float %647, float %608), !dbg !26
  %649 = call float @llvm.maxnum.f32(float %648, float %611), !dbg !26
  %650 = call float @llvm.maxnum.f32(float %649, float %612), !dbg !26
  %651 = call float @llvm.maxnum.f32(float %650, float %613), !dbg !26
  %652 = call float @llvm.maxnum.f32(float %651, float %614), !dbg !26
  %653 = call float @llvm.maxnum.f32(float %652, float %617), !dbg !26
  %654 = call float @llvm.maxnum.f32(float %653, float %618), !dbg !26
  %655 = call float @llvm.maxnum.f32(float %654, float %619), !dbg !26
  %656 = call float @llvm.maxnum.f32(float %655, float %620), !dbg !26
  %657 = call float @llvm.maxnum.f32(float %656, float %623), !dbg !26
  %658 = call float @llvm.maxnum.f32(float %657, float %624), !dbg !26
  %659 = call float @llvm.maxnum.f32(float %658, float %625), !dbg !26
  %660 = call float @llvm.maxnum.f32(float %659, float %626), !dbg !26
  %661 = call float @llvm.maxnum.f32(float %660, float %629), !dbg !26
  %662 = call float @llvm.maxnum.f32(float %661, float %630), !dbg !26
  %663 = call float @llvm.maxnum.f32(float %662, float %631), !dbg !26
  %664 = call float @llvm.maxnum.f32(float %663, float %632), !dbg !26
  %665 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !26
  %666 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %665), !dbg !26
  %667 = add nuw nsw i32 %666, 64, !dbg !26
  %668 = xor i32 %666, 16, !dbg !26
  %669 = and i32 %667, 192, !dbg !26
  %670 = icmp samesign ult i32 %668, %669, !dbg !26
  %671 = select i1 %670, i32 %668, i32 %666, !dbg !26
  %672 = shl nuw nsw i32 %671, 2, !dbg !26
  %673 = bitcast float %664 to i32, !dbg !26
  %674 = call i32 @llvm.amdgcn.ds.bpermute(i32 %672, i32 %673), !dbg !26
  %675 = bitcast i32 %674 to float, !dbg !26
  %676 = call float @llvm.maxnum.f32(float %664, float %675), !dbg !26
  %677 = xor i32 %666, 32, !dbg !26
  %678 = icmp samesign ult i32 %677, %669, !dbg !26
  %679 = select i1 %678, i32 %677, i32 %666, !dbg !26
  %680 = shl nuw nsw i32 %679, 2, !dbg !26
  %681 = bitcast float %676 to i32, !dbg !26
  %682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %681), !dbg !26
  %683 = bitcast i32 %682 to float, !dbg !26
  %684 = call float @llvm.maxnum.f32(float %676, float %683), !dbg !26
  %685 = insertelement <1 x float> poison, float %684, i64 0, !dbg !26
  %686 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %685, <1 x float> %63), !dbg !26
  %687 = shufflevector <1 x float> %686, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %688 = fsub <4 x float> %586, %687, !dbg !27
  %689 = fsub <4 x float> %592, %687, !dbg !27
  %690 = fsub <4 x float> %598, %687, !dbg !27
  %691 = fsub <4 x float> %604, %687, !dbg !27
  %692 = fsub <4 x float> %610, %687, !dbg !27
  %693 = fsub <4 x float> %616, %687, !dbg !27
  %694 = fsub <4 x float> %622, %687, !dbg !27
  %695 = fsub <4 x float> %628, %687, !dbg !27
  %696 = fsub <1 x float> %63, %686, !dbg !28
  %697 = extractelement <1 x float> %696, i64 0, !dbg !29
  %698 = call float @llvm.amdgcn.exp2.f32(float %697), !dbg !29
  %699 = insertelement <4 x float> poison, float %698, i64 0, !dbg !34
  %700 = shufflevector <4 x float> %699, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %701 = fmul <4 x float> %71, %700, !dbg !34
  %702 = fmul <4 x float> %70, %700, !dbg !34
  %703 = fmul <4 x float> %68, %700, !dbg !34
  %704 = fmul <4 x float> %67, %700, !dbg !34
  %705 = fmul <4 x float> %65, %700, !dbg !34
  %706 = fmul <4 x float> %64, %700, !dbg !34
  %707 = fmul <4 x float> %60, %700, !dbg !34
  %708 = fmul <4 x float> %59, %700, !dbg !34
  %709 = extractelement <1 x float> %62, i64 0, !dbg !30
  %710 = fmul float %709, %698, !dbg !30
  %711 = extractelement <4 x float> %688, i64 0, !dbg !31
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !31
  %713 = insertelement <4 x float> poison, float %712, i64 0, !dbg !33
  %714 = extractelement <4 x float> %688, i64 1, !dbg !31
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !31
  %716 = insertelement <4 x float> %713, float %715, i64 1, !dbg !33
  %717 = fadd float %712, %715, !dbg !32
  %718 = extractelement <4 x float> %688, i64 2, !dbg !31
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !31
  %720 = insertelement <4 x float> %716, float %719, i64 2, !dbg !33
  %721 = fadd float %717, %719, !dbg !32
  %722 = extractelement <4 x float> %688, i64 3, !dbg !31
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !31
  %724 = insertelement <4 x float> %720, float %723, i64 3, !dbg !33
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !33, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !33
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !32, !frisk.register_lifetimes !1
  %726 = fadd float %721, %723, !dbg !32
  %727 = extractelement <4 x float> %689, i64 0, !dbg !31
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !31
  %729 = insertelement <4 x float> poison, float %728, i64 0, !dbg !33
  %730 = fadd float %726, %728, !dbg !32
  %731 = extractelement <4 x float> %689, i64 1, !dbg !31
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !31
  %733 = insertelement <4 x float> %729, float %732, i64 1, !dbg !33
  %734 = fadd float %730, %732, !dbg !32
  %735 = extractelement <4 x float> %689, i64 2, !dbg !31
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !31
  %737 = insertelement <4 x float> %733, float %736, i64 2, !dbg !33
  %738 = fadd float %734, %736, !dbg !32
  %739 = extractelement <4 x float> %689, i64 3, !dbg !31
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !31
  %741 = insertelement <4 x float> %737, float %740, i64 3, !dbg !33
  %math.result66 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !33, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result66 to <4 x half>, !dbg !33
  %math.packed67 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !32, !frisk.register_lifetimes !1
  %743 = fadd float %738, %740, !dbg !32
  %744 = extractelement <4 x float> %690, i64 0, !dbg !31
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !31
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !33
  %747 = fadd float %743, %745, !dbg !32
  %748 = extractelement <4 x float> %690, i64 1, !dbg !31
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !31
  %750 = insertelement <4 x float> %746, float %749, i64 1, !dbg !33
  %751 = fadd float %747, %749, !dbg !32
  %752 = extractelement <4 x float> %690, i64 2, !dbg !31
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !31
  %754 = insertelement <4 x float> %750, float %753, i64 2, !dbg !33
  %755 = fadd float %751, %753, !dbg !32
  %756 = extractelement <4 x float> %690, i64 3, !dbg !31
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !31
  %758 = insertelement <4 x float> %754, float %757, i64 3, !dbg !33
  %math.result68 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %758) #9, !dbg !33, !frisk.register_lifetimes !1
  %759 = fptrunc <4 x float> %math.result68 to <4 x half>, !dbg !33
  %math.packed69 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %759) #9, !dbg !32, !frisk.register_lifetimes !1
  %760 = fadd float %755, %757, !dbg !32
  %761 = extractelement <4 x float> %691, i64 0, !dbg !31
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !31
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !33
  %764 = fadd float %760, %762, !dbg !32
  %765 = extractelement <4 x float> %691, i64 1, !dbg !31
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !31
  %767 = insertelement <4 x float> %763, float %766, i64 1, !dbg !33
  %768 = fadd float %764, %766, !dbg !32
  %769 = extractelement <4 x float> %691, i64 2, !dbg !31
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !31
  %771 = insertelement <4 x float> %767, float %770, i64 2, !dbg !33
  %772 = fadd float %768, %770, !dbg !32
  %773 = extractelement <4 x float> %691, i64 3, !dbg !31
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !31
  %775 = insertelement <4 x float> %771, float %774, i64 3, !dbg !33
  %math.result70 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !33, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result70 to <4 x half>, !dbg !33
  %math.packed71 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !32, !frisk.register_lifetimes !1
  %777 = fadd float %772, %774, !dbg !32
  %778 = extractelement <4 x float> %692, i64 0, !dbg !31
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !31
  %780 = insertelement <4 x float> poison, float %779, i64 0, !dbg !33
  %781 = fadd float %777, %779, !dbg !32
  %782 = extractelement <4 x float> %692, i64 1, !dbg !31
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !31
  %784 = insertelement <4 x float> %780, float %783, i64 1, !dbg !33
  %785 = fadd float %781, %783, !dbg !32
  %786 = extractelement <4 x float> %692, i64 2, !dbg !31
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !31
  %788 = insertelement <4 x float> %784, float %787, i64 2, !dbg !33
  %789 = fadd float %785, %787, !dbg !32
  %790 = extractelement <4 x float> %692, i64 3, !dbg !31
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !31
  %792 = insertelement <4 x float> %788, float %791, i64 3, !dbg !33
  %math.result72 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !33, !frisk.register_lifetimes !1
  %793 = fptrunc <4 x float> %math.result72 to <4 x half>, !dbg !33
  %math.packed73 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !32, !frisk.register_lifetimes !1
  %794 = fadd float %789, %791, !dbg !32
  %795 = extractelement <4 x float> %693, i64 0, !dbg !31
  %796 = call float @llvm.amdgcn.exp2.f32(float %795), !dbg !31
  %797 = insertelement <4 x float> poison, float %796, i64 0, !dbg !33
  %798 = fadd float %794, %796, !dbg !32
  %799 = extractelement <4 x float> %693, i64 1, !dbg !31
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !31
  %801 = insertelement <4 x float> %797, float %800, i64 1, !dbg !33
  %802 = fadd float %798, %800, !dbg !32
  %803 = extractelement <4 x float> %693, i64 2, !dbg !31
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !31
  %805 = insertelement <4 x float> %801, float %804, i64 2, !dbg !33
  %806 = fadd float %802, %804, !dbg !32
  %807 = extractelement <4 x float> %693, i64 3, !dbg !31
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !31
  %809 = insertelement <4 x float> %805, float %808, i64 3, !dbg !33
  %math.result74 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #9, !dbg !33, !frisk.register_lifetimes !1
  %810 = fptrunc <4 x float> %math.result74 to <4 x half>, !dbg !33
  %math.packed75 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #9, !dbg !32, !frisk.register_lifetimes !1
  %811 = fadd float %806, %808, !dbg !32
  %812 = extractelement <4 x float> %694, i64 0, !dbg !31
  %813 = call float @llvm.amdgcn.exp2.f32(float %812), !dbg !31
  %814 = insertelement <4 x float> poison, float %813, i64 0, !dbg !33
  %815 = fadd float %811, %813, !dbg !32
  %816 = extractelement <4 x float> %694, i64 1, !dbg !31
  %817 = call float @llvm.amdgcn.exp2.f32(float %816), !dbg !31
  %818 = insertelement <4 x float> %814, float %817, i64 1, !dbg !33
  %819 = fadd float %815, %817, !dbg !32
  %820 = extractelement <4 x float> %694, i64 2, !dbg !31
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !31
  %822 = insertelement <4 x float> %818, float %821, i64 2, !dbg !33
  %823 = fadd float %819, %821, !dbg !32
  %824 = extractelement <4 x float> %694, i64 3, !dbg !31
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !31
  %826 = insertelement <4 x float> %822, float %825, i64 3, !dbg !33
  %math.result76 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %826) #9, !dbg !33, !frisk.register_lifetimes !1
  %827 = fptrunc <4 x float> %math.result76 to <4 x half>, !dbg !33
  %math.packed77 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %827) #9, !dbg !32, !frisk.register_lifetimes !1
  %828 = fadd float %823, %825, !dbg !32
  %829 = extractelement <4 x float> %695, i64 0, !dbg !31
  %830 = call float @llvm.amdgcn.exp2.f32(float %829), !dbg !31
  %831 = insertelement <4 x float> poison, float %830, i64 0, !dbg !33
  %832 = fadd float %828, %830, !dbg !32
  %833 = extractelement <4 x float> %695, i64 1, !dbg !31
  %834 = call float @llvm.amdgcn.exp2.f32(float %833), !dbg !31
  %835 = insertelement <4 x float> %831, float %834, i64 1, !dbg !33
  %836 = fadd float %832, %834, !dbg !32
  %837 = extractelement <4 x float> %695, i64 2, !dbg !31
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !31
  %839 = insertelement <4 x float> %835, float %838, i64 2, !dbg !33
  %840 = fadd float %836, %838, !dbg !32
  %841 = extractelement <4 x float> %695, i64 3, !dbg !31
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !31
  %843 = insertelement <4 x float> %839, float %842, i64 3, !dbg !33
  %math.result78 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #9, !dbg !33, !frisk.register_lifetimes !1
  %844 = fptrunc <4 x float> %math.result78 to <4 x half>, !dbg !33
  %math.packed79 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #9, !dbg !32, !frisk.register_lifetimes !1
  %845 = fadd float %840, %842, !dbg !32
  %846 = bitcast float %845 to i32, !dbg !32
  %847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %672, i32 %846), !dbg !32
  %848 = bitcast i32 %847 to float, !dbg !32
  %849 = fadd float %845, %848, !dbg !32
  %850 = bitcast float %849 to i32, !dbg !32
  %851 = call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %850), !dbg !32
  %852 = bitcast i32 %851 to float, !dbg !32
  %853 = fadd float %849, %852, !dbg !32
  %854 = fadd float %853, %710, !dbg !32
  %855 = insertelement <1 x float> poison, float %854, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread271 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !14, !range !38, !frisk.register_lifetimes !1
  %address.local272 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local273 = and i32 %address.local272, 120, !dbg !14
  %address.local274 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local275 = and i32 %address.local274, 1536, !dbg !14
  %address.local276 = or disjoint i32 %address.local273, %address.local275, !dbg !14
  %address.local277 = zext nneg i32 %address.local276 to i64, !dbg !14
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local277, !dbg !14
  %856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local278, i32 0) #0, !dbg !14
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !14
  %address.local279 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local280 = and i32 %address.local279, 120, !dbg !14
  %address.local281 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local282 = and i32 %address.local281, 1536, !dbg !14
  %address.local283 = or disjoint i32 %address.local280, %address.local282, !dbg !14
  %address.local284 = or disjoint i32 %address.local283, 128, !dbg !14
  %address.local285 = zext nneg i32 %address.local284 to i64, !dbg !14
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local285, !dbg !14
  %858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local286, i32 0) #0, !dbg !14
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !14
  %address.local287 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local288 = and i32 %address.local287, 120, !dbg !14
  %address.local289 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local290 = and i32 %address.local289, 1536, !dbg !14
  %address.local291 = or disjoint i32 %address.local288, %address.local290, !dbg !14
  %address.local292 = or disjoint i32 %address.local291, 256, !dbg !14
  %address.local293 = zext nneg i32 %address.local292 to i64, !dbg !14
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local293, !dbg !14
  %860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local294, i32 0) #0, !dbg !14
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !14
  %address.local295 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local296 = and i32 %address.local295, 120, !dbg !14
  %address.local297 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local298 = and i32 %address.local297, 1536, !dbg !14
  %address.local299 = or disjoint i32 %address.local296, %address.local298, !dbg !14
  %address.local300 = or disjoint i32 %address.local299, 384, !dbg !14
  %address.local301 = zext nneg i32 %address.local300 to i64, !dbg !14
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local301, !dbg !14
  %862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local302, i32 0) #0, !dbg !14
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !14
  %address.local303 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local304 = and i32 %address.local303, 120, !dbg !14
  %address.local305 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local306 = and i32 %address.local305, 1536, !dbg !14
  %address.local307 = or disjoint i32 %address.local304, %address.local306, !dbg !14
  %address.local308 = or disjoint i32 %address.local307, 2048, !dbg !14
  %address.local309 = zext nneg i32 %address.local308 to i64, !dbg !14
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local309, !dbg !14
  %864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local310, i32 0) #0, !dbg !14
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !14
  %address.local311 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local312 = and i32 %address.local311, 120, !dbg !14
  %address.local313 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local314 = and i32 %address.local313, 1536, !dbg !14
  %address.local315 = or disjoint i32 %address.local312, %address.local314, !dbg !14
  %address.local316 = or disjoint i32 %address.local315, 2176, !dbg !14
  %address.local317 = zext nneg i32 %address.local316 to i64, !dbg !14
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local317, !dbg !14
  %866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local318, i32 0) #0, !dbg !14
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !14
  %address.local319 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local320 = and i32 %address.local319, 120, !dbg !14
  %address.local321 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local322 = and i32 %address.local321, 1536, !dbg !14
  %address.local323 = or disjoint i32 %address.local320, %address.local322, !dbg !14
  %address.local324 = or disjoint i32 %address.local323, 2304, !dbg !14
  %address.local325 = zext nneg i32 %address.local324 to i64, !dbg !14
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local325, !dbg !14
  %868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !14
  %address.local327 = shl nuw nsw i32 %address.thread271, 3, !dbg !14
  %address.local328 = and i32 %address.local327, 120, !dbg !14
  %address.local329 = shl nuw nsw i32 %address.thread271, 5, !dbg !14
  %address.local330 = and i32 %address.local329, 1536, !dbg !14
  %address.local331 = or disjoint i32 %address.local328, %address.local330, !dbg !14
  %address.local332 = or disjoint i32 %address.local331, 2432, !dbg !14
  %address.local333 = zext nneg i32 %address.local332 to i64, !dbg !14
  %address.local334 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local333, !dbg !14
  %870 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local334, i32 0) #0, !dbg !14
  %871 = bitcast <4 x i32> %870 to <8 x half>, !dbg !14
  %872 = add nuw nsw i64 %92, %15, !dbg !17
  %873 = shl i64 %872, 19, !dbg !17
  %874 = and i64 %873, 16252928, !dbg !17
  %875 = add i64 %874, %98, !dbg !17
  %876 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %875, !dbg !17
  %877 = getelementptr i8, ptr addrspace(1) %876, i64 64, !dbg !17
  %address.local335 = zext nneg i32 %address.thread271 to i64, !dbg !12
  %address.local336 = shl nuw nsw i64 %address.local335, 3, !dbg !11
  %address.local337 = and i64 %address.local336, 1536, !dbg !15
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local337, !dbg !15
  %address.local339 = ptrtoint ptr addrspace(3) %address.local338 to i64, !dbg !15
  %address.local340 = trunc i64 %address.local339 to i32, !dbg !15
  %878 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !17
  %879 = zext i32 %878 to i64, !dbg !17
  %880 = inttoptr i64 %879 to ptr addrspace(3), !dbg !17
  %address.local341 = shl nuw nsw i64 %address.local335, 5, !dbg !17
  %address.local342 = and i64 %address.local341, 8064, !dbg !17
  %address.local343 = shl nuw nsw i64 %address.local335, 3, !dbg !17
  %address.local344 = and i64 %address.local343, 24, !dbg !17
  %address.local345 = or disjoint i64 %address.local342, %address.local344, !dbg !17
  %address.local346 = getelementptr [2 x i8], ptr addrspace(1) %877, i64 %address.local345, !dbg !17
  %881 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %882 = ptrtoint ptr addrspace(1) %address.local346 to i64, !dbg !17
  %883 = sub i64 %882, %881, !dbg !17
  %dma.buffer.offset654 = trunc i64 %883 to i32, !dbg !17
  %884 = trunc i64 %881 to i32, !dbg !17
  %885 = insertelement <4 x i32> poison, i32 %884, i64 0, !dbg !17
  %886 = lshr i64 %881, 32, !dbg !17
  %887 = trunc i64 %886 to i32, !dbg !17
  %888 = insertelement <4 x i32> %885, i32 %887, i64 1, !dbg !17
  %889 = insertelement <4 x i32> %888, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc655 = insertelement <4 x i32> %889, i32 131072, i64 3, !dbg !17
  %890 = ptrtoint ptr addrspace(3) %880 to i32, !dbg !17
  %dma.buffer.lds656 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %890), !dbg !17
  %891 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds656, i32 %dma.buffer.offset654, <4 x i32> %dma.buffer.rsrc655) #10, !dbg !17
  %address.local347 = or disjoint i64 %address.local335, 256, !dbg !15
  %address.local348 = shl nuw nsw i64 %address.local347, 3, !dbg !15
  %address.local349 = and i64 %address.local348, 3584, !dbg !15
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local349, !dbg !15
  %address.local351 = ptrtoint ptr addrspace(3) %address.local350 to i64, !dbg !15
  %address.local352 = trunc i64 %address.local351 to i32, !dbg !15
  %892 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !17
  %893 = zext i32 %892 to i64, !dbg !17
  %894 = inttoptr i64 %893 to ptr addrspace(3), !dbg !17
  %address.local353 = shl nuw nsw i64 %address.local347, 5, !dbg !17
  %address.local354 = and i64 %address.local353, 16256, !dbg !17
  %address.local355 = shl nuw nsw i64 %address.local335, 3, !dbg !17
  %address.local356 = and i64 %address.local355, 24, !dbg !17
  %address.local357 = or disjoint i64 %address.local354, %address.local356, !dbg !17
  %address.local358 = getelementptr [2 x i8], ptr addrspace(1) %877, i64 %address.local357, !dbg !17
  %895 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %896 = ptrtoint ptr addrspace(1) %address.local358 to i64, !dbg !17
  %897 = sub i64 %896, %895, !dbg !17
  %dma.buffer.offset651 = trunc i64 %897 to i32, !dbg !17
  %898 = trunc i64 %895 to i32, !dbg !17
  %899 = insertelement <4 x i32> poison, i32 %898, i64 0, !dbg !17
  %900 = lshr i64 %895, 32, !dbg !17
  %901 = trunc i64 %900 to i32, !dbg !17
  %902 = insertelement <4 x i32> %899, i32 %901, i64 1, !dbg !17
  %903 = insertelement <4 x i32> %902, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc652 = insertelement <4 x i32> %903, i32 131072, i64 3, !dbg !17
  %904 = ptrtoint ptr addrspace(3) %894 to i32, !dbg !17
  %dma.buffer.lds653 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %904), !dbg !17
  %905 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds653, i32 %dma.buffer.offset651, <4 x i32> %dma.buffer.rsrc652) #10, !dbg !17
  %906 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %906, <4 x float> %708) #9, !dbg !14
  %908 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %908, <4 x float> %907) #9, !dbg !14
  %910 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %910, <4 x float> %909) #9, !dbg !14
  %912 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %912, <4 x float> %911) #9, !dbg !14
  %914 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %914, <4 x float> %913) #9, !dbg !14
  %916 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %916, <4 x float> %915) #9, !dbg !14
  %918 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %918, <4 x float> %917) #9, !dbg !14
  %920 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %920, <4 x float> %919) #9, !dbg !14
  %922 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %922, <4 x float> %707) #9, !dbg !14
  %924 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %924, <4 x float> %923) #9, !dbg !14
  %926 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %926, <4 x float> %925) #9, !dbg !14
  %928 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %928, <4 x float> %927) #9, !dbg !14
  %930 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %930, <4 x float> %929) #9, !dbg !14
  %932 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %932, <4 x float> %931) #9, !dbg !14
  %934 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %934, <4 x float> %933) #9, !dbg !14
  %936 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %936, <4 x float> %935) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local277, !dbg !14
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !14
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !14
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local285, !dbg !14
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !14
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !14
  %address.local361 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local293, !dbg !14
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !14
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local301, !dbg !14
  %944 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !14
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !14
  %address.local363 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !14
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !14
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !14
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local317, !dbg !14
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local364, i32 0) #0, !dbg !14
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !14
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %950 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !14
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !14
  %address.local366 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local333, !dbg !14
  %952 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !14
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %954 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %875, !dbg !17
  %955 = getelementptr i8, ptr addrspace(1) %954, i64 128, !dbg !17
  %956 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !17
  %957 = zext i32 %956 to i64, !dbg !17
  %958 = inttoptr i64 %957 to ptr addrspace(3), !dbg !17
  %address.local367 = getelementptr [2 x i8], ptr addrspace(1) %955, i64 %address.local345, !dbg !17
  %959 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %960 = ptrtoint ptr addrspace(1) %address.local367 to i64, !dbg !17
  %961 = sub i64 %960, %959, !dbg !17
  %dma.buffer.offset648 = trunc i64 %961 to i32, !dbg !17
  %962 = trunc i64 %959 to i32, !dbg !17
  %963 = insertelement <4 x i32> poison, i32 %962, i64 0, !dbg !17
  %964 = lshr i64 %959, 32, !dbg !17
  %965 = trunc i64 %964 to i32, !dbg !17
  %966 = insertelement <4 x i32> %963, i32 %965, i64 1, !dbg !17
  %967 = insertelement <4 x i32> %966, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc649 = insertelement <4 x i32> %967, i32 131072, i64 3, !dbg !17
  %968 = ptrtoint ptr addrspace(3) %958 to i32, !dbg !17
  %dma.buffer.lds650 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %968), !dbg !17
  %969 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds650, i32 %dma.buffer.offset648, <4 x i32> %dma.buffer.rsrc649) #10, !dbg !17
  %970 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !17
  %971 = zext i32 %970 to i64, !dbg !17
  %972 = inttoptr i64 %971 to ptr addrspace(3), !dbg !17
  %address.local368 = getelementptr [2 x i8], ptr addrspace(1) %955, i64 %address.local357, !dbg !17
  %973 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %974 = ptrtoint ptr addrspace(1) %address.local368 to i64, !dbg !17
  %975 = sub i64 %974, %973, !dbg !17
  %dma.buffer.offset645 = trunc i64 %975 to i32, !dbg !17
  %976 = trunc i64 %973 to i32, !dbg !17
  %977 = insertelement <4 x i32> poison, i32 %976, i64 0, !dbg !17
  %978 = lshr i64 %973, 32, !dbg !17
  %979 = trunc i64 %978 to i32, !dbg !17
  %980 = insertelement <4 x i32> %977, i32 %979, i64 1, !dbg !17
  %981 = insertelement <4 x i32> %980, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc646 = insertelement <4 x i32> %981, i32 131072, i64 3, !dbg !17
  %982 = ptrtoint ptr addrspace(3) %972 to i32, !dbg !17
  %dma.buffer.lds647 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %982), !dbg !17
  %983 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds647, i32 %dma.buffer.offset645, <4 x i32> %dma.buffer.rsrc646) #10, !dbg !17
  %984 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %984, <4 x float> %706) #9, !dbg !14
  %986 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %986, <4 x float> %985) #9, !dbg !14
  %988 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %988, <4 x float> %987) #9, !dbg !14
  %990 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %991 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %990, <4 x float> %989) #9, !dbg !14
  %992 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %992, <4 x float> %991) #9, !dbg !14
  %994 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %994, <4 x float> %993) #9, !dbg !14
  %996 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %996, <4 x float> %995) #9, !dbg !14
  %998 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %998, <4 x float> %997) #9, !dbg !14
  %1000 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1000, <4 x float> %705) #9, !dbg !14
  %1002 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1002, <4 x float> %1001) #9, !dbg !14
  %1004 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1004, <4 x float> %1003) #9, !dbg !14
  %1006 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1006, <4 x float> %1005) #9, !dbg !14
  %1008 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1008, <4 x float> %1007) #9, !dbg !14
  %1010 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1010, <4 x float> %1009) #9, !dbg !14
  %1012 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %1012, <4 x float> %1011) #9, !dbg !14
  %1014 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %1014, <4 x float> %1013) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !14
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !14
  %1018 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !14
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !14
  %1020 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !14
  %1022 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !14
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !14
  %1024 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !14
  %1025 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !14
  %1026 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local364, i32 0) #0, !dbg !14
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !14
  %1028 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !14
  %1029 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !14
  %1030 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !14
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1032 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %875, !dbg !17
  %1033 = getelementptr i8, ptr addrspace(1) %1032, i64 192, !dbg !17
  %1034 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local340), !dbg !17
  %1035 = zext i32 %1034 to i64, !dbg !17
  %1036 = inttoptr i64 %1035 to ptr addrspace(3), !dbg !17
  %address.local369 = getelementptr [2 x i8], ptr addrspace(1) %1033, i64 %address.local345, !dbg !17
  %1037 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1038 = ptrtoint ptr addrspace(1) %address.local369 to i64, !dbg !17
  %1039 = sub i64 %1038, %1037, !dbg !17
  %dma.buffer.offset642 = trunc i64 %1039 to i32, !dbg !17
  %1040 = trunc i64 %1037 to i32, !dbg !17
  %1041 = insertelement <4 x i32> poison, i32 %1040, i64 0, !dbg !17
  %1042 = lshr i64 %1037, 32, !dbg !17
  %1043 = trunc i64 %1042 to i32, !dbg !17
  %1044 = insertelement <4 x i32> %1041, i32 %1043, i64 1, !dbg !17
  %1045 = insertelement <4 x i32> %1044, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc643 = insertelement <4 x i32> %1045, i32 131072, i64 3, !dbg !17
  %1046 = ptrtoint ptr addrspace(3) %1036 to i32, !dbg !17
  %dma.buffer.lds644 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1046), !dbg !17
  %1047 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds644, i32 %dma.buffer.offset642, <4 x i32> %dma.buffer.rsrc643) #10, !dbg !17
  %1048 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local352), !dbg !17
  %1049 = zext i32 %1048 to i64, !dbg !17
  %1050 = inttoptr i64 %1049 to ptr addrspace(3), !dbg !17
  %address.local370 = getelementptr [2 x i8], ptr addrspace(1) %1033, i64 %address.local357, !dbg !17
  %1051 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !17
  %1052 = ptrtoint ptr addrspace(1) %address.local370 to i64, !dbg !17
  %1053 = sub i64 %1052, %1051, !dbg !17
  %dma.buffer.offset = trunc i64 %1053 to i32, !dbg !17
  %1054 = trunc i64 %1051 to i32, !dbg !17
  %1055 = insertelement <4 x i32> poison, i32 %1054, i64 0, !dbg !17
  %1056 = lshr i64 %1051, 32, !dbg !17
  %1057 = trunc i64 %1056 to i32, !dbg !17
  %1058 = insertelement <4 x i32> %1055, i32 %1057, i64 1, !dbg !17
  %1059 = insertelement <4 x i32> %1058, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %1059, i32 131072, i64 3, !dbg !17
  %1060 = ptrtoint ptr addrspace(3) %1050 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1060), !dbg !17
  %1061 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %1062 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1062, <4 x float> %704) #9, !dbg !14
  %1064 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1064, <4 x float> %1063) #9, !dbg !14
  %1066 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1066, <4 x float> %1065) #9, !dbg !14
  %1068 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1068, <4 x float> %1067) #9, !dbg !14
  %1070 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1070, <4 x float> %1069) #9, !dbg !14
  %1072 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1072, <4 x float> %1071) #9, !dbg !14
  %1074 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %1074, <4 x float> %1073) #9, !dbg !14
  %1076 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %1076, <4 x float> %1075) #9, !dbg !14
  %1078 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1078, <4 x float> %703) #9, !dbg !14
  %1080 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1080, <4 x float> %1079) #9, !dbg !14
  %1082 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1082, <4 x float> %1081) #9, !dbg !14
  %1084 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1084, <4 x float> %1083) #9, !dbg !14
  %1086 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1086, <4 x float> %1085) #9, !dbg !14
  %1088 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1088, <4 x float> %1087) #9, !dbg !14
  %1090 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %1090, <4 x float> %1089) #9, !dbg !14
  %1092 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %1092, <4 x float> %1091) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local359, i32 0) #0, !dbg !14
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !14
  %1096 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local360, i32 0) #0, !dbg !14
  %1097 = bitcast <4 x i32> %1096 to <8 x half>, !dbg !14
  %1098 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %1099 = bitcast <4 x i32> %1098 to <8 x half>, !dbg !14
  %1100 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local362, i32 0) #0, !dbg !14
  %1101 = bitcast <4 x i32> %1100 to <8 x half>, !dbg !14
  %1102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local363, i32 0) #0, !dbg !14
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !14
  %1104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local364, i32 0) #0, !dbg !14
  %1105 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !14
  %1106 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local365, i32 0) #0, !dbg !14
  %1107 = bitcast <4 x i32> %1106 to <8 x half>, !dbg !14
  %1108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local366, i32 0) #0, !dbg !14
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !14
  %1110 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1111 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1110, <4 x float> %702) #9, !dbg !14
  %1112 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1113 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1112, <4 x float> %1111) #9, !dbg !14
  %1114 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1115 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1114, <4 x float> %1113) #9, !dbg !14
  %1116 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1117 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1116, <4 x float> %1115) #9, !dbg !14
  %1118 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1118, <4 x float> %1117) #9, !dbg !14
  %1120 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1120, <4 x float> %1119) #9, !dbg !14
  %1122 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %1122, <4 x float> %1121) #9, !dbg !14
  %1124 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %1124, <4 x float> %1123) #9, !dbg !14
  %1126 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1126, <4 x float> %701) #9, !dbg !14
  %1128 = shufflevector <8 x half> %1097, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed67, <4 x half> %1128, <4 x float> %1127) #9, !dbg !14
  %1130 = shufflevector <8 x half> %1099, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed69, <4 x half> %1130, <4 x float> %1129) #9, !dbg !14
  %1132 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed71, <4 x half> %1132, <4 x float> %1131) #9, !dbg !14
  %1134 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed73, <4 x half> %1134, <4 x float> %1133) #9, !dbg !14
  %1136 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed75, <4 x half> %1136, <4 x float> %1135) #9, !dbg !14
  %1138 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed77, <4 x half> %1138, <4 x float> %1137) #9, !dbg !14
  %1140 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed79, <4 x half> %1140, <4 x float> %1139) #9, !dbg !14
  %1142 = shufflevector <4 x float> %921, <4 x float> %937, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1143 = shufflevector <4 x float> %999, <4 x float> %1015, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1144 = shufflevector <4 x float> %1077, <4 x float> %1093, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1145 = shufflevector <4 x float> %1125, <4 x float> %1141, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1146 = add nuw nsw i64 %58, 1, !dbg !7
  br label %57, !dbg !7

1147:                                             ; preds = %57
  %1148 = and i32 %34, 16, !dbg !7
  %.not = icmp eq i32 %1148, 0, !dbg !7
  %1149 = and i32 %34, 32, !dbg !7
  %.not1 = icmp eq i32 %1149, 0, !dbg !7
  %1150 = extractelement <8 x float> %72, i64 0, !dbg !7
  %1151 = extractelement <8 x float> %72, i64 1, !dbg !7
  %1152 = extractelement <8 x float> %72, i64 2, !dbg !7
  %1153 = extractelement <8 x float> %72, i64 3, !dbg !7
  %1154 = select i1 %.not, float %1151, float %1150, !dbg !7
  %1155 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1156 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1155), !dbg !7
  %1157 = add nuw nsw i32 %1156, 64, !dbg !7
  %1158 = xor i32 %1156, 16, !dbg !7
  %1159 = and i32 %1157, 192, !dbg !7
  %1160 = icmp samesign ult i32 %1158, %1159, !dbg !7
  %1161 = select i1 %1160, i32 %1158, i32 %1156, !dbg !7
  %1162 = shl nuw nsw i32 %1161, 2, !dbg !7
  %1163 = bitcast float %1154 to i32, !dbg !7
  %1164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1163), !dbg !7
  %1165 = bitcast i32 %1164 to float, !dbg !7
  %1166 = select i1 %.not, float %1150, float %1165, !dbg !7
  %1167 = select i1 %.not, float %1165, float %1151, !dbg !7
  %1168 = select i1 %.not, float %1153, float %1152, !dbg !7
  %1169 = bitcast float %1168 to i32, !dbg !7
  %1170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1169), !dbg !7
  %1171 = bitcast i32 %1170 to float, !dbg !7
  %1172 = select i1 %.not, float %1152, float %1171, !dbg !7
  %1173 = select i1 %.not, float %1171, float %1153, !dbg !7
  %1174 = select i1 %.not1, float %1172, float %1166, !dbg !7
  %1175 = xor i32 %1156, 32, !dbg !7
  %1176 = icmp samesign ult i32 %1175, %1159, !dbg !7
  %1177 = select i1 %1176, i32 %1175, i32 %1156, !dbg !7
  %1178 = shl nuw nsw i32 %1177, 2, !dbg !7
  %1179 = bitcast float %1174 to i32, !dbg !7
  %1180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1179), !dbg !7
  %1181 = bitcast i32 %1180 to float, !dbg !7
  %1182 = select i1 %.not1, float %1166, float %1181, !dbg !7
  %1183 = select i1 %.not1, float %1181, float %1172, !dbg !7
  %1184 = select i1 %.not1, float %1173, float %1167, !dbg !7
  %1185 = bitcast float %1184 to i32, !dbg !7
  %1186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1185), !dbg !7
  %1187 = bitcast i32 %1186 to float, !dbg !7
  %1188 = select i1 %.not1, float %1167, float %1187, !dbg !7
  %1189 = select i1 %.not1, float %1187, float %1173, !dbg !7
  %1190 = extractelement <8 x float> %69, i64 0, !dbg !7
  %1191 = extractelement <8 x float> %69, i64 1, !dbg !7
  %1192 = extractelement <8 x float> %69, i64 2, !dbg !7
  %1193 = extractelement <8 x float> %69, i64 3, !dbg !7
  %1194 = select i1 %.not, float %1191, float %1190, !dbg !7
  %1195 = bitcast float %1194 to i32, !dbg !7
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1195), !dbg !7
  %1197 = bitcast i32 %1196 to float, !dbg !7
  %1198 = select i1 %.not, float %1190, float %1197, !dbg !7
  %1199 = select i1 %.not, float %1197, float %1191, !dbg !7
  %1200 = select i1 %.not, float %1193, float %1192, !dbg !7
  %1201 = bitcast float %1200 to i32, !dbg !7
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1201), !dbg !7
  %1203 = bitcast i32 %1202 to float, !dbg !7
  %1204 = select i1 %.not, float %1192, float %1203, !dbg !7
  %1205 = select i1 %.not, float %1203, float %1193, !dbg !7
  %1206 = select i1 %.not1, float %1204, float %1198, !dbg !7
  %1207 = bitcast float %1206 to i32, !dbg !7
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1207), !dbg !7
  %1209 = bitcast i32 %1208 to float, !dbg !7
  %1210 = select i1 %.not1, float %1198, float %1209, !dbg !7
  %1211 = select i1 %.not1, float %1209, float %1204, !dbg !7
  %1212 = select i1 %.not1, float %1205, float %1199, !dbg !7
  %1213 = bitcast float %1212 to i32, !dbg !7
  %1214 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1213), !dbg !7
  %1215 = bitcast i32 %1214 to float, !dbg !7
  %1216 = select i1 %.not1, float %1199, float %1215, !dbg !7
  %1217 = select i1 %.not1, float %1215, float %1205, !dbg !7
  %1218 = extractelement <8 x float> %66, i64 0, !dbg !7
  %1219 = extractelement <8 x float> %66, i64 1, !dbg !7
  %1220 = extractelement <8 x float> %66, i64 2, !dbg !7
  %1221 = extractelement <8 x float> %66, i64 3, !dbg !7
  %1222 = select i1 %.not, float %1219, float %1218, !dbg !7
  %1223 = bitcast float %1222 to i32, !dbg !7
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1223), !dbg !7
  %1225 = bitcast i32 %1224 to float, !dbg !7
  %1226 = select i1 %.not, float %1218, float %1225, !dbg !7
  %1227 = select i1 %.not, float %1225, float %1219, !dbg !7
  %1228 = select i1 %.not, float %1221, float %1220, !dbg !7
  %1229 = bitcast float %1228 to i32, !dbg !7
  %1230 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1229), !dbg !7
  %1231 = bitcast i32 %1230 to float, !dbg !7
  %1232 = select i1 %.not, float %1220, float %1231, !dbg !7
  %1233 = select i1 %.not, float %1231, float %1221, !dbg !7
  %1234 = select i1 %.not1, float %1232, float %1226, !dbg !7
  %1235 = bitcast float %1234 to i32, !dbg !7
  %1236 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1235), !dbg !7
  %1237 = bitcast i32 %1236 to float, !dbg !7
  %1238 = select i1 %.not1, float %1226, float %1237, !dbg !7
  %1239 = select i1 %.not1, float %1237, float %1232, !dbg !7
  %1240 = select i1 %.not1, float %1233, float %1227, !dbg !7
  %1241 = bitcast float %1240 to i32, !dbg !7
  %1242 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1241), !dbg !7
  %1243 = bitcast i32 %1242 to float, !dbg !7
  %1244 = select i1 %.not1, float %1227, float %1243, !dbg !7
  %1245 = select i1 %.not1, float %1243, float %1233, !dbg !7
  %1246 = extractelement <8 x float> %61, i64 0, !dbg !7
  %1247 = extractelement <8 x float> %61, i64 1, !dbg !7
  %1248 = extractelement <8 x float> %61, i64 2, !dbg !7
  %1249 = extractelement <8 x float> %61, i64 3, !dbg !7
  %1250 = select i1 %.not, float %1247, float %1246, !dbg !7
  %1251 = bitcast float %1250 to i32, !dbg !7
  %1252 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1251), !dbg !7
  %1253 = bitcast i32 %1252 to float, !dbg !7
  %1254 = select i1 %.not, float %1246, float %1253, !dbg !7
  %1255 = select i1 %.not, float %1253, float %1247, !dbg !7
  %1256 = select i1 %.not, float %1249, float %1248, !dbg !7
  %1257 = bitcast float %1256 to i32, !dbg !7
  %1258 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1257), !dbg !7
  %1259 = bitcast i32 %1258 to float, !dbg !7
  %1260 = select i1 %.not, float %1248, float %1259, !dbg !7
  %1261 = select i1 %.not, float %1259, float %1249, !dbg !7
  %1262 = select i1 %.not1, float %1260, float %1254, !dbg !7
  %1263 = bitcast float %1262 to i32, !dbg !7
  %1264 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1263), !dbg !7
  %1265 = select i1 %.not1, float %1261, float %1255, !dbg !7
  %1266 = bitcast float %1265 to i32, !dbg !7
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1266), !dbg !7
  %1268 = shufflevector <1 x float> %62, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !18
  %1269 = extractelement <8 x float> %72, i64 4, !dbg !7
  %1270 = extractelement <8 x float> %72, i64 5, !dbg !7
  %1271 = extractelement <8 x float> %72, i64 6, !dbg !7
  %1272 = extractelement <8 x float> %72, i64 7, !dbg !7
  %1273 = select i1 %.not, float %1270, float %1269, !dbg !7
  %1274 = bitcast float %1273 to i32, !dbg !7
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1274), !dbg !7
  %1276 = bitcast i32 %1275 to float, !dbg !7
  %1277 = select i1 %.not, float %1269, float %1276, !dbg !7
  %1278 = select i1 %.not, float %1276, float %1270, !dbg !7
  %1279 = select i1 %.not, float %1272, float %1271, !dbg !7
  %1280 = bitcast float %1279 to i32, !dbg !7
  %1281 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1280), !dbg !7
  %1282 = bitcast i32 %1281 to float, !dbg !7
  %1283 = select i1 %.not, float %1271, float %1282, !dbg !7
  %1284 = select i1 %.not, float %1282, float %1272, !dbg !7
  %1285 = select i1 %.not1, float %1283, float %1277, !dbg !7
  %1286 = bitcast float %1285 to i32, !dbg !7
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1286), !dbg !7
  %1288 = bitcast i32 %1287 to float, !dbg !7
  %1289 = select i1 %.not1, float %1277, float %1288, !dbg !7
  %1290 = select i1 %.not1, float %1288, float %1283, !dbg !7
  %1291 = select i1 %.not1, float %1284, float %1278, !dbg !7
  %1292 = bitcast float %1291 to i32, !dbg !7
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1292), !dbg !7
  %1294 = bitcast i32 %1293 to float, !dbg !7
  %1295 = select i1 %.not1, float %1278, float %1294, !dbg !7
  %1296 = select i1 %.not1, float %1294, float %1284, !dbg !7
  %1297 = extractelement <8 x float> %69, i64 4, !dbg !7
  %1298 = extractelement <8 x float> %69, i64 5, !dbg !7
  %1299 = extractelement <8 x float> %69, i64 6, !dbg !7
  %1300 = extractelement <8 x float> %69, i64 7, !dbg !7
  %1301 = select i1 %.not, float %1298, float %1297, !dbg !7
  %1302 = bitcast float %1301 to i32, !dbg !7
  %1303 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1302), !dbg !7
  %1304 = bitcast i32 %1303 to float, !dbg !7
  %1305 = select i1 %.not, float %1297, float %1304, !dbg !7
  %1306 = select i1 %.not, float %1304, float %1298, !dbg !7
  %1307 = select i1 %.not, float %1300, float %1299, !dbg !7
  %1308 = bitcast float %1307 to i32, !dbg !7
  %1309 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1308), !dbg !7
  %1310 = bitcast i32 %1309 to float, !dbg !7
  %1311 = select i1 %.not, float %1299, float %1310, !dbg !7
  %1312 = select i1 %.not, float %1310, float %1300, !dbg !7
  %1313 = select i1 %.not1, float %1311, float %1305, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not1, float %1305, float %1316, !dbg !7
  %1318 = select i1 %.not1, float %1316, float %1311, !dbg !7
  %1319 = select i1 %.not1, float %1312, float %1306, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not1, float %1306, float %1322, !dbg !7
  %1324 = select i1 %.not1, float %1322, float %1312, !dbg !7
  %1325 = extractelement <8 x float> %66, i64 4, !dbg !7
  %1326 = extractelement <8 x float> %66, i64 5, !dbg !7
  %1327 = extractelement <8 x float> %66, i64 6, !dbg !7
  %1328 = extractelement <8 x float> %66, i64 7, !dbg !7
  %1329 = select i1 %.not, float %1326, float %1325, !dbg !7
  %1330 = bitcast float %1329 to i32, !dbg !7
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1330), !dbg !7
  %1332 = bitcast i32 %1331 to float, !dbg !7
  %1333 = select i1 %.not, float %1325, float %1332, !dbg !7
  %1334 = select i1 %.not, float %1332, float %1326, !dbg !7
  %1335 = select i1 %.not, float %1328, float %1327, !dbg !7
  %1336 = bitcast float %1335 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1336), !dbg !7
  %1338 = bitcast i32 %1337 to float, !dbg !7
  %1339 = select i1 %.not, float %1327, float %1338, !dbg !7
  %1340 = select i1 %.not, float %1338, float %1328, !dbg !7
  %1341 = select i1 %.not1, float %1339, float %1333, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not1, float %1333, float %1344, !dbg !7
  %1346 = select i1 %.not1, float %1344, float %1339, !dbg !7
  %1347 = select i1 %.not1, float %1340, float %1334, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not1, float %1334, float %1350, !dbg !7
  %1352 = select i1 %.not1, float %1350, float %1340, !dbg !7
  %1353 = extractelement <8 x float> %61, i64 4, !dbg !7
  %1354 = extractelement <8 x float> %61, i64 5, !dbg !7
  %1355 = extractelement <8 x float> %61, i64 6, !dbg !7
  %1356 = extractelement <8 x float> %61, i64 7, !dbg !7
  %1357 = select i1 %.not, float %1354, float %1353, !dbg !7
  %1358 = bitcast float %1357 to i32, !dbg !7
  %1359 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1358), !dbg !7
  %1360 = bitcast i32 %1359 to float, !dbg !7
  %1361 = select i1 %.not, float %1353, float %1360, !dbg !7
  %1362 = select i1 %.not, float %1360, float %1354, !dbg !7
  %1363 = select i1 %.not, float %1356, float %1355, !dbg !7
  %1364 = bitcast float %1363 to i32, !dbg !7
  %1365 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1162, i32 %1364), !dbg !7
  %1366 = bitcast i32 %1365 to float, !dbg !7
  %1367 = select i1 %.not, float %1355, float %1366, !dbg !7
  %1368 = select i1 %.not, float %1366, float %1356, !dbg !7
  %1369 = select i1 %.not1, float %1367, float %1361, !dbg !7
  %1370 = bitcast float %1369 to i32, !dbg !7
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1370), !dbg !7
  %1372 = select i1 %.not1, float %1368, float %1362, !dbg !7
  %1373 = bitcast float %1372 to i32, !dbg !7
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1178, i32 %1373), !dbg !7
  br i1 true, label %1375, label %1409, !dbg !20

1375:                                             ; preds = %1147
  %1376 = bitcast i32 %1371 to float, !dbg !7
  %1377 = select i1 %.not1, float %1361, float %1376, !dbg !7
  %1378 = insertelement <4 x float> poison, float %1377, i64 0, !dbg !18
  %1379 = bitcast i32 %1374 to float, !dbg !7
  %1380 = select i1 %.not1, float %1362, float %1379, !dbg !7
  %1381 = insertelement <4 x float> %1378, float %1380, i64 1, !dbg !18
  %1382 = select i1 %.not1, float %1376, float %1367, !dbg !7
  %1383 = insertelement <4 x float> %1381, float %1382, i64 2, !dbg !18
  %1384 = select i1 %.not1, float %1379, float %1368, !dbg !7
  %1385 = insertelement <4 x float> %1383, float %1384, i64 3, !dbg !18
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1268, !dbg !18
  %normalized = fmul <4 x float> %1385, %row.reciprocal, !dbg !18
  %1386 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !19
  %1387 = extractelement <4 x half> %1386, i64 3, !dbg !19
  %1388 = extractelement <4 x half> %1386, i64 2, !dbg !19
  %1389 = extractelement <4 x half> %1386, i64 1, !dbg !19
  %1390 = extractelement <4 x half> %1386, i64 0, !dbg !19
  %1391 = bitcast i32 %1264 to float, !dbg !7
  %1392 = select i1 %.not1, float %1254, float %1391, !dbg !7
  %1393 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !18
  %1394 = bitcast i32 %1267 to float, !dbg !7
  %1395 = select i1 %.not1, float %1255, float %1394, !dbg !7
  %1396 = insertelement <4 x float> %1393, float %1395, i64 1, !dbg !18
  %1397 = select i1 %.not1, float %1391, float %1260, !dbg !7
  %1398 = insertelement <4 x float> %1396, float %1397, i64 2, !dbg !18
  %1399 = select i1 %.not1, float %1394, float %1261, !dbg !7
  %1400 = insertelement <4 x float> %1398, float %1399, i64 3, !dbg !18
  %normalized36 = fmul <4 x float> %1400, %row.reciprocal, !dbg !18
  %1401 = fptrunc <4 x float> %normalized36 to <4 x half>, !dbg !19
  %1402 = extractelement <4 x half> %1401, i64 3, !dbg !19
  %1403 = extractelement <4 x half> %1401, i64 2, !dbg !19
  %1404 = extractelement <4 x half> %1401, i64 1, !dbg !19
  %1405 = extractelement <4 x half> %1401, i64 0, !dbg !19
  %1406 = add nsw i64 %15, %25, !dbg !20
  %1407 = shl i64 %1406, 19, !dbg !20
  %1408 = and i64 %1407, 16252928, !dbg !20
  %address.thread371 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local372 = zext nneg i32 %address.thread371 to i64, !dbg !12
  %address.local373 = add i64 %28, %address.local372, !dbg !20
  %address.local374 = lshr i64 %address.local372, 6, !dbg !12
  %address.local375 = shl nuw nsw i64 %address.local374, 4, !dbg !20
  %address.local376 = add i64 %address.local373, %address.local375, !dbg !20
  %address.local377 = and i64 %address.local372, 240, !dbg !20
  %address.local378 = sub i64 %address.local376, %address.local377, !dbg !20
  %address.local379 = add i64 %address.local378, %29, !dbg !20
  %address.local380 = shl nuw nsw i64 %address.local379, 7, !dbg !20
  %address.local381 = add nuw nsw i64 %1408, %address.local380, !dbg !20
  %address.local382 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local383 = lshr i64 %address.local372, 2, !dbg !12
  %address.local384 = and i64 %address.local383, 12, !dbg !12
  %address.local385 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local382, i64 %address.local384, !dbg !20
  store half %1405, ptr addrspace(1) %address.local385, align 2, !dbg !20
  %address.local386 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local387 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local386, i64 %address.local384, !dbg !20
  %address.local388 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local387, i64 2, !dbg !20
  store half %1404, ptr addrspace(1) %address.local388, align 2, !dbg !20
  %address.local389 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local390 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local389, i64 %address.local384, !dbg !20
  %address.local391 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local390, i64 4, !dbg !20
  store half %1403, ptr addrspace(1) %address.local391, align 2, !dbg !20
  %address.local392 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local393 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local392, i64 %address.local384, !dbg !20
  %address.local394 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local393, i64 6, !dbg !20
  store half %1402, ptr addrspace(1) %address.local394, align 2, !dbg !20
  %address.local395 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local396 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local395, i64 %address.local384, !dbg !20
  %address.local397 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local396, i64 32, !dbg !20
  store half %1390, ptr addrspace(1) %address.local397, align 2, !dbg !20
  %address.local398 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local399 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local398, i64 %address.local384, !dbg !20
  %address.local400 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local399, i64 34, !dbg !20
  store half %1389, ptr addrspace(1) %address.local400, align 2, !dbg !20
  %address.local401 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local402 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local401, i64 %address.local384, !dbg !20
  %address.local403 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local402, i64 36, !dbg !20
  store half %1388, ptr addrspace(1) %address.local403, align 2, !dbg !20
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local381, !dbg !20
  %address.local405 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local384, !dbg !20
  %address.local406 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local405, i64 38, !dbg !20
  store half %1387, ptr addrspace(1) %address.local406, align 2, !dbg !20
  br label %1409, !dbg !20

1409:                                             ; preds = %1375, %1147
  br i1 true, label %1410, label %1430, !dbg !20

1410:                                             ; preds = %1409
  %1411 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !18
  %1412 = insertelement <4 x float> %1411, float %1351, i64 1, !dbg !18
  %1413 = insertelement <4 x float> %1412, float %1346, i64 2, !dbg !18
  %1414 = insertelement <4 x float> %1413, float %1352, i64 3, !dbg !18
  %row.reciprocal37 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1268, !dbg !18
  %normalized38 = fmul <4 x float> %1414, %row.reciprocal37, !dbg !18
  %1415 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !19
  %1416 = extractelement <4 x half> %1415, i64 3, !dbg !19
  %1417 = extractelement <4 x half> %1415, i64 2, !dbg !19
  %1418 = extractelement <4 x half> %1415, i64 1, !dbg !19
  %1419 = extractelement <4 x half> %1415, i64 0, !dbg !19
  %1420 = insertelement <4 x float> poison, float %1238, i64 0, !dbg !18
  %1421 = insertelement <4 x float> %1420, float %1244, i64 1, !dbg !18
  %1422 = insertelement <4 x float> %1421, float %1239, i64 2, !dbg !18
  %1423 = insertelement <4 x float> %1422, float %1245, i64 3, !dbg !18
  %normalized39 = fmul <4 x float> %1423, %row.reciprocal37, !dbg !18
  %1424 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !19
  %1425 = extractelement <4 x half> %1424, i64 3, !dbg !19
  %1426 = extractelement <4 x half> %1424, i64 2, !dbg !19
  %1427 = extractelement <4 x half> %1424, i64 1, !dbg !19
  %1428 = extractelement <4 x half> %1424, i64 0, !dbg !19
  %1429 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread407 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local408 = zext nneg i32 %address.thread407 to i64, !dbg !12
  %address.local409 = add i64 %28, %address.local408, !dbg !20
  %address.local410 = lshr i64 %address.local408, 6, !dbg !12
  %address.local411 = shl nuw nsw i64 %address.local410, 4, !dbg !20
  %address.local412 = add i64 %address.local409, %address.local411, !dbg !20
  %address.local413 = and i64 %address.local408, 240, !dbg !20
  %address.local414 = sub i64 %address.local412, %address.local413, !dbg !20
  %address.local415 = add i64 %address.local414, %29, !dbg !20
  %address.local416 = shl nuw nsw i64 %address.local415, 7, !dbg !20
  %address.local417 = add nuw nsw i64 %1429, %address.local416, !dbg !20
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local419 = lshr i64 %address.local408, 2, !dbg !12
  %address.local420 = and i64 %address.local419, 12, !dbg !12
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local418, i64 %address.local420, !dbg !20
  %address.local422 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local421, i64 64, !dbg !20
  store half %1428, ptr addrspace(1) %address.local422, align 2, !dbg !20
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local423, i64 %address.local420, !dbg !20
  %address.local425 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local424, i64 66, !dbg !20
  store half %1427, ptr addrspace(1) %address.local425, align 2, !dbg !20
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local427 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local426, i64 %address.local420, !dbg !20
  %address.local428 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local427, i64 68, !dbg !20
  store half %1426, ptr addrspace(1) %address.local428, align 2, !dbg !20
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local420, !dbg !20
  %address.local431 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local430, i64 70, !dbg !20
  store half %1425, ptr addrspace(1) %address.local431, align 2, !dbg !20
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local420, !dbg !20
  %address.local434 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local433, i64 96, !dbg !20
  store half %1419, ptr addrspace(1) %address.local434, align 2, !dbg !20
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local420, !dbg !20
  %address.local437 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local436, i64 98, !dbg !20
  store half %1418, ptr addrspace(1) %address.local437, align 2, !dbg !20
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local420, !dbg !20
  %address.local440 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local439, i64 100, !dbg !20
  store half %1417, ptr addrspace(1) %address.local440, align 2, !dbg !20
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local417, !dbg !20
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local420, !dbg !20
  %address.local443 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local442, i64 102, !dbg !20
  store half %1416, ptr addrspace(1) %address.local443, align 2, !dbg !20
  br label %1430, !dbg !20

1430:                                             ; preds = %1410, %1409
  br i1 true, label %1431, label %1451, !dbg !20

1431:                                             ; preds = %1430
  %1432 = insertelement <4 x float> poison, float %1317, i64 0, !dbg !18
  %1433 = insertelement <4 x float> %1432, float %1323, i64 1, !dbg !18
  %1434 = insertelement <4 x float> %1433, float %1318, i64 2, !dbg !18
  %1435 = insertelement <4 x float> %1434, float %1324, i64 3, !dbg !18
  %row.reciprocal40 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1268, !dbg !18
  %normalized41 = fmul <4 x float> %1435, %row.reciprocal40, !dbg !18
  %1436 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !19
  %1437 = extractelement <4 x half> %1436, i64 3, !dbg !19
  %1438 = extractelement <4 x half> %1436, i64 2, !dbg !19
  %1439 = extractelement <4 x half> %1436, i64 1, !dbg !19
  %1440 = extractelement <4 x half> %1436, i64 0, !dbg !19
  %1441 = insertelement <4 x float> poison, float %1210, i64 0, !dbg !18
  %1442 = insertelement <4 x float> %1441, float %1216, i64 1, !dbg !18
  %1443 = insertelement <4 x float> %1442, float %1211, i64 2, !dbg !18
  %1444 = insertelement <4 x float> %1443, float %1217, i64 3, !dbg !18
  %normalized42 = fmul <4 x float> %1444, %row.reciprocal40, !dbg !18
  %1445 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !19
  %1446 = extractelement <4 x half> %1445, i64 3, !dbg !19
  %1447 = extractelement <4 x half> %1445, i64 2, !dbg !19
  %1448 = extractelement <4 x half> %1445, i64 1, !dbg !19
  %1449 = extractelement <4 x half> %1445, i64 0, !dbg !19
  %1450 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread444 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local445 = zext nneg i32 %address.thread444 to i64, !dbg !12
  %address.local446 = add i64 %28, %address.local445, !dbg !20
  %address.local447 = lshr i64 %address.local445, 6, !dbg !12
  %address.local448 = shl nuw nsw i64 %address.local447, 4, !dbg !20
  %address.local449 = add i64 %address.local446, %address.local448, !dbg !20
  %address.local450 = and i64 %address.local445, 240, !dbg !20
  %address.local451 = sub i64 %address.local449, %address.local450, !dbg !20
  %address.local452 = add i64 %address.local451, %29, !dbg !20
  %address.local453 = shl nuw nsw i64 %address.local452, 7, !dbg !20
  %address.local454 = add nuw nsw i64 %1450, %address.local453, !dbg !20
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local456 = lshr i64 %address.local445, 2, !dbg !12
  %address.local457 = and i64 %address.local456, 12, !dbg !12
  %address.local458 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local455, i64 %address.local457, !dbg !20
  %address.local459 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local458, i64 128, !dbg !20
  store half %1449, ptr addrspace(1) %address.local459, align 2, !dbg !20
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local461 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local457, !dbg !20
  %address.local462 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local461, i64 130, !dbg !20
  store half %1448, ptr addrspace(1) %address.local462, align 2, !dbg !20
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local464 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local457, !dbg !20
  %address.local465 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local464, i64 132, !dbg !20
  store half %1447, ptr addrspace(1) %address.local465, align 2, !dbg !20
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local457, !dbg !20
  %address.local468 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local467, i64 134, !dbg !20
  store half %1446, ptr addrspace(1) %address.local468, align 2, !dbg !20
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local457, !dbg !20
  %address.local471 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local470, i64 160, !dbg !20
  store half %1440, ptr addrspace(1) %address.local471, align 2, !dbg !20
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local457, !dbg !20
  %address.local474 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local473, i64 162, !dbg !20
  store half %1439, ptr addrspace(1) %address.local474, align 2, !dbg !20
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local457, !dbg !20
  %address.local477 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local476, i64 164, !dbg !20
  store half %1438, ptr addrspace(1) %address.local477, align 2, !dbg !20
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !20
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local457, !dbg !20
  %address.local480 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local479, i64 166, !dbg !20
  store half %1437, ptr addrspace(1) %address.local480, align 2, !dbg !20
  br label %1451, !dbg !20

1451:                                             ; preds = %1431, %1430
  br i1 true, label %1452, label %1472, !dbg !20

1452:                                             ; preds = %1451
  %1453 = insertelement <4 x float> poison, float %1289, i64 0, !dbg !18
  %1454 = insertelement <4 x float> %1453, float %1295, i64 1, !dbg !18
  %1455 = insertelement <4 x float> %1454, float %1290, i64 2, !dbg !18
  %1456 = insertelement <4 x float> %1455, float %1296, i64 3, !dbg !18
  %row.reciprocal43 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1268, !dbg !18
  %normalized44 = fmul <4 x float> %1456, %row.reciprocal43, !dbg !18
  %1457 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !19
  %1458 = extractelement <4 x half> %1457, i64 3, !dbg !19
  %1459 = extractelement <4 x half> %1457, i64 2, !dbg !19
  %1460 = extractelement <4 x half> %1457, i64 1, !dbg !19
  %1461 = extractelement <4 x half> %1457, i64 0, !dbg !19
  %1462 = insertelement <4 x float> poison, float %1182, i64 0, !dbg !18
  %1463 = insertelement <4 x float> %1462, float %1188, i64 1, !dbg !18
  %1464 = insertelement <4 x float> %1463, float %1183, i64 2, !dbg !18
  %1465 = insertelement <4 x float> %1464, float %1189, i64 3, !dbg !18
  %normalized45 = fmul <4 x float> %1465, %row.reciprocal43, !dbg !18
  %1466 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !19
  %1467 = extractelement <4 x half> %1466, i64 3, !dbg !19
  %1468 = extractelement <4 x half> %1466, i64 2, !dbg !19
  %1469 = extractelement <4 x half> %1466, i64 1, !dbg !19
  %1470 = extractelement <4 x half> %1466, i64 0, !dbg !19
  %1471 = shl nuw nsw i64 %27, 19, !dbg !20
  %address.thread481 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !20, !range !38, !frisk.register_lifetimes !1
  %address.local482 = zext nneg i32 %address.thread481 to i64, !dbg !12
  %address.local483 = add i64 %28, %address.local482, !dbg !20
  %address.local484 = lshr i64 %address.local482, 6, !dbg !12
  %address.local485 = shl nuw nsw i64 %address.local484, 4, !dbg !20
  %address.local486 = add i64 %address.local483, %address.local485, !dbg !20
  %address.local487 = and i64 %address.local482, 240, !dbg !20
  %address.local488 = sub i64 %address.local486, %address.local487, !dbg !20
  %address.local489 = add i64 %address.local488, %29, !dbg !20
  %address.local490 = shl nuw nsw i64 %address.local489, 7, !dbg !20
  %address.local491 = add nuw nsw i64 %1471, %address.local490, !dbg !20
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local493 = lshr i64 %address.local482, 2, !dbg !12
  %address.local494 = and i64 %address.local493, 12, !dbg !12
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local494, !dbg !20
  %address.local496 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local495, i64 192, !dbg !20
  store half %1470, ptr addrspace(1) %address.local496, align 2, !dbg !20
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local494, !dbg !20
  %address.local499 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local498, i64 194, !dbg !20
  store half %1469, ptr addrspace(1) %address.local499, align 2, !dbg !20
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local500, i64 %address.local494, !dbg !20
  %address.local502 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local501, i64 196, !dbg !20
  store half %1468, ptr addrspace(1) %address.local502, align 2, !dbg !20
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local503, i64 %address.local494, !dbg !20
  %address.local505 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local504, i64 198, !dbg !20
  store half %1467, ptr addrspace(1) %address.local505, align 2, !dbg !20
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local507 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local506, i64 %address.local494, !dbg !20
  %address.local508 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local507, i64 224, !dbg !20
  store half %1461, ptr addrspace(1) %address.local508, align 2, !dbg !20
  %address.local509 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local510 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local509, i64 %address.local494, !dbg !20
  %address.local511 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local510, i64 226, !dbg !20
  store half %1460, ptr addrspace(1) %address.local511, align 2, !dbg !20
  %address.local512 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local513 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local512, i64 %address.local494, !dbg !20
  %address.local514 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local513, i64 228, !dbg !20
  store half %1459, ptr addrspace(1) %address.local514, align 2, !dbg !20
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local491, !dbg !20
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local515, i64 %address.local494, !dbg !20
  %address.local517 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local516, i64 230, !dbg !20
  store half %1458, ptr addrspace(1) %address.local517, align 2, !dbg !20
  br label %1472, !dbg !20

1472:                                             ; preds = %1452, %1451
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/group8-reverse")
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
