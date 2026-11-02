; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/group8-reverse-k128-swizzle/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/group8-reverse-k128-swizzle/input.mlir"

@shm_0 = addrspace(3) global [10240 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
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
  %34 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !39
  %35 = zext nneg i32 %34 to i64, !dbg !12
  %36 = lshr i64 %35, 6, !dbg !12
  %37 = lshr i64 %35, 2, !dbg !12
  %38 = and i64 %37, 12, !dbg !12
  %39 = and i32 %34, 16, !dbg !13
  %.not = icmp eq i32 %39, 0, !dbg !13
  %40 = and i32 %34, 32, !dbg !13
  %.not1 = icmp eq i32 %40, 0, !dbg !13
  %41 = or disjoint i64 %19, 127, !dbg !14
  %42 = add i64 %19, 128, !dbg !14
  %43 = icmp uge i64 %42, %41, !dbg !14
  %44 = shl nsw i64 %16, 2, !dbg !7
  %45 = add nsw i64 %44, 4, !dbg !7
  %46 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %47 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %48 = getelementptr i8, ptr addrspace(1) %47, i64 64, !dbg !11
  %49 = shl i64 %15, 7, !dbg !17
  %50 = and i64 %49, 3968, !dbg !17
  %51 = shl nsw i64 %18, 12, !dbg !17
  %52 = or disjoint i64 %50, %51, !dbg !17
  %53 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %54 = getelementptr i8, ptr addrspace(1) %53, i64 128, !dbg !11
  %55 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %33, !dbg !11
  %56 = getelementptr i8, ptr addrspace(1) %55, i64 192, !dbg !11
  br label %57, !dbg !7

57:                                               ; preds = %1056, %4
  %58 = phi i64 [ %1127, %1056 ], [ 0, %4 ], !dbg !7
  %59 = phi [2 x <8 x float>] [ %1108, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %60 = phi [2 x <1 x float>] [ %1112, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %61 = phi [2 x <1 x float>] [ %1114, %1056 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %62 = phi [2 x <8 x float>] [ %1118, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %63 = phi [2 x <8 x float>] [ %1122, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %64 = phi [2 x <8 x float>] [ %1126, %1056 ], [ zeroinitializer, %4 ], !dbg !7
  %65 = icmp slt i64 %58, %45, !dbg !7
  br i1 %65, label %66, label %1128, !dbg !7

66:                                               ; preds = %57
  %67 = shl nsw i64 %58, 5, !dbg !7
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !11, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local55 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local56 = and i64 %address.local55, 1536, !dbg !11
  %address.local57 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local56, !dbg !11
  %address.local58 = ptrtoint ptr addrspace(3) %address.local57 to i64, !dbg !11
  %address.local59 = trunc i64 %address.local58 to i32, !dbg !11
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local59), !dbg !11
  %69 = zext i32 %68 to i64, !dbg !11
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !11
  %address.local60 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local61 = and i64 %address.local60, 16256, !dbg !11
  %address.local62 = lshr i64 %address.local, 4, !dbg !11
  %address.local63 = and i64 %address.local62, 8, !dbg !11
  %address.local64 = or disjoint i64 %address.local61, %address.local63, !dbg !11
  %address.local65 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local64, !dbg !11
  %71 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %72 = ptrtoint ptr addrspace(1) %address.local65 to i64, !dbg !11
  %73 = sub i64 %72, %71, !dbg !11
  %dma.buffer.offset767 = trunc i64 %73 to i32, !dbg !11
  %74 = trunc i64 %71 to i32, !dbg !11
  %75 = insertelement <4 x i32> poison, i32 %74, i64 0, !dbg !11
  %76 = lshr i64 %71, 32, !dbg !11
  %77 = trunc i64 %76 to i32, !dbg !11
  %78 = insertelement <4 x i32> %75, i32 %77, i64 1, !dbg !11
  %79 = insertelement <4 x i32> %78, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc768 = insertelement <4 x i32> %79, i32 131072, i64 3, !dbg !11
  %80 = ptrtoint ptr addrspace(3) %70 to i32, !dbg !11
  %dma.buffer.lds769 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %80), !dbg !11
  %81 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds769, i32 %dma.buffer.offset767, <4 x i32> %dma.buffer.rsrc768) #10, !dbg !11
  %address.local66 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local67 = shl nuw nsw i64 %address.local66, 3, !dbg !11
  %address.local68 = and i64 %address.local67, 3584, !dbg !11
  %address.local69 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local68, !dbg !11
  %address.local70 = ptrtoint ptr addrspace(3) %address.local69 to i64, !dbg !11
  %address.local71 = trunc i64 %address.local70 to i32, !dbg !11
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local71), !dbg !11
  %83 = zext i32 %82 to i64, !dbg !11
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !11
  %address.local72 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local73 = and i64 %address.local72, 16256, !dbg !11
  %address.local74 = lshr i64 %address.local66, 4, !dbg !11
  %address.local75 = and i64 %address.local74, 24, !dbg !11
  %address.local76 = or disjoint i64 %address.local73, %address.local75, !dbg !11
  %address.local77 = getelementptr [2 x i8], ptr addrspace(1) %46, i64 %address.local76, !dbg !11
  %85 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %86 = ptrtoint ptr addrspace(1) %address.local77 to i64, !dbg !11
  %87 = sub i64 %86, %85, !dbg !11
  %dma.buffer.offset764 = trunc i64 %87 to i32, !dbg !11
  %88 = trunc i64 %85 to i32, !dbg !11
  %89 = insertelement <4 x i32> poison, i32 %88, i64 0, !dbg !11
  %90 = lshr i64 %85, 32, !dbg !11
  %91 = trunc i64 %90 to i32, !dbg !11
  %92 = insertelement <4 x i32> %89, i32 %91, i64 1, !dbg !11
  %93 = insertelement <4 x i32> %92, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc765 = insertelement <4 x i32> %93, i32 131072, i64 3, !dbg !11
  %94 = ptrtoint ptr addrspace(3) %84 to i32, !dbg !11
  %dma.buffer.lds766 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %94), !dbg !11
  %95 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds766, i32 %dma.buffer.offset764, <4 x i32> %dma.buffer.rsrc765) #10, !dbg !11
  %96 = lshr i64 %58, 14, !dbg !17
  %97 = add nuw nsw i64 %96, %15, !dbg !17
  %98 = lshr i64 %58, 7, !dbg !17
  %99 = and i64 %98, 127, !dbg !17
  %100 = shl i64 %97, 19, !dbg !17
  %101 = and i64 %100, 16252928, !dbg !17
  %102 = shl i64 %98, 19, !dbg !17
  %103 = shl i64 %58, 12, !dbg !17
  %104 = sub i64 %103, %102, !dbg !17
  %105 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %101, !dbg !17
  %106 = getelementptr [2 x i8], ptr addrspace(1) %105, i64 %99, !dbg !17
  %107 = getelementptr [2 x i8], ptr addrspace(1) %106, i64 %104, !dbg !17
  br label %108, !dbg !17

108:                                              ; preds = %111, %66
  %109 = phi i64 [ %141, %111 ], [ %35, %66 ], !dbg !17
  %110 = icmp samesign ult i64 %109, 128, !dbg !17
  br i1 %110, label %111, label %142, !dbg !17

111:                                              ; preds = %108
  %112 = and i64 %109, 27, !dbg !17
  %113 = lshr i64 %109, 2, !dbg !17
  %114 = and i64 %113, 4, !dbg !17
  %115 = add nuw nsw i64 %109, %114, !dbg !17
  %116 = and i64 %115, 4, !dbg !17
  %117 = or disjoint i64 %112, %116, !dbg !17
  %118 = lshr i64 %109, 2, !dbg !17
  %119 = and i64 %118, 24, !dbg !17
  %120 = shl nuw nsw i64 %109, 3, !dbg !17
  %121 = and i64 %120, 512, !dbg !17
  %122 = getelementptr [2 x i8], ptr addrspace(1) %107, i64 %119, !dbg !17
  %.idx22 = shl nuw nsw i64 %117, 8, !dbg !17
  %123 = getelementptr i8, ptr addrspace(1) %122, i64 %.idx22, !dbg !17
  %124 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %121, !dbg !17
  %125 = ptrtoint ptr addrspace(3) %124 to i64, !dbg !17
  %126 = trunc i64 %125 to i32, !dbg !17
  %127 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %126), !dbg !17
  %128 = zext i32 %127 to i64, !dbg !17
  %129 = inttoptr i64 %128 to ptr addrspace(3), !dbg !17
  %130 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %131 = ptrtoint ptr addrspace(1) %123 to i64, !dbg !17
  %132 = sub i64 %131, %130, !dbg !17
  %dma.buffer.offset = trunc i64 %132 to i32, !dbg !17
  %133 = trunc i64 %130 to i32, !dbg !17
  %134 = insertelement <4 x i32> poison, i32 %133, i64 0, !dbg !17
  %135 = lshr i64 %130, 32, !dbg !17
  %136 = trunc i64 %135 to i32, !dbg !17
  %137 = insertelement <4 x i32> %134, i32 %136, i64 1, !dbg !17
  %138 = insertelement <4 x i32> %137, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc = insertelement <4 x i32> %138, i32 131072, i64 3, !dbg !17
  %139 = ptrtoint ptr addrspace(3) %129 to i32, !dbg !17
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !17
  %140 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !17
  %141 = or disjoint i64 %35, 256, !dbg !17
  br label %108, !dbg !17

142:                                              ; preds = %108
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.thread78 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %address.local79 = lshr i32 %address.thread78, 2, !dbg !12
  %address.local80 = and i32 %address.local79, 4, !dbg !12
  %address.local81 = shl nuw nsw i32 %address.thread78, 1, !dbg !12
  %address.local82 = and i32 %address.local81, 384, !dbg !12
  %address.local83 = or disjoint i32 %address.local80, %address.local82, !dbg !12
  %address.local84 = shl nuw nsw i32 %address.thread78, 3, !dbg !12
  %address.local85 = and i32 %address.local84, 120, !dbg !12
  %address.local86 = or disjoint i32 %address.local83, %address.local85, !dbg !12
  %address.local87 = shl nuw nsw i32 %address.thread78, 5, !dbg !12
  %address.local88 = and i32 %address.local87, 1024, !dbg !12
  %address.local89 = or disjoint i32 %address.local86, %address.local88, !dbg !12
  %address.local90 = zext nneg i32 %address.local89 to i64, !dbg !12
  %address.local91 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local90, !dbg !12
  %143 = load <4 x half>, ptr addrspace(3) %address.local91, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local91, i64 4096, !dbg !12
  %144 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %lds.const.gep683 = getelementptr i8, ptr addrspace(3) %address.local91, i64 1024, !dbg !12
  %145 = load <4 x half>, ptr addrspace(3) %lds.const.gep683, align 8, !dbg !12
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !12
  %lds.const.gep684 = getelementptr i8, ptr addrspace(3) %address.local91, i64 5120, !dbg !12
  %147 = load <4 x half>, ptr addrspace(3) %lds.const.gep684, align 8, !dbg !12
  %148 = fmul <4 x half> %144, splat (half 0xH3015), !dbg !12
  %149 = fmul <4 x half> %145, splat (half 0xH3015), !dbg !12
  %150 = fmul <4 x half> %147, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local98 = zext nneg i32 %address.thread78 to i64, !dbg !12
  %address.local99 = shl nuw nsw i64 %address.local98, 3, !dbg !12
  %address.local100 = lshr i64 %address.local98, 4, !dbg !12
  %address.local101 = mul nsw i64 %address.local100, -128, !dbg !12
  %address.local102 = add nsw i64 %address.local99, %address.local101, !dbg !16
  %address.local103 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local102, !dbg !16
  %address.local104 = lshr i64 %address.local98, 2, !dbg !12
  %address.local105 = and i64 %address.local104, 12, !dbg !12
  %address.local106 = getelementptr [2 x i8], ptr addrspace(3) %address.local103, i64 %address.local105, !dbg !16
  %address.local107 = and i64 %address.local98, 32, !dbg !16
  %address.local108 = icmp eq i64 %address.local107, 0, !dbg !16
  %address.local109 = select i1 %address.local108, i64 0, i64 248, !dbg !16
  %address.local110 = getelementptr [2 x i8], ptr addrspace(3) %address.local106, i64 %address.local109, !dbg !16
  %151 = load <4 x half>, ptr addrspace(3) %address.local110, align 8, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep685 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep685, align 8, !dbg !16
  %address.local116 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local102, !dbg !16
  %address.local117 = shl i64 %address.local98, 61, !dbg !16
  %address.local118 = ashr exact i64 %address.local117, 57, !dbg !16
  %address.local119 = and i64 %address.local118, -64, !dbg !16
  %address.local120 = getelementptr [2 x i8], ptr addrspace(3) %address.local116, i64 %address.local119, !dbg !16
  %address.local121 = getelementptr [2 x i8], ptr addrspace(3) %address.local120, i64 %address.local105, !dbg !16
  %address.local122 = getelementptr [2 x i8], ptr addrspace(3) %address.local121, i64 %address.local109, !dbg !16
  %address.local123 = getelementptr i8, ptr addrspace(3) %address.local122, i64 320, !dbg !16
  %154 = load <4 x half>, ptr addrspace(3) %address.local123, align 8, !dbg !16
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %153, <4 x float> %152) #9, !dbg !16
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep686 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep686, align 8, !dbg !16
  %158 = load <4 x half>, ptr addrspace(3) %address.local110, align 8, !dbg !16
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %157, <4 x float> %156) #9, !dbg !16
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep687 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep687, align 8, !dbg !16
  %162 = load <4 x half>, ptr addrspace(3) %address.local123, align 8, !dbg !16
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %161, <4 x float> %160) #9, !dbg !16
  %lds.const.gep688 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep688, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %150, <4 x half> %164, <4 x float> %165) #9, !dbg !16
  %address.local133 = shl nuw nsw i64 %address.local98, 3, !dbg !11
  %address.local134 = and i64 %address.local133, 1536, !dbg !11
  %address.local135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local134, !dbg !11
  %address.local136 = ptrtoint ptr addrspace(3) %address.local135 to i64, !dbg !11
  %address.local137 = trunc i64 %address.local136 to i32, !dbg !11
  %167 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local137), !dbg !11
  %168 = zext i32 %167 to i64, !dbg !11
  %169 = inttoptr i64 %168 to ptr addrspace(3), !dbg !11
  %address.local138 = shl nuw nsw i64 %address.local98, 7, !dbg !11
  %address.local139 = and i64 %address.local138, 16256, !dbg !11
  %address.local140 = lshr i64 %address.local98, 4, !dbg !11
  %address.local141 = and i64 %address.local140, 8, !dbg !11
  %address.local142 = or disjoint i64 %address.local139, %address.local141, !dbg !11
  %address.local143 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local142, !dbg !11
  %170 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %171 = ptrtoint ptr addrspace(1) %address.local143 to i64, !dbg !11
  %172 = sub i64 %171, %170, !dbg !11
  %dma.buffer.offset761 = trunc i64 %172 to i32, !dbg !11
  %173 = trunc i64 %170 to i32, !dbg !11
  %174 = insertelement <4 x i32> poison, i32 %173, i64 0, !dbg !11
  %175 = lshr i64 %170, 32, !dbg !11
  %176 = trunc i64 %175 to i32, !dbg !11
  %177 = insertelement <4 x i32> %174, i32 %176, i64 1, !dbg !11
  %178 = insertelement <4 x i32> %177, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc762 = insertelement <4 x i32> %178, i32 131072, i64 3, !dbg !11
  %179 = ptrtoint ptr addrspace(3) %169 to i32, !dbg !11
  %dma.buffer.lds763 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %179), !dbg !11
  %180 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds763, i32 %dma.buffer.offset761, <4 x i32> %dma.buffer.rsrc762) #10, !dbg !11
  %address.local144 = or disjoint i64 %address.local98, 256, !dbg !11
  %address.local145 = shl nuw nsw i64 %address.local144, 3, !dbg !11
  %address.local146 = and i64 %address.local145, 3584, !dbg !11
  %address.local147 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local146, !dbg !11
  %address.local148 = ptrtoint ptr addrspace(3) %address.local147 to i64, !dbg !11
  %address.local149 = trunc i64 %address.local148 to i32, !dbg !11
  %181 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local149), !dbg !11
  %182 = zext i32 %181 to i64, !dbg !11
  %183 = inttoptr i64 %182 to ptr addrspace(3), !dbg !11
  %address.local150 = shl nuw nsw i64 %address.local98, 7, !dbg !11
  %address.local151 = and i64 %address.local150, 16256, !dbg !11
  %address.local152 = lshr i64 %address.local144, 4, !dbg !11
  %address.local153 = and i64 %address.local152, 24, !dbg !11
  %address.local154 = or disjoint i64 %address.local151, %address.local153, !dbg !11
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %48, i64 %address.local154, !dbg !11
  %184 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %185 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !11
  %186 = sub i64 %185, %184, !dbg !11
  %dma.buffer.offset758 = trunc i64 %186 to i32, !dbg !11
  %187 = trunc i64 %184 to i32, !dbg !11
  %188 = insertelement <4 x i32> poison, i32 %187, i64 0, !dbg !11
  %189 = lshr i64 %184, 32, !dbg !11
  %190 = trunc i64 %189 to i32, !dbg !11
  %191 = insertelement <4 x i32> %188, i32 %190, i64 1, !dbg !11
  %192 = insertelement <4 x i32> %191, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc759 = insertelement <4 x i32> %192, i32 131072, i64 3, !dbg !11
  %193 = ptrtoint ptr addrspace(3) %183 to i32, !dbg !11
  %dma.buffer.lds760 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %193), !dbg !11
  %194 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds760, i32 %dma.buffer.offset758, <4 x i32> %dma.buffer.rsrc759) #10, !dbg !11
  %195 = add nuw i64 %58, 4096, !dbg !17
  %196 = icmp slt i64 %195, 0, !dbg !17
  %197 = sub nuw i64 -4097, %58, !dbg !17
  %198 = select i1 %196, i64 %197, i64 %195, !dbg !17
  %199 = sdiv i64 %198, 16384, !dbg !17
  %.lobit = ashr i64 %195, 63, !dbg !17
  %200 = xor i64 %199, %.lobit, !dbg !17
  %201 = add nsw i64 %200, %15, !dbg !17
  %202 = add nsw i64 %98, %52, !dbg !17
  %203 = add nsw i64 %202, 32, !dbg !17
  %204 = icmp slt i64 %202, -32, !dbg !17
  %205 = sub nsw i64 -33, %202, !dbg !17
  %206 = select i1 %204, i64 %205, i64 %203, !dbg !17
  %207 = sdiv i64 %206, 128, !dbg !17
  %208 = sext i1 %204 to i64, !dbg !17
  %209 = xor i64 %207, %208, !dbg !17
  %210 = shl nsw i64 %209, 7, !dbg !17
  %211 = sub i64 %52, %210, !dbg !17
  %212 = shl i64 %201, 19, !dbg !17
  %213 = and i64 %212, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %214 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %213, !dbg !17
  %215 = getelementptr [2 x i8], ptr addrspace(1) %214, i64 %98, !dbg !17
  %216 = getelementptr [2 x i8], ptr addrspace(1) %215, i64 %211, !dbg !17
  %217 = getelementptr i8, ptr addrspace(1) %216, i64 64, !dbg !17
  %218 = getelementptr [2 x i8], ptr addrspace(1) %217, i64 %104, !dbg !17
  br label %219, !dbg !17

219:                                              ; preds = %222, %142
  %220 = phi i64 [ %252, %222 ], [ %35, %142 ], !dbg !17
  %221 = icmp samesign ult i64 %220, 128, !dbg !17
  br i1 %221, label %222, label %253, !dbg !17

222:                                              ; preds = %219
  %223 = and i64 %220, 27, !dbg !17
  %224 = lshr i64 %220, 2, !dbg !17
  %225 = and i64 %224, 4, !dbg !17
  %226 = add nuw nsw i64 %220, %225, !dbg !17
  %227 = and i64 %226, 4, !dbg !17
  %228 = or disjoint i64 %223, %227, !dbg !17
  %229 = lshr i64 %220, 2, !dbg !17
  %230 = and i64 %229, 24, !dbg !17
  %231 = shl nuw nsw i64 %220, 3, !dbg !17
  %232 = and i64 %231, 512, !dbg !17
  %233 = getelementptr [2 x i8], ptr addrspace(1) %218, i64 %230, !dbg !17
  %.idx21 = shl nuw nsw i64 %228, 8, !dbg !17
  %234 = getelementptr i8, ptr addrspace(1) %233, i64 %.idx21, !dbg !17
  %235 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %232, !dbg !17
  %236 = ptrtoint ptr addrspace(3) %235 to i64, !dbg !17
  %237 = trunc i64 %236 to i32, !dbg !17
  %238 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %237), !dbg !17
  %239 = zext i32 %238 to i64, !dbg !17
  %240 = inttoptr i64 %239 to ptr addrspace(3), !dbg !17
  %241 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %242 = ptrtoint ptr addrspace(1) %234 to i64, !dbg !17
  %243 = sub i64 %242, %241, !dbg !17
  %dma.buffer.offset725 = trunc i64 %243 to i32, !dbg !17
  %244 = trunc i64 %241 to i32, !dbg !17
  %245 = insertelement <4 x i32> poison, i32 %244, i64 0, !dbg !17
  %246 = lshr i64 %241, 32, !dbg !17
  %247 = trunc i64 %246 to i32, !dbg !17
  %248 = insertelement <4 x i32> %245, i32 %247, i64 1, !dbg !17
  %249 = insertelement <4 x i32> %248, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc726 = insertelement <4 x i32> %249, i32 131072, i64 3, !dbg !17
  %250 = ptrtoint ptr addrspace(3) %240 to i32, !dbg !17
  %dma.buffer.lds727 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %250), !dbg !17
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds727, i32 %dma.buffer.offset725, <4 x i32> %dma.buffer.rsrc726) #10, !dbg !17
  %252 = or disjoint i64 %35, 256, !dbg !17
  br label %219, !dbg !17

253:                                              ; preds = %219
  %address.thread156 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %lds.const.gep689 = getelementptr i8, ptr addrspace(3) %address.local91, i64 0, !dbg !12
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep689, align 8, !dbg !12
  %lds.const.gep690 = getelementptr i8, ptr addrspace(3) %address.local91, i64 4096, !dbg !12
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep690, align 8, !dbg !12
  %lds.const.gep691 = getelementptr i8, ptr addrspace(3) %address.local91, i64 1024, !dbg !12
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep691, align 8, !dbg !12
  %257 = fmul <4 x half> %254, splat (half 0xH3015), !dbg !12
  %lds.const.gep692 = getelementptr i8, ptr addrspace(3) %address.local91, i64 5120, !dbg !12
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep692, align 8, !dbg !12
  %259 = fmul <4 x half> %255, splat (half 0xH3015), !dbg !12
  %260 = fmul <4 x half> %256, splat (half 0xH3015), !dbg !12
  %261 = fmul <4 x half> %258, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local176 = zext nneg i32 %address.thread156 to i64, !dbg !12
  %lds.const.gep693 = getelementptr i8, ptr addrspace(3) %address.local110, i64 0, !dbg !16
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep693, align 8, !dbg !16
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %257, <4 x half> %262, <4 x float> %155) #9, !dbg !16
  %lds.const.gep694 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %264 = load <4 x half>, ptr addrspace(3) %lds.const.gep694, align 8, !dbg !16
  %lds.const.gep695 = getelementptr i8, ptr addrspace(3) %address.local123, i64 0, !dbg !16
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep695, align 8, !dbg !16
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %259, <4 x half> %264, <4 x float> %263) #9, !dbg !16
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %257, <4 x half> %265, <4 x float> %159) #9, !dbg !16
  %lds.const.gep696 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %268 = load <4 x half>, ptr addrspace(3) %lds.const.gep696, align 8, !dbg !16
  %lds.const.gep697 = getelementptr i8, ptr addrspace(3) %address.local110, i64 0, !dbg !16
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep697, align 8, !dbg !16
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %259, <4 x half> %268, <4 x float> %267) #9, !dbg !16
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %260, <4 x half> %269, <4 x float> %163) #9, !dbg !16
  %lds.const.gep698 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep698, align 8, !dbg !16
  %lds.const.gep699 = getelementptr i8, ptr addrspace(3) %address.local123, i64 0, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep699, align 8, !dbg !16
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %261, <4 x half> %272, <4 x float> %271) #9, !dbg !16
  %lds.const.gep700 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %275 = load <4 x half>, ptr addrspace(3) %lds.const.gep700, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %260, <4 x half> %273, <4 x float> %166) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %261, <4 x half> %275, <4 x float> %276) #9, !dbg !16
  %address.local211 = shl nuw nsw i64 %address.local176, 3, !dbg !11
  %address.local212 = and i64 %address.local211, 1536, !dbg !11
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local212, !dbg !11
  %address.local214 = ptrtoint ptr addrspace(3) %address.local213 to i64, !dbg !11
  %address.local215 = trunc i64 %address.local214 to i32, !dbg !11
  %278 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local215), !dbg !11
  %279 = zext i32 %278 to i64, !dbg !11
  %280 = inttoptr i64 %279 to ptr addrspace(3), !dbg !11
  %address.local216 = shl nuw nsw i64 %address.local176, 7, !dbg !11
  %address.local217 = and i64 %address.local216, 16256, !dbg !11
  %address.local218 = lshr i64 %address.local176, 4, !dbg !11
  %address.local219 = and i64 %address.local218, 8, !dbg !11
  %address.local220 = or disjoint i64 %address.local217, %address.local219, !dbg !11
  %address.local221 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local220, !dbg !11
  %281 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %282 = ptrtoint ptr addrspace(1) %address.local221 to i64, !dbg !11
  %283 = sub i64 %282, %281, !dbg !11
  %dma.buffer.offset755 = trunc i64 %283 to i32, !dbg !11
  %284 = trunc i64 %281 to i32, !dbg !11
  %285 = insertelement <4 x i32> poison, i32 %284, i64 0, !dbg !11
  %286 = lshr i64 %281, 32, !dbg !11
  %287 = trunc i64 %286 to i32, !dbg !11
  %288 = insertelement <4 x i32> %285, i32 %287, i64 1, !dbg !11
  %289 = insertelement <4 x i32> %288, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc756 = insertelement <4 x i32> %289, i32 131072, i64 3, !dbg !11
  %290 = ptrtoint ptr addrspace(3) %280 to i32, !dbg !11
  %dma.buffer.lds757 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %290), !dbg !11
  %291 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds757, i32 %dma.buffer.offset755, <4 x i32> %dma.buffer.rsrc756) #10, !dbg !11
  %address.local222 = or disjoint i64 %address.local176, 256, !dbg !11
  %address.local223 = shl nuw nsw i64 %address.local222, 3, !dbg !11
  %address.local224 = and i64 %address.local223, 3584, !dbg !11
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local224, !dbg !11
  %address.local226 = ptrtoint ptr addrspace(3) %address.local225 to i64, !dbg !11
  %address.local227 = trunc i64 %address.local226 to i32, !dbg !11
  %292 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local227), !dbg !11
  %293 = zext i32 %292 to i64, !dbg !11
  %294 = inttoptr i64 %293 to ptr addrspace(3), !dbg !11
  %address.local228 = shl nuw nsw i64 %address.local176, 7, !dbg !11
  %address.local229 = and i64 %address.local228, 16256, !dbg !11
  %address.local230 = lshr i64 %address.local222, 4, !dbg !11
  %address.local231 = and i64 %address.local230, 24, !dbg !11
  %address.local232 = or disjoint i64 %address.local229, %address.local231, !dbg !11
  %address.local233 = getelementptr [2 x i8], ptr addrspace(1) %54, i64 %address.local232, !dbg !11
  %295 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %296 = ptrtoint ptr addrspace(1) %address.local233 to i64, !dbg !11
  %297 = sub i64 %296, %295, !dbg !11
  %dma.buffer.offset752 = trunc i64 %297 to i32, !dbg !11
  %298 = trunc i64 %295 to i32, !dbg !11
  %299 = insertelement <4 x i32> poison, i32 %298, i64 0, !dbg !11
  %300 = lshr i64 %295, 32, !dbg !11
  %301 = trunc i64 %300 to i32, !dbg !11
  %302 = insertelement <4 x i32> %299, i32 %301, i64 1, !dbg !11
  %303 = insertelement <4 x i32> %302, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc753 = insertelement <4 x i32> %303, i32 131072, i64 3, !dbg !11
  %304 = ptrtoint ptr addrspace(3) %294 to i32, !dbg !11
  %dma.buffer.lds754 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %304), !dbg !11
  %305 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds754, i32 %dma.buffer.offset752, <4 x i32> %dma.buffer.rsrc753) #10, !dbg !11
  %306 = add nuw i64 %58, 8192, !dbg !17
  %307 = icmp slt i64 %306, 0, !dbg !17
  %308 = sub nuw i64 -8193, %58, !dbg !17
  %309 = select i1 %307, i64 %308, i64 %306, !dbg !17
  %310 = sdiv i64 %309, 16384, !dbg !17
  %.lobit2 = ashr i64 %306, 63, !dbg !17
  %311 = xor i64 %310, %.lobit2, !dbg !17
  %312 = add nsw i64 %311, %15, !dbg !17
  %313 = add nsw i64 %202, 64, !dbg !17
  %314 = icmp slt i64 %202, -64, !dbg !17
  %315 = sub nsw i64 -65, %202, !dbg !17
  %316 = select i1 %314, i64 %315, i64 %313, !dbg !17
  %317 = sdiv i64 %316, 128, !dbg !17
  %318 = sext i1 %314 to i64, !dbg !17
  %319 = xor i64 %317, %318, !dbg !17
  %320 = shl nsw i64 %319, 7, !dbg !17
  %321 = sub i64 %52, %320, !dbg !17
  %322 = shl i64 %312, 19, !dbg !17
  %323 = and i64 %322, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %324 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %323, !dbg !17
  %325 = getelementptr [2 x i8], ptr addrspace(1) %324, i64 %98, !dbg !17
  %326 = getelementptr [2 x i8], ptr addrspace(1) %325, i64 %321, !dbg !17
  %327 = getelementptr i8, ptr addrspace(1) %326, i64 128, !dbg !17
  %328 = getelementptr [2 x i8], ptr addrspace(1) %327, i64 %104, !dbg !17
  br label %329, !dbg !17

329:                                              ; preds = %332, %253
  %330 = phi i64 [ %362, %332 ], [ %35, %253 ], !dbg !17
  %331 = icmp samesign ult i64 %330, 128, !dbg !17
  br i1 %331, label %332, label %363, !dbg !17

332:                                              ; preds = %329
  %333 = and i64 %330, 27, !dbg !17
  %334 = lshr i64 %330, 2, !dbg !17
  %335 = and i64 %334, 4, !dbg !17
  %336 = add nuw nsw i64 %330, %335, !dbg !17
  %337 = and i64 %336, 4, !dbg !17
  %338 = or disjoint i64 %333, %337, !dbg !17
  %339 = lshr i64 %330, 2, !dbg !17
  %340 = and i64 %339, 24, !dbg !17
  %341 = shl nuw nsw i64 %330, 3, !dbg !17
  %342 = and i64 %341, 512, !dbg !17
  %343 = getelementptr [2 x i8], ptr addrspace(1) %328, i64 %340, !dbg !17
  %.idx20 = shl nuw nsw i64 %338, 8, !dbg !17
  %344 = getelementptr i8, ptr addrspace(1) %343, i64 %.idx20, !dbg !17
  %345 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %342, !dbg !17
  %346 = ptrtoint ptr addrspace(3) %345 to i64, !dbg !17
  %347 = trunc i64 %346 to i32, !dbg !17
  %348 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %347), !dbg !17
  %349 = zext i32 %348 to i64, !dbg !17
  %350 = inttoptr i64 %349 to ptr addrspace(3), !dbg !17
  %351 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %352 = ptrtoint ptr addrspace(1) %344 to i64, !dbg !17
  %353 = sub i64 %352, %351, !dbg !17
  %dma.buffer.offset728 = trunc i64 %353 to i32, !dbg !17
  %354 = trunc i64 %351 to i32, !dbg !17
  %355 = insertelement <4 x i32> poison, i32 %354, i64 0, !dbg !17
  %356 = lshr i64 %351, 32, !dbg !17
  %357 = trunc i64 %356 to i32, !dbg !17
  %358 = insertelement <4 x i32> %355, i32 %357, i64 1, !dbg !17
  %359 = insertelement <4 x i32> %358, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc729 = insertelement <4 x i32> %359, i32 131072, i64 3, !dbg !17
  %360 = ptrtoint ptr addrspace(3) %350 to i32, !dbg !17
  %dma.buffer.lds730 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %360), !dbg !17
  %361 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds730, i32 %dma.buffer.offset728, <4 x i32> %dma.buffer.rsrc729) #10, !dbg !17
  %362 = or disjoint i64 %35, 256, !dbg !17
  br label %329, !dbg !17

363:                                              ; preds = %329
  %address.thread234 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !12, !range !39, !frisk.register_lifetimes !1
  %lds.const.gep701 = getelementptr i8, ptr addrspace(3) %address.local91, i64 0, !dbg !12
  %364 = load <4 x half>, ptr addrspace(3) %lds.const.gep701, align 8, !dbg !12
  %lds.const.gep702 = getelementptr i8, ptr addrspace(3) %address.local91, i64 4096, !dbg !12
  %365 = load <4 x half>, ptr addrspace(3) %lds.const.gep702, align 8, !dbg !12
  %lds.const.gep703 = getelementptr i8, ptr addrspace(3) %address.local91, i64 1024, !dbg !12
  %366 = load <4 x half>, ptr addrspace(3) %lds.const.gep703, align 8, !dbg !12
  %367 = fmul <4 x half> %364, splat (half 0xH3015), !dbg !12
  %lds.const.gep704 = getelementptr i8, ptr addrspace(3) %address.local91, i64 5120, !dbg !12
  %368 = load <4 x half>, ptr addrspace(3) %lds.const.gep704, align 8, !dbg !12
  %369 = fmul <4 x half> %365, splat (half 0xH3015), !dbg !12
  %370 = fmul <4 x half> %366, splat (half 0xH3015), !dbg !12
  %371 = fmul <4 x half> %368, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.local254 = zext nneg i32 %address.thread234 to i64, !dbg !12
  %lds.const.gep705 = getelementptr i8, ptr addrspace(3) %address.local110, i64 0, !dbg !16
  %372 = load <4 x half>, ptr addrspace(3) %lds.const.gep705, align 8, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %372, <4 x float> %266) #9, !dbg !16
  %lds.const.gep706 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %374 = load <4 x half>, ptr addrspace(3) %lds.const.gep706, align 8, !dbg !16
  %lds.const.gep707 = getelementptr i8, ptr addrspace(3) %address.local123, i64 0, !dbg !16
  %375 = load <4 x half>, ptr addrspace(3) %lds.const.gep707, align 8, !dbg !16
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %369, <4 x half> %374, <4 x float> %373) #9, !dbg !16
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %367, <4 x half> %375, <4 x float> %270) #9, !dbg !16
  %lds.const.gep708 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %378 = load <4 x half>, ptr addrspace(3) %lds.const.gep708, align 8, !dbg !16
  %lds.const.gep709 = getelementptr i8, ptr addrspace(3) %address.local110, i64 0, !dbg !16
  %379 = load <4 x half>, ptr addrspace(3) %lds.const.gep709, align 8, !dbg !16
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %369, <4 x half> %378, <4 x float> %377) #9, !dbg !16
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %379, <4 x float> %274) #9, !dbg !16
  %lds.const.gep710 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %382 = load <4 x half>, ptr addrspace(3) %lds.const.gep710, align 8, !dbg !16
  %lds.const.gep711 = getelementptr i8, ptr addrspace(3) %address.local123, i64 0, !dbg !16
  %383 = load <4 x half>, ptr addrspace(3) %lds.const.gep711, align 8, !dbg !16
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %371, <4 x half> %382, <4 x float> %381) #9, !dbg !16
  %lds.const.gep712 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %385 = load <4 x half>, ptr addrspace(3) %lds.const.gep712, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %383, <4 x float> %277) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %371, <4 x half> %385, <4 x float> %386) #9, !dbg !16
  %address.local289 = shl nuw nsw i64 %address.local254, 3, !dbg !11
  %address.local290 = and i64 %address.local289, 1536, !dbg !11
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local290, !dbg !11
  %address.local292 = ptrtoint ptr addrspace(3) %address.local291 to i64, !dbg !11
  %address.local293 = trunc i64 %address.local292 to i32, !dbg !11
  %388 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local293), !dbg !11
  %389 = zext i32 %388 to i64, !dbg !11
  %390 = inttoptr i64 %389 to ptr addrspace(3), !dbg !11
  %address.local294 = shl nuw nsw i64 %address.local254, 7, !dbg !11
  %address.local295 = and i64 %address.local294, 16256, !dbg !11
  %address.local296 = lshr i64 %address.local254, 4, !dbg !11
  %address.local297 = and i64 %address.local296, 8, !dbg !11
  %address.local298 = or disjoint i64 %address.local295, %address.local297, !dbg !11
  %address.local299 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local298, !dbg !11
  %391 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %392 = ptrtoint ptr addrspace(1) %address.local299 to i64, !dbg !11
  %393 = sub i64 %392, %391, !dbg !11
  %dma.buffer.offset749 = trunc i64 %393 to i32, !dbg !11
  %394 = trunc i64 %391 to i32, !dbg !11
  %395 = insertelement <4 x i32> poison, i32 %394, i64 0, !dbg !11
  %396 = lshr i64 %391, 32, !dbg !11
  %397 = trunc i64 %396 to i32, !dbg !11
  %398 = insertelement <4 x i32> %395, i32 %397, i64 1, !dbg !11
  %399 = insertelement <4 x i32> %398, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc750 = insertelement <4 x i32> %399, i32 131072, i64 3, !dbg !11
  %400 = ptrtoint ptr addrspace(3) %390 to i32, !dbg !11
  %dma.buffer.lds751 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %400), !dbg !11
  %401 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds751, i32 %dma.buffer.offset749, <4 x i32> %dma.buffer.rsrc750) #10, !dbg !11
  %address.local300 = or disjoint i64 %address.local254, 256, !dbg !11
  %address.local301 = shl nuw nsw i64 %address.local300, 3, !dbg !11
  %address.local302 = and i64 %address.local301, 3584, !dbg !11
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !11
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !11
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !11
  %402 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !11
  %403 = zext i32 %402 to i64, !dbg !11
  %404 = inttoptr i64 %403 to ptr addrspace(3), !dbg !11
  %address.local306 = shl nuw nsw i64 %address.local254, 7, !dbg !11
  %address.local307 = and i64 %address.local306, 16256, !dbg !11
  %address.local308 = lshr i64 %address.local300, 4, !dbg !11
  %address.local309 = and i64 %address.local308, 24, !dbg !11
  %address.local310 = or disjoint i64 %address.local307, %address.local309, !dbg !11
  %address.local311 = getelementptr [2 x i8], ptr addrspace(1) %56, i64 %address.local310, !dbg !11
  %405 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %406 = ptrtoint ptr addrspace(1) %address.local311 to i64, !dbg !11
  %407 = sub i64 %406, %405, !dbg !11
  %dma.buffer.offset746 = trunc i64 %407 to i32, !dbg !11
  %408 = trunc i64 %405 to i32, !dbg !11
  %409 = insertelement <4 x i32> poison, i32 %408, i64 0, !dbg !11
  %410 = lshr i64 %405, 32, !dbg !11
  %411 = trunc i64 %410 to i32, !dbg !11
  %412 = insertelement <4 x i32> %409, i32 %411, i64 1, !dbg !11
  %413 = insertelement <4 x i32> %412, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc747 = insertelement <4 x i32> %413, i32 131072, i64 3, !dbg !11
  %414 = ptrtoint ptr addrspace(3) %404 to i32, !dbg !11
  %dma.buffer.lds748 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %414), !dbg !11
  %415 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds748, i32 %dma.buffer.offset746, <4 x i32> %dma.buffer.rsrc747) #10, !dbg !11
  %416 = add nuw i64 %58, 12288, !dbg !17
  %417 = icmp slt i64 %416, 0, !dbg !17
  %418 = sub nuw i64 -12289, %58, !dbg !17
  %419 = select i1 %417, i64 %418, i64 %416, !dbg !17
  %420 = sdiv i64 %419, 16384, !dbg !17
  %.lobit3 = ashr i64 %416, 63, !dbg !17
  %421 = xor i64 %420, %.lobit3, !dbg !17
  %422 = add nsw i64 %421, %15, !dbg !17
  %423 = add nsw i64 %202, 96, !dbg !17
  %424 = icmp slt i64 %202, -96, !dbg !17
  %425 = sub nsw i64 -97, %202, !dbg !17
  %426 = select i1 %424, i64 %425, i64 %423, !dbg !17
  %427 = sdiv i64 %426, 128, !dbg !17
  %428 = sext i1 %424 to i64, !dbg !17
  %429 = xor i64 %427, %428, !dbg !17
  %430 = shl nsw i64 %429, 7, !dbg !17
  %431 = sub i64 %52, %430, !dbg !17
  %432 = shl i64 %422, 19, !dbg !17
  %433 = and i64 %432, 16252928, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %434 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %433, !dbg !17
  %435 = getelementptr [2 x i8], ptr addrspace(1) %434, i64 %98, !dbg !17
  %436 = getelementptr [2 x i8], ptr addrspace(1) %435, i64 %431, !dbg !17
  %437 = getelementptr i8, ptr addrspace(1) %436, i64 192, !dbg !17
  %438 = getelementptr [2 x i8], ptr addrspace(1) %437, i64 %104, !dbg !17
  br label %439, !dbg !17

439:                                              ; preds = %442, %363
  %440 = phi i64 [ %472, %442 ], [ %35, %363 ], !dbg !17
  %441 = icmp samesign ult i64 %440, 128, !dbg !17
  br i1 %441, label %442, label %473, !dbg !17

442:                                              ; preds = %439
  %443 = and i64 %440, 27, !dbg !17
  %444 = lshr i64 %440, 2, !dbg !17
  %445 = and i64 %444, 4, !dbg !17
  %446 = add nuw nsw i64 %440, %445, !dbg !17
  %447 = and i64 %446, 4, !dbg !17
  %448 = or disjoint i64 %443, %447, !dbg !17
  %449 = lshr i64 %440, 2, !dbg !17
  %450 = and i64 %449, 24, !dbg !17
  %451 = shl nuw nsw i64 %440, 3, !dbg !17
  %452 = and i64 %451, 512, !dbg !17
  %453 = getelementptr [2 x i8], ptr addrspace(1) %438, i64 %450, !dbg !17
  %.idx = shl nuw nsw i64 %448, 8, !dbg !17
  %454 = getelementptr i8, ptr addrspace(1) %453, i64 %.idx, !dbg !17
  %455 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %452, !dbg !17
  %456 = ptrtoint ptr addrspace(3) %455 to i64, !dbg !17
  %457 = trunc i64 %456 to i32, !dbg !17
  %458 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %457), !dbg !17
  %459 = zext i32 %458 to i64, !dbg !17
  %460 = inttoptr i64 %459 to ptr addrspace(3), !dbg !17
  %461 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %462 = ptrtoint ptr addrspace(1) %454 to i64, !dbg !17
  %463 = sub i64 %462, %461, !dbg !17
  %dma.buffer.offset731 = trunc i64 %463 to i32, !dbg !17
  %464 = trunc i64 %461 to i32, !dbg !17
  %465 = insertelement <4 x i32> poison, i32 %464, i64 0, !dbg !17
  %466 = lshr i64 %461, 32, !dbg !17
  %467 = trunc i64 %466 to i32, !dbg !17
  %468 = insertelement <4 x i32> %465, i32 %467, i64 1, !dbg !17
  %469 = insertelement <4 x i32> %468, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc732 = insertelement <4 x i32> %469, i32 131072, i64 3, !dbg !17
  %470 = ptrtoint ptr addrspace(3) %460 to i32, !dbg !17
  %dma.buffer.lds733 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %470), !dbg !17
  %471 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds733, i32 %dma.buffer.offset731, <4 x i32> %dma.buffer.rsrc732) #10, !dbg !17
  %472 = or disjoint i64 %35, 256, !dbg !17
  br label %439, !dbg !17

473:                                              ; preds = %439
  %lds.const.gep713 = getelementptr i8, ptr addrspace(3) %address.local91, i64 0, !dbg !12
  %474 = load <4 x half>, ptr addrspace(3) %lds.const.gep713, align 8, !dbg !12
  %lds.const.gep714 = getelementptr i8, ptr addrspace(3) %address.local91, i64 4096, !dbg !12
  %475 = load <4 x half>, ptr addrspace(3) %lds.const.gep714, align 8, !dbg !12
  %lds.const.gep715 = getelementptr i8, ptr addrspace(3) %address.local91, i64 1024, !dbg !12
  %476 = load <4 x half>, ptr addrspace(3) %lds.const.gep715, align 8, !dbg !12
  %477 = fmul <4 x half> %474, splat (half 0xH3015), !dbg !12
  %lds.const.gep716 = getelementptr i8, ptr addrspace(3) %address.local91, i64 5120, !dbg !12
  %478 = load <4 x half>, ptr addrspace(3) %lds.const.gep716, align 8, !dbg !12
  %479 = fmul <4 x half> %475, splat (half 0xH3015), !dbg !12
  %480 = fmul <4 x half> %476, splat (half 0xH3015), !dbg !12
  %481 = fmul <4 x half> %478, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %482 = add nuw nsw i64 %98, %15, !dbg !18
  %483 = shl i64 %482, 19, !dbg !18
  %484 = and i64 %483, 16252928, !dbg !18
  %485 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %484, !dbg !18
  %486 = getelementptr [2 x i8], ptr addrspace(1) %485, i64 %104, !dbg !18
  br label %487, !dbg !18

487:                                              ; preds = %490, %473
  %488 = phi i64 [ %516, %490 ], [ %35, %473 ], !dbg !18
  %489 = icmp samesign ult i64 %488, 128, !dbg !18
  br i1 %489, label %490, label %517, !dbg !18

490:                                              ; preds = %487
  %491 = shl nuw nsw i64 %488, 3, !dbg !18
  %492 = and i64 %491, 24, !dbg !18
  %493 = shl nuw nsw i64 %488, 3, !dbg !18
  %494 = and i64 %493, 512, !dbg !18
  %495 = shl nuw nsw i64 %488, 5, !dbg !18
  %496 = and i64 %495, 3968, !dbg !18
  %497 = getelementptr [2 x i8], ptr addrspace(1) %486, i64 %496, !dbg !18
  %498 = getelementptr [2 x i8], ptr addrspace(1) %497, i64 %492, !dbg !18
  %499 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %494, !dbg !18
  %500 = ptrtoint ptr addrspace(3) %499 to i64, !dbg !18
  %501 = trunc i64 %500 to i32, !dbg !18
  %502 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %501), !dbg !18
  %503 = zext i32 %502 to i64, !dbg !18
  %504 = inttoptr i64 %503 to ptr addrspace(3), !dbg !18
  %505 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %506 = ptrtoint ptr addrspace(1) %498 to i64, !dbg !18
  %507 = sub i64 %506, %505, !dbg !18
  %dma.buffer.offset734 = trunc i64 %507 to i32, !dbg !18
  %508 = trunc i64 %505 to i32, !dbg !18
  %509 = insertelement <4 x i32> poison, i32 %508, i64 0, !dbg !18
  %510 = lshr i64 %505, 32, !dbg !18
  %511 = trunc i64 %510 to i32, !dbg !18
  %512 = insertelement <4 x i32> %509, i32 %511, i64 1, !dbg !18
  %513 = insertelement <4 x i32> %512, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc735 = insertelement <4 x i32> %513, i32 131072, i64 3, !dbg !18
  %514 = ptrtoint ptr addrspace(3) %504 to i32, !dbg !18
  %dma.buffer.lds736 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %514), !dbg !18
  %515 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds736, i32 %dma.buffer.offset734, <4 x i32> %dma.buffer.rsrc735) #10, !dbg !18
  %516 = or disjoint i64 %35, 256, !dbg !18
  br label %487, !dbg !18

517:                                              ; preds = %487
  %lds.const.gep717 = getelementptr i8, ptr addrspace(3) %address.local110, i64 0, !dbg !16
  %518 = load <4 x half>, ptr addrspace(3) %lds.const.gep717, align 8, !dbg !16
  %519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %477, <4 x half> %518, <4 x float> %376) #9, !dbg !16
  %lds.const.gep718 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %520 = load <4 x half>, ptr addrspace(3) %lds.const.gep718, align 8, !dbg !16
  %lds.const.gep719 = getelementptr i8, ptr addrspace(3) %address.local123, i64 0, !dbg !16
  %521 = load <4 x half>, ptr addrspace(3) %lds.const.gep719, align 8, !dbg !16
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %479, <4 x half> %520, <4 x float> %519) #9, !dbg !16
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %477, <4 x half> %521, <4 x float> %380) #9, !dbg !16
  %524 = extractelement <4 x float> %522, i64 0, !dbg !22
  %525 = extractelement <4 x float> %522, i64 1, !dbg !22
  %526 = select i1 %.not, float %525, float %524, !dbg !13
  %527 = extractelement <4 x float> %522, i64 2, !dbg !22
  %528 = extractelement <4 x float> %522, i64 3, !dbg !22
  %lds.const.gep720 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %529 = load <4 x half>, ptr addrspace(3) %lds.const.gep720, align 8, !dbg !16
  %lds.const.gep721 = getelementptr i8, ptr addrspace(3) %address.local110, i64 0, !dbg !16
  %530 = load <4 x half>, ptr addrspace(3) %lds.const.gep721, align 8, !dbg !16
  %531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %479, <4 x half> %529, <4 x float> %523) #9, !dbg !16
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %480, <4 x half> %530, <4 x float> %384) #9, !dbg !16
  %533 = extractelement <4 x float> %531, i64 0, !dbg !22
  %534 = extractelement <4 x float> %531, i64 1, !dbg !22
  %535 = extractelement <4 x float> %531, i64 2, !dbg !22
  %536 = extractelement <4 x float> %531, i64 3, !dbg !22
  %lds.const.gep722 = getelementptr i8, ptr addrspace(3) %address.local110, i64 1024, !dbg !16
  %537 = load <4 x half>, ptr addrspace(3) %lds.const.gep722, align 8, !dbg !16
  %lds.const.gep723 = getelementptr i8, ptr addrspace(3) %address.local123, i64 0, !dbg !16
  %538 = load <4 x half>, ptr addrspace(3) %lds.const.gep723, align 8, !dbg !16
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %481, <4 x half> %537, <4 x float> %532) #9, !dbg !16
  %lds.const.gep724 = getelementptr i8, ptr addrspace(3) %address.local123, i64 1024, !dbg !16
  %540 = load <4 x half>, ptr addrspace(3) %lds.const.gep724, align 8, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %480, <4 x half> %538, <4 x float> %387) #9, !dbg !16
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !22
  %542 = extractelement <4 x float> %539, i64 0, !dbg !22
  %543 = extractelement <4 x float> %539, i64 1, !dbg !22
  %544 = extractelement <4 x float> %539, i64 2, !dbg !22
  %545 = extractelement <4 x float> %539, i64 3, !dbg !22
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %481, <4 x half> %540, <4 x float> %541) #9, !dbg !16
  %547 = extractelement <4 x float> %546, i64 0, !dbg !22
  %548 = extractelement <4 x float> %546, i64 1, !dbg !22
  %549 = extractelement <4 x float> %546, i64 2, !dbg !22
  %550 = extractelement <4 x float> %546, i64 3, !dbg !22
  %551 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %552 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %551), !dbg !13
  %553 = add nuw nsw i32 %552, 64, !dbg !13
  %554 = xor i32 %552, 16, !dbg !13
  %555 = and i32 %553, 192, !dbg !13
  %556 = icmp samesign ult i32 %554, %555, !dbg !13
  %557 = select i1 %556, i32 %554, i32 %552, !dbg !13
  %558 = shl nuw nsw i32 %557, 2, !dbg !13
  %559 = bitcast float %526 to i32, !dbg !13
  %560 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %559), !dbg !13
  %561 = bitcast i32 %560 to float, !dbg !13
  %562 = select i1 %.not, float %524, float %561, !dbg !13
  %563 = select i1 %.not, float %561, float %525, !dbg !13
  %564 = select i1 %.not, float %528, float %527, !dbg !13
  %565 = bitcast float %564 to i32, !dbg !13
  %566 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %565), !dbg !13
  %567 = bitcast i32 %566 to float, !dbg !13
  %568 = select i1 %.not, float %527, float %567, !dbg !13
  %569 = select i1 %.not, float %567, float %528, !dbg !13
  %570 = select i1 %.not1, float %568, float %562, !dbg !13
  %571 = xor i32 %552, 32, !dbg !13
  %572 = icmp samesign ult i32 %571, %555, !dbg !13
  %573 = select i1 %572, i32 %571, i32 %552, !dbg !13
  %574 = shl nuw nsw i32 %573, 2, !dbg !13
  %575 = bitcast float %570 to i32, !dbg !13
  %576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %575), !dbg !13
  %577 = bitcast i32 %576 to float, !dbg !13
  %578 = select i1 %.not1, float %562, float %577, !dbg !13
  %579 = select i1 %.not1, float %577, float %568, !dbg !13
  %580 = select i1 %.not1, float %569, float %563, !dbg !13
  %581 = bitcast float %580 to i32, !dbg !13
  %582 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %581), !dbg !13
  %583 = bitcast i32 %582 to float, !dbg !13
  %584 = select i1 %.not1, float %563, float %583, !dbg !13
  %585 = select i1 %.not1, float %583, float %569, !dbg !13
  %586 = select i1 %.not, float %534, float %533, !dbg !13
  %587 = bitcast float %586 to i32, !dbg !13
  %588 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %587), !dbg !13
  %589 = bitcast i32 %588 to float, !dbg !13
  %590 = select i1 %.not, float %533, float %589, !dbg !13
  %591 = select i1 %.not, float %589, float %534, !dbg !13
  %592 = select i1 %.not, float %536, float %535, !dbg !13
  %593 = bitcast float %592 to i32, !dbg !13
  %594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %593), !dbg !13
  %595 = bitcast i32 %594 to float, !dbg !13
  %596 = select i1 %.not, float %535, float %595, !dbg !13
  %597 = select i1 %.not, float %595, float %536, !dbg !13
  %598 = select i1 %.not1, float %596, float %590, !dbg !13
  %599 = bitcast float %598 to i32, !dbg !13
  %600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %599), !dbg !13
  %601 = bitcast i32 %600 to float, !dbg !13
  %602 = select i1 %.not1, float %590, float %601, !dbg !13
  %603 = select i1 %.not1, float %601, float %596, !dbg !13
  %604 = select i1 %.not1, float %597, float %591, !dbg !13
  %605 = bitcast float %604 to i32, !dbg !13
  %606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %605), !dbg !13
  %607 = bitcast i32 %606 to float, !dbg !13
  %608 = select i1 %.not1, float %591, float %607, !dbg !13
  %609 = select i1 %.not1, float %607, float %597, !dbg !13
  %610 = select i1 %.not, float %543, float %542, !dbg !13
  %611 = bitcast float %610 to i32, !dbg !13
  %612 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %611), !dbg !13
  %613 = bitcast i32 %612 to float, !dbg !13
  %614 = select i1 %.not, float %542, float %613, !dbg !13
  %615 = select i1 %.not, float %613, float %543, !dbg !13
  %616 = select i1 %.not, float %545, float %544, !dbg !13
  %617 = bitcast float %616 to i32, !dbg !13
  %618 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %617), !dbg !13
  %619 = bitcast i32 %618 to float, !dbg !13
  %620 = select i1 %.not, float %544, float %619, !dbg !13
  %621 = select i1 %.not, float %619, float %545, !dbg !13
  %622 = select i1 %.not1, float %620, float %614, !dbg !13
  %623 = bitcast float %622 to i32, !dbg !13
  %624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %623), !dbg !13
  %625 = bitcast i32 %624 to float, !dbg !13
  %626 = select i1 %.not1, float %614, float %625, !dbg !13
  %627 = select i1 %.not1, float %625, float %620, !dbg !13
  %628 = select i1 %.not1, float %621, float %615, !dbg !13
  %629 = bitcast float %628 to i32, !dbg !13
  %630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %629), !dbg !13
  %631 = bitcast i32 %630 to float, !dbg !13
  %632 = select i1 %.not1, float %615, float %631, !dbg !13
  %633 = select i1 %.not1, float %631, float %621, !dbg !13
  %634 = select i1 %.not, float %548, float %547, !dbg !13
  %635 = bitcast float %634 to i32, !dbg !13
  %636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %635), !dbg !13
  %637 = bitcast i32 %636 to float, !dbg !13
  %638 = select i1 %.not, float %547, float %637, !dbg !13
  %639 = select i1 %.not, float %637, float %548, !dbg !13
  %640 = select i1 %.not, float %550, float %549, !dbg !13
  %641 = bitcast float %640 to i32, !dbg !13
  %642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %641), !dbg !13
  %643 = bitcast i32 %642 to float, !dbg !13
  %644 = select i1 %.not, float %549, float %643, !dbg !13
  %645 = select i1 %.not, float %643, float %550, !dbg !13
  %646 = select i1 %.not1, float %644, float %638, !dbg !13
  %647 = bitcast float %646 to i32, !dbg !13
  %648 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %647), !dbg !13
  %649 = bitcast i32 %648 to float, !dbg !13
  %650 = select i1 %.not1, float %638, float %649, !dbg !13
  %651 = select i1 %.not1, float %649, float %644, !dbg !13
  %652 = select i1 %.not1, float %645, float %639, !dbg !13
  %653 = bitcast float %652 to i32, !dbg !13
  %654 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %653), !dbg !13
  %655 = bitcast i32 %654 to float, !dbg !13
  %656 = select i1 %.not1, float %639, float %655, !dbg !13
  %657 = select i1 %.not1, float %655, float %645, !dbg !13
  %658 = or disjoint i64 %67, 31, !dbg !14
  %659 = icmp uge i64 %19, %658, !dbg !14
  %660 = and i1 %43, %659, !dbg !14
  br i1 %660, label %661, label %662, !dbg !14

661:                                              ; preds = %517
  br label %709, !dbg !14

662:                                              ; preds = %517
  %663 = add i64 %19, %35, !dbg !14
  %664 = shl nuw nsw i64 %36, 4, !dbg !14
  %665 = add i64 %663, %664, !dbg !14
  %666 = and i32 %34, 240, !dbg !14
  %index.compact = zext nneg i32 %666 to i64, !dbg !14
  %667 = sub i64 %665, %index.compact, !dbg !14
  %668 = or disjoint i64 %67, %38, !dbg !14
  %669 = add i64 %667, 1, !dbg !23
  %670 = lshr i64 %669, 32, !dbg !24
  %compare.high = trunc i64 %670 to i32, !dbg !24
  %671 = lshr i64 %668, 32, !dbg !24
  %compare.high619 = trunc i64 %671 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high619, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high619, !dbg !24
  %compare.low.base = trunc i64 %669 to i32, !dbg !24
  %compare.low.base620 = trunc i64 %668 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base620, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %672 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low621 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result622 = select i1 %compare.high.equal, i1 %compare.low621, i1 %compare.high.order, !dbg !24
  %673 = select i1 %compare.result622, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base623 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local624 = or i32 %compare.local.base623, 2, !dbg !24
  %compare.low625 = icmp ugt i32 %compare.low.base, %compare.local624, !dbg !24
  %compare.result626 = select i1 %compare.high.equal, i1 %compare.low625, i1 %compare.high.order, !dbg !24
  %674 = select i1 %compare.result626, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base627 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local628 = or i32 %compare.local.base627, 3, !dbg !24
  %compare.low629 = icmp ugt i32 %compare.low.base, %compare.local628, !dbg !24
  %compare.result630 = select i1 %compare.high.equal, i1 %compare.low629, i1 %compare.high.order, !dbg !24
  %675 = select i1 %compare.result630, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base631 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local632 = or i32 %compare.local.base631, 16, !dbg !24
  %compare.low633 = icmp ugt i32 %compare.low.base, %compare.local632, !dbg !24
  %compare.result634 = select i1 %compare.high.equal, i1 %compare.low633, i1 %compare.high.order, !dbg !24
  %676 = select i1 %compare.result634, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base635 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local636 = or i32 %compare.local.base635, 17, !dbg !24
  %compare.low637 = icmp ugt i32 %compare.low.base, %compare.local636, !dbg !24
  %compare.result638 = select i1 %compare.high.equal, i1 %compare.low637, i1 %compare.high.order, !dbg !24
  %677 = select i1 %compare.result638, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base639 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local640 = or i32 %compare.local.base639, 18, !dbg !24
  %compare.low641 = icmp ugt i32 %compare.low.base, %compare.local640, !dbg !24
  %compare.result642 = select i1 %compare.high.equal, i1 %compare.low641, i1 %compare.high.order, !dbg !24
  %678 = select i1 %compare.result642, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base643 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base620, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local644 = or i32 %compare.local.base643, 19, !dbg !24
  %compare.low645 = icmp ugt i32 %compare.low.base, %compare.local644, !dbg !24
  %compare.result646 = select i1 %compare.high.equal, i1 %compare.low645, i1 %compare.high.order, !dbg !24
  %679 = select i1 %compare.result646, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %680 = add i64 %667, 65, !dbg !23
  %681 = lshr i64 %680, 32, !dbg !24
  %compare.high647 = trunc i64 %681 to i32, !dbg !24
  %682 = lshr i64 %668, 32, !dbg !24
  %compare.high648 = trunc i64 %682 to i32, !dbg !24
  %compare.high.equal649 = icmp eq i32 %compare.high647, %compare.high648, !dbg !24
  %compare.high.order650 = icmp ugt i32 %compare.high647, %compare.high648, !dbg !24
  %compare.low.base651 = trunc i64 %680 to i32, !dbg !24
  %compare.low.base652 = trunc i64 %668 to i32, !dbg !24
  %compare.low653 = icmp ugt i32 %compare.low.base651, %compare.low.base652, !dbg !24
  %compare.result654 = select i1 %compare.high.equal649, i1 %compare.low653, i1 %compare.high.order650, !dbg !24
  %683 = select i1 %compare.result654, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base655 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local656 = or i32 %compare.local.base655, 1, !dbg !24
  %compare.low657 = icmp ugt i32 %compare.low.base651, %compare.local656, !dbg !24
  %compare.result658 = select i1 %compare.high.equal649, i1 %compare.low657, i1 %compare.high.order650, !dbg !24
  %684 = select i1 %compare.result658, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base659 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local660 = or i32 %compare.local.base659, 2, !dbg !24
  %compare.low661 = icmp ugt i32 %compare.low.base651, %compare.local660, !dbg !24
  %compare.result662 = select i1 %compare.high.equal649, i1 %compare.low661, i1 %compare.high.order650, !dbg !24
  %685 = select i1 %compare.result662, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base663 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local664 = or i32 %compare.local.base663, 3, !dbg !24
  %compare.low665 = icmp ugt i32 %compare.low.base651, %compare.local664, !dbg !24
  %compare.result666 = select i1 %compare.high.equal649, i1 %compare.low665, i1 %compare.high.order650, !dbg !24
  %686 = select i1 %compare.result666, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base667 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local668 = or i32 %compare.local.base667, 16, !dbg !24
  %compare.low669 = icmp ugt i32 %compare.low.base651, %compare.local668, !dbg !24
  %compare.result670 = select i1 %compare.high.equal649, i1 %compare.low669, i1 %compare.high.order650, !dbg !24
  %687 = select i1 %compare.result670, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base671 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local672 = or i32 %compare.local.base671, 17, !dbg !24
  %compare.low673 = icmp ugt i32 %compare.low.base651, %compare.local672, !dbg !24
  %compare.result674 = select i1 %compare.high.equal649, i1 %compare.low673, i1 %compare.high.order650, !dbg !24
  %688 = select i1 %compare.result674, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base675 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local676 = or i32 %compare.local.base675, 18, !dbg !24
  %compare.low677 = icmp ugt i32 %compare.low.base651, %compare.local676, !dbg !24
  %compare.result678 = select i1 %compare.high.equal649, i1 %compare.low677, i1 %compare.high.order650, !dbg !24
  %689 = select i1 %compare.result678, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base679 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base652, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local680 = or i32 %compare.local.base679, 19, !dbg !24
  %compare.low681 = icmp ugt i32 %compare.low.base651, %compare.local680, !dbg !24
  %compare.result682 = select i1 %compare.high.equal649, i1 %compare.low681, i1 %compare.high.order650, !dbg !24
  %690 = select i1 %compare.result682, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %691 = insertelement <8 x float> poison, float %672, i64 0, !dbg !14
  %692 = insertelement <8 x float> %691, float %673, i64 1, !dbg !14
  %693 = insertelement <8 x float> %692, float %674, i64 2, !dbg !14
  %694 = insertelement <8 x float> %693, float %675, i64 3, !dbg !14
  %695 = insertelement <8 x float> %694, float %676, i64 4, !dbg !14
  %696 = insertelement <8 x float> %695, float %677, i64 5, !dbg !14
  %697 = insertelement <8 x float> %696, float %678, i64 6, !dbg !14
  %698 = insertelement <8 x float> %697, float %679, i64 7, !dbg !14
  %699 = insertvalue [2 x <8 x float>] poison, <8 x float> %698, 0, !dbg !14
  %700 = insertelement <8 x float> poison, float %683, i64 0, !dbg !14
  %701 = insertelement <8 x float> %700, float %684, i64 1, !dbg !14
  %702 = insertelement <8 x float> %701, float %685, i64 2, !dbg !14
  %703 = insertelement <8 x float> %702, float %686, i64 3, !dbg !14
  %704 = insertelement <8 x float> %703, float %687, i64 4, !dbg !14
  %705 = insertelement <8 x float> %704, float %688, i64 5, !dbg !14
  %706 = insertelement <8 x float> %705, float %689, i64 6, !dbg !14
  %707 = insertelement <8 x float> %706, float %690, i64 7, !dbg !14
  %708 = insertvalue [2 x <8 x float>] %699, <8 x float> %707, 1, !dbg !14
  br label %709, !dbg !14

709:                                              ; preds = %661, %662
  %710 = phi [2 x <8 x float>] [ %708, %662 ], [ zeroinitializer, %661 ], !dbg !14
  br label %711, !dbg !14

711:                                              ; preds = %709
  %712 = extractvalue [2 x <8 x float>] %710, 0, !dbg !26
  %713 = insertelement <4 x float> poison, float %578, i64 0, !dbg !26
  %714 = insertelement <4 x float> %713, float %584, i64 1, !dbg !26
  %715 = insertelement <4 x float> %714, float %579, i64 2, !dbg !26
  %716 = insertelement <4 x float> %715, float %585, i64 3, !dbg !26
  %717 = shufflevector <8 x float> %712, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %718 = fadd <4 x float> %716, %717, !dbg !26
  %719 = extractelement <4 x float> %718, i64 0, !dbg !26
  %720 = extractelement <4 x float> %718, i64 1, !dbg !26
  %721 = extractelement <4 x float> %718, i64 2, !dbg !26
  %722 = extractelement <4 x float> %718, i64 3, !dbg !26
  %723 = insertelement <4 x float> poison, float %602, i64 0, !dbg !26
  %724 = insertelement <4 x float> %723, float %608, i64 1, !dbg !26
  %725 = insertelement <4 x float> %724, float %603, i64 2, !dbg !26
  %726 = insertelement <4 x float> %725, float %609, i64 3, !dbg !26
  %727 = shufflevector <8 x float> %712, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %728 = fadd <4 x float> %726, %727, !dbg !26
  %729 = extractelement <4 x float> %728, i64 0, !dbg !26
  %730 = extractelement <4 x float> %728, i64 1, !dbg !26
  %731 = extractelement <4 x float> %728, i64 2, !dbg !26
  %732 = extractelement <4 x float> %728, i64 3, !dbg !26
  %733 = extractvalue [2 x <8 x float>] %710, 1, !dbg !26
  %734 = insertelement <4 x float> poison, float %626, i64 0, !dbg !26
  %735 = insertelement <4 x float> %734, float %632, i64 1, !dbg !26
  %736 = insertelement <4 x float> %735, float %627, i64 2, !dbg !26
  %737 = insertelement <4 x float> %736, float %633, i64 3, !dbg !26
  %738 = shufflevector <8 x float> %733, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %739 = fadd <4 x float> %737, %738, !dbg !26
  %740 = extractelement <4 x float> %739, i64 0, !dbg !26
  %741 = extractelement <4 x float> %739, i64 1, !dbg !26
  %742 = extractelement <4 x float> %739, i64 2, !dbg !26
  %743 = extractelement <4 x float> %739, i64 3, !dbg !26
  %744 = insertelement <4 x float> poison, float %650, i64 0, !dbg !26
  %745 = insertelement <4 x float> %744, float %656, i64 1, !dbg !26
  %746 = insertelement <4 x float> %745, float %651, i64 2, !dbg !26
  %747 = insertelement <4 x float> %746, float %657, i64 3, !dbg !26
  %748 = shufflevector <8 x float> %733, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %749 = fadd <4 x float> %747, %748, !dbg !26
  %750 = extractelement <4 x float> %749, i64 0, !dbg !26
  %751 = extractelement <4 x float> %749, i64 1, !dbg !26
  %752 = extractelement <4 x float> %749, i64 2, !dbg !26
  %753 = extractelement <4 x float> %749, i64 3, !dbg !26
  %754 = call float @llvm.maxnum.f32(float %719, float 0xFFF0000000000000), !dbg !27
  %755 = call float @llvm.maxnum.f32(float %754, float %720), !dbg !27
  %756 = call float @llvm.maxnum.f32(float %755, float %721), !dbg !27
  %757 = call float @llvm.maxnum.f32(float %756, float %722), !dbg !27
  %758 = call float @llvm.maxnum.f32(float %757, float %729), !dbg !27
  %759 = call float @llvm.maxnum.f32(float %758, float %730), !dbg !27
  %760 = call float @llvm.maxnum.f32(float %759, float %731), !dbg !27
  %761 = call float @llvm.maxnum.f32(float %760, float %732), !dbg !27
  %762 = bitcast float %761 to i32, !dbg !27
  %763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %762), !dbg !27
  %764 = bitcast i32 %763 to float, !dbg !27
  %765 = call float @llvm.maxnum.f32(float %761, float %764), !dbg !27
  %766 = bitcast float %765 to i32, !dbg !27
  %767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %766), !dbg !27
  %768 = bitcast i32 %767 to float, !dbg !27
  %769 = call float @llvm.maxnum.f32(float %765, float %768), !dbg !27
  %770 = extractvalue [2 x <1 x float>] %61, 0, !dbg !27
  %771 = insertelement <1 x float> poison, float %769, i64 0, !dbg !27
  %772 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %771, <1 x float> %770), !dbg !27
  %773 = call float @llvm.maxnum.f32(float %740, float 0xFFF0000000000000), !dbg !27
  %774 = call float @llvm.maxnum.f32(float %773, float %741), !dbg !27
  %775 = call float @llvm.maxnum.f32(float %774, float %742), !dbg !27
  %776 = call float @llvm.maxnum.f32(float %775, float %743), !dbg !27
  %777 = call float @llvm.maxnum.f32(float %776, float %750), !dbg !27
  %778 = call float @llvm.maxnum.f32(float %777, float %751), !dbg !27
  %779 = call float @llvm.maxnum.f32(float %778, float %752), !dbg !27
  %780 = call float @llvm.maxnum.f32(float %779, float %753), !dbg !27
  %781 = bitcast float %780 to i32, !dbg !27
  %782 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %781), !dbg !27
  %783 = bitcast i32 %782 to float, !dbg !27
  %784 = call float @llvm.maxnum.f32(float %780, float %783), !dbg !27
  %785 = bitcast float %784 to i32, !dbg !27
  %786 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %785), !dbg !27
  %787 = bitcast i32 %786 to float, !dbg !27
  %788 = call float @llvm.maxnum.f32(float %784, float %787), !dbg !27
  %789 = extractvalue [2 x <1 x float>] %61, 1, !dbg !27
  %790 = insertelement <1 x float> poison, float %788, i64 0, !dbg !27
  %791 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %790, <1 x float> %789), !dbg !27
  %792 = shufflevector <1 x float> %772, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %793 = fsub <4 x float> %718, %792, !dbg !28
  %794 = fsub <4 x float> %728, %792, !dbg !28
  %795 = shufflevector <1 x float> %791, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %796 = fsub <4 x float> %739, %795, !dbg !28
  %797 = fsub <4 x float> %749, %795, !dbg !28
  %798 = fsub <1 x float> %770, %772, !dbg !29
  %799 = extractelement <1 x float> %798, i64 0, !dbg !30
  %800 = call float @llvm.amdgcn.exp2.f32(float %799), !dbg !30
  %801 = extractvalue [2 x <1 x float>] %60, 0, !dbg !31
  %802 = extractelement <1 x float> %801, i64 0, !dbg !31
  %803 = fmul float %802, %800, !dbg !31
  %804 = fsub <1 x float> %789, %791, !dbg !29
  %805 = extractelement <1 x float> %804, i64 0, !dbg !30
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !30
  %807 = extractvalue [2 x <1 x float>] %60, 1, !dbg !31
  %808 = extractelement <1 x float> %807, i64 0, !dbg !31
  %809 = fmul float %808, %806, !dbg !31
  %810 = extractelement <4 x float> %793, i64 0, !dbg !32
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !32
  %812 = insertelement <4 x float> poison, float %811, i64 0, !dbg !34
  %813 = extractelement <4 x float> %793, i64 1, !dbg !32
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !32
  %815 = insertelement <4 x float> %812, float %814, i64 1, !dbg !34
  %816 = fadd float %811, %814, !dbg !33
  %817 = extractelement <4 x float> %793, i64 2, !dbg !32
  %818 = call float @llvm.amdgcn.exp2.f32(float %817), !dbg !32
  %819 = insertelement <4 x float> %815, float %818, i64 2, !dbg !34
  %820 = fadd float %816, %818, !dbg !33
  %821 = extractelement <4 x float> %793, i64 3, !dbg !32
  %822 = call float @llvm.amdgcn.exp2.f32(float %821), !dbg !32
  %823 = insertelement <4 x float> %819, float %822, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %823) #9, !dbg !34, !frisk.register_lifetimes !1
  %824 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %824) #9, !dbg !33, !frisk.register_lifetimes !1
  %825 = fadd float %820, %822, !dbg !33
  %826 = extractelement <4 x float> %794, i64 0, !dbg !32
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !32
  %828 = insertelement <4 x float> poison, float %827, i64 0, !dbg !34
  %829 = fadd float %825, %827, !dbg !33
  %830 = extractelement <4 x float> %794, i64 1, !dbg !32
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !32
  %832 = insertelement <4 x float> %828, float %831, i64 1, !dbg !34
  %833 = fadd float %829, %831, !dbg !33
  %834 = extractelement <4 x float> %794, i64 2, !dbg !32
  %835 = call float @llvm.amdgcn.exp2.f32(float %834), !dbg !32
  %836 = insertelement <4 x float> %832, float %835, i64 2, !dbg !34
  %837 = fadd float %833, %835, !dbg !33
  %838 = extractelement <4 x float> %794, i64 3, !dbg !32
  %839 = call float @llvm.amdgcn.exp2.f32(float %838), !dbg !32
  %840 = insertelement <4 x float> %836, float %839, i64 3, !dbg !34
  %math.result49 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %840) #9, !dbg !34, !frisk.register_lifetimes !1
  %841 = fptrunc <4 x float> %math.result49 to <4 x half>, !dbg !34
  %math.packed50 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %841) #9, !dbg !33, !frisk.register_lifetimes !1
  %842 = fadd float %837, %839, !dbg !33
  %843 = extractelement <4 x float> %796, i64 0, !dbg !32
  %844 = call float @llvm.amdgcn.exp2.f32(float %843), !dbg !32
  %845 = insertelement <4 x float> poison, float %844, i64 0, !dbg !34
  %846 = extractelement <4 x float> %796, i64 1, !dbg !32
  %847 = call float @llvm.amdgcn.exp2.f32(float %846), !dbg !32
  %848 = insertelement <4 x float> %845, float %847, i64 1, !dbg !34
  %849 = fadd float %844, %847, !dbg !33
  %850 = extractelement <4 x float> %796, i64 2, !dbg !32
  %851 = call float @llvm.amdgcn.exp2.f32(float %850), !dbg !32
  %852 = insertelement <4 x float> %848, float %851, i64 2, !dbg !34
  %853 = fadd float %849, %851, !dbg !33
  %854 = extractelement <4 x float> %796, i64 3, !dbg !32
  %855 = call float @llvm.amdgcn.exp2.f32(float %854), !dbg !32
  %856 = insertelement <4 x float> %852, float %855, i64 3, !dbg !34
  %math.result51 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %856) #9, !dbg !34, !frisk.register_lifetimes !1
  %857 = fptrunc <4 x float> %math.result51 to <4 x half>, !dbg !34
  %math.packed52 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %857) #9, !dbg !33, !frisk.register_lifetimes !1
  %858 = fadd float %853, %855, !dbg !33
  %859 = extractelement <4 x float> %797, i64 0, !dbg !32
  %860 = call float @llvm.amdgcn.exp2.f32(float %859), !dbg !32
  %861 = insertelement <4 x float> poison, float %860, i64 0, !dbg !34
  %862 = fadd float %858, %860, !dbg !33
  %863 = extractelement <4 x float> %797, i64 1, !dbg !32
  %864 = call float @llvm.amdgcn.exp2.f32(float %863), !dbg !32
  %865 = insertelement <4 x float> %861, float %864, i64 1, !dbg !34
  %866 = fadd float %862, %864, !dbg !33
  %867 = extractelement <4 x float> %797, i64 2, !dbg !32
  %868 = call float @llvm.amdgcn.exp2.f32(float %867), !dbg !32
  %869 = insertelement <4 x float> %865, float %868, i64 2, !dbg !34
  %870 = fadd float %866, %868, !dbg !33
  %871 = extractelement <4 x float> %797, i64 3, !dbg !32
  %872 = call float @llvm.amdgcn.exp2.f32(float %871), !dbg !32
  %873 = insertelement <4 x float> %869, float %872, i64 3, !dbg !34
  %math.result53 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %873) #9, !dbg !34, !frisk.register_lifetimes !1
  %874 = fptrunc <4 x float> %math.result53 to <4 x half>, !dbg !34
  %math.packed54 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %874) #9, !dbg !33, !frisk.register_lifetimes !1
  %875 = fadd float %870, %872, !dbg !33
  %876 = bitcast float %842 to i32, !dbg !33
  %877 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %876), !dbg !33
  %878 = bitcast i32 %877 to float, !dbg !33
  %879 = fadd float %842, %878, !dbg !33
  %880 = bitcast float %879 to i32, !dbg !33
  %881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %880), !dbg !33
  %882 = bitcast i32 %881 to float, !dbg !33
  %883 = fadd float %879, %882, !dbg !33
  %884 = fadd float %883, %803, !dbg !33
  %885 = bitcast float %875 to i32, !dbg !33
  %886 = call i32 @llvm.amdgcn.ds.bpermute(i32 %558, i32 %885), !dbg !33
  %887 = bitcast i32 %886 to float, !dbg !33
  %888 = fadd float %875, %887, !dbg !33
  %889 = bitcast float %888 to i32, !dbg !33
  %890 = call i32 @llvm.amdgcn.ds.bpermute(i32 %574, i32 %889), !dbg !33
  %891 = bitcast i32 %890 to float, !dbg !33
  %892 = fadd float %888, %891, !dbg !33
  %893 = fadd float %892, %809, !dbg !33
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread368 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local369 = shl nuw nsw i32 %address.thread368, 3, !dbg !15
  %address.local370 = and i32 %address.local369, 504, !dbg !15
  %address.local371 = zext nneg i32 %address.local370 to i64, !dbg !15
  %address.local372 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local371, !dbg !15
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local372, i32 0) #0, !dbg !15
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !15
  %address.local373 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local371, !dbg !15
  %address.local374 = getelementptr i8, ptr addrspace(3) %address.local373, i64 1024, !dbg !15
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local374, i32 0) #0, !dbg !15
  %897 = bitcast <4 x i32> %896 to <8 x half>, !dbg !15
  %898 = add nuw nsw i64 %98, %15, !dbg !18
  %899 = shl i64 %898, 19, !dbg !18
  %900 = and i64 %899, 16252928, !dbg !18
  %901 = add i64 %900, %104, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %902 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !18
  %903 = getelementptr i8, ptr addrspace(1) %902, i64 64, !dbg !18
  br label %904, !dbg !18

904:                                              ; preds = %907, %711
  %905 = phi i64 [ %933, %907 ], [ %35, %711 ], !dbg !18
  %906 = icmp samesign ult i64 %905, 128, !dbg !18
  br i1 %906, label %907, label %934, !dbg !18

907:                                              ; preds = %904
  %908 = shl nuw nsw i64 %905, 3, !dbg !18
  %909 = and i64 %908, 24, !dbg !18
  %910 = shl nuw nsw i64 %905, 3, !dbg !18
  %911 = and i64 %910, 512, !dbg !18
  %912 = shl nuw nsw i64 %905, 5, !dbg !18
  %913 = and i64 %912, 3968, !dbg !18
  %914 = getelementptr [2 x i8], ptr addrspace(1) %903, i64 %913, !dbg !18
  %915 = getelementptr [2 x i8], ptr addrspace(1) %914, i64 %909, !dbg !18
  %916 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %911, !dbg !18
  %917 = ptrtoint ptr addrspace(3) %916 to i64, !dbg !18
  %918 = trunc i64 %917 to i32, !dbg !18
  %919 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %918), !dbg !18
  %920 = zext i32 %919 to i64, !dbg !18
  %921 = inttoptr i64 %920 to ptr addrspace(3), !dbg !18
  %922 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %923 = ptrtoint ptr addrspace(1) %915 to i64, !dbg !18
  %924 = sub i64 %923, %922, !dbg !18
  %dma.buffer.offset737 = trunc i64 %924 to i32, !dbg !18
  %925 = trunc i64 %922 to i32, !dbg !18
  %926 = insertelement <4 x i32> poison, i32 %925, i64 0, !dbg !18
  %927 = lshr i64 %922, 32, !dbg !18
  %928 = trunc i64 %927 to i32, !dbg !18
  %929 = insertelement <4 x i32> %926, i32 %928, i64 1, !dbg !18
  %930 = insertelement <4 x i32> %929, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc738 = insertelement <4 x i32> %930, i32 131072, i64 3, !dbg !18
  %931 = ptrtoint ptr addrspace(3) %921 to i32, !dbg !18
  %dma.buffer.lds739 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %931), !dbg !18
  %932 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds739, i32 %dma.buffer.offset737, <4 x i32> %dma.buffer.rsrc738) #10, !dbg !18
  %933 = or disjoint i64 %35, 256, !dbg !18
  br label %904, !dbg !18

934:                                              ; preds = %904
  %935 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %936 = extractvalue [2 x <8 x float>] %59, 0, !dbg !35
  %937 = shufflevector <8 x float> %936, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %938 = insertelement <4 x float> poison, float %800, i64 0, !dbg !35
  %939 = shufflevector <4 x float> %938, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %940 = fmul <4 x float> %937, %939, !dbg !35
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %935, <4 x float> %940) #9, !dbg !15
  %942 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %942, <4 x float> %941) #9, !dbg !15
  %944 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %945 = shufflevector <8 x float> %936, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %946 = fmul <4 x float> %945, %939, !dbg !35
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %944, <4 x float> %946) #9, !dbg !15
  %948 = shufflevector <8 x half> %897, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %948, <4 x float> %947) #9, !dbg !15
  %950 = extractvalue [2 x <8 x float>] %59, 1, !dbg !35
  %951 = shufflevector <8 x float> %950, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %952 = insertelement <4 x float> poison, float %806, i64 0, !dbg !35
  %953 = shufflevector <4 x float> %952, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %954 = fmul <4 x float> %951, %953, !dbg !35
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %935, <4 x float> %954) #9, !dbg !15
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %942, <4 x float> %955) #9, !dbg !15
  %957 = shufflevector <8 x float> %950, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %958 = fmul <4 x float> %957, %953, !dbg !35
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %944, <4 x float> %958) #9, !dbg !15
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %948, <4 x float> %959) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread375 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local376 = shl nuw nsw i32 %address.thread375, 3, !dbg !15
  %address.local377 = and i32 %address.local376, 504, !dbg !15
  %address.local378 = zext nneg i32 %address.local377 to i64, !dbg !15
  %address.local379 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local378, !dbg !15
  %961 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local379, i32 0) #0, !dbg !15
  %962 = bitcast <4 x i32> %961 to <8 x half>, !dbg !15
  %address.local380 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local378, !dbg !15
  %address.local381 = getelementptr i8, ptr addrspace(3) %address.local380, i64 1024, !dbg !15
  %963 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local381, i32 0) #0, !dbg !15
  %964 = bitcast <4 x i32> %963 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %965 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !18
  %966 = getelementptr i8, ptr addrspace(1) %965, i64 128, !dbg !18
  br label %967, !dbg !18

967:                                              ; preds = %970, %934
  %968 = phi i64 [ %996, %970 ], [ %35, %934 ], !dbg !18
  %969 = icmp samesign ult i64 %968, 128, !dbg !18
  br i1 %969, label %970, label %997, !dbg !18

970:                                              ; preds = %967
  %971 = shl nuw nsw i64 %968, 3, !dbg !18
  %972 = and i64 %971, 24, !dbg !18
  %973 = shl nuw nsw i64 %968, 3, !dbg !18
  %974 = and i64 %973, 512, !dbg !18
  %975 = shl nuw nsw i64 %968, 5, !dbg !18
  %976 = and i64 %975, 3968, !dbg !18
  %977 = getelementptr [2 x i8], ptr addrspace(1) %966, i64 %976, !dbg !18
  %978 = getelementptr [2 x i8], ptr addrspace(1) %977, i64 %972, !dbg !18
  %979 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %974, !dbg !18
  %980 = ptrtoint ptr addrspace(3) %979 to i64, !dbg !18
  %981 = trunc i64 %980 to i32, !dbg !18
  %982 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %981), !dbg !18
  %983 = zext i32 %982 to i64, !dbg !18
  %984 = inttoptr i64 %983 to ptr addrspace(3), !dbg !18
  %985 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %986 = ptrtoint ptr addrspace(1) %978 to i64, !dbg !18
  %987 = sub i64 %986, %985, !dbg !18
  %dma.buffer.offset740 = trunc i64 %987 to i32, !dbg !18
  %988 = trunc i64 %985 to i32, !dbg !18
  %989 = insertelement <4 x i32> poison, i32 %988, i64 0, !dbg !18
  %990 = lshr i64 %985, 32, !dbg !18
  %991 = trunc i64 %990 to i32, !dbg !18
  %992 = insertelement <4 x i32> %989, i32 %991, i64 1, !dbg !18
  %993 = insertelement <4 x i32> %992, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc741 = insertelement <4 x i32> %993, i32 131072, i64 3, !dbg !18
  %994 = ptrtoint ptr addrspace(3) %984 to i32, !dbg !18
  %dma.buffer.lds742 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %994), !dbg !18
  %995 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds742, i32 %dma.buffer.offset740, <4 x i32> %dma.buffer.rsrc741) #10, !dbg !18
  %996 = or disjoint i64 %35, 256, !dbg !18
  br label %967, !dbg !18

997:                                              ; preds = %967
  %998 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %999 = extractvalue [2 x <8 x float>] %62, 0, !dbg !35
  %1000 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1001 = fmul <4 x float> %1000, %939, !dbg !35
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %998, <4 x float> %1001) #9, !dbg !15
  %1003 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1004 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %1003, <4 x float> %1002) #9, !dbg !15
  %1005 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1006 = shufflevector <8 x float> %999, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1007 = fmul <4 x float> %1006, %939, !dbg !35
  %1008 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1005, <4 x float> %1007) #9, !dbg !15
  %1009 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1010 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %1009, <4 x float> %1008) #9, !dbg !15
  %1011 = extractvalue [2 x <8 x float>] %62, 1, !dbg !35
  %1012 = shufflevector <8 x float> %1011, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1013 = fmul <4 x float> %1012, %953, !dbg !35
  %1014 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %998, <4 x float> %1013) #9, !dbg !15
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %1003, <4 x float> %1014) #9, !dbg !15
  %1016 = shufflevector <8 x float> %1011, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1017 = fmul <4 x float> %1016, %953, !dbg !35
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %1005, <4 x float> %1017) #9, !dbg !15
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %1009, <4 x float> %1018) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread382 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local383 = shl nuw nsw i32 %address.thread382, 3, !dbg !15
  %address.local384 = and i32 %address.local383, 504, !dbg !15
  %address.local385 = zext nneg i32 %address.local384 to i64, !dbg !15
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !15
  %1020 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local386, i32 0) #0, !dbg !15
  %1021 = bitcast <4 x i32> %1020 to <8 x half>, !dbg !15
  %address.local387 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local385, !dbg !15
  %address.local388 = getelementptr i8, ptr addrspace(3) %address.local387, i64 1024, !dbg !15
  %1022 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local388, i32 0) #0, !dbg !15
  %1023 = bitcast <4 x i32> %1022 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1024 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !18
  %1025 = getelementptr i8, ptr addrspace(1) %1024, i64 192, !dbg !18
  br label %1026, !dbg !18

1026:                                             ; preds = %1029, %997
  %1027 = phi i64 [ %1055, %1029 ], [ %35, %997 ], !dbg !18
  %1028 = icmp samesign ult i64 %1027, 128, !dbg !18
  br i1 %1028, label %1029, label %1056, !dbg !18

1029:                                             ; preds = %1026
  %1030 = shl nuw nsw i64 %1027, 3, !dbg !18
  %1031 = and i64 %1030, 24, !dbg !18
  %1032 = shl nuw nsw i64 %1027, 3, !dbg !18
  %1033 = and i64 %1032, 512, !dbg !18
  %1034 = shl nuw nsw i64 %1027, 5, !dbg !18
  %1035 = and i64 %1034, 3968, !dbg !18
  %1036 = getelementptr [2 x i8], ptr addrspace(1) %1025, i64 %1035, !dbg !18
  %1037 = getelementptr [2 x i8], ptr addrspace(1) %1036, i64 %1031, !dbg !18
  %1038 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1033, !dbg !18
  %1039 = ptrtoint ptr addrspace(3) %1038 to i64, !dbg !18
  %1040 = trunc i64 %1039 to i32, !dbg !18
  %1041 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1040), !dbg !18
  %1042 = zext i32 %1041 to i64, !dbg !18
  %1043 = inttoptr i64 %1042 to ptr addrspace(3), !dbg !18
  %1044 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1045 = ptrtoint ptr addrspace(1) %1037 to i64, !dbg !18
  %1046 = sub i64 %1045, %1044, !dbg !18
  %dma.buffer.offset743 = trunc i64 %1046 to i32, !dbg !18
  %1047 = trunc i64 %1044 to i32, !dbg !18
  %1048 = insertelement <4 x i32> poison, i32 %1047, i64 0, !dbg !18
  %1049 = lshr i64 %1044, 32, !dbg !18
  %1050 = trunc i64 %1049 to i32, !dbg !18
  %1051 = insertelement <4 x i32> %1048, i32 %1050, i64 1, !dbg !18
  %1052 = insertelement <4 x i32> %1051, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc744 = insertelement <4 x i32> %1052, i32 131072, i64 3, !dbg !18
  %1053 = ptrtoint ptr addrspace(3) %1043 to i32, !dbg !18
  %dma.buffer.lds745 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1053), !dbg !18
  %1054 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds745, i32 %dma.buffer.offset743, <4 x i32> %dma.buffer.rsrc744) #10, !dbg !18
  %1055 = or disjoint i64 %35, 256, !dbg !18
  br label %1026, !dbg !18

1056:                                             ; preds = %1026
  %1057 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1058 = extractvalue [2 x <8 x float>] %63, 0, !dbg !35
  %1059 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1060 = fmul <4 x float> %1059, %939, !dbg !35
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1057, <4 x float> %1060) #9, !dbg !15
  %1062 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %1062, <4 x float> %1061) #9, !dbg !15
  %1064 = shufflevector <8 x half> %1021, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1065 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1066 = fmul <4 x float> %1065, %939, !dbg !35
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1064, <4 x float> %1066) #9, !dbg !15
  %1068 = shufflevector <8 x half> %1023, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %1068, <4 x float> %1067) #9, !dbg !15
  %1070 = extractvalue [2 x <8 x float>] %63, 1, !dbg !35
  %1071 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1072 = fmul <4 x float> %1071, %953, !dbg !35
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %1057, <4 x float> %1072) #9, !dbg !15
  %1074 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %1062, <4 x float> %1073) #9, !dbg !15
  %1075 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1076 = fmul <4 x float> %1075, %953, !dbg !35
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %1064, <4 x float> %1076) #9, !dbg !15
  %1078 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %1068, <4 x float> %1077) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread389 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local390 = shl nuw nsw i32 %address.thread389, 3, !dbg !15
  %address.local391 = and i32 %address.local390, 504, !dbg !15
  %address.local392 = zext nneg i32 %address.local391 to i64, !dbg !15
  %address.local393 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local392, !dbg !15
  %1079 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local393, i32 0) #0, !dbg !15
  %1080 = bitcast <4 x i32> %1079 to <8 x half>, !dbg !15
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local392, !dbg !15
  %address.local395 = getelementptr i8, ptr addrspace(3) %address.local394, i64 1024, !dbg !15
  %1081 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local395, i32 0) #0, !dbg !15
  %1082 = bitcast <4 x i32> %1081 to <8 x half>, !dbg !15
  %1083 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1084 = extractvalue [2 x <8 x float>] %64, 0, !dbg !35
  %1085 = shufflevector <8 x float> %1084, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1086 = fmul <4 x float> %1085, %939, !dbg !35
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1083, <4 x float> %1086) #9, !dbg !15
  %1088 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %1088, <4 x float> %1087) #9, !dbg !15
  %1090 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1091 = shufflevector <8 x float> %1084, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1092 = fmul <4 x float> %1091, %939, !dbg !35
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1090, <4 x float> %1092) #9, !dbg !15
  %1094 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed50, <4 x half> %1094, <4 x float> %1093) #9, !dbg !15
  %1096 = extractvalue [2 x <8 x float>] %64, 1, !dbg !35
  %1097 = shufflevector <8 x float> %1096, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !35
  %1098 = fmul <4 x float> %1097, %953, !dbg !35
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %1083, <4 x float> %1098) #9, !dbg !15
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %1088, <4 x float> %1099) #9, !dbg !15
  %1101 = shufflevector <8 x float> %1096, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1102 = fmul <4 x float> %1101, %953, !dbg !35
  %1103 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed52, <4 x half> %1090, <4 x float> %1102) #9, !dbg !15
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed54, <4 x half> %1094, <4 x float> %1103) #9, !dbg !15
  %1105 = shufflevector <4 x float> %943, <4 x float> %949, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1106 = insertvalue [2 x <8 x float>] poison, <8 x float> %1105, 0, !dbg !7
  %1107 = shufflevector <4 x float> %956, <4 x float> %960, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1108 = insertvalue [2 x <8 x float>] %1106, <8 x float> %1107, 1, !dbg !7
  %1109 = insertelement <1 x float> poison, float %884, i64 0, !dbg !7
  %1110 = insertvalue [2 x <1 x float>] poison, <1 x float> %1109, 0, !dbg !7
  %1111 = insertelement <1 x float> poison, float %893, i64 0, !dbg !7
  %1112 = insertvalue [2 x <1 x float>] %1110, <1 x float> %1111, 1, !dbg !7
  %1113 = insertvalue [2 x <1 x float>] poison, <1 x float> %772, 0, !dbg !7
  %1114 = insertvalue [2 x <1 x float>] %1113, <1 x float> %791, 1, !dbg !7
  %1115 = shufflevector <4 x float> %1004, <4 x float> %1010, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1116 = insertvalue [2 x <8 x float>] poison, <8 x float> %1115, 0, !dbg !7
  %1117 = shufflevector <4 x float> %1015, <4 x float> %1019, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1118 = insertvalue [2 x <8 x float>] %1116, <8 x float> %1117, 1, !dbg !7
  %1119 = shufflevector <4 x float> %1063, <4 x float> %1069, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1120 = insertvalue [2 x <8 x float>] poison, <8 x float> %1119, 0, !dbg !7
  %1121 = shufflevector <4 x float> %1074, <4 x float> %1078, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1122 = insertvalue [2 x <8 x float>] %1120, <8 x float> %1121, 1, !dbg !7
  %1123 = shufflevector <4 x float> %1089, <4 x float> %1095, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1124 = insertvalue [2 x <8 x float>] poison, <8 x float> %1123, 0, !dbg !7
  %1125 = shufflevector <4 x float> %1100, <4 x float> %1104, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1126 = insertvalue [2 x <8 x float>] %1124, <8 x float> %1125, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1127 = add nuw nsw i64 %58, 1, !dbg !7
  br label %57, !dbg !7

1128:                                             ; preds = %57
  %1129 = extractvalue [2 x <8 x float>] %64, 0, !dbg !7
  %1130 = extractelement <8 x float> %1129, i64 0, !dbg !7
  %1131 = extractelement <8 x float> %1129, i64 1, !dbg !7
  %1132 = extractelement <8 x float> %1129, i64 2, !dbg !7
  %1133 = extractelement <8 x float> %1129, i64 3, !dbg !7
  %1134 = select i1 %.not, float %1131, float %1130, !dbg !7
  %1135 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1136 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1135), !dbg !7
  %1137 = add nuw nsw i32 %1136, 64, !dbg !7
  %1138 = xor i32 %1136, 16, !dbg !7
  %1139 = and i32 %1137, 192, !dbg !7
  %1140 = icmp samesign ult i32 %1138, %1139, !dbg !7
  %1141 = select i1 %1140, i32 %1138, i32 %1136, !dbg !7
  %1142 = shl nuw nsw i32 %1141, 2, !dbg !7
  %1143 = bitcast float %1134 to i32, !dbg !7
  %1144 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1143), !dbg !7
  %1145 = bitcast i32 %1144 to float, !dbg !7
  %1146 = select i1 %.not, float %1130, float %1145, !dbg !7
  %1147 = select i1 %.not, float %1145, float %1131, !dbg !7
  %1148 = select i1 %.not, float %1133, float %1132, !dbg !7
  %1149 = bitcast float %1148 to i32, !dbg !7
  %1150 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1149), !dbg !7
  %1151 = bitcast i32 %1150 to float, !dbg !7
  %1152 = select i1 %.not, float %1132, float %1151, !dbg !7
  %1153 = select i1 %.not, float %1151, float %1133, !dbg !7
  %1154 = select i1 %.not1, float %1152, float %1146, !dbg !7
  %1155 = xor i32 %1136, 32, !dbg !7
  %1156 = icmp samesign ult i32 %1155, %1139, !dbg !7
  %1157 = select i1 %1156, i32 %1155, i32 %1136, !dbg !7
  %1158 = shl nuw nsw i32 %1157, 2, !dbg !7
  %1159 = bitcast float %1154 to i32, !dbg !7
  %1160 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1159), !dbg !7
  %1161 = bitcast i32 %1160 to float, !dbg !7
  %1162 = select i1 %.not1, float %1146, float %1161, !dbg !7
  %1163 = select i1 %.not1, float %1161, float %1152, !dbg !7
  %1164 = select i1 %.not1, float %1153, float %1147, !dbg !7
  %1165 = bitcast float %1164 to i32, !dbg !7
  %1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1165), !dbg !7
  %1167 = bitcast i32 %1166 to float, !dbg !7
  %1168 = select i1 %.not1, float %1147, float %1167, !dbg !7
  %1169 = select i1 %.not1, float %1167, float %1153, !dbg !7
  %1170 = extractvalue [2 x <8 x float>] %63, 0, !dbg !7
  %1171 = extractelement <8 x float> %1170, i64 0, !dbg !7
  %1172 = extractelement <8 x float> %1170, i64 1, !dbg !7
  %1173 = extractelement <8 x float> %1170, i64 2, !dbg !7
  %1174 = extractelement <8 x float> %1170, i64 3, !dbg !7
  %1175 = select i1 %.not, float %1172, float %1171, !dbg !7
  %1176 = bitcast float %1175 to i32, !dbg !7
  %1177 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1176), !dbg !7
  %1178 = bitcast i32 %1177 to float, !dbg !7
  %1179 = select i1 %.not, float %1171, float %1178, !dbg !7
  %1180 = select i1 %.not, float %1178, float %1172, !dbg !7
  %1181 = select i1 %.not, float %1174, float %1173, !dbg !7
  %1182 = bitcast float %1181 to i32, !dbg !7
  %1183 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1182), !dbg !7
  %1184 = bitcast i32 %1183 to float, !dbg !7
  %1185 = select i1 %.not, float %1173, float %1184, !dbg !7
  %1186 = select i1 %.not, float %1184, float %1174, !dbg !7
  %1187 = select i1 %.not1, float %1185, float %1179, !dbg !7
  %1188 = bitcast float %1187 to i32, !dbg !7
  %1189 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1188), !dbg !7
  %1190 = bitcast i32 %1189 to float, !dbg !7
  %1191 = select i1 %.not1, float %1179, float %1190, !dbg !7
  %1192 = select i1 %.not1, float %1190, float %1185, !dbg !7
  %1193 = select i1 %.not1, float %1186, float %1180, !dbg !7
  %1194 = bitcast float %1193 to i32, !dbg !7
  %1195 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1194), !dbg !7
  %1196 = bitcast i32 %1195 to float, !dbg !7
  %1197 = select i1 %.not1, float %1180, float %1196, !dbg !7
  %1198 = select i1 %.not1, float %1196, float %1186, !dbg !7
  %1199 = extractvalue [2 x <8 x float>] %62, 0, !dbg !7
  %1200 = extractelement <8 x float> %1199, i64 0, !dbg !7
  %1201 = extractelement <8 x float> %1199, i64 1, !dbg !7
  %1202 = extractelement <8 x float> %1199, i64 2, !dbg !7
  %1203 = extractelement <8 x float> %1199, i64 3, !dbg !7
  %1204 = select i1 %.not, float %1201, float %1200, !dbg !7
  %1205 = bitcast float %1204 to i32, !dbg !7
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1205), !dbg !7
  %1207 = bitcast i32 %1206 to float, !dbg !7
  %1208 = select i1 %.not, float %1200, float %1207, !dbg !7
  %1209 = select i1 %.not, float %1207, float %1201, !dbg !7
  %1210 = select i1 %.not, float %1203, float %1202, !dbg !7
  %1211 = bitcast float %1210 to i32, !dbg !7
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1211), !dbg !7
  %1213 = bitcast i32 %1212 to float, !dbg !7
  %1214 = select i1 %.not, float %1202, float %1213, !dbg !7
  %1215 = select i1 %.not, float %1213, float %1203, !dbg !7
  %1216 = select i1 %.not1, float %1214, float %1208, !dbg !7
  %1217 = bitcast float %1216 to i32, !dbg !7
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1217), !dbg !7
  %1219 = bitcast i32 %1218 to float, !dbg !7
  %1220 = select i1 %.not1, float %1208, float %1219, !dbg !7
  %1221 = select i1 %.not1, float %1219, float %1214, !dbg !7
  %1222 = select i1 %.not1, float %1215, float %1209, !dbg !7
  %1223 = bitcast float %1222 to i32, !dbg !7
  %1224 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1223), !dbg !7
  %1225 = bitcast i32 %1224 to float, !dbg !7
  %1226 = select i1 %.not1, float %1209, float %1225, !dbg !7
  %1227 = select i1 %.not1, float %1225, float %1215, !dbg !7
  %1228 = extractvalue [2 x <8 x float>] %59, 0, !dbg !7
  %1229 = extractelement <8 x float> %1228, i64 0, !dbg !7
  %1230 = extractelement <8 x float> %1228, i64 1, !dbg !7
  %1231 = extractelement <8 x float> %1228, i64 2, !dbg !7
  %1232 = extractelement <8 x float> %1228, i64 3, !dbg !7
  %1233 = select i1 %.not, float %1230, float %1229, !dbg !7
  %1234 = bitcast float %1233 to i32, !dbg !7
  %1235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1234), !dbg !7
  %1236 = bitcast i32 %1235 to float, !dbg !7
  %1237 = select i1 %.not, float %1229, float %1236, !dbg !7
  %1238 = select i1 %.not, float %1236, float %1230, !dbg !7
  %1239 = select i1 %.not, float %1232, float %1231, !dbg !7
  %1240 = bitcast float %1239 to i32, !dbg !7
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1240), !dbg !7
  %1242 = bitcast i32 %1241 to float, !dbg !7
  %1243 = select i1 %.not, float %1231, float %1242, !dbg !7
  %1244 = select i1 %.not, float %1242, float %1232, !dbg !7
  %1245 = select i1 %.not1, float %1243, float %1237, !dbg !7
  %1246 = bitcast float %1245 to i32, !dbg !7
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1246), !dbg !7
  %1248 = select i1 %.not1, float %1244, float %1238, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1249), !dbg !7
  %1251 = extractvalue [2 x <1 x float>] %60, 0, !dbg !19
  %1252 = shufflevector <1 x float> %1251, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1253 = extractelement <8 x float> %1129, i64 4, !dbg !7
  %1254 = extractelement <8 x float> %1129, i64 5, !dbg !7
  %1255 = extractelement <8 x float> %1129, i64 6, !dbg !7
  %1256 = extractelement <8 x float> %1129, i64 7, !dbg !7
  %1257 = select i1 %.not, float %1254, float %1253, !dbg !7
  %1258 = bitcast float %1257 to i32, !dbg !7
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1258), !dbg !7
  %1260 = bitcast i32 %1259 to float, !dbg !7
  %1261 = select i1 %.not, float %1253, float %1260, !dbg !7
  %1262 = select i1 %.not, float %1260, float %1254, !dbg !7
  %1263 = select i1 %.not, float %1256, float %1255, !dbg !7
  %1264 = bitcast float %1263 to i32, !dbg !7
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1264), !dbg !7
  %1266 = bitcast i32 %1265 to float, !dbg !7
  %1267 = select i1 %.not, float %1255, float %1266, !dbg !7
  %1268 = select i1 %.not, float %1266, float %1256, !dbg !7
  %1269 = select i1 %.not1, float %1267, float %1261, !dbg !7
  %1270 = bitcast float %1269 to i32, !dbg !7
  %1271 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1270), !dbg !7
  %1272 = bitcast i32 %1271 to float, !dbg !7
  %1273 = select i1 %.not1, float %1261, float %1272, !dbg !7
  %1274 = select i1 %.not1, float %1272, float %1267, !dbg !7
  %1275 = select i1 %.not1, float %1268, float %1262, !dbg !7
  %1276 = bitcast float %1275 to i32, !dbg !7
  %1277 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1276), !dbg !7
  %1278 = bitcast i32 %1277 to float, !dbg !7
  %1279 = select i1 %.not1, float %1262, float %1278, !dbg !7
  %1280 = select i1 %.not1, float %1278, float %1268, !dbg !7
  %1281 = extractelement <8 x float> %1170, i64 4, !dbg !7
  %1282 = extractelement <8 x float> %1170, i64 5, !dbg !7
  %1283 = extractelement <8 x float> %1170, i64 6, !dbg !7
  %1284 = extractelement <8 x float> %1170, i64 7, !dbg !7
  %1285 = select i1 %.not, float %1282, float %1281, !dbg !7
  %1286 = bitcast float %1285 to i32, !dbg !7
  %1287 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1286), !dbg !7
  %1288 = bitcast i32 %1287 to float, !dbg !7
  %1289 = select i1 %.not, float %1281, float %1288, !dbg !7
  %1290 = select i1 %.not, float %1288, float %1282, !dbg !7
  %1291 = select i1 %.not, float %1284, float %1283, !dbg !7
  %1292 = bitcast float %1291 to i32, !dbg !7
  %1293 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1292), !dbg !7
  %1294 = bitcast i32 %1293 to float, !dbg !7
  %1295 = select i1 %.not, float %1283, float %1294, !dbg !7
  %1296 = select i1 %.not, float %1294, float %1284, !dbg !7
  %1297 = select i1 %.not1, float %1295, float %1289, !dbg !7
  %1298 = bitcast float %1297 to i32, !dbg !7
  %1299 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1298), !dbg !7
  %1300 = bitcast i32 %1299 to float, !dbg !7
  %1301 = select i1 %.not1, float %1289, float %1300, !dbg !7
  %1302 = select i1 %.not1, float %1300, float %1295, !dbg !7
  %1303 = select i1 %.not1, float %1296, float %1290, !dbg !7
  %1304 = bitcast float %1303 to i32, !dbg !7
  %1305 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1304), !dbg !7
  %1306 = bitcast i32 %1305 to float, !dbg !7
  %1307 = select i1 %.not1, float %1290, float %1306, !dbg !7
  %1308 = select i1 %.not1, float %1306, float %1296, !dbg !7
  %1309 = extractelement <8 x float> %1199, i64 4, !dbg !7
  %1310 = extractelement <8 x float> %1199, i64 5, !dbg !7
  %1311 = extractelement <8 x float> %1199, i64 6, !dbg !7
  %1312 = extractelement <8 x float> %1199, i64 7, !dbg !7
  %1313 = select i1 %.not, float %1310, float %1309, !dbg !7
  %1314 = bitcast float %1313 to i32, !dbg !7
  %1315 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1314), !dbg !7
  %1316 = bitcast i32 %1315 to float, !dbg !7
  %1317 = select i1 %.not, float %1309, float %1316, !dbg !7
  %1318 = select i1 %.not, float %1316, float %1310, !dbg !7
  %1319 = select i1 %.not, float %1312, float %1311, !dbg !7
  %1320 = bitcast float %1319 to i32, !dbg !7
  %1321 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1320), !dbg !7
  %1322 = bitcast i32 %1321 to float, !dbg !7
  %1323 = select i1 %.not, float %1311, float %1322, !dbg !7
  %1324 = select i1 %.not, float %1322, float %1312, !dbg !7
  %1325 = select i1 %.not1, float %1323, float %1317, !dbg !7
  %1326 = bitcast float %1325 to i32, !dbg !7
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1326), !dbg !7
  %1328 = bitcast i32 %1327 to float, !dbg !7
  %1329 = select i1 %.not1, float %1317, float %1328, !dbg !7
  %1330 = select i1 %.not1, float %1328, float %1323, !dbg !7
  %1331 = select i1 %.not1, float %1324, float %1318, !dbg !7
  %1332 = bitcast float %1331 to i32, !dbg !7
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1332), !dbg !7
  %1334 = bitcast i32 %1333 to float, !dbg !7
  %1335 = select i1 %.not1, float %1318, float %1334, !dbg !7
  %1336 = select i1 %.not1, float %1334, float %1324, !dbg !7
  %1337 = extractelement <8 x float> %1228, i64 4, !dbg !7
  %1338 = extractelement <8 x float> %1228, i64 5, !dbg !7
  %1339 = extractelement <8 x float> %1228, i64 6, !dbg !7
  %1340 = extractelement <8 x float> %1228, i64 7, !dbg !7
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !7
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !7
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !7
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !7
  %1353 = select i1 %.not1, float %1351, float %1345, !dbg !7
  %1354 = bitcast float %1353 to i32, !dbg !7
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1354), !dbg !7
  %1356 = select i1 %.not1, float %1352, float %1346, !dbg !7
  %1357 = bitcast float %1356 to i32, !dbg !7
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1357), !dbg !7
  %1359 = extractvalue [2 x <8 x float>] %64, 1, !dbg !7
  %1360 = extractelement <8 x float> %1359, i64 0, !dbg !7
  %1361 = extractelement <8 x float> %1359, i64 1, !dbg !7
  %1362 = extractelement <8 x float> %1359, i64 2, !dbg !7
  %1363 = extractelement <8 x float> %1359, i64 3, !dbg !7
  %1364 = select i1 %.not, float %1361, float %1360, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1365), !dbg !7
  %1367 = bitcast i32 %1366 to float, !dbg !7
  %1368 = select i1 %.not, float %1360, float %1367, !dbg !7
  %1369 = select i1 %.not, float %1367, float %1361, !dbg !7
  %1370 = select i1 %.not, float %1363, float %1362, !dbg !7
  %1371 = bitcast float %1370 to i32, !dbg !7
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1371), !dbg !7
  %1373 = bitcast i32 %1372 to float, !dbg !7
  %1374 = select i1 %.not, float %1362, float %1373, !dbg !7
  %1375 = select i1 %.not, float %1373, float %1363, !dbg !7
  %1376 = select i1 %.not1, float %1374, float %1368, !dbg !7
  %1377 = bitcast float %1376 to i32, !dbg !7
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1377), !dbg !7
  %1379 = bitcast i32 %1378 to float, !dbg !7
  %1380 = select i1 %.not1, float %1368, float %1379, !dbg !7
  %1381 = select i1 %.not1, float %1379, float %1374, !dbg !7
  %1382 = select i1 %.not1, float %1375, float %1369, !dbg !7
  %1383 = bitcast float %1382 to i32, !dbg !7
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1383), !dbg !7
  %1385 = bitcast i32 %1384 to float, !dbg !7
  %1386 = select i1 %.not1, float %1369, float %1385, !dbg !7
  %1387 = select i1 %.not1, float %1385, float %1375, !dbg !7
  %1388 = extractvalue [2 x <8 x float>] %63, 1, !dbg !7
  %1389 = extractelement <8 x float> %1388, i64 0, !dbg !7
  %1390 = extractelement <8 x float> %1388, i64 1, !dbg !7
  %1391 = extractelement <8 x float> %1388, i64 2, !dbg !7
  %1392 = extractelement <8 x float> %1388, i64 3, !dbg !7
  %1393 = select i1 %.not, float %1390, float %1389, !dbg !7
  %1394 = bitcast float %1393 to i32, !dbg !7
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1394), !dbg !7
  %1396 = bitcast i32 %1395 to float, !dbg !7
  %1397 = select i1 %.not, float %1389, float %1396, !dbg !7
  %1398 = select i1 %.not, float %1396, float %1390, !dbg !7
  %1399 = select i1 %.not, float %1392, float %1391, !dbg !7
  %1400 = bitcast float %1399 to i32, !dbg !7
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1400), !dbg !7
  %1402 = bitcast i32 %1401 to float, !dbg !7
  %1403 = select i1 %.not, float %1391, float %1402, !dbg !7
  %1404 = select i1 %.not, float %1402, float %1392, !dbg !7
  %1405 = select i1 %.not1, float %1403, float %1397, !dbg !7
  %1406 = bitcast float %1405 to i32, !dbg !7
  %1407 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1406), !dbg !7
  %1408 = bitcast i32 %1407 to float, !dbg !7
  %1409 = select i1 %.not1, float %1397, float %1408, !dbg !7
  %1410 = select i1 %.not1, float %1408, float %1403, !dbg !7
  %1411 = select i1 %.not1, float %1404, float %1398, !dbg !7
  %1412 = bitcast float %1411 to i32, !dbg !7
  %1413 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1412), !dbg !7
  %1414 = bitcast i32 %1413 to float, !dbg !7
  %1415 = select i1 %.not1, float %1398, float %1414, !dbg !7
  %1416 = select i1 %.not1, float %1414, float %1404, !dbg !7
  %1417 = extractvalue [2 x <8 x float>] %62, 1, !dbg !7
  %1418 = extractelement <8 x float> %1417, i64 0, !dbg !7
  %1419 = extractelement <8 x float> %1417, i64 1, !dbg !7
  %1420 = extractelement <8 x float> %1417, i64 2, !dbg !7
  %1421 = extractelement <8 x float> %1417, i64 3, !dbg !7
  %1422 = select i1 %.not, float %1419, float %1418, !dbg !7
  %1423 = bitcast float %1422 to i32, !dbg !7
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1423), !dbg !7
  %1425 = bitcast i32 %1424 to float, !dbg !7
  %1426 = select i1 %.not, float %1418, float %1425, !dbg !7
  %1427 = select i1 %.not, float %1425, float %1419, !dbg !7
  %1428 = select i1 %.not, float %1421, float %1420, !dbg !7
  %1429 = bitcast float %1428 to i32, !dbg !7
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1429), !dbg !7
  %1431 = bitcast i32 %1430 to float, !dbg !7
  %1432 = select i1 %.not, float %1420, float %1431, !dbg !7
  %1433 = select i1 %.not, float %1431, float %1421, !dbg !7
  %1434 = select i1 %.not1, float %1432, float %1426, !dbg !7
  %1435 = bitcast float %1434 to i32, !dbg !7
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1435), !dbg !7
  %1437 = bitcast i32 %1436 to float, !dbg !7
  %1438 = select i1 %.not1, float %1426, float %1437, !dbg !7
  %1439 = select i1 %.not1, float %1437, float %1432, !dbg !7
  %1440 = select i1 %.not1, float %1433, float %1427, !dbg !7
  %1441 = bitcast float %1440 to i32, !dbg !7
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1441), !dbg !7
  %1443 = bitcast i32 %1442 to float, !dbg !7
  %1444 = select i1 %.not1, float %1427, float %1443, !dbg !7
  %1445 = select i1 %.not1, float %1443, float %1433, !dbg !7
  %1446 = extractvalue [2 x <8 x float>] %59, 1, !dbg !7
  %1447 = extractelement <8 x float> %1446, i64 0, !dbg !7
  %1448 = extractelement <8 x float> %1446, i64 1, !dbg !7
  %1449 = extractelement <8 x float> %1446, i64 2, !dbg !7
  %1450 = extractelement <8 x float> %1446, i64 3, !dbg !7
  %1451 = select i1 %.not, float %1448, float %1447, !dbg !7
  %1452 = bitcast float %1451 to i32, !dbg !7
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1452), !dbg !7
  %1454 = bitcast i32 %1453 to float, !dbg !7
  %1455 = select i1 %.not, float %1447, float %1454, !dbg !7
  %1456 = select i1 %.not, float %1454, float %1448, !dbg !7
  %1457 = select i1 %.not, float %1450, float %1449, !dbg !7
  %1458 = bitcast float %1457 to i32, !dbg !7
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1458), !dbg !7
  %1460 = bitcast i32 %1459 to float, !dbg !7
  %1461 = select i1 %.not, float %1449, float %1460, !dbg !7
  %1462 = select i1 %.not, float %1460, float %1450, !dbg !7
  %1463 = select i1 %.not1, float %1461, float %1455, !dbg !7
  %1464 = bitcast float %1463 to i32, !dbg !7
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1464), !dbg !7
  %1466 = select i1 %.not1, float %1462, float %1456, !dbg !7
  %1467 = bitcast float %1466 to i32, !dbg !7
  %1468 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1467), !dbg !7
  %1469 = extractvalue [2 x <1 x float>] %60, 1, !dbg !19
  %1470 = shufflevector <1 x float> %1469, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1471 = extractelement <8 x float> %1359, i64 4, !dbg !7
  %1472 = extractelement <8 x float> %1359, i64 5, !dbg !7
  %1473 = extractelement <8 x float> %1359, i64 6, !dbg !7
  %1474 = extractelement <8 x float> %1359, i64 7, !dbg !7
  %1475 = select i1 %.not, float %1472, float %1471, !dbg !7
  %1476 = bitcast float %1475 to i32, !dbg !7
  %1477 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1476), !dbg !7
  %1478 = bitcast i32 %1477 to float, !dbg !7
  %1479 = select i1 %.not, float %1471, float %1478, !dbg !7
  %1480 = select i1 %.not, float %1478, float %1472, !dbg !7
  %1481 = select i1 %.not, float %1474, float %1473, !dbg !7
  %1482 = bitcast float %1481 to i32, !dbg !7
  %1483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1482), !dbg !7
  %1484 = bitcast i32 %1483 to float, !dbg !7
  %1485 = select i1 %.not, float %1473, float %1484, !dbg !7
  %1486 = select i1 %.not, float %1484, float %1474, !dbg !7
  %1487 = select i1 %.not1, float %1485, float %1479, !dbg !7
  %1488 = bitcast float %1487 to i32, !dbg !7
  %1489 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1488), !dbg !7
  %1490 = bitcast i32 %1489 to float, !dbg !7
  %1491 = select i1 %.not1, float %1479, float %1490, !dbg !7
  %1492 = select i1 %.not1, float %1490, float %1485, !dbg !7
  %1493 = select i1 %.not1, float %1486, float %1480, !dbg !7
  %1494 = bitcast float %1493 to i32, !dbg !7
  %1495 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1494), !dbg !7
  %1496 = bitcast i32 %1495 to float, !dbg !7
  %1497 = select i1 %.not1, float %1480, float %1496, !dbg !7
  %1498 = select i1 %.not1, float %1496, float %1486, !dbg !7
  %1499 = extractelement <8 x float> %1388, i64 4, !dbg !7
  %1500 = extractelement <8 x float> %1388, i64 5, !dbg !7
  %1501 = extractelement <8 x float> %1388, i64 6, !dbg !7
  %1502 = extractelement <8 x float> %1388, i64 7, !dbg !7
  %1503 = select i1 %.not, float %1500, float %1499, !dbg !7
  %1504 = bitcast float %1503 to i32, !dbg !7
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1504), !dbg !7
  %1506 = bitcast i32 %1505 to float, !dbg !7
  %1507 = select i1 %.not, float %1499, float %1506, !dbg !7
  %1508 = select i1 %.not, float %1506, float %1500, !dbg !7
  %1509 = select i1 %.not, float %1502, float %1501, !dbg !7
  %1510 = bitcast float %1509 to i32, !dbg !7
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1510), !dbg !7
  %1512 = bitcast i32 %1511 to float, !dbg !7
  %1513 = select i1 %.not, float %1501, float %1512, !dbg !7
  %1514 = select i1 %.not, float %1512, float %1502, !dbg !7
  %1515 = select i1 %.not1, float %1513, float %1507, !dbg !7
  %1516 = bitcast float %1515 to i32, !dbg !7
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1516), !dbg !7
  %1518 = bitcast i32 %1517 to float, !dbg !7
  %1519 = select i1 %.not1, float %1507, float %1518, !dbg !7
  %1520 = select i1 %.not1, float %1518, float %1513, !dbg !7
  %1521 = select i1 %.not1, float %1514, float %1508, !dbg !7
  %1522 = bitcast float %1521 to i32, !dbg !7
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1522), !dbg !7
  %1524 = bitcast i32 %1523 to float, !dbg !7
  %1525 = select i1 %.not1, float %1508, float %1524, !dbg !7
  %1526 = select i1 %.not1, float %1524, float %1514, !dbg !7
  %1527 = extractelement <8 x float> %1417, i64 4, !dbg !7
  %1528 = extractelement <8 x float> %1417, i64 5, !dbg !7
  %1529 = extractelement <8 x float> %1417, i64 6, !dbg !7
  %1530 = extractelement <8 x float> %1417, i64 7, !dbg !7
  %1531 = select i1 %.not, float %1528, float %1527, !dbg !7
  %1532 = bitcast float %1531 to i32, !dbg !7
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1532), !dbg !7
  %1534 = bitcast i32 %1533 to float, !dbg !7
  %1535 = select i1 %.not, float %1527, float %1534, !dbg !7
  %1536 = select i1 %.not, float %1534, float %1528, !dbg !7
  %1537 = select i1 %.not, float %1530, float %1529, !dbg !7
  %1538 = bitcast float %1537 to i32, !dbg !7
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1538), !dbg !7
  %1540 = bitcast i32 %1539 to float, !dbg !7
  %1541 = select i1 %.not, float %1529, float %1540, !dbg !7
  %1542 = select i1 %.not, float %1540, float %1530, !dbg !7
  %1543 = select i1 %.not1, float %1541, float %1535, !dbg !7
  %1544 = bitcast float %1543 to i32, !dbg !7
  %1545 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1544), !dbg !7
  %1546 = bitcast i32 %1545 to float, !dbg !7
  %1547 = select i1 %.not1, float %1535, float %1546, !dbg !7
  %1548 = select i1 %.not1, float %1546, float %1541, !dbg !7
  %1549 = select i1 %.not1, float %1542, float %1536, !dbg !7
  %1550 = bitcast float %1549 to i32, !dbg !7
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1550), !dbg !7
  %1552 = bitcast i32 %1551 to float, !dbg !7
  %1553 = select i1 %.not1, float %1536, float %1552, !dbg !7
  %1554 = select i1 %.not1, float %1552, float %1542, !dbg !7
  %1555 = extractelement <8 x float> %1446, i64 4, !dbg !7
  %1556 = extractelement <8 x float> %1446, i64 5, !dbg !7
  %1557 = extractelement <8 x float> %1446, i64 6, !dbg !7
  %1558 = extractelement <8 x float> %1446, i64 7, !dbg !7
  %1559 = select i1 %.not, float %1556, float %1555, !dbg !7
  %1560 = bitcast float %1559 to i32, !dbg !7
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1560), !dbg !7
  %1562 = bitcast i32 %1561 to float, !dbg !7
  %1563 = select i1 %.not, float %1555, float %1562, !dbg !7
  %1564 = select i1 %.not, float %1562, float %1556, !dbg !7
  %1565 = select i1 %.not, float %1558, float %1557, !dbg !7
  %1566 = bitcast float %1565 to i32, !dbg !7
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1142, i32 %1566), !dbg !7
  %1568 = bitcast i32 %1567 to float, !dbg !7
  %1569 = select i1 %.not, float %1557, float %1568, !dbg !7
  %1570 = select i1 %.not, float %1568, float %1558, !dbg !7
  %1571 = select i1 %.not1, float %1569, float %1563, !dbg !7
  %1572 = bitcast float %1571 to i32, !dbg !7
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1572), !dbg !7
  %1574 = select i1 %.not1, float %1570, float %1564, !dbg !7
  %1575 = bitcast float %1574 to i32, !dbg !7
  %1576 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1158, i32 %1575), !dbg !7
  br i1 true, label %1577, label %1641, !dbg !21

1577:                                             ; preds = %1128
  %1578 = bitcast i32 %1573 to float, !dbg !7
  %1579 = select i1 %.not1, float %1563, float %1578, !dbg !7
  %1580 = insertelement <4 x float> poison, float %1579, i64 0, !dbg !19
  %1581 = bitcast i32 %1576 to float, !dbg !7
  %1582 = select i1 %.not1, float %1564, float %1581, !dbg !7
  %1583 = insertelement <4 x float> %1580, float %1582, i64 1, !dbg !19
  %1584 = select i1 %.not1, float %1578, float %1569, !dbg !7
  %1585 = insertelement <4 x float> %1583, float %1584, i64 2, !dbg !19
  %1586 = select i1 %.not1, float %1581, float %1570, !dbg !7
  %1587 = insertelement <4 x float> %1585, float %1586, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !19
  %normalized = fmul <4 x float> %1587, %row.reciprocal, !dbg !19
  %1588 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1589 = extractelement <4 x half> %1588, i64 3, !dbg !20
  %1590 = extractelement <4 x half> %1588, i64 2, !dbg !20
  %1591 = extractelement <4 x half> %1588, i64 1, !dbg !20
  %1592 = extractelement <4 x half> %1588, i64 0, !dbg !20
  %1593 = bitcast i32 %1465 to float, !dbg !7
  %1594 = select i1 %.not1, float %1455, float %1593, !dbg !7
  %1595 = insertelement <4 x float> poison, float %1594, i64 0, !dbg !19
  %1596 = bitcast i32 %1468 to float, !dbg !7
  %1597 = select i1 %.not1, float %1456, float %1596, !dbg !7
  %1598 = insertelement <4 x float> %1595, float %1597, i64 1, !dbg !19
  %1599 = select i1 %.not1, float %1593, float %1461, !dbg !7
  %1600 = insertelement <4 x float> %1598, float %1599, i64 2, !dbg !19
  %1601 = select i1 %.not1, float %1596, float %1462, !dbg !7
  %1602 = insertelement <4 x float> %1600, float %1601, i64 3, !dbg !19
  %normalized23 = fmul <4 x float> %1602, %row.reciprocal, !dbg !19
  %1603 = fptrunc <4 x float> %normalized23 to <4 x half>, !dbg !20
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !20
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !20
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !20
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !20
  %1608 = bitcast i32 %1355 to float, !dbg !7
  %1609 = select i1 %.not1, float %1345, float %1608, !dbg !7
  %1610 = insertelement <4 x float> poison, float %1609, i64 0, !dbg !19
  %1611 = bitcast i32 %1358 to float, !dbg !7
  %1612 = select i1 %.not1, float %1346, float %1611, !dbg !7
  %1613 = insertelement <4 x float> %1610, float %1612, i64 1, !dbg !19
  %1614 = select i1 %.not1, float %1608, float %1351, !dbg !7
  %1615 = insertelement <4 x float> %1613, float %1614, i64 2, !dbg !19
  %1616 = select i1 %.not1, float %1611, float %1352, !dbg !7
  %1617 = insertelement <4 x float> %1615, float %1616, i64 3, !dbg !19
  %row.reciprocal24 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !19
  %normalized25 = fmul <4 x float> %1617, %row.reciprocal24, !dbg !19
  %1618 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !20
  %1619 = extractelement <4 x half> %1618, i64 3, !dbg !20
  %1620 = extractelement <4 x half> %1618, i64 2, !dbg !20
  %1621 = extractelement <4 x half> %1618, i64 1, !dbg !20
  %1622 = extractelement <4 x half> %1618, i64 0, !dbg !20
  %1623 = bitcast i32 %1247 to float, !dbg !7
  %1624 = select i1 %.not1, float %1237, float %1623, !dbg !7
  %1625 = insertelement <4 x float> poison, float %1624, i64 0, !dbg !19
  %1626 = bitcast i32 %1250 to float, !dbg !7
  %1627 = select i1 %.not1, float %1238, float %1626, !dbg !7
  %1628 = insertelement <4 x float> %1625, float %1627, i64 1, !dbg !19
  %1629 = select i1 %.not1, float %1623, float %1243, !dbg !7
  %1630 = insertelement <4 x float> %1628, float %1629, i64 2, !dbg !19
  %1631 = select i1 %.not1, float %1626, float %1244, !dbg !7
  %1632 = insertelement <4 x float> %1630, float %1631, i64 3, !dbg !19
  %normalized26 = fmul <4 x float> %1632, %row.reciprocal24, !dbg !19
  %1633 = fptrunc <4 x float> %normalized26 to <4 x half>, !dbg !20
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !20
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !20
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !20
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !20
  %1638 = add nsw i64 %15, %25, !dbg !21
  %1639 = shl i64 %1638, 19, !dbg !21
  %1640 = and i64 %1639, 16252928, !dbg !21
  %address.thread396 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local397 = zext nneg i32 %address.thread396 to i64, !dbg !12
  %address.local398 = add i64 %28, %address.local397, !dbg !21
  %address.local399 = lshr i64 %address.local397, 6, !dbg !12
  %address.local400 = shl nuw nsw i64 %address.local399, 4, !dbg !21
  %address.local401 = add i64 %address.local398, %address.local400, !dbg !21
  %address.local402 = and i64 %address.local397, 240, !dbg !21
  %address.local403 = sub i64 %address.local401, %address.local402, !dbg !21
  %address.local404 = add i64 %address.local403, %29, !dbg !21
  %address.local405 = shl nuw nsw i64 %address.local404, 7, !dbg !21
  %address.local406 = add nuw nsw i64 %1640, %address.local405, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local408 = lshr i64 %address.local397, 2, !dbg !12
  %address.local409 = and i64 %address.local408, 12, !dbg !12
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local409, !dbg !21
  store half %1637, ptr addrspace(1) %address.local410, align 2, !dbg !21
  %address.local411 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local412 = or disjoint i64 %address.local409, 1, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local411, i64 %address.local412, !dbg !21
  store half %1636, ptr addrspace(1) %address.local413, align 2, !dbg !21
  %address.local414 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local415 = or disjoint i64 %address.local409, 2, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local414, i64 %address.local415, !dbg !21
  store half %1635, ptr addrspace(1) %address.local416, align 2, !dbg !21
  %address.local417 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local418 = or disjoint i64 %address.local409, 3, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local417, i64 %address.local418, !dbg !21
  store half %1634, ptr addrspace(1) %address.local419, align 2, !dbg !21
  %address.local420 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local421 = or disjoint i64 %address.local409, 16, !dbg !21
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local420, i64 %address.local421, !dbg !21
  store half %1622, ptr addrspace(1) %address.local422, align 2, !dbg !21
  %address.local423 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local424 = or disjoint i64 %address.local409, 17, !dbg !21
  %address.local425 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local423, i64 %address.local424, !dbg !21
  store half %1621, ptr addrspace(1) %address.local425, align 2, !dbg !21
  %address.local426 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local427 = or disjoint i64 %address.local409, 18, !dbg !21
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local426, i64 %address.local427, !dbg !21
  store half %1620, ptr addrspace(1) %address.local428, align 2, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local406, !dbg !21
  %address.local430 = or disjoint i64 %address.local409, 19, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local429, i64 %address.local430, !dbg !21
  store half %1619, ptr addrspace(1) %address.local431, align 2, !dbg !21
  %address.local432 = shl i64 %address.local404, 7, !dbg !21
  %address.local433 = add i64 %address.local432, 8192, !dbg !21
  %address.local434 = add nuw nsw i64 %1640, %address.local433, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local435, i64 %address.local409, !dbg !21
  store half %1607, ptr addrspace(1) %address.local436, align 2, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local437, i64 %address.local412, !dbg !21
  store half %1606, ptr addrspace(1) %address.local438, align 2, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local439, i64 %address.local415, !dbg !21
  store half %1605, ptr addrspace(1) %address.local440, align 2, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local441, i64 %address.local418, !dbg !21
  store half %1604, ptr addrspace(1) %address.local442, align 2, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local444 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local443, i64 %address.local421, !dbg !21
  store half %1592, ptr addrspace(1) %address.local444, align 2, !dbg !21
  %address.local445 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local446 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local445, i64 %address.local424, !dbg !21
  store half %1591, ptr addrspace(1) %address.local446, align 2, !dbg !21
  %address.local447 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local448 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local447, i64 %address.local427, !dbg !21
  store half %1590, ptr addrspace(1) %address.local448, align 2, !dbg !21
  %address.local449 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local434, !dbg !21
  %address.local450 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local449, i64 %address.local430, !dbg !21
  store half %1589, ptr addrspace(1) %address.local450, align 2, !dbg !21
  br label %1641, !dbg !21

1641:                                             ; preds = %1577, %1128
  br i1 true, label %1642, label %1680, !dbg !21

1642:                                             ; preds = %1641
  %1643 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !19
  %1644 = insertelement <4 x float> %1643, float %1553, i64 1, !dbg !19
  %1645 = insertelement <4 x float> %1644, float %1548, i64 2, !dbg !19
  %1646 = insertelement <4 x float> %1645, float %1554, i64 3, !dbg !19
  %row.reciprocal27 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !19
  %normalized28 = fmul <4 x float> %1646, %row.reciprocal27, !dbg !19
  %1647 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !20
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !20
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !20
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !20
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !20
  %1652 = insertelement <4 x float> poison, float %1438, i64 0, !dbg !19
  %1653 = insertelement <4 x float> %1652, float %1444, i64 1, !dbg !19
  %1654 = insertelement <4 x float> %1653, float %1439, i64 2, !dbg !19
  %1655 = insertelement <4 x float> %1654, float %1445, i64 3, !dbg !19
  %normalized29 = fmul <4 x float> %1655, %row.reciprocal27, !dbg !19
  %1656 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !20
  %1657 = extractelement <4 x half> %1656, i64 3, !dbg !20
  %1658 = extractelement <4 x half> %1656, i64 2, !dbg !20
  %1659 = extractelement <4 x half> %1656, i64 1, !dbg !20
  %1660 = extractelement <4 x half> %1656, i64 0, !dbg !20
  %1661 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !19
  %1662 = insertelement <4 x float> %1661, float %1335, i64 1, !dbg !19
  %1663 = insertelement <4 x float> %1662, float %1330, i64 2, !dbg !19
  %1664 = insertelement <4 x float> %1663, float %1336, i64 3, !dbg !19
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !19
  %normalized31 = fmul <4 x float> %1664, %row.reciprocal30, !dbg !19
  %1665 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !20
  %1666 = extractelement <4 x half> %1665, i64 3, !dbg !20
  %1667 = extractelement <4 x half> %1665, i64 2, !dbg !20
  %1668 = extractelement <4 x half> %1665, i64 1, !dbg !20
  %1669 = extractelement <4 x half> %1665, i64 0, !dbg !20
  %1670 = insertelement <4 x float> poison, float %1220, i64 0, !dbg !19
  %1671 = insertelement <4 x float> %1670, float %1226, i64 1, !dbg !19
  %1672 = insertelement <4 x float> %1671, float %1221, i64 2, !dbg !19
  %1673 = insertelement <4 x float> %1672, float %1227, i64 3, !dbg !19
  %normalized32 = fmul <4 x float> %1673, %row.reciprocal30, !dbg !19
  %1674 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !20
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !20
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !20
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !20
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !20
  %1679 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread451 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local452 = zext nneg i32 %address.thread451 to i64, !dbg !12
  %address.local453 = add i64 %28, %address.local452, !dbg !21
  %address.local454 = lshr i64 %address.local452, 6, !dbg !12
  %address.local455 = shl nuw nsw i64 %address.local454, 4, !dbg !21
  %address.local456 = add i64 %address.local453, %address.local455, !dbg !21
  %address.local457 = and i64 %address.local452, 240, !dbg !21
  %address.local458 = sub i64 %address.local456, %address.local457, !dbg !21
  %address.local459 = add i64 %address.local458, %29, !dbg !21
  %address.local460 = shl nuw nsw i64 %address.local459, 7, !dbg !21
  %address.local461 = add nuw nsw i64 %1679, %address.local460, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local463 = lshr i64 %address.local452, 2, !dbg !12
  %address.local464 = and i64 %address.local463, 12, !dbg !12
  %address.local465 = or disjoint i64 %address.local464, 32, !dbg !21
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local462, i64 %address.local465, !dbg !21
  store half %1678, ptr addrspace(1) %address.local466, align 2, !dbg !21
  %address.local467 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local468 = or disjoint i64 %address.local464, 33, !dbg !21
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local467, i64 %address.local468, !dbg !21
  store half %1677, ptr addrspace(1) %address.local469, align 2, !dbg !21
  %address.local470 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local471 = or disjoint i64 %address.local464, 34, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local470, i64 %address.local471, !dbg !21
  store half %1676, ptr addrspace(1) %address.local472, align 2, !dbg !21
  %address.local473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local474 = or disjoint i64 %address.local464, 35, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local473, i64 %address.local474, !dbg !21
  store half %1675, ptr addrspace(1) %address.local475, align 2, !dbg !21
  %address.local476 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local477 = or disjoint i64 %address.local464, 48, !dbg !21
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local476, i64 %address.local477, !dbg !21
  store half %1669, ptr addrspace(1) %address.local478, align 2, !dbg !21
  %address.local479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local480 = or disjoint i64 %address.local464, 49, !dbg !21
  %address.local481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local479, i64 %address.local480, !dbg !21
  store half %1668, ptr addrspace(1) %address.local481, align 2, !dbg !21
  %address.local482 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local483 = or disjoint i64 %address.local464, 50, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local482, i64 %address.local483, !dbg !21
  store half %1667, ptr addrspace(1) %address.local484, align 2, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local461, !dbg !21
  %address.local486 = or i64 %address.local463, 51, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local485, i64 %address.local486, !dbg !21
  store half %1666, ptr addrspace(1) %address.local487, align 2, !dbg !21
  %address.local488 = shl i64 %address.local459, 7, !dbg !21
  %address.local489 = add i64 %address.local488, 8192, !dbg !21
  %address.local490 = add nuw nsw i64 %1679, %address.local489, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local491, i64 %address.local465, !dbg !21
  store half %1660, ptr addrspace(1) %address.local492, align 2, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local493, i64 %address.local468, !dbg !21
  store half %1659, ptr addrspace(1) %address.local494, align 2, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local495, i64 %address.local471, !dbg !21
  store half %1658, ptr addrspace(1) %address.local496, align 2, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local497, i64 %address.local474, !dbg !21
  store half %1657, ptr addrspace(1) %address.local498, align 2, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local500 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local499, i64 %address.local477, !dbg !21
  store half %1651, ptr addrspace(1) %address.local500, align 2, !dbg !21
  %address.local501 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local502 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local501, i64 %address.local480, !dbg !21
  store half %1650, ptr addrspace(1) %address.local502, align 2, !dbg !21
  %address.local503 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local504 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local503, i64 %address.local483, !dbg !21
  store half %1649, ptr addrspace(1) %address.local504, align 2, !dbg !21
  %address.local505 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local490, !dbg !21
  %address.local506 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local505, i64 %address.local486, !dbg !21
  store half %1648, ptr addrspace(1) %address.local506, align 2, !dbg !21
  br label %1680, !dbg !21

1680:                                             ; preds = %1642, %1641
  br i1 true, label %1681, label %1719, !dbg !21

1681:                                             ; preds = %1680
  %1682 = insertelement <4 x float> poison, float %1519, i64 0, !dbg !19
  %1683 = insertelement <4 x float> %1682, float %1525, i64 1, !dbg !19
  %1684 = insertelement <4 x float> %1683, float %1520, i64 2, !dbg !19
  %1685 = insertelement <4 x float> %1684, float %1526, i64 3, !dbg !19
  %row.reciprocal33 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !19
  %normalized34 = fmul <4 x float> %1685, %row.reciprocal33, !dbg !19
  %1686 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !20
  %1687 = extractelement <4 x half> %1686, i64 3, !dbg !20
  %1688 = extractelement <4 x half> %1686, i64 2, !dbg !20
  %1689 = extractelement <4 x half> %1686, i64 1, !dbg !20
  %1690 = extractelement <4 x half> %1686, i64 0, !dbg !20
  %1691 = insertelement <4 x float> poison, float %1409, i64 0, !dbg !19
  %1692 = insertelement <4 x float> %1691, float %1415, i64 1, !dbg !19
  %1693 = insertelement <4 x float> %1692, float %1410, i64 2, !dbg !19
  %1694 = insertelement <4 x float> %1693, float %1416, i64 3, !dbg !19
  %normalized35 = fmul <4 x float> %1694, %row.reciprocal33, !dbg !19
  %1695 = fptrunc <4 x float> %normalized35 to <4 x half>, !dbg !20
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !20
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !20
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !20
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !20
  %1700 = insertelement <4 x float> poison, float %1301, i64 0, !dbg !19
  %1701 = insertelement <4 x float> %1700, float %1307, i64 1, !dbg !19
  %1702 = insertelement <4 x float> %1701, float %1302, i64 2, !dbg !19
  %1703 = insertelement <4 x float> %1702, float %1308, i64 3, !dbg !19
  %row.reciprocal36 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !19
  %normalized37 = fmul <4 x float> %1703, %row.reciprocal36, !dbg !19
  %1704 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !20
  %1705 = extractelement <4 x half> %1704, i64 3, !dbg !20
  %1706 = extractelement <4 x half> %1704, i64 2, !dbg !20
  %1707 = extractelement <4 x half> %1704, i64 1, !dbg !20
  %1708 = extractelement <4 x half> %1704, i64 0, !dbg !20
  %1709 = insertelement <4 x float> poison, float %1191, i64 0, !dbg !19
  %1710 = insertelement <4 x float> %1709, float %1197, i64 1, !dbg !19
  %1711 = insertelement <4 x float> %1710, float %1192, i64 2, !dbg !19
  %1712 = insertelement <4 x float> %1711, float %1198, i64 3, !dbg !19
  %normalized38 = fmul <4 x float> %1712, %row.reciprocal36, !dbg !19
  %1713 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !20
  %1714 = extractelement <4 x half> %1713, i64 3, !dbg !20
  %1715 = extractelement <4 x half> %1713, i64 2, !dbg !20
  %1716 = extractelement <4 x half> %1713, i64 1, !dbg !20
  %1717 = extractelement <4 x half> %1713, i64 0, !dbg !20
  %1718 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread507 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local508 = zext nneg i32 %address.thread507 to i64, !dbg !12
  %address.local509 = add i64 %28, %address.local508, !dbg !21
  %address.local510 = lshr i64 %address.local508, 6, !dbg !12
  %address.local511 = shl nuw nsw i64 %address.local510, 4, !dbg !21
  %address.local512 = add i64 %address.local509, %address.local511, !dbg !21
  %address.local513 = and i64 %address.local508, 240, !dbg !21
  %address.local514 = sub i64 %address.local512, %address.local513, !dbg !21
  %address.local515 = add i64 %address.local514, %29, !dbg !21
  %address.local516 = shl nuw nsw i64 %address.local515, 7, !dbg !21
  %address.local517 = add nuw nsw i64 %1718, %address.local516, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local519 = lshr i64 %address.local508, 2, !dbg !12
  %address.local520 = and i64 %address.local519, 12, !dbg !12
  %address.local521 = or disjoint i64 %address.local520, 64, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local518, i64 %address.local521, !dbg !21
  store half %1717, ptr addrspace(1) %address.local522, align 2, !dbg !21
  %address.local523 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local524 = or disjoint i64 %address.local520, 65, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local523, i64 %address.local524, !dbg !21
  store half %1716, ptr addrspace(1) %address.local525, align 2, !dbg !21
  %address.local526 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local527 = or disjoint i64 %address.local520, 66, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local526, i64 %address.local527, !dbg !21
  store half %1715, ptr addrspace(1) %address.local528, align 2, !dbg !21
  %address.local529 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local530 = or disjoint i64 %address.local520, 67, !dbg !21
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local529, i64 %address.local530, !dbg !21
  store half %1714, ptr addrspace(1) %address.local531, align 2, !dbg !21
  %address.local532 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local533 = or disjoint i64 %address.local520, 80, !dbg !21
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local532, i64 %address.local533, !dbg !21
  store half %1708, ptr addrspace(1) %address.local534, align 2, !dbg !21
  %address.local535 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local536 = or disjoint i64 %address.local520, 81, !dbg !21
  %address.local537 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local535, i64 %address.local536, !dbg !21
  store half %1707, ptr addrspace(1) %address.local537, align 2, !dbg !21
  %address.local538 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local539 = or disjoint i64 %address.local520, 82, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local538, i64 %address.local539, !dbg !21
  store half %1706, ptr addrspace(1) %address.local540, align 2, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local517, !dbg !21
  %address.local542 = or disjoint i64 %address.local520, 83, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local541, i64 %address.local542, !dbg !21
  store half %1705, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = shl i64 %address.local515, 7, !dbg !21
  %address.local545 = add i64 %address.local544, 8192, !dbg !21
  %address.local546 = add nuw nsw i64 %1718, %address.local545, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local547, i64 %address.local521, !dbg !21
  store half %1699, ptr addrspace(1) %address.local548, align 2, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local549, i64 %address.local524, !dbg !21
  store half %1698, ptr addrspace(1) %address.local550, align 2, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local551, i64 %address.local527, !dbg !21
  store half %1697, ptr addrspace(1) %address.local552, align 2, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local553, i64 %address.local530, !dbg !21
  store half %1696, ptr addrspace(1) %address.local554, align 2, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local556 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local555, i64 %address.local533, !dbg !21
  store half %1690, ptr addrspace(1) %address.local556, align 2, !dbg !21
  %address.local557 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local558 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local557, i64 %address.local536, !dbg !21
  store half %1689, ptr addrspace(1) %address.local558, align 2, !dbg !21
  %address.local559 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local560 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local559, i64 %address.local539, !dbg !21
  store half %1688, ptr addrspace(1) %address.local560, align 2, !dbg !21
  %address.local561 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local546, !dbg !21
  %address.local562 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local561, i64 %address.local542, !dbg !21
  store half %1687, ptr addrspace(1) %address.local562, align 2, !dbg !21
  br label %1719, !dbg !21

1719:                                             ; preds = %1681, %1680
  br i1 true, label %1720, label %1758, !dbg !21

1720:                                             ; preds = %1719
  %1721 = insertelement <4 x float> poison, float %1491, i64 0, !dbg !19
  %1722 = insertelement <4 x float> %1721, float %1497, i64 1, !dbg !19
  %1723 = insertelement <4 x float> %1722, float %1492, i64 2, !dbg !19
  %1724 = insertelement <4 x float> %1723, float %1498, i64 3, !dbg !19
  %row.reciprocal39 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1470, !dbg !19
  %normalized40 = fmul <4 x float> %1724, %row.reciprocal39, !dbg !19
  %1725 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !20
  %1726 = extractelement <4 x half> %1725, i64 3, !dbg !20
  %1727 = extractelement <4 x half> %1725, i64 2, !dbg !20
  %1728 = extractelement <4 x half> %1725, i64 1, !dbg !20
  %1729 = extractelement <4 x half> %1725, i64 0, !dbg !20
  %1730 = insertelement <4 x float> poison, float %1380, i64 0, !dbg !19
  %1731 = insertelement <4 x float> %1730, float %1386, i64 1, !dbg !19
  %1732 = insertelement <4 x float> %1731, float %1381, i64 2, !dbg !19
  %1733 = insertelement <4 x float> %1732, float %1387, i64 3, !dbg !19
  %normalized41 = fmul <4 x float> %1733, %row.reciprocal39, !dbg !19
  %1734 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !20
  %1735 = extractelement <4 x half> %1734, i64 3, !dbg !20
  %1736 = extractelement <4 x half> %1734, i64 2, !dbg !20
  %1737 = extractelement <4 x half> %1734, i64 1, !dbg !20
  %1738 = extractelement <4 x half> %1734, i64 0, !dbg !20
  %1739 = insertelement <4 x float> poison, float %1273, i64 0, !dbg !19
  %1740 = insertelement <4 x float> %1739, float %1279, i64 1, !dbg !19
  %1741 = insertelement <4 x float> %1740, float %1274, i64 2, !dbg !19
  %1742 = insertelement <4 x float> %1741, float %1280, i64 3, !dbg !19
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1252, !dbg !19
  %normalized43 = fmul <4 x float> %1742, %row.reciprocal42, !dbg !19
  %1743 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !20
  %1744 = extractelement <4 x half> %1743, i64 3, !dbg !20
  %1745 = extractelement <4 x half> %1743, i64 2, !dbg !20
  %1746 = extractelement <4 x half> %1743, i64 1, !dbg !20
  %1747 = extractelement <4 x half> %1743, i64 0, !dbg !20
  %1748 = insertelement <4 x float> poison, float %1162, i64 0, !dbg !19
  %1749 = insertelement <4 x float> %1748, float %1168, i64 1, !dbg !19
  %1750 = insertelement <4 x float> %1749, float %1163, i64 2, !dbg !19
  %1751 = insertelement <4 x float> %1750, float %1169, i64 3, !dbg !19
  %normalized44 = fmul <4 x float> %1751, %row.reciprocal42, !dbg !19
  %1752 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !20
  %1753 = extractelement <4 x half> %1752, i64 3, !dbg !20
  %1754 = extractelement <4 x half> %1752, i64 2, !dbg !20
  %1755 = extractelement <4 x half> %1752, i64 1, !dbg !20
  %1756 = extractelement <4 x half> %1752, i64 0, !dbg !20
  %1757 = shl nuw nsw i64 %27, 19, !dbg !21
  %address.thread563 = call i32 asm sideeffect "", "=v,0"(i32 %34) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local564 = zext nneg i32 %address.thread563 to i64, !dbg !12
  %address.local565 = add i64 %28, %address.local564, !dbg !21
  %address.local566 = lshr i64 %address.local564, 6, !dbg !12
  %address.local567 = shl nuw nsw i64 %address.local566, 4, !dbg !21
  %address.local568 = add i64 %address.local565, %address.local567, !dbg !21
  %address.local569 = and i64 %address.local564, 240, !dbg !21
  %address.local570 = sub i64 %address.local568, %address.local569, !dbg !21
  %address.local571 = add i64 %address.local570, %29, !dbg !21
  %address.local572 = shl nuw nsw i64 %address.local571, 7, !dbg !21
  %address.local573 = add nuw nsw i64 %1757, %address.local572, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local575 = lshr i64 %address.local564, 2, !dbg !12
  %address.local576 = and i64 %address.local575, 12, !dbg !12
  %address.local577 = or disjoint i64 %address.local576, 96, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local574, i64 %address.local577, !dbg !21
  store half %1756, ptr addrspace(1) %address.local578, align 2, !dbg !21
  %address.local579 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local580 = or disjoint i64 %address.local576, 97, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local579, i64 %address.local580, !dbg !21
  store half %1755, ptr addrspace(1) %address.local581, align 2, !dbg !21
  %address.local582 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local583 = or disjoint i64 %address.local576, 98, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local582, i64 %address.local583, !dbg !21
  store half %1754, ptr addrspace(1) %address.local584, align 2, !dbg !21
  %address.local585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local586 = or disjoint i64 %address.local576, 99, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local585, i64 %address.local586, !dbg !21
  store half %1753, ptr addrspace(1) %address.local587, align 2, !dbg !21
  %address.local588 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local589 = or disjoint i64 %address.local576, 112, !dbg !21
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local588, i64 %address.local589, !dbg !21
  store half %1747, ptr addrspace(1) %address.local590, align 2, !dbg !21
  %address.local591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local592 = or disjoint i64 %address.local576, 113, !dbg !21
  %address.local593 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local591, i64 %address.local592, !dbg !21
  store half %1746, ptr addrspace(1) %address.local593, align 2, !dbg !21
  %address.local594 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local595 = or disjoint i64 %address.local576, 114, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local594, i64 %address.local595, !dbg !21
  store half %1745, ptr addrspace(1) %address.local596, align 2, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local573, !dbg !21
  %address.local598 = or i64 %address.local575, 115, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local597, i64 %address.local598, !dbg !21
  store half %1744, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = shl i64 %address.local571, 7, !dbg !21
  %address.local601 = add i64 %address.local600, 8192, !dbg !21
  %address.local602 = add nuw nsw i64 %1757, %address.local601, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local603, i64 %address.local577, !dbg !21
  store half %1738, ptr addrspace(1) %address.local604, align 2, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local605, i64 %address.local580, !dbg !21
  store half %1737, ptr addrspace(1) %address.local606, align 2, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local607, i64 %address.local583, !dbg !21
  store half %1736, ptr addrspace(1) %address.local608, align 2, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local609, i64 %address.local586, !dbg !21
  store half %1735, ptr addrspace(1) %address.local610, align 2, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local612 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local611, i64 %address.local589, !dbg !21
  store half %1729, ptr addrspace(1) %address.local612, align 2, !dbg !21
  %address.local613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local613, i64 %address.local592, !dbg !21
  store half %1728, ptr addrspace(1) %address.local614, align 2, !dbg !21
  %address.local615 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local616 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local595, !dbg !21
  store half %1727, ptr addrspace(1) %address.local616, align 2, !dbg !21
  %address.local617 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local602, !dbg !21
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local617, i64 %address.local598, !dbg !21
  store half %1726, ptr addrspace(1) %address.local618, align 2, !dbg !21
  br label %1758, !dbg !21

1758:                                             ; preds = %1720, %1719
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
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0004-test_input_1x32x4096x128-128-32-32-online/group8-reverse-k128-swizzle")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!9 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 70, column: 22, scope: !4)
!12 = !DILocation(line: 72, column: 24, scope: !4)
!13 = !DILocation(line: 63, column: 17, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 73, column: 26, scope: !4)
!17 = !DILocation(line: 71, column: 22, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
!22 = !DILocation(line: 74, column: 23, scope: !4)
!23 = !DILocation(line: 79, column: 24, scope: !4)
!24 = !DILocation(line: 80, column: 23, scope: !4)
!25 = !DILocation(line: 81, column: 26, scope: !4)
!26 = !DILocation(line: 88, column: 21, scope: !4)
!27 = !DILocation(line: 89, column: 20, scope: !4)
!28 = !DILocation(line: 90, column: 22, scope: !4)
!29 = !DILocation(line: 91, column: 23, scope: !4)
!30 = !DILocation(line: 92, column: 20, scope: !4)
!31 = !DILocation(line: 93, column: 22, scope: !4)
!32 = !DILocation(line: 94, column: 22, scope: !4)
!33 = !DILocation(line: 95, column: 20, scope: !4)
!34 = !DILocation(line: 96, column: 16, scope: !4)
!35 = !DILocation(line: 103, column: 22, scope: !4)
!37 = !{i32 256, i32 1, i32 1}
!38 = !{i32 2, !"Debug Info Version", i32 3}
!39 = !{i32 0, i32 256}
