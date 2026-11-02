; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/online-baseline/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/online-baseline/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !37 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %8, 6, !dbg !5
  %10 = add nsw i64 %9, %6, !dbg !5
  %11 = lshr i64 %10, 5, !dbg !5
  %12 = and i64 %11, 576460752303423484, !dbg !5
  %13 = and i64 %6, 3, !dbg !5
  %14 = or disjoint i64 %12, %13, !dbg !5
  %15 = sub nsw i64 63, %14, !dbg !5
  %16 = lshr i64 %10, 2, !dbg !5
  %17 = icmp ugt i64 %10, 2047, !dbg !8
  %18 = add nsw i64 %14, -64, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 64, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %16, %22, !dbg !8
  %24 = and i64 %23, 31, !dbg !8
  %25 = shl nsw i64 %15, 6, !dbg !8
  %26 = mul nsw i64 %22, -4096, !dbg !8
  %27 = add i64 %25, %26, !dbg !8
  %28 = shl nuw nsw i64 %24, 19, !dbg !8
  %29 = shl nsw i64 %27, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !39
  %32 = zext nneg i32 %31 to i64, !dbg !8
  %33 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !8, !range !39, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local277 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local278 = and i64 %address.local277, 1536, !dbg !8
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !8
  %address.local280 = ptrtoint ptr addrspace(3) %address.local279 to i64, !dbg !8
  %address.local281 = trunc i64 %address.local280 to i32, !dbg !8
  %34 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local281), !dbg !8
  %35 = zext i32 %34 to i64, !dbg !8
  %36 = inttoptr i64 %35 to ptr addrspace(3), !dbg !8
  %address.local282 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local283 = and i64 %address.local282, 8064, !dbg !8
  %address.local284 = lshr i64 %address.local, 3, !dbg !8
  %address.local285 = and i64 %address.local284, 24, !dbg !8
  %address.local286 = or disjoint i64 %address.local283, %address.local285, !dbg !8
  %address.local287 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %address.local286, !dbg !8
  %37 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %38 = ptrtoint ptr addrspace(1) %address.local287 to i64, !dbg !8
  %39 = sub i64 %38, %37, !dbg !8
  %dma.buffer.offset1214 = trunc i64 %39 to i32, !dbg !8
  %40 = trunc i64 %37 to i32, !dbg !8
  %41 = insertelement <4 x i32> poison, i32 %40, i64 0, !dbg !8
  %42 = lshr i64 %37, 32, !dbg !8
  %43 = trunc i64 %42 to i32, !dbg !8
  %44 = insertelement <4 x i32> %41, i32 %43, i64 1, !dbg !8
  %45 = insertelement <4 x i32> %44, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1215 = insertelement <4 x i32> %45, i32 131072, i64 3, !dbg !8
  %46 = ptrtoint ptr addrspace(3) %36 to i32, !dbg !8
  %dma.buffer.lds1216 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %47 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1216, i32 %dma.buffer.offset1214, <4 x i32> %dma.buffer.rsrc1215) #10, !dbg !8
  %48 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %49 = sext i32 %48 to i64, !dbg !5
  %50 = shl i64 %15, 6, !dbg !10
  %51 = lshr i64 %32, 6, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %52 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %53 = getelementptr i8, ptr addrspace(1) %52, i64 64, !dbg !8
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local278, !dbg !8
  %address.local289 = ptrtoint ptr addrspace(3) %address.local288 to i64, !dbg !8
  %address.local290 = trunc i64 %address.local289 to i32, !dbg !8
  %54 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !8
  %55 = zext i32 %54 to i64, !dbg !8
  %56 = inttoptr i64 %55 to ptr addrspace(3), !dbg !8
  %address.local291 = getelementptr [2 x i8], ptr addrspace(1) %53, i64 %address.local286, !dbg !8
  %57 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %58 = ptrtoint ptr addrspace(1) %address.local291 to i64, !dbg !8
  %59 = sub i64 %58, %57, !dbg !8
  %dma.buffer.offset1211 = trunc i64 %59 to i32, !dbg !8
  %60 = trunc i64 %57 to i32, !dbg !8
  %61 = insertelement <4 x i32> poison, i32 %60, i64 0, !dbg !8
  %62 = lshr i64 %57, 32, !dbg !8
  %63 = trunc i64 %62 to i32, !dbg !8
  %64 = insertelement <4 x i32> %61, i32 %63, i64 1, !dbg !8
  %65 = insertelement <4 x i32> %64, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1212 = insertelement <4 x i32> %65, i32 131072, i64 3, !dbg !8
  %66 = ptrtoint ptr addrspace(3) %56 to i32, !dbg !8
  %dma.buffer.lds1213 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66), !dbg !8
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1213, i32 %dma.buffer.offset1211, <4 x i32> %dma.buffer.rsrc1212) #10, !dbg !8
  %address.local292 = lshr i32 %address.thread, 2, !dbg !11
  %address.local293 = and i32 %address.local292, 4, !dbg !11
  %address.local294 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local295 = and i32 %address.local294, 384, !dbg !11
  %address.local296 = or disjoint i32 %address.local293, %address.local295, !dbg !11
  %address.local297 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local298 = and i32 %address.local297, 120, !dbg !11
  %address.local299 = or disjoint i32 %address.local296, %address.local298, !dbg !11
  %address.local300 = shl nuw nsw i32 %address.thread, 4, !dbg !11
  %address.local301 = and i32 %address.local300, 512, !dbg !11
  %address.local302 = or disjoint i32 %address.local299, %address.local301, !dbg !11
  %address.local303 = zext nneg i32 %address.local302 to i64, !dbg !11
  %address.local304 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local303, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local304, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local304, i64 2048, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !11
  %71 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %72 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %73 = getelementptr i8, ptr addrspace(1) %72, i64 128, !dbg !8
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local281), !dbg !8
  %75 = zext i32 %74 to i64, !dbg !8
  %76 = inttoptr i64 %75 to ptr addrspace(3), !dbg !8
  %address.local319 = getelementptr [2 x i8], ptr addrspace(1) %73, i64 %address.local286, !dbg !8
  %77 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %78 = ptrtoint ptr addrspace(1) %address.local319 to i64, !dbg !8
  %79 = sub i64 %78, %77, !dbg !8
  %dma.buffer.offset1208 = trunc i64 %79 to i32, !dbg !8
  %80 = trunc i64 %77 to i32, !dbg !8
  %81 = insertelement <4 x i32> poison, i32 %80, i64 0, !dbg !8
  %82 = lshr i64 %77, 32, !dbg !8
  %83 = trunc i64 %82 to i32, !dbg !8
  %84 = insertelement <4 x i32> %81, i32 %83, i64 1, !dbg !8
  %85 = insertelement <4 x i32> %84, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1209 = insertelement <4 x i32> %85, i32 131072, i64 3, !dbg !8
  %86 = ptrtoint ptr addrspace(3) %76 to i32, !dbg !8
  %dma.buffer.lds1210 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %86), !dbg !8
  %87 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1210, i32 %dma.buffer.offset1208, <4 x i32> %dma.buffer.rsrc1209) #10, !dbg !8
  %lds.const.gep1095 = getelementptr i8, ptr addrspace(3) %address.local304, i64 4096, !dbg !11
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1095, align 8, !dbg !11
  %lds.const.gep1096 = getelementptr i8, ptr addrspace(3) %address.local304, i64 6144, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep1096, align 8, !dbg !11
  %90 = fmul <4 x half> %88, splat (half 0xH3015), !dbg !11
  %91 = fmul <4 x half> %89, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %92 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %30, !dbg !8
  %93 = getelementptr i8, ptr addrspace(1) %92, i64 192, !dbg !8
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !8
  %95 = zext i32 %94 to i64, !dbg !8
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !8
  %address.local322 = getelementptr [2 x i8], ptr addrspace(1) %93, i64 %address.local286, !dbg !8
  %97 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %98 = ptrtoint ptr addrspace(1) %address.local322 to i64, !dbg !8
  %99 = sub i64 %98, %97, !dbg !8
  %dma.buffer.offset1205 = trunc i64 %99 to i32, !dbg !8
  %100 = trunc i64 %97 to i32, !dbg !8
  %101 = insertelement <4 x i32> poison, i32 %100, i64 0, !dbg !8
  %102 = lshr i64 %97, 32, !dbg !8
  %103 = trunc i64 %102 to i32, !dbg !8
  %104 = insertelement <4 x i32> %101, i32 %103, i64 1, !dbg !8
  %105 = insertelement <4 x i32> %104, i32 33554432, i64 2, !dbg !8
  %dma.buffer.rsrc1206 = insertelement <4 x i32> %105, i32 131072, i64 3, !dbg !8
  %106 = ptrtoint ptr addrspace(3) %96 to i32, !dbg !8
  %dma.buffer.lds1207 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !8
  %107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1207, i32 %dma.buffer.offset1205, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !8
  %108 = load <4 x half>, ptr addrspace(3) %address.local304, align 8, !dbg !11
  %lds.const.gep1097 = getelementptr i8, ptr addrspace(3) %address.local304, i64 2048, !dbg !11
  %109 = load <4 x half>, ptr addrspace(3) %lds.const.gep1097, align 8, !dbg !11
  %110 = fmul <4 x half> %108, splat (half 0xH3015), !dbg !11
  %111 = fmul <4 x half> %109, splat (half 0xH3015), !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %lds.const.gep1098 = getelementptr i8, ptr addrspace(3) %address.local304, i64 4096, !dbg !11
  %112 = load <4 x half>, ptr addrspace(3) %lds.const.gep1098, align 8, !dbg !11
  %lds.const.gep1099 = getelementptr i8, ptr addrspace(3) %address.local304, i64 6144, !dbg !11
  %113 = load <4 x half>, ptr addrspace(3) %lds.const.gep1099, align 8, !dbg !11
  %114 = fmul <4 x half> %112, splat (half 0xH3015), !dbg !11
  %115 = fmul <4 x half> %113, splat (half 0xH3015), !dbg !11
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %116 = or disjoint i64 %50, 63, !dbg !14
  %117 = add i64 %50, 64, !dbg !14
  %118 = icmp uge i64 %117, %116, !dbg !14
  %119 = icmp ugt i64 %10, 2047, !dbg !7
  %120 = select i1 %119, i64 %18, i64 %15, !dbg !7
  %121 = sdiv i64 %120, 2, !dbg !7
  %122 = sub nsw i64 0, %121, !dbg !7
  %123 = add nsw i64 %121, 1, !dbg !7
  %124 = select i1 %119, i64 %122, i64 %123, !dbg !7
  %125 = shl i64 %16, 7, !dbg !17
  %126 = and i64 %125, 3968, !dbg !17
  %127 = shl nsw i64 %49, 12, !dbg !17
  %128 = or disjoint i64 %126, %127, !dbg !17
  %129 = lshr i64 %32, 2, !dbg !16
  %130 = shl nuw nsw i64 %32, 2, !dbg !16
  %131 = mul nsw i64 %129, -16, !dbg !16
  %132 = add nsw i64 %130, %131, !dbg !16
  %133 = and i32 %31, 1, !dbg !16
  %index.compact235 = zext nneg i32 %133 to i64, !dbg !16
  %134 = or disjoint i64 %132, %index.compact235, !dbg !16
  %135 = add nuw nsw i64 %134, 1, !dbg !16
  %136 = sdiv i64 %135, 2, !dbg !16
  br label %137, !dbg !7

137:                                              ; preds = %592, %4
  %138 = phi i64 [ %1154, %592 ], [ 0, %4 ], !dbg !7
  %139 = phi <4 x float> [ %929, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %140 = phi <4 x float> [ %945, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %141 = phi <8 x float> [ %1150, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %142 = phi <1 x float> [ %863, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %143 = phi <1 x float> [ %694, %592 ], [ splat (float 0xFFF0000000000000), %4 ], !dbg !7
  %144 = phi <4 x float> [ %1007, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %145 = phi <4 x float> [ %1023, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %146 = phi <8 x float> [ %1151, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %147 = phi <4 x float> [ %1085, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %148 = phi <4 x float> [ %1101, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %149 = phi <8 x float> [ %1152, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %150 = phi <4 x float> [ %1133, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %151 = phi <4 x float> [ %1149, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %152 = phi <8 x float> [ %1153, %592 ], [ zeroinitializer, %4 ], !dbg !7
  %153 = icmp slt i64 %138, %124, !dbg !7
  br i1 %153, label %154, label %1155, !dbg !7

154:                                              ; preds = %137
  %155 = shl nsw i64 %138, 7, !dbg !7
  %156 = lshr i64 %138, 12, !dbg !17
  %157 = add nuw nsw i64 %156, %16, !dbg !17
  %158 = lshr i64 %138, 5, !dbg !17
  %159 = and i64 %158, 127, !dbg !17
  %160 = shl i64 %157, 19, !dbg !17
  %161 = and i64 %160, 16252928, !dbg !17
  %162 = shl i64 %158, 19, !dbg !17
  %163 = shl i64 %138, 14, !dbg !17
  %164 = sub i64 %163, %162, !dbg !17
  %165 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %161, !dbg !17
  %166 = getelementptr [2 x i8], ptr addrspace(1) %165, i64 %159, !dbg !17
  %167 = getelementptr [2 x i8], ptr addrspace(1) %166, i64 %164, !dbg !17
  %address.thread323 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !17, !range !39, !frisk.register_lifetimes !1
  %address.local324 = zext nneg i32 %address.thread323 to i64, !dbg !8
  %address.local325 = shl nuw nsw i64 %address.local324, 3, !dbg !8
  %address.local326 = and i64 %address.local325, 1536, !dbg !17
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local326, !dbg !17
  %address.local328 = ptrtoint ptr addrspace(3) %address.local327 to i64, !dbg !17
  %address.local329 = trunc i64 %address.local328 to i32, !dbg !17
  %168 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local329), !dbg !17
  %169 = zext i32 %168 to i64, !dbg !17
  %170 = inttoptr i64 %169 to ptr addrspace(3), !dbg !17
  %address.local330 = lshr i64 %address.local324, 4, !dbg !17
  %address.local331 = and i64 %address.local330, 8, !dbg !17
  %address.local332 = and i64 %address.local324, 123, !dbg !17
  %address.local333 = lshr i64 %address.local324, 2, !dbg !17
  %address.local334 = and i64 %address.local333, 4, !dbg !17
  %address.local335 = add nuw nsw i64 %address.local334, %address.local324, !dbg !17
  %address.local336 = and i64 %address.local335, 4, !dbg !17
  %address.local337 = or disjoint i64 %address.local332, %address.local336, !dbg !17
  %address.local338 = shl nuw nsw i64 %address.local337, 7, !dbg !17
  %address.local339 = or disjoint i64 %address.local331, %address.local338, !dbg !17
  %address.local340 = getelementptr [2 x i8], ptr addrspace(1) %167, i64 %address.local339, !dbg !17
  %171 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %172 = ptrtoint ptr addrspace(1) %address.local340 to i64, !dbg !17
  %173 = sub i64 %172, %171, !dbg !17
  %dma.buffer.offset1202 = trunc i64 %173 to i32, !dbg !17
  %174 = trunc i64 %171 to i32, !dbg !17
  %175 = insertelement <4 x i32> poison, i32 %174, i64 0, !dbg !17
  %176 = lshr i64 %171, 32, !dbg !17
  %177 = trunc i64 %176 to i32, !dbg !17
  %178 = insertelement <4 x i32> %175, i32 %177, i64 1, !dbg !17
  %179 = insertelement <4 x i32> %178, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1203 = insertelement <4 x i32> %179, i32 131072, i64 3, !dbg !17
  %180 = ptrtoint ptr addrspace(3) %170 to i32, !dbg !17
  %dma.buffer.lds1204 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %180), !dbg !17
  %181 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1204, i32 %dma.buffer.offset1202, <4 x i32> %dma.buffer.rsrc1203) #10, !dbg !17
  %address.local341 = or disjoint i64 %address.local324, 256, !dbg !17
  %address.local342 = shl nuw nsw i64 %address.local341, 3, !dbg !17
  %address.local343 = and i64 %address.local342, 3584, !dbg !17
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local343, !dbg !17
  %address.local345 = ptrtoint ptr addrspace(3) %address.local344 to i64, !dbg !17
  %address.local346 = trunc i64 %address.local345 to i32, !dbg !17
  %182 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local346), !dbg !17
  %183 = zext i32 %182 to i64, !dbg !17
  %184 = inttoptr i64 %183 to ptr addrspace(3), !dbg !17
  %address.local347 = lshr i64 %address.local341, 4, !dbg !17
  %address.local348 = and i64 %address.local347, 24, !dbg !17
  %address.local349 = and i64 %address.local324, 123, !dbg !17
  %address.local350 = lshr i64 %address.local324, 2, !dbg !17
  %address.local351 = and i64 %address.local350, 4, !dbg !17
  %address.local352 = add nuw nsw i64 %address.local351, %address.local324, !dbg !17
  %address.local353 = and i64 %address.local352, 4, !dbg !17
  %address.local354 = or disjoint i64 %address.local349, %address.local353, !dbg !17
  %address.local355 = shl nuw nsw i64 %address.local354, 7, !dbg !17
  %address.local356 = or disjoint i64 %address.local348, %address.local355, !dbg !17
  %address.local357 = getelementptr [2 x i8], ptr addrspace(1) %167, i64 %address.local356, !dbg !17
  %185 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %186 = ptrtoint ptr addrspace(1) %address.local357 to i64, !dbg !17
  %187 = sub i64 %186, %185, !dbg !17
  %dma.buffer.offset1199 = trunc i64 %187 to i32, !dbg !17
  %188 = trunc i64 %185 to i32, !dbg !17
  %189 = insertelement <4 x i32> poison, i32 %188, i64 0, !dbg !17
  %190 = lshr i64 %185, 32, !dbg !17
  %191 = trunc i64 %190 to i32, !dbg !17
  %192 = insertelement <4 x i32> %189, i32 %191, i64 1, !dbg !17
  %193 = insertelement <4 x i32> %192, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1200 = insertelement <4 x i32> %193, i32 131072, i64 3, !dbg !17
  %194 = ptrtoint ptr addrspace(3) %184 to i32, !dbg !17
  %dma.buffer.lds1201 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %194), !dbg !17
  %195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1201, i32 %dma.buffer.offset1199, <4 x i32> %dma.buffer.rsrc1200) #10, !dbg !17
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %196 = add nuw i64 %138, 1024, !dbg !17
  %197 = icmp slt i64 %196, 0, !dbg !17
  %198 = sub nuw i64 -1025, %138, !dbg !17
  %199 = select i1 %197, i64 %198, i64 %196, !dbg !17
  %200 = sdiv i64 %199, 4096, !dbg !17
  %.lobit37 = ashr i64 %196, 63, !dbg !17
  %201 = xor i64 %200, %.lobit37, !dbg !17
  %202 = add nsw i64 %201, %16, !dbg !17
  %203 = add nsw i64 %158, %128, !dbg !17
  %204 = add nsw i64 %203, 32, !dbg !17
  %205 = icmp slt i64 %203, -32, !dbg !17
  %206 = sub nsw i64 -33, %203, !dbg !17
  %207 = select i1 %205, i64 %206, i64 %204, !dbg !17
  %208 = sdiv i64 %207, 128, !dbg !17
  %209 = sext i1 %205 to i64, !dbg !17
  %210 = xor i64 %208, %209, !dbg !17
  %211 = shl nsw i64 %210, 7, !dbg !17
  %212 = sub i64 %128, %211, !dbg !17
  %213 = shl i64 %202, 19, !dbg !17
  %214 = and i64 %213, 16252928, !dbg !17
  %215 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %214, !dbg !17
  %216 = getelementptr [2 x i8], ptr addrspace(1) %215, i64 %158, !dbg !17
  %217 = getelementptr [2 x i8], ptr addrspace(1) %216, i64 %212, !dbg !17
  %218 = getelementptr i8, ptr addrspace(1) %217, i64 64, !dbg !17
  %219 = getelementptr [2 x i8], ptr addrspace(1) %218, i64 %164, !dbg !17
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local326, !dbg !17
  %address.local359 = ptrtoint ptr addrspace(3) %address.local358 to i64, !dbg !17
  %address.local360 = trunc i64 %address.local359 to i32, !dbg !17
  %220 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local360), !dbg !17
  %221 = zext i32 %220 to i64, !dbg !17
  %222 = inttoptr i64 %221 to ptr addrspace(3), !dbg !17
  %address.local361 = getelementptr [2 x i8], ptr addrspace(1) %219, i64 %address.local339, !dbg !17
  %223 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %224 = ptrtoint ptr addrspace(1) %address.local361 to i64, !dbg !17
  %225 = sub i64 %224, %223, !dbg !17
  %dma.buffer.offset1196 = trunc i64 %225 to i32, !dbg !17
  %226 = trunc i64 %223 to i32, !dbg !17
  %227 = insertelement <4 x i32> poison, i32 %226, i64 0, !dbg !17
  %228 = lshr i64 %223, 32, !dbg !17
  %229 = trunc i64 %228 to i32, !dbg !17
  %230 = insertelement <4 x i32> %227, i32 %229, i64 1, !dbg !17
  %231 = insertelement <4 x i32> %230, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1197 = insertelement <4 x i32> %231, i32 131072, i64 3, !dbg !17
  %232 = ptrtoint ptr addrspace(3) %222 to i32, !dbg !17
  %dma.buffer.lds1198 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %232), !dbg !17
  %233 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1198, i32 %dma.buffer.offset1196, <4 x i32> %dma.buffer.rsrc1197) #10, !dbg !17
  %address.local362 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local343, !dbg !17
  %address.local363 = ptrtoint ptr addrspace(3) %address.local362 to i64, !dbg !17
  %address.local364 = trunc i64 %address.local363 to i32, !dbg !17
  %234 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !17
  %235 = zext i32 %234 to i64, !dbg !17
  %236 = inttoptr i64 %235 to ptr addrspace(3), !dbg !17
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %219, i64 %address.local356, !dbg !17
  %237 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %238 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !17
  %239 = sub i64 %238, %237, !dbg !17
  %dma.buffer.offset1193 = trunc i64 %239 to i32, !dbg !17
  %240 = trunc i64 %237 to i32, !dbg !17
  %241 = insertelement <4 x i32> poison, i32 %240, i64 0, !dbg !17
  %242 = lshr i64 %237, 32, !dbg !17
  %243 = trunc i64 %242 to i32, !dbg !17
  %244 = insertelement <4 x i32> %241, i32 %243, i64 1, !dbg !17
  %245 = insertelement <4 x i32> %244, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1194 = insertelement <4 x i32> %245, i32 131072, i64 3, !dbg !17
  %246 = ptrtoint ptr addrspace(3) %236 to i32, !dbg !17
  %dma.buffer.lds1195 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %246), !dbg !17
  %247 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1195, i32 %dma.buffer.offset1193, <4 x i32> %dma.buffer.rsrc1194) #10, !dbg !17
  %address.local366 = mul nuw nsw i64 %address.local324, 160, !dbg !16
  %address.local367 = lshr i64 %address.local324, 2, !dbg !16
  %address.local368 = mul nsw i64 %address.local367, -640, !dbg !16
  %address.local369 = add nsw i64 %address.local366, %address.local368, !dbg !16
  %address.local370 = shl nuw nsw i64 %address.local324, 1, !dbg !16
  %address.local371 = and i64 %address.local370, 24, !dbg !16
  %address.local372 = or disjoint i64 %address.local369, %address.local371, !dbg !16
  %address.local373 = shl i64 %address.local324, 62, !dbg !16
  %address.local374 = ashr exact i64 %address.local373, 57, !dbg !16
  %address.local375 = and i64 %address.local374, -64, !dbg !16
  %address.local376 = add nsw i64 %address.local372, %address.local375, !dbg !16
  %address.local377 = lshr i64 %address.local324, 2, !dbg !11
  %address.local378 = and i64 %address.local377, 12, !dbg !11
  %address.local379 = add nsw i64 %address.local376, %address.local378, !dbg !16
  %address.local380 = and i64 %address.local324, 63, !dbg !11
  %address.local381 = lshr i64 %address.local380, 5, !dbg !11
  %address.local382 = mul nuw nsw i64 %address.local381, 1016, !dbg !16
  %address.local383 = add i64 %address.local379, %address.local382, !dbg !16
  %address.local384 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local383, !dbg !16
  %248 = load <4 x half>, ptr addrspace(3) %address.local384, align 8, !dbg !16
  %lds.const.gep1100 = getelementptr i8, ptr addrspace(3) %address.local384, i64 4096, !dbg !16
  %249 = load <4 x half>, ptr addrspace(3) %lds.const.gep1100, align 8, !dbg !16
  %250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %248, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local395 = shl nuw nsw i32 %address.thread323, 1, !dbg !16
  %address.local396 = and i32 %address.local395, 24, !dbg !16
  %address.local397 = shl nuw nsw i32 %address.thread323, 8, !dbg !16
  %address.local398 = and i32 %address.local397, 768, !dbg !16
  %address.local399 = or disjoint i32 %address.local396, %address.local398, !dbg !16
  %address.local400 = shl nuw nsw i32 %address.thread323, 5, !dbg !16
  %address.local401 = and i32 %address.local400, 96, !dbg !16
  %address.local402 = or disjoint i32 %address.local399, %address.local401, !dbg !16
  %address.local403 = zext nneg i32 %address.local402 to i64, !dbg !16
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %136) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local404 = lshr i32 %address.thread323, 1, !dbg !16
  %address.local405 = and i32 %address.local404, 1, !dbg !16
  %address.local406 = zext nneg i32 %address.local405 to i64, !dbg !16
  %address.local407 = add nsw i64 %address.base, %address.local406, !dbg !16
  %address.local408 = shl i64 %address.local407, 6, !dbg !16
  %address.local409 = sub i64 %address.local403, %address.local408, !dbg !16
  %address.local410 = add i64 %address.local409, %address.local378, !dbg !16
  %address.local411 = add i64 %address.local410, %address.local382, !dbg !16
  %address.local412 = add i64 %address.local411, 32, !dbg !16
  %address.local413 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local412, !dbg !16
  %251 = load <4 x half>, ptr addrspace(3) %address.local413, align 8, !dbg !16
  %252 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %249, <4 x float> %250) #9, !dbg !16
  %lds.const.gep1101 = getelementptr i8, ptr addrspace(3) %address.local413, i64 4096, !dbg !16
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1101, align 8, !dbg !16
  %lds.const.gep1102 = getelementptr i8, ptr addrspace(3) %address.local384, i64 128, !dbg !16
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1102, align 8, !dbg !16
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %251, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep1103 = getelementptr i8, ptr addrspace(3) %address.local384, i64 4224, !dbg !16
  %256 = load <4 x half>, ptr addrspace(3) %lds.const.gep1103, align 8, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %253, <4 x float> %255) #9, !dbg !16
  %lds.const.gep1104 = getelementptr i8, ptr addrspace(3) %address.local413, i64 128, !dbg !16
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1104, align 8, !dbg !16
  %lds.const.gep1105 = getelementptr i8, ptr addrspace(3) %address.local413, i64 4224, !dbg !16
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1105, align 8, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %254, <4 x float> zeroinitializer) #9, !dbg !16
  %261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %256, <4 x float> %260) #9, !dbg !16
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %258, <4 x float> zeroinitializer) #9, !dbg !16
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %259, <4 x float> %262) #9, !dbg !16
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local384, i64 1024, !dbg !16
  %264 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !16
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %address.local384, i64 5120, !dbg !16
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !16
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %264, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %address.local413, i64 1024, !dbg !16
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !16
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %265, <4 x float> %266) #9, !dbg !16
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local413, i64 5120, !dbg !16
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !16
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %267, <4 x float> zeroinitializer) #9, !dbg !16
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %269, <4 x float> %270) #9, !dbg !16
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local384, i64 1152, !dbg !16
  %272 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !16
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %address.local384, i64 5248, !dbg !16
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !16
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %272, <4 x float> zeroinitializer) #9, !dbg !16
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %273, <4 x float> %274) #9, !dbg !16
  %lds.const.gep1112 = getelementptr i8, ptr addrspace(3) %address.local413, i64 1152, !dbg !16
  %276 = load <4 x half>, ptr addrspace(3) %lds.const.gep1112, align 8, !dbg !16
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %276, <4 x float> zeroinitializer) #9, !dbg !16
  %lds.const.gep1113 = getelementptr i8, ptr addrspace(3) %address.local413, i64 5248, !dbg !16
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1113, align 8, !dbg !16
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %278, <4 x float> %277) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %280 = add nuw i64 %138, 2048, !dbg !17
  %281 = icmp slt i64 %280, 0, !dbg !17
  %282 = sub nuw i64 -2049, %138, !dbg !17
  %283 = select i1 %281, i64 %282, i64 %280, !dbg !17
  %284 = sdiv i64 %283, 4096, !dbg !17
  %.lobit38 = ashr i64 %280, 63, !dbg !17
  %285 = xor i64 %284, %.lobit38, !dbg !17
  %286 = add nsw i64 %285, %16, !dbg !17
  %287 = add nsw i64 %203, 64, !dbg !17
  %288 = icmp slt i64 %203, -64, !dbg !17
  %289 = sub nsw i64 -65, %203, !dbg !17
  %290 = select i1 %288, i64 %289, i64 %287, !dbg !17
  %291 = sdiv i64 %290, 128, !dbg !17
  %292 = sext i1 %288 to i64, !dbg !17
  %293 = xor i64 %291, %292, !dbg !17
  %294 = shl nsw i64 %293, 7, !dbg !17
  %295 = sub i64 %128, %294, !dbg !17
  %296 = shl i64 %286, 19, !dbg !17
  %297 = and i64 %296, 16252928, !dbg !17
  %298 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %297, !dbg !17
  %299 = getelementptr [2 x i8], ptr addrspace(1) %298, i64 %158, !dbg !17
  %300 = getelementptr [2 x i8], ptr addrspace(1) %299, i64 %295, !dbg !17
  %301 = getelementptr i8, ptr addrspace(1) %300, i64 128, !dbg !17
  %302 = getelementptr [2 x i8], ptr addrspace(1) %301, i64 %164, !dbg !17
  %303 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local329), !dbg !17
  %304 = zext i32 %303 to i64, !dbg !17
  %305 = inttoptr i64 %304 to ptr addrspace(3), !dbg !17
  %address.local722 = getelementptr [2 x i8], ptr addrspace(1) %302, i64 %address.local339, !dbg !17
  %306 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %307 = ptrtoint ptr addrspace(1) %address.local722 to i64, !dbg !17
  %308 = sub i64 %307, %306, !dbg !17
  %dma.buffer.offset1190 = trunc i64 %308 to i32, !dbg !17
  %309 = trunc i64 %306 to i32, !dbg !17
  %310 = insertelement <4 x i32> poison, i32 %309, i64 0, !dbg !17
  %311 = lshr i64 %306, 32, !dbg !17
  %312 = trunc i64 %311 to i32, !dbg !17
  %313 = insertelement <4 x i32> %310, i32 %312, i64 1, !dbg !17
  %314 = insertelement <4 x i32> %313, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1191 = insertelement <4 x i32> %314, i32 131072, i64 3, !dbg !17
  %315 = ptrtoint ptr addrspace(3) %305 to i32, !dbg !17
  %dma.buffer.lds1192 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %315), !dbg !17
  %316 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1192, i32 %dma.buffer.offset1190, <4 x i32> %dma.buffer.rsrc1191) #10, !dbg !17
  %317 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local346), !dbg !17
  %318 = zext i32 %317 to i64, !dbg !17
  %319 = inttoptr i64 %318 to ptr addrspace(3), !dbg !17
  %address.local723 = getelementptr [2 x i8], ptr addrspace(1) %302, i64 %address.local356, !dbg !17
  %320 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %321 = ptrtoint ptr addrspace(1) %address.local723 to i64, !dbg !17
  %322 = sub i64 %321, %320, !dbg !17
  %dma.buffer.offset1187 = trunc i64 %322 to i32, !dbg !17
  %323 = trunc i64 %320 to i32, !dbg !17
  %324 = insertelement <4 x i32> poison, i32 %323, i64 0, !dbg !17
  %325 = lshr i64 %320, 32, !dbg !17
  %326 = trunc i64 %325 to i32, !dbg !17
  %327 = insertelement <4 x i32> %324, i32 %326, i64 1, !dbg !17
  %328 = insertelement <4 x i32> %327, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1188 = insertelement <4 x i32> %328, i32 131072, i64 3, !dbg !17
  %329 = ptrtoint ptr addrspace(3) %319 to i32, !dbg !17
  %dma.buffer.lds1189 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %329), !dbg !17
  %330 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1189, i32 %dma.buffer.offset1187, <4 x i32> %dma.buffer.rsrc1188) #10, !dbg !17
  %lds.const.gep1114 = getelementptr i8, ptr addrspace(3) %address.local384, i64 8192, !dbg !16
  %331 = load <4 x half>, ptr addrspace(3) %lds.const.gep1114, align 8, !dbg !16
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %331, <4 x float> %252) #9, !dbg !16
  %lds.const.gep1115 = getelementptr i8, ptr addrspace(3) %address.local384, i64 12288, !dbg !16
  %333 = load <4 x half>, ptr addrspace(3) %lds.const.gep1115, align 8, !dbg !16
  %lds.const.gep1116 = getelementptr i8, ptr addrspace(3) %address.local413, i64 8192, !dbg !16
  %334 = load <4 x half>, ptr addrspace(3) %lds.const.gep1116, align 8, !dbg !16
  %335 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %333, <4 x float> %332) #9, !dbg !16
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %334, <4 x float> %257) #9, !dbg !16
  %lds.const.gep1117 = getelementptr i8, ptr addrspace(3) %address.local413, i64 12288, !dbg !16
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1117, align 8, !dbg !16
  %lds.const.gep1118 = getelementptr i8, ptr addrspace(3) %address.local384, i64 8320, !dbg !16
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1118, align 8, !dbg !16
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %337, <4 x float> %336) #9, !dbg !16
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %338, <4 x float> %261) #9, !dbg !16
  %lds.const.gep1119 = getelementptr i8, ptr addrspace(3) %address.local384, i64 12416, !dbg !16
  %341 = load <4 x half>, ptr addrspace(3) %lds.const.gep1119, align 8, !dbg !16
  %lds.const.gep1120 = getelementptr i8, ptr addrspace(3) %address.local413, i64 8320, !dbg !16
  %342 = load <4 x half>, ptr addrspace(3) %lds.const.gep1120, align 8, !dbg !16
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %341, <4 x float> %340) #9, !dbg !16
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %342, <4 x float> %263) #9, !dbg !16
  %lds.const.gep1121 = getelementptr i8, ptr addrspace(3) %address.local413, i64 12416, !dbg !16
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1121, align 8, !dbg !16
  %lds.const.gep1122 = getelementptr i8, ptr addrspace(3) %address.local384, i64 9216, !dbg !16
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1122, align 8, !dbg !16
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %345, <4 x float> %344) #9, !dbg !16
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %346, <4 x float> %268) #9, !dbg !16
  %lds.const.gep1123 = getelementptr i8, ptr addrspace(3) %address.local384, i64 13312, !dbg !16
  %349 = load <4 x half>, ptr addrspace(3) %lds.const.gep1123, align 8, !dbg !16
  %lds.const.gep1124 = getelementptr i8, ptr addrspace(3) %address.local413, i64 9216, !dbg !16
  %350 = load <4 x half>, ptr addrspace(3) %lds.const.gep1124, align 8, !dbg !16
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %349, <4 x float> %348) #9, !dbg !16
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %350, <4 x float> %271) #9, !dbg !16
  %lds.const.gep1125 = getelementptr i8, ptr addrspace(3) %address.local413, i64 13312, !dbg !16
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1125, align 8, !dbg !16
  %lds.const.gep1126 = getelementptr i8, ptr addrspace(3) %address.local384, i64 9344, !dbg !16
  %354 = load <4 x half>, ptr addrspace(3) %lds.const.gep1126, align 8, !dbg !16
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %353, <4 x float> %352) #9, !dbg !16
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %354, <4 x float> %275) #9, !dbg !16
  %lds.const.gep1127 = getelementptr i8, ptr addrspace(3) %address.local384, i64 13440, !dbg !16
  %357 = load <4 x half>, ptr addrspace(3) %lds.const.gep1127, align 8, !dbg !16
  %lds.const.gep1128 = getelementptr i8, ptr addrspace(3) %address.local413, i64 9344, !dbg !16
  %358 = load <4 x half>, ptr addrspace(3) %lds.const.gep1128, align 8, !dbg !16
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %357, <4 x float> %356) #9, !dbg !16
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %358, <4 x float> %279) #9, !dbg !16
  %lds.const.gep1129 = getelementptr i8, ptr addrspace(3) %address.local413, i64 13440, !dbg !16
  %361 = load <4 x half>, ptr addrspace(3) %lds.const.gep1129, align 8, !dbg !16
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %361, <4 x float> %360) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %363 = add nuw i64 %138, 3072, !dbg !17
  %364 = icmp slt i64 %363, 0, !dbg !17
  %365 = sub nuw i64 -3073, %138, !dbg !17
  %366 = select i1 %364, i64 %365, i64 %363, !dbg !17
  %367 = sdiv i64 %366, 4096, !dbg !17
  %.lobit39 = ashr i64 %363, 63, !dbg !17
  %368 = xor i64 %367, %.lobit39, !dbg !17
  %369 = add nsw i64 %368, %16, !dbg !17
  %370 = add nsw i64 %203, 96, !dbg !17
  %371 = icmp slt i64 %203, -96, !dbg !17
  %372 = sub nsw i64 -97, %203, !dbg !17
  %373 = select i1 %371, i64 %372, i64 %370, !dbg !17
  %374 = sdiv i64 %373, 128, !dbg !17
  %375 = sext i1 %371 to i64, !dbg !17
  %376 = xor i64 %374, %375, !dbg !17
  %377 = shl nsw i64 %376, 7, !dbg !17
  %378 = sub i64 %128, %377, !dbg !17
  %379 = shl i64 %369, 19, !dbg !17
  %380 = and i64 %379, 16252928, !dbg !17
  %381 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %380, !dbg !17
  %382 = getelementptr [2 x i8], ptr addrspace(1) %381, i64 %158, !dbg !17
  %383 = getelementptr [2 x i8], ptr addrspace(1) %382, i64 %378, !dbg !17
  %384 = getelementptr i8, ptr addrspace(1) %383, i64 192, !dbg !17
  %385 = getelementptr [2 x i8], ptr addrspace(1) %384, i64 %164, !dbg !17
  %386 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local360), !dbg !17
  %387 = zext i32 %386 to i64, !dbg !17
  %388 = inttoptr i64 %387 to ptr addrspace(3), !dbg !17
  %address.local740 = getelementptr [2 x i8], ptr addrspace(1) %385, i64 %address.local339, !dbg !17
  %389 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %390 = ptrtoint ptr addrspace(1) %address.local740 to i64, !dbg !17
  %391 = sub i64 %390, %389, !dbg !17
  %dma.buffer.offset1184 = trunc i64 %391 to i32, !dbg !17
  %392 = trunc i64 %389 to i32, !dbg !17
  %393 = insertelement <4 x i32> poison, i32 %392, i64 0, !dbg !17
  %394 = lshr i64 %389, 32, !dbg !17
  %395 = trunc i64 %394 to i32, !dbg !17
  %396 = insertelement <4 x i32> %393, i32 %395, i64 1, !dbg !17
  %397 = insertelement <4 x i32> %396, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1185 = insertelement <4 x i32> %397, i32 131072, i64 3, !dbg !17
  %398 = ptrtoint ptr addrspace(3) %388 to i32, !dbg !17
  %dma.buffer.lds1186 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %398), !dbg !17
  %399 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1186, i32 %dma.buffer.offset1184, <4 x i32> %dma.buffer.rsrc1185) #10, !dbg !17
  %400 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local364), !dbg !17
  %401 = zext i32 %400 to i64, !dbg !17
  %402 = inttoptr i64 %401 to ptr addrspace(3), !dbg !17
  %address.local741 = getelementptr [2 x i8], ptr addrspace(1) %385, i64 %address.local356, !dbg !17
  %403 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !17
  %404 = ptrtoint ptr addrspace(1) %address.local741 to i64, !dbg !17
  %405 = sub i64 %404, %403, !dbg !17
  %dma.buffer.offset1181 = trunc i64 %405 to i32, !dbg !17
  %406 = trunc i64 %403 to i32, !dbg !17
  %407 = insertelement <4 x i32> poison, i32 %406, i64 0, !dbg !17
  %408 = lshr i64 %403, 32, !dbg !17
  %409 = trunc i64 %408 to i32, !dbg !17
  %410 = insertelement <4 x i32> %407, i32 %409, i64 1, !dbg !17
  %411 = insertelement <4 x i32> %410, i32 33554432, i64 2, !dbg !17
  %dma.buffer.rsrc1182 = insertelement <4 x i32> %411, i32 131072, i64 3, !dbg !17
  %412 = ptrtoint ptr addrspace(3) %402 to i32, !dbg !17
  %dma.buffer.lds1183 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %412), !dbg !17
  %413 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1183, i32 %dma.buffer.offset1181, <4 x i32> %dma.buffer.rsrc1182) #10, !dbg !17
  %414 = load <4 x half>, ptr addrspace(3) %address.local384, align 8, !dbg !16
  %415 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %414, <4 x float> %335) #9, !dbg !16
  %lds.const.gep1130 = getelementptr i8, ptr addrspace(3) %address.local384, i64 4096, !dbg !16
  %416 = load <4 x half>, ptr addrspace(3) %lds.const.gep1130, align 8, !dbg !16
  %417 = load <4 x half>, ptr addrspace(3) %address.local413, align 8, !dbg !16
  %418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %416, <4 x float> %415) #9, !dbg !16
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %417, <4 x float> %339) #9, !dbg !16
  %lds.const.gep1131 = getelementptr i8, ptr addrspace(3) %address.local413, i64 4096, !dbg !16
  %420 = load <4 x half>, ptr addrspace(3) %lds.const.gep1131, align 8, !dbg !16
  %lds.const.gep1132 = getelementptr i8, ptr addrspace(3) %address.local384, i64 128, !dbg !16
  %421 = load <4 x half>, ptr addrspace(3) %lds.const.gep1132, align 8, !dbg !16
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %420, <4 x float> %419) #9, !dbg !16
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %421, <4 x float> %343) #9, !dbg !16
  %lds.const.gep1133 = getelementptr i8, ptr addrspace(3) %address.local384, i64 4224, !dbg !16
  %424 = load <4 x half>, ptr addrspace(3) %lds.const.gep1133, align 8, !dbg !16
  %lds.const.gep1134 = getelementptr i8, ptr addrspace(3) %address.local413, i64 128, !dbg !16
  %425 = load <4 x half>, ptr addrspace(3) %lds.const.gep1134, align 8, !dbg !16
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %424, <4 x float> %423) #9, !dbg !16
  %427 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %425, <4 x float> %347) #9, !dbg !16
  %lds.const.gep1135 = getelementptr i8, ptr addrspace(3) %address.local413, i64 4224, !dbg !16
  %428 = load <4 x half>, ptr addrspace(3) %lds.const.gep1135, align 8, !dbg !16
  %lds.const.gep1136 = getelementptr i8, ptr addrspace(3) %address.local384, i64 1024, !dbg !16
  %429 = load <4 x half>, ptr addrspace(3) %lds.const.gep1136, align 8, !dbg !16
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %428, <4 x float> %427) #9, !dbg !16
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %429, <4 x float> %351) #9, !dbg !16
  %lds.const.gep1137 = getelementptr i8, ptr addrspace(3) %address.local384, i64 5120, !dbg !16
  %432 = load <4 x half>, ptr addrspace(3) %lds.const.gep1137, align 8, !dbg !16
  %lds.const.gep1138 = getelementptr i8, ptr addrspace(3) %address.local413, i64 1024, !dbg !16
  %433 = load <4 x half>, ptr addrspace(3) %lds.const.gep1138, align 8, !dbg !16
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %432, <4 x float> %431) #9, !dbg !16
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %433, <4 x float> %355) #9, !dbg !16
  %lds.const.gep1139 = getelementptr i8, ptr addrspace(3) %address.local413, i64 5120, !dbg !16
  %436 = load <4 x half>, ptr addrspace(3) %lds.const.gep1139, align 8, !dbg !16
  %lds.const.gep1140 = getelementptr i8, ptr addrspace(3) %address.local384, i64 1152, !dbg !16
  %437 = load <4 x half>, ptr addrspace(3) %lds.const.gep1140, align 8, !dbg !16
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %436, <4 x float> %435) #9, !dbg !16
  %439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %437, <4 x float> %359) #9, !dbg !16
  %lds.const.gep1141 = getelementptr i8, ptr addrspace(3) %address.local384, i64 5248, !dbg !16
  %440 = load <4 x half>, ptr addrspace(3) %lds.const.gep1141, align 8, !dbg !16
  %lds.const.gep1142 = getelementptr i8, ptr addrspace(3) %address.local413, i64 1152, !dbg !16
  %441 = load <4 x half>, ptr addrspace(3) %lds.const.gep1142, align 8, !dbg !16
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %440, <4 x float> %439) #9, !dbg !16
  %443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %110, <4 x half> %441, <4 x float> %362) #9, !dbg !16
  %lds.const.gep1143 = getelementptr i8, ptr addrspace(3) %address.local413, i64 5248, !dbg !16
  %444 = load <4 x half>, ptr addrspace(3) %lds.const.gep1143, align 8, !dbg !16
  %445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %111, <4 x half> %444, <4 x float> %443) #9, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %446 = add nuw nsw i64 %158, %16, !dbg !18
  %447 = shl i64 %446, 19, !dbg !18
  %448 = and i64 %447, 16252928, !dbg !18
  %449 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %448, !dbg !18
  %450 = getelementptr [2 x i8], ptr addrspace(1) %449, i64 %164, !dbg !18
  %451 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local329), !dbg !18
  %452 = zext i32 %451 to i64, !dbg !18
  %453 = inttoptr i64 %452 to ptr addrspace(3), !dbg !18
  %address.local742 = shl nuw nsw i64 %address.local324, 5, !dbg !18
  %address.local743 = and i64 %address.local742, 8064, !dbg !18
  %address.local744 = shl nuw nsw i64 %address.local324, 3, !dbg !18
  %address.local745 = and i64 %address.local744, 24, !dbg !18
  %address.local746 = or disjoint i64 %address.local743, %address.local745, !dbg !18
  %address.local747 = getelementptr [2 x i8], ptr addrspace(1) %450, i64 %address.local746, !dbg !18
  %454 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %455 = ptrtoint ptr addrspace(1) %address.local747 to i64, !dbg !18
  %456 = sub i64 %455, %454, !dbg !18
  %dma.buffer.offset1178 = trunc i64 %456 to i32, !dbg !18
  %457 = trunc i64 %454 to i32, !dbg !18
  %458 = insertelement <4 x i32> poison, i32 %457, i64 0, !dbg !18
  %459 = lshr i64 %454, 32, !dbg !18
  %460 = trunc i64 %459 to i32, !dbg !18
  %461 = insertelement <4 x i32> %458, i32 %460, i64 1, !dbg !18
  %462 = insertelement <4 x i32> %461, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1179 = insertelement <4 x i32> %462, i32 131072, i64 3, !dbg !18
  %463 = ptrtoint ptr addrspace(3) %453 to i32, !dbg !18
  %dma.buffer.lds1180 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %463), !dbg !18
  %464 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1180, i32 %dma.buffer.offset1178, <4 x i32> %dma.buffer.rsrc1179) #10, !dbg !18
  %465 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local346), !dbg !18
  %466 = zext i32 %465 to i64, !dbg !18
  %467 = inttoptr i64 %466 to ptr addrspace(3), !dbg !18
  %address.local748 = shl nuw nsw i64 %address.local341, 5, !dbg !18
  %address.local749 = and i64 %address.local748, 16256, !dbg !18
  %address.local750 = shl nuw nsw i64 %address.local324, 3, !dbg !18
  %address.local751 = and i64 %address.local750, 24, !dbg !18
  %address.local752 = or disjoint i64 %address.local749, %address.local751, !dbg !18
  %address.local753 = getelementptr [2 x i8], ptr addrspace(1) %450, i64 %address.local752, !dbg !18
  %468 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %469 = ptrtoint ptr addrspace(1) %address.local753 to i64, !dbg !18
  %470 = sub i64 %469, %468, !dbg !18
  %dma.buffer.offset1175 = trunc i64 %470 to i32, !dbg !18
  %471 = trunc i64 %468 to i32, !dbg !18
  %472 = insertelement <4 x i32> poison, i32 %471, i64 0, !dbg !18
  %473 = lshr i64 %468, 32, !dbg !18
  %474 = trunc i64 %473 to i32, !dbg !18
  %475 = insertelement <4 x i32> %472, i32 %474, i64 1, !dbg !18
  %476 = insertelement <4 x i32> %475, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1176 = insertelement <4 x i32> %476, i32 131072, i64 3, !dbg !18
  %477 = ptrtoint ptr addrspace(3) %467 to i32, !dbg !18
  %dma.buffer.lds1177 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %477), !dbg !18
  %478 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1177, i32 %dma.buffer.offset1175, <4 x i32> %dma.buffer.rsrc1176) #10, !dbg !18
  %lds.const.gep1144 = getelementptr i8, ptr addrspace(3) %address.local384, i64 8192, !dbg !16
  %479 = load <4 x half>, ptr addrspace(3) %lds.const.gep1144, align 8, !dbg !16
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %479, <4 x float> %418) #9, !dbg !16
  %lds.const.gep1145 = getelementptr i8, ptr addrspace(3) %address.local384, i64 12288, !dbg !16
  %481 = load <4 x half>, ptr addrspace(3) %lds.const.gep1145, align 8, !dbg !16
  %lds.const.gep1146 = getelementptr i8, ptr addrspace(3) %address.local413, i64 8192, !dbg !16
  %482 = load <4 x half>, ptr addrspace(3) %lds.const.gep1146, align 8, !dbg !16
  %483 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %481, <4 x float> %480) #9, !dbg !16
  %484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %482, <4 x float> %422) #9, !dbg !16
  %lds.const.gep1147 = getelementptr i8, ptr addrspace(3) %address.local413, i64 12288, !dbg !16
  %485 = load <4 x half>, ptr addrspace(3) %lds.const.gep1147, align 8, !dbg !16
  %lds.const.gep1148 = getelementptr i8, ptr addrspace(3) %address.local384, i64 8320, !dbg !16
  %486 = load <4 x half>, ptr addrspace(3) %lds.const.gep1148, align 8, !dbg !16
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %485, <4 x float> %484) #9, !dbg !16
  %488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %486, <4 x float> %426) #9, !dbg !16
  %lds.const.gep1149 = getelementptr i8, ptr addrspace(3) %address.local384, i64 12416, !dbg !16
  %489 = load <4 x half>, ptr addrspace(3) %lds.const.gep1149, align 8, !dbg !16
  %lds.const.gep1150 = getelementptr i8, ptr addrspace(3) %address.local413, i64 8320, !dbg !16
  %490 = load <4 x half>, ptr addrspace(3) %lds.const.gep1150, align 8, !dbg !16
  %491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %489, <4 x float> %488) #9, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %490, <4 x float> %430) #9, !dbg !16
  %lds.const.gep1151 = getelementptr i8, ptr addrspace(3) %address.local413, i64 12416, !dbg !16
  %493 = load <4 x half>, ptr addrspace(3) %lds.const.gep1151, align 8, !dbg !16
  %lds.const.gep1152 = getelementptr i8, ptr addrspace(3) %address.local384, i64 9216, !dbg !16
  %494 = load <4 x half>, ptr addrspace(3) %lds.const.gep1152, align 8, !dbg !16
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %493, <4 x float> %492) #9, !dbg !16
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %494, <4 x float> %434) #9, !dbg !16
  %lds.const.gep1153 = getelementptr i8, ptr addrspace(3) %address.local384, i64 13312, !dbg !16
  %497 = load <4 x half>, ptr addrspace(3) %lds.const.gep1153, align 8, !dbg !16
  %lds.const.gep1154 = getelementptr i8, ptr addrspace(3) %address.local413, i64 9216, !dbg !16
  %498 = load <4 x half>, ptr addrspace(3) %lds.const.gep1154, align 8, !dbg !16
  %499 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %497, <4 x float> %496) #9, !dbg !16
  %500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %498, <4 x float> %438) #9, !dbg !16
  %lds.const.gep1155 = getelementptr i8, ptr addrspace(3) %address.local413, i64 13312, !dbg !16
  %501 = load <4 x half>, ptr addrspace(3) %lds.const.gep1155, align 8, !dbg !16
  %lds.const.gep1156 = getelementptr i8, ptr addrspace(3) %address.local384, i64 9344, !dbg !16
  %502 = load <4 x half>, ptr addrspace(3) %lds.const.gep1156, align 8, !dbg !16
  %503 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %501, <4 x float> %500) #9, !dbg !16
  %504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %502, <4 x float> %442) #9, !dbg !16
  %lds.const.gep1157 = getelementptr i8, ptr addrspace(3) %address.local384, i64 13440, !dbg !16
  %505 = load <4 x half>, ptr addrspace(3) %lds.const.gep1157, align 8, !dbg !16
  %lds.const.gep1158 = getelementptr i8, ptr addrspace(3) %address.local413, i64 9344, !dbg !16
  %506 = load <4 x half>, ptr addrspace(3) %lds.const.gep1158, align 8, !dbg !16
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %505, <4 x float> %504) #9, !dbg !16
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %114, <4 x half> %506, <4 x float> %445) #9, !dbg !16
  %lds.const.gep1159 = getelementptr i8, ptr addrspace(3) %address.local413, i64 13440, !dbg !16
  %509 = load <4 x half>, ptr addrspace(3) %lds.const.gep1159, align 8, !dbg !16
  %510 = or disjoint i64 %155, 127, !dbg !14
  %511 = icmp uge i64 %50, %510, !dbg !14
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %115, <4 x half> %509, <4 x float> %508) #9, !dbg !16
  %513 = and i1 %118, %511, !dbg !14
  br i1 %513, label %514, label %515, !dbg !14

514:                                              ; preds = %154
  br label %590, !dbg !14

515:                                              ; preds = %154
  %516 = add i64 %50, %32, !dbg !14
  %517 = shl nuw nsw i64 %51, 4, !dbg !14
  %518 = add i64 %516, %517, !dbg !14
  %519 = and i32 %31, 240, !dbg !14
  %index.compact = zext nneg i32 %519 to i64, !dbg !14
  %520 = sub i64 %518, %index.compact, !dbg !14
  %521 = add i64 %520, 1, !dbg !23
  %522 = and i64 %32, 48, !dbg !14
  %523 = or disjoint i64 %155, %522, !dbg !14
  %524 = lshr i64 %521, 32, !dbg !24
  %compare.high = trunc i64 %524 to i32, !dbg !24
  %525 = lshr i64 %523, 32, !dbg !24
  %compare.high971 = trunc i64 %525 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high971, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high971, !dbg !24
  %compare.low.base = trunc i64 %521 to i32, !dbg !24
  %compare.low.base972 = trunc i64 %523 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base972, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %526 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low973 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result974 = select i1 %compare.high.equal, i1 %compare.low973, i1 %compare.high.order, !dbg !24
  %527 = select i1 %compare.result974, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base975 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local976 = or i32 %compare.local.base975, 2, !dbg !24
  %compare.low977 = icmp ugt i32 %compare.low.base, %compare.local976, !dbg !24
  %compare.result978 = select i1 %compare.high.equal, i1 %compare.low977, i1 %compare.high.order, !dbg !24
  %528 = select i1 %compare.result978, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base979 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local980 = or i32 %compare.local.base979, 3, !dbg !24
  %compare.low981 = icmp ugt i32 %compare.low.base, %compare.local980, !dbg !24
  %compare.result982 = select i1 %compare.high.equal, i1 %compare.low981, i1 %compare.high.order, !dbg !24
  %529 = select i1 %compare.result982, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base983 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local984 = or i32 %compare.local.base983, 4, !dbg !24
  %compare.low985 = icmp ugt i32 %compare.low.base, %compare.local984, !dbg !24
  %compare.result986 = select i1 %compare.high.equal, i1 %compare.low985, i1 %compare.high.order, !dbg !24
  %530 = select i1 %compare.result986, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base987 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local988 = or i32 %compare.local.base987, 5, !dbg !24
  %compare.low989 = icmp ugt i32 %compare.low.base, %compare.local988, !dbg !24
  %compare.result990 = select i1 %compare.high.equal, i1 %compare.low989, i1 %compare.high.order, !dbg !24
  %531 = select i1 %compare.result990, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base991 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local992 = or i32 %compare.local.base991, 6, !dbg !24
  %compare.low993 = icmp ugt i32 %compare.low.base, %compare.local992, !dbg !24
  %compare.result994 = select i1 %compare.high.equal, i1 %compare.low993, i1 %compare.high.order, !dbg !24
  %532 = select i1 %compare.result994, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base995 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local996 = or i32 %compare.local.base995, 7, !dbg !24
  %compare.low997 = icmp ugt i32 %compare.low.base, %compare.local996, !dbg !24
  %compare.result998 = select i1 %compare.high.equal, i1 %compare.low997, i1 %compare.high.order, !dbg !24
  %533 = select i1 %compare.result998, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base999 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1000 = or i32 %compare.local.base999, 8, !dbg !24
  %compare.low1001 = icmp ugt i32 %compare.low.base, %compare.local1000, !dbg !24
  %compare.result1002 = select i1 %compare.high.equal, i1 %compare.low1001, i1 %compare.high.order, !dbg !24
  %534 = select i1 %compare.result1002, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1003 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1004 = or i32 %compare.local.base1003, 9, !dbg !24
  %compare.low1005 = icmp ugt i32 %compare.low.base, %compare.local1004, !dbg !24
  %compare.result1006 = select i1 %compare.high.equal, i1 %compare.low1005, i1 %compare.high.order, !dbg !24
  %535 = select i1 %compare.result1006, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1007 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1008 = or i32 %compare.local.base1007, 10, !dbg !24
  %compare.low1009 = icmp ugt i32 %compare.low.base, %compare.local1008, !dbg !24
  %compare.result1010 = select i1 %compare.high.equal, i1 %compare.low1009, i1 %compare.high.order, !dbg !24
  %536 = select i1 %compare.result1010, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1011 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1012 = or i32 %compare.local.base1011, 11, !dbg !24
  %compare.low1013 = icmp ugt i32 %compare.low.base, %compare.local1012, !dbg !24
  %compare.result1014 = select i1 %compare.high.equal, i1 %compare.low1013, i1 %compare.high.order, !dbg !24
  %537 = select i1 %compare.result1014, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1015 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1016 = or i32 %compare.local.base1015, 12, !dbg !24
  %compare.low1017 = icmp ugt i32 %compare.low.base, %compare.local1016, !dbg !24
  %compare.result1018 = select i1 %compare.high.equal, i1 %compare.low1017, i1 %compare.high.order, !dbg !24
  %538 = select i1 %compare.result1018, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1019 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1020 = or i32 %compare.local.base1019, 13, !dbg !24
  %compare.low1021 = icmp ugt i32 %compare.low.base, %compare.local1020, !dbg !24
  %compare.result1022 = select i1 %compare.high.equal, i1 %compare.low1021, i1 %compare.high.order, !dbg !24
  %539 = select i1 %compare.result1022, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1023 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1024 = or i32 %compare.local.base1023, 14, !dbg !24
  %compare.low1025 = icmp ugt i32 %compare.low.base, %compare.local1024, !dbg !24
  %compare.result1026 = select i1 %compare.high.equal, i1 %compare.low1025, i1 %compare.high.order, !dbg !24
  %540 = select i1 %compare.result1026, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1027 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1028 = or i32 %compare.local.base1027, 15, !dbg !24
  %compare.low1029 = icmp ugt i32 %compare.low.base, %compare.local1028, !dbg !24
  %compare.result1030 = select i1 %compare.high.equal, i1 %compare.low1029, i1 %compare.high.order, !dbg !24
  %541 = select i1 %compare.result1030, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1031 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1032 = or i32 %compare.local.base1031, 64, !dbg !24
  %compare.low1033 = icmp ugt i32 %compare.low.base, %compare.local1032, !dbg !24
  %compare.result1034 = select i1 %compare.high.equal, i1 %compare.low1033, i1 %compare.high.order, !dbg !24
  %542 = select i1 %compare.result1034, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1035 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1036 = or i32 %compare.local.base1035, 65, !dbg !24
  %compare.low1037 = icmp ugt i32 %compare.low.base, %compare.local1036, !dbg !24
  %compare.result1038 = select i1 %compare.high.equal, i1 %compare.low1037, i1 %compare.high.order, !dbg !24
  %543 = select i1 %compare.result1038, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1039 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1040 = or i32 %compare.local.base1039, 66, !dbg !24
  %compare.low1041 = icmp ugt i32 %compare.low.base, %compare.local1040, !dbg !24
  %compare.result1042 = select i1 %compare.high.equal, i1 %compare.low1041, i1 %compare.high.order, !dbg !24
  %544 = select i1 %compare.result1042, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1043 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1044 = or i32 %compare.local.base1043, 67, !dbg !24
  %compare.low1045 = icmp ugt i32 %compare.low.base, %compare.local1044, !dbg !24
  %compare.result1046 = select i1 %compare.high.equal, i1 %compare.low1045, i1 %compare.high.order, !dbg !24
  %545 = select i1 %compare.result1046, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1047 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1048 = or i32 %compare.local.base1047, 68, !dbg !24
  %compare.low1049 = icmp ugt i32 %compare.low.base, %compare.local1048, !dbg !24
  %compare.result1050 = select i1 %compare.high.equal, i1 %compare.low1049, i1 %compare.high.order, !dbg !24
  %546 = select i1 %compare.result1050, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1051 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1052 = or i32 %compare.local.base1051, 69, !dbg !24
  %compare.low1053 = icmp ugt i32 %compare.low.base, %compare.local1052, !dbg !24
  %compare.result1054 = select i1 %compare.high.equal, i1 %compare.low1053, i1 %compare.high.order, !dbg !24
  %547 = select i1 %compare.result1054, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1055 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1056 = or i32 %compare.local.base1055, 70, !dbg !24
  %compare.low1057 = icmp ugt i32 %compare.low.base, %compare.local1056, !dbg !24
  %compare.result1058 = select i1 %compare.high.equal, i1 %compare.low1057, i1 %compare.high.order, !dbg !24
  %548 = select i1 %compare.result1058, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1059 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1060 = or i32 %compare.local.base1059, 71, !dbg !24
  %compare.low1061 = icmp ugt i32 %compare.low.base, %compare.local1060, !dbg !24
  %compare.result1062 = select i1 %compare.high.equal, i1 %compare.low1061, i1 %compare.high.order, !dbg !24
  %549 = select i1 %compare.result1062, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1063 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1064 = or i32 %compare.local.base1063, 72, !dbg !24
  %compare.low1065 = icmp ugt i32 %compare.low.base, %compare.local1064, !dbg !24
  %compare.result1066 = select i1 %compare.high.equal, i1 %compare.low1065, i1 %compare.high.order, !dbg !24
  %550 = select i1 %compare.result1066, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1067 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1068 = or i32 %compare.local.base1067, 73, !dbg !24
  %compare.low1069 = icmp ugt i32 %compare.low.base, %compare.local1068, !dbg !24
  %compare.result1070 = select i1 %compare.high.equal, i1 %compare.low1069, i1 %compare.high.order, !dbg !24
  %551 = select i1 %compare.result1070, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1071 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1072 = or i32 %compare.local.base1071, 74, !dbg !24
  %compare.low1073 = icmp ugt i32 %compare.low.base, %compare.local1072, !dbg !24
  %compare.result1074 = select i1 %compare.high.equal, i1 %compare.low1073, i1 %compare.high.order, !dbg !24
  %552 = select i1 %compare.result1074, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1075 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1076 = or i32 %compare.local.base1075, 75, !dbg !24
  %compare.low1077 = icmp ugt i32 %compare.low.base, %compare.local1076, !dbg !24
  %compare.result1078 = select i1 %compare.high.equal, i1 %compare.low1077, i1 %compare.high.order, !dbg !24
  %553 = select i1 %compare.result1078, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1079 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1080 = or i32 %compare.local.base1079, 76, !dbg !24
  %compare.low1081 = icmp ugt i32 %compare.low.base, %compare.local1080, !dbg !24
  %compare.result1082 = select i1 %compare.high.equal, i1 %compare.low1081, i1 %compare.high.order, !dbg !24
  %554 = select i1 %compare.result1082, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1083 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1084 = or i32 %compare.local.base1083, 77, !dbg !24
  %compare.low1085 = icmp ugt i32 %compare.low.base, %compare.local1084, !dbg !24
  %compare.result1086 = select i1 %compare.high.equal, i1 %compare.low1085, i1 %compare.high.order, !dbg !24
  %555 = select i1 %compare.result1086, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1087 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1088 = or i32 %compare.local.base1087, 78, !dbg !24
  %compare.low1089 = icmp ugt i32 %compare.low.base, %compare.local1088, !dbg !24
  %compare.result1090 = select i1 %compare.high.equal, i1 %compare.low1089, i1 %compare.high.order, !dbg !24
  %556 = select i1 %compare.result1090, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base1091 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base972, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local1092 = or i32 %compare.local.base1091, 79, !dbg !24
  %compare.low1093 = icmp ugt i32 %compare.low.base, %compare.local1092, !dbg !24
  %compare.result1094 = select i1 %compare.high.equal, i1 %compare.low1093, i1 %compare.high.order, !dbg !24
  %557 = select i1 %compare.result1094, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %558 = insertelement <32 x float> poison, float %526, i64 0, !dbg !14
  %559 = insertelement <32 x float> %558, float %527, i64 1, !dbg !14
  %560 = insertelement <32 x float> %559, float %528, i64 2, !dbg !14
  %561 = insertelement <32 x float> %560, float %529, i64 3, !dbg !14
  %562 = insertelement <32 x float> %561, float %530, i64 4, !dbg !14
  %563 = insertelement <32 x float> %562, float %531, i64 5, !dbg !14
  %564 = insertelement <32 x float> %563, float %532, i64 6, !dbg !14
  %565 = insertelement <32 x float> %564, float %533, i64 7, !dbg !14
  %566 = insertelement <32 x float> %565, float %534, i64 8, !dbg !14
  %567 = insertelement <32 x float> %566, float %535, i64 9, !dbg !14
  %568 = insertelement <32 x float> %567, float %536, i64 10, !dbg !14
  %569 = insertelement <32 x float> %568, float %537, i64 11, !dbg !14
  %570 = insertelement <32 x float> %569, float %538, i64 12, !dbg !14
  %571 = insertelement <32 x float> %570, float %539, i64 13, !dbg !14
  %572 = insertelement <32 x float> %571, float %540, i64 14, !dbg !14
  %573 = insertelement <32 x float> %572, float %541, i64 15, !dbg !14
  %574 = insertelement <32 x float> %573, float %542, i64 16, !dbg !14
  %575 = insertelement <32 x float> %574, float %543, i64 17, !dbg !14
  %576 = insertelement <32 x float> %575, float %544, i64 18, !dbg !14
  %577 = insertelement <32 x float> %576, float %545, i64 19, !dbg !14
  %578 = insertelement <32 x float> %577, float %546, i64 20, !dbg !14
  %579 = insertelement <32 x float> %578, float %547, i64 21, !dbg !14
  %580 = insertelement <32 x float> %579, float %548, i64 22, !dbg !14
  %581 = insertelement <32 x float> %580, float %549, i64 23, !dbg !14
  %582 = insertelement <32 x float> %581, float %550, i64 24, !dbg !14
  %583 = insertelement <32 x float> %582, float %551, i64 25, !dbg !14
  %584 = insertelement <32 x float> %583, float %552, i64 26, !dbg !14
  %585 = insertelement <32 x float> %584, float %553, i64 27, !dbg !14
  %586 = insertelement <32 x float> %585, float %554, i64 28, !dbg !14
  %587 = insertelement <32 x float> %586, float %555, i64 29, !dbg !14
  %588 = insertelement <32 x float> %587, float %556, i64 30, !dbg !14
  %589 = insertelement <32 x float> %588, float %557, i64 31, !dbg !14
  br label %590, !dbg !14

590:                                              ; preds = %514, %515
  %591 = phi <32 x float> [ %589, %515 ], [ zeroinitializer, %514 ], !dbg !14
  br label %592, !dbg !14

592:                                              ; preds = %590
  %593 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %594 = fadd <4 x float> %483, %593, !dbg !26
  %595 = extractelement <4 x float> %594, i64 0, !dbg !26
  %596 = extractelement <4 x float> %594, i64 1, !dbg !26
  %597 = extractelement <4 x float> %594, i64 2, !dbg !26
  %598 = extractelement <4 x float> %594, i64 3, !dbg !26
  %599 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %600 = fadd <4 x float> %487, %599, !dbg !26
  %601 = extractelement <4 x float> %600, i64 0, !dbg !26
  %602 = extractelement <4 x float> %600, i64 1, !dbg !26
  %603 = extractelement <4 x float> %600, i64 2, !dbg !26
  %604 = extractelement <4 x float> %600, i64 3, !dbg !26
  %605 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %606 = fadd <4 x float> %491, %605, !dbg !26
  %607 = extractelement <4 x float> %606, i64 0, !dbg !26
  %608 = extractelement <4 x float> %606, i64 1, !dbg !26
  %609 = extractelement <4 x float> %606, i64 2, !dbg !26
  %610 = extractelement <4 x float> %606, i64 3, !dbg !26
  %611 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %612 = fadd <4 x float> %495, %611, !dbg !26
  %613 = extractelement <4 x float> %612, i64 0, !dbg !26
  %614 = extractelement <4 x float> %612, i64 1, !dbg !26
  %615 = extractelement <4 x float> %612, i64 2, !dbg !26
  %616 = extractelement <4 x float> %612, i64 3, !dbg !26
  %617 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %618 = fadd <4 x float> %499, %617, !dbg !26
  %619 = extractelement <4 x float> %618, i64 0, !dbg !26
  %620 = extractelement <4 x float> %618, i64 1, !dbg !26
  %621 = extractelement <4 x float> %618, i64 2, !dbg !26
  %622 = extractelement <4 x float> %618, i64 3, !dbg !26
  %623 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %624 = fadd <4 x float> %503, %623, !dbg !26
  %625 = extractelement <4 x float> %624, i64 0, !dbg !26
  %626 = extractelement <4 x float> %624, i64 1, !dbg !26
  %627 = extractelement <4 x float> %624, i64 2, !dbg !26
  %628 = extractelement <4 x float> %624, i64 3, !dbg !26
  %629 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %630 = fadd <4 x float> %507, %629, !dbg !26
  %631 = extractelement <4 x float> %630, i64 0, !dbg !26
  %632 = extractelement <4 x float> %630, i64 1, !dbg !26
  %633 = extractelement <4 x float> %630, i64 2, !dbg !26
  %634 = extractelement <4 x float> %630, i64 3, !dbg !26
  %635 = shufflevector <32 x float> %591, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %636 = fadd <4 x float> %512, %635, !dbg !26
  %637 = extractelement <4 x float> %636, i64 0, !dbg !26
  %638 = extractelement <4 x float> %636, i64 1, !dbg !26
  %639 = extractelement <4 x float> %636, i64 2, !dbg !26
  %640 = extractelement <4 x float> %636, i64 3, !dbg !26
  %641 = call float @llvm.maxnum.f32(float %595, float 0xFFF0000000000000), !dbg !27
  %642 = call float @llvm.maxnum.f32(float %641, float %596), !dbg !27
  %643 = call float @llvm.maxnum.f32(float %642, float %597), !dbg !27
  %644 = call float @llvm.maxnum.f32(float %643, float %598), !dbg !27
  %645 = call float @llvm.maxnum.f32(float %644, float %601), !dbg !27
  %646 = call float @llvm.maxnum.f32(float %645, float %602), !dbg !27
  %647 = call float @llvm.maxnum.f32(float %646, float %603), !dbg !27
  %648 = call float @llvm.maxnum.f32(float %647, float %604), !dbg !27
  %649 = call float @llvm.maxnum.f32(float %648, float %607), !dbg !27
  %650 = call float @llvm.maxnum.f32(float %649, float %608), !dbg !27
  %651 = call float @llvm.maxnum.f32(float %650, float %609), !dbg !27
  %652 = call float @llvm.maxnum.f32(float %651, float %610), !dbg !27
  %653 = call float @llvm.maxnum.f32(float %652, float %613), !dbg !27
  %654 = call float @llvm.maxnum.f32(float %653, float %614), !dbg !27
  %655 = call float @llvm.maxnum.f32(float %654, float %615), !dbg !27
  %656 = call float @llvm.maxnum.f32(float %655, float %616), !dbg !27
  %657 = call float @llvm.maxnum.f32(float %656, float %619), !dbg !27
  %658 = call float @llvm.maxnum.f32(float %657, float %620), !dbg !27
  %659 = call float @llvm.maxnum.f32(float %658, float %621), !dbg !27
  %660 = call float @llvm.maxnum.f32(float %659, float %622), !dbg !27
  %661 = call float @llvm.maxnum.f32(float %660, float %625), !dbg !27
  %662 = call float @llvm.maxnum.f32(float %661, float %626), !dbg !27
  %663 = call float @llvm.maxnum.f32(float %662, float %627), !dbg !27
  %664 = call float @llvm.maxnum.f32(float %663, float %628), !dbg !27
  %665 = call float @llvm.maxnum.f32(float %664, float %631), !dbg !27
  %666 = call float @llvm.maxnum.f32(float %665, float %632), !dbg !27
  %667 = call float @llvm.maxnum.f32(float %666, float %633), !dbg !27
  %668 = call float @llvm.maxnum.f32(float %667, float %634), !dbg !27
  %669 = call float @llvm.maxnum.f32(float %668, float %637), !dbg !27
  %670 = call float @llvm.maxnum.f32(float %669, float %638), !dbg !27
  %671 = call float @llvm.maxnum.f32(float %670, float %639), !dbg !27
  %672 = call float @llvm.maxnum.f32(float %671, float %640), !dbg !27
  %673 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !27
  %674 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %673), !dbg !27
  %675 = add nuw nsw i32 %674, 64, !dbg !27
  %676 = xor i32 %674, 16, !dbg !27
  %677 = and i32 %675, 192, !dbg !27
  %678 = icmp samesign ult i32 %676, %677, !dbg !27
  %679 = select i1 %678, i32 %676, i32 %674, !dbg !27
  %680 = shl nuw nsw i32 %679, 2, !dbg !27
  %681 = bitcast float %672 to i32, !dbg !27
  %682 = call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %681), !dbg !27
  %683 = bitcast i32 %682 to float, !dbg !27
  %684 = call float @llvm.maxnum.f32(float %672, float %683), !dbg !27
  %685 = xor i32 %674, 32, !dbg !27
  %686 = icmp samesign ult i32 %685, %677, !dbg !27
  %687 = select i1 %686, i32 %685, i32 %674, !dbg !27
  %688 = shl nuw nsw i32 %687, 2, !dbg !27
  %689 = bitcast float %684 to i32, !dbg !27
  %690 = call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %689), !dbg !27
  %691 = bitcast i32 %690 to float, !dbg !27
  %692 = call float @llvm.maxnum.f32(float %684, float %691), !dbg !27
  %693 = insertelement <1 x float> poison, float %692, i64 0, !dbg !27
  %694 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %693, <1 x float> %143), !dbg !27
  %695 = shufflevector <1 x float> %694, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %696 = fsub <4 x float> %594, %695, !dbg !28
  %697 = fsub <4 x float> %600, %695, !dbg !28
  %698 = fsub <4 x float> %606, %695, !dbg !28
  %699 = fsub <4 x float> %612, %695, !dbg !28
  %700 = fsub <4 x float> %618, %695, !dbg !28
  %701 = fsub <4 x float> %624, %695, !dbg !28
  %702 = fsub <4 x float> %630, %695, !dbg !28
  %703 = fsub <4 x float> %636, %695, !dbg !28
  %704 = fsub <1 x float> %143, %694, !dbg !29
  %705 = extractelement <1 x float> %704, i64 0, !dbg !30
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !30
  %707 = insertelement <4 x float> poison, float %706, i64 0, !dbg !35
  %708 = shufflevector <4 x float> %707, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !35
  %709 = fmul <4 x float> %151, %708, !dbg !35
  %710 = fmul <4 x float> %150, %708, !dbg !35
  %711 = fmul <4 x float> %148, %708, !dbg !35
  %712 = fmul <4 x float> %147, %708, !dbg !35
  %713 = fmul <4 x float> %145, %708, !dbg !35
  %714 = fmul <4 x float> %144, %708, !dbg !35
  %715 = fmul <4 x float> %140, %708, !dbg !35
  %716 = fmul <4 x float> %139, %708, !dbg !35
  %717 = extractelement <1 x float> %142, i64 0, !dbg !31
  %718 = fmul float %717, %706, !dbg !31
  %719 = extractelement <4 x float> %696, i64 0, !dbg !32
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !32
  %721 = insertelement <4 x float> poison, float %720, i64 0, !dbg !34
  %722 = extractelement <4 x float> %696, i64 1, !dbg !32
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !32
  %724 = insertelement <4 x float> %721, float %723, i64 1, !dbg !34
  %725 = fadd float %720, %723, !dbg !33
  %726 = extractelement <4 x float> %696, i64 2, !dbg !32
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !32
  %728 = insertelement <4 x float> %724, float %727, i64 2, !dbg !34
  %729 = fadd float %725, %727, !dbg !33
  %730 = extractelement <4 x float> %696, i64 3, !dbg !32
  %731 = call float @llvm.amdgcn.exp2.f32(float %730), !dbg !32
  %732 = insertelement <4 x float> %728, float %731, i64 3, !dbg !34
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %732) #9, !dbg !34, !frisk.register_lifetimes !1
  %733 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !34
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %733) #9, !dbg !33, !frisk.register_lifetimes !1
  %734 = fadd float %729, %731, !dbg !33
  %735 = extractelement <4 x float> %697, i64 0, !dbg !32
  %736 = call float @llvm.amdgcn.exp2.f32(float %735), !dbg !32
  %737 = insertelement <4 x float> poison, float %736, i64 0, !dbg !34
  %738 = fadd float %734, %736, !dbg !33
  %739 = extractelement <4 x float> %697, i64 1, !dbg !32
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !32
  %741 = insertelement <4 x float> %737, float %740, i64 1, !dbg !34
  %742 = fadd float %738, %740, !dbg !33
  %743 = extractelement <4 x float> %697, i64 2, !dbg !32
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !32
  %745 = insertelement <4 x float> %741, float %744, i64 2, !dbg !34
  %746 = fadd float %742, %744, !dbg !33
  %747 = extractelement <4 x float> %697, i64 3, !dbg !32
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !32
  %749 = insertelement <4 x float> %745, float %748, i64 3, !dbg !34
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #9, !dbg !34, !frisk.register_lifetimes !1
  %750 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !34
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #9, !dbg !33, !frisk.register_lifetimes !1
  %751 = fadd float %746, %748, !dbg !33
  %752 = extractelement <4 x float> %698, i64 0, !dbg !32
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !32
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !34
  %755 = fadd float %751, %753, !dbg !33
  %756 = extractelement <4 x float> %698, i64 1, !dbg !32
  %757 = call float @llvm.amdgcn.exp2.f32(float %756), !dbg !32
  %758 = insertelement <4 x float> %754, float %757, i64 1, !dbg !34
  %759 = fadd float %755, %757, !dbg !33
  %760 = extractelement <4 x float> %698, i64 2, !dbg !32
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !32
  %762 = insertelement <4 x float> %758, float %761, i64 2, !dbg !34
  %763 = fadd float %759, %761, !dbg !33
  %764 = extractelement <4 x float> %698, i64 3, !dbg !32
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !32
  %766 = insertelement <4 x float> %762, float %765, i64 3, !dbg !34
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #9, !dbg !34, !frisk.register_lifetimes !1
  %767 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !34
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #9, !dbg !33, !frisk.register_lifetimes !1
  %768 = fadd float %763, %765, !dbg !33
  %769 = extractelement <4 x float> %699, i64 0, !dbg !32
  %770 = call float @llvm.amdgcn.exp2.f32(float %769), !dbg !32
  %771 = insertelement <4 x float> poison, float %770, i64 0, !dbg !34
  %772 = fadd float %768, %770, !dbg !33
  %773 = extractelement <4 x float> %699, i64 1, !dbg !32
  %774 = call float @llvm.amdgcn.exp2.f32(float %773), !dbg !32
  %775 = insertelement <4 x float> %771, float %774, i64 1, !dbg !34
  %776 = fadd float %772, %774, !dbg !33
  %777 = extractelement <4 x float> %699, i64 2, !dbg !32
  %778 = call float @llvm.amdgcn.exp2.f32(float %777), !dbg !32
  %779 = insertelement <4 x float> %775, float %778, i64 2, !dbg !34
  %780 = fadd float %776, %778, !dbg !33
  %781 = extractelement <4 x float> %699, i64 3, !dbg !32
  %782 = call float @llvm.amdgcn.exp2.f32(float %781), !dbg !32
  %783 = insertelement <4 x float> %779, float %782, i64 3, !dbg !34
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #9, !dbg !34, !frisk.register_lifetimes !1
  %784 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !34
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #9, !dbg !33, !frisk.register_lifetimes !1
  %785 = fadd float %780, %782, !dbg !33
  %786 = extractelement <4 x float> %700, i64 0, !dbg !32
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !32
  %788 = insertelement <4 x float> poison, float %787, i64 0, !dbg !34
  %789 = fadd float %785, %787, !dbg !33
  %790 = extractelement <4 x float> %700, i64 1, !dbg !32
  %791 = call float @llvm.amdgcn.exp2.f32(float %790), !dbg !32
  %792 = insertelement <4 x float> %788, float %791, i64 1, !dbg !34
  %793 = fadd float %789, %791, !dbg !33
  %794 = extractelement <4 x float> %700, i64 2, !dbg !32
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !32
  %796 = insertelement <4 x float> %792, float %795, i64 2, !dbg !34
  %797 = fadd float %793, %795, !dbg !33
  %798 = extractelement <4 x float> %700, i64 3, !dbg !32
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !32
  %800 = insertelement <4 x float> %796, float %799, i64 3, !dbg !34
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %800) #9, !dbg !34, !frisk.register_lifetimes !1
  %801 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !34
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %801) #9, !dbg !33, !frisk.register_lifetimes !1
  %802 = fadd float %797, %799, !dbg !33
  %803 = extractelement <4 x float> %701, i64 0, !dbg !32
  %804 = call float @llvm.amdgcn.exp2.f32(float %803), !dbg !32
  %805 = insertelement <4 x float> poison, float %804, i64 0, !dbg !34
  %806 = fadd float %802, %804, !dbg !33
  %807 = extractelement <4 x float> %701, i64 1, !dbg !32
  %808 = call float @llvm.amdgcn.exp2.f32(float %807), !dbg !32
  %809 = insertelement <4 x float> %805, float %808, i64 1, !dbg !34
  %810 = fadd float %806, %808, !dbg !33
  %811 = extractelement <4 x float> %701, i64 2, !dbg !32
  %812 = call float @llvm.amdgcn.exp2.f32(float %811), !dbg !32
  %813 = insertelement <4 x float> %809, float %812, i64 2, !dbg !34
  %814 = fadd float %810, %812, !dbg !33
  %815 = extractelement <4 x float> %701, i64 3, !dbg !32
  %816 = call float @llvm.amdgcn.exp2.f32(float %815), !dbg !32
  %817 = insertelement <4 x float> %813, float %816, i64 3, !dbg !34
  %math.result271 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %817) #9, !dbg !34, !frisk.register_lifetimes !1
  %818 = fptrunc <4 x float> %math.result271 to <4 x half>, !dbg !34
  %math.packed272 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %818) #9, !dbg !33, !frisk.register_lifetimes !1
  %819 = fadd float %814, %816, !dbg !33
  %820 = extractelement <4 x float> %702, i64 0, !dbg !32
  %821 = call float @llvm.amdgcn.exp2.f32(float %820), !dbg !32
  %822 = insertelement <4 x float> poison, float %821, i64 0, !dbg !34
  %823 = fadd float %819, %821, !dbg !33
  %824 = extractelement <4 x float> %702, i64 1, !dbg !32
  %825 = call float @llvm.amdgcn.exp2.f32(float %824), !dbg !32
  %826 = insertelement <4 x float> %822, float %825, i64 1, !dbg !34
  %827 = fadd float %823, %825, !dbg !33
  %828 = extractelement <4 x float> %702, i64 2, !dbg !32
  %829 = call float @llvm.amdgcn.exp2.f32(float %828), !dbg !32
  %830 = insertelement <4 x float> %826, float %829, i64 2, !dbg !34
  %831 = fadd float %827, %829, !dbg !33
  %832 = extractelement <4 x float> %702, i64 3, !dbg !32
  %833 = call float @llvm.amdgcn.exp2.f32(float %832), !dbg !32
  %834 = insertelement <4 x float> %830, float %833, i64 3, !dbg !34
  %math.result273 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %834) #9, !dbg !34, !frisk.register_lifetimes !1
  %835 = fptrunc <4 x float> %math.result273 to <4 x half>, !dbg !34
  %math.packed274 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %835) #9, !dbg !33, !frisk.register_lifetimes !1
  %836 = fadd float %831, %833, !dbg !33
  %837 = extractelement <4 x float> %703, i64 0, !dbg !32
  %838 = call float @llvm.amdgcn.exp2.f32(float %837), !dbg !32
  %839 = insertelement <4 x float> poison, float %838, i64 0, !dbg !34
  %840 = fadd float %836, %838, !dbg !33
  %841 = extractelement <4 x float> %703, i64 1, !dbg !32
  %842 = call float @llvm.amdgcn.exp2.f32(float %841), !dbg !32
  %843 = insertelement <4 x float> %839, float %842, i64 1, !dbg !34
  %844 = fadd float %840, %842, !dbg !33
  %845 = extractelement <4 x float> %703, i64 2, !dbg !32
  %846 = call float @llvm.amdgcn.exp2.f32(float %845), !dbg !32
  %847 = insertelement <4 x float> %843, float %846, i64 2, !dbg !34
  %848 = fadd float %844, %846, !dbg !33
  %849 = extractelement <4 x float> %703, i64 3, !dbg !32
  %850 = call float @llvm.amdgcn.exp2.f32(float %849), !dbg !32
  %851 = insertelement <4 x float> %847, float %850, i64 3, !dbg !34
  %math.result275 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %851) #9, !dbg !34, !frisk.register_lifetimes !1
  %852 = fptrunc <4 x float> %math.result275 to <4 x half>, !dbg !34
  %math.packed276 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %852) #9, !dbg !33, !frisk.register_lifetimes !1
  %853 = fadd float %848, %850, !dbg !33
  %854 = bitcast float %853 to i32, !dbg !33
  %855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %680, i32 %854), !dbg !33
  %856 = bitcast i32 %855 to float, !dbg !33
  %857 = fadd float %853, %856, !dbg !33
  %858 = bitcast float %857 to i32, !dbg !33
  %859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %688, i32 %858), !dbg !33
  %860 = bitcast i32 %859 to float, !dbg !33
  %861 = fadd float %857, %860, !dbg !33
  %862 = fadd float %861, %718, !dbg !33
  %863 = insertelement <1 x float> poison, float %862, i64 0, !dbg !7
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread754 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !15, !range !39, !frisk.register_lifetimes !1
  %address.local755 = shl nuw nsw i32 %address.thread754, 3, !dbg !15
  %address.local756 = and i32 %address.local755, 120, !dbg !15
  %address.local757 = shl nuw nsw i32 %address.thread754, 5, !dbg !15
  %address.local758 = and i32 %address.local757, 1536, !dbg !15
  %address.local759 = or disjoint i32 %address.local756, %address.local758, !dbg !15
  %address.local760 = zext nneg i32 %address.local759 to i64, !dbg !15
  %address.local761 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local760, !dbg !15
  %864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local761, i32 0) #0, !dbg !15
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !15
  %address.local762 = zext nneg i32 %address.thread754 to i64, !dbg !8
  %address.local763 = and i64 %address.local762, 60, !dbg !15
  %address.local764 = add nuw nsw i64 %address.local763, %address.local762, !dbg !15
  %address.local765 = shl nuw nsw i64 %address.local764, 3, !dbg !15
  %address.local766 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local765, !dbg !15
  %address.local767 = and i64 %address.local762, 63, !dbg !11
  %address.local768 = lshr i64 %address.local767, 4, !dbg !15
  %address.local769 = mul nuw nsw i64 %address.local768, 384, !dbg !15
  %address.local770 = getelementptr [2 x i8], ptr addrspace(3) %address.local766, i64 %address.local769, !dbg !15
  %address.local771 = lshr i64 %address.local762, 2, !dbg !16
  %address.local772 = mul nsw i64 %address.local771, -32, !dbg !15
  %address.local773 = getelementptr [2 x i8], ptr addrspace(3) %address.local770, i64 %address.local772, !dbg !15
  %address.local774 = getelementptr i8, ptr addrspace(3) %address.local773, i64 256, !dbg !15
  %866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local774, i32 0) #0, !dbg !15
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !15
  %address.local775 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local765, !dbg !15
  %address.local776 = getelementptr [2 x i8], ptr addrspace(3) %address.local775, i64 %address.local769, !dbg !15
  %address.local777 = getelementptr [2 x i8], ptr addrspace(3) %address.local776, i64 %address.local772, !dbg !15
  %address.local778 = getelementptr i8, ptr addrspace(3) %address.local777, i64 512, !dbg !15
  %868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !15
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !15
  %address.local779 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local765, !dbg !15
  %address.local780 = getelementptr [2 x i8], ptr addrspace(3) %address.local779, i64 %address.local769, !dbg !15
  %address.local781 = getelementptr [2 x i8], ptr addrspace(3) %address.local780, i64 %address.local772, !dbg !15
  %address.local782 = getelementptr i8, ptr addrspace(3) %address.local781, i64 768, !dbg !15
  %870 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !15
  %871 = bitcast <4 x i32> %870 to <8 x half>, !dbg !15
  %address.local783 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local760, !dbg !15
  %address.local784 = getelementptr i8, ptr addrspace(3) %address.local783, i64 4096, !dbg !15
  %872 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local784, i32 0) #0, !dbg !15
  %873 = bitcast <4 x i32> %872 to <8 x half>, !dbg !15
  %address.local785 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local765, !dbg !15
  %address.local786 = getelementptr [2 x i8], ptr addrspace(3) %address.local785, i64 %address.local769, !dbg !15
  %address.local787 = getelementptr [2 x i8], ptr addrspace(3) %address.local786, i64 %address.local772, !dbg !15
  %address.local788 = getelementptr i8, ptr addrspace(3) %address.local787, i64 4352, !dbg !15
  %874 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local788, i32 0) #0, !dbg !15
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !15
  %address.local789 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local765, !dbg !15
  %address.local790 = getelementptr [2 x i8], ptr addrspace(3) %address.local789, i64 %address.local769, !dbg !15
  %address.local791 = getelementptr [2 x i8], ptr addrspace(3) %address.local790, i64 %address.local772, !dbg !15
  %address.local792 = getelementptr i8, ptr addrspace(3) %address.local791, i64 4608, !dbg !15
  %876 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local792, i32 0) #0, !dbg !15
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !15
  %address.local793 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local765, !dbg !15
  %address.local794 = getelementptr [2 x i8], ptr addrspace(3) %address.local793, i64 %address.local769, !dbg !15
  %address.local795 = getelementptr [2 x i8], ptr addrspace(3) %address.local794, i64 %address.local772, !dbg !15
  %address.local796 = getelementptr i8, ptr addrspace(3) %address.local795, i64 4864, !dbg !15
  %878 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local796, i32 0) #0, !dbg !15
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !15
  %880 = add nuw nsw i64 %158, %16, !dbg !18
  %881 = shl i64 %880, 19, !dbg !18
  %882 = and i64 %881, 16252928, !dbg !18
  %883 = add i64 %882, %164, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %884 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %883, !dbg !18
  %885 = getelementptr i8, ptr addrspace(1) %884, i64 64, !dbg !18
  %address.local797 = shl nuw nsw i64 %address.local762, 3, !dbg !8
  %address.local798 = and i64 %address.local797, 1536, !dbg !17
  %address.local799 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local798, !dbg !17
  %address.local800 = ptrtoint ptr addrspace(3) %address.local799 to i64, !dbg !17
  %address.local801 = trunc i64 %address.local800 to i32, !dbg !17
  %886 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local801), !dbg !18
  %887 = zext i32 %886 to i64, !dbg !18
  %888 = inttoptr i64 %887 to ptr addrspace(3), !dbg !18
  %address.local802 = shl nuw nsw i64 %address.local762, 5, !dbg !18
  %address.local803 = and i64 %address.local802, 8064, !dbg !18
  %address.local804 = shl nuw nsw i64 %address.local762, 3, !dbg !18
  %address.local805 = and i64 %address.local804, 24, !dbg !18
  %address.local806 = or disjoint i64 %address.local803, %address.local805, !dbg !18
  %address.local807 = getelementptr [2 x i8], ptr addrspace(1) %885, i64 %address.local806, !dbg !18
  %889 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %890 = ptrtoint ptr addrspace(1) %address.local807 to i64, !dbg !18
  %891 = sub i64 %890, %889, !dbg !18
  %dma.buffer.offset1172 = trunc i64 %891 to i32, !dbg !18
  %892 = trunc i64 %889 to i32, !dbg !18
  %893 = insertelement <4 x i32> poison, i32 %892, i64 0, !dbg !18
  %894 = lshr i64 %889, 32, !dbg !18
  %895 = trunc i64 %894 to i32, !dbg !18
  %896 = insertelement <4 x i32> %893, i32 %895, i64 1, !dbg !18
  %897 = insertelement <4 x i32> %896, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1173 = insertelement <4 x i32> %897, i32 131072, i64 3, !dbg !18
  %898 = ptrtoint ptr addrspace(3) %888 to i32, !dbg !18
  %dma.buffer.lds1174 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %898), !dbg !18
  %899 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1174, i32 %dma.buffer.offset1172, <4 x i32> %dma.buffer.rsrc1173) #10, !dbg !18
  %address.local808 = or disjoint i64 %address.local762, 256, !dbg !17
  %address.local809 = shl nuw nsw i64 %address.local808, 3, !dbg !17
  %address.local810 = and i64 %address.local809, 3584, !dbg !17
  %address.local811 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local810, !dbg !17
  %address.local812 = ptrtoint ptr addrspace(3) %address.local811 to i64, !dbg !17
  %address.local813 = trunc i64 %address.local812 to i32, !dbg !17
  %900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local813), !dbg !18
  %901 = zext i32 %900 to i64, !dbg !18
  %902 = inttoptr i64 %901 to ptr addrspace(3), !dbg !18
  %address.local814 = shl nuw nsw i64 %address.local808, 5, !dbg !18
  %address.local815 = and i64 %address.local814, 16256, !dbg !18
  %address.local816 = shl nuw nsw i64 %address.local762, 3, !dbg !18
  %address.local817 = and i64 %address.local816, 24, !dbg !18
  %address.local818 = or disjoint i64 %address.local815, %address.local817, !dbg !18
  %address.local819 = getelementptr [2 x i8], ptr addrspace(1) %885, i64 %address.local818, !dbg !18
  %903 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %904 = ptrtoint ptr addrspace(1) %address.local819 to i64, !dbg !18
  %905 = sub i64 %904, %903, !dbg !18
  %dma.buffer.offset1169 = trunc i64 %905 to i32, !dbg !18
  %906 = trunc i64 %903 to i32, !dbg !18
  %907 = insertelement <4 x i32> poison, i32 %906, i64 0, !dbg !18
  %908 = lshr i64 %903, 32, !dbg !18
  %909 = trunc i64 %908 to i32, !dbg !18
  %910 = insertelement <4 x i32> %907, i32 %909, i64 1, !dbg !18
  %911 = insertelement <4 x i32> %910, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1170 = insertelement <4 x i32> %911, i32 131072, i64 3, !dbg !18
  %912 = ptrtoint ptr addrspace(3) %902 to i32, !dbg !18
  %dma.buffer.lds1171 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %912), !dbg !18
  %913 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1171, i32 %dma.buffer.offset1169, <4 x i32> %dma.buffer.rsrc1170) #10, !dbg !18
  %914 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %914, <4 x float> %716) #9, !dbg !15
  %916 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %916, <4 x float> %915) #9, !dbg !15
  %918 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %918, <4 x float> %917) #9, !dbg !15
  %920 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %920, <4 x float> %919) #9, !dbg !15
  %922 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %922, <4 x float> %921) #9, !dbg !15
  %924 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %924, <4 x float> %923) #9, !dbg !15
  %926 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %926, <4 x float> %925) #9, !dbg !15
  %928 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %928, <4 x float> %927) #9, !dbg !15
  %930 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %930, <4 x float> %715) #9, !dbg !15
  %932 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %933 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %932, <4 x float> %931) #9, !dbg !15
  %934 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %935 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %934, <4 x float> %933) #9, !dbg !15
  %936 = shufflevector <8 x half> %871, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %936, <4 x float> %935) #9, !dbg !15
  %938 = shufflevector <8 x half> %873, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %939 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %938, <4 x float> %937) #9, !dbg !15
  %940 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %940, <4 x float> %939) #9, !dbg !15
  %942 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %942, <4 x float> %941) #9, !dbg !15
  %944 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %944, <4 x float> %943) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local761, i32 0) #0, !dbg !15
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !15
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local774, i32 0) #0, !dbg !15
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !15
  %950 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !15
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !15
  %952 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !15
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !15
  %954 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local784, i32 0) #0, !dbg !15
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !15
  %956 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local788, i32 0) #0, !dbg !15
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !15
  %958 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local792, i32 0) #0, !dbg !15
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !15
  %960 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local796, i32 0) #0, !dbg !15
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %962 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %883, !dbg !18
  %963 = getelementptr i8, ptr addrspace(1) %962, i64 128, !dbg !18
  %964 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local801), !dbg !18
  %965 = zext i32 %964 to i64, !dbg !18
  %966 = inttoptr i64 %965 to ptr addrspace(3), !dbg !18
  %address.local820 = getelementptr [2 x i8], ptr addrspace(1) %963, i64 %address.local806, !dbg !18
  %967 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %968 = ptrtoint ptr addrspace(1) %address.local820 to i64, !dbg !18
  %969 = sub i64 %968, %967, !dbg !18
  %dma.buffer.offset1166 = trunc i64 %969 to i32, !dbg !18
  %970 = trunc i64 %967 to i32, !dbg !18
  %971 = insertelement <4 x i32> poison, i32 %970, i64 0, !dbg !18
  %972 = lshr i64 %967, 32, !dbg !18
  %973 = trunc i64 %972 to i32, !dbg !18
  %974 = insertelement <4 x i32> %971, i32 %973, i64 1, !dbg !18
  %975 = insertelement <4 x i32> %974, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1167 = insertelement <4 x i32> %975, i32 131072, i64 3, !dbg !18
  %976 = ptrtoint ptr addrspace(3) %966 to i32, !dbg !18
  %dma.buffer.lds1168 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %976), !dbg !18
  %977 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1168, i32 %dma.buffer.offset1166, <4 x i32> %dma.buffer.rsrc1167) #10, !dbg !18
  %978 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local813), !dbg !18
  %979 = zext i32 %978 to i64, !dbg !18
  %980 = inttoptr i64 %979 to ptr addrspace(3), !dbg !18
  %address.local821 = getelementptr [2 x i8], ptr addrspace(1) %963, i64 %address.local818, !dbg !18
  %981 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %982 = ptrtoint ptr addrspace(1) %address.local821 to i64, !dbg !18
  %983 = sub i64 %982, %981, !dbg !18
  %dma.buffer.offset1163 = trunc i64 %983 to i32, !dbg !18
  %984 = trunc i64 %981 to i32, !dbg !18
  %985 = insertelement <4 x i32> poison, i32 %984, i64 0, !dbg !18
  %986 = lshr i64 %981, 32, !dbg !18
  %987 = trunc i64 %986 to i32, !dbg !18
  %988 = insertelement <4 x i32> %985, i32 %987, i64 1, !dbg !18
  %989 = insertelement <4 x i32> %988, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1164 = insertelement <4 x i32> %989, i32 131072, i64 3, !dbg !18
  %990 = ptrtoint ptr addrspace(3) %980 to i32, !dbg !18
  %dma.buffer.lds1165 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %990), !dbg !18
  %991 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1165, i32 %dma.buffer.offset1163, <4 x i32> %dma.buffer.rsrc1164) #10, !dbg !18
  %992 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %992, <4 x float> %714) #9, !dbg !15
  %994 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %994, <4 x float> %993) #9, !dbg !15
  %996 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %996, <4 x float> %995) #9, !dbg !15
  %998 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %998, <4 x float> %997) #9, !dbg !15
  %1000 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1000, <4 x float> %999) #9, !dbg !15
  %1002 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1002, <4 x float> %1001) #9, !dbg !15
  %1004 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1005 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1004, <4 x float> %1003) #9, !dbg !15
  %1006 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1007 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1006, <4 x float> %1005) #9, !dbg !15
  %1008 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1009 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1008, <4 x float> %713) #9, !dbg !15
  %1010 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1011 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1010, <4 x float> %1009) #9, !dbg !15
  %1012 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1013 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1012, <4 x float> %1011) #9, !dbg !15
  %1014 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1015 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1014, <4 x float> %1013) #9, !dbg !15
  %1016 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1017 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1016, <4 x float> %1015) #9, !dbg !15
  %1018 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1019 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1018, <4 x float> %1017) #9, !dbg !15
  %1020 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1020, <4 x float> %1019) #9, !dbg !15
  %1022 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1022, <4 x float> %1021) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1024 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local761, i32 0) #0, !dbg !15
  %1025 = bitcast <4 x i32> %1024 to <8 x half>, !dbg !15
  %1026 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local774, i32 0) #0, !dbg !15
  %1027 = bitcast <4 x i32> %1026 to <8 x half>, !dbg !15
  %1028 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !15
  %1029 = bitcast <4 x i32> %1028 to <8 x half>, !dbg !15
  %1030 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !15
  %1031 = bitcast <4 x i32> %1030 to <8 x half>, !dbg !15
  %1032 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local784, i32 0) #0, !dbg !15
  %1033 = bitcast <4 x i32> %1032 to <8 x half>, !dbg !15
  %1034 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local788, i32 0) #0, !dbg !15
  %1035 = bitcast <4 x i32> %1034 to <8 x half>, !dbg !15
  %1036 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local792, i32 0) #0, !dbg !15
  %1037 = bitcast <4 x i32> %1036 to <8 x half>, !dbg !15
  %1038 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local796, i32 0) #0, !dbg !15
  %1039 = bitcast <4 x i32> %1038 to <8 x half>, !dbg !15
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1040 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %883, !dbg !18
  %1041 = getelementptr i8, ptr addrspace(1) %1040, i64 192, !dbg !18
  %1042 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local801), !dbg !18
  %1043 = zext i32 %1042 to i64, !dbg !18
  %1044 = inttoptr i64 %1043 to ptr addrspace(3), !dbg !18
  %address.local822 = getelementptr [2 x i8], ptr addrspace(1) %1041, i64 %address.local806, !dbg !18
  %1045 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1046 = ptrtoint ptr addrspace(1) %address.local822 to i64, !dbg !18
  %1047 = sub i64 %1046, %1045, !dbg !18
  %dma.buffer.offset1160 = trunc i64 %1047 to i32, !dbg !18
  %1048 = trunc i64 %1045 to i32, !dbg !18
  %1049 = insertelement <4 x i32> poison, i32 %1048, i64 0, !dbg !18
  %1050 = lshr i64 %1045, 32, !dbg !18
  %1051 = trunc i64 %1050 to i32, !dbg !18
  %1052 = insertelement <4 x i32> %1049, i32 %1051, i64 1, !dbg !18
  %1053 = insertelement <4 x i32> %1052, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc1161 = insertelement <4 x i32> %1053, i32 131072, i64 3, !dbg !18
  %1054 = ptrtoint ptr addrspace(3) %1044 to i32, !dbg !18
  %dma.buffer.lds1162 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1054), !dbg !18
  %1055 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1162, i32 %dma.buffer.offset1160, <4 x i32> %dma.buffer.rsrc1161) #10, !dbg !18
  %1056 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local813), !dbg !18
  %1057 = zext i32 %1056 to i64, !dbg !18
  %1058 = inttoptr i64 %1057 to ptr addrspace(3), !dbg !18
  %address.local823 = getelementptr [2 x i8], ptr addrspace(1) %1041, i64 %address.local818, !dbg !18
  %1059 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %1060 = ptrtoint ptr addrspace(1) %address.local823 to i64, !dbg !18
  %1061 = sub i64 %1060, %1059, !dbg !18
  %dma.buffer.offset = trunc i64 %1061 to i32, !dbg !18
  %1062 = trunc i64 %1059 to i32, !dbg !18
  %1063 = insertelement <4 x i32> poison, i32 %1062, i64 0, !dbg !18
  %1064 = lshr i64 %1059, 32, !dbg !18
  %1065 = trunc i64 %1064 to i32, !dbg !18
  %1066 = insertelement <4 x i32> %1063, i32 %1065, i64 1, !dbg !18
  %1067 = insertelement <4 x i32> %1066, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %1067, i32 131072, i64 3, !dbg !18
  %1068 = ptrtoint ptr addrspace(3) %1058 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1068), !dbg !18
  %1069 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %1070 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1071 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1070, <4 x float> %712) #9, !dbg !15
  %1072 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1073 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1072, <4 x float> %1071) #9, !dbg !15
  %1074 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1075 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1074, <4 x float> %1073) #9, !dbg !15
  %1076 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1077 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1076, <4 x float> %1075) #9, !dbg !15
  %1078 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1079 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1078, <4 x float> %1077) #9, !dbg !15
  %1080 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1081 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1080, <4 x float> %1079) #9, !dbg !15
  %1082 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1083 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1082, <4 x float> %1081) #9, !dbg !15
  %1084 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1085 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1084, <4 x float> %1083) #9, !dbg !15
  %1086 = shufflevector <8 x half> %1025, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1087 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1086, <4 x float> %711) #9, !dbg !15
  %1088 = shufflevector <8 x half> %1027, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1089 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1088, <4 x float> %1087) #9, !dbg !15
  %1090 = shufflevector <8 x half> %1029, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1091 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1090, <4 x float> %1089) #9, !dbg !15
  %1092 = shufflevector <8 x half> %1031, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1093 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1092, <4 x float> %1091) #9, !dbg !15
  %1094 = shufflevector <8 x half> %1033, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1095 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1094, <4 x float> %1093) #9, !dbg !15
  %1096 = shufflevector <8 x half> %1035, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1097 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1096, <4 x float> %1095) #9, !dbg !15
  %1098 = shufflevector <8 x half> %1037, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1099 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1098, <4 x float> %1097) #9, !dbg !15
  %1100 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1101 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1100, <4 x float> %1099) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1102 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local761, i32 0) #0, !dbg !15
  %1103 = bitcast <4 x i32> %1102 to <8 x half>, !dbg !15
  %1104 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local774, i32 0) #0, !dbg !15
  %1105 = bitcast <4 x i32> %1104 to <8 x half>, !dbg !15
  %1106 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local778, i32 0) #0, !dbg !15
  %1107 = bitcast <4 x i32> %1106 to <8 x half>, !dbg !15
  %1108 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local782, i32 0) #0, !dbg !15
  %1109 = bitcast <4 x i32> %1108 to <8 x half>, !dbg !15
  %1110 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local784, i32 0) #0, !dbg !15
  %1111 = bitcast <4 x i32> %1110 to <8 x half>, !dbg !15
  %1112 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local788, i32 0) #0, !dbg !15
  %1113 = bitcast <4 x i32> %1112 to <8 x half>, !dbg !15
  %1114 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local792, i32 0) #0, !dbg !15
  %1115 = bitcast <4 x i32> %1114 to <8 x half>, !dbg !15
  %1116 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local796, i32 0) #0, !dbg !15
  %1117 = bitcast <4 x i32> %1116 to <8 x half>, !dbg !15
  %1118 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1119 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1118, <4 x float> %710) #9, !dbg !15
  %1120 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1121 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1120, <4 x float> %1119) #9, !dbg !15
  %1122 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1123 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1122, <4 x float> %1121) #9, !dbg !15
  %1124 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1125 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1124, <4 x float> %1123) #9, !dbg !15
  %1126 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1127 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1126, <4 x float> %1125) #9, !dbg !15
  %1128 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1128, <4 x float> %1127) #9, !dbg !15
  %1130 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1131 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1130, <4 x float> %1129) #9, !dbg !15
  %1132 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !15
  %1133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1132, <4 x float> %1131) #9, !dbg !15
  %1134 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1135 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1134, <4 x float> %709) #9, !dbg !15
  %1136 = shufflevector <8 x half> %1105, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1136, <4 x float> %1135) #9, !dbg !15
  %1138 = shufflevector <8 x half> %1107, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1138, <4 x float> %1137) #9, !dbg !15
  %1140 = shufflevector <8 x half> %1109, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1140, <4 x float> %1139) #9, !dbg !15
  %1142 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1142, <4 x float> %1141) #9, !dbg !15
  %1144 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1144, <4 x float> %1143) #9, !dbg !15
  %1146 = shufflevector <8 x half> %1115, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1146, <4 x float> %1145) #9, !dbg !15
  %1148 = shufflevector <8 x half> %1117, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed276, <4 x half> %1148, <4 x float> %1147) #9, !dbg !15
  %1150 = shufflevector <4 x float> %929, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1151 = shufflevector <4 x float> %1007, <4 x float> %1023, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1152 = shufflevector <4 x float> %1085, <4 x float> %1101, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1153 = shufflevector <4 x float> %1133, <4 x float> %1149, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1154 = add nuw nsw i64 %138, 1, !dbg !7
  br label %137, !dbg !7

1155:                                             ; preds = %137
  %1156 = and i32 %31, 16, !dbg !7
  %.not = icmp eq i32 %1156, 0, !dbg !7
  %1157 = and i32 %31, 32, !dbg !7
  %.not36 = icmp eq i32 %1157, 0, !dbg !7
  %1158 = extractelement <8 x float> %152, i64 0, !dbg !7
  %1159 = extractelement <8 x float> %152, i64 1, !dbg !7
  %1160 = extractelement <8 x float> %152, i64 2, !dbg !7
  %1161 = extractelement <8 x float> %152, i64 3, !dbg !7
  %1162 = select i1 %.not, float %1159, float %1158, !dbg !7
  %1163 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %1164 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1163), !dbg !7
  %1165 = add nuw nsw i32 %1164, 64, !dbg !7
  %1166 = xor i32 %1164, 16, !dbg !7
  %1167 = and i32 %1165, 192, !dbg !7
  %1168 = icmp samesign ult i32 %1166, %1167, !dbg !7
  %1169 = select i1 %1168, i32 %1166, i32 %1164, !dbg !7
  %1170 = shl nuw nsw i32 %1169, 2, !dbg !7
  %1171 = bitcast float %1162 to i32, !dbg !7
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1171), !dbg !7
  %1173 = bitcast i32 %1172 to float, !dbg !7
  %1174 = select i1 %.not, float %1158, float %1173, !dbg !7
  %1175 = select i1 %.not, float %1173, float %1159, !dbg !7
  %1176 = select i1 %.not, float %1161, float %1160, !dbg !7
  %1177 = bitcast float %1176 to i32, !dbg !7
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1177), !dbg !7
  %1179 = bitcast i32 %1178 to float, !dbg !7
  %1180 = select i1 %.not, float %1160, float %1179, !dbg !7
  %1181 = select i1 %.not, float %1179, float %1161, !dbg !7
  %1182 = select i1 %.not36, float %1180, float %1174, !dbg !7
  %1183 = xor i32 %1164, 32, !dbg !7
  %1184 = icmp samesign ult i32 %1183, %1167, !dbg !7
  %1185 = select i1 %1184, i32 %1183, i32 %1164, !dbg !7
  %1186 = shl nuw nsw i32 %1185, 2, !dbg !7
  %1187 = bitcast float %1182 to i32, !dbg !7
  %1188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1187), !dbg !7
  %1189 = bitcast i32 %1188 to float, !dbg !7
  %1190 = select i1 %.not36, float %1174, float %1189, !dbg !7
  %1191 = select i1 %.not36, float %1189, float %1180, !dbg !7
  %1192 = select i1 %.not36, float %1181, float %1175, !dbg !7
  %1193 = bitcast float %1192 to i32, !dbg !7
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1193), !dbg !7
  %1195 = bitcast i32 %1194 to float, !dbg !7
  %1196 = select i1 %.not36, float %1175, float %1195, !dbg !7
  %1197 = select i1 %.not36, float %1195, float %1181, !dbg !7
  %1198 = extractelement <8 x float> %149, i64 0, !dbg !7
  %1199 = extractelement <8 x float> %149, i64 1, !dbg !7
  %1200 = extractelement <8 x float> %149, i64 2, !dbg !7
  %1201 = extractelement <8 x float> %149, i64 3, !dbg !7
  %1202 = select i1 %.not, float %1199, float %1198, !dbg !7
  %1203 = bitcast float %1202 to i32, !dbg !7
  %1204 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1203), !dbg !7
  %1205 = bitcast i32 %1204 to float, !dbg !7
  %1206 = select i1 %.not, float %1198, float %1205, !dbg !7
  %1207 = select i1 %.not, float %1205, float %1199, !dbg !7
  %1208 = select i1 %.not, float %1201, float %1200, !dbg !7
  %1209 = bitcast float %1208 to i32, !dbg !7
  %1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1209), !dbg !7
  %1211 = bitcast i32 %1210 to float, !dbg !7
  %1212 = select i1 %.not, float %1200, float %1211, !dbg !7
  %1213 = select i1 %.not, float %1211, float %1201, !dbg !7
  %1214 = select i1 %.not36, float %1212, float %1206, !dbg !7
  %1215 = bitcast float %1214 to i32, !dbg !7
  %1216 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1215), !dbg !7
  %1217 = bitcast i32 %1216 to float, !dbg !7
  %1218 = select i1 %.not36, float %1206, float %1217, !dbg !7
  %1219 = select i1 %.not36, float %1217, float %1212, !dbg !7
  %1220 = select i1 %.not36, float %1213, float %1207, !dbg !7
  %1221 = bitcast float %1220 to i32, !dbg !7
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1221), !dbg !7
  %1223 = bitcast i32 %1222 to float, !dbg !7
  %1224 = select i1 %.not36, float %1207, float %1223, !dbg !7
  %1225 = select i1 %.not36, float %1223, float %1213, !dbg !7
  %1226 = extractelement <8 x float> %146, i64 0, !dbg !7
  %1227 = extractelement <8 x float> %146, i64 1, !dbg !7
  %1228 = extractelement <8 x float> %146, i64 2, !dbg !7
  %1229 = extractelement <8 x float> %146, i64 3, !dbg !7
  %1230 = select i1 %.not, float %1227, float %1226, !dbg !7
  %1231 = bitcast float %1230 to i32, !dbg !7
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1231), !dbg !7
  %1233 = bitcast i32 %1232 to float, !dbg !7
  %1234 = select i1 %.not, float %1226, float %1233, !dbg !7
  %1235 = select i1 %.not, float %1233, float %1227, !dbg !7
  %1236 = select i1 %.not, float %1229, float %1228, !dbg !7
  %1237 = bitcast float %1236 to i32, !dbg !7
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1237), !dbg !7
  %1239 = bitcast i32 %1238 to float, !dbg !7
  %1240 = select i1 %.not, float %1228, float %1239, !dbg !7
  %1241 = select i1 %.not, float %1239, float %1229, !dbg !7
  %1242 = select i1 %.not36, float %1240, float %1234, !dbg !7
  %1243 = bitcast float %1242 to i32, !dbg !7
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1243), !dbg !7
  %1245 = bitcast i32 %1244 to float, !dbg !7
  %1246 = select i1 %.not36, float %1234, float %1245, !dbg !7
  %1247 = select i1 %.not36, float %1245, float %1240, !dbg !7
  %1248 = select i1 %.not36, float %1241, float %1235, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1249), !dbg !7
  %1251 = bitcast i32 %1250 to float, !dbg !7
  %1252 = select i1 %.not36, float %1235, float %1251, !dbg !7
  %1253 = select i1 %.not36, float %1251, float %1241, !dbg !7
  %1254 = extractelement <8 x float> %141, i64 0, !dbg !7
  %1255 = extractelement <8 x float> %141, i64 1, !dbg !7
  %1256 = extractelement <8 x float> %141, i64 2, !dbg !7
  %1257 = extractelement <8 x float> %141, i64 3, !dbg !7
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !7
  %1259 = bitcast float %1258 to i32, !dbg !7
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1259), !dbg !7
  %1261 = bitcast i32 %1260 to float, !dbg !7
  %1262 = select i1 %.not, float %1254, float %1261, !dbg !7
  %1263 = select i1 %.not, float %1261, float %1255, !dbg !7
  %1264 = select i1 %.not, float %1257, float %1256, !dbg !7
  %1265 = bitcast float %1264 to i32, !dbg !7
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1265), !dbg !7
  %1267 = bitcast i32 %1266 to float, !dbg !7
  %1268 = select i1 %.not, float %1256, float %1267, !dbg !7
  %1269 = select i1 %.not, float %1267, float %1257, !dbg !7
  %1270 = select i1 %.not36, float %1268, float %1262, !dbg !7
  %1271 = bitcast float %1270 to i32, !dbg !7
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1271), !dbg !7
  %1273 = select i1 %.not36, float %1269, float %1263, !dbg !7
  %1274 = bitcast float %1273 to i32, !dbg !7
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1274), !dbg !7
  %1276 = shufflevector <1 x float> %142, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1277 = extractelement <8 x float> %152, i64 4, !dbg !7
  %1278 = extractelement <8 x float> %152, i64 5, !dbg !7
  %1279 = extractelement <8 x float> %152, i64 6, !dbg !7
  %1280 = extractelement <8 x float> %152, i64 7, !dbg !7
  %1281 = select i1 %.not, float %1278, float %1277, !dbg !7
  %1282 = bitcast float %1281 to i32, !dbg !7
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1282), !dbg !7
  %1284 = bitcast i32 %1283 to float, !dbg !7
  %1285 = select i1 %.not, float %1277, float %1284, !dbg !7
  %1286 = select i1 %.not, float %1284, float %1278, !dbg !7
  %1287 = select i1 %.not, float %1280, float %1279, !dbg !7
  %1288 = bitcast float %1287 to i32, !dbg !7
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1288), !dbg !7
  %1290 = bitcast i32 %1289 to float, !dbg !7
  %1291 = select i1 %.not, float %1279, float %1290, !dbg !7
  %1292 = select i1 %.not, float %1290, float %1280, !dbg !7
  %1293 = select i1 %.not36, float %1291, float %1285, !dbg !7
  %1294 = bitcast float %1293 to i32, !dbg !7
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1294), !dbg !7
  %1296 = bitcast i32 %1295 to float, !dbg !7
  %1297 = select i1 %.not36, float %1285, float %1296, !dbg !7
  %1298 = select i1 %.not36, float %1296, float %1291, !dbg !7
  %1299 = select i1 %.not36, float %1292, float %1286, !dbg !7
  %1300 = bitcast float %1299 to i32, !dbg !7
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1300), !dbg !7
  %1302 = bitcast i32 %1301 to float, !dbg !7
  %1303 = select i1 %.not36, float %1286, float %1302, !dbg !7
  %1304 = select i1 %.not36, float %1302, float %1292, !dbg !7
  %1305 = extractelement <8 x float> %149, i64 4, !dbg !7
  %1306 = extractelement <8 x float> %149, i64 5, !dbg !7
  %1307 = extractelement <8 x float> %149, i64 6, !dbg !7
  %1308 = extractelement <8 x float> %149, i64 7, !dbg !7
  %1309 = select i1 %.not, float %1306, float %1305, !dbg !7
  %1310 = bitcast float %1309 to i32, !dbg !7
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1310), !dbg !7
  %1312 = bitcast i32 %1311 to float, !dbg !7
  %1313 = select i1 %.not, float %1305, float %1312, !dbg !7
  %1314 = select i1 %.not, float %1312, float %1306, !dbg !7
  %1315 = select i1 %.not, float %1308, float %1307, !dbg !7
  %1316 = bitcast float %1315 to i32, !dbg !7
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1316), !dbg !7
  %1318 = bitcast i32 %1317 to float, !dbg !7
  %1319 = select i1 %.not, float %1307, float %1318, !dbg !7
  %1320 = select i1 %.not, float %1318, float %1308, !dbg !7
  %1321 = select i1 %.not36, float %1319, float %1313, !dbg !7
  %1322 = bitcast float %1321 to i32, !dbg !7
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1322), !dbg !7
  %1324 = bitcast i32 %1323 to float, !dbg !7
  %1325 = select i1 %.not36, float %1313, float %1324, !dbg !7
  %1326 = select i1 %.not36, float %1324, float %1319, !dbg !7
  %1327 = select i1 %.not36, float %1320, float %1314, !dbg !7
  %1328 = bitcast float %1327 to i32, !dbg !7
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1328), !dbg !7
  %1330 = bitcast i32 %1329 to float, !dbg !7
  %1331 = select i1 %.not36, float %1314, float %1330, !dbg !7
  %1332 = select i1 %.not36, float %1330, float %1320, !dbg !7
  %1333 = extractelement <8 x float> %146, i64 4, !dbg !7
  %1334 = extractelement <8 x float> %146, i64 5, !dbg !7
  %1335 = extractelement <8 x float> %146, i64 6, !dbg !7
  %1336 = extractelement <8 x float> %146, i64 7, !dbg !7
  %1337 = select i1 %.not, float %1334, float %1333, !dbg !7
  %1338 = bitcast float %1337 to i32, !dbg !7
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1338), !dbg !7
  %1340 = bitcast i32 %1339 to float, !dbg !7
  %1341 = select i1 %.not, float %1333, float %1340, !dbg !7
  %1342 = select i1 %.not, float %1340, float %1334, !dbg !7
  %1343 = select i1 %.not, float %1336, float %1335, !dbg !7
  %1344 = bitcast float %1343 to i32, !dbg !7
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1344), !dbg !7
  %1346 = bitcast i32 %1345 to float, !dbg !7
  %1347 = select i1 %.not, float %1335, float %1346, !dbg !7
  %1348 = select i1 %.not, float %1346, float %1336, !dbg !7
  %1349 = select i1 %.not36, float %1347, float %1341, !dbg !7
  %1350 = bitcast float %1349 to i32, !dbg !7
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1350), !dbg !7
  %1352 = bitcast i32 %1351 to float, !dbg !7
  %1353 = select i1 %.not36, float %1341, float %1352, !dbg !7
  %1354 = select i1 %.not36, float %1352, float %1347, !dbg !7
  %1355 = select i1 %.not36, float %1348, float %1342, !dbg !7
  %1356 = bitcast float %1355 to i32, !dbg !7
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1356), !dbg !7
  %1358 = bitcast i32 %1357 to float, !dbg !7
  %1359 = select i1 %.not36, float %1342, float %1358, !dbg !7
  %1360 = select i1 %.not36, float %1358, float %1348, !dbg !7
  %1361 = extractelement <8 x float> %141, i64 4, !dbg !7
  %1362 = extractelement <8 x float> %141, i64 5, !dbg !7
  %1363 = extractelement <8 x float> %141, i64 6, !dbg !7
  %1364 = extractelement <8 x float> %141, i64 7, !dbg !7
  %1365 = select i1 %.not, float %1362, float %1361, !dbg !7
  %1366 = bitcast float %1365 to i32, !dbg !7
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1366), !dbg !7
  %1368 = bitcast i32 %1367 to float, !dbg !7
  %1369 = select i1 %.not, float %1361, float %1368, !dbg !7
  %1370 = select i1 %.not, float %1368, float %1362, !dbg !7
  %1371 = select i1 %.not, float %1364, float %1363, !dbg !7
  %1372 = bitcast float %1371 to i32, !dbg !7
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1170, i32 %1372), !dbg !7
  %1374 = bitcast i32 %1373 to float, !dbg !7
  %1375 = select i1 %.not, float %1363, float %1374, !dbg !7
  %1376 = select i1 %.not, float %1374, float %1364, !dbg !7
  %1377 = select i1 %.not36, float %1375, float %1369, !dbg !7
  %1378 = bitcast float %1377 to i32, !dbg !7
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1378), !dbg !7
  %1380 = select i1 %.not36, float %1376, float %1370, !dbg !7
  %1381 = bitcast float %1380 to i32, !dbg !7
  %1382 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1186, i32 %1381), !dbg !7
  br i1 true, label %1383, label %1417, !dbg !21

1383:                                             ; preds = %1155
  %1384 = bitcast i32 %1379 to float, !dbg !7
  %1385 = select i1 %.not36, float %1369, float %1384, !dbg !7
  %1386 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !19
  %1387 = bitcast i32 %1382 to float, !dbg !7
  %1388 = select i1 %.not36, float %1370, float %1387, !dbg !7
  %1389 = insertelement <4 x float> %1386, float %1388, i64 1, !dbg !19
  %1390 = select i1 %.not36, float %1384, float %1375, !dbg !7
  %1391 = insertelement <4 x float> %1389, float %1390, i64 2, !dbg !19
  %1392 = select i1 %.not36, float %1387, float %1376, !dbg !7
  %1393 = insertelement <4 x float> %1391, float %1392, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !19
  %normalized = fmul <4 x float> %1393, %row.reciprocal, !dbg !19
  %1394 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1395 = extractelement <4 x half> %1394, i64 3, !dbg !20
  %1396 = extractelement <4 x half> %1394, i64 2, !dbg !20
  %1397 = extractelement <4 x half> %1394, i64 1, !dbg !20
  %1398 = extractelement <4 x half> %1394, i64 0, !dbg !20
  %1399 = bitcast i32 %1272 to float, !dbg !7
  %1400 = select i1 %.not36, float %1262, float %1399, !dbg !7
  %1401 = insertelement <4 x float> poison, float %1400, i64 0, !dbg !19
  %1402 = bitcast i32 %1275 to float, !dbg !7
  %1403 = select i1 %.not36, float %1263, float %1402, !dbg !7
  %1404 = insertelement <4 x float> %1401, float %1403, i64 1, !dbg !19
  %1405 = select i1 %.not36, float %1399, float %1268, !dbg !7
  %1406 = insertelement <4 x float> %1404, float %1405, i64 2, !dbg !19
  %1407 = select i1 %.not36, float %1402, float %1269, !dbg !7
  %1408 = insertelement <4 x float> %1406, float %1407, i64 3, !dbg !19
  %normalized72 = fmul <4 x float> %1408, %row.reciprocal, !dbg !19
  %1409 = fptrunc <4 x float> %normalized72 to <4 x half>, !dbg !20
  %1410 = extractelement <4 x half> %1409, i64 3, !dbg !20
  %1411 = extractelement <4 x half> %1409, i64 2, !dbg !20
  %1412 = extractelement <4 x half> %1409, i64 1, !dbg !20
  %1413 = extractelement <4 x half> %1409, i64 0, !dbg !20
  %1414 = add nsw i64 %16, %22, !dbg !21
  %1415 = shl i64 %1414, 19, !dbg !21
  %1416 = and i64 %1415, 16252928, !dbg !21
  %address.thread824 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local825 = zext nneg i32 %address.thread824 to i64, !dbg !8
  %address.local826 = add i64 %25, %address.local825, !dbg !21
  %address.local827 = lshr i64 %address.local825, 6, !dbg !11
  %address.local828 = shl nuw nsw i64 %address.local827, 4, !dbg !21
  %address.local829 = add i64 %address.local826, %address.local828, !dbg !21
  %address.local830 = and i64 %address.local825, 240, !dbg !21
  %address.local831 = sub i64 %address.local829, %address.local830, !dbg !21
  %address.local832 = add i64 %address.local831, %26, !dbg !21
  %address.local833 = shl nuw nsw i64 %address.local832, 7, !dbg !21
  %address.local834 = add nuw nsw i64 %1416, %address.local833, !dbg !21
  %address.local835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local836 = lshr i64 %address.local825, 2, !dbg !11
  %address.local837 = and i64 %address.local836, 12, !dbg !11
  %address.local838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local835, i64 %address.local837, !dbg !21
  store half %1413, ptr addrspace(1) %address.local838, align 2, !dbg !21
  %address.local839 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local840 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local839, i64 %address.local837, !dbg !21
  %address.local841 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local840, i64 2, !dbg !21
  store half %1412, ptr addrspace(1) %address.local841, align 2, !dbg !21
  %address.local842 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local843 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local842, i64 %address.local837, !dbg !21
  %address.local844 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local843, i64 4, !dbg !21
  store half %1411, ptr addrspace(1) %address.local844, align 2, !dbg !21
  %address.local845 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local846 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local845, i64 %address.local837, !dbg !21
  %address.local847 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local846, i64 6, !dbg !21
  store half %1410, ptr addrspace(1) %address.local847, align 2, !dbg !21
  %address.local848 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local849 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local837, !dbg !21
  %address.local850 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local849, i64 32, !dbg !21
  store half %1398, ptr addrspace(1) %address.local850, align 2, !dbg !21
  %address.local851 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local852 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local851, i64 %address.local837, !dbg !21
  %address.local853 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local852, i64 34, !dbg !21
  store half %1397, ptr addrspace(1) %address.local853, align 2, !dbg !21
  %address.local854 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local854, i64 %address.local837, !dbg !21
  %address.local856 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local855, i64 36, !dbg !21
  store half %1396, ptr addrspace(1) %address.local856, align 2, !dbg !21
  %address.local857 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local834, !dbg !21
  %address.local858 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local857, i64 %address.local837, !dbg !21
  %address.local859 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local858, i64 38, !dbg !21
  store half %1395, ptr addrspace(1) %address.local859, align 2, !dbg !21
  br label %1417, !dbg !21

1417:                                             ; preds = %1383, %1155
  br i1 true, label %1418, label %1438, !dbg !21

1418:                                             ; preds = %1417
  %1419 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !19
  %1420 = insertelement <4 x float> %1419, float %1359, i64 1, !dbg !19
  %1421 = insertelement <4 x float> %1420, float %1354, i64 2, !dbg !19
  %1422 = insertelement <4 x float> %1421, float %1360, i64 3, !dbg !19
  %row.reciprocal73 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !19
  %normalized74 = fmul <4 x float> %1422, %row.reciprocal73, !dbg !19
  %1423 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %1424 = extractelement <4 x half> %1423, i64 3, !dbg !20
  %1425 = extractelement <4 x half> %1423, i64 2, !dbg !20
  %1426 = extractelement <4 x half> %1423, i64 1, !dbg !20
  %1427 = extractelement <4 x half> %1423, i64 0, !dbg !20
  %1428 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !19
  %1429 = insertelement <4 x float> %1428, float %1252, i64 1, !dbg !19
  %1430 = insertelement <4 x float> %1429, float %1247, i64 2, !dbg !19
  %1431 = insertelement <4 x float> %1430, float %1253, i64 3, !dbg !19
  %normalized75 = fmul <4 x float> %1431, %row.reciprocal73, !dbg !19
  %1432 = fptrunc <4 x float> %normalized75 to <4 x half>, !dbg !20
  %1433 = extractelement <4 x half> %1432, i64 3, !dbg !20
  %1434 = extractelement <4 x half> %1432, i64 2, !dbg !20
  %1435 = extractelement <4 x half> %1432, i64 1, !dbg !20
  %1436 = extractelement <4 x half> %1432, i64 0, !dbg !20
  %1437 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread860 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local861 = zext nneg i32 %address.thread860 to i64, !dbg !8
  %address.local862 = add i64 %25, %address.local861, !dbg !21
  %address.local863 = lshr i64 %address.local861, 6, !dbg !11
  %address.local864 = shl nuw nsw i64 %address.local863, 4, !dbg !21
  %address.local865 = add i64 %address.local862, %address.local864, !dbg !21
  %address.local866 = and i64 %address.local861, 240, !dbg !21
  %address.local867 = sub i64 %address.local865, %address.local866, !dbg !21
  %address.local868 = add i64 %address.local867, %26, !dbg !21
  %address.local869 = shl nuw nsw i64 %address.local868, 7, !dbg !21
  %address.local870 = add nuw nsw i64 %1437, %address.local869, !dbg !21
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local872 = lshr i64 %address.local861, 2, !dbg !11
  %address.local873 = and i64 %address.local872, 12, !dbg !11
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local871, i64 %address.local873, !dbg !21
  %address.local875 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local874, i64 64, !dbg !21
  store half %1436, ptr addrspace(1) %address.local875, align 2, !dbg !21
  %address.local876 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local873, !dbg !21
  %address.local878 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local877, i64 66, !dbg !21
  store half %1435, ptr addrspace(1) %address.local878, align 2, !dbg !21
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local879, i64 %address.local873, !dbg !21
  %address.local881 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local880, i64 68, !dbg !21
  store half %1434, ptr addrspace(1) %address.local881, align 2, !dbg !21
  %address.local882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local882, i64 %address.local873, !dbg !21
  %address.local884 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local883, i64 70, !dbg !21
  store half %1433, ptr addrspace(1) %address.local884, align 2, !dbg !21
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local885, i64 %address.local873, !dbg !21
  %address.local887 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local886, i64 96, !dbg !21
  store half %1427, ptr addrspace(1) %address.local887, align 2, !dbg !21
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local888, i64 %address.local873, !dbg !21
  %address.local890 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local889, i64 98, !dbg !21
  store half %1426, ptr addrspace(1) %address.local890, align 2, !dbg !21
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local891, i64 %address.local873, !dbg !21
  %address.local893 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local892, i64 100, !dbg !21
  store half %1425, ptr addrspace(1) %address.local893, align 2, !dbg !21
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local870, !dbg !21
  %address.local895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local873, !dbg !21
  %address.local896 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local895, i64 102, !dbg !21
  store half %1424, ptr addrspace(1) %address.local896, align 2, !dbg !21
  br label %1438, !dbg !21

1438:                                             ; preds = %1418, %1417
  br i1 true, label %1439, label %1459, !dbg !21

1439:                                             ; preds = %1438
  %1440 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !19
  %1441 = insertelement <4 x float> %1440, float %1331, i64 1, !dbg !19
  %1442 = insertelement <4 x float> %1441, float %1326, i64 2, !dbg !19
  %1443 = insertelement <4 x float> %1442, float %1332, i64 3, !dbg !19
  %row.reciprocal76 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !19
  %normalized77 = fmul <4 x float> %1443, %row.reciprocal76, !dbg !19
  %1444 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %1445 = extractelement <4 x half> %1444, i64 3, !dbg !20
  %1446 = extractelement <4 x half> %1444, i64 2, !dbg !20
  %1447 = extractelement <4 x half> %1444, i64 1, !dbg !20
  %1448 = extractelement <4 x half> %1444, i64 0, !dbg !20
  %1449 = insertelement <4 x float> poison, float %1218, i64 0, !dbg !19
  %1450 = insertelement <4 x float> %1449, float %1224, i64 1, !dbg !19
  %1451 = insertelement <4 x float> %1450, float %1219, i64 2, !dbg !19
  %1452 = insertelement <4 x float> %1451, float %1225, i64 3, !dbg !19
  %normalized78 = fmul <4 x float> %1452, %row.reciprocal76, !dbg !19
  %1453 = fptrunc <4 x float> %normalized78 to <4 x half>, !dbg !20
  %1454 = extractelement <4 x half> %1453, i64 3, !dbg !20
  %1455 = extractelement <4 x half> %1453, i64 2, !dbg !20
  %1456 = extractelement <4 x half> %1453, i64 1, !dbg !20
  %1457 = extractelement <4 x half> %1453, i64 0, !dbg !20
  %1458 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread897 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local898 = zext nneg i32 %address.thread897 to i64, !dbg !8
  %address.local899 = add i64 %25, %address.local898, !dbg !21
  %address.local900 = lshr i64 %address.local898, 6, !dbg !11
  %address.local901 = shl nuw nsw i64 %address.local900, 4, !dbg !21
  %address.local902 = add i64 %address.local899, %address.local901, !dbg !21
  %address.local903 = and i64 %address.local898, 240, !dbg !21
  %address.local904 = sub i64 %address.local902, %address.local903, !dbg !21
  %address.local905 = add i64 %address.local904, %26, !dbg !21
  %address.local906 = shl nuw nsw i64 %address.local905, 7, !dbg !21
  %address.local907 = add nuw nsw i64 %1458, %address.local906, !dbg !21
  %address.local908 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local909 = lshr i64 %address.local898, 2, !dbg !11
  %address.local910 = and i64 %address.local909, 12, !dbg !11
  %address.local911 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local908, i64 %address.local910, !dbg !21
  %address.local912 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local911, i64 128, !dbg !21
  store half %1457, ptr addrspace(1) %address.local912, align 2, !dbg !21
  %address.local913 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local914 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local913, i64 %address.local910, !dbg !21
  %address.local915 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local914, i64 130, !dbg !21
  store half %1456, ptr addrspace(1) %address.local915, align 2, !dbg !21
  %address.local916 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local917 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local916, i64 %address.local910, !dbg !21
  %address.local918 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local917, i64 132, !dbg !21
  store half %1455, ptr addrspace(1) %address.local918, align 2, !dbg !21
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local920 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local919, i64 %address.local910, !dbg !21
  %address.local921 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local920, i64 134, !dbg !21
  store half %1454, ptr addrspace(1) %address.local921, align 2, !dbg !21
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local910, !dbg !21
  %address.local924 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local923, i64 160, !dbg !21
  store half %1448, ptr addrspace(1) %address.local924, align 2, !dbg !21
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local926 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local925, i64 %address.local910, !dbg !21
  %address.local927 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local926, i64 162, !dbg !21
  store half %1447, ptr addrspace(1) %address.local927, align 2, !dbg !21
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local928, i64 %address.local910, !dbg !21
  %address.local930 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local929, i64 164, !dbg !21
  store half %1446, ptr addrspace(1) %address.local930, align 2, !dbg !21
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local907, !dbg !21
  %address.local932 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local931, i64 %address.local910, !dbg !21
  %address.local933 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local932, i64 166, !dbg !21
  store half %1445, ptr addrspace(1) %address.local933, align 2, !dbg !21
  br label %1459, !dbg !21

1459:                                             ; preds = %1439, %1438
  br i1 true, label %1460, label %1480, !dbg !21

1460:                                             ; preds = %1459
  %1461 = insertelement <4 x float> poison, float %1297, i64 0, !dbg !19
  %1462 = insertelement <4 x float> %1461, float %1303, i64 1, !dbg !19
  %1463 = insertelement <4 x float> %1462, float %1298, i64 2, !dbg !19
  %1464 = insertelement <4 x float> %1463, float %1304, i64 3, !dbg !19
  %row.reciprocal79 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1276, !dbg !19
  %normalized80 = fmul <4 x float> %1464, %row.reciprocal79, !dbg !19
  %1465 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %1466 = extractelement <4 x half> %1465, i64 3, !dbg !20
  %1467 = extractelement <4 x half> %1465, i64 2, !dbg !20
  %1468 = extractelement <4 x half> %1465, i64 1, !dbg !20
  %1469 = extractelement <4 x half> %1465, i64 0, !dbg !20
  %1470 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !19
  %1471 = insertelement <4 x float> %1470, float %1196, i64 1, !dbg !19
  %1472 = insertelement <4 x float> %1471, float %1191, i64 2, !dbg !19
  %1473 = insertelement <4 x float> %1472, float %1197, i64 3, !dbg !19
  %normalized81 = fmul <4 x float> %1473, %row.reciprocal79, !dbg !19
  %1474 = fptrunc <4 x float> %normalized81 to <4 x half>, !dbg !20
  %1475 = extractelement <4 x half> %1474, i64 3, !dbg !20
  %1476 = extractelement <4 x half> %1474, i64 2, !dbg !20
  %1477 = extractelement <4 x half> %1474, i64 1, !dbg !20
  %1478 = extractelement <4 x half> %1474, i64 0, !dbg !20
  %1479 = shl nuw nsw i64 %24, 19, !dbg !21
  %address.thread934 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !21, !range !39, !frisk.register_lifetimes !1
  %address.local935 = zext nneg i32 %address.thread934 to i64, !dbg !8
  %address.local936 = add i64 %25, %address.local935, !dbg !21
  %address.local937 = lshr i64 %address.local935, 6, !dbg !11
  %address.local938 = shl nuw nsw i64 %address.local937, 4, !dbg !21
  %address.local939 = add i64 %address.local936, %address.local938, !dbg !21
  %address.local940 = and i64 %address.local935, 240, !dbg !21
  %address.local941 = sub i64 %address.local939, %address.local940, !dbg !21
  %address.local942 = add i64 %address.local941, %26, !dbg !21
  %address.local943 = shl nuw nsw i64 %address.local942, 7, !dbg !21
  %address.local944 = add nuw nsw i64 %1479, %address.local943, !dbg !21
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local946 = lshr i64 %address.local935, 2, !dbg !11
  %address.local947 = and i64 %address.local946, 12, !dbg !11
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local945, i64 %address.local947, !dbg !21
  %address.local949 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local948, i64 192, !dbg !21
  store half %1478, ptr addrspace(1) %address.local949, align 2, !dbg !21
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local951 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local950, i64 %address.local947, !dbg !21
  %address.local952 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local951, i64 194, !dbg !21
  store half %1477, ptr addrspace(1) %address.local952, align 2, !dbg !21
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local953, i64 %address.local947, !dbg !21
  %address.local955 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local954, i64 196, !dbg !21
  store half %1476, ptr addrspace(1) %address.local955, align 2, !dbg !21
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local957 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local956, i64 %address.local947, !dbg !21
  %address.local958 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local957, i64 198, !dbg !21
  store half %1475, ptr addrspace(1) %address.local958, align 2, !dbg !21
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local960 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local959, i64 %address.local947, !dbg !21
  %address.local961 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local960, i64 224, !dbg !21
  store half %1469, ptr addrspace(1) %address.local961, align 2, !dbg !21
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local962, i64 %address.local947, !dbg !21
  %address.local964 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local963, i64 226, !dbg !21
  store half %1468, ptr addrspace(1) %address.local964, align 2, !dbg !21
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local965, i64 %address.local947, !dbg !21
  %address.local967 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local966, i64 228, !dbg !21
  store half %1467, ptr addrspace(1) %address.local967, align 2, !dbg !21
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local944, !dbg !21
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local968, i64 %address.local947, !dbg !21
  %address.local970 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local969, i64 230, !dbg !21
  store half %1466, ptr addrspace(1) %address.local970, align 2, !dbg !21
  br label %1480, !dbg !21

1480:                                             ; preds = %1460, %1459
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!38}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/safe_softmax_tuning_alignment/default/session-20261026-101530-iv100m7k/tuning/run-20261026-101530-001slkfo/input-0003-test_input_1x32x4096x128-64-128-32-online/online-baseline")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 60, column: 24, scope: !4)
!8 = !DILocation(line: 70, column: 22, scope: !4)
!10 = !DILocation(line: 48, column: 17, scope: !4)
!11 = !DILocation(line: 72, column: 24, scope: !4)
!14 = !DILocation(line: 77, column: 19, scope: !4)
!15 = !DILocation(line: 102, column: 23, scope: !4)
!16 = !DILocation(line: 73, column: 26, scope: !4)
!17 = !DILocation(line: 71, column: 22, scope: !4)
!18 = !DILocation(line: 101, column: 20, scope: !4)
!19 = !DILocation(line: 107, column: 23, scope: !4)
!20 = !DILocation(line: 108, column: 19, scope: !4)
!21 = !DILocation(line: 109, column: 9, scope: !4)
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
