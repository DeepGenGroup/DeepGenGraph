; ModuleID = '/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD.mlir'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD.mlir"

@async_lds_staging_0 = private addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !4 !reqd_work_group_size !32 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %6 = sext i32 %5 to i64, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = shl nsw i64 %6, 7, !dbg !12
  %.lobit = ashr i32 %5, 31, !dbg !14
  %10 = xor i32 %5, %.lobit, !dbg !14
  %11 = sdiv i32 %10, 32, !dbg !14
  %.lobit1 = ashr i32 %5, 31, !dbg !14
  %12 = xor i32 %11, %.lobit1, !dbg !14
  %13 = sext i32 %12 to i64, !dbg !14
  %14 = add nsw i64 %8, %13, !dbg !14
  %15 = and i64 %14, 31, !dbg !14
  %16 = shl nsw i64 %6, 7, !dbg !14
  %17 = mul nsw i64 %13, -4096, !dbg !14
  %18 = add nsw i64 %16, %17, !dbg !14
  %19 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !6, !range !34
  %20 = zext nneg i32 %19 to i64, !dbg !6
  %21 = shl nuw nsw i64 %20, 2, !dbg !6
  %22 = lshr i64 %20, 6, !dbg !6
  %23 = lshr i64 %20, 4, !dbg !6
  %24 = and i64 %23, 3, !dbg !6
  %25 = or disjoint i64 %9, 127, !dbg !15
  %26 = add nsw i64 %9, 128, !dbg !15
  %27 = icmp uge i64 %26, %25, !dbg !15
  %28 = lshr i64 %20, 2, !dbg !8
  %29 = and i64 %20, 15, !dbg !8
  %30 = lshr i64 %29, 2, !dbg !8
  %31 = lshr i64 %20, 1, !dbg !8
  %32 = lshr i64 %24, 1, !dbg !8
  %33 = mul nsw i64 %28, -18, !dbg !8
  %34 = add nsw i64 %21, %33, !dbg !8
  %35 = add nsw i64 %34, %31, !dbg !8
  %36 = add nsw i64 %35, %32, !dbg !8
  %.neg = sdiv i64 %36, -2, !dbg !8
  %37 = mul nuw nsw i64 %20, 18, !dbg !8
  %38 = mul nsw i64 %28, -72, !dbg !8
  %39 = add nsw i64 %37, %38, !dbg !8
  %40 = add nsw i64 %39, %30, !dbg !8
  %.neg194 = sdiv i64 %40, -4, !dbg !8
  %41 = shl nuw nsw i64 %20, 4, !dbg !8
  %42 = mul nsw i64 %28, -64, !dbg !8
  %43 = add nsw i64 %41, %42, !dbg !8
  %44 = or disjoint i64 %43, %30, !dbg !8
  %45 = shl nuw nsw i32 %19, 1, !dbg !8
  %46 = and i32 %45, 6, !dbg !8
  %index.compact185 = zext nneg i32 %46 to i64, !dbg !8
  %47 = lshr i32 %19, 1, !dbg !8
  %48 = and i32 %47, 1, !dbg !8
  %index.compact182 = zext nneg i32 %48 to i64, !dbg !8
  %49 = mul nsw i64 %28, -16, !dbg !8
  %50 = add nsw i64 %21, %49, !dbg !8
  %51 = or disjoint i64 %50, %index.compact182, !dbg !8
  %52 = add nuw nsw i64 %51, %32, !dbg !8
  %53 = add nsw i64 %52, 1, !dbg !8
  %.neg197 = sdiv i64 %53, -2, !dbg !8
  %54 = add nuw nsw i64 %44, %index.compact185, !dbg !8
  %.neg199 = sdiv i64 %54, -4, !dbg !8
  br label %55, !dbg !16

55:                                               ; preds = %1384, %4
  %56 = phi i64 [ %1491, %1384 ], [ 0, %4 ], !dbg !16
  %57 = phi [2 x <8 x float>] [ %1451, %1384 ], [ zeroinitializer, %4 ], !dbg !16
  %58 = phi [2 x <1 x float>] [ %1455, %1384 ], [ zeroinitializer, %4 ], !dbg !16
  %59 = phi [2 x <8 x float>] [ %1462, %1384 ], [ zeroinitializer, %4 ], !dbg !16
  %60 = phi [2 x <8 x float>] [ %1468, %1384 ], [ zeroinitializer, %4 ], !dbg !16
  %61 = phi [2 x <8 x float>] [ %1490, %1384 ], [ zeroinitializer, %4 ], !dbg !16
  %.not = icmp sgt i64 %56, %6, !dbg !16
  br i1 %.not, label %1492, label %62, !dbg !16

62:                                               ; preds = %55
  %63 = shl nsw i64 %56, 7, !dbg !16
  %64 = lshr i64 %56, 5, !dbg !17
  %65 = shl i64 %64, 12, !dbg !17
  %66 = sub i64 %63, %65, !dbg !17
  br label %67, !dbg !22

67:                                               ; preds = %148, %62
  %68 = phi i64 [ %283, %148 ], [ 0, %62 ], !dbg !22
  %69 = phi [2 x <32 x float>] [ %282, %148 ], [ zeroinitializer, %62 ], !dbg !22
  %70 = icmp samesign ult i64 %68, 4, !dbg !22
  br i1 %70, label %71, label %284, !dbg !22

71:                                               ; preds = %67
  %72 = shl nuw nsw i64 %68, 5, !dbg !22
  %.idx107 = shl nuw nsw i64 %15, 20, !dbg !14
  %73 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx107, !dbg !14
  %.idx108 = shl nsw i64 %18, 8, !dbg !14
  %74 = getelementptr i8, ptr addrspace(1) %73, i64 %.idx108, !dbg !14
  %75 = getelementptr [2 x i8], ptr addrspace(1) %74, i64 %72, !dbg !14
  br label %76, !dbg !14

76:                                               ; preds = %79, %71
  %77 = phi i64 [ %97, %79 ], [ %20, %71 ], !dbg !14
  %78 = icmp samesign ult i64 %77, 2048, !dbg !14
  br i1 %78, label %79, label %98, !dbg !14

79:                                               ; preds = %76
  %80 = shl nuw nsw i64 %77, 1, !dbg !14
  %81 = lshr i64 %77, 6, !dbg !14
  %82 = and i64 %81, 28, !dbg !14
  %83 = and i64 %80, 2, !dbg !14
  %84 = shl nuw nsw i64 %77, 1, !dbg !14
  %85 = and i64 %84, 3968, !dbg !14
  %86 = shl nuw nsw i64 %77, 6, !dbg !14
  %87 = and i64 %86, 16256, !dbg !14
  %88 = getelementptr [2 x i8], ptr addrspace(1) %75, i64 %87, !dbg !14
  %89 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %82, !dbg !14
  %90 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %83, !dbg !14
  %91 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %85, !dbg !14
  %92 = ptrtoint ptr addrspace(3) %91 to i64, !dbg !14
  %93 = trunc i64 %92 to i32, !dbg !14
  %94 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %93), !dbg !14
  %95 = zext i32 %94 to i64, !dbg !14
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !14
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %90, ptr addrspace(3) %96, i32 4, i32 0, i32 0), !dbg !14
  %97 = add nuw nsw i64 %77, 256, !dbg !14
  br label %76, !dbg !14

98:                                               ; preds = %76
  call void @llvm.amdgcn.asyncmark(), !dbg !14
  %99 = shl nuw nsw i64 %68, 10, !dbg !17
  %100 = add nuw i64 %56, %99, !dbg !17
  %.lobit109 = ashr i64 %100, 63, !dbg !17
  %101 = xor i64 %100, %.lobit109, !dbg !17
  %102 = sdiv i64 %101, 4096, !dbg !17
  %.lobit110 = ashr i64 %100, 63, !dbg !17
  %103 = xor i64 %102, %.lobit110, !dbg !17
  %104 = add nsw i64 %103, %8, !dbg !17
  %105 = add nuw nsw i64 %72, %64, !dbg !17
  %106 = and i64 %105, 127, !dbg !17
  %107 = shl i64 %104, 19, !dbg !17
  %108 = and i64 %107, 16252928, !dbg !17
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %109 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %108, !dbg !17
  %110 = getelementptr [2 x i8], ptr addrspace(1) %109, i64 %106, !dbg !17
  %.idx111 = shl i64 %66, 8, !dbg !17
  %111 = getelementptr i8, ptr addrspace(1) %110, i64 %.idx111, !dbg !17
  br label %112, !dbg !17

112:                                              ; preds = %115, %98
  %113 = phi i64 [ %147, %115 ], [ %20, %98 ], !dbg !17
  %114 = icmp samesign ult i64 %113, 2048, !dbg !17
  br i1 %114, label %115, label %148, !dbg !17

115:                                              ; preds = %112
  %116 = shl nuw nsw i64 %113, 1, !dbg !17
  %117 = lshr i64 %113, 9, !dbg !17
  %118 = shl nuw nsw i64 %113, 1, !dbg !17
  %119 = and i64 %118, 4, !dbg !17
  %120 = or disjoint i64 %119, %117, !dbg !17
  %121 = and i64 %116, 2, !dbg !17
  %122 = lshr i64 %113, 2, !dbg !17
  %123 = lshr i64 %113, 5, !dbg !17
  %124 = and i64 %122, 120, !dbg !17
  %125 = lshr i64 %113, 4, !dbg !17
  %126 = lshr i64 %113, 7, !dbg !17
  %127 = lshr i64 %113, 10, !dbg !17
  %128 = add nuw nsw i64 %126, %127, !dbg !17
  %129 = sub nsw i64 %125, %128, !dbg !17
  %130 = shl nsw i64 %129, 2, !dbg !17
  %131 = and i64 %130, 4, !dbg !17
  %132 = or disjoint i64 %124, %131, !dbg !17
  %133 = sub nsw i64 %122, %123, !dbg !17
  %134 = and i64 %133, 3, !dbg !17
  %135 = or disjoint i64 %132, %134, !dbg !17
  %136 = shl nuw nsw i64 %113, 1, !dbg !17
  %137 = and i64 %136, 3968, !dbg !17
  %.idx112 = shl nuw nsw i64 %120, 3, !dbg !17
  %138 = getelementptr i8, ptr addrspace(1) %111, i64 %.idx112, !dbg !17
  %139 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %121, !dbg !17
  %.idx113 = shl nuw nsw i64 %135, 8, !dbg !17
  %140 = getelementptr i8, ptr addrspace(1) %139, i64 %.idx113, !dbg !17
  %141 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %137, !dbg !17
  %142 = ptrtoint ptr addrspace(3) %141 to i64, !dbg !17
  %143 = trunc i64 %142 to i32, !dbg !17
  %144 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %143), !dbg !17
  %145 = zext i32 %144 to i64, !dbg !17
  %146 = inttoptr i64 %145 to ptr addrspace(3), !dbg !17
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %140, ptr addrspace(3) %146, i32 4, i32 0, i32 0), !dbg !17
  %147 = add nuw nsw i64 %113, 256, !dbg !17
  br label %112, !dbg !17

148:                                              ; preds = %112
  call void @llvm.amdgcn.asyncmark(), !dbg !17
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !6, !range !34, !frisk.register_lifetimes !1
  %address.local = and i32 %address.thread, 192, !dbg !6
  %address.local264 = shl nuw nsw i32 %address.thread, 2, !dbg !6
  %address.local265 = and i32 %address.local264, 60, !dbg !6
  %address.local266 = or disjoint i32 %address.local, %address.local265, !dbg !6
  %address.local267 = shl nuw nsw i32 %address.thread, 5, !dbg !6
  %address.local268 = and i32 %address.local267, 1536, !dbg !6
  %address.local269 = or disjoint i32 %address.local266, %address.local268, !dbg !6
  %address.local270 = zext nneg i32 %address.local269 to i64, !dbg !6
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local270, !dbg !6
  %149 = load <4 x half>, ptr addrspace(3) %address.local271, align 2, !dbg !6
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local270, !dbg !6
  %address.local273 = getelementptr i8, ptr addrspace(3) %address.local272, i64 4096, !dbg !6
  %150 = load <4 x half>, ptr addrspace(3) %address.local273, align 2, !dbg !6
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local270, !dbg !6
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local274, i64 512, !dbg !6
  %151 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !6
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local270, !dbg !6
  %address.local277 = getelementptr i8, ptr addrspace(3) %address.local276, i64 4608, !dbg !6
  %152 = load <4 x half>, ptr addrspace(3) %address.local277, align 2, !dbg !6
  %153 = fmul <4 x half> %149, splat (half 0xH3015), !dbg !6
  %154 = fmul <4 x half> %150, splat (half 0xH3015), !dbg !6
  %155 = fmul <4 x half> %151, splat (half 0xH3015), !dbg !6
  %156 = fmul <4 x half> %152, splat (half 0xH3015), !dbg !6
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local278 = shl i64 %address.base, 6, !dbg !8
  %address.base279 = call i64 asm sideeffect "", "=v,0"(i64 %.neg194) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local280 = shl i64 %address.base279, 5, !dbg !8
  %address.local281 = add i64 %address.local278, %address.local280, !dbg !8
  %address.local282 = zext nneg i32 %address.thread to i64, !dbg !6
  %address.local283 = mul nuw nsw i64 %address.local282, 400, !dbg !8
  %address.local284 = lshr i64 %address.local282, 2, !dbg !8
  %address.local285 = mul nsw i64 %address.local284, -1664, !dbg !8
  %address.local286 = add nsw i64 %address.local283, %address.local285, !dbg !8
  %address.local287 = and i64 %address.local282, 15, !dbg !8
  %address.local288 = lshr i64 %address.local287, 2, !dbg !8
  %address.local289 = shl nuw nsw i64 %address.local288, 3, !dbg !8
  %address.local290 = add nsw i64 %address.local286, %address.local289, !dbg !8
  %address.local291 = lshr i64 %address.local282, 1, !dbg !8
  %address.local292 = shl nuw nsw i64 %address.local291, 5, !dbg !8
  %address.local293 = add nsw i64 %address.local290, %address.local292, !dbg !8
  %address.local294 = lshr i64 %address.local282, 4, !dbg !6
  %address.local295 = and i64 %address.local294, 3, !dbg !6
  %address.local296 = shl nuw nsw i64 %address.local295, 10, !dbg !8
  %address.local297 = add nsw i64 %address.local293, %address.local296, !dbg !8
  %address.local298 = and i32 %address.thread, 32, !dbg !8
  %address.local299 = zext nneg i32 %address.local298 to i64, !dbg !8
  %address.local300 = add i64 %address.local297, %address.local299, !dbg !8
  %address.local301 = add i64 %address.local281, %address.local300, !dbg !8
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local301, !dbg !8
  %157 = load <8 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !8
  %158 = extractvalue [2 x <32 x float>] %69, 0, !dbg !31
  %159 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %160 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %161 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %160, <4 x float> %159) #9, !dbg !8
  %address.base303 = call i64 asm sideeffect "", "=v,0"(i64 %.neg197) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local304 = shl i64 %address.base303, 7, !dbg !8
  %address.local305 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local304, !dbg !8
  %address.base306 = call i64 asm sideeffect "", "=v,0"(i64 %.neg199) #8, !dbg !8, !frisk.register_lifetimes !1
  %address.local307 = shl i64 %address.base306, 6, !dbg !8
  %address.local308 = getelementptr i8, ptr addrspace(3) %address.local305, i64 %address.local307, !dbg !8
  %address.local309 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local310 = and i32 %address.local309, 6, !dbg !8
  %address.local311 = zext nneg i32 %address.local310 to i64, !dbg !8
  %address.local312 = mul nuw nsw i64 %address.local311, 144, !dbg !8
  %address.local313 = getelementptr i8, ptr addrspace(3) %address.local308, i64 %address.local312, !dbg !8
  %address.local314 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local315 = and i32 %address.local314, 24, !dbg !8
  %address.local316 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local317 = and i32 %address.local316, 768, !dbg !8
  %address.local318 = or disjoint i32 %address.local315, %address.local317, !dbg !8
  %address.local319 = zext nneg i32 %address.local318 to i64, !dbg !8
  %address.local320 = getelementptr [2 x i8], ptr addrspace(3) %address.local313, i64 %address.local319, !dbg !8
  %address.local321 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local322 = and i32 %address.local321, 32, !dbg !8
  %address.local323 = zext nneg i32 %address.local322 to i64, !dbg !8
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) %address.local320, i64 %address.local323, !dbg !8
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) %address.local324, i64 %address.local296, !dbg !8
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) %address.local325, i64 %address.local299, !dbg !8
  %address.local327 = getelementptr i8, ptr addrspace(3) %address.local326, i64 64, !dbg !8
  %162 = load <8 x half>, ptr addrspace(3) %address.local327, align 2, !dbg !8
  %163 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %164 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %165 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %163, <4 x float> %161) #9, !dbg !8
  %166 = shufflevector <8 x half> %162, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %167 = shufflevector <8 x half> %162, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %168 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %169 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %166, <4 x float> %164) #9, !dbg !8
  %170 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %171 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %167, <4 x float> %169) #9, !dbg !8
  %172 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %173 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %174 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %175 = shufflevector <32 x float> %158, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %176 = extractvalue [2 x <32 x float>] %69, 1, !dbg !31
  %177 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !31
  %178 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !31
  %179 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !31
  %180 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !31
  %181 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !31
  %182 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !31
  %183 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !31
  %184 = shufflevector <32 x float> %176, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !31
  %address.local328 = shl nuw nsw i64 %address.local282, 2, !dbg !6
  %address.local329 = mul nsw i64 %address.local284, -18, !dbg !8
  %address.local330 = add nsw i64 %address.local328, %address.local329, !dbg !8
  %address.local331 = add nsw i64 %address.local330, %address.local291, !dbg !8
  %address.local332 = lshr i64 %address.local295, 1, !dbg !8
  %address.local333 = add nsw i64 %address.local331, %address.local332, !dbg !8
  %address.local334 = sdiv i64 %address.local333, -2, !dbg !8
  %address.local335 = shl i64 %address.local334, 7, !dbg !8
  %address.local336 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local335, !dbg !8
  %address.local337 = mul nuw nsw i64 %address.local282, 18, !dbg !8
  %address.local338 = mul nsw i64 %address.local284, -72, !dbg !8
  %address.local339 = add nsw i64 %address.local337, %address.local338, !dbg !8
  %address.local340 = add nsw i64 %address.local339, %address.local288, !dbg !8
  %address.local341 = add nsw i64 %address.local340, 1, !dbg !8
  %address.local342 = sdiv i64 %address.local341, -4, !dbg !8
  %address.local343 = shl i64 %address.local342, 6, !dbg !8
  %address.local344 = getelementptr i8, ptr addrspace(3) %address.local336, i64 %address.local343, !dbg !8
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) %address.local344, i64 %address.local283, !dbg !8
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local285, !dbg !8
  %address.local347 = getelementptr [2 x i8], ptr addrspace(3) %address.local346, i64 %address.local289, !dbg !8
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) %address.local347, i64 %address.local292, !dbg !8
  %address.local349 = getelementptr [2 x i8], ptr addrspace(3) %address.local348, i64 %address.local296, !dbg !8
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) %address.local349, i64 %address.local299, !dbg !8
  %address.local351 = getelementptr i8, ptr addrspace(3) %address.local350, i64 144, !dbg !8
  %185 = load <8 x half>, ptr addrspace(3) %address.local351, align 2, !dbg !8
  %address.local352 = mul nsw i64 %address.local284, -16, !dbg !8
  %address.local353 = add nsw i64 %address.local328, %address.local352, !dbg !8
  %address.local354 = lshr i32 %address.thread, 1, !dbg !8
  %address.local355 = and i32 %address.local354, 1, !dbg !8
  %address.local356 = zext nneg i32 %address.local355 to i64, !dbg !8
  %address.local357 = or disjoint i64 %address.local353, %address.local356, !dbg !8
  %address.local358 = add nuw nsw i64 %address.local357, %address.local332, !dbg !8
  %address.local359 = add nsw i64 %address.local358, 3, !dbg !8
  %address.local360 = sdiv i64 %address.local359, -2, !dbg !8
  %address.local361 = shl i64 %address.local360, 7, !dbg !8
  %address.local362 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local361, !dbg !8
  %address.local363 = shl nuw nsw i64 %address.local282, 4, !dbg !8
  %address.local364 = mul nsw i64 %address.local284, -64, !dbg !8
  %address.local365 = add nsw i64 %address.local363, %address.local364, !dbg !8
  %address.local366 = or disjoint i64 %address.local365, %address.local288, !dbg !8
  %address.local367 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local368 = and i32 %address.local367, 6, !dbg !8
  %address.local369 = or disjoint i32 %address.local368, 1, !dbg !8
  %address.local370 = zext nneg i32 %address.local369 to i64, !dbg !8
  %address.local371 = add nuw nsw i64 %address.local366, %address.local370, !dbg !8
  %address.local372 = sdiv i64 %address.local371, -4, !dbg !8
  %address.local373 = shl i64 %address.local372, 6, !dbg !8
  %address.local374 = getelementptr i8, ptr addrspace(3) %address.local362, i64 %address.local373, !dbg !8
  %address.local375 = mul nuw nsw i64 %address.local370, 144, !dbg !8
  %address.local376 = getelementptr i8, ptr addrspace(3) %address.local374, i64 %address.local375, !dbg !8
  %address.local377 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local378 = and i32 %address.local377, 24, !dbg !8
  %address.local379 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local380 = and i32 %address.local379, 768, !dbg !8
  %address.local381 = or disjoint i32 %address.local378, %address.local380, !dbg !8
  %address.local382 = zext nneg i32 %address.local381 to i64, !dbg !8
  %address.local383 = getelementptr [2 x i8], ptr addrspace(3) %address.local376, i64 %address.local382, !dbg !8
  %address.local384 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local385 = and i32 %address.local384, 32, !dbg !8
  %address.local386 = zext nneg i32 %address.local385 to i64, !dbg !8
  %address.local387 = getelementptr [2 x i8], ptr addrspace(3) %address.local383, i64 %address.local386, !dbg !8
  %address.local388 = getelementptr [2 x i8], ptr addrspace(3) %address.local387, i64 %address.local296, !dbg !8
  %address.local389 = getelementptr [2 x i8], ptr addrspace(3) %address.local388, i64 %address.local299, !dbg !8
  %address.local390 = getelementptr i8, ptr addrspace(3) %address.local389, i64 192, !dbg !8
  %186 = load <8 x half>, ptr addrspace(3) %address.local390, align 2, !dbg !8
  %address.local391 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local301, !dbg !8
  %address.local392 = getelementptr i8, ptr addrspace(3) %address.local391, i64 1024, !dbg !8
  %187 = load <8 x half>, ptr addrspace(3) %address.local392, align 2, !dbg !8
  %188 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %189 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %188, <4 x float> %168) #9, !dbg !8
  %190 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %191 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %192 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %193 = shufflevector <8 x half> %187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %194 = shufflevector <8 x half> %187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %address.local393 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local394 = and i32 %address.local393, 6, !dbg !8
  %address.local395 = or disjoint i32 %address.local394, 8, !dbg !8
  %address.local396 = mul nuw nsw i32 %address.local395, 72, !dbg !8
  %address.local397 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local398 = and i32 %address.local397, 24, !dbg !8
  %address.local399 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local400 = and i32 %address.local399, 768, !dbg !8
  %address.local401 = or disjoint i32 %address.local398, %address.local400, !dbg !8
  %address.local402 = or disjoint i32 %address.local401, 1024, !dbg !8
  %address.local403 = add nuw nsw i32 %address.local396, %address.local402, !dbg !8
  %address.local404 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local405 = and i32 %address.local404, 32, !dbg !8
  %address.local406 = or disjoint i32 %address.local405, 64, !dbg !8
  %address.local407 = add nuw nsw i32 %address.local403, %address.local406, !dbg !8
  %address.local408 = zext nneg i32 %address.local407 to i64, !dbg !8
  %address.local409 = add nuw nsw i64 %address.local296, %address.local408, !dbg !8
  %address.local410 = add nuw nsw i64 %address.local409, %address.local299, !dbg !8
  %address.local411 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local412 = and i32 %address.local411, 12, !dbg !8
  %address.local413 = lshr i32 %address.thread, 1, !dbg !8
  %address.local414 = and i32 %address.local413, 1, !dbg !8
  %address.local415 = or disjoint i32 %address.local412, %address.local414, !dbg !8
  %address.local416 = or disjoint i32 %address.local415, 18, !dbg !8
  %address.local417 = zext nneg i32 %address.local416 to i64, !dbg !8
  %address.local418 = add nuw nsw i64 %address.local332, %address.local417, !dbg !8
  %address.local419 = shl nuw nsw i64 %address.local418, 5, !dbg !8
  %address.local420 = add nuw nsw i64 %address.local419, 32, !dbg !8
  %address.local421 = and i64 %address.local420, 274877906880, !dbg !8
  %address.local422 = lshr i32 %address.thread, 2, !dbg !8
  %address.local423 = and i32 %address.local422, 2, !dbg !8
  %address.local424 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local425 = and i32 %address.local424, 48, !dbg !8
  %address.local426 = or disjoint i32 %address.local423, %address.local425, !dbg !8
  %address.local427 = or disjoint i32 %address.local426, 64, !dbg !8
  %address.local428 = add nuw nsw i32 %address.local427, %address.local395, !dbg !8
  %address.local429 = shl nuw nsw i32 %address.local428, 3, !dbg !8
  %address.local430 = and i32 %address.local429, 2016, !dbg !8
  %address.local431 = zext nneg i32 %address.local430 to i64, !dbg !8
  %address.local432 = add nuw nsw i64 %address.local421, %address.local431, !dbg !8
  %address.local433 = sub nsw i64 %address.local410, %address.local432, !dbg !8
  %address.local434 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local433, !dbg !8
  %address.local435 = getelementptr i8, ptr addrspace(3) %address.local434, i64 64, !dbg !8
  %195 = load <8 x half>, ptr addrspace(3) %address.local435, align 2, !dbg !8
  %196 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %190, <4 x float> %189) #9, !dbg !8
  %address.local436 = add nuw nsw i64 %address.local328, 18, !dbg !8
  %address.local437 = add nsw i64 %address.local436, %address.local329, !dbg !8
  %address.local438 = add nsw i64 %address.local437, %address.local291, !dbg !8
  %address.local439 = add nsw i64 %address.local438, %address.local332, !dbg !8
  %address.local440 = sdiv i64 %address.local439, -2, !dbg !8
  %address.local441 = shl i64 %address.local440, 7, !dbg !8
  %address.local442 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local441, !dbg !8
  %address.local443 = add nuw nsw i64 %address.local337, 72, !dbg !8
  %address.local444 = add nsw i64 %address.local443, %address.local338, !dbg !8
  %address.local445 = add nsw i64 %address.local444, %address.local288, !dbg !8
  %address.local446 = add nsw i64 %address.local445, -15, !dbg !8
  %address.local447 = sdiv i64 %address.local446, -4, !dbg !8
  %address.local448 = shl i64 %address.local447, 6, !dbg !8
  %address.local449 = getelementptr i8, ptr addrspace(3) %address.local442, i64 %address.local448, !dbg !8
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) %address.local449, i64 %address.local283, !dbg !8
  %address.local451 = getelementptr [2 x i8], ptr addrspace(3) %address.local450, i64 %address.local285, !dbg !8
  %address.local452 = getelementptr [2 x i8], ptr addrspace(3) %address.local451, i64 %address.local289, !dbg !8
  %address.local453 = getelementptr [2 x i8], ptr addrspace(3) %address.local452, i64 %address.local292, !dbg !8
  %address.local454 = getelementptr [2 x i8], ptr addrspace(3) %address.local453, i64 %address.local296, !dbg !8
  %address.local455 = getelementptr [2 x i8], ptr addrspace(3) %address.local454, i64 %address.local299, !dbg !8
  %address.local456 = getelementptr i8, ptr addrspace(3) %address.local455, i64 3216, !dbg !8
  %197 = load <8 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !8
  %address.local457 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local458 = and i32 %address.local457, 6, !dbg !8
  %address.local459 = or disjoint i32 %address.local458, 9, !dbg !8
  %address.local460 = mul nuw nsw i32 %address.local459, 72, !dbg !8
  %address.local461 = shl nuw nsw i32 %address.thread, 1, !dbg !8
  %address.local462 = and i32 %address.local461, 24, !dbg !8
  %address.local463 = shl nuw nsw i32 %address.thread, 8, !dbg !8
  %address.local464 = and i32 %address.local463, 768, !dbg !8
  %address.local465 = or disjoint i32 %address.local462, %address.local464, !dbg !8
  %address.local466 = or disjoint i32 %address.local465, 1024, !dbg !8
  %address.local467 = add nuw nsw i32 %address.local460, %address.local466, !dbg !8
  %address.local468 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local469 = and i32 %address.local468, 32, !dbg !8
  %address.local470 = or disjoint i32 %address.local469, 64, !dbg !8
  %address.local471 = add nuw nsw i32 %address.local467, %address.local470, !dbg !8
  %address.local472 = zext nneg i32 %address.local471 to i64, !dbg !8
  %address.local473 = add nuw nsw i64 %address.local296, %address.local472, !dbg !8
  %address.local474 = add nuw nsw i64 %address.local473, %address.local299, !dbg !8
  %address.local475 = shl nuw nsw i32 %address.thread, 2, !dbg !8
  %address.local476 = and i32 %address.local475, 12, !dbg !8
  %address.local477 = lshr i32 %address.thread, 1, !dbg !8
  %address.local478 = and i32 %address.local477, 1, !dbg !8
  %address.local479 = or disjoint i32 %address.local476, %address.local478, !dbg !8
  %address.local480 = or disjoint i32 %address.local479, 18, !dbg !8
  %address.local481 = zext nneg i32 %address.local480 to i64, !dbg !8
  %address.local482 = add nuw nsw i64 %address.local332, %address.local481, !dbg !8
  %address.local483 = shl nuw nsw i64 %address.local482, 5, !dbg !8
  %address.local484 = add nuw nsw i64 %address.local483, 96, !dbg !8
  %address.local485 = and i64 %address.local484, 274877906880, !dbg !8
  %address.local486 = lshr i32 %address.thread, 2, !dbg !8
  %address.local487 = and i32 %address.local486, 3, !dbg !8
  %address.local488 = shl nuw nsw i32 %address.thread, 4, !dbg !8
  %address.local489 = and i32 %address.local488, 48, !dbg !8
  %address.local490 = or disjoint i32 %address.local487, %address.local489, !dbg !8
  %address.local491 = or disjoint i32 %address.local490, 64, !dbg !8
  %address.local492 = add nuw nsw i32 %address.local491, %address.local459, !dbg !8
  %address.local493 = shl nuw nsw i32 %address.local492, 3, !dbg !8
  %address.local494 = and i32 %address.local493, 2016, !dbg !8
  %address.local495 = zext nneg i32 %address.local494 to i64, !dbg !8
  %address.local496 = add nuw nsw i64 %address.local485, %address.local495, !dbg !8
  %address.local497 = sub nsw i64 %address.local474, %address.local496, !dbg !8
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local497, !dbg !8
  %address.local499 = getelementptr i8, ptr addrspace(3) %address.local498, i64 192, !dbg !8
  %198 = load <8 x half>, ptr addrspace(3) %address.local499, align 2, !dbg !8
  %199 = shufflevector <8 x half> %195, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %200 = shufflevector <4 x float> %196, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %201 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %191, <4 x float> %170) #9, !dbg !8
  %202 = shufflevector <8 x half> %195, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %203 = shufflevector <8 x half> %197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %204 = shufflevector <8 x half> %197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %205 = shufflevector <8 x half> %198, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %206 = shufflevector <8 x half> %198, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %207 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %192, <4 x float> %201) #9, !dbg !8
  %208 = load <8 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !8
  %209 = shufflevector <4 x float> %207, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %210 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %193, <4 x float> %172) #9, !dbg !8
  %211 = load <8 x half>, ptr addrspace(3) %address.local327, align 2, !dbg !8
  %212 = shufflevector <8 x half> %208, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %213 = shufflevector <8 x half> %208, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %214 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %194, <4 x float> %210) #9, !dbg !8
  %215 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %216 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %217 = load <8 x half>, ptr addrspace(3) %address.local351, align 2, !dbg !8
  %218 = shufflevector <4 x float> %214, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %219 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %199, <4 x float> %173) #9, !dbg !8
  %220 = load <8 x half>, ptr addrspace(3) %address.local390, align 2, !dbg !8
  %221 = shufflevector <8 x half> %217, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %222 = shufflevector <8 x half> %217, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %223 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %202, <4 x float> %219) #9, !dbg !8
  %224 = shufflevector <8 x half> %220, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %225 = shufflevector <8 x half> %220, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %226 = load <8 x half>, ptr addrspace(3) %address.local392, align 2, !dbg !8
  %227 = shufflevector <4 x float> %223, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %228 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %203, <4 x float> %174) #9, !dbg !8
  %229 = load <8 x half>, ptr addrspace(3) %address.local435, align 2, !dbg !8
  %230 = shufflevector <8 x half> %226, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %231 = shufflevector <8 x half> %226, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %232 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %204, <4 x float> %228) #9, !dbg !8
  %233 = shufflevector <8 x half> %229, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %234 = shufflevector <8 x half> %229, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %235 = load <8 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !8
  %236 = shufflevector <4 x float> %232, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %237 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %153, <4 x half> %205, <4 x float> %175) #9, !dbg !8
  %238 = load <8 x half>, ptr addrspace(3) %address.local499, align 2, !dbg !8
  %239 = shufflevector <8 x half> %235, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %240 = shufflevector <8 x half> %235, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %241 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %154, <4 x half> %206, <4 x float> %237) #9, !dbg !8
  %242 = shufflevector <8 x half> %238, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !8
  %243 = shufflevector <8 x half> %238, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !8
  %244 = shufflevector <4 x float> %241, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %245 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %212, <4 x float> %177) #9, !dbg !8
  %246 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %213, <4 x float> %245) #9, !dbg !8
  %247 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %215, <4 x float> %178) #9, !dbg !8
  %248 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %216, <4 x float> %247) #9, !dbg !8
  %249 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %221, <4 x float> %179) #9, !dbg !8
  %250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %222, <4 x float> %249) #9, !dbg !8
  %251 = shufflevector <4 x float> %250, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %252 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %224, <4 x float> %180) #9, !dbg !8
  %253 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %225, <4 x float> %252) #9, !dbg !8
  %254 = shufflevector <4 x float> %253, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %255 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %230, <4 x float> %181) #9, !dbg !8
  %256 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %231, <4 x float> %255) #9, !dbg !8
  %257 = shufflevector <4 x float> %256, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %258 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %233, <4 x float> %182) #9, !dbg !8
  %259 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %234, <4 x float> %258) #9, !dbg !8
  %260 = shufflevector <4 x float> %259, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %261 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %239, <4 x float> %183) #9, !dbg !8
  %262 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %240, <4 x float> %261) #9, !dbg !8
  %263 = shufflevector <4 x float> %262, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %264 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %155, <4 x half> %242, <4 x float> %184) #9, !dbg !8
  %265 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %156, <4 x half> %243, <4 x float> %264) #9, !dbg !8
  %266 = shufflevector <4 x float> %265, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !31
  %267 = shufflevector <4 x float> %165, <4 x float> %171, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %268 = shufflevector <32 x float> %267, <32 x float> %200, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %269 = shufflevector <32 x float> %268, <32 x float> %209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %270 = shufflevector <32 x float> %269, <32 x float> %218, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %271 = shufflevector <32 x float> %270, <32 x float> %227, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %272 = shufflevector <32 x float> %271, <32 x float> %236, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %273 = shufflevector <32 x float> %272, <32 x float> %244, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %274 = insertvalue [2 x <32 x float>] poison, <32 x float> %273, 0, !dbg !22
  %275 = shufflevector <4 x float> %246, <4 x float> %248, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %276 = shufflevector <32 x float> %275, <32 x float> %251, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %277 = shufflevector <32 x float> %276, <32 x float> %254, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %278 = shufflevector <32 x float> %277, <32 x float> %257, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %279 = shufflevector <32 x float> %278, <32 x float> %260, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %280 = shufflevector <32 x float> %279, <32 x float> %263, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !22
  %281 = shufflevector <32 x float> %280, <32 x float> %266, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !22
  %282 = insertvalue [2 x <32 x float>] %274, <32 x float> %281, 1, !dbg !22
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  %283 = add nuw nsw i64 %68, 1, !dbg !22
  br label %67, !dbg !22

284:                                              ; preds = %67
  %285 = add nsw i64 %64, %8, !dbg !7
  %286 = shl i64 %285, 19, !dbg !7
  %287 = and i64 %286, 16252928, !dbg !7
  %288 = shl nsw i64 %66, 7, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %289 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %287, !dbg !7
  %290 = getelementptr [2 x i8], ptr addrspace(1) %289, i64 %288, !dbg !7
  br label %291, !dbg !7

291:                                              ; preds = %294, %284
  %292 = phi i64 [ %309, %294 ], [ %20, %284 ], !dbg !7
  %293 = icmp samesign ult i64 %292, 2048, !dbg !7
  br i1 %293, label %294, label %310, !dbg !7

294:                                              ; preds = %291
  %295 = shl nuw nsw i64 %292, 1, !dbg !7
  %296 = and i64 %295, 30, !dbg !7
  %297 = shl nuw nsw i64 %292, 1, !dbg !7
  %298 = and i64 %297, 3968, !dbg !7
  %299 = shl nuw nsw i64 %292, 3, !dbg !7
  %300 = and i64 %299, 16256, !dbg !7
  %301 = getelementptr [2 x i8], ptr addrspace(1) %290, i64 %300, !dbg !7
  %302 = getelementptr [2 x i8], ptr addrspace(1) %301, i64 %296, !dbg !7
  %303 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %298, !dbg !7
  %304 = ptrtoint ptr addrspace(3) %303 to i64, !dbg !7
  %305 = trunc i64 %304 to i32, !dbg !7
  %306 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %305), !dbg !7
  %307 = zext i32 %306 to i64, !dbg !7
  %308 = inttoptr i64 %307 to ptr addrspace(3), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %302, ptr addrspace(3) %308, i32 4, i32 0, i32 0), !dbg !7
  %309 = add nuw nsw i64 %292, 256, !dbg !7
  br label %291, !dbg !7

310:                                              ; preds = %291
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %311 = or disjoint i64 %63, 127, !dbg !15
  %312 = icmp uge i64 %9, %311, !dbg !15
  %313 = and i1 %27, %312, !dbg !15
  br i1 %313, label %314, label %315, !dbg !15

314:                                              ; preds = %310
  br label %486, !dbg !15

315:                                              ; preds = %310
  %316 = add nsw i64 %9, %20, !dbg !15
  %317 = shl nuw nsw i64 %22, 4, !dbg !15
  %318 = add nsw i64 %316, %317, !dbg !15
  %319 = and i32 %19, 240, !dbg !15
  %index.compact = zext nneg i32 %319 to i64, !dbg !15
  %320 = sub nsw i64 %318, %index.compact, !dbg !15
  %321 = shl nuw nsw i64 %24, 4, !dbg !15
  %322 = or disjoint i64 %63, %321, !dbg !15
  %323 = add nsw i64 %320, 1, !dbg !23
  %.not36 = icmp ugt i64 %323, %322, !dbg !24
  %324 = select i1 %.not36, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %325 = or disjoint i64 %322, 1, !dbg !15
  %.not37 = icmp ugt i64 %323, %325, !dbg !24
  %326 = select i1 %.not37, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %327 = or disjoint i64 %322, 2, !dbg !15
  %.not38 = icmp ugt i64 %323, %327, !dbg !24
  %328 = select i1 %.not38, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %329 = or disjoint i64 %322, 3, !dbg !15
  %.not39 = icmp ugt i64 %323, %329, !dbg !24
  %330 = select i1 %.not39, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %331 = or disjoint i64 %322, 4, !dbg !15
  %.not40 = icmp ugt i64 %323, %331, !dbg !24
  %332 = select i1 %.not40, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %333 = or disjoint i64 %322, 5, !dbg !15
  %.not41 = icmp ugt i64 %323, %333, !dbg !24
  %334 = select i1 %.not41, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %335 = or disjoint i64 %322, 6, !dbg !15
  %.not42 = icmp ugt i64 %323, %335, !dbg !24
  %336 = select i1 %.not42, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %337 = or disjoint i64 %322, 7, !dbg !15
  %.not43 = icmp ugt i64 %323, %337, !dbg !24
  %338 = select i1 %.not43, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %339 = or disjoint i64 %322, 8, !dbg !15
  %.not44 = icmp ugt i64 %323, %339, !dbg !24
  %340 = select i1 %.not44, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %341 = or disjoint i64 %322, 9, !dbg !15
  %.not45 = icmp ugt i64 %323, %341, !dbg !24
  %342 = select i1 %.not45, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %343 = or disjoint i64 %322, 10, !dbg !15
  %.not46 = icmp ugt i64 %323, %343, !dbg !24
  %344 = select i1 %.not46, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %345 = or disjoint i64 %322, 11, !dbg !15
  %.not47 = icmp ugt i64 %323, %345, !dbg !24
  %346 = select i1 %.not47, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %347 = or disjoint i64 %322, 12, !dbg !15
  %.not48 = icmp ugt i64 %323, %347, !dbg !24
  %348 = select i1 %.not48, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %349 = or disjoint i64 %322, 13, !dbg !15
  %.not49 = icmp ugt i64 %323, %349, !dbg !24
  %350 = select i1 %.not49, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %351 = or disjoint i64 %322, 14, !dbg !15
  %.not50 = icmp ugt i64 %323, %351, !dbg !24
  %352 = select i1 %.not50, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %353 = or disjoint i64 %322, 15, !dbg !15
  %.not51 = icmp ugt i64 %323, %353, !dbg !24
  %354 = select i1 %.not51, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %355 = or disjoint i64 %322, 64, !dbg !15
  %.not52 = icmp ugt i64 %323, %355, !dbg !24
  %356 = select i1 %.not52, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %357 = or disjoint i64 %322, 65, !dbg !15
  %.not53 = icmp ugt i64 %323, %357, !dbg !24
  %358 = select i1 %.not53, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %359 = or disjoint i64 %322, 66, !dbg !15
  %.not54 = icmp ugt i64 %323, %359, !dbg !24
  %360 = select i1 %.not54, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %361 = or disjoint i64 %322, 67, !dbg !15
  %.not55 = icmp ugt i64 %323, %361, !dbg !24
  %362 = select i1 %.not55, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %363 = or disjoint i64 %322, 68, !dbg !15
  %.not56 = icmp ugt i64 %323, %363, !dbg !24
  %364 = select i1 %.not56, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %365 = or disjoint i64 %322, 69, !dbg !15
  %.not57 = icmp ugt i64 %323, %365, !dbg !24
  %366 = select i1 %.not57, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %367 = or disjoint i64 %322, 70, !dbg !15
  %.not58 = icmp ugt i64 %323, %367, !dbg !24
  %368 = select i1 %.not58, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %369 = or disjoint i64 %322, 71, !dbg !15
  %.not59 = icmp ugt i64 %323, %369, !dbg !24
  %370 = select i1 %.not59, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %371 = or disjoint i64 %322, 72, !dbg !15
  %.not60 = icmp ugt i64 %323, %371, !dbg !24
  %372 = select i1 %.not60, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %373 = or disjoint i64 %322, 73, !dbg !15
  %.not61 = icmp ugt i64 %323, %373, !dbg !24
  %374 = select i1 %.not61, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %375 = or disjoint i64 %322, 74, !dbg !15
  %.not62 = icmp ugt i64 %323, %375, !dbg !24
  %376 = select i1 %.not62, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %377 = or disjoint i64 %322, 75, !dbg !15
  %.not63 = icmp ugt i64 %323, %377, !dbg !24
  %378 = select i1 %.not63, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %379 = or disjoint i64 %322, 76, !dbg !15
  %.not64 = icmp ugt i64 %323, %379, !dbg !24
  %380 = select i1 %.not64, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %381 = or disjoint i64 %322, 77, !dbg !15
  %.not65 = icmp ugt i64 %323, %381, !dbg !24
  %382 = select i1 %.not65, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %383 = or disjoint i64 %322, 78, !dbg !15
  %.not66 = icmp ugt i64 %323, %383, !dbg !24
  %384 = select i1 %.not66, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %385 = or disjoint i64 %322, 79, !dbg !15
  %.not67 = icmp ugt i64 %323, %385, !dbg !24
  %386 = select i1 %.not67, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %387 = add nsw i64 %320, 65, !dbg !23
  %.not68 = icmp ugt i64 %387, %322, !dbg !24
  %388 = select i1 %.not68, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not69 = icmp ugt i64 %387, %325, !dbg !24
  %389 = select i1 %.not69, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not70 = icmp ugt i64 %387, %327, !dbg !24
  %390 = select i1 %.not70, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not71 = icmp ugt i64 %387, %329, !dbg !24
  %391 = select i1 %.not71, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not72 = icmp ugt i64 %387, %331, !dbg !24
  %392 = select i1 %.not72, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not73 = icmp ugt i64 %387, %333, !dbg !24
  %393 = select i1 %.not73, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not74 = icmp ugt i64 %387, %335, !dbg !24
  %394 = select i1 %.not74, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not75 = icmp ugt i64 %387, %337, !dbg !24
  %395 = select i1 %.not75, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not76 = icmp ugt i64 %387, %339, !dbg !24
  %396 = select i1 %.not76, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not77 = icmp ugt i64 %387, %341, !dbg !24
  %397 = select i1 %.not77, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not78 = icmp ugt i64 %387, %343, !dbg !24
  %398 = select i1 %.not78, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not79 = icmp ugt i64 %387, %345, !dbg !24
  %399 = select i1 %.not79, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not80 = icmp ugt i64 %387, %347, !dbg !24
  %400 = select i1 %.not80, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not81 = icmp ugt i64 %387, %349, !dbg !24
  %401 = select i1 %.not81, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not82 = icmp ugt i64 %387, %351, !dbg !24
  %402 = select i1 %.not82, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not83 = icmp ugt i64 %387, %353, !dbg !24
  %403 = select i1 %.not83, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not84 = icmp ugt i64 %387, %355, !dbg !24
  %404 = select i1 %.not84, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not85 = icmp ugt i64 %387, %357, !dbg !24
  %405 = select i1 %.not85, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not86 = icmp ugt i64 %387, %359, !dbg !24
  %406 = select i1 %.not86, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not87 = icmp ugt i64 %387, %361, !dbg !24
  %407 = select i1 %.not87, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not88 = icmp ugt i64 %387, %363, !dbg !24
  %408 = select i1 %.not88, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not89 = icmp ugt i64 %387, %365, !dbg !24
  %409 = select i1 %.not89, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not90 = icmp ugt i64 %387, %367, !dbg !24
  %410 = select i1 %.not90, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not91 = icmp ugt i64 %387, %369, !dbg !24
  %411 = select i1 %.not91, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not92 = icmp ugt i64 %387, %371, !dbg !24
  %412 = select i1 %.not92, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not93 = icmp ugt i64 %387, %373, !dbg !24
  %413 = select i1 %.not93, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not94 = icmp ugt i64 %387, %375, !dbg !24
  %414 = select i1 %.not94, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not95 = icmp ugt i64 %387, %377, !dbg !24
  %415 = select i1 %.not95, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not96 = icmp ugt i64 %387, %379, !dbg !24
  %416 = select i1 %.not96, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not97 = icmp ugt i64 %387, %381, !dbg !24
  %417 = select i1 %.not97, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not98 = icmp ugt i64 %387, %383, !dbg !24
  %418 = select i1 %.not98, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %.not99 = icmp ugt i64 %387, %385, !dbg !24
  %419 = select i1 %.not99, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %420 = insertelement <32 x float> poison, float %324, i64 0, !dbg !15
  %421 = insertelement <32 x float> %420, float %326, i64 1, !dbg !15
  %422 = insertelement <32 x float> %421, float %328, i64 2, !dbg !15
  %423 = insertelement <32 x float> %422, float %330, i64 3, !dbg !15
  %424 = insertelement <32 x float> %423, float %332, i64 4, !dbg !15
  %425 = insertelement <32 x float> %424, float %334, i64 5, !dbg !15
  %426 = insertelement <32 x float> %425, float %336, i64 6, !dbg !15
  %427 = insertelement <32 x float> %426, float %338, i64 7, !dbg !15
  %428 = insertelement <32 x float> %427, float %340, i64 8, !dbg !15
  %429 = insertelement <32 x float> %428, float %342, i64 9, !dbg !15
  %430 = insertelement <32 x float> %429, float %344, i64 10, !dbg !15
  %431 = insertelement <32 x float> %430, float %346, i64 11, !dbg !15
  %432 = insertelement <32 x float> %431, float %348, i64 12, !dbg !15
  %433 = insertelement <32 x float> %432, float %350, i64 13, !dbg !15
  %434 = insertelement <32 x float> %433, float %352, i64 14, !dbg !15
  %435 = insertelement <32 x float> %434, float %354, i64 15, !dbg !15
  %436 = insertelement <32 x float> %435, float %356, i64 16, !dbg !15
  %437 = insertelement <32 x float> %436, float %358, i64 17, !dbg !15
  %438 = insertelement <32 x float> %437, float %360, i64 18, !dbg !15
  %439 = insertelement <32 x float> %438, float %362, i64 19, !dbg !15
  %440 = insertelement <32 x float> %439, float %364, i64 20, !dbg !15
  %441 = insertelement <32 x float> %440, float %366, i64 21, !dbg !15
  %442 = insertelement <32 x float> %441, float %368, i64 22, !dbg !15
  %443 = insertelement <32 x float> %442, float %370, i64 23, !dbg !15
  %444 = insertelement <32 x float> %443, float %372, i64 24, !dbg !15
  %445 = insertelement <32 x float> %444, float %374, i64 25, !dbg !15
  %446 = insertelement <32 x float> %445, float %376, i64 26, !dbg !15
  %447 = insertelement <32 x float> %446, float %378, i64 27, !dbg !15
  %448 = insertelement <32 x float> %447, float %380, i64 28, !dbg !15
  %449 = insertelement <32 x float> %448, float %382, i64 29, !dbg !15
  %450 = insertelement <32 x float> %449, float %384, i64 30, !dbg !15
  %451 = insertelement <32 x float> %450, float %386, i64 31, !dbg !15
  %452 = insertvalue [2 x <32 x float>] poison, <32 x float> %451, 0, !dbg !15
  %453 = insertelement <32 x float> poison, float %388, i64 0, !dbg !15
  %454 = insertelement <32 x float> %453, float %389, i64 1, !dbg !15
  %455 = insertelement <32 x float> %454, float %390, i64 2, !dbg !15
  %456 = insertelement <32 x float> %455, float %391, i64 3, !dbg !15
  %457 = insertelement <32 x float> %456, float %392, i64 4, !dbg !15
  %458 = insertelement <32 x float> %457, float %393, i64 5, !dbg !15
  %459 = insertelement <32 x float> %458, float %394, i64 6, !dbg !15
  %460 = insertelement <32 x float> %459, float %395, i64 7, !dbg !15
  %461 = insertelement <32 x float> %460, float %396, i64 8, !dbg !15
  %462 = insertelement <32 x float> %461, float %397, i64 9, !dbg !15
  %463 = insertelement <32 x float> %462, float %398, i64 10, !dbg !15
  %464 = insertelement <32 x float> %463, float %399, i64 11, !dbg !15
  %465 = insertelement <32 x float> %464, float %400, i64 12, !dbg !15
  %466 = insertelement <32 x float> %465, float %401, i64 13, !dbg !15
  %467 = insertelement <32 x float> %466, float %402, i64 14, !dbg !15
  %468 = insertelement <32 x float> %467, float %403, i64 15, !dbg !15
  %469 = insertelement <32 x float> %468, float %404, i64 16, !dbg !15
  %470 = insertelement <32 x float> %469, float %405, i64 17, !dbg !15
  %471 = insertelement <32 x float> %470, float %406, i64 18, !dbg !15
  %472 = insertelement <32 x float> %471, float %407, i64 19, !dbg !15
  %473 = insertelement <32 x float> %472, float %408, i64 20, !dbg !15
  %474 = insertelement <32 x float> %473, float %409, i64 21, !dbg !15
  %475 = insertelement <32 x float> %474, float %410, i64 22, !dbg !15
  %476 = insertelement <32 x float> %475, float %411, i64 23, !dbg !15
  %477 = insertelement <32 x float> %476, float %412, i64 24, !dbg !15
  %478 = insertelement <32 x float> %477, float %413, i64 25, !dbg !15
  %479 = insertelement <32 x float> %478, float %414, i64 26, !dbg !15
  %480 = insertelement <32 x float> %479, float %415, i64 27, !dbg !15
  %481 = insertelement <32 x float> %480, float %416, i64 28, !dbg !15
  %482 = insertelement <32 x float> %481, float %417, i64 29, !dbg !15
  %483 = insertelement <32 x float> %482, float %418, i64 30, !dbg !15
  %484 = insertelement <32 x float> %483, float %419, i64 31, !dbg !15
  %485 = insertvalue [2 x <32 x float>] %452, <32 x float> %484, 1, !dbg !15
  br label %486, !dbg !15

486:                                              ; preds = %314, %315
  %487 = phi [2 x <32 x float>] [ %485, %315 ], [ zeroinitializer, %314 ], !dbg !15
  br label %488, !dbg !15

488:                                              ; preds = %486
  %489 = extractvalue [2 x <32 x float>] %69, 0, !dbg !26
  %490 = extractvalue [2 x <32 x float>] %487, 0, !dbg !26
  %491 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %492 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %493 = fadd <4 x float> %491, %492, !dbg !26
  %494 = extractelement <4 x float> %493, i64 0, !dbg !27
  %495 = call float @llvm.amdgcn.exp2.f32(float %494), !dbg !27
  %496 = insertelement <4 x float> poison, float %495, i64 0, !dbg !27
  %497 = extractelement <4 x float> %493, i64 1, !dbg !27
  %498 = call float @llvm.amdgcn.exp2.f32(float %497), !dbg !27
  %499 = fadd float %495, %498, !dbg !28
  %500 = insertelement <4 x float> %496, float %498, i64 1, !dbg !27
  %501 = extractelement <4 x float> %493, i64 2, !dbg !27
  %502 = call float @llvm.amdgcn.exp2.f32(float %501), !dbg !27
  %503 = fadd float %499, %502, !dbg !28
  %504 = insertelement <4 x float> %500, float %502, i64 2, !dbg !27
  %505 = extractelement <4 x float> %493, i64 3, !dbg !27
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !27
  %507 = fadd float %503, %506, !dbg !28
  %508 = insertelement <4 x float> %504, float %506, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %508) #8, !dbg !29, !frisk.register_lifetimes !1
  %509 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %509) #8, !dbg !26, !frisk.register_lifetimes !1
  %510 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %511 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %512 = fadd <4 x float> %510, %511, !dbg !26
  %513 = extractelement <4 x float> %512, i64 0, !dbg !27
  %514 = call float @llvm.amdgcn.exp2.f32(float %513), !dbg !27
  %515 = fadd float %507, %514, !dbg !28
  %516 = insertelement <4 x float> poison, float %514, i64 0, !dbg !27
  %517 = extractelement <4 x float> %512, i64 1, !dbg !27
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !27
  %519 = fadd float %515, %518, !dbg !28
  %520 = insertelement <4 x float> %516, float %518, i64 1, !dbg !27
  %521 = extractelement <4 x float> %512, i64 2, !dbg !27
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !27
  %523 = fadd float %519, %522, !dbg !28
  %524 = insertelement <4 x float> %520, float %522, i64 2, !dbg !27
  %525 = extractelement <4 x float> %512, i64 3, !dbg !27
  %526 = call float @llvm.amdgcn.exp2.f32(float %525), !dbg !27
  %527 = fadd float %523, %526, !dbg !28
  %528 = insertelement <4 x float> %524, float %526, i64 3, !dbg !27
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %528) #8, !dbg !29, !frisk.register_lifetimes !1
  %529 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !29
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %529) #8, !dbg !26, !frisk.register_lifetimes !1
  %530 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %531 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %532 = fadd <4 x float> %530, %531, !dbg !26
  %533 = extractelement <4 x float> %532, i64 0, !dbg !27
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !27
  %535 = fadd float %527, %534, !dbg !28
  %536 = insertelement <4 x float> poison, float %534, i64 0, !dbg !27
  %537 = extractelement <4 x float> %532, i64 1, !dbg !27
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !27
  %539 = fadd float %535, %538, !dbg !28
  %540 = insertelement <4 x float> %536, float %538, i64 1, !dbg !27
  %541 = extractelement <4 x float> %532, i64 2, !dbg !27
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !27
  %543 = fadd float %539, %542, !dbg !28
  %544 = insertelement <4 x float> %540, float %542, i64 2, !dbg !27
  %545 = extractelement <4 x float> %532, i64 3, !dbg !27
  %546 = call float @llvm.amdgcn.exp2.f32(float %545), !dbg !27
  %547 = fadd float %543, %546, !dbg !28
  %548 = insertelement <4 x float> %544, float %546, i64 3, !dbg !27
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %548) #8, !dbg !29, !frisk.register_lifetimes !1
  %549 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !29
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %549) #8, !dbg !26, !frisk.register_lifetimes !1
  %550 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %551 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %552 = fadd <4 x float> %550, %551, !dbg !26
  %553 = extractelement <4 x float> %552, i64 0, !dbg !27
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !27
  %555 = fadd float %547, %554, !dbg !28
  %556 = insertelement <4 x float> poison, float %554, i64 0, !dbg !27
  %557 = extractelement <4 x float> %552, i64 1, !dbg !27
  %558 = call float @llvm.amdgcn.exp2.f32(float %557), !dbg !27
  %559 = fadd float %555, %558, !dbg !28
  %560 = insertelement <4 x float> %556, float %558, i64 1, !dbg !27
  %561 = extractelement <4 x float> %552, i64 2, !dbg !27
  %562 = call float @llvm.amdgcn.exp2.f32(float %561), !dbg !27
  %563 = fadd float %559, %562, !dbg !28
  %564 = insertelement <4 x float> %560, float %562, i64 2, !dbg !27
  %565 = extractelement <4 x float> %552, i64 3, !dbg !27
  %566 = call float @llvm.amdgcn.exp2.f32(float %565), !dbg !27
  %567 = fadd float %563, %566, !dbg !28
  %568 = insertelement <4 x float> %564, float %566, i64 3, !dbg !27
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %568) #8, !dbg !29, !frisk.register_lifetimes !1
  %569 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !29
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %569) #8, !dbg !26, !frisk.register_lifetimes !1
  %570 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %571 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %572 = fadd <4 x float> %570, %571, !dbg !26
  %573 = extractelement <4 x float> %572, i64 0, !dbg !27
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !27
  %575 = fadd float %567, %574, !dbg !28
  %576 = insertelement <4 x float> poison, float %574, i64 0, !dbg !27
  %577 = extractelement <4 x float> %572, i64 1, !dbg !27
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !27
  %579 = fadd float %575, %578, !dbg !28
  %580 = insertelement <4 x float> %576, float %578, i64 1, !dbg !27
  %581 = extractelement <4 x float> %572, i64 2, !dbg !27
  %582 = call float @llvm.amdgcn.exp2.f32(float %581), !dbg !27
  %583 = fadd float %579, %582, !dbg !28
  %584 = insertelement <4 x float> %580, float %582, i64 2, !dbg !27
  %585 = extractelement <4 x float> %572, i64 3, !dbg !27
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !27
  %587 = fadd float %583, %586, !dbg !28
  %588 = insertelement <4 x float> %584, float %586, i64 3, !dbg !27
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #8, !dbg !29, !frisk.register_lifetimes !1
  %589 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !29
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #8, !dbg !26, !frisk.register_lifetimes !1
  %590 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %591 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %592 = fadd <4 x float> %590, %591, !dbg !26
  %593 = extractelement <4 x float> %592, i64 0, !dbg !27
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !27
  %595 = fadd float %587, %594, !dbg !28
  %596 = insertelement <4 x float> poison, float %594, i64 0, !dbg !27
  %597 = extractelement <4 x float> %592, i64 1, !dbg !27
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !27
  %599 = fadd float %595, %598, !dbg !28
  %600 = insertelement <4 x float> %596, float %598, i64 1, !dbg !27
  %601 = extractelement <4 x float> %592, i64 2, !dbg !27
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !27
  %603 = fadd float %599, %602, !dbg !28
  %604 = insertelement <4 x float> %600, float %602, i64 2, !dbg !27
  %605 = extractelement <4 x float> %592, i64 3, !dbg !27
  %606 = call float @llvm.amdgcn.exp2.f32(float %605), !dbg !27
  %607 = fadd float %603, %606, !dbg !28
  %608 = insertelement <4 x float> %604, float %606, i64 3, !dbg !27
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %608) #8, !dbg !29, !frisk.register_lifetimes !1
  %609 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !29
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %609) #8, !dbg !26, !frisk.register_lifetimes !1
  %610 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %611 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %612 = fadd <4 x float> %610, %611, !dbg !26
  %613 = extractelement <4 x float> %612, i64 0, !dbg !27
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !27
  %615 = fadd float %607, %614, !dbg !28
  %616 = insertelement <4 x float> poison, float %614, i64 0, !dbg !27
  %617 = extractelement <4 x float> %612, i64 1, !dbg !27
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !27
  %619 = fadd float %615, %618, !dbg !28
  %620 = insertelement <4 x float> %616, float %618, i64 1, !dbg !27
  %621 = extractelement <4 x float> %612, i64 2, !dbg !27
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !27
  %623 = fadd float %619, %622, !dbg !28
  %624 = insertelement <4 x float> %620, float %622, i64 2, !dbg !27
  %625 = extractelement <4 x float> %612, i64 3, !dbg !27
  %626 = call float @llvm.amdgcn.exp2.f32(float %625), !dbg !27
  %627 = fadd float %623, %626, !dbg !28
  %628 = insertelement <4 x float> %624, float %626, i64 3, !dbg !27
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %628) #8, !dbg !29, !frisk.register_lifetimes !1
  %629 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !29
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %629) #8, !dbg !26, !frisk.register_lifetimes !1
  %630 = shufflevector <32 x float> %489, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %631 = shufflevector <32 x float> %490, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %632 = fadd <4 x float> %630, %631, !dbg !26
  %633 = extractelement <4 x float> %632, i64 0, !dbg !27
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !27
  %635 = fadd float %627, %634, !dbg !28
  %636 = insertelement <4 x float> poison, float %634, i64 0, !dbg !27
  %637 = extractelement <4 x float> %632, i64 1, !dbg !27
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !27
  %639 = fadd float %635, %638, !dbg !28
  %640 = insertelement <4 x float> %636, float %638, i64 1, !dbg !27
  %641 = extractelement <4 x float> %632, i64 2, !dbg !27
  %642 = call float @llvm.amdgcn.exp2.f32(float %641), !dbg !27
  %643 = fadd float %639, %642, !dbg !28
  %644 = insertelement <4 x float> %640, float %642, i64 2, !dbg !27
  %645 = extractelement <4 x float> %632, i64 3, !dbg !27
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !27
  %647 = fadd float %643, %646, !dbg !28
  %648 = insertelement <4 x float> %644, float %646, i64 3, !dbg !27
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %648) #8, !dbg !29, !frisk.register_lifetimes !1
  %649 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !29
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %649) #8, !dbg !26, !frisk.register_lifetimes !1
  %650 = extractvalue [2 x <32 x float>] %69, 1, !dbg !26
  %651 = extractvalue [2 x <32 x float>] %487, 1, !dbg !26
  %652 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %653 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %654 = fadd <4 x float> %652, %653, !dbg !26
  %655 = extractelement <4 x float> %654, i64 0, !dbg !27
  %656 = call float @llvm.amdgcn.exp2.f32(float %655), !dbg !27
  %657 = insertelement <4 x float> poison, float %656, i64 0, !dbg !27
  %658 = extractelement <4 x float> %654, i64 1, !dbg !27
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !27
  %660 = fadd float %656, %659, !dbg !28
  %661 = insertelement <4 x float> %657, float %659, i64 1, !dbg !27
  %662 = extractelement <4 x float> %654, i64 2, !dbg !27
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !27
  %664 = fadd float %660, %663, !dbg !28
  %665 = insertelement <4 x float> %661, float %663, i64 2, !dbg !27
  %666 = extractelement <4 x float> %654, i64 3, !dbg !27
  %667 = call float @llvm.amdgcn.exp2.f32(float %666), !dbg !27
  %668 = fadd float %664, %667, !dbg !28
  %669 = insertelement <4 x float> %665, float %667, i64 3, !dbg !27
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %669) #8, !dbg !29, !frisk.register_lifetimes !1
  %670 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !29
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %670) #8, !dbg !26, !frisk.register_lifetimes !1
  %671 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %672 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %673 = fadd <4 x float> %671, %672, !dbg !26
  %674 = extractelement <4 x float> %673, i64 0, !dbg !27
  %675 = call float @llvm.amdgcn.exp2.f32(float %674), !dbg !27
  %676 = fadd float %668, %675, !dbg !28
  %677 = insertelement <4 x float> poison, float %675, i64 0, !dbg !27
  %678 = extractelement <4 x float> %673, i64 1, !dbg !27
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !27
  %680 = fadd float %676, %679, !dbg !28
  %681 = insertelement <4 x float> %677, float %679, i64 1, !dbg !27
  %682 = extractelement <4 x float> %673, i64 2, !dbg !27
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !27
  %684 = fadd float %680, %683, !dbg !28
  %685 = insertelement <4 x float> %681, float %683, i64 2, !dbg !27
  %686 = extractelement <4 x float> %673, i64 3, !dbg !27
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !27
  %688 = fadd float %684, %687, !dbg !28
  %689 = insertelement <4 x float> %685, float %687, i64 3, !dbg !27
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %689) #8, !dbg !29, !frisk.register_lifetimes !1
  %690 = fptrunc <4 x float> %math.result250 to <4 x half>, !dbg !29
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %690) #8, !dbg !26, !frisk.register_lifetimes !1
  %691 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %692 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %693 = fadd <4 x float> %691, %692, !dbg !26
  %694 = extractelement <4 x float> %693, i64 0, !dbg !27
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !27
  %696 = fadd float %688, %695, !dbg !28
  %697 = insertelement <4 x float> poison, float %695, i64 0, !dbg !27
  %698 = extractelement <4 x float> %693, i64 1, !dbg !27
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !27
  %700 = fadd float %696, %699, !dbg !28
  %701 = insertelement <4 x float> %697, float %699, i64 1, !dbg !27
  %702 = extractelement <4 x float> %693, i64 2, !dbg !27
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !27
  %704 = fadd float %700, %703, !dbg !28
  %705 = insertelement <4 x float> %701, float %703, i64 2, !dbg !27
  %706 = extractelement <4 x float> %693, i64 3, !dbg !27
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !27
  %708 = fadd float %704, %707, !dbg !28
  %709 = insertelement <4 x float> %705, float %707, i64 3, !dbg !27
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %709) #8, !dbg !29, !frisk.register_lifetimes !1
  %710 = fptrunc <4 x float> %math.result252 to <4 x half>, !dbg !29
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %710) #8, !dbg !26, !frisk.register_lifetimes !1
  %711 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %712 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %713 = fadd <4 x float> %711, %712, !dbg !26
  %714 = extractelement <4 x float> %713, i64 0, !dbg !27
  %715 = call float @llvm.amdgcn.exp2.f32(float %714), !dbg !27
  %716 = fadd float %708, %715, !dbg !28
  %717 = insertelement <4 x float> poison, float %715, i64 0, !dbg !27
  %718 = extractelement <4 x float> %713, i64 1, !dbg !27
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !27
  %720 = fadd float %716, %719, !dbg !28
  %721 = insertelement <4 x float> %717, float %719, i64 1, !dbg !27
  %722 = extractelement <4 x float> %713, i64 2, !dbg !27
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !27
  %724 = fadd float %720, %723, !dbg !28
  %725 = insertelement <4 x float> %721, float %723, i64 2, !dbg !27
  %726 = extractelement <4 x float> %713, i64 3, !dbg !27
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !27
  %728 = fadd float %724, %727, !dbg !28
  %729 = insertelement <4 x float> %725, float %727, i64 3, !dbg !27
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #8, !dbg !29, !frisk.register_lifetimes !1
  %730 = fptrunc <4 x float> %math.result254 to <4 x half>, !dbg !29
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #8, !dbg !26, !frisk.register_lifetimes !1
  %731 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %732 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %733 = fadd <4 x float> %731, %732, !dbg !26
  %734 = extractelement <4 x float> %733, i64 0, !dbg !27
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !27
  %736 = fadd float %728, %735, !dbg !28
  %737 = insertelement <4 x float> poison, float %735, i64 0, !dbg !27
  %738 = extractelement <4 x float> %733, i64 1, !dbg !27
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !27
  %740 = fadd float %736, %739, !dbg !28
  %741 = insertelement <4 x float> %737, float %739, i64 1, !dbg !27
  %742 = extractelement <4 x float> %733, i64 2, !dbg !27
  %743 = call float @llvm.amdgcn.exp2.f32(float %742), !dbg !27
  %744 = fadd float %740, %743, !dbg !28
  %745 = insertelement <4 x float> %741, float %743, i64 2, !dbg !27
  %746 = extractelement <4 x float> %733, i64 3, !dbg !27
  %747 = call float @llvm.amdgcn.exp2.f32(float %746), !dbg !27
  %748 = fadd float %744, %747, !dbg !28
  %749 = insertelement <4 x float> %745, float %747, i64 3, !dbg !27
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %749) #8, !dbg !29, !frisk.register_lifetimes !1
  %750 = fptrunc <4 x float> %math.result256 to <4 x half>, !dbg !29
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %750) #8, !dbg !26, !frisk.register_lifetimes !1
  %751 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %752 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %753 = fadd <4 x float> %751, %752, !dbg !26
  %754 = extractelement <4 x float> %753, i64 0, !dbg !27
  %755 = call float @llvm.amdgcn.exp2.f32(float %754), !dbg !27
  %756 = fadd float %748, %755, !dbg !28
  %757 = insertelement <4 x float> poison, float %755, i64 0, !dbg !27
  %758 = extractelement <4 x float> %753, i64 1, !dbg !27
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !27
  %760 = fadd float %756, %759, !dbg !28
  %761 = insertelement <4 x float> %757, float %759, i64 1, !dbg !27
  %762 = extractelement <4 x float> %753, i64 2, !dbg !27
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !27
  %764 = fadd float %760, %763, !dbg !28
  %765 = insertelement <4 x float> %761, float %763, i64 2, !dbg !27
  %766 = extractelement <4 x float> %753, i64 3, !dbg !27
  %767 = call float @llvm.amdgcn.exp2.f32(float %766), !dbg !27
  %768 = fadd float %764, %767, !dbg !28
  %769 = insertelement <4 x float> %765, float %767, i64 3, !dbg !27
  %math.result258 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %769) #8, !dbg !29, !frisk.register_lifetimes !1
  %770 = fptrunc <4 x float> %math.result258 to <4 x half>, !dbg !29
  %math.packed259 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %770) #8, !dbg !26, !frisk.register_lifetimes !1
  %771 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %772 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %773 = fadd <4 x float> %771, %772, !dbg !26
  %774 = extractelement <4 x float> %773, i64 0, !dbg !27
  %775 = call float @llvm.amdgcn.exp2.f32(float %774), !dbg !27
  %776 = fadd float %768, %775, !dbg !28
  %777 = insertelement <4 x float> poison, float %775, i64 0, !dbg !27
  %778 = extractelement <4 x float> %773, i64 1, !dbg !27
  %779 = call float @llvm.amdgcn.exp2.f32(float %778), !dbg !27
  %780 = fadd float %776, %779, !dbg !28
  %781 = insertelement <4 x float> %777, float %779, i64 1, !dbg !27
  %782 = extractelement <4 x float> %773, i64 2, !dbg !27
  %783 = call float @llvm.amdgcn.exp2.f32(float %782), !dbg !27
  %784 = fadd float %780, %783, !dbg !28
  %785 = insertelement <4 x float> %781, float %783, i64 2, !dbg !27
  %786 = extractelement <4 x float> %773, i64 3, !dbg !27
  %787 = call float @llvm.amdgcn.exp2.f32(float %786), !dbg !27
  %788 = fadd float %784, %787, !dbg !28
  %789 = insertelement <4 x float> %785, float %787, i64 3, !dbg !27
  %math.result260 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %789) #8, !dbg !29, !frisk.register_lifetimes !1
  %790 = fptrunc <4 x float> %math.result260 to <4 x half>, !dbg !29
  %math.packed261 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %790) #8, !dbg !26, !frisk.register_lifetimes !1
  %791 = shufflevector <32 x float> %650, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %792 = shufflevector <32 x float> %651, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %793 = fadd <4 x float> %791, %792, !dbg !26
  %794 = extractelement <4 x float> %793, i64 0, !dbg !27
  %795 = call float @llvm.amdgcn.exp2.f32(float %794), !dbg !27
  %796 = fadd float %788, %795, !dbg !28
  %797 = insertelement <4 x float> poison, float %795, i64 0, !dbg !27
  %798 = extractelement <4 x float> %793, i64 1, !dbg !27
  %799 = call float @llvm.amdgcn.exp2.f32(float %798), !dbg !27
  %800 = fadd float %796, %799, !dbg !28
  %801 = insertelement <4 x float> %797, float %799, i64 1, !dbg !27
  %802 = extractelement <4 x float> %793, i64 2, !dbg !27
  %803 = call float @llvm.amdgcn.exp2.f32(float %802), !dbg !27
  %804 = fadd float %800, %803, !dbg !28
  %805 = insertelement <4 x float> %801, float %803, i64 2, !dbg !27
  %806 = extractelement <4 x float> %793, i64 3, !dbg !27
  %807 = call float @llvm.amdgcn.exp2.f32(float %806), !dbg !27
  %808 = fadd float %804, %807, !dbg !28
  %809 = insertelement <4 x float> %805, float %807, i64 3, !dbg !27
  %math.result262 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #8, !dbg !29, !frisk.register_lifetimes !1
  %810 = fptrunc <4 x float> %math.result262 to <4 x half>, !dbg !29
  %math.packed263 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #8, !dbg !28, !frisk.register_lifetimes !1
  %811 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %812 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %811), !dbg !28
  %813 = add nuw nsw i32 %812, 64, !dbg !28
  %814 = xor i32 %812, 16, !dbg !28
  %815 = and i32 %813, 192, !dbg !28
  %816 = icmp samesign ult i32 %814, %815, !dbg !28
  %817 = select i1 %816, i32 %814, i32 %812, !dbg !28
  %818 = shl nuw nsw i32 %817, 2, !dbg !28
  %819 = bitcast float %647 to i32, !dbg !28
  %820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %818, i32 %819), !dbg !28
  %821 = bitcast i32 %820 to float, !dbg !28
  %822 = fadd float %647, %821, !dbg !28
  %823 = xor i32 %812, 32, !dbg !28
  %824 = icmp samesign ult i32 %823, %815, !dbg !28
  %825 = select i1 %824, i32 %823, i32 %812, !dbg !28
  %826 = shl nuw nsw i32 %825, 2, !dbg !28
  %827 = bitcast float %822 to i32, !dbg !28
  %828 = call i32 @llvm.amdgcn.ds.bpermute(i32 %826, i32 %827), !dbg !28
  %829 = bitcast i32 %828 to float, !dbg !28
  %830 = fadd float %822, %829, !dbg !28
  %831 = extractvalue [2 x <1 x float>] %58, 0, !dbg !28
  %832 = extractelement <1 x float> %831, i64 0, !dbg !28
  %833 = fadd float %830, %832, !dbg !28
  %834 = bitcast float %808 to i32, !dbg !28
  %835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %818, i32 %834), !dbg !28
  %836 = bitcast i32 %835 to float, !dbg !28
  %837 = fadd float %808, %836, !dbg !28
  %838 = bitcast float %837 to i32, !dbg !28
  %839 = call i32 @llvm.amdgcn.ds.bpermute(i32 %826, i32 %838), !dbg !28
  %840 = bitcast i32 %839 to float, !dbg !28
  %841 = fadd float %837, %840, !dbg !28
  %842 = extractvalue [2 x <1 x float>] %58, 1, !dbg !28
  %843 = extractelement <1 x float> %842, i64 0, !dbg !28
  %844 = fadd float %841, %843, !dbg !28
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %845, !dbg !7

845:                                              ; preds = %848, %488
  %846 = phi i64 [ %896, %848 ], [ %20, %488 ], !dbg !7
  %847 = icmp samesign ult i64 %846, 2048, !dbg !7
  br i1 %847, label %848, label %897, !dbg !7

848:                                              ; preds = %845
  %849 = shl nuw nsw i64 %846, 1, !dbg !7
  %850 = lshr i64 %846, 2, !dbg !7
  %851 = lshr i64 %846, 5, !dbg !7
  %852 = and i64 %850, 24, !dbg !7
  %853 = lshr i64 %846, 4, !dbg !7
  %854 = lshr i64 %846, 7, !dbg !7
  %855 = sub nsw i64 %853, %854, !dbg !7
  %856 = shl nsw i64 %855, 2, !dbg !7
  %857 = and i64 %856, 4, !dbg !7
  %858 = sub nsw i64 %850, %851, !dbg !7
  %859 = and i64 %858, 3, !dbg !7
  %860 = shl nuw nsw i64 %854, 3, !dbg !7
  %861 = shl nuw nsw i64 %846, 1, !dbg !7
  %862 = and i64 %861, 4, !dbg !7
  %863 = or disjoint i64 %860, %862, !dbg !7
  %864 = and i64 %849, 2, !dbg !7
  %865 = or disjoint i64 %863, %864, !dbg !7
  %866 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %852, !dbg !7
  %867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %866, i64 %857, !dbg !7
  %868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %867, i64 %859, !dbg !7
  %.idx105 = shl nuw nsw i64 %865, 6, !dbg !7
  %869 = getelementptr inbounds nuw i8, ptr addrspace(3) %868, i64 %.idx105, !dbg !7
  %870 = load half, ptr addrspace(3) %869, align 2, !dbg !7
  %871 = lshr i64 %846, 2, !dbg !7
  %872 = lshr i64 %846, 5, !dbg !7
  %873 = and i64 %871, 24, !dbg !7
  %874 = lshr i64 %846, 4, !dbg !7
  %875 = lshr i64 %846, 7, !dbg !7
  %876 = sub nsw i64 %874, %875, !dbg !7
  %877 = shl nsw i64 %876, 2, !dbg !7
  %878 = and i64 %877, 4, !dbg !7
  %879 = sub nsw i64 %871, %872, !dbg !7
  %880 = and i64 %879, 3, !dbg !7
  %881 = shl nuw nsw i64 %875, 3, !dbg !7
  %882 = shl nuw nsw i64 %846, 1, !dbg !7
  %883 = and i64 %882, 4, !dbg !7
  %884 = or disjoint i64 %881, %883, !dbg !7
  %885 = and i64 %849, 2, !dbg !7
  %886 = or disjoint i64 %885, 1, !dbg !7
  %887 = or disjoint i64 %884, %886, !dbg !7
  %888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %873, !dbg !7
  %889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %888, i64 %878, !dbg !7
  %890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %889, i64 %880, !dbg !7
  %.idx106 = shl nuw nsw i64 %887, 6, !dbg !7
  %891 = getelementptr inbounds nuw i8, ptr addrspace(3) %890, i64 %.idx106, !dbg !7
  %892 = load half, ptr addrspace(3) %891, align 2, !dbg !7
  %893 = insertelement <2 x half> poison, half %870, i64 0, !dbg !7
  %894 = insertelement <2 x half> %893, half %892, i64 1, !dbg !7
  %895 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %849, !dbg !7
  store <2 x half> %894, ptr addrspace(3) %895, align 2, !dbg !7
  %896 = add nuw nsw i64 %846, 256, !dbg !7
  br label %845, !dbg !7

897:                                              ; preds = %845
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %898 = add nsw i64 %64, %8, !dbg !7
  %899 = shl i64 %898, 19, !dbg !7
  %900 = and i64 %899, 16252928, !dbg !7
  %901 = add i64 %900, %288, !dbg !7
  %902 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !7
  %903 = getelementptr i8, ptr addrspace(1) %902, i64 64, !dbg !7
  br label %904, !dbg !7

904:                                              ; preds = %907, %897
  %905 = phi i64 [ %922, %907 ], [ %20, %897 ], !dbg !7
  %906 = icmp samesign ult i64 %905, 2048, !dbg !7
  br i1 %906, label %907, label %923, !dbg !7

907:                                              ; preds = %904
  %908 = shl nuw nsw i64 %905, 1, !dbg !7
  %909 = and i64 %908, 30, !dbg !7
  %910 = shl nuw nsw i64 %905, 1, !dbg !7
  %911 = and i64 %910, 3968, !dbg !7
  %912 = shl nuw nsw i64 %905, 3, !dbg !7
  %913 = and i64 %912, 16256, !dbg !7
  %914 = getelementptr [2 x i8], ptr addrspace(1) %903, i64 %913, !dbg !7
  %915 = getelementptr [2 x i8], ptr addrspace(1) %914, i64 %909, !dbg !7
  %916 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %911, !dbg !7
  %917 = ptrtoint ptr addrspace(3) %916 to i64, !dbg !7
  %918 = trunc i64 %917 to i32, !dbg !7
  %919 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %918), !dbg !7
  %920 = zext i32 %919 to i64, !dbg !7
  %921 = inttoptr i64 %920 to ptr addrspace(3), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %915, ptr addrspace(3) %921, i32 4, i32 0, i32 0), !dbg !7
  %922 = add nuw nsw i64 %905, 256, !dbg !7
  br label %904, !dbg !7

923:                                              ; preds = %904
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.thread500 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local501 = zext nneg i32 %address.thread500 to i64, !dbg !6
  %address.local502 = shl nuw nsw i64 %address.local501, 3, !dbg !18
  %address.local503 = lshr i64 %address.local501, 4, !dbg !6
  %address.local504 = mul nsw i64 %address.local503, -128, !dbg !18
  %address.local505 = add nsw i64 %address.local502, %address.local504, !dbg !18
  %address.local506 = and i64 %address.local501, 15, !dbg !8
  %address.local507 = lshr i64 %address.local506, 3, !dbg !18
  %address.local508 = mul nuw nsw i64 %address.local507, 72, !dbg !18
  %address.local509 = add nsw i64 %address.local505, %address.local508, !dbg !18
  %address.local510 = lshr i64 %address.local506, 2, !dbg !8
  %address.local511 = shl nuw nsw i64 %address.local510, 5, !dbg !18
  %address.local512 = add nsw i64 %address.local509, %address.local511, !dbg !18
  %address.local513 = lshr i64 %address.local506, 3, !dbg !18
  %address.local514 = mul nuw nsw i64 %address.local513, -64, !dbg !18
  %address.local515 = add nsw i64 %address.local512, %address.local514, !dbg !18
  %address.local516 = add nuw nsw i64 %address.local506, %address.local507, !dbg !18
  %address.local517 = lshr i64 %address.local516, 2, !dbg !18
  %address.local518 = mul nsw i64 %address.local517, -32, !dbg !18
  %address.local519 = add i64 %address.local515, %address.local518, !dbg !18
  %address.local520 = lshr i64 %address.local501, 4, !dbg !6
  %address.local521 = and i64 %address.local520, 3, !dbg !6
  %address.local522 = shl nuw nsw i64 %address.local521, 9, !dbg !6
  %address.local523 = add i64 %address.local519, %address.local522, !dbg !18
  %address.local524 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local523, !dbg !18
  %924 = load <8 x half>, ptr addrspace(3) %address.local524, align 2, !dbg !18
  %925 = extractvalue [2 x <8 x float>] %57, 0, !dbg !30
  %926 = shufflevector <8 x float> %925, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %927 = shufflevector <8 x half> %924, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %928 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %927, <4 x float> %926) #9, !dbg !18
  %address.local525 = mul nuw nsw i64 %address.local521, 576, !dbg !18
  %address.local526 = add nuw nsw i64 %address.local502, %address.local525, !dbg !18
  %address.local527 = add nsw i64 %address.local526, %address.local504, !dbg !18
  %address.local528 = add nsw i64 %address.local527, %address.local508, !dbg !18
  %address.local529 = add nsw i64 %address.local528, %address.local511, !dbg !18
  %address.local530 = shl nuw nsw i32 %address.thread500, 2, !dbg !18
  %address.local531 = and i32 %address.local530, 192, !dbg !18
  %address.local532 = or disjoint i32 %address.local531, 32, !dbg !18
  %address.local533 = shl nuw nsw i32 %address.thread500, 3, !dbg !18
  %address.local534 = and i32 %address.local533, 96, !dbg !18
  %address.local535 = add nuw nsw i32 %address.local532, %address.local534, !dbg !18
  %address.local536 = and i32 %address.local535, 448, !dbg !18
  %address.local537 = zext nneg i32 %address.local536 to i64, !dbg !18
  %address.local538 = sub nsw i64 %address.local529, %address.local537, !dbg !18
  %address.local539 = add i64 %address.local538, %address.local518, !dbg !18
  %address.local540 = add i64 %address.local539, 288, !dbg !18
  %address.local541 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local540, !dbg !18
  %929 = load <8 x half>, ptr addrspace(3) %address.local541, align 2, !dbg !18
  %930 = shufflevector <8 x half> %924, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %931 = shufflevector <8 x float> %925, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %932 = extractvalue [2 x <8 x float>] %57, 1, !dbg !30
  %933 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %930, <4 x float> %928) #9, !dbg !18
  %934 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %935 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %936 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %937 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %934, <4 x float> %933) #9, !dbg !18
  %938 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %939 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %935, <4 x float> %937) #9, !dbg !18
  %address.local542 = shl nuw nsw i32 %address.thread500, 3, !dbg !18
  %address.local543 = and i32 %address.local542, 120, !dbg !18
  %address.local544 = shl nuw nsw i32 %address.thread500, 5, !dbg !18
  %address.local545 = and i32 %address.local544, 1536, !dbg !18
  %address.local546 = or disjoint i32 %address.local543, %address.local545, !dbg !18
  %address.local547 = zext nneg i32 %address.local546 to i64, !dbg !18
  %address.local548 = add nuw nsw i64 %address.local508, %address.local547, !dbg !18
  %address.local549 = add nuw nsw i64 %address.local548, %address.local511, !dbg !18
  %address.local550 = add nsw i64 %address.local549, %address.local514, !dbg !18
  %address.local551 = add nsw i64 %address.local550, %address.local518, !dbg !18
  %address.local552 = add i64 %address.local551, 2048, !dbg !18
  %address.local553 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local552, !dbg !18
  %940 = load <8 x half>, ptr addrspace(3) %address.local553, align 2, !dbg !18
  %address.local554 = add nuw nsw i64 %address.local526, 2304, !dbg !18
  %address.local555 = add nsw i64 %address.local554, %address.local504, !dbg !18
  %address.local556 = add nsw i64 %address.local555, %address.local508, !dbg !18
  %address.local557 = add nsw i64 %address.local556, %address.local511, !dbg !18
  %address.local558 = shl nuw nsw i32 %address.thread500, 2, !dbg !18
  %address.local559 = and i32 %address.local558, 192, !dbg !18
  %address.local560 = or disjoint i32 %address.local559, 288, !dbg !18
  %address.local561 = shl nuw nsw i32 %address.thread500, 3, !dbg !18
  %address.local562 = and i32 %address.local561, 96, !dbg !18
  %address.local563 = add nuw nsw i32 %address.local560, %address.local562, !dbg !18
  %address.local564 = and i32 %address.local563, 960, !dbg !18
  %address.local565 = zext nneg i32 %address.local564 to i64, !dbg !18
  %address.local566 = sub i64 %address.local557, %address.local565, !dbg !18
  %address.local567 = add i64 %address.local566, %address.local518, !dbg !18
  %address.local568 = add i64 %address.local567, 288, !dbg !18
  %address.local569 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !18
  %941 = load <8 x half>, ptr addrspace(3) %address.local569, align 2, !dbg !18
  %address.local570 = add nsw i64 %address.local509, %address.local511, !dbg !18
  %address.local571 = shl nuw nsw i64 %address.local501, 3, !dbg !18
  %address.local572 = and i64 %address.local571, 64, !dbg !18
  %address.local573 = sub nsw i64 %address.local570, %address.local572, !dbg !18
  %address.local574 = add nuw nsw i64 %address.local516, 18, !dbg !18
  %address.local575 = lshr i64 %address.local574, 2, !dbg !18
  %address.local576 = mul nsw i64 %address.local575, -32, !dbg !18
  %address.local577 = add i64 %address.local573, %address.local576, !dbg !18
  %address.local578 = add i64 %address.local577, %address.local522, !dbg !18
  %address.local579 = add i64 %address.local578, 272, !dbg !18
  %address.local580 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local579, !dbg !18
  %942 = load <8 x half>, ptr addrspace(3) %address.local580, align 2, !dbg !18
  %943 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %944 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %943, <4 x float> %939) #9, !dbg !18
  %945 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %946 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %947 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %948 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %949 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local581 = add nsw i64 %address.local528, %address.local511, !dbg !18
  %address.local582 = shl nuw nsw i32 %address.thread500, 2, !dbg !18
  %address.local583 = and i32 %address.local582, 192, !dbg !18
  %address.local584 = add nuw nsw i32 %address.local583, 160, !dbg !18
  %address.local585 = shl nuw nsw i32 %address.thread500, 3, !dbg !18
  %address.local586 = and i32 %address.local585, 96, !dbg !18
  %address.local587 = add nuw nsw i32 %address.local584, %address.local586, !dbg !18
  %address.local588 = and i32 %address.local587, 960, !dbg !18
  %address.local589 = zext nneg i32 %address.local588 to i64, !dbg !18
  %address.local590 = sub nsw i64 %address.local581, %address.local589, !dbg !18
  %address.local591 = add i64 %address.local590, %address.local576, !dbg !18
  %address.local592 = add i64 %address.local591, 688, !dbg !18
  %address.local593 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local592, !dbg !18
  %950 = load <8 x half>, ptr addrspace(3) %address.local593, align 2, !dbg !18
  %951 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %945, <4 x float> %944) #9, !dbg !18
  %address.local594 = add nsw i64 %address.local526, %address.local504, !dbg !18
  %address.local595 = add nsw i64 %address.local594, %address.local508, !dbg !18
  %address.local596 = add nsw i64 %address.local595, %address.local511, !dbg !18
  %address.local597 = lshr i64 %address.local506, 3, !dbg !18
  %address.local598 = add nuw nsw i64 %address.local597, %address.local521, !dbg !18
  %address.local599 = shl nuw nsw i64 %address.local598, 6, !dbg !18
  %address.local600 = sub nsw i64 %address.local596, %address.local599, !dbg !18
  %address.local601 = add i64 %address.local600, %address.local576, !dbg !18
  %address.local602 = add i64 %address.local601, 2320, !dbg !18
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local602, !dbg !18
  %952 = load <8 x half>, ptr addrspace(3) %address.local603, align 2, !dbg !18
  %address.local604 = add nsw i64 %address.local556, %address.local511, !dbg !18
  %address.local605 = shl nuw nsw i32 %address.thread500, 2, !dbg !18
  %address.local606 = and i32 %address.local605, 192, !dbg !18
  %address.local607 = add nuw nsw i32 %address.local606, 416, !dbg !18
  %address.local608 = shl nuw nsw i32 %address.thread500, 3, !dbg !18
  %address.local609 = and i32 %address.local608, 96, !dbg !18
  %address.local610 = add nuw nsw i32 %address.local607, %address.local609, !dbg !18
  %address.local611 = and i32 %address.local610, 1984, !dbg !18
  %address.local612 = zext nneg i32 %address.local611 to i64, !dbg !18
  %address.local613 = sub i64 %address.local604, %address.local612, !dbg !18
  %address.local614 = add i64 %address.local613, %address.local576, !dbg !18
  %address.local615 = add i64 %address.local614, 688, !dbg !18
  %address.local616 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local615, !dbg !18
  %953 = load <8 x half>, ptr addrspace(3) %address.local616, align 2, !dbg !18
  %954 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %955 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %946, <4 x float> %951) #9, !dbg !18
  %956 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %957 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %958 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %959 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %960 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %961 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %947, <4 x float> %955) #9, !dbg !18
  %962 = load <8 x half>, ptr addrspace(3) %address.local524, align 2, !dbg !18
  %963 = load <8 x half>, ptr addrspace(3) %address.local541, align 2, !dbg !18
  %964 = load <8 x half>, ptr addrspace(3) %address.local553, align 2, !dbg !18
  %965 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %948, <4 x float> %931) #9, !dbg !18
  %966 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %967 = shufflevector <8 x half> %962, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %968 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %969 = shufflevector <8 x half> %963, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %970 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %971 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %972 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %949, <4 x float> %965) #9, !dbg !18
  %973 = load <8 x half>, ptr addrspace(3) %address.local569, align 2, !dbg !18
  %974 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %954, <4 x float> %972) #9, !dbg !18
  %975 = load <8 x half>, ptr addrspace(3) %address.local580, align 2, !dbg !18
  %976 = load <8 x half>, ptr addrspace(3) %address.local593, align 2, !dbg !18
  %977 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %978 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %979 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %956, <4 x float> %974) #9, !dbg !18
  %980 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %981 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %982 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %983 = shufflevector <8 x half> %976, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %984 = load <8 x half>, ptr addrspace(3) %address.local603, align 2, !dbg !18
  %985 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %957, <4 x float> %979) #9, !dbg !18
  %986 = load <8 x half>, ptr addrspace(3) %address.local616, align 2, !dbg !18
  %987 = shufflevector <8 x half> %984, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %988 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %958, <4 x float> %985) #9, !dbg !18
  %989 = shufflevector <8 x half> %984, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %990 = shufflevector <8 x half> %986, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %991 = shufflevector <8 x half> %986, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %992 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %959, <4 x float> %988) #9, !dbg !18
  %993 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %960, <4 x float> %992) #9, !dbg !18
  %994 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %966, <4 x float> %936) #9, !dbg !18
  %995 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %967, <4 x float> %994) #9, !dbg !18
  %996 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %968, <4 x float> %995) #9, !dbg !18
  %997 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %969, <4 x float> %996) #9, !dbg !18
  %998 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %970, <4 x float> %997) #9, !dbg !18
  %999 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %971, <4 x float> %998) #9, !dbg !18
  %1000 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %977, <4 x float> %999) #9, !dbg !18
  %1001 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %978, <4 x float> %1000) #9, !dbg !18
  %1002 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %980, <4 x float> %938) #9, !dbg !18
  %1003 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %981, <4 x float> %1002) #9, !dbg !18
  %1004 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %982, <4 x float> %1003) #9, !dbg !18
  %1005 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %983, <4 x float> %1004) #9, !dbg !18
  %1006 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %987, <4 x float> %1005) #9, !dbg !18
  %1007 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %989, <4 x float> %1006) #9, !dbg !18
  %1008 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %990, <4 x float> %1007) #9, !dbg !18
  %1009 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %991, <4 x float> %1008) #9, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1010, !dbg !7

1010:                                             ; preds = %1013, %923
  %1011 = phi i64 [ %1061, %1013 ], [ %20, %923 ], !dbg !7
  %1012 = icmp samesign ult i64 %1011, 2048, !dbg !7
  br i1 %1012, label %1013, label %1062, !dbg !7

1013:                                             ; preds = %1010
  %1014 = shl nuw nsw i64 %1011, 1, !dbg !7
  %1015 = lshr i64 %1011, 2, !dbg !7
  %1016 = lshr i64 %1011, 5, !dbg !7
  %1017 = and i64 %1015, 24, !dbg !7
  %1018 = lshr i64 %1011, 4, !dbg !7
  %1019 = lshr i64 %1011, 7, !dbg !7
  %1020 = sub nsw i64 %1018, %1019, !dbg !7
  %1021 = shl nsw i64 %1020, 2, !dbg !7
  %1022 = and i64 %1021, 4, !dbg !7
  %1023 = sub nsw i64 %1015, %1016, !dbg !7
  %1024 = and i64 %1023, 3, !dbg !7
  %1025 = shl nuw nsw i64 %1019, 3, !dbg !7
  %1026 = shl nuw nsw i64 %1011, 1, !dbg !7
  %1027 = and i64 %1026, 4, !dbg !7
  %1028 = or disjoint i64 %1025, %1027, !dbg !7
  %1029 = and i64 %1014, 2, !dbg !7
  %1030 = or disjoint i64 %1028, %1029, !dbg !7
  %1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1017, !dbg !7
  %1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1031, i64 %1022, !dbg !7
  %1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1032, i64 %1024, !dbg !7
  %.idx103 = shl nuw nsw i64 %1030, 6, !dbg !7
  %1034 = getelementptr inbounds nuw i8, ptr addrspace(3) %1033, i64 %.idx103, !dbg !7
  %1035 = load half, ptr addrspace(3) %1034, align 2, !dbg !7
  %1036 = lshr i64 %1011, 2, !dbg !7
  %1037 = lshr i64 %1011, 5, !dbg !7
  %1038 = and i64 %1036, 24, !dbg !7
  %1039 = lshr i64 %1011, 4, !dbg !7
  %1040 = lshr i64 %1011, 7, !dbg !7
  %1041 = sub nsw i64 %1039, %1040, !dbg !7
  %1042 = shl nsw i64 %1041, 2, !dbg !7
  %1043 = and i64 %1042, 4, !dbg !7
  %1044 = sub nsw i64 %1036, %1037, !dbg !7
  %1045 = and i64 %1044, 3, !dbg !7
  %1046 = shl nuw nsw i64 %1040, 3, !dbg !7
  %1047 = shl nuw nsw i64 %1011, 1, !dbg !7
  %1048 = and i64 %1047, 4, !dbg !7
  %1049 = or disjoint i64 %1046, %1048, !dbg !7
  %1050 = and i64 %1014, 2, !dbg !7
  %1051 = or disjoint i64 %1050, 1, !dbg !7
  %1052 = or disjoint i64 %1049, %1051, !dbg !7
  %1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1038, !dbg !7
  %1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1053, i64 %1043, !dbg !7
  %1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1054, i64 %1045, !dbg !7
  %.idx104 = shl nuw nsw i64 %1052, 6, !dbg !7
  %1056 = getelementptr inbounds nuw i8, ptr addrspace(3) %1055, i64 %.idx104, !dbg !7
  %1057 = load half, ptr addrspace(3) %1056, align 2, !dbg !7
  %1058 = insertelement <2 x half> poison, half %1035, i64 0, !dbg !7
  %1059 = insertelement <2 x half> %1058, half %1057, i64 1, !dbg !7
  %1060 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %1014, !dbg !7
  store <2 x half> %1059, ptr addrspace(3) %1060, align 2, !dbg !7
  %1061 = add nuw nsw i64 %1011, 256, !dbg !7
  br label %1010, !dbg !7

1062:                                             ; preds = %1010
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1063 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !7
  %1064 = getelementptr i8, ptr addrspace(1) %1063, i64 128, !dbg !7
  br label %1065, !dbg !7

1065:                                             ; preds = %1068, %1062
  %1066 = phi i64 [ %1083, %1068 ], [ %20, %1062 ], !dbg !7
  %1067 = icmp samesign ult i64 %1066, 2048, !dbg !7
  br i1 %1067, label %1068, label %1084, !dbg !7

1068:                                             ; preds = %1065
  %1069 = shl nuw nsw i64 %1066, 1, !dbg !7
  %1070 = and i64 %1069, 30, !dbg !7
  %1071 = shl nuw nsw i64 %1066, 1, !dbg !7
  %1072 = and i64 %1071, 3968, !dbg !7
  %1073 = shl nuw nsw i64 %1066, 3, !dbg !7
  %1074 = and i64 %1073, 16256, !dbg !7
  %1075 = getelementptr [2 x i8], ptr addrspace(1) %1064, i64 %1074, !dbg !7
  %1076 = getelementptr [2 x i8], ptr addrspace(1) %1075, i64 %1070, !dbg !7
  %1077 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1072, !dbg !7
  %1078 = ptrtoint ptr addrspace(3) %1077 to i64, !dbg !7
  %1079 = trunc i64 %1078 to i32, !dbg !7
  %1080 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1079), !dbg !7
  %1081 = zext i32 %1080 to i64, !dbg !7
  %1082 = inttoptr i64 %1081 to ptr addrspace(3), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %1076, ptr addrspace(3) %1082, i32 4, i32 0, i32 0), !dbg !7
  %1083 = add nuw nsw i64 %1066, 256, !dbg !7
  br label %1065, !dbg !7

1084:                                             ; preds = %1065
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.thread617 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local618 = zext nneg i32 %address.thread617 to i64, !dbg !6
  %address.local619 = shl nuw nsw i64 %address.local618, 3, !dbg !18
  %address.local620 = lshr i64 %address.local618, 4, !dbg !6
  %address.local621 = mul nsw i64 %address.local620, -128, !dbg !18
  %address.local622 = add nsw i64 %address.local619, %address.local621, !dbg !18
  %address.local623 = and i64 %address.local618, 15, !dbg !8
  %address.local624 = lshr i64 %address.local623, 3, !dbg !18
  %address.local625 = mul nuw nsw i64 %address.local624, 72, !dbg !18
  %address.local626 = add nsw i64 %address.local622, %address.local625, !dbg !18
  %address.local627 = lshr i64 %address.local623, 2, !dbg !8
  %address.local628 = shl nuw nsw i64 %address.local627, 5, !dbg !18
  %address.local629 = add nsw i64 %address.local626, %address.local628, !dbg !18
  %address.local630 = lshr i64 %address.local623, 3, !dbg !18
  %address.local631 = mul nuw nsw i64 %address.local630, -64, !dbg !18
  %address.local632 = add nsw i64 %address.local629, %address.local631, !dbg !18
  %address.local633 = add nuw nsw i64 %address.local623, %address.local624, !dbg !18
  %address.local634 = lshr i64 %address.local633, 2, !dbg !18
  %address.local635 = mul nsw i64 %address.local634, -32, !dbg !18
  %address.local636 = add i64 %address.local632, %address.local635, !dbg !18
  %address.local637 = lshr i64 %address.local618, 4, !dbg !6
  %address.local638 = and i64 %address.local637, 3, !dbg !6
  %address.local639 = shl nuw nsw i64 %address.local638, 9, !dbg !6
  %address.local640 = add i64 %address.local636, %address.local639, !dbg !18
  %address.local641 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local640, !dbg !18
  %1085 = load <8 x half>, ptr addrspace(3) %address.local641, align 2, !dbg !18
  %1086 = extractvalue [2 x <8 x float>] %59, 0, !dbg !30
  %1087 = shufflevector <8 x float> %1086, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1088 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1089 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1088, <4 x float> %1087) #9, !dbg !18
  %address.local642 = mul nuw nsw i64 %address.local638, 576, !dbg !18
  %address.local643 = add nuw nsw i64 %address.local619, %address.local642, !dbg !18
  %address.local644 = add nsw i64 %address.local643, %address.local621, !dbg !18
  %address.local645 = add nsw i64 %address.local644, %address.local625, !dbg !18
  %address.local646 = add nsw i64 %address.local645, %address.local628, !dbg !18
  %address.local647 = shl nuw nsw i32 %address.thread617, 2, !dbg !18
  %address.local648 = and i32 %address.local647, 192, !dbg !18
  %address.local649 = or disjoint i32 %address.local648, 32, !dbg !18
  %address.local650 = shl nuw nsw i32 %address.thread617, 3, !dbg !18
  %address.local651 = and i32 %address.local650, 96, !dbg !18
  %address.local652 = add nuw nsw i32 %address.local649, %address.local651, !dbg !18
  %address.local653 = and i32 %address.local652, 448, !dbg !18
  %address.local654 = zext nneg i32 %address.local653 to i64, !dbg !18
  %address.local655 = sub nsw i64 %address.local646, %address.local654, !dbg !18
  %address.local656 = add i64 %address.local655, %address.local635, !dbg !18
  %address.local657 = add i64 %address.local656, 288, !dbg !18
  %address.local658 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local657, !dbg !18
  %1090 = load <8 x half>, ptr addrspace(3) %address.local658, align 2, !dbg !18
  %1091 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1092 = shufflevector <8 x float> %1086, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1093 = extractvalue [2 x <8 x float>] %59, 1, !dbg !30
  %1094 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1091, <4 x float> %1089) #9, !dbg !18
  %1095 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1096 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1097 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1098 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1095, <4 x float> %1094) #9, !dbg !18
  %1099 = shufflevector <8 x float> %1093, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1100 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1096, <4 x float> %1098) #9, !dbg !18
  %address.local659 = shl nuw nsw i32 %address.thread617, 3, !dbg !18
  %address.local660 = and i32 %address.local659, 120, !dbg !18
  %address.local661 = shl nuw nsw i32 %address.thread617, 5, !dbg !18
  %address.local662 = and i32 %address.local661, 1536, !dbg !18
  %address.local663 = or disjoint i32 %address.local660, %address.local662, !dbg !18
  %address.local664 = zext nneg i32 %address.local663 to i64, !dbg !18
  %address.local665 = add nuw nsw i64 %address.local625, %address.local664, !dbg !18
  %address.local666 = add nuw nsw i64 %address.local665, %address.local628, !dbg !18
  %address.local667 = add nsw i64 %address.local666, %address.local631, !dbg !18
  %address.local668 = add nsw i64 %address.local667, %address.local635, !dbg !18
  %address.local669 = add i64 %address.local668, 2048, !dbg !18
  %address.local670 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local669, !dbg !18
  %1101 = load <8 x half>, ptr addrspace(3) %address.local670, align 2, !dbg !18
  %address.local671 = add nuw nsw i64 %address.local643, 2304, !dbg !18
  %address.local672 = add nsw i64 %address.local671, %address.local621, !dbg !18
  %address.local673 = add nsw i64 %address.local672, %address.local625, !dbg !18
  %address.local674 = add nsw i64 %address.local673, %address.local628, !dbg !18
  %address.local675 = shl nuw nsw i32 %address.thread617, 2, !dbg !18
  %address.local676 = and i32 %address.local675, 192, !dbg !18
  %address.local677 = or disjoint i32 %address.local676, 288, !dbg !18
  %address.local678 = shl nuw nsw i32 %address.thread617, 3, !dbg !18
  %address.local679 = and i32 %address.local678, 96, !dbg !18
  %address.local680 = add nuw nsw i32 %address.local677, %address.local679, !dbg !18
  %address.local681 = and i32 %address.local680, 960, !dbg !18
  %address.local682 = zext nneg i32 %address.local681 to i64, !dbg !18
  %address.local683 = sub i64 %address.local674, %address.local682, !dbg !18
  %address.local684 = add i64 %address.local683, %address.local635, !dbg !18
  %address.local685 = add i64 %address.local684, 288, !dbg !18
  %address.local686 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local685, !dbg !18
  %1102 = load <8 x half>, ptr addrspace(3) %address.local686, align 2, !dbg !18
  %address.local687 = add nsw i64 %address.local626, %address.local628, !dbg !18
  %address.local688 = shl nuw nsw i64 %address.local618, 3, !dbg !18
  %address.local689 = and i64 %address.local688, 64, !dbg !18
  %address.local690 = sub nsw i64 %address.local687, %address.local689, !dbg !18
  %address.local691 = add nuw nsw i64 %address.local633, 18, !dbg !18
  %address.local692 = lshr i64 %address.local691, 2, !dbg !18
  %address.local693 = mul nsw i64 %address.local692, -32, !dbg !18
  %address.local694 = add i64 %address.local690, %address.local693, !dbg !18
  %address.local695 = add i64 %address.local694, %address.local639, !dbg !18
  %address.local696 = add i64 %address.local695, 272, !dbg !18
  %address.local697 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local696, !dbg !18
  %1103 = load <8 x half>, ptr addrspace(3) %address.local697, align 2, !dbg !18
  %1104 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1105 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1104, <4 x float> %1100) #9, !dbg !18
  %1106 = shufflevector <8 x half> %1101, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1107 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1108 = shufflevector <8 x half> %1102, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1109 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1110 = shufflevector <8 x half> %1103, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local698 = add nsw i64 %address.local645, %address.local628, !dbg !18
  %address.local699 = shl nuw nsw i32 %address.thread617, 2, !dbg !18
  %address.local700 = and i32 %address.local699, 192, !dbg !18
  %address.local701 = add nuw nsw i32 %address.local700, 160, !dbg !18
  %address.local702 = shl nuw nsw i32 %address.thread617, 3, !dbg !18
  %address.local703 = and i32 %address.local702, 96, !dbg !18
  %address.local704 = add nuw nsw i32 %address.local701, %address.local703, !dbg !18
  %address.local705 = and i32 %address.local704, 960, !dbg !18
  %address.local706 = zext nneg i32 %address.local705 to i64, !dbg !18
  %address.local707 = sub nsw i64 %address.local698, %address.local706, !dbg !18
  %address.local708 = add i64 %address.local707, %address.local693, !dbg !18
  %address.local709 = add i64 %address.local708, 688, !dbg !18
  %address.local710 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local709, !dbg !18
  %1111 = load <8 x half>, ptr addrspace(3) %address.local710, align 2, !dbg !18
  %1112 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1106, <4 x float> %1105) #9, !dbg !18
  %address.local711 = add nsw i64 %address.local643, %address.local621, !dbg !18
  %address.local712 = add nsw i64 %address.local711, %address.local625, !dbg !18
  %address.local713 = add nsw i64 %address.local712, %address.local628, !dbg !18
  %address.local714 = lshr i64 %address.local623, 3, !dbg !18
  %address.local715 = add nuw nsw i64 %address.local714, %address.local638, !dbg !18
  %address.local716 = shl nuw nsw i64 %address.local715, 6, !dbg !18
  %address.local717 = sub nsw i64 %address.local713, %address.local716, !dbg !18
  %address.local718 = add i64 %address.local717, %address.local693, !dbg !18
  %address.local719 = add i64 %address.local718, 2320, !dbg !18
  %address.local720 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local719, !dbg !18
  %1113 = load <8 x half>, ptr addrspace(3) %address.local720, align 2, !dbg !18
  %address.local721 = add nsw i64 %address.local673, %address.local628, !dbg !18
  %address.local722 = shl nuw nsw i32 %address.thread617, 2, !dbg !18
  %address.local723 = and i32 %address.local722, 192, !dbg !18
  %address.local724 = add nuw nsw i32 %address.local723, 416, !dbg !18
  %address.local725 = shl nuw nsw i32 %address.thread617, 3, !dbg !18
  %address.local726 = and i32 %address.local725, 96, !dbg !18
  %address.local727 = add nuw nsw i32 %address.local724, %address.local726, !dbg !18
  %address.local728 = and i32 %address.local727, 1984, !dbg !18
  %address.local729 = zext nneg i32 %address.local728 to i64, !dbg !18
  %address.local730 = sub i64 %address.local721, %address.local729, !dbg !18
  %address.local731 = add i64 %address.local730, %address.local693, !dbg !18
  %address.local732 = add i64 %address.local731, 688, !dbg !18
  %address.local733 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local732, !dbg !18
  %1114 = load <8 x half>, ptr addrspace(3) %address.local733, align 2, !dbg !18
  %1115 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1116 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1107, <4 x float> %1112) #9, !dbg !18
  %1117 = shufflevector <8 x half> %1111, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1118 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1119 = shufflevector <8 x half> %1113, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1120 = shufflevector <8 x half> %1114, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1121 = shufflevector <8 x half> %1114, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1122 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1108, <4 x float> %1116) #9, !dbg !18
  %1123 = load <8 x half>, ptr addrspace(3) %address.local641, align 2, !dbg !18
  %1124 = load <8 x half>, ptr addrspace(3) %address.local658, align 2, !dbg !18
  %1125 = load <8 x half>, ptr addrspace(3) %address.local670, align 2, !dbg !18
  %1126 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1109, <4 x float> %1092) #9, !dbg !18
  %1127 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1128 = shufflevector <8 x half> %1123, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1129 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1130 = shufflevector <8 x half> %1124, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1131 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1132 = shufflevector <8 x half> %1125, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1133 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1110, <4 x float> %1126) #9, !dbg !18
  %1134 = load <8 x half>, ptr addrspace(3) %address.local686, align 2, !dbg !18
  %1135 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1115, <4 x float> %1133) #9, !dbg !18
  %1136 = load <8 x half>, ptr addrspace(3) %address.local697, align 2, !dbg !18
  %1137 = load <8 x half>, ptr addrspace(3) %address.local710, align 2, !dbg !18
  %1138 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1139 = shufflevector <8 x half> %1134, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1140 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1117, <4 x float> %1135) #9, !dbg !18
  %1141 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1142 = shufflevector <8 x half> %1136, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1143 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1144 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1145 = load <8 x half>, ptr addrspace(3) %address.local720, align 2, !dbg !18
  %1146 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1118, <4 x float> %1140) #9, !dbg !18
  %1147 = load <8 x half>, ptr addrspace(3) %address.local733, align 2, !dbg !18
  %1148 = shufflevector <8 x half> %1145, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1149 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1119, <4 x float> %1146) #9, !dbg !18
  %1150 = shufflevector <8 x half> %1145, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1151 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1152 = shufflevector <8 x half> %1147, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1153 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1120, <4 x float> %1149) #9, !dbg !18
  %1154 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1121, <4 x float> %1153) #9, !dbg !18
  %1155 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1127, <4 x float> %1097) #9, !dbg !18
  %1156 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1128, <4 x float> %1155) #9, !dbg !18
  %1157 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1129, <4 x float> %1156) #9, !dbg !18
  %1158 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1130, <4 x float> %1157) #9, !dbg !18
  %1159 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1131, <4 x float> %1158) #9, !dbg !18
  %1160 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1132, <4 x float> %1159) #9, !dbg !18
  %1161 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1138, <4 x float> %1160) #9, !dbg !18
  %1162 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1139, <4 x float> %1161) #9, !dbg !18
  %1163 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1141, <4 x float> %1099) #9, !dbg !18
  %1164 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1142, <4 x float> %1163) #9, !dbg !18
  %1165 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1143, <4 x float> %1164) #9, !dbg !18
  %1166 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1144, <4 x float> %1165) #9, !dbg !18
  %1167 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1148, <4 x float> %1166) #9, !dbg !18
  %1168 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1150, <4 x float> %1167) #9, !dbg !18
  %1169 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1151, <4 x float> %1168) #9, !dbg !18
  %1170 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1152, <4 x float> %1169) #9, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1171, !dbg !7

1171:                                             ; preds = %1174, %1084
  %1172 = phi i64 [ %1222, %1174 ], [ %20, %1084 ], !dbg !7
  %1173 = icmp samesign ult i64 %1172, 2048, !dbg !7
  br i1 %1173, label %1174, label %1223, !dbg !7

1174:                                             ; preds = %1171
  %1175 = shl nuw nsw i64 %1172, 1, !dbg !7
  %1176 = lshr i64 %1172, 2, !dbg !7
  %1177 = lshr i64 %1172, 5, !dbg !7
  %1178 = and i64 %1176, 24, !dbg !7
  %1179 = lshr i64 %1172, 4, !dbg !7
  %1180 = lshr i64 %1172, 7, !dbg !7
  %1181 = sub nsw i64 %1179, %1180, !dbg !7
  %1182 = shl nsw i64 %1181, 2, !dbg !7
  %1183 = and i64 %1182, 4, !dbg !7
  %1184 = sub nsw i64 %1176, %1177, !dbg !7
  %1185 = and i64 %1184, 3, !dbg !7
  %1186 = shl nuw nsw i64 %1180, 3, !dbg !7
  %1187 = shl nuw nsw i64 %1172, 1, !dbg !7
  %1188 = and i64 %1187, 4, !dbg !7
  %1189 = or disjoint i64 %1186, %1188, !dbg !7
  %1190 = and i64 %1175, 2, !dbg !7
  %1191 = or disjoint i64 %1189, %1190, !dbg !7
  %1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1178, !dbg !7
  %1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1192, i64 %1183, !dbg !7
  %1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1193, i64 %1185, !dbg !7
  %.idx101 = shl nuw nsw i64 %1191, 6, !dbg !7
  %1195 = getelementptr inbounds nuw i8, ptr addrspace(3) %1194, i64 %.idx101, !dbg !7
  %1196 = load half, ptr addrspace(3) %1195, align 2, !dbg !7
  %1197 = lshr i64 %1172, 2, !dbg !7
  %1198 = lshr i64 %1172, 5, !dbg !7
  %1199 = and i64 %1197, 24, !dbg !7
  %1200 = lshr i64 %1172, 4, !dbg !7
  %1201 = lshr i64 %1172, 7, !dbg !7
  %1202 = sub nsw i64 %1200, %1201, !dbg !7
  %1203 = shl nsw i64 %1202, 2, !dbg !7
  %1204 = and i64 %1203, 4, !dbg !7
  %1205 = sub nsw i64 %1197, %1198, !dbg !7
  %1206 = and i64 %1205, 3, !dbg !7
  %1207 = shl nuw nsw i64 %1201, 3, !dbg !7
  %1208 = shl nuw nsw i64 %1172, 1, !dbg !7
  %1209 = and i64 %1208, 4, !dbg !7
  %1210 = or disjoint i64 %1207, %1209, !dbg !7
  %1211 = and i64 %1175, 2, !dbg !7
  %1212 = or disjoint i64 %1211, 1, !dbg !7
  %1213 = or disjoint i64 %1210, %1212, !dbg !7
  %1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1199, !dbg !7
  %1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1214, i64 %1204, !dbg !7
  %1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1215, i64 %1206, !dbg !7
  %.idx102 = shl nuw nsw i64 %1213, 6, !dbg !7
  %1217 = getelementptr inbounds nuw i8, ptr addrspace(3) %1216, i64 %.idx102, !dbg !7
  %1218 = load half, ptr addrspace(3) %1217, align 2, !dbg !7
  %1219 = insertelement <2 x half> poison, half %1196, i64 0, !dbg !7
  %1220 = insertelement <2 x half> %1219, half %1218, i64 1, !dbg !7
  %1221 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %1175, !dbg !7
  store <2 x half> %1220, ptr addrspace(3) %1221, align 2, !dbg !7
  %1222 = add nuw nsw i64 %1172, 256, !dbg !7
  br label %1171, !dbg !7

1223:                                             ; preds = %1171
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1224 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %901, !dbg !7
  %1225 = getelementptr i8, ptr addrspace(1) %1224, i64 192, !dbg !7
  br label %1226, !dbg !7

1226:                                             ; preds = %1229, %1223
  %1227 = phi i64 [ %1244, %1229 ], [ %20, %1223 ], !dbg !7
  %1228 = icmp samesign ult i64 %1227, 2048, !dbg !7
  br i1 %1228, label %1229, label %1245, !dbg !7

1229:                                             ; preds = %1226
  %1230 = shl nuw nsw i64 %1227, 1, !dbg !7
  %1231 = and i64 %1230, 30, !dbg !7
  %1232 = shl nuw nsw i64 %1227, 1, !dbg !7
  %1233 = and i64 %1232, 3968, !dbg !7
  %1234 = shl nuw nsw i64 %1227, 3, !dbg !7
  %1235 = and i64 %1234, 16256, !dbg !7
  %1236 = getelementptr [2 x i8], ptr addrspace(1) %1225, i64 %1235, !dbg !7
  %1237 = getelementptr [2 x i8], ptr addrspace(1) %1236, i64 %1231, !dbg !7
  %1238 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1233, !dbg !7
  %1239 = ptrtoint ptr addrspace(3) %1238 to i64, !dbg !7
  %1240 = trunc i64 %1239 to i32, !dbg !7
  %1241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1240), !dbg !7
  %1242 = zext i32 %1241 to i64, !dbg !7
  %1243 = inttoptr i64 %1242 to ptr addrspace(3), !dbg !7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %1237, ptr addrspace(3) %1243, i32 4, i32 0, i32 0), !dbg !7
  %1244 = add nuw nsw i64 %1227, 256, !dbg !7
  br label %1226, !dbg !7

1245:                                             ; preds = %1226
  call void @llvm.amdgcn.asyncmark(), !dbg !7
  %address.thread734 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local735 = zext nneg i32 %address.thread734 to i64, !dbg !6
  %address.local736 = shl nuw nsw i64 %address.local735, 3, !dbg !18
  %address.local737 = lshr i64 %address.local735, 4, !dbg !6
  %address.local738 = mul nsw i64 %address.local737, -128, !dbg !18
  %address.local739 = add nsw i64 %address.local736, %address.local738, !dbg !18
  %address.local740 = and i64 %address.local735, 15, !dbg !8
  %address.local741 = lshr i64 %address.local740, 3, !dbg !18
  %address.local742 = mul nuw nsw i64 %address.local741, 72, !dbg !18
  %address.local743 = add nsw i64 %address.local739, %address.local742, !dbg !18
  %address.local744 = lshr i64 %address.local740, 2, !dbg !8
  %address.local745 = shl nuw nsw i64 %address.local744, 5, !dbg !18
  %address.local746 = add nsw i64 %address.local743, %address.local745, !dbg !18
  %address.local747 = lshr i64 %address.local740, 3, !dbg !18
  %address.local748 = mul nuw nsw i64 %address.local747, -64, !dbg !18
  %address.local749 = add nsw i64 %address.local746, %address.local748, !dbg !18
  %address.local750 = add nuw nsw i64 %address.local740, %address.local741, !dbg !18
  %address.local751 = lshr i64 %address.local750, 2, !dbg !18
  %address.local752 = mul nsw i64 %address.local751, -32, !dbg !18
  %address.local753 = add i64 %address.local749, %address.local752, !dbg !18
  %address.local754 = lshr i64 %address.local735, 4, !dbg !6
  %address.local755 = and i64 %address.local754, 3, !dbg !6
  %address.local756 = shl nuw nsw i64 %address.local755, 9, !dbg !6
  %address.local757 = add i64 %address.local753, %address.local756, !dbg !18
  %address.local758 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local757, !dbg !18
  %1246 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !18
  %1247 = extractvalue [2 x <8 x float>] %60, 0, !dbg !30
  %1248 = shufflevector <8 x float> %1247, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1249 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1250 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1249, <4 x float> %1248) #9, !dbg !18
  %address.local759 = mul nuw nsw i64 %address.local755, 576, !dbg !18
  %address.local760 = add nuw nsw i64 %address.local736, %address.local759, !dbg !18
  %address.local761 = add nsw i64 %address.local760, %address.local738, !dbg !18
  %address.local762 = add nsw i64 %address.local761, %address.local742, !dbg !18
  %address.local763 = add nsw i64 %address.local762, %address.local745, !dbg !18
  %address.local764 = shl nuw nsw i32 %address.thread734, 2, !dbg !18
  %address.local765 = and i32 %address.local764, 192, !dbg !18
  %address.local766 = or disjoint i32 %address.local765, 32, !dbg !18
  %address.local767 = shl nuw nsw i32 %address.thread734, 3, !dbg !18
  %address.local768 = and i32 %address.local767, 96, !dbg !18
  %address.local769 = add nuw nsw i32 %address.local766, %address.local768, !dbg !18
  %address.local770 = and i32 %address.local769, 448, !dbg !18
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !18
  %address.local772 = sub nsw i64 %address.local763, %address.local771, !dbg !18
  %address.local773 = add i64 %address.local772, %address.local752, !dbg !18
  %address.local774 = add i64 %address.local773, 288, !dbg !18
  %address.local775 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local774, !dbg !18
  %1251 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !18
  %1252 = shufflevector <8 x half> %1246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1253 = shufflevector <8 x float> %1247, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1254 = extractvalue [2 x <8 x float>] %60, 1, !dbg !30
  %1255 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1252, <4 x float> %1250) #9, !dbg !18
  %1256 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1257 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1258 = shufflevector <8 x float> %1254, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1259 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1256, <4 x float> %1255) #9, !dbg !18
  %1260 = shufflevector <8 x float> %1254, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1261 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1257, <4 x float> %1259) #9, !dbg !18
  %address.local776 = shl nuw nsw i32 %address.thread734, 3, !dbg !18
  %address.local777 = and i32 %address.local776, 120, !dbg !18
  %address.local778 = shl nuw nsw i32 %address.thread734, 5, !dbg !18
  %address.local779 = and i32 %address.local778, 1536, !dbg !18
  %address.local780 = or disjoint i32 %address.local777, %address.local779, !dbg !18
  %address.local781 = zext nneg i32 %address.local780 to i64, !dbg !18
  %address.local782 = add nuw nsw i64 %address.local742, %address.local781, !dbg !18
  %address.local783 = add nuw nsw i64 %address.local782, %address.local745, !dbg !18
  %address.local784 = add nsw i64 %address.local783, %address.local748, !dbg !18
  %address.local785 = add nsw i64 %address.local784, %address.local752, !dbg !18
  %address.local786 = add i64 %address.local785, 2048, !dbg !18
  %address.local787 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local786, !dbg !18
  %1262 = load <8 x half>, ptr addrspace(3) %address.local787, align 2, !dbg !18
  %address.local788 = add nuw nsw i64 %address.local760, 2304, !dbg !18
  %address.local789 = add nsw i64 %address.local788, %address.local738, !dbg !18
  %address.local790 = add nsw i64 %address.local789, %address.local742, !dbg !18
  %address.local791 = add nsw i64 %address.local790, %address.local745, !dbg !18
  %address.local792 = shl nuw nsw i32 %address.thread734, 2, !dbg !18
  %address.local793 = and i32 %address.local792, 192, !dbg !18
  %address.local794 = or disjoint i32 %address.local793, 288, !dbg !18
  %address.local795 = shl nuw nsw i32 %address.thread734, 3, !dbg !18
  %address.local796 = and i32 %address.local795, 96, !dbg !18
  %address.local797 = add nuw nsw i32 %address.local794, %address.local796, !dbg !18
  %address.local798 = and i32 %address.local797, 960, !dbg !18
  %address.local799 = zext nneg i32 %address.local798 to i64, !dbg !18
  %address.local800 = sub i64 %address.local791, %address.local799, !dbg !18
  %address.local801 = add i64 %address.local800, %address.local752, !dbg !18
  %address.local802 = add i64 %address.local801, 288, !dbg !18
  %address.local803 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local802, !dbg !18
  %1263 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %address.local804 = add nsw i64 %address.local743, %address.local745, !dbg !18
  %address.local805 = shl nuw nsw i64 %address.local735, 3, !dbg !18
  %address.local806 = and i64 %address.local805, 64, !dbg !18
  %address.local807 = sub nsw i64 %address.local804, %address.local806, !dbg !18
  %address.local808 = add nuw nsw i64 %address.local750, 18, !dbg !18
  %address.local809 = lshr i64 %address.local808, 2, !dbg !18
  %address.local810 = mul nsw i64 %address.local809, -32, !dbg !18
  %address.local811 = add i64 %address.local807, %address.local810, !dbg !18
  %address.local812 = add i64 %address.local811, %address.local756, !dbg !18
  %address.local813 = add i64 %address.local812, 272, !dbg !18
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local813, !dbg !18
  %1264 = load <8 x half>, ptr addrspace(3) %address.local814, align 2, !dbg !18
  %1265 = shufflevector <8 x half> %1262, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1266 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1265, <4 x float> %1261) #9, !dbg !18
  %1267 = shufflevector <8 x half> %1262, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1268 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1269 = shufflevector <8 x half> %1263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1270 = shufflevector <8 x half> %1264, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1271 = shufflevector <8 x half> %1264, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local815 = add nsw i64 %address.local762, %address.local745, !dbg !18
  %address.local816 = shl nuw nsw i32 %address.thread734, 2, !dbg !18
  %address.local817 = and i32 %address.local816, 192, !dbg !18
  %address.local818 = add nuw nsw i32 %address.local817, 160, !dbg !18
  %address.local819 = shl nuw nsw i32 %address.thread734, 3, !dbg !18
  %address.local820 = and i32 %address.local819, 96, !dbg !18
  %address.local821 = add nuw nsw i32 %address.local818, %address.local820, !dbg !18
  %address.local822 = and i32 %address.local821, 960, !dbg !18
  %address.local823 = zext nneg i32 %address.local822 to i64, !dbg !18
  %address.local824 = sub nsw i64 %address.local815, %address.local823, !dbg !18
  %address.local825 = add i64 %address.local824, %address.local810, !dbg !18
  %address.local826 = add i64 %address.local825, 688, !dbg !18
  %address.local827 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local826, !dbg !18
  %1272 = load <8 x half>, ptr addrspace(3) %address.local827, align 2, !dbg !18
  %1273 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1267, <4 x float> %1266) #9, !dbg !18
  %address.local828 = add nsw i64 %address.local760, %address.local738, !dbg !18
  %address.local829 = add nsw i64 %address.local828, %address.local742, !dbg !18
  %address.local830 = add nsw i64 %address.local829, %address.local745, !dbg !18
  %address.local831 = lshr i64 %address.local740, 3, !dbg !18
  %address.local832 = add nuw nsw i64 %address.local831, %address.local755, !dbg !18
  %address.local833 = shl nuw nsw i64 %address.local832, 6, !dbg !18
  %address.local834 = sub nsw i64 %address.local830, %address.local833, !dbg !18
  %address.local835 = add i64 %address.local834, %address.local810, !dbg !18
  %address.local836 = add i64 %address.local835, 2320, !dbg !18
  %address.local837 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local836, !dbg !18
  %1274 = load <8 x half>, ptr addrspace(3) %address.local837, align 2, !dbg !18
  %address.local838 = add nsw i64 %address.local790, %address.local745, !dbg !18
  %address.local839 = shl nuw nsw i32 %address.thread734, 2, !dbg !18
  %address.local840 = and i32 %address.local839, 192, !dbg !18
  %address.local841 = add nuw nsw i32 %address.local840, 416, !dbg !18
  %address.local842 = shl nuw nsw i32 %address.thread734, 3, !dbg !18
  %address.local843 = and i32 %address.local842, 96, !dbg !18
  %address.local844 = add nuw nsw i32 %address.local841, %address.local843, !dbg !18
  %address.local845 = and i32 %address.local844, 1984, !dbg !18
  %address.local846 = zext nneg i32 %address.local845 to i64, !dbg !18
  %address.local847 = sub i64 %address.local838, %address.local846, !dbg !18
  %address.local848 = add i64 %address.local847, %address.local810, !dbg !18
  %address.local849 = add i64 %address.local848, 688, !dbg !18
  %address.local850 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local849, !dbg !18
  %1275 = load <8 x half>, ptr addrspace(3) %address.local850, align 2, !dbg !18
  %1276 = shufflevector <8 x half> %1272, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1277 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1268, <4 x float> %1273) #9, !dbg !18
  %1278 = shufflevector <8 x half> %1272, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1279 = shufflevector <8 x half> %1274, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1280 = shufflevector <8 x half> %1274, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1281 = shufflevector <8 x half> %1275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1282 = shufflevector <8 x half> %1275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1283 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1269, <4 x float> %1277) #9, !dbg !18
  %1284 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !18
  %1285 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !18
  %1286 = load <8 x half>, ptr addrspace(3) %address.local787, align 2, !dbg !18
  %1287 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1270, <4 x float> %1253) #9, !dbg !18
  %1288 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1289 = shufflevector <8 x half> %1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1290 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1291 = shufflevector <8 x half> %1285, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1292 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1293 = shufflevector <8 x half> %1286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1294 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1271, <4 x float> %1287) #9, !dbg !18
  %1295 = load <8 x half>, ptr addrspace(3) %address.local803, align 2, !dbg !18
  %1296 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1276, <4 x float> %1294) #9, !dbg !18
  %1297 = load <8 x half>, ptr addrspace(3) %address.local814, align 2, !dbg !18
  %1298 = load <8 x half>, ptr addrspace(3) %address.local827, align 2, !dbg !18
  %1299 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1300 = shufflevector <8 x half> %1295, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1301 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1278, <4 x float> %1296) #9, !dbg !18
  %1302 = shufflevector <8 x half> %1297, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1303 = shufflevector <8 x half> %1297, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1304 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1305 = shufflevector <8 x half> %1298, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1306 = load <8 x half>, ptr addrspace(3) %address.local837, align 2, !dbg !18
  %1307 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1279, <4 x float> %1301) #9, !dbg !18
  %1308 = load <8 x half>, ptr addrspace(3) %address.local850, align 2, !dbg !18
  %1309 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1310 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1280, <4 x float> %1307) #9, !dbg !18
  %1311 = shufflevector <8 x half> %1306, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1312 = shufflevector <8 x half> %1308, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1313 = shufflevector <8 x half> %1308, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1314 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1281, <4 x float> %1310) #9, !dbg !18
  %1315 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1282, <4 x float> %1314) #9, !dbg !18
  %1316 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1288, <4 x float> %1258) #9, !dbg !18
  %1317 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1289, <4 x float> %1316) #9, !dbg !18
  %1318 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1290, <4 x float> %1317) #9, !dbg !18
  %1319 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1291, <4 x float> %1318) #9, !dbg !18
  %1320 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1292, <4 x float> %1319) #9, !dbg !18
  %1321 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1293, <4 x float> %1320) #9, !dbg !18
  %1322 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1299, <4 x float> %1321) #9, !dbg !18
  %1323 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1300, <4 x float> %1322) #9, !dbg !18
  %1324 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1302, <4 x float> %1260) #9, !dbg !18
  %1325 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1303, <4 x float> %1324) #9, !dbg !18
  %1326 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1304, <4 x float> %1325) #9, !dbg !18
  %1327 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1305, <4 x float> %1326) #9, !dbg !18
  %1328 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1309, <4 x float> %1327) #9, !dbg !18
  %1329 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1311, <4 x float> %1328) #9, !dbg !18
  %1330 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1312, <4 x float> %1329) #9, !dbg !18
  %1331 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1313, <4 x float> %1330) #9, !dbg !18
  call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  br label %1332, !dbg !7

1332:                                             ; preds = %1335, %1245
  %1333 = phi i64 [ %1383, %1335 ], [ %20, %1245 ], !dbg !7
  %1334 = icmp samesign ult i64 %1333, 2048, !dbg !7
  br i1 %1334, label %1335, label %1384, !dbg !7

1335:                                             ; preds = %1332
  %1336 = shl nuw nsw i64 %1333, 1, !dbg !7
  %1337 = lshr i64 %1333, 2, !dbg !7
  %1338 = lshr i64 %1333, 5, !dbg !7
  %1339 = and i64 %1337, 24, !dbg !7
  %1340 = lshr i64 %1333, 4, !dbg !7
  %1341 = lshr i64 %1333, 7, !dbg !7
  %1342 = sub nsw i64 %1340, %1341, !dbg !7
  %1343 = shl nsw i64 %1342, 2, !dbg !7
  %1344 = and i64 %1343, 4, !dbg !7
  %1345 = sub nsw i64 %1337, %1338, !dbg !7
  %1346 = and i64 %1345, 3, !dbg !7
  %1347 = shl nuw nsw i64 %1341, 3, !dbg !7
  %1348 = shl nuw nsw i64 %1333, 1, !dbg !7
  %1349 = and i64 %1348, 4, !dbg !7
  %1350 = or disjoint i64 %1347, %1349, !dbg !7
  %1351 = and i64 %1336, 2, !dbg !7
  %1352 = or disjoint i64 %1350, %1351, !dbg !7
  %1353 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1339, !dbg !7
  %1354 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1353, i64 %1344, !dbg !7
  %1355 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1354, i64 %1346, !dbg !7
  %.idx = shl nuw nsw i64 %1352, 6, !dbg !7
  %1356 = getelementptr inbounds nuw i8, ptr addrspace(3) %1355, i64 %.idx, !dbg !7
  %1357 = load half, ptr addrspace(3) %1356, align 2, !dbg !7
  %1358 = lshr i64 %1333, 2, !dbg !7
  %1359 = lshr i64 %1333, 5, !dbg !7
  %1360 = and i64 %1358, 24, !dbg !7
  %1361 = lshr i64 %1333, 4, !dbg !7
  %1362 = lshr i64 %1333, 7, !dbg !7
  %1363 = sub nsw i64 %1361, %1362, !dbg !7
  %1364 = shl nsw i64 %1363, 2, !dbg !7
  %1365 = and i64 %1364, 4, !dbg !7
  %1366 = sub nsw i64 %1358, %1359, !dbg !7
  %1367 = and i64 %1366, 3, !dbg !7
  %1368 = shl nuw nsw i64 %1362, 3, !dbg !7
  %1369 = shl nuw nsw i64 %1333, 1, !dbg !7
  %1370 = and i64 %1369, 4, !dbg !7
  %1371 = or disjoint i64 %1368, %1370, !dbg !7
  %1372 = and i64 %1336, 2, !dbg !7
  %1373 = or disjoint i64 %1372, 1, !dbg !7
  %1374 = or disjoint i64 %1371, %1373, !dbg !7
  %1375 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %1360, !dbg !7
  %1376 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1375, i64 %1365, !dbg !7
  %1377 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1376, i64 %1367, !dbg !7
  %.idx100 = shl nuw nsw i64 %1374, 6, !dbg !7
  %1378 = getelementptr inbounds nuw i8, ptr addrspace(3) %1377, i64 %.idx100, !dbg !7
  %1379 = load half, ptr addrspace(3) %1378, align 2, !dbg !7
  %1380 = insertelement <2 x half> poison, half %1357, i64 0, !dbg !7
  %1381 = insertelement <2 x half> %1380, half %1379, i64 1, !dbg !7
  %1382 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %1336, !dbg !7
  store <2 x half> %1381, ptr addrspace(3) %1382, align 2, !dbg !7
  %1383 = add nuw nsw i64 %1333, 256, !dbg !7
  br label %1332, !dbg !7

1384:                                             ; preds = %1332
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %address.thread851 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !18, !range !34, !frisk.register_lifetimes !1
  %address.local852 = zext nneg i32 %address.thread851 to i64, !dbg !6
  %address.local853 = shl nuw nsw i64 %address.local852, 3, !dbg !18
  %address.local854 = lshr i64 %address.local852, 4, !dbg !6
  %address.local855 = mul nsw i64 %address.local854, -128, !dbg !18
  %address.local856 = add nsw i64 %address.local853, %address.local855, !dbg !18
  %address.local857 = and i64 %address.local852, 15, !dbg !8
  %address.local858 = lshr i64 %address.local857, 3, !dbg !18
  %address.local859 = mul nuw nsw i64 %address.local858, 72, !dbg !18
  %address.local860 = add nsw i64 %address.local856, %address.local859, !dbg !18
  %address.local861 = lshr i64 %address.local857, 2, !dbg !8
  %address.local862 = shl nuw nsw i64 %address.local861, 5, !dbg !18
  %address.local863 = add nsw i64 %address.local860, %address.local862, !dbg !18
  %address.local864 = lshr i64 %address.local857, 3, !dbg !18
  %address.local865 = mul nuw nsw i64 %address.local864, -64, !dbg !18
  %address.local866 = add nsw i64 %address.local863, %address.local865, !dbg !18
  %address.local867 = add nuw nsw i64 %address.local857, %address.local858, !dbg !18
  %address.local868 = lshr i64 %address.local867, 2, !dbg !18
  %address.local869 = mul nsw i64 %address.local868, -32, !dbg !18
  %address.local870 = add i64 %address.local866, %address.local869, !dbg !18
  %address.local871 = lshr i64 %address.local852, 4, !dbg !6
  %address.local872 = and i64 %address.local871, 3, !dbg !6
  %address.local873 = shl nuw nsw i64 %address.local872, 9, !dbg !6
  %address.local874 = add i64 %address.local870, %address.local873, !dbg !18
  %address.local875 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local874, !dbg !18
  %1385 = load <8 x half>, ptr addrspace(3) %address.local875, align 2, !dbg !18
  %1386 = extractvalue [2 x <8 x float>] %61, 0, !dbg !30
  %1387 = shufflevector <8 x float> %1386, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1388 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1389 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1388, <4 x float> %1387) #9, !dbg !18
  %address.local876 = mul nuw nsw i64 %address.local872, 576, !dbg !18
  %address.local877 = add nuw nsw i64 %address.local853, %address.local876, !dbg !18
  %address.local878 = add nsw i64 %address.local877, %address.local855, !dbg !18
  %address.local879 = add nsw i64 %address.local878, %address.local859, !dbg !18
  %address.local880 = add nsw i64 %address.local879, %address.local862, !dbg !18
  %address.local881 = shl nuw nsw i32 %address.thread851, 2, !dbg !18
  %address.local882 = and i32 %address.local881, 192, !dbg !18
  %address.local883 = or disjoint i32 %address.local882, 32, !dbg !18
  %address.local884 = shl nuw nsw i32 %address.thread851, 3, !dbg !18
  %address.local885 = and i32 %address.local884, 96, !dbg !18
  %address.local886 = add nuw nsw i32 %address.local883, %address.local885, !dbg !18
  %address.local887 = and i32 %address.local886, 448, !dbg !18
  %address.local888 = zext nneg i32 %address.local887 to i64, !dbg !18
  %address.local889 = sub nsw i64 %address.local880, %address.local888, !dbg !18
  %address.local890 = add i64 %address.local889, %address.local869, !dbg !18
  %address.local891 = add i64 %address.local890, 288, !dbg !18
  %address.local892 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local891, !dbg !18
  %1390 = load <8 x half>, ptr addrspace(3) %address.local892, align 2, !dbg !18
  %1391 = shufflevector <8 x half> %1385, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1392 = shufflevector <8 x float> %1386, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1393 = extractvalue [2 x <8 x float>] %61, 1, !dbg !30
  %1394 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1391, <4 x float> %1389) #9, !dbg !18
  %1395 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1396 = shufflevector <8 x half> %1390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1397 = shufflevector <8 x float> %1393, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %1398 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1395, <4 x float> %1394) #9, !dbg !18
  %1399 = shufflevector <8 x float> %1393, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %1400 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1396, <4 x float> %1398) #9, !dbg !18
  %address.local893 = shl nuw nsw i32 %address.thread851, 3, !dbg !18
  %address.local894 = and i32 %address.local893, 120, !dbg !18
  %address.local895 = shl nuw nsw i32 %address.thread851, 5, !dbg !18
  %address.local896 = and i32 %address.local895, 1536, !dbg !18
  %address.local897 = or disjoint i32 %address.local894, %address.local896, !dbg !18
  %address.local898 = zext nneg i32 %address.local897 to i64, !dbg !18
  %address.local899 = add nuw nsw i64 %address.local859, %address.local898, !dbg !18
  %address.local900 = add nuw nsw i64 %address.local899, %address.local862, !dbg !18
  %address.local901 = add nsw i64 %address.local900, %address.local865, !dbg !18
  %address.local902 = add nsw i64 %address.local901, %address.local869, !dbg !18
  %address.local903 = add i64 %address.local902, 2048, !dbg !18
  %address.local904 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local903, !dbg !18
  %1401 = load <8 x half>, ptr addrspace(3) %address.local904, align 2, !dbg !18
  %address.local905 = add nuw nsw i64 %address.local877, 2304, !dbg !18
  %address.local906 = add nsw i64 %address.local905, %address.local855, !dbg !18
  %address.local907 = add nsw i64 %address.local906, %address.local859, !dbg !18
  %address.local908 = add nsw i64 %address.local907, %address.local862, !dbg !18
  %address.local909 = shl nuw nsw i32 %address.thread851, 2, !dbg !18
  %address.local910 = and i32 %address.local909, 192, !dbg !18
  %address.local911 = or disjoint i32 %address.local910, 288, !dbg !18
  %address.local912 = shl nuw nsw i32 %address.thread851, 3, !dbg !18
  %address.local913 = and i32 %address.local912, 96, !dbg !18
  %address.local914 = add nuw nsw i32 %address.local911, %address.local913, !dbg !18
  %address.local915 = and i32 %address.local914, 960, !dbg !18
  %address.local916 = zext nneg i32 %address.local915 to i64, !dbg !18
  %address.local917 = sub i64 %address.local908, %address.local916, !dbg !18
  %address.local918 = add i64 %address.local917, %address.local869, !dbg !18
  %address.local919 = add i64 %address.local918, 288, !dbg !18
  %address.local920 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local919, !dbg !18
  %1402 = load <8 x half>, ptr addrspace(3) %address.local920, align 2, !dbg !18
  %address.local921 = add nsw i64 %address.local860, %address.local862, !dbg !18
  %address.local922 = shl nuw nsw i64 %address.local852, 3, !dbg !18
  %address.local923 = and i64 %address.local922, 64, !dbg !18
  %address.local924 = sub nsw i64 %address.local921, %address.local923, !dbg !18
  %address.local925 = add nuw nsw i64 %address.local867, 18, !dbg !18
  %address.local926 = lshr i64 %address.local925, 2, !dbg !18
  %address.local927 = mul nsw i64 %address.local926, -32, !dbg !18
  %address.local928 = add i64 %address.local924, %address.local927, !dbg !18
  %address.local929 = add i64 %address.local928, %address.local873, !dbg !18
  %address.local930 = add i64 %address.local929, 272, !dbg !18
  %address.local931 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local930, !dbg !18
  %1403 = load <8 x half>, ptr addrspace(3) %address.local931, align 2, !dbg !18
  %1404 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1405 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1404, <4 x float> %1400) #9, !dbg !18
  %1406 = shufflevector <8 x half> %1401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1407 = shufflevector <8 x half> %1402, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1408 = shufflevector <8 x half> %1402, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1409 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1410 = shufflevector <8 x half> %1403, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %address.local932 = add nsw i64 %address.local879, %address.local862, !dbg !18
  %address.local933 = shl nuw nsw i32 %address.thread851, 2, !dbg !18
  %address.local934 = and i32 %address.local933, 192, !dbg !18
  %address.local935 = add nuw nsw i32 %address.local934, 160, !dbg !18
  %address.local936 = shl nuw nsw i32 %address.thread851, 3, !dbg !18
  %address.local937 = and i32 %address.local936, 96, !dbg !18
  %address.local938 = add nuw nsw i32 %address.local935, %address.local937, !dbg !18
  %address.local939 = and i32 %address.local938, 960, !dbg !18
  %address.local940 = zext nneg i32 %address.local939 to i64, !dbg !18
  %address.local941 = sub nsw i64 %address.local932, %address.local940, !dbg !18
  %address.local942 = add i64 %address.local941, %address.local927, !dbg !18
  %address.local943 = add i64 %address.local942, 688, !dbg !18
  %address.local944 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local943, !dbg !18
  %1411 = load <8 x half>, ptr addrspace(3) %address.local944, align 2, !dbg !18
  %1412 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1406, <4 x float> %1405) #9, !dbg !18
  %address.local945 = add nsw i64 %address.local877, %address.local855, !dbg !18
  %address.local946 = add nsw i64 %address.local945, %address.local859, !dbg !18
  %address.local947 = add nsw i64 %address.local946, %address.local862, !dbg !18
  %address.local948 = lshr i64 %address.local857, 3, !dbg !18
  %address.local949 = add nuw nsw i64 %address.local948, %address.local872, !dbg !18
  %address.local950 = shl nuw nsw i64 %address.local949, 6, !dbg !18
  %address.local951 = sub nsw i64 %address.local947, %address.local950, !dbg !18
  %address.local952 = add i64 %address.local951, %address.local927, !dbg !18
  %address.local953 = add i64 %address.local952, 2320, !dbg !18
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local953, !dbg !18
  %1413 = load <8 x half>, ptr addrspace(3) %address.local954, align 2, !dbg !18
  %address.local955 = add nsw i64 %address.local907, %address.local862, !dbg !18
  %address.local956 = shl nuw nsw i32 %address.thread851, 2, !dbg !18
  %address.local957 = and i32 %address.local956, 192, !dbg !18
  %address.local958 = add nuw nsw i32 %address.local957, 416, !dbg !18
  %address.local959 = shl nuw nsw i32 %address.thread851, 3, !dbg !18
  %address.local960 = and i32 %address.local959, 96, !dbg !18
  %address.local961 = add nuw nsw i32 %address.local958, %address.local960, !dbg !18
  %address.local962 = and i32 %address.local961, 1984, !dbg !18
  %address.local963 = zext nneg i32 %address.local962 to i64, !dbg !18
  %address.local964 = sub i64 %address.local955, %address.local963, !dbg !18
  %address.local965 = add i64 %address.local964, %address.local927, !dbg !18
  %address.local966 = add i64 %address.local965, 688, !dbg !18
  %address.local967 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local966, !dbg !18
  %1414 = load <8 x half>, ptr addrspace(3) %address.local967, align 2, !dbg !18
  %1415 = shufflevector <8 x half> %1411, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1416 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1407, <4 x float> %1412) #9, !dbg !18
  %1417 = shufflevector <8 x half> %1411, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1418 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1419 = shufflevector <8 x half> %1413, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1420 = shufflevector <8 x half> %1414, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1421 = shufflevector <8 x half> %1414, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1422 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1408, <4 x float> %1416) #9, !dbg !18
  %1423 = load <8 x half>, ptr addrspace(3) %address.local875, align 2, !dbg !18
  %1424 = load <8 x half>, ptr addrspace(3) %address.local892, align 2, !dbg !18
  %1425 = load <8 x half>, ptr addrspace(3) %address.local904, align 2, !dbg !18
  %1426 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1409, <4 x float> %1392) #9, !dbg !18
  %1427 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1428 = shufflevector <8 x half> %1423, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1429 = shufflevector <8 x half> %1424, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1430 = shufflevector <8 x half> %1424, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1431 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1432 = shufflevector <8 x half> %1425, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1433 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1410, <4 x float> %1426) #9, !dbg !18
  %1434 = load <8 x half>, ptr addrspace(3) %address.local920, align 2, !dbg !18
  %1435 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1415, <4 x float> %1433) #9, !dbg !18
  %1436 = load <8 x half>, ptr addrspace(3) %address.local931, align 2, !dbg !18
  %1437 = load <8 x half>, ptr addrspace(3) %address.local944, align 2, !dbg !18
  %1438 = shufflevector <8 x half> %1434, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1439 = shufflevector <8 x half> %1434, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1440 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1417, <4 x float> %1435) #9, !dbg !18
  %1441 = shufflevector <8 x half> %1436, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1442 = shufflevector <8 x half> %1436, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1443 = shufflevector <8 x half> %1437, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1444 = shufflevector <8 x half> %1437, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1445 = load <8 x half>, ptr addrspace(3) %address.local954, align 2, !dbg !18
  %1446 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1418, <4 x float> %1440) #9, !dbg !18
  %1447 = load <8 x half>, ptr addrspace(3) %address.local967, align 2, !dbg !18
  %1448 = shufflevector <4 x float> %961, <4 x float> %993, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1449 = insertvalue [2 x <8 x float>] poison, <8 x float> %1448, 0, !dbg !16
  %1450 = shufflevector <4 x float> %1001, <4 x float> %1009, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1451 = insertvalue [2 x <8 x float>] %1449, <8 x float> %1450, 1, !dbg !16
  %1452 = insertelement <1 x float> poison, float %833, i64 0, !dbg !16
  %1453 = insertvalue [2 x <1 x float>] poison, <1 x float> %1452, 0, !dbg !16
  %1454 = insertelement <1 x float> poison, float %844, i64 0, !dbg !16
  %1455 = insertvalue [2 x <1 x float>] %1453, <1 x float> %1454, 1, !dbg !16
  %1456 = shufflevector <8 x half> %1445, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1457 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1419, <4 x float> %1446) #9, !dbg !18
  %1458 = shufflevector <8 x half> %1445, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1459 = shufflevector <4 x float> %1122, <4 x float> %1154, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1460 = insertvalue [2 x <8 x float>] poison, <8 x float> %1459, 0, !dbg !16
  %1461 = shufflevector <4 x float> %1162, <4 x float> %1170, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1462 = insertvalue [2 x <8 x float>] %1460, <8 x float> %1461, 1, !dbg !16
  %1463 = shufflevector <8 x half> %1447, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %1464 = shufflevector <8 x half> %1447, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %1465 = shufflevector <4 x float> %1283, <4 x float> %1315, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1466 = insertvalue [2 x <8 x float>] poison, <8 x float> %1465, 0, !dbg !16
  %1467 = shufflevector <4 x float> %1323, <4 x float> %1331, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1468 = insertvalue [2 x <8 x float>] %1466, <8 x float> %1467, 1, !dbg !16
  %1469 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1420, <4 x float> %1457) #9, !dbg !18
  %1470 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1421, <4 x float> %1469) #9, !dbg !18
  %1471 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1427, <4 x float> %1397) #9, !dbg !18
  %1472 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1428, <4 x float> %1471) #9, !dbg !18
  %1473 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1429, <4 x float> %1472) #9, !dbg !18
  %1474 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1430, <4 x float> %1473) #9, !dbg !18
  %1475 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1431, <4 x float> %1474) #9, !dbg !18
  %1476 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1432, <4 x float> %1475) #9, !dbg !18
  %1477 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1438, <4 x float> %1476) #9, !dbg !18
  %1478 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1439, <4 x float> %1477) #9, !dbg !18
  %1479 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1441, <4 x float> %1399) #9, !dbg !18
  %1480 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1442, <4 x float> %1479) #9, !dbg !18
  %1481 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1443, <4 x float> %1480) #9, !dbg !18
  %1482 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1444, <4 x float> %1481) #9, !dbg !18
  %1483 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1456, <4 x float> %1482) #9, !dbg !18
  %1484 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1458, <4 x float> %1483) #9, !dbg !18
  %1485 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1463, <4 x float> %1484) #9, !dbg !18
  %1486 = call <4 x float> asm sideeffect "s_nop 1\0A\09v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09s_nop 1\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1464, <4 x float> %1485) #9, !dbg !18
  %1487 = shufflevector <4 x float> %1422, <4 x float> %1470, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1488 = insertvalue [2 x <8 x float>] poison, <8 x float> %1487, 0, !dbg !16
  %1489 = shufflevector <4 x float> %1478, <4 x float> %1486, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1490 = insertvalue [2 x <8 x float>] %1488, <8 x float> %1489, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1491 = add nuw nsw i64 %56, 1, !dbg !16
  br label %55, !dbg !16

1492:                                             ; preds = %55
  %1493 = and i32 %19, 16, !dbg !16
  %.not34 = icmp eq i32 %1493, 0, !dbg !16
  %1494 = and i32 %19, 32, !dbg !16
  %.not35 = icmp eq i32 %1494, 0, !dbg !16
  %1495 = extractvalue [2 x <8 x float>] %61, 0, !dbg !16
  %1496 = extractelement <8 x float> %1495, i64 0, !dbg !16
  %1497 = extractelement <8 x float> %1495, i64 1, !dbg !16
  %1498 = extractelement <8 x float> %1495, i64 2, !dbg !16
  %1499 = extractelement <8 x float> %1495, i64 3, !dbg !16
  %1500 = select i1 %.not34, float %1497, float %1496, !dbg !16
  %1501 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %1502 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %1501), !dbg !16
  %1503 = add nuw nsw i32 %1502, 64, !dbg !16
  %1504 = xor i32 %1502, 16, !dbg !16
  %1505 = and i32 %1503, 192, !dbg !16
  %1506 = icmp samesign ult i32 %1504, %1505, !dbg !16
  %1507 = select i1 %1506, i32 %1504, i32 %1502, !dbg !16
  %1508 = shl nuw nsw i32 %1507, 2, !dbg !16
  %1509 = bitcast float %1500 to i32, !dbg !16
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1509), !dbg !16
  %1511 = bitcast i32 %1510 to float, !dbg !16
  %1512 = select i1 %.not34, float %1496, float %1511, !dbg !16
  %1513 = select i1 %.not34, float %1511, float %1497, !dbg !16
  %1514 = select i1 %.not34, float %1499, float %1498, !dbg !16
  %1515 = bitcast float %1514 to i32, !dbg !16
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1515), !dbg !16
  %1517 = bitcast i32 %1516 to float, !dbg !16
  %1518 = select i1 %.not34, float %1498, float %1517, !dbg !16
  %1519 = select i1 %.not34, float %1517, float %1499, !dbg !16
  %1520 = select i1 %.not35, float %1518, float %1512, !dbg !16
  %1521 = xor i32 %1502, 32, !dbg !16
  %1522 = icmp samesign ult i32 %1521, %1505, !dbg !16
  %1523 = select i1 %1522, i32 %1521, i32 %1502, !dbg !16
  %1524 = shl nuw nsw i32 %1523, 2, !dbg !16
  %1525 = bitcast float %1520 to i32, !dbg !16
  %1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1525), !dbg !16
  %1527 = bitcast i32 %1526 to float, !dbg !16
  %1528 = select i1 %.not35, float %1512, float %1527, !dbg !16
  %1529 = select i1 %.not35, float %1527, float %1518, !dbg !16
  %1530 = select i1 %.not35, float %1519, float %1513, !dbg !16
  %1531 = bitcast float %1530 to i32, !dbg !16
  %1532 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1531), !dbg !16
  %1533 = bitcast i32 %1532 to float, !dbg !16
  %1534 = select i1 %.not35, float %1513, float %1533, !dbg !16
  %1535 = select i1 %.not35, float %1533, float %1519, !dbg !16
  %1536 = extractvalue [2 x <8 x float>] %60, 0, !dbg !16
  %1537 = extractelement <8 x float> %1536, i64 0, !dbg !16
  %1538 = extractelement <8 x float> %1536, i64 1, !dbg !16
  %1539 = extractelement <8 x float> %1536, i64 2, !dbg !16
  %1540 = extractelement <8 x float> %1536, i64 3, !dbg !16
  %1541 = select i1 %.not34, float %1538, float %1537, !dbg !16
  %1542 = bitcast float %1541 to i32, !dbg !16
  %1543 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1542), !dbg !16
  %1544 = bitcast i32 %1543 to float, !dbg !16
  %1545 = select i1 %.not34, float %1537, float %1544, !dbg !16
  %1546 = select i1 %.not34, float %1544, float %1538, !dbg !16
  %1547 = select i1 %.not34, float %1540, float %1539, !dbg !16
  %1548 = bitcast float %1547 to i32, !dbg !16
  %1549 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1548), !dbg !16
  %1550 = bitcast i32 %1549 to float, !dbg !16
  %1551 = select i1 %.not34, float %1539, float %1550, !dbg !16
  %1552 = select i1 %.not34, float %1550, float %1540, !dbg !16
  %1553 = select i1 %.not35, float %1551, float %1545, !dbg !16
  %1554 = bitcast float %1553 to i32, !dbg !16
  %1555 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1554), !dbg !16
  %1556 = bitcast i32 %1555 to float, !dbg !16
  %1557 = select i1 %.not35, float %1545, float %1556, !dbg !16
  %1558 = select i1 %.not35, float %1556, float %1551, !dbg !16
  %1559 = select i1 %.not35, float %1552, float %1546, !dbg !16
  %1560 = bitcast float %1559 to i32, !dbg !16
  %1561 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1560), !dbg !16
  %1562 = bitcast i32 %1561 to float, !dbg !16
  %1563 = select i1 %.not35, float %1546, float %1562, !dbg !16
  %1564 = select i1 %.not35, float %1562, float %1552, !dbg !16
  %1565 = extractvalue [2 x <8 x float>] %59, 0, !dbg !16
  %1566 = extractelement <8 x float> %1565, i64 0, !dbg !16
  %1567 = extractelement <8 x float> %1565, i64 1, !dbg !16
  %1568 = extractelement <8 x float> %1565, i64 2, !dbg !16
  %1569 = extractelement <8 x float> %1565, i64 3, !dbg !16
  %1570 = select i1 %.not34, float %1567, float %1566, !dbg !16
  %1571 = bitcast float %1570 to i32, !dbg !16
  %1572 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1571), !dbg !16
  %1573 = bitcast i32 %1572 to float, !dbg !16
  %1574 = select i1 %.not34, float %1566, float %1573, !dbg !16
  %1575 = select i1 %.not34, float %1573, float %1567, !dbg !16
  %1576 = select i1 %.not34, float %1569, float %1568, !dbg !16
  %1577 = bitcast float %1576 to i32, !dbg !16
  %1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1577), !dbg !16
  %1579 = bitcast i32 %1578 to float, !dbg !16
  %1580 = select i1 %.not34, float %1568, float %1579, !dbg !16
  %1581 = select i1 %.not34, float %1579, float %1569, !dbg !16
  %1582 = select i1 %.not35, float %1580, float %1574, !dbg !16
  %1583 = bitcast float %1582 to i32, !dbg !16
  %1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1583), !dbg !16
  %1585 = bitcast i32 %1584 to float, !dbg !16
  %1586 = select i1 %.not35, float %1574, float %1585, !dbg !16
  %1587 = select i1 %.not35, float %1585, float %1580, !dbg !16
  %1588 = select i1 %.not35, float %1581, float %1575, !dbg !16
  %1589 = bitcast float %1588 to i32, !dbg !16
  %1590 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1589), !dbg !16
  %1591 = bitcast i32 %1590 to float, !dbg !16
  %1592 = select i1 %.not35, float %1575, float %1591, !dbg !16
  %1593 = select i1 %.not35, float %1591, float %1581, !dbg !16
  %1594 = extractvalue [2 x <8 x float>] %57, 0, !dbg !16
  %1595 = extractelement <8 x float> %1594, i64 0, !dbg !16
  %1596 = extractelement <8 x float> %1594, i64 1, !dbg !16
  %1597 = extractelement <8 x float> %1594, i64 2, !dbg !16
  %1598 = extractelement <8 x float> %1594, i64 3, !dbg !16
  %1599 = select i1 %.not34, float %1596, float %1595, !dbg !16
  %1600 = bitcast float %1599 to i32, !dbg !16
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1600), !dbg !16
  %1602 = bitcast i32 %1601 to float, !dbg !16
  %1603 = select i1 %.not34, float %1595, float %1602, !dbg !16
  %1604 = select i1 %.not34, float %1602, float %1596, !dbg !16
  %1605 = select i1 %.not34, float %1598, float %1597, !dbg !16
  %1606 = bitcast float %1605 to i32, !dbg !16
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1606), !dbg !16
  %1608 = bitcast i32 %1607 to float, !dbg !16
  %1609 = select i1 %.not34, float %1597, float %1608, !dbg !16
  %1610 = select i1 %.not34, float %1608, float %1598, !dbg !16
  %1611 = select i1 %.not35, float %1609, float %1603, !dbg !16
  %1612 = bitcast float %1611 to i32, !dbg !16
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1612), !dbg !16
  %1614 = select i1 %.not35, float %1610, float %1604, !dbg !16
  %1615 = bitcast float %1614 to i32, !dbg !16
  %1616 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1615), !dbg !16
  %1617 = extractvalue [2 x <1 x float>] %58, 0, !dbg !19
  %1618 = shufflevector <1 x float> %1617, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1619 = extractelement <8 x float> %1495, i64 4, !dbg !16
  %1620 = extractelement <8 x float> %1495, i64 5, !dbg !16
  %1621 = extractelement <8 x float> %1495, i64 6, !dbg !16
  %1622 = extractelement <8 x float> %1495, i64 7, !dbg !16
  %1623 = select i1 %.not34, float %1620, float %1619, !dbg !16
  %1624 = bitcast float %1623 to i32, !dbg !16
  %1625 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1624), !dbg !16
  %1626 = bitcast i32 %1625 to float, !dbg !16
  %1627 = select i1 %.not34, float %1619, float %1626, !dbg !16
  %1628 = select i1 %.not34, float %1626, float %1620, !dbg !16
  %1629 = select i1 %.not34, float %1622, float %1621, !dbg !16
  %1630 = bitcast float %1629 to i32, !dbg !16
  %1631 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1630), !dbg !16
  %1632 = bitcast i32 %1631 to float, !dbg !16
  %1633 = select i1 %.not34, float %1621, float %1632, !dbg !16
  %1634 = select i1 %.not34, float %1632, float %1622, !dbg !16
  %1635 = select i1 %.not35, float %1633, float %1627, !dbg !16
  %1636 = bitcast float %1635 to i32, !dbg !16
  %1637 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1636), !dbg !16
  %1638 = bitcast i32 %1637 to float, !dbg !16
  %1639 = select i1 %.not35, float %1627, float %1638, !dbg !16
  %1640 = select i1 %.not35, float %1638, float %1633, !dbg !16
  %1641 = select i1 %.not35, float %1634, float %1628, !dbg !16
  %1642 = bitcast float %1641 to i32, !dbg !16
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1642), !dbg !16
  %1644 = bitcast i32 %1643 to float, !dbg !16
  %1645 = select i1 %.not35, float %1628, float %1644, !dbg !16
  %1646 = select i1 %.not35, float %1644, float %1634, !dbg !16
  %1647 = extractelement <8 x float> %1536, i64 4, !dbg !16
  %1648 = extractelement <8 x float> %1536, i64 5, !dbg !16
  %1649 = extractelement <8 x float> %1536, i64 6, !dbg !16
  %1650 = extractelement <8 x float> %1536, i64 7, !dbg !16
  %1651 = select i1 %.not34, float %1648, float %1647, !dbg !16
  %1652 = bitcast float %1651 to i32, !dbg !16
  %1653 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1652), !dbg !16
  %1654 = bitcast i32 %1653 to float, !dbg !16
  %1655 = select i1 %.not34, float %1647, float %1654, !dbg !16
  %1656 = select i1 %.not34, float %1654, float %1648, !dbg !16
  %1657 = select i1 %.not34, float %1650, float %1649, !dbg !16
  %1658 = bitcast float %1657 to i32, !dbg !16
  %1659 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1658), !dbg !16
  %1660 = bitcast i32 %1659 to float, !dbg !16
  %1661 = select i1 %.not34, float %1649, float %1660, !dbg !16
  %1662 = select i1 %.not34, float %1660, float %1650, !dbg !16
  %1663 = select i1 %.not35, float %1661, float %1655, !dbg !16
  %1664 = bitcast float %1663 to i32, !dbg !16
  %1665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1664), !dbg !16
  %1666 = bitcast i32 %1665 to float, !dbg !16
  %1667 = select i1 %.not35, float %1655, float %1666, !dbg !16
  %1668 = select i1 %.not35, float %1666, float %1661, !dbg !16
  %1669 = select i1 %.not35, float %1662, float %1656, !dbg !16
  %1670 = bitcast float %1669 to i32, !dbg !16
  %1671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1670), !dbg !16
  %1672 = bitcast i32 %1671 to float, !dbg !16
  %1673 = select i1 %.not35, float %1656, float %1672, !dbg !16
  %1674 = select i1 %.not35, float %1672, float %1662, !dbg !16
  %1675 = extractelement <8 x float> %1565, i64 4, !dbg !16
  %1676 = extractelement <8 x float> %1565, i64 5, !dbg !16
  %1677 = extractelement <8 x float> %1565, i64 6, !dbg !16
  %1678 = extractelement <8 x float> %1565, i64 7, !dbg !16
  %1679 = select i1 %.not34, float %1676, float %1675, !dbg !16
  %1680 = bitcast float %1679 to i32, !dbg !16
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1680), !dbg !16
  %1682 = bitcast i32 %1681 to float, !dbg !16
  %1683 = select i1 %.not34, float %1675, float %1682, !dbg !16
  %1684 = select i1 %.not34, float %1682, float %1676, !dbg !16
  %1685 = select i1 %.not34, float %1678, float %1677, !dbg !16
  %1686 = bitcast float %1685 to i32, !dbg !16
  %1687 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1686), !dbg !16
  %1688 = bitcast i32 %1687 to float, !dbg !16
  %1689 = select i1 %.not34, float %1677, float %1688, !dbg !16
  %1690 = select i1 %.not34, float %1688, float %1678, !dbg !16
  %1691 = select i1 %.not35, float %1689, float %1683, !dbg !16
  %1692 = bitcast float %1691 to i32, !dbg !16
  %1693 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1692), !dbg !16
  %1694 = bitcast i32 %1693 to float, !dbg !16
  %1695 = select i1 %.not35, float %1683, float %1694, !dbg !16
  %1696 = select i1 %.not35, float %1694, float %1689, !dbg !16
  %1697 = select i1 %.not35, float %1690, float %1684, !dbg !16
  %1698 = bitcast float %1697 to i32, !dbg !16
  %1699 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1698), !dbg !16
  %1700 = bitcast i32 %1699 to float, !dbg !16
  %1701 = select i1 %.not35, float %1684, float %1700, !dbg !16
  %1702 = select i1 %.not35, float %1700, float %1690, !dbg !16
  %1703 = extractelement <8 x float> %1594, i64 4, !dbg !16
  %1704 = extractelement <8 x float> %1594, i64 5, !dbg !16
  %1705 = extractelement <8 x float> %1594, i64 6, !dbg !16
  %1706 = extractelement <8 x float> %1594, i64 7, !dbg !16
  %1707 = select i1 %.not34, float %1704, float %1703, !dbg !16
  %1708 = bitcast float %1707 to i32, !dbg !16
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1708), !dbg !16
  %1710 = bitcast i32 %1709 to float, !dbg !16
  %1711 = select i1 %.not34, float %1703, float %1710, !dbg !16
  %1712 = select i1 %.not34, float %1710, float %1704, !dbg !16
  %1713 = select i1 %.not34, float %1706, float %1705, !dbg !16
  %1714 = bitcast float %1713 to i32, !dbg !16
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1714), !dbg !16
  %1716 = bitcast i32 %1715 to float, !dbg !16
  %1717 = select i1 %.not34, float %1705, float %1716, !dbg !16
  %1718 = select i1 %.not34, float %1716, float %1706, !dbg !16
  %1719 = select i1 %.not35, float %1717, float %1711, !dbg !16
  %1720 = bitcast float %1719 to i32, !dbg !16
  %1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1720), !dbg !16
  %1722 = select i1 %.not35, float %1718, float %1712, !dbg !16
  %1723 = bitcast float %1722 to i32, !dbg !16
  %1724 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1723), !dbg !16
  %1725 = extractvalue [2 x <8 x float>] %61, 1, !dbg !16
  %1726 = extractelement <8 x float> %1725, i64 0, !dbg !16
  %1727 = extractelement <8 x float> %1725, i64 1, !dbg !16
  %1728 = extractelement <8 x float> %1725, i64 2, !dbg !16
  %1729 = extractelement <8 x float> %1725, i64 3, !dbg !16
  %1730 = select i1 %.not34, float %1727, float %1726, !dbg !16
  %1731 = bitcast float %1730 to i32, !dbg !16
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1731), !dbg !16
  %1733 = bitcast i32 %1732 to float, !dbg !16
  %1734 = select i1 %.not34, float %1726, float %1733, !dbg !16
  %1735 = select i1 %.not34, float %1733, float %1727, !dbg !16
  %1736 = select i1 %.not34, float %1729, float %1728, !dbg !16
  %1737 = bitcast float %1736 to i32, !dbg !16
  %1738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1737), !dbg !16
  %1739 = bitcast i32 %1738 to float, !dbg !16
  %1740 = select i1 %.not34, float %1728, float %1739, !dbg !16
  %1741 = select i1 %.not34, float %1739, float %1729, !dbg !16
  %1742 = select i1 %.not35, float %1740, float %1734, !dbg !16
  %1743 = bitcast float %1742 to i32, !dbg !16
  %1744 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1743), !dbg !16
  %1745 = bitcast i32 %1744 to float, !dbg !16
  %1746 = select i1 %.not35, float %1734, float %1745, !dbg !16
  %1747 = select i1 %.not35, float %1745, float %1740, !dbg !16
  %1748 = select i1 %.not35, float %1741, float %1735, !dbg !16
  %1749 = bitcast float %1748 to i32, !dbg !16
  %1750 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1749), !dbg !16
  %1751 = bitcast i32 %1750 to float, !dbg !16
  %1752 = select i1 %.not35, float %1735, float %1751, !dbg !16
  %1753 = select i1 %.not35, float %1751, float %1741, !dbg !16
  %1754 = extractvalue [2 x <8 x float>] %60, 1, !dbg !16
  %1755 = extractelement <8 x float> %1754, i64 0, !dbg !16
  %1756 = extractelement <8 x float> %1754, i64 1, !dbg !16
  %1757 = extractelement <8 x float> %1754, i64 2, !dbg !16
  %1758 = extractelement <8 x float> %1754, i64 3, !dbg !16
  %1759 = select i1 %.not34, float %1756, float %1755, !dbg !16
  %1760 = bitcast float %1759 to i32, !dbg !16
  %1761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1760), !dbg !16
  %1762 = bitcast i32 %1761 to float, !dbg !16
  %1763 = select i1 %.not34, float %1755, float %1762, !dbg !16
  %1764 = select i1 %.not34, float %1762, float %1756, !dbg !16
  %1765 = select i1 %.not34, float %1758, float %1757, !dbg !16
  %1766 = bitcast float %1765 to i32, !dbg !16
  %1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1766), !dbg !16
  %1768 = bitcast i32 %1767 to float, !dbg !16
  %1769 = select i1 %.not34, float %1757, float %1768, !dbg !16
  %1770 = select i1 %.not34, float %1768, float %1758, !dbg !16
  %1771 = select i1 %.not35, float %1769, float %1763, !dbg !16
  %1772 = bitcast float %1771 to i32, !dbg !16
  %1773 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1772), !dbg !16
  %1774 = bitcast i32 %1773 to float, !dbg !16
  %1775 = select i1 %.not35, float %1763, float %1774, !dbg !16
  %1776 = select i1 %.not35, float %1774, float %1769, !dbg !16
  %1777 = select i1 %.not35, float %1770, float %1764, !dbg !16
  %1778 = bitcast float %1777 to i32, !dbg !16
  %1779 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1778), !dbg !16
  %1780 = bitcast i32 %1779 to float, !dbg !16
  %1781 = select i1 %.not35, float %1764, float %1780, !dbg !16
  %1782 = select i1 %.not35, float %1780, float %1770, !dbg !16
  %1783 = extractvalue [2 x <8 x float>] %59, 1, !dbg !16
  %1784 = extractelement <8 x float> %1783, i64 0, !dbg !16
  %1785 = extractelement <8 x float> %1783, i64 1, !dbg !16
  %1786 = extractelement <8 x float> %1783, i64 2, !dbg !16
  %1787 = extractelement <8 x float> %1783, i64 3, !dbg !16
  %1788 = select i1 %.not34, float %1785, float %1784, !dbg !16
  %1789 = bitcast float %1788 to i32, !dbg !16
  %1790 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1789), !dbg !16
  %1791 = bitcast i32 %1790 to float, !dbg !16
  %1792 = select i1 %.not34, float %1784, float %1791, !dbg !16
  %1793 = select i1 %.not34, float %1791, float %1785, !dbg !16
  %1794 = select i1 %.not34, float %1787, float %1786, !dbg !16
  %1795 = bitcast float %1794 to i32, !dbg !16
  %1796 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1795), !dbg !16
  %1797 = bitcast i32 %1796 to float, !dbg !16
  %1798 = select i1 %.not34, float %1786, float %1797, !dbg !16
  %1799 = select i1 %.not34, float %1797, float %1787, !dbg !16
  %1800 = select i1 %.not35, float %1798, float %1792, !dbg !16
  %1801 = bitcast float %1800 to i32, !dbg !16
  %1802 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1801), !dbg !16
  %1803 = bitcast i32 %1802 to float, !dbg !16
  %1804 = select i1 %.not35, float %1792, float %1803, !dbg !16
  %1805 = select i1 %.not35, float %1803, float %1798, !dbg !16
  %1806 = select i1 %.not35, float %1799, float %1793, !dbg !16
  %1807 = bitcast float %1806 to i32, !dbg !16
  %1808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1807), !dbg !16
  %1809 = bitcast i32 %1808 to float, !dbg !16
  %1810 = select i1 %.not35, float %1793, float %1809, !dbg !16
  %1811 = select i1 %.not35, float %1809, float %1799, !dbg !16
  %1812 = extractvalue [2 x <8 x float>] %57, 1, !dbg !16
  %1813 = extractelement <8 x float> %1812, i64 0, !dbg !16
  %1814 = extractelement <8 x float> %1812, i64 1, !dbg !16
  %1815 = extractelement <8 x float> %1812, i64 2, !dbg !16
  %1816 = extractelement <8 x float> %1812, i64 3, !dbg !16
  %1817 = select i1 %.not34, float %1814, float %1813, !dbg !16
  %1818 = bitcast float %1817 to i32, !dbg !16
  %1819 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1818), !dbg !16
  %1820 = bitcast i32 %1819 to float, !dbg !16
  %1821 = select i1 %.not34, float %1813, float %1820, !dbg !16
  %1822 = select i1 %.not34, float %1820, float %1814, !dbg !16
  %1823 = select i1 %.not34, float %1816, float %1815, !dbg !16
  %1824 = bitcast float %1823 to i32, !dbg !16
  %1825 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1824), !dbg !16
  %1826 = bitcast i32 %1825 to float, !dbg !16
  %1827 = select i1 %.not34, float %1815, float %1826, !dbg !16
  %1828 = select i1 %.not34, float %1826, float %1816, !dbg !16
  %1829 = select i1 %.not35, float %1827, float %1821, !dbg !16
  %1830 = bitcast float %1829 to i32, !dbg !16
  %1831 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1830), !dbg !16
  %1832 = select i1 %.not35, float %1828, float %1822, !dbg !16
  %1833 = bitcast float %1832 to i32, !dbg !16
  %1834 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1833), !dbg !16
  %1835 = extractvalue [2 x <1 x float>] %58, 1, !dbg !19
  %1836 = shufflevector <1 x float> %1835, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %1837 = extractelement <8 x float> %1725, i64 4, !dbg !16
  %1838 = extractelement <8 x float> %1725, i64 5, !dbg !16
  %1839 = extractelement <8 x float> %1725, i64 6, !dbg !16
  %1840 = extractelement <8 x float> %1725, i64 7, !dbg !16
  %1841 = select i1 %.not34, float %1838, float %1837, !dbg !16
  %1842 = bitcast float %1841 to i32, !dbg !16
  %1843 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1842), !dbg !16
  %1844 = bitcast i32 %1843 to float, !dbg !16
  %1845 = select i1 %.not34, float %1837, float %1844, !dbg !16
  %1846 = select i1 %.not34, float %1844, float %1838, !dbg !16
  %1847 = select i1 %.not34, float %1840, float %1839, !dbg !16
  %1848 = bitcast float %1847 to i32, !dbg !16
  %1849 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1848), !dbg !16
  %1850 = bitcast i32 %1849 to float, !dbg !16
  %1851 = select i1 %.not34, float %1839, float %1850, !dbg !16
  %1852 = select i1 %.not34, float %1850, float %1840, !dbg !16
  %1853 = select i1 %.not35, float %1851, float %1845, !dbg !16
  %1854 = bitcast float %1853 to i32, !dbg !16
  %1855 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1854), !dbg !16
  %1856 = bitcast i32 %1855 to float, !dbg !16
  %1857 = select i1 %.not35, float %1845, float %1856, !dbg !16
  %1858 = select i1 %.not35, float %1856, float %1851, !dbg !16
  %1859 = select i1 %.not35, float %1852, float %1846, !dbg !16
  %1860 = bitcast float %1859 to i32, !dbg !16
  %1861 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1860), !dbg !16
  %1862 = bitcast i32 %1861 to float, !dbg !16
  %1863 = select i1 %.not35, float %1846, float %1862, !dbg !16
  %1864 = select i1 %.not35, float %1862, float %1852, !dbg !16
  %1865 = extractelement <8 x float> %1754, i64 4, !dbg !16
  %1866 = extractelement <8 x float> %1754, i64 5, !dbg !16
  %1867 = extractelement <8 x float> %1754, i64 6, !dbg !16
  %1868 = extractelement <8 x float> %1754, i64 7, !dbg !16
  %1869 = select i1 %.not34, float %1866, float %1865, !dbg !16
  %1870 = bitcast float %1869 to i32, !dbg !16
  %1871 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1870), !dbg !16
  %1872 = bitcast i32 %1871 to float, !dbg !16
  %1873 = select i1 %.not34, float %1865, float %1872, !dbg !16
  %1874 = select i1 %.not34, float %1872, float %1866, !dbg !16
  %1875 = select i1 %.not34, float %1868, float %1867, !dbg !16
  %1876 = bitcast float %1875 to i32, !dbg !16
  %1877 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1876), !dbg !16
  %1878 = bitcast i32 %1877 to float, !dbg !16
  %1879 = select i1 %.not34, float %1867, float %1878, !dbg !16
  %1880 = select i1 %.not34, float %1878, float %1868, !dbg !16
  %1881 = select i1 %.not35, float %1879, float %1873, !dbg !16
  %1882 = bitcast float %1881 to i32, !dbg !16
  %1883 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1882), !dbg !16
  %1884 = bitcast i32 %1883 to float, !dbg !16
  %1885 = select i1 %.not35, float %1873, float %1884, !dbg !16
  %1886 = select i1 %.not35, float %1884, float %1879, !dbg !16
  %1887 = select i1 %.not35, float %1880, float %1874, !dbg !16
  %1888 = bitcast float %1887 to i32, !dbg !16
  %1889 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1888), !dbg !16
  %1890 = bitcast i32 %1889 to float, !dbg !16
  %1891 = select i1 %.not35, float %1874, float %1890, !dbg !16
  %1892 = select i1 %.not35, float %1890, float %1880, !dbg !16
  %1893 = extractelement <8 x float> %1783, i64 4, !dbg !16
  %1894 = extractelement <8 x float> %1783, i64 5, !dbg !16
  %1895 = extractelement <8 x float> %1783, i64 6, !dbg !16
  %1896 = extractelement <8 x float> %1783, i64 7, !dbg !16
  %1897 = select i1 %.not34, float %1894, float %1893, !dbg !16
  %1898 = bitcast float %1897 to i32, !dbg !16
  %1899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1898), !dbg !16
  %1900 = bitcast i32 %1899 to float, !dbg !16
  %1901 = select i1 %.not34, float %1893, float %1900, !dbg !16
  %1902 = select i1 %.not34, float %1900, float %1894, !dbg !16
  %1903 = select i1 %.not34, float %1896, float %1895, !dbg !16
  %1904 = bitcast float %1903 to i32, !dbg !16
  %1905 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1904), !dbg !16
  %1906 = bitcast i32 %1905 to float, !dbg !16
  %1907 = select i1 %.not34, float %1895, float %1906, !dbg !16
  %1908 = select i1 %.not34, float %1906, float %1896, !dbg !16
  %1909 = select i1 %.not35, float %1907, float %1901, !dbg !16
  %1910 = bitcast float %1909 to i32, !dbg !16
  %1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1910), !dbg !16
  %1912 = bitcast i32 %1911 to float, !dbg !16
  %1913 = select i1 %.not35, float %1901, float %1912, !dbg !16
  %1914 = select i1 %.not35, float %1912, float %1907, !dbg !16
  %1915 = select i1 %.not35, float %1908, float %1902, !dbg !16
  %1916 = bitcast float %1915 to i32, !dbg !16
  %1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1916), !dbg !16
  %1918 = bitcast i32 %1917 to float, !dbg !16
  %1919 = select i1 %.not35, float %1902, float %1918, !dbg !16
  %1920 = select i1 %.not35, float %1918, float %1908, !dbg !16
  %1921 = extractelement <8 x float> %1812, i64 4, !dbg !16
  %1922 = extractelement <8 x float> %1812, i64 5, !dbg !16
  %1923 = extractelement <8 x float> %1812, i64 6, !dbg !16
  %1924 = extractelement <8 x float> %1812, i64 7, !dbg !16
  %1925 = select i1 %.not34, float %1922, float %1921, !dbg !16
  %1926 = bitcast float %1925 to i32, !dbg !16
  %1927 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1926), !dbg !16
  %1928 = bitcast i32 %1927 to float, !dbg !16
  %1929 = select i1 %.not34, float %1921, float %1928, !dbg !16
  %1930 = select i1 %.not34, float %1928, float %1922, !dbg !16
  %1931 = select i1 %.not34, float %1924, float %1923, !dbg !16
  %1932 = bitcast float %1931 to i32, !dbg !16
  %1933 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1508, i32 %1932), !dbg !16
  %1934 = bitcast i32 %1933 to float, !dbg !16
  %1935 = select i1 %.not34, float %1923, float %1934, !dbg !16
  %1936 = select i1 %.not34, float %1934, float %1924, !dbg !16
  %1937 = select i1 %.not35, float %1935, float %1929, !dbg !16
  %1938 = bitcast float %1937 to i32, !dbg !16
  %1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1938), !dbg !16
  %1940 = select i1 %.not35, float %1936, float %1930, !dbg !16
  %1941 = bitcast float %1940 to i32, !dbg !16
  %1942 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1524, i32 %1941), !dbg !16
  br i1 true, label %1943, label %2007, !dbg !21

1943:                                             ; preds = %1492
  %1944 = bitcast i32 %1939 to float, !dbg !16
  %1945 = select i1 %.not35, float %1929, float %1944, !dbg !16
  %1946 = insertelement <4 x float> poison, float %1945, i64 0, !dbg !19
  %1947 = bitcast i32 %1942 to float, !dbg !16
  %1948 = select i1 %.not35, float %1930, float %1947, !dbg !16
  %1949 = insertelement <4 x float> %1946, float %1948, i64 1, !dbg !19
  %1950 = select i1 %.not35, float %1944, float %1935, !dbg !16
  %1951 = insertelement <4 x float> %1949, float %1950, i64 2, !dbg !19
  %1952 = select i1 %.not35, float %1947, float %1936, !dbg !16
  %1953 = insertelement <4 x float> %1951, float %1952, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1836, !dbg !19
  %normalized = fmul <4 x float> %1953, %row.reciprocal, !dbg !19
  %1954 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %1955 = extractelement <4 x half> %1954, i64 3, !dbg !20
  %1956 = extractelement <4 x half> %1954, i64 2, !dbg !20
  %1957 = extractelement <4 x half> %1954, i64 1, !dbg !20
  %1958 = extractelement <4 x half> %1954, i64 0, !dbg !20
  %1959 = bitcast i32 %1831 to float, !dbg !16
  %1960 = select i1 %.not35, float %1821, float %1959, !dbg !16
  %1961 = insertelement <4 x float> poison, float %1960, i64 0, !dbg !19
  %1962 = bitcast i32 %1834 to float, !dbg !16
  %1963 = select i1 %.not35, float %1822, float %1962, !dbg !16
  %1964 = insertelement <4 x float> %1961, float %1963, i64 1, !dbg !19
  %1965 = select i1 %.not35, float %1959, float %1827, !dbg !16
  %1966 = insertelement <4 x float> %1964, float %1965, i64 2, !dbg !19
  %1967 = select i1 %.not35, float %1962, float %1828, !dbg !16
  %1968 = insertelement <4 x float> %1966, float %1967, i64 3, !dbg !19
  %normalized114 = fmul <4 x float> %1968, %row.reciprocal, !dbg !19
  %1969 = fptrunc <4 x float> %normalized114 to <4 x half>, !dbg !20
  %1970 = extractelement <4 x half> %1969, i64 3, !dbg !20
  %1971 = extractelement <4 x half> %1969, i64 2, !dbg !20
  %1972 = extractelement <4 x half> %1969, i64 1, !dbg !20
  %1973 = extractelement <4 x half> %1969, i64 0, !dbg !20
  %1974 = bitcast i32 %1721 to float, !dbg !16
  %1975 = select i1 %.not35, float %1711, float %1974, !dbg !16
  %1976 = insertelement <4 x float> poison, float %1975, i64 0, !dbg !19
  %1977 = bitcast i32 %1724 to float, !dbg !16
  %1978 = select i1 %.not35, float %1712, float %1977, !dbg !16
  %1979 = insertelement <4 x float> %1976, float %1978, i64 1, !dbg !19
  %1980 = select i1 %.not35, float %1974, float %1717, !dbg !16
  %1981 = insertelement <4 x float> %1979, float %1980, i64 2, !dbg !19
  %1982 = select i1 %.not35, float %1977, float %1718, !dbg !16
  %1983 = insertelement <4 x float> %1981, float %1982, i64 3, !dbg !19
  %row.reciprocal115 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1618, !dbg !19
  %normalized116 = fmul <4 x float> %1983, %row.reciprocal115, !dbg !19
  %1984 = fptrunc <4 x float> %normalized116 to <4 x half>, !dbg !20
  %1985 = extractelement <4 x half> %1984, i64 3, !dbg !20
  %1986 = extractelement <4 x half> %1984, i64 2, !dbg !20
  %1987 = extractelement <4 x half> %1984, i64 1, !dbg !20
  %1988 = extractelement <4 x half> %1984, i64 0, !dbg !20
  %1989 = bitcast i32 %1613 to float, !dbg !16
  %1990 = select i1 %.not35, float %1603, float %1989, !dbg !16
  %1991 = insertelement <4 x float> poison, float %1990, i64 0, !dbg !19
  %1992 = bitcast i32 %1616 to float, !dbg !16
  %1993 = select i1 %.not35, float %1604, float %1992, !dbg !16
  %1994 = insertelement <4 x float> %1991, float %1993, i64 1, !dbg !19
  %1995 = select i1 %.not35, float %1989, float %1609, !dbg !16
  %1996 = insertelement <4 x float> %1994, float %1995, i64 2, !dbg !19
  %1997 = select i1 %.not35, float %1992, float %1610, !dbg !16
  %1998 = insertelement <4 x float> %1996, float %1997, i64 3, !dbg !19
  %normalized117 = fmul <4 x float> %1998, %row.reciprocal115, !dbg !19
  %1999 = fptrunc <4 x float> %normalized117 to <4 x half>, !dbg !20
  %2000 = extractelement <4 x half> %1999, i64 3, !dbg !20
  %2001 = extractelement <4 x half> %1999, i64 2, !dbg !20
  %2002 = extractelement <4 x half> %1999, i64 1, !dbg !20
  %2003 = extractelement <4 x half> %1999, i64 0, !dbg !20
  %2004 = add nsw i64 %8, %13, !dbg !21
  %2005 = shl nsw i64 %2004, 19, !dbg !21
  %2006 = and i64 %2005, 16252928, !dbg !21
  %address.thread968 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local969 = zext nneg i32 %address.thread968 to i64, !dbg !6
  %address.local970 = add nsw i64 %16, %address.local969, !dbg !21
  %address.local971 = lshr i64 %address.local969, 6, !dbg !6
  %address.local972 = shl nuw nsw i64 %address.local971, 4, !dbg !21
  %address.local973 = add nsw i64 %address.local970, %address.local972, !dbg !21
  %address.local974 = and i64 %address.local969, 240, !dbg !21
  %address.local975 = sub nsw i64 %address.local973, %address.local974, !dbg !21
  %address.local976 = add nsw i64 %address.local975, %17, !dbg !21
  %address.local977 = shl nuw nsw i64 %address.local976, 7, !dbg !21
  %address.local978 = add nuw nsw i64 %2006, %address.local977, !dbg !21
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local980 = lshr i64 %address.local969, 4, !dbg !6
  %address.local981 = and i64 %address.local980, 3, !dbg !6
  %address.local982 = shl nuw nsw i64 %address.local981, 2, !dbg !21
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local979, i64 %address.local982, !dbg !21
  store half %2003, ptr addrspace(1) %address.local983, align 2, !dbg !21
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local985 = or disjoint i64 %address.local982, 1, !dbg !21
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local984, i64 %address.local985, !dbg !21
  store half %2002, ptr addrspace(1) %address.local986, align 2, !dbg !21
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local988 = or disjoint i64 %address.local982, 2, !dbg !21
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local987, i64 %address.local988, !dbg !21
  store half %2001, ptr addrspace(1) %address.local989, align 2, !dbg !21
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local991 = or disjoint i64 %address.local982, 3, !dbg !21
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local990, i64 %address.local991, !dbg !21
  store half %2000, ptr addrspace(1) %address.local992, align 2, !dbg !21
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local994 = or disjoint i64 %address.local982, 16, !dbg !21
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local993, i64 %address.local994, !dbg !21
  store half %1988, ptr addrspace(1) %address.local995, align 2, !dbg !21
  %address.local996 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local997 = or disjoint i64 %address.local982, 17, !dbg !21
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local996, i64 %address.local997, !dbg !21
  store half %1987, ptr addrspace(1) %address.local998, align 2, !dbg !21
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local1000 = or disjoint i64 %address.local982, 18, !dbg !21
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local999, i64 %address.local1000, !dbg !21
  store half %1986, ptr addrspace(1) %address.local1001, align 2, !dbg !21
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local978, !dbg !21
  %address.local1003 = or disjoint i64 %address.local982, 19, !dbg !21
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1002, i64 %address.local1003, !dbg !21
  store half %1985, ptr addrspace(1) %address.local1004, align 2, !dbg !21
  %address.local1005 = shl nsw i64 %address.local976, 7, !dbg !21
  %address.local1006 = add i64 %address.local1005, 8192, !dbg !21
  %address.local1007 = add nuw nsw i64 %2006, %address.local1006, !dbg !21
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1008, i64 %address.local982, !dbg !21
  store half %1973, ptr addrspace(1) %address.local1009, align 2, !dbg !21
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local985, !dbg !21
  store half %1972, ptr addrspace(1) %address.local1011, align 2, !dbg !21
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1012, i64 %address.local988, !dbg !21
  store half %1971, ptr addrspace(1) %address.local1013, align 2, !dbg !21
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local991, !dbg !21
  store half %1970, ptr addrspace(1) %address.local1015, align 2, !dbg !21
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local994, !dbg !21
  store half %1958, ptr addrspace(1) %address.local1017, align 2, !dbg !21
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1018, i64 %address.local997, !dbg !21
  store half %1957, ptr addrspace(1) %address.local1019, align 2, !dbg !21
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1000, !dbg !21
  store half %1956, ptr addrspace(1) %address.local1021, align 2, !dbg !21
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1007, !dbg !21
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1022, i64 %address.local1003, !dbg !21
  store half %1955, ptr addrspace(1) %address.local1023, align 2, !dbg !21
  br label %2007, !dbg !21

2007:                                             ; preds = %1943, %1492
  br i1 true, label %2008, label %2046, !dbg !21

2008:                                             ; preds = %2007
  %2009 = insertelement <4 x float> poison, float %1913, i64 0, !dbg !19
  %2010 = insertelement <4 x float> %2009, float %1919, i64 1, !dbg !19
  %2011 = insertelement <4 x float> %2010, float %1914, i64 2, !dbg !19
  %2012 = insertelement <4 x float> %2011, float %1920, i64 3, !dbg !19
  %row.reciprocal118 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1836, !dbg !19
  %normalized119 = fmul <4 x float> %2012, %row.reciprocal118, !dbg !19
  %2013 = fptrunc <4 x float> %normalized119 to <4 x half>, !dbg !20
  %2014 = extractelement <4 x half> %2013, i64 3, !dbg !20
  %2015 = extractelement <4 x half> %2013, i64 2, !dbg !20
  %2016 = extractelement <4 x half> %2013, i64 1, !dbg !20
  %2017 = extractelement <4 x half> %2013, i64 0, !dbg !20
  %2018 = insertelement <4 x float> poison, float %1804, i64 0, !dbg !19
  %2019 = insertelement <4 x float> %2018, float %1810, i64 1, !dbg !19
  %2020 = insertelement <4 x float> %2019, float %1805, i64 2, !dbg !19
  %2021 = insertelement <4 x float> %2020, float %1811, i64 3, !dbg !19
  %normalized120 = fmul <4 x float> %2021, %row.reciprocal118, !dbg !19
  %2022 = fptrunc <4 x float> %normalized120 to <4 x half>, !dbg !20
  %2023 = extractelement <4 x half> %2022, i64 3, !dbg !20
  %2024 = extractelement <4 x half> %2022, i64 2, !dbg !20
  %2025 = extractelement <4 x half> %2022, i64 1, !dbg !20
  %2026 = extractelement <4 x half> %2022, i64 0, !dbg !20
  %2027 = insertelement <4 x float> poison, float %1695, i64 0, !dbg !19
  %2028 = insertelement <4 x float> %2027, float %1701, i64 1, !dbg !19
  %2029 = insertelement <4 x float> %2028, float %1696, i64 2, !dbg !19
  %2030 = insertelement <4 x float> %2029, float %1702, i64 3, !dbg !19
  %row.reciprocal121 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1618, !dbg !19
  %normalized122 = fmul <4 x float> %2030, %row.reciprocal121, !dbg !19
  %2031 = fptrunc <4 x float> %normalized122 to <4 x half>, !dbg !20
  %2032 = extractelement <4 x half> %2031, i64 3, !dbg !20
  %2033 = extractelement <4 x half> %2031, i64 2, !dbg !20
  %2034 = extractelement <4 x half> %2031, i64 1, !dbg !20
  %2035 = extractelement <4 x half> %2031, i64 0, !dbg !20
  %2036 = insertelement <4 x float> poison, float %1586, i64 0, !dbg !19
  %2037 = insertelement <4 x float> %2036, float %1592, i64 1, !dbg !19
  %2038 = insertelement <4 x float> %2037, float %1587, i64 2, !dbg !19
  %2039 = insertelement <4 x float> %2038, float %1593, i64 3, !dbg !19
  %normalized123 = fmul <4 x float> %2039, %row.reciprocal121, !dbg !19
  %2040 = fptrunc <4 x float> %normalized123 to <4 x half>, !dbg !20
  %2041 = extractelement <4 x half> %2040, i64 3, !dbg !20
  %2042 = extractelement <4 x half> %2040, i64 2, !dbg !20
  %2043 = extractelement <4 x half> %2040, i64 1, !dbg !20
  %2044 = extractelement <4 x half> %2040, i64 0, !dbg !20
  %2045 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1024 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1025 = zext nneg i32 %address.thread1024 to i64, !dbg !6
  %address.local1026 = add nsw i64 %16, %address.local1025, !dbg !21
  %address.local1027 = lshr i64 %address.local1025, 6, !dbg !6
  %address.local1028 = shl nuw nsw i64 %address.local1027, 4, !dbg !21
  %address.local1029 = add nsw i64 %address.local1026, %address.local1028, !dbg !21
  %address.local1030 = and i64 %address.local1025, 240, !dbg !21
  %address.local1031 = sub nsw i64 %address.local1029, %address.local1030, !dbg !21
  %address.local1032 = add nsw i64 %address.local1031, %17, !dbg !21
  %address.local1033 = shl nuw nsw i64 %address.local1032, 7, !dbg !21
  %address.local1034 = add nuw nsw i64 %2045, %address.local1033, !dbg !21
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1036 = lshr i64 %address.local1025, 4, !dbg !6
  %address.local1037 = and i64 %address.local1036, 3, !dbg !6
  %address.local1038 = shl nuw nsw i64 %address.local1037, 2, !dbg !21
  %address.local1039 = or disjoint i64 %address.local1038, 32, !dbg !21
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1039, !dbg !21
  store half %2044, ptr addrspace(1) %address.local1040, align 2, !dbg !21
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1042 = or disjoint i64 %address.local1038, 33, !dbg !21
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1042, !dbg !21
  store half %2043, ptr addrspace(1) %address.local1043, align 2, !dbg !21
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1045 = or disjoint i64 %address.local1038, 34, !dbg !21
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1044, i64 %address.local1045, !dbg !21
  store half %2042, ptr addrspace(1) %address.local1046, align 2, !dbg !21
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1048 = or disjoint i64 %address.local1038, 35, !dbg !21
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1047, i64 %address.local1048, !dbg !21
  store half %2041, ptr addrspace(1) %address.local1049, align 2, !dbg !21
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1051 = or disjoint i64 %address.local1038, 48, !dbg !21
  %address.local1052 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1050, i64 %address.local1051, !dbg !21
  store half %2035, ptr addrspace(1) %address.local1052, align 2, !dbg !21
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1054 = or disjoint i64 %address.local1038, 49, !dbg !21
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1053, i64 %address.local1054, !dbg !21
  store half %2034, ptr addrspace(1) %address.local1055, align 2, !dbg !21
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1057 = or disjoint i64 %address.local1038, 50, !dbg !21
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1056, i64 %address.local1057, !dbg !21
  store half %2033, ptr addrspace(1) %address.local1058, align 2, !dbg !21
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !21
  %address.local1060 = or disjoint i64 %address.local1038, 51, !dbg !21
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1059, i64 %address.local1060, !dbg !21
  store half %2032, ptr addrspace(1) %address.local1061, align 2, !dbg !21
  %address.local1062 = shl nsw i64 %address.local1032, 7, !dbg !21
  %address.local1063 = add i64 %address.local1062, 8192, !dbg !21
  %address.local1064 = add nuw nsw i64 %2045, %address.local1063, !dbg !21
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1065, i64 %address.local1039, !dbg !21
  store half %2026, ptr addrspace(1) %address.local1066, align 2, !dbg !21
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1042, !dbg !21
  store half %2025, ptr addrspace(1) %address.local1068, align 2, !dbg !21
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1045, !dbg !21
  store half %2024, ptr addrspace(1) %address.local1070, align 2, !dbg !21
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1048, !dbg !21
  store half %2023, ptr addrspace(1) %address.local1072, align 2, !dbg !21
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1073, i64 %address.local1051, !dbg !21
  store half %2017, ptr addrspace(1) %address.local1074, align 2, !dbg !21
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1075, i64 %address.local1054, !dbg !21
  store half %2016, ptr addrspace(1) %address.local1076, align 2, !dbg !21
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1057, !dbg !21
  store half %2015, ptr addrspace(1) %address.local1078, align 2, !dbg !21
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1064, !dbg !21
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1079, i64 %address.local1060, !dbg !21
  store half %2014, ptr addrspace(1) %address.local1080, align 2, !dbg !21
  br label %2046, !dbg !21

2046:                                             ; preds = %2008, %2007
  br i1 true, label %2047, label %2085, !dbg !21

2047:                                             ; preds = %2046
  %2048 = insertelement <4 x float> poison, float %1885, i64 0, !dbg !19
  %2049 = insertelement <4 x float> %2048, float %1891, i64 1, !dbg !19
  %2050 = insertelement <4 x float> %2049, float %1886, i64 2, !dbg !19
  %2051 = insertelement <4 x float> %2050, float %1892, i64 3, !dbg !19
  %row.reciprocal124 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1836, !dbg !19
  %normalized125 = fmul <4 x float> %2051, %row.reciprocal124, !dbg !19
  %2052 = fptrunc <4 x float> %normalized125 to <4 x half>, !dbg !20
  %2053 = extractelement <4 x half> %2052, i64 3, !dbg !20
  %2054 = extractelement <4 x half> %2052, i64 2, !dbg !20
  %2055 = extractelement <4 x half> %2052, i64 1, !dbg !20
  %2056 = extractelement <4 x half> %2052, i64 0, !dbg !20
  %2057 = insertelement <4 x float> poison, float %1775, i64 0, !dbg !19
  %2058 = insertelement <4 x float> %2057, float %1781, i64 1, !dbg !19
  %2059 = insertelement <4 x float> %2058, float %1776, i64 2, !dbg !19
  %2060 = insertelement <4 x float> %2059, float %1782, i64 3, !dbg !19
  %normalized126 = fmul <4 x float> %2060, %row.reciprocal124, !dbg !19
  %2061 = fptrunc <4 x float> %normalized126 to <4 x half>, !dbg !20
  %2062 = extractelement <4 x half> %2061, i64 3, !dbg !20
  %2063 = extractelement <4 x half> %2061, i64 2, !dbg !20
  %2064 = extractelement <4 x half> %2061, i64 1, !dbg !20
  %2065 = extractelement <4 x half> %2061, i64 0, !dbg !20
  %2066 = insertelement <4 x float> poison, float %1667, i64 0, !dbg !19
  %2067 = insertelement <4 x float> %2066, float %1673, i64 1, !dbg !19
  %2068 = insertelement <4 x float> %2067, float %1668, i64 2, !dbg !19
  %2069 = insertelement <4 x float> %2068, float %1674, i64 3, !dbg !19
  %row.reciprocal127 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1618, !dbg !19
  %normalized128 = fmul <4 x float> %2069, %row.reciprocal127, !dbg !19
  %2070 = fptrunc <4 x float> %normalized128 to <4 x half>, !dbg !20
  %2071 = extractelement <4 x half> %2070, i64 3, !dbg !20
  %2072 = extractelement <4 x half> %2070, i64 2, !dbg !20
  %2073 = extractelement <4 x half> %2070, i64 1, !dbg !20
  %2074 = extractelement <4 x half> %2070, i64 0, !dbg !20
  %2075 = insertelement <4 x float> poison, float %1557, i64 0, !dbg !19
  %2076 = insertelement <4 x float> %2075, float %1563, i64 1, !dbg !19
  %2077 = insertelement <4 x float> %2076, float %1558, i64 2, !dbg !19
  %2078 = insertelement <4 x float> %2077, float %1564, i64 3, !dbg !19
  %normalized129 = fmul <4 x float> %2078, %row.reciprocal127, !dbg !19
  %2079 = fptrunc <4 x float> %normalized129 to <4 x half>, !dbg !20
  %2080 = extractelement <4 x half> %2079, i64 3, !dbg !20
  %2081 = extractelement <4 x half> %2079, i64 2, !dbg !20
  %2082 = extractelement <4 x half> %2079, i64 1, !dbg !20
  %2083 = extractelement <4 x half> %2079, i64 0, !dbg !20
  %2084 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1081 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1082 = zext nneg i32 %address.thread1081 to i64, !dbg !6
  %address.local1083 = add nsw i64 %16, %address.local1082, !dbg !21
  %address.local1084 = lshr i64 %address.local1082, 6, !dbg !6
  %address.local1085 = shl nuw nsw i64 %address.local1084, 4, !dbg !21
  %address.local1086 = add nsw i64 %address.local1083, %address.local1085, !dbg !21
  %address.local1087 = and i64 %address.local1082, 240, !dbg !21
  %address.local1088 = sub nsw i64 %address.local1086, %address.local1087, !dbg !21
  %address.local1089 = add nsw i64 %address.local1088, %17, !dbg !21
  %address.local1090 = shl nuw nsw i64 %address.local1089, 7, !dbg !21
  %address.local1091 = add nuw nsw i64 %2084, %address.local1090, !dbg !21
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1093 = lshr i64 %address.local1082, 4, !dbg !6
  %address.local1094 = and i64 %address.local1093, 3, !dbg !6
  %address.local1095 = shl nuw nsw i64 %address.local1094, 2, !dbg !21
  %address.local1096 = or disjoint i64 %address.local1095, 64, !dbg !21
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1096, !dbg !21
  store half %2083, ptr addrspace(1) %address.local1097, align 2, !dbg !21
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1099 = or disjoint i64 %address.local1095, 65, !dbg !21
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1099, !dbg !21
  store half %2082, ptr addrspace(1) %address.local1100, align 2, !dbg !21
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1102 = or disjoint i64 %address.local1095, 66, !dbg !21
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1102, !dbg !21
  store half %2081, ptr addrspace(1) %address.local1103, align 2, !dbg !21
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1105 = or disjoint i64 %address.local1095, 67, !dbg !21
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1105, !dbg !21
  store half %2080, ptr addrspace(1) %address.local1106, align 2, !dbg !21
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1108 = or disjoint i64 %address.local1095, 80, !dbg !21
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1107, i64 %address.local1108, !dbg !21
  store half %2074, ptr addrspace(1) %address.local1109, align 2, !dbg !21
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1111 = or disjoint i64 %address.local1095, 81, !dbg !21
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1110, i64 %address.local1111, !dbg !21
  store half %2073, ptr addrspace(1) %address.local1112, align 2, !dbg !21
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1114 = or disjoint i64 %address.local1095, 82, !dbg !21
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1114, !dbg !21
  store half %2072, ptr addrspace(1) %address.local1115, align 2, !dbg !21
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !21
  %address.local1117 = or disjoint i64 %address.local1095, 83, !dbg !21
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1117, !dbg !21
  store half %2071, ptr addrspace(1) %address.local1118, align 2, !dbg !21
  %address.local1119 = shl nsw i64 %address.local1089, 7, !dbg !21
  %address.local1120 = add i64 %address.local1119, 8192, !dbg !21
  %address.local1121 = add nuw nsw i64 %2084, %address.local1120, !dbg !21
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1122, i64 %address.local1096, !dbg !21
  store half %2065, ptr addrspace(1) %address.local1123, align 2, !dbg !21
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1099, !dbg !21
  store half %2064, ptr addrspace(1) %address.local1125, align 2, !dbg !21
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1126, i64 %address.local1102, !dbg !21
  store half %2063, ptr addrspace(1) %address.local1127, align 2, !dbg !21
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1128, i64 %address.local1105, !dbg !21
  store half %2062, ptr addrspace(1) %address.local1129, align 2, !dbg !21
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1130, i64 %address.local1108, !dbg !21
  store half %2056, ptr addrspace(1) %address.local1131, align 2, !dbg !21
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1132, i64 %address.local1111, !dbg !21
  store half %2055, ptr addrspace(1) %address.local1133, align 2, !dbg !21
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1134, i64 %address.local1114, !dbg !21
  store half %2054, ptr addrspace(1) %address.local1135, align 2, !dbg !21
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1121, !dbg !21
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1136, i64 %address.local1117, !dbg !21
  store half %2053, ptr addrspace(1) %address.local1137, align 2, !dbg !21
  br label %2085, !dbg !21

2085:                                             ; preds = %2047, %2046
  br i1 true, label %2086, label %2124, !dbg !21

2086:                                             ; preds = %2085
  %2087 = insertelement <4 x float> poison, float %1857, i64 0, !dbg !19
  %2088 = insertelement <4 x float> %2087, float %1863, i64 1, !dbg !19
  %2089 = insertelement <4 x float> %2088, float %1858, i64 2, !dbg !19
  %2090 = insertelement <4 x float> %2089, float %1864, i64 3, !dbg !19
  %row.reciprocal130 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1836, !dbg !19
  %normalized131 = fmul <4 x float> %2090, %row.reciprocal130, !dbg !19
  %2091 = fptrunc <4 x float> %normalized131 to <4 x half>, !dbg !20
  %2092 = extractelement <4 x half> %2091, i64 3, !dbg !20
  %2093 = extractelement <4 x half> %2091, i64 2, !dbg !20
  %2094 = extractelement <4 x half> %2091, i64 1, !dbg !20
  %2095 = extractelement <4 x half> %2091, i64 0, !dbg !20
  %2096 = insertelement <4 x float> poison, float %1746, i64 0, !dbg !19
  %2097 = insertelement <4 x float> %2096, float %1752, i64 1, !dbg !19
  %2098 = insertelement <4 x float> %2097, float %1747, i64 2, !dbg !19
  %2099 = insertelement <4 x float> %2098, float %1753, i64 3, !dbg !19
  %normalized132 = fmul <4 x float> %2099, %row.reciprocal130, !dbg !19
  %2100 = fptrunc <4 x float> %normalized132 to <4 x half>, !dbg !20
  %2101 = extractelement <4 x half> %2100, i64 3, !dbg !20
  %2102 = extractelement <4 x half> %2100, i64 2, !dbg !20
  %2103 = extractelement <4 x half> %2100, i64 1, !dbg !20
  %2104 = extractelement <4 x half> %2100, i64 0, !dbg !20
  %2105 = insertelement <4 x float> poison, float %1639, i64 0, !dbg !19
  %2106 = insertelement <4 x float> %2105, float %1645, i64 1, !dbg !19
  %2107 = insertelement <4 x float> %2106, float %1640, i64 2, !dbg !19
  %2108 = insertelement <4 x float> %2107, float %1646, i64 3, !dbg !19
  %row.reciprocal133 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1618, !dbg !19
  %normalized134 = fmul <4 x float> %2108, %row.reciprocal133, !dbg !19
  %2109 = fptrunc <4 x float> %normalized134 to <4 x half>, !dbg !20
  %2110 = extractelement <4 x half> %2109, i64 3, !dbg !20
  %2111 = extractelement <4 x half> %2109, i64 2, !dbg !20
  %2112 = extractelement <4 x half> %2109, i64 1, !dbg !20
  %2113 = extractelement <4 x half> %2109, i64 0, !dbg !20
  %2114 = insertelement <4 x float> poison, float %1528, i64 0, !dbg !19
  %2115 = insertelement <4 x float> %2114, float %1534, i64 1, !dbg !19
  %2116 = insertelement <4 x float> %2115, float %1529, i64 2, !dbg !19
  %2117 = insertelement <4 x float> %2116, float %1535, i64 3, !dbg !19
  %normalized135 = fmul <4 x float> %2117, %row.reciprocal133, !dbg !19
  %2118 = fptrunc <4 x float> %normalized135 to <4 x half>, !dbg !20
  %2119 = extractelement <4 x half> %2118, i64 3, !dbg !20
  %2120 = extractelement <4 x half> %2118, i64 2, !dbg !20
  %2121 = extractelement <4 x half> %2118, i64 1, !dbg !20
  %2122 = extractelement <4 x half> %2118, i64 0, !dbg !20
  %2123 = shl nuw nsw i64 %15, 19, !dbg !21
  %address.thread1138 = call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !21, !range !34, !frisk.register_lifetimes !1
  %address.local1139 = zext nneg i32 %address.thread1138 to i64, !dbg !6
  %address.local1140 = add nsw i64 %16, %address.local1139, !dbg !21
  %address.local1141 = lshr i64 %address.local1139, 6, !dbg !6
  %address.local1142 = shl nuw nsw i64 %address.local1141, 4, !dbg !21
  %address.local1143 = add nsw i64 %address.local1140, %address.local1142, !dbg !21
  %address.local1144 = and i64 %address.local1139, 240, !dbg !21
  %address.local1145 = sub nsw i64 %address.local1143, %address.local1144, !dbg !21
  %address.local1146 = add nsw i64 %address.local1145, %17, !dbg !21
  %address.local1147 = shl nuw nsw i64 %address.local1146, 7, !dbg !21
  %address.local1148 = add nuw nsw i64 %2123, %address.local1147, !dbg !21
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1150 = lshr i64 %address.local1139, 4, !dbg !6
  %address.local1151 = and i64 %address.local1150, 3, !dbg !6
  %address.local1152 = shl nuw nsw i64 %address.local1151, 2, !dbg !21
  %address.local1153 = or disjoint i64 %address.local1152, 96, !dbg !21
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1153, !dbg !21
  store half %2122, ptr addrspace(1) %address.local1154, align 2, !dbg !21
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1156 = or disjoint i64 %address.local1152, 97, !dbg !21
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1156, !dbg !21
  store half %2121, ptr addrspace(1) %address.local1157, align 2, !dbg !21
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1159 = or disjoint i64 %address.local1152, 98, !dbg !21
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1158, i64 %address.local1159, !dbg !21
  store half %2120, ptr addrspace(1) %address.local1160, align 2, !dbg !21
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1162 = or disjoint i64 %address.local1152, 99, !dbg !21
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1162, !dbg !21
  store half %2119, ptr addrspace(1) %address.local1163, align 2, !dbg !21
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1165 = or disjoint i64 %address.local1152, 112, !dbg !21
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1164, i64 %address.local1165, !dbg !21
  store half %2113, ptr addrspace(1) %address.local1166, align 2, !dbg !21
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1168 = or disjoint i64 %address.local1152, 113, !dbg !21
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1168, !dbg !21
  store half %2112, ptr addrspace(1) %address.local1169, align 2, !dbg !21
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1171 = or disjoint i64 %address.local1152, 114, !dbg !21
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1170, i64 %address.local1171, !dbg !21
  store half %2111, ptr addrspace(1) %address.local1172, align 2, !dbg !21
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !21
  %address.local1174 = or disjoint i64 %address.local1152, 115, !dbg !21
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1174, !dbg !21
  store half %2110, ptr addrspace(1) %address.local1175, align 2, !dbg !21
  %address.local1176 = shl nsw i64 %address.local1146, 7, !dbg !21
  %address.local1177 = add i64 %address.local1176, 8192, !dbg !21
  %address.local1178 = add nuw nsw i64 %2123, %address.local1177, !dbg !21
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1153, !dbg !21
  store half %2104, ptr addrspace(1) %address.local1180, align 2, !dbg !21
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1156, !dbg !21
  store half %2103, ptr addrspace(1) %address.local1182, align 2, !dbg !21
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1159, !dbg !21
  store half %2102, ptr addrspace(1) %address.local1184, align 2, !dbg !21
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1162, !dbg !21
  store half %2101, ptr addrspace(1) %address.local1186, align 2, !dbg !21
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1187, i64 %address.local1165, !dbg !21
  store half %2095, ptr addrspace(1) %address.local1188, align 2, !dbg !21
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1168, !dbg !21
  store half %2094, ptr addrspace(1) %address.local1190, align 2, !dbg !21
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1171, !dbg !21
  store half %2093, ptr addrspace(1) %address.local1192, align 2, !dbg !21
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1178, !dbg !21
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1193, i64 %address.local1174, !dbg !21
  store half %2092, ptr addrspace(1) %address.local1194, align 2, !dbg !21
  br label %2124, !dbg !21

2124:                                             ; preds = %2086, %2085
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #5

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind memory(none) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!33}

!0 = !DIFile(filename: "test_input_spidD.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!6 = !DILocation(line: 68, column: 24, scope: !4)
!7 = !DILocation(line: 92, column: 20, scope: !4)
!8 = !DILocation(line: 69, column: 26, scope: !4)
!12 = !DILocation(line: 46, column: 17, scope: !4)
!14 = !DILocation(line: 66, column: 22, scope: !4)
!15 = !DILocation(line: 73, column: 19, scope: !4)
!16 = !DILocation(line: 56, column: 24, scope: !4)
!17 = !DILocation(line: 67, column: 22, scope: !4)
!18 = !DILocation(line: 93, column: 23, scope: !4)
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
