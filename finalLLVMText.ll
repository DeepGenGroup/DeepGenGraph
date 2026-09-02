; ModuleID = '3rd/deepgengraph/test/test_input.mlir'
source_filename = "3rd/deepgengraph/test/test_input.mlir"

@shm_5 = addrspace(3) global [64 x [1 x float]] undef, align 16
@shm_4 = addrspace(3) global [64 x [32 x float]] undef, align 16
@shm_3 = addrspace(3) global [64 x [32 x half]] undef, align 16
@shm_2 = addrspace(3) global [32 x [128 x half]] undef, align 16
@shm_1 = addrspace(3) global [64 x [128 x half]] undef, align 16
@shm_0 = addrspace(3) global [8192 x half] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !3 !reqd_work_group_size !6 {
  %5 = alloca [4 x <1 x float>], i64 1, align 4, !dbg !7
  %6 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !10
  %7 = sext i32 %6 to i64, !dbg !10
  %8 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !10
  %9 = sext i32 %8 to i64, !dbg !10
  %10 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10
  %11 = sext i32 %10 to i64, !dbg !10
  %12 = icmp slt i64 %7, 0, !dbg !11
  %13 = sub i64 -1, %7, !dbg !11
  %14 = select i1 %12, i64 %13, i64 %7, !dbg !11
  %15 = sdiv i64 %14, 64, !dbg !11
  %16 = sub i64 -1, %15, !dbg !11
  %17 = select i1 %12, i64 %16, i64 %15, !dbg !11
  %18 = add i64 %9, %17, !dbg !11
  %19 = srem i64 %18, 32, !dbg !11
  %20 = icmp slt i64 %19, 0, !dbg !11
  %21 = add i64 %19, 32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %19, !dbg !11
  %23 = mul nsw i64 %7, 64, !dbg !11
  %24 = icmp slt i64 %11, 0, !dbg !11
  %25 = sub i64 -1, %11, !dbg !11
  %26 = select i1 %24, i64 %25, i64 %11, !dbg !11
  %27 = sdiv i64 %26, 16, !dbg !11
  %28 = sub i64 -1, %27, !dbg !11
  %29 = select i1 %24, i64 %28, i64 %27, !dbg !11
  %30 = mul nsw i64 %29, -16, !dbg !11
  %31 = mul nsw i64 %17, -4096, !dbg !11
  %32 = mul nsw i64 %29, 4, !dbg !11
  br label %33, !dbg !11

33:                                               ; preds = %65, %4
  %34 = phi i64 [ %66, %65 ], [ 0, %4 ], !dbg !11
  %35 = icmp slt i64 %34, 4, !dbg !11
  br i1 %35, label %36, label %67, !dbg !11

36:                                               ; preds = %33
  %37 = mul nsw i64 %34, 16, !dbg !11
  %38 = add i64 %37, %23, !dbg !11
  %39 = add i64 %38, %11, !dbg !11
  %40 = add i64 %39, %30, !dbg !11
  %41 = add i64 %40, %31, !dbg !11
  %42 = add i64 %37, %11, !dbg !11
  %43 = add i64 %42, %30, !dbg !11
  br label %44, !dbg !11

44:                                               ; preds = %47, %36
  %45 = phi i64 [ %64, %47 ], [ 0, %36 ], !dbg !11
  %46 = icmp slt i64 %45, 8, !dbg !11
  br i1 %46, label %47, label %65, !dbg !11

47:                                               ; preds = %44
  %48 = mul nsw i64 %45, 16, !dbg !11
  %49 = add i64 %48, %32, !dbg !11
  %50 = sub i64 128, %49, !dbg !11
  %51 = insertelement <4 x i64> poison, i64 %50, i32 0, !dbg !11
  %52 = shufflevector <4 x i64> %51, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !11
  %53 = icmp sgt <4 x i64> %52, <i64 0, i64 1, i64 2, i64 3>, !dbg !11
  %54 = mul i64 %22, 524288, !dbg !11
  %55 = add i64 0, %54, !dbg !11
  %56 = mul i64 %41, 128, !dbg !11
  %57 = add i64 %55, %56, !dbg !11
  %58 = add i64 %57, %49, !dbg !11
  %59 = getelementptr half, ptr addrspace(1) %0, i64 %58, !dbg !11
  %60 = call <4 x half> @llvm.masked.load.v4f16.p1(ptr addrspace(1) %59, i32 2, <4 x i1> %53, <4 x half> zeroinitializer), !dbg !11
  %61 = mul i64 %43, 128, !dbg !11
  %62 = add i64 %61, %49, !dbg !11
  %63 = getelementptr half, ptr addrspace(3) @shm_0, i64 %62, !dbg !11
  call void @llvm.masked.store.v4f16.p3(<4 x half> %60, ptr addrspace(3) %63, i32 2, <4 x i1> %53), !dbg !11
  %64 = add i64 %45, 1, !dbg !11
  br label %44, !dbg !11

65:                                               ; preds = %44
  %66 = add i64 %34, 1, !dbg !11
  br label %33, !dbg !11

67:                                               ; preds = %33
  %68 = add i64 %11, %30, !dbg !12
  br label %69, !dbg !12

69:                                               ; preds = %245, %67
  %70 = phi i64 [ %246, %245 ], [ 0, %67 ], !dbg !12
  %71 = icmp slt i64 %70, 32, !dbg !12
  br i1 %71, label %72, label %247, !dbg !12

72:                                               ; preds = %69
  %73 = icmp slt i64 %70, 0, !dbg !12
  %74 = sub i64 -1, %70, !dbg !12
  %75 = select i1 %73, i64 %74, i64 %70, !dbg !12
  %76 = sdiv i64 %75, 4, !dbg !12
  %77 = sub i64 -1, %76, !dbg !12
  %78 = select i1 %73, i64 %77, i64 %76, !dbg !12
  %79 = mul nsw i64 %78, -4, !dbg !12
  %80 = mul nsw i64 %78, 12, !dbg !12
  %81 = add i64 %70, %80, !dbg !12
  %82 = add i64 %81, %32, !dbg !12
  br label %83, !dbg !12

83:                                               ; preds = %243, %72
  %84 = phi i64 [ %244, %243 ], [ 0, %72 ], !dbg !12
  %85 = icmp slt i64 %84, 4, !dbg !12
  br i1 %85, label %86, label %245, !dbg !12

86:                                               ; preds = %83
  %87 = mul nsw i64 %84, 16, !dbg !12
  %88 = add i64 %87, %11, !dbg !12
  %89 = add i64 %88, %30, !dbg !12
  br label %90, !dbg !12

90:                                               ; preds = %241, %86
  %91 = phi i64 [ %242, %241 ], [ 0, %86 ], !dbg !12
  %92 = icmp slt i64 %91, 8, !dbg !12
  br i1 %92, label %93, label %243, !dbg !12

93:                                               ; preds = %90
  %94 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !12
  %95 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !12
  %96 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !12
  %97 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !12
  %98 = mul nsw i64 %91, 16, !dbg !12
  %99 = add i64 %98, %70, !dbg !12
  %100 = add i64 %99, %32, !dbg !12
  %101 = add i64 %100, %79, !dbg !12
  %102 = mul i64 %89, 128, !dbg !12
  %103 = add i64 %102, %101, !dbg !12
  %104 = getelementptr half, ptr addrspace(3) @shm_0, i64 %103, !dbg !12
  %105 = load <1 x half>, ptr addrspace(3) %104, align 2, !dbg !12
  %106 = extractelement <1 x half> %105, i64 0, !dbg !12
  %107 = insertelement <4 x half> poison, half %106, i32 0, !dbg !12
  %108 = shufflevector <4 x half> %107, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %compat.splat.insert = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !12
  %compat.splat = shufflevector <4 x half> %compat.splat.insert, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %109 = fmul <4 x half> %108, %compat.splat, !dbg !12
  %110 = extractelement <4 x half> %109, i64 0, !dbg !12
  %111 = insertelement <1 x half> poison, half %110, i64 0, !dbg !12
  %112 = insertvalue [4 x <1 x half>] poison, <1 x half> %111, 0, !dbg !12
  %113 = extractelement <4 x half> %109, i64 1, !dbg !12
  %114 = insertelement <1 x half> poison, half %113, i64 0, !dbg !12
  %115 = insertvalue [4 x <1 x half>] %112, <1 x half> %114, 1, !dbg !12
  %116 = extractelement <4 x half> %109, i64 2, !dbg !12
  %117 = insertelement <1 x half> poison, half %116, i64 0, !dbg !12
  %118 = insertvalue [4 x <1 x half>] %115, <1 x half> %117, 2, !dbg !12
  %119 = extractelement <4 x half> %109, i64 3, !dbg !12
  %120 = insertelement <1 x half> poison, half %119, i64 0, !dbg !12
  %121 = insertvalue [4 x <1 x half>] %118, <1 x half> %120, 3, !dbg !12
  store [4 x <1 x half>] %121, ptr %94, align 2, !dbg !12
  br label %122, !dbg !12

122:                                              ; preds = %134, %93
  %123 = phi i64 [ %135, %134 ], [ 0, %93 ], !dbg !12
  %124 = icmp slt i64 %123, 4, !dbg !12
  br i1 %124, label %125, label %136, !dbg !12

125:                                              ; preds = %122
  %126 = add i64 %123, %68, !dbg !12
  %127 = icmp slt i64 %126, 64, !dbg !12
  br i1 %127, label %128, label %134, !dbg !12

128:                                              ; preds = %125
  %129 = getelementptr <1 x half>, ptr %94, i64 %123, !dbg !12
  %130 = load <1 x half>, ptr %129, align 2, !dbg !12
  %131 = mul i64 %126, 128, !dbg !12
  %132 = add i64 %131, %82, !dbg !12
  %133 = getelementptr half, ptr addrspace(3) @shm_1, i64 %132, !dbg !12
  store <1 x half> %130, ptr addrspace(3) %133, align 2, !dbg !12
  br label %134, !dbg !12

134:                                              ; preds = %128, %125
  %135 = add i64 %123, 1, !dbg !12
  br label %122, !dbg !12

136:                                              ; preds = %122
  %137 = add i64 %101, 16, !dbg !12
  %138 = add i64 %102, %137, !dbg !12
  %139 = getelementptr half, ptr addrspace(3) @shm_0, i64 %138, !dbg !12
  %140 = load <1 x half>, ptr addrspace(3) %139, align 2, !dbg !12
  %141 = extractelement <1 x half> %140, i64 0, !dbg !12
  %142 = insertelement <4 x half> poison, half %141, i32 0, !dbg !12
  %143 = shufflevector <4 x half> %142, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %compat.splat.insert1 = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !12
  %compat.splat2 = shufflevector <4 x half> %compat.splat.insert1, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %144 = fmul <4 x half> %143, %compat.splat2, !dbg !12
  %145 = extractelement <4 x half> %144, i64 0, !dbg !12
  %146 = insertelement <1 x half> poison, half %145, i64 0, !dbg !12
  %147 = insertvalue [4 x <1 x half>] poison, <1 x half> %146, 0, !dbg !12
  %148 = extractelement <4 x half> %144, i64 1, !dbg !12
  %149 = insertelement <1 x half> poison, half %148, i64 0, !dbg !12
  %150 = insertvalue [4 x <1 x half>] %147, <1 x half> %149, 1, !dbg !12
  %151 = extractelement <4 x half> %144, i64 2, !dbg !12
  %152 = insertelement <1 x half> poison, half %151, i64 0, !dbg !12
  %153 = insertvalue [4 x <1 x half>] %150, <1 x half> %152, 2, !dbg !12
  %154 = extractelement <4 x half> %144, i64 3, !dbg !12
  %155 = insertelement <1 x half> poison, half %154, i64 0, !dbg !12
  %156 = insertvalue [4 x <1 x half>] %153, <1 x half> %155, 3, !dbg !12
  store [4 x <1 x half>] %156, ptr %95, align 2, !dbg !12
  br label %157, !dbg !12

157:                                              ; preds = %169, %136
  %158 = phi i64 [ %170, %169 ], [ 0, %136 ], !dbg !12
  %159 = icmp slt i64 %158, 4, !dbg !12
  br i1 %159, label %160, label %171, !dbg !12

160:                                              ; preds = %157
  %161 = add i64 %158, %68, !dbg !12
  %162 = icmp slt i64 %161, 64, !dbg !12
  br i1 %162, label %163, label %169, !dbg !12

163:                                              ; preds = %160
  %164 = getelementptr <1 x half>, ptr %95, i64 %158, !dbg !12
  %165 = load <1 x half>, ptr %164, align 2, !dbg !12
  %166 = mul i64 %161, 128, !dbg !12
  %167 = add i64 %166, %82, !dbg !12
  %168 = getelementptr half, ptr addrspace(3) @shm_1, i64 %167, !dbg !12
  store <1 x half> %165, ptr addrspace(3) %168, align 2, !dbg !12
  br label %169, !dbg !12

169:                                              ; preds = %163, %160
  %170 = add i64 %158, 1, !dbg !12
  br label %157, !dbg !12

171:                                              ; preds = %157
  %172 = add i64 %101, 32, !dbg !12
  %173 = add i64 %102, %172, !dbg !12
  %174 = getelementptr half, ptr addrspace(3) @shm_0, i64 %173, !dbg !12
  %175 = load <1 x half>, ptr addrspace(3) %174, align 2, !dbg !12
  %176 = extractelement <1 x half> %175, i64 0, !dbg !12
  %177 = insertelement <4 x half> poison, half %176, i32 0, !dbg !12
  %178 = shufflevector <4 x half> %177, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %compat.splat.insert3 = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !12
  %compat.splat4 = shufflevector <4 x half> %compat.splat.insert3, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %179 = fmul <4 x half> %178, %compat.splat4, !dbg !12
  %180 = extractelement <4 x half> %179, i64 0, !dbg !12
  %181 = insertelement <1 x half> poison, half %180, i64 0, !dbg !12
  %182 = insertvalue [4 x <1 x half>] poison, <1 x half> %181, 0, !dbg !12
  %183 = extractelement <4 x half> %179, i64 1, !dbg !12
  %184 = insertelement <1 x half> poison, half %183, i64 0, !dbg !12
  %185 = insertvalue [4 x <1 x half>] %182, <1 x half> %184, 1, !dbg !12
  %186 = extractelement <4 x half> %179, i64 2, !dbg !12
  %187 = insertelement <1 x half> poison, half %186, i64 0, !dbg !12
  %188 = insertvalue [4 x <1 x half>] %185, <1 x half> %187, 2, !dbg !12
  %189 = extractelement <4 x half> %179, i64 3, !dbg !12
  %190 = insertelement <1 x half> poison, half %189, i64 0, !dbg !12
  %191 = insertvalue [4 x <1 x half>] %188, <1 x half> %190, 3, !dbg !12
  store [4 x <1 x half>] %191, ptr %96, align 2, !dbg !12
  br label %192, !dbg !12

192:                                              ; preds = %204, %171
  %193 = phi i64 [ %205, %204 ], [ 0, %171 ], !dbg !12
  %194 = icmp slt i64 %193, 4, !dbg !12
  br i1 %194, label %195, label %206, !dbg !12

195:                                              ; preds = %192
  %196 = add i64 %193, %68, !dbg !12
  %197 = icmp slt i64 %196, 64, !dbg !12
  br i1 %197, label %198, label %204, !dbg !12

198:                                              ; preds = %195
  %199 = getelementptr <1 x half>, ptr %96, i64 %193, !dbg !12
  %200 = load <1 x half>, ptr %199, align 2, !dbg !12
  %201 = mul i64 %196, 128, !dbg !12
  %202 = add i64 %201, %82, !dbg !12
  %203 = getelementptr half, ptr addrspace(3) @shm_1, i64 %202, !dbg !12
  store <1 x half> %200, ptr addrspace(3) %203, align 2, !dbg !12
  br label %204, !dbg !12

204:                                              ; preds = %198, %195
  %205 = add i64 %193, 1, !dbg !12
  br label %192, !dbg !12

206:                                              ; preds = %192
  %207 = add i64 %101, 48, !dbg !12
  %208 = add i64 %102, %207, !dbg !12
  %209 = getelementptr half, ptr addrspace(3) @shm_0, i64 %208, !dbg !12
  %210 = load <1 x half>, ptr addrspace(3) %209, align 2, !dbg !12
  %211 = extractelement <1 x half> %210, i64 0, !dbg !12
  %212 = insertelement <4 x half> poison, half %211, i32 0, !dbg !12
  %213 = shufflevector <4 x half> %212, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %compat.splat.insert5 = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !12
  %compat.splat6 = shufflevector <4 x half> %compat.splat.insert5, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %214 = fmul <4 x half> %213, %compat.splat6, !dbg !12
  %215 = extractelement <4 x half> %214, i64 0, !dbg !12
  %216 = insertelement <1 x half> poison, half %215, i64 0, !dbg !12
  %217 = insertvalue [4 x <1 x half>] poison, <1 x half> %216, 0, !dbg !12
  %218 = extractelement <4 x half> %214, i64 1, !dbg !12
  %219 = insertelement <1 x half> poison, half %218, i64 0, !dbg !12
  %220 = insertvalue [4 x <1 x half>] %217, <1 x half> %219, 1, !dbg !12
  %221 = extractelement <4 x half> %214, i64 2, !dbg !12
  %222 = insertelement <1 x half> poison, half %221, i64 0, !dbg !12
  %223 = insertvalue [4 x <1 x half>] %220, <1 x half> %222, 2, !dbg !12
  %224 = extractelement <4 x half> %214, i64 3, !dbg !12
  %225 = insertelement <1 x half> poison, half %224, i64 0, !dbg !12
  %226 = insertvalue [4 x <1 x half>] %223, <1 x half> %225, 3, !dbg !12
  store [4 x <1 x half>] %226, ptr %97, align 2, !dbg !12
  br label %227, !dbg !12

227:                                              ; preds = %239, %206
  %228 = phi i64 [ %240, %239 ], [ 0, %206 ], !dbg !12
  %229 = icmp slt i64 %228, 4, !dbg !12
  br i1 %229, label %230, label %241, !dbg !12

230:                                              ; preds = %227
  %231 = add i64 %228, %68, !dbg !12
  %232 = icmp slt i64 %231, 64, !dbg !12
  br i1 %232, label %233, label %239, !dbg !12

233:                                              ; preds = %230
  %234 = getelementptr <1 x half>, ptr %97, i64 %228, !dbg !12
  %235 = load <1 x half>, ptr %234, align 2, !dbg !12
  %236 = mul i64 %231, 128, !dbg !12
  %237 = add i64 %236, %82, !dbg !12
  %238 = getelementptr half, ptr addrspace(3) @shm_1, i64 %237, !dbg !12
  store <1 x half> %235, ptr addrspace(3) %238, align 2, !dbg !12
  br label %239, !dbg !12

239:                                              ; preds = %233, %230
  %240 = add i64 %228, 1, !dbg !12
  br label %227, !dbg !12

241:                                              ; preds = %227
  %242 = add i64 %91, 4, !dbg !12
  br label %90, !dbg !12

243:                                              ; preds = %90
  %244 = add i64 %84, 1, !dbg !12
  br label %83, !dbg !12

245:                                              ; preds = %83
  %246 = add i64 %70, 1, !dbg !12
  br label %69, !dbg !12

247:                                              ; preds = %69
  %248 = alloca float, i64 128, align 4, addrspace(5), !dbg !13
  br label %249, !dbg !16

249:                                              ; preds = %286, %247
  %250 = phi i64 [ %287, %286 ], [ 0, %247 ], !dbg !17
  %251 = icmp slt i64 %250, 4, !dbg !16
  br i1 %251, label %252, label %288, !dbg !16

252:                                              ; preds = %249
  br label %253, !dbg !16

253:                                              ; preds = %256, %252
  %254 = phi i64 [ %285, %256 ], [ 0, %252 ], !dbg !17
  %255 = icmp slt i64 %254, 32, !dbg !16
  br i1 %255, label %256, label %286, !dbg !16

256:                                              ; preds = %253
  %257 = sub i64 32, %254, !dbg !16
  %258 = insertelement <4 x i64> poison, i64 %257, i32 0, !dbg !16
  %259 = shufflevector <4 x i64> %258, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !16
  %260 = icmp sgt <4 x i64> %259, <i64 0, i64 1, i64 2, i64 3>, !dbg !16
  %261 = mul i64 %250, 32, !dbg !16
  %262 = add i64 %261, %254, !dbg !16
  %263 = getelementptr float, ptr addrspace(5) %248, i64 %262, !dbg !16
  call void @llvm.masked.store.v4f32.p5(<4 x float> zeroinitializer, ptr addrspace(5) %263, i32 4, <4 x i1> %260), !dbg !16
  %264 = add i64 %254, 4, !dbg !16
  %265 = sub i64 32, %264, !dbg !16
  %266 = insertelement <4 x i64> poison, i64 %265, i32 0, !dbg !16
  %267 = shufflevector <4 x i64> %266, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !16
  %268 = icmp sgt <4 x i64> %267, <i64 0, i64 1, i64 2, i64 3>, !dbg !16
  %269 = add i64 %261, %264, !dbg !16
  %270 = getelementptr float, ptr addrspace(5) %248, i64 %269, !dbg !16
  call void @llvm.masked.store.v4f32.p5(<4 x float> zeroinitializer, ptr addrspace(5) %270, i32 4, <4 x i1> %268), !dbg !16
  %271 = add i64 %254, 8, !dbg !16
  %272 = sub i64 32, %271, !dbg !16
  %273 = insertelement <4 x i64> poison, i64 %272, i32 0, !dbg !16
  %274 = shufflevector <4 x i64> %273, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !16
  %275 = icmp sgt <4 x i64> %274, <i64 0, i64 1, i64 2, i64 3>, !dbg !16
  %276 = add i64 %261, %271, !dbg !16
  %277 = getelementptr float, ptr addrspace(5) %248, i64 %276, !dbg !16
  call void @llvm.masked.store.v4f32.p5(<4 x float> zeroinitializer, ptr addrspace(5) %277, i32 4, <4 x i1> %275), !dbg !16
  %278 = add i64 %254, 12, !dbg !16
  %279 = sub i64 32, %278, !dbg !16
  %280 = insertelement <4 x i64> poison, i64 %279, i32 0, !dbg !16
  %281 = shufflevector <4 x i64> %280, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !16
  %282 = icmp sgt <4 x i64> %281, <i64 0, i64 1, i64 2, i64 3>, !dbg !16
  %283 = add i64 %261, %278, !dbg !16
  %284 = getelementptr float, ptr addrspace(5) %248, i64 %283, !dbg !16
  call void @llvm.masked.store.v4f32.p5(<4 x float> zeroinitializer, ptr addrspace(5) %284, i32 4, <4 x i1> %282), !dbg !16
  %285 = add i64 %254, 16, !dbg !16
  br label %253, !dbg !16

286:                                              ; preds = %253
  %287 = add i64 %250, 1, !dbg !16
  br label %249, !dbg !16

288:                                              ; preds = %249
  %289 = alloca float, i64 4, align 4, addrspace(5), !dbg !18
  store [4 x <1 x float>] zeroinitializer, ptr %5, align 4, !dbg !7
  br label %290, !dbg !7

290:                                              ; preds = %293, %288
  %291 = phi i64 [ %298, %293 ], [ 0, %288 ], !dbg !19
  %292 = icmp slt i64 %291, 4, !dbg !7
  br i1 %292, label %293, label %299, !dbg !7

293:                                              ; preds = %290
  %294 = getelementptr <1 x float>, ptr %5, i64 %291, !dbg !7
  %295 = load <1 x float>, ptr %294, align 4, !dbg !7
  %296 = add i64 %291, 0, !dbg !7
  %297 = getelementptr float, ptr addrspace(5) %289, i64 %296, !dbg !7
  store <1 x float> %295, ptr addrspace(5) %297, align 4, !dbg !7
  %298 = add i64 %291, 1, !dbg !7
  br label %290, !dbg !7

299:                                              ; preds = %290
  %300 = alloca half, i64 4, align 1, addrspace(5), !dbg !20
  %301 = alloca half, i64 4, align 1, addrspace(5), !dbg !20
  %302 = alloca float, i64 32, align 1, addrspace(5), !dbg !20
  %303 = alloca float, i64 4, align 1, addrspace(5), !dbg !20
  %304 = alloca float, i64 1, align 4, addrspace(5), !dbg !21
  %305 = alloca float, i64 4, align 1, addrspace(5), !dbg !22
  %306 = alloca half, i64 4, align 1, addrspace(5), !dbg !22
  %307 = alloca half, i64 4, align 1, addrspace(5), !dbg !23
  %308 = alloca half, i64 4, align 1, addrspace(5), !dbg !23
  %309 = alloca float, i64 128, align 1, addrspace(5), !dbg !23
  %310 = alloca float, i64 4, align 1, addrspace(5), !dbg !23
  %311 = mul nsw i64 %7, 2, !dbg !24
  %312 = add i64 %311, 1, !dbg !24
  %313 = mul nsw i64 %7, 256, !dbg !25
  %314 = mul nsw i64 %9, 128, !dbg !25
  %315 = add i64 %311, %314, !dbg !25
  %316 = srem i64 %11, 64, !dbg !21
  %317 = icmp slt i64 %316, 0, !dbg !21
  %318 = add i64 %316, 64, !dbg !21
  %319 = select i1 %317, i64 %318, i64 %316, !dbg !21
  %320 = icmp slt i64 %319, 0, !dbg !21
  %321 = sub i64 -1, %319, !dbg !21
  %322 = select i1 %320, i64 %321, i64 %319, !dbg !21
  %323 = sdiv i64 %322, 16, !dbg !21
  %324 = sub i64 -1, %323, !dbg !21
  %325 = select i1 %320, i64 %324, i64 %323, !dbg !21
  %326 = icmp eq i64 %325, 0, !dbg !21
  %327 = srem i64 %11, 16, !dbg !21
  %328 = icmp slt i64 %327, 0, !dbg !21
  %329 = add i64 %327, 16, !dbg !21
  %330 = select i1 %328, i64 %329, i64 %327, !dbg !21
  %331 = add i64 %330, 16, !dbg !21
  %332 = add i64 %330, 32, !dbg !21
  %333 = add i64 %330, 48, !dbg !21
  br label %334, !dbg !24

334:                                              ; preds = %1085, %299
  %335 = phi i64 [ %1086, %1085 ], [ 0, %299 ], !dbg !24
  %336 = icmp slt i64 %335, %312, !dbg !24
  br i1 %336, label %337, label %1087, !dbg !24

337:                                              ; preds = %334
  %338 = alloca [4 x <1 x float>], i64 1, align 4, !dbg !21
  %339 = alloca [4 x <1 x float>], i64 1, align 4, !dbg !21
  %340 = alloca [4 x <1 x float>], i64 1, align 4, !dbg !21
  %341 = alloca [4 x <1 x float>], i64 1, align 4, !dbg !21
  %342 = add i64 %335, %311, !dbg !26
  %343 = icmp slt i64 %342, 0, !dbg !26
  %344 = sub i64 -1, %342, !dbg !26
  %345 = select i1 %343, i64 %344, i64 %342, !dbg !26
  %346 = sdiv i64 %345, 128, !dbg !26
  %347 = sub i64 -1, %346, !dbg !26
  %348 = select i1 %343, i64 %347, i64 %346, !dbg !26
  %349 = add i64 %348, %9, !dbg !26
  %350 = srem i64 %349, 32, !dbg !26
  %351 = icmp slt i64 %350, 0, !dbg !26
  %352 = add i64 %350, 32, !dbg !26
  %353 = select i1 %351, i64 %352, i64 %350, !dbg !26
  %354 = mul nsw i64 %335, 32, !dbg !26
  %355 = mul nsw i64 %348, -4096, !dbg !26
  br label %356, !dbg !26

356:                                              ; preds = %424, %337
  %357 = phi i64 [ %425, %424 ], [ 0, %337 ], !dbg !26
  %358 = icmp slt i64 %357, 8, !dbg !26
  br i1 %358, label %359, label %426, !dbg !26

359:                                              ; preds = %356
  %360 = mul nsw i64 %357, 16, !dbg !26
  %361 = add i64 %354, %360, !dbg !26
  %362 = add i64 %361, %23, !dbg !26
  %363 = add i64 %362, %355, !dbg !26
  %364 = add i64 %363, %32, !dbg !26
  %365 = add i64 %360, %32, !dbg !26
  br label %366, !dbg !26

366:                                              ; preds = %422, %359
  %367 = phi i64 [ %423, %422 ], [ 0, %359 ], !dbg !26
  %368 = icmp slt i64 %367, 2, !dbg !26
  br i1 %368, label %369, label %424, !dbg !26

369:                                              ; preds = %366
  %370 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !26
  %371 = mul nsw i64 %367, 16, !dbg !26
  %372 = add i64 %371, %11, !dbg !26
  %373 = add i64 %372, %30, !dbg !26
  br label %374, !dbg !26

374:                                              ; preds = %393, %369
  %375 = phi i64 [ %394, %393 ], [ 0, %369 ], !dbg !26
  %376 = phi <4 x half> [ %392, %393 ], [ zeroinitializer, %369 ], !dbg !26
  %377 = icmp slt i64 %375, 4, !dbg !26
  br i1 %377, label %378, label %395, !dbg !26

378:                                              ; preds = %374
  %379 = add i64 %364, %375, !dbg !26
  %380 = icmp slt i64 %379, 4096, !dbg !26
  br i1 %380, label %381, label %390, !dbg !26

381:                                              ; preds = %378
  %382 = mul i64 %353, 524288, !dbg !26
  %383 = add i64 0, %382, !dbg !26
  %384 = mul i64 %379, 128, !dbg !26
  %385 = add i64 %383, %384, !dbg !26
  %386 = add i64 %385, %373, !dbg !26
  %387 = getelementptr half, ptr addrspace(1) %2, i64 %386, !dbg !26
  %388 = load half, ptr addrspace(1) %387, align 2, !dbg !26
  %389 = insertelement <4 x half> %376, half %388, i64 %375, !dbg !26
  br label %391, !dbg !26

390:                                              ; preds = %378
  br label %391, !dbg !26

391:                                              ; preds = %381, %390
  %392 = phi <4 x half> [ %376, %390 ], [ %389, %381 ], !dbg !26
  br label %393, !dbg !26

393:                                              ; preds = %391
  %394 = add i64 %375, 1, !dbg !26
  br label %374, !dbg !26

395:                                              ; preds = %374
  %396 = extractelement <4 x half> %376, i64 0, !dbg !26
  %397 = insertelement <1 x half> poison, half %396, i64 0, !dbg !26
  %398 = insertvalue [4 x <1 x half>] poison, <1 x half> %397, 0, !dbg !26
  %399 = extractelement <4 x half> %376, i64 1, !dbg !26
  %400 = insertelement <1 x half> poison, half %399, i64 0, !dbg !26
  %401 = insertvalue [4 x <1 x half>] %398, <1 x half> %400, 1, !dbg !26
  %402 = extractelement <4 x half> %376, i64 2, !dbg !26
  %403 = insertelement <1 x half> poison, half %402, i64 0, !dbg !26
  %404 = insertvalue [4 x <1 x half>] %401, <1 x half> %403, 2, !dbg !26
  %405 = extractelement <4 x half> %376, i64 3, !dbg !26
  %406 = insertelement <1 x half> poison, half %405, i64 0, !dbg !26
  %407 = insertvalue [4 x <1 x half>] %404, <1 x half> %406, 3, !dbg !26
  store [4 x <1 x half>] %407, ptr %370, align 2, !dbg !26
  br label %408, !dbg !26

408:                                              ; preds = %420, %395
  %409 = phi i64 [ %421, %420 ], [ 0, %395 ], !dbg !26
  %410 = icmp slt i64 %409, 4, !dbg !26
  br i1 %410, label %411, label %422, !dbg !26

411:                                              ; preds = %408
  %412 = add i64 %365, %409, !dbg !26
  %413 = icmp slt i64 %412, 128, !dbg !26
  br i1 %413, label %414, label %420, !dbg !26

414:                                              ; preds = %411
  %415 = getelementptr <1 x half>, ptr %370, i64 %409, !dbg !26
  %416 = load <1 x half>, ptr %415, align 2, !dbg !26
  %417 = mul i64 %412, 32, !dbg !26
  %418 = add i64 %417, %373, !dbg !26
  %419 = getelementptr half, ptr addrspace(3) @shm_0, i64 %418, !dbg !26
  store <1 x half> %416, ptr addrspace(3) %419, align 2, !dbg !26
  br label %420, !dbg !26

420:                                              ; preds = %414, %411
  %421 = add i64 %409, 1, !dbg !26
  br label %408, !dbg !26

422:                                              ; preds = %408
  %423 = add i64 %367, 1, !dbg !26
  br label %366, !dbg !26

424:                                              ; preds = %366
  %425 = add i64 %357, 1, !dbg !26
  br label %356, !dbg !26

426:                                              ; preds = %356
  %427 = add i64 %335, %313, !dbg !25
  %428 = icmp slt i64 %427, 0, !dbg !25
  %429 = sub i64 -1, %427, !dbg !25
  %430 = select i1 %428, i64 %429, i64 %427, !dbg !25
  %431 = sdiv i64 %430, 16384, !dbg !25
  %432 = sub i64 -1, %431, !dbg !25
  %433 = select i1 %428, i64 %432, i64 %431, !dbg !25
  %434 = add i64 %433, %9, !dbg !25
  %435 = srem i64 %434, 32, !dbg !25
  %436 = icmp slt i64 %435, 0, !dbg !25
  %437 = add i64 %435, 32, !dbg !25
  %438 = select i1 %436, i64 %437, i64 %435, !dbg !25
  %439 = icmp slt i64 %335, 0, !dbg !25
  %440 = sub i64 -1, %335, !dbg !25
  %441 = select i1 %439, i64 %440, i64 %335, !dbg !25
  %442 = sdiv i64 %441, 128, !dbg !25
  %443 = sub i64 -1, %442, !dbg !25
  %444 = select i1 %439, i64 %443, i64 %442, !dbg !25
  %445 = add i64 %444, %315, !dbg !25
  %446 = icmp slt i64 %445, 0, !dbg !25
  %447 = sub i64 -1, %445, !dbg !25
  %448 = select i1 %446, i64 %447, i64 %445, !dbg !25
  %449 = sdiv i64 %448, 128, !dbg !25
  %450 = sub i64 -1, %449, !dbg !25
  %451 = select i1 %446, i64 %450, i64 %449, !dbg !25
  %452 = mul nsw i64 %451, -128, !dbg !25
  %453 = mul nsw i64 %444, -4096, !dbg !25
  br label %454, !dbg !25

454:                                              ; preds = %527, %426
  %455 = phi i64 [ %528, %527 ], [ 0, %426 ], !dbg !25
  %456 = icmp slt i64 %455, 2, !dbg !25
  br i1 %456, label %457, label %529, !dbg !25

457:                                              ; preds = %454
  %458 = mul nsw i64 %455, 16, !dbg !25
  %459 = add i64 %458, %311, !dbg !25
  %460 = add i64 %459, %314, !dbg !25
  %461 = add i64 %460, %444, !dbg !25
  %462 = add i64 %461, %452, !dbg !25
  %463 = add i64 %462, %32, !dbg !25
  %464 = add i64 %458, %32, !dbg !25
  br label %465, !dbg !25

465:                                              ; preds = %525, %457
  %466 = phi i64 [ %526, %525 ], [ 0, %457 ], !dbg !25
  %467 = icmp slt i64 %466, 8, !dbg !25
  br i1 %467, label %468, label %527, !dbg !25

468:                                              ; preds = %465
  %469 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !25
  %470 = mul nsw i64 %466, 16, !dbg !25
  %471 = add i64 %354, %470, !dbg !25
  %472 = add i64 %471, %11, !dbg !25
  %473 = add i64 %472, %453, !dbg !25
  %474 = add i64 %473, %30, !dbg !25
  br label %475, !dbg !25

475:                                              ; preds = %494, %468
  %476 = phi i64 [ %495, %494 ], [ 0, %468 ], !dbg !25
  %477 = phi <4 x half> [ %493, %494 ], [ zeroinitializer, %468 ], !dbg !25
  %478 = icmp slt i64 %476, 4, !dbg !25
  br i1 %478, label %479, label %496, !dbg !25

479:                                              ; preds = %475
  %480 = add i64 %463, %476, !dbg !25
  %481 = icmp slt i64 %480, 128, !dbg !25
  br i1 %481, label %482, label %491, !dbg !25

482:                                              ; preds = %479
  %483 = mul i64 %438, 524288, !dbg !25
  %484 = add i64 0, %483, !dbg !25
  %485 = mul i64 %480, 4096, !dbg !25
  %486 = add i64 %484, %485, !dbg !25
  %487 = add i64 %486, %474, !dbg !25
  %488 = getelementptr half, ptr addrspace(1) %1, i64 %487, !dbg !25
  %489 = load half, ptr addrspace(1) %488, align 2, !dbg !25
  %490 = insertelement <4 x half> %477, half %489, i64 %476, !dbg !25
  br label %492, !dbg !25

491:                                              ; preds = %479
  br label %492, !dbg !25

492:                                              ; preds = %482, %491
  %493 = phi <4 x half> [ %477, %491 ], [ %490, %482 ], !dbg !25
  br label %494, !dbg !25

494:                                              ; preds = %492
  %495 = add i64 %476, 1, !dbg !25
  br label %475, !dbg !25

496:                                              ; preds = %475
  %497 = add i64 %470, %11, !dbg !25
  %498 = add i64 %497, %30, !dbg !25
  %499 = extractelement <4 x half> %477, i64 0, !dbg !25
  %500 = insertelement <1 x half> poison, half %499, i64 0, !dbg !25
  %501 = insertvalue [4 x <1 x half>] poison, <1 x half> %500, 0, !dbg !25
  %502 = extractelement <4 x half> %477, i64 1, !dbg !25
  %503 = insertelement <1 x half> poison, half %502, i64 0, !dbg !25
  %504 = insertvalue [4 x <1 x half>] %501, <1 x half> %503, 1, !dbg !25
  %505 = extractelement <4 x half> %477, i64 2, !dbg !25
  %506 = insertelement <1 x half> poison, half %505, i64 0, !dbg !25
  %507 = insertvalue [4 x <1 x half>] %504, <1 x half> %506, 2, !dbg !25
  %508 = extractelement <4 x half> %477, i64 3, !dbg !25
  %509 = insertelement <1 x half> poison, half %508, i64 0, !dbg !25
  %510 = insertvalue [4 x <1 x half>] %507, <1 x half> %509, 3, !dbg !25
  store [4 x <1 x half>] %510, ptr %469, align 2, !dbg !25
  br label %511, !dbg !25

511:                                              ; preds = %523, %496
  %512 = phi i64 [ %524, %523 ], [ 0, %496 ], !dbg !25
  %513 = icmp slt i64 %512, 4, !dbg !25
  br i1 %513, label %514, label %525, !dbg !25

514:                                              ; preds = %511
  %515 = add i64 %464, %512, !dbg !25
  %516 = icmp slt i64 %515, 32, !dbg !25
  br i1 %516, label %517, label %523, !dbg !25

517:                                              ; preds = %514
  %518 = getelementptr <1 x half>, ptr %469, i64 %512, !dbg !25
  %519 = load <1 x half>, ptr %518, align 2, !dbg !25
  %520 = mul i64 %515, 128, !dbg !25
  %521 = add i64 %520, %498, !dbg !25
  %522 = getelementptr half, ptr addrspace(3) @shm_2, i64 %521, !dbg !25
  store <1 x half> %519, ptr addrspace(3) %522, align 2, !dbg !25
  br label %523, !dbg !25

523:                                              ; preds = %517, %514
  %524 = add i64 %512, 1, !dbg !25
  br label %511, !dbg !25

525:                                              ; preds = %511
  %526 = add i64 %466, 1, !dbg !25
  br label %465, !dbg !25

527:                                              ; preds = %465
  %528 = add i64 %455, 1, !dbg !25
  br label %454, !dbg !25

529:                                              ; preds = %454
  br label %530, !dbg !20

530:                                              ; preds = %626, %529
  %531 = phi i64 [ %627, %626 ], [ 0, %529 ], !dbg !20
  %532 = icmp slt i64 %531, 4, !dbg !20
  br i1 %532, label %533, label %628, !dbg !20

533:                                              ; preds = %530
  %534 = mul nsw i64 %531, 64, !dbg !20
  %535 = add i64 %534, %11, !dbg !20
  %536 = add i64 %535, %30, !dbg !20
  br label %537, !dbg !20

537:                                              ; preds = %624, %533
  %538 = phi i64 [ %625, %624 ], [ 0, %533 ], !dbg !20
  %539 = icmp slt i64 %538, 2, !dbg !20
  br i1 %539, label %540, label %626, !dbg !20

540:                                              ; preds = %537
  %541 = getelementptr float, ptr addrspace(5) %303, i64 0, !dbg !20
  store <4 x float> zeroinitializer, ptr addrspace(5) %541, align 4, !dbg !20
  %542 = mul nsw i64 %538, 16, !dbg !20
  %543 = add i64 %542, %11, !dbg !20
  %544 = add i64 %543, %30, !dbg !20
  br label %545, !dbg !20

545:                                              ; preds = %601, %540
  %546 = phi i64 [ %607, %601 ], [ 0, %540 ], !dbg !20
  %547 = icmp slt i64 %546, 8, !dbg !20
  br i1 %547, label %548, label %608, !dbg !20

548:                                              ; preds = %545
  %549 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !20
  %550 = mul nsw i64 %546, 16, !dbg !20
  %551 = add i64 %550, %32, !dbg !20
  %552 = sub i64 128, %551, !dbg !20
  %553 = insertelement <4 x i64> poison, i64 %552, i32 0, !dbg !20
  %554 = shufflevector <4 x i64> %553, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !20
  %555 = icmp sgt <4 x i64> %554, <i64 0, i64 1, i64 2, i64 3>, !dbg !20
  %556 = mul i64 %536, 128, !dbg !20
  %557 = add i64 %556, %551, !dbg !20
  %558 = getelementptr half, ptr addrspace(3) @shm_1, i64 %557, !dbg !20
  %559 = call <4 x half> @llvm.masked.load.v4f16.p3(ptr addrspace(3) %558, i32 2, <4 x i1> %555, <4 x half> zeroinitializer), !dbg !20
  %560 = getelementptr half, ptr addrspace(5) %300, i64 0, !dbg !20
  store <4 x half> %559, ptr addrspace(5) %560, align 2, !dbg !20
  br label %561, !dbg !20

561:                                              ; preds = %577, %548
  %562 = phi i64 [ %578, %577 ], [ 0, %548 ], !dbg !20
  %563 = phi <4 x half> [ %576, %577 ], [ zeroinitializer, %548 ], !dbg !20
  %564 = icmp slt i64 %562, 4, !dbg !20
  br i1 %564, label %565, label %579, !dbg !20

565:                                              ; preds = %561
  %566 = add i64 %551, %562, !dbg !20
  %567 = icmp slt i64 %566, 128, !dbg !20
  br i1 %567, label %568, label %574, !dbg !20

568:                                              ; preds = %565
  %569 = mul i64 %566, 32, !dbg !20
  %570 = add i64 %569, %544, !dbg !20
  %571 = getelementptr half, ptr addrspace(3) @shm_0, i64 %570, !dbg !20
  %572 = load half, ptr addrspace(3) %571, align 2, !dbg !20
  %573 = insertelement <4 x half> %563, half %572, i64 %562, !dbg !20
  br label %575, !dbg !20

574:                                              ; preds = %565
  br label %575, !dbg !20

575:                                              ; preds = %568, %574
  %576 = phi <4 x half> [ %563, %574 ], [ %573, %568 ], !dbg !20
  br label %577, !dbg !20

577:                                              ; preds = %575
  %578 = add i64 %562, 1, !dbg !20
  br label %561, !dbg !20

579:                                              ; preds = %561
  %580 = extractelement <4 x half> %563, i64 0, !dbg !20
  %581 = insertelement <1 x half> poison, half %580, i64 0, !dbg !20
  %582 = insertvalue [4 x <1 x half>] poison, <1 x half> %581, 0, !dbg !20
  %583 = extractelement <4 x half> %563, i64 1, !dbg !20
  %584 = insertelement <1 x half> poison, half %583, i64 0, !dbg !20
  %585 = insertvalue [4 x <1 x half>] %582, <1 x half> %584, 1, !dbg !20
  %586 = extractelement <4 x half> %563, i64 2, !dbg !20
  %587 = insertelement <1 x half> poison, half %586, i64 0, !dbg !20
  %588 = insertvalue [4 x <1 x half>] %585, <1 x half> %587, 2, !dbg !20
  %589 = extractelement <4 x half> %563, i64 3, !dbg !20
  %590 = insertelement <1 x half> poison, half %589, i64 0, !dbg !20
  %591 = insertvalue [4 x <1 x half>] %588, <1 x half> %590, 3, !dbg !20
  store [4 x <1 x half>] %591, ptr %549, align 2, !dbg !20
  br label %592, !dbg !20

592:                                              ; preds = %595, %579
  %593 = phi i64 [ %600, %595 ], [ 0, %579 ], !dbg !20
  %594 = icmp slt i64 %593, 4, !dbg !20
  br i1 %594, label %595, label %601, !dbg !20

595:                                              ; preds = %592
  %596 = getelementptr <1 x half>, ptr %549, i64 %593, !dbg !20
  %597 = load <1 x half>, ptr %596, align 2, !dbg !20
  %598 = add i64 %593, 0, !dbg !20
  %599 = getelementptr half, ptr addrspace(5) %301, i64 %598, !dbg !20
  store <1 x half> %597, ptr addrspace(5) %599, align 2, !dbg !20
  %600 = add i64 %593, 1, !dbg !20
  br label %592, !dbg !20

601:                                              ; preds = %592
  %602 = load <4 x half>, ptr addrspace(5) %560, align 2, !dbg !20
  %603 = getelementptr half, ptr addrspace(5) %301, i64 0, !dbg !20
  %604 = load <4 x half>, ptr addrspace(5) %603, align 2, !dbg !20
  %605 = load <4 x float>, ptr addrspace(5) %541, align 4, !dbg !20
  %606 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %602, <4 x half> %604, <4 x float> %605), !dbg !20
  store <4 x float> %606, ptr addrspace(5) %541, align 4, !dbg !20
  %607 = add i64 %546, 1, !dbg !20
  br label %545, !dbg !20

608:                                              ; preds = %545
  %609 = mul nsw i64 %538, 4, !dbg !20
  br label %610, !dbg !20

610:                                              ; preds = %613, %608
  %611 = phi i64 [ %623, %613 ], [ 0, %608 ], !dbg !20
  %612 = icmp slt i64 %611, 4, !dbg !20
  br i1 %612, label %613, label %624, !dbg !20

613:                                              ; preds = %610
  %614 = load <4 x float>, ptr addrspace(5) %541, align 4, !dbg !20
  %615 = add i64 %609, %611, !dbg !20
  %616 = sub i64 8, %615, !dbg !20
  %617 = insertelement <4 x i64> poison, i64 %616, i32 0, !dbg !20
  %618 = shufflevector <4 x i64> %617, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !20
  %619 = icmp sgt <4 x i64> %618, <i64 0, i64 1, i64 2, i64 3>, !dbg !20
  %620 = mul i64 %531, 8, !dbg !20
  %621 = add i64 %620, %615, !dbg !20
  %622 = getelementptr float, ptr addrspace(5) %302, i64 %621, !dbg !20
  call void @llvm.masked.store.v4f32.p5(<4 x float> %614, ptr addrspace(5) %622, i32 4, <4 x i1> %619), !dbg !20
  %623 = add i64 %611, 1, !dbg !20
  br label %610, !dbg !20

624:                                              ; preds = %610
  %625 = add i64 %538, 1, !dbg !20
  br label %537, !dbg !20

626:                                              ; preds = %537
  %627 = add i64 %531, 1, !dbg !20
  br label %530, !dbg !20

628:                                              ; preds = %530
  br label %629, !dbg !27

629:                                              ; preds = %683, %628
  %630 = phi i64 [ %684, %683 ], [ 0, %628 ], !dbg !27
  %631 = icmp slt i64 %630, 4, !dbg !27
  br i1 %631, label %632, label %685, !dbg !27

632:                                              ; preds = %629
  br label %633, !dbg !27

633:                                              ; preds = %681, %632
  %634 = phi i64 [ %682, %681 ], [ 0, %632 ], !dbg !27
  %635 = icmp slt i64 %634, 2, !dbg !27
  br i1 %635, label %636, label %683, !dbg !27

636:                                              ; preds = %633
  %637 = mul nsw i64 %634, 16, !dbg !27
  %638 = add i64 %354, %637, !dbg !27
  br label %639, !dbg !27

639:                                              ; preds = %679, %636
  %640 = phi i64 [ %680, %679 ], [ 0, %636 ], !dbg !27
  %641 = icmp slt i64 %640, 8, !dbg !27
  br i1 %641, label %642, label %681, !dbg !27

642:                                              ; preds = %639
  %643 = mul nsw i64 %640, 4, !dbg !27
  %644 = add i64 %638, %643, !dbg !27
  %645 = icmp slt i64 %640, 0, !dbg !27
  %646 = sub i64 -1, %640, !dbg !27
  %647 = select i1 %645, i64 %646, i64 %640, !dbg !27
  %648 = sdiv i64 %647, 4, !dbg !27
  %649 = sub i64 -1, %648, !dbg !27
  %650 = select i1 %645, i64 %649, i64 %648, !dbg !27
  %651 = mul nsw i64 %650, -16, !dbg !27
  %652 = add i64 %644, %651, !dbg !27
  %653 = add i64 %652, %29, !dbg !27
  %654 = add i64 %637, %643, !dbg !28
  %655 = add i64 %654, %651, !dbg !28
  %656 = add i64 %655, %29, !dbg !28
  br label %657, !dbg !27

657:                                              ; preds = %660, %642
  %658 = phi i64 [ %678, %660 ], [ 0, %642 ], !dbg !27
  %659 = icmp slt i64 %658, 4, !dbg !27
  br i1 %659, label %660, label %679, !dbg !27

660:                                              ; preds = %657
  %661 = mul nsw i64 %658, 16, !dbg !27
  %662 = add i64 %661, %11, !dbg !27
  %663 = add i64 %662, %23, !dbg !27
  %664 = add i64 %663, %30, !dbg !27
  %665 = add i64 %664, 1, !dbg !29
  %666 = icmp ule i64 %665, %653, !dbg !30
  %667 = select i1 %666, float 0xFFF0000000000000, float 0.000000e+00, !dbg !27
  %668 = mul i64 %630, 8, !dbg !31
  %669 = add i64 %668, %640, !dbg !31
  %670 = getelementptr float, ptr addrspace(5) %302, i64 %669, !dbg !31
  %671 = load float, ptr addrspace(5) %670, align 4, !dbg !31
  %672 = fadd float %671, %667, !dbg !31
  %673 = call float @__ocml_exp2_f32(float %672), !dbg !28
  %674 = add i64 %662, %30, !dbg !28
  %675 = mul i64 %674, 32, !dbg !28
  %676 = add i64 %675, %656, !dbg !28
  %677 = getelementptr float, ptr addrspace(3) @shm_4, i64 %676, !dbg !28
  store float %673, ptr addrspace(3) %677, align 4, !dbg !28
  %678 = add i64 %658, 1, !dbg !27
  br label %657, !dbg !27

679:                                              ; preds = %657
  %680 = add i64 %640, 1, !dbg !27
  br label %639, !dbg !27

681:                                              ; preds = %639
  %682 = add i64 %634, 1, !dbg !27
  br label %633, !dbg !27

683:                                              ; preds = %633
  %684 = add i64 %630, 1, !dbg !27
  br label %629, !dbg !27

685:                                              ; preds = %629
  br label %686, !dbg !21

686:                                              ; preds = %749, %685
  %687 = phi i64 [ %750, %749 ], [ 0, %685 ], !dbg !21
  %688 = icmp slt i64 %687, 4, !dbg !21
  br i1 %688, label %689, label %751, !dbg !21

689:                                              ; preds = %686
  store float 0.000000e+00, ptr addrspace(5) %304, align 4, !dbg !21
  %690 = mul nsw i64 %687, 16, !dbg !21
  %691 = add i64 %690, %11, !dbg !21
  %692 = add i64 %691, %30, !dbg !21
  br label %693, !dbg !21

693:                                              ; preds = %696, %689
  %694 = phi i64 [ %723, %696 ], [ 0, %689 ], !dbg !21
  %695 = icmp slt i64 %694, 8, !dbg !21
  br i1 %695, label %696, label %724, !dbg !21

696:                                              ; preds = %693
  %697 = load float, ptr addrspace(5) %304, align 4, !dbg !21
  %698 = mul nsw i64 %694, 4, !dbg !21
  %699 = add i64 %698, %29, !dbg !21
  %700 = mul i64 %692, 32, !dbg !21
  %701 = add i64 %700, %699, !dbg !21
  %702 = getelementptr float, ptr addrspace(3) @shm_4, i64 %701, !dbg !21
  %703 = load float, ptr addrspace(3) %702, align 4, !dbg !21
  %704 = fadd float %697, %703, !dbg !21
  store float %704, ptr addrspace(5) %304, align 4, !dbg !21
  %705 = load float, ptr addrspace(5) %304, align 4, !dbg !21
  %706 = add i64 %699, 4, !dbg !21
  %707 = add i64 %700, %706, !dbg !21
  %708 = getelementptr float, ptr addrspace(3) @shm_4, i64 %707, !dbg !21
  %709 = load float, ptr addrspace(3) %708, align 4, !dbg !21
  %710 = fadd float %705, %709, !dbg !21
  store float %710, ptr addrspace(5) %304, align 4, !dbg !21
  %711 = load float, ptr addrspace(5) %304, align 4, !dbg !21
  %712 = add i64 %699, 8, !dbg !21
  %713 = add i64 %700, %712, !dbg !21
  %714 = getelementptr float, ptr addrspace(3) @shm_4, i64 %713, !dbg !21
  %715 = load float, ptr addrspace(3) %714, align 4, !dbg !21
  %716 = fadd float %711, %715, !dbg !21
  store float %716, ptr addrspace(5) %304, align 4, !dbg !21
  %717 = load float, ptr addrspace(5) %304, align 4, !dbg !21
  %718 = add i64 %699, 12, !dbg !21
  %719 = add i64 %700, %718, !dbg !21
  %720 = getelementptr float, ptr addrspace(3) @shm_4, i64 %719, !dbg !21
  %721 = load float, ptr addrspace(3) %720, align 4, !dbg !21
  %722 = fadd float %717, %721, !dbg !21
  store float %722, ptr addrspace(5) %304, align 4, !dbg !21
  %723 = add i64 %694, 4, !dbg !21
  br label %693, !dbg !21

724:                                              ; preds = %693
  %725 = load float, ptr addrspace(5) %304, align 4, !dbg !21
  %726 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !21
  %727 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %726), !dbg !21
  %728 = add i32 %727, 64, !dbg !21
  %729 = xor i32 %727, 16, !dbg !21
  %730 = and i32 %728, -64, !dbg !21
  %731 = icmp slt i32 %729, %730, !dbg !21
  %732 = select i1 %731, i32 %729, i32 %727, !dbg !21
  %733 = shl i32 %732, 2, !dbg !21
  %734 = bitcast float %725 to i32, !dbg !21
  %735 = call i32 @llvm.amdgcn.ds.bpermute(i32 %733, i32 %734), !dbg !21
  %736 = bitcast i32 %735 to float, !dbg !21
  %737 = fadd float %725, %736, !dbg !21
  %738 = xor i32 %727, 32, !dbg !21
  %739 = icmp slt i32 %738, %730, !dbg !21
  %740 = select i1 %739, i32 %738, i32 %727, !dbg !21
  %741 = shl i32 %740, 2, !dbg !21
  %742 = bitcast float %737 to i32, !dbg !21
  %743 = call i32 @llvm.amdgcn.ds.bpermute(i32 %741, i32 %742), !dbg !21
  %744 = bitcast i32 %743 to float, !dbg !21
  %745 = fadd float %737, %744, !dbg !21
  br i1 %326, label %746, label %749, !dbg !21

746:                                              ; preds = %724
  %747 = add i64 %692, 0, !dbg !21
  %748 = getelementptr float, ptr addrspace(3) @shm_5, i64 %747, !dbg !21
  store float %745, ptr addrspace(3) %748, align 4, !dbg !21
  br label %749, !dbg !21

749:                                              ; preds = %746, %724
  %750 = add i64 %687, 1, !dbg !21
  br label %686, !dbg !21

751:                                              ; preds = %686
  fence syncscope("workgroup") release, !dbg !21
  call void @llvm.amdgcn.s.barrier(), !dbg !21
  fence syncscope("workgroup") acquire, !dbg !21
  %752 = add i64 %330, 0, !dbg !21
  %753 = getelementptr float, ptr addrspace(3) @shm_5, i64 %752, !dbg !21
  %754 = load <1 x float>, ptr addrspace(3) %753, align 4, !dbg !21
  %755 = extractelement <1 x float> %754, i64 0, !dbg !21
  %756 = insertelement <4 x float> poison, float %755, i32 0, !dbg !21
  %757 = shufflevector <4 x float> %756, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  br label %758, !dbg !21

758:                                              ; preds = %762, %751
  %759 = phi i64 [ %767, %762 ], [ 0, %751 ], !dbg !21
  %760 = phi <4 x float> [ %766, %762 ], [ zeroinitializer, %751 ], !dbg !21
  %761 = icmp slt i64 %759, 4, !dbg !21
  br i1 %761, label %762, label %768, !dbg !21

762:                                              ; preds = %758
  %763 = add i64 %759, 0, !dbg !21
  %764 = getelementptr float, ptr addrspace(5) %289, i64 %763, !dbg !21
  %765 = load float, ptr addrspace(5) %764, align 4, !dbg !21
  %766 = insertelement <4 x float> %760, float %765, i64 %759, !dbg !21
  %767 = add i64 %759, 1, !dbg !21
  br label %758, !dbg !21

768:                                              ; preds = %758
  %769 = fadd <4 x float> %760, %757, !dbg !21
  %770 = extractelement <4 x float> %769, i64 0, !dbg !21
  %771 = insertelement <1 x float> poison, float %770, i64 0, !dbg !21
  %772 = insertvalue [4 x <1 x float>] poison, <1 x float> %771, 0, !dbg !21
  %773 = extractelement <4 x float> %769, i64 1, !dbg !21
  %774 = insertelement <1 x float> poison, float %773, i64 0, !dbg !21
  %775 = insertvalue [4 x <1 x float>] %772, <1 x float> %774, 1, !dbg !21
  %776 = extractelement <4 x float> %769, i64 2, !dbg !21
  %777 = insertelement <1 x float> poison, float %776, i64 0, !dbg !21
  %778 = insertvalue [4 x <1 x float>] %775, <1 x float> %777, 2, !dbg !21
  %779 = extractelement <4 x float> %769, i64 3, !dbg !21
  %780 = insertelement <1 x float> poison, float %779, i64 0, !dbg !21
  %781 = insertvalue [4 x <1 x float>] %778, <1 x float> %780, 3, !dbg !21
  store [4 x <1 x float>] %781, ptr %338, align 4, !dbg !21
  br label %782, !dbg !21

782:                                              ; preds = %785, %768
  %783 = phi i64 [ %790, %785 ], [ 0, %768 ], !dbg !21
  %784 = icmp slt i64 %783, 4, !dbg !21
  br i1 %784, label %785, label %791, !dbg !21

785:                                              ; preds = %782
  %786 = getelementptr <1 x float>, ptr %338, i64 %783, !dbg !21
  %787 = load <1 x float>, ptr %786, align 4, !dbg !21
  %788 = add i64 %783, 0, !dbg !21
  %789 = getelementptr float, ptr addrspace(5) %289, i64 %788, !dbg !21
  store <1 x float> %787, ptr addrspace(5) %789, align 4, !dbg !21
  %790 = add i64 %783, 1, !dbg !21
  br label %782, !dbg !21

791:                                              ; preds = %782
  %792 = add i64 %331, 0, !dbg !21
  %793 = getelementptr float, ptr addrspace(3) @shm_5, i64 %792, !dbg !21
  %794 = load <1 x float>, ptr addrspace(3) %793, align 4, !dbg !21
  %795 = extractelement <1 x float> %794, i64 0, !dbg !21
  %796 = insertelement <4 x float> poison, float %795, i32 0, !dbg !21
  %797 = shufflevector <4 x float> %796, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  br label %798, !dbg !21

798:                                              ; preds = %802, %791
  %799 = phi i64 [ %807, %802 ], [ 0, %791 ], !dbg !21
  %800 = phi <4 x float> [ %806, %802 ], [ zeroinitializer, %791 ], !dbg !21
  %801 = icmp slt i64 %799, 4, !dbg !21
  br i1 %801, label %802, label %808, !dbg !21

802:                                              ; preds = %798
  %803 = add i64 %799, 0, !dbg !21
  %804 = getelementptr float, ptr addrspace(5) %289, i64 %803, !dbg !21
  %805 = load float, ptr addrspace(5) %804, align 4, !dbg !21
  %806 = insertelement <4 x float> %800, float %805, i64 %799, !dbg !21
  %807 = add i64 %799, 1, !dbg !21
  br label %798, !dbg !21

808:                                              ; preds = %798
  %809 = fadd <4 x float> %800, %797, !dbg !21
  %810 = extractelement <4 x float> %809, i64 0, !dbg !21
  %811 = insertelement <1 x float> poison, float %810, i64 0, !dbg !21
  %812 = insertvalue [4 x <1 x float>] poison, <1 x float> %811, 0, !dbg !21
  %813 = extractelement <4 x float> %809, i64 1, !dbg !21
  %814 = insertelement <1 x float> poison, float %813, i64 0, !dbg !21
  %815 = insertvalue [4 x <1 x float>] %812, <1 x float> %814, 1, !dbg !21
  %816 = extractelement <4 x float> %809, i64 2, !dbg !21
  %817 = insertelement <1 x float> poison, float %816, i64 0, !dbg !21
  %818 = insertvalue [4 x <1 x float>] %815, <1 x float> %817, 2, !dbg !21
  %819 = extractelement <4 x float> %809, i64 3, !dbg !21
  %820 = insertelement <1 x float> poison, float %819, i64 0, !dbg !21
  %821 = insertvalue [4 x <1 x float>] %818, <1 x float> %820, 3, !dbg !21
  store [4 x <1 x float>] %821, ptr %339, align 4, !dbg !21
  br label %822, !dbg !21

822:                                              ; preds = %825, %808
  %823 = phi i64 [ %830, %825 ], [ 0, %808 ], !dbg !21
  %824 = icmp slt i64 %823, 4, !dbg !21
  br i1 %824, label %825, label %831, !dbg !21

825:                                              ; preds = %822
  %826 = getelementptr <1 x float>, ptr %339, i64 %823, !dbg !21
  %827 = load <1 x float>, ptr %826, align 4, !dbg !21
  %828 = add i64 %823, 0, !dbg !21
  %829 = getelementptr float, ptr addrspace(5) %289, i64 %828, !dbg !21
  store <1 x float> %827, ptr addrspace(5) %829, align 4, !dbg !21
  %830 = add i64 %823, 1, !dbg !21
  br label %822, !dbg !21

831:                                              ; preds = %822
  %832 = add i64 %332, 0, !dbg !21
  %833 = getelementptr float, ptr addrspace(3) @shm_5, i64 %832, !dbg !21
  %834 = load <1 x float>, ptr addrspace(3) %833, align 4, !dbg !21
  %835 = extractelement <1 x float> %834, i64 0, !dbg !21
  %836 = insertelement <4 x float> poison, float %835, i32 0, !dbg !21
  %837 = shufflevector <4 x float> %836, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  br label %838, !dbg !21

838:                                              ; preds = %842, %831
  %839 = phi i64 [ %847, %842 ], [ 0, %831 ], !dbg !21
  %840 = phi <4 x float> [ %846, %842 ], [ zeroinitializer, %831 ], !dbg !21
  %841 = icmp slt i64 %839, 4, !dbg !21
  br i1 %841, label %842, label %848, !dbg !21

842:                                              ; preds = %838
  %843 = add i64 %839, 0, !dbg !21
  %844 = getelementptr float, ptr addrspace(5) %289, i64 %843, !dbg !21
  %845 = load float, ptr addrspace(5) %844, align 4, !dbg !21
  %846 = insertelement <4 x float> %840, float %845, i64 %839, !dbg !21
  %847 = add i64 %839, 1, !dbg !21
  br label %838, !dbg !21

848:                                              ; preds = %838
  %849 = fadd <4 x float> %840, %837, !dbg !21
  %850 = extractelement <4 x float> %849, i64 0, !dbg !21
  %851 = insertelement <1 x float> poison, float %850, i64 0, !dbg !21
  %852 = insertvalue [4 x <1 x float>] poison, <1 x float> %851, 0, !dbg !21
  %853 = extractelement <4 x float> %849, i64 1, !dbg !21
  %854 = insertelement <1 x float> poison, float %853, i64 0, !dbg !21
  %855 = insertvalue [4 x <1 x float>] %852, <1 x float> %854, 1, !dbg !21
  %856 = extractelement <4 x float> %849, i64 2, !dbg !21
  %857 = insertelement <1 x float> poison, float %856, i64 0, !dbg !21
  %858 = insertvalue [4 x <1 x float>] %855, <1 x float> %857, 2, !dbg !21
  %859 = extractelement <4 x float> %849, i64 3, !dbg !21
  %860 = insertelement <1 x float> poison, float %859, i64 0, !dbg !21
  %861 = insertvalue [4 x <1 x float>] %858, <1 x float> %860, 3, !dbg !21
  store [4 x <1 x float>] %861, ptr %340, align 4, !dbg !21
  br label %862, !dbg !21

862:                                              ; preds = %865, %848
  %863 = phi i64 [ %870, %865 ], [ 0, %848 ], !dbg !21
  %864 = icmp slt i64 %863, 4, !dbg !21
  br i1 %864, label %865, label %871, !dbg !21

865:                                              ; preds = %862
  %866 = getelementptr <1 x float>, ptr %340, i64 %863, !dbg !21
  %867 = load <1 x float>, ptr %866, align 4, !dbg !21
  %868 = add i64 %863, 0, !dbg !21
  %869 = getelementptr float, ptr addrspace(5) %289, i64 %868, !dbg !21
  store <1 x float> %867, ptr addrspace(5) %869, align 4, !dbg !21
  %870 = add i64 %863, 1, !dbg !21
  br label %862, !dbg !21

871:                                              ; preds = %862
  %872 = add i64 %333, 0, !dbg !21
  %873 = getelementptr float, ptr addrspace(3) @shm_5, i64 %872, !dbg !21
  %874 = load <1 x float>, ptr addrspace(3) %873, align 4, !dbg !21
  %875 = extractelement <1 x float> %874, i64 0, !dbg !21
  %876 = insertelement <4 x float> poison, float %875, i32 0, !dbg !21
  %877 = shufflevector <4 x float> %876, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  br label %878, !dbg !21

878:                                              ; preds = %882, %871
  %879 = phi i64 [ %887, %882 ], [ 0, %871 ], !dbg !21
  %880 = phi <4 x float> [ %886, %882 ], [ zeroinitializer, %871 ], !dbg !21
  %881 = icmp slt i64 %879, 4, !dbg !21
  br i1 %881, label %882, label %888, !dbg !21

882:                                              ; preds = %878
  %883 = add i64 %879, 0, !dbg !21
  %884 = getelementptr float, ptr addrspace(5) %289, i64 %883, !dbg !21
  %885 = load float, ptr addrspace(5) %884, align 4, !dbg !21
  %886 = insertelement <4 x float> %880, float %885, i64 %879, !dbg !21
  %887 = add i64 %879, 1, !dbg !21
  br label %878, !dbg !21

888:                                              ; preds = %878
  %889 = fadd <4 x float> %880, %877, !dbg !21
  %890 = extractelement <4 x float> %889, i64 0, !dbg !21
  %891 = insertelement <1 x float> poison, float %890, i64 0, !dbg !21
  %892 = insertvalue [4 x <1 x float>] poison, <1 x float> %891, 0, !dbg !21
  %893 = extractelement <4 x float> %889, i64 1, !dbg !21
  %894 = insertelement <1 x float> poison, float %893, i64 0, !dbg !21
  %895 = insertvalue [4 x <1 x float>] %892, <1 x float> %894, 1, !dbg !21
  %896 = extractelement <4 x float> %889, i64 2, !dbg !21
  %897 = insertelement <1 x float> poison, float %896, i64 0, !dbg !21
  %898 = insertvalue [4 x <1 x float>] %895, <1 x float> %897, 2, !dbg !21
  %899 = extractelement <4 x float> %889, i64 3, !dbg !21
  %900 = insertelement <1 x float> poison, float %899, i64 0, !dbg !21
  %901 = insertvalue [4 x <1 x float>] %898, <1 x float> %900, 3, !dbg !21
  store [4 x <1 x float>] %901, ptr %341, align 4, !dbg !21
  br label %902, !dbg !21

902:                                              ; preds = %905, %888
  %903 = phi i64 [ %910, %905 ], [ 0, %888 ], !dbg !21
  %904 = icmp slt i64 %903, 4, !dbg !21
  br i1 %904, label %905, label %911, !dbg !21

905:                                              ; preds = %902
  %906 = getelementptr <1 x float>, ptr %341, i64 %903, !dbg !21
  %907 = load <1 x float>, ptr %906, align 4, !dbg !21
  %908 = add i64 %903, 0, !dbg !21
  %909 = getelementptr float, ptr addrspace(5) %289, i64 %908, !dbg !21
  store <1 x float> %907, ptr addrspace(5) %909, align 4, !dbg !21
  %910 = add i64 %903, 1, !dbg !21
  br label %902, !dbg !21

911:                                              ; preds = %902
  fence syncscope("workgroup") release, !dbg !22
  call void @llvm.amdgcn.s.barrier(), !dbg !22
  fence syncscope("workgroup") acquire, !dbg !22
  br label %912, !dbg !22

912:                                              ; preds = %940, %911
  %913 = phi i64 [ %941, %940 ], [ 0, %911 ], !dbg !22
  %914 = icmp slt i64 %913, 4, !dbg !22
  br i1 %914, label %915, label %942, !dbg !22

915:                                              ; preds = %912
  %916 = mul nsw i64 %913, 16, !dbg !22
  %917 = add i64 %916, %11, !dbg !22
  %918 = add i64 %917, %30, !dbg !22
  br label %919, !dbg !22

919:                                              ; preds = %922, %915
  %920 = phi i64 [ %939, %922 ], [ 0, %915 ], !dbg !22
  %921 = icmp slt i64 %920, 2, !dbg !22
  br i1 %921, label %922, label %940, !dbg !22

922:                                              ; preds = %919
  %923 = mul nsw i64 %920, 16, !dbg !22
  %924 = add i64 %923, %32, !dbg !22
  %925 = sub i64 32, %924, !dbg !22
  %926 = insertelement <4 x i64> poison, i64 %925, i32 0, !dbg !22
  %927 = shufflevector <4 x i64> %926, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !22
  %928 = icmp sgt <4 x i64> %927, <i64 0, i64 1, i64 2, i64 3>, !dbg !22
  %929 = mul i64 %918, 32, !dbg !22
  %930 = add i64 %929, %924, !dbg !22
  %931 = getelementptr float, ptr addrspace(3) @shm_4, i64 %930, !dbg !22
  %932 = call <4 x float> @llvm.masked.load.v4f32.p3(ptr addrspace(3) %931, i32 4, <4 x i1> %928, <4 x float> zeroinitializer), !dbg !22
  %933 = getelementptr float, ptr addrspace(5) %305, i64 0, !dbg !22
  store <4 x float> %932, ptr addrspace(5) %933, align 4, !dbg !22
  %934 = load <4 x float>, ptr addrspace(5) %933, align 4, !dbg !22
  %935 = fptrunc <4 x float> %934 to <4 x half>, !dbg !22
  %936 = getelementptr half, ptr addrspace(5) %306, i64 0, !dbg !22
  store <4 x half> %935, ptr addrspace(5) %936, align 2, !dbg !22
  %937 = load <4 x half>, ptr addrspace(5) %936, align 2, !dbg !22
  %938 = getelementptr half, ptr addrspace(3) @shm_3, i64 %930, !dbg !22
  call void @llvm.masked.store.v4f16.p3(<4 x half> %937, ptr addrspace(3) %938, i32 2, <4 x i1> %928), !dbg !22
  %939 = add i64 %920, 1, !dbg !22
  br label %919, !dbg !22

940:                                              ; preds = %919
  %941 = add i64 %913, 1, !dbg !22
  br label %912, !dbg !22

942:                                              ; preds = %912
  br label %943, !dbg !23

943:                                              ; preds = %1039, %942
  %944 = phi i64 [ %1040, %1039 ], [ 0, %942 ], !dbg !23
  %945 = icmp slt i64 %944, 4, !dbg !23
  br i1 %945, label %946, label %1041, !dbg !23

946:                                              ; preds = %943
  %947 = mul nsw i64 %944, 16, !dbg !23
  %948 = add i64 %947, %11, !dbg !23
  %949 = add i64 %948, %30, !dbg !23
  br label %950, !dbg !23

950:                                              ; preds = %1037, %946
  %951 = phi i64 [ %1038, %1037 ], [ 0, %946 ], !dbg !23
  %952 = icmp slt i64 %951, 8, !dbg !23
  br i1 %952, label %953, label %1039, !dbg !23

953:                                              ; preds = %950
  %954 = getelementptr float, ptr addrspace(5) %310, i64 0, !dbg !23
  store <4 x float> zeroinitializer, ptr addrspace(5) %954, align 4, !dbg !23
  %955 = mul nsw i64 %951, 16, !dbg !23
  %956 = add i64 %955, %11, !dbg !23
  %957 = add i64 %956, %30, !dbg !23
  br label %958, !dbg !23

958:                                              ; preds = %1014, %953
  %959 = phi i64 [ %1020, %1014 ], [ 0, %953 ], !dbg !23
  %960 = icmp slt i64 %959, 2, !dbg !23
  br i1 %960, label %961, label %1021, !dbg !23

961:                                              ; preds = %958
  %962 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !23
  %963 = mul nsw i64 %959, 16, !dbg !23
  %964 = add i64 %963, %32, !dbg !23
  %965 = sub i64 32, %964, !dbg !23
  %966 = insertelement <4 x i64> poison, i64 %965, i32 0, !dbg !23
  %967 = shufflevector <4 x i64> %966, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !23
  %968 = icmp sgt <4 x i64> %967, <i64 0, i64 1, i64 2, i64 3>, !dbg !23
  %969 = mul i64 %949, 32, !dbg !23
  %970 = add i64 %969, %964, !dbg !23
  %971 = getelementptr half, ptr addrspace(3) @shm_3, i64 %970, !dbg !23
  %972 = call <4 x half> @llvm.masked.load.v4f16.p3(ptr addrspace(3) %971, i32 2, <4 x i1> %968, <4 x half> zeroinitializer), !dbg !23
  %973 = getelementptr half, ptr addrspace(5) %307, i64 0, !dbg !23
  store <4 x half> %972, ptr addrspace(5) %973, align 2, !dbg !23
  br label %974, !dbg !23

974:                                              ; preds = %990, %961
  %975 = phi i64 [ %991, %990 ], [ 0, %961 ], !dbg !23
  %976 = phi <4 x half> [ %989, %990 ], [ zeroinitializer, %961 ], !dbg !23
  %977 = icmp slt i64 %975, 4, !dbg !23
  br i1 %977, label %978, label %992, !dbg !23

978:                                              ; preds = %974
  %979 = add i64 %964, %975, !dbg !23
  %980 = icmp slt i64 %979, 32, !dbg !23
  br i1 %980, label %981, label %987, !dbg !23

981:                                              ; preds = %978
  %982 = mul i64 %979, 128, !dbg !23
  %983 = add i64 %982, %957, !dbg !23
  %984 = getelementptr half, ptr addrspace(3) @shm_2, i64 %983, !dbg !23
  %985 = load half, ptr addrspace(3) %984, align 2, !dbg !23
  %986 = insertelement <4 x half> %976, half %985, i64 %975, !dbg !23
  br label %988, !dbg !23

987:                                              ; preds = %978
  br label %988, !dbg !23

988:                                              ; preds = %981, %987
  %989 = phi <4 x half> [ %976, %987 ], [ %986, %981 ], !dbg !23
  br label %990, !dbg !23

990:                                              ; preds = %988
  %991 = add i64 %975, 1, !dbg !23
  br label %974, !dbg !23

992:                                              ; preds = %974
  %993 = extractelement <4 x half> %976, i64 0, !dbg !23
  %994 = insertelement <1 x half> poison, half %993, i64 0, !dbg !23
  %995 = insertvalue [4 x <1 x half>] poison, <1 x half> %994, 0, !dbg !23
  %996 = extractelement <4 x half> %976, i64 1, !dbg !23
  %997 = insertelement <1 x half> poison, half %996, i64 0, !dbg !23
  %998 = insertvalue [4 x <1 x half>] %995, <1 x half> %997, 1, !dbg !23
  %999 = extractelement <4 x half> %976, i64 2, !dbg !23
  %1000 = insertelement <1 x half> poison, half %999, i64 0, !dbg !23
  %1001 = insertvalue [4 x <1 x half>] %998, <1 x half> %1000, 2, !dbg !23
  %1002 = extractelement <4 x half> %976, i64 3, !dbg !23
  %1003 = insertelement <1 x half> poison, half %1002, i64 0, !dbg !23
  %1004 = insertvalue [4 x <1 x half>] %1001, <1 x half> %1003, 3, !dbg !23
  store [4 x <1 x half>] %1004, ptr %962, align 2, !dbg !23
  br label %1005, !dbg !23

1005:                                             ; preds = %1008, %992
  %1006 = phi i64 [ %1013, %1008 ], [ 0, %992 ], !dbg !23
  %1007 = icmp slt i64 %1006, 4, !dbg !23
  br i1 %1007, label %1008, label %1014, !dbg !23

1008:                                             ; preds = %1005
  %1009 = getelementptr <1 x half>, ptr %962, i64 %1006, !dbg !23
  %1010 = load <1 x half>, ptr %1009, align 2, !dbg !23
  %1011 = add i64 %1006, 0, !dbg !23
  %1012 = getelementptr half, ptr addrspace(5) %308, i64 %1011, !dbg !23
  store <1 x half> %1010, ptr addrspace(5) %1012, align 2, !dbg !23
  %1013 = add i64 %1006, 1, !dbg !23
  br label %1005, !dbg !23

1014:                                             ; preds = %1005
  %1015 = load <4 x half>, ptr addrspace(5) %973, align 2, !dbg !23
  %1016 = getelementptr half, ptr addrspace(5) %308, i64 0, !dbg !23
  %1017 = load <4 x half>, ptr addrspace(5) %1016, align 2, !dbg !23
  %1018 = load <4 x float>, ptr addrspace(5) %954, align 4, !dbg !23
  %1019 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %1015, <4 x half> %1017, <4 x float> %1018), !dbg !23
  store <4 x float> %1019, ptr addrspace(5) %954, align 4, !dbg !23
  %1020 = add i64 %959, 1, !dbg !23
  br label %958, !dbg !23

1021:                                             ; preds = %958
  %1022 = mul nsw i64 %951, 4, !dbg !23
  br label %1023, !dbg !23

1023:                                             ; preds = %1026, %1021
  %1024 = phi i64 [ %1036, %1026 ], [ 0, %1021 ], !dbg !23
  %1025 = icmp slt i64 %1024, 4, !dbg !23
  br i1 %1025, label %1026, label %1037, !dbg !23

1026:                                             ; preds = %1023
  %1027 = load <4 x float>, ptr addrspace(5) %954, align 4, !dbg !23
  %1028 = add i64 %1022, %1024, !dbg !23
  %1029 = sub i64 32, %1028, !dbg !23
  %1030 = insertelement <4 x i64> poison, i64 %1029, i32 0, !dbg !23
  %1031 = shufflevector <4 x i64> %1030, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !23
  %1032 = icmp sgt <4 x i64> %1031, <i64 0, i64 1, i64 2, i64 3>, !dbg !23
  %1033 = mul i64 %944, 32, !dbg !23
  %1034 = add i64 %1033, %1028, !dbg !23
  %1035 = getelementptr float, ptr addrspace(5) %309, i64 %1034, !dbg !23
  call void @llvm.masked.store.v4f32.p5(<4 x float> %1027, ptr addrspace(5) %1035, i32 4, <4 x i1> %1032), !dbg !23
  %1036 = add i64 %1024, 1, !dbg !23
  br label %1023, !dbg !23

1037:                                             ; preds = %1023
  %1038 = add i64 %951, 1, !dbg !23
  br label %950, !dbg !23

1039:                                             ; preds = %950
  %1040 = add i64 %944, 1, !dbg !23
  br label %943, !dbg !23

1041:                                             ; preds = %943
  br label %1042, !dbg !32

1042:                                             ; preds = %1083, %1041
  %1043 = phi i64 [ %1084, %1083 ], [ 0, %1041 ], !dbg !32
  %1044 = icmp slt i64 %1043, 4, !dbg !32
  br i1 %1044, label %1045, label %1085, !dbg !32

1045:                                             ; preds = %1042
  br label %1046, !dbg !32

1046:                                             ; preds = %1081, %1045
  %1047 = phi i64 [ %1082, %1081 ], [ 0, %1045 ], !dbg !32
  %1048 = icmp slt i64 %1047, 32, !dbg !32
  br i1 %1048, label %1049, label %1083, !dbg !32

1049:                                             ; preds = %1046
  br label %1050, !dbg !32

1050:                                             ; preds = %1079, %1049
  %1051 = phi i64 [ %1080, %1079 ], [ 0, %1049 ], !dbg !32
  %1052 = icmp slt i64 %1051, 4, !dbg !32
  br i1 %1052, label %1053, label %1081, !dbg !32

1053:                                             ; preds = %1050
  br label %1054, !dbg !32

1054:                                             ; preds = %1057, %1053
  %1055 = phi i64 [ %1078, %1057 ], [ 0, %1053 ], !dbg !32
  %1056 = icmp slt i64 %1055, 8, !dbg !32
  br i1 %1056, label %1057, label %1079, !dbg !32

1057:                                             ; preds = %1054
  %1058 = sub i64 32, %1047, !dbg !32
  %1059 = insertelement <4 x i64> poison, i64 %1058, i32 0, !dbg !32
  %1060 = shufflevector <4 x i64> %1059, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !32
  %1061 = icmp sgt <4 x i64> %1060, <i64 0, i64 1, i64 2, i64 3>, !dbg !32
  %1062 = mul i64 %1043, 32, !dbg !32
  %1063 = add i64 %1062, %1047, !dbg !32
  %1064 = getelementptr float, ptr addrspace(5) %248, i64 %1063, !dbg !32
  %1065 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1064, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1066 = getelementptr float, ptr addrspace(5) %309, i64 %1063, !dbg !32
  %1067 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1066, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1068 = fadd <4 x float> %1065, %1067, !dbg !32
  call void @llvm.masked.store.v4f32.p5(<4 x float> %1068, ptr addrspace(5) %1064, i32 4, <4 x i1> %1061), !dbg !32
  %1069 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1064, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1070 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1066, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1071 = fadd <4 x float> %1069, %1070, !dbg !32
  call void @llvm.masked.store.v4f32.p5(<4 x float> %1071, ptr addrspace(5) %1064, i32 4, <4 x i1> %1061), !dbg !32
  %1072 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1064, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1073 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1066, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1074 = fadd <4 x float> %1072, %1073, !dbg !32
  call void @llvm.masked.store.v4f32.p5(<4 x float> %1074, ptr addrspace(5) %1064, i32 4, <4 x i1> %1061), !dbg !32
  %1075 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1064, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1076 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1066, i32 4, <4 x i1> %1061, <4 x float> zeroinitializer), !dbg !32
  %1077 = fadd <4 x float> %1075, %1076, !dbg !32
  call void @llvm.masked.store.v4f32.p5(<4 x float> %1077, ptr addrspace(5) %1064, i32 4, <4 x i1> %1061), !dbg !32
  %1078 = add i64 %1055, 4, !dbg !32
  br label %1054, !dbg !32

1079:                                             ; preds = %1054
  %1080 = add i64 %1051, 1, !dbg !32
  br label %1050, !dbg !32

1081:                                             ; preds = %1050
  %1082 = add i64 %1047, 4, !dbg !32
  br label %1046, !dbg !32

1083:                                             ; preds = %1046
  %1084 = add i64 %1043, 1, !dbg !32
  br label %1042, !dbg !32

1085:                                             ; preds = %1042
  %1086 = add i64 %335, 1, !dbg !24
  br label %334, !dbg !24

1087:                                             ; preds = %334
  %1088 = alloca half, i64 128, align 2, addrspace(5), !dbg !33
  br label %1089, !dbg !34

1089:                                             ; preds = %1148, %1087
  %1090 = phi i64 [ %1149, %1148 ], [ 0, %1087 ], !dbg !34
  %1091 = icmp slt i64 %1090, 4, !dbg !34
  br i1 %1091, label %1092, label %1150, !dbg !34

1092:                                             ; preds = %1089
  br label %1093, !dbg !34

1093:                                             ; preds = %1146, %1092
  %1094 = phi i64 [ %1147, %1146 ], [ 0, %1092 ], !dbg !34
  %1095 = icmp slt i64 %1094, 32, !dbg !34
  br i1 %1095, label %1096, label %1148, !dbg !34

1096:                                             ; preds = %1093
  br label %1097, !dbg !34

1097:                                             ; preds = %1144, %1096
  %1098 = phi i64 [ %1145, %1144 ], [ 0, %1096 ], !dbg !34
  %1099 = icmp slt i64 %1098, 4, !dbg !34
  br i1 %1099, label %1100, label %1146, !dbg !34

1100:                                             ; preds = %1097
  br label %1101, !dbg !34

1101:                                             ; preds = %1104, %1100
  %1102 = phi i64 [ %1143, %1104 ], [ 0, %1100 ], !dbg !34
  %1103 = icmp slt i64 %1102, 8, !dbg !34
  br i1 %1103, label %1104, label %1144, !dbg !34

1104:                                             ; preds = %1101
  %1105 = sub i64 32, %1094, !dbg !34
  %1106 = insertelement <4 x i64> poison, i64 %1105, i32 0, !dbg !34
  %1107 = shufflevector <4 x i64> %1106, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !34
  %1108 = icmp sgt <4 x i64> %1107, <i64 0, i64 1, i64 2, i64 3>, !dbg !34
  %1109 = mul i64 %1090, 32, !dbg !34
  %1110 = add i64 %1109, %1094, !dbg !34
  %1111 = getelementptr float, ptr addrspace(5) %248, i64 %1110, !dbg !34
  %1112 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1111, i32 4, <4 x i1> %1108, <4 x float> zeroinitializer), !dbg !34
  %1113 = add i64 %1090, 0, !dbg !34
  %1114 = getelementptr float, ptr addrspace(5) %289, i64 %1113, !dbg !34
  %1115 = load <1 x float>, ptr addrspace(5) %1114, align 4, !dbg !34
  %1116 = extractelement <1 x float> %1115, i64 0, !dbg !34
  %1117 = insertelement <4 x float> poison, float %1116, i32 0, !dbg !34
  %1118 = shufflevector <4 x float> %1117, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1119 = fdiv <4 x float> %1112, %1118, !dbg !34
  %1120 = fptrunc <4 x float> %1119 to <4 x half>, !dbg !10
  %1121 = getelementptr half, ptr addrspace(5) %1088, i64 %1110, !dbg !10
  call void @llvm.masked.store.v4f16.p5(<4 x half> %1120, ptr addrspace(5) %1121, i32 2, <4 x i1> %1108), !dbg !10
  %1122 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1111, i32 4, <4 x i1> %1108, <4 x float> zeroinitializer), !dbg !34
  %1123 = load <1 x float>, ptr addrspace(5) %1114, align 4, !dbg !34
  %1124 = extractelement <1 x float> %1123, i64 0, !dbg !34
  %1125 = insertelement <4 x float> poison, float %1124, i32 0, !dbg !34
  %1126 = shufflevector <4 x float> %1125, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1127 = fdiv <4 x float> %1122, %1126, !dbg !34
  %1128 = fptrunc <4 x float> %1127 to <4 x half>, !dbg !10
  call void @llvm.masked.store.v4f16.p5(<4 x half> %1128, ptr addrspace(5) %1121, i32 2, <4 x i1> %1108), !dbg !10
  %1129 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1111, i32 4, <4 x i1> %1108, <4 x float> zeroinitializer), !dbg !34
  %1130 = load <1 x float>, ptr addrspace(5) %1114, align 4, !dbg !34
  %1131 = extractelement <1 x float> %1130, i64 0, !dbg !34
  %1132 = insertelement <4 x float> poison, float %1131, i32 0, !dbg !34
  %1133 = shufflevector <4 x float> %1132, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1134 = fdiv <4 x float> %1129, %1133, !dbg !34
  %1135 = fptrunc <4 x float> %1134 to <4 x half>, !dbg !10
  call void @llvm.masked.store.v4f16.p5(<4 x half> %1135, ptr addrspace(5) %1121, i32 2, <4 x i1> %1108), !dbg !10
  %1136 = call <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5) %1111, i32 4, <4 x i1> %1108, <4 x float> zeroinitializer), !dbg !34
  %1137 = load <1 x float>, ptr addrspace(5) %1114, align 4, !dbg !34
  %1138 = extractelement <1 x float> %1137, i64 0, !dbg !34
  %1139 = insertelement <4 x float> poison, float %1138, i32 0, !dbg !34
  %1140 = shufflevector <4 x float> %1139, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !34
  %1141 = fdiv <4 x float> %1136, %1140, !dbg !34
  %1142 = fptrunc <4 x float> %1141 to <4 x half>, !dbg !10
  call void @llvm.masked.store.v4f16.p5(<4 x half> %1142, ptr addrspace(5) %1121, i32 2, <4 x i1> %1108), !dbg !10
  %1143 = add i64 %1102, 4, !dbg !34
  br label %1101, !dbg !34

1144:                                             ; preds = %1101
  %1145 = add i64 %1098, 1, !dbg !34
  br label %1097, !dbg !34

1146:                                             ; preds = %1097
  %1147 = add i64 %1094, 4, !dbg !34
  br label %1093, !dbg !34

1148:                                             ; preds = %1093
  %1149 = add i64 %1090, 1, !dbg !34
  br label %1089, !dbg !34

1150:                                             ; preds = %1089
  br label %1151, !dbg !35

1151:                                             ; preds = %1222, %1150
  %1152 = phi i64 [ %1223, %1222 ], [ 0, %1150 ], !dbg !35
  %1153 = icmp slt i64 %1152, 4, !dbg !35
  br i1 %1153, label %1154, label %1224, !dbg !35

1154:                                             ; preds = %1151
  %1155 = add i64 %1152, %23, !dbg !35
  %1156 = add i64 %1155, %31, !dbg !35
  br label %1157, !dbg !35

1157:                                             ; preds = %1160, %1154
  %1158 = phi i64 [ %1221, %1160 ], [ 0, %1154 ], !dbg !35
  %1159 = icmp slt i64 %1158, 32, !dbg !35
  br i1 %1159, label %1160, label %1222, !dbg !35

1160:                                             ; preds = %1157
  %1161 = sub i64 32, %1158, !dbg !35
  %1162 = insertelement <4 x i64> poison, i64 %1161, i32 0, !dbg !35
  %1163 = shufflevector <4 x i64> %1162, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1164 = icmp sgt <4 x i64> %1163, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1165 = mul i64 %1152, 32, !dbg !35
  %1166 = add i64 %1165, %1158, !dbg !35
  %1167 = getelementptr half, ptr addrspace(5) %1088, i64 %1166, !dbg !35
  %1168 = call <4 x half> @llvm.masked.load.v4f16.p5(ptr addrspace(5) %1167, i32 2, <4 x i1> %1164, <4 x half> zeroinitializer), !dbg !35
  %1169 = sub i64 128, %1158, !dbg !35
  %1170 = insertelement <4 x i64> poison, i64 %1169, i32 0, !dbg !35
  %1171 = shufflevector <4 x i64> %1170, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1172 = icmp sgt <4 x i64> %1171, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1173 = mul i64 %22, 524288, !dbg !35
  %1174 = add i64 0, %1173, !dbg !35
  %1175 = mul i64 %1156, 128, !dbg !35
  %1176 = add i64 %1174, %1175, !dbg !35
  %1177 = add i64 %1176, %1158, !dbg !35
  %1178 = getelementptr half, ptr addrspace(1) %3, i64 %1177, !dbg !35
  call void @llvm.masked.store.v4f16.p1(<4 x half> %1168, ptr addrspace(1) %1178, i32 2, <4 x i1> %1172), !dbg !35
  %1179 = add i64 %1158, 4, !dbg !35
  %1180 = sub i64 32, %1179, !dbg !35
  %1181 = insertelement <4 x i64> poison, i64 %1180, i32 0, !dbg !35
  %1182 = shufflevector <4 x i64> %1181, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1183 = icmp sgt <4 x i64> %1182, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1184 = add i64 %1165, %1179, !dbg !35
  %1185 = getelementptr half, ptr addrspace(5) %1088, i64 %1184, !dbg !35
  %1186 = call <4 x half> @llvm.masked.load.v4f16.p5(ptr addrspace(5) %1185, i32 2, <4 x i1> %1183, <4 x half> zeroinitializer), !dbg !35
  %1187 = sub i64 128, %1179, !dbg !35
  %1188 = insertelement <4 x i64> poison, i64 %1187, i32 0, !dbg !35
  %1189 = shufflevector <4 x i64> %1188, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1190 = icmp sgt <4 x i64> %1189, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1191 = add i64 %1176, %1179, !dbg !35
  %1192 = getelementptr half, ptr addrspace(1) %3, i64 %1191, !dbg !35
  call void @llvm.masked.store.v4f16.p1(<4 x half> %1186, ptr addrspace(1) %1192, i32 2, <4 x i1> %1190), !dbg !35
  %1193 = add i64 %1158, 8, !dbg !35
  %1194 = sub i64 32, %1193, !dbg !35
  %1195 = insertelement <4 x i64> poison, i64 %1194, i32 0, !dbg !35
  %1196 = shufflevector <4 x i64> %1195, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1197 = icmp sgt <4 x i64> %1196, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1198 = add i64 %1165, %1193, !dbg !35
  %1199 = getelementptr half, ptr addrspace(5) %1088, i64 %1198, !dbg !35
  %1200 = call <4 x half> @llvm.masked.load.v4f16.p5(ptr addrspace(5) %1199, i32 2, <4 x i1> %1197, <4 x half> zeroinitializer), !dbg !35
  %1201 = sub i64 128, %1193, !dbg !35
  %1202 = insertelement <4 x i64> poison, i64 %1201, i32 0, !dbg !35
  %1203 = shufflevector <4 x i64> %1202, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1204 = icmp sgt <4 x i64> %1203, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1205 = add i64 %1176, %1193, !dbg !35
  %1206 = getelementptr half, ptr addrspace(1) %3, i64 %1205, !dbg !35
  call void @llvm.masked.store.v4f16.p1(<4 x half> %1200, ptr addrspace(1) %1206, i32 2, <4 x i1> %1204), !dbg !35
  %1207 = add i64 %1158, 12, !dbg !35
  %1208 = sub i64 32, %1207, !dbg !35
  %1209 = insertelement <4 x i64> poison, i64 %1208, i32 0, !dbg !35
  %1210 = shufflevector <4 x i64> %1209, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1211 = icmp sgt <4 x i64> %1210, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1212 = add i64 %1165, %1207, !dbg !35
  %1213 = getelementptr half, ptr addrspace(5) %1088, i64 %1212, !dbg !35
  %1214 = call <4 x half> @llvm.masked.load.v4f16.p5(ptr addrspace(5) %1213, i32 2, <4 x i1> %1211, <4 x half> zeroinitializer), !dbg !35
  %1215 = sub i64 128, %1207, !dbg !35
  %1216 = insertelement <4 x i64> poison, i64 %1215, i32 0, !dbg !35
  %1217 = shufflevector <4 x i64> %1216, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !35
  %1218 = icmp sgt <4 x i64> %1217, <i64 0, i64 1, i64 2, i64 3>, !dbg !35
  %1219 = add i64 %1176, %1207, !dbg !35
  %1220 = getelementptr half, ptr addrspace(1) %3, i64 %1219, !dbg !35
  call void @llvm.masked.store.v4f16.p1(<4 x half> %1214, ptr addrspace(1) %1220, i32 2, <4 x i1> %1218), !dbg !35
  %1221 = add i64 %1158, 16, !dbg !35
  br label %1157, !dbg !35

1222:                                             ; preds = %1157
  %1223 = add i64 %1152, 1, !dbg !35
  br label %1151, !dbg !35

1224:                                             ; preds = %1151
  ret void, !dbg !10
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare noundef i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x half> @llvm.masked.load.v4f16.p5(ptr addrspace(5), i32 immarg, <4 x i1>, <4 x half>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.masked.store.v4f16.p1(<4 x half>, ptr addrspace(1), i32 immarg, <4 x i1>) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x float> @llvm.masked.load.v4f32.p5(ptr addrspace(5), i32 immarg, <4 x i1>, <4 x float>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.masked.store.v4f16.p5(<4 x half>, ptr addrspace(5), i32 immarg, <4 x i1>) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.masked.store.v4f32.p5(<4 x float>, ptr addrspace(5), i32 immarg, <4 x i1>) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x half> @llvm.masked.load.v4f16.p3(ptr addrspace(3), i32 immarg, <4 x i1>, <4 x half>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x float> @llvm.masked.load.v4f32.p3(ptr addrspace(3), i32 immarg, <4 x i1>, <4 x float>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.masked.store.v4f16.p3(<4 x half>, ptr addrspace(3), i32 immarg, <4 x i1>) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x half> @llvm.masked.load.v4f16.p1(ptr addrspace(1), i32 immarg, <4 x i1>, <4 x half>) #2

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn }
attributes #2 = { nocallback nofree nosync nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn }
attributes #4 = { convergent nocallback nofree nounwind willreturn }
attributes #5 = { nocallback nofree nosync nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 64, i32 1, i32 1}
!7 = !DILocation(line: 53, column: 13, scope: !8)
!8 = distinct !DILexicalBlockFile(scope: !3, file: !9, discriminator: 0)
!9 = !DIFile(filename: "test_input.mlir_frisk.fill", directory: "3rd/deepgengraph/test")
!10 = !DILocation(line: 22, column: 3, scope: !3)
!11 = !DILocation(line: 48, column: 13, scope: !3)
!12 = !DILocation(line: 51, column: 13, scope: !3)
!13 = !DILocation(line: 52, column: 13, scope: !14)
!14 = distinct !DILexicalBlockFile(scope: !3, file: !15, discriminator: 0)
!15 = !DIFile(filename: "test_input.mlir_frisk.alloc_buffer", directory: "3rd/deepgengraph/test")
!16 = !DILocation(line: 52, column: 13, scope: !8)
!17 = !DILocation(line: 52, column: 13, scope: !3)
!18 = !DILocation(line: 53, column: 13, scope: !14)
!19 = !DILocation(line: 53, column: 13, scope: !3)
!20 = !DILocation(line: 62, column: 15, scope: !3)
!21 = !DILocation(line: 76, column: 15, scope: !3)
!22 = !DILocation(line: 77, column: 15, scope: !3)
!23 = !DILocation(line: 78, column: 15, scope: !3)
!24 = !DILocation(line: 59, column: 15, scope: !3)
!25 = !DILocation(line: 61, column: 15, scope: !3)
!26 = !DILocation(line: 60, column: 15, scope: !3)
!27 = !DILocation(line: 63, column: 15, scope: !3)
!28 = !DILocation(line: 75, column: 15, scope: !3)
!29 = !DILocation(line: 65, column: 17, scope: !3)
!30 = !DILocation(line: 66, column: 17, scope: !3)
!31 = !DILocation(line: 74, column: 15, scope: !3)
!32 = !DILocation(line: 79, column: 15, scope: !3)
!33 = !DILocation(line: 85, column: 13, scope: !3)
!34 = !DILocation(line: 84, column: 13, scope: !3)
!35 = !DILocation(line: 86, column: 7, scope: !3)
