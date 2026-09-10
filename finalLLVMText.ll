; ModuleID = '3rd/deepgengraph/test/test_input.mlir'
source_filename = "3rd/deepgengraph/test/test_input.mlir"

@shm_4 = addrspace(3) global [64 x [32 x half]] undef, align 16
@shm_3 = addrspace(3) global [64 x [1 x float]] undef, align 16
@shm_2 = addrspace(3) global [32 x [128 x half]] undef, align 16
@shm_1 = addrspace(3) global [128 x [32 x half]] undef, align 16
@shm_0 = addrspace(3) global [64 x [128 x half]] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !3 !reqd_work_group_size !6 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = mul i64 %6, 64, !dbg !8
  %12 = icmp slt i64 %6, 0, !dbg !9
  %13 = sub i64 -1, %6, !dbg !9
  %14 = select i1 %12, i64 %13, i64 %6, !dbg !9
  %15 = sdiv i64 %14, 64, !dbg !9
  %16 = sub i64 -1, %15, !dbg !9
  %17 = select i1 %12, i64 %16, i64 %15, !dbg !9
  %18 = add i64 %8, %17, !dbg !9
  %19 = srem i64 %18, 32, !dbg !9
  %20 = icmp slt i64 %19, 0, !dbg !9
  %21 = add i64 %19, 32, !dbg !9
  %22 = select i1 %20, i64 %21, i64 %19, !dbg !9
  %23 = mul nsw i64 %6, 64, !dbg !9
  %24 = icmp slt i64 %10, 0, !dbg !9
  %25 = sub i64 -1, %10, !dbg !9
  %26 = select i1 %24, i64 %25, i64 %10, !dbg !9
  %27 = sdiv i64 %26, 16, !dbg !9
  %28 = sub i64 -1, %27, !dbg !9
  %29 = select i1 %24, i64 %28, i64 %27, !dbg !9
  %30 = mul nsw i64 %29, -16, !dbg !9
  %31 = mul nsw i64 %17, -4096, !dbg !9
  %32 = mul nsw i64 %29, 4, !dbg !9
  br label %33, !dbg !9

33:                                               ; preds = %65, %4
  %34 = phi i64 [ %66, %65 ], [ 0, %4 ], !dbg !9
  %35 = icmp slt i64 %34, 4, !dbg !9
  br i1 %35, label %36, label %67, !dbg !9

36:                                               ; preds = %33
  %37 = mul nsw i64 %34, 16, !dbg !9
  %38 = add i64 %37, %23, !dbg !9
  %39 = add i64 %38, %10, !dbg !9
  %40 = add i64 %39, %30, !dbg !9
  %41 = add i64 %40, %31, !dbg !9
  %42 = add i64 %37, %10, !dbg !9
  %43 = add i64 %42, %30, !dbg !9
  br label %44, !dbg !9

44:                                               ; preds = %47, %36
  %45 = phi i64 [ %64, %47 ], [ 0, %36 ], !dbg !9
  %46 = icmp slt i64 %45, 8, !dbg !9
  br i1 %46, label %47, label %65, !dbg !9

47:                                               ; preds = %44
  %48 = mul nsw i64 %45, 16, !dbg !9
  %49 = add i64 %48, %32, !dbg !9
  %50 = sub i64 128, %49, !dbg !9
  %51 = insertelement <4 x i64> poison, i64 %50, i32 0, !dbg !9
  %52 = shufflevector <4 x i64> %51, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !9
  %53 = icmp sgt <4 x i64> %52, <i64 0, i64 1, i64 2, i64 3>, !dbg !9
  %54 = mul i64 %22, 524288, !dbg !9
  %55 = add i64 0, %54, !dbg !9
  %56 = mul i64 %41, 128, !dbg !9
  %57 = add i64 %55, %56, !dbg !9
  %58 = add i64 %57, %49, !dbg !9
  %59 = getelementptr half, ptr addrspace(1) %0, i64 %58, !dbg !9
  %60 = call <4 x half> @llvm.masked.load.v4f16.p1(ptr addrspace(1) %59, i32 2, <4 x i1> %53, <4 x half> zeroinitializer), !dbg !9
  %61 = mul i64 %43, 128, !dbg !9
  %62 = add i64 %61, %49, !dbg !9
  %63 = getelementptr half, ptr addrspace(3) @shm_0, i64 %62, !dbg !9
  call void @llvm.masked.store.v4f16.p3(<4 x half> %60, ptr addrspace(3) %63, i32 2, <4 x i1> %53), !dbg !9
  %64 = add i64 %45, 1, !dbg !9
  br label %44, !dbg !9

65:                                               ; preds = %44
  %66 = add i64 %34, 1, !dbg !9
  br label %33, !dbg !9

67:                                               ; preds = %33
  %68 = srem i64 %10, 16, !dbg !10
  %69 = icmp slt i64 %68, 0, !dbg !10
  %70 = add i64 %68, 16, !dbg !10
  %71 = select i1 %69, i64 %70, i64 %68, !dbg !10
  %72 = mul i64 %71, 128, !dbg !10
  %73 = add i64 %72, %32, !dbg !10
  %74 = getelementptr half, ptr addrspace(3) @shm_0, i64 %73, !dbg !10
  %75 = load half, ptr addrspace(3) %74, align 2, !dbg !10
  %76 = insertelement <4 x half> zeroinitializer, half %75, i64 0, !dbg !10
  %77 = add i64 %32, 1, !dbg !10
  %78 = add i64 %72, %77, !dbg !10
  %79 = getelementptr half, ptr addrspace(3) @shm_0, i64 %78, !dbg !10
  %80 = load half, ptr addrspace(3) %79, align 2, !dbg !10
  %81 = insertelement <4 x half> %76, half %80, i64 1, !dbg !10
  %82 = add i64 %32, 2, !dbg !10
  %83 = add i64 %72, %82, !dbg !10
  %84 = getelementptr half, ptr addrspace(3) @shm_0, i64 %83, !dbg !10
  %85 = load half, ptr addrspace(3) %84, align 2, !dbg !10
  %86 = insertelement <4 x half> %81, half %85, i64 2, !dbg !10
  %87 = add i64 %32, 3, !dbg !10
  %88 = add i64 %72, %87, !dbg !10
  %89 = getelementptr half, ptr addrspace(3) @shm_0, i64 %88, !dbg !10
  %90 = load half, ptr addrspace(3) %89, align 2, !dbg !10
  %91 = insertelement <4 x half> %86, half %90, i64 3, !dbg !10
  %compat.splat.insert = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !10
  %compat.splat = shufflevector <4 x half> %compat.splat.insert, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !10
  %92 = fmul <4 x half> %91, %compat.splat, !dbg !10
  %93 = alloca float, i64 1, align 4, addrspace(5), !dbg !11
  %94 = alloca float, i64 4, align 1, addrspace(5), !dbg !12
  %95 = alloca half, i64 4, align 1, addrspace(5), !dbg !12
  %96 = srem i64 %10, 64, !dbg !11
  %97 = icmp slt i64 %96, 0, !dbg !11
  %98 = add i64 %96, 64, !dbg !11
  %99 = select i1 %97, i64 %98, i64 %96, !dbg !11
  %100 = icmp slt i64 %99, 0, !dbg !11
  %101 = sub i64 -1, %99, !dbg !11
  %102 = select i1 %100, i64 %101, i64 %99, !dbg !11
  %103 = sdiv i64 %102, 16, !dbg !11
  %104 = sub i64 -1, %103, !dbg !11
  %105 = select i1 %100, i64 %104, i64 %103, !dbg !11
  %106 = icmp eq i64 %105, 0, !dbg !11
  %107 = mul nsw i64 %6, 2, !dbg !13
  %108 = add i64 %107, 1, !dbg !13
  %109 = mul nsw i64 %6, 256, !dbg !14
  %110 = mul nsw i64 %8, 128, !dbg !14
  %111 = add i64 %107, %110, !dbg !14
  %112 = add i64 %71, 1, !dbg !11
  %113 = add i64 %71, 2, !dbg !11
  %114 = add i64 %71, 3, !dbg !11
  br label %115, !dbg !13

115:                                              ; preds = %962, %67
  %116 = phi i64 [ %979, %962 ], [ 0, %67 ], !dbg !13
  %117 = phi [4 x <32 x float>] [ %978, %962 ], [ zeroinitializer, %67 ], !dbg !13
  %118 = phi [4 x <1 x float>] [ %776, %962 ], [ zeroinitializer, %67 ], !dbg !13
  %119 = icmp slt i64 %116, %108, !dbg !13
  br i1 %119, label %120, label %980, !dbg !13

120:                                              ; preds = %115
  %121 = add i64 %116, %107, !dbg !15
  %122 = icmp slt i64 %121, 0, !dbg !15
  %123 = sub i64 -1, %121, !dbg !15
  %124 = select i1 %122, i64 %123, i64 %121, !dbg !15
  %125 = sdiv i64 %124, 128, !dbg !15
  %126 = sub i64 -1, %125, !dbg !15
  %127 = select i1 %122, i64 %126, i64 %125, !dbg !15
  %128 = add i64 %127, %8, !dbg !15
  %129 = srem i64 %128, 32, !dbg !15
  %130 = icmp slt i64 %129, 0, !dbg !15
  %131 = add i64 %129, 32, !dbg !15
  %132 = select i1 %130, i64 %131, i64 %129, !dbg !15
  %133 = mul nsw i64 %116, 32, !dbg !15
  %134 = mul nsw i64 %127, -4096, !dbg !15
  br label %135, !dbg !15

135:                                              ; preds = %211, %120
  %136 = phi i64 [ %212, %211 ], [ 0, %120 ], !dbg !15
  %137 = icmp slt i64 %136, 4, !dbg !15
  br i1 %137, label %138, label %213, !dbg !15

138:                                              ; preds = %135
  %139 = add i64 %133, %136, !dbg !15
  br label %140, !dbg !15

140:                                              ; preds = %209, %138
  %141 = phi i64 [ %210, %209 ], [ 0, %138 ], !dbg !15
  %142 = icmp slt i64 %141, 2, !dbg !15
  br i1 %142, label %143, label %211, !dbg !15

143:                                              ; preds = %140
  %144 = mul nsw i64 %141, 16, !dbg !15
  %145 = add i64 %144, %10, !dbg !15
  %146 = add i64 %145, %30, !dbg !15
  br label %147, !dbg !15

147:                                              ; preds = %207, %143
  %148 = phi i64 [ %208, %207 ], [ 0, %143 ], !dbg !15
  %149 = icmp slt i64 %148, 2, !dbg !15
  br i1 %149, label %150, label %209, !dbg !15

150:                                              ; preds = %147
  %151 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !15
  %152 = mul nsw i64 %148, 64, !dbg !15
  %153 = add i64 %139, %152, !dbg !15
  %154 = add i64 %153, %23, !dbg !15
  %155 = add i64 %154, %134, !dbg !15
  %156 = add i64 %155, %32, !dbg !15
  br label %157, !dbg !15

157:                                              ; preds = %176, %150
  %158 = phi i64 [ %177, %176 ], [ 0, %150 ], !dbg !15
  %159 = phi <4 x half> [ %175, %176 ], [ zeroinitializer, %150 ], !dbg !15
  %160 = icmp slt i64 %158, 4, !dbg !15
  br i1 %160, label %161, label %178, !dbg !15

161:                                              ; preds = %157
  %162 = add i64 %156, %158, !dbg !15
  %163 = icmp slt i64 %162, 4096, !dbg !15
  br i1 %163, label %164, label %173, !dbg !15

164:                                              ; preds = %161
  %165 = mul i64 %132, 524288, !dbg !15
  %166 = add i64 0, %165, !dbg !15
  %167 = mul i64 %162, 128, !dbg !15
  %168 = add i64 %166, %167, !dbg !15
  %169 = add i64 %168, %146, !dbg !15
  %170 = getelementptr half, ptr addrspace(1) %2, i64 %169, !dbg !15
  %171 = load half, ptr addrspace(1) %170, align 2, !dbg !15
  %172 = insertelement <4 x half> %159, half %171, i64 %158, !dbg !15
  br label %174, !dbg !15

173:                                              ; preds = %161
  br label %174, !dbg !15

174:                                              ; preds = %164, %173
  %175 = phi <4 x half> [ %159, %173 ], [ %172, %164 ], !dbg !15
  br label %176, !dbg !15

176:                                              ; preds = %174
  %177 = add i64 %158, 1, !dbg !15
  br label %157, !dbg !15

178:                                              ; preds = %157
  %179 = add i64 %136, %152, !dbg !15
  %180 = add i64 %179, %32, !dbg !15
  %181 = extractelement <4 x half> %159, i64 0, !dbg !15
  %182 = insertelement <1 x half> poison, half %181, i64 0, !dbg !15
  %183 = insertvalue [4 x <1 x half>] poison, <1 x half> %182, 0, !dbg !15
  %184 = extractelement <4 x half> %159, i64 1, !dbg !15
  %185 = insertelement <1 x half> poison, half %184, i64 0, !dbg !15
  %186 = insertvalue [4 x <1 x half>] %183, <1 x half> %185, 1, !dbg !15
  %187 = extractelement <4 x half> %159, i64 2, !dbg !15
  %188 = insertelement <1 x half> poison, half %187, i64 0, !dbg !15
  %189 = insertvalue [4 x <1 x half>] %186, <1 x half> %188, 2, !dbg !15
  %190 = extractelement <4 x half> %159, i64 3, !dbg !15
  %191 = insertelement <1 x half> poison, half %190, i64 0, !dbg !15
  %192 = insertvalue [4 x <1 x half>] %189, <1 x half> %191, 3, !dbg !15
  store [4 x <1 x half>] %192, ptr %151, align 2, !dbg !15
  br label %193, !dbg !15

193:                                              ; preds = %205, %178
  %194 = phi i64 [ %206, %205 ], [ 0, %178 ], !dbg !15
  %195 = icmp slt i64 %194, 4, !dbg !15
  br i1 %195, label %196, label %207, !dbg !15

196:                                              ; preds = %193
  %197 = add i64 %180, %194, !dbg !15
  %198 = icmp slt i64 %197, 128, !dbg !15
  br i1 %198, label %199, label %205, !dbg !15

199:                                              ; preds = %196
  %200 = getelementptr <1 x half>, ptr %151, i64 %194, !dbg !15
  %201 = load <1 x half>, ptr %200, align 2, !dbg !15
  %202 = mul i64 %197, 32, !dbg !15
  %203 = add i64 %202, %146, !dbg !15
  %204 = getelementptr half, ptr addrspace(3) @shm_1, i64 %203, !dbg !15
  store <1 x half> %201, ptr addrspace(3) %204, align 2, !dbg !15
  br label %205, !dbg !15

205:                                              ; preds = %199, %196
  %206 = add i64 %194, 1, !dbg !15
  br label %193, !dbg !15

207:                                              ; preds = %193
  %208 = add i64 %148, 1, !dbg !15
  br label %147, !dbg !15

209:                                              ; preds = %147
  %210 = add i64 %141, 1, !dbg !15
  br label %140, !dbg !15

211:                                              ; preds = %140
  %212 = add i64 %136, 1, !dbg !15
  br label %135, !dbg !15

213:                                              ; preds = %135
  %214 = add i64 %116, %109, !dbg !14
  %215 = icmp slt i64 %214, 0, !dbg !14
  %216 = sub i64 -1, %214, !dbg !14
  %217 = select i1 %215, i64 %216, i64 %214, !dbg !14
  %218 = sdiv i64 %217, 16384, !dbg !14
  %219 = sub i64 -1, %218, !dbg !14
  %220 = select i1 %215, i64 %219, i64 %218, !dbg !14
  %221 = add i64 %220, %8, !dbg !14
  %222 = srem i64 %221, 32, !dbg !14
  %223 = icmp slt i64 %222, 0, !dbg !14
  %224 = add i64 %222, 32, !dbg !14
  %225 = select i1 %223, i64 %224, i64 %222, !dbg !14
  %226 = icmp slt i64 %116, 0, !dbg !14
  %227 = sub i64 -1, %116, !dbg !14
  %228 = select i1 %226, i64 %227, i64 %116, !dbg !14
  %229 = sdiv i64 %228, 128, !dbg !14
  %230 = sub i64 -1, %229, !dbg !14
  %231 = select i1 %226, i64 %230, i64 %229, !dbg !14
  %232 = add i64 %231, %111, !dbg !14
  %233 = icmp slt i64 %232, 0, !dbg !14
  %234 = sub i64 -1, %232, !dbg !14
  %235 = select i1 %233, i64 %234, i64 %232, !dbg !14
  %236 = sdiv i64 %235, 128, !dbg !14
  %237 = sub i64 -1, %236, !dbg !14
  %238 = select i1 %233, i64 %237, i64 %236, !dbg !14
  %239 = mul nsw i64 %238, -128, !dbg !14
  %240 = mul nsw i64 %231, -4096, !dbg !14
  br label %241, !dbg !14

241:                                              ; preds = %313, %213
  %242 = phi i64 [ %314, %313 ], [ 0, %213 ], !dbg !14
  %243 = icmp slt i64 %242, 4, !dbg !14
  br i1 %243, label %244, label %315, !dbg !14

244:                                              ; preds = %241
  %245 = add i64 %242, %107, !dbg !14
  %246 = add i64 %245, %110, !dbg !14
  %247 = add i64 %246, %231, !dbg !14
  %248 = add i64 %247, %239, !dbg !14
  %249 = add i64 %248, %32, !dbg !14
  %250 = add i64 %242, %32, !dbg !14
  br label %251, !dbg !14

251:                                              ; preds = %311, %244
  %252 = phi i64 [ %312, %311 ], [ 0, %244 ], !dbg !14
  %253 = icmp slt i64 %252, 8, !dbg !14
  br i1 %253, label %254, label %313, !dbg !14

254:                                              ; preds = %251
  %255 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !14
  %256 = mul nsw i64 %252, 16, !dbg !14
  %257 = add i64 %133, %256, !dbg !14
  %258 = add i64 %257, %10, !dbg !14
  %259 = add i64 %258, %240, !dbg !14
  %260 = add i64 %259, %30, !dbg !14
  br label %261, !dbg !14

261:                                              ; preds = %280, %254
  %262 = phi i64 [ %281, %280 ], [ 0, %254 ], !dbg !14
  %263 = phi <4 x half> [ %279, %280 ], [ zeroinitializer, %254 ], !dbg !14
  %264 = icmp slt i64 %262, 4, !dbg !14
  br i1 %264, label %265, label %282, !dbg !14

265:                                              ; preds = %261
  %266 = add i64 %249, %262, !dbg !14
  %267 = icmp slt i64 %266, 128, !dbg !14
  br i1 %267, label %268, label %277, !dbg !14

268:                                              ; preds = %265
  %269 = mul i64 %225, 524288, !dbg !14
  %270 = add i64 0, %269, !dbg !14
  %271 = mul i64 %266, 4096, !dbg !14
  %272 = add i64 %270, %271, !dbg !14
  %273 = add i64 %272, %260, !dbg !14
  %274 = getelementptr half, ptr addrspace(1) %1, i64 %273, !dbg !14
  %275 = load half, ptr addrspace(1) %274, align 2, !dbg !14
  %276 = insertelement <4 x half> %263, half %275, i64 %262, !dbg !14
  br label %278, !dbg !14

277:                                              ; preds = %265
  br label %278, !dbg !14

278:                                              ; preds = %268, %277
  %279 = phi <4 x half> [ %263, %277 ], [ %276, %268 ], !dbg !14
  br label %280, !dbg !14

280:                                              ; preds = %278
  %281 = add i64 %262, 1, !dbg !14
  br label %261, !dbg !14

282:                                              ; preds = %261
  %283 = add i64 %256, %10, !dbg !14
  %284 = add i64 %283, %30, !dbg !14
  %285 = extractelement <4 x half> %263, i64 0, !dbg !14
  %286 = insertelement <1 x half> poison, half %285, i64 0, !dbg !14
  %287 = insertvalue [4 x <1 x half>] poison, <1 x half> %286, 0, !dbg !14
  %288 = extractelement <4 x half> %263, i64 1, !dbg !14
  %289 = insertelement <1 x half> poison, half %288, i64 0, !dbg !14
  %290 = insertvalue [4 x <1 x half>] %287, <1 x half> %289, 1, !dbg !14
  %291 = extractelement <4 x half> %263, i64 2, !dbg !14
  %292 = insertelement <1 x half> poison, half %291, i64 0, !dbg !14
  %293 = insertvalue [4 x <1 x half>] %290, <1 x half> %292, 2, !dbg !14
  %294 = extractelement <4 x half> %263, i64 3, !dbg !14
  %295 = insertelement <1 x half> poison, half %294, i64 0, !dbg !14
  %296 = insertvalue [4 x <1 x half>] %293, <1 x half> %295, 3, !dbg !14
  store [4 x <1 x half>] %296, ptr %255, align 2, !dbg !14
  br label %297, !dbg !14

297:                                              ; preds = %309, %282
  %298 = phi i64 [ %310, %309 ], [ 0, %282 ], !dbg !14
  %299 = icmp slt i64 %298, 4, !dbg !14
  br i1 %299, label %300, label %311, !dbg !14

300:                                              ; preds = %297
  %301 = add i64 %250, %298, !dbg !14
  %302 = icmp slt i64 %301, 32, !dbg !14
  br i1 %302, label %303, label %309, !dbg !14

303:                                              ; preds = %300
  %304 = getelementptr <1 x half>, ptr %255, i64 %298, !dbg !14
  %305 = load <1 x half>, ptr %304, align 2, !dbg !14
  %306 = mul i64 %301, 128, !dbg !14
  %307 = add i64 %306, %284, !dbg !14
  %308 = getelementptr half, ptr addrspace(3) @shm_2, i64 %307, !dbg !14
  store <1 x half> %305, ptr addrspace(3) %308, align 2, !dbg !14
  br label %309, !dbg !14

309:                                              ; preds = %303, %300
  %310 = add i64 %298, 1, !dbg !14
  br label %297, !dbg !14

311:                                              ; preds = %297
  %312 = add i64 %252, 1, !dbg !14
  br label %251, !dbg !14

313:                                              ; preds = %251
  %314 = add i64 %242, 1, !dbg !14
  br label %241, !dbg !14

315:                                              ; preds = %241
  br label %316, !dbg !16

316:                                              ; preds = %374, %315
  %317 = phi i64 [ %451, %374 ], [ 0, %315 ], !dbg !16
  %318 = phi [4 x <8 x float>] [ %450, %374 ], [ zeroinitializer, %315 ], !dbg !16
  %319 = icmp slt i64 %317, 8, !dbg !16
  br i1 %319, label %320, label %452, !dbg !16

320:                                              ; preds = %316
  %321 = mul nsw i64 %317, 16, !dbg !16
  %322 = add i64 %321, %10, !dbg !16
  %323 = icmp slt i64 %317, 0, !dbg !16
  %324 = sub i64 -1, %317, !dbg !16
  %325 = select i1 %323, i64 %324, i64 %317, !dbg !16
  %326 = sdiv i64 %325, 2, !dbg !16
  %327 = sub i64 -1, %326, !dbg !16
  %328 = select i1 %323, i64 %327, i64 %326, !dbg !16
  %329 = mul nsw i64 %328, -32, !dbg !16
  %330 = add i64 %322, %329, !dbg !16
  %331 = add i64 %330, %30, !dbg !16
  br label %332, !dbg !16

332:                                              ; preds = %336, %320
  %333 = phi i64 [ %373, %336 ], [ 0, %320 ], !dbg !16
  %334 = phi [1 x <4 x float>] [ %372, %336 ], [ zeroinitializer, %320 ], !dbg !16
  %335 = icmp slt i64 %333, 8, !dbg !16
  br i1 %335, label %336, label %374, !dbg !16

336:                                              ; preds = %332
  %337 = mul nsw i64 %333, 16, !dbg !16
  %338 = add i64 %337, %32, !dbg !16
  %339 = mul i64 %338, 32, !dbg !16
  %340 = add i64 %339, %331, !dbg !16
  %341 = getelementptr half, ptr addrspace(3) @shm_1, i64 %340, !dbg !16
  %342 = load half, ptr addrspace(3) %341, align 2, !dbg !16
  %343 = insertelement <1 x half> zeroinitializer, half %342, i64 0, !dbg !16
  %344 = add i64 %338, 1, !dbg !16
  %345 = mul i64 %344, 32, !dbg !16
  %346 = add i64 %345, %331, !dbg !16
  %347 = getelementptr half, ptr addrspace(3) @shm_1, i64 %346, !dbg !16
  %348 = load half, ptr addrspace(3) %347, align 2, !dbg !16
  %349 = insertelement <1 x half> zeroinitializer, half %348, i64 0, !dbg !16
  %350 = add i64 %338, 2, !dbg !16
  %351 = mul i64 %350, 32, !dbg !16
  %352 = add i64 %351, %331, !dbg !16
  %353 = getelementptr half, ptr addrspace(3) @shm_1, i64 %352, !dbg !16
  %354 = load half, ptr addrspace(3) %353, align 2, !dbg !16
  %355 = insertelement <1 x half> zeroinitializer, half %354, i64 0, !dbg !16
  %356 = add i64 %338, 3, !dbg !16
  %357 = mul i64 %356, 32, !dbg !16
  %358 = add i64 %357, %331, !dbg !16
  %359 = getelementptr half, ptr addrspace(3) @shm_1, i64 %358, !dbg !16
  %360 = load half, ptr addrspace(3) %359, align 2, !dbg !16
  %361 = insertelement <1 x half> zeroinitializer, half %360, i64 0, !dbg !16
  %362 = shufflevector <1 x half> %343, <1 x half> %343, <4 x i32> zeroinitializer, !dbg !16
  %363 = shufflevector <4 x half> %362, <4 x half> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !16
  %364 = shufflevector <1 x half> %349, <1 x half> %349, <4 x i32> zeroinitializer, !dbg !16
  %365 = shufflevector <4 x half> %364, <4 x half> %363, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !16
  %366 = shufflevector <1 x half> %355, <1 x half> %355, <4 x i32> zeroinitializer, !dbg !16
  %367 = shufflevector <4 x half> %366, <4 x half> %365, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !16
  %368 = shufflevector <1 x half> %361, <1 x half> %361, <4 x i32> zeroinitializer, !dbg !16
  %369 = shufflevector <4 x half> %368, <4 x half> %367, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !16
  %370 = extractvalue [1 x <4 x float>] %334, 0, !dbg !16
  %371 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %92, <4 x half> %369, <4 x float> %370), !dbg !16
  %372 = insertvalue [1 x <4 x float>] poison, <4 x float> %371, 0, !dbg !16
  %373 = add i64 %333, 1, !dbg !16
  br label %332, !dbg !16

374:                                              ; preds = %332
  %375 = extractvalue [1 x <4 x float>] %334, 0, !dbg !16
  %376 = extractelement <4 x float> %375, i64 0, !dbg !16
  %377 = mul nsw i64 %317, 4, !dbg !16
  %378 = mul nsw i64 %328, -8, !dbg !16
  %379 = add i64 %377, %378, !dbg !16
  %380 = extractvalue [4 x <8 x float>] %318, 0, !dbg !16
  %381 = shufflevector <8 x float> %380, <8 x float> %380, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %382 = shufflevector <32 x float> %381, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %383 = extractvalue [4 x <8 x float>] %318, 1, !dbg !16
  %384 = shufflevector <8 x float> %383, <8 x float> %383, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %385 = shufflevector <32 x float> %384, <32 x float> %382, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %386 = extractvalue [4 x <8 x float>] %318, 2, !dbg !16
  %387 = shufflevector <8 x float> %386, <8 x float> %386, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %388 = shufflevector <32 x float> %387, <32 x float> %385, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %389 = extractvalue [4 x <8 x float>] %318, 3, !dbg !16
  %390 = shufflevector <8 x float> %389, <8 x float> %389, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %391 = shufflevector <32 x float> %390, <32 x float> %388, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %392 = mul i64 %328, 8, !dbg !16
  %393 = add i64 %392, %379, !dbg !16
  %394 = insertelement <32 x float> %391, float %376, i64 %393, !dbg !16
  %395 = shufflevector <32 x float> %394, <32 x float> %394, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %396 = shufflevector <32 x float> %394, <32 x float> %394, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %397 = shufflevector <32 x float> %394, <32 x float> %394, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !16
  %398 = shufflevector <32 x float> %394, <32 x float> %394, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %399 = extractelement <4 x float> %375, i64 1, !dbg !16
  %400 = add i64 %379, 1, !dbg !16
  %401 = shufflevector <8 x float> %395, <8 x float> %395, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %402 = shufflevector <32 x float> %401, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %403 = shufflevector <8 x float> %396, <8 x float> %396, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %404 = shufflevector <32 x float> %403, <32 x float> %402, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %405 = shufflevector <8 x float> %397, <8 x float> %397, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %406 = shufflevector <32 x float> %405, <32 x float> %404, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %407 = shufflevector <8 x float> %398, <8 x float> %398, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %408 = shufflevector <32 x float> %407, <32 x float> %406, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %409 = add i64 %392, %400, !dbg !16
  %410 = insertelement <32 x float> %408, float %399, i64 %409, !dbg !16
  %411 = shufflevector <32 x float> %410, <32 x float> %410, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %412 = shufflevector <32 x float> %410, <32 x float> %410, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %413 = shufflevector <32 x float> %410, <32 x float> %410, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !16
  %414 = shufflevector <32 x float> %410, <32 x float> %410, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %415 = extractelement <4 x float> %375, i64 2, !dbg !16
  %416 = add i64 %379, 2, !dbg !16
  %417 = shufflevector <8 x float> %411, <8 x float> %411, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %418 = shufflevector <32 x float> %417, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %419 = shufflevector <8 x float> %412, <8 x float> %412, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %420 = shufflevector <32 x float> %419, <32 x float> %418, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %421 = shufflevector <8 x float> %413, <8 x float> %413, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %422 = shufflevector <32 x float> %421, <32 x float> %420, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %423 = shufflevector <8 x float> %414, <8 x float> %414, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %424 = shufflevector <32 x float> %423, <32 x float> %422, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %425 = add i64 %392, %416, !dbg !16
  %426 = insertelement <32 x float> %424, float %415, i64 %425, !dbg !16
  %427 = shufflevector <32 x float> %426, <32 x float> %426, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %428 = shufflevector <32 x float> %426, <32 x float> %426, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %429 = shufflevector <32 x float> %426, <32 x float> %426, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !16
  %430 = shufflevector <32 x float> %426, <32 x float> %426, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %431 = extractelement <4 x float> %375, i64 3, !dbg !16
  %432 = add i64 %379, 3, !dbg !16
  %433 = shufflevector <8 x float> %427, <8 x float> %427, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %434 = shufflevector <32 x float> %433, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %435 = shufflevector <8 x float> %428, <8 x float> %428, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %436 = shufflevector <32 x float> %435, <32 x float> %434, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %437 = shufflevector <8 x float> %429, <8 x float> %429, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %438 = shufflevector <32 x float> %437, <32 x float> %436, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !16
  %439 = shufflevector <8 x float> %430, <8 x float> %430, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !16
  %440 = shufflevector <32 x float> %439, <32 x float> %438, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %441 = add i64 %392, %432, !dbg !16
  %442 = insertelement <32 x float> %440, float %431, i64 %441, !dbg !16
  %443 = shufflevector <32 x float> %442, <32 x float> %442, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %444 = insertvalue [4 x <8 x float>] poison, <8 x float> %443, 0, !dbg !16
  %445 = shufflevector <32 x float> %442, <32 x float> %442, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !16
  %446 = insertvalue [4 x <8 x float>] %444, <8 x float> %445, 1, !dbg !16
  %447 = shufflevector <32 x float> %442, <32 x float> %442, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !16
  %448 = insertvalue [4 x <8 x float>] %446, <8 x float> %447, 2, !dbg !16
  %449 = shufflevector <32 x float> %442, <32 x float> %442, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !16
  %450 = insertvalue [4 x <8 x float>] %448, <8 x float> %449, 3, !dbg !16
  %451 = add i64 %317, 1, !dbg !16
  br label %316, !dbg !16

452:                                              ; preds = %316
  br label %453, !dbg !17

453:                                              ; preds = %552, %452
  %454 = phi i64 [ %553, %552 ], [ 0, %452 ], !dbg !17
  %455 = phi [4 x <8 x float>] [ %465, %552 ], [ zeroinitializer, %452 ], !dbg !17
  %456 = icmp slt i64 %454, 4, !dbg !17
  br i1 %456, label %457, label %554, !dbg !17

457:                                              ; preds = %453
  %458 = mul nsw i64 %454, 16, !dbg !17
  %459 = add i64 %458, %10, !dbg !17
  %460 = add i64 %459, %11, !dbg !17
  %461 = add i64 %460, %30, !dbg !17
  %462 = add i64 %461, 1, !dbg !18
  br label %463, !dbg !17

463:                                              ; preds = %467, %457
  %464 = phi i64 [ %551, %467 ], [ 0, %457 ], !dbg !17
  %465 = phi [4 x <8 x float>] [ %550, %467 ], [ %455, %457 ], !dbg !17
  %466 = icmp slt i64 %464, 2, !dbg !17
  br i1 %466, label %467, label %552, !dbg !17

467:                                              ; preds = %463
  %468 = mul nsw i64 %464, 4, !dbg !17
  %469 = mul nsw i64 %464, 16, !dbg !17
  %470 = add i64 %133, %469, !dbg !17
  %471 = add i64 %470, %29, !dbg !17
  %472 = icmp ule i64 %462, %471, !dbg !19
  %473 = select i1 %472, float 0xFFF0000000000000, float 0.000000e+00, !dbg !20
  %474 = extractvalue [4 x <8 x float>] %465, 0, !dbg !17
  %475 = shufflevector <8 x float> %474, <8 x float> %474, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %476 = shufflevector <32 x float> %475, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %477 = extractvalue [4 x <8 x float>] %465, 1, !dbg !17
  %478 = shufflevector <8 x float> %477, <8 x float> %477, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %479 = shufflevector <32 x float> %478, <32 x float> %476, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %480 = extractvalue [4 x <8 x float>] %465, 2, !dbg !17
  %481 = shufflevector <8 x float> %480, <8 x float> %480, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %482 = shufflevector <32 x float> %481, <32 x float> %479, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %483 = extractvalue [4 x <8 x float>] %465, 3, !dbg !17
  %484 = shufflevector <8 x float> %483, <8 x float> %483, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %485 = shufflevector <32 x float> %484, <32 x float> %482, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %486 = mul i64 %454, 8, !dbg !17
  %487 = add i64 %486, %468, !dbg !17
  %488 = insertelement <32 x float> %485, float %473, i64 %487, !dbg !17
  %489 = shufflevector <32 x float> %488, <32 x float> %488, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %490 = shufflevector <32 x float> %488, <32 x float> %488, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %491 = shufflevector <32 x float> %488, <32 x float> %488, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !17
  %492 = shufflevector <32 x float> %488, <32 x float> %488, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %493 = add i64 %468, 1, !dbg !17
  %494 = add i64 %471, 4, !dbg !17
  %495 = icmp ule i64 %462, %494, !dbg !19
  %496 = select i1 %495, float 0xFFF0000000000000, float 0.000000e+00, !dbg !20
  %497 = shufflevector <8 x float> %489, <8 x float> %489, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %498 = shufflevector <32 x float> %497, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %499 = shufflevector <8 x float> %490, <8 x float> %490, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %500 = shufflevector <32 x float> %499, <32 x float> %498, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %501 = shufflevector <8 x float> %491, <8 x float> %491, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %502 = shufflevector <32 x float> %501, <32 x float> %500, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %503 = shufflevector <8 x float> %492, <8 x float> %492, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %504 = shufflevector <32 x float> %503, <32 x float> %502, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %505 = add i64 %486, %493, !dbg !17
  %506 = insertelement <32 x float> %504, float %496, i64 %505, !dbg !17
  %507 = shufflevector <32 x float> %506, <32 x float> %506, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %508 = shufflevector <32 x float> %506, <32 x float> %506, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %509 = shufflevector <32 x float> %506, <32 x float> %506, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !17
  %510 = shufflevector <32 x float> %506, <32 x float> %506, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %511 = add i64 %468, 2, !dbg !17
  %512 = add i64 %471, 8, !dbg !17
  %513 = icmp ule i64 %462, %512, !dbg !19
  %514 = select i1 %513, float 0xFFF0000000000000, float 0.000000e+00, !dbg !20
  %515 = shufflevector <8 x float> %507, <8 x float> %507, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %516 = shufflevector <32 x float> %515, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %517 = shufflevector <8 x float> %508, <8 x float> %508, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %518 = shufflevector <32 x float> %517, <32 x float> %516, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %519 = shufflevector <8 x float> %509, <8 x float> %509, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %520 = shufflevector <32 x float> %519, <32 x float> %518, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %521 = shufflevector <8 x float> %510, <8 x float> %510, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %522 = shufflevector <32 x float> %521, <32 x float> %520, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %523 = add i64 %486, %511, !dbg !17
  %524 = insertelement <32 x float> %522, float %514, i64 %523, !dbg !17
  %525 = shufflevector <32 x float> %524, <32 x float> %524, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %526 = shufflevector <32 x float> %524, <32 x float> %524, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %527 = shufflevector <32 x float> %524, <32 x float> %524, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !17
  %528 = shufflevector <32 x float> %524, <32 x float> %524, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %529 = add i64 %468, 3, !dbg !17
  %530 = add i64 %471, 12, !dbg !17
  %531 = icmp ule i64 %462, %530, !dbg !19
  %532 = select i1 %531, float 0xFFF0000000000000, float 0.000000e+00, !dbg !20
  %533 = shufflevector <8 x float> %525, <8 x float> %525, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %534 = shufflevector <32 x float> %533, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %535 = shufflevector <8 x float> %526, <8 x float> %526, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %536 = shufflevector <32 x float> %535, <32 x float> %534, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %537 = shufflevector <8 x float> %527, <8 x float> %527, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %538 = shufflevector <32 x float> %537, <32 x float> %536, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !17
  %539 = shufflevector <8 x float> %528, <8 x float> %528, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %540 = shufflevector <32 x float> %539, <32 x float> %538, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %541 = add i64 %486, %529, !dbg !17
  %542 = insertelement <32 x float> %540, float %532, i64 %541, !dbg !17
  %543 = shufflevector <32 x float> %542, <32 x float> %542, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %544 = insertvalue [4 x <8 x float>] poison, <8 x float> %543, 0, !dbg !17
  %545 = shufflevector <32 x float> %542, <32 x float> %542, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %546 = insertvalue [4 x <8 x float>] %544, <8 x float> %545, 1, !dbg !17
  %547 = shufflevector <32 x float> %542, <32 x float> %542, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>, !dbg !17
  %548 = insertvalue [4 x <8 x float>] %546, <8 x float> %547, 2, !dbg !17
  %549 = shufflevector <32 x float> %542, <32 x float> %542, <8 x i32> <i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %550 = insertvalue [4 x <8 x float>] %548, <8 x float> %549, 3, !dbg !17
  %551 = add i64 %464, 1, !dbg !17
  br label %463, !dbg !17

552:                                              ; preds = %463
  %553 = add i64 %454, 1, !dbg !17
  br label %453, !dbg !17

554:                                              ; preds = %453
  %555 = extractvalue [4 x <8 x float>] %318, 0, !dbg !21
  %556 = extractvalue [4 x <8 x float>] %455, 0, !dbg !21
  %557 = fadd <8 x float> %555, %556, !dbg !21
  %558 = extractvalue [4 x <8 x float>] %318, 1, !dbg !21
  %559 = extractvalue [4 x <8 x float>] %455, 1, !dbg !21
  %560 = fadd <8 x float> %558, %559, !dbg !21
  %561 = extractvalue [4 x <8 x float>] %318, 2, !dbg !21
  %562 = extractvalue [4 x <8 x float>] %455, 2, !dbg !21
  %563 = fadd <8 x float> %561, %562, !dbg !21
  %564 = extractvalue [4 x <8 x float>] %318, 3, !dbg !21
  %565 = extractvalue [4 x <8 x float>] %455, 3, !dbg !21
  %566 = fadd <8 x float> %564, %565, !dbg !21
  %567 = extractelement <8 x float> %557, i64 0, !dbg !22
  %568 = call float @__ocml_exp2_f32(float %567), !dbg !22
  %569 = insertelement <8 x float> poison, float %568, i64 0, !dbg !22
  %570 = extractelement <8 x float> %557, i64 1, !dbg !22
  %571 = call float @__ocml_exp2_f32(float %570), !dbg !22
  %572 = insertelement <8 x float> %569, float %571, i64 1, !dbg !22
  %573 = extractelement <8 x float> %557, i64 2, !dbg !22
  %574 = call float @__ocml_exp2_f32(float %573), !dbg !22
  %575 = insertelement <8 x float> %572, float %574, i64 2, !dbg !22
  %576 = extractelement <8 x float> %557, i64 3, !dbg !22
  %577 = call float @__ocml_exp2_f32(float %576), !dbg !22
  %578 = insertelement <8 x float> %575, float %577, i64 3, !dbg !22
  %579 = extractelement <8 x float> %557, i64 4, !dbg !22
  %580 = call float @__ocml_exp2_f32(float %579), !dbg !22
  %581 = insertelement <8 x float> %578, float %580, i64 4, !dbg !22
  %582 = extractelement <8 x float> %557, i64 5, !dbg !22
  %583 = call float @__ocml_exp2_f32(float %582), !dbg !22
  %584 = insertelement <8 x float> %581, float %583, i64 5, !dbg !22
  %585 = extractelement <8 x float> %557, i64 6, !dbg !22
  %586 = call float @__ocml_exp2_f32(float %585), !dbg !22
  %587 = insertelement <8 x float> %584, float %586, i64 6, !dbg !22
  %588 = extractelement <8 x float> %557, i64 7, !dbg !22
  %589 = call float @__ocml_exp2_f32(float %588), !dbg !22
  %590 = insertelement <8 x float> %587, float %589, i64 7, !dbg !22
  %591 = extractelement <8 x float> %560, i64 0, !dbg !22
  %592 = call float @__ocml_exp2_f32(float %591), !dbg !22
  %593 = insertelement <8 x float> poison, float %592, i64 0, !dbg !22
  %594 = extractelement <8 x float> %560, i64 1, !dbg !22
  %595 = call float @__ocml_exp2_f32(float %594), !dbg !22
  %596 = insertelement <8 x float> %593, float %595, i64 1, !dbg !22
  %597 = extractelement <8 x float> %560, i64 2, !dbg !22
  %598 = call float @__ocml_exp2_f32(float %597), !dbg !22
  %599 = insertelement <8 x float> %596, float %598, i64 2, !dbg !22
  %600 = extractelement <8 x float> %560, i64 3, !dbg !22
  %601 = call float @__ocml_exp2_f32(float %600), !dbg !22
  %602 = insertelement <8 x float> %599, float %601, i64 3, !dbg !22
  %603 = extractelement <8 x float> %560, i64 4, !dbg !22
  %604 = call float @__ocml_exp2_f32(float %603), !dbg !22
  %605 = insertelement <8 x float> %602, float %604, i64 4, !dbg !22
  %606 = extractelement <8 x float> %560, i64 5, !dbg !22
  %607 = call float @__ocml_exp2_f32(float %606), !dbg !22
  %608 = insertelement <8 x float> %605, float %607, i64 5, !dbg !22
  %609 = extractelement <8 x float> %560, i64 6, !dbg !22
  %610 = call float @__ocml_exp2_f32(float %609), !dbg !22
  %611 = insertelement <8 x float> %608, float %610, i64 6, !dbg !22
  %612 = extractelement <8 x float> %560, i64 7, !dbg !22
  %613 = call float @__ocml_exp2_f32(float %612), !dbg !22
  %614 = insertelement <8 x float> %611, float %613, i64 7, !dbg !22
  %615 = extractelement <8 x float> %563, i64 0, !dbg !22
  %616 = call float @__ocml_exp2_f32(float %615), !dbg !22
  %617 = insertelement <8 x float> poison, float %616, i64 0, !dbg !22
  %618 = extractelement <8 x float> %563, i64 1, !dbg !22
  %619 = call float @__ocml_exp2_f32(float %618), !dbg !22
  %620 = insertelement <8 x float> %617, float %619, i64 1, !dbg !22
  %621 = extractelement <8 x float> %563, i64 2, !dbg !22
  %622 = call float @__ocml_exp2_f32(float %621), !dbg !22
  %623 = insertelement <8 x float> %620, float %622, i64 2, !dbg !22
  %624 = extractelement <8 x float> %563, i64 3, !dbg !22
  %625 = call float @__ocml_exp2_f32(float %624), !dbg !22
  %626 = insertelement <8 x float> %623, float %625, i64 3, !dbg !22
  %627 = extractelement <8 x float> %563, i64 4, !dbg !22
  %628 = call float @__ocml_exp2_f32(float %627), !dbg !22
  %629 = insertelement <8 x float> %626, float %628, i64 4, !dbg !22
  %630 = extractelement <8 x float> %563, i64 5, !dbg !22
  %631 = call float @__ocml_exp2_f32(float %630), !dbg !22
  %632 = insertelement <8 x float> %629, float %631, i64 5, !dbg !22
  %633 = extractelement <8 x float> %563, i64 6, !dbg !22
  %634 = call float @__ocml_exp2_f32(float %633), !dbg !22
  %635 = insertelement <8 x float> %632, float %634, i64 6, !dbg !22
  %636 = extractelement <8 x float> %563, i64 7, !dbg !22
  %637 = call float @__ocml_exp2_f32(float %636), !dbg !22
  %638 = insertelement <8 x float> %635, float %637, i64 7, !dbg !22
  %639 = extractelement <8 x float> %566, i64 0, !dbg !22
  %640 = call float @__ocml_exp2_f32(float %639), !dbg !22
  %641 = insertelement <8 x float> poison, float %640, i64 0, !dbg !22
  %642 = extractelement <8 x float> %566, i64 1, !dbg !22
  %643 = call float @__ocml_exp2_f32(float %642), !dbg !22
  %644 = insertelement <8 x float> %641, float %643, i64 1, !dbg !22
  %645 = extractelement <8 x float> %566, i64 2, !dbg !22
  %646 = call float @__ocml_exp2_f32(float %645), !dbg !22
  %647 = insertelement <8 x float> %644, float %646, i64 2, !dbg !22
  %648 = extractelement <8 x float> %566, i64 3, !dbg !22
  %649 = call float @__ocml_exp2_f32(float %648), !dbg !22
  %650 = insertelement <8 x float> %647, float %649, i64 3, !dbg !22
  %651 = extractelement <8 x float> %566, i64 4, !dbg !22
  %652 = call float @__ocml_exp2_f32(float %651), !dbg !22
  %653 = insertelement <8 x float> %650, float %652, i64 4, !dbg !22
  %654 = extractelement <8 x float> %566, i64 5, !dbg !22
  %655 = call float @__ocml_exp2_f32(float %654), !dbg !22
  %656 = insertelement <8 x float> %653, float %655, i64 5, !dbg !22
  %657 = extractelement <8 x float> %566, i64 6, !dbg !22
  %658 = call float @__ocml_exp2_f32(float %657), !dbg !22
  %659 = insertelement <8 x float> %656, float %658, i64 6, !dbg !22
  %660 = extractelement <8 x float> %566, i64 7, !dbg !22
  %661 = call float @__ocml_exp2_f32(float %660), !dbg !22
  %662 = insertelement <8 x float> %659, float %661, i64 7, !dbg !22
  br label %663, !dbg !11

663:                                              ; preds = %729, %554
  %664 = phi i64 [ %730, %729 ], [ 0, %554 ], !dbg !11
  %665 = icmp slt i64 %664, 4, !dbg !11
  br i1 %665, label %666, label %731, !dbg !11

666:                                              ; preds = %663
  store float 0.000000e+00, ptr addrspace(5) %93, align 4, !dbg !11
  br label %667, !dbg !11

667:                                              ; preds = %670, %666
  %668 = phi i64 [ %700, %670 ], [ 0, %666 ], !dbg !11
  %669 = icmp slt i64 %668, 2, !dbg !11
  br i1 %669, label %670, label %701, !dbg !11

670:                                              ; preds = %667
  %671 = mul nsw i64 %668, 4, !dbg !11
  %672 = load float, ptr addrspace(5) %93, align 4, !dbg !11
  %673 = shufflevector <8 x float> %590, <8 x float> %590, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !11
  %674 = shufflevector <32 x float> %673, <32 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !11
  %675 = shufflevector <8 x float> %614, <8 x float> %614, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !11
  %676 = shufflevector <32 x float> %675, <32 x float> %674, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !11
  %677 = shufflevector <8 x float> %638, <8 x float> %638, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !11
  %678 = shufflevector <32 x float> %677, <32 x float> %676, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !11
  %679 = shufflevector <8 x float> %662, <8 x float> %662, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !11
  %680 = shufflevector <32 x float> %679, <32 x float> %678, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %681 = mul i64 %664, 8, !dbg !11
  %682 = add i64 %681, %671, !dbg !11
  %683 = extractelement <32 x float> %680, i64 %682, !dbg !11
  %684 = fadd float %672, %683, !dbg !11
  store float %684, ptr addrspace(5) %93, align 4, !dbg !11
  %685 = add i64 %671, 1, !dbg !11
  %686 = load float, ptr addrspace(5) %93, align 4, !dbg !11
  %687 = add i64 %681, %685, !dbg !11
  %688 = extractelement <32 x float> %680, i64 %687, !dbg !11
  %689 = fadd float %686, %688, !dbg !11
  store float %689, ptr addrspace(5) %93, align 4, !dbg !11
  %690 = add i64 %671, 2, !dbg !11
  %691 = load float, ptr addrspace(5) %93, align 4, !dbg !11
  %692 = add i64 %681, %690, !dbg !11
  %693 = extractelement <32 x float> %680, i64 %692, !dbg !11
  %694 = fadd float %691, %693, !dbg !11
  store float %694, ptr addrspace(5) %93, align 4, !dbg !11
  %695 = add i64 %671, 3, !dbg !11
  %696 = load float, ptr addrspace(5) %93, align 4, !dbg !11
  %697 = add i64 %681, %695, !dbg !11
  %698 = extractelement <32 x float> %680, i64 %697, !dbg !11
  %699 = fadd float %696, %698, !dbg !11
  store float %699, ptr addrspace(5) %93, align 4, !dbg !11
  %700 = add i64 %668, 1, !dbg !11
  br label %667, !dbg !11

701:                                              ; preds = %667
  %702 = load float, ptr addrspace(5) %93, align 4, !dbg !11
  %703 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !11
  %704 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %703), !dbg !11
  %705 = add i32 %704, 64, !dbg !11
  %706 = xor i32 %704, 16, !dbg !11
  %707 = and i32 %705, -64, !dbg !11
  %708 = icmp slt i32 %706, %707, !dbg !11
  %709 = select i1 %708, i32 %706, i32 %704, !dbg !11
  %710 = shl i32 %709, 2, !dbg !11
  %711 = bitcast float %702 to i32, !dbg !11
  %712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %710, i32 %711), !dbg !11
  %713 = bitcast i32 %712 to float, !dbg !11
  %714 = fadd float %702, %713, !dbg !11
  %715 = xor i32 %704, 32, !dbg !11
  %716 = icmp slt i32 %715, %707, !dbg !11
  %717 = select i1 %716, i32 %715, i32 %704, !dbg !11
  %718 = shl i32 %717, 2, !dbg !11
  %719 = bitcast float %714 to i32, !dbg !11
  %720 = call i32 @llvm.amdgcn.ds.bpermute(i32 %718, i32 %719), !dbg !11
  %721 = bitcast i32 %720 to float, !dbg !11
  %722 = fadd float %714, %721, !dbg !11
  br i1 %106, label %723, label %729, !dbg !11

723:                                              ; preds = %701
  %724 = mul nsw i64 %664, 16, !dbg !11
  %725 = add i64 %724, %10, !dbg !11
  %726 = add i64 %725, %30, !dbg !11
  %727 = add i64 %726, 0, !dbg !11
  %728 = getelementptr float, ptr addrspace(3) @shm_3, i64 %727, !dbg !11
  store float %722, ptr addrspace(3) %728, align 4, !dbg !11
  br label %729, !dbg !11

729:                                              ; preds = %723, %701
  %730 = add i64 %664, 1, !dbg !11
  br label %663, !dbg !11

731:                                              ; preds = %663
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %732 = add i64 %71, %29, !dbg !11
  %733 = getelementptr float, ptr addrspace(3) @shm_3, i64 %732, !dbg !11
  %734 = load float, ptr addrspace(3) %733, align 4, !dbg !11
  %735 = insertelement <1 x float> zeroinitializer, float %734, i64 0, !dbg !11
  %736 = add i64 %112, %29, !dbg !11
  %737 = getelementptr float, ptr addrspace(3) @shm_3, i64 %736, !dbg !11
  %738 = load float, ptr addrspace(3) %737, align 4, !dbg !11
  %739 = insertelement <1 x float> zeroinitializer, float %738, i64 0, !dbg !11
  %740 = add i64 %113, %29, !dbg !11
  %741 = getelementptr float, ptr addrspace(3) @shm_3, i64 %740, !dbg !11
  %742 = load float, ptr addrspace(3) %741, align 4, !dbg !11
  %743 = insertelement <1 x float> zeroinitializer, float %742, i64 0, !dbg !11
  %744 = add i64 %114, %29, !dbg !11
  %745 = getelementptr float, ptr addrspace(3) @shm_3, i64 %744, !dbg !11
  %746 = load float, ptr addrspace(3) %745, align 4, !dbg !11
  %747 = insertelement <1 x float> zeroinitializer, float %746, i64 0, !dbg !11
  %748 = shufflevector <1 x float> %735, <1 x float> %735, <4 x i32> zeroinitializer, !dbg !11
  %749 = shufflevector <4 x float> %748, <4 x float> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !11
  %750 = shufflevector <1 x float> %739, <1 x float> %739, <4 x i32> zeroinitializer, !dbg !11
  %751 = shufflevector <4 x float> %750, <4 x float> %749, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !11
  %752 = shufflevector <1 x float> %743, <1 x float> %743, <4 x i32> zeroinitializer, !dbg !11
  %753 = shufflevector <4 x float> %752, <4 x float> %751, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !11
  %754 = shufflevector <1 x float> %747, <1 x float> %747, <4 x i32> zeroinitializer, !dbg !11
  %755 = shufflevector <4 x float> %754, <4 x float> %753, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !11
  %756 = extractvalue [4 x <1 x float>] %118, 0, !dbg !11
  %757 = shufflevector <1 x float> %756, <1 x float> %756, <4 x i32> zeroinitializer, !dbg !11
  %758 = shufflevector <4 x float> %757, <4 x float> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !11
  %759 = extractvalue [4 x <1 x float>] %118, 1, !dbg !11
  %760 = shufflevector <1 x float> %759, <1 x float> %759, <4 x i32> zeroinitializer, !dbg !11
  %761 = shufflevector <4 x float> %760, <4 x float> %758, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !11
  %762 = extractvalue [4 x <1 x float>] %118, 2, !dbg !11
  %763 = shufflevector <1 x float> %762, <1 x float> %762, <4 x i32> zeroinitializer, !dbg !11
  %764 = shufflevector <4 x float> %763, <4 x float> %761, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !11
  %765 = extractvalue [4 x <1 x float>] %118, 3, !dbg !11
  %766 = shufflevector <1 x float> %765, <1 x float> %765, <4 x i32> zeroinitializer, !dbg !11
  %767 = shufflevector <4 x float> %766, <4 x float> %764, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !11
  %768 = fadd <4 x float> %755, %767, !dbg !11
  %769 = shufflevector <4 x float> %768, <4 x float> %768, <1 x i32> zeroinitializer, !dbg !11
  %770 = insertvalue [4 x <1 x float>] poison, <1 x float> %769, 0, !dbg !11
  %771 = shufflevector <4 x float> %768, <4 x float> %768, <1 x i32> <i32 1>, !dbg !11
  %772 = insertvalue [4 x <1 x float>] %770, <1 x float> %771, 1, !dbg !11
  %773 = shufflevector <4 x float> %768, <4 x float> %768, <1 x i32> <i32 2>, !dbg !11
  %774 = insertvalue [4 x <1 x float>] %772, <1 x float> %773, 2, !dbg !11
  %775 = shufflevector <4 x float> %768, <4 x float> %768, <1 x i32> <i32 3>, !dbg !11
  %776 = insertvalue [4 x <1 x float>] %774, <1 x float> %775, 3, !dbg !11
  br label %777, !dbg !12

777:                                              ; preds = %803, %731
  %778 = phi i64 [ %804, %803 ], [ 0, %731 ], !dbg !12
  %779 = icmp slt i64 %778, 4, !dbg !12
  br i1 %779, label %780, label %805, !dbg !12

780:                                              ; preds = %777
  %781 = mul nsw i64 %778, 16, !dbg !12
  %782 = add i64 %781, %10, !dbg !12
  %783 = add i64 %782, %30, !dbg !12
  br label %784, !dbg !12

784:                                              ; preds = %787, %780
  %785 = phi i64 [ %802, %787 ], [ 0, %780 ], !dbg !12
  %786 = icmp slt i64 %785, 2, !dbg !12
  br i1 %786, label %787, label %803, !dbg !12

787:                                              ; preds = %784
  %788 = getelementptr float, ptr addrspace(5) %94, i64 0, !dbg !12
  %789 = load <4 x float>, ptr addrspace(5) %788, align 4, !dbg !12
  %790 = fptrunc <4 x float> %789 to <4 x half>, !dbg !12
  %791 = getelementptr half, ptr addrspace(5) %95, i64 0, !dbg !12
  store <4 x half> %790, ptr addrspace(5) %791, align 2, !dbg !12
  %792 = load <4 x half>, ptr addrspace(5) %791, align 2, !dbg !12
  %793 = mul nsw i64 %785, 16, !dbg !12
  %794 = add i64 %793, %32, !dbg !12
  %795 = sub i64 32, %794, !dbg !12
  %796 = insertelement <4 x i64> poison, i64 %795, i32 0, !dbg !12
  %797 = shufflevector <4 x i64> %796, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !12
  %798 = icmp sgt <4 x i64> %797, <i64 0, i64 1, i64 2, i64 3>, !dbg !12
  %799 = mul i64 %783, 32, !dbg !12
  %800 = add i64 %799, %794, !dbg !12
  %801 = getelementptr half, ptr addrspace(3) @shm_4, i64 %800, !dbg !12
  call void @llvm.masked.store.v4f16.p3(<4 x half> %792, ptr addrspace(3) %801, i32 2, <4 x i1> %798), !dbg !12
  %802 = add i64 %785, 1, !dbg !12
  br label %784, !dbg !12

803:                                              ; preds = %784
  %804 = add i64 %778, 1, !dbg !12
  br label %777, !dbg !12

805:                                              ; preds = %777
  br label %806, !dbg !23

806:                                              ; preds = %884, %805
  %807 = phi i64 [ %961, %884 ], [ 0, %805 ], !dbg !23
  %808 = phi [4 x <32 x float>] [ %960, %884 ], [ zeroinitializer, %805 ], !dbg !23
  %809 = icmp slt i64 %807, 32, !dbg !23
  br i1 %809, label %810, label %962, !dbg !23

810:                                              ; preds = %806
  %811 = icmp slt i64 %807, 0, !dbg !23
  %812 = sub i64 -1, %807, !dbg !23
  %813 = select i1 %811, i64 %812, i64 %807, !dbg !23
  %814 = sdiv i64 %813, 8, !dbg !23
  %815 = sub i64 -1, %814, !dbg !23
  %816 = select i1 %811, i64 %815, i64 %814, !dbg !23
  %817 = mul nsw i64 %816, 16, !dbg !23
  %818 = add i64 %817, %10, !dbg !23
  %819 = add i64 %818, %30, !dbg !23
  %820 = mul nsw i64 %807, 16, !dbg !23
  %821 = add i64 %820, %10, !dbg !23
  %822 = mul nsw i64 %816, -128, !dbg !23
  %823 = add i64 %821, %822, !dbg !23
  %824 = add i64 %823, %30, !dbg !23
  br label %825, !dbg !23

825:                                              ; preds = %829, %810
  %826 = phi i64 [ %883, %829 ], [ 0, %810 ], !dbg !23
  %827 = phi [1 x <4 x float>] [ %882, %829 ], [ zeroinitializer, %810 ], !dbg !23
  %828 = icmp slt i64 %826, 2, !dbg !23
  br i1 %828, label %829, label %884, !dbg !23

829:                                              ; preds = %825
  %830 = mul nsw i64 %826, 16, !dbg !23
  %831 = add i64 %830, %32, !dbg !23
  %832 = mul i64 %819, 32, !dbg !23
  %833 = add i64 %832, %831, !dbg !23
  %834 = getelementptr half, ptr addrspace(3) @shm_4, i64 %833, !dbg !23
  %835 = load half, ptr addrspace(3) %834, align 2, !dbg !23
  %836 = insertelement <4 x half> zeroinitializer, half %835, i64 0, !dbg !23
  %837 = add i64 %831, 1, !dbg !23
  %838 = add i64 %832, %837, !dbg !23
  %839 = getelementptr half, ptr addrspace(3) @shm_4, i64 %838, !dbg !23
  %840 = load half, ptr addrspace(3) %839, align 2, !dbg !23
  %841 = insertelement <4 x half> %836, half %840, i64 1, !dbg !23
  %842 = add i64 %831, 2, !dbg !23
  %843 = add i64 %832, %842, !dbg !23
  %844 = getelementptr half, ptr addrspace(3) @shm_4, i64 %843, !dbg !23
  %845 = load half, ptr addrspace(3) %844, align 2, !dbg !23
  %846 = insertelement <4 x half> %841, half %845, i64 2, !dbg !23
  %847 = add i64 %831, 3, !dbg !23
  %848 = add i64 %832, %847, !dbg !23
  %849 = getelementptr half, ptr addrspace(3) @shm_4, i64 %848, !dbg !23
  %850 = load half, ptr addrspace(3) %849, align 2, !dbg !23
  %851 = insertelement <4 x half> %846, half %850, i64 3, !dbg !23
  %852 = mul i64 %831, 128, !dbg !23
  %853 = add i64 %852, %824, !dbg !23
  %854 = getelementptr half, ptr addrspace(3) @shm_2, i64 %853, !dbg !23
  %855 = load half, ptr addrspace(3) %854, align 2, !dbg !23
  %856 = insertelement <1 x half> zeroinitializer, half %855, i64 0, !dbg !23
  %857 = mul i64 %837, 128, !dbg !23
  %858 = add i64 %857, %824, !dbg !23
  %859 = getelementptr half, ptr addrspace(3) @shm_2, i64 %858, !dbg !23
  %860 = load half, ptr addrspace(3) %859, align 2, !dbg !23
  %861 = insertelement <1 x half> zeroinitializer, half %860, i64 0, !dbg !23
  %862 = mul i64 %842, 128, !dbg !23
  %863 = add i64 %862, %824, !dbg !23
  %864 = getelementptr half, ptr addrspace(3) @shm_2, i64 %863, !dbg !23
  %865 = load half, ptr addrspace(3) %864, align 2, !dbg !23
  %866 = insertelement <1 x half> zeroinitializer, half %865, i64 0, !dbg !23
  %867 = mul i64 %847, 128, !dbg !23
  %868 = add i64 %867, %824, !dbg !23
  %869 = getelementptr half, ptr addrspace(3) @shm_2, i64 %868, !dbg !23
  %870 = load half, ptr addrspace(3) %869, align 2, !dbg !23
  %871 = insertelement <1 x half> zeroinitializer, half %870, i64 0, !dbg !23
  %872 = shufflevector <1 x half> %856, <1 x half> %856, <4 x i32> zeroinitializer, !dbg !23
  %873 = shufflevector <4 x half> %872, <4 x half> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !23
  %874 = shufflevector <1 x half> %861, <1 x half> %861, <4 x i32> zeroinitializer, !dbg !23
  %875 = shufflevector <4 x half> %874, <4 x half> %873, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !23
  %876 = shufflevector <1 x half> %866, <1 x half> %866, <4 x i32> zeroinitializer, !dbg !23
  %877 = shufflevector <4 x half> %876, <4 x half> %875, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !23
  %878 = shufflevector <1 x half> %871, <1 x half> %871, <4 x i32> zeroinitializer, !dbg !23
  %879 = shufflevector <4 x half> %878, <4 x half> %877, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !23
  %880 = extractvalue [1 x <4 x float>] %827, 0, !dbg !23
  %881 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %851, <4 x half> %879, <4 x float> %880), !dbg !23
  %882 = insertvalue [1 x <4 x float>] poison, <4 x float> %881, 0, !dbg !23
  %883 = add i64 %826, 1, !dbg !23
  br label %825, !dbg !23

884:                                              ; preds = %825
  %885 = extractvalue [1 x <4 x float>] %827, 0, !dbg !23
  %886 = extractelement <4 x float> %885, i64 0, !dbg !23
  %887 = mul nsw i64 %807, 4, !dbg !23
  %888 = mul nsw i64 %816, -32, !dbg !23
  %889 = add i64 %887, %888, !dbg !23
  %890 = extractvalue [4 x <32 x float>] %808, 0, !dbg !23
  %891 = shufflevector <32 x float> %890, <32 x float> %890, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %892 = shufflevector <128 x float> %891, <128 x float> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %893 = extractvalue [4 x <32 x float>] %808, 1, !dbg !23
  %894 = shufflevector <32 x float> %893, <32 x float> %893, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %895 = shufflevector <128 x float> %894, <128 x float> %892, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %896 = extractvalue [4 x <32 x float>] %808, 2, !dbg !23
  %897 = shufflevector <32 x float> %896, <32 x float> %896, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %898 = shufflevector <128 x float> %897, <128 x float> %895, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %899 = extractvalue [4 x <32 x float>] %808, 3, !dbg !23
  %900 = shufflevector <32 x float> %899, <32 x float> %899, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %901 = shufflevector <128 x float> %900, <128 x float> %898, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %902 = mul i64 %816, 32, !dbg !23
  %903 = add i64 %902, %889, !dbg !23
  %904 = insertelement <128 x float> %901, float %886, i64 %903, !dbg !23
  %905 = shufflevector <128 x float> %904, <128 x float> %904, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %906 = shufflevector <128 x float> %904, <128 x float> %904, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !23
  %907 = shufflevector <128 x float> %904, <128 x float> %904, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>, !dbg !23
  %908 = shufflevector <128 x float> %904, <128 x float> %904, <32 x i32> <i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !23
  %909 = extractelement <4 x float> %885, i64 1, !dbg !23
  %910 = add i64 %889, 1, !dbg !23
  %911 = shufflevector <32 x float> %905, <32 x float> %905, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %912 = shufflevector <128 x float> %911, <128 x float> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %913 = shufflevector <32 x float> %906, <32 x float> %906, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %914 = shufflevector <128 x float> %913, <128 x float> %912, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %915 = shufflevector <32 x float> %907, <32 x float> %907, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %916 = shufflevector <128 x float> %915, <128 x float> %914, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %917 = shufflevector <32 x float> %908, <32 x float> %908, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %918 = shufflevector <128 x float> %917, <128 x float> %916, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %919 = add i64 %902, %910, !dbg !23
  %920 = insertelement <128 x float> %918, float %909, i64 %919, !dbg !23
  %921 = shufflevector <128 x float> %920, <128 x float> %920, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %922 = shufflevector <128 x float> %920, <128 x float> %920, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !23
  %923 = shufflevector <128 x float> %920, <128 x float> %920, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>, !dbg !23
  %924 = shufflevector <128 x float> %920, <128 x float> %920, <32 x i32> <i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !23
  %925 = extractelement <4 x float> %885, i64 2, !dbg !23
  %926 = add i64 %889, 2, !dbg !23
  %927 = shufflevector <32 x float> %921, <32 x float> %921, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %928 = shufflevector <128 x float> %927, <128 x float> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %929 = shufflevector <32 x float> %922, <32 x float> %922, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %930 = shufflevector <128 x float> %929, <128 x float> %928, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %931 = shufflevector <32 x float> %923, <32 x float> %923, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %932 = shufflevector <128 x float> %931, <128 x float> %930, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %933 = shufflevector <32 x float> %924, <32 x float> %924, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %934 = shufflevector <128 x float> %933, <128 x float> %932, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %935 = add i64 %902, %926, !dbg !23
  %936 = insertelement <128 x float> %934, float %925, i64 %935, !dbg !23
  %937 = shufflevector <128 x float> %936, <128 x float> %936, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %938 = shufflevector <128 x float> %936, <128 x float> %936, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !23
  %939 = shufflevector <128 x float> %936, <128 x float> %936, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>, !dbg !23
  %940 = shufflevector <128 x float> %936, <128 x float> %936, <32 x i32> <i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !23
  %941 = extractelement <4 x float> %885, i64 3, !dbg !23
  %942 = add i64 %889, 3, !dbg !23
  %943 = shufflevector <32 x float> %937, <32 x float> %937, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %944 = shufflevector <128 x float> %943, <128 x float> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %945 = shufflevector <32 x float> %938, <32 x float> %938, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %946 = shufflevector <128 x float> %945, <128 x float> %944, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %947 = shufflevector <32 x float> %939, <32 x float> %939, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %948 = shufflevector <128 x float> %947, <128 x float> %946, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !23
  %949 = shufflevector <32 x float> %940, <32 x float> %940, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !23
  %950 = shufflevector <128 x float> %949, <128 x float> %948, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %951 = add i64 %902, %942, !dbg !23
  %952 = insertelement <128 x float> %950, float %941, i64 %951, !dbg !23
  %953 = shufflevector <128 x float> %952, <128 x float> %952, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !23
  %954 = insertvalue [4 x <32 x float>] poison, <32 x float> %953, 0, !dbg !23
  %955 = shufflevector <128 x float> %952, <128 x float> %952, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>, !dbg !23
  %956 = insertvalue [4 x <32 x float>] %954, <32 x float> %955, 1, !dbg !23
  %957 = shufflevector <128 x float> %952, <128 x float> %952, <32 x i32> <i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95>, !dbg !23
  %958 = insertvalue [4 x <32 x float>] %956, <32 x float> %957, 2, !dbg !23
  %959 = shufflevector <128 x float> %952, <128 x float> %952, <32 x i32> <i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127>, !dbg !23
  %960 = insertvalue [4 x <32 x float>] %958, <32 x float> %959, 3, !dbg !23
  %961 = add i64 %807, 1, !dbg !23
  br label %806, !dbg !23

962:                                              ; preds = %806
  %963 = extractvalue [4 x <32 x float>] %117, 0, !dbg !24
  %964 = extractvalue [4 x <32 x float>] %808, 0, !dbg !24
  %965 = fadd <32 x float> %963, %964, !dbg !24
  %966 = insertvalue [4 x <32 x float>] poison, <32 x float> %965, 0, !dbg !24
  %967 = extractvalue [4 x <32 x float>] %117, 1, !dbg !24
  %968 = extractvalue [4 x <32 x float>] %808, 1, !dbg !24
  %969 = fadd <32 x float> %967, %968, !dbg !24
  %970 = insertvalue [4 x <32 x float>] %966, <32 x float> %969, 1, !dbg !24
  %971 = extractvalue [4 x <32 x float>] %117, 2, !dbg !24
  %972 = extractvalue [4 x <32 x float>] %808, 2, !dbg !24
  %973 = fadd <32 x float> %971, %972, !dbg !24
  %974 = insertvalue [4 x <32 x float>] %970, <32 x float> %973, 2, !dbg !24
  %975 = extractvalue [4 x <32 x float>] %117, 3, !dbg !24
  %976 = extractvalue [4 x <32 x float>] %808, 3, !dbg !24
  %977 = fadd <32 x float> %975, %976, !dbg !24
  %978 = insertvalue [4 x <32 x float>] %974, <32 x float> %977, 3, !dbg !24
  %979 = add i64 %116, 1, !dbg !13
  br label %115, !dbg !13

980:                                              ; preds = %115
  %981 = extractvalue [4 x <1 x float>] %118, 0, !dbg !25
  %982 = extractelement <1 x float> %981, i64 0, !dbg !25
  %983 = insertelement <32 x float> poison, float %982, i32 0, !dbg !25
  %984 = shufflevector <32 x float> %983, <32 x float> poison, <32 x i32> zeroinitializer, !dbg !25
  %985 = extractvalue [4 x <1 x float>] %118, 1, !dbg !25
  %986 = extractelement <1 x float> %985, i64 0, !dbg !25
  %987 = insertelement <32 x float> poison, float %986, i32 0, !dbg !25
  %988 = shufflevector <32 x float> %987, <32 x float> poison, <32 x i32> zeroinitializer, !dbg !25
  %989 = extractvalue [4 x <1 x float>] %118, 2, !dbg !25
  %990 = extractelement <1 x float> %989, i64 0, !dbg !25
  %991 = insertelement <32 x float> poison, float %990, i32 0, !dbg !25
  %992 = shufflevector <32 x float> %991, <32 x float> poison, <32 x i32> zeroinitializer, !dbg !25
  %993 = extractvalue [4 x <1 x float>] %118, 3, !dbg !25
  %994 = extractelement <1 x float> %993, i64 0, !dbg !25
  %995 = insertelement <32 x float> poison, float %994, i32 0, !dbg !25
  %996 = shufflevector <32 x float> %995, <32 x float> poison, <32 x i32> zeroinitializer, !dbg !25
  %997 = extractvalue [4 x <32 x float>] %117, 0, !dbg !25
  %998 = fdiv <32 x float> %997, %984, !dbg !25
  %999 = extractvalue [4 x <32 x float>] %117, 1, !dbg !25
  %1000 = fdiv <32 x float> %999, %988, !dbg !25
  %1001 = extractvalue [4 x <32 x float>] %117, 2, !dbg !25
  %1002 = fdiv <32 x float> %1001, %992, !dbg !25
  %1003 = extractvalue [4 x <32 x float>] %117, 3, !dbg !25
  %1004 = fdiv <32 x float> %1003, %996, !dbg !25
  %1005 = fptrunc <32 x float> %998 to <32 x half>, !dbg !26
  %1006 = fptrunc <32 x float> %1000 to <32 x half>, !dbg !26
  %1007 = fptrunc <32 x float> %1002 to <32 x half>, !dbg !26
  %1008 = fptrunc <32 x float> %1004 to <32 x half>, !dbg !26
  br label %1009, !dbg !27

1009:                                             ; preds = %1061, %980
  %1010 = phi i64 [ %1062, %1061 ], [ 0, %980 ], !dbg !27
  %1011 = icmp slt i64 %1010, 4, !dbg !27
  br i1 %1011, label %1012, label %1063, !dbg !27

1012:                                             ; preds = %1009
  %1013 = mul nsw i64 %1010, 16, !dbg !27
  %1014 = add i64 %1013, %23, !dbg !27
  %1015 = add i64 %1014, %10, !dbg !27
  %1016 = add i64 %1015, %30, !dbg !27
  %1017 = add i64 %1016, %31, !dbg !27
  br label %1018, !dbg !27

1018:                                             ; preds = %1021, %1012
  %1019 = phi i64 [ %1060, %1021 ], [ 0, %1012 ], !dbg !27
  %1020 = icmp slt i64 %1019, 8, !dbg !27
  br i1 %1020, label %1021, label %1061, !dbg !27

1021:                                             ; preds = %1018
  %1022 = mul nsw i64 %1019, 4, !dbg !27
  %1023 = shufflevector <32 x half> %1005, <32 x half> %1005, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !27
  %1024 = shufflevector <128 x half> %1023, <128 x half> poison, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !27
  %1025 = shufflevector <32 x half> %1006, <32 x half> %1006, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !27
  %1026 = shufflevector <128 x half> %1025, <128 x half> %1024, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !27
  %1027 = shufflevector <32 x half> %1007, <32 x half> %1007, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !27
  %1028 = shufflevector <128 x half> %1027, <128 x half> %1026, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255>, !dbg !27
  %1029 = shufflevector <32 x half> %1008, <32 x half> %1008, <128 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !27
  %1030 = shufflevector <128 x half> %1029, <128 x half> %1028, <128 x i32> <i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !27
  %1031 = mul i64 %1010, 32, !dbg !27
  %1032 = add i64 %1031, %1022, !dbg !27
  %1033 = extractelement <128 x half> %1030, i64 %1032, !dbg !27
  %1034 = mul nsw i64 %1019, 16, !dbg !27
  %1035 = add i64 %1034, %29, !dbg !27
  %1036 = mul i64 %22, 524288, !dbg !27
  %1037 = add i64 0, %1036, !dbg !27
  %1038 = mul i64 %1017, 128, !dbg !27
  %1039 = add i64 %1037, %1038, !dbg !27
  %1040 = add i64 %1039, %1035, !dbg !27
  %1041 = getelementptr half, ptr addrspace(1) %3, i64 %1040, !dbg !27
  store half %1033, ptr addrspace(1) %1041, align 2, !dbg !27
  %1042 = add i64 %1022, 1, !dbg !27
  %1043 = add i64 %1031, %1042, !dbg !27
  %1044 = extractelement <128 x half> %1030, i64 %1043, !dbg !27
  %1045 = add i64 %1035, 4, !dbg !27
  %1046 = add i64 %1039, %1045, !dbg !27
  %1047 = getelementptr half, ptr addrspace(1) %3, i64 %1046, !dbg !27
  store half %1044, ptr addrspace(1) %1047, align 2, !dbg !27
  %1048 = add i64 %1022, 2, !dbg !27
  %1049 = add i64 %1031, %1048, !dbg !27
  %1050 = extractelement <128 x half> %1030, i64 %1049, !dbg !27
  %1051 = add i64 %1035, 8, !dbg !27
  %1052 = add i64 %1039, %1051, !dbg !27
  %1053 = getelementptr half, ptr addrspace(1) %3, i64 %1052, !dbg !27
  store half %1050, ptr addrspace(1) %1053, align 2, !dbg !27
  %1054 = add i64 %1022, 3, !dbg !27
  %1055 = add i64 %1031, %1054, !dbg !27
  %1056 = extractelement <128 x half> %1030, i64 %1055, !dbg !27
  %1057 = add i64 %1035, 12, !dbg !27
  %1058 = add i64 %1039, %1057, !dbg !27
  %1059 = getelementptr half, ptr addrspace(1) %3, i64 %1058, !dbg !27
  store half %1056, ptr addrspace(1) %1059, align 2, !dbg !27
  %1060 = add i64 %1019, 1, !dbg !27
  br label %1018, !dbg !27

1061:                                             ; preds = %1018
  %1062 = add i64 %1010, 1, !dbg !27
  br label %1009, !dbg !27

1063:                                             ; preds = %1009
  ret void, !dbg !7
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn
declare noundef i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.masked.store.v4f16.p3(<4 x half>, ptr addrspace(3), i32 immarg, <4 x i1>) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare <4 x half> @llvm.masked.load.v4f16.p1(ptr addrspace(1), i32 immarg, <4 x i1>, <4 x half>) #6

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "amdgpu-waves-per-eu"="1" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 64, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 44, column: 12, scope: !3)
!9 = !DILocation(line: 48, column: 13, scope: !3)
!10 = !DILocation(line: 51, column: 13, scope: !3)
!11 = !DILocation(line: 76, column: 15, scope: !3)
!12 = !DILocation(line: 77, column: 15, scope: !3)
!13 = !DILocation(line: 59, column: 15, scope: !3)
!14 = !DILocation(line: 61, column: 15, scope: !3)
!15 = !DILocation(line: 60, column: 15, scope: !3)
!16 = !DILocation(line: 62, column: 15, scope: !3)
!17 = !DILocation(line: 63, column: 15, scope: !3)
!18 = !DILocation(line: 65, column: 17, scope: !3)
!19 = !DILocation(line: 66, column: 17, scope: !3)
!20 = !DILocation(line: 67, column: 17, scope: !3)
!21 = !DILocation(line: 74, column: 15, scope: !3)
!22 = !DILocation(line: 75, column: 15, scope: !3)
!23 = !DILocation(line: 78, column: 15, scope: !3)
!24 = !DILocation(line: 79, column: 15, scope: !3)
!25 = !DILocation(line: 84, column: 13, scope: !3)
!26 = !DILocation(line: 85, column: 13, scope: !3)
!27 = !DILocation(line: 86, column: 7, scope: !3)
