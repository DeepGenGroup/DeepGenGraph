; ModuleID = '3rd/deepgengraph/build/temp.mlir'
source_filename = "3rd/deepgengraph/build/temp.mlir"

@shm_5 = addrspace(3) global [64 x [128 x half]] undef, align 16
@shm_4 = addrspace(3) global [64 x [32 x half]] undef, align 16
@shm_3 = addrspace(3) global [64 x [1 x float]] undef, align 16
@shm_2 = addrspace(3) global [32 x [128 x half]] undef, align 16
@shm_1 = addrspace(3) global [128 x [32 x half]] undef, align 16
@shm_0 = addrspace(3) global [64 x [128 x half]] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !dbg !3 !reqd_work_group_size !6 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !8
  %8 = sext i32 %7 to i64, !dbg !8
  %9 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %10 = sext i32 %9 to i64, !dbg !9
  %11 = mul i64 %6, 64, !dbg !10
  %12 = icmp slt i64 %6, 0, !dbg !11
  %13 = sub i64 -1, %6, !dbg !11
  %14 = select i1 %12, i64 %13, i64 %6, !dbg !11
  %15 = sdiv i64 %14, 64, !dbg !11
  %16 = sub i64 -1, %15, !dbg !11
  %17 = select i1 %12, i64 %16, i64 %15, !dbg !11
  %18 = add i64 %8, %17, !dbg !11
  %19 = srem i64 %18, 32, !dbg !11
  %20 = icmp slt i64 %19, 0, !dbg !11
  %21 = add i64 %19, 32, !dbg !11
  %22 = select i1 %20, i64 %21, i64 %19, !dbg !11
  %23 = mul nsw i64 %6, 64, !dbg !11
  %24 = icmp slt i64 %10, 0, !dbg !11
  %25 = sub i64 -1, %10, !dbg !11
  %26 = select i1 %24, i64 %25, i64 %10, !dbg !11
  %27 = sdiv i64 %26, 16, !dbg !11
  %28 = sub i64 -1, %27, !dbg !11
  %29 = select i1 %24, i64 %28, i64 %27, !dbg !11
  %30 = mul nsw i64 %29, -16, !dbg !11
  %31 = mul nsw i64 %17, -4096, !dbg !11
  %32 = mul nsw i64 %29, 4, !dbg !11
  br label %33, !dbg !11

33:                                               ; preds = %65, %4
  %34 = phi i64 [ %66, %65 ], [ 0, %4 ], !dbg !11
  %35 = icmp slt i64 %34, 2, !dbg !11
  br i1 %35, label %36, label %67, !dbg !11

36:                                               ; preds = %33
  %37 = mul nsw i64 %34, 32, !dbg !11
  %38 = add i64 %37, %23, !dbg !11
  %39 = add i64 %38, %10, !dbg !11
  %40 = add i64 %39, %30, !dbg !11
  %41 = add i64 %40, %31, !dbg !11
  %42 = add i64 %37, %10, !dbg !11
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
  %68 = srem i64 %10, 16, !dbg !12
  %69 = icmp slt i64 %68, 0, !dbg !12
  %70 = add i64 %68, 16, !dbg !12
  %71 = select i1 %69, i64 %70, i64 %68, !dbg !12
  %72 = mul i64 %71, 128, !dbg !12
  %73 = add i64 %72, %32, !dbg !12
  %74 = getelementptr half, ptr addrspace(3) @shm_0, i64 %73, !dbg !12
  %75 = load half, ptr addrspace(3) %74, align 2, !dbg !12
  %76 = insertelement <4 x half> zeroinitializer, half %75, i64 0, !dbg !12
  %77 = add i64 %32, 1, !dbg !12
  %78 = add i64 %72, %77, !dbg !12
  %79 = getelementptr half, ptr addrspace(3) @shm_0, i64 %78, !dbg !12
  %80 = load half, ptr addrspace(3) %79, align 2, !dbg !12
  %81 = insertelement <4 x half> %76, half %80, i64 1, !dbg !12
  %82 = add i64 %32, 2, !dbg !12
  %83 = add i64 %72, %82, !dbg !12
  %84 = getelementptr half, ptr addrspace(3) @shm_0, i64 %83, !dbg !12
  %85 = load half, ptr addrspace(3) %84, align 2, !dbg !12
  %86 = insertelement <4 x half> %81, half %85, i64 2, !dbg !12
  %87 = add i64 %32, 3, !dbg !12
  %88 = add i64 %72, %87, !dbg !12
  %89 = getelementptr half, ptr addrspace(3) @shm_0, i64 %88, !dbg !12
  %90 = load half, ptr addrspace(3) %89, align 2, !dbg !12
  %91 = insertelement <4 x half> %86, half %90, i64 3, !dbg !12
  %compat.splat.insert = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !12
  %compat.splat = shufflevector <4 x half> %compat.splat.insert, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %92 = fmul <4 x half> %91, %compat.splat, !dbg !12
  %93 = alloca float, i64 1, align 4, addrspace(5), !dbg !13
  %94 = alloca float, i64 4, align 1, addrspace(5), !dbg !14
  %95 = alloca half, i64 4, align 1, addrspace(5), !dbg !14
  %96 = srem i64 %10, 64, !dbg !13
  %97 = icmp slt i64 %96, 0, !dbg !13
  %98 = add i64 %96, 64, !dbg !13
  %99 = select i1 %97, i64 %98, i64 %96, !dbg !13
  %100 = icmp slt i64 %99, 0, !dbg !13
  %101 = sub i64 -1, %99, !dbg !13
  %102 = select i1 %100, i64 %101, i64 %99, !dbg !13
  %103 = sdiv i64 %102, 16, !dbg !13
  %104 = sub i64 -1, %103, !dbg !13
  %105 = select i1 %100, i64 %104, i64 %103, !dbg !13
  %106 = icmp eq i64 %105, 0, !dbg !13
  %107 = mul nsw i64 %6, 2, !dbg !15
  %108 = add i64 %107, 1, !dbg !15
  %109 = mul nsw i64 %8, 128, !dbg !16
  %110 = mul nsw i64 %8, 4096, !dbg !16
  %111 = mul nsw i64 %8, 16384, !dbg !17
  %112 = mul nsw i64 %6, 256, !dbg !17
  %113 = add i64 %109, %107, !dbg !17
  br label %114, !dbg !15

114:                                              ; preds = %766, %67
  %115 = phi i64 [ %767, %766 ], [ 0, %67 ], !dbg !18
  %116 = icmp slt i64 %115, %108, !dbg !15
  br i1 %116, label %117, label %768, !dbg !15

117:                                              ; preds = %114
  %118 = add i64 %115, %109, !dbg !16
  %119 = add i64 %118, %107, !dbg !16
  %120 = icmp slt i64 %119, 0, !dbg !16
  %121 = sub i64 -1, %119, !dbg !16
  %122 = select i1 %120, i64 %121, i64 %119, !dbg !16
  %123 = sdiv i64 %122, 128, !dbg !16
  %124 = sub i64 -1, %123, !dbg !16
  %125 = select i1 %120, i64 %124, i64 %123, !dbg !16
  %126 = srem i64 %125, 32, !dbg !16
  %127 = icmp slt i64 %126, 0, !dbg !16
  %128 = add i64 %126, 32, !dbg !16
  %129 = select i1 %127, i64 %128, i64 %126, !dbg !16
  %130 = mul nsw i64 %115, 32, !dbg !16
  %131 = mul nsw i64 %125, -4096, !dbg !16
  br label %132, !dbg !16

132:                                              ; preds = %209, %117
  %133 = phi i64 [ %210, %209 ], [ 0, %117 ], !dbg !16
  %134 = icmp slt i64 %133, 4, !dbg !16
  br i1 %134, label %135, label %211, !dbg !16

135:                                              ; preds = %132
  %136 = add i64 %130, %133, !dbg !16
  br label %137, !dbg !16

137:                                              ; preds = %207, %135
  %138 = phi i64 [ %208, %207 ], [ 0, %135 ], !dbg !16
  %139 = icmp slt i64 %138, 2, !dbg !16
  br i1 %139, label %140, label %209, !dbg !16

140:                                              ; preds = %137
  %141 = mul nsw i64 %138, 16, !dbg !16
  %142 = add i64 %141, %10, !dbg !16
  %143 = add i64 %142, %30, !dbg !16
  br label %144, !dbg !16

144:                                              ; preds = %205, %140
  %145 = phi i64 [ %206, %205 ], [ 0, %140 ], !dbg !16
  %146 = icmp slt i64 %145, 2, !dbg !16
  br i1 %146, label %147, label %207, !dbg !16

147:                                              ; preds = %144
  %148 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !16
  %149 = mul nsw i64 %145, 64, !dbg !16
  %150 = add i64 %136, %149, !dbg !16
  %151 = add i64 %150, %110, !dbg !16
  %152 = add i64 %151, %23, !dbg !16
  %153 = add i64 %152, %131, !dbg !16
  %154 = add i64 %153, %32, !dbg !16
  br label %155, !dbg !16

155:                                              ; preds = %174, %147
  %156 = phi i64 [ %175, %174 ], [ 0, %147 ], !dbg !16
  %157 = phi <4 x half> [ %173, %174 ], [ zeroinitializer, %147 ], !dbg !16
  %158 = icmp slt i64 %156, 4, !dbg !16
  br i1 %158, label %159, label %176, !dbg !16

159:                                              ; preds = %155
  %160 = add i64 %154, %156, !dbg !16
  %161 = icmp slt i64 %160, 4096, !dbg !16
  br i1 %161, label %162, label %171, !dbg !16

162:                                              ; preds = %159
  %163 = mul i64 %129, 524288, !dbg !16
  %164 = add i64 0, %163, !dbg !16
  %165 = mul i64 %160, 128, !dbg !16
  %166 = add i64 %164, %165, !dbg !16
  %167 = add i64 %166, %143, !dbg !16
  %168 = getelementptr half, ptr addrspace(1) %2, i64 %167, !dbg !16
  %169 = load half, ptr addrspace(1) %168, align 2, !dbg !16
  %170 = insertelement <4 x half> %157, half %169, i64 %156, !dbg !16
  br label %172, !dbg !16

171:                                              ; preds = %159
  br label %172, !dbg !16

172:                                              ; preds = %162, %171
  %173 = phi <4 x half> [ %157, %171 ], [ %170, %162 ], !dbg !16
  br label %174, !dbg !16

174:                                              ; preds = %172
  %175 = add i64 %156, 1, !dbg !16
  br label %155, !dbg !16

176:                                              ; preds = %155
  %177 = add i64 %133, %149, !dbg !16
  %178 = add i64 %177, %32, !dbg !16
  %179 = extractelement <4 x half> %157, i64 0, !dbg !16
  %180 = insertelement <1 x half> poison, half %179, i64 0, !dbg !16
  %181 = insertvalue [4 x <1 x half>] poison, <1 x half> %180, 0, !dbg !16
  %182 = extractelement <4 x half> %157, i64 1, !dbg !16
  %183 = insertelement <1 x half> poison, half %182, i64 0, !dbg !16
  %184 = insertvalue [4 x <1 x half>] %181, <1 x half> %183, 1, !dbg !16
  %185 = extractelement <4 x half> %157, i64 2, !dbg !16
  %186 = insertelement <1 x half> poison, half %185, i64 0, !dbg !16
  %187 = insertvalue [4 x <1 x half>] %184, <1 x half> %186, 2, !dbg !16
  %188 = extractelement <4 x half> %157, i64 3, !dbg !16
  %189 = insertelement <1 x half> poison, half %188, i64 0, !dbg !16
  %190 = insertvalue [4 x <1 x half>] %187, <1 x half> %189, 3, !dbg !16
  store [4 x <1 x half>] %190, ptr %148, align 2, !dbg !16
  br label %191, !dbg !16

191:                                              ; preds = %203, %176
  %192 = phi i64 [ %204, %203 ], [ 0, %176 ], !dbg !16
  %193 = icmp slt i64 %192, 4, !dbg !16
  br i1 %193, label %194, label %205, !dbg !16

194:                                              ; preds = %191
  %195 = add i64 %178, %192, !dbg !16
  %196 = icmp slt i64 %195, 128, !dbg !16
  br i1 %196, label %197, label %203, !dbg !16

197:                                              ; preds = %194
  %198 = getelementptr <1 x half>, ptr %148, i64 %192, !dbg !16
  %199 = load <1 x half>, ptr %198, align 2, !dbg !16
  %200 = mul i64 %195, 32, !dbg !16
  %201 = add i64 %200, %143, !dbg !16
  %202 = getelementptr half, ptr addrspace(3) @shm_1, i64 %201, !dbg !16
  store <1 x half> %199, ptr addrspace(3) %202, align 2, !dbg !16
  br label %203, !dbg !16

203:                                              ; preds = %197, %194
  %204 = add i64 %192, 1, !dbg !16
  br label %191, !dbg !16

205:                                              ; preds = %191
  %206 = add i64 %145, 1, !dbg !16
  br label %144, !dbg !16

207:                                              ; preds = %144
  %208 = add i64 %138, 1, !dbg !16
  br label %137, !dbg !16

209:                                              ; preds = %137
  %210 = add i64 %133, 1, !dbg !16
  br label %132, !dbg !16

211:                                              ; preds = %132
  %212 = add i64 %115, %111, !dbg !17
  %213 = add i64 %212, %112, !dbg !17
  %214 = icmp slt i64 %213, 0, !dbg !17
  %215 = sub i64 -1, %213, !dbg !17
  %216 = select i1 %214, i64 %215, i64 %213, !dbg !17
  %217 = sdiv i64 %216, 16384, !dbg !17
  %218 = sub i64 -1, %217, !dbg !17
  %219 = select i1 %214, i64 %218, i64 %217, !dbg !17
  %220 = srem i64 %219, 32, !dbg !17
  %221 = icmp slt i64 %220, 0, !dbg !17
  %222 = add i64 %220, 32, !dbg !17
  %223 = select i1 %221, i64 %222, i64 %220, !dbg !17
  %224 = icmp slt i64 %115, 0, !dbg !17
  %225 = sub i64 -1, %115, !dbg !17
  %226 = select i1 %224, i64 %225, i64 %115, !dbg !17
  %227 = sdiv i64 %226, 128, !dbg !17
  %228 = sub i64 -1, %227, !dbg !17
  %229 = select i1 %224, i64 %228, i64 %227, !dbg !17
  %230 = add i64 %229, %113, !dbg !17
  %231 = icmp slt i64 %230, 0, !dbg !17
  %232 = sub i64 -1, %230, !dbg !17
  %233 = select i1 %231, i64 %232, i64 %230, !dbg !17
  %234 = sdiv i64 %233, 128, !dbg !17
  %235 = sub i64 -1, %234, !dbg !17
  %236 = select i1 %231, i64 %235, i64 %234, !dbg !17
  %237 = mul nsw i64 %236, -128, !dbg !17
  %238 = mul nsw i64 %229, -4096, !dbg !17
  br label %239, !dbg !17

239:                                              ; preds = %311, %211
  %240 = phi i64 [ %312, %311 ], [ 0, %211 ], !dbg !17
  %241 = icmp slt i64 %240, 4, !dbg !17
  br i1 %241, label %242, label %313, !dbg !17

242:                                              ; preds = %239
  %243 = add i64 %240, %109, !dbg !17
  %244 = add i64 %243, %107, !dbg !17
  %245 = add i64 %244, %229, !dbg !17
  %246 = add i64 %245, %237, !dbg !17
  %247 = add i64 %246, %32, !dbg !17
  %248 = add i64 %240, %32, !dbg !17
  br label %249, !dbg !17

249:                                              ; preds = %309, %242
  %250 = phi i64 [ %310, %309 ], [ 0, %242 ], !dbg !17
  %251 = icmp slt i64 %250, 8, !dbg !17
  br i1 %251, label %252, label %311, !dbg !17

252:                                              ; preds = %249
  %253 = alloca [4 x <1 x half>], i64 1, align 2, !dbg !17
  %254 = mul nsw i64 %250, 16, !dbg !17
  %255 = add i64 %130, %254, !dbg !17
  %256 = add i64 %255, %10, !dbg !17
  %257 = add i64 %256, %238, !dbg !17
  %258 = add i64 %257, %30, !dbg !17
  br label %259, !dbg !17

259:                                              ; preds = %278, %252
  %260 = phi i64 [ %279, %278 ], [ 0, %252 ], !dbg !17
  %261 = phi <4 x half> [ %277, %278 ], [ zeroinitializer, %252 ], !dbg !17
  %262 = icmp slt i64 %260, 4, !dbg !17
  br i1 %262, label %263, label %280, !dbg !17

263:                                              ; preds = %259
  %264 = add i64 %247, %260, !dbg !17
  %265 = icmp slt i64 %264, 128, !dbg !17
  br i1 %265, label %266, label %275, !dbg !17

266:                                              ; preds = %263
  %267 = mul i64 %223, 524288, !dbg !17
  %268 = add i64 0, %267, !dbg !17
  %269 = mul i64 %264, 4096, !dbg !17
  %270 = add i64 %268, %269, !dbg !17
  %271 = add i64 %270, %258, !dbg !17
  %272 = getelementptr half, ptr addrspace(1) %1, i64 %271, !dbg !17
  %273 = load half, ptr addrspace(1) %272, align 2, !dbg !17
  %274 = insertelement <4 x half> %261, half %273, i64 %260, !dbg !17
  br label %276, !dbg !17

275:                                              ; preds = %263
  br label %276, !dbg !17

276:                                              ; preds = %266, %275
  %277 = phi <4 x half> [ %261, %275 ], [ %274, %266 ], !dbg !17
  br label %278, !dbg !17

278:                                              ; preds = %276
  %279 = add i64 %260, 1, !dbg !17
  br label %259, !dbg !17

280:                                              ; preds = %259
  %281 = add i64 %254, %10, !dbg !17
  %282 = add i64 %281, %30, !dbg !17
  %283 = extractelement <4 x half> %261, i64 0, !dbg !17
  %284 = insertelement <1 x half> poison, half %283, i64 0, !dbg !17
  %285 = insertvalue [4 x <1 x half>] poison, <1 x half> %284, 0, !dbg !17
  %286 = extractelement <4 x half> %261, i64 1, !dbg !17
  %287 = insertelement <1 x half> poison, half %286, i64 0, !dbg !17
  %288 = insertvalue [4 x <1 x half>] %285, <1 x half> %287, 1, !dbg !17
  %289 = extractelement <4 x half> %261, i64 2, !dbg !17
  %290 = insertelement <1 x half> poison, half %289, i64 0, !dbg !17
  %291 = insertvalue [4 x <1 x half>] %288, <1 x half> %290, 2, !dbg !17
  %292 = extractelement <4 x half> %261, i64 3, !dbg !17
  %293 = insertelement <1 x half> poison, half %292, i64 0, !dbg !17
  %294 = insertvalue [4 x <1 x half>] %291, <1 x half> %293, 3, !dbg !17
  store [4 x <1 x half>] %294, ptr %253, align 2, !dbg !17
  br label %295, !dbg !17

295:                                              ; preds = %307, %280
  %296 = phi i64 [ %308, %307 ], [ 0, %280 ], !dbg !17
  %297 = icmp slt i64 %296, 4, !dbg !17
  br i1 %297, label %298, label %309, !dbg !17

298:                                              ; preds = %295
  %299 = add i64 %248, %296, !dbg !17
  %300 = icmp slt i64 %299, 32, !dbg !17
  br i1 %300, label %301, label %307, !dbg !17

301:                                              ; preds = %298
  %302 = getelementptr <1 x half>, ptr %253, i64 %296, !dbg !17
  %303 = load <1 x half>, ptr %302, align 2, !dbg !17
  %304 = mul i64 %299, 128, !dbg !17
  %305 = add i64 %304, %282, !dbg !17
  %306 = getelementptr half, ptr addrspace(3) @shm_2, i64 %305, !dbg !17
  store <1 x half> %303, ptr addrspace(3) %306, align 2, !dbg !17
  br label %307, !dbg !17

307:                                              ; preds = %301, %298
  %308 = add i64 %296, 1, !dbg !17
  br label %295, !dbg !17

309:                                              ; preds = %295
  %310 = add i64 %250, 1, !dbg !17
  br label %249, !dbg !17

311:                                              ; preds = %249
  %312 = add i64 %240, 1, !dbg !17
  br label %239, !dbg !17

313:                                              ; preds = %239
  br label %314, !dbg !19

314:                                              ; preds = %372, %313
  %315 = phi i64 [ %421, %372 ], [ 0, %313 ], !dbg !19
  %316 = phi [2 x <8 x float>] [ %420, %372 ], [ zeroinitializer, %313 ], !dbg !19
  %317 = icmp slt i64 %315, 4, !dbg !19
  br i1 %317, label %318, label %422, !dbg !19

318:                                              ; preds = %314
  %319 = mul nsw i64 %315, 16, !dbg !19
  %320 = add i64 %319, %10, !dbg !19
  %321 = icmp slt i64 %315, 0, !dbg !19
  %322 = sub i64 -1, %315, !dbg !19
  %323 = select i1 %321, i64 %322, i64 %315, !dbg !19
  %324 = sdiv i64 %323, 2, !dbg !19
  %325 = sub i64 -1, %324, !dbg !19
  %326 = select i1 %321, i64 %325, i64 %324, !dbg !19
  %327 = mul nsw i64 %326, -32, !dbg !19
  %328 = add i64 %320, %327, !dbg !19
  %329 = add i64 %328, %30, !dbg !19
  br label %330, !dbg !19

330:                                              ; preds = %334, %318
  %331 = phi i64 [ %371, %334 ], [ 0, %318 ], !dbg !19
  %332 = phi [1 x <4 x float>] [ %370, %334 ], [ zeroinitializer, %318 ], !dbg !19
  %333 = icmp slt i64 %331, 8, !dbg !19
  br i1 %333, label %334, label %372, !dbg !19

334:                                              ; preds = %330
  %335 = mul nsw i64 %331, 16, !dbg !19
  %336 = add i64 %335, %32, !dbg !19
  %337 = mul i64 %336, 32, !dbg !19
  %338 = add i64 %337, %329, !dbg !19
  %339 = getelementptr half, ptr addrspace(3) @shm_1, i64 %338, !dbg !19
  %340 = load half, ptr addrspace(3) %339, align 2, !dbg !19
  %341 = insertelement <1 x half> zeroinitializer, half %340, i64 0, !dbg !19
  %342 = add i64 %336, 1, !dbg !19
  %343 = mul i64 %342, 32, !dbg !19
  %344 = add i64 %343, %329, !dbg !19
  %345 = getelementptr half, ptr addrspace(3) @shm_1, i64 %344, !dbg !19
  %346 = load half, ptr addrspace(3) %345, align 2, !dbg !19
  %347 = insertelement <1 x half> zeroinitializer, half %346, i64 0, !dbg !19
  %348 = add i64 %336, 2, !dbg !19
  %349 = mul i64 %348, 32, !dbg !19
  %350 = add i64 %349, %329, !dbg !19
  %351 = getelementptr half, ptr addrspace(3) @shm_1, i64 %350, !dbg !19
  %352 = load half, ptr addrspace(3) %351, align 2, !dbg !19
  %353 = insertelement <1 x half> zeroinitializer, half %352, i64 0, !dbg !19
  %354 = add i64 %336, 3, !dbg !19
  %355 = mul i64 %354, 32, !dbg !19
  %356 = add i64 %355, %329, !dbg !19
  %357 = getelementptr half, ptr addrspace(3) @shm_1, i64 %356, !dbg !19
  %358 = load half, ptr addrspace(3) %357, align 2, !dbg !19
  %359 = insertelement <1 x half> zeroinitializer, half %358, i64 0, !dbg !19
  %360 = shufflevector <1 x half> %341, <1 x half> %341, <4 x i32> zeroinitializer, !dbg !19
  %361 = shufflevector <4 x half> %360, <4 x half> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !19
  %362 = shufflevector <1 x half> %347, <1 x half> %347, <4 x i32> zeroinitializer, !dbg !19
  %363 = shufflevector <4 x half> %362, <4 x half> %361, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !19
  %364 = shufflevector <1 x half> %353, <1 x half> %353, <4 x i32> zeroinitializer, !dbg !19
  %365 = shufflevector <4 x half> %364, <4 x half> %363, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !19
  %366 = shufflevector <1 x half> %359, <1 x half> %359, <4 x i32> zeroinitializer, !dbg !19
  %367 = shufflevector <4 x half> %366, <4 x half> %365, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !19
  %368 = extractvalue [1 x <4 x float>] %332, 0, !dbg !19
  %369 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %92, <4 x half> %367, <4 x float> %368), !dbg !19
  %370 = insertvalue [1 x <4 x float>] poison, <4 x float> %369, 0, !dbg !19
  %371 = add i64 %331, 1, !dbg !19
  br label %330, !dbg !19

372:                                              ; preds = %330
  %373 = extractvalue [1 x <4 x float>] %332, 0, !dbg !19
  %374 = extractelement <4 x float> %373, i64 0, !dbg !19
  %375 = mul nsw i64 %315, 4, !dbg !19
  %376 = mul nsw i64 %326, -8, !dbg !19
  %377 = add i64 %375, %376, !dbg !19
  %378 = extractvalue [2 x <8 x float>] %316, 0, !dbg !19
  %379 = shufflevector <8 x float> %378, <8 x float> %378, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %380 = shufflevector <16 x float> %379, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %381 = extractvalue [2 x <8 x float>] %316, 1, !dbg !19
  %382 = shufflevector <8 x float> %381, <8 x float> %381, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %383 = shufflevector <16 x float> %382, <16 x float> %380, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %384 = mul i64 %326, 8, !dbg !19
  %385 = add i64 %384, %377, !dbg !19
  %386 = insertelement <16 x float> %383, float %374, i64 %385, !dbg !19
  %387 = shufflevector <16 x float> %386, <16 x float> %386, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %388 = shufflevector <16 x float> %386, <16 x float> %386, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %389 = extractelement <4 x float> %373, i64 1, !dbg !19
  %390 = add i64 %377, 1, !dbg !19
  %391 = shufflevector <8 x float> %387, <8 x float> %387, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %392 = shufflevector <16 x float> %391, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %393 = shufflevector <8 x float> %388, <8 x float> %388, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %394 = shufflevector <16 x float> %393, <16 x float> %392, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %395 = add i64 %384, %390, !dbg !19
  %396 = insertelement <16 x float> %394, float %389, i64 %395, !dbg !19
  %397 = shufflevector <16 x float> %396, <16 x float> %396, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %398 = shufflevector <16 x float> %396, <16 x float> %396, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %399 = extractelement <4 x float> %373, i64 2, !dbg !19
  %400 = add i64 %377, 2, !dbg !19
  %401 = shufflevector <8 x float> %397, <8 x float> %397, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %402 = shufflevector <16 x float> %401, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %403 = shufflevector <8 x float> %398, <8 x float> %398, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %404 = shufflevector <16 x float> %403, <16 x float> %402, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %405 = add i64 %384, %400, !dbg !19
  %406 = insertelement <16 x float> %404, float %399, i64 %405, !dbg !19
  %407 = shufflevector <16 x float> %406, <16 x float> %406, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %408 = shufflevector <16 x float> %406, <16 x float> %406, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %409 = extractelement <4 x float> %373, i64 3, !dbg !19
  %410 = add i64 %377, 3, !dbg !19
  %411 = shufflevector <8 x float> %407, <8 x float> %407, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %412 = shufflevector <16 x float> %411, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %413 = shufflevector <8 x float> %408, <8 x float> %408, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !19
  %414 = shufflevector <16 x float> %413, <16 x float> %412, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %415 = add i64 %384, %410, !dbg !19
  %416 = insertelement <16 x float> %414, float %409, i64 %415, !dbg !19
  %417 = shufflevector <16 x float> %416, <16 x float> %416, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %418 = insertvalue [2 x <8 x float>] poison, <8 x float> %417, 0, !dbg !19
  %419 = shufflevector <16 x float> %416, <16 x float> %416, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %420 = insertvalue [2 x <8 x float>] %418, <8 x float> %419, 1, !dbg !19
  %421 = add i64 %315, 1, !dbg !19
  br label %314, !dbg !19

422:                                              ; preds = %314
  br label %423, !dbg !20

423:                                              ; preds = %494, %422
  %424 = phi i64 [ %495, %494 ], [ 0, %422 ], !dbg !20
  %425 = phi [2 x <8 x float>] [ %435, %494 ], [ zeroinitializer, %422 ], !dbg !20
  %426 = icmp slt i64 %424, 2, !dbg !20
  br i1 %426, label %427, label %496, !dbg !20

427:                                              ; preds = %423
  %428 = mul nsw i64 %424, 32, !dbg !20
  %429 = add i64 %428, %10, !dbg !20
  %430 = add i64 %429, %11, !dbg !20
  %431 = add i64 %430, %30, !dbg !20
  %432 = add i64 %431, 1, !dbg !21
  br label %433, !dbg !20

433:                                              ; preds = %437, %427
  %434 = phi i64 [ %493, %437 ], [ 0, %427 ], !dbg !20
  %435 = phi [2 x <8 x float>] [ %492, %437 ], [ %425, %427 ], !dbg !20
  %436 = icmp slt i64 %434, 2, !dbg !20
  br i1 %436, label %437, label %494, !dbg !20

437:                                              ; preds = %433
  %438 = mul nsw i64 %434, 4, !dbg !20
  %439 = mul nsw i64 %434, 16, !dbg !20
  %440 = add i64 %130, %439, !dbg !20
  %441 = add i64 %440, %29, !dbg !20
  %442 = icmp ule i64 %432, %441, !dbg !22
  %443 = select i1 %442, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %444 = extractvalue [2 x <8 x float>] %435, 0, !dbg !20
  %445 = shufflevector <8 x float> %444, <8 x float> %444, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %446 = shufflevector <16 x float> %445, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %447 = extractvalue [2 x <8 x float>] %435, 1, !dbg !20
  %448 = shufflevector <8 x float> %447, <8 x float> %447, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %449 = shufflevector <16 x float> %448, <16 x float> %446, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %450 = mul i64 %424, 8, !dbg !20
  %451 = add i64 %450, %438, !dbg !20
  %452 = insertelement <16 x float> %449, float %443, i64 %451, !dbg !20
  %453 = shufflevector <16 x float> %452, <16 x float> %452, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %454 = shufflevector <16 x float> %452, <16 x float> %452, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %455 = add i64 %438, 1, !dbg !20
  %456 = add i64 %441, 4, !dbg !20
  %457 = icmp ule i64 %432, %456, !dbg !22
  %458 = select i1 %457, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %459 = shufflevector <8 x float> %453, <8 x float> %453, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %460 = shufflevector <16 x float> %459, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %461 = shufflevector <8 x float> %454, <8 x float> %454, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %462 = shufflevector <16 x float> %461, <16 x float> %460, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %463 = add i64 %450, %455, !dbg !20
  %464 = insertelement <16 x float> %462, float %458, i64 %463, !dbg !20
  %465 = shufflevector <16 x float> %464, <16 x float> %464, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %466 = shufflevector <16 x float> %464, <16 x float> %464, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %467 = add i64 %438, 2, !dbg !20
  %468 = add i64 %441, 8, !dbg !20
  %469 = icmp ule i64 %432, %468, !dbg !22
  %470 = select i1 %469, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %471 = shufflevector <8 x float> %465, <8 x float> %465, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %472 = shufflevector <16 x float> %471, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %473 = shufflevector <8 x float> %466, <8 x float> %466, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %474 = shufflevector <16 x float> %473, <16 x float> %472, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %475 = add i64 %450, %467, !dbg !20
  %476 = insertelement <16 x float> %474, float %470, i64 %475, !dbg !20
  %477 = shufflevector <16 x float> %476, <16 x float> %476, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %478 = shufflevector <16 x float> %476, <16 x float> %476, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %479 = add i64 %438, 3, !dbg !20
  %480 = add i64 %441, 12, !dbg !20
  %481 = icmp ule i64 %432, %480, !dbg !22
  %482 = select i1 %481, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %483 = shufflevector <8 x float> %477, <8 x float> %477, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %484 = shufflevector <16 x float> %483, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %485 = shufflevector <8 x float> %478, <8 x float> %478, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %486 = shufflevector <16 x float> %485, <16 x float> %484, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %487 = add i64 %450, %479, !dbg !20
  %488 = insertelement <16 x float> %486, float %482, i64 %487, !dbg !20
  %489 = shufflevector <16 x float> %488, <16 x float> %488, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %490 = insertvalue [2 x <8 x float>] poison, <8 x float> %489, 0, !dbg !20
  %491 = shufflevector <16 x float> %488, <16 x float> %488, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %492 = insertvalue [2 x <8 x float>] %490, <8 x float> %491, 1, !dbg !20
  %493 = add i64 %434, 1, !dbg !20
  br label %433, !dbg !20

494:                                              ; preds = %433
  %495 = add i64 %424, 1, !dbg !20
  br label %423, !dbg !20

496:                                              ; preds = %423
  %497 = extractvalue [2 x <8 x float>] %316, 0, !dbg !24
  %498 = extractvalue [2 x <8 x float>] %425, 0, !dbg !24
  %499 = fadd <8 x float> %497, %498, !dbg !24
  %500 = extractvalue [2 x <8 x float>] %316, 1, !dbg !24
  %501 = extractvalue [2 x <8 x float>] %425, 1, !dbg !24
  %502 = fadd <8 x float> %500, %501, !dbg !24
  %503 = extractelement <8 x float> %499, i64 0, !dbg !25
  %504 = call float @__ocml_exp2_f32(float %503), !dbg !25
  %505 = insertelement <8 x float> poison, float %504, i64 0, !dbg !25
  %506 = extractelement <8 x float> %499, i64 1, !dbg !25
  %507 = call float @__ocml_exp2_f32(float %506), !dbg !25
  %508 = insertelement <8 x float> %505, float %507, i64 1, !dbg !25
  %509 = extractelement <8 x float> %499, i64 2, !dbg !25
  %510 = call float @__ocml_exp2_f32(float %509), !dbg !25
  %511 = insertelement <8 x float> %508, float %510, i64 2, !dbg !25
  %512 = extractelement <8 x float> %499, i64 3, !dbg !25
  %513 = call float @__ocml_exp2_f32(float %512), !dbg !25
  %514 = insertelement <8 x float> %511, float %513, i64 3, !dbg !25
  %515 = extractelement <8 x float> %499, i64 4, !dbg !25
  %516 = call float @__ocml_exp2_f32(float %515), !dbg !25
  %517 = insertelement <8 x float> %514, float %516, i64 4, !dbg !25
  %518 = extractelement <8 x float> %499, i64 5, !dbg !25
  %519 = call float @__ocml_exp2_f32(float %518), !dbg !25
  %520 = insertelement <8 x float> %517, float %519, i64 5, !dbg !25
  %521 = extractelement <8 x float> %499, i64 6, !dbg !25
  %522 = call float @__ocml_exp2_f32(float %521), !dbg !25
  %523 = insertelement <8 x float> %520, float %522, i64 6, !dbg !25
  %524 = extractelement <8 x float> %499, i64 7, !dbg !25
  %525 = call float @__ocml_exp2_f32(float %524), !dbg !25
  %526 = insertelement <8 x float> %523, float %525, i64 7, !dbg !25
  %527 = extractelement <8 x float> %502, i64 0, !dbg !25
  %528 = call float @__ocml_exp2_f32(float %527), !dbg !25
  %529 = insertelement <8 x float> poison, float %528, i64 0, !dbg !25
  %530 = extractelement <8 x float> %502, i64 1, !dbg !25
  %531 = call float @__ocml_exp2_f32(float %530), !dbg !25
  %532 = insertelement <8 x float> %529, float %531, i64 1, !dbg !25
  %533 = extractelement <8 x float> %502, i64 2, !dbg !25
  %534 = call float @__ocml_exp2_f32(float %533), !dbg !25
  %535 = insertelement <8 x float> %532, float %534, i64 2, !dbg !25
  %536 = extractelement <8 x float> %502, i64 3, !dbg !25
  %537 = call float @__ocml_exp2_f32(float %536), !dbg !25
  %538 = insertelement <8 x float> %535, float %537, i64 3, !dbg !25
  %539 = extractelement <8 x float> %502, i64 4, !dbg !25
  %540 = call float @__ocml_exp2_f32(float %539), !dbg !25
  %541 = insertelement <8 x float> %538, float %540, i64 4, !dbg !25
  %542 = extractelement <8 x float> %502, i64 5, !dbg !25
  %543 = call float @__ocml_exp2_f32(float %542), !dbg !25
  %544 = insertelement <8 x float> %541, float %543, i64 5, !dbg !25
  %545 = extractelement <8 x float> %502, i64 6, !dbg !25
  %546 = call float @__ocml_exp2_f32(float %545), !dbg !25
  %547 = insertelement <8 x float> %544, float %546, i64 6, !dbg !25
  %548 = extractelement <8 x float> %502, i64 7, !dbg !25
  %549 = call float @__ocml_exp2_f32(float %548), !dbg !25
  %550 = insertelement <8 x float> %547, float %549, i64 7, !dbg !25
  br label %551, !dbg !13

551:                                              ; preds = %613, %496
  %552 = phi i64 [ %614, %613 ], [ 0, %496 ], !dbg !13
  %553 = icmp slt i64 %552, 2, !dbg !13
  br i1 %553, label %554, label %615, !dbg !13

554:                                              ; preds = %551
  store float 0.000000e+00, ptr addrspace(5) %93, align 4, !dbg !13
  br label %555, !dbg !13

555:                                              ; preds = %558, %554
  %556 = phi i64 [ %584, %558 ], [ 0, %554 ], !dbg !13
  %557 = icmp slt i64 %556, 2, !dbg !13
  br i1 %557, label %558, label %585, !dbg !13

558:                                              ; preds = %555
  %559 = mul nsw i64 %556, 4, !dbg !13
  %560 = load float, ptr addrspace(5) %93, align 4, !dbg !13
  %561 = shufflevector <8 x float> %526, <8 x float> %526, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !13
  %562 = shufflevector <16 x float> %561, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !13
  %563 = shufflevector <8 x float> %550, <8 x float> %550, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !13
  %564 = shufflevector <16 x float> %563, <16 x float> %562, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %565 = mul i64 %552, 8, !dbg !13
  %566 = add i64 %565, %559, !dbg !13
  %567 = extractelement <16 x float> %564, i64 %566, !dbg !13
  %568 = fadd float %560, %567, !dbg !13
  store float %568, ptr addrspace(5) %93, align 4, !dbg !13
  %569 = add i64 %559, 1, !dbg !13
  %570 = load float, ptr addrspace(5) %93, align 4, !dbg !13
  %571 = add i64 %565, %569, !dbg !13
  %572 = extractelement <16 x float> %564, i64 %571, !dbg !13
  %573 = fadd float %570, %572, !dbg !13
  store float %573, ptr addrspace(5) %93, align 4, !dbg !13
  %574 = add i64 %559, 2, !dbg !13
  %575 = load float, ptr addrspace(5) %93, align 4, !dbg !13
  %576 = add i64 %565, %574, !dbg !13
  %577 = extractelement <16 x float> %564, i64 %576, !dbg !13
  %578 = fadd float %575, %577, !dbg !13
  store float %578, ptr addrspace(5) %93, align 4, !dbg !13
  %579 = add i64 %559, 3, !dbg !13
  %580 = load float, ptr addrspace(5) %93, align 4, !dbg !13
  %581 = add i64 %565, %579, !dbg !13
  %582 = extractelement <16 x float> %564, i64 %581, !dbg !13
  %583 = fadd float %580, %582, !dbg !13
  store float %583, ptr addrspace(5) %93, align 4, !dbg !13
  %584 = add i64 %556, 1, !dbg !13
  br label %555, !dbg !13

585:                                              ; preds = %555
  %586 = load float, ptr addrspace(5) %93, align 4, !dbg !13
  %587 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %588 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %587), !dbg !13
  %589 = add i32 %588, 64, !dbg !13
  %590 = xor i32 %588, 16, !dbg !13
  %591 = and i32 %589, -64, !dbg !13
  %592 = icmp slt i32 %590, %591, !dbg !13
  %593 = select i1 %592, i32 %590, i32 %588, !dbg !13
  %594 = shl i32 %593, 2, !dbg !13
  %595 = bitcast float %586 to i32, !dbg !13
  %596 = call i32 @llvm.amdgcn.ds.bpermute(i32 %594, i32 %595), !dbg !13
  %597 = bitcast i32 %596 to float, !dbg !13
  %598 = fadd float %586, %597, !dbg !13
  %599 = xor i32 %588, 32, !dbg !13
  %600 = icmp slt i32 %599, %591, !dbg !13
  %601 = select i1 %600, i32 %599, i32 %588, !dbg !13
  %602 = shl i32 %601, 2, !dbg !13
  %603 = bitcast float %598 to i32, !dbg !13
  %604 = call i32 @llvm.amdgcn.ds.bpermute(i32 %602, i32 %603), !dbg !13
  %605 = bitcast i32 %604 to float, !dbg !13
  %606 = fadd float %598, %605, !dbg !13
  br i1 %106, label %607, label %613, !dbg !13

607:                                              ; preds = %585
  %608 = mul nsw i64 %552, 32, !dbg !13
  %609 = add i64 %608, %10, !dbg !13
  %610 = add i64 %609, %30, !dbg !13
  %611 = add i64 %610, 0, !dbg !13
  %612 = getelementptr float, ptr addrspace(3) @shm_3, i64 %611, !dbg !13
  store float %606, ptr addrspace(3) %612, align 4, !dbg !13
  br label %613, !dbg !13

613:                                              ; preds = %607, %585
  %614 = add i64 %552, 1, !dbg !13
  br label %551, !dbg !13

615:                                              ; preds = %551
  fence syncscope("workgroup") release, !dbg !26
  call void @llvm.amdgcn.s.barrier(), !dbg !26
  fence syncscope("workgroup") acquire, !dbg !26
  br label %616, !dbg !26

616:                                              ; preds = %619, %615
  %617 = phi i64 [ %620, %619 ], [ 0, %615 ], !dbg !26
  %618 = icmp slt i64 %617, 2, !dbg !26
  br i1 %618, label %619, label %621, !dbg !26

619:                                              ; preds = %616
  %620 = add i64 %617, 1, !dbg !26
  br label %616, !dbg !26

621:                                              ; preds = %616
  br label %622, !dbg !26

622:                                              ; preds = %625, %621
  %623 = phi i64 [ %626, %625 ], [ 0, %621 ], !dbg !26
  %624 = icmp slt i64 %623, 2, !dbg !26
  br i1 %624, label %625, label %627, !dbg !26

625:                                              ; preds = %622
  %626 = add i64 %623, 1, !dbg !26
  br label %622, !dbg !26

627:                                              ; preds = %622
  br label %628, !dbg !14

628:                                              ; preds = %654, %627
  %629 = phi i64 [ %655, %654 ], [ 0, %627 ], !dbg !14
  %630 = icmp slt i64 %629, 2, !dbg !14
  br i1 %630, label %631, label %656, !dbg !14

631:                                              ; preds = %628
  %632 = mul nsw i64 %629, 32, !dbg !14
  %633 = add i64 %632, %10, !dbg !14
  %634 = add i64 %633, %30, !dbg !14
  br label %635, !dbg !14

635:                                              ; preds = %638, %631
  %636 = phi i64 [ %653, %638 ], [ 0, %631 ], !dbg !14
  %637 = icmp slt i64 %636, 2, !dbg !14
  br i1 %637, label %638, label %654, !dbg !14

638:                                              ; preds = %635
  %639 = getelementptr float, ptr addrspace(5) %94, i64 0, !dbg !14
  %640 = load <4 x float>, ptr addrspace(5) %639, align 4, !dbg !14
  %641 = fptrunc <4 x float> %640 to <4 x half>, !dbg !14
  %642 = getelementptr half, ptr addrspace(5) %95, i64 0, !dbg !14
  store <4 x half> %641, ptr addrspace(5) %642, align 2, !dbg !14
  %643 = load <4 x half>, ptr addrspace(5) %642, align 2, !dbg !14
  %644 = mul nsw i64 %636, 16, !dbg !14
  %645 = add i64 %644, %32, !dbg !14
  %646 = sub i64 32, %645, !dbg !14
  %647 = insertelement <4 x i64> poison, i64 %646, i32 0, !dbg !14
  %648 = shufflevector <4 x i64> %647, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !14
  %649 = icmp sgt <4 x i64> %648, <i64 0, i64 1, i64 2, i64 3>, !dbg !14
  %650 = mul i64 %634, 32, !dbg !14
  %651 = add i64 %650, %645, !dbg !14
  %652 = getelementptr half, ptr addrspace(3) @shm_4, i64 %651, !dbg !14
  call void @llvm.masked.store.v4f16.p3(<4 x half> %643, ptr addrspace(3) %652, i32 2, <4 x i1> %649), !dbg !14
  %653 = add i64 %636, 1, !dbg !14
  br label %635, !dbg !14

654:                                              ; preds = %635
  %655 = add i64 %629, 1, !dbg !14
  br label %628, !dbg !14

656:                                              ; preds = %628
  br label %657, !dbg !27

657:                                              ; preds = %734, %656
  %658 = phi i64 [ %735, %734 ], [ 0, %656 ], !dbg !27
  %659 = icmp slt i64 %658, 16, !dbg !27
  br i1 %659, label %660, label %736, !dbg !27

660:                                              ; preds = %657
  %661 = icmp slt i64 %658, 0, !dbg !27
  %662 = sub i64 -1, %658, !dbg !27
  %663 = select i1 %661, i64 %662, i64 %658, !dbg !27
  %664 = sdiv i64 %663, 8, !dbg !27
  %665 = sub i64 -1, %664, !dbg !27
  %666 = select i1 %661, i64 %665, i64 %664, !dbg !27
  %667 = mul nsw i64 %666, 32, !dbg !27
  %668 = add i64 %667, %10, !dbg !27
  %669 = add i64 %668, %30, !dbg !27
  %670 = mul nsw i64 %658, 16, !dbg !27
  %671 = add i64 %670, %10, !dbg !27
  %672 = mul nsw i64 %666, -128, !dbg !27
  %673 = add i64 %671, %672, !dbg !27
  %674 = add i64 %673, %30, !dbg !27
  br label %675, !dbg !27

675:                                              ; preds = %679, %660
  %676 = phi i64 [ %733, %679 ], [ 0, %660 ], !dbg !27
  %677 = phi [1 x <4 x float>] [ %732, %679 ], [ zeroinitializer, %660 ], !dbg !27
  %678 = icmp slt i64 %676, 2, !dbg !27
  br i1 %678, label %679, label %734, !dbg !27

679:                                              ; preds = %675
  %680 = mul nsw i64 %676, 16, !dbg !27
  %681 = add i64 %680, %32, !dbg !27
  %682 = mul i64 %669, 32, !dbg !27
  %683 = add i64 %682, %681, !dbg !27
  %684 = getelementptr half, ptr addrspace(3) @shm_4, i64 %683, !dbg !27
  %685 = load half, ptr addrspace(3) %684, align 2, !dbg !27
  %686 = insertelement <4 x half> zeroinitializer, half %685, i64 0, !dbg !27
  %687 = add i64 %681, 1, !dbg !27
  %688 = add i64 %682, %687, !dbg !27
  %689 = getelementptr half, ptr addrspace(3) @shm_4, i64 %688, !dbg !27
  %690 = load half, ptr addrspace(3) %689, align 2, !dbg !27
  %691 = insertelement <4 x half> %686, half %690, i64 1, !dbg !27
  %692 = add i64 %681, 2, !dbg !27
  %693 = add i64 %682, %692, !dbg !27
  %694 = getelementptr half, ptr addrspace(3) @shm_4, i64 %693, !dbg !27
  %695 = load half, ptr addrspace(3) %694, align 2, !dbg !27
  %696 = insertelement <4 x half> %691, half %695, i64 2, !dbg !27
  %697 = add i64 %681, 3, !dbg !27
  %698 = add i64 %682, %697, !dbg !27
  %699 = getelementptr half, ptr addrspace(3) @shm_4, i64 %698, !dbg !27
  %700 = load half, ptr addrspace(3) %699, align 2, !dbg !27
  %701 = insertelement <4 x half> %696, half %700, i64 3, !dbg !27
  %702 = mul i64 %681, 128, !dbg !27
  %703 = add i64 %702, %674, !dbg !27
  %704 = getelementptr half, ptr addrspace(3) @shm_2, i64 %703, !dbg !27
  %705 = load half, ptr addrspace(3) %704, align 2, !dbg !27
  %706 = insertelement <1 x half> zeroinitializer, half %705, i64 0, !dbg !27
  %707 = mul i64 %687, 128, !dbg !27
  %708 = add i64 %707, %674, !dbg !27
  %709 = getelementptr half, ptr addrspace(3) @shm_2, i64 %708, !dbg !27
  %710 = load half, ptr addrspace(3) %709, align 2, !dbg !27
  %711 = insertelement <1 x half> zeroinitializer, half %710, i64 0, !dbg !27
  %712 = mul i64 %692, 128, !dbg !27
  %713 = add i64 %712, %674, !dbg !27
  %714 = getelementptr half, ptr addrspace(3) @shm_2, i64 %713, !dbg !27
  %715 = load half, ptr addrspace(3) %714, align 2, !dbg !27
  %716 = insertelement <1 x half> zeroinitializer, half %715, i64 0, !dbg !27
  %717 = mul i64 %697, 128, !dbg !27
  %718 = add i64 %717, %674, !dbg !27
  %719 = getelementptr half, ptr addrspace(3) @shm_2, i64 %718, !dbg !27
  %720 = load half, ptr addrspace(3) %719, align 2, !dbg !27
  %721 = insertelement <1 x half> zeroinitializer, half %720, i64 0, !dbg !27
  %722 = shufflevector <1 x half> %706, <1 x half> %706, <4 x i32> zeroinitializer, !dbg !27
  %723 = shufflevector <4 x half> %722, <4 x half> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !27
  %724 = shufflevector <1 x half> %711, <1 x half> %711, <4 x i32> zeroinitializer, !dbg !27
  %725 = shufflevector <4 x half> %724, <4 x half> %723, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !27
  %726 = shufflevector <1 x half> %716, <1 x half> %716, <4 x i32> zeroinitializer, !dbg !27
  %727 = shufflevector <4 x half> %726, <4 x half> %725, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !27
  %728 = shufflevector <1 x half> %721, <1 x half> %721, <4 x i32> zeroinitializer, !dbg !27
  %729 = shufflevector <4 x half> %728, <4 x half> %727, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !27
  %730 = extractvalue [1 x <4 x float>] %677, 0, !dbg !27
  %731 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %701, <4 x half> %729, <4 x float> %730), !dbg !27
  %732 = insertvalue [1 x <4 x float>] poison, <4 x float> %731, 0, !dbg !27
  %733 = add i64 %676, 1, !dbg !27
  br label %675, !dbg !27

734:                                              ; preds = %675
  %735 = add i64 %658, 1, !dbg !27
  br label %657, !dbg !27

736:                                              ; preds = %657
  br label %737, !dbg !28

737:                                              ; preds = %746, %736
  %738 = phi i64 [ %747, %746 ], [ 0, %736 ], !dbg !28
  %739 = icmp slt i64 %738, 2, !dbg !28
  br i1 %739, label %740, label %748, !dbg !28

740:                                              ; preds = %737
  br label %741, !dbg !28

741:                                              ; preds = %744, %740
  %742 = phi i64 [ %745, %744 ], [ 0, %740 ], !dbg !28
  %743 = icmp slt i64 %742, 8, !dbg !28
  br i1 %743, label %744, label %746, !dbg !28

744:                                              ; preds = %741
  %745 = add i64 %742, 1, !dbg !28
  br label %741, !dbg !28

746:                                              ; preds = %741
  %747 = add i64 %738, 1, !dbg !28
  br label %737, !dbg !28

748:                                              ; preds = %737
  br label %749, !dbg !29

749:                                              ; preds = %758, %748
  %750 = phi i64 [ %759, %758 ], [ 0, %748 ], !dbg !29
  %751 = icmp slt i64 %750, 2, !dbg !29
  br i1 %751, label %752, label %760, !dbg !29

752:                                              ; preds = %749
  br label %753, !dbg !29

753:                                              ; preds = %756, %752
  %754 = phi i64 [ %757, %756 ], [ 0, %752 ], !dbg !29
  %755 = icmp slt i64 %754, 8, !dbg !29
  br i1 %755, label %756, label %758, !dbg !29

756:                                              ; preds = %753
  %757 = add i64 %754, 1, !dbg !29
  br label %753, !dbg !29

758:                                              ; preds = %753
  %759 = add i64 %750, 1, !dbg !29
  br label %749, !dbg !29

760:                                              ; preds = %749
  br label %761, !dbg !30

761:                                              ; preds = %764, %760
  %762 = phi i64 [ %765, %764 ], [ 0, %760 ], !dbg !30
  %763 = icmp slt i64 %762, 2, !dbg !30
  br i1 %763, label %764, label %766, !dbg !30

764:                                              ; preds = %761
  %765 = add i64 %762, 1, !dbg !30
  br label %761, !dbg !30

766:                                              ; preds = %761
  %767 = add i64 %115, 1, !dbg !15
  br label %114, !dbg !15

768:                                              ; preds = %114
  fence syncscope("workgroup") release, !dbg !31
  call void @llvm.amdgcn.s.barrier(), !dbg !31
  fence syncscope("workgroup") acquire, !dbg !31
  %769 = mul nsw i64 %10, -64, !dbg !31
  br label %770, !dbg !31

770:                                              ; preds = %922, %768
  %771 = phi i64 [ %923, %922 ], [ 0, %768 ], !dbg !31
  %772 = icmp slt i64 %771, 16, !dbg !31
  br i1 %772, label %773, label %924, !dbg !31

773:                                              ; preds = %770
  %774 = mul nsw i64 %771, -4, !dbg !31
  %775 = add i64 %774, %769, !dbg !31
  %776 = add i64 %775, 8191, !dbg !31
  %777 = icmp sge i64 %776, 0, !dbg !31
  br i1 %777, label %778, label %811, !dbg !31

778:                                              ; preds = %773
  %779 = sdiv i64 %26, 2, !dbg !31
  %780 = sub i64 -1, %779, !dbg !31
  %781 = select i1 %24, i64 %780, i64 %779, !dbg !31
  %782 = srem i64 %781, 64, !dbg !31
  %783 = icmp slt i64 %782, 0, !dbg !31
  %784 = add i64 %782, 64, !dbg !31
  %785 = select i1 %783, i64 %784, i64 %782, !dbg !31
  %786 = mul nsw i64 %771, 4, !dbg !31
  %787 = mul nsw i64 %10, 64, !dbg !31
  %788 = add i64 %786, %787, !dbg !31
  %789 = mul nsw i64 %781, -128, !dbg !31
  %790 = add i64 %788, %789, !dbg !31
  %791 = mul i64 %785, 128, !dbg !31
  %792 = add i64 %791, %790, !dbg !31
  %793 = getelementptr half, ptr addrspace(3) @shm_5, i64 %792, !dbg !31
  %794 = load half, ptr addrspace(3) %793, align 2, !dbg !31
  %795 = add i64 %23, %781, !dbg !31
  %796 = icmp slt i64 %781, 0, !dbg !31
  %797 = sub i64 -1, %781, !dbg !31
  %798 = select i1 %796, i64 %797, i64 %781, !dbg !31
  %799 = sdiv i64 %798, 64, !dbg !31
  %800 = sub i64 -1, %799, !dbg !31
  %801 = select i1 %796, i64 %800, i64 %799, !dbg !31
  %802 = mul nsw i64 %801, -64, !dbg !31
  %803 = add i64 %795, %802, !dbg !31
  %804 = add i64 %803, %31, !dbg !31
  %805 = mul i64 %22, 524288, !dbg !31
  %806 = add i64 0, %805, !dbg !31
  %807 = mul i64 %804, 128, !dbg !31
  %808 = add i64 %806, %807, !dbg !31
  %809 = add i64 %808, %790, !dbg !31
  %810 = getelementptr half, ptr addrspace(1) %3, i64 %809, !dbg !31
  store half %794, ptr addrspace(1) %810, align 2, !dbg !31
  br label %811, !dbg !31

811:                                              ; preds = %778, %773
  %812 = add i64 %775, 8190, !dbg !31
  %813 = icmp sge i64 %812, 0, !dbg !31
  br i1 %813, label %814, label %848, !dbg !31

814:                                              ; preds = %811
  %815 = sdiv i64 %26, 2, !dbg !31
  %816 = sub i64 -1, %815, !dbg !31
  %817 = select i1 %24, i64 %816, i64 %815, !dbg !31
  %818 = srem i64 %817, 64, !dbg !31
  %819 = icmp slt i64 %818, 0, !dbg !31
  %820 = add i64 %818, 64, !dbg !31
  %821 = select i1 %819, i64 %820, i64 %818, !dbg !31
  %822 = mul nsw i64 %771, 4, !dbg !31
  %823 = mul nsw i64 %10, 64, !dbg !31
  %824 = add i64 %822, %823, !dbg !31
  %825 = mul nsw i64 %817, -128, !dbg !31
  %826 = add i64 %824, %825, !dbg !31
  %827 = add i64 %826, 1, !dbg !31
  %828 = mul i64 %821, 128, !dbg !31
  %829 = add i64 %828, %827, !dbg !31
  %830 = getelementptr half, ptr addrspace(3) @shm_5, i64 %829, !dbg !31
  %831 = load half, ptr addrspace(3) %830, align 2, !dbg !31
  %832 = add i64 %23, %817, !dbg !31
  %833 = icmp slt i64 %817, 0, !dbg !31
  %834 = sub i64 -1, %817, !dbg !31
  %835 = select i1 %833, i64 %834, i64 %817, !dbg !31
  %836 = sdiv i64 %835, 64, !dbg !31
  %837 = sub i64 -1, %836, !dbg !31
  %838 = select i1 %833, i64 %837, i64 %836, !dbg !31
  %839 = mul nsw i64 %838, -64, !dbg !31
  %840 = add i64 %832, %839, !dbg !31
  %841 = add i64 %840, %31, !dbg !31
  %842 = mul i64 %22, 524288, !dbg !31
  %843 = add i64 0, %842, !dbg !31
  %844 = mul i64 %841, 128, !dbg !31
  %845 = add i64 %843, %844, !dbg !31
  %846 = add i64 %845, %827, !dbg !31
  %847 = getelementptr half, ptr addrspace(1) %3, i64 %846, !dbg !31
  store half %831, ptr addrspace(1) %847, align 2, !dbg !31
  br label %848, !dbg !31

848:                                              ; preds = %814, %811
  %849 = add i64 %775, 8189, !dbg !31
  %850 = icmp sge i64 %849, 0, !dbg !31
  br i1 %850, label %851, label %885, !dbg !31

851:                                              ; preds = %848
  %852 = sdiv i64 %26, 2, !dbg !31
  %853 = sub i64 -1, %852, !dbg !31
  %854 = select i1 %24, i64 %853, i64 %852, !dbg !31
  %855 = srem i64 %854, 64, !dbg !31
  %856 = icmp slt i64 %855, 0, !dbg !31
  %857 = add i64 %855, 64, !dbg !31
  %858 = select i1 %856, i64 %857, i64 %855, !dbg !31
  %859 = mul nsw i64 %771, 4, !dbg !31
  %860 = mul nsw i64 %10, 64, !dbg !31
  %861 = add i64 %859, %860, !dbg !31
  %862 = mul nsw i64 %854, -128, !dbg !31
  %863 = add i64 %861, %862, !dbg !31
  %864 = add i64 %863, 2, !dbg !31
  %865 = mul i64 %858, 128, !dbg !31
  %866 = add i64 %865, %864, !dbg !31
  %867 = getelementptr half, ptr addrspace(3) @shm_5, i64 %866, !dbg !31
  %868 = load half, ptr addrspace(3) %867, align 2, !dbg !31
  %869 = add i64 %23, %854, !dbg !31
  %870 = icmp slt i64 %854, 0, !dbg !31
  %871 = sub i64 -1, %854, !dbg !31
  %872 = select i1 %870, i64 %871, i64 %854, !dbg !31
  %873 = sdiv i64 %872, 64, !dbg !31
  %874 = sub i64 -1, %873, !dbg !31
  %875 = select i1 %870, i64 %874, i64 %873, !dbg !31
  %876 = mul nsw i64 %875, -64, !dbg !31
  %877 = add i64 %869, %876, !dbg !31
  %878 = add i64 %877, %31, !dbg !31
  %879 = mul i64 %22, 524288, !dbg !31
  %880 = add i64 0, %879, !dbg !31
  %881 = mul i64 %878, 128, !dbg !31
  %882 = add i64 %880, %881, !dbg !31
  %883 = add i64 %882, %864, !dbg !31
  %884 = getelementptr half, ptr addrspace(1) %3, i64 %883, !dbg !31
  store half %868, ptr addrspace(1) %884, align 2, !dbg !31
  br label %885, !dbg !31

885:                                              ; preds = %851, %848
  %886 = add i64 %775, 8188, !dbg !31
  %887 = icmp sge i64 %886, 0, !dbg !31
  br i1 %887, label %888, label %922, !dbg !31

888:                                              ; preds = %885
  %889 = sdiv i64 %26, 2, !dbg !31
  %890 = sub i64 -1, %889, !dbg !31
  %891 = select i1 %24, i64 %890, i64 %889, !dbg !31
  %892 = srem i64 %891, 64, !dbg !31
  %893 = icmp slt i64 %892, 0, !dbg !31
  %894 = add i64 %892, 64, !dbg !31
  %895 = select i1 %893, i64 %894, i64 %892, !dbg !31
  %896 = mul nsw i64 %771, 4, !dbg !31
  %897 = mul nsw i64 %10, 64, !dbg !31
  %898 = add i64 %896, %897, !dbg !31
  %899 = mul nsw i64 %891, -128, !dbg !31
  %900 = add i64 %898, %899, !dbg !31
  %901 = add i64 %900, 3, !dbg !31
  %902 = mul i64 %895, 128, !dbg !31
  %903 = add i64 %902, %901, !dbg !31
  %904 = getelementptr half, ptr addrspace(3) @shm_5, i64 %903, !dbg !31
  %905 = load half, ptr addrspace(3) %904, align 2, !dbg !31
  %906 = add i64 %23, %891, !dbg !31
  %907 = icmp slt i64 %891, 0, !dbg !31
  %908 = sub i64 -1, %891, !dbg !31
  %909 = select i1 %907, i64 %908, i64 %891, !dbg !31
  %910 = sdiv i64 %909, 64, !dbg !31
  %911 = sub i64 -1, %910, !dbg !31
  %912 = select i1 %907, i64 %911, i64 %910, !dbg !31
  %913 = mul nsw i64 %912, -64, !dbg !31
  %914 = add i64 %906, %913, !dbg !31
  %915 = add i64 %914, %31, !dbg !31
  %916 = mul i64 %22, 524288, !dbg !31
  %917 = add i64 0, %916, !dbg !31
  %918 = mul i64 %915, 128, !dbg !31
  %919 = add i64 %917, %918, !dbg !31
  %920 = add i64 %919, %901, !dbg !31
  %921 = getelementptr half, ptr addrspace(1) %3, i64 %920, !dbg !31
  store half %905, ptr addrspace(1) %921, align 2, !dbg !31
  br label %922, !dbg !31

922:                                              ; preds = %888, %885
  %923 = add i64 %771, 1, !dbg !31
  br label %770, !dbg !31

924:                                              ; preds = %770
  ret void, !dbg !32
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

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "amdgpu-waves-per-eu"="1" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "temp.mlir", directory: "3rd/deepgengraph/build")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 3, type: !4, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 128, i32 1, i32 1}
!7 = !DILocation(line: 5, column: 19, scope: !3)
!8 = !DILocation(line: 6, column: 19, scope: !3)
!9 = !DILocation(line: 7, column: 20, scope: !3)
!10 = !DILocation(line: 13, column: 10, scope: !3)
!11 = !DILocation(line: 16, column: 5, scope: !3)
!12 = !DILocation(line: 18, column: 10, scope: !3)
!13 = !DILocation(line: 45, column: 7, scope: !3)
!14 = !DILocation(line: 48, column: 7, scope: !3)
!15 = !DILocation(line: 25, column: 12, scope: !3)
!16 = !DILocation(line: 27, column: 7, scope: !3)
!17 = !DILocation(line: 29, column: 7, scope: !3)
!18 = !DILocation(line: 25, column: 23, scope: !3)
!19 = !DILocation(line: 30, column: 13, scope: !3)
!20 = !DILocation(line: 31, column: 13, scope: !3)
!21 = !DILocation(line: 33, column: 15, scope: !3)
!22 = !DILocation(line: 34, column: 15, scope: !3)
!23 = !DILocation(line: 35, column: 15, scope: !3)
!24 = !DILocation(line: 42, column: 13, scope: !3)
!25 = !DILocation(line: 43, column: 13, scope: !3)
!26 = !DILocation(line: 46, column: 13, scope: !3)
!27 = !DILocation(line: 49, column: 13, scope: !3)
!28 = !DILocation(line: 50, column: 13, scope: !3)
!29 = !DILocation(line: 51, column: 7, scope: !3)
!30 = !DILocation(line: 52, column: 7, scope: !3)
!31 = !DILocation(line: 58, column: 5, scope: !3)
!32 = !DILocation(line: 59, column: 5, scope: !3)
