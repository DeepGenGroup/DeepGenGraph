; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

@shm_5 = addrspace(3) global [128 x [1 x float]] undef, align 16
@shm_4 = addrspace(3) global [128 x [128 x float]] undef, align 16
@shm_3 = addrspace(3) global [128 x [128 x half]] undef, align 16
@shm_2 = addrspace(3) global [128 x [128 x half]] undef, align 16
@shm_1 = addrspace(3) global [128 x [128 x half]] undef, align 16
@shm_0 = addrspace(3) global [128 x [128 x half]] undef, align 16

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #0 !reqd_work_group_size !1 {
  %5 = call i32 @llvm.amdgcn.workgroup.id.y()
  %6 = call i32 @llvm.amdgcn.workgroup.id.x()
  %7 = call i32 @llvm.amdgcn.workitem.id.x()
  %8 = mul nsw i32 %5, 128
  %9 = icmp slt i32 %6, 0
  %10 = sub i32 -1, %6
  %11 = select i1 %9, i32 %10, i32 %6
  %12 = sdiv i32 %11, 32
  %13 = sub i32 -1, %12
  %14 = select i1 %9, i32 %13, i32 %12
  %15 = add i32 %8, %14
  %16 = srem i32 %15, 4096
  %17 = icmp slt i32 %16, 0
  %18 = add i32 %16, 4096
  %19 = select i1 %17, i32 %18, i32 %16
  %20 = mul nsw i32 %14, -32
  br label %21

21:                                               ; preds = %42, %4
  %22 = phi i32 [ %43, %42 ], [ 0, %4 ]
  %23 = icmp slt i32 %22, 128
  br i1 %23, label %24, label %44

24:                                               ; preds = %21
  %25 = add i32 %22, %6
  %26 = add i32 %25, %20
  br label %27

27:                                               ; preds = %30, %24
  %28 = phi i32 [ %41, %30 ], [ 0, %24 ]
  %29 = icmp slt i32 %28, 128
  br i1 %29, label %30, label %42

30:                                               ; preds = %27
  %31 = mul i32 %19, 4096
  %32 = add i32 0, %31
  %33 = mul i32 %26, 128
  %34 = add i32 %32, %33
  %35 = add i32 %34, %28
  %36 = getelementptr half, ptr addrspace(1) %0, i32 %35
  %37 = load half, ptr addrspace(1) %36, align 2
  %38 = mul i32 %22, 128
  %39 = add i32 %38, %28
  %40 = getelementptr half, ptr addrspace(3) @shm_0, i32 %39
  store half %37, ptr addrspace(3) %40, align 2
  %41 = add i32 %28, 1
  br label %27

42:                                               ; preds = %27
  %43 = add i32 %22, 1
  br label %21

44:                                               ; preds = %21
  %45 = icmp slt i32 %7, 0
  %46 = sub i32 -1, %7
  %47 = select i1 %45, i32 %46, i32 %7
  %48 = sdiv i32 %47, 16
  %49 = sub i32 -1, %48
  %50 = select i1 %45, i32 %49, i32 %48
  %51 = mul nsw i32 %50, -16
  %52 = mul nsw i32 %50, 4
  br label %53

53:                                               ; preds = %94, %44
  %54 = phi i32 [ %95, %94 ], [ 0, %44 ]
  %55 = icmp slt i32 %54, 4
  br i1 %55, label %56, label %96

56:                                               ; preds = %53
  br label %57

57:                                               ; preds = %92, %56
  %58 = phi i32 [ %93, %92 ], [ 0, %56 ]
  %59 = icmp slt i32 %58, 32
  br i1 %59, label %60, label %94

60:                                               ; preds = %57
  %61 = icmp slt i32 %58, 0
  %62 = sub i32 -1, %58
  %63 = select i1 %61, i32 %62, i32 %58
  %64 = sdiv i32 %63, 4
  %65 = sub i32 -1, %64
  %66 = select i1 %61, i32 %65, i32 %64
  %67 = mul nsw i32 %66, -4
  br label %68

68:                                               ; preds = %90, %60
  %69 = phi i32 [ %91, %90 ], [ 0, %60 ]
  %70 = icmp slt i32 %69, 4
  br i1 %70, label %71, label %92

71:                                               ; preds = %68
  %72 = mul nsw i32 %69, 32
  %73 = add i32 %72, %7
  %74 = add i32 %73, %51
  br label %75

75:                                               ; preds = %78, %71
  %76 = phi i32 [ %89, %78 ], [ 0, %71 ]
  %77 = icmp slt i32 %76, 8
  br i1 %77, label %78, label %90

78:                                               ; preds = %75
  %79 = mul nsw i32 %76, 16
  %80 = add i32 %79, %58
  %81 = add i32 %80, %52
  %82 = add i32 %81, %67
  %83 = mul i32 %74, 128
  %84 = add i32 %83, %82
  %85 = getelementptr half, ptr addrspace(3) @shm_0, i32 %84
  %86 = load half, ptr addrspace(3) %85, align 2
  %87 = fmul half %86, 0xH3015
  %88 = getelementptr half, ptr addrspace(3) @shm_1, i32 %84
  store half %87, ptr addrspace(3) %88, align 2
  %89 = add i32 %76, 1
  br label %75

90:                                               ; preds = %75
  %91 = add i32 %69, 1
  br label %68

92:                                               ; preds = %68
  %93 = add i32 %58, 1
  br label %57

94:                                               ; preds = %57
  %95 = add i32 %54, 1
  br label %53

96:                                               ; preds = %53
  %97 = alloca float, i32 128, align 4
  br label %98

98:                                               ; preds = %110, %96
  %99 = phi i32 [ %111, %110 ], [ 0, %96 ]
  %100 = icmp slt i32 %99, 4
  br i1 %100, label %101, label %112

101:                                              ; preds = %98
  br label %102

102:                                              ; preds = %105, %101
  %103 = phi i32 [ %109, %105 ], [ 0, %101 ]
  %104 = icmp slt i32 %103, 32
  br i1 %104, label %105, label %110

105:                                              ; preds = %102
  %106 = mul i32 %99, 32
  %107 = add i32 %106, %103
  %108 = getelementptr float, ptr %97, i32 %107
  store float 0.000000e+00, ptr %108, align 4
  %109 = add i32 %103, 1
  br label %102

110:                                              ; preds = %102
  %111 = add i32 %99, 1
  br label %98

112:                                              ; preds = %98
  %113 = alloca float, i32 4, align 4
  br label %114

114:                                              ; preds = %117, %112
  %115 = phi i32 [ %120, %117 ], [ 0, %112 ]
  %116 = icmp slt i32 %115, 4
  br i1 %116, label %117, label %121

117:                                              ; preds = %114
  %118 = add i32 %115, 0
  %119 = getelementptr float, ptr %113, i32 %118
  store float 0.000000e+00, ptr %119, align 4
  %120 = add i32 %115, 1
  br label %114

121:                                              ; preds = %114
  %122 = alloca half, i32 4, align 1
  %123 = alloca half, i32 4, align 1
  %124 = alloca float, i32 128, align 1
  %125 = alloca float, i32 4, align 1
  %126 = alloca float, align 4
  %127 = alloca float, i32 4, align 1
  %128 = alloca half, i32 4, align 1
  %129 = alloca half, i32 4, align 1
  %130 = alloca half, i32 4, align 1
  %131 = alloca float, i32 128, align 1
  %132 = alloca float, i32 4, align 1
  %133 = add i32 %8, 128
  %134 = icmp slt i32 %14, 0
  %135 = sub i32 -1, %14
  %136 = select i1 %134, i32 %135, i32 %14
  %137 = sdiv i32 %136, 4096
  %138 = sub i32 -1, %137
  %139 = select i1 %134, i32 %138, i32 %137
  %140 = mul nsw i32 %139, -4096
  %141 = srem i32 %7, 64
  %142 = icmp slt i32 %141, 0
  %143 = add i32 %141, 64
  %144 = select i1 %142, i32 %143, i32 %141
  %145 = icmp slt i32 %144, 0
  %146 = sub i32 -1, %144
  %147 = select i1 %145, i32 %146, i32 %144
  %148 = sdiv i32 %147, 16
  %149 = sub i32 -1, %148
  %150 = select i1 %145, i32 %149, i32 %148
  %151 = icmp eq i32 %150, 0
  br label %152

152:                                              ; preds = %603, %121
  %153 = phi i32 [ %604, %603 ], [ 0, %121 ]
  %154 = icmp slt i32 %153, %133
  br i1 %154, label %155, label %605

155:                                              ; preds = %152
  %156 = icmp slt i32 %153, 0
  %157 = sub i32 -1, %153
  %158 = select i1 %156, i32 %157, i32 %153
  %159 = sdiv i32 %158, 128
  %160 = sub i32 -1, %159
  %161 = select i1 %156, i32 %160, i32 %159
  %162 = mul nsw i32 %161, 128
  %163 = add i32 %162, %14
  %164 = add i32 %163, %140
  br label %165

165:                                              ; preds = %186, %155
  %166 = phi i32 [ %187, %186 ], [ 0, %155 ]
  %167 = icmp slt i32 %166, 128
  br i1 %167, label %168, label %188

168:                                              ; preds = %165
  %169 = add i32 %166, %6
  %170 = add i32 %169, %20
  br label %171

171:                                              ; preds = %174, %168
  %172 = phi i32 [ %185, %174 ], [ 0, %168 ]
  %173 = icmp slt i32 %172, 128
  br i1 %173, label %174, label %186

174:                                              ; preds = %171
  %175 = mul i32 %164, 4096
  %176 = add i32 0, %175
  %177 = mul i32 %170, 128
  %178 = add i32 %176, %177
  %179 = add i32 %178, %172
  %180 = getelementptr half, ptr addrspace(1) %2, i32 %179
  %181 = load half, ptr addrspace(1) %180, align 2
  %182 = mul i32 %166, 128
  %183 = add i32 %182, %172
  %184 = getelementptr half, ptr addrspace(3) @shm_0, i32 %183
  store half %181, ptr addrspace(3) %184, align 2
  %185 = add i32 %172, 1
  br label %171

186:                                              ; preds = %171
  %187 = add i32 %166, 1
  br label %165

188:                                              ; preds = %165
  br label %189

189:                                              ; preds = %210, %188
  %190 = phi i32 [ %211, %210 ], [ 0, %188 ]
  %191 = icmp slt i32 %190, 128
  br i1 %191, label %192, label %212

192:                                              ; preds = %189
  %193 = add i32 %190, %6
  %194 = add i32 %193, %20
  br label %195

195:                                              ; preds = %198, %192
  %196 = phi i32 [ %209, %198 ], [ 0, %192 ]
  %197 = icmp slt i32 %196, 128
  br i1 %197, label %198, label %210

198:                                              ; preds = %195
  %199 = mul i32 %164, 4096
  %200 = add i32 0, %199
  %201 = mul i32 %194, 128
  %202 = add i32 %200, %201
  %203 = add i32 %202, %196
  %204 = getelementptr half, ptr addrspace(1) %1, i32 %203
  %205 = load half, ptr addrspace(1) %204, align 2
  %206 = mul i32 %190, 128
  %207 = add i32 %206, %196
  %208 = getelementptr half, ptr addrspace(3) @shm_2, i32 %207
  store half %205, ptr addrspace(3) %208, align 2
  %209 = add i32 %196, 1
  br label %195

210:                                              ; preds = %195
  %211 = add i32 %190, 1
  br label %189

212:                                              ; preds = %189
  br label %213

213:                                              ; preds = %299, %212
  %214 = phi i32 [ %300, %299 ], [ 0, %212 ]
  %215 = icmp slt i32 %214, 4
  br i1 %215, label %216, label %301

216:                                              ; preds = %213
  %217 = mul nsw i32 %214, 128
  %218 = add i32 %217, %7
  %219 = add i32 %218, %51
  br label %220

220:                                              ; preds = %297, %216
  %221 = phi i32 [ %298, %297 ], [ 0, %216 ]
  %222 = icmp slt i32 %221, 8
  br i1 %222, label %223, label %299

223:                                              ; preds = %220
  br label %224

224:                                              ; preds = %227, %223
  %225 = phi i32 [ %230, %227 ], [ 0, %223 ]
  %226 = icmp slt i32 %225, 4
  br i1 %226, label %227, label %231

227:                                              ; preds = %224
  %228 = add i32 0, %225
  %229 = getelementptr float, ptr %125, i32 %228
  store float 0.000000e+00, ptr %229, align 4
  %230 = add i32 %225, 1
  br label %224

231:                                              ; preds = %224
  %232 = mul nsw i32 %221, 16
  %233 = add i32 %232, %7
  %234 = add i32 %233, %51
  br label %235

235:                                              ; preds = %267, %231
  %236 = phi i32 [ %275, %267 ], [ 0, %231 ]
  %237 = icmp slt i32 %236, 8
  br i1 %237, label %238, label %276

238:                                              ; preds = %235
  %239 = mul nsw i32 %236, 16
  br label %240

240:                                              ; preds = %243, %238
  %241 = phi i32 [ %252, %243 ], [ 0, %238 ]
  %242 = icmp slt i32 %241, 4
  br i1 %242, label %243, label %253

243:                                              ; preds = %240
  %244 = add i32 %241, %239
  %245 = add i32 %244, %52
  %246 = mul i32 %219, 128
  %247 = add i32 %246, %245
  %248 = getelementptr half, ptr addrspace(3) @shm_1, i32 %247
  %249 = load half, ptr addrspace(3) %248, align 2
  %250 = add i32 0, %241
  %251 = getelementptr half, ptr %122, i32 %250
  store half %249, ptr %251, align 2
  %252 = add i32 %241, 1
  br label %240

253:                                              ; preds = %240
  br label %254

254:                                              ; preds = %257, %253
  %255 = phi i32 [ %266, %257 ], [ 0, %253 ]
  %256 = icmp slt i32 %255, 4
  br i1 %256, label %257, label %267

257:                                              ; preds = %254
  %258 = add i32 %255, %239
  %259 = add i32 %258, %52
  %260 = mul i32 %259, 128
  %261 = add i32 %260, %234
  %262 = getelementptr half, ptr addrspace(3) @shm_0, i32 %261
  %263 = load half, ptr addrspace(3) %262, align 2
  %264 = add i32 %255, 0
  %265 = getelementptr half, ptr %123, i32 %264
  store half %263, ptr %265, align 2
  %266 = add i32 %255, 1
  br label %254

267:                                              ; preds = %254
  %268 = getelementptr half, ptr %122, i32 0
  %269 = load <4 x half>, ptr %268, align 2
  %270 = getelementptr half, ptr %123, i32 0
  %271 = load <4 x half>, ptr %270, align 2
  %272 = getelementptr float, ptr %125, i32 0
  %273 = load <4 x float>, ptr %272, align 4
  %274 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %269, <4 x half> %271, <4 x float> %273)
  store <4 x float> %274, ptr %272, align 4
  %275 = add i32 %236, 1
  br label %235

276:                                              ; preds = %235
  %277 = mul nsw i32 %221, 4
  br label %278

278:                                              ; preds = %295, %276
  %279 = phi i32 [ %296, %295 ], [ 0, %276 ]
  %280 = icmp slt i32 %279, 4
  br i1 %280, label %281, label %297

281:                                              ; preds = %278
  br label %282

282:                                              ; preds = %285, %281
  %283 = phi i32 [ %294, %285 ], [ 0, %281 ]
  %284 = icmp slt i32 %283, 4
  br i1 %284, label %285, label %295

285:                                              ; preds = %282
  %286 = add i32 0, %283
  %287 = getelementptr float, ptr %125, i32 %286
  %288 = load float, ptr %287, align 4
  %289 = add i32 %283, %277
  %290 = add i32 %289, %279
  %291 = mul i32 %214, 32
  %292 = add i32 %291, %290
  %293 = getelementptr float, ptr %124, i32 %292
  store float %288, ptr %293, align 4
  %294 = add i32 %283, 1
  br label %282

295:                                              ; preds = %282
  %296 = add i32 %279, 1
  br label %278

297:                                              ; preds = %278
  %298 = add i32 %221, 1
  br label %220

299:                                              ; preds = %220
  %300 = add i32 %214, 1
  br label %213

301:                                              ; preds = %213
  br label %302

302:                                              ; preds = %356, %301
  %303 = phi i32 [ %357, %356 ], [ 0, %301 ]
  %304 = icmp slt i32 %303, 4
  br i1 %304, label %305, label %358

305:                                              ; preds = %302
  br label %306

306:                                              ; preds = %354, %305
  %307 = phi i32 [ %355, %354 ], [ 0, %305 ]
  %308 = icmp slt i32 %307, 32
  br i1 %308, label %309, label %356

309:                                              ; preds = %306
  %310 = mul nsw i32 %307, 4
  %311 = icmp slt i32 %307, 0
  %312 = sub i32 -1, %307
  %313 = select i1 %311, i32 %312, i32 %307
  %314 = sdiv i32 %313, 4
  %315 = sub i32 -1, %314
  %316 = select i1 %311, i32 %315, i32 %314
  %317 = mul nsw i32 %316, -16
  br label %318

318:                                              ; preds = %352, %309
  %319 = phi i32 [ %353, %352 ], [ 0, %309 ]
  %320 = icmp slt i32 %319, 4
  br i1 %320, label %321, label %354

321:                                              ; preds = %318
  %322 = mul nsw i32 %319, 32
  %323 = add i32 %322, %7
  %324 = add i32 %323, %8
  %325 = add i32 %324, %51
  %326 = add i32 %325, 1
  %327 = add i32 %323, %51
  br label %328

328:                                              ; preds = %331, %321
  %329 = phi i32 [ %351, %331 ], [ 0, %321 ]
  %330 = icmp slt i32 %329, 8
  br i1 %330, label %331, label %352

331:                                              ; preds = %328
  %332 = mul nsw i32 %329, 16
  %333 = add i32 %153, %332
  %334 = add i32 %333, %310
  %335 = add i32 %334, %317
  %336 = add i32 %335, %50
  %337 = icmp ule i32 %326, %336
  %338 = select i1 %337, float 0xFFF0000000000000, float 0.000000e+00
  %339 = mul i32 %303, 32
  %340 = add i32 %339, %307
  %341 = getelementptr float, ptr %124, i32 %340
  %342 = load float, ptr %341, align 4
  %343 = fadd float %342, %338
  %344 = call float @__ocml_exp2_f32(float %343)
  %345 = add i32 %332, %310
  %346 = add i32 %345, %317
  %347 = add i32 %346, %50
  %348 = mul i32 %327, 128
  %349 = add i32 %348, %347
  %350 = getelementptr float, ptr addrspace(3) @shm_4, i32 %349
  store float %344, ptr addrspace(3) %350, align 4
  %351 = add i32 %329, 1
  br label %328

352:                                              ; preds = %328
  %353 = add i32 %319, 1
  br label %318

354:                                              ; preds = %318
  %355 = add i32 %307, 1
  br label %306

356:                                              ; preds = %306
  %357 = add i32 %303, 1
  br label %302

358:                                              ; preds = %302
  br label %359

359:                                              ; preds = %404, %358
  %360 = phi i32 [ %405, %404 ], [ 0, %358 ]
  %361 = icmp slt i32 %360, 4
  br i1 %361, label %362, label %406

362:                                              ; preds = %359
  store float 0.000000e+00, ptr %126, align 4
  %363 = mul nsw i32 %360, 32
  %364 = add i32 %363, %7
  %365 = add i32 %364, %51
  br label %366

366:                                              ; preds = %369, %362
  %367 = phi i32 [ %378, %369 ], [ 0, %362 ]
  %368 = icmp slt i32 %367, 32
  br i1 %368, label %369, label %379

369:                                              ; preds = %366
  %370 = load float, ptr %126, align 4
  %371 = mul nsw i32 %367, 4
  %372 = add i32 %371, %50
  %373 = mul i32 %365, 128
  %374 = add i32 %373, %372
  %375 = getelementptr float, ptr addrspace(3) @shm_4, i32 %374
  %376 = load float, ptr addrspace(3) %375, align 4
  %377 = fadd float %370, %376
  store float %377, ptr %126, align 4
  %378 = add i32 %367, 1
  br label %366

379:                                              ; preds = %366
  %380 = load float, ptr %126, align 4
  %381 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %382 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %381)
  %383 = add i32 %382, 64
  %384 = xor i32 %382, 16
  %385 = and i32 %383, -64
  %386 = icmp slt i32 %384, %385
  %387 = select i1 %386, i32 %384, i32 %382
  %388 = shl i32 %387, 2
  %389 = bitcast float %380 to i32
  %390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %388, i32 %389)
  %391 = bitcast i32 %390 to float
  %392 = fadd float %380, %391
  %393 = xor i32 %382, 32
  %394 = icmp slt i32 %393, %385
  %395 = select i1 %394, i32 %393, i32 %382
  %396 = shl i32 %395, 2
  %397 = bitcast float %392 to i32
  %398 = call i32 @llvm.amdgcn.ds.bpermute(i32 %396, i32 %397)
  %399 = bitcast i32 %398 to float
  %400 = fadd float %392, %399
  br i1 %151, label %401, label %404

401:                                              ; preds = %379
  %402 = add i32 %365, 0
  %403 = getelementptr float, ptr addrspace(3) @shm_5, i32 %402
  store float %400, ptr addrspace(3) %403, align 4
  br label %404

404:                                              ; preds = %401, %379
  %405 = add i32 %360, 1
  br label %359

406:                                              ; preds = %359
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  br label %407

407:                                              ; preds = %426, %406
  %408 = phi i32 [ %427, %426 ], [ 0, %406 ]
  %409 = icmp slt i32 %408, 4
  br i1 %409, label %410, label %428

410:                                              ; preds = %407
  br label %411

411:                                              ; preds = %414, %410
  %412 = phi i32 [ %425, %414 ], [ 0, %410 ]
  %413 = icmp slt i32 %412, 4
  br i1 %413, label %414, label %426

414:                                              ; preds = %411
  %415 = mul nsw i32 %412, 32
  %416 = add i32 %415, %7
  %417 = add i32 %416, %51
  %418 = add i32 %417, 0
  %419 = getelementptr float, ptr addrspace(3) @shm_5, i32 %418
  %420 = load float, ptr addrspace(3) %419, align 4
  %421 = add i32 %408, 0
  %422 = getelementptr float, ptr %113, i32 %421
  %423 = load float, ptr %422, align 4
  %424 = fadd float %423, %420
  store float %424, ptr %422, align 4
  %425 = add i32 %412, 1
  br label %411

426:                                              ; preds = %411
  %427 = add i32 %408, 1
  br label %407

428:                                              ; preds = %407
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  br label %429

429:                                              ; preds = %481, %428
  %430 = phi i32 [ %482, %481 ], [ 0, %428 ]
  %431 = icmp slt i32 %430, 4
  br i1 %431, label %432, label %483

432:                                              ; preds = %429
  %433 = mul nsw i32 %430, 32
  %434 = add i32 %433, %7
  %435 = add i32 %434, %51
  br label %436

436:                                              ; preds = %479, %432
  %437 = phi i32 [ %480, %479 ], [ 0, %432 ]
  %438 = icmp slt i32 %437, 8
  br i1 %438, label %439, label %481

439:                                              ; preds = %436
  %440 = mul nsw i32 %437, 16
  br label %441

441:                                              ; preds = %444, %439
  %442 = phi i32 [ %453, %444 ], [ 0, %439 ]
  %443 = icmp slt i32 %442, 4
  br i1 %443, label %444, label %454

444:                                              ; preds = %441
  %445 = add i32 %442, %440
  %446 = add i32 %445, %52
  %447 = mul i32 %435, 128
  %448 = add i32 %447, %446
  %449 = getelementptr float, ptr addrspace(3) @shm_4, i32 %448
  %450 = load float, ptr addrspace(3) %449, align 4
  %451 = add i32 0, %442
  %452 = getelementptr float, ptr %127, i32 %451
  store float %450, ptr %452, align 4
  %453 = add i32 %442, 1
  br label %441

454:                                              ; preds = %441
  br label %455

455:                                              ; preds = %458, %454
  %456 = phi i32 [ %464, %458 ], [ 0, %454 ]
  %457 = icmp slt i32 %456, 4
  br i1 %457, label %458, label %465

458:                                              ; preds = %455
  %459 = add i32 0, %456
  %460 = getelementptr float, ptr %127, i32 %459
  %461 = load float, ptr %460, align 4
  %462 = fptrunc float %461 to half
  %463 = getelementptr half, ptr %128, i32 %459
  store half %462, ptr %463, align 2
  %464 = add i32 %456, 1
  br label %455

465:                                              ; preds = %455
  br label %466

466:                                              ; preds = %469, %465
  %467 = phi i32 [ %478, %469 ], [ 0, %465 ]
  %468 = icmp slt i32 %467, 4
  br i1 %468, label %469, label %479

469:                                              ; preds = %466
  %470 = add i32 0, %467
  %471 = getelementptr half, ptr %128, i32 %470
  %472 = load half, ptr %471, align 2
  %473 = add i32 %467, %440
  %474 = add i32 %473, %52
  %475 = mul i32 %435, 128
  %476 = add i32 %475, %474
  %477 = getelementptr half, ptr addrspace(3) @shm_3, i32 %476
  store half %472, ptr addrspace(3) %477, align 2
  %478 = add i32 %467, 1
  br label %466

479:                                              ; preds = %466
  %480 = add i32 %437, 1
  br label %436

481:                                              ; preds = %436
  %482 = add i32 %430, 1
  br label %429

483:                                              ; preds = %429
  br label %484

484:                                              ; preds = %570, %483
  %485 = phi i32 [ %571, %570 ], [ 0, %483 ]
  %486 = icmp slt i32 %485, 4
  br i1 %486, label %487, label %572

487:                                              ; preds = %484
  %488 = mul nsw i32 %485, 32
  %489 = add i32 %488, %7
  %490 = add i32 %489, %51
  br label %491

491:                                              ; preds = %568, %487
  %492 = phi i32 [ %569, %568 ], [ 0, %487 ]
  %493 = icmp slt i32 %492, 8
  br i1 %493, label %494, label %570

494:                                              ; preds = %491
  br label %495

495:                                              ; preds = %498, %494
  %496 = phi i32 [ %501, %498 ], [ 0, %494 ]
  %497 = icmp slt i32 %496, 4
  br i1 %497, label %498, label %502

498:                                              ; preds = %495
  %499 = add i32 0, %496
  %500 = getelementptr float, ptr %132, i32 %499
  store float 0.000000e+00, ptr %500, align 4
  %501 = add i32 %496, 1
  br label %495

502:                                              ; preds = %495
  %503 = mul nsw i32 %492, 16
  %504 = add i32 %503, %7
  %505 = add i32 %504, %51
  br label %506

506:                                              ; preds = %538, %502
  %507 = phi i32 [ %546, %538 ], [ 0, %502 ]
  %508 = icmp slt i32 %507, 8
  br i1 %508, label %509, label %547

509:                                              ; preds = %506
  %510 = mul nsw i32 %507, 16
  br label %511

511:                                              ; preds = %514, %509
  %512 = phi i32 [ %523, %514 ], [ 0, %509 ]
  %513 = icmp slt i32 %512, 4
  br i1 %513, label %514, label %524

514:                                              ; preds = %511
  %515 = add i32 %512, %510
  %516 = add i32 %515, %52
  %517 = mul i32 %490, 128
  %518 = add i32 %517, %516
  %519 = getelementptr half, ptr addrspace(3) @shm_3, i32 %518
  %520 = load half, ptr addrspace(3) %519, align 2
  %521 = add i32 0, %512
  %522 = getelementptr half, ptr %129, i32 %521
  store half %520, ptr %522, align 2
  %523 = add i32 %512, 1
  br label %511

524:                                              ; preds = %511
  br label %525

525:                                              ; preds = %528, %524
  %526 = phi i32 [ %537, %528 ], [ 0, %524 ]
  %527 = icmp slt i32 %526, 4
  br i1 %527, label %528, label %538

528:                                              ; preds = %525
  %529 = add i32 %526, %510
  %530 = add i32 %529, %52
  %531 = mul i32 %530, 128
  %532 = add i32 %531, %505
  %533 = getelementptr half, ptr addrspace(3) @shm_2, i32 %532
  %534 = load half, ptr addrspace(3) %533, align 2
  %535 = add i32 %526, 0
  %536 = getelementptr half, ptr %130, i32 %535
  store half %534, ptr %536, align 2
  %537 = add i32 %526, 1
  br label %525

538:                                              ; preds = %525
  %539 = getelementptr half, ptr %129, i32 0
  %540 = load <4 x half>, ptr %539, align 2
  %541 = getelementptr half, ptr %130, i32 0
  %542 = load <4 x half>, ptr %541, align 2
  %543 = getelementptr float, ptr %132, i32 0
  %544 = load <4 x float>, ptr %543, align 4
  %545 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %540, <4 x half> %542, <4 x float> %544)
  store <4 x float> %545, ptr %543, align 4
  %546 = add i32 %507, 1
  br label %506

547:                                              ; preds = %506
  %548 = mul nsw i32 %492, 4
  br label %549

549:                                              ; preds = %566, %547
  %550 = phi i32 [ %567, %566 ], [ 0, %547 ]
  %551 = icmp slt i32 %550, 4
  br i1 %551, label %552, label %568

552:                                              ; preds = %549
  br label %553

553:                                              ; preds = %556, %552
  %554 = phi i32 [ %565, %556 ], [ 0, %552 ]
  %555 = icmp slt i32 %554, 4
  br i1 %555, label %556, label %566

556:                                              ; preds = %553
  %557 = add i32 0, %554
  %558 = getelementptr float, ptr %132, i32 %557
  %559 = load float, ptr %558, align 4
  %560 = add i32 %554, %548
  %561 = add i32 %560, %550
  %562 = mul i32 %485, 32
  %563 = add i32 %562, %561
  %564 = getelementptr float, ptr %131, i32 %563
  store float %559, ptr %564, align 4
  %565 = add i32 %554, 1
  br label %553

566:                                              ; preds = %553
  %567 = add i32 %550, 1
  br label %549

568:                                              ; preds = %549
  %569 = add i32 %492, 1
  br label %491

570:                                              ; preds = %491
  %571 = add i32 %485, 1
  br label %484

572:                                              ; preds = %484
  br label %573

573:                                              ; preds = %601, %572
  %574 = phi i32 [ %602, %601 ], [ 0, %572 ]
  %575 = icmp slt i32 %574, 4
  br i1 %575, label %576, label %603

576:                                              ; preds = %573
  br label %577

577:                                              ; preds = %599, %576
  %578 = phi i32 [ %600, %599 ], [ 0, %576 ]
  %579 = icmp slt i32 %578, 32
  br i1 %579, label %580, label %601

580:                                              ; preds = %577
  br label %581

581:                                              ; preds = %597, %580
  %582 = phi i32 [ %598, %597 ], [ 0, %580 ]
  %583 = icmp slt i32 %582, 4
  br i1 %583, label %584, label %599

584:                                              ; preds = %581
  br label %585

585:                                              ; preds = %588, %584
  %586 = phi i32 [ %596, %588 ], [ 0, %584 ]
  %587 = icmp slt i32 %586, 8
  br i1 %587, label %588, label %597

588:                                              ; preds = %585
  %589 = mul i32 %574, 32
  %590 = add i32 %589, %578
  %591 = getelementptr float, ptr %97, i32 %590
  %592 = load float, ptr %591, align 4
  %593 = getelementptr float, ptr %131, i32 %590
  %594 = load float, ptr %593, align 4
  %595 = fadd float %592, %594
  store float %595, ptr %591, align 4
  %596 = add i32 %586, 1
  br label %585

597:                                              ; preds = %585
  %598 = add i32 %582, 1
  br label %581

599:                                              ; preds = %581
  %600 = add i32 %578, 1
  br label %577

601:                                              ; preds = %577
  %602 = add i32 %574, 1
  br label %573

603:                                              ; preds = %573
  %604 = add i32 %153, 128
  br label %152

605:                                              ; preds = %152
  %606 = alloca half, i32 128, align 2
  br label %607

607:                                              ; preds = %638, %605
  %608 = phi i32 [ %639, %638 ], [ 0, %605 ]
  %609 = icmp slt i32 %608, 4
  br i1 %609, label %610, label %640

610:                                              ; preds = %607
  br label %611

611:                                              ; preds = %636, %610
  %612 = phi i32 [ %637, %636 ], [ 0, %610 ]
  %613 = icmp slt i32 %612, 32
  br i1 %613, label %614, label %638

614:                                              ; preds = %611
  br label %615

615:                                              ; preds = %634, %614
  %616 = phi i32 [ %635, %634 ], [ 0, %614 ]
  %617 = icmp slt i32 %616, 4
  br i1 %617, label %618, label %636

618:                                              ; preds = %615
  br label %619

619:                                              ; preds = %622, %618
  %620 = phi i32 [ %633, %622 ], [ 0, %618 ]
  %621 = icmp slt i32 %620, 8
  br i1 %621, label %622, label %634

622:                                              ; preds = %619
  %623 = mul i32 %608, 32
  %624 = add i32 %623, %612
  %625 = getelementptr float, ptr %97, i32 %624
  %626 = load float, ptr %625, align 4
  %627 = add i32 %608, 0
  %628 = getelementptr float, ptr %113, i32 %627
  %629 = load float, ptr %628, align 4
  %630 = fdiv float %626, %629
  %631 = fptrunc float %630 to half
  %632 = getelementptr half, ptr %606, i32 %624
  store half %631, ptr %632, align 2
  %633 = add i32 %620, 1
  br label %619

634:                                              ; preds = %619
  %635 = add i32 %616, 1
  br label %615

636:                                              ; preds = %615
  %637 = add i32 %612, 1
  br label %611

638:                                              ; preds = %611
  %639 = add i32 %608, 1
  br label %607

640:                                              ; preds = %607
  br label %641

641:                                              ; preds = %662, %640
  %642 = phi i32 [ %663, %662 ], [ 0, %640 ]
  %643 = icmp slt i32 %642, 4
  br i1 %643, label %644, label %664

644:                                              ; preds = %641
  %645 = add i32 %642, %6
  %646 = add i32 %645, %20
  br label %647

647:                                              ; preds = %650, %644
  %648 = phi i32 [ %661, %650 ], [ 0, %644 ]
  %649 = icmp slt i32 %648, 32
  br i1 %649, label %650, label %662

650:                                              ; preds = %647
  %651 = mul i32 %642, 32
  %652 = add i32 %651, %648
  %653 = getelementptr half, ptr %606, i32 %652
  %654 = load half, ptr %653, align 2
  %655 = mul i32 %19, 4096
  %656 = add i32 0, %655
  %657 = mul i32 %646, 128
  %658 = add i32 %656, %657
  %659 = add i32 %658, %648
  %660 = getelementptr half, ptr addrspace(1) %3, i32 %659
  store half %654, ptr addrspace(1) %660, align 2
  %661 = add i32 %648, 1
  br label %647

662:                                              ; preds = %647
  %663 = add i32 %642, 1
  br label %641

664:                                              ; preds = %641
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 128, i32 1, i32 1}
