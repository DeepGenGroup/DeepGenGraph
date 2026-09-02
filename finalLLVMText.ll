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
  %5 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %6 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %7 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %8 = icmp slt i32 %5, 0, !dbg !8
  %9 = sub i32 -1, %5, !dbg !8
  %10 = select i1 %8, i32 %9, i32 %5, !dbg !8
  %11 = sdiv i32 %10, 64, !dbg !8
  %12 = sub i32 -1, %11, !dbg !8
  %13 = select i1 %8, i32 %12, i32 %11, !dbg !8
  %14 = add i32 %6, %13, !dbg !8
  %15 = srem i32 %14, 32, !dbg !8
  %16 = icmp slt i32 %15, 0, !dbg !8
  %17 = add i32 %15, 32, !dbg !8
  %18 = select i1 %16, i32 %17, i32 %15, !dbg !8
  %19 = mul nsw i32 %5, 64, !dbg !8
  %20 = icmp slt i32 %7, 0, !dbg !8
  %21 = sub i32 -1, %7, !dbg !8
  %22 = select i1 %20, i32 %21, i32 %7, !dbg !8
  %23 = sdiv i32 %22, 16, !dbg !8
  %24 = sub i32 -1, %23, !dbg !8
  %25 = select i1 %20, i32 %24, i32 %23, !dbg !8
  %26 = mul nsw i32 %25, -16, !dbg !8
  %27 = mul nsw i32 %13, -4096, !dbg !8
  %28 = mul nsw i32 %25, 4, !dbg !8
  br label %29, !dbg !8

29:                                               ; preds = %64, %4
  %30 = phi i32 [ %65, %64 ], [ 0, %4 ], !dbg !8
  %31 = icmp slt i32 %30, 4, !dbg !8
  br i1 %31, label %32, label %66, !dbg !8

32:                                               ; preds = %29
  %33 = mul nsw i32 %30, 16, !dbg !8
  %34 = add i32 %33, %19, !dbg !8
  %35 = add i32 %34, %7, !dbg !8
  %36 = add i32 %35, %26, !dbg !8
  %37 = add i32 %36, %27, !dbg !8
  %38 = add i32 %33, %7, !dbg !8
  %39 = add i32 %38, %26, !dbg !8
  br label %40, !dbg !8

40:                                               ; preds = %62, %32
  %41 = phi i32 [ %63, %62 ], [ 0, %32 ], !dbg !8
  %42 = icmp slt i32 %41, 8, !dbg !8
  br i1 %42, label %43, label %64, !dbg !8

43:                                               ; preds = %40
  %44 = mul nsw i32 %41, 16, !dbg !8
  br label %45, !dbg !8

45:                                               ; preds = %48, %43
  %46 = phi i32 [ %61, %48 ], [ 0, %43 ], !dbg !8
  %47 = icmp slt i32 %46, 4, !dbg !8
  br i1 %47, label %48, label %62, !dbg !8

48:                                               ; preds = %45
  %49 = add i32 %44, %46, !dbg !8
  %50 = add i32 %49, %28, !dbg !8
  %51 = mul i32 %18, 524288, !dbg !8
  %52 = add i32 0, %51, !dbg !8
  %53 = mul i32 %37, 128, !dbg !8
  %54 = add i32 %52, %53, !dbg !8
  %55 = add i32 %54, %50, !dbg !8
  %56 = getelementptr half, ptr addrspace(1) %0, i32 %55, !dbg !8
  %57 = load half, ptr addrspace(1) %56, align 2, !dbg !8
  %58 = mul i32 %39, 128, !dbg !8
  %59 = add i32 %58, %50, !dbg !8
  %60 = getelementptr half, ptr addrspace(3) @shm_0, i32 %59, !dbg !8
  store half %57, ptr addrspace(3) %60, align 2, !dbg !8
  %61 = add i32 %46, 1, !dbg !8
  br label %45, !dbg !8

62:                                               ; preds = %45
  %63 = add i32 %41, 1, !dbg !8
  br label %40, !dbg !8

64:                                               ; preds = %40
  %65 = add i32 %30, 1, !dbg !8
  br label %29, !dbg !8

66:                                               ; preds = %29
  br label %67, !dbg !9

67:                                               ; preds = %116, %66
  %68 = phi i32 [ %117, %116 ], [ 0, %66 ], !dbg !9
  %69 = icmp slt i32 %68, 4, !dbg !9
  br i1 %69, label %70, label %118, !dbg !9

70:                                               ; preds = %67
  %71 = mul nsw i32 %68, 16, !dbg !9
  %72 = add i32 %71, %7, !dbg !9
  %73 = add i32 %72, %26, !dbg !9
  br label %74, !dbg !9

74:                                               ; preds = %114, %70
  %75 = phi i32 [ %115, %114 ], [ 0, %70 ], !dbg !9
  %76 = icmp slt i32 %75, 32, !dbg !9
  br i1 %76, label %77, label %116, !dbg !9

77:                                               ; preds = %74
  %78 = icmp slt i32 %75, 0, !dbg !9
  %79 = sub i32 -1, %75, !dbg !9
  %80 = select i1 %78, i32 %79, i32 %75, !dbg !9
  %81 = sdiv i32 %80, 4, !dbg !9
  %82 = sub i32 -1, %81, !dbg !9
  %83 = select i1 %78, i32 %82, i32 %81, !dbg !9
  %84 = mul nsw i32 %83, -4, !dbg !9
  %85 = mul nsw i32 %83, 12, !dbg !9
  %86 = add i32 %75, %85, !dbg !9
  %87 = add i32 %86, %28, !dbg !9
  br label %88, !dbg !9

88:                                               ; preds = %112, %77
  %89 = phi i32 [ %113, %112 ], [ 0, %77 ], !dbg !9
  %90 = icmp slt i32 %89, 4, !dbg !9
  br i1 %90, label %91, label %114, !dbg !9

91:                                               ; preds = %88
  %92 = mul nsw i32 %89, 16, !dbg !9
  %93 = add i32 %92, %7, !dbg !9
  %94 = add i32 %93, %26, !dbg !9
  br label %95, !dbg !9

95:                                               ; preds = %98, %91
  %96 = phi i32 [ %111, %98 ], [ 0, %91 ], !dbg !9
  %97 = icmp slt i32 %96, 8, !dbg !9
  br i1 %97, label %98, label %112, !dbg !9

98:                                               ; preds = %95
  %99 = mul nsw i32 %96, 16, !dbg !9
  %100 = add i32 %99, %75, !dbg !9
  %101 = add i32 %100, %28, !dbg !9
  %102 = add i32 %101, %84, !dbg !9
  %103 = mul i32 %94, 128, !dbg !9
  %104 = add i32 %103, %102, !dbg !9
  %105 = getelementptr half, ptr addrspace(3) @shm_0, i32 %104, !dbg !9
  %106 = load half, ptr addrspace(3) %105, align 2, !dbg !9
  %107 = fmul half %106, 0xH3015, !dbg !9
  %108 = mul i32 %73, 128, !dbg !9
  %109 = add i32 %108, %87, !dbg !9
  %110 = getelementptr half, ptr addrspace(3) @shm_1, i32 %109, !dbg !9
  store half %107, ptr addrspace(3) %110, align 2, !dbg !9
  %111 = add i32 %96, 1, !dbg !9
  br label %95, !dbg !9

112:                                              ; preds = %95
  %113 = add i32 %89, 1, !dbg !9
  br label %88, !dbg !9

114:                                              ; preds = %88
  %115 = add i32 %75, 1, !dbg !9
  br label %74, !dbg !9

116:                                              ; preds = %74
  %117 = add i32 %68, 1, !dbg !9
  br label %67, !dbg !9

118:                                              ; preds = %67
  %119 = alloca float, i32 128, align 4, !dbg !10
  br label %120, !dbg !10

120:                                              ; preds = %132, %118
  %121 = phi i32 [ %133, %132 ], [ 0, %118 ], !dbg !10
  %122 = icmp slt i32 %121, 4, !dbg !10
  br i1 %122, label %123, label %134, !dbg !10

123:                                              ; preds = %120
  br label %124, !dbg !10

124:                                              ; preds = %127, %123
  %125 = phi i32 [ %131, %127 ], [ 0, %123 ], !dbg !10
  %126 = icmp slt i32 %125, 32, !dbg !10
  br i1 %126, label %127, label %132, !dbg !10

127:                                              ; preds = %124
  %128 = mul i32 %121, 32, !dbg !10
  %129 = add i32 %128, %125, !dbg !10
  %130 = getelementptr float, ptr %119, i32 %129, !dbg !10
  store float 0.000000e+00, ptr %130, align 4, !dbg !10
  %131 = add i32 %125, 1, !dbg !10
  br label %124, !dbg !10

132:                                              ; preds = %124
  %133 = add i32 %121, 1, !dbg !10
  br label %120, !dbg !10

134:                                              ; preds = %120
  %135 = alloca float, i32 4, align 4, !dbg !11
  br label %136, !dbg !11

136:                                              ; preds = %139, %134
  %137 = phi i32 [ %142, %139 ], [ 0, %134 ], !dbg !11
  %138 = icmp slt i32 %137, 4, !dbg !11
  br i1 %138, label %139, label %143, !dbg !11

139:                                              ; preds = %136
  %140 = add i32 %137, 0, !dbg !11
  %141 = getelementptr float, ptr %135, i32 %140, !dbg !11
  store float 0.000000e+00, ptr %141, align 4, !dbg !11
  %142 = add i32 %137, 1, !dbg !11
  br label %136, !dbg !11

143:                                              ; preds = %136
  %144 = alloca half, i32 4, align 1, !dbg !12
  %145 = alloca half, i32 4, align 1, !dbg !12
  %146 = alloca float, i32 32, align 1, !dbg !12
  %147 = alloca float, i32 4, align 1, !dbg !12
  %148 = alloca float, align 4, !dbg !13
  %149 = alloca float, i32 4, align 1, !dbg !14
  %150 = alloca half, i32 4, align 1, !dbg !14
  %151 = alloca half, i32 4, align 1, !dbg !15
  %152 = alloca half, i32 4, align 1, !dbg !15
  %153 = alloca float, i32 128, align 1, !dbg !15
  %154 = alloca float, i32 4, align 1, !dbg !15
  %155 = mul nsw i32 %5, 2, !dbg !16
  %156 = add i32 %155, 1, !dbg !16
  %157 = mul nsw i32 %5, 256, !dbg !17
  %158 = mul nsw i32 %6, 128, !dbg !17
  %159 = add i32 %155, %158, !dbg !17
  %160 = srem i32 %7, 64, !dbg !13
  %161 = icmp slt i32 %160, 0, !dbg !13
  %162 = add i32 %160, 64, !dbg !13
  %163 = select i1 %161, i32 %162, i32 %160, !dbg !13
  %164 = icmp slt i32 %163, 0, !dbg !13
  %165 = sub i32 -1, %163, !dbg !13
  %166 = select i1 %164, i32 %165, i32 %163, !dbg !13
  %167 = sdiv i32 %166, 16, !dbg !13
  %168 = sub i32 -1, %167, !dbg !13
  %169 = select i1 %164, i32 %168, i32 %167, !dbg !13
  %170 = icmp eq i32 %169, 0, !dbg !13
  br label %171, !dbg !16

171:                                              ; preds = %688, %143
  %172 = phi i32 [ %689, %688 ], [ 0, %143 ], !dbg !16
  %173 = icmp slt i32 %172, %156, !dbg !16
  br i1 %173, label %174, label %690, !dbg !16

174:                                              ; preds = %171
  %175 = add i32 %172, %155, !dbg !18
  %176 = icmp slt i32 %175, 0, !dbg !18
  %177 = sub i32 -1, %175, !dbg !18
  %178 = select i1 %176, i32 %177, i32 %175, !dbg !18
  %179 = sdiv i32 %178, 128, !dbg !18
  %180 = sub i32 -1, %179, !dbg !18
  %181 = select i1 %176, i32 %180, i32 %179, !dbg !18
  %182 = add i32 %181, %6, !dbg !18
  %183 = srem i32 %182, 32, !dbg !18
  %184 = icmp slt i32 %183, 0, !dbg !18
  %185 = add i32 %183, 32, !dbg !18
  %186 = select i1 %184, i32 %185, i32 %183, !dbg !18
  %187 = mul nsw i32 %172, 32, !dbg !18
  %188 = mul nsw i32 %181, -4096, !dbg !18
  br label %189, !dbg !18

189:                                              ; preds = %225, %174
  %190 = phi i32 [ %226, %225 ], [ 0, %174 ], !dbg !18
  %191 = icmp slt i32 %190, 8, !dbg !18
  br i1 %191, label %192, label %227, !dbg !18

192:                                              ; preds = %189
  %193 = mul nsw i32 %190, 16, !dbg !18
  %194 = add i32 %187, %193, !dbg !18
  br label %195, !dbg !18

195:                                              ; preds = %223, %192
  %196 = phi i32 [ %224, %223 ], [ 0, %192 ], !dbg !18
  %197 = icmp slt i32 %196, 2, !dbg !18
  br i1 %197, label %198, label %225, !dbg !18

198:                                              ; preds = %195
  %199 = mul nsw i32 %196, 16, !dbg !18
  %200 = add i32 %199, %7, !dbg !18
  %201 = add i32 %200, %26, !dbg !18
  br label %202, !dbg !18

202:                                              ; preds = %205, %198
  %203 = phi i32 [ %222, %205 ], [ 0, %198 ], !dbg !18
  %204 = icmp slt i32 %203, 4, !dbg !18
  br i1 %204, label %205, label %223, !dbg !18

205:                                              ; preds = %202
  %206 = add i32 %194, %203, !dbg !18
  %207 = add i32 %206, %19, !dbg !18
  %208 = add i32 %207, %188, !dbg !18
  %209 = add i32 %208, %28, !dbg !18
  %210 = mul i32 %186, 524288, !dbg !18
  %211 = add i32 0, %210, !dbg !18
  %212 = mul i32 %209, 128, !dbg !18
  %213 = add i32 %211, %212, !dbg !18
  %214 = add i32 %213, %201, !dbg !18
  %215 = getelementptr half, ptr addrspace(1) %2, i32 %214, !dbg !18
  %216 = load half, ptr addrspace(1) %215, align 2, !dbg !18
  %217 = add i32 %193, %203, !dbg !18
  %218 = add i32 %217, %28, !dbg !18
  %219 = mul i32 %218, 32, !dbg !18
  %220 = add i32 %219, %201, !dbg !18
  %221 = getelementptr half, ptr addrspace(3) @shm_0, i32 %220, !dbg !18
  store half %216, ptr addrspace(3) %221, align 2, !dbg !18
  %222 = add i32 %203, 1, !dbg !18
  br label %202, !dbg !18

223:                                              ; preds = %202
  %224 = add i32 %196, 1, !dbg !18
  br label %195, !dbg !18

225:                                              ; preds = %195
  %226 = add i32 %190, 1, !dbg !18
  br label %189, !dbg !18

227:                                              ; preds = %189
  %228 = add i32 %172, %157, !dbg !17
  %229 = icmp slt i32 %228, 0, !dbg !17
  %230 = sub i32 -1, %228, !dbg !17
  %231 = select i1 %229, i32 %230, i32 %228, !dbg !17
  %232 = sdiv i32 %231, 16384, !dbg !17
  %233 = sub i32 -1, %232, !dbg !17
  %234 = select i1 %229, i32 %233, i32 %232, !dbg !17
  %235 = add i32 %234, %6, !dbg !17
  %236 = srem i32 %235, 32, !dbg !17
  %237 = icmp slt i32 %236, 0, !dbg !17
  %238 = add i32 %236, 32, !dbg !17
  %239 = select i1 %237, i32 %238, i32 %236, !dbg !17
  %240 = icmp slt i32 %172, 0, !dbg !17
  %241 = sub i32 -1, %172, !dbg !17
  %242 = select i1 %240, i32 %241, i32 %172, !dbg !17
  %243 = sdiv i32 %242, 128, !dbg !17
  %244 = sub i32 -1, %243, !dbg !17
  %245 = select i1 %240, i32 %244, i32 %243, !dbg !17
  %246 = add i32 %245, %159, !dbg !17
  %247 = icmp slt i32 %246, 0, !dbg !17
  %248 = sub i32 -1, %246, !dbg !17
  %249 = select i1 %247, i32 %248, i32 %246, !dbg !17
  %250 = sdiv i32 %249, 128, !dbg !17
  %251 = sub i32 -1, %250, !dbg !17
  %252 = select i1 %247, i32 %251, i32 %250, !dbg !17
  %253 = mul nsw i32 %252, -128, !dbg !17
  %254 = mul nsw i32 %245, -4096, !dbg !17
  br label %255, !dbg !17

255:                                              ; preds = %295, %227
  %256 = phi i32 [ %296, %295 ], [ 0, %227 ], !dbg !17
  %257 = icmp slt i32 %256, 2, !dbg !17
  br i1 %257, label %258, label %297, !dbg !17

258:                                              ; preds = %255
  %259 = mul nsw i32 %256, 16, !dbg !17
  br label %260, !dbg !17

260:                                              ; preds = %293, %258
  %261 = phi i32 [ %294, %293 ], [ 0, %258 ], !dbg !17
  %262 = icmp slt i32 %261, 8, !dbg !17
  br i1 %262, label %263, label %295, !dbg !17

263:                                              ; preds = %260
  %264 = mul nsw i32 %261, 16, !dbg !17
  %265 = add i32 %187, %264, !dbg !17
  %266 = add i32 %265, %7, !dbg !17
  %267 = add i32 %266, %254, !dbg !17
  %268 = add i32 %267, %26, !dbg !17
  %269 = add i32 %264, %7, !dbg !17
  %270 = add i32 %269, %26, !dbg !17
  br label %271, !dbg !17

271:                                              ; preds = %274, %263
  %272 = phi i32 [ %292, %274 ], [ 0, %263 ], !dbg !17
  %273 = icmp slt i32 %272, 4, !dbg !17
  br i1 %273, label %274, label %293, !dbg !17

274:                                              ; preds = %271
  %275 = add i32 %259, %272, !dbg !17
  %276 = add i32 %275, %155, !dbg !17
  %277 = add i32 %276, %158, !dbg !17
  %278 = add i32 %277, %245, !dbg !17
  %279 = add i32 %278, %253, !dbg !17
  %280 = add i32 %279, %28, !dbg !17
  %281 = mul i32 %239, 524288, !dbg !17
  %282 = add i32 0, %281, !dbg !17
  %283 = mul i32 %280, 4096, !dbg !17
  %284 = add i32 %282, %283, !dbg !17
  %285 = add i32 %284, %268, !dbg !17
  %286 = getelementptr half, ptr addrspace(1) %1, i32 %285, !dbg !17
  %287 = load half, ptr addrspace(1) %286, align 2, !dbg !17
  %288 = add i32 %275, %28, !dbg !17
  %289 = mul i32 %288, 128, !dbg !17
  %290 = add i32 %289, %270, !dbg !17
  %291 = getelementptr half, ptr addrspace(3) @shm_2, i32 %290, !dbg !17
  store half %287, ptr addrspace(3) %291, align 2, !dbg !17
  %292 = add i32 %272, 1, !dbg !17
  br label %271, !dbg !17

293:                                              ; preds = %271
  %294 = add i32 %261, 1, !dbg !17
  br label %260, !dbg !17

295:                                              ; preds = %260
  %296 = add i32 %256, 1, !dbg !17
  br label %255, !dbg !17

297:                                              ; preds = %255
  br label %298, !dbg !12

298:                                              ; preds = %384, %297
  %299 = phi i32 [ %385, %384 ], [ 0, %297 ], !dbg !12
  %300 = icmp slt i32 %299, 4, !dbg !12
  br i1 %300, label %301, label %386, !dbg !12

301:                                              ; preds = %298
  %302 = mul nsw i32 %299, 64, !dbg !12
  %303 = add i32 %302, %7, !dbg !12
  %304 = add i32 %303, %26, !dbg !12
  br label %305, !dbg !12

305:                                              ; preds = %382, %301
  %306 = phi i32 [ %383, %382 ], [ 0, %301 ], !dbg !12
  %307 = icmp slt i32 %306, 2, !dbg !12
  br i1 %307, label %308, label %384, !dbg !12

308:                                              ; preds = %305
  br label %309, !dbg !12

309:                                              ; preds = %312, %308
  %310 = phi i32 [ %315, %312 ], [ 0, %308 ], !dbg !12
  %311 = icmp slt i32 %310, 4, !dbg !12
  br i1 %311, label %312, label %316, !dbg !12

312:                                              ; preds = %309
  %313 = add i32 0, %310, !dbg !12
  %314 = getelementptr float, ptr %147, i32 %313, !dbg !12
  store float 0.000000e+00, ptr %314, align 4, !dbg !12
  %315 = add i32 %310, 1, !dbg !12
  br label %309, !dbg !12

316:                                              ; preds = %309
  %317 = mul nsw i32 %306, 16, !dbg !12
  %318 = add i32 %317, %7, !dbg !12
  %319 = add i32 %318, %26, !dbg !12
  br label %320, !dbg !12

320:                                              ; preds = %352, %316
  %321 = phi i32 [ %360, %352 ], [ 0, %316 ], !dbg !12
  %322 = icmp slt i32 %321, 8, !dbg !12
  br i1 %322, label %323, label %361, !dbg !12

323:                                              ; preds = %320
  %324 = mul nsw i32 %321, 16, !dbg !12
  br label %325, !dbg !12

325:                                              ; preds = %328, %323
  %326 = phi i32 [ %337, %328 ], [ 0, %323 ], !dbg !12
  %327 = icmp slt i32 %326, 4, !dbg !12
  br i1 %327, label %328, label %338, !dbg !12

328:                                              ; preds = %325
  %329 = add i32 %326, %324, !dbg !12
  %330 = add i32 %329, %28, !dbg !12
  %331 = mul i32 %304, 128, !dbg !12
  %332 = add i32 %331, %330, !dbg !12
  %333 = getelementptr half, ptr addrspace(3) @shm_1, i32 %332, !dbg !12
  %334 = load half, ptr addrspace(3) %333, align 2, !dbg !12
  %335 = add i32 0, %326, !dbg !12
  %336 = getelementptr half, ptr %144, i32 %335, !dbg !12
  store half %334, ptr %336, align 2, !dbg !12
  %337 = add i32 %326, 1, !dbg !12
  br label %325, !dbg !12

338:                                              ; preds = %325
  br label %339, !dbg !12

339:                                              ; preds = %342, %338
  %340 = phi i32 [ %351, %342 ], [ 0, %338 ], !dbg !12
  %341 = icmp slt i32 %340, 4, !dbg !12
  br i1 %341, label %342, label %352, !dbg !12

342:                                              ; preds = %339
  %343 = add i32 %340, %324, !dbg !12
  %344 = add i32 %343, %28, !dbg !12
  %345 = mul i32 %344, 32, !dbg !12
  %346 = add i32 %345, %319, !dbg !12
  %347 = getelementptr half, ptr addrspace(3) @shm_0, i32 %346, !dbg !12
  %348 = load half, ptr addrspace(3) %347, align 2, !dbg !12
  %349 = add i32 %340, 0, !dbg !12
  %350 = getelementptr half, ptr %145, i32 %349, !dbg !12
  store half %348, ptr %350, align 2, !dbg !12
  %351 = add i32 %340, 1, !dbg !12
  br label %339, !dbg !12

352:                                              ; preds = %339
  %353 = getelementptr half, ptr %144, i32 0, !dbg !12
  %354 = load <4 x half>, ptr %353, align 2, !dbg !12
  %355 = getelementptr half, ptr %145, i32 0, !dbg !12
  %356 = load <4 x half>, ptr %355, align 2, !dbg !12
  %357 = getelementptr float, ptr %147, i32 0, !dbg !12
  %358 = load <4 x float>, ptr %357, align 4, !dbg !12
  %359 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %354, <4 x half> %356, <4 x float> %358), !dbg !12
  store <4 x float> %359, ptr %357, align 4, !dbg !12
  %360 = add i32 %321, 1, !dbg !12
  br label %320, !dbg !12

361:                                              ; preds = %320
  %362 = mul nsw i32 %306, 4, !dbg !12
  br label %363, !dbg !12

363:                                              ; preds = %380, %361
  %364 = phi i32 [ %381, %380 ], [ 0, %361 ], !dbg !12
  %365 = icmp slt i32 %364, 4, !dbg !12
  br i1 %365, label %366, label %382, !dbg !12

366:                                              ; preds = %363
  br label %367, !dbg !12

367:                                              ; preds = %370, %366
  %368 = phi i32 [ %379, %370 ], [ 0, %366 ], !dbg !12
  %369 = icmp slt i32 %368, 4, !dbg !12
  br i1 %369, label %370, label %380, !dbg !12

370:                                              ; preds = %367
  %371 = add i32 0, %368, !dbg !12
  %372 = getelementptr float, ptr %147, i32 %371, !dbg !12
  %373 = load float, ptr %372, align 4, !dbg !12
  %374 = add i32 %368, %362, !dbg !12
  %375 = add i32 %374, %364, !dbg !12
  %376 = mul i32 %299, 8, !dbg !12
  %377 = add i32 %376, %375, !dbg !12
  %378 = getelementptr float, ptr %146, i32 %377, !dbg !12
  store float %373, ptr %378, align 4, !dbg !12
  %379 = add i32 %368, 1, !dbg !12
  br label %367, !dbg !12

380:                                              ; preds = %367
  %381 = add i32 %364, 1, !dbg !12
  br label %363, !dbg !12

382:                                              ; preds = %363
  %383 = add i32 %306, 1, !dbg !12
  br label %305, !dbg !12

384:                                              ; preds = %305
  %385 = add i32 %299, 1, !dbg !12
  br label %298, !dbg !12

386:                                              ; preds = %298
  br label %387, !dbg !19

387:                                              ; preds = %441, %386
  %388 = phi i32 [ %442, %441 ], [ 0, %386 ], !dbg !19
  %389 = icmp slt i32 %388, 4, !dbg !19
  br i1 %389, label %390, label %443, !dbg !19

390:                                              ; preds = %387
  br label %391, !dbg !19

391:                                              ; preds = %439, %390
  %392 = phi i32 [ %440, %439 ], [ 0, %390 ], !dbg !19
  %393 = icmp slt i32 %392, 8, !dbg !19
  br i1 %393, label %394, label %441, !dbg !19

394:                                              ; preds = %391
  %395 = mul nsw i32 %392, 4, !dbg !19
  %396 = icmp slt i32 %392, 0, !dbg !19
  %397 = sub i32 -1, %392, !dbg !19
  %398 = select i1 %396, i32 %397, i32 %392, !dbg !19
  %399 = sdiv i32 %398, 4, !dbg !19
  %400 = sub i32 -1, %399, !dbg !19
  %401 = select i1 %396, i32 %400, i32 %399, !dbg !19
  %402 = mul nsw i32 %401, -16, !dbg !19
  br label %403, !dbg !19

403:                                              ; preds = %437, %394
  %404 = phi i32 [ %438, %437 ], [ 0, %394 ], !dbg !19
  %405 = icmp slt i32 %404, 4, !dbg !19
  br i1 %405, label %406, label %439, !dbg !19

406:                                              ; preds = %403
  %407 = mul nsw i32 %404, 16, !dbg !19
  %408 = add i32 %407, %7, !dbg !19
  %409 = add i32 %408, %19, !dbg !19
  %410 = add i32 %409, %26, !dbg !19
  %411 = add i32 %410, 1, !dbg !20
  %412 = add i32 %408, %26, !dbg !21
  br label %413, !dbg !19

413:                                              ; preds = %416, %406
  %414 = phi i32 [ %436, %416 ], [ 0, %406 ], !dbg !19
  %415 = icmp slt i32 %414, 2, !dbg !19
  br i1 %415, label %416, label %437, !dbg !19

416:                                              ; preds = %413
  %417 = mul nsw i32 %414, 16, !dbg !19
  %418 = add i32 %187, %417, !dbg !19
  %419 = add i32 %418, %395, !dbg !19
  %420 = add i32 %419, %402, !dbg !19
  %421 = add i32 %420, %25, !dbg !19
  %422 = icmp ule i32 %411, %421, !dbg !22
  %423 = select i1 %422, float 0xFFF0000000000000, float 0.000000e+00, !dbg !19
  %424 = mul i32 %388, 8, !dbg !23
  %425 = add i32 %424, %392, !dbg !23
  %426 = getelementptr float, ptr %146, i32 %425, !dbg !23
  %427 = load float, ptr %426, align 4, !dbg !23
  %428 = fadd float %427, %423, !dbg !23
  %429 = call float @__ocml_exp2_f32(float %428), !dbg !21
  %430 = add i32 %417, %395, !dbg !21
  %431 = add i32 %430, %402, !dbg !21
  %432 = add i32 %431, %25, !dbg !21
  %433 = mul i32 %412, 32, !dbg !21
  %434 = add i32 %433, %432, !dbg !21
  %435 = getelementptr float, ptr addrspace(3) @shm_4, i32 %434, !dbg !21
  store float %429, ptr addrspace(3) %435, align 4, !dbg !21
  %436 = add i32 %414, 1, !dbg !19
  br label %413, !dbg !19

437:                                              ; preds = %413
  %438 = add i32 %404, 1, !dbg !19
  br label %403, !dbg !19

439:                                              ; preds = %403
  %440 = add i32 %392, 1, !dbg !19
  br label %391, !dbg !19

441:                                              ; preds = %391
  %442 = add i32 %388, 1, !dbg !19
  br label %387, !dbg !19

443:                                              ; preds = %387
  br label %444, !dbg !13

444:                                              ; preds = %489, %443
  %445 = phi i32 [ %490, %489 ], [ 0, %443 ], !dbg !13
  %446 = icmp slt i32 %445, 4, !dbg !13
  br i1 %446, label %447, label %491, !dbg !13

447:                                              ; preds = %444
  store float 0.000000e+00, ptr %148, align 4, !dbg !13
  %448 = mul nsw i32 %445, 16, !dbg !13
  %449 = add i32 %448, %7, !dbg !13
  %450 = add i32 %449, %26, !dbg !13
  br label %451, !dbg !13

451:                                              ; preds = %454, %447
  %452 = phi i32 [ %463, %454 ], [ 0, %447 ], !dbg !13
  %453 = icmp slt i32 %452, 8, !dbg !13
  br i1 %453, label %454, label %464, !dbg !13

454:                                              ; preds = %451
  %455 = load float, ptr %148, align 4, !dbg !13
  %456 = mul nsw i32 %452, 4, !dbg !13
  %457 = add i32 %456, %25, !dbg !13
  %458 = mul i32 %450, 32, !dbg !13
  %459 = add i32 %458, %457, !dbg !13
  %460 = getelementptr float, ptr addrspace(3) @shm_4, i32 %459, !dbg !13
  %461 = load float, ptr addrspace(3) %460, align 4, !dbg !13
  %462 = fadd float %455, %461, !dbg !13
  store float %462, ptr %148, align 4, !dbg !13
  %463 = add i32 %452, 1, !dbg !13
  br label %451, !dbg !13

464:                                              ; preds = %451
  %465 = load float, ptr %148, align 4, !dbg !13
  %466 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %467 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %466), !dbg !13
  %468 = add i32 %467, 64, !dbg !13
  %469 = xor i32 %467, 16, !dbg !13
  %470 = and i32 %468, -64, !dbg !13
  %471 = icmp slt i32 %469, %470, !dbg !13
  %472 = select i1 %471, i32 %469, i32 %467, !dbg !13
  %473 = shl i32 %472, 2, !dbg !13
  %474 = bitcast float %465 to i32, !dbg !13
  %475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %473, i32 %474), !dbg !13
  %476 = bitcast i32 %475 to float, !dbg !13
  %477 = fadd float %465, %476, !dbg !13
  %478 = xor i32 %467, 32, !dbg !13
  %479 = icmp slt i32 %478, %470, !dbg !13
  %480 = select i1 %479, i32 %478, i32 %467, !dbg !13
  %481 = shl i32 %480, 2, !dbg !13
  %482 = bitcast float %477 to i32, !dbg !13
  %483 = call i32 @llvm.amdgcn.ds.bpermute(i32 %481, i32 %482), !dbg !13
  %484 = bitcast i32 %483 to float, !dbg !13
  %485 = fadd float %477, %484, !dbg !13
  br i1 %170, label %486, label %489, !dbg !13

486:                                              ; preds = %464
  %487 = add i32 %450, 0, !dbg !13
  %488 = getelementptr float, ptr addrspace(3) @shm_5, i32 %487, !dbg !13
  store float %485, ptr addrspace(3) %488, align 4, !dbg !13
  br label %489, !dbg !13

489:                                              ; preds = %486, %464
  %490 = add i32 %445, 1, !dbg !13
  br label %444, !dbg !13

491:                                              ; preds = %444
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  br label %492, !dbg !13

492:                                              ; preds = %511, %491
  %493 = phi i32 [ %512, %511 ], [ 0, %491 ], !dbg !13
  %494 = icmp slt i32 %493, 4, !dbg !13
  br i1 %494, label %495, label %513, !dbg !13

495:                                              ; preds = %492
  br label %496, !dbg !13

496:                                              ; preds = %499, %495
  %497 = phi i32 [ %510, %499 ], [ 0, %495 ], !dbg !13
  %498 = icmp slt i32 %497, 4, !dbg !13
  br i1 %498, label %499, label %511, !dbg !13

499:                                              ; preds = %496
  %500 = mul nsw i32 %497, 16, !dbg !13
  %501 = add i32 %500, %7, !dbg !13
  %502 = add i32 %501, %26, !dbg !13
  %503 = add i32 %502, 0, !dbg !13
  %504 = getelementptr float, ptr addrspace(3) @shm_5, i32 %503, !dbg !13
  %505 = load float, ptr addrspace(3) %504, align 4, !dbg !13
  %506 = add i32 %493, 0, !dbg !13
  %507 = getelementptr float, ptr %135, i32 %506, !dbg !13
  %508 = load float, ptr %507, align 4, !dbg !13
  %509 = fadd float %508, %505, !dbg !13
  store float %509, ptr %507, align 4, !dbg !13
  %510 = add i32 %497, 1, !dbg !13
  br label %496, !dbg !13

511:                                              ; preds = %496
  %512 = add i32 %493, 1, !dbg !13
  br label %492, !dbg !13

513:                                              ; preds = %492
  fence syncscope("workgroup") release, !dbg !14
  call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  br label %514, !dbg !14

514:                                              ; preds = %566, %513
  %515 = phi i32 [ %567, %566 ], [ 0, %513 ], !dbg !14
  %516 = icmp slt i32 %515, 4, !dbg !14
  br i1 %516, label %517, label %568, !dbg !14

517:                                              ; preds = %514
  %518 = mul nsw i32 %515, 16, !dbg !14
  %519 = add i32 %518, %7, !dbg !14
  %520 = add i32 %519, %26, !dbg !14
  br label %521, !dbg !14

521:                                              ; preds = %564, %517
  %522 = phi i32 [ %565, %564 ], [ 0, %517 ], !dbg !14
  %523 = icmp slt i32 %522, 2, !dbg !14
  br i1 %523, label %524, label %566, !dbg !14

524:                                              ; preds = %521
  %525 = mul nsw i32 %522, 16, !dbg !14
  br label %526, !dbg !14

526:                                              ; preds = %529, %524
  %527 = phi i32 [ %538, %529 ], [ 0, %524 ], !dbg !14
  %528 = icmp slt i32 %527, 4, !dbg !14
  br i1 %528, label %529, label %539, !dbg !14

529:                                              ; preds = %526
  %530 = add i32 %527, %525, !dbg !14
  %531 = add i32 %530, %28, !dbg !14
  %532 = mul i32 %520, 32, !dbg !14
  %533 = add i32 %532, %531, !dbg !14
  %534 = getelementptr float, ptr addrspace(3) @shm_4, i32 %533, !dbg !14
  %535 = load float, ptr addrspace(3) %534, align 4, !dbg !14
  %536 = add i32 0, %527, !dbg !14
  %537 = getelementptr float, ptr %149, i32 %536, !dbg !14
  store float %535, ptr %537, align 4, !dbg !14
  %538 = add i32 %527, 1, !dbg !14
  br label %526, !dbg !14

539:                                              ; preds = %526
  br label %540, !dbg !14

540:                                              ; preds = %543, %539
  %541 = phi i32 [ %549, %543 ], [ 0, %539 ], !dbg !14
  %542 = icmp slt i32 %541, 4, !dbg !14
  br i1 %542, label %543, label %550, !dbg !14

543:                                              ; preds = %540
  %544 = add i32 0, %541, !dbg !14
  %545 = getelementptr float, ptr %149, i32 %544, !dbg !14
  %546 = load float, ptr %545, align 4, !dbg !14
  %547 = fptrunc float %546 to half, !dbg !14
  %548 = getelementptr half, ptr %150, i32 %544, !dbg !14
  store half %547, ptr %548, align 2, !dbg !14
  %549 = add i32 %541, 1, !dbg !14
  br label %540, !dbg !14

550:                                              ; preds = %540
  br label %551, !dbg !14

551:                                              ; preds = %554, %550
  %552 = phi i32 [ %563, %554 ], [ 0, %550 ], !dbg !14
  %553 = icmp slt i32 %552, 4, !dbg !14
  br i1 %553, label %554, label %564, !dbg !14

554:                                              ; preds = %551
  %555 = add i32 0, %552, !dbg !14
  %556 = getelementptr half, ptr %150, i32 %555, !dbg !14
  %557 = load half, ptr %556, align 2, !dbg !14
  %558 = add i32 %552, %525, !dbg !14
  %559 = add i32 %558, %28, !dbg !14
  %560 = mul i32 %520, 32, !dbg !14
  %561 = add i32 %560, %559, !dbg !14
  %562 = getelementptr half, ptr addrspace(3) @shm_3, i32 %561, !dbg !14
  store half %557, ptr addrspace(3) %562, align 2, !dbg !14
  %563 = add i32 %552, 1, !dbg !14
  br label %551, !dbg !14

564:                                              ; preds = %551
  %565 = add i32 %522, 1, !dbg !14
  br label %521, !dbg !14

566:                                              ; preds = %521
  %567 = add i32 %515, 1, !dbg !14
  br label %514, !dbg !14

568:                                              ; preds = %514
  br label %569, !dbg !15

569:                                              ; preds = %655, %568
  %570 = phi i32 [ %656, %655 ], [ 0, %568 ], !dbg !15
  %571 = icmp slt i32 %570, 4, !dbg !15
  br i1 %571, label %572, label %657, !dbg !15

572:                                              ; preds = %569
  %573 = mul nsw i32 %570, 16, !dbg !15
  %574 = add i32 %573, %7, !dbg !15
  %575 = add i32 %574, %26, !dbg !15
  br label %576, !dbg !15

576:                                              ; preds = %653, %572
  %577 = phi i32 [ %654, %653 ], [ 0, %572 ], !dbg !15
  %578 = icmp slt i32 %577, 8, !dbg !15
  br i1 %578, label %579, label %655, !dbg !15

579:                                              ; preds = %576
  br label %580, !dbg !15

580:                                              ; preds = %583, %579
  %581 = phi i32 [ %586, %583 ], [ 0, %579 ], !dbg !15
  %582 = icmp slt i32 %581, 4, !dbg !15
  br i1 %582, label %583, label %587, !dbg !15

583:                                              ; preds = %580
  %584 = add i32 0, %581, !dbg !15
  %585 = getelementptr float, ptr %154, i32 %584, !dbg !15
  store float 0.000000e+00, ptr %585, align 4, !dbg !15
  %586 = add i32 %581, 1, !dbg !15
  br label %580, !dbg !15

587:                                              ; preds = %580
  %588 = mul nsw i32 %577, 16, !dbg !15
  %589 = add i32 %588, %7, !dbg !15
  %590 = add i32 %589, %26, !dbg !15
  br label %591, !dbg !15

591:                                              ; preds = %623, %587
  %592 = phi i32 [ %631, %623 ], [ 0, %587 ], !dbg !15
  %593 = icmp slt i32 %592, 2, !dbg !15
  br i1 %593, label %594, label %632, !dbg !15

594:                                              ; preds = %591
  %595 = mul nsw i32 %592, 16, !dbg !15
  br label %596, !dbg !15

596:                                              ; preds = %599, %594
  %597 = phi i32 [ %608, %599 ], [ 0, %594 ], !dbg !15
  %598 = icmp slt i32 %597, 4, !dbg !15
  br i1 %598, label %599, label %609, !dbg !15

599:                                              ; preds = %596
  %600 = add i32 %597, %595, !dbg !15
  %601 = add i32 %600, %28, !dbg !15
  %602 = mul i32 %575, 32, !dbg !15
  %603 = add i32 %602, %601, !dbg !15
  %604 = getelementptr half, ptr addrspace(3) @shm_3, i32 %603, !dbg !15
  %605 = load half, ptr addrspace(3) %604, align 2, !dbg !15
  %606 = add i32 0, %597, !dbg !15
  %607 = getelementptr half, ptr %151, i32 %606, !dbg !15
  store half %605, ptr %607, align 2, !dbg !15
  %608 = add i32 %597, 1, !dbg !15
  br label %596, !dbg !15

609:                                              ; preds = %596
  br label %610, !dbg !15

610:                                              ; preds = %613, %609
  %611 = phi i32 [ %622, %613 ], [ 0, %609 ], !dbg !15
  %612 = icmp slt i32 %611, 4, !dbg !15
  br i1 %612, label %613, label %623, !dbg !15

613:                                              ; preds = %610
  %614 = add i32 %611, %595, !dbg !15
  %615 = add i32 %614, %28, !dbg !15
  %616 = mul i32 %615, 128, !dbg !15
  %617 = add i32 %616, %590, !dbg !15
  %618 = getelementptr half, ptr addrspace(3) @shm_2, i32 %617, !dbg !15
  %619 = load half, ptr addrspace(3) %618, align 2, !dbg !15
  %620 = add i32 %611, 0, !dbg !15
  %621 = getelementptr half, ptr %152, i32 %620, !dbg !15
  store half %619, ptr %621, align 2, !dbg !15
  %622 = add i32 %611, 1, !dbg !15
  br label %610, !dbg !15

623:                                              ; preds = %610
  %624 = getelementptr half, ptr %151, i32 0, !dbg !15
  %625 = load <4 x half>, ptr %624, align 2, !dbg !15
  %626 = getelementptr half, ptr %152, i32 0, !dbg !15
  %627 = load <4 x half>, ptr %626, align 2, !dbg !15
  %628 = getelementptr float, ptr %154, i32 0, !dbg !15
  %629 = load <4 x float>, ptr %628, align 4, !dbg !15
  %630 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %625, <4 x half> %627, <4 x float> %629), !dbg !15
  store <4 x float> %630, ptr %628, align 4, !dbg !15
  %631 = add i32 %592, 1, !dbg !15
  br label %591, !dbg !15

632:                                              ; preds = %591
  %633 = mul nsw i32 %577, 4, !dbg !15
  br label %634, !dbg !15

634:                                              ; preds = %651, %632
  %635 = phi i32 [ %652, %651 ], [ 0, %632 ], !dbg !15
  %636 = icmp slt i32 %635, 4, !dbg !15
  br i1 %636, label %637, label %653, !dbg !15

637:                                              ; preds = %634
  br label %638, !dbg !15

638:                                              ; preds = %641, %637
  %639 = phi i32 [ %650, %641 ], [ 0, %637 ], !dbg !15
  %640 = icmp slt i32 %639, 4, !dbg !15
  br i1 %640, label %641, label %651, !dbg !15

641:                                              ; preds = %638
  %642 = add i32 0, %639, !dbg !15
  %643 = getelementptr float, ptr %154, i32 %642, !dbg !15
  %644 = load float, ptr %643, align 4, !dbg !15
  %645 = add i32 %639, %633, !dbg !15
  %646 = add i32 %645, %635, !dbg !15
  %647 = mul i32 %570, 32, !dbg !15
  %648 = add i32 %647, %646, !dbg !15
  %649 = getelementptr float, ptr %153, i32 %648, !dbg !15
  store float %644, ptr %649, align 4, !dbg !15
  %650 = add i32 %639, 1, !dbg !15
  br label %638, !dbg !15

651:                                              ; preds = %638
  %652 = add i32 %635, 1, !dbg !15
  br label %634, !dbg !15

653:                                              ; preds = %634
  %654 = add i32 %577, 1, !dbg !15
  br label %576, !dbg !15

655:                                              ; preds = %576
  %656 = add i32 %570, 1, !dbg !15
  br label %569, !dbg !15

657:                                              ; preds = %569
  br label %658, !dbg !24

658:                                              ; preds = %686, %657
  %659 = phi i32 [ %687, %686 ], [ 0, %657 ], !dbg !24
  %660 = icmp slt i32 %659, 4, !dbg !24
  br i1 %660, label %661, label %688, !dbg !24

661:                                              ; preds = %658
  br label %662, !dbg !24

662:                                              ; preds = %684, %661
  %663 = phi i32 [ %685, %684 ], [ 0, %661 ], !dbg !24
  %664 = icmp slt i32 %663, 32, !dbg !24
  br i1 %664, label %665, label %686, !dbg !24

665:                                              ; preds = %662
  br label %666, !dbg !24

666:                                              ; preds = %682, %665
  %667 = phi i32 [ %683, %682 ], [ 0, %665 ], !dbg !24
  %668 = icmp slt i32 %667, 4, !dbg !24
  br i1 %668, label %669, label %684, !dbg !24

669:                                              ; preds = %666
  br label %670, !dbg !24

670:                                              ; preds = %673, %669
  %671 = phi i32 [ %681, %673 ], [ 0, %669 ], !dbg !24
  %672 = icmp slt i32 %671, 8, !dbg !24
  br i1 %672, label %673, label %682, !dbg !24

673:                                              ; preds = %670
  %674 = mul i32 %659, 32, !dbg !24
  %675 = add i32 %674, %663, !dbg !24
  %676 = getelementptr float, ptr %119, i32 %675, !dbg !24
  %677 = load float, ptr %676, align 4, !dbg !24
  %678 = getelementptr float, ptr %153, i32 %675, !dbg !24
  %679 = load float, ptr %678, align 4, !dbg !24
  %680 = fadd float %677, %679, !dbg !24
  store float %680, ptr %676, align 4, !dbg !24
  %681 = add i32 %671, 1, !dbg !24
  br label %670, !dbg !24

682:                                              ; preds = %670
  %683 = add i32 %667, 1, !dbg !24
  br label %666, !dbg !24

684:                                              ; preds = %666
  %685 = add i32 %663, 1, !dbg !24
  br label %662, !dbg !24

686:                                              ; preds = %662
  %687 = add i32 %659, 1, !dbg !24
  br label %658, !dbg !24

688:                                              ; preds = %658
  %689 = add i32 %172, 1, !dbg !16
  br label %171, !dbg !16

690:                                              ; preds = %171
  %691 = alloca half, i32 128, align 2, !dbg !25
  br label %692, !dbg !26

692:                                              ; preds = %723, %690
  %693 = phi i32 [ %724, %723 ], [ 0, %690 ], !dbg !26
  %694 = icmp slt i32 %693, 4, !dbg !26
  br i1 %694, label %695, label %725, !dbg !26

695:                                              ; preds = %692
  br label %696, !dbg !26

696:                                              ; preds = %721, %695
  %697 = phi i32 [ %722, %721 ], [ 0, %695 ], !dbg !26
  %698 = icmp slt i32 %697, 32, !dbg !26
  br i1 %698, label %699, label %723, !dbg !26

699:                                              ; preds = %696
  br label %700, !dbg !26

700:                                              ; preds = %719, %699
  %701 = phi i32 [ %720, %719 ], [ 0, %699 ], !dbg !26
  %702 = icmp slt i32 %701, 4, !dbg !26
  br i1 %702, label %703, label %721, !dbg !26

703:                                              ; preds = %700
  br label %704, !dbg !26

704:                                              ; preds = %707, %703
  %705 = phi i32 [ %718, %707 ], [ 0, %703 ], !dbg !26
  %706 = icmp slt i32 %705, 8, !dbg !26
  br i1 %706, label %707, label %719, !dbg !26

707:                                              ; preds = %704
  %708 = mul i32 %693, 32, !dbg !26
  %709 = add i32 %708, %697, !dbg !26
  %710 = getelementptr float, ptr %119, i32 %709, !dbg !26
  %711 = load float, ptr %710, align 4, !dbg !26
  %712 = add i32 %693, 0, !dbg !26
  %713 = getelementptr float, ptr %135, i32 %712, !dbg !26
  %714 = load float, ptr %713, align 4, !dbg !26
  %715 = fdiv float %711, %714, !dbg !26
  %716 = fptrunc float %715 to half, !dbg !7
  %717 = getelementptr half, ptr %691, i32 %709, !dbg !7
  store half %716, ptr %717, align 2, !dbg !7
  %718 = add i32 %705, 1, !dbg !26
  br label %704, !dbg !26

719:                                              ; preds = %704
  %720 = add i32 %701, 1, !dbg !26
  br label %700, !dbg !26

721:                                              ; preds = %700
  %722 = add i32 %697, 1, !dbg !26
  br label %696, !dbg !26

723:                                              ; preds = %696
  %724 = add i32 %693, 1, !dbg !26
  br label %692, !dbg !26

725:                                              ; preds = %692
  br label %726, !dbg !27

726:                                              ; preds = %747, %725
  %727 = phi i32 [ %748, %747 ], [ 0, %725 ], !dbg !27
  %728 = icmp slt i32 %727, 4, !dbg !27
  br i1 %728, label %729, label %749, !dbg !27

729:                                              ; preds = %726
  %730 = add i32 %727, %19, !dbg !27
  %731 = add i32 %730, %27, !dbg !27
  br label %732, !dbg !27

732:                                              ; preds = %735, %729
  %733 = phi i32 [ %746, %735 ], [ 0, %729 ], !dbg !27
  %734 = icmp slt i32 %733, 32, !dbg !27
  br i1 %734, label %735, label %747, !dbg !27

735:                                              ; preds = %732
  %736 = mul i32 %727, 32, !dbg !27
  %737 = add i32 %736, %733, !dbg !27
  %738 = getelementptr half, ptr %691, i32 %737, !dbg !27
  %739 = load half, ptr %738, align 2, !dbg !27
  %740 = mul i32 %18, 524288, !dbg !27
  %741 = add i32 0, %740, !dbg !27
  %742 = mul i32 %731, 128, !dbg !27
  %743 = add i32 %741, %742, !dbg !27
  %744 = add i32 %743, %733, !dbg !27
  %745 = getelementptr half, ptr addrspace(1) %3, i32 %744, !dbg !27
  store half %739, ptr addrspace(1) %745, align 2, !dbg !27
  %746 = add i32 %733, 1, !dbg !27
  br label %732, !dbg !27

747:                                              ; preds = %732
  %748 = add i32 %727, 1, !dbg !27
  br label %726, !dbg !27

749:                                              ; preds = %726
  ret void, !dbg !7
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

attributes #0 = { "amdgpu-flat-work-group-size"="64,64" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nocallback nofree nounwind willreturn memory(none) }

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
!8 = !DILocation(line: 48, column: 13, scope: !3)
!9 = !DILocation(line: 51, column: 13, scope: !3)
!10 = !DILocation(line: 52, column: 13, scope: !3)
!11 = !DILocation(line: 53, column: 13, scope: !3)
!12 = !DILocation(line: 62, column: 15, scope: !3)
!13 = !DILocation(line: 76, column: 15, scope: !3)
!14 = !DILocation(line: 77, column: 15, scope: !3)
!15 = !DILocation(line: 78, column: 15, scope: !3)
!16 = !DILocation(line: 59, column: 15, scope: !3)
!17 = !DILocation(line: 61, column: 15, scope: !3)
!18 = !DILocation(line: 60, column: 15, scope: !3)
!19 = !DILocation(line: 63, column: 15, scope: !3)
!20 = !DILocation(line: 65, column: 17, scope: !3)
!21 = !DILocation(line: 75, column: 15, scope: !3)
!22 = !DILocation(line: 66, column: 17, scope: !3)
!23 = !DILocation(line: 74, column: 15, scope: !3)
!24 = !DILocation(line: 79, column: 15, scope: !3)
!25 = !DILocation(line: 85, column: 13, scope: !3)
!26 = !DILocation(line: 84, column: 13, scope: !3)
!27 = !DILocation(line: 86, column: 7, scope: !3)
