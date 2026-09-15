; ModuleID = 'test/test_friskBase.mlir'
source_filename = "test/test_friskBase.mlir"

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
  %12 = mul nsw i64 %10, -64, !dbg !11
  br label %13, !dbg !11

13:                                               ; preds = %225, %4
  %14 = phi i64 [ %226, %225 ], [ 0, %4 ], !dbg !11
  %15 = icmp slt i64 %14, 16, !dbg !11
  br i1 %15, label %16, label %227, !dbg !11

16:                                               ; preds = %13
  %17 = mul nsw i64 %14, -4, !dbg !11
  %18 = add i64 %17, %12, !dbg !11
  %19 = add i64 %18, 8191, !dbg !11
  %20 = icmp sge i64 %19, 0, !dbg !11
  br i1 %20, label %21, label %69, !dbg !11

21:                                               ; preds = %16
  %22 = srem i64 %8, 32, !dbg !11
  %23 = icmp slt i64 %22, 0, !dbg !11
  %24 = add i64 %22, 32, !dbg !11
  %25 = select i1 %23, i64 %24, i64 %22, !dbg !11
  %26 = mul nsw i64 %6, 64, !dbg !11
  %27 = icmp slt i64 %10, 0, !dbg !11
  %28 = sub i64 -1, %10, !dbg !11
  %29 = select i1 %27, i64 %28, i64 %10, !dbg !11
  %30 = sdiv i64 %29, 2, !dbg !11
  %31 = sub i64 -1, %30, !dbg !11
  %32 = select i1 %27, i64 %31, i64 %30, !dbg !11
  %33 = add i64 %26, %32, !dbg !11
  %34 = icmp slt i64 %32, 0, !dbg !11
  %35 = sub i64 -1, %32, !dbg !11
  %36 = select i1 %34, i64 %35, i64 %32, !dbg !11
  %37 = sdiv i64 %36, 64, !dbg !11
  %38 = sub i64 -1, %37, !dbg !11
  %39 = select i1 %34, i64 %38, i64 %37, !dbg !11
  %40 = mul nsw i64 %39, -64, !dbg !11
  %41 = add i64 %33, %40, !dbg !11
  %42 = icmp slt i64 %6, 0, !dbg !11
  %43 = sub i64 -1, %6, !dbg !11
  %44 = select i1 %42, i64 %43, i64 %6, !dbg !11
  %45 = sdiv i64 %44, 64, !dbg !11
  %46 = sub i64 -1, %45, !dbg !11
  %47 = select i1 %42, i64 %46, i64 %45, !dbg !11
  %48 = mul nsw i64 %47, -4096, !dbg !11
  %49 = add i64 %41, %48, !dbg !11
  %50 = mul nsw i64 %14, 4, !dbg !11
  %51 = mul nsw i64 %10, 64, !dbg !11
  %52 = add i64 %50, %51, !dbg !11
  %53 = mul nsw i64 %32, -128, !dbg !11
  %54 = add i64 %52, %53, !dbg !11
  %55 = mul i64 %25, 524288, !dbg !11
  %56 = add i64 0, %55, !dbg !11
  %57 = mul i64 %49, 128, !dbg !11
  %58 = add i64 %56, %57, !dbg !11
  %59 = add i64 %58, %54, !dbg !11
  %60 = getelementptr half, ptr addrspace(1) %0, i64 %59, !dbg !11
  %61 = load half, ptr addrspace(1) %60, align 2, !dbg !11
  %62 = srem i64 %32, 64, !dbg !11
  %63 = icmp slt i64 %62, 0, !dbg !11
  %64 = add i64 %62, 64, !dbg !11
  %65 = select i1 %63, i64 %64, i64 %62, !dbg !11
  %66 = mul i64 %65, 128, !dbg !11
  %67 = add i64 %66, %54, !dbg !11
  %68 = getelementptr half, ptr addrspace(3) @shm_0, i64 %67, !dbg !11
  store half %61, ptr addrspace(3) %68, align 2, !dbg !11
  br label %69, !dbg !11

69:                                               ; preds = %21, %16
  %70 = add i64 %18, 8190, !dbg !11
  %71 = icmp sge i64 %70, 0, !dbg !11
  br i1 %71, label %72, label %121, !dbg !11

72:                                               ; preds = %69
  %73 = srem i64 %8, 32, !dbg !11
  %74 = icmp slt i64 %73, 0, !dbg !11
  %75 = add i64 %73, 32, !dbg !11
  %76 = select i1 %74, i64 %75, i64 %73, !dbg !11
  %77 = mul nsw i64 %6, 64, !dbg !11
  %78 = icmp slt i64 %10, 0, !dbg !11
  %79 = sub i64 -1, %10, !dbg !11
  %80 = select i1 %78, i64 %79, i64 %10, !dbg !11
  %81 = sdiv i64 %80, 2, !dbg !11
  %82 = sub i64 -1, %81, !dbg !11
  %83 = select i1 %78, i64 %82, i64 %81, !dbg !11
  %84 = add i64 %77, %83, !dbg !11
  %85 = icmp slt i64 %83, 0, !dbg !11
  %86 = sub i64 -1, %83, !dbg !11
  %87 = select i1 %85, i64 %86, i64 %83, !dbg !11
  %88 = sdiv i64 %87, 64, !dbg !11
  %89 = sub i64 -1, %88, !dbg !11
  %90 = select i1 %85, i64 %89, i64 %88, !dbg !11
  %91 = mul nsw i64 %90, -64, !dbg !11
  %92 = add i64 %84, %91, !dbg !11
  %93 = icmp slt i64 %6, 0, !dbg !11
  %94 = sub i64 -1, %6, !dbg !11
  %95 = select i1 %93, i64 %94, i64 %6, !dbg !11
  %96 = sdiv i64 %95, 64, !dbg !11
  %97 = sub i64 -1, %96, !dbg !11
  %98 = select i1 %93, i64 %97, i64 %96, !dbg !11
  %99 = mul nsw i64 %98, -4096, !dbg !11
  %100 = add i64 %92, %99, !dbg !11
  %101 = mul nsw i64 %14, 4, !dbg !11
  %102 = mul nsw i64 %10, 64, !dbg !11
  %103 = add i64 %101, %102, !dbg !11
  %104 = mul nsw i64 %83, -128, !dbg !11
  %105 = add i64 %103, %104, !dbg !11
  %106 = add i64 %105, 1, !dbg !11
  %107 = mul i64 %76, 524288, !dbg !11
  %108 = add i64 0, %107, !dbg !11
  %109 = mul i64 %100, 128, !dbg !11
  %110 = add i64 %108, %109, !dbg !11
  %111 = add i64 %110, %106, !dbg !11
  %112 = getelementptr half, ptr addrspace(1) %0, i64 %111, !dbg !11
  %113 = load half, ptr addrspace(1) %112, align 2, !dbg !11
  %114 = srem i64 %83, 64, !dbg !11
  %115 = icmp slt i64 %114, 0, !dbg !11
  %116 = add i64 %114, 64, !dbg !11
  %117 = select i1 %115, i64 %116, i64 %114, !dbg !11
  %118 = mul i64 %117, 128, !dbg !11
  %119 = add i64 %118, %106, !dbg !11
  %120 = getelementptr half, ptr addrspace(3) @shm_0, i64 %119, !dbg !11
  store half %113, ptr addrspace(3) %120, align 2, !dbg !11
  br label %121, !dbg !11

121:                                              ; preds = %72, %69
  %122 = add i64 %18, 8189, !dbg !11
  %123 = icmp sge i64 %122, 0, !dbg !11
  br i1 %123, label %124, label %173, !dbg !11

124:                                              ; preds = %121
  %125 = srem i64 %8, 32, !dbg !11
  %126 = icmp slt i64 %125, 0, !dbg !11
  %127 = add i64 %125, 32, !dbg !11
  %128 = select i1 %126, i64 %127, i64 %125, !dbg !11
  %129 = mul nsw i64 %6, 64, !dbg !11
  %130 = icmp slt i64 %10, 0, !dbg !11
  %131 = sub i64 -1, %10, !dbg !11
  %132 = select i1 %130, i64 %131, i64 %10, !dbg !11
  %133 = sdiv i64 %132, 2, !dbg !11
  %134 = sub i64 -1, %133, !dbg !11
  %135 = select i1 %130, i64 %134, i64 %133, !dbg !11
  %136 = add i64 %129, %135, !dbg !11
  %137 = icmp slt i64 %135, 0, !dbg !11
  %138 = sub i64 -1, %135, !dbg !11
  %139 = select i1 %137, i64 %138, i64 %135, !dbg !11
  %140 = sdiv i64 %139, 64, !dbg !11
  %141 = sub i64 -1, %140, !dbg !11
  %142 = select i1 %137, i64 %141, i64 %140, !dbg !11
  %143 = mul nsw i64 %142, -64, !dbg !11
  %144 = add i64 %136, %143, !dbg !11
  %145 = icmp slt i64 %6, 0, !dbg !11
  %146 = sub i64 -1, %6, !dbg !11
  %147 = select i1 %145, i64 %146, i64 %6, !dbg !11
  %148 = sdiv i64 %147, 64, !dbg !11
  %149 = sub i64 -1, %148, !dbg !11
  %150 = select i1 %145, i64 %149, i64 %148, !dbg !11
  %151 = mul nsw i64 %150, -4096, !dbg !11
  %152 = add i64 %144, %151, !dbg !11
  %153 = mul nsw i64 %14, 4, !dbg !11
  %154 = mul nsw i64 %10, 64, !dbg !11
  %155 = add i64 %153, %154, !dbg !11
  %156 = mul nsw i64 %135, -128, !dbg !11
  %157 = add i64 %155, %156, !dbg !11
  %158 = add i64 %157, 2, !dbg !11
  %159 = mul i64 %128, 524288, !dbg !11
  %160 = add i64 0, %159, !dbg !11
  %161 = mul i64 %152, 128, !dbg !11
  %162 = add i64 %160, %161, !dbg !11
  %163 = add i64 %162, %158, !dbg !11
  %164 = getelementptr half, ptr addrspace(1) %0, i64 %163, !dbg !11
  %165 = load half, ptr addrspace(1) %164, align 2, !dbg !11
  %166 = srem i64 %135, 64, !dbg !11
  %167 = icmp slt i64 %166, 0, !dbg !11
  %168 = add i64 %166, 64, !dbg !11
  %169 = select i1 %167, i64 %168, i64 %166, !dbg !11
  %170 = mul i64 %169, 128, !dbg !11
  %171 = add i64 %170, %158, !dbg !11
  %172 = getelementptr half, ptr addrspace(3) @shm_0, i64 %171, !dbg !11
  store half %165, ptr addrspace(3) %172, align 2, !dbg !11
  br label %173, !dbg !11

173:                                              ; preds = %124, %121
  %174 = add i64 %18, 8188, !dbg !11
  %175 = icmp sge i64 %174, 0, !dbg !11
  br i1 %175, label %176, label %225, !dbg !11

176:                                              ; preds = %173
  %177 = srem i64 %8, 32, !dbg !11
  %178 = icmp slt i64 %177, 0, !dbg !11
  %179 = add i64 %177, 32, !dbg !11
  %180 = select i1 %178, i64 %179, i64 %177, !dbg !11
  %181 = mul nsw i64 %6, 64, !dbg !11
  %182 = icmp slt i64 %10, 0, !dbg !11
  %183 = sub i64 -1, %10, !dbg !11
  %184 = select i1 %182, i64 %183, i64 %10, !dbg !11
  %185 = sdiv i64 %184, 2, !dbg !11
  %186 = sub i64 -1, %185, !dbg !11
  %187 = select i1 %182, i64 %186, i64 %185, !dbg !11
  %188 = add i64 %181, %187, !dbg !11
  %189 = icmp slt i64 %187, 0, !dbg !11
  %190 = sub i64 -1, %187, !dbg !11
  %191 = select i1 %189, i64 %190, i64 %187, !dbg !11
  %192 = sdiv i64 %191, 64, !dbg !11
  %193 = sub i64 -1, %192, !dbg !11
  %194 = select i1 %189, i64 %193, i64 %192, !dbg !11
  %195 = mul nsw i64 %194, -64, !dbg !11
  %196 = add i64 %188, %195, !dbg !11
  %197 = icmp slt i64 %6, 0, !dbg !11
  %198 = sub i64 -1, %6, !dbg !11
  %199 = select i1 %197, i64 %198, i64 %6, !dbg !11
  %200 = sdiv i64 %199, 64, !dbg !11
  %201 = sub i64 -1, %200, !dbg !11
  %202 = select i1 %197, i64 %201, i64 %200, !dbg !11
  %203 = mul nsw i64 %202, -4096, !dbg !11
  %204 = add i64 %196, %203, !dbg !11
  %205 = mul nsw i64 %14, 4, !dbg !11
  %206 = mul nsw i64 %10, 64, !dbg !11
  %207 = add i64 %205, %206, !dbg !11
  %208 = mul nsw i64 %187, -128, !dbg !11
  %209 = add i64 %207, %208, !dbg !11
  %210 = add i64 %209, 3, !dbg !11
  %211 = mul i64 %180, 524288, !dbg !11
  %212 = add i64 0, %211, !dbg !11
  %213 = mul i64 %204, 128, !dbg !11
  %214 = add i64 %212, %213, !dbg !11
  %215 = add i64 %214, %210, !dbg !11
  %216 = getelementptr half, ptr addrspace(1) %0, i64 %215, !dbg !11
  %217 = load half, ptr addrspace(1) %216, align 2, !dbg !11
  %218 = srem i64 %187, 64, !dbg !11
  %219 = icmp slt i64 %218, 0, !dbg !11
  %220 = add i64 %218, 64, !dbg !11
  %221 = select i1 %219, i64 %220, i64 %218, !dbg !11
  %222 = mul i64 %221, 128, !dbg !11
  %223 = add i64 %222, %210, !dbg !11
  %224 = getelementptr half, ptr addrspace(3) @shm_0, i64 %223, !dbg !11
  store half %217, ptr addrspace(3) %224, align 2, !dbg !11
  br label %225, !dbg !11

225:                                              ; preds = %176, %173
  %226 = add i64 %14, 1, !dbg !11
  br label %13, !dbg !11

227:                                              ; preds = %13
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %228 = srem i64 %10, 16, !dbg !12
  %229 = icmp slt i64 %228, 0, !dbg !12
  %230 = add i64 %228, 16, !dbg !12
  %231 = select i1 %229, i64 %230, i64 %228, !dbg !12
  %232 = icmp slt i64 %10, 0, !dbg !12
  %233 = sub i64 -1, %10, !dbg !12
  %234 = select i1 %232, i64 %233, i64 %10, !dbg !12
  %235 = sdiv i64 %234, 16, !dbg !12
  %236 = sub i64 -1, %235, !dbg !12
  %237 = select i1 %232, i64 %236, i64 %235, !dbg !12
  %238 = mul nsw i64 %237, 4, !dbg !12
  %239 = mul i64 %231, 128, !dbg !12
  %240 = add i64 %239, %238, !dbg !12
  %241 = getelementptr half, ptr addrspace(3) @shm_0, i64 %240, !dbg !12
  %242 = load half, ptr addrspace(3) %241, align 2, !dbg !12
  %243 = insertelement <4 x half> zeroinitializer, half %242, i64 0, !dbg !12
  %244 = add i64 %238, 1, !dbg !12
  %245 = add i64 %239, %244, !dbg !12
  %246 = getelementptr half, ptr addrspace(3) @shm_0, i64 %245, !dbg !12
  %247 = load half, ptr addrspace(3) %246, align 2, !dbg !12
  %248 = insertelement <4 x half> %243, half %247, i64 1, !dbg !12
  %249 = add i64 %238, 2, !dbg !12
  %250 = add i64 %239, %249, !dbg !12
  %251 = getelementptr half, ptr addrspace(3) @shm_0, i64 %250, !dbg !12
  %252 = load half, ptr addrspace(3) %251, align 2, !dbg !12
  %253 = insertelement <4 x half> %248, half %252, i64 2, !dbg !12
  %254 = add i64 %238, 3, !dbg !12
  %255 = add i64 %239, %254, !dbg !12
  %256 = getelementptr half, ptr addrspace(3) @shm_0, i64 %255, !dbg !12
  %257 = load half, ptr addrspace(3) %256, align 2, !dbg !12
  %258 = insertelement <4 x half> %253, half %257, i64 3, !dbg !12
  %compat.splat.insert = insertelement <4 x half> poison, half 0xH3015, i32 0, !dbg !12
  %compat.splat = shufflevector <4 x half> %compat.splat.insert, <4 x half> poison, <4 x i32> zeroinitializer, !dbg !12
  %259 = fmul <4 x half> %258, %compat.splat, !dbg !12
  %260 = srem i64 %10, 64, !dbg !13
  %261 = icmp slt i64 %260, 0, !dbg !13
  %262 = add i64 %260, 64, !dbg !13
  %263 = select i1 %261, i64 %262, i64 %260, !dbg !13
  %264 = icmp slt i64 %263, 0, !dbg !13
  %265 = sub i64 -1, %263, !dbg !13
  %266 = select i1 %264, i64 %265, i64 %263, !dbg !13
  %267 = sdiv i64 %266, 16, !dbg !13
  %268 = sub i64 -1, %267, !dbg !13
  %269 = select i1 %264, i64 %268, i64 %267, !dbg !13
  %270 = icmp eq i64 %269, 0, !dbg !13
  %271 = alloca float, i64 1, align 4, addrspace(5), !dbg !13
  %272 = alloca float, i64 4, align 1, addrspace(5), !dbg !14
  %273 = alloca half, i64 4, align 1, addrspace(5), !dbg !14
  %274 = mul nsw i64 %6, 2, !dbg !15
  %275 = add i64 %274, 1, !dbg !15
  %276 = mul nsw i64 %10, -32, !dbg !16
  %277 = mul nsw i64 %237, -16, !dbg !17
  br label %278, !dbg !15

278:                                              ; preds = %1084, %227
  %279 = phi i64 [ %1085, %1084 ], [ 0, %227 ], !dbg !18
  %280 = icmp slt i64 %279, %275, !dbg !15
  br i1 %280, label %281, label %1086, !dbg !15

281:                                              ; preds = %278
  br label %282, !dbg !16

282:                                              ; preds = %425, %281
  %283 = phi i64 [ %426, %425 ], [ 0, %281 ], !dbg !16
  %284 = icmp slt i64 %283, 8, !dbg !16
  br i1 %284, label %285, label %427, !dbg !16

285:                                              ; preds = %282
  %286 = mul nsw i64 %283, -4, !dbg !16
  %287 = add i64 %286, %276, !dbg !16
  %288 = add i64 %287, 4095, !dbg !16
  %289 = icmp sge i64 %288, 0, !dbg !16
  br i1 %289, label %290, label %320, !dbg !16

290:                                              ; preds = %285
  %291 = srem i64 %8, 32, !dbg !16
  %292 = icmp slt i64 %291, 0, !dbg !16
  %293 = add i64 %291, 32, !dbg !16
  %294 = select i1 %292, i64 %293, i64 %291, !dbg !16
  %295 = srem i64 %10, 128, !dbg !16
  %296 = icmp slt i64 %295, 0, !dbg !16
  %297 = add i64 %295, 128, !dbg !16
  %298 = select i1 %296, i64 %297, i64 %295, !dbg !16
  %299 = mul nsw i64 %279, 32, !dbg !16
  %300 = mul nsw i64 %283, 4, !dbg !16
  %301 = add i64 %299, %300, !dbg !16
  %302 = icmp slt i64 %279, 0, !dbg !16
  %303 = sub i64 -1, %279, !dbg !16
  %304 = select i1 %302, i64 %303, i64 %279, !dbg !16
  %305 = sdiv i64 %304, 128, !dbg !16
  %306 = sub i64 -1, %305, !dbg !16
  %307 = select i1 %302, i64 %306, i64 %305, !dbg !16
  %308 = mul nsw i64 %307, -4096, !dbg !16
  %309 = add i64 %301, %308, !dbg !16
  %310 = mul i64 %294, 524288, !dbg !16
  %311 = add i64 0, %310, !dbg !16
  %312 = mul i64 %298, 4096, !dbg !16
  %313 = add i64 %311, %312, !dbg !16
  %314 = add i64 %313, %309, !dbg !16
  %315 = getelementptr half, ptr addrspace(1) %2, i64 %314, !dbg !16
  %316 = load half, ptr addrspace(1) %315, align 2, !dbg !16
  %317 = mul i64 %298, 32, !dbg !16
  %318 = add i64 %317, %300, !dbg !16
  %319 = getelementptr half, ptr addrspace(3) @shm_1, i64 %318, !dbg !16
  store half %316, ptr addrspace(3) %319, align 2, !dbg !16
  br label %320, !dbg !16

320:                                              ; preds = %290, %285
  %321 = add i64 %287, 4094, !dbg !16
  %322 = icmp sge i64 %321, 0, !dbg !16
  br i1 %322, label %323, label %355, !dbg !16

323:                                              ; preds = %320
  %324 = srem i64 %8, 32, !dbg !16
  %325 = icmp slt i64 %324, 0, !dbg !16
  %326 = add i64 %324, 32, !dbg !16
  %327 = select i1 %325, i64 %326, i64 %324, !dbg !16
  %328 = srem i64 %10, 128, !dbg !16
  %329 = icmp slt i64 %328, 0, !dbg !16
  %330 = add i64 %328, 128, !dbg !16
  %331 = select i1 %329, i64 %330, i64 %328, !dbg !16
  %332 = mul nsw i64 %279, 32, !dbg !16
  %333 = mul nsw i64 %283, 4, !dbg !16
  %334 = add i64 %332, %333, !dbg !16
  %335 = icmp slt i64 %279, 0, !dbg !16
  %336 = sub i64 -1, %279, !dbg !16
  %337 = select i1 %335, i64 %336, i64 %279, !dbg !16
  %338 = sdiv i64 %337, 128, !dbg !16
  %339 = sub i64 -1, %338, !dbg !16
  %340 = select i1 %335, i64 %339, i64 %338, !dbg !16
  %341 = mul nsw i64 %340, -4096, !dbg !16
  %342 = add i64 %334, %341, !dbg !16
  %343 = add i64 %342, 1, !dbg !16
  %344 = mul i64 %327, 524288, !dbg !16
  %345 = add i64 0, %344, !dbg !16
  %346 = mul i64 %331, 4096, !dbg !16
  %347 = add i64 %345, %346, !dbg !16
  %348 = add i64 %347, %343, !dbg !16
  %349 = getelementptr half, ptr addrspace(1) %2, i64 %348, !dbg !16
  %350 = load half, ptr addrspace(1) %349, align 2, !dbg !16
  %351 = add i64 %333, 1, !dbg !16
  %352 = mul i64 %331, 32, !dbg !16
  %353 = add i64 %352, %351, !dbg !16
  %354 = getelementptr half, ptr addrspace(3) @shm_1, i64 %353, !dbg !16
  store half %350, ptr addrspace(3) %354, align 2, !dbg !16
  br label %355, !dbg !16

355:                                              ; preds = %323, %320
  %356 = add i64 %287, 4093, !dbg !16
  %357 = icmp sge i64 %356, 0, !dbg !16
  br i1 %357, label %358, label %390, !dbg !16

358:                                              ; preds = %355
  %359 = srem i64 %8, 32, !dbg !16
  %360 = icmp slt i64 %359, 0, !dbg !16
  %361 = add i64 %359, 32, !dbg !16
  %362 = select i1 %360, i64 %361, i64 %359, !dbg !16
  %363 = srem i64 %10, 128, !dbg !16
  %364 = icmp slt i64 %363, 0, !dbg !16
  %365 = add i64 %363, 128, !dbg !16
  %366 = select i1 %364, i64 %365, i64 %363, !dbg !16
  %367 = mul nsw i64 %279, 32, !dbg !16
  %368 = mul nsw i64 %283, 4, !dbg !16
  %369 = add i64 %367, %368, !dbg !16
  %370 = icmp slt i64 %279, 0, !dbg !16
  %371 = sub i64 -1, %279, !dbg !16
  %372 = select i1 %370, i64 %371, i64 %279, !dbg !16
  %373 = sdiv i64 %372, 128, !dbg !16
  %374 = sub i64 -1, %373, !dbg !16
  %375 = select i1 %370, i64 %374, i64 %373, !dbg !16
  %376 = mul nsw i64 %375, -4096, !dbg !16
  %377 = add i64 %369, %376, !dbg !16
  %378 = add i64 %377, 2, !dbg !16
  %379 = mul i64 %362, 524288, !dbg !16
  %380 = add i64 0, %379, !dbg !16
  %381 = mul i64 %366, 4096, !dbg !16
  %382 = add i64 %380, %381, !dbg !16
  %383 = add i64 %382, %378, !dbg !16
  %384 = getelementptr half, ptr addrspace(1) %2, i64 %383, !dbg !16
  %385 = load half, ptr addrspace(1) %384, align 2, !dbg !16
  %386 = add i64 %368, 2, !dbg !16
  %387 = mul i64 %366, 32, !dbg !16
  %388 = add i64 %387, %386, !dbg !16
  %389 = getelementptr half, ptr addrspace(3) @shm_1, i64 %388, !dbg !16
  store half %385, ptr addrspace(3) %389, align 2, !dbg !16
  br label %390, !dbg !16

390:                                              ; preds = %358, %355
  %391 = add i64 %287, 4092, !dbg !16
  %392 = icmp sge i64 %391, 0, !dbg !16
  br i1 %392, label %393, label %425, !dbg !16

393:                                              ; preds = %390
  %394 = srem i64 %8, 32, !dbg !16
  %395 = icmp slt i64 %394, 0, !dbg !16
  %396 = add i64 %394, 32, !dbg !16
  %397 = select i1 %395, i64 %396, i64 %394, !dbg !16
  %398 = srem i64 %10, 128, !dbg !16
  %399 = icmp slt i64 %398, 0, !dbg !16
  %400 = add i64 %398, 128, !dbg !16
  %401 = select i1 %399, i64 %400, i64 %398, !dbg !16
  %402 = mul nsw i64 %279, 32, !dbg !16
  %403 = mul nsw i64 %283, 4, !dbg !16
  %404 = add i64 %402, %403, !dbg !16
  %405 = icmp slt i64 %279, 0, !dbg !16
  %406 = sub i64 -1, %279, !dbg !16
  %407 = select i1 %405, i64 %406, i64 %279, !dbg !16
  %408 = sdiv i64 %407, 128, !dbg !16
  %409 = sub i64 -1, %408, !dbg !16
  %410 = select i1 %405, i64 %409, i64 %408, !dbg !16
  %411 = mul nsw i64 %410, -4096, !dbg !16
  %412 = add i64 %404, %411, !dbg !16
  %413 = add i64 %412, 3, !dbg !16
  %414 = mul i64 %397, 524288, !dbg !16
  %415 = add i64 0, %414, !dbg !16
  %416 = mul i64 %401, 4096, !dbg !16
  %417 = add i64 %415, %416, !dbg !16
  %418 = add i64 %417, %413, !dbg !16
  %419 = getelementptr half, ptr addrspace(1) %2, i64 %418, !dbg !16
  %420 = load half, ptr addrspace(1) %419, align 2, !dbg !16
  %421 = add i64 %403, 3, !dbg !16
  %422 = mul i64 %401, 32, !dbg !16
  %423 = add i64 %422, %421, !dbg !16
  %424 = getelementptr half, ptr addrspace(3) @shm_1, i64 %423, !dbg !16
  store half %420, ptr addrspace(3) %424, align 2, !dbg !16
  br label %425, !dbg !16

425:                                              ; preds = %393, %390
  %426 = add i64 %283, 1, !dbg !16
  br label %282, !dbg !16

427:                                              ; preds = %282
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  br label %428, !dbg !19

428:                                              ; preds = %628, %427
  %429 = phi i64 [ %629, %628 ], [ 0, %427 ], !dbg !19
  %430 = icmp slt i64 %429, 8, !dbg !19
  br i1 %430, label %431, label %630, !dbg !19

431:                                              ; preds = %428
  %432 = mul nsw i64 %429, -4, !dbg !19
  %433 = add i64 %432, %276, !dbg !19
  %434 = add i64 %433, 4095, !dbg !19
  %435 = icmp sge i64 %434, 0, !dbg !19
  br i1 %435, label %436, label %481, !dbg !19

436:                                              ; preds = %431
  %437 = srem i64 %8, 32, !dbg !19
  %438 = icmp slt i64 %437, 0, !dbg !19
  %439 = add i64 %437, 32, !dbg !19
  %440 = select i1 %438, i64 %439, i64 %437, !dbg !19
  %441 = mul nsw i64 %279, 32, !dbg !19
  %442 = icmp slt i64 %279, 0, !dbg !19
  %443 = sub i64 -1, %279, !dbg !19
  %444 = select i1 %442, i64 %443, i64 %279, !dbg !19
  %445 = sdiv i64 %444, 128, !dbg !19
  %446 = sub i64 -1, %445, !dbg !19
  %447 = select i1 %442, i64 %446, i64 %445, !dbg !19
  %448 = mul nsw i64 %447, -4096, !dbg !19
  %449 = add i64 %441, %448, !dbg !19
  %450 = sdiv i64 %234, 4, !dbg !19
  %451 = sub i64 -1, %450, !dbg !19
  %452 = select i1 %232, i64 %451, i64 %450, !dbg !19
  %453 = add i64 %449, %452, !dbg !19
  %454 = icmp slt i64 %452, 0, !dbg !19
  %455 = sub i64 -1, %452, !dbg !19
  %456 = select i1 %454, i64 %455, i64 %452, !dbg !19
  %457 = sdiv i64 %456, 32, !dbg !19
  %458 = sub i64 -1, %457, !dbg !19
  %459 = select i1 %454, i64 %458, i64 %457, !dbg !19
  %460 = mul nsw i64 %459, -32, !dbg !19
  %461 = add i64 %453, %460, !dbg !19
  %462 = mul nsw i64 %429, 4, !dbg !19
  %463 = mul nsw i64 %10, 32, !dbg !19
  %464 = add i64 %462, %463, !dbg !19
  %465 = mul nsw i64 %452, -128, !dbg !19
  %466 = add i64 %464, %465, !dbg !19
  %467 = mul i64 %440, 524288, !dbg !19
  %468 = add i64 0, %467, !dbg !19
  %469 = mul i64 %461, 128, !dbg !19
  %470 = add i64 %468, %469, !dbg !19
  %471 = add i64 %470, %466, !dbg !19
  %472 = getelementptr half, ptr addrspace(1) %1, i64 %471, !dbg !19
  %473 = load half, ptr addrspace(1) %472, align 2, !dbg !19
  %474 = srem i64 %452, 32, !dbg !19
  %475 = icmp slt i64 %474, 0, !dbg !19
  %476 = add i64 %474, 32, !dbg !19
  %477 = select i1 %475, i64 %476, i64 %474, !dbg !19
  %478 = mul i64 %477, 128, !dbg !19
  %479 = add i64 %478, %466, !dbg !19
  %480 = getelementptr half, ptr addrspace(3) @shm_2, i64 %479, !dbg !19
  store half %473, ptr addrspace(3) %480, align 2, !dbg !19
  br label %481, !dbg !19

481:                                              ; preds = %436, %431
  %482 = add i64 %433, 4094, !dbg !19
  %483 = icmp sge i64 %482, 0, !dbg !19
  br i1 %483, label %484, label %530, !dbg !19

484:                                              ; preds = %481
  %485 = srem i64 %8, 32, !dbg !19
  %486 = icmp slt i64 %485, 0, !dbg !19
  %487 = add i64 %485, 32, !dbg !19
  %488 = select i1 %486, i64 %487, i64 %485, !dbg !19
  %489 = mul nsw i64 %279, 32, !dbg !19
  %490 = icmp slt i64 %279, 0, !dbg !19
  %491 = sub i64 -1, %279, !dbg !19
  %492 = select i1 %490, i64 %491, i64 %279, !dbg !19
  %493 = sdiv i64 %492, 128, !dbg !19
  %494 = sub i64 -1, %493, !dbg !19
  %495 = select i1 %490, i64 %494, i64 %493, !dbg !19
  %496 = mul nsw i64 %495, -4096, !dbg !19
  %497 = add i64 %489, %496, !dbg !19
  %498 = sdiv i64 %234, 4, !dbg !19
  %499 = sub i64 -1, %498, !dbg !19
  %500 = select i1 %232, i64 %499, i64 %498, !dbg !19
  %501 = add i64 %497, %500, !dbg !19
  %502 = icmp slt i64 %500, 0, !dbg !19
  %503 = sub i64 -1, %500, !dbg !19
  %504 = select i1 %502, i64 %503, i64 %500, !dbg !19
  %505 = sdiv i64 %504, 32, !dbg !19
  %506 = sub i64 -1, %505, !dbg !19
  %507 = select i1 %502, i64 %506, i64 %505, !dbg !19
  %508 = mul nsw i64 %507, -32, !dbg !19
  %509 = add i64 %501, %508, !dbg !19
  %510 = mul nsw i64 %429, 4, !dbg !19
  %511 = mul nsw i64 %10, 32, !dbg !19
  %512 = add i64 %510, %511, !dbg !19
  %513 = mul nsw i64 %500, -128, !dbg !19
  %514 = add i64 %512, %513, !dbg !19
  %515 = add i64 %514, 1, !dbg !19
  %516 = mul i64 %488, 524288, !dbg !19
  %517 = add i64 0, %516, !dbg !19
  %518 = mul i64 %509, 128, !dbg !19
  %519 = add i64 %517, %518, !dbg !19
  %520 = add i64 %519, %515, !dbg !19
  %521 = getelementptr half, ptr addrspace(1) %1, i64 %520, !dbg !19
  %522 = load half, ptr addrspace(1) %521, align 2, !dbg !19
  %523 = srem i64 %500, 32, !dbg !19
  %524 = icmp slt i64 %523, 0, !dbg !19
  %525 = add i64 %523, 32, !dbg !19
  %526 = select i1 %524, i64 %525, i64 %523, !dbg !19
  %527 = mul i64 %526, 128, !dbg !19
  %528 = add i64 %527, %515, !dbg !19
  %529 = getelementptr half, ptr addrspace(3) @shm_2, i64 %528, !dbg !19
  store half %522, ptr addrspace(3) %529, align 2, !dbg !19
  br label %530, !dbg !19

530:                                              ; preds = %484, %481
  %531 = add i64 %433, 4093, !dbg !19
  %532 = icmp sge i64 %531, 0, !dbg !19
  br i1 %532, label %533, label %579, !dbg !19

533:                                              ; preds = %530
  %534 = srem i64 %8, 32, !dbg !19
  %535 = icmp slt i64 %534, 0, !dbg !19
  %536 = add i64 %534, 32, !dbg !19
  %537 = select i1 %535, i64 %536, i64 %534, !dbg !19
  %538 = mul nsw i64 %279, 32, !dbg !19
  %539 = icmp slt i64 %279, 0, !dbg !19
  %540 = sub i64 -1, %279, !dbg !19
  %541 = select i1 %539, i64 %540, i64 %279, !dbg !19
  %542 = sdiv i64 %541, 128, !dbg !19
  %543 = sub i64 -1, %542, !dbg !19
  %544 = select i1 %539, i64 %543, i64 %542, !dbg !19
  %545 = mul nsw i64 %544, -4096, !dbg !19
  %546 = add i64 %538, %545, !dbg !19
  %547 = sdiv i64 %234, 4, !dbg !19
  %548 = sub i64 -1, %547, !dbg !19
  %549 = select i1 %232, i64 %548, i64 %547, !dbg !19
  %550 = add i64 %546, %549, !dbg !19
  %551 = icmp slt i64 %549, 0, !dbg !19
  %552 = sub i64 -1, %549, !dbg !19
  %553 = select i1 %551, i64 %552, i64 %549, !dbg !19
  %554 = sdiv i64 %553, 32, !dbg !19
  %555 = sub i64 -1, %554, !dbg !19
  %556 = select i1 %551, i64 %555, i64 %554, !dbg !19
  %557 = mul nsw i64 %556, -32, !dbg !19
  %558 = add i64 %550, %557, !dbg !19
  %559 = mul nsw i64 %429, 4, !dbg !19
  %560 = mul nsw i64 %10, 32, !dbg !19
  %561 = add i64 %559, %560, !dbg !19
  %562 = mul nsw i64 %549, -128, !dbg !19
  %563 = add i64 %561, %562, !dbg !19
  %564 = add i64 %563, 2, !dbg !19
  %565 = mul i64 %537, 524288, !dbg !19
  %566 = add i64 0, %565, !dbg !19
  %567 = mul i64 %558, 128, !dbg !19
  %568 = add i64 %566, %567, !dbg !19
  %569 = add i64 %568, %564, !dbg !19
  %570 = getelementptr half, ptr addrspace(1) %1, i64 %569, !dbg !19
  %571 = load half, ptr addrspace(1) %570, align 2, !dbg !19
  %572 = srem i64 %549, 32, !dbg !19
  %573 = icmp slt i64 %572, 0, !dbg !19
  %574 = add i64 %572, 32, !dbg !19
  %575 = select i1 %573, i64 %574, i64 %572, !dbg !19
  %576 = mul i64 %575, 128, !dbg !19
  %577 = add i64 %576, %564, !dbg !19
  %578 = getelementptr half, ptr addrspace(3) @shm_2, i64 %577, !dbg !19
  store half %571, ptr addrspace(3) %578, align 2, !dbg !19
  br label %579, !dbg !19

579:                                              ; preds = %533, %530
  %580 = add i64 %433, 4092, !dbg !19
  %581 = icmp sge i64 %580, 0, !dbg !19
  br i1 %581, label %582, label %628, !dbg !19

582:                                              ; preds = %579
  %583 = srem i64 %8, 32, !dbg !19
  %584 = icmp slt i64 %583, 0, !dbg !19
  %585 = add i64 %583, 32, !dbg !19
  %586 = select i1 %584, i64 %585, i64 %583, !dbg !19
  %587 = mul nsw i64 %279, 32, !dbg !19
  %588 = icmp slt i64 %279, 0, !dbg !19
  %589 = sub i64 -1, %279, !dbg !19
  %590 = select i1 %588, i64 %589, i64 %279, !dbg !19
  %591 = sdiv i64 %590, 128, !dbg !19
  %592 = sub i64 -1, %591, !dbg !19
  %593 = select i1 %588, i64 %592, i64 %591, !dbg !19
  %594 = mul nsw i64 %593, -4096, !dbg !19
  %595 = add i64 %587, %594, !dbg !19
  %596 = sdiv i64 %234, 4, !dbg !19
  %597 = sub i64 -1, %596, !dbg !19
  %598 = select i1 %232, i64 %597, i64 %596, !dbg !19
  %599 = add i64 %595, %598, !dbg !19
  %600 = icmp slt i64 %598, 0, !dbg !19
  %601 = sub i64 -1, %598, !dbg !19
  %602 = select i1 %600, i64 %601, i64 %598, !dbg !19
  %603 = sdiv i64 %602, 32, !dbg !19
  %604 = sub i64 -1, %603, !dbg !19
  %605 = select i1 %600, i64 %604, i64 %603, !dbg !19
  %606 = mul nsw i64 %605, -32, !dbg !19
  %607 = add i64 %599, %606, !dbg !19
  %608 = mul nsw i64 %429, 4, !dbg !19
  %609 = mul nsw i64 %10, 32, !dbg !19
  %610 = add i64 %608, %609, !dbg !19
  %611 = mul nsw i64 %598, -128, !dbg !19
  %612 = add i64 %610, %611, !dbg !19
  %613 = add i64 %612, 3, !dbg !19
  %614 = mul i64 %586, 524288, !dbg !19
  %615 = add i64 0, %614, !dbg !19
  %616 = mul i64 %607, 128, !dbg !19
  %617 = add i64 %615, %616, !dbg !19
  %618 = add i64 %617, %613, !dbg !19
  %619 = getelementptr half, ptr addrspace(1) %1, i64 %618, !dbg !19
  %620 = load half, ptr addrspace(1) %619, align 2, !dbg !19
  %621 = srem i64 %598, 32, !dbg !19
  %622 = icmp slt i64 %621, 0, !dbg !19
  %623 = add i64 %621, 32, !dbg !19
  %624 = select i1 %622, i64 %623, i64 %621, !dbg !19
  %625 = mul i64 %624, 128, !dbg !19
  %626 = add i64 %625, %613, !dbg !19
  %627 = getelementptr half, ptr addrspace(3) @shm_2, i64 %626, !dbg !19
  store half %620, ptr addrspace(3) %627, align 2, !dbg !19
  br label %628, !dbg !19

628:                                              ; preds = %582, %579
  %629 = add i64 %429, 1, !dbg !19
  br label %428, !dbg !19

630:                                              ; preds = %428
  fence syncscope("workgroup") release, !dbg !19
  call void @llvm.amdgcn.s.barrier(), !dbg !19
  fence syncscope("workgroup") acquire, !dbg !19
  br label %631, !dbg !17

631:                                              ; preds = %689, %630
  %632 = phi i64 [ %738, %689 ], [ 0, %630 ], !dbg !17
  %633 = phi [2 x <8 x float>] [ %737, %689 ], [ zeroinitializer, %630 ], !dbg !17
  %634 = icmp slt i64 %632, 4, !dbg !17
  br i1 %634, label %635, label %739, !dbg !17

635:                                              ; preds = %631
  %636 = mul nsw i64 %632, 16, !dbg !17
  %637 = add i64 %636, %10, !dbg !17
  %638 = icmp slt i64 %632, 0, !dbg !17
  %639 = sub i64 -1, %632, !dbg !17
  %640 = select i1 %638, i64 %639, i64 %632, !dbg !17
  %641 = sdiv i64 %640, 2, !dbg !17
  %642 = sub i64 -1, %641, !dbg !17
  %643 = select i1 %638, i64 %642, i64 %641, !dbg !17
  %644 = mul nsw i64 %643, -32, !dbg !17
  %645 = add i64 %637, %644, !dbg !17
  %646 = add i64 %645, %277, !dbg !17
  br label %647, !dbg !17

647:                                              ; preds = %651, %635
  %648 = phi i64 [ %688, %651 ], [ 0, %635 ], !dbg !17
  %649 = phi [1 x <4 x float>] [ %687, %651 ], [ zeroinitializer, %635 ], !dbg !17
  %650 = icmp slt i64 %648, 8, !dbg !17
  br i1 %650, label %651, label %689, !dbg !17

651:                                              ; preds = %647
  %652 = mul nsw i64 %648, 16, !dbg !17
  %653 = add i64 %652, %238, !dbg !17
  %654 = mul i64 %653, 32, !dbg !17
  %655 = add i64 %654, %646, !dbg !17
  %656 = getelementptr half, ptr addrspace(3) @shm_1, i64 %655, !dbg !17
  %657 = load half, ptr addrspace(3) %656, align 2, !dbg !17
  %658 = insertelement <1 x half> zeroinitializer, half %657, i64 0, !dbg !17
  %659 = add i64 %653, 1, !dbg !17
  %660 = mul i64 %659, 32, !dbg !17
  %661 = add i64 %660, %646, !dbg !17
  %662 = getelementptr half, ptr addrspace(3) @shm_1, i64 %661, !dbg !17
  %663 = load half, ptr addrspace(3) %662, align 2, !dbg !17
  %664 = insertelement <1 x half> zeroinitializer, half %663, i64 0, !dbg !17
  %665 = add i64 %653, 2, !dbg !17
  %666 = mul i64 %665, 32, !dbg !17
  %667 = add i64 %666, %646, !dbg !17
  %668 = getelementptr half, ptr addrspace(3) @shm_1, i64 %667, !dbg !17
  %669 = load half, ptr addrspace(3) %668, align 2, !dbg !17
  %670 = insertelement <1 x half> zeroinitializer, half %669, i64 0, !dbg !17
  %671 = add i64 %653, 3, !dbg !17
  %672 = mul i64 %671, 32, !dbg !17
  %673 = add i64 %672, %646, !dbg !17
  %674 = getelementptr half, ptr addrspace(3) @shm_1, i64 %673, !dbg !17
  %675 = load half, ptr addrspace(3) %674, align 2, !dbg !17
  %676 = insertelement <1 x half> zeroinitializer, half %675, i64 0, !dbg !17
  %677 = shufflevector <1 x half> %658, <1 x half> %658, <4 x i32> zeroinitializer, !dbg !17
  %678 = shufflevector <4 x half> %677, <4 x half> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !17
  %679 = shufflevector <1 x half> %664, <1 x half> %664, <4 x i32> zeroinitializer, !dbg !17
  %680 = shufflevector <4 x half> %679, <4 x half> %678, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !17
  %681 = shufflevector <1 x half> %670, <1 x half> %670, <4 x i32> zeroinitializer, !dbg !17
  %682 = shufflevector <4 x half> %681, <4 x half> %680, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !17
  %683 = shufflevector <1 x half> %676, <1 x half> %676, <4 x i32> zeroinitializer, !dbg !17
  %684 = shufflevector <4 x half> %683, <4 x half> %682, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !17
  %685 = extractvalue [1 x <4 x float>] %649, 0, !dbg !17
  %686 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %259, <4 x half> %684, <4 x float> %685), !dbg !17
  %687 = insertvalue [1 x <4 x float>] poison, <4 x float> %686, 0, !dbg !17
  %688 = add i64 %648, 1, !dbg !17
  br label %647, !dbg !17

689:                                              ; preds = %647
  %690 = extractvalue [1 x <4 x float>] %649, 0, !dbg !17
  %691 = extractelement <4 x float> %690, i64 0, !dbg !17
  %692 = mul nsw i64 %632, 4, !dbg !17
  %693 = mul nsw i64 %643, -8, !dbg !17
  %694 = add i64 %692, %693, !dbg !17
  %695 = extractvalue [2 x <8 x float>] %633, 0, !dbg !17
  %696 = shufflevector <8 x float> %695, <8 x float> %695, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %697 = shufflevector <16 x float> %696, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %698 = extractvalue [2 x <8 x float>] %633, 1, !dbg !17
  %699 = shufflevector <8 x float> %698, <8 x float> %698, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %700 = shufflevector <16 x float> %699, <16 x float> %697, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %701 = mul i64 %643, 8, !dbg !17
  %702 = add i64 %701, %694, !dbg !17
  %703 = insertelement <16 x float> %700, float %691, i64 %702, !dbg !17
  %704 = shufflevector <16 x float> %703, <16 x float> %703, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %705 = shufflevector <16 x float> %703, <16 x float> %703, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %706 = extractelement <4 x float> %690, i64 1, !dbg !17
  %707 = add i64 %694, 1, !dbg !17
  %708 = shufflevector <8 x float> %704, <8 x float> %704, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %709 = shufflevector <16 x float> %708, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %710 = shufflevector <8 x float> %705, <8 x float> %705, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %711 = shufflevector <16 x float> %710, <16 x float> %709, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %712 = add i64 %701, %707, !dbg !17
  %713 = insertelement <16 x float> %711, float %706, i64 %712, !dbg !17
  %714 = shufflevector <16 x float> %713, <16 x float> %713, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %715 = shufflevector <16 x float> %713, <16 x float> %713, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %716 = extractelement <4 x float> %690, i64 2, !dbg !17
  %717 = add i64 %694, 2, !dbg !17
  %718 = shufflevector <8 x float> %714, <8 x float> %714, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %719 = shufflevector <16 x float> %718, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %720 = shufflevector <8 x float> %715, <8 x float> %715, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %721 = shufflevector <16 x float> %720, <16 x float> %719, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %722 = add i64 %701, %717, !dbg !17
  %723 = insertelement <16 x float> %721, float %716, i64 %722, !dbg !17
  %724 = shufflevector <16 x float> %723, <16 x float> %723, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %725 = shufflevector <16 x float> %723, <16 x float> %723, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %726 = extractelement <4 x float> %690, i64 3, !dbg !17
  %727 = add i64 %694, 3, !dbg !17
  %728 = shufflevector <8 x float> %724, <8 x float> %724, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %729 = shufflevector <16 x float> %728, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !17
  %730 = shufflevector <8 x float> %725, <8 x float> %725, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !17
  %731 = shufflevector <16 x float> %730, <16 x float> %729, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %732 = add i64 %701, %727, !dbg !17
  %733 = insertelement <16 x float> %731, float %726, i64 %732, !dbg !17
  %734 = shufflevector <16 x float> %733, <16 x float> %733, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %735 = insertvalue [2 x <8 x float>] poison, <8 x float> %734, 0, !dbg !17
  %736 = shufflevector <16 x float> %733, <16 x float> %733, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !17
  %737 = insertvalue [2 x <8 x float>] %735, <8 x float> %736, 1, !dbg !17
  %738 = add i64 %632, 1, !dbg !17
  br label %631, !dbg !17

739:                                              ; preds = %631
  %740 = mul nsw i64 %279, 32, !dbg !20
  br label %741, !dbg !20

741:                                              ; preds = %812, %739
  %742 = phi i64 [ %813, %812 ], [ 0, %739 ], !dbg !20
  %743 = phi [2 x <8 x float>] [ %753, %812 ], [ zeroinitializer, %739 ], !dbg !20
  %744 = icmp slt i64 %742, 2, !dbg !20
  br i1 %744, label %745, label %814, !dbg !20

745:                                              ; preds = %741
  %746 = mul nsw i64 %742, 32, !dbg !20
  %747 = add i64 %746, %10, !dbg !20
  %748 = add i64 %747, %11, !dbg !20
  %749 = add i64 %748, %277, !dbg !20
  %750 = add i64 %749, 1, !dbg !21
  br label %751, !dbg !20

751:                                              ; preds = %755, %745
  %752 = phi i64 [ %811, %755 ], [ 0, %745 ], !dbg !20
  %753 = phi [2 x <8 x float>] [ %810, %755 ], [ %743, %745 ], !dbg !20
  %754 = icmp slt i64 %752, 2, !dbg !20
  br i1 %754, label %755, label %812, !dbg !20

755:                                              ; preds = %751
  %756 = mul nsw i64 %752, 4, !dbg !20
  %757 = mul nsw i64 %752, 16, !dbg !20
  %758 = add i64 %740, %757, !dbg !20
  %759 = add i64 %758, %237, !dbg !20
  %760 = icmp ule i64 %750, %759, !dbg !22
  %761 = select i1 %760, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %762 = extractvalue [2 x <8 x float>] %753, 0, !dbg !20
  %763 = shufflevector <8 x float> %762, <8 x float> %762, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %764 = shufflevector <16 x float> %763, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %765 = extractvalue [2 x <8 x float>] %753, 1, !dbg !20
  %766 = shufflevector <8 x float> %765, <8 x float> %765, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %767 = shufflevector <16 x float> %766, <16 x float> %764, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %768 = mul i64 %742, 8, !dbg !20
  %769 = add i64 %768, %756, !dbg !20
  %770 = insertelement <16 x float> %767, float %761, i64 %769, !dbg !20
  %771 = shufflevector <16 x float> %770, <16 x float> %770, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %772 = shufflevector <16 x float> %770, <16 x float> %770, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %773 = add i64 %756, 1, !dbg !20
  %774 = add i64 %759, 4, !dbg !20
  %775 = icmp ule i64 %750, %774, !dbg !22
  %776 = select i1 %775, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %777 = shufflevector <8 x float> %771, <8 x float> %771, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %778 = shufflevector <16 x float> %777, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %779 = shufflevector <8 x float> %772, <8 x float> %772, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %780 = shufflevector <16 x float> %779, <16 x float> %778, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %781 = add i64 %768, %773, !dbg !20
  %782 = insertelement <16 x float> %780, float %776, i64 %781, !dbg !20
  %783 = shufflevector <16 x float> %782, <16 x float> %782, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %784 = shufflevector <16 x float> %782, <16 x float> %782, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %785 = add i64 %756, 2, !dbg !20
  %786 = add i64 %759, 8, !dbg !20
  %787 = icmp ule i64 %750, %786, !dbg !22
  %788 = select i1 %787, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %789 = shufflevector <8 x float> %783, <8 x float> %783, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %790 = shufflevector <16 x float> %789, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %791 = shufflevector <8 x float> %784, <8 x float> %784, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %792 = shufflevector <16 x float> %791, <16 x float> %790, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %793 = add i64 %768, %785, !dbg !20
  %794 = insertelement <16 x float> %792, float %788, i64 %793, !dbg !20
  %795 = shufflevector <16 x float> %794, <16 x float> %794, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %796 = shufflevector <16 x float> %794, <16 x float> %794, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %797 = add i64 %756, 3, !dbg !20
  %798 = add i64 %759, 12, !dbg !20
  %799 = icmp ule i64 %750, %798, !dbg !22
  %800 = select i1 %799, float 0xFFF0000000000000, float 0.000000e+00, !dbg !23
  %801 = shufflevector <8 x float> %795, <8 x float> %795, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %802 = shufflevector <16 x float> %801, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !20
  %803 = shufflevector <8 x float> %796, <8 x float> %796, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !20
  %804 = shufflevector <16 x float> %803, <16 x float> %802, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %805 = add i64 %768, %797, !dbg !20
  %806 = insertelement <16 x float> %804, float %800, i64 %805, !dbg !20
  %807 = shufflevector <16 x float> %806, <16 x float> %806, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %808 = insertvalue [2 x <8 x float>] poison, <8 x float> %807, 0, !dbg !20
  %809 = shufflevector <16 x float> %806, <16 x float> %806, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !20
  %810 = insertvalue [2 x <8 x float>] %808, <8 x float> %809, 1, !dbg !20
  %811 = add i64 %752, 1, !dbg !20
  br label %751, !dbg !20

812:                                              ; preds = %751
  %813 = add i64 %742, 1, !dbg !20
  br label %741, !dbg !20

814:                                              ; preds = %741
  %815 = extractvalue [2 x <8 x float>] %633, 0, !dbg !24
  %816 = extractvalue [2 x <8 x float>] %743, 0, !dbg !24
  %817 = fadd <8 x float> %815, %816, !dbg !24
  %818 = extractvalue [2 x <8 x float>] %633, 1, !dbg !24
  %819 = extractvalue [2 x <8 x float>] %743, 1, !dbg !24
  %820 = fadd <8 x float> %818, %819, !dbg !24
  %821 = extractelement <8 x float> %817, i64 0, !dbg !25
  %822 = call float @__ocml_exp2_f32(float %821), !dbg !25
  %823 = insertelement <8 x float> poison, float %822, i64 0, !dbg !25
  %824 = extractelement <8 x float> %817, i64 1, !dbg !25
  %825 = call float @__ocml_exp2_f32(float %824), !dbg !25
  %826 = insertelement <8 x float> %823, float %825, i64 1, !dbg !25
  %827 = extractelement <8 x float> %817, i64 2, !dbg !25
  %828 = call float @__ocml_exp2_f32(float %827), !dbg !25
  %829 = insertelement <8 x float> %826, float %828, i64 2, !dbg !25
  %830 = extractelement <8 x float> %817, i64 3, !dbg !25
  %831 = call float @__ocml_exp2_f32(float %830), !dbg !25
  %832 = insertelement <8 x float> %829, float %831, i64 3, !dbg !25
  %833 = extractelement <8 x float> %817, i64 4, !dbg !25
  %834 = call float @__ocml_exp2_f32(float %833), !dbg !25
  %835 = insertelement <8 x float> %832, float %834, i64 4, !dbg !25
  %836 = extractelement <8 x float> %817, i64 5, !dbg !25
  %837 = call float @__ocml_exp2_f32(float %836), !dbg !25
  %838 = insertelement <8 x float> %835, float %837, i64 5, !dbg !25
  %839 = extractelement <8 x float> %817, i64 6, !dbg !25
  %840 = call float @__ocml_exp2_f32(float %839), !dbg !25
  %841 = insertelement <8 x float> %838, float %840, i64 6, !dbg !25
  %842 = extractelement <8 x float> %817, i64 7, !dbg !25
  %843 = call float @__ocml_exp2_f32(float %842), !dbg !25
  %844 = insertelement <8 x float> %841, float %843, i64 7, !dbg !25
  %845 = extractelement <8 x float> %820, i64 0, !dbg !25
  %846 = call float @__ocml_exp2_f32(float %845), !dbg !25
  %847 = insertelement <8 x float> poison, float %846, i64 0, !dbg !25
  %848 = extractelement <8 x float> %820, i64 1, !dbg !25
  %849 = call float @__ocml_exp2_f32(float %848), !dbg !25
  %850 = insertelement <8 x float> %847, float %849, i64 1, !dbg !25
  %851 = extractelement <8 x float> %820, i64 2, !dbg !25
  %852 = call float @__ocml_exp2_f32(float %851), !dbg !25
  %853 = insertelement <8 x float> %850, float %852, i64 2, !dbg !25
  %854 = extractelement <8 x float> %820, i64 3, !dbg !25
  %855 = call float @__ocml_exp2_f32(float %854), !dbg !25
  %856 = insertelement <8 x float> %853, float %855, i64 3, !dbg !25
  %857 = extractelement <8 x float> %820, i64 4, !dbg !25
  %858 = call float @__ocml_exp2_f32(float %857), !dbg !25
  %859 = insertelement <8 x float> %856, float %858, i64 4, !dbg !25
  %860 = extractelement <8 x float> %820, i64 5, !dbg !25
  %861 = call float @__ocml_exp2_f32(float %860), !dbg !25
  %862 = insertelement <8 x float> %859, float %861, i64 5, !dbg !25
  %863 = extractelement <8 x float> %820, i64 6, !dbg !25
  %864 = call float @__ocml_exp2_f32(float %863), !dbg !25
  %865 = insertelement <8 x float> %862, float %864, i64 6, !dbg !25
  %866 = extractelement <8 x float> %820, i64 7, !dbg !25
  %867 = call float @__ocml_exp2_f32(float %866), !dbg !25
  %868 = insertelement <8 x float> %865, float %867, i64 7, !dbg !25
  br label %869, !dbg !13

869:                                              ; preds = %931, %814
  %870 = phi i64 [ %932, %931 ], [ 0, %814 ], !dbg !13
  %871 = icmp slt i64 %870, 2, !dbg !13
  br i1 %871, label %872, label %933, !dbg !13

872:                                              ; preds = %869
  store float 0.000000e+00, ptr addrspace(5) %271, align 4, !dbg !13
  br label %873, !dbg !13

873:                                              ; preds = %876, %872
  %874 = phi i64 [ %902, %876 ], [ 0, %872 ], !dbg !13
  %875 = icmp slt i64 %874, 2, !dbg !13
  br i1 %875, label %876, label %903, !dbg !13

876:                                              ; preds = %873
  %877 = mul nsw i64 %874, 4, !dbg !13
  %878 = load float, ptr addrspace(5) %271, align 4, !dbg !13
  %879 = shufflevector <8 x float> %844, <8 x float> %844, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !13
  %880 = shufflevector <16 x float> %879, <16 x float> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>, !dbg !13
  %881 = shufflevector <8 x float> %868, <8 x float> %868, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>, !dbg !13
  %882 = shufflevector <16 x float> %881, <16 x float> %880, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %883 = mul i64 %870, 8, !dbg !13
  %884 = add i64 %883, %877, !dbg !13
  %885 = extractelement <16 x float> %882, i64 %884, !dbg !13
  %886 = fadd float %878, %885, !dbg !13
  store float %886, ptr addrspace(5) %271, align 4, !dbg !13
  %887 = add i64 %877, 1, !dbg !13
  %888 = load float, ptr addrspace(5) %271, align 4, !dbg !13
  %889 = add i64 %883, %887, !dbg !13
  %890 = extractelement <16 x float> %882, i64 %889, !dbg !13
  %891 = fadd float %888, %890, !dbg !13
  store float %891, ptr addrspace(5) %271, align 4, !dbg !13
  %892 = add i64 %877, 2, !dbg !13
  %893 = load float, ptr addrspace(5) %271, align 4, !dbg !13
  %894 = add i64 %883, %892, !dbg !13
  %895 = extractelement <16 x float> %882, i64 %894, !dbg !13
  %896 = fadd float %893, %895, !dbg !13
  store float %896, ptr addrspace(5) %271, align 4, !dbg !13
  %897 = add i64 %877, 3, !dbg !13
  %898 = load float, ptr addrspace(5) %271, align 4, !dbg !13
  %899 = add i64 %883, %897, !dbg !13
  %900 = extractelement <16 x float> %882, i64 %899, !dbg !13
  %901 = fadd float %898, %900, !dbg !13
  store float %901, ptr addrspace(5) %271, align 4, !dbg !13
  %902 = add i64 %874, 1, !dbg !13
  br label %873, !dbg !13

903:                                              ; preds = %873
  %904 = load float, ptr addrspace(5) %271, align 4, !dbg !13
  %905 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %906 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %905), !dbg !13
  %907 = add i32 %906, 64, !dbg !13
  %908 = xor i32 %906, 16, !dbg !13
  %909 = and i32 %907, -64, !dbg !13
  %910 = icmp slt i32 %908, %909, !dbg !13
  %911 = select i1 %910, i32 %908, i32 %906, !dbg !13
  %912 = shl i32 %911, 2, !dbg !13
  %913 = bitcast float %904 to i32, !dbg !13
  %914 = call i32 @llvm.amdgcn.ds.bpermute(i32 %912, i32 %913), !dbg !13
  %915 = bitcast i32 %914 to float, !dbg !13
  %916 = fadd float %904, %915, !dbg !13
  %917 = xor i32 %906, 32, !dbg !13
  %918 = icmp slt i32 %917, %909, !dbg !13
  %919 = select i1 %918, i32 %917, i32 %906, !dbg !13
  %920 = shl i32 %919, 2, !dbg !13
  %921 = bitcast float %916 to i32, !dbg !13
  %922 = call i32 @llvm.amdgcn.ds.bpermute(i32 %920, i32 %921), !dbg !13
  %923 = bitcast i32 %922 to float, !dbg !13
  %924 = fadd float %916, %923, !dbg !13
  br i1 %270, label %925, label %931, !dbg !13

925:                                              ; preds = %903
  %926 = mul nsw i64 %870, 32, !dbg !13
  %927 = add i64 %926, %10, !dbg !13
  %928 = add i64 %927, %277, !dbg !13
  %929 = add i64 %928, 0, !dbg !13
  %930 = getelementptr float, ptr addrspace(3) @shm_3, i64 %929, !dbg !13
  store float %924, ptr addrspace(3) %930, align 4, !dbg !13
  br label %931, !dbg !13

931:                                              ; preds = %925, %903
  %932 = add i64 %870, 1, !dbg !13
  br label %869, !dbg !13

933:                                              ; preds = %869
  fence syncscope("workgroup") release, !dbg !26
  call void @llvm.amdgcn.s.barrier(), !dbg !26
  fence syncscope("workgroup") acquire, !dbg !26
  br label %934, !dbg !26

934:                                              ; preds = %937, %933
  %935 = phi i64 [ %938, %937 ], [ 0, %933 ], !dbg !26
  %936 = icmp slt i64 %935, 2, !dbg !26
  br i1 %936, label %937, label %939, !dbg !26

937:                                              ; preds = %934
  %938 = add i64 %935, 1, !dbg !26
  br label %934, !dbg !26

939:                                              ; preds = %934
  br label %940, !dbg !26

940:                                              ; preds = %943, %939
  %941 = phi i64 [ %944, %943 ], [ 0, %939 ], !dbg !26
  %942 = icmp slt i64 %941, 2, !dbg !26
  br i1 %942, label %943, label %945, !dbg !26

943:                                              ; preds = %940
  %944 = add i64 %941, 1, !dbg !26
  br label %940, !dbg !26

945:                                              ; preds = %940
  br label %946, !dbg !14

946:                                              ; preds = %972, %945
  %947 = phi i64 [ %973, %972 ], [ 0, %945 ], !dbg !14
  %948 = icmp slt i64 %947, 2, !dbg !14
  br i1 %948, label %949, label %974, !dbg !14

949:                                              ; preds = %946
  %950 = mul nsw i64 %947, 32, !dbg !14
  %951 = add i64 %950, %10, !dbg !14
  %952 = add i64 %951, %277, !dbg !14
  br label %953, !dbg !14

953:                                              ; preds = %956, %949
  %954 = phi i64 [ %971, %956 ], [ 0, %949 ], !dbg !14
  %955 = icmp slt i64 %954, 2, !dbg !14
  br i1 %955, label %956, label %972, !dbg !14

956:                                              ; preds = %953
  %957 = getelementptr float, ptr addrspace(5) %272, i64 0, !dbg !14
  %958 = load <4 x float>, ptr addrspace(5) %957, align 4, !dbg !14
  %959 = fptrunc <4 x float> %958 to <4 x half>, !dbg !14
  %960 = getelementptr half, ptr addrspace(5) %273, i64 0, !dbg !14
  store <4 x half> %959, ptr addrspace(5) %960, align 2, !dbg !14
  %961 = load <4 x half>, ptr addrspace(5) %960, align 2, !dbg !14
  %962 = mul nsw i64 %954, 16, !dbg !14
  %963 = add i64 %962, %238, !dbg !14
  %964 = sub i64 32, %963, !dbg !14
  %965 = insertelement <4 x i64> poison, i64 %964, i32 0, !dbg !14
  %966 = shufflevector <4 x i64> %965, <4 x i64> poison, <4 x i32> zeroinitializer, !dbg !14
  %967 = icmp sgt <4 x i64> %966, <i64 0, i64 1, i64 2, i64 3>, !dbg !14
  %968 = mul i64 %952, 32, !dbg !14
  %969 = add i64 %968, %963, !dbg !14
  %970 = getelementptr half, ptr addrspace(3) @shm_4, i64 %969, !dbg !14
  call void @llvm.masked.store.v4f16.p3(<4 x half> %961, ptr addrspace(3) %970, i32 2, <4 x i1> %967), !dbg !14
  %971 = add i64 %954, 1, !dbg !14
  br label %953, !dbg !14

972:                                              ; preds = %953
  %973 = add i64 %947, 1, !dbg !14
  br label %946, !dbg !14

974:                                              ; preds = %946
  br label %975, !dbg !27

975:                                              ; preds = %1052, %974
  %976 = phi i64 [ %1053, %1052 ], [ 0, %974 ], !dbg !27
  %977 = icmp slt i64 %976, 16, !dbg !27
  br i1 %977, label %978, label %1054, !dbg !27

978:                                              ; preds = %975
  %979 = icmp slt i64 %976, 0, !dbg !27
  %980 = sub i64 -1, %976, !dbg !27
  %981 = select i1 %979, i64 %980, i64 %976, !dbg !27
  %982 = sdiv i64 %981, 8, !dbg !27
  %983 = sub i64 -1, %982, !dbg !27
  %984 = select i1 %979, i64 %983, i64 %982, !dbg !27
  %985 = mul nsw i64 %984, 32, !dbg !27
  %986 = add i64 %985, %10, !dbg !27
  %987 = add i64 %986, %277, !dbg !27
  %988 = mul nsw i64 %976, 16, !dbg !27
  %989 = add i64 %988, %10, !dbg !27
  %990 = mul nsw i64 %984, -128, !dbg !27
  %991 = add i64 %989, %990, !dbg !27
  %992 = add i64 %991, %277, !dbg !27
  br label %993, !dbg !27

993:                                              ; preds = %997, %978
  %994 = phi i64 [ %1051, %997 ], [ 0, %978 ], !dbg !27
  %995 = phi [1 x <4 x float>] [ %1050, %997 ], [ zeroinitializer, %978 ], !dbg !27
  %996 = icmp slt i64 %994, 2, !dbg !27
  br i1 %996, label %997, label %1052, !dbg !27

997:                                              ; preds = %993
  %998 = mul nsw i64 %994, 16, !dbg !27
  %999 = add i64 %998, %238, !dbg !27
  %1000 = mul i64 %987, 32, !dbg !27
  %1001 = add i64 %1000, %999, !dbg !27
  %1002 = getelementptr half, ptr addrspace(3) @shm_4, i64 %1001, !dbg !27
  %1003 = load half, ptr addrspace(3) %1002, align 2, !dbg !27
  %1004 = insertelement <4 x half> zeroinitializer, half %1003, i64 0, !dbg !27
  %1005 = add i64 %999, 1, !dbg !27
  %1006 = add i64 %1000, %1005, !dbg !27
  %1007 = getelementptr half, ptr addrspace(3) @shm_4, i64 %1006, !dbg !27
  %1008 = load half, ptr addrspace(3) %1007, align 2, !dbg !27
  %1009 = insertelement <4 x half> %1004, half %1008, i64 1, !dbg !27
  %1010 = add i64 %999, 2, !dbg !27
  %1011 = add i64 %1000, %1010, !dbg !27
  %1012 = getelementptr half, ptr addrspace(3) @shm_4, i64 %1011, !dbg !27
  %1013 = load half, ptr addrspace(3) %1012, align 2, !dbg !27
  %1014 = insertelement <4 x half> %1009, half %1013, i64 2, !dbg !27
  %1015 = add i64 %999, 3, !dbg !27
  %1016 = add i64 %1000, %1015, !dbg !27
  %1017 = getelementptr half, ptr addrspace(3) @shm_4, i64 %1016, !dbg !27
  %1018 = load half, ptr addrspace(3) %1017, align 2, !dbg !27
  %1019 = insertelement <4 x half> %1014, half %1018, i64 3, !dbg !27
  %1020 = mul i64 %999, 128, !dbg !27
  %1021 = add i64 %1020, %992, !dbg !27
  %1022 = getelementptr half, ptr addrspace(3) @shm_2, i64 %1021, !dbg !27
  %1023 = load half, ptr addrspace(3) %1022, align 2, !dbg !27
  %1024 = insertelement <1 x half> zeroinitializer, half %1023, i64 0, !dbg !27
  %1025 = mul i64 %1005, 128, !dbg !27
  %1026 = add i64 %1025, %992, !dbg !27
  %1027 = getelementptr half, ptr addrspace(3) @shm_2, i64 %1026, !dbg !27
  %1028 = load half, ptr addrspace(3) %1027, align 2, !dbg !27
  %1029 = insertelement <1 x half> zeroinitializer, half %1028, i64 0, !dbg !27
  %1030 = mul i64 %1010, 128, !dbg !27
  %1031 = add i64 %1030, %992, !dbg !27
  %1032 = getelementptr half, ptr addrspace(3) @shm_2, i64 %1031, !dbg !27
  %1033 = load half, ptr addrspace(3) %1032, align 2, !dbg !27
  %1034 = insertelement <1 x half> zeroinitializer, half %1033, i64 0, !dbg !27
  %1035 = mul i64 %1015, 128, !dbg !27
  %1036 = add i64 %1035, %992, !dbg !27
  %1037 = getelementptr half, ptr addrspace(3) @shm_2, i64 %1036, !dbg !27
  %1038 = load half, ptr addrspace(3) %1037, align 2, !dbg !27
  %1039 = insertelement <1 x half> zeroinitializer, half %1038, i64 0, !dbg !27
  %1040 = shufflevector <1 x half> %1024, <1 x half> %1024, <4 x i32> zeroinitializer, !dbg !27
  %1041 = shufflevector <4 x half> %1040, <4 x half> poison, <4 x i32> <i32 0, i32 5, i32 6, i32 7>, !dbg !27
  %1042 = shufflevector <1 x half> %1029, <1 x half> %1029, <4 x i32> zeroinitializer, !dbg !27
  %1043 = shufflevector <4 x half> %1042, <4 x half> %1041, <4 x i32> <i32 4, i32 0, i32 6, i32 7>, !dbg !27
  %1044 = shufflevector <1 x half> %1034, <1 x half> %1034, <4 x i32> zeroinitializer, !dbg !27
  %1045 = shufflevector <4 x half> %1044, <4 x half> %1043, <4 x i32> <i32 4, i32 5, i32 0, i32 7>, !dbg !27
  %1046 = shufflevector <1 x half> %1039, <1 x half> %1039, <4 x i32> zeroinitializer, !dbg !27
  %1047 = shufflevector <4 x half> %1046, <4 x half> %1045, <4 x i32> <i32 4, i32 5, i32 6, i32 0>, !dbg !27
  %1048 = extractvalue [1 x <4 x float>] %995, 0, !dbg !27
  %1049 = call <4 x float> asm sideeffect "v_mmac_f32_16x16x16_f16 $0, $2, $1, $3", "=v,v,v,0"(<4 x half> %1019, <4 x half> %1047, <4 x float> %1048), !dbg !27
  %1050 = insertvalue [1 x <4 x float>] poison, <4 x float> %1049, 0, !dbg !27
  %1051 = add i64 %994, 1, !dbg !27
  br label %993, !dbg !27

1052:                                             ; preds = %993
  %1053 = add i64 %976, 1, !dbg !27
  br label %975, !dbg !27

1054:                                             ; preds = %975
  br label %1055, !dbg !28

1055:                                             ; preds = %1064, %1054
  %1056 = phi i64 [ %1065, %1064 ], [ 0, %1054 ], !dbg !28
  %1057 = icmp slt i64 %1056, 2, !dbg !28
  br i1 %1057, label %1058, label %1066, !dbg !28

1058:                                             ; preds = %1055
  br label %1059, !dbg !28

1059:                                             ; preds = %1062, %1058
  %1060 = phi i64 [ %1063, %1062 ], [ 0, %1058 ], !dbg !28
  %1061 = icmp slt i64 %1060, 8, !dbg !28
  br i1 %1061, label %1062, label %1064, !dbg !28

1062:                                             ; preds = %1059
  %1063 = add i64 %1060, 1, !dbg !28
  br label %1059, !dbg !28

1064:                                             ; preds = %1059
  %1065 = add i64 %1056, 1, !dbg !28
  br label %1055, !dbg !28

1066:                                             ; preds = %1055
  br label %1067, !dbg !29

1067:                                             ; preds = %1076, %1066
  %1068 = phi i64 [ %1077, %1076 ], [ 0, %1066 ], !dbg !29
  %1069 = icmp slt i64 %1068, 2, !dbg !29
  br i1 %1069, label %1070, label %1078, !dbg !29

1070:                                             ; preds = %1067
  br label %1071, !dbg !29

1071:                                             ; preds = %1074, %1070
  %1072 = phi i64 [ %1075, %1074 ], [ 0, %1070 ], !dbg !29
  %1073 = icmp slt i64 %1072, 8, !dbg !29
  br i1 %1073, label %1074, label %1076, !dbg !29

1074:                                             ; preds = %1071
  %1075 = add i64 %1072, 1, !dbg !29
  br label %1071, !dbg !29

1076:                                             ; preds = %1071
  %1077 = add i64 %1068, 1, !dbg !29
  br label %1067, !dbg !29

1078:                                             ; preds = %1067
  br label %1079, !dbg !30

1079:                                             ; preds = %1082, %1078
  %1080 = phi i64 [ %1083, %1082 ], [ 0, %1078 ], !dbg !30
  %1081 = icmp slt i64 %1080, 2, !dbg !30
  br i1 %1081, label %1082, label %1084, !dbg !30

1082:                                             ; preds = %1079
  %1083 = add i64 %1080, 1, !dbg !30
  br label %1079, !dbg !30

1084:                                             ; preds = %1079
  %1085 = add i64 %279, 1, !dbg !15
  br label %278, !dbg !15

1086:                                             ; preds = %278
  fence syncscope("workgroup") release, !dbg !31
  call void @llvm.amdgcn.s.barrier(), !dbg !31
  fence syncscope("workgroup") acquire, !dbg !31
  br label %1087, !dbg !31

1087:                                             ; preds = %1291, %1086
  %1088 = phi i64 [ %1292, %1291 ], [ 0, %1086 ], !dbg !31
  %1089 = icmp slt i64 %1088, 16, !dbg !31
  br i1 %1089, label %1090, label %1293, !dbg !31

1090:                                             ; preds = %1087
  %1091 = mul nsw i64 %1088, -4, !dbg !31
  %1092 = add i64 %1091, %12, !dbg !31
  %1093 = add i64 %1092, 8191, !dbg !31
  %1094 = icmp sge i64 %1093, 0, !dbg !31
  br i1 %1094, label %1095, label %1141, !dbg !31

1095:                                             ; preds = %1090
  %1096 = sdiv i64 %234, 2, !dbg !31
  %1097 = sub i64 -1, %1096, !dbg !31
  %1098 = select i1 %232, i64 %1097, i64 %1096, !dbg !31
  %1099 = srem i64 %1098, 64, !dbg !31
  %1100 = icmp slt i64 %1099, 0, !dbg !31
  %1101 = add i64 %1099, 64, !dbg !31
  %1102 = select i1 %1100, i64 %1101, i64 %1099, !dbg !31
  %1103 = mul nsw i64 %1088, 4, !dbg !31
  %1104 = mul nsw i64 %10, 64, !dbg !31
  %1105 = add i64 %1103, %1104, !dbg !31
  %1106 = mul nsw i64 %1098, -128, !dbg !31
  %1107 = add i64 %1105, %1106, !dbg !31
  %1108 = mul i64 %1102, 128, !dbg !31
  %1109 = add i64 %1108, %1107, !dbg !31
  %1110 = getelementptr half, ptr addrspace(3) @shm_5, i64 %1109, !dbg !31
  %1111 = load half, ptr addrspace(3) %1110, align 2, !dbg !31
  %1112 = icmp slt i64 %6, 0, !dbg !31
  %1113 = sub i64 -1, %6, !dbg !31
  %1114 = select i1 %1112, i64 %1113, i64 %6, !dbg !31
  %1115 = sdiv i64 %1114, 64, !dbg !31
  %1116 = sub i64 -1, %1115, !dbg !31
  %1117 = select i1 %1112, i64 %1116, i64 %1115, !dbg !31
  %1118 = add i64 %8, %1117, !dbg !31
  %1119 = srem i64 %1118, 32, !dbg !31
  %1120 = icmp slt i64 %1119, 0, !dbg !31
  %1121 = add i64 %1119, 32, !dbg !31
  %1122 = select i1 %1120, i64 %1121, i64 %1119, !dbg !31
  %1123 = mul nsw i64 %6, 64, !dbg !31
  %1124 = add i64 %1123, %1098, !dbg !31
  %1125 = icmp slt i64 %1098, 0, !dbg !31
  %1126 = sub i64 -1, %1098, !dbg !31
  %1127 = select i1 %1125, i64 %1126, i64 %1098, !dbg !31
  %1128 = sdiv i64 %1127, 64, !dbg !31
  %1129 = sub i64 -1, %1128, !dbg !31
  %1130 = select i1 %1125, i64 %1129, i64 %1128, !dbg !31
  %1131 = mul nsw i64 %1130, -64, !dbg !31
  %1132 = add i64 %1124, %1131, !dbg !31
  %1133 = mul nsw i64 %1117, -4096, !dbg !31
  %1134 = add i64 %1132, %1133, !dbg !31
  %1135 = mul i64 %1122, 524288, !dbg !31
  %1136 = add i64 0, %1135, !dbg !31
  %1137 = mul i64 %1134, 128, !dbg !31
  %1138 = add i64 %1136, %1137, !dbg !31
  %1139 = add i64 %1138, %1107, !dbg !31
  %1140 = getelementptr half, ptr addrspace(1) %3, i64 %1139, !dbg !31
  store half %1111, ptr addrspace(1) %1140, align 2, !dbg !31
  br label %1141, !dbg !31

1141:                                             ; preds = %1095, %1090
  %1142 = add i64 %1092, 8190, !dbg !31
  %1143 = icmp sge i64 %1142, 0, !dbg !31
  br i1 %1143, label %1144, label %1191, !dbg !31

1144:                                             ; preds = %1141
  %1145 = sdiv i64 %234, 2, !dbg !31
  %1146 = sub i64 -1, %1145, !dbg !31
  %1147 = select i1 %232, i64 %1146, i64 %1145, !dbg !31
  %1148 = srem i64 %1147, 64, !dbg !31
  %1149 = icmp slt i64 %1148, 0, !dbg !31
  %1150 = add i64 %1148, 64, !dbg !31
  %1151 = select i1 %1149, i64 %1150, i64 %1148, !dbg !31
  %1152 = mul nsw i64 %1088, 4, !dbg !31
  %1153 = mul nsw i64 %10, 64, !dbg !31
  %1154 = add i64 %1152, %1153, !dbg !31
  %1155 = mul nsw i64 %1147, -128, !dbg !31
  %1156 = add i64 %1154, %1155, !dbg !31
  %1157 = add i64 %1156, 1, !dbg !31
  %1158 = mul i64 %1151, 128, !dbg !31
  %1159 = add i64 %1158, %1157, !dbg !31
  %1160 = getelementptr half, ptr addrspace(3) @shm_5, i64 %1159, !dbg !31
  %1161 = load half, ptr addrspace(3) %1160, align 2, !dbg !31
  %1162 = icmp slt i64 %6, 0, !dbg !31
  %1163 = sub i64 -1, %6, !dbg !31
  %1164 = select i1 %1162, i64 %1163, i64 %6, !dbg !31
  %1165 = sdiv i64 %1164, 64, !dbg !31
  %1166 = sub i64 -1, %1165, !dbg !31
  %1167 = select i1 %1162, i64 %1166, i64 %1165, !dbg !31
  %1168 = add i64 %8, %1167, !dbg !31
  %1169 = srem i64 %1168, 32, !dbg !31
  %1170 = icmp slt i64 %1169, 0, !dbg !31
  %1171 = add i64 %1169, 32, !dbg !31
  %1172 = select i1 %1170, i64 %1171, i64 %1169, !dbg !31
  %1173 = mul nsw i64 %6, 64, !dbg !31
  %1174 = add i64 %1173, %1147, !dbg !31
  %1175 = icmp slt i64 %1147, 0, !dbg !31
  %1176 = sub i64 -1, %1147, !dbg !31
  %1177 = select i1 %1175, i64 %1176, i64 %1147, !dbg !31
  %1178 = sdiv i64 %1177, 64, !dbg !31
  %1179 = sub i64 -1, %1178, !dbg !31
  %1180 = select i1 %1175, i64 %1179, i64 %1178, !dbg !31
  %1181 = mul nsw i64 %1180, -64, !dbg !31
  %1182 = add i64 %1174, %1181, !dbg !31
  %1183 = mul nsw i64 %1167, -4096, !dbg !31
  %1184 = add i64 %1182, %1183, !dbg !31
  %1185 = mul i64 %1172, 524288, !dbg !31
  %1186 = add i64 0, %1185, !dbg !31
  %1187 = mul i64 %1184, 128, !dbg !31
  %1188 = add i64 %1186, %1187, !dbg !31
  %1189 = add i64 %1188, %1157, !dbg !31
  %1190 = getelementptr half, ptr addrspace(1) %3, i64 %1189, !dbg !31
  store half %1161, ptr addrspace(1) %1190, align 2, !dbg !31
  br label %1191, !dbg !31

1191:                                             ; preds = %1144, %1141
  %1192 = add i64 %1092, 8189, !dbg !31
  %1193 = icmp sge i64 %1192, 0, !dbg !31
  br i1 %1193, label %1194, label %1241, !dbg !31

1194:                                             ; preds = %1191
  %1195 = sdiv i64 %234, 2, !dbg !31
  %1196 = sub i64 -1, %1195, !dbg !31
  %1197 = select i1 %232, i64 %1196, i64 %1195, !dbg !31
  %1198 = srem i64 %1197, 64, !dbg !31
  %1199 = icmp slt i64 %1198, 0, !dbg !31
  %1200 = add i64 %1198, 64, !dbg !31
  %1201 = select i1 %1199, i64 %1200, i64 %1198, !dbg !31
  %1202 = mul nsw i64 %1088, 4, !dbg !31
  %1203 = mul nsw i64 %10, 64, !dbg !31
  %1204 = add i64 %1202, %1203, !dbg !31
  %1205 = mul nsw i64 %1197, -128, !dbg !31
  %1206 = add i64 %1204, %1205, !dbg !31
  %1207 = add i64 %1206, 2, !dbg !31
  %1208 = mul i64 %1201, 128, !dbg !31
  %1209 = add i64 %1208, %1207, !dbg !31
  %1210 = getelementptr half, ptr addrspace(3) @shm_5, i64 %1209, !dbg !31
  %1211 = load half, ptr addrspace(3) %1210, align 2, !dbg !31
  %1212 = icmp slt i64 %6, 0, !dbg !31
  %1213 = sub i64 -1, %6, !dbg !31
  %1214 = select i1 %1212, i64 %1213, i64 %6, !dbg !31
  %1215 = sdiv i64 %1214, 64, !dbg !31
  %1216 = sub i64 -1, %1215, !dbg !31
  %1217 = select i1 %1212, i64 %1216, i64 %1215, !dbg !31
  %1218 = add i64 %8, %1217, !dbg !31
  %1219 = srem i64 %1218, 32, !dbg !31
  %1220 = icmp slt i64 %1219, 0, !dbg !31
  %1221 = add i64 %1219, 32, !dbg !31
  %1222 = select i1 %1220, i64 %1221, i64 %1219, !dbg !31
  %1223 = mul nsw i64 %6, 64, !dbg !31
  %1224 = add i64 %1223, %1197, !dbg !31
  %1225 = icmp slt i64 %1197, 0, !dbg !31
  %1226 = sub i64 -1, %1197, !dbg !31
  %1227 = select i1 %1225, i64 %1226, i64 %1197, !dbg !31
  %1228 = sdiv i64 %1227, 64, !dbg !31
  %1229 = sub i64 -1, %1228, !dbg !31
  %1230 = select i1 %1225, i64 %1229, i64 %1228, !dbg !31
  %1231 = mul nsw i64 %1230, -64, !dbg !31
  %1232 = add i64 %1224, %1231, !dbg !31
  %1233 = mul nsw i64 %1217, -4096, !dbg !31
  %1234 = add i64 %1232, %1233, !dbg !31
  %1235 = mul i64 %1222, 524288, !dbg !31
  %1236 = add i64 0, %1235, !dbg !31
  %1237 = mul i64 %1234, 128, !dbg !31
  %1238 = add i64 %1236, %1237, !dbg !31
  %1239 = add i64 %1238, %1207, !dbg !31
  %1240 = getelementptr half, ptr addrspace(1) %3, i64 %1239, !dbg !31
  store half %1211, ptr addrspace(1) %1240, align 2, !dbg !31
  br label %1241, !dbg !31

1241:                                             ; preds = %1194, %1191
  %1242 = add i64 %1092, 8188, !dbg !31
  %1243 = icmp sge i64 %1242, 0, !dbg !31
  br i1 %1243, label %1244, label %1291, !dbg !31

1244:                                             ; preds = %1241
  %1245 = sdiv i64 %234, 2, !dbg !31
  %1246 = sub i64 -1, %1245, !dbg !31
  %1247 = select i1 %232, i64 %1246, i64 %1245, !dbg !31
  %1248 = srem i64 %1247, 64, !dbg !31
  %1249 = icmp slt i64 %1248, 0, !dbg !31
  %1250 = add i64 %1248, 64, !dbg !31
  %1251 = select i1 %1249, i64 %1250, i64 %1248, !dbg !31
  %1252 = mul nsw i64 %1088, 4, !dbg !31
  %1253 = mul nsw i64 %10, 64, !dbg !31
  %1254 = add i64 %1252, %1253, !dbg !31
  %1255 = mul nsw i64 %1247, -128, !dbg !31
  %1256 = add i64 %1254, %1255, !dbg !31
  %1257 = add i64 %1256, 3, !dbg !31
  %1258 = mul i64 %1251, 128, !dbg !31
  %1259 = add i64 %1258, %1257, !dbg !31
  %1260 = getelementptr half, ptr addrspace(3) @shm_5, i64 %1259, !dbg !31
  %1261 = load half, ptr addrspace(3) %1260, align 2, !dbg !31
  %1262 = icmp slt i64 %6, 0, !dbg !31
  %1263 = sub i64 -1, %6, !dbg !31
  %1264 = select i1 %1262, i64 %1263, i64 %6, !dbg !31
  %1265 = sdiv i64 %1264, 64, !dbg !31
  %1266 = sub i64 -1, %1265, !dbg !31
  %1267 = select i1 %1262, i64 %1266, i64 %1265, !dbg !31
  %1268 = add i64 %8, %1267, !dbg !31
  %1269 = srem i64 %1268, 32, !dbg !31
  %1270 = icmp slt i64 %1269, 0, !dbg !31
  %1271 = add i64 %1269, 32, !dbg !31
  %1272 = select i1 %1270, i64 %1271, i64 %1269, !dbg !31
  %1273 = mul nsw i64 %6, 64, !dbg !31
  %1274 = add i64 %1273, %1247, !dbg !31
  %1275 = icmp slt i64 %1247, 0, !dbg !31
  %1276 = sub i64 -1, %1247, !dbg !31
  %1277 = select i1 %1275, i64 %1276, i64 %1247, !dbg !31
  %1278 = sdiv i64 %1277, 64, !dbg !31
  %1279 = sub i64 -1, %1278, !dbg !31
  %1280 = select i1 %1275, i64 %1279, i64 %1278, !dbg !31
  %1281 = mul nsw i64 %1280, -64, !dbg !31
  %1282 = add i64 %1274, %1281, !dbg !31
  %1283 = mul nsw i64 %1267, -4096, !dbg !31
  %1284 = add i64 %1282, %1283, !dbg !31
  %1285 = mul i64 %1272, 524288, !dbg !31
  %1286 = add i64 0, %1285, !dbg !31
  %1287 = mul i64 %1284, 128, !dbg !31
  %1288 = add i64 %1286, %1287, !dbg !31
  %1289 = add i64 %1288, %1257, !dbg !31
  %1290 = getelementptr half, ptr addrspace(1) %3, i64 %1289, !dbg !31
  store half %1261, ptr addrspace(1) %1290, align 2, !dbg !31
  br label %1291, !dbg !31

1291:                                             ; preds = %1244, %1241
  %1292 = add i64 %1088, 1, !dbg !31
  br label %1087, !dbg !31

1293:                                             ; preds = %1087
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

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" "amdgpu-waves-per-eu"="1" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { nocallback nofree nosync nounwind willreturn }
attributes #4 = { nocallback nofree nosync nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_friskBase.mlir", directory: "test")
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
!12 = !DILocation(line: 18, column: 19, scope: !3)
!13 = !DILocation(line: 45, column: 7, scope: !3)
!14 = !DILocation(line: 48, column: 7, scope: !3)
!15 = !DILocation(line: 25, column: 12, scope: !3)
!16 = !DILocation(line: 27, column: 7, scope: !3)
!17 = !DILocation(line: 30, column: 13, scope: !3)
!18 = !DILocation(line: 25, column: 23, scope: !3)
!19 = !DILocation(line: 29, column: 7, scope: !3)
!20 = !DILocation(line: 31, column: 19, scope: !3)
!21 = !DILocation(line: 33, column: 15, scope: !3)
!22 = !DILocation(line: 34, column: 15, scope: !3)
!23 = !DILocation(line: 35, column: 15, scope: !3)
!24 = !DILocation(line: 42, column: 13, scope: !3)
!25 = !DILocation(line: 43, column: 15, scope: !3)
!26 = !DILocation(line: 46, column: 13, scope: !3)
!27 = !DILocation(line: 49, column: 16, scope: !3)
!28 = !DILocation(line: 50, column: 13, scope: !3)
!29 = !DILocation(line: 51, column: 7, scope: !3)
!30 = !DILocation(line: 52, column: 7, scope: !3)
!31 = !DILocation(line: 58, column: 5, scope: !3)
!32 = !DILocation(line: 59, column: 5, scope: !3)
