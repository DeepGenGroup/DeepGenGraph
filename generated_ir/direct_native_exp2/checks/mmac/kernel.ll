; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !28 {
  %5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %6 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5), !dbg !8
  %7 = add nuw nsw i32 %6, 64, !dbg !8
  %8 = xor i32 %6, 16, !dbg !8
  %9 = and i32 %7, 192, !dbg !8
  %10 = icmp samesign ult i32 %8, %9, !dbg !8
  %11 = select i1 %10, i32 %8, i32 %6, !dbg !8
  %12 = shl nuw nsw i32 %11, 2, !dbg !8
  %13 = xor i32 %6, 32, !dbg !8
  %14 = icmp samesign ult i32 %13, %9, !dbg !8
  %15 = select i1 %14, i32 %13, i32 %6, !dbg !8
  %16 = shl nuw nsw i32 %15, 2, !dbg !8
  %17 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %20 = sext i32 %19 to i64, !dbg !5
  %.lobit = ashr i32 %19, 31, !dbg !11
  %21 = xor i32 %19, %.lobit, !dbg !11
  %22 = sdiv i32 %21, 32, !dbg !11
  %.lobit1 = ashr i32 %19, 31, !dbg !11
  %23 = xor i32 %22, %.lobit1, !dbg !11
  %24 = sext i32 %23 to i64, !dbg !11
  %25 = add nsw i64 %18, %24, !dbg !11
  %26 = and i64 %25, 31, !dbg !11
  %27 = shl nsw i64 %20, 7, !dbg !11
  %28 = mul nsw i64 %24, -4096, !dbg !11
  %29 = add nsw i64 %27, %28, !dbg !11
  %30 = shl nsw i64 %29, 7, !dbg !11
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !30
  %32 = zext nneg i32 %31 to i64, !dbg !13
  %33 = shl nuw nsw i64 %32, 2, !dbg !13
  %34 = add nsw i64 %18, %24, !dbg !11
  %35 = and i64 %34, 31, !dbg !11
  %36 = shl nuw nsw i64 %35, 19, !dbg !11
  %37 = add nsw i64 %36, %30, !dbg !11
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !11, !range !30, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local249 = lshr i64 %address.local, 6, !dbg !11
  %address.local250 = shl nuw nsw i64 %address.local249, 7, !dbg !11
  %address.local251 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local250, !dbg !11
  %address.local252 = ptrtoint ptr addrspace(3) %address.local251 to i64, !dbg !11
  %address.local253 = trunc i64 %address.local252 to i32, !dbg !11
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local253), !dbg !11
  %39 = zext i32 %38 to i64, !dbg !11
  %40 = inttoptr i64 %39 to ptr addrspace(3), !dbg !11
  %address.local254 = lshr i64 %address.local, 6, !dbg !11
  %address.local255 = or disjoint i64 %address.local254, 4, !dbg !11
  %address.local256 = shl nuw nsw i64 %address.local255, 7, !dbg !11
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local256, !dbg !11
  %address.local258 = ptrtoint ptr addrspace(3) %address.local257 to i64, !dbg !11
  %address.local259 = trunc i64 %address.local258 to i32, !dbg !11
  %41 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local259), !dbg !11
  %42 = zext i32 %41 to i64, !dbg !11
  %43 = inttoptr i64 %42 to ptr addrspace(3), !dbg !11
  %address.local260 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local261 = and i64 %address.local260, 384, !dbg !11
  %address.local262 = or disjoint i64 %address.local261, 1024, !dbg !11
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local262, !dbg !11
  %address.local264 = ptrtoint ptr addrspace(3) %address.local263 to i64, !dbg !11
  %address.local265 = trunc i64 %address.local264 to i32, !dbg !11
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local265), !dbg !11
  %45 = zext i32 %44 to i64, !dbg !11
  %46 = inttoptr i64 %45 to ptr addrspace(3), !dbg !11
  %address.local266 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local267 = and i64 %address.local266, 384, !dbg !11
  %address.local268 = or disjoint i64 %address.local267, 1536, !dbg !11
  %address.local269 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local268, !dbg !11
  %address.local270 = ptrtoint ptr addrspace(3) %address.local269 to i64, !dbg !11
  %address.local271 = trunc i64 %address.local270 to i32, !dbg !11
  %47 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local271), !dbg !11
  %48 = zext i32 %47 to i64, !dbg !11
  %49 = inttoptr i64 %48 to ptr addrspace(3), !dbg !11
  %address.local272 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local273 = and i64 %address.local272, 384, !dbg !11
  %address.local274 = or disjoint i64 %address.local273, 2048, !dbg !11
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local274, !dbg !11
  %address.local276 = ptrtoint ptr addrspace(3) %address.local275 to i64, !dbg !11
  %address.local277 = trunc i64 %address.local276 to i32, !dbg !11
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local277), !dbg !11
  %51 = zext i32 %50 to i64, !dbg !11
  %52 = inttoptr i64 %51 to ptr addrspace(3), !dbg !11
  %address.local278 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local279 = and i64 %address.local278, 384, !dbg !11
  %address.local280 = or disjoint i64 %address.local279, 2560, !dbg !11
  %address.local281 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local280, !dbg !11
  %address.local282 = ptrtoint ptr addrspace(3) %address.local281 to i64, !dbg !11
  %address.local283 = trunc i64 %address.local282 to i32, !dbg !11
  %53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local283), !dbg !11
  %54 = zext i32 %53 to i64, !dbg !11
  %55 = inttoptr i64 %54 to ptr addrspace(3), !dbg !11
  %address.local284 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local285 = and i64 %address.local284, 384, !dbg !11
  %address.local286 = or disjoint i64 %address.local285, 3072, !dbg !11
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local286, !dbg !11
  %address.local288 = ptrtoint ptr addrspace(3) %address.local287 to i64, !dbg !11
  %address.local289 = trunc i64 %address.local288 to i32, !dbg !11
  %56 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local289), !dbg !11
  %57 = zext i32 %56 to i64, !dbg !11
  %58 = inttoptr i64 %57 to ptr addrspace(3), !dbg !11
  %address.local290 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local291 = and i64 %address.local290, 384, !dbg !11
  %address.local292 = or disjoint i64 %address.local291, 3584, !dbg !11
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local292, !dbg !11
  %address.local294 = ptrtoint ptr addrspace(3) %address.local293 to i64, !dbg !11
  %address.local295 = trunc i64 %address.local294 to i32, !dbg !11
  %59 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local295), !dbg !11
  %60 = zext i32 %59 to i64, !dbg !11
  %61 = inttoptr i64 %60 to ptr addrspace(3), !dbg !11
  %address.local296 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local250, !dbg !12
  %address.local297 = ptrtoint ptr addrspace(3) %address.local296 to i64, !dbg !12
  %address.local298 = trunc i64 %address.local297 to i32, !dbg !12
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local298), !dbg !12
  %63 = zext i32 %62 to i64, !dbg !12
  %64 = inttoptr i64 %63 to ptr addrspace(3), !dbg !12
  %address.local299 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !12
  %address.local300 = ptrtoint ptr addrspace(3) %address.local299 to i64, !dbg !12
  %address.local301 = trunc i64 %address.local300 to i32, !dbg !12
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local301), !dbg !12
  %66 = zext i32 %65 to i64, !dbg !12
  %67 = inttoptr i64 %66 to ptr addrspace(3), !dbg !12
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local262, !dbg !12
  %address.local303 = ptrtoint ptr addrspace(3) %address.local302 to i64, !dbg !12
  %address.local304 = trunc i64 %address.local303 to i32, !dbg !12
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local304), !dbg !12
  %69 = zext i32 %68 to i64, !dbg !12
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !12
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local268, !dbg !12
  %address.local306 = ptrtoint ptr addrspace(3) %address.local305 to i64, !dbg !12
  %address.local307 = trunc i64 %address.local306 to i32, !dbg !12
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local307), !dbg !12
  %72 = zext i32 %71 to i64, !dbg !12
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !12
  %address.local308 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local274, !dbg !12
  %address.local309 = ptrtoint ptr addrspace(3) %address.local308 to i64, !dbg !12
  %address.local310 = trunc i64 %address.local309 to i32, !dbg !12
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local310), !dbg !12
  %75 = zext i32 %74 to i64, !dbg !12
  %76 = inttoptr i64 %75 to ptr addrspace(3), !dbg !12
  %address.local311 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local280, !dbg !12
  %address.local312 = ptrtoint ptr addrspace(3) %address.local311 to i64, !dbg !12
  %address.local313 = trunc i64 %address.local312 to i32, !dbg !12
  %77 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local313), !dbg !12
  %78 = zext i32 %77 to i64, !dbg !12
  %79 = inttoptr i64 %78 to ptr addrspace(3), !dbg !12
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local286, !dbg !12
  %address.local315 = ptrtoint ptr addrspace(3) %address.local314 to i64, !dbg !12
  %address.local316 = trunc i64 %address.local315 to i32, !dbg !12
  %80 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local316), !dbg !12
  %81 = zext i32 %80 to i64, !dbg !12
  %82 = inttoptr i64 %81 to ptr addrspace(3), !dbg !12
  %address.local317 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local292, !dbg !12
  %address.local318 = ptrtoint ptr addrspace(3) %address.local317 to i64, !dbg !12
  %address.local319 = trunc i64 %address.local318 to i32, !dbg !12
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local319), !dbg !12
  %84 = zext i32 %83 to i64, !dbg !12
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !12
  %address.local320 = shl nuw nsw i64 %address.local249, 10, !dbg !15
  %address.local321 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !15
  %address.local322 = ptrtoint ptr addrspace(3) %address.local321 to i64, !dbg !15
  %address.local323 = trunc i64 %address.local322 to i32, !dbg !15
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local323), !dbg !15
  %address.local324 = shl nuw nsw i64 %address.local255, 10, !dbg !15
  %address.local325 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local324, !dbg !15
  %address.local326 = ptrtoint ptr addrspace(3) %address.local325 to i64, !dbg !15
  %address.local327 = trunc i64 %address.local326 to i32, !dbg !15
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local327), !dbg !15
  %.idx = shl nuw nsw i64 %26, 20, !dbg !11
  %88 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx, !dbg !11
  %89 = getelementptr [2 x i8], ptr addrspace(1) %88, i64 %30, !dbg !11
  %90 = lshr i64 %32, 2, !dbg !16
  %91 = lshr i64 %32, 2, !dbg !16
  %92 = and i64 %91, 3, !dbg !16
  %93 = lshr i64 %32, 1, !dbg !16
  %94 = lshr i64 %32, 5, !dbg !16
  %95 = and i64 %94, 1, !dbg !16
  %96 = mul nsw i64 %90, -18, !dbg !16
  %97 = add nsw i64 %33, %96, !dbg !16
  %98 = add nsw i64 %97, %93, !dbg !16
  %99 = add nsw i64 %98, %95, !dbg !16
  %.neg = sdiv i64 %99, -2, !dbg !16
  %100 = mul nuw nsw i64 %32, 18, !dbg !16
  %101 = mul nsw i64 %90, -72, !dbg !16
  %102 = add nsw i64 %100, %101, !dbg !16
  %103 = add nsw i64 %102, %92, !dbg !16
  %.neg179 = sdiv i64 %103, -4, !dbg !16
  %104 = shl nuw nsw i64 %32, 4, !dbg !16
  %105 = shl nuw nsw i64 %90, 6, !dbg !16
  %106 = sub nsw i64 %104, %105, !dbg !16
  %107 = or disjoint i64 %106, %92, !dbg !16
  %108 = shl nuw nsw i32 %31, 1, !dbg !16
  %109 = and i32 %108, 6, !dbg !16
  %index.compact109 = zext nneg i32 %109 to i64, !dbg !16
  %110 = lshr i32 %31, 1, !dbg !16
  %111 = and i32 %110, 1, !dbg !16
  %index.compact106 = zext nneg i32 %111 to i64, !dbg !16
  %112 = shl nuw nsw i64 %90, 4, !dbg !16
  %113 = sub nsw i64 %33, %112, !dbg !16
  %114 = or disjoint i64 %113, %index.compact106, !dbg !16
  %115 = add nuw nsw i64 %114, %95, !dbg !16
  %116 = add nsw i64 %115, 1, !dbg !16
  %.neg182 = sdiv i64 %116, -2, !dbg !16
  %117 = add nuw nsw i64 %107, %index.compact109, !dbg !16
  %.neg184 = sdiv i64 %117, -4, !dbg !16
  %118 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %37, !dbg !11
  %119 = getelementptr i8, ptr addrspace(1) %118, i64 64, !dbg !11
  %120 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %37, !dbg !11
  %121 = getelementptr i8, ptr addrspace(1) %120, i64 128, !dbg !11
  %122 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %37, !dbg !11
  %123 = getelementptr i8, ptr addrspace(1) %122, i64 192, !dbg !11
  br label %124, !dbg !6

124:                                              ; preds = %133, %4
  %125 = phi i64 [ %1328, %133 ], [ 0, %4 ], !dbg !6
  %126 = phi [2 x <8 x float>] [ %1313, %133 ], [ zeroinitializer, %4 ], !dbg !6
  %127 = phi [2 x <1 x float>] [ %986, %133 ], [ zeroinitializer, %4 ], !dbg !6
  %128 = phi [2 x <1 x float>] [ %1315, %133 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %129 = phi [2 x <8 x float>] [ %1319, %133 ], [ zeroinitializer, %4 ], !dbg !6
  %130 = phi [2 x <8 x float>] [ %1323, %133 ], [ zeroinitializer, %4 ], !dbg !6
  %131 = phi [2 x <8 x float>] [ %1327, %133 ], [ zeroinitializer, %4 ], !dbg !6
  %132 = icmp samesign ult i64 %125, 32, !dbg !6
  br i1 %132, label %133, label %1329, !dbg !6

133:                                              ; preds = %124
  %address.thread328 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !11, !range !30, !frisk.register_lifetimes !1
  %address.local329 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local330 = and i32 %address.local329, 2, !dbg !11
  %address.local331 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local332 = and i32 %address.local331, 16256, !dbg !11
  %address.local333 = or disjoint i32 %address.local330, %address.local332, !dbg !11
  %address.local334 = zext nneg i32 %address.local333 to i64, !dbg !11
  %address.local335 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local334, !dbg !11
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local335 to i64, !dbg !11
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #10, !dbg !11
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local336 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local337 = and i32 %address.local336, 2, !dbg !11
  %address.local338 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local339 = and i32 %address.local338, 16256, !dbg !11
  %address.local340 = or disjoint i32 %address.local337, %address.local339, !dbg !11
  %address.local341 = or disjoint i32 %address.local340, 4, !dbg !11
  %address.local342 = zext nneg i32 %address.local341 to i64, !dbg !11
  %address.local343 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local342, !dbg !11
  %dma.source.bits1231 = ptrtoint ptr addrspace(1) %address.local343 to i64, !dbg !11
  %dma.source.opaque1232 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1231) #10, !dbg !11
  %dma.source.ptr1233 = inttoptr i64 %dma.source.opaque1232 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1233, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local344 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local345 = and i32 %address.local344, 2, !dbg !11
  %address.local346 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local347 = and i32 %address.local346, 16256, !dbg !11
  %address.local348 = or disjoint i32 %address.local345, %address.local347, !dbg !11
  %address.local349 = or disjoint i32 %address.local348, 8, !dbg !11
  %address.local350 = zext nneg i32 %address.local349 to i64, !dbg !11
  %address.local351 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local350, !dbg !11
  %dma.source.bits1234 = ptrtoint ptr addrspace(1) %address.local351 to i64, !dbg !11
  %dma.source.opaque1235 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1234) #10, !dbg !11
  %dma.source.ptr1236 = inttoptr i64 %dma.source.opaque1235 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1236, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local352 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local353 = and i32 %address.local352, 2, !dbg !11
  %address.local354 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local355 = and i32 %address.local354, 16256, !dbg !11
  %address.local356 = or disjoint i32 %address.local353, %address.local355, !dbg !11
  %address.local357 = or disjoint i32 %address.local356, 12, !dbg !11
  %address.local358 = zext nneg i32 %address.local357 to i64, !dbg !11
  %address.local359 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local358, !dbg !11
  %dma.source.bits1237 = ptrtoint ptr addrspace(1) %address.local359 to i64, !dbg !11
  %dma.source.opaque1238 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1237) #10, !dbg !11
  %dma.source.ptr1239 = inttoptr i64 %dma.source.opaque1238 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1239, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local360 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local361 = and i32 %address.local360, 2, !dbg !11
  %address.local362 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local363 = and i32 %address.local362, 16256, !dbg !11
  %address.local364 = or disjoint i32 %address.local361, %address.local363, !dbg !11
  %address.local365 = or disjoint i32 %address.local364, 16, !dbg !11
  %address.local366 = zext nneg i32 %address.local365 to i64, !dbg !11
  %address.local367 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local366, !dbg !11
  %dma.source.bits1240 = ptrtoint ptr addrspace(1) %address.local367 to i64, !dbg !11
  %dma.source.opaque1241 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1240) #10, !dbg !11
  %dma.source.ptr1242 = inttoptr i64 %dma.source.opaque1241 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1242, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local368 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local369 = and i32 %address.local368, 2, !dbg !11
  %address.local370 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local371 = and i32 %address.local370, 16256, !dbg !11
  %address.local372 = or disjoint i32 %address.local369, %address.local371, !dbg !11
  %address.local373 = or disjoint i32 %address.local372, 20, !dbg !11
  %address.local374 = zext nneg i32 %address.local373 to i64, !dbg !11
  %address.local375 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local374, !dbg !11
  %dma.source.bits1243 = ptrtoint ptr addrspace(1) %address.local375 to i64, !dbg !11
  %dma.source.opaque1244 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1243) #10, !dbg !11
  %dma.source.ptr1245 = inttoptr i64 %dma.source.opaque1244 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1245, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local376 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local377 = and i32 %address.local376, 2, !dbg !11
  %address.local378 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local379 = and i32 %address.local378, 16256, !dbg !11
  %address.local380 = or disjoint i32 %address.local377, %address.local379, !dbg !11
  %address.local381 = or disjoint i32 %address.local380, 24, !dbg !11
  %address.local382 = zext nneg i32 %address.local381 to i64, !dbg !11
  %address.local383 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local382, !dbg !11
  %dma.source.bits1246 = ptrtoint ptr addrspace(1) %address.local383 to i64, !dbg !11
  %dma.source.opaque1247 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1246) #10, !dbg !11
  %dma.source.ptr1248 = inttoptr i64 %dma.source.opaque1247 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1248, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local384 = shl nuw nsw i32 %address.thread328, 1, !dbg !11
  %address.local385 = and i32 %address.local384, 2, !dbg !11
  %address.local386 = shl nuw nsw i32 %address.thread328, 6, !dbg !11
  %address.local387 = and i32 %address.local386, 16256, !dbg !11
  %address.local388 = or disjoint i32 %address.local385, %address.local387, !dbg !11
  %address.local389 = or disjoint i32 %address.local388, 28, !dbg !11
  %address.local390 = zext nneg i32 %address.local389 to i64, !dbg !11
  %address.local391 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local390, !dbg !11
  %dma.source.bits1249 = ptrtoint ptr addrspace(1) %address.local391 to i64, !dbg !11
  %dma.source.opaque1250 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1249) #10, !dbg !11
  %dma.source.ptr1251 = inttoptr i64 %dma.source.opaque1250 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1251, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %134 = shl nsw i64 %18, 19, !dbg !12
  %135 = and i64 %134, 16252928, !dbg !12
  %136 = shl nuw nsw i64 %125, 14, !dbg !12
  %137 = or disjoint i64 %135, %136, !dbg !12
  %138 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %137, !dbg !12
  %address.local392 = zext nneg i32 %address.thread328 to i64, !dbg !13
  %address.local393 = lshr i64 %address.local392, 2, !dbg !12
  %address.local394 = lshr i64 %address.local392, 5, !dbg !12
  %address.local395 = mul nuw nsw i64 %address.local394, 7, !dbg !12
  %address.local396 = add nuw nsw i64 %address.local393, %address.local395, !dbg !12
  %address.local397 = and i64 %address.local393, 60, !dbg !12
  %address.local398 = add nuw nsw i64 %address.local396, %address.local397, !dbg !12
  %address.local399 = lshr i64 %address.local392, 7, !dbg !12
  %address.local400 = sub nsw i64 %address.local393, %address.local394, !dbg !12
  %address.local401 = sdiv i64 %address.local400, 4, !dbg !12
  %address.local402 = add nsw i64 %address.local399, %address.local401, !dbg !12
  %address.local403 = shl nsw i64 %address.local402, 2, !dbg !12
  %address.local404 = lshr i64 %address.local392, 4, !dbg !12
  %address.local405 = sub nsw i64 %address.local404, %address.local399, !dbg !12
  %address.local406 = sdiv i64 %address.local405, 2, !dbg !12
  %address.local407 = shl nsw i64 %address.local406, 3, !dbg !12
  %address.local408 = add nsw i64 %address.local403, %address.local407, !dbg !12
  %address.local409 = sub i64 %address.local398, %address.local408, !dbg !12
  %address.local410 = shl i64 %address.local409, 7, !dbg !12
  %address.local411 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local412 = and i32 %address.local411, 2, !dbg !12
  %address.local413 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local414 = and i32 %address.local413, 16, !dbg !12
  %address.local415 = or disjoint i32 %address.local412, %address.local414, !dbg !12
  %address.local416 = zext nneg i32 %address.local415 to i64, !dbg !12
  %address.local417 = or disjoint i64 %address.local410, %address.local416, !dbg !12
  %address.local418 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local417, !dbg !12
  %dma.source.bits1252 = ptrtoint ptr addrspace(1) %address.local418 to i64, !dbg !12
  %dma.source.opaque1253 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1252) #10, !dbg !12
  %dma.source.ptr1254 = inttoptr i64 %dma.source.opaque1253 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1254, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local419 = or disjoint i64 %address.local392, 256, !dbg !11
  %address.local420 = lshr i64 %address.local419, 2, !dbg !12
  %address.local421 = lshr i64 %address.local419, 5, !dbg !12
  %address.local422 = mul nuw nsw i64 %address.local421, 7, !dbg !12
  %address.local423 = add nuw nsw i64 %address.local420, %address.local422, !dbg !12
  %address.local424 = and i64 %address.local420, 124, !dbg !12
  %address.local425 = add nuw nsw i64 %address.local423, %address.local424, !dbg !12
  %address.local426 = sub nuw nsw i64 %address.local420, %address.local421, !dbg !12
  %address.local427 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local428 = and i32 %address.local427, 4, !dbg !12
  %address.local429 = or disjoint i32 %address.local428, 8, !dbg !12
  %address.local430 = zext nneg i32 %address.local429 to i64, !dbg !12
  %address.local431 = add nuw nsw i64 %address.local426, %address.local430, !dbg !12
  %address.local432 = and i64 %address.local431, 1073741820, !dbg !12
  %address.local433 = lshr i64 %address.local419, 4, !dbg !12
  %address.local434 = lshr i32 %address.thread328, 7, !dbg !12
  %address.local435 = or disjoint i32 %address.local434, 2, !dbg !12
  %address.local436 = zext nneg i32 %address.local435 to i64, !dbg !12
  %address.local437 = sub nuw nsw i64 %address.local433, %address.local436, !dbg !12
  %address.local438 = shl nuw nsw i64 %address.local437, 2, !dbg !12
  %address.local439 = and i64 %address.local438, 536870904, !dbg !12
  %address.local440 = add nuw nsw i64 %address.local432, %address.local439, !dbg !12
  %address.local441 = sub nsw i64 %address.local425, %address.local440, !dbg !12
  %address.local442 = shl i64 %address.local441, 7, !dbg !12
  %address.local443 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local444 = and i32 %address.local443, 2, !dbg !12
  %address.local445 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local446 = and i32 %address.local445, 16, !dbg !12
  %address.local447 = or disjoint i32 %address.local444, %address.local446, !dbg !12
  %address.local448 = zext nneg i32 %address.local447 to i64, !dbg !12
  %address.local449 = or disjoint i64 %address.local442, %address.local448, !dbg !12
  %address.local450 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local449, !dbg !12
  %dma.source.bits1255 = ptrtoint ptr addrspace(1) %address.local450 to i64, !dbg !12
  %dma.source.opaque1256 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1255) #10, !dbg !12
  %dma.source.ptr1257 = inttoptr i64 %dma.source.opaque1256 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1257, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local451 = lshr i64 %address.local392, 2, !dbg !12
  %address.local452 = lshr i64 %address.local392, 5, !dbg !12
  %address.local453 = mul nuw nsw i64 %address.local452, 7, !dbg !12
  %address.local454 = add nuw nsw i64 %address.local451, %address.local453, !dbg !12
  %address.local455 = and i64 %address.local451, 60, !dbg !12
  %address.local456 = add nuw nsw i64 %address.local454, %address.local455, !dbg !12
  %address.local457 = sub nsw i64 %address.local451, %address.local452, !dbg !12
  %address.local458 = sdiv i64 %address.local457, 4, !dbg !12
  %address.local459 = lshr i32 %address.thread328, 7, !dbg !12
  %address.local460 = zext nneg i32 %address.local459 to i64, !dbg !12
  %address.local461 = add nsw i64 %address.local458, %address.local460, !dbg !12
  %address.local462 = shl nsw i64 %address.local461, 2, !dbg !12
  %address.local463 = lshr i64 %address.local392, 4, !dbg !12
  %address.local464 = lshr i32 %address.thread328, 7, !dbg !12
  %address.local465 = zext nneg i32 %address.local464 to i64, !dbg !12
  %address.local466 = sub nsw i64 %address.local463, %address.local465, !dbg !12
  %address.local467 = sdiv i64 %address.local466, 2, !dbg !12
  %address.local468 = shl nsw i64 %address.local467, 3, !dbg !12
  %address.local469 = add nsw i64 %address.local462, %address.local468, !dbg !12
  %address.local470 = sub i64 %address.local456, %address.local469, !dbg !12
  %address.local471 = shl i64 %address.local470, 7, !dbg !12
  %address.local472 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local473 = and i32 %address.local472, 2, !dbg !12
  %address.local474 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local475 = and i32 %address.local474, 16, !dbg !12
  %address.local476 = or disjoint i32 %address.local473, %address.local475, !dbg !12
  %address.local477 = or disjoint i32 %address.local476, 4, !dbg !12
  %address.local478 = zext nneg i32 %address.local477 to i64, !dbg !12
  %address.local479 = or disjoint i64 %address.local471, %address.local478, !dbg !12
  %address.local480 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local479, !dbg !12
  %dma.source.bits1258 = ptrtoint ptr addrspace(1) %address.local480 to i64, !dbg !12
  %dma.source.opaque1259 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1258) #10, !dbg !12
  %dma.source.ptr1260 = inttoptr i64 %dma.source.opaque1259 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1260, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local481 = lshr i64 %address.local392, 2, !dbg !12
  %address.local482 = or disjoint i64 %address.local481, 192, !dbg !12
  %address.local483 = and i64 %address.local482, 127, !dbg !12
  %address.local484 = lshr i64 %address.local483, 3, !dbg !12
  %address.local485 = mul nuw nsw i64 %address.local484, 7, !dbg !12
  %address.local486 = add nuw nsw i64 %address.local483, %address.local485, !dbg !12
  %address.local487 = and i64 %address.local482, 124, !dbg !12
  %address.local488 = add nuw nsw i64 %address.local486, %address.local487, !dbg !12
  %address.local489 = sub nuw nsw i64 %address.local483, %address.local484, !dbg !12
  %address.local490 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local491 = and i32 %address.local490, 4, !dbg !12
  %address.local492 = or disjoint i32 %address.local491, 8, !dbg !12
  %address.local493 = zext nneg i32 %address.local492 to i64, !dbg !12
  %address.local494 = add nuw nsw i64 %address.local489, %address.local493, !dbg !12
  %address.local495 = and i64 %address.local494, 252, !dbg !12
  %address.local496 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local497 = and i32 %address.local496, 4, !dbg !12
  %address.local498 = or disjoint i32 %address.local497, 8, !dbg !12
  %address.local499 = zext nneg i32 %address.local498 to i64, !dbg !12
  %address.local500 = sub nuw nsw i64 %address.local482, %address.local499, !dbg !12
  %address.local501 = and i64 %address.local500, 120, !dbg !12
  %address.local502 = add nuw nsw i64 %address.local495, %address.local501, !dbg !12
  %address.local503 = sub nsw i64 %address.local488, %address.local502, !dbg !12
  %address.local504 = shl nsw i64 %address.local503, 7, !dbg !12
  %address.local505 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local506 = and i32 %address.local505, 2, !dbg !12
  %address.local507 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local508 = and i32 %address.local507, 16, !dbg !12
  %address.local509 = or disjoint i32 %address.local506, %address.local508, !dbg !12
  %address.local510 = or disjoint i32 %address.local509, 4, !dbg !12
  %address.local511 = zext nneg i32 %address.local510 to i64, !dbg !12
  %address.local512 = or disjoint i64 %address.local504, %address.local511, !dbg !12
  %address.local513 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local512, !dbg !12
  %dma.source.bits1261 = ptrtoint ptr addrspace(1) %address.local513 to i64, !dbg !12
  %dma.source.opaque1262 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1261) #10, !dbg !12
  %dma.source.ptr1263 = inttoptr i64 %dma.source.opaque1262 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1263, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local514 = lshr i64 %address.local392, 2, !dbg !12
  %address.local515 = lshr i64 %address.local392, 5, !dbg !12
  %address.local516 = mul nuw nsw i64 %address.local515, 7, !dbg !12
  %address.local517 = add nuw nsw i64 %address.local514, %address.local516, !dbg !12
  %address.local518 = and i64 %address.local514, 60, !dbg !12
  %address.local519 = add nuw nsw i64 %address.local517, %address.local518, !dbg !12
  %address.local520 = add nuw nsw i64 %address.local519, 144115188075855868, !dbg !12
  %address.local521 = lshr i64 %address.local392, 7, !dbg !12
  %address.local522 = sub nsw i64 %address.local514, %address.local515, !dbg !12
  %address.local523 = sdiv i64 %address.local522, 4, !dbg !12
  %address.local524 = add nsw i64 %address.local521, %address.local523, !dbg !12
  %address.local525 = shl nsw i64 %address.local524, 2, !dbg !12
  %address.local526 = lshr i64 %address.local392, 4, !dbg !12
  %address.local527 = xor i64 %address.local521, -1, !dbg !12
  %address.local528 = add nsw i64 %address.local526, %address.local527, !dbg !12
  %address.local529 = icmp slt i64 %address.local528, 0, !dbg !12
  %address.local530 = sub nsw i64 %address.local521, %address.local526, !dbg !12
  %address.local531 = select i1 %address.local529, i64 %address.local530, i64 %address.local528, !dbg !12
  %address.local532 = sdiv i64 %address.local531, 2, !dbg !12
  %address.local533 = ashr i64 %address.local528, 63, !dbg !12
  %address.local534 = xor i64 %address.local532, %address.local533, !dbg !12
  %address.local535 = shl nsw i64 %address.local534, 3, !dbg !12
  %address.local536 = add i64 %address.local525, %address.local535, !dbg !12
  %address.local537 = sub i64 %address.local520, %address.local536, !dbg !12
  %address.local538 = shl i64 %address.local537, 7, !dbg !12
  %address.local539 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local540 = and i32 %address.local539, 2, !dbg !12
  %address.local541 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local542 = and i32 %address.local541, 16, !dbg !12
  %address.local543 = or disjoint i32 %address.local540, %address.local542, !dbg !12
  %address.local544 = or disjoint i32 %address.local543, 8, !dbg !12
  %address.local545 = zext nneg i32 %address.local544 to i64, !dbg !12
  %address.local546 = or disjoint i64 %address.local538, %address.local545, !dbg !12
  %address.local547 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local546, !dbg !12
  %dma.source.bits1264 = ptrtoint ptr addrspace(1) %address.local547 to i64, !dbg !12
  %dma.source.opaque1265 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1264) #10, !dbg !12
  %dma.source.ptr1266 = inttoptr i64 %dma.source.opaque1265 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1266, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local548 = lshr i64 %address.local392, 2, !dbg !12
  %address.local549 = or disjoint i64 %address.local548, 320, !dbg !12
  %address.local550 = and i64 %address.local549, 127, !dbg !12
  %address.local551 = lshr i64 %address.local550, 3, !dbg !12
  %address.local552 = mul nuw nsw i64 %address.local551, 7, !dbg !12
  %address.local553 = add nuw nsw i64 %address.local550, %address.local552, !dbg !12
  %address.local554 = and i64 %address.local549, 124, !dbg !12
  %address.local555 = add nuw nsw i64 %address.local553, %address.local554, !dbg !12
  %address.local556 = sub nuw nsw i64 %address.local550, %address.local551, !dbg !12
  %address.local557 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local558 = and i32 %address.local557, 4, !dbg !12
  %address.local559 = add nuw nsw i32 %address.local558, 12, !dbg !12
  %address.local560 = zext nneg i32 %address.local559 to i64, !dbg !12
  %address.local561 = add nuw nsw i64 %address.local556, %address.local560, !dbg !12
  %address.local562 = and i64 %address.local561, 252, !dbg !12
  %address.local563 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local564 = and i32 %address.local563, 4, !dbg !12
  %address.local565 = add nuw nsw i32 %address.local564, 12, !dbg !12
  %address.local566 = zext nneg i32 %address.local565 to i64, !dbg !12
  %address.local567 = sub nuw nsw i64 %address.local549, %address.local566, !dbg !12
  %address.local568 = and i64 %address.local567, 120, !dbg !12
  %address.local569 = add nuw nsw i64 %address.local562, %address.local568, !dbg !12
  %address.local570 = sub nsw i64 %address.local555, %address.local569, !dbg !12
  %address.local571 = shl nsw i64 %address.local570, 7, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local573 = and i32 %address.local572, 2, !dbg !12
  %address.local574 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local575 = and i32 %address.local574, 16, !dbg !12
  %address.local576 = or disjoint i32 %address.local573, %address.local575, !dbg !12
  %address.local577 = or disjoint i32 %address.local576, 8, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.local579 = or disjoint i64 %address.local571, %address.local578, !dbg !12
  %address.local580 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local579, !dbg !12
  %dma.source.bits1267 = ptrtoint ptr addrspace(1) %address.local580 to i64, !dbg !12
  %dma.source.opaque1268 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1267) #10, !dbg !12
  %dma.source.ptr1269 = inttoptr i64 %dma.source.opaque1268 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1269, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local581 = lshr i64 %address.local392, 2, !dbg !12
  %address.local582 = lshr i64 %address.local392, 5, !dbg !12
  %address.local583 = mul nuw nsw i64 %address.local582, 7, !dbg !12
  %address.local584 = add nuw nsw i64 %address.local581, %address.local583, !dbg !12
  %address.local585 = and i64 %address.local581, 60, !dbg !12
  %address.local586 = add nuw nsw i64 %address.local584, %address.local585, !dbg !12
  %address.local587 = sub nsw i64 %address.local581, %address.local582, !dbg !12
  %address.local588 = sdiv i64 %address.local587, 4, !dbg !12
  %address.local589 = lshr i32 %address.thread328, 7, !dbg !12
  %address.local590 = add nuw nsw i32 %address.local589, 1, !dbg !12
  %address.local591 = zext nneg i32 %address.local590 to i64, !dbg !12
  %address.local592 = add nsw i64 %address.local588, %address.local591, !dbg !12
  %address.local593 = shl nsw i64 %address.local592, 2, !dbg !12
  %address.local594 = lshr i64 %address.local392, 4, !dbg !12
  %address.local595 = lshr i32 %address.thread328, 7, !dbg !12
  %address.local596 = add nuw nsw i32 %address.local595, 1, !dbg !12
  %address.local597 = zext nneg i32 %address.local596 to i64, !dbg !12
  %address.local598 = sub nsw i64 %address.local594, %address.local597, !dbg !12
  %address.local599 = ashr i64 %address.local598, 63, !dbg !12
  %address.local600 = xor i64 %address.local598, %address.local599, !dbg !12
  %address.local601 = sdiv i64 %address.local600, 2, !dbg !12
  %address.local602 = ashr i64 %address.local598, 63, !dbg !12
  %address.local603 = xor i64 %address.local601, %address.local602, !dbg !12
  %address.local604 = shl nsw i64 %address.local603, 3, !dbg !12
  %address.local605 = add i64 %address.local593, %address.local604, !dbg !12
  %address.local606 = sub i64 %address.local586, %address.local605, !dbg !12
  %address.local607 = shl i64 %address.local606, 7, !dbg !12
  %address.local608 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local609 = and i32 %address.local608, 2, !dbg !12
  %address.local610 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local611 = and i32 %address.local610, 16, !dbg !12
  %address.local612 = or disjoint i32 %address.local609, %address.local611, !dbg !12
  %address.local613 = or disjoint i32 %address.local612, 12, !dbg !12
  %address.local614 = zext nneg i32 %address.local613 to i64, !dbg !12
  %address.local615 = or disjoint i64 %address.local607, %address.local614, !dbg !12
  %address.local616 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local615, !dbg !12
  %dma.source.bits1270 = ptrtoint ptr addrspace(1) %address.local616 to i64, !dbg !12
  %dma.source.opaque1271 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1270) #10, !dbg !12
  %dma.source.ptr1272 = inttoptr i64 %dma.source.opaque1271 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1272, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local617 = lshr i64 %address.local392, 2, !dbg !12
  %address.local618 = or disjoint i64 %address.local617, 448, !dbg !12
  %address.local619 = and i64 %address.local618, 127, !dbg !12
  %address.local620 = lshr i64 %address.local619, 3, !dbg !12
  %address.local621 = mul nuw nsw i64 %address.local620, 7, !dbg !12
  %address.local622 = add nuw nsw i64 %address.local619, %address.local621, !dbg !12
  %address.local623 = and i64 %address.local618, 124, !dbg !12
  %address.local624 = add nuw nsw i64 %address.local622, %address.local623, !dbg !12
  %address.local625 = sub nuw nsw i64 %address.local619, %address.local620, !dbg !12
  %address.local626 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local627 = and i32 %address.local626, 4, !dbg !12
  %address.local628 = add nuw nsw i32 %address.local627, 12, !dbg !12
  %address.local629 = zext nneg i32 %address.local628 to i64, !dbg !12
  %address.local630 = add nuw nsw i64 %address.local625, %address.local629, !dbg !12
  %address.local631 = and i64 %address.local630, 252, !dbg !12
  %address.local632 = lshr i32 %address.thread328, 5, !dbg !12
  %address.local633 = and i32 %address.local632, 4, !dbg !12
  %address.local634 = add nuw nsw i32 %address.local633, 12, !dbg !12
  %address.local635 = zext nneg i32 %address.local634 to i64, !dbg !12
  %address.local636 = sub nuw nsw i64 %address.local618, %address.local635, !dbg !12
  %address.local637 = and i64 %address.local636, 120, !dbg !12
  %address.local638 = add nuw nsw i64 %address.local631, %address.local637, !dbg !12
  %address.local639 = sub nsw i64 %address.local624, %address.local638, !dbg !12
  %address.local640 = shl nsw i64 %address.local639, 7, !dbg !12
  %address.local641 = shl nuw nsw i32 %address.thread328, 1, !dbg !12
  %address.local642 = and i32 %address.local641, 2, !dbg !12
  %address.local643 = shl nuw nsw i32 %address.thread328, 3, !dbg !12
  %address.local644 = and i32 %address.local643, 16, !dbg !12
  %address.local645 = or disjoint i32 %address.local642, %address.local644, !dbg !12
  %address.local646 = or disjoint i32 %address.local645, 12, !dbg !12
  %address.local647 = zext nneg i32 %address.local646 to i64, !dbg !12
  %address.local648 = or disjoint i64 %address.local640, %address.local647, !dbg !12
  %address.local649 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local648, !dbg !12
  %dma.source.bits1273 = ptrtoint ptr addrspace(1) %address.local649 to i64, !dbg !12
  %dma.source.opaque1274 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1273) #10, !dbg !12
  %dma.source.ptr1275 = inttoptr i64 %dma.source.opaque1274 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1275, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local650 = and i32 %address.thread328, 192, !dbg !13
  %address.local651 = shl nuw nsw i32 %address.thread328, 2, !dbg !13
  %address.local652 = and i32 %address.local651, 60, !dbg !13
  %address.local653 = or disjoint i32 %address.local650, %address.local652, !dbg !13
  %address.local654 = shl nuw nsw i32 %address.thread328, 5, !dbg !13
  %address.local655 = and i32 %address.local654, 1536, !dbg !13
  %address.local656 = or disjoint i32 %address.local653, %address.local655, !dbg !13
  %address.local657 = zext nneg i32 %address.local656 to i64, !dbg !13
  %address.local658 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local657, !dbg !13
  %139 = load <4 x half>, ptr addrspace(3) %address.local658, align 2, !dbg !13
  %address.local659 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local657, !dbg !13
  %address.local660 = getelementptr i8, ptr addrspace(3) %address.local659, i64 4096, !dbg !13
  %140 = load <4 x half>, ptr addrspace(3) %address.local660, align 2, !dbg !13
  %address.local661 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local657, !dbg !13
  %address.local662 = getelementptr i8, ptr addrspace(3) %address.local661, i64 512, !dbg !13
  %141 = load <4 x half>, ptr addrspace(3) %address.local662, align 2, !dbg !13
  %142 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !13
  %address.local663 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local657, !dbg !13
  %address.local664 = getelementptr i8, ptr addrspace(3) %address.local663, i64 4608, !dbg !13
  %143 = load <4 x half>, ptr addrspace(3) %address.local664, align 2, !dbg !13
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !13
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !13
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local665 = shl i64 %address.base, 6, !dbg !16
  %address.base666 = call i64 asm sideeffect "", "=v,0"(i64 %.neg179) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local667 = shl i64 %address.base666, 5, !dbg !16
  %address.local668 = add i64 %address.local665, %address.local667, !dbg !16
  %address.local669 = mul nuw nsw i64 %address.local392, 400, !dbg !16
  %address.local670 = lshr i64 %address.local392, 2, !dbg !16
  %address.local671 = mul nsw i64 %address.local670, -1664, !dbg !16
  %address.local672 = add nsw i64 %address.local669, %address.local671, !dbg !16
  %address.local673 = lshr i64 %address.local392, 2, !dbg !16
  %address.local674 = and i64 %address.local673, 3, !dbg !16
  %address.local675 = shl nuw nsw i64 %address.local674, 3, !dbg !16
  %address.local676 = add nsw i64 %address.local672, %address.local675, !dbg !16
  %address.local677 = lshr i64 %address.local392, 1, !dbg !16
  %address.local678 = shl nuw nsw i64 %address.local677, 5, !dbg !16
  %address.local679 = add nsw i64 %address.local676, %address.local678, !dbg !16
  %address.local680 = and i64 %address.local392, 63, !dbg !13
  %147 = shl nuw nsw i64 %address.local392, 6, !dbg !16
  %address.local682 = and i64 %147, 3072, !dbg !16
  %address.local683 = add nsw i64 %address.local679, %address.local682, !dbg !16
  %address.local684 = and i64 %address.local392, 32, !dbg !16
  %address.local685 = add i64 %address.local683, %address.local684, !dbg !16
  %address.local686 = add i64 %address.local668, %address.local685, !dbg !16
  %address.local687 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local686, !dbg !16
  %148 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %149 = shufflevector <8 x half> %148, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %151 = shufflevector <8 x half> %148, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %151, <4 x float> %150) #9, !dbg !16
  %address.base688 = call i64 asm sideeffect "", "=v,0"(i64 %.neg182) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local689 = shl i64 %address.base688, 7, !dbg !16
  %address.local690 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local689, !dbg !16
  %address.base691 = call i64 asm sideeffect "", "=v,0"(i64 %.neg184) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local692 = shl i64 %address.base691, 6, !dbg !16
  %address.local693 = getelementptr i8, ptr addrspace(3) %address.local690, i64 %address.local692, !dbg !16
  %address.local694 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local695 = and i32 %address.local694, 6, !dbg !16
  %narrow = mul nuw nsw i32 %address.local695, 144, !dbg !16
  %address.local697 = zext nneg i32 %narrow to i64, !dbg !16
  %address.local698 = getelementptr i8, ptr addrspace(3) %address.local693, i64 %address.local697, !dbg !16
  %address.local699 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local700 = and i32 %address.local699, 24, !dbg !16
  %address.local701 = shl nuw nsw i32 %address.thread328, 8, !dbg !16
  %address.local702 = and i32 %address.local701, 768, !dbg !16
  %address.local703 = or disjoint i32 %address.local700, %address.local702, !dbg !16
  %address.local704 = zext nneg i32 %address.local703 to i64, !dbg !16
  %address.local705 = getelementptr [2 x i8], ptr addrspace(3) %address.local698, i64 %address.local704, !dbg !16
  %address.local706 = shl nuw nsw i32 %address.thread328, 4, !dbg !16
  %address.local707 = and i32 %address.local706, 32, !dbg !16
  %address.local708 = zext nneg i32 %address.local707 to i64, !dbg !16
  %address.local709 = getelementptr [2 x i8], ptr addrspace(3) %address.local705, i64 %address.local708, !dbg !16
  %address.local710 = getelementptr [2 x i8], ptr addrspace(3) %address.local709, i64 %address.local682, !dbg !16
  %address.local711 = getelementptr [2 x i8], ptr addrspace(3) %address.local710, i64 %address.local684, !dbg !16
  %address.local712 = getelementptr i8, ptr addrspace(3) %address.local711, i64 64, !dbg !16
  %153 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %address.local713 = shl nuw nsw i64 %address.local392, 2, !dbg !13
  %address.local714 = mul nsw i64 %address.local670, -18, !dbg !16
  %address.local715 = add nsw i64 %address.local713, %address.local714, !dbg !16
  %address.local716 = add nsw i64 %address.local715, %address.local677, !dbg !16
  %address.local717 = lshr i64 %address.local680, 5, !dbg !16
  %address.local718 = add nsw i64 %address.local716, %address.local717, !dbg !16
  %address.local719 = sdiv i64 %address.local718, -2, !dbg !16
  %address.local720 = shl i64 %address.local719, 7, !dbg !16
  %address.local721 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local720, !dbg !16
  %address.local722 = mul nuw nsw i64 %address.local392, 18, !dbg !16
  %address.local723 = mul nsw i64 %address.local670, -72, !dbg !16
  %address.local724 = add nsw i64 %address.local722, %address.local723, !dbg !16
  %address.local725 = add nsw i64 %address.local724, %address.local674, !dbg !16
  %address.local726 = add nsw i64 %address.local725, 1, !dbg !16
  %address.local727 = sdiv i64 %address.local726, -4, !dbg !16
  %address.local728 = shl i64 %address.local727, 6, !dbg !16
  %address.local729 = getelementptr i8, ptr addrspace(3) %address.local721, i64 %address.local728, !dbg !16
  %address.local730 = getelementptr [2 x i8], ptr addrspace(3) %address.local729, i64 %address.local669, !dbg !16
  %address.local731 = getelementptr [2 x i8], ptr addrspace(3) %address.local730, i64 %address.local671, !dbg !16
  %address.local732 = getelementptr [2 x i8], ptr addrspace(3) %address.local731, i64 %address.local675, !dbg !16
  %address.local733 = getelementptr [2 x i8], ptr addrspace(3) %address.local732, i64 %address.local678, !dbg !16
  %address.local734 = getelementptr [2 x i8], ptr addrspace(3) %address.local733, i64 %address.local682, !dbg !16
  %address.local735 = getelementptr [2 x i8], ptr addrspace(3) %address.local734, i64 %address.local684, !dbg !16
  %address.local736 = getelementptr i8, ptr addrspace(3) %address.local735, i64 144, !dbg !16
  %154 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %155 = shl nuw nsw i64 %address.local670, 4, !dbg !16
  %address.local738 = sub nsw i64 %address.local713, %155, !dbg !16
  %address.local739 = lshr i32 %address.thread328, 1, !dbg !16
  %address.local740 = and i32 %address.local739, 1, !dbg !16
  %address.local741 = zext nneg i32 %address.local740 to i64, !dbg !16
  %address.local742 = or disjoint i64 %address.local738, %address.local741, !dbg !16
  %address.local743 = add nuw nsw i64 %address.local742, %address.local717, !dbg !16
  %address.local744 = add nsw i64 %address.local743, 3, !dbg !16
  %address.local745 = sdiv i64 %address.local744, -2, !dbg !16
  %address.local746 = shl i64 %address.local745, 7, !dbg !16
  %address.local747 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local746, !dbg !16
  %address.local748 = shl nuw nsw i64 %address.local392, 4, !dbg !16
  %156 = shl nuw nsw i64 %address.local670, 6, !dbg !16
  %address.local750 = sub nsw i64 %address.local748, %156, !dbg !16
  %address.local751 = or disjoint i64 %address.local750, %address.local674, !dbg !16
  %address.local752 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local753 = and i32 %address.local752, 6, !dbg !16
  %address.local754 = or disjoint i32 %address.local753, 1, !dbg !16
  %address.local755 = zext nneg i32 %address.local754 to i64, !dbg !16
  %address.local756 = add nuw nsw i64 %address.local751, %address.local755, !dbg !16
  %address.local757 = sdiv i64 %address.local756, -4, !dbg !16
  %address.local758 = shl i64 %address.local757, 6, !dbg !16
  %address.local759 = getelementptr i8, ptr addrspace(3) %address.local747, i64 %address.local758, !dbg !16
  %address.local760 = mul nuw nsw i64 %address.local755, 144, !dbg !16
  %address.local761 = getelementptr i8, ptr addrspace(3) %address.local759, i64 %address.local760, !dbg !16
  %address.local762 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local763 = and i32 %address.local762, 24, !dbg !16
  %address.local764 = shl nuw nsw i32 %address.thread328, 8, !dbg !16
  %address.local765 = and i32 %address.local764, 768, !dbg !16
  %address.local766 = or disjoint i32 %address.local763, %address.local765, !dbg !16
  %address.local767 = zext nneg i32 %address.local766 to i64, !dbg !16
  %address.local768 = getelementptr [2 x i8], ptr addrspace(3) %address.local761, i64 %address.local767, !dbg !16
  %address.local769 = shl nuw nsw i32 %address.thread328, 4, !dbg !16
  %address.local770 = and i32 %address.local769, 32, !dbg !16
  %address.local771 = zext nneg i32 %address.local770 to i64, !dbg !16
  %address.local772 = getelementptr [2 x i8], ptr addrspace(3) %address.local768, i64 %address.local771, !dbg !16
  %address.local773 = getelementptr [2 x i8], ptr addrspace(3) %address.local772, i64 %address.local682, !dbg !16
  %address.local774 = getelementptr [2 x i8], ptr addrspace(3) %address.local773, i64 %address.local684, !dbg !16
  %address.local775 = getelementptr i8, ptr addrspace(3) %address.local774, i64 192, !dbg !16
  %157 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %158 = shufflevector <8 x half> %153, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !16
  %160 = shufflevector <8 x half> %153, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %160, <4 x float> %159) #9, !dbg !16
  %162 = shufflevector <8 x half> %154, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %163 = shufflevector <8 x half> %154, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !16
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %163, <4 x float> %164) #9, !dbg !16
  %166 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %167 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %166, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local776 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local686, !dbg !16
  %address.local777 = getelementptr i8, ptr addrspace(3) %address.local776, i64 1024, !dbg !16
  %169 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %167, <4 x float> %168) #9, !dbg !16
  %address.local778 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local779 = and i32 %address.local778, 6, !dbg !16
  %address.local780 = or disjoint i32 %address.local779, 8, !dbg !16
  %address.local781 = mul nuw nsw i32 %address.local780, 72, !dbg !16
  %address.local782 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local783 = and i32 %address.local782, 24, !dbg !16
  %address.local784 = shl nuw nsw i32 %address.thread328, 8, !dbg !16
  %address.local785 = and i32 %address.local784, 768, !dbg !16
  %address.local786 = or disjoint i32 %address.local783, %address.local785, !dbg !16
  %address.local787 = or disjoint i32 %address.local786, 1024, !dbg !16
  %address.local788 = add nuw nsw i32 %address.local781, %address.local787, !dbg !16
  %address.local789 = shl nuw nsw i32 %address.thread328, 4, !dbg !16
  %address.local790 = and i32 %address.local789, 32, !dbg !16
  %address.local791 = or disjoint i32 %address.local790, 64, !dbg !16
  %address.local792 = add nuw nsw i32 %address.local788, %address.local791, !dbg !16
  %address.local793 = zext nneg i32 %address.local792 to i64, !dbg !16
  %address.local794 = add nuw nsw i64 %address.local682, %address.local793, !dbg !16
  %address.local795 = add nuw nsw i64 %address.local794, %address.local684, !dbg !16
  %address.local796 = shl nuw nsw i32 %address.thread328, 2, !dbg !16
  %address.local797 = and i32 %address.local796, 12, !dbg !16
  %address.local798 = lshr i32 %address.thread328, 1, !dbg !16
  %address.local799 = and i32 %address.local798, 1, !dbg !16
  %address.local800 = or disjoint i32 %address.local797, %address.local799, !dbg !16
  %address.local801 = or disjoint i32 %address.local800, 18, !dbg !16
  %address.local802 = zext nneg i32 %address.local801 to i64, !dbg !16
  %address.local803 = add nuw nsw i64 %address.local717, %address.local802, !dbg !16
  %address.local804 = shl nuw nsw i64 %address.local803, 5, !dbg !16
  %address.local805 = add nuw nsw i64 %address.local804, 32, !dbg !16
  %address.local806 = and i64 %address.local805, 274877906880, !dbg !16
  %address.local807 = lshr i32 %address.thread328, 2, !dbg !16
  %address.local808 = and i32 %address.local807, 2, !dbg !16
  %address.local809 = shl nuw nsw i32 %address.thread328, 4, !dbg !16
  %address.local810 = and i32 %address.local809, 48, !dbg !16
  %address.local811 = or disjoint i32 %address.local808, %address.local810, !dbg !16
  %address.local812 = or disjoint i32 %address.local811, 64, !dbg !16
  %address.local813 = add nuw nsw i32 %address.local812, %address.local780, !dbg !16
  %address.local814 = shl nuw nsw i32 %address.local813, 3, !dbg !16
  %address.local815 = and i32 %address.local814, 2016, !dbg !16
  %address.local816 = zext nneg i32 %address.local815 to i64, !dbg !16
  %address.local817 = add nuw nsw i64 %address.local806, %address.local816, !dbg !16
  %address.local818 = sub nsw i64 %address.local795, %address.local817, !dbg !16
  %address.local819 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local818, !dbg !16
  %address.local820 = getelementptr i8, ptr addrspace(3) %address.local819, i64 64, !dbg !16
  %171 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %address.local821 = add nuw nsw i64 %address.local713, 18, !dbg !16
  %address.local822 = add nsw i64 %address.local821, %address.local714, !dbg !16
  %address.local823 = add nsw i64 %address.local822, %address.local677, !dbg !16
  %address.local824 = add nsw i64 %address.local823, %address.local717, !dbg !16
  %address.local825 = sdiv i64 %address.local824, -2, !dbg !16
  %address.local826 = shl i64 %address.local825, 7, !dbg !16
  %address.local827 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local826, !dbg !16
  %address.local828 = add nuw nsw i64 %address.local722, 72, !dbg !16
  %address.local829 = add nsw i64 %address.local828, %address.local723, !dbg !16
  %address.local830 = add nsw i64 %address.local829, %address.local674, !dbg !16
  %address.local831 = add nsw i64 %address.local830, -15, !dbg !16
  %address.local832 = sdiv i64 %address.local831, -4, !dbg !16
  %address.local833 = shl i64 %address.local832, 6, !dbg !16
  %address.local834 = getelementptr i8, ptr addrspace(3) %address.local827, i64 %address.local833, !dbg !16
  %address.local835 = getelementptr [2 x i8], ptr addrspace(3) %address.local834, i64 %address.local669, !dbg !16
  %address.local836 = getelementptr [2 x i8], ptr addrspace(3) %address.local835, i64 %address.local671, !dbg !16
  %address.local837 = getelementptr [2 x i8], ptr addrspace(3) %address.local836, i64 %address.local675, !dbg !16
  %address.local838 = getelementptr [2 x i8], ptr addrspace(3) %address.local837, i64 %address.local678, !dbg !16
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) %address.local838, i64 %address.local682, !dbg !16
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) %address.local839, i64 %address.local684, !dbg !16
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local840, i64 3216, !dbg !16
  %172 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %173 = shufflevector <8 x half> %169, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %173, <4 x float> zeroinitializer) #9, !dbg !16
  %175 = shufflevector <8 x half> %169, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %175, <4 x float> %174) #9, !dbg !16
  %177 = shufflevector <8 x half> %171, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %178 = shufflevector <8 x half> %171, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %177, <4 x float> zeroinitializer) #9, !dbg !16
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %178, <4 x float> %179) #9, !dbg !16
  %181 = shufflevector <8 x half> %172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %182 = shufflevector <8 x half> %172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %181, <4 x float> zeroinitializer) #9, !dbg !16
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %182, <4 x float> %183) #9, !dbg !16
  %address.local842 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local843 = and i32 %address.local842, 6, !dbg !16
  %address.local844 = or disjoint i32 %address.local843, 9, !dbg !16
  %address.local845 = mul nuw nsw i32 %address.local844, 72, !dbg !16
  %address.local846 = shl nuw nsw i32 %address.thread328, 1, !dbg !16
  %address.local847 = and i32 %address.local846, 24, !dbg !16
  %address.local848 = shl nuw nsw i32 %address.thread328, 8, !dbg !16
  %address.local849 = and i32 %address.local848, 768, !dbg !16
  %address.local850 = or disjoint i32 %address.local847, %address.local849, !dbg !16
  %address.local851 = or disjoint i32 %address.local850, 1024, !dbg !16
  %address.local852 = add nuw nsw i32 %address.local845, %address.local851, !dbg !16
  %address.local853 = shl nuw nsw i32 %address.thread328, 4, !dbg !16
  %address.local854 = and i32 %address.local853, 32, !dbg !16
  %address.local855 = or disjoint i32 %address.local854, 64, !dbg !16
  %address.local856 = add nuw nsw i32 %address.local852, %address.local855, !dbg !16
  %address.local857 = zext nneg i32 %address.local856 to i64, !dbg !16
  %address.local858 = add nuw nsw i64 %address.local682, %address.local857, !dbg !16
  %address.local859 = add nuw nsw i64 %address.local858, %address.local684, !dbg !16
  %address.local860 = shl nuw nsw i32 %address.thread328, 2, !dbg !16
  %address.local861 = and i32 %address.local860, 12, !dbg !16
  %address.local862 = lshr i32 %address.thread328, 1, !dbg !16
  %address.local863 = and i32 %address.local862, 1, !dbg !16
  %address.local864 = or disjoint i32 %address.local861, %address.local863, !dbg !16
  %address.local865 = or disjoint i32 %address.local864, 18, !dbg !16
  %address.local866 = zext nneg i32 %address.local865 to i64, !dbg !16
  %address.local867 = add nuw nsw i64 %address.local717, %address.local866, !dbg !16
  %address.local868 = shl nuw nsw i64 %address.local867, 5, !dbg !16
  %address.local869 = add nuw nsw i64 %address.local868, 96, !dbg !16
  %address.local870 = and i64 %address.local869, 274877906880, !dbg !16
  %address.local871 = lshr i32 %address.thread328, 2, !dbg !16
  %address.local872 = and i32 %address.local871, 3, !dbg !16
  %address.local873 = shl nuw nsw i32 %address.thread328, 4, !dbg !16
  %address.local874 = and i32 %address.local873, 48, !dbg !16
  %address.local875 = or disjoint i32 %address.local872, %address.local874, !dbg !16
  %address.local876 = or disjoint i32 %address.local875, 64, !dbg !16
  %address.local877 = add nuw nsw i32 %address.local876, %address.local844, !dbg !16
  %address.local878 = shl nuw nsw i32 %address.local877, 3, !dbg !16
  %address.local879 = and i32 %address.local878, 2016, !dbg !16
  %address.local880 = zext nneg i32 %address.local879 to i64, !dbg !16
  %address.local881 = add nuw nsw i64 %address.local870, %address.local880, !dbg !16
  %address.local882 = sub nsw i64 %address.local859, %address.local881, !dbg !16
  %address.local883 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local882, !dbg !16
  %address.local884 = getelementptr i8, ptr addrspace(3) %address.local883, i64 192, !dbg !16
  %185 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %186 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %187 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !16
  %189 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %189, <4 x float> %188) #9, !dbg !16
  %191 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %192 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !16
  %194 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %192, <4 x float> %193) #9, !dbg !16
  %196 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %197 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %198 = shufflevector <8 x half> %194, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %198, <4 x float> zeroinitializer) #9, !dbg !16
  %200 = shufflevector <8 x half> %194, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %200, <4 x float> %199) #9, !dbg !16
  %202 = shufflevector <8 x half> %196, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %203 = shufflevector <8 x half> %196, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %202, <4 x float> zeroinitializer) #9, !dbg !16
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %203, <4 x float> %204) #9, !dbg !16
  %206 = shufflevector <8 x half> %197, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %207 = shufflevector <8 x half> %197, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %208 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %206, <4 x float> zeroinitializer) #9, !dbg !16
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %207, <4 x float> %208) #9, !dbg !16
  %210 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %211 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %212 = shufflevector <8 x half> %210, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %212, <4 x float> zeroinitializer) #9, !dbg !16
  %214 = shufflevector <8 x half> %210, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %214, <4 x float> %213) #9, !dbg !16
  %216 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %217 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %216, <4 x float> zeroinitializer) #9, !dbg !16
  %219 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %217, <4 x float> %218) #9, !dbg !16
  %221 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %222 = shufflevector <8 x half> %219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %222, <4 x float> zeroinitializer) #9, !dbg !16
  %224 = shufflevector <8 x half> %219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %224, <4 x float> %223) #9, !dbg !16
  %226 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %227 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %226, <4 x float> zeroinitializer) #9, !dbg !16
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %227, <4 x float> %228) #9, !dbg !16
  %address.local885 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local334, !dbg !11
  %dma.source.bits1276 = ptrtoint ptr addrspace(1) %address.local885 to i64, !dbg !11
  %dma.source.opaque1277 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1276) #10, !dbg !11
  %dma.source.ptr1278 = inttoptr i64 %dma.source.opaque1277 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1278, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local886 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local342, !dbg !11
  %dma.source.bits1279 = ptrtoint ptr addrspace(1) %address.local886 to i64, !dbg !11
  %dma.source.opaque1280 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1279) #10, !dbg !11
  %dma.source.ptr1281 = inttoptr i64 %dma.source.opaque1280 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1281, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local887 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local350, !dbg !11
  %dma.source.bits1282 = ptrtoint ptr addrspace(1) %address.local887 to i64, !dbg !11
  %dma.source.opaque1283 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1282) #10, !dbg !11
  %dma.source.ptr1284 = inttoptr i64 %dma.source.opaque1283 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1284, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local888 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local358, !dbg !11
  %dma.source.bits1285 = ptrtoint ptr addrspace(1) %address.local888 to i64, !dbg !11
  %dma.source.opaque1286 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1285) #10, !dbg !11
  %dma.source.ptr1287 = inttoptr i64 %dma.source.opaque1286 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1287, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local889 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local366, !dbg !11
  %dma.source.bits1288 = ptrtoint ptr addrspace(1) %address.local889 to i64, !dbg !11
  %dma.source.opaque1289 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1288) #10, !dbg !11
  %dma.source.ptr1290 = inttoptr i64 %dma.source.opaque1289 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1290, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local890 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local374, !dbg !11
  %dma.source.bits1291 = ptrtoint ptr addrspace(1) %address.local890 to i64, !dbg !11
  %dma.source.opaque1292 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1291) #10, !dbg !11
  %dma.source.ptr1293 = inttoptr i64 %dma.source.opaque1292 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1293, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local891 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local382, !dbg !11
  %dma.source.bits1294 = ptrtoint ptr addrspace(1) %address.local891 to i64, !dbg !11
  %dma.source.opaque1295 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1294) #10, !dbg !11
  %dma.source.ptr1296 = inttoptr i64 %dma.source.opaque1295 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1296, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local892 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local390, !dbg !11
  %dma.source.bits1297 = ptrtoint ptr addrspace(1) %address.local892 to i64, !dbg !11
  %dma.source.opaque1298 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1297) #10, !dbg !11
  %dma.source.ptr1299 = inttoptr i64 %dma.source.opaque1298 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1299, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %230 = or disjoint i64 %137, 32, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %231 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %230, !dbg !12
  %address.local893 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local417, !dbg !12
  %dma.source.bits1300 = ptrtoint ptr addrspace(1) %address.local893 to i64, !dbg !12
  %dma.source.opaque1301 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1300) #10, !dbg !12
  %dma.source.ptr1302 = inttoptr i64 %dma.source.opaque1301 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1302, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local894 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local449, !dbg !12
  %dma.source.bits1303 = ptrtoint ptr addrspace(1) %address.local894 to i64, !dbg !12
  %dma.source.opaque1304 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1303) #10, !dbg !12
  %dma.source.ptr1305 = inttoptr i64 %dma.source.opaque1304 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1305, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local895 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local479, !dbg !12
  %dma.source.bits1306 = ptrtoint ptr addrspace(1) %address.local895 to i64, !dbg !12
  %dma.source.opaque1307 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1306) #10, !dbg !12
  %dma.source.ptr1308 = inttoptr i64 %dma.source.opaque1307 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1308, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local896 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local512, !dbg !12
  %dma.source.bits1309 = ptrtoint ptr addrspace(1) %address.local896 to i64, !dbg !12
  %dma.source.opaque1310 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1309) #10, !dbg !12
  %dma.source.ptr1311 = inttoptr i64 %dma.source.opaque1310 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1311, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local897 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local546, !dbg !12
  %dma.source.bits1312 = ptrtoint ptr addrspace(1) %address.local897 to i64, !dbg !12
  %dma.source.opaque1313 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1312) #10, !dbg !12
  %dma.source.ptr1314 = inttoptr i64 %dma.source.opaque1313 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1314, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local898 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local579, !dbg !12
  %dma.source.bits1315 = ptrtoint ptr addrspace(1) %address.local898 to i64, !dbg !12
  %dma.source.opaque1316 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1315) #10, !dbg !12
  %dma.source.ptr1317 = inttoptr i64 %dma.source.opaque1316 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1317, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local899 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local615, !dbg !12
  %dma.source.bits1318 = ptrtoint ptr addrspace(1) %address.local899 to i64, !dbg !12
  %dma.source.opaque1319 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1318) #10, !dbg !12
  %dma.source.ptr1320 = inttoptr i64 %dma.source.opaque1319 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1320, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local900 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local648, !dbg !12
  %dma.source.bits1321 = ptrtoint ptr addrspace(1) %address.local900 to i64, !dbg !12
  %dma.source.opaque1322 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1321) #10, !dbg !12
  %dma.source.ptr1323 = inttoptr i64 %dma.source.opaque1322 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1323, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %232 = load <4 x half>, ptr addrspace(3) %address.local658, align 2, !dbg !13
  %233 = load <4 x half>, ptr addrspace(3) %address.local660, align 2, !dbg !13
  %234 = load <4 x half>, ptr addrspace(3) %address.local662, align 2, !dbg !13
  %235 = fmul <4 x half> %232, splat (half 0xH3015), !dbg !13
  %236 = load <4 x half>, ptr addrspace(3) %address.local664, align 2, !dbg !13
  %237 = fmul <4 x half> %233, splat (half 0xH3015), !dbg !13
  %238 = fmul <4 x half> %234, splat (half 0xH3015), !dbg !13
  %239 = fmul <4 x half> %236, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %240 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %241 = shufflevector <8 x half> %240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %241, <4 x float> %152) #9, !dbg !16
  %243 = shufflevector <8 x half> %240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %243, <4 x float> %242) #9, !dbg !16
  %245 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %246 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %247 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %248 = shufflevector <8 x half> %245, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %248, <4 x float> %161) #9, !dbg !16
  %250 = shufflevector <8 x half> %245, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %250, <4 x float> %249) #9, !dbg !16
  %252 = shufflevector <8 x half> %246, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %253 = shufflevector <8 x half> %246, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %254 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %252, <4 x float> %165) #9, !dbg !16
  %255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %253, <4 x float> %254) #9, !dbg !16
  %256 = shufflevector <8 x half> %247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %257 = shufflevector <8 x half> %247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %256, <4 x float> %170) #9, !dbg !16
  %259 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %257, <4 x float> %258) #9, !dbg !16
  %261 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %262 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %263 = shufflevector <8 x half> %259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %264 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %263, <4 x float> %176) #9, !dbg !16
  %265 = shufflevector <8 x half> %259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %265, <4 x float> %264) #9, !dbg !16
  %267 = shufflevector <8 x half> %261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %268 = shufflevector <8 x half> %261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %267, <4 x float> %180) #9, !dbg !16
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %268, <4 x float> %269) #9, !dbg !16
  %271 = shufflevector <8 x half> %262, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %272 = shufflevector <8 x half> %262, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %271, <4 x float> %184) #9, !dbg !16
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %272, <4 x float> %273) #9, !dbg !16
  %275 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %276 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %277 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %277, <4 x float> %190) #9, !dbg !16
  %279 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %279, <4 x float> %278) #9, !dbg !16
  %281 = shufflevector <8 x half> %276, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %282 = shufflevector <8 x half> %276, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %281, <4 x float> %195) #9, !dbg !16
  %284 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %282, <4 x float> %283) #9, !dbg !16
  %286 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %287 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %288 = shufflevector <8 x half> %284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %288, <4 x float> %201) #9, !dbg !16
  %290 = shufflevector <8 x half> %284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %290, <4 x float> %289) #9, !dbg !16
  %292 = shufflevector <8 x half> %286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %293 = shufflevector <8 x half> %286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %292, <4 x float> %205) #9, !dbg !16
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %293, <4 x float> %294) #9, !dbg !16
  %296 = shufflevector <8 x half> %287, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %297 = shufflevector <8 x half> %287, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %296, <4 x float> %209) #9, !dbg !16
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %297, <4 x float> %298) #9, !dbg !16
  %300 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %301 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %302 = shufflevector <8 x half> %300, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %302, <4 x float> %215) #9, !dbg !16
  %304 = shufflevector <8 x half> %300, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %304, <4 x float> %303) #9, !dbg !16
  %306 = shufflevector <8 x half> %301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %307 = shufflevector <8 x half> %301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %306, <4 x float> %220) #9, !dbg !16
  %309 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %307, <4 x float> %308) #9, !dbg !16
  %311 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %312 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %312, <4 x float> %225) #9, !dbg !16
  %314 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %314, <4 x float> %313) #9, !dbg !16
  %316 = shufflevector <8 x half> %311, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %317 = shufflevector <8 x half> %311, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %316, <4 x float> %229) #9, !dbg !16
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %317, <4 x float> %318) #9, !dbg !16
  %address.local901 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local334, !dbg !11
  %dma.source.bits1324 = ptrtoint ptr addrspace(1) %address.local901 to i64, !dbg !11
  %dma.source.opaque1325 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1324) #10, !dbg !11
  %dma.source.ptr1326 = inttoptr i64 %dma.source.opaque1325 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1326, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local902 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local342, !dbg !11
  %dma.source.bits1327 = ptrtoint ptr addrspace(1) %address.local902 to i64, !dbg !11
  %dma.source.opaque1328 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1327) #10, !dbg !11
  %dma.source.ptr1329 = inttoptr i64 %dma.source.opaque1328 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1329, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local903 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local350, !dbg !11
  %dma.source.bits1330 = ptrtoint ptr addrspace(1) %address.local903 to i64, !dbg !11
  %dma.source.opaque1331 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1330) #10, !dbg !11
  %dma.source.ptr1332 = inttoptr i64 %dma.source.opaque1331 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1332, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local904 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local358, !dbg !11
  %dma.source.bits1333 = ptrtoint ptr addrspace(1) %address.local904 to i64, !dbg !11
  %dma.source.opaque1334 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1333) #10, !dbg !11
  %dma.source.ptr1335 = inttoptr i64 %dma.source.opaque1334 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1335, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local905 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local366, !dbg !11
  %dma.source.bits1336 = ptrtoint ptr addrspace(1) %address.local905 to i64, !dbg !11
  %dma.source.opaque1337 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1336) #10, !dbg !11
  %dma.source.ptr1338 = inttoptr i64 %dma.source.opaque1337 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1338, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local906 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local374, !dbg !11
  %dma.source.bits1339 = ptrtoint ptr addrspace(1) %address.local906 to i64, !dbg !11
  %dma.source.opaque1340 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1339) #10, !dbg !11
  %dma.source.ptr1341 = inttoptr i64 %dma.source.opaque1340 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1341, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local907 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local382, !dbg !11
  %dma.source.bits1342 = ptrtoint ptr addrspace(1) %address.local907 to i64, !dbg !11
  %dma.source.opaque1343 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1342) #10, !dbg !11
  %dma.source.ptr1344 = inttoptr i64 %dma.source.opaque1343 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1344, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local908 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local390, !dbg !11
  %dma.source.bits1345 = ptrtoint ptr addrspace(1) %address.local908 to i64, !dbg !11
  %dma.source.opaque1346 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1345) #10, !dbg !11
  %dma.source.ptr1347 = inttoptr i64 %dma.source.opaque1346 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1347, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %320 = or disjoint i64 %137, 64, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %321 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %320, !dbg !12
  %address.local909 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local417, !dbg !12
  %dma.source.bits1348 = ptrtoint ptr addrspace(1) %address.local909 to i64, !dbg !12
  %dma.source.opaque1349 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1348) #10, !dbg !12
  %dma.source.ptr1350 = inttoptr i64 %dma.source.opaque1349 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1350, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local910 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local449, !dbg !12
  %dma.source.bits1351 = ptrtoint ptr addrspace(1) %address.local910 to i64, !dbg !12
  %dma.source.opaque1352 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1351) #10, !dbg !12
  %dma.source.ptr1353 = inttoptr i64 %dma.source.opaque1352 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1353, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local911 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local479, !dbg !12
  %dma.source.bits1354 = ptrtoint ptr addrspace(1) %address.local911 to i64, !dbg !12
  %dma.source.opaque1355 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1354) #10, !dbg !12
  %dma.source.ptr1356 = inttoptr i64 %dma.source.opaque1355 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1356, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local912 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local512, !dbg !12
  %dma.source.bits1357 = ptrtoint ptr addrspace(1) %address.local912 to i64, !dbg !12
  %dma.source.opaque1358 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1357) #10, !dbg !12
  %dma.source.ptr1359 = inttoptr i64 %dma.source.opaque1358 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1359, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local913 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local546, !dbg !12
  %dma.source.bits1360 = ptrtoint ptr addrspace(1) %address.local913 to i64, !dbg !12
  %dma.source.opaque1361 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1360) #10, !dbg !12
  %dma.source.ptr1362 = inttoptr i64 %dma.source.opaque1361 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1362, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local914 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local579, !dbg !12
  %dma.source.bits1363 = ptrtoint ptr addrspace(1) %address.local914 to i64, !dbg !12
  %dma.source.opaque1364 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1363) #10, !dbg !12
  %dma.source.ptr1365 = inttoptr i64 %dma.source.opaque1364 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1365, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local915 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local615, !dbg !12
  %dma.source.bits1366 = ptrtoint ptr addrspace(1) %address.local915 to i64, !dbg !12
  %dma.source.opaque1367 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1366) #10, !dbg !12
  %dma.source.ptr1368 = inttoptr i64 %dma.source.opaque1367 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1368, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local916 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local648, !dbg !12
  %dma.source.bits1369 = ptrtoint ptr addrspace(1) %address.local916 to i64, !dbg !12
  %dma.source.opaque1370 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1369) #10, !dbg !12
  %dma.source.ptr1371 = inttoptr i64 %dma.source.opaque1370 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1371, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %address.local658, align 2, !dbg !13
  %323 = load <4 x half>, ptr addrspace(3) %address.local660, align 2, !dbg !13
  %324 = load <4 x half>, ptr addrspace(3) %address.local662, align 2, !dbg !13
  %325 = fmul <4 x half> %322, splat (half 0xH3015), !dbg !13
  %326 = load <4 x half>, ptr addrspace(3) %address.local664, align 2, !dbg !13
  %327 = fmul <4 x half> %323, splat (half 0xH3015), !dbg !13
  %328 = fmul <4 x half> %324, splat (half 0xH3015), !dbg !13
  %329 = fmul <4 x half> %326, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %330 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %331 = shufflevector <8 x half> %330, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %331, <4 x float> %244) #9, !dbg !16
  %333 = shufflevector <8 x half> %330, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %333, <4 x float> %332) #9, !dbg !16
  %335 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %336 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %337 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %338 = shufflevector <8 x half> %335, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %338, <4 x float> %251) #9, !dbg !16
  %340 = shufflevector <8 x half> %335, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %340, <4 x float> %339) #9, !dbg !16
  %342 = shufflevector <8 x half> %336, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %343 = shufflevector <8 x half> %336, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %342, <4 x float> %255) #9, !dbg !16
  %345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %343, <4 x float> %344) #9, !dbg !16
  %346 = shufflevector <8 x half> %337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %347 = shufflevector <8 x half> %337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %346, <4 x float> %260) #9, !dbg !16
  %349 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %347, <4 x float> %348) #9, !dbg !16
  %351 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %352 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %353 = shufflevector <8 x half> %349, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %354 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %353, <4 x float> %266) #9, !dbg !16
  %355 = shufflevector <8 x half> %349, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %355, <4 x float> %354) #9, !dbg !16
  %357 = shufflevector <8 x half> %351, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %358 = shufflevector <8 x half> %351, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %357, <4 x float> %270) #9, !dbg !16
  %360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %358, <4 x float> %359) #9, !dbg !16
  %361 = shufflevector <8 x half> %352, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %362 = shufflevector <8 x half> %352, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %361, <4 x float> %274) #9, !dbg !16
  %364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %362, <4 x float> %363) #9, !dbg !16
  %365 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %366 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %367 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %367, <4 x float> %280) #9, !dbg !16
  %369 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %369, <4 x float> %368) #9, !dbg !16
  %371 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %372 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %371, <4 x float> %285) #9, !dbg !16
  %374 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %372, <4 x float> %373) #9, !dbg !16
  %376 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %377 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %378 = shufflevector <8 x half> %374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %379 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %378, <4 x float> %291) #9, !dbg !16
  %380 = shufflevector <8 x half> %374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %380, <4 x float> %379) #9, !dbg !16
  %382 = shufflevector <8 x half> %376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %383 = shufflevector <8 x half> %376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %382, <4 x float> %295) #9, !dbg !16
  %385 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %383, <4 x float> %384) #9, !dbg !16
  %386 = shufflevector <8 x half> %377, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %387 = shufflevector <8 x half> %377, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %386, <4 x float> %299) #9, !dbg !16
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %387, <4 x float> %388) #9, !dbg !16
  %390 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %391 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %392 = shufflevector <8 x half> %390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %392, <4 x float> %305) #9, !dbg !16
  %394 = shufflevector <8 x half> %390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %394, <4 x float> %393) #9, !dbg !16
  %396 = shufflevector <8 x half> %391, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %397 = shufflevector <8 x half> %391, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %396, <4 x float> %310) #9, !dbg !16
  %399 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %397, <4 x float> %398) #9, !dbg !16
  %401 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %402 = shufflevector <8 x half> %399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %402, <4 x float> %315) #9, !dbg !16
  %404 = shufflevector <8 x half> %399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %404, <4 x float> %403) #9, !dbg !16
  %406 = shufflevector <8 x half> %401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %407 = shufflevector <8 x half> %401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %406, <4 x float> %319) #9, !dbg !16
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %407, <4 x float> %408) #9, !dbg !16
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local334, !dbg !11
  %dma.source.bits1372 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !11
  %dma.source.opaque1373 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1372) #10, !dbg !11
  %dma.source.ptr1374 = inttoptr i64 %dma.source.opaque1373 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1374, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local918 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local342, !dbg !11
  %dma.source.bits1375 = ptrtoint ptr addrspace(1) %address.local918 to i64, !dbg !11
  %dma.source.opaque1376 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1375) #10, !dbg !11
  %dma.source.ptr1377 = inttoptr i64 %dma.source.opaque1376 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1377, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local919 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local350, !dbg !11
  %dma.source.bits1378 = ptrtoint ptr addrspace(1) %address.local919 to i64, !dbg !11
  %dma.source.opaque1379 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1378) #10, !dbg !11
  %dma.source.ptr1380 = inttoptr i64 %dma.source.opaque1379 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1380, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local920 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local358, !dbg !11
  %dma.source.bits1381 = ptrtoint ptr addrspace(1) %address.local920 to i64, !dbg !11
  %dma.source.opaque1382 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1381) #10, !dbg !11
  %dma.source.ptr1383 = inttoptr i64 %dma.source.opaque1382 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1383, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local921 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local366, !dbg !11
  %dma.source.bits1384 = ptrtoint ptr addrspace(1) %address.local921 to i64, !dbg !11
  %dma.source.opaque1385 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1384) #10, !dbg !11
  %dma.source.ptr1386 = inttoptr i64 %dma.source.opaque1385 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1386, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local922 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local374, !dbg !11
  %dma.source.bits1387 = ptrtoint ptr addrspace(1) %address.local922 to i64, !dbg !11
  %dma.source.opaque1388 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1387) #10, !dbg !11
  %dma.source.ptr1389 = inttoptr i64 %dma.source.opaque1388 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1389, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local923 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local382, !dbg !11
  %dma.source.bits1390 = ptrtoint ptr addrspace(1) %address.local923 to i64, !dbg !11
  %dma.source.opaque1391 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1390) #10, !dbg !11
  %dma.source.ptr1392 = inttoptr i64 %dma.source.opaque1391 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1392, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local924 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local390, !dbg !11
  %dma.source.bits1393 = ptrtoint ptr addrspace(1) %address.local924 to i64, !dbg !11
  %dma.source.opaque1394 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1393) #10, !dbg !11
  %dma.source.ptr1395 = inttoptr i64 %dma.source.opaque1394 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1395, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %410 = or disjoint i64 %137, 96, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %411 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %410, !dbg !12
  %address.local925 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local417, !dbg !12
  %dma.source.bits1396 = ptrtoint ptr addrspace(1) %address.local925 to i64, !dbg !12
  %dma.source.opaque1397 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1396) #10, !dbg !12
  %dma.source.ptr1398 = inttoptr i64 %dma.source.opaque1397 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1398, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local926 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local449, !dbg !12
  %dma.source.bits1399 = ptrtoint ptr addrspace(1) %address.local926 to i64, !dbg !12
  %dma.source.opaque1400 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1399) #10, !dbg !12
  %dma.source.ptr1401 = inttoptr i64 %dma.source.opaque1400 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1401, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local927 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local479, !dbg !12
  %dma.source.bits1402 = ptrtoint ptr addrspace(1) %address.local927 to i64, !dbg !12
  %dma.source.opaque1403 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1402) #10, !dbg !12
  %dma.source.ptr1404 = inttoptr i64 %dma.source.opaque1403 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1404, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local928 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local512, !dbg !12
  %dma.source.bits1405 = ptrtoint ptr addrspace(1) %address.local928 to i64, !dbg !12
  %dma.source.opaque1406 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1405) #10, !dbg !12
  %dma.source.ptr1407 = inttoptr i64 %dma.source.opaque1406 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1407, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local546, !dbg !12
  %dma.source.bits1408 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !12
  %dma.source.opaque1409 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1408) #10, !dbg !12
  %dma.source.ptr1410 = inttoptr i64 %dma.source.opaque1409 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1410, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local579, !dbg !12
  %dma.source.bits1411 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !12
  %dma.source.opaque1412 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1411) #10, !dbg !12
  %dma.source.ptr1413 = inttoptr i64 %dma.source.opaque1412 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1413, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local615, !dbg !12
  %dma.source.bits1414 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !12
  %dma.source.opaque1415 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1414) #10, !dbg !12
  %dma.source.ptr1416 = inttoptr i64 %dma.source.opaque1415 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1416, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local648, !dbg !12
  %dma.source.bits1417 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !12
  %dma.source.opaque1418 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1417) #10, !dbg !12
  %dma.source.ptr1419 = inttoptr i64 %dma.source.opaque1418 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1419, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %412 = load <4 x half>, ptr addrspace(3) %address.local658, align 2, !dbg !13
  %413 = load <4 x half>, ptr addrspace(3) %address.local660, align 2, !dbg !13
  %414 = load <4 x half>, ptr addrspace(3) %address.local662, align 2, !dbg !13
  %415 = fmul <4 x half> %412, splat (half 0xH3015), !dbg !13
  %416 = load <4 x half>, ptr addrspace(3) %address.local664, align 2, !dbg !13
  %417 = fmul <4 x half> %413, splat (half 0xH3015), !dbg !13
  %418 = fmul <4 x half> %414, splat (half 0xH3015), !dbg !13
  %419 = fmul <4 x half> %416, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local933 = shl nuw nsw i64 %address.local392, 5, !dbg !15
  %address.local934 = and i64 %address.local933, 8064, !dbg !15
  %address.local935 = shl nuw nsw i64 %address.local392, 3, !dbg !15
  %address.local936 = shl nuw nsw i64 %address.local392, 3, !dbg !15
  %address.local937 = and i64 %address.local936, 2016, !dbg !15
  %address.local938 = sub nsw i64 %address.local935, %address.local937, !dbg !15
  %address.local939 = add nsw i64 %address.local934, %address.local938, !dbg !15
  %420 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14201421 = add nsw i64 %137, %address.local939, !dbg !15
  %.idx14201421.tr = trunc i64 %.idx14201421 to i32, !dbg !15
  %dma.buffer.offset1229 = shl i32 %.idx14201421.tr, 1, !dbg !15
  %421 = insertelement <2 x i64> poison, i64 %420, i64 0, !dbg !15
  %422 = bitcast <2 x i64> %421 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1230 = shufflevector <4 x i32> %422, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %423 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1229, <4 x i32> %dma.buffer.rsrc1230) #11, !dbg !15
  %address.local941 = shl nuw nsw i32 %address.thread328, 3, !dbg !15
  %address.local942 = and i32 %address.local941, 24, !dbg !15
  %address.local943 = shl nuw nsw i32 %address.thread328, 5, !dbg !15
  %address.local944 = and i32 %address.local943, 8064, !dbg !15
  %address.local945 = or disjoint i32 %address.local942, %address.local944, !dbg !15
  %address.local946 = or disjoint i32 %address.local945, 8192, !dbg !15
  %424 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %425 = trunc nuw i64 %137 to i32, !dbg !15
  %.idx14221423.tr = or disjoint i32 %address.local946, %425, !dbg !15
  %dma.buffer.offset1227 = shl nuw nsw i32 %.idx14221423.tr, 1, !dbg !15
  %426 = insertelement <2 x i64> poison, i64 %424, i64 0, !dbg !15
  %427 = bitcast <2 x i64> %426 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1228 = shufflevector <4 x i32> %427, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1227, <4 x i32> %dma.buffer.rsrc1228) #11, !dbg !15
  %429 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %430 = shufflevector <8 x half> %429, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %430, <4 x float> %334) #9, !dbg !16
  %432 = shufflevector <8 x half> %429, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %432, <4 x float> %431) #9, !dbg !16
  %434 = extractelement <4 x float> %433, i64 0, !dbg !16
  %435 = call float @llvm.maxnum.f32(float %434, float 0xFFF0000000000000), !dbg !8
  %436 = extractelement <4 x float> %433, i64 1, !dbg !16
  %437 = call float @llvm.maxnum.f32(float %435, float %436), !dbg !8
  %438 = extractelement <4 x float> %433, i64 2, !dbg !16
  %439 = call float @llvm.maxnum.f32(float %437, float %438), !dbg !8
  %440 = extractelement <4 x float> %433, i64 3, !dbg !16
  %441 = call float @llvm.maxnum.f32(float %439, float %440), !dbg !8
  %442 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %443 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %444 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %445 = shufflevector <8 x half> %442, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %445, <4 x float> %341) #9, !dbg !16
  %447 = shufflevector <8 x half> %442, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %447, <4 x float> %446) #9, !dbg !16
  %449 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %450 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %449, <4 x float> %345) #9, !dbg !16
  %452 = extractelement <4 x float> %448, i64 0, !dbg !16
  %453 = call float @llvm.maxnum.f32(float %441, float %452), !dbg !8
  %454 = extractelement <4 x float> %448, i64 1, !dbg !16
  %455 = call float @llvm.maxnum.f32(float %453, float %454), !dbg !8
  %456 = extractelement <4 x float> %448, i64 2, !dbg !16
  %457 = call float @llvm.maxnum.f32(float %455, float %456), !dbg !8
  %458 = extractelement <4 x float> %448, i64 3, !dbg !16
  %459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %450, <4 x float> %451) #9, !dbg !16
  %460 = call float @llvm.maxnum.f32(float %457, float %458), !dbg !8
  %461 = extractelement <4 x float> %459, i64 0, !dbg !16
  %462 = call float @llvm.maxnum.f32(float %460, float %461), !dbg !8
  %463 = extractelement <4 x float> %459, i64 1, !dbg !16
  %464 = call float @llvm.maxnum.f32(float %462, float %463), !dbg !8
  %465 = extractelement <4 x float> %459, i64 2, !dbg !16
  %466 = call float @llvm.maxnum.f32(float %464, float %465), !dbg !8
  %467 = extractelement <4 x float> %459, i64 3, !dbg !16
  %468 = call float @llvm.maxnum.f32(float %466, float %467), !dbg !8
  %469 = shufflevector <8 x half> %444, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %469, <4 x float> %350) #9, !dbg !16
  %471 = shufflevector <8 x half> %444, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %472 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %473 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %471, <4 x float> %470) #9, !dbg !16
  %475 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %476 = extractelement <4 x float> %474, i64 0, !dbg !16
  %477 = call float @llvm.maxnum.f32(float %468, float %476), !dbg !8
  %478 = extractelement <4 x float> %474, i64 1, !dbg !16
  %479 = call float @llvm.maxnum.f32(float %477, float %478), !dbg !8
  %480 = extractelement <4 x float> %474, i64 2, !dbg !16
  %481 = call float @llvm.maxnum.f32(float %479, float %480), !dbg !8
  %482 = extractelement <4 x float> %474, i64 3, !dbg !16
  %483 = call float @llvm.maxnum.f32(float %481, float %482), !dbg !8
  %484 = shufflevector <8 x half> %472, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %484, <4 x float> %356) #9, !dbg !16
  %486 = shufflevector <8 x half> %472, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %487 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %486, <4 x float> %485) #9, !dbg !16
  %488 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %489 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %488, <4 x float> %360) #9, !dbg !16
  %491 = extractelement <4 x float> %487, i64 0, !dbg !16
  %492 = call float @llvm.maxnum.f32(float %483, float %491), !dbg !8
  %493 = extractelement <4 x float> %487, i64 1, !dbg !16
  %494 = call float @llvm.maxnum.f32(float %492, float %493), !dbg !8
  %495 = extractelement <4 x float> %487, i64 2, !dbg !16
  %496 = call float @llvm.maxnum.f32(float %494, float %495), !dbg !8
  %497 = extractelement <4 x float> %487, i64 3, !dbg !16
  %498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %489, <4 x float> %490) #9, !dbg !16
  %499 = call float @llvm.maxnum.f32(float %496, float %497), !dbg !8
  %500 = extractelement <4 x float> %498, i64 0, !dbg !16
  %501 = call float @llvm.maxnum.f32(float %499, float %500), !dbg !8
  %502 = extractelement <4 x float> %498, i64 1, !dbg !16
  %503 = call float @llvm.maxnum.f32(float %501, float %502), !dbg !8
  %504 = extractelement <4 x float> %498, i64 2, !dbg !16
  %505 = call float @llvm.maxnum.f32(float %503, float %504), !dbg !8
  %506 = extractelement <4 x float> %498, i64 3, !dbg !16
  %507 = call float @llvm.maxnum.f32(float %505, float %506), !dbg !8
  %508 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %508, <4 x float> %364) #9, !dbg !16
  %510 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %511 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %512 = load <8 x half>, ptr addrspace(3) %address.local687, align 2, !dbg !16
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %510, <4 x float> %509) #9, !dbg !16
  %514 = load <8 x half>, ptr addrspace(3) %address.local712, align 2, !dbg !16
  %515 = extractelement <4 x float> %513, i64 0, !dbg !16
  %516 = call float @llvm.maxnum.f32(float %507, float %515), !dbg !8
  %517 = extractelement <4 x float> %513, i64 1, !dbg !16
  %518 = call float @llvm.maxnum.f32(float %516, float %517), !dbg !8
  %519 = extractelement <4 x float> %513, i64 2, !dbg !16
  %520 = call float @llvm.maxnum.f32(float %518, float %519), !dbg !8
  %521 = extractelement <4 x float> %513, i64 3, !dbg !16
  %522 = call float @llvm.maxnum.f32(float %520, float %521), !dbg !8
  %523 = shufflevector <8 x half> %511, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %524 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %415, <4 x half> %523, <4 x float> %370) #9, !dbg !16
  %525 = shufflevector <8 x half> %511, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %525, <4 x float> %524) #9, !dbg !16
  %527 = shufflevector <8 x half> %512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %528 = shufflevector <8 x half> %512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %527, <4 x float> %375) #9, !dbg !16
  %530 = extractelement <4 x float> %526, i64 0, !dbg !16
  %531 = call float @llvm.maxnum.f32(float %522, float %530), !dbg !8
  %532 = extractelement <4 x float> %526, i64 1, !dbg !16
  %533 = call float @llvm.maxnum.f32(float %531, float %532), !dbg !8
  %534 = extractelement <4 x float> %526, i64 2, !dbg !16
  %535 = call float @llvm.maxnum.f32(float %533, float %534), !dbg !8
  %536 = extractelement <4 x float> %526, i64 3, !dbg !16
  %537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %528, <4 x float> %529) #9, !dbg !16
  %538 = call float @llvm.maxnum.f32(float %535, float %536), !dbg !8
  %539 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %539, <4 x float> %381) #9, !dbg !16
  %541 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %542 = extractelement <4 x float> %537, i64 0, !dbg !16
  %543 = extractelement <4 x float> %537, i64 1, !dbg !16
  %544 = extractelement <4 x float> %537, i64 2, !dbg !16
  %545 = extractelement <4 x float> %537, i64 3, !dbg !16
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %541, <4 x float> %540) #9, !dbg !16
  %547 = load <8 x half>, ptr addrspace(3) %address.local736, align 2, !dbg !16
  %548 = load <8 x half>, ptr addrspace(3) %address.local775, align 2, !dbg !16
  %549 = load <8 x half>, ptr addrspace(3) %address.local777, align 2, !dbg !16
  %550 = extractelement <4 x float> %546, i64 0, !dbg !16
  %551 = shufflevector <8 x half> %547, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %552 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %551, <4 x float> %385) #9, !dbg !16
  %553 = shufflevector <8 x half> %547, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %553, <4 x float> %552) #9, !dbg !16
  %555 = shufflevector <8 x half> %548, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %556 = shufflevector <8 x half> %548, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %555, <4 x float> %389) #9, !dbg !16
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %556, <4 x float> %557) #9, !dbg !16
  %559 = shufflevector <8 x half> %549, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %560 = shufflevector <8 x half> %549, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %561 = extractelement <4 x float> %546, i64 1, !dbg !16
  %562 = extractelement <4 x float> %546, i64 2, !dbg !16
  %563 = extractelement <4 x float> %546, i64 3, !dbg !16
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %559, <4 x float> %395) #9, !dbg !16
  %565 = extractelement <4 x float> %554, i64 0, !dbg !16
  %566 = extractelement <4 x float> %554, i64 1, !dbg !16
  %567 = extractelement <4 x float> %554, i64 2, !dbg !16
  %568 = extractelement <4 x float> %554, i64 3, !dbg !16
  %569 = extractelement <4 x float> %558, i64 0, !dbg !16
  %570 = extractelement <4 x float> %558, i64 1, !dbg !16
  %571 = extractelement <4 x float> %558, i64 2, !dbg !16
  %572 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %560, <4 x float> %564) #9, !dbg !16
  %573 = extractelement <4 x float> %558, i64 3, !dbg !16
  %574 = load <8 x half>, ptr addrspace(3) %address.local820, align 2, !dbg !16
  %575 = load <8 x half>, ptr addrspace(3) %address.local841, align 2, !dbg !16
  %576 = extractelement <4 x float> %572, i64 0, !dbg !16
  %577 = extractelement <4 x float> %572, i64 1, !dbg !16
  %578 = extractelement <4 x float> %572, i64 2, !dbg !16
  %579 = extractelement <4 x float> %572, i64 3, !dbg !16
  %580 = shufflevector <8 x half> %574, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %581 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %580, <4 x float> %400) #9, !dbg !16
  %582 = shufflevector <8 x half> %574, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %583 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %582, <4 x float> %581) #9, !dbg !16
  %584 = shufflevector <8 x half> %575, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %585 = shufflevector <8 x half> %575, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %586 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %584, <4 x float> %405) #9, !dbg !16
  %587 = extractelement <4 x float> %583, i64 0, !dbg !16
  %588 = extractelement <4 x float> %583, i64 1, !dbg !16
  %589 = extractelement <4 x float> %583, i64 2, !dbg !16
  %590 = extractelement <4 x float> %583, i64 3, !dbg !16
  %591 = load <8 x half>, ptr addrspace(3) %address.local884, align 2, !dbg !16
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %585, <4 x float> %586) #9, !dbg !16
  %593 = shufflevector <8 x half> %591, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %594 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %418, <4 x half> %593, <4 x float> %409) #9, !dbg !16
  %595 = shufflevector <8 x half> %591, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %596 = extractelement <4 x float> %592, i64 0, !dbg !16
  %597 = extractelement <4 x float> %592, i64 1, !dbg !16
  %598 = extractelement <4 x float> %592, i64 2, !dbg !16
  %599 = extractelement <4 x float> %592, i64 3, !dbg !16
  %600 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %419, <4 x half> %595, <4 x float> %594) #9, !dbg !16
  %601 = extractelement <4 x float> %600, i64 0, !dbg !16
  %602 = extractelement <4 x float> %600, i64 1, !dbg !16
  %603 = extractelement <4 x float> %600, i64 2, !dbg !16
  %604 = extractelement <4 x float> %600, i64 3, !dbg !16
  %605 = bitcast float %538 to i32, !dbg !8
  %606 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %605), !dbg !8
  %607 = bitcast i32 %606 to float, !dbg !8
  %608 = call float @llvm.maxnum.f32(float %538, float %607), !dbg !8
  %609 = bitcast float %608 to i32, !dbg !8
  %610 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %609), !dbg !8
  %611 = bitcast i32 %610 to float, !dbg !8
  %612 = call float @llvm.maxnum.f32(float %608, float %611), !dbg !8
  %613 = call float @llvm.maxnum.f32(float %542, float 0xFFF0000000000000), !dbg !8
  %614 = call float @llvm.maxnum.f32(float %613, float %543), !dbg !8
  %615 = call float @llvm.maxnum.f32(float %614, float %544), !dbg !8
  %616 = call float @llvm.maxnum.f32(float %615, float %545), !dbg !8
  %617 = call float @llvm.maxnum.f32(float %616, float %550), !dbg !8
  %618 = call float @llvm.maxnum.f32(float %617, float %561), !dbg !8
  %619 = call float @llvm.maxnum.f32(float %618, float %562), !dbg !8
  %620 = call float @llvm.maxnum.f32(float %619, float %563), !dbg !8
  %621 = call float @llvm.maxnum.f32(float %620, float %565), !dbg !8
  %622 = call float @llvm.maxnum.f32(float %621, float %566), !dbg !8
  %623 = call float @llvm.maxnum.f32(float %622, float %567), !dbg !8
  %624 = call float @llvm.maxnum.f32(float %623, float %568), !dbg !8
  %625 = call float @llvm.maxnum.f32(float %624, float %569), !dbg !8
  %626 = call float @llvm.maxnum.f32(float %625, float %570), !dbg !8
  %627 = call float @llvm.maxnum.f32(float %626, float %571), !dbg !8
  %628 = call float @llvm.maxnum.f32(float %627, float %573), !dbg !8
  %629 = call float @llvm.maxnum.f32(float %628, float %576), !dbg !8
  %630 = call float @llvm.maxnum.f32(float %629, float %577), !dbg !8
  %631 = call float @llvm.maxnum.f32(float %630, float %578), !dbg !8
  %632 = call float @llvm.maxnum.f32(float %631, float %579), !dbg !8
  %633 = call float @llvm.maxnum.f32(float %632, float %587), !dbg !8
  %634 = call float @llvm.maxnum.f32(float %633, float %588), !dbg !8
  %635 = call float @llvm.maxnum.f32(float %634, float %589), !dbg !8
  %636 = call float @llvm.maxnum.f32(float %635, float %590), !dbg !8
  %637 = call float @llvm.maxnum.f32(float %636, float %596), !dbg !8
  %638 = call float @llvm.maxnum.f32(float %637, float %597), !dbg !8
  %639 = call float @llvm.maxnum.f32(float %638, float %598), !dbg !8
  %640 = call float @llvm.maxnum.f32(float %639, float %599), !dbg !8
  %641 = call float @llvm.maxnum.f32(float %640, float %601), !dbg !8
  %642 = call float @llvm.maxnum.f32(float %641, float %602), !dbg !8
  %643 = call float @llvm.maxnum.f32(float %642, float %603), !dbg !8
  %644 = call float @llvm.maxnum.f32(float %643, float %604), !dbg !8
  %645 = bitcast float %644 to i32, !dbg !8
  %646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %645), !dbg !8
  %647 = bitcast i32 %646 to float, !dbg !8
  %648 = call float @llvm.maxnum.f32(float %644, float %647), !dbg !8
  %649 = bitcast float %648 to i32, !dbg !8
  %650 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %649), !dbg !8
  %651 = bitcast i32 %650 to float, !dbg !8
  %652 = call float @llvm.maxnum.f32(float %648, float %651), !dbg !8
  %653 = extractvalue [2 x <1 x float>] %128, 0, !dbg !8
  %654 = insertelement <1 x float> poison, float %612, i64 0, !dbg !8
  %655 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %654, <1 x float> %653), !dbg !8
  %656 = extractvalue [2 x <1 x float>] %128, 1, !dbg !8
  %657 = insertelement <1 x float> poison, float %652, i64 0, !dbg !8
  %658 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %657, <1 x float> %656), !dbg !8
  %659 = shufflevector <1 x float> %655, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %660 = fsub <4 x float> %433, %659, !dbg !20
  %661 = fsub <4 x float> %448, %659, !dbg !20
  %662 = fsub <4 x float> %459, %659, !dbg !20
  %663 = fsub <4 x float> %474, %659, !dbg !20
  %664 = fsub <4 x float> %487, %659, !dbg !20
  %665 = fsub <4 x float> %498, %659, !dbg !20
  %666 = fsub <4 x float> %513, %659, !dbg !20
  %667 = fsub <4 x float> %526, %659, !dbg !20
  %668 = shufflevector <1 x float> %658, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %669 = fsub <4 x float> %537, %668, !dbg !20
  %670 = fsub <4 x float> %546, %668, !dbg !20
  %671 = fsub <4 x float> %554, %668, !dbg !20
  %672 = fsub <4 x float> %558, %668, !dbg !20
  %673 = fsub <4 x float> %572, %668, !dbg !20
  %674 = fsub <4 x float> %583, %668, !dbg !20
  %675 = fsub <4 x float> %592, %668, !dbg !20
  %676 = fsub <4 x float> %600, %668, !dbg !20
  %677 = fsub <1 x float> %653, %655, !dbg !21
  %678 = extractelement <1 x float> %677, i64 0, !dbg !22
  %679 = call float @__ocml_exp2_f32(float %678), !dbg !22
  %680 = insertelement <4 x float> poison, float %679, i64 0, !dbg !27
  %681 = shufflevector <4 x float> %680, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %682 = extractvalue [2 x <1 x float>] %127, 0, !dbg !23
  %683 = extractelement <1 x float> %682, i64 0, !dbg !23
  %684 = fmul float %683, %679, !dbg !23
  %685 = fsub <1 x float> %656, %658, !dbg !21
  %686 = extractelement <1 x float> %685, i64 0, !dbg !22
  %687 = call float @__ocml_exp2_f32(float %686), !dbg !22
  %688 = insertelement <4 x float> poison, float %687, i64 0, !dbg !27
  %689 = shufflevector <4 x float> %688, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %690 = extractvalue [2 x <1 x float>] %127, 1, !dbg !23
  %691 = extractelement <1 x float> %690, i64 0, !dbg !23
  %692 = fmul float %691, %687, !dbg !23
  %693 = extractelement <4 x float> %660, i64 0, !dbg !24
  %694 = call float @__ocml_exp2_f32(float %693), !dbg !24
  %695 = insertelement <4 x float> poison, float %694, i64 0, !dbg !26
  %696 = fadd float %694, 0x0000000000000000, !dbg !25
  %697 = extractelement <4 x float> %660, i64 1, !dbg !24
  %698 = call float @__ocml_exp2_f32(float %697), !dbg !24
  %699 = insertelement <4 x float> %695, float %698, i64 1, !dbg !26
  %700 = fadd float %696, %698, !dbg !25
  %701 = extractelement <4 x float> %660, i64 2, !dbg !24
  %702 = call float @__ocml_exp2_f32(float %701), !dbg !24
  %703 = insertelement <4 x float> %699, float %702, i64 2, !dbg !26
  %704 = fadd float %700, %702, !dbg !25
  %705 = extractelement <4 x float> %660, i64 3, !dbg !24
  %706 = call float @__ocml_exp2_f32(float %705), !dbg !24
  %707 = insertelement <4 x float> %703, float %706, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #9, !dbg !26, !frisk.register_lifetimes !1
  %708 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !26
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #9, !dbg !25, !frisk.register_lifetimes !1
  %709 = fadd float %704, %706, !dbg !25
  %710 = extractelement <4 x float> %661, i64 0, !dbg !24
  %711 = call float @__ocml_exp2_f32(float %710), !dbg !24
  %712 = insertelement <4 x float> poison, float %711, i64 0, !dbg !26
  %713 = fadd float %709, %711, !dbg !25
  %714 = extractelement <4 x float> %661, i64 1, !dbg !24
  %715 = call float @__ocml_exp2_f32(float %714), !dbg !24
  %716 = insertelement <4 x float> %712, float %715, i64 1, !dbg !26
  %717 = fadd float %713, %715, !dbg !25
  %718 = extractelement <4 x float> %661, i64 2, !dbg !24
  %719 = call float @__ocml_exp2_f32(float %718), !dbg !24
  %720 = insertelement <4 x float> %716, float %719, i64 2, !dbg !26
  %721 = fadd float %717, %719, !dbg !25
  %722 = extractelement <4 x float> %661, i64 3, !dbg !24
  %723 = call float @__ocml_exp2_f32(float %722), !dbg !24
  %724 = insertelement <4 x float> %720, float %723, i64 3, !dbg !26
  %math.result219 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !26, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result219 to <4 x half>, !dbg !26
  %math.packed220 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !25, !frisk.register_lifetimes !1
  %726 = fadd float %721, %723, !dbg !25
  %727 = extractelement <4 x float> %662, i64 0, !dbg !24
  %728 = call float @__ocml_exp2_f32(float %727), !dbg !24
  %729 = insertelement <4 x float> poison, float %728, i64 0, !dbg !26
  %730 = fadd float %726, %728, !dbg !25
  %731 = extractelement <4 x float> %662, i64 1, !dbg !24
  %732 = call float @__ocml_exp2_f32(float %731), !dbg !24
  %733 = insertelement <4 x float> %729, float %732, i64 1, !dbg !26
  %734 = fadd float %730, %732, !dbg !25
  %735 = extractelement <4 x float> %662, i64 2, !dbg !24
  %736 = call float @__ocml_exp2_f32(float %735), !dbg !24
  %737 = insertelement <4 x float> %733, float %736, i64 2, !dbg !26
  %738 = fadd float %734, %736, !dbg !25
  %739 = extractelement <4 x float> %662, i64 3, !dbg !24
  %740 = call float @__ocml_exp2_f32(float %739), !dbg !24
  %741 = insertelement <4 x float> %737, float %740, i64 3, !dbg !26
  %math.result221 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !26, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result221 to <4 x half>, !dbg !26
  %math.packed222 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !25, !frisk.register_lifetimes !1
  %743 = fadd float %738, %740, !dbg !25
  %744 = extractelement <4 x float> %663, i64 0, !dbg !24
  %745 = call float @__ocml_exp2_f32(float %744), !dbg !24
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !26
  %747 = fadd float %743, %745, !dbg !25
  %748 = extractelement <4 x float> %663, i64 1, !dbg !24
  %749 = call float @__ocml_exp2_f32(float %748), !dbg !24
  %750 = insertelement <4 x float> %746, float %749, i64 1, !dbg !26
  %751 = fadd float %747, %749, !dbg !25
  %752 = extractelement <4 x float> %663, i64 2, !dbg !24
  %753 = call float @__ocml_exp2_f32(float %752), !dbg !24
  %754 = insertelement <4 x float> %750, float %753, i64 2, !dbg !26
  %755 = fadd float %751, %753, !dbg !25
  %756 = extractelement <4 x float> %663, i64 3, !dbg !24
  %757 = call float @__ocml_exp2_f32(float %756), !dbg !24
  %758 = insertelement <4 x float> %754, float %757, i64 3, !dbg !26
  %math.result223 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %758) #9, !dbg !26, !frisk.register_lifetimes !1
  %759 = fptrunc <4 x float> %math.result223 to <4 x half>, !dbg !26
  %math.packed224 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %759) #9, !dbg !25, !frisk.register_lifetimes !1
  %760 = fadd float %755, %757, !dbg !25
  %761 = extractelement <4 x float> %664, i64 0, !dbg !24
  %762 = call float @__ocml_exp2_f32(float %761), !dbg !24
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !26
  %764 = fadd float %760, %762, !dbg !25
  %765 = extractelement <4 x float> %664, i64 1, !dbg !24
  %766 = call float @__ocml_exp2_f32(float %765), !dbg !24
  %767 = insertelement <4 x float> %763, float %766, i64 1, !dbg !26
  %768 = fadd float %764, %766, !dbg !25
  %769 = extractelement <4 x float> %664, i64 2, !dbg !24
  %770 = call float @__ocml_exp2_f32(float %769), !dbg !24
  %771 = insertelement <4 x float> %767, float %770, i64 2, !dbg !26
  %772 = fadd float %768, %770, !dbg !25
  %773 = extractelement <4 x float> %664, i64 3, !dbg !24
  %774 = call float @__ocml_exp2_f32(float %773), !dbg !24
  %775 = insertelement <4 x float> %771, float %774, i64 3, !dbg !26
  %math.result225 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !26, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result225 to <4 x half>, !dbg !26
  %math.packed226 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !25, !frisk.register_lifetimes !1
  %777 = fadd float %772, %774, !dbg !25
  %778 = extractelement <4 x float> %665, i64 0, !dbg !24
  %779 = call float @__ocml_exp2_f32(float %778), !dbg !24
  %780 = insertelement <4 x float> poison, float %779, i64 0, !dbg !26
  %781 = fadd float %777, %779, !dbg !25
  %782 = extractelement <4 x float> %665, i64 1, !dbg !24
  %783 = call float @__ocml_exp2_f32(float %782), !dbg !24
  %784 = insertelement <4 x float> %780, float %783, i64 1, !dbg !26
  %785 = fadd float %781, %783, !dbg !25
  %786 = extractelement <4 x float> %665, i64 2, !dbg !24
  %787 = call float @__ocml_exp2_f32(float %786), !dbg !24
  %788 = insertelement <4 x float> %784, float %787, i64 2, !dbg !26
  %789 = fadd float %785, %787, !dbg !25
  %790 = extractelement <4 x float> %665, i64 3, !dbg !24
  %791 = call float @__ocml_exp2_f32(float %790), !dbg !24
  %792 = insertelement <4 x float> %788, float %791, i64 3, !dbg !26
  %math.result227 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !26, !frisk.register_lifetimes !1
  %793 = fptrunc <4 x float> %math.result227 to <4 x half>, !dbg !26
  %math.packed228 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !25, !frisk.register_lifetimes !1
  %794 = fadd float %789, %791, !dbg !25
  %795 = extractelement <4 x float> %666, i64 0, !dbg !24
  %796 = call float @__ocml_exp2_f32(float %795), !dbg !24
  %797 = insertelement <4 x float> poison, float %796, i64 0, !dbg !26
  %798 = fadd float %794, %796, !dbg !25
  %799 = extractelement <4 x float> %666, i64 1, !dbg !24
  %800 = call float @__ocml_exp2_f32(float %799), !dbg !24
  %801 = insertelement <4 x float> %797, float %800, i64 1, !dbg !26
  %802 = fadd float %798, %800, !dbg !25
  %803 = extractelement <4 x float> %666, i64 2, !dbg !24
  %804 = call float @__ocml_exp2_f32(float %803), !dbg !24
  %805 = insertelement <4 x float> %801, float %804, i64 2, !dbg !26
  %806 = fadd float %802, %804, !dbg !25
  %807 = extractelement <4 x float> %666, i64 3, !dbg !24
  %808 = call float @__ocml_exp2_f32(float %807), !dbg !24
  %809 = insertelement <4 x float> %805, float %808, i64 3, !dbg !26
  %math.result229 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #9, !dbg !26, !frisk.register_lifetimes !1
  %810 = fptrunc <4 x float> %math.result229 to <4 x half>, !dbg !26
  %math.packed230 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #9, !dbg !25, !frisk.register_lifetimes !1
  %811 = fadd float %806, %808, !dbg !25
  %812 = extractelement <4 x float> %667, i64 0, !dbg !24
  %813 = call float @__ocml_exp2_f32(float %812), !dbg !24
  %814 = insertelement <4 x float> poison, float %813, i64 0, !dbg !26
  %815 = fadd float %811, %813, !dbg !25
  %816 = extractelement <4 x float> %667, i64 1, !dbg !24
  %817 = call float @__ocml_exp2_f32(float %816), !dbg !24
  %818 = insertelement <4 x float> %814, float %817, i64 1, !dbg !26
  %819 = fadd float %815, %817, !dbg !25
  %820 = extractelement <4 x float> %667, i64 2, !dbg !24
  %821 = call float @__ocml_exp2_f32(float %820), !dbg !24
  %822 = insertelement <4 x float> %818, float %821, i64 2, !dbg !26
  %823 = fadd float %819, %821, !dbg !25
  %824 = extractelement <4 x float> %667, i64 3, !dbg !24
  %825 = call float @__ocml_exp2_f32(float %824), !dbg !24
  %826 = insertelement <4 x float> %822, float %825, i64 3, !dbg !26
  %math.result231 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %826) #9, !dbg !26, !frisk.register_lifetimes !1
  %827 = fptrunc <4 x float> %math.result231 to <4 x half>, !dbg !26
  %math.packed232 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %827) #9, !dbg !25, !frisk.register_lifetimes !1
  %828 = fadd float %823, %825, !dbg !25
  %829 = extractelement <4 x float> %669, i64 0, !dbg !24
  %830 = call float @__ocml_exp2_f32(float %829), !dbg !24
  %831 = insertelement <4 x float> poison, float %830, i64 0, !dbg !26
  %832 = fadd float %830, 0x0000000000000000, !dbg !25
  %833 = extractelement <4 x float> %669, i64 1, !dbg !24
  %834 = call float @__ocml_exp2_f32(float %833), !dbg !24
  %835 = insertelement <4 x float> %831, float %834, i64 1, !dbg !26
  %836 = fadd float %832, %834, !dbg !25
  %837 = extractelement <4 x float> %669, i64 2, !dbg !24
  %838 = call float @__ocml_exp2_f32(float %837), !dbg !24
  %839 = insertelement <4 x float> %835, float %838, i64 2, !dbg !26
  %840 = fadd float %836, %838, !dbg !25
  %841 = extractelement <4 x float> %669, i64 3, !dbg !24
  %842 = call float @__ocml_exp2_f32(float %841), !dbg !24
  %843 = insertelement <4 x float> %839, float %842, i64 3, !dbg !26
  %math.result233 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #9, !dbg !26, !frisk.register_lifetimes !1
  %844 = fptrunc <4 x float> %math.result233 to <4 x half>, !dbg !26
  %math.packed234 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #9, !dbg !25, !frisk.register_lifetimes !1
  %845 = fadd float %840, %842, !dbg !25
  %846 = extractelement <4 x float> %670, i64 0, !dbg !24
  %847 = call float @__ocml_exp2_f32(float %846), !dbg !24
  %848 = insertelement <4 x float> poison, float %847, i64 0, !dbg !26
  %849 = fadd float %845, %847, !dbg !25
  %850 = extractelement <4 x float> %670, i64 1, !dbg !24
  %851 = call float @__ocml_exp2_f32(float %850), !dbg !24
  %852 = insertelement <4 x float> %848, float %851, i64 1, !dbg !26
  %853 = fadd float %849, %851, !dbg !25
  %854 = extractelement <4 x float> %670, i64 2, !dbg !24
  %855 = call float @__ocml_exp2_f32(float %854), !dbg !24
  %856 = insertelement <4 x float> %852, float %855, i64 2, !dbg !26
  %857 = fadd float %853, %855, !dbg !25
  %858 = extractelement <4 x float> %670, i64 3, !dbg !24
  %859 = call float @__ocml_exp2_f32(float %858), !dbg !24
  %860 = insertelement <4 x float> %856, float %859, i64 3, !dbg !26
  %math.result235 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %860) #9, !dbg !26, !frisk.register_lifetimes !1
  %861 = fptrunc <4 x float> %math.result235 to <4 x half>, !dbg !26
  %math.packed236 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %861) #9, !dbg !25, !frisk.register_lifetimes !1
  %862 = fadd float %857, %859, !dbg !25
  %863 = extractelement <4 x float> %671, i64 0, !dbg !24
  %864 = call float @__ocml_exp2_f32(float %863), !dbg !24
  %865 = insertelement <4 x float> poison, float %864, i64 0, !dbg !26
  %866 = fadd float %862, %864, !dbg !25
  %867 = extractelement <4 x float> %671, i64 1, !dbg !24
  %868 = call float @__ocml_exp2_f32(float %867), !dbg !24
  %869 = insertelement <4 x float> %865, float %868, i64 1, !dbg !26
  %870 = fadd float %866, %868, !dbg !25
  %871 = extractelement <4 x float> %671, i64 2, !dbg !24
  %872 = call float @__ocml_exp2_f32(float %871), !dbg !24
  %873 = insertelement <4 x float> %869, float %872, i64 2, !dbg !26
  %874 = fadd float %870, %872, !dbg !25
  %875 = extractelement <4 x float> %671, i64 3, !dbg !24
  %876 = call float @__ocml_exp2_f32(float %875), !dbg !24
  %877 = insertelement <4 x float> %873, float %876, i64 3, !dbg !26
  %math.result237 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %877) #9, !dbg !26, !frisk.register_lifetimes !1
  %878 = fptrunc <4 x float> %math.result237 to <4 x half>, !dbg !26
  %math.packed238 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %878) #9, !dbg !25, !frisk.register_lifetimes !1
  %879 = fadd float %874, %876, !dbg !25
  %880 = extractelement <4 x float> %672, i64 0, !dbg !24
  %881 = call float @__ocml_exp2_f32(float %880), !dbg !24
  %882 = insertelement <4 x float> poison, float %881, i64 0, !dbg !26
  %883 = fadd float %879, %881, !dbg !25
  %884 = extractelement <4 x float> %672, i64 1, !dbg !24
  %885 = call float @__ocml_exp2_f32(float %884), !dbg !24
  %886 = insertelement <4 x float> %882, float %885, i64 1, !dbg !26
  %887 = fadd float %883, %885, !dbg !25
  %888 = extractelement <4 x float> %672, i64 2, !dbg !24
  %889 = call float @__ocml_exp2_f32(float %888), !dbg !24
  %890 = insertelement <4 x float> %886, float %889, i64 2, !dbg !26
  %891 = fadd float %887, %889, !dbg !25
  %892 = extractelement <4 x float> %672, i64 3, !dbg !24
  %893 = call float @__ocml_exp2_f32(float %892), !dbg !24
  %894 = insertelement <4 x float> %890, float %893, i64 3, !dbg !26
  %math.result239 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %894) #9, !dbg !26, !frisk.register_lifetimes !1
  %895 = fptrunc <4 x float> %math.result239 to <4 x half>, !dbg !26
  %math.packed240 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %895) #9, !dbg !25, !frisk.register_lifetimes !1
  %896 = fadd float %891, %893, !dbg !25
  %897 = extractelement <4 x float> %673, i64 0, !dbg !24
  %898 = call float @__ocml_exp2_f32(float %897), !dbg !24
  %899 = insertelement <4 x float> poison, float %898, i64 0, !dbg !26
  %900 = fadd float %896, %898, !dbg !25
  %901 = extractelement <4 x float> %673, i64 1, !dbg !24
  %902 = call float @__ocml_exp2_f32(float %901), !dbg !24
  %903 = insertelement <4 x float> %899, float %902, i64 1, !dbg !26
  %904 = fadd float %900, %902, !dbg !25
  %905 = extractelement <4 x float> %673, i64 2, !dbg !24
  %906 = call float @__ocml_exp2_f32(float %905), !dbg !24
  %907 = insertelement <4 x float> %903, float %906, i64 2, !dbg !26
  %908 = fadd float %904, %906, !dbg !25
  %909 = extractelement <4 x float> %673, i64 3, !dbg !24
  %910 = call float @__ocml_exp2_f32(float %909), !dbg !24
  %911 = insertelement <4 x float> %907, float %910, i64 3, !dbg !26
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %911) #9, !dbg !26, !frisk.register_lifetimes !1
  %912 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !26
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %912) #9, !dbg !25, !frisk.register_lifetimes !1
  %913 = fadd float %908, %910, !dbg !25
  %914 = extractelement <4 x float> %674, i64 0, !dbg !24
  %915 = call float @__ocml_exp2_f32(float %914), !dbg !24
  %916 = insertelement <4 x float> poison, float %915, i64 0, !dbg !26
  %917 = fadd float %913, %915, !dbg !25
  %918 = extractelement <4 x float> %674, i64 1, !dbg !24
  %919 = call float @__ocml_exp2_f32(float %918), !dbg !24
  %920 = insertelement <4 x float> %916, float %919, i64 1, !dbg !26
  %921 = fadd float %917, %919, !dbg !25
  %922 = extractelement <4 x float> %674, i64 2, !dbg !24
  %923 = call float @__ocml_exp2_f32(float %922), !dbg !24
  %924 = insertelement <4 x float> %920, float %923, i64 2, !dbg !26
  %925 = fadd float %921, %923, !dbg !25
  %926 = extractelement <4 x float> %674, i64 3, !dbg !24
  %927 = call float @__ocml_exp2_f32(float %926), !dbg !24
  %928 = insertelement <4 x float> %924, float %927, i64 3, !dbg !26
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %928) #9, !dbg !26, !frisk.register_lifetimes !1
  %929 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !26
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %929) #9, !dbg !25, !frisk.register_lifetimes !1
  %930 = fadd float %925, %927, !dbg !25
  %931 = extractelement <4 x float> %675, i64 0, !dbg !24
  %932 = call float @__ocml_exp2_f32(float %931), !dbg !24
  %933 = insertelement <4 x float> poison, float %932, i64 0, !dbg !26
  %934 = fadd float %930, %932, !dbg !25
  %935 = extractelement <4 x float> %675, i64 1, !dbg !24
  %936 = call float @__ocml_exp2_f32(float %935), !dbg !24
  %937 = insertelement <4 x float> %933, float %936, i64 1, !dbg !26
  %938 = fadd float %934, %936, !dbg !25
  %939 = extractelement <4 x float> %675, i64 2, !dbg !24
  %940 = call float @__ocml_exp2_f32(float %939), !dbg !24
  %941 = insertelement <4 x float> %937, float %940, i64 2, !dbg !26
  %942 = fadd float %938, %940, !dbg !25
  %943 = extractelement <4 x float> %675, i64 3, !dbg !24
  %944 = call float @__ocml_exp2_f32(float %943), !dbg !24
  %945 = insertelement <4 x float> %941, float %944, i64 3, !dbg !26
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %945) #9, !dbg !26, !frisk.register_lifetimes !1
  %946 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !26
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %946) #9, !dbg !25, !frisk.register_lifetimes !1
  %947 = fadd float %942, %944, !dbg !25
  %948 = extractelement <4 x float> %676, i64 0, !dbg !24
  %949 = call float @__ocml_exp2_f32(float %948), !dbg !24
  %950 = insertelement <4 x float> poison, float %949, i64 0, !dbg !26
  %951 = fadd float %947, %949, !dbg !25
  %952 = extractelement <4 x float> %676, i64 1, !dbg !24
  %953 = call float @__ocml_exp2_f32(float %952), !dbg !24
  %954 = insertelement <4 x float> %950, float %953, i64 1, !dbg !26
  %955 = fadd float %951, %953, !dbg !25
  %956 = extractelement <4 x float> %676, i64 2, !dbg !24
  %957 = call float @__ocml_exp2_f32(float %956), !dbg !24
  %958 = insertelement <4 x float> %954, float %957, i64 2, !dbg !26
  %959 = fadd float %955, %957, !dbg !25
  %960 = extractelement <4 x float> %676, i64 3, !dbg !24
  %961 = call float @__ocml_exp2_f32(float %960), !dbg !24
  %962 = insertelement <4 x float> %958, float %961, i64 3, !dbg !26
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %962) #9, !dbg !26, !frisk.register_lifetimes !1
  %963 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !26
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %963) #9, !dbg !25, !frisk.register_lifetimes !1
  %964 = fadd float %959, %961, !dbg !25
  %965 = bitcast float %828 to i32, !dbg !25
  %966 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %965), !dbg !25
  %967 = bitcast i32 %966 to float, !dbg !25
  %968 = fadd float %828, %967, !dbg !25
  %969 = bitcast float %968 to i32, !dbg !25
  %970 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %969), !dbg !25
  %971 = bitcast i32 %970 to float, !dbg !25
  %972 = fadd float %968, %971, !dbg !25
  %973 = fadd float %972, %684, !dbg !25
  %974 = insertelement <1 x float> poison, float %973, i64 0, !dbg !6
  %975 = insertvalue [2 x <1 x float>] poison, <1 x float> %974, 0, !dbg !6
  %976 = bitcast float %964 to i32, !dbg !25
  %977 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %976), !dbg !25
  %978 = bitcast i32 %977 to float, !dbg !25
  %979 = fadd float %964, %978, !dbg !25
  %980 = bitcast float %979 to i32, !dbg !25
  %981 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %980), !dbg !25
  %982 = bitcast i32 %981 to float, !dbg !25
  %983 = fadd float %979, %982, !dbg !25
  %984 = fadd float %983, %692, !dbg !25
  %985 = insertelement <1 x float> poison, float %984, i64 0, !dbg !6
  %986 = insertvalue [2 x <1 x float>] %975, <1 x float> %985, 1, !dbg !6
  %987 = extractvalue [2 x <8 x float>] %126, 0, !dbg !27
  %988 = shufflevector <8 x float> %987, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %989 = fmul <4 x float> %988, %681, !dbg !27
  %990 = shufflevector <8 x float> %987, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %991 = fmul <4 x float> %990, %681, !dbg !27
  %992 = extractvalue [2 x <8 x float>] %126, 1, !dbg !27
  %993 = shufflevector <8 x float> %992, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %994 = fmul <4 x float> %993, %689, !dbg !27
  %995 = shufflevector <8 x float> %992, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %996 = fmul <4 x float> %995, %689, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local949 = shl nuw nsw i32 %address.thread328, 3, !dbg !14
  %address.local950 = and i32 %address.local949, 120, !dbg !14
  %address.local951 = shl nuw nsw i32 %address.thread328, 5, !dbg !14
  %address.local952 = and i32 %address.local951, 1536, !dbg !14
  %address.local953 = or disjoint i32 %address.local950, %address.local952, !dbg !14
  %address.local954 = zext nneg i32 %address.local953 to i64, !dbg !14
  %address.local955 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local954, !dbg !14
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !14
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !14
  %address.local956 = and i64 %address.local392, 60, !dbg !14
  %address.local957 = add nuw nsw i64 %address.local956, %address.local392, !dbg !14
  %address.local958 = shl nuw nsw i64 %address.local957, 3, !dbg !14
  %address.local959 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local958, !dbg !14
  %address.local960 = lshr i64 %address.local680, 4, !dbg !14
  %address.local961 = mul nuw nsw i64 %address.local960, 384, !dbg !14
  %address.local962 = getelementptr [2 x i8], ptr addrspace(3) %address.local959, i64 %address.local961, !dbg !14
  %address.local963 = mul nsw i64 %address.local670, -32, !dbg !14
  %address.local964 = getelementptr [2 x i8], ptr addrspace(3) %address.local962, i64 %address.local963, !dbg !14
  %address.local965 = getelementptr i8, ptr addrspace(3) %address.local964, i64 256, !dbg !14
  %999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local965, i32 0) #0, !dbg !14
  %1000 = bitcast <4 x i32> %999 to <8 x half>, !dbg !14
  %address.local966 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local958, !dbg !14
  %address.local967 = getelementptr [2 x i8], ptr addrspace(3) %address.local966, i64 %address.local961, !dbg !14
  %address.local968 = getelementptr [2 x i8], ptr addrspace(3) %address.local967, i64 %address.local963, !dbg !14
  %address.local969 = getelementptr i8, ptr addrspace(3) %address.local968, i64 512, !dbg !14
  %1001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !14
  %1002 = bitcast <4 x i32> %1001 to <8 x half>, !dbg !14
  %address.local970 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local958, !dbg !14
  %address.local971 = getelementptr [2 x i8], ptr addrspace(3) %address.local970, i64 %address.local961, !dbg !14
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) %address.local971, i64 %address.local963, !dbg !14
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local972, i64 768, !dbg !14
  %1003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14
  %1004 = bitcast <4 x i32> %1003 to <8 x half>, !dbg !14
  %address.local974 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local954, !dbg !14
  %address.local975 = getelementptr i8, ptr addrspace(3) %address.local974, i64 4096, !dbg !14
  %1005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !14
  %1006 = bitcast <4 x i32> %1005 to <8 x half>, !dbg !14
  %address.local976 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local958, !dbg !14
  %address.local977 = getelementptr [2 x i8], ptr addrspace(3) %address.local976, i64 %address.local961, !dbg !14
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) %address.local977, i64 %address.local963, !dbg !14
  %address.local979 = getelementptr i8, ptr addrspace(3) %address.local978, i64 4352, !dbg !14
  %1007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !14
  %1008 = bitcast <4 x i32> %1007 to <8 x half>, !dbg !14
  %address.local980 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local958, !dbg !14
  %address.local981 = getelementptr [2 x i8], ptr addrspace(3) %address.local980, i64 %address.local961, !dbg !14
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) %address.local981, i64 %address.local963, !dbg !14
  %address.local983 = getelementptr i8, ptr addrspace(3) %address.local982, i64 4608, !dbg !14
  %1009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !14
  %1010 = bitcast <4 x i32> %1009 to <8 x half>, !dbg !14
  %address.local984 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local958, !dbg !14
  %address.local985 = getelementptr [2 x i8], ptr addrspace(3) %address.local984, i64 %address.local961, !dbg !14
  %address.local986 = getelementptr [2 x i8], ptr addrspace(3) %address.local985, i64 %address.local963, !dbg !14
  %address.local987 = getelementptr i8, ptr addrspace(3) %address.local986, i64 4864, !dbg !14
  %1011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1012 = bitcast <4 x i32> %1011 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1013 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14241425 = add nsw i64 %230, %address.local939, !dbg !15
  %.idx14241425.tr = trunc i64 %.idx14241425 to i32, !dbg !15
  %dma.buffer.offset1225 = shl i32 %.idx14241425.tr, 1, !dbg !15
  %1014 = insertelement <2 x i64> poison, i64 %1013, i64 0, !dbg !15
  %1015 = bitcast <2 x i64> %1014 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1226 = shufflevector <4 x i32> %1015, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1016 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1225, <4 x i32> %dma.buffer.rsrc1226) #11, !dbg !15
  %1017 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1018 = trunc nuw i64 %230 to i32, !dbg !15
  %.idx14261427.tr = or disjoint i32 %address.local946, %1018, !dbg !15
  %dma.buffer.offset1223 = shl nuw nsw i32 %.idx14261427.tr, 1, !dbg !15
  %1019 = insertelement <2 x i64> poison, i64 %1017, i64 0, !dbg !15
  %1020 = bitcast <2 x i64> %1019 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1224 = shufflevector <4 x i32> %1020, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1021 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1223, <4 x i32> %dma.buffer.rsrc1224) #11, !dbg !15
  %1022 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1022, <4 x float> %989) #9, !dbg !14
  %1024 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1024, <4 x float> %1023) #9, !dbg !14
  %1026 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1026, <4 x float> %1025) #9, !dbg !14
  %1028 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1028, <4 x float> %1027) #9, !dbg !14
  %1030 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1030, <4 x float> %1029) #9, !dbg !14
  %1032 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1032, <4 x float> %1031) #9, !dbg !14
  %1034 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1034, <4 x float> %1033) #9, !dbg !14
  %1036 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1036, <4 x float> %1035) #9, !dbg !14
  %1038 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1038, <4 x float> %991) #9, !dbg !14
  %1040 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1040, <4 x float> %1039) #9, !dbg !14
  %1042 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1042, <4 x float> %1041) #9, !dbg !14
  %1044 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1044, <4 x float> %1043) #9, !dbg !14
  %1046 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1046, <4 x float> %1045) #9, !dbg !14
  %1048 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1048, <4 x float> %1047) #9, !dbg !14
  %1050 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1050, <4 x float> %1049) #9, !dbg !14
  %1052 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1052, <4 x float> %1051) #9, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1022, <4 x float> %994) #9, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1024, <4 x float> %1054) #9, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1026, <4 x float> %1055) #9, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1028, <4 x float> %1056) #9, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1030, <4 x float> %1057) #9, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1032, <4 x float> %1058) #9, !dbg !14
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1034, <4 x float> %1059) #9, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1036, <4 x float> %1060) #9, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1038, <4 x float> %996) #9, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1040, <4 x float> %1062) #9, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1042, <4 x float> %1063) #9, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1044, <4 x float> %1064) #9, !dbg !14
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1046, <4 x float> %1065) #9, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1048, <4 x float> %1066) #9, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1050, <4 x float> %1067) #9, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1052, <4 x float> %1068) #9, !dbg !14
  %1070 = extractvalue [2 x <8 x float>] %129, 0, !dbg !27
  %1071 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1072 = fmul <4 x float> %1071, %681, !dbg !27
  %1073 = shufflevector <8 x float> %1070, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1074 = fmul <4 x float> %1073, %681, !dbg !27
  %1075 = extractvalue [2 x <8 x float>] %129, 1, !dbg !27
  %1076 = shufflevector <8 x float> %1075, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1077 = fmul <4 x float> %1076, %689, !dbg !27
  %1078 = shufflevector <8 x float> %1075, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1079 = fmul <4 x float> %1078, %689, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1080 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !14
  %1081 = bitcast <4 x i32> %1080 to <8 x half>, !dbg !14
  %1082 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local965, i32 0) #0, !dbg !14
  %1083 = bitcast <4 x i32> %1082 to <8 x half>, !dbg !14
  %1084 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !14
  %1085 = bitcast <4 x i32> %1084 to <8 x half>, !dbg !14
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !14
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !14
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !14
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !14
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !14
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !14
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !14
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1096 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14281429 = add nsw i64 %320, %address.local939, !dbg !15
  %.idx14281429.tr = trunc i64 %.idx14281429 to i32, !dbg !15
  %dma.buffer.offset1221 = shl i32 %.idx14281429.tr, 1, !dbg !15
  %1097 = insertelement <2 x i64> poison, i64 %1096, i64 0, !dbg !15
  %1098 = bitcast <2 x i64> %1097 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1222 = shufflevector <4 x i32> %1098, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1099 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1221, <4 x i32> %dma.buffer.rsrc1222) #11, !dbg !15
  %1100 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1101 = trunc nuw i64 %320 to i32, !dbg !15
  %.idx14301431.tr = or disjoint i32 %address.local946, %1101, !dbg !15
  %dma.buffer.offset1219 = shl nuw nsw i32 %.idx14301431.tr, 1, !dbg !15
  %1102 = insertelement <2 x i64> poison, i64 %1100, i64 0, !dbg !15
  %1103 = bitcast <2 x i64> %1102 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1220 = shufflevector <4 x i32> %1103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1219, <4 x i32> %dma.buffer.rsrc1220) #11, !dbg !15
  %1105 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1105, <4 x float> %1072) #9, !dbg !14
  %1107 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1107, <4 x float> %1106) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1115, <4 x float> %1114) #9, !dbg !14
  %1117 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1117, <4 x float> %1116) #9, !dbg !14
  %1119 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1119, <4 x float> %1118) #9, !dbg !14
  %1121 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1121, <4 x float> %1074) #9, !dbg !14
  %1123 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1123, <4 x float> %1122) #9, !dbg !14
  %1125 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1125, <4 x float> %1124) #9, !dbg !14
  %1127 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1127, <4 x float> %1126) #9, !dbg !14
  %1129 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1129, <4 x float> %1128) #9, !dbg !14
  %1131 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1131, <4 x float> %1130) #9, !dbg !14
  %1133 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1133, <4 x float> %1132) #9, !dbg !14
  %1135 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1135, <4 x float> %1134) #9, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1105, <4 x float> %1077) #9, !dbg !14
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1107, <4 x float> %1137) #9, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1109, <4 x float> %1138) #9, !dbg !14
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1111, <4 x float> %1139) #9, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1113, <4 x float> %1140) #9, !dbg !14
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1115, <4 x float> %1141) #9, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1117, <4 x float> %1142) #9, !dbg !14
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1119, <4 x float> %1143) #9, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1121, <4 x float> %1079) #9, !dbg !14
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1123, <4 x float> %1145) #9, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1125, <4 x float> %1146) #9, !dbg !14
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1127, <4 x float> %1147) #9, !dbg !14
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1129, <4 x float> %1148) #9, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1131, <4 x float> %1149) #9, !dbg !14
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1133, <4 x float> %1150) #9, !dbg !14
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1135, <4 x float> %1151) #9, !dbg !14
  %1153 = extractvalue [2 x <8 x float>] %130, 0, !dbg !27
  %1154 = shufflevector <8 x float> %1153, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1155 = fmul <4 x float> %1154, %681, !dbg !27
  %1156 = shufflevector <8 x float> %1153, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1157 = fmul <4 x float> %1156, %681, !dbg !27
  %1158 = extractvalue [2 x <8 x float>] %130, 1, !dbg !27
  %1159 = shufflevector <8 x float> %1158, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1160 = fmul <4 x float> %1159, %689, !dbg !27
  %1161 = shufflevector <8 x float> %1158, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1162 = fmul <4 x float> %1161, %689, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1163 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !14
  %1164 = bitcast <4 x i32> %1163 to <8 x half>, !dbg !14
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local965, i32 0) #0, !dbg !14
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !14
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !14
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !14
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !14
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !14
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1179 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14321433 = add nsw i64 %410, %address.local939, !dbg !15
  %.idx14321433.tr = trunc i64 %.idx14321433 to i32, !dbg !15
  %dma.buffer.offset1217 = shl i32 %.idx14321433.tr, 1, !dbg !15
  %1180 = insertelement <2 x i64> poison, i64 %1179, i64 0, !dbg !15
  %1181 = bitcast <2 x i64> %1180 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1218 = shufflevector <4 x i32> %1181, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1182 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1217, <4 x i32> %dma.buffer.rsrc1218) #11, !dbg !15
  %1183 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.tr = trunc nuw i64 %410 to i32, !dbg !15
  %1184 = or disjoint i32 %address.local946, %.tr, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %1184, 1, !dbg !15
  %1185 = insertelement <2 x i64> poison, i64 %1183, i64 0, !dbg !15
  %1186 = bitcast <2 x i64> %1185 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %1186, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1187 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #11, !dbg !15
  %1188 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1188, <4 x float> %1155) #9, !dbg !14
  %1190 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1190, <4 x float> %1189) #9, !dbg !14
  %1192 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1192, <4 x float> %1191) #9, !dbg !14
  %1194 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1194, <4 x float> %1193) #9, !dbg !14
  %1196 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1196, <4 x float> %1195) #9, !dbg !14
  %1198 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1198, <4 x float> %1197) #9, !dbg !14
  %1200 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1200, <4 x float> %1199) #9, !dbg !14
  %1202 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1202, <4 x float> %1201) #9, !dbg !14
  %1204 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1204, <4 x float> %1157) #9, !dbg !14
  %1206 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1206, <4 x float> %1205) #9, !dbg !14
  %1208 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1208, <4 x float> %1207) #9, !dbg !14
  %1210 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1210, <4 x float> %1209) #9, !dbg !14
  %1212 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1212, <4 x float> %1211) #9, !dbg !14
  %1214 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1214, <4 x float> %1213) #9, !dbg !14
  %1216 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1216, <4 x float> %1215) #9, !dbg !14
  %1218 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1218, <4 x float> %1217) #9, !dbg !14
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1188, <4 x float> %1160) #9, !dbg !14
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1190, <4 x float> %1220) #9, !dbg !14
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1192, <4 x float> %1221) #9, !dbg !14
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1194, <4 x float> %1222) #9, !dbg !14
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1196, <4 x float> %1223) #9, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1198, <4 x float> %1224) #9, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1200, <4 x float> %1225) #9, !dbg !14
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1202, <4 x float> %1226) #9, !dbg !14
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1204, <4 x float> %1162) #9, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1206, <4 x float> %1228) #9, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1208, <4 x float> %1229) #9, !dbg !14
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1210, <4 x float> %1230) #9, !dbg !14
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1212, <4 x float> %1231) #9, !dbg !14
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1214, <4 x float> %1232) #9, !dbg !14
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1216, <4 x float> %1233) #9, !dbg !14
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1218, <4 x float> %1234) #9, !dbg !14
  %1236 = extractvalue [2 x <8 x float>] %131, 0, !dbg !27
  %1237 = shufflevector <8 x float> %1236, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1238 = fmul <4 x float> %1237, %681, !dbg !27
  %1239 = shufflevector <8 x float> %1236, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1240 = fmul <4 x float> %1239, %681, !dbg !27
  %1241 = extractvalue [2 x <8 x float>] %131, 1, !dbg !27
  %1242 = shufflevector <8 x float> %1241, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1243 = fmul <4 x float> %1242, %689, !dbg !27
  %1244 = shufflevector <8 x float> %1241, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1245 = fmul <4 x float> %1244, %689, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1246 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !14
  %1247 = bitcast <4 x i32> %1246 to <8 x half>, !dbg !14
  %1248 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local965, i32 0) #0, !dbg !14
  %1249 = bitcast <4 x i32> %1248 to <8 x half>, !dbg !14
  %1250 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local969, i32 0) #0, !dbg !14
  %1251 = bitcast <4 x i32> %1250 to <8 x half>, !dbg !14
  %1252 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14
  %1253 = bitcast <4 x i32> %1252 to <8 x half>, !dbg !14
  %1254 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local975, i32 0) #0, !dbg !14
  %1255 = bitcast <4 x i32> %1254 to <8 x half>, !dbg !14
  %1256 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local979, i32 0) #0, !dbg !14
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !14
  %1258 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local983, i32 0) #0, !dbg !14
  %1259 = bitcast <4 x i32> %1258 to <8 x half>, !dbg !14
  %1260 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1261 = bitcast <4 x i32> %1260 to <8 x half>, !dbg !14
  %1262 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1262, <4 x float> %1238) #9, !dbg !14
  %1264 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1264, <4 x float> %1263) #9, !dbg !14
  %1266 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1266, <4 x float> %1265) #9, !dbg !14
  %1268 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1268, <4 x float> %1267) #9, !dbg !14
  %1270 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1270, <4 x float> %1269) #9, !dbg !14
  %1272 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1272, <4 x float> %1271) #9, !dbg !14
  %1274 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1274, <4 x float> %1273) #9, !dbg !14
  %1276 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1276, <4 x float> %1275) #9, !dbg !14
  %1278 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1278, <4 x float> %1240) #9, !dbg !14
  %1280 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed220, <4 x half> %1280, <4 x float> %1279) #9, !dbg !14
  %1282 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed222, <4 x half> %1282, <4 x float> %1281) #9, !dbg !14
  %1284 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed224, <4 x half> %1284, <4 x float> %1283) #9, !dbg !14
  %1286 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed226, <4 x half> %1286, <4 x float> %1285) #9, !dbg !14
  %1288 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed228, <4 x half> %1288, <4 x float> %1287) #9, !dbg !14
  %1290 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed230, <4 x half> %1290, <4 x float> %1289) #9, !dbg !14
  %1292 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed232, <4 x half> %1292, <4 x float> %1291) #9, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1262, <4 x float> %1243) #9, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1264, <4 x float> %1294) #9, !dbg !14
  %1296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1266, <4 x float> %1295) #9, !dbg !14
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1268, <4 x float> %1296) #9, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1270, <4 x float> %1297) #9, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1272, <4 x float> %1298) #9, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1274, <4 x float> %1299) #9, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1276, <4 x float> %1300) #9, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed234, <4 x half> %1278, <4 x float> %1245) #9, !dbg !14
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed236, <4 x half> %1280, <4 x float> %1302) #9, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed238, <4 x half> %1282, <4 x float> %1303) #9, !dbg !14
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed240, <4 x half> %1284, <4 x float> %1304) #9, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1286, <4 x float> %1305) #9, !dbg !14
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1288, <4 x float> %1306) #9, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1290, <4 x float> %1307) #9, !dbg !14
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1292, <4 x float> %1308) #9, !dbg !14
  %1310 = shufflevector <4 x float> %1037, <4 x float> %1053, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1311 = insertvalue [2 x <8 x float>] poison, <8 x float> %1310, 0, !dbg !6
  %1312 = shufflevector <4 x float> %1061, <4 x float> %1069, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1313 = insertvalue [2 x <8 x float>] %1311, <8 x float> %1312, 1, !dbg !6
  %1314 = insertvalue [2 x <1 x float>] poison, <1 x float> %655, 0, !dbg !6
  %1315 = insertvalue [2 x <1 x float>] %1314, <1 x float> %658, 1, !dbg !6
  %1316 = shufflevector <4 x float> %1120, <4 x float> %1136, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1317 = insertvalue [2 x <8 x float>] poison, <8 x float> %1316, 0, !dbg !6
  %1318 = shufflevector <4 x float> %1144, <4 x float> %1152, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1319 = insertvalue [2 x <8 x float>] %1317, <8 x float> %1318, 1, !dbg !6
  %1320 = shufflevector <4 x float> %1203, <4 x float> %1219, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1321 = insertvalue [2 x <8 x float>] poison, <8 x float> %1320, 0, !dbg !6
  %1322 = shufflevector <4 x float> %1227, <4 x float> %1235, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1323 = insertvalue [2 x <8 x float>] %1321, <8 x float> %1322, 1, !dbg !6
  %1324 = shufflevector <4 x float> %1277, <4 x float> %1293, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1325 = insertvalue [2 x <8 x float>] poison, <8 x float> %1324, 0, !dbg !6
  %1326 = shufflevector <4 x float> %1301, <4 x float> %1309, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1327 = insertvalue [2 x <8 x float>] %1325, <8 x float> %1326, 1, !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %1328 = add nuw nsw i64 %125, 1, !dbg !6
  br label %124, !dbg !6

1329:                                             ; preds = %124
  %1330 = extractvalue [2 x <1 x float>] %127, 0, !dbg !17
  %1331 = shufflevector <1 x float> %1330, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1332 = extractvalue [2 x <1 x float>] %127, 1, !dbg !17
  %1333 = shufflevector <1 x float> %1332, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1334 = extractvalue [2 x <8 x float>] %126, 1, !dbg !17
  %1335 = shufflevector <8 x float> %1334, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1336 = fdiv <4 x float> %1335, %1333, !dbg !17
  %1337 = fptrunc <4 x float> %1336 to <4 x half>, !dbg !18
  %1338 = extractelement <4 x half> %1337, i64 3, !dbg !18
  %1339 = extractelement <4 x half> %1337, i64 2, !dbg !18
  %1340 = extractelement <4 x half> %1337, i64 1, !dbg !18
  %1341 = extractelement <4 x half> %1337, i64 0, !dbg !18
  %1342 = shufflevector <8 x float> %1334, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1343 = fdiv <4 x float> %1342, %1333, !dbg !17
  %1344 = fptrunc <4 x float> %1343 to <4 x half>, !dbg !18
  %1345 = extractelement <4 x half> %1344, i64 3, !dbg !18
  %1346 = extractelement <4 x half> %1344, i64 2, !dbg !18
  %1347 = extractelement <4 x half> %1344, i64 1, !dbg !18
  %1348 = extractelement <4 x half> %1344, i64 0, !dbg !18
  %1349 = extractvalue [2 x <8 x float>] %126, 0, !dbg !17
  %1350 = shufflevector <8 x float> %1349, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1351 = fdiv <4 x float> %1350, %1331, !dbg !17
  %1352 = fptrunc <4 x float> %1351 to <4 x half>, !dbg !18
  %1353 = extractelement <4 x half> %1352, i64 3, !dbg !18
  %1354 = extractelement <4 x half> %1352, i64 2, !dbg !18
  %1355 = extractelement <4 x half> %1352, i64 1, !dbg !18
  %1356 = extractelement <4 x half> %1352, i64 0, !dbg !18
  %1357 = shufflevector <8 x float> %1349, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1358 = fdiv <4 x float> %1357, %1331, !dbg !17
  %1359 = fptrunc <4 x float> %1358 to <4 x half>, !dbg !18
  %1360 = extractelement <4 x half> %1359, i64 3, !dbg !18
  %1361 = extractelement <4 x half> %1359, i64 2, !dbg !18
  %1362 = extractelement <4 x half> %1359, i64 1, !dbg !18
  %1363 = extractelement <4 x half> %1359, i64 0, !dbg !18
  %1364 = shl nuw nsw i64 %26, 19, !dbg !19
  %address.thread994 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local995 = zext nneg i32 %address.thread994 to i64, !dbg !13
  %address.local996 = add nsw i64 %27, %address.local995, !dbg !19
  %1365 = lshr i64 %address.local995, 2, !dbg !19
  %address.local998 = and i64 %1365, 48, !dbg !19
  %address.local999 = add nsw i64 %address.local996, %address.local998, !dbg !19
  %address.local1000 = and i64 %address.local995, 240, !dbg !19
  %address.local1001 = sub nsw i64 %address.local999, %address.local1000, !dbg !19
  %address.local1002 = add nsw i64 %address.local1001, %28, !dbg !19
  %address.local1003 = shl nuw nsw i64 %address.local1002, 7, !dbg !19
  %address.local1004 = add nuw nsw i64 %1364, %address.local1003, !dbg !19
  %address.local1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1006 = lshr i64 %address.local995, 4, !dbg !13
  %address.local1007 = and i64 %address.local1006, 3, !dbg !13
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1005, i64 %address.local1007, !dbg !19
  store half %1363, ptr addrspace(1) %address.local1008, align 2, !dbg !19
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1010 = or disjoint i64 %address.local1007, 4, !dbg !19
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1009, i64 %address.local1010, !dbg !19
  store half %1362, ptr addrspace(1) %address.local1011, align 2, !dbg !19
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1013 = or disjoint i64 %address.local1007, 8, !dbg !19
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1012, i64 %address.local1013, !dbg !19
  store half %1361, ptr addrspace(1) %address.local1014, align 2, !dbg !19
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1016 = or i64 %address.local1006, 12, !dbg !19
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1015, i64 %address.local1016, !dbg !19
  store half %1360, ptr addrspace(1) %address.local1017, align 2, !dbg !19
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1019 = or disjoint i64 %address.local1007, 16, !dbg !19
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1018, i64 %address.local1019, !dbg !19
  store half %1356, ptr addrspace(1) %address.local1020, align 2, !dbg !19
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1022 = or disjoint i64 %address.local1007, 20, !dbg !19
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1021, i64 %address.local1022, !dbg !19
  store half %1355, ptr addrspace(1) %address.local1023, align 2, !dbg !19
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1025 = or disjoint i64 %address.local1007, 24, !dbg !19
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1024, i64 %address.local1025, !dbg !19
  store half %1354, ptr addrspace(1) %address.local1026, align 2, !dbg !19
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1004, !dbg !19
  %address.local1028 = or i64 %address.local1006, 28, !dbg !19
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1028, !dbg !19
  store half %1353, ptr addrspace(1) %address.local1029, align 2, !dbg !19
  %address.local1030 = shl nsw i64 %address.local1002, 7, !dbg !19
  %address.local1031 = add i64 %address.local1030, 8192, !dbg !19
  %address.local1032 = add nuw nsw i64 %1364, %address.local1031, !dbg !19
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1033, i64 %address.local1007, !dbg !19
  store half %1348, ptr addrspace(1) %address.local1034, align 2, !dbg !19
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1010, !dbg !19
  store half %1347, ptr addrspace(1) %address.local1036, align 2, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1013, !dbg !19
  store half %1346, ptr addrspace(1) %address.local1038, align 2, !dbg !19
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1016, !dbg !19
  store half %1345, ptr addrspace(1) %address.local1040, align 2, !dbg !19
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1019, !dbg !19
  store half %1341, ptr addrspace(1) %address.local1042, align 2, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1022, !dbg !19
  store half %1340, ptr addrspace(1) %address.local1044, align 2, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1025, !dbg !19
  store half %1339, ptr addrspace(1) %address.local1046, align 2, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1032, !dbg !19
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1047, i64 %address.local1028, !dbg !19
  store half %1338, ptr addrspace(1) %address.local1048, align 2, !dbg !19
  %1366 = extractvalue [2 x <8 x float>] %129, 1, !dbg !17
  %1367 = shufflevector <8 x float> %1366, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1368 = fdiv <4 x float> %1367, %1333, !dbg !17
  %1369 = fptrunc <4 x float> %1368 to <4 x half>, !dbg !18
  %1370 = extractelement <4 x half> %1369, i64 3, !dbg !18
  %1371 = extractelement <4 x half> %1369, i64 2, !dbg !18
  %1372 = extractelement <4 x half> %1369, i64 1, !dbg !18
  %1373 = extractelement <4 x half> %1369, i64 0, !dbg !18
  %1374 = shufflevector <8 x float> %1366, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1375 = fdiv <4 x float> %1374, %1333, !dbg !17
  %1376 = fptrunc <4 x float> %1375 to <4 x half>, !dbg !18
  %1377 = extractelement <4 x half> %1376, i64 3, !dbg !18
  %1378 = extractelement <4 x half> %1376, i64 2, !dbg !18
  %1379 = extractelement <4 x half> %1376, i64 1, !dbg !18
  %1380 = extractelement <4 x half> %1376, i64 0, !dbg !18
  %1381 = extractvalue [2 x <8 x float>] %129, 0, !dbg !17
  %1382 = shufflevector <8 x float> %1381, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1383 = fdiv <4 x float> %1382, %1331, !dbg !17
  %1384 = fptrunc <4 x float> %1383 to <4 x half>, !dbg !18
  %1385 = extractelement <4 x half> %1384, i64 3, !dbg !18
  %1386 = extractelement <4 x half> %1384, i64 2, !dbg !18
  %1387 = extractelement <4 x half> %1384, i64 1, !dbg !18
  %1388 = extractelement <4 x half> %1384, i64 0, !dbg !18
  %1389 = shufflevector <8 x float> %1381, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1390 = fdiv <4 x float> %1389, %1331, !dbg !17
  %1391 = fptrunc <4 x float> %1390 to <4 x half>, !dbg !18
  %1392 = extractelement <4 x half> %1391, i64 3, !dbg !18
  %1393 = extractelement <4 x half> %1391, i64 2, !dbg !18
  %1394 = extractelement <4 x half> %1391, i64 1, !dbg !18
  %1395 = extractelement <4 x half> %1391, i64 0, !dbg !18
  %1396 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1049 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1050 = zext nneg i32 %address.thread1049 to i64, !dbg !13
  %address.local1051 = add nsw i64 %27, %address.local1050, !dbg !19
  %1397 = lshr i64 %address.local1050, 2, !dbg !19
  %address.local1053 = and i64 %1397, 48, !dbg !19
  %address.local1054 = add nsw i64 %address.local1051, %address.local1053, !dbg !19
  %address.local1055 = and i64 %address.local1050, 240, !dbg !19
  %address.local1056 = sub nsw i64 %address.local1054, %address.local1055, !dbg !19
  %address.local1057 = add nsw i64 %address.local1056, %28, !dbg !19
  %address.local1058 = shl nuw nsw i64 %address.local1057, 7, !dbg !19
  %address.local1059 = add nuw nsw i64 %1396, %address.local1058, !dbg !19
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1061 = lshr i64 %address.local1050, 4, !dbg !13
  %address.local1062 = and i64 %address.local1061, 3, !dbg !13
  %address.local1063 = or disjoint i64 %address.local1062, 32, !dbg !19
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1060, i64 %address.local1063, !dbg !19
  store half %1395, ptr addrspace(1) %address.local1064, align 2, !dbg !19
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1066 = or disjoint i64 %address.local1062, 36, !dbg !19
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1065, i64 %address.local1066, !dbg !19
  store half %1394, ptr addrspace(1) %address.local1067, align 2, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1069 = or disjoint i64 %address.local1062, 40, !dbg !19
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1068, i64 %address.local1069, !dbg !19
  store half %1393, ptr addrspace(1) %address.local1070, align 2, !dbg !19
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1072 = or i64 %address.local1061, 44, !dbg !19
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1072, !dbg !19
  store half %1392, ptr addrspace(1) %address.local1073, align 2, !dbg !19
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1075 = or disjoint i64 %address.local1062, 48, !dbg !19
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1074, i64 %address.local1075, !dbg !19
  store half %1388, ptr addrspace(1) %address.local1076, align 2, !dbg !19
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1078 = or disjoint i64 %address.local1062, 52, !dbg !19
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1078, !dbg !19
  store half %1387, ptr addrspace(1) %address.local1079, align 2, !dbg !19
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1081 = or disjoint i64 %address.local1062, 56, !dbg !19
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1080, i64 %address.local1081, !dbg !19
  store half %1386, ptr addrspace(1) %address.local1082, align 2, !dbg !19
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1059, !dbg !19
  %address.local1084 = or i64 %address.local1061, 60, !dbg !19
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1084, !dbg !19
  store half %1385, ptr addrspace(1) %address.local1085, align 2, !dbg !19
  %address.local1086 = shl nsw i64 %address.local1057, 7, !dbg !19
  %address.local1087 = add i64 %address.local1086, 8192, !dbg !19
  %address.local1088 = add nuw nsw i64 %1396, %address.local1087, !dbg !19
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1089, i64 %address.local1063, !dbg !19
  store half %1380, ptr addrspace(1) %address.local1090, align 2, !dbg !19
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1091, i64 %address.local1066, !dbg !19
  store half %1379, ptr addrspace(1) %address.local1092, align 2, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1069, !dbg !19
  store half %1378, ptr addrspace(1) %address.local1094, align 2, !dbg !19
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1095, i64 %address.local1072, !dbg !19
  store half %1377, ptr addrspace(1) %address.local1096, align 2, !dbg !19
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1097, i64 %address.local1075, !dbg !19
  store half %1373, ptr addrspace(1) %address.local1098, align 2, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1078, !dbg !19
  store half %1372, ptr addrspace(1) %address.local1100, align 2, !dbg !19
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1101, i64 %address.local1081, !dbg !19
  store half %1371, ptr addrspace(1) %address.local1102, align 2, !dbg !19
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1088, !dbg !19
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1103, i64 %address.local1084, !dbg !19
  store half %1370, ptr addrspace(1) %address.local1104, align 2, !dbg !19
  %1398 = extractvalue [2 x <8 x float>] %130, 1, !dbg !17
  %1399 = shufflevector <8 x float> %1398, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1400 = fdiv <4 x float> %1399, %1333, !dbg !17
  %1401 = fptrunc <4 x float> %1400 to <4 x half>, !dbg !18
  %1402 = extractelement <4 x half> %1401, i64 3, !dbg !18
  %1403 = extractelement <4 x half> %1401, i64 2, !dbg !18
  %1404 = extractelement <4 x half> %1401, i64 1, !dbg !18
  %1405 = extractelement <4 x half> %1401, i64 0, !dbg !18
  %1406 = shufflevector <8 x float> %1398, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1407 = fdiv <4 x float> %1406, %1333, !dbg !17
  %1408 = fptrunc <4 x float> %1407 to <4 x half>, !dbg !18
  %1409 = extractelement <4 x half> %1408, i64 3, !dbg !18
  %1410 = extractelement <4 x half> %1408, i64 2, !dbg !18
  %1411 = extractelement <4 x half> %1408, i64 1, !dbg !18
  %1412 = extractelement <4 x half> %1408, i64 0, !dbg !18
  %1413 = extractvalue [2 x <8 x float>] %130, 0, !dbg !17
  %1414 = shufflevector <8 x float> %1413, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1415 = fdiv <4 x float> %1414, %1331, !dbg !17
  %1416 = fptrunc <4 x float> %1415 to <4 x half>, !dbg !18
  %1417 = extractelement <4 x half> %1416, i64 3, !dbg !18
  %1418 = extractelement <4 x half> %1416, i64 2, !dbg !18
  %1419 = extractelement <4 x half> %1416, i64 1, !dbg !18
  %1420 = extractelement <4 x half> %1416, i64 0, !dbg !18
  %1421 = shufflevector <8 x float> %1413, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1422 = fdiv <4 x float> %1421, %1331, !dbg !17
  %1423 = fptrunc <4 x float> %1422 to <4 x half>, !dbg !18
  %1424 = extractelement <4 x half> %1423, i64 3, !dbg !18
  %1425 = extractelement <4 x half> %1423, i64 2, !dbg !18
  %1426 = extractelement <4 x half> %1423, i64 1, !dbg !18
  %1427 = extractelement <4 x half> %1423, i64 0, !dbg !18
  %1428 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1105 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1106 = zext nneg i32 %address.thread1105 to i64, !dbg !13
  %address.local1107 = add nsw i64 %27, %address.local1106, !dbg !19
  %1429 = lshr i64 %address.local1106, 2, !dbg !19
  %address.local1109 = and i64 %1429, 48, !dbg !19
  %address.local1110 = add nsw i64 %address.local1107, %address.local1109, !dbg !19
  %address.local1111 = and i64 %address.local1106, 240, !dbg !19
  %address.local1112 = sub nsw i64 %address.local1110, %address.local1111, !dbg !19
  %address.local1113 = add nsw i64 %address.local1112, %28, !dbg !19
  %address.local1114 = shl nuw nsw i64 %address.local1113, 7, !dbg !19
  %address.local1115 = add nuw nsw i64 %1428, %address.local1114, !dbg !19
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1117 = lshr i64 %address.local1106, 4, !dbg !13
  %address.local1118 = and i64 %address.local1117, 3, !dbg !13
  %address.local1119 = or disjoint i64 %address.local1118, 64, !dbg !19
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1116, i64 %address.local1119, !dbg !19
  store half %1427, ptr addrspace(1) %address.local1120, align 2, !dbg !19
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1122 = or disjoint i64 %address.local1118, 68, !dbg !19
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1122, !dbg !19
  store half %1426, ptr addrspace(1) %address.local1123, align 2, !dbg !19
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1125 = or disjoint i64 %address.local1118, 72, !dbg !19
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1124, i64 %address.local1125, !dbg !19
  store half %1425, ptr addrspace(1) %address.local1126, align 2, !dbg !19
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1128 = or i64 %address.local1117, 76, !dbg !19
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1127, i64 %address.local1128, !dbg !19
  store half %1424, ptr addrspace(1) %address.local1129, align 2, !dbg !19
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1131 = or disjoint i64 %address.local1118, 80, !dbg !19
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1130, i64 %address.local1131, !dbg !19
  store half %1420, ptr addrspace(1) %address.local1132, align 2, !dbg !19
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1134 = or disjoint i64 %address.local1118, 84, !dbg !19
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1133, i64 %address.local1134, !dbg !19
  store half %1419, ptr addrspace(1) %address.local1135, align 2, !dbg !19
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1137 = or disjoint i64 %address.local1118, 88, !dbg !19
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1136, i64 %address.local1137, !dbg !19
  store half %1418, ptr addrspace(1) %address.local1138, align 2, !dbg !19
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1115, !dbg !19
  %address.local1140 = or i64 %address.local1117, 92, !dbg !19
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1139, i64 %address.local1140, !dbg !19
  store half %1417, ptr addrspace(1) %address.local1141, align 2, !dbg !19
  %address.local1142 = shl nsw i64 %address.local1113, 7, !dbg !19
  %address.local1143 = add i64 %address.local1142, 8192, !dbg !19
  %address.local1144 = add nuw nsw i64 %1428, %address.local1143, !dbg !19
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1145, i64 %address.local1119, !dbg !19
  store half %1412, ptr addrspace(1) %address.local1146, align 2, !dbg !19
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1147, i64 %address.local1122, !dbg !19
  store half %1411, ptr addrspace(1) %address.local1148, align 2, !dbg !19
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1125, !dbg !19
  store half %1410, ptr addrspace(1) %address.local1150, align 2, !dbg !19
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1128, !dbg !19
  store half %1409, ptr addrspace(1) %address.local1152, align 2, !dbg !19
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1131, !dbg !19
  store half %1405, ptr addrspace(1) %address.local1154, align 2, !dbg !19
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1134, !dbg !19
  store half %1404, ptr addrspace(1) %address.local1156, align 2, !dbg !19
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1137, !dbg !19
  store half %1403, ptr addrspace(1) %address.local1158, align 2, !dbg !19
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1144, !dbg !19
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1140, !dbg !19
  store half %1402, ptr addrspace(1) %address.local1160, align 2, !dbg !19
  %1430 = extractvalue [2 x <8 x float>] %131, 1, !dbg !17
  %1431 = shufflevector <8 x float> %1430, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1432 = fdiv <4 x float> %1431, %1333, !dbg !17
  %1433 = fptrunc <4 x float> %1432 to <4 x half>, !dbg !18
  %1434 = extractelement <4 x half> %1433, i64 3, !dbg !18
  %1435 = extractelement <4 x half> %1433, i64 2, !dbg !18
  %1436 = extractelement <4 x half> %1433, i64 1, !dbg !18
  %1437 = extractelement <4 x half> %1433, i64 0, !dbg !18
  %1438 = shufflevector <8 x float> %1430, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1439 = fdiv <4 x float> %1438, %1333, !dbg !17
  %1440 = fptrunc <4 x float> %1439 to <4 x half>, !dbg !18
  %1441 = extractelement <4 x half> %1440, i64 3, !dbg !18
  %1442 = extractelement <4 x half> %1440, i64 2, !dbg !18
  %1443 = extractelement <4 x half> %1440, i64 1, !dbg !18
  %1444 = extractelement <4 x half> %1440, i64 0, !dbg !18
  %1445 = extractvalue [2 x <8 x float>] %131, 0, !dbg !17
  %1446 = shufflevector <8 x float> %1445, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %1447 = fdiv <4 x float> %1446, %1331, !dbg !17
  %1448 = fptrunc <4 x float> %1447 to <4 x half>, !dbg !18
  %1449 = extractelement <4 x half> %1448, i64 3, !dbg !18
  %1450 = extractelement <4 x half> %1448, i64 2, !dbg !18
  %1451 = extractelement <4 x half> %1448, i64 1, !dbg !18
  %1452 = extractelement <4 x half> %1448, i64 0, !dbg !18
  %1453 = shufflevector <8 x float> %1445, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %1454 = fdiv <4 x float> %1453, %1331, !dbg !17
  %1455 = fptrunc <4 x float> %1454 to <4 x half>, !dbg !18
  %1456 = extractelement <4 x half> %1455, i64 3, !dbg !18
  %1457 = extractelement <4 x half> %1455, i64 2, !dbg !18
  %1458 = extractelement <4 x half> %1455, i64 1, !dbg !18
  %1459 = extractelement <4 x half> %1455, i64 0, !dbg !18
  %1460 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1161 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1162 = zext nneg i32 %address.thread1161 to i64, !dbg !13
  %address.local1163 = add nsw i64 %27, %address.local1162, !dbg !19
  %1461 = lshr i64 %address.local1162, 2, !dbg !19
  %address.local1165 = and i64 %1461, 48, !dbg !19
  %address.local1166 = add nsw i64 %address.local1163, %address.local1165, !dbg !19
  %address.local1167 = and i64 %address.local1162, 240, !dbg !19
  %address.local1168 = sub nsw i64 %address.local1166, %address.local1167, !dbg !19
  %address.local1169 = add nsw i64 %address.local1168, %28, !dbg !19
  %address.local1170 = shl nuw nsw i64 %address.local1169, 7, !dbg !19
  %address.local1171 = add nuw nsw i64 %1460, %address.local1170, !dbg !19
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1173 = lshr i64 %address.local1162, 4, !dbg !13
  %address.local1174 = and i64 %address.local1173, 3, !dbg !13
  %address.local1175 = or disjoint i64 %address.local1174, 96, !dbg !19
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1172, i64 %address.local1175, !dbg !19
  store half %1459, ptr addrspace(1) %address.local1176, align 2, !dbg !19
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1178 = or disjoint i64 %address.local1174, 100, !dbg !19
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1178, !dbg !19
  store half %1458, ptr addrspace(1) %address.local1179, align 2, !dbg !19
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1181 = or disjoint i64 %address.local1174, 104, !dbg !19
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1180, i64 %address.local1181, !dbg !19
  store half %1457, ptr addrspace(1) %address.local1182, align 2, !dbg !19
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1184 = or i64 %address.local1173, 108, !dbg !19
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1183, i64 %address.local1184, !dbg !19
  store half %1456, ptr addrspace(1) %address.local1185, align 2, !dbg !19
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1187 = or disjoint i64 %address.local1174, 112, !dbg !19
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1186, i64 %address.local1187, !dbg !19
  store half %1452, ptr addrspace(1) %address.local1188, align 2, !dbg !19
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1190 = or disjoint i64 %address.local1174, 116, !dbg !19
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1189, i64 %address.local1190, !dbg !19
  store half %1451, ptr addrspace(1) %address.local1191, align 2, !dbg !19
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1193 = or disjoint i64 %address.local1174, 120, !dbg !19
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1192, i64 %address.local1193, !dbg !19
  store half %1450, ptr addrspace(1) %address.local1194, align 2, !dbg !19
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1171, !dbg !19
  %address.local1196 = or i64 %address.local1173, 124, !dbg !19
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1195, i64 %address.local1196, !dbg !19
  store half %1449, ptr addrspace(1) %address.local1197, align 2, !dbg !19
  %address.local1198 = shl nsw i64 %address.local1169, 7, !dbg !19
  %address.local1199 = add i64 %address.local1198, 8192, !dbg !19
  %address.local1200 = add nuw nsw i64 %1460, %address.local1199, !dbg !19
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1201, i64 %address.local1175, !dbg !19
  store half %1444, ptr addrspace(1) %address.local1202, align 2, !dbg !19
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1203, i64 %address.local1178, !dbg !19
  store half %1443, ptr addrspace(1) %address.local1204, align 2, !dbg !19
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1181, !dbg !19
  store half %1442, ptr addrspace(1) %address.local1206, align 2, !dbg !19
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1207, i64 %address.local1184, !dbg !19
  store half %1441, ptr addrspace(1) %address.local1208, align 2, !dbg !19
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1209, i64 %address.local1187, !dbg !19
  store half %1437, ptr addrspace(1) %address.local1210, align 2, !dbg !19
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1190, !dbg !19
  store half %1436, ptr addrspace(1) %address.local1212, align 2, !dbg !19
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1213, i64 %address.local1193, !dbg !19
  store half %1435, ptr addrspace(1) %address.local1214, align 2, !dbg !19
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1200, !dbg !19
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1215, i64 %address.local1196, !dbg !19
  store half %1434, ptr addrspace(1) %address.local1216, align 2, !dbg !19
  ret void, !dbg !5
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind memory(none) }
attributes #11 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!29}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 74, column: 20, scope: !4)
!11 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 69, column: 24, scope: !4)
!14 = !DILocation(line: 89, column: 20, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 71, column: 23, scope: !4)
!17 = !DILocation(line: 92, column: 23, scope: !4)
!18 = !DILocation(line: 93, column: 19, scope: !4)
!19 = !DILocation(line: 94, column: 9, scope: !4)
!20 = !DILocation(line: 75, column: 22, scope: !4)
!21 = !DILocation(line: 76, column: 23, scope: !4)
!22 = !DILocation(line: 77, column: 20, scope: !4)
!23 = !DILocation(line: 78, column: 22, scope: !4)
!24 = !DILocation(line: 79, column: 22, scope: !4)
!25 = !DILocation(line: 80, column: 20, scope: !4)
!26 = !DILocation(line: 81, column: 16, scope: !4)
!27 = !DILocation(line: 88, column: 22, scope: !4)
!28 = !{i32 256, i32 1, i32 1}
!29 = !{i32 2, !"Debug Info Version", i32 3}
!30 = !{i32 0, i32 256}
