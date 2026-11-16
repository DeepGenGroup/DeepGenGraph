; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir"

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
  %address.local250 = lshr i64 %address.local, 6, !dbg !11
  %address.local251 = shl nuw nsw i64 %address.local250, 7, !dbg !11
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local251, !dbg !11
  %address.local253 = ptrtoint ptr addrspace(3) %address.local252 to i64, !dbg !11
  %address.local254 = trunc i64 %address.local253 to i32, !dbg !11
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local254), !dbg !11
  %39 = zext i32 %38 to i64, !dbg !11
  %40 = inttoptr i64 %39 to ptr addrspace(3), !dbg !11
  %address.local255 = lshr i64 %address.local, 6, !dbg !11
  %address.local256 = or disjoint i64 %address.local255, 4, !dbg !11
  %address.local257 = shl nuw nsw i64 %address.local256, 7, !dbg !11
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local257, !dbg !11
  %address.local259 = ptrtoint ptr addrspace(3) %address.local258 to i64, !dbg !11
  %address.local260 = trunc i64 %address.local259 to i32, !dbg !11
  %41 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local260), !dbg !11
  %42 = zext i32 %41 to i64, !dbg !11
  %43 = inttoptr i64 %42 to ptr addrspace(3), !dbg !11
  %address.local261 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local262 = and i64 %address.local261, 384, !dbg !11
  %address.local263 = or disjoint i64 %address.local262, 1024, !dbg !11
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local263, !dbg !11
  %address.local265 = ptrtoint ptr addrspace(3) %address.local264 to i64, !dbg !11
  %address.local266 = trunc i64 %address.local265 to i32, !dbg !11
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local266), !dbg !11
  %45 = zext i32 %44 to i64, !dbg !11
  %46 = inttoptr i64 %45 to ptr addrspace(3), !dbg !11
  %address.local267 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local268 = and i64 %address.local267, 384, !dbg !11
  %address.local269 = or disjoint i64 %address.local268, 1536, !dbg !11
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local269, !dbg !11
  %address.local271 = ptrtoint ptr addrspace(3) %address.local270 to i64, !dbg !11
  %address.local272 = trunc i64 %address.local271 to i32, !dbg !11
  %47 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local272), !dbg !11
  %48 = zext i32 %47 to i64, !dbg !11
  %49 = inttoptr i64 %48 to ptr addrspace(3), !dbg !11
  %address.local273 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local274 = and i64 %address.local273, 384, !dbg !11
  %address.local275 = or disjoint i64 %address.local274, 2048, !dbg !11
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local275, !dbg !11
  %address.local277 = ptrtoint ptr addrspace(3) %address.local276 to i64, !dbg !11
  %address.local278 = trunc i64 %address.local277 to i32, !dbg !11
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local278), !dbg !11
  %51 = zext i32 %50 to i64, !dbg !11
  %52 = inttoptr i64 %51 to ptr addrspace(3), !dbg !11
  %address.local279 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local280 = and i64 %address.local279, 384, !dbg !11
  %address.local281 = or disjoint i64 %address.local280, 2560, !dbg !11
  %address.local282 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local281, !dbg !11
  %address.local283 = ptrtoint ptr addrspace(3) %address.local282 to i64, !dbg !11
  %address.local284 = trunc i64 %address.local283 to i32, !dbg !11
  %53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local284), !dbg !11
  %54 = zext i32 %53 to i64, !dbg !11
  %55 = inttoptr i64 %54 to ptr addrspace(3), !dbg !11
  %address.local285 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local286 = and i64 %address.local285, 384, !dbg !11
  %address.local287 = or disjoint i64 %address.local286, 3072, !dbg !11
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local287, !dbg !11
  %address.local289 = ptrtoint ptr addrspace(3) %address.local288 to i64, !dbg !11
  %address.local290 = trunc i64 %address.local289 to i32, !dbg !11
  %56 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local290), !dbg !11
  %57 = zext i32 %56 to i64, !dbg !11
  %58 = inttoptr i64 %57 to ptr addrspace(3), !dbg !11
  %address.local291 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local292 = and i64 %address.local291, 384, !dbg !11
  %address.local293 = or disjoint i64 %address.local292, 3584, !dbg !11
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local293, !dbg !11
  %address.local295 = ptrtoint ptr addrspace(3) %address.local294 to i64, !dbg !11
  %address.local296 = trunc i64 %address.local295 to i32, !dbg !11
  %59 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !11
  %60 = zext i32 %59 to i64, !dbg !11
  %61 = inttoptr i64 %60 to ptr addrspace(3), !dbg !11
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local251, !dbg !12
  %address.local298 = ptrtoint ptr addrspace(3) %address.local297 to i64, !dbg !12
  %address.local299 = trunc i64 %address.local298 to i32, !dbg !12
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local299), !dbg !12
  %63 = zext i32 %62 to i64, !dbg !12
  %64 = inttoptr i64 %63 to ptr addrspace(3), !dbg !12
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local257, !dbg !12
  %address.local301 = ptrtoint ptr addrspace(3) %address.local300 to i64, !dbg !12
  %address.local302 = trunc i64 %address.local301 to i32, !dbg !12
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local302), !dbg !12
  %66 = zext i32 %65 to i64, !dbg !12
  %67 = inttoptr i64 %66 to ptr addrspace(3), !dbg !12
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local263, !dbg !12
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !12
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !12
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !12
  %69 = zext i32 %68 to i64, !dbg !12
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !12
  %address.local306 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !12
  %address.local307 = ptrtoint ptr addrspace(3) %address.local306 to i64, !dbg !12
  %address.local308 = trunc i64 %address.local307 to i32, !dbg !12
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local308), !dbg !12
  %72 = zext i32 %71 to i64, !dbg !12
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !12
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local275, !dbg !12
  %address.local310 = ptrtoint ptr addrspace(3) %address.local309 to i64, !dbg !12
  %address.local311 = trunc i64 %address.local310 to i32, !dbg !12
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local311), !dbg !12
  %75 = zext i32 %74 to i64, !dbg !12
  %76 = inttoptr i64 %75 to ptr addrspace(3), !dbg !12
  %address.local312 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local281, !dbg !12
  %address.local313 = ptrtoint ptr addrspace(3) %address.local312 to i64, !dbg !12
  %address.local314 = trunc i64 %address.local313 to i32, !dbg !12
  %77 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local314), !dbg !12
  %78 = zext i32 %77 to i64, !dbg !12
  %79 = inttoptr i64 %78 to ptr addrspace(3), !dbg !12
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local287, !dbg !12
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !12
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !12
  %80 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !12
  %81 = zext i32 %80 to i64, !dbg !12
  %82 = inttoptr i64 %81 to ptr addrspace(3), !dbg !12
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local293, !dbg !12
  %address.local319 = ptrtoint ptr addrspace(3) %address.local318 to i64, !dbg !12
  %address.local320 = trunc i64 %address.local319 to i32, !dbg !12
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local320), !dbg !12
  %84 = zext i32 %83 to i64, !dbg !12
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !12
  %address.local321 = shl nuw nsw i64 %address.local250, 10, !dbg !15
  %address.local322 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local321, !dbg !15
  %address.local323 = ptrtoint ptr addrspace(3) %address.local322 to i64, !dbg !15
  %address.local324 = trunc i64 %address.local323 to i32, !dbg !15
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local324), !dbg !15
  %address.local325 = shl nuw nsw i64 %address.local256, 10, !dbg !15
  %address.local326 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !15
  %address.local327 = ptrtoint ptr addrspace(3) %address.local326 to i64, !dbg !15
  %address.local328 = trunc i64 %address.local327 to i32, !dbg !15
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local328), !dbg !15
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
  %.neg180 = sdiv i64 %103, -4, !dbg !16
  %104 = shl nuw nsw i64 %32, 4, !dbg !16
  %105 = shl nuw nsw i64 %90, 6, !dbg !16
  %106 = sub nsw i64 %104, %105, !dbg !16
  %107 = or disjoint i64 %106, %92, !dbg !16
  %108 = shl nuw nsw i32 %31, 1, !dbg !16
  %109 = and i32 %108, 6, !dbg !16
  %index.compact110 = zext nneg i32 %109 to i64, !dbg !16
  %110 = lshr i32 %31, 1, !dbg !16
  %111 = and i32 %110, 1, !dbg !16
  %index.compact107 = zext nneg i32 %111 to i64, !dbg !16
  %112 = shl nuw nsw i64 %90, 4, !dbg !16
  %113 = sub nsw i64 %33, %112, !dbg !16
  %114 = or disjoint i64 %113, %index.compact107, !dbg !16
  %115 = add nuw nsw i64 %114, %95, !dbg !16
  %116 = add nsw i64 %115, 1, !dbg !16
  %.neg183 = sdiv i64 %116, -2, !dbg !16
  %117 = add nuw nsw i64 %107, %index.compact110, !dbg !16
  %.neg185 = sdiv i64 %117, -4, !dbg !16
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
  %address.thread329 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !11, !range !30, !frisk.register_lifetimes !1
  %address.local330 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local331 = and i32 %address.local330, 2, !dbg !11
  %address.local332 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local333 = and i32 %address.local332, 16256, !dbg !11
  %address.local334 = or disjoint i32 %address.local331, %address.local333, !dbg !11
  %address.local335 = zext nneg i32 %address.local334 to i64, !dbg !11
  %address.local336 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local335, !dbg !11
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local336 to i64, !dbg !11
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #10, !dbg !11
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local337 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local338 = and i32 %address.local337, 2, !dbg !11
  %address.local339 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local340 = and i32 %address.local339, 16256, !dbg !11
  %address.local341 = or disjoint i32 %address.local338, %address.local340, !dbg !11
  %address.local342 = or disjoint i32 %address.local341, 4, !dbg !11
  %address.local343 = zext nneg i32 %address.local342 to i64, !dbg !11
  %address.local344 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local343, !dbg !11
  %dma.source.bits1236 = ptrtoint ptr addrspace(1) %address.local344 to i64, !dbg !11
  %dma.source.opaque1237 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1236) #10, !dbg !11
  %dma.source.ptr1238 = inttoptr i64 %dma.source.opaque1237 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1238, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local345 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local346 = and i32 %address.local345, 2, !dbg !11
  %address.local347 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local348 = and i32 %address.local347, 16256, !dbg !11
  %address.local349 = or disjoint i32 %address.local346, %address.local348, !dbg !11
  %address.local350 = or disjoint i32 %address.local349, 8, !dbg !11
  %address.local351 = zext nneg i32 %address.local350 to i64, !dbg !11
  %address.local352 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local351, !dbg !11
  %dma.source.bits1239 = ptrtoint ptr addrspace(1) %address.local352 to i64, !dbg !11
  %dma.source.opaque1240 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1239) #10, !dbg !11
  %dma.source.ptr1241 = inttoptr i64 %dma.source.opaque1240 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1241, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local353 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local354 = and i32 %address.local353, 2, !dbg !11
  %address.local355 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local356 = and i32 %address.local355, 16256, !dbg !11
  %address.local357 = or disjoint i32 %address.local354, %address.local356, !dbg !11
  %address.local358 = or disjoint i32 %address.local357, 12, !dbg !11
  %address.local359 = zext nneg i32 %address.local358 to i64, !dbg !11
  %address.local360 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local359, !dbg !11
  %dma.source.bits1242 = ptrtoint ptr addrspace(1) %address.local360 to i64, !dbg !11
  %dma.source.opaque1243 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1242) #10, !dbg !11
  %dma.source.ptr1244 = inttoptr i64 %dma.source.opaque1243 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1244, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local361 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local362 = and i32 %address.local361, 2, !dbg !11
  %address.local363 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local364 = and i32 %address.local363, 16256, !dbg !11
  %address.local365 = or disjoint i32 %address.local362, %address.local364, !dbg !11
  %address.local366 = or disjoint i32 %address.local365, 16, !dbg !11
  %address.local367 = zext nneg i32 %address.local366 to i64, !dbg !11
  %address.local368 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local367, !dbg !11
  %dma.source.bits1245 = ptrtoint ptr addrspace(1) %address.local368 to i64, !dbg !11
  %dma.source.opaque1246 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1245) #10, !dbg !11
  %dma.source.ptr1247 = inttoptr i64 %dma.source.opaque1246 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1247, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local369 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local370 = and i32 %address.local369, 2, !dbg !11
  %address.local371 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local372 = and i32 %address.local371, 16256, !dbg !11
  %address.local373 = or disjoint i32 %address.local370, %address.local372, !dbg !11
  %address.local374 = or disjoint i32 %address.local373, 20, !dbg !11
  %address.local375 = zext nneg i32 %address.local374 to i64, !dbg !11
  %address.local376 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local375, !dbg !11
  %dma.source.bits1248 = ptrtoint ptr addrspace(1) %address.local376 to i64, !dbg !11
  %dma.source.opaque1249 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1248) #10, !dbg !11
  %dma.source.ptr1250 = inttoptr i64 %dma.source.opaque1249 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1250, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local377 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local378 = and i32 %address.local377, 2, !dbg !11
  %address.local379 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local380 = and i32 %address.local379, 16256, !dbg !11
  %address.local381 = or disjoint i32 %address.local378, %address.local380, !dbg !11
  %address.local382 = or disjoint i32 %address.local381, 24, !dbg !11
  %address.local383 = zext nneg i32 %address.local382 to i64, !dbg !11
  %address.local384 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local383, !dbg !11
  %dma.source.bits1251 = ptrtoint ptr addrspace(1) %address.local384 to i64, !dbg !11
  %dma.source.opaque1252 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1251) #10, !dbg !11
  %dma.source.ptr1253 = inttoptr i64 %dma.source.opaque1252 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1253, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local385 = shl nuw nsw i32 %address.thread329, 1, !dbg !11
  %address.local386 = and i32 %address.local385, 2, !dbg !11
  %address.local387 = shl nuw nsw i32 %address.thread329, 6, !dbg !11
  %address.local388 = and i32 %address.local387, 16256, !dbg !11
  %address.local389 = or disjoint i32 %address.local386, %address.local388, !dbg !11
  %address.local390 = or disjoint i32 %address.local389, 28, !dbg !11
  %address.local391 = zext nneg i32 %address.local390 to i64, !dbg !11
  %address.local392 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local391, !dbg !11
  %dma.source.bits1254 = ptrtoint ptr addrspace(1) %address.local392 to i64, !dbg !11
  %dma.source.opaque1255 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1254) #10, !dbg !11
  %dma.source.ptr1256 = inttoptr i64 %dma.source.opaque1255 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1256, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %134 = shl nsw i64 %18, 19, !dbg !12
  %135 = and i64 %134, 16252928, !dbg !12
  %136 = shl nuw nsw i64 %125, 14, !dbg !12
  %137 = or disjoint i64 %135, %136, !dbg !12
  %138 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %137, !dbg !12
  %address.local393 = zext nneg i32 %address.thread329 to i64, !dbg !13
  %address.local394 = lshr i64 %address.local393, 2, !dbg !12
  %address.local395 = lshr i64 %address.local393, 5, !dbg !12
  %address.local396 = mul nuw nsw i64 %address.local395, 7, !dbg !12
  %address.local397 = add nuw nsw i64 %address.local394, %address.local396, !dbg !12
  %address.local398 = and i64 %address.local394, 60, !dbg !12
  %address.local399 = add nuw nsw i64 %address.local397, %address.local398, !dbg !12
  %address.local400 = lshr i64 %address.local393, 7, !dbg !12
  %address.local401 = sub nsw i64 %address.local394, %address.local395, !dbg !12
  %address.local402 = sdiv i64 %address.local401, 4, !dbg !12
  %address.local403 = add nsw i64 %address.local400, %address.local402, !dbg !12
  %address.local404 = shl nsw i64 %address.local403, 2, !dbg !12
  %address.local405 = lshr i64 %address.local393, 4, !dbg !12
  %address.local406 = sub nsw i64 %address.local405, %address.local400, !dbg !12
  %address.local407 = sdiv i64 %address.local406, 2, !dbg !12
  %address.local408 = shl nsw i64 %address.local407, 3, !dbg !12
  %address.local409 = add nsw i64 %address.local404, %address.local408, !dbg !12
  %address.local410 = sub i64 %address.local399, %address.local409, !dbg !12
  %address.local411 = shl i64 %address.local410, 7, !dbg !12
  %address.local412 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local413 = and i32 %address.local412, 2, !dbg !12
  %address.local414 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local415 = and i32 %address.local414, 16, !dbg !12
  %address.local416 = or disjoint i32 %address.local413, %address.local415, !dbg !12
  %address.local417 = zext nneg i32 %address.local416 to i64, !dbg !12
  %address.local418 = or disjoint i64 %address.local411, %address.local417, !dbg !12
  %address.local419 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local418, !dbg !12
  %dma.source.bits1257 = ptrtoint ptr addrspace(1) %address.local419 to i64, !dbg !12
  %dma.source.opaque1258 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1257) #10, !dbg !12
  %dma.source.ptr1259 = inttoptr i64 %dma.source.opaque1258 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1259, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local420 = or disjoint i64 %address.local393, 256, !dbg !11
  %address.local421 = lshr i64 %address.local420, 2, !dbg !12
  %address.local422 = lshr i64 %address.local420, 5, !dbg !12
  %address.local423 = mul nuw nsw i64 %address.local422, 7, !dbg !12
  %address.local424 = add nuw nsw i64 %address.local421, %address.local423, !dbg !12
  %address.local425 = and i64 %address.local421, 124, !dbg !12
  %address.local426 = add nuw nsw i64 %address.local424, %address.local425, !dbg !12
  %address.local427 = sub nuw nsw i64 %address.local421, %address.local422, !dbg !12
  %address.local428 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local429 = and i32 %address.local428, 4, !dbg !12
  %address.local430 = or disjoint i32 %address.local429, 8, !dbg !12
  %address.local431 = zext nneg i32 %address.local430 to i64, !dbg !12
  %address.local432 = add nuw nsw i64 %address.local427, %address.local431, !dbg !12
  %address.local433 = and i64 %address.local432, 1073741820, !dbg !12
  %address.local434 = lshr i64 %address.local420, 4, !dbg !12
  %address.local435 = lshr i32 %address.thread329, 7, !dbg !12
  %address.local436 = or disjoint i32 %address.local435, 2, !dbg !12
  %address.local437 = zext nneg i32 %address.local436 to i64, !dbg !12
  %address.local438 = sub nuw nsw i64 %address.local434, %address.local437, !dbg !12
  %address.local439 = shl nuw nsw i64 %address.local438, 2, !dbg !12
  %address.local440 = and i64 %address.local439, 536870904, !dbg !12
  %address.local441 = add nuw nsw i64 %address.local433, %address.local440, !dbg !12
  %address.local442 = sub nsw i64 %address.local426, %address.local441, !dbg !12
  %address.local443 = shl i64 %address.local442, 7, !dbg !12
  %address.local444 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local445 = and i32 %address.local444, 2, !dbg !12
  %address.local446 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local447 = and i32 %address.local446, 16, !dbg !12
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !12
  %address.local449 = zext nneg i32 %address.local448 to i64, !dbg !12
  %address.local450 = or disjoint i64 %address.local443, %address.local449, !dbg !12
  %address.local451 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local450, !dbg !12
  %dma.source.bits1260 = ptrtoint ptr addrspace(1) %address.local451 to i64, !dbg !12
  %dma.source.opaque1261 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1260) #10, !dbg !12
  %dma.source.ptr1262 = inttoptr i64 %dma.source.opaque1261 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1262, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local452 = lshr i64 %address.local393, 2, !dbg !12
  %address.local453 = lshr i64 %address.local393, 5, !dbg !12
  %address.local454 = mul nuw nsw i64 %address.local453, 7, !dbg !12
  %address.local455 = add nuw nsw i64 %address.local452, %address.local454, !dbg !12
  %address.local456 = and i64 %address.local452, 60, !dbg !12
  %address.local457 = add nuw nsw i64 %address.local455, %address.local456, !dbg !12
  %address.local458 = sub nsw i64 %address.local452, %address.local453, !dbg !12
  %address.local459 = sdiv i64 %address.local458, 4, !dbg !12
  %address.local460 = lshr i32 %address.thread329, 7, !dbg !12
  %address.local461 = zext nneg i32 %address.local460 to i64, !dbg !12
  %address.local462 = add nsw i64 %address.local459, %address.local461, !dbg !12
  %address.local463 = shl nsw i64 %address.local462, 2, !dbg !12
  %address.local464 = lshr i64 %address.local393, 4, !dbg !12
  %address.local465 = lshr i32 %address.thread329, 7, !dbg !12
  %address.local466 = zext nneg i32 %address.local465 to i64, !dbg !12
  %address.local467 = sub nsw i64 %address.local464, %address.local466, !dbg !12
  %address.local468 = sdiv i64 %address.local467, 2, !dbg !12
  %address.local469 = shl nsw i64 %address.local468, 3, !dbg !12
  %address.local470 = add nsw i64 %address.local463, %address.local469, !dbg !12
  %address.local471 = sub i64 %address.local457, %address.local470, !dbg !12
  %address.local472 = shl i64 %address.local471, 7, !dbg !12
  %address.local473 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local474 = and i32 %address.local473, 2, !dbg !12
  %address.local475 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local476 = and i32 %address.local475, 16, !dbg !12
  %address.local477 = or disjoint i32 %address.local474, %address.local476, !dbg !12
  %address.local478 = or disjoint i32 %address.local477, 4, !dbg !12
  %address.local479 = zext nneg i32 %address.local478 to i64, !dbg !12
  %address.local480 = or disjoint i64 %address.local472, %address.local479, !dbg !12
  %address.local481 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local480, !dbg !12
  %dma.source.bits1263 = ptrtoint ptr addrspace(1) %address.local481 to i64, !dbg !12
  %dma.source.opaque1264 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1263) #10, !dbg !12
  %dma.source.ptr1265 = inttoptr i64 %dma.source.opaque1264 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1265, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local482 = lshr i64 %address.local393, 2, !dbg !12
  %address.local483 = or disjoint i64 %address.local482, 192, !dbg !12
  %address.local484 = and i64 %address.local483, 127, !dbg !12
  %address.local485 = lshr i64 %address.local484, 3, !dbg !12
  %address.local486 = mul nuw nsw i64 %address.local485, 7, !dbg !12
  %address.local487 = add nuw nsw i64 %address.local484, %address.local486, !dbg !12
  %address.local488 = and i64 %address.local483, 124, !dbg !12
  %address.local489 = add nuw nsw i64 %address.local487, %address.local488, !dbg !12
  %address.local490 = sub nuw nsw i64 %address.local484, %address.local485, !dbg !12
  %address.local491 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local492 = and i32 %address.local491, 4, !dbg !12
  %address.local493 = or disjoint i32 %address.local492, 8, !dbg !12
  %address.local494 = zext nneg i32 %address.local493 to i64, !dbg !12
  %address.local495 = add nuw nsw i64 %address.local490, %address.local494, !dbg !12
  %address.local496 = and i64 %address.local495, 252, !dbg !12
  %address.local497 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local498 = and i32 %address.local497, 4, !dbg !12
  %address.local499 = or disjoint i32 %address.local498, 8, !dbg !12
  %address.local500 = zext nneg i32 %address.local499 to i64, !dbg !12
  %address.local501 = sub nuw nsw i64 %address.local483, %address.local500, !dbg !12
  %address.local502 = and i64 %address.local501, 120, !dbg !12
  %address.local503 = add nuw nsw i64 %address.local496, %address.local502, !dbg !12
  %address.local504 = sub nsw i64 %address.local489, %address.local503, !dbg !12
  %address.local505 = shl nsw i64 %address.local504, 7, !dbg !12
  %address.local506 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local507 = and i32 %address.local506, 2, !dbg !12
  %address.local508 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local509 = and i32 %address.local508, 16, !dbg !12
  %address.local510 = or disjoint i32 %address.local507, %address.local509, !dbg !12
  %address.local511 = or disjoint i32 %address.local510, 4, !dbg !12
  %address.local512 = zext nneg i32 %address.local511 to i64, !dbg !12
  %address.local513 = or disjoint i64 %address.local505, %address.local512, !dbg !12
  %address.local514 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local513, !dbg !12
  %dma.source.bits1266 = ptrtoint ptr addrspace(1) %address.local514 to i64, !dbg !12
  %dma.source.opaque1267 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1266) #10, !dbg !12
  %dma.source.ptr1268 = inttoptr i64 %dma.source.opaque1267 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1268, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local515 = lshr i64 %address.local393, 2, !dbg !12
  %address.local516 = lshr i64 %address.local393, 5, !dbg !12
  %address.local517 = mul nuw nsw i64 %address.local516, 7, !dbg !12
  %address.local518 = add nuw nsw i64 %address.local515, %address.local517, !dbg !12
  %address.local519 = and i64 %address.local515, 60, !dbg !12
  %address.local520 = add nuw nsw i64 %address.local518, %address.local519, !dbg !12
  %address.local521 = add nuw nsw i64 %address.local520, 144115188075855868, !dbg !12
  %address.local522 = lshr i64 %address.local393, 7, !dbg !12
  %address.local523 = sub nsw i64 %address.local515, %address.local516, !dbg !12
  %address.local524 = sdiv i64 %address.local523, 4, !dbg !12
  %address.local525 = add nsw i64 %address.local522, %address.local524, !dbg !12
  %address.local526 = shl nsw i64 %address.local525, 2, !dbg !12
  %address.local527 = lshr i64 %address.local393, 4, !dbg !12
  %address.local528 = xor i64 %address.local522, -1, !dbg !12
  %address.local529 = add nsw i64 %address.local527, %address.local528, !dbg !12
  %address.local530 = icmp slt i64 %address.local529, 0, !dbg !12
  %address.local531 = sub nsw i64 %address.local522, %address.local527, !dbg !12
  %address.local532 = select i1 %address.local530, i64 %address.local531, i64 %address.local529, !dbg !12
  %address.local533 = sdiv i64 %address.local532, 2, !dbg !12
  %address.local534 = ashr i64 %address.local529, 63, !dbg !12
  %address.local535 = xor i64 %address.local533, %address.local534, !dbg !12
  %address.local536 = shl nsw i64 %address.local535, 3, !dbg !12
  %address.local537 = add i64 %address.local526, %address.local536, !dbg !12
  %address.local538 = sub i64 %address.local521, %address.local537, !dbg !12
  %address.local539 = shl i64 %address.local538, 7, !dbg !12
  %address.local540 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local541 = and i32 %address.local540, 2, !dbg !12
  %address.local542 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local543 = and i32 %address.local542, 16, !dbg !12
  %address.local544 = or disjoint i32 %address.local541, %address.local543, !dbg !12
  %address.local545 = or disjoint i32 %address.local544, 8, !dbg !12
  %address.local546 = zext nneg i32 %address.local545 to i64, !dbg !12
  %address.local547 = or disjoint i64 %address.local539, %address.local546, !dbg !12
  %address.local548 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local547, !dbg !12
  %dma.source.bits1269 = ptrtoint ptr addrspace(1) %address.local548 to i64, !dbg !12
  %dma.source.opaque1270 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1269) #10, !dbg !12
  %dma.source.ptr1271 = inttoptr i64 %dma.source.opaque1270 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1271, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local549 = lshr i64 %address.local393, 2, !dbg !12
  %address.local550 = or disjoint i64 %address.local549, 320, !dbg !12
  %address.local551 = and i64 %address.local550, 127, !dbg !12
  %address.local552 = lshr i64 %address.local551, 3, !dbg !12
  %address.local553 = mul nuw nsw i64 %address.local552, 7, !dbg !12
  %address.local554 = add nuw nsw i64 %address.local551, %address.local553, !dbg !12
  %address.local555 = and i64 %address.local550, 124, !dbg !12
  %address.local556 = add nuw nsw i64 %address.local554, %address.local555, !dbg !12
  %address.local557 = sub nuw nsw i64 %address.local551, %address.local552, !dbg !12
  %address.local558 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local559 = and i32 %address.local558, 4, !dbg !12
  %address.local560 = add nuw nsw i32 %address.local559, 12, !dbg !12
  %address.local561 = zext nneg i32 %address.local560 to i64, !dbg !12
  %address.local562 = add nuw nsw i64 %address.local557, %address.local561, !dbg !12
  %address.local563 = and i64 %address.local562, 252, !dbg !12
  %address.local564 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local565 = and i32 %address.local564, 4, !dbg !12
  %address.local566 = add nuw nsw i32 %address.local565, 12, !dbg !12
  %address.local567 = zext nneg i32 %address.local566 to i64, !dbg !12
  %address.local568 = sub nuw nsw i64 %address.local550, %address.local567, !dbg !12
  %address.local569 = and i64 %address.local568, 120, !dbg !12
  %address.local570 = add nuw nsw i64 %address.local563, %address.local569, !dbg !12
  %address.local571 = sub nsw i64 %address.local556, %address.local570, !dbg !12
  %address.local572 = shl nsw i64 %address.local571, 7, !dbg !12
  %address.local573 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local574 = and i32 %address.local573, 2, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local576 = and i32 %address.local575, 16, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = or disjoint i32 %address.local577, 8, !dbg !12
  %address.local579 = zext nneg i32 %address.local578 to i64, !dbg !12
  %address.local580 = or disjoint i64 %address.local572, %address.local579, !dbg !12
  %address.local581 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local580, !dbg !12
  %dma.source.bits1272 = ptrtoint ptr addrspace(1) %address.local581 to i64, !dbg !12
  %dma.source.opaque1273 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1272) #10, !dbg !12
  %dma.source.ptr1274 = inttoptr i64 %dma.source.opaque1273 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1274, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local582 = lshr i64 %address.local393, 2, !dbg !12
  %address.local583 = lshr i64 %address.local393, 5, !dbg !12
  %address.local584 = mul nuw nsw i64 %address.local583, 7, !dbg !12
  %address.local585 = add nuw nsw i64 %address.local582, %address.local584, !dbg !12
  %address.local586 = and i64 %address.local582, 60, !dbg !12
  %address.local587 = add nuw nsw i64 %address.local585, %address.local586, !dbg !12
  %address.local588 = sub nsw i64 %address.local582, %address.local583, !dbg !12
  %address.local589 = sdiv i64 %address.local588, 4, !dbg !12
  %address.local590 = lshr i32 %address.thread329, 7, !dbg !12
  %address.local591 = add nuw nsw i32 %address.local590, 1, !dbg !12
  %address.local592 = zext nneg i32 %address.local591 to i64, !dbg !12
  %address.local593 = add nsw i64 %address.local589, %address.local592, !dbg !12
  %address.local594 = shl nsw i64 %address.local593, 2, !dbg !12
  %address.local595 = lshr i64 %address.local393, 4, !dbg !12
  %address.local596 = lshr i32 %address.thread329, 7, !dbg !12
  %address.local597 = add nuw nsw i32 %address.local596, 1, !dbg !12
  %address.local598 = zext nneg i32 %address.local597 to i64, !dbg !12
  %address.local599 = sub nsw i64 %address.local595, %address.local598, !dbg !12
  %address.local600 = ashr i64 %address.local599, 63, !dbg !12
  %address.local601 = xor i64 %address.local599, %address.local600, !dbg !12
  %address.local602 = sdiv i64 %address.local601, 2, !dbg !12
  %address.local603 = ashr i64 %address.local599, 63, !dbg !12
  %address.local604 = xor i64 %address.local602, %address.local603, !dbg !12
  %address.local605 = shl nsw i64 %address.local604, 3, !dbg !12
  %address.local606 = add i64 %address.local594, %address.local605, !dbg !12
  %address.local607 = sub i64 %address.local587, %address.local606, !dbg !12
  %address.local608 = shl i64 %address.local607, 7, !dbg !12
  %address.local609 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local610 = and i32 %address.local609, 2, !dbg !12
  %address.local611 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local612 = and i32 %address.local611, 16, !dbg !12
  %address.local613 = or disjoint i32 %address.local610, %address.local612, !dbg !12
  %address.local614 = or disjoint i32 %address.local613, 12, !dbg !12
  %address.local615 = zext nneg i32 %address.local614 to i64, !dbg !12
  %address.local616 = or disjoint i64 %address.local608, %address.local615, !dbg !12
  %address.local617 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local616, !dbg !12
  %dma.source.bits1275 = ptrtoint ptr addrspace(1) %address.local617 to i64, !dbg !12
  %dma.source.opaque1276 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1275) #10, !dbg !12
  %dma.source.ptr1277 = inttoptr i64 %dma.source.opaque1276 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1277, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local618 = lshr i64 %address.local393, 2, !dbg !12
  %address.local619 = or disjoint i64 %address.local618, 448, !dbg !12
  %address.local620 = and i64 %address.local619, 127, !dbg !12
  %address.local621 = lshr i64 %address.local620, 3, !dbg !12
  %address.local622 = mul nuw nsw i64 %address.local621, 7, !dbg !12
  %address.local623 = add nuw nsw i64 %address.local620, %address.local622, !dbg !12
  %address.local624 = and i64 %address.local619, 124, !dbg !12
  %address.local625 = add nuw nsw i64 %address.local623, %address.local624, !dbg !12
  %address.local626 = sub nuw nsw i64 %address.local620, %address.local621, !dbg !12
  %address.local627 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local628 = and i32 %address.local627, 4, !dbg !12
  %address.local629 = add nuw nsw i32 %address.local628, 12, !dbg !12
  %address.local630 = zext nneg i32 %address.local629 to i64, !dbg !12
  %address.local631 = add nuw nsw i64 %address.local626, %address.local630, !dbg !12
  %address.local632 = and i64 %address.local631, 252, !dbg !12
  %address.local633 = lshr i32 %address.thread329, 5, !dbg !12
  %address.local634 = and i32 %address.local633, 4, !dbg !12
  %address.local635 = add nuw nsw i32 %address.local634, 12, !dbg !12
  %address.local636 = zext nneg i32 %address.local635 to i64, !dbg !12
  %address.local637 = sub nuw nsw i64 %address.local619, %address.local636, !dbg !12
  %address.local638 = and i64 %address.local637, 120, !dbg !12
  %address.local639 = add nuw nsw i64 %address.local632, %address.local638, !dbg !12
  %address.local640 = sub nsw i64 %address.local625, %address.local639, !dbg !12
  %address.local641 = shl nsw i64 %address.local640, 7, !dbg !12
  %address.local642 = shl nuw nsw i32 %address.thread329, 1, !dbg !12
  %address.local643 = and i32 %address.local642, 2, !dbg !12
  %address.local644 = shl nuw nsw i32 %address.thread329, 3, !dbg !12
  %address.local645 = and i32 %address.local644, 16, !dbg !12
  %address.local646 = or disjoint i32 %address.local643, %address.local645, !dbg !12
  %address.local647 = or disjoint i32 %address.local646, 12, !dbg !12
  %address.local648 = zext nneg i32 %address.local647 to i64, !dbg !12
  %address.local649 = or disjoint i64 %address.local641, %address.local648, !dbg !12
  %address.local650 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local649, !dbg !12
  %dma.source.bits1278 = ptrtoint ptr addrspace(1) %address.local650 to i64, !dbg !12
  %dma.source.opaque1279 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1278) #10, !dbg !12
  %dma.source.ptr1280 = inttoptr i64 %dma.source.opaque1279 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1280, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local651 = and i32 %address.thread329, 192, !dbg !13
  %address.local652 = shl nuw nsw i32 %address.thread329, 2, !dbg !13
  %address.local653 = and i32 %address.local652, 60, !dbg !13
  %address.local654 = or disjoint i32 %address.local651, %address.local653, !dbg !13
  %address.local655 = shl nuw nsw i32 %address.thread329, 5, !dbg !13
  %address.local656 = and i32 %address.local655, 1536, !dbg !13
  %address.local657 = or disjoint i32 %address.local654, %address.local656, !dbg !13
  %address.local658 = zext nneg i32 %address.local657 to i64, !dbg !13
  %address.local659 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local658, !dbg !13
  %139 = load <4 x half>, ptr addrspace(3) %address.local659, align 2, !dbg !13
  %address.local660 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local658, !dbg !13
  %address.local661 = getelementptr i8, ptr addrspace(3) %address.local660, i64 4096, !dbg !13
  %140 = load <4 x half>, ptr addrspace(3) %address.local661, align 2, !dbg !13
  %address.local662 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local658, !dbg !13
  %address.local663 = getelementptr i8, ptr addrspace(3) %address.local662, i64 512, !dbg !13
  %141 = load <4 x half>, ptr addrspace(3) %address.local663, align 2, !dbg !13
  %142 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !13
  %address.local664 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local658, !dbg !13
  %address.local665 = getelementptr i8, ptr addrspace(3) %address.local664, i64 4608, !dbg !13
  %143 = load <4 x half>, ptr addrspace(3) %address.local665, align 2, !dbg !13
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !13
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !13
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local666 = shl i64 %address.base, 6, !dbg !16
  %address.base667 = call i64 asm sideeffect "", "=v,0"(i64 %.neg180) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local668 = shl i64 %address.base667, 5, !dbg !16
  %address.local669 = add i64 %address.local666, %address.local668, !dbg !16
  %address.local670 = mul nuw nsw i64 %address.local393, 400, !dbg !16
  %address.local671 = lshr i64 %address.local393, 2, !dbg !16
  %address.local672 = mul nsw i64 %address.local671, -1664, !dbg !16
  %address.local673 = add nsw i64 %address.local670, %address.local672, !dbg !16
  %address.local674 = lshr i64 %address.local393, 2, !dbg !16
  %address.local675 = and i64 %address.local674, 3, !dbg !16
  %address.local676 = shl nuw nsw i64 %address.local675, 3, !dbg !16
  %address.local677 = add nsw i64 %address.local673, %address.local676, !dbg !16
  %address.local678 = lshr i64 %address.local393, 1, !dbg !16
  %address.local679 = shl nuw nsw i64 %address.local678, 5, !dbg !16
  %address.local680 = add nsw i64 %address.local677, %address.local679, !dbg !16
  %address.local681 = and i64 %address.local393, 63, !dbg !13
  %147 = shl nuw nsw i64 %address.local393, 6, !dbg !16
  %address.local683 = and i64 %147, 3072, !dbg !16
  %address.local684 = add nsw i64 %address.local680, %address.local683, !dbg !16
  %address.local685 = and i64 %address.local393, 32, !dbg !16
  %address.local686 = add i64 %address.local684, %address.local685, !dbg !16
  %address.local687 = add i64 %address.local669, %address.local686, !dbg !16
  %address.local688 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local687, !dbg !16
  %148 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %149 = shufflevector <8 x half> %148, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %151 = shufflevector <8 x half> %148, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %151, <4 x float> %150) #9, !dbg !16
  %address.base689 = call i64 asm sideeffect "", "=v,0"(i64 %.neg183) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local690 = shl i64 %address.base689, 7, !dbg !16
  %address.local691 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local690, !dbg !16
  %address.base692 = call i64 asm sideeffect "", "=v,0"(i64 %.neg185) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local693 = shl i64 %address.base692, 6, !dbg !16
  %address.local694 = getelementptr i8, ptr addrspace(3) %address.local691, i64 %address.local693, !dbg !16
  %address.local695 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local696 = and i32 %address.local695, 6, !dbg !16
  %narrow = mul nuw nsw i32 %address.local696, 144, !dbg !16
  %address.local698 = zext nneg i32 %narrow to i64, !dbg !16
  %address.local699 = getelementptr i8, ptr addrspace(3) %address.local694, i64 %address.local698, !dbg !16
  %address.local700 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local701 = and i32 %address.local700, 24, !dbg !16
  %address.local702 = shl nuw nsw i32 %address.thread329, 8, !dbg !16
  %address.local703 = and i32 %address.local702, 768, !dbg !16
  %address.local704 = or disjoint i32 %address.local701, %address.local703, !dbg !16
  %address.local705 = zext nneg i32 %address.local704 to i64, !dbg !16
  %address.local706 = getelementptr [2 x i8], ptr addrspace(3) %address.local699, i64 %address.local705, !dbg !16
  %address.local707 = shl nuw nsw i32 %address.thread329, 4, !dbg !16
  %address.local708 = and i32 %address.local707, 32, !dbg !16
  %address.local709 = zext nneg i32 %address.local708 to i64, !dbg !16
  %address.local710 = getelementptr [2 x i8], ptr addrspace(3) %address.local706, i64 %address.local709, !dbg !16
  %address.local711 = getelementptr [2 x i8], ptr addrspace(3) %address.local710, i64 %address.local683, !dbg !16
  %address.local712 = getelementptr [2 x i8], ptr addrspace(3) %address.local711, i64 %address.local685, !dbg !16
  %address.local713 = getelementptr i8, ptr addrspace(3) %address.local712, i64 64, !dbg !16
  %153 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %address.local714 = shl nuw nsw i64 %address.local393, 2, !dbg !13
  %address.local715 = mul nsw i64 %address.local671, -18, !dbg !16
  %address.local716 = add nsw i64 %address.local714, %address.local715, !dbg !16
  %address.local717 = add nsw i64 %address.local716, %address.local678, !dbg !16
  %address.local718 = lshr i64 %address.local681, 5, !dbg !16
  %address.local719 = add nsw i64 %address.local717, %address.local718, !dbg !16
  %address.local720 = sdiv i64 %address.local719, -2, !dbg !16
  %address.local721 = shl i64 %address.local720, 7, !dbg !16
  %address.local722 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local721, !dbg !16
  %address.local723 = mul nuw nsw i64 %address.local393, 18, !dbg !16
  %address.local724 = mul nsw i64 %address.local671, -72, !dbg !16
  %address.local725 = add nsw i64 %address.local723, %address.local724, !dbg !16
  %address.local726 = add nsw i64 %address.local725, %address.local675, !dbg !16
  %address.local727 = add nsw i64 %address.local726, 1, !dbg !16
  %address.local728 = sdiv i64 %address.local727, -4, !dbg !16
  %address.local729 = shl i64 %address.local728, 6, !dbg !16
  %address.local730 = getelementptr i8, ptr addrspace(3) %address.local722, i64 %address.local729, !dbg !16
  %address.local731 = getelementptr [2 x i8], ptr addrspace(3) %address.local730, i64 %address.local670, !dbg !16
  %address.local732 = getelementptr [2 x i8], ptr addrspace(3) %address.local731, i64 %address.local672, !dbg !16
  %address.local733 = getelementptr [2 x i8], ptr addrspace(3) %address.local732, i64 %address.local676, !dbg !16
  %address.local734 = getelementptr [2 x i8], ptr addrspace(3) %address.local733, i64 %address.local679, !dbg !16
  %address.local735 = getelementptr [2 x i8], ptr addrspace(3) %address.local734, i64 %address.local683, !dbg !16
  %address.local736 = getelementptr [2 x i8], ptr addrspace(3) %address.local735, i64 %address.local685, !dbg !16
  %address.local737 = getelementptr i8, ptr addrspace(3) %address.local736, i64 144, !dbg !16
  %154 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %155 = shl nuw nsw i64 %address.local671, 4, !dbg !16
  %address.local739 = sub nsw i64 %address.local714, %155, !dbg !16
  %address.local740 = lshr i32 %address.thread329, 1, !dbg !16
  %address.local741 = and i32 %address.local740, 1, !dbg !16
  %address.local742 = zext nneg i32 %address.local741 to i64, !dbg !16
  %address.local743 = or disjoint i64 %address.local739, %address.local742, !dbg !16
  %address.local744 = add nuw nsw i64 %address.local743, %address.local718, !dbg !16
  %address.local745 = add nsw i64 %address.local744, 3, !dbg !16
  %address.local746 = sdiv i64 %address.local745, -2, !dbg !16
  %address.local747 = shl i64 %address.local746, 7, !dbg !16
  %address.local748 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local747, !dbg !16
  %address.local749 = shl nuw nsw i64 %address.local393, 4, !dbg !16
  %156 = shl nuw nsw i64 %address.local671, 6, !dbg !16
  %address.local751 = sub nsw i64 %address.local749, %156, !dbg !16
  %address.local752 = or disjoint i64 %address.local751, %address.local675, !dbg !16
  %address.local753 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local754 = and i32 %address.local753, 6, !dbg !16
  %address.local755 = or disjoint i32 %address.local754, 1, !dbg !16
  %address.local756 = zext nneg i32 %address.local755 to i64, !dbg !16
  %address.local757 = add nuw nsw i64 %address.local752, %address.local756, !dbg !16
  %address.local758 = sdiv i64 %address.local757, -4, !dbg !16
  %address.local759 = shl i64 %address.local758, 6, !dbg !16
  %address.local760 = getelementptr i8, ptr addrspace(3) %address.local748, i64 %address.local759, !dbg !16
  %address.local761 = mul nuw nsw i64 %address.local756, 144, !dbg !16
  %address.local762 = getelementptr i8, ptr addrspace(3) %address.local760, i64 %address.local761, !dbg !16
  %address.local763 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local764 = and i32 %address.local763, 24, !dbg !16
  %address.local765 = shl nuw nsw i32 %address.thread329, 8, !dbg !16
  %address.local766 = and i32 %address.local765, 768, !dbg !16
  %address.local767 = or disjoint i32 %address.local764, %address.local766, !dbg !16
  %address.local768 = zext nneg i32 %address.local767 to i64, !dbg !16
  %address.local769 = getelementptr [2 x i8], ptr addrspace(3) %address.local762, i64 %address.local768, !dbg !16
  %address.local770 = shl nuw nsw i32 %address.thread329, 4, !dbg !16
  %address.local771 = and i32 %address.local770, 32, !dbg !16
  %address.local772 = zext nneg i32 %address.local771 to i64, !dbg !16
  %address.local773 = getelementptr [2 x i8], ptr addrspace(3) %address.local769, i64 %address.local772, !dbg !16
  %address.local774 = getelementptr [2 x i8], ptr addrspace(3) %address.local773, i64 %address.local683, !dbg !16
  %address.local775 = getelementptr [2 x i8], ptr addrspace(3) %address.local774, i64 %address.local685, !dbg !16
  %address.local776 = getelementptr i8, ptr addrspace(3) %address.local775, i64 192, !dbg !16
  %157 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %address.local777 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local687, !dbg !16
  %address.local778 = getelementptr i8, ptr addrspace(3) %address.local777, i64 1024, !dbg !16
  %169 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %167, <4 x float> %168) #9, !dbg !16
  %address.local779 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local780 = and i32 %address.local779, 6, !dbg !16
  %address.local781 = or disjoint i32 %address.local780, 8, !dbg !16
  %address.local782 = mul nuw nsw i32 %address.local781, 72, !dbg !16
  %address.local783 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local784 = and i32 %address.local783, 24, !dbg !16
  %address.local785 = shl nuw nsw i32 %address.thread329, 8, !dbg !16
  %address.local786 = and i32 %address.local785, 768, !dbg !16
  %address.local787 = or disjoint i32 %address.local784, %address.local786, !dbg !16
  %address.local788 = or disjoint i32 %address.local787, 1024, !dbg !16
  %address.local789 = add nuw nsw i32 %address.local782, %address.local788, !dbg !16
  %address.local790 = shl nuw nsw i32 %address.thread329, 4, !dbg !16
  %address.local791 = and i32 %address.local790, 32, !dbg !16
  %address.local792 = or disjoint i32 %address.local791, 64, !dbg !16
  %address.local793 = add nuw nsw i32 %address.local789, %address.local792, !dbg !16
  %address.local794 = zext nneg i32 %address.local793 to i64, !dbg !16
  %address.local795 = add nuw nsw i64 %address.local683, %address.local794, !dbg !16
  %address.local796 = add nuw nsw i64 %address.local795, %address.local685, !dbg !16
  %address.local797 = shl nuw nsw i32 %address.thread329, 2, !dbg !16
  %address.local798 = and i32 %address.local797, 12, !dbg !16
  %address.local799 = lshr i32 %address.thread329, 1, !dbg !16
  %address.local800 = and i32 %address.local799, 1, !dbg !16
  %address.local801 = or disjoint i32 %address.local798, %address.local800, !dbg !16
  %address.local802 = or disjoint i32 %address.local801, 18, !dbg !16
  %address.local803 = zext nneg i32 %address.local802 to i64, !dbg !16
  %address.local804 = add nuw nsw i64 %address.local718, %address.local803, !dbg !16
  %address.local805 = shl nuw nsw i64 %address.local804, 5, !dbg !16
  %address.local806 = add nuw nsw i64 %address.local805, 32, !dbg !16
  %address.local807 = and i64 %address.local806, 274877906880, !dbg !16
  %address.local808 = lshr i32 %address.thread329, 2, !dbg !16
  %address.local809 = and i32 %address.local808, 2, !dbg !16
  %address.local810 = shl nuw nsw i32 %address.thread329, 4, !dbg !16
  %address.local811 = and i32 %address.local810, 48, !dbg !16
  %address.local812 = or disjoint i32 %address.local809, %address.local811, !dbg !16
  %address.local813 = or disjoint i32 %address.local812, 64, !dbg !16
  %address.local814 = add nuw nsw i32 %address.local813, %address.local781, !dbg !16
  %address.local815 = shl nuw nsw i32 %address.local814, 3, !dbg !16
  %address.local816 = and i32 %address.local815, 2016, !dbg !16
  %address.local817 = zext nneg i32 %address.local816 to i64, !dbg !16
  %address.local818 = add nuw nsw i64 %address.local807, %address.local817, !dbg !16
  %address.local819 = sub nsw i64 %address.local796, %address.local818, !dbg !16
  %address.local820 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local819, !dbg !16
  %address.local821 = getelementptr i8, ptr addrspace(3) %address.local820, i64 64, !dbg !16
  %171 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %address.local822 = add nuw nsw i64 %address.local714, 18, !dbg !16
  %address.local823 = add nsw i64 %address.local822, %address.local715, !dbg !16
  %address.local824 = add nsw i64 %address.local823, %address.local678, !dbg !16
  %address.local825 = add nsw i64 %address.local824, %address.local718, !dbg !16
  %address.local826 = sdiv i64 %address.local825, -2, !dbg !16
  %address.local827 = shl i64 %address.local826, 7, !dbg !16
  %address.local828 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local827, !dbg !16
  %address.local829 = add nuw nsw i64 %address.local723, 72, !dbg !16
  %address.local830 = add nsw i64 %address.local829, %address.local724, !dbg !16
  %address.local831 = add nsw i64 %address.local830, %address.local675, !dbg !16
  %address.local832 = add nsw i64 %address.local831, -15, !dbg !16
  %address.local833 = sdiv i64 %address.local832, -4, !dbg !16
  %address.local834 = shl i64 %address.local833, 6, !dbg !16
  %address.local835 = getelementptr i8, ptr addrspace(3) %address.local828, i64 %address.local834, !dbg !16
  %address.local836 = getelementptr [2 x i8], ptr addrspace(3) %address.local835, i64 %address.local670, !dbg !16
  %address.local837 = getelementptr [2 x i8], ptr addrspace(3) %address.local836, i64 %address.local672, !dbg !16
  %address.local838 = getelementptr [2 x i8], ptr addrspace(3) %address.local837, i64 %address.local676, !dbg !16
  %address.local839 = getelementptr [2 x i8], ptr addrspace(3) %address.local838, i64 %address.local679, !dbg !16
  %address.local840 = getelementptr [2 x i8], ptr addrspace(3) %address.local839, i64 %address.local683, !dbg !16
  %address.local841 = getelementptr [2 x i8], ptr addrspace(3) %address.local840, i64 %address.local685, !dbg !16
  %address.local842 = getelementptr i8, ptr addrspace(3) %address.local841, i64 3216, !dbg !16
  %172 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
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
  %address.local843 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local844 = and i32 %address.local843, 6, !dbg !16
  %address.local845 = or disjoint i32 %address.local844, 9, !dbg !16
  %address.local846 = mul nuw nsw i32 %address.local845, 72, !dbg !16
  %address.local847 = shl nuw nsw i32 %address.thread329, 1, !dbg !16
  %address.local848 = and i32 %address.local847, 24, !dbg !16
  %address.local849 = shl nuw nsw i32 %address.thread329, 8, !dbg !16
  %address.local850 = and i32 %address.local849, 768, !dbg !16
  %address.local851 = or disjoint i32 %address.local848, %address.local850, !dbg !16
  %address.local852 = or disjoint i32 %address.local851, 1024, !dbg !16
  %address.local853 = add nuw nsw i32 %address.local846, %address.local852, !dbg !16
  %address.local854 = shl nuw nsw i32 %address.thread329, 4, !dbg !16
  %address.local855 = and i32 %address.local854, 32, !dbg !16
  %address.local856 = or disjoint i32 %address.local855, 64, !dbg !16
  %address.local857 = add nuw nsw i32 %address.local853, %address.local856, !dbg !16
  %address.local858 = zext nneg i32 %address.local857 to i64, !dbg !16
  %address.local859 = add nuw nsw i64 %address.local683, %address.local858, !dbg !16
  %address.local860 = add nuw nsw i64 %address.local859, %address.local685, !dbg !16
  %address.local861 = shl nuw nsw i32 %address.thread329, 2, !dbg !16
  %address.local862 = and i32 %address.local861, 12, !dbg !16
  %address.local863 = lshr i32 %address.thread329, 1, !dbg !16
  %address.local864 = and i32 %address.local863, 1, !dbg !16
  %address.local865 = or disjoint i32 %address.local862, %address.local864, !dbg !16
  %address.local866 = or disjoint i32 %address.local865, 18, !dbg !16
  %address.local867 = zext nneg i32 %address.local866 to i64, !dbg !16
  %address.local868 = add nuw nsw i64 %address.local718, %address.local867, !dbg !16
  %address.local869 = shl nuw nsw i64 %address.local868, 5, !dbg !16
  %address.local870 = add nuw nsw i64 %address.local869, 96, !dbg !16
  %address.local871 = and i64 %address.local870, 274877906880, !dbg !16
  %address.local872 = lshr i32 %address.thread329, 2, !dbg !16
  %address.local873 = and i32 %address.local872, 3, !dbg !16
  %address.local874 = shl nuw nsw i32 %address.thread329, 4, !dbg !16
  %address.local875 = and i32 %address.local874, 48, !dbg !16
  %address.local876 = or disjoint i32 %address.local873, %address.local875, !dbg !16
  %address.local877 = or disjoint i32 %address.local876, 64, !dbg !16
  %address.local878 = add nuw nsw i32 %address.local877, %address.local845, !dbg !16
  %address.local879 = shl nuw nsw i32 %address.local878, 3, !dbg !16
  %address.local880 = and i32 %address.local879, 2016, !dbg !16
  %address.local881 = zext nneg i32 %address.local880 to i64, !dbg !16
  %address.local882 = add nuw nsw i64 %address.local871, %address.local881, !dbg !16
  %address.local883 = sub nsw i64 %address.local860, %address.local882, !dbg !16
  %address.local884 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local883, !dbg !16
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local884, i64 192, !dbg !16
  %185 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %186 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %187 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !16
  %189 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %189, <4 x float> %188) #9, !dbg !16
  %191 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %192 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !16
  %194 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %192, <4 x float> %193) #9, !dbg !16
  %196 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %197 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %210 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %211 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %212 = shufflevector <8 x half> %210, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %212, <4 x float> zeroinitializer) #9, !dbg !16
  %214 = shufflevector <8 x half> %210, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %214, <4 x float> %213) #9, !dbg !16
  %216 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %217 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %216, <4 x float> zeroinitializer) #9, !dbg !16
  %219 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %217, <4 x float> %218) #9, !dbg !16
  %221 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %222 = shufflevector <8 x half> %219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %222, <4 x float> zeroinitializer) #9, !dbg !16
  %224 = shufflevector <8 x half> %219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %224, <4 x float> %223) #9, !dbg !16
  %226 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %227 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %226, <4 x float> zeroinitializer) #9, !dbg !16
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %227, <4 x float> %228) #9, !dbg !16
  %address.local886 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local335, !dbg !11
  %dma.source.bits1281 = ptrtoint ptr addrspace(1) %address.local886 to i64, !dbg !11
  %dma.source.opaque1282 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1281) #10, !dbg !11
  %dma.source.ptr1283 = inttoptr i64 %dma.source.opaque1282 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1283, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local887 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local343, !dbg !11
  %dma.source.bits1284 = ptrtoint ptr addrspace(1) %address.local887 to i64, !dbg !11
  %dma.source.opaque1285 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1284) #10, !dbg !11
  %dma.source.ptr1286 = inttoptr i64 %dma.source.opaque1285 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1286, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local888 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local351, !dbg !11
  %dma.source.bits1287 = ptrtoint ptr addrspace(1) %address.local888 to i64, !dbg !11
  %dma.source.opaque1288 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1287) #10, !dbg !11
  %dma.source.ptr1289 = inttoptr i64 %dma.source.opaque1288 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1289, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local889 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local359, !dbg !11
  %dma.source.bits1290 = ptrtoint ptr addrspace(1) %address.local889 to i64, !dbg !11
  %dma.source.opaque1291 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1290) #10, !dbg !11
  %dma.source.ptr1292 = inttoptr i64 %dma.source.opaque1291 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1292, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local890 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local367, !dbg !11
  %dma.source.bits1293 = ptrtoint ptr addrspace(1) %address.local890 to i64, !dbg !11
  %dma.source.opaque1294 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1293) #10, !dbg !11
  %dma.source.ptr1295 = inttoptr i64 %dma.source.opaque1294 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1295, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local891 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local375, !dbg !11
  %dma.source.bits1296 = ptrtoint ptr addrspace(1) %address.local891 to i64, !dbg !11
  %dma.source.opaque1297 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1296) #10, !dbg !11
  %dma.source.ptr1298 = inttoptr i64 %dma.source.opaque1297 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1298, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local892 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local383, !dbg !11
  %dma.source.bits1299 = ptrtoint ptr addrspace(1) %address.local892 to i64, !dbg !11
  %dma.source.opaque1300 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1299) #10, !dbg !11
  %dma.source.ptr1301 = inttoptr i64 %dma.source.opaque1300 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1301, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local893 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local391, !dbg !11
  %dma.source.bits1302 = ptrtoint ptr addrspace(1) %address.local893 to i64, !dbg !11
  %dma.source.opaque1303 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1302) #10, !dbg !11
  %dma.source.ptr1304 = inttoptr i64 %dma.source.opaque1303 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1304, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %230 = or disjoint i64 %137, 32, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %231 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %230, !dbg !12
  %address.local894 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local418, !dbg !12
  %dma.source.bits1305 = ptrtoint ptr addrspace(1) %address.local894 to i64, !dbg !12
  %dma.source.opaque1306 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1305) #10, !dbg !12
  %dma.source.ptr1307 = inttoptr i64 %dma.source.opaque1306 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1307, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local895 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local450, !dbg !12
  %dma.source.bits1308 = ptrtoint ptr addrspace(1) %address.local895 to i64, !dbg !12
  %dma.source.opaque1309 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1308) #10, !dbg !12
  %dma.source.ptr1310 = inttoptr i64 %dma.source.opaque1309 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1310, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local896 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local480, !dbg !12
  %dma.source.bits1311 = ptrtoint ptr addrspace(1) %address.local896 to i64, !dbg !12
  %dma.source.opaque1312 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1311) #10, !dbg !12
  %dma.source.ptr1313 = inttoptr i64 %dma.source.opaque1312 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1313, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local897 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local513, !dbg !12
  %dma.source.bits1314 = ptrtoint ptr addrspace(1) %address.local897 to i64, !dbg !12
  %dma.source.opaque1315 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1314) #10, !dbg !12
  %dma.source.ptr1316 = inttoptr i64 %dma.source.opaque1315 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1316, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local898 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local547, !dbg !12
  %dma.source.bits1317 = ptrtoint ptr addrspace(1) %address.local898 to i64, !dbg !12
  %dma.source.opaque1318 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1317) #10, !dbg !12
  %dma.source.ptr1319 = inttoptr i64 %dma.source.opaque1318 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1319, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local899 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local580, !dbg !12
  %dma.source.bits1320 = ptrtoint ptr addrspace(1) %address.local899 to i64, !dbg !12
  %dma.source.opaque1321 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1320) #10, !dbg !12
  %dma.source.ptr1322 = inttoptr i64 %dma.source.opaque1321 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1322, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local900 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local616, !dbg !12
  %dma.source.bits1323 = ptrtoint ptr addrspace(1) %address.local900 to i64, !dbg !12
  %dma.source.opaque1324 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1323) #10, !dbg !12
  %dma.source.ptr1325 = inttoptr i64 %dma.source.opaque1324 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1325, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local901 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local649, !dbg !12
  %dma.source.bits1326 = ptrtoint ptr addrspace(1) %address.local901 to i64, !dbg !12
  %dma.source.opaque1327 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1326) #10, !dbg !12
  %dma.source.ptr1328 = inttoptr i64 %dma.source.opaque1327 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1328, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %232 = load <4 x half>, ptr addrspace(3) %address.local659, align 2, !dbg !13
  %233 = load <4 x half>, ptr addrspace(3) %address.local661, align 2, !dbg !13
  %234 = load <4 x half>, ptr addrspace(3) %address.local663, align 2, !dbg !13
  %235 = fmul <4 x half> %232, splat (half 0xH3015), !dbg !13
  %236 = load <4 x half>, ptr addrspace(3) %address.local665, align 2, !dbg !13
  %237 = fmul <4 x half> %233, splat (half 0xH3015), !dbg !13
  %238 = fmul <4 x half> %234, splat (half 0xH3015), !dbg !13
  %239 = fmul <4 x half> %236, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %240 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %241 = shufflevector <8 x half> %240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %241, <4 x float> %152) #9, !dbg !16
  %243 = shufflevector <8 x half> %240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %243, <4 x float> %242) #9, !dbg !16
  %245 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %246 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %247 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %259 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %257, <4 x float> %258) #9, !dbg !16
  %261 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %262 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
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
  %275 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %276 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %277 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %277, <4 x float> %190) #9, !dbg !16
  %279 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %279, <4 x float> %278) #9, !dbg !16
  %281 = shufflevector <8 x half> %276, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %282 = shufflevector <8 x half> %276, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %281, <4 x float> %195) #9, !dbg !16
  %284 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %282, <4 x float> %283) #9, !dbg !16
  %286 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %287 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %300 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %301 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %302 = shufflevector <8 x half> %300, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %302, <4 x float> %215) #9, !dbg !16
  %304 = shufflevector <8 x half> %300, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %304, <4 x float> %303) #9, !dbg !16
  %306 = shufflevector <8 x half> %301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %307 = shufflevector <8 x half> %301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %306, <4 x float> %220) #9, !dbg !16
  %309 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %307, <4 x float> %308) #9, !dbg !16
  %311 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %312 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %312, <4 x float> %225) #9, !dbg !16
  %314 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %314, <4 x float> %313) #9, !dbg !16
  %316 = shufflevector <8 x half> %311, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %317 = shufflevector <8 x half> %311, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %316, <4 x float> %229) #9, !dbg !16
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %317, <4 x float> %318) #9, !dbg !16
  %address.local902 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local335, !dbg !11
  %dma.source.bits1329 = ptrtoint ptr addrspace(1) %address.local902 to i64, !dbg !11
  %dma.source.opaque1330 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1329) #10, !dbg !11
  %dma.source.ptr1331 = inttoptr i64 %dma.source.opaque1330 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1331, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local903 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local343, !dbg !11
  %dma.source.bits1332 = ptrtoint ptr addrspace(1) %address.local903 to i64, !dbg !11
  %dma.source.opaque1333 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1332) #10, !dbg !11
  %dma.source.ptr1334 = inttoptr i64 %dma.source.opaque1333 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1334, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local904 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local351, !dbg !11
  %dma.source.bits1335 = ptrtoint ptr addrspace(1) %address.local904 to i64, !dbg !11
  %dma.source.opaque1336 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1335) #10, !dbg !11
  %dma.source.ptr1337 = inttoptr i64 %dma.source.opaque1336 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1337, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local905 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local359, !dbg !11
  %dma.source.bits1338 = ptrtoint ptr addrspace(1) %address.local905 to i64, !dbg !11
  %dma.source.opaque1339 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1338) #10, !dbg !11
  %dma.source.ptr1340 = inttoptr i64 %dma.source.opaque1339 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1340, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local906 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local367, !dbg !11
  %dma.source.bits1341 = ptrtoint ptr addrspace(1) %address.local906 to i64, !dbg !11
  %dma.source.opaque1342 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1341) #10, !dbg !11
  %dma.source.ptr1343 = inttoptr i64 %dma.source.opaque1342 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1343, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local907 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local375, !dbg !11
  %dma.source.bits1344 = ptrtoint ptr addrspace(1) %address.local907 to i64, !dbg !11
  %dma.source.opaque1345 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1344) #10, !dbg !11
  %dma.source.ptr1346 = inttoptr i64 %dma.source.opaque1345 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1346, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local908 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local383, !dbg !11
  %dma.source.bits1347 = ptrtoint ptr addrspace(1) %address.local908 to i64, !dbg !11
  %dma.source.opaque1348 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1347) #10, !dbg !11
  %dma.source.ptr1349 = inttoptr i64 %dma.source.opaque1348 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1349, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local909 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local391, !dbg !11
  %dma.source.bits1350 = ptrtoint ptr addrspace(1) %address.local909 to i64, !dbg !11
  %dma.source.opaque1351 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1350) #10, !dbg !11
  %dma.source.ptr1352 = inttoptr i64 %dma.source.opaque1351 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1352, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %320 = or disjoint i64 %137, 64, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %321 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %320, !dbg !12
  %address.local910 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local418, !dbg !12
  %dma.source.bits1353 = ptrtoint ptr addrspace(1) %address.local910 to i64, !dbg !12
  %dma.source.opaque1354 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1353) #10, !dbg !12
  %dma.source.ptr1355 = inttoptr i64 %dma.source.opaque1354 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1355, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local911 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local450, !dbg !12
  %dma.source.bits1356 = ptrtoint ptr addrspace(1) %address.local911 to i64, !dbg !12
  %dma.source.opaque1357 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1356) #10, !dbg !12
  %dma.source.ptr1358 = inttoptr i64 %dma.source.opaque1357 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1358, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local912 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local480, !dbg !12
  %dma.source.bits1359 = ptrtoint ptr addrspace(1) %address.local912 to i64, !dbg !12
  %dma.source.opaque1360 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1359) #10, !dbg !12
  %dma.source.ptr1361 = inttoptr i64 %dma.source.opaque1360 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1361, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local913 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local513, !dbg !12
  %dma.source.bits1362 = ptrtoint ptr addrspace(1) %address.local913 to i64, !dbg !12
  %dma.source.opaque1363 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1362) #10, !dbg !12
  %dma.source.ptr1364 = inttoptr i64 %dma.source.opaque1363 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1364, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local914 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local547, !dbg !12
  %dma.source.bits1365 = ptrtoint ptr addrspace(1) %address.local914 to i64, !dbg !12
  %dma.source.opaque1366 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1365) #10, !dbg !12
  %dma.source.ptr1367 = inttoptr i64 %dma.source.opaque1366 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1367, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local915 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local580, !dbg !12
  %dma.source.bits1368 = ptrtoint ptr addrspace(1) %address.local915 to i64, !dbg !12
  %dma.source.opaque1369 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1368) #10, !dbg !12
  %dma.source.ptr1370 = inttoptr i64 %dma.source.opaque1369 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1370, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local916 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local616, !dbg !12
  %dma.source.bits1371 = ptrtoint ptr addrspace(1) %address.local916 to i64, !dbg !12
  %dma.source.opaque1372 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1371) #10, !dbg !12
  %dma.source.ptr1373 = inttoptr i64 %dma.source.opaque1372 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1373, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local649, !dbg !12
  %dma.source.bits1374 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !12
  %dma.source.opaque1375 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1374) #10, !dbg !12
  %dma.source.ptr1376 = inttoptr i64 %dma.source.opaque1375 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1376, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %address.local659, align 2, !dbg !13
  %323 = load <4 x half>, ptr addrspace(3) %address.local661, align 2, !dbg !13
  %324 = load <4 x half>, ptr addrspace(3) %address.local663, align 2, !dbg !13
  %325 = fmul <4 x half> %322, splat (half 0xH3015), !dbg !13
  %326 = load <4 x half>, ptr addrspace(3) %address.local665, align 2, !dbg !13
  %327 = fmul <4 x half> %323, splat (half 0xH3015), !dbg !13
  %328 = fmul <4 x half> %324, splat (half 0xH3015), !dbg !13
  %329 = fmul <4 x half> %326, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %330 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %331 = shufflevector <8 x half> %330, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %331, <4 x float> %244) #9, !dbg !16
  %333 = shufflevector <8 x half> %330, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %333, <4 x float> %332) #9, !dbg !16
  %335 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %336 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %337 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %349 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %347, <4 x float> %348) #9, !dbg !16
  %351 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %352 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
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
  %365 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %366 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %367 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %367, <4 x float> %280) #9, !dbg !16
  %369 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %369, <4 x float> %368) #9, !dbg !16
  %371 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %372 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %371, <4 x float> %285) #9, !dbg !16
  %374 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %372, <4 x float> %373) #9, !dbg !16
  %376 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %377 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %390 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %391 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %392 = shufflevector <8 x half> %390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %392, <4 x float> %305) #9, !dbg !16
  %394 = shufflevector <8 x half> %390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %394, <4 x float> %393) #9, !dbg !16
  %396 = shufflevector <8 x half> %391, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %397 = shufflevector <8 x half> %391, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %396, <4 x float> %310) #9, !dbg !16
  %399 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %397, <4 x float> %398) #9, !dbg !16
  %401 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %402 = shufflevector <8 x half> %399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %402, <4 x float> %315) #9, !dbg !16
  %404 = shufflevector <8 x half> %399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %404, <4 x float> %403) #9, !dbg !16
  %406 = shufflevector <8 x half> %401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %407 = shufflevector <8 x half> %401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %406, <4 x float> %319) #9, !dbg !16
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %407, <4 x float> %408) #9, !dbg !16
  %address.local918 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local335, !dbg !11
  %dma.source.bits1377 = ptrtoint ptr addrspace(1) %address.local918 to i64, !dbg !11
  %dma.source.opaque1378 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1377) #10, !dbg !11
  %dma.source.ptr1379 = inttoptr i64 %dma.source.opaque1378 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1379, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local919 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local343, !dbg !11
  %dma.source.bits1380 = ptrtoint ptr addrspace(1) %address.local919 to i64, !dbg !11
  %dma.source.opaque1381 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1380) #10, !dbg !11
  %dma.source.ptr1382 = inttoptr i64 %dma.source.opaque1381 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1382, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local920 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local351, !dbg !11
  %dma.source.bits1383 = ptrtoint ptr addrspace(1) %address.local920 to i64, !dbg !11
  %dma.source.opaque1384 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1383) #10, !dbg !11
  %dma.source.ptr1385 = inttoptr i64 %dma.source.opaque1384 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1385, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local921 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local359, !dbg !11
  %dma.source.bits1386 = ptrtoint ptr addrspace(1) %address.local921 to i64, !dbg !11
  %dma.source.opaque1387 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1386) #10, !dbg !11
  %dma.source.ptr1388 = inttoptr i64 %dma.source.opaque1387 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1388, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local922 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local367, !dbg !11
  %dma.source.bits1389 = ptrtoint ptr addrspace(1) %address.local922 to i64, !dbg !11
  %dma.source.opaque1390 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1389) #10, !dbg !11
  %dma.source.ptr1391 = inttoptr i64 %dma.source.opaque1390 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1391, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local923 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local375, !dbg !11
  %dma.source.bits1392 = ptrtoint ptr addrspace(1) %address.local923 to i64, !dbg !11
  %dma.source.opaque1393 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1392) #10, !dbg !11
  %dma.source.ptr1394 = inttoptr i64 %dma.source.opaque1393 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1394, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local924 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local383, !dbg !11
  %dma.source.bits1395 = ptrtoint ptr addrspace(1) %address.local924 to i64, !dbg !11
  %dma.source.opaque1396 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1395) #10, !dbg !11
  %dma.source.ptr1397 = inttoptr i64 %dma.source.opaque1396 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1397, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local925 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local391, !dbg !11
  %dma.source.bits1398 = ptrtoint ptr addrspace(1) %address.local925 to i64, !dbg !11
  %dma.source.opaque1399 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1398) #10, !dbg !11
  %dma.source.ptr1400 = inttoptr i64 %dma.source.opaque1399 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1400, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %410 = or disjoint i64 %137, 96, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %411 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %410, !dbg !12
  %address.local926 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local418, !dbg !12
  %dma.source.bits1401 = ptrtoint ptr addrspace(1) %address.local926 to i64, !dbg !12
  %dma.source.opaque1402 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1401) #10, !dbg !12
  %dma.source.ptr1403 = inttoptr i64 %dma.source.opaque1402 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1403, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local927 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local450, !dbg !12
  %dma.source.bits1404 = ptrtoint ptr addrspace(1) %address.local927 to i64, !dbg !12
  %dma.source.opaque1405 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1404) #10, !dbg !12
  %dma.source.ptr1406 = inttoptr i64 %dma.source.opaque1405 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1406, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local928 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local480, !dbg !12
  %dma.source.bits1407 = ptrtoint ptr addrspace(1) %address.local928 to i64, !dbg !12
  %dma.source.opaque1408 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1407) #10, !dbg !12
  %dma.source.ptr1409 = inttoptr i64 %dma.source.opaque1408 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1409, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local513, !dbg !12
  %dma.source.bits1410 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !12
  %dma.source.opaque1411 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1410) #10, !dbg !12
  %dma.source.ptr1412 = inttoptr i64 %dma.source.opaque1411 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1412, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local547, !dbg !12
  %dma.source.bits1413 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !12
  %dma.source.opaque1414 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1413) #10, !dbg !12
  %dma.source.ptr1415 = inttoptr i64 %dma.source.opaque1414 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1415, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local580, !dbg !12
  %dma.source.bits1416 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !12
  %dma.source.opaque1417 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1416) #10, !dbg !12
  %dma.source.ptr1418 = inttoptr i64 %dma.source.opaque1417 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1418, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local616, !dbg !12
  %dma.source.bits1419 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !12
  %dma.source.opaque1420 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1419) #10, !dbg !12
  %dma.source.ptr1421 = inttoptr i64 %dma.source.opaque1420 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1421, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local649, !dbg !12
  %dma.source.bits1422 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !12
  %dma.source.opaque1423 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1422) #10, !dbg !12
  %dma.source.ptr1424 = inttoptr i64 %dma.source.opaque1423 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1424, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %412 = load <4 x half>, ptr addrspace(3) %address.local659, align 2, !dbg !13
  %413 = load <4 x half>, ptr addrspace(3) %address.local661, align 2, !dbg !13
  %414 = load <4 x half>, ptr addrspace(3) %address.local663, align 2, !dbg !13
  %415 = fmul <4 x half> %412, splat (half 0xH3015), !dbg !13
  %416 = load <4 x half>, ptr addrspace(3) %address.local665, align 2, !dbg !13
  %417 = fmul <4 x half> %413, splat (half 0xH3015), !dbg !13
  %418 = fmul <4 x half> %414, splat (half 0xH3015), !dbg !13
  %419 = fmul <4 x half> %416, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local934 = shl nuw nsw i64 %address.local393, 5, !dbg !15
  %address.local935 = and i64 %address.local934, 8064, !dbg !15
  %address.local936 = shl nuw nsw i64 %address.local393, 3, !dbg !15
  %address.local937 = shl nuw nsw i64 %address.local393, 3, !dbg !15
  %address.local938 = and i64 %address.local937, 2016, !dbg !15
  %address.local939 = sub nsw i64 %address.local936, %address.local938, !dbg !15
  %address.local940 = add nsw i64 %address.local935, %address.local939, !dbg !15
  %420 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14251426 = add nsw i64 %137, %address.local940, !dbg !15
  %.idx14251426.tr = trunc i64 %.idx14251426 to i32, !dbg !15
  %dma.buffer.offset1234 = shl i32 %.idx14251426.tr, 1, !dbg !15
  %421 = insertelement <2 x i64> poison, i64 %420, i64 0, !dbg !15
  %422 = bitcast <2 x i64> %421 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1235 = shufflevector <4 x i32> %422, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %423 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1234, <4 x i32> %dma.buffer.rsrc1235) #11, !dbg !15
  %address.local942 = shl nuw nsw i32 %address.thread329, 3, !dbg !15
  %address.local943 = and i32 %address.local942, 24, !dbg !15
  %address.local944 = shl nuw nsw i32 %address.thread329, 5, !dbg !15
  %address.local945 = and i32 %address.local944, 8064, !dbg !15
  %address.local946 = or disjoint i32 %address.local943, %address.local945, !dbg !15
  %address.local947 = or disjoint i32 %address.local946, 8192, !dbg !15
  %424 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %425 = trunc nuw i64 %137 to i32, !dbg !15
  %.idx14271428.tr = or disjoint i32 %address.local947, %425, !dbg !15
  %dma.buffer.offset1232 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !15
  %426 = insertelement <2 x i64> poison, i64 %424, i64 0, !dbg !15
  %427 = bitcast <2 x i64> %426 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1233 = shufflevector <4 x i32> %427, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1232, <4 x i32> %dma.buffer.rsrc1233) #11, !dbg !15
  %429 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
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
  %442 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
  %443 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %444 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
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
  %472 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
  %473 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %471, <4 x float> %470) #9, !dbg !16
  %475 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
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
  %511 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
  %512 = load <8 x half>, ptr addrspace(3) %address.local688, align 2, !dbg !16
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %510, <4 x float> %509) #9, !dbg !16
  %514 = load <8 x half>, ptr addrspace(3) %address.local713, align 2, !dbg !16
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
  %547 = load <8 x half>, ptr addrspace(3) %address.local737, align 2, !dbg !16
  %548 = load <8 x half>, ptr addrspace(3) %address.local776, align 2, !dbg !16
  %549 = load <8 x half>, ptr addrspace(3) %address.local778, align 2, !dbg !16
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
  %574 = load <8 x half>, ptr addrspace(3) %address.local821, align 2, !dbg !16
  %575 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
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
  %591 = load <8 x half>, ptr addrspace(3) %address.local885, align 2, !dbg !16
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
  %math.result220 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !26, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result220 to <4 x half>, !dbg !26
  %math.packed221 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result222 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !26, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result222 to <4 x half>, !dbg !26
  %math.packed223 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result224 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %758) #9, !dbg !26, !frisk.register_lifetimes !1
  %759 = fptrunc <4 x float> %math.result224 to <4 x half>, !dbg !26
  %math.packed225 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %759) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result226 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !26, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result226 to <4 x half>, !dbg !26
  %math.packed227 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result228 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !26, !frisk.register_lifetimes !1
  %793 = fptrunc <4 x float> %math.result228 to <4 x half>, !dbg !26
  %math.packed229 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result230 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #9, !dbg !26, !frisk.register_lifetimes !1
  %810 = fptrunc <4 x float> %math.result230 to <4 x half>, !dbg !26
  %math.packed231 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result232 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %826) #9, !dbg !26, !frisk.register_lifetimes !1
  %827 = fptrunc <4 x float> %math.result232 to <4 x half>, !dbg !26
  %math.packed233 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %827) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #9, !dbg !26, !frisk.register_lifetimes !1
  %844 = fptrunc <4 x float> %math.result234 to <4 x half>, !dbg !26
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %860) #9, !dbg !26, !frisk.register_lifetimes !1
  %861 = fptrunc <4 x float> %math.result236 to <4 x half>, !dbg !26
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %861) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %877) #9, !dbg !26, !frisk.register_lifetimes !1
  %878 = fptrunc <4 x float> %math.result238 to <4 x half>, !dbg !26
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %878) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %894) #9, !dbg !26, !frisk.register_lifetimes !1
  %895 = fptrunc <4 x float> %math.result240 to <4 x half>, !dbg !26
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %895) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %911) #9, !dbg !26, !frisk.register_lifetimes !1
  %912 = fptrunc <4 x float> %math.result242 to <4 x half>, !dbg !26
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %912) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %928) #9, !dbg !26, !frisk.register_lifetimes !1
  %929 = fptrunc <4 x float> %math.result244 to <4 x half>, !dbg !26
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %929) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %945) #9, !dbg !26, !frisk.register_lifetimes !1
  %946 = fptrunc <4 x float> %math.result246 to <4 x half>, !dbg !26
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %946) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %962) #9, !dbg !26, !frisk.register_lifetimes !1
  %963 = fptrunc <4 x float> %math.result248 to <4 x half>, !dbg !26
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %963) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %address.local950 = shl nuw nsw i32 %address.thread329, 3, !dbg !14
  %address.local951 = and i32 %address.local950, 120, !dbg !14
  %address.local952 = shl nuw nsw i32 %address.thread329, 5, !dbg !14
  %address.local953 = and i32 %address.local952, 1536, !dbg !14
  %address.local954 = or disjoint i32 %address.local951, %address.local953, !dbg !14
  %address.local955 = zext nneg i32 %address.local954 to i64, !dbg !14
  %address.local956 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local955, !dbg !14
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !14
  %address.local957 = and i64 %address.local393, 60, !dbg !14
  %address.local958 = add nuw nsw i64 %address.local957, %address.local393, !dbg !14
  %address.local959 = shl nuw nsw i64 %address.local958, 3, !dbg !14
  %address.local960 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local959, !dbg !14
  %address.local961 = lshr i64 %address.local681, 4, !dbg !14
  %address.local962 = mul nuw nsw i64 %address.local961, 384, !dbg !14
  %address.local963 = getelementptr [2 x i8], ptr addrspace(3) %address.local960, i64 %address.local962, !dbg !14
  %address.local964 = mul nsw i64 %address.local671, -32, !dbg !14
  %address.local965 = getelementptr [2 x i8], ptr addrspace(3) %address.local963, i64 %address.local964, !dbg !14
  %address.local966 = getelementptr i8, ptr addrspace(3) %address.local965, i64 256, !dbg !14
  %999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1000 = bitcast <4 x i32> %999 to <8 x half>, !dbg !14
  %address.local967 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local959, !dbg !14
  %address.local968 = getelementptr [2 x i8], ptr addrspace(3) %address.local967, i64 %address.local962, !dbg !14
  %address.local969 = getelementptr [2 x i8], ptr addrspace(3) %address.local968, i64 %address.local964, !dbg !14
  %address.local970 = getelementptr i8, ptr addrspace(3) %address.local969, i64 512, !dbg !14
  %1001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #0, !dbg !14
  %1002 = bitcast <4 x i32> %1001 to <8 x half>, !dbg !14
  %address.local971 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local959, !dbg !14
  %address.local972 = getelementptr [2 x i8], ptr addrspace(3) %address.local971, i64 %address.local962, !dbg !14
  %address.local973 = getelementptr [2 x i8], ptr addrspace(3) %address.local972, i64 %address.local964, !dbg !14
  %address.local974 = getelementptr i8, ptr addrspace(3) %address.local973, i64 768, !dbg !14
  %1003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #0, !dbg !14
  %1004 = bitcast <4 x i32> %1003 to <8 x half>, !dbg !14
  %address.local975 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local955, !dbg !14
  %address.local976 = getelementptr i8, ptr addrspace(3) %address.local975, i64 4096, !dbg !14
  %1005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local976, i32 0) #0, !dbg !14
  %1006 = bitcast <4 x i32> %1005 to <8 x half>, !dbg !14
  %address.local977 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local959, !dbg !14
  %address.local978 = getelementptr [2 x i8], ptr addrspace(3) %address.local977, i64 %address.local962, !dbg !14
  %address.local979 = getelementptr [2 x i8], ptr addrspace(3) %address.local978, i64 %address.local964, !dbg !14
  %address.local980 = getelementptr i8, ptr addrspace(3) %address.local979, i64 4352, !dbg !14
  %1007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local980, i32 0) #0, !dbg !14
  %1008 = bitcast <4 x i32> %1007 to <8 x half>, !dbg !14
  %address.local981 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local959, !dbg !14
  %address.local982 = getelementptr [2 x i8], ptr addrspace(3) %address.local981, i64 %address.local962, !dbg !14
  %address.local983 = getelementptr [2 x i8], ptr addrspace(3) %address.local982, i64 %address.local964, !dbg !14
  %address.local984 = getelementptr i8, ptr addrspace(3) %address.local983, i64 4608, !dbg !14
  %1009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local984, i32 0) #0, !dbg !14
  %1010 = bitcast <4 x i32> %1009 to <8 x half>, !dbg !14
  %address.local985 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local959, !dbg !14
  %address.local986 = getelementptr [2 x i8], ptr addrspace(3) %address.local985, i64 %address.local962, !dbg !14
  %address.local987 = getelementptr [2 x i8], ptr addrspace(3) %address.local986, i64 %address.local964, !dbg !14
  %address.local988 = getelementptr i8, ptr addrspace(3) %address.local987, i64 4864, !dbg !14
  %1011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local988, i32 0) #0, !dbg !14
  %1012 = bitcast <4 x i32> %1011 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1013 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14291430 = add nsw i64 %230, %address.local940, !dbg !15
  %.idx14291430.tr = trunc i64 %.idx14291430 to i32, !dbg !15
  %dma.buffer.offset1230 = shl i32 %.idx14291430.tr, 1, !dbg !15
  %1014 = insertelement <2 x i64> poison, i64 %1013, i64 0, !dbg !15
  %1015 = bitcast <2 x i64> %1014 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1231 = shufflevector <4 x i32> %1015, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1016 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1230, <4 x i32> %dma.buffer.rsrc1231) #11, !dbg !15
  %1017 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1018 = trunc nuw i64 %230 to i32, !dbg !15
  %.idx14311432.tr = or disjoint i32 %address.local947, %1018, !dbg !15
  %dma.buffer.offset1228 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !15
  %1019 = insertelement <2 x i64> poison, i64 %1017, i64 0, !dbg !15
  %1020 = bitcast <2 x i64> %1019 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1229 = shufflevector <4 x i32> %1020, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1021 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1228, <4 x i32> %dma.buffer.rsrc1229) #11, !dbg !15
  %1022 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1022, <4 x float> %989) #9, !dbg !14
  %1024 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1024, <4 x float> %1023) #9, !dbg !14
  %1026 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1026, <4 x float> %1025) #9, !dbg !14
  %1028 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1028, <4 x float> %1027) #9, !dbg !14
  %1030 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1030, <4 x float> %1029) #9, !dbg !14
  %1032 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1032, <4 x float> %1031) #9, !dbg !14
  %1034 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1034, <4 x float> %1033) #9, !dbg !14
  %1036 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1036, <4 x float> %1035) #9, !dbg !14
  %1038 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1038, <4 x float> %991) #9, !dbg !14
  %1040 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1040, <4 x float> %1039) #9, !dbg !14
  %1042 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1042, <4 x float> %1041) #9, !dbg !14
  %1044 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1044, <4 x float> %1043) #9, !dbg !14
  %1046 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1046, <4 x float> %1045) #9, !dbg !14
  %1048 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1048, <4 x float> %1047) #9, !dbg !14
  %1050 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1050, <4 x float> %1049) #9, !dbg !14
  %1052 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1052, <4 x float> %1051) #9, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1022, <4 x float> %994) #9, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1024, <4 x float> %1054) #9, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1026, <4 x float> %1055) #9, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1028, <4 x float> %1056) #9, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1030, <4 x float> %1057) #9, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1032, <4 x float> %1058) #9, !dbg !14
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1034, <4 x float> %1059) #9, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1036, <4 x float> %1060) #9, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1038, <4 x float> %996) #9, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1040, <4 x float> %1062) #9, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1042, <4 x float> %1063) #9, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1044, <4 x float> %1064) #9, !dbg !14
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1046, <4 x float> %1065) #9, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1048, <4 x float> %1066) #9, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1050, <4 x float> %1067) #9, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1052, <4 x float> %1068) #9, !dbg !14
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
  %1080 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14
  %1081 = bitcast <4 x i32> %1080 to <8 x half>, !dbg !14
  %1082 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1083 = bitcast <4 x i32> %1082 to <8 x half>, !dbg !14
  %1084 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #0, !dbg !14
  %1085 = bitcast <4 x i32> %1084 to <8 x half>, !dbg !14
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #0, !dbg !14
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !14
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local976, i32 0) #0, !dbg !14
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !14
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local980, i32 0) #0, !dbg !14
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !14
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local984, i32 0) #0, !dbg !14
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !14
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local988, i32 0) #0, !dbg !14
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1096 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14331434 = add nsw i64 %320, %address.local940, !dbg !15
  %.idx14331434.tr = trunc i64 %.idx14331434 to i32, !dbg !15
  %dma.buffer.offset1226 = shl i32 %.idx14331434.tr, 1, !dbg !15
  %1097 = insertelement <2 x i64> poison, i64 %1096, i64 0, !dbg !15
  %1098 = bitcast <2 x i64> %1097 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1227 = shufflevector <4 x i32> %1098, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1099 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1226, <4 x i32> %dma.buffer.rsrc1227) #11, !dbg !15
  %1100 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1101 = trunc nuw i64 %320 to i32, !dbg !15
  %.idx14351436.tr = or disjoint i32 %address.local947, %1101, !dbg !15
  %dma.buffer.offset1224 = shl nuw nsw i32 %.idx14351436.tr, 1, !dbg !15
  %1102 = insertelement <2 x i64> poison, i64 %1100, i64 0, !dbg !15
  %1103 = bitcast <2 x i64> %1102 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1225 = shufflevector <4 x i32> %1103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1224, <4 x i32> %dma.buffer.rsrc1225) #11, !dbg !15
  %1105 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1105, <4 x float> %1072) #9, !dbg !14
  %1107 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1107, <4 x float> %1106) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1115, <4 x float> %1114) #9, !dbg !14
  %1117 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1117, <4 x float> %1116) #9, !dbg !14
  %1119 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1119, <4 x float> %1118) #9, !dbg !14
  %1121 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1121, <4 x float> %1074) #9, !dbg !14
  %1123 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1123, <4 x float> %1122) #9, !dbg !14
  %1125 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1125, <4 x float> %1124) #9, !dbg !14
  %1127 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1127, <4 x float> %1126) #9, !dbg !14
  %1129 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1129, <4 x float> %1128) #9, !dbg !14
  %1131 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1131, <4 x float> %1130) #9, !dbg !14
  %1133 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1133, <4 x float> %1132) #9, !dbg !14
  %1135 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1135, <4 x float> %1134) #9, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1105, <4 x float> %1077) #9, !dbg !14
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1107, <4 x float> %1137) #9, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1109, <4 x float> %1138) #9, !dbg !14
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1111, <4 x float> %1139) #9, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1113, <4 x float> %1140) #9, !dbg !14
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1115, <4 x float> %1141) #9, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1117, <4 x float> %1142) #9, !dbg !14
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1119, <4 x float> %1143) #9, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1121, <4 x float> %1079) #9, !dbg !14
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1123, <4 x float> %1145) #9, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1125, <4 x float> %1146) #9, !dbg !14
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1127, <4 x float> %1147) #9, !dbg !14
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1129, <4 x float> %1148) #9, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1131, <4 x float> %1149) #9, !dbg !14
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1133, <4 x float> %1150) #9, !dbg !14
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1135, <4 x float> %1151) #9, !dbg !14
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
  %1163 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14
  %1164 = bitcast <4 x i32> %1163 to <8 x half>, !dbg !14
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !14
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #0, !dbg !14
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #0, !dbg !14
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local976, i32 0) #0, !dbg !14
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local980, i32 0) #0, !dbg !14
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local984, i32 0) #0, !dbg !14
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local988, i32 0) #0, !dbg !14
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1179 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14371438 = add nsw i64 %410, %address.local940, !dbg !15
  %.idx14371438.tr = trunc i64 %.idx14371438 to i32, !dbg !15
  %dma.buffer.offset1222 = shl i32 %.idx14371438.tr, 1, !dbg !15
  %1180 = insertelement <2 x i64> poison, i64 %1179, i64 0, !dbg !15
  %1181 = bitcast <2 x i64> %1180 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1223 = shufflevector <4 x i32> %1181, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1182 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1222, <4 x i32> %dma.buffer.rsrc1223) #11, !dbg !15
  %1183 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.tr = trunc nuw i64 %410 to i32, !dbg !15
  %1184 = or disjoint i32 %address.local947, %.tr, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %1184, 1, !dbg !15
  %1185 = insertelement <2 x i64> poison, i64 %1183, i64 0, !dbg !15
  %1186 = bitcast <2 x i64> %1185 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %1186, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1187 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #11, !dbg !15
  %1188 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1188, <4 x float> %1155) #9, !dbg !14
  %1190 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1190, <4 x float> %1189) #9, !dbg !14
  %1192 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1192, <4 x float> %1191) #9, !dbg !14
  %1194 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1194, <4 x float> %1193) #9, !dbg !14
  %1196 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1196, <4 x float> %1195) #9, !dbg !14
  %1198 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1198, <4 x float> %1197) #9, !dbg !14
  %1200 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1200, <4 x float> %1199) #9, !dbg !14
  %1202 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1202, <4 x float> %1201) #9, !dbg !14
  %1204 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1204, <4 x float> %1157) #9, !dbg !14
  %1206 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1206, <4 x float> %1205) #9, !dbg !14
  %1208 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1208, <4 x float> %1207) #9, !dbg !14
  %1210 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1210, <4 x float> %1209) #9, !dbg !14
  %1212 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1212, <4 x float> %1211) #9, !dbg !14
  %1214 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1214, <4 x float> %1213) #9, !dbg !14
  %1216 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1216, <4 x float> %1215) #9, !dbg !14
  %1218 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1218, <4 x float> %1217) #9, !dbg !14
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1188, <4 x float> %1160) #9, !dbg !14
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1190, <4 x float> %1220) #9, !dbg !14
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1192, <4 x float> %1221) #9, !dbg !14
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1194, <4 x float> %1222) #9, !dbg !14
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1196, <4 x float> %1223) #9, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1198, <4 x float> %1224) #9, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1200, <4 x float> %1225) #9, !dbg !14
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1202, <4 x float> %1226) #9, !dbg !14
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1204, <4 x float> %1162) #9, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1206, <4 x float> %1228) #9, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1208, <4 x float> %1229) #9, !dbg !14
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1210, <4 x float> %1230) #9, !dbg !14
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1212, <4 x float> %1231) #9, !dbg !14
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1214, <4 x float> %1232) #9, !dbg !14
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1216, <4 x float> %1233) #9, !dbg !14
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1218, <4 x float> %1234) #9, !dbg !14
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
  %1246 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14
  %1247 = bitcast <4 x i32> %1246 to <8 x half>, !dbg !14
  %1248 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #0, !dbg !14
  %1249 = bitcast <4 x i32> %1248 to <8 x half>, !dbg !14
  %1250 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #0, !dbg !14
  %1251 = bitcast <4 x i32> %1250 to <8 x half>, !dbg !14
  %1252 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #0, !dbg !14
  %1253 = bitcast <4 x i32> %1252 to <8 x half>, !dbg !14
  %1254 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local976, i32 0) #0, !dbg !14
  %1255 = bitcast <4 x i32> %1254 to <8 x half>, !dbg !14
  %1256 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local980, i32 0) #0, !dbg !14
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !14
  %1258 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local984, i32 0) #0, !dbg !14
  %1259 = bitcast <4 x i32> %1258 to <8 x half>, !dbg !14
  %1260 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local988, i32 0) #0, !dbg !14
  %1261 = bitcast <4 x i32> %1260 to <8 x half>, !dbg !14
  %1262 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1262, <4 x float> %1238) #9, !dbg !14
  %1264 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1264, <4 x float> %1263) #9, !dbg !14
  %1266 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1266, <4 x float> %1265) #9, !dbg !14
  %1268 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1268, <4 x float> %1267) #9, !dbg !14
  %1270 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1270, <4 x float> %1269) #9, !dbg !14
  %1272 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1272, <4 x float> %1271) #9, !dbg !14
  %1274 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1274, <4 x float> %1273) #9, !dbg !14
  %1276 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1276, <4 x float> %1275) #9, !dbg !14
  %1278 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1278, <4 x float> %1240) #9, !dbg !14
  %1280 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed221, <4 x half> %1280, <4 x float> %1279) #9, !dbg !14
  %1282 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed223, <4 x half> %1282, <4 x float> %1281) #9, !dbg !14
  %1284 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed225, <4 x half> %1284, <4 x float> %1283) #9, !dbg !14
  %1286 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed227, <4 x half> %1286, <4 x float> %1285) #9, !dbg !14
  %1288 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed229, <4 x half> %1288, <4 x float> %1287) #9, !dbg !14
  %1290 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed231, <4 x half> %1290, <4 x float> %1289) #9, !dbg !14
  %1292 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed233, <4 x half> %1292, <4 x float> %1291) #9, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1262, <4 x float> %1243) #9, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1264, <4 x float> %1294) #9, !dbg !14
  %1296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1266, <4 x float> %1295) #9, !dbg !14
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1268, <4 x float> %1296) #9, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1270, <4 x float> %1297) #9, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1272, <4 x float> %1298) #9, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1274, <4 x float> %1299) #9, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1276, <4 x float> %1300) #9, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1278, <4 x float> %1245) #9, !dbg !14
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1280, <4 x float> %1302) #9, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1282, <4 x float> %1303) #9, !dbg !14
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1284, <4 x float> %1304) #9, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1286, <4 x float> %1305) #9, !dbg !14
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1288, <4 x float> %1306) #9, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1290, <4 x float> %1307) #9, !dbg !14
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1292, <4 x float> %1308) #9, !dbg !14
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
  %1330 = and i32 %31, 16, !dbg !6
  %.not = icmp eq i32 %1330, 0, !dbg !6
  %1331 = and i32 %31, 32, !dbg !6
  %.not65 = icmp eq i32 %1331, 0, !dbg !6
  %1332 = extractvalue [2 x <8 x float>] %131, 0, !dbg !6
  %1333 = extractelement <8 x float> %1332, i64 0, !dbg !6
  %1334 = extractelement <8 x float> %1332, i64 1, !dbg !6
  %1335 = extractelement <8 x float> %1332, i64 2, !dbg !6
  %1336 = extractelement <8 x float> %1332, i64 3, !dbg !6
  %1337 = select i1 %.not, float %1334, float %1333, !dbg !6
  %1338 = bitcast float %1337 to i32, !dbg !6
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1338), !dbg !6
  %1340 = bitcast i32 %1339 to float, !dbg !6
  %1341 = select i1 %.not, float %1333, float %1340, !dbg !6
  %1342 = select i1 %.not, float %1340, float %1334, !dbg !6
  %1343 = select i1 %.not, float %1336, float %1335, !dbg !6
  %1344 = bitcast float %1343 to i32, !dbg !6
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1344), !dbg !6
  %1346 = bitcast i32 %1345 to float, !dbg !6
  %1347 = select i1 %.not, float %1335, float %1346, !dbg !6
  %1348 = select i1 %.not, float %1346, float %1336, !dbg !6
  %1349 = select i1 %.not65, float %1347, float %1341, !dbg !6
  %1350 = bitcast float %1349 to i32, !dbg !6
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1350), !dbg !6
  %1352 = bitcast i32 %1351 to float, !dbg !6
  %1353 = select i1 %.not65, float %1341, float %1352, !dbg !6
  %1354 = select i1 %.not65, float %1352, float %1347, !dbg !6
  %1355 = select i1 %.not65, float %1348, float %1342, !dbg !6
  %1356 = bitcast float %1355 to i32, !dbg !6
  %1357 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1356), !dbg !6
  %1358 = bitcast i32 %1357 to float, !dbg !6
  %1359 = select i1 %.not65, float %1342, float %1358, !dbg !6
  %1360 = select i1 %.not65, float %1358, float %1348, !dbg !6
  %1361 = extractelement <8 x float> %1332, i64 4, !dbg !6
  %1362 = extractelement <8 x float> %1332, i64 5, !dbg !6
  %1363 = extractelement <8 x float> %1332, i64 6, !dbg !6
  %1364 = extractelement <8 x float> %1332, i64 7, !dbg !6
  %1365 = select i1 %.not, float %1362, float %1361, !dbg !6
  %1366 = bitcast float %1365 to i32, !dbg !6
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1366), !dbg !6
  %1368 = bitcast i32 %1367 to float, !dbg !6
  %1369 = select i1 %.not, float %1361, float %1368, !dbg !6
  %1370 = select i1 %.not, float %1368, float %1362, !dbg !6
  %1371 = select i1 %.not, float %1364, float %1363, !dbg !6
  %1372 = bitcast float %1371 to i32, !dbg !6
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1372), !dbg !6
  %1374 = bitcast i32 %1373 to float, !dbg !6
  %1375 = select i1 %.not, float %1363, float %1374, !dbg !6
  %1376 = select i1 %.not, float %1374, float %1364, !dbg !6
  %1377 = select i1 %.not65, float %1375, float %1369, !dbg !6
  %1378 = bitcast float %1377 to i32, !dbg !6
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1378), !dbg !6
  %1380 = bitcast i32 %1379 to float, !dbg !6
  %1381 = select i1 %.not65, float %1369, float %1380, !dbg !6
  %1382 = select i1 %.not65, float %1380, float %1375, !dbg !6
  %1383 = select i1 %.not65, float %1376, float %1370, !dbg !6
  %1384 = bitcast float %1383 to i32, !dbg !6
  %1385 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1384), !dbg !6
  %1386 = bitcast i32 %1385 to float, !dbg !6
  %1387 = select i1 %.not65, float %1370, float %1386, !dbg !6
  %1388 = select i1 %.not65, float %1386, float %1376, !dbg !6
  %1389 = extractvalue [2 x <8 x float>] %131, 1, !dbg !6
  %1390 = extractelement <8 x float> %1389, i64 0, !dbg !6
  %1391 = extractelement <8 x float> %1389, i64 1, !dbg !6
  %1392 = extractelement <8 x float> %1389, i64 2, !dbg !6
  %1393 = extractelement <8 x float> %1389, i64 3, !dbg !6
  %1394 = select i1 %.not, float %1391, float %1390, !dbg !6
  %1395 = bitcast float %1394 to i32, !dbg !6
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1395), !dbg !6
  %1397 = bitcast i32 %1396 to float, !dbg !6
  %1398 = select i1 %.not, float %1390, float %1397, !dbg !6
  %1399 = select i1 %.not, float %1397, float %1391, !dbg !6
  %1400 = select i1 %.not, float %1393, float %1392, !dbg !6
  %1401 = bitcast float %1400 to i32, !dbg !6
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1401), !dbg !6
  %1403 = bitcast i32 %1402 to float, !dbg !6
  %1404 = select i1 %.not, float %1392, float %1403, !dbg !6
  %1405 = select i1 %.not, float %1403, float %1393, !dbg !6
  %1406 = select i1 %.not65, float %1404, float %1398, !dbg !6
  %1407 = bitcast float %1406 to i32, !dbg !6
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1407), !dbg !6
  %1409 = bitcast i32 %1408 to float, !dbg !6
  %1410 = select i1 %.not65, float %1398, float %1409, !dbg !6
  %1411 = select i1 %.not65, float %1409, float %1404, !dbg !6
  %1412 = select i1 %.not65, float %1405, float %1399, !dbg !6
  %1413 = bitcast float %1412 to i32, !dbg !6
  %1414 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1413), !dbg !6
  %1415 = bitcast i32 %1414 to float, !dbg !6
  %1416 = select i1 %.not65, float %1399, float %1415, !dbg !6
  %1417 = select i1 %.not65, float %1415, float %1405, !dbg !6
  %1418 = extractelement <8 x float> %1389, i64 4, !dbg !6
  %1419 = extractelement <8 x float> %1389, i64 5, !dbg !6
  %1420 = extractelement <8 x float> %1389, i64 6, !dbg !6
  %1421 = extractelement <8 x float> %1389, i64 7, !dbg !6
  %1422 = select i1 %.not, float %1419, float %1418, !dbg !6
  %1423 = bitcast float %1422 to i32, !dbg !6
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1423), !dbg !6
  %1425 = bitcast i32 %1424 to float, !dbg !6
  %1426 = select i1 %.not, float %1418, float %1425, !dbg !6
  %1427 = select i1 %.not, float %1425, float %1419, !dbg !6
  %1428 = select i1 %.not, float %1421, float %1420, !dbg !6
  %1429 = bitcast float %1428 to i32, !dbg !6
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1429), !dbg !6
  %1431 = bitcast i32 %1430 to float, !dbg !6
  %1432 = select i1 %.not, float %1420, float %1431, !dbg !6
  %1433 = select i1 %.not, float %1431, float %1421, !dbg !6
  %1434 = select i1 %.not65, float %1432, float %1426, !dbg !6
  %1435 = bitcast float %1434 to i32, !dbg !6
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1435), !dbg !6
  %1437 = bitcast i32 %1436 to float, !dbg !6
  %1438 = select i1 %.not65, float %1426, float %1437, !dbg !6
  %1439 = select i1 %.not65, float %1437, float %1432, !dbg !6
  %1440 = select i1 %.not65, float %1433, float %1427, !dbg !6
  %1441 = bitcast float %1440 to i32, !dbg !6
  %1442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1441), !dbg !6
  %1443 = bitcast i32 %1442 to float, !dbg !6
  %1444 = select i1 %.not65, float %1427, float %1443, !dbg !6
  %1445 = select i1 %.not65, float %1443, float %1433, !dbg !6
  %1446 = extractvalue [2 x <8 x float>] %130, 0, !dbg !6
  %1447 = extractelement <8 x float> %1446, i64 0, !dbg !6
  %1448 = extractelement <8 x float> %1446, i64 1, !dbg !6
  %1449 = extractelement <8 x float> %1446, i64 2, !dbg !6
  %1450 = extractelement <8 x float> %1446, i64 3, !dbg !6
  %1451 = select i1 %.not, float %1448, float %1447, !dbg !6
  %1452 = bitcast float %1451 to i32, !dbg !6
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1452), !dbg !6
  %1454 = bitcast i32 %1453 to float, !dbg !6
  %1455 = select i1 %.not, float %1447, float %1454, !dbg !6
  %1456 = select i1 %.not, float %1454, float %1448, !dbg !6
  %1457 = select i1 %.not, float %1450, float %1449, !dbg !6
  %1458 = bitcast float %1457 to i32, !dbg !6
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1458), !dbg !6
  %1460 = bitcast i32 %1459 to float, !dbg !6
  %1461 = select i1 %.not, float %1449, float %1460, !dbg !6
  %1462 = select i1 %.not, float %1460, float %1450, !dbg !6
  %1463 = select i1 %.not65, float %1461, float %1455, !dbg !6
  %1464 = bitcast float %1463 to i32, !dbg !6
  %1465 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1464), !dbg !6
  %1466 = bitcast i32 %1465 to float, !dbg !6
  %1467 = select i1 %.not65, float %1455, float %1466, !dbg !6
  %1468 = select i1 %.not65, float %1466, float %1461, !dbg !6
  %1469 = select i1 %.not65, float %1462, float %1456, !dbg !6
  %1470 = bitcast float %1469 to i32, !dbg !6
  %1471 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1470), !dbg !6
  %1472 = bitcast i32 %1471 to float, !dbg !6
  %1473 = select i1 %.not65, float %1456, float %1472, !dbg !6
  %1474 = select i1 %.not65, float %1472, float %1462, !dbg !6
  %1475 = extractelement <8 x float> %1446, i64 4, !dbg !6
  %1476 = extractelement <8 x float> %1446, i64 5, !dbg !6
  %1477 = extractelement <8 x float> %1446, i64 6, !dbg !6
  %1478 = extractelement <8 x float> %1446, i64 7, !dbg !6
  %1479 = select i1 %.not, float %1476, float %1475, !dbg !6
  %1480 = bitcast float %1479 to i32, !dbg !6
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1480), !dbg !6
  %1482 = bitcast i32 %1481 to float, !dbg !6
  %1483 = select i1 %.not, float %1475, float %1482, !dbg !6
  %1484 = select i1 %.not, float %1482, float %1476, !dbg !6
  %1485 = select i1 %.not, float %1478, float %1477, !dbg !6
  %1486 = bitcast float %1485 to i32, !dbg !6
  %1487 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1486), !dbg !6
  %1488 = bitcast i32 %1487 to float, !dbg !6
  %1489 = select i1 %.not, float %1477, float %1488, !dbg !6
  %1490 = select i1 %.not, float %1488, float %1478, !dbg !6
  %1491 = select i1 %.not65, float %1489, float %1483, !dbg !6
  %1492 = bitcast float %1491 to i32, !dbg !6
  %1493 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1492), !dbg !6
  %1494 = bitcast i32 %1493 to float, !dbg !6
  %1495 = select i1 %.not65, float %1483, float %1494, !dbg !6
  %1496 = select i1 %.not65, float %1494, float %1489, !dbg !6
  %1497 = select i1 %.not65, float %1490, float %1484, !dbg !6
  %1498 = bitcast float %1497 to i32, !dbg !6
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1498), !dbg !6
  %1500 = bitcast i32 %1499 to float, !dbg !6
  %1501 = select i1 %.not65, float %1484, float %1500, !dbg !6
  %1502 = select i1 %.not65, float %1500, float %1490, !dbg !6
  %1503 = extractvalue [2 x <8 x float>] %130, 1, !dbg !6
  %1504 = extractelement <8 x float> %1503, i64 0, !dbg !6
  %1505 = extractelement <8 x float> %1503, i64 1, !dbg !6
  %1506 = extractelement <8 x float> %1503, i64 2, !dbg !6
  %1507 = extractelement <8 x float> %1503, i64 3, !dbg !6
  %1508 = select i1 %.not, float %1505, float %1504, !dbg !6
  %1509 = bitcast float %1508 to i32, !dbg !6
  %1510 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1509), !dbg !6
  %1511 = bitcast i32 %1510 to float, !dbg !6
  %1512 = select i1 %.not, float %1504, float %1511, !dbg !6
  %1513 = select i1 %.not, float %1511, float %1505, !dbg !6
  %1514 = select i1 %.not, float %1507, float %1506, !dbg !6
  %1515 = bitcast float %1514 to i32, !dbg !6
  %1516 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1515), !dbg !6
  %1517 = bitcast i32 %1516 to float, !dbg !6
  %1518 = select i1 %.not, float %1506, float %1517, !dbg !6
  %1519 = select i1 %.not, float %1517, float %1507, !dbg !6
  %1520 = select i1 %.not65, float %1518, float %1512, !dbg !6
  %1521 = bitcast float %1520 to i32, !dbg !6
  %1522 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1521), !dbg !6
  %1523 = bitcast i32 %1522 to float, !dbg !6
  %1524 = select i1 %.not65, float %1512, float %1523, !dbg !6
  %1525 = select i1 %.not65, float %1523, float %1518, !dbg !6
  %1526 = select i1 %.not65, float %1519, float %1513, !dbg !6
  %1527 = bitcast float %1526 to i32, !dbg !6
  %1528 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1527), !dbg !6
  %1529 = bitcast i32 %1528 to float, !dbg !6
  %1530 = select i1 %.not65, float %1513, float %1529, !dbg !6
  %1531 = select i1 %.not65, float %1529, float %1519, !dbg !6
  %1532 = extractelement <8 x float> %1503, i64 4, !dbg !6
  %1533 = extractelement <8 x float> %1503, i64 5, !dbg !6
  %1534 = extractelement <8 x float> %1503, i64 6, !dbg !6
  %1535 = extractelement <8 x float> %1503, i64 7, !dbg !6
  %1536 = select i1 %.not, float %1533, float %1532, !dbg !6
  %1537 = bitcast float %1536 to i32, !dbg !6
  %1538 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1537), !dbg !6
  %1539 = bitcast i32 %1538 to float, !dbg !6
  %1540 = select i1 %.not, float %1532, float %1539, !dbg !6
  %1541 = select i1 %.not, float %1539, float %1533, !dbg !6
  %1542 = select i1 %.not, float %1535, float %1534, !dbg !6
  %1543 = bitcast float %1542 to i32, !dbg !6
  %1544 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1543), !dbg !6
  %1545 = bitcast i32 %1544 to float, !dbg !6
  %1546 = select i1 %.not, float %1534, float %1545, !dbg !6
  %1547 = select i1 %.not, float %1545, float %1535, !dbg !6
  %1548 = select i1 %.not65, float %1546, float %1540, !dbg !6
  %1549 = bitcast float %1548 to i32, !dbg !6
  %1550 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1549), !dbg !6
  %1551 = bitcast i32 %1550 to float, !dbg !6
  %1552 = select i1 %.not65, float %1540, float %1551, !dbg !6
  %1553 = select i1 %.not65, float %1551, float %1546, !dbg !6
  %1554 = select i1 %.not65, float %1547, float %1541, !dbg !6
  %1555 = bitcast float %1554 to i32, !dbg !6
  %1556 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1555), !dbg !6
  %1557 = bitcast i32 %1556 to float, !dbg !6
  %1558 = select i1 %.not65, float %1541, float %1557, !dbg !6
  %1559 = select i1 %.not65, float %1557, float %1547, !dbg !6
  %1560 = extractvalue [2 x <8 x float>] %129, 0, !dbg !6
  %1561 = extractelement <8 x float> %1560, i64 0, !dbg !6
  %1562 = extractelement <8 x float> %1560, i64 1, !dbg !6
  %1563 = extractelement <8 x float> %1560, i64 2, !dbg !6
  %1564 = extractelement <8 x float> %1560, i64 3, !dbg !6
  %1565 = select i1 %.not, float %1562, float %1561, !dbg !6
  %1566 = bitcast float %1565 to i32, !dbg !6
  %1567 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1566), !dbg !6
  %1568 = bitcast i32 %1567 to float, !dbg !6
  %1569 = select i1 %.not, float %1561, float %1568, !dbg !6
  %1570 = select i1 %.not, float %1568, float %1562, !dbg !6
  %1571 = select i1 %.not, float %1564, float %1563, !dbg !6
  %1572 = bitcast float %1571 to i32, !dbg !6
  %1573 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1572), !dbg !6
  %1574 = bitcast i32 %1573 to float, !dbg !6
  %1575 = select i1 %.not, float %1563, float %1574, !dbg !6
  %1576 = select i1 %.not, float %1574, float %1564, !dbg !6
  %1577 = select i1 %.not65, float %1575, float %1569, !dbg !6
  %1578 = bitcast float %1577 to i32, !dbg !6
  %1579 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1578), !dbg !6
  %1580 = bitcast i32 %1579 to float, !dbg !6
  %1581 = select i1 %.not65, float %1569, float %1580, !dbg !6
  %1582 = select i1 %.not65, float %1580, float %1575, !dbg !6
  %1583 = select i1 %.not65, float %1576, float %1570, !dbg !6
  %1584 = bitcast float %1583 to i32, !dbg !6
  %1585 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1584), !dbg !6
  %1586 = bitcast i32 %1585 to float, !dbg !6
  %1587 = select i1 %.not65, float %1570, float %1586, !dbg !6
  %1588 = select i1 %.not65, float %1586, float %1576, !dbg !6
  %1589 = extractelement <8 x float> %1560, i64 4, !dbg !6
  %1590 = extractelement <8 x float> %1560, i64 5, !dbg !6
  %1591 = extractelement <8 x float> %1560, i64 6, !dbg !6
  %1592 = extractelement <8 x float> %1560, i64 7, !dbg !6
  %1593 = select i1 %.not, float %1590, float %1589, !dbg !6
  %1594 = bitcast float %1593 to i32, !dbg !6
  %1595 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1594), !dbg !6
  %1596 = bitcast i32 %1595 to float, !dbg !6
  %1597 = select i1 %.not, float %1589, float %1596, !dbg !6
  %1598 = select i1 %.not, float %1596, float %1590, !dbg !6
  %1599 = select i1 %.not, float %1592, float %1591, !dbg !6
  %1600 = bitcast float %1599 to i32, !dbg !6
  %1601 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1600), !dbg !6
  %1602 = bitcast i32 %1601 to float, !dbg !6
  %1603 = select i1 %.not, float %1591, float %1602, !dbg !6
  %1604 = select i1 %.not, float %1602, float %1592, !dbg !6
  %1605 = select i1 %.not65, float %1603, float %1597, !dbg !6
  %1606 = bitcast float %1605 to i32, !dbg !6
  %1607 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1606), !dbg !6
  %1608 = bitcast i32 %1607 to float, !dbg !6
  %1609 = select i1 %.not65, float %1597, float %1608, !dbg !6
  %1610 = select i1 %.not65, float %1608, float %1603, !dbg !6
  %1611 = select i1 %.not65, float %1604, float %1598, !dbg !6
  %1612 = bitcast float %1611 to i32, !dbg !6
  %1613 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1612), !dbg !6
  %1614 = bitcast i32 %1613 to float, !dbg !6
  %1615 = select i1 %.not65, float %1598, float %1614, !dbg !6
  %1616 = select i1 %.not65, float %1614, float %1604, !dbg !6
  %1617 = extractvalue [2 x <8 x float>] %129, 1, !dbg !6
  %1618 = extractelement <8 x float> %1617, i64 0, !dbg !6
  %1619 = extractelement <8 x float> %1617, i64 1, !dbg !6
  %1620 = extractelement <8 x float> %1617, i64 2, !dbg !6
  %1621 = extractelement <8 x float> %1617, i64 3, !dbg !6
  %1622 = select i1 %.not, float %1619, float %1618, !dbg !6
  %1623 = bitcast float %1622 to i32, !dbg !6
  %1624 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1623), !dbg !6
  %1625 = bitcast i32 %1624 to float, !dbg !6
  %1626 = select i1 %.not, float %1618, float %1625, !dbg !6
  %1627 = select i1 %.not, float %1625, float %1619, !dbg !6
  %1628 = select i1 %.not, float %1621, float %1620, !dbg !6
  %1629 = bitcast float %1628 to i32, !dbg !6
  %1630 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1629), !dbg !6
  %1631 = bitcast i32 %1630 to float, !dbg !6
  %1632 = select i1 %.not, float %1620, float %1631, !dbg !6
  %1633 = select i1 %.not, float %1631, float %1621, !dbg !6
  %1634 = select i1 %.not65, float %1632, float %1626, !dbg !6
  %1635 = bitcast float %1634 to i32, !dbg !6
  %1636 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1635), !dbg !6
  %1637 = bitcast i32 %1636 to float, !dbg !6
  %1638 = select i1 %.not65, float %1626, float %1637, !dbg !6
  %1639 = select i1 %.not65, float %1637, float %1632, !dbg !6
  %1640 = select i1 %.not65, float %1633, float %1627, !dbg !6
  %1641 = bitcast float %1640 to i32, !dbg !6
  %1642 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1641), !dbg !6
  %1643 = bitcast i32 %1642 to float, !dbg !6
  %1644 = select i1 %.not65, float %1627, float %1643, !dbg !6
  %1645 = select i1 %.not65, float %1643, float %1633, !dbg !6
  %1646 = extractelement <8 x float> %1617, i64 4, !dbg !6
  %1647 = extractelement <8 x float> %1617, i64 5, !dbg !6
  %1648 = extractelement <8 x float> %1617, i64 6, !dbg !6
  %1649 = extractelement <8 x float> %1617, i64 7, !dbg !6
  %1650 = select i1 %.not, float %1647, float %1646, !dbg !6
  %1651 = bitcast float %1650 to i32, !dbg !6
  %1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1651), !dbg !6
  %1653 = bitcast i32 %1652 to float, !dbg !6
  %1654 = select i1 %.not, float %1646, float %1653, !dbg !6
  %1655 = select i1 %.not, float %1653, float %1647, !dbg !6
  %1656 = select i1 %.not, float %1649, float %1648, !dbg !6
  %1657 = bitcast float %1656 to i32, !dbg !6
  %1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1657), !dbg !6
  %1659 = bitcast i32 %1658 to float, !dbg !6
  %1660 = select i1 %.not, float %1648, float %1659, !dbg !6
  %1661 = select i1 %.not, float %1659, float %1649, !dbg !6
  %1662 = select i1 %.not65, float %1660, float %1654, !dbg !6
  %1663 = bitcast float %1662 to i32, !dbg !6
  %1664 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1663), !dbg !6
  %1665 = bitcast i32 %1664 to float, !dbg !6
  %1666 = select i1 %.not65, float %1654, float %1665, !dbg !6
  %1667 = select i1 %.not65, float %1665, float %1660, !dbg !6
  %1668 = select i1 %.not65, float %1661, float %1655, !dbg !6
  %1669 = bitcast float %1668 to i32, !dbg !6
  %1670 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1669), !dbg !6
  %1671 = bitcast i32 %1670 to float, !dbg !6
  %1672 = select i1 %.not65, float %1655, float %1671, !dbg !6
  %1673 = select i1 %.not65, float %1671, float %1661, !dbg !6
  %1674 = extractvalue [2 x <8 x float>] %126, 0, !dbg !6
  %1675 = extractelement <8 x float> %1674, i64 0, !dbg !6
  %1676 = extractelement <8 x float> %1674, i64 1, !dbg !6
  %1677 = extractelement <8 x float> %1674, i64 2, !dbg !6
  %1678 = extractelement <8 x float> %1674, i64 3, !dbg !6
  %1679 = select i1 %.not, float %1676, float %1675, !dbg !6
  %1680 = bitcast float %1679 to i32, !dbg !6
  %1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1680), !dbg !6
  %1682 = bitcast i32 %1681 to float, !dbg !6
  %1683 = select i1 %.not, float %1675, float %1682, !dbg !6
  %1684 = select i1 %.not, float %1682, float %1676, !dbg !6
  %1685 = select i1 %.not, float %1678, float %1677, !dbg !6
  %1686 = bitcast float %1685 to i32, !dbg !6
  %1687 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1686), !dbg !6
  %1688 = bitcast i32 %1687 to float, !dbg !6
  %1689 = select i1 %.not, float %1677, float %1688, !dbg !6
  %1690 = select i1 %.not, float %1688, float %1678, !dbg !6
  %1691 = select i1 %.not65, float %1689, float %1683, !dbg !6
  %1692 = bitcast float %1691 to i32, !dbg !6
  %1693 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1692), !dbg !6
  %1694 = select i1 %.not65, float %1690, float %1684, !dbg !6
  %1695 = bitcast float %1694 to i32, !dbg !6
  %1696 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1695), !dbg !6
  %1697 = extractelement <8 x float> %1674, i64 4, !dbg !6
  %1698 = extractelement <8 x float> %1674, i64 5, !dbg !6
  %1699 = extractelement <8 x float> %1674, i64 6, !dbg !6
  %1700 = extractelement <8 x float> %1674, i64 7, !dbg !6
  %1701 = select i1 %.not, float %1698, float %1697, !dbg !6
  %1702 = bitcast float %1701 to i32, !dbg !6
  %1703 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1702), !dbg !6
  %1704 = bitcast i32 %1703 to float, !dbg !6
  %1705 = select i1 %.not, float %1697, float %1704, !dbg !6
  %1706 = select i1 %.not, float %1704, float %1698, !dbg !6
  %1707 = select i1 %.not, float %1700, float %1699, !dbg !6
  %1708 = bitcast float %1707 to i32, !dbg !6
  %1709 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1708), !dbg !6
  %1710 = bitcast i32 %1709 to float, !dbg !6
  %1711 = select i1 %.not, float %1699, float %1710, !dbg !6
  %1712 = select i1 %.not, float %1710, float %1700, !dbg !6
  %1713 = select i1 %.not65, float %1711, float %1705, !dbg !6
  %1714 = bitcast float %1713 to i32, !dbg !6
  %1715 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1714), !dbg !6
  %1716 = select i1 %.not65, float %1712, float %1706, !dbg !6
  %1717 = bitcast float %1716 to i32, !dbg !6
  %1718 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1717), !dbg !6
  %1719 = extractvalue [2 x <8 x float>] %126, 1, !dbg !6
  %1720 = extractelement <8 x float> %1719, i64 0, !dbg !6
  %1721 = extractelement <8 x float> %1719, i64 1, !dbg !6
  %1722 = extractelement <8 x float> %1719, i64 2, !dbg !6
  %1723 = extractelement <8 x float> %1719, i64 3, !dbg !6
  %1724 = select i1 %.not, float %1721, float %1720, !dbg !6
  %1725 = bitcast float %1724 to i32, !dbg !6
  %1726 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1725), !dbg !6
  %1727 = bitcast i32 %1726 to float, !dbg !6
  %1728 = select i1 %.not, float %1720, float %1727, !dbg !6
  %1729 = select i1 %.not, float %1727, float %1721, !dbg !6
  %1730 = select i1 %.not, float %1723, float %1722, !dbg !6
  %1731 = bitcast float %1730 to i32, !dbg !6
  %1732 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1731), !dbg !6
  %1733 = bitcast i32 %1732 to float, !dbg !6
  %1734 = select i1 %.not, float %1722, float %1733, !dbg !6
  %1735 = select i1 %.not, float %1733, float %1723, !dbg !6
  %1736 = select i1 %.not65, float %1734, float %1728, !dbg !6
  %1737 = bitcast float %1736 to i32, !dbg !6
  %1738 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1737), !dbg !6
  %1739 = select i1 %.not65, float %1735, float %1729, !dbg !6
  %1740 = bitcast float %1739 to i32, !dbg !6
  %1741 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1740), !dbg !6
  %1742 = extractelement <8 x float> %1719, i64 4, !dbg !6
  %1743 = extractelement <8 x float> %1719, i64 5, !dbg !6
  %1744 = extractelement <8 x float> %1719, i64 6, !dbg !6
  %1745 = extractelement <8 x float> %1719, i64 7, !dbg !6
  %1746 = select i1 %.not, float %1743, float %1742, !dbg !6
  %1747 = bitcast float %1746 to i32, !dbg !6
  %1748 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1747), !dbg !6
  %1749 = bitcast i32 %1748 to float, !dbg !6
  %1750 = select i1 %.not, float %1742, float %1749, !dbg !6
  %1751 = select i1 %.not, float %1749, float %1743, !dbg !6
  %1752 = select i1 %.not, float %1745, float %1744, !dbg !6
  %1753 = bitcast float %1752 to i32, !dbg !6
  %1754 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1753), !dbg !6
  %1755 = bitcast i32 %1754 to float, !dbg !6
  %1756 = select i1 %.not, float %1744, float %1755, !dbg !6
  %1757 = select i1 %.not, float %1755, float %1745, !dbg !6
  %1758 = select i1 %.not65, float %1756, float %1750, !dbg !6
  %1759 = bitcast float %1758 to i32, !dbg !6
  %1760 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1759), !dbg !6
  %1761 = select i1 %.not65, float %1757, float %1751, !dbg !6
  %1762 = bitcast float %1761 to i32, !dbg !6
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1762), !dbg !6
  %1764 = extractvalue [2 x <1 x float>] %127, 0, !dbg !17
  %1765 = shufflevector <1 x float> %1764, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1766 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1765, !dbg !17
  %1767 = extractvalue [2 x <1 x float>] %127, 1, !dbg !17
  %1768 = shufflevector <1 x float> %1767, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1769 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1768, !dbg !17
  %1770 = bitcast i32 %1760 to float, !dbg !6
  %1771 = select i1 %.not65, float %1750, float %1770, !dbg !6
  %1772 = insertelement <4 x float> poison, float %1771, i64 0, !dbg !17
  %1773 = bitcast i32 %1763 to float, !dbg !6
  %1774 = select i1 %.not65, float %1751, float %1773, !dbg !6
  %1775 = insertelement <4 x float> %1772, float %1774, i64 1, !dbg !17
  %1776 = select i1 %.not65, float %1770, float %1756, !dbg !6
  %1777 = insertelement <4 x float> %1775, float %1776, i64 2, !dbg !17
  %1778 = select i1 %.not65, float %1773, float %1757, !dbg !6
  %1779 = insertelement <4 x float> %1777, float %1778, i64 3, !dbg !17
  %1780 = fmul <4 x float> %1779, %1769, !dbg !17
  %1781 = fptrunc <4 x float> %1780 to <4 x half>, !dbg !18
  %1782 = extractelement <4 x half> %1781, i64 3, !dbg !18
  %1783 = extractelement <4 x half> %1781, i64 2, !dbg !18
  %1784 = extractelement <4 x half> %1781, i64 1, !dbg !18
  %1785 = extractelement <4 x half> %1781, i64 0, !dbg !18
  %1786 = bitcast i32 %1738 to float, !dbg !6
  %1787 = select i1 %.not65, float %1728, float %1786, !dbg !6
  %1788 = insertelement <4 x float> poison, float %1787, i64 0, !dbg !17
  %1789 = bitcast i32 %1741 to float, !dbg !6
  %1790 = select i1 %.not65, float %1729, float %1789, !dbg !6
  %1791 = insertelement <4 x float> %1788, float %1790, i64 1, !dbg !17
  %1792 = select i1 %.not65, float %1786, float %1734, !dbg !6
  %1793 = insertelement <4 x float> %1791, float %1792, i64 2, !dbg !17
  %1794 = select i1 %.not65, float %1789, float %1735, !dbg !6
  %1795 = insertelement <4 x float> %1793, float %1794, i64 3, !dbg !17
  %1796 = fmul <4 x float> %1795, %1769, !dbg !17
  %1797 = fptrunc <4 x float> %1796 to <4 x half>, !dbg !18
  %1798 = extractelement <4 x half> %1797, i64 3, !dbg !18
  %1799 = extractelement <4 x half> %1797, i64 2, !dbg !18
  %1800 = extractelement <4 x half> %1797, i64 1, !dbg !18
  %1801 = extractelement <4 x half> %1797, i64 0, !dbg !18
  %1802 = bitcast i32 %1715 to float, !dbg !6
  %1803 = select i1 %.not65, float %1705, float %1802, !dbg !6
  %1804 = insertelement <4 x float> poison, float %1803, i64 0, !dbg !17
  %1805 = bitcast i32 %1718 to float, !dbg !6
  %1806 = select i1 %.not65, float %1706, float %1805, !dbg !6
  %1807 = insertelement <4 x float> %1804, float %1806, i64 1, !dbg !17
  %1808 = select i1 %.not65, float %1802, float %1711, !dbg !6
  %1809 = insertelement <4 x float> %1807, float %1808, i64 2, !dbg !17
  %1810 = select i1 %.not65, float %1805, float %1712, !dbg !6
  %1811 = insertelement <4 x float> %1809, float %1810, i64 3, !dbg !17
  %1812 = fmul <4 x float> %1811, %1766, !dbg !17
  %1813 = fptrunc <4 x float> %1812 to <4 x half>, !dbg !18
  %1814 = extractelement <4 x half> %1813, i64 3, !dbg !18
  %1815 = extractelement <4 x half> %1813, i64 2, !dbg !18
  %1816 = extractelement <4 x half> %1813, i64 1, !dbg !18
  %1817 = extractelement <4 x half> %1813, i64 0, !dbg !18
  %1818 = bitcast i32 %1693 to float, !dbg !6
  %1819 = select i1 %.not65, float %1683, float %1818, !dbg !6
  %1820 = insertelement <4 x float> poison, float %1819, i64 0, !dbg !17
  %1821 = bitcast i32 %1696 to float, !dbg !6
  %1822 = select i1 %.not65, float %1684, float %1821, !dbg !6
  %1823 = insertelement <4 x float> %1820, float %1822, i64 1, !dbg !17
  %1824 = select i1 %.not65, float %1818, float %1689, !dbg !6
  %1825 = insertelement <4 x float> %1823, float %1824, i64 2, !dbg !17
  %1826 = select i1 %.not65, float %1821, float %1690, !dbg !6
  %1827 = insertelement <4 x float> %1825, float %1826, i64 3, !dbg !17
  %1828 = fmul <4 x float> %1827, %1766, !dbg !17
  %1829 = fptrunc <4 x float> %1828 to <4 x half>, !dbg !18
  %1830 = extractelement <4 x half> %1829, i64 3, !dbg !18
  %1831 = extractelement <4 x half> %1829, i64 2, !dbg !18
  %1832 = extractelement <4 x half> %1829, i64 1, !dbg !18
  %1833 = extractelement <4 x half> %1829, i64 0, !dbg !18
  %1834 = shl nuw nsw i64 %26, 19, !dbg !19
  %address.thread995 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local996 = zext nneg i32 %address.thread995 to i64, !dbg !13
  %address.local997 = add nsw i64 %27, %address.local996, !dbg !19
  %1835 = lshr i64 %address.local996, 2, !dbg !19
  %address.local999 = and i64 %1835, 48, !dbg !19
  %address.local1000 = add nsw i64 %address.local997, %address.local999, !dbg !19
  %address.local1001 = and i64 %address.local996, 240, !dbg !19
  %address.local1002 = sub nsw i64 %address.local1000, %address.local1001, !dbg !19
  %address.local1003 = add nsw i64 %address.local1002, %28, !dbg !19
  %address.local1004 = shl nuw nsw i64 %address.local1003, 7, !dbg !19
  %address.local1005 = add nuw nsw i64 %1834, %address.local1004, !dbg !19
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %1836 = lshr i64 %address.local996, 2, !dbg !19
  %address.local1009 = and i64 %1836, 12, !dbg !19
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1006, i64 %address.local1009, !dbg !19
  store half %1833, ptr addrspace(1) %address.local1010, align 2, !dbg !19
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1012 = or disjoint i64 %address.local1009, 1, !dbg !19
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1011, i64 %address.local1012, !dbg !19
  store half %1832, ptr addrspace(1) %address.local1013, align 2, !dbg !19
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1015 = or disjoint i64 %address.local1009, 2, !dbg !19
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1014, i64 %address.local1015, !dbg !19
  store half %1831, ptr addrspace(1) %address.local1016, align 2, !dbg !19
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1018 = or disjoint i64 %address.local1009, 3, !dbg !19
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1017, i64 %address.local1018, !dbg !19
  store half %1830, ptr addrspace(1) %address.local1019, align 2, !dbg !19
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1021 = or disjoint i64 %address.local1009, 16, !dbg !19
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1020, i64 %address.local1021, !dbg !19
  store half %1817, ptr addrspace(1) %address.local1022, align 2, !dbg !19
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1024 = or disjoint i64 %address.local1009, 17, !dbg !19
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1023, i64 %address.local1024, !dbg !19
  store half %1816, ptr addrspace(1) %address.local1025, align 2, !dbg !19
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1027 = or disjoint i64 %address.local1009, 18, !dbg !19
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1026, i64 %address.local1027, !dbg !19
  store half %1815, ptr addrspace(1) %address.local1028, align 2, !dbg !19
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1005, !dbg !19
  %address.local1030 = or disjoint i64 %address.local1009, 19, !dbg !19
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1029, i64 %address.local1030, !dbg !19
  store half %1814, ptr addrspace(1) %address.local1031, align 2, !dbg !19
  %address.local1032 = shl nsw i64 %address.local1003, 7, !dbg !19
  %address.local1033 = add i64 %address.local1032, 8192, !dbg !19
  %address.local1034 = add nuw nsw i64 %1834, %address.local1033, !dbg !19
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1035, i64 %address.local1009, !dbg !19
  store half %1801, ptr addrspace(1) %address.local1036, align 2, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1012, !dbg !19
  store half %1800, ptr addrspace(1) %address.local1038, align 2, !dbg !19
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1039, i64 %address.local1015, !dbg !19
  store half %1799, ptr addrspace(1) %address.local1040, align 2, !dbg !19
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1041, i64 %address.local1018, !dbg !19
  store half %1798, ptr addrspace(1) %address.local1042, align 2, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1021, !dbg !19
  store half %1785, ptr addrspace(1) %address.local1044, align 2, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1045, i64 %address.local1024, !dbg !19
  store half %1784, ptr addrspace(1) %address.local1046, align 2, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1047, i64 %address.local1027, !dbg !19
  store half %1783, ptr addrspace(1) %address.local1048, align 2, !dbg !19
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1034, !dbg !19
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1049, i64 %address.local1030, !dbg !19
  store half %1782, ptr addrspace(1) %address.local1050, align 2, !dbg !19
  %1837 = insertelement <4 x float> poison, float %1666, i64 0, !dbg !17
  %1838 = insertelement <4 x float> %1837, float %1672, i64 1, !dbg !17
  %1839 = insertelement <4 x float> %1838, float %1667, i64 2, !dbg !17
  %1840 = insertelement <4 x float> %1839, float %1673, i64 3, !dbg !17
  %1841 = fmul <4 x float> %1840, %1769, !dbg !17
  %1842 = fptrunc <4 x float> %1841 to <4 x half>, !dbg !18
  %1843 = extractelement <4 x half> %1842, i64 3, !dbg !18
  %1844 = extractelement <4 x half> %1842, i64 2, !dbg !18
  %1845 = extractelement <4 x half> %1842, i64 1, !dbg !18
  %1846 = extractelement <4 x half> %1842, i64 0, !dbg !18
  %1847 = insertelement <4 x float> poison, float %1638, i64 0, !dbg !17
  %1848 = insertelement <4 x float> %1847, float %1644, i64 1, !dbg !17
  %1849 = insertelement <4 x float> %1848, float %1639, i64 2, !dbg !17
  %1850 = insertelement <4 x float> %1849, float %1645, i64 3, !dbg !17
  %1851 = fmul <4 x float> %1850, %1769, !dbg !17
  %1852 = fptrunc <4 x float> %1851 to <4 x half>, !dbg !18
  %1853 = extractelement <4 x half> %1852, i64 3, !dbg !18
  %1854 = extractelement <4 x half> %1852, i64 2, !dbg !18
  %1855 = extractelement <4 x half> %1852, i64 1, !dbg !18
  %1856 = extractelement <4 x half> %1852, i64 0, !dbg !18
  %1857 = insertelement <4 x float> poison, float %1609, i64 0, !dbg !17
  %1858 = insertelement <4 x float> %1857, float %1615, i64 1, !dbg !17
  %1859 = insertelement <4 x float> %1858, float %1610, i64 2, !dbg !17
  %1860 = insertelement <4 x float> %1859, float %1616, i64 3, !dbg !17
  %1861 = fmul <4 x float> %1860, %1766, !dbg !17
  %1862 = fptrunc <4 x float> %1861 to <4 x half>, !dbg !18
  %1863 = extractelement <4 x half> %1862, i64 3, !dbg !18
  %1864 = extractelement <4 x half> %1862, i64 2, !dbg !18
  %1865 = extractelement <4 x half> %1862, i64 1, !dbg !18
  %1866 = extractelement <4 x half> %1862, i64 0, !dbg !18
  %1867 = insertelement <4 x float> poison, float %1581, i64 0, !dbg !17
  %1868 = insertelement <4 x float> %1867, float %1587, i64 1, !dbg !17
  %1869 = insertelement <4 x float> %1868, float %1582, i64 2, !dbg !17
  %1870 = insertelement <4 x float> %1869, float %1588, i64 3, !dbg !17
  %1871 = fmul <4 x float> %1870, %1766, !dbg !17
  %1872 = fptrunc <4 x float> %1871 to <4 x half>, !dbg !18
  %1873 = extractelement <4 x half> %1872, i64 3, !dbg !18
  %1874 = extractelement <4 x half> %1872, i64 2, !dbg !18
  %1875 = extractelement <4 x half> %1872, i64 1, !dbg !18
  %1876 = extractelement <4 x half> %1872, i64 0, !dbg !18
  %1877 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1051 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1052 = zext nneg i32 %address.thread1051 to i64, !dbg !13
  %address.local1053 = add nsw i64 %27, %address.local1052, !dbg !19
  %1878 = lshr i64 %address.local1052, 2, !dbg !19
  %address.local1055 = and i64 %1878, 48, !dbg !19
  %address.local1056 = add nsw i64 %address.local1053, %address.local1055, !dbg !19
  %address.local1057 = and i64 %address.local1052, 240, !dbg !19
  %address.local1058 = sub nsw i64 %address.local1056, %address.local1057, !dbg !19
  %address.local1059 = add nsw i64 %address.local1058, %28, !dbg !19
  %address.local1060 = shl nuw nsw i64 %address.local1059, 7, !dbg !19
  %address.local1061 = add nuw nsw i64 %1877, %address.local1060, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %1879 = lshr i64 %address.local1052, 2, !dbg !19
  %address.local1065 = and i64 %1879, 12, !dbg !19
  %address.local1066 = or disjoint i64 %address.local1065, 32, !dbg !19
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1062, i64 %address.local1066, !dbg !19
  store half %1876, ptr addrspace(1) %address.local1067, align 2, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1069 = or disjoint i64 %address.local1065, 33, !dbg !19
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1068, i64 %address.local1069, !dbg !19
  store half %1875, ptr addrspace(1) %address.local1070, align 2, !dbg !19
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1072 = or disjoint i64 %address.local1065, 34, !dbg !19
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1071, i64 %address.local1072, !dbg !19
  store half %1874, ptr addrspace(1) %address.local1073, align 2, !dbg !19
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1075 = or disjoint i64 %address.local1065, 35, !dbg !19
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1074, i64 %address.local1075, !dbg !19
  store half %1873, ptr addrspace(1) %address.local1076, align 2, !dbg !19
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1078 = or disjoint i64 %address.local1065, 48, !dbg !19
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1077, i64 %address.local1078, !dbg !19
  store half %1866, ptr addrspace(1) %address.local1079, align 2, !dbg !19
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1081 = or disjoint i64 %address.local1065, 49, !dbg !19
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1080, i64 %address.local1081, !dbg !19
  store half %1865, ptr addrspace(1) %address.local1082, align 2, !dbg !19
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1084 = or disjoint i64 %address.local1065, 50, !dbg !19
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1083, i64 %address.local1084, !dbg !19
  store half %1864, ptr addrspace(1) %address.local1085, align 2, !dbg !19
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1061, !dbg !19
  %address.local1087 = or i64 %1879, 51, !dbg !19
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1086, i64 %address.local1087, !dbg !19
  store half %1863, ptr addrspace(1) %address.local1088, align 2, !dbg !19
  %address.local1089 = shl nsw i64 %address.local1059, 7, !dbg !19
  %address.local1090 = add i64 %address.local1089, 8192, !dbg !19
  %address.local1091 = add nuw nsw i64 %1877, %address.local1090, !dbg !19
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1092, i64 %address.local1066, !dbg !19
  store half %1856, ptr addrspace(1) %address.local1093, align 2, !dbg !19
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1094, i64 %address.local1069, !dbg !19
  store half %1855, ptr addrspace(1) %address.local1095, align 2, !dbg !19
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1072, !dbg !19
  store half %1854, ptr addrspace(1) %address.local1097, align 2, !dbg !19
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1098, i64 %address.local1075, !dbg !19
  store half %1853, ptr addrspace(1) %address.local1099, align 2, !dbg !19
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1100, i64 %address.local1078, !dbg !19
  store half %1846, ptr addrspace(1) %address.local1101, align 2, !dbg !19
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1102, i64 %address.local1081, !dbg !19
  store half %1845, ptr addrspace(1) %address.local1103, align 2, !dbg !19
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1104, i64 %address.local1084, !dbg !19
  store half %1844, ptr addrspace(1) %address.local1105, align 2, !dbg !19
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1091, !dbg !19
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1106, i64 %address.local1087, !dbg !19
  store half %1843, ptr addrspace(1) %address.local1107, align 2, !dbg !19
  %1880 = insertelement <4 x float> poison, float %1552, i64 0, !dbg !17
  %1881 = insertelement <4 x float> %1880, float %1558, i64 1, !dbg !17
  %1882 = insertelement <4 x float> %1881, float %1553, i64 2, !dbg !17
  %1883 = insertelement <4 x float> %1882, float %1559, i64 3, !dbg !17
  %1884 = fmul <4 x float> %1883, %1769, !dbg !17
  %1885 = fptrunc <4 x float> %1884 to <4 x half>, !dbg !18
  %1886 = extractelement <4 x half> %1885, i64 3, !dbg !18
  %1887 = extractelement <4 x half> %1885, i64 2, !dbg !18
  %1888 = extractelement <4 x half> %1885, i64 1, !dbg !18
  %1889 = extractelement <4 x half> %1885, i64 0, !dbg !18
  %1890 = insertelement <4 x float> poison, float %1524, i64 0, !dbg !17
  %1891 = insertelement <4 x float> %1890, float %1530, i64 1, !dbg !17
  %1892 = insertelement <4 x float> %1891, float %1525, i64 2, !dbg !17
  %1893 = insertelement <4 x float> %1892, float %1531, i64 3, !dbg !17
  %1894 = fmul <4 x float> %1893, %1769, !dbg !17
  %1895 = fptrunc <4 x float> %1894 to <4 x half>, !dbg !18
  %1896 = extractelement <4 x half> %1895, i64 3, !dbg !18
  %1897 = extractelement <4 x half> %1895, i64 2, !dbg !18
  %1898 = extractelement <4 x half> %1895, i64 1, !dbg !18
  %1899 = extractelement <4 x half> %1895, i64 0, !dbg !18
  %1900 = insertelement <4 x float> poison, float %1495, i64 0, !dbg !17
  %1901 = insertelement <4 x float> %1900, float %1501, i64 1, !dbg !17
  %1902 = insertelement <4 x float> %1901, float %1496, i64 2, !dbg !17
  %1903 = insertelement <4 x float> %1902, float %1502, i64 3, !dbg !17
  %1904 = fmul <4 x float> %1903, %1766, !dbg !17
  %1905 = fptrunc <4 x float> %1904 to <4 x half>, !dbg !18
  %1906 = extractelement <4 x half> %1905, i64 3, !dbg !18
  %1907 = extractelement <4 x half> %1905, i64 2, !dbg !18
  %1908 = extractelement <4 x half> %1905, i64 1, !dbg !18
  %1909 = extractelement <4 x half> %1905, i64 0, !dbg !18
  %1910 = insertelement <4 x float> poison, float %1467, i64 0, !dbg !17
  %1911 = insertelement <4 x float> %1910, float %1473, i64 1, !dbg !17
  %1912 = insertelement <4 x float> %1911, float %1468, i64 2, !dbg !17
  %1913 = insertelement <4 x float> %1912, float %1474, i64 3, !dbg !17
  %1914 = fmul <4 x float> %1913, %1766, !dbg !17
  %1915 = fptrunc <4 x float> %1914 to <4 x half>, !dbg !18
  %1916 = extractelement <4 x half> %1915, i64 3, !dbg !18
  %1917 = extractelement <4 x half> %1915, i64 2, !dbg !18
  %1918 = extractelement <4 x half> %1915, i64 1, !dbg !18
  %1919 = extractelement <4 x half> %1915, i64 0, !dbg !18
  %1920 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1108 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1109 = zext nneg i32 %address.thread1108 to i64, !dbg !13
  %address.local1110 = add nsw i64 %27, %address.local1109, !dbg !19
  %1921 = lshr i64 %address.local1109, 2, !dbg !19
  %address.local1112 = and i64 %1921, 48, !dbg !19
  %address.local1113 = add nsw i64 %address.local1110, %address.local1112, !dbg !19
  %address.local1114 = and i64 %address.local1109, 240, !dbg !19
  %address.local1115 = sub nsw i64 %address.local1113, %address.local1114, !dbg !19
  %address.local1116 = add nsw i64 %address.local1115, %28, !dbg !19
  %address.local1117 = shl nuw nsw i64 %address.local1116, 7, !dbg !19
  %address.local1118 = add nuw nsw i64 %1920, %address.local1117, !dbg !19
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %1922 = lshr i64 %address.local1109, 2, !dbg !19
  %address.local1122 = and i64 %1922, 12, !dbg !19
  %address.local1123 = or disjoint i64 %address.local1122, 64, !dbg !19
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1123, !dbg !19
  store half %1919, ptr addrspace(1) %address.local1124, align 2, !dbg !19
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1126 = or disjoint i64 %address.local1122, 65, !dbg !19
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1126, !dbg !19
  store half %1918, ptr addrspace(1) %address.local1127, align 2, !dbg !19
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1129 = or disjoint i64 %address.local1122, 66, !dbg !19
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1128, i64 %address.local1129, !dbg !19
  store half %1917, ptr addrspace(1) %address.local1130, align 2, !dbg !19
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1132 = or disjoint i64 %address.local1122, 67, !dbg !19
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1131, i64 %address.local1132, !dbg !19
  store half %1916, ptr addrspace(1) %address.local1133, align 2, !dbg !19
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1135 = or disjoint i64 %address.local1122, 80, !dbg !19
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1134, i64 %address.local1135, !dbg !19
  store half %1909, ptr addrspace(1) %address.local1136, align 2, !dbg !19
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1138 = or disjoint i64 %address.local1122, 81, !dbg !19
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1137, i64 %address.local1138, !dbg !19
  store half %1908, ptr addrspace(1) %address.local1139, align 2, !dbg !19
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1141 = or disjoint i64 %address.local1122, 82, !dbg !19
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1140, i64 %address.local1141, !dbg !19
  store half %1907, ptr addrspace(1) %address.local1142, align 2, !dbg !19
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1118, !dbg !19
  %address.local1144 = or disjoint i64 %address.local1122, 83, !dbg !19
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1144, !dbg !19
  store half %1906, ptr addrspace(1) %address.local1145, align 2, !dbg !19
  %address.local1146 = shl nsw i64 %address.local1116, 7, !dbg !19
  %address.local1147 = add i64 %address.local1146, 8192, !dbg !19
  %address.local1148 = add nuw nsw i64 %1920, %address.local1147, !dbg !19
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1123, !dbg !19
  store half %1899, ptr addrspace(1) %address.local1150, align 2, !dbg !19
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1151, i64 %address.local1126, !dbg !19
  store half %1898, ptr addrspace(1) %address.local1152, align 2, !dbg !19
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1153, i64 %address.local1129, !dbg !19
  store half %1897, ptr addrspace(1) %address.local1154, align 2, !dbg !19
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1132, !dbg !19
  store half %1896, ptr addrspace(1) %address.local1156, align 2, !dbg !19
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1157, i64 %address.local1135, !dbg !19
  store half %1889, ptr addrspace(1) %address.local1158, align 2, !dbg !19
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1159, i64 %address.local1138, !dbg !19
  store half %1888, ptr addrspace(1) %address.local1160, align 2, !dbg !19
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1141, !dbg !19
  store half %1887, ptr addrspace(1) %address.local1162, align 2, !dbg !19
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1148, !dbg !19
  %address.local1164 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1163, i64 %address.local1144, !dbg !19
  store half %1886, ptr addrspace(1) %address.local1164, align 2, !dbg !19
  %1923 = insertelement <4 x float> poison, float %1438, i64 0, !dbg !17
  %1924 = insertelement <4 x float> %1923, float %1444, i64 1, !dbg !17
  %1925 = insertelement <4 x float> %1924, float %1439, i64 2, !dbg !17
  %1926 = insertelement <4 x float> %1925, float %1445, i64 3, !dbg !17
  %1927 = fmul <4 x float> %1926, %1769, !dbg !17
  %1928 = fptrunc <4 x float> %1927 to <4 x half>, !dbg !18
  %1929 = extractelement <4 x half> %1928, i64 3, !dbg !18
  %1930 = extractelement <4 x half> %1928, i64 2, !dbg !18
  %1931 = extractelement <4 x half> %1928, i64 1, !dbg !18
  %1932 = extractelement <4 x half> %1928, i64 0, !dbg !18
  %1933 = insertelement <4 x float> poison, float %1410, i64 0, !dbg !17
  %1934 = insertelement <4 x float> %1933, float %1416, i64 1, !dbg !17
  %1935 = insertelement <4 x float> %1934, float %1411, i64 2, !dbg !17
  %1936 = insertelement <4 x float> %1935, float %1417, i64 3, !dbg !17
  %1937 = fmul <4 x float> %1936, %1769, !dbg !17
  %1938 = fptrunc <4 x float> %1937 to <4 x half>, !dbg !18
  %1939 = extractelement <4 x half> %1938, i64 3, !dbg !18
  %1940 = extractelement <4 x half> %1938, i64 2, !dbg !18
  %1941 = extractelement <4 x half> %1938, i64 1, !dbg !18
  %1942 = extractelement <4 x half> %1938, i64 0, !dbg !18
  %1943 = insertelement <4 x float> poison, float %1381, i64 0, !dbg !17
  %1944 = insertelement <4 x float> %1943, float %1387, i64 1, !dbg !17
  %1945 = insertelement <4 x float> %1944, float %1382, i64 2, !dbg !17
  %1946 = insertelement <4 x float> %1945, float %1388, i64 3, !dbg !17
  %1947 = fmul <4 x float> %1946, %1766, !dbg !17
  %1948 = fptrunc <4 x float> %1947 to <4 x half>, !dbg !18
  %1949 = extractelement <4 x half> %1948, i64 3, !dbg !18
  %1950 = extractelement <4 x half> %1948, i64 2, !dbg !18
  %1951 = extractelement <4 x half> %1948, i64 1, !dbg !18
  %1952 = extractelement <4 x half> %1948, i64 0, !dbg !18
  %1953 = insertelement <4 x float> poison, float %1353, i64 0, !dbg !17
  %1954 = insertelement <4 x float> %1953, float %1359, i64 1, !dbg !17
  %1955 = insertelement <4 x float> %1954, float %1354, i64 2, !dbg !17
  %1956 = insertelement <4 x float> %1955, float %1360, i64 3, !dbg !17
  %1957 = fmul <4 x float> %1956, %1766, !dbg !17
  %1958 = fptrunc <4 x float> %1957 to <4 x half>, !dbg !18
  %1959 = extractelement <4 x half> %1958, i64 3, !dbg !18
  %1960 = extractelement <4 x half> %1958, i64 2, !dbg !18
  %1961 = extractelement <4 x half> %1958, i64 1, !dbg !18
  %1962 = extractelement <4 x half> %1958, i64 0, !dbg !18
  %1963 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1165 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1166 = zext nneg i32 %address.thread1165 to i64, !dbg !13
  %address.local1167 = add nsw i64 %27, %address.local1166, !dbg !19
  %1964 = lshr i64 %address.local1166, 2, !dbg !19
  %address.local1169 = and i64 %1964, 48, !dbg !19
  %address.local1170 = add nsw i64 %address.local1167, %address.local1169, !dbg !19
  %address.local1171 = and i64 %address.local1166, 240, !dbg !19
  %address.local1172 = sub nsw i64 %address.local1170, %address.local1171, !dbg !19
  %address.local1173 = add nsw i64 %address.local1172, %28, !dbg !19
  %address.local1174 = shl nuw nsw i64 %address.local1173, 7, !dbg !19
  %address.local1175 = add nuw nsw i64 %1963, %address.local1174, !dbg !19
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %1965 = lshr i64 %address.local1166, 2, !dbg !19
  %address.local1179 = and i64 %1965, 12, !dbg !19
  %address.local1180 = or disjoint i64 %address.local1179, 96, !dbg !19
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1176, i64 %address.local1180, !dbg !19
  store half %1962, ptr addrspace(1) %address.local1181, align 2, !dbg !19
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1183 = or disjoint i64 %address.local1179, 97, !dbg !19
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1182, i64 %address.local1183, !dbg !19
  store half %1961, ptr addrspace(1) %address.local1184, align 2, !dbg !19
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1186 = or disjoint i64 %address.local1179, 98, !dbg !19
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1185, i64 %address.local1186, !dbg !19
  store half %1960, ptr addrspace(1) %address.local1187, align 2, !dbg !19
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1189 = or disjoint i64 %address.local1179, 99, !dbg !19
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1188, i64 %address.local1189, !dbg !19
  store half %1959, ptr addrspace(1) %address.local1190, align 2, !dbg !19
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1192 = or disjoint i64 %address.local1179, 112, !dbg !19
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1191, i64 %address.local1192, !dbg !19
  store half %1952, ptr addrspace(1) %address.local1193, align 2, !dbg !19
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1195 = or disjoint i64 %address.local1179, 113, !dbg !19
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1194, i64 %address.local1195, !dbg !19
  store half %1951, ptr addrspace(1) %address.local1196, align 2, !dbg !19
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1198 = or disjoint i64 %address.local1179, 114, !dbg !19
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1197, i64 %address.local1198, !dbg !19
  store half %1950, ptr addrspace(1) %address.local1199, align 2, !dbg !19
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1175, !dbg !19
  %address.local1201 = or i64 %1965, 115, !dbg !19
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1200, i64 %address.local1201, !dbg !19
  store half %1949, ptr addrspace(1) %address.local1202, align 2, !dbg !19
  %address.local1203 = shl nsw i64 %address.local1173, 7, !dbg !19
  %address.local1204 = add i64 %address.local1203, 8192, !dbg !19
  %address.local1205 = add nuw nsw i64 %1963, %address.local1204, !dbg !19
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1206, i64 %address.local1180, !dbg !19
  store half %1942, ptr addrspace(1) %address.local1207, align 2, !dbg !19
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1183, !dbg !19
  store half %1941, ptr addrspace(1) %address.local1209, align 2, !dbg !19
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1210, i64 %address.local1186, !dbg !19
  store half %1940, ptr addrspace(1) %address.local1211, align 2, !dbg !19
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1212, i64 %address.local1189, !dbg !19
  store half %1939, ptr addrspace(1) %address.local1213, align 2, !dbg !19
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1192, !dbg !19
  store half %1932, ptr addrspace(1) %address.local1215, align 2, !dbg !19
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1216, i64 %address.local1195, !dbg !19
  store half %1931, ptr addrspace(1) %address.local1217, align 2, !dbg !19
  %address.local1218 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1218, i64 %address.local1198, !dbg !19
  store half %1930, ptr addrspace(1) %address.local1219, align 2, !dbg !19
  %address.local1220 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1205, !dbg !19
  %address.local1221 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1220, i64 %address.local1201, !dbg !19
  store half %1929, ptr addrspace(1) %address.local1221, align 2, !dbg !19
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

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #6

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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind memory(none) }
attributes #11 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!29}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math")
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
