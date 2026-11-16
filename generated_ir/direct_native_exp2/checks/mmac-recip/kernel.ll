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
  %address.local271 = lshr i64 %address.local, 6, !dbg !11
  %address.local272 = shl nuw nsw i64 %address.local271, 7, !dbg !11
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local272, !dbg !11
  %address.local274 = ptrtoint ptr addrspace(3) %address.local273 to i64, !dbg !11
  %address.local275 = trunc i64 %address.local274 to i32, !dbg !11
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local275), !dbg !11
  %39 = zext i32 %38 to i64, !dbg !11
  %40 = inttoptr i64 %39 to ptr addrspace(3), !dbg !11
  %address.local276 = lshr i64 %address.local, 6, !dbg !11
  %address.local277 = or disjoint i64 %address.local276, 4, !dbg !11
  %address.local278 = shl nuw nsw i64 %address.local277, 7, !dbg !11
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !11
  %address.local280 = ptrtoint ptr addrspace(3) %address.local279 to i64, !dbg !11
  %address.local281 = trunc i64 %address.local280 to i32, !dbg !11
  %41 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local281), !dbg !11
  %42 = zext i32 %41 to i64, !dbg !11
  %43 = inttoptr i64 %42 to ptr addrspace(3), !dbg !11
  %address.local282 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local283 = and i64 %address.local282, 384, !dbg !11
  %address.local284 = or disjoint i64 %address.local283, 1024, !dbg !11
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local284, !dbg !11
  %address.local286 = ptrtoint ptr addrspace(3) %address.local285 to i64, !dbg !11
  %address.local287 = trunc i64 %address.local286 to i32, !dbg !11
  %44 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local287), !dbg !11
  %45 = zext i32 %44 to i64, !dbg !11
  %46 = inttoptr i64 %45 to ptr addrspace(3), !dbg !11
  %address.local288 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local289 = and i64 %address.local288, 384, !dbg !11
  %address.local290 = or disjoint i64 %address.local289, 1536, !dbg !11
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local290, !dbg !11
  %address.local292 = ptrtoint ptr addrspace(3) %address.local291 to i64, !dbg !11
  %address.local293 = trunc i64 %address.local292 to i32, !dbg !11
  %47 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local293), !dbg !11
  %48 = zext i32 %47 to i64, !dbg !11
  %49 = inttoptr i64 %48 to ptr addrspace(3), !dbg !11
  %address.local294 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local295 = and i64 %address.local294, 384, !dbg !11
  %address.local296 = or disjoint i64 %address.local295, 2048, !dbg !11
  %address.local297 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local296, !dbg !11
  %address.local298 = ptrtoint ptr addrspace(3) %address.local297 to i64, !dbg !11
  %address.local299 = trunc i64 %address.local298 to i32, !dbg !11
  %50 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local299), !dbg !11
  %51 = zext i32 %50 to i64, !dbg !11
  %52 = inttoptr i64 %51 to ptr addrspace(3), !dbg !11
  %address.local300 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local301 = and i64 %address.local300, 384, !dbg !11
  %address.local302 = or disjoint i64 %address.local301, 2560, !dbg !11
  %address.local303 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local302, !dbg !11
  %address.local304 = ptrtoint ptr addrspace(3) %address.local303 to i64, !dbg !11
  %address.local305 = trunc i64 %address.local304 to i32, !dbg !11
  %53 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local305), !dbg !11
  %54 = zext i32 %53 to i64, !dbg !11
  %55 = inttoptr i64 %54 to ptr addrspace(3), !dbg !11
  %address.local306 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local307 = and i64 %address.local306, 384, !dbg !11
  %address.local308 = or disjoint i64 %address.local307, 3072, !dbg !11
  %address.local309 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local308, !dbg !11
  %address.local310 = ptrtoint ptr addrspace(3) %address.local309 to i64, !dbg !11
  %address.local311 = trunc i64 %address.local310 to i32, !dbg !11
  %56 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local311), !dbg !11
  %57 = zext i32 %56 to i64, !dbg !11
  %58 = inttoptr i64 %57 to ptr addrspace(3), !dbg !11
  %address.local312 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local313 = and i64 %address.local312, 384, !dbg !11
  %address.local314 = or disjoint i64 %address.local313, 3584, !dbg !11
  %address.local315 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local314, !dbg !11
  %address.local316 = ptrtoint ptr addrspace(3) %address.local315 to i64, !dbg !11
  %address.local317 = trunc i64 %address.local316 to i32, !dbg !11
  %59 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local317), !dbg !11
  %60 = zext i32 %59 to i64, !dbg !11
  %61 = inttoptr i64 %60 to ptr addrspace(3), !dbg !11
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local272, !dbg !12
  %address.local319 = ptrtoint ptr addrspace(3) %address.local318 to i64, !dbg !12
  %address.local320 = trunc i64 %address.local319 to i32, !dbg !12
  %62 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local320), !dbg !12
  %63 = zext i32 %62 to i64, !dbg !12
  %64 = inttoptr i64 %63 to ptr addrspace(3), !dbg !12
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local278, !dbg !12
  %address.local322 = ptrtoint ptr addrspace(3) %address.local321 to i64, !dbg !12
  %address.local323 = trunc i64 %address.local322 to i32, !dbg !12
  %65 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local323), !dbg !12
  %66 = zext i32 %65 to i64, !dbg !12
  %67 = inttoptr i64 %66 to ptr addrspace(3), !dbg !12
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local284, !dbg !12
  %address.local325 = ptrtoint ptr addrspace(3) %address.local324 to i64, !dbg !12
  %address.local326 = trunc i64 %address.local325 to i32, !dbg !12
  %68 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local326), !dbg !12
  %69 = zext i32 %68 to i64, !dbg !12
  %70 = inttoptr i64 %69 to ptr addrspace(3), !dbg !12
  %address.local327 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local290, !dbg !12
  %address.local328 = ptrtoint ptr addrspace(3) %address.local327 to i64, !dbg !12
  %address.local329 = trunc i64 %address.local328 to i32, !dbg !12
  %71 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local329), !dbg !12
  %72 = zext i32 %71 to i64, !dbg !12
  %73 = inttoptr i64 %72 to ptr addrspace(3), !dbg !12
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local296, !dbg !12
  %address.local331 = ptrtoint ptr addrspace(3) %address.local330 to i64, !dbg !12
  %address.local332 = trunc i64 %address.local331 to i32, !dbg !12
  %74 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local332), !dbg !12
  %75 = zext i32 %74 to i64, !dbg !12
  %76 = inttoptr i64 %75 to ptr addrspace(3), !dbg !12
  %address.local333 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local302, !dbg !12
  %address.local334 = ptrtoint ptr addrspace(3) %address.local333 to i64, !dbg !12
  %address.local335 = trunc i64 %address.local334 to i32, !dbg !12
  %77 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local335), !dbg !12
  %78 = zext i32 %77 to i64, !dbg !12
  %79 = inttoptr i64 %78 to ptr addrspace(3), !dbg !12
  %address.local336 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local308, !dbg !12
  %address.local337 = ptrtoint ptr addrspace(3) %address.local336 to i64, !dbg !12
  %address.local338 = trunc i64 %address.local337 to i32, !dbg !12
  %80 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local338), !dbg !12
  %81 = zext i32 %80 to i64, !dbg !12
  %82 = inttoptr i64 %81 to ptr addrspace(3), !dbg !12
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local314, !dbg !12
  %address.local340 = ptrtoint ptr addrspace(3) %address.local339 to i64, !dbg !12
  %address.local341 = trunc i64 %address.local340 to i32, !dbg !12
  %83 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local341), !dbg !12
  %84 = zext i32 %83 to i64, !dbg !12
  %85 = inttoptr i64 %84 to ptr addrspace(3), !dbg !12
  %address.local342 = shl nuw nsw i64 %address.local271, 10, !dbg !15
  %address.local343 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local342, !dbg !15
  %address.local344 = ptrtoint ptr addrspace(3) %address.local343 to i64, !dbg !15
  %address.local345 = trunc i64 %address.local344 to i32, !dbg !15
  %86 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local345), !dbg !15
  %address.local346 = shl nuw nsw i64 %address.local277, 10, !dbg !15
  %address.local347 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local346, !dbg !15
  %address.local348 = ptrtoint ptr addrspace(3) %address.local347 to i64, !dbg !15
  %address.local349 = trunc i64 %address.local348 to i32, !dbg !15
  %87 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local349), !dbg !15
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
  %.neg201 = sdiv i64 %103, -4, !dbg !16
  %104 = shl nuw nsw i64 %32, 4, !dbg !16
  %105 = shl nuw nsw i64 %90, 6, !dbg !16
  %106 = sub nsw i64 %104, %105, !dbg !16
  %107 = or disjoint i64 %106, %92, !dbg !16
  %108 = shl nuw nsw i32 %31, 1, !dbg !16
  %109 = and i32 %108, 6, !dbg !16
  %index.compact131 = zext nneg i32 %109 to i64, !dbg !16
  %110 = lshr i32 %31, 1, !dbg !16
  %111 = and i32 %110, 1, !dbg !16
  %index.compact128 = zext nneg i32 %111 to i64, !dbg !16
  %112 = shl nuw nsw i64 %90, 4, !dbg !16
  %113 = sub nsw i64 %33, %112, !dbg !16
  %114 = or disjoint i64 %113, %index.compact128, !dbg !16
  %115 = add nuw nsw i64 %114, %95, !dbg !16
  %116 = add nsw i64 %115, 1, !dbg !16
  %.neg204 = sdiv i64 %116, -2, !dbg !16
  %117 = add nuw nsw i64 %107, %index.compact131, !dbg !16
  %.neg206 = sdiv i64 %117, -4, !dbg !16
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
  %address.thread350 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !11, !range !30, !frisk.register_lifetimes !1
  %address.local351 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local352 = and i32 %address.local351, 2, !dbg !11
  %address.local353 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local354 = and i32 %address.local353, 16256, !dbg !11
  %address.local355 = or disjoint i32 %address.local352, %address.local354, !dbg !11
  %address.local356 = zext nneg i32 %address.local355 to i64, !dbg !11
  %address.local357 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local356, !dbg !11
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local357 to i64, !dbg !11
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #10, !dbg !11
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local358 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local359 = and i32 %address.local358, 2, !dbg !11
  %address.local360 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local361 = and i32 %address.local360, 16256, !dbg !11
  %address.local362 = or disjoint i32 %address.local359, %address.local361, !dbg !11
  %address.local363 = or disjoint i32 %address.local362, 4, !dbg !11
  %address.local364 = zext nneg i32 %address.local363 to i64, !dbg !11
  %address.local365 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local364, !dbg !11
  %dma.source.bits1253 = ptrtoint ptr addrspace(1) %address.local365 to i64, !dbg !11
  %dma.source.opaque1254 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1253) #10, !dbg !11
  %dma.source.ptr1255 = inttoptr i64 %dma.source.opaque1254 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1255, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local366 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local367 = and i32 %address.local366, 2, !dbg !11
  %address.local368 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local369 = and i32 %address.local368, 16256, !dbg !11
  %address.local370 = or disjoint i32 %address.local367, %address.local369, !dbg !11
  %address.local371 = or disjoint i32 %address.local370, 8, !dbg !11
  %address.local372 = zext nneg i32 %address.local371 to i64, !dbg !11
  %address.local373 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local372, !dbg !11
  %dma.source.bits1256 = ptrtoint ptr addrspace(1) %address.local373 to i64, !dbg !11
  %dma.source.opaque1257 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1256) #10, !dbg !11
  %dma.source.ptr1258 = inttoptr i64 %dma.source.opaque1257 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1258, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local374 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local375 = and i32 %address.local374, 2, !dbg !11
  %address.local376 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local377 = and i32 %address.local376, 16256, !dbg !11
  %address.local378 = or disjoint i32 %address.local375, %address.local377, !dbg !11
  %address.local379 = or disjoint i32 %address.local378, 12, !dbg !11
  %address.local380 = zext nneg i32 %address.local379 to i64, !dbg !11
  %address.local381 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local380, !dbg !11
  %dma.source.bits1259 = ptrtoint ptr addrspace(1) %address.local381 to i64, !dbg !11
  %dma.source.opaque1260 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1259) #10, !dbg !11
  %dma.source.ptr1261 = inttoptr i64 %dma.source.opaque1260 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1261, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local382 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local383 = and i32 %address.local382, 2, !dbg !11
  %address.local384 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local385 = and i32 %address.local384, 16256, !dbg !11
  %address.local386 = or disjoint i32 %address.local383, %address.local385, !dbg !11
  %address.local387 = or disjoint i32 %address.local386, 16, !dbg !11
  %address.local388 = zext nneg i32 %address.local387 to i64, !dbg !11
  %address.local389 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local388, !dbg !11
  %dma.source.bits1262 = ptrtoint ptr addrspace(1) %address.local389 to i64, !dbg !11
  %dma.source.opaque1263 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1262) #10, !dbg !11
  %dma.source.ptr1264 = inttoptr i64 %dma.source.opaque1263 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1264, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local390 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local391 = and i32 %address.local390, 2, !dbg !11
  %address.local392 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local393 = and i32 %address.local392, 16256, !dbg !11
  %address.local394 = or disjoint i32 %address.local391, %address.local393, !dbg !11
  %address.local395 = or disjoint i32 %address.local394, 20, !dbg !11
  %address.local396 = zext nneg i32 %address.local395 to i64, !dbg !11
  %address.local397 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local396, !dbg !11
  %dma.source.bits1265 = ptrtoint ptr addrspace(1) %address.local397 to i64, !dbg !11
  %dma.source.opaque1266 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1265) #10, !dbg !11
  %dma.source.ptr1267 = inttoptr i64 %dma.source.opaque1266 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1267, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local398 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local399 = and i32 %address.local398, 2, !dbg !11
  %address.local400 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local401 = and i32 %address.local400, 16256, !dbg !11
  %address.local402 = or disjoint i32 %address.local399, %address.local401, !dbg !11
  %address.local403 = or disjoint i32 %address.local402, 24, !dbg !11
  %address.local404 = zext nneg i32 %address.local403 to i64, !dbg !11
  %address.local405 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local404, !dbg !11
  %dma.source.bits1268 = ptrtoint ptr addrspace(1) %address.local405 to i64, !dbg !11
  %dma.source.opaque1269 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1268) #10, !dbg !11
  %dma.source.ptr1270 = inttoptr i64 %dma.source.opaque1269 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1270, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local406 = shl nuw nsw i32 %address.thread350, 1, !dbg !11
  %address.local407 = and i32 %address.local406, 2, !dbg !11
  %address.local408 = shl nuw nsw i32 %address.thread350, 6, !dbg !11
  %address.local409 = and i32 %address.local408, 16256, !dbg !11
  %address.local410 = or disjoint i32 %address.local407, %address.local409, !dbg !11
  %address.local411 = or disjoint i32 %address.local410, 28, !dbg !11
  %address.local412 = zext nneg i32 %address.local411 to i64, !dbg !11
  %address.local413 = getelementptr [2 x i8], ptr addrspace(1) %89, i64 %address.local412, !dbg !11
  %dma.source.bits1271 = ptrtoint ptr addrspace(1) %address.local413 to i64, !dbg !11
  %dma.source.opaque1272 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1271) #10, !dbg !11
  %dma.source.ptr1273 = inttoptr i64 %dma.source.opaque1272 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1273, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %134 = shl nsw i64 %18, 19, !dbg !12
  %135 = and i64 %134, 16252928, !dbg !12
  %136 = shl nuw nsw i64 %125, 14, !dbg !12
  %137 = or disjoint i64 %135, %136, !dbg !12
  %138 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %137, !dbg !12
  %address.local414 = zext nneg i32 %address.thread350 to i64, !dbg !13
  %address.local415 = lshr i64 %address.local414, 2, !dbg !12
  %address.local416 = lshr i64 %address.local414, 5, !dbg !12
  %address.local417 = mul nuw nsw i64 %address.local416, 7, !dbg !12
  %address.local418 = add nuw nsw i64 %address.local415, %address.local417, !dbg !12
  %address.local419 = and i64 %address.local415, 60, !dbg !12
  %address.local420 = add nuw nsw i64 %address.local418, %address.local419, !dbg !12
  %address.local421 = lshr i64 %address.local414, 7, !dbg !12
  %address.local422 = sub nsw i64 %address.local415, %address.local416, !dbg !12
  %address.local423 = sdiv i64 %address.local422, 4, !dbg !12
  %address.local424 = add nsw i64 %address.local421, %address.local423, !dbg !12
  %address.local425 = shl nsw i64 %address.local424, 2, !dbg !12
  %address.local426 = lshr i64 %address.local414, 4, !dbg !12
  %address.local427 = sub nsw i64 %address.local426, %address.local421, !dbg !12
  %address.local428 = sdiv i64 %address.local427, 2, !dbg !12
  %address.local429 = shl nsw i64 %address.local428, 3, !dbg !12
  %address.local430 = add nsw i64 %address.local425, %address.local429, !dbg !12
  %address.local431 = sub i64 %address.local420, %address.local430, !dbg !12
  %address.local432 = shl i64 %address.local431, 7, !dbg !12
  %address.local433 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local434 = and i32 %address.local433, 2, !dbg !12
  %address.local435 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local436 = and i32 %address.local435, 16, !dbg !12
  %address.local437 = or disjoint i32 %address.local434, %address.local436, !dbg !12
  %address.local438 = zext nneg i32 %address.local437 to i64, !dbg !12
  %address.local439 = or disjoint i64 %address.local432, %address.local438, !dbg !12
  %address.local440 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local439, !dbg !12
  %dma.source.bits1274 = ptrtoint ptr addrspace(1) %address.local440 to i64, !dbg !12
  %dma.source.opaque1275 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1274) #10, !dbg !12
  %dma.source.ptr1276 = inttoptr i64 %dma.source.opaque1275 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1276, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local441 = or disjoint i64 %address.local414, 256, !dbg !11
  %address.local442 = lshr i64 %address.local441, 2, !dbg !12
  %address.local443 = lshr i64 %address.local441, 5, !dbg !12
  %address.local444 = mul nuw nsw i64 %address.local443, 7, !dbg !12
  %address.local445 = add nuw nsw i64 %address.local442, %address.local444, !dbg !12
  %address.local446 = and i64 %address.local442, 124, !dbg !12
  %address.local447 = add nuw nsw i64 %address.local445, %address.local446, !dbg !12
  %address.local448 = sub nuw nsw i64 %address.local442, %address.local443, !dbg !12
  %address.local449 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local450 = and i32 %address.local449, 4, !dbg !12
  %address.local451 = or disjoint i32 %address.local450, 8, !dbg !12
  %address.local452 = zext nneg i32 %address.local451 to i64, !dbg !12
  %address.local453 = add nuw nsw i64 %address.local448, %address.local452, !dbg !12
  %address.local454 = and i64 %address.local453, 1073741820, !dbg !12
  %address.local455 = lshr i64 %address.local441, 4, !dbg !12
  %address.local456 = lshr i32 %address.thread350, 7, !dbg !12
  %address.local457 = or disjoint i32 %address.local456, 2, !dbg !12
  %address.local458 = zext nneg i32 %address.local457 to i64, !dbg !12
  %address.local459 = sub nuw nsw i64 %address.local455, %address.local458, !dbg !12
  %address.local460 = shl nuw nsw i64 %address.local459, 2, !dbg !12
  %address.local461 = and i64 %address.local460, 536870904, !dbg !12
  %address.local462 = add nuw nsw i64 %address.local454, %address.local461, !dbg !12
  %address.local463 = sub nsw i64 %address.local447, %address.local462, !dbg !12
  %address.local464 = shl i64 %address.local463, 7, !dbg !12
  %address.local465 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local466 = and i32 %address.local465, 2, !dbg !12
  %address.local467 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local468 = and i32 %address.local467, 16, !dbg !12
  %address.local469 = or disjoint i32 %address.local466, %address.local468, !dbg !12
  %address.local470 = zext nneg i32 %address.local469 to i64, !dbg !12
  %address.local471 = or disjoint i64 %address.local464, %address.local470, !dbg !12
  %address.local472 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local471, !dbg !12
  %dma.source.bits1277 = ptrtoint ptr addrspace(1) %address.local472 to i64, !dbg !12
  %dma.source.opaque1278 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1277) #10, !dbg !12
  %dma.source.ptr1279 = inttoptr i64 %dma.source.opaque1278 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1279, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local473 = lshr i64 %address.local414, 2, !dbg !12
  %address.local474 = lshr i64 %address.local414, 5, !dbg !12
  %address.local475 = mul nuw nsw i64 %address.local474, 7, !dbg !12
  %address.local476 = add nuw nsw i64 %address.local473, %address.local475, !dbg !12
  %address.local477 = and i64 %address.local473, 60, !dbg !12
  %address.local478 = add nuw nsw i64 %address.local476, %address.local477, !dbg !12
  %address.local479 = sub nsw i64 %address.local473, %address.local474, !dbg !12
  %address.local480 = sdiv i64 %address.local479, 4, !dbg !12
  %address.local481 = lshr i32 %address.thread350, 7, !dbg !12
  %address.local482 = zext nneg i32 %address.local481 to i64, !dbg !12
  %address.local483 = add nsw i64 %address.local480, %address.local482, !dbg !12
  %address.local484 = shl nsw i64 %address.local483, 2, !dbg !12
  %address.local485 = lshr i64 %address.local414, 4, !dbg !12
  %address.local486 = lshr i32 %address.thread350, 7, !dbg !12
  %address.local487 = zext nneg i32 %address.local486 to i64, !dbg !12
  %address.local488 = sub nsw i64 %address.local485, %address.local487, !dbg !12
  %address.local489 = sdiv i64 %address.local488, 2, !dbg !12
  %address.local490 = shl nsw i64 %address.local489, 3, !dbg !12
  %address.local491 = add nsw i64 %address.local484, %address.local490, !dbg !12
  %address.local492 = sub i64 %address.local478, %address.local491, !dbg !12
  %address.local493 = shl i64 %address.local492, 7, !dbg !12
  %address.local494 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local495 = and i32 %address.local494, 2, !dbg !12
  %address.local496 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local497 = and i32 %address.local496, 16, !dbg !12
  %address.local498 = or disjoint i32 %address.local495, %address.local497, !dbg !12
  %address.local499 = or disjoint i32 %address.local498, 4, !dbg !12
  %address.local500 = zext nneg i32 %address.local499 to i64, !dbg !12
  %address.local501 = or disjoint i64 %address.local493, %address.local500, !dbg !12
  %address.local502 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local501, !dbg !12
  %dma.source.bits1280 = ptrtoint ptr addrspace(1) %address.local502 to i64, !dbg !12
  %dma.source.opaque1281 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1280) #10, !dbg !12
  %dma.source.ptr1282 = inttoptr i64 %dma.source.opaque1281 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1282, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local503 = lshr i64 %address.local414, 2, !dbg !12
  %address.local504 = or disjoint i64 %address.local503, 192, !dbg !12
  %address.local505 = and i64 %address.local504, 127, !dbg !12
  %address.local506 = lshr i64 %address.local505, 3, !dbg !12
  %address.local507 = mul nuw nsw i64 %address.local506, 7, !dbg !12
  %address.local508 = add nuw nsw i64 %address.local505, %address.local507, !dbg !12
  %address.local509 = and i64 %address.local504, 124, !dbg !12
  %address.local510 = add nuw nsw i64 %address.local508, %address.local509, !dbg !12
  %address.local511 = sub nuw nsw i64 %address.local505, %address.local506, !dbg !12
  %address.local512 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local513 = and i32 %address.local512, 4, !dbg !12
  %address.local514 = or disjoint i32 %address.local513, 8, !dbg !12
  %address.local515 = zext nneg i32 %address.local514 to i64, !dbg !12
  %address.local516 = add nuw nsw i64 %address.local511, %address.local515, !dbg !12
  %address.local517 = and i64 %address.local516, 252, !dbg !12
  %address.local518 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local519 = and i32 %address.local518, 4, !dbg !12
  %address.local520 = or disjoint i32 %address.local519, 8, !dbg !12
  %address.local521 = zext nneg i32 %address.local520 to i64, !dbg !12
  %address.local522 = sub nuw nsw i64 %address.local504, %address.local521, !dbg !12
  %address.local523 = and i64 %address.local522, 120, !dbg !12
  %address.local524 = add nuw nsw i64 %address.local517, %address.local523, !dbg !12
  %address.local525 = sub nsw i64 %address.local510, %address.local524, !dbg !12
  %address.local526 = shl nsw i64 %address.local525, 7, !dbg !12
  %address.local527 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local528 = and i32 %address.local527, 2, !dbg !12
  %address.local529 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local530 = and i32 %address.local529, 16, !dbg !12
  %address.local531 = or disjoint i32 %address.local528, %address.local530, !dbg !12
  %address.local532 = or disjoint i32 %address.local531, 4, !dbg !12
  %address.local533 = zext nneg i32 %address.local532 to i64, !dbg !12
  %address.local534 = or disjoint i64 %address.local526, %address.local533, !dbg !12
  %address.local535 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local534, !dbg !12
  %dma.source.bits1283 = ptrtoint ptr addrspace(1) %address.local535 to i64, !dbg !12
  %dma.source.opaque1284 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1283) #10, !dbg !12
  %dma.source.ptr1285 = inttoptr i64 %dma.source.opaque1284 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1285, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local536 = lshr i64 %address.local414, 2, !dbg !12
  %address.local537 = lshr i64 %address.local414, 5, !dbg !12
  %address.local538 = mul nuw nsw i64 %address.local537, 7, !dbg !12
  %address.local539 = add nuw nsw i64 %address.local536, %address.local538, !dbg !12
  %address.local540 = and i64 %address.local536, 60, !dbg !12
  %address.local541 = add nuw nsw i64 %address.local539, %address.local540, !dbg !12
  %address.local542 = add nuw nsw i64 %address.local541, 144115188075855868, !dbg !12
  %address.local543 = lshr i64 %address.local414, 7, !dbg !12
  %address.local544 = sub nsw i64 %address.local536, %address.local537, !dbg !12
  %address.local545 = sdiv i64 %address.local544, 4, !dbg !12
  %address.local546 = add nsw i64 %address.local543, %address.local545, !dbg !12
  %address.local547 = shl nsw i64 %address.local546, 2, !dbg !12
  %address.local548 = lshr i64 %address.local414, 4, !dbg !12
  %address.local549 = xor i64 %address.local543, -1, !dbg !12
  %address.local550 = add nsw i64 %address.local548, %address.local549, !dbg !12
  %address.local551 = icmp slt i64 %address.local550, 0, !dbg !12
  %address.local552 = sub nsw i64 %address.local543, %address.local548, !dbg !12
  %address.local553 = select i1 %address.local551, i64 %address.local552, i64 %address.local550, !dbg !12
  %address.local554 = sdiv i64 %address.local553, 2, !dbg !12
  %address.local555 = ashr i64 %address.local550, 63, !dbg !12
  %address.local556 = xor i64 %address.local554, %address.local555, !dbg !12
  %address.local557 = shl nsw i64 %address.local556, 3, !dbg !12
  %address.local558 = add i64 %address.local547, %address.local557, !dbg !12
  %address.local559 = sub i64 %address.local542, %address.local558, !dbg !12
  %address.local560 = shl i64 %address.local559, 7, !dbg !12
  %address.local561 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local562 = and i32 %address.local561, 2, !dbg !12
  %address.local563 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local564 = and i32 %address.local563, 16, !dbg !12
  %address.local565 = or disjoint i32 %address.local562, %address.local564, !dbg !12
  %address.local566 = or disjoint i32 %address.local565, 8, !dbg !12
  %address.local567 = zext nneg i32 %address.local566 to i64, !dbg !12
  %address.local568 = or disjoint i64 %address.local560, %address.local567, !dbg !12
  %address.local569 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local568, !dbg !12
  %dma.source.bits1286 = ptrtoint ptr addrspace(1) %address.local569 to i64, !dbg !12
  %dma.source.opaque1287 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1286) #10, !dbg !12
  %dma.source.ptr1288 = inttoptr i64 %dma.source.opaque1287 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1288, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local570 = lshr i64 %address.local414, 2, !dbg !12
  %address.local571 = or disjoint i64 %address.local570, 320, !dbg !12
  %address.local572 = and i64 %address.local571, 127, !dbg !12
  %address.local573 = lshr i64 %address.local572, 3, !dbg !12
  %address.local574 = mul nuw nsw i64 %address.local573, 7, !dbg !12
  %address.local575 = add nuw nsw i64 %address.local572, %address.local574, !dbg !12
  %address.local576 = and i64 %address.local571, 124, !dbg !12
  %address.local577 = add nuw nsw i64 %address.local575, %address.local576, !dbg !12
  %address.local578 = sub nuw nsw i64 %address.local572, %address.local573, !dbg !12
  %address.local579 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local580 = and i32 %address.local579, 4, !dbg !12
  %address.local581 = add nuw nsw i32 %address.local580, 12, !dbg !12
  %address.local582 = zext nneg i32 %address.local581 to i64, !dbg !12
  %address.local583 = add nuw nsw i64 %address.local578, %address.local582, !dbg !12
  %address.local584 = and i64 %address.local583, 252, !dbg !12
  %address.local585 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local586 = and i32 %address.local585, 4, !dbg !12
  %address.local587 = add nuw nsw i32 %address.local586, 12, !dbg !12
  %address.local588 = zext nneg i32 %address.local587 to i64, !dbg !12
  %address.local589 = sub nuw nsw i64 %address.local571, %address.local588, !dbg !12
  %address.local590 = and i64 %address.local589, 120, !dbg !12
  %address.local591 = add nuw nsw i64 %address.local584, %address.local590, !dbg !12
  %address.local592 = sub nsw i64 %address.local577, %address.local591, !dbg !12
  %address.local593 = shl nsw i64 %address.local592, 7, !dbg !12
  %address.local594 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local595 = and i32 %address.local594, 2, !dbg !12
  %address.local596 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local597 = and i32 %address.local596, 16, !dbg !12
  %address.local598 = or disjoint i32 %address.local595, %address.local597, !dbg !12
  %address.local599 = or disjoint i32 %address.local598, 8, !dbg !12
  %address.local600 = zext nneg i32 %address.local599 to i64, !dbg !12
  %address.local601 = or disjoint i64 %address.local593, %address.local600, !dbg !12
  %address.local602 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local601, !dbg !12
  %dma.source.bits1289 = ptrtoint ptr addrspace(1) %address.local602 to i64, !dbg !12
  %dma.source.opaque1290 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1289) #10, !dbg !12
  %dma.source.ptr1291 = inttoptr i64 %dma.source.opaque1290 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1291, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local603 = lshr i64 %address.local414, 2, !dbg !12
  %address.local604 = lshr i64 %address.local414, 5, !dbg !12
  %address.local605 = mul nuw nsw i64 %address.local604, 7, !dbg !12
  %address.local606 = add nuw nsw i64 %address.local603, %address.local605, !dbg !12
  %address.local607 = and i64 %address.local603, 60, !dbg !12
  %address.local608 = add nuw nsw i64 %address.local606, %address.local607, !dbg !12
  %address.local609 = sub nsw i64 %address.local603, %address.local604, !dbg !12
  %address.local610 = sdiv i64 %address.local609, 4, !dbg !12
  %address.local611 = lshr i32 %address.thread350, 7, !dbg !12
  %address.local612 = add nuw nsw i32 %address.local611, 1, !dbg !12
  %address.local613 = zext nneg i32 %address.local612 to i64, !dbg !12
  %address.local614 = add nsw i64 %address.local610, %address.local613, !dbg !12
  %address.local615 = shl nsw i64 %address.local614, 2, !dbg !12
  %address.local616 = lshr i64 %address.local414, 4, !dbg !12
  %address.local617 = lshr i32 %address.thread350, 7, !dbg !12
  %address.local618 = add nuw nsw i32 %address.local617, 1, !dbg !12
  %address.local619 = zext nneg i32 %address.local618 to i64, !dbg !12
  %address.local620 = sub nsw i64 %address.local616, %address.local619, !dbg !12
  %address.local621 = ashr i64 %address.local620, 63, !dbg !12
  %address.local622 = xor i64 %address.local620, %address.local621, !dbg !12
  %address.local623 = sdiv i64 %address.local622, 2, !dbg !12
  %address.local624 = ashr i64 %address.local620, 63, !dbg !12
  %address.local625 = xor i64 %address.local623, %address.local624, !dbg !12
  %address.local626 = shl nsw i64 %address.local625, 3, !dbg !12
  %address.local627 = add i64 %address.local615, %address.local626, !dbg !12
  %address.local628 = sub i64 %address.local608, %address.local627, !dbg !12
  %address.local629 = shl i64 %address.local628, 7, !dbg !12
  %address.local630 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local631 = and i32 %address.local630, 2, !dbg !12
  %address.local632 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local633 = and i32 %address.local632, 16, !dbg !12
  %address.local634 = or disjoint i32 %address.local631, %address.local633, !dbg !12
  %address.local635 = or disjoint i32 %address.local634, 12, !dbg !12
  %address.local636 = zext nneg i32 %address.local635 to i64, !dbg !12
  %address.local637 = or disjoint i64 %address.local629, %address.local636, !dbg !12
  %address.local638 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local637, !dbg !12
  %dma.source.bits1292 = ptrtoint ptr addrspace(1) %address.local638 to i64, !dbg !12
  %dma.source.opaque1293 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1292) #10, !dbg !12
  %dma.source.ptr1294 = inttoptr i64 %dma.source.opaque1293 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1294, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local639 = lshr i64 %address.local414, 2, !dbg !12
  %address.local640 = or disjoint i64 %address.local639, 448, !dbg !12
  %address.local641 = and i64 %address.local640, 127, !dbg !12
  %address.local642 = lshr i64 %address.local641, 3, !dbg !12
  %address.local643 = mul nuw nsw i64 %address.local642, 7, !dbg !12
  %address.local644 = add nuw nsw i64 %address.local641, %address.local643, !dbg !12
  %address.local645 = and i64 %address.local640, 124, !dbg !12
  %address.local646 = add nuw nsw i64 %address.local644, %address.local645, !dbg !12
  %address.local647 = sub nuw nsw i64 %address.local641, %address.local642, !dbg !12
  %address.local648 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local649 = and i32 %address.local648, 4, !dbg !12
  %address.local650 = add nuw nsw i32 %address.local649, 12, !dbg !12
  %address.local651 = zext nneg i32 %address.local650 to i64, !dbg !12
  %address.local652 = add nuw nsw i64 %address.local647, %address.local651, !dbg !12
  %address.local653 = and i64 %address.local652, 252, !dbg !12
  %address.local654 = lshr i32 %address.thread350, 5, !dbg !12
  %address.local655 = and i32 %address.local654, 4, !dbg !12
  %address.local656 = add nuw nsw i32 %address.local655, 12, !dbg !12
  %address.local657 = zext nneg i32 %address.local656 to i64, !dbg !12
  %address.local658 = sub nuw nsw i64 %address.local640, %address.local657, !dbg !12
  %address.local659 = and i64 %address.local658, 120, !dbg !12
  %address.local660 = add nuw nsw i64 %address.local653, %address.local659, !dbg !12
  %address.local661 = sub nsw i64 %address.local646, %address.local660, !dbg !12
  %address.local662 = shl nsw i64 %address.local661, 7, !dbg !12
  %address.local663 = shl nuw nsw i32 %address.thread350, 1, !dbg !12
  %address.local664 = and i32 %address.local663, 2, !dbg !12
  %address.local665 = shl nuw nsw i32 %address.thread350, 3, !dbg !12
  %address.local666 = and i32 %address.local665, 16, !dbg !12
  %address.local667 = or disjoint i32 %address.local664, %address.local666, !dbg !12
  %address.local668 = or disjoint i32 %address.local667, 12, !dbg !12
  %address.local669 = zext nneg i32 %address.local668 to i64, !dbg !12
  %address.local670 = or disjoint i64 %address.local662, %address.local669, !dbg !12
  %address.local671 = getelementptr [2 x i8], ptr addrspace(1) %138, i64 %address.local670, !dbg !12
  %dma.source.bits1295 = ptrtoint ptr addrspace(1) %address.local671 to i64, !dbg !12
  %dma.source.opaque1296 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1295) #10, !dbg !12
  %dma.source.ptr1297 = inttoptr i64 %dma.source.opaque1296 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1297, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local672 = and i32 %address.thread350, 192, !dbg !13
  %address.local673 = shl nuw nsw i32 %address.thread350, 2, !dbg !13
  %address.local674 = and i32 %address.local673, 60, !dbg !13
  %address.local675 = or disjoint i32 %address.local672, %address.local674, !dbg !13
  %address.local676 = shl nuw nsw i32 %address.thread350, 5, !dbg !13
  %address.local677 = and i32 %address.local676, 1536, !dbg !13
  %address.local678 = or disjoint i32 %address.local675, %address.local677, !dbg !13
  %address.local679 = zext nneg i32 %address.local678 to i64, !dbg !13
  %address.local680 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local679, !dbg !13
  %139 = load <4 x half>, ptr addrspace(3) %address.local680, align 2, !dbg !13
  %address.local681 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local679, !dbg !13
  %address.local682 = getelementptr i8, ptr addrspace(3) %address.local681, i64 4096, !dbg !13
  %140 = load <4 x half>, ptr addrspace(3) %address.local682, align 2, !dbg !13
  %address.local683 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local679, !dbg !13
  %address.local684 = getelementptr i8, ptr addrspace(3) %address.local683, i64 512, !dbg !13
  %141 = load <4 x half>, ptr addrspace(3) %address.local684, align 2, !dbg !13
  %142 = fmul <4 x half> %139, splat (half 0xH3015), !dbg !13
  %address.local685 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local679, !dbg !13
  %address.local686 = getelementptr i8, ptr addrspace(3) %address.local685, i64 4608, !dbg !13
  %143 = load <4 x half>, ptr addrspace(3) %address.local686, align 2, !dbg !13
  %144 = fmul <4 x half> %140, splat (half 0xH3015), !dbg !13
  %145 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !13
  %146 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local687 = shl i64 %address.base, 6, !dbg !16
  %address.base688 = call i64 asm sideeffect "", "=v,0"(i64 %.neg201) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local689 = shl i64 %address.base688, 5, !dbg !16
  %address.local690 = add i64 %address.local687, %address.local689, !dbg !16
  %address.local691 = mul nuw nsw i64 %address.local414, 400, !dbg !16
  %address.local692 = lshr i64 %address.local414, 2, !dbg !16
  %address.local693 = mul nsw i64 %address.local692, -1664, !dbg !16
  %address.local694 = add nsw i64 %address.local691, %address.local693, !dbg !16
  %address.local695 = lshr i64 %address.local414, 2, !dbg !16
  %address.local696 = and i64 %address.local695, 3, !dbg !16
  %address.local697 = shl nuw nsw i64 %address.local696, 3, !dbg !16
  %address.local698 = add nsw i64 %address.local694, %address.local697, !dbg !16
  %address.local699 = lshr i64 %address.local414, 1, !dbg !16
  %address.local700 = shl nuw nsw i64 %address.local699, 5, !dbg !16
  %address.local701 = add nsw i64 %address.local698, %address.local700, !dbg !16
  %address.local702 = and i64 %address.local414, 63, !dbg !13
  %147 = shl nuw nsw i64 %address.local414, 6, !dbg !16
  %address.local704 = and i64 %147, 3072, !dbg !16
  %address.local705 = add nsw i64 %address.local701, %address.local704, !dbg !16
  %address.local706 = and i64 %address.local414, 32, !dbg !16
  %address.local707 = add i64 %address.local705, %address.local706, !dbg !16
  %address.local708 = add i64 %address.local690, %address.local707, !dbg !16
  %address.local709 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local708, !dbg !16
  %148 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %149 = shufflevector <8 x half> %148, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %149, <4 x float> zeroinitializer) #9, !dbg !16
  %151 = shufflevector <8 x half> %148, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %151, <4 x float> %150) #9, !dbg !16
  %address.base710 = call i64 asm sideeffect "", "=v,0"(i64 %.neg204) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local711 = shl i64 %address.base710, 7, !dbg !16
  %address.local712 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local711, !dbg !16
  %address.base713 = call i64 asm sideeffect "", "=v,0"(i64 %.neg206) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local714 = shl i64 %address.base713, 6, !dbg !16
  %address.local715 = getelementptr i8, ptr addrspace(3) %address.local712, i64 %address.local714, !dbg !16
  %address.local716 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local717 = and i32 %address.local716, 6, !dbg !16
  %narrow = mul nuw nsw i32 %address.local717, 144, !dbg !16
  %address.local719 = zext nneg i32 %narrow to i64, !dbg !16
  %address.local720 = getelementptr i8, ptr addrspace(3) %address.local715, i64 %address.local719, !dbg !16
  %address.local721 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local722 = and i32 %address.local721, 24, !dbg !16
  %address.local723 = shl nuw nsw i32 %address.thread350, 8, !dbg !16
  %address.local724 = and i32 %address.local723, 768, !dbg !16
  %address.local725 = or disjoint i32 %address.local722, %address.local724, !dbg !16
  %address.local726 = zext nneg i32 %address.local725 to i64, !dbg !16
  %address.local727 = getelementptr [2 x i8], ptr addrspace(3) %address.local720, i64 %address.local726, !dbg !16
  %address.local728 = shl nuw nsw i32 %address.thread350, 4, !dbg !16
  %address.local729 = and i32 %address.local728, 32, !dbg !16
  %address.local730 = zext nneg i32 %address.local729 to i64, !dbg !16
  %address.local731 = getelementptr [2 x i8], ptr addrspace(3) %address.local727, i64 %address.local730, !dbg !16
  %address.local732 = getelementptr [2 x i8], ptr addrspace(3) %address.local731, i64 %address.local704, !dbg !16
  %address.local733 = getelementptr [2 x i8], ptr addrspace(3) %address.local732, i64 %address.local706, !dbg !16
  %address.local734 = getelementptr i8, ptr addrspace(3) %address.local733, i64 64, !dbg !16
  %153 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %address.local735 = shl nuw nsw i64 %address.local414, 2, !dbg !13
  %address.local736 = mul nsw i64 %address.local692, -18, !dbg !16
  %address.local737 = add nsw i64 %address.local735, %address.local736, !dbg !16
  %address.local738 = add nsw i64 %address.local737, %address.local699, !dbg !16
  %address.local739 = lshr i64 %address.local702, 5, !dbg !16
  %address.local740 = add nsw i64 %address.local738, %address.local739, !dbg !16
  %address.local741 = sdiv i64 %address.local740, -2, !dbg !16
  %address.local742 = shl i64 %address.local741, 7, !dbg !16
  %address.local743 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local742, !dbg !16
  %address.local744 = mul nuw nsw i64 %address.local414, 18, !dbg !16
  %address.local745 = mul nsw i64 %address.local692, -72, !dbg !16
  %address.local746 = add nsw i64 %address.local744, %address.local745, !dbg !16
  %address.local747 = add nsw i64 %address.local746, %address.local696, !dbg !16
  %address.local748 = add nsw i64 %address.local747, 1, !dbg !16
  %address.local749 = sdiv i64 %address.local748, -4, !dbg !16
  %address.local750 = shl i64 %address.local749, 6, !dbg !16
  %address.local751 = getelementptr i8, ptr addrspace(3) %address.local743, i64 %address.local750, !dbg !16
  %address.local752 = getelementptr [2 x i8], ptr addrspace(3) %address.local751, i64 %address.local691, !dbg !16
  %address.local753 = getelementptr [2 x i8], ptr addrspace(3) %address.local752, i64 %address.local693, !dbg !16
  %address.local754 = getelementptr [2 x i8], ptr addrspace(3) %address.local753, i64 %address.local697, !dbg !16
  %address.local755 = getelementptr [2 x i8], ptr addrspace(3) %address.local754, i64 %address.local700, !dbg !16
  %address.local756 = getelementptr [2 x i8], ptr addrspace(3) %address.local755, i64 %address.local704, !dbg !16
  %address.local757 = getelementptr [2 x i8], ptr addrspace(3) %address.local756, i64 %address.local706, !dbg !16
  %address.local758 = getelementptr i8, ptr addrspace(3) %address.local757, i64 144, !dbg !16
  %154 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %155 = shl nuw nsw i64 %address.local692, 4, !dbg !16
  %address.local760 = sub nsw i64 %address.local735, %155, !dbg !16
  %address.local761 = lshr i32 %address.thread350, 1, !dbg !16
  %address.local762 = and i32 %address.local761, 1, !dbg !16
  %address.local763 = zext nneg i32 %address.local762 to i64, !dbg !16
  %address.local764 = or disjoint i64 %address.local760, %address.local763, !dbg !16
  %address.local765 = add nuw nsw i64 %address.local764, %address.local739, !dbg !16
  %address.local766 = add nsw i64 %address.local765, 3, !dbg !16
  %address.local767 = sdiv i64 %address.local766, -2, !dbg !16
  %address.local768 = shl i64 %address.local767, 7, !dbg !16
  %address.local769 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local768, !dbg !16
  %address.local770 = shl nuw nsw i64 %address.local414, 4, !dbg !16
  %156 = shl nuw nsw i64 %address.local692, 6, !dbg !16
  %address.local772 = sub nsw i64 %address.local770, %156, !dbg !16
  %address.local773 = or disjoint i64 %address.local772, %address.local696, !dbg !16
  %address.local774 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local775 = and i32 %address.local774, 6, !dbg !16
  %address.local776 = or disjoint i32 %address.local775, 1, !dbg !16
  %address.local777 = zext nneg i32 %address.local776 to i64, !dbg !16
  %address.local778 = add nuw nsw i64 %address.local773, %address.local777, !dbg !16
  %address.local779 = sdiv i64 %address.local778, -4, !dbg !16
  %address.local780 = shl i64 %address.local779, 6, !dbg !16
  %address.local781 = getelementptr i8, ptr addrspace(3) %address.local769, i64 %address.local780, !dbg !16
  %address.local782 = mul nuw nsw i64 %address.local777, 144, !dbg !16
  %address.local783 = getelementptr i8, ptr addrspace(3) %address.local781, i64 %address.local782, !dbg !16
  %address.local784 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local785 = and i32 %address.local784, 24, !dbg !16
  %address.local786 = shl nuw nsw i32 %address.thread350, 8, !dbg !16
  %address.local787 = and i32 %address.local786, 768, !dbg !16
  %address.local788 = or disjoint i32 %address.local785, %address.local787, !dbg !16
  %address.local789 = zext nneg i32 %address.local788 to i64, !dbg !16
  %address.local790 = getelementptr [2 x i8], ptr addrspace(3) %address.local783, i64 %address.local789, !dbg !16
  %address.local791 = shl nuw nsw i32 %address.thread350, 4, !dbg !16
  %address.local792 = and i32 %address.local791, 32, !dbg !16
  %address.local793 = zext nneg i32 %address.local792 to i64, !dbg !16
  %address.local794 = getelementptr [2 x i8], ptr addrspace(3) %address.local790, i64 %address.local793, !dbg !16
  %address.local795 = getelementptr [2 x i8], ptr addrspace(3) %address.local794, i64 %address.local704, !dbg !16
  %address.local796 = getelementptr [2 x i8], ptr addrspace(3) %address.local795, i64 %address.local706, !dbg !16
  %address.local797 = getelementptr i8, ptr addrspace(3) %address.local796, i64 192, !dbg !16
  %157 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local708, !dbg !16
  %address.local799 = getelementptr i8, ptr addrspace(3) %address.local798, i64 1024, !dbg !16
  %169 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %167, <4 x float> %168) #9, !dbg !16
  %address.local800 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local801 = and i32 %address.local800, 6, !dbg !16
  %address.local802 = or disjoint i32 %address.local801, 8, !dbg !16
  %address.local803 = mul nuw nsw i32 %address.local802, 72, !dbg !16
  %address.local804 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local805 = and i32 %address.local804, 24, !dbg !16
  %address.local806 = shl nuw nsw i32 %address.thread350, 8, !dbg !16
  %address.local807 = and i32 %address.local806, 768, !dbg !16
  %address.local808 = or disjoint i32 %address.local805, %address.local807, !dbg !16
  %address.local809 = or disjoint i32 %address.local808, 1024, !dbg !16
  %address.local810 = add nuw nsw i32 %address.local803, %address.local809, !dbg !16
  %address.local811 = shl nuw nsw i32 %address.thread350, 4, !dbg !16
  %address.local812 = and i32 %address.local811, 32, !dbg !16
  %address.local813 = or disjoint i32 %address.local812, 64, !dbg !16
  %address.local814 = add nuw nsw i32 %address.local810, %address.local813, !dbg !16
  %address.local815 = zext nneg i32 %address.local814 to i64, !dbg !16
  %address.local816 = add nuw nsw i64 %address.local704, %address.local815, !dbg !16
  %address.local817 = add nuw nsw i64 %address.local816, %address.local706, !dbg !16
  %address.local818 = shl nuw nsw i32 %address.thread350, 2, !dbg !16
  %address.local819 = and i32 %address.local818, 12, !dbg !16
  %address.local820 = lshr i32 %address.thread350, 1, !dbg !16
  %address.local821 = and i32 %address.local820, 1, !dbg !16
  %address.local822 = or disjoint i32 %address.local819, %address.local821, !dbg !16
  %address.local823 = or disjoint i32 %address.local822, 18, !dbg !16
  %address.local824 = zext nneg i32 %address.local823 to i64, !dbg !16
  %address.local825 = add nuw nsw i64 %address.local739, %address.local824, !dbg !16
  %address.local826 = shl nuw nsw i64 %address.local825, 5, !dbg !16
  %address.local827 = add nuw nsw i64 %address.local826, 32, !dbg !16
  %address.local828 = and i64 %address.local827, 274877906880, !dbg !16
  %address.local829 = lshr i32 %address.thread350, 2, !dbg !16
  %address.local830 = and i32 %address.local829, 2, !dbg !16
  %address.local831 = shl nuw nsw i32 %address.thread350, 4, !dbg !16
  %address.local832 = and i32 %address.local831, 48, !dbg !16
  %address.local833 = or disjoint i32 %address.local830, %address.local832, !dbg !16
  %address.local834 = or disjoint i32 %address.local833, 64, !dbg !16
  %address.local835 = add nuw nsw i32 %address.local834, %address.local802, !dbg !16
  %address.local836 = shl nuw nsw i32 %address.local835, 3, !dbg !16
  %address.local837 = and i32 %address.local836, 2016, !dbg !16
  %address.local838 = zext nneg i32 %address.local837 to i64, !dbg !16
  %address.local839 = add nuw nsw i64 %address.local828, %address.local838, !dbg !16
  %address.local840 = sub nsw i64 %address.local817, %address.local839, !dbg !16
  %address.local841 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local840, !dbg !16
  %address.local842 = getelementptr i8, ptr addrspace(3) %address.local841, i64 64, !dbg !16
  %171 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %address.local843 = add nuw nsw i64 %address.local735, 18, !dbg !16
  %address.local844 = add nsw i64 %address.local843, %address.local736, !dbg !16
  %address.local845 = add nsw i64 %address.local844, %address.local699, !dbg !16
  %address.local846 = add nsw i64 %address.local845, %address.local739, !dbg !16
  %address.local847 = sdiv i64 %address.local846, -2, !dbg !16
  %address.local848 = shl i64 %address.local847, 7, !dbg !16
  %address.local849 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local848, !dbg !16
  %address.local850 = add nuw nsw i64 %address.local744, 72, !dbg !16
  %address.local851 = add nsw i64 %address.local850, %address.local745, !dbg !16
  %address.local852 = add nsw i64 %address.local851, %address.local696, !dbg !16
  %address.local853 = add nsw i64 %address.local852, -15, !dbg !16
  %address.local854 = sdiv i64 %address.local853, -4, !dbg !16
  %address.local855 = shl i64 %address.local854, 6, !dbg !16
  %address.local856 = getelementptr i8, ptr addrspace(3) %address.local849, i64 %address.local855, !dbg !16
  %address.local857 = getelementptr [2 x i8], ptr addrspace(3) %address.local856, i64 %address.local691, !dbg !16
  %address.local858 = getelementptr [2 x i8], ptr addrspace(3) %address.local857, i64 %address.local693, !dbg !16
  %address.local859 = getelementptr [2 x i8], ptr addrspace(3) %address.local858, i64 %address.local697, !dbg !16
  %address.local860 = getelementptr [2 x i8], ptr addrspace(3) %address.local859, i64 %address.local700, !dbg !16
  %address.local861 = getelementptr [2 x i8], ptr addrspace(3) %address.local860, i64 %address.local704, !dbg !16
  %address.local862 = getelementptr [2 x i8], ptr addrspace(3) %address.local861, i64 %address.local706, !dbg !16
  %address.local863 = getelementptr i8, ptr addrspace(3) %address.local862, i64 3216, !dbg !16
  %172 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
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
  %address.local864 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local865 = and i32 %address.local864, 6, !dbg !16
  %address.local866 = or disjoint i32 %address.local865, 9, !dbg !16
  %address.local867 = mul nuw nsw i32 %address.local866, 72, !dbg !16
  %address.local868 = shl nuw nsw i32 %address.thread350, 1, !dbg !16
  %address.local869 = and i32 %address.local868, 24, !dbg !16
  %address.local870 = shl nuw nsw i32 %address.thread350, 8, !dbg !16
  %address.local871 = and i32 %address.local870, 768, !dbg !16
  %address.local872 = or disjoint i32 %address.local869, %address.local871, !dbg !16
  %address.local873 = or disjoint i32 %address.local872, 1024, !dbg !16
  %address.local874 = add nuw nsw i32 %address.local867, %address.local873, !dbg !16
  %address.local875 = shl nuw nsw i32 %address.thread350, 4, !dbg !16
  %address.local876 = and i32 %address.local875, 32, !dbg !16
  %address.local877 = or disjoint i32 %address.local876, 64, !dbg !16
  %address.local878 = add nuw nsw i32 %address.local874, %address.local877, !dbg !16
  %address.local879 = zext nneg i32 %address.local878 to i64, !dbg !16
  %address.local880 = add nuw nsw i64 %address.local704, %address.local879, !dbg !16
  %address.local881 = add nuw nsw i64 %address.local880, %address.local706, !dbg !16
  %address.local882 = shl nuw nsw i32 %address.thread350, 2, !dbg !16
  %address.local883 = and i32 %address.local882, 12, !dbg !16
  %address.local884 = lshr i32 %address.thread350, 1, !dbg !16
  %address.local885 = and i32 %address.local884, 1, !dbg !16
  %address.local886 = or disjoint i32 %address.local883, %address.local885, !dbg !16
  %address.local887 = or disjoint i32 %address.local886, 18, !dbg !16
  %address.local888 = zext nneg i32 %address.local887 to i64, !dbg !16
  %address.local889 = add nuw nsw i64 %address.local739, %address.local888, !dbg !16
  %address.local890 = shl nuw nsw i64 %address.local889, 5, !dbg !16
  %address.local891 = add nuw nsw i64 %address.local890, 96, !dbg !16
  %address.local892 = and i64 %address.local891, 274877906880, !dbg !16
  %address.local893 = lshr i32 %address.thread350, 2, !dbg !16
  %address.local894 = and i32 %address.local893, 3, !dbg !16
  %address.local895 = shl nuw nsw i32 %address.thread350, 4, !dbg !16
  %address.local896 = and i32 %address.local895, 48, !dbg !16
  %address.local897 = or disjoint i32 %address.local894, %address.local896, !dbg !16
  %address.local898 = or disjoint i32 %address.local897, 64, !dbg !16
  %address.local899 = add nuw nsw i32 %address.local898, %address.local866, !dbg !16
  %address.local900 = shl nuw nsw i32 %address.local899, 3, !dbg !16
  %address.local901 = and i32 %address.local900, 2016, !dbg !16
  %address.local902 = zext nneg i32 %address.local901 to i64, !dbg !16
  %address.local903 = add nuw nsw i64 %address.local892, %address.local902, !dbg !16
  %address.local904 = sub nsw i64 %address.local881, %address.local903, !dbg !16
  %address.local905 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local904, !dbg !16
  %address.local906 = getelementptr i8, ptr addrspace(3) %address.local905, i64 192, !dbg !16
  %185 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %186 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %187 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %142, <4 x half> %187, <4 x float> zeroinitializer) #9, !dbg !16
  %189 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %189, <4 x float> %188) #9, !dbg !16
  %191 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %192 = shufflevector <8 x half> %186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %191, <4 x float> zeroinitializer) #9, !dbg !16
  %194 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %192, <4 x float> %193) #9, !dbg !16
  %196 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %197 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %210 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %211 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %212 = shufflevector <8 x half> %210, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %212, <4 x float> zeroinitializer) #9, !dbg !16
  %214 = shufflevector <8 x half> %210, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %214, <4 x float> %213) #9, !dbg !16
  %216 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %217 = shufflevector <8 x half> %211, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %216, <4 x float> zeroinitializer) #9, !dbg !16
  %219 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %217, <4 x float> %218) #9, !dbg !16
  %221 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %222 = shufflevector <8 x half> %219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %222, <4 x float> zeroinitializer) #9, !dbg !16
  %224 = shufflevector <8 x half> %219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %224, <4 x float> %223) #9, !dbg !16
  %226 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %227 = shufflevector <8 x half> %221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %226, <4 x float> zeroinitializer) #9, !dbg !16
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %227, <4 x float> %228) #9, !dbg !16
  %address.local907 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local356, !dbg !11
  %dma.source.bits1298 = ptrtoint ptr addrspace(1) %address.local907 to i64, !dbg !11
  %dma.source.opaque1299 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1298) #10, !dbg !11
  %dma.source.ptr1300 = inttoptr i64 %dma.source.opaque1299 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1300, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local908 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local364, !dbg !11
  %dma.source.bits1301 = ptrtoint ptr addrspace(1) %address.local908 to i64, !dbg !11
  %dma.source.opaque1302 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1301) #10, !dbg !11
  %dma.source.ptr1303 = inttoptr i64 %dma.source.opaque1302 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1303, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local909 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local372, !dbg !11
  %dma.source.bits1304 = ptrtoint ptr addrspace(1) %address.local909 to i64, !dbg !11
  %dma.source.opaque1305 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1304) #10, !dbg !11
  %dma.source.ptr1306 = inttoptr i64 %dma.source.opaque1305 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1306, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local910 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local380, !dbg !11
  %dma.source.bits1307 = ptrtoint ptr addrspace(1) %address.local910 to i64, !dbg !11
  %dma.source.opaque1308 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1307) #10, !dbg !11
  %dma.source.ptr1309 = inttoptr i64 %dma.source.opaque1308 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1309, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local911 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local388, !dbg !11
  %dma.source.bits1310 = ptrtoint ptr addrspace(1) %address.local911 to i64, !dbg !11
  %dma.source.opaque1311 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1310) #10, !dbg !11
  %dma.source.ptr1312 = inttoptr i64 %dma.source.opaque1311 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1312, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local912 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local396, !dbg !11
  %dma.source.bits1313 = ptrtoint ptr addrspace(1) %address.local912 to i64, !dbg !11
  %dma.source.opaque1314 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1313) #10, !dbg !11
  %dma.source.ptr1315 = inttoptr i64 %dma.source.opaque1314 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1315, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local913 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local404, !dbg !11
  %dma.source.bits1316 = ptrtoint ptr addrspace(1) %address.local913 to i64, !dbg !11
  %dma.source.opaque1317 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1316) #10, !dbg !11
  %dma.source.ptr1318 = inttoptr i64 %dma.source.opaque1317 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1318, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local914 = getelementptr [2 x i8], ptr addrspace(1) %119, i64 %address.local412, !dbg !11
  %dma.source.bits1319 = ptrtoint ptr addrspace(1) %address.local914 to i64, !dbg !11
  %dma.source.opaque1320 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1319) #10, !dbg !11
  %dma.source.ptr1321 = inttoptr i64 %dma.source.opaque1320 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1321, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %230 = or disjoint i64 %137, 32, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %231 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %230, !dbg !12
  %address.local915 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local439, !dbg !12
  %dma.source.bits1322 = ptrtoint ptr addrspace(1) %address.local915 to i64, !dbg !12
  %dma.source.opaque1323 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1322) #10, !dbg !12
  %dma.source.ptr1324 = inttoptr i64 %dma.source.opaque1323 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1324, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local916 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local471, !dbg !12
  %dma.source.bits1325 = ptrtoint ptr addrspace(1) %address.local916 to i64, !dbg !12
  %dma.source.opaque1326 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1325) #10, !dbg !12
  %dma.source.ptr1327 = inttoptr i64 %dma.source.opaque1326 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1327, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local917 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local501, !dbg !12
  %dma.source.bits1328 = ptrtoint ptr addrspace(1) %address.local917 to i64, !dbg !12
  %dma.source.opaque1329 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1328) #10, !dbg !12
  %dma.source.ptr1330 = inttoptr i64 %dma.source.opaque1329 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1330, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local918 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local534, !dbg !12
  %dma.source.bits1331 = ptrtoint ptr addrspace(1) %address.local918 to i64, !dbg !12
  %dma.source.opaque1332 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1331) #10, !dbg !12
  %dma.source.ptr1333 = inttoptr i64 %dma.source.opaque1332 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1333, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local919 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local568, !dbg !12
  %dma.source.bits1334 = ptrtoint ptr addrspace(1) %address.local919 to i64, !dbg !12
  %dma.source.opaque1335 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1334) #10, !dbg !12
  %dma.source.ptr1336 = inttoptr i64 %dma.source.opaque1335 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1336, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local920 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local601, !dbg !12
  %dma.source.bits1337 = ptrtoint ptr addrspace(1) %address.local920 to i64, !dbg !12
  %dma.source.opaque1338 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1337) #10, !dbg !12
  %dma.source.ptr1339 = inttoptr i64 %dma.source.opaque1338 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1339, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local921 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local637, !dbg !12
  %dma.source.bits1340 = ptrtoint ptr addrspace(1) %address.local921 to i64, !dbg !12
  %dma.source.opaque1341 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1340) #10, !dbg !12
  %dma.source.ptr1342 = inttoptr i64 %dma.source.opaque1341 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1342, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local922 = getelementptr [2 x i8], ptr addrspace(1) %231, i64 %address.local670, !dbg !12
  %dma.source.bits1343 = ptrtoint ptr addrspace(1) %address.local922 to i64, !dbg !12
  %dma.source.opaque1344 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1343) #10, !dbg !12
  %dma.source.ptr1345 = inttoptr i64 %dma.source.opaque1344 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1345, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %232 = load <4 x half>, ptr addrspace(3) %address.local680, align 2, !dbg !13
  %233 = load <4 x half>, ptr addrspace(3) %address.local682, align 2, !dbg !13
  %234 = load <4 x half>, ptr addrspace(3) %address.local684, align 2, !dbg !13
  %235 = fmul <4 x half> %232, splat (half 0xH3015), !dbg !13
  %236 = load <4 x half>, ptr addrspace(3) %address.local686, align 2, !dbg !13
  %237 = fmul <4 x half> %233, splat (half 0xH3015), !dbg !13
  %238 = fmul <4 x half> %234, splat (half 0xH3015), !dbg !13
  %239 = fmul <4 x half> %236, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %240 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %241 = shufflevector <8 x half> %240, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %241, <4 x float> %152) #9, !dbg !16
  %243 = shufflevector <8 x half> %240, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %243, <4 x float> %242) #9, !dbg !16
  %245 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %246 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %247 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %259 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %257, <4 x float> %258) #9, !dbg !16
  %261 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %262 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
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
  %275 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %276 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %277 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %235, <4 x half> %277, <4 x float> %190) #9, !dbg !16
  %279 = shufflevector <8 x half> %275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %237, <4 x half> %279, <4 x float> %278) #9, !dbg !16
  %281 = shufflevector <8 x half> %276, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %282 = shufflevector <8 x half> %276, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %281, <4 x float> %195) #9, !dbg !16
  %284 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %282, <4 x float> %283) #9, !dbg !16
  %286 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %287 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %300 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %301 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %302 = shufflevector <8 x half> %300, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %302, <4 x float> %215) #9, !dbg !16
  %304 = shufflevector <8 x half> %300, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %304, <4 x float> %303) #9, !dbg !16
  %306 = shufflevector <8 x half> %301, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %307 = shufflevector <8 x half> %301, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %306, <4 x float> %220) #9, !dbg !16
  %309 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
  %310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %307, <4 x float> %308) #9, !dbg !16
  %311 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %312 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %312, <4 x float> %225) #9, !dbg !16
  %314 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %314, <4 x float> %313) #9, !dbg !16
  %316 = shufflevector <8 x half> %311, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %317 = shufflevector <8 x half> %311, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %238, <4 x half> %316, <4 x float> %229) #9, !dbg !16
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %239, <4 x half> %317, <4 x float> %318) #9, !dbg !16
  %address.local923 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local356, !dbg !11
  %dma.source.bits1346 = ptrtoint ptr addrspace(1) %address.local923 to i64, !dbg !11
  %dma.source.opaque1347 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1346) #10, !dbg !11
  %dma.source.ptr1348 = inttoptr i64 %dma.source.opaque1347 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1348, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local924 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local364, !dbg !11
  %dma.source.bits1349 = ptrtoint ptr addrspace(1) %address.local924 to i64, !dbg !11
  %dma.source.opaque1350 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1349) #10, !dbg !11
  %dma.source.ptr1351 = inttoptr i64 %dma.source.opaque1350 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1351, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local925 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local372, !dbg !11
  %dma.source.bits1352 = ptrtoint ptr addrspace(1) %address.local925 to i64, !dbg !11
  %dma.source.opaque1353 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1352) #10, !dbg !11
  %dma.source.ptr1354 = inttoptr i64 %dma.source.opaque1353 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1354, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local926 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local380, !dbg !11
  %dma.source.bits1355 = ptrtoint ptr addrspace(1) %address.local926 to i64, !dbg !11
  %dma.source.opaque1356 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1355) #10, !dbg !11
  %dma.source.ptr1357 = inttoptr i64 %dma.source.opaque1356 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1357, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local927 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local388, !dbg !11
  %dma.source.bits1358 = ptrtoint ptr addrspace(1) %address.local927 to i64, !dbg !11
  %dma.source.opaque1359 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1358) #10, !dbg !11
  %dma.source.ptr1360 = inttoptr i64 %dma.source.opaque1359 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1360, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local928 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local396, !dbg !11
  %dma.source.bits1361 = ptrtoint ptr addrspace(1) %address.local928 to i64, !dbg !11
  %dma.source.opaque1362 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1361) #10, !dbg !11
  %dma.source.ptr1363 = inttoptr i64 %dma.source.opaque1362 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1363, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local929 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local404, !dbg !11
  %dma.source.bits1364 = ptrtoint ptr addrspace(1) %address.local929 to i64, !dbg !11
  %dma.source.opaque1365 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1364) #10, !dbg !11
  %dma.source.ptr1366 = inttoptr i64 %dma.source.opaque1365 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1366, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local930 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local412, !dbg !11
  %dma.source.bits1367 = ptrtoint ptr addrspace(1) %address.local930 to i64, !dbg !11
  %dma.source.opaque1368 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1367) #10, !dbg !11
  %dma.source.ptr1369 = inttoptr i64 %dma.source.opaque1368 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1369, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %320 = or disjoint i64 %137, 64, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %321 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %320, !dbg !12
  %address.local931 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local439, !dbg !12
  %dma.source.bits1370 = ptrtoint ptr addrspace(1) %address.local931 to i64, !dbg !12
  %dma.source.opaque1371 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1370) #10, !dbg !12
  %dma.source.ptr1372 = inttoptr i64 %dma.source.opaque1371 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1372, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local932 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local471, !dbg !12
  %dma.source.bits1373 = ptrtoint ptr addrspace(1) %address.local932 to i64, !dbg !12
  %dma.source.opaque1374 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1373) #10, !dbg !12
  %dma.source.ptr1375 = inttoptr i64 %dma.source.opaque1374 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1375, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local933 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local501, !dbg !12
  %dma.source.bits1376 = ptrtoint ptr addrspace(1) %address.local933 to i64, !dbg !12
  %dma.source.opaque1377 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1376) #10, !dbg !12
  %dma.source.ptr1378 = inttoptr i64 %dma.source.opaque1377 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1378, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local934 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local534, !dbg !12
  %dma.source.bits1379 = ptrtoint ptr addrspace(1) %address.local934 to i64, !dbg !12
  %dma.source.opaque1380 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1379) #10, !dbg !12
  %dma.source.ptr1381 = inttoptr i64 %dma.source.opaque1380 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1381, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local935 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local568, !dbg !12
  %dma.source.bits1382 = ptrtoint ptr addrspace(1) %address.local935 to i64, !dbg !12
  %dma.source.opaque1383 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1382) #10, !dbg !12
  %dma.source.ptr1384 = inttoptr i64 %dma.source.opaque1383 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1384, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local936 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local601, !dbg !12
  %dma.source.bits1385 = ptrtoint ptr addrspace(1) %address.local936 to i64, !dbg !12
  %dma.source.opaque1386 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1385) #10, !dbg !12
  %dma.source.ptr1387 = inttoptr i64 %dma.source.opaque1386 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1387, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local937 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local637, !dbg !12
  %dma.source.bits1388 = ptrtoint ptr addrspace(1) %address.local937 to i64, !dbg !12
  %dma.source.opaque1389 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1388) #10, !dbg !12
  %dma.source.ptr1390 = inttoptr i64 %dma.source.opaque1389 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1390, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local938 = getelementptr [2 x i8], ptr addrspace(1) %321, i64 %address.local670, !dbg !12
  %dma.source.bits1391 = ptrtoint ptr addrspace(1) %address.local938 to i64, !dbg !12
  %dma.source.opaque1392 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1391) #10, !dbg !12
  %dma.source.ptr1393 = inttoptr i64 %dma.source.opaque1392 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1393, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %address.local680, align 2, !dbg !13
  %323 = load <4 x half>, ptr addrspace(3) %address.local682, align 2, !dbg !13
  %324 = load <4 x half>, ptr addrspace(3) %address.local684, align 2, !dbg !13
  %325 = fmul <4 x half> %322, splat (half 0xH3015), !dbg !13
  %326 = load <4 x half>, ptr addrspace(3) %address.local686, align 2, !dbg !13
  %327 = fmul <4 x half> %323, splat (half 0xH3015), !dbg !13
  %328 = fmul <4 x half> %324, splat (half 0xH3015), !dbg !13
  %329 = fmul <4 x half> %326, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %330 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %331 = shufflevector <8 x half> %330, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %331, <4 x float> %244) #9, !dbg !16
  %333 = shufflevector <8 x half> %330, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %333, <4 x float> %332) #9, !dbg !16
  %335 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %336 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %337 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %349 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %347, <4 x float> %348) #9, !dbg !16
  %351 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %352 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
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
  %365 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %366 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %367 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %325, <4 x half> %367, <4 x float> %280) #9, !dbg !16
  %369 = shufflevector <8 x half> %365, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %327, <4 x half> %369, <4 x float> %368) #9, !dbg !16
  %371 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %372 = shufflevector <8 x half> %366, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %371, <4 x float> %285) #9, !dbg !16
  %374 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %372, <4 x float> %373) #9, !dbg !16
  %376 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %377 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %390 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %391 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %392 = shufflevector <8 x half> %390, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %392, <4 x float> %305) #9, !dbg !16
  %394 = shufflevector <8 x half> %390, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %394, <4 x float> %393) #9, !dbg !16
  %396 = shufflevector <8 x half> %391, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %397 = shufflevector <8 x half> %391, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %396, <4 x float> %310) #9, !dbg !16
  %399 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %397, <4 x float> %398) #9, !dbg !16
  %401 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %402 = shufflevector <8 x half> %399, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %403 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %402, <4 x float> %315) #9, !dbg !16
  %404 = shufflevector <8 x half> %399, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %404, <4 x float> %403) #9, !dbg !16
  %406 = shufflevector <8 x half> %401, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %407 = shufflevector <8 x half> %401, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %328, <4 x half> %406, <4 x float> %319) #9, !dbg !16
  %409 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %329, <4 x half> %407, <4 x float> %408) #9, !dbg !16
  %address.local939 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local356, !dbg !11
  %dma.source.bits1394 = ptrtoint ptr addrspace(1) %address.local939 to i64, !dbg !11
  %dma.source.opaque1395 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1394) #10, !dbg !11
  %dma.source.ptr1396 = inttoptr i64 %dma.source.opaque1395 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1396, ptr addrspace(3) %40, i32 4, i32 0, i32 0), !dbg !11
  %address.local940 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local364, !dbg !11
  %dma.source.bits1397 = ptrtoint ptr addrspace(1) %address.local940 to i64, !dbg !11
  %dma.source.opaque1398 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1397) #10, !dbg !11
  %dma.source.ptr1399 = inttoptr i64 %dma.source.opaque1398 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1399, ptr addrspace(3) %43, i32 4, i32 0, i32 0), !dbg !11
  %address.local941 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local372, !dbg !11
  %dma.source.bits1400 = ptrtoint ptr addrspace(1) %address.local941 to i64, !dbg !11
  %dma.source.opaque1401 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1400) #10, !dbg !11
  %dma.source.ptr1402 = inttoptr i64 %dma.source.opaque1401 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1402, ptr addrspace(3) %46, i32 4, i32 0, i32 0), !dbg !11
  %address.local942 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local380, !dbg !11
  %dma.source.bits1403 = ptrtoint ptr addrspace(1) %address.local942 to i64, !dbg !11
  %dma.source.opaque1404 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1403) #10, !dbg !11
  %dma.source.ptr1405 = inttoptr i64 %dma.source.opaque1404 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1405, ptr addrspace(3) %49, i32 4, i32 0, i32 0), !dbg !11
  %address.local943 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local388, !dbg !11
  %dma.source.bits1406 = ptrtoint ptr addrspace(1) %address.local943 to i64, !dbg !11
  %dma.source.opaque1407 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1406) #10, !dbg !11
  %dma.source.ptr1408 = inttoptr i64 %dma.source.opaque1407 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1408, ptr addrspace(3) %52, i32 4, i32 0, i32 0), !dbg !11
  %address.local944 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local396, !dbg !11
  %dma.source.bits1409 = ptrtoint ptr addrspace(1) %address.local944 to i64, !dbg !11
  %dma.source.opaque1410 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1409) #10, !dbg !11
  %dma.source.ptr1411 = inttoptr i64 %dma.source.opaque1410 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1411, ptr addrspace(3) %55, i32 4, i32 0, i32 0), !dbg !11
  %address.local945 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local404, !dbg !11
  %dma.source.bits1412 = ptrtoint ptr addrspace(1) %address.local945 to i64, !dbg !11
  %dma.source.opaque1413 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1412) #10, !dbg !11
  %dma.source.ptr1414 = inttoptr i64 %dma.source.opaque1413 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1414, ptr addrspace(3) %58, i32 4, i32 0, i32 0), !dbg !11
  %address.local946 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local412, !dbg !11
  %dma.source.bits1415 = ptrtoint ptr addrspace(1) %address.local946 to i64, !dbg !11
  %dma.source.opaque1416 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1415) #10, !dbg !11
  %dma.source.ptr1417 = inttoptr i64 %dma.source.opaque1416 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1417, ptr addrspace(3) %61, i32 4, i32 0, i32 0), !dbg !11
  %410 = or disjoint i64 %137, 96, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %411 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %410, !dbg !12
  %address.local947 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local439, !dbg !12
  %dma.source.bits1418 = ptrtoint ptr addrspace(1) %address.local947 to i64, !dbg !12
  %dma.source.opaque1419 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1418) #10, !dbg !12
  %dma.source.ptr1420 = inttoptr i64 %dma.source.opaque1419 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1420, ptr addrspace(3) %64, i32 4, i32 0, i32 0), !dbg !12
  %address.local948 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local471, !dbg !12
  %dma.source.bits1421 = ptrtoint ptr addrspace(1) %address.local948 to i64, !dbg !12
  %dma.source.opaque1422 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1421) #10, !dbg !12
  %dma.source.ptr1423 = inttoptr i64 %dma.source.opaque1422 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1423, ptr addrspace(3) %67, i32 4, i32 0, i32 0), !dbg !12
  %address.local949 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local501, !dbg !12
  %dma.source.bits1424 = ptrtoint ptr addrspace(1) %address.local949 to i64, !dbg !12
  %dma.source.opaque1425 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1424) #10, !dbg !12
  %dma.source.ptr1426 = inttoptr i64 %dma.source.opaque1425 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1426, ptr addrspace(3) %70, i32 4, i32 0, i32 0), !dbg !12
  %address.local950 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local534, !dbg !12
  %dma.source.bits1427 = ptrtoint ptr addrspace(1) %address.local950 to i64, !dbg !12
  %dma.source.opaque1428 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1427) #10, !dbg !12
  %dma.source.ptr1429 = inttoptr i64 %dma.source.opaque1428 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1429, ptr addrspace(3) %73, i32 4, i32 0, i32 0), !dbg !12
  %address.local951 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local568, !dbg !12
  %dma.source.bits1430 = ptrtoint ptr addrspace(1) %address.local951 to i64, !dbg !12
  %dma.source.opaque1431 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1430) #10, !dbg !12
  %dma.source.ptr1432 = inttoptr i64 %dma.source.opaque1431 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1432, ptr addrspace(3) %76, i32 4, i32 0, i32 0), !dbg !12
  %address.local952 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local601, !dbg !12
  %dma.source.bits1433 = ptrtoint ptr addrspace(1) %address.local952 to i64, !dbg !12
  %dma.source.opaque1434 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1433) #10, !dbg !12
  %dma.source.ptr1435 = inttoptr i64 %dma.source.opaque1434 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1435, ptr addrspace(3) %79, i32 4, i32 0, i32 0), !dbg !12
  %address.local953 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local637, !dbg !12
  %dma.source.bits1436 = ptrtoint ptr addrspace(1) %address.local953 to i64, !dbg !12
  %dma.source.opaque1437 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1436) #10, !dbg !12
  %dma.source.ptr1438 = inttoptr i64 %dma.source.opaque1437 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1438, ptr addrspace(3) %82, i32 4, i32 0, i32 0), !dbg !12
  %address.local954 = getelementptr [2 x i8], ptr addrspace(1) %411, i64 %address.local670, !dbg !12
  %dma.source.bits1439 = ptrtoint ptr addrspace(1) %address.local954 to i64, !dbg !12
  %dma.source.opaque1440 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1439) #10, !dbg !12
  %dma.source.ptr1441 = inttoptr i64 %dma.source.opaque1440 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1441, ptr addrspace(3) %85, i32 4, i32 0, i32 0), !dbg !12
  %412 = load <4 x half>, ptr addrspace(3) %address.local680, align 2, !dbg !13
  %413 = load <4 x half>, ptr addrspace(3) %address.local682, align 2, !dbg !13
  %414 = load <4 x half>, ptr addrspace(3) %address.local684, align 2, !dbg !13
  %415 = fmul <4 x half> %412, splat (half 0xH3015), !dbg !13
  %416 = load <4 x half>, ptr addrspace(3) %address.local686, align 2, !dbg !13
  %417 = fmul <4 x half> %413, splat (half 0xH3015), !dbg !13
  %418 = fmul <4 x half> %414, splat (half 0xH3015), !dbg !13
  %419 = fmul <4 x half> %416, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local955 = shl nuw nsw i64 %address.local414, 5, !dbg !15
  %address.local956 = and i64 %address.local955, 8064, !dbg !15
  %address.local957 = shl nuw nsw i64 %address.local414, 3, !dbg !15
  %address.local958 = shl nuw nsw i64 %address.local414, 3, !dbg !15
  %address.local959 = and i64 %address.local958, 2016, !dbg !15
  %address.local960 = sub nsw i64 %address.local957, %address.local959, !dbg !15
  %address.local961 = add nsw i64 %address.local956, %address.local960, !dbg !15
  %420 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14421443 = add nsw i64 %137, %address.local961, !dbg !15
  %.idx14421443.tr = trunc i64 %.idx14421443 to i32, !dbg !15
  %dma.buffer.offset1251 = shl i32 %.idx14421443.tr, 1, !dbg !15
  %421 = insertelement <2 x i64> poison, i64 %420, i64 0, !dbg !15
  %422 = bitcast <2 x i64> %421 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1252 = shufflevector <4 x i32> %422, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %423 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1251, <4 x i32> %dma.buffer.rsrc1252) #11, !dbg !15
  %address.local963 = shl nuw nsw i32 %address.thread350, 3, !dbg !15
  %address.local964 = and i32 %address.local963, 24, !dbg !15
  %address.local965 = shl nuw nsw i32 %address.thread350, 5, !dbg !15
  %address.local966 = and i32 %address.local965, 8064, !dbg !15
  %address.local967 = or disjoint i32 %address.local964, %address.local966, !dbg !15
  %address.local968 = or disjoint i32 %address.local967, 8192, !dbg !15
  %424 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %425 = trunc nuw i64 %137 to i32, !dbg !15
  %.idx14441445.tr = or disjoint i32 %address.local968, %425, !dbg !15
  %dma.buffer.offset1249 = shl nuw nsw i32 %.idx14441445.tr, 1, !dbg !15
  %426 = insertelement <2 x i64> poison, i64 %424, i64 0, !dbg !15
  %427 = bitcast <2 x i64> %426 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1250 = shufflevector <4 x i32> %427, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %428 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1249, <4 x i32> %dma.buffer.rsrc1250) #11, !dbg !15
  %429 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
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
  %442 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
  %443 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %444 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
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
  %472 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
  %473 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %474 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %471, <4 x float> %470) #9, !dbg !16
  %475 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
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
  %511 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
  %512 = load <8 x half>, ptr addrspace(3) %address.local709, align 2, !dbg !16
  %513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %417, <4 x half> %510, <4 x float> %509) #9, !dbg !16
  %514 = load <8 x half>, ptr addrspace(3) %address.local734, align 2, !dbg !16
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
  %547 = load <8 x half>, ptr addrspace(3) %address.local758, align 2, !dbg !16
  %548 = load <8 x half>, ptr addrspace(3) %address.local797, align 2, !dbg !16
  %549 = load <8 x half>, ptr addrspace(3) %address.local799, align 2, !dbg !16
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
  %574 = load <8 x half>, ptr addrspace(3) %address.local842, align 2, !dbg !16
  %575 = load <8 x half>, ptr addrspace(3) %address.local863, align 2, !dbg !16
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
  %591 = load <8 x half>, ptr addrspace(3) %address.local906, align 2, !dbg !16
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
  %math.result241 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !26, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !26
  %math.packed242 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result243 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %741) #9, !dbg !26, !frisk.register_lifetimes !1
  %742 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !26
  %math.packed244 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %742) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %758) #9, !dbg !26, !frisk.register_lifetimes !1
  %759 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !26
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %759) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !26, !frisk.register_lifetimes !1
  %776 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !26
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %792) #9, !dbg !26, !frisk.register_lifetimes !1
  %793 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !26
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %793) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %809) #9, !dbg !26, !frisk.register_lifetimes !1
  %810 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !26
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %810) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %826) #9, !dbg !26, !frisk.register_lifetimes !1
  %827 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !26
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %827) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %843) #9, !dbg !26, !frisk.register_lifetimes !1
  %844 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !26
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %844) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %860) #9, !dbg !26, !frisk.register_lifetimes !1
  %861 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !26
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %861) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %877) #9, !dbg !26, !frisk.register_lifetimes !1
  %878 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !26
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %878) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %894) #9, !dbg !26, !frisk.register_lifetimes !1
  %895 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !26
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %895) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %911) #9, !dbg !26, !frisk.register_lifetimes !1
  %912 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !26
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %912) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %928) #9, !dbg !26, !frisk.register_lifetimes !1
  %929 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !26
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %929) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %945) #9, !dbg !26, !frisk.register_lifetimes !1
  %946 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !26
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %946) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %962) #9, !dbg !26, !frisk.register_lifetimes !1
  %963 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !26
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %963) #9, !dbg !25, !frisk.register_lifetimes !1
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
  %address.local971 = shl nuw nsw i32 %address.thread350, 3, !dbg !14
  %address.local972 = and i32 %address.local971, 120, !dbg !14
  %address.local973 = shl nuw nsw i32 %address.thread350, 5, !dbg !14
  %address.local974 = and i32 %address.local973, 1536, !dbg !14
  %address.local975 = or disjoint i32 %address.local972, %address.local974, !dbg !14
  %address.local976 = zext nneg i32 %address.local975 to i64, !dbg !14
  %address.local977 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local976, !dbg !14
  %997 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %998 = bitcast <4 x i32> %997 to <8 x half>, !dbg !14
  %address.local978 = and i64 %address.local414, 60, !dbg !14
  %address.local979 = add nuw nsw i64 %address.local978, %address.local414, !dbg !14
  %address.local980 = shl nuw nsw i64 %address.local979, 3, !dbg !14
  %address.local981 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !14
  %address.local982 = lshr i64 %address.local702, 4, !dbg !14
  %address.local983 = mul nuw nsw i64 %address.local982, 384, !dbg !14
  %address.local984 = getelementptr [2 x i8], ptr addrspace(3) %address.local981, i64 %address.local983, !dbg !14
  %address.local985 = mul nsw i64 %address.local692, -32, !dbg !14
  %address.local986 = getelementptr [2 x i8], ptr addrspace(3) %address.local984, i64 %address.local985, !dbg !14
  %address.local987 = getelementptr i8, ptr addrspace(3) %address.local986, i64 256, !dbg !14
  %999 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1000 = bitcast <4 x i32> %999 to <8 x half>, !dbg !14
  %address.local988 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !14
  %address.local989 = getelementptr [2 x i8], ptr addrspace(3) %address.local988, i64 %address.local983, !dbg !14
  %address.local990 = getelementptr [2 x i8], ptr addrspace(3) %address.local989, i64 %address.local985, !dbg !14
  %address.local991 = getelementptr i8, ptr addrspace(3) %address.local990, i64 512, !dbg !14
  %1001 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local991, i32 0) #0, !dbg !14
  %1002 = bitcast <4 x i32> %1001 to <8 x half>, !dbg !14
  %address.local992 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !14
  %address.local993 = getelementptr [2 x i8], ptr addrspace(3) %address.local992, i64 %address.local983, !dbg !14
  %address.local994 = getelementptr [2 x i8], ptr addrspace(3) %address.local993, i64 %address.local985, !dbg !14
  %address.local995 = getelementptr i8, ptr addrspace(3) %address.local994, i64 768, !dbg !14
  %1003 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local995, i32 0) #0, !dbg !14
  %1004 = bitcast <4 x i32> %1003 to <8 x half>, !dbg !14
  %address.local996 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local976, !dbg !14
  %address.local997 = getelementptr i8, ptr addrspace(3) %address.local996, i64 4096, !dbg !14
  %1005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local997, i32 0) #0, !dbg !14
  %1006 = bitcast <4 x i32> %1005 to <8 x half>, !dbg !14
  %address.local998 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !14
  %address.local999 = getelementptr [2 x i8], ptr addrspace(3) %address.local998, i64 %address.local983, !dbg !14
  %address.local1000 = getelementptr [2 x i8], ptr addrspace(3) %address.local999, i64 %address.local985, !dbg !14
  %address.local1001 = getelementptr i8, ptr addrspace(3) %address.local1000, i64 4352, !dbg !14
  %1007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1001, i32 0) #0, !dbg !14
  %1008 = bitcast <4 x i32> %1007 to <8 x half>, !dbg !14
  %address.local1002 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !14
  %address.local1003 = getelementptr [2 x i8], ptr addrspace(3) %address.local1002, i64 %address.local983, !dbg !14
  %address.local1004 = getelementptr [2 x i8], ptr addrspace(3) %address.local1003, i64 %address.local985, !dbg !14
  %address.local1005 = getelementptr i8, ptr addrspace(3) %address.local1004, i64 4608, !dbg !14
  %1009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1005, i32 0) #0, !dbg !14
  %1010 = bitcast <4 x i32> %1009 to <8 x half>, !dbg !14
  %address.local1006 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local980, !dbg !14
  %address.local1007 = getelementptr [2 x i8], ptr addrspace(3) %address.local1006, i64 %address.local983, !dbg !14
  %address.local1008 = getelementptr [2 x i8], ptr addrspace(3) %address.local1007, i64 %address.local985, !dbg !14
  %address.local1009 = getelementptr i8, ptr addrspace(3) %address.local1008, i64 4864, !dbg !14
  %1011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #0, !dbg !14
  %1012 = bitcast <4 x i32> %1011 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1013 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14461447 = add nsw i64 %230, %address.local961, !dbg !15
  %.idx14461447.tr = trunc i64 %.idx14461447 to i32, !dbg !15
  %dma.buffer.offset1247 = shl i32 %.idx14461447.tr, 1, !dbg !15
  %1014 = insertelement <2 x i64> poison, i64 %1013, i64 0, !dbg !15
  %1015 = bitcast <2 x i64> %1014 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1248 = shufflevector <4 x i32> %1015, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1016 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1247, <4 x i32> %dma.buffer.rsrc1248) #11, !dbg !15
  %1017 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1018 = trunc nuw i64 %230 to i32, !dbg !15
  %.idx14481449.tr = or disjoint i32 %address.local968, %1018, !dbg !15
  %dma.buffer.offset1245 = shl nuw nsw i32 %.idx14481449.tr, 1, !dbg !15
  %1019 = insertelement <2 x i64> poison, i64 %1017, i64 0, !dbg !15
  %1020 = bitcast <2 x i64> %1019 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1246 = shufflevector <4 x i32> %1020, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1021 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1245, <4 x i32> %dma.buffer.rsrc1246) #11, !dbg !15
  %1022 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1022, <4 x float> %989) #9, !dbg !14
  %1024 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1024, <4 x float> %1023) #9, !dbg !14
  %1026 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1026, <4 x float> %1025) #9, !dbg !14
  %1028 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1028, <4 x float> %1027) #9, !dbg !14
  %1030 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1030, <4 x float> %1029) #9, !dbg !14
  %1032 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1032, <4 x float> %1031) #9, !dbg !14
  %1034 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1034, <4 x float> %1033) #9, !dbg !14
  %1036 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1036, <4 x float> %1035) #9, !dbg !14
  %1038 = shufflevector <8 x half> %998, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1038, <4 x float> %991) #9, !dbg !14
  %1040 = shufflevector <8 x half> %1000, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1040, <4 x float> %1039) #9, !dbg !14
  %1042 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1042, <4 x float> %1041) #9, !dbg !14
  %1044 = shufflevector <8 x half> %1004, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1044, <4 x float> %1043) #9, !dbg !14
  %1046 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1046, <4 x float> %1045) #9, !dbg !14
  %1048 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1048, <4 x float> %1047) #9, !dbg !14
  %1050 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1050, <4 x float> %1049) #9, !dbg !14
  %1052 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1052, <4 x float> %1051) #9, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1022, <4 x float> %994) #9, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1024, <4 x float> %1054) #9, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1026, <4 x float> %1055) #9, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1028, <4 x float> %1056) #9, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1030, <4 x float> %1057) #9, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1032, <4 x float> %1058) #9, !dbg !14
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1034, <4 x float> %1059) #9, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1036, <4 x float> %1060) #9, !dbg !14
  %1062 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1038, <4 x float> %996) #9, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1040, <4 x float> %1062) #9, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1042, <4 x float> %1063) #9, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1044, <4 x float> %1064) #9, !dbg !14
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1046, <4 x float> %1065) #9, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1048, <4 x float> %1066) #9, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1050, <4 x float> %1067) #9, !dbg !14
  %1069 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1052, <4 x float> %1068) #9, !dbg !14
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
  %1080 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1081 = bitcast <4 x i32> %1080 to <8 x half>, !dbg !14
  %1082 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1083 = bitcast <4 x i32> %1082 to <8 x half>, !dbg !14
  %1084 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local991, i32 0) #0, !dbg !14
  %1085 = bitcast <4 x i32> %1084 to <8 x half>, !dbg !14
  %1086 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local995, i32 0) #0, !dbg !14
  %1087 = bitcast <4 x i32> %1086 to <8 x half>, !dbg !14
  %1088 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local997, i32 0) #0, !dbg !14
  %1089 = bitcast <4 x i32> %1088 to <8 x half>, !dbg !14
  %1090 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1001, i32 0) #0, !dbg !14
  %1091 = bitcast <4 x i32> %1090 to <8 x half>, !dbg !14
  %1092 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1005, i32 0) #0, !dbg !14
  %1093 = bitcast <4 x i32> %1092 to <8 x half>, !dbg !14
  %1094 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #0, !dbg !14
  %1095 = bitcast <4 x i32> %1094 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1096 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14501451 = add nsw i64 %320, %address.local961, !dbg !15
  %.idx14501451.tr = trunc i64 %.idx14501451 to i32, !dbg !15
  %dma.buffer.offset1243 = shl i32 %.idx14501451.tr, 1, !dbg !15
  %1097 = insertelement <2 x i64> poison, i64 %1096, i64 0, !dbg !15
  %1098 = bitcast <2 x i64> %1097 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1244 = shufflevector <4 x i32> %1098, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1099 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1243, <4 x i32> %dma.buffer.rsrc1244) #11, !dbg !15
  %1100 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1101 = trunc nuw i64 %320 to i32, !dbg !15
  %.idx14521453.tr = or disjoint i32 %address.local968, %1101, !dbg !15
  %dma.buffer.offset1241 = shl nuw nsw i32 %.idx14521453.tr, 1, !dbg !15
  %1102 = insertelement <2 x i64> poison, i64 %1100, i64 0, !dbg !15
  %1103 = bitcast <2 x i64> %1102 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1242 = shufflevector <4 x i32> %1103, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1104 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset1241, <4 x i32> %dma.buffer.rsrc1242) #11, !dbg !15
  %1105 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1105, <4 x float> %1072) #9, !dbg !14
  %1107 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1107, <4 x float> %1106) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1116 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1115, <4 x float> %1114) #9, !dbg !14
  %1117 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1118 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1117, <4 x float> %1116) #9, !dbg !14
  %1119 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1119, <4 x float> %1118) #9, !dbg !14
  %1121 = shufflevector <8 x half> %1081, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1121, <4 x float> %1074) #9, !dbg !14
  %1123 = shufflevector <8 x half> %1083, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1123, <4 x float> %1122) #9, !dbg !14
  %1125 = shufflevector <8 x half> %1085, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1125, <4 x float> %1124) #9, !dbg !14
  %1127 = shufflevector <8 x half> %1087, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1127, <4 x float> %1126) #9, !dbg !14
  %1129 = shufflevector <8 x half> %1089, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1129, <4 x float> %1128) #9, !dbg !14
  %1131 = shufflevector <8 x half> %1091, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1131, <4 x float> %1130) #9, !dbg !14
  %1133 = shufflevector <8 x half> %1093, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1133, <4 x float> %1132) #9, !dbg !14
  %1135 = shufflevector <8 x half> %1095, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1136 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1135, <4 x float> %1134) #9, !dbg !14
  %1137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1105, <4 x float> %1077) #9, !dbg !14
  %1138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1107, <4 x float> %1137) #9, !dbg !14
  %1139 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1109, <4 x float> %1138) #9, !dbg !14
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1111, <4 x float> %1139) #9, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1113, <4 x float> %1140) #9, !dbg !14
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1115, <4 x float> %1141) #9, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1117, <4 x float> %1142) #9, !dbg !14
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1119, <4 x float> %1143) #9, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1121, <4 x float> %1079) #9, !dbg !14
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1123, <4 x float> %1145) #9, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1125, <4 x float> %1146) #9, !dbg !14
  %1148 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1127, <4 x float> %1147) #9, !dbg !14
  %1149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1129, <4 x float> %1148) #9, !dbg !14
  %1150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1131, <4 x float> %1149) #9, !dbg !14
  %1151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1133, <4 x float> %1150) #9, !dbg !14
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1135, <4 x float> %1151) #9, !dbg !14
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
  %1163 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1164 = bitcast <4 x i32> %1163 to <8 x half>, !dbg !14
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !14
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local991, i32 0) #0, !dbg !14
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local995, i32 0) #0, !dbg !14
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local997, i32 0) #0, !dbg !14
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1001, i32 0) #0, !dbg !14
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1005, i32 0) #0, !dbg !14
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #0, !dbg !14
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1179 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14541455 = add nsw i64 %410, %address.local961, !dbg !15
  %.idx14541455.tr = trunc i64 %.idx14541455 to i32, !dbg !15
  %dma.buffer.offset1239 = shl i32 %.idx14541455.tr, 1, !dbg !15
  %1180 = insertelement <2 x i64> poison, i64 %1179, i64 0, !dbg !15
  %1181 = bitcast <2 x i64> %1180 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1240 = shufflevector <4 x i32> %1181, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1182 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %86, i32 %dma.buffer.offset1239, <4 x i32> %dma.buffer.rsrc1240) #11, !dbg !15
  %1183 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.tr = trunc nuw i64 %410 to i32, !dbg !15
  %1184 = or disjoint i32 %address.local968, %.tr, !dbg !15
  %dma.buffer.offset = shl nuw nsw i32 %1184, 1, !dbg !15
  %1185 = insertelement <2 x i64> poison, i64 %1183, i64 0, !dbg !15
  %1186 = bitcast <2 x i64> %1185 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc = shufflevector <4 x i32> %1186, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %1187 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %87, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #11, !dbg !15
  %1188 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1188, <4 x float> %1155) #9, !dbg !14
  %1190 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1190, <4 x float> %1189) #9, !dbg !14
  %1192 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1192, <4 x float> %1191) #9, !dbg !14
  %1194 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1195 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1194, <4 x float> %1193) #9, !dbg !14
  %1196 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1197 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1196, <4 x float> %1195) #9, !dbg !14
  %1198 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1199 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1198, <4 x float> %1197) #9, !dbg !14
  %1200 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1201 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1200, <4 x float> %1199) #9, !dbg !14
  %1202 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1202, <4 x float> %1201) #9, !dbg !14
  %1204 = shufflevector <8 x half> %1164, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1204, <4 x float> %1157) #9, !dbg !14
  %1206 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1206, <4 x float> %1205) #9, !dbg !14
  %1208 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1208, <4 x float> %1207) #9, !dbg !14
  %1210 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1210, <4 x float> %1209) #9, !dbg !14
  %1212 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1212, <4 x float> %1211) #9, !dbg !14
  %1214 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1214, <4 x float> %1213) #9, !dbg !14
  %1216 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1216, <4 x float> %1215) #9, !dbg !14
  %1218 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1218, <4 x float> %1217) #9, !dbg !14
  %1220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1188, <4 x float> %1160) #9, !dbg !14
  %1221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1190, <4 x float> %1220) #9, !dbg !14
  %1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1192, <4 x float> %1221) #9, !dbg !14
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1194, <4 x float> %1222) #9, !dbg !14
  %1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1196, <4 x float> %1223) #9, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1198, <4 x float> %1224) #9, !dbg !14
  %1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1200, <4 x float> %1225) #9, !dbg !14
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1202, <4 x float> %1226) #9, !dbg !14
  %1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1204, <4 x float> %1162) #9, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1206, <4 x float> %1228) #9, !dbg !14
  %1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1208, <4 x float> %1229) #9, !dbg !14
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1210, <4 x float> %1230) #9, !dbg !14
  %1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1212, <4 x float> %1231) #9, !dbg !14
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1214, <4 x float> %1232) #9, !dbg !14
  %1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1216, <4 x float> %1233) #9, !dbg !14
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1218, <4 x float> %1234) #9, !dbg !14
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
  %1246 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14
  %1247 = bitcast <4 x i32> %1246 to <8 x half>, !dbg !14
  %1248 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local987, i32 0) #0, !dbg !14
  %1249 = bitcast <4 x i32> %1248 to <8 x half>, !dbg !14
  %1250 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local991, i32 0) #0, !dbg !14
  %1251 = bitcast <4 x i32> %1250 to <8 x half>, !dbg !14
  %1252 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local995, i32 0) #0, !dbg !14
  %1253 = bitcast <4 x i32> %1252 to <8 x half>, !dbg !14
  %1254 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local997, i32 0) #0, !dbg !14
  %1255 = bitcast <4 x i32> %1254 to <8 x half>, !dbg !14
  %1256 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1001, i32 0) #0, !dbg !14
  %1257 = bitcast <4 x i32> %1256 to <8 x half>, !dbg !14
  %1258 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1005, i32 0) #0, !dbg !14
  %1259 = bitcast <4 x i32> %1258 to <8 x half>, !dbg !14
  %1260 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1009, i32 0) #0, !dbg !14
  %1261 = bitcast <4 x i32> %1260 to <8 x half>, !dbg !14
  %1262 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1262, <4 x float> %1238) #9, !dbg !14
  %1264 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1264, <4 x float> %1263) #9, !dbg !14
  %1266 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1266, <4 x float> %1265) #9, !dbg !14
  %1268 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1268, <4 x float> %1267) #9, !dbg !14
  %1270 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1270, <4 x float> %1269) #9, !dbg !14
  %1272 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1272, <4 x float> %1271) #9, !dbg !14
  %1274 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1274, <4 x float> %1273) #9, !dbg !14
  %1276 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1276, <4 x float> %1275) #9, !dbg !14
  %1278 = shufflevector <8 x half> %1247, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1278, <4 x float> %1240) #9, !dbg !14
  %1280 = shufflevector <8 x half> %1249, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1280, <4 x float> %1279) #9, !dbg !14
  %1282 = shufflevector <8 x half> %1251, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1282, <4 x float> %1281) #9, !dbg !14
  %1284 = shufflevector <8 x half> %1253, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1284, <4 x float> %1283) #9, !dbg !14
  %1286 = shufflevector <8 x half> %1255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1286, <4 x float> %1285) #9, !dbg !14
  %1288 = shufflevector <8 x half> %1257, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1288, <4 x float> %1287) #9, !dbg !14
  %1290 = shufflevector <8 x half> %1259, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1290, <4 x float> %1289) #9, !dbg !14
  %1292 = shufflevector <8 x half> %1261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1292, <4 x float> %1291) #9, !dbg !14
  %1294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1262, <4 x float> %1243) #9, !dbg !14
  %1295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1264, <4 x float> %1294) #9, !dbg !14
  %1296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1266, <4 x float> %1295) #9, !dbg !14
  %1297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1268, <4 x float> %1296) #9, !dbg !14
  %1298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1270, <4 x float> %1297) #9, !dbg !14
  %1299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1272, <4 x float> %1298) #9, !dbg !14
  %1300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1274, <4 x float> %1299) #9, !dbg !14
  %1301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1276, <4 x float> %1300) #9, !dbg !14
  %1302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1278, <4 x float> %1245) #9, !dbg !14
  %1303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1280, <4 x float> %1302) #9, !dbg !14
  %1304 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1282, <4 x float> %1303) #9, !dbg !14
  %1305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1284, <4 x float> %1304) #9, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1286, <4 x float> %1305) #9, !dbg !14
  %1307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1288, <4 x float> %1306) #9, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1290, <4 x float> %1307) #9, !dbg !14
  %1309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1292, <4 x float> %1308) #9, !dbg !14
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
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1333, !dbg !17
  %normalized = fmul <4 x float> %1335, %row.reciprocal, !dbg !17
  %1336 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1337 = extractelement <4 x half> %1336, i64 3, !dbg !18
  %1338 = extractelement <4 x half> %1336, i64 2, !dbg !18
  %1339 = extractelement <4 x half> %1336, i64 1, !dbg !18
  %1340 = extractelement <4 x half> %1336, i64 0, !dbg !18
  %1341 = shufflevector <8 x float> %1334, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized65 = fmul <4 x float> %1341, %row.reciprocal, !dbg !17
  %1342 = fptrunc <4 x float> %normalized65 to <4 x half>, !dbg !18
  %1343 = extractelement <4 x half> %1342, i64 3, !dbg !18
  %1344 = extractelement <4 x half> %1342, i64 2, !dbg !18
  %1345 = extractelement <4 x half> %1342, i64 1, !dbg !18
  %1346 = extractelement <4 x half> %1342, i64 0, !dbg !18
  %1347 = extractvalue [2 x <8 x float>] %126, 0, !dbg !17
  %1348 = shufflevector <8 x float> %1347, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal66 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1331, !dbg !17
  %normalized67 = fmul <4 x float> %1348, %row.reciprocal66, !dbg !17
  %1349 = fptrunc <4 x float> %normalized67 to <4 x half>, !dbg !18
  %1350 = extractelement <4 x half> %1349, i64 3, !dbg !18
  %1351 = extractelement <4 x half> %1349, i64 2, !dbg !18
  %1352 = extractelement <4 x half> %1349, i64 1, !dbg !18
  %1353 = extractelement <4 x half> %1349, i64 0, !dbg !18
  %1354 = shufflevector <8 x float> %1347, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized68 = fmul <4 x float> %1354, %row.reciprocal66, !dbg !17
  %1355 = fptrunc <4 x float> %normalized68 to <4 x half>, !dbg !18
  %1356 = extractelement <4 x half> %1355, i64 3, !dbg !18
  %1357 = extractelement <4 x half> %1355, i64 2, !dbg !18
  %1358 = extractelement <4 x half> %1355, i64 1, !dbg !18
  %1359 = extractelement <4 x half> %1355, i64 0, !dbg !18
  %1360 = shl nuw nsw i64 %26, 19, !dbg !19
  %address.thread1016 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1017 = zext nneg i32 %address.thread1016 to i64, !dbg !13
  %address.local1018 = add nsw i64 %27, %address.local1017, !dbg !19
  %1361 = lshr i64 %address.local1017, 2, !dbg !19
  %address.local1020 = and i64 %1361, 48, !dbg !19
  %address.local1021 = add nsw i64 %address.local1018, %address.local1020, !dbg !19
  %address.local1022 = and i64 %address.local1017, 240, !dbg !19
  %address.local1023 = sub nsw i64 %address.local1021, %address.local1022, !dbg !19
  %address.local1024 = add nsw i64 %address.local1023, %28, !dbg !19
  %address.local1025 = shl nuw nsw i64 %address.local1024, 7, !dbg !19
  %address.local1026 = add nuw nsw i64 %1360, %address.local1025, !dbg !19
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1028 = lshr i64 %address.local1017, 4, !dbg !13
  %address.local1029 = and i64 %address.local1028, 3, !dbg !13
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1027, i64 %address.local1029, !dbg !19
  store half %1359, ptr addrspace(1) %address.local1030, align 2, !dbg !19
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1032 = or disjoint i64 %address.local1029, 4, !dbg !19
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1032, !dbg !19
  store half %1358, ptr addrspace(1) %address.local1033, align 2, !dbg !19
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1035 = or disjoint i64 %address.local1029, 8, !dbg !19
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1034, i64 %address.local1035, !dbg !19
  store half %1357, ptr addrspace(1) %address.local1036, align 2, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1038 = or i64 %address.local1028, 12, !dbg !19
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1037, i64 %address.local1038, !dbg !19
  store half %1356, ptr addrspace(1) %address.local1039, align 2, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1041 = or disjoint i64 %address.local1029, 16, !dbg !19
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1040, i64 %address.local1041, !dbg !19
  store half %1353, ptr addrspace(1) %address.local1042, align 2, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1044 = or disjoint i64 %address.local1029, 20, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1043, i64 %address.local1044, !dbg !19
  store half %1352, ptr addrspace(1) %address.local1045, align 2, !dbg !19
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1047 = or disjoint i64 %address.local1029, 24, !dbg !19
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1046, i64 %address.local1047, !dbg !19
  store half %1351, ptr addrspace(1) %address.local1048, align 2, !dbg !19
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1026, !dbg !19
  %address.local1050 = or i64 %address.local1028, 28, !dbg !19
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1049, i64 %address.local1050, !dbg !19
  store half %1350, ptr addrspace(1) %address.local1051, align 2, !dbg !19
  %address.local1052 = shl nsw i64 %address.local1024, 7, !dbg !19
  %address.local1053 = add i64 %address.local1052, 8192, !dbg !19
  %address.local1054 = add nuw nsw i64 %1360, %address.local1053, !dbg !19
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1055, i64 %address.local1029, !dbg !19
  store half %1346, ptr addrspace(1) %address.local1056, align 2, !dbg !19
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1057, i64 %address.local1032, !dbg !19
  store half %1345, ptr addrspace(1) %address.local1058, align 2, !dbg !19
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1059, i64 %address.local1035, !dbg !19
  store half %1344, ptr addrspace(1) %address.local1060, align 2, !dbg !19
  %address.local1061 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1061, i64 %address.local1038, !dbg !19
  store half %1343, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1063, i64 %address.local1041, !dbg !19
  store half %1340, ptr addrspace(1) %address.local1064, align 2, !dbg !19
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1065, i64 %address.local1044, !dbg !19
  store half %1339, ptr addrspace(1) %address.local1066, align 2, !dbg !19
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1067, i64 %address.local1047, !dbg !19
  store half %1338, ptr addrspace(1) %address.local1068, align 2, !dbg !19
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1054, !dbg !19
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1069, i64 %address.local1050, !dbg !19
  store half %1337, ptr addrspace(1) %address.local1070, align 2, !dbg !19
  %1362 = extractvalue [2 x <8 x float>] %129, 1, !dbg !17
  %1363 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal69 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1333, !dbg !17
  %normalized70 = fmul <4 x float> %1363, %row.reciprocal69, !dbg !17
  %1364 = fptrunc <4 x float> %normalized70 to <4 x half>, !dbg !18
  %1365 = extractelement <4 x half> %1364, i64 3, !dbg !18
  %1366 = extractelement <4 x half> %1364, i64 2, !dbg !18
  %1367 = extractelement <4 x half> %1364, i64 1, !dbg !18
  %1368 = extractelement <4 x half> %1364, i64 0, !dbg !18
  %1369 = shufflevector <8 x float> %1362, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized71 = fmul <4 x float> %1369, %row.reciprocal69, !dbg !17
  %1370 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !18
  %1371 = extractelement <4 x half> %1370, i64 3, !dbg !18
  %1372 = extractelement <4 x half> %1370, i64 2, !dbg !18
  %1373 = extractelement <4 x half> %1370, i64 1, !dbg !18
  %1374 = extractelement <4 x half> %1370, i64 0, !dbg !18
  %1375 = extractvalue [2 x <8 x float>] %129, 0, !dbg !17
  %1376 = shufflevector <8 x float> %1375, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1331, !dbg !17
  %normalized73 = fmul <4 x float> %1376, %row.reciprocal72, !dbg !17
  %1377 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !18
  %1378 = extractelement <4 x half> %1377, i64 3, !dbg !18
  %1379 = extractelement <4 x half> %1377, i64 2, !dbg !18
  %1380 = extractelement <4 x half> %1377, i64 1, !dbg !18
  %1381 = extractelement <4 x half> %1377, i64 0, !dbg !18
  %1382 = shufflevector <8 x float> %1375, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized74 = fmul <4 x float> %1382, %row.reciprocal72, !dbg !17
  %1383 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !18
  %1384 = extractelement <4 x half> %1383, i64 3, !dbg !18
  %1385 = extractelement <4 x half> %1383, i64 2, !dbg !18
  %1386 = extractelement <4 x half> %1383, i64 1, !dbg !18
  %1387 = extractelement <4 x half> %1383, i64 0, !dbg !18
  %1388 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1071 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1072 = zext nneg i32 %address.thread1071 to i64, !dbg !13
  %address.local1073 = add nsw i64 %27, %address.local1072, !dbg !19
  %1389 = lshr i64 %address.local1072, 2, !dbg !19
  %address.local1075 = and i64 %1389, 48, !dbg !19
  %address.local1076 = add nsw i64 %address.local1073, %address.local1075, !dbg !19
  %address.local1077 = and i64 %address.local1072, 240, !dbg !19
  %address.local1078 = sub nsw i64 %address.local1076, %address.local1077, !dbg !19
  %address.local1079 = add nsw i64 %address.local1078, %28, !dbg !19
  %address.local1080 = shl nuw nsw i64 %address.local1079, 7, !dbg !19
  %address.local1081 = add nuw nsw i64 %1388, %address.local1080, !dbg !19
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1083 = lshr i64 %address.local1072, 4, !dbg !13
  %address.local1084 = and i64 %address.local1083, 3, !dbg !13
  %address.local1085 = or disjoint i64 %address.local1084, 32, !dbg !19
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1082, i64 %address.local1085, !dbg !19
  store half %1387, ptr addrspace(1) %address.local1086, align 2, !dbg !19
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1088 = or disjoint i64 %address.local1084, 36, !dbg !19
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1087, i64 %address.local1088, !dbg !19
  store half %1386, ptr addrspace(1) %address.local1089, align 2, !dbg !19
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1091 = or disjoint i64 %address.local1084, 40, !dbg !19
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1090, i64 %address.local1091, !dbg !19
  store half %1385, ptr addrspace(1) %address.local1092, align 2, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1094 = or i64 %address.local1083, 44, !dbg !19
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1093, i64 %address.local1094, !dbg !19
  store half %1384, ptr addrspace(1) %address.local1095, align 2, !dbg !19
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1097 = or disjoint i64 %address.local1084, 48, !dbg !19
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1096, i64 %address.local1097, !dbg !19
  store half %1381, ptr addrspace(1) %address.local1098, align 2, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1100 = or disjoint i64 %address.local1084, 52, !dbg !19
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1099, i64 %address.local1100, !dbg !19
  store half %1380, ptr addrspace(1) %address.local1101, align 2, !dbg !19
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1103 = or disjoint i64 %address.local1084, 56, !dbg !19
  %address.local1104 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1102, i64 %address.local1103, !dbg !19
  store half %1379, ptr addrspace(1) %address.local1104, align 2, !dbg !19
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1081, !dbg !19
  %address.local1106 = or i64 %address.local1083, 60, !dbg !19
  %address.local1107 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1105, i64 %address.local1106, !dbg !19
  store half %1378, ptr addrspace(1) %address.local1107, align 2, !dbg !19
  %address.local1108 = shl nsw i64 %address.local1079, 7, !dbg !19
  %address.local1109 = add i64 %address.local1108, 8192, !dbg !19
  %address.local1110 = add nuw nsw i64 %1388, %address.local1109, !dbg !19
  %address.local1111 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1111, i64 %address.local1085, !dbg !19
  store half %1374, ptr addrspace(1) %address.local1112, align 2, !dbg !19
  %address.local1113 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1113, i64 %address.local1088, !dbg !19
  store half %1373, ptr addrspace(1) %address.local1114, align 2, !dbg !19
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1115, i64 %address.local1091, !dbg !19
  store half %1372, ptr addrspace(1) %address.local1116, align 2, !dbg !19
  %address.local1117 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1117, i64 %address.local1094, !dbg !19
  store half %1371, ptr addrspace(1) %address.local1118, align 2, !dbg !19
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1119, i64 %address.local1097, !dbg !19
  store half %1368, ptr addrspace(1) %address.local1120, align 2, !dbg !19
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1121, i64 %address.local1100, !dbg !19
  store half %1367, ptr addrspace(1) %address.local1122, align 2, !dbg !19
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1123, i64 %address.local1103, !dbg !19
  store half %1366, ptr addrspace(1) %address.local1124, align 2, !dbg !19
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1110, !dbg !19
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1125, i64 %address.local1106, !dbg !19
  store half %1365, ptr addrspace(1) %address.local1126, align 2, !dbg !19
  %1390 = extractvalue [2 x <8 x float>] %130, 1, !dbg !17
  %1391 = shufflevector <8 x float> %1390, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1333, !dbg !17
  %normalized76 = fmul <4 x float> %1391, %row.reciprocal75, !dbg !17
  %1392 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !18
  %1393 = extractelement <4 x half> %1392, i64 3, !dbg !18
  %1394 = extractelement <4 x half> %1392, i64 2, !dbg !18
  %1395 = extractelement <4 x half> %1392, i64 1, !dbg !18
  %1396 = extractelement <4 x half> %1392, i64 0, !dbg !18
  %1397 = shufflevector <8 x float> %1390, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized77 = fmul <4 x float> %1397, %row.reciprocal75, !dbg !17
  %1398 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !18
  %1399 = extractelement <4 x half> %1398, i64 3, !dbg !18
  %1400 = extractelement <4 x half> %1398, i64 2, !dbg !18
  %1401 = extractelement <4 x half> %1398, i64 1, !dbg !18
  %1402 = extractelement <4 x half> %1398, i64 0, !dbg !18
  %1403 = extractvalue [2 x <8 x float>] %130, 0, !dbg !17
  %1404 = shufflevector <8 x float> %1403, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1331, !dbg !17
  %normalized79 = fmul <4 x float> %1404, %row.reciprocal78, !dbg !17
  %1405 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !18
  %1406 = extractelement <4 x half> %1405, i64 3, !dbg !18
  %1407 = extractelement <4 x half> %1405, i64 2, !dbg !18
  %1408 = extractelement <4 x half> %1405, i64 1, !dbg !18
  %1409 = extractelement <4 x half> %1405, i64 0, !dbg !18
  %1410 = shufflevector <8 x float> %1403, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized80 = fmul <4 x float> %1410, %row.reciprocal78, !dbg !17
  %1411 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !18
  %1412 = extractelement <4 x half> %1411, i64 3, !dbg !18
  %1413 = extractelement <4 x half> %1411, i64 2, !dbg !18
  %1414 = extractelement <4 x half> %1411, i64 1, !dbg !18
  %1415 = extractelement <4 x half> %1411, i64 0, !dbg !18
  %1416 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1127 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1128 = zext nneg i32 %address.thread1127 to i64, !dbg !13
  %address.local1129 = add nsw i64 %27, %address.local1128, !dbg !19
  %1417 = lshr i64 %address.local1128, 2, !dbg !19
  %address.local1131 = and i64 %1417, 48, !dbg !19
  %address.local1132 = add nsw i64 %address.local1129, %address.local1131, !dbg !19
  %address.local1133 = and i64 %address.local1128, 240, !dbg !19
  %address.local1134 = sub nsw i64 %address.local1132, %address.local1133, !dbg !19
  %address.local1135 = add nsw i64 %address.local1134, %28, !dbg !19
  %address.local1136 = shl nuw nsw i64 %address.local1135, 7, !dbg !19
  %address.local1137 = add nuw nsw i64 %1416, %address.local1136, !dbg !19
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1139 = lshr i64 %address.local1128, 4, !dbg !13
  %address.local1140 = and i64 %address.local1139, 3, !dbg !13
  %address.local1141 = or disjoint i64 %address.local1140, 64, !dbg !19
  %address.local1142 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1138, i64 %address.local1141, !dbg !19
  store half %1415, ptr addrspace(1) %address.local1142, align 2, !dbg !19
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1144 = or disjoint i64 %address.local1140, 68, !dbg !19
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1143, i64 %address.local1144, !dbg !19
  store half %1414, ptr addrspace(1) %address.local1145, align 2, !dbg !19
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1147 = or disjoint i64 %address.local1140, 72, !dbg !19
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1146, i64 %address.local1147, !dbg !19
  store half %1413, ptr addrspace(1) %address.local1148, align 2, !dbg !19
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1150 = or i64 %address.local1139, 76, !dbg !19
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1149, i64 %address.local1150, !dbg !19
  store half %1412, ptr addrspace(1) %address.local1151, align 2, !dbg !19
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1153 = or disjoint i64 %address.local1140, 80, !dbg !19
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1152, i64 %address.local1153, !dbg !19
  store half %1409, ptr addrspace(1) %address.local1154, align 2, !dbg !19
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1156 = or disjoint i64 %address.local1140, 84, !dbg !19
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1155, i64 %address.local1156, !dbg !19
  store half %1408, ptr addrspace(1) %address.local1157, align 2, !dbg !19
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1159 = or disjoint i64 %address.local1140, 88, !dbg !19
  %address.local1160 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1158, i64 %address.local1159, !dbg !19
  store half %1407, ptr addrspace(1) %address.local1160, align 2, !dbg !19
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1137, !dbg !19
  %address.local1162 = or i64 %address.local1139, 92, !dbg !19
  %address.local1163 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1161, i64 %address.local1162, !dbg !19
  store half %1406, ptr addrspace(1) %address.local1163, align 2, !dbg !19
  %address.local1164 = shl nsw i64 %address.local1135, 7, !dbg !19
  %address.local1165 = add i64 %address.local1164, 8192, !dbg !19
  %address.local1166 = add nuw nsw i64 %1416, %address.local1165, !dbg !19
  %address.local1167 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1167, i64 %address.local1141, !dbg !19
  store half %1402, ptr addrspace(1) %address.local1168, align 2, !dbg !19
  %address.local1169 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1169, i64 %address.local1144, !dbg !19
  store half %1401, ptr addrspace(1) %address.local1170, align 2, !dbg !19
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1171, i64 %address.local1147, !dbg !19
  store half %1400, ptr addrspace(1) %address.local1172, align 2, !dbg !19
  %address.local1173 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1173, i64 %address.local1150, !dbg !19
  store half %1399, ptr addrspace(1) %address.local1174, align 2, !dbg !19
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1175, i64 %address.local1153, !dbg !19
  store half %1396, ptr addrspace(1) %address.local1176, align 2, !dbg !19
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1177, i64 %address.local1156, !dbg !19
  store half %1395, ptr addrspace(1) %address.local1178, align 2, !dbg !19
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1179, i64 %address.local1159, !dbg !19
  store half %1394, ptr addrspace(1) %address.local1180, align 2, !dbg !19
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1166, !dbg !19
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1181, i64 %address.local1162, !dbg !19
  store half %1393, ptr addrspace(1) %address.local1182, align 2, !dbg !19
  %1418 = extractvalue [2 x <8 x float>] %131, 1, !dbg !17
  %1419 = shufflevector <8 x float> %1418, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1333, !dbg !17
  %normalized82 = fmul <4 x float> %1419, %row.reciprocal81, !dbg !17
  %1420 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !18
  %1421 = extractelement <4 x half> %1420, i64 3, !dbg !18
  %1422 = extractelement <4 x half> %1420, i64 2, !dbg !18
  %1423 = extractelement <4 x half> %1420, i64 1, !dbg !18
  %1424 = extractelement <4 x half> %1420, i64 0, !dbg !18
  %1425 = shufflevector <8 x float> %1418, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized83 = fmul <4 x float> %1425, %row.reciprocal81, !dbg !17
  %1426 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !18
  %1427 = extractelement <4 x half> %1426, i64 3, !dbg !18
  %1428 = extractelement <4 x half> %1426, i64 2, !dbg !18
  %1429 = extractelement <4 x half> %1426, i64 1, !dbg !18
  %1430 = extractelement <4 x half> %1426, i64 0, !dbg !18
  %1431 = extractvalue [2 x <8 x float>] %131, 0, !dbg !17
  %1432 = shufflevector <8 x float> %1431, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1331, !dbg !17
  %normalized85 = fmul <4 x float> %1432, %row.reciprocal84, !dbg !17
  %1433 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !18
  %1434 = extractelement <4 x half> %1433, i64 3, !dbg !18
  %1435 = extractelement <4 x half> %1433, i64 2, !dbg !18
  %1436 = extractelement <4 x half> %1433, i64 1, !dbg !18
  %1437 = extractelement <4 x half> %1433, i64 0, !dbg !18
  %1438 = shufflevector <8 x float> %1431, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized86 = fmul <4 x float> %1438, %row.reciprocal84, !dbg !17
  %1439 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !18
  %1440 = extractelement <4 x half> %1439, i64 3, !dbg !18
  %1441 = extractelement <4 x half> %1439, i64 2, !dbg !18
  %1442 = extractelement <4 x half> %1439, i64 1, !dbg !18
  %1443 = extractelement <4 x half> %1439, i64 0, !dbg !18
  %1444 = shl nuw nsw i64 %35, 19, !dbg !19
  %address.thread1183 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local1184 = zext nneg i32 %address.thread1183 to i64, !dbg !13
  %address.local1185 = add nsw i64 %27, %address.local1184, !dbg !19
  %1445 = lshr i64 %address.local1184, 2, !dbg !19
  %address.local1187 = and i64 %1445, 48, !dbg !19
  %address.local1188 = add nsw i64 %address.local1185, %address.local1187, !dbg !19
  %address.local1189 = and i64 %address.local1184, 240, !dbg !19
  %address.local1190 = sub nsw i64 %address.local1188, %address.local1189, !dbg !19
  %address.local1191 = add nsw i64 %address.local1190, %28, !dbg !19
  %address.local1192 = shl nuw nsw i64 %address.local1191, 7, !dbg !19
  %address.local1193 = add nuw nsw i64 %1444, %address.local1192, !dbg !19
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1195 = lshr i64 %address.local1184, 4, !dbg !13
  %address.local1196 = and i64 %address.local1195, 3, !dbg !13
  %address.local1197 = or disjoint i64 %address.local1196, 96, !dbg !19
  %address.local1198 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1194, i64 %address.local1197, !dbg !19
  store half %1443, ptr addrspace(1) %address.local1198, align 2, !dbg !19
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1200 = or disjoint i64 %address.local1196, 100, !dbg !19
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1199, i64 %address.local1200, !dbg !19
  store half %1442, ptr addrspace(1) %address.local1201, align 2, !dbg !19
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1203 = or disjoint i64 %address.local1196, 104, !dbg !19
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1202, i64 %address.local1203, !dbg !19
  store half %1441, ptr addrspace(1) %address.local1204, align 2, !dbg !19
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1206 = or i64 %address.local1195, 108, !dbg !19
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1205, i64 %address.local1206, !dbg !19
  store half %1440, ptr addrspace(1) %address.local1207, align 2, !dbg !19
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1209 = or disjoint i64 %address.local1196, 112, !dbg !19
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1208, i64 %address.local1209, !dbg !19
  store half %1437, ptr addrspace(1) %address.local1210, align 2, !dbg !19
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1212 = or disjoint i64 %address.local1196, 116, !dbg !19
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1211, i64 %address.local1212, !dbg !19
  store half %1436, ptr addrspace(1) %address.local1213, align 2, !dbg !19
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1215 = or disjoint i64 %address.local1196, 120, !dbg !19
  %address.local1216 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1214, i64 %address.local1215, !dbg !19
  store half %1435, ptr addrspace(1) %address.local1216, align 2, !dbg !19
  %address.local1217 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1193, !dbg !19
  %address.local1218 = or i64 %address.local1195, 124, !dbg !19
  %address.local1219 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1217, i64 %address.local1218, !dbg !19
  store half %1434, ptr addrspace(1) %address.local1219, align 2, !dbg !19
  %address.local1220 = shl nsw i64 %address.local1191, 7, !dbg !19
  %address.local1221 = add i64 %address.local1220, 8192, !dbg !19
  %address.local1222 = add nuw nsw i64 %1444, %address.local1221, !dbg !19
  %address.local1223 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1224 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1223, i64 %address.local1197, !dbg !19
  store half %1430, ptr addrspace(1) %address.local1224, align 2, !dbg !19
  %address.local1225 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1226 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1225, i64 %address.local1200, !dbg !19
  store half %1429, ptr addrspace(1) %address.local1226, align 2, !dbg !19
  %address.local1227 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1228 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1227, i64 %address.local1203, !dbg !19
  store half %1428, ptr addrspace(1) %address.local1228, align 2, !dbg !19
  %address.local1229 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1230 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1229, i64 %address.local1206, !dbg !19
  store half %1427, ptr addrspace(1) %address.local1230, align 2, !dbg !19
  %address.local1231 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1232 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1231, i64 %address.local1209, !dbg !19
  store half %1424, ptr addrspace(1) %address.local1232, align 2, !dbg !19
  %address.local1233 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1234 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1233, i64 %address.local1212, !dbg !19
  store half %1423, ptr addrspace(1) %address.local1234, align 2, !dbg !19
  %address.local1235 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1236 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1235, i64 %address.local1215, !dbg !19
  store half %1422, ptr addrspace(1) %address.local1236, align 2, !dbg !19
  %address.local1237 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1222, !dbg !19
  %address.local1238 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1237, i64 %address.local1218, !dbg !19
  store half %1421, ptr addrspace(1) %address.local1238, align 2, !dbg !19
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
