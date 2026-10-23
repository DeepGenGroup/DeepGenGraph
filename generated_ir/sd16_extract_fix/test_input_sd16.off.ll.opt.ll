; ModuleID = 'generated_ir/sd16_extract_fix/test_input_sd16.off.ll'
source_filename = "3rd/deepgengraph/test/test_input_sd16.mlir"

@async_lds_staging_1 = private addrspace(3) global [2048 x half] undef, align 16
@async_lds_staging_0 = private addrspace(3) global [2048 x half] undef, align 16
@shm_0 = addrspace(3) global [8192 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float) local_unnamed_addr

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = zext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %6, 7, !dbg !8
  %.lobit = ashr i32 %5, 31, !dbg !9
  %10 = xor i32 %.lobit, %5, !dbg !9
  %11 = sdiv i32 %10, 32, !dbg !9
  %12 = xor i32 %11, %.lobit, !dbg !9
  %13 = sext i32 %12 to i64, !dbg !9
  %14 = add nsw i64 %13, %8, !dbg !9
  %15 = shl nsw i64 %13, 12, !dbg !9
  %16 = sub nsw i64 %9, %15, !dbg !9
  %17 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %18 = zext nneg i32 %17 to i64, !dbg !10
  %19 = and i32 %17, 16, !dbg !12
  %.not = icmp eq i32 %19, 0, !dbg !12
  %20 = and i32 %17, 32, !dbg !12
  %.not155 = icmp eq i32 %20, 0, !dbg !12
  %21 = add nsw i64 %9, %18, !dbg !13
  %22 = lshr i32 %17, 2, !dbg !13
  %23 = and i32 %22, 48, !dbg !13
  %index.compact2 = zext nneg i32 %23 to i64, !dbg !13
  %24 = and i32 %17, 240, !dbg !13
  %25 = sub nsw i32 0, %24, !dbg !13
  %index.compact1 = sext i32 %25 to i64, !dbg !13
  %26 = add nsw i64 %21, %index.compact1, !dbg !13
  %27 = add nsw i64 %26, %index.compact2, !dbg !13
  %28 = and i32 %22, 12, !dbg !13
  %29 = add nsw i64 %27, 1, !dbg !14
  %30 = add nsw i64 %27, 65, !dbg !14
  %.not16111 = icmp slt i32 %5, 0
  %31 = shl nsw i64 %14, 20
  %.idx234 = and i64 %31, 32505856
  %32 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx234
  %.idx235 = shl nsw i64 %16, 8
  %33 = getelementptr i8, ptr addrspace(1) %32, i64 %.idx235
  %34 = lshr i64 %29, 32
  %compare.high = trunc nuw i64 %34 to i32
  %compare.low.base = trunc i64 %29 to i32
  %35 = lshr i64 %30, 32
  %compare.high853 = trunc nuw i64 %35 to i32
  %compare.low.base857 = trunc i64 %30 to i32
  %36 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i64 %.idx234
  %.mask = shl nuw nsw i32 %17, 1, !dbg !15
  %37 = and i32 %.mask, 6, !dbg !15
  %38 = zext nneg i32 %37 to i64, !dbg !15
  %39 = lshr i64 %18, 2, !dbg !15
  %.lobit42 = and i64 %39, 1, !dbg !15
  %40 = and i64 %18, 120, !dbg !15
  %41 = lshr i64 %18, 5, !dbg !15
  %.lobit43 = and i64 %41, 1, !dbg !15
  %42 = shl nuw nsw i64 %18, 1, !dbg !15
  %43 = and i64 %42, 4, !dbg !15
  %44 = and i64 %18, 192, !dbg !15
  %.mask45 = shl nuw nsw i32 %17, 5, !dbg !15
  %45 = and i32 %.mask45, 32, !dbg !15
  %46 = zext nneg i32 %45 to i64, !dbg !15
  %47 = or disjoint i64 %40, %38
  %invariant.gep = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %.lobit42
  %.idx239 = shl nuw nsw i64 %47, 5
  %invariant.gep71 = getelementptr inbounds nuw i8, ptr addrspace(3) %invariant.gep, i64 %.idx239
  %.idx47 = shl nuw nsw i64 %.lobit43, 4
  %48 = getelementptr inbounds nuw i8, ptr addrspace(3) @async_lds_staging_1, i64 %.idx47
  %49 = insertelement <4 x i1> poison, i1 %.not155, i64 0, !dbg !16
  %50 = shufflevector <4 x i1> %49, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  br label %51, !dbg !15

51:                                               ; preds = %4, %._crit_edge
  %52 = phi i64 [ 0, %4 ], [ %1423, %._crit_edge ]
  %53 = shl nuw nsw i64 %52, 5, !dbg !15
  br i1 %.not16111, label %.._crit_edge_crit_edge, label %.lr.ph, !dbg !16

.._crit_edge_crit_edge:                           ; preds = %51
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %.pre32 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !16
  %.pre34 = add nuw nsw i32 %.pre32, 64, !dbg !16
  %.pre36 = xor i32 %.pre32, 16, !dbg !16
  %.pre38 = and i32 %.pre34, 192, !dbg !16
  %.pre40 = xor i32 %.pre32, 32, !dbg !16
  br label %._crit_edge, !dbg !16

.lr.ph:                                           ; preds = %51, %1196
  %54 = phi [2 x <4 x float>] [ %1251, %1196 ], [ zeroinitializer, %51 ]
  %55 = phi [2 x <1 x float>] [ %1246, %1196 ], [ zeroinitializer, %51 ]
  %56 = phi [2 x <4 x float>] [ %1223, %1196 ], [ zeroinitializer, %51 ]
  %57 = phi i64 [ %1252, %1196 ], [ 0, %51 ]
  %58 = lshr i64 %57, 5, !dbg !17
  %59 = shl i64 %58, 20
  %60 = shl i64 %57, 15
  %.idx238 = sub i64 %60, %59
  br label %61, !dbg !12

61:                                               ; preds = %.lr.ph, %136
  %62 = phi [2 x <32 x float>] [ zeroinitializer, %.lr.ph ], [ %231, %136 ]
  %63 = phi i64 [ 0, %.lr.ph ], [ %232, %136 ]
  %64 = shl nuw nsw i64 %63, 4, !dbg !12
  %65 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %64, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %66 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %67 = and i32 %66, 384, !dbg !9
  %address.local272 = zext nneg i32 %67 to i64, !dbg !9
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local272, !dbg !9
  %address.local277 = and i32 %66, 2, !dbg !9
  %address.local278 = shl nuw nsw i32 %address.thread, 6, !dbg !9
  %address.local279 = and i32 %address.local278, 16256, !dbg !9
  %address.local280 = or disjoint i32 %address.local277, %address.local279, !dbg !9
  %address.local281 = zext nneg i32 %address.local280 to i64, !dbg !9
  %address.local282 = getelementptr [2 x i8], ptr addrspace(1) %65, i64 %address.local281, !dbg !9
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local282 to i64, !dbg !9
  %dma.source.opaque = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !9
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !9
  %address.local285 = or disjoint i64 %address.local272, 512, !dbg !9
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local285, !dbg !9
  %address.local296 = getelementptr i8, ptr addrspace(1) %address.local282, i64 8, !dbg !9
  %dma.source.bits985 = ptrtoint ptr addrspace(1) %address.local296 to i64, !dbg !9
  %dma.source.opaque986 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits985) #9, !dbg !9
  %dma.source.ptr987 = inttoptr i64 %dma.source.opaque986 to ptr addrspace(1), !dbg !9
  %address.local299 = or disjoint i64 %address.local272, 1024, !dbg !9
  %address.local300 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local299, !dbg !9
  %address.local310 = getelementptr i8, ptr addrspace(1) %address.local282, i64 16, !dbg !9
  %dma.source.bits988 = ptrtoint ptr addrspace(1) %address.local310 to i64, !dbg !9
  %dma.source.opaque989 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits988) #9, !dbg !9
  %dma.source.ptr990 = inttoptr i64 %dma.source.opaque989 to ptr addrspace(1), !dbg !9
  %address.local313 = or disjoint i64 %address.local272, 1536, !dbg !9
  %address.local314 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local313, !dbg !9
  %address.local324 = getelementptr i8, ptr addrspace(1) %address.local282, i64 24, !dbg !9
  %dma.source.bits991 = ptrtoint ptr addrspace(1) %address.local324 to i64, !dbg !9
  %dma.source.opaque992 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits991) #9, !dbg !9
  %dma.source.ptr993 = inttoptr i64 %dma.source.opaque992 to ptr addrspace(1), !dbg !9
  %68 = shl nuw nsw i64 %63, 9, !dbg !17
  %69 = add nuw nsw i64 %68, %57, !dbg !17
  %70 = lshr i64 %69, 12, !dbg !17
  %71 = add nuw nsw i64 %70, %8, !dbg !17
  %72 = add nuw nsw i64 %64, %58, !dbg !17
  %73 = and i64 %72, 127, !dbg !17
  %74 = shl i64 %71, 19, !dbg !17
  %75 = and i64 %74, 16252928, !dbg !17
  %76 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %75, !dbg !17
  %77 = getelementptr [2 x i8], ptr addrspace(1) %76, i64 %73, !dbg !17
  %78 = getelementptr i8, ptr addrspace(1) %77, i64 %.idx238, !dbg !17
  %address.local325 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local272, !dbg !17
  %address.local329 = and i32 %66, 14, !dbg !17
  %address.local330 = shl nuw nsw i32 %address.thread, 4, !dbg !17
  %address.local331 = and i32 %address.local330, 3968, !dbg !17
  %address.local332 = or disjoint i32 %address.local329, %address.local331, !dbg !17
  %address.local333 = zext nneg i32 %address.local332 to i64, !dbg !17
  %address.local334 = getelementptr [2 x i8], ptr addrspace(1) %78, i64 %address.local333, !dbg !17
  %dma.source.bits994 = ptrtoint ptr addrspace(1) %address.local334 to i64, !dbg !17
  %dma.source.opaque995 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits994) #9, !dbg !17
  %dma.source.ptr996 = inttoptr i64 %dma.source.opaque995 to ptr addrspace(1), !dbg !17
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local285, !dbg !17
  %address.local345 = getelementptr i8, ptr addrspace(1) %address.local334, i64 8192, !dbg !17
  %dma.source.bits997 = ptrtoint ptr addrspace(1) %address.local345 to i64, !dbg !17
  %dma.source.opaque998 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits997) #9, !dbg !17
  %dma.source.ptr999 = inttoptr i64 %dma.source.opaque998 to ptr addrspace(1), !dbg !17
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local299, !dbg !17
  %address.local356 = getelementptr i8, ptr addrspace(1) %address.local334, i64 16384, !dbg !17
  %dma.source.bits1000 = ptrtoint ptr addrspace(1) %address.local356 to i64, !dbg !17
  %dma.source.opaque1001 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1000) #9, !dbg !17
  %dma.source.ptr1002 = inttoptr i64 %dma.source.opaque1001 to ptr addrspace(1), !dbg !17
  %address.local357 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local313, !dbg !17
  %79 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local273, i64 0, !dbg !9
  %80 = insertelement <8 x ptr addrspace(3)> %79, ptr addrspace(3) %address.local286, i64 1, !dbg !9
  %81 = insertelement <8 x ptr addrspace(3)> %80, ptr addrspace(3) %address.local300, i64 2, !dbg !9
  %82 = insertelement <8 x ptr addrspace(3)> %81, ptr addrspace(3) %address.local314, i64 3, !dbg !9
  %83 = insertelement <8 x ptr addrspace(3)> %82, ptr addrspace(3) %address.local325, i64 4, !dbg !9
  %84 = insertelement <8 x ptr addrspace(3)> %83, ptr addrspace(3) %address.local335, i64 5, !dbg !9
  %85 = insertelement <8 x ptr addrspace(3)> %84, ptr addrspace(3) %address.local346, i64 6, !dbg !9
  %86 = insertelement <8 x ptr addrspace(3)> %85, ptr addrspace(3) %address.local357, i64 7, !dbg !9
  %87 = ptrtoint <8 x ptr addrspace(3)> %86 to <8 x i64>, !dbg !9
  %88 = trunc <8 x i64> %87 to <8 x i32>, !dbg !9
  %89 = extractelement <8 x i32> %88, i64 0, !dbg !9
  %90 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89), !dbg !9
  %91 = zext i32 %90 to i64, !dbg !9
  %92 = inttoptr i64 %91 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %92, i32 4, i32 0, i32 0), !dbg !9
  %93 = extractelement <8 x i32> %88, i64 1, !dbg !9
  %94 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %93), !dbg !9
  %95 = zext i32 %94 to i64, !dbg !9
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr987, ptr addrspace(3) %96, i32 4, i32 0, i32 0), !dbg !9
  %97 = extractelement <8 x i32> %88, i64 2, !dbg !9
  %98 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !9
  %99 = zext i32 %98 to i64, !dbg !9
  %100 = inttoptr i64 %99 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr990, ptr addrspace(3) %100, i32 4, i32 0, i32 0), !dbg !9
  %101 = extractelement <8 x i32> %88, i64 3, !dbg !9
  %102 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101), !dbg !9
  %103 = zext i32 %102 to i64, !dbg !9
  %104 = inttoptr i64 %103 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr993, ptr addrspace(3) %104, i32 4, i32 0, i32 0), !dbg !9
  tail call void @llvm.amdgcn.asyncmark(), !dbg !9
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %105 = extractelement <8 x i32> %88, i64 4, !dbg !17
  %106 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !17
  %107 = zext i32 %106 to i64, !dbg !17
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr996, ptr addrspace(3) %108, i32 4, i32 0, i32 0), !dbg !17
  %109 = extractelement <8 x i32> %88, i64 5, !dbg !17
  %110 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !17
  %111 = zext i32 %110 to i64, !dbg !17
  %112 = inttoptr i64 %111 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr999, ptr addrspace(3) %112, i32 4, i32 0, i32 0), !dbg !17
  %113 = extractelement <8 x i32> %88, i64 6, !dbg !17
  %114 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113), !dbg !17
  %115 = zext i32 %114 to i64, !dbg !17
  %116 = inttoptr i64 %115 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1002, ptr addrspace(3) %116, i32 4, i32 0, i32 0), !dbg !17
  %117 = extractelement <8 x i32> %88, i64 7, !dbg !17
  %118 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %117), !dbg !17
  %119 = zext i32 %118 to i64, !dbg !17
  %120 = inttoptr i64 %119 to ptr addrspace(3), !dbg !17
  %address.local367 = getelementptr i8, ptr addrspace(1) %address.local334, i64 24576, !dbg !17
  %dma.source.bits1003 = ptrtoint ptr addrspace(1) %address.local367 to i64, !dbg !17
  %dma.source.opaque1004 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1003) #9, !dbg !17
  %dma.source.ptr1005 = inttoptr i64 %dma.source.opaque1004 to ptr addrspace(1), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1005, ptr addrspace(3) %120, i32 4, i32 0, i32 0), !dbg !17
  tail call void @llvm.amdgcn.asyncmark(), !dbg !17
  %address.local368 = and i32 %address.thread, 192, !dbg !10
  %address.local369 = shl nuw nsw i32 %address.thread, 2, !dbg !10
  %address.local370 = and i32 %address.local369, 60, !dbg !10
  %address.local371 = or disjoint i32 %address.local370, %address.local368, !dbg !10
  %address.local372 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local373 = and i32 %address.local372, 1536, !dbg !10
  %address.local374 = or disjoint i32 %address.local371, %address.local373, !dbg !10
  %address.local375 = zext nneg i32 %address.local374 to i64, !dbg !10
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !10
  %121 = load <4 x half>, ptr addrspace(3) %address.local376, align 8, !dbg !10
  %address.local386 = getelementptr i8, ptr addrspace(3) %address.local376, i64 512, !dbg !10
  %122 = load <4 x half>, ptr addrspace(3) %address.local386, align 8, !dbg !10
  %123 = fmul <4 x half> %121, splat (half 1.275630e-01), !dbg !10
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  br label %124, !dbg !17

124:                                              ; preds = %61, %124
  %125 = phi i64 [ %18, %61 ], [ %134, %124 ]
  %126 = lshr i64 %125, 6, !dbg !17
  %127 = and i64 %126, 14, !dbg !17
  %gep72 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %invariant.gep71, i64 %127, !dbg !17
  %128 = load half, ptr addrspace(3) %gep72, align 2, !dbg !17
  %129 = getelementptr inbounds nuw i8, ptr addrspace(3) %gep72, i64 32, !dbg !17
  %130 = load half, ptr addrspace(3) %129, align 2, !dbg !17
  %131 = insertelement <2 x half> poison, half %128, i64 0, !dbg !17
  %132 = insertelement <2 x half> %131, half %130, i64 1, !dbg !17
  %.idx46 = shl i64 %125, 2, !dbg !17
  %133 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %.idx46, !dbg !17
  store <2 x half> %132, ptr addrspace(3) %133, align 4, !dbg !17
  %134 = add nuw nsw i64 %125, 256, !dbg !17
  %135 = icmp samesign ult i64 %125, 768, !dbg !17
  br i1 %135, label %124, label %136, !dbg !17

136:                                              ; preds = %124
  %137 = fmul <4 x half> %122, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread387 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !18, !range !11, !frisk.register_lifetimes !5
  %address.local388 = shl nuw nsw i32 %address.thread387, 1, !dbg !18
  %address.local389 = and i32 %address.local388, 8, !dbg !18
  %address.local390 = shl nuw nsw i32 %address.thread387, 4, !dbg !18
  %address.local391 = and i32 %address.local390, 48, !dbg !18
  %address.local392 = or disjoint i32 %address.local389, %address.local391, !dbg !18
  %address.local393 = shl nuw nsw i32 %address.thread387, 5, !dbg !18
  %address.local394 = and i32 %address.local393, 1792, !dbg !18
  %address.local395 = or disjoint i32 %address.local392, %address.local394, !dbg !18
  %address.local396 = zext nneg i32 %address.local395 to i64, !dbg !18
  %address.local397 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local396, !dbg !18
  %138 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local397, i32 0) #10, !dbg !18
  %139 = bitcast <4 x i32> %138 to <8 x half>, !dbg !18
  %address.local408 = getelementptr i8, ptr addrspace(3) %address.local397, i64 128, !dbg !18
  %140 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local408, i32 0) #10, !dbg !18
  %141 = bitcast <4 x i32> %140 to <8 x half>, !dbg !18
  %address.local419 = getelementptr i8, ptr addrspace(3) %address.local397, i64 256, !dbg !18
  %142 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local419, i32 0) #10, !dbg !18
  %143 = bitcast <4 x i32> %142 to <8 x half>, !dbg !18
  %address.local430 = getelementptr i8, ptr addrspace(3) %address.local397, i64 384, !dbg !18
  %144 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local430, i32 0) #10, !dbg !18
  %145 = bitcast <4 x i32> %144 to <8 x half>, !dbg !18
  %146 = shufflevector <8 x half> %139, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !18
  %148 = extractvalue [2 x <32 x float>] %62, 0, !dbg !19
  %149 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %150 = fadd <4 x float> %149, %147, !dbg !19
  %151 = shufflevector <8 x half> %139, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %151, <4 x float> zeroinitializer) #8, !dbg !18
  %153 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %154 = fadd <4 x float> %153, %152, !dbg !19
  %155 = shufflevector <8 x half> %141, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %155, <4 x float> zeroinitializer) #8, !dbg !18
  %157 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %158 = fadd <4 x float> %157, %156, !dbg !19
  %159 = shufflevector <4 x float> %158, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %160 = shufflevector <8 x half> %141, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !18
  %162 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %163 = fadd <4 x float> %162, %161, !dbg !19
  %164 = shufflevector <4 x float> %163, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %165 = shufflevector <8 x half> %143, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %165, <4 x float> zeroinitializer) #8, !dbg !18
  %167 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %168 = fadd <4 x float> %167, %166, !dbg !19
  %169 = shufflevector <4 x float> %168, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %170 = shufflevector <8 x half> %143, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %170, <4 x float> zeroinitializer) #8, !dbg !18
  %172 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %173 = fadd <4 x float> %172, %171, !dbg !19
  %174 = shufflevector <4 x float> %173, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %175 = shufflevector <8 x half> %145, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %175, <4 x float> zeroinitializer) #8, !dbg !18
  %177 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %178 = fadd <4 x float> %177, %176, !dbg !19
  %179 = shufflevector <4 x float> %178, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %180 = shufflevector <8 x half> %145, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %123, <4 x half> %180, <4 x float> zeroinitializer) #8, !dbg !18
  %182 = shufflevector <32 x float> %148, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %183 = fadd <4 x float> %182, %181, !dbg !19
  %184 = shufflevector <4 x float> %183, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !18
  %186 = extractvalue [2 x <32 x float>] %62, 1, !dbg !19
  %187 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %188 = fadd <4 x float> %187, %185, !dbg !19
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %151, <4 x float> zeroinitializer) #8, !dbg !18
  %190 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %191 = fadd <4 x float> %190, %189, !dbg !19
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %155, <4 x float> zeroinitializer) #8, !dbg !18
  %193 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %194 = fadd <4 x float> %193, %192, !dbg !19
  %195 = shufflevector <4 x float> %194, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !18
  %197 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %198 = fadd <4 x float> %197, %196, !dbg !19
  %199 = shufflevector <4 x float> %198, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %165, <4 x float> zeroinitializer) #8, !dbg !18
  %201 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %202 = fadd <4 x float> %201, %200, !dbg !19
  %203 = shufflevector <4 x float> %202, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %170, <4 x float> zeroinitializer) #8, !dbg !18
  %205 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %206 = fadd <4 x float> %205, %204, !dbg !19
  %207 = shufflevector <4 x float> %206, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %175, <4 x float> zeroinitializer) #8, !dbg !18
  %209 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %210 = fadd <4 x float> %209, %208, !dbg !19
  %211 = shufflevector <4 x float> %210, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %137, <4 x half> %180, <4 x float> zeroinitializer) #8, !dbg !18
  %213 = shufflevector <32 x float> %186, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %214 = fadd <4 x float> %213, %212, !dbg !19
  %215 = shufflevector <4 x float> %214, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %216 = shufflevector <4 x float> %150, <4 x float> %154, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %217 = shufflevector <32 x float> %216, <32 x float> %159, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %218 = shufflevector <32 x float> %217, <32 x float> %164, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %219 = shufflevector <32 x float> %218, <32 x float> %169, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %220 = shufflevector <32 x float> %219, <32 x float> %174, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %221 = shufflevector <32 x float> %220, <32 x float> %179, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %222 = shufflevector <32 x float> %221, <32 x float> %184, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !12
  %223 = insertvalue [2 x <32 x float>] poison, <32 x float> %222, 0, !dbg !12
  %224 = shufflevector <4 x float> %188, <4 x float> %191, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %225 = shufflevector <32 x float> %224, <32 x float> %195, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %226 = shufflevector <32 x float> %225, <32 x float> %199, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %227 = shufflevector <32 x float> %226, <32 x float> %203, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %228 = shufflevector <32 x float> %227, <32 x float> %207, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %229 = shufflevector <32 x float> %228, <32 x float> %211, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %230 = shufflevector <32 x float> %229, <32 x float> %215, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !12
  %231 = insertvalue [2 x <32 x float>] %223, <32 x float> %230, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %232 = add nuw nsw i64 %63, 1, !dbg !12
  %233 = icmp samesign ult i64 %63, 7, !dbg !12
  br i1 %233, label %61, label %234, !dbg !12

234:                                              ; preds = %136
  %235 = add nuw nsw i64 %58, %8, !dbg !20
  %236 = shl i64 %235, 19, !dbg !20
  %237 = and i64 %236, 16252928, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %238 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %237, !dbg !20
  %239 = getelementptr i8, ptr addrspace(1) %238, i64 %.idx238, !dbg !20
  %240 = getelementptr [2 x i8], ptr addrspace(1) %239, i64 %53, !dbg !20
  %address.thread431 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !20, !range !11, !frisk.register_lifetimes !5
  %241 = shl nuw nsw i32 %address.thread431, 1, !dbg !9
  %242 = and i32 %241, 384, !dbg !9
  %address.local434 = zext nneg i32 %242 to i64, !dbg !9
  %address.local435 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local434, !dbg !20
  %address.local439 = and i32 %241, 14, !dbg !17
  %address.local440 = shl nuw nsw i32 %address.thread431, 4, !dbg !17
  %address.local441 = and i32 %address.local440, 3968, !dbg !17
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !17
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !17
  %address.local444 = getelementptr [2 x i8], ptr addrspace(1) %240, i64 %address.local443, !dbg !20
  %dma.source.bits1006 = ptrtoint ptr addrspace(1) %address.local444 to i64, !dbg !20
  %dma.source.opaque1007 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1006) #9, !dbg !20
  %dma.source.ptr1008 = inttoptr i64 %dma.source.opaque1007 to ptr addrspace(1), !dbg !20
  %address.local448 = getelementptr i8, ptr addrspace(3) %address.local435, i64 1024, !dbg !20
  %address.local458 = getelementptr i8, ptr addrspace(1) %address.local444, i64 8192, !dbg !20
  %dma.source.bits1009 = ptrtoint ptr addrspace(1) %address.local458 to i64, !dbg !20
  %dma.source.opaque1010 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1009) #9, !dbg !20
  %dma.source.ptr1011 = inttoptr i64 %dma.source.opaque1010 to ptr addrspace(1), !dbg !20
  %address.local462 = getelementptr i8, ptr addrspace(3) %address.local435, i64 2048, !dbg !20
  %address.local472 = getelementptr i8, ptr addrspace(1) %address.local444, i64 16384, !dbg !20
  %dma.source.bits1012 = ptrtoint ptr addrspace(1) %address.local472 to i64, !dbg !20
  %dma.source.opaque1013 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1012) #9, !dbg !20
  %dma.source.ptr1014 = inttoptr i64 %dma.source.opaque1013 to ptr addrspace(1), !dbg !20
  %address.local476 = getelementptr i8, ptr addrspace(3) %address.local435, i64 3072, !dbg !20
  %243 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local435, i64 0, !dbg !20
  %244 = insertelement <4 x ptr addrspace(3)> %243, ptr addrspace(3) %address.local448, i64 1, !dbg !20
  %245 = insertelement <4 x ptr addrspace(3)> %244, ptr addrspace(3) %address.local462, i64 2, !dbg !20
  %246 = insertelement <4 x ptr addrspace(3)> %245, ptr addrspace(3) %address.local476, i64 3, !dbg !20
  %247 = ptrtoint <4 x ptr addrspace(3)> %246 to <4 x i64>, !dbg !20
  %248 = trunc <4 x i64> %247 to <4 x i32>, !dbg !20
  %249 = extractelement <4 x i32> %248, i64 0, !dbg !20
  %250 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %249), !dbg !20
  %251 = zext i32 %250 to i64, !dbg !20
  %252 = inttoptr i64 %251 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1008, ptr addrspace(3) %252, i32 4, i32 0, i32 0), !dbg !20
  %253 = extractelement <4 x i32> %248, i64 1, !dbg !20
  %254 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %253), !dbg !20
  %255 = zext i32 %254 to i64, !dbg !20
  %256 = inttoptr i64 %255 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1011, ptr addrspace(3) %256, i32 4, i32 0, i32 0), !dbg !20
  %257 = extractelement <4 x i32> %248, i64 2, !dbg !20
  %258 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %257), !dbg !20
  %259 = zext i32 %258 to i64, !dbg !20
  %260 = inttoptr i64 %259 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1014, ptr addrspace(3) %260, i32 4, i32 0, i32 0), !dbg !20
  %261 = extractelement <4 x i32> %248, i64 3, !dbg !20
  %262 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %261), !dbg !20
  %263 = zext i32 %262 to i64, !dbg !20
  %264 = inttoptr i64 %263 to ptr addrspace(3), !dbg !20
  %address.local486 = getelementptr i8, ptr addrspace(1) %address.local444, i64 24576, !dbg !20
  %dma.source.bits1015 = ptrtoint ptr addrspace(1) %address.local486 to i64, !dbg !20
  %dma.source.opaque1016 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1015) #9, !dbg !20
  %dma.source.ptr1017 = inttoptr i64 %dma.source.opaque1016 to ptr addrspace(1), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1017, ptr addrspace(3) %264, i32 4, i32 0, i32 0), !dbg !20
  tail call void @llvm.amdgcn.asyncmark(), !dbg !20
  %265 = extractelement <4 x float> %150, i64 0, !dbg !12
  %266 = extractelement <4 x float> %150, i64 1, !dbg !12
  %267 = extractelement <4 x float> %150, i64 2, !dbg !12
  %268 = extractelement <4 x float> %150, i64 3, !dbg !12
  %269 = select i1 %.not, float %266, float %265, !dbg !12
  %270 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %271 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %270), !dbg !12
  %272 = add nuw nsw i32 %271, 64, !dbg !12
  %273 = xor i32 %271, 16, !dbg !12
  %274 = and i32 %272, 192, !dbg !12
  %275 = icmp samesign ult i32 %273, %274, !dbg !12
  %276 = select i1 %275, i32 %273, i32 %271, !dbg !12
  %277 = shl nuw nsw i32 %276, 2, !dbg !12
  %278 = bitcast float %269 to i32, !dbg !12
  %279 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %278), !dbg !12
  %280 = bitcast i32 %279 to float, !dbg !12
  %281 = select i1 %.not, float %265, float %280, !dbg !12
  %282 = select i1 %.not, float %280, float %266, !dbg !12
  %283 = select i1 %.not, float %268, float %267, !dbg !12
  %284 = bitcast float %283 to i32, !dbg !12
  %285 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %284), !dbg !12
  %286 = bitcast i32 %285 to float, !dbg !12
  %287 = select i1 %.not, float %267, float %286, !dbg !12
  %288 = select i1 %.not, float %286, float %268, !dbg !12
  %289 = select i1 %.not155, float %287, float %281, !dbg !12
  %290 = xor i32 %271, 32, !dbg !12
  %291 = icmp samesign ult i32 %290, %274, !dbg !12
  %292 = select i1 %291, i32 %290, i32 %271, !dbg !12
  %293 = shl nuw nsw i32 %292, 2, !dbg !12
  %294 = bitcast float %289 to i32, !dbg !12
  %295 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %294), !dbg !12
  %296 = bitcast i32 %295 to float, !dbg !12
  %297 = select i1 %.not155, float %281, float %296, !dbg !12
  %298 = select i1 %.not155, float %296, float %287, !dbg !12
  %299 = select i1 %.not155, float %288, float %282, !dbg !12
  %300 = bitcast float %299 to i32, !dbg !12
  %301 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %300), !dbg !12
  %302 = bitcast i32 %301 to float, !dbg !12
  %303 = select i1 %.not155, float %282, float %302, !dbg !12
  %304 = select i1 %.not155, float %302, float %288, !dbg !12
  %305 = lshr i64 %57, 25, !dbg !21
  %compare.high729 = trunc nuw i64 %305 to i32, !dbg !21
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high729, !dbg !21
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high729, !dbg !21
  %.tr = trunc i64 %57 to i32, !dbg !21
  %306 = shl i32 %.tr, 7, !dbg !21
  %compare.low.base730 = or disjoint i32 %28, %306, !dbg !21
  %compare.low = icmp ult i32 %compare.low.base730, %compare.low.base, !dbg !21
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !21
  %307 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !22
  %308 = fadd float %307, %297, !dbg !13
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 1) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local = or i32 %compare.local.base, 1, !dbg !21
  %compare.low731 = icmp ult i32 %compare.local, %compare.low.base, !dbg !21
  %compare.result732 = select i1 %compare.high.equal, i1 %compare.low731, i1 %compare.high.order, !dbg !21
  %309 = select i1 %compare.result732, float 0.000000e+00, float -inf, !dbg !22
  %310 = fadd float %303, %309, !dbg !13
  %compare.local.base733 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 2) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local734 = or i32 %compare.local.base733, 2, !dbg !21
  %compare.low735 = icmp ult i32 %compare.local734, %compare.low.base, !dbg !21
  %compare.result736 = select i1 %compare.high.equal, i1 %compare.low735, i1 %compare.high.order, !dbg !21
  %311 = select i1 %compare.result736, float 0.000000e+00, float -inf, !dbg !22
  %312 = fadd float %298, %311, !dbg !13
  %compare.local.base737 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 3) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local738 = or i32 %compare.local.base737, 3, !dbg !21
  %compare.low739 = icmp ult i32 %compare.local738, %compare.low.base, !dbg !21
  %compare.result740 = select i1 %compare.high.equal, i1 %compare.low739, i1 %compare.high.order, !dbg !21
  %313 = select i1 %compare.result740, float 0.000000e+00, float -inf, !dbg !22
  %314 = fadd float %304, %313, !dbg !13
  %315 = tail call float @__ocml_exp2_f32(float %308), !dbg !23
  %316 = insertelement <4 x float> poison, float %315, i64 0, !dbg !24
  %317 = fadd float %315, 0.000000e+00, !dbg !25
  %318 = tail call float @__ocml_exp2_f32(float %310), !dbg !23
  %319 = insertelement <4 x float> %316, float %318, i64 1, !dbg !24
  %320 = fadd float %317, %318, !dbg !25
  %321 = tail call float @__ocml_exp2_f32(float %312), !dbg !23
  %322 = insertelement <4 x float> %319, float %321, i64 2, !dbg !24
  %323 = fadd float %320, %321, !dbg !25
  %324 = tail call float @__ocml_exp2_f32(float %314), !dbg !23
  %325 = insertelement <4 x float> %322, float %324, i64 3, !dbg !24
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %325) #8, !dbg !24, !frisk.register_lifetimes !5
  %326 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !24
  %327 = fadd float %323, %324, !dbg !25
  %328 = extractelement <4 x float> %154, i64 0, !dbg !12
  %329 = extractelement <4 x float> %154, i64 1, !dbg !12
  %330 = extractelement <4 x float> %154, i64 2, !dbg !12
  %331 = extractelement <4 x float> %154, i64 3, !dbg !12
  %332 = select i1 %.not, float %329, float %328, !dbg !12
  %333 = bitcast float %332 to i32, !dbg !12
  %334 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %333), !dbg !12
  %335 = bitcast i32 %334 to float, !dbg !12
  %336 = select i1 %.not, float %328, float %335, !dbg !12
  %337 = select i1 %.not, float %335, float %329, !dbg !12
  %338 = select i1 %.not, float %331, float %330, !dbg !12
  %339 = bitcast float %338 to i32, !dbg !12
  %340 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %339), !dbg !12
  %341 = bitcast i32 %340 to float, !dbg !12
  %342 = select i1 %.not, float %330, float %341, !dbg !12
  %343 = select i1 %.not, float %341, float %331, !dbg !12
  %344 = select i1 %.not155, float %342, float %336, !dbg !12
  %345 = bitcast float %344 to i32, !dbg !12
  %346 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %345), !dbg !12
  %347 = bitcast i32 %346 to float, !dbg !12
  %348 = select i1 %.not155, float %336, float %347, !dbg !12
  %349 = select i1 %.not155, float %347, float %342, !dbg !12
  %350 = select i1 %.not155, float %343, float %337, !dbg !12
  %351 = bitcast float %350 to i32, !dbg !12
  %352 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %351), !dbg !12
  %353 = bitcast i32 %352 to float, !dbg !12
  %354 = select i1 %.not155, float %337, float %353, !dbg !12
  %355 = select i1 %.not155, float %353, float %343, !dbg !12
  %compare.local.base741 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 4) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local742 = or i32 %compare.local.base741, 16, !dbg !21
  %compare.low743 = icmp ult i32 %compare.local742, %compare.low.base, !dbg !21
  %compare.result744 = select i1 %compare.high.equal, i1 %compare.low743, i1 %compare.high.order, !dbg !21
  %356 = select i1 %compare.result744, float 0.000000e+00, float -inf, !dbg !22
  %357 = fadd float %348, %356, !dbg !13
  %compare.local.base745 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 5) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local746 = or i32 %compare.local.base745, 17, !dbg !21
  %compare.low747 = icmp ult i32 %compare.local746, %compare.low.base, !dbg !21
  %compare.result748 = select i1 %compare.high.equal, i1 %compare.low747, i1 %compare.high.order, !dbg !21
  %358 = select i1 %compare.result748, float 0.000000e+00, float -inf, !dbg !22
  %359 = fadd float %354, %358, !dbg !13
  %compare.local.base749 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 6) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local750 = or i32 %compare.local.base749, 18, !dbg !21
  %compare.low751 = icmp ult i32 %compare.local750, %compare.low.base, !dbg !21
  %compare.result752 = select i1 %compare.high.equal, i1 %compare.low751, i1 %compare.high.order, !dbg !21
  %360 = select i1 %compare.result752, float 0.000000e+00, float -inf, !dbg !22
  %361 = fadd float %349, %360, !dbg !13
  %compare.local.base753 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 7) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local754 = or i32 %compare.local.base753, 19, !dbg !21
  %compare.low755 = icmp ult i32 %compare.local754, %compare.low.base, !dbg !21
  %compare.result756 = select i1 %compare.high.equal, i1 %compare.low755, i1 %compare.high.order, !dbg !21
  %362 = select i1 %compare.result756, float 0.000000e+00, float -inf, !dbg !22
  %363 = fadd float %355, %362, !dbg !13
  %364 = tail call float @__ocml_exp2_f32(float %357), !dbg !23
  %365 = insertelement <4 x float> poison, float %364, i64 0, !dbg !24
  %366 = fadd float %327, %364, !dbg !25
  %367 = tail call float @__ocml_exp2_f32(float %359), !dbg !23
  %368 = insertelement <4 x float> %365, float %367, i64 1, !dbg !24
  %369 = fadd float %366, %367, !dbg !25
  %370 = tail call float @__ocml_exp2_f32(float %361), !dbg !23
  %371 = insertelement <4 x float> %368, float %370, i64 2, !dbg !24
  %372 = fadd float %369, %370, !dbg !25
  %373 = tail call float @__ocml_exp2_f32(float %363), !dbg !23
  %374 = insertelement <4 x float> %371, float %373, i64 3, !dbg !24
  %math.result241 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %374) #8, !dbg !24, !frisk.register_lifetimes !5
  %375 = fptrunc <4 x float> %math.result241 to <4 x half>, !dbg !24
  %376 = fadd float %372, %373, !dbg !25
  %377 = extractelement <4 x float> %158, i64 0, !dbg !12
  %378 = extractelement <4 x float> %158, i64 1, !dbg !12
  %379 = extractelement <4 x float> %158, i64 2, !dbg !12
  %380 = extractelement <4 x float> %158, i64 3, !dbg !12
  %381 = select i1 %.not, float %378, float %377, !dbg !12
  %382 = bitcast float %381 to i32, !dbg !12
  %383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %382), !dbg !12
  %384 = bitcast i32 %383 to float, !dbg !12
  %385 = select i1 %.not, float %377, float %384, !dbg !12
  %386 = select i1 %.not, float %384, float %378, !dbg !12
  %387 = select i1 %.not, float %380, float %379, !dbg !12
  %388 = bitcast float %387 to i32, !dbg !12
  %389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %388), !dbg !12
  %390 = bitcast i32 %389 to float, !dbg !12
  %391 = select i1 %.not, float %379, float %390, !dbg !12
  %392 = select i1 %.not, float %390, float %380, !dbg !12
  %393 = select i1 %.not155, float %391, float %385, !dbg !12
  %394 = bitcast float %393 to i32, !dbg !12
  %395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %394), !dbg !12
  %396 = bitcast i32 %395 to float, !dbg !12
  %397 = select i1 %.not155, float %385, float %396, !dbg !12
  %398 = select i1 %.not155, float %396, float %391, !dbg !12
  %399 = select i1 %.not155, float %392, float %386, !dbg !12
  %400 = bitcast float %399 to i32, !dbg !12
  %401 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %400), !dbg !12
  %402 = bitcast i32 %401 to float, !dbg !12
  %403 = select i1 %.not155, float %386, float %402, !dbg !12
  %404 = select i1 %.not155, float %402, float %392, !dbg !12
  %compare.local.base757 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 8) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local758 = or i32 %compare.local.base757, 32, !dbg !21
  %compare.low759 = icmp ult i32 %compare.local758, %compare.low.base, !dbg !21
  %compare.result760 = select i1 %compare.high.equal, i1 %compare.low759, i1 %compare.high.order, !dbg !21
  %405 = select i1 %compare.result760, float 0.000000e+00, float -inf, !dbg !22
  %406 = fadd float %397, %405, !dbg !13
  %compare.local.base761 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 9) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local762 = or i32 %compare.local.base761, 33, !dbg !21
  %compare.low763 = icmp ult i32 %compare.local762, %compare.low.base, !dbg !21
  %compare.result764 = select i1 %compare.high.equal, i1 %compare.low763, i1 %compare.high.order, !dbg !21
  %407 = select i1 %compare.result764, float 0.000000e+00, float -inf, !dbg !22
  %408 = fadd float %403, %407, !dbg !13
  %compare.local.base765 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 10) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local766 = or i32 %compare.local.base765, 34, !dbg !21
  %compare.low767 = icmp ult i32 %compare.local766, %compare.low.base, !dbg !21
  %compare.result768 = select i1 %compare.high.equal, i1 %compare.low767, i1 %compare.high.order, !dbg !21
  %409 = select i1 %compare.result768, float 0.000000e+00, float -inf, !dbg !22
  %410 = fadd float %398, %409, !dbg !13
  %compare.local.base769 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 11) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local770 = or i32 %compare.local.base769, 35, !dbg !21
  %compare.low771 = icmp ult i32 %compare.local770, %compare.low.base, !dbg !21
  %compare.result772 = select i1 %compare.high.equal, i1 %compare.low771, i1 %compare.high.order, !dbg !21
  %411 = select i1 %compare.result772, float 0.000000e+00, float -inf, !dbg !22
  %412 = fadd float %404, %411, !dbg !13
  %413 = tail call float @__ocml_exp2_f32(float %406), !dbg !23
  %414 = insertelement <4 x float> poison, float %413, i64 0, !dbg !24
  %415 = fadd float %376, %413, !dbg !25
  %416 = tail call float @__ocml_exp2_f32(float %408), !dbg !23
  %417 = insertelement <4 x float> %414, float %416, i64 1, !dbg !24
  %418 = fadd float %415, %416, !dbg !25
  %419 = tail call float @__ocml_exp2_f32(float %410), !dbg !23
  %420 = insertelement <4 x float> %417, float %419, i64 2, !dbg !24
  %421 = fadd float %418, %419, !dbg !25
  %422 = tail call float @__ocml_exp2_f32(float %412), !dbg !23
  %423 = insertelement <4 x float> %420, float %422, i64 3, !dbg !24
  %math.result243 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %423) #8, !dbg !24, !frisk.register_lifetimes !5
  %424 = fptrunc <4 x float> %math.result243 to <4 x half>, !dbg !24
  %425 = fadd float %421, %422, !dbg !25
  %426 = extractelement <4 x float> %163, i64 0, !dbg !12
  %427 = extractelement <4 x float> %163, i64 1, !dbg !12
  %428 = extractelement <4 x float> %163, i64 2, !dbg !12
  %429 = extractelement <4 x float> %163, i64 3, !dbg !12
  %430 = select i1 %.not, float %427, float %426, !dbg !12
  %431 = bitcast float %430 to i32, !dbg !12
  %432 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %431), !dbg !12
  %433 = bitcast i32 %432 to float, !dbg !12
  %434 = select i1 %.not, float %426, float %433, !dbg !12
  %435 = select i1 %.not, float %433, float %427, !dbg !12
  %436 = select i1 %.not, float %429, float %428, !dbg !12
  %437 = bitcast float %436 to i32, !dbg !12
  %438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %437), !dbg !12
  %439 = bitcast i32 %438 to float, !dbg !12
  %440 = select i1 %.not, float %428, float %439, !dbg !12
  %441 = select i1 %.not, float %439, float %429, !dbg !12
  %442 = select i1 %.not155, float %440, float %434, !dbg !12
  %443 = bitcast float %442 to i32, !dbg !12
  %444 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %443), !dbg !12
  %445 = bitcast i32 %444 to float, !dbg !12
  %446 = select i1 %.not155, float %434, float %445, !dbg !12
  %447 = select i1 %.not155, float %445, float %440, !dbg !12
  %448 = select i1 %.not155, float %441, float %435, !dbg !12
  %449 = bitcast float %448 to i32, !dbg !12
  %450 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %449), !dbg !12
  %451 = bitcast i32 %450 to float, !dbg !12
  %452 = select i1 %.not155, float %435, float %451, !dbg !12
  %453 = select i1 %.not155, float %451, float %441, !dbg !12
  %compare.local.base773 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 12) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local774 = or i32 %compare.local.base773, 48, !dbg !21
  %compare.low775 = icmp ult i32 %compare.local774, %compare.low.base, !dbg !21
  %compare.result776 = select i1 %compare.high.equal, i1 %compare.low775, i1 %compare.high.order, !dbg !21
  %454 = select i1 %compare.result776, float 0.000000e+00, float -inf, !dbg !22
  %455 = fadd float %446, %454, !dbg !13
  %compare.local.base777 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 13) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local778 = or i32 %compare.local.base777, 49, !dbg !21
  %compare.low779 = icmp ult i32 %compare.local778, %compare.low.base, !dbg !21
  %compare.result780 = select i1 %compare.high.equal, i1 %compare.low779, i1 %compare.high.order, !dbg !21
  %456 = select i1 %compare.result780, float 0.000000e+00, float -inf, !dbg !22
  %457 = fadd float %452, %456, !dbg !13
  %compare.local.base781 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 14) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local782 = or i32 %compare.local.base781, 50, !dbg !21
  %compare.low783 = icmp ult i32 %compare.local782, %compare.low.base, !dbg !21
  %compare.result784 = select i1 %compare.high.equal, i1 %compare.low783, i1 %compare.high.order, !dbg !21
  %458 = select i1 %compare.result784, float 0.000000e+00, float -inf, !dbg !22
  %459 = fadd float %447, %458, !dbg !13
  %compare.local.base785 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 15) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local786 = or i32 %compare.local.base785, 51, !dbg !21
  %compare.low787 = icmp ult i32 %compare.local786, %compare.low.base, !dbg !21
  %compare.result788 = select i1 %compare.high.equal, i1 %compare.low787, i1 %compare.high.order, !dbg !21
  %460 = select i1 %compare.result788, float 0.000000e+00, float -inf, !dbg !22
  %461 = fadd float %453, %460, !dbg !13
  %462 = tail call float @__ocml_exp2_f32(float %455), !dbg !23
  %463 = insertelement <4 x float> poison, float %462, i64 0, !dbg !24
  %464 = fadd float %425, %462, !dbg !25
  %465 = tail call float @__ocml_exp2_f32(float %457), !dbg !23
  %466 = insertelement <4 x float> %463, float %465, i64 1, !dbg !24
  %467 = fadd float %464, %465, !dbg !25
  %468 = tail call float @__ocml_exp2_f32(float %459), !dbg !23
  %469 = insertelement <4 x float> %466, float %468, i64 2, !dbg !24
  %470 = fadd float %467, %468, !dbg !25
  %471 = tail call float @__ocml_exp2_f32(float %461), !dbg !23
  %472 = insertelement <4 x float> %469, float %471, i64 3, !dbg !24
  %math.result245 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %472) #8, !dbg !24, !frisk.register_lifetimes !5
  %473 = fptrunc <4 x float> %math.result245 to <4 x half>, !dbg !24
  %474 = fadd float %470, %471, !dbg !25
  %475 = extractelement <4 x float> %168, i64 0, !dbg !12
  %476 = extractelement <4 x float> %168, i64 1, !dbg !12
  %477 = extractelement <4 x float> %168, i64 2, !dbg !12
  %478 = extractelement <4 x float> %168, i64 3, !dbg !12
  %479 = select i1 %.not, float %476, float %475, !dbg !12
  %480 = bitcast float %479 to i32, !dbg !12
  %481 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %480), !dbg !12
  %482 = bitcast i32 %481 to float, !dbg !12
  %483 = select i1 %.not, float %475, float %482, !dbg !12
  %484 = select i1 %.not, float %482, float %476, !dbg !12
  %485 = select i1 %.not, float %478, float %477, !dbg !12
  %486 = bitcast float %485 to i32, !dbg !12
  %487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %486), !dbg !12
  %488 = bitcast i32 %487 to float, !dbg !12
  %489 = select i1 %.not, float %477, float %488, !dbg !12
  %490 = select i1 %.not, float %488, float %478, !dbg !12
  %491 = select i1 %.not155, float %489, float %483, !dbg !12
  %492 = bitcast float %491 to i32, !dbg !12
  %493 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %492), !dbg !12
  %494 = bitcast i32 %493 to float, !dbg !12
  %495 = select i1 %.not155, float %483, float %494, !dbg !12
  %496 = select i1 %.not155, float %494, float %489, !dbg !12
  %497 = select i1 %.not155, float %490, float %484, !dbg !12
  %498 = bitcast float %497 to i32, !dbg !12
  %499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %498), !dbg !12
  %500 = bitcast i32 %499 to float, !dbg !12
  %501 = select i1 %.not155, float %484, float %500, !dbg !12
  %502 = select i1 %.not155, float %500, float %490, !dbg !12
  %compare.local.base789 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 16) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local790 = or i32 %compare.local.base789, 64, !dbg !21
  %compare.low791 = icmp ult i32 %compare.local790, %compare.low.base, !dbg !21
  %compare.result792 = select i1 %compare.high.equal, i1 %compare.low791, i1 %compare.high.order, !dbg !21
  %503 = select i1 %compare.result792, float 0.000000e+00, float -inf, !dbg !22
  %504 = fadd float %495, %503, !dbg !13
  %compare.local.base793 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 17) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local794 = or i32 %compare.local.base793, 65, !dbg !21
  %compare.low795 = icmp ult i32 %compare.local794, %compare.low.base, !dbg !21
  %compare.result796 = select i1 %compare.high.equal, i1 %compare.low795, i1 %compare.high.order, !dbg !21
  %505 = select i1 %compare.result796, float 0.000000e+00, float -inf, !dbg !22
  %506 = fadd float %501, %505, !dbg !13
  %compare.local.base797 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 18) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local798 = or i32 %compare.local.base797, 66, !dbg !21
  %compare.low799 = icmp ult i32 %compare.local798, %compare.low.base, !dbg !21
  %compare.result800 = select i1 %compare.high.equal, i1 %compare.low799, i1 %compare.high.order, !dbg !21
  %507 = select i1 %compare.result800, float 0.000000e+00, float -inf, !dbg !22
  %508 = fadd float %496, %507, !dbg !13
  %compare.local.base801 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 19) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local802 = or i32 %compare.local.base801, 67, !dbg !21
  %compare.low803 = icmp ult i32 %compare.local802, %compare.low.base, !dbg !21
  %compare.result804 = select i1 %compare.high.equal, i1 %compare.low803, i1 %compare.high.order, !dbg !21
  %509 = select i1 %compare.result804, float 0.000000e+00, float -inf, !dbg !22
  %510 = fadd float %502, %509, !dbg !13
  %511 = tail call float @__ocml_exp2_f32(float %504), !dbg !23
  %512 = insertelement <4 x float> poison, float %511, i64 0, !dbg !24
  %513 = fadd float %474, %511, !dbg !25
  %514 = tail call float @__ocml_exp2_f32(float %506), !dbg !23
  %515 = insertelement <4 x float> %512, float %514, i64 1, !dbg !24
  %516 = fadd float %513, %514, !dbg !25
  %517 = tail call float @__ocml_exp2_f32(float %508), !dbg !23
  %518 = insertelement <4 x float> %515, float %517, i64 2, !dbg !24
  %519 = fadd float %516, %517, !dbg !25
  %520 = tail call float @__ocml_exp2_f32(float %510), !dbg !23
  %521 = insertelement <4 x float> %518, float %520, i64 3, !dbg !24
  %math.result247 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %521) #8, !dbg !24, !frisk.register_lifetimes !5
  %522 = fptrunc <4 x float> %math.result247 to <4 x half>, !dbg !24
  %523 = fadd float %519, %520, !dbg !25
  %524 = extractelement <4 x float> %173, i64 0, !dbg !12
  %525 = extractelement <4 x float> %173, i64 1, !dbg !12
  %526 = extractelement <4 x float> %173, i64 2, !dbg !12
  %527 = extractelement <4 x float> %173, i64 3, !dbg !12
  %528 = select i1 %.not, float %525, float %524, !dbg !12
  %529 = bitcast float %528 to i32, !dbg !12
  %530 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %529), !dbg !12
  %531 = bitcast i32 %530 to float, !dbg !12
  %532 = select i1 %.not, float %524, float %531, !dbg !12
  %533 = select i1 %.not, float %531, float %525, !dbg !12
  %534 = select i1 %.not, float %527, float %526, !dbg !12
  %535 = bitcast float %534 to i32, !dbg !12
  %536 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %535), !dbg !12
  %537 = bitcast i32 %536 to float, !dbg !12
  %538 = select i1 %.not, float %526, float %537, !dbg !12
  %539 = select i1 %.not, float %537, float %527, !dbg !12
  %540 = select i1 %.not155, float %538, float %532, !dbg !12
  %541 = bitcast float %540 to i32, !dbg !12
  %542 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %541), !dbg !12
  %543 = bitcast i32 %542 to float, !dbg !12
  %544 = select i1 %.not155, float %532, float %543, !dbg !12
  %545 = select i1 %.not155, float %543, float %538, !dbg !12
  %546 = select i1 %.not155, float %539, float %533, !dbg !12
  %547 = bitcast float %546 to i32, !dbg !12
  %548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %547), !dbg !12
  %549 = bitcast i32 %548 to float, !dbg !12
  %550 = select i1 %.not155, float %533, float %549, !dbg !12
  %551 = select i1 %.not155, float %549, float %539, !dbg !12
  %compare.local.base805 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 20) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local806 = or i32 %compare.local.base805, 80, !dbg !21
  %compare.low807 = icmp ult i32 %compare.local806, %compare.low.base, !dbg !21
  %compare.result808 = select i1 %compare.high.equal, i1 %compare.low807, i1 %compare.high.order, !dbg !21
  %552 = select i1 %compare.result808, float 0.000000e+00, float -inf, !dbg !22
  %553 = fadd float %544, %552, !dbg !13
  %compare.local.base809 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 21) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local810 = or i32 %compare.local.base809, 81, !dbg !21
  %compare.low811 = icmp ult i32 %compare.local810, %compare.low.base, !dbg !21
  %compare.result812 = select i1 %compare.high.equal, i1 %compare.low811, i1 %compare.high.order, !dbg !21
  %554 = select i1 %compare.result812, float 0.000000e+00, float -inf, !dbg !22
  %555 = fadd float %550, %554, !dbg !13
  %compare.local.base813 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 22) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local814 = or i32 %compare.local.base813, 82, !dbg !21
  %compare.low815 = icmp ult i32 %compare.local814, %compare.low.base, !dbg !21
  %compare.result816 = select i1 %compare.high.equal, i1 %compare.low815, i1 %compare.high.order, !dbg !21
  %556 = select i1 %compare.result816, float 0.000000e+00, float -inf, !dbg !22
  %557 = fadd float %545, %556, !dbg !13
  %compare.local.base817 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 23) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local818 = or i32 %compare.local.base817, 83, !dbg !21
  %compare.low819 = icmp ult i32 %compare.local818, %compare.low.base, !dbg !21
  %compare.result820 = select i1 %compare.high.equal, i1 %compare.low819, i1 %compare.high.order, !dbg !21
  %558 = select i1 %compare.result820, float 0.000000e+00, float -inf, !dbg !22
  %559 = fadd float %551, %558, !dbg !13
  %560 = tail call float @__ocml_exp2_f32(float %553), !dbg !23
  %561 = insertelement <4 x float> poison, float %560, i64 0, !dbg !24
  %562 = fadd float %523, %560, !dbg !25
  %563 = tail call float @__ocml_exp2_f32(float %555), !dbg !23
  %564 = insertelement <4 x float> %561, float %563, i64 1, !dbg !24
  %565 = fadd float %562, %563, !dbg !25
  %566 = tail call float @__ocml_exp2_f32(float %557), !dbg !23
  %567 = insertelement <4 x float> %564, float %566, i64 2, !dbg !24
  %568 = fadd float %565, %566, !dbg !25
  %569 = tail call float @__ocml_exp2_f32(float %559), !dbg !23
  %570 = insertelement <4 x float> %567, float %569, i64 3, !dbg !24
  %math.result249 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %570) #8, !dbg !24, !frisk.register_lifetimes !5
  %571 = fptrunc <4 x float> %math.result249 to <4 x half>, !dbg !24
  %572 = fadd float %568, %569, !dbg !25
  %573 = extractelement <4 x float> %178, i64 0, !dbg !12
  %574 = extractelement <4 x float> %178, i64 1, !dbg !12
  %575 = extractelement <4 x float> %178, i64 2, !dbg !12
  %576 = extractelement <4 x float> %178, i64 3, !dbg !12
  %577 = select i1 %.not, float %574, float %573, !dbg !12
  %578 = bitcast float %577 to i32, !dbg !12
  %579 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %578), !dbg !12
  %580 = bitcast i32 %579 to float, !dbg !12
  %581 = select i1 %.not, float %573, float %580, !dbg !12
  %582 = select i1 %.not, float %580, float %574, !dbg !12
  %583 = select i1 %.not, float %576, float %575, !dbg !12
  %584 = bitcast float %583 to i32, !dbg !12
  %585 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %584), !dbg !12
  %586 = bitcast i32 %585 to float, !dbg !12
  %587 = select i1 %.not, float %575, float %586, !dbg !12
  %588 = select i1 %.not, float %586, float %576, !dbg !12
  %589 = select i1 %.not155, float %587, float %581, !dbg !12
  %590 = bitcast float %589 to i32, !dbg !12
  %591 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %590), !dbg !12
  %592 = bitcast i32 %591 to float, !dbg !12
  %593 = select i1 %.not155, float %581, float %592, !dbg !12
  %594 = select i1 %.not155, float %592, float %587, !dbg !12
  %595 = select i1 %.not155, float %588, float %582, !dbg !12
  %596 = bitcast float %595 to i32, !dbg !12
  %597 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %596), !dbg !12
  %598 = bitcast i32 %597 to float, !dbg !12
  %599 = select i1 %.not155, float %582, float %598, !dbg !12
  %600 = select i1 %.not155, float %598, float %588, !dbg !12
  %compare.local.base821 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 24) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local822 = or i32 %compare.local.base821, 96, !dbg !21
  %compare.low823 = icmp ult i32 %compare.local822, %compare.low.base, !dbg !21
  %compare.result824 = select i1 %compare.high.equal, i1 %compare.low823, i1 %compare.high.order, !dbg !21
  %601 = select i1 %compare.result824, float 0.000000e+00, float -inf, !dbg !22
  %602 = fadd float %593, %601, !dbg !13
  %compare.local.base825 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 25) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local826 = or i32 %compare.local.base825, 97, !dbg !21
  %compare.low827 = icmp ult i32 %compare.local826, %compare.low.base, !dbg !21
  %compare.result828 = select i1 %compare.high.equal, i1 %compare.low827, i1 %compare.high.order, !dbg !21
  %603 = select i1 %compare.result828, float 0.000000e+00, float -inf, !dbg !22
  %604 = fadd float %599, %603, !dbg !13
  %compare.local.base829 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 26) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local830 = or i32 %compare.local.base829, 98, !dbg !21
  %compare.low831 = icmp ult i32 %compare.local830, %compare.low.base, !dbg !21
  %compare.result832 = select i1 %compare.high.equal, i1 %compare.low831, i1 %compare.high.order, !dbg !21
  %605 = select i1 %compare.result832, float 0.000000e+00, float -inf, !dbg !22
  %606 = fadd float %594, %605, !dbg !13
  %compare.local.base833 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 27) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local834 = or i32 %compare.local.base833, 99, !dbg !21
  %compare.low835 = icmp ult i32 %compare.local834, %compare.low.base, !dbg !21
  %compare.result836 = select i1 %compare.high.equal, i1 %compare.low835, i1 %compare.high.order, !dbg !21
  %607 = select i1 %compare.result836, float 0.000000e+00, float -inf, !dbg !22
  %608 = fadd float %600, %607, !dbg !13
  %609 = tail call float @__ocml_exp2_f32(float %602), !dbg !23
  %610 = insertelement <4 x float> poison, float %609, i64 0, !dbg !24
  %611 = fadd float %572, %609, !dbg !25
  %612 = tail call float @__ocml_exp2_f32(float %604), !dbg !23
  %613 = insertelement <4 x float> %610, float %612, i64 1, !dbg !24
  %614 = fadd float %611, %612, !dbg !25
  %615 = tail call float @__ocml_exp2_f32(float %606), !dbg !23
  %616 = insertelement <4 x float> %613, float %615, i64 2, !dbg !24
  %617 = fadd float %614, %615, !dbg !25
  %618 = tail call float @__ocml_exp2_f32(float %608), !dbg !23
  %619 = insertelement <4 x float> %616, float %618, i64 3, !dbg !24
  %math.result251 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %619) #8, !dbg !24, !frisk.register_lifetimes !5
  %620 = fptrunc <4 x float> %math.result251 to <4 x half>, !dbg !24
  %621 = fadd float %617, %618, !dbg !25
  %622 = extractelement <4 x float> %183, i64 0, !dbg !12
  %623 = extractelement <4 x float> %183, i64 1, !dbg !12
  %624 = extractelement <4 x float> %183, i64 2, !dbg !12
  %625 = extractelement <4 x float> %183, i64 3, !dbg !12
  %626 = select i1 %.not, float %623, float %622, !dbg !12
  %627 = bitcast float %626 to i32, !dbg !12
  %628 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %627), !dbg !12
  %629 = bitcast i32 %628 to float, !dbg !12
  %630 = select i1 %.not, float %622, float %629, !dbg !12
  %631 = select i1 %.not, float %629, float %623, !dbg !12
  %632 = select i1 %.not, float %625, float %624, !dbg !12
  %633 = bitcast float %632 to i32, !dbg !12
  %634 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %633), !dbg !12
  %635 = bitcast i32 %634 to float, !dbg !12
  %636 = select i1 %.not, float %624, float %635, !dbg !12
  %637 = select i1 %.not, float %635, float %625, !dbg !12
  %638 = select i1 %.not155, float %636, float %630, !dbg !12
  %639 = bitcast float %638 to i32, !dbg !12
  %640 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %639), !dbg !12
  %641 = bitcast i32 %640 to float, !dbg !12
  %642 = select i1 %.not155, float %630, float %641, !dbg !12
  %643 = select i1 %.not155, float %641, float %636, !dbg !12
  %644 = select i1 %.not155, float %637, float %631, !dbg !12
  %645 = bitcast float %644 to i32, !dbg !12
  %646 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %645), !dbg !12
  %647 = bitcast i32 %646 to float, !dbg !12
  %648 = select i1 %.not155, float %631, float %647, !dbg !12
  %649 = select i1 %.not155, float %647, float %637, !dbg !12
  %compare.local.base837 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 28) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local838 = or i32 %compare.local.base837, 112, !dbg !21
  %compare.low839 = icmp ult i32 %compare.local838, %compare.low.base, !dbg !21
  %compare.result840 = select i1 %compare.high.equal, i1 %compare.low839, i1 %compare.high.order, !dbg !21
  %650 = select i1 %compare.result840, float 0.000000e+00, float -inf, !dbg !22
  %651 = fadd float %642, %650, !dbg !13
  %compare.local.base841 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 29) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local842 = or i32 %compare.local.base841, 113, !dbg !21
  %compare.low843 = icmp ult i32 %compare.local842, %compare.low.base, !dbg !21
  %compare.result844 = select i1 %compare.high.equal, i1 %compare.low843, i1 %compare.high.order, !dbg !21
  %652 = select i1 %compare.result844, float 0.000000e+00, float -inf, !dbg !22
  %653 = fadd float %648, %652, !dbg !13
  %compare.local.base845 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 30) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local846 = or i32 %compare.local.base845, 114, !dbg !21
  %compare.low847 = icmp ult i32 %compare.local846, %compare.low.base, !dbg !21
  %compare.result848 = select i1 %compare.high.equal, i1 %compare.low847, i1 %compare.high.order, !dbg !21
  %654 = select i1 %compare.result848, float 0.000000e+00, float -inf, !dbg !22
  %655 = fadd float %643, %654, !dbg !13
  %compare.local.base849 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 31) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local850 = or i32 %compare.local.base849, 115, !dbg !21
  %compare.low851 = icmp ult i32 %compare.local850, %compare.low.base, !dbg !21
  %compare.result852 = select i1 %compare.high.equal, i1 %compare.low851, i1 %compare.high.order, !dbg !21
  %656 = select i1 %compare.result852, float 0.000000e+00, float -inf, !dbg !22
  %657 = fadd float %649, %656, !dbg !13
  %658 = tail call float @__ocml_exp2_f32(float %651), !dbg !23
  %659 = insertelement <4 x float> poison, float %658, i64 0, !dbg !24
  %660 = fadd float %621, %658, !dbg !25
  %661 = tail call float @__ocml_exp2_f32(float %653), !dbg !23
  %662 = insertelement <4 x float> %659, float %661, i64 1, !dbg !24
  %663 = fadd float %660, %661, !dbg !25
  %664 = tail call float @__ocml_exp2_f32(float %655), !dbg !23
  %665 = insertelement <4 x float> %662, float %664, i64 2, !dbg !24
  %666 = fadd float %663, %664, !dbg !25
  %667 = tail call float @__ocml_exp2_f32(float %657), !dbg !23
  %668 = insertelement <4 x float> %665, float %667, i64 3, !dbg !24
  %math.result253 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %668) #8, !dbg !24, !frisk.register_lifetimes !5
  %669 = fptrunc <4 x float> %math.result253 to <4 x half>, !dbg !24
  %670 = fadd float %666, %667, !dbg !25
  %671 = extractelement <4 x float> %188, i64 0, !dbg !12
  %672 = extractelement <4 x float> %188, i64 1, !dbg !12
  %673 = extractelement <4 x float> %188, i64 2, !dbg !12
  %674 = extractelement <4 x float> %188, i64 3, !dbg !12
  %675 = select i1 %.not, float %672, float %671, !dbg !12
  %676 = bitcast float %675 to i32, !dbg !12
  %677 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %676), !dbg !12
  %678 = bitcast i32 %677 to float, !dbg !12
  %679 = select i1 %.not, float %671, float %678, !dbg !12
  %680 = select i1 %.not, float %678, float %672, !dbg !12
  %681 = select i1 %.not, float %674, float %673, !dbg !12
  %682 = bitcast float %681 to i32, !dbg !12
  %683 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %682), !dbg !12
  %684 = bitcast i32 %683 to float, !dbg !12
  %685 = select i1 %.not, float %673, float %684, !dbg !12
  %686 = select i1 %.not, float %684, float %674, !dbg !12
  %687 = select i1 %.not155, float %685, float %679, !dbg !12
  %688 = bitcast float %687 to i32, !dbg !12
  %689 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %688), !dbg !12
  %690 = bitcast i32 %689 to float, !dbg !12
  %691 = select i1 %.not155, float %679, float %690, !dbg !12
  %692 = select i1 %.not155, float %690, float %685, !dbg !12
  %693 = select i1 %.not155, float %686, float %680, !dbg !12
  %694 = bitcast float %693 to i32, !dbg !12
  %695 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %694), !dbg !12
  %696 = bitcast i32 %695 to float, !dbg !12
  %697 = select i1 %.not155, float %680, float %696, !dbg !12
  %698 = select i1 %.not155, float %696, float %686, !dbg !12
  %compare.high.equal855 = icmp eq i32 %compare.high853, %compare.high729, !dbg !21
  %compare.high.order856 = icmp ugt i32 %compare.high853, %compare.high729, !dbg !21
  %compare.low859 = icmp ult i32 %compare.low.base730, %compare.low.base857, !dbg !21
  %compare.result860 = select i1 %compare.high.equal855, i1 %compare.low859, i1 %compare.high.order856, !dbg !21
  %699 = select i1 %compare.result860, float 0.000000e+00, float -inf, !dbg !22
  %700 = fadd float %699, %691, !dbg !13
  %compare.local.base861 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 32) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local862 = or i32 %compare.local.base861, 1, !dbg !21
  %compare.low863 = icmp ult i32 %compare.local862, %compare.low.base857, !dbg !21
  %compare.result864 = select i1 %compare.high.equal855, i1 %compare.low863, i1 %compare.high.order856, !dbg !21
  %701 = select i1 %compare.result864, float 0.000000e+00, float -inf, !dbg !22
  %702 = fadd float %697, %701, !dbg !13
  %compare.local.base865 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 33) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local866 = or i32 %compare.local.base865, 2, !dbg !21
  %compare.low867 = icmp ult i32 %compare.local866, %compare.low.base857, !dbg !21
  %compare.result868 = select i1 %compare.high.equal855, i1 %compare.low867, i1 %compare.high.order856, !dbg !21
  %703 = select i1 %compare.result868, float 0.000000e+00, float -inf, !dbg !22
  %704 = fadd float %692, %703, !dbg !13
  %compare.local.base869 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 34) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local870 = or i32 %compare.local.base869, 3, !dbg !21
  %compare.low871 = icmp ult i32 %compare.local870, %compare.low.base857, !dbg !21
  %compare.result872 = select i1 %compare.high.equal855, i1 %compare.low871, i1 %compare.high.order856, !dbg !21
  %705 = select i1 %compare.result872, float 0.000000e+00, float -inf, !dbg !22
  %706 = fadd float %698, %705, !dbg !13
  %707 = tail call float @__ocml_exp2_f32(float %700), !dbg !23
  %708 = insertelement <4 x float> poison, float %707, i64 0, !dbg !24
  %709 = fadd float %707, 0.000000e+00, !dbg !25
  %710 = tail call float @__ocml_exp2_f32(float %702), !dbg !23
  %711 = insertelement <4 x float> %708, float %710, i64 1, !dbg !24
  %712 = fadd float %709, %710, !dbg !25
  %713 = tail call float @__ocml_exp2_f32(float %704), !dbg !23
  %714 = insertelement <4 x float> %711, float %713, i64 2, !dbg !24
  %715 = fadd float %712, %713, !dbg !25
  %716 = tail call float @__ocml_exp2_f32(float %706), !dbg !23
  %717 = insertelement <4 x float> %714, float %716, i64 3, !dbg !24
  %math.result255 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %717) #8, !dbg !24, !frisk.register_lifetimes !5
  %718 = fptrunc <4 x float> %math.result255 to <4 x half>, !dbg !24
  %719 = fadd float %715, %716, !dbg !25
  %720 = extractelement <4 x float> %191, i64 0, !dbg !12
  %721 = extractelement <4 x float> %191, i64 1, !dbg !12
  %722 = extractelement <4 x float> %191, i64 2, !dbg !12
  %723 = extractelement <4 x float> %191, i64 3, !dbg !12
  %724 = select i1 %.not, float %721, float %720, !dbg !12
  %725 = bitcast float %724 to i32, !dbg !12
  %726 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %725), !dbg !12
  %727 = bitcast i32 %726 to float, !dbg !12
  %728 = select i1 %.not, float %720, float %727, !dbg !12
  %729 = select i1 %.not, float %727, float %721, !dbg !12
  %730 = select i1 %.not, float %723, float %722, !dbg !12
  %731 = bitcast float %730 to i32, !dbg !12
  %732 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %731), !dbg !12
  %733 = bitcast i32 %732 to float, !dbg !12
  %734 = select i1 %.not, float %722, float %733, !dbg !12
  %735 = select i1 %.not, float %733, float %723, !dbg !12
  %736 = select i1 %.not155, float %734, float %728, !dbg !12
  %737 = bitcast float %736 to i32, !dbg !12
  %738 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %737), !dbg !12
  %739 = bitcast i32 %738 to float, !dbg !12
  %740 = select i1 %.not155, float %728, float %739, !dbg !12
  %741 = select i1 %.not155, float %739, float %734, !dbg !12
  %742 = select i1 %.not155, float %735, float %729, !dbg !12
  %743 = bitcast float %742 to i32, !dbg !12
  %744 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %743), !dbg !12
  %745 = bitcast i32 %744 to float, !dbg !12
  %746 = select i1 %.not155, float %729, float %745, !dbg !12
  %747 = select i1 %.not155, float %745, float %735, !dbg !12
  %compare.local.base873 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 35) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local874 = or i32 %compare.local.base873, 16, !dbg !21
  %compare.low875 = icmp ult i32 %compare.local874, %compare.low.base857, !dbg !21
  %compare.result876 = select i1 %compare.high.equal855, i1 %compare.low875, i1 %compare.high.order856, !dbg !21
  %748 = select i1 %compare.result876, float 0.000000e+00, float -inf, !dbg !22
  %749 = fadd float %740, %748, !dbg !13
  %compare.local.base877 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 36) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local878 = or i32 %compare.local.base877, 17, !dbg !21
  %compare.low879 = icmp ult i32 %compare.local878, %compare.low.base857, !dbg !21
  %compare.result880 = select i1 %compare.high.equal855, i1 %compare.low879, i1 %compare.high.order856, !dbg !21
  %750 = select i1 %compare.result880, float 0.000000e+00, float -inf, !dbg !22
  %751 = fadd float %746, %750, !dbg !13
  %compare.local.base881 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 37) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local882 = or i32 %compare.local.base881, 18, !dbg !21
  %compare.low883 = icmp ult i32 %compare.local882, %compare.low.base857, !dbg !21
  %compare.result884 = select i1 %compare.high.equal855, i1 %compare.low883, i1 %compare.high.order856, !dbg !21
  %752 = select i1 %compare.result884, float 0.000000e+00, float -inf, !dbg !22
  %753 = fadd float %741, %752, !dbg !13
  %compare.local.base885 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 38) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local886 = or i32 %compare.local.base885, 19, !dbg !21
  %compare.low887 = icmp ult i32 %compare.local886, %compare.low.base857, !dbg !21
  %compare.result888 = select i1 %compare.high.equal855, i1 %compare.low887, i1 %compare.high.order856, !dbg !21
  %754 = select i1 %compare.result888, float 0.000000e+00, float -inf, !dbg !22
  %755 = fadd float %747, %754, !dbg !13
  %756 = tail call float @__ocml_exp2_f32(float %749), !dbg !23
  %757 = insertelement <4 x float> poison, float %756, i64 0, !dbg !24
  %758 = fadd float %719, %756, !dbg !25
  %759 = tail call float @__ocml_exp2_f32(float %751), !dbg !23
  %760 = insertelement <4 x float> %757, float %759, i64 1, !dbg !24
  %761 = fadd float %758, %759, !dbg !25
  %762 = tail call float @__ocml_exp2_f32(float %753), !dbg !23
  %763 = insertelement <4 x float> %760, float %762, i64 2, !dbg !24
  %764 = fadd float %761, %762, !dbg !25
  %765 = tail call float @__ocml_exp2_f32(float %755), !dbg !23
  %766 = insertelement <4 x float> %763, float %765, i64 3, !dbg !24
  %math.result257 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %766) #8, !dbg !24, !frisk.register_lifetimes !5
  %767 = fptrunc <4 x float> %math.result257 to <4 x half>, !dbg !24
  %768 = fadd float %764, %765, !dbg !25
  %769 = extractelement <4 x float> %194, i64 0, !dbg !12
  %770 = extractelement <4 x float> %194, i64 1, !dbg !12
  %771 = extractelement <4 x float> %194, i64 2, !dbg !12
  %772 = extractelement <4 x float> %194, i64 3, !dbg !12
  %773 = select i1 %.not, float %770, float %769, !dbg !12
  %774 = bitcast float %773 to i32, !dbg !12
  %775 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %774), !dbg !12
  %776 = bitcast i32 %775 to float, !dbg !12
  %777 = select i1 %.not, float %769, float %776, !dbg !12
  %778 = select i1 %.not, float %776, float %770, !dbg !12
  %779 = select i1 %.not, float %772, float %771, !dbg !12
  %780 = bitcast float %779 to i32, !dbg !12
  %781 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %780), !dbg !12
  %782 = bitcast i32 %781 to float, !dbg !12
  %783 = select i1 %.not, float %771, float %782, !dbg !12
  %784 = select i1 %.not, float %782, float %772, !dbg !12
  %785 = select i1 %.not155, float %783, float %777, !dbg !12
  %786 = bitcast float %785 to i32, !dbg !12
  %787 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %786), !dbg !12
  %788 = bitcast i32 %787 to float, !dbg !12
  %789 = select i1 %.not155, float %777, float %788, !dbg !12
  %790 = select i1 %.not155, float %788, float %783, !dbg !12
  %791 = select i1 %.not155, float %784, float %778, !dbg !12
  %792 = bitcast float %791 to i32, !dbg !12
  %793 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %792), !dbg !12
  %794 = bitcast i32 %793 to float, !dbg !12
  %795 = select i1 %.not155, float %778, float %794, !dbg !12
  %796 = select i1 %.not155, float %794, float %784, !dbg !12
  %compare.local.base889 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 39) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local890 = or i32 %compare.local.base889, 32, !dbg !21
  %compare.low891 = icmp ult i32 %compare.local890, %compare.low.base857, !dbg !21
  %compare.result892 = select i1 %compare.high.equal855, i1 %compare.low891, i1 %compare.high.order856, !dbg !21
  %797 = select i1 %compare.result892, float 0.000000e+00, float -inf, !dbg !22
  %798 = fadd float %789, %797, !dbg !13
  %compare.local.base893 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 40) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local894 = or i32 %compare.local.base893, 33, !dbg !21
  %compare.low895 = icmp ult i32 %compare.local894, %compare.low.base857, !dbg !21
  %compare.result896 = select i1 %compare.high.equal855, i1 %compare.low895, i1 %compare.high.order856, !dbg !21
  %799 = select i1 %compare.result896, float 0.000000e+00, float -inf, !dbg !22
  %800 = fadd float %795, %799, !dbg !13
  %compare.local.base897 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 41) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local898 = or i32 %compare.local.base897, 34, !dbg !21
  %compare.low899 = icmp ult i32 %compare.local898, %compare.low.base857, !dbg !21
  %compare.result900 = select i1 %compare.high.equal855, i1 %compare.low899, i1 %compare.high.order856, !dbg !21
  %801 = select i1 %compare.result900, float 0.000000e+00, float -inf, !dbg !22
  %802 = fadd float %790, %801, !dbg !13
  %compare.local.base901 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 42) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local902 = or i32 %compare.local.base901, 35, !dbg !21
  %compare.low903 = icmp ult i32 %compare.local902, %compare.low.base857, !dbg !21
  %compare.result904 = select i1 %compare.high.equal855, i1 %compare.low903, i1 %compare.high.order856, !dbg !21
  %803 = select i1 %compare.result904, float 0.000000e+00, float -inf, !dbg !22
  %804 = fadd float %796, %803, !dbg !13
  %805 = tail call float @__ocml_exp2_f32(float %798), !dbg !23
  %806 = insertelement <4 x float> poison, float %805, i64 0, !dbg !24
  %807 = fadd float %768, %805, !dbg !25
  %808 = tail call float @__ocml_exp2_f32(float %800), !dbg !23
  %809 = insertelement <4 x float> %806, float %808, i64 1, !dbg !24
  %810 = fadd float %807, %808, !dbg !25
  %811 = tail call float @__ocml_exp2_f32(float %802), !dbg !23
  %812 = insertelement <4 x float> %809, float %811, i64 2, !dbg !24
  %813 = fadd float %810, %811, !dbg !25
  %814 = tail call float @__ocml_exp2_f32(float %804), !dbg !23
  %815 = insertelement <4 x float> %812, float %814, i64 3, !dbg !24
  %math.result259 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %815) #8, !dbg !24, !frisk.register_lifetimes !5
  %816 = fptrunc <4 x float> %math.result259 to <4 x half>, !dbg !24
  %817 = fadd float %813, %814, !dbg !25
  %818 = extractelement <4 x float> %198, i64 0, !dbg !12
  %819 = extractelement <4 x float> %198, i64 1, !dbg !12
  %820 = extractelement <4 x float> %198, i64 2, !dbg !12
  %821 = extractelement <4 x float> %198, i64 3, !dbg !12
  %822 = select i1 %.not, float %819, float %818, !dbg !12
  %823 = bitcast float %822 to i32, !dbg !12
  %824 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %823), !dbg !12
  %825 = bitcast i32 %824 to float, !dbg !12
  %826 = select i1 %.not, float %818, float %825, !dbg !12
  %827 = select i1 %.not, float %825, float %819, !dbg !12
  %828 = select i1 %.not, float %821, float %820, !dbg !12
  %829 = bitcast float %828 to i32, !dbg !12
  %830 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %829), !dbg !12
  %831 = bitcast i32 %830 to float, !dbg !12
  %832 = select i1 %.not, float %820, float %831, !dbg !12
  %833 = select i1 %.not, float %831, float %821, !dbg !12
  %834 = select i1 %.not155, float %832, float %826, !dbg !12
  %835 = bitcast float %834 to i32, !dbg !12
  %836 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %835), !dbg !12
  %837 = bitcast i32 %836 to float, !dbg !12
  %838 = select i1 %.not155, float %826, float %837, !dbg !12
  %839 = select i1 %.not155, float %837, float %832, !dbg !12
  %840 = select i1 %.not155, float %833, float %827, !dbg !12
  %841 = bitcast float %840 to i32, !dbg !12
  %842 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %841), !dbg !12
  %843 = bitcast i32 %842 to float, !dbg !12
  %844 = select i1 %.not155, float %827, float %843, !dbg !12
  %845 = select i1 %.not155, float %843, float %833, !dbg !12
  %compare.local.base905 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 43) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local906 = or i32 %compare.local.base905, 48, !dbg !21
  %compare.low907 = icmp ult i32 %compare.local906, %compare.low.base857, !dbg !21
  %compare.result908 = select i1 %compare.high.equal855, i1 %compare.low907, i1 %compare.high.order856, !dbg !21
  %846 = select i1 %compare.result908, float 0.000000e+00, float -inf, !dbg !22
  %847 = fadd float %838, %846, !dbg !13
  %compare.local.base909 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 44) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local910 = or i32 %compare.local.base909, 49, !dbg !21
  %compare.low911 = icmp ult i32 %compare.local910, %compare.low.base857, !dbg !21
  %compare.result912 = select i1 %compare.high.equal855, i1 %compare.low911, i1 %compare.high.order856, !dbg !21
  %848 = select i1 %compare.result912, float 0.000000e+00, float -inf, !dbg !22
  %849 = fadd float %844, %848, !dbg !13
  %compare.local.base913 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 45) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local914 = or i32 %compare.local.base913, 50, !dbg !21
  %compare.low915 = icmp ult i32 %compare.local914, %compare.low.base857, !dbg !21
  %compare.result916 = select i1 %compare.high.equal855, i1 %compare.low915, i1 %compare.high.order856, !dbg !21
  %850 = select i1 %compare.result916, float 0.000000e+00, float -inf, !dbg !22
  %851 = fadd float %839, %850, !dbg !13
  %compare.local.base917 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 46) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local918 = or i32 %compare.local.base917, 51, !dbg !21
  %compare.low919 = icmp ult i32 %compare.local918, %compare.low.base857, !dbg !21
  %compare.result920 = select i1 %compare.high.equal855, i1 %compare.low919, i1 %compare.high.order856, !dbg !21
  %852 = select i1 %compare.result920, float 0.000000e+00, float -inf, !dbg !22
  %853 = fadd float %845, %852, !dbg !13
  %854 = tail call float @__ocml_exp2_f32(float %847), !dbg !23
  %855 = insertelement <4 x float> poison, float %854, i64 0, !dbg !24
  %856 = fadd float %817, %854, !dbg !25
  %857 = tail call float @__ocml_exp2_f32(float %849), !dbg !23
  %858 = insertelement <4 x float> %855, float %857, i64 1, !dbg !24
  %859 = fadd float %856, %857, !dbg !25
  %860 = tail call float @__ocml_exp2_f32(float %851), !dbg !23
  %861 = insertelement <4 x float> %858, float %860, i64 2, !dbg !24
  %862 = fadd float %859, %860, !dbg !25
  %863 = tail call float @__ocml_exp2_f32(float %853), !dbg !23
  %864 = insertelement <4 x float> %861, float %863, i64 3, !dbg !24
  %math.result261 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %864) #8, !dbg !24, !frisk.register_lifetimes !5
  %865 = fptrunc <4 x float> %math.result261 to <4 x half>, !dbg !24
  %866 = fadd float %862, %863, !dbg !25
  %867 = extractelement <4 x float> %202, i64 0, !dbg !12
  %868 = extractelement <4 x float> %202, i64 1, !dbg !12
  %869 = extractelement <4 x float> %202, i64 2, !dbg !12
  %870 = extractelement <4 x float> %202, i64 3, !dbg !12
  %871 = select i1 %.not, float %868, float %867, !dbg !12
  %872 = bitcast float %871 to i32, !dbg !12
  %873 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %872), !dbg !12
  %874 = bitcast i32 %873 to float, !dbg !12
  %875 = select i1 %.not, float %867, float %874, !dbg !12
  %876 = select i1 %.not, float %874, float %868, !dbg !12
  %877 = select i1 %.not, float %870, float %869, !dbg !12
  %878 = bitcast float %877 to i32, !dbg !12
  %879 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %878), !dbg !12
  %880 = bitcast i32 %879 to float, !dbg !12
  %881 = select i1 %.not, float %869, float %880, !dbg !12
  %882 = select i1 %.not, float %880, float %870, !dbg !12
  %883 = select i1 %.not155, float %881, float %875, !dbg !12
  %884 = bitcast float %883 to i32, !dbg !12
  %885 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %884), !dbg !12
  %886 = bitcast i32 %885 to float, !dbg !12
  %887 = select i1 %.not155, float %875, float %886, !dbg !12
  %888 = select i1 %.not155, float %886, float %881, !dbg !12
  %889 = select i1 %.not155, float %882, float %876, !dbg !12
  %890 = bitcast float %889 to i32, !dbg !12
  %891 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %890), !dbg !12
  %892 = bitcast i32 %891 to float, !dbg !12
  %893 = select i1 %.not155, float %876, float %892, !dbg !12
  %894 = select i1 %.not155, float %892, float %882, !dbg !12
  %compare.local.base921 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 47) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local922 = or i32 %compare.local.base921, 64, !dbg !21
  %compare.low923 = icmp ult i32 %compare.local922, %compare.low.base857, !dbg !21
  %compare.result924 = select i1 %compare.high.equal855, i1 %compare.low923, i1 %compare.high.order856, !dbg !21
  %895 = select i1 %compare.result924, float 0.000000e+00, float -inf, !dbg !22
  %896 = fadd float %887, %895, !dbg !13
  %compare.local.base925 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 48) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local926 = or i32 %compare.local.base925, 65, !dbg !21
  %compare.low927 = icmp ult i32 %compare.local926, %compare.low.base857, !dbg !21
  %compare.result928 = select i1 %compare.high.equal855, i1 %compare.low927, i1 %compare.high.order856, !dbg !21
  %897 = select i1 %compare.result928, float 0.000000e+00, float -inf, !dbg !22
  %898 = fadd float %893, %897, !dbg !13
  %compare.local.base929 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 49) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local930 = or i32 %compare.local.base929, 66, !dbg !21
  %compare.low931 = icmp ult i32 %compare.local930, %compare.low.base857, !dbg !21
  %compare.result932 = select i1 %compare.high.equal855, i1 %compare.low931, i1 %compare.high.order856, !dbg !21
  %899 = select i1 %compare.result932, float 0.000000e+00, float -inf, !dbg !22
  %900 = fadd float %888, %899, !dbg !13
  %compare.local.base933 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 50) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local934 = or i32 %compare.local.base933, 67, !dbg !21
  %compare.low935 = icmp ult i32 %compare.local934, %compare.low.base857, !dbg !21
  %compare.result936 = select i1 %compare.high.equal855, i1 %compare.low935, i1 %compare.high.order856, !dbg !21
  %901 = select i1 %compare.result936, float 0.000000e+00, float -inf, !dbg !22
  %902 = fadd float %894, %901, !dbg !13
  %903 = tail call float @__ocml_exp2_f32(float %896), !dbg !23
  %904 = insertelement <4 x float> poison, float %903, i64 0, !dbg !24
  %905 = fadd float %866, %903, !dbg !25
  %906 = tail call float @__ocml_exp2_f32(float %898), !dbg !23
  %907 = insertelement <4 x float> %904, float %906, i64 1, !dbg !24
  %908 = fadd float %905, %906, !dbg !25
  %909 = tail call float @__ocml_exp2_f32(float %900), !dbg !23
  %910 = insertelement <4 x float> %907, float %909, i64 2, !dbg !24
  %911 = fadd float %908, %909, !dbg !25
  %912 = tail call float @__ocml_exp2_f32(float %902), !dbg !23
  %913 = insertelement <4 x float> %910, float %912, i64 3, !dbg !24
  %math.result263 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %913) #8, !dbg !24, !frisk.register_lifetimes !5
  %914 = fptrunc <4 x float> %math.result263 to <4 x half>, !dbg !24
  %915 = fadd float %911, %912, !dbg !25
  %916 = extractelement <4 x float> %206, i64 0, !dbg !12
  %917 = extractelement <4 x float> %206, i64 1, !dbg !12
  %918 = extractelement <4 x float> %206, i64 2, !dbg !12
  %919 = extractelement <4 x float> %206, i64 3, !dbg !12
  %920 = select i1 %.not, float %917, float %916, !dbg !12
  %921 = bitcast float %920 to i32, !dbg !12
  %922 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %921), !dbg !12
  %923 = bitcast i32 %922 to float, !dbg !12
  %924 = select i1 %.not, float %916, float %923, !dbg !12
  %925 = select i1 %.not, float %923, float %917, !dbg !12
  %926 = select i1 %.not, float %919, float %918, !dbg !12
  %927 = bitcast float %926 to i32, !dbg !12
  %928 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %927), !dbg !12
  %929 = bitcast i32 %928 to float, !dbg !12
  %930 = select i1 %.not, float %918, float %929, !dbg !12
  %931 = select i1 %.not, float %929, float %919, !dbg !12
  %932 = select i1 %.not155, float %930, float %924, !dbg !12
  %933 = bitcast float %932 to i32, !dbg !12
  %934 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %933), !dbg !12
  %935 = bitcast i32 %934 to float, !dbg !12
  %936 = select i1 %.not155, float %924, float %935, !dbg !12
  %937 = select i1 %.not155, float %935, float %930, !dbg !12
  %938 = select i1 %.not155, float %931, float %925, !dbg !12
  %939 = bitcast float %938 to i32, !dbg !12
  %940 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %939), !dbg !12
  %941 = bitcast i32 %940 to float, !dbg !12
  %942 = select i1 %.not155, float %925, float %941, !dbg !12
  %943 = select i1 %.not155, float %941, float %931, !dbg !12
  %compare.local.base937 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 51) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local938 = or i32 %compare.local.base937, 80, !dbg !21
  %compare.low939 = icmp ult i32 %compare.local938, %compare.low.base857, !dbg !21
  %compare.result940 = select i1 %compare.high.equal855, i1 %compare.low939, i1 %compare.high.order856, !dbg !21
  %944 = select i1 %compare.result940, float 0.000000e+00, float -inf, !dbg !22
  %945 = fadd float %936, %944, !dbg !13
  %compare.local.base941 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 52) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local942 = or i32 %compare.local.base941, 81, !dbg !21
  %compare.low943 = icmp ult i32 %compare.local942, %compare.low.base857, !dbg !21
  %compare.result944 = select i1 %compare.high.equal855, i1 %compare.low943, i1 %compare.high.order856, !dbg !21
  %946 = select i1 %compare.result944, float 0.000000e+00, float -inf, !dbg !22
  %947 = fadd float %942, %946, !dbg !13
  %compare.local.base945 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 53) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local946 = or i32 %compare.local.base945, 82, !dbg !21
  %compare.low947 = icmp ult i32 %compare.local946, %compare.low.base857, !dbg !21
  %compare.result948 = select i1 %compare.high.equal855, i1 %compare.low947, i1 %compare.high.order856, !dbg !21
  %948 = select i1 %compare.result948, float 0.000000e+00, float -inf, !dbg !22
  %949 = fadd float %937, %948, !dbg !13
  %compare.local.base949 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 54) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local950 = or i32 %compare.local.base949, 83, !dbg !21
  %compare.low951 = icmp ult i32 %compare.local950, %compare.low.base857, !dbg !21
  %compare.result952 = select i1 %compare.high.equal855, i1 %compare.low951, i1 %compare.high.order856, !dbg !21
  %950 = select i1 %compare.result952, float 0.000000e+00, float -inf, !dbg !22
  %951 = fadd float %943, %950, !dbg !13
  %952 = tail call float @__ocml_exp2_f32(float %945), !dbg !23
  %953 = insertelement <4 x float> poison, float %952, i64 0, !dbg !24
  %954 = fadd float %915, %952, !dbg !25
  %955 = tail call float @__ocml_exp2_f32(float %947), !dbg !23
  %956 = insertelement <4 x float> %953, float %955, i64 1, !dbg !24
  %957 = fadd float %954, %955, !dbg !25
  %958 = tail call float @__ocml_exp2_f32(float %949), !dbg !23
  %959 = insertelement <4 x float> %956, float %958, i64 2, !dbg !24
  %960 = fadd float %957, %958, !dbg !25
  %961 = tail call float @__ocml_exp2_f32(float %951), !dbg !23
  %962 = insertelement <4 x float> %959, float %961, i64 3, !dbg !24
  %math.result265 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %962) #8, !dbg !24, !frisk.register_lifetimes !5
  %963 = fptrunc <4 x float> %math.result265 to <4 x half>, !dbg !24
  %964 = fadd float %960, %961, !dbg !25
  %965 = extractelement <4 x float> %210, i64 0, !dbg !12
  %966 = extractelement <4 x float> %210, i64 1, !dbg !12
  %967 = extractelement <4 x float> %210, i64 2, !dbg !12
  %968 = extractelement <4 x float> %210, i64 3, !dbg !12
  %969 = select i1 %.not, float %966, float %965, !dbg !12
  %970 = bitcast float %969 to i32, !dbg !12
  %971 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %970), !dbg !12
  %972 = bitcast i32 %971 to float, !dbg !12
  %973 = select i1 %.not, float %965, float %972, !dbg !12
  %974 = select i1 %.not, float %972, float %966, !dbg !12
  %975 = select i1 %.not, float %968, float %967, !dbg !12
  %976 = bitcast float %975 to i32, !dbg !12
  %977 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %976), !dbg !12
  %978 = bitcast i32 %977 to float, !dbg !12
  %979 = select i1 %.not, float %967, float %978, !dbg !12
  %980 = select i1 %.not, float %978, float %968, !dbg !12
  %981 = select i1 %.not155, float %979, float %973, !dbg !12
  %982 = bitcast float %981 to i32, !dbg !12
  %983 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %982), !dbg !12
  %984 = bitcast i32 %983 to float, !dbg !12
  %985 = select i1 %.not155, float %973, float %984, !dbg !12
  %986 = select i1 %.not155, float %984, float %979, !dbg !12
  %987 = select i1 %.not155, float %980, float %974, !dbg !12
  %988 = bitcast float %987 to i32, !dbg !12
  %989 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %988), !dbg !12
  %990 = bitcast i32 %989 to float, !dbg !12
  %991 = select i1 %.not155, float %974, float %990, !dbg !12
  %992 = select i1 %.not155, float %990, float %980, !dbg !12
  %compare.local.base953 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 55) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local954 = or i32 %compare.local.base953, 96, !dbg !21
  %compare.low955 = icmp ult i32 %compare.local954, %compare.low.base857, !dbg !21
  %compare.result956 = select i1 %compare.high.equal855, i1 %compare.low955, i1 %compare.high.order856, !dbg !21
  %993 = select i1 %compare.result956, float 0.000000e+00, float -inf, !dbg !22
  %994 = fadd float %985, %993, !dbg !13
  %compare.local.base957 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 56) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local958 = or i32 %compare.local.base957, 97, !dbg !21
  %compare.low959 = icmp ult i32 %compare.local958, %compare.low.base857, !dbg !21
  %compare.result960 = select i1 %compare.high.equal855, i1 %compare.low959, i1 %compare.high.order856, !dbg !21
  %995 = select i1 %compare.result960, float 0.000000e+00, float -inf, !dbg !22
  %996 = fadd float %991, %995, !dbg !13
  %compare.local.base961 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 57) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local962 = or i32 %compare.local.base961, 98, !dbg !21
  %compare.low963 = icmp ult i32 %compare.local962, %compare.low.base857, !dbg !21
  %compare.result964 = select i1 %compare.high.equal855, i1 %compare.low963, i1 %compare.high.order856, !dbg !21
  %997 = select i1 %compare.result964, float 0.000000e+00, float -inf, !dbg !22
  %998 = fadd float %986, %997, !dbg !13
  %compare.local.base965 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 58) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local966 = or i32 %compare.local.base965, 99, !dbg !21
  %compare.low967 = icmp ult i32 %compare.local966, %compare.low.base857, !dbg !21
  %compare.result968 = select i1 %compare.high.equal855, i1 %compare.low967, i1 %compare.high.order856, !dbg !21
  %999 = select i1 %compare.result968, float 0.000000e+00, float -inf, !dbg !22
  %1000 = fadd float %992, %999, !dbg !13
  %1001 = tail call float @__ocml_exp2_f32(float %994), !dbg !23
  %1002 = insertelement <4 x float> poison, float %1001, i64 0, !dbg !24
  %1003 = fadd float %964, %1001, !dbg !25
  %1004 = tail call float @__ocml_exp2_f32(float %996), !dbg !23
  %1005 = insertelement <4 x float> %1002, float %1004, i64 1, !dbg !24
  %1006 = fadd float %1003, %1004, !dbg !25
  %1007 = tail call float @__ocml_exp2_f32(float %998), !dbg !23
  %1008 = insertelement <4 x float> %1005, float %1007, i64 2, !dbg !24
  %1009 = fadd float %1006, %1007, !dbg !25
  %1010 = tail call float @__ocml_exp2_f32(float %1000), !dbg !23
  %1011 = insertelement <4 x float> %1008, float %1010, i64 3, !dbg !24
  %math.result267 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1011) #8, !dbg !24, !frisk.register_lifetimes !5
  %1012 = fptrunc <4 x float> %math.result267 to <4 x half>, !dbg !24
  %1013 = fadd float %1009, %1010, !dbg !25
  %1014 = extractelement <4 x float> %214, i64 0, !dbg !12
  %1015 = extractelement <4 x float> %214, i64 1, !dbg !12
  %1016 = extractelement <4 x float> %214, i64 2, !dbg !12
  %1017 = extractelement <4 x float> %214, i64 3, !dbg !12
  %1018 = select i1 %.not, float %1015, float %1014, !dbg !12
  %1019 = bitcast float %1018 to i32, !dbg !12
  %1020 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %1019), !dbg !12
  %1021 = bitcast i32 %1020 to float, !dbg !12
  %1022 = select i1 %.not, float %1014, float %1021, !dbg !12
  %1023 = select i1 %.not, float %1021, float %1015, !dbg !12
  %1024 = select i1 %.not, float %1017, float %1016, !dbg !12
  %1025 = bitcast float %1024 to i32, !dbg !12
  %1026 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %1025), !dbg !12
  %1027 = bitcast i32 %1026 to float, !dbg !12
  %1028 = select i1 %.not, float %1016, float %1027, !dbg !12
  %1029 = select i1 %.not, float %1027, float %1017, !dbg !12
  %1030 = select i1 %.not155, float %1028, float %1022, !dbg !12
  %1031 = bitcast float %1030 to i32, !dbg !12
  %1032 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %1031), !dbg !12
  %1033 = bitcast i32 %1032 to float, !dbg !12
  %1034 = select i1 %.not155, float %1022, float %1033, !dbg !12
  %1035 = select i1 %.not155, float %1033, float %1028, !dbg !12
  %1036 = select i1 %.not155, float %1029, float %1023, !dbg !12
  %1037 = bitcast float %1036 to i32, !dbg !12
  %1038 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %1037), !dbg !12
  %1039 = bitcast i32 %1038 to float, !dbg !12
  %1040 = select i1 %.not155, float %1023, float %1039, !dbg !12
  %1041 = select i1 %.not155, float %1039, float %1029, !dbg !12
  %compare.local.base969 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 59) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local970 = or i32 %compare.local.base969, 112, !dbg !21
  %compare.low971 = icmp ult i32 %compare.local970, %compare.low.base857, !dbg !21
  %compare.result972 = select i1 %compare.high.equal855, i1 %compare.low971, i1 %compare.high.order856, !dbg !21
  %1042 = select i1 %compare.result972, float 0.000000e+00, float -inf, !dbg !22
  %1043 = fadd float %1034, %1042, !dbg !13
  %compare.local.base973 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 60) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local974 = or i32 %compare.local.base973, 113, !dbg !21
  %compare.low975 = icmp ult i32 %compare.local974, %compare.low.base857, !dbg !21
  %compare.result976 = select i1 %compare.high.equal855, i1 %compare.low975, i1 %compare.high.order856, !dbg !21
  %1044 = select i1 %compare.result976, float 0.000000e+00, float -inf, !dbg !22
  %1045 = fadd float %1040, %1044, !dbg !13
  %compare.local.base977 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 61) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local978 = or i32 %compare.local.base977, 114, !dbg !21
  %compare.low979 = icmp ult i32 %compare.local978, %compare.low.base857, !dbg !21
  %compare.result980 = select i1 %compare.high.equal855, i1 %compare.low979, i1 %compare.high.order856, !dbg !21
  %1046 = select i1 %compare.result980, float 0.000000e+00, float -inf, !dbg !22
  %1047 = fadd float %1035, %1046, !dbg !13
  %compare.local.base981 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base730, i32 62) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local982 = or i32 %compare.local.base981, 115, !dbg !21
  %compare.low983 = icmp ult i32 %compare.local982, %compare.low.base857, !dbg !21
  %compare.result984 = select i1 %compare.high.equal855, i1 %compare.low983, i1 %compare.high.order856, !dbg !21
  %1048 = select i1 %compare.result984, float 0.000000e+00, float -inf, !dbg !22
  %1049 = fadd float %1041, %1048, !dbg !13
  %1050 = tail call float @__ocml_exp2_f32(float %1043), !dbg !23
  %1051 = insertelement <4 x float> poison, float %1050, i64 0, !dbg !24
  %1052 = fadd float %1013, %1050, !dbg !25
  %1053 = tail call float @__ocml_exp2_f32(float %1045), !dbg !23
  %1054 = insertelement <4 x float> %1051, float %1053, i64 1, !dbg !24
  %1055 = fadd float %1052, %1053, !dbg !25
  %1056 = tail call float @__ocml_exp2_f32(float %1047), !dbg !23
  %1057 = insertelement <4 x float> %1054, float %1056, i64 2, !dbg !24
  %1058 = fadd float %1055, %1056, !dbg !25
  %1059 = tail call float @__ocml_exp2_f32(float %1049), !dbg !23
  %1060 = insertelement <4 x float> %1057, float %1059, i64 3, !dbg !24
  %math.result269 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1060) #8, !dbg !24, !frisk.register_lifetimes !5
  %1061 = fptrunc <4 x float> %math.result269 to <4 x half>, !dbg !24
  %1062 = fadd float %1058, %1059, !dbg !25
  %1063 = bitcast float %670 to i32, !dbg !25
  %1064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %1063), !dbg !25
  %1065 = bitcast i32 %1064 to float, !dbg !25
  %1066 = fadd float %670, %1065, !dbg !25
  %1067 = bitcast float %1066 to i32, !dbg !25
  %1068 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %1067), !dbg !25
  %1069 = bitcast i32 %1068 to float, !dbg !25
  %1070 = extractvalue [2 x <1 x float>] %55, 0, !dbg !25
  %1071 = extractelement <1 x float> %1070, i64 0, !dbg !25
  %1072 = bitcast float %1062 to i32, !dbg !25
  %1073 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %277, i32 %1072), !dbg !25
  %1074 = bitcast i32 %1073 to float, !dbg !25
  %1075 = fadd float %1062, %1074, !dbg !25
  %1076 = bitcast float %1075 to i32, !dbg !25
  %1077 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %293, i32 %1076), !dbg !25
  %1078 = bitcast i32 %1077 to float, !dbg !25
  %1079 = extractvalue [2 x <1 x float>] %55, 1, !dbg !25
  %1080 = extractelement <1 x float> %1079, i64 0, !dbg !25
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  br label %1081, !dbg !20

1081:                                             ; preds = %234, %1081
  %1082 = phi i64 [ %18, %234 ], [ %1105, %1081 ]
  %1083 = lshr i64 %1082, 2, !dbg !20
  %1084 = lshr i64 %1082, 5, !dbg !20
  %1085 = lshr i64 %1082, 4, !dbg !20
  %1086 = lshr i64 %1082, 7, !dbg !20
  %1087 = sub nsw i64 %1085, %1086, !dbg !20
  %1088 = shl nsw i64 %1087, 2, !dbg !20
  %1089 = and i64 %1088, 4, !dbg !20
  %1090 = sub nsw i64 %1083, %.lobit43, !dbg !20
  %1091 = and i64 %1090, 3, !dbg !20
  %1092 = and i64 %1084, 24, !dbg !20
  %1093 = or disjoint i64 %1092, %43, !dbg !20
  %1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %48, i64 %1089, !dbg !20
  %1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1094, i64 %1091, !dbg !20
  %1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1095, i64 %44, !dbg !20
  %.idx230 = shl nuw nsw i64 %1093, 7, !dbg !20
  %1097 = getelementptr inbounds nuw i8, ptr addrspace(3) %1096, i64 %.idx230, !dbg !20
  %1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1097, i64 %46, !dbg !20
  %1099 = load half, ptr addrspace(3) %1098, align 2, !dbg !20
  %1100 = getelementptr inbounds nuw i8, ptr addrspace(3) %1098, i64 32, !dbg !20
  %1101 = load half, ptr addrspace(3) %1100, align 2, !dbg !20
  %1102 = insertelement <2 x half> poison, half %1099, i64 0, !dbg !20
  %1103 = insertelement <2 x half> %1102, half %1101, i64 1, !dbg !20
  %.idx233 = shl nuw nsw i64 %1082, 2, !dbg !20
  %1104 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx233, !dbg !20
  store <2 x half> %1103, ptr addrspace(3) %1104, align 4, !dbg !20
  %1105 = add nuw nsw i64 %1082, 256, !dbg !20
  %1106 = icmp samesign ult i64 %1082, 768, !dbg !20
  br i1 %1106, label %1081, label %1107, !dbg !20

1107:                                             ; preds = %1081
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %326) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed242 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %375) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed244 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %424) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed246 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %473) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed248 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %522) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed250 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %571) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed252 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %620) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed256 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %718) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed258 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %767) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed260 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %816) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed262 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %865) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed264 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %914) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed266 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %963) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed268 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1012) #8, !dbg !25, !frisk.register_lifetimes !5
  %math.packed270 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1061) #8, !dbg !25, !frisk.register_lifetimes !5
  %1108 = fadd float %1066, %1069, !dbg !25
  %1109 = fadd float %1075, %1078, !dbg !25
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %1110 = getelementptr i8, ptr addrspace(1) %240, i64 32, !dbg !20
  %address.thread487 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !20, !range !11, !frisk.register_lifetimes !5
  %1111 = shl nuw nsw i32 %address.thread487, 1, !dbg !9
  %1112 = and i32 %1111, 384, !dbg !9
  %address.local490 = zext nneg i32 %1112 to i64, !dbg !9
  %address.local491 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_1, i64 %address.local490, !dbg !20
  %address.local495 = and i32 %1111, 14, !dbg !17
  %address.local496 = shl nuw nsw i32 %address.thread487, 4, !dbg !17
  %address.local497 = and i32 %address.local496, 3968, !dbg !17
  %address.local498 = or disjoint i32 %address.local495, %address.local497, !dbg !17
  %address.local499 = zext nneg i32 %address.local498 to i64, !dbg !17
  %address.local500 = getelementptr [2 x i8], ptr addrspace(1) %1110, i64 %address.local499, !dbg !20
  %dma.source.bits1018 = ptrtoint ptr addrspace(1) %address.local500 to i64, !dbg !20
  %dma.source.opaque1019 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1018) #9, !dbg !20
  %dma.source.ptr1020 = inttoptr i64 %dma.source.opaque1019 to ptr addrspace(1), !dbg !20
  %address.local504 = getelementptr i8, ptr addrspace(3) %address.local491, i64 1024, !dbg !20
  %address.local514 = getelementptr i8, ptr addrspace(1) %address.local500, i64 8192, !dbg !20
  %dma.source.bits1021 = ptrtoint ptr addrspace(1) %address.local514 to i64, !dbg !20
  %dma.source.opaque1022 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1021) #9, !dbg !20
  %dma.source.ptr1023 = inttoptr i64 %dma.source.opaque1022 to ptr addrspace(1), !dbg !20
  %address.local518 = getelementptr i8, ptr addrspace(3) %address.local491, i64 2048, !dbg !20
  %address.local528 = getelementptr i8, ptr addrspace(1) %address.local500, i64 16384, !dbg !20
  %dma.source.bits1024 = ptrtoint ptr addrspace(1) %address.local528 to i64, !dbg !20
  %dma.source.opaque1025 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1024) #9, !dbg !20
  %dma.source.ptr1026 = inttoptr i64 %dma.source.opaque1025 to ptr addrspace(1), !dbg !20
  %address.local532 = getelementptr i8, ptr addrspace(3) %address.local491, i64 3072, !dbg !20
  %1113 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local491, i64 0, !dbg !20
  %1114 = insertelement <4 x ptr addrspace(3)> %1113, ptr addrspace(3) %address.local504, i64 1, !dbg !20
  %1115 = insertelement <4 x ptr addrspace(3)> %1114, ptr addrspace(3) %address.local518, i64 2, !dbg !20
  %1116 = insertelement <4 x ptr addrspace(3)> %1115, ptr addrspace(3) %address.local532, i64 3, !dbg !20
  %1117 = ptrtoint <4 x ptr addrspace(3)> %1116 to <4 x i64>, !dbg !20
  %1118 = trunc <4 x i64> %1117 to <4 x i32>, !dbg !20
  %1119 = extractelement <4 x i32> %1118, i64 0, !dbg !20
  %1120 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1119), !dbg !20
  %1121 = zext i32 %1120 to i64, !dbg !20
  %1122 = inttoptr i64 %1121 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1020, ptr addrspace(3) %1122, i32 4, i32 0, i32 0), !dbg !20
  %1123 = extractelement <4 x i32> %1118, i64 1, !dbg !20
  %1124 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1123), !dbg !20
  %1125 = zext i32 %1124 to i64, !dbg !20
  %1126 = inttoptr i64 %1125 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1023, ptr addrspace(3) %1126, i32 4, i32 0, i32 0), !dbg !20
  %1127 = extractelement <4 x i32> %1118, i64 2, !dbg !20
  %1128 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1127), !dbg !20
  %1129 = zext i32 %1128 to i64, !dbg !20
  %1130 = inttoptr i64 %1129 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1026, ptr addrspace(3) %1130, i32 4, i32 0, i32 0), !dbg !20
  %1131 = extractelement <4 x i32> %1118, i64 3, !dbg !20
  %1132 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %1131), !dbg !20
  %1133 = zext i32 %1132 to i64, !dbg !20
  %1134 = inttoptr i64 %1133 to ptr addrspace(3), !dbg !20
  %address.local542 = getelementptr i8, ptr addrspace(1) %address.local500, i64 24576, !dbg !20
  %dma.source.bits1027 = ptrtoint ptr addrspace(1) %address.local542 to i64, !dbg !20
  %dma.source.opaque1028 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1027) #9, !dbg !20
  %dma.source.ptr1029 = inttoptr i64 %dma.source.opaque1028 to ptr addrspace(1), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1029, ptr addrspace(3) %1134, i32 4, i32 0, i32 0), !dbg !20
  tail call void @llvm.amdgcn.asyncmark(), !dbg !20
  %address.local543 = lshr i32 %address.thread487, 5, !dbg !26
  %address.local544 = and i32 %address.local543, 1, !dbg !26
  %address.local545 = lshr i32 %address.thread487, 2, !dbg !26
  %address.local546 = and i32 %address.local545, 3, !dbg !26
  %address.local547 = add nuw nsw i32 %address.local544, %address.local546, !dbg !26
  %address.local548 = lshr i32 %address.local547, 1, !dbg !26
  %address.local549 = zext nneg i32 %address.local548 to i64, !dbg !26
  %address.local550 = mul nsw i64 %address.local549, -64, !dbg !26
  %address.local551 = lshr i32 %address.thread487, 3, !dbg !26
  %address.local552 = and i32 %address.local551, 1, !dbg !26
  %address.local553 = mul nuw nsw i32 %address.local552, 72, !dbg !26
  %address.local554 = shl nuw nsw i32 %address.thread487, 3, !dbg !26
  %address.local555 = and i32 %address.local554, 504, !dbg !26
  %address.local558 = and i32 %address.local554, 96, !dbg !26
  %address.local560 = and i32 %address.thread487, 32, !dbg !26
  %address.local556 = add nuw nsw i32 %address.local555, %address.local560, !dbg !26
  %address.local559 = add nuw nsw i32 %address.local556, %address.local558, !dbg !26
  %address.local561 = add nuw nsw i32 %address.local559, %address.local553, !dbg !26
  %address.local562 = zext nneg i32 %address.local561 to i64, !dbg !26
  %address.local564 = and i32 %address.thread487, 15, !dbg !26
  %address.local565 = add nuw nsw i32 %address.local552, %address.local564, !dbg !26
  %address.local566 = lshr i32 %address.local565, 2, !dbg !26
  %address.local567 = zext nneg i32 %address.local566 to i64, !dbg !26
  %address.local568 = mul nsw i64 %address.local567, -32, !dbg !26
  %1135 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local550, !dbg !26
  %1136 = getelementptr [2 x i8], ptr addrspace(3) %1135, i64 %address.local562, !dbg !26
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) %1136, i64 %address.local568, !dbg !26
  %1137 = load <8 x half>, ptr addrspace(3) %address.local570, align 2, !dbg !26
  %1138 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1138, <4 x float> zeroinitializer) #8, !dbg !26
  %1140 = shufflevector <8 x half> %1137, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1140, <4 x float> %1139) #8, !dbg !26
  %address.local572 = and i32 %address.local554, 1656, !dbg !26
  %address.local574 = and i32 %address.local496, 768, !dbg !26
  %address.local575 = or disjoint i32 %address.local574, %address.local560, !dbg !26
  %address.local576 = add nuw nsw i32 %address.local575, %address.local572, !dbg !26
  %address.local577 = zext nneg i32 %address.local576 to i64, !dbg !26
  %address.local579 = and i32 %address.local554, 1920, !dbg !26
  %address.local580 = sub nsw i32 0, %address.local579, !dbg !26
  %address.local581 = sext i32 %address.local580 to i64, !dbg !26
  %address.local583 = zext nneg i32 %address.local558 to i64, !dbg !26
  %address.local585 = zext nneg i32 %address.local553 to i64, !dbg !26
  %1142 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local577, !dbg !26
  %1143 = getelementptr [2 x i8], ptr addrspace(3) %1142, i64 %address.local581, !dbg !26
  %1144 = getelementptr [2 x i8], ptr addrspace(3) %1143, i64 %address.local583, !dbg !26
  %1145 = getelementptr [2 x i8], ptr addrspace(3) %1144, i64 %address.local585, !dbg !26
  %1146 = getelementptr [2 x i8], ptr addrspace(3) %1145, i64 %address.local550, !dbg !26
  %1147 = getelementptr [2 x i8], ptr addrspace(3) %1146, i64 %address.local568, !dbg !26
  %address.local590 = getelementptr i8, ptr addrspace(3) %1147, i64 1024, !dbg !26
  %1148 = load <8 x half>, ptr addrspace(3) %address.local590, align 2, !dbg !26
  %address.local592 = getelementptr i8, ptr addrspace(3) %1147, i64 2048, !dbg !26
  %1149 = load <8 x half>, ptr addrspace(3) %address.local592, align 2, !dbg !26
  %address.local594 = getelementptr i8, ptr addrspace(3) %1147, i64 3072, !dbg !26
  %1150 = load <8 x half>, ptr addrspace(3) %address.local594, align 2, !dbg !26
  %1151 = shufflevector <8 x half> %1148, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1151, <4 x float> %1141) #8, !dbg !26
  %1153 = shufflevector <8 x half> %1148, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1153, <4 x float> %1152) #8, !dbg !26
  %1155 = shufflevector <8 x half> %1149, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1155, <4 x float> %1154) #8, !dbg !26
  %1157 = shufflevector <8 x half> %1149, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1157, <4 x float> %1156) #8, !dbg !26
  %1159 = shufflevector <8 x half> %1150, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1160 = shufflevector <8 x half> %1150, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1159, <4 x float> %1158) #8, !dbg !26
  %1162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1138, <4 x float> zeroinitializer) #8, !dbg !26
  %1163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1140, <4 x float> %1162) #8, !dbg !26
  %1164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1151, <4 x float> %1163) #8, !dbg !26
  %1165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1153, <4 x float> %1164) #8, !dbg !26
  %1166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1155, <4 x float> %1165) #8, !dbg !26
  %1167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1157, <4 x float> %1166) #8, !dbg !26
  %1168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1159, <4 x float> %1167) #8, !dbg !26
  %1169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1160, <4 x float> %1168) #8, !dbg !26
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  br label %1170, !dbg !20

1170:                                             ; preds = %1107, %1170
  %1171 = phi i64 [ %18, %1107 ], [ %1194, %1170 ]
  %1172 = lshr i64 %1171, 2, !dbg !20
  %1173 = lshr i64 %1171, 5, !dbg !20
  %1174 = lshr i64 %1171, 4, !dbg !20
  %1175 = lshr i64 %1171, 7, !dbg !20
  %1176 = sub nsw i64 %1174, %1175, !dbg !20
  %1177 = shl nsw i64 %1176, 2, !dbg !20
  %1178 = and i64 %1177, 4, !dbg !20
  %1179 = sub nsw i64 %1172, %.lobit43, !dbg !20
  %1180 = and i64 %1179, 3, !dbg !20
  %1181 = and i64 %1173, 24, !dbg !20
  %1182 = or disjoint i64 %1181, %43, !dbg !20
  %1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %48, i64 %1178, !dbg !20
  %1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1183, i64 %1180, !dbg !20
  %1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1184, i64 %44, !dbg !20
  %.idx = shl nuw nsw i64 %1182, 7, !dbg !20
  %1186 = getelementptr inbounds nuw i8, ptr addrspace(3) %1185, i64 %.idx, !dbg !20
  %1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %1186, i64 %46, !dbg !20
  %1188 = load half, ptr addrspace(3) %1187, align 2, !dbg !20
  %1189 = getelementptr inbounds nuw i8, ptr addrspace(3) %1187, i64 32, !dbg !20
  %1190 = load half, ptr addrspace(3) %1189, align 2, !dbg !20
  %1191 = insertelement <2 x half> poison, half %1188, i64 0, !dbg !20
  %1192 = insertelement <2 x half> %1191, half %1190, i64 1, !dbg !20
  %.idx228 = shl nuw nsw i64 %1171, 2, !dbg !20
  %1193 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %.idx228, !dbg !20
  store <2 x half> %1192, ptr addrspace(3) %1193, align 4, !dbg !20
  %1194 = add nuw nsw i64 %1171, 256, !dbg !20
  %1195 = icmp samesign ult i64 %1171, 768, !dbg !20
  br i1 %1195, label %1170, label %1196, !dbg !20

1196:                                             ; preds = %1170
  %math.packed254 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %669) #8, !dbg !25, !frisk.register_lifetimes !5
  %1197 = fadd float %1071, %1108, !dbg !25
  %1198 = fadd float %1080, %1109, !dbg !25
  %1199 = extractvalue [2 x <4 x float>] %56, 0, !dbg !27
  %1200 = extractvalue [2 x <4 x float>] %56, 1, !dbg !27
  %1201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1160, <4 x float> %1161) #8, !dbg !26
  %1202 = fadd <4 x float> %1199, %1201, !dbg !27
  %1203 = fadd <4 x float> %1200, %1169, !dbg !27
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.thread595 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !26, !range !11, !frisk.register_lifetimes !5
  %address.local596 = lshr i32 %address.thread595, 5, !dbg !26
  %address.local597 = and i32 %address.local596, 1, !dbg !26
  %address.local598 = lshr i32 %address.thread595, 2, !dbg !26
  %address.local599 = and i32 %address.local598, 3, !dbg !26
  %address.local600 = add nuw nsw i32 %address.local597, %address.local599, !dbg !26
  %address.local601 = lshr i32 %address.local600, 1, !dbg !26
  %address.local602 = zext nneg i32 %address.local601 to i64, !dbg !26
  %address.local603 = mul nsw i64 %address.local602, -64, !dbg !26
  %address.local604 = lshr i32 %address.thread595, 3, !dbg !26
  %address.local605 = and i32 %address.local604, 1, !dbg !26
  %address.local606 = mul nuw nsw i32 %address.local605, 72, !dbg !26
  %address.local607 = shl nuw nsw i32 %address.thread595, 3, !dbg !26
  %address.local608 = and i32 %address.local607, 504, !dbg !26
  %address.local611 = and i32 %address.local607, 96, !dbg !26
  %address.local613 = and i32 %address.thread595, 32, !dbg !26
  %address.local609 = add nuw nsw i32 %address.local608, %address.local613, !dbg !26
  %address.local612 = add nuw nsw i32 %address.local609, %address.local611, !dbg !26
  %address.local614 = add nuw nsw i32 %address.local612, %address.local606, !dbg !26
  %address.local615 = zext nneg i32 %address.local614 to i64, !dbg !26
  %address.local617 = and i32 %address.thread595, 15, !dbg !26
  %address.local618 = add nuw nsw i32 %address.local605, %address.local617, !dbg !26
  %address.local619 = lshr i32 %address.local618, 2, !dbg !26
  %address.local620 = zext nneg i32 %address.local619 to i64, !dbg !26
  %address.local621 = mul nsw i64 %address.local620, -32, !dbg !26
  %1204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local603, !dbg !26
  %1205 = getelementptr [2 x i8], ptr addrspace(3) %1204, i64 %address.local615, !dbg !26
  %address.local623 = getelementptr [2 x i8], ptr addrspace(3) %1205, i64 %address.local621, !dbg !26
  %1206 = load <8 x half>, ptr addrspace(3) %address.local623, align 2, !dbg !26
  %1207 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1207, <4 x float> zeroinitializer) #8, !dbg !26
  %1209 = shufflevector <8 x half> %1206, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1210 = extractvalue [2 x <4 x float>] %54, 0, !dbg !27
  %1211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed242, <4 x half> %1209, <4 x float> %1208) #8, !dbg !26
  %1212 = extractvalue [2 x <4 x float>] %54, 1, !dbg !27
  %address.local625 = and i32 %address.local607, 1656, !dbg !26
  %address.local626 = shl nuw nsw i32 %address.thread595, 4, !dbg !26
  %address.local627 = and i32 %address.local626, 768, !dbg !26
  %address.local628 = or disjoint i32 %address.local627, %address.local613, !dbg !26
  %address.local629 = add nuw nsw i32 %address.local628, %address.local625, !dbg !26
  %address.local630 = zext nneg i32 %address.local629 to i64, !dbg !26
  %address.local632 = and i32 %address.local607, 1920, !dbg !26
  %address.local633 = sub nsw i32 0, %address.local632, !dbg !26
  %address.local634 = sext i32 %address.local633 to i64, !dbg !26
  %address.local636 = zext nneg i32 %address.local611 to i64, !dbg !26
  %address.local638 = zext nneg i32 %address.local606 to i64, !dbg !26
  %1213 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 4096), i64 %address.local630, !dbg !26
  %1214 = getelementptr [2 x i8], ptr addrspace(3) %1213, i64 %address.local634, !dbg !26
  %1215 = getelementptr [2 x i8], ptr addrspace(3) %1214, i64 %address.local636, !dbg !26
  %1216 = getelementptr [2 x i8], ptr addrspace(3) %1215, i64 %address.local638, !dbg !26
  %1217 = getelementptr [2 x i8], ptr addrspace(3) %1216, i64 %address.local603, !dbg !26
  %1218 = getelementptr [2 x i8], ptr addrspace(3) %1217, i64 %address.local621, !dbg !26
  %address.local643 = getelementptr i8, ptr addrspace(3) %1218, i64 1024, !dbg !26
  %1219 = load <8 x half>, ptr addrspace(3) %address.local643, align 2, !dbg !26
  %address.local645 = getelementptr i8, ptr addrspace(3) %1218, i64 2048, !dbg !26
  %1220 = load <8 x half>, ptr addrspace(3) %address.local645, align 2, !dbg !26
  %address.local647 = getelementptr i8, ptr addrspace(3) %1218, i64 3072, !dbg !26
  %1221 = load <8 x half>, ptr addrspace(3) %address.local647, align 2, !dbg !26
  %1222 = insertvalue [2 x <4 x float>] poison, <4 x float> %1202, 0, !dbg !16
  %1223 = insertvalue [2 x <4 x float>] %1222, <4 x float> %1203, 1, !dbg !16
  %1224 = shufflevector <8 x half> %1219, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed244, <4 x half> %1224, <4 x float> %1211) #8, !dbg !26
  %1226 = shufflevector <8 x half> %1219, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1226, <4 x float> %1225) #8, !dbg !26
  %1228 = shufflevector <8 x half> %1220, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1228, <4 x float> %1227) #8, !dbg !26
  %1230 = shufflevector <8 x half> %1220, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1230, <4 x float> %1229) #8, !dbg !26
  %1232 = shufflevector <8 x half> %1221, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1233 = shufflevector <8 x half> %1221, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1232, <4 x float> %1231) #8, !dbg !26
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1233, <4 x float> %1234) #8, !dbg !26
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1207, <4 x float> zeroinitializer) #8, !dbg !26
  %1237 = fadd <4 x float> %1210, %1235, !dbg !27
  %1238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1209, <4 x float> %1236) #8, !dbg !26
  %1239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1224, <4 x float> %1238) #8, !dbg !26
  %1240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1226, <4 x float> %1239) #8, !dbg !26
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1228, <4 x float> %1240) #8, !dbg !26
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1230, <4 x float> %1241) #8, !dbg !26
  %1243 = insertelement <1 x float> poison, float %1197, i64 0, !dbg !16
  %1244 = insertvalue [2 x <1 x float>] poison, <1 x float> %1243, 0, !dbg !16
  %1245 = insertelement <1 x float> poison, float %1198, i64 0, !dbg !16
  %1246 = insertvalue [2 x <1 x float>] %1244, <1 x float> %1245, 1, !dbg !16
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1232, <4 x float> %1242) #8, !dbg !26
  %1248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1233, <4 x float> %1247) #8, !dbg !26
  %1249 = fadd <4 x float> %1212, %1248, !dbg !27
  %1250 = insertvalue [2 x <4 x float>] poison, <4 x float> %1237, 0, !dbg !16
  %1251 = insertvalue [2 x <4 x float>] %1250, <4 x float> %1249, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1252 = add nuw nsw i64 %57, 1, !dbg !16
  %.not161.not = icmp slt i64 %57, %6, !dbg !16
  br i1 %.not161.not, label %.lr.ph, label %._crit_edge, !dbg !16

._crit_edge:                                      ; preds = %1196, %.._crit_edge_crit_edge
  %.pre-phi41 = phi i32 [ %.pre40, %.._crit_edge_crit_edge ], [ %290, %1196 ], !dbg !16
  %.pre-phi39 = phi i32 [ %.pre38, %.._crit_edge_crit_edge ], [ %274, %1196 ], !dbg !16
  %.pre-phi37 = phi i32 [ %.pre36, %.._crit_edge_crit_edge ], [ %273, %1196 ], !dbg !16
  %.pre-phi33 = phi i32 [ %.pre32, %.._crit_edge_crit_edge ], [ %271, %1196 ], !dbg !16
  %.lcssa10 = phi [2 x <4 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1223, %1196 ], !dbg !16
  %.lcssa9 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1246, %1196 ], !dbg !16
  %.lcssa8 = phi [2 x <4 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1251, %1196 ], !dbg !16
  %1253 = extractvalue [2 x <4 x float>] %.lcssa8, 0, !dbg !16
  %1254 = extractelement <4 x float> %1253, i64 0, !dbg !16
  %1255 = extractelement <4 x float> %1253, i64 1, !dbg !16
  %1256 = extractelement <4 x float> %1253, i64 2, !dbg !16
  %1257 = extractelement <4 x float> %1253, i64 3, !dbg !16
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !16
  %1259 = icmp samesign ult i32 %.pre-phi37, %.pre-phi39, !dbg !16
  %1260 = select i1 %1259, i32 %.pre-phi37, i32 %.pre-phi33, !dbg !16
  %1261 = shl nuw nsw i32 %1260, 2, !dbg !16
  %1262 = bitcast float %1258 to i32, !dbg !16
  %1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1262), !dbg !16
  %1264 = bitcast i32 %1263 to float, !dbg !16
  %1265 = select i1 %.not, float %1254, float %1264, !dbg !16
  %1266 = select i1 %.not, float %1264, float %1255, !dbg !16
  %1267 = select i1 %.not, float %1257, float %1256, !dbg !16
  %1268 = bitcast float %1267 to i32, !dbg !16
  %1269 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1268), !dbg !16
  %1270 = bitcast i32 %1269 to float, !dbg !16
  %1271 = select i1 %.not, float %1256, float %1270, !dbg !16
  %1272 = select i1 %.not, float %1270, float %1257, !dbg !16
  %1273 = select i1 %.not155, float %1271, float %1265, !dbg !16
  %1274 = icmp samesign ult i32 %.pre-phi41, %.pre-phi39, !dbg !16
  %1275 = select i1 %1274, i32 %.pre-phi41, i32 %.pre-phi33, !dbg !16
  %1276 = shl nuw nsw i32 %1275, 2, !dbg !16
  %1277 = bitcast float %1273 to i32, !dbg !16
  %1278 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1277), !dbg !16
  %1279 = bitcast i32 %1278 to float, !dbg !16
  %1280 = select i1 %.not155, float %1272, float %1266, !dbg !16
  %1281 = bitcast float %1280 to i32, !dbg !16
  %1282 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1281), !dbg !16
  %1283 = bitcast i32 %1282 to float, !dbg !16
  %1284 = insertelement <4 x float> poison, float %1265, i64 0, !dbg !16
  %1285 = insertelement <4 x float> %1284, float %1266, i64 1, !dbg !16
  %1286 = insertelement <4 x float> %1285, float %1279, i64 2, !dbg !16
  %1287 = insertelement <4 x float> %1286, float %1283, i64 3, !dbg !16
  %1288 = insertelement <4 x float> poison, float %1279, i64 0, !dbg !16
  %1289 = insertelement <4 x float> %1288, float %1283, i64 1, !dbg !16
  %1290 = insertelement <4 x float> %1289, float %1271, i64 2, !dbg !16
  %1291 = insertelement <4 x float> %1290, float %1272, i64 3, !dbg !16
  %1292 = select <4 x i1> %50, <4 x float> %1287, <4 x float> %1291, !dbg !16
  %1293 = extractvalue [2 x <4 x float>] %.lcssa10, 0, !dbg !16
  %1294 = extractelement <4 x float> %1293, i64 0, !dbg !16
  %1295 = extractelement <4 x float> %1293, i64 1, !dbg !16
  %1296 = extractelement <4 x float> %1293, i64 2, !dbg !16
  %1297 = extractelement <4 x float> %1293, i64 3, !dbg !16
  %1298 = select i1 %.not, float %1295, float %1294, !dbg !16
  %1299 = bitcast float %1298 to i32, !dbg !16
  %1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1299), !dbg !16
  %1301 = bitcast i32 %1300 to float, !dbg !16
  %1302 = select i1 %.not, float %1294, float %1301, !dbg !16
  %1303 = select i1 %.not, float %1301, float %1295, !dbg !16
  %1304 = select i1 %.not, float %1297, float %1296, !dbg !16
  %1305 = bitcast float %1304 to i32, !dbg !16
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1305), !dbg !16
  %1307 = bitcast i32 %1306 to float, !dbg !16
  %1308 = select i1 %.not, float %1296, float %1307, !dbg !16
  %1309 = select i1 %.not, float %1307, float %1297, !dbg !16
  %1310 = select i1 %.not155, float %1308, float %1302, !dbg !16
  %1311 = bitcast float %1310 to i32, !dbg !16
  %1312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1311), !dbg !16
  %1313 = select i1 %.not155, float %1309, float %1303, !dbg !16
  %1314 = bitcast float %1313 to i32, !dbg !16
  %1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1314), !dbg !16
  %1316 = extractvalue [2 x <1 x float>] %.lcssa9, 0, !dbg !28
  %1317 = shufflevector <1 x float> %1316, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1318 = extractvalue [2 x <4 x float>] %.lcssa8, 1, !dbg !16
  %1319 = extractelement <4 x float> %1318, i64 0, !dbg !16
  %1320 = extractelement <4 x float> %1318, i64 1, !dbg !16
  %1321 = extractelement <4 x float> %1318, i64 2, !dbg !16
  %1322 = extractelement <4 x float> %1318, i64 3, !dbg !16
  %1323 = select i1 %.not, float %1320, float %1319, !dbg !16
  %1324 = bitcast float %1323 to i32, !dbg !16
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1324), !dbg !16
  %1326 = bitcast i32 %1325 to float, !dbg !16
  %1327 = select i1 %.not, float %1319, float %1326, !dbg !16
  %1328 = select i1 %.not, float %1326, float %1320, !dbg !16
  %1329 = select i1 %.not, float %1322, float %1321, !dbg !16
  %1330 = bitcast float %1329 to i32, !dbg !16
  %1331 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1330), !dbg !16
  %1332 = bitcast i32 %1331 to float, !dbg !16
  %1333 = select i1 %.not, float %1321, float %1332, !dbg !16
  %1334 = select i1 %.not, float %1332, float %1322, !dbg !16
  %1335 = select i1 %.not155, float %1333, float %1327, !dbg !16
  %1336 = bitcast float %1335 to i32, !dbg !16
  %1337 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1336), !dbg !16
  %1338 = bitcast i32 %1337 to float, !dbg !16
  %1339 = select i1 %.not155, float %1334, float %1328, !dbg !16
  %1340 = bitcast float %1339 to i32, !dbg !16
  %1341 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1340), !dbg !16
  %1342 = bitcast i32 %1341 to float, !dbg !16
  %1343 = insertelement <4 x float> poison, float %1327, i64 0, !dbg !16
  %1344 = insertelement <4 x float> %1343, float %1328, i64 1, !dbg !16
  %1345 = insertelement <4 x float> %1344, float %1338, i64 2, !dbg !16
  %1346 = insertelement <4 x float> %1345, float %1342, i64 3, !dbg !16
  %1347 = insertelement <4 x float> poison, float %1338, i64 0, !dbg !16
  %1348 = insertelement <4 x float> %1347, float %1342, i64 1, !dbg !16
  %1349 = insertelement <4 x float> %1348, float %1333, i64 2, !dbg !16
  %1350 = insertelement <4 x float> %1349, float %1334, i64 3, !dbg !16
  %1351 = select <4 x i1> %50, <4 x float> %1346, <4 x float> %1350, !dbg !16
  %1352 = extractvalue [2 x <4 x float>] %.lcssa10, 1, !dbg !16
  %1353 = extractelement <4 x float> %1352, i64 0, !dbg !16
  %1354 = extractelement <4 x float> %1352, i64 1, !dbg !16
  %1355 = extractelement <4 x float> %1352, i64 2, !dbg !16
  %1356 = extractelement <4 x float> %1352, i64 3, !dbg !16
  %1357 = select i1 %.not, float %1354, float %1353, !dbg !16
  %1358 = bitcast float %1357 to i32, !dbg !16
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1358), !dbg !16
  %1360 = bitcast i32 %1359 to float, !dbg !16
  %1361 = select i1 %.not, float %1353, float %1360, !dbg !16
  %1362 = select i1 %.not, float %1360, float %1354, !dbg !16
  %1363 = select i1 %.not, float %1356, float %1355, !dbg !16
  %1364 = bitcast float %1363 to i32, !dbg !16
  %1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1261, i32 %1364), !dbg !16
  %1366 = bitcast i32 %1365 to float, !dbg !16
  %1367 = select i1 %.not, float %1355, float %1366, !dbg !16
  %1368 = select i1 %.not, float %1366, float %1356, !dbg !16
  %1369 = select i1 %.not155, float %1367, float %1361, !dbg !16
  %1370 = bitcast float %1369 to i32, !dbg !16
  %1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1370), !dbg !16
  %1372 = select i1 %.not155, float %1368, float %1362, !dbg !16
  %1373 = bitcast float %1372 to i32, !dbg !16
  %1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1276, i32 %1373), !dbg !16
  %1375 = extractvalue [2 x <1 x float>] %.lcssa9, 1, !dbg !28
  %1376 = shufflevector <1 x float> %1375, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1377 = bitcast i32 %1371 to float, !dbg !16
  %1378 = bitcast i32 %1374 to float, !dbg !16
  %1379 = insertelement <4 x float> poison, float %1361, i64 0, !dbg !16
  %1380 = insertelement <4 x float> %1379, float %1362, i64 1, !dbg !16
  %1381 = insertelement <4 x float> %1380, float %1377, i64 2, !dbg !16
  %1382 = insertelement <4 x float> %1381, float %1378, i64 3, !dbg !16
  %1383 = insertelement <4 x float> poison, float %1377, i64 0, !dbg !16
  %1384 = insertelement <4 x float> %1383, float %1378, i64 1, !dbg !16
  %1385 = insertelement <4 x float> %1384, float %1367, i64 2, !dbg !16
  %1386 = insertelement <4 x float> %1385, float %1368, i64 3, !dbg !16
  %1387 = select <4 x i1> %50, <4 x float> %1382, <4 x float> %1386, !dbg !16
  %1388 = fdiv <4 x float> %1387, %1376, !dbg !28
  %1389 = fptrunc <4 x float> %1388 to <4 x half>, !dbg !29
  %1390 = extractelement <4 x half> %1389, i64 3, !dbg !29
  %1391 = extractelement <4 x half> %1389, i64 2, !dbg !29
  %1392 = extractelement <4 x half> %1389, i64 1, !dbg !29
  %1393 = extractelement <4 x half> %1389, i64 0, !dbg !29
  %1394 = bitcast i32 %1312 to float, !dbg !16
  %1395 = bitcast i32 %1315 to float, !dbg !16
  %1396 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !16
  %1397 = insertelement <4 x float> %1396, float %1303, i64 1, !dbg !16
  %1398 = insertelement <4 x float> %1397, float %1394, i64 2, !dbg !16
  %1399 = insertelement <4 x float> %1398, float %1395, i64 3, !dbg !16
  %1400 = insertelement <4 x float> poison, float %1394, i64 0, !dbg !16
  %1401 = insertelement <4 x float> %1400, float %1395, i64 1, !dbg !16
  %1402 = insertelement <4 x float> %1401, float %1308, i64 2, !dbg !16
  %1403 = insertelement <4 x float> %1402, float %1309, i64 3, !dbg !16
  %1404 = select <4 x i1> %50, <4 x float> %1399, <4 x float> %1403, !dbg !16
  %1405 = fdiv <4 x float> %1404, %1317, !dbg !28
  %1406 = fptrunc <4 x float> %1405 to <4 x half>, !dbg !29
  %1407 = extractelement <4 x half> %1406, i64 3, !dbg !29
  %1408 = extractelement <4 x half> %1406, i64 2, !dbg !29
  %1409 = extractelement <4 x half> %1406, i64 1, !dbg !29
  %1410 = extractelement <4 x half> %1406, i64 0, !dbg !29
  %address.thread648 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !30, !range !11, !frisk.register_lifetimes !5
  %address.local649 = zext nneg i32 %address.thread648 to i64, !dbg !10
  %address.local652 = lshr i32 %address.thread648, 2, !dbg !13
  %address.local653 = and i32 %address.local652, 48, !dbg !13
  %address.local654 = zext nneg i32 %address.local653 to i64, !dbg !13
  %address.local656 = and i32 %address.thread648, 240, !dbg !13
  %address.local657 = sub nsw i32 0, %address.local656, !dbg !13
  %address.local658 = sext i32 %address.local657 to i64, !dbg !13
  %address.local651 = add nsw i64 %16, %address.local649, !dbg !30
  %address.local655 = add nsw i64 %address.local651, %address.local658, !dbg !30
  %address.local659 = add nsw i64 %address.local655, %address.local654, !dbg !30
  %address.local662.idx = shl nuw nsw i64 %address.local659, 8, !dbg !30
  %address.local662 = getelementptr inbounds nuw i8, ptr addrspace(1) %36, i64 %address.local662.idx, !dbg !30
  %address.local664 = and i32 %address.local652, 12, !dbg !13
  %address.local665 = zext nneg i32 %address.local664 to i64, !dbg !13
  %address.local666 = or disjoint i64 %53, %address.local665, !dbg !30
  %address.local667 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local666, !dbg !30
  store half %1410, ptr addrspace(1) %address.local667, align 2, !dbg !30
  %address.local669 = or disjoint i64 %address.local666, 1, !dbg !30
  %address.local670 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local669, !dbg !30
  store half %1409, ptr addrspace(1) %address.local670, align 2, !dbg !30
  %address.local672 = or disjoint i64 %address.local666, 2, !dbg !30
  %address.local673 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local672, !dbg !30
  store half %1408, ptr addrspace(1) %address.local673, align 2, !dbg !30
  %address.local675 = or disjoint i64 %address.local666, 3, !dbg !30
  %address.local676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local675, !dbg !30
  store half %1407, ptr addrspace(1) %address.local676, align 2, !dbg !30
  %address.local680 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local662, i64 16384, !dbg !30
  %address.local681 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local666, !dbg !30
  store half %1393, ptr addrspace(1) %address.local681, align 2, !dbg !30
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local669, !dbg !30
  store half %1392, ptr addrspace(1) %address.local683, align 2, !dbg !30
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local672, !dbg !30
  store half %1391, ptr addrspace(1) %address.local685, align 2, !dbg !30
  %address.local687 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local675, !dbg !30
  store half %1390, ptr addrspace(1) %address.local687, align 2, !dbg !30
  %1411 = fdiv <4 x float> %1351, %1376, !dbg !28
  %1412 = fptrunc <4 x float> %1411 to <4 x half>, !dbg !29
  %1413 = extractelement <4 x half> %1412, i64 3, !dbg !29
  %1414 = extractelement <4 x half> %1412, i64 2, !dbg !29
  %1415 = extractelement <4 x half> %1412, i64 1, !dbg !29
  %1416 = extractelement <4 x half> %1412, i64 0, !dbg !29
  %1417 = fdiv <4 x float> %1292, %1317, !dbg !28
  %1418 = fptrunc <4 x float> %1417 to <4 x half>, !dbg !29
  %1419 = extractelement <4 x half> %1418, i64 3, !dbg !29
  %1420 = extractelement <4 x half> %1418, i64 2, !dbg !29
  %1421 = extractelement <4 x half> %1418, i64 1, !dbg !29
  %1422 = extractelement <4 x half> %1418, i64 0, !dbg !29
  %address.local707 = or disjoint i64 %address.local666, 16, !dbg !30
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local707, !dbg !30
  store half %1422, ptr addrspace(1) %address.local708, align 2, !dbg !30
  %address.local710 = or disjoint i64 %address.local666, 17, !dbg !30
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local710, !dbg !30
  store half %1421, ptr addrspace(1) %address.local711, align 2, !dbg !30
  %address.local713 = or disjoint i64 %address.local666, 18, !dbg !30
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local713, !dbg !30
  store half %1420, ptr addrspace(1) %address.local714, align 2, !dbg !30
  %address.local716 = or disjoint i64 %address.local666, 19, !dbg !30
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local662, i64 %address.local716, !dbg !30
  store half %1419, ptr addrspace(1) %address.local717, align 2, !dbg !30
  %address.local722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local707, !dbg !30
  store half %1416, ptr addrspace(1) %address.local722, align 2, !dbg !30
  %address.local724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local710, !dbg !30
  store half %1415, ptr addrspace(1) %address.local724, align 2, !dbg !30
  %address.local726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local713, !dbg !30
  store half %1414, ptr addrspace(1) %address.local726, align 2, !dbg !30
  %address.local728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local680, i64 %address.local716, !dbg !30
  store half %1413, ptr addrspace(1) %address.local728, align 2, !dbg !30
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1423 = add nuw nsw i64 %52, 1, !dbg !15
  %1424 = icmp samesign ult i64 %52, 3, !dbg !15
  br i1 %1424, label %51, label %1425, !dbg !15

1425:                                             ; preds = %._crit_edge
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #6

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #7

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #7

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind memory(none) }
attributes #10 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input_sd16.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 46, column: 17, scope: !3)
!9 = !DILocation(line: 66, column: 22, scope: !3)
!10 = !DILocation(line: 68, column: 24, scope: !3)
!11 = !{i32 0, i32 256}
!12 = !DILocation(line: 59, column: 17, scope: !3)
!13 = !DILocation(line: 73, column: 19, scope: !3)
!14 = !DILocation(line: 75, column: 24, scope: !3)
!15 = !DILocation(line: 51, column: 7, scope: !3)
!16 = !DILocation(line: 56, column: 24, scope: !3)
!17 = !DILocation(line: 67, column: 22, scope: !3)
!18 = !DILocation(line: 69, column: 26, scope: !3)
!19 = !DILocation(line: 70, column: 23, scope: !3)
!20 = !DILocation(line: 92, column: 20, scope: !3)
!21 = !DILocation(line: 76, column: 23, scope: !3)
!22 = !DILocation(line: 77, column: 26, scope: !3)
!23 = !DILocation(line: 85, column: 22, scope: !3)
!24 = !DILocation(line: 87, column: 16, scope: !3)
!25 = !DILocation(line: 86, column: 20, scope: !3)
!26 = !DILocation(line: 93, column: 23, scope: !3)
!27 = !DILocation(line: 94, column: 20, scope: !3)
!28 = !DILocation(line: 97, column: 23, scope: !3)
!29 = !DILocation(line: 98, column: 19, scope: !3)
!30 = !DILocation(line: 99, column: 9, scope: !3)
