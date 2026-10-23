; ModuleID = 'generated_ir/sd16_extract_fix/test_input_spidD.off.ll'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@async_lds_staging_0 = private addrspace(3) global [4096 x half] undef, align 16
@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

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
  %.not314 = icmp eq i32 %20, 0, !dbg !12
  %21 = add nsw i64 %9, %18, !dbg !13
  %22 = lshr i32 %17, 2, !dbg !13
  %23 = and i32 %22, 48, !dbg !13
  %index.compact13 = zext nneg i32 %23 to i64, !dbg !13
  %24 = and i32 %17, 240, !dbg !13
  %25 = sub nsw i32 0, %24, !dbg !13
  %index.compact12 = sext i32 %25 to i64, !dbg !13
  %26 = add nsw i64 %21, %index.compact12, !dbg !13
  %27 = add nsw i64 %26, %index.compact13, !dbg !13
  %28 = and i32 %22, 12, !dbg !13
  %29 = add nsw i64 %27, 1, !dbg !14
  %30 = add nsw i64 %27, 65, !dbg !14
  %.not3157 = icmp slt i32 %5, 0
  %31 = shl nsw i64 %14, 20
  %.idx380 = and i64 %31, 32505856
  %32 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx380
  %.idx381 = shl nsw i64 %16, 8
  %33 = getelementptr i8, ptr addrspace(1) %32, i64 %.idx381
  %34 = lshr i64 %29, 32
  %compare.high = trunc nuw i64 %34 to i32
  %compare.low.base = trunc i64 %29 to i32
  %35 = lshr i64 %30, 32
  %compare.high1074 = trunc nuw i64 %35 to i32
  %compare.low.base1078 = trunc i64 %30 to i32
  %36 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i64 %.idx380
  %.mask = shl nuw nsw i32 %17, 1, !dbg !15
  %37 = and i32 %.mask, 6, !dbg !15
  %38 = zext nneg i32 %37 to i64, !dbg !15
  %39 = lshr i64 %18, 2, !dbg !15
  %.lobit36 = and i64 %39, 1, !dbg !15
  %40 = and i64 %18, 120, !dbg !15
  %41 = or disjoint i64 %40, %38
  %invariant.gep = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %.lobit36
  %.idx385 = shl nuw nsw i64 %41, 6
  %invariant.gep58 = getelementptr inbounds nuw i8, ptr addrspace(3) %invariant.gep, i64 %.idx385
  %42 = insertelement <4 x i1> poison, i1 %.not314, i64 0, !dbg !16
  %43 = shufflevector <4 x i1> %42, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !16
  br label %44, !dbg !15

44:                                               ; preds = %4, %._crit_edge
  %45 = phi i64 [ 0, %4 ], [ %1424, %._crit_edge ]
  %46 = shl nuw nsw i64 %45, 5, !dbg !15
  br i1 %.not3157, label %.._crit_edge_crit_edge, label %.lr.ph, !dbg !16

.._crit_edge_crit_edge:                           ; preds = %44
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %.pre26 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !16
  %.pre28 = add nuw nsw i32 %.pre26, 64, !dbg !16
  %.pre30 = xor i32 %.pre26, 16, !dbg !16
  %.pre32 = and i32 %.pre28, 192, !dbg !16
  %.pre34 = xor i32 %.pre26, 32, !dbg !16
  br label %._crit_edge, !dbg !16

.lr.ph:                                           ; preds = %44, %302
  %47 = phi [2 x <1 x float>] [ %1176, %302 ], [ zeroinitializer, %44 ]
  %48 = phi [2 x <8 x float>] [ %1254, %302 ], [ zeroinitializer, %44 ]
  %49 = phi i64 [ %1255, %302 ], [ 0, %44 ]
  %50 = lshr i64 %49, 5, !dbg !17
  %51 = shl i64 %50, 20
  %52 = shl i64 %49, 15
  %.idx384 = sub i64 %52, %51
  br label %53, !dbg !12

53:                                               ; preds = %.lr.ph, %171
  %54 = phi [2 x <32 x float>] [ zeroinitializer, %.lr.ph ], [ %299, %171 ]
  %55 = phi i64 [ 0, %.lr.ph ], [ %300, %171 ]
  %56 = shl nuw nsw i64 %55, 5, !dbg !12
  %57 = getelementptr [2 x i8], ptr addrspace(1) %33, i64 %56, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %58 = shl nuw nsw i32 %address.thread, 1, !dbg !9
  %59 = and i32 %58, 384, !dbg !9
  %address.local418 = zext nneg i32 %59 to i64, !dbg !9
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local418, !dbg !9
  %address.local423 = and i32 %58, 2, !dbg !9
  %address.local424 = shl nuw nsw i32 %address.thread, 6, !dbg !9
  %address.local425 = and i32 %address.local424, 16256, !dbg !9
  %address.local426 = or disjoint i32 %address.local423, %address.local425, !dbg !9
  %address.local427 = zext nneg i32 %address.local426 to i64, !dbg !9
  %address.local428 = getelementptr [2 x i8], ptr addrspace(1) %57, i64 %address.local427, !dbg !9
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local428 to i64, !dbg !9
  %dma.source.opaque = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #9, !dbg !9
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !9
  %address.local431 = or disjoint i64 %address.local418, 512, !dbg !9
  %address.local432 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local431, !dbg !9
  %address.local442 = getelementptr i8, ptr addrspace(1) %address.local428, i64 8, !dbg !9
  %dma.source.bits1206 = ptrtoint ptr addrspace(1) %address.local442 to i64, !dbg !9
  %dma.source.opaque1207 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1206) #9, !dbg !9
  %dma.source.ptr1208 = inttoptr i64 %dma.source.opaque1207 to ptr addrspace(1), !dbg !9
  %address.local445 = or disjoint i64 %address.local418, 1024, !dbg !9
  %address.local446 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local445, !dbg !9
  %address.local456 = getelementptr i8, ptr addrspace(1) %address.local428, i64 16, !dbg !9
  %dma.source.bits1209 = ptrtoint ptr addrspace(1) %address.local456 to i64, !dbg !9
  %dma.source.opaque1210 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1209) #9, !dbg !9
  %dma.source.ptr1211 = inttoptr i64 %dma.source.opaque1210 to ptr addrspace(1), !dbg !9
  %address.local459 = or disjoint i64 %address.local418, 1536, !dbg !9
  %address.local460 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local459, !dbg !9
  %address.local470 = getelementptr i8, ptr addrspace(1) %address.local428, i64 24, !dbg !9
  %dma.source.bits1212 = ptrtoint ptr addrspace(1) %address.local470 to i64, !dbg !9
  %dma.source.opaque1213 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1212) #9, !dbg !9
  %dma.source.ptr1214 = inttoptr i64 %dma.source.opaque1213 to ptr addrspace(1), !dbg !9
  %address.local473 = or disjoint i64 %address.local418, 2048, !dbg !9
  %address.local474 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local473, !dbg !9
  %address.local484 = getelementptr i8, ptr addrspace(1) %address.local428, i64 32, !dbg !9
  %dma.source.bits1215 = ptrtoint ptr addrspace(1) %address.local484 to i64, !dbg !9
  %dma.source.opaque1216 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1215) #9, !dbg !9
  %dma.source.ptr1217 = inttoptr i64 %dma.source.opaque1216 to ptr addrspace(1), !dbg !9
  %address.local487 = or disjoint i64 %address.local418, 2560, !dbg !9
  %address.local488 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local487, !dbg !9
  %address.local498 = getelementptr i8, ptr addrspace(1) %address.local428, i64 40, !dbg !9
  %dma.source.bits1218 = ptrtoint ptr addrspace(1) %address.local498 to i64, !dbg !9
  %dma.source.opaque1219 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1218) #9, !dbg !9
  %dma.source.ptr1220 = inttoptr i64 %dma.source.opaque1219 to ptr addrspace(1), !dbg !9
  %address.local501 = or disjoint i64 %address.local418, 3072, !dbg !9
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !9
  %address.local512 = getelementptr i8, ptr addrspace(1) %address.local428, i64 48, !dbg !9
  %dma.source.bits1221 = ptrtoint ptr addrspace(1) %address.local512 to i64, !dbg !9
  %dma.source.opaque1222 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1221) #9, !dbg !9
  %dma.source.ptr1223 = inttoptr i64 %dma.source.opaque1222 to ptr addrspace(1), !dbg !9
  %address.local515 = or disjoint i64 %address.local418, 3584, !dbg !9
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local515, !dbg !9
  %address.local526 = getelementptr i8, ptr addrspace(1) %address.local428, i64 56, !dbg !9
  %dma.source.bits1224 = ptrtoint ptr addrspace(1) %address.local526 to i64, !dbg !9
  %dma.source.opaque1225 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1224) #9, !dbg !9
  %dma.source.ptr1226 = inttoptr i64 %dma.source.opaque1225 to ptr addrspace(1), !dbg !9
  %60 = shl nuw nsw i64 %55, 10, !dbg !17
  %61 = add nuw nsw i64 %60, %49, !dbg !17
  %62 = lshr i64 %61, 12, !dbg !17
  %63 = add nuw nsw i64 %62, %8, !dbg !17
  %64 = add nuw nsw i64 %56, %50, !dbg !17
  %65 = and i64 %64, 127, !dbg !17
  %66 = shl i64 %63, 19, !dbg !17
  %67 = and i64 %66, 16252928, !dbg !17
  %68 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %67, !dbg !17
  %69 = getelementptr [2 x i8], ptr addrspace(1) %68, i64 %65, !dbg !17
  %70 = getelementptr i8, ptr addrspace(1) %69, i64 %.idx384, !dbg !17
  %address.local527 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local418, !dbg !17
  %address.local531 = and i32 %58, 30, !dbg !17
  %address.local532 = shl nuw nsw i32 %address.thread, 3, !dbg !17
  %address.local533 = and i32 %address.local532, 1920, !dbg !17
  %address.local534 = or disjoint i32 %address.local531, %address.local533, !dbg !17
  %address.local535 = zext nneg i32 %address.local534 to i64, !dbg !17
  %address.local536 = getelementptr [2 x i8], ptr addrspace(1) %70, i64 %address.local535, !dbg !17
  %dma.source.bits1227 = ptrtoint ptr addrspace(1) %address.local536 to i64, !dbg !17
  %dma.source.opaque1228 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1227) #9, !dbg !17
  %dma.source.ptr1229 = inttoptr i64 %dma.source.opaque1228 to ptr addrspace(1), !dbg !17
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local431, !dbg !17
  %address.local547 = getelementptr i8, ptr addrspace(1) %address.local536, i64 4096, !dbg !17
  %dma.source.bits1230 = ptrtoint ptr addrspace(1) %address.local547 to i64, !dbg !17
  %dma.source.opaque1231 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1230) #9, !dbg !17
  %dma.source.ptr1232 = inttoptr i64 %dma.source.opaque1231 to ptr addrspace(1), !dbg !17
  %address.local548 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local445, !dbg !17
  %address.local558 = getelementptr i8, ptr addrspace(1) %address.local536, i64 8192, !dbg !17
  %dma.source.bits1233 = ptrtoint ptr addrspace(1) %address.local558 to i64, !dbg !17
  %dma.source.opaque1234 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1233) #9, !dbg !17
  %dma.source.ptr1235 = inttoptr i64 %dma.source.opaque1234 to ptr addrspace(1), !dbg !17
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local459, !dbg !17
  %address.local569 = getelementptr i8, ptr addrspace(1) %address.local536, i64 12288, !dbg !17
  %dma.source.bits1236 = ptrtoint ptr addrspace(1) %address.local569 to i64, !dbg !17
  %dma.source.opaque1237 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1236) #9, !dbg !17
  %dma.source.ptr1238 = inttoptr i64 %dma.source.opaque1237 to ptr addrspace(1), !dbg !17
  %address.local570 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local473, !dbg !17
  %address.local580 = getelementptr i8, ptr addrspace(1) %address.local536, i64 16384, !dbg !17
  %dma.source.bits1239 = ptrtoint ptr addrspace(1) %address.local580 to i64, !dbg !17
  %dma.source.opaque1240 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1239) #9, !dbg !17
  %dma.source.ptr1241 = inttoptr i64 %dma.source.opaque1240 to ptr addrspace(1), !dbg !17
  %address.local581 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local487, !dbg !17
  %address.local591 = getelementptr i8, ptr addrspace(1) %address.local536, i64 20480, !dbg !17
  %dma.source.bits1242 = ptrtoint ptr addrspace(1) %address.local591 to i64, !dbg !17
  %dma.source.opaque1243 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1242) #9, !dbg !17
  %dma.source.ptr1244 = inttoptr i64 %dma.source.opaque1243 to ptr addrspace(1), !dbg !17
  %address.local592 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local501, !dbg !17
  %address.local602 = getelementptr i8, ptr addrspace(1) %address.local536, i64 24576, !dbg !17
  %dma.source.bits1245 = ptrtoint ptr addrspace(1) %address.local602 to i64, !dbg !17
  %dma.source.opaque1246 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1245) #9, !dbg !17
  %dma.source.ptr1247 = inttoptr i64 %dma.source.opaque1246 to ptr addrspace(1), !dbg !17
  %address.local603 = getelementptr [2 x i8], ptr addrspace(3) @async_lds_staging_0, i64 %address.local515, !dbg !17
  %71 = insertelement <16 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local419, i64 0, !dbg !9
  %72 = insertelement <16 x ptr addrspace(3)> %71, ptr addrspace(3) %address.local432, i64 1, !dbg !9
  %73 = insertelement <16 x ptr addrspace(3)> %72, ptr addrspace(3) %address.local446, i64 2, !dbg !9
  %74 = insertelement <16 x ptr addrspace(3)> %73, ptr addrspace(3) %address.local460, i64 3, !dbg !9
  %75 = insertelement <16 x ptr addrspace(3)> %74, ptr addrspace(3) %address.local474, i64 4, !dbg !9
  %76 = insertelement <16 x ptr addrspace(3)> %75, ptr addrspace(3) %address.local488, i64 5, !dbg !9
  %77 = insertelement <16 x ptr addrspace(3)> %76, ptr addrspace(3) %address.local502, i64 6, !dbg !9
  %78 = insertelement <16 x ptr addrspace(3)> %77, ptr addrspace(3) %address.local516, i64 7, !dbg !9
  %79 = insertelement <16 x ptr addrspace(3)> %78, ptr addrspace(3) %address.local527, i64 8, !dbg !9
  %80 = insertelement <16 x ptr addrspace(3)> %79, ptr addrspace(3) %address.local537, i64 9, !dbg !9
  %81 = insertelement <16 x ptr addrspace(3)> %80, ptr addrspace(3) %address.local548, i64 10, !dbg !9
  %82 = insertelement <16 x ptr addrspace(3)> %81, ptr addrspace(3) %address.local559, i64 11, !dbg !9
  %83 = insertelement <16 x ptr addrspace(3)> %82, ptr addrspace(3) %address.local570, i64 12, !dbg !9
  %84 = insertelement <16 x ptr addrspace(3)> %83, ptr addrspace(3) %address.local581, i64 13, !dbg !9
  %85 = insertelement <16 x ptr addrspace(3)> %84, ptr addrspace(3) %address.local592, i64 14, !dbg !9
  %86 = insertelement <16 x ptr addrspace(3)> %85, ptr addrspace(3) %address.local603, i64 15, !dbg !9
  %87 = ptrtoint <16 x ptr addrspace(3)> %86 to <16 x i64>, !dbg !9
  %88 = trunc <16 x i64> %87 to <16 x i32>, !dbg !9
  %89 = extractelement <16 x i32> %88, i64 0, !dbg !9
  %90 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %89), !dbg !9
  %91 = zext i32 %90 to i64, !dbg !9
  %92 = inttoptr i64 %91 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %92, i32 4, i32 0, i32 0), !dbg !9
  %93 = extractelement <16 x i32> %88, i64 1, !dbg !9
  %94 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %93), !dbg !9
  %95 = zext i32 %94 to i64, !dbg !9
  %96 = inttoptr i64 %95 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1208, ptr addrspace(3) %96, i32 4, i32 0, i32 0), !dbg !9
  %97 = extractelement <16 x i32> %88, i64 2, !dbg !9
  %98 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %97), !dbg !9
  %99 = zext i32 %98 to i64, !dbg !9
  %100 = inttoptr i64 %99 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1211, ptr addrspace(3) %100, i32 4, i32 0, i32 0), !dbg !9
  %101 = extractelement <16 x i32> %88, i64 3, !dbg !9
  %102 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %101), !dbg !9
  %103 = zext i32 %102 to i64, !dbg !9
  %104 = inttoptr i64 %103 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1214, ptr addrspace(3) %104, i32 4, i32 0, i32 0), !dbg !9
  %105 = extractelement <16 x i32> %88, i64 4, !dbg !9
  %106 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !9
  %107 = zext i32 %106 to i64, !dbg !9
  %108 = inttoptr i64 %107 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1217, ptr addrspace(3) %108, i32 4, i32 0, i32 0), !dbg !9
  %109 = extractelement <16 x i32> %88, i64 5, !dbg !9
  %110 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !9
  %111 = zext i32 %110 to i64, !dbg !9
  %112 = inttoptr i64 %111 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1220, ptr addrspace(3) %112, i32 4, i32 0, i32 0), !dbg !9
  %113 = extractelement <16 x i32> %88, i64 6, !dbg !9
  %114 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %113), !dbg !9
  %115 = zext i32 %114 to i64, !dbg !9
  %116 = inttoptr i64 %115 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1223, ptr addrspace(3) %116, i32 4, i32 0, i32 0), !dbg !9
  %117 = extractelement <16 x i32> %88, i64 7, !dbg !9
  %118 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %117), !dbg !9
  %119 = zext i32 %118 to i64, !dbg !9
  %120 = inttoptr i64 %119 to ptr addrspace(3), !dbg !9
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1226, ptr addrspace(3) %120, i32 4, i32 0, i32 0), !dbg !9
  tail call void @llvm.amdgcn.asyncmark(), !dbg !9
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %121 = extractelement <16 x i32> %88, i64 8, !dbg !17
  %122 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %121), !dbg !17
  %123 = zext i32 %122 to i64, !dbg !17
  %124 = inttoptr i64 %123 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1229, ptr addrspace(3) %124, i32 4, i32 0, i32 0), !dbg !17
  %125 = extractelement <16 x i32> %88, i64 9, !dbg !17
  %126 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %125), !dbg !17
  %127 = zext i32 %126 to i64, !dbg !17
  %128 = inttoptr i64 %127 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1232, ptr addrspace(3) %128, i32 4, i32 0, i32 0), !dbg !17
  %129 = extractelement <16 x i32> %88, i64 10, !dbg !17
  %130 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !17
  %131 = zext i32 %130 to i64, !dbg !17
  %132 = inttoptr i64 %131 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1235, ptr addrspace(3) %132, i32 4, i32 0, i32 0), !dbg !17
  %133 = extractelement <16 x i32> %88, i64 11, !dbg !17
  %134 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !17
  %135 = zext i32 %134 to i64, !dbg !17
  %136 = inttoptr i64 %135 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1238, ptr addrspace(3) %136, i32 4, i32 0, i32 0), !dbg !17
  %137 = extractelement <16 x i32> %88, i64 12, !dbg !17
  %138 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %137), !dbg !17
  %139 = zext i32 %138 to i64, !dbg !17
  %140 = inttoptr i64 %139 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1241, ptr addrspace(3) %140, i32 4, i32 0, i32 0), !dbg !17
  %141 = extractelement <16 x i32> %88, i64 13, !dbg !17
  %142 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %141), !dbg !17
  %143 = zext i32 %142 to i64, !dbg !17
  %144 = inttoptr i64 %143 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1244, ptr addrspace(3) %144, i32 4, i32 0, i32 0), !dbg !17
  %145 = extractelement <16 x i32> %88, i64 14, !dbg !17
  %146 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %145), !dbg !17
  %147 = zext i32 %146 to i64, !dbg !17
  %148 = inttoptr i64 %147 to ptr addrspace(3), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1247, ptr addrspace(3) %148, i32 4, i32 0, i32 0), !dbg !17
  %149 = extractelement <16 x i32> %88, i64 15, !dbg !17
  %150 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %149), !dbg !17
  %151 = zext i32 %150 to i64, !dbg !17
  %152 = inttoptr i64 %151 to ptr addrspace(3), !dbg !17
  %address.local613 = getelementptr i8, ptr addrspace(1) %address.local536, i64 28672, !dbg !17
  %dma.source.bits1248 = ptrtoint ptr addrspace(1) %address.local613 to i64, !dbg !17
  %dma.source.opaque1249 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1248) #9, !dbg !17
  %dma.source.ptr1250 = inttoptr i64 %dma.source.opaque1249 to ptr addrspace(1), !dbg !17
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1250, ptr addrspace(3) %152, i32 4, i32 0, i32 0), !dbg !17
  tail call void @llvm.amdgcn.asyncmark(), !dbg !17
  %address.local614 = and i32 %address.thread, 192, !dbg !10
  %address.local615 = shl nuw nsw i32 %address.thread, 2, !dbg !10
  %address.local616 = and i32 %address.local615, 60, !dbg !10
  %address.local617 = or disjoint i32 %address.local616, %address.local614, !dbg !10
  %address.local618 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local619 = and i32 %address.local618, 1536, !dbg !10
  %address.local620 = or disjoint i32 %address.local617, %address.local619, !dbg !10
  %address.local621 = zext nneg i32 %address.local620 to i64, !dbg !10
  %address.local622 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local621, !dbg !10
  %153 = load <4 x half>, ptr addrspace(3) %address.local622, align 8, !dbg !10
  %address.local632 = getelementptr i8, ptr addrspace(3) %address.local622, i64 4096, !dbg !10
  %154 = load <4 x half>, ptr addrspace(3) %address.local632, align 8, !dbg !10
  %address.local642 = getelementptr i8, ptr addrspace(3) %address.local622, i64 512, !dbg !10
  %155 = load <4 x half>, ptr addrspace(3) %address.local642, align 8, !dbg !10
  %address.local652 = getelementptr i8, ptr addrspace(3) %address.local622, i64 4608, !dbg !10
  %156 = load <4 x half>, ptr addrspace(3) %address.local652, align 8, !dbg !10
  %157 = fmul <4 x half> %153, splat (half 1.275630e-01), !dbg !10
  %158 = fmul <4 x half> %155, splat (half 1.275630e-01), !dbg !10
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !17
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  br label %159, !dbg !17

159:                                              ; preds = %53, %159
  %160 = phi i64 [ %18, %53 ], [ %169, %159 ]
  %161 = lshr i64 %160, 6, !dbg !17
  %162 = and i64 %161, 30, !dbg !17
  %gep59 = getelementptr inbounds nuw [2 x i8], ptr addrspace(3) %invariant.gep58, i64 %162, !dbg !17
  %163 = load half, ptr addrspace(3) %gep59, align 2, !dbg !17
  %164 = getelementptr inbounds nuw i8, ptr addrspace(3) %gep59, i64 64, !dbg !17
  %165 = load half, ptr addrspace(3) %164, align 2, !dbg !17
  %166 = insertelement <2 x half> poison, half %163, i64 0, !dbg !17
  %167 = insertelement <2 x half> %166, half %165, i64 1, !dbg !17
  %.idx = shl i64 %160, 2, !dbg !17
  %168 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %.idx, !dbg !17
  store <2 x half> %167, ptr addrspace(3) %168, align 4, !dbg !17
  %169 = add nuw nsw i64 %160, 256, !dbg !17
  %170 = icmp samesign ult i64 %160, 1792, !dbg !17
  br i1 %170, label %159, label %171, !dbg !17

171:                                              ; preds = %159
  %172 = fmul <4 x half> %154, splat (half 1.275630e-01), !dbg !10
  %173 = fmul <4 x half> %156, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !17
  tail call void @llvm.amdgcn.s.barrier(), !dbg !17
  fence syncscope("workgroup") acquire, !dbg !17
  %address.thread653 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !18, !range !11, !frisk.register_lifetimes !5
  %address.local654 = shl nuw nsw i32 %address.thread653, 1, !dbg !18
  %address.local655 = and i32 %address.local654, 8, !dbg !18
  %address.local656 = shl nuw nsw i32 %address.thread653, 4, !dbg !18
  %address.local657 = and i32 %address.local656, 48, !dbg !18
  %address.local658 = or disjoint i32 %address.local655, %address.local657, !dbg !18
  %address.local659 = shl nuw nsw i32 %address.thread653, 5, !dbg !18
  %address.local660 = and i32 %address.local659, 1792, !dbg !18
  %address.local661 = or disjoint i32 %address.local658, %address.local660, !dbg !18
  %address.local662 = zext nneg i32 %address.local661 to i64, !dbg !18
  %address.local663 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local662, !dbg !18
  %174 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local663, i32 0) #10, !dbg !18
  %175 = bitcast <4 x i32> %174 to <8 x half>, !dbg !18
  %address.local674 = getelementptr i8, ptr addrspace(3) %address.local663, i64 128, !dbg !18
  %176 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local674, i32 0) #10, !dbg !18
  %177 = bitcast <4 x i32> %176 to <8 x half>, !dbg !18
  %address.local685 = getelementptr i8, ptr addrspace(3) %address.local663, i64 256, !dbg !18
  %178 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local685, i32 0) #10, !dbg !18
  %179 = bitcast <4 x i32> %178 to <8 x half>, !dbg !18
  %address.local696 = getelementptr i8, ptr addrspace(3) %address.local663, i64 384, !dbg !18
  %180 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local696, i32 0) #10, !dbg !18
  %181 = bitcast <4 x i32> %180 to <8 x half>, !dbg !18
  %address.local707 = getelementptr i8, ptr addrspace(3) %address.local663, i64 4096, !dbg !18
  %182 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local707, i32 0) #10, !dbg !18
  %183 = bitcast <4 x i32> %182 to <8 x half>, !dbg !18
  %address.local718 = getelementptr i8, ptr addrspace(3) %address.local663, i64 4224, !dbg !18
  %184 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local718, i32 0) #10, !dbg !18
  %185 = bitcast <4 x i32> %184 to <8 x half>, !dbg !18
  %address.local729 = getelementptr i8, ptr addrspace(3) %address.local663, i64 4352, !dbg !18
  %186 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local729, i32 0) #10, !dbg !18
  %187 = bitcast <4 x i32> %186 to <8 x half>, !dbg !18
  %address.local740 = getelementptr i8, ptr addrspace(3) %address.local663, i64 4480, !dbg !18
  %188 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local740, i32 0) #10, !dbg !18
  %189 = bitcast <4 x i32> %188 to <8 x half>, !dbg !18
  %190 = shufflevector <8 x half> %175, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %190, <4 x float> zeroinitializer) #8, !dbg !18
  %192 = shufflevector <8 x half> %183, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %192, <4 x float> %191) #8, !dbg !18
  %194 = extractvalue [2 x <32 x float>] %54, 0, !dbg !19
  %195 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %196 = fadd <4 x float> %195, %193, !dbg !19
  %197 = shufflevector <8 x half> %175, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %197, <4 x float> zeroinitializer) #8, !dbg !18
  %199 = shufflevector <8 x half> %183, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %199, <4 x float> %198) #8, !dbg !18
  %201 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %202 = fadd <4 x float> %201, %200, !dbg !19
  %203 = shufflevector <8 x half> %177, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %203, <4 x float> zeroinitializer) #8, !dbg !18
  %205 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %205, <4 x float> %204) #8, !dbg !18
  %207 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %208 = fadd <4 x float> %207, %206, !dbg !19
  %209 = shufflevector <4 x float> %208, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %210 = shufflevector <8 x half> %177, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !18
  %212 = shufflevector <8 x half> %185, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %212, <4 x float> %211) #8, !dbg !18
  %214 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %215 = fadd <4 x float> %214, %213, !dbg !19
  %216 = shufflevector <4 x float> %215, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %217 = shufflevector <8 x half> %179, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %217, <4 x float> zeroinitializer) #8, !dbg !18
  %219 = shufflevector <8 x half> %187, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %219, <4 x float> %218) #8, !dbg !18
  %221 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %222 = fadd <4 x float> %221, %220, !dbg !19
  %223 = shufflevector <4 x float> %222, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %224 = shufflevector <8 x half> %179, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !18
  %226 = shufflevector <8 x half> %187, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %226, <4 x float> %225) #8, !dbg !18
  %228 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %229 = fadd <4 x float> %228, %227, !dbg !19
  %230 = shufflevector <4 x float> %229, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %231 = shufflevector <8 x half> %181, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %231, <4 x float> zeroinitializer) #8, !dbg !18
  %233 = shufflevector <8 x half> %189, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !18
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %233, <4 x float> %232) #8, !dbg !18
  %235 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %236 = fadd <4 x float> %235, %234, !dbg !19
  %237 = shufflevector <4 x float> %236, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %238 = shufflevector <8 x half> %181, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %157, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !18
  %240 = shufflevector <8 x half> %189, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !18
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %172, <4 x half> %240, <4 x float> %239) #8, !dbg !18
  %242 = shufflevector <32 x float> %194, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %243 = fadd <4 x float> %242, %241, !dbg !19
  %244 = shufflevector <4 x float> %243, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %190, <4 x float> zeroinitializer) #8, !dbg !18
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %192, <4 x float> %245) #8, !dbg !18
  %247 = extractvalue [2 x <32 x float>] %54, 1, !dbg !19
  %248 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !19
  %249 = fadd <4 x float> %248, %246, !dbg !19
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %197, <4 x float> zeroinitializer) #8, !dbg !18
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %199, <4 x float> %250) #8, !dbg !18
  %252 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !19
  %253 = fadd <4 x float> %252, %251, !dbg !19
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %203, <4 x float> zeroinitializer) #8, !dbg !18
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %205, <4 x float> %254) #8, !dbg !18
  %256 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !19
  %257 = fadd <4 x float> %256, %255, !dbg !19
  %258 = shufflevector <4 x float> %257, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %210, <4 x float> zeroinitializer) #8, !dbg !18
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %212, <4 x float> %259) #8, !dbg !18
  %261 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !19
  %262 = fadd <4 x float> %261, %260, !dbg !19
  %263 = shufflevector <4 x float> %262, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %217, <4 x float> zeroinitializer) #8, !dbg !18
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %219, <4 x float> %264) #8, !dbg !18
  %266 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !19
  %267 = fadd <4 x float> %266, %265, !dbg !19
  %268 = shufflevector <4 x float> %267, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %224, <4 x float> zeroinitializer) #8, !dbg !18
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %226, <4 x float> %269) #8, !dbg !18
  %271 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !19
  %272 = fadd <4 x float> %271, %270, !dbg !19
  %273 = shufflevector <4 x float> %272, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %231, <4 x float> zeroinitializer) #8, !dbg !18
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %233, <4 x float> %274) #8, !dbg !18
  %276 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !19
  %277 = fadd <4 x float> %276, %275, !dbg !19
  %278 = shufflevector <4 x float> %277, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %158, <4 x half> %238, <4 x float> zeroinitializer) #8, !dbg !18
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %173, <4 x half> %240, <4 x float> %279) #8, !dbg !18
  %281 = shufflevector <32 x float> %247, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !19
  %282 = fadd <4 x float> %281, %280, !dbg !19
  %283 = shufflevector <4 x float> %282, <4 x float> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !19
  %284 = shufflevector <4 x float> %196, <4 x float> %202, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %285 = shufflevector <32 x float> %284, <32 x float> %209, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %286 = shufflevector <32 x float> %285, <32 x float> %216, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %287 = shufflevector <32 x float> %286, <32 x float> %223, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %288 = shufflevector <32 x float> %287, <32 x float> %230, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %289 = shufflevector <32 x float> %288, <32 x float> %237, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %290 = shufflevector <32 x float> %289, <32 x float> %244, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !12
  %291 = insertvalue [2 x <32 x float>] poison, <32 x float> %290, 0, !dbg !12
  %292 = shufflevector <4 x float> %249, <4 x float> %253, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %293 = shufflevector <32 x float> %292, <32 x float> %258, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %294 = shufflevector <32 x float> %293, <32 x float> %263, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %295 = shufflevector <32 x float> %294, <32 x float> %268, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %296 = shufflevector <32 x float> %295, <32 x float> %273, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %297 = shufflevector <32 x float> %296, <32 x float> %278, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 32, i32 33, i32 34, i32 35, i32 poison, i32 poison, i32 poison, i32 poison>, !dbg !12
  %298 = shufflevector <32 x float> %297, <32 x float> %283, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 32, i32 33, i32 34, i32 35>, !dbg !12
  %299 = insertvalue [2 x <32 x float>] %291, <32 x float> %298, 1, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %300 = add nuw nsw i64 %55, 1, !dbg !12
  %301 = icmp samesign ult i64 %55, 3, !dbg !12
  br i1 %301, label %53, label %302, !dbg !12

302:                                              ; preds = %171
  %303 = add nuw nsw i64 %50, %8, !dbg !20
  %304 = shl i64 %303, 19, !dbg !20
  %305 = and i64 %304, 16252928, !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %306 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %305, !dbg !20
  %307 = getelementptr i8, ptr addrspace(1) %306, i64 %.idx384, !dbg !20
  %308 = getelementptr [2 x i8], ptr addrspace(1) %307, i64 %46, !dbg !20
  %address.thread741 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !20, !range !11, !frisk.register_lifetimes !5
  %309 = shl nuw nsw i32 %address.thread741, 1, !dbg !9
  %310 = and i32 %309, 384, !dbg !9
  %address.local744 = zext nneg i32 %310 to i64, !dbg !9
  %address.local745 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local744, !dbg !9
  %address.local749 = and i32 %309, 30, !dbg !20
  %address.local750 = shl nuw nsw i32 %address.thread741, 3, !dbg !20
  %address.local751 = and i32 %address.local750, 1920, !dbg !20
  %address.local752 = or disjoint i32 %address.local749, %address.local751, !dbg !20
  %address.local753 = zext nneg i32 %address.local752 to i64, !dbg !20
  %address.local754 = getelementptr [2 x i8], ptr addrspace(1) %308, i64 %address.local753, !dbg !20
  %dma.source.bits1251 = ptrtoint ptr addrspace(1) %address.local754 to i64, !dbg !20
  %dma.source.opaque1252 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1251) #9, !dbg !20
  %dma.source.ptr1253 = inttoptr i64 %dma.source.opaque1252 to ptr addrspace(1), !dbg !20
  %address.local758 = getelementptr i8, ptr addrspace(3) %address.local745, i64 1024, !dbg !9
  %address.local768 = getelementptr i8, ptr addrspace(1) %address.local754, i64 4096, !dbg !20
  %dma.source.bits1254 = ptrtoint ptr addrspace(1) %address.local768 to i64, !dbg !20
  %dma.source.opaque1255 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1254) #9, !dbg !20
  %dma.source.ptr1256 = inttoptr i64 %dma.source.opaque1255 to ptr addrspace(1), !dbg !20
  %address.local772 = getelementptr i8, ptr addrspace(3) %address.local745, i64 2048, !dbg !9
  %address.local782 = getelementptr i8, ptr addrspace(1) %address.local754, i64 8192, !dbg !20
  %dma.source.bits1257 = ptrtoint ptr addrspace(1) %address.local782 to i64, !dbg !20
  %dma.source.opaque1258 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1257) #9, !dbg !20
  %dma.source.ptr1259 = inttoptr i64 %dma.source.opaque1258 to ptr addrspace(1), !dbg !20
  %address.local786 = getelementptr i8, ptr addrspace(3) %address.local745, i64 3072, !dbg !9
  %address.local796 = getelementptr i8, ptr addrspace(1) %address.local754, i64 12288, !dbg !20
  %dma.source.bits1260 = ptrtoint ptr addrspace(1) %address.local796 to i64, !dbg !20
  %dma.source.opaque1261 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1260) #9, !dbg !20
  %dma.source.ptr1262 = inttoptr i64 %dma.source.opaque1261 to ptr addrspace(1), !dbg !20
  %address.local800 = getelementptr i8, ptr addrspace(3) %address.local745, i64 4096, !dbg !9
  %address.local810 = getelementptr i8, ptr addrspace(1) %address.local754, i64 16384, !dbg !20
  %dma.source.bits1263 = ptrtoint ptr addrspace(1) %address.local810 to i64, !dbg !20
  %dma.source.opaque1264 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1263) #9, !dbg !20
  %dma.source.ptr1265 = inttoptr i64 %dma.source.opaque1264 to ptr addrspace(1), !dbg !20
  %address.local814 = getelementptr i8, ptr addrspace(3) %address.local745, i64 5120, !dbg !9
  %address.local824 = getelementptr i8, ptr addrspace(1) %address.local754, i64 20480, !dbg !20
  %dma.source.bits1266 = ptrtoint ptr addrspace(1) %address.local824 to i64, !dbg !20
  %dma.source.opaque1267 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1266) #9, !dbg !20
  %dma.source.ptr1268 = inttoptr i64 %dma.source.opaque1267 to ptr addrspace(1), !dbg !20
  %address.local828 = getelementptr i8, ptr addrspace(3) %address.local745, i64 6144, !dbg !9
  %address.local838 = getelementptr i8, ptr addrspace(1) %address.local754, i64 24576, !dbg !20
  %dma.source.bits1269 = ptrtoint ptr addrspace(1) %address.local838 to i64, !dbg !20
  %dma.source.opaque1270 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1269) #9, !dbg !20
  %dma.source.ptr1271 = inttoptr i64 %dma.source.opaque1270 to ptr addrspace(1), !dbg !20
  %address.local842 = getelementptr i8, ptr addrspace(3) %address.local745, i64 7168, !dbg !9
  %311 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local745, i64 0, !dbg !9
  %312 = insertelement <8 x ptr addrspace(3)> %311, ptr addrspace(3) %address.local758, i64 1, !dbg !9
  %313 = insertelement <8 x ptr addrspace(3)> %312, ptr addrspace(3) %address.local772, i64 2, !dbg !9
  %314 = insertelement <8 x ptr addrspace(3)> %313, ptr addrspace(3) %address.local786, i64 3, !dbg !9
  %315 = insertelement <8 x ptr addrspace(3)> %314, ptr addrspace(3) %address.local800, i64 4, !dbg !9
  %316 = insertelement <8 x ptr addrspace(3)> %315, ptr addrspace(3) %address.local814, i64 5, !dbg !9
  %317 = insertelement <8 x ptr addrspace(3)> %316, ptr addrspace(3) %address.local828, i64 6, !dbg !9
  %318 = insertelement <8 x ptr addrspace(3)> %317, ptr addrspace(3) %address.local842, i64 7, !dbg !9
  %319 = ptrtoint <8 x ptr addrspace(3)> %318 to <8 x i64>, !dbg !9
  %320 = trunc <8 x i64> %319 to <8 x i32>, !dbg !9
  %321 = extractelement <8 x i32> %320, i64 0, !dbg !20
  %322 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %321), !dbg !20
  %323 = zext i32 %322 to i64, !dbg !20
  %324 = inttoptr i64 %323 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1253, ptr addrspace(3) %324, i32 4, i32 0, i32 0), !dbg !20
  %325 = extractelement <8 x i32> %320, i64 1, !dbg !20
  %326 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %325), !dbg !20
  %327 = zext i32 %326 to i64, !dbg !20
  %328 = inttoptr i64 %327 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1256, ptr addrspace(3) %328, i32 4, i32 0, i32 0), !dbg !20
  %329 = extractelement <8 x i32> %320, i64 2, !dbg !20
  %330 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %329), !dbg !20
  %331 = zext i32 %330 to i64, !dbg !20
  %332 = inttoptr i64 %331 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1259, ptr addrspace(3) %332, i32 4, i32 0, i32 0), !dbg !20
  %333 = extractelement <8 x i32> %320, i64 3, !dbg !20
  %334 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %333), !dbg !20
  %335 = zext i32 %334 to i64, !dbg !20
  %336 = inttoptr i64 %335 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1262, ptr addrspace(3) %336, i32 4, i32 0, i32 0), !dbg !20
  %337 = extractelement <8 x i32> %320, i64 4, !dbg !20
  %338 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %337), !dbg !20
  %339 = zext i32 %338 to i64, !dbg !20
  %340 = inttoptr i64 %339 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1265, ptr addrspace(3) %340, i32 4, i32 0, i32 0), !dbg !20
  %341 = extractelement <8 x i32> %320, i64 5, !dbg !20
  %342 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %341), !dbg !20
  %343 = zext i32 %342 to i64, !dbg !20
  %344 = inttoptr i64 %343 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1268, ptr addrspace(3) %344, i32 4, i32 0, i32 0), !dbg !20
  %345 = extractelement <8 x i32> %320, i64 6, !dbg !20
  %346 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %345), !dbg !20
  %347 = zext i32 %346 to i64, !dbg !20
  %348 = inttoptr i64 %347 to ptr addrspace(3), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1271, ptr addrspace(3) %348, i32 4, i32 0, i32 0), !dbg !20
  %349 = extractelement <8 x i32> %320, i64 7, !dbg !20
  %350 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %349), !dbg !20
  %351 = zext i32 %350 to i64, !dbg !20
  %352 = inttoptr i64 %351 to ptr addrspace(3), !dbg !20
  %address.local852 = getelementptr i8, ptr addrspace(1) %address.local754, i64 28672, !dbg !20
  %dma.source.bits1272 = ptrtoint ptr addrspace(1) %address.local852 to i64, !dbg !20
  %dma.source.opaque1273 = tail call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1272) #9, !dbg !20
  %dma.source.ptr1274 = inttoptr i64 %dma.source.opaque1273 to ptr addrspace(1), !dbg !20
  tail call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1274, ptr addrspace(3) %352, i32 4, i32 0, i32 0), !dbg !20
  tail call void @llvm.amdgcn.asyncmark(), !dbg !20
  %353 = extractelement <4 x float> %196, i64 0, !dbg !12
  %354 = extractelement <4 x float> %196, i64 1, !dbg !12
  %355 = extractelement <4 x float> %196, i64 2, !dbg !12
  %356 = extractelement <4 x float> %196, i64 3, !dbg !12
  %357 = select i1 %.not, float %354, float %353, !dbg !12
  %358 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !12
  %359 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %358), !dbg !12
  %360 = add nuw nsw i32 %359, 64, !dbg !12
  %361 = xor i32 %359, 16, !dbg !12
  %362 = and i32 %360, 192, !dbg !12
  %363 = icmp samesign ult i32 %361, %362, !dbg !12
  %364 = select i1 %363, i32 %361, i32 %359, !dbg !12
  %365 = shl nuw nsw i32 %364, 2, !dbg !12
  %366 = bitcast float %357 to i32, !dbg !12
  %367 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %366), !dbg !12
  %368 = bitcast i32 %367 to float, !dbg !12
  %369 = select i1 %.not, float %353, float %368, !dbg !12
  %370 = select i1 %.not, float %368, float %354, !dbg !12
  %371 = select i1 %.not, float %356, float %355, !dbg !12
  %372 = bitcast float %371 to i32, !dbg !12
  %373 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %372), !dbg !12
  %374 = bitcast i32 %373 to float, !dbg !12
  %375 = select i1 %.not, float %355, float %374, !dbg !12
  %376 = select i1 %.not, float %374, float %356, !dbg !12
  %377 = select i1 %.not314, float %375, float %369, !dbg !12
  %378 = xor i32 %359, 32, !dbg !12
  %379 = icmp samesign ult i32 %378, %362, !dbg !12
  %380 = select i1 %379, i32 %378, i32 %359, !dbg !12
  %381 = shl nuw nsw i32 %380, 2, !dbg !12
  %382 = bitcast float %377 to i32, !dbg !12
  %383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %382), !dbg !12
  %384 = bitcast i32 %383 to float, !dbg !12
  %385 = select i1 %.not314, float %369, float %384, !dbg !12
  %386 = select i1 %.not314, float %384, float %375, !dbg !12
  %387 = select i1 %.not314, float %376, float %370, !dbg !12
  %388 = bitcast float %387 to i32, !dbg !12
  %389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %388), !dbg !12
  %390 = bitcast i32 %389 to float, !dbg !12
  %391 = select i1 %.not314, float %370, float %390, !dbg !12
  %392 = select i1 %.not314, float %390, float %376, !dbg !12
  %393 = lshr i64 %49, 25, !dbg !21
  %compare.high950 = trunc nuw i64 %393 to i32, !dbg !21
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high950, !dbg !21
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high950, !dbg !21
  %.tr = trunc i64 %49 to i32, !dbg !21
  %394 = shl i32 %.tr, 7, !dbg !21
  %compare.low.base951 = or disjoint i32 %28, %394, !dbg !21
  %compare.low = icmp ult i32 %compare.low.base951, %compare.low.base, !dbg !21
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !21
  %395 = select i1 %compare.result, float 0.000000e+00, float -inf, !dbg !22
  %396 = fadd float %395, %385, !dbg !13
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 1) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local = or i32 %compare.local.base, 1, !dbg !21
  %compare.low952 = icmp ult i32 %compare.local, %compare.low.base, !dbg !21
  %compare.result953 = select i1 %compare.high.equal, i1 %compare.low952, i1 %compare.high.order, !dbg !21
  %397 = select i1 %compare.result953, float 0.000000e+00, float -inf, !dbg !22
  %398 = fadd float %391, %397, !dbg !13
  %compare.local.base954 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 2) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local955 = or i32 %compare.local.base954, 2, !dbg !21
  %compare.low956 = icmp ult i32 %compare.local955, %compare.low.base, !dbg !21
  %compare.result957 = select i1 %compare.high.equal, i1 %compare.low956, i1 %compare.high.order, !dbg !21
  %399 = select i1 %compare.result957, float 0.000000e+00, float -inf, !dbg !22
  %400 = fadd float %386, %399, !dbg !13
  %compare.local.base958 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 3) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local959 = or i32 %compare.local.base958, 3, !dbg !21
  %compare.low960 = icmp ult i32 %compare.local959, %compare.low.base, !dbg !21
  %compare.result961 = select i1 %compare.high.equal, i1 %compare.low960, i1 %compare.high.order, !dbg !21
  %401 = select i1 %compare.result961, float 0.000000e+00, float -inf, !dbg !22
  %402 = fadd float %392, %401, !dbg !13
  %403 = tail call float @__ocml_exp2_f32(float %396), !dbg !23
  %404 = insertelement <4 x float> poison, float %403, i64 0, !dbg !24
  %405 = fadd float %403, 0.000000e+00, !dbg !25
  %406 = tail call float @__ocml_exp2_f32(float %398), !dbg !23
  %407 = insertelement <4 x float> %404, float %406, i64 1, !dbg !24
  %408 = fadd float %405, %406, !dbg !25
  %409 = tail call float @__ocml_exp2_f32(float %400), !dbg !23
  %410 = insertelement <4 x float> %407, float %409, i64 2, !dbg !24
  %411 = fadd float %408, %409, !dbg !25
  %412 = tail call float @__ocml_exp2_f32(float %402), !dbg !23
  %413 = insertelement <4 x float> %410, float %412, i64 3, !dbg !24
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %413) #8, !dbg !24, !frisk.register_lifetimes !5
  %414 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !24
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %414) #8, !dbg !25, !frisk.register_lifetimes !5
  %415 = fadd float %411, %412, !dbg !25
  %416 = extractelement <4 x float> %202, i64 0, !dbg !12
  %417 = extractelement <4 x float> %202, i64 1, !dbg !12
  %418 = extractelement <4 x float> %202, i64 2, !dbg !12
  %419 = extractelement <4 x float> %202, i64 3, !dbg !12
  %420 = select i1 %.not, float %417, float %416, !dbg !12
  %421 = bitcast float %420 to i32, !dbg !12
  %422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %421), !dbg !12
  %423 = bitcast i32 %422 to float, !dbg !12
  %424 = select i1 %.not, float %416, float %423, !dbg !12
  %425 = select i1 %.not, float %423, float %417, !dbg !12
  %426 = select i1 %.not, float %419, float %418, !dbg !12
  %427 = bitcast float %426 to i32, !dbg !12
  %428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %427), !dbg !12
  %429 = bitcast i32 %428 to float, !dbg !12
  %430 = select i1 %.not, float %418, float %429, !dbg !12
  %431 = select i1 %.not, float %429, float %419, !dbg !12
  %432 = select i1 %.not314, float %430, float %424, !dbg !12
  %433 = bitcast float %432 to i32, !dbg !12
  %434 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %433), !dbg !12
  %435 = bitcast i32 %434 to float, !dbg !12
  %436 = select i1 %.not314, float %424, float %435, !dbg !12
  %437 = select i1 %.not314, float %435, float %430, !dbg !12
  %438 = select i1 %.not314, float %431, float %425, !dbg !12
  %439 = bitcast float %438 to i32, !dbg !12
  %440 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %439), !dbg !12
  %441 = bitcast i32 %440 to float, !dbg !12
  %442 = select i1 %.not314, float %425, float %441, !dbg !12
  %443 = select i1 %.not314, float %441, float %431, !dbg !12
  %compare.local.base962 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 4) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local963 = or i32 %compare.local.base962, 16, !dbg !21
  %compare.low964 = icmp ult i32 %compare.local963, %compare.low.base, !dbg !21
  %compare.result965 = select i1 %compare.high.equal, i1 %compare.low964, i1 %compare.high.order, !dbg !21
  %444 = select i1 %compare.result965, float 0.000000e+00, float -inf, !dbg !22
  %445 = fadd float %436, %444, !dbg !13
  %compare.local.base966 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 5) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local967 = or i32 %compare.local.base966, 17, !dbg !21
  %compare.low968 = icmp ult i32 %compare.local967, %compare.low.base, !dbg !21
  %compare.result969 = select i1 %compare.high.equal, i1 %compare.low968, i1 %compare.high.order, !dbg !21
  %446 = select i1 %compare.result969, float 0.000000e+00, float -inf, !dbg !22
  %447 = fadd float %442, %446, !dbg !13
  %compare.local.base970 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 6) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local971 = or i32 %compare.local.base970, 18, !dbg !21
  %compare.low972 = icmp ult i32 %compare.local971, %compare.low.base, !dbg !21
  %compare.result973 = select i1 %compare.high.equal, i1 %compare.low972, i1 %compare.high.order, !dbg !21
  %448 = select i1 %compare.result973, float 0.000000e+00, float -inf, !dbg !22
  %449 = fadd float %437, %448, !dbg !13
  %compare.local.base974 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 7) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local975 = or i32 %compare.local.base974, 19, !dbg !21
  %compare.low976 = icmp ult i32 %compare.local975, %compare.low.base, !dbg !21
  %compare.result977 = select i1 %compare.high.equal, i1 %compare.low976, i1 %compare.high.order, !dbg !21
  %450 = select i1 %compare.result977, float 0.000000e+00, float -inf, !dbg !22
  %451 = fadd float %443, %450, !dbg !13
  %452 = tail call float @__ocml_exp2_f32(float %445), !dbg !23
  %453 = insertelement <4 x float> poison, float %452, i64 0, !dbg !24
  %454 = fadd float %415, %452, !dbg !25
  %455 = tail call float @__ocml_exp2_f32(float %447), !dbg !23
  %456 = insertelement <4 x float> %453, float %455, i64 1, !dbg !24
  %457 = fadd float %454, %455, !dbg !25
  %458 = tail call float @__ocml_exp2_f32(float %449), !dbg !23
  %459 = insertelement <4 x float> %456, float %458, i64 2, !dbg !24
  %460 = fadd float %457, %458, !dbg !25
  %461 = tail call float @__ocml_exp2_f32(float %451), !dbg !23
  %462 = insertelement <4 x float> %459, float %461, i64 3, !dbg !24
  %math.result387 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %462) #8, !dbg !24, !frisk.register_lifetimes !5
  %463 = fptrunc <4 x float> %math.result387 to <4 x half>, !dbg !24
  %math.packed388 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %463) #8, !dbg !25, !frisk.register_lifetimes !5
  %464 = fadd float %460, %461, !dbg !25
  %465 = extractelement <4 x float> %208, i64 0, !dbg !12
  %466 = extractelement <4 x float> %208, i64 1, !dbg !12
  %467 = extractelement <4 x float> %208, i64 2, !dbg !12
  %468 = extractelement <4 x float> %208, i64 3, !dbg !12
  %469 = select i1 %.not, float %466, float %465, !dbg !12
  %470 = bitcast float %469 to i32, !dbg !12
  %471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %470), !dbg !12
  %472 = bitcast i32 %471 to float, !dbg !12
  %473 = select i1 %.not, float %465, float %472, !dbg !12
  %474 = select i1 %.not, float %472, float %466, !dbg !12
  %475 = select i1 %.not, float %468, float %467, !dbg !12
  %476 = bitcast float %475 to i32, !dbg !12
  %477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %476), !dbg !12
  %478 = bitcast i32 %477 to float, !dbg !12
  %479 = select i1 %.not, float %467, float %478, !dbg !12
  %480 = select i1 %.not, float %478, float %468, !dbg !12
  %481 = select i1 %.not314, float %479, float %473, !dbg !12
  %482 = bitcast float %481 to i32, !dbg !12
  %483 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %482), !dbg !12
  %484 = bitcast i32 %483 to float, !dbg !12
  %485 = select i1 %.not314, float %473, float %484, !dbg !12
  %486 = select i1 %.not314, float %484, float %479, !dbg !12
  %487 = select i1 %.not314, float %480, float %474, !dbg !12
  %488 = bitcast float %487 to i32, !dbg !12
  %489 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %488), !dbg !12
  %490 = bitcast i32 %489 to float, !dbg !12
  %491 = select i1 %.not314, float %474, float %490, !dbg !12
  %492 = select i1 %.not314, float %490, float %480, !dbg !12
  %compare.local.base978 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 8) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local979 = or i32 %compare.local.base978, 32, !dbg !21
  %compare.low980 = icmp ult i32 %compare.local979, %compare.low.base, !dbg !21
  %compare.result981 = select i1 %compare.high.equal, i1 %compare.low980, i1 %compare.high.order, !dbg !21
  %493 = select i1 %compare.result981, float 0.000000e+00, float -inf, !dbg !22
  %494 = fadd float %485, %493, !dbg !13
  %compare.local.base982 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 9) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local983 = or i32 %compare.local.base982, 33, !dbg !21
  %compare.low984 = icmp ult i32 %compare.local983, %compare.low.base, !dbg !21
  %compare.result985 = select i1 %compare.high.equal, i1 %compare.low984, i1 %compare.high.order, !dbg !21
  %495 = select i1 %compare.result985, float 0.000000e+00, float -inf, !dbg !22
  %496 = fadd float %491, %495, !dbg !13
  %compare.local.base986 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 10) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local987 = or i32 %compare.local.base986, 34, !dbg !21
  %compare.low988 = icmp ult i32 %compare.local987, %compare.low.base, !dbg !21
  %compare.result989 = select i1 %compare.high.equal, i1 %compare.low988, i1 %compare.high.order, !dbg !21
  %497 = select i1 %compare.result989, float 0.000000e+00, float -inf, !dbg !22
  %498 = fadd float %486, %497, !dbg !13
  %compare.local.base990 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 11) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local991 = or i32 %compare.local.base990, 35, !dbg !21
  %compare.low992 = icmp ult i32 %compare.local991, %compare.low.base, !dbg !21
  %compare.result993 = select i1 %compare.high.equal, i1 %compare.low992, i1 %compare.high.order, !dbg !21
  %499 = select i1 %compare.result993, float 0.000000e+00, float -inf, !dbg !22
  %500 = fadd float %492, %499, !dbg !13
  %501 = tail call float @__ocml_exp2_f32(float %494), !dbg !23
  %502 = insertelement <4 x float> poison, float %501, i64 0, !dbg !24
  %503 = fadd float %464, %501, !dbg !25
  %504 = tail call float @__ocml_exp2_f32(float %496), !dbg !23
  %505 = insertelement <4 x float> %502, float %504, i64 1, !dbg !24
  %506 = fadd float %503, %504, !dbg !25
  %507 = tail call float @__ocml_exp2_f32(float %498), !dbg !23
  %508 = insertelement <4 x float> %505, float %507, i64 2, !dbg !24
  %509 = fadd float %506, %507, !dbg !25
  %510 = tail call float @__ocml_exp2_f32(float %500), !dbg !23
  %511 = insertelement <4 x float> %508, float %510, i64 3, !dbg !24
  %math.result389 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %511) #8, !dbg !24, !frisk.register_lifetimes !5
  %512 = fptrunc <4 x float> %math.result389 to <4 x half>, !dbg !24
  %math.packed390 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %512) #8, !dbg !25, !frisk.register_lifetimes !5
  %513 = fadd float %509, %510, !dbg !25
  %514 = extractelement <4 x float> %215, i64 0, !dbg !12
  %515 = extractelement <4 x float> %215, i64 1, !dbg !12
  %516 = extractelement <4 x float> %215, i64 2, !dbg !12
  %517 = extractelement <4 x float> %215, i64 3, !dbg !12
  %518 = select i1 %.not, float %515, float %514, !dbg !12
  %519 = bitcast float %518 to i32, !dbg !12
  %520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %519), !dbg !12
  %521 = bitcast i32 %520 to float, !dbg !12
  %522 = select i1 %.not, float %514, float %521, !dbg !12
  %523 = select i1 %.not, float %521, float %515, !dbg !12
  %524 = select i1 %.not, float %517, float %516, !dbg !12
  %525 = bitcast float %524 to i32, !dbg !12
  %526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %525), !dbg !12
  %527 = bitcast i32 %526 to float, !dbg !12
  %528 = select i1 %.not, float %516, float %527, !dbg !12
  %529 = select i1 %.not, float %527, float %517, !dbg !12
  %530 = select i1 %.not314, float %528, float %522, !dbg !12
  %531 = bitcast float %530 to i32, !dbg !12
  %532 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %531), !dbg !12
  %533 = bitcast i32 %532 to float, !dbg !12
  %534 = select i1 %.not314, float %522, float %533, !dbg !12
  %535 = select i1 %.not314, float %533, float %528, !dbg !12
  %536 = select i1 %.not314, float %529, float %523, !dbg !12
  %537 = bitcast float %536 to i32, !dbg !12
  %538 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %537), !dbg !12
  %539 = bitcast i32 %538 to float, !dbg !12
  %540 = select i1 %.not314, float %523, float %539, !dbg !12
  %541 = select i1 %.not314, float %539, float %529, !dbg !12
  %compare.local.base994 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 12) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local995 = or i32 %compare.local.base994, 48, !dbg !21
  %compare.low996 = icmp ult i32 %compare.local995, %compare.low.base, !dbg !21
  %compare.result997 = select i1 %compare.high.equal, i1 %compare.low996, i1 %compare.high.order, !dbg !21
  %542 = select i1 %compare.result997, float 0.000000e+00, float -inf, !dbg !22
  %543 = fadd float %534, %542, !dbg !13
  %compare.local.base998 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 13) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local999 = or i32 %compare.local.base998, 49, !dbg !21
  %compare.low1000 = icmp ult i32 %compare.local999, %compare.low.base, !dbg !21
  %compare.result1001 = select i1 %compare.high.equal, i1 %compare.low1000, i1 %compare.high.order, !dbg !21
  %544 = select i1 %compare.result1001, float 0.000000e+00, float -inf, !dbg !22
  %545 = fadd float %540, %544, !dbg !13
  %compare.local.base1002 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 14) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1003 = or i32 %compare.local.base1002, 50, !dbg !21
  %compare.low1004 = icmp ult i32 %compare.local1003, %compare.low.base, !dbg !21
  %compare.result1005 = select i1 %compare.high.equal, i1 %compare.low1004, i1 %compare.high.order, !dbg !21
  %546 = select i1 %compare.result1005, float 0.000000e+00, float -inf, !dbg !22
  %547 = fadd float %535, %546, !dbg !13
  %compare.local.base1006 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 15) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1007 = or i32 %compare.local.base1006, 51, !dbg !21
  %compare.low1008 = icmp ult i32 %compare.local1007, %compare.low.base, !dbg !21
  %compare.result1009 = select i1 %compare.high.equal, i1 %compare.low1008, i1 %compare.high.order, !dbg !21
  %548 = select i1 %compare.result1009, float 0.000000e+00, float -inf, !dbg !22
  %549 = fadd float %541, %548, !dbg !13
  %550 = tail call float @__ocml_exp2_f32(float %543), !dbg !23
  %551 = insertelement <4 x float> poison, float %550, i64 0, !dbg !24
  %552 = fadd float %513, %550, !dbg !25
  %553 = tail call float @__ocml_exp2_f32(float %545), !dbg !23
  %554 = insertelement <4 x float> %551, float %553, i64 1, !dbg !24
  %555 = fadd float %552, %553, !dbg !25
  %556 = tail call float @__ocml_exp2_f32(float %547), !dbg !23
  %557 = insertelement <4 x float> %554, float %556, i64 2, !dbg !24
  %558 = fadd float %555, %556, !dbg !25
  %559 = tail call float @__ocml_exp2_f32(float %549), !dbg !23
  %560 = insertelement <4 x float> %557, float %559, i64 3, !dbg !24
  %math.result391 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %560) #8, !dbg !24, !frisk.register_lifetimes !5
  %561 = fptrunc <4 x float> %math.result391 to <4 x half>, !dbg !24
  %math.packed392 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %561) #8, !dbg !25, !frisk.register_lifetimes !5
  %562 = fadd float %558, %559, !dbg !25
  %563 = extractelement <4 x float> %222, i64 0, !dbg !12
  %564 = extractelement <4 x float> %222, i64 1, !dbg !12
  %565 = extractelement <4 x float> %222, i64 2, !dbg !12
  %566 = extractelement <4 x float> %222, i64 3, !dbg !12
  %567 = select i1 %.not, float %564, float %563, !dbg !12
  %568 = bitcast float %567 to i32, !dbg !12
  %569 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %568), !dbg !12
  %570 = bitcast i32 %569 to float, !dbg !12
  %571 = select i1 %.not, float %563, float %570, !dbg !12
  %572 = select i1 %.not, float %570, float %564, !dbg !12
  %573 = select i1 %.not, float %566, float %565, !dbg !12
  %574 = bitcast float %573 to i32, !dbg !12
  %575 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %574), !dbg !12
  %576 = bitcast i32 %575 to float, !dbg !12
  %577 = select i1 %.not, float %565, float %576, !dbg !12
  %578 = select i1 %.not, float %576, float %566, !dbg !12
  %579 = select i1 %.not314, float %577, float %571, !dbg !12
  %580 = bitcast float %579 to i32, !dbg !12
  %581 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %580), !dbg !12
  %582 = bitcast i32 %581 to float, !dbg !12
  %583 = select i1 %.not314, float %571, float %582, !dbg !12
  %584 = select i1 %.not314, float %582, float %577, !dbg !12
  %585 = select i1 %.not314, float %578, float %572, !dbg !12
  %586 = bitcast float %585 to i32, !dbg !12
  %587 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %586), !dbg !12
  %588 = bitcast i32 %587 to float, !dbg !12
  %589 = select i1 %.not314, float %572, float %588, !dbg !12
  %590 = select i1 %.not314, float %588, float %578, !dbg !12
  %compare.local.base1010 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 16) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1011 = or i32 %compare.local.base1010, 64, !dbg !21
  %compare.low1012 = icmp ult i32 %compare.local1011, %compare.low.base, !dbg !21
  %compare.result1013 = select i1 %compare.high.equal, i1 %compare.low1012, i1 %compare.high.order, !dbg !21
  %591 = select i1 %compare.result1013, float 0.000000e+00, float -inf, !dbg !22
  %592 = fadd float %583, %591, !dbg !13
  %compare.local.base1014 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 17) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1015 = or i32 %compare.local.base1014, 65, !dbg !21
  %compare.low1016 = icmp ult i32 %compare.local1015, %compare.low.base, !dbg !21
  %compare.result1017 = select i1 %compare.high.equal, i1 %compare.low1016, i1 %compare.high.order, !dbg !21
  %593 = select i1 %compare.result1017, float 0.000000e+00, float -inf, !dbg !22
  %594 = fadd float %589, %593, !dbg !13
  %compare.local.base1018 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 18) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1019 = or i32 %compare.local.base1018, 66, !dbg !21
  %compare.low1020 = icmp ult i32 %compare.local1019, %compare.low.base, !dbg !21
  %compare.result1021 = select i1 %compare.high.equal, i1 %compare.low1020, i1 %compare.high.order, !dbg !21
  %595 = select i1 %compare.result1021, float 0.000000e+00, float -inf, !dbg !22
  %596 = fadd float %584, %595, !dbg !13
  %compare.local.base1022 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 19) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1023 = or i32 %compare.local.base1022, 67, !dbg !21
  %compare.low1024 = icmp ult i32 %compare.local1023, %compare.low.base, !dbg !21
  %compare.result1025 = select i1 %compare.high.equal, i1 %compare.low1024, i1 %compare.high.order, !dbg !21
  %597 = select i1 %compare.result1025, float 0.000000e+00, float -inf, !dbg !22
  %598 = fadd float %590, %597, !dbg !13
  %599 = tail call float @__ocml_exp2_f32(float %592), !dbg !23
  %600 = insertelement <4 x float> poison, float %599, i64 0, !dbg !24
  %601 = fadd float %562, %599, !dbg !25
  %602 = tail call float @__ocml_exp2_f32(float %594), !dbg !23
  %603 = insertelement <4 x float> %600, float %602, i64 1, !dbg !24
  %604 = fadd float %601, %602, !dbg !25
  %605 = tail call float @__ocml_exp2_f32(float %596), !dbg !23
  %606 = insertelement <4 x float> %603, float %605, i64 2, !dbg !24
  %607 = fadd float %604, %605, !dbg !25
  %608 = tail call float @__ocml_exp2_f32(float %598), !dbg !23
  %609 = insertelement <4 x float> %606, float %608, i64 3, !dbg !24
  %math.result393 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %609) #8, !dbg !24, !frisk.register_lifetimes !5
  %610 = fptrunc <4 x float> %math.result393 to <4 x half>, !dbg !24
  %math.packed394 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %610) #8, !dbg !25, !frisk.register_lifetimes !5
  %611 = fadd float %607, %608, !dbg !25
  %612 = extractelement <4 x float> %229, i64 0, !dbg !12
  %613 = extractelement <4 x float> %229, i64 1, !dbg !12
  %614 = extractelement <4 x float> %229, i64 2, !dbg !12
  %615 = extractelement <4 x float> %229, i64 3, !dbg !12
  %616 = select i1 %.not, float %613, float %612, !dbg !12
  %617 = bitcast float %616 to i32, !dbg !12
  %618 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %617), !dbg !12
  %619 = bitcast i32 %618 to float, !dbg !12
  %620 = select i1 %.not, float %612, float %619, !dbg !12
  %621 = select i1 %.not, float %619, float %613, !dbg !12
  %622 = select i1 %.not, float %615, float %614, !dbg !12
  %623 = bitcast float %622 to i32, !dbg !12
  %624 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %623), !dbg !12
  %625 = bitcast i32 %624 to float, !dbg !12
  %626 = select i1 %.not, float %614, float %625, !dbg !12
  %627 = select i1 %.not, float %625, float %615, !dbg !12
  %628 = select i1 %.not314, float %626, float %620, !dbg !12
  %629 = bitcast float %628 to i32, !dbg !12
  %630 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %629), !dbg !12
  %631 = bitcast i32 %630 to float, !dbg !12
  %632 = select i1 %.not314, float %620, float %631, !dbg !12
  %633 = select i1 %.not314, float %631, float %626, !dbg !12
  %634 = select i1 %.not314, float %627, float %621, !dbg !12
  %635 = bitcast float %634 to i32, !dbg !12
  %636 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %635), !dbg !12
  %637 = bitcast i32 %636 to float, !dbg !12
  %638 = select i1 %.not314, float %621, float %637, !dbg !12
  %639 = select i1 %.not314, float %637, float %627, !dbg !12
  %compare.local.base1026 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 20) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1027 = or i32 %compare.local.base1026, 80, !dbg !21
  %compare.low1028 = icmp ult i32 %compare.local1027, %compare.low.base, !dbg !21
  %compare.result1029 = select i1 %compare.high.equal, i1 %compare.low1028, i1 %compare.high.order, !dbg !21
  %640 = select i1 %compare.result1029, float 0.000000e+00, float -inf, !dbg !22
  %641 = fadd float %632, %640, !dbg !13
  %compare.local.base1030 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 21) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1031 = or i32 %compare.local.base1030, 81, !dbg !21
  %compare.low1032 = icmp ult i32 %compare.local1031, %compare.low.base, !dbg !21
  %compare.result1033 = select i1 %compare.high.equal, i1 %compare.low1032, i1 %compare.high.order, !dbg !21
  %642 = select i1 %compare.result1033, float 0.000000e+00, float -inf, !dbg !22
  %643 = fadd float %638, %642, !dbg !13
  %compare.local.base1034 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 22) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1035 = or i32 %compare.local.base1034, 82, !dbg !21
  %compare.low1036 = icmp ult i32 %compare.local1035, %compare.low.base, !dbg !21
  %compare.result1037 = select i1 %compare.high.equal, i1 %compare.low1036, i1 %compare.high.order, !dbg !21
  %644 = select i1 %compare.result1037, float 0.000000e+00, float -inf, !dbg !22
  %645 = fadd float %633, %644, !dbg !13
  %compare.local.base1038 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 23) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1039 = or i32 %compare.local.base1038, 83, !dbg !21
  %compare.low1040 = icmp ult i32 %compare.local1039, %compare.low.base, !dbg !21
  %compare.result1041 = select i1 %compare.high.equal, i1 %compare.low1040, i1 %compare.high.order, !dbg !21
  %646 = select i1 %compare.result1041, float 0.000000e+00, float -inf, !dbg !22
  %647 = fadd float %639, %646, !dbg !13
  %648 = tail call float @__ocml_exp2_f32(float %641), !dbg !23
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !24
  %650 = fadd float %611, %648, !dbg !25
  %651 = tail call float @__ocml_exp2_f32(float %643), !dbg !23
  %652 = insertelement <4 x float> %649, float %651, i64 1, !dbg !24
  %653 = fadd float %650, %651, !dbg !25
  %654 = tail call float @__ocml_exp2_f32(float %645), !dbg !23
  %655 = insertelement <4 x float> %652, float %654, i64 2, !dbg !24
  %656 = fadd float %653, %654, !dbg !25
  %657 = tail call float @__ocml_exp2_f32(float %647), !dbg !23
  %658 = insertelement <4 x float> %655, float %657, i64 3, !dbg !24
  %math.result395 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %658) #8, !dbg !24, !frisk.register_lifetimes !5
  %659 = fptrunc <4 x float> %math.result395 to <4 x half>, !dbg !24
  %math.packed396 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %659) #8, !dbg !25, !frisk.register_lifetimes !5
  %660 = fadd float %656, %657, !dbg !25
  %661 = extractelement <4 x float> %236, i64 0, !dbg !12
  %662 = extractelement <4 x float> %236, i64 1, !dbg !12
  %663 = extractelement <4 x float> %236, i64 2, !dbg !12
  %664 = extractelement <4 x float> %236, i64 3, !dbg !12
  %665 = select i1 %.not, float %662, float %661, !dbg !12
  %666 = bitcast float %665 to i32, !dbg !12
  %667 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %666), !dbg !12
  %668 = bitcast i32 %667 to float, !dbg !12
  %669 = select i1 %.not, float %661, float %668, !dbg !12
  %670 = select i1 %.not, float %668, float %662, !dbg !12
  %671 = select i1 %.not, float %664, float %663, !dbg !12
  %672 = bitcast float %671 to i32, !dbg !12
  %673 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %672), !dbg !12
  %674 = bitcast i32 %673 to float, !dbg !12
  %675 = select i1 %.not, float %663, float %674, !dbg !12
  %676 = select i1 %.not, float %674, float %664, !dbg !12
  %677 = select i1 %.not314, float %675, float %669, !dbg !12
  %678 = bitcast float %677 to i32, !dbg !12
  %679 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %678), !dbg !12
  %680 = bitcast i32 %679 to float, !dbg !12
  %681 = select i1 %.not314, float %669, float %680, !dbg !12
  %682 = select i1 %.not314, float %680, float %675, !dbg !12
  %683 = select i1 %.not314, float %676, float %670, !dbg !12
  %684 = bitcast float %683 to i32, !dbg !12
  %685 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %684), !dbg !12
  %686 = bitcast i32 %685 to float, !dbg !12
  %687 = select i1 %.not314, float %670, float %686, !dbg !12
  %688 = select i1 %.not314, float %686, float %676, !dbg !12
  %compare.local.base1042 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 24) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1043 = or i32 %compare.local.base1042, 96, !dbg !21
  %compare.low1044 = icmp ult i32 %compare.local1043, %compare.low.base, !dbg !21
  %compare.result1045 = select i1 %compare.high.equal, i1 %compare.low1044, i1 %compare.high.order, !dbg !21
  %689 = select i1 %compare.result1045, float 0.000000e+00, float -inf, !dbg !22
  %690 = fadd float %681, %689, !dbg !13
  %compare.local.base1046 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 25) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1047 = or i32 %compare.local.base1046, 97, !dbg !21
  %compare.low1048 = icmp ult i32 %compare.local1047, %compare.low.base, !dbg !21
  %compare.result1049 = select i1 %compare.high.equal, i1 %compare.low1048, i1 %compare.high.order, !dbg !21
  %691 = select i1 %compare.result1049, float 0.000000e+00, float -inf, !dbg !22
  %692 = fadd float %687, %691, !dbg !13
  %compare.local.base1050 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 26) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1051 = or i32 %compare.local.base1050, 98, !dbg !21
  %compare.low1052 = icmp ult i32 %compare.local1051, %compare.low.base, !dbg !21
  %compare.result1053 = select i1 %compare.high.equal, i1 %compare.low1052, i1 %compare.high.order, !dbg !21
  %693 = select i1 %compare.result1053, float 0.000000e+00, float -inf, !dbg !22
  %694 = fadd float %682, %693, !dbg !13
  %compare.local.base1054 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 27) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1055 = or i32 %compare.local.base1054, 99, !dbg !21
  %compare.low1056 = icmp ult i32 %compare.local1055, %compare.low.base, !dbg !21
  %compare.result1057 = select i1 %compare.high.equal, i1 %compare.low1056, i1 %compare.high.order, !dbg !21
  %695 = select i1 %compare.result1057, float 0.000000e+00, float -inf, !dbg !22
  %696 = fadd float %688, %695, !dbg !13
  %697 = tail call float @__ocml_exp2_f32(float %690), !dbg !23
  %698 = insertelement <4 x float> poison, float %697, i64 0, !dbg !24
  %699 = fadd float %660, %697, !dbg !25
  %700 = tail call float @__ocml_exp2_f32(float %692), !dbg !23
  %701 = insertelement <4 x float> %698, float %700, i64 1, !dbg !24
  %702 = fadd float %699, %700, !dbg !25
  %703 = tail call float @__ocml_exp2_f32(float %694), !dbg !23
  %704 = insertelement <4 x float> %701, float %703, i64 2, !dbg !24
  %705 = fadd float %702, %703, !dbg !25
  %706 = tail call float @__ocml_exp2_f32(float %696), !dbg !23
  %707 = insertelement <4 x float> %704, float %706, i64 3, !dbg !24
  %math.result397 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #8, !dbg !24, !frisk.register_lifetimes !5
  %708 = fptrunc <4 x float> %math.result397 to <4 x half>, !dbg !24
  %math.packed398 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #8, !dbg !25, !frisk.register_lifetimes !5
  %709 = fadd float %705, %706, !dbg !25
  %710 = extractelement <4 x float> %243, i64 0, !dbg !12
  %711 = extractelement <4 x float> %243, i64 1, !dbg !12
  %712 = extractelement <4 x float> %243, i64 2, !dbg !12
  %713 = extractelement <4 x float> %243, i64 3, !dbg !12
  %714 = select i1 %.not, float %711, float %710, !dbg !12
  %715 = bitcast float %714 to i32, !dbg !12
  %716 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %715), !dbg !12
  %717 = bitcast i32 %716 to float, !dbg !12
  %718 = select i1 %.not, float %710, float %717, !dbg !12
  %719 = select i1 %.not, float %717, float %711, !dbg !12
  %720 = select i1 %.not, float %713, float %712, !dbg !12
  %721 = bitcast float %720 to i32, !dbg !12
  %722 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %721), !dbg !12
  %723 = bitcast i32 %722 to float, !dbg !12
  %724 = select i1 %.not, float %712, float %723, !dbg !12
  %725 = select i1 %.not, float %723, float %713, !dbg !12
  %726 = select i1 %.not314, float %724, float %718, !dbg !12
  %727 = bitcast float %726 to i32, !dbg !12
  %728 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %727), !dbg !12
  %729 = bitcast i32 %728 to float, !dbg !12
  %730 = select i1 %.not314, float %718, float %729, !dbg !12
  %731 = select i1 %.not314, float %729, float %724, !dbg !12
  %732 = select i1 %.not314, float %725, float %719, !dbg !12
  %733 = bitcast float %732 to i32, !dbg !12
  %734 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %733), !dbg !12
  %735 = bitcast i32 %734 to float, !dbg !12
  %736 = select i1 %.not314, float %719, float %735, !dbg !12
  %737 = select i1 %.not314, float %735, float %725, !dbg !12
  %compare.local.base1058 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 28) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1059 = or i32 %compare.local.base1058, 112, !dbg !21
  %compare.low1060 = icmp ult i32 %compare.local1059, %compare.low.base, !dbg !21
  %compare.result1061 = select i1 %compare.high.equal, i1 %compare.low1060, i1 %compare.high.order, !dbg !21
  %738 = select i1 %compare.result1061, float 0.000000e+00, float -inf, !dbg !22
  %739 = fadd float %730, %738, !dbg !13
  %compare.local.base1062 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 29) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1063 = or i32 %compare.local.base1062, 113, !dbg !21
  %compare.low1064 = icmp ult i32 %compare.local1063, %compare.low.base, !dbg !21
  %compare.result1065 = select i1 %compare.high.equal, i1 %compare.low1064, i1 %compare.high.order, !dbg !21
  %740 = select i1 %compare.result1065, float 0.000000e+00, float -inf, !dbg !22
  %741 = fadd float %736, %740, !dbg !13
  %compare.local.base1066 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 30) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1067 = or i32 %compare.local.base1066, 114, !dbg !21
  %compare.low1068 = icmp ult i32 %compare.local1067, %compare.low.base, !dbg !21
  %compare.result1069 = select i1 %compare.high.equal, i1 %compare.low1068, i1 %compare.high.order, !dbg !21
  %742 = select i1 %compare.result1069, float 0.000000e+00, float -inf, !dbg !22
  %743 = fadd float %731, %742, !dbg !13
  %compare.local.base1070 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 31) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1071 = or i32 %compare.local.base1070, 115, !dbg !21
  %compare.low1072 = icmp ult i32 %compare.local1071, %compare.low.base, !dbg !21
  %compare.result1073 = select i1 %compare.high.equal, i1 %compare.low1072, i1 %compare.high.order, !dbg !21
  %744 = select i1 %compare.result1073, float 0.000000e+00, float -inf, !dbg !22
  %745 = fadd float %737, %744, !dbg !13
  %746 = tail call float @__ocml_exp2_f32(float %739), !dbg !23
  %747 = insertelement <4 x float> poison, float %746, i64 0, !dbg !24
  %748 = fadd float %709, %746, !dbg !25
  %749 = tail call float @__ocml_exp2_f32(float %741), !dbg !23
  %750 = insertelement <4 x float> %747, float %749, i64 1, !dbg !24
  %751 = fadd float %748, %749, !dbg !25
  %752 = tail call float @__ocml_exp2_f32(float %743), !dbg !23
  %753 = insertelement <4 x float> %750, float %752, i64 2, !dbg !24
  %754 = fadd float %751, %752, !dbg !25
  %755 = tail call float @__ocml_exp2_f32(float %745), !dbg !23
  %756 = insertelement <4 x float> %753, float %755, i64 3, !dbg !24
  %math.result399 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #8, !dbg !24, !frisk.register_lifetimes !5
  %757 = fptrunc <4 x float> %math.result399 to <4 x half>, !dbg !24
  %math.packed400 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #8, !dbg !25, !frisk.register_lifetimes !5
  %758 = fadd float %754, %755, !dbg !25
  %759 = extractelement <4 x float> %249, i64 0, !dbg !12
  %760 = extractelement <4 x float> %249, i64 1, !dbg !12
  %761 = extractelement <4 x float> %249, i64 2, !dbg !12
  %762 = extractelement <4 x float> %249, i64 3, !dbg !12
  %763 = select i1 %.not, float %760, float %759, !dbg !12
  %764 = bitcast float %763 to i32, !dbg !12
  %765 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %764), !dbg !12
  %766 = bitcast i32 %765 to float, !dbg !12
  %767 = select i1 %.not, float %759, float %766, !dbg !12
  %768 = select i1 %.not, float %766, float %760, !dbg !12
  %769 = select i1 %.not, float %762, float %761, !dbg !12
  %770 = bitcast float %769 to i32, !dbg !12
  %771 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %770), !dbg !12
  %772 = bitcast i32 %771 to float, !dbg !12
  %773 = select i1 %.not, float %761, float %772, !dbg !12
  %774 = select i1 %.not, float %772, float %762, !dbg !12
  %775 = select i1 %.not314, float %773, float %767, !dbg !12
  %776 = bitcast float %775 to i32, !dbg !12
  %777 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %776), !dbg !12
  %778 = bitcast i32 %777 to float, !dbg !12
  %779 = select i1 %.not314, float %767, float %778, !dbg !12
  %780 = select i1 %.not314, float %778, float %773, !dbg !12
  %781 = select i1 %.not314, float %774, float %768, !dbg !12
  %782 = bitcast float %781 to i32, !dbg !12
  %783 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %782), !dbg !12
  %784 = bitcast i32 %783 to float, !dbg !12
  %785 = select i1 %.not314, float %768, float %784, !dbg !12
  %786 = select i1 %.not314, float %784, float %774, !dbg !12
  %compare.high.equal1076 = icmp eq i32 %compare.high1074, %compare.high950, !dbg !21
  %compare.high.order1077 = icmp ugt i32 %compare.high1074, %compare.high950, !dbg !21
  %compare.low1080 = icmp ult i32 %compare.low.base951, %compare.low.base1078, !dbg !21
  %compare.result1081 = select i1 %compare.high.equal1076, i1 %compare.low1080, i1 %compare.high.order1077, !dbg !21
  %787 = select i1 %compare.result1081, float 0.000000e+00, float -inf, !dbg !22
  %788 = fadd float %787, %779, !dbg !13
  %compare.local.base1082 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 32) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1083 = or i32 %compare.local.base1082, 1, !dbg !21
  %compare.low1084 = icmp ult i32 %compare.local1083, %compare.low.base1078, !dbg !21
  %compare.result1085 = select i1 %compare.high.equal1076, i1 %compare.low1084, i1 %compare.high.order1077, !dbg !21
  %789 = select i1 %compare.result1085, float 0.000000e+00, float -inf, !dbg !22
  %790 = fadd float %785, %789, !dbg !13
  %compare.local.base1086 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 33) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1087 = or i32 %compare.local.base1086, 2, !dbg !21
  %compare.low1088 = icmp ult i32 %compare.local1087, %compare.low.base1078, !dbg !21
  %compare.result1089 = select i1 %compare.high.equal1076, i1 %compare.low1088, i1 %compare.high.order1077, !dbg !21
  %791 = select i1 %compare.result1089, float 0.000000e+00, float -inf, !dbg !22
  %792 = fadd float %780, %791, !dbg !13
  %compare.local.base1090 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 34) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1091 = or i32 %compare.local.base1090, 3, !dbg !21
  %compare.low1092 = icmp ult i32 %compare.local1091, %compare.low.base1078, !dbg !21
  %compare.result1093 = select i1 %compare.high.equal1076, i1 %compare.low1092, i1 %compare.high.order1077, !dbg !21
  %793 = select i1 %compare.result1093, float 0.000000e+00, float -inf, !dbg !22
  %794 = fadd float %786, %793, !dbg !13
  %795 = tail call float @__ocml_exp2_f32(float %788), !dbg !23
  %796 = insertelement <4 x float> poison, float %795, i64 0, !dbg !24
  %797 = fadd float %795, 0.000000e+00, !dbg !25
  %798 = tail call float @__ocml_exp2_f32(float %790), !dbg !23
  %799 = insertelement <4 x float> %796, float %798, i64 1, !dbg !24
  %800 = fadd float %797, %798, !dbg !25
  %801 = tail call float @__ocml_exp2_f32(float %792), !dbg !23
  %802 = insertelement <4 x float> %799, float %801, i64 2, !dbg !24
  %803 = fadd float %800, %801, !dbg !25
  %804 = tail call float @__ocml_exp2_f32(float %794), !dbg !23
  %805 = insertelement <4 x float> %802, float %804, i64 3, !dbg !24
  %math.result401 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %805) #8, !dbg !24, !frisk.register_lifetimes !5
  %806 = fptrunc <4 x float> %math.result401 to <4 x half>, !dbg !24
  %math.packed402 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %806) #8, !dbg !25, !frisk.register_lifetimes !5
  %807 = fadd float %803, %804, !dbg !25
  %808 = extractelement <4 x float> %253, i64 0, !dbg !12
  %809 = extractelement <4 x float> %253, i64 1, !dbg !12
  %810 = extractelement <4 x float> %253, i64 2, !dbg !12
  %811 = extractelement <4 x float> %253, i64 3, !dbg !12
  %812 = select i1 %.not, float %809, float %808, !dbg !12
  %813 = bitcast float %812 to i32, !dbg !12
  %814 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %813), !dbg !12
  %815 = bitcast i32 %814 to float, !dbg !12
  %816 = select i1 %.not, float %808, float %815, !dbg !12
  %817 = select i1 %.not, float %815, float %809, !dbg !12
  %818 = select i1 %.not, float %811, float %810, !dbg !12
  %819 = bitcast float %818 to i32, !dbg !12
  %820 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %819), !dbg !12
  %821 = bitcast i32 %820 to float, !dbg !12
  %822 = select i1 %.not, float %810, float %821, !dbg !12
  %823 = select i1 %.not, float %821, float %811, !dbg !12
  %824 = select i1 %.not314, float %822, float %816, !dbg !12
  %825 = bitcast float %824 to i32, !dbg !12
  %826 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %825), !dbg !12
  %827 = bitcast i32 %826 to float, !dbg !12
  %828 = select i1 %.not314, float %816, float %827, !dbg !12
  %829 = select i1 %.not314, float %827, float %822, !dbg !12
  %830 = select i1 %.not314, float %823, float %817, !dbg !12
  %831 = bitcast float %830 to i32, !dbg !12
  %832 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %831), !dbg !12
  %833 = bitcast i32 %832 to float, !dbg !12
  %834 = select i1 %.not314, float %817, float %833, !dbg !12
  %835 = select i1 %.not314, float %833, float %823, !dbg !12
  %compare.local.base1094 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 35) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1095 = or i32 %compare.local.base1094, 16, !dbg !21
  %compare.low1096 = icmp ult i32 %compare.local1095, %compare.low.base1078, !dbg !21
  %compare.result1097 = select i1 %compare.high.equal1076, i1 %compare.low1096, i1 %compare.high.order1077, !dbg !21
  %836 = select i1 %compare.result1097, float 0.000000e+00, float -inf, !dbg !22
  %837 = fadd float %828, %836, !dbg !13
  %compare.local.base1098 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 36) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1099 = or i32 %compare.local.base1098, 17, !dbg !21
  %compare.low1100 = icmp ult i32 %compare.local1099, %compare.low.base1078, !dbg !21
  %compare.result1101 = select i1 %compare.high.equal1076, i1 %compare.low1100, i1 %compare.high.order1077, !dbg !21
  %838 = select i1 %compare.result1101, float 0.000000e+00, float -inf, !dbg !22
  %839 = fadd float %834, %838, !dbg !13
  %compare.local.base1102 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 37) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1103 = or i32 %compare.local.base1102, 18, !dbg !21
  %compare.low1104 = icmp ult i32 %compare.local1103, %compare.low.base1078, !dbg !21
  %compare.result1105 = select i1 %compare.high.equal1076, i1 %compare.low1104, i1 %compare.high.order1077, !dbg !21
  %840 = select i1 %compare.result1105, float 0.000000e+00, float -inf, !dbg !22
  %841 = fadd float %829, %840, !dbg !13
  %compare.local.base1106 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 38) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1107 = or i32 %compare.local.base1106, 19, !dbg !21
  %compare.low1108 = icmp ult i32 %compare.local1107, %compare.low.base1078, !dbg !21
  %compare.result1109 = select i1 %compare.high.equal1076, i1 %compare.low1108, i1 %compare.high.order1077, !dbg !21
  %842 = select i1 %compare.result1109, float 0.000000e+00, float -inf, !dbg !22
  %843 = fadd float %835, %842, !dbg !13
  %844 = tail call float @__ocml_exp2_f32(float %837), !dbg !23
  %845 = insertelement <4 x float> poison, float %844, i64 0, !dbg !24
  %846 = fadd float %807, %844, !dbg !25
  %847 = tail call float @__ocml_exp2_f32(float %839), !dbg !23
  %848 = insertelement <4 x float> %845, float %847, i64 1, !dbg !24
  %849 = fadd float %846, %847, !dbg !25
  %850 = tail call float @__ocml_exp2_f32(float %841), !dbg !23
  %851 = insertelement <4 x float> %848, float %850, i64 2, !dbg !24
  %852 = fadd float %849, %850, !dbg !25
  %853 = tail call float @__ocml_exp2_f32(float %843), !dbg !23
  %854 = insertelement <4 x float> %851, float %853, i64 3, !dbg !24
  %math.result403 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %854) #8, !dbg !24, !frisk.register_lifetimes !5
  %855 = fptrunc <4 x float> %math.result403 to <4 x half>, !dbg !24
  %math.packed404 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %855) #8, !dbg !25, !frisk.register_lifetimes !5
  %856 = fadd float %852, %853, !dbg !25
  %857 = extractelement <4 x float> %257, i64 0, !dbg !12
  %858 = extractelement <4 x float> %257, i64 1, !dbg !12
  %859 = extractelement <4 x float> %257, i64 2, !dbg !12
  %860 = extractelement <4 x float> %257, i64 3, !dbg !12
  %861 = select i1 %.not, float %858, float %857, !dbg !12
  %862 = bitcast float %861 to i32, !dbg !12
  %863 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %862), !dbg !12
  %864 = bitcast i32 %863 to float, !dbg !12
  %865 = select i1 %.not, float %857, float %864, !dbg !12
  %866 = select i1 %.not, float %864, float %858, !dbg !12
  %867 = select i1 %.not, float %860, float %859, !dbg !12
  %868 = bitcast float %867 to i32, !dbg !12
  %869 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %868), !dbg !12
  %870 = bitcast i32 %869 to float, !dbg !12
  %871 = select i1 %.not, float %859, float %870, !dbg !12
  %872 = select i1 %.not, float %870, float %860, !dbg !12
  %873 = select i1 %.not314, float %871, float %865, !dbg !12
  %874 = bitcast float %873 to i32, !dbg !12
  %875 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %874), !dbg !12
  %876 = bitcast i32 %875 to float, !dbg !12
  %877 = select i1 %.not314, float %865, float %876, !dbg !12
  %878 = select i1 %.not314, float %876, float %871, !dbg !12
  %879 = select i1 %.not314, float %872, float %866, !dbg !12
  %880 = bitcast float %879 to i32, !dbg !12
  %881 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %880), !dbg !12
  %882 = bitcast i32 %881 to float, !dbg !12
  %883 = select i1 %.not314, float %866, float %882, !dbg !12
  %884 = select i1 %.not314, float %882, float %872, !dbg !12
  %compare.local.base1110 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 39) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1111 = or i32 %compare.local.base1110, 32, !dbg !21
  %compare.low1112 = icmp ult i32 %compare.local1111, %compare.low.base1078, !dbg !21
  %compare.result1113 = select i1 %compare.high.equal1076, i1 %compare.low1112, i1 %compare.high.order1077, !dbg !21
  %885 = select i1 %compare.result1113, float 0.000000e+00, float -inf, !dbg !22
  %886 = fadd float %877, %885, !dbg !13
  %compare.local.base1114 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 40) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1115 = or i32 %compare.local.base1114, 33, !dbg !21
  %compare.low1116 = icmp ult i32 %compare.local1115, %compare.low.base1078, !dbg !21
  %compare.result1117 = select i1 %compare.high.equal1076, i1 %compare.low1116, i1 %compare.high.order1077, !dbg !21
  %887 = select i1 %compare.result1117, float 0.000000e+00, float -inf, !dbg !22
  %888 = fadd float %883, %887, !dbg !13
  %compare.local.base1118 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 41) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1119 = or i32 %compare.local.base1118, 34, !dbg !21
  %compare.low1120 = icmp ult i32 %compare.local1119, %compare.low.base1078, !dbg !21
  %compare.result1121 = select i1 %compare.high.equal1076, i1 %compare.low1120, i1 %compare.high.order1077, !dbg !21
  %889 = select i1 %compare.result1121, float 0.000000e+00, float -inf, !dbg !22
  %890 = fadd float %878, %889, !dbg !13
  %compare.local.base1122 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 42) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1123 = or i32 %compare.local.base1122, 35, !dbg !21
  %compare.low1124 = icmp ult i32 %compare.local1123, %compare.low.base1078, !dbg !21
  %compare.result1125 = select i1 %compare.high.equal1076, i1 %compare.low1124, i1 %compare.high.order1077, !dbg !21
  %891 = select i1 %compare.result1125, float 0.000000e+00, float -inf, !dbg !22
  %892 = fadd float %884, %891, !dbg !13
  %893 = tail call float @__ocml_exp2_f32(float %886), !dbg !23
  %894 = insertelement <4 x float> poison, float %893, i64 0, !dbg !24
  %895 = fadd float %856, %893, !dbg !25
  %896 = tail call float @__ocml_exp2_f32(float %888), !dbg !23
  %897 = insertelement <4 x float> %894, float %896, i64 1, !dbg !24
  %898 = fadd float %895, %896, !dbg !25
  %899 = tail call float @__ocml_exp2_f32(float %890), !dbg !23
  %900 = insertelement <4 x float> %897, float %899, i64 2, !dbg !24
  %901 = fadd float %898, %899, !dbg !25
  %902 = tail call float @__ocml_exp2_f32(float %892), !dbg !23
  %903 = insertelement <4 x float> %900, float %902, i64 3, !dbg !24
  %math.result405 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %903) #8, !dbg !24, !frisk.register_lifetimes !5
  %904 = fptrunc <4 x float> %math.result405 to <4 x half>, !dbg !24
  %math.packed406 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %904) #8, !dbg !25, !frisk.register_lifetimes !5
  %905 = fadd float %901, %902, !dbg !25
  %906 = extractelement <4 x float> %262, i64 0, !dbg !12
  %907 = extractelement <4 x float> %262, i64 1, !dbg !12
  %908 = extractelement <4 x float> %262, i64 2, !dbg !12
  %909 = extractelement <4 x float> %262, i64 3, !dbg !12
  %910 = select i1 %.not, float %907, float %906, !dbg !12
  %911 = bitcast float %910 to i32, !dbg !12
  %912 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %911), !dbg !12
  %913 = bitcast i32 %912 to float, !dbg !12
  %914 = select i1 %.not, float %906, float %913, !dbg !12
  %915 = select i1 %.not, float %913, float %907, !dbg !12
  %916 = select i1 %.not, float %909, float %908, !dbg !12
  %917 = bitcast float %916 to i32, !dbg !12
  %918 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %917), !dbg !12
  %919 = bitcast i32 %918 to float, !dbg !12
  %920 = select i1 %.not, float %908, float %919, !dbg !12
  %921 = select i1 %.not, float %919, float %909, !dbg !12
  %922 = select i1 %.not314, float %920, float %914, !dbg !12
  %923 = bitcast float %922 to i32, !dbg !12
  %924 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %923), !dbg !12
  %925 = bitcast i32 %924 to float, !dbg !12
  %926 = select i1 %.not314, float %914, float %925, !dbg !12
  %927 = select i1 %.not314, float %925, float %920, !dbg !12
  %928 = select i1 %.not314, float %921, float %915, !dbg !12
  %929 = bitcast float %928 to i32, !dbg !12
  %930 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %929), !dbg !12
  %931 = bitcast i32 %930 to float, !dbg !12
  %932 = select i1 %.not314, float %915, float %931, !dbg !12
  %933 = select i1 %.not314, float %931, float %921, !dbg !12
  %compare.local.base1126 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 43) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1127 = or i32 %compare.local.base1126, 48, !dbg !21
  %compare.low1128 = icmp ult i32 %compare.local1127, %compare.low.base1078, !dbg !21
  %compare.result1129 = select i1 %compare.high.equal1076, i1 %compare.low1128, i1 %compare.high.order1077, !dbg !21
  %934 = select i1 %compare.result1129, float 0.000000e+00, float -inf, !dbg !22
  %935 = fadd float %926, %934, !dbg !13
  %compare.local.base1130 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 44) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1131 = or i32 %compare.local.base1130, 49, !dbg !21
  %compare.low1132 = icmp ult i32 %compare.local1131, %compare.low.base1078, !dbg !21
  %compare.result1133 = select i1 %compare.high.equal1076, i1 %compare.low1132, i1 %compare.high.order1077, !dbg !21
  %936 = select i1 %compare.result1133, float 0.000000e+00, float -inf, !dbg !22
  %937 = fadd float %932, %936, !dbg !13
  %compare.local.base1134 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 45) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1135 = or i32 %compare.local.base1134, 50, !dbg !21
  %compare.low1136 = icmp ult i32 %compare.local1135, %compare.low.base1078, !dbg !21
  %compare.result1137 = select i1 %compare.high.equal1076, i1 %compare.low1136, i1 %compare.high.order1077, !dbg !21
  %938 = select i1 %compare.result1137, float 0.000000e+00, float -inf, !dbg !22
  %939 = fadd float %927, %938, !dbg !13
  %compare.local.base1138 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 46) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1139 = or i32 %compare.local.base1138, 51, !dbg !21
  %compare.low1140 = icmp ult i32 %compare.local1139, %compare.low.base1078, !dbg !21
  %compare.result1141 = select i1 %compare.high.equal1076, i1 %compare.low1140, i1 %compare.high.order1077, !dbg !21
  %940 = select i1 %compare.result1141, float 0.000000e+00, float -inf, !dbg !22
  %941 = fadd float %933, %940, !dbg !13
  %942 = tail call float @__ocml_exp2_f32(float %935), !dbg !23
  %943 = insertelement <4 x float> poison, float %942, i64 0, !dbg !24
  %944 = fadd float %905, %942, !dbg !25
  %945 = tail call float @__ocml_exp2_f32(float %937), !dbg !23
  %946 = insertelement <4 x float> %943, float %945, i64 1, !dbg !24
  %947 = fadd float %944, %945, !dbg !25
  %948 = tail call float @__ocml_exp2_f32(float %939), !dbg !23
  %949 = insertelement <4 x float> %946, float %948, i64 2, !dbg !24
  %950 = fadd float %947, %948, !dbg !25
  %951 = tail call float @__ocml_exp2_f32(float %941), !dbg !23
  %952 = insertelement <4 x float> %949, float %951, i64 3, !dbg !24
  %math.result407 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %952) #8, !dbg !24, !frisk.register_lifetimes !5
  %953 = fptrunc <4 x float> %math.result407 to <4 x half>, !dbg !24
  %math.packed408 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %953) #8, !dbg !25, !frisk.register_lifetimes !5
  %954 = fadd float %950, %951, !dbg !25
  %955 = extractelement <4 x float> %267, i64 0, !dbg !12
  %956 = extractelement <4 x float> %267, i64 1, !dbg !12
  %957 = extractelement <4 x float> %267, i64 2, !dbg !12
  %958 = extractelement <4 x float> %267, i64 3, !dbg !12
  %959 = select i1 %.not, float %956, float %955, !dbg !12
  %960 = bitcast float %959 to i32, !dbg !12
  %961 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %960), !dbg !12
  %962 = bitcast i32 %961 to float, !dbg !12
  %963 = select i1 %.not, float %955, float %962, !dbg !12
  %964 = select i1 %.not, float %962, float %956, !dbg !12
  %965 = select i1 %.not, float %958, float %957, !dbg !12
  %966 = bitcast float %965 to i32, !dbg !12
  %967 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %966), !dbg !12
  %968 = bitcast i32 %967 to float, !dbg !12
  %969 = select i1 %.not, float %957, float %968, !dbg !12
  %970 = select i1 %.not, float %968, float %958, !dbg !12
  %971 = select i1 %.not314, float %969, float %963, !dbg !12
  %972 = bitcast float %971 to i32, !dbg !12
  %973 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %972), !dbg !12
  %974 = bitcast i32 %973 to float, !dbg !12
  %975 = select i1 %.not314, float %963, float %974, !dbg !12
  %976 = select i1 %.not314, float %974, float %969, !dbg !12
  %977 = select i1 %.not314, float %970, float %964, !dbg !12
  %978 = bitcast float %977 to i32, !dbg !12
  %979 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %978), !dbg !12
  %980 = bitcast i32 %979 to float, !dbg !12
  %981 = select i1 %.not314, float %964, float %980, !dbg !12
  %982 = select i1 %.not314, float %980, float %970, !dbg !12
  %compare.local.base1142 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 47) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1143 = or i32 %compare.local.base1142, 64, !dbg !21
  %compare.low1144 = icmp ult i32 %compare.local1143, %compare.low.base1078, !dbg !21
  %compare.result1145 = select i1 %compare.high.equal1076, i1 %compare.low1144, i1 %compare.high.order1077, !dbg !21
  %983 = select i1 %compare.result1145, float 0.000000e+00, float -inf, !dbg !22
  %984 = fadd float %975, %983, !dbg !13
  %compare.local.base1146 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 48) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1147 = or i32 %compare.local.base1146, 65, !dbg !21
  %compare.low1148 = icmp ult i32 %compare.local1147, %compare.low.base1078, !dbg !21
  %compare.result1149 = select i1 %compare.high.equal1076, i1 %compare.low1148, i1 %compare.high.order1077, !dbg !21
  %985 = select i1 %compare.result1149, float 0.000000e+00, float -inf, !dbg !22
  %986 = fadd float %981, %985, !dbg !13
  %compare.local.base1150 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 49) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1151 = or i32 %compare.local.base1150, 66, !dbg !21
  %compare.low1152 = icmp ult i32 %compare.local1151, %compare.low.base1078, !dbg !21
  %compare.result1153 = select i1 %compare.high.equal1076, i1 %compare.low1152, i1 %compare.high.order1077, !dbg !21
  %987 = select i1 %compare.result1153, float 0.000000e+00, float -inf, !dbg !22
  %988 = fadd float %976, %987, !dbg !13
  %compare.local.base1154 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 50) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1155 = or i32 %compare.local.base1154, 67, !dbg !21
  %compare.low1156 = icmp ult i32 %compare.local1155, %compare.low.base1078, !dbg !21
  %compare.result1157 = select i1 %compare.high.equal1076, i1 %compare.low1156, i1 %compare.high.order1077, !dbg !21
  %989 = select i1 %compare.result1157, float 0.000000e+00, float -inf, !dbg !22
  %990 = fadd float %982, %989, !dbg !13
  %991 = tail call float @__ocml_exp2_f32(float %984), !dbg !23
  %992 = insertelement <4 x float> poison, float %991, i64 0, !dbg !24
  %993 = fadd float %954, %991, !dbg !25
  %994 = tail call float @__ocml_exp2_f32(float %986), !dbg !23
  %995 = insertelement <4 x float> %992, float %994, i64 1, !dbg !24
  %996 = fadd float %993, %994, !dbg !25
  %997 = tail call float @__ocml_exp2_f32(float %988), !dbg !23
  %998 = insertelement <4 x float> %995, float %997, i64 2, !dbg !24
  %999 = fadd float %996, %997, !dbg !25
  %1000 = tail call float @__ocml_exp2_f32(float %990), !dbg !23
  %1001 = insertelement <4 x float> %998, float %1000, i64 3, !dbg !24
  %math.result409 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1001) #8, !dbg !24, !frisk.register_lifetimes !5
  %1002 = fptrunc <4 x float> %math.result409 to <4 x half>, !dbg !24
  %math.packed410 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1002) #8, !dbg !25, !frisk.register_lifetimes !5
  %1003 = fadd float %999, %1000, !dbg !25
  %1004 = extractelement <4 x float> %272, i64 0, !dbg !12
  %1005 = extractelement <4 x float> %272, i64 1, !dbg !12
  %1006 = extractelement <4 x float> %272, i64 2, !dbg !12
  %1007 = extractelement <4 x float> %272, i64 3, !dbg !12
  %1008 = select i1 %.not, float %1005, float %1004, !dbg !12
  %1009 = bitcast float %1008 to i32, !dbg !12
  %1010 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1009), !dbg !12
  %1011 = bitcast i32 %1010 to float, !dbg !12
  %1012 = select i1 %.not, float %1004, float %1011, !dbg !12
  %1013 = select i1 %.not, float %1011, float %1005, !dbg !12
  %1014 = select i1 %.not, float %1007, float %1006, !dbg !12
  %1015 = bitcast float %1014 to i32, !dbg !12
  %1016 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1015), !dbg !12
  %1017 = bitcast i32 %1016 to float, !dbg !12
  %1018 = select i1 %.not, float %1006, float %1017, !dbg !12
  %1019 = select i1 %.not, float %1017, float %1007, !dbg !12
  %1020 = select i1 %.not314, float %1018, float %1012, !dbg !12
  %1021 = bitcast float %1020 to i32, !dbg !12
  %1022 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1021), !dbg !12
  %1023 = bitcast i32 %1022 to float, !dbg !12
  %1024 = select i1 %.not314, float %1012, float %1023, !dbg !12
  %1025 = select i1 %.not314, float %1023, float %1018, !dbg !12
  %1026 = select i1 %.not314, float %1019, float %1013, !dbg !12
  %1027 = bitcast float %1026 to i32, !dbg !12
  %1028 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1027), !dbg !12
  %1029 = bitcast i32 %1028 to float, !dbg !12
  %1030 = select i1 %.not314, float %1013, float %1029, !dbg !12
  %1031 = select i1 %.not314, float %1029, float %1019, !dbg !12
  %compare.local.base1158 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 51) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1159 = or i32 %compare.local.base1158, 80, !dbg !21
  %compare.low1160 = icmp ult i32 %compare.local1159, %compare.low.base1078, !dbg !21
  %compare.result1161 = select i1 %compare.high.equal1076, i1 %compare.low1160, i1 %compare.high.order1077, !dbg !21
  %1032 = select i1 %compare.result1161, float 0.000000e+00, float -inf, !dbg !22
  %1033 = fadd float %1024, %1032, !dbg !13
  %compare.local.base1162 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 52) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1163 = or i32 %compare.local.base1162, 81, !dbg !21
  %compare.low1164 = icmp ult i32 %compare.local1163, %compare.low.base1078, !dbg !21
  %compare.result1165 = select i1 %compare.high.equal1076, i1 %compare.low1164, i1 %compare.high.order1077, !dbg !21
  %1034 = select i1 %compare.result1165, float 0.000000e+00, float -inf, !dbg !22
  %1035 = fadd float %1030, %1034, !dbg !13
  %compare.local.base1166 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 53) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1167 = or i32 %compare.local.base1166, 82, !dbg !21
  %compare.low1168 = icmp ult i32 %compare.local1167, %compare.low.base1078, !dbg !21
  %compare.result1169 = select i1 %compare.high.equal1076, i1 %compare.low1168, i1 %compare.high.order1077, !dbg !21
  %1036 = select i1 %compare.result1169, float 0.000000e+00, float -inf, !dbg !22
  %1037 = fadd float %1025, %1036, !dbg !13
  %compare.local.base1170 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 54) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1171 = or i32 %compare.local.base1170, 83, !dbg !21
  %compare.low1172 = icmp ult i32 %compare.local1171, %compare.low.base1078, !dbg !21
  %compare.result1173 = select i1 %compare.high.equal1076, i1 %compare.low1172, i1 %compare.high.order1077, !dbg !21
  %1038 = select i1 %compare.result1173, float 0.000000e+00, float -inf, !dbg !22
  %1039 = fadd float %1031, %1038, !dbg !13
  %1040 = tail call float @__ocml_exp2_f32(float %1033), !dbg !23
  %1041 = insertelement <4 x float> poison, float %1040, i64 0, !dbg !24
  %1042 = fadd float %1003, %1040, !dbg !25
  %1043 = tail call float @__ocml_exp2_f32(float %1035), !dbg !23
  %1044 = insertelement <4 x float> %1041, float %1043, i64 1, !dbg !24
  %1045 = fadd float %1042, %1043, !dbg !25
  %1046 = tail call float @__ocml_exp2_f32(float %1037), !dbg !23
  %1047 = insertelement <4 x float> %1044, float %1046, i64 2, !dbg !24
  %1048 = fadd float %1045, %1046, !dbg !25
  %1049 = tail call float @__ocml_exp2_f32(float %1039), !dbg !23
  %1050 = insertelement <4 x float> %1047, float %1049, i64 3, !dbg !24
  %math.result411 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1050) #8, !dbg !24, !frisk.register_lifetimes !5
  %1051 = fptrunc <4 x float> %math.result411 to <4 x half>, !dbg !24
  %math.packed412 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1051) #8, !dbg !25, !frisk.register_lifetimes !5
  %1052 = fadd float %1048, %1049, !dbg !25
  %1053 = extractelement <4 x float> %277, i64 0, !dbg !12
  %1054 = extractelement <4 x float> %277, i64 1, !dbg !12
  %1055 = extractelement <4 x float> %277, i64 2, !dbg !12
  %1056 = extractelement <4 x float> %277, i64 3, !dbg !12
  %1057 = select i1 %.not, float %1054, float %1053, !dbg !12
  %1058 = bitcast float %1057 to i32, !dbg !12
  %1059 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1058), !dbg !12
  %1060 = bitcast i32 %1059 to float, !dbg !12
  %1061 = select i1 %.not, float %1053, float %1060, !dbg !12
  %1062 = select i1 %.not, float %1060, float %1054, !dbg !12
  %1063 = select i1 %.not, float %1056, float %1055, !dbg !12
  %1064 = bitcast float %1063 to i32, !dbg !12
  %1065 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1064), !dbg !12
  %1066 = bitcast i32 %1065 to float, !dbg !12
  %1067 = select i1 %.not, float %1055, float %1066, !dbg !12
  %1068 = select i1 %.not, float %1066, float %1056, !dbg !12
  %1069 = select i1 %.not314, float %1067, float %1061, !dbg !12
  %1070 = bitcast float %1069 to i32, !dbg !12
  %1071 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1070), !dbg !12
  %1072 = bitcast i32 %1071 to float, !dbg !12
  %1073 = select i1 %.not314, float %1061, float %1072, !dbg !12
  %1074 = select i1 %.not314, float %1072, float %1067, !dbg !12
  %1075 = select i1 %.not314, float %1068, float %1062, !dbg !12
  %1076 = bitcast float %1075 to i32, !dbg !12
  %1077 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1076), !dbg !12
  %1078 = bitcast i32 %1077 to float, !dbg !12
  %1079 = select i1 %.not314, float %1062, float %1078, !dbg !12
  %1080 = select i1 %.not314, float %1078, float %1068, !dbg !12
  %compare.local.base1174 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 55) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1175 = or i32 %compare.local.base1174, 96, !dbg !21
  %compare.low1176 = icmp ult i32 %compare.local1175, %compare.low.base1078, !dbg !21
  %compare.result1177 = select i1 %compare.high.equal1076, i1 %compare.low1176, i1 %compare.high.order1077, !dbg !21
  %1081 = select i1 %compare.result1177, float 0.000000e+00, float -inf, !dbg !22
  %1082 = fadd float %1073, %1081, !dbg !13
  %compare.local.base1178 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 56) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1179 = or i32 %compare.local.base1178, 97, !dbg !21
  %compare.low1180 = icmp ult i32 %compare.local1179, %compare.low.base1078, !dbg !21
  %compare.result1181 = select i1 %compare.high.equal1076, i1 %compare.low1180, i1 %compare.high.order1077, !dbg !21
  %1083 = select i1 %compare.result1181, float 0.000000e+00, float -inf, !dbg !22
  %1084 = fadd float %1079, %1083, !dbg !13
  %compare.local.base1182 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 57) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1183 = or i32 %compare.local.base1182, 98, !dbg !21
  %compare.low1184 = icmp ult i32 %compare.local1183, %compare.low.base1078, !dbg !21
  %compare.result1185 = select i1 %compare.high.equal1076, i1 %compare.low1184, i1 %compare.high.order1077, !dbg !21
  %1085 = select i1 %compare.result1185, float 0.000000e+00, float -inf, !dbg !22
  %1086 = fadd float %1074, %1085, !dbg !13
  %compare.local.base1186 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 58) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1187 = or i32 %compare.local.base1186, 99, !dbg !21
  %compare.low1188 = icmp ult i32 %compare.local1187, %compare.low.base1078, !dbg !21
  %compare.result1189 = select i1 %compare.high.equal1076, i1 %compare.low1188, i1 %compare.high.order1077, !dbg !21
  %1087 = select i1 %compare.result1189, float 0.000000e+00, float -inf, !dbg !22
  %1088 = fadd float %1080, %1087, !dbg !13
  %1089 = tail call float @__ocml_exp2_f32(float %1082), !dbg !23
  %1090 = insertelement <4 x float> poison, float %1089, i64 0, !dbg !24
  %1091 = fadd float %1052, %1089, !dbg !25
  %1092 = tail call float @__ocml_exp2_f32(float %1084), !dbg !23
  %1093 = insertelement <4 x float> %1090, float %1092, i64 1, !dbg !24
  %1094 = fadd float %1091, %1092, !dbg !25
  %1095 = tail call float @__ocml_exp2_f32(float %1086), !dbg !23
  %1096 = insertelement <4 x float> %1093, float %1095, i64 2, !dbg !24
  %1097 = fadd float %1094, %1095, !dbg !25
  %1098 = tail call float @__ocml_exp2_f32(float %1088), !dbg !23
  %1099 = insertelement <4 x float> %1096, float %1098, i64 3, !dbg !24
  %math.result413 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1099) #8, !dbg !24, !frisk.register_lifetimes !5
  %1100 = fptrunc <4 x float> %math.result413 to <4 x half>, !dbg !24
  %math.packed414 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1100) #8, !dbg !25, !frisk.register_lifetimes !5
  %1101 = fadd float %1097, %1098, !dbg !25
  %1102 = extractelement <4 x float> %282, i64 0, !dbg !12
  %1103 = extractelement <4 x float> %282, i64 1, !dbg !12
  %1104 = extractelement <4 x float> %282, i64 2, !dbg !12
  %1105 = extractelement <4 x float> %282, i64 3, !dbg !12
  %1106 = select i1 %.not, float %1103, float %1102, !dbg !12
  %1107 = bitcast float %1106 to i32, !dbg !12
  %1108 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1107), !dbg !12
  %1109 = bitcast i32 %1108 to float, !dbg !12
  %1110 = select i1 %.not, float %1102, float %1109, !dbg !12
  %1111 = select i1 %.not, float %1109, float %1103, !dbg !12
  %1112 = select i1 %.not, float %1105, float %1104, !dbg !12
  %1113 = bitcast float %1112 to i32, !dbg !12
  %1114 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1113), !dbg !12
  %1115 = bitcast i32 %1114 to float, !dbg !12
  %1116 = select i1 %.not, float %1104, float %1115, !dbg !12
  %1117 = select i1 %.not, float %1115, float %1105, !dbg !12
  %1118 = select i1 %.not314, float %1116, float %1110, !dbg !12
  %1119 = bitcast float %1118 to i32, !dbg !12
  %1120 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1119), !dbg !12
  %1121 = bitcast i32 %1120 to float, !dbg !12
  %1122 = select i1 %.not314, float %1110, float %1121, !dbg !12
  %1123 = select i1 %.not314, float %1121, float %1116, !dbg !12
  %1124 = select i1 %.not314, float %1117, float %1111, !dbg !12
  %1125 = bitcast float %1124 to i32, !dbg !12
  %1126 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1125), !dbg !12
  %1127 = bitcast i32 %1126 to float, !dbg !12
  %1128 = select i1 %.not314, float %1111, float %1127, !dbg !12
  %1129 = select i1 %.not314, float %1127, float %1117, !dbg !12
  %compare.local.base1190 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 59) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1191 = or i32 %compare.local.base1190, 112, !dbg !21
  %compare.low1192 = icmp ult i32 %compare.local1191, %compare.low.base1078, !dbg !21
  %compare.result1193 = select i1 %compare.high.equal1076, i1 %compare.low1192, i1 %compare.high.order1077, !dbg !21
  %1130 = select i1 %compare.result1193, float 0.000000e+00, float -inf, !dbg !22
  %1131 = fadd float %1122, %1130, !dbg !13
  %compare.local.base1194 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 60) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1195 = or i32 %compare.local.base1194, 113, !dbg !21
  %compare.low1196 = icmp ult i32 %compare.local1195, %compare.low.base1078, !dbg !21
  %compare.result1197 = select i1 %compare.high.equal1076, i1 %compare.low1196, i1 %compare.high.order1077, !dbg !21
  %1132 = select i1 %compare.result1197, float 0.000000e+00, float -inf, !dbg !22
  %1133 = fadd float %1128, %1132, !dbg !13
  %compare.local.base1198 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 61) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1199 = or i32 %compare.local.base1198, 114, !dbg !21
  %compare.low1200 = icmp ult i32 %compare.local1199, %compare.low.base1078, !dbg !21
  %compare.result1201 = select i1 %compare.high.equal1076, i1 %compare.low1200, i1 %compare.high.order1077, !dbg !21
  %1134 = select i1 %compare.result1201, float 0.000000e+00, float -inf, !dbg !22
  %1135 = fadd float %1123, %1134, !dbg !13
  %compare.local.base1202 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base951, i32 62) #8, !dbg !21, !frisk.register_lifetimes !5
  %compare.local1203 = or i32 %compare.local.base1202, 115, !dbg !21
  %compare.low1204 = icmp ult i32 %compare.local1203, %compare.low.base1078, !dbg !21
  %compare.result1205 = select i1 %compare.high.equal1076, i1 %compare.low1204, i1 %compare.high.order1077, !dbg !21
  %1136 = select i1 %compare.result1205, float 0.000000e+00, float -inf, !dbg !22
  %1137 = fadd float %1129, %1136, !dbg !13
  %1138 = tail call float @__ocml_exp2_f32(float %1131), !dbg !23
  %1139 = insertelement <4 x float> poison, float %1138, i64 0, !dbg !24
  %1140 = fadd float %1101, %1138, !dbg !25
  %1141 = tail call float @__ocml_exp2_f32(float %1133), !dbg !23
  %1142 = insertelement <4 x float> %1139, float %1141, i64 1, !dbg !24
  %1143 = fadd float %1140, %1141, !dbg !25
  %1144 = tail call float @__ocml_exp2_f32(float %1135), !dbg !23
  %1145 = insertelement <4 x float> %1142, float %1144, i64 2, !dbg !24
  %1146 = fadd float %1143, %1144, !dbg !25
  %1147 = tail call float @__ocml_exp2_f32(float %1137), !dbg !23
  %1148 = insertelement <4 x float> %1145, float %1147, i64 3, !dbg !24
  %math.result415 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1148) #8, !dbg !24, !frisk.register_lifetimes !5
  %1149 = fptrunc <4 x float> %math.result415 to <4 x half>, !dbg !24
  %math.packed416 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1149) #8, !dbg !25, !frisk.register_lifetimes !5
  %1150 = fadd float %1146, %1147, !dbg !25
  %1151 = bitcast float %758 to i32, !dbg !25
  %1152 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1151), !dbg !25
  %1153 = bitcast i32 %1152 to float, !dbg !25
  %1154 = fadd float %758, %1153, !dbg !25
  %1155 = bitcast float %1154 to i32, !dbg !25
  %1156 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1155), !dbg !25
  %1157 = bitcast i32 %1156 to float, !dbg !25
  %1158 = fadd float %1154, %1157, !dbg !25
  %1159 = extractvalue [2 x <1 x float>] %47, 0, !dbg !25
  %1160 = extractelement <1 x float> %1159, i64 0, !dbg !25
  %1161 = fadd float %1160, %1158, !dbg !25
  %1162 = insertelement <1 x float> poison, float %1161, i64 0, !dbg !16
  %1163 = insertvalue [2 x <1 x float>] poison, <1 x float> %1162, 0, !dbg !16
  %1164 = bitcast float %1150 to i32, !dbg !25
  %1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %365, i32 %1164), !dbg !25
  %1166 = bitcast i32 %1165 to float, !dbg !25
  %1167 = fadd float %1150, %1166, !dbg !25
  %1168 = bitcast float %1167 to i32, !dbg !25
  %1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %381, i32 %1168), !dbg !25
  %1170 = bitcast i32 %1169 to float, !dbg !25
  %1171 = fadd float %1167, %1170, !dbg !25
  %1172 = extractvalue [2 x <1 x float>] %47, 1, !dbg !25
  %1173 = extractelement <1 x float> %1172, i64 0, !dbg !25
  %1174 = fadd float %1173, %1171, !dbg !25
  %1175 = insertelement <1 x float> poison, float %1174, i64 0, !dbg !16
  %1176 = insertvalue [2 x <1 x float>] %1163, <1 x float> %1175, 1, !dbg !16
  tail call void @llvm.amdgcn.wait.asyncmark(i16 0), !dbg !20
  fence syncscope("workgroup") release, !dbg !20
  tail call void @llvm.amdgcn.s.barrier(), !dbg !20
  fence syncscope("workgroup") acquire, !dbg !20
  %address.local854 = and i32 %address.local750, 504, !dbg !26
  %address.local855 = zext nneg i32 %address.local854 to i64, !dbg !26
  %address.local856 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local855, !dbg !26
  %1177 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local856, i32 0) #10, !dbg !26
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !26
  %address.local861 = getelementptr i8, ptr addrspace(3) %address.local856, i64 1024, !dbg !26
  %1179 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local861, i32 0) #10, !dbg !26
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !26
  %address.local866 = getelementptr i8, ptr addrspace(3) %address.local856, i64 2048, !dbg !26
  %1181 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local866, i32 0) #10, !dbg !26
  %1182 = bitcast <4 x i32> %1181 to <8 x half>, !dbg !26
  %address.local868 = or i32 %address.local750, 1536, !dbg !26
  %address.local869 = zext nneg i32 %address.local868 to i64, !dbg !26
  %address.local870 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local869, !dbg !26
  %1183 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local870, i32 0) #10, !dbg !26
  %1184 = bitcast <4 x i32> %1183 to <8 x half>, !dbg !26
  %address.local875 = getelementptr i8, ptr addrspace(3) %address.local856, i64 4096, !dbg !26
  %1185 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #10, !dbg !26
  %1186 = bitcast <4 x i32> %1185 to <8 x half>, !dbg !26
  %address.local880 = getelementptr i8, ptr addrspace(3) %address.local856, i64 5120, !dbg !26
  %1187 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local880, i32 0) #10, !dbg !26
  %1188 = bitcast <4 x i32> %1187 to <8 x half>, !dbg !26
  %address.local885 = getelementptr i8, ptr addrspace(3) %address.local856, i64 6144, !dbg !26
  %1189 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local885, i32 0) #10, !dbg !26
  %1190 = bitcast <4 x i32> %1189 to <8 x half>, !dbg !26
  %address.local887 = or i32 %address.local750, 3584, !dbg !26
  %address.local888 = zext nneg i32 %address.local887 to i64, !dbg !26
  %address.local889 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local888, !dbg !26
  %1191 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local889, i32 0) #10, !dbg !26
  %1192 = bitcast <4 x i32> %1191 to <8 x half>, !dbg !26
  %1193 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1193, <4 x float> zeroinitializer) #8, !dbg !26
  %1195 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1195, <4 x float> %1194) #8, !dbg !26
  %1197 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1197, <4 x float> %1196) #8, !dbg !26
  %1199 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %1199, <4 x float> %1198) #8, !dbg !26
  %1201 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %1201, <4 x float> %1200) #8, !dbg !26
  %1203 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %1203, <4 x float> %1202) #8, !dbg !26
  %1205 = shufflevector <8 x half> %1190, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1205, <4 x float> %1204) #8, !dbg !26
  %1207 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1207, <4 x float> %1206) #8, !dbg !26
  %1209 = extractvalue [2 x <8 x float>] %48, 0, !dbg !27
  %1210 = shufflevector <8 x float> %1209, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1211 = fadd <4 x float> %1210, %1208, !dbg !27
  %1212 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1212, <4 x float> zeroinitializer) #8, !dbg !26
  %1214 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1214, <4 x float> %1213) #8, !dbg !26
  %1216 = shufflevector <8 x half> %1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1216, <4 x float> %1215) #8, !dbg !26
  %1218 = shufflevector <8 x half> %1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %1218, <4 x float> %1217) #8, !dbg !26
  %1220 = shufflevector <8 x half> %1186, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %1220, <4 x float> %1219) #8, !dbg !26
  %1222 = shufflevector <8 x half> %1188, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %1222, <4 x float> %1221) #8, !dbg !26
  %1224 = shufflevector <8 x half> %1190, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1224, <4 x float> %1223) #8, !dbg !26
  %1226 = shufflevector <8 x half> %1192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1226, <4 x float> %1225) #8, !dbg !26
  %1228 = shufflevector <8 x float> %1209, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1229 = fadd <4 x float> %1228, %1227, !dbg !27
  %1230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1193, <4 x float> zeroinitializer) #8, !dbg !26
  %1231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1195, <4 x float> %1230) #8, !dbg !26
  %1232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1197, <4 x float> %1231) #8, !dbg !26
  %1233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %1199, <4 x float> %1232) #8, !dbg !26
  %1234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %1201, <4 x float> %1233) #8, !dbg !26
  %1235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %1203, <4 x float> %1234) #8, !dbg !26
  %1236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %1205, <4 x float> %1235) #8, !dbg !26
  %1237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %1207, <4 x float> %1236) #8, !dbg !26
  %1238 = extractvalue [2 x <8 x float>] %48, 1, !dbg !27
  %1239 = shufflevector <8 x float> %1238, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1240 = fadd <4 x float> %1239, %1237, !dbg !27
  %1241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1212, <4 x float> zeroinitializer) #8, !dbg !26
  %1242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1214, <4 x float> %1241) #8, !dbg !26
  %1243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1216, <4 x float> %1242) #8, !dbg !26
  %1244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %1218, <4 x float> %1243) #8, !dbg !26
  %1245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %1220, <4 x float> %1244) #8, !dbg !26
  %1246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %1222, <4 x float> %1245) #8, !dbg !26
  %1247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %1224, <4 x float> %1246) #8, !dbg !26
  %1248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %1226, <4 x float> %1247) #8, !dbg !26
  %1249 = shufflevector <8 x float> %1238, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1250 = fadd <4 x float> %1249, %1248, !dbg !27
  %1251 = shufflevector <4 x float> %1211, <4 x float> %1229, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1252 = insertvalue [2 x <8 x float>] poison, <8 x float> %1251, 0, !dbg !16
  %1253 = shufflevector <4 x float> %1240, <4 x float> %1250, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %1254 = insertvalue [2 x <8 x float>] %1252, <8 x float> %1253, 1, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1255 = add nuw nsw i64 %49, 1, !dbg !16
  %.not315.not = icmp slt i64 %49, %6, !dbg !16
  br i1 %.not315.not, label %.lr.ph, label %._crit_edge, !dbg !16

._crit_edge:                                      ; preds = %302, %.._crit_edge_crit_edge
  %.pre-phi35 = phi i32 [ %.pre34, %.._crit_edge_crit_edge ], [ %378, %302 ], !dbg !16
  %.pre-phi33 = phi i32 [ %.pre32, %.._crit_edge_crit_edge ], [ %362, %302 ], !dbg !16
  %.pre-phi31 = phi i32 [ %.pre30, %.._crit_edge_crit_edge ], [ %361, %302 ], !dbg !16
  %.pre-phi27 = phi i32 [ %.pre26, %.._crit_edge_crit_edge ], [ %359, %302 ], !dbg !16
  %.lcssa6 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1254, %302 ], !dbg !16
  %.lcssa5 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1176, %302 ], !dbg !16
  %1256 = extractvalue [2 x <8 x float>] %.lcssa6, 0, !dbg !16
  %1257 = extractelement <8 x float> %1256, i64 0, !dbg !16
  %1258 = extractelement <8 x float> %1256, i64 1, !dbg !16
  %1259 = extractelement <8 x float> %1256, i64 2, !dbg !16
  %1260 = extractelement <8 x float> %1256, i64 3, !dbg !16
  %1261 = select i1 %.not, float %1258, float %1257, !dbg !16
  %1262 = icmp samesign ult i32 %.pre-phi31, %.pre-phi33, !dbg !16
  %1263 = select i1 %1262, i32 %.pre-phi31, i32 %.pre-phi27, !dbg !16
  %1264 = shl nuw nsw i32 %1263, 2, !dbg !16
  %1265 = bitcast float %1261 to i32, !dbg !16
  %1266 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1265), !dbg !16
  %1267 = bitcast i32 %1266 to float, !dbg !16
  %1268 = select i1 %.not, float %1257, float %1267, !dbg !16
  %1269 = select i1 %.not, float %1267, float %1258, !dbg !16
  %1270 = select i1 %.not, float %1260, float %1259, !dbg !16
  %1271 = bitcast float %1270 to i32, !dbg !16
  %1272 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1271), !dbg !16
  %1273 = bitcast i32 %1272 to float, !dbg !16
  %1274 = select i1 %.not, float %1259, float %1273, !dbg !16
  %1275 = select i1 %.not, float %1273, float %1260, !dbg !16
  %1276 = select i1 %.not314, float %1274, float %1268, !dbg !16
  %1277 = icmp samesign ult i32 %.pre-phi35, %.pre-phi33, !dbg !16
  %1278 = select i1 %1277, i32 %.pre-phi35, i32 %.pre-phi27, !dbg !16
  %1279 = shl nuw nsw i32 %1278, 2, !dbg !16
  %1280 = bitcast float %1276 to i32, !dbg !16
  %1281 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1280), !dbg !16
  %1282 = select i1 %.not314, float %1275, float %1269, !dbg !16
  %1283 = bitcast float %1282 to i32, !dbg !16
  %1284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1283), !dbg !16
  %1285 = extractelement <8 x float> %1256, i64 4, !dbg !16
  %1286 = extractelement <8 x float> %1256, i64 5, !dbg !16
  %1287 = extractelement <8 x float> %1256, i64 6, !dbg !16
  %1288 = extractelement <8 x float> %1256, i64 7, !dbg !16
  %1289 = select i1 %.not, float %1286, float %1285, !dbg !16
  %1290 = bitcast float %1289 to i32, !dbg !16
  %1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1290), !dbg !16
  %1292 = bitcast i32 %1291 to float, !dbg !16
  %1293 = select i1 %.not, float %1285, float %1292, !dbg !16
  %1294 = select i1 %.not, float %1292, float %1286, !dbg !16
  %1295 = select i1 %.not, float %1288, float %1287, !dbg !16
  %1296 = bitcast float %1295 to i32, !dbg !16
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1296), !dbg !16
  %1298 = bitcast i32 %1297 to float, !dbg !16
  %1299 = select i1 %.not, float %1287, float %1298, !dbg !16
  %1300 = select i1 %.not, float %1298, float %1288, !dbg !16
  %1301 = select i1 %.not314, float %1299, float %1293, !dbg !16
  %1302 = bitcast float %1301 to i32, !dbg !16
  %1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1302), !dbg !16
  %1304 = select i1 %.not314, float %1300, float %1294, !dbg !16
  %1305 = bitcast float %1304 to i32, !dbg !16
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1305), !dbg !16
  %1307 = extractvalue [2 x <8 x float>] %.lcssa6, 1, !dbg !16
  %1308 = extractelement <8 x float> %1307, i64 0, !dbg !16
  %1309 = extractelement <8 x float> %1307, i64 1, !dbg !16
  %1310 = extractelement <8 x float> %1307, i64 2, !dbg !16
  %1311 = extractelement <8 x float> %1307, i64 3, !dbg !16
  %1312 = select i1 %.not, float %1309, float %1308, !dbg !16
  %1313 = bitcast float %1312 to i32, !dbg !16
  %1314 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1313), !dbg !16
  %1315 = bitcast i32 %1314 to float, !dbg !16
  %1316 = select i1 %.not, float %1308, float %1315, !dbg !16
  %1317 = select i1 %.not, float %1315, float %1309, !dbg !16
  %1318 = select i1 %.not, float %1311, float %1310, !dbg !16
  %1319 = bitcast float %1318 to i32, !dbg !16
  %1320 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1319), !dbg !16
  %1321 = bitcast i32 %1320 to float, !dbg !16
  %1322 = select i1 %.not, float %1310, float %1321, !dbg !16
  %1323 = select i1 %.not, float %1321, float %1311, !dbg !16
  %1324 = select i1 %.not314, float %1322, float %1316, !dbg !16
  %1325 = bitcast float %1324 to i32, !dbg !16
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1325), !dbg !16
  %1327 = select i1 %.not314, float %1323, float %1317, !dbg !16
  %1328 = bitcast float %1327 to i32, !dbg !16
  %1329 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1328), !dbg !16
  %1330 = extractelement <8 x float> %1307, i64 4, !dbg !16
  %1331 = extractelement <8 x float> %1307, i64 5, !dbg !16
  %1332 = extractelement <8 x float> %1307, i64 6, !dbg !16
  %1333 = extractelement <8 x float> %1307, i64 7, !dbg !16
  %1334 = select i1 %.not, float %1331, float %1330, !dbg !16
  %1335 = bitcast float %1334 to i32, !dbg !16
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1335), !dbg !16
  %1337 = bitcast i32 %1336 to float, !dbg !16
  %1338 = select i1 %.not, float %1330, float %1337, !dbg !16
  %1339 = select i1 %.not, float %1337, float %1331, !dbg !16
  %1340 = select i1 %.not, float %1333, float %1332, !dbg !16
  %1341 = bitcast float %1340 to i32, !dbg !16
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1264, i32 %1341), !dbg !16
  %1343 = bitcast i32 %1342 to float, !dbg !16
  %1344 = select i1 %.not, float %1332, float %1343, !dbg !16
  %1345 = select i1 %.not, float %1343, float %1333, !dbg !16
  %1346 = select i1 %.not314, float %1344, float %1338, !dbg !16
  %1347 = bitcast float %1346 to i32, !dbg !16
  %1348 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1347), !dbg !16
  %1349 = select i1 %.not314, float %1345, float %1339, !dbg !16
  %1350 = bitcast float %1349 to i32, !dbg !16
  %1351 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1279, i32 %1350), !dbg !16
  %1352 = bitcast i32 %1348 to float, !dbg !16
  %1353 = bitcast i32 %1351 to float, !dbg !16
  %1354 = insertelement <4 x float> poison, float %1338, i64 0, !dbg !16
  %1355 = insertelement <4 x float> %1354, float %1339, i64 1, !dbg !16
  %1356 = insertelement <4 x float> %1355, float %1352, i64 2, !dbg !16
  %1357 = insertelement <4 x float> %1356, float %1353, i64 3, !dbg !16
  %1358 = insertelement <4 x float> poison, float %1352, i64 0, !dbg !16
  %1359 = insertelement <4 x float> %1358, float %1353, i64 1, !dbg !16
  %1360 = insertelement <4 x float> %1359, float %1344, i64 2, !dbg !16
  %1361 = insertelement <4 x float> %1360, float %1345, i64 3, !dbg !16
  %1362 = select <4 x i1> %43, <4 x float> %1357, <4 x float> %1361, !dbg !16
  %1363 = extractvalue [2 x <1 x float>] %.lcssa5, 1, !dbg !28
  %1364 = shufflevector <1 x float> %1363, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1365 = fdiv <4 x float> %1362, %1364, !dbg !28
  %1366 = fptrunc <4 x float> %1365 to <4 x half>, !dbg !29
  %1367 = extractelement <4 x half> %1366, i64 3, !dbg !29
  %1368 = extractelement <4 x half> %1366, i64 2, !dbg !29
  %1369 = extractelement <4 x half> %1366, i64 1, !dbg !29
  %1370 = extractelement <4 x half> %1366, i64 0, !dbg !29
  %1371 = bitcast i32 %1326 to float, !dbg !16
  %1372 = bitcast i32 %1329 to float, !dbg !16
  %1373 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !16
  %1374 = insertelement <4 x float> %1373, float %1317, i64 1, !dbg !16
  %1375 = insertelement <4 x float> %1374, float %1371, i64 2, !dbg !16
  %1376 = insertelement <4 x float> %1375, float %1372, i64 3, !dbg !16
  %1377 = insertelement <4 x float> poison, float %1371, i64 0, !dbg !16
  %1378 = insertelement <4 x float> %1377, float %1372, i64 1, !dbg !16
  %1379 = insertelement <4 x float> %1378, float %1322, i64 2, !dbg !16
  %1380 = insertelement <4 x float> %1379, float %1323, i64 3, !dbg !16
  %1381 = select <4 x i1> %43, <4 x float> %1376, <4 x float> %1380, !dbg !16
  %1382 = fdiv <4 x float> %1381, %1364, !dbg !28
  %1383 = fptrunc <4 x float> %1382 to <4 x half>, !dbg !29
  %1384 = extractelement <4 x half> %1383, i64 3, !dbg !29
  %1385 = extractelement <4 x half> %1383, i64 2, !dbg !29
  %1386 = extractelement <4 x half> %1383, i64 1, !dbg !29
  %1387 = extractelement <4 x half> %1383, i64 0, !dbg !29
  %1388 = bitcast i32 %1303 to float, !dbg !16
  %1389 = bitcast i32 %1306 to float, !dbg !16
  %1390 = insertelement <4 x float> poison, float %1293, i64 0, !dbg !16
  %1391 = insertelement <4 x float> %1390, float %1294, i64 1, !dbg !16
  %1392 = insertelement <4 x float> %1391, float %1388, i64 2, !dbg !16
  %1393 = insertelement <4 x float> %1392, float %1389, i64 3, !dbg !16
  %1394 = insertelement <4 x float> poison, float %1388, i64 0, !dbg !16
  %1395 = insertelement <4 x float> %1394, float %1389, i64 1, !dbg !16
  %1396 = insertelement <4 x float> %1395, float %1299, i64 2, !dbg !16
  %1397 = insertelement <4 x float> %1396, float %1300, i64 3, !dbg !16
  %1398 = select <4 x i1> %43, <4 x float> %1393, <4 x float> %1397, !dbg !16
  %1399 = extractvalue [2 x <1 x float>] %.lcssa5, 0, !dbg !28
  %1400 = shufflevector <1 x float> %1399, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1401 = fdiv <4 x float> %1398, %1400, !dbg !28
  %1402 = fptrunc <4 x float> %1401 to <4 x half>, !dbg !29
  %1403 = extractelement <4 x half> %1402, i64 3, !dbg !29
  %1404 = extractelement <4 x half> %1402, i64 2, !dbg !29
  %1405 = extractelement <4 x half> %1402, i64 1, !dbg !29
  %1406 = extractelement <4 x half> %1402, i64 0, !dbg !29
  %1407 = bitcast i32 %1281 to float, !dbg !16
  %1408 = bitcast i32 %1284 to float, !dbg !16
  %1409 = insertelement <4 x float> poison, float %1268, i64 0, !dbg !16
  %1410 = insertelement <4 x float> %1409, float %1269, i64 1, !dbg !16
  %1411 = insertelement <4 x float> %1410, float %1407, i64 2, !dbg !16
  %1412 = insertelement <4 x float> %1411, float %1408, i64 3, !dbg !16
  %1413 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !16
  %1414 = insertelement <4 x float> %1413, float %1408, i64 1, !dbg !16
  %1415 = insertelement <4 x float> %1414, float %1274, i64 2, !dbg !16
  %1416 = insertelement <4 x float> %1415, float %1275, i64 3, !dbg !16
  %1417 = select <4 x i1> %43, <4 x float> %1412, <4 x float> %1416, !dbg !16
  %1418 = fdiv <4 x float> %1417, %1400, !dbg !28
  %1419 = fptrunc <4 x float> %1418 to <4 x half>, !dbg !29
  %1420 = extractelement <4 x half> %1419, i64 3, !dbg !29
  %1421 = extractelement <4 x half> %1419, i64 2, !dbg !29
  %1422 = extractelement <4 x half> %1419, i64 1, !dbg !29
  %1423 = extractelement <4 x half> %1419, i64 0, !dbg !29
  %address.thread890 = tail call i32 asm sideeffect "", "=v,0"(i32 %17) #8, !dbg !30, !range !11, !frisk.register_lifetimes !5
  %address.local891 = zext nneg i32 %address.thread890 to i64, !dbg !10
  %address.local894 = lshr i32 %address.thread890, 2, !dbg !13
  %address.local895 = and i32 %address.local894, 48, !dbg !13
  %address.local896 = zext nneg i32 %address.local895 to i64, !dbg !13
  %address.local898 = and i32 %address.thread890, 240, !dbg !13
  %address.local899 = sub nsw i32 0, %address.local898, !dbg !13
  %address.local900 = sext i32 %address.local899 to i64, !dbg !13
  %address.local893 = add nsw i64 %16, %address.local891, !dbg !30
  %address.local897 = add nsw i64 %address.local893, %address.local900, !dbg !30
  %address.local901 = add nsw i64 %address.local897, %address.local896, !dbg !30
  %address.local904.idx = shl nuw nsw i64 %address.local901, 8, !dbg !30
  %address.local904 = getelementptr inbounds nuw i8, ptr addrspace(1) %36, i64 %address.local904.idx, !dbg !30
  %address.local906 = and i32 %address.local894, 12, !dbg !13
  %address.local907 = zext nneg i32 %address.local906 to i64, !dbg !13
  %address.local908 = or disjoint i64 %46, %address.local907, !dbg !30
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local908, !dbg !30
  store half %1423, ptr addrspace(1) %address.local909, align 2, !dbg !30
  %address.local911 = or disjoint i64 %address.local908, 1, !dbg !30
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local911, !dbg !30
  store half %1422, ptr addrspace(1) %address.local912, align 2, !dbg !30
  %address.local914 = or disjoint i64 %address.local908, 2, !dbg !30
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local914, !dbg !30
  store half %1421, ptr addrspace(1) %address.local915, align 2, !dbg !30
  %address.local917 = or disjoint i64 %address.local908, 3, !dbg !30
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local917, !dbg !30
  store half %1420, ptr addrspace(1) %address.local918, align 2, !dbg !30
  %address.local920 = or disjoint i64 %address.local908, 16, !dbg !30
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local920, !dbg !30
  store half %1406, ptr addrspace(1) %address.local921, align 2, !dbg !30
  %address.local923 = or disjoint i64 %address.local908, 17, !dbg !30
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local923, !dbg !30
  store half %1405, ptr addrspace(1) %address.local924, align 2, !dbg !30
  %address.local926 = or disjoint i64 %address.local908, 18, !dbg !30
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local926, !dbg !30
  store half %1404, ptr addrspace(1) %address.local927, align 2, !dbg !30
  %address.local929 = or disjoint i64 %address.local908, 19, !dbg !30
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local904, i64 %address.local929, !dbg !30
  store half %1403, ptr addrspace(1) %address.local930, align 2, !dbg !30
  %address.local934 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local904, i64 16384, !dbg !30
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local908, !dbg !30
  store half %1387, ptr addrspace(1) %address.local935, align 2, !dbg !30
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local911, !dbg !30
  store half %1386, ptr addrspace(1) %address.local937, align 2, !dbg !30
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local914, !dbg !30
  store half %1385, ptr addrspace(1) %address.local939, align 2, !dbg !30
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local917, !dbg !30
  store half %1384, ptr addrspace(1) %address.local941, align 2, !dbg !30
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local920, !dbg !30
  store half %1370, ptr addrspace(1) %address.local943, align 2, !dbg !30
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local923, !dbg !30
  store half %1369, ptr addrspace(1) %address.local945, align 2, !dbg !30
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local926, !dbg !30
  store half %1368, ptr addrspace(1) %address.local947, align 2, !dbg !30
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local934, i64 %address.local929, !dbg !30
  store half %1367, ptr addrspace(1) %address.local949, align 2, !dbg !30
  fence syncscope("workgroup") release, !dbg !15
  tail call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1424 = add nuw nsw i64 %45, 1, !dbg !15
  %1425 = icmp samesign ult i64 %45, 3, !dbg !15
  br i1 %1425, label %44, label %1426, !dbg !15

1426:                                             ; preds = %._crit_edge
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
!1 = !DIFile(filename: "test_input_spidD.mlir", directory: "3rd/deepgengraph/test")
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
