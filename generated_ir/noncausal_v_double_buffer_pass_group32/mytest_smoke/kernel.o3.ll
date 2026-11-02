; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_v_double_buffer_pass_group32/mytest_smoke/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test/test_input_spidD_noncasual.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = shl nsw i64 %8, 5, !dbg !7
  %10 = add nsw i64 %9, %6, !dbg !7
  %11 = lshr i64 %10, 5, !dbg !7
  %12 = and i64 %11, 576460752303423456, !dbg !7
  %13 = and i64 %6, 31, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = lshr i64 %10, 10, !dbg !8
  %16 = add nuw nsw i64 %11, %15, !dbg !8
  %17 = shl nuw nsw i64 %14, 7, !dbg !8
  %18 = shl i64 %15, 12, !dbg !8
  %19 = sub i64 %17, %18, !dbg !8
  %20 = shl i64 %16, 19, !dbg !8
  %21 = and i64 %20, 16252928, !dbg !8
  %22 = shl nsw i64 %19, 7, !dbg !8
  %23 = add i64 %22, %21, !dbg !8
  %24 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %24) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %25 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %23, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %26 = insertelement <2 x i64> poison, i64 %25, i64 0, !dbg !8
  %27 = bitcast <2 x i64> %26 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %27, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %23, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %28 = or disjoint i64 %23, 32, !dbg !8
  %29 = or disjoint i64 %address.local413, %28, !dbg !8
  %.tr = trunc i64 %29 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %30 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %31 = insertelement <4 x ptr addrspace(3)> %30, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %32 = insertelement <4 x ptr addrspace(3)> %31, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %33 = insertelement <4 x ptr addrspace(3)> %32, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %34 = ptrtoint <4 x ptr addrspace(3)> %33 to <4 x i64>, !dbg !8
  %35 = trunc <4 x i64> %34 to <4 x i32>, !dbg !8
  %36 = extractelement <4 x i32> %35, i64 0, !dbg !8
  %37 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %36), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %38 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %39 = extractelement <4 x i32> %35, i64 1, !dbg !8
  %40 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %39), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40), !dbg !8
  %41 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %42 = extractelement <4 x i32> %35, i64 2, !dbg !8
  %43 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %42), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %44 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %45 = extractelement <4 x i32> %35, i64 3, !dbg !8
  %46 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %45), !dbg !8
  %47 = or disjoint i64 %address.local425, %28, !dbg !8
  %.tr1401 = trunc i64 %47 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %address.local435 = lshr i32 %address.thread, 2, !dbg !10
  %address.local436 = and i32 %address.local435, 4, !dbg !10
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local438 = and i32 %address.local437, 384, !dbg !10
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !10
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local441 = and i32 %address.local440, 120, !dbg !10
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !10
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local444 = and i32 %address.local443, 1024, !dbg !10
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !10
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !10
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !10
  %49 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %50 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %51 = fmul <4 x half> %49, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %52 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %53 = fmul <4 x half> %50, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %55 = fmul <4 x half> %52, splat (half 1.275630e-01), !dbg !10
  %56 = fmul <4 x half> %54, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %57 = or disjoint i64 %23, 64, !dbg !8
  %58 = or disjoint i64 %address.local413, %57, !dbg !8
  %.tr1402 = trunc i64 %58 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %59 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %60 = or disjoint i64 %address.local425, %57, !dbg !8
  %.tr1403 = trunc i64 %60 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %61 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %62 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %65 = fmul <4 x half> %62, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %66 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %67 = fmul <4 x half> %63, splat (half 1.275630e-01), !dbg !10
  %68 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %70 = or disjoint i64 %23, 96, !dbg !8
  %71 = or disjoint i64 %address.local413, %70, !dbg !8
  %.tr1404 = trunc i64 %71 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %72 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %73 = or disjoint i64 %address.local425, %70, !dbg !8
  %.tr1405 = trunc i64 %73 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %74 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %75 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %78 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !10
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %81 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !10
  %82 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %86 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !10
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %88 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %90 = fmul <4 x half> %87, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %91 = shl nuw nsw i32 %24, 2, !dbg !12
  %92 = and i32 %91, 12, !dbg !12
  %93 = and i32 %24, 1, !dbg !12
  %94 = add nuw nsw i32 %93, 1, !dbg !12
  %narrow = or disjoint i32 %94, %92, !dbg !12
  %95 = lshr i32 %narrow, 1, !dbg !12
  %96 = zext nneg i32 %95 to i64, !dbg !12
  %97 = shl i64 %11, 19
  %98 = and i64 %97, 16252928
  %99 = ptrtoint ptr addrspace(1) %1 to i64
  %100 = insertelement <2 x i64> poison, i64 %99, i64 0
  %101 = bitcast <2 x i64> %100 to <4 x i32>
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %101, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %102 = ptrtoint ptr addrspace(1) %2 to i64
  %103 = insertelement <2 x i64> poison, i64 %102, i64 0
  %104 = bitcast <2 x i64> %103 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %104, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %105, !dbg !11

105:                                              ; preds = %4, %105
  %106 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1013, %105 ]
  %107 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1009, %105 ]
  %108 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1005, %105 ]
  %109 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1001, %105 ]
  %110 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %693, %105 ]
  %111 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %999, %105 ]
  %112 = phi i64 [ 0, %4 ], [ %1014, %105 ]
  %113 = shl nuw nsw i64 %112, 14, !dbg !13
  %114 = add nuw nsw i64 %113, %98, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %24) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local505 = lshr i64 %address.local499, 4, !dbg !13
  %address.local506 = and i64 %address.local505, 8, !dbg !13
  %address.local507 = and i64 %address.local499, 123, !dbg !13
  %address.local508 = lshr i64 %address.local499, 2, !dbg !13
  %address.local509 = and i64 %address.local508, 4, !dbg !13
  %address.local510 = add nuw nsw i64 %address.local509, %address.local499, !dbg !13
  %address.local511 = and i64 %address.local510, 4, !dbg !13
  %address.local512 = or disjoint i64 %address.local511, %address.local507, !dbg !13
  %address.local513 = shl nuw nsw i64 %address.local512, 7, !dbg !13
  %address.local514 = or disjoint i64 %address.local513, %address.local506, !dbg !13
  %.idx1406 = or disjoint i64 %address.local514, %114, !dbg !13
  %.idx1406.tr = trunc nuw nsw i64 %.idx1406 to i32, !dbg !13
  %dma.buffer.offset1372 = shl nuw nsw i32 %.idx1406.tr, 1, !dbg !13
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local522 = lshr i64 %address.local516, 4, !dbg !13
  %address.local523 = and i64 %address.local522, 24, !dbg !13
  %address.local531 = or disjoint i64 %address.local513, %address.local523, !dbg !13
  %.idx1407 = or disjoint i64 %address.local531, %114, !dbg !13
  %.idx1407.tr = trunc nuw nsw i64 %.idx1407 to i32, !dbg !13
  %dma.buffer.offset1369 = shl nuw nsw i32 %.idx1407.tr, 1, !dbg !13
  %115 = or disjoint i64 %114, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %.idx1408 = or disjoint i64 %address.local514, %115, !dbg !13
  %.idx1408.tr = trunc nuw nsw i64 %.idx1408 to i32, !dbg !13
  %dma.buffer.offset1366 = shl nuw nsw i32 %.idx1408.tr, 1, !dbg !13
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %116 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0, !dbg !8
  %117 = insertelement <4 x ptr addrspace(3)> %116, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %118 = insertelement <4 x ptr addrspace(3)> %117, ptr addrspace(3) %address.local533, i64 2, !dbg !8
  %119 = insertelement <4 x ptr addrspace(3)> %118, ptr addrspace(3) %address.local537, i64 3, !dbg !8
  %120 = ptrtoint <4 x ptr addrspace(3)> %119 to <4 x i64>, !dbg !8
  %121 = trunc <4 x i64> %120 to <4 x i32>, !dbg !8
  %122 = extractelement <4 x i32> %121, i64 0, !dbg !13
  %123 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %122), !dbg !13
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !13
  %124 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1372, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %125 = extractelement <4 x i32> %121, i64 1, !dbg !13
  %126 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %125), !dbg !13
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %126), !dbg !13
  %127 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %128 = extractelement <4 x i32> %121, i64 2, !dbg !13
  %129 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %128), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !13
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %131 = extractelement <4 x i32> %121, i64 3, !dbg !13
  %132 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %131), !dbg !13
  %.idx1409 = or disjoint i64 %address.local531, %115, !dbg !13
  %.idx1409.tr = trunc nuw nsw i64 %.idx1409 to i32, !dbg !13
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !13
  %133 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !12
  %address.local546 = and i64 %address.local545, 24, !dbg !12
  %address.local548 = shl i64 %address.local499, 62, !dbg !12
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !12
  %address.local550 = and i64 %address.local549, -64, !dbg !12
  %address.local553 = and i64 %address.local508, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !12
  %134 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local508, -1280, !dbg !12
  %135 = getelementptr i8, ptr addrspace(3) %134, i64 %.idx1410, !dbg !12
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local546, !dbg !12
  %137 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local550, !dbg !12
  %138 = getelementptr [2 x i8], ptr addrspace(3) %137, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %138, i64 %address.local557, !dbg !12
  %139 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %140 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %139, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !12
  %141 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %96) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %142 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %143 = getelementptr [2 x i8], ptr addrspace(3) %142, i64 %address.local553, !dbg !12
  %144 = getelementptr [2 x i8], ptr addrspace(3) %143, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %144, i64 64, !dbg !12
  %145 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %141, <4 x float> %140) #8, !dbg !12
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %145, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %144, i64 4160, !dbg !12
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !12
  %149 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %150 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %148, <4 x float> %147) #8, !dbg !12
  %151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %149, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !12
  %152 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %144, i64 192, !dbg !12
  %153 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %152, <4 x float> %151) #8, !dbg !12
  %155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %153, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %144, i64 4288, !dbg !12
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !12
  %157 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %156, <4 x float> %155) #8, !dbg !12
  %159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %157, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !12
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %144, i64 1088, !dbg !12
  %161 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %160, <4 x float> %159) #8, !dbg !12
  %163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %161, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %144, i64 5184, !dbg !12
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !12
  %165 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %164, <4 x float> %163) #8, !dbg !12
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %165, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !12
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %144, i64 1216, !dbg !12
  %169 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %168, <4 x float> %167) #8, !dbg !12
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %51, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %144, i64 5312, !dbg !12
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %53, <4 x half> %172, <4 x float> %171) #8, !dbg !12
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %139, <4 x float> zeroinitializer) #8, !dbg !12
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %141, <4 x float> %174) #8, !dbg !12
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %145, <4 x float> zeroinitializer) #8, !dbg !12
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %148, <4 x float> %176) #8, !dbg !12
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %149, <4 x float> zeroinitializer) #8, !dbg !12
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %152, <4 x float> %178) #8, !dbg !12
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %153, <4 x float> zeroinitializer) #8, !dbg !12
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %156, <4 x float> %180) #8, !dbg !12
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %157, <4 x float> zeroinitializer) #8, !dbg !12
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %160, <4 x float> %182) #8, !dbg !12
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %161, <4 x float> zeroinitializer) #8, !dbg !12
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %164, <4 x float> %184) #8, !dbg !12
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %165, <4 x float> zeroinitializer) #8, !dbg !12
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %168, <4 x float> %186) #8, !dbg !12
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %55, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !12
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %172, <4 x float> %188) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %190 = or disjoint i64 %114, 64, !dbg !13
  %.idx14111412 = or disjoint i64 %address.local514, %190, !dbg !13
  %.idx14111412.tr = trunc nuw nsw i64 %.idx14111412 to i32, !dbg !13
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !13
  %191 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14131414 = or disjoint i64 %address.local531, %190, !dbg !13
  %.idx14131414.tr = trunc nuw nsw i64 %.idx14131414 to i32, !dbg !13
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !13
  %192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !12
  %193 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %193, <4 x float> %146) #8, !dbg !12
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !12
  %195 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %144, i64 8256, !dbg !12
  %196 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %195, <4 x float> %194) #8, !dbg !12
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %196, <4 x float> %150) #8, !dbg !12
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %144, i64 12352, !dbg !12
  %199 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !12
  %200 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %199, <4 x float> %198) #8, !dbg !12
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %200, <4 x float> %154) #8, !dbg !12
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !12
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %144, i64 8384, !dbg !12
  %204 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %203, <4 x float> %202) #8, !dbg !12
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %204, <4 x float> %158) #8, !dbg !12
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %144, i64 12480, !dbg !12
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !12
  %208 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %207, <4 x float> %206) #8, !dbg !12
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %208, <4 x float> %162) #8, !dbg !12
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !12
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %144, i64 9280, !dbg !12
  %212 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %211, <4 x float> %210) #8, !dbg !12
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %212, <4 x float> %166) #8, !dbg !12
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %144, i64 13376, !dbg !12
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !12
  %216 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %215, <4 x float> %214) #8, !dbg !12
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %216, <4 x float> %170) #8, !dbg !12
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !12
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %144, i64 9408, !dbg !12
  %220 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %219, <4 x float> %218) #8, !dbg !12
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %220, <4 x float> %173) #8, !dbg !12
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %144, i64 13504, !dbg !12
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %223, <4 x float> %222) #8, !dbg !12
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %193, <4 x float> %175) #8, !dbg !12
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %195, <4 x float> %225) #8, !dbg !12
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %196, <4 x float> %177) #8, !dbg !12
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %199, <4 x float> %227) #8, !dbg !12
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %200, <4 x float> %179) #8, !dbg !12
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %203, <4 x float> %229) #8, !dbg !12
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %204, <4 x float> %181) #8, !dbg !12
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %207, <4 x float> %231) #8, !dbg !12
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %208, <4 x float> %183) #8, !dbg !12
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %211, <4 x float> %233) #8, !dbg !12
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %212, <4 x float> %185) #8, !dbg !12
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %215, <4 x float> %235) #8, !dbg !12
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %216, <4 x float> %187) #8, !dbg !12
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %219, <4 x float> %237) #8, !dbg !12
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %220, <4 x float> %189) #8, !dbg !12
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %223, <4 x float> %239) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %241 = or disjoint i64 %114, 96, !dbg !13
  %.idx14151416 = or disjoint i64 %address.local514, %241, !dbg !13
  %.idx14151416.tr = trunc nuw nsw i64 %.idx14151416 to i32, !dbg !13
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !13
  %242 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14171418 = or disjoint i64 %address.local531, %241, !dbg !13
  %.idx14171418.tr = trunc nuw nsw i64 %.idx14171418 to i32, !dbg !13
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !13
  %243 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %244 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %244, <4 x float> %197) #8, !dbg !12
  %246 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %247 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %246, <4 x float> %245) #8, !dbg !12
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %247, <4 x float> %201) #8, !dbg !12
  %250 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %251 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %250, <4 x float> %249) #8, !dbg !12
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %251, <4 x float> %205) #8, !dbg !12
  %254 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %255 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %254, <4 x float> %253) #8, !dbg !12
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %255, <4 x float> %209) #8, !dbg !12
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %259 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %258, <4 x float> %257) #8, !dbg !12
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %259, <4 x float> %213) #8, !dbg !12
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %263 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %262, <4 x float> %261) #8, !dbg !12
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %263, <4 x float> %217) #8, !dbg !12
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %267 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %266, <4 x float> %265) #8, !dbg !12
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %267, <4 x float> %221) #8, !dbg !12
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %271 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %270, <4 x float> %269) #8, !dbg !12
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %78, <4 x half> %271, <4 x float> %224) #8, !dbg !12
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %274, <4 x float> %273) #8, !dbg !12
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %244, <4 x float> %226) #8, !dbg !12
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %246, <4 x float> %276) #8, !dbg !12
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %247, <4 x float> %228) #8, !dbg !12
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %250, <4 x float> %278) #8, !dbg !12
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %251, <4 x float> %230) #8, !dbg !12
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %254, <4 x float> %280) #8, !dbg !12
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %255, <4 x float> %232) #8, !dbg !12
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %258, <4 x float> %282) #8, !dbg !12
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %259, <4 x float> %234) #8, !dbg !12
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %262, <4 x float> %284) #8, !dbg !12
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %263, <4 x float> %236) #8, !dbg !12
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %266, <4 x float> %286) #8, !dbg !12
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %267, <4 x float> %238) #8, !dbg !12
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %270, <4 x float> %288) #8, !dbg !12
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %271, <4 x float> %240) #8, !dbg !12
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %274, <4 x float> %290) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !14
  %address.local918 = and i64 %address.local917, 8064, !dbg !14
  %address.local920 = and i64 %address.local500, 24, !dbg !14
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !14
  %.idx14191420 = or disjoint i64 %address.local921, %114, !dbg !14
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !14
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !14
  %292 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !14
  %address.local924 = and i64 %address.local923, 16256, !dbg !14
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !14
  %.idx14211422 = or disjoint i64 %address.local927, %114, !dbg !14
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !14
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !14
  %293 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %294 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %294, <4 x float> %248) #8, !dbg !12
  %296 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %297 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %296, <4 x float> %295) #8, !dbg !12
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %297, <4 x float> %252) #8, !dbg !12
  %300 = extractelement <4 x float> %298, i64 0, !dbg !15
  %301 = tail call float @llvm.maxnum.f32(float %300, float -inf), !dbg !16, !frisk.local_reduction !5
  %302 = extractelement <4 x float> %298, i64 1, !dbg !15
  %303 = extractelement <4 x float> %298, i64 2, !dbg !15
  %304 = extractelement <4 x float> %298, i64 3, !dbg !15
  %305 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %306 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %305, <4 x float> %299) #8, !dbg !12
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %306, <4 x float> %256) #8, !dbg !12
  %309 = extractelement <4 x float> %307, i64 0, !dbg !15
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %301, float %309), !dbg !16, !frisk.local_reduction !5
  %310 = extractelement <4 x float> %307, i64 1, !dbg !15
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %302, float %310), !dbg !16, !frisk.local_reduction !5
  %311 = extractelement <4 x float> %307, i64 2, !dbg !15
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %303, float %311), !dbg !16, !frisk.local_reduction !5
  %312 = extractelement <4 x float> %307, i64 3, !dbg !15
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %304, float %312), !dbg !16, !frisk.local_reduction !5
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %314 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %313, <4 x float> %308) #8, !dbg !12
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %314, <4 x float> %260) #8, !dbg !12
  %317 = extractelement <4 x float> %315, i64 0, !dbg !15
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %317), !dbg !16, !frisk.local_reduction !5
  %318 = extractelement <4 x float> %315, i64 1, !dbg !15
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %318), !dbg !16, !frisk.local_reduction !5
  %319 = extractelement <4 x float> %315, i64 2, !dbg !15
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %319), !dbg !16, !frisk.local_reduction !5
  %320 = extractelement <4 x float> %315, i64 3, !dbg !15
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %320), !dbg !16, !frisk.local_reduction !5
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %322 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %321, <4 x float> %316) #8, !dbg !12
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %322, <4 x float> %264) #8, !dbg !12
  %325 = extractelement <4 x float> %323, i64 0, !dbg !15
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %325), !dbg !16, !frisk.local_reduction !5
  %326 = extractelement <4 x float> %323, i64 1, !dbg !15
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %326), !dbg !16, !frisk.local_reduction !5
  %327 = extractelement <4 x float> %323, i64 2, !dbg !15
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %327), !dbg !16, !frisk.local_reduction !5
  %328 = extractelement <4 x float> %323, i64 3, !dbg !15
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %328), !dbg !16, !frisk.local_reduction !5
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %330 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %329, <4 x float> %324) #8, !dbg !12
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %330, <4 x float> %268) #8, !dbg !12
  %333 = extractelement <4 x float> %331, i64 0, !dbg !15
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %333), !dbg !16, !frisk.local_reduction !5
  %334 = extractelement <4 x float> %331, i64 1, !dbg !15
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %334), !dbg !16, !frisk.local_reduction !5
  %335 = extractelement <4 x float> %331, i64 2, !dbg !15
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %335), !dbg !16, !frisk.local_reduction !5
  %336 = extractelement <4 x float> %331, i64 3, !dbg !15
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %336), !dbg !16, !frisk.local_reduction !5
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %338 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %337, <4 x float> %332) #8, !dbg !12
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %338, <4 x float> %272) #8, !dbg !12
  %341 = extractelement <4 x float> %339, i64 0, !dbg !15
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %341), !dbg !16, !frisk.local_reduction !5
  %342 = extractelement <4 x float> %339, i64 1, !dbg !15
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %342), !dbg !16, !frisk.local_reduction !5
  %343 = extractelement <4 x float> %339, i64 2, !dbg !15
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %343), !dbg !16, !frisk.local_reduction !5
  %344 = extractelement <4 x float> %339, i64 3, !dbg !15
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %344), !dbg !16, !frisk.local_reduction !5
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %346 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %345, <4 x float> %340) #8, !dbg !12
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %86, <4 x half> %346, <4 x float> %275) #8, !dbg !12
  %349 = extractelement <4 x float> %347, i64 0, !dbg !15
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %349), !dbg !16, !frisk.local_reduction !5
  %350 = extractelement <4 x float> %347, i64 1, !dbg !15
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %350), !dbg !16, !frisk.local_reduction !5
  %351 = extractelement <4 x float> %347, i64 2, !dbg !15
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %351), !dbg !16, !frisk.local_reduction !5
  %352 = extractelement <4 x float> %347, i64 3, !dbg !15
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %352), !dbg !16, !frisk.local_reduction !5
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %353, <4 x float> %348) #8, !dbg !12
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %294, <4 x float> %277) #8, !dbg !12
  %356 = extractelement <4 x float> %354, i64 0, !dbg !15
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %356), !dbg !16, !frisk.local_reduction !5
  %357 = extractelement <4 x float> %354, i64 1, !dbg !15
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %357), !dbg !16, !frisk.local_reduction !5
  %358 = extractelement <4 x float> %354, i64 2, !dbg !15
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %358), !dbg !16, !frisk.local_reduction !5
  %359 = extractelement <4 x float> %354, i64 3, !dbg !15
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %359), !dbg !16, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !16, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !16, !frisk.local_reduction !5
  %360 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !16, !frisk.local_reduction !5
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %296, <4 x float> %355) #8, !dbg !12
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %297, <4 x float> %279) #8, !dbg !12
  %363 = extractelement <4 x float> %361, i64 0, !dbg !15
  %364 = extractelement <4 x float> %361, i64 1, !dbg !15
  %365 = extractelement <4 x float> %361, i64 2, !dbg !15
  %366 = extractelement <4 x float> %361, i64 3, !dbg !15
  %367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %305, <4 x float> %362) #8, !dbg !12
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %306, <4 x float> %281) #8, !dbg !12
  %369 = extractelement <4 x float> %367, i64 0, !dbg !15
  %370 = extractelement <4 x float> %367, i64 1, !dbg !15
  %371 = extractelement <4 x float> %367, i64 2, !dbg !15
  %372 = extractelement <4 x float> %367, i64 3, !dbg !15
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %313, <4 x float> %368) #8, !dbg !12
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %314, <4 x float> %283) #8, !dbg !12
  %375 = extractelement <4 x float> %373, i64 0, !dbg !15
  %376 = extractelement <4 x float> %373, i64 1, !dbg !15
  %377 = extractelement <4 x float> %373, i64 2, !dbg !15
  %378 = extractelement <4 x float> %373, i64 3, !dbg !15
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %321, <4 x float> %374) #8, !dbg !12
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %322, <4 x float> %285) #8, !dbg !12
  %381 = extractelement <4 x float> %379, i64 0, !dbg !15
  %382 = extractelement <4 x float> %379, i64 1, !dbg !15
  %383 = extractelement <4 x float> %379, i64 2, !dbg !15
  %384 = extractelement <4 x float> %379, i64 3, !dbg !15
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %329, <4 x float> %380) #8, !dbg !12
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %330, <4 x float> %287) #8, !dbg !12
  %387 = extractelement <4 x float> %385, i64 0, !dbg !15
  %388 = extractelement <4 x float> %385, i64 1, !dbg !15
  %389 = extractelement <4 x float> %385, i64 2, !dbg !15
  %390 = extractelement <4 x float> %385, i64 3, !dbg !15
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %337, <4 x float> %386) #8, !dbg !12
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %338, <4 x float> %289) #8, !dbg !12
  %393 = extractelement <4 x float> %391, i64 0, !dbg !15
  %394 = extractelement <4 x float> %391, i64 1, !dbg !15
  %395 = extractelement <4 x float> %391, i64 2, !dbg !15
  %396 = extractelement <4 x float> %391, i64 3, !dbg !15
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %345, <4 x float> %392) #8, !dbg !12
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %346, <4 x float> %291) #8, !dbg !12
  %399 = extractelement <4 x float> %397, i64 0, !dbg !15
  %400 = extractelement <4 x float> %397, i64 1, !dbg !15
  %401 = extractelement <4 x float> %397, i64 2, !dbg !15
  %402 = extractelement <4 x float> %397, i64 3, !dbg !15
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %353, <4 x float> %398) #8, !dbg !12
  %404 = extractelement <4 x float> %403, i64 0, !dbg !15
  %405 = extractelement <4 x float> %403, i64 1, !dbg !15
  %406 = extractelement <4 x float> %403, i64 2, !dbg !15
  %407 = extractelement <4 x float> %403, i64 3, !dbg !15
  %408 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %409 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %408), !dbg !16
  %410 = add nuw nsw i32 %409, 64, !dbg !16
  %411 = xor i32 %409, 16, !dbg !16
  %412 = and i32 %410, 192, !dbg !16
  %413 = icmp samesign ult i32 %411, %412, !dbg !16
  %414 = select i1 %413, i32 %411, i32 %409, !dbg !16
  %415 = shl nuw nsw i32 %414, 2, !dbg !16
  %416 = bitcast float %360 to i32, !dbg !16
  %417 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %416), !dbg !16
  %418 = bitcast i32 %417 to float, !dbg !16
  %419 = tail call float @llvm.maxnum.f32(float %360, float %418), !dbg !16
  %420 = xor i32 %409, 32, !dbg !16
  %421 = icmp samesign ult i32 %420, %412, !dbg !16
  %422 = select i1 %421, i32 %420, i32 %409, !dbg !16
  %423 = shl nuw nsw i32 %422, 2, !dbg !16
  %424 = bitcast float %419 to i32, !dbg !16
  %425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %424), !dbg !16
  %426 = bitcast i32 %425 to float, !dbg !16
  %427 = tail call float @llvm.maxnum.f32(float %419, float %426), !dbg !16
  %428 = extractvalue [2 x <1 x float>] %109, 0, !dbg !16
  %429 = insertelement <1 x float> poison, float %427, i64 0, !dbg !16
  %430 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %429, <1 x float> %428), !dbg !16
  %431 = tail call float @llvm.maxnum.f32(float %363, float -inf), !dbg !16, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %431, float %369), !dbg !16, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %364, float %370), !dbg !16, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %365, float %371), !dbg !16, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %366, float %372), !dbg !16, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %375), !dbg !16, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %376), !dbg !16, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %377), !dbg !16, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %378), !dbg !16, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %381), !dbg !16, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %382), !dbg !16, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %383), !dbg !16, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %384), !dbg !16, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %387), !dbg !16, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %388), !dbg !16, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %389), !dbg !16, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %390), !dbg !16, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %393), !dbg !16, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %394), !dbg !16, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %395), !dbg !16, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %396), !dbg !16, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %399), !dbg !16, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %400), !dbg !16, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %401), !dbg !16, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %402), !dbg !16, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %404), !dbg !16, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %405), !dbg !16, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %406), !dbg !16, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %407), !dbg !16, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !16, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !16, !frisk.local_reduction !5
  %432 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !16, !frisk.local_reduction !5
  %433 = bitcast float %432 to i32, !dbg !16
  %434 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %433), !dbg !16
  %435 = bitcast i32 %434 to float, !dbg !16
  %436 = tail call float @llvm.maxnum.f32(float %432, float %435), !dbg !16
  %437 = bitcast float %436 to i32, !dbg !16
  %438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %437), !dbg !16
  %439 = bitcast i32 %438 to float, !dbg !16
  %440 = tail call float @llvm.maxnum.f32(float %436, float %439), !dbg !16
  %441 = extractvalue [2 x <1 x float>] %109, 1, !dbg !16
  %442 = insertelement <1 x float> poison, float %440, i64 0, !dbg !16
  %443 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %442, <1 x float> %441), !dbg !16
  %444 = shufflevector <1 x float> %430, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %445 = fsub <4 x float> %298, %444, !dbg !17
  %446 = fsub <4 x float> %307, %444, !dbg !17
  %447 = fsub <4 x float> %315, %444, !dbg !17
  %448 = fsub <4 x float> %323, %444, !dbg !17
  %449 = fsub <4 x float> %331, %444, !dbg !17
  %450 = fsub <4 x float> %339, %444, !dbg !17
  %451 = fsub <4 x float> %347, %444, !dbg !17
  %452 = fsub <4 x float> %354, %444, !dbg !17
  %453 = shufflevector <1 x float> %443, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %454 = fsub <4 x float> %361, %453, !dbg !17
  %455 = fsub <4 x float> %367, %453, !dbg !17
  %456 = fsub <4 x float> %373, %453, !dbg !17
  %457 = fsub <4 x float> %379, %453, !dbg !17
  %458 = fsub <4 x float> %385, %453, !dbg !17
  %459 = fsub <4 x float> %391, %453, !dbg !17
  %460 = fsub <4 x float> %397, %453, !dbg !17
  %461 = fsub <4 x float> %403, %453, !dbg !17
  %462 = fsub <1 x float> %428, %430, !dbg !18
  %463 = extractelement <1 x float> %462, i64 0, !dbg !19
  %464 = tail call float @llvm.amdgcn.exp2.f32(float %463), !dbg !19
  %465 = insertelement <4 x float> poison, float %464, i64 0, !dbg !20
  %466 = shufflevector <4 x float> %465, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %467 = extractvalue [2 x <1 x float>] %110, 0, !dbg !21
  %468 = extractelement <1 x float> %467, i64 0, !dbg !21
  %469 = fmul float %468, %464, !dbg !21
  %470 = fsub <1 x float> %441, %443, !dbg !18
  %471 = extractelement <1 x float> %470, i64 0, !dbg !19
  %472 = tail call float @llvm.amdgcn.exp2.f32(float %471), !dbg !19
  %473 = insertelement <4 x float> poison, float %472, i64 0, !dbg !20
  %474 = shufflevector <4 x float> %473, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %475 = extractvalue [2 x <1 x float>] %110, 1, !dbg !21
  %476 = extractelement <1 x float> %475, i64 0, !dbg !21
  %477 = fmul float %476, %472, !dbg !21
  %478 = extractelement <4 x float> %445, i64 0, !dbg !22
  %479 = tail call float @llvm.amdgcn.exp2.f32(float %478), !dbg !22
  %480 = insertelement <4 x float> poison, float %479, i64 0, !dbg !23
  %481 = extractelement <4 x float> %445, i64 1, !dbg !22
  %482 = tail call float @llvm.amdgcn.exp2.f32(float %481), !dbg !22
  %483 = insertelement <4 x float> %480, float %482, i64 1, !dbg !23
  %484 = extractelement <4 x float> %445, i64 2, !dbg !22
  %485 = tail call float @llvm.amdgcn.exp2.f32(float %484), !dbg !22
  %486 = insertelement <4 x float> %483, float %485, i64 2, !dbg !23
  %487 = extractelement <4 x float> %445, i64 3, !dbg !22
  %488 = tail call float @llvm.amdgcn.exp2.f32(float %487), !dbg !22
  %489 = insertelement <4 x float> %486, float %488, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %489) #8, !dbg !23, !frisk.register_lifetimes !5
  %490 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %490) #8, !dbg !22, !frisk.register_lifetimes !5
  %491 = extractelement <4 x float> %446, i64 0, !dbg !22
  %492 = tail call float @llvm.amdgcn.exp2.f32(float %491), !dbg !22
  %493 = insertelement <4 x float> poison, float %492, i64 0, !dbg !23
  %local.reduce89 = fadd float %479, %492, !dbg !24, !frisk.local_reduction !5
  %494 = extractelement <4 x float> %446, i64 1, !dbg !22
  %495 = tail call float @llvm.amdgcn.exp2.f32(float %494), !dbg !22
  %496 = insertelement <4 x float> %493, float %495, i64 1, !dbg !23
  %local.reduce90 = fadd float %482, %495, !dbg !24, !frisk.local_reduction !5
  %497 = extractelement <4 x float> %446, i64 2, !dbg !22
  %498 = tail call float @llvm.amdgcn.exp2.f32(float %497), !dbg !22
  %499 = insertelement <4 x float> %496, float %498, i64 2, !dbg !23
  %local.reduce91 = fadd float %485, %498, !dbg !24, !frisk.local_reduction !5
  %500 = extractelement <4 x float> %446, i64 3, !dbg !22
  %501 = tail call float @llvm.amdgcn.exp2.f32(float %500), !dbg !22
  %502 = insertelement <4 x float> %499, float %501, i64 3, !dbg !23
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %502) #8, !dbg !23, !frisk.register_lifetimes !5
  %503 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !23
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %503) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %488, %501, !dbg !24, !frisk.local_reduction !5
  %504 = extractelement <4 x float> %447, i64 0, !dbg !22
  %505 = tail call float @llvm.amdgcn.exp2.f32(float %504), !dbg !22
  %506 = insertelement <4 x float> poison, float %505, i64 0, !dbg !23
  %local.reduce93 = fadd float %505, %local.reduce89, !dbg !24, !frisk.local_reduction !5
  %507 = extractelement <4 x float> %447, i64 1, !dbg !22
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %507), !dbg !22
  %509 = insertelement <4 x float> %506, float %508, i64 1, !dbg !23
  %local.reduce94 = fadd float %508, %local.reduce90, !dbg !24, !frisk.local_reduction !5
  %510 = extractelement <4 x float> %447, i64 2, !dbg !22
  %511 = tail call float @llvm.amdgcn.exp2.f32(float %510), !dbg !22
  %512 = insertelement <4 x float> %509, float %511, i64 2, !dbg !23
  %local.reduce95 = fadd float %511, %local.reduce91, !dbg !24, !frisk.local_reduction !5
  %513 = extractelement <4 x float> %447, i64 3, !dbg !22
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %513), !dbg !22
  %515 = insertelement <4 x float> %512, float %514, i64 3, !dbg !23
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %515) #8, !dbg !23, !frisk.register_lifetimes !5
  %516 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !23
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %516) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %514, %local.reduce92, !dbg !24, !frisk.local_reduction !5
  %517 = extractelement <4 x float> %448, i64 0, !dbg !22
  %518 = tail call float @llvm.amdgcn.exp2.f32(float %517), !dbg !22
  %519 = insertelement <4 x float> poison, float %518, i64 0, !dbg !23
  %local.reduce97 = fadd float %518, %local.reduce93, !dbg !24, !frisk.local_reduction !5
  %520 = extractelement <4 x float> %448, i64 1, !dbg !22
  %521 = tail call float @llvm.amdgcn.exp2.f32(float %520), !dbg !22
  %522 = insertelement <4 x float> %519, float %521, i64 1, !dbg !23
  %local.reduce98 = fadd float %521, %local.reduce94, !dbg !24, !frisk.local_reduction !5
  %523 = extractelement <4 x float> %448, i64 2, !dbg !22
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %523), !dbg !22
  %525 = insertelement <4 x float> %522, float %524, i64 2, !dbg !23
  %local.reduce99 = fadd float %524, %local.reduce95, !dbg !24, !frisk.local_reduction !5
  %526 = extractelement <4 x float> %448, i64 3, !dbg !22
  %527 = tail call float @llvm.amdgcn.exp2.f32(float %526), !dbg !22
  %528 = insertelement <4 x float> %525, float %527, i64 3, !dbg !23
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %528) #8, !dbg !23, !frisk.register_lifetimes !5
  %529 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !23
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %529) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %527, %local.reduce96, !dbg !24, !frisk.local_reduction !5
  %530 = extractelement <4 x float> %449, i64 0, !dbg !22
  %531 = tail call float @llvm.amdgcn.exp2.f32(float %530), !dbg !22
  %532 = insertelement <4 x float> poison, float %531, i64 0, !dbg !23
  %local.reduce101 = fadd float %531, %local.reduce97, !dbg !24, !frisk.local_reduction !5
  %533 = extractelement <4 x float> %449, i64 1, !dbg !22
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !22
  %535 = insertelement <4 x float> %532, float %534, i64 1, !dbg !23
  %local.reduce102 = fadd float %534, %local.reduce98, !dbg !24, !frisk.local_reduction !5
  %536 = extractelement <4 x float> %449, i64 2, !dbg !22
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !22
  %538 = insertelement <4 x float> %535, float %537, i64 2, !dbg !23
  %local.reduce103 = fadd float %537, %local.reduce99, !dbg !24, !frisk.local_reduction !5
  %539 = extractelement <4 x float> %449, i64 3, !dbg !22
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %539), !dbg !22
  %541 = insertelement <4 x float> %538, float %540, i64 3, !dbg !23
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %541) #8, !dbg !23, !frisk.register_lifetimes !5
  %542 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !23
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %542) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %540, %local.reduce100, !dbg !24, !frisk.local_reduction !5
  %543 = extractelement <4 x float> %450, i64 0, !dbg !22
  %544 = tail call float @llvm.amdgcn.exp2.f32(float %543), !dbg !22
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !23
  %local.reduce105 = fadd float %544, %local.reduce101, !dbg !24, !frisk.local_reduction !5
  %546 = extractelement <4 x float> %450, i64 1, !dbg !22
  %547 = tail call float @llvm.amdgcn.exp2.f32(float %546), !dbg !22
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !23
  %local.reduce106 = fadd float %547, %local.reduce102, !dbg !24, !frisk.local_reduction !5
  %549 = extractelement <4 x float> %450, i64 2, !dbg !22
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %549), !dbg !22
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !23
  %local.reduce107 = fadd float %550, %local.reduce103, !dbg !24, !frisk.local_reduction !5
  %552 = extractelement <4 x float> %450, i64 3, !dbg !22
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !22
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !23
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #8, !dbg !23, !frisk.register_lifetimes !5
  %555 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !23
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %553, %local.reduce104, !dbg !24, !frisk.local_reduction !5
  %556 = extractelement <4 x float> %451, i64 0, !dbg !22
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %556), !dbg !22
  %558 = insertelement <4 x float> poison, float %557, i64 0, !dbg !23
  %local.reduce109 = fadd float %557, %local.reduce105, !dbg !24, !frisk.local_reduction !5
  %559 = extractelement <4 x float> %451, i64 1, !dbg !22
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !22
  %561 = insertelement <4 x float> %558, float %560, i64 1, !dbg !23
  %local.reduce110 = fadd float %560, %local.reduce106, !dbg !24, !frisk.local_reduction !5
  %562 = extractelement <4 x float> %451, i64 2, !dbg !22
  %563 = tail call float @llvm.amdgcn.exp2.f32(float %562), !dbg !22
  %564 = insertelement <4 x float> %561, float %563, i64 2, !dbg !23
  %local.reduce111 = fadd float %563, %local.reduce107, !dbg !24, !frisk.local_reduction !5
  %565 = extractelement <4 x float> %451, i64 3, !dbg !22
  %566 = tail call float @llvm.amdgcn.exp2.f32(float %565), !dbg !22
  %567 = insertelement <4 x float> %564, float %566, i64 3, !dbg !23
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %567) #8, !dbg !23, !frisk.register_lifetimes !5
  %568 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !23
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %568) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %566, %local.reduce108, !dbg !24, !frisk.local_reduction !5
  %569 = extractelement <4 x float> %452, i64 0, !dbg !22
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %569), !dbg !22
  %571 = insertelement <4 x float> poison, float %570, i64 0, !dbg !23
  %local.reduce113 = fadd float %570, %local.reduce109, !dbg !24, !frisk.local_reduction !5
  %572 = extractelement <4 x float> %452, i64 1, !dbg !22
  %573 = tail call float @llvm.amdgcn.exp2.f32(float %572), !dbg !22
  %574 = insertelement <4 x float> %571, float %573, i64 1, !dbg !23
  %local.reduce114 = fadd float %573, %local.reduce110, !dbg !24, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !24, !frisk.local_reduction !5
  %575 = extractelement <4 x float> %452, i64 2, !dbg !22
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %575), !dbg !22
  %577 = insertelement <4 x float> %574, float %576, i64 2, !dbg !23
  %local.reduce115 = fadd float %576, %local.reduce111, !dbg !24, !frisk.local_reduction !5
  %578 = extractelement <4 x float> %452, i64 3, !dbg !22
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !22
  %580 = insertelement <4 x float> %577, float %579, i64 3, !dbg !23
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %580) #8, !dbg !23, !frisk.register_lifetimes !5
  %581 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !23
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %581) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %579, %local.reduce112, !dbg !24, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !24, !frisk.local_reduction !5
  %582 = fadd float %local.reduce117, %local.reduce118, !dbg !24, !frisk.local_reduction !5
  %583 = fadd float %469, %582, !dbg !24
  %584 = insertelement <1 x float> poison, float %583, i64 0, !dbg !11
  %585 = insertvalue [2 x <1 x float>] poison, <1 x float> %584, 0, !dbg !11
  %586 = extractelement <4 x float> %454, i64 0, !dbg !22
  %587 = tail call float @llvm.amdgcn.exp2.f32(float %586), !dbg !22
  %588 = insertelement <4 x float> poison, float %587, i64 0, !dbg !23
  %589 = extractelement <4 x float> %454, i64 1, !dbg !22
  %590 = tail call float @llvm.amdgcn.exp2.f32(float %589), !dbg !22
  %591 = insertelement <4 x float> %588, float %590, i64 1, !dbg !23
  %592 = extractelement <4 x float> %454, i64 2, !dbg !22
  %593 = tail call float @llvm.amdgcn.exp2.f32(float %592), !dbg !22
  %594 = insertelement <4 x float> %591, float %593, i64 2, !dbg !23
  %595 = extractelement <4 x float> %454, i64 3, !dbg !22
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %595), !dbg !22
  %597 = insertelement <4 x float> %594, float %596, i64 3, !dbg !23
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %597) #8, !dbg !23, !frisk.register_lifetimes !5
  %598 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !23
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %598) #8, !dbg !22, !frisk.register_lifetimes !5
  %599 = extractelement <4 x float> %455, i64 0, !dbg !22
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %599), !dbg !22
  %601 = insertelement <4 x float> poison, float %600, i64 0, !dbg !23
  %local.reduce = fadd float %587, %600, !dbg !24, !frisk.local_reduction !5
  %602 = extractelement <4 x float> %455, i64 1, !dbg !22
  %603 = tail call float @llvm.amdgcn.exp2.f32(float %602), !dbg !22
  %604 = insertelement <4 x float> %601, float %603, i64 1, !dbg !23
  %local.reduce59 = fadd float %590, %603, !dbg !24, !frisk.local_reduction !5
  %605 = extractelement <4 x float> %455, i64 2, !dbg !22
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %605), !dbg !22
  %607 = insertelement <4 x float> %604, float %606, i64 2, !dbg !23
  %local.reduce60 = fadd float %593, %606, !dbg !24, !frisk.local_reduction !5
  %608 = extractelement <4 x float> %455, i64 3, !dbg !22
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %608), !dbg !22
  %610 = insertelement <4 x float> %607, float %609, i64 3, !dbg !23
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %610) #8, !dbg !23, !frisk.register_lifetimes !5
  %611 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !23
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %611) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce61 = fadd float %596, %609, !dbg !24, !frisk.local_reduction !5
  %612 = extractelement <4 x float> %456, i64 0, !dbg !22
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !22
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !23
  %local.reduce62 = fadd float %613, %local.reduce, !dbg !24, !frisk.local_reduction !5
  %615 = extractelement <4 x float> %456, i64 1, !dbg !22
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %615), !dbg !22
  %617 = insertelement <4 x float> %614, float %616, i64 1, !dbg !23
  %local.reduce63 = fadd float %616, %local.reduce59, !dbg !24, !frisk.local_reduction !5
  %618 = extractelement <4 x float> %456, i64 2, !dbg !22
  %619 = tail call float @llvm.amdgcn.exp2.f32(float %618), !dbg !22
  %620 = insertelement <4 x float> %617, float %619, i64 2, !dbg !23
  %local.reduce64 = fadd float %619, %local.reduce60, !dbg !24, !frisk.local_reduction !5
  %621 = extractelement <4 x float> %456, i64 3, !dbg !22
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %621), !dbg !22
  %623 = insertelement <4 x float> %620, float %622, i64 3, !dbg !23
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #8, !dbg !23, !frisk.register_lifetimes !5
  %624 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !23
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce65 = fadd float %622, %local.reduce61, !dbg !24, !frisk.local_reduction !5
  %625 = extractelement <4 x float> %457, i64 0, !dbg !22
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %625), !dbg !22
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !23
  %local.reduce66 = fadd float %626, %local.reduce62, !dbg !24, !frisk.local_reduction !5
  %628 = extractelement <4 x float> %457, i64 1, !dbg !22
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !22
  %630 = insertelement <4 x float> %627, float %629, i64 1, !dbg !23
  %local.reduce67 = fadd float %629, %local.reduce63, !dbg !24, !frisk.local_reduction !5
  %631 = extractelement <4 x float> %457, i64 2, !dbg !22
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %631), !dbg !22
  %633 = insertelement <4 x float> %630, float %632, i64 2, !dbg !23
  %local.reduce68 = fadd float %632, %local.reduce64, !dbg !24, !frisk.local_reduction !5
  %634 = extractelement <4 x float> %457, i64 3, !dbg !22
  %635 = tail call float @llvm.amdgcn.exp2.f32(float %634), !dbg !22
  %636 = insertelement <4 x float> %633, float %635, i64 3, !dbg !23
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #8, !dbg !23, !frisk.register_lifetimes !5
  %637 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !23
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce69 = fadd float %635, %local.reduce65, !dbg !24, !frisk.local_reduction !5
  %638 = extractelement <4 x float> %458, i64 0, !dbg !22
  %639 = tail call float @llvm.amdgcn.exp2.f32(float %638), !dbg !22
  %640 = insertelement <4 x float> poison, float %639, i64 0, !dbg !23
  %local.reduce70 = fadd float %639, %local.reduce66, !dbg !24, !frisk.local_reduction !5
  %641 = extractelement <4 x float> %458, i64 1, !dbg !22
  %642 = tail call float @llvm.amdgcn.exp2.f32(float %641), !dbg !22
  %643 = insertelement <4 x float> %640, float %642, i64 1, !dbg !23
  %local.reduce71 = fadd float %642, %local.reduce67, !dbg !24, !frisk.local_reduction !5
  %644 = extractelement <4 x float> %458, i64 2, !dbg !22
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %644), !dbg !22
  %646 = insertelement <4 x float> %643, float %645, i64 2, !dbg !23
  %local.reduce72 = fadd float %645, %local.reduce68, !dbg !24, !frisk.local_reduction !5
  %647 = extractelement <4 x float> %458, i64 3, !dbg !22
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !22
  %649 = insertelement <4 x float> %646, float %648, i64 3, !dbg !23
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #8, !dbg !23, !frisk.register_lifetimes !5
  %650 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !23
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce73 = fadd float %648, %local.reduce69, !dbg !24, !frisk.local_reduction !5
  %651 = extractelement <4 x float> %459, i64 0, !dbg !22
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !22
  %653 = insertelement <4 x float> poison, float %652, i64 0, !dbg !23
  %local.reduce74 = fadd float %652, %local.reduce70, !dbg !24, !frisk.local_reduction !5
  %654 = extractelement <4 x float> %459, i64 1, !dbg !22
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !22
  %656 = insertelement <4 x float> %653, float %655, i64 1, !dbg !23
  %local.reduce75 = fadd float %655, %local.reduce71, !dbg !24, !frisk.local_reduction !5
  %657 = extractelement <4 x float> %459, i64 2, !dbg !22
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %657), !dbg !22
  %659 = insertelement <4 x float> %656, float %658, i64 2, !dbg !23
  %local.reduce76 = fadd float %658, %local.reduce72, !dbg !24, !frisk.local_reduction !5
  %660 = extractelement <4 x float> %459, i64 3, !dbg !22
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %660), !dbg !22
  %662 = insertelement <4 x float> %659, float %661, i64 3, !dbg !23
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %662) #8, !dbg !23, !frisk.register_lifetimes !5
  %663 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !23
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %663) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce77 = fadd float %661, %local.reduce73, !dbg !24, !frisk.local_reduction !5
  %664 = extractelement <4 x float> %460, i64 0, !dbg !22
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %664), !dbg !22
  %666 = insertelement <4 x float> poison, float %665, i64 0, !dbg !23
  %local.reduce78 = fadd float %665, %local.reduce74, !dbg !24, !frisk.local_reduction !5
  %667 = extractelement <4 x float> %460, i64 1, !dbg !22
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %667), !dbg !22
  %669 = insertelement <4 x float> %666, float %668, i64 1, !dbg !23
  %local.reduce79 = fadd float %668, %local.reduce75, !dbg !24, !frisk.local_reduction !5
  %670 = extractelement <4 x float> %460, i64 2, !dbg !22
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !22
  %672 = insertelement <4 x float> %669, float %671, i64 2, !dbg !23
  %local.reduce80 = fadd float %671, %local.reduce76, !dbg !24, !frisk.local_reduction !5
  %673 = extractelement <4 x float> %460, i64 3, !dbg !22
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %673), !dbg !22
  %675 = insertelement <4 x float> %672, float %674, i64 3, !dbg !23
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #8, !dbg !23, !frisk.register_lifetimes !5
  %676 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !23
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce81 = fadd float %674, %local.reduce77, !dbg !24, !frisk.local_reduction !5
  %677 = extractelement <4 x float> %461, i64 0, !dbg !22
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !22
  %679 = insertelement <4 x float> poison, float %678, i64 0, !dbg !23
  %local.reduce82 = fadd float %678, %local.reduce78, !dbg !24, !frisk.local_reduction !5
  %680 = extractelement <4 x float> %461, i64 1, !dbg !22
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %680), !dbg !22
  %682 = insertelement <4 x float> %679, float %681, i64 1, !dbg !23
  %local.reduce83 = fadd float %681, %local.reduce79, !dbg !24, !frisk.local_reduction !5
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !24, !frisk.local_reduction !5
  %683 = extractelement <4 x float> %461, i64 2, !dbg !22
  %684 = tail call float @llvm.amdgcn.exp2.f32(float %683), !dbg !22
  %685 = insertelement <4 x float> %682, float %684, i64 2, !dbg !23
  %local.reduce84 = fadd float %684, %local.reduce80, !dbg !24, !frisk.local_reduction !5
  %686 = extractelement <4 x float> %461, i64 3, !dbg !22
  %687 = tail call float @llvm.amdgcn.exp2.f32(float %686), !dbg !22
  %688 = insertelement <4 x float> %685, float %687, i64 3, !dbg !23
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %688) #8, !dbg !23, !frisk.register_lifetimes !5
  %689 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !23
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %689) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce85 = fadd float %687, %local.reduce81, !dbg !24, !frisk.local_reduction !5
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !24, !frisk.local_reduction !5
  %690 = fadd float %local.reduce86, %local.reduce87, !dbg !24, !frisk.local_reduction !5
  %691 = fadd float %477, %690, !dbg !24
  %692 = insertelement <1 x float> poison, float %691, i64 0, !dbg !11
  %693 = insertvalue [2 x <1 x float>] %585, <1 x float> %692, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %vdb.address1436 = or disjoint i64 %address.local921, %115, !dbg !14
  %vdb.address1437 = trunc nuw nsw i64 %vdb.address1436 to i32, !dbg !14
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !14
  %vdb.address1445 = or disjoint i64 %address.local927, %115, !dbg !14
  %vdb.address1446 = trunc nuw nsw i64 %vdb.address1445 to i32, !dbg !14
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !14
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !25
  %694 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !25
  %695 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !25
  %address.local930 = and i32 %address.local929, 120, !dbg !25
  %address.local932 = and i32 %address.local575, 1536, !dbg !25
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !25
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !25
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !25
  %696 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %697 = bitcast <4 x i32> %696 to <8 x half>, !dbg !25
  %address.local936 = and i64 %address.local499, 60, !dbg !25
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !25
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !25
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !25
  %address.local940 = lshr i64 %address.local555, 4, !dbg !25
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !25
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !25
  %address.local944.idx = mul nsw i64 %address.local508, -64, !dbg !25
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !25
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !25
  %698 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %699 = bitcast <4 x i32> %698 to <8 x half>, !dbg !25
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !25
  %700 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %701 = bitcast <4 x i32> %700 to <8 x half>, !dbg !25
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !25
  %702 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %703 = bitcast <4 x i32> %702 to <8 x half>, !dbg !25
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !25
  %704 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %705 = bitcast <4 x i32> %704 to <8 x half>, !dbg !25
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !25
  %706 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !25
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !25
  %708 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !25
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !25
  %710 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %711 = shufflevector <8 x half> %697, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %712 = extractvalue [2 x <8 x float>] %111, 0, !dbg !20
  %713 = shufflevector <8 x float> %712, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %714 = fmul <4 x float> %713, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %715 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %711, <4 x float> %714) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %716 = shufflevector <8 x half> %697, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %717 = shufflevector <8 x float> %712, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %718 = fmul <4 x float> %717, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %719 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %716, <4 x float> %718) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %720 = extractvalue [2 x <8 x float>] %111, 1, !dbg !20
  %721 = shufflevector <8 x float> %720, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %722 = fmul <4 x float> %721, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %723 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %711, <4 x float> %722) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %724 = shufflevector <8 x float> %720, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %725 = fmul <4 x float> %724, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %726 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %716, <4 x float> %725) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %727 = shufflevector <8 x half> %699, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %728 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %727, <4 x float> %715) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %729 = shufflevector <8 x half> %699, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %730 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %729, <4 x float> %719) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %727, <4 x float> %723) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %729, <4 x float> %726) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %733 = shufflevector <8 x half> %701, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %733, <4 x float> %728) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %735 = shufflevector <8 x half> %701, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %735, <4 x float> %730) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %733, <4 x float> %731) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %738 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %735, <4 x float> %732) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %739 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %740 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %739, <4 x float> %734) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %741 = shufflevector <8 x half> %703, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %741, <4 x float> %736) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %739, <4 x float> %737) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %741, <4 x float> %738) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %745 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %746 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %745, <4 x float> %740) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %747 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %747, <4 x float> %742) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %745, <4 x float> %743) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %750 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %747, <4 x float> %744) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %751 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %752 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %751, <4 x float> %746) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %753 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %754 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %753, <4 x float> %748) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %751, <4 x float> %749) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %756 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %753, <4 x float> %750) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %757 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %758 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %757, <4 x float> %752) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %759 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %760 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %759, <4 x float> %754) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %761 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %757, <4 x float> %755) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %759, <4 x float> %756) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %763 = bitcast <4 x i32> %710 to <8 x half>, !dbg !25
  %764 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %765 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %764, <4 x float> %758) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %766 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %766, <4 x float> %760) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %768 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %764, <4 x float> %761) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %769 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %766, <4 x float> %762) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %vdb.address1462 = or disjoint i64 %address.local921, %190, !dbg !14
  %vdb.address1463 = trunc nuw nsw i64 %vdb.address1462 to i32, !dbg !14
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !14
  %vdb.address1471 = or disjoint i64 %address.local927, %190, !dbg !14
  %vdb.address1472 = trunc nuw nsw i64 %vdb.address1471 to i32, !dbg !14
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !14
  %770 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %771 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !25
  %772 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %773 = bitcast <4 x i32> %772 to <8 x half>, !dbg !25
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !25
  %774 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %775 = bitcast <4 x i32> %774 to <8 x half>, !dbg !25
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !25
  %776 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %777 = bitcast <4 x i32> %776 to <8 x half>, !dbg !25
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !25
  %778 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %779 = bitcast <4 x i32> %778 to <8 x half>, !dbg !25
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !25
  %780 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !25
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !25
  %782 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !25
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !25
  %784 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !25
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !25
  %786 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %787 = shufflevector <8 x half> %773, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %788 = extractvalue [2 x <8 x float>] %108, 0, !dbg !20
  %789 = shufflevector <8 x float> %788, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %790 = fmul <4 x float> %789, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %787, <4 x float> %790) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %792 = shufflevector <8 x half> %773, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %793 = shufflevector <8 x float> %788, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %794 = fmul <4 x float> %793, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %792, <4 x float> %794) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %796 = extractvalue [2 x <8 x float>] %108, 1, !dbg !20
  %797 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %798 = fmul <4 x float> %797, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %799 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %787, <4 x float> %798) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %800 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %801 = fmul <4 x float> %800, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %792, <4 x float> %801) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %803 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %803, <4 x float> %791) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %805 = shufflevector <8 x half> %775, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %805, <4 x float> %795) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %803, <4 x float> %799) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %805, <4 x float> %802) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %809 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %809, <4 x float> %804) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %811 = shufflevector <8 x half> %777, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %811, <4 x float> %806) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %809, <4 x float> %807) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %811, <4 x float> %808) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %815 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %815, <4 x float> %810) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %817 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %817, <4 x float> %812) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %815, <4 x float> %813) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %817, <4 x float> %814) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %821 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %821, <4 x float> %816) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %823 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %823, <4 x float> %818) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %821, <4 x float> %819) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %823, <4 x float> %820) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %827 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %827, <4 x float> %822) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %829 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %829, <4 x float> %824) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %827, <4 x float> %825) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %829, <4 x float> %826) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %833 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %833, <4 x float> %828) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %835 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %835, <4 x float> %830) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %833, <4 x float> %831) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %835, <4 x float> %832) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %839 = bitcast <4 x i32> %786 to <8 x half>, !dbg !25
  %840 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %840, <4 x float> %834) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %842 = shufflevector <8 x half> %839, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %842, <4 x float> %836) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %840, <4 x float> %837) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %842, <4 x float> %838) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %vdb.address1480 = or disjoint i64 %address.local921, %241, !dbg !14
  %vdb.address1481 = trunc nuw nsw i64 %vdb.address1480 to i32, !dbg !14
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !14
  %vdb.address1489 = or disjoint i64 %address.local927, %241, !dbg !14
  %vdb.address1490 = trunc nuw nsw i64 %vdb.address1489 to i32, !dbg !14
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !14
  %846 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %847 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %848 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %849 = bitcast <4 x i32> %848 to <8 x half>, !dbg !25
  %850 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %851 = bitcast <4 x i32> %850 to <8 x half>, !dbg !25
  %852 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %853 = bitcast <4 x i32> %852 to <8 x half>, !dbg !25
  %854 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %855 = bitcast <4 x i32> %854 to <8 x half>, !dbg !25
  %856 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !25
  %858 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !25
  %860 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !25
  %862 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %863 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %864 = extractvalue [2 x <8 x float>] %107, 0, !dbg !20
  %865 = shufflevector <8 x float> %864, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %866 = fmul <4 x float> %865, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %863, <4 x float> %866) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %868 = shufflevector <8 x half> %849, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %869 = shufflevector <8 x float> %864, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %870 = fmul <4 x float> %869, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %868, <4 x float> %870) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %872 = extractvalue [2 x <8 x float>] %107, 1, !dbg !20
  %873 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %874 = fmul <4 x float> %873, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %863, <4 x float> %874) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %876 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %877 = fmul <4 x float> %876, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %868, <4 x float> %877) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %879 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %879, <4 x float> %867) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %881 = shufflevector <8 x half> %851, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %881, <4 x float> %871) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %879, <4 x float> %875) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %881, <4 x float> %878) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %885 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %885, <4 x float> %880) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %887 = shufflevector <8 x half> %853, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %887, <4 x float> %882) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %885, <4 x float> %883) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %887, <4 x float> %884) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %891 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %891, <4 x float> %886) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %893 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %893, <4 x float> %888) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %891, <4 x float> %889) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %893, <4 x float> %890) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %897 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %897, <4 x float> %892) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %899 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %899, <4 x float> %894) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %897, <4 x float> %895) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %899, <4 x float> %896) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %903 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %903, <4 x float> %898) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %905 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %905, <4 x float> %900) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %903, <4 x float> %901) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %905, <4 x float> %902) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %909 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %909, <4 x float> %904) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %911 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %911, <4 x float> %906) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %909, <4 x float> %907) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %911, <4 x float> %908) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %915 = bitcast <4 x i32> %862 to <8 x half>, !dbg !25
  %916 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %916, <4 x float> %910) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %918 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %918, <4 x float> %912) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %916, <4 x float> %913) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %918, <4 x float> %914) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %922 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %923 = bitcast <4 x i32> %922 to <8 x half>, !dbg !25
  %924 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %925 = bitcast <4 x i32> %924 to <8 x half>, !dbg !25
  %926 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %927 = bitcast <4 x i32> %926 to <8 x half>, !dbg !25
  %928 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %929 = bitcast <4 x i32> %928 to <8 x half>, !dbg !25
  %930 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !25
  %932 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %933 = bitcast <4 x i32> %932 to <8 x half>, !dbg !25
  %934 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !25
  %936 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #10, !dbg !25, !frisk.v_double_buffer !26
  %937 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %938 = extractvalue [2 x <8 x float>] %106, 0, !dbg !20
  %939 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %940 = fmul <4 x float> %939, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %937, <4 x float> %940) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %942 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %943 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %944 = fmul <4 x float> %943, %466, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %942, <4 x float> %944) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %946 = extractvalue [2 x <8 x float>] %106, 1, !dbg !20
  %947 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %948 = fmul <4 x float> %947, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %937, <4 x float> %948) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20
  %950 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %951 = fmul <4 x float> %950, %474, !dbg !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %942, <4 x float> %951) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %953 = shufflevector <8 x half> %925, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %953, <4 x float> %941) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %955 = shufflevector <8 x half> %925, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %955, <4 x float> %945) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %953, <4 x float> %949) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %955, <4 x float> %952) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %959 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %959, <4 x float> %954) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %961 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %961, <4 x float> %956) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %959, <4 x float> %957) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %961, <4 x float> %958) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %965 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %965, <4 x float> %960) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %967 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %967, <4 x float> %962) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %965, <4 x float> %963) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %967, <4 x float> %964) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %971 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %971, <4 x float> %966) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %973 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %973, <4 x float> %968) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %971, <4 x float> %969) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %973, <4 x float> %970) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %977 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %977, <4 x float> %972) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %979 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %979, <4 x float> %974) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %977, <4 x float> %975) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %979, <4 x float> %976) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %983 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %983, <4 x float> %978) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %985 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %985, <4 x float> %980) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %983, <4 x float> %981) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %985, <4 x float> %982) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %989 = bitcast <4 x i32> %936 to <8 x half>, !dbg !25
  %990 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %990, <4 x float> %984) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %992 = shufflevector <8 x half> %989, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %992, <4 x float> %986) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %990, <4 x float> %987) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !25
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %992, <4 x float> %988) #8, !dbg !25, !frisk.pv_interleave !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !11
  %996 = shufflevector <4 x float> %765, <4 x float> %767, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %997 = insertvalue [2 x <8 x float>] poison, <8 x float> %996, 0, !dbg !11
  %998 = shufflevector <4 x float> %768, <4 x float> %769, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %999 = insertvalue [2 x <8 x float>] %997, <8 x float> %998, 1, !dbg !11
  %1000 = insertvalue [2 x <1 x float>] poison, <1 x float> %430, 0, !dbg !11
  %1001 = insertvalue [2 x <1 x float>] %1000, <1 x float> %443, 1, !dbg !11
  %1002 = shufflevector <4 x float> %841, <4 x float> %843, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1003 = insertvalue [2 x <8 x float>] poison, <8 x float> %1002, 0, !dbg !11
  %1004 = shufflevector <4 x float> %844, <4 x float> %845, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1005 = insertvalue [2 x <8 x float>] %1003, <8 x float> %1004, 1, !dbg !11
  %1006 = shufflevector <4 x float> %917, <4 x float> %919, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1007 = insertvalue [2 x <8 x float>] poison, <8 x float> %1006, 0, !dbg !11
  %1008 = shufflevector <4 x float> %920, <4 x float> %921, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1009 = insertvalue [2 x <8 x float>] %1007, <8 x float> %1008, 1, !dbg !11
  %1010 = shufflevector <4 x float> %991, <4 x float> %993, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1011 = insertvalue [2 x <8 x float>] poison, <8 x float> %1010, 0, !dbg !11
  %1012 = shufflevector <4 x float> %994, <4 x float> %995, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1013 = insertvalue [2 x <8 x float>] %1011, <8 x float> %1012, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1014 = add nuw nsw i64 %112, 1, !dbg !11
  %1015 = icmp samesign ult i64 %112, 31, !dbg !11
  br i1 %1015, label %105, label %1016, !dbg !11

1016:                                             ; preds = %105
  %1017 = shl nuw nsw i32 %409, 2, !dbg !11
  %1018 = xor i32 %1017, 64, !dbg !11
  %1019 = bitcast float %691 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1018, i32 %1019), !dbg !11
  %1020 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %691, %1020, !dbg !11
  %1021 = xor i32 %1017, 128, !dbg !11
  %1022 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1021, i32 %1022), !dbg !11
  %1023 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %1023, !dbg !11
  %1024 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %1025 = bitcast float %583 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1018, i32 %1025), !dbg !11
  %1026 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %583, %1026, !dbg !11
  %1027 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1021, i32 %1027), !dbg !11
  %1028 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %1028, !dbg !11
  %1029 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %1030 = and i32 %24, 16, !dbg !11
  %.not = icmp eq i32 %1030, 0, !dbg !11
  %1031 = and i32 %24, 32, !dbg !11
  %.not36 = icmp eq i32 %1031, 0, !dbg !11
  %1032 = extractelement <4 x float> %991, i64 0, !dbg !11
  %1033 = extractelement <4 x float> %991, i64 1, !dbg !11
  %1034 = extractelement <4 x float> %991, i64 2, !dbg !11
  %1035 = extractelement <4 x float> %991, i64 3, !dbg !11
  %1036 = select i1 %.not, float %1033, float %1032, !dbg !11
  %1037 = bitcast float %1036 to i32, !dbg !11
  %1038 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1037), !dbg !11
  %1039 = bitcast i32 %1038 to float, !dbg !11
  %1040 = select i1 %.not, float %1032, float %1039, !dbg !11
  %1041 = select i1 %.not, float %1039, float %1033, !dbg !11
  %1042 = select i1 %.not, float %1035, float %1034, !dbg !11
  %1043 = bitcast float %1042 to i32, !dbg !11
  %1044 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1043), !dbg !11
  %1045 = bitcast i32 %1044 to float, !dbg !11
  %1046 = select i1 %.not, float %1034, float %1045, !dbg !11
  %1047 = select i1 %.not, float %1045, float %1035, !dbg !11
  %1048 = select i1 %.not36, float %1046, float %1040, !dbg !11
  %1049 = bitcast float %1048 to i32, !dbg !11
  %1050 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1049), !dbg !11
  %1051 = bitcast i32 %1050 to float, !dbg !11
  %1052 = select i1 %.not36, float %1047, float %1041, !dbg !11
  %1053 = bitcast float %1052 to i32, !dbg !11
  %1054 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1053), !dbg !11
  %1055 = bitcast i32 %1054 to float, !dbg !11
  %1056 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1057 = shufflevector <4 x i1> %1056, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1058 = insertelement <4 x float> poison, float %1040, i64 0, !dbg !11
  %1059 = insertelement <4 x float> %1058, float %1041, i64 1, !dbg !11
  %1060 = insertelement <4 x float> %1059, float %1051, i64 2, !dbg !11
  %1061 = insertelement <4 x float> %1060, float %1055, i64 3, !dbg !11
  %1062 = insertelement <4 x float> poison, float %1051, i64 0, !dbg !11
  %1063 = insertelement <4 x float> %1062, float %1055, i64 1, !dbg !11
  %1064 = insertelement <4 x float> %1063, float %1046, i64 2, !dbg !11
  %1065 = insertelement <4 x float> %1064, float %1047, i64 3, !dbg !11
  %1066 = select <4 x i1> %1057, <4 x float> %1061, <4 x float> %1065, !dbg !11
  %1067 = extractelement <4 x float> %917, i64 0, !dbg !11
  %1068 = extractelement <4 x float> %917, i64 1, !dbg !11
  %1069 = extractelement <4 x float> %917, i64 2, !dbg !11
  %1070 = extractelement <4 x float> %917, i64 3, !dbg !11
  %1071 = select i1 %.not, float %1068, float %1067, !dbg !11
  %1072 = bitcast float %1071 to i32, !dbg !11
  %1073 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1072), !dbg !11
  %1074 = bitcast i32 %1073 to float, !dbg !11
  %1075 = select i1 %.not, float %1067, float %1074, !dbg !11
  %1076 = select i1 %.not, float %1074, float %1068, !dbg !11
  %1077 = select i1 %.not, float %1070, float %1069, !dbg !11
  %1078 = bitcast float %1077 to i32, !dbg !11
  %1079 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1078), !dbg !11
  %1080 = bitcast i32 %1079 to float, !dbg !11
  %1081 = select i1 %.not, float %1069, float %1080, !dbg !11
  %1082 = select i1 %.not, float %1080, float %1070, !dbg !11
  %1083 = select i1 %.not36, float %1081, float %1075, !dbg !11
  %1084 = bitcast float %1083 to i32, !dbg !11
  %1085 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1084), !dbg !11
  %1086 = bitcast i32 %1085 to float, !dbg !11
  %1087 = select i1 %.not36, float %1082, float %1076, !dbg !11
  %1088 = bitcast float %1087 to i32, !dbg !11
  %1089 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1088), !dbg !11
  %1090 = bitcast i32 %1089 to float, !dbg !11
  %1091 = insertelement <4 x float> poison, float %1075, i64 0, !dbg !11
  %1092 = insertelement <4 x float> %1091, float %1076, i64 1, !dbg !11
  %1093 = insertelement <4 x float> %1092, float %1086, i64 2, !dbg !11
  %1094 = insertelement <4 x float> %1093, float %1090, i64 3, !dbg !11
  %1095 = insertelement <4 x float> poison, float %1086, i64 0, !dbg !11
  %1096 = insertelement <4 x float> %1095, float %1090, i64 1, !dbg !11
  %1097 = insertelement <4 x float> %1096, float %1081, i64 2, !dbg !11
  %1098 = insertelement <4 x float> %1097, float %1082, i64 3, !dbg !11
  %1099 = select <4 x i1> %1057, <4 x float> %1094, <4 x float> %1098, !dbg !11
  %1100 = extractelement <4 x float> %841, i64 0, !dbg !11
  %1101 = extractelement <4 x float> %841, i64 1, !dbg !11
  %1102 = extractelement <4 x float> %841, i64 2, !dbg !11
  %1103 = extractelement <4 x float> %841, i64 3, !dbg !11
  %1104 = select i1 %.not, float %1101, float %1100, !dbg !11
  %1105 = bitcast float %1104 to i32, !dbg !11
  %1106 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1105), !dbg !11
  %1107 = bitcast i32 %1106 to float, !dbg !11
  %1108 = select i1 %.not, float %1100, float %1107, !dbg !11
  %1109 = select i1 %.not, float %1107, float %1101, !dbg !11
  %1110 = select i1 %.not, float %1103, float %1102, !dbg !11
  %1111 = bitcast float %1110 to i32, !dbg !11
  %1112 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1111), !dbg !11
  %1113 = bitcast i32 %1112 to float, !dbg !11
  %1114 = select i1 %.not, float %1102, float %1113, !dbg !11
  %1115 = select i1 %.not, float %1113, float %1103, !dbg !11
  %1116 = select i1 %.not36, float %1114, float %1108, !dbg !11
  %1117 = bitcast float %1116 to i32, !dbg !11
  %1118 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1117), !dbg !11
  %1119 = bitcast i32 %1118 to float, !dbg !11
  %1120 = select i1 %.not36, float %1115, float %1109, !dbg !11
  %1121 = bitcast float %1120 to i32, !dbg !11
  %1122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1121), !dbg !11
  %1123 = bitcast i32 %1122 to float, !dbg !11
  %1124 = insertelement <4 x float> poison, float %1108, i64 0, !dbg !11
  %1125 = insertelement <4 x float> %1124, float %1109, i64 1, !dbg !11
  %1126 = insertelement <4 x float> %1125, float %1119, i64 2, !dbg !11
  %1127 = insertelement <4 x float> %1126, float %1123, i64 3, !dbg !11
  %1128 = insertelement <4 x float> poison, float %1119, i64 0, !dbg !11
  %1129 = insertelement <4 x float> %1128, float %1123, i64 1, !dbg !11
  %1130 = insertelement <4 x float> %1129, float %1114, i64 2, !dbg !11
  %1131 = insertelement <4 x float> %1130, float %1115, i64 3, !dbg !11
  %1132 = select <4 x i1> %1057, <4 x float> %1127, <4 x float> %1131, !dbg !11
  %1133 = extractelement <4 x float> %765, i64 0, !dbg !11
  %1134 = extractelement <4 x float> %765, i64 1, !dbg !11
  %1135 = extractelement <4 x float> %765, i64 2, !dbg !11
  %1136 = extractelement <4 x float> %765, i64 3, !dbg !11
  %1137 = select i1 %.not, float %1134, float %1133, !dbg !11
  %1138 = bitcast float %1137 to i32, !dbg !11
  %1139 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1138), !dbg !11
  %1140 = bitcast i32 %1139 to float, !dbg !11
  %1141 = select i1 %.not, float %1133, float %1140, !dbg !11
  %1142 = select i1 %.not, float %1140, float %1134, !dbg !11
  %1143 = select i1 %.not, float %1136, float %1135, !dbg !11
  %1144 = bitcast float %1143 to i32, !dbg !11
  %1145 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1144), !dbg !11
  %1146 = bitcast i32 %1145 to float, !dbg !11
  %1147 = select i1 %.not, float %1135, float %1146, !dbg !11
  %1148 = select i1 %.not, float %1146, float %1136, !dbg !11
  %1149 = select i1 %.not36, float %1147, float %1141, !dbg !11
  %1150 = bitcast float %1149 to i32, !dbg !11
  %1151 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1150), !dbg !11
  %1152 = select i1 %.not36, float %1148, float %1142, !dbg !11
  %1153 = bitcast float %1152 to i32, !dbg !11
  %1154 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1153), !dbg !11
  %1155 = shufflevector <1 x float> %1029, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1156 = extractelement <4 x float> %993, i64 0, !dbg !11
  %1157 = extractelement <4 x float> %993, i64 1, !dbg !11
  %1158 = extractelement <4 x float> %993, i64 2, !dbg !11
  %1159 = extractelement <4 x float> %993, i64 3, !dbg !11
  %1160 = select i1 %.not, float %1157, float %1156, !dbg !11
  %1161 = bitcast float %1160 to i32, !dbg !11
  %1162 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1161), !dbg !11
  %1163 = bitcast i32 %1162 to float, !dbg !11
  %1164 = select i1 %.not, float %1156, float %1163, !dbg !11
  %1165 = select i1 %.not, float %1163, float %1157, !dbg !11
  %1166 = select i1 %.not, float %1159, float %1158, !dbg !11
  %1167 = bitcast float %1166 to i32, !dbg !11
  %1168 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1167), !dbg !11
  %1169 = bitcast i32 %1168 to float, !dbg !11
  %1170 = select i1 %.not, float %1158, float %1169, !dbg !11
  %1171 = select i1 %.not, float %1169, float %1159, !dbg !11
  %1172 = select i1 %.not36, float %1170, float %1164, !dbg !11
  %1173 = bitcast float %1172 to i32, !dbg !11
  %1174 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1173), !dbg !11
  %1175 = bitcast i32 %1174 to float, !dbg !11
  %1176 = select i1 %.not36, float %1171, float %1165, !dbg !11
  %1177 = bitcast float %1176 to i32, !dbg !11
  %1178 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1177), !dbg !11
  %1179 = bitcast i32 %1178 to float, !dbg !11
  %1180 = insertelement <4 x float> poison, float %1164, i64 0, !dbg !11
  %1181 = insertelement <4 x float> %1180, float %1165, i64 1, !dbg !11
  %1182 = insertelement <4 x float> %1181, float %1175, i64 2, !dbg !11
  %1183 = insertelement <4 x float> %1182, float %1179, i64 3, !dbg !11
  %1184 = insertelement <4 x float> poison, float %1175, i64 0, !dbg !11
  %1185 = insertelement <4 x float> %1184, float %1179, i64 1, !dbg !11
  %1186 = insertelement <4 x float> %1185, float %1170, i64 2, !dbg !11
  %1187 = insertelement <4 x float> %1186, float %1171, i64 3, !dbg !11
  %1188 = select <4 x i1> %1057, <4 x float> %1183, <4 x float> %1187, !dbg !11
  %1189 = extractelement <4 x float> %919, i64 0, !dbg !11
  %1190 = extractelement <4 x float> %919, i64 1, !dbg !11
  %1191 = extractelement <4 x float> %919, i64 2, !dbg !11
  %1192 = extractelement <4 x float> %919, i64 3, !dbg !11
  %1193 = select i1 %.not, float %1190, float %1189, !dbg !11
  %1194 = bitcast float %1193 to i32, !dbg !11
  %1195 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1194), !dbg !11
  %1196 = bitcast i32 %1195 to float, !dbg !11
  %1197 = select i1 %.not, float %1189, float %1196, !dbg !11
  %1198 = select i1 %.not, float %1196, float %1190, !dbg !11
  %1199 = select i1 %.not, float %1192, float %1191, !dbg !11
  %1200 = bitcast float %1199 to i32, !dbg !11
  %1201 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1200), !dbg !11
  %1202 = bitcast i32 %1201 to float, !dbg !11
  %1203 = select i1 %.not, float %1191, float %1202, !dbg !11
  %1204 = select i1 %.not, float %1202, float %1192, !dbg !11
  %1205 = select i1 %.not36, float %1203, float %1197, !dbg !11
  %1206 = bitcast float %1205 to i32, !dbg !11
  %1207 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1206), !dbg !11
  %1208 = bitcast i32 %1207 to float, !dbg !11
  %1209 = select i1 %.not36, float %1204, float %1198, !dbg !11
  %1210 = bitcast float %1209 to i32, !dbg !11
  %1211 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1210), !dbg !11
  %1212 = bitcast i32 %1211 to float, !dbg !11
  %1213 = insertelement <4 x float> poison, float %1197, i64 0, !dbg !11
  %1214 = insertelement <4 x float> %1213, float %1198, i64 1, !dbg !11
  %1215 = insertelement <4 x float> %1214, float %1208, i64 2, !dbg !11
  %1216 = insertelement <4 x float> %1215, float %1212, i64 3, !dbg !11
  %1217 = insertelement <4 x float> poison, float %1208, i64 0, !dbg !11
  %1218 = insertelement <4 x float> %1217, float %1212, i64 1, !dbg !11
  %1219 = insertelement <4 x float> %1218, float %1203, i64 2, !dbg !11
  %1220 = insertelement <4 x float> %1219, float %1204, i64 3, !dbg !11
  %1221 = select <4 x i1> %1057, <4 x float> %1216, <4 x float> %1220, !dbg !11
  %1222 = extractelement <4 x float> %843, i64 0, !dbg !11
  %1223 = extractelement <4 x float> %843, i64 1, !dbg !11
  %1224 = extractelement <4 x float> %843, i64 2, !dbg !11
  %1225 = extractelement <4 x float> %843, i64 3, !dbg !11
  %1226 = select i1 %.not, float %1223, float %1222, !dbg !11
  %1227 = bitcast float %1226 to i32, !dbg !11
  %1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1227), !dbg !11
  %1229 = bitcast i32 %1228 to float, !dbg !11
  %1230 = select i1 %.not, float %1222, float %1229, !dbg !11
  %1231 = select i1 %.not, float %1229, float %1223, !dbg !11
  %1232 = select i1 %.not, float %1225, float %1224, !dbg !11
  %1233 = bitcast float %1232 to i32, !dbg !11
  %1234 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1233), !dbg !11
  %1235 = bitcast i32 %1234 to float, !dbg !11
  %1236 = select i1 %.not, float %1224, float %1235, !dbg !11
  %1237 = select i1 %.not, float %1235, float %1225, !dbg !11
  %1238 = select i1 %.not36, float %1236, float %1230, !dbg !11
  %1239 = bitcast float %1238 to i32, !dbg !11
  %1240 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1239), !dbg !11
  %1241 = bitcast i32 %1240 to float, !dbg !11
  %1242 = select i1 %.not36, float %1237, float %1231, !dbg !11
  %1243 = bitcast float %1242 to i32, !dbg !11
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1243), !dbg !11
  %1245 = bitcast i32 %1244 to float, !dbg !11
  %1246 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !11
  %1247 = insertelement <4 x float> %1246, float %1231, i64 1, !dbg !11
  %1248 = insertelement <4 x float> %1247, float %1241, i64 2, !dbg !11
  %1249 = insertelement <4 x float> %1248, float %1245, i64 3, !dbg !11
  %1250 = insertelement <4 x float> poison, float %1241, i64 0, !dbg !11
  %1251 = insertelement <4 x float> %1250, float %1245, i64 1, !dbg !11
  %1252 = insertelement <4 x float> %1251, float %1236, i64 2, !dbg !11
  %1253 = insertelement <4 x float> %1252, float %1237, i64 3, !dbg !11
  %1254 = select <4 x i1> %1057, <4 x float> %1249, <4 x float> %1253, !dbg !11
  %1255 = extractelement <4 x float> %767, i64 0, !dbg !11
  %1256 = extractelement <4 x float> %767, i64 1, !dbg !11
  %1257 = extractelement <4 x float> %767, i64 2, !dbg !11
  %1258 = extractelement <4 x float> %767, i64 3, !dbg !11
  %1259 = select i1 %.not, float %1256, float %1255, !dbg !11
  %1260 = bitcast float %1259 to i32, !dbg !11
  %1261 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1260), !dbg !11
  %1262 = bitcast i32 %1261 to float, !dbg !11
  %1263 = select i1 %.not, float %1255, float %1262, !dbg !11
  %1264 = select i1 %.not, float %1262, float %1256, !dbg !11
  %1265 = select i1 %.not, float %1258, float %1257, !dbg !11
  %1266 = bitcast float %1265 to i32, !dbg !11
  %1267 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1266), !dbg !11
  %1268 = bitcast i32 %1267 to float, !dbg !11
  %1269 = select i1 %.not, float %1257, float %1268, !dbg !11
  %1270 = select i1 %.not, float %1268, float %1258, !dbg !11
  %1271 = select i1 %.not36, float %1269, float %1263, !dbg !11
  %1272 = bitcast float %1271 to i32, !dbg !11
  %1273 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1272), !dbg !11
  %1274 = select i1 %.not36, float %1270, float %1264, !dbg !11
  %1275 = bitcast float %1274 to i32, !dbg !11
  %1276 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1275), !dbg !11
  %1277 = extractelement <4 x float> %994, i64 0, !dbg !11
  %1278 = extractelement <4 x float> %994, i64 1, !dbg !11
  %1279 = extractelement <4 x float> %994, i64 2, !dbg !11
  %1280 = extractelement <4 x float> %994, i64 3, !dbg !11
  %1281 = select i1 %.not, float %1278, float %1277, !dbg !11
  %1282 = bitcast float %1281 to i32, !dbg !11
  %1283 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1282), !dbg !11
  %1284 = bitcast i32 %1283 to float, !dbg !11
  %1285 = select i1 %.not, float %1277, float %1284, !dbg !11
  %1286 = select i1 %.not, float %1284, float %1278, !dbg !11
  %1287 = select i1 %.not, float %1280, float %1279, !dbg !11
  %1288 = bitcast float %1287 to i32, !dbg !11
  %1289 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1288), !dbg !11
  %1290 = bitcast i32 %1289 to float, !dbg !11
  %1291 = select i1 %.not, float %1279, float %1290, !dbg !11
  %1292 = select i1 %.not, float %1290, float %1280, !dbg !11
  %1293 = select i1 %.not36, float %1291, float %1285, !dbg !11
  %1294 = bitcast float %1293 to i32, !dbg !11
  %1295 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1294), !dbg !11
  %1296 = bitcast i32 %1295 to float, !dbg !11
  %1297 = select i1 %.not36, float %1292, float %1286, !dbg !11
  %1298 = bitcast float %1297 to i32, !dbg !11
  %1299 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1298), !dbg !11
  %1300 = bitcast i32 %1299 to float, !dbg !11
  %1301 = insertelement <4 x float> poison, float %1285, i64 0, !dbg !11
  %1302 = insertelement <4 x float> %1301, float %1286, i64 1, !dbg !11
  %1303 = insertelement <4 x float> %1302, float %1296, i64 2, !dbg !11
  %1304 = insertelement <4 x float> %1303, float %1300, i64 3, !dbg !11
  %1305 = insertelement <4 x float> poison, float %1296, i64 0, !dbg !11
  %1306 = insertelement <4 x float> %1305, float %1300, i64 1, !dbg !11
  %1307 = insertelement <4 x float> %1306, float %1291, i64 2, !dbg !11
  %1308 = insertelement <4 x float> %1307, float %1292, i64 3, !dbg !11
  %1309 = select <4 x i1> %1057, <4 x float> %1304, <4 x float> %1308, !dbg !11
  %1310 = extractelement <4 x float> %920, i64 0, !dbg !11
  %1311 = extractelement <4 x float> %920, i64 1, !dbg !11
  %1312 = extractelement <4 x float> %920, i64 2, !dbg !11
  %1313 = extractelement <4 x float> %920, i64 3, !dbg !11
  %1314 = select i1 %.not, float %1311, float %1310, !dbg !11
  %1315 = bitcast float %1314 to i32, !dbg !11
  %1316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1315), !dbg !11
  %1317 = bitcast i32 %1316 to float, !dbg !11
  %1318 = select i1 %.not, float %1310, float %1317, !dbg !11
  %1319 = select i1 %.not, float %1317, float %1311, !dbg !11
  %1320 = select i1 %.not, float %1313, float %1312, !dbg !11
  %1321 = bitcast float %1320 to i32, !dbg !11
  %1322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1321), !dbg !11
  %1323 = bitcast i32 %1322 to float, !dbg !11
  %1324 = select i1 %.not, float %1312, float %1323, !dbg !11
  %1325 = select i1 %.not, float %1323, float %1313, !dbg !11
  %1326 = select i1 %.not36, float %1324, float %1318, !dbg !11
  %1327 = bitcast float %1326 to i32, !dbg !11
  %1328 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1327), !dbg !11
  %1329 = bitcast i32 %1328 to float, !dbg !11
  %1330 = select i1 %.not36, float %1325, float %1319, !dbg !11
  %1331 = bitcast float %1330 to i32, !dbg !11
  %1332 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1331), !dbg !11
  %1333 = bitcast i32 %1332 to float, !dbg !11
  %1334 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !11
  %1335 = insertelement <4 x float> %1334, float %1319, i64 1, !dbg !11
  %1336 = insertelement <4 x float> %1335, float %1329, i64 2, !dbg !11
  %1337 = insertelement <4 x float> %1336, float %1333, i64 3, !dbg !11
  %1338 = insertelement <4 x float> poison, float %1329, i64 0, !dbg !11
  %1339 = insertelement <4 x float> %1338, float %1333, i64 1, !dbg !11
  %1340 = insertelement <4 x float> %1339, float %1324, i64 2, !dbg !11
  %1341 = insertelement <4 x float> %1340, float %1325, i64 3, !dbg !11
  %1342 = select <4 x i1> %1057, <4 x float> %1337, <4 x float> %1341, !dbg !11
  %1343 = extractelement <4 x float> %844, i64 0, !dbg !11
  %1344 = extractelement <4 x float> %844, i64 1, !dbg !11
  %1345 = extractelement <4 x float> %844, i64 2, !dbg !11
  %1346 = extractelement <4 x float> %844, i64 3, !dbg !11
  %1347 = select i1 %.not, float %1344, float %1343, !dbg !11
  %1348 = bitcast float %1347 to i32, !dbg !11
  %1349 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1348), !dbg !11
  %1350 = bitcast i32 %1349 to float, !dbg !11
  %1351 = select i1 %.not, float %1343, float %1350, !dbg !11
  %1352 = select i1 %.not, float %1350, float %1344, !dbg !11
  %1353 = select i1 %.not, float %1346, float %1345, !dbg !11
  %1354 = bitcast float %1353 to i32, !dbg !11
  %1355 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1354), !dbg !11
  %1356 = bitcast i32 %1355 to float, !dbg !11
  %1357 = select i1 %.not, float %1345, float %1356, !dbg !11
  %1358 = select i1 %.not, float %1356, float %1346, !dbg !11
  %1359 = select i1 %.not36, float %1357, float %1351, !dbg !11
  %1360 = bitcast float %1359 to i32, !dbg !11
  %1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1360), !dbg !11
  %1362 = bitcast i32 %1361 to float, !dbg !11
  %1363 = select i1 %.not36, float %1358, float %1352, !dbg !11
  %1364 = bitcast float %1363 to i32, !dbg !11
  %1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1364), !dbg !11
  %1366 = bitcast i32 %1365 to float, !dbg !11
  %1367 = insertelement <4 x float> poison, float %1351, i64 0, !dbg !11
  %1368 = insertelement <4 x float> %1367, float %1352, i64 1, !dbg !11
  %1369 = insertelement <4 x float> %1368, float %1362, i64 2, !dbg !11
  %1370 = insertelement <4 x float> %1369, float %1366, i64 3, !dbg !11
  %1371 = insertelement <4 x float> poison, float %1362, i64 0, !dbg !11
  %1372 = insertelement <4 x float> %1371, float %1366, i64 1, !dbg !11
  %1373 = insertelement <4 x float> %1372, float %1357, i64 2, !dbg !11
  %1374 = insertelement <4 x float> %1373, float %1358, i64 3, !dbg !11
  %1375 = select <4 x i1> %1057, <4 x float> %1370, <4 x float> %1374, !dbg !11
  %1376 = extractelement <4 x float> %768, i64 0, !dbg !11
  %1377 = extractelement <4 x float> %768, i64 1, !dbg !11
  %1378 = extractelement <4 x float> %768, i64 2, !dbg !11
  %1379 = extractelement <4 x float> %768, i64 3, !dbg !11
  %1380 = select i1 %.not, float %1377, float %1376, !dbg !11
  %1381 = bitcast float %1380 to i32, !dbg !11
  %1382 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1381), !dbg !11
  %1383 = bitcast i32 %1382 to float, !dbg !11
  %1384 = select i1 %.not, float %1376, float %1383, !dbg !11
  %1385 = select i1 %.not, float %1383, float %1377, !dbg !11
  %1386 = select i1 %.not, float %1379, float %1378, !dbg !11
  %1387 = bitcast float %1386 to i32, !dbg !11
  %1388 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1387), !dbg !11
  %1389 = bitcast i32 %1388 to float, !dbg !11
  %1390 = select i1 %.not, float %1378, float %1389, !dbg !11
  %1391 = select i1 %.not, float %1389, float %1379, !dbg !11
  %1392 = select i1 %.not36, float %1390, float %1384, !dbg !11
  %1393 = bitcast float %1392 to i32, !dbg !11
  %1394 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1393), !dbg !11
  %1395 = select i1 %.not36, float %1391, float %1385, !dbg !11
  %1396 = bitcast float %1395 to i32, !dbg !11
  %1397 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1396), !dbg !11
  %1398 = shufflevector <1 x float> %1024, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %1399 = extractelement <4 x float> %995, i64 0, !dbg !11
  %1400 = extractelement <4 x float> %995, i64 1, !dbg !11
  %1401 = extractelement <4 x float> %995, i64 2, !dbg !11
  %1402 = extractelement <4 x float> %995, i64 3, !dbg !11
  %1403 = select i1 %.not, float %1400, float %1399, !dbg !11
  %1404 = bitcast float %1403 to i32, !dbg !11
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1404), !dbg !11
  %1406 = bitcast i32 %1405 to float, !dbg !11
  %1407 = select i1 %.not, float %1399, float %1406, !dbg !11
  %1408 = select i1 %.not, float %1406, float %1400, !dbg !11
  %1409 = select i1 %.not, float %1402, float %1401, !dbg !11
  %1410 = bitcast float %1409 to i32, !dbg !11
  %1411 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1410), !dbg !11
  %1412 = bitcast i32 %1411 to float, !dbg !11
  %1413 = select i1 %.not, float %1401, float %1412, !dbg !11
  %1414 = select i1 %.not, float %1412, float %1402, !dbg !11
  %1415 = select i1 %.not36, float %1413, float %1407, !dbg !11
  %1416 = bitcast float %1415 to i32, !dbg !11
  %1417 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1416), !dbg !11
  %1418 = bitcast i32 %1417 to float, !dbg !11
  %1419 = select i1 %.not36, float %1414, float %1408, !dbg !11
  %1420 = bitcast float %1419 to i32, !dbg !11
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1420), !dbg !11
  %1422 = bitcast i32 %1421 to float, !dbg !11
  %1423 = insertelement <4 x float> poison, float %1407, i64 0, !dbg !11
  %1424 = insertelement <4 x float> %1423, float %1408, i64 1, !dbg !11
  %1425 = insertelement <4 x float> %1424, float %1418, i64 2, !dbg !11
  %1426 = insertelement <4 x float> %1425, float %1422, i64 3, !dbg !11
  %1427 = insertelement <4 x float> poison, float %1418, i64 0, !dbg !11
  %1428 = insertelement <4 x float> %1427, float %1422, i64 1, !dbg !11
  %1429 = insertelement <4 x float> %1428, float %1413, i64 2, !dbg !11
  %1430 = insertelement <4 x float> %1429, float %1414, i64 3, !dbg !11
  %1431 = select <4 x i1> %1057, <4 x float> %1426, <4 x float> %1430, !dbg !11
  %1432 = extractelement <4 x float> %921, i64 0, !dbg !11
  %1433 = extractelement <4 x float> %921, i64 1, !dbg !11
  %1434 = extractelement <4 x float> %921, i64 2, !dbg !11
  %1435 = extractelement <4 x float> %921, i64 3, !dbg !11
  %1436 = select i1 %.not, float %1433, float %1432, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1437), !dbg !11
  %1439 = bitcast i32 %1438 to float, !dbg !11
  %1440 = select i1 %.not, float %1432, float %1439, !dbg !11
  %1441 = select i1 %.not, float %1439, float %1433, !dbg !11
  %1442 = select i1 %.not, float %1435, float %1434, !dbg !11
  %1443 = bitcast float %1442 to i32, !dbg !11
  %1444 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1443), !dbg !11
  %1445 = bitcast i32 %1444 to float, !dbg !11
  %1446 = select i1 %.not, float %1434, float %1445, !dbg !11
  %1447 = select i1 %.not, float %1445, float %1435, !dbg !11
  %1448 = select i1 %.not36, float %1446, float %1440, !dbg !11
  %1449 = bitcast float %1448 to i32, !dbg !11
  %1450 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1449), !dbg !11
  %1451 = bitcast i32 %1450 to float, !dbg !11
  %1452 = select i1 %.not36, float %1447, float %1441, !dbg !11
  %1453 = bitcast float %1452 to i32, !dbg !11
  %1454 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1453), !dbg !11
  %1455 = bitcast i32 %1454 to float, !dbg !11
  %1456 = insertelement <4 x float> poison, float %1440, i64 0, !dbg !11
  %1457 = insertelement <4 x float> %1456, float %1441, i64 1, !dbg !11
  %1458 = insertelement <4 x float> %1457, float %1451, i64 2, !dbg !11
  %1459 = insertelement <4 x float> %1458, float %1455, i64 3, !dbg !11
  %1460 = insertelement <4 x float> poison, float %1451, i64 0, !dbg !11
  %1461 = insertelement <4 x float> %1460, float %1455, i64 1, !dbg !11
  %1462 = insertelement <4 x float> %1461, float %1446, i64 2, !dbg !11
  %1463 = insertelement <4 x float> %1462, float %1447, i64 3, !dbg !11
  %1464 = select <4 x i1> %1057, <4 x float> %1459, <4 x float> %1463, !dbg !11
  %1465 = extractelement <4 x float> %845, i64 0, !dbg !11
  %1466 = extractelement <4 x float> %845, i64 1, !dbg !11
  %1467 = extractelement <4 x float> %845, i64 2, !dbg !11
  %1468 = extractelement <4 x float> %845, i64 3, !dbg !11
  %1469 = select i1 %.not, float %1466, float %1465, !dbg !11
  %1470 = bitcast float %1469 to i32, !dbg !11
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1470), !dbg !11
  %1472 = bitcast i32 %1471 to float, !dbg !11
  %1473 = select i1 %.not, float %1465, float %1472, !dbg !11
  %1474 = select i1 %.not, float %1472, float %1466, !dbg !11
  %1475 = select i1 %.not, float %1468, float %1467, !dbg !11
  %1476 = bitcast float %1475 to i32, !dbg !11
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1476), !dbg !11
  %1478 = bitcast i32 %1477 to float, !dbg !11
  %1479 = select i1 %.not, float %1467, float %1478, !dbg !11
  %1480 = select i1 %.not, float %1478, float %1468, !dbg !11
  %1481 = select i1 %.not36, float %1479, float %1473, !dbg !11
  %1482 = bitcast float %1481 to i32, !dbg !11
  %1483 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1482), !dbg !11
  %1484 = bitcast i32 %1483 to float, !dbg !11
  %1485 = select i1 %.not36, float %1480, float %1474, !dbg !11
  %1486 = bitcast float %1485 to i32, !dbg !11
  %1487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1486), !dbg !11
  %1488 = bitcast i32 %1487 to float, !dbg !11
  %1489 = insertelement <4 x float> poison, float %1473, i64 0, !dbg !11
  %1490 = insertelement <4 x float> %1489, float %1474, i64 1, !dbg !11
  %1491 = insertelement <4 x float> %1490, float %1484, i64 2, !dbg !11
  %1492 = insertelement <4 x float> %1491, float %1488, i64 3, !dbg !11
  %1493 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !11
  %1494 = insertelement <4 x float> %1493, float %1488, i64 1, !dbg !11
  %1495 = insertelement <4 x float> %1494, float %1479, i64 2, !dbg !11
  %1496 = insertelement <4 x float> %1495, float %1480, i64 3, !dbg !11
  %1497 = select <4 x i1> %1057, <4 x float> %1492, <4 x float> %1496, !dbg !11
  %1498 = extractelement <4 x float> %769, i64 0, !dbg !11
  %1499 = extractelement <4 x float> %769, i64 1, !dbg !11
  %1500 = extractelement <4 x float> %769, i64 2, !dbg !11
  %1501 = extractelement <4 x float> %769, i64 3, !dbg !11
  %1502 = select i1 %.not, float %1499, float %1498, !dbg !11
  %1503 = bitcast float %1502 to i32, !dbg !11
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1503), !dbg !11
  %1505 = bitcast i32 %1504 to float, !dbg !11
  %1506 = select i1 %.not, float %1498, float %1505, !dbg !11
  %1507 = select i1 %.not, float %1505, float %1499, !dbg !11
  %1508 = select i1 %.not, float %1501, float %1500, !dbg !11
  %1509 = bitcast float %1508 to i32, !dbg !11
  %1510 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %415, i32 %1509), !dbg !11
  %1511 = bitcast i32 %1510 to float, !dbg !11
  %1512 = select i1 %.not, float %1500, float %1511, !dbg !11
  %1513 = select i1 %.not, float %1511, float %1501, !dbg !11
  %1514 = select i1 %.not36, float %1512, float %1506, !dbg !11
  %1515 = bitcast float %1514 to i32, !dbg !11
  %1516 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1515), !dbg !11
  %1517 = select i1 %.not36, float %1513, float %1507, !dbg !11
  %1518 = bitcast float %1517 to i32, !dbg !11
  %1519 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %1518), !dbg !11
  %1520 = bitcast i32 %1516 to float, !dbg !11
  %1521 = bitcast i32 %1519 to float, !dbg !11
  %1522 = insertelement <4 x float> poison, float %1506, i64 0, !dbg !11
  %1523 = insertelement <4 x float> %1522, float %1507, i64 1, !dbg !11
  %1524 = insertelement <4 x float> %1523, float %1520, i64 2, !dbg !11
  %1525 = insertelement <4 x float> %1524, float %1521, i64 3, !dbg !11
  %1526 = insertelement <4 x float> poison, float %1520, i64 0, !dbg !11
  %1527 = insertelement <4 x float> %1526, float %1521, i64 1, !dbg !11
  %1528 = insertelement <4 x float> %1527, float %1512, i64 2, !dbg !11
  %1529 = insertelement <4 x float> %1528, float %1513, i64 3, !dbg !11
  %1530 = select <4 x i1> %1057, <4 x float> %1525, <4 x float> %1529, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1398, !dbg !28
  %normalized = fmul <4 x float> %row.reciprocal, %1530, !dbg !28
  %1531 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !29
  %1532 = extractelement <4 x half> %1531, i64 3, !dbg !29
  %1533 = extractelement <4 x half> %1531, i64 2, !dbg !29
  %1534 = extractelement <4 x half> %1531, i64 1, !dbg !29
  %1535 = extractelement <4 x half> %1531, i64 0, !dbg !29
  %1536 = bitcast i32 %1394 to float, !dbg !11
  %1537 = bitcast i32 %1397 to float, !dbg !11
  %1538 = insertelement <4 x float> poison, float %1384, i64 0, !dbg !11
  %1539 = insertelement <4 x float> %1538, float %1385, i64 1, !dbg !11
  %1540 = insertelement <4 x float> %1539, float %1536, i64 2, !dbg !11
  %1541 = insertelement <4 x float> %1540, float %1537, i64 3, !dbg !11
  %1542 = insertelement <4 x float> poison, float %1536, i64 0, !dbg !11
  %1543 = insertelement <4 x float> %1542, float %1537, i64 1, !dbg !11
  %1544 = insertelement <4 x float> %1543, float %1390, i64 2, !dbg !11
  %1545 = insertelement <4 x float> %1544, float %1391, i64 3, !dbg !11
  %1546 = select <4 x i1> %1057, <4 x float> %1541, <4 x float> %1545, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %1546, !dbg !28
  %1547 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !29
  %1548 = extractelement <4 x half> %1547, i64 3, !dbg !29
  %1549 = extractelement <4 x half> %1547, i64 2, !dbg !29
  %1550 = extractelement <4 x half> %1547, i64 1, !dbg !29
  %1551 = extractelement <4 x half> %1547, i64 0, !dbg !29
  %1552 = bitcast i32 %1273 to float, !dbg !11
  %1553 = bitcast i32 %1276 to float, !dbg !11
  %1554 = insertelement <4 x float> poison, float %1263, i64 0, !dbg !11
  %1555 = insertelement <4 x float> %1554, float %1264, i64 1, !dbg !11
  %1556 = insertelement <4 x float> %1555, float %1552, i64 2, !dbg !11
  %1557 = insertelement <4 x float> %1556, float %1553, i64 3, !dbg !11
  %1558 = insertelement <4 x float> poison, float %1552, i64 0, !dbg !11
  %1559 = insertelement <4 x float> %1558, float %1553, i64 1, !dbg !11
  %1560 = insertelement <4 x float> %1559, float %1269, i64 2, !dbg !11
  %1561 = insertelement <4 x float> %1560, float %1270, i64 3, !dbg !11
  %1562 = select <4 x i1> %1057, <4 x float> %1557, <4 x float> %1561, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1155, !dbg !28
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1562, !dbg !28
  %1563 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !29
  %1564 = extractelement <4 x half> %1563, i64 3, !dbg !29
  %1565 = extractelement <4 x half> %1563, i64 2, !dbg !29
  %1566 = extractelement <4 x half> %1563, i64 1, !dbg !29
  %1567 = extractelement <4 x half> %1563, i64 0, !dbg !29
  %1568 = bitcast i32 %1151 to float, !dbg !11
  %1569 = bitcast i32 %1154 to float, !dbg !11
  %1570 = insertelement <4 x float> poison, float %1141, i64 0, !dbg !11
  %1571 = insertelement <4 x float> %1570, float %1142, i64 1, !dbg !11
  %1572 = insertelement <4 x float> %1571, float %1568, i64 2, !dbg !11
  %1573 = insertelement <4 x float> %1572, float %1569, i64 3, !dbg !11
  %1574 = insertelement <4 x float> poison, float %1568, i64 0, !dbg !11
  %1575 = insertelement <4 x float> %1574, float %1569, i64 1, !dbg !11
  %1576 = insertelement <4 x float> %1575, float %1147, i64 2, !dbg !11
  %1577 = insertelement <4 x float> %1576, float %1148, i64 3, !dbg !11
  %1578 = select <4 x i1> %1057, <4 x float> %1573, <4 x float> %1577, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1578, !dbg !28
  %1579 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !29
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !29
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !29
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !29
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !29
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %24) #8, !dbg !30, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %1584 = lshr i64 %address.local975, 2, !dbg !30
  %address.local978 = and i64 %1584, 48, !dbg !30
  %address.local980 = and i64 %address.local975, 240, !dbg !30
  %address.local979 = add i64 %19, %address.local975, !dbg !30
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !30
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !30
  %1585 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %21, !dbg !30
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !30
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %1585, i64 %address.local985.idx, !dbg !30
  %address.local987 = and i64 %1584, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !30
  store half %1583, ptr addrspace(1) %address.local988, align 2, !dbg !30
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !30
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !30
  store half %1582, ptr addrspace(1) %address.local991, align 2, !dbg !30
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !30
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !30
  store half %1581, ptr addrspace(1) %address.local994, align 2, !dbg !30
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !30
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !30
  store half %1580, ptr addrspace(1) %address.local997, align 2, !dbg !30
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !30
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !30
  store half %1567, ptr addrspace(1) %address.local1000, align 2, !dbg !30
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !30
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !30
  store half %1566, ptr addrspace(1) %address.local1003, align 2, !dbg !30
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !30
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !30
  store half %1565, ptr addrspace(1) %address.local1006, align 2, !dbg !30
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !30
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !30
  store half %1564, ptr addrspace(1) %address.local1009, align 2, !dbg !30
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !30
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !30
  store half %1551, ptr addrspace(1) %address.local1014, align 2, !dbg !30
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !30
  store half %1550, ptr addrspace(1) %address.local1016, align 2, !dbg !30
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !30
  store half %1549, ptr addrspace(1) %address.local1018, align 2, !dbg !30
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !30
  store half %1548, ptr addrspace(1) %address.local1020, align 2, !dbg !30
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !30
  store half %1535, ptr addrspace(1) %address.local1022, align 2, !dbg !30
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !30
  store half %1534, ptr addrspace(1) %address.local1024, align 2, !dbg !30
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !30
  store half %1533, ptr addrspace(1) %address.local1026, align 2, !dbg !30
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !30
  store half %1532, ptr addrspace(1) %address.local1028, align 2, !dbg !30
  %normalized42 = fmul <4 x float> %row.reciprocal, %1497, !dbg !28
  %1586 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !29
  %1587 = extractelement <4 x half> %1586, i64 3, !dbg !29
  %1588 = extractelement <4 x half> %1586, i64 2, !dbg !29
  %1589 = extractelement <4 x half> %1586, i64 1, !dbg !29
  %1590 = extractelement <4 x half> %1586, i64 0, !dbg !29
  %normalized43 = fmul <4 x float> %row.reciprocal, %1375, !dbg !28
  %1591 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !29
  %1592 = extractelement <4 x half> %1591, i64 3, !dbg !29
  %1593 = extractelement <4 x half> %1591, i64 2, !dbg !29
  %1594 = extractelement <4 x half> %1591, i64 1, !dbg !29
  %1595 = extractelement <4 x half> %1591, i64 0, !dbg !29
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1254, !dbg !28
  %1596 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !29
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !29
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !29
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !29
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !29
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1132, !dbg !28
  %1601 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !29
  %1602 = extractelement <4 x half> %1601, i64 3, !dbg !29
  %1603 = extractelement <4 x half> %1601, i64 2, !dbg !29
  %1604 = extractelement <4 x half> %1601, i64 1, !dbg !29
  %1605 = extractelement <4 x half> %1601, i64 0, !dbg !29
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !30
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !30
  store half %1605, ptr addrspace(1) %address.local1044, align 2, !dbg !30
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !30
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !30
  store half %1604, ptr addrspace(1) %address.local1047, align 2, !dbg !30
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !30
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !30
  store half %1603, ptr addrspace(1) %address.local1050, align 2, !dbg !30
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !30
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !30
  store half %1602, ptr addrspace(1) %address.local1053, align 2, !dbg !30
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !30
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !30
  store half %1600, ptr addrspace(1) %address.local1056, align 2, !dbg !30
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !30
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !30
  store half %1599, ptr addrspace(1) %address.local1059, align 2, !dbg !30
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !30
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !30
  store half %1598, ptr addrspace(1) %address.local1062, align 2, !dbg !30
  %address.local1064 = or i64 %1584, 51, !dbg !30
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !30
  store half %1597, ptr addrspace(1) %address.local1065, align 2, !dbg !30
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !30
  store half %1595, ptr addrspace(1) %address.local1070, align 2, !dbg !30
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !30
  store half %1594, ptr addrspace(1) %address.local1072, align 2, !dbg !30
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !30
  store half %1593, ptr addrspace(1) %address.local1074, align 2, !dbg !30
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !30
  store half %1592, ptr addrspace(1) %address.local1076, align 2, !dbg !30
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !30
  store half %1590, ptr addrspace(1) %address.local1078, align 2, !dbg !30
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !30
  store half %1589, ptr addrspace(1) %address.local1080, align 2, !dbg !30
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !30
  store half %1588, ptr addrspace(1) %address.local1082, align 2, !dbg !30
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !30
  store half %1587, ptr addrspace(1) %address.local1084, align 2, !dbg !30
  %normalized48 = fmul <4 x float> %row.reciprocal, %1464, !dbg !28
  %1606 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !29
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !29
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !29
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !29
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !29
  %normalized49 = fmul <4 x float> %row.reciprocal, %1342, !dbg !28
  %1611 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !29
  %1612 = extractelement <4 x half> %1611, i64 3, !dbg !29
  %1613 = extractelement <4 x half> %1611, i64 2, !dbg !29
  %1614 = extractelement <4 x half> %1611, i64 1, !dbg !29
  %1615 = extractelement <4 x half> %1611, i64 0, !dbg !29
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1221, !dbg !28
  %1616 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !29
  %1617 = extractelement <4 x half> %1616, i64 3, !dbg !29
  %1618 = extractelement <4 x half> %1616, i64 2, !dbg !29
  %1619 = extractelement <4 x half> %1616, i64 1, !dbg !29
  %1620 = extractelement <4 x half> %1616, i64 0, !dbg !29
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1099, !dbg !28
  %1621 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !29
  %1622 = extractelement <4 x half> %1621, i64 3, !dbg !29
  %1623 = extractelement <4 x half> %1621, i64 2, !dbg !29
  %1624 = extractelement <4 x half> %1621, i64 1, !dbg !29
  %1625 = extractelement <4 x half> %1621, i64 0, !dbg !29
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !30
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !30
  store half %1625, ptr addrspace(1) %address.local1100, align 2, !dbg !30
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !30
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !30
  store half %1624, ptr addrspace(1) %address.local1103, align 2, !dbg !30
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !30
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !30
  store half %1623, ptr addrspace(1) %address.local1106, align 2, !dbg !30
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !30
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !30
  store half %1622, ptr addrspace(1) %address.local1109, align 2, !dbg !30
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !30
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !30
  store half %1620, ptr addrspace(1) %address.local1112, align 2, !dbg !30
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !30
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !30
  store half %1619, ptr addrspace(1) %address.local1115, align 2, !dbg !30
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !30
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !30
  store half %1618, ptr addrspace(1) %address.local1118, align 2, !dbg !30
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !30
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !30
  store half %1617, ptr addrspace(1) %address.local1121, align 2, !dbg !30
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !30
  store half %1615, ptr addrspace(1) %address.local1126, align 2, !dbg !30
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !30
  store half %1614, ptr addrspace(1) %address.local1128, align 2, !dbg !30
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !30
  store half %1613, ptr addrspace(1) %address.local1130, align 2, !dbg !30
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !30
  store half %1612, ptr addrspace(1) %address.local1132, align 2, !dbg !30
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !30
  store half %1610, ptr addrspace(1) %address.local1134, align 2, !dbg !30
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !30
  store half %1609, ptr addrspace(1) %address.local1136, align 2, !dbg !30
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !30
  store half %1608, ptr addrspace(1) %address.local1138, align 2, !dbg !30
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !30
  store half %1607, ptr addrspace(1) %address.local1140, align 2, !dbg !30
  %normalized54 = fmul <4 x float> %row.reciprocal, %1431, !dbg !28
  %1626 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !29
  %1627 = extractelement <4 x half> %1626, i64 3, !dbg !29
  %1628 = extractelement <4 x half> %1626, i64 2, !dbg !29
  %1629 = extractelement <4 x half> %1626, i64 1, !dbg !29
  %1630 = extractelement <4 x half> %1626, i64 0, !dbg !29
  %normalized55 = fmul <4 x float> %row.reciprocal, %1309, !dbg !28
  %1631 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !29
  %1632 = extractelement <4 x half> %1631, i64 3, !dbg !29
  %1633 = extractelement <4 x half> %1631, i64 2, !dbg !29
  %1634 = extractelement <4 x half> %1631, i64 1, !dbg !29
  %1635 = extractelement <4 x half> %1631, i64 0, !dbg !29
  %normalized57 = fmul <4 x float> %row.reciprocal38, %1188, !dbg !28
  %1636 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !29
  %1637 = extractelement <4 x half> %1636, i64 3, !dbg !29
  %1638 = extractelement <4 x half> %1636, i64 2, !dbg !29
  %1639 = extractelement <4 x half> %1636, i64 1, !dbg !29
  %1640 = extractelement <4 x half> %1636, i64 0, !dbg !29
  %normalized58 = fmul <4 x float> %row.reciprocal38, %1066, !dbg !28
  %1641 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !29
  %1642 = extractelement <4 x half> %1641, i64 3, !dbg !29
  %1643 = extractelement <4 x half> %1641, i64 2, !dbg !29
  %1644 = extractelement <4 x half> %1641, i64 1, !dbg !29
  %1645 = extractelement <4 x half> %1641, i64 0, !dbg !29
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !30
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !30
  store half %1645, ptr addrspace(1) %address.local1156, align 2, !dbg !30
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !30
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !30
  store half %1644, ptr addrspace(1) %address.local1159, align 2, !dbg !30
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !30
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !30
  store half %1643, ptr addrspace(1) %address.local1162, align 2, !dbg !30
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !30
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !30
  store half %1642, ptr addrspace(1) %address.local1165, align 2, !dbg !30
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !30
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !30
  store half %1640, ptr addrspace(1) %address.local1168, align 2, !dbg !30
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !30
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !30
  store half %1639, ptr addrspace(1) %address.local1171, align 2, !dbg !30
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !30
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !30
  store half %1638, ptr addrspace(1) %address.local1174, align 2, !dbg !30
  %address.local1176 = or i64 %1584, 115, !dbg !30
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !30
  store half %1637, ptr addrspace(1) %address.local1177, align 2, !dbg !30
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !30
  store half %1635, ptr addrspace(1) %address.local1182, align 2, !dbg !30
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !30
  store half %1634, ptr addrspace(1) %address.local1184, align 2, !dbg !30
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !30
  store half %1633, ptr addrspace(1) %address.local1186, align 2, !dbg !30
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !30
  store half %1632, ptr addrspace(1) %address.local1188, align 2, !dbg !30
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !30
  store half %1630, ptr addrspace(1) %address.local1190, align 2, !dbg !30
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !30
  store half %1629, ptr addrspace(1) %address.local1192, align 2, !dbg !30
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !30
  store half %1628, ptr addrspace(1) %address.local1194, align 2, !dbg !30
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !30
  store half %1627, ptr addrspace(1) %address.local1196, align 2, !dbg !30
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { convergent willreturn memory(read) }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input_spidD_noncasual.mlir", directory: "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 67, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 69, column: 24, scope: !3)
!11 = !DILocation(line: 57, column: 24, scope: !3)
!12 = !DILocation(line: 70, column: 26, scope: !3)
!13 = !DILocation(line: 68, column: 22, scope: !3)
!14 = !DILocation(line: 86, column: 20, scope: !3)
!15 = !DILocation(line: 71, column: 23, scope: !3)
!16 = !DILocation(line: 74, column: 20, scope: !3)
!17 = !DILocation(line: 75, column: 22, scope: !3)
!18 = !DILocation(line: 76, column: 23, scope: !3)
!19 = !DILocation(line: 77, column: 20, scope: !3)
!20 = !DILocation(line: 88, column: 22, scope: !3)
!21 = !DILocation(line: 78, column: 22, scope: !3)
!22 = !DILocation(line: 79, column: 22, scope: !3)
!23 = !DILocation(line: 81, column: 16, scope: !3)
!24 = !DILocation(line: 80, column: 20, scope: !3)
!25 = !DILocation(line: 87, column: 23, scope: !3)
!26 = !{i64 8192}
!27 = !{i32 4}
!28 = !DILocation(line: 92, column: 23, scope: !3)
!29 = !DILocation(line: 93, column: 19, scope: !3)
!30 = !DILocation(line: 94, column: 9, scope: !3)
