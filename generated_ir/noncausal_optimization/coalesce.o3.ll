; ModuleID = 'generated_ir/noncausal_optimization/coalesce/run-20261027-164012-mybq6yab/online-baseline/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/coalesce/run-20261027-164012-mybq6yab/online-baseline/input.mlir"

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
  %12 = and i64 %11, 576460752303423484, !dbg !7
  %13 = and i64 %6, 3, !dbg !7
  %14 = or disjoint i64 %12, %13, !dbg !7
  %15 = sub nsw i64 31, %14, !dbg !7
  %16 = lshr i64 %10, 2, !dbg !7
  %17 = icmp ugt i64 %10, 1023, !dbg !8
  %18 = add nsw i64 %14, -32, !dbg !8
  %19 = select i1 %17, i64 %18, i64 %15, !dbg !8
  %20 = sdiv i64 %19, 32, !dbg !8
  %21 = sext i1 %17 to i64, !dbg !8
  %22 = xor i64 %20, %21, !dbg !8
  %23 = add nsw i64 %22, %16, !dbg !8
  %24 = shl nsw i64 %15, 7, !dbg !8
  %25 = shl i64 %22, 12, !dbg !8
  %26 = sub i64 %24, %25, !dbg !8
  %27 = shl i64 %23, 19, !dbg !8
  %28 = and i64 %27, 16252928, !dbg !8
  %29 = shl nsw i64 %26, 7, !dbg !8
  %30 = add i64 %28, %29, !dbg !8
  %31 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8, !range !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %32 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %30, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %33 = insertelement <2 x i64> poison, i64 %32, i64 0, !dbg !8
  %34 = bitcast <2 x i64> %33 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %34, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %30, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %35 = or disjoint i64 %30, 32, !dbg !8
  %36 = or disjoint i64 %address.local413, %35, !dbg !8
  %.tr = trunc i64 %36 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %37 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> %37, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %39 = insertelement <4 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> %39, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %41 = ptrtoint <4 x ptr addrspace(3)> %40 to <4 x i64>, !dbg !8
  %42 = trunc <4 x i64> %41 to <4 x i32>, !dbg !8
  %43 = extractelement <4 x i32> %42, i64 0, !dbg !8
  %44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %45 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %46 = extractelement <4 x i32> %42, i64 1, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %49 = extractelement <4 x i32> %42, i64 2, !dbg !8
  %50 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %49), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %51 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %52 = extractelement <4 x i32> %42, i64 3, !dbg !8
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !8
  %54 = or disjoint i64 %address.local425, %35, !dbg !8
  %.tr1401 = trunc i64 %54 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
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
  %56 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %58 = fmul <4 x half> %56, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %60 = fmul <4 x half> %57, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %62 = fmul <4 x half> %59, splat (half 1.275630e-01), !dbg !10
  %63 = fmul <4 x half> %61, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %64 = or disjoint i64 %30, 64, !dbg !8
  %65 = or disjoint i64 %address.local413, %64, !dbg !8
  %.tr1402 = trunc i64 %65 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %66 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %67 = or disjoint i64 %address.local425, %64, !dbg !8
  %.tr1403 = trunc i64 %67 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %68 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %74 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !10
  %75 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !10
  %76 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %77 = or disjoint i64 %30, 96, !dbg !8
  %78 = or disjoint i64 %address.local413, %77, !dbg !8
  %.tr1404 = trunc i64 %78 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %79 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %80 = or disjoint i64 %address.local425, %77, !dbg !8
  %.tr1405 = trunc i64 %80 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %82 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %87 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !10
  %88 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %89 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %93 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %95 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !10
  %96 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  %97 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %98 = shl nuw nsw i32 %31, 2, !dbg !12
  %99 = and i32 %98, 12, !dbg !12
  %100 = and i32 %31, 1, !dbg !12
  %101 = add nuw nsw i32 %100, 1, !dbg !12
  %narrow = or disjoint i32 %101, %99, !dbg !12
  %102 = lshr i32 %narrow, 1, !dbg !12
  %103 = zext nneg i32 %102 to i64, !dbg !12
  %104 = shl i64 %16, 19
  %105 = and i64 %104, 16252928
  %106 = ptrtoint ptr addrspace(1) %1 to i64
  %107 = insertelement <2 x i64> poison, i64 %106, i64 0
  %108 = bitcast <2 x i64> %107 to <4 x i32>
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %108, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %109 = ptrtoint ptr addrspace(1) %2 to i64
  %110 = insertelement <2 x i64> poison, i64 %109, i64 0
  %111 = bitcast <2 x i64> %110 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %111, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %112, !dbg !11

112:                                              ; preds = %4, %112
  %113 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1020, %112 ]
  %114 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1016, %112 ]
  %115 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1012, %112 ]
  %116 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1008, %112 ]
  %117 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %700, %112 ]
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1006, %112 ]
  %119 = phi i64 [ 0, %4 ], [ %1021, %112 ]
  %120 = shl nuw nsw i64 %119, 14, !dbg !13
  %121 = add nuw nsw i64 %120, %105, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
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
  %.idx1406 = or disjoint i64 %address.local514, %121, !dbg !13
  %.idx1406.tr = trunc nuw nsw i64 %.idx1406 to i32, !dbg !13
  %dma.buffer.offset1372 = shl nuw nsw i32 %.idx1406.tr, 1, !dbg !13
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local522 = lshr i64 %address.local516, 4, !dbg !13
  %address.local523 = and i64 %address.local522, 24, !dbg !13
  %address.local531 = or disjoint i64 %address.local513, %address.local523, !dbg !13
  %.idx1407 = or disjoint i64 %address.local531, %121, !dbg !13
  %.idx1407.tr = trunc nuw nsw i64 %.idx1407 to i32, !dbg !13
  %dma.buffer.offset1369 = shl nuw nsw i32 %.idx1407.tr, 1, !dbg !13
  %122 = or disjoint i64 %121, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %.idx1408 = or disjoint i64 %address.local514, %122, !dbg !13
  %.idx1408.tr = trunc nuw nsw i64 %.idx1408 to i32, !dbg !13
  %dma.buffer.offset1366 = shl nuw nsw i32 %.idx1408.tr, 1, !dbg !13
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %123 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0, !dbg !8
  %124 = insertelement <4 x ptr addrspace(3)> %123, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %125 = insertelement <4 x ptr addrspace(3)> %124, ptr addrspace(3) %address.local533, i64 2, !dbg !8
  %126 = insertelement <4 x ptr addrspace(3)> %125, ptr addrspace(3) %address.local537, i64 3, !dbg !8
  %127 = ptrtoint <4 x ptr addrspace(3)> %126 to <4 x i64>, !dbg !8
  %128 = trunc <4 x i64> %127 to <4 x i32>, !dbg !8
  %129 = extractelement <4 x i32> %128, i64 0, !dbg !13
  %130 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %129), !dbg !13
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %130), !dbg !13
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1372, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %132 = extractelement <4 x i32> %128, i64 1, !dbg !13
  %133 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !13
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !13
  %134 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %135 = extractelement <4 x i32> %128, i64 2, !dbg !13
  %136 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %136), !dbg !13
  %137 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1366, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %138 = extractelement <4 x i32> %128, i64 3, !dbg !13
  %139 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %138), !dbg !13
  %.idx1409 = or disjoint i64 %address.local531, %122, !dbg !13
  %.idx1409.tr = trunc nuw nsw i64 %.idx1409 to i32, !dbg !13
  %dma.buffer.offset1363 = shl nuw nsw i32 %.idx1409.tr, 1, !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %139), !dbg !13
  %140 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
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
  %141 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local508, -1280, !dbg !12
  %142 = getelementptr i8, ptr addrspace(3) %141, i64 %.idx1410, !dbg !12
  %143 = getelementptr [2 x i8], ptr addrspace(3) %142, i64 %address.local546, !dbg !12
  %144 = getelementptr [2 x i8], ptr addrspace(3) %143, i64 %address.local550, !dbg !12
  %145 = getelementptr [2 x i8], ptr addrspace(3) %144, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %145, i64 %address.local557, !dbg !12
  %146 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1210 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4096, !dbg !12
  %148 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %103) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %149 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %150 = getelementptr [2 x i8], ptr addrspace(3) %149, i64 %address.local553, !dbg !12
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %151, i64 64, !dbg !12
  %152 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %148, <4 x float> %147) #8, !dbg !12
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1211 = getelementptr i8, ptr addrspace(3) %151, i64 4160, !dbg !12
  %155 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %lds.const.gep1212 = getelementptr i8, ptr addrspace(3) %address.local559, i64 128, !dbg !12
  %156 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %155, <4 x float> %154) #8, !dbg !12
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %156, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1213 = getelementptr i8, ptr addrspace(3) %address.local559, i64 4224, !dbg !12
  %159 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %lds.const.gep1214 = getelementptr i8, ptr addrspace(3) %151, i64 192, !dbg !12
  %160 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %159, <4 x float> %158) #8, !dbg !12
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %151, i64 4288, !dbg !12
  %163 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1024, !dbg !12
  %164 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %163, <4 x float> %162) #8, !dbg !12
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %164, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5120, !dbg !12
  %167 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %151, i64 1088, !dbg !12
  %168 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %167, <4 x float> %166) #8, !dbg !12
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %151, i64 5184, !dbg !12
  %171 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local559, i64 1152, !dbg !12
  %172 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %171, <4 x float> %170) #8, !dbg !12
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %172, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local559, i64 5248, !dbg !12
  %175 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %lds.const.gep1222 = getelementptr i8, ptr addrspace(3) %151, i64 1216, !dbg !12
  %176 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %175, <4 x float> %174) #8, !dbg !12
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %176, <4 x float> zeroinitializer) #8, !dbg !12
  %lds.const.gep1223 = getelementptr i8, ptr addrspace(3) %151, i64 5312, !dbg !12
  %179 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %179, <4 x float> %178) #8, !dbg !12
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !12
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %148, <4 x float> %181) #8, !dbg !12
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !12
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %155, <4 x float> %183) #8, !dbg !12
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %156, <4 x float> zeroinitializer) #8, !dbg !12
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %159, <4 x float> %185) #8, !dbg !12
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !12
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %163, <4 x float> %187) #8, !dbg !12
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %164, <4 x float> zeroinitializer) #8, !dbg !12
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %167, <4 x float> %189) #8, !dbg !12
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !12
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %171, <4 x float> %191) #8, !dbg !12
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %172, <4 x float> zeroinitializer) #8, !dbg !12
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %175, <4 x float> %193) #8, !dbg !12
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %176, <4 x float> zeroinitializer) #8, !dbg !12
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %179, <4 x float> %195) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %197 = or disjoint i64 %121, 64, !dbg !13
  %.idx14111412 = or disjoint i64 %address.local514, %197, !dbg !13
  %.idx14111412.tr = trunc nuw nsw i64 %.idx14111412 to i32, !dbg !13
  %dma.buffer.offset1360 = shl nuw nsw i32 %.idx14111412.tr, 1, !dbg !13
  %198 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14131414 = or disjoint i64 %address.local531, %197, !dbg !13
  %.idx14131414.tr = trunc nuw nsw i64 %.idx14131414 to i32, !dbg !13
  %dma.buffer.offset1357 = shl nuw nsw i32 %.idx14131414.tr, 1, !dbg !13
  %199 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %lds.const.gep1238 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8192, !dbg !12
  %200 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %200, <4 x float> %153) #8, !dbg !12
  %lds.const.gep1239 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12288, !dbg !12
  %202 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %lds.const.gep1240 = getelementptr i8, ptr addrspace(3) %151, i64 8256, !dbg !12
  %203 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %202, <4 x float> %201) #8, !dbg !12
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %203, <4 x float> %157) #8, !dbg !12
  %lds.const.gep1241 = getelementptr i8, ptr addrspace(3) %151, i64 12352, !dbg !12
  %206 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %lds.const.gep1242 = getelementptr i8, ptr addrspace(3) %address.local559, i64 8320, !dbg !12
  %207 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %206, <4 x float> %205) #8, !dbg !12
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %207, <4 x float> %161) #8, !dbg !12
  %lds.const.gep1243 = getelementptr i8, ptr addrspace(3) %address.local559, i64 12416, !dbg !12
  %210 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %lds.const.gep1244 = getelementptr i8, ptr addrspace(3) %151, i64 8384, !dbg !12
  %211 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %210, <4 x float> %209) #8, !dbg !12
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %211, <4 x float> %165) #8, !dbg !12
  %lds.const.gep1245 = getelementptr i8, ptr addrspace(3) %151, i64 12480, !dbg !12
  %214 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %lds.const.gep1246 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9216, !dbg !12
  %215 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %214, <4 x float> %213) #8, !dbg !12
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %215, <4 x float> %169) #8, !dbg !12
  %lds.const.gep1247 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13312, !dbg !12
  %218 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %lds.const.gep1248 = getelementptr i8, ptr addrspace(3) %151, i64 9280, !dbg !12
  %219 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %218, <4 x float> %217) #8, !dbg !12
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %219, <4 x float> %173) #8, !dbg !12
  %lds.const.gep1249 = getelementptr i8, ptr addrspace(3) %151, i64 13376, !dbg !12
  %222 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %lds.const.gep1250 = getelementptr i8, ptr addrspace(3) %address.local559, i64 9344, !dbg !12
  %223 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %222, <4 x float> %221) #8, !dbg !12
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %223, <4 x float> %177) #8, !dbg !12
  %lds.const.gep1251 = getelementptr i8, ptr addrspace(3) %address.local559, i64 13440, !dbg !12
  %226 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %lds.const.gep1252 = getelementptr i8, ptr addrspace(3) %151, i64 9408, !dbg !12
  %227 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %226, <4 x float> %225) #8, !dbg !12
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %227, <4 x float> %180) #8, !dbg !12
  %lds.const.gep1253 = getelementptr i8, ptr addrspace(3) %151, i64 13504, !dbg !12
  %230 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %230, <4 x float> %229) #8, !dbg !12
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %200, <4 x float> %182) #8, !dbg !12
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %202, <4 x float> %232) #8, !dbg !12
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %203, <4 x float> %184) #8, !dbg !12
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %206, <4 x float> %234) #8, !dbg !12
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %207, <4 x float> %186) #8, !dbg !12
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %210, <4 x float> %236) #8, !dbg !12
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %211, <4 x float> %188) #8, !dbg !12
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %214, <4 x float> %238) #8, !dbg !12
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %215, <4 x float> %190) #8, !dbg !12
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %218, <4 x float> %240) #8, !dbg !12
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %219, <4 x float> %192) #8, !dbg !12
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %222, <4 x float> %242) #8, !dbg !12
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %223, <4 x float> %194) #8, !dbg !12
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %226, <4 x float> %244) #8, !dbg !12
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %227, <4 x float> %196) #8, !dbg !12
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %230, <4 x float> %246) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %248 = or disjoint i64 %121, 96, !dbg !13
  %.idx14151416 = or disjoint i64 %address.local514, %248, !dbg !13
  %.idx14151416.tr = trunc nuw nsw i64 %.idx14151416 to i32, !dbg !13
  %dma.buffer.offset1354 = shl nuw nsw i32 %.idx14151416.tr, 1, !dbg !13
  %249 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %.idx14171418 = or disjoint i64 %address.local531, %248, !dbg !13
  %.idx14171418.tr = trunc nuw nsw i64 %.idx14171418 to i32, !dbg !13
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14171418.tr, 1, !dbg !13
  %250 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1373) #9, !dbg !13
  %251 = load <4 x half>, ptr addrspace(3) %address.local559, align 8, !dbg !12
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %251, <4 x float> %204) #8, !dbg !12
  %253 = load <4 x half>, ptr addrspace(3) %lds.const.gep1210, align 8, !dbg !12
  %254 = load <4 x half>, ptr addrspace(3) %address.local588, align 8, !dbg !12
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %253, <4 x float> %252) #8, !dbg !12
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %254, <4 x float> %208) #8, !dbg !12
  %257 = load <4 x half>, ptr addrspace(3) %lds.const.gep1211, align 8, !dbg !12
  %258 = load <4 x half>, ptr addrspace(3) %lds.const.gep1212, align 8, !dbg !12
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %257, <4 x float> %256) #8, !dbg !12
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %258, <4 x float> %212) #8, !dbg !12
  %261 = load <4 x half>, ptr addrspace(3) %lds.const.gep1213, align 8, !dbg !12
  %262 = load <4 x half>, ptr addrspace(3) %lds.const.gep1214, align 8, !dbg !12
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %261, <4 x float> %260) #8, !dbg !12
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %262, <4 x float> %216) #8, !dbg !12
  %265 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %266 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %265, <4 x float> %264) #8, !dbg !12
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %266, <4 x float> %220) #8, !dbg !12
  %269 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %270 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %269, <4 x float> %268) #8, !dbg !12
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %270, <4 x float> %224) #8, !dbg !12
  %273 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %274 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %273, <4 x float> %272) #8, !dbg !12
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %274, <4 x float> %228) #8, !dbg !12
  %277 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %278 = load <4 x half>, ptr addrspace(3) %lds.const.gep1222, align 8, !dbg !12
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %277, <4 x float> %276) #8, !dbg !12
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %278, <4 x float> %231) #8, !dbg !12
  %281 = load <4 x half>, ptr addrspace(3) %lds.const.gep1223, align 8, !dbg !12
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %281, <4 x float> %280) #8, !dbg !12
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %251, <4 x float> %233) #8, !dbg !12
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %253, <4 x float> %283) #8, !dbg !12
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %254, <4 x float> %235) #8, !dbg !12
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %257, <4 x float> %285) #8, !dbg !12
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %258, <4 x float> %237) #8, !dbg !12
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %261, <4 x float> %287) #8, !dbg !12
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %262, <4 x float> %239) #8, !dbg !12
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %265, <4 x float> %289) #8, !dbg !12
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %266, <4 x float> %241) #8, !dbg !12
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %269, <4 x float> %291) #8, !dbg !12
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %270, <4 x float> %243) #8, !dbg !12
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %273, <4 x float> %293) #8, !dbg !12
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %274, <4 x float> %245) #8, !dbg !12
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %277, <4 x float> %295) #8, !dbg !12
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %278, <4 x float> %247) #8, !dbg !12
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %281, <4 x float> %297) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !14
  %address.local918 = and i64 %address.local917, 8064, !dbg !14
  %address.local920 = and i64 %address.local500, 24, !dbg !14
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !14
  %.idx14191420 = or disjoint i64 %address.local921, %121, !dbg !14
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !14
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !14
  %299 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !14
  %address.local924 = and i64 %address.local923, 16256, !dbg !14
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !14
  %.idx14211422 = or disjoint i64 %address.local927, %121, !dbg !14
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !14
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !14
  %300 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %301 = load <4 x half>, ptr addrspace(3) %lds.const.gep1238, align 8, !dbg !12
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %301, <4 x float> %255) #8, !dbg !12
  %303 = load <4 x half>, ptr addrspace(3) %lds.const.gep1239, align 8, !dbg !12
  %304 = load <4 x half>, ptr addrspace(3) %lds.const.gep1240, align 8, !dbg !12
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %303, <4 x float> %302) #8, !dbg !12
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %304, <4 x float> %259) #8, !dbg !12
  %307 = extractelement <4 x float> %305, i64 0, !dbg !15
  %308 = tail call float @llvm.maxnum.f32(float %307, float -inf), !dbg !16, !frisk.local_reduction !5
  %309 = extractelement <4 x float> %305, i64 1, !dbg !15
  %310 = extractelement <4 x float> %305, i64 2, !dbg !15
  %311 = extractelement <4 x float> %305, i64 3, !dbg !15
  %312 = load <4 x half>, ptr addrspace(3) %lds.const.gep1241, align 8, !dbg !12
  %313 = load <4 x half>, ptr addrspace(3) %lds.const.gep1242, align 8, !dbg !12
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %312, <4 x float> %306) #8, !dbg !12
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %313, <4 x float> %263) #8, !dbg !12
  %316 = extractelement <4 x float> %314, i64 0, !dbg !15
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %308, float %316), !dbg !16, !frisk.local_reduction !5
  %317 = extractelement <4 x float> %314, i64 1, !dbg !15
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %309, float %317), !dbg !16, !frisk.local_reduction !5
  %318 = extractelement <4 x float> %314, i64 2, !dbg !15
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %310, float %318), !dbg !16, !frisk.local_reduction !5
  %319 = extractelement <4 x float> %314, i64 3, !dbg !15
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %311, float %319), !dbg !16, !frisk.local_reduction !5
  %320 = load <4 x half>, ptr addrspace(3) %lds.const.gep1243, align 8, !dbg !12
  %321 = load <4 x half>, ptr addrspace(3) %lds.const.gep1244, align 8, !dbg !12
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %320, <4 x float> %315) #8, !dbg !12
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %321, <4 x float> %267) #8, !dbg !12
  %324 = extractelement <4 x float> %322, i64 0, !dbg !15
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %324), !dbg !16, !frisk.local_reduction !5
  %325 = extractelement <4 x float> %322, i64 1, !dbg !15
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %325), !dbg !16, !frisk.local_reduction !5
  %326 = extractelement <4 x float> %322, i64 2, !dbg !15
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %326), !dbg !16, !frisk.local_reduction !5
  %327 = extractelement <4 x float> %322, i64 3, !dbg !15
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %327), !dbg !16, !frisk.local_reduction !5
  %328 = load <4 x half>, ptr addrspace(3) %lds.const.gep1245, align 8, !dbg !12
  %329 = load <4 x half>, ptr addrspace(3) %lds.const.gep1246, align 8, !dbg !12
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %328, <4 x float> %323) #8, !dbg !12
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %329, <4 x float> %271) #8, !dbg !12
  %332 = extractelement <4 x float> %330, i64 0, !dbg !15
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %332), !dbg !16, !frisk.local_reduction !5
  %333 = extractelement <4 x float> %330, i64 1, !dbg !15
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %333), !dbg !16, !frisk.local_reduction !5
  %334 = extractelement <4 x float> %330, i64 2, !dbg !15
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %334), !dbg !16, !frisk.local_reduction !5
  %335 = extractelement <4 x float> %330, i64 3, !dbg !15
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %335), !dbg !16, !frisk.local_reduction !5
  %336 = load <4 x half>, ptr addrspace(3) %lds.const.gep1247, align 8, !dbg !12
  %337 = load <4 x half>, ptr addrspace(3) %lds.const.gep1248, align 8, !dbg !12
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %336, <4 x float> %331) #8, !dbg !12
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %337, <4 x float> %275) #8, !dbg !12
  %340 = extractelement <4 x float> %338, i64 0, !dbg !15
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %340), !dbg !16, !frisk.local_reduction !5
  %341 = extractelement <4 x float> %338, i64 1, !dbg !15
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %341), !dbg !16, !frisk.local_reduction !5
  %342 = extractelement <4 x float> %338, i64 2, !dbg !15
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %342), !dbg !16, !frisk.local_reduction !5
  %343 = extractelement <4 x float> %338, i64 3, !dbg !15
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %343), !dbg !16, !frisk.local_reduction !5
  %344 = load <4 x half>, ptr addrspace(3) %lds.const.gep1249, align 8, !dbg !12
  %345 = load <4 x half>, ptr addrspace(3) %lds.const.gep1250, align 8, !dbg !12
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %344, <4 x float> %339) #8, !dbg !12
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %345, <4 x float> %279) #8, !dbg !12
  %348 = extractelement <4 x float> %346, i64 0, !dbg !15
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %348), !dbg !16, !frisk.local_reduction !5
  %349 = extractelement <4 x float> %346, i64 1, !dbg !15
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %349), !dbg !16, !frisk.local_reduction !5
  %350 = extractelement <4 x float> %346, i64 2, !dbg !15
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %350), !dbg !16, !frisk.local_reduction !5
  %351 = extractelement <4 x float> %346, i64 3, !dbg !15
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %351), !dbg !16, !frisk.local_reduction !5
  %352 = load <4 x half>, ptr addrspace(3) %lds.const.gep1251, align 8, !dbg !12
  %353 = load <4 x half>, ptr addrspace(3) %lds.const.gep1252, align 8, !dbg !12
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %352, <4 x float> %347) #8, !dbg !12
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %353, <4 x float> %282) #8, !dbg !12
  %356 = extractelement <4 x float> %354, i64 0, !dbg !15
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %356), !dbg !16, !frisk.local_reduction !5
  %357 = extractelement <4 x float> %354, i64 1, !dbg !15
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %357), !dbg !16, !frisk.local_reduction !5
  %358 = extractelement <4 x float> %354, i64 2, !dbg !15
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %358), !dbg !16, !frisk.local_reduction !5
  %359 = extractelement <4 x float> %354, i64 3, !dbg !15
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %359), !dbg !16, !frisk.local_reduction !5
  %360 = load <4 x half>, ptr addrspace(3) %lds.const.gep1253, align 8, !dbg !12
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %360, <4 x float> %355) #8, !dbg !12
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %301, <4 x float> %284) #8, !dbg !12
  %363 = extractelement <4 x float> %361, i64 0, !dbg !15
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %363), !dbg !16, !frisk.local_reduction !5
  %364 = extractelement <4 x float> %361, i64 1, !dbg !15
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %364), !dbg !16, !frisk.local_reduction !5
  %365 = extractelement <4 x float> %361, i64 2, !dbg !15
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %365), !dbg !16, !frisk.local_reduction !5
  %366 = extractelement <4 x float> %361, i64 3, !dbg !15
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %366), !dbg !16, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !16, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !16, !frisk.local_reduction !5
  %367 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !16, !frisk.local_reduction !5
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %303, <4 x float> %362) #8, !dbg !12
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %304, <4 x float> %286) #8, !dbg !12
  %370 = extractelement <4 x float> %368, i64 0, !dbg !15
  %371 = extractelement <4 x float> %368, i64 1, !dbg !15
  %372 = extractelement <4 x float> %368, i64 2, !dbg !15
  %373 = extractelement <4 x float> %368, i64 3, !dbg !15
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %312, <4 x float> %369) #8, !dbg !12
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %313, <4 x float> %288) #8, !dbg !12
  %376 = extractelement <4 x float> %374, i64 0, !dbg !15
  %377 = extractelement <4 x float> %374, i64 1, !dbg !15
  %378 = extractelement <4 x float> %374, i64 2, !dbg !15
  %379 = extractelement <4 x float> %374, i64 3, !dbg !15
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %320, <4 x float> %375) #8, !dbg !12
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %321, <4 x float> %290) #8, !dbg !12
  %382 = extractelement <4 x float> %380, i64 0, !dbg !15
  %383 = extractelement <4 x float> %380, i64 1, !dbg !15
  %384 = extractelement <4 x float> %380, i64 2, !dbg !15
  %385 = extractelement <4 x float> %380, i64 3, !dbg !15
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %328, <4 x float> %381) #8, !dbg !12
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %329, <4 x float> %292) #8, !dbg !12
  %388 = extractelement <4 x float> %386, i64 0, !dbg !15
  %389 = extractelement <4 x float> %386, i64 1, !dbg !15
  %390 = extractelement <4 x float> %386, i64 2, !dbg !15
  %391 = extractelement <4 x float> %386, i64 3, !dbg !15
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %336, <4 x float> %387) #8, !dbg !12
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %337, <4 x float> %294) #8, !dbg !12
  %394 = extractelement <4 x float> %392, i64 0, !dbg !15
  %395 = extractelement <4 x float> %392, i64 1, !dbg !15
  %396 = extractelement <4 x float> %392, i64 2, !dbg !15
  %397 = extractelement <4 x float> %392, i64 3, !dbg !15
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %344, <4 x float> %393) #8, !dbg !12
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %345, <4 x float> %296) #8, !dbg !12
  %400 = extractelement <4 x float> %398, i64 0, !dbg !15
  %401 = extractelement <4 x float> %398, i64 1, !dbg !15
  %402 = extractelement <4 x float> %398, i64 2, !dbg !15
  %403 = extractelement <4 x float> %398, i64 3, !dbg !15
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %352, <4 x float> %399) #8, !dbg !12
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %353, <4 x float> %298) #8, !dbg !12
  %406 = extractelement <4 x float> %404, i64 0, !dbg !15
  %407 = extractelement <4 x float> %404, i64 1, !dbg !15
  %408 = extractelement <4 x float> %404, i64 2, !dbg !15
  %409 = extractelement <4 x float> %404, i64 3, !dbg !15
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %360, <4 x float> %405) #8, !dbg !12
  %411 = extractelement <4 x float> %410, i64 0, !dbg !15
  %412 = extractelement <4 x float> %410, i64 1, !dbg !15
  %413 = extractelement <4 x float> %410, i64 2, !dbg !15
  %414 = extractelement <4 x float> %410, i64 3, !dbg !15
  %415 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !16
  %416 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %415), !dbg !16
  %417 = add nuw nsw i32 %416, 64, !dbg !16
  %418 = xor i32 %416, 16, !dbg !16
  %419 = and i32 %417, 192, !dbg !16
  %420 = icmp samesign ult i32 %418, %419, !dbg !16
  %421 = select i1 %420, i32 %418, i32 %416, !dbg !16
  %422 = shl nuw nsw i32 %421, 2, !dbg !16
  %423 = bitcast float %367 to i32, !dbg !16
  %424 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %423), !dbg !16
  %425 = bitcast i32 %424 to float, !dbg !16
  %426 = tail call float @llvm.maxnum.f32(float %367, float %425), !dbg !16
  %427 = xor i32 %416, 32, !dbg !16
  %428 = icmp samesign ult i32 %427, %419, !dbg !16
  %429 = select i1 %428, i32 %427, i32 %416, !dbg !16
  %430 = shl nuw nsw i32 %429, 2, !dbg !16
  %431 = bitcast float %426 to i32, !dbg !16
  %432 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %431), !dbg !16
  %433 = bitcast i32 %432 to float, !dbg !16
  %434 = tail call float @llvm.maxnum.f32(float %426, float %433), !dbg !16
  %435 = extractvalue [2 x <1 x float>] %116, 0, !dbg !16
  %436 = insertelement <1 x float> poison, float %434, i64 0, !dbg !16
  %437 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %436, <1 x float> %435), !dbg !16
  %438 = tail call float @llvm.maxnum.f32(float %370, float -inf), !dbg !16, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %438, float %376), !dbg !16, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %371, float %377), !dbg !16, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %372, float %378), !dbg !16, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %373, float %379), !dbg !16, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %382), !dbg !16, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %383), !dbg !16, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %384), !dbg !16, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %385), !dbg !16, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %388), !dbg !16, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %389), !dbg !16, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %390), !dbg !16, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %391), !dbg !16, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %394), !dbg !16, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %395), !dbg !16, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %396), !dbg !16, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %397), !dbg !16, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %400), !dbg !16, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %401), !dbg !16, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %402), !dbg !16, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %403), !dbg !16, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %406), !dbg !16, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %407), !dbg !16, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %408), !dbg !16, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %409), !dbg !16, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %411), !dbg !16, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %412), !dbg !16, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %413), !dbg !16, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %414), !dbg !16, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !16, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !16, !frisk.local_reduction !5
  %439 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !16, !frisk.local_reduction !5
  %440 = bitcast float %439 to i32, !dbg !16
  %441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %440), !dbg !16
  %442 = bitcast i32 %441 to float, !dbg !16
  %443 = tail call float @llvm.maxnum.f32(float %439, float %442), !dbg !16
  %444 = bitcast float %443 to i32, !dbg !16
  %445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %444), !dbg !16
  %446 = bitcast i32 %445 to float, !dbg !16
  %447 = tail call float @llvm.maxnum.f32(float %443, float %446), !dbg !16
  %448 = extractvalue [2 x <1 x float>] %116, 1, !dbg !16
  %449 = insertelement <1 x float> poison, float %447, i64 0, !dbg !16
  %450 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %449, <1 x float> %448), !dbg !16
  %451 = shufflevector <1 x float> %437, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %452 = fsub <4 x float> %305, %451, !dbg !17
  %453 = fsub <4 x float> %314, %451, !dbg !17
  %454 = fsub <4 x float> %322, %451, !dbg !17
  %455 = fsub <4 x float> %330, %451, !dbg !17
  %456 = fsub <4 x float> %338, %451, !dbg !17
  %457 = fsub <4 x float> %346, %451, !dbg !17
  %458 = fsub <4 x float> %354, %451, !dbg !17
  %459 = fsub <4 x float> %361, %451, !dbg !17
  %460 = shufflevector <1 x float> %450, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %461 = fsub <4 x float> %368, %460, !dbg !17
  %462 = fsub <4 x float> %374, %460, !dbg !17
  %463 = fsub <4 x float> %380, %460, !dbg !17
  %464 = fsub <4 x float> %386, %460, !dbg !17
  %465 = fsub <4 x float> %392, %460, !dbg !17
  %466 = fsub <4 x float> %398, %460, !dbg !17
  %467 = fsub <4 x float> %404, %460, !dbg !17
  %468 = fsub <4 x float> %410, %460, !dbg !17
  %469 = fsub <1 x float> %435, %437, !dbg !18
  %470 = extractelement <1 x float> %469, i64 0, !dbg !19
  %471 = tail call float @llvm.amdgcn.exp2.f32(float %470), !dbg !19
  %472 = insertelement <4 x float> poison, float %471, i64 0, !dbg !20
  %473 = shufflevector <4 x float> %472, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %474 = extractvalue [2 x <1 x float>] %117, 0, !dbg !21
  %475 = extractelement <1 x float> %474, i64 0, !dbg !21
  %476 = fmul float %475, %471, !dbg !21
  %477 = fsub <1 x float> %448, %450, !dbg !18
  %478 = extractelement <1 x float> %477, i64 0, !dbg !19
  %479 = tail call float @llvm.amdgcn.exp2.f32(float %478), !dbg !19
  %480 = insertelement <4 x float> poison, float %479, i64 0, !dbg !20
  %481 = shufflevector <4 x float> %480, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %482 = extractvalue [2 x <1 x float>] %117, 1, !dbg !21
  %483 = extractelement <1 x float> %482, i64 0, !dbg !21
  %484 = fmul float %483, %479, !dbg !21
  %485 = extractelement <4 x float> %452, i64 0, !dbg !22
  %486 = tail call float @llvm.amdgcn.exp2.f32(float %485), !dbg !22
  %487 = insertelement <4 x float> poison, float %486, i64 0, !dbg !23
  %488 = extractelement <4 x float> %452, i64 1, !dbg !22
  %489 = tail call float @llvm.amdgcn.exp2.f32(float %488), !dbg !22
  %490 = insertelement <4 x float> %487, float %489, i64 1, !dbg !23
  %491 = extractelement <4 x float> %452, i64 2, !dbg !22
  %492 = tail call float @llvm.amdgcn.exp2.f32(float %491), !dbg !22
  %493 = insertelement <4 x float> %490, float %492, i64 2, !dbg !23
  %494 = extractelement <4 x float> %452, i64 3, !dbg !22
  %495 = tail call float @llvm.amdgcn.exp2.f32(float %494), !dbg !22
  %496 = insertelement <4 x float> %493, float %495, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %496) #8, !dbg !23, !frisk.register_lifetimes !5
  %497 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %497) #8, !dbg !22, !frisk.register_lifetimes !5
  %498 = extractelement <4 x float> %453, i64 0, !dbg !22
  %499 = tail call float @llvm.amdgcn.exp2.f32(float %498), !dbg !22
  %500 = insertelement <4 x float> poison, float %499, i64 0, !dbg !23
  %local.reduce89 = fadd float %486, %499, !dbg !24, !frisk.local_reduction !5
  %501 = extractelement <4 x float> %453, i64 1, !dbg !22
  %502 = tail call float @llvm.amdgcn.exp2.f32(float %501), !dbg !22
  %503 = insertelement <4 x float> %500, float %502, i64 1, !dbg !23
  %local.reduce90 = fadd float %489, %502, !dbg !24, !frisk.local_reduction !5
  %504 = extractelement <4 x float> %453, i64 2, !dbg !22
  %505 = tail call float @llvm.amdgcn.exp2.f32(float %504), !dbg !22
  %506 = insertelement <4 x float> %503, float %505, i64 2, !dbg !23
  %local.reduce91 = fadd float %492, %505, !dbg !24, !frisk.local_reduction !5
  %507 = extractelement <4 x float> %453, i64 3, !dbg !22
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %507), !dbg !22
  %509 = insertelement <4 x float> %506, float %508, i64 3, !dbg !23
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %509) #8, !dbg !23, !frisk.register_lifetimes !5
  %510 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !23
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %510) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce92 = fadd float %495, %508, !dbg !24, !frisk.local_reduction !5
  %511 = extractelement <4 x float> %454, i64 0, !dbg !22
  %512 = tail call float @llvm.amdgcn.exp2.f32(float %511), !dbg !22
  %513 = insertelement <4 x float> poison, float %512, i64 0, !dbg !23
  %local.reduce93 = fadd float %512, %local.reduce89, !dbg !24, !frisk.local_reduction !5
  %514 = extractelement <4 x float> %454, i64 1, !dbg !22
  %515 = tail call float @llvm.amdgcn.exp2.f32(float %514), !dbg !22
  %516 = insertelement <4 x float> %513, float %515, i64 1, !dbg !23
  %local.reduce94 = fadd float %515, %local.reduce90, !dbg !24, !frisk.local_reduction !5
  %517 = extractelement <4 x float> %454, i64 2, !dbg !22
  %518 = tail call float @llvm.amdgcn.exp2.f32(float %517), !dbg !22
  %519 = insertelement <4 x float> %516, float %518, i64 2, !dbg !23
  %local.reduce95 = fadd float %518, %local.reduce91, !dbg !24, !frisk.local_reduction !5
  %520 = extractelement <4 x float> %454, i64 3, !dbg !22
  %521 = tail call float @llvm.amdgcn.exp2.f32(float %520), !dbg !22
  %522 = insertelement <4 x float> %519, float %521, i64 3, !dbg !23
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %522) #8, !dbg !23, !frisk.register_lifetimes !5
  %523 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !23
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %523) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce96 = fadd float %521, %local.reduce92, !dbg !24, !frisk.local_reduction !5
  %524 = extractelement <4 x float> %455, i64 0, !dbg !22
  %525 = tail call float @llvm.amdgcn.exp2.f32(float %524), !dbg !22
  %526 = insertelement <4 x float> poison, float %525, i64 0, !dbg !23
  %local.reduce97 = fadd float %525, %local.reduce93, !dbg !24, !frisk.local_reduction !5
  %527 = extractelement <4 x float> %455, i64 1, !dbg !22
  %528 = tail call float @llvm.amdgcn.exp2.f32(float %527), !dbg !22
  %529 = insertelement <4 x float> %526, float %528, i64 1, !dbg !23
  %local.reduce98 = fadd float %528, %local.reduce94, !dbg !24, !frisk.local_reduction !5
  %530 = extractelement <4 x float> %455, i64 2, !dbg !22
  %531 = tail call float @llvm.amdgcn.exp2.f32(float %530), !dbg !22
  %532 = insertelement <4 x float> %529, float %531, i64 2, !dbg !23
  %local.reduce99 = fadd float %531, %local.reduce95, !dbg !24, !frisk.local_reduction !5
  %533 = extractelement <4 x float> %455, i64 3, !dbg !22
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !22
  %535 = insertelement <4 x float> %532, float %534, i64 3, !dbg !23
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %535) #8, !dbg !23, !frisk.register_lifetimes !5
  %536 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !23
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %536) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce100 = fadd float %534, %local.reduce96, !dbg !24, !frisk.local_reduction !5
  %537 = extractelement <4 x float> %456, i64 0, !dbg !22
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %537), !dbg !22
  %539 = insertelement <4 x float> poison, float %538, i64 0, !dbg !23
  %local.reduce101 = fadd float %538, %local.reduce97, !dbg !24, !frisk.local_reduction !5
  %540 = extractelement <4 x float> %456, i64 1, !dbg !22
  %541 = tail call float @llvm.amdgcn.exp2.f32(float %540), !dbg !22
  %542 = insertelement <4 x float> %539, float %541, i64 1, !dbg !23
  %local.reduce102 = fadd float %541, %local.reduce98, !dbg !24, !frisk.local_reduction !5
  %543 = extractelement <4 x float> %456, i64 2, !dbg !22
  %544 = tail call float @llvm.amdgcn.exp2.f32(float %543), !dbg !22
  %545 = insertelement <4 x float> %542, float %544, i64 2, !dbg !23
  %local.reduce103 = fadd float %544, %local.reduce99, !dbg !24, !frisk.local_reduction !5
  %546 = extractelement <4 x float> %456, i64 3, !dbg !22
  %547 = tail call float @llvm.amdgcn.exp2.f32(float %546), !dbg !22
  %548 = insertelement <4 x float> %545, float %547, i64 3, !dbg !23
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %548) #8, !dbg !23, !frisk.register_lifetimes !5
  %549 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !23
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %549) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce104 = fadd float %547, %local.reduce100, !dbg !24, !frisk.local_reduction !5
  %550 = extractelement <4 x float> %457, i64 0, !dbg !22
  %551 = tail call float @llvm.amdgcn.exp2.f32(float %550), !dbg !22
  %552 = insertelement <4 x float> poison, float %551, i64 0, !dbg !23
  %local.reduce105 = fadd float %551, %local.reduce101, !dbg !24, !frisk.local_reduction !5
  %553 = extractelement <4 x float> %457, i64 1, !dbg !22
  %554 = tail call float @llvm.amdgcn.exp2.f32(float %553), !dbg !22
  %555 = insertelement <4 x float> %552, float %554, i64 1, !dbg !23
  %local.reduce106 = fadd float %554, %local.reduce102, !dbg !24, !frisk.local_reduction !5
  %556 = extractelement <4 x float> %457, i64 2, !dbg !22
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %556), !dbg !22
  %558 = insertelement <4 x float> %555, float %557, i64 2, !dbg !23
  %local.reduce107 = fadd float %557, %local.reduce103, !dbg !24, !frisk.local_reduction !5
  %559 = extractelement <4 x float> %457, i64 3, !dbg !22
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !22
  %561 = insertelement <4 x float> %558, float %560, i64 3, !dbg !23
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %561) #8, !dbg !23, !frisk.register_lifetimes !5
  %562 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !23
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %562) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce108 = fadd float %560, %local.reduce104, !dbg !24, !frisk.local_reduction !5
  %563 = extractelement <4 x float> %458, i64 0, !dbg !22
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %563), !dbg !22
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !23
  %local.reduce109 = fadd float %564, %local.reduce105, !dbg !24, !frisk.local_reduction !5
  %566 = extractelement <4 x float> %458, i64 1, !dbg !22
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %566), !dbg !22
  %568 = insertelement <4 x float> %565, float %567, i64 1, !dbg !23
  %local.reduce110 = fadd float %567, %local.reduce106, !dbg !24, !frisk.local_reduction !5
  %569 = extractelement <4 x float> %458, i64 2, !dbg !22
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %569), !dbg !22
  %571 = insertelement <4 x float> %568, float %570, i64 2, !dbg !23
  %local.reduce111 = fadd float %570, %local.reduce107, !dbg !24, !frisk.local_reduction !5
  %572 = extractelement <4 x float> %458, i64 3, !dbg !22
  %573 = tail call float @llvm.amdgcn.exp2.f32(float %572), !dbg !22
  %574 = insertelement <4 x float> %571, float %573, i64 3, !dbg !23
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #8, !dbg !23, !frisk.register_lifetimes !5
  %575 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !23
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce112 = fadd float %573, %local.reduce108, !dbg !24, !frisk.local_reduction !5
  %576 = extractelement <4 x float> %459, i64 0, !dbg !22
  %577 = tail call float @llvm.amdgcn.exp2.f32(float %576), !dbg !22
  %578 = insertelement <4 x float> poison, float %577, i64 0, !dbg !23
  %local.reduce113 = fadd float %577, %local.reduce109, !dbg !24, !frisk.local_reduction !5
  %579 = extractelement <4 x float> %459, i64 1, !dbg !22
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %579), !dbg !22
  %581 = insertelement <4 x float> %578, float %580, i64 1, !dbg !23
  %local.reduce114 = fadd float %580, %local.reduce110, !dbg !24, !frisk.local_reduction !5
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !24, !frisk.local_reduction !5
  %582 = extractelement <4 x float> %459, i64 2, !dbg !22
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !22
  %584 = insertelement <4 x float> %581, float %583, i64 2, !dbg !23
  %local.reduce115 = fadd float %583, %local.reduce111, !dbg !24, !frisk.local_reduction !5
  %585 = extractelement <4 x float> %459, i64 3, !dbg !22
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %585), !dbg !22
  %587 = insertelement <4 x float> %584, float %586, i64 3, !dbg !23
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %587) #8, !dbg !23, !frisk.register_lifetimes !5
  %588 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !23
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %588) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce116 = fadd float %586, %local.reduce112, !dbg !24, !frisk.local_reduction !5
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !24, !frisk.local_reduction !5
  %589 = fadd float %local.reduce117, %local.reduce118, !dbg !24, !frisk.local_reduction !5
  %590 = fadd float %476, %589, !dbg !24
  %591 = insertelement <1 x float> poison, float %590, i64 0, !dbg !11
  %592 = insertvalue [2 x <1 x float>] poison, <1 x float> %591, 0, !dbg !11
  %593 = extractelement <4 x float> %461, i64 0, !dbg !22
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %593), !dbg !22
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !23
  %596 = extractelement <4 x float> %461, i64 1, !dbg !22
  %597 = tail call float @llvm.amdgcn.exp2.f32(float %596), !dbg !22
  %598 = insertelement <4 x float> %595, float %597, i64 1, !dbg !23
  %599 = extractelement <4 x float> %461, i64 2, !dbg !22
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %599), !dbg !22
  %601 = insertelement <4 x float> %598, float %600, i64 2, !dbg !23
  %602 = extractelement <4 x float> %461, i64 3, !dbg !22
  %603 = tail call float @llvm.amdgcn.exp2.f32(float %602), !dbg !22
  %604 = insertelement <4 x float> %601, float %603, i64 3, !dbg !23
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %604) #8, !dbg !23, !frisk.register_lifetimes !5
  %605 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !23
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %605) #8, !dbg !22, !frisk.register_lifetimes !5
  %606 = extractelement <4 x float> %462, i64 0, !dbg !22
  %607 = tail call float @llvm.amdgcn.exp2.f32(float %606), !dbg !22
  %608 = insertelement <4 x float> poison, float %607, i64 0, !dbg !23
  %local.reduce = fadd float %594, %607, !dbg !24, !frisk.local_reduction !5
  %609 = extractelement <4 x float> %462, i64 1, !dbg !22
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %609), !dbg !22
  %611 = insertelement <4 x float> %608, float %610, i64 1, !dbg !23
  %local.reduce59 = fadd float %597, %610, !dbg !24, !frisk.local_reduction !5
  %612 = extractelement <4 x float> %462, i64 2, !dbg !22
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !22
  %614 = insertelement <4 x float> %611, float %613, i64 2, !dbg !23
  %local.reduce60 = fadd float %600, %613, !dbg !24, !frisk.local_reduction !5
  %615 = extractelement <4 x float> %462, i64 3, !dbg !22
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %615), !dbg !22
  %617 = insertelement <4 x float> %614, float %616, i64 3, !dbg !23
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %617) #8, !dbg !23, !frisk.register_lifetimes !5
  %618 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !23
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %618) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce61 = fadd float %603, %616, !dbg !24, !frisk.local_reduction !5
  %619 = extractelement <4 x float> %463, i64 0, !dbg !22
  %620 = tail call float @llvm.amdgcn.exp2.f32(float %619), !dbg !22
  %621 = insertelement <4 x float> poison, float %620, i64 0, !dbg !23
  %local.reduce62 = fadd float %620, %local.reduce, !dbg !24, !frisk.local_reduction !5
  %622 = extractelement <4 x float> %463, i64 1, !dbg !22
  %623 = tail call float @llvm.amdgcn.exp2.f32(float %622), !dbg !22
  %624 = insertelement <4 x float> %621, float %623, i64 1, !dbg !23
  %local.reduce63 = fadd float %623, %local.reduce59, !dbg !24, !frisk.local_reduction !5
  %625 = extractelement <4 x float> %463, i64 2, !dbg !22
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %625), !dbg !22
  %627 = insertelement <4 x float> %624, float %626, i64 2, !dbg !23
  %local.reduce64 = fadd float %626, %local.reduce60, !dbg !24, !frisk.local_reduction !5
  %628 = extractelement <4 x float> %463, i64 3, !dbg !22
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !22
  %630 = insertelement <4 x float> %627, float %629, i64 3, !dbg !23
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %630) #8, !dbg !23, !frisk.register_lifetimes !5
  %631 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !23
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %631) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce65 = fadd float %629, %local.reduce61, !dbg !24, !frisk.local_reduction !5
  %632 = extractelement <4 x float> %464, i64 0, !dbg !22
  %633 = tail call float @llvm.amdgcn.exp2.f32(float %632), !dbg !22
  %634 = insertelement <4 x float> poison, float %633, i64 0, !dbg !23
  %local.reduce66 = fadd float %633, %local.reduce62, !dbg !24, !frisk.local_reduction !5
  %635 = extractelement <4 x float> %464, i64 1, !dbg !22
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %635), !dbg !22
  %637 = insertelement <4 x float> %634, float %636, i64 1, !dbg !23
  %local.reduce67 = fadd float %636, %local.reduce63, !dbg !24, !frisk.local_reduction !5
  %638 = extractelement <4 x float> %464, i64 2, !dbg !22
  %639 = tail call float @llvm.amdgcn.exp2.f32(float %638), !dbg !22
  %640 = insertelement <4 x float> %637, float %639, i64 2, !dbg !23
  %local.reduce68 = fadd float %639, %local.reduce64, !dbg !24, !frisk.local_reduction !5
  %641 = extractelement <4 x float> %464, i64 3, !dbg !22
  %642 = tail call float @llvm.amdgcn.exp2.f32(float %641), !dbg !22
  %643 = insertelement <4 x float> %640, float %642, i64 3, !dbg !23
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %643) #8, !dbg !23, !frisk.register_lifetimes !5
  %644 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !23
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %644) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce69 = fadd float %642, %local.reduce65, !dbg !24, !frisk.local_reduction !5
  %645 = extractelement <4 x float> %465, i64 0, !dbg !22
  %646 = tail call float @llvm.amdgcn.exp2.f32(float %645), !dbg !22
  %647 = insertelement <4 x float> poison, float %646, i64 0, !dbg !23
  %local.reduce70 = fadd float %646, %local.reduce66, !dbg !24, !frisk.local_reduction !5
  %648 = extractelement <4 x float> %465, i64 1, !dbg !22
  %649 = tail call float @llvm.amdgcn.exp2.f32(float %648), !dbg !22
  %650 = insertelement <4 x float> %647, float %649, i64 1, !dbg !23
  %local.reduce71 = fadd float %649, %local.reduce67, !dbg !24, !frisk.local_reduction !5
  %651 = extractelement <4 x float> %465, i64 2, !dbg !22
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !22
  %653 = insertelement <4 x float> %650, float %652, i64 2, !dbg !23
  %local.reduce72 = fadd float %652, %local.reduce68, !dbg !24, !frisk.local_reduction !5
  %654 = extractelement <4 x float> %465, i64 3, !dbg !22
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !22
  %656 = insertelement <4 x float> %653, float %655, i64 3, !dbg !23
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %656) #8, !dbg !23, !frisk.register_lifetimes !5
  %657 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !23
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %657) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce73 = fadd float %655, %local.reduce69, !dbg !24, !frisk.local_reduction !5
  %658 = extractelement <4 x float> %466, i64 0, !dbg !22
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %658), !dbg !22
  %660 = insertelement <4 x float> poison, float %659, i64 0, !dbg !23
  %local.reduce74 = fadd float %659, %local.reduce70, !dbg !24, !frisk.local_reduction !5
  %661 = extractelement <4 x float> %466, i64 1, !dbg !22
  %662 = tail call float @llvm.amdgcn.exp2.f32(float %661), !dbg !22
  %663 = insertelement <4 x float> %660, float %662, i64 1, !dbg !23
  %local.reduce75 = fadd float %662, %local.reduce71, !dbg !24, !frisk.local_reduction !5
  %664 = extractelement <4 x float> %466, i64 2, !dbg !22
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %664), !dbg !22
  %666 = insertelement <4 x float> %663, float %665, i64 2, !dbg !23
  %local.reduce76 = fadd float %665, %local.reduce72, !dbg !24, !frisk.local_reduction !5
  %667 = extractelement <4 x float> %466, i64 3, !dbg !22
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %667), !dbg !22
  %669 = insertelement <4 x float> %666, float %668, i64 3, !dbg !23
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %669) #8, !dbg !23, !frisk.register_lifetimes !5
  %670 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !23
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %670) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce77 = fadd float %668, %local.reduce73, !dbg !24, !frisk.local_reduction !5
  %671 = extractelement <4 x float> %467, i64 0, !dbg !22
  %672 = tail call float @llvm.amdgcn.exp2.f32(float %671), !dbg !22
  %673 = insertelement <4 x float> poison, float %672, i64 0, !dbg !23
  %local.reduce78 = fadd float %672, %local.reduce74, !dbg !24, !frisk.local_reduction !5
  %674 = extractelement <4 x float> %467, i64 1, !dbg !22
  %675 = tail call float @llvm.amdgcn.exp2.f32(float %674), !dbg !22
  %676 = insertelement <4 x float> %673, float %675, i64 1, !dbg !23
  %local.reduce79 = fadd float %675, %local.reduce75, !dbg !24, !frisk.local_reduction !5
  %677 = extractelement <4 x float> %467, i64 2, !dbg !22
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !22
  %679 = insertelement <4 x float> %676, float %678, i64 2, !dbg !23
  %local.reduce80 = fadd float %678, %local.reduce76, !dbg !24, !frisk.local_reduction !5
  %680 = extractelement <4 x float> %467, i64 3, !dbg !22
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %680), !dbg !22
  %682 = insertelement <4 x float> %679, float %681, i64 3, !dbg !23
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %682) #8, !dbg !23, !frisk.register_lifetimes !5
  %683 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !23
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %683) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce81 = fadd float %681, %local.reduce77, !dbg !24, !frisk.local_reduction !5
  %684 = extractelement <4 x float> %468, i64 0, !dbg !22
  %685 = tail call float @llvm.amdgcn.exp2.f32(float %684), !dbg !22
  %686 = insertelement <4 x float> poison, float %685, i64 0, !dbg !23
  %local.reduce82 = fadd float %685, %local.reduce78, !dbg !24, !frisk.local_reduction !5
  %687 = extractelement <4 x float> %468, i64 1, !dbg !22
  %688 = tail call float @llvm.amdgcn.exp2.f32(float %687), !dbg !22
  %689 = insertelement <4 x float> %686, float %688, i64 1, !dbg !23
  %local.reduce83 = fadd float %688, %local.reduce79, !dbg !24, !frisk.local_reduction !5
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !24, !frisk.local_reduction !5
  %690 = extractelement <4 x float> %468, i64 2, !dbg !22
  %691 = tail call float @llvm.amdgcn.exp2.f32(float %690), !dbg !22
  %692 = insertelement <4 x float> %689, float %691, i64 2, !dbg !23
  %local.reduce84 = fadd float %691, %local.reduce80, !dbg !24, !frisk.local_reduction !5
  %693 = extractelement <4 x float> %468, i64 3, !dbg !22
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %693), !dbg !22
  %695 = insertelement <4 x float> %692, float %694, i64 3, !dbg !23
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %695) #8, !dbg !23, !frisk.register_lifetimes !5
  %696 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !23
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %696) #8, !dbg !24, !frisk.register_lifetimes !5
  %local.reduce85 = fadd float %694, %local.reduce81, !dbg !24, !frisk.local_reduction !5
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !24, !frisk.local_reduction !5
  %697 = fadd float %local.reduce86, %local.reduce87, !dbg !24, !frisk.local_reduction !5
  %698 = fadd float %484, %697, !dbg !24
  %699 = insertelement <1 x float> poison, float %698, i64 0, !dbg !11
  %700 = insertvalue [2 x <1 x float>] %592, <1 x float> %699, 1, !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !25
  %address.local930 = and i32 %address.local929, 120, !dbg !25
  %address.local932 = and i32 %address.local575, 1536, !dbg !25
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !25
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !25
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !25
  %701 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %702 = bitcast <4 x i32> %701 to <8 x half>, !dbg !25
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
  %703 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %704 = bitcast <4 x i32> %703 to <8 x half>, !dbg !25
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !25
  %705 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %706 = bitcast <4 x i32> %705 to <8 x half>, !dbg !25
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !25
  %707 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %708 = bitcast <4 x i32> %707 to <8 x half>, !dbg !25
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !25
  %709 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %710 = bitcast <4 x i32> %709 to <8 x half>, !dbg !25
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !25
  %711 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %712 = bitcast <4 x i32> %711 to <8 x half>, !dbg !25
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !25
  %713 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %714 = bitcast <4 x i32> %713 to <8 x half>, !dbg !25
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !25
  %715 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %716 = bitcast <4 x i32> %715 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14231424 = or disjoint i64 %address.local921, %122, !dbg !14
  %.idx14231424.tr = trunc nuw nsw i64 %.idx14231424 to i32, !dbg !14
  %dma.buffer.offset1342 = shl nuw nsw i32 %.idx14231424.tr, 1, !dbg !14
  %717 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1342, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14251426 = or disjoint i64 %address.local927, %122, !dbg !14
  %.idx14251426.tr = trunc nuw nsw i64 %.idx14251426 to i32, !dbg !14
  %dma.buffer.offset1339 = shl nuw nsw i32 %.idx14251426.tr, 1, !dbg !14
  %718 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %719 = shufflevector <8 x half> %702, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %720 = extractvalue [2 x <8 x float>] %118, 0, !dbg !20
  %721 = shufflevector <8 x float> %720, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %722 = fmul <4 x float> %721, %473, !dbg !20
  %723 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %719, <4 x float> %722) #8, !dbg !25
  %724 = shufflevector <8 x half> %704, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %724, <4 x float> %723) #8, !dbg !25
  %726 = shufflevector <8 x half> %706, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %727 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %726, <4 x float> %725) #8, !dbg !25
  %728 = shufflevector <8 x half> %708, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %729 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %728, <4 x float> %727) #8, !dbg !25
  %730 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %730, <4 x float> %729) #8, !dbg !25
  %732 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %733 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %732, <4 x float> %731) #8, !dbg !25
  %734 = shufflevector <8 x half> %714, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %734, <4 x float> %733) #8, !dbg !25
  %736 = shufflevector <8 x half> %716, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %736, <4 x float> %735) #8, !dbg !25
  %738 = shufflevector <8 x half> %702, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %739 = shufflevector <8 x float> %720, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %740 = fmul <4 x float> %739, %473, !dbg !20
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %738, <4 x float> %740) #8, !dbg !25
  %742 = shufflevector <8 x half> %704, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %742, <4 x float> %741) #8, !dbg !25
  %744 = shufflevector <8 x half> %706, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %744, <4 x float> %743) #8, !dbg !25
  %746 = shufflevector <8 x half> %708, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %746, <4 x float> %745) #8, !dbg !25
  %748 = shufflevector <8 x half> %710, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %748, <4 x float> %747) #8, !dbg !25
  %750 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %751 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %750, <4 x float> %749) #8, !dbg !25
  %752 = shufflevector <8 x half> %714, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %753 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %752, <4 x float> %751) #8, !dbg !25
  %754 = shufflevector <8 x half> %716, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %754, <4 x float> %753) #8, !dbg !25
  %756 = extractvalue [2 x <8 x float>] %118, 1, !dbg !20
  %757 = shufflevector <8 x float> %756, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %758 = fmul <4 x float> %757, %481, !dbg !20
  %759 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %719, <4 x float> %758) #8, !dbg !25
  %760 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %724, <4 x float> %759) #8, !dbg !25
  %761 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %726, <4 x float> %760) #8, !dbg !25
  %762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %728, <4 x float> %761) #8, !dbg !25
  %763 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %730, <4 x float> %762) #8, !dbg !25
  %764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %732, <4 x float> %763) #8, !dbg !25
  %765 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %734, <4 x float> %764) #8, !dbg !25
  %766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %736, <4 x float> %765) #8, !dbg !25
  %767 = shufflevector <8 x float> %756, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %768 = fmul <4 x float> %767, %481, !dbg !20
  %769 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %738, <4 x float> %768) #8, !dbg !25
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %742, <4 x float> %769) #8, !dbg !25
  %771 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %744, <4 x float> %770) #8, !dbg !25
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %746, <4 x float> %771) #8, !dbg !25
  %773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %748, <4 x float> %772) #8, !dbg !25
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %750, <4 x float> %773) #8, !dbg !25
  %775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %752, <4 x float> %774) #8, !dbg !25
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %754, <4 x float> %775) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %777 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %778 = bitcast <4 x i32> %777 to <8 x half>, !dbg !25
  %779 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %780 = bitcast <4 x i32> %779 to <8 x half>, !dbg !25
  %781 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %782 = bitcast <4 x i32> %781 to <8 x half>, !dbg !25
  %783 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %784 = bitcast <4 x i32> %783 to <8 x half>, !dbg !25
  %785 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %786 = bitcast <4 x i32> %785 to <8 x half>, !dbg !25
  %787 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %788 = bitcast <4 x i32> %787 to <8 x half>, !dbg !25
  %789 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %790 = bitcast <4 x i32> %789 to <8 x half>, !dbg !25
  %791 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %792 = bitcast <4 x i32> %791 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14271428 = or disjoint i64 %address.local921, %197, !dbg !14
  %.idx14271428.tr = trunc nuw nsw i64 %.idx14271428 to i32, !dbg !14
  %dma.buffer.offset1336 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !14
  %793 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1336, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14291430 = or disjoint i64 %address.local927, %197, !dbg !14
  %.idx14291430.tr = trunc nuw nsw i64 %.idx14291430 to i32, !dbg !14
  %dma.buffer.offset1333 = shl nuw nsw i32 %.idx14291430.tr, 1, !dbg !14
  %794 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1333, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %795 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %796 = extractvalue [2 x <8 x float>] %115, 0, !dbg !20
  %797 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %798 = fmul <4 x float> %797, %473, !dbg !20
  %799 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %795, <4 x float> %798) #8, !dbg !25
  %800 = shufflevector <8 x half> %780, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %800, <4 x float> %799) #8, !dbg !25
  %802 = shufflevector <8 x half> %782, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %802, <4 x float> %801) #8, !dbg !25
  %804 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %804, <4 x float> %803) #8, !dbg !25
  %806 = shufflevector <8 x half> %786, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %806, <4 x float> %805) #8, !dbg !25
  %808 = shufflevector <8 x half> %788, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %809 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %808, <4 x float> %807) #8, !dbg !25
  %810 = shufflevector <8 x half> %790, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %810, <4 x float> %809) #8, !dbg !25
  %812 = shufflevector <8 x half> %792, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %812, <4 x float> %811) #8, !dbg !25
  %814 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %815 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %816 = fmul <4 x float> %815, %473, !dbg !20
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %814, <4 x float> %816) #8, !dbg !25
  %818 = shufflevector <8 x half> %780, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %818, <4 x float> %817) #8, !dbg !25
  %820 = shufflevector <8 x half> %782, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %820, <4 x float> %819) #8, !dbg !25
  %822 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %823 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %822, <4 x float> %821) #8, !dbg !25
  %824 = shufflevector <8 x half> %786, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %824, <4 x float> %823) #8, !dbg !25
  %826 = shufflevector <8 x half> %788, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %827 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %826, <4 x float> %825) #8, !dbg !25
  %828 = shufflevector <8 x half> %790, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %829 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %828, <4 x float> %827) #8, !dbg !25
  %830 = shufflevector <8 x half> %792, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %830, <4 x float> %829) #8, !dbg !25
  %832 = extractvalue [2 x <8 x float>] %115, 1, !dbg !20
  %833 = shufflevector <8 x float> %832, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %834 = fmul <4 x float> %833, %481, !dbg !20
  %835 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %795, <4 x float> %834) #8, !dbg !25
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %800, <4 x float> %835) #8, !dbg !25
  %837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %802, <4 x float> %836) #8, !dbg !25
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %804, <4 x float> %837) #8, !dbg !25
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %806, <4 x float> %838) #8, !dbg !25
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %808, <4 x float> %839) #8, !dbg !25
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %810, <4 x float> %840) #8, !dbg !25
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %812, <4 x float> %841) #8, !dbg !25
  %843 = shufflevector <8 x float> %832, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %844 = fmul <4 x float> %843, %481, !dbg !20
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %814, <4 x float> %844) #8, !dbg !25
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %818, <4 x float> %845) #8, !dbg !25
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %820, <4 x float> %846) #8, !dbg !25
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %822, <4 x float> %847) #8, !dbg !25
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %824, <4 x float> %848) #8, !dbg !25
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %826, <4 x float> %849) #8, !dbg !25
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %828, <4 x float> %850) #8, !dbg !25
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %830, <4 x float> %851) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %853 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %854 = bitcast <4 x i32> %853 to <8 x half>, !dbg !25
  %855 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %856 = bitcast <4 x i32> %855 to <8 x half>, !dbg !25
  %857 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %858 = bitcast <4 x i32> %857 to <8 x half>, !dbg !25
  %859 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !25
  %861 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !25
  %863 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !25
  %865 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !25
  %867 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %.idx14311432 = or disjoint i64 %address.local921, %248, !dbg !14
  %.idx14311432.tr = trunc nuw nsw i64 %.idx14311432 to i32, !dbg !14
  %dma.buffer.offset1330 = shl nuw nsw i32 %.idx14311432.tr, 1, !dbg !14
  %869 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1330, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %.idx14331434 = or disjoint i64 %address.local927, %248, !dbg !14
  %.idx14331434.tr = trunc nuw nsw i64 %.idx14331434 to i32, !dbg !14
  %dma.buffer.offset = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !14
  %870 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !14
  %871 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %872 = extractvalue [2 x <8 x float>] %114, 0, !dbg !20
  %873 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %874 = fmul <4 x float> %873, %473, !dbg !20
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %871, <4 x float> %874) #8, !dbg !25
  %876 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %876, <4 x float> %875) #8, !dbg !25
  %878 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %878, <4 x float> %877) #8, !dbg !25
  %880 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %880, <4 x float> %879) #8, !dbg !25
  %882 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %882, <4 x float> %881) #8, !dbg !25
  %884 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %884, <4 x float> %883) #8, !dbg !25
  %886 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %886, <4 x float> %885) #8, !dbg !25
  %888 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %888, <4 x float> %887) #8, !dbg !25
  %890 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %891 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %892 = fmul <4 x float> %891, %473, !dbg !20
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %890, <4 x float> %892) #8, !dbg !25
  %894 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %894, <4 x float> %893) #8, !dbg !25
  %896 = shufflevector <8 x half> %858, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %896, <4 x float> %895) #8, !dbg !25
  %898 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %898, <4 x float> %897) #8, !dbg !25
  %900 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %900, <4 x float> %899) #8, !dbg !25
  %902 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %902, <4 x float> %901) #8, !dbg !25
  %904 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %904, <4 x float> %903) #8, !dbg !25
  %906 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %906, <4 x float> %905) #8, !dbg !25
  %908 = extractvalue [2 x <8 x float>] %114, 1, !dbg !20
  %909 = shufflevector <8 x float> %908, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %910 = fmul <4 x float> %909, %481, !dbg !20
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %871, <4 x float> %910) #8, !dbg !25
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %876, <4 x float> %911) #8, !dbg !25
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %878, <4 x float> %912) #8, !dbg !25
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %880, <4 x float> %913) #8, !dbg !25
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %882, <4 x float> %914) #8, !dbg !25
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %884, <4 x float> %915) #8, !dbg !25
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %886, <4 x float> %916) #8, !dbg !25
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %888, <4 x float> %917) #8, !dbg !25
  %919 = shufflevector <8 x float> %908, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %920 = fmul <4 x float> %919, %481, !dbg !20
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %890, <4 x float> %920) #8, !dbg !25
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %894, <4 x float> %921) #8, !dbg !25
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %896, <4 x float> %922) #8, !dbg !25
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %898, <4 x float> %923) #8, !dbg !25
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %900, <4 x float> %924) #8, !dbg !25
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %902, <4 x float> %925) #8, !dbg !25
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %904, <4 x float> %926) #8, !dbg !25
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %906, <4 x float> %927) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %929 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #10, !dbg !25
  %930 = bitcast <4 x i32> %929 to <8 x half>, !dbg !25
  %931 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #10, !dbg !25
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !25
  %933 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #10, !dbg !25
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !25
  %935 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #10, !dbg !25
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !25
  %937 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #10, !dbg !25
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !25
  %939 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #10, !dbg !25
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !25
  %941 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #10, !dbg !25
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !25
  %943 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #10, !dbg !25
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !25
  %945 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %946 = extractvalue [2 x <8 x float>] %113, 0, !dbg !20
  %947 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %948 = fmul <4 x float> %947, %473, !dbg !20
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %945, <4 x float> %948) #8, !dbg !25
  %950 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %950, <4 x float> %949) #8, !dbg !25
  %952 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %952, <4 x float> %951) #8, !dbg !25
  %954 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %954, <4 x float> %953) #8, !dbg !25
  %956 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %956, <4 x float> %955) #8, !dbg !25
  %958 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %958, <4 x float> %957) #8, !dbg !25
  %960 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %960, <4 x float> %959) #8, !dbg !25
  %962 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %962, <4 x float> %961) #8, !dbg !25
  %964 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %965 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %966 = fmul <4 x float> %965, %473, !dbg !20
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %964, <4 x float> %966) #8, !dbg !25
  %968 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %968, <4 x float> %967) #8, !dbg !25
  %970 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %970, <4 x float> %969) #8, !dbg !25
  %972 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %972, <4 x float> %971) #8, !dbg !25
  %974 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %974, <4 x float> %973) #8, !dbg !25
  %976 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %976, <4 x float> %975) #8, !dbg !25
  %978 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %978, <4 x float> %977) #8, !dbg !25
  %980 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %980, <4 x float> %979) #8, !dbg !25
  %982 = extractvalue [2 x <8 x float>] %113, 1, !dbg !20
  %983 = shufflevector <8 x float> %982, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !20
  %984 = fmul <4 x float> %983, %481, !dbg !20
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %945, <4 x float> %984) #8, !dbg !25
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %950, <4 x float> %985) #8, !dbg !25
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %952, <4 x float> %986) #8, !dbg !25
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %954, <4 x float> %987) #8, !dbg !25
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %956, <4 x float> %988) #8, !dbg !25
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %958, <4 x float> %989) #8, !dbg !25
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %960, <4 x float> %990) #8, !dbg !25
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %962, <4 x float> %991) #8, !dbg !25
  %993 = shufflevector <8 x float> %982, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !20
  %994 = fmul <4 x float> %993, %481, !dbg !20
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %964, <4 x float> %994) #8, !dbg !25
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %968, <4 x float> %995) #8, !dbg !25
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %970, <4 x float> %996) #8, !dbg !25
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %972, <4 x float> %997) #8, !dbg !25
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %974, <4 x float> %998) #8, !dbg !25
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %976, <4 x float> %999) #8, !dbg !25
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %978, <4 x float> %1000) #8, !dbg !25
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %980, <4 x float> %1001) #8, !dbg !25
  %1003 = shufflevector <4 x float> %737, <4 x float> %755, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1004 = insertvalue [2 x <8 x float>] poison, <8 x float> %1003, 0, !dbg !11
  %1005 = shufflevector <4 x float> %766, <4 x float> %776, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1006 = insertvalue [2 x <8 x float>] %1004, <8 x float> %1005, 1, !dbg !11
  %1007 = insertvalue [2 x <1 x float>] poison, <1 x float> %437, 0, !dbg !11
  %1008 = insertvalue [2 x <1 x float>] %1007, <1 x float> %450, 1, !dbg !11
  %1009 = shufflevector <4 x float> %813, <4 x float> %831, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1010 = insertvalue [2 x <8 x float>] poison, <8 x float> %1009, 0, !dbg !11
  %1011 = shufflevector <4 x float> %842, <4 x float> %852, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1012 = insertvalue [2 x <8 x float>] %1010, <8 x float> %1011, 1, !dbg !11
  %1013 = shufflevector <4 x float> %889, <4 x float> %907, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1014 = insertvalue [2 x <8 x float>] poison, <8 x float> %1013, 0, !dbg !11
  %1015 = shufflevector <4 x float> %918, <4 x float> %928, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1016 = insertvalue [2 x <8 x float>] %1014, <8 x float> %1015, 1, !dbg !11
  %1017 = shufflevector <4 x float> %963, <4 x float> %981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1018 = insertvalue [2 x <8 x float>] poison, <8 x float> %1017, 0, !dbg !11
  %1019 = shufflevector <4 x float> %992, <4 x float> %1002, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1020 = insertvalue [2 x <8 x float>] %1018, <8 x float> %1019, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1021 = add nuw nsw i64 %119, 1, !dbg !11
  %1022 = icmp samesign ult i64 %119, 31, !dbg !11
  br i1 %1022, label %112, label %1023, !dbg !11

1023:                                             ; preds = %112
  %1024 = shl nuw nsw i32 %416, 2, !dbg !11
  %1025 = xor i32 %1024, 64, !dbg !11
  %1026 = bitcast float %698 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1025, i32 %1026), !dbg !11
  %1027 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %698, %1027, !dbg !11
  %1028 = xor i32 %1024, 128, !dbg !11
  %1029 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1029), !dbg !11
  %1030 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %1030, !dbg !11
  %1031 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %1032 = bitcast float %590 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1025, i32 %1032), !dbg !11
  %1033 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %590, %1033, !dbg !11
  %1034 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1034), !dbg !11
  %1035 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %1035, !dbg !11
  %1036 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %1037 = and i32 %31, 16, !dbg !11
  %.not = icmp eq i32 %1037, 0, !dbg !11
  %1038 = and i32 %31, 32, !dbg !11
  %.not36 = icmp eq i32 %1038, 0, !dbg !11
  %1039 = extractelement <4 x float> %963, i64 0, !dbg !11
  %1040 = extractelement <4 x float> %963, i64 1, !dbg !11
  %1041 = extractelement <4 x float> %963, i64 2, !dbg !11
  %1042 = extractelement <4 x float> %963, i64 3, !dbg !11
  %1043 = select i1 %.not, float %1040, float %1039, !dbg !11
  %1044 = bitcast float %1043 to i32, !dbg !11
  %1045 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1044), !dbg !11
  %1046 = bitcast i32 %1045 to float, !dbg !11
  %1047 = select i1 %.not, float %1039, float %1046, !dbg !11
  %1048 = select i1 %.not, float %1046, float %1040, !dbg !11
  %1049 = select i1 %.not, float %1042, float %1041, !dbg !11
  %1050 = bitcast float %1049 to i32, !dbg !11
  %1051 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1050), !dbg !11
  %1052 = bitcast i32 %1051 to float, !dbg !11
  %1053 = select i1 %.not, float %1041, float %1052, !dbg !11
  %1054 = select i1 %.not, float %1052, float %1042, !dbg !11
  %1055 = select i1 %.not36, float %1053, float %1047, !dbg !11
  %1056 = bitcast float %1055 to i32, !dbg !11
  %1057 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1056), !dbg !11
  %1058 = bitcast i32 %1057 to float, !dbg !11
  %1059 = select i1 %.not36, float %1054, float %1048, !dbg !11
  %1060 = bitcast float %1059 to i32, !dbg !11
  %1061 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1060), !dbg !11
  %1062 = bitcast i32 %1061 to float, !dbg !11
  %1063 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1064 = shufflevector <4 x i1> %1063, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1065 = insertelement <4 x float> poison, float %1047, i64 0, !dbg !11
  %1066 = insertelement <4 x float> %1065, float %1048, i64 1, !dbg !11
  %1067 = insertelement <4 x float> %1066, float %1058, i64 2, !dbg !11
  %1068 = insertelement <4 x float> %1067, float %1062, i64 3, !dbg !11
  %1069 = insertelement <4 x float> poison, float %1058, i64 0, !dbg !11
  %1070 = insertelement <4 x float> %1069, float %1062, i64 1, !dbg !11
  %1071 = insertelement <4 x float> %1070, float %1053, i64 2, !dbg !11
  %1072 = insertelement <4 x float> %1071, float %1054, i64 3, !dbg !11
  %1073 = select <4 x i1> %1064, <4 x float> %1068, <4 x float> %1072, !dbg !11
  %1074 = extractelement <4 x float> %889, i64 0, !dbg !11
  %1075 = extractelement <4 x float> %889, i64 1, !dbg !11
  %1076 = extractelement <4 x float> %889, i64 2, !dbg !11
  %1077 = extractelement <4 x float> %889, i64 3, !dbg !11
  %1078 = select i1 %.not, float %1075, float %1074, !dbg !11
  %1079 = bitcast float %1078 to i32, !dbg !11
  %1080 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1079), !dbg !11
  %1081 = bitcast i32 %1080 to float, !dbg !11
  %1082 = select i1 %.not, float %1074, float %1081, !dbg !11
  %1083 = select i1 %.not, float %1081, float %1075, !dbg !11
  %1084 = select i1 %.not, float %1077, float %1076, !dbg !11
  %1085 = bitcast float %1084 to i32, !dbg !11
  %1086 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1085), !dbg !11
  %1087 = bitcast i32 %1086 to float, !dbg !11
  %1088 = select i1 %.not, float %1076, float %1087, !dbg !11
  %1089 = select i1 %.not, float %1087, float %1077, !dbg !11
  %1090 = select i1 %.not36, float %1088, float %1082, !dbg !11
  %1091 = bitcast float %1090 to i32, !dbg !11
  %1092 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1091), !dbg !11
  %1093 = bitcast i32 %1092 to float, !dbg !11
  %1094 = select i1 %.not36, float %1089, float %1083, !dbg !11
  %1095 = bitcast float %1094 to i32, !dbg !11
  %1096 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1095), !dbg !11
  %1097 = bitcast i32 %1096 to float, !dbg !11
  %1098 = insertelement <4 x float> poison, float %1082, i64 0, !dbg !11
  %1099 = insertelement <4 x float> %1098, float %1083, i64 1, !dbg !11
  %1100 = insertelement <4 x float> %1099, float %1093, i64 2, !dbg !11
  %1101 = insertelement <4 x float> %1100, float %1097, i64 3, !dbg !11
  %1102 = insertelement <4 x float> poison, float %1093, i64 0, !dbg !11
  %1103 = insertelement <4 x float> %1102, float %1097, i64 1, !dbg !11
  %1104 = insertelement <4 x float> %1103, float %1088, i64 2, !dbg !11
  %1105 = insertelement <4 x float> %1104, float %1089, i64 3, !dbg !11
  %1106 = select <4 x i1> %1064, <4 x float> %1101, <4 x float> %1105, !dbg !11
  %1107 = extractelement <4 x float> %813, i64 0, !dbg !11
  %1108 = extractelement <4 x float> %813, i64 1, !dbg !11
  %1109 = extractelement <4 x float> %813, i64 2, !dbg !11
  %1110 = extractelement <4 x float> %813, i64 3, !dbg !11
  %1111 = select i1 %.not, float %1108, float %1107, !dbg !11
  %1112 = bitcast float %1111 to i32, !dbg !11
  %1113 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1112), !dbg !11
  %1114 = bitcast i32 %1113 to float, !dbg !11
  %1115 = select i1 %.not, float %1107, float %1114, !dbg !11
  %1116 = select i1 %.not, float %1114, float %1108, !dbg !11
  %1117 = select i1 %.not, float %1110, float %1109, !dbg !11
  %1118 = bitcast float %1117 to i32, !dbg !11
  %1119 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1118), !dbg !11
  %1120 = bitcast i32 %1119 to float, !dbg !11
  %1121 = select i1 %.not, float %1109, float %1120, !dbg !11
  %1122 = select i1 %.not, float %1120, float %1110, !dbg !11
  %1123 = select i1 %.not36, float %1121, float %1115, !dbg !11
  %1124 = bitcast float %1123 to i32, !dbg !11
  %1125 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1124), !dbg !11
  %1126 = bitcast i32 %1125 to float, !dbg !11
  %1127 = select i1 %.not36, float %1122, float %1116, !dbg !11
  %1128 = bitcast float %1127 to i32, !dbg !11
  %1129 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1128), !dbg !11
  %1130 = bitcast i32 %1129 to float, !dbg !11
  %1131 = insertelement <4 x float> poison, float %1115, i64 0, !dbg !11
  %1132 = insertelement <4 x float> %1131, float %1116, i64 1, !dbg !11
  %1133 = insertelement <4 x float> %1132, float %1126, i64 2, !dbg !11
  %1134 = insertelement <4 x float> %1133, float %1130, i64 3, !dbg !11
  %1135 = insertelement <4 x float> poison, float %1126, i64 0, !dbg !11
  %1136 = insertelement <4 x float> %1135, float %1130, i64 1, !dbg !11
  %1137 = insertelement <4 x float> %1136, float %1121, i64 2, !dbg !11
  %1138 = insertelement <4 x float> %1137, float %1122, i64 3, !dbg !11
  %1139 = select <4 x i1> %1064, <4 x float> %1134, <4 x float> %1138, !dbg !11
  %1140 = extractelement <4 x float> %737, i64 0, !dbg !11
  %1141 = extractelement <4 x float> %737, i64 1, !dbg !11
  %1142 = extractelement <4 x float> %737, i64 2, !dbg !11
  %1143 = extractelement <4 x float> %737, i64 3, !dbg !11
  %1144 = select i1 %.not, float %1141, float %1140, !dbg !11
  %1145 = bitcast float %1144 to i32, !dbg !11
  %1146 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1145), !dbg !11
  %1147 = bitcast i32 %1146 to float, !dbg !11
  %1148 = select i1 %.not, float %1140, float %1147, !dbg !11
  %1149 = select i1 %.not, float %1147, float %1141, !dbg !11
  %1150 = select i1 %.not, float %1143, float %1142, !dbg !11
  %1151 = bitcast float %1150 to i32, !dbg !11
  %1152 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1151), !dbg !11
  %1153 = bitcast i32 %1152 to float, !dbg !11
  %1154 = select i1 %.not, float %1142, float %1153, !dbg !11
  %1155 = select i1 %.not, float %1153, float %1143, !dbg !11
  %1156 = select i1 %.not36, float %1154, float %1148, !dbg !11
  %1157 = bitcast float %1156 to i32, !dbg !11
  %1158 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1157), !dbg !11
  %1159 = select i1 %.not36, float %1155, float %1149, !dbg !11
  %1160 = bitcast float %1159 to i32, !dbg !11
  %1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1160), !dbg !11
  %1162 = shufflevector <1 x float> %1036, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %1163 = extractelement <4 x float> %981, i64 0, !dbg !11
  %1164 = extractelement <4 x float> %981, i64 1, !dbg !11
  %1165 = extractelement <4 x float> %981, i64 2, !dbg !11
  %1166 = extractelement <4 x float> %981, i64 3, !dbg !11
  %1167 = select i1 %.not, float %1164, float %1163, !dbg !11
  %1168 = bitcast float %1167 to i32, !dbg !11
  %1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1168), !dbg !11
  %1170 = bitcast i32 %1169 to float, !dbg !11
  %1171 = select i1 %.not, float %1163, float %1170, !dbg !11
  %1172 = select i1 %.not, float %1170, float %1164, !dbg !11
  %1173 = select i1 %.not, float %1166, float %1165, !dbg !11
  %1174 = bitcast float %1173 to i32, !dbg !11
  %1175 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1174), !dbg !11
  %1176 = bitcast i32 %1175 to float, !dbg !11
  %1177 = select i1 %.not, float %1165, float %1176, !dbg !11
  %1178 = select i1 %.not, float %1176, float %1166, !dbg !11
  %1179 = select i1 %.not36, float %1177, float %1171, !dbg !11
  %1180 = bitcast float %1179 to i32, !dbg !11
  %1181 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1180), !dbg !11
  %1182 = bitcast i32 %1181 to float, !dbg !11
  %1183 = select i1 %.not36, float %1178, float %1172, !dbg !11
  %1184 = bitcast float %1183 to i32, !dbg !11
  %1185 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1184), !dbg !11
  %1186 = bitcast i32 %1185 to float, !dbg !11
  %1187 = insertelement <4 x float> poison, float %1171, i64 0, !dbg !11
  %1188 = insertelement <4 x float> %1187, float %1172, i64 1, !dbg !11
  %1189 = insertelement <4 x float> %1188, float %1182, i64 2, !dbg !11
  %1190 = insertelement <4 x float> %1189, float %1186, i64 3, !dbg !11
  %1191 = insertelement <4 x float> poison, float %1182, i64 0, !dbg !11
  %1192 = insertelement <4 x float> %1191, float %1186, i64 1, !dbg !11
  %1193 = insertelement <4 x float> %1192, float %1177, i64 2, !dbg !11
  %1194 = insertelement <4 x float> %1193, float %1178, i64 3, !dbg !11
  %1195 = select <4 x i1> %1064, <4 x float> %1190, <4 x float> %1194, !dbg !11
  %1196 = extractelement <4 x float> %907, i64 0, !dbg !11
  %1197 = extractelement <4 x float> %907, i64 1, !dbg !11
  %1198 = extractelement <4 x float> %907, i64 2, !dbg !11
  %1199 = extractelement <4 x float> %907, i64 3, !dbg !11
  %1200 = select i1 %.not, float %1197, float %1196, !dbg !11
  %1201 = bitcast float %1200 to i32, !dbg !11
  %1202 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1201), !dbg !11
  %1203 = bitcast i32 %1202 to float, !dbg !11
  %1204 = select i1 %.not, float %1196, float %1203, !dbg !11
  %1205 = select i1 %.not, float %1203, float %1197, !dbg !11
  %1206 = select i1 %.not, float %1199, float %1198, !dbg !11
  %1207 = bitcast float %1206 to i32, !dbg !11
  %1208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1207), !dbg !11
  %1209 = bitcast i32 %1208 to float, !dbg !11
  %1210 = select i1 %.not, float %1198, float %1209, !dbg !11
  %1211 = select i1 %.not, float %1209, float %1199, !dbg !11
  %1212 = select i1 %.not36, float %1210, float %1204, !dbg !11
  %1213 = bitcast float %1212 to i32, !dbg !11
  %1214 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1213), !dbg !11
  %1215 = bitcast i32 %1214 to float, !dbg !11
  %1216 = select i1 %.not36, float %1211, float %1205, !dbg !11
  %1217 = bitcast float %1216 to i32, !dbg !11
  %1218 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1217), !dbg !11
  %1219 = bitcast i32 %1218 to float, !dbg !11
  %1220 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !11
  %1221 = insertelement <4 x float> %1220, float %1205, i64 1, !dbg !11
  %1222 = insertelement <4 x float> %1221, float %1215, i64 2, !dbg !11
  %1223 = insertelement <4 x float> %1222, float %1219, i64 3, !dbg !11
  %1224 = insertelement <4 x float> poison, float %1215, i64 0, !dbg !11
  %1225 = insertelement <4 x float> %1224, float %1219, i64 1, !dbg !11
  %1226 = insertelement <4 x float> %1225, float %1210, i64 2, !dbg !11
  %1227 = insertelement <4 x float> %1226, float %1211, i64 3, !dbg !11
  %1228 = select <4 x i1> %1064, <4 x float> %1223, <4 x float> %1227, !dbg !11
  %1229 = extractelement <4 x float> %831, i64 0, !dbg !11
  %1230 = extractelement <4 x float> %831, i64 1, !dbg !11
  %1231 = extractelement <4 x float> %831, i64 2, !dbg !11
  %1232 = extractelement <4 x float> %831, i64 3, !dbg !11
  %1233 = select i1 %.not, float %1230, float %1229, !dbg !11
  %1234 = bitcast float %1233 to i32, !dbg !11
  %1235 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1234), !dbg !11
  %1236 = bitcast i32 %1235 to float, !dbg !11
  %1237 = select i1 %.not, float %1229, float %1236, !dbg !11
  %1238 = select i1 %.not, float %1236, float %1230, !dbg !11
  %1239 = select i1 %.not, float %1232, float %1231, !dbg !11
  %1240 = bitcast float %1239 to i32, !dbg !11
  %1241 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1240), !dbg !11
  %1242 = bitcast i32 %1241 to float, !dbg !11
  %1243 = select i1 %.not, float %1231, float %1242, !dbg !11
  %1244 = select i1 %.not, float %1242, float %1232, !dbg !11
  %1245 = select i1 %.not36, float %1243, float %1237, !dbg !11
  %1246 = bitcast float %1245 to i32, !dbg !11
  %1247 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1246), !dbg !11
  %1248 = bitcast i32 %1247 to float, !dbg !11
  %1249 = select i1 %.not36, float %1244, float %1238, !dbg !11
  %1250 = bitcast float %1249 to i32, !dbg !11
  %1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1250), !dbg !11
  %1252 = bitcast i32 %1251 to float, !dbg !11
  %1253 = insertelement <4 x float> poison, float %1237, i64 0, !dbg !11
  %1254 = insertelement <4 x float> %1253, float %1238, i64 1, !dbg !11
  %1255 = insertelement <4 x float> %1254, float %1248, i64 2, !dbg !11
  %1256 = insertelement <4 x float> %1255, float %1252, i64 3, !dbg !11
  %1257 = insertelement <4 x float> poison, float %1248, i64 0, !dbg !11
  %1258 = insertelement <4 x float> %1257, float %1252, i64 1, !dbg !11
  %1259 = insertelement <4 x float> %1258, float %1243, i64 2, !dbg !11
  %1260 = insertelement <4 x float> %1259, float %1244, i64 3, !dbg !11
  %1261 = select <4 x i1> %1064, <4 x float> %1256, <4 x float> %1260, !dbg !11
  %1262 = extractelement <4 x float> %755, i64 0, !dbg !11
  %1263 = extractelement <4 x float> %755, i64 1, !dbg !11
  %1264 = extractelement <4 x float> %755, i64 2, !dbg !11
  %1265 = extractelement <4 x float> %755, i64 3, !dbg !11
  %1266 = select i1 %.not, float %1263, float %1262, !dbg !11
  %1267 = bitcast float %1266 to i32, !dbg !11
  %1268 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1267), !dbg !11
  %1269 = bitcast i32 %1268 to float, !dbg !11
  %1270 = select i1 %.not, float %1262, float %1269, !dbg !11
  %1271 = select i1 %.not, float %1269, float %1263, !dbg !11
  %1272 = select i1 %.not, float %1265, float %1264, !dbg !11
  %1273 = bitcast float %1272 to i32, !dbg !11
  %1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1273), !dbg !11
  %1275 = bitcast i32 %1274 to float, !dbg !11
  %1276 = select i1 %.not, float %1264, float %1275, !dbg !11
  %1277 = select i1 %.not, float %1275, float %1265, !dbg !11
  %1278 = select i1 %.not36, float %1276, float %1270, !dbg !11
  %1279 = bitcast float %1278 to i32, !dbg !11
  %1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1279), !dbg !11
  %1281 = select i1 %.not36, float %1277, float %1271, !dbg !11
  %1282 = bitcast float %1281 to i32, !dbg !11
  %1283 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1282), !dbg !11
  %1284 = extractelement <4 x float> %992, i64 0, !dbg !11
  %1285 = extractelement <4 x float> %992, i64 1, !dbg !11
  %1286 = extractelement <4 x float> %992, i64 2, !dbg !11
  %1287 = extractelement <4 x float> %992, i64 3, !dbg !11
  %1288 = select i1 %.not, float %1285, float %1284, !dbg !11
  %1289 = bitcast float %1288 to i32, !dbg !11
  %1290 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1289), !dbg !11
  %1291 = bitcast i32 %1290 to float, !dbg !11
  %1292 = select i1 %.not, float %1284, float %1291, !dbg !11
  %1293 = select i1 %.not, float %1291, float %1285, !dbg !11
  %1294 = select i1 %.not, float %1287, float %1286, !dbg !11
  %1295 = bitcast float %1294 to i32, !dbg !11
  %1296 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1295), !dbg !11
  %1297 = bitcast i32 %1296 to float, !dbg !11
  %1298 = select i1 %.not, float %1286, float %1297, !dbg !11
  %1299 = select i1 %.not, float %1297, float %1287, !dbg !11
  %1300 = select i1 %.not36, float %1298, float %1292, !dbg !11
  %1301 = bitcast float %1300 to i32, !dbg !11
  %1302 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1301), !dbg !11
  %1303 = bitcast i32 %1302 to float, !dbg !11
  %1304 = select i1 %.not36, float %1299, float %1293, !dbg !11
  %1305 = bitcast float %1304 to i32, !dbg !11
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1305), !dbg !11
  %1307 = bitcast i32 %1306 to float, !dbg !11
  %1308 = insertelement <4 x float> poison, float %1292, i64 0, !dbg !11
  %1309 = insertelement <4 x float> %1308, float %1293, i64 1, !dbg !11
  %1310 = insertelement <4 x float> %1309, float %1303, i64 2, !dbg !11
  %1311 = insertelement <4 x float> %1310, float %1307, i64 3, !dbg !11
  %1312 = insertelement <4 x float> poison, float %1303, i64 0, !dbg !11
  %1313 = insertelement <4 x float> %1312, float %1307, i64 1, !dbg !11
  %1314 = insertelement <4 x float> %1313, float %1298, i64 2, !dbg !11
  %1315 = insertelement <4 x float> %1314, float %1299, i64 3, !dbg !11
  %1316 = select <4 x i1> %1064, <4 x float> %1311, <4 x float> %1315, !dbg !11
  %1317 = extractelement <4 x float> %918, i64 0, !dbg !11
  %1318 = extractelement <4 x float> %918, i64 1, !dbg !11
  %1319 = extractelement <4 x float> %918, i64 2, !dbg !11
  %1320 = extractelement <4 x float> %918, i64 3, !dbg !11
  %1321 = select i1 %.not, float %1318, float %1317, !dbg !11
  %1322 = bitcast float %1321 to i32, !dbg !11
  %1323 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1322), !dbg !11
  %1324 = bitcast i32 %1323 to float, !dbg !11
  %1325 = select i1 %.not, float %1317, float %1324, !dbg !11
  %1326 = select i1 %.not, float %1324, float %1318, !dbg !11
  %1327 = select i1 %.not, float %1320, float %1319, !dbg !11
  %1328 = bitcast float %1327 to i32, !dbg !11
  %1329 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1328), !dbg !11
  %1330 = bitcast i32 %1329 to float, !dbg !11
  %1331 = select i1 %.not, float %1319, float %1330, !dbg !11
  %1332 = select i1 %.not, float %1330, float %1320, !dbg !11
  %1333 = select i1 %.not36, float %1331, float %1325, !dbg !11
  %1334 = bitcast float %1333 to i32, !dbg !11
  %1335 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1334), !dbg !11
  %1336 = bitcast i32 %1335 to float, !dbg !11
  %1337 = select i1 %.not36, float %1332, float %1326, !dbg !11
  %1338 = bitcast float %1337 to i32, !dbg !11
  %1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1338), !dbg !11
  %1340 = bitcast i32 %1339 to float, !dbg !11
  %1341 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !11
  %1342 = insertelement <4 x float> %1341, float %1326, i64 1, !dbg !11
  %1343 = insertelement <4 x float> %1342, float %1336, i64 2, !dbg !11
  %1344 = insertelement <4 x float> %1343, float %1340, i64 3, !dbg !11
  %1345 = insertelement <4 x float> poison, float %1336, i64 0, !dbg !11
  %1346 = insertelement <4 x float> %1345, float %1340, i64 1, !dbg !11
  %1347 = insertelement <4 x float> %1346, float %1331, i64 2, !dbg !11
  %1348 = insertelement <4 x float> %1347, float %1332, i64 3, !dbg !11
  %1349 = select <4 x i1> %1064, <4 x float> %1344, <4 x float> %1348, !dbg !11
  %1350 = extractelement <4 x float> %842, i64 0, !dbg !11
  %1351 = extractelement <4 x float> %842, i64 1, !dbg !11
  %1352 = extractelement <4 x float> %842, i64 2, !dbg !11
  %1353 = extractelement <4 x float> %842, i64 3, !dbg !11
  %1354 = select i1 %.not, float %1351, float %1350, !dbg !11
  %1355 = bitcast float %1354 to i32, !dbg !11
  %1356 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1355), !dbg !11
  %1357 = bitcast i32 %1356 to float, !dbg !11
  %1358 = select i1 %.not, float %1350, float %1357, !dbg !11
  %1359 = select i1 %.not, float %1357, float %1351, !dbg !11
  %1360 = select i1 %.not, float %1353, float %1352, !dbg !11
  %1361 = bitcast float %1360 to i32, !dbg !11
  %1362 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1361), !dbg !11
  %1363 = bitcast i32 %1362 to float, !dbg !11
  %1364 = select i1 %.not, float %1352, float %1363, !dbg !11
  %1365 = select i1 %.not, float %1363, float %1353, !dbg !11
  %1366 = select i1 %.not36, float %1364, float %1358, !dbg !11
  %1367 = bitcast float %1366 to i32, !dbg !11
  %1368 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1367), !dbg !11
  %1369 = bitcast i32 %1368 to float, !dbg !11
  %1370 = select i1 %.not36, float %1365, float %1359, !dbg !11
  %1371 = bitcast float %1370 to i32, !dbg !11
  %1372 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1371), !dbg !11
  %1373 = bitcast i32 %1372 to float, !dbg !11
  %1374 = insertelement <4 x float> poison, float %1358, i64 0, !dbg !11
  %1375 = insertelement <4 x float> %1374, float %1359, i64 1, !dbg !11
  %1376 = insertelement <4 x float> %1375, float %1369, i64 2, !dbg !11
  %1377 = insertelement <4 x float> %1376, float %1373, i64 3, !dbg !11
  %1378 = insertelement <4 x float> poison, float %1369, i64 0, !dbg !11
  %1379 = insertelement <4 x float> %1378, float %1373, i64 1, !dbg !11
  %1380 = insertelement <4 x float> %1379, float %1364, i64 2, !dbg !11
  %1381 = insertelement <4 x float> %1380, float %1365, i64 3, !dbg !11
  %1382 = select <4 x i1> %1064, <4 x float> %1377, <4 x float> %1381, !dbg !11
  %1383 = extractelement <4 x float> %766, i64 0, !dbg !11
  %1384 = extractelement <4 x float> %766, i64 1, !dbg !11
  %1385 = extractelement <4 x float> %766, i64 2, !dbg !11
  %1386 = extractelement <4 x float> %766, i64 3, !dbg !11
  %1387 = select i1 %.not, float %1384, float %1383, !dbg !11
  %1388 = bitcast float %1387 to i32, !dbg !11
  %1389 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1388), !dbg !11
  %1390 = bitcast i32 %1389 to float, !dbg !11
  %1391 = select i1 %.not, float %1383, float %1390, !dbg !11
  %1392 = select i1 %.not, float %1390, float %1384, !dbg !11
  %1393 = select i1 %.not, float %1386, float %1385, !dbg !11
  %1394 = bitcast float %1393 to i32, !dbg !11
  %1395 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1394), !dbg !11
  %1396 = bitcast i32 %1395 to float, !dbg !11
  %1397 = select i1 %.not, float %1385, float %1396, !dbg !11
  %1398 = select i1 %.not, float %1396, float %1386, !dbg !11
  %1399 = select i1 %.not36, float %1397, float %1391, !dbg !11
  %1400 = bitcast float %1399 to i32, !dbg !11
  %1401 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1400), !dbg !11
  %1402 = select i1 %.not36, float %1398, float %1392, !dbg !11
  %1403 = bitcast float %1402 to i32, !dbg !11
  %1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1403), !dbg !11
  %1405 = shufflevector <1 x float> %1031, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !26
  %1406 = extractelement <4 x float> %1002, i64 0, !dbg !11
  %1407 = extractelement <4 x float> %1002, i64 1, !dbg !11
  %1408 = extractelement <4 x float> %1002, i64 2, !dbg !11
  %1409 = extractelement <4 x float> %1002, i64 3, !dbg !11
  %1410 = select i1 %.not, float %1407, float %1406, !dbg !11
  %1411 = bitcast float %1410 to i32, !dbg !11
  %1412 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1411), !dbg !11
  %1413 = bitcast i32 %1412 to float, !dbg !11
  %1414 = select i1 %.not, float %1406, float %1413, !dbg !11
  %1415 = select i1 %.not, float %1413, float %1407, !dbg !11
  %1416 = select i1 %.not, float %1409, float %1408, !dbg !11
  %1417 = bitcast float %1416 to i32, !dbg !11
  %1418 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1417), !dbg !11
  %1419 = bitcast i32 %1418 to float, !dbg !11
  %1420 = select i1 %.not, float %1408, float %1419, !dbg !11
  %1421 = select i1 %.not, float %1419, float %1409, !dbg !11
  %1422 = select i1 %.not36, float %1420, float %1414, !dbg !11
  %1423 = bitcast float %1422 to i32, !dbg !11
  %1424 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1423), !dbg !11
  %1425 = bitcast i32 %1424 to float, !dbg !11
  %1426 = select i1 %.not36, float %1421, float %1415, !dbg !11
  %1427 = bitcast float %1426 to i32, !dbg !11
  %1428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1427), !dbg !11
  %1429 = bitcast i32 %1428 to float, !dbg !11
  %1430 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !11
  %1431 = insertelement <4 x float> %1430, float %1415, i64 1, !dbg !11
  %1432 = insertelement <4 x float> %1431, float %1425, i64 2, !dbg !11
  %1433 = insertelement <4 x float> %1432, float %1429, i64 3, !dbg !11
  %1434 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !11
  %1435 = insertelement <4 x float> %1434, float %1429, i64 1, !dbg !11
  %1436 = insertelement <4 x float> %1435, float %1420, i64 2, !dbg !11
  %1437 = insertelement <4 x float> %1436, float %1421, i64 3, !dbg !11
  %1438 = select <4 x i1> %1064, <4 x float> %1433, <4 x float> %1437, !dbg !11
  %1439 = extractelement <4 x float> %928, i64 0, !dbg !11
  %1440 = extractelement <4 x float> %928, i64 1, !dbg !11
  %1441 = extractelement <4 x float> %928, i64 2, !dbg !11
  %1442 = extractelement <4 x float> %928, i64 3, !dbg !11
  %1443 = select i1 %.not, float %1440, float %1439, !dbg !11
  %1444 = bitcast float %1443 to i32, !dbg !11
  %1445 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1444), !dbg !11
  %1446 = bitcast i32 %1445 to float, !dbg !11
  %1447 = select i1 %.not, float %1439, float %1446, !dbg !11
  %1448 = select i1 %.not, float %1446, float %1440, !dbg !11
  %1449 = select i1 %.not, float %1442, float %1441, !dbg !11
  %1450 = bitcast float %1449 to i32, !dbg !11
  %1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1450), !dbg !11
  %1452 = bitcast i32 %1451 to float, !dbg !11
  %1453 = select i1 %.not, float %1441, float %1452, !dbg !11
  %1454 = select i1 %.not, float %1452, float %1442, !dbg !11
  %1455 = select i1 %.not36, float %1453, float %1447, !dbg !11
  %1456 = bitcast float %1455 to i32, !dbg !11
  %1457 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1456), !dbg !11
  %1458 = bitcast i32 %1457 to float, !dbg !11
  %1459 = select i1 %.not36, float %1454, float %1448, !dbg !11
  %1460 = bitcast float %1459 to i32, !dbg !11
  %1461 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1460), !dbg !11
  %1462 = bitcast i32 %1461 to float, !dbg !11
  %1463 = insertelement <4 x float> poison, float %1447, i64 0, !dbg !11
  %1464 = insertelement <4 x float> %1463, float %1448, i64 1, !dbg !11
  %1465 = insertelement <4 x float> %1464, float %1458, i64 2, !dbg !11
  %1466 = insertelement <4 x float> %1465, float %1462, i64 3, !dbg !11
  %1467 = insertelement <4 x float> poison, float %1458, i64 0, !dbg !11
  %1468 = insertelement <4 x float> %1467, float %1462, i64 1, !dbg !11
  %1469 = insertelement <4 x float> %1468, float %1453, i64 2, !dbg !11
  %1470 = insertelement <4 x float> %1469, float %1454, i64 3, !dbg !11
  %1471 = select <4 x i1> %1064, <4 x float> %1466, <4 x float> %1470, !dbg !11
  %1472 = extractelement <4 x float> %852, i64 0, !dbg !11
  %1473 = extractelement <4 x float> %852, i64 1, !dbg !11
  %1474 = extractelement <4 x float> %852, i64 2, !dbg !11
  %1475 = extractelement <4 x float> %852, i64 3, !dbg !11
  %1476 = select i1 %.not, float %1473, float %1472, !dbg !11
  %1477 = bitcast float %1476 to i32, !dbg !11
  %1478 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1477), !dbg !11
  %1479 = bitcast i32 %1478 to float, !dbg !11
  %1480 = select i1 %.not, float %1472, float %1479, !dbg !11
  %1481 = select i1 %.not, float %1479, float %1473, !dbg !11
  %1482 = select i1 %.not, float %1475, float %1474, !dbg !11
  %1483 = bitcast float %1482 to i32, !dbg !11
  %1484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1483), !dbg !11
  %1485 = bitcast i32 %1484 to float, !dbg !11
  %1486 = select i1 %.not, float %1474, float %1485, !dbg !11
  %1487 = select i1 %.not, float %1485, float %1475, !dbg !11
  %1488 = select i1 %.not36, float %1486, float %1480, !dbg !11
  %1489 = bitcast float %1488 to i32, !dbg !11
  %1490 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1489), !dbg !11
  %1491 = bitcast i32 %1490 to float, !dbg !11
  %1492 = select i1 %.not36, float %1487, float %1481, !dbg !11
  %1493 = bitcast float %1492 to i32, !dbg !11
  %1494 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1493), !dbg !11
  %1495 = bitcast i32 %1494 to float, !dbg !11
  %1496 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !11
  %1497 = insertelement <4 x float> %1496, float %1481, i64 1, !dbg !11
  %1498 = insertelement <4 x float> %1497, float %1491, i64 2, !dbg !11
  %1499 = insertelement <4 x float> %1498, float %1495, i64 3, !dbg !11
  %1500 = insertelement <4 x float> poison, float %1491, i64 0, !dbg !11
  %1501 = insertelement <4 x float> %1500, float %1495, i64 1, !dbg !11
  %1502 = insertelement <4 x float> %1501, float %1486, i64 2, !dbg !11
  %1503 = insertelement <4 x float> %1502, float %1487, i64 3, !dbg !11
  %1504 = select <4 x i1> %1064, <4 x float> %1499, <4 x float> %1503, !dbg !11
  %1505 = extractelement <4 x float> %776, i64 0, !dbg !11
  %1506 = extractelement <4 x float> %776, i64 1, !dbg !11
  %1507 = extractelement <4 x float> %776, i64 2, !dbg !11
  %1508 = extractelement <4 x float> %776, i64 3, !dbg !11
  %1509 = select i1 %.not, float %1506, float %1505, !dbg !11
  %1510 = bitcast float %1509 to i32, !dbg !11
  %1511 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1510), !dbg !11
  %1512 = bitcast i32 %1511 to float, !dbg !11
  %1513 = select i1 %.not, float %1505, float %1512, !dbg !11
  %1514 = select i1 %.not, float %1512, float %1506, !dbg !11
  %1515 = select i1 %.not, float %1508, float %1507, !dbg !11
  %1516 = bitcast float %1515 to i32, !dbg !11
  %1517 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %422, i32 %1516), !dbg !11
  %1518 = bitcast i32 %1517 to float, !dbg !11
  %1519 = select i1 %.not, float %1507, float %1518, !dbg !11
  %1520 = select i1 %.not, float %1518, float %1508, !dbg !11
  %1521 = select i1 %.not36, float %1519, float %1513, !dbg !11
  %1522 = bitcast float %1521 to i32, !dbg !11
  %1523 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1522), !dbg !11
  %1524 = select i1 %.not36, float %1520, float %1514, !dbg !11
  %1525 = bitcast float %1524 to i32, !dbg !11
  %1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %430, i32 %1525), !dbg !11
  %1527 = bitcast i32 %1523 to float, !dbg !11
  %1528 = bitcast i32 %1526 to float, !dbg !11
  %1529 = insertelement <4 x float> poison, float %1513, i64 0, !dbg !11
  %1530 = insertelement <4 x float> %1529, float %1514, i64 1, !dbg !11
  %1531 = insertelement <4 x float> %1530, float %1527, i64 2, !dbg !11
  %1532 = insertelement <4 x float> %1531, float %1528, i64 3, !dbg !11
  %1533 = insertelement <4 x float> poison, float %1527, i64 0, !dbg !11
  %1534 = insertelement <4 x float> %1533, float %1528, i64 1, !dbg !11
  %1535 = insertelement <4 x float> %1534, float %1519, i64 2, !dbg !11
  %1536 = insertelement <4 x float> %1535, float %1520, i64 3, !dbg !11
  %1537 = select <4 x i1> %1064, <4 x float> %1532, <4 x float> %1536, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1405, !dbg !26
  %normalized = fmul <4 x float> %row.reciprocal, %1537, !dbg !26
  %1538 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !27
  %1539 = extractelement <4 x half> %1538, i64 3, !dbg !27
  %1540 = extractelement <4 x half> %1538, i64 2, !dbg !27
  %1541 = extractelement <4 x half> %1538, i64 1, !dbg !27
  %1542 = extractelement <4 x half> %1538, i64 0, !dbg !27
  %1543 = bitcast i32 %1401 to float, !dbg !11
  %1544 = bitcast i32 %1404 to float, !dbg !11
  %1545 = insertelement <4 x float> poison, float %1391, i64 0, !dbg !11
  %1546 = insertelement <4 x float> %1545, float %1392, i64 1, !dbg !11
  %1547 = insertelement <4 x float> %1546, float %1543, i64 2, !dbg !11
  %1548 = insertelement <4 x float> %1547, float %1544, i64 3, !dbg !11
  %1549 = insertelement <4 x float> poison, float %1543, i64 0, !dbg !11
  %1550 = insertelement <4 x float> %1549, float %1544, i64 1, !dbg !11
  %1551 = insertelement <4 x float> %1550, float %1397, i64 2, !dbg !11
  %1552 = insertelement <4 x float> %1551, float %1398, i64 3, !dbg !11
  %1553 = select <4 x i1> %1064, <4 x float> %1548, <4 x float> %1552, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %1553, !dbg !26
  %1554 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !27
  %1555 = extractelement <4 x half> %1554, i64 3, !dbg !27
  %1556 = extractelement <4 x half> %1554, i64 2, !dbg !27
  %1557 = extractelement <4 x half> %1554, i64 1, !dbg !27
  %1558 = extractelement <4 x half> %1554, i64 0, !dbg !27
  %1559 = bitcast i32 %1280 to float, !dbg !11
  %1560 = bitcast i32 %1283 to float, !dbg !11
  %1561 = insertelement <4 x float> poison, float %1270, i64 0, !dbg !11
  %1562 = insertelement <4 x float> %1561, float %1271, i64 1, !dbg !11
  %1563 = insertelement <4 x float> %1562, float %1559, i64 2, !dbg !11
  %1564 = insertelement <4 x float> %1563, float %1560, i64 3, !dbg !11
  %1565 = insertelement <4 x float> poison, float %1559, i64 0, !dbg !11
  %1566 = insertelement <4 x float> %1565, float %1560, i64 1, !dbg !11
  %1567 = insertelement <4 x float> %1566, float %1276, i64 2, !dbg !11
  %1568 = insertelement <4 x float> %1567, float %1277, i64 3, !dbg !11
  %1569 = select <4 x i1> %1064, <4 x float> %1564, <4 x float> %1568, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1162, !dbg !26
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1569, !dbg !26
  %1570 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !27
  %1571 = extractelement <4 x half> %1570, i64 3, !dbg !27
  %1572 = extractelement <4 x half> %1570, i64 2, !dbg !27
  %1573 = extractelement <4 x half> %1570, i64 1, !dbg !27
  %1574 = extractelement <4 x half> %1570, i64 0, !dbg !27
  %1575 = bitcast i32 %1158 to float, !dbg !11
  %1576 = bitcast i32 %1161 to float, !dbg !11
  %1577 = insertelement <4 x float> poison, float %1148, i64 0, !dbg !11
  %1578 = insertelement <4 x float> %1577, float %1149, i64 1, !dbg !11
  %1579 = insertelement <4 x float> %1578, float %1575, i64 2, !dbg !11
  %1580 = insertelement <4 x float> %1579, float %1576, i64 3, !dbg !11
  %1581 = insertelement <4 x float> poison, float %1575, i64 0, !dbg !11
  %1582 = insertelement <4 x float> %1581, float %1576, i64 1, !dbg !11
  %1583 = insertelement <4 x float> %1582, float %1154, i64 2, !dbg !11
  %1584 = insertelement <4 x float> %1583, float %1155, i64 3, !dbg !11
  %1585 = select <4 x i1> %1064, <4 x float> %1580, <4 x float> %1584, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1585, !dbg !26
  %1586 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !27
  %1587 = extractelement <4 x half> %1586, i64 3, !dbg !27
  %1588 = extractelement <4 x half> %1586, i64 2, !dbg !27
  %1589 = extractelement <4 x half> %1586, i64 1, !dbg !27
  %1590 = extractelement <4 x half> %1586, i64 0, !dbg !27
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %31) #8, !dbg !28, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %1591 = lshr i64 %address.local975, 2, !dbg !28
  %address.local978 = and i64 %1591, 48, !dbg !28
  %address.local980 = and i64 %address.local975, 240, !dbg !28
  %address.local979 = add i64 %26, %address.local975, !dbg !28
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !28
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !28
  %1592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %28, !dbg !28
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !28
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %1592, i64 %address.local985.idx, !dbg !28
  %address.local987 = and i64 %1591, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !28
  store half %1590, ptr addrspace(1) %address.local988, align 2, !dbg !28
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !28
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !28
  store half %1589, ptr addrspace(1) %address.local991, align 2, !dbg !28
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !28
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !28
  store half %1588, ptr addrspace(1) %address.local994, align 2, !dbg !28
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !28
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !28
  store half %1587, ptr addrspace(1) %address.local997, align 2, !dbg !28
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !28
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !28
  store half %1574, ptr addrspace(1) %address.local1000, align 2, !dbg !28
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !28
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !28
  store half %1573, ptr addrspace(1) %address.local1003, align 2, !dbg !28
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !28
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !28
  store half %1572, ptr addrspace(1) %address.local1006, align 2, !dbg !28
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !28
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !28
  store half %1571, ptr addrspace(1) %address.local1009, align 2, !dbg !28
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !28
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !28
  store half %1558, ptr addrspace(1) %address.local1014, align 2, !dbg !28
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !28
  store half %1557, ptr addrspace(1) %address.local1016, align 2, !dbg !28
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !28
  store half %1556, ptr addrspace(1) %address.local1018, align 2, !dbg !28
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !28
  store half %1555, ptr addrspace(1) %address.local1020, align 2, !dbg !28
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !28
  store half %1542, ptr addrspace(1) %address.local1022, align 2, !dbg !28
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !28
  store half %1541, ptr addrspace(1) %address.local1024, align 2, !dbg !28
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !28
  store half %1540, ptr addrspace(1) %address.local1026, align 2, !dbg !28
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !28
  store half %1539, ptr addrspace(1) %address.local1028, align 2, !dbg !28
  %normalized42 = fmul <4 x float> %row.reciprocal, %1504, !dbg !26
  %1593 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !27
  %1594 = extractelement <4 x half> %1593, i64 3, !dbg !27
  %1595 = extractelement <4 x half> %1593, i64 2, !dbg !27
  %1596 = extractelement <4 x half> %1593, i64 1, !dbg !27
  %1597 = extractelement <4 x half> %1593, i64 0, !dbg !27
  %normalized43 = fmul <4 x float> %row.reciprocal, %1382, !dbg !26
  %1598 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !27
  %1599 = extractelement <4 x half> %1598, i64 3, !dbg !27
  %1600 = extractelement <4 x half> %1598, i64 2, !dbg !27
  %1601 = extractelement <4 x half> %1598, i64 1, !dbg !27
  %1602 = extractelement <4 x half> %1598, i64 0, !dbg !27
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1261, !dbg !26
  %1603 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !27
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !27
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !27
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !27
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !27
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1139, !dbg !26
  %1608 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !27
  %1609 = extractelement <4 x half> %1608, i64 3, !dbg !27
  %1610 = extractelement <4 x half> %1608, i64 2, !dbg !27
  %1611 = extractelement <4 x half> %1608, i64 1, !dbg !27
  %1612 = extractelement <4 x half> %1608, i64 0, !dbg !27
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !28
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !28
  store half %1612, ptr addrspace(1) %address.local1044, align 2, !dbg !28
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !28
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !28
  store half %1611, ptr addrspace(1) %address.local1047, align 2, !dbg !28
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !28
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !28
  store half %1610, ptr addrspace(1) %address.local1050, align 2, !dbg !28
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !28
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !28
  store half %1609, ptr addrspace(1) %address.local1053, align 2, !dbg !28
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !28
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !28
  store half %1607, ptr addrspace(1) %address.local1056, align 2, !dbg !28
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !28
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !28
  store half %1606, ptr addrspace(1) %address.local1059, align 2, !dbg !28
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !28
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !28
  store half %1605, ptr addrspace(1) %address.local1062, align 2, !dbg !28
  %address.local1064 = or i64 %1591, 51, !dbg !28
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !28
  store half %1604, ptr addrspace(1) %address.local1065, align 2, !dbg !28
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !28
  store half %1602, ptr addrspace(1) %address.local1070, align 2, !dbg !28
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !28
  store half %1601, ptr addrspace(1) %address.local1072, align 2, !dbg !28
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !28
  store half %1600, ptr addrspace(1) %address.local1074, align 2, !dbg !28
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !28
  store half %1599, ptr addrspace(1) %address.local1076, align 2, !dbg !28
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !28
  store half %1597, ptr addrspace(1) %address.local1078, align 2, !dbg !28
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !28
  store half %1596, ptr addrspace(1) %address.local1080, align 2, !dbg !28
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !28
  store half %1595, ptr addrspace(1) %address.local1082, align 2, !dbg !28
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !28
  store half %1594, ptr addrspace(1) %address.local1084, align 2, !dbg !28
  %normalized48 = fmul <4 x float> %row.reciprocal, %1471, !dbg !26
  %1613 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !27
  %1614 = extractelement <4 x half> %1613, i64 3, !dbg !27
  %1615 = extractelement <4 x half> %1613, i64 2, !dbg !27
  %1616 = extractelement <4 x half> %1613, i64 1, !dbg !27
  %1617 = extractelement <4 x half> %1613, i64 0, !dbg !27
  %normalized49 = fmul <4 x float> %row.reciprocal, %1349, !dbg !26
  %1618 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !27
  %1619 = extractelement <4 x half> %1618, i64 3, !dbg !27
  %1620 = extractelement <4 x half> %1618, i64 2, !dbg !27
  %1621 = extractelement <4 x half> %1618, i64 1, !dbg !27
  %1622 = extractelement <4 x half> %1618, i64 0, !dbg !27
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1228, !dbg !26
  %1623 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !27
  %1624 = extractelement <4 x half> %1623, i64 3, !dbg !27
  %1625 = extractelement <4 x half> %1623, i64 2, !dbg !27
  %1626 = extractelement <4 x half> %1623, i64 1, !dbg !27
  %1627 = extractelement <4 x half> %1623, i64 0, !dbg !27
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1106, !dbg !26
  %1628 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !27
  %1629 = extractelement <4 x half> %1628, i64 3, !dbg !27
  %1630 = extractelement <4 x half> %1628, i64 2, !dbg !27
  %1631 = extractelement <4 x half> %1628, i64 1, !dbg !27
  %1632 = extractelement <4 x half> %1628, i64 0, !dbg !27
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !28
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !28
  store half %1632, ptr addrspace(1) %address.local1100, align 2, !dbg !28
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !28
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !28
  store half %1631, ptr addrspace(1) %address.local1103, align 2, !dbg !28
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !28
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !28
  store half %1630, ptr addrspace(1) %address.local1106, align 2, !dbg !28
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !28
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !28
  store half %1629, ptr addrspace(1) %address.local1109, align 2, !dbg !28
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !28
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !28
  store half %1627, ptr addrspace(1) %address.local1112, align 2, !dbg !28
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !28
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !28
  store half %1626, ptr addrspace(1) %address.local1115, align 2, !dbg !28
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !28
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !28
  store half %1625, ptr addrspace(1) %address.local1118, align 2, !dbg !28
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !28
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !28
  store half %1624, ptr addrspace(1) %address.local1121, align 2, !dbg !28
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !28
  store half %1622, ptr addrspace(1) %address.local1126, align 2, !dbg !28
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !28
  store half %1621, ptr addrspace(1) %address.local1128, align 2, !dbg !28
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !28
  store half %1620, ptr addrspace(1) %address.local1130, align 2, !dbg !28
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !28
  store half %1619, ptr addrspace(1) %address.local1132, align 2, !dbg !28
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !28
  store half %1617, ptr addrspace(1) %address.local1134, align 2, !dbg !28
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !28
  store half %1616, ptr addrspace(1) %address.local1136, align 2, !dbg !28
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !28
  store half %1615, ptr addrspace(1) %address.local1138, align 2, !dbg !28
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !28
  store half %1614, ptr addrspace(1) %address.local1140, align 2, !dbg !28
  %normalized54 = fmul <4 x float> %row.reciprocal, %1438, !dbg !26
  %1633 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !27
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !27
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !27
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !27
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !27
  %normalized55 = fmul <4 x float> %row.reciprocal, %1316, !dbg !26
  %1638 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !27
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !27
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !27
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !27
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !27
  %normalized57 = fmul <4 x float> %row.reciprocal38, %1195, !dbg !26
  %1643 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !27
  %1644 = extractelement <4 x half> %1643, i64 3, !dbg !27
  %1645 = extractelement <4 x half> %1643, i64 2, !dbg !27
  %1646 = extractelement <4 x half> %1643, i64 1, !dbg !27
  %1647 = extractelement <4 x half> %1643, i64 0, !dbg !27
  %normalized58 = fmul <4 x float> %row.reciprocal38, %1073, !dbg !26
  %1648 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !27
  %1649 = extractelement <4 x half> %1648, i64 3, !dbg !27
  %1650 = extractelement <4 x half> %1648, i64 2, !dbg !27
  %1651 = extractelement <4 x half> %1648, i64 1, !dbg !27
  %1652 = extractelement <4 x half> %1648, i64 0, !dbg !27
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !28
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !28
  store half %1652, ptr addrspace(1) %address.local1156, align 2, !dbg !28
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !28
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !28
  store half %1651, ptr addrspace(1) %address.local1159, align 2, !dbg !28
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !28
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !28
  store half %1650, ptr addrspace(1) %address.local1162, align 2, !dbg !28
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !28
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !28
  store half %1649, ptr addrspace(1) %address.local1165, align 2, !dbg !28
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !28
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !28
  store half %1647, ptr addrspace(1) %address.local1168, align 2, !dbg !28
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !28
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !28
  store half %1646, ptr addrspace(1) %address.local1171, align 2, !dbg !28
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !28
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !28
  store half %1645, ptr addrspace(1) %address.local1174, align 2, !dbg !28
  %address.local1176 = or i64 %1591, 115, !dbg !28
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !28
  store half %1644, ptr addrspace(1) %address.local1177, align 2, !dbg !28
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !28
  store half %1642, ptr addrspace(1) %address.local1182, align 2, !dbg !28
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !28
  store half %1641, ptr addrspace(1) %address.local1184, align 2, !dbg !28
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !28
  store half %1640, ptr addrspace(1) %address.local1186, align 2, !dbg !28
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !28
  store half %1639, ptr addrspace(1) %address.local1188, align 2, !dbg !28
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !28
  store half %1637, ptr addrspace(1) %address.local1190, align 2, !dbg !28
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !28
  store half %1636, ptr addrspace(1) %address.local1192, align 2, !dbg !28
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !28
  store half %1635, ptr addrspace(1) %address.local1194, align 2, !dbg !28
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !28
  store half %1634, ptr addrspace(1) %address.local1196, align 2, !dbg !28
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

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_optimization/coalesce/run-20261027-164012-mybq6yab/online-baseline")
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
!26 = !DILocation(line: 92, column: 23, scope: !3)
!27 = !DILocation(line: 93, column: 19, scope: !3)
!28 = !DILocation(line: 94, column: 9, scope: !3)
