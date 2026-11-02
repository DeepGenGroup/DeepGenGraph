; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments/baseline.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments/baseline.input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !5
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !5
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !5
  %5 = and i32 %kdma.lane, 31, !dbg !5
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !5
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !5
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !5
  %kdma.intra = and i32 %6, 8, !dbg !5
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !5
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %8 = sext i32 %7 to i64, !dbg !5
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %10 = sext i32 %9 to i64, !dbg !5
  %11 = shl nsw i64 %10, 5, !dbg !5
  %12 = add nsw i64 %11, %8, !dbg !5
  %13 = lshr i64 %12, 5, !dbg !5
  %14 = and i64 %13, 576460752303423456, !dbg !5
  %15 = and i64 %8, 31, !dbg !5
  %16 = or disjoint i64 %14, %15, !dbg !5
  %17 = lshr i64 %12, 10, !dbg !8
  %18 = add nuw nsw i64 %13, %17, !dbg !8
  %19 = shl nsw i64 %16, 7, !dbg !8
  %20 = mul nsw i64 %17, -4096, !dbg !8
  %21 = add i64 %19, %20, !dbg !8
  %22 = shl i64 %18, 19, !dbg !8
  %23 = and i64 %22, 16252928, !dbg !8
  %24 = shl nsw i64 %21, 7, !dbg !8
  %25 = add i64 %23, %24, !dbg !8
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !8, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !8
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !8
  %26 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %27 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %25, %address.local413, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %28 = insertelement <2 x i64> poison, i64 %27, i64 0, !dbg !8
  %29 = bitcast <2 x i64> %28 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %29, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %26), !dbg !8
  %30 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !8
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !8
  %31 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local410, %address.local424, !dbg !8
  %.idx1400 = or disjoint i64 %25, %address.local425, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %31), !dbg !8
  %32 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !8
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !8
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !8
  %34 = or disjoint i64 %25, 32, !dbg !8
  %35 = or disjoint i64 %address.local413, %34, !dbg !8
  %.tr = trunc i64 %35 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !8
  %36 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !8
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !8
  %37 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !8
  %38 = or disjoint i64 %address.local425, %34, !dbg !8
  %.tr1401 = trunc i64 %38 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %39 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
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
  %40 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %41 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %42 = fmul <4 x half> %40, splat (half 0xH3015), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %43 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %44 = fmul <4 x half> %41, splat (half 0xH3015), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %45 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %46 = fmul <4 x half> %43, splat (half 0xH3015), !dbg !10
  %47 = fmul <4 x half> %45, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %48 = or disjoint i64 %25, 64, !dbg !8
  %49 = or disjoint i64 %address.local413, %48, !dbg !8
  %.tr1402 = trunc i64 %49 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %51 = or disjoint i64 %address.local425, %48, !dbg !8
  %.tr1403 = trunc i64 %51 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %56 = fmul <4 x half> %53, splat (half 0xH3015), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %58 = fmul <4 x half> %54, splat (half 0xH3015), !dbg !10
  %59 = fmul <4 x half> %55, splat (half 0xH3015), !dbg !10
  %60 = fmul <4 x half> %57, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %61 = or disjoint i64 %25, 96, !dbg !8
  %62 = or disjoint i64 %address.local413, %61, !dbg !8
  %.tr1404 = trunc i64 %62 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %64 = or disjoint i64 %address.local425, %61, !dbg !8
  %.tr1405 = trunc i64 %64 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !8
  %66 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %69 = fmul <4 x half> %66, splat (half 0xH3015), !dbg !10
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %71 = fmul <4 x half> %67, splat (half 0xH3015), !dbg !10
  %72 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !10
  %73 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !10
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %77 = fmul <4 x half> %74, splat (half 0xH3015), !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %79 = fmul <4 x half> %75, splat (half 0xH3015), !dbg !10
  %80 = fmul <4 x half> %76, splat (half 0xH3015), !dbg !10
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !10
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %82 = shl nuw nsw i32 %kdma.tid, 2, !dbg !14
  %83 = and i32 %82, 3084, !dbg !14
  %84 = and i32 %kdma.tid, 1, !dbg !14
  %85 = or disjoint i32 %83, %84, !dbg !14
  %narrow = add nuw nsw i32 %85, 1, !dbg !14
  %86 = lshr i32 %narrow, 1, !dbg !14
  %87 = zext nneg i32 %86 to i64, !dbg !14
  br label %88, !dbg !7

88:                                               ; preds = %97, %4
  %89 = phi i64 [ %1022, %97 ], [ 0, %4 ], !dbg !7
  %90 = phi [2 x <8 x float>] [ %1007, %97 ], [ zeroinitializer, %4 ], !dbg !7
  %91 = phi [2 x <1 x float>] [ %773, %97 ], [ zeroinitializer, %4 ], !dbg !7
  %92 = phi [2 x <1 x float>] [ %1009, %97 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !7
  %93 = phi [2 x <8 x float>] [ %1013, %97 ], [ zeroinitializer, %4 ], !dbg !7
  %94 = phi [2 x <8 x float>] [ %1017, %97 ], [ zeroinitializer, %4 ], !dbg !7
  %95 = phi [2 x <8 x float>] [ %1021, %97 ], [ zeroinitializer, %4 ], !dbg !7
  %96 = icmp samesign ult i64 %89, 32, !dbg !7
  br i1 %96, label %97, label %1023, !dbg !7

97:                                               ; preds = %88
  %98 = shl i64 %13, 19, !dbg !12
  %99 = and i64 %98, 16252928, !dbg !12
  %100 = shl nuw nsw i64 %89, 14, !dbg !12
  %101 = or disjoint i64 %99, %100, !dbg !12
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !12, !range !32, !frisk.register_lifetimes !1
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !8
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !8
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !12
  %103 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %104 = insertelement <2 x i64> poison, i64 %103, i64 0, !dbg !12
  %105 = bitcast <2 x i64> %104 to <4 x i32>, !dbg !12
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %105, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !12
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !12
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !12
  %kdma.source1508 = and i64 %kdma.source, 91, !dbg !12
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !12
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !12
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !12
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !12
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !12
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !12
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !12
  %kdma.source1516 = or disjoint i64 %101, %kdma.source1515, !dbg !12
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !12
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !12
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !12
  %106 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !12
  %kdma.source1521 = and i64 %kdma.source1520, 8, !dbg !12
  %kdma.source1522 = and i64 %kdma.source1519, 123, !dbg !12
  %kdma.source1523 = lshr i64 %kdma.source1519, 2, !dbg !12
  %kdma.source1524 = and i64 %kdma.source1523, 4, !dbg !12
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519, !dbg !12
  %kdma.source1526 = and i64 %kdma.source1525, 4, !dbg !12
  %kdma.source1527 = or disjoint i64 %kdma.source1522, %kdma.source1526, !dbg !12
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7, !dbg !12
  %kdma.source1529 = or disjoint i64 %kdma.source1521, %kdma.source1528, !dbg !12
  %kdma.source1530 = or disjoint i64 %101, %kdma.source1529, !dbg !12
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !12
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !12
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra.hi, !dbg !12
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !12
  %107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !8
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !8
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !12
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !12
  %kdma.source1546 = or disjoint i64 %kdma.source1515, 16, !dbg !12
  %kdma.source1547 = or disjoint i64 %101, %kdma.source1546, !dbg !12
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !12
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !12
  %kdma.offset1550 = or disjoint i32 %kdma.source1549, %kdma.intra, !dbg !12
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1550, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1563 = or disjoint i64 %kdma.source1529, 16, !dbg !12
  %kdma.source1564 = or disjoint i64 %101, %kdma.source1563, !dbg !12
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !12
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !12
  %kdma.offset1567 = or disjoint i32 %kdma.source1566, %kdma.intra.hi, !dbg !12
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !12
  %110 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1567, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %111 = or disjoint i64 %101, 32, !dbg !12
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !8
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !8
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !12
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %112), !dbg !12
  %kdma.source1580 = or disjoint i64 %111, %kdma.source1515, !dbg !12
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !12
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !12
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !12
  %113 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1596 = or disjoint i64 %111, %kdma.source1529, !dbg !12
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !12
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !12
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !12
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !12
  %114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !8
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !8
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !12
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115), !dbg !12
  %kdma.source1613 = or disjoint i64 %111, %kdma.source1546, !dbg !12
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !12
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !12
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !12
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1630 = or disjoint i64 %111, %kdma.source1563, !dbg !12
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !12
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !12
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !12
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !12
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %address.local542 = lshr i64 %address.local499, 2, !dbg !14
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !14
  %address.local546 = and i64 %address.local545, 24, !dbg !14
  %address.local548 = shl i64 %address.local499, 62, !dbg !14
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !14
  %address.local550 = and i64 %address.local549, -64, !dbg !14
  %address.local553 = and i64 %address.local542, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !14
  %118 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !14
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !14
  %119 = getelementptr i8, ptr addrspace(3) %118, i64 %.idx1410, !dbg !14
  %120 = getelementptr [2 x i8], ptr addrspace(3) %119, i64 %address.local546, !dbg !14
  %121 = getelementptr [2 x i8], ptr addrspace(3) %120, i64 %address.local550, !dbg !14
  %122 = getelementptr [2 x i8], ptr addrspace(3) %121, i64 %address.local553, !dbg !14
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %122, i64 %address.local557, !dbg !14
  %123 = add nsw i64 %.idx, %.idx1410, !dbg !14
  %.idx1893 = shl nuw nsw i64 %address.local546, 1, !dbg !14
  %124 = or disjoint i64 %123, %.idx1893, !dbg !14
  %125 = or disjoint i64 %address.local550, %address.local553, !dbg !14
  %126 = shl nsw i64 %125, 1, !dbg !14
  %127 = add nsw i64 %126, %124, !dbg !14
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !14
  %128 = add nsw i64 %127, %address.local559.idx, !dbg !14
  %129 = lshr i64 %128, 6, !dbg !14
  %130 = and i64 %129, 8, !dbg !14
  %131 = xor i64 %128, %130, !dbg !14
  %132 = sub i64 %131, %128, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %132, !dbg !14
  %133 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !14
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %135 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !14
  %address.local571 = and i32 %address.local570, 24, !dbg !14
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !14
  %address.local573 = and i32 %address.local572, 768, !dbg !14
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !14
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !14
  %address.local576 = and i32 %address.local575, 96, !dbg !14
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !14
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !14
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %87) #9, !dbg !14, !frisk.register_lifetimes !1
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !14
  %address.local580 = and i32 %address.local579, 1, !dbg !14
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !14
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !14
  %address.local583 = shl i64 %address.local582, 6, !dbg !14
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !14
  %136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !14
  %137 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local553, !dbg !14
  %138 = getelementptr [2 x i8], ptr addrspace(3) %137, i64 %address.local557, !dbg !14
  %address.local588 = getelementptr i8, ptr addrspace(3) %138, i64 64, !dbg !14
  %.idx18961899 = add i64 %address.local584, %address.local553, !dbg !14
  %139 = shl i64 %.idx18961899, 1, !dbg !14
  %140 = add i64 %139, %address.local559.idx, !dbg !14
  %141 = add i64 %140, 64, !dbg !14
  %142 = lshr i64 %141, 6, !dbg !14
  %143 = and i64 %142, 8, !dbg !14
  %144 = xor i64 %141, %143, !dbg !14
  %145 = sub i64 %144, %141, !dbg !14
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %145, !dbg !14
  %146 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !14, !frisk.k_lds_pair !35
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !14
  %147 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !14, !frisk.k_lds_pair !35
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !14
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !14, !frisk.k_lds_pair !36
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !14
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !14, !frisk.k_lds_pair !36
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !14
  %150 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !14, !frisk.k_lds_pair !37
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !14, !frisk.k_lds_pair !37
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !14
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !14, !frisk.k_lds_pair !38
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !14, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %135, <4 x float> %134) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !14
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !14, !frisk.k_lds_pair !33
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !14
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !14, !frisk.k_lds_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !14
  %158 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !14, !frisk.k_lds_pair !39
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !14
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !14, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %160 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %148, <4 x float> %157) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !14
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !14, !frisk.k_lds_pair !40
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !14
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !14, !frisk.k_lds_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %163 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %152, <4 x float> %163) #9, !dbg !14
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !14
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %149, <4 x float> %165) #9, !dbg !14
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !14
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %153, <4 x float> %167) #9, !dbg !14
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !14
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %158, <4 x float> %169) #9, !dbg !14
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !14
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %162, <4 x float> %171) #9, !dbg !14
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %156, <4 x float> zeroinitializer) #9, !dbg !14
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %159, <4 x float> %173) #9, !dbg !14
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !14
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %135, <4 x float> %175) #9, !dbg !14
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !14
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %148, <4 x float> %177) #9, !dbg !14
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !14
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %152, <4 x float> %179) #9, !dbg !14
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !14
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %149, <4 x float> %181) #9, !dbg !14
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !14
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %153, <4 x float> %183) #9, !dbg !14
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !14
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %158, <4 x float> %185) #9, !dbg !14
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %161, <4 x float> zeroinitializer) #9, !dbg !14
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %162, <4 x float> %187) #9, !dbg !14
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %156, <4 x float> zeroinitializer) #9, !dbg !14
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %159, <4 x float> %189) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %191 = or disjoint i64 %101, 64, !dbg !12
  %kdma.source1646 = or disjoint i64 %191, %kdma.source1515, !dbg !12
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !12
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !12
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !12
  %192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1662 = or disjoint i64 %191, %kdma.source1529, !dbg !12
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !12
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !12
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !12
  %193 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1679 = or disjoint i64 %191, %kdma.source1546, !dbg !12
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !12
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !12
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !12
  %194 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1696 = or disjoint i64 %191, %kdma.source1563, !dbg !12
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !12
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !12
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !12
  %195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !14, !frisk.k_lds_pair !41
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !14, !frisk.k_lds_pair !41
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !14, !frisk.k_lds_pair !42
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !14, !frisk.k_lds_pair !42
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !14, !frisk.k_lds_pair !43
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !14, !frisk.k_lds_pair !43
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !14, !frisk.k_lds_pair !44
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !14, !frisk.k_lds_pair !44
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !14, !frisk.k_lds_pair !45
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !14
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !14, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %196, <4 x float> %154) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !14, !frisk.k_lds_pair !46
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !14
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !14, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %198, <4 x float> %206) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !14
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !14, !frisk.k_lds_pair !47
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !14
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !14, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %212 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %200, <4 x float> %160) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !14
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !14, !frisk.k_lds_pair !48
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !14
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14, !frisk.k_lds_pair !48
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %202, <4 x float> %212) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %197, <4 x float> %164) #9, !dbg !14
  %217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %199, <4 x float> %216) #9, !dbg !14
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %201, <4 x float> %166) #9, !dbg !14
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %203, <4 x float> %218) #9, !dbg !14
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %204, <4 x float> %168) #9, !dbg !14
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %207, <4 x float> %220) #9, !dbg !14
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %210, <4 x float> %170) #9, !dbg !14
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %213, <4 x float> %222) #9, !dbg !14
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %205, <4 x float> %172) #9, !dbg !14
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %208, <4 x float> %224) #9, !dbg !14
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %211, <4 x float> %174) #9, !dbg !14
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %214, <4 x float> %226) #9, !dbg !14
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %196, <4 x float> %176) #9, !dbg !14
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %198, <4 x float> %228) #9, !dbg !14
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %200, <4 x float> %178) #9, !dbg !14
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %202, <4 x float> %230) #9, !dbg !14
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %197, <4 x float> %180) #9, !dbg !14
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %199, <4 x float> %232) #9, !dbg !14
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %201, <4 x float> %182) #9, !dbg !14
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %203, <4 x float> %234) #9, !dbg !14
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %204, <4 x float> %184) #9, !dbg !14
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %207, <4 x float> %236) #9, !dbg !14
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %210, <4 x float> %186) #9, !dbg !14
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %213, <4 x float> %238) #9, !dbg !14
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %205, <4 x float> %188) #9, !dbg !14
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %208, <4 x float> %240) #9, !dbg !14
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %211, <4 x float> %190) #9, !dbg !14
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %214, <4 x float> %242) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %244 = or disjoint i64 %101, 96, !dbg !12
  %kdma.source1712 = or disjoint i64 %244, %kdma.source1515, !dbg !12
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !12
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !12
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !12
  %245 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1728 = or disjoint i64 %244, %kdma.source1529, !dbg !12
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !12
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !12
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !12
  %246 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1745 = or disjoint i64 %244, %kdma.source1546, !dbg !12
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !12
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !12
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !12
  %247 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %kdma.source1762 = or disjoint i64 %244, %kdma.source1563, !dbg !12
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !12
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !12
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !12
  %248 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !12
  %249 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !49
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !14, !frisk.k_lds_pair !49
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !50
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !14, !frisk.k_lds_pair !50
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !14, !frisk.k_lds_pair !51
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !14, !frisk.k_lds_pair !51
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !14, !frisk.k_lds_pair !52
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !14, !frisk.k_lds_pair !52
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !14, !frisk.k_lds_pair !53
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !14, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %249, <4 x float> %209) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !14, !frisk.k_lds_pair !54
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !14, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %251, <4 x float> %259) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !14, !frisk.k_lds_pair !55
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !14, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %253, <4 x float> %215) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !14, !frisk.k_lds_pair !56
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !14, !frisk.k_lds_pair !56
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %255, <4 x float> %265) #9, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %250, <4 x float> %217) #9, !dbg !14
  %270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %252, <4 x float> %269) #9, !dbg !14
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %254, <4 x float> %219) #9, !dbg !14
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %256, <4 x float> %271) #9, !dbg !14
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %257, <4 x float> %221) #9, !dbg !14
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %260, <4 x float> %273) #9, !dbg !14
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %263, <4 x float> %223) #9, !dbg !14
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %266, <4 x float> %275) #9, !dbg !14
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %258, <4 x float> %225) #9, !dbg !14
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %261, <4 x float> %277) #9, !dbg !14
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %264, <4 x float> %227) #9, !dbg !14
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %267, <4 x float> %279) #9, !dbg !14
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %249, <4 x float> %229) #9, !dbg !14
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %251, <4 x float> %281) #9, !dbg !14
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %253, <4 x float> %231) #9, !dbg !14
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %255, <4 x float> %283) #9, !dbg !14
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %250, <4 x float> %233) #9, !dbg !14
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %252, <4 x float> %285) #9, !dbg !14
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %254, <4 x float> %235) #9, !dbg !14
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %256, <4 x float> %287) #9, !dbg !14
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %257, <4 x float> %237) #9, !dbg !14
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %260, <4 x float> %289) #9, !dbg !14
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %263, <4 x float> %239) #9, !dbg !14
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %266, <4 x float> %291) #9, !dbg !14
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %258, <4 x float> %241) #9, !dbg !14
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %261, <4 x float> %293) #9, !dbg !14
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %264, <4 x float> %243) #9, !dbg !14
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %267, <4 x float> %295) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !15
  %address.local918 = and i64 %address.local917, 8064, !dbg !15
  %address.local920 = and i64 %address.local500, 24, !dbg !15
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !15
  %297 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %.idx14191420 = or disjoint i64 %101, %address.local921, !dbg !15
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !15
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !15
  %298 = insertelement <2 x i64> poison, i64 %297, i64 0, !dbg !15
  %299 = bitcast <2 x i64> %298 to <4 x i32>, !dbg !15
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %299, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !15
  %300 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !15
  %address.local924 = and i64 %address.local923, 16256, !dbg !15
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !15
  %.idx14211422 = or disjoint i64 %101, %address.local927, !dbg !15
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !15
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !15
  %301 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !14
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %302, <4 x float> %262) #9, !dbg !14
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !14
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !14
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %304, <4 x float> %303) #9, !dbg !14
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %305, <4 x float> %268) #9, !dbg !14
  %308 = extractelement <4 x float> %306, i64 0, !dbg !19
  %309 = call float @llvm.maxnum.f32(float %308, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %310 = extractelement <4 x float> %306, i64 1, !dbg !19
  %311 = extractelement <4 x float> %306, i64 2, !dbg !19
  %312 = extractelement <4 x float> %306, i64 3, !dbg !19
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !14
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !14
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %313, <4 x float> %307) #9, !dbg !14
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %314, <4 x float> %270) #9, !dbg !14
  %317 = extractelement <4 x float> %315, i64 0, !dbg !19
  %local.reduce151 = call float @llvm.maxnum.f32(float %309, float %317), !dbg !20, !frisk.local_reduction !1
  %318 = extractelement <4 x float> %315, i64 1, !dbg !19
  %local.reduce152 = call float @llvm.maxnum.f32(float %310, float %318), !dbg !20, !frisk.local_reduction !1
  %319 = extractelement <4 x float> %315, i64 2, !dbg !19
  %local.reduce153 = call float @llvm.maxnum.f32(float %311, float %319), !dbg !20, !frisk.local_reduction !1
  %320 = extractelement <4 x float> %315, i64 3, !dbg !19
  %local.reduce154 = call float @llvm.maxnum.f32(float %312, float %320), !dbg !20, !frisk.local_reduction !1
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !14
  %322 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !14
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %321, <4 x float> %316) #9, !dbg !14
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %322, <4 x float> %272) #9, !dbg !14
  %325 = extractelement <4 x float> %323, i64 0, !dbg !19
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %325), !dbg !20, !frisk.local_reduction !1
  %326 = extractelement <4 x float> %323, i64 1, !dbg !19
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %326), !dbg !20, !frisk.local_reduction !1
  %327 = extractelement <4 x float> %323, i64 2, !dbg !19
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %327), !dbg !20, !frisk.local_reduction !1
  %328 = extractelement <4 x float> %323, i64 3, !dbg !19
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %328), !dbg !20, !frisk.local_reduction !1
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !14
  %330 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !14
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %329, <4 x float> %324) #9, !dbg !14
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %330, <4 x float> %274) #9, !dbg !14
  %333 = extractelement <4 x float> %331, i64 0, !dbg !19
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %333), !dbg !20, !frisk.local_reduction !1
  %334 = extractelement <4 x float> %331, i64 1, !dbg !19
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %334), !dbg !20, !frisk.local_reduction !1
  %335 = extractelement <4 x float> %331, i64 2, !dbg !19
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %335), !dbg !20, !frisk.local_reduction !1
  %336 = extractelement <4 x float> %331, i64 3, !dbg !19
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %336), !dbg !20, !frisk.local_reduction !1
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !14
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !14
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %337, <4 x float> %332) #9, !dbg !14
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %338, <4 x float> %276) #9, !dbg !14
  %341 = extractelement <4 x float> %339, i64 0, !dbg !19
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %341), !dbg !20, !frisk.local_reduction !1
  %342 = extractelement <4 x float> %339, i64 1, !dbg !19
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %342), !dbg !20, !frisk.local_reduction !1
  %343 = extractelement <4 x float> %339, i64 2, !dbg !19
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %343), !dbg !20, !frisk.local_reduction !1
  %344 = extractelement <4 x float> %339, i64 3, !dbg !19
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %344), !dbg !20, !frisk.local_reduction !1
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !14
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !14
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %345, <4 x float> %340) #9, !dbg !14
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %346, <4 x float> %278) #9, !dbg !14
  %349 = extractelement <4 x float> %347, i64 0, !dbg !19
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %349), !dbg !20, !frisk.local_reduction !1
  %350 = extractelement <4 x float> %347, i64 1, !dbg !19
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %350), !dbg !20, !frisk.local_reduction !1
  %351 = extractelement <4 x float> %347, i64 2, !dbg !19
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %351), !dbg !20, !frisk.local_reduction !1
  %352 = extractelement <4 x float> %347, i64 3, !dbg !19
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %352), !dbg !20, !frisk.local_reduction !1
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !14
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !14
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %353, <4 x float> %348) #9, !dbg !14
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %354, <4 x float> %280) #9, !dbg !14
  %357 = extractelement <4 x float> %355, i64 0, !dbg !19
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %357), !dbg !20, !frisk.local_reduction !1
  %358 = extractelement <4 x float> %355, i64 1, !dbg !19
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %358), !dbg !20, !frisk.local_reduction !1
  %359 = extractelement <4 x float> %355, i64 2, !dbg !19
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %359), !dbg !20, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %355, i64 3, !dbg !19
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %360), !dbg !20, !frisk.local_reduction !1
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !14
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %361, <4 x float> %356) #9, !dbg !14
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %302, <4 x float> %282) #9, !dbg !14
  %364 = extractelement <4 x float> %362, i64 0, !dbg !19
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %364), !dbg !20, !frisk.local_reduction !1
  %365 = extractelement <4 x float> %362, i64 1, !dbg !19
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %365), !dbg !20, !frisk.local_reduction !1
  %366 = extractelement <4 x float> %362, i64 2, !dbg !19
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %366), !dbg !20, !frisk.local_reduction !1
  %367 = extractelement <4 x float> %362, i64 3, !dbg !19
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %367), !dbg !20, !frisk.local_reduction !1
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !20, !frisk.local_reduction !1
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !20, !frisk.local_reduction !1
  %368 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !20, !frisk.local_reduction !1
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %363) #9, !dbg !14
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %305, <4 x float> %284) #9, !dbg !14
  %371 = extractelement <4 x float> %369, i64 0, !dbg !19
  %372 = extractelement <4 x float> %369, i64 1, !dbg !19
  %373 = extractelement <4 x float> %369, i64 2, !dbg !19
  %374 = extractelement <4 x float> %369, i64 3, !dbg !19
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %313, <4 x float> %370) #9, !dbg !14
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %314, <4 x float> %286) #9, !dbg !14
  %377 = extractelement <4 x float> %375, i64 0, !dbg !19
  %378 = extractelement <4 x float> %375, i64 1, !dbg !19
  %379 = extractelement <4 x float> %375, i64 2, !dbg !19
  %380 = extractelement <4 x float> %375, i64 3, !dbg !19
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %321, <4 x float> %376) #9, !dbg !14
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %322, <4 x float> %288) #9, !dbg !14
  %383 = extractelement <4 x float> %381, i64 0, !dbg !19
  %384 = extractelement <4 x float> %381, i64 1, !dbg !19
  %385 = extractelement <4 x float> %381, i64 2, !dbg !19
  %386 = extractelement <4 x float> %381, i64 3, !dbg !19
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %329, <4 x float> %382) #9, !dbg !14
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %330, <4 x float> %290) #9, !dbg !14
  %389 = extractelement <4 x float> %387, i64 0, !dbg !19
  %390 = extractelement <4 x float> %387, i64 1, !dbg !19
  %391 = extractelement <4 x float> %387, i64 2, !dbg !19
  %392 = extractelement <4 x float> %387, i64 3, !dbg !19
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %337, <4 x float> %388) #9, !dbg !14
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %338, <4 x float> %292) #9, !dbg !14
  %395 = extractelement <4 x float> %393, i64 0, !dbg !19
  %396 = extractelement <4 x float> %393, i64 1, !dbg !19
  %397 = extractelement <4 x float> %393, i64 2, !dbg !19
  %398 = extractelement <4 x float> %393, i64 3, !dbg !19
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %345, <4 x float> %394) #9, !dbg !14
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %346, <4 x float> %294) #9, !dbg !14
  %401 = extractelement <4 x float> %399, i64 0, !dbg !19
  %402 = extractelement <4 x float> %399, i64 1, !dbg !19
  %403 = extractelement <4 x float> %399, i64 2, !dbg !19
  %404 = extractelement <4 x float> %399, i64 3, !dbg !19
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %353, <4 x float> %400) #9, !dbg !14
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %354, <4 x float> %296) #9, !dbg !14
  %407 = extractelement <4 x float> %405, i64 0, !dbg !19
  %408 = extractelement <4 x float> %405, i64 1, !dbg !19
  %409 = extractelement <4 x float> %405, i64 2, !dbg !19
  %410 = extractelement <4 x float> %405, i64 3, !dbg !19
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %361, <4 x float> %406) #9, !dbg !14
  %412 = extractelement <4 x float> %411, i64 0, !dbg !19
  %413 = extractelement <4 x float> %411, i64 1, !dbg !19
  %414 = extractelement <4 x float> %411, i64 2, !dbg !19
  %415 = extractelement <4 x float> %411, i64 3, !dbg !19
  %416 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !20
  %417 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %416), !dbg !20
  %418 = add nuw nsw i32 %417, 64, !dbg !20
  %419 = xor i32 %417, 16, !dbg !20
  %420 = and i32 %418, 192, !dbg !20
  %421 = icmp samesign ult i32 %419, %420, !dbg !20
  %422 = select i1 %421, i32 %419, i32 %417, !dbg !20
  %423 = shl nuw nsw i32 %422, 2, !dbg !20
  %424 = bitcast float %368 to i32, !dbg !20
  %425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %424), !dbg !20
  %426 = bitcast i32 %425 to float, !dbg !20
  %427 = call float @llvm.maxnum.f32(float %368, float %426), !dbg !20
  %428 = xor i32 %417, 32, !dbg !20
  %429 = icmp samesign ult i32 %428, %420, !dbg !20
  %430 = select i1 %429, i32 %428, i32 %417, !dbg !20
  %431 = shl nuw nsw i32 %430, 2, !dbg !20
  %432 = bitcast float %427 to i32, !dbg !20
  %433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %432), !dbg !20
  %434 = bitcast i32 %433 to float, !dbg !20
  %435 = call float @llvm.maxnum.f32(float %427, float %434), !dbg !20
  %436 = extractvalue [2 x <1 x float>] %92, 0, !dbg !20
  %437 = insertelement <1 x float> poison, float %435, i64 0, !dbg !20
  %438 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %437, <1 x float> %436), !dbg !20
  %439 = call float @llvm.maxnum.f32(float %371, float 0xFFF0000000000000), !dbg !20, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %439, float %377), !dbg !20, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %372, float %378), !dbg !20, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %373, float %379), !dbg !20, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %374, float %380), !dbg !20, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %383), !dbg !20, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %384), !dbg !20, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %385), !dbg !20, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %386), !dbg !20, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %389), !dbg !20, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %390), !dbg !20, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %391), !dbg !20, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %392), !dbg !20, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %395), !dbg !20, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %396), !dbg !20, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %397), !dbg !20, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %398), !dbg !20, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %401), !dbg !20, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %402), !dbg !20, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %403), !dbg !20, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %404), !dbg !20, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %407), !dbg !20, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %408), !dbg !20, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %409), !dbg !20, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %410), !dbg !20, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %412), !dbg !20, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %413), !dbg !20, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %414), !dbg !20, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %415), !dbg !20, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !20, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !20, !frisk.local_reduction !1
  %440 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !20, !frisk.local_reduction !1
  %441 = bitcast float %440 to i32, !dbg !20
  %442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %441), !dbg !20
  %443 = bitcast i32 %442 to float, !dbg !20
  %444 = call float @llvm.maxnum.f32(float %440, float %443), !dbg !20
  %445 = bitcast float %444 to i32, !dbg !20
  %446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %445), !dbg !20
  %447 = bitcast i32 %446 to float, !dbg !20
  %448 = call float @llvm.maxnum.f32(float %444, float %447), !dbg !20
  %449 = extractvalue [2 x <1 x float>] %92, 1, !dbg !20
  %450 = insertelement <1 x float> poison, float %448, i64 0, !dbg !20
  %451 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %450, <1 x float> %449), !dbg !20
  %452 = shufflevector <1 x float> %438, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %453 = shufflevector <1 x float> %451, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %454 = fsub <1 x float> %436, %438, !dbg !22
  %455 = extractelement <1 x float> %454, i64 0, !dbg !23
  %456 = call float @llvm.amdgcn.exp2.f32(float %455), !dbg !23
  %457 = insertelement <4 x float> poison, float %456, i64 0, !dbg !28
  %458 = shufflevector <4 x float> %457, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %459 = extractvalue [2 x <1 x float>] %91, 0, !dbg !24
  %460 = extractelement <1 x float> %459, i64 0, !dbg !24
  %461 = fmul float %460, %456, !dbg !24
  %462 = fsub <1 x float> %449, %451, !dbg !22
  %463 = extractelement <1 x float> %462, i64 0, !dbg !23
  %464 = call float @llvm.amdgcn.exp2.f32(float %463), !dbg !23
  %465 = insertelement <4 x float> poison, float %464, i64 0, !dbg !28
  %466 = shufflevector <4 x float> %465, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %467 = extractvalue [2 x <1 x float>] %91, 1, !dbg !24
  %468 = extractelement <1 x float> %467, i64 0, !dbg !24
  %469 = fmul float %468, %464, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1436 = or disjoint i64 %111, %address.local921, !dbg !15
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !15
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !15
  %vdb.address1445 = or disjoint i64 %111, %address.local927, !dbg !15
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !15
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !15
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !13
  %470 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !13
  %471 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !13
  %address.local930 = and i32 %address.local929, 120, !dbg !13
  %address.local932 = and i32 %address.local575, 1536, !dbg !13
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !13
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !13
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !13
  %472 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %473 = bitcast <4 x i32> %472 to <8 x half>, !dbg !13
  %address.local936 = and i64 %address.local499, 60, !dbg !13
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !13
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !13
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !13
  %address.local940 = lshr i64 %address.local555, 4, !dbg !13
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !13
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !13
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !13
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !13
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !13
  %474 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %475 = bitcast <4 x i32> %474 to <8 x half>, !dbg !13
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !13
  %476 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %477 = bitcast <4 x i32> %476 to <8 x half>, !dbg !13
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !13
  %478 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %479 = bitcast <4 x i32> %478 to <8 x half>, !dbg !13
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !13
  %480 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %481 = bitcast <4 x i32> %480 to <8 x half>, !dbg !13
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !13
  %482 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %483 = bitcast <4 x i32> %482 to <8 x half>, !dbg !13
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !13
  %484 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %485 = bitcast <4 x i32> %484 to <8 x half>, !dbg !13
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !13
  %486 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %487 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %488 = extractvalue [2 x <8 x float>] %90, 0, !dbg !28
  %489 = shufflevector <8 x float> %488, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %490 = fmul <4 x float> %489, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %491 = fsub <4 x float> %306, %452, !dbg !21
  %492 = extractelement <4 x float> %491, i64 0, !dbg !25
  %493 = call float @llvm.amdgcn.exp2.f32(float %492), !dbg !25
  %494 = insertelement <4 x float> poison, float %493, i64 0, !dbg !27
  %495 = extractelement <4 x float> %491, i64 1, !dbg !25
  %496 = call float @llvm.amdgcn.exp2.f32(float %495), !dbg !25
  %497 = insertelement <4 x float> %494, float %496, i64 1, !dbg !27
  %498 = extractelement <4 x float> %491, i64 2, !dbg !25
  %499 = call float @llvm.amdgcn.exp2.f32(float %498), !dbg !25
  %500 = insertelement <4 x float> %497, float %499, i64 2, !dbg !27
  %501 = extractelement <4 x float> %491, i64 3, !dbg !25
  %502 = call float @llvm.amdgcn.exp2.f32(float %501), !dbg !25
  %503 = insertelement <4 x float> %500, float %502, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %503) #9, !dbg !27, !frisk.register_lifetimes !1
  %504 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %504) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %487, <4 x float> %490) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %506 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %507 = shufflevector <8 x float> %488, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %508 = fmul <4 x float> %507, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %506, <4 x float> %508) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %510 = extractvalue [2 x <8 x float>] %90, 1, !dbg !28
  %511 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %512 = fmul <4 x float> %511, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %513 = fsub <4 x float> %369, %453, !dbg !21
  %514 = extractelement <4 x float> %513, i64 0, !dbg !25
  %515 = call float @llvm.amdgcn.exp2.f32(float %514), !dbg !25
  %516 = insertelement <4 x float> poison, float %515, i64 0, !dbg !27
  %517 = extractelement <4 x float> %513, i64 1, !dbg !25
  %518 = call float @llvm.amdgcn.exp2.f32(float %517), !dbg !25
  %519 = insertelement <4 x float> %516, float %518, i64 1, !dbg !27
  %520 = extractelement <4 x float> %513, i64 2, !dbg !25
  %521 = call float @llvm.amdgcn.exp2.f32(float %520), !dbg !25
  %522 = insertelement <4 x float> %519, float %521, i64 2, !dbg !27
  %523 = extractelement <4 x float> %513, i64 3, !dbg !25
  %524 = call float @llvm.amdgcn.exp2.f32(float %523), !dbg !25
  %525 = insertelement <4 x float> %522, float %524, i64 3, !dbg !27
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %525) #9, !dbg !27, !frisk.register_lifetimes !1
  %526 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !27
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %526) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %487, <4 x float> %512) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %528 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %529 = fmul <4 x float> %528, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %506, <4 x float> %529) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %531 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %532 = fsub <4 x float> %315, %452, !dbg !21
  %533 = extractelement <4 x float> %532, i64 0, !dbg !25
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !25
  %local.reduce89 = fadd float %493, %534, !dbg !26, !frisk.local_reduction !1
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !27
  %536 = extractelement <4 x float> %532, i64 1, !dbg !25
  %537 = call float @llvm.amdgcn.exp2.f32(float %536), !dbg !25
  %local.reduce90 = fadd float %496, %537, !dbg !26, !frisk.local_reduction !1
  %538 = insertelement <4 x float> %535, float %537, i64 1, !dbg !27
  %539 = extractelement <4 x float> %532, i64 2, !dbg !25
  %540 = call float @llvm.amdgcn.exp2.f32(float %539), !dbg !25
  %local.reduce91 = fadd float %499, %540, !dbg !26, !frisk.local_reduction !1
  %541 = insertelement <4 x float> %538, float %540, i64 2, !dbg !27
  %542 = extractelement <4 x float> %532, i64 3, !dbg !25
  %543 = call float @llvm.amdgcn.exp2.f32(float %542), !dbg !25
  %local.reduce92 = fadd float %502, %543, !dbg !26, !frisk.local_reduction !1
  %544 = insertelement <4 x float> %541, float %543, i64 3, !dbg !27
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %544) #9, !dbg !27, !frisk.register_lifetimes !1
  %545 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !27
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %545) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %531, <4 x float> %505) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %547 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %547, <4 x float> %509) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %549 = fsub <4 x float> %375, %453, !dbg !21
  %550 = extractelement <4 x float> %549, i64 0, !dbg !25
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !25
  %local.reduce = fadd float %515, %551, !dbg !26, !frisk.local_reduction !1
  %552 = insertelement <4 x float> poison, float %551, i64 0, !dbg !27
  %553 = extractelement <4 x float> %549, i64 1, !dbg !25
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !25
  %local.reduce59 = fadd float %518, %554, !dbg !26, !frisk.local_reduction !1
  %555 = insertelement <4 x float> %552, float %554, i64 1, !dbg !27
  %556 = extractelement <4 x float> %549, i64 2, !dbg !25
  %557 = call float @llvm.amdgcn.exp2.f32(float %556), !dbg !25
  %local.reduce60 = fadd float %521, %557, !dbg !26, !frisk.local_reduction !1
  %558 = insertelement <4 x float> %555, float %557, i64 2, !dbg !27
  %559 = extractelement <4 x float> %549, i64 3, !dbg !25
  %560 = call float @llvm.amdgcn.exp2.f32(float %559), !dbg !25
  %local.reduce61 = fadd float %524, %560, !dbg !26, !frisk.local_reduction !1
  %561 = insertelement <4 x float> %558, float %560, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %561) #9, !dbg !27, !frisk.register_lifetimes !1
  %562 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %562) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %563 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %531, <4 x float> %527) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %564 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %547, <4 x float> %530) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %565 = shufflevector <8 x half> %477, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %566 = fsub <4 x float> %323, %452, !dbg !21
  %567 = extractelement <4 x float> %566, i64 0, !dbg !25
  %568 = call float @llvm.amdgcn.exp2.f32(float %567), !dbg !25
  %local.reduce93 = fadd float %local.reduce89, %568, !dbg !26, !frisk.local_reduction !1
  %569 = insertelement <4 x float> poison, float %568, i64 0, !dbg !27
  %570 = extractelement <4 x float> %566, i64 1, !dbg !25
  %571 = call float @llvm.amdgcn.exp2.f32(float %570), !dbg !25
  %local.reduce94 = fadd float %local.reduce90, %571, !dbg !26, !frisk.local_reduction !1
  %572 = insertelement <4 x float> %569, float %571, i64 1, !dbg !27
  %573 = extractelement <4 x float> %566, i64 2, !dbg !25
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !25
  %local.reduce95 = fadd float %local.reduce91, %574, !dbg !26, !frisk.local_reduction !1
  %575 = insertelement <4 x float> %572, float %574, i64 2, !dbg !27
  %576 = extractelement <4 x float> %566, i64 3, !dbg !25
  %577 = call float @llvm.amdgcn.exp2.f32(float %576), !dbg !25
  %local.reduce96 = fadd float %local.reduce92, %577, !dbg !26, !frisk.local_reduction !1
  %578 = insertelement <4 x float> %575, float %577, i64 3, !dbg !27
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %578) #9, !dbg !27, !frisk.register_lifetimes !1
  %579 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !27
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %579) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %580 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %565, <4 x float> %546) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %581 = shufflevector <8 x half> %477, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %582 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %581, <4 x float> %548) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %583 = fsub <4 x float> %381, %453, !dbg !21
  %584 = extractelement <4 x float> %583, i64 0, !dbg !25
  %585 = call float @llvm.amdgcn.exp2.f32(float %584), !dbg !25
  %local.reduce62 = fadd float %local.reduce, %585, !dbg !26, !frisk.local_reduction !1
  %586 = insertelement <4 x float> poison, float %585, i64 0, !dbg !27
  %587 = extractelement <4 x float> %583, i64 1, !dbg !25
  %588 = call float @llvm.amdgcn.exp2.f32(float %587), !dbg !25
  %local.reduce63 = fadd float %local.reduce59, %588, !dbg !26, !frisk.local_reduction !1
  %589 = insertelement <4 x float> %586, float %588, i64 1, !dbg !27
  %590 = extractelement <4 x float> %583, i64 2, !dbg !25
  %591 = call float @llvm.amdgcn.exp2.f32(float %590), !dbg !25
  %local.reduce64 = fadd float %local.reduce60, %591, !dbg !26, !frisk.local_reduction !1
  %592 = insertelement <4 x float> %589, float %591, i64 2, !dbg !27
  %593 = extractelement <4 x float> %583, i64 3, !dbg !25
  %594 = call float @llvm.amdgcn.exp2.f32(float %593), !dbg !25
  %local.reduce65 = fadd float %local.reduce61, %594, !dbg !26, !frisk.local_reduction !1
  %595 = insertelement <4 x float> %592, float %594, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %595) #9, !dbg !27, !frisk.register_lifetimes !1
  %596 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %596) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %565, <4 x float> %563) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %581, <4 x float> %564) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %599 = shufflevector <8 x half> %479, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %600 = fsub <4 x float> %331, %452, !dbg !21
  %601 = extractelement <4 x float> %600, i64 0, !dbg !25
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !25
  %local.reduce97 = fadd float %local.reduce93, %602, !dbg !26, !frisk.local_reduction !1
  %603 = insertelement <4 x float> poison, float %602, i64 0, !dbg !27
  %604 = extractelement <4 x float> %600, i64 1, !dbg !25
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !25
  %local.reduce98 = fadd float %local.reduce94, %605, !dbg !26, !frisk.local_reduction !1
  %606 = insertelement <4 x float> %603, float %605, i64 1, !dbg !27
  %607 = extractelement <4 x float> %600, i64 2, !dbg !25
  %608 = call float @llvm.amdgcn.exp2.f32(float %607), !dbg !25
  %local.reduce99 = fadd float %local.reduce95, %608, !dbg !26, !frisk.local_reduction !1
  %609 = insertelement <4 x float> %606, float %608, i64 2, !dbg !27
  %610 = extractelement <4 x float> %600, i64 3, !dbg !25
  %611 = call float @llvm.amdgcn.exp2.f32(float %610), !dbg !25
  %local.reduce100 = fadd float %local.reduce96, %611, !dbg !26, !frisk.local_reduction !1
  %612 = insertelement <4 x float> %609, float %611, i64 3, !dbg !27
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %612) #9, !dbg !27, !frisk.register_lifetimes !1
  %613 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !27
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %613) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %599, <4 x float> %580) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %615 = shufflevector <8 x half> %479, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %615, <4 x float> %582) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %617 = fsub <4 x float> %387, %453, !dbg !21
  %618 = extractelement <4 x float> %617, i64 0, !dbg !25
  %619 = call float @llvm.amdgcn.exp2.f32(float %618), !dbg !25
  %local.reduce66 = fadd float %local.reduce62, %619, !dbg !26, !frisk.local_reduction !1
  %620 = insertelement <4 x float> poison, float %619, i64 0, !dbg !27
  %621 = extractelement <4 x float> %617, i64 1, !dbg !25
  %622 = call float @llvm.amdgcn.exp2.f32(float %621), !dbg !25
  %local.reduce67 = fadd float %local.reduce63, %622, !dbg !26, !frisk.local_reduction !1
  %623 = insertelement <4 x float> %620, float %622, i64 1, !dbg !27
  %624 = extractelement <4 x float> %617, i64 2, !dbg !25
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !25
  %local.reduce68 = fadd float %local.reduce64, %625, !dbg !26, !frisk.local_reduction !1
  %626 = insertelement <4 x float> %623, float %625, i64 2, !dbg !27
  %627 = extractelement <4 x float> %617, i64 3, !dbg !25
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !25
  %local.reduce69 = fadd float %local.reduce65, %628, !dbg !26, !frisk.local_reduction !1
  %629 = insertelement <4 x float> %626, float %628, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %629) #9, !dbg !27, !frisk.register_lifetimes !1
  %630 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %630) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %631 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %599, <4 x float> %597) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %615, <4 x float> %598) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %633 = shufflevector <8 x half> %481, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %634 = fsub <4 x float> %339, %452, !dbg !21
  %635 = extractelement <4 x float> %634, i64 0, !dbg !25
  %636 = call float @llvm.amdgcn.exp2.f32(float %635), !dbg !25
  %local.reduce101 = fadd float %local.reduce97, %636, !dbg !26, !frisk.local_reduction !1
  %637 = insertelement <4 x float> poison, float %636, i64 0, !dbg !27
  %638 = extractelement <4 x float> %634, i64 1, !dbg !25
  %639 = call float @llvm.amdgcn.exp2.f32(float %638), !dbg !25
  %local.reduce102 = fadd float %local.reduce98, %639, !dbg !26, !frisk.local_reduction !1
  %640 = insertelement <4 x float> %637, float %639, i64 1, !dbg !27
  %641 = extractelement <4 x float> %634, i64 2, !dbg !25
  %642 = call float @llvm.amdgcn.exp2.f32(float %641), !dbg !25
  %local.reduce103 = fadd float %local.reduce99, %642, !dbg !26, !frisk.local_reduction !1
  %643 = insertelement <4 x float> %640, float %642, i64 2, !dbg !27
  %644 = extractelement <4 x float> %634, i64 3, !dbg !25
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !25
  %local.reduce104 = fadd float %local.reduce100, %645, !dbg !26, !frisk.local_reduction !1
  %646 = insertelement <4 x float> %643, float %645, i64 3, !dbg !27
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %646) #9, !dbg !27, !frisk.register_lifetimes !1
  %647 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !27
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %647) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %633, <4 x float> %614) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %649 = shufflevector <8 x half> %481, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %650 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %649, <4 x float> %616) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %651 = fsub <4 x float> %393, %453, !dbg !21
  %652 = extractelement <4 x float> %651, i64 0, !dbg !25
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !25
  %local.reduce70 = fadd float %local.reduce66, %653, !dbg !26, !frisk.local_reduction !1
  %654 = insertelement <4 x float> poison, float %653, i64 0, !dbg !27
  %655 = extractelement <4 x float> %651, i64 1, !dbg !25
  %656 = call float @llvm.amdgcn.exp2.f32(float %655), !dbg !25
  %local.reduce71 = fadd float %local.reduce67, %656, !dbg !26, !frisk.local_reduction !1
  %657 = insertelement <4 x float> %654, float %656, i64 1, !dbg !27
  %658 = extractelement <4 x float> %651, i64 2, !dbg !25
  %659 = call float @llvm.amdgcn.exp2.f32(float %658), !dbg !25
  %local.reduce72 = fadd float %local.reduce68, %659, !dbg !26, !frisk.local_reduction !1
  %660 = insertelement <4 x float> %657, float %659, i64 2, !dbg !27
  %661 = extractelement <4 x float> %651, i64 3, !dbg !25
  %662 = call float @llvm.amdgcn.exp2.f32(float %661), !dbg !25
  %local.reduce73 = fadd float %local.reduce69, %662, !dbg !26, !frisk.local_reduction !1
  %663 = insertelement <4 x float> %660, float %662, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %663) #9, !dbg !27, !frisk.register_lifetimes !1
  %664 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %664) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %633, <4 x float> %631) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %666 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %649, <4 x float> %632) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %667 = shufflevector <8 x half> %483, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %668 = fsub <4 x float> %347, %452, !dbg !21
  %669 = extractelement <4 x float> %668, i64 0, !dbg !25
  %670 = call float @llvm.amdgcn.exp2.f32(float %669), !dbg !25
  %local.reduce105 = fadd float %local.reduce101, %670, !dbg !26, !frisk.local_reduction !1
  %671 = insertelement <4 x float> poison, float %670, i64 0, !dbg !27
  %672 = extractelement <4 x float> %668, i64 1, !dbg !25
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !25
  %local.reduce106 = fadd float %local.reduce102, %673, !dbg !26, !frisk.local_reduction !1
  %674 = insertelement <4 x float> %671, float %673, i64 1, !dbg !27
  %675 = extractelement <4 x float> %668, i64 2, !dbg !25
  %676 = call float @llvm.amdgcn.exp2.f32(float %675), !dbg !25
  %local.reduce107 = fadd float %local.reduce103, %676, !dbg !26, !frisk.local_reduction !1
  %677 = insertelement <4 x float> %674, float %676, i64 2, !dbg !27
  %678 = extractelement <4 x float> %668, i64 3, !dbg !25
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !25
  %local.reduce108 = fadd float %local.reduce104, %679, !dbg !26, !frisk.local_reduction !1
  %680 = insertelement <4 x float> %677, float %679, i64 3, !dbg !27
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %680) #9, !dbg !27, !frisk.register_lifetimes !1
  %681 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !27
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %681) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %682 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %667, <4 x float> %648) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %683 = shufflevector <8 x half> %483, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %684 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %683, <4 x float> %650) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %685 = fsub <4 x float> %399, %453, !dbg !21
  %686 = extractelement <4 x float> %685, i64 0, !dbg !25
  %687 = call float @llvm.amdgcn.exp2.f32(float %686), !dbg !25
  %local.reduce74 = fadd float %local.reduce70, %687, !dbg !26, !frisk.local_reduction !1
  %688 = insertelement <4 x float> poison, float %687, i64 0, !dbg !27
  %689 = extractelement <4 x float> %685, i64 1, !dbg !25
  %690 = call float @llvm.amdgcn.exp2.f32(float %689), !dbg !25
  %local.reduce75 = fadd float %local.reduce71, %690, !dbg !26, !frisk.local_reduction !1
  %691 = insertelement <4 x float> %688, float %690, i64 1, !dbg !27
  %692 = extractelement <4 x float> %685, i64 2, !dbg !25
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !25
  %local.reduce76 = fadd float %local.reduce72, %693, !dbg !26, !frisk.local_reduction !1
  %694 = insertelement <4 x float> %691, float %693, i64 2, !dbg !27
  %695 = extractelement <4 x float> %685, i64 3, !dbg !25
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !25
  %local.reduce77 = fadd float %local.reduce73, %696, !dbg !26, !frisk.local_reduction !1
  %697 = insertelement <4 x float> %694, float %696, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %697) #9, !dbg !27, !frisk.register_lifetimes !1
  %698 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %698) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %667, <4 x float> %665) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %700 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %683, <4 x float> %666) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %701 = shufflevector <8 x half> %485, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %702 = fsub <4 x float> %355, %452, !dbg !21
  %703 = extractelement <4 x float> %702, i64 0, !dbg !25
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !25
  %local.reduce109 = fadd float %local.reduce105, %704, !dbg !26, !frisk.local_reduction !1
  %705 = insertelement <4 x float> poison, float %704, i64 0, !dbg !27
  %706 = extractelement <4 x float> %702, i64 1, !dbg !25
  %707 = call float @llvm.amdgcn.exp2.f32(float %706), !dbg !25
  %local.reduce110 = fadd float %local.reduce106, %707, !dbg !26, !frisk.local_reduction !1
  %708 = insertelement <4 x float> %705, float %707, i64 1, !dbg !27
  %709 = extractelement <4 x float> %702, i64 2, !dbg !25
  %710 = call float @llvm.amdgcn.exp2.f32(float %709), !dbg !25
  %local.reduce111 = fadd float %local.reduce107, %710, !dbg !26, !frisk.local_reduction !1
  %711 = insertelement <4 x float> %708, float %710, i64 2, !dbg !27
  %712 = extractelement <4 x float> %702, i64 3, !dbg !25
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !25
  %local.reduce112 = fadd float %local.reduce108, %713, !dbg !26, !frisk.local_reduction !1
  %714 = insertelement <4 x float> %711, float %713, i64 3, !dbg !27
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %714) #9, !dbg !27, !frisk.register_lifetimes !1
  %715 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !27
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %715) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %701, <4 x float> %682) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %717 = shufflevector <8 x half> %485, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %717, <4 x float> %684) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %719 = fsub <4 x float> %405, %453, !dbg !21
  %720 = extractelement <4 x float> %719, i64 0, !dbg !25
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !25
  %local.reduce78 = fadd float %local.reduce74, %721, !dbg !26, !frisk.local_reduction !1
  %722 = insertelement <4 x float> poison, float %721, i64 0, !dbg !27
  %723 = extractelement <4 x float> %719, i64 1, !dbg !25
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !25
  %local.reduce79 = fadd float %local.reduce75, %724, !dbg !26, !frisk.local_reduction !1
  %725 = insertelement <4 x float> %722, float %724, i64 1, !dbg !27
  %726 = extractelement <4 x float> %719, i64 2, !dbg !25
  %727 = call float @llvm.amdgcn.exp2.f32(float %726), !dbg !25
  %local.reduce80 = fadd float %local.reduce76, %727, !dbg !26, !frisk.local_reduction !1
  %728 = insertelement <4 x float> %725, float %727, i64 2, !dbg !27
  %729 = extractelement <4 x float> %719, i64 3, !dbg !25
  %730 = call float @llvm.amdgcn.exp2.f32(float %729), !dbg !25
  %local.reduce81 = fadd float %local.reduce77, %730, !dbg !26, !frisk.local_reduction !1
  %731 = insertelement <4 x float> %728, float %730, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %731) #9, !dbg !27, !frisk.register_lifetimes !1
  %732 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %732) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %733 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %701, <4 x float> %699) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %717, <4 x float> %700) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %735 = bitcast <4 x i32> %486 to <8 x half>, !dbg !13
  %736 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %737 = fsub <4 x float> %362, %452, !dbg !21
  %738 = extractelement <4 x float> %737, i64 0, !dbg !25
  %739 = call float @llvm.amdgcn.exp2.f32(float %738), !dbg !25
  %local.reduce113 = fadd float %local.reduce109, %739, !dbg !26, !frisk.local_reduction !1
  %740 = insertelement <4 x float> poison, float %739, i64 0, !dbg !27
  %741 = extractelement <4 x float> %737, i64 1, !dbg !25
  %742 = call float @llvm.amdgcn.exp2.f32(float %741), !dbg !25
  %local.reduce114 = fadd float %local.reduce110, %742, !dbg !26, !frisk.local_reduction !1
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !26, !frisk.local_reduction !1
  %743 = insertelement <4 x float> %740, float %742, i64 1, !dbg !27
  %744 = extractelement <4 x float> %737, i64 2, !dbg !25
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !25
  %local.reduce115 = fadd float %local.reduce111, %745, !dbg !26, !frisk.local_reduction !1
  %746 = insertelement <4 x float> %743, float %745, i64 2, !dbg !27
  %747 = extractelement <4 x float> %737, i64 3, !dbg !25
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !25
  %local.reduce116 = fadd float %local.reduce112, %748, !dbg !26, !frisk.local_reduction !1
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !26, !frisk.local_reduction !1
  %749 = fadd float %local.reduce117, %local.reduce118, !dbg !26, !frisk.local_reduction !1
  %750 = fadd float %749, %461, !dbg !26
  %751 = insertelement <1 x float> poison, float %750, i64 0, !dbg !7
  %752 = insertvalue [2 x <1 x float>] poison, <1 x float> %751, 0, !dbg !7
  %753 = insertelement <4 x float> %746, float %748, i64 3, !dbg !27
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %753) #9, !dbg !27, !frisk.register_lifetimes !1
  %754 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !27
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %754) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %736, <4 x float> %716) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %756 = shufflevector <8 x half> %735, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %756, <4 x float> %718) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %758 = fsub <4 x float> %411, %453, !dbg !21
  %759 = extractelement <4 x float> %758, i64 0, !dbg !25
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !25
  %local.reduce82 = fadd float %local.reduce78, %760, !dbg !26, !frisk.local_reduction !1
  %761 = insertelement <4 x float> poison, float %760, i64 0, !dbg !27
  %762 = extractelement <4 x float> %758, i64 1, !dbg !25
  %763 = call float @llvm.amdgcn.exp2.f32(float %762), !dbg !25
  %local.reduce83 = fadd float %local.reduce79, %763, !dbg !26, !frisk.local_reduction !1
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !26, !frisk.local_reduction !1
  %764 = insertelement <4 x float> %761, float %763, i64 1, !dbg !27
  %765 = extractelement <4 x float> %758, i64 2, !dbg !25
  %766 = call float @llvm.amdgcn.exp2.f32(float %765), !dbg !25
  %local.reduce84 = fadd float %local.reduce80, %766, !dbg !26, !frisk.local_reduction !1
  %767 = insertelement <4 x float> %764, float %766, i64 2, !dbg !27
  %768 = extractelement <4 x float> %758, i64 3, !dbg !25
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !25
  %local.reduce85 = fadd float %local.reduce81, %769, !dbg !26, !frisk.local_reduction !1
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !26, !frisk.local_reduction !1
  %770 = fadd float %local.reduce86, %local.reduce87, !dbg !26, !frisk.local_reduction !1
  %771 = fadd float %770, %469, !dbg !26
  %772 = insertelement <1 x float> poison, float %771, i64 0, !dbg !7
  %773 = insertvalue [2 x <1 x float>] %752, <1 x float> %772, 1, !dbg !7
  %774 = insertelement <4 x float> %767, float %769, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %774) #9, !dbg !27, !frisk.register_lifetimes !1
  %775 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %775) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13, !frisk.softmax_fragments !1
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %736, <4 x float> %733) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %756, <4 x float> %734) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1462 = or disjoint i64 %191, %address.local921, !dbg !15
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !15
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !15
  %vdb.address1471 = or disjoint i64 %191, %address.local927, !dbg !15
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !15
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !15
  %778 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %779 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !13
  %780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !13
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !13
  %782 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !13
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !13
  %784 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !13
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !13
  %786 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %787 = bitcast <4 x i32> %786 to <8 x half>, !dbg !13
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !13
  %788 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %789 = bitcast <4 x i32> %788 to <8 x half>, !dbg !13
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !13
  %790 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %791 = bitcast <4 x i32> %790 to <8 x half>, !dbg !13
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !13
  %792 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %793 = bitcast <4 x i32> %792 to <8 x half>, !dbg !13
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !13
  %794 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %795 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %796 = extractvalue [2 x <8 x float>] %93, 0, !dbg !28
  %797 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %798 = fmul <4 x float> %797, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %795, <4 x float> %798) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %800 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %801 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %802 = fmul <4 x float> %801, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %800, <4 x float> %802) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %804 = extractvalue [2 x <8 x float>] %93, 1, !dbg !28
  %805 = shufflevector <8 x float> %804, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %806 = fmul <4 x float> %805, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %807 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %795, <4 x float> %806) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %808 = shufflevector <8 x float> %804, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %809 = fmul <4 x float> %808, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %800, <4 x float> %809) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %811 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %811, <4 x float> %799) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %813 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %813, <4 x float> %803) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %811, <4 x float> %807) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %813, <4 x float> %810) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %817 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %817, <4 x float> %812) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %819 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %819, <4 x float> %814) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %817, <4 x float> %815) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %819, <4 x float> %816) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %823 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %823, <4 x float> %818) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %825 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %825, <4 x float> %820) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %823, <4 x float> %821) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %825, <4 x float> %822) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %829 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %829, <4 x float> %824) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %831 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %831, <4 x float> %826) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %829, <4 x float> %827) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %831, <4 x float> %828) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %835 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %835, <4 x float> %830) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %837 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %837, <4 x float> %832) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %835, <4 x float> %833) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %837, <4 x float> %834) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %841 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %841, <4 x float> %836) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %843 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %843, <4 x float> %838) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %841, <4 x float> %839) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %843, <4 x float> %840) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %847 = bitcast <4 x i32> %794 to <8 x half>, !dbg !13
  %848 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %848, <4 x float> %842) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %850 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %850, <4 x float> %844) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %848, <4 x float> %845) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %850, <4 x float> %846) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %vdb.address1480 = or disjoint i64 %244, %address.local921, !dbg !15
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !15
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !15
  %vdb.address1489 = or disjoint i64 %244, %address.local927, !dbg !15
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !15
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !15
  %854 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %855 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !15
  %856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !13
  %858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !13
  %860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !13
  %862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !13
  %864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !13
  %866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !13
  %868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !13
  %870 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %871 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %872 = extractvalue [2 x <8 x float>] %94, 0, !dbg !28
  %873 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %874 = fmul <4 x float> %873, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %871, <4 x float> %874) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %876 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %877 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %878 = fmul <4 x float> %877, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %876, <4 x float> %878) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %880 = extractvalue [2 x <8 x float>] %94, 1, !dbg !28
  %881 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %882 = fmul <4 x float> %881, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %871, <4 x float> %882) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %884 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %885 = fmul <4 x float> %884, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %876, <4 x float> %885) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %887 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %887, <4 x float> %875) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %889 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %889, <4 x float> %879) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %887, <4 x float> %883) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %889, <4 x float> %886) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %893 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %893, <4 x float> %888) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %895 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %895, <4 x float> %890) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %893, <4 x float> %891) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %895, <4 x float> %892) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %899 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %899, <4 x float> %894) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %901 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %901, <4 x float> %896) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %899, <4 x float> %897) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %901, <4 x float> %898) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %905 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %905, <4 x float> %900) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %907 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %907, <4 x float> %902) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %905, <4 x float> %903) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %907, <4 x float> %904) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %911 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %911, <4 x float> %906) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %913 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %913, <4 x float> %908) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %911, <4 x float> %909) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %913, <4 x float> %910) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %917 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %917, <4 x float> %912) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %919 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %919, <4 x float> %914) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %917, <4 x float> %915) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %919, <4 x float> %916) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %923 = bitcast <4 x i32> %870 to <8 x half>, !dbg !13
  %924 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %924, <4 x float> %918) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %926 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %926, <4 x float> %920) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %924, <4 x float> %921) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %926, <4 x float> %922) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !13
  %932 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %933 = bitcast <4 x i32> %932 to <8 x half>, !dbg !13
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !13
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !13
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !13
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !13
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !13
  %944 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !13, !frisk.v_double_buffer !34
  %945 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  %946 = extractvalue [2 x <8 x float>] %95, 0, !dbg !28
  %947 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %948 = fmul <4 x float> %947, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %945, <4 x float> %948) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %950 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %951 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %952 = fmul <4 x float> %951, %458, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %950, <4 x float> %952) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %954 = extractvalue [2 x <8 x float>] %95, 1, !dbg !28
  %955 = shufflevector <8 x float> %954, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %956 = fmul <4 x float> %955, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %945, <4 x float> %956) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %958 = shufflevector <8 x float> %954, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %959 = fmul <4 x float> %958, %466, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %950, <4 x float> %959) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %961 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %961, <4 x float> %949) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %963 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %963, <4 x float> %953) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %961, <4 x float> %957) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %963, <4 x float> %960) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %967 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %967, <4 x float> %962) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %969 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %969, <4 x float> %964) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %967, <4 x float> %965) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %969, <4 x float> %966) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %973 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %973, <4 x float> %968) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %975 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %975, <4 x float> %970) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %973, <4 x float> %971) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %975, <4 x float> %972) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %979 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %979, <4 x float> %974) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %981 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %981, <4 x float> %976) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %979, <4 x float> %977) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %981, <4 x float> %978) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %985 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %985, <4 x float> %980) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %987 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %987, <4 x float> %982) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %985, <4 x float> %983) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %987, <4 x float> %984) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %991 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %991, <4 x float> %986) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %993 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %993, <4 x float> %988) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %991, <4 x float> %989) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %993, <4 x float> %990) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %997 = bitcast <4 x i32> %944 to <8 x half>, !dbg !13
  %998 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %998, <4 x float> %992) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1000 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1000, <4 x float> %994) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %998, <4 x float> %995) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1000, <4 x float> %996) #9, !dbg !13, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !7
  %1004 = shufflevector <4 x float> %755, <4 x float> %757, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1005 = insertvalue [2 x <8 x float>] poison, <8 x float> %1004, 0, !dbg !7
  %1006 = shufflevector <4 x float> %776, <4 x float> %777, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1007 = insertvalue [2 x <8 x float>] %1005, <8 x float> %1006, 1, !dbg !7
  %1008 = insertvalue [2 x <1 x float>] poison, <1 x float> %438, 0, !dbg !7
  %1009 = insertvalue [2 x <1 x float>] %1008, <1 x float> %451, 1, !dbg !7
  %1010 = shufflevector <4 x float> %849, <4 x float> %851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1011 = insertvalue [2 x <8 x float>] poison, <8 x float> %1010, 0, !dbg !7
  %1012 = shufflevector <4 x float> %852, <4 x float> %853, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1013 = insertvalue [2 x <8 x float>] %1011, <8 x float> %1012, 1, !dbg !7
  %1014 = shufflevector <4 x float> %925, <4 x float> %927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1015 = insertvalue [2 x <8 x float>] poison, <8 x float> %1014, 0, !dbg !7
  %1016 = shufflevector <4 x float> %928, <4 x float> %929, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1017 = insertvalue [2 x <8 x float>] %1015, <8 x float> %1016, 1, !dbg !7
  %1018 = shufflevector <4 x float> %999, <4 x float> %1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1019 = insertvalue [2 x <8 x float>] poison, <8 x float> %1018, 0, !dbg !7
  %1020 = shufflevector <4 x float> %1002, <4 x float> %1003, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !7
  %1021 = insertvalue [2 x <8 x float>] %1019, <8 x float> %1020, 1, !dbg !7
  fence syncscope("workgroup") release, !dbg !7
  call void @llvm.amdgcn.s.barrier(), !dbg !7
  fence syncscope("workgroup") acquire, !dbg !7
  %1022 = add nuw nsw i64 %89, 1, !dbg !7
  br label %88, !dbg !7

1023:                                             ; preds = %88
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %91, 1, !dbg !7
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !7
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !7
  %1024 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !7
  %1025 = xor i32 %1024, 64, !dbg !7
  %1026 = bitcast float %lane.sum.local185 to i32, !dbg !7
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1025, i32 %1026), !dbg !7
  %1027 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !7
  %lane.sum.final189 = fadd float %lane.sum.local185, %1027, !dbg !7
  %1028 = xor i32 %1024, 128, !dbg !7
  %1029 = bitcast float %lane.sum.final189 to i32, !dbg !7
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1029), !dbg !7
  %1030 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !7
  %lane.sum.final191 = fadd float %lane.sum.final189, %1030, !dbg !7
  %1031 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !7
  %lane.sum.field = extractvalue [2 x <1 x float>] %91, 0, !dbg !7
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !7
  %1032 = bitcast float %lane.sum.local to i32, !dbg !7
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1025, i32 %1032), !dbg !7
  %1033 = bitcast i32 %lane.sum.shuffle to float, !dbg !7
  %lane.sum.final = fadd float %lane.sum.local, %1033, !dbg !7
  %1034 = bitcast float %lane.sum.final to i32, !dbg !7
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1034), !dbg !7
  %1035 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !7
  %lane.sum.final183 = fadd float %lane.sum.final, %1035, !dbg !7
  %1036 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !7
  %1037 = and i32 %kdma.tid, 16, !dbg !7
  %.not = icmp eq i32 %1037, 0, !dbg !7
  %1038 = and i32 %kdma.tid, 32, !dbg !7
  %.not36 = icmp eq i32 %1038, 0, !dbg !7
  %1039 = extractvalue [2 x <8 x float>] %95, 0, !dbg !7
  %1040 = extractelement <8 x float> %1039, i64 0, !dbg !7
  %1041 = extractelement <8 x float> %1039, i64 1, !dbg !7
  %1042 = extractelement <8 x float> %1039, i64 2, !dbg !7
  %1043 = extractelement <8 x float> %1039, i64 3, !dbg !7
  %1044 = select i1 %.not, float %1041, float %1040, !dbg !7
  %1045 = add nuw nsw i32 %lane.sum.lane187, 64, !dbg !7
  %1046 = xor i32 %lane.sum.lane187, 16, !dbg !7
  %1047 = and i32 %1045, 192, !dbg !7
  %1048 = icmp samesign ult i32 %1046, %1047, !dbg !7
  %1049 = select i1 %1048, i32 %1046, i32 %lane.sum.lane187, !dbg !7
  %1050 = shl nuw nsw i32 %1049, 2, !dbg !7
  %1051 = bitcast float %1044 to i32, !dbg !7
  %1052 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1051), !dbg !7
  %1053 = bitcast i32 %1052 to float, !dbg !7
  %1054 = select i1 %.not, float %1040, float %1053, !dbg !7
  %1055 = select i1 %.not, float %1053, float %1041, !dbg !7
  %1056 = select i1 %.not, float %1043, float %1042, !dbg !7
  %1057 = bitcast float %1056 to i32, !dbg !7
  %1058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1057), !dbg !7
  %1059 = bitcast i32 %1058 to float, !dbg !7
  %1060 = select i1 %.not, float %1042, float %1059, !dbg !7
  %1061 = select i1 %.not, float %1059, float %1043, !dbg !7
  %1062 = select i1 %.not36, float %1060, float %1054, !dbg !7
  %1063 = xor i32 %lane.sum.lane187, 32, !dbg !7
  %1064 = icmp samesign ult i32 %1063, %1047, !dbg !7
  %1065 = select i1 %1064, i32 %1063, i32 %lane.sum.lane187, !dbg !7
  %1066 = shl nuw nsw i32 %1065, 2, !dbg !7
  %1067 = bitcast float %1062 to i32, !dbg !7
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1067), !dbg !7
  %1069 = bitcast i32 %1068 to float, !dbg !7
  %1070 = select i1 %.not36, float %1054, float %1069, !dbg !7
  %1071 = select i1 %.not36, float %1069, float %1060, !dbg !7
  %1072 = select i1 %.not36, float %1061, float %1055, !dbg !7
  %1073 = bitcast float %1072 to i32, !dbg !7
  %1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1073), !dbg !7
  %1075 = bitcast i32 %1074 to float, !dbg !7
  %1076 = select i1 %.not36, float %1055, float %1075, !dbg !7
  %1077 = select i1 %.not36, float %1075, float %1061, !dbg !7
  %1078 = extractvalue [2 x <8 x float>] %94, 0, !dbg !7
  %1079 = extractelement <8 x float> %1078, i64 0, !dbg !7
  %1080 = extractelement <8 x float> %1078, i64 1, !dbg !7
  %1081 = extractelement <8 x float> %1078, i64 2, !dbg !7
  %1082 = extractelement <8 x float> %1078, i64 3, !dbg !7
  %1083 = select i1 %.not, float %1080, float %1079, !dbg !7
  %1084 = bitcast float %1083 to i32, !dbg !7
  %1085 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1084), !dbg !7
  %1086 = bitcast i32 %1085 to float, !dbg !7
  %1087 = select i1 %.not, float %1079, float %1086, !dbg !7
  %1088 = select i1 %.not, float %1086, float %1080, !dbg !7
  %1089 = select i1 %.not, float %1082, float %1081, !dbg !7
  %1090 = bitcast float %1089 to i32, !dbg !7
  %1091 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1090), !dbg !7
  %1092 = bitcast i32 %1091 to float, !dbg !7
  %1093 = select i1 %.not, float %1081, float %1092, !dbg !7
  %1094 = select i1 %.not, float %1092, float %1082, !dbg !7
  %1095 = select i1 %.not36, float %1093, float %1087, !dbg !7
  %1096 = bitcast float %1095 to i32, !dbg !7
  %1097 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1096), !dbg !7
  %1098 = bitcast i32 %1097 to float, !dbg !7
  %1099 = select i1 %.not36, float %1087, float %1098, !dbg !7
  %1100 = select i1 %.not36, float %1098, float %1093, !dbg !7
  %1101 = select i1 %.not36, float %1094, float %1088, !dbg !7
  %1102 = bitcast float %1101 to i32, !dbg !7
  %1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1102), !dbg !7
  %1104 = bitcast i32 %1103 to float, !dbg !7
  %1105 = select i1 %.not36, float %1088, float %1104, !dbg !7
  %1106 = select i1 %.not36, float %1104, float %1094, !dbg !7
  %1107 = extractvalue [2 x <8 x float>] %93, 0, !dbg !7
  %1108 = extractelement <8 x float> %1107, i64 0, !dbg !7
  %1109 = extractelement <8 x float> %1107, i64 1, !dbg !7
  %1110 = extractelement <8 x float> %1107, i64 2, !dbg !7
  %1111 = extractelement <8 x float> %1107, i64 3, !dbg !7
  %1112 = select i1 %.not, float %1109, float %1108, !dbg !7
  %1113 = bitcast float %1112 to i32, !dbg !7
  %1114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1113), !dbg !7
  %1115 = bitcast i32 %1114 to float, !dbg !7
  %1116 = select i1 %.not, float %1108, float %1115, !dbg !7
  %1117 = select i1 %.not, float %1115, float %1109, !dbg !7
  %1118 = select i1 %.not, float %1111, float %1110, !dbg !7
  %1119 = bitcast float %1118 to i32, !dbg !7
  %1120 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1119), !dbg !7
  %1121 = bitcast i32 %1120 to float, !dbg !7
  %1122 = select i1 %.not, float %1110, float %1121, !dbg !7
  %1123 = select i1 %.not, float %1121, float %1111, !dbg !7
  %1124 = select i1 %.not36, float %1122, float %1116, !dbg !7
  %1125 = bitcast float %1124 to i32, !dbg !7
  %1126 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1125), !dbg !7
  %1127 = bitcast i32 %1126 to float, !dbg !7
  %1128 = select i1 %.not36, float %1116, float %1127, !dbg !7
  %1129 = select i1 %.not36, float %1127, float %1122, !dbg !7
  %1130 = select i1 %.not36, float %1123, float %1117, !dbg !7
  %1131 = bitcast float %1130 to i32, !dbg !7
  %1132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1131), !dbg !7
  %1133 = bitcast i32 %1132 to float, !dbg !7
  %1134 = select i1 %.not36, float %1117, float %1133, !dbg !7
  %1135 = select i1 %.not36, float %1133, float %1123, !dbg !7
  %1136 = extractvalue [2 x <8 x float>] %90, 0, !dbg !7
  %1137 = extractelement <8 x float> %1136, i64 0, !dbg !7
  %1138 = extractelement <8 x float> %1136, i64 1, !dbg !7
  %1139 = extractelement <8 x float> %1136, i64 2, !dbg !7
  %1140 = extractelement <8 x float> %1136, i64 3, !dbg !7
  %1141 = select i1 %.not, float %1138, float %1137, !dbg !7
  %1142 = bitcast float %1141 to i32, !dbg !7
  %1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1142), !dbg !7
  %1144 = bitcast i32 %1143 to float, !dbg !7
  %1145 = select i1 %.not, float %1137, float %1144, !dbg !7
  %1146 = select i1 %.not, float %1144, float %1138, !dbg !7
  %1147 = select i1 %.not, float %1140, float %1139, !dbg !7
  %1148 = bitcast float %1147 to i32, !dbg !7
  %1149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1148), !dbg !7
  %1150 = bitcast i32 %1149 to float, !dbg !7
  %1151 = select i1 %.not, float %1139, float %1150, !dbg !7
  %1152 = select i1 %.not, float %1150, float %1140, !dbg !7
  %1153 = select i1 %.not36, float %1151, float %1145, !dbg !7
  %1154 = bitcast float %1153 to i32, !dbg !7
  %1155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1154), !dbg !7
  %1156 = select i1 %.not36, float %1152, float %1146, !dbg !7
  %1157 = bitcast float %1156 to i32, !dbg !7
  %1158 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1157), !dbg !7
  %1159 = shufflevector <1 x float> %1036, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1160 = extractelement <8 x float> %1039, i64 4, !dbg !7
  %1161 = extractelement <8 x float> %1039, i64 5, !dbg !7
  %1162 = extractelement <8 x float> %1039, i64 6, !dbg !7
  %1163 = extractelement <8 x float> %1039, i64 7, !dbg !7
  %1164 = select i1 %.not, float %1161, float %1160, !dbg !7
  %1165 = bitcast float %1164 to i32, !dbg !7
  %1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1165), !dbg !7
  %1167 = bitcast i32 %1166 to float, !dbg !7
  %1168 = select i1 %.not, float %1160, float %1167, !dbg !7
  %1169 = select i1 %.not, float %1167, float %1161, !dbg !7
  %1170 = select i1 %.not, float %1163, float %1162, !dbg !7
  %1171 = bitcast float %1170 to i32, !dbg !7
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1171), !dbg !7
  %1173 = bitcast i32 %1172 to float, !dbg !7
  %1174 = select i1 %.not, float %1162, float %1173, !dbg !7
  %1175 = select i1 %.not, float %1173, float %1163, !dbg !7
  %1176 = select i1 %.not36, float %1174, float %1168, !dbg !7
  %1177 = bitcast float %1176 to i32, !dbg !7
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1177), !dbg !7
  %1179 = bitcast i32 %1178 to float, !dbg !7
  %1180 = select i1 %.not36, float %1168, float %1179, !dbg !7
  %1181 = select i1 %.not36, float %1179, float %1174, !dbg !7
  %1182 = select i1 %.not36, float %1175, float %1169, !dbg !7
  %1183 = bitcast float %1182 to i32, !dbg !7
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1183), !dbg !7
  %1185 = bitcast i32 %1184 to float, !dbg !7
  %1186 = select i1 %.not36, float %1169, float %1185, !dbg !7
  %1187 = select i1 %.not36, float %1185, float %1175, !dbg !7
  %1188 = extractelement <8 x float> %1078, i64 4, !dbg !7
  %1189 = extractelement <8 x float> %1078, i64 5, !dbg !7
  %1190 = extractelement <8 x float> %1078, i64 6, !dbg !7
  %1191 = extractelement <8 x float> %1078, i64 7, !dbg !7
  %1192 = select i1 %.not, float %1189, float %1188, !dbg !7
  %1193 = bitcast float %1192 to i32, !dbg !7
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1193), !dbg !7
  %1195 = bitcast i32 %1194 to float, !dbg !7
  %1196 = select i1 %.not, float %1188, float %1195, !dbg !7
  %1197 = select i1 %.not, float %1195, float %1189, !dbg !7
  %1198 = select i1 %.not, float %1191, float %1190, !dbg !7
  %1199 = bitcast float %1198 to i32, !dbg !7
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1199), !dbg !7
  %1201 = bitcast i32 %1200 to float, !dbg !7
  %1202 = select i1 %.not, float %1190, float %1201, !dbg !7
  %1203 = select i1 %.not, float %1201, float %1191, !dbg !7
  %1204 = select i1 %.not36, float %1202, float %1196, !dbg !7
  %1205 = bitcast float %1204 to i32, !dbg !7
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1205), !dbg !7
  %1207 = bitcast i32 %1206 to float, !dbg !7
  %1208 = select i1 %.not36, float %1196, float %1207, !dbg !7
  %1209 = select i1 %.not36, float %1207, float %1202, !dbg !7
  %1210 = select i1 %.not36, float %1203, float %1197, !dbg !7
  %1211 = bitcast float %1210 to i32, !dbg !7
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1211), !dbg !7
  %1213 = bitcast i32 %1212 to float, !dbg !7
  %1214 = select i1 %.not36, float %1197, float %1213, !dbg !7
  %1215 = select i1 %.not36, float %1213, float %1203, !dbg !7
  %1216 = extractelement <8 x float> %1107, i64 4, !dbg !7
  %1217 = extractelement <8 x float> %1107, i64 5, !dbg !7
  %1218 = extractelement <8 x float> %1107, i64 6, !dbg !7
  %1219 = extractelement <8 x float> %1107, i64 7, !dbg !7
  %1220 = select i1 %.not, float %1217, float %1216, !dbg !7
  %1221 = bitcast float %1220 to i32, !dbg !7
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1221), !dbg !7
  %1223 = bitcast i32 %1222 to float, !dbg !7
  %1224 = select i1 %.not, float %1216, float %1223, !dbg !7
  %1225 = select i1 %.not, float %1223, float %1217, !dbg !7
  %1226 = select i1 %.not, float %1219, float %1218, !dbg !7
  %1227 = bitcast float %1226 to i32, !dbg !7
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1227), !dbg !7
  %1229 = bitcast i32 %1228 to float, !dbg !7
  %1230 = select i1 %.not, float %1218, float %1229, !dbg !7
  %1231 = select i1 %.not, float %1229, float %1219, !dbg !7
  %1232 = select i1 %.not36, float %1230, float %1224, !dbg !7
  %1233 = bitcast float %1232 to i32, !dbg !7
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1233), !dbg !7
  %1235 = bitcast i32 %1234 to float, !dbg !7
  %1236 = select i1 %.not36, float %1224, float %1235, !dbg !7
  %1237 = select i1 %.not36, float %1235, float %1230, !dbg !7
  %1238 = select i1 %.not36, float %1231, float %1225, !dbg !7
  %1239 = bitcast float %1238 to i32, !dbg !7
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1239), !dbg !7
  %1241 = bitcast i32 %1240 to float, !dbg !7
  %1242 = select i1 %.not36, float %1225, float %1241, !dbg !7
  %1243 = select i1 %.not36, float %1241, float %1231, !dbg !7
  %1244 = extractelement <8 x float> %1136, i64 4, !dbg !7
  %1245 = extractelement <8 x float> %1136, i64 5, !dbg !7
  %1246 = extractelement <8 x float> %1136, i64 6, !dbg !7
  %1247 = extractelement <8 x float> %1136, i64 7, !dbg !7
  %1248 = select i1 %.not, float %1245, float %1244, !dbg !7
  %1249 = bitcast float %1248 to i32, !dbg !7
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1249), !dbg !7
  %1251 = bitcast i32 %1250 to float, !dbg !7
  %1252 = select i1 %.not, float %1244, float %1251, !dbg !7
  %1253 = select i1 %.not, float %1251, float %1245, !dbg !7
  %1254 = select i1 %.not, float %1247, float %1246, !dbg !7
  %1255 = bitcast float %1254 to i32, !dbg !7
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1255), !dbg !7
  %1257 = bitcast i32 %1256 to float, !dbg !7
  %1258 = select i1 %.not, float %1246, float %1257, !dbg !7
  %1259 = select i1 %.not, float %1257, float %1247, !dbg !7
  %1260 = select i1 %.not36, float %1258, float %1252, !dbg !7
  %1261 = bitcast float %1260 to i32, !dbg !7
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1261), !dbg !7
  %1263 = select i1 %.not36, float %1259, float %1253, !dbg !7
  %1264 = bitcast float %1263 to i32, !dbg !7
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1264), !dbg !7
  %1266 = extractvalue [2 x <8 x float>] %95, 1, !dbg !7
  %1267 = extractelement <8 x float> %1266, i64 0, !dbg !7
  %1268 = extractelement <8 x float> %1266, i64 1, !dbg !7
  %1269 = extractelement <8 x float> %1266, i64 2, !dbg !7
  %1270 = extractelement <8 x float> %1266, i64 3, !dbg !7
  %1271 = select i1 %.not, float %1268, float %1267, !dbg !7
  %1272 = bitcast float %1271 to i32, !dbg !7
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1272), !dbg !7
  %1274 = bitcast i32 %1273 to float, !dbg !7
  %1275 = select i1 %.not, float %1267, float %1274, !dbg !7
  %1276 = select i1 %.not, float %1274, float %1268, !dbg !7
  %1277 = select i1 %.not, float %1270, float %1269, !dbg !7
  %1278 = bitcast float %1277 to i32, !dbg !7
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1278), !dbg !7
  %1280 = bitcast i32 %1279 to float, !dbg !7
  %1281 = select i1 %.not, float %1269, float %1280, !dbg !7
  %1282 = select i1 %.not, float %1280, float %1270, !dbg !7
  %1283 = select i1 %.not36, float %1281, float %1275, !dbg !7
  %1284 = bitcast float %1283 to i32, !dbg !7
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1284), !dbg !7
  %1286 = bitcast i32 %1285 to float, !dbg !7
  %1287 = select i1 %.not36, float %1275, float %1286, !dbg !7
  %1288 = select i1 %.not36, float %1286, float %1281, !dbg !7
  %1289 = select i1 %.not36, float %1282, float %1276, !dbg !7
  %1290 = bitcast float %1289 to i32, !dbg !7
  %1291 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1290), !dbg !7
  %1292 = bitcast i32 %1291 to float, !dbg !7
  %1293 = select i1 %.not36, float %1276, float %1292, !dbg !7
  %1294 = select i1 %.not36, float %1292, float %1282, !dbg !7
  %1295 = extractvalue [2 x <8 x float>] %94, 1, !dbg !7
  %1296 = extractelement <8 x float> %1295, i64 0, !dbg !7
  %1297 = extractelement <8 x float> %1295, i64 1, !dbg !7
  %1298 = extractelement <8 x float> %1295, i64 2, !dbg !7
  %1299 = extractelement <8 x float> %1295, i64 3, !dbg !7
  %1300 = select i1 %.not, float %1297, float %1296, !dbg !7
  %1301 = bitcast float %1300 to i32, !dbg !7
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1301), !dbg !7
  %1303 = bitcast i32 %1302 to float, !dbg !7
  %1304 = select i1 %.not, float %1296, float %1303, !dbg !7
  %1305 = select i1 %.not, float %1303, float %1297, !dbg !7
  %1306 = select i1 %.not, float %1299, float %1298, !dbg !7
  %1307 = bitcast float %1306 to i32, !dbg !7
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1307), !dbg !7
  %1309 = bitcast i32 %1308 to float, !dbg !7
  %1310 = select i1 %.not, float %1298, float %1309, !dbg !7
  %1311 = select i1 %.not, float %1309, float %1299, !dbg !7
  %1312 = select i1 %.not36, float %1310, float %1304, !dbg !7
  %1313 = bitcast float %1312 to i32, !dbg !7
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1313), !dbg !7
  %1315 = bitcast i32 %1314 to float, !dbg !7
  %1316 = select i1 %.not36, float %1304, float %1315, !dbg !7
  %1317 = select i1 %.not36, float %1315, float %1310, !dbg !7
  %1318 = select i1 %.not36, float %1311, float %1305, !dbg !7
  %1319 = bitcast float %1318 to i32, !dbg !7
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1319), !dbg !7
  %1321 = bitcast i32 %1320 to float, !dbg !7
  %1322 = select i1 %.not36, float %1305, float %1321, !dbg !7
  %1323 = select i1 %.not36, float %1321, float %1311, !dbg !7
  %1324 = extractvalue [2 x <8 x float>] %93, 1, !dbg !7
  %1325 = extractelement <8 x float> %1324, i64 0, !dbg !7
  %1326 = extractelement <8 x float> %1324, i64 1, !dbg !7
  %1327 = extractelement <8 x float> %1324, i64 2, !dbg !7
  %1328 = extractelement <8 x float> %1324, i64 3, !dbg !7
  %1329 = select i1 %.not, float %1326, float %1325, !dbg !7
  %1330 = bitcast float %1329 to i32, !dbg !7
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1330), !dbg !7
  %1332 = bitcast i32 %1331 to float, !dbg !7
  %1333 = select i1 %.not, float %1325, float %1332, !dbg !7
  %1334 = select i1 %.not, float %1332, float %1326, !dbg !7
  %1335 = select i1 %.not, float %1328, float %1327, !dbg !7
  %1336 = bitcast float %1335 to i32, !dbg !7
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1336), !dbg !7
  %1338 = bitcast i32 %1337 to float, !dbg !7
  %1339 = select i1 %.not, float %1327, float %1338, !dbg !7
  %1340 = select i1 %.not, float %1338, float %1328, !dbg !7
  %1341 = select i1 %.not36, float %1339, float %1333, !dbg !7
  %1342 = bitcast float %1341 to i32, !dbg !7
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1342), !dbg !7
  %1344 = bitcast i32 %1343 to float, !dbg !7
  %1345 = select i1 %.not36, float %1333, float %1344, !dbg !7
  %1346 = select i1 %.not36, float %1344, float %1339, !dbg !7
  %1347 = select i1 %.not36, float %1340, float %1334, !dbg !7
  %1348 = bitcast float %1347 to i32, !dbg !7
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1348), !dbg !7
  %1350 = bitcast i32 %1349 to float, !dbg !7
  %1351 = select i1 %.not36, float %1334, float %1350, !dbg !7
  %1352 = select i1 %.not36, float %1350, float %1340, !dbg !7
  %1353 = extractvalue [2 x <8 x float>] %90, 1, !dbg !7
  %1354 = extractelement <8 x float> %1353, i64 0, !dbg !7
  %1355 = extractelement <8 x float> %1353, i64 1, !dbg !7
  %1356 = extractelement <8 x float> %1353, i64 2, !dbg !7
  %1357 = extractelement <8 x float> %1353, i64 3, !dbg !7
  %1358 = select i1 %.not, float %1355, float %1354, !dbg !7
  %1359 = bitcast float %1358 to i32, !dbg !7
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1359), !dbg !7
  %1361 = bitcast i32 %1360 to float, !dbg !7
  %1362 = select i1 %.not, float %1354, float %1361, !dbg !7
  %1363 = select i1 %.not, float %1361, float %1355, !dbg !7
  %1364 = select i1 %.not, float %1357, float %1356, !dbg !7
  %1365 = bitcast float %1364 to i32, !dbg !7
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1365), !dbg !7
  %1367 = bitcast i32 %1366 to float, !dbg !7
  %1368 = select i1 %.not, float %1356, float %1367, !dbg !7
  %1369 = select i1 %.not, float %1367, float %1357, !dbg !7
  %1370 = select i1 %.not36, float %1368, float %1362, !dbg !7
  %1371 = bitcast float %1370 to i32, !dbg !7
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1371), !dbg !7
  %1373 = select i1 %.not36, float %1369, float %1363, !dbg !7
  %1374 = bitcast float %1373 to i32, !dbg !7
  %1375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1374), !dbg !7
  %1376 = shufflevector <1 x float> %1031, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !16
  %1377 = extractelement <8 x float> %1266, i64 4, !dbg !7
  %1378 = extractelement <8 x float> %1266, i64 5, !dbg !7
  %1379 = extractelement <8 x float> %1266, i64 6, !dbg !7
  %1380 = extractelement <8 x float> %1266, i64 7, !dbg !7
  %1381 = select i1 %.not, float %1378, float %1377, !dbg !7
  %1382 = bitcast float %1381 to i32, !dbg !7
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1382), !dbg !7
  %1384 = bitcast i32 %1383 to float, !dbg !7
  %1385 = select i1 %.not, float %1377, float %1384, !dbg !7
  %1386 = select i1 %.not, float %1384, float %1378, !dbg !7
  %1387 = select i1 %.not, float %1380, float %1379, !dbg !7
  %1388 = bitcast float %1387 to i32, !dbg !7
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1388), !dbg !7
  %1390 = bitcast i32 %1389 to float, !dbg !7
  %1391 = select i1 %.not, float %1379, float %1390, !dbg !7
  %1392 = select i1 %.not, float %1390, float %1380, !dbg !7
  %1393 = select i1 %.not36, float %1391, float %1385, !dbg !7
  %1394 = bitcast float %1393 to i32, !dbg !7
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1394), !dbg !7
  %1396 = bitcast i32 %1395 to float, !dbg !7
  %1397 = select i1 %.not36, float %1385, float %1396, !dbg !7
  %1398 = select i1 %.not36, float %1396, float %1391, !dbg !7
  %1399 = select i1 %.not36, float %1392, float %1386, !dbg !7
  %1400 = bitcast float %1399 to i32, !dbg !7
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1400), !dbg !7
  %1402 = bitcast i32 %1401 to float, !dbg !7
  %1403 = select i1 %.not36, float %1386, float %1402, !dbg !7
  %1404 = select i1 %.not36, float %1402, float %1392, !dbg !7
  %1405 = extractelement <8 x float> %1295, i64 4, !dbg !7
  %1406 = extractelement <8 x float> %1295, i64 5, !dbg !7
  %1407 = extractelement <8 x float> %1295, i64 6, !dbg !7
  %1408 = extractelement <8 x float> %1295, i64 7, !dbg !7
  %1409 = select i1 %.not, float %1406, float %1405, !dbg !7
  %1410 = bitcast float %1409 to i32, !dbg !7
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1410), !dbg !7
  %1412 = bitcast i32 %1411 to float, !dbg !7
  %1413 = select i1 %.not, float %1405, float %1412, !dbg !7
  %1414 = select i1 %.not, float %1412, float %1406, !dbg !7
  %1415 = select i1 %.not, float %1408, float %1407, !dbg !7
  %1416 = bitcast float %1415 to i32, !dbg !7
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1416), !dbg !7
  %1418 = bitcast i32 %1417 to float, !dbg !7
  %1419 = select i1 %.not, float %1407, float %1418, !dbg !7
  %1420 = select i1 %.not, float %1418, float %1408, !dbg !7
  %1421 = select i1 %.not36, float %1419, float %1413, !dbg !7
  %1422 = bitcast float %1421 to i32, !dbg !7
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1422), !dbg !7
  %1424 = bitcast i32 %1423 to float, !dbg !7
  %1425 = select i1 %.not36, float %1413, float %1424, !dbg !7
  %1426 = select i1 %.not36, float %1424, float %1419, !dbg !7
  %1427 = select i1 %.not36, float %1420, float %1414, !dbg !7
  %1428 = bitcast float %1427 to i32, !dbg !7
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1428), !dbg !7
  %1430 = bitcast i32 %1429 to float, !dbg !7
  %1431 = select i1 %.not36, float %1414, float %1430, !dbg !7
  %1432 = select i1 %.not36, float %1430, float %1420, !dbg !7
  %1433 = extractelement <8 x float> %1324, i64 4, !dbg !7
  %1434 = extractelement <8 x float> %1324, i64 5, !dbg !7
  %1435 = extractelement <8 x float> %1324, i64 6, !dbg !7
  %1436 = extractelement <8 x float> %1324, i64 7, !dbg !7
  %1437 = select i1 %.not, float %1434, float %1433, !dbg !7
  %1438 = bitcast float %1437 to i32, !dbg !7
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1438), !dbg !7
  %1440 = bitcast i32 %1439 to float, !dbg !7
  %1441 = select i1 %.not, float %1433, float %1440, !dbg !7
  %1442 = select i1 %.not, float %1440, float %1434, !dbg !7
  %1443 = select i1 %.not, float %1436, float %1435, !dbg !7
  %1444 = bitcast float %1443 to i32, !dbg !7
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1444), !dbg !7
  %1446 = bitcast i32 %1445 to float, !dbg !7
  %1447 = select i1 %.not, float %1435, float %1446, !dbg !7
  %1448 = select i1 %.not, float %1446, float %1436, !dbg !7
  %1449 = select i1 %.not36, float %1447, float %1441, !dbg !7
  %1450 = bitcast float %1449 to i32, !dbg !7
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1450), !dbg !7
  %1452 = bitcast i32 %1451 to float, !dbg !7
  %1453 = select i1 %.not36, float %1441, float %1452, !dbg !7
  %1454 = select i1 %.not36, float %1452, float %1447, !dbg !7
  %1455 = select i1 %.not36, float %1448, float %1442, !dbg !7
  %1456 = bitcast float %1455 to i32, !dbg !7
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1456), !dbg !7
  %1458 = bitcast i32 %1457 to float, !dbg !7
  %1459 = select i1 %.not36, float %1442, float %1458, !dbg !7
  %1460 = select i1 %.not36, float %1458, float %1448, !dbg !7
  %1461 = extractelement <8 x float> %1353, i64 4, !dbg !7
  %1462 = extractelement <8 x float> %1353, i64 5, !dbg !7
  %1463 = extractelement <8 x float> %1353, i64 6, !dbg !7
  %1464 = extractelement <8 x float> %1353, i64 7, !dbg !7
  %1465 = select i1 %.not, float %1462, float %1461, !dbg !7
  %1466 = bitcast float %1465 to i32, !dbg !7
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1466), !dbg !7
  %1468 = bitcast i32 %1467 to float, !dbg !7
  %1469 = select i1 %.not, float %1461, float %1468, !dbg !7
  %1470 = select i1 %.not, float %1468, float %1462, !dbg !7
  %1471 = select i1 %.not, float %1464, float %1463, !dbg !7
  %1472 = bitcast float %1471 to i32, !dbg !7
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1472), !dbg !7
  %1474 = bitcast i32 %1473 to float, !dbg !7
  %1475 = select i1 %.not, float %1463, float %1474, !dbg !7
  %1476 = select i1 %.not, float %1474, float %1464, !dbg !7
  %1477 = select i1 %.not36, float %1475, float %1469, !dbg !7
  %1478 = bitcast float %1477 to i32, !dbg !7
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1478), !dbg !7
  %1480 = select i1 %.not36, float %1476, float %1470, !dbg !7
  %1481 = bitcast float %1480 to i32, !dbg !7
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1481), !dbg !7
  %1483 = bitcast i32 %1479 to float, !dbg !7
  %1484 = select i1 %.not36, float %1469, float %1483, !dbg !7
  %1485 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !16
  %1486 = bitcast i32 %1482 to float, !dbg !7
  %1487 = select i1 %.not36, float %1470, float %1486, !dbg !7
  %1488 = insertelement <4 x float> %1485, float %1487, i64 1, !dbg !16
  %1489 = select i1 %.not36, float %1483, float %1475, !dbg !7
  %1490 = insertelement <4 x float> %1488, float %1489, i64 2, !dbg !16
  %1491 = select i1 %.not36, float %1486, float %1476, !dbg !7
  %1492 = insertelement <4 x float> %1490, float %1491, i64 3, !dbg !16
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1376, !dbg !16
  %normalized = fmul <4 x float> %1492, %row.reciprocal, !dbg !16
  %1493 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !17
  %1494 = extractelement <4 x half> %1493, i64 3, !dbg !17
  %1495 = extractelement <4 x half> %1493, i64 2, !dbg !17
  %1496 = extractelement <4 x half> %1493, i64 1, !dbg !17
  %1497 = extractelement <4 x half> %1493, i64 0, !dbg !17
  %1498 = bitcast i32 %1372 to float, !dbg !7
  %1499 = select i1 %.not36, float %1362, float %1498, !dbg !7
  %1500 = insertelement <4 x float> poison, float %1499, i64 0, !dbg !16
  %1501 = bitcast i32 %1375 to float, !dbg !7
  %1502 = select i1 %.not36, float %1363, float %1501, !dbg !7
  %1503 = insertelement <4 x float> %1500, float %1502, i64 1, !dbg !16
  %1504 = select i1 %.not36, float %1498, float %1368, !dbg !7
  %1505 = insertelement <4 x float> %1503, float %1504, i64 2, !dbg !16
  %1506 = select i1 %.not36, float %1501, float %1369, !dbg !7
  %1507 = insertelement <4 x float> %1505, float %1506, i64 3, !dbg !16
  %normalized37 = fmul <4 x float> %1507, %row.reciprocal, !dbg !16
  %1508 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !17
  %1509 = extractelement <4 x half> %1508, i64 3, !dbg !17
  %1510 = extractelement <4 x half> %1508, i64 2, !dbg !17
  %1511 = extractelement <4 x half> %1508, i64 1, !dbg !17
  %1512 = extractelement <4 x half> %1508, i64 0, !dbg !17
  %1513 = bitcast i32 %1262 to float, !dbg !7
  %1514 = select i1 %.not36, float %1252, float %1513, !dbg !7
  %1515 = insertelement <4 x float> poison, float %1514, i64 0, !dbg !16
  %1516 = bitcast i32 %1265 to float, !dbg !7
  %1517 = select i1 %.not36, float %1253, float %1516, !dbg !7
  %1518 = insertelement <4 x float> %1515, float %1517, i64 1, !dbg !16
  %1519 = select i1 %.not36, float %1513, float %1258, !dbg !7
  %1520 = insertelement <4 x float> %1518, float %1519, i64 2, !dbg !16
  %1521 = select i1 %.not36, float %1516, float %1259, !dbg !7
  %1522 = insertelement <4 x float> %1520, float %1521, i64 3, !dbg !16
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1159, !dbg !16
  %normalized39 = fmul <4 x float> %1522, %row.reciprocal38, !dbg !16
  %1523 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !17
  %1524 = extractelement <4 x half> %1523, i64 3, !dbg !17
  %1525 = extractelement <4 x half> %1523, i64 2, !dbg !17
  %1526 = extractelement <4 x half> %1523, i64 1, !dbg !17
  %1527 = extractelement <4 x half> %1523, i64 0, !dbg !17
  %1528 = bitcast i32 %1155 to float, !dbg !7
  %1529 = select i1 %.not36, float %1145, float %1528, !dbg !7
  %1530 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !16
  %1531 = bitcast i32 %1158 to float, !dbg !7
  %1532 = select i1 %.not36, float %1146, float %1531, !dbg !7
  %1533 = insertelement <4 x float> %1530, float %1532, i64 1, !dbg !16
  %1534 = select i1 %.not36, float %1528, float %1151, !dbg !7
  %1535 = insertelement <4 x float> %1533, float %1534, i64 2, !dbg !16
  %1536 = select i1 %.not36, float %1531, float %1152, !dbg !7
  %1537 = insertelement <4 x float> %1535, float %1536, i64 3, !dbg !16
  %normalized40 = fmul <4 x float> %1537, %row.reciprocal38, !dbg !16
  %1538 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !17
  %1539 = extractelement <4 x half> %1538, i64 3, !dbg !17
  %1540 = extractelement <4 x half> %1538, i64 2, !dbg !17
  %1541 = extractelement <4 x half> %1538, i64 1, !dbg !17
  %1542 = extractelement <4 x half> %1538, i64 0, !dbg !17
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !18, !range !32, !frisk.register_lifetimes !1
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %address.local976 = add nuw i64 %19, %address.local975, !dbg !18
  %1543 = lshr i64 %address.local975, 2, !dbg !18
  %address.local978 = and i64 %1543, 48, !dbg !18
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !18
  %address.local980 = and i64 %address.local975, 240, !dbg !18
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !18
  %address.local982 = add i64 %address.local981, %20, !dbg !18
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !18
  %1544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %23, !dbg !18
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1544, i64 %address.local983, !dbg !18
  %address.local987 = and i64 %1543, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !18
  store half %1542, ptr addrspace(1) %address.local988, align 2, !dbg !18
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !18
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !18
  store half %1541, ptr addrspace(1) %address.local991, align 2, !dbg !18
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !18
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !18
  store half %1540, ptr addrspace(1) %address.local994, align 2, !dbg !18
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !18
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !18
  store half %1539, ptr addrspace(1) %address.local997, align 2, !dbg !18
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !18
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !18
  store half %1527, ptr addrspace(1) %address.local1000, align 2, !dbg !18
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !18
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !18
  store half %1526, ptr addrspace(1) %address.local1003, align 2, !dbg !18
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !18
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !18
  store half %1525, ptr addrspace(1) %address.local1006, align 2, !dbg !18
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !18
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !18
  store half %1524, ptr addrspace(1) %address.local1009, align 2, !dbg !18
  %address.local1011 = add nuw i64 %address.local983, 8192, !dbg !18
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1544, i64 %address.local1011, !dbg !18
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !18
  store half %1512, ptr addrspace(1) %address.local1014, align 2, !dbg !18
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !18
  store half %1511, ptr addrspace(1) %address.local1016, align 2, !dbg !18
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !18
  store half %1510, ptr addrspace(1) %address.local1018, align 2, !dbg !18
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !18
  store half %1509, ptr addrspace(1) %address.local1020, align 2, !dbg !18
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !18
  store half %1497, ptr addrspace(1) %address.local1022, align 2, !dbg !18
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !18
  store half %1496, ptr addrspace(1) %address.local1024, align 2, !dbg !18
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !18
  store half %1495, ptr addrspace(1) %address.local1026, align 2, !dbg !18
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !18
  store half %1494, ptr addrspace(1) %address.local1028, align 2, !dbg !18
  %1545 = insertelement <4 x float> poison, float %1453, i64 0, !dbg !16
  %1546 = insertelement <4 x float> %1545, float %1459, i64 1, !dbg !16
  %1547 = insertelement <4 x float> %1546, float %1454, i64 2, !dbg !16
  %1548 = insertelement <4 x float> %1547, float %1460, i64 3, !dbg !16
  %normalized42 = fmul <4 x float> %1548, %row.reciprocal, !dbg !16
  %1549 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !17
  %1550 = extractelement <4 x half> %1549, i64 3, !dbg !17
  %1551 = extractelement <4 x half> %1549, i64 2, !dbg !17
  %1552 = extractelement <4 x half> %1549, i64 1, !dbg !17
  %1553 = extractelement <4 x half> %1549, i64 0, !dbg !17
  %1554 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !16
  %1555 = insertelement <4 x float> %1554, float %1351, i64 1, !dbg !16
  %1556 = insertelement <4 x float> %1555, float %1346, i64 2, !dbg !16
  %1557 = insertelement <4 x float> %1556, float %1352, i64 3, !dbg !16
  %normalized43 = fmul <4 x float> %1557, %row.reciprocal, !dbg !16
  %1558 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !17
  %1559 = extractelement <4 x half> %1558, i64 3, !dbg !17
  %1560 = extractelement <4 x half> %1558, i64 2, !dbg !17
  %1561 = extractelement <4 x half> %1558, i64 1, !dbg !17
  %1562 = extractelement <4 x half> %1558, i64 0, !dbg !17
  %1563 = insertelement <4 x float> poison, float %1236, i64 0, !dbg !16
  %1564 = insertelement <4 x float> %1563, float %1242, i64 1, !dbg !16
  %1565 = insertelement <4 x float> %1564, float %1237, i64 2, !dbg !16
  %1566 = insertelement <4 x float> %1565, float %1243, i64 3, !dbg !16
  %normalized45 = fmul <4 x float> %1566, %row.reciprocal38, !dbg !16
  %1567 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !17
  %1568 = extractelement <4 x half> %1567, i64 3, !dbg !17
  %1569 = extractelement <4 x half> %1567, i64 2, !dbg !17
  %1570 = extractelement <4 x half> %1567, i64 1, !dbg !17
  %1571 = extractelement <4 x half> %1567, i64 0, !dbg !17
  %1572 = insertelement <4 x float> poison, float %1128, i64 0, !dbg !16
  %1573 = insertelement <4 x float> %1572, float %1134, i64 1, !dbg !16
  %1574 = insertelement <4 x float> %1573, float %1129, i64 2, !dbg !16
  %1575 = insertelement <4 x float> %1574, float %1135, i64 3, !dbg !16
  %normalized46 = fmul <4 x float> %1575, %row.reciprocal38, !dbg !16
  %1576 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !17
  %1577 = extractelement <4 x half> %1576, i64 3, !dbg !17
  %1578 = extractelement <4 x half> %1576, i64 2, !dbg !17
  %1579 = extractelement <4 x half> %1576, i64 1, !dbg !17
  %1580 = extractelement <4 x half> %1576, i64 0, !dbg !17
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !18
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !18
  store half %1580, ptr addrspace(1) %address.local1044, align 2, !dbg !18
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !18
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !18
  store half %1579, ptr addrspace(1) %address.local1047, align 2, !dbg !18
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !18
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !18
  store half %1578, ptr addrspace(1) %address.local1050, align 2, !dbg !18
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !18
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !18
  store half %1577, ptr addrspace(1) %address.local1053, align 2, !dbg !18
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !18
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !18
  store half %1571, ptr addrspace(1) %address.local1056, align 2, !dbg !18
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !18
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !18
  store half %1570, ptr addrspace(1) %address.local1059, align 2, !dbg !18
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !18
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !18
  store half %1569, ptr addrspace(1) %address.local1062, align 2, !dbg !18
  %address.local1064 = or i64 %1543, 51, !dbg !18
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !18
  store half %1568, ptr addrspace(1) %address.local1065, align 2, !dbg !18
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !18
  store half %1562, ptr addrspace(1) %address.local1070, align 2, !dbg !18
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !18
  store half %1561, ptr addrspace(1) %address.local1072, align 2, !dbg !18
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !18
  store half %1560, ptr addrspace(1) %address.local1074, align 2, !dbg !18
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !18
  store half %1559, ptr addrspace(1) %address.local1076, align 2, !dbg !18
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !18
  store half %1553, ptr addrspace(1) %address.local1078, align 2, !dbg !18
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !18
  store half %1552, ptr addrspace(1) %address.local1080, align 2, !dbg !18
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !18
  store half %1551, ptr addrspace(1) %address.local1082, align 2, !dbg !18
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !18
  store half %1550, ptr addrspace(1) %address.local1084, align 2, !dbg !18
  %1581 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !16
  %1582 = insertelement <4 x float> %1581, float %1431, i64 1, !dbg !16
  %1583 = insertelement <4 x float> %1582, float %1426, i64 2, !dbg !16
  %1584 = insertelement <4 x float> %1583, float %1432, i64 3, !dbg !16
  %normalized48 = fmul <4 x float> %1584, %row.reciprocal, !dbg !16
  %1585 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !17
  %1586 = extractelement <4 x half> %1585, i64 3, !dbg !17
  %1587 = extractelement <4 x half> %1585, i64 2, !dbg !17
  %1588 = extractelement <4 x half> %1585, i64 1, !dbg !17
  %1589 = extractelement <4 x half> %1585, i64 0, !dbg !17
  %1590 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !16
  %1591 = insertelement <4 x float> %1590, float %1322, i64 1, !dbg !16
  %1592 = insertelement <4 x float> %1591, float %1317, i64 2, !dbg !16
  %1593 = insertelement <4 x float> %1592, float %1323, i64 3, !dbg !16
  %normalized49 = fmul <4 x float> %1593, %row.reciprocal, !dbg !16
  %1594 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !17
  %1595 = extractelement <4 x half> %1594, i64 3, !dbg !17
  %1596 = extractelement <4 x half> %1594, i64 2, !dbg !17
  %1597 = extractelement <4 x half> %1594, i64 1, !dbg !17
  %1598 = extractelement <4 x half> %1594, i64 0, !dbg !17
  %1599 = insertelement <4 x float> poison, float %1208, i64 0, !dbg !16
  %1600 = insertelement <4 x float> %1599, float %1214, i64 1, !dbg !16
  %1601 = insertelement <4 x float> %1600, float %1209, i64 2, !dbg !16
  %1602 = insertelement <4 x float> %1601, float %1215, i64 3, !dbg !16
  %normalized51 = fmul <4 x float> %1602, %row.reciprocal38, !dbg !16
  %1603 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !17
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !17
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !17
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !17
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !17
  %1608 = insertelement <4 x float> poison, float %1099, i64 0, !dbg !16
  %1609 = insertelement <4 x float> %1608, float %1105, i64 1, !dbg !16
  %1610 = insertelement <4 x float> %1609, float %1100, i64 2, !dbg !16
  %1611 = insertelement <4 x float> %1610, float %1106, i64 3, !dbg !16
  %normalized52 = fmul <4 x float> %1611, %row.reciprocal38, !dbg !16
  %1612 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !17
  %1613 = extractelement <4 x half> %1612, i64 3, !dbg !17
  %1614 = extractelement <4 x half> %1612, i64 2, !dbg !17
  %1615 = extractelement <4 x half> %1612, i64 1, !dbg !17
  %1616 = extractelement <4 x half> %1612, i64 0, !dbg !17
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !18
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !18
  store half %1616, ptr addrspace(1) %address.local1100, align 2, !dbg !18
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !18
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !18
  store half %1615, ptr addrspace(1) %address.local1103, align 2, !dbg !18
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !18
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !18
  store half %1614, ptr addrspace(1) %address.local1106, align 2, !dbg !18
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !18
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !18
  store half %1613, ptr addrspace(1) %address.local1109, align 2, !dbg !18
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !18
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !18
  store half %1607, ptr addrspace(1) %address.local1112, align 2, !dbg !18
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !18
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !18
  store half %1606, ptr addrspace(1) %address.local1115, align 2, !dbg !18
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !18
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !18
  store half %1605, ptr addrspace(1) %address.local1118, align 2, !dbg !18
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !18
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !18
  store half %1604, ptr addrspace(1) %address.local1121, align 2, !dbg !18
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !18
  store half %1598, ptr addrspace(1) %address.local1126, align 2, !dbg !18
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !18
  store half %1597, ptr addrspace(1) %address.local1128, align 2, !dbg !18
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !18
  store half %1596, ptr addrspace(1) %address.local1130, align 2, !dbg !18
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !18
  store half %1595, ptr addrspace(1) %address.local1132, align 2, !dbg !18
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !18
  store half %1589, ptr addrspace(1) %address.local1134, align 2, !dbg !18
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !18
  store half %1588, ptr addrspace(1) %address.local1136, align 2, !dbg !18
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !18
  store half %1587, ptr addrspace(1) %address.local1138, align 2, !dbg !18
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !18
  store half %1586, ptr addrspace(1) %address.local1140, align 2, !dbg !18
  %1617 = insertelement <4 x float> poison, float %1397, i64 0, !dbg !16
  %1618 = insertelement <4 x float> %1617, float %1403, i64 1, !dbg !16
  %1619 = insertelement <4 x float> %1618, float %1398, i64 2, !dbg !16
  %1620 = insertelement <4 x float> %1619, float %1404, i64 3, !dbg !16
  %normalized54 = fmul <4 x float> %1620, %row.reciprocal, !dbg !16
  %1621 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !17
  %1622 = extractelement <4 x half> %1621, i64 3, !dbg !17
  %1623 = extractelement <4 x half> %1621, i64 2, !dbg !17
  %1624 = extractelement <4 x half> %1621, i64 1, !dbg !17
  %1625 = extractelement <4 x half> %1621, i64 0, !dbg !17
  %1626 = insertelement <4 x float> poison, float %1287, i64 0, !dbg !16
  %1627 = insertelement <4 x float> %1626, float %1293, i64 1, !dbg !16
  %1628 = insertelement <4 x float> %1627, float %1288, i64 2, !dbg !16
  %1629 = insertelement <4 x float> %1628, float %1294, i64 3, !dbg !16
  %normalized55 = fmul <4 x float> %1629, %row.reciprocal, !dbg !16
  %1630 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !17
  %1631 = extractelement <4 x half> %1630, i64 3, !dbg !17
  %1632 = extractelement <4 x half> %1630, i64 2, !dbg !17
  %1633 = extractelement <4 x half> %1630, i64 1, !dbg !17
  %1634 = extractelement <4 x half> %1630, i64 0, !dbg !17
  %1635 = insertelement <4 x float> poison, float %1180, i64 0, !dbg !16
  %1636 = insertelement <4 x float> %1635, float %1186, i64 1, !dbg !16
  %1637 = insertelement <4 x float> %1636, float %1181, i64 2, !dbg !16
  %1638 = insertelement <4 x float> %1637, float %1187, i64 3, !dbg !16
  %normalized57 = fmul <4 x float> %1638, %row.reciprocal38, !dbg !16
  %1639 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !17
  %1640 = extractelement <4 x half> %1639, i64 3, !dbg !17
  %1641 = extractelement <4 x half> %1639, i64 2, !dbg !17
  %1642 = extractelement <4 x half> %1639, i64 1, !dbg !17
  %1643 = extractelement <4 x half> %1639, i64 0, !dbg !17
  %1644 = insertelement <4 x float> poison, float %1070, i64 0, !dbg !16
  %1645 = insertelement <4 x float> %1644, float %1076, i64 1, !dbg !16
  %1646 = insertelement <4 x float> %1645, float %1071, i64 2, !dbg !16
  %1647 = insertelement <4 x float> %1646, float %1077, i64 3, !dbg !16
  %normalized58 = fmul <4 x float> %1647, %row.reciprocal38, !dbg !16
  %1648 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !17
  %1649 = extractelement <4 x half> %1648, i64 3, !dbg !17
  %1650 = extractelement <4 x half> %1648, i64 2, !dbg !17
  %1651 = extractelement <4 x half> %1648, i64 1, !dbg !17
  %1652 = extractelement <4 x half> %1648, i64 0, !dbg !17
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !18
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !18
  store half %1652, ptr addrspace(1) %address.local1156, align 2, !dbg !18
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !18
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !18
  store half %1651, ptr addrspace(1) %address.local1159, align 2, !dbg !18
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !18
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !18
  store half %1650, ptr addrspace(1) %address.local1162, align 2, !dbg !18
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !18
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !18
  store half %1649, ptr addrspace(1) %address.local1165, align 2, !dbg !18
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !18
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !18
  store half %1643, ptr addrspace(1) %address.local1168, align 2, !dbg !18
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !18
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !18
  store half %1642, ptr addrspace(1) %address.local1171, align 2, !dbg !18
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !18
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !18
  store half %1641, ptr addrspace(1) %address.local1174, align 2, !dbg !18
  %address.local1176 = or i64 %1543, 115, !dbg !18
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !18
  store half %1640, ptr addrspace(1) %address.local1177, align 2, !dbg !18
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !18
  store half %1634, ptr addrspace(1) %address.local1182, align 2, !dbg !18
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !18
  store half %1633, ptr addrspace(1) %address.local1184, align 2, !dbg !18
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !18
  store half %1632, ptr addrspace(1) %address.local1186, align 2, !dbg !18
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !18
  store half %1631, ptr addrspace(1) %address.local1188, align 2, !dbg !18
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !18
  store half %1625, ptr addrspace(1) %address.local1190, align 2, !dbg !18
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !18
  store half %1624, ptr addrspace(1) %address.local1192, align 2, !dbg !18
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !18
  store half %1623, ptr addrspace(1) %address.local1194, align 2, !dbg !18
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !18
  store half %1622, ptr addrspace(1) %address.local1196, align 2, !dbg !18
  ret void, !dbg !5
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { nocallback nofree nounwind willreturn }
attributes #5 = { convergent nocallback nofree nounwind willreturn }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "baseline.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_softmax_fragments")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!7 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 67, column: 22, scope: !4)
!10 = !DILocation(line: 69, column: 24, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 87, column: 23, scope: !4)
!14 = !DILocation(line: 70, column: 26, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 92, column: 23, scope: !4)
!17 = !DILocation(line: 93, column: 19, scope: !4)
!18 = !DILocation(line: 94, column: 9, scope: !4)
!19 = !DILocation(line: 71, column: 23, scope: !4)
!20 = !DILocation(line: 74, column: 20, scope: !4)
!21 = !DILocation(line: 75, column: 22, scope: !4)
!22 = !DILocation(line: 76, column: 23, scope: !4)
!23 = !DILocation(line: 77, column: 20, scope: !4)
!24 = !DILocation(line: 78, column: 22, scope: !4)
!25 = !DILocation(line: 79, column: 22, scope: !4)
!26 = !DILocation(line: 80, column: 20, scope: !4)
!27 = !DILocation(line: 81, column: 16, scope: !4)
!28 = !DILocation(line: 88, column: 22, scope: !4)
!30 = !{i32 256, i32 1, i32 1}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 0, i32 256}
!33 = !{i32 4}
!34 = !{i64 8192}
!35 = !{i32 0}
!36 = !{i32 1}
!37 = !{i32 2}
!38 = !{i32 3}
!39 = !{i32 5}
!40 = !{i32 6}
!41 = !{i32 7}
!42 = !{i32 8}
!43 = !{i32 9}
!44 = !{i32 10}
!45 = !{i32 11}
!46 = !{i32 12}
!47 = !{i32 13}
!48 = !{i32 14}
!49 = !{i32 15}
!50 = !{i32 16}
!51 = !{i32 17}
!52 = !{i32 18}
!53 = !{i32 19}
!54 = !{i32 20}
!55 = !{i32 21}
!56 = !{i32 22}
