; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest_k_swizzle64.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !33 !reqd_work_group_size !55 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !34
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !34
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !34
  %5 = and i32 %kdma.lane, 31, !dbg !34
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !34
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !34
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !34
  %kdma.intra = and i32 %6, 8, !dbg !34
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !34
  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !34
  %8 = sext i32 %7 to i64, !dbg !34
  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !34
  %10 = sext i32 %9 to i64, !dbg !34
  %11 = shl nsw i64 %10, 5, !dbg !34
  %12 = add nsw i64 %11, %8, !dbg !34
  %13 = lshr i64 %12, 5, !dbg !34
  %14 = and i64 %13, 576460752303423456, !dbg !34
  %15 = and i64 %8, 31, !dbg !34
  %16 = or disjoint i64 %14, %15, !dbg !34
  %17 = lshr i64 %12, 10, !dbg !36
  %18 = add nuw nsw i64 %13, %17, !dbg !36
  %19 = shl nsw i64 %16, 7, !dbg !36
  %20 = mul nsw i64 %17, -4096, !dbg !36
  %21 = add i64 %19, %20, !dbg !36
  %22 = shl i64 %18, 19, !dbg !36
  %23 = and i64 %22, 16252928, !dbg !36
  %24 = shl nsw i64 %21, 7, !dbg !36
  %25 = add i64 %23, %24, !dbg !36
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !36, !range !57, !frisk.register_lifetimes !30
  %address.local = zext nneg i32 %address.thread to i64, !dbg !36
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !36
  %address.local405 = and i64 %address.local404, 1536, !dbg !36
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !36
  %address.local407 = ptrtoint ptr addrspace(3) %address.local406 to i64, !dbg !36
  %address.local408 = trunc i64 %address.local407 to i32, !dbg !36
  %26 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local408), !dbg !36
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !36
  %address.local410 = and i64 %address.local409, 16256, !dbg !36
  %address.local411 = lshr i64 %address.local, 4, !dbg !36
  %address.local412 = and i64 %address.local411, 8, !dbg !36
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !36
  %27 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !36
  %.idx1399 = or disjoint i64 %25, %address.local413, !dbg !36
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !36
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !36
  %28 = insertelement <2 x i64> poison, i64 %27, i64 0, !dbg !36
  %29 = bitcast <2 x i64> %28 to <4 x i32>, !dbg !36
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %29, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !36
  %dma.buffer.lds1398 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %26), !dbg !36
  %30 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !36
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !36
  %address.local417 = and i64 %address.local416, 3584, !dbg !36
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !36
  %address.local419 = ptrtoint ptr addrspace(3) %address.local418 to i64, !dbg !36
  %address.local420 = trunc i64 %address.local419 to i32, !dbg !36
  %31 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local420), !dbg !36
  %address.local423 = lshr i64 %address.local415, 4, !dbg !36
  %address.local424 = and i64 %address.local423, 24, !dbg !36
  %address.local425 = or disjoint i64 %address.local410, %address.local424, !dbg !36
  %.idx1400 = or disjoint i64 %25, %address.local425, !dbg !36
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !36
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !36
  %dma.buffer.lds1395 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %31), !dbg !36
  %32 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !36
  %address.local428 = ptrtoint ptr addrspace(3) %address.local427 to i64, !dbg !36
  %address.local429 = trunc i64 %address.local428 to i32, !dbg !36
  %33 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local429), !dbg !36
  %34 = or disjoint i64 %25, 32, !dbg !36
  %35 = or disjoint i64 %address.local413, %34, !dbg !36
  %.tr = trunc i64 %35 to i32, !dbg !36
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !36
  %dma.buffer.lds1392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %33), !dbg !36
  %36 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !36
  %address.local432 = ptrtoint ptr addrspace(3) %address.local431 to i64, !dbg !36
  %address.local433 = trunc i64 %address.local432 to i32, !dbg !36
  %37 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local433), !dbg !36
  %38 = or disjoint i64 %address.local425, %34, !dbg !36
  %.tr1401 = trunc i64 %38 to i32, !dbg !36
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !36
  %dma.buffer.lds1389 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !36
  %39 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %address.local435 = lshr i32 %address.thread, 2, !dbg !37
  %address.local436 = and i32 %address.local435, 4, !dbg !37
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !37
  %address.local438 = and i32 %address.local437, 384, !dbg !37
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !37
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !37
  %address.local441 = and i32 %address.local440, 120, !dbg !37
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !37
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !37
  %address.local444 = and i32 %address.local443, 1024, !dbg !37
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !37
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !37
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !37
  %40 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !37
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !37
  %41 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !37
  %42 = fmul <4 x half> %40, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !37
  %43 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !37
  %44 = fmul <4 x half> %41, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !37
  %45 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !37
  %46 = fmul <4 x half> %43, splat (half 1.275630e-01), !dbg !37
  %47 = fmul <4 x half> %45, splat (half 1.275630e-01), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %48 = or disjoint i64 %25, 64, !dbg !36
  %49 = or disjoint i64 %address.local413, %48, !dbg !36
  %.tr1402 = trunc i64 %49 to i32, !dbg !36
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !36
  %50 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %51 = or disjoint i64 %address.local425, %48, !dbg !36
  %.tr1403 = trunc i64 %51 to i32, !dbg !36
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !36
  %52 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !37
  %53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !37
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !37
  %54 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !37
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !37
  %55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !37
  %56 = fmul <4 x half> %53, splat (half 1.275630e-01), !dbg !37
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !37
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !37
  %58 = fmul <4 x half> %54, splat (half 1.275630e-01), !dbg !37
  %59 = fmul <4 x half> %55, splat (half 1.275630e-01), !dbg !37
  %60 = fmul <4 x half> %57, splat (half 1.275630e-01), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %61 = or disjoint i64 %25, 96, !dbg !36
  %62 = or disjoint i64 %address.local413, %61, !dbg !36
  %.tr1404 = trunc i64 %62 to i32, !dbg !36
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !36
  %63 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %64 = or disjoint i64 %address.local425, %61, !dbg !36
  %.tr1405 = trunc i64 %64 to i32, !dbg !36
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !36
  %65 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #10, !dbg !36
  %66 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !37
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !37
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !37
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !37
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !37
  %71 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !37
  %72 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !37
  %73 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !37
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !36
  fence syncscope("workgroup") release, !dbg !36
  call void @llvm.amdgcn.s.barrier(), !dbg !36
  fence syncscope("workgroup") acquire, !dbg !36
  %74 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !37
  %75 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !37
  %76 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !37
  %77 = fmul <4 x half> %74, splat (half 1.275630e-01), !dbg !37
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !37
  %79 = fmul <4 x half> %75, splat (half 1.275630e-01), !dbg !37
  %80 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !37
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !37
  fence syncscope("workgroup") release, !dbg !35
  call void @llvm.amdgcn.s.barrier(), !dbg !35
  fence syncscope("workgroup") acquire, !dbg !35
  %82 = shl nuw nsw i32 %kdma.tid, 2, !dbg !40
  %83 = and i32 %82, 3084, !dbg !40
  %84 = and i32 %kdma.tid, 1, !dbg !40
  %85 = or disjoint i32 %83, %84, !dbg !40
  %narrow = add nuw nsw i32 %85, 1, !dbg !40
  %86 = lshr i32 %narrow, 1, !dbg !40
  %87 = zext nneg i32 %86 to i64, !dbg !40
  br label %88, !dbg !35

88:                                               ; preds = %97, %4
  %89 = phi i64 [ %1022, %97 ], [ 0, %4 ], !dbg !35
  %90 = phi [2 x <8 x float>] [ %1007, %97 ], [ zeroinitializer, %4 ], !dbg !35
  %91 = phi [2 x <1 x float>] [ %701, %97 ], [ zeroinitializer, %4 ], !dbg !35
  %92 = phi [2 x <1 x float>] [ %1009, %97 ], [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], !dbg !35
  %93 = phi [2 x <8 x float>] [ %1013, %97 ], [ zeroinitializer, %4 ], !dbg !35
  %94 = phi [2 x <8 x float>] [ %1017, %97 ], [ zeroinitializer, %4 ], !dbg !35
  %95 = phi [2 x <8 x float>] [ %1021, %97 ], [ zeroinitializer, %4 ], !dbg !35
  %96 = icmp samesign ult i64 %89, 32, !dbg !35
  br i1 %96, label %97, label %1023, !dbg !35

97:                                               ; preds = %88
  %98 = shl i64 %13, 19, !dbg !38
  %99 = and i64 %98, 16252928, !dbg !38
  %100 = shl nuw nsw i64 %89, 14, !dbg !38
  %101 = or disjoint i64 %99, %100, !dbg !38
  %address.thread498 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !38, !range !57, !frisk.register_lifetimes !30
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !36
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !36
  %address.local501 = and i64 %address.local500, 1536, !dbg !36
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !36
  %address.local503 = ptrtoint ptr addrspace(3) %address.local502 to i64, !dbg !36
  %address.local504 = trunc i64 %address.local503 to i32, !dbg !36
  %102 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local504), !dbg !38
  %103 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !38
  %104 = insertelement <2 x i64> poison, i64 %103, i64 0, !dbg !38
  %105 = bitcast <2 x i64> %104 to <4 x i32>, !dbg !38
  %dma.buffer.rsrc1373 = shufflevector <4 x i32> %105, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !38
  %dma.buffer.lds1374 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !38
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !38
  %kdma.source1506 = lshr i64 %kdma.source, 4, !dbg !38
  %kdma.source1507 = and i64 %kdma.source1506, 8, !dbg !38
  %kdma.source1508 = and i64 %kdma.source, 91, !dbg !38
  %kdma.source1509 = lshr i64 %kdma.source, 2, !dbg !38
  %kdma.source1510 = and i64 %kdma.source1509, 4, !dbg !38
  %kdma.source1511 = add nuw nsw i64 %kdma.source1510, %kdma.source, !dbg !38
  %kdma.source1512 = and i64 %kdma.source1511, 4, !dbg !38
  %kdma.source1513 = or disjoint i64 %kdma.source1508, %kdma.source1512, !dbg !38
  %kdma.source1514 = shl nuw nsw i64 %kdma.source1513, 7, !dbg !38
  %kdma.source1515 = or disjoint i64 %kdma.source1507, %kdma.source1514, !dbg !38
  %kdma.source1516 = or disjoint i64 %101, %kdma.source1515, !dbg !38
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !38
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !38
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !38
  %106 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !38
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !38
  %kdma.source1521 = and i64 %kdma.source1520, 8, !dbg !38
  %kdma.source1522 = and i64 %kdma.source1519, 123, !dbg !38
  %kdma.source1523 = lshr i64 %kdma.source1519, 2, !dbg !38
  %kdma.source1524 = and i64 %kdma.source1523, 4, !dbg !38
  %kdma.source1525 = add nuw nsw i64 %kdma.source1524, %kdma.source1519, !dbg !38
  %kdma.source1526 = and i64 %kdma.source1525, 4, !dbg !38
  %kdma.source1527 = or disjoint i64 %kdma.source1522, %kdma.source1526, !dbg !38
  %kdma.source1528 = shl nuw nsw i64 %kdma.source1527, 7, !dbg !38
  %kdma.source1529 = or disjoint i64 %kdma.source1521, %kdma.source1528, !dbg !38
  %kdma.source1530 = or disjoint i64 %101, %kdma.source1529, !dbg !38
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !38
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !38
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra.hi, !dbg !38
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !38
  %107 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !36
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !36
  %address.local518 = and i64 %address.local517, 3584, !dbg !36
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !36
  %address.local520 = ptrtoint ptr addrspace(3) %address.local519 to i64, !dbg !36
  %address.local521 = trunc i64 %address.local520 to i32, !dbg !36
  %108 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local521), !dbg !38
  %dma.buffer.lds1371 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !38
  %kdma.source1546 = or disjoint i64 %kdma.source1515, 16, !dbg !38
  %kdma.source1547 = or disjoint i64 %101, %kdma.source1546, !dbg !38
  %kdma.source1548 = trunc nuw i64 %kdma.source1547 to i32, !dbg !38
  %kdma.source1549 = shl nuw nsw i32 %kdma.source1548, 1, !dbg !38
  %kdma.offset1550 = or disjoint i32 %kdma.source1549, %kdma.intra, !dbg !38
  %109 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1550, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1563 = or disjoint i64 %kdma.source1529, 16, !dbg !38
  %kdma.source1564 = or disjoint i64 %101, %kdma.source1563, !dbg !38
  %kdma.source1565 = trunc nuw i64 %kdma.source1564 to i32, !dbg !38
  %kdma.source1566 = shl nuw nsw i32 %kdma.source1565, 1, !dbg !38
  %kdma.offset1567 = or disjoint i32 %kdma.source1566, %kdma.intra.hi, !dbg !38
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !38
  %110 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1567, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %111 = or disjoint i64 %101, 32, !dbg !38
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !36
  %address.local534 = ptrtoint ptr addrspace(3) %address.local533 to i64, !dbg !36
  %address.local535 = trunc i64 %address.local534 to i32, !dbg !36
  %112 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local535), !dbg !38
  %dma.buffer.lds1368 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %112), !dbg !38
  %kdma.source1580 = or disjoint i64 %111, %kdma.source1515, !dbg !38
  %kdma.source1581 = trunc nuw i64 %kdma.source1580 to i32, !dbg !38
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !38
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !38
  %113 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1596 = or disjoint i64 %111, %kdma.source1529, !dbg !38
  %kdma.source1597 = trunc nuw i64 %kdma.source1596 to i32, !dbg !38
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !38
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !38
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !38
  %114 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !36
  %address.local538 = ptrtoint ptr addrspace(3) %address.local537 to i64, !dbg !36
  %address.local539 = trunc i64 %address.local538 to i32, !dbg !36
  %115 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local539), !dbg !38
  %dma.buffer.lds1365 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115), !dbg !38
  %kdma.source1613 = or disjoint i64 %111, %kdma.source1546, !dbg !38
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !38
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !38
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !38
  %116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1630 = or disjoint i64 %111, %kdma.source1563, !dbg !38
  %kdma.source1631 = trunc nuw i64 %kdma.source1630 to i32, !dbg !38
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !38
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !38
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !38
  %117 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %address.local542 = lshr i64 %address.local499, 2, !dbg !40
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !40
  %address.local546 = and i64 %address.local545, 24, !dbg !40
  %address.local548 = shl i64 %address.local499, 62, !dbg !40
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !40
  %address.local550 = and i64 %address.local549, -64, !dbg !40
  %address.local553 = and i64 %address.local542, 12, !dbg !37
  %address.local555 = and i64 %address.local499, 63, !dbg !37
  %address.local556 = lshr i64 %address.local555, 5, !dbg !37
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !37
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !40
  %118 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !40
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !40
  %119 = getelementptr i8, ptr addrspace(3) %118, i64 %.idx1410, !dbg !40
  %120 = getelementptr [2 x i8], ptr addrspace(3) %119, i64 %address.local546, !dbg !40
  %121 = getelementptr [2 x i8], ptr addrspace(3) %120, i64 %address.local550, !dbg !40
  %122 = getelementptr [2 x i8], ptr addrspace(3) %121, i64 %address.local553, !dbg !40
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %122, i64 %address.local557, !dbg !40
  %123 = add nsw i64 %.idx, %.idx1410, !dbg !40
  %.idx1 = shl nuw nsw i64 %address.local546, 1, !dbg !40
  %124 = or disjoint i64 %123, %.idx1, !dbg !40
  %125 = or disjoint i64 %address.local550, %address.local553, !dbg !40
  %126 = shl nsw i64 %125, 1, !dbg !40
  %127 = add nsw i64 %126, %124, !dbg !40
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !40
  %128 = add nsw i64 %127, %address.local559.idx, !dbg !40
  %129 = lshr i64 %128, 6, !dbg !40
  %130 = and i64 %129, 8, !dbg !40
  %131 = xor i64 %128, %130, !dbg !40
  %132 = sub i64 %131, %128, !dbg !40
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %132, !dbg !40
  %133 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !40
  %134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !40
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !40
  %135 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !40
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !40
  %address.local571 = and i32 %address.local570, 24, !dbg !40
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !40
  %address.local573 = and i32 %address.local572, 768, !dbg !40
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !40
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !40
  %address.local576 = and i32 %address.local575, 96, !dbg !40
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !40
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !40
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %87) #9, !dbg !40, !frisk.register_lifetimes !30
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !40
  %address.local580 = and i32 %address.local579, 1, !dbg !40
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !40
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !40
  %address.local583 = shl i64 %address.local582, 6, !dbg !40
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !40
  %136 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !40
  %137 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local553, !dbg !40
  %138 = getelementptr [2 x i8], ptr addrspace(3) %137, i64 %address.local557, !dbg !40
  %address.local588 = getelementptr i8, ptr addrspace(3) %138, i64 64, !dbg !40
  %.idx47 = add i64 %address.local584, %address.local553, !dbg !40
  %139 = shl i64 %.idx47, 1, !dbg !40
  %140 = add i64 %139, %address.local559.idx, !dbg !40
  %141 = add i64 %140, 64, !dbg !40
  %142 = lshr i64 %141, 6, !dbg !40
  %143 = and i64 %142, 8, !dbg !40
  %144 = xor i64 %141, %143, !dbg !40
  %145 = sub i64 %144, %141, !dbg !40
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %145, !dbg !40
  %146 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !40, !frisk.k_lds_pair !60
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !40
  %147 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !40, !frisk.k_lds_pair !60
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !40
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !40, !frisk.k_lds_pair !61
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !40
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !40, !frisk.k_lds_pair !61
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !40
  %150 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !40, !frisk.k_lds_pair !62
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !40
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !40, !frisk.k_lds_pair !62
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !40
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !40, !frisk.k_lds_pair !63
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !40
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !40, !frisk.k_lds_pair !63
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %135, <4 x float> %134) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !40
  %156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %148, <4 x float> %155) #9, !dbg !40
  %157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !40
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %152, <4 x float> %157) #9, !dbg !40
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !40
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !40, !frisk.k_lds_pair !58
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !40
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !40, !frisk.k_lds_pair !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !40
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !40, !frisk.k_lds_pair !64
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !40
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !40, !frisk.k_lds_pair !64
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %149, <4 x float> %161) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !40
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !40, !frisk.k_lds_pair !65
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !40
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !40, !frisk.k_lds_pair !65
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %153, <4 x float> %167) #9, !dbg !40
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !40
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %162, <4 x float> %169) #9, !dbg !40
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !40
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %166, <4 x float> %171) #9, !dbg !40
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %42, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !40
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %44, <4 x half> %163, <4 x float> %173) #9, !dbg !40
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %133, <4 x float> zeroinitializer) #9, !dbg !40
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %135, <4 x float> %175) #9, !dbg !40
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %146, <4 x float> zeroinitializer) #9, !dbg !40
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %148, <4 x float> %177) #9, !dbg !40
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !40
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %152, <4 x float> %179) #9, !dbg !40
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %147, <4 x float> zeroinitializer) #9, !dbg !40
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %149, <4 x float> %181) #9, !dbg !40
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !40
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %153, <4 x float> %183) #9, !dbg !40
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !40
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %162, <4 x float> %185) #9, !dbg !40
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !40
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %166, <4 x float> %187) #9, !dbg !40
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %46, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !40
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %47, <4 x half> %163, <4 x float> %189) #9, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %191 = or disjoint i64 %101, 64, !dbg !38
  %kdma.source1646 = or disjoint i64 %191, %kdma.source1515, !dbg !38
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !38
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !38
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !38
  %192 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1662 = or disjoint i64 %191, %kdma.source1529, !dbg !38
  %kdma.source1663 = trunc nuw i64 %kdma.source1662 to i32, !dbg !38
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !38
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !38
  %193 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1679 = or disjoint i64 %191, %kdma.source1546, !dbg !38
  %kdma.source1680 = trunc nuw i64 %kdma.source1679 to i32, !dbg !38
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !38
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !38
  %194 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1696 = or disjoint i64 %191, %kdma.source1563, !dbg !38
  %kdma.source1697 = trunc nuw i64 %kdma.source1696 to i32, !dbg !38
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !38
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !38
  %195 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !40
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !40, !frisk.k_lds_pair !66
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !40
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !40, !frisk.k_lds_pair !66
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !40
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !40, !frisk.k_lds_pair !67
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !40
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !40, !frisk.k_lds_pair !67
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !40
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !40, !frisk.k_lds_pair !68
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !40
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !40, !frisk.k_lds_pair !68
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !40
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !40, !frisk.k_lds_pair !69
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !40
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !40, !frisk.k_lds_pair !69
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %204 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %196, <4 x float> %154) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %198, <4 x float> %204) #9, !dbg !40
  %206 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %200, <4 x float> %156) #9, !dbg !40
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %202, <4 x float> %206) #9, !dbg !40
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !40
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !40, !frisk.k_lds_pair !70
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !40
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !40, !frisk.k_lds_pair !70
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %197, <4 x float> %158) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !40
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !40, !frisk.k_lds_pair !71
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !40
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !40, !frisk.k_lds_pair !71
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %199, <4 x float> %210) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !40
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !40, !frisk.k_lds_pair !72
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !40
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !40, !frisk.k_lds_pair !72
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %201, <4 x float> %164) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !40
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !40, !frisk.k_lds_pair !73
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !40
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !40, !frisk.k_lds_pair !73
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %203, <4 x float> %216) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %208, <4 x float> %168) #9, !dbg !40
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %211, <4 x float> %220) #9, !dbg !40
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %214, <4 x float> %170) #9, !dbg !40
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %217, <4 x float> %222) #9, !dbg !40
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %209, <4 x float> %172) #9, !dbg !40
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %212, <4 x float> %224) #9, !dbg !40
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %215, <4 x float> %174) #9, !dbg !40
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %218, <4 x float> %226) #9, !dbg !40
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %196, <4 x float> %176) #9, !dbg !40
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %198, <4 x float> %228) #9, !dbg !40
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %200, <4 x float> %178) #9, !dbg !40
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %202, <4 x float> %230) #9, !dbg !40
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %197, <4 x float> %180) #9, !dbg !40
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %199, <4 x float> %232) #9, !dbg !40
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %201, <4 x float> %182) #9, !dbg !40
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %203, <4 x float> %234) #9, !dbg !40
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %208, <4 x float> %184) #9, !dbg !40
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %211, <4 x float> %236) #9, !dbg !40
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %214, <4 x float> %186) #9, !dbg !40
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %217, <4 x float> %238) #9, !dbg !40
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %209, <4 x float> %188) #9, !dbg !40
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %212, <4 x float> %240) #9, !dbg !40
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %59, <4 x half> %215, <4 x float> %190) #9, !dbg !40
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %218, <4 x float> %242) #9, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %244 = or disjoint i64 %101, 96, !dbg !38
  %kdma.source1712 = or disjoint i64 %244, %kdma.source1515, !dbg !38
  %kdma.source1713 = trunc nuw i64 %kdma.source1712 to i32, !dbg !38
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !38
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !38
  %245 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1728 = or disjoint i64 %244, %kdma.source1529, !dbg !38
  %kdma.source1729 = trunc nuw i64 %kdma.source1728 to i32, !dbg !38
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !38
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !38
  %246 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1745 = or disjoint i64 %244, %kdma.source1546, !dbg !38
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !38
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !38
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !38
  %247 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %kdma.source1762 = or disjoint i64 %244, %kdma.source1563, !dbg !38
  %kdma.source1763 = trunc nuw i64 %kdma.source1762 to i32, !dbg !38
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !38
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !38
  %248 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %dma.buffer.rsrc1373) #10, !dbg !38
  %249 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !40, !frisk.k_lds_pair !74
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !40, !frisk.k_lds_pair !74
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !40, !frisk.k_lds_pair !75
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !40, !frisk.k_lds_pair !75
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !40, !frisk.k_lds_pair !76
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !40, !frisk.k_lds_pair !76
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !40, !frisk.k_lds_pair !77
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !40, !frisk.k_lds_pair !77
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %249, <4 x float> %205) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %251, <4 x float> %257) #9, !dbg !40
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %253, <4 x float> %207) #9, !dbg !40
  %260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %255, <4 x float> %259) #9, !dbg !40
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !40, !frisk.k_lds_pair !78
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !40, !frisk.k_lds_pair !78
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %250, <4 x float> %213) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !40, !frisk.k_lds_pair !79
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !40, !frisk.k_lds_pair !79
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %252, <4 x float> %263) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !40, !frisk.k_lds_pair !80
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !40, !frisk.k_lds_pair !80
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %254, <4 x float> %219) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !40, !frisk.k_lds_pair !81
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !40, !frisk.k_lds_pair !81
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %256, <4 x float> %269) #9, !dbg !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !40, !frisk.k_lds_prefetch !30
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %261, <4 x float> %221) #9, !dbg !40
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %264, <4 x float> %273) #9, !dbg !40
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %267, <4 x float> %223) #9, !dbg !40
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %270, <4 x float> %275) #9, !dbg !40
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %262, <4 x float> %225) #9, !dbg !40
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %265, <4 x float> %277) #9, !dbg !40
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %268, <4 x float> %227) #9, !dbg !40
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %271, <4 x float> %279) #9, !dbg !40
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %249, <4 x float> %229) #9, !dbg !40
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %251, <4 x float> %281) #9, !dbg !40
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %253, <4 x float> %231) #9, !dbg !40
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %255, <4 x float> %283) #9, !dbg !40
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %250, <4 x float> %233) #9, !dbg !40
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %252, <4 x float> %285) #9, !dbg !40
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %254, <4 x float> %235) #9, !dbg !40
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %256, <4 x float> %287) #9, !dbg !40
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %261, <4 x float> %237) #9, !dbg !40
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %264, <4 x float> %289) #9, !dbg !40
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %267, <4 x float> %239) #9, !dbg !40
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %270, <4 x float> %291) #9, !dbg !40
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %262, <4 x float> %241) #9, !dbg !40
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %265, <4 x float> %293) #9, !dbg !40
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %268, <4 x float> %243) #9, !dbg !40
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %271, <4 x float> %295) #9, !dbg !40
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !41
  %address.local918 = and i64 %address.local917, 8064, !dbg !41
  %address.local920 = and i64 %address.local500, 24, !dbg !41
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !41
  %297 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !41
  %.idx14191420 = or disjoint i64 %101, %address.local921, !dbg !41
  %.idx14191420.tr = trunc nuw i64 %.idx14191420 to i32, !dbg !41
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !41
  %298 = insertelement <2 x i64> poison, i64 %297, i64 0, !dbg !41
  %299 = bitcast <2 x i64> %298 to <4 x i32>, !dbg !41
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %299, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !41
  %300 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !41
  %address.local924 = and i64 %address.local923, 16256, !dbg !41
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !41
  %.idx14211422 = or disjoint i64 %101, %address.local927, !dbg !41
  %.idx14211422.tr = trunc nuw i64 %.idx14211422 to i32, !dbg !41
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !41
  %301 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !40
  %303 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %302, <4 x float> %258) #9, !dbg !40
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !40
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !40
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %304, <4 x float> %303) #9, !dbg !40
  %307 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %305, <4 x float> %260) #9, !dbg !40
  %308 = extractelement <4 x float> %306, i64 0, !dbg !45
  %309 = call float @llvm.maxnum.f32(float %308, float -inf), !dbg !46, !frisk.local_reduction !30
  %310 = extractelement <4 x float> %306, i64 1, !dbg !45
  %311 = extractelement <4 x float> %306, i64 2, !dbg !45
  %312 = extractelement <4 x float> %306, i64 3, !dbg !45
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !40
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !40
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %313, <4 x float> %307) #9, !dbg !40
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %314, <4 x float> %266) #9, !dbg !40
  %317 = extractelement <4 x float> %315, i64 0, !dbg !45
  %local.reduce151 = call float @llvm.maxnum.f32(float %309, float %317), !dbg !46, !frisk.local_reduction !30
  %318 = extractelement <4 x float> %315, i64 1, !dbg !45
  %local.reduce152 = call float @llvm.maxnum.f32(float %310, float %318), !dbg !46, !frisk.local_reduction !30
  %319 = extractelement <4 x float> %315, i64 2, !dbg !45
  %local.reduce153 = call float @llvm.maxnum.f32(float %311, float %319), !dbg !46, !frisk.local_reduction !30
  %320 = extractelement <4 x float> %315, i64 3, !dbg !45
  %local.reduce154 = call float @llvm.maxnum.f32(float %312, float %320), !dbg !46, !frisk.local_reduction !30
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !40
  %322 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !40
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %321, <4 x float> %316) #9, !dbg !40
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %322, <4 x float> %272) #9, !dbg !40
  %325 = extractelement <4 x float> %323, i64 0, !dbg !45
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %325), !dbg !46, !frisk.local_reduction !30
  %326 = extractelement <4 x float> %323, i64 1, !dbg !45
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %326), !dbg !46, !frisk.local_reduction !30
  %327 = extractelement <4 x float> %323, i64 2, !dbg !45
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %327), !dbg !46, !frisk.local_reduction !30
  %328 = extractelement <4 x float> %323, i64 3, !dbg !45
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %328), !dbg !46, !frisk.local_reduction !30
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !40
  %330 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !40
  %331 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %329, <4 x float> %324) #9, !dbg !40
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %330, <4 x float> %274) #9, !dbg !40
  %333 = extractelement <4 x float> %331, i64 0, !dbg !45
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %333), !dbg !46, !frisk.local_reduction !30
  %334 = extractelement <4 x float> %331, i64 1, !dbg !45
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %334), !dbg !46, !frisk.local_reduction !30
  %335 = extractelement <4 x float> %331, i64 2, !dbg !45
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %335), !dbg !46, !frisk.local_reduction !30
  %336 = extractelement <4 x float> %331, i64 3, !dbg !45
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %336), !dbg !46, !frisk.local_reduction !30
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !40
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !40
  %339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %337, <4 x float> %332) #9, !dbg !40
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %338, <4 x float> %276) #9, !dbg !40
  %341 = extractelement <4 x float> %339, i64 0, !dbg !45
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %341), !dbg !46, !frisk.local_reduction !30
  %342 = extractelement <4 x float> %339, i64 1, !dbg !45
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %342), !dbg !46, !frisk.local_reduction !30
  %343 = extractelement <4 x float> %339, i64 2, !dbg !45
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %343), !dbg !46, !frisk.local_reduction !30
  %344 = extractelement <4 x float> %339, i64 3, !dbg !45
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %344), !dbg !46, !frisk.local_reduction !30
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !40
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !40
  %347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %345, <4 x float> %340) #9, !dbg !40
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %346, <4 x float> %278) #9, !dbg !40
  %349 = extractelement <4 x float> %347, i64 0, !dbg !45
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %349), !dbg !46, !frisk.local_reduction !30
  %350 = extractelement <4 x float> %347, i64 1, !dbg !45
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %350), !dbg !46, !frisk.local_reduction !30
  %351 = extractelement <4 x float> %347, i64 2, !dbg !45
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %351), !dbg !46, !frisk.local_reduction !30
  %352 = extractelement <4 x float> %347, i64 3, !dbg !45
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %352), !dbg !46, !frisk.local_reduction !30
  %353 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !40
  %354 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !40
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %353, <4 x float> %348) #9, !dbg !40
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %77, <4 x half> %354, <4 x float> %280) #9, !dbg !40
  %357 = extractelement <4 x float> %355, i64 0, !dbg !45
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %357), !dbg !46, !frisk.local_reduction !30
  %358 = extractelement <4 x float> %355, i64 1, !dbg !45
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %358), !dbg !46, !frisk.local_reduction !30
  %359 = extractelement <4 x float> %355, i64 2, !dbg !45
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %359), !dbg !46, !frisk.local_reduction !30
  %360 = extractelement <4 x float> %355, i64 3, !dbg !45
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %360), !dbg !46, !frisk.local_reduction !30
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !40
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %361, <4 x float> %356) #9, !dbg !40
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %302, <4 x float> %282) #9, !dbg !40
  %364 = extractelement <4 x float> %362, i64 0, !dbg !45
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %364), !dbg !46, !frisk.local_reduction !30
  %365 = extractelement <4 x float> %362, i64 1, !dbg !45
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %365), !dbg !46, !frisk.local_reduction !30
  %366 = extractelement <4 x float> %362, i64 2, !dbg !45
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %366), !dbg !46, !frisk.local_reduction !30
  %367 = extractelement <4 x float> %362, i64 3, !dbg !45
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %367), !dbg !46, !frisk.local_reduction !30
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !46, !frisk.local_reduction !30
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !46, !frisk.local_reduction !30
  %368 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !46, !frisk.local_reduction !30
  %369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %304, <4 x float> %363) #9, !dbg !40
  %370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %305, <4 x float> %284) #9, !dbg !40
  %371 = extractelement <4 x float> %369, i64 0, !dbg !45
  %372 = extractelement <4 x float> %369, i64 1, !dbg !45
  %373 = extractelement <4 x float> %369, i64 2, !dbg !45
  %374 = extractelement <4 x float> %369, i64 3, !dbg !45
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %313, <4 x float> %370) #9, !dbg !40
  %376 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %314, <4 x float> %286) #9, !dbg !40
  %377 = extractelement <4 x float> %375, i64 0, !dbg !45
  %378 = extractelement <4 x float> %375, i64 1, !dbg !45
  %379 = extractelement <4 x float> %375, i64 2, !dbg !45
  %380 = extractelement <4 x float> %375, i64 3, !dbg !45
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %321, <4 x float> %376) #9, !dbg !40
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %322, <4 x float> %288) #9, !dbg !40
  %383 = extractelement <4 x float> %381, i64 0, !dbg !45
  %384 = extractelement <4 x float> %381, i64 1, !dbg !45
  %385 = extractelement <4 x float> %381, i64 2, !dbg !45
  %386 = extractelement <4 x float> %381, i64 3, !dbg !45
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %329, <4 x float> %382) #9, !dbg !40
  %388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %330, <4 x float> %290) #9, !dbg !40
  %389 = extractelement <4 x float> %387, i64 0, !dbg !45
  %390 = extractelement <4 x float> %387, i64 1, !dbg !45
  %391 = extractelement <4 x float> %387, i64 2, !dbg !45
  %392 = extractelement <4 x float> %387, i64 3, !dbg !45
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %337, <4 x float> %388) #9, !dbg !40
  %394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %338, <4 x float> %292) #9, !dbg !40
  %395 = extractelement <4 x float> %393, i64 0, !dbg !45
  %396 = extractelement <4 x float> %393, i64 1, !dbg !45
  %397 = extractelement <4 x float> %393, i64 2, !dbg !45
  %398 = extractelement <4 x float> %393, i64 3, !dbg !45
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %345, <4 x float> %394) #9, !dbg !40
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %346, <4 x float> %294) #9, !dbg !40
  %401 = extractelement <4 x float> %399, i64 0, !dbg !45
  %402 = extractelement <4 x float> %399, i64 1, !dbg !45
  %403 = extractelement <4 x float> %399, i64 2, !dbg !45
  %404 = extractelement <4 x float> %399, i64 3, !dbg !45
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %353, <4 x float> %400) #9, !dbg !40
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %80, <4 x half> %354, <4 x float> %296) #9, !dbg !40
  %407 = extractelement <4 x float> %405, i64 0, !dbg !45
  %408 = extractelement <4 x float> %405, i64 1, !dbg !45
  %409 = extractelement <4 x float> %405, i64 2, !dbg !45
  %410 = extractelement <4 x float> %405, i64 3, !dbg !45
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %361, <4 x float> %406) #9, !dbg !40
  %412 = extractelement <4 x float> %411, i64 0, !dbg !45
  %413 = extractelement <4 x float> %411, i64 1, !dbg !45
  %414 = extractelement <4 x float> %411, i64 2, !dbg !45
  %415 = extractelement <4 x float> %411, i64 3, !dbg !45
  %416 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !46
  %417 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %416), !dbg !46
  %418 = add nuw nsw i32 %417, 64, !dbg !46
  %419 = xor i32 %417, 16, !dbg !46
  %420 = and i32 %418, 192, !dbg !46
  %421 = icmp samesign ult i32 %419, %420, !dbg !46
  %422 = select i1 %421, i32 %419, i32 %417, !dbg !46
  %423 = shl nuw nsw i32 %422, 2, !dbg !46
  %424 = bitcast float %368 to i32, !dbg !46
  %425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %424), !dbg !46
  %426 = bitcast i32 %425 to float, !dbg !46
  %427 = call float @llvm.maxnum.f32(float %368, float %426), !dbg !46
  %428 = xor i32 %417, 32, !dbg !46
  %429 = icmp samesign ult i32 %428, %420, !dbg !46
  %430 = select i1 %429, i32 %428, i32 %417, !dbg !46
  %431 = shl nuw nsw i32 %430, 2, !dbg !46
  %432 = bitcast float %427 to i32, !dbg !46
  %433 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %432), !dbg !46
  %434 = bitcast i32 %433 to float, !dbg !46
  %435 = call float @llvm.maxnum.f32(float %427, float %434), !dbg !46
  %436 = extractvalue [2 x <1 x float>] %92, 0, !dbg !46
  %437 = insertelement <1 x float> poison, float %435, i64 0, !dbg !46
  %438 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %437, <1 x float> %436), !dbg !46
  %439 = call float @llvm.maxnum.f32(float %371, float -inf), !dbg !46, !frisk.local_reduction !30
  %local.reduce120 = call float @llvm.maxnum.f32(float %439, float %377), !dbg !46, !frisk.local_reduction !30
  %local.reduce121 = call float @llvm.maxnum.f32(float %372, float %378), !dbg !46, !frisk.local_reduction !30
  %local.reduce122 = call float @llvm.maxnum.f32(float %373, float %379), !dbg !46, !frisk.local_reduction !30
  %local.reduce123 = call float @llvm.maxnum.f32(float %374, float %380), !dbg !46, !frisk.local_reduction !30
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %383), !dbg !46, !frisk.local_reduction !30
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %384), !dbg !46, !frisk.local_reduction !30
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %385), !dbg !46, !frisk.local_reduction !30
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %386), !dbg !46, !frisk.local_reduction !30
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %389), !dbg !46, !frisk.local_reduction !30
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %390), !dbg !46, !frisk.local_reduction !30
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %391), !dbg !46, !frisk.local_reduction !30
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %392), !dbg !46, !frisk.local_reduction !30
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %395), !dbg !46, !frisk.local_reduction !30
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %396), !dbg !46, !frisk.local_reduction !30
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %397), !dbg !46, !frisk.local_reduction !30
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %398), !dbg !46, !frisk.local_reduction !30
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %401), !dbg !46, !frisk.local_reduction !30
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %402), !dbg !46, !frisk.local_reduction !30
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %403), !dbg !46, !frisk.local_reduction !30
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %404), !dbg !46, !frisk.local_reduction !30
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %407), !dbg !46, !frisk.local_reduction !30
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %408), !dbg !46, !frisk.local_reduction !30
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %409), !dbg !46, !frisk.local_reduction !30
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %410), !dbg !46, !frisk.local_reduction !30
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %412), !dbg !46, !frisk.local_reduction !30
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %413), !dbg !46, !frisk.local_reduction !30
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %414), !dbg !46, !frisk.local_reduction !30
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %415), !dbg !46, !frisk.local_reduction !30
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !46, !frisk.local_reduction !30
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !46, !frisk.local_reduction !30
  %440 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !46, !frisk.local_reduction !30
  %441 = bitcast float %440 to i32, !dbg !46
  %442 = call i32 @llvm.amdgcn.ds.bpermute(i32 %423, i32 %441), !dbg !46
  %443 = bitcast i32 %442 to float, !dbg !46
  %444 = call float @llvm.maxnum.f32(float %440, float %443), !dbg !46
  %445 = bitcast float %444 to i32, !dbg !46
  %446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %431, i32 %445), !dbg !46
  %447 = bitcast i32 %446 to float, !dbg !46
  %448 = call float @llvm.maxnum.f32(float %444, float %447), !dbg !46
  %449 = extractvalue [2 x <1 x float>] %92, 1, !dbg !46
  %450 = insertelement <1 x float> poison, float %448, i64 0, !dbg !46
  %451 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %450, <1 x float> %449), !dbg !46
  %452 = shufflevector <1 x float> %438, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %453 = fsub <4 x float> %306, %452, !dbg !47
  %454 = fsub <4 x float> %315, %452, !dbg !47
  %455 = fsub <4 x float> %323, %452, !dbg !47
  %456 = fsub <4 x float> %331, %452, !dbg !47
  %457 = fsub <4 x float> %339, %452, !dbg !47
  %458 = fsub <4 x float> %347, %452, !dbg !47
  %459 = fsub <4 x float> %355, %452, !dbg !47
  %460 = fsub <4 x float> %362, %452, !dbg !47
  %461 = shufflevector <1 x float> %451, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %462 = fsub <4 x float> %369, %461, !dbg !47
  %463 = fsub <4 x float> %375, %461, !dbg !47
  %464 = fsub <4 x float> %381, %461, !dbg !47
  %465 = fsub <4 x float> %387, %461, !dbg !47
  %466 = fsub <4 x float> %393, %461, !dbg !47
  %467 = fsub <4 x float> %399, %461, !dbg !47
  %468 = fsub <4 x float> %405, %461, !dbg !47
  %469 = fsub <4 x float> %411, %461, !dbg !47
  %470 = fsub <1 x float> %436, %438, !dbg !48
  %471 = extractelement <1 x float> %470, i64 0, !dbg !49
  %472 = call float @llvm.amdgcn.exp2.f32(float %471), !dbg !49
  %473 = insertelement <4 x float> poison, float %472, i64 0, !dbg !54
  %474 = shufflevector <4 x float> %473, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %475 = extractvalue [2 x <1 x float>] %91, 0, !dbg !50
  %476 = extractelement <1 x float> %475, i64 0, !dbg !50
  %477 = fmul float %476, %472, !dbg !50
  %478 = fsub <1 x float> %449, %451, !dbg !48
  %479 = extractelement <1 x float> %478, i64 0, !dbg !49
  %480 = call float @llvm.amdgcn.exp2.f32(float %479), !dbg !49
  %481 = insertelement <4 x float> poison, float %480, i64 0, !dbg !54
  %482 = shufflevector <4 x float> %481, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !54
  %483 = extractvalue [2 x <1 x float>] %91, 1, !dbg !50
  %484 = extractelement <1 x float> %483, i64 0, !dbg !50
  %485 = fmul float %484, %480, !dbg !50
  %486 = extractelement <4 x float> %453, i64 0, !dbg !51
  %487 = call float @llvm.amdgcn.exp2.f32(float %486), !dbg !51
  %488 = insertelement <4 x float> poison, float %487, i64 0, !dbg !53
  %489 = extractelement <4 x float> %453, i64 1, !dbg !51
  %490 = call float @llvm.amdgcn.exp2.f32(float %489), !dbg !51
  %491 = insertelement <4 x float> %488, float %490, i64 1, !dbg !53
  %492 = extractelement <4 x float> %453, i64 2, !dbg !51
  %493 = call float @llvm.amdgcn.exp2.f32(float %492), !dbg !51
  %494 = insertelement <4 x float> %491, float %493, i64 2, !dbg !53
  %495 = extractelement <4 x float> %453, i64 3, !dbg !51
  %496 = call float @llvm.amdgcn.exp2.f32(float %495), !dbg !51
  %497 = insertelement <4 x float> %494, float %496, i64 3, !dbg !53
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %497) #9, !dbg !53, !frisk.register_lifetimes !30
  %498 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !53
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %498) #9, !dbg !51, !frisk.register_lifetimes !30
  %499 = extractelement <4 x float> %454, i64 0, !dbg !51
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !51
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !53
  %local.reduce89 = fadd float %487, %500, !dbg !52, !frisk.local_reduction !30
  %502 = extractelement <4 x float> %454, i64 1, !dbg !51
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !51
  %504 = insertelement <4 x float> %501, float %503, i64 1, !dbg !53
  %local.reduce90 = fadd float %490, %503, !dbg !52, !frisk.local_reduction !30
  %505 = extractelement <4 x float> %454, i64 2, !dbg !51
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !51
  %507 = insertelement <4 x float> %504, float %506, i64 2, !dbg !53
  %local.reduce91 = fadd float %493, %506, !dbg !52, !frisk.local_reduction !30
  %508 = extractelement <4 x float> %454, i64 3, !dbg !51
  %509 = call float @llvm.amdgcn.exp2.f32(float %508), !dbg !51
  %510 = insertelement <4 x float> %507, float %509, i64 3, !dbg !53
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %510) #9, !dbg !53, !frisk.register_lifetimes !30
  %511 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !53
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %511) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce92 = fadd float %496, %509, !dbg !52, !frisk.local_reduction !30
  %512 = extractelement <4 x float> %455, i64 0, !dbg !51
  %513 = call float @llvm.amdgcn.exp2.f32(float %512), !dbg !51
  %514 = insertelement <4 x float> poison, float %513, i64 0, !dbg !53
  %local.reduce93 = fadd float %local.reduce89, %513, !dbg !52, !frisk.local_reduction !30
  %515 = extractelement <4 x float> %455, i64 1, !dbg !51
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !51
  %517 = insertelement <4 x float> %514, float %516, i64 1, !dbg !53
  %local.reduce94 = fadd float %local.reduce90, %516, !dbg !52, !frisk.local_reduction !30
  %518 = extractelement <4 x float> %455, i64 2, !dbg !51
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !51
  %520 = insertelement <4 x float> %517, float %519, i64 2, !dbg !53
  %local.reduce95 = fadd float %local.reduce91, %519, !dbg !52, !frisk.local_reduction !30
  %521 = extractelement <4 x float> %455, i64 3, !dbg !51
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !51
  %523 = insertelement <4 x float> %520, float %522, i64 3, !dbg !53
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %523) #9, !dbg !53, !frisk.register_lifetimes !30
  %524 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !53
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %524) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce96 = fadd float %local.reduce92, %522, !dbg !52, !frisk.local_reduction !30
  %525 = extractelement <4 x float> %456, i64 0, !dbg !51
  %526 = call float @llvm.amdgcn.exp2.f32(float %525), !dbg !51
  %527 = insertelement <4 x float> poison, float %526, i64 0, !dbg !53
  %local.reduce97 = fadd float %local.reduce93, %526, !dbg !52, !frisk.local_reduction !30
  %528 = extractelement <4 x float> %456, i64 1, !dbg !51
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !51
  %530 = insertelement <4 x float> %527, float %529, i64 1, !dbg !53
  %local.reduce98 = fadd float %local.reduce94, %529, !dbg !52, !frisk.local_reduction !30
  %531 = extractelement <4 x float> %456, i64 2, !dbg !51
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !51
  %533 = insertelement <4 x float> %530, float %532, i64 2, !dbg !53
  %local.reduce99 = fadd float %local.reduce95, %532, !dbg !52, !frisk.local_reduction !30
  %534 = extractelement <4 x float> %456, i64 3, !dbg !51
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !51
  %536 = insertelement <4 x float> %533, float %535, i64 3, !dbg !53
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %536) #9, !dbg !53, !frisk.register_lifetimes !30
  %537 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !53
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %537) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce100 = fadd float %local.reduce96, %535, !dbg !52, !frisk.local_reduction !30
  %538 = extractelement <4 x float> %457, i64 0, !dbg !51
  %539 = call float @llvm.amdgcn.exp2.f32(float %538), !dbg !51
  %540 = insertelement <4 x float> poison, float %539, i64 0, !dbg !53
  %local.reduce101 = fadd float %local.reduce97, %539, !dbg !52, !frisk.local_reduction !30
  %541 = extractelement <4 x float> %457, i64 1, !dbg !51
  %542 = call float @llvm.amdgcn.exp2.f32(float %541), !dbg !51
  %543 = insertelement <4 x float> %540, float %542, i64 1, !dbg !53
  %local.reduce102 = fadd float %local.reduce98, %542, !dbg !52, !frisk.local_reduction !30
  %544 = extractelement <4 x float> %457, i64 2, !dbg !51
  %545 = call float @llvm.amdgcn.exp2.f32(float %544), !dbg !51
  %546 = insertelement <4 x float> %543, float %545, i64 2, !dbg !53
  %local.reduce103 = fadd float %local.reduce99, %545, !dbg !52, !frisk.local_reduction !30
  %547 = extractelement <4 x float> %457, i64 3, !dbg !51
  %548 = call float @llvm.amdgcn.exp2.f32(float %547), !dbg !51
  %549 = insertelement <4 x float> %546, float %548, i64 3, !dbg !53
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %549) #9, !dbg !53, !frisk.register_lifetimes !30
  %550 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !53
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %550) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce104 = fadd float %local.reduce100, %548, !dbg !52, !frisk.local_reduction !30
  %551 = extractelement <4 x float> %458, i64 0, !dbg !51
  %552 = call float @llvm.amdgcn.exp2.f32(float %551), !dbg !51
  %553 = insertelement <4 x float> poison, float %552, i64 0, !dbg !53
  %local.reduce105 = fadd float %local.reduce101, %552, !dbg !52, !frisk.local_reduction !30
  %554 = extractelement <4 x float> %458, i64 1, !dbg !51
  %555 = call float @llvm.amdgcn.exp2.f32(float %554), !dbg !51
  %556 = insertelement <4 x float> %553, float %555, i64 1, !dbg !53
  %local.reduce106 = fadd float %local.reduce102, %555, !dbg !52, !frisk.local_reduction !30
  %557 = extractelement <4 x float> %458, i64 2, !dbg !51
  %558 = call float @llvm.amdgcn.exp2.f32(float %557), !dbg !51
  %559 = insertelement <4 x float> %556, float %558, i64 2, !dbg !53
  %local.reduce107 = fadd float %local.reduce103, %558, !dbg !52, !frisk.local_reduction !30
  %560 = extractelement <4 x float> %458, i64 3, !dbg !51
  %561 = call float @llvm.amdgcn.exp2.f32(float %560), !dbg !51
  %562 = insertelement <4 x float> %559, float %561, i64 3, !dbg !53
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %562) #9, !dbg !53, !frisk.register_lifetimes !30
  %563 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !53
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %563) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce108 = fadd float %local.reduce104, %561, !dbg !52, !frisk.local_reduction !30
  %564 = extractelement <4 x float> %459, i64 0, !dbg !51
  %565 = call float @llvm.amdgcn.exp2.f32(float %564), !dbg !51
  %566 = insertelement <4 x float> poison, float %565, i64 0, !dbg !53
  %local.reduce109 = fadd float %local.reduce105, %565, !dbg !52, !frisk.local_reduction !30
  %567 = extractelement <4 x float> %459, i64 1, !dbg !51
  %568 = call float @llvm.amdgcn.exp2.f32(float %567), !dbg !51
  %569 = insertelement <4 x float> %566, float %568, i64 1, !dbg !53
  %local.reduce110 = fadd float %local.reduce106, %568, !dbg !52, !frisk.local_reduction !30
  %570 = extractelement <4 x float> %459, i64 2, !dbg !51
  %571 = call float @llvm.amdgcn.exp2.f32(float %570), !dbg !51
  %572 = insertelement <4 x float> %569, float %571, i64 2, !dbg !53
  %local.reduce111 = fadd float %local.reduce107, %571, !dbg !52, !frisk.local_reduction !30
  %573 = extractelement <4 x float> %459, i64 3, !dbg !51
  %574 = call float @llvm.amdgcn.exp2.f32(float %573), !dbg !51
  %575 = insertelement <4 x float> %572, float %574, i64 3, !dbg !53
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %575) #9, !dbg !53, !frisk.register_lifetimes !30
  %576 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !53
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %576) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce112 = fadd float %local.reduce108, %574, !dbg !52, !frisk.local_reduction !30
  %577 = extractelement <4 x float> %460, i64 0, !dbg !51
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !51
  %579 = insertelement <4 x float> poison, float %578, i64 0, !dbg !53
  %local.reduce113 = fadd float %local.reduce109, %578, !dbg !52, !frisk.local_reduction !30
  %580 = extractelement <4 x float> %460, i64 1, !dbg !51
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !51
  %582 = insertelement <4 x float> %579, float %581, i64 1, !dbg !53
  %local.reduce114 = fadd float %local.reduce110, %581, !dbg !52, !frisk.local_reduction !30
  %local.reduce117 = fadd float %local.reduce113, %local.reduce114, !dbg !52, !frisk.local_reduction !30
  %583 = extractelement <4 x float> %460, i64 2, !dbg !51
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !51
  %585 = insertelement <4 x float> %582, float %584, i64 2, !dbg !53
  %local.reduce115 = fadd float %local.reduce111, %584, !dbg !52, !frisk.local_reduction !30
  %586 = extractelement <4 x float> %460, i64 3, !dbg !51
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !51
  %588 = insertelement <4 x float> %585, float %587, i64 3, !dbg !53
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #9, !dbg !53, !frisk.register_lifetimes !30
  %589 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !53
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce116 = fadd float %local.reduce112, %587, !dbg !52, !frisk.local_reduction !30
  %local.reduce118 = fadd float %local.reduce115, %local.reduce116, !dbg !52, !frisk.local_reduction !30
  %590 = fadd float %local.reduce117, %local.reduce118, !dbg !52, !frisk.local_reduction !30
  %591 = fadd float %590, %477, !dbg !52
  %592 = insertelement <1 x float> poison, float %591, i64 0, !dbg !35
  %593 = insertvalue [2 x <1 x float>] poison, <1 x float> %592, 0, !dbg !35
  %594 = extractelement <4 x float> %462, i64 0, !dbg !51
  %595 = call float @llvm.amdgcn.exp2.f32(float %594), !dbg !51
  %596 = insertelement <4 x float> poison, float %595, i64 0, !dbg !53
  %597 = extractelement <4 x float> %462, i64 1, !dbg !51
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !51
  %599 = insertelement <4 x float> %596, float %598, i64 1, !dbg !53
  %600 = extractelement <4 x float> %462, i64 2, !dbg !51
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !51
  %602 = insertelement <4 x float> %599, float %601, i64 2, !dbg !53
  %603 = extractelement <4 x float> %462, i64 3, !dbg !51
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !51
  %605 = insertelement <4 x float> %602, float %604, i64 3, !dbg !53
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %605) #9, !dbg !53, !frisk.register_lifetimes !30
  %606 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !53
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %606) #9, !dbg !51, !frisk.register_lifetimes !30
  %607 = extractelement <4 x float> %463, i64 0, !dbg !51
  %608 = call float @llvm.amdgcn.exp2.f32(float %607), !dbg !51
  %609 = insertelement <4 x float> poison, float %608, i64 0, !dbg !53
  %local.reduce = fadd float %595, %608, !dbg !52, !frisk.local_reduction !30
  %610 = extractelement <4 x float> %463, i64 1, !dbg !51
  %611 = call float @llvm.amdgcn.exp2.f32(float %610), !dbg !51
  %612 = insertelement <4 x float> %609, float %611, i64 1, !dbg !53
  %local.reduce59 = fadd float %598, %611, !dbg !52, !frisk.local_reduction !30
  %613 = extractelement <4 x float> %463, i64 2, !dbg !51
  %614 = call float @llvm.amdgcn.exp2.f32(float %613), !dbg !51
  %615 = insertelement <4 x float> %612, float %614, i64 2, !dbg !53
  %local.reduce60 = fadd float %601, %614, !dbg !52, !frisk.local_reduction !30
  %616 = extractelement <4 x float> %463, i64 3, !dbg !51
  %617 = call float @llvm.amdgcn.exp2.f32(float %616), !dbg !51
  %618 = insertelement <4 x float> %615, float %617, i64 3, !dbg !53
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %618) #9, !dbg !53, !frisk.register_lifetimes !30
  %619 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !53
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %619) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce61 = fadd float %604, %617, !dbg !52, !frisk.local_reduction !30
  %620 = extractelement <4 x float> %464, i64 0, !dbg !51
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !51
  %622 = insertelement <4 x float> poison, float %621, i64 0, !dbg !53
  %local.reduce62 = fadd float %local.reduce, %621, !dbg !52, !frisk.local_reduction !30
  %623 = extractelement <4 x float> %464, i64 1, !dbg !51
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !51
  %625 = insertelement <4 x float> %622, float %624, i64 1, !dbg !53
  %local.reduce63 = fadd float %local.reduce59, %624, !dbg !52, !frisk.local_reduction !30
  %626 = extractelement <4 x float> %464, i64 2, !dbg !51
  %627 = call float @llvm.amdgcn.exp2.f32(float %626), !dbg !51
  %628 = insertelement <4 x float> %625, float %627, i64 2, !dbg !53
  %local.reduce64 = fadd float %local.reduce60, %627, !dbg !52, !frisk.local_reduction !30
  %629 = extractelement <4 x float> %464, i64 3, !dbg !51
  %630 = call float @llvm.amdgcn.exp2.f32(float %629), !dbg !51
  %631 = insertelement <4 x float> %628, float %630, i64 3, !dbg !53
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %631) #9, !dbg !53, !frisk.register_lifetimes !30
  %632 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !53
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %632) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce65 = fadd float %local.reduce61, %630, !dbg !52, !frisk.local_reduction !30
  %633 = extractelement <4 x float> %465, i64 0, !dbg !51
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !51
  %635 = insertelement <4 x float> poison, float %634, i64 0, !dbg !53
  %local.reduce66 = fadd float %local.reduce62, %634, !dbg !52, !frisk.local_reduction !30
  %636 = extractelement <4 x float> %465, i64 1, !dbg !51
  %637 = call float @llvm.amdgcn.exp2.f32(float %636), !dbg !51
  %638 = insertelement <4 x float> %635, float %637, i64 1, !dbg !53
  %local.reduce67 = fadd float %local.reduce63, %637, !dbg !52, !frisk.local_reduction !30
  %639 = extractelement <4 x float> %465, i64 2, !dbg !51
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !51
  %641 = insertelement <4 x float> %638, float %640, i64 2, !dbg !53
  %local.reduce68 = fadd float %local.reduce64, %640, !dbg !52, !frisk.local_reduction !30
  %642 = extractelement <4 x float> %465, i64 3, !dbg !51
  %643 = call float @llvm.amdgcn.exp2.f32(float %642), !dbg !51
  %644 = insertelement <4 x float> %641, float %643, i64 3, !dbg !53
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %644) #9, !dbg !53, !frisk.register_lifetimes !30
  %645 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !53
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %645) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce69 = fadd float %local.reduce65, %643, !dbg !52, !frisk.local_reduction !30
  %646 = extractelement <4 x float> %466, i64 0, !dbg !51
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !51
  %648 = insertelement <4 x float> poison, float %647, i64 0, !dbg !53
  %local.reduce70 = fadd float %local.reduce66, %647, !dbg !52, !frisk.local_reduction !30
  %649 = extractelement <4 x float> %466, i64 1, !dbg !51
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !51
  %651 = insertelement <4 x float> %648, float %650, i64 1, !dbg !53
  %local.reduce71 = fadd float %local.reduce67, %650, !dbg !52, !frisk.local_reduction !30
  %652 = extractelement <4 x float> %466, i64 2, !dbg !51
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !51
  %654 = insertelement <4 x float> %651, float %653, i64 2, !dbg !53
  %local.reduce72 = fadd float %local.reduce68, %653, !dbg !52, !frisk.local_reduction !30
  %655 = extractelement <4 x float> %466, i64 3, !dbg !51
  %656 = call float @llvm.amdgcn.exp2.f32(float %655), !dbg !51
  %657 = insertelement <4 x float> %654, float %656, i64 3, !dbg !53
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %657) #9, !dbg !53, !frisk.register_lifetimes !30
  %658 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !53
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %658) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce73 = fadd float %local.reduce69, %656, !dbg !52, !frisk.local_reduction !30
  %659 = extractelement <4 x float> %467, i64 0, !dbg !51
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !51
  %661 = insertelement <4 x float> poison, float %660, i64 0, !dbg !53
  %local.reduce74 = fadd float %local.reduce70, %660, !dbg !52, !frisk.local_reduction !30
  %662 = extractelement <4 x float> %467, i64 1, !dbg !51
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !51
  %664 = insertelement <4 x float> %661, float %663, i64 1, !dbg !53
  %local.reduce75 = fadd float %local.reduce71, %663, !dbg !52, !frisk.local_reduction !30
  %665 = extractelement <4 x float> %467, i64 2, !dbg !51
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !51
  %667 = insertelement <4 x float> %664, float %666, i64 2, !dbg !53
  %local.reduce76 = fadd float %local.reduce72, %666, !dbg !52, !frisk.local_reduction !30
  %668 = extractelement <4 x float> %467, i64 3, !dbg !51
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !51
  %670 = insertelement <4 x float> %667, float %669, i64 3, !dbg !53
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #9, !dbg !53, !frisk.register_lifetimes !30
  %671 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !53
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce77 = fadd float %local.reduce73, %669, !dbg !52, !frisk.local_reduction !30
  %672 = extractelement <4 x float> %468, i64 0, !dbg !51
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !51
  %674 = insertelement <4 x float> poison, float %673, i64 0, !dbg !53
  %local.reduce78 = fadd float %local.reduce74, %673, !dbg !52, !frisk.local_reduction !30
  %675 = extractelement <4 x float> %468, i64 1, !dbg !51
  %676 = call float @llvm.amdgcn.exp2.f32(float %675), !dbg !51
  %677 = insertelement <4 x float> %674, float %676, i64 1, !dbg !53
  %local.reduce79 = fadd float %local.reduce75, %676, !dbg !52, !frisk.local_reduction !30
  %678 = extractelement <4 x float> %468, i64 2, !dbg !51
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !51
  %680 = insertelement <4 x float> %677, float %679, i64 2, !dbg !53
  %local.reduce80 = fadd float %local.reduce76, %679, !dbg !52, !frisk.local_reduction !30
  %681 = extractelement <4 x float> %468, i64 3, !dbg !51
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !51
  %683 = insertelement <4 x float> %680, float %682, i64 3, !dbg !53
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %683) #9, !dbg !53, !frisk.register_lifetimes !30
  %684 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !53
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %684) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce81 = fadd float %local.reduce77, %682, !dbg !52, !frisk.local_reduction !30
  %685 = extractelement <4 x float> %469, i64 0, !dbg !51
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !51
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !53
  %local.reduce82 = fadd float %local.reduce78, %686, !dbg !52, !frisk.local_reduction !30
  %688 = extractelement <4 x float> %469, i64 1, !dbg !51
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !51
  %690 = insertelement <4 x float> %687, float %689, i64 1, !dbg !53
  %local.reduce83 = fadd float %local.reduce79, %689, !dbg !52, !frisk.local_reduction !30
  %local.reduce86 = fadd float %local.reduce82, %local.reduce83, !dbg !52, !frisk.local_reduction !30
  %691 = extractelement <4 x float> %469, i64 2, !dbg !51
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !51
  %693 = insertelement <4 x float> %690, float %692, i64 2, !dbg !53
  %local.reduce84 = fadd float %local.reduce80, %692, !dbg !52, !frisk.local_reduction !30
  %694 = extractelement <4 x float> %469, i64 3, !dbg !51
  %695 = call float @llvm.amdgcn.exp2.f32(float %694), !dbg !51
  %696 = insertelement <4 x float> %693, float %695, i64 3, !dbg !53
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %696) #9, !dbg !53, !frisk.register_lifetimes !30
  %697 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !53
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %697) #9, !dbg !52, !frisk.register_lifetimes !30
  %local.reduce85 = fadd float %local.reduce81, %695, !dbg !52, !frisk.local_reduction !30
  %local.reduce87 = fadd float %local.reduce84, %local.reduce85, !dbg !52, !frisk.local_reduction !30
  %698 = fadd float %local.reduce86, %local.reduce87, !dbg !52, !frisk.local_reduction !30
  %699 = fadd float %698, %485, !dbg !52
  %700 = insertelement <1 x float> poison, float %699, i64 0, !dbg !35
  %701 = insertvalue [2 x <1 x float>] %593, <1 x float> %700, 1, !dbg !35
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1436 = or disjoint i64 %111, %address.local921, !dbg !41
  %vdb.address1437 = trunc nuw i64 %vdb.address1436 to i32, !dbg !41
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !41
  %vdb.address1445 = or disjoint i64 %111, %address.local927, !dbg !41
  %vdb.address1446 = trunc nuw i64 %vdb.address1445 to i32, !dbg !41
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !41
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !39
  %702 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !39
  %703 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !39
  %address.local930 = and i32 %address.local929, 120, !dbg !39
  %address.local932 = and i32 %address.local575, 1536, !dbg !39
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !39
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !39
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !39
  %704 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %705 = bitcast <4 x i32> %704 to <8 x half>, !dbg !39
  %address.local936 = and i64 %address.local499, 60, !dbg !39
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !39
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !39
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !39
  %address.local940 = lshr i64 %address.local555, 4, !dbg !39
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !39
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !39
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !39
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !39
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !39
  %706 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %707 = bitcast <4 x i32> %706 to <8 x half>, !dbg !39
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !39
  %708 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %709 = bitcast <4 x i32> %708 to <8 x half>, !dbg !39
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !39
  %710 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %711 = bitcast <4 x i32> %710 to <8 x half>, !dbg !39
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !39
  %712 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %713 = bitcast <4 x i32> %712 to <8 x half>, !dbg !39
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !39
  %714 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %715 = bitcast <4 x i32> %714 to <8 x half>, !dbg !39
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !39
  %716 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %717 = bitcast <4 x i32> %716 to <8 x half>, !dbg !39
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !39
  %718 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %719 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %720 = extractvalue [2 x <8 x float>] %90, 0, !dbg !54
  %721 = shufflevector <8 x float> %720, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %722 = fmul <4 x float> %721, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %723 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %719, <4 x float> %722) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %724 = shufflevector <8 x half> %705, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %725 = shufflevector <8 x float> %720, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %726 = fmul <4 x float> %725, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %724, <4 x float> %726) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %728 = extractvalue [2 x <8 x float>] %90, 1, !dbg !54
  %729 = shufflevector <8 x float> %728, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %730 = fmul <4 x float> %729, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %731 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %719, <4 x float> %730) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %732 = shufflevector <8 x float> %728, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %733 = fmul <4 x float> %732, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %734 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %724, <4 x float> %733) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %735 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %735, <4 x float> %723) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %737 = shufflevector <8 x half> %707, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %737, <4 x float> %727) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %735, <4 x float> %731) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %740 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %737, <4 x float> %734) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %741 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %742 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %741, <4 x float> %736) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %743 = shufflevector <8 x half> %709, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %744 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %743, <4 x float> %738) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %745 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %741, <4 x float> %739) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %746 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %743, <4 x float> %740) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %747 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %747, <4 x float> %742) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %749 = shufflevector <8 x half> %711, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %749, <4 x float> %744) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %751 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %747, <4 x float> %745) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %752 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %749, <4 x float> %746) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %753 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %753, <4 x float> %748) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %755 = shufflevector <8 x half> %713, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %755, <4 x float> %750) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %753, <4 x float> %751) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %755, <4 x float> %752) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %759 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %760 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %759, <4 x float> %754) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %761 = shufflevector <8 x half> %715, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %762 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %761, <4 x float> %756) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %759, <4 x float> %757) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %761, <4 x float> %758) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %765 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %766 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %765, <4 x float> %760) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %767 = shufflevector <8 x half> %717, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %768 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %767, <4 x float> %762) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %765, <4 x float> %763) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %767, <4 x float> %764) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %771 = bitcast <4 x i32> %718 to <8 x half>, !dbg !39
  %772 = shufflevector <8 x half> %771, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %772, <4 x float> %766) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %774 = shufflevector <8 x half> %771, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %774, <4 x float> %768) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %772, <4 x float> %769) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %774, <4 x float> %770) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1462 = or disjoint i64 %191, %address.local921, !dbg !41
  %vdb.address1463 = trunc nuw i64 %vdb.address1462 to i32, !dbg !41
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !41
  %vdb.address1471 = or disjoint i64 %191, %address.local927, !dbg !41
  %vdb.address1472 = trunc nuw i64 %vdb.address1471 to i32, !dbg !41
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !41
  %778 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %779 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !39
  %780 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !39
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !39
  %782 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !39
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !39
  %784 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !39
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !39
  %786 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %787 = bitcast <4 x i32> %786 to <8 x half>, !dbg !39
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !39
  %788 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %789 = bitcast <4 x i32> %788 to <8 x half>, !dbg !39
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !39
  %790 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %791 = bitcast <4 x i32> %790 to <8 x half>, !dbg !39
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !39
  %792 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %793 = bitcast <4 x i32> %792 to <8 x half>, !dbg !39
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !39
  %794 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %795 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %796 = extractvalue [2 x <8 x float>] %93, 0, !dbg !54
  %797 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %798 = fmul <4 x float> %797, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %795, <4 x float> %798) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %800 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %801 = shufflevector <8 x float> %796, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %802 = fmul <4 x float> %801, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %800, <4 x float> %802) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %804 = extractvalue [2 x <8 x float>] %93, 1, !dbg !54
  %805 = shufflevector <8 x float> %804, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %806 = fmul <4 x float> %805, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %807 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %795, <4 x float> %806) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %808 = shufflevector <8 x float> %804, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %809 = fmul <4 x float> %808, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %810 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %800, <4 x float> %809) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %811 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %812 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %811, <4 x float> %799) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %813 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %813, <4 x float> %803) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %811, <4 x float> %807) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %816 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %813, <4 x float> %810) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %817 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %818 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %817, <4 x float> %812) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %819 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %819, <4 x float> %814) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %817, <4 x float> %815) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %822 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %819, <4 x float> %816) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %823 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %823, <4 x float> %818) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %825 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %825, <4 x float> %820) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %823, <4 x float> %821) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %828 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %825, <4 x float> %822) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %829 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %829, <4 x float> %824) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %831 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %831, <4 x float> %826) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %829, <4 x float> %827) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %834 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %831, <4 x float> %828) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %835 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %835, <4 x float> %830) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %837 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %837, <4 x float> %832) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %835, <4 x float> %833) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %840 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %837, <4 x float> %834) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %841 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %841, <4 x float> %836) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %843 = shufflevector <8 x half> %793, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %843, <4 x float> %838) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %841, <4 x float> %839) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %843, <4 x float> %840) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %847 = bitcast <4 x i32> %794 to <8 x half>, !dbg !39
  %848 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %848, <4 x float> %842) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %850 = shufflevector <8 x half> %847, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %850, <4 x float> %844) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %852 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %848, <4 x float> %845) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %850, <4 x float> %846) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %vdb.address1480 = or disjoint i64 %244, %address.local921, !dbg !41
  %vdb.address1481 = trunc nuw i64 %vdb.address1480 to i32, !dbg !41
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !41
  %vdb.address1489 = or disjoint i64 %244, %address.local927, !dbg !41
  %vdb.address1490 = trunc nuw i64 %vdb.address1489 to i32, !dbg !41
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !41
  %854 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %855 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #10, !dbg !41
  %856 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !39
  %858 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !39
  %860 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !39
  %862 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !39
  %864 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !39
  %866 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !39
  %868 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !39
  %870 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %871 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %872 = extractvalue [2 x <8 x float>] %94, 0, !dbg !54
  %873 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %874 = fmul <4 x float> %873, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %871, <4 x float> %874) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %876 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %877 = shufflevector <8 x float> %872, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %878 = fmul <4 x float> %877, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %876, <4 x float> %878) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %880 = extractvalue [2 x <8 x float>] %94, 1, !dbg !54
  %881 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %882 = fmul <4 x float> %881, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %871, <4 x float> %882) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %884 = shufflevector <8 x float> %880, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %885 = fmul <4 x float> %884, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %876, <4 x float> %885) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %887 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %888 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %887, <4 x float> %875) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %889 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %889, <4 x float> %879) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %887, <4 x float> %883) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %889, <4 x float> %886) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %893 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %893, <4 x float> %888) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %895 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %895, <4 x float> %890) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %893, <4 x float> %891) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %895, <4 x float> %892) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %899 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %899, <4 x float> %894) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %901 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %901, <4 x float> %896) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %899, <4 x float> %897) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %901, <4 x float> %898) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %905 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %905, <4 x float> %900) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %907 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %907, <4 x float> %902) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %905, <4 x float> %903) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %907, <4 x float> %904) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %911 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %911, <4 x float> %906) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %913 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %913, <4 x float> %908) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %911, <4 x float> %909) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %913, <4 x float> %910) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %917 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %917, <4 x float> %912) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %919 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %919, <4 x float> %914) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %917, <4 x float> %915) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %919, <4 x float> %916) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %923 = bitcast <4 x i32> %870 to <8 x half>, !dbg !39
  %924 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %924, <4 x float> %918) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %926 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %927 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %926, <4 x float> %920) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %924, <4 x float> %921) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %929 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %926, <4 x float> %922) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !41
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !41
  fence syncscope("workgroup") release, !dbg !41
  call void @llvm.amdgcn.s.barrier(), !dbg !41
  fence syncscope("workgroup") acquire, !dbg !41
  %930 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %931 = bitcast <4 x i32> %930 to <8 x half>, !dbg !39
  %932 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %933 = bitcast <4 x i32> %932 to <8 x half>, !dbg !39
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !39
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !39
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !39
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !39
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !39
  %944 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !39, !frisk.v_double_buffer !59
  %945 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  %946 = extractvalue [2 x <8 x float>] %95, 0, !dbg !54
  %947 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %948 = fmul <4 x float> %947, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %945, <4 x float> %948) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %950 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  %951 = shufflevector <8 x float> %946, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %952 = fmul <4 x float> %951, %474, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %950, <4 x float> %952) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %954 = extractvalue [2 x <8 x float>] %95, 1, !dbg !54
  %955 = shufflevector <8 x float> %954, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !54
  %956 = fmul <4 x float> %955, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %945, <4 x float> %956) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !54
  %958 = shufflevector <8 x float> %954, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !54
  %959 = fmul <4 x float> %958, %482, !dbg !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %950, <4 x float> %959) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %961 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %961, <4 x float> %949) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %963 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %963, <4 x float> %953) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %961, <4 x float> %957) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %963, <4 x float> %960) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %967 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %967, <4 x float> %962) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %969 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %969, <4 x float> %964) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %967, <4 x float> %965) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %969, <4 x float> %966) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %973 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %973, <4 x float> %968) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %975 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %975, <4 x float> %970) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %973, <4 x float> %971) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %975, <4 x float> %972) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %979 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %979, <4 x float> %974) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %981 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %981, <4 x float> %976) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %979, <4 x float> %977) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %981, <4 x float> %978) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %985 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %985, <4 x float> %980) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %987 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %987, <4 x float> %982) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %985, <4 x float> %983) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %987, <4 x float> %984) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %991 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %991, <4 x float> %986) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %993 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %993, <4 x float> %988) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %991, <4 x float> %989) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %993, <4 x float> %990) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %997 = bitcast <4 x i32> %944 to <8 x half>, !dbg !39
  %998 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %999 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %998, <4 x float> %992) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1000 = shufflevector <8 x half> %997, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1001 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1000, <4 x float> %994) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1002 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %998, <4 x float> %995) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !39
  %1003 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1000, <4 x float> %996) #9, !dbg !39, !frisk.pv_interleave !58
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !35
  %1004 = shufflevector <4 x float> %773, <4 x float> %775, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1005 = insertvalue [2 x <8 x float>] poison, <8 x float> %1004, 0, !dbg !35
  %1006 = shufflevector <4 x float> %776, <4 x float> %777, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1007 = insertvalue [2 x <8 x float>] %1005, <8 x float> %1006, 1, !dbg !35
  %1008 = insertvalue [2 x <1 x float>] poison, <1 x float> %438, 0, !dbg !35
  %1009 = insertvalue [2 x <1 x float>] %1008, <1 x float> %451, 1, !dbg !35
  %1010 = shufflevector <4 x float> %849, <4 x float> %851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1011 = insertvalue [2 x <8 x float>] poison, <8 x float> %1010, 0, !dbg !35
  %1012 = shufflevector <4 x float> %852, <4 x float> %853, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1013 = insertvalue [2 x <8 x float>] %1011, <8 x float> %1012, 1, !dbg !35
  %1014 = shufflevector <4 x float> %925, <4 x float> %927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1015 = insertvalue [2 x <8 x float>] poison, <8 x float> %1014, 0, !dbg !35
  %1016 = shufflevector <4 x float> %928, <4 x float> %929, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1017 = insertvalue [2 x <8 x float>] %1015, <8 x float> %1016, 1, !dbg !35
  %1018 = shufflevector <4 x float> %999, <4 x float> %1001, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1019 = insertvalue [2 x <8 x float>] poison, <8 x float> %1018, 0, !dbg !35
  %1020 = shufflevector <4 x float> %1002, <4 x float> %1003, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !35
  %1021 = insertvalue [2 x <8 x float>] %1019, <8 x float> %1020, 1, !dbg !35
  fence syncscope("workgroup") release, !dbg !35
  call void @llvm.amdgcn.s.barrier(), !dbg !35
  fence syncscope("workgroup") acquire, !dbg !35
  %1022 = add nuw nsw i64 %89, 1, !dbg !35
  br label %88, !dbg !35

1023:                                             ; preds = %88
  %lane.sum.field184 = extractvalue [2 x <1 x float>] %91, 1, !dbg !35
  %lane.sum.local185 = extractelement <1 x float> %lane.sum.field184, i64 0, !dbg !35
  %lane.sum.lane.lo186 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !35
  %lane.sum.lane187 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lane.sum.lane.lo186), !dbg !35
  %1024 = shl nuw nsw i32 %lane.sum.lane187, 2, !dbg !35
  %1025 = xor i32 %1024, 64, !dbg !35
  %1026 = bitcast float %lane.sum.local185 to i32, !dbg !35
  %lane.sum.shuffle188 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1025, i32 %1026), !dbg !35
  %1027 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !35
  %lane.sum.final189 = fadd float %lane.sum.local185, %1027, !dbg !35
  %1028 = xor i32 %1024, 128, !dbg !35
  %1029 = bitcast float %lane.sum.final189 to i32, !dbg !35
  %lane.sum.shuffle190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1029), !dbg !35
  %1030 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !35
  %lane.sum.final191 = fadd float %lane.sum.final189, %1030, !dbg !35
  %1031 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !35
  %lane.sum.field = extractvalue [2 x <1 x float>] %91, 0, !dbg !35
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !35
  %1032 = bitcast float %lane.sum.local to i32, !dbg !35
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1025, i32 %1032), !dbg !35
  %1033 = bitcast i32 %lane.sum.shuffle to float, !dbg !35
  %lane.sum.final = fadd float %lane.sum.local, %1033, !dbg !35
  %1034 = bitcast float %lane.sum.final to i32, !dbg !35
  %lane.sum.shuffle182 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1034), !dbg !35
  %1035 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !35
  %lane.sum.final183 = fadd float %lane.sum.final, %1035, !dbg !35
  %1036 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !35
  %1037 = and i32 %kdma.tid, 16, !dbg !35
  %.not = icmp eq i32 %1037, 0, !dbg !35
  %1038 = and i32 %kdma.tid, 32, !dbg !35
  %.not36 = icmp eq i32 %1038, 0, !dbg !35
  %1039 = extractvalue [2 x <8 x float>] %95, 0, !dbg !35
  %1040 = extractelement <8 x float> %1039, i64 0, !dbg !35
  %1041 = extractelement <8 x float> %1039, i64 1, !dbg !35
  %1042 = extractelement <8 x float> %1039, i64 2, !dbg !35
  %1043 = extractelement <8 x float> %1039, i64 3, !dbg !35
  %1044 = select i1 %.not, float %1041, float %1040, !dbg !35
  %1045 = add nuw nsw i32 %lane.sum.lane187, 64, !dbg !35
  %1046 = xor i32 %lane.sum.lane187, 16, !dbg !35
  %1047 = and i32 %1045, 192, !dbg !35
  %1048 = icmp samesign ult i32 %1046, %1047, !dbg !35
  %1049 = select i1 %1048, i32 %1046, i32 %lane.sum.lane187, !dbg !35
  %1050 = shl nuw nsw i32 %1049, 2, !dbg !35
  %1051 = bitcast float %1044 to i32, !dbg !35
  %1052 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1051), !dbg !35
  %1053 = bitcast i32 %1052 to float, !dbg !35
  %1054 = select i1 %.not, float %1040, float %1053, !dbg !35
  %1055 = select i1 %.not, float %1053, float %1041, !dbg !35
  %1056 = select i1 %.not, float %1043, float %1042, !dbg !35
  %1057 = bitcast float %1056 to i32, !dbg !35
  %1058 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1057), !dbg !35
  %1059 = bitcast i32 %1058 to float, !dbg !35
  %1060 = select i1 %.not, float %1042, float %1059, !dbg !35
  %1061 = select i1 %.not, float %1059, float %1043, !dbg !35
  %1062 = select i1 %.not36, float %1060, float %1054, !dbg !35
  %1063 = xor i32 %lane.sum.lane187, 32, !dbg !35
  %1064 = icmp samesign ult i32 %1063, %1047, !dbg !35
  %1065 = select i1 %1064, i32 %1063, i32 %lane.sum.lane187, !dbg !35
  %1066 = shl nuw nsw i32 %1065, 2, !dbg !35
  %1067 = bitcast float %1062 to i32, !dbg !35
  %1068 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1067), !dbg !35
  %1069 = bitcast i32 %1068 to float, !dbg !35
  %1070 = select i1 %.not36, float %1054, float %1069, !dbg !35
  %1071 = select i1 %.not36, float %1069, float %1060, !dbg !35
  %1072 = select i1 %.not36, float %1061, float %1055, !dbg !35
  %1073 = bitcast float %1072 to i32, !dbg !35
  %1074 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1073), !dbg !35
  %1075 = bitcast i32 %1074 to float, !dbg !35
  %1076 = select i1 %.not36, float %1055, float %1075, !dbg !35
  %1077 = select i1 %.not36, float %1075, float %1061, !dbg !35
  %1078 = extractvalue [2 x <8 x float>] %94, 0, !dbg !35
  %1079 = extractelement <8 x float> %1078, i64 0, !dbg !35
  %1080 = extractelement <8 x float> %1078, i64 1, !dbg !35
  %1081 = extractelement <8 x float> %1078, i64 2, !dbg !35
  %1082 = extractelement <8 x float> %1078, i64 3, !dbg !35
  %1083 = select i1 %.not, float %1080, float %1079, !dbg !35
  %1084 = bitcast float %1083 to i32, !dbg !35
  %1085 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1084), !dbg !35
  %1086 = bitcast i32 %1085 to float, !dbg !35
  %1087 = select i1 %.not, float %1079, float %1086, !dbg !35
  %1088 = select i1 %.not, float %1086, float %1080, !dbg !35
  %1089 = select i1 %.not, float %1082, float %1081, !dbg !35
  %1090 = bitcast float %1089 to i32, !dbg !35
  %1091 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1090), !dbg !35
  %1092 = bitcast i32 %1091 to float, !dbg !35
  %1093 = select i1 %.not, float %1081, float %1092, !dbg !35
  %1094 = select i1 %.not, float %1092, float %1082, !dbg !35
  %1095 = select i1 %.not36, float %1093, float %1087, !dbg !35
  %1096 = bitcast float %1095 to i32, !dbg !35
  %1097 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1096), !dbg !35
  %1098 = bitcast i32 %1097 to float, !dbg !35
  %1099 = select i1 %.not36, float %1087, float %1098, !dbg !35
  %1100 = select i1 %.not36, float %1098, float %1093, !dbg !35
  %1101 = select i1 %.not36, float %1094, float %1088, !dbg !35
  %1102 = bitcast float %1101 to i32, !dbg !35
  %1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1102), !dbg !35
  %1104 = bitcast i32 %1103 to float, !dbg !35
  %1105 = select i1 %.not36, float %1088, float %1104, !dbg !35
  %1106 = select i1 %.not36, float %1104, float %1094, !dbg !35
  %1107 = extractvalue [2 x <8 x float>] %93, 0, !dbg !35
  %1108 = extractelement <8 x float> %1107, i64 0, !dbg !35
  %1109 = extractelement <8 x float> %1107, i64 1, !dbg !35
  %1110 = extractelement <8 x float> %1107, i64 2, !dbg !35
  %1111 = extractelement <8 x float> %1107, i64 3, !dbg !35
  %1112 = select i1 %.not, float %1109, float %1108, !dbg !35
  %1113 = bitcast float %1112 to i32, !dbg !35
  %1114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1113), !dbg !35
  %1115 = bitcast i32 %1114 to float, !dbg !35
  %1116 = select i1 %.not, float %1108, float %1115, !dbg !35
  %1117 = select i1 %.not, float %1115, float %1109, !dbg !35
  %1118 = select i1 %.not, float %1111, float %1110, !dbg !35
  %1119 = bitcast float %1118 to i32, !dbg !35
  %1120 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1119), !dbg !35
  %1121 = bitcast i32 %1120 to float, !dbg !35
  %1122 = select i1 %.not, float %1110, float %1121, !dbg !35
  %1123 = select i1 %.not, float %1121, float %1111, !dbg !35
  %1124 = select i1 %.not36, float %1122, float %1116, !dbg !35
  %1125 = bitcast float %1124 to i32, !dbg !35
  %1126 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1125), !dbg !35
  %1127 = bitcast i32 %1126 to float, !dbg !35
  %1128 = select i1 %.not36, float %1116, float %1127, !dbg !35
  %1129 = select i1 %.not36, float %1127, float %1122, !dbg !35
  %1130 = select i1 %.not36, float %1123, float %1117, !dbg !35
  %1131 = bitcast float %1130 to i32, !dbg !35
  %1132 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1131), !dbg !35
  %1133 = bitcast i32 %1132 to float, !dbg !35
  %1134 = select i1 %.not36, float %1117, float %1133, !dbg !35
  %1135 = select i1 %.not36, float %1133, float %1123, !dbg !35
  %1136 = extractvalue [2 x <8 x float>] %90, 0, !dbg !35
  %1137 = extractelement <8 x float> %1136, i64 0, !dbg !35
  %1138 = extractelement <8 x float> %1136, i64 1, !dbg !35
  %1139 = extractelement <8 x float> %1136, i64 2, !dbg !35
  %1140 = extractelement <8 x float> %1136, i64 3, !dbg !35
  %1141 = select i1 %.not, float %1138, float %1137, !dbg !35
  %1142 = bitcast float %1141 to i32, !dbg !35
  %1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1142), !dbg !35
  %1144 = bitcast i32 %1143 to float, !dbg !35
  %1145 = select i1 %.not, float %1137, float %1144, !dbg !35
  %1146 = select i1 %.not, float %1144, float %1138, !dbg !35
  %1147 = select i1 %.not, float %1140, float %1139, !dbg !35
  %1148 = bitcast float %1147 to i32, !dbg !35
  %1149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1148), !dbg !35
  %1150 = bitcast i32 %1149 to float, !dbg !35
  %1151 = select i1 %.not, float %1139, float %1150, !dbg !35
  %1152 = select i1 %.not, float %1150, float %1140, !dbg !35
  %1153 = select i1 %.not36, float %1151, float %1145, !dbg !35
  %1154 = bitcast float %1153 to i32, !dbg !35
  %1155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1154), !dbg !35
  %1156 = select i1 %.not36, float %1152, float %1146, !dbg !35
  %1157 = bitcast float %1156 to i32, !dbg !35
  %1158 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1157), !dbg !35
  %1159 = shufflevector <1 x float> %1036, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %1160 = extractelement <8 x float> %1039, i64 4, !dbg !35
  %1161 = extractelement <8 x float> %1039, i64 5, !dbg !35
  %1162 = extractelement <8 x float> %1039, i64 6, !dbg !35
  %1163 = extractelement <8 x float> %1039, i64 7, !dbg !35
  %1164 = select i1 %.not, float %1161, float %1160, !dbg !35
  %1165 = bitcast float %1164 to i32, !dbg !35
  %1166 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1165), !dbg !35
  %1167 = bitcast i32 %1166 to float, !dbg !35
  %1168 = select i1 %.not, float %1160, float %1167, !dbg !35
  %1169 = select i1 %.not, float %1167, float %1161, !dbg !35
  %1170 = select i1 %.not, float %1163, float %1162, !dbg !35
  %1171 = bitcast float %1170 to i32, !dbg !35
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1171), !dbg !35
  %1173 = bitcast i32 %1172 to float, !dbg !35
  %1174 = select i1 %.not, float %1162, float %1173, !dbg !35
  %1175 = select i1 %.not, float %1173, float %1163, !dbg !35
  %1176 = select i1 %.not36, float %1174, float %1168, !dbg !35
  %1177 = bitcast float %1176 to i32, !dbg !35
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1177), !dbg !35
  %1179 = bitcast i32 %1178 to float, !dbg !35
  %1180 = select i1 %.not36, float %1168, float %1179, !dbg !35
  %1181 = select i1 %.not36, float %1179, float %1174, !dbg !35
  %1182 = select i1 %.not36, float %1175, float %1169, !dbg !35
  %1183 = bitcast float %1182 to i32, !dbg !35
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1183), !dbg !35
  %1185 = bitcast i32 %1184 to float, !dbg !35
  %1186 = select i1 %.not36, float %1169, float %1185, !dbg !35
  %1187 = select i1 %.not36, float %1185, float %1175, !dbg !35
  %1188 = extractelement <8 x float> %1078, i64 4, !dbg !35
  %1189 = extractelement <8 x float> %1078, i64 5, !dbg !35
  %1190 = extractelement <8 x float> %1078, i64 6, !dbg !35
  %1191 = extractelement <8 x float> %1078, i64 7, !dbg !35
  %1192 = select i1 %.not, float %1189, float %1188, !dbg !35
  %1193 = bitcast float %1192 to i32, !dbg !35
  %1194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1193), !dbg !35
  %1195 = bitcast i32 %1194 to float, !dbg !35
  %1196 = select i1 %.not, float %1188, float %1195, !dbg !35
  %1197 = select i1 %.not, float %1195, float %1189, !dbg !35
  %1198 = select i1 %.not, float %1191, float %1190, !dbg !35
  %1199 = bitcast float %1198 to i32, !dbg !35
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1199), !dbg !35
  %1201 = bitcast i32 %1200 to float, !dbg !35
  %1202 = select i1 %.not, float %1190, float %1201, !dbg !35
  %1203 = select i1 %.not, float %1201, float %1191, !dbg !35
  %1204 = select i1 %.not36, float %1202, float %1196, !dbg !35
  %1205 = bitcast float %1204 to i32, !dbg !35
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1205), !dbg !35
  %1207 = bitcast i32 %1206 to float, !dbg !35
  %1208 = select i1 %.not36, float %1196, float %1207, !dbg !35
  %1209 = select i1 %.not36, float %1207, float %1202, !dbg !35
  %1210 = select i1 %.not36, float %1203, float %1197, !dbg !35
  %1211 = bitcast float %1210 to i32, !dbg !35
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1211), !dbg !35
  %1213 = bitcast i32 %1212 to float, !dbg !35
  %1214 = select i1 %.not36, float %1197, float %1213, !dbg !35
  %1215 = select i1 %.not36, float %1213, float %1203, !dbg !35
  %1216 = extractelement <8 x float> %1107, i64 4, !dbg !35
  %1217 = extractelement <8 x float> %1107, i64 5, !dbg !35
  %1218 = extractelement <8 x float> %1107, i64 6, !dbg !35
  %1219 = extractelement <8 x float> %1107, i64 7, !dbg !35
  %1220 = select i1 %.not, float %1217, float %1216, !dbg !35
  %1221 = bitcast float %1220 to i32, !dbg !35
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1221), !dbg !35
  %1223 = bitcast i32 %1222 to float, !dbg !35
  %1224 = select i1 %.not, float %1216, float %1223, !dbg !35
  %1225 = select i1 %.not, float %1223, float %1217, !dbg !35
  %1226 = select i1 %.not, float %1219, float %1218, !dbg !35
  %1227 = bitcast float %1226 to i32, !dbg !35
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1227), !dbg !35
  %1229 = bitcast i32 %1228 to float, !dbg !35
  %1230 = select i1 %.not, float %1218, float %1229, !dbg !35
  %1231 = select i1 %.not, float %1229, float %1219, !dbg !35
  %1232 = select i1 %.not36, float %1230, float %1224, !dbg !35
  %1233 = bitcast float %1232 to i32, !dbg !35
  %1234 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1233), !dbg !35
  %1235 = bitcast i32 %1234 to float, !dbg !35
  %1236 = select i1 %.not36, float %1224, float %1235, !dbg !35
  %1237 = select i1 %.not36, float %1235, float %1230, !dbg !35
  %1238 = select i1 %.not36, float %1231, float %1225, !dbg !35
  %1239 = bitcast float %1238 to i32, !dbg !35
  %1240 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1239), !dbg !35
  %1241 = bitcast i32 %1240 to float, !dbg !35
  %1242 = select i1 %.not36, float %1225, float %1241, !dbg !35
  %1243 = select i1 %.not36, float %1241, float %1231, !dbg !35
  %1244 = extractelement <8 x float> %1136, i64 4, !dbg !35
  %1245 = extractelement <8 x float> %1136, i64 5, !dbg !35
  %1246 = extractelement <8 x float> %1136, i64 6, !dbg !35
  %1247 = extractelement <8 x float> %1136, i64 7, !dbg !35
  %1248 = select i1 %.not, float %1245, float %1244, !dbg !35
  %1249 = bitcast float %1248 to i32, !dbg !35
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1249), !dbg !35
  %1251 = bitcast i32 %1250 to float, !dbg !35
  %1252 = select i1 %.not, float %1244, float %1251, !dbg !35
  %1253 = select i1 %.not, float %1251, float %1245, !dbg !35
  %1254 = select i1 %.not, float %1247, float %1246, !dbg !35
  %1255 = bitcast float %1254 to i32, !dbg !35
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1255), !dbg !35
  %1257 = bitcast i32 %1256 to float, !dbg !35
  %1258 = select i1 %.not, float %1246, float %1257, !dbg !35
  %1259 = select i1 %.not, float %1257, float %1247, !dbg !35
  %1260 = select i1 %.not36, float %1258, float %1252, !dbg !35
  %1261 = bitcast float %1260 to i32, !dbg !35
  %1262 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1261), !dbg !35
  %1263 = select i1 %.not36, float %1259, float %1253, !dbg !35
  %1264 = bitcast float %1263 to i32, !dbg !35
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1264), !dbg !35
  %1266 = extractvalue [2 x <8 x float>] %95, 1, !dbg !35
  %1267 = extractelement <8 x float> %1266, i64 0, !dbg !35
  %1268 = extractelement <8 x float> %1266, i64 1, !dbg !35
  %1269 = extractelement <8 x float> %1266, i64 2, !dbg !35
  %1270 = extractelement <8 x float> %1266, i64 3, !dbg !35
  %1271 = select i1 %.not, float %1268, float %1267, !dbg !35
  %1272 = bitcast float %1271 to i32, !dbg !35
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1272), !dbg !35
  %1274 = bitcast i32 %1273 to float, !dbg !35
  %1275 = select i1 %.not, float %1267, float %1274, !dbg !35
  %1276 = select i1 %.not, float %1274, float %1268, !dbg !35
  %1277 = select i1 %.not, float %1270, float %1269, !dbg !35
  %1278 = bitcast float %1277 to i32, !dbg !35
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1278), !dbg !35
  %1280 = bitcast i32 %1279 to float, !dbg !35
  %1281 = select i1 %.not, float %1269, float %1280, !dbg !35
  %1282 = select i1 %.not, float %1280, float %1270, !dbg !35
  %1283 = select i1 %.not36, float %1281, float %1275, !dbg !35
  %1284 = bitcast float %1283 to i32, !dbg !35
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1284), !dbg !35
  %1286 = bitcast i32 %1285 to float, !dbg !35
  %1287 = select i1 %.not36, float %1275, float %1286, !dbg !35
  %1288 = select i1 %.not36, float %1286, float %1281, !dbg !35
  %1289 = select i1 %.not36, float %1282, float %1276, !dbg !35
  %1290 = bitcast float %1289 to i32, !dbg !35
  %1291 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1290), !dbg !35
  %1292 = bitcast i32 %1291 to float, !dbg !35
  %1293 = select i1 %.not36, float %1276, float %1292, !dbg !35
  %1294 = select i1 %.not36, float %1292, float %1282, !dbg !35
  %1295 = extractvalue [2 x <8 x float>] %94, 1, !dbg !35
  %1296 = extractelement <8 x float> %1295, i64 0, !dbg !35
  %1297 = extractelement <8 x float> %1295, i64 1, !dbg !35
  %1298 = extractelement <8 x float> %1295, i64 2, !dbg !35
  %1299 = extractelement <8 x float> %1295, i64 3, !dbg !35
  %1300 = select i1 %.not, float %1297, float %1296, !dbg !35
  %1301 = bitcast float %1300 to i32, !dbg !35
  %1302 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1301), !dbg !35
  %1303 = bitcast i32 %1302 to float, !dbg !35
  %1304 = select i1 %.not, float %1296, float %1303, !dbg !35
  %1305 = select i1 %.not, float %1303, float %1297, !dbg !35
  %1306 = select i1 %.not, float %1299, float %1298, !dbg !35
  %1307 = bitcast float %1306 to i32, !dbg !35
  %1308 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1307), !dbg !35
  %1309 = bitcast i32 %1308 to float, !dbg !35
  %1310 = select i1 %.not, float %1298, float %1309, !dbg !35
  %1311 = select i1 %.not, float %1309, float %1299, !dbg !35
  %1312 = select i1 %.not36, float %1310, float %1304, !dbg !35
  %1313 = bitcast float %1312 to i32, !dbg !35
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1313), !dbg !35
  %1315 = bitcast i32 %1314 to float, !dbg !35
  %1316 = select i1 %.not36, float %1304, float %1315, !dbg !35
  %1317 = select i1 %.not36, float %1315, float %1310, !dbg !35
  %1318 = select i1 %.not36, float %1311, float %1305, !dbg !35
  %1319 = bitcast float %1318 to i32, !dbg !35
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1319), !dbg !35
  %1321 = bitcast i32 %1320 to float, !dbg !35
  %1322 = select i1 %.not36, float %1305, float %1321, !dbg !35
  %1323 = select i1 %.not36, float %1321, float %1311, !dbg !35
  %1324 = extractvalue [2 x <8 x float>] %93, 1, !dbg !35
  %1325 = extractelement <8 x float> %1324, i64 0, !dbg !35
  %1326 = extractelement <8 x float> %1324, i64 1, !dbg !35
  %1327 = extractelement <8 x float> %1324, i64 2, !dbg !35
  %1328 = extractelement <8 x float> %1324, i64 3, !dbg !35
  %1329 = select i1 %.not, float %1326, float %1325, !dbg !35
  %1330 = bitcast float %1329 to i32, !dbg !35
  %1331 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1330), !dbg !35
  %1332 = bitcast i32 %1331 to float, !dbg !35
  %1333 = select i1 %.not, float %1325, float %1332, !dbg !35
  %1334 = select i1 %.not, float %1332, float %1326, !dbg !35
  %1335 = select i1 %.not, float %1328, float %1327, !dbg !35
  %1336 = bitcast float %1335 to i32, !dbg !35
  %1337 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1336), !dbg !35
  %1338 = bitcast i32 %1337 to float, !dbg !35
  %1339 = select i1 %.not, float %1327, float %1338, !dbg !35
  %1340 = select i1 %.not, float %1338, float %1328, !dbg !35
  %1341 = select i1 %.not36, float %1339, float %1333, !dbg !35
  %1342 = bitcast float %1341 to i32, !dbg !35
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1342), !dbg !35
  %1344 = bitcast i32 %1343 to float, !dbg !35
  %1345 = select i1 %.not36, float %1333, float %1344, !dbg !35
  %1346 = select i1 %.not36, float %1344, float %1339, !dbg !35
  %1347 = select i1 %.not36, float %1340, float %1334, !dbg !35
  %1348 = bitcast float %1347 to i32, !dbg !35
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1348), !dbg !35
  %1350 = bitcast i32 %1349 to float, !dbg !35
  %1351 = select i1 %.not36, float %1334, float %1350, !dbg !35
  %1352 = select i1 %.not36, float %1350, float %1340, !dbg !35
  %1353 = extractvalue [2 x <8 x float>] %90, 1, !dbg !35
  %1354 = extractelement <8 x float> %1353, i64 0, !dbg !35
  %1355 = extractelement <8 x float> %1353, i64 1, !dbg !35
  %1356 = extractelement <8 x float> %1353, i64 2, !dbg !35
  %1357 = extractelement <8 x float> %1353, i64 3, !dbg !35
  %1358 = select i1 %.not, float %1355, float %1354, !dbg !35
  %1359 = bitcast float %1358 to i32, !dbg !35
  %1360 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1359), !dbg !35
  %1361 = bitcast i32 %1360 to float, !dbg !35
  %1362 = select i1 %.not, float %1354, float %1361, !dbg !35
  %1363 = select i1 %.not, float %1361, float %1355, !dbg !35
  %1364 = select i1 %.not, float %1357, float %1356, !dbg !35
  %1365 = bitcast float %1364 to i32, !dbg !35
  %1366 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1365), !dbg !35
  %1367 = bitcast i32 %1366 to float, !dbg !35
  %1368 = select i1 %.not, float %1356, float %1367, !dbg !35
  %1369 = select i1 %.not, float %1367, float %1357, !dbg !35
  %1370 = select i1 %.not36, float %1368, float %1362, !dbg !35
  %1371 = bitcast float %1370 to i32, !dbg !35
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1371), !dbg !35
  %1373 = select i1 %.not36, float %1369, float %1363, !dbg !35
  %1374 = bitcast float %1373 to i32, !dbg !35
  %1375 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1374), !dbg !35
  %1376 = shufflevector <1 x float> %1031, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !42
  %1377 = extractelement <8 x float> %1266, i64 4, !dbg !35
  %1378 = extractelement <8 x float> %1266, i64 5, !dbg !35
  %1379 = extractelement <8 x float> %1266, i64 6, !dbg !35
  %1380 = extractelement <8 x float> %1266, i64 7, !dbg !35
  %1381 = select i1 %.not, float %1378, float %1377, !dbg !35
  %1382 = bitcast float %1381 to i32, !dbg !35
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1382), !dbg !35
  %1384 = bitcast i32 %1383 to float, !dbg !35
  %1385 = select i1 %.not, float %1377, float %1384, !dbg !35
  %1386 = select i1 %.not, float %1384, float %1378, !dbg !35
  %1387 = select i1 %.not, float %1380, float %1379, !dbg !35
  %1388 = bitcast float %1387 to i32, !dbg !35
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1388), !dbg !35
  %1390 = bitcast i32 %1389 to float, !dbg !35
  %1391 = select i1 %.not, float %1379, float %1390, !dbg !35
  %1392 = select i1 %.not, float %1390, float %1380, !dbg !35
  %1393 = select i1 %.not36, float %1391, float %1385, !dbg !35
  %1394 = bitcast float %1393 to i32, !dbg !35
  %1395 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1394), !dbg !35
  %1396 = bitcast i32 %1395 to float, !dbg !35
  %1397 = select i1 %.not36, float %1385, float %1396, !dbg !35
  %1398 = select i1 %.not36, float %1396, float %1391, !dbg !35
  %1399 = select i1 %.not36, float %1392, float %1386, !dbg !35
  %1400 = bitcast float %1399 to i32, !dbg !35
  %1401 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1400), !dbg !35
  %1402 = bitcast i32 %1401 to float, !dbg !35
  %1403 = select i1 %.not36, float %1386, float %1402, !dbg !35
  %1404 = select i1 %.not36, float %1402, float %1392, !dbg !35
  %1405 = extractelement <8 x float> %1295, i64 4, !dbg !35
  %1406 = extractelement <8 x float> %1295, i64 5, !dbg !35
  %1407 = extractelement <8 x float> %1295, i64 6, !dbg !35
  %1408 = extractelement <8 x float> %1295, i64 7, !dbg !35
  %1409 = select i1 %.not, float %1406, float %1405, !dbg !35
  %1410 = bitcast float %1409 to i32, !dbg !35
  %1411 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1410), !dbg !35
  %1412 = bitcast i32 %1411 to float, !dbg !35
  %1413 = select i1 %.not, float %1405, float %1412, !dbg !35
  %1414 = select i1 %.not, float %1412, float %1406, !dbg !35
  %1415 = select i1 %.not, float %1408, float %1407, !dbg !35
  %1416 = bitcast float %1415 to i32, !dbg !35
  %1417 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1416), !dbg !35
  %1418 = bitcast i32 %1417 to float, !dbg !35
  %1419 = select i1 %.not, float %1407, float %1418, !dbg !35
  %1420 = select i1 %.not, float %1418, float %1408, !dbg !35
  %1421 = select i1 %.not36, float %1419, float %1413, !dbg !35
  %1422 = bitcast float %1421 to i32, !dbg !35
  %1423 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1422), !dbg !35
  %1424 = bitcast i32 %1423 to float, !dbg !35
  %1425 = select i1 %.not36, float %1413, float %1424, !dbg !35
  %1426 = select i1 %.not36, float %1424, float %1419, !dbg !35
  %1427 = select i1 %.not36, float %1420, float %1414, !dbg !35
  %1428 = bitcast float %1427 to i32, !dbg !35
  %1429 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1428), !dbg !35
  %1430 = bitcast i32 %1429 to float, !dbg !35
  %1431 = select i1 %.not36, float %1414, float %1430, !dbg !35
  %1432 = select i1 %.not36, float %1430, float %1420, !dbg !35
  %1433 = extractelement <8 x float> %1324, i64 4, !dbg !35
  %1434 = extractelement <8 x float> %1324, i64 5, !dbg !35
  %1435 = extractelement <8 x float> %1324, i64 6, !dbg !35
  %1436 = extractelement <8 x float> %1324, i64 7, !dbg !35
  %1437 = select i1 %.not, float %1434, float %1433, !dbg !35
  %1438 = bitcast float %1437 to i32, !dbg !35
  %1439 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1438), !dbg !35
  %1440 = bitcast i32 %1439 to float, !dbg !35
  %1441 = select i1 %.not, float %1433, float %1440, !dbg !35
  %1442 = select i1 %.not, float %1440, float %1434, !dbg !35
  %1443 = select i1 %.not, float %1436, float %1435, !dbg !35
  %1444 = bitcast float %1443 to i32, !dbg !35
  %1445 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1444), !dbg !35
  %1446 = bitcast i32 %1445 to float, !dbg !35
  %1447 = select i1 %.not, float %1435, float %1446, !dbg !35
  %1448 = select i1 %.not, float %1446, float %1436, !dbg !35
  %1449 = select i1 %.not36, float %1447, float %1441, !dbg !35
  %1450 = bitcast float %1449 to i32, !dbg !35
  %1451 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1450), !dbg !35
  %1452 = bitcast i32 %1451 to float, !dbg !35
  %1453 = select i1 %.not36, float %1441, float %1452, !dbg !35
  %1454 = select i1 %.not36, float %1452, float %1447, !dbg !35
  %1455 = select i1 %.not36, float %1448, float %1442, !dbg !35
  %1456 = bitcast float %1455 to i32, !dbg !35
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1456), !dbg !35
  %1458 = bitcast i32 %1457 to float, !dbg !35
  %1459 = select i1 %.not36, float %1442, float %1458, !dbg !35
  %1460 = select i1 %.not36, float %1458, float %1448, !dbg !35
  %1461 = extractelement <8 x float> %1353, i64 4, !dbg !35
  %1462 = extractelement <8 x float> %1353, i64 5, !dbg !35
  %1463 = extractelement <8 x float> %1353, i64 6, !dbg !35
  %1464 = extractelement <8 x float> %1353, i64 7, !dbg !35
  %1465 = select i1 %.not, float %1462, float %1461, !dbg !35
  %1466 = bitcast float %1465 to i32, !dbg !35
  %1467 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1466), !dbg !35
  %1468 = bitcast i32 %1467 to float, !dbg !35
  %1469 = select i1 %.not, float %1461, float %1468, !dbg !35
  %1470 = select i1 %.not, float %1468, float %1462, !dbg !35
  %1471 = select i1 %.not, float %1464, float %1463, !dbg !35
  %1472 = bitcast float %1471 to i32, !dbg !35
  %1473 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1050, i32 %1472), !dbg !35
  %1474 = bitcast i32 %1473 to float, !dbg !35
  %1475 = select i1 %.not, float %1463, float %1474, !dbg !35
  %1476 = select i1 %.not, float %1474, float %1464, !dbg !35
  %1477 = select i1 %.not36, float %1475, float %1469, !dbg !35
  %1478 = bitcast float %1477 to i32, !dbg !35
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1478), !dbg !35
  %1480 = select i1 %.not36, float %1476, float %1470, !dbg !35
  %1481 = bitcast float %1480 to i32, !dbg !35
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1066, i32 %1481), !dbg !35
  %1483 = bitcast i32 %1479 to float, !dbg !35
  %1484 = select i1 %.not36, float %1469, float %1483, !dbg !35
  %1485 = insertelement <4 x float> poison, float %1484, i64 0, !dbg !42
  %1486 = bitcast i32 %1482 to float, !dbg !35
  %1487 = select i1 %.not36, float %1470, float %1486, !dbg !35
  %1488 = insertelement <4 x float> %1485, float %1487, i64 1, !dbg !42
  %1489 = select i1 %.not36, float %1483, float %1475, !dbg !35
  %1490 = insertelement <4 x float> %1488, float %1489, i64 2, !dbg !42
  %1491 = select i1 %.not36, float %1486, float %1476, !dbg !35
  %1492 = insertelement <4 x float> %1490, float %1491, i64 3, !dbg !42
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1376, !dbg !42
  %normalized = fmul <4 x float> %1492, %row.reciprocal, !dbg !42
  %1493 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !43
  %1494 = extractelement <4 x half> %1493, i64 3, !dbg !43
  %1495 = extractelement <4 x half> %1493, i64 2, !dbg !43
  %1496 = extractelement <4 x half> %1493, i64 1, !dbg !43
  %1497 = extractelement <4 x half> %1493, i64 0, !dbg !43
  %1498 = bitcast i32 %1372 to float, !dbg !35
  %1499 = select i1 %.not36, float %1362, float %1498, !dbg !35
  %1500 = insertelement <4 x float> poison, float %1499, i64 0, !dbg !42
  %1501 = bitcast i32 %1375 to float, !dbg !35
  %1502 = select i1 %.not36, float %1363, float %1501, !dbg !35
  %1503 = insertelement <4 x float> %1500, float %1502, i64 1, !dbg !42
  %1504 = select i1 %.not36, float %1498, float %1368, !dbg !35
  %1505 = insertelement <4 x float> %1503, float %1504, i64 2, !dbg !42
  %1506 = select i1 %.not36, float %1501, float %1369, !dbg !35
  %1507 = insertelement <4 x float> %1505, float %1506, i64 3, !dbg !42
  %normalized37 = fmul <4 x float> %1507, %row.reciprocal, !dbg !42
  %1508 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !43
  %1509 = extractelement <4 x half> %1508, i64 3, !dbg !43
  %1510 = extractelement <4 x half> %1508, i64 2, !dbg !43
  %1511 = extractelement <4 x half> %1508, i64 1, !dbg !43
  %1512 = extractelement <4 x half> %1508, i64 0, !dbg !43
  %1513 = bitcast i32 %1262 to float, !dbg !35
  %1514 = select i1 %.not36, float %1252, float %1513, !dbg !35
  %1515 = insertelement <4 x float> poison, float %1514, i64 0, !dbg !42
  %1516 = bitcast i32 %1265 to float, !dbg !35
  %1517 = select i1 %.not36, float %1253, float %1516, !dbg !35
  %1518 = insertelement <4 x float> %1515, float %1517, i64 1, !dbg !42
  %1519 = select i1 %.not36, float %1513, float %1258, !dbg !35
  %1520 = insertelement <4 x float> %1518, float %1519, i64 2, !dbg !42
  %1521 = select i1 %.not36, float %1516, float %1259, !dbg !35
  %1522 = insertelement <4 x float> %1520, float %1521, i64 3, !dbg !42
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1159, !dbg !42
  %normalized39 = fmul <4 x float> %1522, %row.reciprocal38, !dbg !42
  %1523 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !43
  %1524 = extractelement <4 x half> %1523, i64 3, !dbg !43
  %1525 = extractelement <4 x half> %1523, i64 2, !dbg !43
  %1526 = extractelement <4 x half> %1523, i64 1, !dbg !43
  %1527 = extractelement <4 x half> %1523, i64 0, !dbg !43
  %1528 = bitcast i32 %1155 to float, !dbg !35
  %1529 = select i1 %.not36, float %1145, float %1528, !dbg !35
  %1530 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !42
  %1531 = bitcast i32 %1158 to float, !dbg !35
  %1532 = select i1 %.not36, float %1146, float %1531, !dbg !35
  %1533 = insertelement <4 x float> %1530, float %1532, i64 1, !dbg !42
  %1534 = select i1 %.not36, float %1528, float %1151, !dbg !35
  %1535 = insertelement <4 x float> %1533, float %1534, i64 2, !dbg !42
  %1536 = select i1 %.not36, float %1531, float %1152, !dbg !35
  %1537 = insertelement <4 x float> %1535, float %1536, i64 3, !dbg !42
  %normalized40 = fmul <4 x float> %1537, %row.reciprocal38, !dbg !42
  %1538 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !43
  %1539 = extractelement <4 x half> %1538, i64 3, !dbg !43
  %1540 = extractelement <4 x half> %1538, i64 2, !dbg !43
  %1541 = extractelement <4 x half> %1538, i64 1, !dbg !43
  %1542 = extractelement <4 x half> %1538, i64 0, !dbg !43
  %address.thread974 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !44, !range !57, !frisk.register_lifetimes !30
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !36
  %address.local976 = add nuw i64 %19, %address.local975, !dbg !44
  %1543 = lshr i64 %address.local975, 2, !dbg !44
  %address.local978 = and i64 %1543, 48, !dbg !44
  %address.local979 = add i64 %address.local976, %address.local978, !dbg !44
  %address.local980 = and i64 %address.local975, 240, !dbg !44
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !44
  %address.local982 = add i64 %address.local981, %20, !dbg !44
  %address.local983 = shl nuw nsw i64 %address.local982, 7, !dbg !44
  %1544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %23, !dbg !44
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1544, i64 %address.local983, !dbg !44
  %address.local987 = and i64 %1543, 12, !dbg !37
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !44
  store half %1542, ptr addrspace(1) %address.local988, align 2, !dbg !44
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !44
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !44
  store half %1541, ptr addrspace(1) %address.local991, align 2, !dbg !44
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !44
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !44
  store half %1540, ptr addrspace(1) %address.local994, align 2, !dbg !44
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !44
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !44
  store half %1539, ptr addrspace(1) %address.local997, align 2, !dbg !44
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !44
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !44
  store half %1527, ptr addrspace(1) %address.local1000, align 2, !dbg !44
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !44
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !44
  store half %1526, ptr addrspace(1) %address.local1003, align 2, !dbg !44
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !44
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !44
  store half %1525, ptr addrspace(1) %address.local1006, align 2, !dbg !44
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !44
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !44
  store half %1524, ptr addrspace(1) %address.local1009, align 2, !dbg !44
  %address.local1011 = add nuw i64 %address.local983, 8192, !dbg !44
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1544, i64 %address.local1011, !dbg !44
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !44
  store half %1512, ptr addrspace(1) %address.local1014, align 2, !dbg !44
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !44
  store half %1511, ptr addrspace(1) %address.local1016, align 2, !dbg !44
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !44
  store half %1510, ptr addrspace(1) %address.local1018, align 2, !dbg !44
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !44
  store half %1509, ptr addrspace(1) %address.local1020, align 2, !dbg !44
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !44
  store half %1497, ptr addrspace(1) %address.local1022, align 2, !dbg !44
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !44
  store half %1496, ptr addrspace(1) %address.local1024, align 2, !dbg !44
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !44
  store half %1495, ptr addrspace(1) %address.local1026, align 2, !dbg !44
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !44
  store half %1494, ptr addrspace(1) %address.local1028, align 2, !dbg !44
  %1545 = insertelement <4 x float> poison, float %1453, i64 0, !dbg !42
  %1546 = insertelement <4 x float> %1545, float %1459, i64 1, !dbg !42
  %1547 = insertelement <4 x float> %1546, float %1454, i64 2, !dbg !42
  %1548 = insertelement <4 x float> %1547, float %1460, i64 3, !dbg !42
  %normalized42 = fmul <4 x float> %1548, %row.reciprocal, !dbg !42
  %1549 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !43
  %1550 = extractelement <4 x half> %1549, i64 3, !dbg !43
  %1551 = extractelement <4 x half> %1549, i64 2, !dbg !43
  %1552 = extractelement <4 x half> %1549, i64 1, !dbg !43
  %1553 = extractelement <4 x half> %1549, i64 0, !dbg !43
  %1554 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !42
  %1555 = insertelement <4 x float> %1554, float %1351, i64 1, !dbg !42
  %1556 = insertelement <4 x float> %1555, float %1346, i64 2, !dbg !42
  %1557 = insertelement <4 x float> %1556, float %1352, i64 3, !dbg !42
  %normalized43 = fmul <4 x float> %1557, %row.reciprocal, !dbg !42
  %1558 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !43
  %1559 = extractelement <4 x half> %1558, i64 3, !dbg !43
  %1560 = extractelement <4 x half> %1558, i64 2, !dbg !43
  %1561 = extractelement <4 x half> %1558, i64 1, !dbg !43
  %1562 = extractelement <4 x half> %1558, i64 0, !dbg !43
  %1563 = insertelement <4 x float> poison, float %1236, i64 0, !dbg !42
  %1564 = insertelement <4 x float> %1563, float %1242, i64 1, !dbg !42
  %1565 = insertelement <4 x float> %1564, float %1237, i64 2, !dbg !42
  %1566 = insertelement <4 x float> %1565, float %1243, i64 3, !dbg !42
  %normalized45 = fmul <4 x float> %1566, %row.reciprocal38, !dbg !42
  %1567 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !43
  %1568 = extractelement <4 x half> %1567, i64 3, !dbg !43
  %1569 = extractelement <4 x half> %1567, i64 2, !dbg !43
  %1570 = extractelement <4 x half> %1567, i64 1, !dbg !43
  %1571 = extractelement <4 x half> %1567, i64 0, !dbg !43
  %1572 = insertelement <4 x float> poison, float %1128, i64 0, !dbg !42
  %1573 = insertelement <4 x float> %1572, float %1134, i64 1, !dbg !42
  %1574 = insertelement <4 x float> %1573, float %1129, i64 2, !dbg !42
  %1575 = insertelement <4 x float> %1574, float %1135, i64 3, !dbg !42
  %normalized46 = fmul <4 x float> %1575, %row.reciprocal38, !dbg !42
  %1576 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !43
  %1577 = extractelement <4 x half> %1576, i64 3, !dbg !43
  %1578 = extractelement <4 x half> %1576, i64 2, !dbg !43
  %1579 = extractelement <4 x half> %1576, i64 1, !dbg !43
  %1580 = extractelement <4 x half> %1576, i64 0, !dbg !43
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !44
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !44
  store half %1580, ptr addrspace(1) %address.local1044, align 2, !dbg !44
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !44
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !44
  store half %1579, ptr addrspace(1) %address.local1047, align 2, !dbg !44
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !44
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !44
  store half %1578, ptr addrspace(1) %address.local1050, align 2, !dbg !44
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !44
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !44
  store half %1577, ptr addrspace(1) %address.local1053, align 2, !dbg !44
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !44
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !44
  store half %1571, ptr addrspace(1) %address.local1056, align 2, !dbg !44
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !44
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !44
  store half %1570, ptr addrspace(1) %address.local1059, align 2, !dbg !44
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !44
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !44
  store half %1569, ptr addrspace(1) %address.local1062, align 2, !dbg !44
  %address.local1064 = or i64 %1543, 51, !dbg !44
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !44
  store half %1568, ptr addrspace(1) %address.local1065, align 2, !dbg !44
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !44
  store half %1562, ptr addrspace(1) %address.local1070, align 2, !dbg !44
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !44
  store half %1561, ptr addrspace(1) %address.local1072, align 2, !dbg !44
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !44
  store half %1560, ptr addrspace(1) %address.local1074, align 2, !dbg !44
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !44
  store half %1559, ptr addrspace(1) %address.local1076, align 2, !dbg !44
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !44
  store half %1553, ptr addrspace(1) %address.local1078, align 2, !dbg !44
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !44
  store half %1552, ptr addrspace(1) %address.local1080, align 2, !dbg !44
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !44
  store half %1551, ptr addrspace(1) %address.local1082, align 2, !dbg !44
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !44
  store half %1550, ptr addrspace(1) %address.local1084, align 2, !dbg !44
  %1581 = insertelement <4 x float> poison, float %1425, i64 0, !dbg !42
  %1582 = insertelement <4 x float> %1581, float %1431, i64 1, !dbg !42
  %1583 = insertelement <4 x float> %1582, float %1426, i64 2, !dbg !42
  %1584 = insertelement <4 x float> %1583, float %1432, i64 3, !dbg !42
  %normalized48 = fmul <4 x float> %1584, %row.reciprocal, !dbg !42
  %1585 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !43
  %1586 = extractelement <4 x half> %1585, i64 3, !dbg !43
  %1587 = extractelement <4 x half> %1585, i64 2, !dbg !43
  %1588 = extractelement <4 x half> %1585, i64 1, !dbg !43
  %1589 = extractelement <4 x half> %1585, i64 0, !dbg !43
  %1590 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !42
  %1591 = insertelement <4 x float> %1590, float %1322, i64 1, !dbg !42
  %1592 = insertelement <4 x float> %1591, float %1317, i64 2, !dbg !42
  %1593 = insertelement <4 x float> %1592, float %1323, i64 3, !dbg !42
  %normalized49 = fmul <4 x float> %1593, %row.reciprocal, !dbg !42
  %1594 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !43
  %1595 = extractelement <4 x half> %1594, i64 3, !dbg !43
  %1596 = extractelement <4 x half> %1594, i64 2, !dbg !43
  %1597 = extractelement <4 x half> %1594, i64 1, !dbg !43
  %1598 = extractelement <4 x half> %1594, i64 0, !dbg !43
  %1599 = insertelement <4 x float> poison, float %1208, i64 0, !dbg !42
  %1600 = insertelement <4 x float> %1599, float %1214, i64 1, !dbg !42
  %1601 = insertelement <4 x float> %1600, float %1209, i64 2, !dbg !42
  %1602 = insertelement <4 x float> %1601, float %1215, i64 3, !dbg !42
  %normalized51 = fmul <4 x float> %1602, %row.reciprocal38, !dbg !42
  %1603 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !43
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !43
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !43
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !43
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !43
  %1608 = insertelement <4 x float> poison, float %1099, i64 0, !dbg !42
  %1609 = insertelement <4 x float> %1608, float %1105, i64 1, !dbg !42
  %1610 = insertelement <4 x float> %1609, float %1100, i64 2, !dbg !42
  %1611 = insertelement <4 x float> %1610, float %1106, i64 3, !dbg !42
  %normalized52 = fmul <4 x float> %1611, %row.reciprocal38, !dbg !42
  %1612 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !43
  %1613 = extractelement <4 x half> %1612, i64 3, !dbg !43
  %1614 = extractelement <4 x half> %1612, i64 2, !dbg !43
  %1615 = extractelement <4 x half> %1612, i64 1, !dbg !43
  %1616 = extractelement <4 x half> %1612, i64 0, !dbg !43
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !44
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !44
  store half %1616, ptr addrspace(1) %address.local1100, align 2, !dbg !44
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !44
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !44
  store half %1615, ptr addrspace(1) %address.local1103, align 2, !dbg !44
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !44
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !44
  store half %1614, ptr addrspace(1) %address.local1106, align 2, !dbg !44
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !44
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !44
  store half %1613, ptr addrspace(1) %address.local1109, align 2, !dbg !44
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !44
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !44
  store half %1607, ptr addrspace(1) %address.local1112, align 2, !dbg !44
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !44
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !44
  store half %1606, ptr addrspace(1) %address.local1115, align 2, !dbg !44
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !44
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !44
  store half %1605, ptr addrspace(1) %address.local1118, align 2, !dbg !44
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !44
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !44
  store half %1604, ptr addrspace(1) %address.local1121, align 2, !dbg !44
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !44
  store half %1598, ptr addrspace(1) %address.local1126, align 2, !dbg !44
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !44
  store half %1597, ptr addrspace(1) %address.local1128, align 2, !dbg !44
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !44
  store half %1596, ptr addrspace(1) %address.local1130, align 2, !dbg !44
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !44
  store half %1595, ptr addrspace(1) %address.local1132, align 2, !dbg !44
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !44
  store half %1589, ptr addrspace(1) %address.local1134, align 2, !dbg !44
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !44
  store half %1588, ptr addrspace(1) %address.local1136, align 2, !dbg !44
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !44
  store half %1587, ptr addrspace(1) %address.local1138, align 2, !dbg !44
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !44
  store half %1586, ptr addrspace(1) %address.local1140, align 2, !dbg !44
  %1617 = insertelement <4 x float> poison, float %1397, i64 0, !dbg !42
  %1618 = insertelement <4 x float> %1617, float %1403, i64 1, !dbg !42
  %1619 = insertelement <4 x float> %1618, float %1398, i64 2, !dbg !42
  %1620 = insertelement <4 x float> %1619, float %1404, i64 3, !dbg !42
  %normalized54 = fmul <4 x float> %1620, %row.reciprocal, !dbg !42
  %1621 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !43
  %1622 = extractelement <4 x half> %1621, i64 3, !dbg !43
  %1623 = extractelement <4 x half> %1621, i64 2, !dbg !43
  %1624 = extractelement <4 x half> %1621, i64 1, !dbg !43
  %1625 = extractelement <4 x half> %1621, i64 0, !dbg !43
  %1626 = insertelement <4 x float> poison, float %1287, i64 0, !dbg !42
  %1627 = insertelement <4 x float> %1626, float %1293, i64 1, !dbg !42
  %1628 = insertelement <4 x float> %1627, float %1288, i64 2, !dbg !42
  %1629 = insertelement <4 x float> %1628, float %1294, i64 3, !dbg !42
  %normalized55 = fmul <4 x float> %1629, %row.reciprocal, !dbg !42
  %1630 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !43
  %1631 = extractelement <4 x half> %1630, i64 3, !dbg !43
  %1632 = extractelement <4 x half> %1630, i64 2, !dbg !43
  %1633 = extractelement <4 x half> %1630, i64 1, !dbg !43
  %1634 = extractelement <4 x half> %1630, i64 0, !dbg !43
  %1635 = insertelement <4 x float> poison, float %1180, i64 0, !dbg !42
  %1636 = insertelement <4 x float> %1635, float %1186, i64 1, !dbg !42
  %1637 = insertelement <4 x float> %1636, float %1181, i64 2, !dbg !42
  %1638 = insertelement <4 x float> %1637, float %1187, i64 3, !dbg !42
  %normalized57 = fmul <4 x float> %1638, %row.reciprocal38, !dbg !42
  %1639 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !43
  %1640 = extractelement <4 x half> %1639, i64 3, !dbg !43
  %1641 = extractelement <4 x half> %1639, i64 2, !dbg !43
  %1642 = extractelement <4 x half> %1639, i64 1, !dbg !43
  %1643 = extractelement <4 x half> %1639, i64 0, !dbg !43
  %1644 = insertelement <4 x float> poison, float %1070, i64 0, !dbg !42
  %1645 = insertelement <4 x float> %1644, float %1076, i64 1, !dbg !42
  %1646 = insertelement <4 x float> %1645, float %1071, i64 2, !dbg !42
  %1647 = insertelement <4 x float> %1646, float %1077, i64 3, !dbg !42
  %normalized58 = fmul <4 x float> %1647, %row.reciprocal38, !dbg !42
  %1648 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !43
  %1649 = extractelement <4 x half> %1648, i64 3, !dbg !43
  %1650 = extractelement <4 x half> %1648, i64 2, !dbg !43
  %1651 = extractelement <4 x half> %1648, i64 1, !dbg !43
  %1652 = extractelement <4 x half> %1648, i64 0, !dbg !43
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !44
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !44
  store half %1652, ptr addrspace(1) %address.local1156, align 2, !dbg !44
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !44
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !44
  store half %1651, ptr addrspace(1) %address.local1159, align 2, !dbg !44
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !44
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !44
  store half %1650, ptr addrspace(1) %address.local1162, align 2, !dbg !44
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !44
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !44
  store half %1649, ptr addrspace(1) %address.local1165, align 2, !dbg !44
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !44
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !44
  store half %1643, ptr addrspace(1) %address.local1168, align 2, !dbg !44
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !44
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !44
  store half %1642, ptr addrspace(1) %address.local1171, align 2, !dbg !44
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !44
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !44
  store half %1641, ptr addrspace(1) %address.local1174, align 2, !dbg !44
  %address.local1176 = or i64 %1543, 115, !dbg !44
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !44
  store half %1640, ptr addrspace(1) %address.local1177, align 2, !dbg !44
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !44
  store half %1634, ptr addrspace(1) %address.local1182, align 2, !dbg !44
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !44
  store half %1633, ptr addrspace(1) %address.local1184, align 2, !dbg !44
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !44
  store half %1632, ptr addrspace(1) %address.local1186, align 2, !dbg !44
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !44
  store half %1631, ptr addrspace(1) %address.local1188, align 2, !dbg !44
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !44
  store half %1625, ptr addrspace(1) %address.local1190, align 2, !dbg !44
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !44
  store half %1624, ptr addrspace(1) %address.local1192, align 2, !dbg !44
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !44
  store half %1623, ptr addrspace(1) %address.local1194, align 2, !dbg !44
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !44
  store half %1622, ptr addrspace(1) %address.local1196, align 2, !dbg !44
  ret void, !dbg !34
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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
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

!llvm.dbg.cu = !{!32}
!llvm.module.flags = !{!56}

!29 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/noncausal_k_dma_pass/mytest/run-20261030-163658-kcakvj_l/grid-000003")
!30 = !{}
!31 = !DISubroutineType(types: !30)
!32 = distinct !DICompileUnit(language: DW_LANG_C, file: !29, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!33 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !29, file: !29, line: 23, type: !31, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !32)
!34 = !DILocation(line: 23, column: 3, scope: !33)
!35 = !DILocation(line: 57, column: 24, scope: !33)
!36 = !DILocation(line: 67, column: 22, scope: !33)
!37 = !DILocation(line: 69, column: 24, scope: !33)
!38 = !DILocation(line: 68, column: 22, scope: !33)
!39 = !DILocation(line: 87, column: 23, scope: !33)
!40 = !DILocation(line: 70, column: 26, scope: !33)
!41 = !DILocation(line: 86, column: 20, scope: !33)
!42 = !DILocation(line: 92, column: 23, scope: !33)
!43 = !DILocation(line: 93, column: 19, scope: !33)
!44 = !DILocation(line: 94, column: 9, scope: !33)
!45 = !DILocation(line: 71, column: 23, scope: !33)
!46 = !DILocation(line: 74, column: 20, scope: !33)
!47 = !DILocation(line: 75, column: 22, scope: !33)
!48 = !DILocation(line: 76, column: 23, scope: !33)
!49 = !DILocation(line: 77, column: 20, scope: !33)
!50 = !DILocation(line: 78, column: 22, scope: !33)
!51 = !DILocation(line: 79, column: 22, scope: !33)
!52 = !DILocation(line: 80, column: 20, scope: !33)
!53 = !DILocation(line: 81, column: 16, scope: !33)
!54 = !DILocation(line: 88, column: 22, scope: !33)
!55 = !{i32 256, i32 1, i32 1}
!56 = !{i32 2, !"Debug Info Version", i32 3}
!57 = !{i32 0, i32 256}
!58 = !{i32 4}
!59 = !{i64 8192}
!60 = !{i32 0}
!61 = !{i32 1}
!62 = !{i32 2}
!63 = !{i32 3}
!64 = !{i32 5}
!65 = !{i32 6}
!66 = !{i32 7}
!67 = !{i32 8}
!68 = !{i32 9}
!69 = !{i32 10}
!70 = !{i32 11}
!71 = !{i32 12}
!72 = !{i32 13}
!73 = !{i32 14}
!74 = !{i32 15}
!75 = !{i32 16}
!76 = !{i32 17}
!77 = !{i32 18}
!78 = !{i32 19}
!79 = !{i32 20}
!80 = !{i32 21}
!81 = !{i32 22}
