; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %5 = and i32 %kdma.lane, 31, !dbg !7
  %kdma.owner = or disjoint i32 %5, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %6, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !7
  %8 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7), !dbg !7
  %9 = add nuw nsw i32 %8, 64, !dbg !7
  %10 = xor i32 %8, 16, !dbg !7
  %11 = and i32 %9, 192, !dbg !7
  %12 = icmp samesign ult i32 %10, %11, !dbg !7
  %13 = select i1 %12, i32 %10, i32 %8, !dbg !7
  %14 = shl nuw nsw i32 %13, 2, !dbg !7
  %15 = xor i32 %8, 32, !dbg !7
  %16 = icmp samesign ult i32 %15, %11, !dbg !7
  %17 = select i1 %16, i32 %15, i32 %8, !dbg !7
  %18 = shl nuw nsw i32 %17, 2, !dbg !7
  %19 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !8
  %20 = sext i32 %19 to i64, !dbg !8
  %21 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !8
  %22 = sext i32 %21 to i64, !dbg !8
  %23 = shl nsw i64 %22, 5, !dbg !8
  %24 = add nsw i64 %23, %20, !dbg !8
  %25 = lshr i64 %24, 5, !dbg !8
  %26 = and i64 %25, 576460752303423456, !dbg !8
  %27 = and i64 %20, 31, !dbg !8
  %28 = or disjoint i64 %26, %27, !dbg !8
  %29 = lshr i64 %24, 10, !dbg !9
  %30 = add nuw nsw i64 %25, %29, !dbg !9
  %31 = and i64 %30, 31, !dbg !9
  %32 = shl nuw nsw i64 %28, 7, !dbg !9
  %33 = shl i64 %29, 12, !dbg !9
  %34 = sub i64 %32, %33, !dbg !9
  %35 = shl nsw i64 %34, 7, !dbg !9
  %.idx = shl nuw nsw i64 %31, 20, !dbg !9
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !9, !range !10, !frisk.register_lifetimes !5
  %36 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %37 = and i32 %36, 1536, !dbg !9
  %address.local399 = zext nneg i32 %37 to i64, !dbg !9
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local399, !dbg !9
  %address.local403 = lshr i32 %address.thread, 4, !dbg !9
  %address.local404 = and i32 %address.local403, 8, !dbg !9
  %address.local405 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local406 = and i32 %address.local405, 16256, !dbg !9
  %address.local407 = or disjoint i32 %address.local404, %address.local406, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1373 = shl i64 %34, 8, !dbg !9
  %39 = add i64 %.idx1373, %.idx, !dbg !9
  %40 = shl nuw nsw i32 %address.local407, 1, !dbg !9
  %41 = trunc i64 %39 to i32, !dbg !9
  %dma.buffer.offset1371 = or disjoint i32 %40, %41, !dbg !9
  %42 = insertelement <2 x i64> poison, i64 %38, i64 0, !dbg !9
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1372 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local412 = or disjoint i64 %address.local399, 2048, !dbg !9
  %address.local413 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local412, !dbg !9
  %address.local421 = or disjoint i32 %address.local407, 16, !dbg !9
  %44 = shl nuw nsw i32 %address.local421, 1, !dbg !9
  %dma.buffer.offset1369 = or disjoint i32 %44, %41, !dbg !9
  %45 = shl nuw nsw i64 %31, 19, !dbg !9
  %46 = add i64 %35, %45, !dbg !9
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local399, !dbg !9
  %47 = trunc i64 %46 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local407, %47, !dbg !9
  %48 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1367 = or disjoint i32 %48, 64, !dbg !9
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local412, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local400, i64 0, !dbg !9
  %50 = insertelement <4 x ptr addrspace(3)> %49, ptr addrspace(3) %address.local413, i64 1, !dbg !9
  %51 = insertelement <4 x ptr addrspace(3)> %50, ptr addrspace(3) %address.local424, i64 2, !dbg !9
  %52 = insertelement <4 x ptr addrspace(3)> %51, ptr addrspace(3) %address.local428, i64 3, !dbg !9
  %53 = ptrtoint <4 x ptr addrspace(3)> %52 to <4 x i64>, !dbg !9
  %54 = trunc <4 x i64> %53 to <4 x i32>, !dbg !9
  %55 = extractelement <4 x i32> %54, i64 0, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1371, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %58 = extractelement <4 x i32> %54, i64 1, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %61 = extractelement <4 x i32> %54, i64 2, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1367, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %64 = extractelement <4 x i32> %54, i64 3, !dbg !9
  %65 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !9
  %.tr1377 = or disjoint i32 %address.local421, %47, !dbg !9
  %66 = shl i32 %.tr1377, 1, !dbg !9
  %dma.buffer.offset1365 = or disjoint i32 %66, 64, !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1365, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %address.local432 = lshr i32 %address.thread, 2, !dbg !11
  %address.local433 = and i32 %address.local432, 4, !dbg !11
  %address.local434 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local435 = and i32 %address.local434, 384, !dbg !11
  %address.local436 = or disjoint i32 %address.local433, %address.local435, !dbg !11
  %address.local438 = and i32 %36, 120, !dbg !11
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !11
  %address.local440 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local441 = and i32 %address.local440, 1024, !dbg !11
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !11
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !11
  %address.local444 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local443, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local444, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local444, i64 4096, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1194 = getelementptr i8, ptr addrspace(3) %address.local444, i64 1024, !dbg !11
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !11
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1195 = getelementptr i8, ptr addrspace(3) %address.local444, i64 5120, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !11
  %75 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1363 = or disjoint i32 %48, 128, !dbg !9
  %76 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %dma.buffer.offset1361 = or disjoint i32 %66, 128, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1361, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %lds.const.gep1196 = getelementptr i8, ptr addrspace(3) %address.local444, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !11
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local444, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !11
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local444, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local444, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1359 = or disjoint i32 %48, 192, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1359, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %dma.buffer.offset1357 = or disjoint i32 %66, 192, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1372) #9, !dbg !9
  %88 = load <4 x half>, ptr addrspace(3) %address.local444, align 8, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !11
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !11
  %101 = fmul <4 x half> %97, splat (half 1.275630e-01), !dbg !11
  %102 = fmul <4 x half> %98, splat (half 1.275630e-01), !dbg !11
  %103 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %104 = shl nuw nsw i32 %kdma.tid, 2, !dbg !13
  %105 = and i32 %104, 3084, !dbg !13
  %106 = and i32 %kdma.tid, 1, !dbg !13
  %107 = add nuw nsw i32 %106, 1, !dbg !13
  %narrow = or disjoint i32 %107, %105, !dbg !13
  %108 = lshr i32 %narrow, 1, !dbg !13
  %109 = zext nneg i32 %108 to i64, !dbg !13
  %k0.prime. = shl i64 %25, 19, !dbg !14
  %k0.prime.2040 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1479 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1480 = shl nuw nsw i64 %k0.prime.kdma.source1479, 9, !dbg !14
  %k0.prime.kdma.source1481 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1482 = add nuw nsw i64 %k0.prime.kdma.source1480, %k0.prime.kdma.source1481, !dbg !14
  %k0.prime.kdma.source1483 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1484 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1485 = and i64 %k0.prime.kdma.source1484, 1, !dbg !14
  %k0.prime.kdma.source1486 = and i64 %k0.prime.kdma.source1485, %k0.prime.kdma.source1479, !dbg !14
  %k0.prime.kdma.source1487 = or disjoint i64 %k0.prime.kdma.source1486, %k0.prime.kdma.source1483, !dbg !14
  %110 = shl nuw nsw i64 %k0.prime.kdma.source1487, 10, !dbg !14
  %k0.prime.kdma.source1489 = sub nsw i64 %k0.prime.kdma.source1482, %110, !dbg !14
  %k0.prime.kdma.source1490 = and i64 %k0.prime.kdma.source1489, 16256, !dbg !14
  %k0.prime.kdma.source1491 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1492 = and i32 %k0.prime.kdma.source1491, 8, !dbg !14
  %k0.prime.kdma.source1493 = zext nneg i32 %k0.prime.kdma.source1492 to i64, !dbg !14
  %k0.prime.kdma.source1494 = or disjoint i64 %k0.prime.kdma.source1490, %k0.prime.kdma.source1493, !dbg !14
  %k0.prime.kdma.source1495 = or disjoint i64 %k0.prime.kdma.source1494, %k0.prime.2040, !dbg !14
  %k0.prime.kdma.source1496 = trunc nuw nsw i64 %k0.prime.kdma.source1495 to i32, !dbg !14
  %k0.prime.kdma.source1497 = shl nuw nsw i32 %k0.prime.kdma.source1496, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1497, %kdma.intra, !dbg !14
  %k0.prime.2043 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.2044 = insertelement <2 x i64> poison, i64 %k0.prime.2043, i64 0, !dbg !14
  %k0.prime.2045 = bitcast <2 x i64> %k0.prime.2044 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1356 = shufflevector <4 x i32> %k0.prime.2045, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.2046 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1519 = add i32 %56, 512, !dbg !14
  %k0.prime.kdma.source1498 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1499 = lshr i64 %k0.prime.kdma.source1498, 4, !dbg !14
  %k0.prime.kdma.source1500 = shl nuw nsw i64 %k0.prime.kdma.source1499, 9, !dbg !14
  %k0.prime.kdma.source1501 = shl nuw nsw i64 %k0.prime.kdma.source1498, 7, !dbg !14
  %k0.prime.kdma.source1502 = add nuw nsw i64 %k0.prime.kdma.source1500, %k0.prime.kdma.source1501, !dbg !14
  %k0.prime.kdma.source1503 = lshr i64 %k0.prime.kdma.source1498, 5, !dbg !14
  %k0.prime.kdma.source1504 = lshr i64 %k0.prime.kdma.source1498, 2, !dbg !14
  %k0.prime.kdma.source1505 = and i64 %k0.prime.kdma.source1504, 1, !dbg !14
  %k0.prime.kdma.source1506 = and i64 %k0.prime.kdma.source1505, %k0.prime.kdma.source1499, !dbg !14
  %k0.prime.kdma.source1507 = add nuw nsw i64 %k0.prime.kdma.source1506, %k0.prime.kdma.source1503, !dbg !14
  %111 = shl nuw nsw i64 %k0.prime.kdma.source1507, 10, !dbg !14
  %k0.prime.kdma.source1509 = sub nsw i64 %k0.prime.kdma.source1502, %111, !dbg !14
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 16256, !dbg !14
  %k0.prime.kdma.source1514 = or disjoint i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source1493, !dbg !14
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1514, %k0.prime.2040, !dbg !14
  %k0.prime.kdma.source1516 = trunc nuw nsw i64 %k0.prime.kdma.source1515 to i32, !dbg !14
  %k0.prime.kdma.source1517 = shl nuw nsw i32 %k0.prime.kdma.source1516, 1, !dbg !14
  %k0.prime.kdma.offset1518 = or disjoint i32 %k0.prime.kdma.source1517, %kdma.intra.hi, !dbg !14
  %k0.prime.2047 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1519, i32 %k0.prime.kdma.offset1518, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1521 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1522 = lshr i64 %k0.prime.kdma.source1521, 4, !dbg !14
  %k0.prime.kdma.source1523 = shl nuw nsw i64 %k0.prime.kdma.source1522, 9, !dbg !14
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1523, %k0.prime.kdma.source1481, !dbg !14
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source1521, 5, !dbg !14
  %k0.prime.kdma.source1529 = and i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1485, !dbg !14
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.kdma.source1529, %k0.prime.kdma.source1526, !dbg !14
  %112 = shl nuw nsw i64 %k0.prime.kdma.source1530, 10, !dbg !14
  %k0.prime.kdma.source1532 = sub nsw i64 %k0.prime.kdma.source1525, %112, !dbg !14
  %k0.prime.kdma.source1533 = and i64 %k0.prime.kdma.source1532, 16256, !dbg !14
  %k0.prime.kdma.source1536 = or disjoint i32 %k0.prime.kdma.source1492, 16, !dbg !14
  %k0.prime.kdma.source1537 = zext nneg i32 %k0.prime.kdma.source1536 to i64, !dbg !14
  %k0.prime.kdma.source1538 = or disjoint i64 %k0.prime.kdma.source1533, %k0.prime.kdma.source1537, !dbg !14
  %k0.prime.kdma.source1539 = or disjoint i64 %k0.prime.kdma.source1538, %k0.prime.2040, !dbg !14
  %k0.prime.kdma.source1540 = trunc nuw nsw i64 %k0.prime.kdma.source1539 to i32, !dbg !14
  %k0.prime.kdma.source1541 = shl nuw nsw i32 %k0.prime.kdma.source1540, 1, !dbg !14
  %k0.prime.kdma.offset1542 = add nuw nsw i32 %k0.prime.kdma.source1541, %kdma.intra, !dbg !14
  %k0.prime.2048 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.prime.kdma.offset1542, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1567 = add i32 %59, 512, !dbg !14
  %k0.prime.kdma.source1545 = or disjoint i64 %k0.prime.kdma.source1498, 256, !dbg !14
  %k0.prime.kdma.source1546 = lshr i64 %k0.prime.kdma.source1545, 4, !dbg !14
  %k0.prime.kdma.source1547 = shl nuw nsw i64 %k0.prime.kdma.source1546, 9, !dbg !14
  %k0.prime.kdma.source1549 = add nuw nsw i64 %k0.prime.kdma.source1547, %k0.prime.kdma.source1501, !dbg !14
  %k0.prime.kdma.source1550 = lshr i64 %k0.prime.kdma.source1545, 5, !dbg !14
  %k0.prime.kdma.source1553 = and i64 %k0.prime.kdma.source1546, %k0.prime.kdma.source1505, !dbg !14
  %k0.prime.kdma.source1554 = add nuw nsw i64 %k0.prime.kdma.source1553, %k0.prime.kdma.source1550, !dbg !14
  %113 = shl nuw nsw i64 %k0.prime.kdma.source1554, 10, !dbg !14
  %k0.prime.kdma.source1556 = sub nsw i64 %k0.prime.kdma.source1549, %113, !dbg !14
  %k0.prime.kdma.source1557 = and i64 %k0.prime.kdma.source1556, 16256, !dbg !14
  %k0.prime.kdma.source1562 = or disjoint i64 %k0.prime.kdma.source1557, %k0.prime.kdma.source1537, !dbg !14
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.kdma.source1562, %k0.prime.2040, !dbg !14
  %k0.prime.kdma.source1564 = trunc nuw nsw i64 %k0.prime.kdma.source1563 to i32, !dbg !14
  %k0.prime.kdma.source1565 = shl nuw nsw i32 %k0.prime.kdma.source1564, 1, !dbg !14
  %k0.prime.kdma.offset1566 = add nuw nsw i32 %k0.prime.kdma.source1565, %kdma.intra.hi, !dbg !14
  %k0.prime.2049 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1567, i32 %k0.prime.kdma.offset1566, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1611 = add i32 %62, 512
  %kdma.lds1659 = add i32 %65, 512
  %114 = ptrtoint ptr addrspace(1) %2 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1340 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %56, 8192
  %vdb.dma.base1433 = add i32 %59, 8192
  br label %117, !dbg !12

117:                                              ; preds = %4, %k0.join
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %904, %k0.join ]
  %119 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %900, %k0.join ]
  %120 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %896, %k0.join ]
  %121 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %lazy.base.join, %k0.join ]
  %122 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %669, %k0.join ]
  %123 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %892, %k0.join ]
  %124 = phi i64 [ 0, %4 ], [ %824, %k0.join ]
  %125 = shl nuw nsw i64 %124, 14, !dbg !14
  %126 = add nuw nsw i64 %125, %k0.prime.2040, !dbg !14
  %address.thread495 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local496 = zext nneg i32 %address.thread495 to i64, !dbg !9
  %address.local502 = lshr i64 %address.local496, 2, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %127 = or disjoint i64 %126, 32, !dbg !14
  %kdma.source1585 = or disjoint i64 %127, %k0.prime.kdma.source1494, !dbg !14
  %kdma.source1586 = trunc nuw nsw i64 %kdma.source1585 to i32, !dbg !14
  %kdma.source1587 = shl nuw nsw i32 %kdma.source1586, 1, !dbg !14
  %kdma.offset1588 = or disjoint i32 %kdma.source1587, %kdma.intra, !dbg !14
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1607 = or disjoint i64 %127, %k0.prime.kdma.source1514, !dbg !14
  %kdma.source1608 = trunc nuw nsw i64 %kdma.source1607 to i32, !dbg !14
  %kdma.source1609 = shl nuw nsw i32 %kdma.source1608, 1, !dbg !14
  %kdma.offset1610 = or disjoint i32 %kdma.source1609, %kdma.intra.hi, !dbg !14
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1611, i32 %kdma.offset1610, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1631 = or disjoint i64 %127, %k0.prime.kdma.source1538, !dbg !14
  %kdma.source1632 = trunc nuw nsw i64 %kdma.source1631 to i32, !dbg !14
  %kdma.source1633 = shl nuw nsw i32 %kdma.source1632, 1, !dbg !14
  %kdma.offset1634 = add nuw nsw i32 %kdma.source1633, %kdma.intra, !dbg !14
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1634, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1655 = or disjoint i64 %127, %k0.prime.kdma.source1562, !dbg !14
  %kdma.source1656 = trunc nuw nsw i64 %kdma.source1655 to i32, !dbg !14
  %kdma.source1657 = shl nuw nsw i32 %kdma.source1656, 1, !dbg !14
  %kdma.offset1658 = add nuw nsw i32 %kdma.source1657, %kdma.intra.hi, !dbg !14
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1659, i32 %kdma.offset1658, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %address.local539 = shl nuw nsw i64 %address.local496, 1, !dbg !13
  %address.local540 = and i64 %address.local539, 24, !dbg !13
  %address.local542 = shl i64 %address.local496, 62, !dbg !13
  %address.local543 = ashr exact i64 %address.local542, 57, !dbg !13
  %address.local544 = and i64 %address.local543, -64, !dbg !13
  %address.local547 = and i64 %address.local502, 12, !dbg !11
  %address.local549 = and i64 %address.local496, 63, !dbg !11
  %address.local550 = lshr i64 %address.local549, 5, !dbg !11
  %address.local551 = mul nuw nsw i64 %address.local550, 1016, !dbg !11
  %.idx1394 = mul nuw nsw i64 %address.local496, 320, !dbg !13
  %132 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1394, !dbg !13
  %.idx1395 = mul nsw i64 %address.local502, -1280, !dbg !13
  %133 = getelementptr i8, ptr addrspace(3) %132, i64 %.idx1395, !dbg !13
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local540, !dbg !13
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local544, !dbg !13
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local547, !dbg !13
  %address.local553 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local551, !dbg !13
  %137 = add nsw i64 %.idx1395, %.idx1394, !dbg !13
  %.idx1970 = shl nuw nsw i64 %address.local540, 1, !dbg !13
  %138 = or disjoint i64 %137, %.idx1970, !dbg !13
  %139 = or disjoint i64 %address.local544, %address.local547, !dbg !13
  %140 = shl nsw i64 %139, 1, !dbg !13
  %address.local553.idx = mul nuw nsw i64 %address.local550, 2032, !dbg !13
  %141 = add nsw i64 %138, %address.local553.idx, !dbg !13
  %142 = add nsw i64 %141, %140, !dbg !13
  %143 = lshr i64 %142, 6, !dbg !13
  %144 = and i64 %143, 8, !dbg !13
  %145 = xor i64 %144, %142, !dbg !13
  %146 = sub i64 %145, %142, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local553, i64 %146, !dbg !13
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %121, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base2055 = extractvalue [2 x <1 x float>] %121, 1, !dbg !13
  %lazy.seed2056 = shufflevector <1 x float> %lazy.base2055, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %147, <4 x float> %lazy.seed) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local564 = shl nuw nsw i32 %address.thread495, 1, !dbg !13
  %address.local565 = and i32 %address.local564, 24, !dbg !13
  %address.local566 = shl nuw nsw i32 %address.thread495, 8, !dbg !13
  %address.local567 = and i32 %address.local566, 768, !dbg !13
  %address.local568 = or disjoint i32 %address.local565, %address.local567, !dbg !13
  %address.local569 = shl nuw nsw i32 %address.thread495, 5, !dbg !13
  %address.local570 = and i32 %address.local569, 96, !dbg !13
  %address.local571 = or disjoint i32 %address.local568, %address.local570, !dbg !13
  %address.local572 = zext nneg i32 %address.local571 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %109) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local573 = lshr i32 %address.thread495, 1, !dbg !13
  %address.local574 = and i32 %address.local573, 1, !dbg !13
  %address.local575 = zext nneg i32 %address.local574 to i64, !dbg !13
  %address.local576 = add nsw i64 %address.base, %address.local575, !dbg !13
  %address.local577 = shl i64 %address.local576, 6, !dbg !13
  %address.local578 = sub i64 %address.local572, %address.local577, !dbg !13
  %150 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local578, !dbg !13
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local547, !dbg !13
  %152 = getelementptr [2 x i8], ptr addrspace(3) %151, i64 %address.local551, !dbg !13
  %address.local582 = getelementptr i8, ptr addrspace(3) %152, i64 64, !dbg !13
  %.idx19731976 = add i64 %address.local578, %address.local547, !dbg !13
  %153 = shl i64 %.idx19731976, 1, !dbg !13
  %154 = add nuw nsw i64 %address.local553.idx, 64, !dbg !13
  %155 = add i64 %154, %153, !dbg !13
  %156 = lshr i64 %155, 6, !dbg !13
  %157 = and i64 %156, 8, !dbg !13
  %158 = xor i64 %157, %155, !dbg !13
  %159 = sub i64 %158, %155, !dbg !13
  %kdma.swizzled1844 = getelementptr i8, ptr addrspace(3) %address.local582, i64 %159, !dbg !13
  %160 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1844, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 128, !dbg !13
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 4096, !dbg !13
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 4224, !dbg !13
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %167 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %148) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 1024, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 1152, !dbg !13
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %160, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 5120, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 5248, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %171) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %176 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !13, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %164, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %166, <4 x float> %177) #8, !dbg !13
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %161, <4 x float> %lazy.seed) #8, !dbg !13
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %163, <4 x float> %179) #8, !dbg !13
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %165, <4 x float> %lazy.seed) #8, !dbg !13
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %167, <4 x float> %181) #8, !dbg !13
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %169, <4 x float> %lazy.seed) #8, !dbg !13
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %172, <4 x float> %183) #8, !dbg !13
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %175, <4 x float> %lazy.seed) #8, !dbg !13
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %176, <4 x float> %185) #8, !dbg !13
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %170, <4 x float> %lazy.seed) #8, !dbg !13
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %173, <4 x float> %187) #8, !dbg !13
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %147, <4 x float> %lazy.seed2056) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %149, <4 x float> %189) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %160, <4 x float> %lazy.seed2056) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %162, <4 x float> %191) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %164, <4 x float> %lazy.seed2056) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %166, <4 x float> %193) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %161, <4 x float> %lazy.seed2056) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %163, <4 x float> %195) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %165, <4 x float> %lazy.seed2056) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %167, <4 x float> %197) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %169, <4 x float> %lazy.seed2056) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %172, <4 x float> %199) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %175, <4 x float> %lazy.seed2056) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %176, <4 x float> %201) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %170, <4 x float> %lazy.seed2056) #8, !dbg !13
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %173, <4 x float> %203) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %205 = or disjoint i64 %126, 64, !dbg !14
  %kdma.source1677 = or disjoint i64 %205, %k0.prime.kdma.source1494, !dbg !14
  %kdma.source1678 = trunc nuw nsw i64 %kdma.source1677 to i32, !dbg !14
  %kdma.source1679 = shl nuw nsw i32 %kdma.source1678, 1, !dbg !14
  %kdma.offset1680 = or disjoint i32 %kdma.source1679, %kdma.intra, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1680, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1699 = or disjoint i64 %205, %k0.prime.kdma.source1514, !dbg !14
  %kdma.source1700 = trunc nuw nsw i64 %kdma.source1699 to i32, !dbg !14
  %kdma.source1701 = shl nuw nsw i32 %kdma.source1700, 1, !dbg !14
  %kdma.offset1702 = or disjoint i32 %kdma.source1701, %kdma.intra.hi, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1519, i32 %kdma.offset1702, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1723 = or disjoint i64 %205, %k0.prime.kdma.source1538, !dbg !14
  %kdma.source1724 = trunc nuw nsw i64 %kdma.source1723 to i32, !dbg !14
  %kdma.source1725 = shl nuw nsw i32 %kdma.source1724, 1, !dbg !14
  %kdma.offset1726 = add nuw nsw i32 %kdma.source1725, %kdma.intra, !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1726, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1747 = or disjoint i64 %205, %k0.prime.kdma.source1562, !dbg !14
  %kdma.source1748 = trunc nuw nsw i64 %kdma.source1747 to i32, !dbg !14
  %kdma.source1749 = shl nuw nsw i32 %kdma.source1748, 1, !dbg !14
  %kdma.offset1750 = add nuw nsw i32 %kdma.source1749, %kdma.intra.hi, !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1567, i32 %kdma.offset1750, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 8192, !dbg !13
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 8320, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 12288, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 12416, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %210, <4 x float> %168) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %212, <4 x float> %220) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 9216, !dbg !13
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 9344, !dbg !13
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %214, <4 x float> %174) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 13312, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 13440, !dbg !13
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !13, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %216, <4 x float> %226) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %211, <4 x float> %178) #8, !dbg !13
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %213, <4 x float> %230) #8, !dbg !13
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %215, <4 x float> %180) #8, !dbg !13
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %217, <4 x float> %232) #8, !dbg !13
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %218, <4 x float> %182) #8, !dbg !13
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %221, <4 x float> %234) #8, !dbg !13
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %224, <4 x float> %184) #8, !dbg !13
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %227, <4 x float> %236) #8, !dbg !13
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %219, <4 x float> %186) #8, !dbg !13
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %222, <4 x float> %238) #8, !dbg !13
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %225, <4 x float> %188) #8, !dbg !13
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %228, <4 x float> %240) #8, !dbg !13
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %210, <4 x float> %190) #8, !dbg !13
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %212, <4 x float> %242) #8, !dbg !13
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %214, <4 x float> %192) #8, !dbg !13
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %216, <4 x float> %244) #8, !dbg !13
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %211, <4 x float> %194) #8, !dbg !13
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %213, <4 x float> %246) #8, !dbg !13
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %215, <4 x float> %196) #8, !dbg !13
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %217, <4 x float> %248) #8, !dbg !13
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %218, <4 x float> %198) #8, !dbg !13
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %221, <4 x float> %250) #8, !dbg !13
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %224, <4 x float> %200) #8, !dbg !13
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %227, <4 x float> %252) #8, !dbg !13
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %219, <4 x float> %202) #8, !dbg !13
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %222, <4 x float> %254) #8, !dbg !13
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %225, <4 x float> %204) #8, !dbg !13
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %228, <4 x float> %256) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %258 = or disjoint i64 %126, 96, !dbg !14
  %kdma.source1769 = or disjoint i64 %258, %k0.prime.kdma.source1494, !dbg !14
  %kdma.source1770 = trunc nuw nsw i64 %kdma.source1769 to i32, !dbg !14
  %kdma.source1771 = shl nuw nsw i32 %kdma.source1770, 1, !dbg !14
  %kdma.offset1772 = or disjoint i32 %kdma.source1771, %kdma.intra, !dbg !14
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1772, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1791 = or disjoint i64 %258, %k0.prime.kdma.source1514, !dbg !14
  %kdma.source1792 = trunc nuw nsw i64 %kdma.source1791 to i32, !dbg !14
  %kdma.source1793 = shl nuw nsw i32 %kdma.source1792, 1, !dbg !14
  %kdma.offset1794 = or disjoint i32 %kdma.source1793, %kdma.intra.hi, !dbg !14
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1611, i32 %kdma.offset1794, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1815 = or disjoint i64 %258, %k0.prime.kdma.source1538, !dbg !14
  %kdma.source1816 = trunc nuw nsw i64 %kdma.source1815 to i32, !dbg !14
  %kdma.source1817 = shl nuw nsw i32 %kdma.source1816, 1, !dbg !14
  %kdma.offset1818 = add nuw nsw i32 %kdma.source1817, %kdma.intra, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1818, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %kdma.source1839 = or disjoint i64 %258, %k0.prime.kdma.source1562, !dbg !14
  %kdma.source1840 = trunc nuw nsw i64 %kdma.source1839 to i32, !dbg !14
  %kdma.source1841 = shl nuw nsw i32 %kdma.source1840, 1, !dbg !14
  %kdma.offset1842 = add nuw nsw i32 %kdma.source1841, %kdma.intra.hi, !dbg !14
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1659, i32 %kdma.offset1842, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !13, !frisk.k_lds_pair !30
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !13, !frisk.k_lds_pair !31
  %267 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1844, align 8, !dbg !13, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !13, !frisk.k_lds_pair !32
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !13, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !13, !frisk.k_lds_pair !33
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !13, !frisk.k_lds_pair !34
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %263, <4 x float> %223) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !13, !frisk.k_lds_pair !35
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %265, <4 x float> %273) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !13, !frisk.k_lds_pair !36
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %267, <4 x float> %229) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !13, !frisk.k_lds_pair !37
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !13, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %269, <4 x float> %279) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %264, <4 x float> %231) #8, !dbg !13
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %266, <4 x float> %283) #8, !dbg !13
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %268, <4 x float> %233) #8, !dbg !13
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %270, <4 x float> %285) #8, !dbg !13
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %271, <4 x float> %235) #8, !dbg !13
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %274, <4 x float> %287) #8, !dbg !13
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %277, <4 x float> %237) #8, !dbg !13
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %280, <4 x float> %289) #8, !dbg !13
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %272, <4 x float> %239) #8, !dbg !13
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %275, <4 x float> %291) #8, !dbg !13
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %278, <4 x float> %241) #8, !dbg !13
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %281, <4 x float> %293) #8, !dbg !13
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %263, <4 x float> %243) #8, !dbg !13
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %265, <4 x float> %295) #8, !dbg !13
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %267, <4 x float> %245) #8, !dbg !13
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %269, <4 x float> %297) #8, !dbg !13
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %264, <4 x float> %247) #8, !dbg !13
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %266, <4 x float> %299) #8, !dbg !13
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %268, <4 x float> %249) #8, !dbg !13
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %270, <4 x float> %301) #8, !dbg !13
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %271, <4 x float> %251) #8, !dbg !13
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %274, <4 x float> %303) #8, !dbg !13
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %277, <4 x float> %253) #8, !dbg !13
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %280, <4 x float> %305) #8, !dbg !13
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %272, <4 x float> %255) #8, !dbg !13
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %275, <4 x float> %307) #8, !dbg !13
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %278, <4 x float> %257) #8, !dbg !13
  %310 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %281, <4 x float> %309) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local911 = shl nuw nsw i64 %address.local496, 5, !dbg !38
  %address.local912 = and i64 %address.local911, 8064, !dbg !38
  %address.local913 = shl nuw nsw i64 %address.local496, 3, !dbg !9
  %address.local914 = shl nuw nsw i64 %address.local502, 5, !dbg !38
  %address.local915 = sub nsw i64 %address.local913, %address.local914, !dbg !38
  %address.local916 = add nsw i64 %address.local915, %address.local912, !dbg !38
  %.idx14041405 = add nsw i64 %address.local916, %126, !dbg !38
  %.idx14041405.tr = trunc nsw i64 %.idx14041405 to i32, !dbg !38
  %dma.buffer.offset1339 = shl nsw i32 %.idx14041405.tr, 1, !dbg !38
  %311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %address.local918 = shl nuw nsw i32 %address.thread495, 3, !dbg !38
  %address.local919 = and i32 %address.local918, 24, !dbg !38
  %address.local921 = and i32 %address.local569, 8064, !dbg !38
  %address.local922 = or disjoint i32 %address.local921, %address.local919, !dbg !38
  %address.local923 = or disjoint i32 %address.local922, 8192, !dbg !38
  %312 = trunc nuw nsw i64 %126 to i32, !dbg !38
  %.idx14061407.tr = or disjoint i32 %address.local923, %312, !dbg !38
  %dma.buffer.offset1337 = shl nuw nsw i32 %.idx14061407.tr, 1, !dbg !38
  %313 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1337, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_tail_pair !15
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_tail_pair !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_tail_pair !16
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !13, !frisk.k_tail_pair !16
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_tail_pair !17
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !13, !frisk.k_tail_pair !17
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_tail_pair !18
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !13, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %314, <4 x float> %276) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %316, <4 x float> %322) #8, !dbg !13
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %318, <4 x float> %282) #8, !dbg !13
  %325 = extractelement <4 x float> %323, i64 0, !dbg !13
  %326 = tail call float @llvm.maxnum.f32(float %325, float -inf), !dbg !7, !frisk.local_reduction !5
  %327 = extractelement <4 x float> %323, i64 1, !dbg !13
  %328 = extractelement <4 x float> %323, i64 2, !dbg !13
  %329 = extractelement <4 x float> %323, i64 3, !dbg !13
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %320, <4 x float> %324) #8, !dbg !13
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !13, !frisk.k_tail_pair !19
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !13, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %315, <4 x float> %284) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %334 = extractelement <4 x float> %330, i64 0, !dbg !13
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %326, float %334), !dbg !7, !frisk.local_reduction !5
  %335 = extractelement <4 x float> %330, i64 1, !dbg !13
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %327, float %335), !dbg !7, !frisk.local_reduction !5
  %336 = extractelement <4 x float> %330, i64 2, !dbg !13
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %328, float %336), !dbg !7, !frisk.local_reduction !5
  %337 = extractelement <4 x float> %330, i64 3, !dbg !13
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %329, float %337), !dbg !7, !frisk.local_reduction !5
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !13, !frisk.k_tail_pair !20
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %317, <4 x float> %333) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !13, !frisk.k_tail_pair !21
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %319, <4 x float> %286) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %344 = extractelement <4 x float> %340, i64 0, !dbg !13
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %344), !dbg !7, !frisk.local_reduction !5
  %345 = extractelement <4 x float> %340, i64 1, !dbg !13
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %345), !dbg !7, !frisk.local_reduction !5
  %346 = extractelement <4 x float> %340, i64 2, !dbg !13
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %346), !dbg !7, !frisk.local_reduction !5
  %347 = extractelement <4 x float> %340, i64 3, !dbg !13
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %347), !dbg !7, !frisk.local_reduction !5
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !13, !frisk.k_tail_pair !22
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %321, <4 x float> %343) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %331, <4 x float> %288) #8, !dbg !13
  %352 = extractelement <4 x float> %350, i64 0, !dbg !13
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %352), !dbg !7, !frisk.local_reduction !5
  %353 = extractelement <4 x float> %350, i64 1, !dbg !13
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %353), !dbg !7, !frisk.local_reduction !5
  %354 = extractelement <4 x float> %350, i64 2, !dbg !13
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %354), !dbg !7, !frisk.local_reduction !5
  %355 = extractelement <4 x float> %350, i64 3, !dbg !13
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %355), !dbg !7, !frisk.local_reduction !5
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %338, <4 x float> %351) #8, !dbg !13
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %341, <4 x float> %290) #8, !dbg !13
  %358 = extractelement <4 x float> %356, i64 0, !dbg !13
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %358), !dbg !7, !frisk.local_reduction !5
  %359 = extractelement <4 x float> %356, i64 1, !dbg !13
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %359), !dbg !7, !frisk.local_reduction !5
  %360 = extractelement <4 x float> %356, i64 2, !dbg !13
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %360), !dbg !7, !frisk.local_reduction !5
  %361 = extractelement <4 x float> %356, i64 3, !dbg !13
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %361), !dbg !7, !frisk.local_reduction !5
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %348, <4 x float> %357) #8, !dbg !13
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %332, <4 x float> %292) #8, !dbg !13
  %364 = extractelement <4 x float> %362, i64 0, !dbg !13
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %364), !dbg !7, !frisk.local_reduction !5
  %365 = extractelement <4 x float> %362, i64 1, !dbg !13
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %365), !dbg !7, !frisk.local_reduction !5
  %366 = extractelement <4 x float> %362, i64 2, !dbg !13
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %366), !dbg !7, !frisk.local_reduction !5
  %367 = extractelement <4 x float> %362, i64 3, !dbg !13
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %367), !dbg !7, !frisk.local_reduction !5
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %339, <4 x float> %363) #8, !dbg !13
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %342, <4 x float> %294) #8, !dbg !13
  %370 = extractelement <4 x float> %368, i64 0, !dbg !13
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %local.reduce149, float %370), !dbg !7, !frisk.local_reduction !5
  %371 = extractelement <4 x float> %368, i64 1, !dbg !13
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %local.reduce150, float %371), !dbg !7, !frisk.local_reduction !5
  %372 = extractelement <4 x float> %368, i64 2, !dbg !13
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %372), !dbg !7, !frisk.local_reduction !5
  %373 = extractelement <4 x float> %368, i64 3, !dbg !13
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %373), !dbg !7, !frisk.local_reduction !5
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %349, <4 x float> %369) #8, !dbg !13
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %314, <4 x float> %296) #8, !dbg !13
  %376 = extractelement <4 x float> %374, i64 0, !dbg !13
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %376), !dbg !7, !frisk.local_reduction !5
  %377 = extractelement <4 x float> %374, i64 1, !dbg !13
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %377), !dbg !7, !frisk.local_reduction !5
  %378 = extractelement <4 x float> %374, i64 2, !dbg !13
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %378), !dbg !7, !frisk.local_reduction !5
  %379 = extractelement <4 x float> %374, i64 3, !dbg !13
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %379), !dbg !7, !frisk.local_reduction !5
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %local.reduce158), !dbg !7, !frisk.local_reduction !5
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %local.reduce160), !dbg !7, !frisk.local_reduction !5
  %380 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %local.reduce162), !dbg !7, !frisk.local_reduction !5
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %316, <4 x float> %375) #8, !dbg !13
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %318, <4 x float> %298) #8, !dbg !13
  %383 = extractelement <4 x float> %381, i64 0, !dbg !13
  %384 = extractelement <4 x float> %381, i64 1, !dbg !13
  %385 = extractelement <4 x float> %381, i64 2, !dbg !13
  %386 = extractelement <4 x float> %381, i64 3, !dbg !13
  %387 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %320, <4 x float> %382) #8, !dbg !13
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %315, <4 x float> %300) #8, !dbg !13
  %389 = extractelement <4 x float> %387, i64 0, !dbg !13
  %390 = extractelement <4 x float> %387, i64 1, !dbg !13
  %391 = extractelement <4 x float> %387, i64 2, !dbg !13
  %392 = extractelement <4 x float> %387, i64 3, !dbg !13
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %317, <4 x float> %388) #8, !dbg !13
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %319, <4 x float> %302) #8, !dbg !13
  %395 = extractelement <4 x float> %393, i64 0, !dbg !13
  %396 = extractelement <4 x float> %393, i64 1, !dbg !13
  %397 = extractelement <4 x float> %393, i64 2, !dbg !13
  %398 = extractelement <4 x float> %393, i64 3, !dbg !13
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %321, <4 x float> %394) #8, !dbg !13
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %331, <4 x float> %304) #8, !dbg !13
  %401 = extractelement <4 x float> %399, i64 0, !dbg !13
  %402 = extractelement <4 x float> %399, i64 1, !dbg !13
  %403 = extractelement <4 x float> %399, i64 2, !dbg !13
  %404 = extractelement <4 x float> %399, i64 3, !dbg !13
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %338, <4 x float> %400) #8, !dbg !13
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %341, <4 x float> %306) #8, !dbg !13
  %407 = extractelement <4 x float> %405, i64 0, !dbg !13
  %408 = extractelement <4 x float> %405, i64 1, !dbg !13
  %409 = extractelement <4 x float> %405, i64 2, !dbg !13
  %410 = extractelement <4 x float> %405, i64 3, !dbg !13
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %348, <4 x float> %406) #8, !dbg !13
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %332, <4 x float> %308) #8, !dbg !13
  %413 = extractelement <4 x float> %411, i64 0, !dbg !13
  %414 = extractelement <4 x float> %411, i64 1, !dbg !13
  %415 = extractelement <4 x float> %411, i64 2, !dbg !13
  %416 = extractelement <4 x float> %411, i64 3, !dbg !13
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %339, <4 x float> %412) #8, !dbg !13
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %342, <4 x float> %310) #8, !dbg !13
  %419 = extractelement <4 x float> %417, i64 0, !dbg !13
  %420 = extractelement <4 x float> %417, i64 1, !dbg !13
  %421 = extractelement <4 x float> %417, i64 2, !dbg !13
  %422 = extractelement <4 x float> %417, i64 3, !dbg !13
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %349, <4 x float> %418) #8, !dbg !13
  %424 = extractelement <4 x float> %423, i64 0, !dbg !13
  %425 = extractelement <4 x float> %423, i64 1, !dbg !13
  %426 = extractelement <4 x float> %423, i64 2, !dbg !13
  %427 = extractelement <4 x float> %423, i64 3, !dbg !13
  %428 = bitcast float %380 to i32, !dbg !7
  %429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %428), !dbg !7
  %430 = bitcast i32 %429 to float, !dbg !7
  %431 = tail call float @llvm.maxnum.f32(float %380, float %430), !dbg !7
  %432 = bitcast float %431 to i32, !dbg !7
  %433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %432), !dbg !7
  %434 = bitcast i32 %433 to float, !dbg !7
  %435 = tail call float @llvm.maxnum.f32(float %431, float %434), !dbg !7
  %436 = tail call float @llvm.maxnum.f32(float %383, float -inf), !dbg !7, !frisk.local_reduction !5
  %local.reduce102 = tail call float @llvm.maxnum.f32(float %436, float %389), !dbg !7, !frisk.local_reduction !5
  %local.reduce103 = tail call float @llvm.maxnum.f32(float %384, float %390), !dbg !7, !frisk.local_reduction !5
  %local.reduce104 = tail call float @llvm.maxnum.f32(float %385, float %391), !dbg !7, !frisk.local_reduction !5
  %local.reduce105 = tail call float @llvm.maxnum.f32(float %386, float %392), !dbg !7, !frisk.local_reduction !5
  %local.reduce106 = tail call float @llvm.maxnum.f32(float %local.reduce102, float %395), !dbg !7, !frisk.local_reduction !5
  %local.reduce107 = tail call float @llvm.maxnum.f32(float %local.reduce103, float %396), !dbg !7, !frisk.local_reduction !5
  %local.reduce108 = tail call float @llvm.maxnum.f32(float %local.reduce104, float %397), !dbg !7, !frisk.local_reduction !5
  %local.reduce109 = tail call float @llvm.maxnum.f32(float %local.reduce105, float %398), !dbg !7, !frisk.local_reduction !5
  %local.reduce110 = tail call float @llvm.maxnum.f32(float %local.reduce106, float %401), !dbg !7, !frisk.local_reduction !5
  %local.reduce111 = tail call float @llvm.maxnum.f32(float %local.reduce107, float %402), !dbg !7, !frisk.local_reduction !5
  %local.reduce112 = tail call float @llvm.maxnum.f32(float %local.reduce108, float %403), !dbg !7, !frisk.local_reduction !5
  %local.reduce113 = tail call float @llvm.maxnum.f32(float %local.reduce109, float %404), !dbg !7, !frisk.local_reduction !5
  %local.reduce114 = tail call float @llvm.maxnum.f32(float %local.reduce110, float %407), !dbg !7, !frisk.local_reduction !5
  %local.reduce115 = tail call float @llvm.maxnum.f32(float %local.reduce111, float %408), !dbg !7, !frisk.local_reduction !5
  %local.reduce116 = tail call float @llvm.maxnum.f32(float %local.reduce112, float %409), !dbg !7, !frisk.local_reduction !5
  %local.reduce117 = tail call float @llvm.maxnum.f32(float %local.reduce113, float %410), !dbg !7, !frisk.local_reduction !5
  %local.reduce118 = tail call float @llvm.maxnum.f32(float %local.reduce114, float %413), !dbg !7, !frisk.local_reduction !5
  %local.reduce119 = tail call float @llvm.maxnum.f32(float %local.reduce115, float %414), !dbg !7, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %local.reduce116, float %415), !dbg !7, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %local.reduce117, float %416), !dbg !7, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %local.reduce118, float %419), !dbg !7, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %local.reduce119, float %420), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %421), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %422), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %424), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %425), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %426), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %427), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %local.reduce127), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %local.reduce129), !dbg !7, !frisk.local_reduction !5
  %437 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !7, !frisk.local_reduction !5
  %438 = bitcast float %437 to i32, !dbg !7
  %439 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %438), !dbg !7
  %440 = bitcast i32 %439 to float, !dbg !7
  %441 = tail call float @llvm.maxnum.f32(float %437, float %440), !dbg !7
  %442 = bitcast float %441 to i32, !dbg !7
  %443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %442), !dbg !7
  %444 = bitcast i32 %443 to float, !dbg !7
  %445 = tail call float @llvm.maxnum.f32(float %441, float %444), !dbg !7
  %lazy.peak = tail call float @llvm.maxnum.f32(float %435, float %445), !dbg !39
  %lazy.over = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !39
  %lazy.mask = tail call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !39
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !39
  %lazy.first = icmp eq i64 %124, 0, !dbg !39
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !39
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !39

lazy.slow:                                        ; preds = %117
  %lazy.positive = tail call float @llvm.maxnum.f32(float %435, float 0.000000e+00), !dbg !39
  %lazy.delta = select i1 %lazy.first, float %435, float %lazy.positive, !dbg !39
  %lazy.negated = fsub float 0.000000e+00, %lazy.delta, !dbg !39
  %lazy.exp = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !39
  %lazy.alpha = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp, !dbg !39
  %lazy.positive2057 = tail call float @llvm.maxnum.f32(float %445, float 0.000000e+00), !dbg !39
  %lazy.delta2058 = select i1 %lazy.first, float %445, float %lazy.positive2057, !dbg !39
  %lazy.negated2059 = fsub float 0.000000e+00, %lazy.delta2058, !dbg !39
  %lazy.exp2060 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated2059), !dbg !39
  %lazy.alpha2061 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp2060, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %323, %lazy.delta.v, !dbg !39
  %lazy.shifted2064 = fsub <4 x float> %330, %lazy.delta.v, !dbg !39
  %lazy.shifted2067 = fsub <4 x float> %340, %lazy.delta.v, !dbg !39
  %lazy.shifted2070 = fsub <4 x float> %350, %lazy.delta.v, !dbg !39
  %lazy.shifted2073 = fsub <4 x float> %356, %lazy.delta.v, !dbg !39
  %lazy.shifted2076 = fsub <4 x float> %362, %lazy.delta.v, !dbg !39
  %lazy.shifted2079 = fsub <4 x float> %368, %lazy.delta.v, !dbg !39
  %lazy.shifted2082 = fsub <4 x float> %374, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane2083 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !39
  %lazy.delta.v2084 = shufflevector <1 x float> %lazy.delta.v.lane2083, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2085 = fsub <4 x float> %381, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2088 = fsub <4 x float> %387, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2091 = fsub <4 x float> %393, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2094 = fsub <4 x float> %399, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2097 = fsub <4 x float> %405, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2100 = fsub <4 x float> %411, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2103 = fsub <4 x float> %417, %lazy.delta.v2084, !dbg !39
  %lazy.shifted2106 = fsub <4 x float> %423, %lazy.delta.v2084, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %122, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state2107 = extractvalue [2 x <1 x float>] %122, 1, !dbg !39
  %lazy.alpha.v.lane2108 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !39
  %lazy.scaled2109 = fmul <1 x float> %lazy.state2107, %lazy.alpha.v.lane2108, !dbg !39
  %lazy.state.next2110 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2109, 1, !dbg !39
  %lazy.state2111 = extractvalue [2 x <8 x float>] %118, 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2113 = fmul <8 x float> %lazy.state2111, %lazy.alpha.v, !dbg !39
  %lazy.state.next2114 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2113, 0, !dbg !39
  %lazy.state2115 = extractvalue [2 x <8 x float>] %118, 1, !dbg !39
  %lazy.alpha.v2117 = shufflevector <1 x float> %lazy.alpha.v.lane2108, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2118 = fmul <8 x float> %lazy.state2115, %lazy.alpha.v2117, !dbg !39
  %lazy.state.next2119 = insertvalue [2 x <8 x float>] %lazy.state.next2114, <8 x float> %lazy.scaled2118, 1, !dbg !39
  %lazy.state2120 = extractvalue [2 x <8 x float>] %119, 0, !dbg !39
  %lazy.scaled2123 = fmul <8 x float> %lazy.state2120, %lazy.alpha.v, !dbg !39
  %lazy.state.next2124 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2123, 0, !dbg !39
  %lazy.state2125 = extractvalue [2 x <8 x float>] %119, 1, !dbg !39
  %lazy.scaled2128 = fmul <8 x float> %lazy.state2125, %lazy.alpha.v2117, !dbg !39
  %lazy.state.next2129 = insertvalue [2 x <8 x float>] %lazy.state.next2124, <8 x float> %lazy.scaled2128, 1, !dbg !39
  %lazy.state2130 = extractvalue [2 x <8 x float>] %120, 0, !dbg !39
  %lazy.scaled2133 = fmul <8 x float> %lazy.state2130, %lazy.alpha.v, !dbg !39
  %lazy.state.next2134 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2133, 0, !dbg !39
  %lazy.state2135 = extractvalue [2 x <8 x float>] %120, 1, !dbg !39
  %lazy.scaled2138 = fmul <8 x float> %lazy.state2135, %lazy.alpha.v2117, !dbg !39
  %lazy.state.next2139 = insertvalue [2 x <8 x float>] %lazy.state.next2134, <8 x float> %lazy.scaled2138, 1, !dbg !39
  %lazy.state2140 = extractvalue [2 x <8 x float>] %123, 0, !dbg !39
  %lazy.scaled2143 = fmul <8 x float> %lazy.state2140, %lazy.alpha.v, !dbg !39
  %lazy.state.next2144 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2143, 0, !dbg !39
  %lazy.state2145 = extractvalue [2 x <8 x float>] %123, 1, !dbg !39
  %lazy.scaled2148 = fmul <8 x float> %lazy.state2145, %lazy.alpha.v2117, !dbg !39
  %lazy.state.next2149 = insertvalue [2 x <8 x float>] %lazy.state.next2144, <8 x float> %lazy.scaled2148, 1, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.next2152 = fsub <1 x float> %lazy.base2055, %lazy.delta.v.lane2083, !dbg !39
  %lazy.base.state2153 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2152, 1, !dbg !39
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !40
  %.pre25 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !40
  %.pre27 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !40
  %.pre29 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !40
  %.pre31 = extractelement <4 x float> %lazy.shifted2085, i64 0, !dbg !40
  %.pre33 = extractelement <4 x float> %lazy.shifted2085, i64 1, !dbg !40
  %.pre35 = extractelement <4 x float> %lazy.shifted2085, i64 2, !dbg !40
  %.pre37 = extractelement <4 x float> %lazy.shifted2085, i64 3, !dbg !40
  %.pre39 = extractelement <4 x float> %lazy.shifted2064, i64 0, !dbg !40
  %.pre41 = extractelement <4 x float> %lazy.shifted2064, i64 1, !dbg !40
  %.pre43 = extractelement <4 x float> %lazy.shifted2064, i64 2, !dbg !40
  %.pre45 = extractelement <4 x float> %lazy.shifted2064, i64 3, !dbg !40
  %.pre47 = extractelement <4 x float> %lazy.shifted2088, i64 0, !dbg !40
  %.pre49 = extractelement <4 x float> %lazy.shifted2088, i64 1, !dbg !40
  %.pre51 = extractelement <4 x float> %lazy.shifted2088, i64 2, !dbg !40
  %.pre53 = extractelement <4 x float> %lazy.shifted2088, i64 3, !dbg !40
  %.pre55 = extractelement <4 x float> %lazy.shifted2067, i64 0, !dbg !40
  %.pre57 = extractelement <4 x float> %lazy.shifted2067, i64 1, !dbg !40
  %.pre59 = extractelement <4 x float> %lazy.shifted2067, i64 2, !dbg !40
  %.pre61 = extractelement <4 x float> %lazy.shifted2067, i64 3, !dbg !40
  %.pre63 = extractelement <4 x float> %lazy.shifted2091, i64 0, !dbg !40
  %.pre65 = extractelement <4 x float> %lazy.shifted2091, i64 1, !dbg !40
  %.pre67 = extractelement <4 x float> %lazy.shifted2091, i64 2, !dbg !40
  %.pre69 = extractelement <4 x float> %lazy.shifted2091, i64 3, !dbg !40
  %.pre71 = extractelement <4 x float> %lazy.shifted2070, i64 0, !dbg !40
  %.pre73 = extractelement <4 x float> %lazy.shifted2070, i64 1, !dbg !40
  %.pre75 = extractelement <4 x float> %lazy.shifted2070, i64 2, !dbg !40
  %.pre77 = extractelement <4 x float> %lazy.shifted2070, i64 3, !dbg !40
  %.pre79 = extractelement <4 x float> %lazy.shifted2094, i64 0, !dbg !40
  %.pre81 = extractelement <4 x float> %lazy.shifted2094, i64 1, !dbg !40
  %.pre83 = extractelement <4 x float> %lazy.shifted2094, i64 2, !dbg !40
  %.pre85 = extractelement <4 x float> %lazy.shifted2094, i64 3, !dbg !40
  %.pre87 = extractelement <4 x float> %lazy.shifted2073, i64 0, !dbg !40
  %.pre89 = extractelement <4 x float> %lazy.shifted2073, i64 1, !dbg !40
  %.pre91 = extractelement <4 x float> %lazy.shifted2073, i64 2, !dbg !40
  %.pre93 = extractelement <4 x float> %lazy.shifted2073, i64 3, !dbg !40
  %.pre95 = extractelement <4 x float> %lazy.shifted2097, i64 0, !dbg !40
  %.pre97 = extractelement <4 x float> %lazy.shifted2097, i64 1, !dbg !40
  %.pre99 = extractelement <4 x float> %lazy.shifted2097, i64 2, !dbg !40
  %.pre101 = extractelement <4 x float> %lazy.shifted2097, i64 3, !dbg !40
  %.pre103 = extractelement <4 x float> %lazy.shifted2076, i64 0, !dbg !40
  %.pre105 = extractelement <4 x float> %lazy.shifted2076, i64 1, !dbg !40
  %.pre107 = extractelement <4 x float> %lazy.shifted2076, i64 2, !dbg !40
  %.pre109 = extractelement <4 x float> %lazy.shifted2076, i64 3, !dbg !40
  %.pre111 = extractelement <4 x float> %lazy.shifted2100, i64 0, !dbg !40
  %.pre113 = extractelement <4 x float> %lazy.shifted2100, i64 1, !dbg !40
  %.pre115 = extractelement <4 x float> %lazy.shifted2100, i64 2, !dbg !40
  %.pre117 = extractelement <4 x float> %lazy.shifted2100, i64 3, !dbg !40
  %.pre119 = extractelement <4 x float> %lazy.shifted2079, i64 0, !dbg !40
  %.pre121 = extractelement <4 x float> %lazy.shifted2079, i64 1, !dbg !40
  %.pre123 = extractelement <4 x float> %lazy.shifted2079, i64 2, !dbg !40
  %.pre125 = extractelement <4 x float> %lazy.shifted2079, i64 3, !dbg !40
  %.pre127 = extractelement <4 x float> %lazy.shifted2103, i64 0, !dbg !40
  %.pre129 = extractelement <4 x float> %lazy.shifted2103, i64 1, !dbg !40
  %.pre131 = extractelement <4 x float> %lazy.shifted2103, i64 2, !dbg !40
  %.pre133 = extractelement <4 x float> %lazy.shifted2103, i64 3, !dbg !40
  %.pre135 = extractelement <4 x float> %lazy.shifted2082, i64 0, !dbg !40
  %.pre137 = extractelement <4 x float> %lazy.shifted2082, i64 1, !dbg !40
  %.pre139 = extractelement <4 x float> %lazy.shifted2082, i64 2, !dbg !40
  %.pre141 = extractelement <4 x float> %lazy.shifted2082, i64 3, !dbg !40
  %.pre143 = extractelement <4 x float> %lazy.shifted2106, i64 0, !dbg !40
  %.pre145 = extractelement <4 x float> %lazy.shifted2106, i64 1, !dbg !40
  %.pre147 = extractelement <4 x float> %lazy.shifted2106, i64 2, !dbg !40
  %.pre149 = extractelement <4 x float> %lazy.shifted2106, i64 3, !dbg !40
  br label %lazy.join, !dbg !39

lazy.join:                                        ; preds = %lazy.slow, %117
  %.pre-phi150 = phi float [ %.pre149, %lazy.slow ], [ %427, %117 ], !dbg !40
  %.pre-phi148 = phi float [ %.pre147, %lazy.slow ], [ %426, %117 ], !dbg !40
  %.pre-phi146 = phi float [ %.pre145, %lazy.slow ], [ %425, %117 ], !dbg !40
  %.pre-phi144 = phi float [ %.pre143, %lazy.slow ], [ %424, %117 ], !dbg !40
  %.pre-phi142 = phi float [ %.pre141, %lazy.slow ], [ %379, %117 ], !dbg !40
  %.pre-phi140 = phi float [ %.pre139, %lazy.slow ], [ %378, %117 ], !dbg !40
  %.pre-phi138 = phi float [ %.pre137, %lazy.slow ], [ %377, %117 ], !dbg !40
  %.pre-phi136 = phi float [ %.pre135, %lazy.slow ], [ %376, %117 ], !dbg !40
  %.pre-phi134 = phi float [ %.pre133, %lazy.slow ], [ %422, %117 ], !dbg !40
  %.pre-phi132 = phi float [ %.pre131, %lazy.slow ], [ %421, %117 ], !dbg !40
  %.pre-phi130 = phi float [ %.pre129, %lazy.slow ], [ %420, %117 ], !dbg !40
  %.pre-phi128 = phi float [ %.pre127, %lazy.slow ], [ %419, %117 ], !dbg !40
  %.pre-phi126 = phi float [ %.pre125, %lazy.slow ], [ %373, %117 ], !dbg !40
  %.pre-phi124 = phi float [ %.pre123, %lazy.slow ], [ %372, %117 ], !dbg !40
  %.pre-phi122 = phi float [ %.pre121, %lazy.slow ], [ %371, %117 ], !dbg !40
  %.pre-phi120 = phi float [ %.pre119, %lazy.slow ], [ %370, %117 ], !dbg !40
  %.pre-phi118 = phi float [ %.pre117, %lazy.slow ], [ %416, %117 ], !dbg !40
  %.pre-phi116 = phi float [ %.pre115, %lazy.slow ], [ %415, %117 ], !dbg !40
  %.pre-phi114 = phi float [ %.pre113, %lazy.slow ], [ %414, %117 ], !dbg !40
  %.pre-phi112 = phi float [ %.pre111, %lazy.slow ], [ %413, %117 ], !dbg !40
  %.pre-phi110 = phi float [ %.pre109, %lazy.slow ], [ %367, %117 ], !dbg !40
  %.pre-phi108 = phi float [ %.pre107, %lazy.slow ], [ %366, %117 ], !dbg !40
  %.pre-phi106 = phi float [ %.pre105, %lazy.slow ], [ %365, %117 ], !dbg !40
  %.pre-phi104 = phi float [ %.pre103, %lazy.slow ], [ %364, %117 ], !dbg !40
  %.pre-phi102 = phi float [ %.pre101, %lazy.slow ], [ %410, %117 ], !dbg !40
  %.pre-phi100 = phi float [ %.pre99, %lazy.slow ], [ %409, %117 ], !dbg !40
  %.pre-phi98 = phi float [ %.pre97, %lazy.slow ], [ %408, %117 ], !dbg !40
  %.pre-phi96 = phi float [ %.pre95, %lazy.slow ], [ %407, %117 ], !dbg !40
  %.pre-phi94 = phi float [ %.pre93, %lazy.slow ], [ %361, %117 ], !dbg !40
  %.pre-phi92 = phi float [ %.pre91, %lazy.slow ], [ %360, %117 ], !dbg !40
  %.pre-phi90 = phi float [ %.pre89, %lazy.slow ], [ %359, %117 ], !dbg !40
  %.pre-phi88 = phi float [ %.pre87, %lazy.slow ], [ %358, %117 ], !dbg !40
  %.pre-phi86 = phi float [ %.pre85, %lazy.slow ], [ %404, %117 ], !dbg !40
  %.pre-phi84 = phi float [ %.pre83, %lazy.slow ], [ %403, %117 ], !dbg !40
  %.pre-phi82 = phi float [ %.pre81, %lazy.slow ], [ %402, %117 ], !dbg !40
  %.pre-phi80 = phi float [ %.pre79, %lazy.slow ], [ %401, %117 ], !dbg !40
  %.pre-phi78 = phi float [ %.pre77, %lazy.slow ], [ %355, %117 ], !dbg !40
  %.pre-phi76 = phi float [ %.pre75, %lazy.slow ], [ %354, %117 ], !dbg !40
  %.pre-phi74 = phi float [ %.pre73, %lazy.slow ], [ %353, %117 ], !dbg !40
  %.pre-phi72 = phi float [ %.pre71, %lazy.slow ], [ %352, %117 ], !dbg !40
  %.pre-phi70 = phi float [ %.pre69, %lazy.slow ], [ %398, %117 ], !dbg !40
  %.pre-phi68 = phi float [ %.pre67, %lazy.slow ], [ %397, %117 ], !dbg !40
  %.pre-phi66 = phi float [ %.pre65, %lazy.slow ], [ %396, %117 ], !dbg !40
  %.pre-phi64 = phi float [ %.pre63, %lazy.slow ], [ %395, %117 ], !dbg !40
  %.pre-phi62 = phi float [ %.pre61, %lazy.slow ], [ %347, %117 ], !dbg !40
  %.pre-phi60 = phi float [ %.pre59, %lazy.slow ], [ %346, %117 ], !dbg !40
  %.pre-phi58 = phi float [ %.pre57, %lazy.slow ], [ %345, %117 ], !dbg !40
  %.pre-phi56 = phi float [ %.pre55, %lazy.slow ], [ %344, %117 ], !dbg !40
  %.pre-phi54 = phi float [ %.pre53, %lazy.slow ], [ %392, %117 ], !dbg !40
  %.pre-phi52 = phi float [ %.pre51, %lazy.slow ], [ %391, %117 ], !dbg !40
  %.pre-phi50 = phi float [ %.pre49, %lazy.slow ], [ %390, %117 ], !dbg !40
  %.pre-phi48 = phi float [ %.pre47, %lazy.slow ], [ %389, %117 ], !dbg !40
  %.pre-phi46 = phi float [ %.pre45, %lazy.slow ], [ %337, %117 ], !dbg !40
  %.pre-phi44 = phi float [ %.pre43, %lazy.slow ], [ %336, %117 ], !dbg !40
  %.pre-phi42 = phi float [ %.pre41, %lazy.slow ], [ %335, %117 ], !dbg !40
  %.pre-phi40 = phi float [ %.pre39, %lazy.slow ], [ %334, %117 ], !dbg !40
  %.pre-phi38 = phi float [ %.pre37, %lazy.slow ], [ %386, %117 ], !dbg !40
  %.pre-phi36 = phi float [ %.pre35, %lazy.slow ], [ %385, %117 ], !dbg !40
  %.pre-phi34 = phi float [ %.pre33, %lazy.slow ], [ %384, %117 ], !dbg !40
  %.pre-phi32 = phi float [ %.pre31, %lazy.slow ], [ %383, %117 ], !dbg !40
  %.pre-phi30 = phi float [ %.pre29, %lazy.slow ], [ %329, %117 ], !dbg !40
  %.pre-phi28 = phi float [ %.pre27, %lazy.slow ], [ %328, %117 ], !dbg !40
  %.pre-phi26 = phi float [ %.pre25, %lazy.slow ], [ %327, %117 ], !dbg !40
  %.pre-phi = phi float [ %.pre, %lazy.slow ], [ %325, %117 ], !dbg !40
  %lazy.state2169 = phi [2 x <1 x float>] [ %lazy.state.next2110, %lazy.slow ], [ %122, %117 ], !dbg !39
  %lazy.state2170 = phi [2 x <8 x float>] [ %lazy.state.next2119, %lazy.slow ], [ %118, %117 ], !dbg !39
  %lazy.state2171 = phi [2 x <8 x float>] [ %lazy.state.next2129, %lazy.slow ], [ %119, %117 ], !dbg !39
  %lazy.state2172 = phi [2 x <8 x float>] [ %lazy.state.next2139, %lazy.slow ], [ %120, %117 ], !dbg !39
  %lazy.state2173 = phi [2 x <8 x float>] [ %lazy.state.next2149, %lazy.slow ], [ %123, %117 ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2153, %lazy.slow ], [ %121, %117 ], !dbg !39
  %446 = extractvalue [2 x <1 x float>] %lazy.state2169, 0, !dbg !39
  %447 = extractelement <1 x float> %446, i64 0, !dbg !39
  %448 = extractvalue [2 x <1 x float>] %lazy.state2169, 1, !dbg !39
  %449 = extractelement <1 x float> %448, i64 0, !dbg !39
  %450 = extractvalue [2 x <8 x float>] %lazy.state2173, 0, !dbg !41
  %451 = shufflevector <8 x float> %450, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %452 = shufflevector <8 x float> %450, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %453 = extractvalue [2 x <8 x float>] %lazy.state2173, 1, !dbg !41
  %454 = shufflevector <8 x float> %453, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %455 = shufflevector <8 x float> %453, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = add nsw i64 %address.local916, %127, !dbg !38
  %vdb.address1420 = trunc nsw i64 %vdb.address to i32, !dbg !38
  %vdb.address1421 = shl nsw i32 %vdb.address1420, 1, !dbg !38
  %vdb.address1426 = trunc nuw nsw i64 %127 to i32, !dbg !38
  %vdb.address1427 = or disjoint i32 %address.local923, %vdb.address1426, !dbg !38
  %vdb.address1428 = shl nuw nsw i32 %vdb.address1427, 1, !dbg !38
  %456 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1421, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %457 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1428, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %address.local927 = and i32 %address.local918, 120, !dbg !42
  %address.local929 = and i32 %address.local569, 1536, !dbg !42
  %address.local930 = or disjoint i32 %address.local927, %address.local929, !dbg !42
  %address.local931 = zext nneg i32 %address.local930 to i64, !dbg !42
  %address.local932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local931, !dbg !42
  %458 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %459 = bitcast <4 x i32> %458 to <8 x half>, !dbg !42
  %address.local933 = and i64 %address.local496, 60, !dbg !42
  %address.local934 = add nuw nsw i64 %address.local933, %address.local496, !dbg !42
  %address.local936.idx = shl nuw nsw i64 %address.local934, 4, !dbg !42
  %address.local936 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local936.idx, !dbg !42
  %address.local937 = lshr i64 %address.local549, 4, !dbg !42
  %address.local939.idx = mul nuw nsw i64 %address.local937, 768, !dbg !42
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local936, i64 %address.local939.idx, !dbg !42
  %address.local941.idx = mul nsw i64 %address.local502, -64, !dbg !42
  %address.local941 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local941.idx, !dbg !42
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local941, i64 256, !dbg !42
  %460 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %461 = bitcast <4 x i32> %460 to <8 x half>, !dbg !42
  %address.local946 = getelementptr i8, ptr addrspace(3) %address.local941, i64 512, !dbg !42
  %462 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %463 = bitcast <4 x i32> %462 to <8 x half>, !dbg !42
  %address.local950 = getelementptr i8, ptr addrspace(3) %address.local941, i64 768, !dbg !42
  %464 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %465 = bitcast <4 x i32> %464 to <8 x half>, !dbg !42
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local932, i64 4096, !dbg !42
  %466 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %467 = bitcast <4 x i32> %466 to <8 x half>, !dbg !42
  %address.local956 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4352, !dbg !42
  %468 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %469 = bitcast <4 x i32> %468 to <8 x half>, !dbg !42
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4608, !dbg !42
  %470 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %471 = bitcast <4 x i32> %470 to <8 x half>, !dbg !42
  %address.local964 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4864, !dbg !42
  %472 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %473 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %474 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !40
  %475 = insertelement <4 x float> poison, float %474, i64 0, !dbg !44
  %476 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi26), !dbg !40
  %477 = insertelement <4 x float> %475, float %476, i64 1, !dbg !44
  %478 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi28), !dbg !40
  %479 = insertelement <4 x float> %477, float %478, i64 2, !dbg !44
  %480 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi30), !dbg !40
  %481 = insertelement <4 x float> %479, float %480, i64 3, !dbg !44
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %481) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !44
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !44
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !44
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !44
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !44
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !44
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !44
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !44
  %482 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !44
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %482) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %483 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %473, <4 x float> %451) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %484 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %485 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %484, <4 x float> %452) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %486 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi32), !dbg !40
  %487 = insertelement <4 x float> poison, float %486, i64 0, !dbg !44
  %488 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi34), !dbg !40
  %489 = insertelement <4 x float> %487, float %488, i64 1, !dbg !44
  %490 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi36), !dbg !40
  %491 = insertelement <4 x float> %489, float %490, i64 2, !dbg !44
  %492 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi38), !dbg !40
  %493 = insertelement <4 x float> %491, float %492, i64 3, !dbg !44
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %493) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02174 = extractelement <4 x float> %math.result382, i64 0, !dbg !44
  %.cvtpack.f12175 = extractelement <4 x float> %math.result382, i64 1, !dbg !44
  %.cvtpack.f22176 = extractelement <4 x float> %math.result382, i64 2, !dbg !44
  %.cvtpack.f32177 = extractelement <4 x float> %math.result382, i64 3, !dbg !44
  %.cvtpack.p02178 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02174, float %.cvtpack.f12175) #12, !dbg !44
  %.cvtpack.p12179 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22176, float %.cvtpack.f32177) #12, !dbg !44
  %.cvtpack.v02180 = insertelement <2 x i32> poison, i32 %.cvtpack.p02178, i64 0, !dbg !44
  %.cvtpack.v12181 = insertelement <2 x i32> %.cvtpack.v02180, i32 %.cvtpack.p12179, i64 1, !dbg !44
  %494 = bitcast <2 x i32> %.cvtpack.v12181 to <4 x half>, !dbg !44
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %494) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %495 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %473, <4 x float> %454) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %484, <4 x float> %455) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %497 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %498 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi40), !dbg !40
  %local.reduce71 = fadd float %474, %498, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %499 = insertelement <4 x float> poison, float %498, i64 0, !dbg !44
  %500 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi42), !dbg !40
  %local.reduce72 = fadd float %476, %500, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1977 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %501 = insertelement <4 x float> %499, float %500, i64 1, !dbg !44
  %502 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi44), !dbg !40
  %local.reduce73 = fadd float %478, %502, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1978 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %503 = insertelement <4 x float> %501, float %502, i64 2, !dbg !44
  %504 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi46), !dbg !40
  %local.reduce74 = fadd float %480, %504, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1979 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %505 = insertelement <4 x float> %503, float %504, i64 3, !dbg !44
  %math.result368 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %505) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02182 = extractelement <4 x float> %math.result368, i64 0, !dbg !44
  %.cvtpack.f12183 = extractelement <4 x float> %math.result368, i64 1, !dbg !44
  %.cvtpack.f22184 = extractelement <4 x float> %math.result368, i64 2, !dbg !44
  %.cvtpack.f32185 = extractelement <4 x float> %math.result368, i64 3, !dbg !44
  %.cvtpack.p02186 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02182, float %.cvtpack.f12183) #12, !dbg !44
  %.cvtpack.p12187 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22184, float %.cvtpack.f32185) #12, !dbg !44
  %.cvtpack.v02188 = insertelement <2 x i32> poison, i32 %.cvtpack.p02186, i64 0, !dbg !44
  %.cvtpack.v12189 = insertelement <2 x i32> %.cvtpack.v02188, i32 %.cvtpack.p12187, i64 1, !dbg !44
  %506 = bitcast <2 x i32> %.cvtpack.v12189 to <4 x half>, !dbg !44
  %math.packed369 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %506) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %497, <4 x float> %483) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %508 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %509 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %508, <4 x float> %485) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %510 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi48), !dbg !40
  %local.reduce = fadd float %486, %510, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1980 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %511 = insertelement <4 x float> poison, float %510, i64 0, !dbg !44
  %512 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi50), !dbg !40
  %local.reduce41 = fadd float %488, %512, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1981 = tail call float asm sideeffect "", "=v,0"(float %local.reduce41) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %513 = insertelement <4 x float> %511, float %512, i64 1, !dbg !44
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi52), !dbg !40
  %local.reduce42 = fadd float %490, %514, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1982 = tail call float asm sideeffect "", "=v,0"(float %local.reduce42) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %515 = insertelement <4 x float> %513, float %514, i64 2, !dbg !44
  %516 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi54), !dbg !40
  %local.reduce43 = fadd float %492, %516, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1983 = tail call float asm sideeffect "", "=v,0"(float %local.reduce43) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %517 = insertelement <4 x float> %515, float %516, i64 3, !dbg !44
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %517) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02190 = extractelement <4 x float> %math.result384, i64 0, !dbg !44
  %.cvtpack.f12191 = extractelement <4 x float> %math.result384, i64 1, !dbg !44
  %.cvtpack.f22192 = extractelement <4 x float> %math.result384, i64 2, !dbg !44
  %.cvtpack.f32193 = extractelement <4 x float> %math.result384, i64 3, !dbg !44
  %.cvtpack.p02194 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02190, float %.cvtpack.f12191) #12, !dbg !44
  %.cvtpack.p12195 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22192, float %.cvtpack.f32193) #12, !dbg !44
  %.cvtpack.v02196 = insertelement <2 x i32> poison, i32 %.cvtpack.p02194, i64 0, !dbg !44
  %.cvtpack.v12197 = insertelement <2 x i32> %.cvtpack.v02196, i32 %.cvtpack.p12195, i64 1, !dbg !44
  %518 = bitcast <2 x i32> %.cvtpack.v12197 to <4 x half>, !dbg !44
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %518) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %519 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %497, <4 x float> %495) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %520 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %508, <4 x float> %496) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %521 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %522 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi56), !dbg !40
  %local.reduce75 = fadd float %522, %k0.sum.anchor, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1984 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %523 = insertelement <4 x float> poison, float %522, i64 0, !dbg !44
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi58), !dbg !40
  %local.reduce76 = fadd float %524, %k0.sum.anchor1977, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1985 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %525 = insertelement <4 x float> %523, float %524, i64 1, !dbg !44
  %526 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi60), !dbg !40
  %local.reduce77 = fadd float %526, %k0.sum.anchor1978, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1986 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %527 = insertelement <4 x float> %525, float %526, i64 2, !dbg !44
  %528 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi62), !dbg !40
  %local.reduce78 = fadd float %528, %k0.sum.anchor1979, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1987 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %529 = insertelement <4 x float> %527, float %528, i64 3, !dbg !44
  %math.result370 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %529) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02198 = extractelement <4 x float> %math.result370, i64 0, !dbg !44
  %.cvtpack.f12199 = extractelement <4 x float> %math.result370, i64 1, !dbg !44
  %.cvtpack.f22200 = extractelement <4 x float> %math.result370, i64 2, !dbg !44
  %.cvtpack.f32201 = extractelement <4 x float> %math.result370, i64 3, !dbg !44
  %.cvtpack.p02202 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02198, float %.cvtpack.f12199) #12, !dbg !44
  %.cvtpack.p12203 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22200, float %.cvtpack.f32201) #12, !dbg !44
  %.cvtpack.v02204 = insertelement <2 x i32> poison, i32 %.cvtpack.p02202, i64 0, !dbg !44
  %.cvtpack.v12205 = insertelement <2 x i32> %.cvtpack.v02204, i32 %.cvtpack.p12203, i64 1, !dbg !44
  %530 = bitcast <2 x i32> %.cvtpack.v12205 to <4 x half>, !dbg !44
  %math.packed371 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %530) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %521, <4 x float> %507) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %532 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %532, <4 x float> %509) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi64), !dbg !40
  %local.reduce44 = fadd float %534, %k0.sum.anchor1980, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1988 = tail call float asm sideeffect "", "=v,0"(float %local.reduce44) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !44
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi66), !dbg !40
  %local.reduce45 = fadd float %536, %k0.sum.anchor1981, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1989 = tail call float asm sideeffect "", "=v,0"(float %local.reduce45) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %537 = insertelement <4 x float> %535, float %536, i64 1, !dbg !44
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi68), !dbg !40
  %local.reduce46 = fadd float %538, %k0.sum.anchor1982, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1990 = tail call float asm sideeffect "", "=v,0"(float %local.reduce46) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %539 = insertelement <4 x float> %537, float %538, i64 2, !dbg !44
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi70), !dbg !40
  %local.reduce47 = fadd float %540, %k0.sum.anchor1983, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1991 = tail call float asm sideeffect "", "=v,0"(float %local.reduce47) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %541 = insertelement <4 x float> %539, float %540, i64 3, !dbg !44
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %541) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02206 = extractelement <4 x float> %math.result386, i64 0, !dbg !44
  %.cvtpack.f12207 = extractelement <4 x float> %math.result386, i64 1, !dbg !44
  %.cvtpack.f22208 = extractelement <4 x float> %math.result386, i64 2, !dbg !44
  %.cvtpack.f32209 = extractelement <4 x float> %math.result386, i64 3, !dbg !44
  %.cvtpack.p02210 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02206, float %.cvtpack.f12207) #12, !dbg !44
  %.cvtpack.p12211 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22208, float %.cvtpack.f32209) #12, !dbg !44
  %.cvtpack.v02212 = insertelement <2 x i32> poison, i32 %.cvtpack.p02210, i64 0, !dbg !44
  %.cvtpack.v12213 = insertelement <2 x i32> %.cvtpack.v02212, i32 %.cvtpack.p12211, i64 1, !dbg !44
  %542 = bitcast <2 x i32> %.cvtpack.v12213 to <4 x half>, !dbg !44
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %542) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %521, <4 x float> %519) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %532, <4 x float> %520) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %545 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi72), !dbg !40
  %local.reduce79 = fadd float %546, %k0.sum.anchor1984, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1992 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !44
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi74), !dbg !40
  %local.reduce80 = fadd float %548, %k0.sum.anchor1985, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1993 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %549 = insertelement <4 x float> %547, float %548, i64 1, !dbg !44
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi76), !dbg !40
  %local.reduce81 = fadd float %550, %k0.sum.anchor1986, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1994 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %551 = insertelement <4 x float> %549, float %550, i64 2, !dbg !44
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi78), !dbg !40
  %local.reduce82 = fadd float %552, %k0.sum.anchor1987, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1995 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %553 = insertelement <4 x float> %551, float %552, i64 3, !dbg !44
  %math.result372 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %553) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02214 = extractelement <4 x float> %math.result372, i64 0, !dbg !44
  %.cvtpack.f12215 = extractelement <4 x float> %math.result372, i64 1, !dbg !44
  %.cvtpack.f22216 = extractelement <4 x float> %math.result372, i64 2, !dbg !44
  %.cvtpack.f32217 = extractelement <4 x float> %math.result372, i64 3, !dbg !44
  %.cvtpack.p02218 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02214, float %.cvtpack.f12215) #12, !dbg !44
  %.cvtpack.p12219 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22216, float %.cvtpack.f32217) #12, !dbg !44
  %.cvtpack.v02220 = insertelement <2 x i32> poison, i32 %.cvtpack.p02218, i64 0, !dbg !44
  %.cvtpack.v12221 = insertelement <2 x i32> %.cvtpack.v02220, i32 %.cvtpack.p12219, i64 1, !dbg !44
  %554 = bitcast <2 x i32> %.cvtpack.v12221 to <4 x half>, !dbg !44
  %math.packed373 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %554) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %545, <4 x float> %531) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %556 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %556, <4 x float> %533) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %558 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi80), !dbg !40
  %local.reduce48 = fadd float %558, %k0.sum.anchor1988, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1996 = tail call float asm sideeffect "", "=v,0"(float %local.reduce48) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %559 = insertelement <4 x float> poison, float %558, i64 0, !dbg !44
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi82), !dbg !40
  %local.reduce49 = fadd float %560, %k0.sum.anchor1989, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1997 = tail call float asm sideeffect "", "=v,0"(float %local.reduce49) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %561 = insertelement <4 x float> %559, float %560, i64 1, !dbg !44
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi84), !dbg !40
  %local.reduce50 = fadd float %562, %k0.sum.anchor1990, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1998 = tail call float asm sideeffect "", "=v,0"(float %local.reduce50) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> %561, float %562, i64 2, !dbg !44
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi86), !dbg !40
  %local.reduce51 = fadd float %564, %k0.sum.anchor1991, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1999 = tail call float asm sideeffect "", "=v,0"(float %local.reduce51) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %565 = insertelement <4 x float> %563, float %564, i64 3, !dbg !44
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %565) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02222 = extractelement <4 x float> %math.result388, i64 0, !dbg !44
  %.cvtpack.f12223 = extractelement <4 x float> %math.result388, i64 1, !dbg !44
  %.cvtpack.f22224 = extractelement <4 x float> %math.result388, i64 2, !dbg !44
  %.cvtpack.f32225 = extractelement <4 x float> %math.result388, i64 3, !dbg !44
  %.cvtpack.p02226 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02222, float %.cvtpack.f12223) #12, !dbg !44
  %.cvtpack.p12227 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22224, float %.cvtpack.f32225) #12, !dbg !44
  %.cvtpack.v02228 = insertelement <2 x i32> poison, i32 %.cvtpack.p02226, i64 0, !dbg !44
  %.cvtpack.v12229 = insertelement <2 x i32> %.cvtpack.v02228, i32 %.cvtpack.p12227, i64 1, !dbg !44
  %566 = bitcast <2 x i32> %.cvtpack.v12229 to <4 x half>, !dbg !44
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %566) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %567 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %545, <4 x float> %543) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %556, <4 x float> %544) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %569 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi88), !dbg !40
  %local.reduce83 = fadd float %570, %k0.sum.anchor1992, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2000 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %571 = insertelement <4 x float> poison, float %570, i64 0, !dbg !44
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi90), !dbg !40
  %local.reduce84 = fadd float %572, %k0.sum.anchor1993, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2001 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %573 = insertelement <4 x float> %571, float %572, i64 1, !dbg !44
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi92), !dbg !40
  %local.reduce85 = fadd float %574, %k0.sum.anchor1994, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2002 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %575 = insertelement <4 x float> %573, float %574, i64 2, !dbg !44
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi94), !dbg !40
  %local.reduce86 = fadd float %576, %k0.sum.anchor1995, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2003 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %577 = insertelement <4 x float> %575, float %576, i64 3, !dbg !44
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %577) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02230 = extractelement <4 x float> %math.result374, i64 0, !dbg !44
  %.cvtpack.f12231 = extractelement <4 x float> %math.result374, i64 1, !dbg !44
  %.cvtpack.f22232 = extractelement <4 x float> %math.result374, i64 2, !dbg !44
  %.cvtpack.f32233 = extractelement <4 x float> %math.result374, i64 3, !dbg !44
  %.cvtpack.p02234 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02230, float %.cvtpack.f12231) #12, !dbg !44
  %.cvtpack.p12235 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22232, float %.cvtpack.f32233) #12, !dbg !44
  %.cvtpack.v02236 = insertelement <2 x i32> poison, i32 %.cvtpack.p02234, i64 0, !dbg !44
  %.cvtpack.v12237 = insertelement <2 x i32> %.cvtpack.v02236, i32 %.cvtpack.p12235, i64 1, !dbg !44
  %578 = bitcast <2 x i32> %.cvtpack.v12237 to <4 x half>, !dbg !44
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %578) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %569, <4 x float> %555) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %580 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %580, <4 x float> %557) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %582 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi96), !dbg !40
  %local.reduce52 = fadd float %582, %k0.sum.anchor1996, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2004 = tail call float asm sideeffect "", "=v,0"(float %local.reduce52) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %583 = insertelement <4 x float> poison, float %582, i64 0, !dbg !44
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi98), !dbg !40
  %local.reduce53 = fadd float %584, %k0.sum.anchor1997, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2005 = tail call float asm sideeffect "", "=v,0"(float %local.reduce53) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %583, float %584, i64 1, !dbg !44
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi100), !dbg !40
  %local.reduce54 = fadd float %586, %k0.sum.anchor1998, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2006 = tail call float asm sideeffect "", "=v,0"(float %local.reduce54) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %587 = insertelement <4 x float> %585, float %586, i64 2, !dbg !44
  %588 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi102), !dbg !40
  %local.reduce55 = fadd float %588, %k0.sum.anchor1999, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2007 = tail call float asm sideeffect "", "=v,0"(float %local.reduce55) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %589 = insertelement <4 x float> %587, float %588, i64 3, !dbg !44
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %589) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02238 = extractelement <4 x float> %math.result390, i64 0, !dbg !44
  %.cvtpack.f12239 = extractelement <4 x float> %math.result390, i64 1, !dbg !44
  %.cvtpack.f22240 = extractelement <4 x float> %math.result390, i64 2, !dbg !44
  %.cvtpack.f32241 = extractelement <4 x float> %math.result390, i64 3, !dbg !44
  %.cvtpack.p02242 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02238, float %.cvtpack.f12239) #12, !dbg !44
  %.cvtpack.p12243 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22240, float %.cvtpack.f32241) #12, !dbg !44
  %.cvtpack.v02244 = insertelement <2 x i32> poison, i32 %.cvtpack.p02242, i64 0, !dbg !44
  %.cvtpack.v12245 = insertelement <2 x i32> %.cvtpack.v02244, i32 %.cvtpack.p12243, i64 1, !dbg !44
  %590 = bitcast <2 x i32> %.cvtpack.v12245 to <4 x half>, !dbg !44
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %590) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %569, <4 x float> %567) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %580, <4 x float> %568) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %593 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi104), !dbg !40
  %local.reduce87 = fadd float %594, %k0.sum.anchor2000, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2008 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !44
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi106), !dbg !40
  %local.reduce88 = fadd float %596, %k0.sum.anchor2001, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2009 = tail call float asm sideeffect "", "=v,0"(float %local.reduce88) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %597 = insertelement <4 x float> %595, float %596, i64 1, !dbg !44
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi108), !dbg !40
  %local.reduce89 = fadd float %598, %k0.sum.anchor2002, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2010 = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %599 = insertelement <4 x float> %597, float %598, i64 2, !dbg !44
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi110), !dbg !40
  %local.reduce90 = fadd float %600, %k0.sum.anchor2003, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2011 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> %599, float %600, i64 3, !dbg !44
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %601) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02246 = extractelement <4 x float> %math.result376, i64 0, !dbg !44
  %.cvtpack.f12247 = extractelement <4 x float> %math.result376, i64 1, !dbg !44
  %.cvtpack.f22248 = extractelement <4 x float> %math.result376, i64 2, !dbg !44
  %.cvtpack.f32249 = extractelement <4 x float> %math.result376, i64 3, !dbg !44
  %.cvtpack.p02250 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02246, float %.cvtpack.f12247) #12, !dbg !44
  %.cvtpack.p12251 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22248, float %.cvtpack.f32249) #12, !dbg !44
  %.cvtpack.v02252 = insertelement <2 x i32> poison, i32 %.cvtpack.p02250, i64 0, !dbg !44
  %.cvtpack.v12253 = insertelement <2 x i32> %.cvtpack.v02252, i32 %.cvtpack.p12251, i64 1, !dbg !44
  %602 = bitcast <2 x i32> %.cvtpack.v12253 to <4 x half>, !dbg !44
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %602) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %593, <4 x float> %579) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %604 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %604, <4 x float> %581) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi112), !dbg !40
  %local.reduce56 = fadd float %606, %k0.sum.anchor2004, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2012 = tail call float asm sideeffect "", "=v,0"(float %local.reduce56) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %607 = insertelement <4 x float> poison, float %606, i64 0, !dbg !44
  %608 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi114), !dbg !40
  %local.reduce57 = fadd float %608, %k0.sum.anchor2005, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2013 = tail call float asm sideeffect "", "=v,0"(float %local.reduce57) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %609 = insertelement <4 x float> %607, float %608, i64 1, !dbg !44
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi116), !dbg !40
  %local.reduce58 = fadd float %610, %k0.sum.anchor2006, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2014 = tail call float asm sideeffect "", "=v,0"(float %local.reduce58) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> %609, float %610, i64 2, !dbg !44
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi118), !dbg !40
  %local.reduce59 = fadd float %612, %k0.sum.anchor2007, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2015 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %613 = insertelement <4 x float> %611, float %612, i64 3, !dbg !44
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %613) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02254 = extractelement <4 x float> %math.result392, i64 0, !dbg !44
  %.cvtpack.f12255 = extractelement <4 x float> %math.result392, i64 1, !dbg !44
  %.cvtpack.f22256 = extractelement <4 x float> %math.result392, i64 2, !dbg !44
  %.cvtpack.f32257 = extractelement <4 x float> %math.result392, i64 3, !dbg !44
  %.cvtpack.p02258 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02254, float %.cvtpack.f12255) #12, !dbg !44
  %.cvtpack.p12259 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22256, float %.cvtpack.f32257) #12, !dbg !44
  %.cvtpack.v02260 = insertelement <2 x i32> poison, i32 %.cvtpack.p02258, i64 0, !dbg !44
  %.cvtpack.v12261 = insertelement <2 x i32> %.cvtpack.v02260, i32 %.cvtpack.p12259, i64 1, !dbg !44
  %614 = bitcast <2 x i32> %.cvtpack.v12261 to <4 x half>, !dbg !44
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %614) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %593, <4 x float> %591) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %604, <4 x float> %592) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %617 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %618 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi120), !dbg !40
  %local.reduce91 = fadd float %618, %k0.sum.anchor2008, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2016 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %619 = insertelement <4 x float> poison, float %618, i64 0, !dbg !44
  %620 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi122), !dbg !40
  %local.reduce92 = fadd float %620, %k0.sum.anchor2009, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2017 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %621 = insertelement <4 x float> %619, float %620, i64 1, !dbg !44
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi124), !dbg !40
  %local.reduce93 = fadd float %622, %k0.sum.anchor2010, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2018 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %623 = insertelement <4 x float> %621, float %622, i64 2, !dbg !44
  %624 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi126), !dbg !40
  %local.reduce94 = fadd float %624, %k0.sum.anchor2011, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2019 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %625 = insertelement <4 x float> %623, float %624, i64 3, !dbg !44
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %625) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02262 = extractelement <4 x float> %math.result378, i64 0, !dbg !44
  %.cvtpack.f12263 = extractelement <4 x float> %math.result378, i64 1, !dbg !44
  %.cvtpack.f22264 = extractelement <4 x float> %math.result378, i64 2, !dbg !44
  %.cvtpack.f32265 = extractelement <4 x float> %math.result378, i64 3, !dbg !44
  %.cvtpack.p02266 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02262, float %.cvtpack.f12263) #12, !dbg !44
  %.cvtpack.p12267 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22264, float %.cvtpack.f32265) #12, !dbg !44
  %.cvtpack.v02268 = insertelement <2 x i32> poison, i32 %.cvtpack.p02266, i64 0, !dbg !44
  %.cvtpack.v12269 = insertelement <2 x i32> %.cvtpack.v02268, i32 %.cvtpack.p12267, i64 1, !dbg !44
  %626 = bitcast <2 x i32> %.cvtpack.v12269 to <4 x half>, !dbg !44
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %626) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %617, <4 x float> %603) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %628 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %628, <4 x float> %605) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi128), !dbg !40
  %local.reduce60 = fadd float %630, %k0.sum.anchor2012, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2020 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %631 = insertelement <4 x float> poison, float %630, i64 0, !dbg !44
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi130), !dbg !40
  %local.reduce61 = fadd float %632, %k0.sum.anchor2013, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2021 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %631, float %632, i64 1, !dbg !44
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi132), !dbg !40
  %local.reduce62 = fadd float %634, %k0.sum.anchor2014, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2022 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %635 = insertelement <4 x float> %633, float %634, i64 2, !dbg !44
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi134), !dbg !40
  %local.reduce63 = fadd float %636, %k0.sum.anchor2015, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2023 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %637 = insertelement <4 x float> %635, float %636, i64 3, !dbg !44
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %637) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02270 = extractelement <4 x float> %math.result394, i64 0, !dbg !44
  %.cvtpack.f12271 = extractelement <4 x float> %math.result394, i64 1, !dbg !44
  %.cvtpack.f22272 = extractelement <4 x float> %math.result394, i64 2, !dbg !44
  %.cvtpack.f32273 = extractelement <4 x float> %math.result394, i64 3, !dbg !44
  %.cvtpack.p02274 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02270, float %.cvtpack.f12271) #12, !dbg !44
  %.cvtpack.p12275 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22272, float %.cvtpack.f32273) #12, !dbg !44
  %.cvtpack.v02276 = insertelement <2 x i32> poison, i32 %.cvtpack.p02274, i64 0, !dbg !44
  %.cvtpack.v12277 = insertelement <2 x i32> %.cvtpack.v02276, i32 %.cvtpack.p12275, i64 1, !dbg !44
  %638 = bitcast <2 x i32> %.cvtpack.v12277 to <4 x half>, !dbg !44
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %638) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %617, <4 x float> %615) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %640 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %628, <4 x float> %616) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %641 = bitcast <4 x i32> %472 to <8 x half>, !dbg !42
  %642 = shufflevector <8 x half> %641, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi136), !dbg !40
  %local.reduce95 = fadd float %643, %k0.sum.anchor2016, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2024 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %644 = insertelement <4 x float> poison, float %643, i64 0, !dbg !44
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi138), !dbg !40
  %local.reduce96 = fadd float %645, %k0.sum.anchor2017, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2025 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce99 = fadd float %k0.sum.anchor2024, %k0.sum.anchor2025, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2026 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> %644, float %645, i64 1, !dbg !44
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi140), !dbg !40
  %local.reduce97 = fadd float %647, %k0.sum.anchor2018, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2027 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %648 = insertelement <4 x float> %646, float %647, i64 2, !dbg !44
  %649 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi142), !dbg !40
  %local.reduce98 = fadd float %649, %k0.sum.anchor2019, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2028 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce100 = fadd float %k0.sum.anchor2027, %k0.sum.anchor2028, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2029 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %650 = fadd float %k0.sum.anchor2026, %k0.sum.anchor2029, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2030 = tail call float asm sideeffect "", "=v,0"(float %650) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %651 = fadd float %447, %k0.sum.anchor2030, !dbg !45
  %k0.sum.anchor2031 = tail call float asm sideeffect "", "=v,0"(float %651) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %652 = insertelement <1 x float> poison, float %k0.sum.anchor2031, i64 0, !dbg !12
  %653 = insertvalue [2 x <1 x float>] poison, <1 x float> %652, 0, !dbg !12
  %654 = insertelement <4 x float> %648, float %649, i64 3, !dbg !44
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02278 = extractelement <4 x float> %math.result380, i64 0, !dbg !44
  %.cvtpack.f12279 = extractelement <4 x float> %math.result380, i64 1, !dbg !44
  %.cvtpack.f22280 = extractelement <4 x float> %math.result380, i64 2, !dbg !44
  %.cvtpack.f32281 = extractelement <4 x float> %math.result380, i64 3, !dbg !44
  %.cvtpack.p02282 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02278, float %.cvtpack.f12279) #12, !dbg !44
  %.cvtpack.p12283 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22280, float %.cvtpack.f32281) #12, !dbg !44
  %.cvtpack.v02284 = insertelement <2 x i32> poison, i32 %.cvtpack.p02282, i64 0, !dbg !44
  %.cvtpack.v12285 = insertelement <2 x i32> %.cvtpack.v02284, i32 %.cvtpack.p12283, i64 1, !dbg !44
  %655 = bitcast <2 x i32> %.cvtpack.v12285 to <4 x half>, !dbg !44
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %656 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %642, <4 x float> %627) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %657 = shufflevector <8 x half> %641, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %657, <4 x float> %629) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi144), !dbg !40
  %local.reduce64 = fadd float %659, %k0.sum.anchor2020, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2032 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %660 = insertelement <4 x float> poison, float %659, i64 0, !dbg !44
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi146), !dbg !40
  %local.reduce65 = fadd float %661, %k0.sum.anchor2021, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2033 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce68 = fadd float %k0.sum.anchor2032, %k0.sum.anchor2033, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2034 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %660, float %661, i64 1, !dbg !44
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi148), !dbg !40
  %local.reduce66 = fadd float %663, %k0.sum.anchor2022, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2035 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %664 = insertelement <4 x float> %662, float %663, i64 2, !dbg !44
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi150), !dbg !40
  %local.reduce67 = fadd float %665, %k0.sum.anchor2023, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2036 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce69 = fadd float %k0.sum.anchor2035, %k0.sum.anchor2036, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2037 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %666 = fadd float %k0.sum.anchor2034, %k0.sum.anchor2037, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2038 = tail call float asm sideeffect "", "=v,0"(float %666) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %667 = fadd float %449, %k0.sum.anchor2038, !dbg !45
  %k0.sum.anchor2039 = tail call float asm sideeffect "", "=v,0"(float %667) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %668 = insertelement <1 x float> poison, float %k0.sum.anchor2039, i64 0, !dbg !12
  %669 = insertvalue [2 x <1 x float>] %653, <1 x float> %668, 1, !dbg !12
  %670 = insertelement <4 x float> %664, float %665, i64 3, !dbg !44
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02286 = extractelement <4 x float> %math.result396, i64 0, !dbg !44
  %.cvtpack.f12287 = extractelement <4 x float> %math.result396, i64 1, !dbg !44
  %.cvtpack.f22288 = extractelement <4 x float> %math.result396, i64 2, !dbg !44
  %.cvtpack.f32289 = extractelement <4 x float> %math.result396, i64 3, !dbg !44
  %.cvtpack.p02290 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02286, float %.cvtpack.f12287) #12, !dbg !44
  %.cvtpack.p12291 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22288, float %.cvtpack.f32289) #12, !dbg !44
  %.cvtpack.v02292 = insertelement <2 x i32> poison, i32 %.cvtpack.p02290, i64 0, !dbg !44
  %.cvtpack.v12293 = insertelement <2 x i32> %.cvtpack.v02292, i32 %.cvtpack.p12291, i64 1, !dbg !44
  %671 = bitcast <2 x i32> %.cvtpack.v12293 to <4 x half>, !dbg !44
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %672 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %642, <4 x float> %639) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %657, <4 x float> %640) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %674 = extractvalue [2 x <8 x float>] %lazy.state2172, 0, !dbg !41
  %675 = shufflevector <8 x float> %674, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %676 = shufflevector <8 x float> %674, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %677 = extractvalue [2 x <8 x float>] %lazy.state2172, 1, !dbg !41
  %678 = shufflevector <8 x float> %677, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %679 = shufflevector <8 x float> %677, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1441 = add nsw i64 %address.local916, %205, !dbg !38
  %vdb.address1442 = trunc nsw i64 %vdb.address1441 to i32, !dbg !38
  %vdb.address1443 = shl nsw i32 %vdb.address1442, 1, !dbg !38
  %vdb.address1448 = trunc nuw nsw i64 %205 to i32, !dbg !38
  %vdb.address1449 = or disjoint i32 %address.local923, %vdb.address1448, !dbg !38
  %vdb.address1450 = shl nuw nsw i32 %vdb.address1449, 1, !dbg !38
  %680 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1443, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %681 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %vdb.address1450, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local932, i64 8192, !dbg !42
  %682 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %683 = bitcast <4 x i32> %682 to <8 x half>, !dbg !42
  %vdb.read1434 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8448, !dbg !42
  %684 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %685 = bitcast <4 x i32> %684 to <8 x half>, !dbg !42
  %vdb.read1435 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8704, !dbg !42
  %686 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %687 = bitcast <4 x i32> %686 to <8 x half>, !dbg !42
  %vdb.read1436 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8960, !dbg !42
  %688 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %689 = bitcast <4 x i32> %688 to <8 x half>, !dbg !42
  %vdb.read1437 = getelementptr i8, ptr addrspace(3) %address.local932, i64 12288, !dbg !42
  %690 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %691 = bitcast <4 x i32> %690 to <8 x half>, !dbg !42
  %vdb.read1438 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12544, !dbg !42
  %692 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %693 = bitcast <4 x i32> %692 to <8 x half>, !dbg !42
  %vdb.read1439 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12800, !dbg !42
  %694 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %695 = bitcast <4 x i32> %694 to <8 x half>, !dbg !42
  %vdb.read1440 = getelementptr i8, ptr addrspace(3) %address.local941, i64 13056, !dbg !42
  %696 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %697 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %697, <4 x float> %675) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %699 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %699, <4 x float> %676) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %697, <4 x float> %678) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %699, <4 x float> %679) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %703 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %704 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %703, <4 x float> %698) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %705 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %706 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %705, <4 x float> %700) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %707 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %703, <4 x float> %701) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %708 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %705, <4 x float> %702) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %709 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %710 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %709, <4 x float> %704) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %711 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %712 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %711, <4 x float> %706) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %713 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %709, <4 x float> %707) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %714 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %711, <4 x float> %708) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %715 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %715, <4 x float> %710) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %717 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %718 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %717, <4 x float> %712) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %719 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %715, <4 x float> %713) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %720 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %717, <4 x float> %714) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %721 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %722 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %721, <4 x float> %716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %723 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %724 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %723, <4 x float> %718) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %721, <4 x float> %719) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %726 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %723, <4 x float> %720) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %727 = shufflevector <8 x half> %693, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %728 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %727, <4 x float> %722) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %729 = shufflevector <8 x half> %693, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %730 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %729, <4 x float> %724) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %727, <4 x float> %725) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %729, <4 x float> %726) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %733 = shufflevector <8 x half> %695, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %733, <4 x float> %728) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %735 = shufflevector <8 x half> %695, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %735, <4 x float> %730) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %733, <4 x float> %731) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %738 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %735, <4 x float> %732) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %739 = bitcast <4 x i32> %696 to <8 x half>, !dbg !42
  %740 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %740, <4 x float> %734) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %742 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %742, <4 x float> %736) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %740, <4 x float> %737) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %742, <4 x float> %738) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %746 = extractvalue [2 x <8 x float>] %lazy.state2171, 0, !dbg !41
  %747 = shufflevector <8 x float> %746, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %748 = shufflevector <8 x float> %746, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %749 = extractvalue [2 x <8 x float>] %lazy.state2171, 1, !dbg !41
  %750 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %751 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1455 = add nsw i64 %address.local916, %258, !dbg !38
  %vdb.address1456 = trunc nsw i64 %vdb.address1455 to i32, !dbg !38
  %vdb.address1457 = shl nsw i32 %vdb.address1456, 1, !dbg !38
  %vdb.address1462 = trunc nuw nsw i64 %258 to i32, !dbg !38
  %vdb.address1463 = or disjoint i32 %address.local923, %vdb.address1462, !dbg !38
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !38
  %752 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1457, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %753 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1340) #9, !dbg !38
  %754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %755 = bitcast <4 x i32> %754 to <8 x half>, !dbg !42
  %756 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %757 = bitcast <4 x i32> %756 to <8 x half>, !dbg !42
  %758 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !42
  %760 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %761 = bitcast <4 x i32> %760 to <8 x half>, !dbg !42
  %762 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %763 = bitcast <4 x i32> %762 to <8 x half>, !dbg !42
  %764 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %765 = bitcast <4 x i32> %764 to <8 x half>, !dbg !42
  %766 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %767 = bitcast <4 x i32> %766 to <8 x half>, !dbg !42
  %768 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %769 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %769, <4 x float> %747) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %771 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %771, <4 x float> %748) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %769, <4 x float> %750) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %771, <4 x float> %751) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %775 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %775, <4 x float> %770) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %777 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %777, <4 x float> %772) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %779 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %775, <4 x float> %773) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %777, <4 x float> %774) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %781 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %781, <4 x float> %776) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %783 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %783, <4 x float> %778) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %785 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %781, <4 x float> %779) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %783, <4 x float> %780) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %787 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %788 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %787, <4 x float> %782) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %789 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %790 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %789, <4 x float> %784) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %787, <4 x float> %785) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %789, <4 x float> %786) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %793 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %793, <4 x float> %788) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %795 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %795, <4 x float> %790) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %793, <4 x float> %791) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %795, <4 x float> %792) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %799 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %799, <4 x float> %794) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %801 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %801, <4 x float> %796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %799, <4 x float> %797) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %801, <4 x float> %798) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %805 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %805, <4 x float> %800) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %807 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %807, <4 x float> %802) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %809 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %805, <4 x float> %803) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %807, <4 x float> %804) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %811 = bitcast <4 x i32> %768 to <8 x half>, !dbg !42
  %812 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %812, <4 x float> %806) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %814 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %814, <4 x float> %808) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %812, <4 x float> %809) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %814, <4 x float> %810) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %818 = extractvalue [2 x <8 x float>] %lazy.state2170, 0, !dbg !41
  %819 = shufflevector <8 x float> %818, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %820 = shufflevector <8 x float> %818, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %821 = extractvalue [2 x <8 x float>] %lazy.state2170, 1, !dbg !41
  %822 = shufflevector <8 x float> %821, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %823 = shufflevector <8 x float> %821, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %824 = add nuw nsw i64 %124, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %124, 31, !dbg !42
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !42

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %824, 14, !dbg !14
  %k0.next.2050 = add nuw nsw i64 %k0.next., %k0.prime.2040, !dbg !14
  %k0.next.kdma.source1495 = or disjoint i64 %k0.next.2050, %k0.prime.kdma.source1494, !dbg !14
  %k0.next.kdma.source1496 = trunc nuw nsw i64 %k0.next.kdma.source1495 to i32, !dbg !14
  %k0.next.kdma.source1497 = shl nuw nsw i32 %k0.next.kdma.source1496, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1497, %kdma.intra, !dbg !14
  %k0.next.2051 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1515 = or disjoint i64 %k0.next.2050, %k0.prime.kdma.source1514, !dbg !14
  %k0.next.kdma.source1516 = trunc nuw nsw i64 %k0.next.kdma.source1515 to i32, !dbg !14
  %k0.next.kdma.source1517 = shl nuw nsw i32 %k0.next.kdma.source1516, 1, !dbg !14
  %k0.next.kdma.offset1518 = or disjoint i32 %k0.next.kdma.source1517, %kdma.intra.hi, !dbg !14
  %k0.next.2052 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1519, i32 %k0.next.kdma.offset1518, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1539 = or disjoint i64 %k0.next.2050, %k0.prime.kdma.source1538, !dbg !14
  %k0.next.kdma.source1540 = trunc nuw nsw i64 %k0.next.kdma.source1539 to i32, !dbg !14
  %k0.next.kdma.source1541 = shl nuw nsw i32 %k0.next.kdma.source1540, 1, !dbg !14
  %k0.next.kdma.offset1542 = add nuw nsw i32 %k0.next.kdma.source1541, %kdma.intra, !dbg !14
  %k0.next.2053 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.next.kdma.offset1542, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1563 = or disjoint i64 %k0.next.2050, %k0.prime.kdma.source1562, !dbg !14
  %k0.next.kdma.source1564 = trunc nuw nsw i64 %k0.next.kdma.source1563 to i32, !dbg !14
  %k0.next.kdma.source1565 = shl nuw nsw i32 %k0.next.kdma.source1564, 1, !dbg !14
  %k0.next.kdma.offset1566 = add nuw nsw i32 %k0.next.kdma.source1565, %kdma.intra.hi, !dbg !14
  %k0.next.2054 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1567, i32 %k0.next.kdma.offset1566, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !42
  %827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %828 = bitcast <4 x i32> %827 to <8 x half>, !dbg !42
  %829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %830 = bitcast <4 x i32> %829 to <8 x half>, !dbg !42
  %831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %832 = bitcast <4 x i32> %831 to <8 x half>, !dbg !42
  %833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !42
  %835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %836 = bitcast <4 x i32> %835 to <8 x half>, !dbg !42
  %837 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %838 = bitcast <4 x i32> %837 to <8 x half>, !dbg !42
  %839 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %840 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %840, <4 x float> %819) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %842 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %842, <4 x float> %820) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %840, <4 x float> %822) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %842, <4 x float> %823) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %846 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %846, <4 x float> %841) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %848 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %848, <4 x float> %843) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %846, <4 x float> %844) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %848, <4 x float> %845) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %852 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %852, <4 x float> %847) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %854 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %854, <4 x float> %849) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %852, <4 x float> %850) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %854, <4 x float> %851) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %858 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %858, <4 x float> %853) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %860 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %860, <4 x float> %855) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %858, <4 x float> %856) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %860, <4 x float> %857) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %864 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %864, <4 x float> %859) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %866 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %866, <4 x float> %861) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %864, <4 x float> %862) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %866, <4 x float> %863) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %870 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %870, <4 x float> %865) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %872 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %872, <4 x float> %867) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %870, <4 x float> %868) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %872, <4 x float> %869) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %876 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %876, <4 x float> %871) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %878 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %878, <4 x float> %873) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %876, <4 x float> %874) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %878, <4 x float> %875) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %882 = bitcast <4 x i32> %839 to <8 x half>, !dbg !42
  %883 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %883, <4 x float> %877) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %885 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %885, <4 x float> %879) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %883, <4 x float> %880) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %885, <4 x float> %881) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %889 = shufflevector <4 x float> %656, <4 x float> %658, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %890 = insertvalue [2 x <8 x float>] poison, <8 x float> %889, 0, !dbg !12
  %891 = shufflevector <4 x float> %672, <4 x float> %673, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %892 = insertvalue [2 x <8 x float>] %890, <8 x float> %891, 1, !dbg !12
  %893 = shufflevector <4 x float> %741, <4 x float> %743, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %894 = insertvalue [2 x <8 x float>] poison, <8 x float> %893, 0, !dbg !12
  %895 = shufflevector <4 x float> %744, <4 x float> %745, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %896 = insertvalue [2 x <8 x float>] %894, <8 x float> %895, 1, !dbg !12
  %897 = shufflevector <4 x float> %813, <4 x float> %815, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %898 = insertvalue [2 x <8 x float>] poison, <8 x float> %897, 0, !dbg !12
  %899 = shufflevector <4 x float> %816, <4 x float> %817, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %900 = insertvalue [2 x <8 x float>] %898, <8 x float> %899, 1, !dbg !12
  %901 = shufflevector <4 x float> %884, <4 x float> %886, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %902 = insertvalue [2 x <8 x float>] poison, <8 x float> %901, 0, !dbg !12
  %903 = shufflevector <4 x float> %887, <4 x float> %888, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %904 = insertvalue [2 x <8 x float>] %902, <8 x float> %903, 1, !dbg !12
  %905 = icmp samesign ult i64 %124, 31, !dbg !12
  br i1 %905, label %117, label %906, !dbg !12

906:                                              ; preds = %k0.join
  %907 = shl nuw nsw i32 %8, 2, !dbg !12
  %908 = xor i32 %907, 64, !dbg !12
  %909 = bitcast float %k0.sum.anchor2039 to i32, !dbg !12
  %lane.sum.shuffle170 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %908, i32 %909), !dbg !12
  %910 = bitcast i32 %lane.sum.shuffle170 to float, !dbg !12
  %lane.sum.final171 = fadd float %k0.sum.anchor2039, %910, !dbg !12
  %911 = xor i32 %907, 128, !dbg !12
  %912 = bitcast float %lane.sum.final171 to i32, !dbg !12
  %lane.sum.shuffle172 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %911, i32 %912), !dbg !12
  %913 = bitcast i32 %lane.sum.shuffle172 to float, !dbg !12
  %lane.sum.final173 = fadd float %lane.sum.final171, %913, !dbg !12
  %914 = insertelement <1 x float> poison, float %lane.sum.final173, i64 0, !dbg !12
  %915 = bitcast float %k0.sum.anchor2031 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %908, i32 %915), !dbg !12
  %916 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor2031, %916, !dbg !12
  %917 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle164 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %911, i32 %917), !dbg !12
  %918 = bitcast i32 %lane.sum.shuffle164 to float, !dbg !12
  %lane.sum.final165 = fadd float %lane.sum.final, %918, !dbg !12
  %919 = insertelement <1 x float> poison, float %lane.sum.final165, i64 0, !dbg !12
  %920 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %920, 0, !dbg !12
  %921 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %921, 0, !dbg !12
  %922 = extractelement <4 x float> %884, i64 0, !dbg !12
  %923 = extractelement <4 x float> %884, i64 1, !dbg !12
  %924 = extractelement <4 x float> %884, i64 2, !dbg !12
  %925 = extractelement <4 x float> %884, i64 3, !dbg !12
  %926 = select i1 %.not, float %923, float %922, !dbg !12
  %927 = bitcast float %926 to i32, !dbg !12
  %928 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %927), !dbg !12
  %929 = bitcast i32 %928 to float, !dbg !12
  %930 = select i1 %.not, float %922, float %929, !dbg !12
  %931 = select i1 %.not, float %929, float %923, !dbg !12
  %932 = select i1 %.not, float %925, float %924, !dbg !12
  %933 = bitcast float %932 to i32, !dbg !12
  %934 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %933), !dbg !12
  %935 = bitcast i32 %934 to float, !dbg !12
  %936 = select i1 %.not, float %924, float %935, !dbg !12
  %937 = select i1 %.not, float %935, float %925, !dbg !12
  %938 = select i1 %.not40, float %936, float %930, !dbg !12
  %939 = bitcast float %938 to i32, !dbg !12
  %940 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %939), !dbg !12
  %941 = bitcast i32 %940 to float, !dbg !12
  %942 = select i1 %.not40, float %937, float %931, !dbg !12
  %943 = bitcast float %942 to i32, !dbg !12
  %944 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %943), !dbg !12
  %945 = bitcast i32 %944 to float, !dbg !12
  %946 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %947 = shufflevector <4 x i1> %946, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %948 = insertelement <4 x float> poison, float %930, i64 0, !dbg !12
  %949 = insertelement <4 x float> %948, float %931, i64 1, !dbg !12
  %950 = insertelement <4 x float> %949, float %941, i64 2, !dbg !12
  %951 = insertelement <4 x float> %950, float %945, i64 3, !dbg !12
  %952 = insertelement <4 x float> poison, float %941, i64 0, !dbg !12
  %953 = insertelement <4 x float> %952, float %945, i64 1, !dbg !12
  %954 = insertelement <4 x float> %953, float %936, i64 2, !dbg !12
  %955 = insertelement <4 x float> %954, float %937, i64 3, !dbg !12
  %956 = select <4 x i1> %947, <4 x float> %951, <4 x float> %955, !dbg !12
  %957 = extractelement <4 x float> %886, i64 0, !dbg !12
  %958 = extractelement <4 x float> %886, i64 1, !dbg !12
  %959 = extractelement <4 x float> %886, i64 2, !dbg !12
  %960 = extractelement <4 x float> %886, i64 3, !dbg !12
  %961 = select i1 %.not, float %958, float %957, !dbg !12
  %962 = bitcast float %961 to i32, !dbg !12
  %963 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %962), !dbg !12
  %964 = bitcast i32 %963 to float, !dbg !12
  %965 = select i1 %.not, float %957, float %964, !dbg !12
  %966 = select i1 %.not, float %964, float %958, !dbg !12
  %967 = select i1 %.not, float %960, float %959, !dbg !12
  %968 = bitcast float %967 to i32, !dbg !12
  %969 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %968), !dbg !12
  %970 = bitcast i32 %969 to float, !dbg !12
  %971 = select i1 %.not, float %959, float %970, !dbg !12
  %972 = select i1 %.not, float %970, float %960, !dbg !12
  %973 = select i1 %.not40, float %971, float %965, !dbg !12
  %974 = bitcast float %973 to i32, !dbg !12
  %975 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %974), !dbg !12
  %976 = bitcast i32 %975 to float, !dbg !12
  %977 = select i1 %.not40, float %972, float %966, !dbg !12
  %978 = bitcast float %977 to i32, !dbg !12
  %979 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %978), !dbg !12
  %980 = bitcast i32 %979 to float, !dbg !12
  %981 = insertelement <4 x float> poison, float %965, i64 0, !dbg !12
  %982 = insertelement <4 x float> %981, float %966, i64 1, !dbg !12
  %983 = insertelement <4 x float> %982, float %976, i64 2, !dbg !12
  %984 = insertelement <4 x float> %983, float %980, i64 3, !dbg !12
  %985 = insertelement <4 x float> poison, float %976, i64 0, !dbg !12
  %986 = insertelement <4 x float> %985, float %980, i64 1, !dbg !12
  %987 = insertelement <4 x float> %986, float %971, i64 2, !dbg !12
  %988 = insertelement <4 x float> %987, float %972, i64 3, !dbg !12
  %989 = select <4 x i1> %947, <4 x float> %984, <4 x float> %988, !dbg !12
  %990 = extractelement <4 x float> %887, i64 0, !dbg !12
  %991 = extractelement <4 x float> %887, i64 1, !dbg !12
  %992 = extractelement <4 x float> %887, i64 2, !dbg !12
  %993 = extractelement <4 x float> %887, i64 3, !dbg !12
  %994 = select i1 %.not, float %991, float %990, !dbg !12
  %995 = bitcast float %994 to i32, !dbg !12
  %996 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %995), !dbg !12
  %997 = bitcast i32 %996 to float, !dbg !12
  %998 = select i1 %.not, float %990, float %997, !dbg !12
  %999 = select i1 %.not, float %997, float %991, !dbg !12
  %1000 = select i1 %.not, float %993, float %992, !dbg !12
  %1001 = bitcast float %1000 to i32, !dbg !12
  %1002 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1001), !dbg !12
  %1003 = bitcast i32 %1002 to float, !dbg !12
  %1004 = select i1 %.not, float %992, float %1003, !dbg !12
  %1005 = select i1 %.not, float %1003, float %993, !dbg !12
  %1006 = select i1 %.not40, float %1004, float %998, !dbg !12
  %1007 = bitcast float %1006 to i32, !dbg !12
  %1008 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1007), !dbg !12
  %1009 = bitcast i32 %1008 to float, !dbg !12
  %1010 = select i1 %.not40, float %1005, float %999, !dbg !12
  %1011 = bitcast float %1010 to i32, !dbg !12
  %1012 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1011), !dbg !12
  %1013 = bitcast i32 %1012 to float, !dbg !12
  %1014 = insertelement <4 x float> poison, float %998, i64 0, !dbg !12
  %1015 = insertelement <4 x float> %1014, float %999, i64 1, !dbg !12
  %1016 = insertelement <4 x float> %1015, float %1009, i64 2, !dbg !12
  %1017 = insertelement <4 x float> %1016, float %1013, i64 3, !dbg !12
  %1018 = insertelement <4 x float> poison, float %1009, i64 0, !dbg !12
  %1019 = insertelement <4 x float> %1018, float %1013, i64 1, !dbg !12
  %1020 = insertelement <4 x float> %1019, float %1004, i64 2, !dbg !12
  %1021 = insertelement <4 x float> %1020, float %1005, i64 3, !dbg !12
  %1022 = select <4 x i1> %947, <4 x float> %1017, <4 x float> %1021, !dbg !12
  %1023 = extractelement <4 x float> %888, i64 0, !dbg !12
  %1024 = extractelement <4 x float> %888, i64 1, !dbg !12
  %1025 = extractelement <4 x float> %888, i64 2, !dbg !12
  %1026 = extractelement <4 x float> %888, i64 3, !dbg !12
  %1027 = select i1 %.not, float %1024, float %1023, !dbg !12
  %1028 = bitcast float %1027 to i32, !dbg !12
  %1029 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1028), !dbg !12
  %1030 = bitcast i32 %1029 to float, !dbg !12
  %1031 = select i1 %.not, float %1023, float %1030, !dbg !12
  %1032 = select i1 %.not, float %1030, float %1024, !dbg !12
  %1033 = select i1 %.not, float %1026, float %1025, !dbg !12
  %1034 = bitcast float %1033 to i32, !dbg !12
  %1035 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1034), !dbg !12
  %1036 = bitcast i32 %1035 to float, !dbg !12
  %1037 = select i1 %.not, float %1025, float %1036, !dbg !12
  %1038 = select i1 %.not, float %1036, float %1026, !dbg !12
  %1039 = select i1 %.not40, float %1037, float %1031, !dbg !12
  %1040 = bitcast float %1039 to i32, !dbg !12
  %1041 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1040), !dbg !12
  %1042 = bitcast i32 %1041 to float, !dbg !12
  %1043 = select i1 %.not40, float %1038, float %1032, !dbg !12
  %1044 = bitcast float %1043 to i32, !dbg !12
  %1045 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1044), !dbg !12
  %1046 = bitcast i32 %1045 to float, !dbg !12
  %1047 = insertelement <4 x float> poison, float %1031, i64 0, !dbg !12
  %1048 = insertelement <4 x float> %1047, float %1032, i64 1, !dbg !12
  %1049 = insertelement <4 x float> %1048, float %1042, i64 2, !dbg !12
  %1050 = insertelement <4 x float> %1049, float %1046, i64 3, !dbg !12
  %1051 = insertelement <4 x float> poison, float %1042, i64 0, !dbg !12
  %1052 = insertelement <4 x float> %1051, float %1046, i64 1, !dbg !12
  %1053 = insertelement <4 x float> %1052, float %1037, i64 2, !dbg !12
  %1054 = insertelement <4 x float> %1053, float %1038, i64 3, !dbg !12
  %1055 = select <4 x i1> %947, <4 x float> %1050, <4 x float> %1054, !dbg !12
  %1056 = extractelement <4 x float> %813, i64 0, !dbg !12
  %1057 = extractelement <4 x float> %813, i64 1, !dbg !12
  %1058 = extractelement <4 x float> %813, i64 2, !dbg !12
  %1059 = extractelement <4 x float> %813, i64 3, !dbg !12
  %1060 = select i1 %.not, float %1057, float %1056, !dbg !12
  %1061 = bitcast float %1060 to i32, !dbg !12
  %1062 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1061), !dbg !12
  %1063 = bitcast i32 %1062 to float, !dbg !12
  %1064 = select i1 %.not, float %1056, float %1063, !dbg !12
  %1065 = select i1 %.not, float %1063, float %1057, !dbg !12
  %1066 = select i1 %.not, float %1059, float %1058, !dbg !12
  %1067 = bitcast float %1066 to i32, !dbg !12
  %1068 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1067), !dbg !12
  %1069 = bitcast i32 %1068 to float, !dbg !12
  %1070 = select i1 %.not, float %1058, float %1069, !dbg !12
  %1071 = select i1 %.not, float %1069, float %1059, !dbg !12
  %1072 = select i1 %.not40, float %1070, float %1064, !dbg !12
  %1073 = bitcast float %1072 to i32, !dbg !12
  %1074 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1073), !dbg !12
  %1075 = bitcast i32 %1074 to float, !dbg !12
  %1076 = select i1 %.not40, float %1071, float %1065, !dbg !12
  %1077 = bitcast float %1076 to i32, !dbg !12
  %1078 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1077), !dbg !12
  %1079 = bitcast i32 %1078 to float, !dbg !12
  %1080 = insertelement <4 x float> poison, float %1064, i64 0, !dbg !12
  %1081 = insertelement <4 x float> %1080, float %1065, i64 1, !dbg !12
  %1082 = insertelement <4 x float> %1081, float %1075, i64 2, !dbg !12
  %1083 = insertelement <4 x float> %1082, float %1079, i64 3, !dbg !12
  %1084 = insertelement <4 x float> poison, float %1075, i64 0, !dbg !12
  %1085 = insertelement <4 x float> %1084, float %1079, i64 1, !dbg !12
  %1086 = insertelement <4 x float> %1085, float %1070, i64 2, !dbg !12
  %1087 = insertelement <4 x float> %1086, float %1071, i64 3, !dbg !12
  %1088 = select <4 x i1> %947, <4 x float> %1083, <4 x float> %1087, !dbg !12
  %1089 = extractelement <4 x float> %815, i64 0, !dbg !12
  %1090 = extractelement <4 x float> %815, i64 1, !dbg !12
  %1091 = extractelement <4 x float> %815, i64 2, !dbg !12
  %1092 = extractelement <4 x float> %815, i64 3, !dbg !12
  %1093 = select i1 %.not, float %1090, float %1089, !dbg !12
  %1094 = bitcast float %1093 to i32, !dbg !12
  %1095 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1094), !dbg !12
  %1096 = bitcast i32 %1095 to float, !dbg !12
  %1097 = select i1 %.not, float %1089, float %1096, !dbg !12
  %1098 = select i1 %.not, float %1096, float %1090, !dbg !12
  %1099 = select i1 %.not, float %1092, float %1091, !dbg !12
  %1100 = bitcast float %1099 to i32, !dbg !12
  %1101 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1100), !dbg !12
  %1102 = bitcast i32 %1101 to float, !dbg !12
  %1103 = select i1 %.not, float %1091, float %1102, !dbg !12
  %1104 = select i1 %.not, float %1102, float %1092, !dbg !12
  %1105 = select i1 %.not40, float %1103, float %1097, !dbg !12
  %1106 = bitcast float %1105 to i32, !dbg !12
  %1107 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1106), !dbg !12
  %1108 = bitcast i32 %1107 to float, !dbg !12
  %1109 = select i1 %.not40, float %1104, float %1098, !dbg !12
  %1110 = bitcast float %1109 to i32, !dbg !12
  %1111 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1110), !dbg !12
  %1112 = bitcast i32 %1111 to float, !dbg !12
  %1113 = insertelement <4 x float> poison, float %1097, i64 0, !dbg !12
  %1114 = insertelement <4 x float> %1113, float %1098, i64 1, !dbg !12
  %1115 = insertelement <4 x float> %1114, float %1108, i64 2, !dbg !12
  %1116 = insertelement <4 x float> %1115, float %1112, i64 3, !dbg !12
  %1117 = insertelement <4 x float> poison, float %1108, i64 0, !dbg !12
  %1118 = insertelement <4 x float> %1117, float %1112, i64 1, !dbg !12
  %1119 = insertelement <4 x float> %1118, float %1103, i64 2, !dbg !12
  %1120 = insertelement <4 x float> %1119, float %1104, i64 3, !dbg !12
  %1121 = select <4 x i1> %947, <4 x float> %1116, <4 x float> %1120, !dbg !12
  %1122 = extractelement <4 x float> %816, i64 0, !dbg !12
  %1123 = extractelement <4 x float> %816, i64 1, !dbg !12
  %1124 = extractelement <4 x float> %816, i64 2, !dbg !12
  %1125 = extractelement <4 x float> %816, i64 3, !dbg !12
  %1126 = select i1 %.not, float %1123, float %1122, !dbg !12
  %1127 = bitcast float %1126 to i32, !dbg !12
  %1128 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1127), !dbg !12
  %1129 = bitcast i32 %1128 to float, !dbg !12
  %1130 = select i1 %.not, float %1122, float %1129, !dbg !12
  %1131 = select i1 %.not, float %1129, float %1123, !dbg !12
  %1132 = select i1 %.not, float %1125, float %1124, !dbg !12
  %1133 = bitcast float %1132 to i32, !dbg !12
  %1134 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1133), !dbg !12
  %1135 = bitcast i32 %1134 to float, !dbg !12
  %1136 = select i1 %.not, float %1124, float %1135, !dbg !12
  %1137 = select i1 %.not, float %1135, float %1125, !dbg !12
  %1138 = select i1 %.not40, float %1136, float %1130, !dbg !12
  %1139 = bitcast float %1138 to i32, !dbg !12
  %1140 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1139), !dbg !12
  %1141 = bitcast i32 %1140 to float, !dbg !12
  %1142 = select i1 %.not40, float %1137, float %1131, !dbg !12
  %1143 = bitcast float %1142 to i32, !dbg !12
  %1144 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1143), !dbg !12
  %1145 = bitcast i32 %1144 to float, !dbg !12
  %1146 = insertelement <4 x float> poison, float %1130, i64 0, !dbg !12
  %1147 = insertelement <4 x float> %1146, float %1131, i64 1, !dbg !12
  %1148 = insertelement <4 x float> %1147, float %1141, i64 2, !dbg !12
  %1149 = insertelement <4 x float> %1148, float %1145, i64 3, !dbg !12
  %1150 = insertelement <4 x float> poison, float %1141, i64 0, !dbg !12
  %1151 = insertelement <4 x float> %1150, float %1145, i64 1, !dbg !12
  %1152 = insertelement <4 x float> %1151, float %1136, i64 2, !dbg !12
  %1153 = insertelement <4 x float> %1152, float %1137, i64 3, !dbg !12
  %1154 = select <4 x i1> %947, <4 x float> %1149, <4 x float> %1153, !dbg !12
  %1155 = extractelement <4 x float> %817, i64 0, !dbg !12
  %1156 = extractelement <4 x float> %817, i64 1, !dbg !12
  %1157 = extractelement <4 x float> %817, i64 2, !dbg !12
  %1158 = extractelement <4 x float> %817, i64 3, !dbg !12
  %1159 = select i1 %.not, float %1156, float %1155, !dbg !12
  %1160 = bitcast float %1159 to i32, !dbg !12
  %1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1160), !dbg !12
  %1162 = bitcast i32 %1161 to float, !dbg !12
  %1163 = select i1 %.not, float %1155, float %1162, !dbg !12
  %1164 = select i1 %.not, float %1162, float %1156, !dbg !12
  %1165 = select i1 %.not, float %1158, float %1157, !dbg !12
  %1166 = bitcast float %1165 to i32, !dbg !12
  %1167 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1166), !dbg !12
  %1168 = bitcast i32 %1167 to float, !dbg !12
  %1169 = select i1 %.not, float %1157, float %1168, !dbg !12
  %1170 = select i1 %.not, float %1168, float %1158, !dbg !12
  %1171 = select i1 %.not40, float %1169, float %1163, !dbg !12
  %1172 = bitcast float %1171 to i32, !dbg !12
  %1173 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1172), !dbg !12
  %1174 = bitcast i32 %1173 to float, !dbg !12
  %1175 = select i1 %.not40, float %1170, float %1164, !dbg !12
  %1176 = bitcast float %1175 to i32, !dbg !12
  %1177 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1176), !dbg !12
  %1178 = bitcast i32 %1177 to float, !dbg !12
  %1179 = insertelement <4 x float> poison, float %1163, i64 0, !dbg !12
  %1180 = insertelement <4 x float> %1179, float %1164, i64 1, !dbg !12
  %1181 = insertelement <4 x float> %1180, float %1174, i64 2, !dbg !12
  %1182 = insertelement <4 x float> %1181, float %1178, i64 3, !dbg !12
  %1183 = insertelement <4 x float> poison, float %1174, i64 0, !dbg !12
  %1184 = insertelement <4 x float> %1183, float %1178, i64 1, !dbg !12
  %1185 = insertelement <4 x float> %1184, float %1169, i64 2, !dbg !12
  %1186 = insertelement <4 x float> %1185, float %1170, i64 3, !dbg !12
  %1187 = select <4 x i1> %947, <4 x float> %1182, <4 x float> %1186, !dbg !12
  %1188 = extractelement <4 x float> %741, i64 0, !dbg !12
  %1189 = extractelement <4 x float> %741, i64 1, !dbg !12
  %1190 = extractelement <4 x float> %741, i64 2, !dbg !12
  %1191 = extractelement <4 x float> %741, i64 3, !dbg !12
  %1192 = select i1 %.not, float %1189, float %1188, !dbg !12
  %1193 = bitcast float %1192 to i32, !dbg !12
  %1194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1193), !dbg !12
  %1195 = bitcast i32 %1194 to float, !dbg !12
  %1196 = select i1 %.not, float %1188, float %1195, !dbg !12
  %1197 = select i1 %.not, float %1195, float %1189, !dbg !12
  %1198 = select i1 %.not, float %1191, float %1190, !dbg !12
  %1199 = bitcast float %1198 to i32, !dbg !12
  %1200 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1199), !dbg !12
  %1201 = bitcast i32 %1200 to float, !dbg !12
  %1202 = select i1 %.not, float %1190, float %1201, !dbg !12
  %1203 = select i1 %.not, float %1201, float %1191, !dbg !12
  %1204 = select i1 %.not40, float %1202, float %1196, !dbg !12
  %1205 = bitcast float %1204 to i32, !dbg !12
  %1206 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1205), !dbg !12
  %1207 = bitcast i32 %1206 to float, !dbg !12
  %1208 = select i1 %.not40, float %1203, float %1197, !dbg !12
  %1209 = bitcast float %1208 to i32, !dbg !12
  %1210 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1209), !dbg !12
  %1211 = bitcast i32 %1210 to float, !dbg !12
  %1212 = insertelement <4 x float> poison, float %1196, i64 0, !dbg !12
  %1213 = insertelement <4 x float> %1212, float %1197, i64 1, !dbg !12
  %1214 = insertelement <4 x float> %1213, float %1207, i64 2, !dbg !12
  %1215 = insertelement <4 x float> %1214, float %1211, i64 3, !dbg !12
  %1216 = insertelement <4 x float> poison, float %1207, i64 0, !dbg !12
  %1217 = insertelement <4 x float> %1216, float %1211, i64 1, !dbg !12
  %1218 = insertelement <4 x float> %1217, float %1202, i64 2, !dbg !12
  %1219 = insertelement <4 x float> %1218, float %1203, i64 3, !dbg !12
  %1220 = select <4 x i1> %947, <4 x float> %1215, <4 x float> %1219, !dbg !12
  %1221 = extractelement <4 x float> %743, i64 0, !dbg !12
  %1222 = extractelement <4 x float> %743, i64 1, !dbg !12
  %1223 = extractelement <4 x float> %743, i64 2, !dbg !12
  %1224 = extractelement <4 x float> %743, i64 3, !dbg !12
  %1225 = select i1 %.not, float %1222, float %1221, !dbg !12
  %1226 = bitcast float %1225 to i32, !dbg !12
  %1227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1226), !dbg !12
  %1228 = bitcast i32 %1227 to float, !dbg !12
  %1229 = select i1 %.not, float %1221, float %1228, !dbg !12
  %1230 = select i1 %.not, float %1228, float %1222, !dbg !12
  %1231 = select i1 %.not, float %1224, float %1223, !dbg !12
  %1232 = bitcast float %1231 to i32, !dbg !12
  %1233 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1232), !dbg !12
  %1234 = bitcast i32 %1233 to float, !dbg !12
  %1235 = select i1 %.not, float %1223, float %1234, !dbg !12
  %1236 = select i1 %.not, float %1234, float %1224, !dbg !12
  %1237 = select i1 %.not40, float %1235, float %1229, !dbg !12
  %1238 = bitcast float %1237 to i32, !dbg !12
  %1239 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1238), !dbg !12
  %1240 = bitcast i32 %1239 to float, !dbg !12
  %1241 = select i1 %.not40, float %1236, float %1230, !dbg !12
  %1242 = bitcast float %1241 to i32, !dbg !12
  %1243 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1242), !dbg !12
  %1244 = bitcast i32 %1243 to float, !dbg !12
  %1245 = insertelement <4 x float> poison, float %1229, i64 0, !dbg !12
  %1246 = insertelement <4 x float> %1245, float %1230, i64 1, !dbg !12
  %1247 = insertelement <4 x float> %1246, float %1240, i64 2, !dbg !12
  %1248 = insertelement <4 x float> %1247, float %1244, i64 3, !dbg !12
  %1249 = insertelement <4 x float> poison, float %1240, i64 0, !dbg !12
  %1250 = insertelement <4 x float> %1249, float %1244, i64 1, !dbg !12
  %1251 = insertelement <4 x float> %1250, float %1235, i64 2, !dbg !12
  %1252 = insertelement <4 x float> %1251, float %1236, i64 3, !dbg !12
  %1253 = select <4 x i1> %947, <4 x float> %1248, <4 x float> %1252, !dbg !12
  %1254 = extractelement <4 x float> %744, i64 0, !dbg !12
  %1255 = extractelement <4 x float> %744, i64 1, !dbg !12
  %1256 = extractelement <4 x float> %744, i64 2, !dbg !12
  %1257 = extractelement <4 x float> %744, i64 3, !dbg !12
  %1258 = select i1 %.not, float %1255, float %1254, !dbg !12
  %1259 = bitcast float %1258 to i32, !dbg !12
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1259), !dbg !12
  %1261 = bitcast i32 %1260 to float, !dbg !12
  %1262 = select i1 %.not, float %1254, float %1261, !dbg !12
  %1263 = select i1 %.not, float %1261, float %1255, !dbg !12
  %1264 = select i1 %.not, float %1257, float %1256, !dbg !12
  %1265 = bitcast float %1264 to i32, !dbg !12
  %1266 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1265), !dbg !12
  %1267 = bitcast i32 %1266 to float, !dbg !12
  %1268 = select i1 %.not, float %1256, float %1267, !dbg !12
  %1269 = select i1 %.not, float %1267, float %1257, !dbg !12
  %1270 = select i1 %.not40, float %1268, float %1262, !dbg !12
  %1271 = bitcast float %1270 to i32, !dbg !12
  %1272 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1271), !dbg !12
  %1273 = bitcast i32 %1272 to float, !dbg !12
  %1274 = select i1 %.not40, float %1269, float %1263, !dbg !12
  %1275 = bitcast float %1274 to i32, !dbg !12
  %1276 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1275), !dbg !12
  %1277 = bitcast i32 %1276 to float, !dbg !12
  %1278 = insertelement <4 x float> poison, float %1262, i64 0, !dbg !12
  %1279 = insertelement <4 x float> %1278, float %1263, i64 1, !dbg !12
  %1280 = insertelement <4 x float> %1279, float %1273, i64 2, !dbg !12
  %1281 = insertelement <4 x float> %1280, float %1277, i64 3, !dbg !12
  %1282 = insertelement <4 x float> poison, float %1273, i64 0, !dbg !12
  %1283 = insertelement <4 x float> %1282, float %1277, i64 1, !dbg !12
  %1284 = insertelement <4 x float> %1283, float %1268, i64 2, !dbg !12
  %1285 = insertelement <4 x float> %1284, float %1269, i64 3, !dbg !12
  %1286 = select <4 x i1> %947, <4 x float> %1281, <4 x float> %1285, !dbg !12
  %1287 = extractelement <4 x float> %745, i64 0, !dbg !12
  %1288 = extractelement <4 x float> %745, i64 1, !dbg !12
  %1289 = extractelement <4 x float> %745, i64 2, !dbg !12
  %1290 = extractelement <4 x float> %745, i64 3, !dbg !12
  %1291 = select i1 %.not, float %1288, float %1287, !dbg !12
  %1292 = bitcast float %1291 to i32, !dbg !12
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1292), !dbg !12
  %1294 = bitcast i32 %1293 to float, !dbg !12
  %1295 = select i1 %.not, float %1287, float %1294, !dbg !12
  %1296 = select i1 %.not, float %1294, float %1288, !dbg !12
  %1297 = select i1 %.not, float %1290, float %1289, !dbg !12
  %1298 = bitcast float %1297 to i32, !dbg !12
  %1299 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1298), !dbg !12
  %1300 = bitcast i32 %1299 to float, !dbg !12
  %1301 = select i1 %.not, float %1289, float %1300, !dbg !12
  %1302 = select i1 %.not, float %1300, float %1290, !dbg !12
  %1303 = select i1 %.not40, float %1301, float %1295, !dbg !12
  %1304 = bitcast float %1303 to i32, !dbg !12
  %1305 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1304), !dbg !12
  %1306 = bitcast i32 %1305 to float, !dbg !12
  %1307 = select i1 %.not40, float %1302, float %1296, !dbg !12
  %1308 = bitcast float %1307 to i32, !dbg !12
  %1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1308), !dbg !12
  %1310 = bitcast i32 %1309 to float, !dbg !12
  %1311 = insertelement <4 x float> poison, float %1295, i64 0, !dbg !12
  %1312 = insertelement <4 x float> %1311, float %1296, i64 1, !dbg !12
  %1313 = insertelement <4 x float> %1312, float %1306, i64 2, !dbg !12
  %1314 = insertelement <4 x float> %1313, float %1310, i64 3, !dbg !12
  %1315 = insertelement <4 x float> poison, float %1306, i64 0, !dbg !12
  %1316 = insertelement <4 x float> %1315, float %1310, i64 1, !dbg !12
  %1317 = insertelement <4 x float> %1316, float %1301, i64 2, !dbg !12
  %1318 = insertelement <4 x float> %1317, float %1302, i64 3, !dbg !12
  %1319 = select <4 x i1> %947, <4 x float> %1314, <4 x float> %1318, !dbg !12
  %1320 = extractelement <4 x float> %656, i64 0, !dbg !12
  %1321 = extractelement <4 x float> %656, i64 1, !dbg !12
  %1322 = extractelement <4 x float> %656, i64 2, !dbg !12
  %1323 = extractelement <4 x float> %656, i64 3, !dbg !12
  %1324 = select i1 %.not, float %1321, float %1320, !dbg !12
  %1325 = bitcast float %1324 to i32, !dbg !12
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1325), !dbg !12
  %1327 = bitcast i32 %1326 to float, !dbg !12
  %1328 = select i1 %.not, float %1320, float %1327, !dbg !12
  %1329 = select i1 %.not, float %1327, float %1321, !dbg !12
  %1330 = select i1 %.not, float %1323, float %1322, !dbg !12
  %1331 = bitcast float %1330 to i32, !dbg !12
  %1332 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1331), !dbg !12
  %1333 = bitcast i32 %1332 to float, !dbg !12
  %1334 = select i1 %.not, float %1322, float %1333, !dbg !12
  %1335 = select i1 %.not, float %1333, float %1323, !dbg !12
  %1336 = select i1 %.not40, float %1334, float %1328, !dbg !12
  %1337 = bitcast float %1336 to i32, !dbg !12
  %1338 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1337), !dbg !12
  %1339 = select i1 %.not40, float %1335, float %1329, !dbg !12
  %1340 = bitcast float %1339 to i32, !dbg !12
  %1341 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1340), !dbg !12
  %1342 = extractelement <4 x float> %658, i64 0, !dbg !12
  %1343 = extractelement <4 x float> %658, i64 1, !dbg !12
  %1344 = extractelement <4 x float> %658, i64 2, !dbg !12
  %1345 = extractelement <4 x float> %658, i64 3, !dbg !12
  %1346 = select i1 %.not, float %1343, float %1342, !dbg !12
  %1347 = bitcast float %1346 to i32, !dbg !12
  %1348 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1347), !dbg !12
  %1349 = bitcast i32 %1348 to float, !dbg !12
  %1350 = select i1 %.not, float %1342, float %1349, !dbg !12
  %1351 = select i1 %.not, float %1349, float %1343, !dbg !12
  %1352 = select i1 %.not, float %1345, float %1344, !dbg !12
  %1353 = bitcast float %1352 to i32, !dbg !12
  %1354 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1353), !dbg !12
  %1355 = bitcast i32 %1354 to float, !dbg !12
  %1356 = select i1 %.not, float %1344, float %1355, !dbg !12
  %1357 = select i1 %.not, float %1355, float %1345, !dbg !12
  %1358 = select i1 %.not40, float %1356, float %1350, !dbg !12
  %1359 = bitcast float %1358 to i32, !dbg !12
  %1360 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1359), !dbg !12
  %1361 = select i1 %.not40, float %1357, float %1351, !dbg !12
  %1362 = bitcast float %1361 to i32, !dbg !12
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1362), !dbg !12
  %1364 = extractelement <4 x float> %672, i64 0, !dbg !12
  %1365 = extractelement <4 x float> %672, i64 1, !dbg !12
  %1366 = extractelement <4 x float> %672, i64 2, !dbg !12
  %1367 = extractelement <4 x float> %672, i64 3, !dbg !12
  %1368 = select i1 %.not, float %1365, float %1364, !dbg !12
  %1369 = bitcast float %1368 to i32, !dbg !12
  %1370 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1369), !dbg !12
  %1371 = bitcast i32 %1370 to float, !dbg !12
  %1372 = select i1 %.not, float %1364, float %1371, !dbg !12
  %1373 = select i1 %.not, float %1371, float %1365, !dbg !12
  %1374 = select i1 %.not, float %1367, float %1366, !dbg !12
  %1375 = bitcast float %1374 to i32, !dbg !12
  %1376 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1375), !dbg !12
  %1377 = bitcast i32 %1376 to float, !dbg !12
  %1378 = select i1 %.not, float %1366, float %1377, !dbg !12
  %1379 = select i1 %.not, float %1377, float %1367, !dbg !12
  %1380 = select i1 %.not40, float %1378, float %1372, !dbg !12
  %1381 = bitcast float %1380 to i32, !dbg !12
  %1382 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1381), !dbg !12
  %1383 = select i1 %.not40, float %1379, float %1373, !dbg !12
  %1384 = bitcast float %1383 to i32, !dbg !12
  %1385 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1384), !dbg !12
  %1386 = extractelement <4 x float> %673, i64 0, !dbg !12
  %1387 = extractelement <4 x float> %673, i64 1, !dbg !12
  %1388 = extractelement <4 x float> %673, i64 2, !dbg !12
  %1389 = extractelement <4 x float> %673, i64 3, !dbg !12
  %1390 = select i1 %.not, float %1387, float %1386, !dbg !12
  %1391 = bitcast float %1390 to i32, !dbg !12
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1391), !dbg !12
  %1393 = bitcast i32 %1392 to float, !dbg !12
  %1394 = select i1 %.not, float %1386, float %1393, !dbg !12
  %1395 = select i1 %.not, float %1393, float %1387, !dbg !12
  %1396 = select i1 %.not, float %1389, float %1388, !dbg !12
  %1397 = bitcast float %1396 to i32, !dbg !12
  %1398 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1397), !dbg !12
  %1399 = bitcast i32 %1398 to float, !dbg !12
  %1400 = select i1 %.not, float %1388, float %1399, !dbg !12
  %1401 = select i1 %.not, float %1399, float %1389, !dbg !12
  %1402 = select i1 %.not40, float %1400, float %1394, !dbg !12
  %1403 = bitcast float %1402 to i32, !dbg !12
  %1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1403), !dbg !12
  %1405 = select i1 %.not40, float %1401, float %1395, !dbg !12
  %1406 = bitcast float %1405 to i32, !dbg !12
  %1407 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1406), !dbg !12
  %1408 = shufflevector <1 x float> %919, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %1409 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1408, !dbg !46
  %1410 = shufflevector <1 x float> %914, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %1411 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1410, !dbg !46
  %1412 = bitcast i32 %1404 to float, !dbg !12
  %1413 = bitcast i32 %1407 to float, !dbg !12
  %1414 = insertelement <4 x float> poison, float %1394, i64 0, !dbg !12
  %1415 = insertelement <4 x float> %1414, float %1395, i64 1, !dbg !12
  %1416 = insertelement <4 x float> %1415, float %1412, i64 2, !dbg !12
  %1417 = insertelement <4 x float> %1416, float %1413, i64 3, !dbg !12
  %1418 = insertelement <4 x float> poison, float %1412, i64 0, !dbg !12
  %1419 = insertelement <4 x float> %1418, float %1413, i64 1, !dbg !12
  %1420 = insertelement <4 x float> %1419, float %1400, i64 2, !dbg !12
  %1421 = insertelement <4 x float> %1420, float %1401, i64 3, !dbg !12
  %1422 = select <4 x i1> %947, <4 x float> %1417, <4 x float> %1421, !dbg !12
  %1423 = fmul <4 x float> %1411, %1422, !dbg !46
  %1424 = fptrunc <4 x float> %1423 to <4 x half>, !dbg !47
  %1425 = extractelement <4 x half> %1424, i64 3, !dbg !47
  %1426 = extractelement <4 x half> %1424, i64 2, !dbg !47
  %1427 = extractelement <4 x half> %1424, i64 1, !dbg !47
  %1428 = extractelement <4 x half> %1424, i64 0, !dbg !47
  %1429 = bitcast i32 %1382 to float, !dbg !12
  %1430 = bitcast i32 %1385 to float, !dbg !12
  %1431 = insertelement <4 x float> poison, float %1372, i64 0, !dbg !12
  %1432 = insertelement <4 x float> %1431, float %1373, i64 1, !dbg !12
  %1433 = insertelement <4 x float> %1432, float %1429, i64 2, !dbg !12
  %1434 = insertelement <4 x float> %1433, float %1430, i64 3, !dbg !12
  %1435 = insertelement <4 x float> poison, float %1429, i64 0, !dbg !12
  %1436 = insertelement <4 x float> %1435, float %1430, i64 1, !dbg !12
  %1437 = insertelement <4 x float> %1436, float %1378, i64 2, !dbg !12
  %1438 = insertelement <4 x float> %1437, float %1379, i64 3, !dbg !12
  %1439 = select <4 x i1> %947, <4 x float> %1434, <4 x float> %1438, !dbg !12
  %1440 = fmul <4 x float> %1411, %1439, !dbg !46
  %1441 = fptrunc <4 x float> %1440 to <4 x half>, !dbg !47
  %1442 = extractelement <4 x half> %1441, i64 3, !dbg !47
  %1443 = extractelement <4 x half> %1441, i64 2, !dbg !47
  %1444 = extractelement <4 x half> %1441, i64 1, !dbg !47
  %1445 = extractelement <4 x half> %1441, i64 0, !dbg !47
  %1446 = bitcast i32 %1360 to float, !dbg !12
  %1447 = bitcast i32 %1363 to float, !dbg !12
  %1448 = insertelement <4 x float> poison, float %1350, i64 0, !dbg !12
  %1449 = insertelement <4 x float> %1448, float %1351, i64 1, !dbg !12
  %1450 = insertelement <4 x float> %1449, float %1446, i64 2, !dbg !12
  %1451 = insertelement <4 x float> %1450, float %1447, i64 3, !dbg !12
  %1452 = insertelement <4 x float> poison, float %1446, i64 0, !dbg !12
  %1453 = insertelement <4 x float> %1452, float %1447, i64 1, !dbg !12
  %1454 = insertelement <4 x float> %1453, float %1356, i64 2, !dbg !12
  %1455 = insertelement <4 x float> %1454, float %1357, i64 3, !dbg !12
  %1456 = select <4 x i1> %947, <4 x float> %1451, <4 x float> %1455, !dbg !12
  %1457 = fmul <4 x float> %1409, %1456, !dbg !46
  %1458 = fptrunc <4 x float> %1457 to <4 x half>, !dbg !47
  %1459 = extractelement <4 x half> %1458, i64 3, !dbg !47
  %1460 = extractelement <4 x half> %1458, i64 2, !dbg !47
  %1461 = extractelement <4 x half> %1458, i64 1, !dbg !47
  %1462 = extractelement <4 x half> %1458, i64 0, !dbg !47
  %1463 = bitcast i32 %1338 to float, !dbg !12
  %1464 = bitcast i32 %1341 to float, !dbg !12
  %1465 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !12
  %1466 = insertelement <4 x float> %1465, float %1329, i64 1, !dbg !12
  %1467 = insertelement <4 x float> %1466, float %1463, i64 2, !dbg !12
  %1468 = insertelement <4 x float> %1467, float %1464, i64 3, !dbg !12
  %1469 = insertelement <4 x float> poison, float %1463, i64 0, !dbg !12
  %1470 = insertelement <4 x float> %1469, float %1464, i64 1, !dbg !12
  %1471 = insertelement <4 x float> %1470, float %1334, i64 2, !dbg !12
  %1472 = insertelement <4 x float> %1471, float %1335, i64 3, !dbg !12
  %1473 = select <4 x i1> %947, <4 x float> %1468, <4 x float> %1472, !dbg !12
  %1474 = fmul <4 x float> %1409, %1473, !dbg !46
  %1475 = fptrunc <4 x float> %1474 to <4 x half>, !dbg !47
  %1476 = extractelement <4 x half> %1475, i64 3, !dbg !47
  %1477 = extractelement <4 x half> %1475, i64 2, !dbg !47
  %1478 = extractelement <4 x half> %1475, i64 1, !dbg !47
  %1479 = extractelement <4 x half> %1475, i64 0, !dbg !47
  %address.thread971 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local972 = zext nneg i32 %address.thread971 to i64, !dbg !9
  %1480 = lshr i64 %address.local972, 2, !dbg !48
  %address.local975 = and i64 %1480, 48, !dbg !48
  %address.local977 = and i64 %address.local972, 240, !dbg !48
  %address.local976 = add i64 %34, %address.local972, !dbg !48
  %address.local978 = sub i64 %address.local976, %address.local977, !dbg !48
  %address.local979 = add i64 %address.local978, %address.local975, !dbg !48
  %1481 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !48
  %address.local982.idx = shl nuw nsw i64 %address.local979, 8, !dbg !48
  %address.local982 = getelementptr inbounds nuw i8, ptr addrspace(1) %1481, i64 %address.local982.idx, !dbg !48
  %address.local984 = and i64 %1480, 12, !dbg !11
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local984, !dbg !48
  store half %1479, ptr addrspace(1) %address.local985, align 2, !dbg !48
  %address.local987 = or disjoint i64 %address.local984, 1, !dbg !48
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local987, !dbg !48
  store half %1478, ptr addrspace(1) %address.local988, align 2, !dbg !48
  %address.local990 = or disjoint i64 %address.local984, 2, !dbg !48
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local990, !dbg !48
  store half %1477, ptr addrspace(1) %address.local991, align 2, !dbg !48
  %address.local993 = or disjoint i64 %address.local984, 3, !dbg !48
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local993, !dbg !48
  store half %1476, ptr addrspace(1) %address.local994, align 2, !dbg !48
  %address.local996 = or disjoint i64 %address.local984, 16, !dbg !48
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local996, !dbg !48
  store half %1462, ptr addrspace(1) %address.local997, align 2, !dbg !48
  %address.local999 = or disjoint i64 %address.local984, 17, !dbg !48
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local999, !dbg !48
  store half %1461, ptr addrspace(1) %address.local1000, align 2, !dbg !48
  %address.local1002 = or disjoint i64 %address.local984, 18, !dbg !48
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1002, !dbg !48
  store half %1460, ptr addrspace(1) %address.local1003, align 2, !dbg !48
  %address.local1005 = or disjoint i64 %address.local984, 19, !dbg !48
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1005, !dbg !48
  store half %1459, ptr addrspace(1) %address.local1006, align 2, !dbg !48
  %address.local1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local982, i64 16384, !dbg !48
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local984, !dbg !48
  store half %1445, ptr addrspace(1) %address.local1011, align 2, !dbg !48
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local987, !dbg !48
  store half %1444, ptr addrspace(1) %address.local1013, align 2, !dbg !48
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local990, !dbg !48
  store half %1443, ptr addrspace(1) %address.local1015, align 2, !dbg !48
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local993, !dbg !48
  store half %1442, ptr addrspace(1) %address.local1017, align 2, !dbg !48
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local996, !dbg !48
  store half %1428, ptr addrspace(1) %address.local1019, align 2, !dbg !48
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local999, !dbg !48
  store half %1427, ptr addrspace(1) %address.local1021, align 2, !dbg !48
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1002, !dbg !48
  store half %1426, ptr addrspace(1) %address.local1023, align 2, !dbg !48
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1005, !dbg !48
  store half %1425, ptr addrspace(1) %address.local1025, align 2, !dbg !48
  %1482 = fmul <4 x float> %1411, %1319, !dbg !46
  %1483 = fptrunc <4 x float> %1482 to <4 x half>, !dbg !47
  %1484 = extractelement <4 x half> %1483, i64 3, !dbg !47
  %1485 = extractelement <4 x half> %1483, i64 2, !dbg !47
  %1486 = extractelement <4 x half> %1483, i64 1, !dbg !47
  %1487 = extractelement <4 x half> %1483, i64 0, !dbg !47
  %1488 = fmul <4 x float> %1411, %1286, !dbg !46
  %1489 = fptrunc <4 x float> %1488 to <4 x half>, !dbg !47
  %1490 = extractelement <4 x half> %1489, i64 3, !dbg !47
  %1491 = extractelement <4 x half> %1489, i64 2, !dbg !47
  %1492 = extractelement <4 x half> %1489, i64 1, !dbg !47
  %1493 = extractelement <4 x half> %1489, i64 0, !dbg !47
  %1494 = fmul <4 x float> %1409, %1253, !dbg !46
  %1495 = fptrunc <4 x float> %1494 to <4 x half>, !dbg !47
  %1496 = extractelement <4 x half> %1495, i64 3, !dbg !47
  %1497 = extractelement <4 x half> %1495, i64 2, !dbg !47
  %1498 = extractelement <4 x half> %1495, i64 1, !dbg !47
  %1499 = extractelement <4 x half> %1495, i64 0, !dbg !47
  %1500 = fmul <4 x float> %1409, %1220, !dbg !46
  %1501 = fptrunc <4 x float> %1500 to <4 x half>, !dbg !47
  %1502 = extractelement <4 x half> %1501, i64 3, !dbg !47
  %1503 = extractelement <4 x half> %1501, i64 2, !dbg !47
  %1504 = extractelement <4 x half> %1501, i64 1, !dbg !47
  %1505 = extractelement <4 x half> %1501, i64 0, !dbg !47
  %address.local1040 = or disjoint i64 %address.local984, 32, !dbg !48
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1040, !dbg !48
  store half %1505, ptr addrspace(1) %address.local1041, align 2, !dbg !48
  %address.local1043 = or disjoint i64 %address.local984, 33, !dbg !48
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1043, !dbg !48
  store half %1504, ptr addrspace(1) %address.local1044, align 2, !dbg !48
  %address.local1046 = or disjoint i64 %address.local984, 34, !dbg !48
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1046, !dbg !48
  store half %1503, ptr addrspace(1) %address.local1047, align 2, !dbg !48
  %address.local1049 = or disjoint i64 %address.local984, 35, !dbg !48
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1049, !dbg !48
  store half %1502, ptr addrspace(1) %address.local1050, align 2, !dbg !48
  %address.local1052 = or disjoint i64 %address.local984, 48, !dbg !48
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1052, !dbg !48
  store half %1499, ptr addrspace(1) %address.local1053, align 2, !dbg !48
  %address.local1055 = or disjoint i64 %address.local984, 49, !dbg !48
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1055, !dbg !48
  store half %1498, ptr addrspace(1) %address.local1056, align 2, !dbg !48
  %address.local1058 = or disjoint i64 %address.local984, 50, !dbg !48
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1058, !dbg !48
  store half %1497, ptr addrspace(1) %address.local1059, align 2, !dbg !48
  %address.local1061 = or i64 %1480, 51, !dbg !48
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1061, !dbg !48
  store half %1496, ptr addrspace(1) %address.local1062, align 2, !dbg !48
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1040, !dbg !48
  store half %1493, ptr addrspace(1) %address.local1067, align 2, !dbg !48
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1043, !dbg !48
  store half %1492, ptr addrspace(1) %address.local1069, align 2, !dbg !48
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1046, !dbg !48
  store half %1491, ptr addrspace(1) %address.local1071, align 2, !dbg !48
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1049, !dbg !48
  store half %1490, ptr addrspace(1) %address.local1073, align 2, !dbg !48
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1052, !dbg !48
  store half %1487, ptr addrspace(1) %address.local1075, align 2, !dbg !48
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1055, !dbg !48
  store half %1486, ptr addrspace(1) %address.local1077, align 2, !dbg !48
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1058, !dbg !48
  store half %1485, ptr addrspace(1) %address.local1079, align 2, !dbg !48
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1061, !dbg !48
  store half %1484, ptr addrspace(1) %address.local1081, align 2, !dbg !48
  %1506 = fmul <4 x float> %1411, %1187, !dbg !46
  %1507 = fptrunc <4 x float> %1506 to <4 x half>, !dbg !47
  %1508 = extractelement <4 x half> %1507, i64 3, !dbg !47
  %1509 = extractelement <4 x half> %1507, i64 2, !dbg !47
  %1510 = extractelement <4 x half> %1507, i64 1, !dbg !47
  %1511 = extractelement <4 x half> %1507, i64 0, !dbg !47
  %1512 = fmul <4 x float> %1411, %1154, !dbg !46
  %1513 = fptrunc <4 x float> %1512 to <4 x half>, !dbg !47
  %1514 = extractelement <4 x half> %1513, i64 3, !dbg !47
  %1515 = extractelement <4 x half> %1513, i64 2, !dbg !47
  %1516 = extractelement <4 x half> %1513, i64 1, !dbg !47
  %1517 = extractelement <4 x half> %1513, i64 0, !dbg !47
  %1518 = fmul <4 x float> %1409, %1121, !dbg !46
  %1519 = fptrunc <4 x float> %1518 to <4 x half>, !dbg !47
  %1520 = extractelement <4 x half> %1519, i64 3, !dbg !47
  %1521 = extractelement <4 x half> %1519, i64 2, !dbg !47
  %1522 = extractelement <4 x half> %1519, i64 1, !dbg !47
  %1523 = extractelement <4 x half> %1519, i64 0, !dbg !47
  %1524 = fmul <4 x float> %1409, %1088, !dbg !46
  %1525 = fptrunc <4 x float> %1524 to <4 x half>, !dbg !47
  %1526 = extractelement <4 x half> %1525, i64 3, !dbg !47
  %1527 = extractelement <4 x half> %1525, i64 2, !dbg !47
  %1528 = extractelement <4 x half> %1525, i64 1, !dbg !47
  %1529 = extractelement <4 x half> %1525, i64 0, !dbg !47
  %address.local1096 = or disjoint i64 %address.local984, 64, !dbg !48
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1096, !dbg !48
  store half %1529, ptr addrspace(1) %address.local1097, align 2, !dbg !48
  %address.local1099 = or disjoint i64 %address.local984, 65, !dbg !48
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1099, !dbg !48
  store half %1528, ptr addrspace(1) %address.local1100, align 2, !dbg !48
  %address.local1102 = or disjoint i64 %address.local984, 66, !dbg !48
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1102, !dbg !48
  store half %1527, ptr addrspace(1) %address.local1103, align 2, !dbg !48
  %address.local1105 = or disjoint i64 %address.local984, 67, !dbg !48
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1105, !dbg !48
  store half %1526, ptr addrspace(1) %address.local1106, align 2, !dbg !48
  %address.local1108 = or disjoint i64 %address.local984, 80, !dbg !48
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1108, !dbg !48
  store half %1523, ptr addrspace(1) %address.local1109, align 2, !dbg !48
  %address.local1111 = or disjoint i64 %address.local984, 81, !dbg !48
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1111, !dbg !48
  store half %1522, ptr addrspace(1) %address.local1112, align 2, !dbg !48
  %address.local1114 = or disjoint i64 %address.local984, 82, !dbg !48
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1114, !dbg !48
  store half %1521, ptr addrspace(1) %address.local1115, align 2, !dbg !48
  %address.local1117 = or disjoint i64 %address.local984, 83, !dbg !48
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1117, !dbg !48
  store half %1520, ptr addrspace(1) %address.local1118, align 2, !dbg !48
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1096, !dbg !48
  store half %1517, ptr addrspace(1) %address.local1123, align 2, !dbg !48
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1099, !dbg !48
  store half %1516, ptr addrspace(1) %address.local1125, align 2, !dbg !48
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1102, !dbg !48
  store half %1515, ptr addrspace(1) %address.local1127, align 2, !dbg !48
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1105, !dbg !48
  store half %1514, ptr addrspace(1) %address.local1129, align 2, !dbg !48
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1108, !dbg !48
  store half %1511, ptr addrspace(1) %address.local1131, align 2, !dbg !48
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1111, !dbg !48
  store half %1510, ptr addrspace(1) %address.local1133, align 2, !dbg !48
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1114, !dbg !48
  store half %1509, ptr addrspace(1) %address.local1135, align 2, !dbg !48
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1117, !dbg !48
  store half %1508, ptr addrspace(1) %address.local1137, align 2, !dbg !48
  %1530 = fmul <4 x float> %1411, %1055, !dbg !46
  %1531 = fptrunc <4 x float> %1530 to <4 x half>, !dbg !47
  %1532 = extractelement <4 x half> %1531, i64 3, !dbg !47
  %1533 = extractelement <4 x half> %1531, i64 2, !dbg !47
  %1534 = extractelement <4 x half> %1531, i64 1, !dbg !47
  %1535 = extractelement <4 x half> %1531, i64 0, !dbg !47
  %1536 = fmul <4 x float> %1411, %1022, !dbg !46
  %1537 = fptrunc <4 x float> %1536 to <4 x half>, !dbg !47
  %1538 = extractelement <4 x half> %1537, i64 3, !dbg !47
  %1539 = extractelement <4 x half> %1537, i64 2, !dbg !47
  %1540 = extractelement <4 x half> %1537, i64 1, !dbg !47
  %1541 = extractelement <4 x half> %1537, i64 0, !dbg !47
  %1542 = fmul <4 x float> %1409, %989, !dbg !46
  %1543 = fptrunc <4 x float> %1542 to <4 x half>, !dbg !47
  %1544 = extractelement <4 x half> %1543, i64 3, !dbg !47
  %1545 = extractelement <4 x half> %1543, i64 2, !dbg !47
  %1546 = extractelement <4 x half> %1543, i64 1, !dbg !47
  %1547 = extractelement <4 x half> %1543, i64 0, !dbg !47
  %1548 = fmul <4 x float> %1409, %956, !dbg !46
  %1549 = fptrunc <4 x float> %1548 to <4 x half>, !dbg !47
  %1550 = extractelement <4 x half> %1549, i64 3, !dbg !47
  %1551 = extractelement <4 x half> %1549, i64 2, !dbg !47
  %1552 = extractelement <4 x half> %1549, i64 1, !dbg !47
  %1553 = extractelement <4 x half> %1549, i64 0, !dbg !47
  %address.local1152 = or disjoint i64 %address.local984, 96, !dbg !48
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1152, !dbg !48
  store half %1553, ptr addrspace(1) %address.local1153, align 2, !dbg !48
  %address.local1155 = or disjoint i64 %address.local984, 97, !dbg !48
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1155, !dbg !48
  store half %1552, ptr addrspace(1) %address.local1156, align 2, !dbg !48
  %address.local1158 = or disjoint i64 %address.local984, 98, !dbg !48
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1158, !dbg !48
  store half %1551, ptr addrspace(1) %address.local1159, align 2, !dbg !48
  %address.local1161 = or disjoint i64 %address.local984, 99, !dbg !48
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1161, !dbg !48
  store half %1550, ptr addrspace(1) %address.local1162, align 2, !dbg !48
  %address.local1164 = or disjoint i64 %address.local984, 112, !dbg !48
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1164, !dbg !48
  store half %1547, ptr addrspace(1) %address.local1165, align 2, !dbg !48
  %address.local1167 = or disjoint i64 %address.local984, 113, !dbg !48
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1167, !dbg !48
  store half %1546, ptr addrspace(1) %address.local1168, align 2, !dbg !48
  %address.local1170 = or disjoint i64 %address.local984, 114, !dbg !48
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1170, !dbg !48
  store half %1545, ptr addrspace(1) %address.local1171, align 2, !dbg !48
  %address.local1173 = or i64 %1480, 115, !dbg !48
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1173, !dbg !48
  store half %1544, ptr addrspace(1) %address.local1174, align 2, !dbg !48
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1152, !dbg !48
  store half %1541, ptr addrspace(1) %address.local1179, align 2, !dbg !48
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1155, !dbg !48
  store half %1540, ptr addrspace(1) %address.local1181, align 2, !dbg !48
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1158, !dbg !48
  store half %1539, ptr addrspace(1) %address.local1183, align 2, !dbg !48
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1161, !dbg !48
  store half %1538, ptr addrspace(1) %address.local1185, align 2, !dbg !48
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1164, !dbg !48
  store half %1535, ptr addrspace(1) %address.local1187, align 2, !dbg !48
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1167, !dbg !48
  store half %1534, ptr addrspace(1) %address.local1189, align 2, !dbg !48
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1170, !dbg !48
  store half %1533, ptr addrspace(1) %address.local1191, align 2, !dbg !48
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1173, !dbg !48
  store half %1532, ptr addrspace(1) %address.local1193, align 2, !dbg !48
  ret void, !dbg !8
}

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.lazy_softmax_base"="v1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }
attributes #12 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 74, column: 20, scope: !3)
!8 = !DILocation(line: 23, column: 3, scope: !3)
!9 = !DILocation(line: 67, column: 22, scope: !3)
!10 = !{i32 0, i32 256}
!11 = !DILocation(line: 69, column: 24, scope: !3)
!12 = !DILocation(line: 57, column: 24, scope: !3)
!13 = !DILocation(line: 71, column: 23, scope: !3)
!14 = !DILocation(line: 68, column: 22, scope: !3)
!15 = !{i32 0}
!16 = !{i32 1}
!17 = !{i32 2}
!18 = !{i32 3}
!19 = !{i32 4}
!20 = !{i32 5}
!21 = !{i32 6}
!22 = !{i32 7}
!23 = !{i32 8}
!24 = !{i32 9}
!25 = !{i32 10}
!26 = !{i32 11}
!27 = !{i32 12}
!28 = !{i32 13}
!29 = !{i32 14}
!30 = !{i32 15}
!31 = !{i32 16}
!32 = !{i32 17}
!33 = !{i32 18}
!34 = !{i32 19}
!35 = !{i32 20}
!36 = !{i32 21}
!37 = !{i32 22}
!38 = !DILocation(line: 86, column: 20, scope: !3)
!39 = !DILocation(line: 78, column: 22, scope: !3)
!40 = !DILocation(line: 79, column: 22, scope: !3)
!41 = !DILocation(line: 88, column: 22, scope: !3)
!42 = !DILocation(line: 89, column: 20, scope: !3)
!43 = !{i64 8192}
!44 = !DILocation(line: 81, column: 16, scope: !3)
!45 = !DILocation(line: 80, column: 20, scope: !3)
!46 = !DILocation(line: 92, column: 23, scope: !3)
!47 = !DILocation(line: 93, column: 19, scope: !3)
!48 = !DILocation(line: 94, column: 9, scope: !3)
