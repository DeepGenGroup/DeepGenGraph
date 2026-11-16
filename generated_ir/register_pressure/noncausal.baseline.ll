; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/fragment_level_optimized_noncausal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/fragment_level_optimized_noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !28 {
  %kdma.tid = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !8
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !8
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !8
  %5 = and i32 %kdma.lane, 31, !dbg !8
  %kdma.owner = or disjoint i32 %kdma.wave, %5, !dbg !8
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !8
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !8
  %kdma.intra = and i32 %6, 8, !dbg !8
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !8
  %7 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %8 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %7), !dbg !8
  %9 = add nuw nsw i32 %8, 64, !dbg !8
  %10 = xor i32 %8, 16, !dbg !8
  %11 = and i32 %9, 192, !dbg !8
  %12 = icmp samesign ult i32 %10, %11, !dbg !8
  %13 = select i1 %12, i32 %10, i32 %8, !dbg !8
  %14 = shl nuw nsw i32 %13, 2, !dbg !8
  %15 = xor i32 %8, 32, !dbg !8
  %16 = icmp samesign ult i32 %15, %11, !dbg !8
  %17 = select i1 %16, i32 %15, i32 %8, !dbg !8
  %18 = shl nuw nsw i32 %17, 2, !dbg !8
  %19 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %20 = sext i32 %19 to i64, !dbg !5
  %21 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %22 = sext i32 %21 to i64, !dbg !5
  %23 = shl nsw i64 %22, 5, !dbg !5
  %24 = add nsw i64 %23, %20, !dbg !5
  %25 = lshr i64 %24, 5, !dbg !5
  %26 = and i64 %25, 576460752303423456, !dbg !5
  %27 = and i64 %20, 31, !dbg !5
  %28 = or disjoint i64 %26, %27, !dbg !5
  %29 = lshr i64 %24, 10, !dbg !9
  %30 = add nuw nsw i64 %25, %29, !dbg !9
  %31 = and i64 %30, 31, !dbg !9
  %32 = shl nsw i64 %28, 7, !dbg !9
  %33 = mul nsw i64 %29, -4096, !dbg !9
  %34 = add i64 %32, %33, !dbg !9
  %35 = shl nsw i64 %34, 7, !dbg !9
  %.idx = shl nuw nsw i64 %31, 20, !dbg !9
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !9, !range !30, !frisk.register_lifetimes !1
  %36 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %37 = and i32 %36, 1536, !dbg !9
  %address.local399 = zext nneg i32 %37 to i64, !dbg !9
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local399, !dbg !9
  %address.local401 = ptrtoint ptr addrspace(3) %address.local400 to i64, !dbg !9
  %address.local402 = trunc i64 %address.local401 to i32, !dbg !9
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local402), !dbg !9
  %address.local403 = lshr i32 %address.thread, 4, !dbg !9
  %address.local404 = and i32 %address.local403, 8, !dbg !9
  %address.local405 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local406 = and i32 %address.local405, 16256, !dbg !9
  %address.local407 = or disjoint i32 %address.local404, %address.local406, !dbg !9
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1373 = shl i64 %34, 8, !dbg !9
  %40 = add i64 %.idx, %.idx1373, !dbg !9
  %41 = shl nuw nsw i32 %address.local407, 1, !dbg !9
  %42 = trunc i64 %40 to i32, !dbg !9
  %dma.buffer.offset1371 = or disjoint i32 %41, %42, !dbg !9
  %43 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !9
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1372 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1371, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %address.local412 = or disjoint i64 %address.local399, 2048, !dbg !9
  %address.local413 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local412, !dbg !9
  %address.local414 = ptrtoint ptr addrspace(3) %address.local413 to i64, !dbg !9
  %address.local415 = trunc i64 %address.local414 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local415), !dbg !9
  %address.local421 = or disjoint i32 %address.local407, 16, !dbg !9
  %47 = shl nuw nsw i32 %address.local421, 1, !dbg !9
  %dma.buffer.offset1369 = or disjoint i32 %47, %42, !dbg !9
  %48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1369, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %49 = shl nuw nsw i64 %31, 19, !dbg !9
  %50 = add i64 %49, %35, !dbg !9
  %address.local424 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local399, !dbg !9
  %address.local425 = ptrtoint ptr addrspace(3) %address.local424 to i64, !dbg !9
  %address.local426 = trunc i64 %address.local425 to i32, !dbg !9
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local426), !dbg !9
  %52 = trunc i64 %50 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local407, %52, !dbg !9
  %53 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1367 = or disjoint i32 %53, 64, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1367, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local412, !dbg !9
  %address.local429 = ptrtoint ptr addrspace(3) %address.local428 to i64, !dbg !9
  %address.local430 = trunc i64 %address.local429 to i32, !dbg !9
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local430), !dbg !9
  %.tr1377 = or disjoint i32 %address.local421, %52, !dbg !9
  %56 = shl i32 %.tr1377, 1, !dbg !9
  %dma.buffer.offset1365 = or disjoint i32 %56, 64, !dbg !9
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1365, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %address.local432 = lshr i32 %address.thread, 2, !dbg !12
  %address.local433 = and i32 %address.local432, 4, !dbg !12
  %address.local434 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local435 = and i32 %address.local434, 384, !dbg !12
  %address.local436 = or disjoint i32 %address.local433, %address.local435, !dbg !12
  %address.local438 = and i32 %36, 120, !dbg !12
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !12
  %address.local440 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local441 = and i32 %address.local440, 1024, !dbg !12
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !12
  %address.local443 = zext nneg i32 %address.local442 to i64, !dbg !12
  %address.local444 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local443, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local444, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local444, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep1194 = getelementptr i8, ptr addrspace(3) %address.local444, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep1195 = getelementptr i8, ptr addrspace(3) %address.local444, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1363 = or disjoint i32 %53, 128, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1363, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %dma.buffer.offset1361 = or disjoint i32 %56, 128, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1361, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %lds.const.gep1196 = getelementptr i8, ptr addrspace(3) %address.local444, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !12
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local444, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !12
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local444, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local444, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1359 = or disjoint i32 %53, 192, !dbg !9
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1359, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %dma.buffer.offset1357 = or disjoint i32 %56, 192, !dbg !9
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1357, <4 x i32> %dma.buffer.rsrc1372) #10, !dbg !9
  %78 = load <4 x half>, ptr addrspace(3) %address.local444, align 8, !dbg !12
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1194, align 8, !dbg !12
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1195, align 8, !dbg !12
  %83 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !12
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !12
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1196, align 8, !dbg !12
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !12
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !12
  %91 = fmul <4 x half> %87, splat (half 0xH3015), !dbg !12
  %92 = fmul <4 x half> %88, splat (half 0xH3015), !dbg !12
  %93 = fmul <4 x half> %90, splat (half 0xH3015), !dbg !12
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %94 = shl nuw nsw i32 %kdma.tid, 2, !dbg !15
  %95 = and i32 %94, 3084, !dbg !15
  %96 = and i32 %kdma.tid, 1, !dbg !15
  %97 = or disjoint i32 %95, %96, !dbg !15
  %narrow = add nuw nsw i32 %97, 1, !dbg !15
  %98 = lshr i32 %narrow, 1, !dbg !15
  %99 = zext nneg i32 %98 to i64, !dbg !15
  %k0.prime. = shl i64 %25, 19, !dbg !13
  %k0.prime.2040 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.2041 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.2042 = or disjoint i64 %k0.prime.2040, %k0.prime.2041, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1479 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1480 = shl nuw nsw i64 %k0.prime.kdma.source1479, 9, !dbg !13
  %k0.prime.kdma.source1481 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !13
  %k0.prime.kdma.source1482 = add nuw nsw i64 %k0.prime.kdma.source1480, %k0.prime.kdma.source1481, !dbg !13
  %k0.prime.kdma.source1483 = lshr i64 %k0.prime.kdma.source, 5, !dbg !13
  %k0.prime.kdma.source1484 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1485 = and i64 %k0.prime.kdma.source1484, 1, !dbg !13
  %k0.prime.kdma.source1486 = and i64 %k0.prime.kdma.source1479, %k0.prime.kdma.source1485, !dbg !13
  %k0.prime.kdma.source1487 = or disjoint i64 %k0.prime.kdma.source1483, %k0.prime.kdma.source1486, !dbg !13
  %k0.prime.kdma.source1488 = shl nuw nsw i64 %k0.prime.kdma.source1487, 10, !dbg !13
  %k0.prime.kdma.source1489 = sub nsw i64 %k0.prime.kdma.source1482, %k0.prime.kdma.source1488, !dbg !13
  %k0.prime.kdma.source1490 = and i64 %k0.prime.kdma.source1489, 16256, !dbg !13
  %k0.prime.kdma.source1491 = lshr i32 %kdma.tid, 4, !dbg !13
  %k0.prime.kdma.source1492 = and i32 %k0.prime.kdma.source1491, 8, !dbg !13
  %k0.prime.kdma.source1493 = zext nneg i32 %k0.prime.kdma.source1492 to i64, !dbg !13
  %k0.prime.kdma.source1494 = or disjoint i64 %k0.prime.kdma.source1490, %k0.prime.kdma.source1493, !dbg !13
  %k0.prime.kdma.source1495 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1494, !dbg !13
  %k0.prime.kdma.source1496 = trunc nuw i64 %k0.prime.kdma.source1495 to i32, !dbg !13
  %k0.prime.kdma.source1497 = shl nuw nsw i32 %k0.prime.kdma.source1496, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1497, %kdma.intra, !dbg !13
  %k0.prime.2043 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.2044 = insertelement <2 x i64> poison, i64 %k0.prime.2043, i64 0, !dbg !13
  %k0.prime.2045 = bitcast <2 x i64> %k0.prime.2044 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1356 = shufflevector <4 x i32> %k0.prime.2045, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.2046 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1519 = add i32 %38, 512, !dbg !13
  %k0.prime.kdma.source1498 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1499 = lshr i64 %k0.prime.kdma.source1498, 4, !dbg !13
  %k0.prime.kdma.source1500 = shl nuw nsw i64 %k0.prime.kdma.source1499, 9, !dbg !13
  %k0.prime.kdma.source1501 = shl nuw nsw i64 %k0.prime.kdma.source1498, 7, !dbg !13
  %k0.prime.kdma.source1502 = add nuw nsw i64 %k0.prime.kdma.source1500, %k0.prime.kdma.source1501, !dbg !13
  %k0.prime.kdma.source1503 = lshr i64 %k0.prime.kdma.source1498, 5, !dbg !13
  %k0.prime.kdma.source1504 = lshr i64 %k0.prime.kdma.source1498, 2, !dbg !13
  %k0.prime.kdma.source1505 = and i64 %k0.prime.kdma.source1504, 1, !dbg !13
  %k0.prime.kdma.source1506 = and i64 %k0.prime.kdma.source1499, %k0.prime.kdma.source1505, !dbg !13
  %k0.prime.kdma.source1507 = add nuw nsw i64 %k0.prime.kdma.source1503, %k0.prime.kdma.source1506, !dbg !13
  %k0.prime.kdma.source1508 = shl nuw nsw i64 %k0.prime.kdma.source1507, 10, !dbg !13
  %k0.prime.kdma.source1509 = sub nsw i64 %k0.prime.kdma.source1502, %k0.prime.kdma.source1508, !dbg !13
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 16256, !dbg !13
  %k0.prime.kdma.source1514 = or disjoint i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source1493, !dbg !13
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1514, !dbg !13
  %k0.prime.kdma.source1516 = trunc nuw i64 %k0.prime.kdma.source1515 to i32, !dbg !13
  %k0.prime.kdma.source1517 = shl nuw nsw i32 %k0.prime.kdma.source1516, 1, !dbg !13
  %k0.prime.kdma.offset1518 = or disjoint i32 %k0.prime.kdma.source1517, %kdma.intra.hi, !dbg !13
  %k0.prime.2047 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1519, i32 %k0.prime.kdma.offset1518, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1521 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !13
  %k0.prime.kdma.source1522 = lshr i64 %k0.prime.kdma.source1521, 4, !dbg !13
  %k0.prime.kdma.source1523 = shl nuw nsw i64 %k0.prime.kdma.source1522, 9, !dbg !13
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1523, %k0.prime.kdma.source1481, !dbg !13
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source1521, 5, !dbg !13
  %k0.prime.kdma.source1529 = and i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1485, !dbg !13
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.kdma.source1526, %k0.prime.kdma.source1529, !dbg !13
  %k0.prime.kdma.source1531 = shl nuw nsw i64 %k0.prime.kdma.source1530, 10, !dbg !13
  %k0.prime.kdma.source1532 = sub nsw i64 %k0.prime.kdma.source1525, %k0.prime.kdma.source1531, !dbg !13
  %k0.prime.kdma.source1533 = and i64 %k0.prime.kdma.source1532, 16256, !dbg !13
  %k0.prime.kdma.source1536 = or disjoint i32 %k0.prime.kdma.source1492, 16, !dbg !13
  %k0.prime.kdma.source1537 = zext nneg i32 %k0.prime.kdma.source1536 to i64, !dbg !13
  %k0.prime.kdma.source1538 = or disjoint i64 %k0.prime.kdma.source1533, %k0.prime.kdma.source1537, !dbg !13
  %k0.prime.kdma.source1539 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1538, !dbg !13
  %k0.prime.kdma.source1540 = trunc nuw i64 %k0.prime.kdma.source1539 to i32, !dbg !13
  %k0.prime.kdma.source1541 = shl nuw nsw i32 %k0.prime.kdma.source1540, 1, !dbg !13
  %k0.prime.kdma.offset1542 = add nuw i32 %k0.prime.kdma.source1541, %kdma.intra, !dbg !13
  %k0.prime.2048 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.prime.kdma.offset1542, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1567 = add i32 %46, 512, !dbg !13
  %k0.prime.kdma.source1545 = or disjoint i64 %k0.prime.kdma.source1498, 256, !dbg !13
  %k0.prime.kdma.source1546 = lshr i64 %k0.prime.kdma.source1545, 4, !dbg !13
  %k0.prime.kdma.source1547 = shl nuw nsw i64 %k0.prime.kdma.source1546, 9, !dbg !13
  %k0.prime.kdma.source1549 = add nuw nsw i64 %k0.prime.kdma.source1547, %k0.prime.kdma.source1501, !dbg !13
  %k0.prime.kdma.source1550 = lshr i64 %k0.prime.kdma.source1545, 5, !dbg !13
  %k0.prime.kdma.source1553 = and i64 %k0.prime.kdma.source1546, %k0.prime.kdma.source1505, !dbg !13
  %k0.prime.kdma.source1554 = add nuw nsw i64 %k0.prime.kdma.source1550, %k0.prime.kdma.source1553, !dbg !13
  %k0.prime.kdma.source1555 = shl nuw nsw i64 %k0.prime.kdma.source1554, 10, !dbg !13
  %k0.prime.kdma.source1556 = sub nsw i64 %k0.prime.kdma.source1549, %k0.prime.kdma.source1555, !dbg !13
  %k0.prime.kdma.source1557 = and i64 %k0.prime.kdma.source1556, 16256, !dbg !13
  %k0.prime.kdma.source1562 = or disjoint i64 %k0.prime.kdma.source1557, %k0.prime.kdma.source1537, !dbg !13
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.2042, %k0.prime.kdma.source1562, !dbg !13
  %k0.prime.kdma.source1564 = trunc nuw i64 %k0.prime.kdma.source1563 to i32, !dbg !13
  %k0.prime.kdma.source1565 = shl nuw nsw i32 %k0.prime.kdma.source1564, 1, !dbg !13
  %k0.prime.kdma.offset1566 = add nuw i32 %k0.prime.kdma.source1565, %kdma.intra.hi, !dbg !13
  %k0.prime.2049 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1567, i32 %k0.prime.kdma.offset1566, <4 x i32> %k0.prime.dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %100, !dbg !6

100:                                              ; preds = %k0.join, %4
  %101 = phi i64 [ %933, %k0.join ], [ 0, %4 ], !dbg !6
  %102 = phi [2 x <8 x float>] [ %1001, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %103 = phi [2 x <1 x float>] [ %766, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %104 = phi [2 x <1 x float>] [ %1003, %k0.join ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %105 = phi [2 x <8 x float>] [ %1007, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %106 = phi [2 x <8 x float>] [ %1011, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %107 = phi [2 x <8 x float>] [ %1015, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %108 = icmp samesign ult i64 %101, 32, !dbg !6
  br i1 %108, label %109, label %1016, !dbg !6

109:                                              ; preds = %100
  %110 = shl i64 %25, 19, !dbg !13
  %111 = and i64 %110, 16252928, !dbg !13
  %112 = shl nuw nsw i64 %101, 14, !dbg !13
  %113 = or disjoint i64 %111, %112, !dbg !13
  %address.thread495 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %address.local496 = zext nneg i32 %address.thread495 to i64, !dbg !9
  %address.local502 = lshr i64 %address.local496, 2, !dbg !13
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0, !dbg !13
  %116 = bitcast <2 x i64> %115 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1356 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1479 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1480 = shl nuw nsw i64 %kdma.source1479, 9, !dbg !13
  %kdma.source1481 = shl nuw nsw i64 %kdma.source, 7, !dbg !13
  %kdma.source1482 = add nuw nsw i64 %kdma.source1480, %kdma.source1481, !dbg !13
  %kdma.source1483 = lshr i64 %kdma.source, 5, !dbg !13
  %kdma.source1484 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1485 = and i64 %kdma.source1484, 1, !dbg !13
  %kdma.source1486 = and i64 %kdma.source1479, %kdma.source1485, !dbg !13
  %kdma.source1487 = or disjoint i64 %kdma.source1483, %kdma.source1486, !dbg !13
  %kdma.source1488 = shl nuw nsw i64 %kdma.source1487, 10, !dbg !13
  %kdma.source1489 = sub nsw i64 %kdma.source1482, %kdma.source1488, !dbg !13
  %kdma.source1490 = and i64 %kdma.source1489, 16256, !dbg !13
  %kdma.source1491 = lshr i32 %kdma.tid, 4, !dbg !13
  %kdma.source1492 = and i32 %kdma.source1491, 8, !dbg !13
  %kdma.source1493 = zext nneg i32 %kdma.source1492 to i64, !dbg !13
  %kdma.source1494 = or disjoint i64 %kdma.source1490, %kdma.source1493, !dbg !13
  %kdma.source1495 = or disjoint i64 %113, %kdma.source1494, !dbg !13
  %kdma.source1496 = trunc nuw i64 %kdma.source1495 to i32, !dbg !13
  %kdma.source1497 = shl nuw nsw i32 %kdma.source1496, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1497, %kdma.intra, !dbg !13
  %kdma.source1498 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1499 = lshr i64 %kdma.source1498, 4, !dbg !13
  %kdma.source1500 = shl nuw nsw i64 %kdma.source1499, 9, !dbg !13
  %kdma.source1501 = shl nuw nsw i64 %kdma.source1498, 7, !dbg !13
  %kdma.source1502 = add nuw nsw i64 %kdma.source1500, %kdma.source1501, !dbg !13
  %kdma.source1503 = lshr i64 %kdma.source1498, 5, !dbg !13
  %kdma.source1504 = lshr i64 %kdma.source1498, 2, !dbg !13
  %kdma.source1505 = and i64 %kdma.source1504, 1, !dbg !13
  %kdma.source1506 = and i64 %kdma.source1499, %kdma.source1505, !dbg !13
  %kdma.source1507 = add nuw nsw i64 %kdma.source1503, %kdma.source1506, !dbg !13
  %kdma.source1508 = shl nuw nsw i64 %kdma.source1507, 10, !dbg !13
  %kdma.source1509 = sub nsw i64 %kdma.source1502, %kdma.source1508, !dbg !13
  %kdma.source1510 = and i64 %kdma.source1509, 16256, !dbg !13
  %kdma.source1514 = or disjoint i64 %kdma.source1510, %kdma.source1493, !dbg !13
  %kdma.source1515 = or disjoint i64 %113, %kdma.source1514, !dbg !13
  %kdma.source1516 = trunc nuw i64 %kdma.source1515 to i32, !dbg !13
  %kdma.source1517 = shl nuw nsw i32 %kdma.source1516, 1, !dbg !13
  %kdma.offset1518 = or disjoint i32 %kdma.source1517, %kdma.intra.hi, !dbg !13
  %kdma.lds1519 = add i32 %38, 512, !dbg !13
  %kdma.source1521 = or disjoint i64 %kdma.source, 256, !dbg !13
  %kdma.source1522 = lshr i64 %kdma.source1521, 4, !dbg !13
  %kdma.source1523 = shl nuw nsw i64 %kdma.source1522, 9, !dbg !13
  %kdma.source1525 = add nuw nsw i64 %kdma.source1523, %kdma.source1481, !dbg !13
  %kdma.source1526 = lshr i64 %kdma.source1521, 5, !dbg !13
  %kdma.source1529 = and i64 %kdma.source1522, %kdma.source1485, !dbg !13
  %kdma.source1530 = or disjoint i64 %kdma.source1526, %kdma.source1529, !dbg !13
  %kdma.source1531 = shl nuw nsw i64 %kdma.source1530, 10, !dbg !13
  %kdma.source1532 = sub nsw i64 %kdma.source1525, %kdma.source1531, !dbg !13
  %kdma.source1533 = and i64 %kdma.source1532, 16256, !dbg !13
  %kdma.source1536 = or disjoint i32 %kdma.source1492, 16, !dbg !13
  %kdma.source1537 = zext nneg i32 %kdma.source1536 to i64, !dbg !13
  %kdma.source1538 = or disjoint i64 %kdma.source1533, %kdma.source1537, !dbg !13
  %kdma.source1539 = or disjoint i64 %113, %kdma.source1538, !dbg !13
  %kdma.source1540 = trunc nuw i64 %kdma.source1539 to i32, !dbg !13
  %kdma.source1541 = shl nuw nsw i32 %kdma.source1540, 1, !dbg !13
  %kdma.offset1542 = add nuw i32 %kdma.source1541, %kdma.intra, !dbg !13
  %kdma.source1545 = or disjoint i64 %kdma.source1498, 256, !dbg !13
  %kdma.source1546 = lshr i64 %kdma.source1545, 4, !dbg !13
  %kdma.source1547 = shl nuw nsw i64 %kdma.source1546, 9, !dbg !13
  %kdma.source1549 = add nuw nsw i64 %kdma.source1547, %kdma.source1501, !dbg !13
  %kdma.source1550 = lshr i64 %kdma.source1545, 5, !dbg !13
  %kdma.source1553 = and i64 %kdma.source1546, %kdma.source1505, !dbg !13
  %kdma.source1554 = add nuw nsw i64 %kdma.source1550, %kdma.source1553, !dbg !13
  %kdma.source1555 = shl nuw nsw i64 %kdma.source1554, 10, !dbg !13
  %kdma.source1556 = sub nsw i64 %kdma.source1549, %kdma.source1555, !dbg !13
  %kdma.source1557 = and i64 %kdma.source1556, 16256, !dbg !13
  %kdma.source1562 = or disjoint i64 %kdma.source1557, %kdma.source1537, !dbg !13
  %kdma.source1563 = or disjoint i64 %113, %kdma.source1562, !dbg !13
  %kdma.source1564 = trunc nuw i64 %kdma.source1563 to i32, !dbg !13
  %kdma.source1565 = shl nuw nsw i32 %kdma.source1564, 1, !dbg !13
  %kdma.offset1566 = add nuw i32 %kdma.source1565, %kdma.intra.hi, !dbg !13
  %kdma.lds1567 = add i32 %46, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %117 = or disjoint i64 %113, 32, !dbg !13
  %kdma.source1585 = or disjoint i64 %117, %kdma.source1494, !dbg !13
  %kdma.source1586 = trunc nuw i64 %kdma.source1585 to i32, !dbg !13
  %kdma.source1587 = shl nuw nsw i32 %kdma.source1586, 1, !dbg !13
  %kdma.offset1588 = or disjoint i32 %kdma.source1587, %kdma.intra, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1588, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1607 = or disjoint i64 %117, %kdma.source1514, !dbg !13
  %kdma.source1608 = trunc nuw i64 %kdma.source1607 to i32, !dbg !13
  %kdma.source1609 = shl nuw nsw i32 %kdma.source1608, 1, !dbg !13
  %kdma.offset1610 = or disjoint i32 %kdma.source1609, %kdma.intra.hi, !dbg !13
  %kdma.lds1611 = add i32 %51, 512, !dbg !13
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1611, i32 %kdma.offset1610, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1631 = or disjoint i64 %117, %kdma.source1538, !dbg !13
  %kdma.source1632 = trunc nuw i64 %kdma.source1631 to i32, !dbg !13
  %kdma.source1633 = shl nuw nsw i32 %kdma.source1632, 1, !dbg !13
  %kdma.offset1634 = add nuw i32 %kdma.source1633, %kdma.intra, !dbg !13
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1634, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1655 = or disjoint i64 %117, %kdma.source1562, !dbg !13
  %kdma.source1656 = trunc nuw i64 %kdma.source1655 to i32, !dbg !13
  %kdma.source1657 = shl nuw nsw i32 %kdma.source1656, 1, !dbg !13
  %kdma.offset1658 = add nuw i32 %kdma.source1657, %kdma.intra.hi, !dbg !13
  %kdma.lds1659 = add i32 %55, 512, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1659, i32 %kdma.offset1658, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %address.local539 = shl nuw nsw i64 %address.local496, 1, !dbg !15
  %address.local540 = and i64 %address.local539, 24, !dbg !15
  %address.local542 = shl i64 %address.local496, 62, !dbg !15
  %address.local543 = ashr exact i64 %address.local542, 57, !dbg !15
  %address.local544 = and i64 %address.local543, -64, !dbg !15
  %address.local547 = and i64 %address.local502, 12, !dbg !12
  %address.local549 = and i64 %address.local496, 63, !dbg !12
  %address.local550 = lshr i64 %address.local549, 5, !dbg !12
  %address.local551 = mul nuw nsw i64 %address.local550, 1016, !dbg !12
  %.idx1394 = mul nuw nsw i64 %address.local496, 320, !dbg !15
  %122 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1394, !dbg !15
  %.idx1395 = mul nsw i64 %address.local502, -1280, !dbg !15
  %123 = getelementptr i8, ptr addrspace(3) %122, i64 %.idx1395, !dbg !15
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local540, !dbg !15
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local544, !dbg !15
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local547, !dbg !15
  %address.local553 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local551, !dbg !15
  %127 = add nsw i64 %.idx1394, %.idx1395, !dbg !15
  %.idx1970 = shl nuw nsw i64 %address.local540, 1, !dbg !15
  %128 = or disjoint i64 %127, %.idx1970, !dbg !15
  %129 = or disjoint i64 %address.local544, %address.local547, !dbg !15
  %130 = shl nsw i64 %129, 1, !dbg !15
  %131 = add nsw i64 %130, %128, !dbg !15
  %address.local553.idx = mul nuw nsw i64 %address.local550, 2032, !dbg !15
  %132 = add nsw i64 %131, %address.local553.idx, !dbg !15
  %133 = lshr i64 %132, 6, !dbg !15
  %134 = and i64 %133, 8, !dbg !15
  %135 = xor i64 %132, %134, !dbg !15
  %136 = sub i64 %135, %132, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local553, i64 %136, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local564 = shl nuw nsw i32 %address.thread495, 1, !dbg !15
  %address.local565 = and i32 %address.local564, 24, !dbg !15
  %address.local566 = shl nuw nsw i32 %address.thread495, 8, !dbg !15
  %address.local567 = and i32 %address.local566, 768, !dbg !15
  %address.local568 = or disjoint i32 %address.local565, %address.local567, !dbg !15
  %address.local569 = shl nuw nsw i32 %address.thread495, 5, !dbg !15
  %address.local570 = and i32 %address.local569, 96, !dbg !15
  %address.local571 = or disjoint i32 %address.local568, %address.local570, !dbg !15
  %address.local572 = zext nneg i32 %address.local571 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %99) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local573 = lshr i32 %address.thread495, 1, !dbg !15
  %address.local574 = and i32 %address.local573, 1, !dbg !15
  %address.local575 = zext nneg i32 %address.local574 to i64, !dbg !15
  %address.local576 = add nsw i64 %address.base, %address.local575, !dbg !15
  %address.local577 = shl i64 %address.local576, 6, !dbg !15
  %address.local578 = sub i64 %address.local572, %address.local577, !dbg !15
  %140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local578, !dbg !15
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local547, !dbg !15
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local551, !dbg !15
  %address.local582 = getelementptr i8, ptr addrspace(3) %142, i64 64, !dbg !15
  %.idx19731976 = add i64 %address.local578, %address.local547, !dbg !15
  %143 = shl i64 %.idx19731976, 1, !dbg !15
  %144 = add i64 %143, %address.local553.idx, !dbg !15
  %145 = add i64 %144, 64, !dbg !15
  %146 = lshr i64 %145, 6, !dbg !15
  %147 = and i64 %146, 8, !dbg !15
  %148 = xor i64 %145, %147, !dbg !15
  %149 = sub i64 %148, %145, !dbg !15
  %kdma.swizzled1844 = getelementptr i8, ptr addrspace(3) %address.local582, i64 %149, !dbg !15
  %150 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1844, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 128, !dbg !15
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 4096, !dbg !15
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 4224, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !15, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %139, <4 x float> %138) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 1024, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 1152, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !15, !frisk.k_lds_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 5120, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1857 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 5248, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !15, !frisk.k_lds_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !15, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %156, <4 x float> %167) #9, !dbg !15
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !15
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %153, <4 x float> %169) #9, !dbg !15
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %157, <4 x float> %171) #9, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !15
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %162, <4 x float> %173) #9, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %166, <4 x float> %175) #9, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !15
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %163, <4 x float> %177) #9, !dbg !15
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> zeroinitializer) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> zeroinitializer) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> zeroinitializer) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> zeroinitializer) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> zeroinitializer) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> zeroinitializer) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> zeroinitializer) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %163, <4 x float> %193) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %195 = or disjoint i64 %113, 64, !dbg !13
  %kdma.source1677 = or disjoint i64 %195, %kdma.source1494, !dbg !13
  %kdma.source1678 = trunc nuw i64 %kdma.source1677 to i32, !dbg !13
  %kdma.source1679 = shl nuw nsw i32 %kdma.source1678, 1, !dbg !13
  %kdma.offset1680 = or disjoint i32 %kdma.source1679, %kdma.intra, !dbg !13
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %kdma.offset1680, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1699 = or disjoint i64 %195, %kdma.source1514, !dbg !13
  %kdma.source1700 = trunc nuw i64 %kdma.source1699 to i32, !dbg !13
  %kdma.source1701 = shl nuw nsw i32 %kdma.source1700, 1, !dbg !13
  %kdma.offset1702 = or disjoint i32 %kdma.source1701, %kdma.intra.hi, !dbg !13
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1519, i32 %kdma.offset1702, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1723 = or disjoint i64 %195, %kdma.source1538, !dbg !13
  %kdma.source1724 = trunc nuw i64 %kdma.source1723 to i32, !dbg !13
  %kdma.source1725 = shl nuw nsw i32 %kdma.source1724, 1, !dbg !13
  %kdma.offset1726 = add nuw i32 %kdma.source1725, %kdma.intra, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %kdma.offset1726, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1747 = or disjoint i64 %195, %kdma.source1562, !dbg !13
  %kdma.source1748 = trunc nuw i64 %kdma.source1747 to i32, !dbg !13
  %kdma.source1749 = shl nuw nsw i32 %kdma.source1748, 1, !dbg !13
  %kdma.offset1750 = add nuw i32 %kdma.source1749, %kdma.intra.hi, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1567, i32 %kdma.offset1750, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 8192, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1880 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 8320, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 12288, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1881 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 12416, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1882 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1886 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !15, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %200, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1883 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1887 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %210) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1884 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 9216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1888 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 9344, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1885 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 13312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1889 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1844, i64 13440, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %206, <4 x float> %216) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %220 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %201, <4 x float> %168) #9, !dbg !15
  %221 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %203, <4 x float> %220) #9, !dbg !15
  %222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %205, <4 x float> %170) #9, !dbg !15
  %223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %207, <4 x float> %222) #9, !dbg !15
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %208, <4 x float> %172) #9, !dbg !15
  %225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %211, <4 x float> %224) #9, !dbg !15
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %214, <4 x float> %174) #9, !dbg !15
  %227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %217, <4 x float> %226) #9, !dbg !15
  %228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %209, <4 x float> %176) #9, !dbg !15
  %229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %212, <4 x float> %228) #9, !dbg !15
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %215, <4 x float> %178) #9, !dbg !15
  %231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %218, <4 x float> %230) #9, !dbg !15
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %200, <4 x float> %180) #9, !dbg !15
  %233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %202, <4 x float> %232) #9, !dbg !15
  %234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %204, <4 x float> %182) #9, !dbg !15
  %235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %206, <4 x float> %234) #9, !dbg !15
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %201, <4 x float> %184) #9, !dbg !15
  %237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %203, <4 x float> %236) #9, !dbg !15
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %205, <4 x float> %186) #9, !dbg !15
  %239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %207, <4 x float> %238) #9, !dbg !15
  %240 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %208, <4 x float> %188) #9, !dbg !15
  %241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %211, <4 x float> %240) #9, !dbg !15
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %214, <4 x float> %190) #9, !dbg !15
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %217, <4 x float> %242) #9, !dbg !15
  %244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %209, <4 x float> %192) #9, !dbg !15
  %245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %212, <4 x float> %244) #9, !dbg !15
  %246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %215, <4 x float> %194) #9, !dbg !15
  %247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %218, <4 x float> %246) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %248 = or disjoint i64 %113, 96, !dbg !13
  %kdma.source1769 = or disjoint i64 %248, %kdma.source1494, !dbg !13
  %kdma.source1770 = trunc nuw i64 %kdma.source1769 to i32, !dbg !13
  %kdma.source1771 = shl nuw nsw i32 %kdma.source1770, 1, !dbg !13
  %kdma.offset1772 = or disjoint i32 %kdma.source1771, %kdma.intra, !dbg !13
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1772, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1791 = or disjoint i64 %248, %kdma.source1514, !dbg !13
  %kdma.source1792 = trunc nuw i64 %kdma.source1791 to i32, !dbg !13
  %kdma.source1793 = shl nuw nsw i32 %kdma.source1792, 1, !dbg !13
  %kdma.offset1794 = or disjoint i32 %kdma.source1793, %kdma.intra.hi, !dbg !13
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1611, i32 %kdma.offset1794, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1815 = or disjoint i64 %248, %kdma.source1538, !dbg !13
  %kdma.source1816 = trunc nuw i64 %kdma.source1815 to i32, !dbg !13
  %kdma.source1817 = shl nuw nsw i32 %kdma.source1816, 1, !dbg !13
  %kdma.offset1818 = add nuw i32 %kdma.source1817, %kdma.intra, !dbg !13
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1818, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %kdma.source1839 = or disjoint i64 %248, %kdma.source1562, !dbg !13
  %kdma.source1840 = trunc nuw i64 %kdma.source1839 to i32, !dbg !13
  %kdma.source1841 = shl nuw nsw i32 %kdma.source1840, 1, !dbg !13
  %kdma.offset1842 = add nuw i32 %kdma.source1841, %kdma.intra.hi, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1659, i32 %kdma.offset1842, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !47
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !15, !frisk.k_lds_pair !47
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !48
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !15, !frisk.k_lds_pair !48
  %257 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1844, align 8, !dbg !15, !frisk.k_lds_pair !49
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !15, !frisk.k_lds_pair !49
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !15, !frisk.k_lds_pair !50
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !15, !frisk.k_lds_pair !50
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !15, !frisk.k_lds_pair !51
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %253, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !15, !frisk.k_lds_pair !52
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %255, <4 x float> %263) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !15, !frisk.k_lds_pair !53
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %257, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !15, !frisk.k_lds_pair !54
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1857, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %259, <4 x float> %269) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %273 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %254, <4 x float> %221) #9, !dbg !15
  %274 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %256, <4 x float> %273) #9, !dbg !15
  %275 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %258, <4 x float> %223) #9, !dbg !15
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %260, <4 x float> %275) #9, !dbg !15
  %277 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %261, <4 x float> %225) #9, !dbg !15
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %264, <4 x float> %277) #9, !dbg !15
  %279 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %267, <4 x float> %227) #9, !dbg !15
  %280 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %270, <4 x float> %279) #9, !dbg !15
  %281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %262, <4 x float> %229) #9, !dbg !15
  %282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %265, <4 x float> %281) #9, !dbg !15
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %268, <4 x float> %231) #9, !dbg !15
  %284 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %271, <4 x float> %283) #9, !dbg !15
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %253, <4 x float> %233) #9, !dbg !15
  %286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %255, <4 x float> %285) #9, !dbg !15
  %287 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %257, <4 x float> %235) #9, !dbg !15
  %288 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %259, <4 x float> %287) #9, !dbg !15
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %254, <4 x float> %237) #9, !dbg !15
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %256, <4 x float> %289) #9, !dbg !15
  %291 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %258, <4 x float> %239) #9, !dbg !15
  %292 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %260, <4 x float> %291) #9, !dbg !15
  %293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %261, <4 x float> %241) #9, !dbg !15
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %264, <4 x float> %293) #9, !dbg !15
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %267, <4 x float> %243) #9, !dbg !15
  %296 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %270, <4 x float> %295) #9, !dbg !15
  %297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %262, <4 x float> %245) #9, !dbg !15
  %298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %265, <4 x float> %297) #9, !dbg !15
  %299 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %268, <4 x float> %247) #9, !dbg !15
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %271, <4 x float> %299) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local911 = shl nuw nsw i64 %address.local496, 5, !dbg !16
  %address.local912 = and i64 %address.local911, 8064, !dbg !16
  %address.local913 = shl nuw nsw i64 %address.local496, 3, !dbg !9
  %address.local914 = shl nuw nsw i64 %address.local502, 5, !dbg !16
  %address.local915 = sub nsw i64 %address.local913, %address.local914, !dbg !16
  %address.local916 = add nsw i64 %address.local912, %address.local915, !dbg !16
  %301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx14041405 = add nsw i64 %113, %address.local916, !dbg !16
  %.idx14041405.tr = trunc i64 %.idx14041405 to i32, !dbg !16
  %dma.buffer.offset1339 = shl i32 %.idx14041405.tr, 1, !dbg !16
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !16
  %303 = bitcast <2 x i64> %302 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1340 = shufflevector <4 x i32> %303, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1339, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %address.local918 = shl nuw nsw i32 %address.thread495, 3, !dbg !16
  %address.local919 = and i32 %address.local918, 24, !dbg !16
  %address.local921 = and i32 %address.local569, 8064, !dbg !16
  %address.local922 = or disjoint i32 %address.local919, %address.local921, !dbg !16
  %address.local923 = or disjoint i32 %address.local922, 8192, !dbg !16
  %305 = trunc nuw i64 %113 to i32, !dbg !16
  %.idx14061407.tr = or disjoint i32 %address.local923, %305, !dbg !16
  %dma.buffer.offset1337 = shl nuw nsw i32 %.idx14061407.tr, 1, !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1337, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_tail_pair !33
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_tail_pair !33
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_tail_pair !34
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !15, !frisk.k_tail_pair !34
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_tail_pair !35
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1880, align 8, !dbg !15, !frisk.k_tail_pair !35
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_tail_pair !36
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1881, align 8, !dbg !15, !frisk.k_tail_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %307, <4 x float> %266) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %309, <4 x float> %315) #9, !dbg !15
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %311, <4 x float> %272) #9, !dbg !15
  %318 = extractelement <4 x float> %316, i64 0, !dbg !15
  %319 = call float @llvm.maxnum.f32(float %318, float 0xFFF0000000000000), !dbg !8, !frisk.local_reduction !1
  %320 = extractelement <4 x float> %316, i64 1, !dbg !15
  %321 = extractelement <4 x float> %316, i64 2, !dbg !15
  %322 = extractelement <4 x float> %316, i64 3, !dbg !15
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %313, <4 x float> %317) #9, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1882, align 8, !dbg !15, !frisk.k_tail_pair !31
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1886, align 8, !dbg !15, !frisk.k_tail_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %308, <4 x float> %274) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %327 = extractelement <4 x float> %323, i64 0, !dbg !15
  %local.reduce133 = call float @llvm.maxnum.f32(float %319, float %327), !dbg !8, !frisk.local_reduction !1
  %328 = extractelement <4 x float> %323, i64 1, !dbg !15
  %local.reduce134 = call float @llvm.maxnum.f32(float %320, float %328), !dbg !8, !frisk.local_reduction !1
  %329 = extractelement <4 x float> %323, i64 2, !dbg !15
  %local.reduce135 = call float @llvm.maxnum.f32(float %321, float %329), !dbg !8, !frisk.local_reduction !1
  %330 = extractelement <4 x float> %323, i64 3, !dbg !15
  %local.reduce136 = call float @llvm.maxnum.f32(float %322, float %330), !dbg !8, !frisk.local_reduction !1
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1883, align 8, !dbg !15, !frisk.k_tail_pair !37
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1887, align 8, !dbg !15, !frisk.k_tail_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %326) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1884, align 8, !dbg !15, !frisk.k_tail_pair !38
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1888, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %312, <4 x float> %276) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %337 = extractelement <4 x float> %333, i64 0, !dbg !15
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %337), !dbg !8, !frisk.local_reduction !1
  %338 = extractelement <4 x float> %333, i64 1, !dbg !15
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %338), !dbg !8, !frisk.local_reduction !1
  %339 = extractelement <4 x float> %333, i64 2, !dbg !15
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %339), !dbg !8, !frisk.local_reduction !1
  %340 = extractelement <4 x float> %333, i64 3, !dbg !15
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %340), !dbg !8, !frisk.local_reduction !1
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1885, align 8, !dbg !15, !frisk.k_tail_pair !39
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1889, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %314, <4 x float> %336) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %324, <4 x float> %278) #9, !dbg !15
  %345 = extractelement <4 x float> %343, i64 0, !dbg !15
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %345), !dbg !8, !frisk.local_reduction !1
  %346 = extractelement <4 x float> %343, i64 1, !dbg !15
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %346), !dbg !8, !frisk.local_reduction !1
  %347 = extractelement <4 x float> %343, i64 2, !dbg !15
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %347), !dbg !8, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %343, i64 3, !dbg !15
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %348), !dbg !8, !frisk.local_reduction !1
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %331, <4 x float> %344) #9, !dbg !15
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %334, <4 x float> %280) #9, !dbg !15
  %351 = extractelement <4 x float> %349, i64 0, !dbg !15
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %351), !dbg !8, !frisk.local_reduction !1
  %352 = extractelement <4 x float> %349, i64 1, !dbg !15
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %352), !dbg !8, !frisk.local_reduction !1
  %353 = extractelement <4 x float> %349, i64 2, !dbg !15
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %353), !dbg !8, !frisk.local_reduction !1
  %354 = extractelement <4 x float> %349, i64 3, !dbg !15
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %354), !dbg !8, !frisk.local_reduction !1
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %341, <4 x float> %350) #9, !dbg !15
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %325, <4 x float> %282) #9, !dbg !15
  %357 = extractelement <4 x float> %355, i64 0, !dbg !15
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %357), !dbg !8, !frisk.local_reduction !1
  %358 = extractelement <4 x float> %355, i64 1, !dbg !15
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %358), !dbg !8, !frisk.local_reduction !1
  %359 = extractelement <4 x float> %355, i64 2, !dbg !15
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %359), !dbg !8, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %355, i64 3, !dbg !15
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce148, float %360), !dbg !8, !frisk.local_reduction !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %332, <4 x float> %356) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %335, <4 x float> %284) #9, !dbg !15
  %363 = extractelement <4 x float> %361, i64 0, !dbg !15
  %local.reduce153 = call float @llvm.maxnum.f32(float %local.reduce149, float %363), !dbg !8, !frisk.local_reduction !1
  %364 = extractelement <4 x float> %361, i64 1, !dbg !15
  %local.reduce154 = call float @llvm.maxnum.f32(float %local.reduce150, float %364), !dbg !8, !frisk.local_reduction !1
  %365 = extractelement <4 x float> %361, i64 2, !dbg !15
  %local.reduce155 = call float @llvm.maxnum.f32(float %local.reduce151, float %365), !dbg !8, !frisk.local_reduction !1
  %366 = extractelement <4 x float> %361, i64 3, !dbg !15
  %local.reduce156 = call float @llvm.maxnum.f32(float %local.reduce152, float %366), !dbg !8, !frisk.local_reduction !1
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %342, <4 x float> %362) #9, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %307, <4 x float> %286) #9, !dbg !15
  %369 = extractelement <4 x float> %367, i64 0, !dbg !15
  %local.reduce157 = call float @llvm.maxnum.f32(float %local.reduce153, float %369), !dbg !8, !frisk.local_reduction !1
  %370 = extractelement <4 x float> %367, i64 1, !dbg !15
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %370), !dbg !8, !frisk.local_reduction !1
  %371 = extractelement <4 x float> %367, i64 2, !dbg !15
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %371), !dbg !8, !frisk.local_reduction !1
  %372 = extractelement <4 x float> %367, i64 3, !dbg !15
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %372), !dbg !8, !frisk.local_reduction !1
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %local.reduce158), !dbg !8, !frisk.local_reduction !1
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce159, float %local.reduce160), !dbg !8, !frisk.local_reduction !1
  %373 = call float @llvm.maxnum.f32(float %local.reduce161, float %local.reduce162), !dbg !8, !frisk.local_reduction !1
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %309, <4 x float> %368) #9, !dbg !15
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %311, <4 x float> %288) #9, !dbg !15
  %376 = extractelement <4 x float> %374, i64 0, !dbg !15
  %377 = extractelement <4 x float> %374, i64 1, !dbg !15
  %378 = extractelement <4 x float> %374, i64 2, !dbg !15
  %379 = extractelement <4 x float> %374, i64 3, !dbg !15
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %313, <4 x float> %375) #9, !dbg !15
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %308, <4 x float> %290) #9, !dbg !15
  %382 = extractelement <4 x float> %380, i64 0, !dbg !15
  %383 = extractelement <4 x float> %380, i64 1, !dbg !15
  %384 = extractelement <4 x float> %380, i64 2, !dbg !15
  %385 = extractelement <4 x float> %380, i64 3, !dbg !15
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %310, <4 x float> %381) #9, !dbg !15
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %312, <4 x float> %292) #9, !dbg !15
  %388 = extractelement <4 x float> %386, i64 0, !dbg !15
  %389 = extractelement <4 x float> %386, i64 1, !dbg !15
  %390 = extractelement <4 x float> %386, i64 2, !dbg !15
  %391 = extractelement <4 x float> %386, i64 3, !dbg !15
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %314, <4 x float> %387) #9, !dbg !15
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %324, <4 x float> %294) #9, !dbg !15
  %394 = extractelement <4 x float> %392, i64 0, !dbg !15
  %395 = extractelement <4 x float> %392, i64 1, !dbg !15
  %396 = extractelement <4 x float> %392, i64 2, !dbg !15
  %397 = extractelement <4 x float> %392, i64 3, !dbg !15
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %331, <4 x float> %393) #9, !dbg !15
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %334, <4 x float> %296) #9, !dbg !15
  %400 = extractelement <4 x float> %398, i64 0, !dbg !15
  %401 = extractelement <4 x float> %398, i64 1, !dbg !15
  %402 = extractelement <4 x float> %398, i64 2, !dbg !15
  %403 = extractelement <4 x float> %398, i64 3, !dbg !15
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %341, <4 x float> %399) #9, !dbg !15
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %325, <4 x float> %298) #9, !dbg !15
  %406 = extractelement <4 x float> %404, i64 0, !dbg !15
  %407 = extractelement <4 x float> %404, i64 1, !dbg !15
  %408 = extractelement <4 x float> %404, i64 2, !dbg !15
  %409 = extractelement <4 x float> %404, i64 3, !dbg !15
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %332, <4 x float> %405) #9, !dbg !15
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %335, <4 x float> %300) #9, !dbg !15
  %412 = extractelement <4 x float> %410, i64 0, !dbg !15
  %413 = extractelement <4 x float> %410, i64 1, !dbg !15
  %414 = extractelement <4 x float> %410, i64 2, !dbg !15
  %415 = extractelement <4 x float> %410, i64 3, !dbg !15
  %416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %342, <4 x float> %411) #9, !dbg !15
  %417 = extractelement <4 x float> %416, i64 0, !dbg !15
  %418 = extractelement <4 x float> %416, i64 1, !dbg !15
  %419 = extractelement <4 x float> %416, i64 2, !dbg !15
  %420 = extractelement <4 x float> %416, i64 3, !dbg !15
  %421 = bitcast float %373 to i32, !dbg !8
  %422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %421), !dbg !8
  %423 = bitcast i32 %422 to float, !dbg !8
  %424 = call float @llvm.maxnum.f32(float %373, float %423), !dbg !8
  %425 = bitcast float %424 to i32, !dbg !8
  %426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %425), !dbg !8
  %427 = bitcast i32 %426 to float, !dbg !8
  %428 = call float @llvm.maxnum.f32(float %424, float %427), !dbg !8
  %429 = call float @llvm.maxnum.f32(float %376, float 0xFFF0000000000000), !dbg !8, !frisk.local_reduction !1
  %local.reduce102 = call float @llvm.maxnum.f32(float %429, float %382), !dbg !8, !frisk.local_reduction !1
  %local.reduce103 = call float @llvm.maxnum.f32(float %377, float %383), !dbg !8, !frisk.local_reduction !1
  %local.reduce104 = call float @llvm.maxnum.f32(float %378, float %384), !dbg !8, !frisk.local_reduction !1
  %local.reduce105 = call float @llvm.maxnum.f32(float %379, float %385), !dbg !8, !frisk.local_reduction !1
  %local.reduce106 = call float @llvm.maxnum.f32(float %local.reduce102, float %388), !dbg !8, !frisk.local_reduction !1
  %local.reduce107 = call float @llvm.maxnum.f32(float %local.reduce103, float %389), !dbg !8, !frisk.local_reduction !1
  %local.reduce108 = call float @llvm.maxnum.f32(float %local.reduce104, float %390), !dbg !8, !frisk.local_reduction !1
  %local.reduce109 = call float @llvm.maxnum.f32(float %local.reduce105, float %391), !dbg !8, !frisk.local_reduction !1
  %local.reduce110 = call float @llvm.maxnum.f32(float %local.reduce106, float %394), !dbg !8, !frisk.local_reduction !1
  %local.reduce111 = call float @llvm.maxnum.f32(float %local.reduce107, float %395), !dbg !8, !frisk.local_reduction !1
  %local.reduce112 = call float @llvm.maxnum.f32(float %local.reduce108, float %396), !dbg !8, !frisk.local_reduction !1
  %local.reduce113 = call float @llvm.maxnum.f32(float %local.reduce109, float %397), !dbg !8, !frisk.local_reduction !1
  %local.reduce114 = call float @llvm.maxnum.f32(float %local.reduce110, float %400), !dbg !8, !frisk.local_reduction !1
  %local.reduce115 = call float @llvm.maxnum.f32(float %local.reduce111, float %401), !dbg !8, !frisk.local_reduction !1
  %local.reduce116 = call float @llvm.maxnum.f32(float %local.reduce112, float %402), !dbg !8, !frisk.local_reduction !1
  %local.reduce117 = call float @llvm.maxnum.f32(float %local.reduce113, float %403), !dbg !8, !frisk.local_reduction !1
  %local.reduce118 = call float @llvm.maxnum.f32(float %local.reduce114, float %406), !dbg !8, !frisk.local_reduction !1
  %local.reduce119 = call float @llvm.maxnum.f32(float %local.reduce115, float %407), !dbg !8, !frisk.local_reduction !1
  %local.reduce120 = call float @llvm.maxnum.f32(float %local.reduce116, float %408), !dbg !8, !frisk.local_reduction !1
  %local.reduce121 = call float @llvm.maxnum.f32(float %local.reduce117, float %409), !dbg !8, !frisk.local_reduction !1
  %local.reduce122 = call float @llvm.maxnum.f32(float %local.reduce118, float %412), !dbg !8, !frisk.local_reduction !1
  %local.reduce123 = call float @llvm.maxnum.f32(float %local.reduce119, float %413), !dbg !8, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %local.reduce120, float %414), !dbg !8, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %local.reduce121, float %415), !dbg !8, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %local.reduce122, float %417), !dbg !8, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %418), !dbg !8, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %419), !dbg !8, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %420), !dbg !8, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %local.reduce127), !dbg !8, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce128, float %local.reduce129), !dbg !8, !frisk.local_reduction !1
  %430 = call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !8, !frisk.local_reduction !1
  %431 = bitcast float %430 to i32, !dbg !8
  %432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %431), !dbg !8
  %433 = bitcast i32 %432 to float, !dbg !8
  %434 = call float @llvm.maxnum.f32(float %430, float %433), !dbg !8
  %435 = bitcast float %434 to i32, !dbg !8
  %436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %435), !dbg !8
  %437 = bitcast i32 %436 to float, !dbg !8
  %438 = call float @llvm.maxnum.f32(float %434, float %437), !dbg !8
  %439 = extractvalue [2 x <1 x float>] %104, 0, !dbg !8
  %440 = insertelement <1 x float> poison, float %428, i64 0, !dbg !8
  %441 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %440, <1 x float> %439), !dbg !8
  %442 = extractvalue [2 x <1 x float>] %104, 1, !dbg !8
  %443 = insertelement <1 x float> poison, float %438, i64 0, !dbg !8
  %444 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %443, <1 x float> %442), !dbg !8
  %445 = shufflevector <1 x float> %441, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %446 = shufflevector <1 x float> %444, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !20
  %447 = fsub <1 x float> %439, %441, !dbg !21
  %448 = extractelement <1 x float> %447, i64 0, !dbg !22
  %449 = call float @llvm.amdgcn.exp2.f32(float %448), !dbg !22
  %450 = insertelement <4 x float> poison, float %449, i64 0, !dbg !27
  %451 = shufflevector <4 x float> %450, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %452 = extractvalue [2 x <1 x float>] %103, 0, !dbg !23
  %453 = extractelement <1 x float> %452, i64 0, !dbg !23
  %454 = fmul float %453, %449, !dbg !23
  %455 = fsub <1 x float> %442, %444, !dbg !21
  %456 = extractelement <1 x float> %455, i64 0, !dbg !22
  %457 = call float @llvm.amdgcn.exp2.f32(float %456), !dbg !22
  %458 = insertelement <4 x float> poison, float %457, i64 0, !dbg !27
  %459 = shufflevector <4 x float> %458, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %460 = extractvalue [2 x <1 x float>] %103, 1, !dbg !23
  %461 = extractelement <1 x float> %460, i64 0, !dbg !23
  %462 = fmul float %461, %457, !dbg !23
  %463 = extractvalue [2 x <8 x float>] %102, 0, !dbg !27
  %464 = shufflevector <8 x float> %463, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %465 = fmul <4 x float> %464, %451, !dbg !27
  %466 = shufflevector <8 x float> %463, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %467 = fmul <4 x float> %466, %451, !dbg !27
  %468 = extractvalue [2 x <8 x float>] %102, 1, !dbg !27
  %469 = shufflevector <8 x float> %468, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %470 = fmul <4 x float> %469, %459, !dbg !27
  %471 = shufflevector <8 x float> %468, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %472 = fmul <4 x float> %471, %459, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = add nsw i64 %117, %address.local916, !dbg !16
  %vdb.address1420 = trunc i64 %vdb.address to i32, !dbg !16
  %vdb.address1421 = shl i32 %vdb.address1420, 1, !dbg !16
  %vdb.address1426 = trunc nuw i64 %117 to i32, !dbg !16
  %vdb.address1427 = or disjoint i32 %address.local923, %vdb.address1426, !dbg !16
  %vdb.address1428 = shl nuw nsw i32 %vdb.address1427, 1, !dbg !16
  %vdb.dma.base = add i32 %38, 8192, !dbg !14
  %473 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1421, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %vdb.dma.base1433 = add i32 %46, 8192, !dbg !14
  %474 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1428, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %address.local927 = and i32 %address.local918, 120, !dbg !14
  %address.local929 = and i32 %address.local569, 1536, !dbg !14
  %address.local930 = or disjoint i32 %address.local927, %address.local929, !dbg !14
  %address.local931 = zext nneg i32 %address.local930 to i64, !dbg !14
  %address.local932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local931, !dbg !14
  %475 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %476 = bitcast <4 x i32> %475 to <8 x half>, !dbg !14
  %address.local933 = and i64 %address.local496, 60, !dbg !14
  %address.local934 = add nuw nsw i64 %address.local933, %address.local496, !dbg !14
  %address.local936.idx = shl nuw nsw i64 %address.local934, 4, !dbg !14
  %address.local936 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local936.idx, !dbg !14
  %address.local937 = lshr i64 %address.local549, 4, !dbg !14
  %address.local939.idx = mul nuw nsw i64 %address.local937, 768, !dbg !14
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local936, i64 %address.local939.idx, !dbg !14
  %address.local941.idx = mul nsw i64 %address.local502, -64, !dbg !14
  %address.local941 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local941.idx, !dbg !14
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local941, i64 256, !dbg !14
  %477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %478 = bitcast <4 x i32> %477 to <8 x half>, !dbg !14
  %address.local946 = getelementptr i8, ptr addrspace(3) %address.local941, i64 512, !dbg !14
  %479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %480 = bitcast <4 x i32> %479 to <8 x half>, !dbg !14
  %address.local950 = getelementptr i8, ptr addrspace(3) %address.local941, i64 768, !dbg !14
  %481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %482 = bitcast <4 x i32> %481 to <8 x half>, !dbg !14
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local932, i64 4096, !dbg !14
  %483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %484 = bitcast <4 x i32> %483 to <8 x half>, !dbg !14
  %address.local956 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4352, !dbg !14
  %485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %486 = bitcast <4 x i32> %485 to <8 x half>, !dbg !14
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4608, !dbg !14
  %487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %488 = bitcast <4 x i32> %487 to <8 x half>, !dbg !14
  %address.local964 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4864, !dbg !14
  %489 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %490 = shufflevector <8 x half> %476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %491 = fsub <4 x float> %316, %445, !dbg !20
  %492 = extractelement <4 x float> %491, i64 0, !dbg !24
  %493 = call float @llvm.amdgcn.exp2.f32(float %492), !dbg !24
  %494 = insertelement <4 x float> poison, float %493, i64 0, !dbg !26
  %495 = extractelement <4 x float> %491, i64 1, !dbg !24
  %496 = call float @llvm.amdgcn.exp2.f32(float %495), !dbg !24
  %497 = insertelement <4 x float> %494, float %496, i64 1, !dbg !26
  %498 = extractelement <4 x float> %491, i64 2, !dbg !24
  %499 = call float @llvm.amdgcn.exp2.f32(float %498), !dbg !24
  %500 = insertelement <4 x float> %497, float %499, i64 2, !dbg !26
  %501 = extractelement <4 x float> %491, i64 3, !dbg !24
  %502 = call float @llvm.amdgcn.exp2.f32(float %501), !dbg !24
  %503 = insertelement <4 x float> %500, float %502, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %503) #9, !dbg !26, !frisk.register_lifetimes !1
  %504 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !26
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %504) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %505 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %490, <4 x float> %465) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %506 = shufflevector <8 x half> %476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %507 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %506, <4 x float> %467) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %508 = fsub <4 x float> %374, %446, !dbg !20
  %509 = extractelement <4 x float> %508, i64 0, !dbg !24
  %510 = call float @llvm.amdgcn.exp2.f32(float %509), !dbg !24
  %511 = insertelement <4 x float> poison, float %510, i64 0, !dbg !26
  %512 = extractelement <4 x float> %508, i64 1, !dbg !24
  %513 = call float @llvm.amdgcn.exp2.f32(float %512), !dbg !24
  %514 = insertelement <4 x float> %511, float %513, i64 1, !dbg !26
  %515 = extractelement <4 x float> %508, i64 2, !dbg !24
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !24
  %517 = insertelement <4 x float> %514, float %516, i64 2, !dbg !26
  %518 = extractelement <4 x float> %508, i64 3, !dbg !24
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !24
  %520 = insertelement <4 x float> %517, float %519, i64 3, !dbg !26
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %520) #9, !dbg !26, !frisk.register_lifetimes !1
  %521 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !26
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %521) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %522 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %490, <4 x float> %470) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %506, <4 x float> %472) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %524 = shufflevector <8 x half> %478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %525 = fsub <4 x float> %323, %445, !dbg !20
  %526 = extractelement <4 x float> %525, i64 0, !dbg !24
  %527 = call float @llvm.amdgcn.exp2.f32(float %526), !dbg !24
  %local.reduce71 = fadd float %493, %527, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %528 = insertelement <4 x float> poison, float %527, i64 0, !dbg !26
  %529 = extractelement <4 x float> %525, i64 1, !dbg !24
  %530 = call float @llvm.amdgcn.exp2.f32(float %529), !dbg !24
  %local.reduce72 = fadd float %496, %530, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1977 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %531 = insertelement <4 x float> %528, float %530, i64 1, !dbg !26
  %532 = extractelement <4 x float> %525, i64 2, !dbg !24
  %533 = call float @llvm.amdgcn.exp2.f32(float %532), !dbg !24
  %local.reduce73 = fadd float %499, %533, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1978 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %534 = insertelement <4 x float> %531, float %533, i64 2, !dbg !26
  %535 = extractelement <4 x float> %525, i64 3, !dbg !24
  %536 = call float @llvm.amdgcn.exp2.f32(float %535), !dbg !24
  %local.reduce74 = fadd float %502, %536, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1979 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %537 = insertelement <4 x float> %534, float %536, i64 3, !dbg !26
  %math.result368 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %537) #9, !dbg !26, !frisk.register_lifetimes !1
  %538 = fptrunc <4 x float> %math.result368 to <4 x half>, !dbg !26
  %math.packed369 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %538) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %524, <4 x float> %505) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %540 = shufflevector <8 x half> %478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %540, <4 x float> %507) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %542 = fsub <4 x float> %380, %446, !dbg !20
  %543 = extractelement <4 x float> %542, i64 0, !dbg !24
  %544 = call float @llvm.amdgcn.exp2.f32(float %543), !dbg !24
  %local.reduce = fadd float %510, %544, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1980 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !26
  %546 = extractelement <4 x float> %542, i64 1, !dbg !24
  %547 = call float @llvm.amdgcn.exp2.f32(float %546), !dbg !24
  %local.reduce41 = fadd float %513, %547, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1981 = call float asm sideeffect "", "=v,0"(float %local.reduce41) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !26
  %549 = extractelement <4 x float> %542, i64 2, !dbg !24
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !24
  %local.reduce42 = fadd float %516, %550, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1982 = call float asm sideeffect "", "=v,0"(float %local.reduce42) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !26
  %552 = extractelement <4 x float> %542, i64 3, !dbg !24
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !24
  %local.reduce43 = fadd float %519, %553, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1983 = call float asm sideeffect "", "=v,0"(float %local.reduce43) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !26
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #9, !dbg !26, !frisk.register_lifetimes !1
  %555 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !26
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %524, <4 x float> %522) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %540, <4 x float> %523) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %558 = shufflevector <8 x half> %480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %559 = fsub <4 x float> %333, %445, !dbg !20
  %560 = extractelement <4 x float> %559, i64 0, !dbg !24
  %561 = call float @llvm.amdgcn.exp2.f32(float %560), !dbg !24
  %local.reduce75 = fadd float %k0.sum.anchor, %561, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1984 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %562 = insertelement <4 x float> poison, float %561, i64 0, !dbg !26
  %563 = extractelement <4 x float> %559, i64 1, !dbg !24
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !24
  %local.reduce76 = fadd float %k0.sum.anchor1977, %564, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1985 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %565 = insertelement <4 x float> %562, float %564, i64 1, !dbg !26
  %566 = extractelement <4 x float> %559, i64 2, !dbg !24
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !24
  %local.reduce77 = fadd float %k0.sum.anchor1978, %567, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1986 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> %565, float %567, i64 2, !dbg !26
  %569 = extractelement <4 x float> %559, i64 3, !dbg !24
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !24
  %local.reduce78 = fadd float %k0.sum.anchor1979, %570, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1987 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 3, !dbg !26
  %math.result370 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %571) #9, !dbg !26, !frisk.register_lifetimes !1
  %572 = fptrunc <4 x float> %math.result370 to <4 x half>, !dbg !26
  %math.packed371 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %572) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %573 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %558, <4 x float> %539) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %574 = shufflevector <8 x half> %480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %575 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %574, <4 x float> %541) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %576 = fsub <4 x float> %386, %446, !dbg !20
  %577 = extractelement <4 x float> %576, i64 0, !dbg !24
  %578 = call float @llvm.amdgcn.exp2.f32(float %577), !dbg !24
  %local.reduce44 = fadd float %k0.sum.anchor1980, %578, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1988 = call float asm sideeffect "", "=v,0"(float %local.reduce44) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %579 = insertelement <4 x float> poison, float %578, i64 0, !dbg !26
  %580 = extractelement <4 x float> %576, i64 1, !dbg !24
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !24
  %local.reduce45 = fadd float %k0.sum.anchor1981, %581, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1989 = call float asm sideeffect "", "=v,0"(float %local.reduce45) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %582 = insertelement <4 x float> %579, float %581, i64 1, !dbg !26
  %583 = extractelement <4 x float> %576, i64 2, !dbg !24
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !24
  %local.reduce46 = fadd float %k0.sum.anchor1982, %584, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1990 = call float asm sideeffect "", "=v,0"(float %local.reduce46) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %585 = insertelement <4 x float> %582, float %584, i64 2, !dbg !26
  %586 = extractelement <4 x float> %576, i64 3, !dbg !24
  %587 = call float @llvm.amdgcn.exp2.f32(float %586), !dbg !24
  %local.reduce47 = fadd float %k0.sum.anchor1983, %587, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1991 = call float asm sideeffect "", "=v,0"(float %local.reduce47) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %588 = insertelement <4 x float> %585, float %587, i64 3, !dbg !26
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %588) #9, !dbg !26, !frisk.register_lifetimes !1
  %589 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !26
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %589) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %558, <4 x float> %556) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %591 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %574, <4 x float> %557) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %592 = shufflevector <8 x half> %482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %593 = fsub <4 x float> %343, %445, !dbg !20
  %594 = extractelement <4 x float> %593, i64 0, !dbg !24
  %595 = call float @llvm.amdgcn.exp2.f32(float %594), !dbg !24
  %local.reduce79 = fadd float %k0.sum.anchor1984, %595, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1992 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %596 = insertelement <4 x float> poison, float %595, i64 0, !dbg !26
  %597 = extractelement <4 x float> %593, i64 1, !dbg !24
  %598 = call float @llvm.amdgcn.exp2.f32(float %597), !dbg !24
  %local.reduce80 = fadd float %k0.sum.anchor1985, %598, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1993 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %599 = insertelement <4 x float> %596, float %598, i64 1, !dbg !26
  %600 = extractelement <4 x float> %593, i64 2, !dbg !24
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !24
  %local.reduce81 = fadd float %k0.sum.anchor1986, %601, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1994 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> %599, float %601, i64 2, !dbg !26
  %603 = extractelement <4 x float> %593, i64 3, !dbg !24
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !24
  %local.reduce82 = fadd float %k0.sum.anchor1987, %604, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1995 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %605 = insertelement <4 x float> %602, float %604, i64 3, !dbg !26
  %math.result372 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %605) #9, !dbg !26, !frisk.register_lifetimes !1
  %606 = fptrunc <4 x float> %math.result372 to <4 x half>, !dbg !26
  %math.packed373 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %606) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %607 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %592, <4 x float> %573) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %608 = shufflevector <8 x half> %482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %609 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %608, <4 x float> %575) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %610 = fsub <4 x float> %392, %446, !dbg !20
  %611 = extractelement <4 x float> %610, i64 0, !dbg !24
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !24
  %local.reduce48 = fadd float %k0.sum.anchor1988, %612, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1996 = call float asm sideeffect "", "=v,0"(float %local.reduce48) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %613 = insertelement <4 x float> poison, float %612, i64 0, !dbg !26
  %614 = extractelement <4 x float> %610, i64 1, !dbg !24
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !24
  %local.reduce49 = fadd float %k0.sum.anchor1989, %615, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1997 = call float asm sideeffect "", "=v,0"(float %local.reduce49) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %616 = insertelement <4 x float> %613, float %615, i64 1, !dbg !26
  %617 = extractelement <4 x float> %610, i64 2, !dbg !24
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !24
  %local.reduce50 = fadd float %k0.sum.anchor1990, %618, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1998 = call float asm sideeffect "", "=v,0"(float %local.reduce50) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> %616, float %618, i64 2, !dbg !26
  %620 = extractelement <4 x float> %610, i64 3, !dbg !24
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !24
  %local.reduce51 = fadd float %k0.sum.anchor1991, %621, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1999 = call float asm sideeffect "", "=v,0"(float %local.reduce51) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 3, !dbg !26
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %622) #9, !dbg !26, !frisk.register_lifetimes !1
  %623 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !26
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %623) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %592, <4 x float> %590) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %608, <4 x float> %591) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %626 = shufflevector <8 x half> %484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = fsub <4 x float> %349, %445, !dbg !20
  %628 = extractelement <4 x float> %627, i64 0, !dbg !24
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !24
  %local.reduce83 = fadd float %k0.sum.anchor1992, %629, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2000 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %630 = insertelement <4 x float> poison, float %629, i64 0, !dbg !26
  %631 = extractelement <4 x float> %627, i64 1, !dbg !24
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !24
  %local.reduce84 = fadd float %k0.sum.anchor1993, %632, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2001 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %633 = insertelement <4 x float> %630, float %632, i64 1, !dbg !26
  %634 = extractelement <4 x float> %627, i64 2, !dbg !24
  %635 = call float @llvm.amdgcn.exp2.f32(float %634), !dbg !24
  %local.reduce85 = fadd float %k0.sum.anchor1994, %635, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2002 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %636 = insertelement <4 x float> %633, float %635, i64 2, !dbg !26
  %637 = extractelement <4 x float> %627, i64 3, !dbg !24
  %638 = call float @llvm.amdgcn.exp2.f32(float %637), !dbg !24
  %local.reduce86 = fadd float %k0.sum.anchor1995, %638, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2003 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %639 = insertelement <4 x float> %636, float %638, i64 3, !dbg !26
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %639) #9, !dbg !26, !frisk.register_lifetimes !1
  %640 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !26
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %640) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %626, <4 x float> %607) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %642 = shufflevector <8 x half> %484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %643 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %642, <4 x float> %609) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %644 = fsub <4 x float> %398, %446, !dbg !20
  %645 = extractelement <4 x float> %644, i64 0, !dbg !24
  %646 = call float @llvm.amdgcn.exp2.f32(float %645), !dbg !24
  %local.reduce52 = fadd float %k0.sum.anchor1996, %646, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2004 = call float asm sideeffect "", "=v,0"(float %local.reduce52) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %647 = insertelement <4 x float> poison, float %646, i64 0, !dbg !26
  %648 = extractelement <4 x float> %644, i64 1, !dbg !24
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !24
  %local.reduce53 = fadd float %k0.sum.anchor1997, %649, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2005 = call float asm sideeffect "", "=v,0"(float %local.reduce53) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %650 = insertelement <4 x float> %647, float %649, i64 1, !dbg !26
  %651 = extractelement <4 x float> %644, i64 2, !dbg !24
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !24
  %local.reduce54 = fadd float %k0.sum.anchor1998, %652, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2006 = call float asm sideeffect "", "=v,0"(float %local.reduce54) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %653 = insertelement <4 x float> %650, float %652, i64 2, !dbg !26
  %654 = extractelement <4 x float> %644, i64 3, !dbg !24
  %655 = call float @llvm.amdgcn.exp2.f32(float %654), !dbg !24
  %local.reduce55 = fadd float %k0.sum.anchor1999, %655, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2007 = call float asm sideeffect "", "=v,0"(float %local.reduce55) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %656 = insertelement <4 x float> %653, float %655, i64 3, !dbg !26
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %656) #9, !dbg !26, !frisk.register_lifetimes !1
  %657 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !26
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %657) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %626, <4 x float> %624) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %642, <4 x float> %625) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %660 = shufflevector <8 x half> %486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %661 = fsub <4 x float> %355, %445, !dbg !20
  %662 = extractelement <4 x float> %661, i64 0, !dbg !24
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !24
  %local.reduce87 = fadd float %k0.sum.anchor2000, %663, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2008 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %664 = insertelement <4 x float> poison, float %663, i64 0, !dbg !26
  %665 = extractelement <4 x float> %661, i64 1, !dbg !24
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !24
  %local.reduce88 = fadd float %k0.sum.anchor2001, %666, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2009 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> %664, float %666, i64 1, !dbg !26
  %668 = extractelement <4 x float> %661, i64 2, !dbg !24
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !24
  %local.reduce89 = fadd float %k0.sum.anchor2002, %669, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2010 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %667, float %669, i64 2, !dbg !26
  %671 = extractelement <4 x float> %661, i64 3, !dbg !24
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !24
  %local.reduce90 = fadd float %k0.sum.anchor2003, %672, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2011 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %673 = insertelement <4 x float> %670, float %672, i64 3, !dbg !26
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %673) #9, !dbg !26, !frisk.register_lifetimes !1
  %674 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !26
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %674) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %675 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %660, <4 x float> %641) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %676 = shufflevector <8 x half> %486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %676, <4 x float> %643) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %678 = fsub <4 x float> %404, %446, !dbg !20
  %679 = extractelement <4 x float> %678, i64 0, !dbg !24
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !24
  %local.reduce56 = fadd float %k0.sum.anchor2004, %680, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2012 = call float asm sideeffect "", "=v,0"(float %local.reduce56) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %681 = insertelement <4 x float> poison, float %680, i64 0, !dbg !26
  %682 = extractelement <4 x float> %678, i64 1, !dbg !24
  %683 = call float @llvm.amdgcn.exp2.f32(float %682), !dbg !24
  %local.reduce57 = fadd float %k0.sum.anchor2005, %683, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2013 = call float asm sideeffect "", "=v,0"(float %local.reduce57) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %684 = insertelement <4 x float> %681, float %683, i64 1, !dbg !26
  %685 = extractelement <4 x float> %678, i64 2, !dbg !24
  %686 = call float @llvm.amdgcn.exp2.f32(float %685), !dbg !24
  %local.reduce58 = fadd float %k0.sum.anchor2006, %686, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2014 = call float asm sideeffect "", "=v,0"(float %local.reduce58) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %687 = insertelement <4 x float> %684, float %686, i64 2, !dbg !26
  %688 = extractelement <4 x float> %678, i64 3, !dbg !24
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !24
  %local.reduce59 = fadd float %k0.sum.anchor2007, %689, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2015 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %690 = insertelement <4 x float> %687, float %689, i64 3, !dbg !26
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %690) #9, !dbg !26, !frisk.register_lifetimes !1
  %691 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !26
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %691) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %692 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %660, <4 x float> %658) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %693 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %676, <4 x float> %659) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %694 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %695 = fsub <4 x float> %361, %445, !dbg !20
  %696 = extractelement <4 x float> %695, i64 0, !dbg !24
  %697 = call float @llvm.amdgcn.exp2.f32(float %696), !dbg !24
  %local.reduce91 = fadd float %k0.sum.anchor2008, %697, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2016 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %698 = insertelement <4 x float> poison, float %697, i64 0, !dbg !26
  %699 = extractelement <4 x float> %695, i64 1, !dbg !24
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !24
  %local.reduce92 = fadd float %k0.sum.anchor2009, %700, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2017 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %701 = insertelement <4 x float> %698, float %700, i64 1, !dbg !26
  %702 = extractelement <4 x float> %695, i64 2, !dbg !24
  %703 = call float @llvm.amdgcn.exp2.f32(float %702), !dbg !24
  %local.reduce93 = fadd float %k0.sum.anchor2010, %703, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2018 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %704 = insertelement <4 x float> %701, float %703, i64 2, !dbg !26
  %705 = extractelement <4 x float> %695, i64 3, !dbg !24
  %706 = call float @llvm.amdgcn.exp2.f32(float %705), !dbg !24
  %local.reduce94 = fadd float %k0.sum.anchor2011, %706, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2019 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %707 = insertelement <4 x float> %704, float %706, i64 3, !dbg !26
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #9, !dbg !26, !frisk.register_lifetimes !1
  %708 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !26
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %694, <4 x float> %675) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %710 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %710, <4 x float> %677) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %712 = fsub <4 x float> %410, %446, !dbg !20
  %713 = extractelement <4 x float> %712, i64 0, !dbg !24
  %714 = call float @llvm.amdgcn.exp2.f32(float %713), !dbg !24
  %local.reduce60 = fadd float %k0.sum.anchor2012, %714, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2020 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %715 = insertelement <4 x float> poison, float %714, i64 0, !dbg !26
  %716 = extractelement <4 x float> %712, i64 1, !dbg !24
  %717 = call float @llvm.amdgcn.exp2.f32(float %716), !dbg !24
  %local.reduce61 = fadd float %k0.sum.anchor2013, %717, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2021 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %718 = insertelement <4 x float> %715, float %717, i64 1, !dbg !26
  %719 = extractelement <4 x float> %712, i64 2, !dbg !24
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !24
  %local.reduce62 = fadd float %k0.sum.anchor2014, %720, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2022 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %721 = insertelement <4 x float> %718, float %720, i64 2, !dbg !26
  %722 = extractelement <4 x float> %712, i64 3, !dbg !24
  %723 = call float @llvm.amdgcn.exp2.f32(float %722), !dbg !24
  %local.reduce63 = fadd float %k0.sum.anchor2015, %723, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2023 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %724 = insertelement <4 x float> %721, float %723, i64 3, !dbg !26
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %724) #9, !dbg !26, !frisk.register_lifetimes !1
  %725 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !26
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %725) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %726 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %694, <4 x float> %692) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %727 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %710, <4 x float> %693) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %728 = bitcast <4 x i32> %489 to <8 x half>, !dbg !14
  %729 = shufflevector <8 x half> %728, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %730 = fsub <4 x float> %367, %445, !dbg !20
  %731 = extractelement <4 x float> %730, i64 0, !dbg !24
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !24
  %local.reduce95 = fadd float %k0.sum.anchor2016, %732, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2024 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %733 = insertelement <4 x float> poison, float %732, i64 0, !dbg !26
  %734 = extractelement <4 x float> %730, i64 1, !dbg !24
  %735 = call float @llvm.amdgcn.exp2.f32(float %734), !dbg !24
  %local.reduce96 = fadd float %k0.sum.anchor2017, %735, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2025 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce99 = fadd float %k0.sum.anchor2024, %k0.sum.anchor2025, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2026 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %736 = insertelement <4 x float> %733, float %735, i64 1, !dbg !26
  %737 = extractelement <4 x float> %730, i64 2, !dbg !24
  %738 = call float @llvm.amdgcn.exp2.f32(float %737), !dbg !24
  %local.reduce97 = fadd float %k0.sum.anchor2018, %738, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2027 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %739 = insertelement <4 x float> %736, float %738, i64 2, !dbg !26
  %740 = extractelement <4 x float> %730, i64 3, !dbg !24
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !24
  %local.reduce98 = fadd float %k0.sum.anchor2019, %741, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2028 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce100 = fadd float %k0.sum.anchor2027, %k0.sum.anchor2028, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2029 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %742 = fadd float %k0.sum.anchor2026, %k0.sum.anchor2029, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2030 = call float asm sideeffect "", "=v,0"(float %742) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %743 = fadd float %k0.sum.anchor2030, %454, !dbg !25
  %k0.sum.anchor2031 = call float asm sideeffect "", "=v,0"(float %743) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %744 = insertelement <1 x float> poison, float %k0.sum.anchor2031, i64 0, !dbg !6
  %745 = insertvalue [2 x <1 x float>] poison, <1 x float> %744, 0, !dbg !6
  %746 = insertelement <4 x float> %739, float %741, i64 3, !dbg !26
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %746) #9, !dbg !26, !frisk.register_lifetimes !1
  %747 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !26
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %747) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %748 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %729, <4 x float> %709) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %749 = shufflevector <8 x half> %728, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %750 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %749, <4 x float> %711) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %751 = fsub <4 x float> %416, %446, !dbg !20
  %752 = extractelement <4 x float> %751, i64 0, !dbg !24
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !24
  %local.reduce64 = fadd float %k0.sum.anchor2020, %753, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2032 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %754 = insertelement <4 x float> poison, float %753, i64 0, !dbg !26
  %755 = extractelement <4 x float> %751, i64 1, !dbg !24
  %756 = call float @llvm.amdgcn.exp2.f32(float %755), !dbg !24
  %local.reduce65 = fadd float %k0.sum.anchor2021, %756, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2033 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce68 = fadd float %k0.sum.anchor2032, %k0.sum.anchor2033, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2034 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %757 = insertelement <4 x float> %754, float %756, i64 1, !dbg !26
  %758 = extractelement <4 x float> %751, i64 2, !dbg !24
  %759 = call float @llvm.amdgcn.exp2.f32(float %758), !dbg !24
  %local.reduce66 = fadd float %k0.sum.anchor2022, %759, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2035 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %760 = insertelement <4 x float> %757, float %759, i64 2, !dbg !26
  %761 = extractelement <4 x float> %751, i64 3, !dbg !24
  %762 = call float @llvm.amdgcn.exp2.f32(float %761), !dbg !24
  %local.reduce67 = fadd float %k0.sum.anchor2023, %762, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2036 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce69 = fadd float %k0.sum.anchor2035, %k0.sum.anchor2036, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2037 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %763 = fadd float %k0.sum.anchor2034, %k0.sum.anchor2037, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2038 = call float asm sideeffect "", "=v,0"(float %763) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %764 = fadd float %k0.sum.anchor2038, %462, !dbg !25
  %k0.sum.anchor2039 = call float asm sideeffect "", "=v,0"(float %764) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %765 = insertelement <1 x float> poison, float %k0.sum.anchor2039, i64 0, !dbg !6
  %766 = insertvalue [2 x <1 x float>] %745, <1 x float> %765, 1, !dbg !6
  %767 = insertelement <4 x float> %760, float %762, i64 3, !dbg !26
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %767) #9, !dbg !26, !frisk.register_lifetimes !1
  %768 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !26
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %768) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %729, <4 x float> %726) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %749, <4 x float> %727) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %771 = extractvalue [2 x <8 x float>] %105, 0, !dbg !27
  %772 = shufflevector <8 x float> %771, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %773 = fmul <4 x float> %772, %451, !dbg !27
  %774 = shufflevector <8 x float> %771, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %775 = fmul <4 x float> %774, %451, !dbg !27
  %776 = extractvalue [2 x <8 x float>] %105, 1, !dbg !27
  %777 = shufflevector <8 x float> %776, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %778 = fmul <4 x float> %777, %459, !dbg !27
  %779 = shufflevector <8 x float> %776, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %780 = fmul <4 x float> %779, %459, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1441 = add nsw i64 %195, %address.local916, !dbg !16
  %vdb.address1442 = trunc i64 %vdb.address1441 to i32, !dbg !16
  %vdb.address1443 = shl i32 %vdb.address1442, 1, !dbg !16
  %vdb.address1448 = trunc nuw i64 %195 to i32, !dbg !16
  %vdb.address1449 = or disjoint i32 %address.local923, %vdb.address1448, !dbg !16
  %vdb.address1450 = shl nuw nsw i32 %vdb.address1449, 1, !dbg !16
  %781 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %vdb.address1443, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %782 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %vdb.address1450, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local932, i64 8192, !dbg !14
  %783 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %784 = bitcast <4 x i32> %783 to <8 x half>, !dbg !14
  %vdb.read1434 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8448, !dbg !14
  %785 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %786 = bitcast <4 x i32> %785 to <8 x half>, !dbg !14
  %vdb.read1435 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8704, !dbg !14
  %787 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %788 = bitcast <4 x i32> %787 to <8 x half>, !dbg !14
  %vdb.read1436 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8960, !dbg !14
  %789 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %790 = bitcast <4 x i32> %789 to <8 x half>, !dbg !14
  %vdb.read1437 = getelementptr i8, ptr addrspace(3) %address.local932, i64 12288, !dbg !14
  %791 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %792 = bitcast <4 x i32> %791 to <8 x half>, !dbg !14
  %vdb.read1438 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12544, !dbg !14
  %793 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %794 = bitcast <4 x i32> %793 to <8 x half>, !dbg !14
  %vdb.read1439 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12800, !dbg !14
  %795 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %796 = bitcast <4 x i32> %795 to <8 x half>, !dbg !14
  %vdb.read1440 = getelementptr i8, ptr addrspace(3) %address.local941, i64 13056, !dbg !14
  %797 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %798 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %798, <4 x float> %773) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = shufflevector <8 x half> %784, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %800, <4 x float> %775) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %798, <4 x float> %778) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %803 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %800, <4 x float> %780) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %804 = shufflevector <8 x half> %786, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %805 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %804, <4 x float> %799) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %806 = shufflevector <8 x half> %786, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %807 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %806, <4 x float> %801) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %808 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %804, <4 x float> %802) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %809 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %806, <4 x float> %803) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %810 = shufflevector <8 x half> %788, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %811 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %810, <4 x float> %805) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %812 = shufflevector <8 x half> %788, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %813 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %812, <4 x float> %807) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %814 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %810, <4 x float> %808) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %815 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %812, <4 x float> %809) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %816 = shufflevector <8 x half> %790, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %817 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %816, <4 x float> %811) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %818 = shufflevector <8 x half> %790, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %819 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %818, <4 x float> %813) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %820 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %816, <4 x float> %814) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %821 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %818, <4 x float> %815) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %822 = shufflevector <8 x half> %792, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %822, <4 x float> %817) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %824 = shufflevector <8 x half> %792, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %824, <4 x float> %819) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %822, <4 x float> %820) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %824, <4 x float> %821) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = shufflevector <8 x half> %794, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %828, <4 x float> %823) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %830 = shufflevector <8 x half> %794, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %830, <4 x float> %825) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %832 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %828, <4 x float> %826) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %830, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %834, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %836, <4 x float> %831) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %834, <4 x float> %832) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %836, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = bitcast <4 x i32> %797 to <8 x half>, !dbg !14
  %841 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %841, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %843, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %841, <4 x float> %838) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %843, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %847 = extractvalue [2 x <8 x float>] %106, 0, !dbg !27
  %848 = shufflevector <8 x float> %847, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %849 = fmul <4 x float> %848, %451, !dbg !27
  %850 = shufflevector <8 x float> %847, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %851 = fmul <4 x float> %850, %451, !dbg !27
  %852 = extractvalue [2 x <8 x float>] %106, 1, !dbg !27
  %853 = shufflevector <8 x float> %852, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %854 = fmul <4 x float> %853, %459, !dbg !27
  %855 = shufflevector <8 x float> %852, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %856 = fmul <4 x float> %855, %459, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1455 = add nsw i64 %248, %address.local916, !dbg !16
  %vdb.address1456 = trunc i64 %vdb.address1455 to i32, !dbg !16
  %vdb.address1457 = shl i32 %vdb.address1456, 1, !dbg !16
  %vdb.address1462 = trunc nuw i64 %248 to i32, !dbg !16
  %vdb.address1463 = or disjoint i32 %address.local923, %vdb.address1462, !dbg !16
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !16
  %857 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1457, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %858 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !14
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !14
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !14
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !14
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !14
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %874 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %874, <4 x float> %849) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %876, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %874, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %876, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %880, <4 x float> %875) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %882, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %880, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %882, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %886, <4 x float> %881) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %888, <4 x float> %883) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %886, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %888, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %892, <4 x float> %887) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %894, <4 x float> %889) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %892, <4 x float> %890) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %894, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %898, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %900, <4 x float> %895) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %898, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %900, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %904, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %906, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %904, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %906, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %910, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %912, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %910, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %912, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = bitcast <4 x i32> %873 to <8 x half>, !dbg !14
  %917 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %917, <4 x float> %911) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %919, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %917, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %919, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %923 = extractvalue [2 x <8 x float>] %107, 0, !dbg !27
  %924 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %925 = fmul <4 x float> %924, %451, !dbg !27
  %926 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %927 = fmul <4 x float> %926, %451, !dbg !27
  %928 = extractvalue [2 x <8 x float>] %107, 1, !dbg !27
  %929 = shufflevector <8 x float> %928, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %930 = fmul <4 x float> %929, %459, !dbg !27
  %931 = shufflevector <8 x float> %928, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %932 = fmul <4 x float> %931, %459, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %933 = add nuw nsw i64 %101, 1, !dbg !6
  %k0.has.next = icmp ult i64 %933, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %109
  %k0.next. = shl nuw nsw i64 %933, 14, !dbg !13
  %k0.next.2050 = or disjoint i64 %111, %k0.next., !dbg !13
  %k0.next.kdma.source1495 = or disjoint i64 %k0.next.2050, %kdma.source1494, !dbg !13
  %k0.next.kdma.source1496 = trunc nuw i64 %k0.next.kdma.source1495 to i32, !dbg !13
  %k0.next.kdma.source1497 = shl nuw nsw i32 %k0.next.kdma.source1496, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1497, %kdma.intra, !dbg !13
  %k0.next.2051 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1515 = or disjoint i64 %k0.next.2050, %kdma.source1514, !dbg !13
  %k0.next.kdma.source1516 = trunc nuw i64 %k0.next.kdma.source1515 to i32, !dbg !13
  %k0.next.kdma.source1517 = shl nuw nsw i32 %k0.next.kdma.source1516, 1, !dbg !13
  %k0.next.kdma.offset1518 = or disjoint i32 %k0.next.kdma.source1517, %kdma.intra.hi, !dbg !13
  %k0.next.2052 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1519, i32 %k0.next.kdma.offset1518, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1539 = or disjoint i64 %k0.next.2050, %kdma.source1538, !dbg !13
  %k0.next.kdma.source1540 = trunc nuw i64 %k0.next.kdma.source1539 to i32, !dbg !13
  %k0.next.kdma.source1541 = shl nuw nsw i32 %k0.next.kdma.source1540, 1, !dbg !13
  %k0.next.kdma.offset1542 = add nuw i32 %k0.next.kdma.source1541, %kdma.intra, !dbg !13
  %k0.next.2053 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.next.kdma.offset1542, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1563 = or disjoint i64 %k0.next.2050, %kdma.source1562, !dbg !13
  %k0.next.kdma.source1564 = trunc nuw i64 %k0.next.kdma.source1563 to i32, !dbg !13
  %k0.next.kdma.source1565 = shl nuw nsw i32 %k0.next.kdma.source1564, 1, !dbg !13
  %k0.next.kdma.offset1566 = add nuw i32 %k0.next.kdma.source1565, %kdma.intra.hi, !dbg !13
  %k0.next.2054 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1567, i32 %k0.next.kdma.offset1566, <4 x i32> %dma.buffer.rsrc1356) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %109, %k0.issue
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !14
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !14
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !14
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !14
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !14
  %944 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !14
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !14
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %949 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %949, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %951, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %949, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %951, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %955, <4 x float> %950) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %957, <4 x float> %952) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %955, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %957, <4 x float> %954) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %961, <4 x float> %956) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %963, <4 x float> %958) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %961, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %963, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %967, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %969, <4 x float> %964) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %967, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %969, <4 x float> %966) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %973, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %975, <4 x float> %970) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %973, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %975, <4 x float> %972) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %979, <4 x float> %974) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %981, <4 x float> %976) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %979, <4 x float> %977) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %981, <4 x float> %978) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %985, <4 x float> %980) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %987, <4 x float> %982) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %985, <4 x float> %983) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %987, <4 x float> %984) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = bitcast <4 x i32> %948 to <8 x half>, !dbg !14
  %992 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %992, <4 x float> %986) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %994, <4 x float> %988) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %992, <4 x float> %989) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %994, <4 x float> %990) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %998 = shufflevector <4 x float> %748, <4 x float> %750, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %999 = insertvalue [2 x <8 x float>] poison, <8 x float> %998, 0, !dbg !6
  %1000 = shufflevector <4 x float> %769, <4 x float> %770, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1001 = insertvalue [2 x <8 x float>] %999, <8 x float> %1000, 1, !dbg !6
  %1002 = insertvalue [2 x <1 x float>] poison, <1 x float> %441, 0, !dbg !6
  %1003 = insertvalue [2 x <1 x float>] %1002, <1 x float> %444, 1, !dbg !6
  %1004 = shufflevector <4 x float> %842, <4 x float> %844, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1005 = insertvalue [2 x <8 x float>] poison, <8 x float> %1004, 0, !dbg !6
  %1006 = shufflevector <4 x float> %845, <4 x float> %846, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1007 = insertvalue [2 x <8 x float>] %1005, <8 x float> %1006, 1, !dbg !6
  %1008 = shufflevector <4 x float> %918, <4 x float> %920, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1009 = insertvalue [2 x <8 x float>] poison, <8 x float> %1008, 0, !dbg !6
  %1010 = shufflevector <4 x float> %921, <4 x float> %922, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1011 = insertvalue [2 x <8 x float>] %1009, <8 x float> %1010, 1, !dbg !6
  %1012 = shufflevector <4 x float> %993, <4 x float> %995, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1013 = insertvalue [2 x <8 x float>] poison, <8 x float> %1012, 0, !dbg !6
  %1014 = shufflevector <4 x float> %996, <4 x float> %997, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1015 = insertvalue [2 x <8 x float>] %1013, <8 x float> %1014, 1, !dbg !6
  br label %100, !dbg !6

1016:                                             ; preds = %100
  %lane.sum.field166 = extractvalue [2 x <1 x float>] %103, 1, !dbg !6
  %lane.sum.local167 = extractelement <1 x float> %lane.sum.field166, i64 0, !dbg !6
  %1017 = shl nuw nsw i32 %8, 2, !dbg !6
  %1018 = xor i32 %1017, 64, !dbg !6
  %1019 = bitcast float %lane.sum.local167 to i32, !dbg !6
  %lane.sum.shuffle170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1018, i32 %1019), !dbg !6
  %1020 = bitcast i32 %lane.sum.shuffle170 to float, !dbg !6
  %lane.sum.final171 = fadd float %lane.sum.local167, %1020, !dbg !6
  %1021 = xor i32 %1017, 128, !dbg !6
  %1022 = bitcast float %lane.sum.final171 to i32, !dbg !6
  %lane.sum.shuffle172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1021, i32 %1022), !dbg !6
  %1023 = bitcast i32 %lane.sum.shuffle172 to float, !dbg !6
  %lane.sum.final173 = fadd float %lane.sum.final171, %1023, !dbg !6
  %1024 = insertelement <1 x float> poison, float %lane.sum.final173, i64 0, !dbg !6
  %lane.sum.field = extractvalue [2 x <1 x float>] %103, 0, !dbg !6
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !6
  %1025 = bitcast float %lane.sum.local to i32, !dbg !6
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1018, i32 %1025), !dbg !6
  %1026 = bitcast i32 %lane.sum.shuffle to float, !dbg !6
  %lane.sum.final = fadd float %lane.sum.local, %1026, !dbg !6
  %1027 = bitcast float %lane.sum.final to i32, !dbg !6
  %lane.sum.shuffle164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1021, i32 %1027), !dbg !6
  %1028 = bitcast i32 %lane.sum.shuffle164 to float, !dbg !6
  %lane.sum.final165 = fadd float %lane.sum.final, %1028, !dbg !6
  %1029 = insertelement <1 x float> poison, float %lane.sum.final165, i64 0, !dbg !6
  %1030 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1030, 0, !dbg !6
  %1031 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1031, 0, !dbg !6
  %1032 = extractvalue [2 x <8 x float>] %107, 0, !dbg !6
  %1033 = extractelement <8 x float> %1032, i64 0, !dbg !6
  %1034 = extractelement <8 x float> %1032, i64 1, !dbg !6
  %1035 = extractelement <8 x float> %1032, i64 2, !dbg !6
  %1036 = extractelement <8 x float> %1032, i64 3, !dbg !6
  %1037 = select i1 %.not, float %1034, float %1033, !dbg !6
  %1038 = bitcast float %1037 to i32, !dbg !6
  %1039 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1038), !dbg !6
  %1040 = bitcast i32 %1039 to float, !dbg !6
  %1041 = select i1 %.not, float %1033, float %1040, !dbg !6
  %1042 = select i1 %.not, float %1040, float %1034, !dbg !6
  %1043 = select i1 %.not, float %1036, float %1035, !dbg !6
  %1044 = bitcast float %1043 to i32, !dbg !6
  %1045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1044), !dbg !6
  %1046 = bitcast i32 %1045 to float, !dbg !6
  %1047 = select i1 %.not, float %1035, float %1046, !dbg !6
  %1048 = select i1 %.not, float %1046, float %1036, !dbg !6
  %1049 = select i1 %.not40, float %1047, float %1041, !dbg !6
  %1050 = bitcast float %1049 to i32, !dbg !6
  %1051 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1050), !dbg !6
  %1052 = bitcast i32 %1051 to float, !dbg !6
  %1053 = select i1 %.not40, float %1041, float %1052, !dbg !6
  %1054 = select i1 %.not40, float %1052, float %1047, !dbg !6
  %1055 = select i1 %.not40, float %1048, float %1042, !dbg !6
  %1056 = bitcast float %1055 to i32, !dbg !6
  %1057 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1056), !dbg !6
  %1058 = bitcast i32 %1057 to float, !dbg !6
  %1059 = select i1 %.not40, float %1042, float %1058, !dbg !6
  %1060 = select i1 %.not40, float %1058, float %1048, !dbg !6
  %1061 = extractelement <8 x float> %1032, i64 4, !dbg !6
  %1062 = extractelement <8 x float> %1032, i64 5, !dbg !6
  %1063 = extractelement <8 x float> %1032, i64 6, !dbg !6
  %1064 = extractelement <8 x float> %1032, i64 7, !dbg !6
  %1065 = select i1 %.not, float %1062, float %1061, !dbg !6
  %1066 = bitcast float %1065 to i32, !dbg !6
  %1067 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1066), !dbg !6
  %1068 = bitcast i32 %1067 to float, !dbg !6
  %1069 = select i1 %.not, float %1061, float %1068, !dbg !6
  %1070 = select i1 %.not, float %1068, float %1062, !dbg !6
  %1071 = select i1 %.not, float %1064, float %1063, !dbg !6
  %1072 = bitcast float %1071 to i32, !dbg !6
  %1073 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1072), !dbg !6
  %1074 = bitcast i32 %1073 to float, !dbg !6
  %1075 = select i1 %.not, float %1063, float %1074, !dbg !6
  %1076 = select i1 %.not, float %1074, float %1064, !dbg !6
  %1077 = select i1 %.not40, float %1075, float %1069, !dbg !6
  %1078 = bitcast float %1077 to i32, !dbg !6
  %1079 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1078), !dbg !6
  %1080 = bitcast i32 %1079 to float, !dbg !6
  %1081 = select i1 %.not40, float %1069, float %1080, !dbg !6
  %1082 = select i1 %.not40, float %1080, float %1075, !dbg !6
  %1083 = select i1 %.not40, float %1076, float %1070, !dbg !6
  %1084 = bitcast float %1083 to i32, !dbg !6
  %1085 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1084), !dbg !6
  %1086 = bitcast i32 %1085 to float, !dbg !6
  %1087 = select i1 %.not40, float %1070, float %1086, !dbg !6
  %1088 = select i1 %.not40, float %1086, float %1076, !dbg !6
  %1089 = extractvalue [2 x <8 x float>] %107, 1, !dbg !6
  %1090 = extractelement <8 x float> %1089, i64 0, !dbg !6
  %1091 = extractelement <8 x float> %1089, i64 1, !dbg !6
  %1092 = extractelement <8 x float> %1089, i64 2, !dbg !6
  %1093 = extractelement <8 x float> %1089, i64 3, !dbg !6
  %1094 = select i1 %.not, float %1091, float %1090, !dbg !6
  %1095 = bitcast float %1094 to i32, !dbg !6
  %1096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1095), !dbg !6
  %1097 = bitcast i32 %1096 to float, !dbg !6
  %1098 = select i1 %.not, float %1090, float %1097, !dbg !6
  %1099 = select i1 %.not, float %1097, float %1091, !dbg !6
  %1100 = select i1 %.not, float %1093, float %1092, !dbg !6
  %1101 = bitcast float %1100 to i32, !dbg !6
  %1102 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1101), !dbg !6
  %1103 = bitcast i32 %1102 to float, !dbg !6
  %1104 = select i1 %.not, float %1092, float %1103, !dbg !6
  %1105 = select i1 %.not, float %1103, float %1093, !dbg !6
  %1106 = select i1 %.not40, float %1104, float %1098, !dbg !6
  %1107 = bitcast float %1106 to i32, !dbg !6
  %1108 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1107), !dbg !6
  %1109 = bitcast i32 %1108 to float, !dbg !6
  %1110 = select i1 %.not40, float %1098, float %1109, !dbg !6
  %1111 = select i1 %.not40, float %1109, float %1104, !dbg !6
  %1112 = select i1 %.not40, float %1105, float %1099, !dbg !6
  %1113 = bitcast float %1112 to i32, !dbg !6
  %1114 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1113), !dbg !6
  %1115 = bitcast i32 %1114 to float, !dbg !6
  %1116 = select i1 %.not40, float %1099, float %1115, !dbg !6
  %1117 = select i1 %.not40, float %1115, float %1105, !dbg !6
  %1118 = extractelement <8 x float> %1089, i64 4, !dbg !6
  %1119 = extractelement <8 x float> %1089, i64 5, !dbg !6
  %1120 = extractelement <8 x float> %1089, i64 6, !dbg !6
  %1121 = extractelement <8 x float> %1089, i64 7, !dbg !6
  %1122 = select i1 %.not, float %1119, float %1118, !dbg !6
  %1123 = bitcast float %1122 to i32, !dbg !6
  %1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1123), !dbg !6
  %1125 = bitcast i32 %1124 to float, !dbg !6
  %1126 = select i1 %.not, float %1118, float %1125, !dbg !6
  %1127 = select i1 %.not, float %1125, float %1119, !dbg !6
  %1128 = select i1 %.not, float %1121, float %1120, !dbg !6
  %1129 = bitcast float %1128 to i32, !dbg !6
  %1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1129), !dbg !6
  %1131 = bitcast i32 %1130 to float, !dbg !6
  %1132 = select i1 %.not, float %1120, float %1131, !dbg !6
  %1133 = select i1 %.not, float %1131, float %1121, !dbg !6
  %1134 = select i1 %.not40, float %1132, float %1126, !dbg !6
  %1135 = bitcast float %1134 to i32, !dbg !6
  %1136 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1135), !dbg !6
  %1137 = bitcast i32 %1136 to float, !dbg !6
  %1138 = select i1 %.not40, float %1126, float %1137, !dbg !6
  %1139 = select i1 %.not40, float %1137, float %1132, !dbg !6
  %1140 = select i1 %.not40, float %1133, float %1127, !dbg !6
  %1141 = bitcast float %1140 to i32, !dbg !6
  %1142 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1141), !dbg !6
  %1143 = bitcast i32 %1142 to float, !dbg !6
  %1144 = select i1 %.not40, float %1127, float %1143, !dbg !6
  %1145 = select i1 %.not40, float %1143, float %1133, !dbg !6
  %1146 = extractvalue [2 x <8 x float>] %106, 0, !dbg !6
  %1147 = extractelement <8 x float> %1146, i64 0, !dbg !6
  %1148 = extractelement <8 x float> %1146, i64 1, !dbg !6
  %1149 = extractelement <8 x float> %1146, i64 2, !dbg !6
  %1150 = extractelement <8 x float> %1146, i64 3, !dbg !6
  %1151 = select i1 %.not, float %1148, float %1147, !dbg !6
  %1152 = bitcast float %1151 to i32, !dbg !6
  %1153 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1152), !dbg !6
  %1154 = bitcast i32 %1153 to float, !dbg !6
  %1155 = select i1 %.not, float %1147, float %1154, !dbg !6
  %1156 = select i1 %.not, float %1154, float %1148, !dbg !6
  %1157 = select i1 %.not, float %1150, float %1149, !dbg !6
  %1158 = bitcast float %1157 to i32, !dbg !6
  %1159 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1158), !dbg !6
  %1160 = bitcast i32 %1159 to float, !dbg !6
  %1161 = select i1 %.not, float %1149, float %1160, !dbg !6
  %1162 = select i1 %.not, float %1160, float %1150, !dbg !6
  %1163 = select i1 %.not40, float %1161, float %1155, !dbg !6
  %1164 = bitcast float %1163 to i32, !dbg !6
  %1165 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1164), !dbg !6
  %1166 = bitcast i32 %1165 to float, !dbg !6
  %1167 = select i1 %.not40, float %1155, float %1166, !dbg !6
  %1168 = select i1 %.not40, float %1166, float %1161, !dbg !6
  %1169 = select i1 %.not40, float %1162, float %1156, !dbg !6
  %1170 = bitcast float %1169 to i32, !dbg !6
  %1171 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1170), !dbg !6
  %1172 = bitcast i32 %1171 to float, !dbg !6
  %1173 = select i1 %.not40, float %1156, float %1172, !dbg !6
  %1174 = select i1 %.not40, float %1172, float %1162, !dbg !6
  %1175 = extractelement <8 x float> %1146, i64 4, !dbg !6
  %1176 = extractelement <8 x float> %1146, i64 5, !dbg !6
  %1177 = extractelement <8 x float> %1146, i64 6, !dbg !6
  %1178 = extractelement <8 x float> %1146, i64 7, !dbg !6
  %1179 = select i1 %.not, float %1176, float %1175, !dbg !6
  %1180 = bitcast float %1179 to i32, !dbg !6
  %1181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1180), !dbg !6
  %1182 = bitcast i32 %1181 to float, !dbg !6
  %1183 = select i1 %.not, float %1175, float %1182, !dbg !6
  %1184 = select i1 %.not, float %1182, float %1176, !dbg !6
  %1185 = select i1 %.not, float %1178, float %1177, !dbg !6
  %1186 = bitcast float %1185 to i32, !dbg !6
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1186), !dbg !6
  %1188 = bitcast i32 %1187 to float, !dbg !6
  %1189 = select i1 %.not, float %1177, float %1188, !dbg !6
  %1190 = select i1 %.not, float %1188, float %1178, !dbg !6
  %1191 = select i1 %.not40, float %1189, float %1183, !dbg !6
  %1192 = bitcast float %1191 to i32, !dbg !6
  %1193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1192), !dbg !6
  %1194 = bitcast i32 %1193 to float, !dbg !6
  %1195 = select i1 %.not40, float %1183, float %1194, !dbg !6
  %1196 = select i1 %.not40, float %1194, float %1189, !dbg !6
  %1197 = select i1 %.not40, float %1190, float %1184, !dbg !6
  %1198 = bitcast float %1197 to i32, !dbg !6
  %1199 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1198), !dbg !6
  %1200 = bitcast i32 %1199 to float, !dbg !6
  %1201 = select i1 %.not40, float %1184, float %1200, !dbg !6
  %1202 = select i1 %.not40, float %1200, float %1190, !dbg !6
  %1203 = extractvalue [2 x <8 x float>] %106, 1, !dbg !6
  %1204 = extractelement <8 x float> %1203, i64 0, !dbg !6
  %1205 = extractelement <8 x float> %1203, i64 1, !dbg !6
  %1206 = extractelement <8 x float> %1203, i64 2, !dbg !6
  %1207 = extractelement <8 x float> %1203, i64 3, !dbg !6
  %1208 = select i1 %.not, float %1205, float %1204, !dbg !6
  %1209 = bitcast float %1208 to i32, !dbg !6
  %1210 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1209), !dbg !6
  %1211 = bitcast i32 %1210 to float, !dbg !6
  %1212 = select i1 %.not, float %1204, float %1211, !dbg !6
  %1213 = select i1 %.not, float %1211, float %1205, !dbg !6
  %1214 = select i1 %.not, float %1207, float %1206, !dbg !6
  %1215 = bitcast float %1214 to i32, !dbg !6
  %1216 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1215), !dbg !6
  %1217 = bitcast i32 %1216 to float, !dbg !6
  %1218 = select i1 %.not, float %1206, float %1217, !dbg !6
  %1219 = select i1 %.not, float %1217, float %1207, !dbg !6
  %1220 = select i1 %.not40, float %1218, float %1212, !dbg !6
  %1221 = bitcast float %1220 to i32, !dbg !6
  %1222 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1221), !dbg !6
  %1223 = bitcast i32 %1222 to float, !dbg !6
  %1224 = select i1 %.not40, float %1212, float %1223, !dbg !6
  %1225 = select i1 %.not40, float %1223, float %1218, !dbg !6
  %1226 = select i1 %.not40, float %1219, float %1213, !dbg !6
  %1227 = bitcast float %1226 to i32, !dbg !6
  %1228 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1227), !dbg !6
  %1229 = bitcast i32 %1228 to float, !dbg !6
  %1230 = select i1 %.not40, float %1213, float %1229, !dbg !6
  %1231 = select i1 %.not40, float %1229, float %1219, !dbg !6
  %1232 = extractelement <8 x float> %1203, i64 4, !dbg !6
  %1233 = extractelement <8 x float> %1203, i64 5, !dbg !6
  %1234 = extractelement <8 x float> %1203, i64 6, !dbg !6
  %1235 = extractelement <8 x float> %1203, i64 7, !dbg !6
  %1236 = select i1 %.not, float %1233, float %1232, !dbg !6
  %1237 = bitcast float %1236 to i32, !dbg !6
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1237), !dbg !6
  %1239 = bitcast i32 %1238 to float, !dbg !6
  %1240 = select i1 %.not, float %1232, float %1239, !dbg !6
  %1241 = select i1 %.not, float %1239, float %1233, !dbg !6
  %1242 = select i1 %.not, float %1235, float %1234, !dbg !6
  %1243 = bitcast float %1242 to i32, !dbg !6
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1243), !dbg !6
  %1245 = bitcast i32 %1244 to float, !dbg !6
  %1246 = select i1 %.not, float %1234, float %1245, !dbg !6
  %1247 = select i1 %.not, float %1245, float %1235, !dbg !6
  %1248 = select i1 %.not40, float %1246, float %1240, !dbg !6
  %1249 = bitcast float %1248 to i32, !dbg !6
  %1250 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1249), !dbg !6
  %1251 = bitcast i32 %1250 to float, !dbg !6
  %1252 = select i1 %.not40, float %1240, float %1251, !dbg !6
  %1253 = select i1 %.not40, float %1251, float %1246, !dbg !6
  %1254 = select i1 %.not40, float %1247, float %1241, !dbg !6
  %1255 = bitcast float %1254 to i32, !dbg !6
  %1256 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1255), !dbg !6
  %1257 = bitcast i32 %1256 to float, !dbg !6
  %1258 = select i1 %.not40, float %1241, float %1257, !dbg !6
  %1259 = select i1 %.not40, float %1257, float %1247, !dbg !6
  %1260 = extractvalue [2 x <8 x float>] %105, 0, !dbg !6
  %1261 = extractelement <8 x float> %1260, i64 0, !dbg !6
  %1262 = extractelement <8 x float> %1260, i64 1, !dbg !6
  %1263 = extractelement <8 x float> %1260, i64 2, !dbg !6
  %1264 = extractelement <8 x float> %1260, i64 3, !dbg !6
  %1265 = select i1 %.not, float %1262, float %1261, !dbg !6
  %1266 = bitcast float %1265 to i32, !dbg !6
  %1267 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1266), !dbg !6
  %1268 = bitcast i32 %1267 to float, !dbg !6
  %1269 = select i1 %.not, float %1261, float %1268, !dbg !6
  %1270 = select i1 %.not, float %1268, float %1262, !dbg !6
  %1271 = select i1 %.not, float %1264, float %1263, !dbg !6
  %1272 = bitcast float %1271 to i32, !dbg !6
  %1273 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1272), !dbg !6
  %1274 = bitcast i32 %1273 to float, !dbg !6
  %1275 = select i1 %.not, float %1263, float %1274, !dbg !6
  %1276 = select i1 %.not, float %1274, float %1264, !dbg !6
  %1277 = select i1 %.not40, float %1275, float %1269, !dbg !6
  %1278 = bitcast float %1277 to i32, !dbg !6
  %1279 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1278), !dbg !6
  %1280 = bitcast i32 %1279 to float, !dbg !6
  %1281 = select i1 %.not40, float %1269, float %1280, !dbg !6
  %1282 = select i1 %.not40, float %1280, float %1275, !dbg !6
  %1283 = select i1 %.not40, float %1276, float %1270, !dbg !6
  %1284 = bitcast float %1283 to i32, !dbg !6
  %1285 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1284), !dbg !6
  %1286 = bitcast i32 %1285 to float, !dbg !6
  %1287 = select i1 %.not40, float %1270, float %1286, !dbg !6
  %1288 = select i1 %.not40, float %1286, float %1276, !dbg !6
  %1289 = extractelement <8 x float> %1260, i64 4, !dbg !6
  %1290 = extractelement <8 x float> %1260, i64 5, !dbg !6
  %1291 = extractelement <8 x float> %1260, i64 6, !dbg !6
  %1292 = extractelement <8 x float> %1260, i64 7, !dbg !6
  %1293 = select i1 %.not, float %1290, float %1289, !dbg !6
  %1294 = bitcast float %1293 to i32, !dbg !6
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1294), !dbg !6
  %1296 = bitcast i32 %1295 to float, !dbg !6
  %1297 = select i1 %.not, float %1289, float %1296, !dbg !6
  %1298 = select i1 %.not, float %1296, float %1290, !dbg !6
  %1299 = select i1 %.not, float %1292, float %1291, !dbg !6
  %1300 = bitcast float %1299 to i32, !dbg !6
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1300), !dbg !6
  %1302 = bitcast i32 %1301 to float, !dbg !6
  %1303 = select i1 %.not, float %1291, float %1302, !dbg !6
  %1304 = select i1 %.not, float %1302, float %1292, !dbg !6
  %1305 = select i1 %.not40, float %1303, float %1297, !dbg !6
  %1306 = bitcast float %1305 to i32, !dbg !6
  %1307 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1306), !dbg !6
  %1308 = bitcast i32 %1307 to float, !dbg !6
  %1309 = select i1 %.not40, float %1297, float %1308, !dbg !6
  %1310 = select i1 %.not40, float %1308, float %1303, !dbg !6
  %1311 = select i1 %.not40, float %1304, float %1298, !dbg !6
  %1312 = bitcast float %1311 to i32, !dbg !6
  %1313 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1312), !dbg !6
  %1314 = bitcast i32 %1313 to float, !dbg !6
  %1315 = select i1 %.not40, float %1298, float %1314, !dbg !6
  %1316 = select i1 %.not40, float %1314, float %1304, !dbg !6
  %1317 = extractvalue [2 x <8 x float>] %105, 1, !dbg !6
  %1318 = extractelement <8 x float> %1317, i64 0, !dbg !6
  %1319 = extractelement <8 x float> %1317, i64 1, !dbg !6
  %1320 = extractelement <8 x float> %1317, i64 2, !dbg !6
  %1321 = extractelement <8 x float> %1317, i64 3, !dbg !6
  %1322 = select i1 %.not, float %1319, float %1318, !dbg !6
  %1323 = bitcast float %1322 to i32, !dbg !6
  %1324 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1323), !dbg !6
  %1325 = bitcast i32 %1324 to float, !dbg !6
  %1326 = select i1 %.not, float %1318, float %1325, !dbg !6
  %1327 = select i1 %.not, float %1325, float %1319, !dbg !6
  %1328 = select i1 %.not, float %1321, float %1320, !dbg !6
  %1329 = bitcast float %1328 to i32, !dbg !6
  %1330 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1329), !dbg !6
  %1331 = bitcast i32 %1330 to float, !dbg !6
  %1332 = select i1 %.not, float %1320, float %1331, !dbg !6
  %1333 = select i1 %.not, float %1331, float %1321, !dbg !6
  %1334 = select i1 %.not40, float %1332, float %1326, !dbg !6
  %1335 = bitcast float %1334 to i32, !dbg !6
  %1336 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1335), !dbg !6
  %1337 = bitcast i32 %1336 to float, !dbg !6
  %1338 = select i1 %.not40, float %1326, float %1337, !dbg !6
  %1339 = select i1 %.not40, float %1337, float %1332, !dbg !6
  %1340 = select i1 %.not40, float %1333, float %1327, !dbg !6
  %1341 = bitcast float %1340 to i32, !dbg !6
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1341), !dbg !6
  %1343 = bitcast i32 %1342 to float, !dbg !6
  %1344 = select i1 %.not40, float %1327, float %1343, !dbg !6
  %1345 = select i1 %.not40, float %1343, float %1333, !dbg !6
  %1346 = extractelement <8 x float> %1317, i64 4, !dbg !6
  %1347 = extractelement <8 x float> %1317, i64 5, !dbg !6
  %1348 = extractelement <8 x float> %1317, i64 6, !dbg !6
  %1349 = extractelement <8 x float> %1317, i64 7, !dbg !6
  %1350 = select i1 %.not, float %1347, float %1346, !dbg !6
  %1351 = bitcast float %1350 to i32, !dbg !6
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1351), !dbg !6
  %1353 = bitcast i32 %1352 to float, !dbg !6
  %1354 = select i1 %.not, float %1346, float %1353, !dbg !6
  %1355 = select i1 %.not, float %1353, float %1347, !dbg !6
  %1356 = select i1 %.not, float %1349, float %1348, !dbg !6
  %1357 = bitcast float %1356 to i32, !dbg !6
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1357), !dbg !6
  %1359 = bitcast i32 %1358 to float, !dbg !6
  %1360 = select i1 %.not, float %1348, float %1359, !dbg !6
  %1361 = select i1 %.not, float %1359, float %1349, !dbg !6
  %1362 = select i1 %.not40, float %1360, float %1354, !dbg !6
  %1363 = bitcast float %1362 to i32, !dbg !6
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1363), !dbg !6
  %1365 = bitcast i32 %1364 to float, !dbg !6
  %1366 = select i1 %.not40, float %1354, float %1365, !dbg !6
  %1367 = select i1 %.not40, float %1365, float %1360, !dbg !6
  %1368 = select i1 %.not40, float %1361, float %1355, !dbg !6
  %1369 = bitcast float %1368 to i32, !dbg !6
  %1370 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1369), !dbg !6
  %1371 = bitcast i32 %1370 to float, !dbg !6
  %1372 = select i1 %.not40, float %1355, float %1371, !dbg !6
  %1373 = select i1 %.not40, float %1371, float %1361, !dbg !6
  %1374 = extractvalue [2 x <8 x float>] %102, 0, !dbg !6
  %1375 = extractelement <8 x float> %1374, i64 0, !dbg !6
  %1376 = extractelement <8 x float> %1374, i64 1, !dbg !6
  %1377 = extractelement <8 x float> %1374, i64 2, !dbg !6
  %1378 = extractelement <8 x float> %1374, i64 3, !dbg !6
  %1379 = select i1 %.not, float %1376, float %1375, !dbg !6
  %1380 = bitcast float %1379 to i32, !dbg !6
  %1381 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1380), !dbg !6
  %1382 = bitcast i32 %1381 to float, !dbg !6
  %1383 = select i1 %.not, float %1375, float %1382, !dbg !6
  %1384 = select i1 %.not, float %1382, float %1376, !dbg !6
  %1385 = select i1 %.not, float %1378, float %1377, !dbg !6
  %1386 = bitcast float %1385 to i32, !dbg !6
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1386), !dbg !6
  %1388 = bitcast i32 %1387 to float, !dbg !6
  %1389 = select i1 %.not, float %1377, float %1388, !dbg !6
  %1390 = select i1 %.not, float %1388, float %1378, !dbg !6
  %1391 = select i1 %.not40, float %1389, float %1383, !dbg !6
  %1392 = bitcast float %1391 to i32, !dbg !6
  %1393 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1392), !dbg !6
  %1394 = select i1 %.not40, float %1390, float %1384, !dbg !6
  %1395 = bitcast float %1394 to i32, !dbg !6
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1395), !dbg !6
  %1397 = extractelement <8 x float> %1374, i64 4, !dbg !6
  %1398 = extractelement <8 x float> %1374, i64 5, !dbg !6
  %1399 = extractelement <8 x float> %1374, i64 6, !dbg !6
  %1400 = extractelement <8 x float> %1374, i64 7, !dbg !6
  %1401 = select i1 %.not, float %1398, float %1397, !dbg !6
  %1402 = bitcast float %1401 to i32, !dbg !6
  %1403 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1402), !dbg !6
  %1404 = bitcast i32 %1403 to float, !dbg !6
  %1405 = select i1 %.not, float %1397, float %1404, !dbg !6
  %1406 = select i1 %.not, float %1404, float %1398, !dbg !6
  %1407 = select i1 %.not, float %1400, float %1399, !dbg !6
  %1408 = bitcast float %1407 to i32, !dbg !6
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1408), !dbg !6
  %1410 = bitcast i32 %1409 to float, !dbg !6
  %1411 = select i1 %.not, float %1399, float %1410, !dbg !6
  %1412 = select i1 %.not, float %1410, float %1400, !dbg !6
  %1413 = select i1 %.not40, float %1411, float %1405, !dbg !6
  %1414 = bitcast float %1413 to i32, !dbg !6
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1414), !dbg !6
  %1416 = select i1 %.not40, float %1412, float %1406, !dbg !6
  %1417 = bitcast float %1416 to i32, !dbg !6
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1417), !dbg !6
  %1419 = extractvalue [2 x <8 x float>] %102, 1, !dbg !6
  %1420 = extractelement <8 x float> %1419, i64 0, !dbg !6
  %1421 = extractelement <8 x float> %1419, i64 1, !dbg !6
  %1422 = extractelement <8 x float> %1419, i64 2, !dbg !6
  %1423 = extractelement <8 x float> %1419, i64 3, !dbg !6
  %1424 = select i1 %.not, float %1421, float %1420, !dbg !6
  %1425 = bitcast float %1424 to i32, !dbg !6
  %1426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1425), !dbg !6
  %1427 = bitcast i32 %1426 to float, !dbg !6
  %1428 = select i1 %.not, float %1420, float %1427, !dbg !6
  %1429 = select i1 %.not, float %1427, float %1421, !dbg !6
  %1430 = select i1 %.not, float %1423, float %1422, !dbg !6
  %1431 = bitcast float %1430 to i32, !dbg !6
  %1432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1431), !dbg !6
  %1433 = bitcast i32 %1432 to float, !dbg !6
  %1434 = select i1 %.not, float %1422, float %1433, !dbg !6
  %1435 = select i1 %.not, float %1433, float %1423, !dbg !6
  %1436 = select i1 %.not40, float %1434, float %1428, !dbg !6
  %1437 = bitcast float %1436 to i32, !dbg !6
  %1438 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1437), !dbg !6
  %1439 = select i1 %.not40, float %1435, float %1429, !dbg !6
  %1440 = bitcast float %1439 to i32, !dbg !6
  %1441 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1440), !dbg !6
  %1442 = extractelement <8 x float> %1419, i64 4, !dbg !6
  %1443 = extractelement <8 x float> %1419, i64 5, !dbg !6
  %1444 = extractelement <8 x float> %1419, i64 6, !dbg !6
  %1445 = extractelement <8 x float> %1419, i64 7, !dbg !6
  %1446 = select i1 %.not, float %1443, float %1442, !dbg !6
  %1447 = bitcast float %1446 to i32, !dbg !6
  %1448 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1447), !dbg !6
  %1449 = bitcast i32 %1448 to float, !dbg !6
  %1450 = select i1 %.not, float %1442, float %1449, !dbg !6
  %1451 = select i1 %.not, float %1449, float %1443, !dbg !6
  %1452 = select i1 %.not, float %1445, float %1444, !dbg !6
  %1453 = bitcast float %1452 to i32, !dbg !6
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1453), !dbg !6
  %1455 = bitcast i32 %1454 to float, !dbg !6
  %1456 = select i1 %.not, float %1444, float %1455, !dbg !6
  %1457 = select i1 %.not, float %1455, float %1445, !dbg !6
  %1458 = select i1 %.not40, float %1456, float %1450, !dbg !6
  %1459 = bitcast float %1458 to i32, !dbg !6
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1459), !dbg !6
  %1461 = select i1 %.not40, float %1457, float %1451, !dbg !6
  %1462 = bitcast float %1461 to i32, !dbg !6
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1462), !dbg !6
  %1464 = shufflevector <1 x float> %1029, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1465 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1464, !dbg !17
  %1466 = shufflevector <1 x float> %1024, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1467 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1466, !dbg !17
  %1468 = bitcast i32 %1460 to float, !dbg !6
  %1469 = select i1 %.not40, float %1450, float %1468, !dbg !6
  %1470 = insertelement <4 x float> poison, float %1469, i64 0, !dbg !17
  %1471 = bitcast i32 %1463 to float, !dbg !6
  %1472 = select i1 %.not40, float %1451, float %1471, !dbg !6
  %1473 = insertelement <4 x float> %1470, float %1472, i64 1, !dbg !17
  %1474 = select i1 %.not40, float %1468, float %1456, !dbg !6
  %1475 = insertelement <4 x float> %1473, float %1474, i64 2, !dbg !17
  %1476 = select i1 %.not40, float %1471, float %1457, !dbg !6
  %1477 = insertelement <4 x float> %1475, float %1476, i64 3, !dbg !17
  %1478 = fmul <4 x float> %1477, %1467, !dbg !17
  %1479 = fptrunc <4 x float> %1478 to <4 x half>, !dbg !18
  %1480 = extractelement <4 x half> %1479, i64 3, !dbg !18
  %1481 = extractelement <4 x half> %1479, i64 2, !dbg !18
  %1482 = extractelement <4 x half> %1479, i64 1, !dbg !18
  %1483 = extractelement <4 x half> %1479, i64 0, !dbg !18
  %1484 = bitcast i32 %1438 to float, !dbg !6
  %1485 = select i1 %.not40, float %1428, float %1484, !dbg !6
  %1486 = insertelement <4 x float> poison, float %1485, i64 0, !dbg !17
  %1487 = bitcast i32 %1441 to float, !dbg !6
  %1488 = select i1 %.not40, float %1429, float %1487, !dbg !6
  %1489 = insertelement <4 x float> %1486, float %1488, i64 1, !dbg !17
  %1490 = select i1 %.not40, float %1484, float %1434, !dbg !6
  %1491 = insertelement <4 x float> %1489, float %1490, i64 2, !dbg !17
  %1492 = select i1 %.not40, float %1487, float %1435, !dbg !6
  %1493 = insertelement <4 x float> %1491, float %1492, i64 3, !dbg !17
  %1494 = fmul <4 x float> %1493, %1467, !dbg !17
  %1495 = fptrunc <4 x float> %1494 to <4 x half>, !dbg !18
  %1496 = extractelement <4 x half> %1495, i64 3, !dbg !18
  %1497 = extractelement <4 x half> %1495, i64 2, !dbg !18
  %1498 = extractelement <4 x half> %1495, i64 1, !dbg !18
  %1499 = extractelement <4 x half> %1495, i64 0, !dbg !18
  %1500 = bitcast i32 %1415 to float, !dbg !6
  %1501 = select i1 %.not40, float %1405, float %1500, !dbg !6
  %1502 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !17
  %1503 = bitcast i32 %1418 to float, !dbg !6
  %1504 = select i1 %.not40, float %1406, float %1503, !dbg !6
  %1505 = insertelement <4 x float> %1502, float %1504, i64 1, !dbg !17
  %1506 = select i1 %.not40, float %1500, float %1411, !dbg !6
  %1507 = insertelement <4 x float> %1505, float %1506, i64 2, !dbg !17
  %1508 = select i1 %.not40, float %1503, float %1412, !dbg !6
  %1509 = insertelement <4 x float> %1507, float %1508, i64 3, !dbg !17
  %1510 = fmul <4 x float> %1509, %1465, !dbg !17
  %1511 = fptrunc <4 x float> %1510 to <4 x half>, !dbg !18
  %1512 = extractelement <4 x half> %1511, i64 3, !dbg !18
  %1513 = extractelement <4 x half> %1511, i64 2, !dbg !18
  %1514 = extractelement <4 x half> %1511, i64 1, !dbg !18
  %1515 = extractelement <4 x half> %1511, i64 0, !dbg !18
  %1516 = bitcast i32 %1393 to float, !dbg !6
  %1517 = select i1 %.not40, float %1383, float %1516, !dbg !6
  %1518 = insertelement <4 x float> poison, float %1517, i64 0, !dbg !17
  %1519 = bitcast i32 %1396 to float, !dbg !6
  %1520 = select i1 %.not40, float %1384, float %1519, !dbg !6
  %1521 = insertelement <4 x float> %1518, float %1520, i64 1, !dbg !17
  %1522 = select i1 %.not40, float %1516, float %1389, !dbg !6
  %1523 = insertelement <4 x float> %1521, float %1522, i64 2, !dbg !17
  %1524 = select i1 %.not40, float %1519, float %1390, !dbg !6
  %1525 = insertelement <4 x float> %1523, float %1524, i64 3, !dbg !17
  %1526 = fmul <4 x float> %1525, %1465, !dbg !17
  %1527 = fptrunc <4 x float> %1526 to <4 x half>, !dbg !18
  %1528 = extractelement <4 x half> %1527, i64 3, !dbg !18
  %1529 = extractelement <4 x half> %1527, i64 2, !dbg !18
  %1530 = extractelement <4 x half> %1527, i64 1, !dbg !18
  %1531 = extractelement <4 x half> %1527, i64 0, !dbg !18
  %address.thread971 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local972 = zext nneg i32 %address.thread971 to i64, !dbg !9
  %address.local973 = add nuw i64 %32, %address.local972, !dbg !19
  %1532 = lshr i64 %address.local972, 2, !dbg !19
  %address.local975 = and i64 %1532, 48, !dbg !19
  %address.local976 = add i64 %address.local973, %address.local975, !dbg !19
  %address.local977 = and i64 %address.local972, 240, !dbg !19
  %address.local978 = sub i64 %address.local976, %address.local977, !dbg !19
  %address.local979 = add i64 %address.local978, %33, !dbg !19
  %address.local980 = shl nuw nsw i64 %address.local979, 7, !dbg !19
  %1533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1533, i64 %address.local980, !dbg !19
  %address.local984 = and i64 %1532, 12, !dbg !12
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local984, !dbg !19
  store half %1531, ptr addrspace(1) %address.local985, align 2, !dbg !19
  %address.local987 = or disjoint i64 %address.local984, 1, !dbg !19
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local987, !dbg !19
  store half %1530, ptr addrspace(1) %address.local988, align 2, !dbg !19
  %address.local990 = or disjoint i64 %address.local984, 2, !dbg !19
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local990, !dbg !19
  store half %1529, ptr addrspace(1) %address.local991, align 2, !dbg !19
  %address.local993 = or disjoint i64 %address.local984, 3, !dbg !19
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local993, !dbg !19
  store half %1528, ptr addrspace(1) %address.local994, align 2, !dbg !19
  %address.local996 = or disjoint i64 %address.local984, 16, !dbg !19
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local996, !dbg !19
  store half %1515, ptr addrspace(1) %address.local997, align 2, !dbg !19
  %address.local999 = or disjoint i64 %address.local984, 17, !dbg !19
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local999, !dbg !19
  store half %1514, ptr addrspace(1) %address.local1000, align 2, !dbg !19
  %address.local1002 = or disjoint i64 %address.local984, 18, !dbg !19
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1002, !dbg !19
  store half %1513, ptr addrspace(1) %address.local1003, align 2, !dbg !19
  %address.local1005 = or disjoint i64 %address.local984, 19, !dbg !19
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1005, !dbg !19
  store half %1512, ptr addrspace(1) %address.local1006, align 2, !dbg !19
  %address.local1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local982, i64 16384, !dbg !19
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local984, !dbg !19
  store half %1499, ptr addrspace(1) %address.local1011, align 2, !dbg !19
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local987, !dbg !19
  store half %1498, ptr addrspace(1) %address.local1013, align 2, !dbg !19
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local990, !dbg !19
  store half %1497, ptr addrspace(1) %address.local1015, align 2, !dbg !19
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local993, !dbg !19
  store half %1496, ptr addrspace(1) %address.local1017, align 2, !dbg !19
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local996, !dbg !19
  store half %1483, ptr addrspace(1) %address.local1019, align 2, !dbg !19
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local999, !dbg !19
  store half %1482, ptr addrspace(1) %address.local1021, align 2, !dbg !19
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1002, !dbg !19
  store half %1481, ptr addrspace(1) %address.local1023, align 2, !dbg !19
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1005, !dbg !19
  store half %1480, ptr addrspace(1) %address.local1025, align 2, !dbg !19
  %1534 = insertelement <4 x float> poison, float %1366, i64 0, !dbg !17
  %1535 = insertelement <4 x float> %1534, float %1372, i64 1, !dbg !17
  %1536 = insertelement <4 x float> %1535, float %1367, i64 2, !dbg !17
  %1537 = insertelement <4 x float> %1536, float %1373, i64 3, !dbg !17
  %1538 = fmul <4 x float> %1537, %1467, !dbg !17
  %1539 = fptrunc <4 x float> %1538 to <4 x half>, !dbg !18
  %1540 = extractelement <4 x half> %1539, i64 3, !dbg !18
  %1541 = extractelement <4 x half> %1539, i64 2, !dbg !18
  %1542 = extractelement <4 x half> %1539, i64 1, !dbg !18
  %1543 = extractelement <4 x half> %1539, i64 0, !dbg !18
  %1544 = insertelement <4 x float> poison, float %1338, i64 0, !dbg !17
  %1545 = insertelement <4 x float> %1544, float %1344, i64 1, !dbg !17
  %1546 = insertelement <4 x float> %1545, float %1339, i64 2, !dbg !17
  %1547 = insertelement <4 x float> %1546, float %1345, i64 3, !dbg !17
  %1548 = fmul <4 x float> %1547, %1467, !dbg !17
  %1549 = fptrunc <4 x float> %1548 to <4 x half>, !dbg !18
  %1550 = extractelement <4 x half> %1549, i64 3, !dbg !18
  %1551 = extractelement <4 x half> %1549, i64 2, !dbg !18
  %1552 = extractelement <4 x half> %1549, i64 1, !dbg !18
  %1553 = extractelement <4 x half> %1549, i64 0, !dbg !18
  %1554 = insertelement <4 x float> poison, float %1309, i64 0, !dbg !17
  %1555 = insertelement <4 x float> %1554, float %1315, i64 1, !dbg !17
  %1556 = insertelement <4 x float> %1555, float %1310, i64 2, !dbg !17
  %1557 = insertelement <4 x float> %1556, float %1316, i64 3, !dbg !17
  %1558 = fmul <4 x float> %1557, %1465, !dbg !17
  %1559 = fptrunc <4 x float> %1558 to <4 x half>, !dbg !18
  %1560 = extractelement <4 x half> %1559, i64 3, !dbg !18
  %1561 = extractelement <4 x half> %1559, i64 2, !dbg !18
  %1562 = extractelement <4 x half> %1559, i64 1, !dbg !18
  %1563 = extractelement <4 x half> %1559, i64 0, !dbg !18
  %1564 = insertelement <4 x float> poison, float %1281, i64 0, !dbg !17
  %1565 = insertelement <4 x float> %1564, float %1287, i64 1, !dbg !17
  %1566 = insertelement <4 x float> %1565, float %1282, i64 2, !dbg !17
  %1567 = insertelement <4 x float> %1566, float %1288, i64 3, !dbg !17
  %1568 = fmul <4 x float> %1567, %1465, !dbg !17
  %1569 = fptrunc <4 x float> %1568 to <4 x half>, !dbg !18
  %1570 = extractelement <4 x half> %1569, i64 3, !dbg !18
  %1571 = extractelement <4 x half> %1569, i64 2, !dbg !18
  %1572 = extractelement <4 x half> %1569, i64 1, !dbg !18
  %1573 = extractelement <4 x half> %1569, i64 0, !dbg !18
  %address.local1040 = or disjoint i64 %address.local984, 32, !dbg !19
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1040, !dbg !19
  store half %1573, ptr addrspace(1) %address.local1041, align 2, !dbg !19
  %address.local1043 = or disjoint i64 %address.local984, 33, !dbg !19
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1043, !dbg !19
  store half %1572, ptr addrspace(1) %address.local1044, align 2, !dbg !19
  %address.local1046 = or disjoint i64 %address.local984, 34, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1046, !dbg !19
  store half %1571, ptr addrspace(1) %address.local1047, align 2, !dbg !19
  %address.local1049 = or disjoint i64 %address.local984, 35, !dbg !19
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1049, !dbg !19
  store half %1570, ptr addrspace(1) %address.local1050, align 2, !dbg !19
  %address.local1052 = or disjoint i64 %address.local984, 48, !dbg !19
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1052, !dbg !19
  store half %1563, ptr addrspace(1) %address.local1053, align 2, !dbg !19
  %address.local1055 = or disjoint i64 %address.local984, 49, !dbg !19
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1055, !dbg !19
  store half %1562, ptr addrspace(1) %address.local1056, align 2, !dbg !19
  %address.local1058 = or disjoint i64 %address.local984, 50, !dbg !19
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1058, !dbg !19
  store half %1561, ptr addrspace(1) %address.local1059, align 2, !dbg !19
  %address.local1061 = or i64 %1532, 51, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1061, !dbg !19
  store half %1560, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1040, !dbg !19
  store half %1553, ptr addrspace(1) %address.local1067, align 2, !dbg !19
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1043, !dbg !19
  store half %1552, ptr addrspace(1) %address.local1069, align 2, !dbg !19
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1046, !dbg !19
  store half %1551, ptr addrspace(1) %address.local1071, align 2, !dbg !19
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1049, !dbg !19
  store half %1550, ptr addrspace(1) %address.local1073, align 2, !dbg !19
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1052, !dbg !19
  store half %1543, ptr addrspace(1) %address.local1075, align 2, !dbg !19
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1055, !dbg !19
  store half %1542, ptr addrspace(1) %address.local1077, align 2, !dbg !19
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1058, !dbg !19
  store half %1541, ptr addrspace(1) %address.local1079, align 2, !dbg !19
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1061, !dbg !19
  store half %1540, ptr addrspace(1) %address.local1081, align 2, !dbg !19
  %1574 = insertelement <4 x float> poison, float %1252, i64 0, !dbg !17
  %1575 = insertelement <4 x float> %1574, float %1258, i64 1, !dbg !17
  %1576 = insertelement <4 x float> %1575, float %1253, i64 2, !dbg !17
  %1577 = insertelement <4 x float> %1576, float %1259, i64 3, !dbg !17
  %1578 = fmul <4 x float> %1577, %1467, !dbg !17
  %1579 = fptrunc <4 x float> %1578 to <4 x half>, !dbg !18
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !18
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !18
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !18
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !18
  %1584 = insertelement <4 x float> poison, float %1224, i64 0, !dbg !17
  %1585 = insertelement <4 x float> %1584, float %1230, i64 1, !dbg !17
  %1586 = insertelement <4 x float> %1585, float %1225, i64 2, !dbg !17
  %1587 = insertelement <4 x float> %1586, float %1231, i64 3, !dbg !17
  %1588 = fmul <4 x float> %1587, %1467, !dbg !17
  %1589 = fptrunc <4 x float> %1588 to <4 x half>, !dbg !18
  %1590 = extractelement <4 x half> %1589, i64 3, !dbg !18
  %1591 = extractelement <4 x half> %1589, i64 2, !dbg !18
  %1592 = extractelement <4 x half> %1589, i64 1, !dbg !18
  %1593 = extractelement <4 x half> %1589, i64 0, !dbg !18
  %1594 = insertelement <4 x float> poison, float %1195, i64 0, !dbg !17
  %1595 = insertelement <4 x float> %1594, float %1201, i64 1, !dbg !17
  %1596 = insertelement <4 x float> %1595, float %1196, i64 2, !dbg !17
  %1597 = insertelement <4 x float> %1596, float %1202, i64 3, !dbg !17
  %1598 = fmul <4 x float> %1597, %1465, !dbg !17
  %1599 = fptrunc <4 x float> %1598 to <4 x half>, !dbg !18
  %1600 = extractelement <4 x half> %1599, i64 3, !dbg !18
  %1601 = extractelement <4 x half> %1599, i64 2, !dbg !18
  %1602 = extractelement <4 x half> %1599, i64 1, !dbg !18
  %1603 = extractelement <4 x half> %1599, i64 0, !dbg !18
  %1604 = insertelement <4 x float> poison, float %1167, i64 0, !dbg !17
  %1605 = insertelement <4 x float> %1604, float %1173, i64 1, !dbg !17
  %1606 = insertelement <4 x float> %1605, float %1168, i64 2, !dbg !17
  %1607 = insertelement <4 x float> %1606, float %1174, i64 3, !dbg !17
  %1608 = fmul <4 x float> %1607, %1465, !dbg !17
  %1609 = fptrunc <4 x float> %1608 to <4 x half>, !dbg !18
  %1610 = extractelement <4 x half> %1609, i64 3, !dbg !18
  %1611 = extractelement <4 x half> %1609, i64 2, !dbg !18
  %1612 = extractelement <4 x half> %1609, i64 1, !dbg !18
  %1613 = extractelement <4 x half> %1609, i64 0, !dbg !18
  %address.local1096 = or disjoint i64 %address.local984, 64, !dbg !19
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1096, !dbg !19
  store half %1613, ptr addrspace(1) %address.local1097, align 2, !dbg !19
  %address.local1099 = or disjoint i64 %address.local984, 65, !dbg !19
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1099, !dbg !19
  store half %1612, ptr addrspace(1) %address.local1100, align 2, !dbg !19
  %address.local1102 = or disjoint i64 %address.local984, 66, !dbg !19
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1102, !dbg !19
  store half %1611, ptr addrspace(1) %address.local1103, align 2, !dbg !19
  %address.local1105 = or disjoint i64 %address.local984, 67, !dbg !19
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1105, !dbg !19
  store half %1610, ptr addrspace(1) %address.local1106, align 2, !dbg !19
  %address.local1108 = or disjoint i64 %address.local984, 80, !dbg !19
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1108, !dbg !19
  store half %1603, ptr addrspace(1) %address.local1109, align 2, !dbg !19
  %address.local1111 = or disjoint i64 %address.local984, 81, !dbg !19
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1111, !dbg !19
  store half %1602, ptr addrspace(1) %address.local1112, align 2, !dbg !19
  %address.local1114 = or disjoint i64 %address.local984, 82, !dbg !19
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1114, !dbg !19
  store half %1601, ptr addrspace(1) %address.local1115, align 2, !dbg !19
  %address.local1117 = or disjoint i64 %address.local984, 83, !dbg !19
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1117, !dbg !19
  store half %1600, ptr addrspace(1) %address.local1118, align 2, !dbg !19
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1096, !dbg !19
  store half %1593, ptr addrspace(1) %address.local1123, align 2, !dbg !19
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1099, !dbg !19
  store half %1592, ptr addrspace(1) %address.local1125, align 2, !dbg !19
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1102, !dbg !19
  store half %1591, ptr addrspace(1) %address.local1127, align 2, !dbg !19
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1105, !dbg !19
  store half %1590, ptr addrspace(1) %address.local1129, align 2, !dbg !19
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1108, !dbg !19
  store half %1583, ptr addrspace(1) %address.local1131, align 2, !dbg !19
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1111, !dbg !19
  store half %1582, ptr addrspace(1) %address.local1133, align 2, !dbg !19
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1114, !dbg !19
  store half %1581, ptr addrspace(1) %address.local1135, align 2, !dbg !19
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1117, !dbg !19
  store half %1580, ptr addrspace(1) %address.local1137, align 2, !dbg !19
  %1614 = insertelement <4 x float> poison, float %1138, i64 0, !dbg !17
  %1615 = insertelement <4 x float> %1614, float %1144, i64 1, !dbg !17
  %1616 = insertelement <4 x float> %1615, float %1139, i64 2, !dbg !17
  %1617 = insertelement <4 x float> %1616, float %1145, i64 3, !dbg !17
  %1618 = fmul <4 x float> %1617, %1467, !dbg !17
  %1619 = fptrunc <4 x float> %1618 to <4 x half>, !dbg !18
  %1620 = extractelement <4 x half> %1619, i64 3, !dbg !18
  %1621 = extractelement <4 x half> %1619, i64 2, !dbg !18
  %1622 = extractelement <4 x half> %1619, i64 1, !dbg !18
  %1623 = extractelement <4 x half> %1619, i64 0, !dbg !18
  %1624 = insertelement <4 x float> poison, float %1110, i64 0, !dbg !17
  %1625 = insertelement <4 x float> %1624, float %1116, i64 1, !dbg !17
  %1626 = insertelement <4 x float> %1625, float %1111, i64 2, !dbg !17
  %1627 = insertelement <4 x float> %1626, float %1117, i64 3, !dbg !17
  %1628 = fmul <4 x float> %1627, %1467, !dbg !17
  %1629 = fptrunc <4 x float> %1628 to <4 x half>, !dbg !18
  %1630 = extractelement <4 x half> %1629, i64 3, !dbg !18
  %1631 = extractelement <4 x half> %1629, i64 2, !dbg !18
  %1632 = extractelement <4 x half> %1629, i64 1, !dbg !18
  %1633 = extractelement <4 x half> %1629, i64 0, !dbg !18
  %1634 = insertelement <4 x float> poison, float %1081, i64 0, !dbg !17
  %1635 = insertelement <4 x float> %1634, float %1087, i64 1, !dbg !17
  %1636 = insertelement <4 x float> %1635, float %1082, i64 2, !dbg !17
  %1637 = insertelement <4 x float> %1636, float %1088, i64 3, !dbg !17
  %1638 = fmul <4 x float> %1637, %1465, !dbg !17
  %1639 = fptrunc <4 x float> %1638 to <4 x half>, !dbg !18
  %1640 = extractelement <4 x half> %1639, i64 3, !dbg !18
  %1641 = extractelement <4 x half> %1639, i64 2, !dbg !18
  %1642 = extractelement <4 x half> %1639, i64 1, !dbg !18
  %1643 = extractelement <4 x half> %1639, i64 0, !dbg !18
  %1644 = insertelement <4 x float> poison, float %1053, i64 0, !dbg !17
  %1645 = insertelement <4 x float> %1644, float %1059, i64 1, !dbg !17
  %1646 = insertelement <4 x float> %1645, float %1054, i64 2, !dbg !17
  %1647 = insertelement <4 x float> %1646, float %1060, i64 3, !dbg !17
  %1648 = fmul <4 x float> %1647, %1465, !dbg !17
  %1649 = fptrunc <4 x float> %1648 to <4 x half>, !dbg !18
  %1650 = extractelement <4 x half> %1649, i64 3, !dbg !18
  %1651 = extractelement <4 x half> %1649, i64 2, !dbg !18
  %1652 = extractelement <4 x half> %1649, i64 1, !dbg !18
  %1653 = extractelement <4 x half> %1649, i64 0, !dbg !18
  %address.local1152 = or disjoint i64 %address.local984, 96, !dbg !19
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1152, !dbg !19
  store half %1653, ptr addrspace(1) %address.local1153, align 2, !dbg !19
  %address.local1155 = or disjoint i64 %address.local984, 97, !dbg !19
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1155, !dbg !19
  store half %1652, ptr addrspace(1) %address.local1156, align 2, !dbg !19
  %address.local1158 = or disjoint i64 %address.local984, 98, !dbg !19
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1158, !dbg !19
  store half %1651, ptr addrspace(1) %address.local1159, align 2, !dbg !19
  %address.local1161 = or disjoint i64 %address.local984, 99, !dbg !19
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1161, !dbg !19
  store half %1650, ptr addrspace(1) %address.local1162, align 2, !dbg !19
  %address.local1164 = or disjoint i64 %address.local984, 112, !dbg !19
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1164, !dbg !19
  store half %1643, ptr addrspace(1) %address.local1165, align 2, !dbg !19
  %address.local1167 = or disjoint i64 %address.local984, 113, !dbg !19
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1167, !dbg !19
  store half %1642, ptr addrspace(1) %address.local1168, align 2, !dbg !19
  %address.local1170 = or disjoint i64 %address.local984, 114, !dbg !19
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1170, !dbg !19
  store half %1641, ptr addrspace(1) %address.local1171, align 2, !dbg !19
  %address.local1173 = or i64 %1532, 115, !dbg !19
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1173, !dbg !19
  store half %1640, ptr addrspace(1) %address.local1174, align 2, !dbg !19
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1152, !dbg !19
  store half %1633, ptr addrspace(1) %address.local1179, align 2, !dbg !19
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1155, !dbg !19
  store half %1632, ptr addrspace(1) %address.local1181, align 2, !dbg !19
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1158, !dbg !19
  store half %1631, ptr addrspace(1) %address.local1183, align 2, !dbg !19
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1161, !dbg !19
  store half %1630, ptr addrspace(1) %address.local1185, align 2, !dbg !19
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1164, !dbg !19
  store half %1623, ptr addrspace(1) %address.local1187, align 2, !dbg !19
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1167, !dbg !19
  store half %1622, ptr addrspace(1) %address.local1189, align 2, !dbg !19
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1170, !dbg !19
  store half %1621, ptr addrspace(1) %address.local1191, align 2, !dbg !19
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1173, !dbg !19
  store half %1620, ptr addrspace(1) %address.local1193, align 2, !dbg !19
  ret void, !dbg !5
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #5

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #8

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nounwind willreturn }
attributes #6 = { convergent nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!29}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/fragment_level_optimized_noncausal")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 74, column: 20, scope: !4)
!9 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 69, column: 24, scope: !4)
!13 = !DILocation(line: 68, column: 22, scope: !4)
!14 = !DILocation(line: 89, column: 20, scope: !4)
!15 = !DILocation(line: 71, column: 23, scope: !4)
!16 = !DILocation(line: 86, column: 20, scope: !4)
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
!31 = !{i32 4}
!32 = !{i64 8192}
!33 = !{i32 0}
!34 = !{i32 1}
!35 = !{i32 2}
!36 = !{i32 3}
!37 = !{i32 5}
!38 = !{i32 6}
!39 = !{i32 7}
!40 = !{i32 8}
!41 = !{i32 9}
!42 = !{i32 10}
!43 = !{i32 11}
!44 = !{i32 12}
!45 = !{i32 13}
!46 = !{i32 14}
!47 = !{i32 15}
!48 = !{i32 16}
!49 = !{i32 17}
!50 = !{i32 18}
!51 = !{i32 19}
!52 = !{i32 20}
!53 = !{i32 21}
!54 = !{i32 22}
