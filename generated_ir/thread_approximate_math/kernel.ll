; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir"

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
  %101 = phi i64 [ %881, %k0.join ], [ 0, %4 ], !dbg !6
  %102 = phi [2 x <8 x float>] [ %949, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %103 = phi [2 x <1 x float>] [ %726, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %104 = phi [2 x <1 x float>] [ %lazy.base.join, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %105 = phi [2 x <8 x float>] [ %953, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %106 = phi [2 x <8 x float>] [ %957, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %107 = phi [2 x <8 x float>] [ %961, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %108 = icmp samesign ult i64 %101, 32, !dbg !6
  br i1 %108, label %109, label %962, !dbg !6

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
  %lazy.base = extractvalue [2 x <1 x float>] %104, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base2055 = extractvalue [2 x <1 x float>] %104, 1, !dbg !15
  %lazy.seed2056 = shufflevector <1 x float> %lazy.base2055, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> %lazy.seed) #9, !dbg !15
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
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> %lazy.seed) #9, !dbg !15
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
  %167 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %154, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %168 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %156, <4 x float> %167) #9, !dbg !15
  %169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %151, <4 x float> %lazy.seed) #9, !dbg !15
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %153, <4 x float> %169) #9, !dbg !15
  %171 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %155, <4 x float> %lazy.seed) #9, !dbg !15
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %157, <4 x float> %171) #9, !dbg !15
  %173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %159, <4 x float> %lazy.seed) #9, !dbg !15
  %174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %162, <4 x float> %173) #9, !dbg !15
  %175 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %165, <4 x float> %lazy.seed) #9, !dbg !15
  %176 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %166, <4 x float> %175) #9, !dbg !15
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %160, <4 x float> %lazy.seed) #9, !dbg !15
  %178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %163, <4 x float> %177) #9, !dbg !15
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> %lazy.seed2056) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> %lazy.seed2056) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> %lazy.seed2056) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> %lazy.seed2056) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> %lazy.seed2056) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> %lazy.seed2056) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> %lazy.seed2056) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> %lazy.seed2056) #9, !dbg !15
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
  %lazy.peak = call float @llvm.maxnum.f32(float %428, float %438), !dbg !23
  %lazy.over = fcmp ogt float %lazy.peak, 0x4020000000000000, !dbg !23
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !23
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !23
  %lazy.first = icmp eq i64 %101, 0, !dbg !23
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !23
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !23

lazy.slow:                                        ; preds = %109
  %lazy.positive = call float @llvm.maxnum.f32(float %428, float 0x0000000000000000), !dbg !23
  %lazy.delta = select i1 %lazy.first, float %428, float %lazy.positive, !dbg !23
  %lazy.negated = fsub float 0x0000000000000000, %lazy.delta, !dbg !23
  %lazy.exp = call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !23
  %lazy.alpha = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp, !dbg !23
  %lazy.positive2057 = call float @llvm.maxnum.f32(float %438, float 0x0000000000000000), !dbg !23
  %lazy.delta2058 = select i1 %lazy.first, float %438, float %lazy.positive2057, !dbg !23
  %lazy.negated2059 = fsub float 0x0000000000000000, %lazy.delta2058, !dbg !23
  %lazy.exp2060 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated2059), !dbg !23
  %lazy.alpha2061 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp2060, !dbg !23
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted = fsub <4 x float> %316, %lazy.delta.v, !dbg !23
  %lazy.delta.v.lane2062 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2063 = shufflevector <1 x float> %lazy.delta.v.lane2062, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2064 = fsub <4 x float> %323, %lazy.delta.v2063, !dbg !23
  %lazy.delta.v.lane2065 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2066 = shufflevector <1 x float> %lazy.delta.v.lane2065, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2067 = fsub <4 x float> %333, %lazy.delta.v2066, !dbg !23
  %lazy.delta.v.lane2068 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2069 = shufflevector <1 x float> %lazy.delta.v.lane2068, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2070 = fsub <4 x float> %343, %lazy.delta.v2069, !dbg !23
  %lazy.delta.v.lane2071 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2072 = shufflevector <1 x float> %lazy.delta.v.lane2071, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2073 = fsub <4 x float> %349, %lazy.delta.v2072, !dbg !23
  %lazy.delta.v.lane2074 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2075 = shufflevector <1 x float> %lazy.delta.v.lane2074, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2076 = fsub <4 x float> %355, %lazy.delta.v2075, !dbg !23
  %lazy.delta.v.lane2077 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2078 = shufflevector <1 x float> %lazy.delta.v.lane2077, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2079 = fsub <4 x float> %361, %lazy.delta.v2078, !dbg !23
  %lazy.delta.v.lane2080 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2081 = shufflevector <1 x float> %lazy.delta.v.lane2080, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2082 = fsub <4 x float> %367, %lazy.delta.v2081, !dbg !23
  %lazy.delta.v.lane2083 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2084 = shufflevector <1 x float> %lazy.delta.v.lane2083, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2085 = fsub <4 x float> %374, %lazy.delta.v2084, !dbg !23
  %lazy.delta.v.lane2086 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2087 = shufflevector <1 x float> %lazy.delta.v.lane2086, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2088 = fsub <4 x float> %380, %lazy.delta.v2087, !dbg !23
  %lazy.delta.v.lane2089 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2090 = shufflevector <1 x float> %lazy.delta.v.lane2089, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2091 = fsub <4 x float> %386, %lazy.delta.v2090, !dbg !23
  %lazy.delta.v.lane2092 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2093 = shufflevector <1 x float> %lazy.delta.v.lane2092, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2094 = fsub <4 x float> %392, %lazy.delta.v2093, !dbg !23
  %lazy.delta.v.lane2095 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2096 = shufflevector <1 x float> %lazy.delta.v.lane2095, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2097 = fsub <4 x float> %398, %lazy.delta.v2096, !dbg !23
  %lazy.delta.v.lane2098 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2099 = shufflevector <1 x float> %lazy.delta.v.lane2098, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2100 = fsub <4 x float> %404, %lazy.delta.v2099, !dbg !23
  %lazy.delta.v.lane2101 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2102 = shufflevector <1 x float> %lazy.delta.v.lane2101, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2103 = fsub <4 x float> %410, %lazy.delta.v2102, !dbg !23
  %lazy.delta.v.lane2104 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.delta.v2105 = shufflevector <1 x float> %lazy.delta.v.lane2104, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2106 = fsub <4 x float> %416, %lazy.delta.v2105, !dbg !23
  %lazy.state = extractvalue [2 x <1 x float>] %103, 0, !dbg !23
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !23
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !23
  %lazy.state2107 = extractvalue [2 x <1 x float>] %103, 1, !dbg !23
  %lazy.alpha.v.lane2108 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !23
  %lazy.scaled2109 = fmul <1 x float> %lazy.state2107, %lazy.alpha.v.lane2108, !dbg !23
  %lazy.state.next2110 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2109, 1, !dbg !23
  %lazy.state2111 = extractvalue [2 x <8 x float>] %107, 0, !dbg !23
  %lazy.alpha.v.lane2112 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane2112, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2113 = fmul <8 x float> %lazy.state2111, %lazy.alpha.v, !dbg !23
  %lazy.state.next2114 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2113, 0, !dbg !23
  %lazy.state2115 = extractvalue [2 x <8 x float>] %107, 1, !dbg !23
  %lazy.alpha.v.lane2116 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !23
  %lazy.alpha.v2117 = shufflevector <1 x float> %lazy.alpha.v.lane2116, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2118 = fmul <8 x float> %lazy.state2115, %lazy.alpha.v2117, !dbg !23
  %lazy.state.next2119 = insertvalue [2 x <8 x float>] %lazy.state.next2114, <8 x float> %lazy.scaled2118, 1, !dbg !23
  %lazy.state2120 = extractvalue [2 x <8 x float>] %106, 0, !dbg !23
  %lazy.alpha.v.lane2121 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2122 = shufflevector <1 x float> %lazy.alpha.v.lane2121, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2123 = fmul <8 x float> %lazy.state2120, %lazy.alpha.v2122, !dbg !23
  %lazy.state.next2124 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2123, 0, !dbg !23
  %lazy.state2125 = extractvalue [2 x <8 x float>] %106, 1, !dbg !23
  %lazy.alpha.v.lane2126 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !23
  %lazy.alpha.v2127 = shufflevector <1 x float> %lazy.alpha.v.lane2126, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2128 = fmul <8 x float> %lazy.state2125, %lazy.alpha.v2127, !dbg !23
  %lazy.state.next2129 = insertvalue [2 x <8 x float>] %lazy.state.next2124, <8 x float> %lazy.scaled2128, 1, !dbg !23
  %lazy.state2130 = extractvalue [2 x <8 x float>] %105, 0, !dbg !23
  %lazy.alpha.v.lane2131 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2132 = shufflevector <1 x float> %lazy.alpha.v.lane2131, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2133 = fmul <8 x float> %lazy.state2130, %lazy.alpha.v2132, !dbg !23
  %lazy.state.next2134 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2133, 0, !dbg !23
  %lazy.state2135 = extractvalue [2 x <8 x float>] %105, 1, !dbg !23
  %lazy.alpha.v.lane2136 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !23
  %lazy.alpha.v2137 = shufflevector <1 x float> %lazy.alpha.v.lane2136, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2138 = fmul <8 x float> %lazy.state2135, %lazy.alpha.v2137, !dbg !23
  %lazy.state.next2139 = insertvalue [2 x <8 x float>] %lazy.state.next2134, <8 x float> %lazy.scaled2138, 1, !dbg !23
  %lazy.state2140 = extractvalue [2 x <8 x float>] %102, 0, !dbg !23
  %lazy.alpha.v.lane2141 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2142 = shufflevector <1 x float> %lazy.alpha.v.lane2141, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2143 = fmul <8 x float> %lazy.state2140, %lazy.alpha.v2142, !dbg !23
  %lazy.state.next2144 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2143, 0, !dbg !23
  %lazy.state2145 = extractvalue [2 x <8 x float>] %102, 1, !dbg !23
  %lazy.alpha.v.lane2146 = insertelement <1 x float> poison, float %lazy.alpha2061, i64 0, !dbg !23
  %lazy.alpha.v2147 = shufflevector <1 x float> %lazy.alpha.v.lane2146, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2148 = fmul <8 x float> %lazy.state2145, %lazy.alpha.v2147, !dbg !23
  %lazy.state.next2149 = insertvalue [2 x <8 x float>] %lazy.state.next2144, <8 x float> %lazy.scaled2148, 1, !dbg !23
  %lazy.base.old = extractvalue [2 x <1 x float>] %104, 0, !dbg !23
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !23
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !23
  %lazy.base.old2150 = extractvalue [2 x <1 x float>] %104, 1, !dbg !23
  %lazy.delta.b.lane2151 = insertelement <1 x float> poison, float %lazy.delta2058, i64 0, !dbg !23
  %lazy.base.next2152 = fsub <1 x float> %lazy.base.old2150, %lazy.delta.b.lane2151, !dbg !23
  %lazy.base.state2153 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2152, 1, !dbg !23
  br label %lazy.join, !dbg !23

lazy.join:                                        ; preds = %lazy.slow, %109
  %lazy.score = phi <4 x float> [ %316, %109 ], [ %lazy.shifted, %lazy.slow ], !dbg !23
  %lazy.score2154 = phi <4 x float> [ %323, %109 ], [ %lazy.shifted2064, %lazy.slow ], !dbg !23
  %lazy.score2155 = phi <4 x float> [ %333, %109 ], [ %lazy.shifted2067, %lazy.slow ], !dbg !23
  %lazy.score2156 = phi <4 x float> [ %343, %109 ], [ %lazy.shifted2070, %lazy.slow ], !dbg !23
  %lazy.score2157 = phi <4 x float> [ %349, %109 ], [ %lazy.shifted2073, %lazy.slow ], !dbg !23
  %lazy.score2158 = phi <4 x float> [ %355, %109 ], [ %lazy.shifted2076, %lazy.slow ], !dbg !23
  %lazy.score2159 = phi <4 x float> [ %361, %109 ], [ %lazy.shifted2079, %lazy.slow ], !dbg !23
  %lazy.score2160 = phi <4 x float> [ %367, %109 ], [ %lazy.shifted2082, %lazy.slow ], !dbg !23
  %lazy.score2161 = phi <4 x float> [ %374, %109 ], [ %lazy.shifted2085, %lazy.slow ], !dbg !23
  %lazy.score2162 = phi <4 x float> [ %380, %109 ], [ %lazy.shifted2088, %lazy.slow ], !dbg !23
  %lazy.score2163 = phi <4 x float> [ %386, %109 ], [ %lazy.shifted2091, %lazy.slow ], !dbg !23
  %lazy.score2164 = phi <4 x float> [ %392, %109 ], [ %lazy.shifted2094, %lazy.slow ], !dbg !23
  %lazy.score2165 = phi <4 x float> [ %398, %109 ], [ %lazy.shifted2097, %lazy.slow ], !dbg !23
  %lazy.score2166 = phi <4 x float> [ %404, %109 ], [ %lazy.shifted2100, %lazy.slow ], !dbg !23
  %lazy.score2167 = phi <4 x float> [ %410, %109 ], [ %lazy.shifted2103, %lazy.slow ], !dbg !23
  %lazy.score2168 = phi <4 x float> [ %416, %109 ], [ %lazy.shifted2106, %lazy.slow ], !dbg !23
  %lazy.state2169 = phi [2 x <1 x float>] [ %103, %109 ], [ %lazy.state.next2110, %lazy.slow ], !dbg !23
  %lazy.state2170 = phi [2 x <8 x float>] [ %107, %109 ], [ %lazy.state.next2119, %lazy.slow ], !dbg !23
  %lazy.state2171 = phi [2 x <8 x float>] [ %106, %109 ], [ %lazy.state.next2129, %lazy.slow ], !dbg !23
  %lazy.state2172 = phi [2 x <8 x float>] [ %105, %109 ], [ %lazy.state.next2139, %lazy.slow ], !dbg !23
  %lazy.state2173 = phi [2 x <8 x float>] [ %102, %109 ], [ %lazy.state.next2149, %lazy.slow ], !dbg !23
  %lazy.base.join = phi [2 x <1 x float>] [ %104, %109 ], [ %lazy.base.state2153, %lazy.slow ], !dbg !23
  %439 = extractvalue [2 x <1 x float>] %lazy.state2169, 0, !dbg !23
  %440 = extractelement <1 x float> %439, i64 0, !dbg !23
  %441 = extractvalue [2 x <1 x float>] %lazy.state2169, 1, !dbg !23
  %442 = extractelement <1 x float> %441, i64 0, !dbg !23
  %443 = extractvalue [2 x <8 x float>] %lazy.state2173, 0, !dbg !27
  %444 = shufflevector <8 x float> %443, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %445 = shufflevector <8 x float> %443, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %446 = extractvalue [2 x <8 x float>] %lazy.state2173, 1, !dbg !27
  %447 = shufflevector <8 x float> %446, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %448 = shufflevector <8 x float> %446, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
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
  %449 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1421, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %vdb.dma.base1433 = add i32 %46, 8192, !dbg !14
  %450 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1428, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %address.local927 = and i32 %address.local918, 120, !dbg !14
  %address.local929 = and i32 %address.local569, 1536, !dbg !14
  %address.local930 = or disjoint i32 %address.local927, %address.local929, !dbg !14
  %address.local931 = zext nneg i32 %address.local930 to i64, !dbg !14
  %address.local932 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local931, !dbg !14
  %451 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %452 = bitcast <4 x i32> %451 to <8 x half>, !dbg !14
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
  %453 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %454 = bitcast <4 x i32> %453 to <8 x half>, !dbg !14
  %address.local946 = getelementptr i8, ptr addrspace(3) %address.local941, i64 512, !dbg !14
  %455 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %456 = bitcast <4 x i32> %455 to <8 x half>, !dbg !14
  %address.local950 = getelementptr i8, ptr addrspace(3) %address.local941, i64 768, !dbg !14
  %457 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %458 = bitcast <4 x i32> %457 to <8 x half>, !dbg !14
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local932, i64 4096, !dbg !14
  %459 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %460 = bitcast <4 x i32> %459 to <8 x half>, !dbg !14
  %address.local956 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4352, !dbg !14
  %461 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %462 = bitcast <4 x i32> %461 to <8 x half>, !dbg !14
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4608, !dbg !14
  %463 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %464 = bitcast <4 x i32> %463 to <8 x half>, !dbg !14
  %address.local964 = getelementptr i8, ptr addrspace(3) %address.local941, i64 4864, !dbg !14
  %465 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %466 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %467 = extractelement <4 x float> %lazy.score, i64 0, !dbg !24
  %468 = call float @llvm.amdgcn.exp2.f32(float %467), !dbg !24
  %469 = insertelement <4 x float> poison, float %468, i64 0, !dbg !26
  %470 = extractelement <4 x float> %lazy.score, i64 1, !dbg !24
  %471 = call float @llvm.amdgcn.exp2.f32(float %470), !dbg !24
  %472 = insertelement <4 x float> %469, float %471, i64 1, !dbg !26
  %473 = extractelement <4 x float> %lazy.score, i64 2, !dbg !24
  %474 = call float @llvm.amdgcn.exp2.f32(float %473), !dbg !24
  %475 = insertelement <4 x float> %472, float %474, i64 2, !dbg !26
  %476 = extractelement <4 x float> %lazy.score, i64 3, !dbg !24
  %477 = call float @llvm.amdgcn.exp2.f32(float %476), !dbg !24
  %478 = insertelement <4 x float> %475, float %477, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %478) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !26
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !26
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !26
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !26
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !26
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !26
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !26
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !26
  %479 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !26
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %479) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %466, <4 x float> %444) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %481 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %482 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %481, <4 x float> %445) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %483 = extractelement <4 x float> %lazy.score2161, i64 0, !dbg !24
  %484 = call float @llvm.amdgcn.exp2.f32(float %483), !dbg !24
  %485 = insertelement <4 x float> poison, float %484, i64 0, !dbg !26
  %486 = extractelement <4 x float> %lazy.score2161, i64 1, !dbg !24
  %487 = call float @llvm.amdgcn.exp2.f32(float %486), !dbg !24
  %488 = insertelement <4 x float> %485, float %487, i64 1, !dbg !26
  %489 = extractelement <4 x float> %lazy.score2161, i64 2, !dbg !24
  %490 = call float @llvm.amdgcn.exp2.f32(float %489), !dbg !24
  %491 = insertelement <4 x float> %488, float %490, i64 2, !dbg !26
  %492 = extractelement <4 x float> %lazy.score2161, i64 3, !dbg !24
  %493 = call float @llvm.amdgcn.exp2.f32(float %492), !dbg !24
  %494 = insertelement <4 x float> %491, float %493, i64 3, !dbg !26
  %math.result382 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %494) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02174 = extractelement <4 x float> %math.result382, i32 0, !dbg !26
  %.cvtpack.f12175 = extractelement <4 x float> %math.result382, i32 1, !dbg !26
  %.cvtpack.f22176 = extractelement <4 x float> %math.result382, i32 2, !dbg !26
  %.cvtpack.f32177 = extractelement <4 x float> %math.result382, i32 3, !dbg !26
  %.cvtpack.p02178 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02174, float %.cvtpack.f12175), !dbg !26
  %.cvtpack.p12179 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22176, float %.cvtpack.f32177), !dbg !26
  %.cvtpack.v02180 = insertelement <2 x i32> poison, i32 %.cvtpack.p02178, i32 0, !dbg !26
  %.cvtpack.v12181 = insertelement <2 x i32> %.cvtpack.v02180, i32 %.cvtpack.p12179, i32 1, !dbg !26
  %495 = bitcast <2 x i32> %.cvtpack.v12181 to <4 x half>, !dbg !26
  %math.packed383 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %495) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %466, <4 x float> %447) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %481, <4 x float> %448) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %498 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %499 = extractelement <4 x float> %lazy.score2154, i64 0, !dbg !24
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !24
  %local.reduce71 = fadd float %468, %500, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !26
  %502 = extractelement <4 x float> %lazy.score2154, i64 1, !dbg !24
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !24
  %local.reduce72 = fadd float %471, %503, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1977 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %504 = insertelement <4 x float> %501, float %503, i64 1, !dbg !26
  %505 = extractelement <4 x float> %lazy.score2154, i64 2, !dbg !24
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !24
  %local.reduce73 = fadd float %474, %506, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1978 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %507 = insertelement <4 x float> %504, float %506, i64 2, !dbg !26
  %508 = extractelement <4 x float> %lazy.score2154, i64 3, !dbg !24
  %509 = call float @llvm.amdgcn.exp2.f32(float %508), !dbg !24
  %local.reduce74 = fadd float %477, %509, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1979 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %510 = insertelement <4 x float> %507, float %509, i64 3, !dbg !26
  %math.result368 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %510) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02182 = extractelement <4 x float> %math.result368, i32 0, !dbg !26
  %.cvtpack.f12183 = extractelement <4 x float> %math.result368, i32 1, !dbg !26
  %.cvtpack.f22184 = extractelement <4 x float> %math.result368, i32 2, !dbg !26
  %.cvtpack.f32185 = extractelement <4 x float> %math.result368, i32 3, !dbg !26
  %.cvtpack.p02186 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02182, float %.cvtpack.f12183), !dbg !26
  %.cvtpack.p12187 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22184, float %.cvtpack.f32185), !dbg !26
  %.cvtpack.v02188 = insertelement <2 x i32> poison, i32 %.cvtpack.p02186, i32 0, !dbg !26
  %.cvtpack.v12189 = insertelement <2 x i32> %.cvtpack.v02188, i32 %.cvtpack.p12187, i32 1, !dbg !26
  %511 = bitcast <2 x i32> %.cvtpack.v12189 to <4 x half>, !dbg !26
  %math.packed369 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %511) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %498, <4 x float> %480) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %513 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %513, <4 x float> %482) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %515 = extractelement <4 x float> %lazy.score2162, i64 0, !dbg !24
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !24
  %local.reduce = fadd float %484, %516, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1980 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %517 = insertelement <4 x float> poison, float %516, i64 0, !dbg !26
  %518 = extractelement <4 x float> %lazy.score2162, i64 1, !dbg !24
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !24
  %local.reduce41 = fadd float %487, %519, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1981 = call float asm sideeffect "", "=v,0"(float %local.reduce41) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %520 = insertelement <4 x float> %517, float %519, i64 1, !dbg !26
  %521 = extractelement <4 x float> %lazy.score2162, i64 2, !dbg !24
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !24
  %local.reduce42 = fadd float %490, %522, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1982 = call float asm sideeffect "", "=v,0"(float %local.reduce42) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %523 = insertelement <4 x float> %520, float %522, i64 2, !dbg !26
  %524 = extractelement <4 x float> %lazy.score2162, i64 3, !dbg !24
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !24
  %local.reduce43 = fadd float %493, %525, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1983 = call float asm sideeffect "", "=v,0"(float %local.reduce43) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %526 = insertelement <4 x float> %523, float %525, i64 3, !dbg !26
  %math.result384 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %526) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02190 = extractelement <4 x float> %math.result384, i32 0, !dbg !26
  %.cvtpack.f12191 = extractelement <4 x float> %math.result384, i32 1, !dbg !26
  %.cvtpack.f22192 = extractelement <4 x float> %math.result384, i32 2, !dbg !26
  %.cvtpack.f32193 = extractelement <4 x float> %math.result384, i32 3, !dbg !26
  %.cvtpack.p02194 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02190, float %.cvtpack.f12191), !dbg !26
  %.cvtpack.p12195 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22192, float %.cvtpack.f32193), !dbg !26
  %.cvtpack.v02196 = insertelement <2 x i32> poison, i32 %.cvtpack.p02194, i32 0, !dbg !26
  %.cvtpack.v12197 = insertelement <2 x i32> %.cvtpack.v02196, i32 %.cvtpack.p12195, i32 1, !dbg !26
  %527 = bitcast <2 x i32> %.cvtpack.v12197 to <4 x half>, !dbg !26
  %math.packed385 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %527) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %498, <4 x float> %496) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %513, <4 x float> %497) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %530 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %531 = extractelement <4 x float> %lazy.score2155, i64 0, !dbg !24
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !24
  %local.reduce75 = fadd float %k0.sum.anchor, %532, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1984 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %533 = insertelement <4 x float> poison, float %532, i64 0, !dbg !26
  %534 = extractelement <4 x float> %lazy.score2155, i64 1, !dbg !24
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !24
  %local.reduce76 = fadd float %k0.sum.anchor1977, %535, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1985 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %536 = insertelement <4 x float> %533, float %535, i64 1, !dbg !26
  %537 = extractelement <4 x float> %lazy.score2155, i64 2, !dbg !24
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !24
  %local.reduce77 = fadd float %k0.sum.anchor1978, %538, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1986 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %539 = insertelement <4 x float> %536, float %538, i64 2, !dbg !26
  %540 = extractelement <4 x float> %lazy.score2155, i64 3, !dbg !24
  %541 = call float @llvm.amdgcn.exp2.f32(float %540), !dbg !24
  %local.reduce78 = fadd float %k0.sum.anchor1979, %541, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1987 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %542 = insertelement <4 x float> %539, float %541, i64 3, !dbg !26
  %math.result370 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02198 = extractelement <4 x float> %math.result370, i32 0, !dbg !26
  %.cvtpack.f12199 = extractelement <4 x float> %math.result370, i32 1, !dbg !26
  %.cvtpack.f22200 = extractelement <4 x float> %math.result370, i32 2, !dbg !26
  %.cvtpack.f32201 = extractelement <4 x float> %math.result370, i32 3, !dbg !26
  %.cvtpack.p02202 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02198, float %.cvtpack.f12199), !dbg !26
  %.cvtpack.p12203 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22200, float %.cvtpack.f32201), !dbg !26
  %.cvtpack.v02204 = insertelement <2 x i32> poison, i32 %.cvtpack.p02202, i32 0, !dbg !26
  %.cvtpack.v12205 = insertelement <2 x i32> %.cvtpack.v02204, i32 %.cvtpack.p12203, i32 1, !dbg !26
  %543 = bitcast <2 x i32> %.cvtpack.v12205 to <4 x half>, !dbg !26
  %math.packed371 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %530, <4 x float> %512) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %545 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %545, <4 x float> %514) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %547 = extractelement <4 x float> %lazy.score2163, i64 0, !dbg !24
  %548 = call float @llvm.amdgcn.exp2.f32(float %547), !dbg !24
  %local.reduce44 = fadd float %k0.sum.anchor1980, %548, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1988 = call float asm sideeffect "", "=v,0"(float %local.reduce44) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %549 = insertelement <4 x float> poison, float %548, i64 0, !dbg !26
  %550 = extractelement <4 x float> %lazy.score2163, i64 1, !dbg !24
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !24
  %local.reduce45 = fadd float %k0.sum.anchor1981, %551, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1989 = call float asm sideeffect "", "=v,0"(float %local.reduce45) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %552 = insertelement <4 x float> %549, float %551, i64 1, !dbg !26
  %553 = extractelement <4 x float> %lazy.score2163, i64 2, !dbg !24
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !24
  %local.reduce46 = fadd float %k0.sum.anchor1982, %554, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1990 = call float asm sideeffect "", "=v,0"(float %local.reduce46) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %555 = insertelement <4 x float> %552, float %554, i64 2, !dbg !26
  %556 = extractelement <4 x float> %lazy.score2163, i64 3, !dbg !24
  %557 = call float @llvm.amdgcn.exp2.f32(float %556), !dbg !24
  %local.reduce47 = fadd float %k0.sum.anchor1983, %557, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1991 = call float asm sideeffect "", "=v,0"(float %local.reduce47) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %558 = insertelement <4 x float> %555, float %557, i64 3, !dbg !26
  %math.result386 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %558) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02206 = extractelement <4 x float> %math.result386, i32 0, !dbg !26
  %.cvtpack.f12207 = extractelement <4 x float> %math.result386, i32 1, !dbg !26
  %.cvtpack.f22208 = extractelement <4 x float> %math.result386, i32 2, !dbg !26
  %.cvtpack.f32209 = extractelement <4 x float> %math.result386, i32 3, !dbg !26
  %.cvtpack.p02210 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02206, float %.cvtpack.f12207), !dbg !26
  %.cvtpack.p12211 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22208, float %.cvtpack.f32209), !dbg !26
  %.cvtpack.v02212 = insertelement <2 x i32> poison, i32 %.cvtpack.p02210, i32 0, !dbg !26
  %.cvtpack.v12213 = insertelement <2 x i32> %.cvtpack.v02212, i32 %.cvtpack.p12211, i32 1, !dbg !26
  %559 = bitcast <2 x i32> %.cvtpack.v12213 to <4 x half>, !dbg !26
  %math.packed387 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %559) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %530, <4 x float> %528) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %545, <4 x float> %529) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %562 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %563 = extractelement <4 x float> %lazy.score2156, i64 0, !dbg !24
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !24
  %local.reduce79 = fadd float %k0.sum.anchor1984, %564, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1992 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !26
  %566 = extractelement <4 x float> %lazy.score2156, i64 1, !dbg !24
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !24
  %local.reduce80 = fadd float %k0.sum.anchor1985, %567, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1993 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> %565, float %567, i64 1, !dbg !26
  %569 = extractelement <4 x float> %lazy.score2156, i64 2, !dbg !24
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !24
  %local.reduce81 = fadd float %k0.sum.anchor1986, %570, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1994 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 2, !dbg !26
  %572 = extractelement <4 x float> %lazy.score2156, i64 3, !dbg !24
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !24
  %local.reduce82 = fadd float %k0.sum.anchor1987, %573, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1995 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %571, float %573, i64 3, !dbg !26
  %math.result372 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02214 = extractelement <4 x float> %math.result372, i32 0, !dbg !26
  %.cvtpack.f12215 = extractelement <4 x float> %math.result372, i32 1, !dbg !26
  %.cvtpack.f22216 = extractelement <4 x float> %math.result372, i32 2, !dbg !26
  %.cvtpack.f32217 = extractelement <4 x float> %math.result372, i32 3, !dbg !26
  %.cvtpack.p02218 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02214, float %.cvtpack.f12215), !dbg !26
  %.cvtpack.p12219 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22216, float %.cvtpack.f32217), !dbg !26
  %.cvtpack.v02220 = insertelement <2 x i32> poison, i32 %.cvtpack.p02218, i32 0, !dbg !26
  %.cvtpack.v12221 = insertelement <2 x i32> %.cvtpack.v02220, i32 %.cvtpack.p12219, i32 1, !dbg !26
  %575 = bitcast <2 x i32> %.cvtpack.v12221 to <4 x half>, !dbg !26
  %math.packed373 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %562, <4 x float> %544) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %577 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %577, <4 x float> %546) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = extractelement <4 x float> %lazy.score2164, i64 0, !dbg !24
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !24
  %local.reduce48 = fadd float %k0.sum.anchor1988, %580, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1996 = call float asm sideeffect "", "=v,0"(float %local.reduce48) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %581 = insertelement <4 x float> poison, float %580, i64 0, !dbg !26
  %582 = extractelement <4 x float> %lazy.score2164, i64 1, !dbg !24
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !24
  %local.reduce49 = fadd float %k0.sum.anchor1989, %583, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1997 = call float asm sideeffect "", "=v,0"(float %local.reduce49) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %584 = insertelement <4 x float> %581, float %583, i64 1, !dbg !26
  %585 = extractelement <4 x float> %lazy.score2164, i64 2, !dbg !24
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !24
  %local.reduce50 = fadd float %k0.sum.anchor1990, %586, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1998 = call float asm sideeffect "", "=v,0"(float %local.reduce50) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %587 = insertelement <4 x float> %584, float %586, i64 2, !dbg !26
  %588 = extractelement <4 x float> %lazy.score2164, i64 3, !dbg !24
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !24
  %local.reduce51 = fadd float %k0.sum.anchor1991, %589, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1999 = call float asm sideeffect "", "=v,0"(float %local.reduce51) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %590 = insertelement <4 x float> %587, float %589, i64 3, !dbg !26
  %math.result388 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %590) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02222 = extractelement <4 x float> %math.result388, i32 0, !dbg !26
  %.cvtpack.f12223 = extractelement <4 x float> %math.result388, i32 1, !dbg !26
  %.cvtpack.f22224 = extractelement <4 x float> %math.result388, i32 2, !dbg !26
  %.cvtpack.f32225 = extractelement <4 x float> %math.result388, i32 3, !dbg !26
  %.cvtpack.p02226 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02222, float %.cvtpack.f12223), !dbg !26
  %.cvtpack.p12227 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22224, float %.cvtpack.f32225), !dbg !26
  %.cvtpack.v02228 = insertelement <2 x i32> poison, i32 %.cvtpack.p02226, i32 0, !dbg !26
  %.cvtpack.v12229 = insertelement <2 x i32> %.cvtpack.v02228, i32 %.cvtpack.p12227, i32 1, !dbg !26
  %591 = bitcast <2 x i32> %.cvtpack.v12229 to <4 x half>, !dbg !26
  %math.packed389 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %591) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %562, <4 x float> %560) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %577, <4 x float> %561) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %594 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %595 = extractelement <4 x float> %lazy.score2157, i64 0, !dbg !24
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !24
  %local.reduce83 = fadd float %k0.sum.anchor1992, %596, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2000 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %597 = insertelement <4 x float> poison, float %596, i64 0, !dbg !26
  %598 = extractelement <4 x float> %lazy.score2157, i64 1, !dbg !24
  %599 = call float @llvm.amdgcn.exp2.f32(float %598), !dbg !24
  %local.reduce84 = fadd float %k0.sum.anchor1993, %599, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2001 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %600 = insertelement <4 x float> %597, float %599, i64 1, !dbg !26
  %601 = extractelement <4 x float> %lazy.score2157, i64 2, !dbg !24
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !24
  %local.reduce85 = fadd float %k0.sum.anchor1994, %602, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2002 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %603 = insertelement <4 x float> %600, float %602, i64 2, !dbg !26
  %604 = extractelement <4 x float> %lazy.score2157, i64 3, !dbg !24
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !24
  %local.reduce86 = fadd float %k0.sum.anchor1995, %605, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2003 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %606 = insertelement <4 x float> %603, float %605, i64 3, !dbg !26
  %math.result374 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02230 = extractelement <4 x float> %math.result374, i32 0, !dbg !26
  %.cvtpack.f12231 = extractelement <4 x float> %math.result374, i32 1, !dbg !26
  %.cvtpack.f22232 = extractelement <4 x float> %math.result374, i32 2, !dbg !26
  %.cvtpack.f32233 = extractelement <4 x float> %math.result374, i32 3, !dbg !26
  %.cvtpack.p02234 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02230, float %.cvtpack.f12231), !dbg !26
  %.cvtpack.p12235 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22232, float %.cvtpack.f32233), !dbg !26
  %.cvtpack.v02236 = insertelement <2 x i32> poison, i32 %.cvtpack.p02234, i32 0, !dbg !26
  %.cvtpack.v12237 = insertelement <2 x i32> %.cvtpack.v02236, i32 %.cvtpack.p12235, i32 1, !dbg !26
  %607 = bitcast <2 x i32> %.cvtpack.v12237 to <4 x half>, !dbg !26
  %math.packed375 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %594, <4 x float> %576) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %609 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %609, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %611 = extractelement <4 x float> %lazy.score2165, i64 0, !dbg !24
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !24
  %local.reduce52 = fadd float %k0.sum.anchor1996, %612, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2004 = call float asm sideeffect "", "=v,0"(float %local.reduce52) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %613 = insertelement <4 x float> poison, float %612, i64 0, !dbg !26
  %614 = extractelement <4 x float> %lazy.score2165, i64 1, !dbg !24
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !24
  %local.reduce53 = fadd float %k0.sum.anchor1997, %615, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2005 = call float asm sideeffect "", "=v,0"(float %local.reduce53) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %616 = insertelement <4 x float> %613, float %615, i64 1, !dbg !26
  %617 = extractelement <4 x float> %lazy.score2165, i64 2, !dbg !24
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !24
  %local.reduce54 = fadd float %k0.sum.anchor1998, %618, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2006 = call float asm sideeffect "", "=v,0"(float %local.reduce54) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> %616, float %618, i64 2, !dbg !26
  %620 = extractelement <4 x float> %lazy.score2165, i64 3, !dbg !24
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !24
  %local.reduce55 = fadd float %k0.sum.anchor1999, %621, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2007 = call float asm sideeffect "", "=v,0"(float %local.reduce55) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 3, !dbg !26
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %622) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02238 = extractelement <4 x float> %math.result390, i32 0, !dbg !26
  %.cvtpack.f12239 = extractelement <4 x float> %math.result390, i32 1, !dbg !26
  %.cvtpack.f22240 = extractelement <4 x float> %math.result390, i32 2, !dbg !26
  %.cvtpack.f32241 = extractelement <4 x float> %math.result390, i32 3, !dbg !26
  %.cvtpack.p02242 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02238, float %.cvtpack.f12239), !dbg !26
  %.cvtpack.p12243 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22240, float %.cvtpack.f32241), !dbg !26
  %.cvtpack.v02244 = insertelement <2 x i32> poison, i32 %.cvtpack.p02242, i32 0, !dbg !26
  %.cvtpack.v12245 = insertelement <2 x i32> %.cvtpack.v02244, i32 %.cvtpack.p12243, i32 1, !dbg !26
  %623 = bitcast <2 x i32> %.cvtpack.v12245 to <4 x half>, !dbg !26
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %623) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %594, <4 x float> %592) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %609, <4 x float> %593) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %626 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = extractelement <4 x float> %lazy.score2158, i64 0, !dbg !24
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !24
  %local.reduce87 = fadd float %k0.sum.anchor2000, %628, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2008 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !26
  %630 = extractelement <4 x float> %lazy.score2158, i64 1, !dbg !24
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !24
  %local.reduce88 = fadd float %k0.sum.anchor2001, %631, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2009 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %632 = insertelement <4 x float> %629, float %631, i64 1, !dbg !26
  %633 = extractelement <4 x float> %lazy.score2158, i64 2, !dbg !24
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !24
  %local.reduce89 = fadd float %k0.sum.anchor2002, %634, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2010 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %635 = insertelement <4 x float> %632, float %634, i64 2, !dbg !26
  %636 = extractelement <4 x float> %lazy.score2158, i64 3, !dbg !24
  %637 = call float @llvm.amdgcn.exp2.f32(float %636), !dbg !24
  %local.reduce90 = fadd float %k0.sum.anchor2003, %637, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2011 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %638 = insertelement <4 x float> %635, float %637, i64 3, !dbg !26
  %math.result376 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %638) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02246 = extractelement <4 x float> %math.result376, i32 0, !dbg !26
  %.cvtpack.f12247 = extractelement <4 x float> %math.result376, i32 1, !dbg !26
  %.cvtpack.f22248 = extractelement <4 x float> %math.result376, i32 2, !dbg !26
  %.cvtpack.f32249 = extractelement <4 x float> %math.result376, i32 3, !dbg !26
  %.cvtpack.p02250 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02246, float %.cvtpack.f12247), !dbg !26
  %.cvtpack.p12251 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22248, float %.cvtpack.f32249), !dbg !26
  %.cvtpack.v02252 = insertelement <2 x i32> poison, i32 %.cvtpack.p02250, i32 0, !dbg !26
  %.cvtpack.v12253 = insertelement <2 x i32> %.cvtpack.v02252, i32 %.cvtpack.p12251, i32 1, !dbg !26
  %639 = bitcast <2 x i32> %.cvtpack.v12253 to <4 x half>, !dbg !26
  %math.packed377 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %639) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %626, <4 x float> %608) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %641 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %641, <4 x float> %610) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %643 = extractelement <4 x float> %lazy.score2166, i64 0, !dbg !24
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !24
  %local.reduce56 = fadd float %k0.sum.anchor2004, %644, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2012 = call float asm sideeffect "", "=v,0"(float %local.reduce56) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %645 = insertelement <4 x float> poison, float %644, i64 0, !dbg !26
  %646 = extractelement <4 x float> %lazy.score2166, i64 1, !dbg !24
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !24
  %local.reduce57 = fadd float %k0.sum.anchor2005, %647, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2013 = call float asm sideeffect "", "=v,0"(float %local.reduce57) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %648 = insertelement <4 x float> %645, float %647, i64 1, !dbg !26
  %649 = extractelement <4 x float> %lazy.score2166, i64 2, !dbg !24
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !24
  %local.reduce58 = fadd float %k0.sum.anchor2006, %650, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2014 = call float asm sideeffect "", "=v,0"(float %local.reduce58) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %651 = insertelement <4 x float> %648, float %650, i64 2, !dbg !26
  %652 = extractelement <4 x float> %lazy.score2166, i64 3, !dbg !24
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !24
  %local.reduce59 = fadd float %k0.sum.anchor2007, %653, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2015 = call float asm sideeffect "", "=v,0"(float %local.reduce59) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %654 = insertelement <4 x float> %651, float %653, i64 3, !dbg !26
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02254 = extractelement <4 x float> %math.result392, i32 0, !dbg !26
  %.cvtpack.f12255 = extractelement <4 x float> %math.result392, i32 1, !dbg !26
  %.cvtpack.f22256 = extractelement <4 x float> %math.result392, i32 2, !dbg !26
  %.cvtpack.f32257 = extractelement <4 x float> %math.result392, i32 3, !dbg !26
  %.cvtpack.p02258 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02254, float %.cvtpack.f12255), !dbg !26
  %.cvtpack.p12259 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22256, float %.cvtpack.f32257), !dbg !26
  %.cvtpack.v02260 = insertelement <2 x i32> poison, i32 %.cvtpack.p02258, i32 0, !dbg !26
  %.cvtpack.v12261 = insertelement <2 x i32> %.cvtpack.v02260, i32 %.cvtpack.p12259, i32 1, !dbg !26
  %655 = bitcast <2 x i32> %.cvtpack.v12261 to <4 x half>, !dbg !26
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %626, <4 x float> %624) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %641, <4 x float> %625) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %658 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = extractelement <4 x float> %lazy.score2159, i64 0, !dbg !24
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !24
  %local.reduce91 = fadd float %k0.sum.anchor2008, %660, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2016 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %661 = insertelement <4 x float> poison, float %660, i64 0, !dbg !26
  %662 = extractelement <4 x float> %lazy.score2159, i64 1, !dbg !24
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !24
  %local.reduce92 = fadd float %k0.sum.anchor2009, %663, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2017 = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %664 = insertelement <4 x float> %661, float %663, i64 1, !dbg !26
  %665 = extractelement <4 x float> %lazy.score2159, i64 2, !dbg !24
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !24
  %local.reduce93 = fadd float %k0.sum.anchor2010, %666, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2018 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> %664, float %666, i64 2, !dbg !26
  %668 = extractelement <4 x float> %lazy.score2159, i64 3, !dbg !24
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !24
  %local.reduce94 = fadd float %k0.sum.anchor2011, %669, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2019 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %667, float %669, i64 3, !dbg !26
  %math.result378 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02262 = extractelement <4 x float> %math.result378, i32 0, !dbg !26
  %.cvtpack.f12263 = extractelement <4 x float> %math.result378, i32 1, !dbg !26
  %.cvtpack.f22264 = extractelement <4 x float> %math.result378, i32 2, !dbg !26
  %.cvtpack.f32265 = extractelement <4 x float> %math.result378, i32 3, !dbg !26
  %.cvtpack.p02266 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02262, float %.cvtpack.f12263), !dbg !26
  %.cvtpack.p12267 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22264, float %.cvtpack.f32265), !dbg !26
  %.cvtpack.v02268 = insertelement <2 x i32> poison, i32 %.cvtpack.p02266, i32 0, !dbg !26
  %.cvtpack.v12269 = insertelement <2 x i32> %.cvtpack.v02268, i32 %.cvtpack.p12267, i32 1, !dbg !26
  %671 = bitcast <2 x i32> %.cvtpack.v12269 to <4 x half>, !dbg !26
  %math.packed379 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %658, <4 x float> %640) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %673 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %673, <4 x float> %642) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %675 = extractelement <4 x float> %lazy.score2167, i64 0, !dbg !24
  %676 = call float @llvm.amdgcn.exp2.f32(float %675), !dbg !24
  %local.reduce60 = fadd float %k0.sum.anchor2012, %676, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2020 = call float asm sideeffect "", "=v,0"(float %local.reduce60) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %677 = insertelement <4 x float> poison, float %676, i64 0, !dbg !26
  %678 = extractelement <4 x float> %lazy.score2167, i64 1, !dbg !24
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !24
  %local.reduce61 = fadd float %k0.sum.anchor2013, %679, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2021 = call float asm sideeffect "", "=v,0"(float %local.reduce61) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %680 = insertelement <4 x float> %677, float %679, i64 1, !dbg !26
  %681 = extractelement <4 x float> %lazy.score2167, i64 2, !dbg !24
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !24
  %local.reduce62 = fadd float %k0.sum.anchor2014, %682, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2022 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %683 = insertelement <4 x float> %680, float %682, i64 2, !dbg !26
  %684 = extractelement <4 x float> %lazy.score2167, i64 3, !dbg !24
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !24
  %local.reduce63 = fadd float %k0.sum.anchor2015, %685, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2023 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %686 = insertelement <4 x float> %683, float %685, i64 3, !dbg !26
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %686) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02270 = extractelement <4 x float> %math.result394, i32 0, !dbg !26
  %.cvtpack.f12271 = extractelement <4 x float> %math.result394, i32 1, !dbg !26
  %.cvtpack.f22272 = extractelement <4 x float> %math.result394, i32 2, !dbg !26
  %.cvtpack.f32273 = extractelement <4 x float> %math.result394, i32 3, !dbg !26
  %.cvtpack.p02274 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02270, float %.cvtpack.f12271), !dbg !26
  %.cvtpack.p12275 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22272, float %.cvtpack.f32273), !dbg !26
  %.cvtpack.v02276 = insertelement <2 x i32> poison, i32 %.cvtpack.p02274, i32 0, !dbg !26
  %.cvtpack.v12277 = insertelement <2 x i32> %.cvtpack.v02276, i32 %.cvtpack.p12275, i32 1, !dbg !26
  %687 = bitcast <2 x i32> %.cvtpack.v12277 to <4 x half>, !dbg !26
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %687) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %658, <4 x float> %656) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %673, <4 x float> %657) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %690 = bitcast <4 x i32> %465 to <8 x half>, !dbg !14
  %691 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %692 = extractelement <4 x float> %lazy.score2160, i64 0, !dbg !24
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !24
  %local.reduce95 = fadd float %k0.sum.anchor2016, %693, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2024 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %694 = insertelement <4 x float> poison, float %693, i64 0, !dbg !26
  %695 = extractelement <4 x float> %lazy.score2160, i64 1, !dbg !24
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !24
  %local.reduce96 = fadd float %k0.sum.anchor2017, %696, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2025 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce99 = fadd float %k0.sum.anchor2024, %k0.sum.anchor2025, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2026 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %697 = insertelement <4 x float> %694, float %696, i64 1, !dbg !26
  %698 = extractelement <4 x float> %lazy.score2160, i64 2, !dbg !24
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !24
  %local.reduce97 = fadd float %k0.sum.anchor2018, %699, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2027 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %700 = insertelement <4 x float> %697, float %699, i64 2, !dbg !26
  %701 = extractelement <4 x float> %lazy.score2160, i64 3, !dbg !24
  %702 = call float @llvm.amdgcn.exp2.f32(float %701), !dbg !24
  %local.reduce98 = fadd float %k0.sum.anchor2019, %702, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2028 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce100 = fadd float %k0.sum.anchor2027, %k0.sum.anchor2028, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2029 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %703 = fadd float %k0.sum.anchor2026, %k0.sum.anchor2029, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2030 = call float asm sideeffect "", "=v,0"(float %703) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %704 = fadd float %k0.sum.anchor2030, %440, !dbg !25
  %k0.sum.anchor2031 = call float asm sideeffect "", "=v,0"(float %704) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %705 = insertelement <1 x float> poison, float %k0.sum.anchor2031, i64 0, !dbg !6
  %706 = insertvalue [2 x <1 x float>] poison, <1 x float> %705, 0, !dbg !6
  %707 = insertelement <4 x float> %700, float %702, i64 3, !dbg !26
  %math.result380 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02278 = extractelement <4 x float> %math.result380, i32 0, !dbg !26
  %.cvtpack.f12279 = extractelement <4 x float> %math.result380, i32 1, !dbg !26
  %.cvtpack.f22280 = extractelement <4 x float> %math.result380, i32 2, !dbg !26
  %.cvtpack.f32281 = extractelement <4 x float> %math.result380, i32 3, !dbg !26
  %.cvtpack.p02282 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02278, float %.cvtpack.f12279), !dbg !26
  %.cvtpack.p12283 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22280, float %.cvtpack.f32281), !dbg !26
  %.cvtpack.v02284 = insertelement <2 x i32> poison, i32 %.cvtpack.p02282, i32 0, !dbg !26
  %.cvtpack.v12285 = insertelement <2 x i32> %.cvtpack.v02284, i32 %.cvtpack.p12283, i32 1, !dbg !26
  %708 = bitcast <2 x i32> %.cvtpack.v12285 to <4 x half>, !dbg !26
  %math.packed381 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %691, <4 x float> %672) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %710 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %710, <4 x float> %674) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %712 = extractelement <4 x float> %lazy.score2168, i64 0, !dbg !24
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !24
  %local.reduce64 = fadd float %k0.sum.anchor2020, %713, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2032 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %714 = insertelement <4 x float> poison, float %713, i64 0, !dbg !26
  %715 = extractelement <4 x float> %lazy.score2168, i64 1, !dbg !24
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !24
  %local.reduce65 = fadd float %k0.sum.anchor2021, %716, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2033 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce68 = fadd float %k0.sum.anchor2032, %k0.sum.anchor2033, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2034 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %717 = insertelement <4 x float> %714, float %716, i64 1, !dbg !26
  %718 = extractelement <4 x float> %lazy.score2168, i64 2, !dbg !24
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !24
  %local.reduce66 = fadd float %k0.sum.anchor2022, %719, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2035 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %720 = insertelement <4 x float> %717, float %719, i64 2, !dbg !26
  %721 = extractelement <4 x float> %lazy.score2168, i64 3, !dbg !24
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !24
  %local.reduce67 = fadd float %k0.sum.anchor2023, %722, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2036 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce69 = fadd float %k0.sum.anchor2035, %k0.sum.anchor2036, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2037 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %723 = fadd float %k0.sum.anchor2034, %k0.sum.anchor2037, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2038 = call float asm sideeffect "", "=v,0"(float %723) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %724 = fadd float %k0.sum.anchor2038, %442, !dbg !25
  %k0.sum.anchor2039 = call float asm sideeffect "", "=v,0"(float %724) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %725 = insertelement <1 x float> poison, float %k0.sum.anchor2039, i64 0, !dbg !6
  %726 = insertvalue [2 x <1 x float>] %706, <1 x float> %725, 1, !dbg !6
  %727 = insertelement <4 x float> %720, float %722, i64 3, !dbg !26
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02286 = extractelement <4 x float> %math.result396, i32 0, !dbg !26
  %.cvtpack.f12287 = extractelement <4 x float> %math.result396, i32 1, !dbg !26
  %.cvtpack.f22288 = extractelement <4 x float> %math.result396, i32 2, !dbg !26
  %.cvtpack.f32289 = extractelement <4 x float> %math.result396, i32 3, !dbg !26
  %.cvtpack.p02290 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02286, float %.cvtpack.f12287), !dbg !26
  %.cvtpack.p12291 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22288, float %.cvtpack.f32289), !dbg !26
  %.cvtpack.v02292 = insertelement <2 x i32> poison, i32 %.cvtpack.p02290, i32 0, !dbg !26
  %.cvtpack.v12293 = insertelement <2 x i32> %.cvtpack.v02292, i32 %.cvtpack.p12291, i32 1, !dbg !26
  %728 = bitcast <2 x i32> %.cvtpack.v12293 to <4 x half>, !dbg !26
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %691, <4 x float> %688) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %710, <4 x float> %689) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %731 = extractvalue [2 x <8 x float>] %lazy.state2172, 0, !dbg !27
  %732 = shufflevector <8 x float> %731, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %733 = shufflevector <8 x float> %731, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %734 = extractvalue [2 x <8 x float>] %lazy.state2172, 1, !dbg !27
  %735 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %736 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
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
  %737 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %vdb.address1443, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %738 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %vdb.address1450, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local932, i64 8192, !dbg !14
  %739 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %740 = bitcast <4 x i32> %739 to <8 x half>, !dbg !14
  %vdb.read1434 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8448, !dbg !14
  %741 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %742 = bitcast <4 x i32> %741 to <8 x half>, !dbg !14
  %vdb.read1435 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8704, !dbg !14
  %743 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %744 = bitcast <4 x i32> %743 to <8 x half>, !dbg !14
  %vdb.read1436 = getelementptr i8, ptr addrspace(3) %address.local941, i64 8960, !dbg !14
  %745 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %746 = bitcast <4 x i32> %745 to <8 x half>, !dbg !14
  %vdb.read1437 = getelementptr i8, ptr addrspace(3) %address.local932, i64 12288, !dbg !14
  %747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !14
  %vdb.read1438 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12544, !dbg !14
  %749 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %750 = bitcast <4 x i32> %749 to <8 x half>, !dbg !14
  %vdb.read1439 = getelementptr i8, ptr addrspace(3) %address.local941, i64 12800, !dbg !14
  %751 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %752 = bitcast <4 x i32> %751 to <8 x half>, !dbg !14
  %vdb.read1440 = getelementptr i8, ptr addrspace(3) %address.local941, i64 13056, !dbg !14
  %753 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %754 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %754, <4 x float> %732) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %756 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %756, <4 x float> %733) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %754, <4 x float> %735) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %756, <4 x float> %736) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %760 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %760, <4 x float> %755) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %762 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %762, <4 x float> %757) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %760, <4 x float> %758) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %762, <4 x float> %759) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %766 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %766, <4 x float> %761) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %768 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %768, <4 x float> %763) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %766, <4 x float> %764) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %768, <4 x float> %765) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %772 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %772, <4 x float> %767) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %774 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %774, <4 x float> %769) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %772, <4 x float> %770) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %774, <4 x float> %771) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %778, <4 x float> %773) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %780 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %780, <4 x float> %775) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %778, <4 x float> %776) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %780, <4 x float> %777) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %784 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %784, <4 x float> %779) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %786 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %786, <4 x float> %781) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %788 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %784, <4 x float> %782) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %786, <4 x float> %783) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %790 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %790, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %792 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %792, <4 x float> %787) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %790, <4 x float> %788) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %792, <4 x float> %789) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = bitcast <4 x i32> %753 to <8 x half>, !dbg !14
  %797 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %797, <4 x float> %791) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %799, <4 x float> %793) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %797, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %799, <4 x float> %795) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %803 = extractvalue [2 x <8 x float>] %lazy.state2171, 0, !dbg !27
  %804 = shufflevector <8 x float> %803, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %805 = shufflevector <8 x float> %803, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %806 = extractvalue [2 x <8 x float>] %lazy.state2171, 1, !dbg !27
  %807 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %808 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
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
  %809 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1457, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %810 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1433, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1340) #10, !dbg !16
  %811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local932, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %812 = bitcast <4 x i32> %811 to <8 x half>, !dbg !14
  %813 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local942, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %814 = bitcast <4 x i32> %813 to <8 x half>, !dbg !14
  %815 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local946, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %816 = bitcast <4 x i32> %815 to <8 x half>, !dbg !14
  %817 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local950, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %818 = bitcast <4 x i32> %817 to <8 x half>, !dbg !14
  %819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %820 = bitcast <4 x i32> %819 to <8 x half>, !dbg !14
  %821 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %822 = bitcast <4 x i32> %821 to <8 x half>, !dbg !14
  %823 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local960, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !14
  %825 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %826 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %826, <4 x float> %804) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %828, <4 x float> %805) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %826, <4 x float> %807) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %828, <4 x float> %808) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %832 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %832, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %834, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %832, <4 x float> %830) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %834, <4 x float> %831) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %838, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %840, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %838, <4 x float> %836) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %840, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %844, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %846, <4 x float> %841) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %844, <4 x float> %842) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %846, <4 x float> %843) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %850, <4 x float> %845) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %852, <4 x float> %847) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %850, <4 x float> %848) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %852, <4 x float> %849) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %856 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %856, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %858, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %856, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %858, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %862 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %862, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %864, <4 x float> %859) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %862, <4 x float> %860) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %864, <4 x float> %861) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = bitcast <4 x i32> %825 to <8 x half>, !dbg !14
  %869 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %869, <4 x float> %863) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %871, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %869, <4 x float> %866) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %871, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %875 = extractvalue [2 x <8 x float>] %lazy.state2170, 0, !dbg !27
  %876 = shufflevector <8 x float> %875, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %877 = shufflevector <8 x float> %875, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %878 = extractvalue [2 x <8 x float>] %lazy.state2170, 1, !dbg !27
  %879 = shufflevector <8 x float> %878, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %880 = shufflevector <8 x float> %878, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %881 = add nuw nsw i64 %101, 1, !dbg !6
  %k0.has.next = icmp ult i64 %881, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %881, 14, !dbg !13
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

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %882 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !14
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1434, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !14
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1435, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !14
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1436, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !14
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1437, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !14
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1438, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !14
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1439, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !14
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1440, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %897 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %897, <4 x float> %876) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %899, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %897, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %899, <4 x float> %880) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %903, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed369, <4 x half> %905, <4 x float> %900) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %903, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %905, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %909, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed371, <4 x half> %911, <4 x float> %906) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %909, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %911, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %915, <4 x float> %910) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed373, <4 x half> %917, <4 x float> %912) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %915, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %917, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %921, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %923, <4 x float> %918) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %921, <4 x float> %919) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %923, <4 x float> %920) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %927, <4 x float> %922) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %929, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %927, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %929, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %933, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %935 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %935, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %933, <4 x float> %931) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %935, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = bitcast <4 x i32> %896 to <8 x half>, !dbg !14
  %940 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %940, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %942 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %942, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %940, <4 x float> %937) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %942, <4 x float> %938) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %946 = shufflevector <4 x float> %709, <4 x float> %711, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %947 = insertvalue [2 x <8 x float>] poison, <8 x float> %946, 0, !dbg !6
  %948 = shufflevector <4 x float> %729, <4 x float> %730, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %949 = insertvalue [2 x <8 x float>] %947, <8 x float> %948, 1, !dbg !6
  %950 = shufflevector <4 x float> %798, <4 x float> %800, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %951 = insertvalue [2 x <8 x float>] poison, <8 x float> %950, 0, !dbg !6
  %952 = shufflevector <4 x float> %801, <4 x float> %802, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %953 = insertvalue [2 x <8 x float>] %951, <8 x float> %952, 1, !dbg !6
  %954 = shufflevector <4 x float> %870, <4 x float> %872, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %955 = insertvalue [2 x <8 x float>] poison, <8 x float> %954, 0, !dbg !6
  %956 = shufflevector <4 x float> %873, <4 x float> %874, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %957 = insertvalue [2 x <8 x float>] %955, <8 x float> %956, 1, !dbg !6
  %958 = shufflevector <4 x float> %941, <4 x float> %943, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %959 = insertvalue [2 x <8 x float>] poison, <8 x float> %958, 0, !dbg !6
  %960 = shufflevector <4 x float> %944, <4 x float> %945, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %961 = insertvalue [2 x <8 x float>] %959, <8 x float> %960, 1, !dbg !6
  br label %100, !dbg !6

962:                                              ; preds = %100
  %lane.sum.field166 = extractvalue [2 x <1 x float>] %103, 1, !dbg !6
  %lane.sum.local167 = extractelement <1 x float> %lane.sum.field166, i64 0, !dbg !6
  %963 = shl nuw nsw i32 %8, 2, !dbg !6
  %964 = xor i32 %963, 64, !dbg !6
  %965 = bitcast float %lane.sum.local167 to i32, !dbg !6
  %lane.sum.shuffle170 = call i32 @llvm.amdgcn.ds.bpermute(i32 %964, i32 %965), !dbg !6
  %966 = bitcast i32 %lane.sum.shuffle170 to float, !dbg !6
  %lane.sum.final171 = fadd float %lane.sum.local167, %966, !dbg !6
  %967 = xor i32 %963, 128, !dbg !6
  %968 = bitcast float %lane.sum.final171 to i32, !dbg !6
  %lane.sum.shuffle172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %968), !dbg !6
  %969 = bitcast i32 %lane.sum.shuffle172 to float, !dbg !6
  %lane.sum.final173 = fadd float %lane.sum.final171, %969, !dbg !6
  %970 = insertelement <1 x float> poison, float %lane.sum.final173, i64 0, !dbg !6
  %lane.sum.field = extractvalue [2 x <1 x float>] %103, 0, !dbg !6
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !6
  %971 = bitcast float %lane.sum.local to i32, !dbg !6
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %964, i32 %971), !dbg !6
  %972 = bitcast i32 %lane.sum.shuffle to float, !dbg !6
  %lane.sum.final = fadd float %lane.sum.local, %972, !dbg !6
  %973 = bitcast float %lane.sum.final to i32, !dbg !6
  %lane.sum.shuffle164 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %973), !dbg !6
  %974 = bitcast i32 %lane.sum.shuffle164 to float, !dbg !6
  %lane.sum.final165 = fadd float %lane.sum.final, %974, !dbg !6
  %975 = insertelement <1 x float> poison, float %lane.sum.final165, i64 0, !dbg !6
  %976 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %976, 0, !dbg !6
  %977 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %977, 0, !dbg !6
  %978 = extractvalue [2 x <8 x float>] %107, 0, !dbg !6
  %979 = extractelement <8 x float> %978, i64 0, !dbg !6
  %980 = extractelement <8 x float> %978, i64 1, !dbg !6
  %981 = extractelement <8 x float> %978, i64 2, !dbg !6
  %982 = extractelement <8 x float> %978, i64 3, !dbg !6
  %983 = select i1 %.not, float %980, float %979, !dbg !6
  %984 = bitcast float %983 to i32, !dbg !6
  %985 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %984), !dbg !6
  %986 = bitcast i32 %985 to float, !dbg !6
  %987 = select i1 %.not, float %979, float %986, !dbg !6
  %988 = select i1 %.not, float %986, float %980, !dbg !6
  %989 = select i1 %.not, float %982, float %981, !dbg !6
  %990 = bitcast float %989 to i32, !dbg !6
  %991 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %990), !dbg !6
  %992 = bitcast i32 %991 to float, !dbg !6
  %993 = select i1 %.not, float %981, float %992, !dbg !6
  %994 = select i1 %.not, float %992, float %982, !dbg !6
  %995 = select i1 %.not40, float %993, float %987, !dbg !6
  %996 = bitcast float %995 to i32, !dbg !6
  %997 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %996), !dbg !6
  %998 = bitcast i32 %997 to float, !dbg !6
  %999 = select i1 %.not40, float %987, float %998, !dbg !6
  %1000 = select i1 %.not40, float %998, float %993, !dbg !6
  %1001 = select i1 %.not40, float %994, float %988, !dbg !6
  %1002 = bitcast float %1001 to i32, !dbg !6
  %1003 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1002), !dbg !6
  %1004 = bitcast i32 %1003 to float, !dbg !6
  %1005 = select i1 %.not40, float %988, float %1004, !dbg !6
  %1006 = select i1 %.not40, float %1004, float %994, !dbg !6
  %1007 = extractelement <8 x float> %978, i64 4, !dbg !6
  %1008 = extractelement <8 x float> %978, i64 5, !dbg !6
  %1009 = extractelement <8 x float> %978, i64 6, !dbg !6
  %1010 = extractelement <8 x float> %978, i64 7, !dbg !6
  %1011 = select i1 %.not, float %1008, float %1007, !dbg !6
  %1012 = bitcast float %1011 to i32, !dbg !6
  %1013 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1012), !dbg !6
  %1014 = bitcast i32 %1013 to float, !dbg !6
  %1015 = select i1 %.not, float %1007, float %1014, !dbg !6
  %1016 = select i1 %.not, float %1014, float %1008, !dbg !6
  %1017 = select i1 %.not, float %1010, float %1009, !dbg !6
  %1018 = bitcast float %1017 to i32, !dbg !6
  %1019 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1018), !dbg !6
  %1020 = bitcast i32 %1019 to float, !dbg !6
  %1021 = select i1 %.not, float %1009, float %1020, !dbg !6
  %1022 = select i1 %.not, float %1020, float %1010, !dbg !6
  %1023 = select i1 %.not40, float %1021, float %1015, !dbg !6
  %1024 = bitcast float %1023 to i32, !dbg !6
  %1025 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1024), !dbg !6
  %1026 = bitcast i32 %1025 to float, !dbg !6
  %1027 = select i1 %.not40, float %1015, float %1026, !dbg !6
  %1028 = select i1 %.not40, float %1026, float %1021, !dbg !6
  %1029 = select i1 %.not40, float %1022, float %1016, !dbg !6
  %1030 = bitcast float %1029 to i32, !dbg !6
  %1031 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1030), !dbg !6
  %1032 = bitcast i32 %1031 to float, !dbg !6
  %1033 = select i1 %.not40, float %1016, float %1032, !dbg !6
  %1034 = select i1 %.not40, float %1032, float %1022, !dbg !6
  %1035 = extractvalue [2 x <8 x float>] %107, 1, !dbg !6
  %1036 = extractelement <8 x float> %1035, i64 0, !dbg !6
  %1037 = extractelement <8 x float> %1035, i64 1, !dbg !6
  %1038 = extractelement <8 x float> %1035, i64 2, !dbg !6
  %1039 = extractelement <8 x float> %1035, i64 3, !dbg !6
  %1040 = select i1 %.not, float %1037, float %1036, !dbg !6
  %1041 = bitcast float %1040 to i32, !dbg !6
  %1042 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1041), !dbg !6
  %1043 = bitcast i32 %1042 to float, !dbg !6
  %1044 = select i1 %.not, float %1036, float %1043, !dbg !6
  %1045 = select i1 %.not, float %1043, float %1037, !dbg !6
  %1046 = select i1 %.not, float %1039, float %1038, !dbg !6
  %1047 = bitcast float %1046 to i32, !dbg !6
  %1048 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1047), !dbg !6
  %1049 = bitcast i32 %1048 to float, !dbg !6
  %1050 = select i1 %.not, float %1038, float %1049, !dbg !6
  %1051 = select i1 %.not, float %1049, float %1039, !dbg !6
  %1052 = select i1 %.not40, float %1050, float %1044, !dbg !6
  %1053 = bitcast float %1052 to i32, !dbg !6
  %1054 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1053), !dbg !6
  %1055 = bitcast i32 %1054 to float, !dbg !6
  %1056 = select i1 %.not40, float %1044, float %1055, !dbg !6
  %1057 = select i1 %.not40, float %1055, float %1050, !dbg !6
  %1058 = select i1 %.not40, float %1051, float %1045, !dbg !6
  %1059 = bitcast float %1058 to i32, !dbg !6
  %1060 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1059), !dbg !6
  %1061 = bitcast i32 %1060 to float, !dbg !6
  %1062 = select i1 %.not40, float %1045, float %1061, !dbg !6
  %1063 = select i1 %.not40, float %1061, float %1051, !dbg !6
  %1064 = extractelement <8 x float> %1035, i64 4, !dbg !6
  %1065 = extractelement <8 x float> %1035, i64 5, !dbg !6
  %1066 = extractelement <8 x float> %1035, i64 6, !dbg !6
  %1067 = extractelement <8 x float> %1035, i64 7, !dbg !6
  %1068 = select i1 %.not, float %1065, float %1064, !dbg !6
  %1069 = bitcast float %1068 to i32, !dbg !6
  %1070 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1069), !dbg !6
  %1071 = bitcast i32 %1070 to float, !dbg !6
  %1072 = select i1 %.not, float %1064, float %1071, !dbg !6
  %1073 = select i1 %.not, float %1071, float %1065, !dbg !6
  %1074 = select i1 %.not, float %1067, float %1066, !dbg !6
  %1075 = bitcast float %1074 to i32, !dbg !6
  %1076 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1075), !dbg !6
  %1077 = bitcast i32 %1076 to float, !dbg !6
  %1078 = select i1 %.not, float %1066, float %1077, !dbg !6
  %1079 = select i1 %.not, float %1077, float %1067, !dbg !6
  %1080 = select i1 %.not40, float %1078, float %1072, !dbg !6
  %1081 = bitcast float %1080 to i32, !dbg !6
  %1082 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1081), !dbg !6
  %1083 = bitcast i32 %1082 to float, !dbg !6
  %1084 = select i1 %.not40, float %1072, float %1083, !dbg !6
  %1085 = select i1 %.not40, float %1083, float %1078, !dbg !6
  %1086 = select i1 %.not40, float %1079, float %1073, !dbg !6
  %1087 = bitcast float %1086 to i32, !dbg !6
  %1088 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1087), !dbg !6
  %1089 = bitcast i32 %1088 to float, !dbg !6
  %1090 = select i1 %.not40, float %1073, float %1089, !dbg !6
  %1091 = select i1 %.not40, float %1089, float %1079, !dbg !6
  %1092 = extractvalue [2 x <8 x float>] %106, 0, !dbg !6
  %1093 = extractelement <8 x float> %1092, i64 0, !dbg !6
  %1094 = extractelement <8 x float> %1092, i64 1, !dbg !6
  %1095 = extractelement <8 x float> %1092, i64 2, !dbg !6
  %1096 = extractelement <8 x float> %1092, i64 3, !dbg !6
  %1097 = select i1 %.not, float %1094, float %1093, !dbg !6
  %1098 = bitcast float %1097 to i32, !dbg !6
  %1099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1098), !dbg !6
  %1100 = bitcast i32 %1099 to float, !dbg !6
  %1101 = select i1 %.not, float %1093, float %1100, !dbg !6
  %1102 = select i1 %.not, float %1100, float %1094, !dbg !6
  %1103 = select i1 %.not, float %1096, float %1095, !dbg !6
  %1104 = bitcast float %1103 to i32, !dbg !6
  %1105 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1104), !dbg !6
  %1106 = bitcast i32 %1105 to float, !dbg !6
  %1107 = select i1 %.not, float %1095, float %1106, !dbg !6
  %1108 = select i1 %.not, float %1106, float %1096, !dbg !6
  %1109 = select i1 %.not40, float %1107, float %1101, !dbg !6
  %1110 = bitcast float %1109 to i32, !dbg !6
  %1111 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1110), !dbg !6
  %1112 = bitcast i32 %1111 to float, !dbg !6
  %1113 = select i1 %.not40, float %1101, float %1112, !dbg !6
  %1114 = select i1 %.not40, float %1112, float %1107, !dbg !6
  %1115 = select i1 %.not40, float %1108, float %1102, !dbg !6
  %1116 = bitcast float %1115 to i32, !dbg !6
  %1117 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1116), !dbg !6
  %1118 = bitcast i32 %1117 to float, !dbg !6
  %1119 = select i1 %.not40, float %1102, float %1118, !dbg !6
  %1120 = select i1 %.not40, float %1118, float %1108, !dbg !6
  %1121 = extractelement <8 x float> %1092, i64 4, !dbg !6
  %1122 = extractelement <8 x float> %1092, i64 5, !dbg !6
  %1123 = extractelement <8 x float> %1092, i64 6, !dbg !6
  %1124 = extractelement <8 x float> %1092, i64 7, !dbg !6
  %1125 = select i1 %.not, float %1122, float %1121, !dbg !6
  %1126 = bitcast float %1125 to i32, !dbg !6
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1126), !dbg !6
  %1128 = bitcast i32 %1127 to float, !dbg !6
  %1129 = select i1 %.not, float %1121, float %1128, !dbg !6
  %1130 = select i1 %.not, float %1128, float %1122, !dbg !6
  %1131 = select i1 %.not, float %1124, float %1123, !dbg !6
  %1132 = bitcast float %1131 to i32, !dbg !6
  %1133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1132), !dbg !6
  %1134 = bitcast i32 %1133 to float, !dbg !6
  %1135 = select i1 %.not, float %1123, float %1134, !dbg !6
  %1136 = select i1 %.not, float %1134, float %1124, !dbg !6
  %1137 = select i1 %.not40, float %1135, float %1129, !dbg !6
  %1138 = bitcast float %1137 to i32, !dbg !6
  %1139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1138), !dbg !6
  %1140 = bitcast i32 %1139 to float, !dbg !6
  %1141 = select i1 %.not40, float %1129, float %1140, !dbg !6
  %1142 = select i1 %.not40, float %1140, float %1135, !dbg !6
  %1143 = select i1 %.not40, float %1136, float %1130, !dbg !6
  %1144 = bitcast float %1143 to i32, !dbg !6
  %1145 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1144), !dbg !6
  %1146 = bitcast i32 %1145 to float, !dbg !6
  %1147 = select i1 %.not40, float %1130, float %1146, !dbg !6
  %1148 = select i1 %.not40, float %1146, float %1136, !dbg !6
  %1149 = extractvalue [2 x <8 x float>] %106, 1, !dbg !6
  %1150 = extractelement <8 x float> %1149, i64 0, !dbg !6
  %1151 = extractelement <8 x float> %1149, i64 1, !dbg !6
  %1152 = extractelement <8 x float> %1149, i64 2, !dbg !6
  %1153 = extractelement <8 x float> %1149, i64 3, !dbg !6
  %1154 = select i1 %.not, float %1151, float %1150, !dbg !6
  %1155 = bitcast float %1154 to i32, !dbg !6
  %1156 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1155), !dbg !6
  %1157 = bitcast i32 %1156 to float, !dbg !6
  %1158 = select i1 %.not, float %1150, float %1157, !dbg !6
  %1159 = select i1 %.not, float %1157, float %1151, !dbg !6
  %1160 = select i1 %.not, float %1153, float %1152, !dbg !6
  %1161 = bitcast float %1160 to i32, !dbg !6
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1161), !dbg !6
  %1163 = bitcast i32 %1162 to float, !dbg !6
  %1164 = select i1 %.not, float %1152, float %1163, !dbg !6
  %1165 = select i1 %.not, float %1163, float %1153, !dbg !6
  %1166 = select i1 %.not40, float %1164, float %1158, !dbg !6
  %1167 = bitcast float %1166 to i32, !dbg !6
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1167), !dbg !6
  %1169 = bitcast i32 %1168 to float, !dbg !6
  %1170 = select i1 %.not40, float %1158, float %1169, !dbg !6
  %1171 = select i1 %.not40, float %1169, float %1164, !dbg !6
  %1172 = select i1 %.not40, float %1165, float %1159, !dbg !6
  %1173 = bitcast float %1172 to i32, !dbg !6
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1173), !dbg !6
  %1175 = bitcast i32 %1174 to float, !dbg !6
  %1176 = select i1 %.not40, float %1159, float %1175, !dbg !6
  %1177 = select i1 %.not40, float %1175, float %1165, !dbg !6
  %1178 = extractelement <8 x float> %1149, i64 4, !dbg !6
  %1179 = extractelement <8 x float> %1149, i64 5, !dbg !6
  %1180 = extractelement <8 x float> %1149, i64 6, !dbg !6
  %1181 = extractelement <8 x float> %1149, i64 7, !dbg !6
  %1182 = select i1 %.not, float %1179, float %1178, !dbg !6
  %1183 = bitcast float %1182 to i32, !dbg !6
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1183), !dbg !6
  %1185 = bitcast i32 %1184 to float, !dbg !6
  %1186 = select i1 %.not, float %1178, float %1185, !dbg !6
  %1187 = select i1 %.not, float %1185, float %1179, !dbg !6
  %1188 = select i1 %.not, float %1181, float %1180, !dbg !6
  %1189 = bitcast float %1188 to i32, !dbg !6
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1189), !dbg !6
  %1191 = bitcast i32 %1190 to float, !dbg !6
  %1192 = select i1 %.not, float %1180, float %1191, !dbg !6
  %1193 = select i1 %.not, float %1191, float %1181, !dbg !6
  %1194 = select i1 %.not40, float %1192, float %1186, !dbg !6
  %1195 = bitcast float %1194 to i32, !dbg !6
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1195), !dbg !6
  %1197 = bitcast i32 %1196 to float, !dbg !6
  %1198 = select i1 %.not40, float %1186, float %1197, !dbg !6
  %1199 = select i1 %.not40, float %1197, float %1192, !dbg !6
  %1200 = select i1 %.not40, float %1193, float %1187, !dbg !6
  %1201 = bitcast float %1200 to i32, !dbg !6
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1201), !dbg !6
  %1203 = bitcast i32 %1202 to float, !dbg !6
  %1204 = select i1 %.not40, float %1187, float %1203, !dbg !6
  %1205 = select i1 %.not40, float %1203, float %1193, !dbg !6
  %1206 = extractvalue [2 x <8 x float>] %105, 0, !dbg !6
  %1207 = extractelement <8 x float> %1206, i64 0, !dbg !6
  %1208 = extractelement <8 x float> %1206, i64 1, !dbg !6
  %1209 = extractelement <8 x float> %1206, i64 2, !dbg !6
  %1210 = extractelement <8 x float> %1206, i64 3, !dbg !6
  %1211 = select i1 %.not, float %1208, float %1207, !dbg !6
  %1212 = bitcast float %1211 to i32, !dbg !6
  %1213 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1212), !dbg !6
  %1214 = bitcast i32 %1213 to float, !dbg !6
  %1215 = select i1 %.not, float %1207, float %1214, !dbg !6
  %1216 = select i1 %.not, float %1214, float %1208, !dbg !6
  %1217 = select i1 %.not, float %1210, float %1209, !dbg !6
  %1218 = bitcast float %1217 to i32, !dbg !6
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1218), !dbg !6
  %1220 = bitcast i32 %1219 to float, !dbg !6
  %1221 = select i1 %.not, float %1209, float %1220, !dbg !6
  %1222 = select i1 %.not, float %1220, float %1210, !dbg !6
  %1223 = select i1 %.not40, float %1221, float %1215, !dbg !6
  %1224 = bitcast float %1223 to i32, !dbg !6
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1224), !dbg !6
  %1226 = bitcast i32 %1225 to float, !dbg !6
  %1227 = select i1 %.not40, float %1215, float %1226, !dbg !6
  %1228 = select i1 %.not40, float %1226, float %1221, !dbg !6
  %1229 = select i1 %.not40, float %1222, float %1216, !dbg !6
  %1230 = bitcast float %1229 to i32, !dbg !6
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1230), !dbg !6
  %1232 = bitcast i32 %1231 to float, !dbg !6
  %1233 = select i1 %.not40, float %1216, float %1232, !dbg !6
  %1234 = select i1 %.not40, float %1232, float %1222, !dbg !6
  %1235 = extractelement <8 x float> %1206, i64 4, !dbg !6
  %1236 = extractelement <8 x float> %1206, i64 5, !dbg !6
  %1237 = extractelement <8 x float> %1206, i64 6, !dbg !6
  %1238 = extractelement <8 x float> %1206, i64 7, !dbg !6
  %1239 = select i1 %.not, float %1236, float %1235, !dbg !6
  %1240 = bitcast float %1239 to i32, !dbg !6
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1240), !dbg !6
  %1242 = bitcast i32 %1241 to float, !dbg !6
  %1243 = select i1 %.not, float %1235, float %1242, !dbg !6
  %1244 = select i1 %.not, float %1242, float %1236, !dbg !6
  %1245 = select i1 %.not, float %1238, float %1237, !dbg !6
  %1246 = bitcast float %1245 to i32, !dbg !6
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1246), !dbg !6
  %1248 = bitcast i32 %1247 to float, !dbg !6
  %1249 = select i1 %.not, float %1237, float %1248, !dbg !6
  %1250 = select i1 %.not, float %1248, float %1238, !dbg !6
  %1251 = select i1 %.not40, float %1249, float %1243, !dbg !6
  %1252 = bitcast float %1251 to i32, !dbg !6
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1252), !dbg !6
  %1254 = bitcast i32 %1253 to float, !dbg !6
  %1255 = select i1 %.not40, float %1243, float %1254, !dbg !6
  %1256 = select i1 %.not40, float %1254, float %1249, !dbg !6
  %1257 = select i1 %.not40, float %1250, float %1244, !dbg !6
  %1258 = bitcast float %1257 to i32, !dbg !6
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1258), !dbg !6
  %1260 = bitcast i32 %1259 to float, !dbg !6
  %1261 = select i1 %.not40, float %1244, float %1260, !dbg !6
  %1262 = select i1 %.not40, float %1260, float %1250, !dbg !6
  %1263 = extractvalue [2 x <8 x float>] %105, 1, !dbg !6
  %1264 = extractelement <8 x float> %1263, i64 0, !dbg !6
  %1265 = extractelement <8 x float> %1263, i64 1, !dbg !6
  %1266 = extractelement <8 x float> %1263, i64 2, !dbg !6
  %1267 = extractelement <8 x float> %1263, i64 3, !dbg !6
  %1268 = select i1 %.not, float %1265, float %1264, !dbg !6
  %1269 = bitcast float %1268 to i32, !dbg !6
  %1270 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1269), !dbg !6
  %1271 = bitcast i32 %1270 to float, !dbg !6
  %1272 = select i1 %.not, float %1264, float %1271, !dbg !6
  %1273 = select i1 %.not, float %1271, float %1265, !dbg !6
  %1274 = select i1 %.not, float %1267, float %1266, !dbg !6
  %1275 = bitcast float %1274 to i32, !dbg !6
  %1276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1275), !dbg !6
  %1277 = bitcast i32 %1276 to float, !dbg !6
  %1278 = select i1 %.not, float %1266, float %1277, !dbg !6
  %1279 = select i1 %.not, float %1277, float %1267, !dbg !6
  %1280 = select i1 %.not40, float %1278, float %1272, !dbg !6
  %1281 = bitcast float %1280 to i32, !dbg !6
  %1282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1281), !dbg !6
  %1283 = bitcast i32 %1282 to float, !dbg !6
  %1284 = select i1 %.not40, float %1272, float %1283, !dbg !6
  %1285 = select i1 %.not40, float %1283, float %1278, !dbg !6
  %1286 = select i1 %.not40, float %1279, float %1273, !dbg !6
  %1287 = bitcast float %1286 to i32, !dbg !6
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1287), !dbg !6
  %1289 = bitcast i32 %1288 to float, !dbg !6
  %1290 = select i1 %.not40, float %1273, float %1289, !dbg !6
  %1291 = select i1 %.not40, float %1289, float %1279, !dbg !6
  %1292 = extractelement <8 x float> %1263, i64 4, !dbg !6
  %1293 = extractelement <8 x float> %1263, i64 5, !dbg !6
  %1294 = extractelement <8 x float> %1263, i64 6, !dbg !6
  %1295 = extractelement <8 x float> %1263, i64 7, !dbg !6
  %1296 = select i1 %.not, float %1293, float %1292, !dbg !6
  %1297 = bitcast float %1296 to i32, !dbg !6
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1297), !dbg !6
  %1299 = bitcast i32 %1298 to float, !dbg !6
  %1300 = select i1 %.not, float %1292, float %1299, !dbg !6
  %1301 = select i1 %.not, float %1299, float %1293, !dbg !6
  %1302 = select i1 %.not, float %1295, float %1294, !dbg !6
  %1303 = bitcast float %1302 to i32, !dbg !6
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1303), !dbg !6
  %1305 = bitcast i32 %1304 to float, !dbg !6
  %1306 = select i1 %.not, float %1294, float %1305, !dbg !6
  %1307 = select i1 %.not, float %1305, float %1295, !dbg !6
  %1308 = select i1 %.not40, float %1306, float %1300, !dbg !6
  %1309 = bitcast float %1308 to i32, !dbg !6
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1309), !dbg !6
  %1311 = bitcast i32 %1310 to float, !dbg !6
  %1312 = select i1 %.not40, float %1300, float %1311, !dbg !6
  %1313 = select i1 %.not40, float %1311, float %1306, !dbg !6
  %1314 = select i1 %.not40, float %1307, float %1301, !dbg !6
  %1315 = bitcast float %1314 to i32, !dbg !6
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1315), !dbg !6
  %1317 = bitcast i32 %1316 to float, !dbg !6
  %1318 = select i1 %.not40, float %1301, float %1317, !dbg !6
  %1319 = select i1 %.not40, float %1317, float %1307, !dbg !6
  %1320 = extractvalue [2 x <8 x float>] %102, 0, !dbg !6
  %1321 = extractelement <8 x float> %1320, i64 0, !dbg !6
  %1322 = extractelement <8 x float> %1320, i64 1, !dbg !6
  %1323 = extractelement <8 x float> %1320, i64 2, !dbg !6
  %1324 = extractelement <8 x float> %1320, i64 3, !dbg !6
  %1325 = select i1 %.not, float %1322, float %1321, !dbg !6
  %1326 = bitcast float %1325 to i32, !dbg !6
  %1327 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1326), !dbg !6
  %1328 = bitcast i32 %1327 to float, !dbg !6
  %1329 = select i1 %.not, float %1321, float %1328, !dbg !6
  %1330 = select i1 %.not, float %1328, float %1322, !dbg !6
  %1331 = select i1 %.not, float %1324, float %1323, !dbg !6
  %1332 = bitcast float %1331 to i32, !dbg !6
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1332), !dbg !6
  %1334 = bitcast i32 %1333 to float, !dbg !6
  %1335 = select i1 %.not, float %1323, float %1334, !dbg !6
  %1336 = select i1 %.not, float %1334, float %1324, !dbg !6
  %1337 = select i1 %.not40, float %1335, float %1329, !dbg !6
  %1338 = bitcast float %1337 to i32, !dbg !6
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1338), !dbg !6
  %1340 = select i1 %.not40, float %1336, float %1330, !dbg !6
  %1341 = bitcast float %1340 to i32, !dbg !6
  %1342 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1341), !dbg !6
  %1343 = extractelement <8 x float> %1320, i64 4, !dbg !6
  %1344 = extractelement <8 x float> %1320, i64 5, !dbg !6
  %1345 = extractelement <8 x float> %1320, i64 6, !dbg !6
  %1346 = extractelement <8 x float> %1320, i64 7, !dbg !6
  %1347 = select i1 %.not, float %1344, float %1343, !dbg !6
  %1348 = bitcast float %1347 to i32, !dbg !6
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1348), !dbg !6
  %1350 = bitcast i32 %1349 to float, !dbg !6
  %1351 = select i1 %.not, float %1343, float %1350, !dbg !6
  %1352 = select i1 %.not, float %1350, float %1344, !dbg !6
  %1353 = select i1 %.not, float %1346, float %1345, !dbg !6
  %1354 = bitcast float %1353 to i32, !dbg !6
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1354), !dbg !6
  %1356 = bitcast i32 %1355 to float, !dbg !6
  %1357 = select i1 %.not, float %1345, float %1356, !dbg !6
  %1358 = select i1 %.not, float %1356, float %1346, !dbg !6
  %1359 = select i1 %.not40, float %1357, float %1351, !dbg !6
  %1360 = bitcast float %1359 to i32, !dbg !6
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1360), !dbg !6
  %1362 = select i1 %.not40, float %1358, float %1352, !dbg !6
  %1363 = bitcast float %1362 to i32, !dbg !6
  %1364 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1363), !dbg !6
  %1365 = extractvalue [2 x <8 x float>] %102, 1, !dbg !6
  %1366 = extractelement <8 x float> %1365, i64 0, !dbg !6
  %1367 = extractelement <8 x float> %1365, i64 1, !dbg !6
  %1368 = extractelement <8 x float> %1365, i64 2, !dbg !6
  %1369 = extractelement <8 x float> %1365, i64 3, !dbg !6
  %1370 = select i1 %.not, float %1367, float %1366, !dbg !6
  %1371 = bitcast float %1370 to i32, !dbg !6
  %1372 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1371), !dbg !6
  %1373 = bitcast i32 %1372 to float, !dbg !6
  %1374 = select i1 %.not, float %1366, float %1373, !dbg !6
  %1375 = select i1 %.not, float %1373, float %1367, !dbg !6
  %1376 = select i1 %.not, float %1369, float %1368, !dbg !6
  %1377 = bitcast float %1376 to i32, !dbg !6
  %1378 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1377), !dbg !6
  %1379 = bitcast i32 %1378 to float, !dbg !6
  %1380 = select i1 %.not, float %1368, float %1379, !dbg !6
  %1381 = select i1 %.not, float %1379, float %1369, !dbg !6
  %1382 = select i1 %.not40, float %1380, float %1374, !dbg !6
  %1383 = bitcast float %1382 to i32, !dbg !6
  %1384 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1383), !dbg !6
  %1385 = select i1 %.not40, float %1381, float %1375, !dbg !6
  %1386 = bitcast float %1385 to i32, !dbg !6
  %1387 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1386), !dbg !6
  %1388 = extractelement <8 x float> %1365, i64 4, !dbg !6
  %1389 = extractelement <8 x float> %1365, i64 5, !dbg !6
  %1390 = extractelement <8 x float> %1365, i64 6, !dbg !6
  %1391 = extractelement <8 x float> %1365, i64 7, !dbg !6
  %1392 = select i1 %.not, float %1389, float %1388, !dbg !6
  %1393 = bitcast float %1392 to i32, !dbg !6
  %1394 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1393), !dbg !6
  %1395 = bitcast i32 %1394 to float, !dbg !6
  %1396 = select i1 %.not, float %1388, float %1395, !dbg !6
  %1397 = select i1 %.not, float %1395, float %1389, !dbg !6
  %1398 = select i1 %.not, float %1391, float %1390, !dbg !6
  %1399 = bitcast float %1398 to i32, !dbg !6
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1399), !dbg !6
  %1401 = bitcast i32 %1400 to float, !dbg !6
  %1402 = select i1 %.not, float %1390, float %1401, !dbg !6
  %1403 = select i1 %.not, float %1401, float %1391, !dbg !6
  %1404 = select i1 %.not40, float %1402, float %1396, !dbg !6
  %1405 = bitcast float %1404 to i32, !dbg !6
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1405), !dbg !6
  %1407 = select i1 %.not40, float %1403, float %1397, !dbg !6
  %1408 = bitcast float %1407 to i32, !dbg !6
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1408), !dbg !6
  %1410 = shufflevector <1 x float> %975, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1411 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1410, !dbg !17
  %1412 = shufflevector <1 x float> %970, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1413 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1412, !dbg !17
  %1414 = bitcast i32 %1406 to float, !dbg !6
  %1415 = select i1 %.not40, float %1396, float %1414, !dbg !6
  %1416 = insertelement <4 x float> poison, float %1415, i64 0, !dbg !17
  %1417 = bitcast i32 %1409 to float, !dbg !6
  %1418 = select i1 %.not40, float %1397, float %1417, !dbg !6
  %1419 = insertelement <4 x float> %1416, float %1418, i64 1, !dbg !17
  %1420 = select i1 %.not40, float %1414, float %1402, !dbg !6
  %1421 = insertelement <4 x float> %1419, float %1420, i64 2, !dbg !17
  %1422 = select i1 %.not40, float %1417, float %1403, !dbg !6
  %1423 = insertelement <4 x float> %1421, float %1422, i64 3, !dbg !17
  %1424 = fmul <4 x float> %1423, %1413, !dbg !17
  %1425 = fptrunc <4 x float> %1424 to <4 x half>, !dbg !18
  %1426 = extractelement <4 x half> %1425, i64 3, !dbg !18
  %1427 = extractelement <4 x half> %1425, i64 2, !dbg !18
  %1428 = extractelement <4 x half> %1425, i64 1, !dbg !18
  %1429 = extractelement <4 x half> %1425, i64 0, !dbg !18
  %1430 = bitcast i32 %1384 to float, !dbg !6
  %1431 = select i1 %.not40, float %1374, float %1430, !dbg !6
  %1432 = insertelement <4 x float> poison, float %1431, i64 0, !dbg !17
  %1433 = bitcast i32 %1387 to float, !dbg !6
  %1434 = select i1 %.not40, float %1375, float %1433, !dbg !6
  %1435 = insertelement <4 x float> %1432, float %1434, i64 1, !dbg !17
  %1436 = select i1 %.not40, float %1430, float %1380, !dbg !6
  %1437 = insertelement <4 x float> %1435, float %1436, i64 2, !dbg !17
  %1438 = select i1 %.not40, float %1433, float %1381, !dbg !6
  %1439 = insertelement <4 x float> %1437, float %1438, i64 3, !dbg !17
  %1440 = fmul <4 x float> %1439, %1413, !dbg !17
  %1441 = fptrunc <4 x float> %1440 to <4 x half>, !dbg !18
  %1442 = extractelement <4 x half> %1441, i64 3, !dbg !18
  %1443 = extractelement <4 x half> %1441, i64 2, !dbg !18
  %1444 = extractelement <4 x half> %1441, i64 1, !dbg !18
  %1445 = extractelement <4 x half> %1441, i64 0, !dbg !18
  %1446 = bitcast i32 %1361 to float, !dbg !6
  %1447 = select i1 %.not40, float %1351, float %1446, !dbg !6
  %1448 = insertelement <4 x float> poison, float %1447, i64 0, !dbg !17
  %1449 = bitcast i32 %1364 to float, !dbg !6
  %1450 = select i1 %.not40, float %1352, float %1449, !dbg !6
  %1451 = insertelement <4 x float> %1448, float %1450, i64 1, !dbg !17
  %1452 = select i1 %.not40, float %1446, float %1357, !dbg !6
  %1453 = insertelement <4 x float> %1451, float %1452, i64 2, !dbg !17
  %1454 = select i1 %.not40, float %1449, float %1358, !dbg !6
  %1455 = insertelement <4 x float> %1453, float %1454, i64 3, !dbg !17
  %1456 = fmul <4 x float> %1455, %1411, !dbg !17
  %1457 = fptrunc <4 x float> %1456 to <4 x half>, !dbg !18
  %1458 = extractelement <4 x half> %1457, i64 3, !dbg !18
  %1459 = extractelement <4 x half> %1457, i64 2, !dbg !18
  %1460 = extractelement <4 x half> %1457, i64 1, !dbg !18
  %1461 = extractelement <4 x half> %1457, i64 0, !dbg !18
  %1462 = bitcast i32 %1339 to float, !dbg !6
  %1463 = select i1 %.not40, float %1329, float %1462, !dbg !6
  %1464 = insertelement <4 x float> poison, float %1463, i64 0, !dbg !17
  %1465 = bitcast i32 %1342 to float, !dbg !6
  %1466 = select i1 %.not40, float %1330, float %1465, !dbg !6
  %1467 = insertelement <4 x float> %1464, float %1466, i64 1, !dbg !17
  %1468 = select i1 %.not40, float %1462, float %1335, !dbg !6
  %1469 = insertelement <4 x float> %1467, float %1468, i64 2, !dbg !17
  %1470 = select i1 %.not40, float %1465, float %1336, !dbg !6
  %1471 = insertelement <4 x float> %1469, float %1470, i64 3, !dbg !17
  %1472 = fmul <4 x float> %1471, %1411, !dbg !17
  %1473 = fptrunc <4 x float> %1472 to <4 x half>, !dbg !18
  %1474 = extractelement <4 x half> %1473, i64 3, !dbg !18
  %1475 = extractelement <4 x half> %1473, i64 2, !dbg !18
  %1476 = extractelement <4 x half> %1473, i64 1, !dbg !18
  %1477 = extractelement <4 x half> %1473, i64 0, !dbg !18
  %address.thread971 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local972 = zext nneg i32 %address.thread971 to i64, !dbg !9
  %address.local973 = add nuw i64 %32, %address.local972, !dbg !19
  %1478 = lshr i64 %address.local972, 2, !dbg !19
  %address.local975 = and i64 %1478, 48, !dbg !19
  %address.local976 = add i64 %address.local973, %address.local975, !dbg !19
  %address.local977 = and i64 %address.local972, 240, !dbg !19
  %address.local978 = sub i64 %address.local976, %address.local977, !dbg !19
  %address.local979 = add i64 %address.local978, %33, !dbg !19
  %address.local980 = shl nuw nsw i64 %address.local979, 7, !dbg !19
  %1479 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local982 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1479, i64 %address.local980, !dbg !19
  %address.local984 = and i64 %1478, 12, !dbg !12
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local984, !dbg !19
  store half %1477, ptr addrspace(1) %address.local985, align 2, !dbg !19
  %address.local987 = or disjoint i64 %address.local984, 1, !dbg !19
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local987, !dbg !19
  store half %1476, ptr addrspace(1) %address.local988, align 2, !dbg !19
  %address.local990 = or disjoint i64 %address.local984, 2, !dbg !19
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local990, !dbg !19
  store half %1475, ptr addrspace(1) %address.local991, align 2, !dbg !19
  %address.local993 = or disjoint i64 %address.local984, 3, !dbg !19
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local993, !dbg !19
  store half %1474, ptr addrspace(1) %address.local994, align 2, !dbg !19
  %address.local996 = or disjoint i64 %address.local984, 16, !dbg !19
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local996, !dbg !19
  store half %1461, ptr addrspace(1) %address.local997, align 2, !dbg !19
  %address.local999 = or disjoint i64 %address.local984, 17, !dbg !19
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local999, !dbg !19
  store half %1460, ptr addrspace(1) %address.local1000, align 2, !dbg !19
  %address.local1002 = or disjoint i64 %address.local984, 18, !dbg !19
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1002, !dbg !19
  store half %1459, ptr addrspace(1) %address.local1003, align 2, !dbg !19
  %address.local1005 = or disjoint i64 %address.local984, 19, !dbg !19
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1005, !dbg !19
  store half %1458, ptr addrspace(1) %address.local1006, align 2, !dbg !19
  %address.local1010 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local982, i64 16384, !dbg !19
  %address.local1011 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local984, !dbg !19
  store half %1445, ptr addrspace(1) %address.local1011, align 2, !dbg !19
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local987, !dbg !19
  store half %1444, ptr addrspace(1) %address.local1013, align 2, !dbg !19
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local990, !dbg !19
  store half %1443, ptr addrspace(1) %address.local1015, align 2, !dbg !19
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local993, !dbg !19
  store half %1442, ptr addrspace(1) %address.local1017, align 2, !dbg !19
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local996, !dbg !19
  store half %1429, ptr addrspace(1) %address.local1019, align 2, !dbg !19
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local999, !dbg !19
  store half %1428, ptr addrspace(1) %address.local1021, align 2, !dbg !19
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1002, !dbg !19
  store half %1427, ptr addrspace(1) %address.local1023, align 2, !dbg !19
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1005, !dbg !19
  store half %1426, ptr addrspace(1) %address.local1025, align 2, !dbg !19
  %1480 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !17
  %1481 = insertelement <4 x float> %1480, float %1318, i64 1, !dbg !17
  %1482 = insertelement <4 x float> %1481, float %1313, i64 2, !dbg !17
  %1483 = insertelement <4 x float> %1482, float %1319, i64 3, !dbg !17
  %1484 = fmul <4 x float> %1483, %1413, !dbg !17
  %1485 = fptrunc <4 x float> %1484 to <4 x half>, !dbg !18
  %1486 = extractelement <4 x half> %1485, i64 3, !dbg !18
  %1487 = extractelement <4 x half> %1485, i64 2, !dbg !18
  %1488 = extractelement <4 x half> %1485, i64 1, !dbg !18
  %1489 = extractelement <4 x half> %1485, i64 0, !dbg !18
  %1490 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !17
  %1491 = insertelement <4 x float> %1490, float %1290, i64 1, !dbg !17
  %1492 = insertelement <4 x float> %1491, float %1285, i64 2, !dbg !17
  %1493 = insertelement <4 x float> %1492, float %1291, i64 3, !dbg !17
  %1494 = fmul <4 x float> %1493, %1413, !dbg !17
  %1495 = fptrunc <4 x float> %1494 to <4 x half>, !dbg !18
  %1496 = extractelement <4 x half> %1495, i64 3, !dbg !18
  %1497 = extractelement <4 x half> %1495, i64 2, !dbg !18
  %1498 = extractelement <4 x half> %1495, i64 1, !dbg !18
  %1499 = extractelement <4 x half> %1495, i64 0, !dbg !18
  %1500 = insertelement <4 x float> poison, float %1255, i64 0, !dbg !17
  %1501 = insertelement <4 x float> %1500, float %1261, i64 1, !dbg !17
  %1502 = insertelement <4 x float> %1501, float %1256, i64 2, !dbg !17
  %1503 = insertelement <4 x float> %1502, float %1262, i64 3, !dbg !17
  %1504 = fmul <4 x float> %1503, %1411, !dbg !17
  %1505 = fptrunc <4 x float> %1504 to <4 x half>, !dbg !18
  %1506 = extractelement <4 x half> %1505, i64 3, !dbg !18
  %1507 = extractelement <4 x half> %1505, i64 2, !dbg !18
  %1508 = extractelement <4 x half> %1505, i64 1, !dbg !18
  %1509 = extractelement <4 x half> %1505, i64 0, !dbg !18
  %1510 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !17
  %1511 = insertelement <4 x float> %1510, float %1233, i64 1, !dbg !17
  %1512 = insertelement <4 x float> %1511, float %1228, i64 2, !dbg !17
  %1513 = insertelement <4 x float> %1512, float %1234, i64 3, !dbg !17
  %1514 = fmul <4 x float> %1513, %1411, !dbg !17
  %1515 = fptrunc <4 x float> %1514 to <4 x half>, !dbg !18
  %1516 = extractelement <4 x half> %1515, i64 3, !dbg !18
  %1517 = extractelement <4 x half> %1515, i64 2, !dbg !18
  %1518 = extractelement <4 x half> %1515, i64 1, !dbg !18
  %1519 = extractelement <4 x half> %1515, i64 0, !dbg !18
  %address.local1040 = or disjoint i64 %address.local984, 32, !dbg !19
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1040, !dbg !19
  store half %1519, ptr addrspace(1) %address.local1041, align 2, !dbg !19
  %address.local1043 = or disjoint i64 %address.local984, 33, !dbg !19
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1043, !dbg !19
  store half %1518, ptr addrspace(1) %address.local1044, align 2, !dbg !19
  %address.local1046 = or disjoint i64 %address.local984, 34, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1046, !dbg !19
  store half %1517, ptr addrspace(1) %address.local1047, align 2, !dbg !19
  %address.local1049 = or disjoint i64 %address.local984, 35, !dbg !19
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1049, !dbg !19
  store half %1516, ptr addrspace(1) %address.local1050, align 2, !dbg !19
  %address.local1052 = or disjoint i64 %address.local984, 48, !dbg !19
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1052, !dbg !19
  store half %1509, ptr addrspace(1) %address.local1053, align 2, !dbg !19
  %address.local1055 = or disjoint i64 %address.local984, 49, !dbg !19
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1055, !dbg !19
  store half %1508, ptr addrspace(1) %address.local1056, align 2, !dbg !19
  %address.local1058 = or disjoint i64 %address.local984, 50, !dbg !19
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1058, !dbg !19
  store half %1507, ptr addrspace(1) %address.local1059, align 2, !dbg !19
  %address.local1061 = or i64 %1478, 51, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1061, !dbg !19
  store half %1506, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1067 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1040, !dbg !19
  store half %1499, ptr addrspace(1) %address.local1067, align 2, !dbg !19
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1043, !dbg !19
  store half %1498, ptr addrspace(1) %address.local1069, align 2, !dbg !19
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1046, !dbg !19
  store half %1497, ptr addrspace(1) %address.local1071, align 2, !dbg !19
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1049, !dbg !19
  store half %1496, ptr addrspace(1) %address.local1073, align 2, !dbg !19
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1052, !dbg !19
  store half %1489, ptr addrspace(1) %address.local1075, align 2, !dbg !19
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1055, !dbg !19
  store half %1488, ptr addrspace(1) %address.local1077, align 2, !dbg !19
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1058, !dbg !19
  store half %1487, ptr addrspace(1) %address.local1079, align 2, !dbg !19
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1061, !dbg !19
  store half %1486, ptr addrspace(1) %address.local1081, align 2, !dbg !19
  %1520 = insertelement <4 x float> poison, float %1198, i64 0, !dbg !17
  %1521 = insertelement <4 x float> %1520, float %1204, i64 1, !dbg !17
  %1522 = insertelement <4 x float> %1521, float %1199, i64 2, !dbg !17
  %1523 = insertelement <4 x float> %1522, float %1205, i64 3, !dbg !17
  %1524 = fmul <4 x float> %1523, %1413, !dbg !17
  %1525 = fptrunc <4 x float> %1524 to <4 x half>, !dbg !18
  %1526 = extractelement <4 x half> %1525, i64 3, !dbg !18
  %1527 = extractelement <4 x half> %1525, i64 2, !dbg !18
  %1528 = extractelement <4 x half> %1525, i64 1, !dbg !18
  %1529 = extractelement <4 x half> %1525, i64 0, !dbg !18
  %1530 = insertelement <4 x float> poison, float %1170, i64 0, !dbg !17
  %1531 = insertelement <4 x float> %1530, float %1176, i64 1, !dbg !17
  %1532 = insertelement <4 x float> %1531, float %1171, i64 2, !dbg !17
  %1533 = insertelement <4 x float> %1532, float %1177, i64 3, !dbg !17
  %1534 = fmul <4 x float> %1533, %1413, !dbg !17
  %1535 = fptrunc <4 x float> %1534 to <4 x half>, !dbg !18
  %1536 = extractelement <4 x half> %1535, i64 3, !dbg !18
  %1537 = extractelement <4 x half> %1535, i64 2, !dbg !18
  %1538 = extractelement <4 x half> %1535, i64 1, !dbg !18
  %1539 = extractelement <4 x half> %1535, i64 0, !dbg !18
  %1540 = insertelement <4 x float> poison, float %1141, i64 0, !dbg !17
  %1541 = insertelement <4 x float> %1540, float %1147, i64 1, !dbg !17
  %1542 = insertelement <4 x float> %1541, float %1142, i64 2, !dbg !17
  %1543 = insertelement <4 x float> %1542, float %1148, i64 3, !dbg !17
  %1544 = fmul <4 x float> %1543, %1411, !dbg !17
  %1545 = fptrunc <4 x float> %1544 to <4 x half>, !dbg !18
  %1546 = extractelement <4 x half> %1545, i64 3, !dbg !18
  %1547 = extractelement <4 x half> %1545, i64 2, !dbg !18
  %1548 = extractelement <4 x half> %1545, i64 1, !dbg !18
  %1549 = extractelement <4 x half> %1545, i64 0, !dbg !18
  %1550 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !17
  %1551 = insertelement <4 x float> %1550, float %1119, i64 1, !dbg !17
  %1552 = insertelement <4 x float> %1551, float %1114, i64 2, !dbg !17
  %1553 = insertelement <4 x float> %1552, float %1120, i64 3, !dbg !17
  %1554 = fmul <4 x float> %1553, %1411, !dbg !17
  %1555 = fptrunc <4 x float> %1554 to <4 x half>, !dbg !18
  %1556 = extractelement <4 x half> %1555, i64 3, !dbg !18
  %1557 = extractelement <4 x half> %1555, i64 2, !dbg !18
  %1558 = extractelement <4 x half> %1555, i64 1, !dbg !18
  %1559 = extractelement <4 x half> %1555, i64 0, !dbg !18
  %address.local1096 = or disjoint i64 %address.local984, 64, !dbg !19
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1096, !dbg !19
  store half %1559, ptr addrspace(1) %address.local1097, align 2, !dbg !19
  %address.local1099 = or disjoint i64 %address.local984, 65, !dbg !19
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1099, !dbg !19
  store half %1558, ptr addrspace(1) %address.local1100, align 2, !dbg !19
  %address.local1102 = or disjoint i64 %address.local984, 66, !dbg !19
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1102, !dbg !19
  store half %1557, ptr addrspace(1) %address.local1103, align 2, !dbg !19
  %address.local1105 = or disjoint i64 %address.local984, 67, !dbg !19
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1105, !dbg !19
  store half %1556, ptr addrspace(1) %address.local1106, align 2, !dbg !19
  %address.local1108 = or disjoint i64 %address.local984, 80, !dbg !19
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1108, !dbg !19
  store half %1549, ptr addrspace(1) %address.local1109, align 2, !dbg !19
  %address.local1111 = or disjoint i64 %address.local984, 81, !dbg !19
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1111, !dbg !19
  store half %1548, ptr addrspace(1) %address.local1112, align 2, !dbg !19
  %address.local1114 = or disjoint i64 %address.local984, 82, !dbg !19
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1114, !dbg !19
  store half %1547, ptr addrspace(1) %address.local1115, align 2, !dbg !19
  %address.local1117 = or disjoint i64 %address.local984, 83, !dbg !19
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1117, !dbg !19
  store half %1546, ptr addrspace(1) %address.local1118, align 2, !dbg !19
  %address.local1123 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1096, !dbg !19
  store half %1539, ptr addrspace(1) %address.local1123, align 2, !dbg !19
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1099, !dbg !19
  store half %1538, ptr addrspace(1) %address.local1125, align 2, !dbg !19
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1102, !dbg !19
  store half %1537, ptr addrspace(1) %address.local1127, align 2, !dbg !19
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1105, !dbg !19
  store half %1536, ptr addrspace(1) %address.local1129, align 2, !dbg !19
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1108, !dbg !19
  store half %1529, ptr addrspace(1) %address.local1131, align 2, !dbg !19
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1111, !dbg !19
  store half %1528, ptr addrspace(1) %address.local1133, align 2, !dbg !19
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1114, !dbg !19
  store half %1527, ptr addrspace(1) %address.local1135, align 2, !dbg !19
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1117, !dbg !19
  store half %1526, ptr addrspace(1) %address.local1137, align 2, !dbg !19
  %1560 = insertelement <4 x float> poison, float %1084, i64 0, !dbg !17
  %1561 = insertelement <4 x float> %1560, float %1090, i64 1, !dbg !17
  %1562 = insertelement <4 x float> %1561, float %1085, i64 2, !dbg !17
  %1563 = insertelement <4 x float> %1562, float %1091, i64 3, !dbg !17
  %1564 = fmul <4 x float> %1563, %1413, !dbg !17
  %1565 = fptrunc <4 x float> %1564 to <4 x half>, !dbg !18
  %1566 = extractelement <4 x half> %1565, i64 3, !dbg !18
  %1567 = extractelement <4 x half> %1565, i64 2, !dbg !18
  %1568 = extractelement <4 x half> %1565, i64 1, !dbg !18
  %1569 = extractelement <4 x half> %1565, i64 0, !dbg !18
  %1570 = insertelement <4 x float> poison, float %1056, i64 0, !dbg !17
  %1571 = insertelement <4 x float> %1570, float %1062, i64 1, !dbg !17
  %1572 = insertelement <4 x float> %1571, float %1057, i64 2, !dbg !17
  %1573 = insertelement <4 x float> %1572, float %1063, i64 3, !dbg !17
  %1574 = fmul <4 x float> %1573, %1413, !dbg !17
  %1575 = fptrunc <4 x float> %1574 to <4 x half>, !dbg !18
  %1576 = extractelement <4 x half> %1575, i64 3, !dbg !18
  %1577 = extractelement <4 x half> %1575, i64 2, !dbg !18
  %1578 = extractelement <4 x half> %1575, i64 1, !dbg !18
  %1579 = extractelement <4 x half> %1575, i64 0, !dbg !18
  %1580 = insertelement <4 x float> poison, float %1027, i64 0, !dbg !17
  %1581 = insertelement <4 x float> %1580, float %1033, i64 1, !dbg !17
  %1582 = insertelement <4 x float> %1581, float %1028, i64 2, !dbg !17
  %1583 = insertelement <4 x float> %1582, float %1034, i64 3, !dbg !17
  %1584 = fmul <4 x float> %1583, %1411, !dbg !17
  %1585 = fptrunc <4 x float> %1584 to <4 x half>, !dbg !18
  %1586 = extractelement <4 x half> %1585, i64 3, !dbg !18
  %1587 = extractelement <4 x half> %1585, i64 2, !dbg !18
  %1588 = extractelement <4 x half> %1585, i64 1, !dbg !18
  %1589 = extractelement <4 x half> %1585, i64 0, !dbg !18
  %1590 = insertelement <4 x float> poison, float %999, i64 0, !dbg !17
  %1591 = insertelement <4 x float> %1590, float %1005, i64 1, !dbg !17
  %1592 = insertelement <4 x float> %1591, float %1000, i64 2, !dbg !17
  %1593 = insertelement <4 x float> %1592, float %1006, i64 3, !dbg !17
  %1594 = fmul <4 x float> %1593, %1411, !dbg !17
  %1595 = fptrunc <4 x float> %1594 to <4 x half>, !dbg !18
  %1596 = extractelement <4 x half> %1595, i64 3, !dbg !18
  %1597 = extractelement <4 x half> %1595, i64 2, !dbg !18
  %1598 = extractelement <4 x half> %1595, i64 1, !dbg !18
  %1599 = extractelement <4 x half> %1595, i64 0, !dbg !18
  %address.local1152 = or disjoint i64 %address.local984, 96, !dbg !19
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1152, !dbg !19
  store half %1599, ptr addrspace(1) %address.local1153, align 2, !dbg !19
  %address.local1155 = or disjoint i64 %address.local984, 97, !dbg !19
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1155, !dbg !19
  store half %1598, ptr addrspace(1) %address.local1156, align 2, !dbg !19
  %address.local1158 = or disjoint i64 %address.local984, 98, !dbg !19
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1158, !dbg !19
  store half %1597, ptr addrspace(1) %address.local1159, align 2, !dbg !19
  %address.local1161 = or disjoint i64 %address.local984, 99, !dbg !19
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1161, !dbg !19
  store half %1596, ptr addrspace(1) %address.local1162, align 2, !dbg !19
  %address.local1164 = or disjoint i64 %address.local984, 112, !dbg !19
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1164, !dbg !19
  store half %1589, ptr addrspace(1) %address.local1165, align 2, !dbg !19
  %address.local1167 = or disjoint i64 %address.local984, 113, !dbg !19
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1167, !dbg !19
  store half %1588, ptr addrspace(1) %address.local1168, align 2, !dbg !19
  %address.local1170 = or disjoint i64 %address.local984, 114, !dbg !19
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1170, !dbg !19
  store half %1587, ptr addrspace(1) %address.local1171, align 2, !dbg !19
  %address.local1173 = or i64 %1478, 115, !dbg !19
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local982, i64 %address.local1173, !dbg !19
  store half %1586, ptr addrspace(1) %address.local1174, align 2, !dbg !19
  %address.local1179 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1152, !dbg !19
  store half %1579, ptr addrspace(1) %address.local1179, align 2, !dbg !19
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1155, !dbg !19
  store half %1578, ptr addrspace(1) %address.local1181, align 2, !dbg !19
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1158, !dbg !19
  store half %1577, ptr addrspace(1) %address.local1183, align 2, !dbg !19
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1161, !dbg !19
  store half %1576, ptr addrspace(1) %address.local1185, align 2, !dbg !19
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1164, !dbg !19
  store half %1569, ptr addrspace(1) %address.local1187, align 2, !dbg !19
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1167, !dbg !19
  store half %1568, ptr addrspace(1) %address.local1189, align 2, !dbg !19
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1170, !dbg !19
  store half %1567, ptr addrspace(1) %address.local1191, align 2, !dbg !19
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1010, i64 %address.local1173, !dbg !19
  store half %1566, ptr addrspace(1) %address.local1193, align 2, !dbg !19
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

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i64 @llvm.amdgcn.ballot.i64(i1) #4

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.lazy_softmax_base"="v1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_cvt_pack"="rtz" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math")
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
