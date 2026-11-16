; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2/input.mlir"

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
  %address.local420 = zext nneg i32 %37 to i64, !dbg !9
  %address.local421 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local420, !dbg !9
  %address.local422 = ptrtoint ptr addrspace(3) %address.local421 to i64, !dbg !9
  %address.local423 = trunc i64 %address.local422 to i32, !dbg !9
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local423), !dbg !9
  %address.local424 = lshr i32 %address.thread, 4, !dbg !9
  %address.local425 = and i32 %address.local424, 8, !dbg !9
  %address.local426 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local427 = and i32 %address.local426, 16256, !dbg !9
  %address.local428 = or disjoint i32 %address.local425, %address.local427, !dbg !9
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1394 = shl i64 %34, 8, !dbg !9
  %40 = add i64 %.idx, %.idx1394, !dbg !9
  %41 = shl nuw nsw i32 %address.local428, 1, !dbg !9
  %42 = trunc i64 %40 to i32, !dbg !9
  %dma.buffer.offset1392 = or disjoint i32 %41, %42, !dbg !9
  %43 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !9
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1393 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1392, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %address.local433 = or disjoint i64 %address.local420, 2048, !dbg !9
  %address.local434 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local433, !dbg !9
  %address.local435 = ptrtoint ptr addrspace(3) %address.local434 to i64, !dbg !9
  %address.local436 = trunc i64 %address.local435 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local436), !dbg !9
  %address.local442 = or disjoint i32 %address.local428, 16, !dbg !9
  %47 = shl nuw nsw i32 %address.local442, 1, !dbg !9
  %dma.buffer.offset1390 = or disjoint i32 %47, %42, !dbg !9
  %48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %49 = shl nuw nsw i64 %31, 19, !dbg !9
  %50 = add i64 %49, %35, !dbg !9
  %address.local445 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local420, !dbg !9
  %address.local446 = ptrtoint ptr addrspace(3) %address.local445 to i64, !dbg !9
  %address.local447 = trunc i64 %address.local446 to i32, !dbg !9
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local447), !dbg !9
  %52 = trunc i64 %50 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local428, %52, !dbg !9
  %53 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1388 = or disjoint i32 %53, 64, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1388, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %address.local449 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local433, !dbg !9
  %address.local450 = ptrtoint ptr addrspace(3) %address.local449 to i64, !dbg !9
  %address.local451 = trunc i64 %address.local450 to i32, !dbg !9
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local451), !dbg !9
  %.tr1398 = or disjoint i32 %address.local442, %52, !dbg !9
  %56 = shl i32 %.tr1398, 1, !dbg !9
  %dma.buffer.offset1386 = or disjoint i32 %56, 64, !dbg !9
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1386, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %address.local453 = lshr i32 %address.thread, 2, !dbg !12
  %address.local454 = and i32 %address.local453, 4, !dbg !12
  %address.local455 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local456 = and i32 %address.local455, 384, !dbg !12
  %address.local457 = or disjoint i32 %address.local454, %address.local456, !dbg !12
  %address.local459 = and i32 %36, 120, !dbg !12
  %address.local460 = or disjoint i32 %address.local457, %address.local459, !dbg !12
  %address.local461 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local462 = and i32 %address.local461, 1024, !dbg !12
  %address.local463 = or disjoint i32 %address.local460, %address.local462, !dbg !12
  %address.local464 = zext nneg i32 %address.local463 to i64, !dbg !12
  %address.local465 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local465, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local465, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %address.local465, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local465, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1384 = or disjoint i32 %53, 128, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %dma.buffer.offset1382 = or disjoint i32 %56, 128, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1382, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local465, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %address.local465, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %address.local465, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local465, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1380 = or disjoint i32 %53, 192, !dbg !9
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1380, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %dma.buffer.offset1378 = or disjoint i32 %56, 192, !dbg !9
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1393) #10, !dbg !9
  %78 = load <4 x half>, ptr addrspace(3) %address.local465, align 8, !dbg !12
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !12
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %83 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !12
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !12
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
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
  %k0.prime.2061 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.2062 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.2063 = or disjoint i64 %k0.prime.2061, %k0.prime.2062, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1500 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1501 = shl nuw nsw i64 %k0.prime.kdma.source1500, 9, !dbg !13
  %k0.prime.kdma.source1502 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !13
  %k0.prime.kdma.source1503 = add nuw nsw i64 %k0.prime.kdma.source1501, %k0.prime.kdma.source1502, !dbg !13
  %k0.prime.kdma.source1504 = lshr i64 %k0.prime.kdma.source, 5, !dbg !13
  %k0.prime.kdma.source1505 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1506 = and i64 %k0.prime.kdma.source1505, 1, !dbg !13
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1500, %k0.prime.kdma.source1506, !dbg !13
  %k0.prime.kdma.source1508 = or disjoint i64 %k0.prime.kdma.source1504, %k0.prime.kdma.source1507, !dbg !13
  %k0.prime.kdma.source1509 = shl nuw nsw i64 %k0.prime.kdma.source1508, 10, !dbg !13
  %k0.prime.kdma.source1510 = sub nsw i64 %k0.prime.kdma.source1503, %k0.prime.kdma.source1509, !dbg !13
  %k0.prime.kdma.source1511 = and i64 %k0.prime.kdma.source1510, 16256, !dbg !13
  %k0.prime.kdma.source1512 = lshr i32 %kdma.tid, 4, !dbg !13
  %k0.prime.kdma.source1513 = and i32 %k0.prime.kdma.source1512, 8, !dbg !13
  %k0.prime.kdma.source1514 = zext nneg i32 %k0.prime.kdma.source1513 to i64, !dbg !13
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1511, %k0.prime.kdma.source1514, !dbg !13
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.2063, %k0.prime.kdma.source1515, !dbg !13
  %k0.prime.kdma.source1517 = trunc nuw i64 %k0.prime.kdma.source1516 to i32, !dbg !13
  %k0.prime.kdma.source1518 = shl nuw nsw i32 %k0.prime.kdma.source1517, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1518, %kdma.intra, !dbg !13
  %k0.prime.2064 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.2065 = insertelement <2 x i64> poison, i64 %k0.prime.2064, i64 0, !dbg !13
  %k0.prime.2066 = bitcast <2 x i64> %k0.prime.2065 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1377 = shufflevector <4 x i32> %k0.prime.2066, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.2067 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1540 = add i32 %38, 512, !dbg !13
  %k0.prime.kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1520 = lshr i64 %k0.prime.kdma.source1519, 4, !dbg !13
  %k0.prime.kdma.source1521 = shl nuw nsw i64 %k0.prime.kdma.source1520, 9, !dbg !13
  %k0.prime.kdma.source1522 = shl nuw nsw i64 %k0.prime.kdma.source1519, 7, !dbg !13
  %k0.prime.kdma.source1523 = add nuw nsw i64 %k0.prime.kdma.source1521, %k0.prime.kdma.source1522, !dbg !13
  %k0.prime.kdma.source1524 = lshr i64 %k0.prime.kdma.source1519, 5, !dbg !13
  %k0.prime.kdma.source1525 = lshr i64 %k0.prime.kdma.source1519, 2, !dbg !13
  %k0.prime.kdma.source1526 = and i64 %k0.prime.kdma.source1525, 1, !dbg !13
  %k0.prime.kdma.source1527 = and i64 %k0.prime.kdma.source1520, %k0.prime.kdma.source1526, !dbg !13
  %k0.prime.kdma.source1528 = add nuw nsw i64 %k0.prime.kdma.source1524, %k0.prime.kdma.source1527, !dbg !13
  %k0.prime.kdma.source1529 = shl nuw nsw i64 %k0.prime.kdma.source1528, 10, !dbg !13
  %k0.prime.kdma.source1530 = sub nsw i64 %k0.prime.kdma.source1523, %k0.prime.kdma.source1529, !dbg !13
  %k0.prime.kdma.source1531 = and i64 %k0.prime.kdma.source1530, 16256, !dbg !13
  %k0.prime.kdma.source1535 = or disjoint i64 %k0.prime.kdma.source1531, %k0.prime.kdma.source1514, !dbg !13
  %k0.prime.kdma.source1536 = or disjoint i64 %k0.prime.2063, %k0.prime.kdma.source1535, !dbg !13
  %k0.prime.kdma.source1537 = trunc nuw i64 %k0.prime.kdma.source1536 to i32, !dbg !13
  %k0.prime.kdma.source1538 = shl nuw nsw i32 %k0.prime.kdma.source1537, 1, !dbg !13
  %k0.prime.kdma.offset1539 = or disjoint i32 %k0.prime.kdma.source1538, %kdma.intra.hi, !dbg !13
  %k0.prime.2068 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1540, i32 %k0.prime.kdma.offset1539, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1542 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !13
  %k0.prime.kdma.source1543 = lshr i64 %k0.prime.kdma.source1542, 4, !dbg !13
  %k0.prime.kdma.source1544 = shl nuw nsw i64 %k0.prime.kdma.source1543, 9, !dbg !13
  %k0.prime.kdma.source1546 = add nuw nsw i64 %k0.prime.kdma.source1544, %k0.prime.kdma.source1502, !dbg !13
  %k0.prime.kdma.source1547 = lshr i64 %k0.prime.kdma.source1542, 5, !dbg !13
  %k0.prime.kdma.source1550 = and i64 %k0.prime.kdma.source1543, %k0.prime.kdma.source1506, !dbg !13
  %k0.prime.kdma.source1551 = or disjoint i64 %k0.prime.kdma.source1547, %k0.prime.kdma.source1550, !dbg !13
  %k0.prime.kdma.source1552 = shl nuw nsw i64 %k0.prime.kdma.source1551, 10, !dbg !13
  %k0.prime.kdma.source1553 = sub nsw i64 %k0.prime.kdma.source1546, %k0.prime.kdma.source1552, !dbg !13
  %k0.prime.kdma.source1554 = and i64 %k0.prime.kdma.source1553, 16256, !dbg !13
  %k0.prime.kdma.source1557 = or disjoint i32 %k0.prime.kdma.source1513, 16, !dbg !13
  %k0.prime.kdma.source1558 = zext nneg i32 %k0.prime.kdma.source1557 to i64, !dbg !13
  %k0.prime.kdma.source1559 = or disjoint i64 %k0.prime.kdma.source1554, %k0.prime.kdma.source1558, !dbg !13
  %k0.prime.kdma.source1560 = or disjoint i64 %k0.prime.2063, %k0.prime.kdma.source1559, !dbg !13
  %k0.prime.kdma.source1561 = trunc nuw i64 %k0.prime.kdma.source1560 to i32, !dbg !13
  %k0.prime.kdma.source1562 = shl nuw nsw i32 %k0.prime.kdma.source1561, 1, !dbg !13
  %k0.prime.kdma.offset1563 = add nuw i32 %k0.prime.kdma.source1562, %kdma.intra, !dbg !13
  %k0.prime.2069 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.prime.kdma.offset1563, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1588 = add i32 %46, 512, !dbg !13
  %k0.prime.kdma.source1566 = or disjoint i64 %k0.prime.kdma.source1519, 256, !dbg !13
  %k0.prime.kdma.source1567 = lshr i64 %k0.prime.kdma.source1566, 4, !dbg !13
  %k0.prime.kdma.source1568 = shl nuw nsw i64 %k0.prime.kdma.source1567, 9, !dbg !13
  %k0.prime.kdma.source1570 = add nuw nsw i64 %k0.prime.kdma.source1568, %k0.prime.kdma.source1522, !dbg !13
  %k0.prime.kdma.source1571 = lshr i64 %k0.prime.kdma.source1566, 5, !dbg !13
  %k0.prime.kdma.source1574 = and i64 %k0.prime.kdma.source1567, %k0.prime.kdma.source1526, !dbg !13
  %k0.prime.kdma.source1575 = add nuw nsw i64 %k0.prime.kdma.source1571, %k0.prime.kdma.source1574, !dbg !13
  %k0.prime.kdma.source1576 = shl nuw nsw i64 %k0.prime.kdma.source1575, 10, !dbg !13
  %k0.prime.kdma.source1577 = sub nsw i64 %k0.prime.kdma.source1570, %k0.prime.kdma.source1576, !dbg !13
  %k0.prime.kdma.source1578 = and i64 %k0.prime.kdma.source1577, 16256, !dbg !13
  %k0.prime.kdma.source1583 = or disjoint i64 %k0.prime.kdma.source1578, %k0.prime.kdma.source1558, !dbg !13
  %k0.prime.kdma.source1584 = or disjoint i64 %k0.prime.2063, %k0.prime.kdma.source1583, !dbg !13
  %k0.prime.kdma.source1585 = trunc nuw i64 %k0.prime.kdma.source1584 to i32, !dbg !13
  %k0.prime.kdma.source1586 = shl nuw nsw i32 %k0.prime.kdma.source1585, 1, !dbg !13
  %k0.prime.kdma.offset1587 = add nuw i32 %k0.prime.kdma.source1586, %kdma.intra.hi, !dbg !13
  %k0.prime.2070 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1588, i32 %k0.prime.kdma.offset1587, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
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
  %address.thread516 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %address.local517 = zext nneg i32 %address.thread516 to i64, !dbg !9
  %address.local523 = lshr i64 %address.local517, 2, !dbg !13
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0, !dbg !13
  %116 = bitcast <2 x i64> %115 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1377 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1500 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1501 = shl nuw nsw i64 %kdma.source1500, 9, !dbg !13
  %kdma.source1502 = shl nuw nsw i64 %kdma.source, 7, !dbg !13
  %kdma.source1503 = add nuw nsw i64 %kdma.source1501, %kdma.source1502, !dbg !13
  %kdma.source1504 = lshr i64 %kdma.source, 5, !dbg !13
  %kdma.source1505 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1506 = and i64 %kdma.source1505, 1, !dbg !13
  %kdma.source1507 = and i64 %kdma.source1500, %kdma.source1506, !dbg !13
  %kdma.source1508 = or disjoint i64 %kdma.source1504, %kdma.source1507, !dbg !13
  %kdma.source1509 = shl nuw nsw i64 %kdma.source1508, 10, !dbg !13
  %kdma.source1510 = sub nsw i64 %kdma.source1503, %kdma.source1509, !dbg !13
  %kdma.source1511 = and i64 %kdma.source1510, 16256, !dbg !13
  %kdma.source1512 = lshr i32 %kdma.tid, 4, !dbg !13
  %kdma.source1513 = and i32 %kdma.source1512, 8, !dbg !13
  %kdma.source1514 = zext nneg i32 %kdma.source1513 to i64, !dbg !13
  %kdma.source1515 = or disjoint i64 %kdma.source1511, %kdma.source1514, !dbg !13
  %kdma.source1516 = or disjoint i64 %113, %kdma.source1515, !dbg !13
  %kdma.source1517 = trunc nuw i64 %kdma.source1516 to i32, !dbg !13
  %kdma.source1518 = shl nuw nsw i32 %kdma.source1517, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1518, %kdma.intra, !dbg !13
  %kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1520 = lshr i64 %kdma.source1519, 4, !dbg !13
  %kdma.source1521 = shl nuw nsw i64 %kdma.source1520, 9, !dbg !13
  %kdma.source1522 = shl nuw nsw i64 %kdma.source1519, 7, !dbg !13
  %kdma.source1523 = add nuw nsw i64 %kdma.source1521, %kdma.source1522, !dbg !13
  %kdma.source1524 = lshr i64 %kdma.source1519, 5, !dbg !13
  %kdma.source1525 = lshr i64 %kdma.source1519, 2, !dbg !13
  %kdma.source1526 = and i64 %kdma.source1525, 1, !dbg !13
  %kdma.source1527 = and i64 %kdma.source1520, %kdma.source1526, !dbg !13
  %kdma.source1528 = add nuw nsw i64 %kdma.source1524, %kdma.source1527, !dbg !13
  %kdma.source1529 = shl nuw nsw i64 %kdma.source1528, 10, !dbg !13
  %kdma.source1530 = sub nsw i64 %kdma.source1523, %kdma.source1529, !dbg !13
  %kdma.source1531 = and i64 %kdma.source1530, 16256, !dbg !13
  %kdma.source1535 = or disjoint i64 %kdma.source1531, %kdma.source1514, !dbg !13
  %kdma.source1536 = or disjoint i64 %113, %kdma.source1535, !dbg !13
  %kdma.source1537 = trunc nuw i64 %kdma.source1536 to i32, !dbg !13
  %kdma.source1538 = shl nuw nsw i32 %kdma.source1537, 1, !dbg !13
  %kdma.offset1539 = or disjoint i32 %kdma.source1538, %kdma.intra.hi, !dbg !13
  %kdma.lds1540 = add i32 %38, 512, !dbg !13
  %kdma.source1542 = or disjoint i64 %kdma.source, 256, !dbg !13
  %kdma.source1543 = lshr i64 %kdma.source1542, 4, !dbg !13
  %kdma.source1544 = shl nuw nsw i64 %kdma.source1543, 9, !dbg !13
  %kdma.source1546 = add nuw nsw i64 %kdma.source1544, %kdma.source1502, !dbg !13
  %kdma.source1547 = lshr i64 %kdma.source1542, 5, !dbg !13
  %kdma.source1550 = and i64 %kdma.source1543, %kdma.source1506, !dbg !13
  %kdma.source1551 = or disjoint i64 %kdma.source1547, %kdma.source1550, !dbg !13
  %kdma.source1552 = shl nuw nsw i64 %kdma.source1551, 10, !dbg !13
  %kdma.source1553 = sub nsw i64 %kdma.source1546, %kdma.source1552, !dbg !13
  %kdma.source1554 = and i64 %kdma.source1553, 16256, !dbg !13
  %kdma.source1557 = or disjoint i32 %kdma.source1513, 16, !dbg !13
  %kdma.source1558 = zext nneg i32 %kdma.source1557 to i64, !dbg !13
  %kdma.source1559 = or disjoint i64 %kdma.source1554, %kdma.source1558, !dbg !13
  %kdma.source1560 = or disjoint i64 %113, %kdma.source1559, !dbg !13
  %kdma.source1561 = trunc nuw i64 %kdma.source1560 to i32, !dbg !13
  %kdma.source1562 = shl nuw nsw i32 %kdma.source1561, 1, !dbg !13
  %kdma.offset1563 = add nuw i32 %kdma.source1562, %kdma.intra, !dbg !13
  %kdma.source1566 = or disjoint i64 %kdma.source1519, 256, !dbg !13
  %kdma.source1567 = lshr i64 %kdma.source1566, 4, !dbg !13
  %kdma.source1568 = shl nuw nsw i64 %kdma.source1567, 9, !dbg !13
  %kdma.source1570 = add nuw nsw i64 %kdma.source1568, %kdma.source1522, !dbg !13
  %kdma.source1571 = lshr i64 %kdma.source1566, 5, !dbg !13
  %kdma.source1574 = and i64 %kdma.source1567, %kdma.source1526, !dbg !13
  %kdma.source1575 = add nuw nsw i64 %kdma.source1571, %kdma.source1574, !dbg !13
  %kdma.source1576 = shl nuw nsw i64 %kdma.source1575, 10, !dbg !13
  %kdma.source1577 = sub nsw i64 %kdma.source1570, %kdma.source1576, !dbg !13
  %kdma.source1578 = and i64 %kdma.source1577, 16256, !dbg !13
  %kdma.source1583 = or disjoint i64 %kdma.source1578, %kdma.source1558, !dbg !13
  %kdma.source1584 = or disjoint i64 %113, %kdma.source1583, !dbg !13
  %kdma.source1585 = trunc nuw i64 %kdma.source1584 to i32, !dbg !13
  %kdma.source1586 = shl nuw nsw i32 %kdma.source1585, 1, !dbg !13
  %kdma.offset1587 = add nuw i32 %kdma.source1586, %kdma.intra.hi, !dbg !13
  %kdma.lds1588 = add i32 %46, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %117 = or disjoint i64 %113, 32, !dbg !13
  %kdma.source1606 = or disjoint i64 %117, %kdma.source1515, !dbg !13
  %kdma.source1607 = trunc nuw i64 %kdma.source1606 to i32, !dbg !13
  %kdma.source1608 = shl nuw nsw i32 %kdma.source1607, 1, !dbg !13
  %kdma.offset1609 = or disjoint i32 %kdma.source1608, %kdma.intra, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1609, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1628 = or disjoint i64 %117, %kdma.source1535, !dbg !13
  %kdma.source1629 = trunc nuw i64 %kdma.source1628 to i32, !dbg !13
  %kdma.source1630 = shl nuw nsw i32 %kdma.source1629, 1, !dbg !13
  %kdma.offset1631 = or disjoint i32 %kdma.source1630, %kdma.intra.hi, !dbg !13
  %kdma.lds1632 = add i32 %51, 512, !dbg !13
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1632, i32 %kdma.offset1631, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1652 = or disjoint i64 %117, %kdma.source1559, !dbg !13
  %kdma.source1653 = trunc nuw i64 %kdma.source1652 to i32, !dbg !13
  %kdma.source1654 = shl nuw nsw i32 %kdma.source1653, 1, !dbg !13
  %kdma.offset1655 = add nuw i32 %kdma.source1654, %kdma.intra, !dbg !13
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1655, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1676 = or disjoint i64 %117, %kdma.source1583, !dbg !13
  %kdma.source1677 = trunc nuw i64 %kdma.source1676 to i32, !dbg !13
  %kdma.source1678 = shl nuw nsw i32 %kdma.source1677, 1, !dbg !13
  %kdma.offset1679 = add nuw i32 %kdma.source1678, %kdma.intra.hi, !dbg !13
  %kdma.lds1680 = add i32 %55, 512, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1680, i32 %kdma.offset1679, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %address.local560 = shl nuw nsw i64 %address.local517, 1, !dbg !15
  %address.local561 = and i64 %address.local560, 24, !dbg !15
  %address.local563 = shl i64 %address.local517, 62, !dbg !15
  %address.local564 = ashr exact i64 %address.local563, 57, !dbg !15
  %address.local565 = and i64 %address.local564, -64, !dbg !15
  %address.local567 = and i64 %address.local517, 63, !dbg !12
  %address.local569 = and i64 %address.local523, 12, !dbg !12
  %address.local571 = lshr i64 %address.local567, 5, !dbg !12
  %address.local572 = mul nuw nsw i64 %address.local571, 1016, !dbg !12
  %.idx1415 = mul nuw nsw i64 %address.local517, 320, !dbg !15
  %122 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1415, !dbg !15
  %.idx1416 = mul nsw i64 %address.local523, -1280, !dbg !15
  %123 = getelementptr i8, ptr addrspace(3) %122, i64 %.idx1416, !dbg !15
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local561, !dbg !15
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local565, !dbg !15
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local569, !dbg !15
  %address.local574 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local572, !dbg !15
  %127 = add nsw i64 %.idx1415, %.idx1416, !dbg !15
  %.idx1991 = shl nuw nsw i64 %address.local561, 1, !dbg !15
  %128 = or disjoint i64 %127, %.idx1991, !dbg !15
  %129 = or disjoint i64 %address.local565, %address.local569, !dbg !15
  %130 = shl nsw i64 %129, 1, !dbg !15
  %131 = add nsw i64 %130, %128, !dbg !15
  %address.local574.idx = mul nuw nsw i64 %address.local571, 2032, !dbg !15
  %132 = add nsw i64 %131, %address.local574.idx, !dbg !15
  %133 = lshr i64 %132, 6, !dbg !15
  %134 = and i64 %133, 8, !dbg !15
  %135 = xor i64 %132, %134, !dbg !15
  %136 = sub i64 %135, %132, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local574, i64 %136, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %lazy.base = extractvalue [2 x <1 x float>] %104, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base2076 = extractvalue [2 x <1 x float>] %104, 1, !dbg !15
  %lazy.seed2077 = shufflevector <1 x float> %lazy.base2076, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> %lazy.seed) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local585 = shl nuw nsw i32 %address.thread516, 1, !dbg !15
  %address.local586 = and i32 %address.local585, 24, !dbg !15
  %address.local587 = shl nuw nsw i32 %address.thread516, 8, !dbg !15
  %address.local588 = and i32 %address.local587, 768, !dbg !15
  %address.local589 = or disjoint i32 %address.local586, %address.local588, !dbg !15
  %address.local590 = shl nuw nsw i32 %address.thread516, 5, !dbg !15
  %address.local591 = and i32 %address.local590, 96, !dbg !15
  %address.local592 = or disjoint i32 %address.local589, %address.local591, !dbg !15
  %address.local593 = zext nneg i32 %address.local592 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %99) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local594 = lshr i32 %address.thread516, 1, !dbg !15
  %address.local595 = and i32 %address.local594, 1, !dbg !15
  %address.local596 = zext nneg i32 %address.local595 to i64, !dbg !15
  %address.local597 = add nsw i64 %address.base, %address.local596, !dbg !15
  %address.local598 = shl i64 %address.local597, 6, !dbg !15
  %address.local599 = sub i64 %address.local593, %address.local598, !dbg !15
  %140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local599, !dbg !15
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local569, !dbg !15
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local572, !dbg !15
  %address.local603 = getelementptr i8, ptr addrspace(3) %142, i64 64, !dbg !15
  %.idx19941997 = add i64 %address.local599, %address.local569, !dbg !15
  %143 = shl i64 %.idx19941997, 1, !dbg !15
  %144 = add i64 %143, %address.local574.idx, !dbg !15
  %145 = add i64 %144, 64, !dbg !15
  %146 = lshr i64 %145, 6, !dbg !15
  %147 = and i64 %146, 8, !dbg !15
  %148 = xor i64 %145, %147, !dbg !15
  %149 = sub i64 %148, %145, !dbg !15
  %kdma.swizzled1865 = getelementptr i8, ptr addrspace(3) %address.local603, i64 %149, !dbg !15
  %150 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1865, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 128, !dbg !15
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1866 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 4096, !dbg !15
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1866, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 4224, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !15, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %139, <4 x float> %138) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 1024, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 1152, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_lds_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 5120, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 5248, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_lds_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_lds_pair !38
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
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> %lazy.seed2077) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> %lazy.seed2077) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> %lazy.seed2077) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> %lazy.seed2077) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> %lazy.seed2077) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> %lazy.seed2077) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> %lazy.seed2077) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> %lazy.seed2077) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %163, <4 x float> %193) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %195 = or disjoint i64 %113, 64, !dbg !13
  %kdma.source1698 = or disjoint i64 %195, %kdma.source1515, !dbg !13
  %kdma.source1699 = trunc nuw i64 %kdma.source1698 to i32, !dbg !13
  %kdma.source1700 = shl nuw nsw i32 %kdma.source1699, 1, !dbg !13
  %kdma.offset1701 = or disjoint i32 %kdma.source1700, %kdma.intra, !dbg !13
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %kdma.offset1701, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1720 = or disjoint i64 %195, %kdma.source1535, !dbg !13
  %kdma.source1721 = trunc nuw i64 %kdma.source1720 to i32, !dbg !13
  %kdma.source1722 = shl nuw nsw i32 %kdma.source1721, 1, !dbg !13
  %kdma.offset1723 = or disjoint i32 %kdma.source1722, %kdma.intra.hi, !dbg !13
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1540, i32 %kdma.offset1723, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1744 = or disjoint i64 %195, %kdma.source1559, !dbg !13
  %kdma.source1745 = trunc nuw i64 %kdma.source1744 to i32, !dbg !13
  %kdma.source1746 = shl nuw nsw i32 %kdma.source1745, 1, !dbg !13
  %kdma.offset1747 = add nuw i32 %kdma.source1746, %kdma.intra, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %kdma.offset1747, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1768 = or disjoint i64 %195, %kdma.source1583, !dbg !13
  %kdma.source1769 = trunc nuw i64 %kdma.source1768 to i32, !dbg !13
  %kdma.source1770 = shl nuw nsw i32 %kdma.source1769, 1, !dbg !13
  %kdma.offset1771 = add nuw i32 %kdma.source1770, %kdma.intra.hi, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1588, i32 %kdma.offset1771, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.gep1895 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1895, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1899 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1896 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1900 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1897 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 8192, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1901 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 8320, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1898 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 12288, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1902 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 12416, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1903 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !15, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %200, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1904 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %210) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1905 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 9216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 9344, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1906 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 13312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 13440, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !15, !frisk.k_lds_pair !46
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
  %kdma.source1790 = or disjoint i64 %248, %kdma.source1515, !dbg !13
  %kdma.source1791 = trunc nuw i64 %kdma.source1790 to i32, !dbg !13
  %kdma.source1792 = shl nuw nsw i32 %kdma.source1791, 1, !dbg !13
  %kdma.offset1793 = or disjoint i32 %kdma.source1792, %kdma.intra, !dbg !13
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1793, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1812 = or disjoint i64 %248, %kdma.source1535, !dbg !13
  %kdma.source1813 = trunc nuw i64 %kdma.source1812 to i32, !dbg !13
  %kdma.source1814 = shl nuw nsw i32 %kdma.source1813, 1, !dbg !13
  %kdma.offset1815 = or disjoint i32 %kdma.source1814, %kdma.intra.hi, !dbg !13
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1632, i32 %kdma.offset1815, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1836 = or disjoint i64 %248, %kdma.source1559, !dbg !13
  %kdma.source1837 = trunc nuw i64 %kdma.source1836 to i32, !dbg !13
  %kdma.source1838 = shl nuw nsw i32 %kdma.source1837, 1, !dbg !13
  %kdma.offset1839 = add nuw i32 %kdma.source1838, %kdma.intra, !dbg !13
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1839, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %kdma.source1860 = or disjoint i64 %248, %kdma.source1583, !dbg !13
  %kdma.source1861 = trunc nuw i64 %kdma.source1860 to i32, !dbg !13
  %kdma.source1862 = shl nuw nsw i32 %kdma.source1861, 1, !dbg !13
  %kdma.offset1863 = add nuw i32 %kdma.source1862, %kdma.intra.hi, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1680, i32 %kdma.offset1863, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !47
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !15, !frisk.k_lds_pair !47
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !48
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !15, !frisk.k_lds_pair !48
  %257 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1865, align 8, !dbg !15, !frisk.k_lds_pair !49
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !15, !frisk.k_lds_pair !49
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1866, align 8, !dbg !15, !frisk.k_lds_pair !50
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !15, !frisk.k_lds_pair !50
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !15, !frisk.k_lds_pair !51
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %253, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !15, !frisk.k_lds_pair !52
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %255, <4 x float> %263) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !15, !frisk.k_lds_pair !53
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %257, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_lds_pair !54
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_lds_pair !54
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
  %address.local932 = shl nuw nsw i64 %address.local517, 5, !dbg !16
  %address.local933 = and i64 %address.local932, 8064, !dbg !16
  %address.local934 = shl nuw nsw i64 %address.local517, 3, !dbg !9
  %address.local935 = shl nuw nsw i64 %address.local523, 5, !dbg !16
  %address.local936 = sub nsw i64 %address.local934, %address.local935, !dbg !16
  %address.local937 = add nsw i64 %address.local933, %address.local936, !dbg !16
  %301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx14251426 = add nsw i64 %113, %address.local937, !dbg !16
  %.idx14251426.tr = trunc i64 %.idx14251426 to i32, !dbg !16
  %dma.buffer.offset1360 = shl i32 %.idx14251426.tr, 1, !dbg !16
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !16
  %303 = bitcast <2 x i64> %302 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %303, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %address.local939 = shl nuw nsw i32 %address.thread516, 3, !dbg !16
  %address.local940 = and i32 %address.local939, 24, !dbg !16
  %address.local942 = and i32 %address.local590, 8064, !dbg !16
  %address.local943 = or disjoint i32 %address.local940, %address.local942, !dbg !16
  %address.local944 = or disjoint i32 %address.local943, 8192, !dbg !16
  %305 = trunc nuw i64 %113 to i32, !dbg !16
  %.idx14271428.tr = or disjoint i32 %address.local944, %305, !dbg !16
  %dma.buffer.offset1358 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1358, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1895, align 8, !dbg !15, !frisk.k_tail_pair !33
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !15, !frisk.k_tail_pair !33
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !15, !frisk.k_tail_pair !34
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !15, !frisk.k_tail_pair !34
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !15, !frisk.k_tail_pair !35
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !15, !frisk.k_tail_pair !35
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !15, !frisk.k_tail_pair !36
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !15, !frisk.k_tail_pair !36
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
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !15, !frisk.k_tail_pair !31
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !15, !frisk.k_tail_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %308, <4 x float> %274) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %327 = extractelement <4 x float> %323, i64 0, !dbg !15
  %local.reduce154 = call float @llvm.maxnum.f32(float %319, float %327), !dbg !8, !frisk.local_reduction !1
  %328 = extractelement <4 x float> %323, i64 1, !dbg !15
  %local.reduce155 = call float @llvm.maxnum.f32(float %320, float %328), !dbg !8, !frisk.local_reduction !1
  %329 = extractelement <4 x float> %323, i64 2, !dbg !15
  %local.reduce156 = call float @llvm.maxnum.f32(float %321, float %329), !dbg !8, !frisk.local_reduction !1
  %330 = extractelement <4 x float> %323, i64 3, !dbg !15
  %local.reduce157 = call float @llvm.maxnum.f32(float %322, float %330), !dbg !8, !frisk.local_reduction !1
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !15, !frisk.k_tail_pair !37
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !15, !frisk.k_tail_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %326) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !15, !frisk.k_tail_pair !38
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %312, <4 x float> %276) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %337 = extractelement <4 x float> %333, i64 0, !dbg !15
  %local.reduce158 = call float @llvm.maxnum.f32(float %local.reduce154, float %337), !dbg !8, !frisk.local_reduction !1
  %338 = extractelement <4 x float> %333, i64 1, !dbg !15
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %338), !dbg !8, !frisk.local_reduction !1
  %339 = extractelement <4 x float> %333, i64 2, !dbg !15
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %339), !dbg !8, !frisk.local_reduction !1
  %340 = extractelement <4 x float> %333, i64 3, !dbg !15
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %340), !dbg !8, !frisk.local_reduction !1
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !15, !frisk.k_tail_pair !39
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %314, <4 x float> %336) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %324, <4 x float> %278) #9, !dbg !15
  %345 = extractelement <4 x float> %343, i64 0, !dbg !15
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %345), !dbg !8, !frisk.local_reduction !1
  %346 = extractelement <4 x float> %343, i64 1, !dbg !15
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %346), !dbg !8, !frisk.local_reduction !1
  %347 = extractelement <4 x float> %343, i64 2, !dbg !15
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %347), !dbg !8, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %343, i64 3, !dbg !15
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %348), !dbg !8, !frisk.local_reduction !1
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %331, <4 x float> %344) #9, !dbg !15
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %334, <4 x float> %280) #9, !dbg !15
  %351 = extractelement <4 x float> %349, i64 0, !dbg !15
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %351), !dbg !8, !frisk.local_reduction !1
  %352 = extractelement <4 x float> %349, i64 1, !dbg !15
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %352), !dbg !8, !frisk.local_reduction !1
  %353 = extractelement <4 x float> %349, i64 2, !dbg !15
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %353), !dbg !8, !frisk.local_reduction !1
  %354 = extractelement <4 x float> %349, i64 3, !dbg !15
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %354), !dbg !8, !frisk.local_reduction !1
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %341, <4 x float> %350) #9, !dbg !15
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %325, <4 x float> %282) #9, !dbg !15
  %357 = extractelement <4 x float> %355, i64 0, !dbg !15
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %357), !dbg !8, !frisk.local_reduction !1
  %358 = extractelement <4 x float> %355, i64 1, !dbg !15
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %358), !dbg !8, !frisk.local_reduction !1
  %359 = extractelement <4 x float> %355, i64 2, !dbg !15
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %359), !dbg !8, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %355, i64 3, !dbg !15
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %360), !dbg !8, !frisk.local_reduction !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %332, <4 x float> %356) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %335, <4 x float> %284) #9, !dbg !15
  %363 = extractelement <4 x float> %361, i64 0, !dbg !15
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %363), !dbg !8, !frisk.local_reduction !1
  %364 = extractelement <4 x float> %361, i64 1, !dbg !15
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %364), !dbg !8, !frisk.local_reduction !1
  %365 = extractelement <4 x float> %361, i64 2, !dbg !15
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %365), !dbg !8, !frisk.local_reduction !1
  %366 = extractelement <4 x float> %361, i64 3, !dbg !15
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %366), !dbg !8, !frisk.local_reduction !1
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %342, <4 x float> %362) #9, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %307, <4 x float> %286) #9, !dbg !15
  %369 = extractelement <4 x float> %367, i64 0, !dbg !15
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %369), !dbg !8, !frisk.local_reduction !1
  %370 = extractelement <4 x float> %367, i64 1, !dbg !15
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %370), !dbg !8, !frisk.local_reduction !1
  %371 = extractelement <4 x float> %367, i64 2, !dbg !15
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce176, float %371), !dbg !8, !frisk.local_reduction !1
  %372 = extractelement <4 x float> %367, i64 3, !dbg !15
  %local.reduce181 = call float @llvm.maxnum.f32(float %local.reduce177, float %372), !dbg !8, !frisk.local_reduction !1
  %local.reduce182 = call float @llvm.maxnum.f32(float %local.reduce178, float %local.reduce179), !dbg !8, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %local.reduce180, float %local.reduce181), !dbg !8, !frisk.local_reduction !1
  %373 = call float @llvm.maxnum.f32(float %local.reduce182, float %local.reduce183), !dbg !8, !frisk.local_reduction !1
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
  %local.reduce123 = call float @llvm.maxnum.f32(float %429, float %382), !dbg !8, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %377, float %383), !dbg !8, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %378, float %384), !dbg !8, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %379, float %385), !dbg !8, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %local.reduce123, float %388), !dbg !8, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %389), !dbg !8, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %390), !dbg !8, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %391), !dbg !8, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %394), !dbg !8, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %395), !dbg !8, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %396), !dbg !8, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %397), !dbg !8, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %400), !dbg !8, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %401), !dbg !8, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %402), !dbg !8, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %403), !dbg !8, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %406), !dbg !8, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %407), !dbg !8, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %408), !dbg !8, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %409), !dbg !8, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %412), !dbg !8, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %413), !dbg !8, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %414), !dbg !8, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %415), !dbg !8, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %417), !dbg !8, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %418), !dbg !8, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %419), !dbg !8, !frisk.local_reduction !1
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %420), !dbg !8, !frisk.local_reduction !1
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %local.reduce148), !dbg !8, !frisk.local_reduction !1
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce149, float %local.reduce150), !dbg !8, !frisk.local_reduction !1
  %430 = call float @llvm.maxnum.f32(float %local.reduce151, float %local.reduce152), !dbg !8, !frisk.local_reduction !1
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
  %lazy.positive2078 = call float @llvm.maxnum.f32(float %438, float 0x0000000000000000), !dbg !23
  %lazy.delta2079 = select i1 %lazy.first, float %438, float %lazy.positive2078, !dbg !23
  %lazy.negated2080 = fsub float 0x0000000000000000, %lazy.delta2079, !dbg !23
  %lazy.exp2081 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated2080), !dbg !23
  %lazy.alpha2082 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp2081, !dbg !23
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted = fsub <4 x float> %316, %lazy.delta.v, !dbg !23
  %lazy.delta.v.lane2083 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2084 = shufflevector <1 x float> %lazy.delta.v.lane2083, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2085 = fsub <4 x float> %323, %lazy.delta.v2084, !dbg !23
  %lazy.delta.v.lane2086 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2087 = shufflevector <1 x float> %lazy.delta.v.lane2086, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2088 = fsub <4 x float> %333, %lazy.delta.v2087, !dbg !23
  %lazy.delta.v.lane2089 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2090 = shufflevector <1 x float> %lazy.delta.v.lane2089, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2091 = fsub <4 x float> %343, %lazy.delta.v2090, !dbg !23
  %lazy.delta.v.lane2092 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2093 = shufflevector <1 x float> %lazy.delta.v.lane2092, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2094 = fsub <4 x float> %349, %lazy.delta.v2093, !dbg !23
  %lazy.delta.v.lane2095 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2096 = shufflevector <1 x float> %lazy.delta.v.lane2095, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2097 = fsub <4 x float> %355, %lazy.delta.v2096, !dbg !23
  %lazy.delta.v.lane2098 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2099 = shufflevector <1 x float> %lazy.delta.v.lane2098, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2100 = fsub <4 x float> %361, %lazy.delta.v2099, !dbg !23
  %lazy.delta.v.lane2101 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v2102 = shufflevector <1 x float> %lazy.delta.v.lane2101, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2103 = fsub <4 x float> %367, %lazy.delta.v2102, !dbg !23
  %lazy.delta.v.lane2104 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2105 = shufflevector <1 x float> %lazy.delta.v.lane2104, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2106 = fsub <4 x float> %374, %lazy.delta.v2105, !dbg !23
  %lazy.delta.v.lane2107 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2108 = shufflevector <1 x float> %lazy.delta.v.lane2107, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2109 = fsub <4 x float> %380, %lazy.delta.v2108, !dbg !23
  %lazy.delta.v.lane2110 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2111 = shufflevector <1 x float> %lazy.delta.v.lane2110, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2112 = fsub <4 x float> %386, %lazy.delta.v2111, !dbg !23
  %lazy.delta.v.lane2113 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2114 = shufflevector <1 x float> %lazy.delta.v.lane2113, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2115 = fsub <4 x float> %392, %lazy.delta.v2114, !dbg !23
  %lazy.delta.v.lane2116 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2117 = shufflevector <1 x float> %lazy.delta.v.lane2116, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2118 = fsub <4 x float> %398, %lazy.delta.v2117, !dbg !23
  %lazy.delta.v.lane2119 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2120 = shufflevector <1 x float> %lazy.delta.v.lane2119, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2121 = fsub <4 x float> %404, %lazy.delta.v2120, !dbg !23
  %lazy.delta.v.lane2122 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2123 = shufflevector <1 x float> %lazy.delta.v.lane2122, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2124 = fsub <4 x float> %410, %lazy.delta.v2123, !dbg !23
  %lazy.delta.v.lane2125 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.delta.v2126 = shufflevector <1 x float> %lazy.delta.v.lane2125, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted2127 = fsub <4 x float> %416, %lazy.delta.v2126, !dbg !23
  %lazy.state = extractvalue [2 x <1 x float>] %103, 0, !dbg !23
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !23
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !23
  %lazy.state2128 = extractvalue [2 x <1 x float>] %103, 1, !dbg !23
  %lazy.alpha.v.lane2129 = insertelement <1 x float> poison, float %lazy.alpha2082, i64 0, !dbg !23
  %lazy.scaled2130 = fmul <1 x float> %lazy.state2128, %lazy.alpha.v.lane2129, !dbg !23
  %lazy.state.next2131 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2130, 1, !dbg !23
  %lazy.state2132 = extractvalue [2 x <8 x float>] %107, 0, !dbg !23
  %lazy.alpha.v.lane2133 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane2133, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2134 = fmul <8 x float> %lazy.state2132, %lazy.alpha.v, !dbg !23
  %lazy.state.next2135 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2134, 0, !dbg !23
  %lazy.state2136 = extractvalue [2 x <8 x float>] %107, 1, !dbg !23
  %lazy.alpha.v.lane2137 = insertelement <1 x float> poison, float %lazy.alpha2082, i64 0, !dbg !23
  %lazy.alpha.v2138 = shufflevector <1 x float> %lazy.alpha.v.lane2137, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2139 = fmul <8 x float> %lazy.state2136, %lazy.alpha.v2138, !dbg !23
  %lazy.state.next2140 = insertvalue [2 x <8 x float>] %lazy.state.next2135, <8 x float> %lazy.scaled2139, 1, !dbg !23
  %lazy.state2141 = extractvalue [2 x <8 x float>] %106, 0, !dbg !23
  %lazy.alpha.v.lane2142 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2143 = shufflevector <1 x float> %lazy.alpha.v.lane2142, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2144 = fmul <8 x float> %lazy.state2141, %lazy.alpha.v2143, !dbg !23
  %lazy.state.next2145 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2144, 0, !dbg !23
  %lazy.state2146 = extractvalue [2 x <8 x float>] %106, 1, !dbg !23
  %lazy.alpha.v.lane2147 = insertelement <1 x float> poison, float %lazy.alpha2082, i64 0, !dbg !23
  %lazy.alpha.v2148 = shufflevector <1 x float> %lazy.alpha.v.lane2147, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2149 = fmul <8 x float> %lazy.state2146, %lazy.alpha.v2148, !dbg !23
  %lazy.state.next2150 = insertvalue [2 x <8 x float>] %lazy.state.next2145, <8 x float> %lazy.scaled2149, 1, !dbg !23
  %lazy.state2151 = extractvalue [2 x <8 x float>] %105, 0, !dbg !23
  %lazy.alpha.v.lane2152 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2153 = shufflevector <1 x float> %lazy.alpha.v.lane2152, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2154 = fmul <8 x float> %lazy.state2151, %lazy.alpha.v2153, !dbg !23
  %lazy.state.next2155 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2154, 0, !dbg !23
  %lazy.state2156 = extractvalue [2 x <8 x float>] %105, 1, !dbg !23
  %lazy.alpha.v.lane2157 = insertelement <1 x float> poison, float %lazy.alpha2082, i64 0, !dbg !23
  %lazy.alpha.v2158 = shufflevector <1 x float> %lazy.alpha.v.lane2157, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2159 = fmul <8 x float> %lazy.state2156, %lazy.alpha.v2158, !dbg !23
  %lazy.state.next2160 = insertvalue [2 x <8 x float>] %lazy.state.next2155, <8 x float> %lazy.scaled2159, 1, !dbg !23
  %lazy.state2161 = extractvalue [2 x <8 x float>] %102, 0, !dbg !23
  %lazy.alpha.v.lane2162 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2163 = shufflevector <1 x float> %lazy.alpha.v.lane2162, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2164 = fmul <8 x float> %lazy.state2161, %lazy.alpha.v2163, !dbg !23
  %lazy.state.next2165 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2164, 0, !dbg !23
  %lazy.state2166 = extractvalue [2 x <8 x float>] %102, 1, !dbg !23
  %lazy.alpha.v.lane2167 = insertelement <1 x float> poison, float %lazy.alpha2082, i64 0, !dbg !23
  %lazy.alpha.v2168 = shufflevector <1 x float> %lazy.alpha.v.lane2167, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2169 = fmul <8 x float> %lazy.state2166, %lazy.alpha.v2168, !dbg !23
  %lazy.state.next2170 = insertvalue [2 x <8 x float>] %lazy.state.next2165, <8 x float> %lazy.scaled2169, 1, !dbg !23
  %lazy.base.old = extractvalue [2 x <1 x float>] %104, 0, !dbg !23
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !23
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !23
  %lazy.base.old2171 = extractvalue [2 x <1 x float>] %104, 1, !dbg !23
  %lazy.delta.b.lane2172 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !23
  %lazy.base.next2173 = fsub <1 x float> %lazy.base.old2171, %lazy.delta.b.lane2172, !dbg !23
  %lazy.base.state2174 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2173, 1, !dbg !23
  br label %lazy.join, !dbg !23

lazy.join:                                        ; preds = %lazy.slow, %109
  %lazy.score = phi <4 x float> [ %316, %109 ], [ %lazy.shifted, %lazy.slow ], !dbg !23
  %lazy.score2175 = phi <4 x float> [ %323, %109 ], [ %lazy.shifted2085, %lazy.slow ], !dbg !23
  %lazy.score2176 = phi <4 x float> [ %333, %109 ], [ %lazy.shifted2088, %lazy.slow ], !dbg !23
  %lazy.score2177 = phi <4 x float> [ %343, %109 ], [ %lazy.shifted2091, %lazy.slow ], !dbg !23
  %lazy.score2178 = phi <4 x float> [ %349, %109 ], [ %lazy.shifted2094, %lazy.slow ], !dbg !23
  %lazy.score2179 = phi <4 x float> [ %355, %109 ], [ %lazy.shifted2097, %lazy.slow ], !dbg !23
  %lazy.score2180 = phi <4 x float> [ %361, %109 ], [ %lazy.shifted2100, %lazy.slow ], !dbg !23
  %lazy.score2181 = phi <4 x float> [ %367, %109 ], [ %lazy.shifted2103, %lazy.slow ], !dbg !23
  %lazy.score2182 = phi <4 x float> [ %374, %109 ], [ %lazy.shifted2106, %lazy.slow ], !dbg !23
  %lazy.score2183 = phi <4 x float> [ %380, %109 ], [ %lazy.shifted2109, %lazy.slow ], !dbg !23
  %lazy.score2184 = phi <4 x float> [ %386, %109 ], [ %lazy.shifted2112, %lazy.slow ], !dbg !23
  %lazy.score2185 = phi <4 x float> [ %392, %109 ], [ %lazy.shifted2115, %lazy.slow ], !dbg !23
  %lazy.score2186 = phi <4 x float> [ %398, %109 ], [ %lazy.shifted2118, %lazy.slow ], !dbg !23
  %lazy.score2187 = phi <4 x float> [ %404, %109 ], [ %lazy.shifted2121, %lazy.slow ], !dbg !23
  %lazy.score2188 = phi <4 x float> [ %410, %109 ], [ %lazy.shifted2124, %lazy.slow ], !dbg !23
  %lazy.score2189 = phi <4 x float> [ %416, %109 ], [ %lazy.shifted2127, %lazy.slow ], !dbg !23
  %lazy.state2190 = phi [2 x <1 x float>] [ %103, %109 ], [ %lazy.state.next2131, %lazy.slow ], !dbg !23
  %lazy.state2191 = phi [2 x <8 x float>] [ %107, %109 ], [ %lazy.state.next2140, %lazy.slow ], !dbg !23
  %lazy.state2192 = phi [2 x <8 x float>] [ %106, %109 ], [ %lazy.state.next2150, %lazy.slow ], !dbg !23
  %lazy.state2193 = phi [2 x <8 x float>] [ %105, %109 ], [ %lazy.state.next2160, %lazy.slow ], !dbg !23
  %lazy.state2194 = phi [2 x <8 x float>] [ %102, %109 ], [ %lazy.state.next2170, %lazy.slow ], !dbg !23
  %lazy.base.join = phi [2 x <1 x float>] [ %104, %109 ], [ %lazy.base.state2174, %lazy.slow ], !dbg !23
  %439 = extractvalue [2 x <1 x float>] %lazy.state2190, 0, !dbg !23
  %440 = extractelement <1 x float> %439, i64 0, !dbg !23
  %441 = extractvalue [2 x <1 x float>] %lazy.state2190, 1, !dbg !23
  %442 = extractelement <1 x float> %441, i64 0, !dbg !23
  %443 = extractvalue [2 x <8 x float>] %lazy.state2194, 0, !dbg !27
  %444 = shufflevector <8 x float> %443, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %445 = shufflevector <8 x float> %443, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %446 = extractvalue [2 x <8 x float>] %lazy.state2194, 1, !dbg !27
  %447 = shufflevector <8 x float> %446, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %448 = shufflevector <8 x float> %446, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = add nsw i64 %117, %address.local937, !dbg !16
  %vdb.address1441 = trunc i64 %vdb.address to i32, !dbg !16
  %vdb.address1442 = shl i32 %vdb.address1441, 1, !dbg !16
  %vdb.address1447 = trunc nuw i64 %117 to i32, !dbg !16
  %vdb.address1448 = or disjoint i32 %address.local944, %vdb.address1447, !dbg !16
  %vdb.address1449 = shl nuw nsw i32 %vdb.address1448, 1, !dbg !16
  %vdb.dma.base = add i32 %38, 8192, !dbg !14
  %449 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1442, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %vdb.dma.base1454 = add i32 %46, 8192, !dbg !14
  %450 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1454, i32 %vdb.address1449, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %address.local948 = and i32 %address.local939, 120, !dbg !14
  %address.local950 = and i32 %address.local590, 1536, !dbg !14
  %address.local951 = or disjoint i32 %address.local948, %address.local950, !dbg !14
  %address.local952 = zext nneg i32 %address.local951 to i64, !dbg !14
  %address.local953 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local952, !dbg !14
  %451 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %452 = bitcast <4 x i32> %451 to <8 x half>, !dbg !14
  %address.local954 = and i64 %address.local517, 60, !dbg !14
  %address.local955 = add nuw nsw i64 %address.local954, %address.local517, !dbg !14
  %address.local957.idx = shl nuw nsw i64 %address.local955, 4, !dbg !14
  %address.local957 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local957.idx, !dbg !14
  %address.local958 = lshr i64 %address.local567, 4, !dbg !14
  %address.local960.idx = mul nuw nsw i64 %address.local958, 768, !dbg !14
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local957, i64 %address.local960.idx, !dbg !14
  %address.local962.idx = mul nsw i64 %address.local523, -64, !dbg !14
  %address.local962 = getelementptr i8, ptr addrspace(3) %address.local960, i64 %address.local962.idx, !dbg !14
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 256, !dbg !14
  %453 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %454 = bitcast <4 x i32> %453 to <8 x half>, !dbg !14
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local962, i64 512, !dbg !14
  %455 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %456 = bitcast <4 x i32> %455 to <8 x half>, !dbg !14
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local962, i64 768, !dbg !14
  %457 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %458 = bitcast <4 x i32> %457 to <8 x half>, !dbg !14
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local953, i64 4096, !dbg !14
  %459 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %460 = bitcast <4 x i32> %459 to <8 x half>, !dbg !14
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4352, !dbg !14
  %461 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %462 = bitcast <4 x i32> %461 to <8 x half>, !dbg !14
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !14
  %463 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %464 = bitcast <4 x i32> %463 to <8 x half>, !dbg !14
  %address.local985 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4864, !dbg !14
  %465 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local985, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
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
  %483 = extractelement <4 x float> %lazy.score2182, i64 0, !dbg !24
  %484 = call float @llvm.amdgcn.exp2.f32(float %483), !dbg !24
  %485 = insertelement <4 x float> poison, float %484, i64 0, !dbg !26
  %486 = extractelement <4 x float> %lazy.score2182, i64 1, !dbg !24
  %487 = call float @llvm.amdgcn.exp2.f32(float %486), !dbg !24
  %488 = insertelement <4 x float> %485, float %487, i64 1, !dbg !26
  %489 = extractelement <4 x float> %lazy.score2182, i64 2, !dbg !24
  %490 = call float @llvm.amdgcn.exp2.f32(float %489), !dbg !24
  %491 = insertelement <4 x float> %488, float %490, i64 2, !dbg !26
  %492 = extractelement <4 x float> %lazy.score2182, i64 3, !dbg !24
  %493 = call float @llvm.amdgcn.exp2.f32(float %492), !dbg !24
  %494 = insertelement <4 x float> %491, float %493, i64 3, !dbg !26
  %math.result403 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %494) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02195 = extractelement <4 x float> %math.result403, i32 0, !dbg !26
  %.cvtpack.f12196 = extractelement <4 x float> %math.result403, i32 1, !dbg !26
  %.cvtpack.f22197 = extractelement <4 x float> %math.result403, i32 2, !dbg !26
  %.cvtpack.f32198 = extractelement <4 x float> %math.result403, i32 3, !dbg !26
  %.cvtpack.p02199 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02195, float %.cvtpack.f12196), !dbg !26
  %.cvtpack.p12200 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22197, float %.cvtpack.f32198), !dbg !26
  %.cvtpack.v02201 = insertelement <2 x i32> poison, i32 %.cvtpack.p02199, i32 0, !dbg !26
  %.cvtpack.v12202 = insertelement <2 x i32> %.cvtpack.v02201, i32 %.cvtpack.p12200, i32 1, !dbg !26
  %495 = bitcast <2 x i32> %.cvtpack.v12202 to <4 x half>, !dbg !26
  %math.packed404 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %495) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %466, <4 x float> %447) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %481, <4 x float> %448) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %498 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %499 = extractelement <4 x float> %lazy.score2175, i64 0, !dbg !24
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !24
  %local.reduce92 = fadd float %468, %500, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce92) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !26
  %502 = extractelement <4 x float> %lazy.score2175, i64 1, !dbg !24
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !24
  %local.reduce93 = fadd float %471, %503, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1998 = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %504 = insertelement <4 x float> %501, float %503, i64 1, !dbg !26
  %505 = extractelement <4 x float> %lazy.score2175, i64 2, !dbg !24
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !24
  %local.reduce94 = fadd float %474, %506, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor1999 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %507 = insertelement <4 x float> %504, float %506, i64 2, !dbg !26
  %508 = extractelement <4 x float> %lazy.score2175, i64 3, !dbg !24
  %509 = call float @llvm.amdgcn.exp2.f32(float %508), !dbg !24
  %local.reduce95 = fadd float %477, %509, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2000 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %510 = insertelement <4 x float> %507, float %509, i64 3, !dbg !26
  %math.result389 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %510) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02203 = extractelement <4 x float> %math.result389, i32 0, !dbg !26
  %.cvtpack.f12204 = extractelement <4 x float> %math.result389, i32 1, !dbg !26
  %.cvtpack.f22205 = extractelement <4 x float> %math.result389, i32 2, !dbg !26
  %.cvtpack.f32206 = extractelement <4 x float> %math.result389, i32 3, !dbg !26
  %.cvtpack.p02207 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02203, float %.cvtpack.f12204), !dbg !26
  %.cvtpack.p12208 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22205, float %.cvtpack.f32206), !dbg !26
  %.cvtpack.v02209 = insertelement <2 x i32> poison, i32 %.cvtpack.p02207, i32 0, !dbg !26
  %.cvtpack.v12210 = insertelement <2 x i32> %.cvtpack.v02209, i32 %.cvtpack.p12208, i32 1, !dbg !26
  %511 = bitcast <2 x i32> %.cvtpack.v12210 to <4 x half>, !dbg !26
  %math.packed390 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %511) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %498, <4 x float> %480) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %513 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %513, <4 x float> %482) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %515 = extractelement <4 x float> %lazy.score2183, i64 0, !dbg !24
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !24
  %local.reduce = fadd float %484, %516, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2001 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %517 = insertelement <4 x float> poison, float %516, i64 0, !dbg !26
  %518 = extractelement <4 x float> %lazy.score2183, i64 1, !dbg !24
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !24
  %local.reduce62 = fadd float %487, %519, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2002 = call float asm sideeffect "", "=v,0"(float %local.reduce62) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %520 = insertelement <4 x float> %517, float %519, i64 1, !dbg !26
  %521 = extractelement <4 x float> %lazy.score2183, i64 2, !dbg !24
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !24
  %local.reduce63 = fadd float %490, %522, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2003 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %523 = insertelement <4 x float> %520, float %522, i64 2, !dbg !26
  %524 = extractelement <4 x float> %lazy.score2183, i64 3, !dbg !24
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !24
  %local.reduce64 = fadd float %493, %525, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2004 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %526 = insertelement <4 x float> %523, float %525, i64 3, !dbg !26
  %math.result405 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %526) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02211 = extractelement <4 x float> %math.result405, i32 0, !dbg !26
  %.cvtpack.f12212 = extractelement <4 x float> %math.result405, i32 1, !dbg !26
  %.cvtpack.f22213 = extractelement <4 x float> %math.result405, i32 2, !dbg !26
  %.cvtpack.f32214 = extractelement <4 x float> %math.result405, i32 3, !dbg !26
  %.cvtpack.p02215 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02211, float %.cvtpack.f12212), !dbg !26
  %.cvtpack.p12216 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22213, float %.cvtpack.f32214), !dbg !26
  %.cvtpack.v02217 = insertelement <2 x i32> poison, i32 %.cvtpack.p02215, i32 0, !dbg !26
  %.cvtpack.v12218 = insertelement <2 x i32> %.cvtpack.v02217, i32 %.cvtpack.p12216, i32 1, !dbg !26
  %527 = bitcast <2 x i32> %.cvtpack.v12218 to <4 x half>, !dbg !26
  %math.packed406 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %527) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %498, <4 x float> %496) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %513, <4 x float> %497) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %530 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %531 = extractelement <4 x float> %lazy.score2176, i64 0, !dbg !24
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !24
  %local.reduce96 = fadd float %k0.sum.anchor, %532, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2005 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %533 = insertelement <4 x float> poison, float %532, i64 0, !dbg !26
  %534 = extractelement <4 x float> %lazy.score2176, i64 1, !dbg !24
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !24
  %local.reduce97 = fadd float %k0.sum.anchor1998, %535, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2006 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %536 = insertelement <4 x float> %533, float %535, i64 1, !dbg !26
  %537 = extractelement <4 x float> %lazy.score2176, i64 2, !dbg !24
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !24
  %local.reduce98 = fadd float %k0.sum.anchor1999, %538, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2007 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %539 = insertelement <4 x float> %536, float %538, i64 2, !dbg !26
  %540 = extractelement <4 x float> %lazy.score2176, i64 3, !dbg !24
  %541 = call float @llvm.amdgcn.exp2.f32(float %540), !dbg !24
  %local.reduce99 = fadd float %k0.sum.anchor2000, %541, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2008 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %542 = insertelement <4 x float> %539, float %541, i64 3, !dbg !26
  %math.result391 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02219 = extractelement <4 x float> %math.result391, i32 0, !dbg !26
  %.cvtpack.f12220 = extractelement <4 x float> %math.result391, i32 1, !dbg !26
  %.cvtpack.f22221 = extractelement <4 x float> %math.result391, i32 2, !dbg !26
  %.cvtpack.f32222 = extractelement <4 x float> %math.result391, i32 3, !dbg !26
  %.cvtpack.p02223 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02219, float %.cvtpack.f12220), !dbg !26
  %.cvtpack.p12224 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22221, float %.cvtpack.f32222), !dbg !26
  %.cvtpack.v02225 = insertelement <2 x i32> poison, i32 %.cvtpack.p02223, i32 0, !dbg !26
  %.cvtpack.v12226 = insertelement <2 x i32> %.cvtpack.v02225, i32 %.cvtpack.p12224, i32 1, !dbg !26
  %543 = bitcast <2 x i32> %.cvtpack.v12226 to <4 x half>, !dbg !26
  %math.packed392 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %530, <4 x float> %512) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %545 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %545, <4 x float> %514) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %547 = extractelement <4 x float> %lazy.score2184, i64 0, !dbg !24
  %548 = call float @llvm.amdgcn.exp2.f32(float %547), !dbg !24
  %local.reduce65 = fadd float %k0.sum.anchor2001, %548, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2009 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %549 = insertelement <4 x float> poison, float %548, i64 0, !dbg !26
  %550 = extractelement <4 x float> %lazy.score2184, i64 1, !dbg !24
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !24
  %local.reduce66 = fadd float %k0.sum.anchor2002, %551, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2010 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %552 = insertelement <4 x float> %549, float %551, i64 1, !dbg !26
  %553 = extractelement <4 x float> %lazy.score2184, i64 2, !dbg !24
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !24
  %local.reduce67 = fadd float %k0.sum.anchor2003, %554, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2011 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %555 = insertelement <4 x float> %552, float %554, i64 2, !dbg !26
  %556 = extractelement <4 x float> %lazy.score2184, i64 3, !dbg !24
  %557 = call float @llvm.amdgcn.exp2.f32(float %556), !dbg !24
  %local.reduce68 = fadd float %k0.sum.anchor2004, %557, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2012 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %558 = insertelement <4 x float> %555, float %557, i64 3, !dbg !26
  %math.result407 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %558) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02227 = extractelement <4 x float> %math.result407, i32 0, !dbg !26
  %.cvtpack.f12228 = extractelement <4 x float> %math.result407, i32 1, !dbg !26
  %.cvtpack.f22229 = extractelement <4 x float> %math.result407, i32 2, !dbg !26
  %.cvtpack.f32230 = extractelement <4 x float> %math.result407, i32 3, !dbg !26
  %.cvtpack.p02231 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02227, float %.cvtpack.f12228), !dbg !26
  %.cvtpack.p12232 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22229, float %.cvtpack.f32230), !dbg !26
  %.cvtpack.v02233 = insertelement <2 x i32> poison, i32 %.cvtpack.p02231, i32 0, !dbg !26
  %.cvtpack.v12234 = insertelement <2 x i32> %.cvtpack.v02233, i32 %.cvtpack.p12232, i32 1, !dbg !26
  %559 = bitcast <2 x i32> %.cvtpack.v12234 to <4 x half>, !dbg !26
  %math.packed408 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %559) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %530, <4 x float> %528) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %545, <4 x float> %529) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %562 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %563 = extractelement <4 x float> %lazy.score2177, i64 0, !dbg !24
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !24
  %local.reduce100 = fadd float %k0.sum.anchor2005, %564, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2013 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !26
  %566 = extractelement <4 x float> %lazy.score2177, i64 1, !dbg !24
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !24
  %local.reduce101 = fadd float %k0.sum.anchor2006, %567, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2014 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> %565, float %567, i64 1, !dbg !26
  %569 = extractelement <4 x float> %lazy.score2177, i64 2, !dbg !24
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !24
  %local.reduce102 = fadd float %k0.sum.anchor2007, %570, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2015 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 2, !dbg !26
  %572 = extractelement <4 x float> %lazy.score2177, i64 3, !dbg !24
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !24
  %local.reduce103 = fadd float %k0.sum.anchor2008, %573, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2016 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %571, float %573, i64 3, !dbg !26
  %math.result393 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02235 = extractelement <4 x float> %math.result393, i32 0, !dbg !26
  %.cvtpack.f12236 = extractelement <4 x float> %math.result393, i32 1, !dbg !26
  %.cvtpack.f22237 = extractelement <4 x float> %math.result393, i32 2, !dbg !26
  %.cvtpack.f32238 = extractelement <4 x float> %math.result393, i32 3, !dbg !26
  %.cvtpack.p02239 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02235, float %.cvtpack.f12236), !dbg !26
  %.cvtpack.p12240 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22237, float %.cvtpack.f32238), !dbg !26
  %.cvtpack.v02241 = insertelement <2 x i32> poison, i32 %.cvtpack.p02239, i32 0, !dbg !26
  %.cvtpack.v12242 = insertelement <2 x i32> %.cvtpack.v02241, i32 %.cvtpack.p12240, i32 1, !dbg !26
  %575 = bitcast <2 x i32> %.cvtpack.v12242 to <4 x half>, !dbg !26
  %math.packed394 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %562, <4 x float> %544) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %577 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %577, <4 x float> %546) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = extractelement <4 x float> %lazy.score2185, i64 0, !dbg !24
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !24
  %local.reduce69 = fadd float %k0.sum.anchor2009, %580, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2017 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %581 = insertelement <4 x float> poison, float %580, i64 0, !dbg !26
  %582 = extractelement <4 x float> %lazy.score2185, i64 1, !dbg !24
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !24
  %local.reduce70 = fadd float %k0.sum.anchor2010, %583, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2018 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %584 = insertelement <4 x float> %581, float %583, i64 1, !dbg !26
  %585 = extractelement <4 x float> %lazy.score2185, i64 2, !dbg !24
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !24
  %local.reduce71 = fadd float %k0.sum.anchor2011, %586, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2019 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %587 = insertelement <4 x float> %584, float %586, i64 2, !dbg !26
  %588 = extractelement <4 x float> %lazy.score2185, i64 3, !dbg !24
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !24
  %local.reduce72 = fadd float %k0.sum.anchor2012, %589, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2020 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %590 = insertelement <4 x float> %587, float %589, i64 3, !dbg !26
  %math.result409 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %590) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02243 = extractelement <4 x float> %math.result409, i32 0, !dbg !26
  %.cvtpack.f12244 = extractelement <4 x float> %math.result409, i32 1, !dbg !26
  %.cvtpack.f22245 = extractelement <4 x float> %math.result409, i32 2, !dbg !26
  %.cvtpack.f32246 = extractelement <4 x float> %math.result409, i32 3, !dbg !26
  %.cvtpack.p02247 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02243, float %.cvtpack.f12244), !dbg !26
  %.cvtpack.p12248 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22245, float %.cvtpack.f32246), !dbg !26
  %.cvtpack.v02249 = insertelement <2 x i32> poison, i32 %.cvtpack.p02247, i32 0, !dbg !26
  %.cvtpack.v12250 = insertelement <2 x i32> %.cvtpack.v02249, i32 %.cvtpack.p12248, i32 1, !dbg !26
  %591 = bitcast <2 x i32> %.cvtpack.v12250 to <4 x half>, !dbg !26
  %math.packed410 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %591) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %562, <4 x float> %560) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %577, <4 x float> %561) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %594 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %595 = extractelement <4 x float> %lazy.score2178, i64 0, !dbg !24
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !24
  %local.reduce104 = fadd float %k0.sum.anchor2013, %596, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2021 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %597 = insertelement <4 x float> poison, float %596, i64 0, !dbg !26
  %598 = extractelement <4 x float> %lazy.score2178, i64 1, !dbg !24
  %599 = call float @llvm.amdgcn.exp2.f32(float %598), !dbg !24
  %local.reduce105 = fadd float %k0.sum.anchor2014, %599, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2022 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %600 = insertelement <4 x float> %597, float %599, i64 1, !dbg !26
  %601 = extractelement <4 x float> %lazy.score2178, i64 2, !dbg !24
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !24
  %local.reduce106 = fadd float %k0.sum.anchor2015, %602, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2023 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %603 = insertelement <4 x float> %600, float %602, i64 2, !dbg !26
  %604 = extractelement <4 x float> %lazy.score2178, i64 3, !dbg !24
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !24
  %local.reduce107 = fadd float %k0.sum.anchor2016, %605, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2024 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %606 = insertelement <4 x float> %603, float %605, i64 3, !dbg !26
  %math.result395 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02251 = extractelement <4 x float> %math.result395, i32 0, !dbg !26
  %.cvtpack.f12252 = extractelement <4 x float> %math.result395, i32 1, !dbg !26
  %.cvtpack.f22253 = extractelement <4 x float> %math.result395, i32 2, !dbg !26
  %.cvtpack.f32254 = extractelement <4 x float> %math.result395, i32 3, !dbg !26
  %.cvtpack.p02255 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02251, float %.cvtpack.f12252), !dbg !26
  %.cvtpack.p12256 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22253, float %.cvtpack.f32254), !dbg !26
  %.cvtpack.v02257 = insertelement <2 x i32> poison, i32 %.cvtpack.p02255, i32 0, !dbg !26
  %.cvtpack.v12258 = insertelement <2 x i32> %.cvtpack.v02257, i32 %.cvtpack.p12256, i32 1, !dbg !26
  %607 = bitcast <2 x i32> %.cvtpack.v12258 to <4 x half>, !dbg !26
  %math.packed396 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %594, <4 x float> %576) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %609 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %609, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %611 = extractelement <4 x float> %lazy.score2186, i64 0, !dbg !24
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !24
  %local.reduce73 = fadd float %k0.sum.anchor2017, %612, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2025 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %613 = insertelement <4 x float> poison, float %612, i64 0, !dbg !26
  %614 = extractelement <4 x float> %lazy.score2186, i64 1, !dbg !24
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !24
  %local.reduce74 = fadd float %k0.sum.anchor2018, %615, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2026 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %616 = insertelement <4 x float> %613, float %615, i64 1, !dbg !26
  %617 = extractelement <4 x float> %lazy.score2186, i64 2, !dbg !24
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !24
  %local.reduce75 = fadd float %k0.sum.anchor2019, %618, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2027 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> %616, float %618, i64 2, !dbg !26
  %620 = extractelement <4 x float> %lazy.score2186, i64 3, !dbg !24
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !24
  %local.reduce76 = fadd float %k0.sum.anchor2020, %621, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2028 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 3, !dbg !26
  %math.result411 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %622) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02259 = extractelement <4 x float> %math.result411, i32 0, !dbg !26
  %.cvtpack.f12260 = extractelement <4 x float> %math.result411, i32 1, !dbg !26
  %.cvtpack.f22261 = extractelement <4 x float> %math.result411, i32 2, !dbg !26
  %.cvtpack.f32262 = extractelement <4 x float> %math.result411, i32 3, !dbg !26
  %.cvtpack.p02263 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02259, float %.cvtpack.f12260), !dbg !26
  %.cvtpack.p12264 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22261, float %.cvtpack.f32262), !dbg !26
  %.cvtpack.v02265 = insertelement <2 x i32> poison, i32 %.cvtpack.p02263, i32 0, !dbg !26
  %.cvtpack.v12266 = insertelement <2 x i32> %.cvtpack.v02265, i32 %.cvtpack.p12264, i32 1, !dbg !26
  %623 = bitcast <2 x i32> %.cvtpack.v12266 to <4 x half>, !dbg !26
  %math.packed412 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %623) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %594, <4 x float> %592) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %609, <4 x float> %593) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %626 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = extractelement <4 x float> %lazy.score2179, i64 0, !dbg !24
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !24
  %local.reduce108 = fadd float %k0.sum.anchor2021, %628, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2029 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !26
  %630 = extractelement <4 x float> %lazy.score2179, i64 1, !dbg !24
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !24
  %local.reduce109 = fadd float %k0.sum.anchor2022, %631, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2030 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %632 = insertelement <4 x float> %629, float %631, i64 1, !dbg !26
  %633 = extractelement <4 x float> %lazy.score2179, i64 2, !dbg !24
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !24
  %local.reduce110 = fadd float %k0.sum.anchor2023, %634, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2031 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %635 = insertelement <4 x float> %632, float %634, i64 2, !dbg !26
  %636 = extractelement <4 x float> %lazy.score2179, i64 3, !dbg !24
  %637 = call float @llvm.amdgcn.exp2.f32(float %636), !dbg !24
  %local.reduce111 = fadd float %k0.sum.anchor2024, %637, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2032 = call float asm sideeffect "", "=v,0"(float %local.reduce111) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %638 = insertelement <4 x float> %635, float %637, i64 3, !dbg !26
  %math.result397 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %638) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02267 = extractelement <4 x float> %math.result397, i32 0, !dbg !26
  %.cvtpack.f12268 = extractelement <4 x float> %math.result397, i32 1, !dbg !26
  %.cvtpack.f22269 = extractelement <4 x float> %math.result397, i32 2, !dbg !26
  %.cvtpack.f32270 = extractelement <4 x float> %math.result397, i32 3, !dbg !26
  %.cvtpack.p02271 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02267, float %.cvtpack.f12268), !dbg !26
  %.cvtpack.p12272 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22269, float %.cvtpack.f32270), !dbg !26
  %.cvtpack.v02273 = insertelement <2 x i32> poison, i32 %.cvtpack.p02271, i32 0, !dbg !26
  %.cvtpack.v12274 = insertelement <2 x i32> %.cvtpack.v02273, i32 %.cvtpack.p12272, i32 1, !dbg !26
  %639 = bitcast <2 x i32> %.cvtpack.v12274 to <4 x half>, !dbg !26
  %math.packed398 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %639) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %626, <4 x float> %608) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %641 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %641, <4 x float> %610) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %643 = extractelement <4 x float> %lazy.score2187, i64 0, !dbg !24
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !24
  %local.reduce77 = fadd float %k0.sum.anchor2025, %644, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2033 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %645 = insertelement <4 x float> poison, float %644, i64 0, !dbg !26
  %646 = extractelement <4 x float> %lazy.score2187, i64 1, !dbg !24
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !24
  %local.reduce78 = fadd float %k0.sum.anchor2026, %647, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2034 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %648 = insertelement <4 x float> %645, float %647, i64 1, !dbg !26
  %649 = extractelement <4 x float> %lazy.score2187, i64 2, !dbg !24
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !24
  %local.reduce79 = fadd float %k0.sum.anchor2027, %650, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2035 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %651 = insertelement <4 x float> %648, float %650, i64 2, !dbg !26
  %652 = extractelement <4 x float> %lazy.score2187, i64 3, !dbg !24
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !24
  %local.reduce80 = fadd float %k0.sum.anchor2028, %653, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2036 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %654 = insertelement <4 x float> %651, float %653, i64 3, !dbg !26
  %math.result413 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02275 = extractelement <4 x float> %math.result413, i32 0, !dbg !26
  %.cvtpack.f12276 = extractelement <4 x float> %math.result413, i32 1, !dbg !26
  %.cvtpack.f22277 = extractelement <4 x float> %math.result413, i32 2, !dbg !26
  %.cvtpack.f32278 = extractelement <4 x float> %math.result413, i32 3, !dbg !26
  %.cvtpack.p02279 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02275, float %.cvtpack.f12276), !dbg !26
  %.cvtpack.p12280 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22277, float %.cvtpack.f32278), !dbg !26
  %.cvtpack.v02281 = insertelement <2 x i32> poison, i32 %.cvtpack.p02279, i32 0, !dbg !26
  %.cvtpack.v12282 = insertelement <2 x i32> %.cvtpack.v02281, i32 %.cvtpack.p12280, i32 1, !dbg !26
  %655 = bitcast <2 x i32> %.cvtpack.v12282 to <4 x half>, !dbg !26
  %math.packed414 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %626, <4 x float> %624) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %641, <4 x float> %625) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %658 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = extractelement <4 x float> %lazy.score2180, i64 0, !dbg !24
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !24
  %local.reduce112 = fadd float %k0.sum.anchor2029, %660, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2037 = call float asm sideeffect "", "=v,0"(float %local.reduce112) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %661 = insertelement <4 x float> poison, float %660, i64 0, !dbg !26
  %662 = extractelement <4 x float> %lazy.score2180, i64 1, !dbg !24
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !24
  %local.reduce113 = fadd float %k0.sum.anchor2030, %663, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2038 = call float asm sideeffect "", "=v,0"(float %local.reduce113) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %664 = insertelement <4 x float> %661, float %663, i64 1, !dbg !26
  %665 = extractelement <4 x float> %lazy.score2180, i64 2, !dbg !24
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !24
  %local.reduce114 = fadd float %k0.sum.anchor2031, %666, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2039 = call float asm sideeffect "", "=v,0"(float %local.reduce114) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> %664, float %666, i64 2, !dbg !26
  %668 = extractelement <4 x float> %lazy.score2180, i64 3, !dbg !24
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !24
  %local.reduce115 = fadd float %k0.sum.anchor2032, %669, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2040 = call float asm sideeffect "", "=v,0"(float %local.reduce115) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %667, float %669, i64 3, !dbg !26
  %math.result399 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02283 = extractelement <4 x float> %math.result399, i32 0, !dbg !26
  %.cvtpack.f12284 = extractelement <4 x float> %math.result399, i32 1, !dbg !26
  %.cvtpack.f22285 = extractelement <4 x float> %math.result399, i32 2, !dbg !26
  %.cvtpack.f32286 = extractelement <4 x float> %math.result399, i32 3, !dbg !26
  %.cvtpack.p02287 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02283, float %.cvtpack.f12284), !dbg !26
  %.cvtpack.p12288 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22285, float %.cvtpack.f32286), !dbg !26
  %.cvtpack.v02289 = insertelement <2 x i32> poison, i32 %.cvtpack.p02287, i32 0, !dbg !26
  %.cvtpack.v12290 = insertelement <2 x i32> %.cvtpack.v02289, i32 %.cvtpack.p12288, i32 1, !dbg !26
  %671 = bitcast <2 x i32> %.cvtpack.v12290 to <4 x half>, !dbg !26
  %math.packed400 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %658, <4 x float> %640) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %673 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %673, <4 x float> %642) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %675 = extractelement <4 x float> %lazy.score2188, i64 0, !dbg !24
  %676 = call float @llvm.amdgcn.exp2.f32(float %675), !dbg !24
  %local.reduce81 = fadd float %k0.sum.anchor2033, %676, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2041 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %677 = insertelement <4 x float> poison, float %676, i64 0, !dbg !26
  %678 = extractelement <4 x float> %lazy.score2188, i64 1, !dbg !24
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !24
  %local.reduce82 = fadd float %k0.sum.anchor2034, %679, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2042 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %680 = insertelement <4 x float> %677, float %679, i64 1, !dbg !26
  %681 = extractelement <4 x float> %lazy.score2188, i64 2, !dbg !24
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !24
  %local.reduce83 = fadd float %k0.sum.anchor2035, %682, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2043 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %683 = insertelement <4 x float> %680, float %682, i64 2, !dbg !26
  %684 = extractelement <4 x float> %lazy.score2188, i64 3, !dbg !24
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !24
  %local.reduce84 = fadd float %k0.sum.anchor2036, %685, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2044 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %686 = insertelement <4 x float> %683, float %685, i64 3, !dbg !26
  %math.result415 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %686) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02291 = extractelement <4 x float> %math.result415, i32 0, !dbg !26
  %.cvtpack.f12292 = extractelement <4 x float> %math.result415, i32 1, !dbg !26
  %.cvtpack.f22293 = extractelement <4 x float> %math.result415, i32 2, !dbg !26
  %.cvtpack.f32294 = extractelement <4 x float> %math.result415, i32 3, !dbg !26
  %.cvtpack.p02295 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02291, float %.cvtpack.f12292), !dbg !26
  %.cvtpack.p12296 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22293, float %.cvtpack.f32294), !dbg !26
  %.cvtpack.v02297 = insertelement <2 x i32> poison, i32 %.cvtpack.p02295, i32 0, !dbg !26
  %.cvtpack.v12298 = insertelement <2 x i32> %.cvtpack.v02297, i32 %.cvtpack.p12296, i32 1, !dbg !26
  %687 = bitcast <2 x i32> %.cvtpack.v12298 to <4 x half>, !dbg !26
  %math.packed416 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %687) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %658, <4 x float> %656) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %673, <4 x float> %657) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %690 = bitcast <4 x i32> %465 to <8 x half>, !dbg !14
  %691 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %692 = extractelement <4 x float> %lazy.score2181, i64 0, !dbg !24
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !24
  %local.reduce116 = fadd float %k0.sum.anchor2037, %693, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2045 = call float asm sideeffect "", "=v,0"(float %local.reduce116) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %694 = insertelement <4 x float> poison, float %693, i64 0, !dbg !26
  %695 = extractelement <4 x float> %lazy.score2181, i64 1, !dbg !24
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !24
  %local.reduce117 = fadd float %k0.sum.anchor2038, %696, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2046 = call float asm sideeffect "", "=v,0"(float %local.reduce117) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce120 = fadd float %k0.sum.anchor2045, %k0.sum.anchor2046, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2047 = call float asm sideeffect "", "=v,0"(float %local.reduce120) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %697 = insertelement <4 x float> %694, float %696, i64 1, !dbg !26
  %698 = extractelement <4 x float> %lazy.score2181, i64 2, !dbg !24
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !24
  %local.reduce118 = fadd float %k0.sum.anchor2039, %699, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2048 = call float asm sideeffect "", "=v,0"(float %local.reduce118) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %700 = insertelement <4 x float> %697, float %699, i64 2, !dbg !26
  %701 = extractelement <4 x float> %lazy.score2181, i64 3, !dbg !24
  %702 = call float @llvm.amdgcn.exp2.f32(float %701), !dbg !24
  %local.reduce119 = fadd float %k0.sum.anchor2040, %702, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2049 = call float asm sideeffect "", "=v,0"(float %local.reduce119) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce121 = fadd float %k0.sum.anchor2048, %k0.sum.anchor2049, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2050 = call float asm sideeffect "", "=v,0"(float %local.reduce121) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %703 = fadd float %k0.sum.anchor2047, %k0.sum.anchor2050, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2051 = call float asm sideeffect "", "=v,0"(float %703) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %704 = fadd float %k0.sum.anchor2051, %440, !dbg !25
  %k0.sum.anchor2052 = call float asm sideeffect "", "=v,0"(float %704) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %705 = insertelement <1 x float> poison, float %k0.sum.anchor2052, i64 0, !dbg !6
  %706 = insertvalue [2 x <1 x float>] poison, <1 x float> %705, 0, !dbg !6
  %707 = insertelement <4 x float> %700, float %702, i64 3, !dbg !26
  %math.result401 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02299 = extractelement <4 x float> %math.result401, i32 0, !dbg !26
  %.cvtpack.f12300 = extractelement <4 x float> %math.result401, i32 1, !dbg !26
  %.cvtpack.f22301 = extractelement <4 x float> %math.result401, i32 2, !dbg !26
  %.cvtpack.f32302 = extractelement <4 x float> %math.result401, i32 3, !dbg !26
  %.cvtpack.p02303 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02299, float %.cvtpack.f12300), !dbg !26
  %.cvtpack.p12304 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22301, float %.cvtpack.f32302), !dbg !26
  %.cvtpack.v02305 = insertelement <2 x i32> poison, i32 %.cvtpack.p02303, i32 0, !dbg !26
  %.cvtpack.v12306 = insertelement <2 x i32> %.cvtpack.v02305, i32 %.cvtpack.p12304, i32 1, !dbg !26
  %708 = bitcast <2 x i32> %.cvtpack.v12306 to <4 x half>, !dbg !26
  %math.packed402 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %691, <4 x float> %672) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %710 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %710, <4 x float> %674) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %712 = extractelement <4 x float> %lazy.score2189, i64 0, !dbg !24
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !24
  %local.reduce85 = fadd float %k0.sum.anchor2041, %713, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2053 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %714 = insertelement <4 x float> poison, float %713, i64 0, !dbg !26
  %715 = extractelement <4 x float> %lazy.score2189, i64 1, !dbg !24
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !24
  %local.reduce86 = fadd float %k0.sum.anchor2042, %716, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2054 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce89 = fadd float %k0.sum.anchor2053, %k0.sum.anchor2054, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2055 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %717 = insertelement <4 x float> %714, float %716, i64 1, !dbg !26
  %718 = extractelement <4 x float> %lazy.score2189, i64 2, !dbg !24
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !24
  %local.reduce87 = fadd float %k0.sum.anchor2043, %719, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2056 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %720 = insertelement <4 x float> %717, float %719, i64 2, !dbg !26
  %721 = extractelement <4 x float> %lazy.score2189, i64 3, !dbg !24
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !24
  %local.reduce88 = fadd float %k0.sum.anchor2044, %722, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2057 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %local.reduce90 = fadd float %k0.sum.anchor2056, %k0.sum.anchor2057, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2058 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %723 = fadd float %k0.sum.anchor2055, %k0.sum.anchor2058, !dbg !25, !frisk.local_reduction !1
  %k0.sum.anchor2059 = call float asm sideeffect "", "=v,0"(float %723) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %724 = fadd float %k0.sum.anchor2059, %442, !dbg !25
  %k0.sum.anchor2060 = call float asm sideeffect "", "=v,0"(float %724) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %725 = insertelement <1 x float> poison, float %k0.sum.anchor2060, i64 0, !dbg !6
  %726 = insertvalue [2 x <1 x float>] %706, <1 x float> %725, 1, !dbg !6
  %727 = insertelement <4 x float> %720, float %722, i64 3, !dbg !26
  %math.result417 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02307 = extractelement <4 x float> %math.result417, i32 0, !dbg !26
  %.cvtpack.f12308 = extractelement <4 x float> %math.result417, i32 1, !dbg !26
  %.cvtpack.f22309 = extractelement <4 x float> %math.result417, i32 2, !dbg !26
  %.cvtpack.f32310 = extractelement <4 x float> %math.result417, i32 3, !dbg !26
  %.cvtpack.p02311 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02307, float %.cvtpack.f12308), !dbg !26
  %.cvtpack.p12312 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22309, float %.cvtpack.f32310), !dbg !26
  %.cvtpack.v02313 = insertelement <2 x i32> poison, i32 %.cvtpack.p02311, i32 0, !dbg !26
  %.cvtpack.v12314 = insertelement <2 x i32> %.cvtpack.v02313, i32 %.cvtpack.p12312, i32 1, !dbg !26
  %728 = bitcast <2 x i32> %.cvtpack.v12314 to <4 x half>, !dbg !26
  %math.packed418 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %691, <4 x float> %688) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %710, <4 x float> %689) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %731 = extractvalue [2 x <8 x float>] %lazy.state2193, 0, !dbg !27
  %732 = shufflevector <8 x float> %731, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %733 = shufflevector <8 x float> %731, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %734 = extractvalue [2 x <8 x float>] %lazy.state2193, 1, !dbg !27
  %735 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %736 = shufflevector <8 x float> %734, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1462 = add nsw i64 %195, %address.local937, !dbg !16
  %vdb.address1463 = trunc i64 %vdb.address1462 to i32, !dbg !16
  %vdb.address1464 = shl i32 %vdb.address1463, 1, !dbg !16
  %vdb.address1469 = trunc nuw i64 %195 to i32, !dbg !16
  %vdb.address1470 = or disjoint i32 %address.local944, %vdb.address1469, !dbg !16
  %vdb.address1471 = shl nuw nsw i32 %vdb.address1470, 1, !dbg !16
  %737 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %738 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %vdb.address1471, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !14
  %739 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %740 = bitcast <4 x i32> %739 to <8 x half>, !dbg !14
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local962, i64 8448, !dbg !14
  %741 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %742 = bitcast <4 x i32> %741 to <8 x half>, !dbg !14
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local962, i64 8704, !dbg !14
  %743 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %744 = bitcast <4 x i32> %743 to <8 x half>, !dbg !14
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local962, i64 8960, !dbg !14
  %745 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %746 = bitcast <4 x i32> %745 to <8 x half>, !dbg !14
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local953, i64 12288, !dbg !14
  %747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %748 = bitcast <4 x i32> %747 to <8 x half>, !dbg !14
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local962, i64 12544, !dbg !14
  %749 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %750 = bitcast <4 x i32> %749 to <8 x half>, !dbg !14
  %vdb.read1460 = getelementptr i8, ptr addrspace(3) %address.local962, i64 12800, !dbg !14
  %751 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %752 = bitcast <4 x i32> %751 to <8 x half>, !dbg !14
  %vdb.read1461 = getelementptr i8, ptr addrspace(3) %address.local962, i64 13056, !dbg !14
  %753 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %754 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %755 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %754, <4 x float> %732) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %756 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %756, <4 x float> %733) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %754, <4 x float> %735) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %756, <4 x float> %736) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %760 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %760, <4 x float> %755) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %762 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %762, <4 x float> %757) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %760, <4 x float> %758) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %762, <4 x float> %759) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %766 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %766, <4 x float> %761) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %768 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %768, <4 x float> %763) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %766, <4 x float> %764) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %768, <4 x float> %765) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %772 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %772, <4 x float> %767) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %774 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %774, <4 x float> %769) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %772, <4 x float> %770) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %774, <4 x float> %771) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %778, <4 x float> %773) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %780 = shufflevector <8 x half> %748, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %780, <4 x float> %775) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %778, <4 x float> %776) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %780, <4 x float> %777) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %784 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %784, <4 x float> %779) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %786 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %786, <4 x float> %781) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %788 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %784, <4 x float> %782) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %786, <4 x float> %783) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %790 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %790, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %792 = shufflevector <8 x half> %752, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %792, <4 x float> %787) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %790, <4 x float> %788) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %792, <4 x float> %789) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = bitcast <4 x i32> %753 to <8 x half>, !dbg !14
  %797 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %797, <4 x float> %791) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %799, <4 x float> %793) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %797, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %799, <4 x float> %795) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %803 = extractvalue [2 x <8 x float>] %lazy.state2192, 0, !dbg !27
  %804 = shufflevector <8 x float> %803, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %805 = shufflevector <8 x float> %803, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %806 = extractvalue [2 x <8 x float>] %lazy.state2192, 1, !dbg !27
  %807 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %808 = shufflevector <8 x float> %806, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1476 = add nsw i64 %248, %address.local937, !dbg !16
  %vdb.address1477 = trunc i64 %vdb.address1476 to i32, !dbg !16
  %vdb.address1478 = shl i32 %vdb.address1477, 1, !dbg !16
  %vdb.address1483 = trunc nuw i64 %248 to i32, !dbg !16
  %vdb.address1484 = or disjoint i32 %address.local944, %vdb.address1483, !dbg !16
  %vdb.address1485 = shl nuw nsw i32 %vdb.address1484, 1, !dbg !16
  %809 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1478, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %810 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1454, i32 %vdb.address1485, <4 x i32> %dma.buffer.rsrc1361) #10, !dbg !16
  %811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %812 = bitcast <4 x i32> %811 to <8 x half>, !dbg !14
  %813 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %814 = bitcast <4 x i32> %813 to <8 x half>, !dbg !14
  %815 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %816 = bitcast <4 x i32> %815 to <8 x half>, !dbg !14
  %817 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %818 = bitcast <4 x i32> %817 to <8 x half>, !dbg !14
  %819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %820 = bitcast <4 x i32> %819 to <8 x half>, !dbg !14
  %821 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %822 = bitcast <4 x i32> %821 to <8 x half>, !dbg !14
  %823 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !14
  %825 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local985, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %826 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %826, <4 x float> %804) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %828 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %828, <4 x float> %805) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %830 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %826, <4 x float> %807) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %828, <4 x float> %808) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %832 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %832, <4 x float> %827) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %834, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %832, <4 x float> %830) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %834, <4 x float> %831) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %838, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %840, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %838, <4 x float> %836) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %840, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %844, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %846, <4 x float> %841) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %844, <4 x float> %842) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %846, <4 x float> %843) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %850, <4 x float> %845) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %852, <4 x float> %847) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %850, <4 x float> %848) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %852, <4 x float> %849) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %856 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %856, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %858, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %856, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %858, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %862 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %862, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %864, <4 x float> %859) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %862, <4 x float> %860) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %864, <4 x float> %861) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = bitcast <4 x i32> %825 to <8 x half>, !dbg !14
  %869 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %869, <4 x float> %863) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %871, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %869, <4 x float> %866) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %871, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %875 = extractvalue [2 x <8 x float>] %lazy.state2191, 0, !dbg !27
  %876 = shufflevector <8 x float> %875, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %877 = shufflevector <8 x float> %875, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %878 = extractvalue [2 x <8 x float>] %lazy.state2191, 1, !dbg !27
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
  %k0.next.2071 = or disjoint i64 %111, %k0.next., !dbg !13
  %k0.next.kdma.source1516 = or disjoint i64 %k0.next.2071, %kdma.source1515, !dbg !13
  %k0.next.kdma.source1517 = trunc nuw i64 %k0.next.kdma.source1516 to i32, !dbg !13
  %k0.next.kdma.source1518 = shl nuw nsw i32 %k0.next.kdma.source1517, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1518, %kdma.intra, !dbg !13
  %k0.next.2072 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1536 = or disjoint i64 %k0.next.2071, %kdma.source1535, !dbg !13
  %k0.next.kdma.source1537 = trunc nuw i64 %k0.next.kdma.source1536 to i32, !dbg !13
  %k0.next.kdma.source1538 = shl nuw nsw i32 %k0.next.kdma.source1537, 1, !dbg !13
  %k0.next.kdma.offset1539 = or disjoint i32 %k0.next.kdma.source1538, %kdma.intra.hi, !dbg !13
  %k0.next.2073 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1540, i32 %k0.next.kdma.offset1539, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1560 = or disjoint i64 %k0.next.2071, %kdma.source1559, !dbg !13
  %k0.next.kdma.source1561 = trunc nuw i64 %k0.next.kdma.source1560 to i32, !dbg !13
  %k0.next.kdma.source1562 = shl nuw nsw i32 %k0.next.kdma.source1561, 1, !dbg !13
  %k0.next.kdma.offset1563 = add nuw i32 %k0.next.kdma.source1562, %kdma.intra, !dbg !13
  %k0.next.2074 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.next.kdma.offset1563, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1584 = or disjoint i64 %k0.next.2071, %kdma.source1583, !dbg !13
  %k0.next.kdma.source1585 = trunc nuw i64 %k0.next.kdma.source1584 to i32, !dbg !13
  %k0.next.kdma.source1586 = shl nuw nsw i32 %k0.next.kdma.source1585, 1, !dbg !13
  %k0.next.kdma.offset1587 = add nuw i32 %k0.next.kdma.source1586, %kdma.intra.hi, !dbg !13
  %k0.next.2075 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1588, i32 %k0.next.kdma.offset1587, <4 x i32> %dma.buffer.rsrc1377) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %882 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !14
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !14
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !14
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !14
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %891 = bitcast <4 x i32> %890 to <8 x half>, !dbg !14
  %892 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %893 = bitcast <4 x i32> %892 to <8 x half>, !dbg !14
  %894 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %895 = bitcast <4 x i32> %894 to <8 x half>, !dbg !14
  %896 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %897 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %897, <4 x float> %876) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %899, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %897, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %899, <4 x float> %880) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %903, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %905, <4 x float> %900) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %903, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %905, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %909, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %911, <4 x float> %906) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %909, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %911, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %915, <4 x float> %910) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %917, <4 x float> %912) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %915, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %917, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %921, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = shufflevector <8 x half> %891, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %923, <4 x float> %918) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %921, <4 x float> %919) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %923, <4 x float> %920) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %927, <4 x float> %922) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = shufflevector <8 x half> %893, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %929, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %927, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %929, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %933, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %935 = shufflevector <8 x half> %895, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %935, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %933, <4 x float> %931) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %935, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = bitcast <4 x i32> %896 to <8 x half>, !dbg !14
  %940 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %940, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %942 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %942, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %940, <4 x float> %937) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %942, <4 x float> %938) #9, !dbg !14, !frisk.pv_interleave !31
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
  %lane.sum.field187 = extractvalue [2 x <1 x float>] %103, 1, !dbg !17
  %lane.sum.local188 = extractelement <1 x float> %lane.sum.field187, i64 0, !dbg !17
  %963 = shl nuw nsw i32 %8, 2, !dbg !17
  %964 = xor i32 %963, 64, !dbg !17
  %965 = bitcast float %lane.sum.local188 to i32, !dbg !17
  %lane.sum.shuffle191 = call i32 @llvm.amdgcn.ds.bpermute(i32 %964, i32 %965), !dbg !17
  %966 = bitcast i32 %lane.sum.shuffle191 to float, !dbg !17
  %lane.sum.final192 = fadd float %lane.sum.local188, %966, !dbg !17
  %967 = xor i32 %963, 128, !dbg !17
  %968 = bitcast float %lane.sum.final192 to i32, !dbg !17
  %lane.sum.shuffle193 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %968), !dbg !17
  %969 = bitcast i32 %lane.sum.shuffle193 to float, !dbg !17
  %lane.sum.final194 = fadd float %lane.sum.final192, %969, !dbg !17
  %970 = insertelement <1 x float> poison, float %lane.sum.final194, i64 0, !dbg !17
  %lane.sum.field = extractvalue [2 x <1 x float>] %103, 0, !dbg !17
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !17
  %971 = bitcast float %lane.sum.local to i32, !dbg !17
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %964, i32 %971), !dbg !17
  %972 = bitcast i32 %lane.sum.shuffle to float, !dbg !17
  %lane.sum.final = fadd float %lane.sum.local, %972, !dbg !17
  %973 = bitcast float %lane.sum.final to i32, !dbg !17
  %lane.sum.shuffle185 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %973), !dbg !17
  %974 = bitcast i32 %lane.sum.shuffle185 to float, !dbg !17
  %lane.sum.final186 = fadd float %lane.sum.final, %974, !dbg !17
  %975 = insertelement <1 x float> poison, float %lane.sum.final186, i64 0, !dbg !17
  %976 = shufflevector <1 x float> %975, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %977 = shufflevector <1 x float> %970, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %978 = extractvalue [2 x <8 x float>] %102, 1, !dbg !17
  %979 = shufflevector <8 x float> %978, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %977, !dbg !17
  %normalized = fmul <4 x float> %979, %row.reciprocal, !dbg !17
  %980 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %981 = extractelement <4 x half> %980, i64 3, !dbg !18
  %982 = extractelement <4 x half> %980, i64 2, !dbg !18
  %983 = extractelement <4 x half> %980, i64 1, !dbg !18
  %984 = extractelement <4 x half> %980, i64 0, !dbg !18
  %985 = shufflevector <8 x float> %978, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized40 = fmul <4 x float> %985, %row.reciprocal, !dbg !17
  %986 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !18
  %987 = extractelement <4 x half> %986, i64 3, !dbg !18
  %988 = extractelement <4 x half> %986, i64 2, !dbg !18
  %989 = extractelement <4 x half> %986, i64 1, !dbg !18
  %990 = extractelement <4 x half> %986, i64 0, !dbg !18
  %991 = extractvalue [2 x <8 x float>] %102, 0, !dbg !17
  %992 = shufflevector <8 x float> %991, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %976, !dbg !17
  %normalized42 = fmul <4 x float> %992, %row.reciprocal41, !dbg !17
  %993 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !18
  %994 = extractelement <4 x half> %993, i64 3, !dbg !18
  %995 = extractelement <4 x half> %993, i64 2, !dbg !18
  %996 = extractelement <4 x half> %993, i64 1, !dbg !18
  %997 = extractelement <4 x half> %993, i64 0, !dbg !18
  %998 = shufflevector <8 x float> %991, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized43 = fmul <4 x float> %998, %row.reciprocal41, !dbg !17
  %999 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !18
  %1000 = extractelement <4 x half> %999, i64 3, !dbg !18
  %1001 = extractelement <4 x half> %999, i64 2, !dbg !18
  %1002 = extractelement <4 x half> %999, i64 1, !dbg !18
  %1003 = extractelement <4 x half> %999, i64 0, !dbg !18
  %address.thread992 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local993 = zext nneg i32 %address.thread992 to i64, !dbg !9
  %address.local994 = add nuw i64 %32, %address.local993, !dbg !19
  %1004 = lshr i64 %address.local993, 2, !dbg !19
  %address.local996 = and i64 %1004, 48, !dbg !19
  %address.local997 = add i64 %address.local994, %address.local996, !dbg !19
  %address.local998 = and i64 %address.local993, 240, !dbg !19
  %address.local999 = sub i64 %address.local997, %address.local998, !dbg !19
  %address.local1000 = add i64 %address.local999, %33, !dbg !19
  %address.local1001 = shl nuw nsw i64 %address.local1000, 7, !dbg !19
  %1005 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1005, i64 %address.local1001, !dbg !19
  %address.local1004 = lshr i64 %address.local993, 4, !dbg !12
  %address.local1005 = and i64 %address.local1004, 3, !dbg !12
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1005, !dbg !19
  store half %1003, ptr addrspace(1) %address.local1006, align 2, !dbg !19
  %address.local1008 = or disjoint i64 %address.local1005, 4, !dbg !19
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1008, !dbg !19
  store half %1002, ptr addrspace(1) %address.local1009, align 2, !dbg !19
  %address.local1011 = or disjoint i64 %address.local1005, 8, !dbg !19
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1011, !dbg !19
  store half %1001, ptr addrspace(1) %address.local1012, align 2, !dbg !19
  %address.local1014 = or i64 %address.local1004, 12, !dbg !19
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1014, !dbg !19
  store half %1000, ptr addrspace(1) %address.local1015, align 2, !dbg !19
  %address.local1017 = or disjoint i64 %address.local1005, 16, !dbg !19
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1017, !dbg !19
  store half %997, ptr addrspace(1) %address.local1018, align 2, !dbg !19
  %address.local1020 = or disjoint i64 %address.local1005, 20, !dbg !19
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1020, !dbg !19
  store half %996, ptr addrspace(1) %address.local1021, align 2, !dbg !19
  %address.local1023 = or disjoint i64 %address.local1005, 24, !dbg !19
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1023, !dbg !19
  store half %995, ptr addrspace(1) %address.local1024, align 2, !dbg !19
  %address.local1026 = or i64 %address.local1004, 28, !dbg !19
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1026, !dbg !19
  store half %994, ptr addrspace(1) %address.local1027, align 2, !dbg !19
  %address.local1031 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1003, i64 16384, !dbg !19
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1005, !dbg !19
  store half %990, ptr addrspace(1) %address.local1032, align 2, !dbg !19
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1008, !dbg !19
  store half %989, ptr addrspace(1) %address.local1034, align 2, !dbg !19
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1011, !dbg !19
  store half %988, ptr addrspace(1) %address.local1036, align 2, !dbg !19
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1014, !dbg !19
  store half %987, ptr addrspace(1) %address.local1038, align 2, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1017, !dbg !19
  store half %984, ptr addrspace(1) %address.local1040, align 2, !dbg !19
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1020, !dbg !19
  store half %983, ptr addrspace(1) %address.local1042, align 2, !dbg !19
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1023, !dbg !19
  store half %982, ptr addrspace(1) %address.local1044, align 2, !dbg !19
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1026, !dbg !19
  store half %981, ptr addrspace(1) %address.local1046, align 2, !dbg !19
  %1006 = extractvalue [2 x <8 x float>] %105, 1, !dbg !17
  %1007 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %normalized45 = fmul <4 x float> %1007, %row.reciprocal, !dbg !17
  %1008 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !18
  %1009 = extractelement <4 x half> %1008, i64 3, !dbg !18
  %1010 = extractelement <4 x half> %1008, i64 2, !dbg !18
  %1011 = extractelement <4 x half> %1008, i64 1, !dbg !18
  %1012 = extractelement <4 x half> %1008, i64 0, !dbg !18
  %1013 = shufflevector <8 x float> %1006, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized46 = fmul <4 x float> %1013, %row.reciprocal, !dbg !17
  %1014 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !18
  %1015 = extractelement <4 x half> %1014, i64 3, !dbg !18
  %1016 = extractelement <4 x half> %1014, i64 2, !dbg !18
  %1017 = extractelement <4 x half> %1014, i64 1, !dbg !18
  %1018 = extractelement <4 x half> %1014, i64 0, !dbg !18
  %1019 = extractvalue [2 x <8 x float>] %105, 0, !dbg !17
  %1020 = shufflevector <8 x float> %1019, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %normalized48 = fmul <4 x float> %1020, %row.reciprocal41, !dbg !17
  %1021 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !18
  %1022 = extractelement <4 x half> %1021, i64 3, !dbg !18
  %1023 = extractelement <4 x half> %1021, i64 2, !dbg !18
  %1024 = extractelement <4 x half> %1021, i64 1, !dbg !18
  %1025 = extractelement <4 x half> %1021, i64 0, !dbg !18
  %1026 = shufflevector <8 x float> %1019, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized49 = fmul <4 x float> %1026, %row.reciprocal41, !dbg !17
  %1027 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !18
  %1028 = extractelement <4 x half> %1027, i64 3, !dbg !18
  %1029 = extractelement <4 x half> %1027, i64 2, !dbg !18
  %1030 = extractelement <4 x half> %1027, i64 1, !dbg !18
  %1031 = extractelement <4 x half> %1027, i64 0, !dbg !18
  %address.local1061 = or disjoint i64 %address.local1005, 32, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1061, !dbg !19
  store half %1031, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1064 = or disjoint i64 %address.local1005, 36, !dbg !19
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1064, !dbg !19
  store half %1030, ptr addrspace(1) %address.local1065, align 2, !dbg !19
  %address.local1067 = or disjoint i64 %address.local1005, 40, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1067, !dbg !19
  store half %1029, ptr addrspace(1) %address.local1068, align 2, !dbg !19
  %address.local1070 = or i64 %address.local1004, 44, !dbg !19
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1070, !dbg !19
  store half %1028, ptr addrspace(1) %address.local1071, align 2, !dbg !19
  %address.local1073 = or disjoint i64 %address.local1005, 48, !dbg !19
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1073, !dbg !19
  store half %1025, ptr addrspace(1) %address.local1074, align 2, !dbg !19
  %address.local1076 = or disjoint i64 %address.local1005, 52, !dbg !19
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1076, !dbg !19
  store half %1024, ptr addrspace(1) %address.local1077, align 2, !dbg !19
  %address.local1079 = or disjoint i64 %address.local1005, 56, !dbg !19
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1079, !dbg !19
  store half %1023, ptr addrspace(1) %address.local1080, align 2, !dbg !19
  %address.local1082 = or i64 %address.local1004, 60, !dbg !19
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1082, !dbg !19
  store half %1022, ptr addrspace(1) %address.local1083, align 2, !dbg !19
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1061, !dbg !19
  store half %1018, ptr addrspace(1) %address.local1088, align 2, !dbg !19
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1064, !dbg !19
  store half %1017, ptr addrspace(1) %address.local1090, align 2, !dbg !19
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1067, !dbg !19
  store half %1016, ptr addrspace(1) %address.local1092, align 2, !dbg !19
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1070, !dbg !19
  store half %1015, ptr addrspace(1) %address.local1094, align 2, !dbg !19
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1073, !dbg !19
  store half %1012, ptr addrspace(1) %address.local1096, align 2, !dbg !19
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1076, !dbg !19
  store half %1011, ptr addrspace(1) %address.local1098, align 2, !dbg !19
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1079, !dbg !19
  store half %1010, ptr addrspace(1) %address.local1100, align 2, !dbg !19
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1082, !dbg !19
  store half %1009, ptr addrspace(1) %address.local1102, align 2, !dbg !19
  %1032 = extractvalue [2 x <8 x float>] %106, 1, !dbg !17
  %1033 = shufflevector <8 x float> %1032, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %normalized51 = fmul <4 x float> %1033, %row.reciprocal, !dbg !17
  %1034 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !18
  %1035 = extractelement <4 x half> %1034, i64 3, !dbg !18
  %1036 = extractelement <4 x half> %1034, i64 2, !dbg !18
  %1037 = extractelement <4 x half> %1034, i64 1, !dbg !18
  %1038 = extractelement <4 x half> %1034, i64 0, !dbg !18
  %1039 = shufflevector <8 x float> %1032, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized52 = fmul <4 x float> %1039, %row.reciprocal, !dbg !17
  %1040 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !18
  %1041 = extractelement <4 x half> %1040, i64 3, !dbg !18
  %1042 = extractelement <4 x half> %1040, i64 2, !dbg !18
  %1043 = extractelement <4 x half> %1040, i64 1, !dbg !18
  %1044 = extractelement <4 x half> %1040, i64 0, !dbg !18
  %1045 = extractvalue [2 x <8 x float>] %106, 0, !dbg !17
  %1046 = shufflevector <8 x float> %1045, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %normalized54 = fmul <4 x float> %1046, %row.reciprocal41, !dbg !17
  %1047 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !18
  %1048 = extractelement <4 x half> %1047, i64 3, !dbg !18
  %1049 = extractelement <4 x half> %1047, i64 2, !dbg !18
  %1050 = extractelement <4 x half> %1047, i64 1, !dbg !18
  %1051 = extractelement <4 x half> %1047, i64 0, !dbg !18
  %1052 = shufflevector <8 x float> %1045, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized55 = fmul <4 x float> %1052, %row.reciprocal41, !dbg !17
  %1053 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !18
  %1054 = extractelement <4 x half> %1053, i64 3, !dbg !18
  %1055 = extractelement <4 x half> %1053, i64 2, !dbg !18
  %1056 = extractelement <4 x half> %1053, i64 1, !dbg !18
  %1057 = extractelement <4 x half> %1053, i64 0, !dbg !18
  %address.local1117 = or disjoint i64 %address.local1005, 64, !dbg !19
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1117, !dbg !19
  store half %1057, ptr addrspace(1) %address.local1118, align 2, !dbg !19
  %address.local1120 = or disjoint i64 %address.local1005, 68, !dbg !19
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1120, !dbg !19
  store half %1056, ptr addrspace(1) %address.local1121, align 2, !dbg !19
  %address.local1123 = or disjoint i64 %address.local1005, 72, !dbg !19
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1123, !dbg !19
  store half %1055, ptr addrspace(1) %address.local1124, align 2, !dbg !19
  %address.local1126 = or i64 %address.local1004, 76, !dbg !19
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1126, !dbg !19
  store half %1054, ptr addrspace(1) %address.local1127, align 2, !dbg !19
  %address.local1129 = or disjoint i64 %address.local1005, 80, !dbg !19
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1129, !dbg !19
  store half %1051, ptr addrspace(1) %address.local1130, align 2, !dbg !19
  %address.local1132 = or disjoint i64 %address.local1005, 84, !dbg !19
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1132, !dbg !19
  store half %1050, ptr addrspace(1) %address.local1133, align 2, !dbg !19
  %address.local1135 = or disjoint i64 %address.local1005, 88, !dbg !19
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1135, !dbg !19
  store half %1049, ptr addrspace(1) %address.local1136, align 2, !dbg !19
  %address.local1138 = or i64 %address.local1004, 92, !dbg !19
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1138, !dbg !19
  store half %1048, ptr addrspace(1) %address.local1139, align 2, !dbg !19
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1117, !dbg !19
  store half %1044, ptr addrspace(1) %address.local1144, align 2, !dbg !19
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1120, !dbg !19
  store half %1043, ptr addrspace(1) %address.local1146, align 2, !dbg !19
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1123, !dbg !19
  store half %1042, ptr addrspace(1) %address.local1148, align 2, !dbg !19
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1126, !dbg !19
  store half %1041, ptr addrspace(1) %address.local1150, align 2, !dbg !19
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1129, !dbg !19
  store half %1038, ptr addrspace(1) %address.local1152, align 2, !dbg !19
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1132, !dbg !19
  store half %1037, ptr addrspace(1) %address.local1154, align 2, !dbg !19
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1135, !dbg !19
  store half %1036, ptr addrspace(1) %address.local1156, align 2, !dbg !19
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1138, !dbg !19
  store half %1035, ptr addrspace(1) %address.local1158, align 2, !dbg !19
  %1058 = extractvalue [2 x <8 x float>] %107, 1, !dbg !17
  %1059 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %normalized57 = fmul <4 x float> %1059, %row.reciprocal, !dbg !17
  %1060 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !18
  %1061 = extractelement <4 x half> %1060, i64 3, !dbg !18
  %1062 = extractelement <4 x half> %1060, i64 2, !dbg !18
  %1063 = extractelement <4 x half> %1060, i64 1, !dbg !18
  %1064 = extractelement <4 x half> %1060, i64 0, !dbg !18
  %1065 = shufflevector <8 x float> %1058, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized58 = fmul <4 x float> %1065, %row.reciprocal, !dbg !17
  %1066 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !18
  %1067 = extractelement <4 x half> %1066, i64 3, !dbg !18
  %1068 = extractelement <4 x half> %1066, i64 2, !dbg !18
  %1069 = extractelement <4 x half> %1066, i64 1, !dbg !18
  %1070 = extractelement <4 x half> %1066, i64 0, !dbg !18
  %1071 = extractvalue [2 x <8 x float>] %107, 0, !dbg !17
  %1072 = shufflevector <8 x float> %1071, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !17
  %normalized60 = fmul <4 x float> %1072, %row.reciprocal41, !dbg !17
  %1073 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !18
  %1074 = extractelement <4 x half> %1073, i64 3, !dbg !18
  %1075 = extractelement <4 x half> %1073, i64 2, !dbg !18
  %1076 = extractelement <4 x half> %1073, i64 1, !dbg !18
  %1077 = extractelement <4 x half> %1073, i64 0, !dbg !18
  %1078 = shufflevector <8 x float> %1071, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !17
  %normalized61 = fmul <4 x float> %1078, %row.reciprocal41, !dbg !17
  %1079 = fptrunc <4 x float> %normalized61 to <4 x half>, !dbg !18
  %1080 = extractelement <4 x half> %1079, i64 3, !dbg !18
  %1081 = extractelement <4 x half> %1079, i64 2, !dbg !18
  %1082 = extractelement <4 x half> %1079, i64 1, !dbg !18
  %1083 = extractelement <4 x half> %1079, i64 0, !dbg !18
  %address.local1173 = or disjoint i64 %address.local1005, 96, !dbg !19
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1173, !dbg !19
  store half %1083, ptr addrspace(1) %address.local1174, align 2, !dbg !19
  %address.local1176 = or disjoint i64 %address.local1005, 100, !dbg !19
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1176, !dbg !19
  store half %1082, ptr addrspace(1) %address.local1177, align 2, !dbg !19
  %address.local1179 = or disjoint i64 %address.local1005, 104, !dbg !19
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1179, !dbg !19
  store half %1081, ptr addrspace(1) %address.local1180, align 2, !dbg !19
  %address.local1182 = or i64 %address.local1004, 108, !dbg !19
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1182, !dbg !19
  store half %1080, ptr addrspace(1) %address.local1183, align 2, !dbg !19
  %address.local1185 = or disjoint i64 %address.local1005, 112, !dbg !19
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1185, !dbg !19
  store half %1077, ptr addrspace(1) %address.local1186, align 2, !dbg !19
  %address.local1188 = or disjoint i64 %address.local1005, 116, !dbg !19
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1188, !dbg !19
  store half %1076, ptr addrspace(1) %address.local1189, align 2, !dbg !19
  %address.local1191 = or disjoint i64 %address.local1005, 120, !dbg !19
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1191, !dbg !19
  store half %1075, ptr addrspace(1) %address.local1192, align 2, !dbg !19
  %address.local1194 = or i64 %address.local1004, 124, !dbg !19
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1194, !dbg !19
  store half %1074, ptr addrspace(1) %address.local1195, align 2, !dbg !19
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1173, !dbg !19
  store half %1070, ptr addrspace(1) %address.local1200, align 2, !dbg !19
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1176, !dbg !19
  store half %1069, ptr addrspace(1) %address.local1202, align 2, !dbg !19
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1179, !dbg !19
  store half %1068, ptr addrspace(1) %address.local1204, align 2, !dbg !19
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1182, !dbg !19
  store half %1067, ptr addrspace(1) %address.local1206, align 2, !dbg !19
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1185, !dbg !19
  store half %1064, ptr addrspace(1) %address.local1208, align 2, !dbg !19
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1188, !dbg !19
  store half %1063, ptr addrspace(1) %address.local1210, align 2, !dbg !19
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1191, !dbg !19
  store half %1062, ptr addrspace(1) %address.local1212, align 2, !dbg !19
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1194, !dbg !19
  store half %1061, ptr addrspace(1) %address.local1214, align 2, !dbg !19
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #8

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

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
attributes #7 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind }
attributes #11 = { memory(inaccessiblemem: readwrite) }

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
