; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_early_wave/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_early_wave/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
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
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !9, !range !32, !frisk.register_lifetimes !1
  %36 = shl nuw nsw i32 %address.thread, 3, !dbg !9
  %37 = and i32 %36, 1536, !dbg !9
  %address.local421 = zext nneg i32 %37 to i64, !dbg !9
  %address.local422 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local421, !dbg !9
  %address.local423 = ptrtoint ptr addrspace(3) %address.local422 to i64, !dbg !9
  %address.local424 = trunc i64 %address.local423 to i32, !dbg !9
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local424), !dbg !9
  %address.local425 = lshr i32 %address.thread, 4, !dbg !9
  %address.local426 = and i32 %address.local425, 8, !dbg !9
  %address.local427 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local428 = and i32 %address.local427, 16256, !dbg !9
  %address.local429 = or disjoint i32 %address.local426, %address.local428, !dbg !9
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1395 = shl i64 %34, 8, !dbg !9
  %40 = add i64 %.idx, %.idx1395, !dbg !9
  %41 = shl nuw nsw i32 %address.local429, 1, !dbg !9
  %42 = trunc i64 %40 to i32, !dbg !9
  %dma.buffer.offset1393 = or disjoint i32 %41, %42, !dbg !9
  %43 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !9
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1394 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %address.local434 = or disjoint i64 %address.local421, 2048, !dbg !9
  %address.local435 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local434, !dbg !9
  %address.local436 = ptrtoint ptr addrspace(3) %address.local435 to i64, !dbg !9
  %address.local437 = trunc i64 %address.local436 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local437), !dbg !9
  %address.local443 = or disjoint i32 %address.local429, 16, !dbg !9
  %47 = shl nuw nsw i32 %address.local443, 1, !dbg !9
  %dma.buffer.offset1391 = or disjoint i32 %47, %42, !dbg !9
  %48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1391, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %49 = shl nuw nsw i64 %31, 19, !dbg !9
  %50 = add i64 %49, %35, !dbg !9
  %address.local446 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local421, !dbg !9
  %address.local447 = ptrtoint ptr addrspace(3) %address.local446 to i64, !dbg !9
  %address.local448 = trunc i64 %address.local447 to i32, !dbg !9
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local448), !dbg !9
  %52 = trunc i64 %50 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local429, %52, !dbg !9
  %53 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1389 = or disjoint i32 %53, 64, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1389, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local434, !dbg !9
  %address.local451 = ptrtoint ptr addrspace(3) %address.local450 to i64, !dbg !9
  %address.local452 = trunc i64 %address.local451 to i32, !dbg !9
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local452), !dbg !9
  %.tr1399 = or disjoint i32 %address.local443, %52, !dbg !9
  %56 = shl i32 %.tr1399, 1, !dbg !9
  %dma.buffer.offset1387 = or disjoint i32 %56, 64, !dbg !9
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %address.local454 = lshr i32 %address.thread, 2, !dbg !12
  %address.local455 = and i32 %address.local454, 4, !dbg !12
  %address.local456 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local457 = and i32 %address.local456, 384, !dbg !12
  %address.local458 = or disjoint i32 %address.local455, %address.local457, !dbg !12
  %address.local460 = and i32 %36, 120, !dbg !12
  %address.local461 = or disjoint i32 %address.local458, %address.local460, !dbg !12
  %address.local462 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local463 = and i32 %address.local462, 1024, !dbg !12
  %address.local464 = or disjoint i32 %address.local461, %address.local463, !dbg !12
  %address.local465 = zext nneg i32 %address.local464 to i64, !dbg !12
  %address.local466 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local465, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local466, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local466, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local466, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local466, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1385 = or disjoint i32 %53, 128, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1385, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %dma.buffer.offset1383 = or disjoint i32 %56, 128, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1383, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %address.local466, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %address.local466, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local466, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1221 = getelementptr i8, ptr addrspace(3) %address.local466, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1381 = or disjoint i32 %53, 192, !dbg !9
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %dma.buffer.offset1379 = or disjoint i32 %56, 192, !dbg !9
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1379, <4 x i32> %dma.buffer.rsrc1394) #10, !dbg !9
  %78 = load <4 x half>, ptr addrspace(3) %address.local466, align 8, !dbg !12
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !12
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !12
  %83 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !12
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !12
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !12
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !12
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1221, align 8, !dbg !12
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
  %k0.prime.2062 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.2063 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.2064 = or disjoint i64 %k0.prime.2062, %k0.prime.2063, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1501 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1502 = shl nuw nsw i64 %k0.prime.kdma.source1501, 9, !dbg !13
  %k0.prime.kdma.source1503 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !13
  %k0.prime.kdma.source1504 = add nuw nsw i64 %k0.prime.kdma.source1502, %k0.prime.kdma.source1503, !dbg !13
  %k0.prime.kdma.source1505 = lshr i64 %k0.prime.kdma.source, 5, !dbg !13
  %k0.prime.kdma.source1506 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, 1, !dbg !13
  %k0.prime.kdma.source1508 = and i64 %k0.prime.kdma.source1501, %k0.prime.kdma.source1507, !dbg !13
  %k0.prime.kdma.source1509 = or disjoint i64 %k0.prime.kdma.source1505, %k0.prime.kdma.source1508, !dbg !13
  %k0.prime.kdma.source1510 = shl nuw nsw i64 %k0.prime.kdma.source1509, 10, !dbg !13
  %k0.prime.kdma.source1511 = sub nsw i64 %k0.prime.kdma.source1504, %k0.prime.kdma.source1510, !dbg !13
  %k0.prime.kdma.source1512 = and i64 %k0.prime.kdma.source1511, 16256, !dbg !13
  %k0.prime.kdma.source1513 = lshr i32 %kdma.tid, 4, !dbg !13
  %k0.prime.kdma.source1514 = and i32 %k0.prime.kdma.source1513, 8, !dbg !13
  %k0.prime.kdma.source1515 = zext nneg i32 %k0.prime.kdma.source1514 to i64, !dbg !13
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.kdma.source1512, %k0.prime.kdma.source1515, !dbg !13
  %k0.prime.kdma.source1517 = or disjoint i64 %k0.prime.2064, %k0.prime.kdma.source1516, !dbg !13
  %k0.prime.kdma.source1518 = trunc nuw i64 %k0.prime.kdma.source1517 to i32, !dbg !13
  %k0.prime.kdma.source1519 = shl nuw nsw i32 %k0.prime.kdma.source1518, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1519, %kdma.intra, !dbg !13
  %k0.prime.2065 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.2066 = insertelement <2 x i64> poison, i64 %k0.prime.2065, i64 0, !dbg !13
  %k0.prime.2067 = bitcast <2 x i64> %k0.prime.2066 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1378 = shufflevector <4 x i32> %k0.prime.2067, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.2068 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1541 = add i32 %38, 512, !dbg !13
  %k0.prime.kdma.source1520 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1521 = lshr i64 %k0.prime.kdma.source1520, 4, !dbg !13
  %k0.prime.kdma.source1522 = shl nuw nsw i64 %k0.prime.kdma.source1521, 9, !dbg !13
  %k0.prime.kdma.source1523 = shl nuw nsw i64 %k0.prime.kdma.source1520, 7, !dbg !13
  %k0.prime.kdma.source1524 = add nuw nsw i64 %k0.prime.kdma.source1522, %k0.prime.kdma.source1523, !dbg !13
  %k0.prime.kdma.source1525 = lshr i64 %k0.prime.kdma.source1520, 5, !dbg !13
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source1520, 2, !dbg !13
  %k0.prime.kdma.source1527 = and i64 %k0.prime.kdma.source1526, 1, !dbg !13
  %k0.prime.kdma.source1528 = and i64 %k0.prime.kdma.source1521, %k0.prime.kdma.source1527, !dbg !13
  %k0.prime.kdma.source1529 = add nuw nsw i64 %k0.prime.kdma.source1525, %k0.prime.kdma.source1528, !dbg !13
  %k0.prime.kdma.source1530 = shl nuw nsw i64 %k0.prime.kdma.source1529, 10, !dbg !13
  %k0.prime.kdma.source1531 = sub nsw i64 %k0.prime.kdma.source1524, %k0.prime.kdma.source1530, !dbg !13
  %k0.prime.kdma.source1532 = and i64 %k0.prime.kdma.source1531, 16256, !dbg !13
  %k0.prime.kdma.source1536 = or disjoint i64 %k0.prime.kdma.source1532, %k0.prime.kdma.source1515, !dbg !13
  %k0.prime.kdma.source1537 = or disjoint i64 %k0.prime.2064, %k0.prime.kdma.source1536, !dbg !13
  %k0.prime.kdma.source1538 = trunc nuw i64 %k0.prime.kdma.source1537 to i32, !dbg !13
  %k0.prime.kdma.source1539 = shl nuw nsw i32 %k0.prime.kdma.source1538, 1, !dbg !13
  %k0.prime.kdma.offset1540 = or disjoint i32 %k0.prime.kdma.source1539, %kdma.intra.hi, !dbg !13
  %k0.prime.2069 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1541, i32 %k0.prime.kdma.offset1540, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1543 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !13
  %k0.prime.kdma.source1544 = lshr i64 %k0.prime.kdma.source1543, 4, !dbg !13
  %k0.prime.kdma.source1545 = shl nuw nsw i64 %k0.prime.kdma.source1544, 9, !dbg !13
  %k0.prime.kdma.source1547 = add nuw nsw i64 %k0.prime.kdma.source1545, %k0.prime.kdma.source1503, !dbg !13
  %k0.prime.kdma.source1548 = lshr i64 %k0.prime.kdma.source1543, 5, !dbg !13
  %k0.prime.kdma.source1551 = and i64 %k0.prime.kdma.source1544, %k0.prime.kdma.source1507, !dbg !13
  %k0.prime.kdma.source1552 = or disjoint i64 %k0.prime.kdma.source1548, %k0.prime.kdma.source1551, !dbg !13
  %k0.prime.kdma.source1553 = shl nuw nsw i64 %k0.prime.kdma.source1552, 10, !dbg !13
  %k0.prime.kdma.source1554 = sub nsw i64 %k0.prime.kdma.source1547, %k0.prime.kdma.source1553, !dbg !13
  %k0.prime.kdma.source1555 = and i64 %k0.prime.kdma.source1554, 16256, !dbg !13
  %k0.prime.kdma.source1558 = or disjoint i32 %k0.prime.kdma.source1514, 16, !dbg !13
  %k0.prime.kdma.source1559 = zext nneg i32 %k0.prime.kdma.source1558 to i64, !dbg !13
  %k0.prime.kdma.source1560 = or disjoint i64 %k0.prime.kdma.source1555, %k0.prime.kdma.source1559, !dbg !13
  %k0.prime.kdma.source1561 = or disjoint i64 %k0.prime.2064, %k0.prime.kdma.source1560, !dbg !13
  %k0.prime.kdma.source1562 = trunc nuw i64 %k0.prime.kdma.source1561 to i32, !dbg !13
  %k0.prime.kdma.source1563 = shl nuw nsw i32 %k0.prime.kdma.source1562, 1, !dbg !13
  %k0.prime.kdma.offset1564 = add nuw i32 %k0.prime.kdma.source1563, %kdma.intra, !dbg !13
  %k0.prime.2070 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.prime.kdma.offset1564, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1589 = add i32 %46, 512, !dbg !13
  %k0.prime.kdma.source1567 = or disjoint i64 %k0.prime.kdma.source1520, 256, !dbg !13
  %k0.prime.kdma.source1568 = lshr i64 %k0.prime.kdma.source1567, 4, !dbg !13
  %k0.prime.kdma.source1569 = shl nuw nsw i64 %k0.prime.kdma.source1568, 9, !dbg !13
  %k0.prime.kdma.source1571 = add nuw nsw i64 %k0.prime.kdma.source1569, %k0.prime.kdma.source1523, !dbg !13
  %k0.prime.kdma.source1572 = lshr i64 %k0.prime.kdma.source1567, 5, !dbg !13
  %k0.prime.kdma.source1575 = and i64 %k0.prime.kdma.source1568, %k0.prime.kdma.source1527, !dbg !13
  %k0.prime.kdma.source1576 = add nuw nsw i64 %k0.prime.kdma.source1572, %k0.prime.kdma.source1575, !dbg !13
  %k0.prime.kdma.source1577 = shl nuw nsw i64 %k0.prime.kdma.source1576, 10, !dbg !13
  %k0.prime.kdma.source1578 = sub nsw i64 %k0.prime.kdma.source1571, %k0.prime.kdma.source1577, !dbg !13
  %k0.prime.kdma.source1579 = and i64 %k0.prime.kdma.source1578, 16256, !dbg !13
  %k0.prime.kdma.source1584 = or disjoint i64 %k0.prime.kdma.source1579, %k0.prime.kdma.source1559, !dbg !13
  %k0.prime.kdma.source1585 = or disjoint i64 %k0.prime.2064, %k0.prime.kdma.source1584, !dbg !13
  %k0.prime.kdma.source1586 = trunc nuw i64 %k0.prime.kdma.source1585 to i32, !dbg !13
  %k0.prime.kdma.source1587 = shl nuw nsw i32 %k0.prime.kdma.source1586, 1, !dbg !13
  %k0.prime.kdma.offset1588 = add nuw i32 %k0.prime.kdma.source1587, %kdma.intra.hi, !dbg !13
  %k0.prime.2071 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1589, i32 %k0.prime.kdma.offset1588, <4 x i32> %k0.prime.dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %100, !dbg !6

100:                                              ; preds = %k0.join, %4
  %101 = phi i64 [ %875, %k0.join ], [ 0, %4 ], !dbg !6
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
  %address.thread517 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !32, !frisk.register_lifetimes !1
  %address.local518 = zext nneg i32 %address.thread517 to i64, !dbg !9
  %address.local524 = lshr i64 %address.local518, 2, !dbg !13
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0, !dbg !13
  %116 = bitcast <2 x i64> %115 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1378 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1501 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1502 = shl nuw nsw i64 %kdma.source1501, 9, !dbg !13
  %kdma.source1503 = shl nuw nsw i64 %kdma.source, 7, !dbg !13
  %kdma.source1504 = add nuw nsw i64 %kdma.source1502, %kdma.source1503, !dbg !13
  %kdma.source1505 = lshr i64 %kdma.source, 5, !dbg !13
  %kdma.source1506 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1507 = and i64 %kdma.source1506, 1, !dbg !13
  %kdma.source1508 = and i64 %kdma.source1501, %kdma.source1507, !dbg !13
  %kdma.source1509 = or disjoint i64 %kdma.source1505, %kdma.source1508, !dbg !13
  %kdma.source1510 = shl nuw nsw i64 %kdma.source1509, 10, !dbg !13
  %kdma.source1511 = sub nsw i64 %kdma.source1504, %kdma.source1510, !dbg !13
  %kdma.source1512 = and i64 %kdma.source1511, 16256, !dbg !13
  %kdma.source1513 = lshr i32 %kdma.tid, 4, !dbg !13
  %kdma.source1514 = and i32 %kdma.source1513, 8, !dbg !13
  %kdma.source1515 = zext nneg i32 %kdma.source1514 to i64, !dbg !13
  %kdma.source1516 = or disjoint i64 %kdma.source1512, %kdma.source1515, !dbg !13
  %kdma.source1517 = or disjoint i64 %113, %kdma.source1516, !dbg !13
  %kdma.source1518 = trunc nuw i64 %kdma.source1517 to i32, !dbg !13
  %kdma.source1519 = shl nuw nsw i32 %kdma.source1518, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1519, %kdma.intra, !dbg !13
  %kdma.source1520 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1521 = lshr i64 %kdma.source1520, 4, !dbg !13
  %kdma.source1522 = shl nuw nsw i64 %kdma.source1521, 9, !dbg !13
  %kdma.source1523 = shl nuw nsw i64 %kdma.source1520, 7, !dbg !13
  %kdma.source1524 = add nuw nsw i64 %kdma.source1522, %kdma.source1523, !dbg !13
  %kdma.source1525 = lshr i64 %kdma.source1520, 5, !dbg !13
  %kdma.source1526 = lshr i64 %kdma.source1520, 2, !dbg !13
  %kdma.source1527 = and i64 %kdma.source1526, 1, !dbg !13
  %kdma.source1528 = and i64 %kdma.source1521, %kdma.source1527, !dbg !13
  %kdma.source1529 = add nuw nsw i64 %kdma.source1525, %kdma.source1528, !dbg !13
  %kdma.source1530 = shl nuw nsw i64 %kdma.source1529, 10, !dbg !13
  %kdma.source1531 = sub nsw i64 %kdma.source1524, %kdma.source1530, !dbg !13
  %kdma.source1532 = and i64 %kdma.source1531, 16256, !dbg !13
  %kdma.source1536 = or disjoint i64 %kdma.source1532, %kdma.source1515, !dbg !13
  %kdma.source1537 = or disjoint i64 %113, %kdma.source1536, !dbg !13
  %kdma.source1538 = trunc nuw i64 %kdma.source1537 to i32, !dbg !13
  %kdma.source1539 = shl nuw nsw i32 %kdma.source1538, 1, !dbg !13
  %kdma.offset1540 = or disjoint i32 %kdma.source1539, %kdma.intra.hi, !dbg !13
  %kdma.lds1541 = add i32 %38, 512, !dbg !13
  %kdma.source1543 = or disjoint i64 %kdma.source, 256, !dbg !13
  %kdma.source1544 = lshr i64 %kdma.source1543, 4, !dbg !13
  %kdma.source1545 = shl nuw nsw i64 %kdma.source1544, 9, !dbg !13
  %kdma.source1547 = add nuw nsw i64 %kdma.source1545, %kdma.source1503, !dbg !13
  %kdma.source1548 = lshr i64 %kdma.source1543, 5, !dbg !13
  %kdma.source1551 = and i64 %kdma.source1544, %kdma.source1507, !dbg !13
  %kdma.source1552 = or disjoint i64 %kdma.source1548, %kdma.source1551, !dbg !13
  %kdma.source1553 = shl nuw nsw i64 %kdma.source1552, 10, !dbg !13
  %kdma.source1554 = sub nsw i64 %kdma.source1547, %kdma.source1553, !dbg !13
  %kdma.source1555 = and i64 %kdma.source1554, 16256, !dbg !13
  %kdma.source1558 = or disjoint i32 %kdma.source1514, 16, !dbg !13
  %kdma.source1559 = zext nneg i32 %kdma.source1558 to i64, !dbg !13
  %kdma.source1560 = or disjoint i64 %kdma.source1555, %kdma.source1559, !dbg !13
  %kdma.source1561 = or disjoint i64 %113, %kdma.source1560, !dbg !13
  %kdma.source1562 = trunc nuw i64 %kdma.source1561 to i32, !dbg !13
  %kdma.source1563 = shl nuw nsw i32 %kdma.source1562, 1, !dbg !13
  %kdma.offset1564 = add nuw i32 %kdma.source1563, %kdma.intra, !dbg !13
  %kdma.source1567 = or disjoint i64 %kdma.source1520, 256, !dbg !13
  %kdma.source1568 = lshr i64 %kdma.source1567, 4, !dbg !13
  %kdma.source1569 = shl nuw nsw i64 %kdma.source1568, 9, !dbg !13
  %kdma.source1571 = add nuw nsw i64 %kdma.source1569, %kdma.source1523, !dbg !13
  %kdma.source1572 = lshr i64 %kdma.source1567, 5, !dbg !13
  %kdma.source1575 = and i64 %kdma.source1568, %kdma.source1527, !dbg !13
  %kdma.source1576 = add nuw nsw i64 %kdma.source1572, %kdma.source1575, !dbg !13
  %kdma.source1577 = shl nuw nsw i64 %kdma.source1576, 10, !dbg !13
  %kdma.source1578 = sub nsw i64 %kdma.source1571, %kdma.source1577, !dbg !13
  %kdma.source1579 = and i64 %kdma.source1578, 16256, !dbg !13
  %kdma.source1584 = or disjoint i64 %kdma.source1579, %kdma.source1559, !dbg !13
  %kdma.source1585 = or disjoint i64 %113, %kdma.source1584, !dbg !13
  %kdma.source1586 = trunc nuw i64 %kdma.source1585 to i32, !dbg !13
  %kdma.source1587 = shl nuw nsw i32 %kdma.source1586, 1, !dbg !13
  %kdma.offset1588 = add nuw i32 %kdma.source1587, %kdma.intra.hi, !dbg !13
  %kdma.lds1589 = add i32 %46, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %117 = or disjoint i64 %113, 32, !dbg !13
  %kdma.source1607 = or disjoint i64 %117, %kdma.source1516, !dbg !13
  %kdma.source1608 = trunc nuw i64 %kdma.source1607 to i32, !dbg !13
  %kdma.source1609 = shl nuw nsw i32 %kdma.source1608, 1, !dbg !13
  %kdma.offset1610 = or disjoint i32 %kdma.source1609, %kdma.intra, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1610, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1629 = or disjoint i64 %117, %kdma.source1536, !dbg !13
  %kdma.source1630 = trunc nuw i64 %kdma.source1629 to i32, !dbg !13
  %kdma.source1631 = shl nuw nsw i32 %kdma.source1630, 1, !dbg !13
  %kdma.offset1632 = or disjoint i32 %kdma.source1631, %kdma.intra.hi, !dbg !13
  %kdma.lds1633 = add i32 %51, 512, !dbg !13
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1633, i32 %kdma.offset1632, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1653 = or disjoint i64 %117, %kdma.source1560, !dbg !13
  %kdma.source1654 = trunc nuw i64 %kdma.source1653 to i32, !dbg !13
  %kdma.source1655 = shl nuw nsw i32 %kdma.source1654, 1, !dbg !13
  %kdma.offset1656 = add nuw i32 %kdma.source1655, %kdma.intra, !dbg !13
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1656, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1677 = or disjoint i64 %117, %kdma.source1584, !dbg !13
  %kdma.source1678 = trunc nuw i64 %kdma.source1677 to i32, !dbg !13
  %kdma.source1679 = shl nuw nsw i32 %kdma.source1678, 1, !dbg !13
  %kdma.offset1680 = add nuw i32 %kdma.source1679, %kdma.intra.hi, !dbg !13
  %kdma.lds1681 = add i32 %55, 512, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1681, i32 %kdma.offset1680, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %address.local561 = shl nuw nsw i64 %address.local518, 1, !dbg !15
  %address.local562 = and i64 %address.local561, 24, !dbg !15
  %address.local564 = shl i64 %address.local518, 62, !dbg !15
  %address.local565 = ashr exact i64 %address.local564, 57, !dbg !15
  %address.local566 = and i64 %address.local565, -64, !dbg !15
  %address.local569 = and i64 %address.local524, 12, !dbg !12
  %address.local571 = and i64 %address.local518, 63, !dbg !12
  %address.local572 = lshr i64 %address.local571, 5, !dbg !12
  %address.local573 = mul nuw nsw i64 %address.local572, 1016, !dbg !12
  %.idx1416 = mul nuw nsw i64 %address.local518, 320, !dbg !15
  %122 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1416, !dbg !15
  %.idx1417 = mul nsw i64 %address.local524, -1280, !dbg !15
  %123 = getelementptr i8, ptr addrspace(3) %122, i64 %.idx1417, !dbg !15
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local562, !dbg !15
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local566, !dbg !15
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local569, !dbg !15
  %address.local575 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local573, !dbg !15
  %127 = add nsw i64 %.idx1416, %.idx1417, !dbg !15
  %.idx1992 = shl nuw nsw i64 %address.local562, 1, !dbg !15
  %128 = or disjoint i64 %127, %.idx1992, !dbg !15
  %129 = or disjoint i64 %address.local566, %address.local569, !dbg !15
  %130 = shl nsw i64 %129, 1, !dbg !15
  %131 = add nsw i64 %130, %128, !dbg !15
  %address.local575.idx = mul nuw nsw i64 %address.local572, 2032, !dbg !15
  %132 = add nsw i64 %131, %address.local575.idx, !dbg !15
  %133 = lshr i64 %132, 6, !dbg !15
  %134 = and i64 %133, 8, !dbg !15
  %135 = xor i64 %132, %134, !dbg !15
  %136 = sub i64 %135, %132, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local575, i64 %136, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %lazy.base = extractvalue [2 x <1 x float>] %104, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base2077 = extractvalue [2 x <1 x float>] %104, 1, !dbg !15
  %lazy.seed2078 = shufflevector <1 x float> %lazy.base2077, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> %lazy.seed) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local586 = shl nuw nsw i32 %address.thread517, 1, !dbg !15
  %address.local587 = and i32 %address.local586, 24, !dbg !15
  %address.local588 = shl nuw nsw i32 %address.thread517, 8, !dbg !15
  %address.local589 = and i32 %address.local588, 768, !dbg !15
  %address.local590 = or disjoint i32 %address.local587, %address.local589, !dbg !15
  %address.local591 = shl nuw nsw i32 %address.thread517, 5, !dbg !15
  %address.local592 = and i32 %address.local591, 96, !dbg !15
  %address.local593 = or disjoint i32 %address.local590, %address.local592, !dbg !15
  %address.local594 = zext nneg i32 %address.local593 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %99) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local595 = lshr i32 %address.thread517, 1, !dbg !15
  %address.local596 = and i32 %address.local595, 1, !dbg !15
  %address.local597 = zext nneg i32 %address.local596 to i64, !dbg !15
  %address.local598 = add nsw i64 %address.base, %address.local597, !dbg !15
  %address.local599 = shl i64 %address.local598, 6, !dbg !15
  %address.local600 = sub i64 %address.local594, %address.local599, !dbg !15
  %140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local600, !dbg !15
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local569, !dbg !15
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local573, !dbg !15
  %address.local604 = getelementptr i8, ptr addrspace(3) %142, i64 64, !dbg !15
  %.idx19951998 = add i64 %address.local600, %address.local569, !dbg !15
  %143 = shl i64 %.idx19951998, 1, !dbg !15
  %144 = add i64 %143, %address.local575.idx, !dbg !15
  %145 = add i64 %144, 64, !dbg !15
  %146 = lshr i64 %145, 6, !dbg !15
  %147 = and i64 %146, 8, !dbg !15
  %148 = xor i64 %145, %147, !dbg !15
  %149 = sub i64 %148, %145, !dbg !15
  %kdma.swizzled1866 = getelementptr i8, ptr addrspace(3) %address.local604, i64 %149, !dbg !15
  %150 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1866, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 128, !dbg !15
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 4096, !dbg !15
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 4224, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !15, !frisk.k_lds_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %139, <4 x float> %138) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 1024, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 1152, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_lds_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 5120, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1879 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 5248, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !15, !frisk.k_lds_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_lds_pair !40
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
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> %lazy.seed2078) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> %lazy.seed2078) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> %lazy.seed2078) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> %lazy.seed2078) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> %lazy.seed2078) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> %lazy.seed2078) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> %lazy.seed2078) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> %lazy.seed2078) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %163, <4 x float> %193) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %195 = or disjoint i64 %113, 64, !dbg !13
  %kdma.source1699 = or disjoint i64 %195, %kdma.source1516, !dbg !13
  %kdma.source1700 = trunc nuw i64 %kdma.source1699 to i32, !dbg !13
  %kdma.source1701 = shl nuw nsw i32 %kdma.source1700, 1, !dbg !13
  %kdma.offset1702 = or disjoint i32 %kdma.source1701, %kdma.intra, !dbg !13
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %kdma.offset1702, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1721 = or disjoint i64 %195, %kdma.source1536, !dbg !13
  %kdma.source1722 = trunc nuw i64 %kdma.source1721 to i32, !dbg !13
  %kdma.source1723 = shl nuw nsw i32 %kdma.source1722, 1, !dbg !13
  %kdma.offset1724 = or disjoint i32 %kdma.source1723, %kdma.intra.hi, !dbg !13
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1541, i32 %kdma.offset1724, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1745 = or disjoint i64 %195, %kdma.source1560, !dbg !13
  %kdma.source1746 = trunc nuw i64 %kdma.source1745 to i32, !dbg !13
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !13
  %kdma.offset1748 = add nuw i32 %kdma.source1747, %kdma.intra, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %kdma.offset1748, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1769 = or disjoint i64 %195, %kdma.source1584, !dbg !13
  %kdma.source1770 = trunc nuw i64 %kdma.source1769 to i32, !dbg !13
  %kdma.source1771 = shl nuw nsw i32 %kdma.source1770, 1, !dbg !13
  %kdma.offset1772 = add nuw i32 %kdma.source1771, %kdma.intra.hi, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1589, i32 %kdma.offset1772, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.gep1896 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1900 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1897 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1901 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1898 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 8192, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1902 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 8320, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1899 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 12288, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1903 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 12416, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1904 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %200, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1905 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !15, !frisk.k_lds_pair !46
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %210) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1906 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 9216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !15, !frisk.k_lds_pair !47
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 9344, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !15, !frisk.k_lds_pair !47
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 13312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !15, !frisk.k_lds_pair !48
  %kdma.gep1911 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1866, i64 13440, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !15, !frisk.k_lds_pair !48
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
  %kdma.source1791 = or disjoint i64 %248, %kdma.source1516, !dbg !13
  %kdma.source1792 = trunc nuw i64 %kdma.source1791 to i32, !dbg !13
  %kdma.source1793 = shl nuw nsw i32 %kdma.source1792, 1, !dbg !13
  %kdma.offset1794 = or disjoint i32 %kdma.source1793, %kdma.intra, !dbg !13
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1794, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1813 = or disjoint i64 %248, %kdma.source1536, !dbg !13
  %kdma.source1814 = trunc nuw i64 %kdma.source1813 to i32, !dbg !13
  %kdma.source1815 = shl nuw nsw i32 %kdma.source1814, 1, !dbg !13
  %kdma.offset1816 = or disjoint i32 %kdma.source1815, %kdma.intra.hi, !dbg !13
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1633, i32 %kdma.offset1816, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1837 = or disjoint i64 %248, %kdma.source1560, !dbg !13
  %kdma.source1838 = trunc nuw i64 %kdma.source1837 to i32, !dbg !13
  %kdma.source1839 = shl nuw nsw i32 %kdma.source1838, 1, !dbg !13
  %kdma.offset1840 = add nuw i32 %kdma.source1839, %kdma.intra, !dbg !13
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1840, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %kdma.source1861 = or disjoint i64 %248, %kdma.source1584, !dbg !13
  %kdma.source1862 = trunc nuw i64 %kdma.source1861 to i32, !dbg !13
  %kdma.source1863 = shl nuw nsw i32 %kdma.source1862, 1, !dbg !13
  %kdma.offset1864 = add nuw i32 %kdma.source1863, %kdma.intra.hi, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1681, i32 %kdma.offset1864, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !49
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !15, !frisk.k_lds_pair !49
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !50
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !15, !frisk.k_lds_pair !50
  %257 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1866, align 8, !dbg !15, !frisk.k_lds_pair !51
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !15, !frisk.k_lds_pair !51
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !15, !frisk.k_lds_pair !52
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !15, !frisk.k_lds_pair !52
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !15, !frisk.k_lds_pair !53
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %253, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !15, !frisk.k_lds_pair !54
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !15, !frisk.k_lds_pair !54
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %255, <4 x float> %263) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !15, !frisk.k_lds_pair !55
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !15, !frisk.k_lds_pair !55
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %257, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !15, !frisk.k_lds_pair !56
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1879, align 8, !dbg !15, !frisk.k_lds_pair !56
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
  %address.local933 = shl nuw nsw i64 %address.local518, 5, !dbg !16
  %address.local934 = and i64 %address.local933, 8064, !dbg !16
  %address.local935 = shl nuw nsw i64 %address.local518, 3, !dbg !9
  %address.local936 = shl nuw nsw i64 %address.local524, 5, !dbg !16
  %address.local937 = sub nsw i64 %address.local935, %address.local936, !dbg !16
  %address.local938 = add nsw i64 %address.local934, %address.local937, !dbg !16
  %301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx14261427 = add nsw i64 %113, %address.local938, !dbg !16
  %.idx14261427.tr = trunc i64 %.idx14261427 to i32, !dbg !16
  %dma.buffer.offset1361 = shl i32 %.idx14261427.tr, 1, !dbg !16
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !16
  %303 = bitcast <2 x i64> %302 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1362 = shufflevector <4 x i32> %303, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1361, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %address.local940 = shl nuw nsw i32 %address.thread517, 3, !dbg !16
  %address.local941 = and i32 %address.local940, 24, !dbg !16
  %address.local943 = and i32 %address.local591, 8064, !dbg !16
  %address.local944 = or disjoint i32 %address.local941, %address.local943, !dbg !16
  %address.local945 = or disjoint i32 %address.local944, 8192, !dbg !16
  %305 = trunc nuw i64 %113 to i32, !dbg !16
  %.idx14281429.tr = or disjoint i32 %address.local945, %305, !dbg !16
  %dma.buffer.offset1359 = shl nuw nsw i32 %.idx14281429.tr, 1, !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1359, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !15, !frisk.k_tail_pair !35
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !15, !frisk.k_tail_pair !35
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !15, !frisk.k_tail_pair !36
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !15, !frisk.k_tail_pair !36
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !15, !frisk.k_tail_pair !37
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !15, !frisk.k_tail_pair !37
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !15, !frisk.k_tail_pair !38
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %307, <4 x float> %266) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %309, <4 x float> %315) #9, !dbg !15
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %311, <4 x float> %272) #9, !dbg !15
  %318 = extractelement <4 x float> %316, i64 0, !dbg !20
  %319 = call float @llvm.maxnum.f32(float %318, float 0xFFF0000000000000), !dbg !8, !frisk.local_reduction !1
  %320 = extractelement <4 x float> %316, i64 1, !dbg !20
  %321 = extractelement <4 x float> %316, i64 2, !dbg !20
  %322 = extractelement <4 x float> %316, i64 3, !dbg !20
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %313, <4 x float> %317) #9, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !15, !frisk.k_tail_pair !33
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !15, !frisk.k_tail_pair !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %308, <4 x float> %274) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %327 = extractelement <4 x float> %323, i64 0, !dbg !20
  %local.reduce155 = call float @llvm.maxnum.f32(float %319, float %327), !dbg !8, !frisk.local_reduction !1
  %328 = extractelement <4 x float> %323, i64 1, !dbg !20
  %local.reduce156 = call float @llvm.maxnum.f32(float %320, float %328), !dbg !8, !frisk.local_reduction !1
  %329 = extractelement <4 x float> %323, i64 2, !dbg !20
  %local.reduce157 = call float @llvm.maxnum.f32(float %321, float %329), !dbg !8, !frisk.local_reduction !1
  %330 = extractelement <4 x float> %323, i64 3, !dbg !20
  %local.reduce158 = call float @llvm.maxnum.f32(float %322, float %330), !dbg !8, !frisk.local_reduction !1
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !15, !frisk.k_tail_pair !39
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %326) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %334 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !15, !frisk.k_tail_pair !40
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !15, !frisk.k_tail_pair !40
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %312, <4 x float> %276) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !20, !frisk.k_tail_prefetch !1
  %337 = extractelement <4 x float> %333, i64 0, !dbg !20
  %local.reduce159 = call float @llvm.maxnum.f32(float %local.reduce155, float %337), !dbg !8, !frisk.local_reduction !1
  %338 = extractelement <4 x float> %333, i64 1, !dbg !20
  %local.reduce160 = call float @llvm.maxnum.f32(float %local.reduce156, float %338), !dbg !8, !frisk.local_reduction !1
  %339 = extractelement <4 x float> %333, i64 2, !dbg !20
  %local.reduce161 = call float @llvm.maxnum.f32(float %local.reduce157, float %339), !dbg !8, !frisk.local_reduction !1
  %340 = extractelement <4 x float> %333, i64 3, !dbg !20
  %local.reduce162 = call float @llvm.maxnum.f32(float %local.reduce158, float %340), !dbg !8, !frisk.local_reduction !1
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !15, !frisk.k_tail_pair !41
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1911, align 8, !dbg !15, !frisk.k_tail_pair !41
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %314, <4 x float> %336) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %324, <4 x float> %278) #9, !dbg !15
  %345 = extractelement <4 x float> %343, i64 0, !dbg !20
  %local.reduce163 = call float @llvm.maxnum.f32(float %local.reduce159, float %345), !dbg !8, !frisk.local_reduction !1
  %346 = extractelement <4 x float> %343, i64 1, !dbg !20
  %local.reduce164 = call float @llvm.maxnum.f32(float %local.reduce160, float %346), !dbg !8, !frisk.local_reduction !1
  %347 = extractelement <4 x float> %343, i64 2, !dbg !20
  %local.reduce165 = call float @llvm.maxnum.f32(float %local.reduce161, float %347), !dbg !8, !frisk.local_reduction !1
  %348 = extractelement <4 x float> %343, i64 3, !dbg !20
  %local.reduce166 = call float @llvm.maxnum.f32(float %local.reduce162, float %348), !dbg !8, !frisk.local_reduction !1
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %331, <4 x float> %344) #9, !dbg !15
  %350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %334, <4 x float> %280) #9, !dbg !15
  %351 = extractelement <4 x float> %349, i64 0, !dbg !20
  %local.reduce167 = call float @llvm.maxnum.f32(float %local.reduce163, float %351), !dbg !8, !frisk.local_reduction !1
  %352 = extractelement <4 x float> %349, i64 1, !dbg !20
  %local.reduce168 = call float @llvm.maxnum.f32(float %local.reduce164, float %352), !dbg !8, !frisk.local_reduction !1
  %353 = extractelement <4 x float> %349, i64 2, !dbg !20
  %local.reduce169 = call float @llvm.maxnum.f32(float %local.reduce165, float %353), !dbg !8, !frisk.local_reduction !1
  %354 = extractelement <4 x float> %349, i64 3, !dbg !20
  %local.reduce170 = call float @llvm.maxnum.f32(float %local.reduce166, float %354), !dbg !8, !frisk.local_reduction !1
  %355 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %341, <4 x float> %350) #9, !dbg !15
  %356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %325, <4 x float> %282) #9, !dbg !15
  %357 = extractelement <4 x float> %355, i64 0, !dbg !20
  %local.reduce171 = call float @llvm.maxnum.f32(float %local.reduce167, float %357), !dbg !8, !frisk.local_reduction !1
  %358 = extractelement <4 x float> %355, i64 1, !dbg !20
  %local.reduce172 = call float @llvm.maxnum.f32(float %local.reduce168, float %358), !dbg !8, !frisk.local_reduction !1
  %359 = extractelement <4 x float> %355, i64 2, !dbg !20
  %local.reduce173 = call float @llvm.maxnum.f32(float %local.reduce169, float %359), !dbg !8, !frisk.local_reduction !1
  %360 = extractelement <4 x float> %355, i64 3, !dbg !20
  %local.reduce174 = call float @llvm.maxnum.f32(float %local.reduce170, float %360), !dbg !8, !frisk.local_reduction !1
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %332, <4 x float> %356) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %335, <4 x float> %284) #9, !dbg !15
  %363 = extractelement <4 x float> %361, i64 0, !dbg !20
  %local.reduce175 = call float @llvm.maxnum.f32(float %local.reduce171, float %363), !dbg !8, !frisk.local_reduction !1
  %364 = extractelement <4 x float> %361, i64 1, !dbg !20
  %local.reduce176 = call float @llvm.maxnum.f32(float %local.reduce172, float %364), !dbg !8, !frisk.local_reduction !1
  %365 = extractelement <4 x float> %361, i64 2, !dbg !20
  %local.reduce177 = call float @llvm.maxnum.f32(float %local.reduce173, float %365), !dbg !8, !frisk.local_reduction !1
  %366 = extractelement <4 x float> %361, i64 3, !dbg !20
  %local.reduce178 = call float @llvm.maxnum.f32(float %local.reduce174, float %366), !dbg !8, !frisk.local_reduction !1
  %367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %342, <4 x float> %362) #9, !dbg !15
  %368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %307, <4 x float> %286) #9, !dbg !15
  %369 = extractelement <4 x float> %367, i64 0, !dbg !20
  %local.reduce179 = call float @llvm.maxnum.f32(float %local.reduce175, float %369), !dbg !8, !frisk.local_reduction !1
  %370 = extractelement <4 x float> %367, i64 1, !dbg !20
  %local.reduce180 = call float @llvm.maxnum.f32(float %local.reduce176, float %370), !dbg !8, !frisk.local_reduction !1
  %371 = extractelement <4 x float> %367, i64 2, !dbg !20
  %local.reduce181 = call float @llvm.maxnum.f32(float %local.reduce177, float %371), !dbg !8, !frisk.local_reduction !1
  %372 = extractelement <4 x float> %367, i64 3, !dbg !20
  %local.reduce182 = call float @llvm.maxnum.f32(float %local.reduce178, float %372), !dbg !8, !frisk.local_reduction !1
  %local.reduce183 = call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !8, !frisk.local_reduction !1
  %local.reduce184 = call float @llvm.maxnum.f32(float %local.reduce181, float %local.reduce182), !dbg !8, !frisk.local_reduction !1
  %373 = call float @llvm.maxnum.f32(float %local.reduce183, float %local.reduce184), !dbg !8, !frisk.local_reduction !1
  %374 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %309, <4 x float> %368) #9, !dbg !15
  %375 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %311, <4 x float> %288) #9, !dbg !15
  %376 = extractelement <4 x float> %374, i64 0, !dbg !20
  %377 = extractelement <4 x float> %374, i64 1, !dbg !20
  %378 = extractelement <4 x float> %374, i64 2, !dbg !20
  %379 = extractelement <4 x float> %374, i64 3, !dbg !20
  %380 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %313, <4 x float> %375) #9, !dbg !15
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %308, <4 x float> %290) #9, !dbg !15
  %382 = extractelement <4 x float> %380, i64 0, !dbg !20
  %383 = extractelement <4 x float> %380, i64 1, !dbg !20
  %384 = extractelement <4 x float> %380, i64 2, !dbg !20
  %385 = extractelement <4 x float> %380, i64 3, !dbg !20
  %386 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %310, <4 x float> %381) #9, !dbg !15
  %387 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %312, <4 x float> %292) #9, !dbg !15
  %388 = extractelement <4 x float> %386, i64 0, !dbg !20
  %389 = extractelement <4 x float> %386, i64 1, !dbg !20
  %390 = extractelement <4 x float> %386, i64 2, !dbg !20
  %391 = extractelement <4 x float> %386, i64 3, !dbg !20
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %314, <4 x float> %387) #9, !dbg !15
  %393 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %324, <4 x float> %294) #9, !dbg !15
  %394 = extractelement <4 x float> %392, i64 0, !dbg !20
  %395 = extractelement <4 x float> %392, i64 1, !dbg !20
  %396 = extractelement <4 x float> %392, i64 2, !dbg !20
  %397 = extractelement <4 x float> %392, i64 3, !dbg !20
  %398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %331, <4 x float> %393) #9, !dbg !15
  %399 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %334, <4 x float> %296) #9, !dbg !15
  %400 = extractelement <4 x float> %398, i64 0, !dbg !20
  %401 = extractelement <4 x float> %398, i64 1, !dbg !20
  %402 = extractelement <4 x float> %398, i64 2, !dbg !20
  %403 = extractelement <4 x float> %398, i64 3, !dbg !20
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %341, <4 x float> %399) #9, !dbg !15
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %325, <4 x float> %298) #9, !dbg !15
  %406 = extractelement <4 x float> %404, i64 0, !dbg !20
  %407 = extractelement <4 x float> %404, i64 1, !dbg !20
  %408 = extractelement <4 x float> %404, i64 2, !dbg !20
  %409 = extractelement <4 x float> %404, i64 3, !dbg !20
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %332, <4 x float> %405) #9, !dbg !15
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %335, <4 x float> %300) #9, !dbg !15
  %412 = extractelement <4 x float> %410, i64 0, !dbg !20
  %413 = extractelement <4 x float> %410, i64 1, !dbg !20
  %414 = extractelement <4 x float> %410, i64 2, !dbg !20
  %415 = extractelement <4 x float> %410, i64 3, !dbg !20
  %416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %342, <4 x float> %411) #9, !dbg !15
  %417 = extractelement <4 x float> %416, i64 0, !dbg !20
  %418 = extractelement <4 x float> %416, i64 1, !dbg !20
  %419 = extractelement <4 x float> %416, i64 2, !dbg !20
  %420 = extractelement <4 x float> %416, i64 3, !dbg !20
  %421 = bitcast float %373 to i32, !dbg !8
  %422 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %421), !dbg !8
  %423 = bitcast i32 %422 to float, !dbg !8
  %424 = call float @llvm.maxnum.f32(float %373, float %423), !dbg !8
  %425 = bitcast float %424 to i32, !dbg !8
  %426 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %425), !dbg !8
  %427 = bitcast i32 %426 to float, !dbg !8
  %428 = call float @llvm.maxnum.f32(float %424, float %427), !dbg !8
  %429 = call float @llvm.maxnum.f32(float %376, float 0xFFF0000000000000), !dbg !8, !frisk.local_reduction !1
  %local.reduce124 = call float @llvm.maxnum.f32(float %429, float %382), !dbg !8, !frisk.local_reduction !1
  %local.reduce125 = call float @llvm.maxnum.f32(float %377, float %383), !dbg !8, !frisk.local_reduction !1
  %local.reduce126 = call float @llvm.maxnum.f32(float %378, float %384), !dbg !8, !frisk.local_reduction !1
  %local.reduce127 = call float @llvm.maxnum.f32(float %379, float %385), !dbg !8, !frisk.local_reduction !1
  %local.reduce128 = call float @llvm.maxnum.f32(float %local.reduce124, float %388), !dbg !8, !frisk.local_reduction !1
  %local.reduce129 = call float @llvm.maxnum.f32(float %local.reduce125, float %389), !dbg !8, !frisk.local_reduction !1
  %local.reduce130 = call float @llvm.maxnum.f32(float %local.reduce126, float %390), !dbg !8, !frisk.local_reduction !1
  %local.reduce131 = call float @llvm.maxnum.f32(float %local.reduce127, float %391), !dbg !8, !frisk.local_reduction !1
  %local.reduce132 = call float @llvm.maxnum.f32(float %local.reduce128, float %394), !dbg !8, !frisk.local_reduction !1
  %local.reduce133 = call float @llvm.maxnum.f32(float %local.reduce129, float %395), !dbg !8, !frisk.local_reduction !1
  %local.reduce134 = call float @llvm.maxnum.f32(float %local.reduce130, float %396), !dbg !8, !frisk.local_reduction !1
  %local.reduce135 = call float @llvm.maxnum.f32(float %local.reduce131, float %397), !dbg !8, !frisk.local_reduction !1
  %local.reduce136 = call float @llvm.maxnum.f32(float %local.reduce132, float %400), !dbg !8, !frisk.local_reduction !1
  %local.reduce137 = call float @llvm.maxnum.f32(float %local.reduce133, float %401), !dbg !8, !frisk.local_reduction !1
  %local.reduce138 = call float @llvm.maxnum.f32(float %local.reduce134, float %402), !dbg !8, !frisk.local_reduction !1
  %local.reduce139 = call float @llvm.maxnum.f32(float %local.reduce135, float %403), !dbg !8, !frisk.local_reduction !1
  %local.reduce140 = call float @llvm.maxnum.f32(float %local.reduce136, float %406), !dbg !8, !frisk.local_reduction !1
  %local.reduce141 = call float @llvm.maxnum.f32(float %local.reduce137, float %407), !dbg !8, !frisk.local_reduction !1
  %local.reduce142 = call float @llvm.maxnum.f32(float %local.reduce138, float %408), !dbg !8, !frisk.local_reduction !1
  %local.reduce143 = call float @llvm.maxnum.f32(float %local.reduce139, float %409), !dbg !8, !frisk.local_reduction !1
  %local.reduce144 = call float @llvm.maxnum.f32(float %local.reduce140, float %412), !dbg !8, !frisk.local_reduction !1
  %local.reduce145 = call float @llvm.maxnum.f32(float %local.reduce141, float %413), !dbg !8, !frisk.local_reduction !1
  %local.reduce146 = call float @llvm.maxnum.f32(float %local.reduce142, float %414), !dbg !8, !frisk.local_reduction !1
  %local.reduce147 = call float @llvm.maxnum.f32(float %local.reduce143, float %415), !dbg !8, !frisk.local_reduction !1
  %local.reduce148 = call float @llvm.maxnum.f32(float %local.reduce144, float %417), !dbg !8, !frisk.local_reduction !1
  %local.reduce149 = call float @llvm.maxnum.f32(float %local.reduce145, float %418), !dbg !8, !frisk.local_reduction !1
  %local.reduce150 = call float @llvm.maxnum.f32(float %local.reduce146, float %419), !dbg !8, !frisk.local_reduction !1
  %local.reduce151 = call float @llvm.maxnum.f32(float %local.reduce147, float %420), !dbg !8, !frisk.local_reduction !1
  %local.reduce152 = call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !8, !frisk.local_reduction !1
  %local.reduce153 = call float @llvm.maxnum.f32(float %local.reduce150, float %local.reduce151), !dbg !8, !frisk.local_reduction !1
  %430 = call float @llvm.maxnum.f32(float %local.reduce152, float %local.reduce153), !dbg !8, !frisk.local_reduction !1
  %431 = bitcast float %430 to i32, !dbg !8
  %432 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %431), !dbg !8
  %433 = bitcast i32 %432 to float, !dbg !8
  %434 = call float @llvm.maxnum.f32(float %430, float %433), !dbg !8
  %435 = bitcast float %434 to i32, !dbg !8
  %436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %435), !dbg !8
  %437 = bitcast i32 %436 to float, !dbg !8
  %438 = call float @llvm.maxnum.f32(float %434, float %437), !dbg !8
  %lazy.peak = call float @llvm.maxnum.f32(float %428, float %438), !dbg !24
  %lazy.over = fcmp ogt float %lazy.peak, 0x4020000000000000, !dbg !24
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !24
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !24
  %lazy.first = icmp eq i64 %101, 0, !dbg !24
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !24
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !24

lazy.slow:                                        ; preds = %109
  %lazy.positive = call float @llvm.maxnum.f32(float %428, float 0x0000000000000000), !dbg !24
  %lazy.delta = select i1 %lazy.first, float %428, float %lazy.positive, !dbg !24
  %lazy.negated = fsub float 0x0000000000000000, %lazy.delta, !dbg !24
  %lazy.exp = call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !24
  %lazy.alpha = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp, !dbg !24
  %lazy.positive2079 = call float @llvm.maxnum.f32(float %438, float 0x0000000000000000), !dbg !24
  %lazy.delta2080 = select i1 %lazy.first, float %438, float %lazy.positive2079, !dbg !24
  %lazy.negated2081 = fsub float 0x0000000000000000, %lazy.delta2080, !dbg !24
  %lazy.exp2082 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated2081), !dbg !24
  %lazy.alpha2083 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp2082, !dbg !24
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted = fsub <4 x float> %316, %lazy.delta.v, !dbg !24
  %lazy.delta.v.lane2084 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2085 = shufflevector <1 x float> %lazy.delta.v.lane2084, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2086 = fsub <4 x float> %323, %lazy.delta.v2085, !dbg !24
  %lazy.delta.v.lane2087 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2088 = shufflevector <1 x float> %lazy.delta.v.lane2087, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2089 = fsub <4 x float> %333, %lazy.delta.v2088, !dbg !24
  %lazy.delta.v.lane2090 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2091 = shufflevector <1 x float> %lazy.delta.v.lane2090, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2092 = fsub <4 x float> %343, %lazy.delta.v2091, !dbg !24
  %lazy.delta.v.lane2093 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2094 = shufflevector <1 x float> %lazy.delta.v.lane2093, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2095 = fsub <4 x float> %349, %lazy.delta.v2094, !dbg !24
  %lazy.delta.v.lane2096 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2097 = shufflevector <1 x float> %lazy.delta.v.lane2096, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2098 = fsub <4 x float> %355, %lazy.delta.v2097, !dbg !24
  %lazy.delta.v.lane2099 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2100 = shufflevector <1 x float> %lazy.delta.v.lane2099, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2101 = fsub <4 x float> %361, %lazy.delta.v2100, !dbg !24
  %lazy.delta.v.lane2102 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.delta.v2103 = shufflevector <1 x float> %lazy.delta.v.lane2102, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2104 = fsub <4 x float> %367, %lazy.delta.v2103, !dbg !24
  %lazy.delta.v.lane2105 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2106 = shufflevector <1 x float> %lazy.delta.v.lane2105, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2107 = fsub <4 x float> %374, %lazy.delta.v2106, !dbg !24
  %lazy.delta.v.lane2108 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2109 = shufflevector <1 x float> %lazy.delta.v.lane2108, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2110 = fsub <4 x float> %380, %lazy.delta.v2109, !dbg !24
  %lazy.delta.v.lane2111 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2112 = shufflevector <1 x float> %lazy.delta.v.lane2111, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2113 = fsub <4 x float> %386, %lazy.delta.v2112, !dbg !24
  %lazy.delta.v.lane2114 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2115 = shufflevector <1 x float> %lazy.delta.v.lane2114, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2116 = fsub <4 x float> %392, %lazy.delta.v2115, !dbg !24
  %lazy.delta.v.lane2117 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2118 = shufflevector <1 x float> %lazy.delta.v.lane2117, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2119 = fsub <4 x float> %398, %lazy.delta.v2118, !dbg !24
  %lazy.delta.v.lane2120 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2121 = shufflevector <1 x float> %lazy.delta.v.lane2120, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2122 = fsub <4 x float> %404, %lazy.delta.v2121, !dbg !24
  %lazy.delta.v.lane2123 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2124 = shufflevector <1 x float> %lazy.delta.v.lane2123, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2125 = fsub <4 x float> %410, %lazy.delta.v2124, !dbg !24
  %lazy.delta.v.lane2126 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.delta.v2127 = shufflevector <1 x float> %lazy.delta.v.lane2126, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !24
  %lazy.shifted2128 = fsub <4 x float> %416, %lazy.delta.v2127, !dbg !24
  %lazy.state = extractvalue [2 x <1 x float>] %103, 0, !dbg !24
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !24
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !24
  %lazy.state2129 = extractvalue [2 x <1 x float>] %103, 1, !dbg !24
  %lazy.alpha.v.lane2130 = insertelement <1 x float> poison, float %lazy.alpha2083, i64 0, !dbg !24
  %lazy.scaled2131 = fmul <1 x float> %lazy.state2129, %lazy.alpha.v.lane2130, !dbg !24
  %lazy.state.next2132 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2131, 1, !dbg !24
  %lazy.state2133 = extractvalue [2 x <8 x float>] %107, 0, !dbg !24
  %lazy.alpha.v.lane2134 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane2134, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2135 = fmul <8 x float> %lazy.state2133, %lazy.alpha.v, !dbg !24
  %lazy.state.next2136 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2135, 0, !dbg !24
  %lazy.state2137 = extractvalue [2 x <8 x float>] %107, 1, !dbg !24
  %lazy.alpha.v.lane2138 = insertelement <1 x float> poison, float %lazy.alpha2083, i64 0, !dbg !24
  %lazy.alpha.v2139 = shufflevector <1 x float> %lazy.alpha.v.lane2138, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2140 = fmul <8 x float> %lazy.state2137, %lazy.alpha.v2139, !dbg !24
  %lazy.state.next2141 = insertvalue [2 x <8 x float>] %lazy.state.next2136, <8 x float> %lazy.scaled2140, 1, !dbg !24
  %lazy.state2142 = extractvalue [2 x <8 x float>] %106, 0, !dbg !24
  %lazy.alpha.v.lane2143 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v2144 = shufflevector <1 x float> %lazy.alpha.v.lane2143, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2145 = fmul <8 x float> %lazy.state2142, %lazy.alpha.v2144, !dbg !24
  %lazy.state.next2146 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2145, 0, !dbg !24
  %lazy.state2147 = extractvalue [2 x <8 x float>] %106, 1, !dbg !24
  %lazy.alpha.v.lane2148 = insertelement <1 x float> poison, float %lazy.alpha2083, i64 0, !dbg !24
  %lazy.alpha.v2149 = shufflevector <1 x float> %lazy.alpha.v.lane2148, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2150 = fmul <8 x float> %lazy.state2147, %lazy.alpha.v2149, !dbg !24
  %lazy.state.next2151 = insertvalue [2 x <8 x float>] %lazy.state.next2146, <8 x float> %lazy.scaled2150, 1, !dbg !24
  %lazy.state2152 = extractvalue [2 x <8 x float>] %105, 0, !dbg !24
  %lazy.alpha.v.lane2153 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v2154 = shufflevector <1 x float> %lazy.alpha.v.lane2153, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2155 = fmul <8 x float> %lazy.state2152, %lazy.alpha.v2154, !dbg !24
  %lazy.state.next2156 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2155, 0, !dbg !24
  %lazy.state2157 = extractvalue [2 x <8 x float>] %105, 1, !dbg !24
  %lazy.alpha.v.lane2158 = insertelement <1 x float> poison, float %lazy.alpha2083, i64 0, !dbg !24
  %lazy.alpha.v2159 = shufflevector <1 x float> %lazy.alpha.v.lane2158, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2160 = fmul <8 x float> %lazy.state2157, %lazy.alpha.v2159, !dbg !24
  %lazy.state.next2161 = insertvalue [2 x <8 x float>] %lazy.state.next2156, <8 x float> %lazy.scaled2160, 1, !dbg !24
  %lazy.state2162 = extractvalue [2 x <8 x float>] %102, 0, !dbg !24
  %lazy.alpha.v.lane2163 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !24
  %lazy.alpha.v2164 = shufflevector <1 x float> %lazy.alpha.v.lane2163, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2165 = fmul <8 x float> %lazy.state2162, %lazy.alpha.v2164, !dbg !24
  %lazy.state.next2166 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2165, 0, !dbg !24
  %lazy.state2167 = extractvalue [2 x <8 x float>] %102, 1, !dbg !24
  %lazy.alpha.v.lane2168 = insertelement <1 x float> poison, float %lazy.alpha2083, i64 0, !dbg !24
  %lazy.alpha.v2169 = shufflevector <1 x float> %lazy.alpha.v.lane2168, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !24
  %lazy.scaled2170 = fmul <8 x float> %lazy.state2167, %lazy.alpha.v2169, !dbg !24
  %lazy.state.next2171 = insertvalue [2 x <8 x float>] %lazy.state.next2166, <8 x float> %lazy.scaled2170, 1, !dbg !24
  %lazy.base.old = extractvalue [2 x <1 x float>] %104, 0, !dbg !24
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !24
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !24
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !24
  %lazy.base.old2172 = extractvalue [2 x <1 x float>] %104, 1, !dbg !24
  %lazy.delta.b.lane2173 = insertelement <1 x float> poison, float %lazy.delta2080, i64 0, !dbg !24
  %lazy.base.next2174 = fsub <1 x float> %lazy.base.old2172, %lazy.delta.b.lane2173, !dbg !24
  %lazy.base.state2175 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2174, 1, !dbg !24
  br label %lazy.join, !dbg !24

lazy.join:                                        ; preds = %lazy.slow, %109
  %lazy.score = phi <4 x float> [ %316, %109 ], [ %lazy.shifted, %lazy.slow ], !dbg !24
  %lazy.score2176 = phi <4 x float> [ %323, %109 ], [ %lazy.shifted2086, %lazy.slow ], !dbg !24
  %lazy.score2177 = phi <4 x float> [ %333, %109 ], [ %lazy.shifted2089, %lazy.slow ], !dbg !24
  %lazy.score2178 = phi <4 x float> [ %343, %109 ], [ %lazy.shifted2092, %lazy.slow ], !dbg !24
  %lazy.score2179 = phi <4 x float> [ %349, %109 ], [ %lazy.shifted2095, %lazy.slow ], !dbg !24
  %lazy.score2180 = phi <4 x float> [ %355, %109 ], [ %lazy.shifted2098, %lazy.slow ], !dbg !24
  %lazy.score2181 = phi <4 x float> [ %361, %109 ], [ %lazy.shifted2101, %lazy.slow ], !dbg !24
  %lazy.score2182 = phi <4 x float> [ %367, %109 ], [ %lazy.shifted2104, %lazy.slow ], !dbg !24
  %lazy.score2183 = phi <4 x float> [ %374, %109 ], [ %lazy.shifted2107, %lazy.slow ], !dbg !24
  %lazy.score2184 = phi <4 x float> [ %380, %109 ], [ %lazy.shifted2110, %lazy.slow ], !dbg !24
  %lazy.score2185 = phi <4 x float> [ %386, %109 ], [ %lazy.shifted2113, %lazy.slow ], !dbg !24
  %lazy.score2186 = phi <4 x float> [ %392, %109 ], [ %lazy.shifted2116, %lazy.slow ], !dbg !24
  %lazy.score2187 = phi <4 x float> [ %398, %109 ], [ %lazy.shifted2119, %lazy.slow ], !dbg !24
  %lazy.score2188 = phi <4 x float> [ %404, %109 ], [ %lazy.shifted2122, %lazy.slow ], !dbg !24
  %lazy.score2189 = phi <4 x float> [ %410, %109 ], [ %lazy.shifted2125, %lazy.slow ], !dbg !24
  %lazy.score2190 = phi <4 x float> [ %416, %109 ], [ %lazy.shifted2128, %lazy.slow ], !dbg !24
  %lazy.state2191 = phi [2 x <1 x float>] [ %103, %109 ], [ %lazy.state.next2132, %lazy.slow ], !dbg !24
  %lazy.state2192 = phi [2 x <8 x float>] [ %107, %109 ], [ %lazy.state.next2141, %lazy.slow ], !dbg !24
  %lazy.state2193 = phi [2 x <8 x float>] [ %106, %109 ], [ %lazy.state.next2151, %lazy.slow ], !dbg !24
  %lazy.state2194 = phi [2 x <8 x float>] [ %105, %109 ], [ %lazy.state.next2161, %lazy.slow ], !dbg !24
  %lazy.state2195 = phi [2 x <8 x float>] [ %102, %109 ], [ %lazy.state.next2171, %lazy.slow ], !dbg !24
  %lazy.base.join = phi [2 x <1 x float>] [ %104, %109 ], [ %lazy.base.state2175, %lazy.slow ], !dbg !24
  %439 = extractvalue [2 x <1 x float>] %lazy.state2191, 0, !dbg !24
  %440 = extractelement <1 x float> %439, i64 0, !dbg !24
  %441 = extractvalue [2 x <1 x float>] %lazy.state2191, 1, !dbg !24
  %442 = extractelement <1 x float> %441, i64 0, !dbg !24
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = add nsw i64 %117, %address.local938, !dbg !16
  %vdb.address1442 = trunc i64 %vdb.address to i32, !dbg !16
  %vdb.address1443 = shl i32 %vdb.address1442, 1, !dbg !16
  %vdb.address1448 = trunc nuw i64 %117 to i32, !dbg !16
  %vdb.address1449 = or disjoint i32 %address.local945, %vdb.address1448, !dbg !16
  %vdb.address1450 = shl nuw nsw i32 %vdb.address1449, 1, !dbg !16
  %vdb.dma.base = add i32 %38, 8192, !dbg !14
  %443 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1443, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %vdb.dma.base1455 = add i32 %46, 8192, !dbg !14
  %444 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1455, i32 %vdb.address1450, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %address.local949 = and i32 %address.local940, 120, !dbg !14
  %address.local951 = and i32 %address.local591, 1536, !dbg !14
  %address.local952 = or disjoint i32 %address.local949, %address.local951, !dbg !14
  %address.local953 = zext nneg i32 %address.local952 to i64, !dbg !14
  %address.local954 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local953, !dbg !14
  %445 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local954, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %446 = bitcast <4 x i32> %445 to <8 x half>, !dbg !14
  %address.local955 = and i64 %address.local518, 60, !dbg !14
  %address.local956 = add nuw nsw i64 %address.local955, %address.local518, !dbg !14
  %address.local958.idx = shl nuw nsw i64 %address.local956, 4, !dbg !14
  %address.local958 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local958.idx, !dbg !14
  %address.local959 = lshr i64 %address.local571, 4, !dbg !14
  %address.local961.idx = mul nuw nsw i64 %address.local959, 768, !dbg !14
  %address.local961 = getelementptr i8, ptr addrspace(3) %address.local958, i64 %address.local961.idx, !dbg !14
  %address.local963.idx = mul nsw i64 %address.local524, -64, !dbg !14
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local961, i64 %address.local963.idx, !dbg !14
  %address.local964 = getelementptr i8, ptr addrspace(3) %address.local963, i64 256, !dbg !14
  %447 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %448 = bitcast <4 x i32> %447 to <8 x half>, !dbg !14
  %address.local968 = getelementptr i8, ptr addrspace(3) %address.local963, i64 512, !dbg !14
  %449 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local968, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %450 = bitcast <4 x i32> %449 to <8 x half>, !dbg !14
  %address.local972 = getelementptr i8, ptr addrspace(3) %address.local963, i64 768, !dbg !14
  %451 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local972, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %452 = bitcast <4 x i32> %451 to <8 x half>, !dbg !14
  %address.local974 = getelementptr i8, ptr addrspace(3) %address.local954, i64 4096, !dbg !14
  %453 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %454 = bitcast <4 x i32> %453 to <8 x half>, !dbg !14
  %address.local978 = getelementptr i8, ptr addrspace(3) %address.local963, i64 4352, !dbg !14
  %455 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local978, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %456 = bitcast <4 x i32> %455 to <8 x half>, !dbg !14
  %address.local982 = getelementptr i8, ptr addrspace(3) %address.local963, i64 4608, !dbg !14
  %457 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local982, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %458 = bitcast <4 x i32> %457 to <8 x half>, !dbg !14
  %address.local986 = getelementptr i8, ptr addrspace(3) %address.local963, i64 4864, !dbg !14
  %459 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local986, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %460 = shufflevector <8 x half> %446, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %461 = extractvalue [2 x <8 x float>] %lazy.state2195, 0, !dbg !28
  %462 = shufflevector <8 x float> %461, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %463 = extractelement <4 x float> %lazy.score, i64 0, !dbg !25
  %464 = call float @llvm.amdgcn.exp2.f32(float %463), !dbg !25
  %465 = insertelement <4 x float> poison, float %464, i64 0, !dbg !27
  %466 = extractelement <4 x float> %lazy.score, i64 1, !dbg !25
  %467 = call float @llvm.amdgcn.exp2.f32(float %466), !dbg !25
  %468 = insertelement <4 x float> %465, float %467, i64 1, !dbg !27
  %469 = extractelement <4 x float> %lazy.score, i64 2, !dbg !25
  %470 = call float @llvm.amdgcn.exp2.f32(float %469), !dbg !25
  %471 = insertelement <4 x float> %468, float %470, i64 2, !dbg !27
  %472 = extractelement <4 x float> %lazy.score, i64 3, !dbg !25
  %473 = call float @llvm.amdgcn.exp2.f32(float %472), !dbg !25
  %474 = insertelement <4 x float> %471, float %473, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %474) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !27
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !27
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !27
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !27
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !27
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !27
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !27
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !27
  %475 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %475) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %460, <4 x float> %462) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %477 = shufflevector <8 x half> %446, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %478 = shufflevector <8 x float> %461, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %479 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %477, <4 x float> %478) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %480 = extractvalue [2 x <8 x float>] %lazy.state2195, 1, !dbg !28
  %481 = shufflevector <8 x float> %480, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %482 = extractelement <4 x float> %lazy.score2183, i64 0, !dbg !25
  %483 = call float @llvm.amdgcn.exp2.f32(float %482), !dbg !25
  %484 = insertelement <4 x float> poison, float %483, i64 0, !dbg !27
  %485 = extractelement <4 x float> %lazy.score2183, i64 1, !dbg !25
  %486 = call float @llvm.amdgcn.exp2.f32(float %485), !dbg !25
  %487 = insertelement <4 x float> %484, float %486, i64 1, !dbg !27
  %488 = extractelement <4 x float> %lazy.score2183, i64 2, !dbg !25
  %489 = call float @llvm.amdgcn.exp2.f32(float %488), !dbg !25
  %490 = insertelement <4 x float> %487, float %489, i64 2, !dbg !27
  %491 = extractelement <4 x float> %lazy.score2183, i64 3, !dbg !25
  %492 = call float @llvm.amdgcn.exp2.f32(float %491), !dbg !25
  %493 = insertelement <4 x float> %490, float %492, i64 3, !dbg !27
  %math.result404 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %493) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02196 = extractelement <4 x float> %math.result404, i32 0, !dbg !27
  %.cvtpack.f12197 = extractelement <4 x float> %math.result404, i32 1, !dbg !27
  %.cvtpack.f22198 = extractelement <4 x float> %math.result404, i32 2, !dbg !27
  %.cvtpack.f32199 = extractelement <4 x float> %math.result404, i32 3, !dbg !27
  %.cvtpack.p02200 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02196, float %.cvtpack.f12197), !dbg !27
  %.cvtpack.p12201 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22198, float %.cvtpack.f32199), !dbg !27
  %.cvtpack.v02202 = insertelement <2 x i32> poison, i32 %.cvtpack.p02200, i32 0, !dbg !27
  %.cvtpack.v12203 = insertelement <2 x i32> %.cvtpack.v02202, i32 %.cvtpack.p12201, i32 1, !dbg !27
  %494 = bitcast <2 x i32> %.cvtpack.v12203 to <4 x half>, !dbg !27
  %math.packed405 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %494) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %495 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %460, <4 x float> %481) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %496 = shufflevector <8 x float> %480, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %497 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %477, <4 x float> %496) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %498 = shufflevector <8 x half> %448, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %499 = extractelement <4 x float> %lazy.score2176, i64 0, !dbg !25
  %500 = call float @llvm.amdgcn.exp2.f32(float %499), !dbg !25
  %local.reduce93 = fadd float %464, %500, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %local.reduce93) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !27
  %502 = extractelement <4 x float> %lazy.score2176, i64 1, !dbg !25
  %503 = call float @llvm.amdgcn.exp2.f32(float %502), !dbg !25
  %local.reduce94 = fadd float %467, %503, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor1999 = call float asm sideeffect "", "=v,0"(float %local.reduce94) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %504 = insertelement <4 x float> %501, float %503, i64 1, !dbg !27
  %505 = extractelement <4 x float> %lazy.score2176, i64 2, !dbg !25
  %506 = call float @llvm.amdgcn.exp2.f32(float %505), !dbg !25
  %local.reduce95 = fadd float %470, %506, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2000 = call float asm sideeffect "", "=v,0"(float %local.reduce95) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %507 = insertelement <4 x float> %504, float %506, i64 2, !dbg !27
  %508 = extractelement <4 x float> %lazy.score2176, i64 3, !dbg !25
  %509 = call float @llvm.amdgcn.exp2.f32(float %508), !dbg !25
  %local.reduce96 = fadd float %473, %509, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2001 = call float asm sideeffect "", "=v,0"(float %local.reduce96) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %510 = insertelement <4 x float> %507, float %509, i64 3, !dbg !27
  %math.result390 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %510) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02204 = extractelement <4 x float> %math.result390, i32 0, !dbg !27
  %.cvtpack.f12205 = extractelement <4 x float> %math.result390, i32 1, !dbg !27
  %.cvtpack.f22206 = extractelement <4 x float> %math.result390, i32 2, !dbg !27
  %.cvtpack.f32207 = extractelement <4 x float> %math.result390, i32 3, !dbg !27
  %.cvtpack.p02208 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02204, float %.cvtpack.f12205), !dbg !27
  %.cvtpack.p12209 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22206, float %.cvtpack.f32207), !dbg !27
  %.cvtpack.v02210 = insertelement <2 x i32> poison, i32 %.cvtpack.p02208, i32 0, !dbg !27
  %.cvtpack.v12211 = insertelement <2 x i32> %.cvtpack.v02210, i32 %.cvtpack.p12209, i32 1, !dbg !27
  %511 = bitcast <2 x i32> %.cvtpack.v12211 to <4 x half>, !dbg !27
  %math.packed391 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %511) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %498, <4 x float> %476) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %513 = shufflevector <8 x half> %448, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %514 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %513, <4 x float> %479) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %515 = extractelement <4 x float> %lazy.score2184, i64 0, !dbg !25
  %516 = call float @llvm.amdgcn.exp2.f32(float %515), !dbg !25
  %local.reduce = fadd float %483, %516, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2002 = call float asm sideeffect "", "=v,0"(float %local.reduce) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %517 = insertelement <4 x float> poison, float %516, i64 0, !dbg !27
  %518 = extractelement <4 x float> %lazy.score2184, i64 1, !dbg !25
  %519 = call float @llvm.amdgcn.exp2.f32(float %518), !dbg !25
  %local.reduce63 = fadd float %486, %519, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2003 = call float asm sideeffect "", "=v,0"(float %local.reduce63) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %520 = insertelement <4 x float> %517, float %519, i64 1, !dbg !27
  %521 = extractelement <4 x float> %lazy.score2184, i64 2, !dbg !25
  %522 = call float @llvm.amdgcn.exp2.f32(float %521), !dbg !25
  %local.reduce64 = fadd float %489, %522, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2004 = call float asm sideeffect "", "=v,0"(float %local.reduce64) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %523 = insertelement <4 x float> %520, float %522, i64 2, !dbg !27
  %524 = extractelement <4 x float> %lazy.score2184, i64 3, !dbg !25
  %525 = call float @llvm.amdgcn.exp2.f32(float %524), !dbg !25
  %local.reduce65 = fadd float %492, %525, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2005 = call float asm sideeffect "", "=v,0"(float %local.reduce65) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %526 = insertelement <4 x float> %523, float %525, i64 3, !dbg !27
  %math.result406 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %526) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02212 = extractelement <4 x float> %math.result406, i32 0, !dbg !27
  %.cvtpack.f12213 = extractelement <4 x float> %math.result406, i32 1, !dbg !27
  %.cvtpack.f22214 = extractelement <4 x float> %math.result406, i32 2, !dbg !27
  %.cvtpack.f32215 = extractelement <4 x float> %math.result406, i32 3, !dbg !27
  %.cvtpack.p02216 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02212, float %.cvtpack.f12213), !dbg !27
  %.cvtpack.p12217 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22214, float %.cvtpack.f32215), !dbg !27
  %.cvtpack.v02218 = insertelement <2 x i32> poison, i32 %.cvtpack.p02216, i32 0, !dbg !27
  %.cvtpack.v12219 = insertelement <2 x i32> %.cvtpack.v02218, i32 %.cvtpack.p12217, i32 1, !dbg !27
  %527 = bitcast <2 x i32> %.cvtpack.v12219 to <4 x half>, !dbg !27
  %math.packed407 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %527) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %498, <4 x float> %495) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %513, <4 x float> %497) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %530 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %531 = extractelement <4 x float> %lazy.score2177, i64 0, !dbg !25
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !25
  %local.reduce97 = fadd float %k0.sum.anchor, %532, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2006 = call float asm sideeffect "", "=v,0"(float %local.reduce97) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %533 = insertelement <4 x float> poison, float %532, i64 0, !dbg !27
  %534 = extractelement <4 x float> %lazy.score2177, i64 1, !dbg !25
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !25
  %local.reduce98 = fadd float %k0.sum.anchor1999, %535, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2007 = call float asm sideeffect "", "=v,0"(float %local.reduce98) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %536 = insertelement <4 x float> %533, float %535, i64 1, !dbg !27
  %537 = extractelement <4 x float> %lazy.score2177, i64 2, !dbg !25
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !25
  %local.reduce99 = fadd float %k0.sum.anchor2000, %538, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2008 = call float asm sideeffect "", "=v,0"(float %local.reduce99) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %539 = insertelement <4 x float> %536, float %538, i64 2, !dbg !27
  %540 = extractelement <4 x float> %lazy.score2177, i64 3, !dbg !25
  %541 = call float @llvm.amdgcn.exp2.f32(float %540), !dbg !25
  %local.reduce100 = fadd float %k0.sum.anchor2001, %541, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2009 = call float asm sideeffect "", "=v,0"(float %local.reduce100) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %542 = insertelement <4 x float> %539, float %541, i64 3, !dbg !27
  %math.result392 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02220 = extractelement <4 x float> %math.result392, i32 0, !dbg !27
  %.cvtpack.f12221 = extractelement <4 x float> %math.result392, i32 1, !dbg !27
  %.cvtpack.f22222 = extractelement <4 x float> %math.result392, i32 2, !dbg !27
  %.cvtpack.f32223 = extractelement <4 x float> %math.result392, i32 3, !dbg !27
  %.cvtpack.p02224 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02220, float %.cvtpack.f12221), !dbg !27
  %.cvtpack.p12225 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22222, float %.cvtpack.f32223), !dbg !27
  %.cvtpack.v02226 = insertelement <2 x i32> poison, i32 %.cvtpack.p02224, i32 0, !dbg !27
  %.cvtpack.v12227 = insertelement <2 x i32> %.cvtpack.v02226, i32 %.cvtpack.p12225, i32 1, !dbg !27
  %543 = bitcast <2 x i32> %.cvtpack.v12227 to <4 x half>, !dbg !27
  %math.packed393 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %544 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %530, <4 x float> %512) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %545 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %546 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %545, <4 x float> %514) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %547 = extractelement <4 x float> %lazy.score2185, i64 0, !dbg !25
  %548 = call float @llvm.amdgcn.exp2.f32(float %547), !dbg !25
  %local.reduce66 = fadd float %k0.sum.anchor2002, %548, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2010 = call float asm sideeffect "", "=v,0"(float %local.reduce66) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %549 = insertelement <4 x float> poison, float %548, i64 0, !dbg !27
  %550 = extractelement <4 x float> %lazy.score2185, i64 1, !dbg !25
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !25
  %local.reduce67 = fadd float %k0.sum.anchor2003, %551, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2011 = call float asm sideeffect "", "=v,0"(float %local.reduce67) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %552 = insertelement <4 x float> %549, float %551, i64 1, !dbg !27
  %553 = extractelement <4 x float> %lazy.score2185, i64 2, !dbg !25
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !25
  %local.reduce68 = fadd float %k0.sum.anchor2004, %554, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2012 = call float asm sideeffect "", "=v,0"(float %local.reduce68) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %555 = insertelement <4 x float> %552, float %554, i64 2, !dbg !27
  %556 = extractelement <4 x float> %lazy.score2185, i64 3, !dbg !25
  %557 = call float @llvm.amdgcn.exp2.f32(float %556), !dbg !25
  %local.reduce69 = fadd float %k0.sum.anchor2005, %557, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2013 = call float asm sideeffect "", "=v,0"(float %local.reduce69) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %558 = insertelement <4 x float> %555, float %557, i64 3, !dbg !27
  %math.result408 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %558) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02228 = extractelement <4 x float> %math.result408, i32 0, !dbg !27
  %.cvtpack.f12229 = extractelement <4 x float> %math.result408, i32 1, !dbg !27
  %.cvtpack.f22230 = extractelement <4 x float> %math.result408, i32 2, !dbg !27
  %.cvtpack.f32231 = extractelement <4 x float> %math.result408, i32 3, !dbg !27
  %.cvtpack.p02232 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02228, float %.cvtpack.f12229), !dbg !27
  %.cvtpack.p12233 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22230, float %.cvtpack.f32231), !dbg !27
  %.cvtpack.v02234 = insertelement <2 x i32> poison, i32 %.cvtpack.p02232, i32 0, !dbg !27
  %.cvtpack.v12235 = insertelement <2 x i32> %.cvtpack.v02234, i32 %.cvtpack.p12233, i32 1, !dbg !27
  %559 = bitcast <2 x i32> %.cvtpack.v12235 to <4 x half>, !dbg !27
  %math.packed409 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %559) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %560 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %530, <4 x float> %528) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %561 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %545, <4 x float> %529) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %562 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %563 = extractelement <4 x float> %lazy.score2178, i64 0, !dbg !25
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !25
  %local.reduce101 = fadd float %k0.sum.anchor2006, %564, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2014 = call float asm sideeffect "", "=v,0"(float %local.reduce101) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %565 = insertelement <4 x float> poison, float %564, i64 0, !dbg !27
  %566 = extractelement <4 x float> %lazy.score2178, i64 1, !dbg !25
  %567 = call float @llvm.amdgcn.exp2.f32(float %566), !dbg !25
  %local.reduce102 = fadd float %k0.sum.anchor2007, %567, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2015 = call float asm sideeffect "", "=v,0"(float %local.reduce102) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %568 = insertelement <4 x float> %565, float %567, i64 1, !dbg !27
  %569 = extractelement <4 x float> %lazy.score2178, i64 2, !dbg !25
  %570 = call float @llvm.amdgcn.exp2.f32(float %569), !dbg !25
  %local.reduce103 = fadd float %k0.sum.anchor2008, %570, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2016 = call float asm sideeffect "", "=v,0"(float %local.reduce103) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %568, float %570, i64 2, !dbg !27
  %572 = extractelement <4 x float> %lazy.score2178, i64 3, !dbg !25
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !25
  %local.reduce104 = fadd float %k0.sum.anchor2009, %573, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2017 = call float asm sideeffect "", "=v,0"(float %local.reduce104) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %571, float %573, i64 3, !dbg !27
  %math.result394 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02236 = extractelement <4 x float> %math.result394, i32 0, !dbg !27
  %.cvtpack.f12237 = extractelement <4 x float> %math.result394, i32 1, !dbg !27
  %.cvtpack.f22238 = extractelement <4 x float> %math.result394, i32 2, !dbg !27
  %.cvtpack.f32239 = extractelement <4 x float> %math.result394, i32 3, !dbg !27
  %.cvtpack.p02240 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02236, float %.cvtpack.f12237), !dbg !27
  %.cvtpack.p12241 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22238, float %.cvtpack.f32239), !dbg !27
  %.cvtpack.v02242 = insertelement <2 x i32> poison, i32 %.cvtpack.p02240, i32 0, !dbg !27
  %.cvtpack.v12243 = insertelement <2 x i32> %.cvtpack.v02242, i32 %.cvtpack.p12241, i32 1, !dbg !27
  %575 = bitcast <2 x i32> %.cvtpack.v12243 to <4 x half>, !dbg !27
  %math.packed395 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %562, <4 x float> %544) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %577 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %577, <4 x float> %546) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = extractelement <4 x float> %lazy.score2186, i64 0, !dbg !25
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !25
  %local.reduce70 = fadd float %k0.sum.anchor2010, %580, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2018 = call float asm sideeffect "", "=v,0"(float %local.reduce70) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %581 = insertelement <4 x float> poison, float %580, i64 0, !dbg !27
  %582 = extractelement <4 x float> %lazy.score2186, i64 1, !dbg !25
  %583 = call float @llvm.amdgcn.exp2.f32(float %582), !dbg !25
  %local.reduce71 = fadd float %k0.sum.anchor2011, %583, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2019 = call float asm sideeffect "", "=v,0"(float %local.reduce71) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %584 = insertelement <4 x float> %581, float %583, i64 1, !dbg !27
  %585 = extractelement <4 x float> %lazy.score2186, i64 2, !dbg !25
  %586 = call float @llvm.amdgcn.exp2.f32(float %585), !dbg !25
  %local.reduce72 = fadd float %k0.sum.anchor2012, %586, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2020 = call float asm sideeffect "", "=v,0"(float %local.reduce72) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %587 = insertelement <4 x float> %584, float %586, i64 2, !dbg !27
  %588 = extractelement <4 x float> %lazy.score2186, i64 3, !dbg !25
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !25
  %local.reduce73 = fadd float %k0.sum.anchor2013, %589, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2021 = call float asm sideeffect "", "=v,0"(float %local.reduce73) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %590 = insertelement <4 x float> %587, float %589, i64 3, !dbg !27
  %math.result410 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %590) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02244 = extractelement <4 x float> %math.result410, i32 0, !dbg !27
  %.cvtpack.f12245 = extractelement <4 x float> %math.result410, i32 1, !dbg !27
  %.cvtpack.f22246 = extractelement <4 x float> %math.result410, i32 2, !dbg !27
  %.cvtpack.f32247 = extractelement <4 x float> %math.result410, i32 3, !dbg !27
  %.cvtpack.p02248 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02244, float %.cvtpack.f12245), !dbg !27
  %.cvtpack.p12249 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22246, float %.cvtpack.f32247), !dbg !27
  %.cvtpack.v02250 = insertelement <2 x i32> poison, i32 %.cvtpack.p02248, i32 0, !dbg !27
  %.cvtpack.v12251 = insertelement <2 x i32> %.cvtpack.v02250, i32 %.cvtpack.p12249, i32 1, !dbg !27
  %591 = bitcast <2 x i32> %.cvtpack.v12251 to <4 x half>, !dbg !27
  %math.packed411 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %591) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %562, <4 x float> %560) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %593 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %577, <4 x float> %561) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %594 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %595 = extractelement <4 x float> %lazy.score2179, i64 0, !dbg !25
  %596 = call float @llvm.amdgcn.exp2.f32(float %595), !dbg !25
  %local.reduce105 = fadd float %k0.sum.anchor2014, %596, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2022 = call float asm sideeffect "", "=v,0"(float %local.reduce105) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %597 = insertelement <4 x float> poison, float %596, i64 0, !dbg !27
  %598 = extractelement <4 x float> %lazy.score2179, i64 1, !dbg !25
  %599 = call float @llvm.amdgcn.exp2.f32(float %598), !dbg !25
  %local.reduce106 = fadd float %k0.sum.anchor2015, %599, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2023 = call float asm sideeffect "", "=v,0"(float %local.reduce106) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %600 = insertelement <4 x float> %597, float %599, i64 1, !dbg !27
  %601 = extractelement <4 x float> %lazy.score2179, i64 2, !dbg !25
  %602 = call float @llvm.amdgcn.exp2.f32(float %601), !dbg !25
  %local.reduce107 = fadd float %k0.sum.anchor2016, %602, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2024 = call float asm sideeffect "", "=v,0"(float %local.reduce107) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %603 = insertelement <4 x float> %600, float %602, i64 2, !dbg !27
  %604 = extractelement <4 x float> %lazy.score2179, i64 3, !dbg !25
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !25
  %local.reduce108 = fadd float %k0.sum.anchor2017, %605, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2025 = call float asm sideeffect "", "=v,0"(float %local.reduce108) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %606 = insertelement <4 x float> %603, float %605, i64 3, !dbg !27
  %math.result396 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02252 = extractelement <4 x float> %math.result396, i32 0, !dbg !27
  %.cvtpack.f12253 = extractelement <4 x float> %math.result396, i32 1, !dbg !27
  %.cvtpack.f22254 = extractelement <4 x float> %math.result396, i32 2, !dbg !27
  %.cvtpack.f32255 = extractelement <4 x float> %math.result396, i32 3, !dbg !27
  %.cvtpack.p02256 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02252, float %.cvtpack.f12253), !dbg !27
  %.cvtpack.p12257 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22254, float %.cvtpack.f32255), !dbg !27
  %.cvtpack.v02258 = insertelement <2 x i32> poison, i32 %.cvtpack.p02256, i32 0, !dbg !27
  %.cvtpack.v12259 = insertelement <2 x i32> %.cvtpack.v02258, i32 %.cvtpack.p12257, i32 1, !dbg !27
  %607 = bitcast <2 x i32> %.cvtpack.v12259 to <4 x half>, !dbg !27
  %math.packed397 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %594, <4 x float> %576) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %609 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %610 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %609, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %611 = extractelement <4 x float> %lazy.score2187, i64 0, !dbg !25
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !25
  %local.reduce74 = fadd float %k0.sum.anchor2018, %612, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2026 = call float asm sideeffect "", "=v,0"(float %local.reduce74) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %613 = insertelement <4 x float> poison, float %612, i64 0, !dbg !27
  %614 = extractelement <4 x float> %lazy.score2187, i64 1, !dbg !25
  %615 = call float @llvm.amdgcn.exp2.f32(float %614), !dbg !25
  %local.reduce75 = fadd float %k0.sum.anchor2019, %615, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2027 = call float asm sideeffect "", "=v,0"(float %local.reduce75) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %616 = insertelement <4 x float> %613, float %615, i64 1, !dbg !27
  %617 = extractelement <4 x float> %lazy.score2187, i64 2, !dbg !25
  %618 = call float @llvm.amdgcn.exp2.f32(float %617), !dbg !25
  %local.reduce76 = fadd float %k0.sum.anchor2020, %618, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2028 = call float asm sideeffect "", "=v,0"(float %local.reduce76) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %619 = insertelement <4 x float> %616, float %618, i64 2, !dbg !27
  %620 = extractelement <4 x float> %lazy.score2187, i64 3, !dbg !25
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !25
  %local.reduce77 = fadd float %k0.sum.anchor2021, %621, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2029 = call float asm sideeffect "", "=v,0"(float %local.reduce77) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> %619, float %621, i64 3, !dbg !27
  %math.result412 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %622) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02260 = extractelement <4 x float> %math.result412, i32 0, !dbg !27
  %.cvtpack.f12261 = extractelement <4 x float> %math.result412, i32 1, !dbg !27
  %.cvtpack.f22262 = extractelement <4 x float> %math.result412, i32 2, !dbg !27
  %.cvtpack.f32263 = extractelement <4 x float> %math.result412, i32 3, !dbg !27
  %.cvtpack.p02264 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02260, float %.cvtpack.f12261), !dbg !27
  %.cvtpack.p12265 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22262, float %.cvtpack.f32263), !dbg !27
  %.cvtpack.v02266 = insertelement <2 x i32> poison, i32 %.cvtpack.p02264, i32 0, !dbg !27
  %.cvtpack.v12267 = insertelement <2 x i32> %.cvtpack.v02266, i32 %.cvtpack.p12265, i32 1, !dbg !27
  %623 = bitcast <2 x i32> %.cvtpack.v12267 to <4 x half>, !dbg !27
  %math.packed413 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %623) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %594, <4 x float> %592) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %609, <4 x float> %593) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %626 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %627 = extractelement <4 x float> %lazy.score2180, i64 0, !dbg !25
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !25
  %local.reduce109 = fadd float %k0.sum.anchor2022, %628, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2030 = call float asm sideeffect "", "=v,0"(float %local.reduce109) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !27
  %630 = extractelement <4 x float> %lazy.score2180, i64 1, !dbg !25
  %631 = call float @llvm.amdgcn.exp2.f32(float %630), !dbg !25
  %local.reduce110 = fadd float %k0.sum.anchor2023, %631, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2031 = call float asm sideeffect "", "=v,0"(float %local.reduce110) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %632 = insertelement <4 x float> %629, float %631, i64 1, !dbg !27
  %633 = extractelement <4 x float> %lazy.score2180, i64 2, !dbg !25
  %634 = call float @llvm.amdgcn.exp2.f32(float %633), !dbg !25
  %local.reduce111 = fadd float %k0.sum.anchor2024, %634, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2032 = call float asm sideeffect "", "=v,0"(float %local.reduce111) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %635 = insertelement <4 x float> %632, float %634, i64 2, !dbg !27
  %636 = extractelement <4 x float> %lazy.score2180, i64 3, !dbg !25
  %637 = call float @llvm.amdgcn.exp2.f32(float %636), !dbg !25
  %local.reduce112 = fadd float %k0.sum.anchor2025, %637, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2033 = call float asm sideeffect "", "=v,0"(float %local.reduce112) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %638 = insertelement <4 x float> %635, float %637, i64 3, !dbg !27
  %math.result398 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %638) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02268 = extractelement <4 x float> %math.result398, i32 0, !dbg !27
  %.cvtpack.f12269 = extractelement <4 x float> %math.result398, i32 1, !dbg !27
  %.cvtpack.f22270 = extractelement <4 x float> %math.result398, i32 2, !dbg !27
  %.cvtpack.f32271 = extractelement <4 x float> %math.result398, i32 3, !dbg !27
  %.cvtpack.p02272 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02268, float %.cvtpack.f12269), !dbg !27
  %.cvtpack.p12273 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22270, float %.cvtpack.f32271), !dbg !27
  %.cvtpack.v02274 = insertelement <2 x i32> poison, i32 %.cvtpack.p02272, i32 0, !dbg !27
  %.cvtpack.v12275 = insertelement <2 x i32> %.cvtpack.v02274, i32 %.cvtpack.p12273, i32 1, !dbg !27
  %639 = bitcast <2 x i32> %.cvtpack.v12275 to <4 x half>, !dbg !27
  %math.packed399 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %639) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %626, <4 x float> %608) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %641 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %641, <4 x float> %610) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %643 = extractelement <4 x float> %lazy.score2188, i64 0, !dbg !25
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !25
  %local.reduce78 = fadd float %k0.sum.anchor2026, %644, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2034 = call float asm sideeffect "", "=v,0"(float %local.reduce78) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %645 = insertelement <4 x float> poison, float %644, i64 0, !dbg !27
  %646 = extractelement <4 x float> %lazy.score2188, i64 1, !dbg !25
  %647 = call float @llvm.amdgcn.exp2.f32(float %646), !dbg !25
  %local.reduce79 = fadd float %k0.sum.anchor2027, %647, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2035 = call float asm sideeffect "", "=v,0"(float %local.reduce79) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %648 = insertelement <4 x float> %645, float %647, i64 1, !dbg !27
  %649 = extractelement <4 x float> %lazy.score2188, i64 2, !dbg !25
  %650 = call float @llvm.amdgcn.exp2.f32(float %649), !dbg !25
  %local.reduce80 = fadd float %k0.sum.anchor2028, %650, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2036 = call float asm sideeffect "", "=v,0"(float %local.reduce80) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %651 = insertelement <4 x float> %648, float %650, i64 2, !dbg !27
  %652 = extractelement <4 x float> %lazy.score2188, i64 3, !dbg !25
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !25
  %local.reduce81 = fadd float %k0.sum.anchor2029, %653, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2037 = call float asm sideeffect "", "=v,0"(float %local.reduce81) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %654 = insertelement <4 x float> %651, float %653, i64 3, !dbg !27
  %math.result414 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02276 = extractelement <4 x float> %math.result414, i32 0, !dbg !27
  %.cvtpack.f12277 = extractelement <4 x float> %math.result414, i32 1, !dbg !27
  %.cvtpack.f22278 = extractelement <4 x float> %math.result414, i32 2, !dbg !27
  %.cvtpack.f32279 = extractelement <4 x float> %math.result414, i32 3, !dbg !27
  %.cvtpack.p02280 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02276, float %.cvtpack.f12277), !dbg !27
  %.cvtpack.p12281 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22278, float %.cvtpack.f32279), !dbg !27
  %.cvtpack.v02282 = insertelement <2 x i32> poison, i32 %.cvtpack.p02280, i32 0, !dbg !27
  %.cvtpack.v12283 = insertelement <2 x i32> %.cvtpack.v02282, i32 %.cvtpack.p12281, i32 1, !dbg !27
  %655 = bitcast <2 x i32> %.cvtpack.v12283 to <4 x half>, !dbg !27
  %math.packed415 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %626, <4 x float> %624) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %641, <4 x float> %625) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %658 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = extractelement <4 x float> %lazy.score2181, i64 0, !dbg !25
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !25
  %local.reduce113 = fadd float %k0.sum.anchor2030, %660, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2038 = call float asm sideeffect "", "=v,0"(float %local.reduce113) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %661 = insertelement <4 x float> poison, float %660, i64 0, !dbg !27
  %662 = extractelement <4 x float> %lazy.score2181, i64 1, !dbg !25
  %663 = call float @llvm.amdgcn.exp2.f32(float %662), !dbg !25
  %local.reduce114 = fadd float %k0.sum.anchor2031, %663, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2039 = call float asm sideeffect "", "=v,0"(float %local.reduce114) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %664 = insertelement <4 x float> %661, float %663, i64 1, !dbg !27
  %665 = extractelement <4 x float> %lazy.score2181, i64 2, !dbg !25
  %666 = call float @llvm.amdgcn.exp2.f32(float %665), !dbg !25
  %local.reduce115 = fadd float %k0.sum.anchor2032, %666, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2040 = call float asm sideeffect "", "=v,0"(float %local.reduce115) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> %664, float %666, i64 2, !dbg !27
  %668 = extractelement <4 x float> %lazy.score2181, i64 3, !dbg !25
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !25
  %local.reduce116 = fadd float %k0.sum.anchor2033, %669, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2041 = call float asm sideeffect "", "=v,0"(float %local.reduce116) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %667, float %669, i64 3, !dbg !27
  %math.result400 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02284 = extractelement <4 x float> %math.result400, i32 0, !dbg !27
  %.cvtpack.f12285 = extractelement <4 x float> %math.result400, i32 1, !dbg !27
  %.cvtpack.f22286 = extractelement <4 x float> %math.result400, i32 2, !dbg !27
  %.cvtpack.f32287 = extractelement <4 x float> %math.result400, i32 3, !dbg !27
  %.cvtpack.p02288 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02284, float %.cvtpack.f12285), !dbg !27
  %.cvtpack.p12289 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22286, float %.cvtpack.f32287), !dbg !27
  %.cvtpack.v02290 = insertelement <2 x i32> poison, i32 %.cvtpack.p02288, i32 0, !dbg !27
  %.cvtpack.v12291 = insertelement <2 x i32> %.cvtpack.v02290, i32 %.cvtpack.p12289, i32 1, !dbg !27
  %671 = bitcast <2 x i32> %.cvtpack.v12291 to <4 x half>, !dbg !27
  %math.packed401 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %658, <4 x float> %640) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %673 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %673, <4 x float> %642) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %675 = extractelement <4 x float> %lazy.score2189, i64 0, !dbg !25
  %676 = call float @llvm.amdgcn.exp2.f32(float %675), !dbg !25
  %local.reduce82 = fadd float %k0.sum.anchor2034, %676, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2042 = call float asm sideeffect "", "=v,0"(float %local.reduce82) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %677 = insertelement <4 x float> poison, float %676, i64 0, !dbg !27
  %678 = extractelement <4 x float> %lazy.score2189, i64 1, !dbg !25
  %679 = call float @llvm.amdgcn.exp2.f32(float %678), !dbg !25
  %local.reduce83 = fadd float %k0.sum.anchor2035, %679, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2043 = call float asm sideeffect "", "=v,0"(float %local.reduce83) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %680 = insertelement <4 x float> %677, float %679, i64 1, !dbg !27
  %681 = extractelement <4 x float> %lazy.score2189, i64 2, !dbg !25
  %682 = call float @llvm.amdgcn.exp2.f32(float %681), !dbg !25
  %local.reduce84 = fadd float %k0.sum.anchor2036, %682, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2044 = call float asm sideeffect "", "=v,0"(float %local.reduce84) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %683 = insertelement <4 x float> %680, float %682, i64 2, !dbg !27
  %684 = extractelement <4 x float> %lazy.score2189, i64 3, !dbg !25
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !25
  %local.reduce85 = fadd float %k0.sum.anchor2037, %685, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2045 = call float asm sideeffect "", "=v,0"(float %local.reduce85) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %686 = insertelement <4 x float> %683, float %685, i64 3, !dbg !27
  %math.result416 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %686) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02292 = extractelement <4 x float> %math.result416, i32 0, !dbg !27
  %.cvtpack.f12293 = extractelement <4 x float> %math.result416, i32 1, !dbg !27
  %.cvtpack.f22294 = extractelement <4 x float> %math.result416, i32 2, !dbg !27
  %.cvtpack.f32295 = extractelement <4 x float> %math.result416, i32 3, !dbg !27
  %.cvtpack.p02296 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02292, float %.cvtpack.f12293), !dbg !27
  %.cvtpack.p12297 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22294, float %.cvtpack.f32295), !dbg !27
  %.cvtpack.v02298 = insertelement <2 x i32> poison, i32 %.cvtpack.p02296, i32 0, !dbg !27
  %.cvtpack.v12299 = insertelement <2 x i32> %.cvtpack.v02298, i32 %.cvtpack.p12297, i32 1, !dbg !27
  %687 = bitcast <2 x i32> %.cvtpack.v12299 to <4 x half>, !dbg !27
  %math.packed417 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %687) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %688 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %658, <4 x float> %656) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %689 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %673, <4 x float> %657) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %690 = bitcast <4 x i32> %459 to <8 x half>, !dbg !14
  %691 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %692 = extractelement <4 x float> %lazy.score2182, i64 0, !dbg !25
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !25
  %local.reduce117 = fadd float %k0.sum.anchor2038, %693, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2046 = call float asm sideeffect "", "=v,0"(float %local.reduce117) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %694 = insertelement <4 x float> poison, float %693, i64 0, !dbg !27
  %695 = extractelement <4 x float> %lazy.score2182, i64 1, !dbg !25
  %696 = call float @llvm.amdgcn.exp2.f32(float %695), !dbg !25
  %local.reduce118 = fadd float %k0.sum.anchor2039, %696, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2047 = call float asm sideeffect "", "=v,0"(float %local.reduce118) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce121 = fadd float %k0.sum.anchor2046, %k0.sum.anchor2047, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2048 = call float asm sideeffect "", "=v,0"(float %local.reduce121) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %697 = insertelement <4 x float> %694, float %696, i64 1, !dbg !27
  %698 = extractelement <4 x float> %lazy.score2182, i64 2, !dbg !25
  %699 = call float @llvm.amdgcn.exp2.f32(float %698), !dbg !25
  %local.reduce119 = fadd float %k0.sum.anchor2040, %699, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2049 = call float asm sideeffect "", "=v,0"(float %local.reduce119) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %700 = insertelement <4 x float> %697, float %699, i64 2, !dbg !27
  %701 = extractelement <4 x float> %lazy.score2182, i64 3, !dbg !25
  %702 = call float @llvm.amdgcn.exp2.f32(float %701), !dbg !25
  %local.reduce120 = fadd float %k0.sum.anchor2041, %702, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2050 = call float asm sideeffect "", "=v,0"(float %local.reduce120) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce122 = fadd float %k0.sum.anchor2049, %k0.sum.anchor2050, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2051 = call float asm sideeffect "", "=v,0"(float %local.reduce122) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %703 = fadd float %k0.sum.anchor2048, %k0.sum.anchor2051, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2052 = call float asm sideeffect "", "=v,0"(float %703) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %704 = fadd float %k0.sum.anchor2052, %440, !dbg !26
  %k0.sum.anchor2053 = call float asm sideeffect "", "=v,0"(float %704) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %705 = insertelement <1 x float> poison, float %k0.sum.anchor2053, i64 0, !dbg !6
  %706 = insertvalue [2 x <1 x float>] poison, <1 x float> %705, 0, !dbg !6
  %707 = insertelement <4 x float> %700, float %702, i64 3, !dbg !27
  %math.result402 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %707) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02300 = extractelement <4 x float> %math.result402, i32 0, !dbg !27
  %.cvtpack.f12301 = extractelement <4 x float> %math.result402, i32 1, !dbg !27
  %.cvtpack.f22302 = extractelement <4 x float> %math.result402, i32 2, !dbg !27
  %.cvtpack.f32303 = extractelement <4 x float> %math.result402, i32 3, !dbg !27
  %.cvtpack.p02304 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02300, float %.cvtpack.f12301), !dbg !27
  %.cvtpack.p12305 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22302, float %.cvtpack.f32303), !dbg !27
  %.cvtpack.v02306 = insertelement <2 x i32> poison, i32 %.cvtpack.p02304, i32 0, !dbg !27
  %.cvtpack.v12307 = insertelement <2 x i32> %.cvtpack.v02306, i32 %.cvtpack.p12305, i32 1, !dbg !27
  %708 = bitcast <2 x i32> %.cvtpack.v12307 to <4 x half>, !dbg !27
  %math.packed403 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %708) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %709 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %691, <4 x float> %672) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %710 = shufflevector <8 x half> %690, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %711 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %710, <4 x float> %674) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %712 = extractelement <4 x float> %lazy.score2190, i64 0, !dbg !25
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !25
  %local.reduce86 = fadd float %k0.sum.anchor2042, %713, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2054 = call float asm sideeffect "", "=v,0"(float %local.reduce86) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %714 = insertelement <4 x float> poison, float %713, i64 0, !dbg !27
  %715 = extractelement <4 x float> %lazy.score2190, i64 1, !dbg !25
  %716 = call float @llvm.amdgcn.exp2.f32(float %715), !dbg !25
  %local.reduce87 = fadd float %k0.sum.anchor2043, %716, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2055 = call float asm sideeffect "", "=v,0"(float %local.reduce87) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce90 = fadd float %k0.sum.anchor2054, %k0.sum.anchor2055, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2056 = call float asm sideeffect "", "=v,0"(float %local.reduce90) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %717 = insertelement <4 x float> %714, float %716, i64 1, !dbg !27
  %718 = extractelement <4 x float> %lazy.score2190, i64 2, !dbg !25
  %719 = call float @llvm.amdgcn.exp2.f32(float %718), !dbg !25
  %local.reduce88 = fadd float %k0.sum.anchor2044, %719, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2057 = call float asm sideeffect "", "=v,0"(float %local.reduce88) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %720 = insertelement <4 x float> %717, float %719, i64 2, !dbg !27
  %721 = extractelement <4 x float> %lazy.score2190, i64 3, !dbg !25
  %722 = call float @llvm.amdgcn.exp2.f32(float %721), !dbg !25
  %local.reduce89 = fadd float %k0.sum.anchor2045, %722, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2058 = call float asm sideeffect "", "=v,0"(float %local.reduce89) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %local.reduce91 = fadd float %k0.sum.anchor2057, %k0.sum.anchor2058, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2059 = call float asm sideeffect "", "=v,0"(float %local.reduce91) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %723 = fadd float %k0.sum.anchor2056, %k0.sum.anchor2059, !dbg !26, !frisk.local_reduction !1
  %k0.sum.anchor2060 = call float asm sideeffect "", "=v,0"(float %723) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %724 = fadd float %k0.sum.anchor2060, %442, !dbg !26
  %k0.sum.anchor2061 = call float asm sideeffect "", "=v,0"(float %724) #9, !dbg !26, !frisk.cross_bn_sum_anchor !1
  %725 = insertelement <1 x float> poison, float %k0.sum.anchor2061, i64 0, !dbg !6
  %726 = insertvalue [2 x <1 x float>] %706, <1 x float> %725, 1, !dbg !6
  %727 = insertelement <4 x float> %720, float %722, i64 3, !dbg !27
  %math.result418 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %727) #9, !dbg !27, !frisk.register_lifetimes !1
  %.cvtpack.f02308 = extractelement <4 x float> %math.result418, i32 0, !dbg !27
  %.cvtpack.f12309 = extractelement <4 x float> %math.result418, i32 1, !dbg !27
  %.cvtpack.f22310 = extractelement <4 x float> %math.result418, i32 2, !dbg !27
  %.cvtpack.f32311 = extractelement <4 x float> %math.result418, i32 3, !dbg !27
  %.cvtpack.p02312 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02308, float %.cvtpack.f12309), !dbg !27
  %.cvtpack.p12313 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22310, float %.cvtpack.f32311), !dbg !27
  %.cvtpack.v02314 = insertelement <2 x i32> poison, i32 %.cvtpack.p02312, i32 0, !dbg !27
  %.cvtpack.v12315 = insertelement <2 x i32> %.cvtpack.v02314, i32 %.cvtpack.p12313, i32 1, !dbg !27
  %728 = bitcast <2 x i32> %.cvtpack.v12315 to <4 x half>, !dbg !27
  %math.packed419 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %728) #9, !dbg !26, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %729 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %691, <4 x float> %688) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %730 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %710, <4 x float> %689) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1463 = add nsw i64 %195, %address.local938, !dbg !16
  %vdb.address1464 = trunc i64 %vdb.address1463 to i32, !dbg !16
  %vdb.address1465 = shl i32 %vdb.address1464, 1, !dbg !16
  %vdb.address1470 = trunc nuw i64 %195 to i32, !dbg !16
  %vdb.address1471 = or disjoint i32 %address.local945, %vdb.address1470, !dbg !16
  %vdb.address1472 = shl nuw nsw i32 %vdb.address1471, 1, !dbg !16
  %731 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %vdb.address1465, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %732 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %vdb.address1472, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local954, i64 8192, !dbg !14
  %733 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %734 = bitcast <4 x i32> %733 to <8 x half>, !dbg !14
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8448, !dbg !14
  %735 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %736 = bitcast <4 x i32> %735 to <8 x half>, !dbg !14
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8704, !dbg !14
  %737 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %738 = bitcast <4 x i32> %737 to <8 x half>, !dbg !14
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local963, i64 8960, !dbg !14
  %739 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %740 = bitcast <4 x i32> %739 to <8 x half>, !dbg !14
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local954, i64 12288, !dbg !14
  %741 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %742 = bitcast <4 x i32> %741 to <8 x half>, !dbg !14
  %vdb.read1460 = getelementptr i8, ptr addrspace(3) %address.local963, i64 12544, !dbg !14
  %743 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %744 = bitcast <4 x i32> %743 to <8 x half>, !dbg !14
  %vdb.read1461 = getelementptr i8, ptr addrspace(3) %address.local963, i64 12800, !dbg !14
  %745 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %746 = bitcast <4 x i32> %745 to <8 x half>, !dbg !14
  %vdb.read1462 = getelementptr i8, ptr addrspace(3) %address.local963, i64 13056, !dbg !14
  %747 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1462, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %748 = shufflevector <8 x half> %734, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %749 = extractvalue [2 x <8 x float>] %lazy.state2194, 0, !dbg !28
  %750 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %751 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %748, <4 x float> %750) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %752 = shufflevector <8 x half> %734, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %753 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %754 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %752, <4 x float> %753) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %755 = extractvalue [2 x <8 x float>] %lazy.state2194, 1, !dbg !28
  %756 = shufflevector <8 x float> %755, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %748, <4 x float> %756) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %758 = shufflevector <8 x float> %755, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %752, <4 x float> %758) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %760 = shufflevector <8 x half> %736, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %761 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %760, <4 x float> %751) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %762 = shufflevector <8 x half> %736, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %763 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %762, <4 x float> %754) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %764 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %760, <4 x float> %757) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %765 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %762, <4 x float> %759) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %766 = shufflevector <8 x half> %738, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %767 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %766, <4 x float> %761) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %768 = shufflevector <8 x half> %738, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %769 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %768, <4 x float> %763) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %770 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %766, <4 x float> %764) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %771 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %768, <4 x float> %765) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %772 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %773 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %772, <4 x float> %767) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %774 = shufflevector <8 x half> %740, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %775 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %774, <4 x float> %769) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %772, <4 x float> %770) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %774, <4 x float> %771) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %778, <4 x float> %773) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %780 = shufflevector <8 x half> %742, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %781 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %780, <4 x float> %775) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %782 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %778, <4 x float> %776) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %783 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %780, <4 x float> %777) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %784 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %785 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %784, <4 x float> %779) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %786 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %787 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %786, <4 x float> %781) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %788 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %784, <4 x float> %782) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %789 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %786, <4 x float> %783) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %790 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %791 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %790, <4 x float> %785) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %792 = shufflevector <8 x half> %746, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %793 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %792, <4 x float> %787) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %794 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %790, <4 x float> %788) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %795 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %792, <4 x float> %789) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %796 = bitcast <4 x i32> %747 to <8 x half>, !dbg !14
  %797 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %797, <4 x float> %791) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %799, <4 x float> %793) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %797, <4 x float> %794) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %802 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %799, <4 x float> %795) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1477 = add nsw i64 %248, %address.local938, !dbg !16
  %vdb.address1478 = trunc i64 %vdb.address1477 to i32, !dbg !16
  %vdb.address1479 = shl i32 %vdb.address1478, 1, !dbg !16
  %vdb.address1484 = trunc nuw i64 %248 to i32, !dbg !16
  %vdb.address1485 = or disjoint i32 %address.local945, %vdb.address1484, !dbg !16
  %vdb.address1486 = shl nuw nsw i32 %vdb.address1485, 1, !dbg !16
  %803 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1479, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %804 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1455, i32 %vdb.address1486, <4 x i32> %dma.buffer.rsrc1362) #10, !dbg !16
  %805 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local954, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %806 = bitcast <4 x i32> %805 to <8 x half>, !dbg !14
  %807 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local964, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %808 = bitcast <4 x i32> %807 to <8 x half>, !dbg !14
  %809 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local968, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %810 = bitcast <4 x i32> %809 to <8 x half>, !dbg !14
  %811 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local972, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %812 = bitcast <4 x i32> %811 to <8 x half>, !dbg !14
  %813 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local974, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %814 = bitcast <4 x i32> %813 to <8 x half>, !dbg !14
  %815 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local978, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %816 = bitcast <4 x i32> %815 to <8 x half>, !dbg !14
  %817 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local982, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %818 = bitcast <4 x i32> %817 to <8 x half>, !dbg !14
  %819 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local986, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %820 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %821 = extractvalue [2 x <8 x float>] %lazy.state2193, 0, !dbg !28
  %822 = shufflevector <8 x float> %821, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %820, <4 x float> %822) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %824 = shufflevector <8 x half> %806, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %825 = shufflevector <8 x float> %821, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %824, <4 x float> %825) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %827 = extractvalue [2 x <8 x float>] %lazy.state2193, 1, !dbg !28
  %828 = shufflevector <8 x float> %827, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %829 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %820, <4 x float> %828) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %830 = shufflevector <8 x float> %827, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %831 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %824, <4 x float> %830) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %832 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %833 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %832, <4 x float> %823) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %834 = shufflevector <8 x half> %808, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %835 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %834, <4 x float> %826) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %836 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %832, <4 x float> %829) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %837 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %834, <4 x float> %831) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %838 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %839 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %838, <4 x float> %833) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %840 = shufflevector <8 x half> %810, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %841 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %840, <4 x float> %835) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %842 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %838, <4 x float> %836) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %843 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %840, <4 x float> %837) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %844 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %845 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %844, <4 x float> %839) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %846 = shufflevector <8 x half> %812, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %847 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %846, <4 x float> %841) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %848 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %844, <4 x float> %842) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %846, <4 x float> %843) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %850, <4 x float> %845) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %852 = shufflevector <8 x half> %814, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %853 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %852, <4 x float> %847) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %854 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %850, <4 x float> %848) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %855 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %852, <4 x float> %849) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %856 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %857 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %856, <4 x float> %851) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %858 = shufflevector <8 x half> %816, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %859 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %858, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %860 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %856, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %861 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %858, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %862 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %863 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %862, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %864 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %865 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %864, <4 x float> %859) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %866 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %862, <4 x float> %860) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %867 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %864, <4 x float> %861) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %868 = bitcast <4 x i32> %819 to <8 x half>, !dbg !14
  %869 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %870 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %869, <4 x float> %863) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %871 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %872 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %871, <4 x float> %865) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %873 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %869, <4 x float> %866) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %874 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %871, <4 x float> %867) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %875 = add nuw nsw i64 %101, 1, !dbg !6
  %k0.has.next = icmp ult i64 %875, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %875, 14, !dbg !13
  %k0.next.2072 = or disjoint i64 %111, %k0.next., !dbg !13
  %k0.next.kdma.source1517 = or disjoint i64 %k0.next.2072, %kdma.source1516, !dbg !13
  %k0.next.kdma.source1518 = trunc nuw i64 %k0.next.kdma.source1517 to i32, !dbg !13
  %k0.next.kdma.source1519 = shl nuw nsw i32 %k0.next.kdma.source1518, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1519, %kdma.intra, !dbg !13
  %k0.next.2073 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1537 = or disjoint i64 %k0.next.2072, %kdma.source1536, !dbg !13
  %k0.next.kdma.source1538 = trunc nuw i64 %k0.next.kdma.source1537 to i32, !dbg !13
  %k0.next.kdma.source1539 = shl nuw nsw i32 %k0.next.kdma.source1538, 1, !dbg !13
  %k0.next.kdma.offset1540 = or disjoint i32 %k0.next.kdma.source1539, %kdma.intra.hi, !dbg !13
  %k0.next.2074 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1541, i32 %k0.next.kdma.offset1540, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1561 = or disjoint i64 %k0.next.2072, %kdma.source1560, !dbg !13
  %k0.next.kdma.source1562 = trunc nuw i64 %k0.next.kdma.source1561 to i32, !dbg !13
  %k0.next.kdma.source1563 = shl nuw nsw i32 %k0.next.kdma.source1562, 1, !dbg !13
  %k0.next.kdma.offset1564 = add nuw i32 %k0.next.kdma.source1563, %kdma.intra, !dbg !13
  %k0.next.2075 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.next.kdma.offset1564, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1585 = or disjoint i64 %k0.next.2072, %kdma.source1584, !dbg !13
  %k0.next.kdma.source1586 = trunc nuw i64 %k0.next.kdma.source1585 to i32, !dbg !13
  %k0.next.kdma.source1587 = shl nuw nsw i32 %k0.next.kdma.source1586, 1, !dbg !13
  %k0.next.kdma.offset1588 = add nuw i32 %k0.next.kdma.source1587, %kdma.intra.hi, !dbg !13
  %k0.next.2076 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1589, i32 %k0.next.kdma.offset1588, <4 x i32> %dma.buffer.rsrc1378) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %876 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !14
  %878 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !14
  %880 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !14
  %882 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !14
  %884 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !14
  %886 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !14
  %888 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %889 = bitcast <4 x i32> %888 to <8 x half>, !dbg !14
  %890 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1462, i32 0) #0, !dbg !14, !frisk.v_double_buffer !34
  %891 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %892 = extractvalue [2 x <8 x float>] %lazy.state2192, 0, !dbg !28
  %893 = shufflevector <8 x float> %892, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %891, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %896 = shufflevector <8 x float> %892, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %895, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %898 = extractvalue [2 x <8 x float>] %lazy.state2192, 1, !dbg !28
  %899 = shufflevector <8 x float> %898, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %891, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !28
  %901 = shufflevector <8 x float> %898, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed405, <4 x half> %895, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %903, <4 x float> %894) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %905, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %903, <4 x float> %900) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed407, <4 x half> %905, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %909, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %911, <4 x float> %906) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %909, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed409, <4 x half> %911, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %915, <4 x float> %910) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %917, <4 x float> %912) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %915, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed411, <4 x half> %917, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %921, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %923, <4 x float> %918) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %925 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %921, <4 x float> %919) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %926 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed413, <4 x half> %923, <4 x float> %920) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %927 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %928 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %927, <4 x float> %922) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %929 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %930 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %929, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %931 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %927, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %932 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed415, <4 x half> %929, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %933 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %934 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %933, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %935 = shufflevector <8 x half> %889, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %936 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %935, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %937 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %933, <4 x float> %931) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %938 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed417, <4 x half> %935, <4 x float> %932) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %939 = bitcast <4 x i32> %890 to <8 x half>, !dbg !14
  %940 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %941 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %940, <4 x float> %934) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %942 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %943 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %942, <4 x float> %936) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %944 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %940, <4 x float> %937) #9, !dbg !14, !frisk.pv_interleave !33
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %945 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed419, <4 x half> %942, <4 x float> %938) #9, !dbg !14, !frisk.pv_interleave !33
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
  %lane.sum.field188 = extractvalue [2 x <1 x float>] %103, 1, !dbg !6
  %lane.sum.local189 = extractelement <1 x float> %lane.sum.field188, i64 0, !dbg !6
  %963 = shl nuw nsw i32 %8, 2, !dbg !6
  %964 = xor i32 %963, 64, !dbg !6
  %965 = bitcast float %lane.sum.local189 to i32, !dbg !6
  %lane.sum.shuffle192 = call i32 @llvm.amdgcn.ds.bpermute(i32 %964, i32 %965), !dbg !6
  %966 = bitcast i32 %lane.sum.shuffle192 to float, !dbg !6
  %lane.sum.final193 = fadd float %lane.sum.local189, %966, !dbg !6
  %967 = xor i32 %963, 128, !dbg !6
  %968 = bitcast float %lane.sum.final193 to i32, !dbg !6
  %lane.sum.shuffle194 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %968), !dbg !6
  %969 = bitcast i32 %lane.sum.shuffle194 to float, !dbg !6
  %lane.sum.final195 = fadd float %lane.sum.final193, %969, !dbg !6
  %970 = insertelement <1 x float> poison, float %lane.sum.final195, i64 0, !dbg !6
  %lane.sum.field = extractvalue [2 x <1 x float>] %103, 0, !dbg !6
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !6
  %971 = bitcast float %lane.sum.local to i32, !dbg !6
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %964, i32 %971), !dbg !6
  %972 = bitcast i32 %lane.sum.shuffle to float, !dbg !6
  %lane.sum.final = fadd float %lane.sum.local, %972, !dbg !6
  %973 = bitcast float %lane.sum.final to i32, !dbg !6
  %lane.sum.shuffle186 = call i32 @llvm.amdgcn.ds.bpermute(i32 %967, i32 %973), !dbg !6
  %974 = bitcast i32 %lane.sum.shuffle186 to float, !dbg !6
  %lane.sum.final187 = fadd float %lane.sum.final, %974, !dbg !6
  %975 = insertelement <1 x float> poison, float %lane.sum.final187, i64 0, !dbg !6
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
  %1411 = shufflevector <1 x float> %970, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1412 = bitcast i32 %1406 to float, !dbg !6
  %1413 = select i1 %.not40, float %1396, float %1412, !dbg !6
  %1414 = insertelement <4 x float> poison, float %1413, i64 0, !dbg !17
  %1415 = bitcast i32 %1409 to float, !dbg !6
  %1416 = select i1 %.not40, float %1397, float %1415, !dbg !6
  %1417 = insertelement <4 x float> %1414, float %1416, i64 1, !dbg !17
  %1418 = select i1 %.not40, float %1412, float %1402, !dbg !6
  %1419 = insertelement <4 x float> %1417, float %1418, i64 2, !dbg !17
  %1420 = select i1 %.not40, float %1415, float %1403, !dbg !6
  %1421 = insertelement <4 x float> %1419, float %1420, i64 3, !dbg !17
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1411, !dbg !17
  %normalized = fmul <4 x float> %1421, %row.reciprocal, !dbg !17
  %1422 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !18
  %1423 = extractelement <4 x half> %1422, i64 3, !dbg !18
  %1424 = extractelement <4 x half> %1422, i64 2, !dbg !18
  %1425 = extractelement <4 x half> %1422, i64 1, !dbg !18
  %1426 = extractelement <4 x half> %1422, i64 0, !dbg !18
  %1427 = bitcast i32 %1384 to float, !dbg !6
  %1428 = select i1 %.not40, float %1374, float %1427, !dbg !6
  %1429 = insertelement <4 x float> poison, float %1428, i64 0, !dbg !17
  %1430 = bitcast i32 %1387 to float, !dbg !6
  %1431 = select i1 %.not40, float %1375, float %1430, !dbg !6
  %1432 = insertelement <4 x float> %1429, float %1431, i64 1, !dbg !17
  %1433 = select i1 %.not40, float %1427, float %1380, !dbg !6
  %1434 = insertelement <4 x float> %1432, float %1433, i64 2, !dbg !17
  %1435 = select i1 %.not40, float %1430, float %1381, !dbg !6
  %1436 = insertelement <4 x float> %1434, float %1435, i64 3, !dbg !17
  %normalized41 = fmul <4 x float> %1436, %row.reciprocal, !dbg !17
  %1437 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !18
  %1438 = extractelement <4 x half> %1437, i64 3, !dbg !18
  %1439 = extractelement <4 x half> %1437, i64 2, !dbg !18
  %1440 = extractelement <4 x half> %1437, i64 1, !dbg !18
  %1441 = extractelement <4 x half> %1437, i64 0, !dbg !18
  %1442 = bitcast i32 %1361 to float, !dbg !6
  %1443 = select i1 %.not40, float %1351, float %1442, !dbg !6
  %1444 = insertelement <4 x float> poison, float %1443, i64 0, !dbg !17
  %1445 = bitcast i32 %1364 to float, !dbg !6
  %1446 = select i1 %.not40, float %1352, float %1445, !dbg !6
  %1447 = insertelement <4 x float> %1444, float %1446, i64 1, !dbg !17
  %1448 = select i1 %.not40, float %1442, float %1357, !dbg !6
  %1449 = insertelement <4 x float> %1447, float %1448, i64 2, !dbg !17
  %1450 = select i1 %.not40, float %1445, float %1358, !dbg !6
  %1451 = insertelement <4 x float> %1449, float %1450, i64 3, !dbg !17
  %row.reciprocal42 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1410, !dbg !17
  %normalized43 = fmul <4 x float> %1451, %row.reciprocal42, !dbg !17
  %1452 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !18
  %1453 = extractelement <4 x half> %1452, i64 3, !dbg !18
  %1454 = extractelement <4 x half> %1452, i64 2, !dbg !18
  %1455 = extractelement <4 x half> %1452, i64 1, !dbg !18
  %1456 = extractelement <4 x half> %1452, i64 0, !dbg !18
  %1457 = bitcast i32 %1339 to float, !dbg !6
  %1458 = select i1 %.not40, float %1329, float %1457, !dbg !6
  %1459 = insertelement <4 x float> poison, float %1458, i64 0, !dbg !17
  %1460 = bitcast i32 %1342 to float, !dbg !6
  %1461 = select i1 %.not40, float %1330, float %1460, !dbg !6
  %1462 = insertelement <4 x float> %1459, float %1461, i64 1, !dbg !17
  %1463 = select i1 %.not40, float %1457, float %1335, !dbg !6
  %1464 = insertelement <4 x float> %1462, float %1463, i64 2, !dbg !17
  %1465 = select i1 %.not40, float %1460, float %1336, !dbg !6
  %1466 = insertelement <4 x float> %1464, float %1465, i64 3, !dbg !17
  %normalized44 = fmul <4 x float> %1466, %row.reciprocal42, !dbg !17
  %1467 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !18
  %1468 = extractelement <4 x half> %1467, i64 3, !dbg !18
  %1469 = extractelement <4 x half> %1467, i64 2, !dbg !18
  %1470 = extractelement <4 x half> %1467, i64 1, !dbg !18
  %1471 = extractelement <4 x half> %1467, i64 0, !dbg !18
  %address.thread993 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local994 = zext nneg i32 %address.thread993 to i64, !dbg !9
  %address.local995 = add nuw i64 %32, %address.local994, !dbg !19
  %1472 = lshr i64 %address.local994, 2, !dbg !19
  %address.local997 = and i64 %1472, 48, !dbg !19
  %address.local998 = add i64 %address.local995, %address.local997, !dbg !19
  %address.local999 = and i64 %address.local994, 240, !dbg !19
  %address.local1000 = sub i64 %address.local998, %address.local999, !dbg !19
  %address.local1001 = add i64 %address.local1000, %33, !dbg !19
  %address.local1002 = shl nuw nsw i64 %address.local1001, 7, !dbg !19
  %1473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1473, i64 %address.local1002, !dbg !19
  %address.local1006 = and i64 %1472, 12, !dbg !12
  %address.local1007 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1006, !dbg !19
  store half %1471, ptr addrspace(1) %address.local1007, align 2, !dbg !19
  %address.local1009 = or disjoint i64 %address.local1006, 1, !dbg !19
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1009, !dbg !19
  store half %1470, ptr addrspace(1) %address.local1010, align 2, !dbg !19
  %address.local1012 = or disjoint i64 %address.local1006, 2, !dbg !19
  %address.local1013 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1012, !dbg !19
  store half %1469, ptr addrspace(1) %address.local1013, align 2, !dbg !19
  %address.local1015 = or disjoint i64 %address.local1006, 3, !dbg !19
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1015, !dbg !19
  store half %1468, ptr addrspace(1) %address.local1016, align 2, !dbg !19
  %address.local1018 = or disjoint i64 %address.local1006, 16, !dbg !19
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1018, !dbg !19
  store half %1456, ptr addrspace(1) %address.local1019, align 2, !dbg !19
  %address.local1021 = or disjoint i64 %address.local1006, 17, !dbg !19
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1021, !dbg !19
  store half %1455, ptr addrspace(1) %address.local1022, align 2, !dbg !19
  %address.local1024 = or disjoint i64 %address.local1006, 18, !dbg !19
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1024, !dbg !19
  store half %1454, ptr addrspace(1) %address.local1025, align 2, !dbg !19
  %address.local1027 = or disjoint i64 %address.local1006, 19, !dbg !19
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1027, !dbg !19
  store half %1453, ptr addrspace(1) %address.local1028, align 2, !dbg !19
  %address.local1032 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1004, i64 16384, !dbg !19
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1006, !dbg !19
  store half %1441, ptr addrspace(1) %address.local1033, align 2, !dbg !19
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1009, !dbg !19
  store half %1440, ptr addrspace(1) %address.local1035, align 2, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1012, !dbg !19
  store half %1439, ptr addrspace(1) %address.local1037, align 2, !dbg !19
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1015, !dbg !19
  store half %1438, ptr addrspace(1) %address.local1039, align 2, !dbg !19
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1018, !dbg !19
  store half %1426, ptr addrspace(1) %address.local1041, align 2, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1021, !dbg !19
  store half %1425, ptr addrspace(1) %address.local1043, align 2, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1024, !dbg !19
  store half %1424, ptr addrspace(1) %address.local1045, align 2, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1027, !dbg !19
  store half %1423, ptr addrspace(1) %address.local1047, align 2, !dbg !19
  %1474 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !17
  %1475 = insertelement <4 x float> %1474, float %1318, i64 1, !dbg !17
  %1476 = insertelement <4 x float> %1475, float %1313, i64 2, !dbg !17
  %1477 = insertelement <4 x float> %1476, float %1319, i64 3, !dbg !17
  %normalized46 = fmul <4 x float> %1477, %row.reciprocal, !dbg !17
  %1478 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !18
  %1479 = extractelement <4 x half> %1478, i64 3, !dbg !18
  %1480 = extractelement <4 x half> %1478, i64 2, !dbg !18
  %1481 = extractelement <4 x half> %1478, i64 1, !dbg !18
  %1482 = extractelement <4 x half> %1478, i64 0, !dbg !18
  %1483 = insertelement <4 x float> poison, float %1284, i64 0, !dbg !17
  %1484 = insertelement <4 x float> %1483, float %1290, i64 1, !dbg !17
  %1485 = insertelement <4 x float> %1484, float %1285, i64 2, !dbg !17
  %1486 = insertelement <4 x float> %1485, float %1291, i64 3, !dbg !17
  %normalized47 = fmul <4 x float> %1486, %row.reciprocal, !dbg !17
  %1487 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !18
  %1488 = extractelement <4 x half> %1487, i64 3, !dbg !18
  %1489 = extractelement <4 x half> %1487, i64 2, !dbg !18
  %1490 = extractelement <4 x half> %1487, i64 1, !dbg !18
  %1491 = extractelement <4 x half> %1487, i64 0, !dbg !18
  %1492 = insertelement <4 x float> poison, float %1255, i64 0, !dbg !17
  %1493 = insertelement <4 x float> %1492, float %1261, i64 1, !dbg !17
  %1494 = insertelement <4 x float> %1493, float %1256, i64 2, !dbg !17
  %1495 = insertelement <4 x float> %1494, float %1262, i64 3, !dbg !17
  %normalized49 = fmul <4 x float> %1495, %row.reciprocal42, !dbg !17
  %1496 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !18
  %1497 = extractelement <4 x half> %1496, i64 3, !dbg !18
  %1498 = extractelement <4 x half> %1496, i64 2, !dbg !18
  %1499 = extractelement <4 x half> %1496, i64 1, !dbg !18
  %1500 = extractelement <4 x half> %1496, i64 0, !dbg !18
  %1501 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !17
  %1502 = insertelement <4 x float> %1501, float %1233, i64 1, !dbg !17
  %1503 = insertelement <4 x float> %1502, float %1228, i64 2, !dbg !17
  %1504 = insertelement <4 x float> %1503, float %1234, i64 3, !dbg !17
  %normalized50 = fmul <4 x float> %1504, %row.reciprocal42, !dbg !17
  %1505 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !18
  %1506 = extractelement <4 x half> %1505, i64 3, !dbg !18
  %1507 = extractelement <4 x half> %1505, i64 2, !dbg !18
  %1508 = extractelement <4 x half> %1505, i64 1, !dbg !18
  %1509 = extractelement <4 x half> %1505, i64 0, !dbg !18
  %address.local1062 = or disjoint i64 %address.local1006, 32, !dbg !19
  %address.local1063 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1062, !dbg !19
  store half %1509, ptr addrspace(1) %address.local1063, align 2, !dbg !19
  %address.local1065 = or disjoint i64 %address.local1006, 33, !dbg !19
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1065, !dbg !19
  store half %1508, ptr addrspace(1) %address.local1066, align 2, !dbg !19
  %address.local1068 = or disjoint i64 %address.local1006, 34, !dbg !19
  %address.local1069 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1068, !dbg !19
  store half %1507, ptr addrspace(1) %address.local1069, align 2, !dbg !19
  %address.local1071 = or disjoint i64 %address.local1006, 35, !dbg !19
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1071, !dbg !19
  store half %1506, ptr addrspace(1) %address.local1072, align 2, !dbg !19
  %address.local1074 = or disjoint i64 %address.local1006, 48, !dbg !19
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1074, !dbg !19
  store half %1500, ptr addrspace(1) %address.local1075, align 2, !dbg !19
  %address.local1077 = or disjoint i64 %address.local1006, 49, !dbg !19
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1077, !dbg !19
  store half %1499, ptr addrspace(1) %address.local1078, align 2, !dbg !19
  %address.local1080 = or disjoint i64 %address.local1006, 50, !dbg !19
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1080, !dbg !19
  store half %1498, ptr addrspace(1) %address.local1081, align 2, !dbg !19
  %address.local1083 = or i64 %1472, 51, !dbg !19
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1083, !dbg !19
  store half %1497, ptr addrspace(1) %address.local1084, align 2, !dbg !19
  %address.local1089 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1062, !dbg !19
  store half %1491, ptr addrspace(1) %address.local1089, align 2, !dbg !19
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1065, !dbg !19
  store half %1490, ptr addrspace(1) %address.local1091, align 2, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1068, !dbg !19
  store half %1489, ptr addrspace(1) %address.local1093, align 2, !dbg !19
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1071, !dbg !19
  store half %1488, ptr addrspace(1) %address.local1095, align 2, !dbg !19
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1074, !dbg !19
  store half %1482, ptr addrspace(1) %address.local1097, align 2, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1077, !dbg !19
  store half %1481, ptr addrspace(1) %address.local1099, align 2, !dbg !19
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1080, !dbg !19
  store half %1480, ptr addrspace(1) %address.local1101, align 2, !dbg !19
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1083, !dbg !19
  store half %1479, ptr addrspace(1) %address.local1103, align 2, !dbg !19
  %1510 = insertelement <4 x float> poison, float %1198, i64 0, !dbg !17
  %1511 = insertelement <4 x float> %1510, float %1204, i64 1, !dbg !17
  %1512 = insertelement <4 x float> %1511, float %1199, i64 2, !dbg !17
  %1513 = insertelement <4 x float> %1512, float %1205, i64 3, !dbg !17
  %normalized52 = fmul <4 x float> %1513, %row.reciprocal, !dbg !17
  %1514 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !18
  %1515 = extractelement <4 x half> %1514, i64 3, !dbg !18
  %1516 = extractelement <4 x half> %1514, i64 2, !dbg !18
  %1517 = extractelement <4 x half> %1514, i64 1, !dbg !18
  %1518 = extractelement <4 x half> %1514, i64 0, !dbg !18
  %1519 = insertelement <4 x float> poison, float %1170, i64 0, !dbg !17
  %1520 = insertelement <4 x float> %1519, float %1176, i64 1, !dbg !17
  %1521 = insertelement <4 x float> %1520, float %1171, i64 2, !dbg !17
  %1522 = insertelement <4 x float> %1521, float %1177, i64 3, !dbg !17
  %normalized53 = fmul <4 x float> %1522, %row.reciprocal, !dbg !17
  %1523 = fptrunc <4 x float> %normalized53 to <4 x half>, !dbg !18
  %1524 = extractelement <4 x half> %1523, i64 3, !dbg !18
  %1525 = extractelement <4 x half> %1523, i64 2, !dbg !18
  %1526 = extractelement <4 x half> %1523, i64 1, !dbg !18
  %1527 = extractelement <4 x half> %1523, i64 0, !dbg !18
  %1528 = insertelement <4 x float> poison, float %1141, i64 0, !dbg !17
  %1529 = insertelement <4 x float> %1528, float %1147, i64 1, !dbg !17
  %1530 = insertelement <4 x float> %1529, float %1142, i64 2, !dbg !17
  %1531 = insertelement <4 x float> %1530, float %1148, i64 3, !dbg !17
  %normalized55 = fmul <4 x float> %1531, %row.reciprocal42, !dbg !17
  %1532 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !18
  %1533 = extractelement <4 x half> %1532, i64 3, !dbg !18
  %1534 = extractelement <4 x half> %1532, i64 2, !dbg !18
  %1535 = extractelement <4 x half> %1532, i64 1, !dbg !18
  %1536 = extractelement <4 x half> %1532, i64 0, !dbg !18
  %1537 = insertelement <4 x float> poison, float %1113, i64 0, !dbg !17
  %1538 = insertelement <4 x float> %1537, float %1119, i64 1, !dbg !17
  %1539 = insertelement <4 x float> %1538, float %1114, i64 2, !dbg !17
  %1540 = insertelement <4 x float> %1539, float %1120, i64 3, !dbg !17
  %normalized56 = fmul <4 x float> %1540, %row.reciprocal42, !dbg !17
  %1541 = fptrunc <4 x float> %normalized56 to <4 x half>, !dbg !18
  %1542 = extractelement <4 x half> %1541, i64 3, !dbg !18
  %1543 = extractelement <4 x half> %1541, i64 2, !dbg !18
  %1544 = extractelement <4 x half> %1541, i64 1, !dbg !18
  %1545 = extractelement <4 x half> %1541, i64 0, !dbg !18
  %address.local1118 = or disjoint i64 %address.local1006, 64, !dbg !19
  %address.local1119 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1118, !dbg !19
  store half %1545, ptr addrspace(1) %address.local1119, align 2, !dbg !19
  %address.local1121 = or disjoint i64 %address.local1006, 65, !dbg !19
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1121, !dbg !19
  store half %1544, ptr addrspace(1) %address.local1122, align 2, !dbg !19
  %address.local1124 = or disjoint i64 %address.local1006, 66, !dbg !19
  %address.local1125 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1124, !dbg !19
  store half %1543, ptr addrspace(1) %address.local1125, align 2, !dbg !19
  %address.local1127 = or disjoint i64 %address.local1006, 67, !dbg !19
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1127, !dbg !19
  store half %1542, ptr addrspace(1) %address.local1128, align 2, !dbg !19
  %address.local1130 = or disjoint i64 %address.local1006, 80, !dbg !19
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1130, !dbg !19
  store half %1536, ptr addrspace(1) %address.local1131, align 2, !dbg !19
  %address.local1133 = or disjoint i64 %address.local1006, 81, !dbg !19
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1133, !dbg !19
  store half %1535, ptr addrspace(1) %address.local1134, align 2, !dbg !19
  %address.local1136 = or disjoint i64 %address.local1006, 82, !dbg !19
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1136, !dbg !19
  store half %1534, ptr addrspace(1) %address.local1137, align 2, !dbg !19
  %address.local1139 = or disjoint i64 %address.local1006, 83, !dbg !19
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1139, !dbg !19
  store half %1533, ptr addrspace(1) %address.local1140, align 2, !dbg !19
  %address.local1145 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1118, !dbg !19
  store half %1527, ptr addrspace(1) %address.local1145, align 2, !dbg !19
  %address.local1147 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1121, !dbg !19
  store half %1526, ptr addrspace(1) %address.local1147, align 2, !dbg !19
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1124, !dbg !19
  store half %1525, ptr addrspace(1) %address.local1149, align 2, !dbg !19
  %address.local1151 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1127, !dbg !19
  store half %1524, ptr addrspace(1) %address.local1151, align 2, !dbg !19
  %address.local1153 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1130, !dbg !19
  store half %1518, ptr addrspace(1) %address.local1153, align 2, !dbg !19
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1133, !dbg !19
  store half %1517, ptr addrspace(1) %address.local1155, align 2, !dbg !19
  %address.local1157 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1136, !dbg !19
  store half %1516, ptr addrspace(1) %address.local1157, align 2, !dbg !19
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1139, !dbg !19
  store half %1515, ptr addrspace(1) %address.local1159, align 2, !dbg !19
  %1546 = insertelement <4 x float> poison, float %1084, i64 0, !dbg !17
  %1547 = insertelement <4 x float> %1546, float %1090, i64 1, !dbg !17
  %1548 = insertelement <4 x float> %1547, float %1085, i64 2, !dbg !17
  %1549 = insertelement <4 x float> %1548, float %1091, i64 3, !dbg !17
  %normalized58 = fmul <4 x float> %1549, %row.reciprocal, !dbg !17
  %1550 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !18
  %1551 = extractelement <4 x half> %1550, i64 3, !dbg !18
  %1552 = extractelement <4 x half> %1550, i64 2, !dbg !18
  %1553 = extractelement <4 x half> %1550, i64 1, !dbg !18
  %1554 = extractelement <4 x half> %1550, i64 0, !dbg !18
  %1555 = insertelement <4 x float> poison, float %1056, i64 0, !dbg !17
  %1556 = insertelement <4 x float> %1555, float %1062, i64 1, !dbg !17
  %1557 = insertelement <4 x float> %1556, float %1057, i64 2, !dbg !17
  %1558 = insertelement <4 x float> %1557, float %1063, i64 3, !dbg !17
  %normalized59 = fmul <4 x float> %1558, %row.reciprocal, !dbg !17
  %1559 = fptrunc <4 x float> %normalized59 to <4 x half>, !dbg !18
  %1560 = extractelement <4 x half> %1559, i64 3, !dbg !18
  %1561 = extractelement <4 x half> %1559, i64 2, !dbg !18
  %1562 = extractelement <4 x half> %1559, i64 1, !dbg !18
  %1563 = extractelement <4 x half> %1559, i64 0, !dbg !18
  %1564 = insertelement <4 x float> poison, float %1027, i64 0, !dbg !17
  %1565 = insertelement <4 x float> %1564, float %1033, i64 1, !dbg !17
  %1566 = insertelement <4 x float> %1565, float %1028, i64 2, !dbg !17
  %1567 = insertelement <4 x float> %1566, float %1034, i64 3, !dbg !17
  %normalized61 = fmul <4 x float> %1567, %row.reciprocal42, !dbg !17
  %1568 = fptrunc <4 x float> %normalized61 to <4 x half>, !dbg !18
  %1569 = extractelement <4 x half> %1568, i64 3, !dbg !18
  %1570 = extractelement <4 x half> %1568, i64 2, !dbg !18
  %1571 = extractelement <4 x half> %1568, i64 1, !dbg !18
  %1572 = extractelement <4 x half> %1568, i64 0, !dbg !18
  %1573 = insertelement <4 x float> poison, float %999, i64 0, !dbg !17
  %1574 = insertelement <4 x float> %1573, float %1005, i64 1, !dbg !17
  %1575 = insertelement <4 x float> %1574, float %1000, i64 2, !dbg !17
  %1576 = insertelement <4 x float> %1575, float %1006, i64 3, !dbg !17
  %normalized62 = fmul <4 x float> %1576, %row.reciprocal42, !dbg !17
  %1577 = fptrunc <4 x float> %normalized62 to <4 x half>, !dbg !18
  %1578 = extractelement <4 x half> %1577, i64 3, !dbg !18
  %1579 = extractelement <4 x half> %1577, i64 2, !dbg !18
  %1580 = extractelement <4 x half> %1577, i64 1, !dbg !18
  %1581 = extractelement <4 x half> %1577, i64 0, !dbg !18
  %address.local1174 = or disjoint i64 %address.local1006, 96, !dbg !19
  %address.local1175 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1174, !dbg !19
  store half %1581, ptr addrspace(1) %address.local1175, align 2, !dbg !19
  %address.local1177 = or disjoint i64 %address.local1006, 97, !dbg !19
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1177, !dbg !19
  store half %1580, ptr addrspace(1) %address.local1178, align 2, !dbg !19
  %address.local1180 = or disjoint i64 %address.local1006, 98, !dbg !19
  %address.local1181 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1180, !dbg !19
  store half %1579, ptr addrspace(1) %address.local1181, align 2, !dbg !19
  %address.local1183 = or disjoint i64 %address.local1006, 99, !dbg !19
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1183, !dbg !19
  store half %1578, ptr addrspace(1) %address.local1184, align 2, !dbg !19
  %address.local1186 = or disjoint i64 %address.local1006, 112, !dbg !19
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1186, !dbg !19
  store half %1572, ptr addrspace(1) %address.local1187, align 2, !dbg !19
  %address.local1189 = or disjoint i64 %address.local1006, 113, !dbg !19
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1189, !dbg !19
  store half %1571, ptr addrspace(1) %address.local1190, align 2, !dbg !19
  %address.local1192 = or disjoint i64 %address.local1006, 114, !dbg !19
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1192, !dbg !19
  store half %1570, ptr addrspace(1) %address.local1193, align 2, !dbg !19
  %address.local1195 = or i64 %1472, 115, !dbg !19
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1004, i64 %address.local1195, !dbg !19
  store half %1569, ptr addrspace(1) %address.local1196, align 2, !dbg !19
  %address.local1201 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1174, !dbg !19
  store half %1563, ptr addrspace(1) %address.local1201, align 2, !dbg !19
  %address.local1203 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1177, !dbg !19
  store half %1562, ptr addrspace(1) %address.local1203, align 2, !dbg !19
  %address.local1205 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1180, !dbg !19
  store half %1561, ptr addrspace(1) %address.local1205, align 2, !dbg !19
  %address.local1207 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1183, !dbg !19
  store half %1560, ptr addrspace(1) %address.local1207, align 2, !dbg !19
  %address.local1209 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1186, !dbg !19
  store half %1554, ptr addrspace(1) %address.local1209, align 2, !dbg !19
  %address.local1211 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1189, !dbg !19
  store half %1553, ptr addrspace(1) %address.local1211, align 2, !dbg !19
  %address.local1213 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1192, !dbg !19
  store half %1552, ptr addrspace(1) %address.local1213, align 2, !dbg !19
  %address.local1215 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1032, i64 %address.local1195, !dbg !19
  store half %1551, ptr addrspace(1) %address.local1215, align 2, !dbg !19
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
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_early_wave")
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
!14 = !DILocation(line: 87, column: 23, scope: !4)
!15 = !DILocation(line: 70, column: 26, scope: !4)
!16 = !DILocation(line: 86, column: 20, scope: !4)
!17 = !DILocation(line: 92, column: 23, scope: !4)
!18 = !DILocation(line: 93, column: 19, scope: !4)
!19 = !DILocation(line: 94, column: 9, scope: !4)
!20 = !DILocation(line: 71, column: 23, scope: !4)
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
