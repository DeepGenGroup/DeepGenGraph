; ModuleID = 'generated_ir/direct_native_exp2/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2/input.mlir"

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
  %address.local420 = zext nneg i32 %37 to i64, !dbg !9
  %address.local421 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local420, !dbg !9
  %address.local424 = lshr i32 %address.thread, 4, !dbg !9
  %address.local425 = and i32 %address.local424, 8, !dbg !9
  %address.local426 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local427 = and i32 %address.local426, 16256, !dbg !9
  %address.local428 = or disjoint i32 %address.local425, %address.local427, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1394 = shl i64 %34, 8, !dbg !9
  %39 = add i64 %.idx1394, %.idx, !dbg !9
  %40 = shl nuw nsw i32 %address.local428, 1, !dbg !9
  %41 = trunc i64 %39 to i32, !dbg !9
  %dma.buffer.offset1392 = or disjoint i32 %40, %41, !dbg !9
  %42 = insertelement <2 x i64> poison, i64 %38, i64 0, !dbg !9
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1393 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local433 = or disjoint i64 %address.local420, 2048, !dbg !9
  %address.local434 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local433, !dbg !9
  %address.local442 = or disjoint i32 %address.local428, 16, !dbg !9
  %44 = shl nuw nsw i32 %address.local442, 1, !dbg !9
  %dma.buffer.offset1390 = or disjoint i32 %44, %41, !dbg !9
  %45 = shl nuw nsw i64 %31, 19, !dbg !9
  %46 = add i64 %35, %45, !dbg !9
  %address.local445 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local420, !dbg !9
  %47 = trunc i64 %46 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local428, %47, !dbg !9
  %48 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1388 = or disjoint i32 %48, 64, !dbg !9
  %address.local449 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local433, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local421, i64 0, !dbg !9
  %50 = insertelement <4 x ptr addrspace(3)> %49, ptr addrspace(3) %address.local434, i64 1, !dbg !9
  %51 = insertelement <4 x ptr addrspace(3)> %50, ptr addrspace(3) %address.local445, i64 2, !dbg !9
  %52 = insertelement <4 x ptr addrspace(3)> %51, ptr addrspace(3) %address.local449, i64 3, !dbg !9
  %53 = ptrtoint <4 x ptr addrspace(3)> %52 to <4 x i64>, !dbg !9
  %54 = trunc <4 x i64> %53 to <4 x i32>, !dbg !9
  %55 = extractelement <4 x i32> %54, i64 0, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1392, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %58 = extractelement <4 x i32> %54, i64 1, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %61 = extractelement <4 x i32> %54, i64 2, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1388, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %64 = extractelement <4 x i32> %54, i64 3, !dbg !9
  %65 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !9
  %.tr1398 = or disjoint i32 %address.local442, %47, !dbg !9
  %66 = shl i32 %.tr1398, 1, !dbg !9
  %dma.buffer.offset1386 = or disjoint i32 %66, 64, !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1386, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %address.local453 = lshr i32 %address.thread, 2, !dbg !11
  %address.local454 = and i32 %address.local453, 4, !dbg !11
  %address.local455 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local456 = and i32 %address.local455, 384, !dbg !11
  %address.local457 = or disjoint i32 %address.local454, %address.local456, !dbg !11
  %address.local459 = and i32 %36, 120, !dbg !11
  %address.local460 = or disjoint i32 %address.local457, %address.local459, !dbg !11
  %address.local461 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local462 = and i32 %address.local461, 1024, !dbg !11
  %address.local463 = or disjoint i32 %address.local460, %address.local462, !dbg !11
  %address.local464 = zext nneg i32 %address.local463 to i64, !dbg !11
  %address.local465 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local464, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local465, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local465, i64 4096, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1215 = getelementptr i8, ptr addrspace(3) %address.local465, i64 1024, !dbg !11
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !11
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1216 = getelementptr i8, ptr addrspace(3) %address.local465, i64 5120, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !11
  %75 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1384 = or disjoint i32 %48, 128, !dbg !9
  %76 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %dma.buffer.offset1382 = or disjoint i32 %66, 128, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1382, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %lds.const.gep1217 = getelementptr i8, ptr addrspace(3) %address.local465, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !11
  %lds.const.gep1218 = getelementptr i8, ptr addrspace(3) %address.local465, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !11
  %lds.const.gep1219 = getelementptr i8, ptr addrspace(3) %address.local465, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1220 = getelementptr i8, ptr addrspace(3) %address.local465, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1380 = or disjoint i32 %48, 192, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1380, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %dma.buffer.offset1378 = or disjoint i32 %66, 192, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1393) #9, !dbg !9
  %88 = load <4 x half>, ptr addrspace(3) %address.local465, align 8, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1215, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1216, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1217, align 8, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1218, align 8, !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1219, align 8, !dbg !11
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1220, align 8, !dbg !11
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
  %k0.prime.2061 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1500 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1501 = shl nuw nsw i64 %k0.prime.kdma.source1500, 9, !dbg !14
  %k0.prime.kdma.source1502 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1503 = add nuw nsw i64 %k0.prime.kdma.source1501, %k0.prime.kdma.source1502, !dbg !14
  %k0.prime.kdma.source1504 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1505 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1506 = and i64 %k0.prime.kdma.source1505, 1, !dbg !14
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, %k0.prime.kdma.source1500, !dbg !14
  %k0.prime.kdma.source1508 = or disjoint i64 %k0.prime.kdma.source1507, %k0.prime.kdma.source1504, !dbg !14
  %110 = shl nuw nsw i64 %k0.prime.kdma.source1508, 10, !dbg !14
  %k0.prime.kdma.source1510 = sub nsw i64 %k0.prime.kdma.source1503, %110, !dbg !14
  %k0.prime.kdma.source1511 = and i64 %k0.prime.kdma.source1510, 16256, !dbg !14
  %k0.prime.kdma.source1512 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1513 = and i32 %k0.prime.kdma.source1512, 8, !dbg !14
  %k0.prime.kdma.source1514 = zext nneg i32 %k0.prime.kdma.source1513 to i64, !dbg !14
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1511, %k0.prime.kdma.source1514, !dbg !14
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.kdma.source1515, %k0.prime.2061, !dbg !14
  %k0.prime.kdma.source1517 = trunc nuw nsw i64 %k0.prime.kdma.source1516 to i32, !dbg !14
  %k0.prime.kdma.source1518 = shl nuw nsw i32 %k0.prime.kdma.source1517, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1518, %kdma.intra, !dbg !14
  %k0.prime.2064 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.2065 = insertelement <2 x i64> poison, i64 %k0.prime.2064, i64 0, !dbg !14
  %k0.prime.2066 = bitcast <2 x i64> %k0.prime.2065 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1377 = shufflevector <4 x i32> %k0.prime.2066, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.2067 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1540 = add i32 %56, 512, !dbg !14
  %k0.prime.kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1520 = lshr i64 %k0.prime.kdma.source1519, 4, !dbg !14
  %k0.prime.kdma.source1521 = shl nuw nsw i64 %k0.prime.kdma.source1520, 9, !dbg !14
  %k0.prime.kdma.source1522 = shl nuw nsw i64 %k0.prime.kdma.source1519, 7, !dbg !14
  %k0.prime.kdma.source1523 = add nuw nsw i64 %k0.prime.kdma.source1521, %k0.prime.kdma.source1522, !dbg !14
  %k0.prime.kdma.source1524 = lshr i64 %k0.prime.kdma.source1519, 5, !dbg !14
  %k0.prime.kdma.source1525 = lshr i64 %k0.prime.kdma.source1519, 2, !dbg !14
  %k0.prime.kdma.source1526 = and i64 %k0.prime.kdma.source1525, 1, !dbg !14
  %k0.prime.kdma.source1527 = and i64 %k0.prime.kdma.source1526, %k0.prime.kdma.source1520, !dbg !14
  %k0.prime.kdma.source1528 = add nuw nsw i64 %k0.prime.kdma.source1527, %k0.prime.kdma.source1524, !dbg !14
  %111 = shl nuw nsw i64 %k0.prime.kdma.source1528, 10, !dbg !14
  %k0.prime.kdma.source1530 = sub nsw i64 %k0.prime.kdma.source1523, %111, !dbg !14
  %k0.prime.kdma.source1531 = and i64 %k0.prime.kdma.source1530, 16256, !dbg !14
  %k0.prime.kdma.source1535 = or disjoint i64 %k0.prime.kdma.source1531, %k0.prime.kdma.source1514, !dbg !14
  %k0.prime.kdma.source1536 = or disjoint i64 %k0.prime.kdma.source1535, %k0.prime.2061, !dbg !14
  %k0.prime.kdma.source1537 = trunc nuw nsw i64 %k0.prime.kdma.source1536 to i32, !dbg !14
  %k0.prime.kdma.source1538 = shl nuw nsw i32 %k0.prime.kdma.source1537, 1, !dbg !14
  %k0.prime.kdma.offset1539 = or disjoint i32 %k0.prime.kdma.source1538, %kdma.intra.hi, !dbg !14
  %k0.prime.2068 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1540, i32 %k0.prime.kdma.offset1539, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1542 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1543 = lshr i64 %k0.prime.kdma.source1542, 4, !dbg !14
  %k0.prime.kdma.source1544 = shl nuw nsw i64 %k0.prime.kdma.source1543, 9, !dbg !14
  %k0.prime.kdma.source1546 = add nuw nsw i64 %k0.prime.kdma.source1544, %k0.prime.kdma.source1502, !dbg !14
  %k0.prime.kdma.source1547 = lshr i64 %k0.prime.kdma.source1542, 5, !dbg !14
  %k0.prime.kdma.source1550 = and i64 %k0.prime.kdma.source1543, %k0.prime.kdma.source1506, !dbg !14
  %k0.prime.kdma.source1551 = or disjoint i64 %k0.prime.kdma.source1550, %k0.prime.kdma.source1547, !dbg !14
  %112 = shl nuw nsw i64 %k0.prime.kdma.source1551, 10, !dbg !14
  %k0.prime.kdma.source1553 = sub nsw i64 %k0.prime.kdma.source1546, %112, !dbg !14
  %k0.prime.kdma.source1554 = and i64 %k0.prime.kdma.source1553, 16256, !dbg !14
  %k0.prime.kdma.source1557 = or disjoint i32 %k0.prime.kdma.source1513, 16, !dbg !14
  %k0.prime.kdma.source1558 = zext nneg i32 %k0.prime.kdma.source1557 to i64, !dbg !14
  %k0.prime.kdma.source1559 = or disjoint i64 %k0.prime.kdma.source1554, %k0.prime.kdma.source1558, !dbg !14
  %k0.prime.kdma.source1560 = or disjoint i64 %k0.prime.kdma.source1559, %k0.prime.2061, !dbg !14
  %k0.prime.kdma.source1561 = trunc nuw nsw i64 %k0.prime.kdma.source1560 to i32, !dbg !14
  %k0.prime.kdma.source1562 = shl nuw nsw i32 %k0.prime.kdma.source1561, 1, !dbg !14
  %k0.prime.kdma.offset1563 = add nuw nsw i32 %k0.prime.kdma.source1562, %kdma.intra, !dbg !14
  %k0.prime.2069 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.prime.kdma.offset1563, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1588 = add i32 %59, 512, !dbg !14
  %k0.prime.kdma.source1566 = or disjoint i64 %k0.prime.kdma.source1519, 256, !dbg !14
  %k0.prime.kdma.source1567 = lshr i64 %k0.prime.kdma.source1566, 4, !dbg !14
  %k0.prime.kdma.source1568 = shl nuw nsw i64 %k0.prime.kdma.source1567, 9, !dbg !14
  %k0.prime.kdma.source1570 = add nuw nsw i64 %k0.prime.kdma.source1568, %k0.prime.kdma.source1522, !dbg !14
  %k0.prime.kdma.source1571 = lshr i64 %k0.prime.kdma.source1566, 5, !dbg !14
  %k0.prime.kdma.source1574 = and i64 %k0.prime.kdma.source1567, %k0.prime.kdma.source1526, !dbg !14
  %k0.prime.kdma.source1575 = add nuw nsw i64 %k0.prime.kdma.source1574, %k0.prime.kdma.source1571, !dbg !14
  %113 = shl nuw nsw i64 %k0.prime.kdma.source1575, 10, !dbg !14
  %k0.prime.kdma.source1577 = sub nsw i64 %k0.prime.kdma.source1570, %113, !dbg !14
  %k0.prime.kdma.source1578 = and i64 %k0.prime.kdma.source1577, 16256, !dbg !14
  %k0.prime.kdma.source1583 = or disjoint i64 %k0.prime.kdma.source1578, %k0.prime.kdma.source1558, !dbg !14
  %k0.prime.kdma.source1584 = or disjoint i64 %k0.prime.kdma.source1583, %k0.prime.2061, !dbg !14
  %k0.prime.kdma.source1585 = trunc nuw nsw i64 %k0.prime.kdma.source1584 to i32, !dbg !14
  %k0.prime.kdma.source1586 = shl nuw nsw i32 %k0.prime.kdma.source1585, 1, !dbg !14
  %k0.prime.kdma.offset1587 = add nuw nsw i32 %k0.prime.kdma.source1586, %kdma.intra.hi, !dbg !14
  %k0.prime.2070 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1588, i32 %k0.prime.kdma.offset1587, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1632 = add i32 %62, 512
  %kdma.lds1680 = add i32 %65, 512
  %114 = ptrtoint ptr addrspace(1) %2 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1361 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %56, 8192
  %vdb.dma.base1454 = add i32 %59, 8192
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
  %126 = add nuw nsw i64 %125, %k0.prime.2061, !dbg !14
  %address.thread516 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local517 = zext nneg i32 %address.thread516 to i64, !dbg !9
  %address.local523 = lshr i64 %address.local517, 2, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %127 = or disjoint i64 %126, 32, !dbg !14
  %kdma.source1606 = or disjoint i64 %127, %k0.prime.kdma.source1515, !dbg !14
  %kdma.source1607 = trunc nuw nsw i64 %kdma.source1606 to i32, !dbg !14
  %kdma.source1608 = shl nuw nsw i32 %kdma.source1607, 1, !dbg !14
  %kdma.offset1609 = or disjoint i32 %kdma.source1608, %kdma.intra, !dbg !14
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1609, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1628 = or disjoint i64 %127, %k0.prime.kdma.source1535, !dbg !14
  %kdma.source1629 = trunc nuw nsw i64 %kdma.source1628 to i32, !dbg !14
  %kdma.source1630 = shl nuw nsw i32 %kdma.source1629, 1, !dbg !14
  %kdma.offset1631 = or disjoint i32 %kdma.source1630, %kdma.intra.hi, !dbg !14
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1632, i32 %kdma.offset1631, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1652 = or disjoint i64 %127, %k0.prime.kdma.source1559, !dbg !14
  %kdma.source1653 = trunc nuw nsw i64 %kdma.source1652 to i32, !dbg !14
  %kdma.source1654 = shl nuw nsw i32 %kdma.source1653, 1, !dbg !14
  %kdma.offset1655 = add nuw nsw i32 %kdma.source1654, %kdma.intra, !dbg !14
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1655, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1676 = or disjoint i64 %127, %k0.prime.kdma.source1583, !dbg !14
  %kdma.source1677 = trunc nuw nsw i64 %kdma.source1676 to i32, !dbg !14
  %kdma.source1678 = shl nuw nsw i32 %kdma.source1677, 1, !dbg !14
  %kdma.offset1679 = add nuw nsw i32 %kdma.source1678, %kdma.intra.hi, !dbg !14
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1680, i32 %kdma.offset1679, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %address.local560 = shl nuw nsw i64 %address.local517, 1, !dbg !13
  %address.local561 = and i64 %address.local560, 24, !dbg !13
  %address.local563 = shl i64 %address.local517, 62, !dbg !13
  %address.local564 = ashr exact i64 %address.local563, 57, !dbg !13
  %address.local565 = and i64 %address.local564, -64, !dbg !13
  %address.local567 = and i64 %address.local517, 63, !dbg !11
  %address.local569 = and i64 %address.local523, 12, !dbg !11
  %address.local571 = lshr i64 %address.local567, 5, !dbg !11
  %address.local572 = mul nuw nsw i64 %address.local571, 1016, !dbg !11
  %.idx1415 = mul nuw nsw i64 %address.local517, 320, !dbg !13
  %132 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1415, !dbg !13
  %.idx1416 = mul nsw i64 %address.local523, -1280, !dbg !13
  %133 = getelementptr i8, ptr addrspace(3) %132, i64 %.idx1416, !dbg !13
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local561, !dbg !13
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local565, !dbg !13
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local569, !dbg !13
  %address.local574 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local572, !dbg !13
  %137 = add nsw i64 %.idx1416, %.idx1415, !dbg !13
  %.idx1991 = shl nuw nsw i64 %address.local561, 1, !dbg !13
  %138 = or disjoint i64 %137, %.idx1991, !dbg !13
  %139 = or disjoint i64 %address.local565, %address.local569, !dbg !13
  %140 = shl nsw i64 %139, 1, !dbg !13
  %address.local574.idx = mul nuw nsw i64 %address.local571, 2032, !dbg !13
  %141 = add nsw i64 %138, %address.local574.idx, !dbg !13
  %142 = add nsw i64 %141, %140, !dbg !13
  %143 = lshr i64 %142, 6, !dbg !13
  %144 = and i64 %143, 8, !dbg !13
  %145 = xor i64 %144, %142, !dbg !13
  %146 = sub i64 %145, %142, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local574, i64 %146, !dbg !13
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %121, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base2076 = extractvalue [2 x <1 x float>] %121, 1, !dbg !13
  %lazy.seed2077 = shufflevector <1 x float> %lazy.base2076, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %147, <4 x float> %lazy.seed) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local585 = shl nuw nsw i32 %address.thread516, 1, !dbg !13
  %address.local586 = and i32 %address.local585, 24, !dbg !13
  %address.local587 = shl nuw nsw i32 %address.thread516, 8, !dbg !13
  %address.local588 = and i32 %address.local587, 768, !dbg !13
  %address.local589 = or disjoint i32 %address.local586, %address.local588, !dbg !13
  %address.local590 = shl nuw nsw i32 %address.thread516, 5, !dbg !13
  %address.local591 = and i32 %address.local590, 96, !dbg !13
  %address.local592 = or disjoint i32 %address.local589, %address.local591, !dbg !13
  %address.local593 = zext nneg i32 %address.local592 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %109) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local594 = lshr i32 %address.thread516, 1, !dbg !13
  %address.local595 = and i32 %address.local594, 1, !dbg !13
  %address.local596 = zext nneg i32 %address.local595 to i64, !dbg !13
  %address.local597 = add nsw i64 %address.base, %address.local596, !dbg !13
  %address.local598 = shl i64 %address.local597, 6, !dbg !13
  %address.local599 = sub i64 %address.local593, %address.local598, !dbg !13
  %150 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local599, !dbg !13
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local569, !dbg !13
  %152 = getelementptr [2 x i8], ptr addrspace(3) %151, i64 %address.local572, !dbg !13
  %address.local603 = getelementptr i8, ptr addrspace(3) %152, i64 64, !dbg !13
  %.idx19941997 = add i64 %address.local599, %address.local569, !dbg !13
  %153 = shl i64 %.idx19941997, 1, !dbg !13
  %154 = add nuw nsw i64 %address.local574.idx, 64, !dbg !13
  %155 = add i64 %154, %153, !dbg !13
  %156 = lshr i64 %155, 6, !dbg !13
  %157 = and i64 %156, 8, !dbg !13
  %158 = xor i64 %157, %155, !dbg !13
  %159 = sub i64 %158, %155, !dbg !13
  %kdma.swizzled1865 = getelementptr i8, ptr addrspace(3) %address.local603, i64 %159, !dbg !13
  %160 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1865, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1869 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 128, !dbg !13
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1866 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 4096, !dbg !13
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1866, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1870 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 4224, !dbg !13
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1867 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1871 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1868 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1872 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %167 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %148) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1873 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 1024, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1877 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 1152, !dbg !13
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %160, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1874 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 5120, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1878 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 5248, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %171) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1875 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1876 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %176 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_lds_pair !21
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
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %147, <4 x float> %lazy.seed2077) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %149, <4 x float> %189) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %160, <4 x float> %lazy.seed2077) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %162, <4 x float> %191) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %164, <4 x float> %lazy.seed2077) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %166, <4 x float> %193) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %161, <4 x float> %lazy.seed2077) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %163, <4 x float> %195) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %165, <4 x float> %lazy.seed2077) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %167, <4 x float> %197) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %169, <4 x float> %lazy.seed2077) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %172, <4 x float> %199) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %175, <4 x float> %lazy.seed2077) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %176, <4 x float> %201) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %170, <4 x float> %lazy.seed2077) #8, !dbg !13
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %173, <4 x float> %203) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %205 = or disjoint i64 %126, 64, !dbg !14
  %kdma.source1698 = or disjoint i64 %205, %k0.prime.kdma.source1515, !dbg !14
  %kdma.source1699 = trunc nuw nsw i64 %kdma.source1698 to i32, !dbg !14
  %kdma.source1700 = shl nuw nsw i32 %kdma.source1699, 1, !dbg !14
  %kdma.offset1701 = or disjoint i32 %kdma.source1700, %kdma.intra, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1701, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1720 = or disjoint i64 %205, %k0.prime.kdma.source1535, !dbg !14
  %kdma.source1721 = trunc nuw nsw i64 %kdma.source1720 to i32, !dbg !14
  %kdma.source1722 = shl nuw nsw i32 %kdma.source1721, 1, !dbg !14
  %kdma.offset1723 = or disjoint i32 %kdma.source1722, %kdma.intra.hi, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1540, i32 %kdma.offset1723, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1744 = or disjoint i64 %205, %k0.prime.kdma.source1559, !dbg !14
  %kdma.source1745 = trunc nuw nsw i64 %kdma.source1744 to i32, !dbg !14
  %kdma.source1746 = shl nuw nsw i32 %kdma.source1745, 1, !dbg !14
  %kdma.offset1747 = add nuw nsw i32 %kdma.source1746, %kdma.intra, !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1747, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1768 = or disjoint i64 %205, %k0.prime.kdma.source1583, !dbg !14
  %kdma.source1769 = trunc nuw nsw i64 %kdma.source1768 to i32, !dbg !14
  %kdma.source1770 = shl nuw nsw i32 %kdma.source1769, 1, !dbg !14
  %kdma.offset1771 = add nuw nsw i32 %kdma.source1770, %kdma.intra.hi, !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1588, i32 %kdma.offset1771, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.gep1895 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1895, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1899 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1896 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1900 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1897 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 8192, !dbg !13
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1901 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 8320, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1898 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 12288, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1902 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 12416, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1903 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1907 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %210, <4 x float> %168) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1904 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1908 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %212, <4 x float> %220) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1905 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 9216, !dbg !13
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1909 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 9344, !dbg !13
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %214, <4 x float> %174) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1906 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 13312, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1910 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1865, i64 13440, !dbg !13
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !13, !frisk.k_lds_pair !29
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
  %kdma.source1790 = or disjoint i64 %258, %k0.prime.kdma.source1515, !dbg !14
  %kdma.source1791 = trunc nuw nsw i64 %kdma.source1790 to i32, !dbg !14
  %kdma.source1792 = shl nuw nsw i32 %kdma.source1791, 1, !dbg !14
  %kdma.offset1793 = or disjoint i32 %kdma.source1792, %kdma.intra, !dbg !14
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1793, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1812 = or disjoint i64 %258, %k0.prime.kdma.source1535, !dbg !14
  %kdma.source1813 = trunc nuw nsw i64 %kdma.source1812 to i32, !dbg !14
  %kdma.source1814 = shl nuw nsw i32 %kdma.source1813, 1, !dbg !14
  %kdma.offset1815 = or disjoint i32 %kdma.source1814, %kdma.intra.hi, !dbg !14
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1632, i32 %kdma.offset1815, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1836 = or disjoint i64 %258, %k0.prime.kdma.source1559, !dbg !14
  %kdma.source1837 = trunc nuw nsw i64 %kdma.source1836 to i32, !dbg !14
  %kdma.source1838 = shl nuw nsw i32 %kdma.source1837, 1, !dbg !14
  %kdma.offset1839 = add nuw nsw i32 %kdma.source1838, %kdma.intra, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1839, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %kdma.source1860 = or disjoint i64 %258, %k0.prime.kdma.source1583, !dbg !14
  %kdma.source1861 = trunc nuw nsw i64 %kdma.source1860 to i32, !dbg !14
  %kdma.source1862 = shl nuw nsw i32 %kdma.source1861, 1, !dbg !14
  %kdma.offset1863 = add nuw nsw i32 %kdma.source1862, %kdma.intra.hi, !dbg !14
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1680, i32 %kdma.offset1863, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1867, align 8, !dbg !13, !frisk.k_lds_pair !30
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1868, align 8, !dbg !13, !frisk.k_lds_pair !31
  %267 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1865, align 8, !dbg !13, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1869, align 8, !dbg !13, !frisk.k_lds_pair !32
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1866, align 8, !dbg !13, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1870, align 8, !dbg !13, !frisk.k_lds_pair !33
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1871, align 8, !dbg !13, !frisk.k_lds_pair !34
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1875, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %263, <4 x float> %223) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1872, align 8, !dbg !13, !frisk.k_lds_pair !35
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1876, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %265, <4 x float> %273) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1873, align 8, !dbg !13, !frisk.k_lds_pair !36
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1877, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %267, <4 x float> %229) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1874, align 8, !dbg !13, !frisk.k_lds_pair !37
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1878, align 8, !dbg !13, !frisk.k_lds_pair !37
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
  %address.local932 = shl nuw nsw i64 %address.local517, 5, !dbg !38
  %address.local933 = and i64 %address.local932, 8064, !dbg !38
  %address.local934 = shl nuw nsw i64 %address.local517, 3, !dbg !9
  %address.local935 = shl nuw nsw i64 %address.local523, 5, !dbg !38
  %address.local936 = sub nsw i64 %address.local934, %address.local935, !dbg !38
  %address.local937 = add nsw i64 %address.local936, %address.local933, !dbg !38
  %.idx14251426 = add nsw i64 %address.local937, %126, !dbg !38
  %.idx14251426.tr = trunc nsw i64 %.idx14251426 to i32, !dbg !38
  %dma.buffer.offset1360 = shl nsw i32 %.idx14251426.tr, 1, !dbg !38
  %311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1360, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %address.local939 = shl nuw nsw i32 %address.thread516, 3, !dbg !38
  %address.local940 = and i32 %address.local939, 24, !dbg !38
  %address.local942 = and i32 %address.local590, 8064, !dbg !38
  %address.local943 = or disjoint i32 %address.local942, %address.local940, !dbg !38
  %address.local944 = or disjoint i32 %address.local943, 8192, !dbg !38
  %312 = trunc nuw nsw i64 %126 to i32, !dbg !38
  %.idx14271428.tr = or disjoint i32 %address.local944, %312, !dbg !38
  %dma.buffer.offset1358 = shl nuw nsw i32 %.idx14271428.tr, 1, !dbg !38
  %313 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1358, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1895, align 8, !dbg !13, !frisk.k_tail_pair !15
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1899, align 8, !dbg !13, !frisk.k_tail_pair !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1896, align 8, !dbg !13, !frisk.k_tail_pair !16
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1900, align 8, !dbg !13, !frisk.k_tail_pair !16
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1897, align 8, !dbg !13, !frisk.k_tail_pair !17
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1901, align 8, !dbg !13, !frisk.k_tail_pair !17
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1898, align 8, !dbg !13, !frisk.k_tail_pair !18
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1902, align 8, !dbg !13, !frisk.k_tail_pair !18
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
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1903, align 8, !dbg !13, !frisk.k_tail_pair !19
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1907, align 8, !dbg !13, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %315, <4 x float> %284) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %334 = extractelement <4 x float> %330, i64 0, !dbg !13
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %326, float %334), !dbg !7, !frisk.local_reduction !5
  %335 = extractelement <4 x float> %330, i64 1, !dbg !13
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %327, float %335), !dbg !7, !frisk.local_reduction !5
  %336 = extractelement <4 x float> %330, i64 2, !dbg !13
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %328, float %336), !dbg !7, !frisk.local_reduction !5
  %337 = extractelement <4 x float> %330, i64 3, !dbg !13
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %329, float %337), !dbg !7, !frisk.local_reduction !5
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1904, align 8, !dbg !13, !frisk.k_tail_pair !20
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1908, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %317, <4 x float> %333) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %341 = load <4 x half>, ptr addrspace(3) %kdma.gep1905, align 8, !dbg !13, !frisk.k_tail_pair !21
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1909, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %319, <4 x float> %286) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %344 = extractelement <4 x float> %340, i64 0, !dbg !13
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %344), !dbg !7, !frisk.local_reduction !5
  %345 = extractelement <4 x float> %340, i64 1, !dbg !13
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %345), !dbg !7, !frisk.local_reduction !5
  %346 = extractelement <4 x float> %340, i64 2, !dbg !13
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %346), !dbg !7, !frisk.local_reduction !5
  %347 = extractelement <4 x float> %340, i64 3, !dbg !13
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %347), !dbg !7, !frisk.local_reduction !5
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1906, align 8, !dbg !13, !frisk.k_tail_pair !22
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1910, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %321, <4 x float> %343) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %331, <4 x float> %288) #8, !dbg !13
  %352 = extractelement <4 x float> %350, i64 0, !dbg !13
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %352), !dbg !7, !frisk.local_reduction !5
  %353 = extractelement <4 x float> %350, i64 1, !dbg !13
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %353), !dbg !7, !frisk.local_reduction !5
  %354 = extractelement <4 x float> %350, i64 2, !dbg !13
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %354), !dbg !7, !frisk.local_reduction !5
  %355 = extractelement <4 x float> %350, i64 3, !dbg !13
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %355), !dbg !7, !frisk.local_reduction !5
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %338, <4 x float> %351) #8, !dbg !13
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %341, <4 x float> %290) #8, !dbg !13
  %358 = extractelement <4 x float> %356, i64 0, !dbg !13
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %358), !dbg !7, !frisk.local_reduction !5
  %359 = extractelement <4 x float> %356, i64 1, !dbg !13
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %359), !dbg !7, !frisk.local_reduction !5
  %360 = extractelement <4 x float> %356, i64 2, !dbg !13
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %360), !dbg !7, !frisk.local_reduction !5
  %361 = extractelement <4 x float> %356, i64 3, !dbg !13
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %361), !dbg !7, !frisk.local_reduction !5
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %348, <4 x float> %357) #8, !dbg !13
  %363 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %332, <4 x float> %292) #8, !dbg !13
  %364 = extractelement <4 x float> %362, i64 0, !dbg !13
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %364), !dbg !7, !frisk.local_reduction !5
  %365 = extractelement <4 x float> %362, i64 1, !dbg !13
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %365), !dbg !7, !frisk.local_reduction !5
  %366 = extractelement <4 x float> %362, i64 2, !dbg !13
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %366), !dbg !7, !frisk.local_reduction !5
  %367 = extractelement <4 x float> %362, i64 3, !dbg !13
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %367), !dbg !7, !frisk.local_reduction !5
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %339, <4 x float> %363) #8, !dbg !13
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %342, <4 x float> %294) #8, !dbg !13
  %370 = extractelement <4 x float> %368, i64 0, !dbg !13
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %370), !dbg !7, !frisk.local_reduction !5
  %371 = extractelement <4 x float> %368, i64 1, !dbg !13
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %371), !dbg !7, !frisk.local_reduction !5
  %372 = extractelement <4 x float> %368, i64 2, !dbg !13
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %372), !dbg !7, !frisk.local_reduction !5
  %373 = extractelement <4 x float> %368, i64 3, !dbg !13
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %373), !dbg !7, !frisk.local_reduction !5
  %374 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %349, <4 x float> %369) #8, !dbg !13
  %375 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %314, <4 x float> %296) #8, !dbg !13
  %376 = extractelement <4 x float> %374, i64 0, !dbg !13
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %376), !dbg !7, !frisk.local_reduction !5
  %377 = extractelement <4 x float> %374, i64 1, !dbg !13
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %377), !dbg !7, !frisk.local_reduction !5
  %378 = extractelement <4 x float> %374, i64 2, !dbg !13
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce176, float %378), !dbg !7, !frisk.local_reduction !5
  %379 = extractelement <4 x float> %374, i64 3, !dbg !13
  %local.reduce181 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %379), !dbg !7, !frisk.local_reduction !5
  %local.reduce182 = tail call float @llvm.maxnum.f32(float %local.reduce178, float %local.reduce179), !dbg !7, !frisk.local_reduction !5
  %local.reduce183 = tail call float @llvm.maxnum.f32(float %local.reduce180, float %local.reduce181), !dbg !7, !frisk.local_reduction !5
  %380 = tail call float @llvm.maxnum.f32(float %local.reduce182, float %local.reduce183), !dbg !7, !frisk.local_reduction !5
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
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %436, float %389), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %384, float %390), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %385, float %391), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %386, float %392), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %395), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %396), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %397), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %398), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %401), !dbg !7, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %402), !dbg !7, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %403), !dbg !7, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %404), !dbg !7, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %407), !dbg !7, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %408), !dbg !7, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %409), !dbg !7, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %410), !dbg !7, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %413), !dbg !7, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %414), !dbg !7, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %415), !dbg !7, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %416), !dbg !7, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %419), !dbg !7, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %420), !dbg !7, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %421), !dbg !7, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %422), !dbg !7, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %424), !dbg !7, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %425), !dbg !7, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %426), !dbg !7, !frisk.local_reduction !5
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %427), !dbg !7, !frisk.local_reduction !5
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %local.reduce148), !dbg !7, !frisk.local_reduction !5
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %local.reduce149, float %local.reduce150), !dbg !7, !frisk.local_reduction !5
  %437 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %local.reduce152), !dbg !7, !frisk.local_reduction !5
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
  %lazy.positive2078 = tail call float @llvm.maxnum.f32(float %445, float 0.000000e+00), !dbg !39
  %lazy.delta2079 = select i1 %lazy.first, float %445, float %lazy.positive2078, !dbg !39
  %lazy.negated2080 = fsub float 0.000000e+00, %lazy.delta2079, !dbg !39
  %lazy.exp2081 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated2080), !dbg !39
  %lazy.alpha2082 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp2081, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %323, %lazy.delta.v, !dbg !39
  %lazy.shifted2085 = fsub <4 x float> %330, %lazy.delta.v, !dbg !39
  %lazy.shifted2088 = fsub <4 x float> %340, %lazy.delta.v, !dbg !39
  %lazy.shifted2091 = fsub <4 x float> %350, %lazy.delta.v, !dbg !39
  %lazy.shifted2094 = fsub <4 x float> %356, %lazy.delta.v, !dbg !39
  %lazy.shifted2097 = fsub <4 x float> %362, %lazy.delta.v, !dbg !39
  %lazy.shifted2100 = fsub <4 x float> %368, %lazy.delta.v, !dbg !39
  %lazy.shifted2103 = fsub <4 x float> %374, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane2104 = insertelement <1 x float> poison, float %lazy.delta2079, i64 0, !dbg !39
  %lazy.delta.v2105 = shufflevector <1 x float> %lazy.delta.v.lane2104, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted2106 = fsub <4 x float> %381, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2109 = fsub <4 x float> %387, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2112 = fsub <4 x float> %393, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2115 = fsub <4 x float> %399, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2118 = fsub <4 x float> %405, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2121 = fsub <4 x float> %411, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2124 = fsub <4 x float> %417, %lazy.delta.v2105, !dbg !39
  %lazy.shifted2127 = fsub <4 x float> %423, %lazy.delta.v2105, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %122, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state2128 = extractvalue [2 x <1 x float>] %122, 1, !dbg !39
  %lazy.alpha.v.lane2129 = insertelement <1 x float> poison, float %lazy.alpha2082, i64 0, !dbg !39
  %lazy.scaled2130 = fmul <1 x float> %lazy.state2128, %lazy.alpha.v.lane2129, !dbg !39
  %lazy.state.next2131 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled2130, 1, !dbg !39
  %lazy.state2132 = extractvalue [2 x <8 x float>] %118, 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2134 = fmul <8 x float> %lazy.state2132, %lazy.alpha.v, !dbg !39
  %lazy.state.next2135 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2134, 0, !dbg !39
  %lazy.state2136 = extractvalue [2 x <8 x float>] %118, 1, !dbg !39
  %lazy.alpha.v2138 = shufflevector <1 x float> %lazy.alpha.v.lane2129, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled2139 = fmul <8 x float> %lazy.state2136, %lazy.alpha.v2138, !dbg !39
  %lazy.state.next2140 = insertvalue [2 x <8 x float>] %lazy.state.next2135, <8 x float> %lazy.scaled2139, 1, !dbg !39
  %lazy.state2141 = extractvalue [2 x <8 x float>] %119, 0, !dbg !39
  %lazy.scaled2144 = fmul <8 x float> %lazy.state2141, %lazy.alpha.v, !dbg !39
  %lazy.state.next2145 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2144, 0, !dbg !39
  %lazy.state2146 = extractvalue [2 x <8 x float>] %119, 1, !dbg !39
  %lazy.scaled2149 = fmul <8 x float> %lazy.state2146, %lazy.alpha.v2138, !dbg !39
  %lazy.state.next2150 = insertvalue [2 x <8 x float>] %lazy.state.next2145, <8 x float> %lazy.scaled2149, 1, !dbg !39
  %lazy.state2151 = extractvalue [2 x <8 x float>] %120, 0, !dbg !39
  %lazy.scaled2154 = fmul <8 x float> %lazy.state2151, %lazy.alpha.v, !dbg !39
  %lazy.state.next2155 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2154, 0, !dbg !39
  %lazy.state2156 = extractvalue [2 x <8 x float>] %120, 1, !dbg !39
  %lazy.scaled2159 = fmul <8 x float> %lazy.state2156, %lazy.alpha.v2138, !dbg !39
  %lazy.state.next2160 = insertvalue [2 x <8 x float>] %lazy.state.next2155, <8 x float> %lazy.scaled2159, 1, !dbg !39
  %lazy.state2161 = extractvalue [2 x <8 x float>] %123, 0, !dbg !39
  %lazy.scaled2164 = fmul <8 x float> %lazy.state2161, %lazy.alpha.v, !dbg !39
  %lazy.state.next2165 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2164, 0, !dbg !39
  %lazy.state2166 = extractvalue [2 x <8 x float>] %123, 1, !dbg !39
  %lazy.scaled2169 = fmul <8 x float> %lazy.state2166, %lazy.alpha.v2138, !dbg !39
  %lazy.state.next2170 = insertvalue [2 x <8 x float>] %lazy.state.next2165, <8 x float> %lazy.scaled2169, 1, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.next2173 = fsub <1 x float> %lazy.base2076, %lazy.delta.v.lane2104, !dbg !39
  %lazy.base.state2174 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2173, 1, !dbg !39
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !40
  %.pre25 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !40
  %.pre27 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !40
  %.pre29 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !40
  %.pre31 = extractelement <4 x float> %lazy.shifted2106, i64 0, !dbg !40
  %.pre33 = extractelement <4 x float> %lazy.shifted2106, i64 1, !dbg !40
  %.pre35 = extractelement <4 x float> %lazy.shifted2106, i64 2, !dbg !40
  %.pre37 = extractelement <4 x float> %lazy.shifted2106, i64 3, !dbg !40
  %.pre39 = extractelement <4 x float> %lazy.shifted2085, i64 0, !dbg !40
  %.pre41 = extractelement <4 x float> %lazy.shifted2085, i64 1, !dbg !40
  %.pre43 = extractelement <4 x float> %lazy.shifted2085, i64 2, !dbg !40
  %.pre45 = extractelement <4 x float> %lazy.shifted2085, i64 3, !dbg !40
  %.pre47 = extractelement <4 x float> %lazy.shifted2109, i64 0, !dbg !40
  %.pre49 = extractelement <4 x float> %lazy.shifted2109, i64 1, !dbg !40
  %.pre51 = extractelement <4 x float> %lazy.shifted2109, i64 2, !dbg !40
  %.pre53 = extractelement <4 x float> %lazy.shifted2109, i64 3, !dbg !40
  %.pre55 = extractelement <4 x float> %lazy.shifted2088, i64 0, !dbg !40
  %.pre57 = extractelement <4 x float> %lazy.shifted2088, i64 1, !dbg !40
  %.pre59 = extractelement <4 x float> %lazy.shifted2088, i64 2, !dbg !40
  %.pre61 = extractelement <4 x float> %lazy.shifted2088, i64 3, !dbg !40
  %.pre63 = extractelement <4 x float> %lazy.shifted2112, i64 0, !dbg !40
  %.pre65 = extractelement <4 x float> %lazy.shifted2112, i64 1, !dbg !40
  %.pre67 = extractelement <4 x float> %lazy.shifted2112, i64 2, !dbg !40
  %.pre69 = extractelement <4 x float> %lazy.shifted2112, i64 3, !dbg !40
  %.pre71 = extractelement <4 x float> %lazy.shifted2091, i64 0, !dbg !40
  %.pre73 = extractelement <4 x float> %lazy.shifted2091, i64 1, !dbg !40
  %.pre75 = extractelement <4 x float> %lazy.shifted2091, i64 2, !dbg !40
  %.pre77 = extractelement <4 x float> %lazy.shifted2091, i64 3, !dbg !40
  %.pre79 = extractelement <4 x float> %lazy.shifted2115, i64 0, !dbg !40
  %.pre81 = extractelement <4 x float> %lazy.shifted2115, i64 1, !dbg !40
  %.pre83 = extractelement <4 x float> %lazy.shifted2115, i64 2, !dbg !40
  %.pre85 = extractelement <4 x float> %lazy.shifted2115, i64 3, !dbg !40
  %.pre87 = extractelement <4 x float> %lazy.shifted2094, i64 0, !dbg !40
  %.pre89 = extractelement <4 x float> %lazy.shifted2094, i64 1, !dbg !40
  %.pre91 = extractelement <4 x float> %lazy.shifted2094, i64 2, !dbg !40
  %.pre93 = extractelement <4 x float> %lazy.shifted2094, i64 3, !dbg !40
  %.pre95 = extractelement <4 x float> %lazy.shifted2118, i64 0, !dbg !40
  %.pre97 = extractelement <4 x float> %lazy.shifted2118, i64 1, !dbg !40
  %.pre99 = extractelement <4 x float> %lazy.shifted2118, i64 2, !dbg !40
  %.pre101 = extractelement <4 x float> %lazy.shifted2118, i64 3, !dbg !40
  %.pre103 = extractelement <4 x float> %lazy.shifted2097, i64 0, !dbg !40
  %.pre105 = extractelement <4 x float> %lazy.shifted2097, i64 1, !dbg !40
  %.pre107 = extractelement <4 x float> %lazy.shifted2097, i64 2, !dbg !40
  %.pre109 = extractelement <4 x float> %lazy.shifted2097, i64 3, !dbg !40
  %.pre111 = extractelement <4 x float> %lazy.shifted2121, i64 0, !dbg !40
  %.pre113 = extractelement <4 x float> %lazy.shifted2121, i64 1, !dbg !40
  %.pre115 = extractelement <4 x float> %lazy.shifted2121, i64 2, !dbg !40
  %.pre117 = extractelement <4 x float> %lazy.shifted2121, i64 3, !dbg !40
  %.pre119 = extractelement <4 x float> %lazy.shifted2100, i64 0, !dbg !40
  %.pre121 = extractelement <4 x float> %lazy.shifted2100, i64 1, !dbg !40
  %.pre123 = extractelement <4 x float> %lazy.shifted2100, i64 2, !dbg !40
  %.pre125 = extractelement <4 x float> %lazy.shifted2100, i64 3, !dbg !40
  %.pre127 = extractelement <4 x float> %lazy.shifted2124, i64 0, !dbg !40
  %.pre129 = extractelement <4 x float> %lazy.shifted2124, i64 1, !dbg !40
  %.pre131 = extractelement <4 x float> %lazy.shifted2124, i64 2, !dbg !40
  %.pre133 = extractelement <4 x float> %lazy.shifted2124, i64 3, !dbg !40
  %.pre135 = extractelement <4 x float> %lazy.shifted2103, i64 0, !dbg !40
  %.pre137 = extractelement <4 x float> %lazy.shifted2103, i64 1, !dbg !40
  %.pre139 = extractelement <4 x float> %lazy.shifted2103, i64 2, !dbg !40
  %.pre141 = extractelement <4 x float> %lazy.shifted2103, i64 3, !dbg !40
  %.pre143 = extractelement <4 x float> %lazy.shifted2127, i64 0, !dbg !40
  %.pre145 = extractelement <4 x float> %lazy.shifted2127, i64 1, !dbg !40
  %.pre147 = extractelement <4 x float> %lazy.shifted2127, i64 2, !dbg !40
  %.pre149 = extractelement <4 x float> %lazy.shifted2127, i64 3, !dbg !40
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
  %lazy.state2190 = phi [2 x <1 x float>] [ %lazy.state.next2131, %lazy.slow ], [ %122, %117 ], !dbg !39
  %lazy.state2191 = phi [2 x <8 x float>] [ %lazy.state.next2140, %lazy.slow ], [ %118, %117 ], !dbg !39
  %lazy.state2192 = phi [2 x <8 x float>] [ %lazy.state.next2150, %lazy.slow ], [ %119, %117 ], !dbg !39
  %lazy.state2193 = phi [2 x <8 x float>] [ %lazy.state.next2160, %lazy.slow ], [ %120, %117 ], !dbg !39
  %lazy.state2194 = phi [2 x <8 x float>] [ %lazy.state.next2170, %lazy.slow ], [ %123, %117 ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2174, %lazy.slow ], [ %121, %117 ], !dbg !39
  %446 = extractvalue [2 x <1 x float>] %lazy.state2190, 0, !dbg !39
  %447 = extractelement <1 x float> %446, i64 0, !dbg !39
  %448 = extractvalue [2 x <1 x float>] %lazy.state2190, 1, !dbg !39
  %449 = extractelement <1 x float> %448, i64 0, !dbg !39
  %450 = extractvalue [2 x <8 x float>] %lazy.state2194, 0, !dbg !41
  %451 = shufflevector <8 x float> %450, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %452 = shufflevector <8 x float> %450, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %453 = extractvalue [2 x <8 x float>] %lazy.state2194, 1, !dbg !41
  %454 = shufflevector <8 x float> %453, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %455 = shufflevector <8 x float> %453, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = add nsw i64 %address.local937, %127, !dbg !38
  %vdb.address1441 = trunc nsw i64 %vdb.address to i32, !dbg !38
  %vdb.address1442 = shl nsw i32 %vdb.address1441, 1, !dbg !38
  %vdb.address1447 = trunc nuw nsw i64 %127 to i32, !dbg !38
  %vdb.address1448 = or disjoint i32 %address.local944, %vdb.address1447, !dbg !38
  %vdb.address1449 = shl nuw nsw i32 %vdb.address1448, 1, !dbg !38
  %456 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1442, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %457 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1454, i32 %vdb.address1449, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %address.local948 = and i32 %address.local939, 120, !dbg !42
  %address.local950 = and i32 %address.local590, 1536, !dbg !42
  %address.local951 = or disjoint i32 %address.local948, %address.local950, !dbg !42
  %address.local952 = zext nneg i32 %address.local951 to i64, !dbg !42
  %address.local953 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local952, !dbg !42
  %458 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %459 = bitcast <4 x i32> %458 to <8 x half>, !dbg !42
  %address.local954 = and i64 %address.local517, 60, !dbg !42
  %address.local955 = add nuw nsw i64 %address.local954, %address.local517, !dbg !42
  %address.local957.idx = shl nuw nsw i64 %address.local955, 4, !dbg !42
  %address.local957 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local957.idx, !dbg !42
  %address.local958 = lshr i64 %address.local567, 4, !dbg !42
  %address.local960.idx = mul nuw nsw i64 %address.local958, 768, !dbg !42
  %address.local960 = getelementptr i8, ptr addrspace(3) %address.local957, i64 %address.local960.idx, !dbg !42
  %address.local962.idx = mul nsw i64 %address.local523, -64, !dbg !42
  %address.local962 = getelementptr i8, ptr addrspace(3) %address.local960, i64 %address.local962.idx, !dbg !42
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local962, i64 256, !dbg !42
  %460 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %461 = bitcast <4 x i32> %460 to <8 x half>, !dbg !42
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local962, i64 512, !dbg !42
  %462 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %463 = bitcast <4 x i32> %462 to <8 x half>, !dbg !42
  %address.local971 = getelementptr i8, ptr addrspace(3) %address.local962, i64 768, !dbg !42
  %464 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %465 = bitcast <4 x i32> %464 to <8 x half>, !dbg !42
  %address.local973 = getelementptr i8, ptr addrspace(3) %address.local953, i64 4096, !dbg !42
  %466 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %467 = bitcast <4 x i32> %466 to <8 x half>, !dbg !42
  %address.local977 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4352, !dbg !42
  %468 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %469 = bitcast <4 x i32> %468 to <8 x half>, !dbg !42
  %address.local981 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4608, !dbg !42
  %470 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %471 = bitcast <4 x i32> %470 to <8 x half>, !dbg !42
  %address.local985 = getelementptr i8, ptr addrspace(3) %address.local962, i64 4864, !dbg !42
  %472 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local985, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
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
  %math.result403 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %493) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02195 = extractelement <4 x float> %math.result403, i64 0, !dbg !44
  %.cvtpack.f12196 = extractelement <4 x float> %math.result403, i64 1, !dbg !44
  %.cvtpack.f22197 = extractelement <4 x float> %math.result403, i64 2, !dbg !44
  %.cvtpack.f32198 = extractelement <4 x float> %math.result403, i64 3, !dbg !44
  %.cvtpack.p02199 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02195, float %.cvtpack.f12196) #12, !dbg !44
  %.cvtpack.p12200 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22197, float %.cvtpack.f32198) #12, !dbg !44
  %.cvtpack.v02201 = insertelement <2 x i32> poison, i32 %.cvtpack.p02199, i64 0, !dbg !44
  %.cvtpack.v12202 = insertelement <2 x i32> %.cvtpack.v02201, i32 %.cvtpack.p12200, i64 1, !dbg !44
  %494 = bitcast <2 x i32> %.cvtpack.v12202 to <4 x half>, !dbg !44
  %math.packed404 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %494) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %495 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %473, <4 x float> %454) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %496 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %484, <4 x float> %455) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %497 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %498 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi40), !dbg !40
  %local.reduce92 = fadd float %474, %498, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %499 = insertelement <4 x float> poison, float %498, i64 0, !dbg !44
  %500 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi42), !dbg !40
  %local.reduce93 = fadd float %476, %500, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1998 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %501 = insertelement <4 x float> %499, float %500, i64 1, !dbg !44
  %502 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi44), !dbg !40
  %local.reduce94 = fadd float %478, %502, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor1999 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %503 = insertelement <4 x float> %501, float %502, i64 2, !dbg !44
  %504 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi46), !dbg !40
  %local.reduce95 = fadd float %480, %504, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2000 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %505 = insertelement <4 x float> %503, float %504, i64 3, !dbg !44
  %math.result389 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %505) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02203 = extractelement <4 x float> %math.result389, i64 0, !dbg !44
  %.cvtpack.f12204 = extractelement <4 x float> %math.result389, i64 1, !dbg !44
  %.cvtpack.f22205 = extractelement <4 x float> %math.result389, i64 2, !dbg !44
  %.cvtpack.f32206 = extractelement <4 x float> %math.result389, i64 3, !dbg !44
  %.cvtpack.p02207 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02203, float %.cvtpack.f12204) #12, !dbg !44
  %.cvtpack.p12208 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22205, float %.cvtpack.f32206) #12, !dbg !44
  %.cvtpack.v02209 = insertelement <2 x i32> poison, i32 %.cvtpack.p02207, i64 0, !dbg !44
  %.cvtpack.v12210 = insertelement <2 x i32> %.cvtpack.v02209, i32 %.cvtpack.p12208, i64 1, !dbg !44
  %506 = bitcast <2 x i32> %.cvtpack.v12210 to <4 x half>, !dbg !44
  %math.packed390 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %506) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %497, <4 x float> %483) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %508 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %509 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %508, <4 x float> %485) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %510 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi48), !dbg !40
  %local.reduce = fadd float %486, %510, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2001 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %511 = insertelement <4 x float> poison, float %510, i64 0, !dbg !44
  %512 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi50), !dbg !40
  %local.reduce62 = fadd float %488, %512, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2002 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %513 = insertelement <4 x float> %511, float %512, i64 1, !dbg !44
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi52), !dbg !40
  %local.reduce63 = fadd float %490, %514, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2003 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %515 = insertelement <4 x float> %513, float %514, i64 2, !dbg !44
  %516 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi54), !dbg !40
  %local.reduce64 = fadd float %492, %516, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2004 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %517 = insertelement <4 x float> %515, float %516, i64 3, !dbg !44
  %math.result405 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %517) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02211 = extractelement <4 x float> %math.result405, i64 0, !dbg !44
  %.cvtpack.f12212 = extractelement <4 x float> %math.result405, i64 1, !dbg !44
  %.cvtpack.f22213 = extractelement <4 x float> %math.result405, i64 2, !dbg !44
  %.cvtpack.f32214 = extractelement <4 x float> %math.result405, i64 3, !dbg !44
  %.cvtpack.p02215 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02211, float %.cvtpack.f12212) #12, !dbg !44
  %.cvtpack.p12216 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22213, float %.cvtpack.f32214) #12, !dbg !44
  %.cvtpack.v02217 = insertelement <2 x i32> poison, i32 %.cvtpack.p02215, i64 0, !dbg !44
  %.cvtpack.v12218 = insertelement <2 x i32> %.cvtpack.v02217, i32 %.cvtpack.p12216, i64 1, !dbg !44
  %518 = bitcast <2 x i32> %.cvtpack.v12218 to <4 x half>, !dbg !44
  %math.packed406 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %518) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %519 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %497, <4 x float> %495) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %520 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %508, <4 x float> %496) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %521 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %522 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi56), !dbg !40
  %local.reduce96 = fadd float %522, %k0.sum.anchor, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2005 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %523 = insertelement <4 x float> poison, float %522, i64 0, !dbg !44
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi58), !dbg !40
  %local.reduce97 = fadd float %524, %k0.sum.anchor1998, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2006 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %525 = insertelement <4 x float> %523, float %524, i64 1, !dbg !44
  %526 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi60), !dbg !40
  %local.reduce98 = fadd float %526, %k0.sum.anchor1999, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2007 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %527 = insertelement <4 x float> %525, float %526, i64 2, !dbg !44
  %528 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi62), !dbg !40
  %local.reduce99 = fadd float %528, %k0.sum.anchor2000, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2008 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %529 = insertelement <4 x float> %527, float %528, i64 3, !dbg !44
  %math.result391 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %529) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02219 = extractelement <4 x float> %math.result391, i64 0, !dbg !44
  %.cvtpack.f12220 = extractelement <4 x float> %math.result391, i64 1, !dbg !44
  %.cvtpack.f22221 = extractelement <4 x float> %math.result391, i64 2, !dbg !44
  %.cvtpack.f32222 = extractelement <4 x float> %math.result391, i64 3, !dbg !44
  %.cvtpack.p02223 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02219, float %.cvtpack.f12220) #12, !dbg !44
  %.cvtpack.p12224 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22221, float %.cvtpack.f32222) #12, !dbg !44
  %.cvtpack.v02225 = insertelement <2 x i32> poison, i32 %.cvtpack.p02223, i64 0, !dbg !44
  %.cvtpack.v12226 = insertelement <2 x i32> %.cvtpack.v02225, i32 %.cvtpack.p12224, i64 1, !dbg !44
  %530 = bitcast <2 x i32> %.cvtpack.v12226 to <4 x half>, !dbg !44
  %math.packed392 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %530) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %521, <4 x float> %507) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %532 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %532, <4 x float> %509) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi64), !dbg !40
  %local.reduce65 = fadd float %534, %k0.sum.anchor2001, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2009 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !44
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi66), !dbg !40
  %local.reduce66 = fadd float %536, %k0.sum.anchor2002, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2010 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %537 = insertelement <4 x float> %535, float %536, i64 1, !dbg !44
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi68), !dbg !40
  %local.reduce67 = fadd float %538, %k0.sum.anchor2003, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2011 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %539 = insertelement <4 x float> %537, float %538, i64 2, !dbg !44
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi70), !dbg !40
  %local.reduce68 = fadd float %540, %k0.sum.anchor2004, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2012 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %541 = insertelement <4 x float> %539, float %540, i64 3, !dbg !44
  %math.result407 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %541) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02227 = extractelement <4 x float> %math.result407, i64 0, !dbg !44
  %.cvtpack.f12228 = extractelement <4 x float> %math.result407, i64 1, !dbg !44
  %.cvtpack.f22229 = extractelement <4 x float> %math.result407, i64 2, !dbg !44
  %.cvtpack.f32230 = extractelement <4 x float> %math.result407, i64 3, !dbg !44
  %.cvtpack.p02231 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02227, float %.cvtpack.f12228) #12, !dbg !44
  %.cvtpack.p12232 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22229, float %.cvtpack.f32230) #12, !dbg !44
  %.cvtpack.v02233 = insertelement <2 x i32> poison, i32 %.cvtpack.p02231, i64 0, !dbg !44
  %.cvtpack.v12234 = insertelement <2 x i32> %.cvtpack.v02233, i32 %.cvtpack.p12232, i64 1, !dbg !44
  %542 = bitcast <2 x i32> %.cvtpack.v12234 to <4 x half>, !dbg !44
  %math.packed408 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %542) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %521, <4 x float> %519) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %532, <4 x float> %520) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %545 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi72), !dbg !40
  %local.reduce100 = fadd float %546, %k0.sum.anchor2005, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2013 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !44
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi74), !dbg !40
  %local.reduce101 = fadd float %548, %k0.sum.anchor2006, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2014 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %549 = insertelement <4 x float> %547, float %548, i64 1, !dbg !44
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi76), !dbg !40
  %local.reduce102 = fadd float %550, %k0.sum.anchor2007, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2015 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %551 = insertelement <4 x float> %549, float %550, i64 2, !dbg !44
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi78), !dbg !40
  %local.reduce103 = fadd float %552, %k0.sum.anchor2008, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2016 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %553 = insertelement <4 x float> %551, float %552, i64 3, !dbg !44
  %math.result393 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %553) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02235 = extractelement <4 x float> %math.result393, i64 0, !dbg !44
  %.cvtpack.f12236 = extractelement <4 x float> %math.result393, i64 1, !dbg !44
  %.cvtpack.f22237 = extractelement <4 x float> %math.result393, i64 2, !dbg !44
  %.cvtpack.f32238 = extractelement <4 x float> %math.result393, i64 3, !dbg !44
  %.cvtpack.p02239 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02235, float %.cvtpack.f12236) #12, !dbg !44
  %.cvtpack.p12240 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22237, float %.cvtpack.f32238) #12, !dbg !44
  %.cvtpack.v02241 = insertelement <2 x i32> poison, i32 %.cvtpack.p02239, i64 0, !dbg !44
  %.cvtpack.v12242 = insertelement <2 x i32> %.cvtpack.v02241, i32 %.cvtpack.p12240, i64 1, !dbg !44
  %554 = bitcast <2 x i32> %.cvtpack.v12242 to <4 x half>, !dbg !44
  %math.packed394 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %554) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %545, <4 x float> %531) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %556 = shufflevector <8 x half> %465, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %556, <4 x float> %533) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %558 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi80), !dbg !40
  %local.reduce69 = fadd float %558, %k0.sum.anchor2009, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2017 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %559 = insertelement <4 x float> poison, float %558, i64 0, !dbg !44
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi82), !dbg !40
  %local.reduce70 = fadd float %560, %k0.sum.anchor2010, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2018 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %561 = insertelement <4 x float> %559, float %560, i64 1, !dbg !44
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi84), !dbg !40
  %local.reduce71 = fadd float %562, %k0.sum.anchor2011, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2019 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> %561, float %562, i64 2, !dbg !44
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi86), !dbg !40
  %local.reduce72 = fadd float %564, %k0.sum.anchor2012, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2020 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %565 = insertelement <4 x float> %563, float %564, i64 3, !dbg !44
  %math.result409 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %565) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02243 = extractelement <4 x float> %math.result409, i64 0, !dbg !44
  %.cvtpack.f12244 = extractelement <4 x float> %math.result409, i64 1, !dbg !44
  %.cvtpack.f22245 = extractelement <4 x float> %math.result409, i64 2, !dbg !44
  %.cvtpack.f32246 = extractelement <4 x float> %math.result409, i64 3, !dbg !44
  %.cvtpack.p02247 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02243, float %.cvtpack.f12244) #12, !dbg !44
  %.cvtpack.p12248 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22245, float %.cvtpack.f32246) #12, !dbg !44
  %.cvtpack.v02249 = insertelement <2 x i32> poison, i32 %.cvtpack.p02247, i64 0, !dbg !44
  %.cvtpack.v12250 = insertelement <2 x i32> %.cvtpack.v02249, i32 %.cvtpack.p12248, i64 1, !dbg !44
  %566 = bitcast <2 x i32> %.cvtpack.v12250 to <4 x half>, !dbg !44
  %math.packed410 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %566) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %567 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %545, <4 x float> %543) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %556, <4 x float> %544) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %569 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi88), !dbg !40
  %local.reduce104 = fadd float %570, %k0.sum.anchor2013, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2021 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %571 = insertelement <4 x float> poison, float %570, i64 0, !dbg !44
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi90), !dbg !40
  %local.reduce105 = fadd float %572, %k0.sum.anchor2014, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2022 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %573 = insertelement <4 x float> %571, float %572, i64 1, !dbg !44
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi92), !dbg !40
  %local.reduce106 = fadd float %574, %k0.sum.anchor2015, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2023 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %575 = insertelement <4 x float> %573, float %574, i64 2, !dbg !44
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi94), !dbg !40
  %local.reduce107 = fadd float %576, %k0.sum.anchor2016, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2024 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %577 = insertelement <4 x float> %575, float %576, i64 3, !dbg !44
  %math.result395 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %577) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02251 = extractelement <4 x float> %math.result395, i64 0, !dbg !44
  %.cvtpack.f12252 = extractelement <4 x float> %math.result395, i64 1, !dbg !44
  %.cvtpack.f22253 = extractelement <4 x float> %math.result395, i64 2, !dbg !44
  %.cvtpack.f32254 = extractelement <4 x float> %math.result395, i64 3, !dbg !44
  %.cvtpack.p02255 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02251, float %.cvtpack.f12252) #12, !dbg !44
  %.cvtpack.p12256 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22253, float %.cvtpack.f32254) #12, !dbg !44
  %.cvtpack.v02257 = insertelement <2 x i32> poison, i32 %.cvtpack.p02255, i64 0, !dbg !44
  %.cvtpack.v12258 = insertelement <2 x i32> %.cvtpack.v02257, i32 %.cvtpack.p12256, i64 1, !dbg !44
  %578 = bitcast <2 x i32> %.cvtpack.v12258 to <4 x half>, !dbg !44
  %math.packed396 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %578) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %569, <4 x float> %555) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %580 = shufflevector <8 x half> %467, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %581 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %580, <4 x float> %557) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %582 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi96), !dbg !40
  %local.reduce73 = fadd float %582, %k0.sum.anchor2017, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2025 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %583 = insertelement <4 x float> poison, float %582, i64 0, !dbg !44
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi98), !dbg !40
  %local.reduce74 = fadd float %584, %k0.sum.anchor2018, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2026 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %583, float %584, i64 1, !dbg !44
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi100), !dbg !40
  %local.reduce75 = fadd float %586, %k0.sum.anchor2019, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2027 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %587 = insertelement <4 x float> %585, float %586, i64 2, !dbg !44
  %588 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi102), !dbg !40
  %local.reduce76 = fadd float %588, %k0.sum.anchor2020, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2028 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %589 = insertelement <4 x float> %587, float %588, i64 3, !dbg !44
  %math.result411 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %589) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02259 = extractelement <4 x float> %math.result411, i64 0, !dbg !44
  %.cvtpack.f12260 = extractelement <4 x float> %math.result411, i64 1, !dbg !44
  %.cvtpack.f22261 = extractelement <4 x float> %math.result411, i64 2, !dbg !44
  %.cvtpack.f32262 = extractelement <4 x float> %math.result411, i64 3, !dbg !44
  %.cvtpack.p02263 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02259, float %.cvtpack.f12260) #12, !dbg !44
  %.cvtpack.p12264 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22261, float %.cvtpack.f32262) #12, !dbg !44
  %.cvtpack.v02265 = insertelement <2 x i32> poison, i32 %.cvtpack.p02263, i64 0, !dbg !44
  %.cvtpack.v12266 = insertelement <2 x i32> %.cvtpack.v02265, i32 %.cvtpack.p12264, i64 1, !dbg !44
  %590 = bitcast <2 x i32> %.cvtpack.v12266 to <4 x half>, !dbg !44
  %math.packed412 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %590) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %569, <4 x float> %567) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %592 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %580, <4 x float> %568) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %593 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi104), !dbg !40
  %local.reduce108 = fadd float %594, %k0.sum.anchor2021, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2029 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !44
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi106), !dbg !40
  %local.reduce109 = fadd float %596, %k0.sum.anchor2022, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2030 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %597 = insertelement <4 x float> %595, float %596, i64 1, !dbg !44
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi108), !dbg !40
  %local.reduce110 = fadd float %598, %k0.sum.anchor2023, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2031 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %599 = insertelement <4 x float> %597, float %598, i64 2, !dbg !44
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi110), !dbg !40
  %local.reduce111 = fadd float %600, %k0.sum.anchor2024, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2032 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> %599, float %600, i64 3, !dbg !44
  %math.result397 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %601) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02267 = extractelement <4 x float> %math.result397, i64 0, !dbg !44
  %.cvtpack.f12268 = extractelement <4 x float> %math.result397, i64 1, !dbg !44
  %.cvtpack.f22269 = extractelement <4 x float> %math.result397, i64 2, !dbg !44
  %.cvtpack.f32270 = extractelement <4 x float> %math.result397, i64 3, !dbg !44
  %.cvtpack.p02271 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02267, float %.cvtpack.f12268) #12, !dbg !44
  %.cvtpack.p12272 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22269, float %.cvtpack.f32270) #12, !dbg !44
  %.cvtpack.v02273 = insertelement <2 x i32> poison, i32 %.cvtpack.p02271, i64 0, !dbg !44
  %.cvtpack.v12274 = insertelement <2 x i32> %.cvtpack.v02273, i32 %.cvtpack.p12272, i64 1, !dbg !44
  %602 = bitcast <2 x i32> %.cvtpack.v12274 to <4 x half>, !dbg !44
  %math.packed398 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %602) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %593, <4 x float> %579) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %604 = shufflevector <8 x half> %469, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %604, <4 x float> %581) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi112), !dbg !40
  %local.reduce77 = fadd float %606, %k0.sum.anchor2025, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2033 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %607 = insertelement <4 x float> poison, float %606, i64 0, !dbg !44
  %608 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi114), !dbg !40
  %local.reduce78 = fadd float %608, %k0.sum.anchor2026, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2034 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %609 = insertelement <4 x float> %607, float %608, i64 1, !dbg !44
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi116), !dbg !40
  %local.reduce79 = fadd float %610, %k0.sum.anchor2027, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2035 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> %609, float %610, i64 2, !dbg !44
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi118), !dbg !40
  %local.reduce80 = fadd float %612, %k0.sum.anchor2028, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2036 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %613 = insertelement <4 x float> %611, float %612, i64 3, !dbg !44
  %math.result413 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %613) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02275 = extractelement <4 x float> %math.result413, i64 0, !dbg !44
  %.cvtpack.f12276 = extractelement <4 x float> %math.result413, i64 1, !dbg !44
  %.cvtpack.f22277 = extractelement <4 x float> %math.result413, i64 2, !dbg !44
  %.cvtpack.f32278 = extractelement <4 x float> %math.result413, i64 3, !dbg !44
  %.cvtpack.p02279 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02275, float %.cvtpack.f12276) #12, !dbg !44
  %.cvtpack.p12280 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22277, float %.cvtpack.f32278) #12, !dbg !44
  %.cvtpack.v02281 = insertelement <2 x i32> poison, i32 %.cvtpack.p02279, i64 0, !dbg !44
  %.cvtpack.v12282 = insertelement <2 x i32> %.cvtpack.v02281, i32 %.cvtpack.p12280, i64 1, !dbg !44
  %614 = bitcast <2 x i32> %.cvtpack.v12282 to <4 x half>, !dbg !44
  %math.packed414 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %614) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %615 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %593, <4 x float> %591) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %604, <4 x float> %592) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %617 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %618 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi120), !dbg !40
  %local.reduce112 = fadd float %618, %k0.sum.anchor2029, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2037 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %619 = insertelement <4 x float> poison, float %618, i64 0, !dbg !44
  %620 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi122), !dbg !40
  %local.reduce113 = fadd float %620, %k0.sum.anchor2030, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2038 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %621 = insertelement <4 x float> %619, float %620, i64 1, !dbg !44
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi124), !dbg !40
  %local.reduce114 = fadd float %622, %k0.sum.anchor2031, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2039 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %623 = insertelement <4 x float> %621, float %622, i64 2, !dbg !44
  %624 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi126), !dbg !40
  %local.reduce115 = fadd float %624, %k0.sum.anchor2032, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2040 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %625 = insertelement <4 x float> %623, float %624, i64 3, !dbg !44
  %math.result399 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %625) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02283 = extractelement <4 x float> %math.result399, i64 0, !dbg !44
  %.cvtpack.f12284 = extractelement <4 x float> %math.result399, i64 1, !dbg !44
  %.cvtpack.f22285 = extractelement <4 x float> %math.result399, i64 2, !dbg !44
  %.cvtpack.f32286 = extractelement <4 x float> %math.result399, i64 3, !dbg !44
  %.cvtpack.p02287 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02283, float %.cvtpack.f12284) #12, !dbg !44
  %.cvtpack.p12288 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22285, float %.cvtpack.f32286) #12, !dbg !44
  %.cvtpack.v02289 = insertelement <2 x i32> poison, i32 %.cvtpack.p02287, i64 0, !dbg !44
  %.cvtpack.v12290 = insertelement <2 x i32> %.cvtpack.v02289, i32 %.cvtpack.p12288, i64 1, !dbg !44
  %626 = bitcast <2 x i32> %.cvtpack.v12290 to <4 x half>, !dbg !44
  %math.packed400 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %626) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %617, <4 x float> %603) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %628 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %628, <4 x float> %605) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi128), !dbg !40
  %local.reduce81 = fadd float %630, %k0.sum.anchor2033, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2041 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %631 = insertelement <4 x float> poison, float %630, i64 0, !dbg !44
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi130), !dbg !40
  %local.reduce82 = fadd float %632, %k0.sum.anchor2034, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2042 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %631, float %632, i64 1, !dbg !44
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi132), !dbg !40
  %local.reduce83 = fadd float %634, %k0.sum.anchor2035, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2043 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %635 = insertelement <4 x float> %633, float %634, i64 2, !dbg !44
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi134), !dbg !40
  %local.reduce84 = fadd float %636, %k0.sum.anchor2036, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2044 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %637 = insertelement <4 x float> %635, float %636, i64 3, !dbg !44
  %math.result415 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %637) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02291 = extractelement <4 x float> %math.result415, i64 0, !dbg !44
  %.cvtpack.f12292 = extractelement <4 x float> %math.result415, i64 1, !dbg !44
  %.cvtpack.f22293 = extractelement <4 x float> %math.result415, i64 2, !dbg !44
  %.cvtpack.f32294 = extractelement <4 x float> %math.result415, i64 3, !dbg !44
  %.cvtpack.p02295 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02291, float %.cvtpack.f12292) #12, !dbg !44
  %.cvtpack.p12296 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22293, float %.cvtpack.f32294) #12, !dbg !44
  %.cvtpack.v02297 = insertelement <2 x i32> poison, i32 %.cvtpack.p02295, i64 0, !dbg !44
  %.cvtpack.v12298 = insertelement <2 x i32> %.cvtpack.v02297, i32 %.cvtpack.p12296, i64 1, !dbg !44
  %638 = bitcast <2 x i32> %.cvtpack.v12298 to <4 x half>, !dbg !44
  %math.packed416 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %638) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %617, <4 x float> %615) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %640 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %628, <4 x float> %616) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %641 = bitcast <4 x i32> %472 to <8 x half>, !dbg !42
  %642 = shufflevector <8 x half> %641, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi136), !dbg !40
  %local.reduce116 = fadd float %643, %k0.sum.anchor2037, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2045 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %644 = insertelement <4 x float> poison, float %643, i64 0, !dbg !44
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi138), !dbg !40
  %local.reduce117 = fadd float %645, %k0.sum.anchor2038, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2046 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce120 = fadd float %k0.sum.anchor2045, %k0.sum.anchor2046, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2047 = tail call float asm sideeffect "", "=v,0"(float %local.reduce120) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> %644, float %645, i64 1, !dbg !44
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi140), !dbg !40
  %local.reduce118 = fadd float %647, %k0.sum.anchor2039, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2048 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %648 = insertelement <4 x float> %646, float %647, i64 2, !dbg !44
  %649 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi142), !dbg !40
  %local.reduce119 = fadd float %649, %k0.sum.anchor2040, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2049 = tail call float asm sideeffect "", "=v,0"(float %local.reduce119) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce121 = fadd float %k0.sum.anchor2048, %k0.sum.anchor2049, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2050 = tail call float asm sideeffect "", "=v,0"(float %local.reduce121) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %650 = fadd float %k0.sum.anchor2047, %k0.sum.anchor2050, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2051 = tail call float asm sideeffect "", "=v,0"(float %650) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %651 = fadd float %447, %k0.sum.anchor2051, !dbg !45
  %k0.sum.anchor2052 = tail call float asm sideeffect "", "=v,0"(float %651) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %652 = insertelement <1 x float> poison, float %k0.sum.anchor2052, i64 0, !dbg !12
  %653 = insertvalue [2 x <1 x float>] poison, <1 x float> %652, 0, !dbg !12
  %654 = insertelement <4 x float> %648, float %649, i64 3, !dbg !44
  %math.result401 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02299 = extractelement <4 x float> %math.result401, i64 0, !dbg !44
  %.cvtpack.f12300 = extractelement <4 x float> %math.result401, i64 1, !dbg !44
  %.cvtpack.f22301 = extractelement <4 x float> %math.result401, i64 2, !dbg !44
  %.cvtpack.f32302 = extractelement <4 x float> %math.result401, i64 3, !dbg !44
  %.cvtpack.p02303 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02299, float %.cvtpack.f12300) #12, !dbg !44
  %.cvtpack.p12304 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22301, float %.cvtpack.f32302) #12, !dbg !44
  %.cvtpack.v02305 = insertelement <2 x i32> poison, i32 %.cvtpack.p02303, i64 0, !dbg !44
  %.cvtpack.v12306 = insertelement <2 x i32> %.cvtpack.v02305, i32 %.cvtpack.p12304, i64 1, !dbg !44
  %655 = bitcast <2 x i32> %.cvtpack.v12306 to <4 x half>, !dbg !44
  %math.packed402 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %656 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %642, <4 x float> %627) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %657 = shufflevector <8 x half> %641, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %658 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %657, <4 x float> %629) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi144), !dbg !40
  %local.reduce85 = fadd float %659, %k0.sum.anchor2041, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2053 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %660 = insertelement <4 x float> poison, float %659, i64 0, !dbg !44
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi146), !dbg !40
  %local.reduce86 = fadd float %661, %k0.sum.anchor2042, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2054 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce89 = fadd float %k0.sum.anchor2053, %k0.sum.anchor2054, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2055 = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %660, float %661, i64 1, !dbg !44
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi148), !dbg !40
  %local.reduce87 = fadd float %663, %k0.sum.anchor2043, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2056 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %664 = insertelement <4 x float> %662, float %663, i64 2, !dbg !44
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi150), !dbg !40
  %local.reduce88 = fadd float %665, %k0.sum.anchor2044, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2057 = tail call float asm sideeffect "", "=v,0"(float %local.reduce88) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %local.reduce90 = fadd float %k0.sum.anchor2056, %k0.sum.anchor2057, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2058 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %666 = fadd float %k0.sum.anchor2055, %k0.sum.anchor2058, !dbg !45, !frisk.local_reduction !5
  %k0.sum.anchor2059 = tail call float asm sideeffect "", "=v,0"(float %666) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %667 = fadd float %449, %k0.sum.anchor2059, !dbg !45
  %k0.sum.anchor2060 = tail call float asm sideeffect "", "=v,0"(float %667) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %668 = insertelement <1 x float> poison, float %k0.sum.anchor2060, i64 0, !dbg !12
  %669 = insertvalue [2 x <1 x float>] %653, <1 x float> %668, 1, !dbg !12
  %670 = insertelement <4 x float> %664, float %665, i64 3, !dbg !44
  %math.result417 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02307 = extractelement <4 x float> %math.result417, i64 0, !dbg !44
  %.cvtpack.f12308 = extractelement <4 x float> %math.result417, i64 1, !dbg !44
  %.cvtpack.f22309 = extractelement <4 x float> %math.result417, i64 2, !dbg !44
  %.cvtpack.f32310 = extractelement <4 x float> %math.result417, i64 3, !dbg !44
  %.cvtpack.p02311 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02307, float %.cvtpack.f12308) #12, !dbg !44
  %.cvtpack.p12312 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22309, float %.cvtpack.f32310) #12, !dbg !44
  %.cvtpack.v02313 = insertelement <2 x i32> poison, i32 %.cvtpack.p02311, i64 0, !dbg !44
  %.cvtpack.v12314 = insertelement <2 x i32> %.cvtpack.v02313, i32 %.cvtpack.p12312, i64 1, !dbg !44
  %671 = bitcast <2 x i32> %.cvtpack.v12314 to <4 x half>, !dbg !44
  %math.packed418 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %672 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %642, <4 x float> %639) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %657, <4 x float> %640) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %674 = extractvalue [2 x <8 x float>] %lazy.state2193, 0, !dbg !41
  %675 = shufflevector <8 x float> %674, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %676 = shufflevector <8 x float> %674, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %677 = extractvalue [2 x <8 x float>] %lazy.state2193, 1, !dbg !41
  %678 = shufflevector <8 x float> %677, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %679 = shufflevector <8 x float> %677, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1462 = add nsw i64 %address.local937, %205, !dbg !38
  %vdb.address1463 = trunc nsw i64 %vdb.address1462 to i32, !dbg !38
  %vdb.address1464 = shl nsw i32 %vdb.address1463, 1, !dbg !38
  %vdb.address1469 = trunc nuw nsw i64 %205 to i32, !dbg !38
  %vdb.address1470 = or disjoint i32 %address.local944, %vdb.address1469, !dbg !38
  %vdb.address1471 = shl nuw nsw i32 %vdb.address1470, 1, !dbg !38
  %680 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %681 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %vdb.address1471, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local953, i64 8192, !dbg !42
  %682 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %683 = bitcast <4 x i32> %682 to <8 x half>, !dbg !42
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local962, i64 8448, !dbg !42
  %684 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %685 = bitcast <4 x i32> %684 to <8 x half>, !dbg !42
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local962, i64 8704, !dbg !42
  %686 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %687 = bitcast <4 x i32> %686 to <8 x half>, !dbg !42
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local962, i64 8960, !dbg !42
  %688 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %689 = bitcast <4 x i32> %688 to <8 x half>, !dbg !42
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local953, i64 12288, !dbg !42
  %690 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %691 = bitcast <4 x i32> %690 to <8 x half>, !dbg !42
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local962, i64 12544, !dbg !42
  %692 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %693 = bitcast <4 x i32> %692 to <8 x half>, !dbg !42
  %vdb.read1460 = getelementptr i8, ptr addrspace(3) %address.local962, i64 12800, !dbg !42
  %694 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %695 = bitcast <4 x i32> %694 to <8 x half>, !dbg !42
  %vdb.read1461 = getelementptr i8, ptr addrspace(3) %address.local962, i64 13056, !dbg !42
  %696 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %697 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %697, <4 x float> %675) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %699 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %699, <4 x float> %676) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %697, <4 x float> %678) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %699, <4 x float> %679) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %703 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %704 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %703, <4 x float> %698) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %705 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %706 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %705, <4 x float> %700) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %707 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %703, <4 x float> %701) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %708 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %705, <4 x float> %702) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %709 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %710 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %709, <4 x float> %704) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %711 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %712 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %711, <4 x float> %706) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %713 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %709, <4 x float> %707) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %714 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %711, <4 x float> %708) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %715 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %715, <4 x float> %710) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %717 = shufflevector <8 x half> %689, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %718 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %717, <4 x float> %712) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %719 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %715, <4 x float> %713) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %720 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %717, <4 x float> %714) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %721 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %722 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %721, <4 x float> %716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %723 = shufflevector <8 x half> %691, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %724 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %723, <4 x float> %718) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %721, <4 x float> %719) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %726 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %723, <4 x float> %720) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %727 = shufflevector <8 x half> %693, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %728 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %727, <4 x float> %722) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %729 = shufflevector <8 x half> %693, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %730 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %729, <4 x float> %724) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %727, <4 x float> %725) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %729, <4 x float> %726) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %733 = shufflevector <8 x half> %695, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %733, <4 x float> %728) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %735 = shufflevector <8 x half> %695, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %735, <4 x float> %730) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %737 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %733, <4 x float> %731) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %738 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %735, <4 x float> %732) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %739 = bitcast <4 x i32> %696 to <8 x half>, !dbg !42
  %740 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %740, <4 x float> %734) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %742 = shufflevector <8 x half> %739, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %742, <4 x float> %736) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %744 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %740, <4 x float> %737) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %745 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %742, <4 x float> %738) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %746 = extractvalue [2 x <8 x float>] %lazy.state2192, 0, !dbg !41
  %747 = shufflevector <8 x float> %746, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %748 = shufflevector <8 x float> %746, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %749 = extractvalue [2 x <8 x float>] %lazy.state2192, 1, !dbg !41
  %750 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %751 = shufflevector <8 x float> %749, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1476 = add nsw i64 %address.local937, %258, !dbg !38
  %vdb.address1477 = trunc nsw i64 %vdb.address1476 to i32, !dbg !38
  %vdb.address1478 = shl nsw i32 %vdb.address1477, 1, !dbg !38
  %vdb.address1483 = trunc nuw nsw i64 %258 to i32, !dbg !38
  %vdb.address1484 = or disjoint i32 %address.local944, %vdb.address1483, !dbg !38
  %vdb.address1485 = shl nuw nsw i32 %vdb.address1484, 1, !dbg !38
  %752 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1478, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %753 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1454, i32 %vdb.address1485, <4 x i32> %dma.buffer.rsrc1361) #9, !dbg !38
  %754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %755 = bitcast <4 x i32> %754 to <8 x half>, !dbg !42
  %756 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %757 = bitcast <4 x i32> %756 to <8 x half>, !dbg !42
  %758 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !42
  %760 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local971, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %761 = bitcast <4 x i32> %760 to <8 x half>, !dbg !42
  %762 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local973, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %763 = bitcast <4 x i32> %762 to <8 x half>, !dbg !42
  %764 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local977, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %765 = bitcast <4 x i32> %764 to <8 x half>, !dbg !42
  %766 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local981, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %767 = bitcast <4 x i32> %766 to <8 x half>, !dbg !42
  %768 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local985, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %769 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %769, <4 x float> %747) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %771 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %771, <4 x float> %748) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %769, <4 x float> %750) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %771, <4 x float> %751) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %775 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %775, <4 x float> %770) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %777 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %777, <4 x float> %772) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %779 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %775, <4 x float> %773) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %777, <4 x float> %774) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %781 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %781, <4 x float> %776) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %783 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %783, <4 x float> %778) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %785 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %781, <4 x float> %779) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %783, <4 x float> %780) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %787 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %788 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %787, <4 x float> %782) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %789 = shufflevector <8 x half> %761, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %790 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %789, <4 x float> %784) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %787, <4 x float> %785) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %789, <4 x float> %786) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %793 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %793, <4 x float> %788) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %795 = shufflevector <8 x half> %763, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %795, <4 x float> %790) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %793, <4 x float> %791) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %795, <4 x float> %792) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %799 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %799, <4 x float> %794) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %801 = shufflevector <8 x half> %765, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %801, <4 x float> %796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %799, <4 x float> %797) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %801, <4 x float> %798) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %805 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %805, <4 x float> %800) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %807 = shufflevector <8 x half> %767, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %807, <4 x float> %802) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %809 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %805, <4 x float> %803) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %807, <4 x float> %804) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %811 = bitcast <4 x i32> %768 to <8 x half>, !dbg !42
  %812 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %812, <4 x float> %806) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %814 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %814, <4 x float> %808) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %812, <4 x float> %809) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %814, <4 x float> %810) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %818 = extractvalue [2 x <8 x float>] %lazy.state2191, 0, !dbg !41
  %819 = shufflevector <8 x float> %818, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %820 = shufflevector <8 x float> %818, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %821 = extractvalue [2 x <8 x float>] %lazy.state2191, 1, !dbg !41
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
  %k0.next.2071 = add nuw nsw i64 %k0.next., %k0.prime.2061, !dbg !14
  %k0.next.kdma.source1516 = or disjoint i64 %k0.next.2071, %k0.prime.kdma.source1515, !dbg !14
  %k0.next.kdma.source1517 = trunc nuw nsw i64 %k0.next.kdma.source1516 to i32, !dbg !14
  %k0.next.kdma.source1518 = shl nuw nsw i32 %k0.next.kdma.source1517, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1518, %kdma.intra, !dbg !14
  %k0.next.2072 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1536 = or disjoint i64 %k0.next.2071, %k0.prime.kdma.source1535, !dbg !14
  %k0.next.kdma.source1537 = trunc nuw nsw i64 %k0.next.kdma.source1536 to i32, !dbg !14
  %k0.next.kdma.source1538 = shl nuw nsw i32 %k0.next.kdma.source1537, 1, !dbg !14
  %k0.next.kdma.offset1539 = or disjoint i32 %k0.next.kdma.source1538, %kdma.intra.hi, !dbg !14
  %k0.next.2073 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1540, i32 %k0.next.kdma.offset1539, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1560 = or disjoint i64 %k0.next.2071, %k0.prime.kdma.source1559, !dbg !14
  %k0.next.kdma.source1561 = trunc nuw nsw i64 %k0.next.kdma.source1560 to i32, !dbg !14
  %k0.next.kdma.source1562 = shl nuw nsw i32 %k0.next.kdma.source1561, 1, !dbg !14
  %k0.next.kdma.offset1563 = add nuw nsw i32 %k0.next.kdma.source1562, %kdma.intra, !dbg !14
  %k0.next.2074 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.next.kdma.offset1563, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1584 = or disjoint i64 %k0.next.2071, %k0.prime.kdma.source1583, !dbg !14
  %k0.next.kdma.source1585 = trunc nuw nsw i64 %k0.next.kdma.source1584 to i32, !dbg !14
  %k0.next.kdma.source1586 = shl nuw nsw i32 %k0.next.kdma.source1585, 1, !dbg !14
  %k0.next.kdma.offset1587 = add nuw nsw i32 %k0.next.kdma.source1586, %kdma.intra.hi, !dbg !14
  %k0.next.2075 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1588, i32 %k0.next.kdma.offset1587, <4 x i32> %k0.prime.dma.buffer.rsrc1377) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !42
  %827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %828 = bitcast <4 x i32> %827 to <8 x half>, !dbg !42
  %829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %830 = bitcast <4 x i32> %829 to <8 x half>, !dbg !42
  %831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %832 = bitcast <4 x i32> %831 to <8 x half>, !dbg !42
  %833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !42
  %835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %836 = bitcast <4 x i32> %835 to <8 x half>, !dbg !42
  %837 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1460, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %838 = bitcast <4 x i32> %837 to <8 x half>, !dbg !42
  %839 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1461, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %840 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %840, <4 x float> %819) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %842 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %842, <4 x float> %820) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %840, <4 x float> %822) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %842, <4 x float> %823) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %846 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %846, <4 x float> %841) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %848 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %848, <4 x float> %843) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %846, <4 x float> %844) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %848, <4 x float> %845) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %852 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %852, <4 x float> %847) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %854 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %854, <4 x float> %849) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %852, <4 x float> %850) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed408, <4 x half> %854, <4 x float> %851) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %858 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %858, <4 x float> %853) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %860 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %860, <4 x float> %855) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %858, <4 x float> %856) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed410, <4 x half> %860, <4 x float> %857) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %864 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %864, <4 x float> %859) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %866 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %866, <4 x float> %861) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %864, <4 x float> %862) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed412, <4 x half> %866, <4 x float> %863) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %870 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %870, <4 x float> %865) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %872 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %872, <4 x float> %867) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %870, <4 x float> %868) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed414, <4 x half> %872, <4 x float> %869) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %876 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %876, <4 x float> %871) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %878 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %878, <4 x float> %873) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %876, <4 x float> %874) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed416, <4 x half> %878, <4 x float> %875) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %882 = bitcast <4 x i32> %839 to <8 x half>, !dbg !42
  %883 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %883, <4 x float> %877) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %885 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %885, <4 x float> %879) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %883, <4 x float> %880) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed418, <4 x half> %885, <4 x float> %881) #8, !dbg !42, !frisk.pv_interleave !19
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
  %907 = shl nuw nsw i32 %8, 2, !dbg !46
  %908 = xor i32 %907, 64, !dbg !46
  %909 = bitcast float %k0.sum.anchor2060 to i32, !dbg !46
  %lane.sum.shuffle191 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %908, i32 %909), !dbg !46
  %910 = bitcast i32 %lane.sum.shuffle191 to float, !dbg !46
  %lane.sum.final192 = fadd float %k0.sum.anchor2060, %910, !dbg !46
  %911 = xor i32 %907, 128, !dbg !46
  %912 = bitcast float %lane.sum.final192 to i32, !dbg !46
  %lane.sum.shuffle193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %911, i32 %912), !dbg !46
  %913 = bitcast i32 %lane.sum.shuffle193 to float, !dbg !46
  %lane.sum.final194 = fadd float %lane.sum.final192, %913, !dbg !46
  %914 = insertelement <1 x float> poison, float %lane.sum.final194, i64 0, !dbg !46
  %915 = bitcast float %k0.sum.anchor2052 to i32, !dbg !46
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %908, i32 %915), !dbg !46
  %916 = bitcast i32 %lane.sum.shuffle to float, !dbg !46
  %lane.sum.final = fadd float %k0.sum.anchor2052, %916, !dbg !46
  %917 = bitcast float %lane.sum.final to i32, !dbg !46
  %lane.sum.shuffle185 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %911, i32 %917), !dbg !46
  %918 = bitcast i32 %lane.sum.shuffle185 to float, !dbg !46
  %lane.sum.final186 = fadd float %lane.sum.final, %918, !dbg !46
  %919 = insertelement <1 x float> poison, float %lane.sum.final186, i64 0, !dbg !46
  %920 = shufflevector <1 x float> %919, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %921 = shufflevector <1 x float> %914, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %921, !dbg !46
  %normalized = fmul <4 x float> %673, %row.reciprocal, !dbg !46
  %922 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !47
  %923 = extractelement <4 x half> %922, i64 3, !dbg !47
  %924 = extractelement <4 x half> %922, i64 2, !dbg !47
  %925 = extractelement <4 x half> %922, i64 1, !dbg !47
  %926 = extractelement <4 x half> %922, i64 0, !dbg !47
  %normalized40 = fmul <4 x float> %672, %row.reciprocal, !dbg !46
  %927 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !47
  %928 = extractelement <4 x half> %927, i64 3, !dbg !47
  %929 = extractelement <4 x half> %927, i64 2, !dbg !47
  %930 = extractelement <4 x half> %927, i64 1, !dbg !47
  %931 = extractelement <4 x half> %927, i64 0, !dbg !47
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 1.000000e+00), %920, !dbg !46
  %normalized42 = fmul <4 x float> %658, %row.reciprocal41, !dbg !46
  %932 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !47
  %933 = extractelement <4 x half> %932, i64 3, !dbg !47
  %934 = extractelement <4 x half> %932, i64 2, !dbg !47
  %935 = extractelement <4 x half> %932, i64 1, !dbg !47
  %936 = extractelement <4 x half> %932, i64 0, !dbg !47
  %normalized43 = fmul <4 x float> %656, %row.reciprocal41, !dbg !46
  %937 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !47
  %938 = extractelement <4 x half> %937, i64 3, !dbg !47
  %939 = extractelement <4 x half> %937, i64 2, !dbg !47
  %940 = extractelement <4 x half> %937, i64 1, !dbg !47
  %941 = extractelement <4 x half> %937, i64 0, !dbg !47
  %address.thread992 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local993 = zext nneg i32 %address.thread992 to i64, !dbg !9
  %942 = lshr i64 %address.local993, 2, !dbg !48
  %address.local996 = and i64 %942, 48, !dbg !48
  %address.local998 = and i64 %address.local993, 240, !dbg !48
  %address.local997 = add i64 %34, %address.local993, !dbg !48
  %address.local999 = sub i64 %address.local997, %address.local998, !dbg !48
  %address.local1000 = add i64 %address.local999, %address.local996, !dbg !48
  %943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !48
  %address.local1003.idx = shl nuw nsw i64 %address.local1000, 8, !dbg !48
  %address.local1003 = getelementptr inbounds nuw i8, ptr addrspace(1) %943, i64 %address.local1003.idx, !dbg !48
  %address.local1004 = lshr i64 %address.local993, 4, !dbg !11
  %address.local1005 = and i64 %address.local1004, 3, !dbg !11
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1005, !dbg !48
  store half %941, ptr addrspace(1) %address.local1006, align 2, !dbg !48
  %address.local1008 = or disjoint i64 %address.local1005, 4, !dbg !48
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1008, !dbg !48
  store half %940, ptr addrspace(1) %address.local1009, align 2, !dbg !48
  %address.local1011 = or disjoint i64 %address.local1005, 8, !dbg !48
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1011, !dbg !48
  store half %939, ptr addrspace(1) %address.local1012, align 2, !dbg !48
  %address.local1014 = or i64 %address.local1004, 12, !dbg !48
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1014, !dbg !48
  store half %938, ptr addrspace(1) %address.local1015, align 2, !dbg !48
  %address.local1017 = or disjoint i64 %address.local1005, 16, !dbg !48
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1017, !dbg !48
  store half %936, ptr addrspace(1) %address.local1018, align 2, !dbg !48
  %address.local1020 = or disjoint i64 %address.local1005, 20, !dbg !48
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1020, !dbg !48
  store half %935, ptr addrspace(1) %address.local1021, align 2, !dbg !48
  %address.local1023 = or disjoint i64 %address.local1005, 24, !dbg !48
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1023, !dbg !48
  store half %934, ptr addrspace(1) %address.local1024, align 2, !dbg !48
  %address.local1026 = or i64 %address.local1004, 28, !dbg !48
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1026, !dbg !48
  store half %933, ptr addrspace(1) %address.local1027, align 2, !dbg !48
  %address.local1031 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local1003, i64 16384, !dbg !48
  %address.local1032 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1005, !dbg !48
  store half %931, ptr addrspace(1) %address.local1032, align 2, !dbg !48
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1008, !dbg !48
  store half %930, ptr addrspace(1) %address.local1034, align 2, !dbg !48
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1011, !dbg !48
  store half %929, ptr addrspace(1) %address.local1036, align 2, !dbg !48
  %address.local1038 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1014, !dbg !48
  store half %928, ptr addrspace(1) %address.local1038, align 2, !dbg !48
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1017, !dbg !48
  store half %926, ptr addrspace(1) %address.local1040, align 2, !dbg !48
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1020, !dbg !48
  store half %925, ptr addrspace(1) %address.local1042, align 2, !dbg !48
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1023, !dbg !48
  store half %924, ptr addrspace(1) %address.local1044, align 2, !dbg !48
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1026, !dbg !48
  store half %923, ptr addrspace(1) %address.local1046, align 2, !dbg !48
  %normalized45 = fmul <4 x float> %745, %row.reciprocal, !dbg !46
  %944 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !47
  %945 = extractelement <4 x half> %944, i64 3, !dbg !47
  %946 = extractelement <4 x half> %944, i64 2, !dbg !47
  %947 = extractelement <4 x half> %944, i64 1, !dbg !47
  %948 = extractelement <4 x half> %944, i64 0, !dbg !47
  %normalized46 = fmul <4 x float> %744, %row.reciprocal, !dbg !46
  %949 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !47
  %950 = extractelement <4 x half> %949, i64 3, !dbg !47
  %951 = extractelement <4 x half> %949, i64 2, !dbg !47
  %952 = extractelement <4 x half> %949, i64 1, !dbg !47
  %953 = extractelement <4 x half> %949, i64 0, !dbg !47
  %normalized48 = fmul <4 x float> %743, %row.reciprocal41, !dbg !46
  %954 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !47
  %955 = extractelement <4 x half> %954, i64 3, !dbg !47
  %956 = extractelement <4 x half> %954, i64 2, !dbg !47
  %957 = extractelement <4 x half> %954, i64 1, !dbg !47
  %958 = extractelement <4 x half> %954, i64 0, !dbg !47
  %normalized49 = fmul <4 x float> %741, %row.reciprocal41, !dbg !46
  %959 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !47
  %960 = extractelement <4 x half> %959, i64 3, !dbg !47
  %961 = extractelement <4 x half> %959, i64 2, !dbg !47
  %962 = extractelement <4 x half> %959, i64 1, !dbg !47
  %963 = extractelement <4 x half> %959, i64 0, !dbg !47
  %address.local1061 = or disjoint i64 %address.local1005, 32, !dbg !48
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1061, !dbg !48
  store half %963, ptr addrspace(1) %address.local1062, align 2, !dbg !48
  %address.local1064 = or disjoint i64 %address.local1005, 36, !dbg !48
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1064, !dbg !48
  store half %962, ptr addrspace(1) %address.local1065, align 2, !dbg !48
  %address.local1067 = or disjoint i64 %address.local1005, 40, !dbg !48
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1067, !dbg !48
  store half %961, ptr addrspace(1) %address.local1068, align 2, !dbg !48
  %address.local1070 = or i64 %address.local1004, 44, !dbg !48
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1070, !dbg !48
  store half %960, ptr addrspace(1) %address.local1071, align 2, !dbg !48
  %address.local1073 = or disjoint i64 %address.local1005, 48, !dbg !48
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1073, !dbg !48
  store half %958, ptr addrspace(1) %address.local1074, align 2, !dbg !48
  %address.local1076 = or disjoint i64 %address.local1005, 52, !dbg !48
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1076, !dbg !48
  store half %957, ptr addrspace(1) %address.local1077, align 2, !dbg !48
  %address.local1079 = or disjoint i64 %address.local1005, 56, !dbg !48
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1079, !dbg !48
  store half %956, ptr addrspace(1) %address.local1080, align 2, !dbg !48
  %address.local1082 = or i64 %address.local1004, 60, !dbg !48
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1082, !dbg !48
  store half %955, ptr addrspace(1) %address.local1083, align 2, !dbg !48
  %address.local1088 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1061, !dbg !48
  store half %953, ptr addrspace(1) %address.local1088, align 2, !dbg !48
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1064, !dbg !48
  store half %952, ptr addrspace(1) %address.local1090, align 2, !dbg !48
  %address.local1092 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1067, !dbg !48
  store half %951, ptr addrspace(1) %address.local1092, align 2, !dbg !48
  %address.local1094 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1070, !dbg !48
  store half %950, ptr addrspace(1) %address.local1094, align 2, !dbg !48
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1073, !dbg !48
  store half %948, ptr addrspace(1) %address.local1096, align 2, !dbg !48
  %address.local1098 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1076, !dbg !48
  store half %947, ptr addrspace(1) %address.local1098, align 2, !dbg !48
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1079, !dbg !48
  store half %946, ptr addrspace(1) %address.local1100, align 2, !dbg !48
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1082, !dbg !48
  store half %945, ptr addrspace(1) %address.local1102, align 2, !dbg !48
  %normalized51 = fmul <4 x float> %817, %row.reciprocal, !dbg !46
  %964 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !47
  %965 = extractelement <4 x half> %964, i64 3, !dbg !47
  %966 = extractelement <4 x half> %964, i64 2, !dbg !47
  %967 = extractelement <4 x half> %964, i64 1, !dbg !47
  %968 = extractelement <4 x half> %964, i64 0, !dbg !47
  %normalized52 = fmul <4 x float> %816, %row.reciprocal, !dbg !46
  %969 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !47
  %970 = extractelement <4 x half> %969, i64 3, !dbg !47
  %971 = extractelement <4 x half> %969, i64 2, !dbg !47
  %972 = extractelement <4 x half> %969, i64 1, !dbg !47
  %973 = extractelement <4 x half> %969, i64 0, !dbg !47
  %normalized54 = fmul <4 x float> %815, %row.reciprocal41, !dbg !46
  %974 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !47
  %975 = extractelement <4 x half> %974, i64 3, !dbg !47
  %976 = extractelement <4 x half> %974, i64 2, !dbg !47
  %977 = extractelement <4 x half> %974, i64 1, !dbg !47
  %978 = extractelement <4 x half> %974, i64 0, !dbg !47
  %normalized55 = fmul <4 x float> %813, %row.reciprocal41, !dbg !46
  %979 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !47
  %980 = extractelement <4 x half> %979, i64 3, !dbg !47
  %981 = extractelement <4 x half> %979, i64 2, !dbg !47
  %982 = extractelement <4 x half> %979, i64 1, !dbg !47
  %983 = extractelement <4 x half> %979, i64 0, !dbg !47
  %address.local1117 = or disjoint i64 %address.local1005, 64, !dbg !48
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1117, !dbg !48
  store half %983, ptr addrspace(1) %address.local1118, align 2, !dbg !48
  %address.local1120 = or disjoint i64 %address.local1005, 68, !dbg !48
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1120, !dbg !48
  store half %982, ptr addrspace(1) %address.local1121, align 2, !dbg !48
  %address.local1123 = or disjoint i64 %address.local1005, 72, !dbg !48
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1123, !dbg !48
  store half %981, ptr addrspace(1) %address.local1124, align 2, !dbg !48
  %address.local1126 = or i64 %address.local1004, 76, !dbg !48
  %address.local1127 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1126, !dbg !48
  store half %980, ptr addrspace(1) %address.local1127, align 2, !dbg !48
  %address.local1129 = or disjoint i64 %address.local1005, 80, !dbg !48
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1129, !dbg !48
  store half %978, ptr addrspace(1) %address.local1130, align 2, !dbg !48
  %address.local1132 = or disjoint i64 %address.local1005, 84, !dbg !48
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1132, !dbg !48
  store half %977, ptr addrspace(1) %address.local1133, align 2, !dbg !48
  %address.local1135 = or disjoint i64 %address.local1005, 88, !dbg !48
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1135, !dbg !48
  store half %976, ptr addrspace(1) %address.local1136, align 2, !dbg !48
  %address.local1138 = or i64 %address.local1004, 92, !dbg !48
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1138, !dbg !48
  store half %975, ptr addrspace(1) %address.local1139, align 2, !dbg !48
  %address.local1144 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1117, !dbg !48
  store half %973, ptr addrspace(1) %address.local1144, align 2, !dbg !48
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1120, !dbg !48
  store half %972, ptr addrspace(1) %address.local1146, align 2, !dbg !48
  %address.local1148 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1123, !dbg !48
  store half %971, ptr addrspace(1) %address.local1148, align 2, !dbg !48
  %address.local1150 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1126, !dbg !48
  store half %970, ptr addrspace(1) %address.local1150, align 2, !dbg !48
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1129, !dbg !48
  store half %968, ptr addrspace(1) %address.local1152, align 2, !dbg !48
  %address.local1154 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1132, !dbg !48
  store half %967, ptr addrspace(1) %address.local1154, align 2, !dbg !48
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1135, !dbg !48
  store half %966, ptr addrspace(1) %address.local1156, align 2, !dbg !48
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1138, !dbg !48
  store half %965, ptr addrspace(1) %address.local1158, align 2, !dbg !48
  %normalized57 = fmul <4 x float> %888, %row.reciprocal, !dbg !46
  %984 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !47
  %985 = extractelement <4 x half> %984, i64 3, !dbg !47
  %986 = extractelement <4 x half> %984, i64 2, !dbg !47
  %987 = extractelement <4 x half> %984, i64 1, !dbg !47
  %988 = extractelement <4 x half> %984, i64 0, !dbg !47
  %normalized58 = fmul <4 x float> %887, %row.reciprocal, !dbg !46
  %989 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !47
  %990 = extractelement <4 x half> %989, i64 3, !dbg !47
  %991 = extractelement <4 x half> %989, i64 2, !dbg !47
  %992 = extractelement <4 x half> %989, i64 1, !dbg !47
  %993 = extractelement <4 x half> %989, i64 0, !dbg !47
  %normalized60 = fmul <4 x float> %886, %row.reciprocal41, !dbg !46
  %994 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !47
  %995 = extractelement <4 x half> %994, i64 3, !dbg !47
  %996 = extractelement <4 x half> %994, i64 2, !dbg !47
  %997 = extractelement <4 x half> %994, i64 1, !dbg !47
  %998 = extractelement <4 x half> %994, i64 0, !dbg !47
  %normalized61 = fmul <4 x float> %884, %row.reciprocal41, !dbg !46
  %999 = fptrunc <4 x float> %normalized61 to <4 x half>, !dbg !47
  %1000 = extractelement <4 x half> %999, i64 3, !dbg !47
  %1001 = extractelement <4 x half> %999, i64 2, !dbg !47
  %1002 = extractelement <4 x half> %999, i64 1, !dbg !47
  %1003 = extractelement <4 x half> %999, i64 0, !dbg !47
  %address.local1173 = or disjoint i64 %address.local1005, 96, !dbg !48
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1173, !dbg !48
  store half %1003, ptr addrspace(1) %address.local1174, align 2, !dbg !48
  %address.local1176 = or disjoint i64 %address.local1005, 100, !dbg !48
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1176, !dbg !48
  store half %1002, ptr addrspace(1) %address.local1177, align 2, !dbg !48
  %address.local1179 = or disjoint i64 %address.local1005, 104, !dbg !48
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1179, !dbg !48
  store half %1001, ptr addrspace(1) %address.local1180, align 2, !dbg !48
  %address.local1182 = or i64 %address.local1004, 108, !dbg !48
  %address.local1183 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1182, !dbg !48
  store half %1000, ptr addrspace(1) %address.local1183, align 2, !dbg !48
  %address.local1185 = or disjoint i64 %address.local1005, 112, !dbg !48
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1185, !dbg !48
  store half %998, ptr addrspace(1) %address.local1186, align 2, !dbg !48
  %address.local1188 = or disjoint i64 %address.local1005, 116, !dbg !48
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1188, !dbg !48
  store half %997, ptr addrspace(1) %address.local1189, align 2, !dbg !48
  %address.local1191 = or disjoint i64 %address.local1005, 120, !dbg !48
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1191, !dbg !48
  store half %996, ptr addrspace(1) %address.local1192, align 2, !dbg !48
  %address.local1194 = or i64 %address.local1004, 124, !dbg !48
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1003, i64 %address.local1194, !dbg !48
  store half %995, ptr addrspace(1) %address.local1195, align 2, !dbg !48
  %address.local1200 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1173, !dbg !48
  store half %993, ptr addrspace(1) %address.local1200, align 2, !dbg !48
  %address.local1202 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1176, !dbg !48
  store half %992, ptr addrspace(1) %address.local1202, align 2, !dbg !48
  %address.local1204 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1179, !dbg !48
  store half %991, ptr addrspace(1) %address.local1204, align 2, !dbg !48
  %address.local1206 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1182, !dbg !48
  store half %990, ptr addrspace(1) %address.local1206, align 2, !dbg !48
  %address.local1208 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1185, !dbg !48
  store half %988, ptr addrspace(1) %address.local1208, align 2, !dbg !48
  %address.local1210 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1188, !dbg !48
  store half %987, ptr addrspace(1) %address.local1210, align 2, !dbg !48
  %address.local1212 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1191, !dbg !48
  store half %986, ptr addrspace(1) %address.local1212, align 2, !dbg !48
  %address.local1214 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1031, i64 %address.local1194, !dbg !48
  store half %985, ptr addrspace(1) %address.local1214, align 2, !dbg !48
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

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #6

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
attributes #6 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }
attributes #12 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/direct_native_exp2")
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
