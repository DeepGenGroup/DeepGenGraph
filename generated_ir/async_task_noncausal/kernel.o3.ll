; ModuleID = 'generated_ir/async_task_noncausal/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal/input.mlir"

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
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = shl nsw i64 %10, 5, !dbg !7
  %12 = add nsw i64 %11, %8, !dbg !7
  %13 = lshr i64 %12, 5, !dbg !7
  %14 = and i64 %13, 576460752303423456, !dbg !7
  %15 = and i64 %8, 31, !dbg !7
  %16 = or disjoint i64 %14, %15, !dbg !7
  %17 = and i64 %13, 31, !dbg !7
  %18 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %19 = sext i32 %18 to i64, !dbg !7
  %20 = lshr i64 %12, 10, !dbg !8
  %21 = add nuw nsw i64 %13, %20, !dbg !8
  %22 = and i64 %21, 31, !dbg !8
  %23 = shl nuw nsw i64 %16, 7, !dbg !8
  %24 = shl i64 %20, 12, !dbg !8
  %25 = sub i64 %23, %24, !dbg !8
  %26 = shl nsw i64 %25, 7, !dbg !8
  %.idx = shl nuw nsw i64 %22, 20, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local407 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local408 = and i64 %address.local407, 1536, !dbg !8
  %address.local409 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local408, !dbg !8
  %address.local412 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local413 = and i64 %address.local412, 16256, !dbg !8
  %address.local414 = lshr i64 %address.local, 4, !dbg !8
  %address.local415 = and i64 %address.local414, 8, !dbg !8
  %address.local416 = or disjoint i64 %address.local413, %address.local415, !dbg !8
  %27 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1402 = shl i64 %25, 8, !dbg !8
  %28 = add i64 %.idx1402, %.idx, !dbg !8
  %address.local417.idx = shl nuw nsw i64 %address.local416, 1, !dbg !8
  %29 = or disjoint i64 %address.local417.idx, %28, !dbg !8
  %dma.buffer.offset1399 = trunc i64 %29 to i32, !dbg !8
  %30 = insertelement <2 x i64> poison, i64 %27, i64 0, !dbg !8
  %31 = bitcast <2 x i64> %30 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1400 = shufflevector <4 x i32> %31, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local418 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local419 = shl nuw nsw i64 %address.local418, 3, !dbg !8
  %address.local420 = and i64 %address.local419, 3584, !dbg !8
  %address.local421 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local420, !dbg !8
  %address.local426 = lshr i64 %address.local418, 4, !dbg !8
  %address.local427 = and i64 %address.local426, 24, !dbg !8
  %address.local428 = or disjoint i64 %address.local427, %address.local413, !dbg !8
  %address.local429.idx = shl nuw nsw i64 %address.local428, 1, !dbg !8
  %32 = or disjoint i64 %address.local429.idx, %28, !dbg !8
  %dma.buffer.offset1396 = trunc i64 %32 to i32, !dbg !8
  %33 = shl nuw nsw i64 %22, 19, !dbg !8
  %34 = add i64 %26, %33, !dbg !8
  %address.local430 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local408, !dbg !8
  %35 = or disjoint i64 %34, 32, !dbg !8
  %36 = or disjoint i64 %address.local416, %35, !dbg !8
  %.tr = trunc i64 %36 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.tr, 1, !dbg !8
  %address.local434 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local420, !dbg !8
  %37 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local409, i64 0, !dbg !8
  %38 = insertelement <4 x ptr addrspace(3)> %37, ptr addrspace(3) %address.local421, i64 1, !dbg !8
  %39 = insertelement <4 x ptr addrspace(3)> %38, ptr addrspace(3) %address.local430, i64 2, !dbg !8
  %40 = insertelement <4 x ptr addrspace(3)> %39, ptr addrspace(3) %address.local434, i64 3, !dbg !8
  %41 = ptrtoint <4 x ptr addrspace(3)> %40 to <4 x i64>, !dbg !8
  %42 = trunc <4 x i64> %41 to <4 x i32>, !dbg !8
  %43 = extractelement <4 x i32> %42, i64 0, !dbg !8
  %44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %dma.buffer.lds1401 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %45 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1401, i32 %dma.buffer.offset1399, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %46 = extractelement <4 x i32> %42, i64 1, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %48 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %49 = extractelement <4 x i32> %42, i64 2, !dbg !8
  %50 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %49), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %50), !dbg !8
  %51 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %52 = extractelement <4 x i32> %42, i64 3, !dbg !8
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !8
  %54 = or disjoint i64 %address.local428, %35, !dbg !8
  %.tr1406 = trunc i64 %54 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr1406, 1, !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %53), !dbg !8
  %55 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %address.local438 = lshr i32 %address.thread, 2, !dbg !10
  %address.local439 = and i32 %address.local438, 4, !dbg !10
  %address.local440 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local441 = and i32 %address.local440, 384, !dbg !10
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !10
  %address.local443 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local444 = and i32 %address.local443, 120, !dbg !10
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !10
  %address.local446 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local447 = and i32 %address.local446, 1024, !dbg !10
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !10
  %address.local449 = zext nneg i32 %address.local448 to i64, !dbg !10
  %address.local450 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local449, !dbg !10
  %56 = load <4 x half>, ptr addrspace(3) %address.local450, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local450, i64 4096, !dbg !10
  %57 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %58 = fmul <4 x half> %56, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local450, i64 1024, !dbg !10
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %60 = fmul <4 x half> %57, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local450, i64 5120, !dbg !10
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %62 = fmul <4 x half> %59, splat (half 1.275630e-01), !dbg !10
  %63 = fmul <4 x half> %61, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %64 = or disjoint i64 %34, 64, !dbg !8
  %65 = or disjoint i64 %address.local416, %64, !dbg !8
  %.tr1408 = trunc i64 %65 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1408, 1, !dbg !8
  %66 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1401, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %67 = or disjoint i64 %address.local428, %64, !dbg !8
  %.tr1410 = trunc i64 %67 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1410, 1, !dbg !8
  %68 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local450, i64 8192, !dbg !10
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %lds.const.gep1203 = getelementptr i8, ptr addrspace(3) %address.local450, i64 12288, !dbg !10
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !10
  %lds.const.gep1204 = getelementptr i8, ptr addrspace(3) %address.local450, i64 9216, !dbg !10
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !10
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1205 = getelementptr i8, ptr addrspace(3) %address.local450, i64 13312, !dbg !10
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !10
  %74 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !10
  %75 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !10
  %76 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %77 = or disjoint i64 %34, 96, !dbg !8
  %78 = or disjoint i64 %address.local416, %77, !dbg !8
  %.tr1412 = trunc i64 %78 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1412, 1, !dbg !8
  %79 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %80 = or disjoint i64 %address.local428, %77, !dbg !8
  %.tr1414 = trunc i64 %80 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1414, 1, !dbg !8
  %81 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1400) #9, !dbg !8
  %82 = load <4 x half>, ptr addrspace(3) %address.local450, align 8, !dbg !10
  %83 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %87 = fmul <4 x half> %83, splat (half 1.275630e-01), !dbg !10
  %88 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %89 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %91 = load <4 x half>, ptr addrspace(3) %lds.const.gep1203, align 8, !dbg !10
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1204, align 8, !dbg !10
  %93 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !10
  %94 = load <4 x half>, ptr addrspace(3) %lds.const.gep1205, align 8, !dbg !10
  %95 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !10
  %96 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !10
  %97 = fmul <4 x half> %94, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %98 = shl nuw nsw i64 %17, 12, !dbg !12
  %99 = shl nsw i64 %19, 17, !dbg !12
  %100 = shl nuw nsw i32 %kdma.tid, 2, !dbg !13
  %101 = and i32 %100, 3084, !dbg !13
  %102 = and i32 %kdma.tid, 1, !dbg !13
  %103 = add nuw nsw i32 %102, 1, !dbg !13
  %narrow = or disjoint i32 %103, %101, !dbg !13
  %104 = lshr i32 %narrow, 1, !dbg !13
  %105 = zext nneg i32 %104 to i64, !dbg !13
  %k0.prime.2007 = shl nuw nsw i64 %17, 19, !dbg !12
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !12
  %k0.prime.kdma.source1526 = lshr i64 %k0.prime.kdma.source, 4, !dbg !12
  %k0.prime.kdma.source1527 = and i64 %k0.prime.kdma.source1526, 8, !dbg !12
  %k0.prime.kdma.source1528 = and i64 %k0.prime.kdma.source, 91, !dbg !12
  %k0.prime.kdma.source1529 = lshr i64 %k0.prime.kdma.source, 2, !dbg !12
  %k0.prime.kdma.source1530 = and i64 %k0.prime.kdma.source1529, 4, !dbg !12
  %k0.prime.kdma.source1531 = add nuw nsw i64 %k0.prime.kdma.source1530, %k0.prime.kdma.source, !dbg !12
  %k0.prime.kdma.source1532 = and i64 %k0.prime.kdma.source1531, 4, !dbg !12
  %k0.prime.kdma.source1533 = or disjoint i64 %k0.prime.kdma.source1532, %k0.prime.kdma.source1528, !dbg !12
  %k0.prime.kdma.source1534 = shl nuw nsw i64 %k0.prime.kdma.source1533, 7, !dbg !12
  %k0.prime.kdma.source1535 = or disjoint i64 %k0.prime.kdma.source1534, %k0.prime.kdma.source1527, !dbg !12
  %k0.prime.kdma.source1536 = or disjoint i64 %k0.prime.kdma.source1535, %k0.prime.2007, !dbg !12
  %k0.prime.kdma.source1537 = trunc nuw nsw i64 %k0.prime.kdma.source1536 to i32, !dbg !12
  %k0.prime.kdma.source1538 = shl nuw nsw i32 %k0.prime.kdma.source1537, 1, !dbg !12
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1538, %kdma.intra, !dbg !12
  %k0.prime.2010 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !12
  %k0.prime.2011 = insertelement <2 x i64> poison, i64 %k0.prime.2010, i64 0, !dbg !12
  %k0.prime.2012 = bitcast <2 x i64> %k0.prime.2011 to <4 x i32>, !dbg !12
  %k0.prime.dma.buffer.rsrc1376 = shufflevector <4 x i32> %k0.prime.2012, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !12
  %k0.prime.2013 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1401, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1554 = add i32 %dma.buffer.lds1401, 512, !dbg !12
  %k0.prime.kdma.source1539 = zext nneg i32 %kdma.owner.hi to i64, !dbg !12
  %k0.prime.kdma.source1540 = lshr i64 %k0.prime.kdma.source1539, 4, !dbg !12
  %k0.prime.kdma.source1541 = and i64 %k0.prime.kdma.source1540, 8, !dbg !12
  %k0.prime.kdma.source1542 = and i64 %k0.prime.kdma.source1539, 123, !dbg !12
  %k0.prime.kdma.source1543 = lshr i64 %k0.prime.kdma.source1539, 2, !dbg !12
  %k0.prime.kdma.source1544 = and i64 %k0.prime.kdma.source1543, 4, !dbg !12
  %k0.prime.kdma.source1545 = add nuw nsw i64 %k0.prime.kdma.source1544, %k0.prime.kdma.source1539, !dbg !12
  %k0.prime.kdma.source1546 = and i64 %k0.prime.kdma.source1545, 4, !dbg !12
  %k0.prime.kdma.source1547 = or disjoint i64 %k0.prime.kdma.source1546, %k0.prime.kdma.source1542, !dbg !12
  %k0.prime.kdma.source1548 = shl nuw nsw i64 %k0.prime.kdma.source1547, 7, !dbg !12
  %k0.prime.kdma.source1549 = or disjoint i64 %k0.prime.kdma.source1548, %k0.prime.kdma.source1541, !dbg !12
  %k0.prime.kdma.source1550 = or disjoint i64 %k0.prime.kdma.source1549, %k0.prime.2007, !dbg !12
  %k0.prime.kdma.source1551 = trunc nuw nsw i64 %k0.prime.kdma.source1550 to i32, !dbg !12
  %k0.prime.kdma.source1552 = shl nuw nsw i32 %k0.prime.kdma.source1551, 1, !dbg !12
  %k0.prime.kdma.offset1553 = or disjoint i32 %k0.prime.kdma.source1552, %kdma.intra.hi, !dbg !12
  %k0.prime.2014 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1554, i32 %k0.prime.kdma.offset1553, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1566 = or disjoint i64 %k0.prime.kdma.source1535, 16, !dbg !12
  %k0.prime.kdma.source1567 = or disjoint i64 %k0.prime.kdma.source1566, %k0.prime.2007, !dbg !12
  %k0.prime.kdma.source1568 = trunc nuw nsw i64 %k0.prime.kdma.source1567 to i32, !dbg !12
  %k0.prime.kdma.source1569 = shl nuw nsw i32 %k0.prime.kdma.source1568, 1, !dbg !12
  %k0.prime.kdma.offset1570 = or disjoint i32 %k0.prime.kdma.source1569, %kdma.intra, !dbg !12
  %k0.prime.2016 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %k0.prime.kdma.offset1570, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1588 = add i32 %dma.buffer.lds1398, 512, !dbg !12
  %k0.prime.kdma.source1583 = or disjoint i64 %k0.prime.kdma.source1549, 16, !dbg !12
  %k0.prime.kdma.source1584 = or disjoint i64 %k0.prime.kdma.source1583, %k0.prime.2007, !dbg !12
  %k0.prime.kdma.source1585 = trunc nuw nsw i64 %k0.prime.kdma.source1584 to i32, !dbg !12
  %k0.prime.kdma.source1586 = shl nuw nsw i32 %k0.prime.kdma.source1585, 1, !dbg !12
  %k0.prime.kdma.offset1587 = or disjoint i32 %k0.prime.kdma.source1586, %kdma.intra.hi, !dbg !12
  %k0.prime.2017 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1588, i32 %k0.prime.kdma.offset1587, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %106 = or disjoint i64 %98, %99
  %107 = icmp slt i32 %18, 0
  %.lobit37 = ashr i32 %18, 31
  %108 = zext i32 %.lobit37 to i64
  %109 = ptrtoint ptr addrspace(1) %2 to i64
  %110 = insertelement <2 x i64> poison, i64 %109, i64 0
  %111 = bitcast <2 x i64> %110 to <4 x i32>
  %dma.buffer.rsrc1352 = shufflevector <4 x i32> %111, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %112 = insertelement <2 x i1> poison, i1 %107, i64 0, !dbg !12
  %113 = shufflevector <2 x i1> %112, <2 x i1> poison, <2 x i32> zeroinitializer, !dbg !12
  %invariant.op102 = or disjoint i32 %kdma.intra, 64, !dbg !11
  %invariant.op = sub i32 72, %kdma.intra, !dbg !11
  %invariant.op103 = or disjoint i32 %kdma.intra, 128, !dbg !11
  %invariant.op104 = sub i32 136, %kdma.intra, !dbg !11
  %invariant.op105 = or disjoint i32 %kdma.intra, 192, !dbg !11
  %invariant.op106 = sub i32 200, %kdma.intra, !dbg !11
  br label %114, !dbg !11

114:                                              ; preds = %4, %k0.join
  %115 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1063, %k0.join ]
  %116 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1059, %k0.join ]
  %117 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1055, %k0.join ]
  %118 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1051, %k0.join ]
  %119 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %814, %k0.join ]
  %120 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1049, %k0.join ]
  %121 = phi i64 [ 0, %4 ], [ %971, %k0.join ]
  %122 = shl nuw nsw i64 %121, 14, !dbg !12
  %123 = add nuw nsw i64 %122, %k0.prime.2007, !dbg !12
  %address.thread501 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !12, !range !9, !frisk.register_lifetimes !5
  %address.local502 = zext nneg i32 %address.thread501 to i64, !dbg !8
  %address.local503 = shl nuw nsw i64 %address.local502, 3, !dbg !8
  %address.local504 = and i64 %address.local503, 1536, !dbg !8
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local504, !dbg !8
  %address.local519 = or disjoint i64 %address.local502, 256, !dbg !8
  %address.local520 = shl nuw nsw i64 %address.local519, 3, !dbg !8
  %address.local521 = and i64 %address.local520, 3584, !dbg !8
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local521, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %124 = add nuw nsw i64 %121, %106, !dbg !12
  %address.local536 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local504, !dbg !8
  %.idx1420 = shl nuw nsw i64 %121, 15, !dbg !12
  %address.local540 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local521, !dbg !8
  %125 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local505, i64 0, !dbg !8
  %126 = insertelement <4 x ptr addrspace(3)> %125, ptr addrspace(3) %address.local522, i64 1, !dbg !8
  %127 = insertelement <4 x ptr addrspace(3)> %126, ptr addrspace(3) %address.local536, i64 2, !dbg !8
  %128 = insertelement <4 x ptr addrspace(3)> %127, ptr addrspace(3) %address.local540, i64 3, !dbg !8
  %129 = ptrtoint <4 x ptr addrspace(3)> %128 to <4 x i64>, !dbg !8
  %130 = trunc <4 x i64> %129 to <4 x i32>, !dbg !8
  %131 = extractelement <4 x i32> %130, i64 0, !dbg !12
  %132 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %131), !dbg !12
  %dma.buffer.lds1377 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %132), !dbg !12
  %kdma.lds1554 = add i32 %dma.buffer.lds1377, 512, !dbg !12
  %133 = extractelement <4 x i32> %130, i64 1, !dbg !12
  %134 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !12
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %134), !dbg !12
  %kdma.lds1588 = add i32 %dma.buffer.lds1374, 512, !dbg !12
  %135 = extractelement <4 x i32> %130, i64 2, !dbg !12
  %136 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %135), !dbg !12
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %136), !dbg !12
  %kdma.lds1624 = add i32 %dma.buffer.lds1371, 512, !dbg !12
  %137 = extractelement <4 x i32> %130, i64 3, !dbg !12
  %address.local545 = lshr i64 %address.local502, 2, !dbg !13
  %address.local548 = shl nuw nsw i64 %address.local502, 1, !dbg !13
  %address.local549 = and i64 %address.local548, 24, !dbg !13
  %address.local551 = shl i64 %address.local502, 62, !dbg !13
  %address.local552 = ashr exact i64 %address.local551, 57, !dbg !13
  %address.local553 = and i64 %address.local552, -64, !dbg !13
  %address.local556 = and i64 %address.local545, 12, !dbg !10
  %address.local558 = and i64 %address.local502, 63, !dbg !10
  %address.local559 = lshr i64 %address.local558, 5, !dbg !10
  %address.local560 = mul nuw nsw i64 %address.local559, 1016, !dbg !10
  %.idx1423 = mul nuw nsw i64 %address.local502, 320, !dbg !13
  %138 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1423, !dbg !13
  %.idx1424 = mul nsw i64 %address.local545, -1280, !dbg !13
  %139 = getelementptr i8, ptr addrspace(3) %138, i64 %.idx1424, !dbg !13
  %140 = getelementptr [2 x i8], ptr addrspace(3) %139, i64 %address.local549, !dbg !13
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local553, !dbg !13
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local556, !dbg !13
  %address.local562 = getelementptr [2 x i8], ptr addrspace(3) %142, i64 %address.local560, !dbg !13
  %143 = add nsw i64 %.idx1424, %.idx1423, !dbg !13
  %.idx1937 = shl nuw nsw i64 %address.local549, 1, !dbg !13
  %144 = or disjoint i64 %143, %.idx1937, !dbg !13
  %145 = or disjoint i64 %address.local553, %address.local556, !dbg !13
  %146 = shl nsw i64 %145, 1, !dbg !13
  %address.local562.idx = mul nuw nsw i64 %address.local559, 2032, !dbg !13
  %147 = add nsw i64 %144, %address.local562.idx, !dbg !13
  %148 = add nsw i64 %147, %146, !dbg !13
  %149 = lshr i64 %148, 6, !dbg !13
  %150 = and i64 %149, 8, !dbg !13
  %151 = xor i64 %150, %148, !dbg !13
  %152 = sub i64 %151, %148, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local562, i64 %152, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %address.local573 = shl nuw nsw i32 %address.thread501, 1, !dbg !13
  %address.local574 = and i32 %address.local573, 24, !dbg !13
  %address.local575 = shl nuw nsw i32 %address.thread501, 8, !dbg !13
  %address.local576 = and i32 %address.local575, 768, !dbg !13
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !13
  %address.local578 = shl nuw nsw i32 %address.thread501, 5, !dbg !13
  %address.local579 = and i32 %address.local578, 96, !dbg !13
  %address.local580 = or disjoint i32 %address.local577, %address.local579, !dbg !13
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !13
  %address.local582 = lshr i32 %address.thread501, 1, !dbg !13
  %address.local583 = and i32 %address.local582, 1, !dbg !13
  %address.local584 = zext nneg i32 %address.local583 to i64, !dbg !13
  %153 = add nuw nsw i64 %address.local562.idx, 64, !dbg !13
  %kdma.gep1813 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %kdma.gep1817 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %kdma.gep1814 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %kdma.gep1818 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %kdma.gep1821 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %kdma.gep1822 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %154 = insertelement <2 x i64> poison, i64 %124, i64 0, !dbg !12
  %155 = shufflevector <2 x i64> %154, <2 x i64> poison, <2 x i32> zeroinitializer, !dbg !12
  %156 = or disjoint <2 x i64> %155, <i64 1024, i64 2048>, !dbg !12
  %157 = sub nuw nsw <2 x i64> <i64 -1025, i64 -2049>, %155, !dbg !12
  %158 = select <2 x i1> %113, <2 x i64> %157, <2 x i64> %156, !dbg !12
  %159 = sdiv <2 x i64> %158, splat (i64 4096), !dbg !12
  %160 = extractelement <2 x i64> %159, i64 0, !dbg !12
  %161 = xor i64 %160, %108, !dbg !12
  %162 = shl i64 %161, 19, !dbg !12
  %163 = and i64 %162, 16252928, !dbg !12
  %kdma.source1600 = or disjoint i64 %163, %k0.prime.kdma.source1535, !dbg !12
  %kdma.source1601 = shl nuw nsw i64 %kdma.source1600, 1, !dbg !12
  %kdma.source1602 = or i64 %kdma.source1601, %.idx1420, !dbg !12
  %kdma.source1603 = trunc nuw nsw i64 %kdma.source1602 to i32, !dbg !12
  %kdma.offset1605.reass.reass = or disjoint i32 %kdma.source1603, %invariant.op102
  %kdma.source1618 = or disjoint i64 %163, %k0.prime.kdma.source1549, !dbg !12
  %kdma.source1619 = shl nuw nsw i64 %kdma.source1618, 1, !dbg !12
  %kdma.source1620 = or i64 %kdma.source1619, %.idx1420, !dbg !12
  %kdma.source1621 = trunc nuw nsw i64 %kdma.source1620 to i32, !dbg !12
  %kdma.offset1623.reass.reass = add i32 %kdma.source1621, %invariant.op
  %164 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1605.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %165 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1624, i32 %kdma.offset1623.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %166 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %137), !dbg !12
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %166), !dbg !12
  %kdma.source1637 = or disjoint i64 %163, %k0.prime.kdma.source1566, !dbg !12
  %kdma.source1638 = shl nuw nsw i64 %kdma.source1637, 1, !dbg !12
  %kdma.source1639 = or i64 %kdma.source1638, %.idx1420, !dbg !12
  %kdma.source1640 = trunc nuw nsw i64 %kdma.source1639 to i32, !dbg !12
  %kdma.source1641 = or i32 %kdma.source1640, 64, !dbg !12
  %kdma.offset1642 = add nuw nsw i32 %kdma.source1641, %kdma.intra, !dbg !12
  %167 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1642, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1656 = or disjoint i64 %163, %k0.prime.kdma.source1583, !dbg !12
  %kdma.source1657 = shl nuw nsw i64 %kdma.source1656, 1, !dbg !12
  %kdma.source1658 = or i64 %kdma.source1657, %.idx1420, !dbg !12
  %kdma.source1659 = trunc nuw nsw i64 %kdma.source1658 to i32, !dbg !12
  %kdma.source1660 = or i32 %kdma.source1659, 64, !dbg !12
  %kdma.offset1661 = add nuw nsw i32 %kdma.source1660, %kdma.intra.hi, !dbg !12
  %kdma.lds1662 = add i32 %dma.buffer.lds1368, 512, !dbg !12
  %168 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1662, i32 %kdma.offset1661, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %169 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !13
  %171 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %105) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local585 = add nsw i64 %address.base, %address.local584, !dbg !13
  %address.local586 = shl i64 %address.local585, 6, !dbg !13
  %address.local587 = sub i64 %address.local581, %address.local586, !dbg !13
  %172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local587, !dbg !13
  %173 = getelementptr [2 x i8], ptr addrspace(3) %172, i64 %address.local556, !dbg !13
  %174 = getelementptr [2 x i8], ptr addrspace(3) %173, i64 %address.local560, !dbg !13
  %address.local591 = getelementptr i8, ptr addrspace(3) %174, i64 64, !dbg !13
  %.idx19401943 = add i64 %address.local587, %address.local556, !dbg !13
  %175 = shl i64 %.idx19401943, 1, !dbg !13
  %176 = add i64 %153, %175, !dbg !13
  %177 = lshr i64 %176, 6, !dbg !13
  %178 = and i64 %177, 8, !dbg !13
  %179 = xor i64 %178, %176, !dbg !13
  %180 = sub i64 %179, %176, !dbg !13
  %kdma.swizzled1811 = getelementptr i8, ptr addrspace(3) %address.local591, i64 %180, !dbg !13
  %181 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1811, align 8, !dbg !13, !frisk.k_lds_pair !14
  %kdma.gep1815 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 128, !dbg !13
  %182 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !13, !frisk.k_lds_pair !14
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 4096, !dbg !13
  %183 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1816 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 4224, !dbg !13
  %184 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !13, !frisk.k_lds_pair !15
  %185 = load <4 x half>, ptr addrspace(3) %kdma.gep1813, align 8, !dbg !13, !frisk.k_lds_pair !16
  %186 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !13, !frisk.k_lds_pair !16
  %187 = load <4 x half>, ptr addrspace(3) %kdma.gep1814, align 8, !dbg !13, !frisk.k_lds_pair !17
  %188 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !13, !frisk.k_lds_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %171, <4 x float> %170) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1819 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 1024, !dbg !13
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1823 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 1152, !dbg !13
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %181, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1820 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 5120, !dbg !13
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1824 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 5248, !dbg !13
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %183, <4 x float> %192) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !13, !frisk.k_lds_pair !20
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %185, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %187, <4 x float> %198) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %182, <4 x float> zeroinitializer) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %184, <4 x float> %200) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %186, <4 x float> zeroinitializer) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %188, <4 x float> %202) #8, !dbg !13
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %190, <4 x float> zeroinitializer) #8, !dbg !13
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %193, <4 x float> %204) #8, !dbg !13
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %196, <4 x float> zeroinitializer) #8, !dbg !13
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %197, <4 x float> %206) #8, !dbg !13
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %58, <4 x half> %191, <4 x float> zeroinitializer) #8, !dbg !13
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %194, <4 x float> %208) #8, !dbg !13
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !13
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %171, <4 x float> %210) #8, !dbg !13
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %181, <4 x float> zeroinitializer) #8, !dbg !13
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %183, <4 x float> %212) #8, !dbg !13
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %185, <4 x float> zeroinitializer) #8, !dbg !13
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %187, <4 x float> %214) #8, !dbg !13
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %182, <4 x float> zeroinitializer) #8, !dbg !13
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %184, <4 x float> %216) #8, !dbg !13
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %186, <4 x float> zeroinitializer) #8, !dbg !13
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %188, <4 x float> %218) #8, !dbg !13
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %190, <4 x float> zeroinitializer) #8, !dbg !13
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %193, <4 x float> %220) #8, !dbg !13
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %196, <4 x float> zeroinitializer) #8, !dbg !13
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %197, <4 x float> %222) #8, !dbg !13
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %191, <4 x float> zeroinitializer) #8, !dbg !13
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %63, <4 x half> %194, <4 x float> %224) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %226 = extractelement <2 x i64> %159, i64 1, !dbg !12
  %227 = xor i64 %226, %108, !dbg !12
  %228 = shl i64 %227, 19, !dbg !12
  %229 = and i64 %228, 16252928, !dbg !12
  %kdma.source1674 = or disjoint i64 %229, %k0.prime.kdma.source1535, !dbg !12
  %kdma.source1675 = shl nuw nsw i64 %kdma.source1674, 1, !dbg !12
  %kdma.source1676 = or i64 %kdma.source1675, %.idx1420, !dbg !12
  %kdma.source1677 = trunc nuw nsw i64 %kdma.source1676 to i32, !dbg !12
  %kdma.offset1679.reass.reass = or disjoint i32 %kdma.source1677, %invariant.op103
  %230 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %kdma.offset1679.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1692 = or disjoint i64 %229, %k0.prime.kdma.source1549, !dbg !12
  %kdma.source1693 = shl nuw nsw i64 %kdma.source1692, 1, !dbg !12
  %kdma.source1694 = or i64 %kdma.source1693, %.idx1420, !dbg !12
  %kdma.source1695 = trunc nuw nsw i64 %kdma.source1694 to i32, !dbg !12
  %kdma.offset1697.reass.reass = add i32 %kdma.source1695, %invariant.op104
  %231 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1554, i32 %kdma.offset1697.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1711 = or disjoint i64 %229, %k0.prime.kdma.source1566, !dbg !12
  %kdma.source1712 = shl nuw nsw i64 %kdma.source1711, 1, !dbg !12
  %kdma.source1713 = or i64 %kdma.source1712, %.idx1420, !dbg !12
  %kdma.source1714 = trunc nuw nsw i64 %kdma.source1713 to i32, !dbg !12
  %kdma.source1715 = or i32 %kdma.source1714, 128, !dbg !12
  %kdma.offset1716 = add nuw nsw i32 %kdma.source1715, %kdma.intra, !dbg !12
  %232 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1716, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1730 = or disjoint i64 %229, %k0.prime.kdma.source1583, !dbg !12
  %kdma.source1731 = shl nuw nsw i64 %kdma.source1730, 1, !dbg !12
  %kdma.source1732 = or i64 %kdma.source1731, %.idx1420, !dbg !12
  %kdma.source1733 = trunc nuw nsw i64 %kdma.source1732 to i32, !dbg !12
  %kdma.source1734 = or i32 %kdma.source1733, 128, !dbg !12
  %kdma.offset1735 = add nuw nsw i32 %kdma.source1734, %kdma.intra.hi, !dbg !12
  %233 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1588, i32 %kdma.offset1735, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.gep1841 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %234 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1845 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %235 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1842 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %236 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1846 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %237 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1843 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 8192, !dbg !13
  %238 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1847 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 8320, !dbg !13
  %239 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1844 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 12288, !dbg !13
  %240 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1848 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 12416, !dbg !13
  %241 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1849 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1853 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !13, !frisk.k_lds_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %234, <4 x float> %189) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1850 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %245 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1854 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %236, <4 x float> %244) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1851 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 9216, !dbg !13
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1855 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 9344, !dbg !13
  %249 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %238, <4 x float> %195) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1852 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 13312, !dbg !13
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1856 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1811, i64 13440, !dbg !13
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %240, <4 x float> %250) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %235, <4 x float> %199) #8, !dbg !13
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %237, <4 x float> %254) #8, !dbg !13
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %239, <4 x float> %201) #8, !dbg !13
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %241, <4 x float> %256) #8, !dbg !13
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %242, <4 x float> %203) #8, !dbg !13
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %245, <4 x float> %258) #8, !dbg !13
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %248, <4 x float> %205) #8, !dbg !13
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %251, <4 x float> %260) #8, !dbg !13
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %243, <4 x float> %207) #8, !dbg !13
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %246, <4 x float> %262) #8, !dbg !13
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %249, <4 x float> %209) #8, !dbg !13
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %252, <4 x float> %264) #8, !dbg !13
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %234, <4 x float> %211) #8, !dbg !13
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %236, <4 x float> %266) #8, !dbg !13
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %238, <4 x float> %213) #8, !dbg !13
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %240, <4 x float> %268) #8, !dbg !13
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %235, <4 x float> %215) #8, !dbg !13
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %237, <4 x float> %270) #8, !dbg !13
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %239, <4 x float> %217) #8, !dbg !13
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %241, <4 x float> %272) #8, !dbg !13
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %242, <4 x float> %219) #8, !dbg !13
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %245, <4 x float> %274) #8, !dbg !13
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %248, <4 x float> %221) #8, !dbg !13
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %251, <4 x float> %276) #8, !dbg !13
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %243, <4 x float> %223) #8, !dbg !13
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %246, <4 x float> %278) #8, !dbg !13
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %249, <4 x float> %225) #8, !dbg !13
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %76, <4 x half> %252, <4 x float> %280) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %282 = or disjoint i64 %124, 3072, !dbg !12
  %283 = sub nuw nsw i64 -3073, %124, !dbg !12
  %284 = select i1 %107, i64 %283, i64 %282, !dbg !12
  %285 = sdiv i64 %284, 4096, !dbg !12
  %286 = xor i64 %285, %108, !dbg !12
  %287 = shl i64 %286, 19, !dbg !12
  %288 = and i64 %287, 16252928, !dbg !12
  %kdma.source1748 = or disjoint i64 %288, %k0.prime.kdma.source1535, !dbg !12
  %kdma.source1749 = shl nuw nsw i64 %kdma.source1748, 1, !dbg !12
  %kdma.source1750 = or i64 %kdma.source1749, %.idx1420, !dbg !12
  %kdma.source1751 = trunc nuw nsw i64 %kdma.source1750 to i32, !dbg !12
  %kdma.offset1753.reass.reass = or disjoint i32 %kdma.source1751, %invariant.op105
  %289 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1753.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1766 = or disjoint i64 %288, %k0.prime.kdma.source1549, !dbg !12
  %kdma.source1767 = shl nuw nsw i64 %kdma.source1766, 1, !dbg !12
  %kdma.source1768 = or i64 %kdma.source1767, %.idx1420, !dbg !12
  %kdma.source1769 = trunc nuw nsw i64 %kdma.source1768 to i32, !dbg !12
  %kdma.offset1771.reass.reass = add i32 %kdma.source1769, %invariant.op106
  %290 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1624, i32 %kdma.offset1771.reass.reass, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1785 = or disjoint i64 %288, %k0.prime.kdma.source1566, !dbg !12
  %kdma.source1786 = shl nuw nsw i64 %kdma.source1785, 1, !dbg !12
  %kdma.source1787 = or i64 %kdma.source1786, %.idx1420, !dbg !12
  %kdma.source1788 = trunc nuw nsw i64 %kdma.source1787 to i32, !dbg !12
  %kdma.source1789 = or i32 %kdma.source1788, 192, !dbg !12
  %kdma.offset1790 = add nuw nsw i32 %kdma.source1789, %kdma.intra, !dbg !12
  %291 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1790, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %kdma.source1804 = or disjoint i64 %288, %k0.prime.kdma.source1583, !dbg !12
  %kdma.source1805 = shl nuw nsw i64 %kdma.source1804, 1, !dbg !12
  %kdma.source1806 = or i64 %kdma.source1805, %.idx1420, !dbg !12
  %kdma.source1807 = trunc nuw nsw i64 %kdma.source1806 to i32, !dbg !12
  %kdma.source1808 = or i32 %kdma.source1807, 192, !dbg !12
  %kdma.offset1809 = add nuw nsw i32 %kdma.source1808, %kdma.intra.hi, !dbg !12
  %292 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1662, i32 %kdma.offset1809, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12
  %293 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !29
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1813, align 8, !dbg !13, !frisk.k_lds_pair !29
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !30
  %296 = load <4 x half>, ptr addrspace(3) %kdma.gep1814, align 8, !dbg !13, !frisk.k_lds_pair !30
  %297 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1811, align 8, !dbg !13, !frisk.k_lds_pair !31
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1815, align 8, !dbg !13, !frisk.k_lds_pair !31
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !13, !frisk.k_lds_pair !32
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1816, align 8, !dbg !13, !frisk.k_lds_pair !32
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1817, align 8, !dbg !13, !frisk.k_lds_pair !33
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1821, align 8, !dbg !13, !frisk.k_lds_pair !33
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %293, <4 x float> %247) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1818, align 8, !dbg !13, !frisk.k_lds_pair !34
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1822, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %295, <4 x float> %303) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1819, align 8, !dbg !13, !frisk.k_lds_pair !35
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1823, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %297, <4 x float> %253) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1820, align 8, !dbg !13, !frisk.k_lds_pair !36
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1824, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %312 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %299, <4 x float> %309) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %294, <4 x float> %255) #8, !dbg !13
  %314 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %296, <4 x float> %313) #8, !dbg !13
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %298, <4 x float> %257) #8, !dbg !13
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %300, <4 x float> %315) #8, !dbg !13
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %301, <4 x float> %259) #8, !dbg !13
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %304, <4 x float> %317) #8, !dbg !13
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %307, <4 x float> %261) #8, !dbg !13
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %310, <4 x float> %319) #8, !dbg !13
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %302, <4 x float> %263) #8, !dbg !13
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %305, <4 x float> %321) #8, !dbg !13
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %308, <4 x float> %265) #8, !dbg !13
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %311, <4 x float> %323) #8, !dbg !13
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %293, <4 x float> %267) #8, !dbg !13
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %295, <4 x float> %325) #8, !dbg !13
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %297, <4 x float> %269) #8, !dbg !13
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %299, <4 x float> %327) #8, !dbg !13
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %294, <4 x float> %271) #8, !dbg !13
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %296, <4 x float> %329) #8, !dbg !13
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %298, <4 x float> %273) #8, !dbg !13
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %300, <4 x float> %331) #8, !dbg !13
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %301, <4 x float> %275) #8, !dbg !13
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %304, <4 x float> %333) #8, !dbg !13
  %335 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %307, <4 x float> %277) #8, !dbg !13
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %310, <4 x float> %335) #8, !dbg !13
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %302, <4 x float> %279) #8, !dbg !13
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %305, <4 x float> %337) #8, !dbg !13
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %88, <4 x half> %308, <4 x float> %281) #8, !dbg !13
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %311, <4 x float> %339) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local920 = shl nuw nsw i64 %address.local502, 5, !dbg !37
  %address.local921 = and i64 %address.local920, 8064, !dbg !37
  %address.local923 = and i64 %address.local503, 24, !dbg !37
  %address.local924 = or disjoint i64 %address.local921, %address.local923, !dbg !37
  %.idx14331434 = or disjoint i64 %address.local924, %123, !dbg !37
  %.idx14331434.tr = trunc nuw nsw i64 %.idx14331434 to i32, !dbg !37
  %dma.buffer.offset1351 = shl nuw nsw i32 %.idx14331434.tr, 1, !dbg !37
  %341 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %dma.buffer.offset1351, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %address.local926 = shl nuw nsw i64 %address.local519, 5, !dbg !37
  %address.local927 = and i64 %address.local926, 16256, !dbg !37
  %address.local930 = or disjoint i64 %address.local927, %address.local923, !dbg !37
  %.idx14351436 = or disjoint i64 %address.local930, %123, !dbg !37
  %.idx14351436.tr = trunc nuw nsw i64 %.idx14351436 to i32, !dbg !37
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14351436.tr, 1, !dbg !37
  %342 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep1841, align 8, !dbg !13, !frisk.k_tail_pair !14
  %344 = load <4 x half>, ptr addrspace(3) %kdma.gep1845, align 8, !dbg !13, !frisk.k_tail_pair !14
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1842, align 8, !dbg !13, !frisk.k_tail_pair !15
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1846, align 8, !dbg !13, !frisk.k_tail_pair !15
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep1843, align 8, !dbg !13, !frisk.k_tail_pair !16
  %348 = load <4 x half>, ptr addrspace(3) %kdma.gep1847, align 8, !dbg !13, !frisk.k_tail_pair !16
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1844, align 8, !dbg !13, !frisk.k_tail_pair !17
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1848, align 8, !dbg !13, !frisk.k_tail_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %343, <4 x float> %306) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %345, <4 x float> %351) #8, !dbg !13
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %347, <4 x float> %312) #8, !dbg !13
  %354 = extractelement <4 x float> %352, i64 0, !dbg !38
  %355 = tail call float @llvm.maxnum.f32(float %354, float -inf), !dbg !39, !frisk.local_reduction !5
  %356 = extractelement <4 x float> %352, i64 1, !dbg !38
  %357 = extractelement <4 x float> %352, i64 2, !dbg !38
  %358 = extractelement <4 x float> %352, i64 3, !dbg !38
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %349, <4 x float> %353) #8, !dbg !13
  %360 = load <4 x half>, ptr addrspace(3) %kdma.gep1849, align 8, !dbg !13, !frisk.k_tail_pair !18
  %361 = load <4 x half>, ptr addrspace(3) %kdma.gep1853, align 8, !dbg !13, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %362 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %344, <4 x float> %314) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38, !frisk.k_tail_prefetch !5
  %363 = extractelement <4 x float> %359, i64 0, !dbg !38
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %355, float %363), !dbg !39, !frisk.local_reduction !5
  %364 = extractelement <4 x float> %359, i64 1, !dbg !38
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %356, float %364), !dbg !39, !frisk.local_reduction !5
  %365 = extractelement <4 x float> %359, i64 2, !dbg !38
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %357, float %365), !dbg !39, !frisk.local_reduction !5
  %366 = extractelement <4 x float> %359, i64 3, !dbg !38
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %358, float %366), !dbg !39, !frisk.local_reduction !5
  %367 = load <4 x half>, ptr addrspace(3) %kdma.gep1850, align 8, !dbg !13, !frisk.k_tail_pair !19
  %368 = load <4 x half>, ptr addrspace(3) %kdma.gep1854, align 8, !dbg !13, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %346, <4 x float> %362) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %370 = load <4 x half>, ptr addrspace(3) %kdma.gep1851, align 8, !dbg !13, !frisk.k_tail_pair !20
  %371 = load <4 x half>, ptr addrspace(3) %kdma.gep1855, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %348, <4 x float> %316) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38, !frisk.k_tail_prefetch !5
  %373 = extractelement <4 x float> %369, i64 0, !dbg !38
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %373), !dbg !39, !frisk.local_reduction !5
  %374 = extractelement <4 x float> %369, i64 1, !dbg !38
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %374), !dbg !39, !frisk.local_reduction !5
  %375 = extractelement <4 x float> %369, i64 2, !dbg !38
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %375), !dbg !39, !frisk.local_reduction !5
  %376 = extractelement <4 x float> %369, i64 3, !dbg !38
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %376), !dbg !39, !frisk.local_reduction !5
  %377 = load <4 x half>, ptr addrspace(3) %kdma.gep1852, align 8, !dbg !13, !frisk.k_tail_pair !21
  %378 = load <4 x half>, ptr addrspace(3) %kdma.gep1856, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %350, <4 x float> %372) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %360, <4 x float> %318) #8, !dbg !13
  %381 = extractelement <4 x float> %379, i64 0, !dbg !38
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %381), !dbg !39, !frisk.local_reduction !5
  %382 = extractelement <4 x float> %379, i64 1, !dbg !38
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %382), !dbg !39, !frisk.local_reduction !5
  %383 = extractelement <4 x float> %379, i64 2, !dbg !38
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %383), !dbg !39, !frisk.local_reduction !5
  %384 = extractelement <4 x float> %379, i64 3, !dbg !38
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %384), !dbg !39, !frisk.local_reduction !5
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %367, <4 x float> %380) #8, !dbg !13
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %370, <4 x float> %320) #8, !dbg !13
  %387 = extractelement <4 x float> %385, i64 0, !dbg !38
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %387), !dbg !39, !frisk.local_reduction !5
  %388 = extractelement <4 x float> %385, i64 1, !dbg !38
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %388), !dbg !39, !frisk.local_reduction !5
  %389 = extractelement <4 x float> %385, i64 2, !dbg !38
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %389), !dbg !39, !frisk.local_reduction !5
  %390 = extractelement <4 x float> %385, i64 3, !dbg !38
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %390), !dbg !39, !frisk.local_reduction !5
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %377, <4 x float> %386) #8, !dbg !13
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %361, <4 x float> %322) #8, !dbg !13
  %393 = extractelement <4 x float> %391, i64 0, !dbg !38
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %393), !dbg !39, !frisk.local_reduction !5
  %394 = extractelement <4 x float> %391, i64 1, !dbg !38
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %394), !dbg !39, !frisk.local_reduction !5
  %395 = extractelement <4 x float> %391, i64 2, !dbg !38
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %395), !dbg !39, !frisk.local_reduction !5
  %396 = extractelement <4 x float> %391, i64 3, !dbg !38
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %396), !dbg !39, !frisk.local_reduction !5
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %368, <4 x float> %392) #8, !dbg !13
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %371, <4 x float> %324) #8, !dbg !13
  %399 = extractelement <4 x float> %397, i64 0, !dbg !38
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %399), !dbg !39, !frisk.local_reduction !5
  %400 = extractelement <4 x float> %397, i64 1, !dbg !38
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %400), !dbg !39, !frisk.local_reduction !5
  %401 = extractelement <4 x float> %397, i64 2, !dbg !38
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %401), !dbg !39, !frisk.local_reduction !5
  %402 = extractelement <4 x float> %397, i64 3, !dbg !38
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %402), !dbg !39, !frisk.local_reduction !5
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %95, <4 x half> %378, <4 x float> %398) #8, !dbg !13
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %343, <4 x float> %326) #8, !dbg !13
  %405 = extractelement <4 x float> %403, i64 0, !dbg !38
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %405), !dbg !39, !frisk.local_reduction !5
  %406 = extractelement <4 x float> %403, i64 1, !dbg !38
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %406), !dbg !39, !frisk.local_reduction !5
  %407 = extractelement <4 x float> %403, i64 2, !dbg !38
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce176, float %407), !dbg !39, !frisk.local_reduction !5
  %408 = extractelement <4 x float> %403, i64 3, !dbg !38
  %local.reduce181 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %408), !dbg !39, !frisk.local_reduction !5
  %local.reduce182 = tail call float @llvm.maxnum.f32(float %local.reduce178, float %local.reduce179), !dbg !39, !frisk.local_reduction !5
  %local.reduce183 = tail call float @llvm.maxnum.f32(float %local.reduce180, float %local.reduce181), !dbg !39, !frisk.local_reduction !5
  %409 = tail call float @llvm.maxnum.f32(float %local.reduce182, float %local.reduce183), !dbg !39, !frisk.local_reduction !5
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %345, <4 x float> %404) #8, !dbg !13
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %347, <4 x float> %328) #8, !dbg !13
  %412 = extractelement <4 x float> %410, i64 0, !dbg !38
  %413 = extractelement <4 x float> %410, i64 1, !dbg !38
  %414 = extractelement <4 x float> %410, i64 2, !dbg !38
  %415 = extractelement <4 x float> %410, i64 3, !dbg !38
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %349, <4 x float> %411) #8, !dbg !13
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %344, <4 x float> %330) #8, !dbg !13
  %418 = extractelement <4 x float> %416, i64 0, !dbg !38
  %419 = extractelement <4 x float> %416, i64 1, !dbg !38
  %420 = extractelement <4 x float> %416, i64 2, !dbg !38
  %421 = extractelement <4 x float> %416, i64 3, !dbg !38
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %346, <4 x float> %417) #8, !dbg !13
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %348, <4 x float> %332) #8, !dbg !13
  %424 = extractelement <4 x float> %422, i64 0, !dbg !38
  %425 = extractelement <4 x float> %422, i64 1, !dbg !38
  %426 = extractelement <4 x float> %422, i64 2, !dbg !38
  %427 = extractelement <4 x float> %422, i64 3, !dbg !38
  %428 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %350, <4 x float> %423) #8, !dbg !13
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %360, <4 x float> %334) #8, !dbg !13
  %430 = extractelement <4 x float> %428, i64 0, !dbg !38
  %431 = extractelement <4 x float> %428, i64 1, !dbg !38
  %432 = extractelement <4 x float> %428, i64 2, !dbg !38
  %433 = extractelement <4 x float> %428, i64 3, !dbg !38
  %434 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %367, <4 x float> %429) #8, !dbg !13
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %370, <4 x float> %336) #8, !dbg !13
  %436 = extractelement <4 x float> %434, i64 0, !dbg !38
  %437 = extractelement <4 x float> %434, i64 1, !dbg !38
  %438 = extractelement <4 x float> %434, i64 2, !dbg !38
  %439 = extractelement <4 x float> %434, i64 3, !dbg !38
  %440 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %377, <4 x float> %435) #8, !dbg !13
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %361, <4 x float> %338) #8, !dbg !13
  %442 = extractelement <4 x float> %440, i64 0, !dbg !38
  %443 = extractelement <4 x float> %440, i64 1, !dbg !38
  %444 = extractelement <4 x float> %440, i64 2, !dbg !38
  %445 = extractelement <4 x float> %440, i64 3, !dbg !38
  %446 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %368, <4 x float> %441) #8, !dbg !13
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %371, <4 x float> %340) #8, !dbg !13
  %448 = extractelement <4 x float> %446, i64 0, !dbg !38
  %449 = extractelement <4 x float> %446, i64 1, !dbg !38
  %450 = extractelement <4 x float> %446, i64 2, !dbg !38
  %451 = extractelement <4 x float> %446, i64 3, !dbg !38
  %452 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %378, <4 x float> %447) #8, !dbg !13
  %453 = extractelement <4 x float> %452, i64 0, !dbg !38
  %454 = extractelement <4 x float> %452, i64 1, !dbg !38
  %455 = extractelement <4 x float> %452, i64 2, !dbg !38
  %456 = extractelement <4 x float> %452, i64 3, !dbg !38
  %457 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !39
  %458 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %457), !dbg !39
  %459 = add nuw nsw i32 %458, 64, !dbg !39
  %460 = xor i32 %458, 16, !dbg !39
  %461 = and i32 %459, 192, !dbg !39
  %462 = icmp samesign ult i32 %460, %461, !dbg !39
  %463 = select i1 %462, i32 %460, i32 %458, !dbg !39
  %464 = shl nuw nsw i32 %463, 2, !dbg !39
  %465 = bitcast float %409 to i32, !dbg !39
  %466 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %465), !dbg !39
  %467 = bitcast i32 %466 to float, !dbg !39
  %468 = tail call float @llvm.maxnum.f32(float %409, float %467), !dbg !39
  %469 = xor i32 %458, 32, !dbg !39
  %470 = icmp samesign ult i32 %469, %461, !dbg !39
  %471 = select i1 %470, i32 %469, i32 %458, !dbg !39
  %472 = shl nuw nsw i32 %471, 2, !dbg !39
  %473 = bitcast float %468 to i32, !dbg !39
  %474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %473), !dbg !39
  %475 = bitcast i32 %474 to float, !dbg !39
  %476 = tail call float @llvm.maxnum.f32(float %468, float %475), !dbg !39
  %477 = tail call float @llvm.maxnum.f32(float %412, float -inf), !dbg !39, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %477, float %418), !dbg !39, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %413, float %419), !dbg !39, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %414, float %420), !dbg !39, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %415, float %421), !dbg !39, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %424), !dbg !39, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %425), !dbg !39, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %426), !dbg !39, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %427), !dbg !39, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %430), !dbg !39, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %431), !dbg !39, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %432), !dbg !39, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %433), !dbg !39, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %436), !dbg !39, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %437), !dbg !39, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %438), !dbg !39, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %439), !dbg !39, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %442), !dbg !39, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %443), !dbg !39, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %444), !dbg !39, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %445), !dbg !39, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %448), !dbg !39, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %449), !dbg !39, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %450), !dbg !39, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %451), !dbg !39, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %453), !dbg !39, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %454), !dbg !39, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %455), !dbg !39, !frisk.local_reduction !5
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %456), !dbg !39, !frisk.local_reduction !5
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %local.reduce148), !dbg !39, !frisk.local_reduction !5
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %local.reduce149, float %local.reduce150), !dbg !39, !frisk.local_reduction !5
  %478 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %local.reduce152), !dbg !39, !frisk.local_reduction !5
  %479 = bitcast float %478 to i32, !dbg !39
  %480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %479), !dbg !39
  %481 = bitcast i32 %480 to float, !dbg !39
  %482 = tail call float @llvm.maxnum.f32(float %478, float %481), !dbg !39
  %483 = bitcast float %482 to i32, !dbg !39
  %484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %483), !dbg !39
  %485 = bitcast i32 %484 to float, !dbg !39
  %486 = tail call float @llvm.maxnum.f32(float %482, float %485), !dbg !39
  %487 = extractvalue [2 x <1 x float>] %118, 0, !dbg !39
  %488 = insertelement <1 x float> poison, float %476, i64 0, !dbg !39
  %489 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %488, <1 x float> %487), !dbg !39
  %490 = extractvalue [2 x <1 x float>] %118, 1, !dbg !39
  %491 = insertelement <1 x float> poison, float %486, i64 0, !dbg !39
  %492 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %491, <1 x float> %490), !dbg !39
  %493 = shufflevector <1 x float> %489, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %494 = shufflevector <1 x float> %492, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %495 = fsub <1 x float> %487, %489, !dbg !41
  %496 = extractelement <1 x float> %495, i64 0, !dbg !42
  %497 = tail call float @llvm.amdgcn.exp2.f32(float %496), !dbg !42
  %498 = insertelement <4 x float> poison, float %497, i64 0, !dbg !43
  %499 = shufflevector <4 x float> %498, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %500 = extractvalue [2 x <1 x float>] %119, 0, !dbg !44
  %501 = extractelement <1 x float> %500, i64 0, !dbg !44
  %502 = fmul float %501, %497, !dbg !44
  %503 = fsub <1 x float> %490, %492, !dbg !41
  %504 = extractelement <1 x float> %503, i64 0, !dbg !42
  %505 = tail call float @llvm.amdgcn.exp2.f32(float %504), !dbg !42
  %506 = insertelement <4 x float> poison, float %505, i64 0, !dbg !43
  %507 = shufflevector <4 x float> %506, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %508 = extractvalue [2 x <1 x float>] %119, 1, !dbg !44
  %509 = extractelement <1 x float> %508, i64 0, !dbg !44
  %510 = fmul float %509, %505, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1450 = or disjoint i64 %123, 32, !dbg !37
  %vdb.address1451 = or disjoint i64 %address.local924, %vdb.address1450, !dbg !37
  %vdb.address1452 = trunc nuw nsw i64 %vdb.address1451 to i32, !dbg !37
  %vdb.address1453 = shl nuw nsw i32 %vdb.address1452, 1, !dbg !37
  %vdb.address1461 = or disjoint i64 %address.local930, %vdb.address1450, !dbg !37
  %vdb.address1462 = trunc nuw nsw i64 %vdb.address1461 to i32, !dbg !37
  %vdb.address1463 = shl nuw nsw i32 %vdb.address1462, 1, !dbg !37
  %vdb.dma.base = add i32 %dma.buffer.lds1377, 8192, !dbg !45
  %511 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1453, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %vdb.dma.base1468 = add i32 %dma.buffer.lds1374, 8192, !dbg !45
  %512 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1468, i32 %vdb.address1463, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %address.local932 = shl nuw nsw i32 %address.thread501, 3, !dbg !45
  %address.local933 = and i32 %address.local932, 120, !dbg !45
  %address.local935 = and i32 %address.local578, 1536, !dbg !45
  %address.local936 = or disjoint i32 %address.local933, %address.local935, !dbg !45
  %address.local937 = zext nneg i32 %address.local936 to i64, !dbg !45
  %address.local938 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local937, !dbg !45
  %513 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local938, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %514 = bitcast <4 x i32> %513 to <8 x half>, !dbg !45
  %address.local939 = and i64 %address.local502, 60, !dbg !45
  %address.local940 = add nuw nsw i64 %address.local939, %address.local502, !dbg !45
  %address.local942.idx = shl nuw nsw i64 %address.local940, 4, !dbg !45
  %address.local942 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local942.idx, !dbg !45
  %address.local943 = lshr i64 %address.local558, 4, !dbg !45
  %address.local945.idx = mul nuw nsw i64 %address.local943, 768, !dbg !45
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local945.idx, !dbg !45
  %address.local947.idx = mul nsw i64 %address.local545, -64, !dbg !45
  %address.local947 = getelementptr i8, ptr addrspace(3) %address.local945, i64 %address.local947.idx, !dbg !45
  %address.local948 = getelementptr i8, ptr addrspace(3) %address.local947, i64 256, !dbg !45
  %515 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local948, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %516 = bitcast <4 x i32> %515 to <8 x half>, !dbg !45
  %address.local952 = getelementptr i8, ptr addrspace(3) %address.local947, i64 512, !dbg !45
  %517 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %518 = bitcast <4 x i32> %517 to <8 x half>, !dbg !45
  %address.local956 = getelementptr i8, ptr addrspace(3) %address.local947, i64 768, !dbg !45
  %519 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %520 = bitcast <4 x i32> %519 to <8 x half>, !dbg !45
  %address.local958 = getelementptr i8, ptr addrspace(3) %address.local938, i64 4096, !dbg !45
  %521 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local958, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %522 = bitcast <4 x i32> %521 to <8 x half>, !dbg !45
  %address.local962 = getelementptr i8, ptr addrspace(3) %address.local947, i64 4352, !dbg !45
  %523 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local962, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %524 = bitcast <4 x i32> %523 to <8 x half>, !dbg !45
  %address.local966 = getelementptr i8, ptr addrspace(3) %address.local947, i64 4608, !dbg !45
  %525 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %526 = bitcast <4 x i32> %525 to <8 x half>, !dbg !45
  %address.local970 = getelementptr i8, ptr addrspace(3) %address.local947, i64 4864, !dbg !45
  %527 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %528 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %529 = extractvalue [2 x <8 x float>] %120, 0, !dbg !43
  %530 = shufflevector <8 x float> %529, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %531 = fmul <4 x float> %530, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %532 = fsub <4 x float> %352, %493, !dbg !40
  %533 = extractelement <4 x float> %532, i64 0, !dbg !47
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !47
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !48
  %536 = extractelement <4 x float> %532, i64 1, !dbg !47
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !47
  %538 = insertelement <4 x float> %535, float %537, i64 1, !dbg !48
  %539 = extractelement <4 x float> %532, i64 2, !dbg !47
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %539), !dbg !47
  %541 = insertelement <4 x float> %538, float %540, i64 2, !dbg !48
  %542 = extractelement <4 x float> %532, i64 3, !dbg !47
  %543 = tail call float @llvm.amdgcn.exp2.f32(float %542), !dbg !47
  %544 = insertelement <4 x float> %541, float %543, i64 3, !dbg !48
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %544) #8, !dbg !48, !frisk.register_lifetimes !5
  %545 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !48
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %545) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %528, <4 x float> %531) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %547 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %548 = shufflevector <8 x float> %529, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %549 = fmul <4 x float> %548, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %550 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %547, <4 x float> %549) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %551 = extractvalue [2 x <8 x float>] %120, 1, !dbg !43
  %552 = shufflevector <8 x float> %551, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %553 = fmul <4 x float> %552, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %554 = fsub <4 x float> %410, %494, !dbg !40
  %555 = extractelement <4 x float> %554, i64 0, !dbg !47
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !47
  %557 = insertelement <4 x float> poison, float %556, i64 0, !dbg !48
  %558 = extractelement <4 x float> %554, i64 1, !dbg !47
  %559 = tail call float @llvm.amdgcn.exp2.f32(float %558), !dbg !47
  %560 = insertelement <4 x float> %557, float %559, i64 1, !dbg !48
  %561 = extractelement <4 x float> %554, i64 2, !dbg !47
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %561), !dbg !47
  %563 = insertelement <4 x float> %560, float %562, i64 2, !dbg !48
  %564 = extractelement <4 x float> %554, i64 3, !dbg !47
  %565 = tail call float @llvm.amdgcn.exp2.f32(float %564), !dbg !47
  %566 = insertelement <4 x float> %563, float %565, i64 3, !dbg !48
  %math.result391 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %566) #8, !dbg !48, !frisk.register_lifetimes !5
  %567 = fptrunc <4 x float> %math.result391 to <4 x half>, !dbg !48
  %math.packed392 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %567) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %568 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %528, <4 x float> %553) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %569 = shufflevector <8 x float> %551, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %570 = fmul <4 x float> %569, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %571 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %547, <4 x float> %570) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %572 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %573 = fsub <4 x float> %359, %493, !dbg !40
  %574 = extractelement <4 x float> %573, i64 0, !dbg !47
  %575 = tail call float @llvm.amdgcn.exp2.f32(float %574), !dbg !47
  %local.reduce92 = fadd float %534, %575, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %576 = insertelement <4 x float> poison, float %575, i64 0, !dbg !48
  %577 = extractelement <4 x float> %573, i64 1, !dbg !47
  %578 = tail call float @llvm.amdgcn.exp2.f32(float %577), !dbg !47
  %local.reduce93 = fadd float %537, %578, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1944 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %579 = insertelement <4 x float> %576, float %578, i64 1, !dbg !48
  %580 = extractelement <4 x float> %573, i64 2, !dbg !47
  %581 = tail call float @llvm.amdgcn.exp2.f32(float %580), !dbg !47
  %local.reduce94 = fadd float %540, %581, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1945 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %582 = insertelement <4 x float> %579, float %581, i64 2, !dbg !48
  %583 = extractelement <4 x float> %573, i64 3, !dbg !47
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %583), !dbg !47
  %local.reduce95 = fadd float %543, %584, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1946 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %582, float %584, i64 3, !dbg !48
  %math.result377 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %585) #8, !dbg !48, !frisk.register_lifetimes !5
  %586 = fptrunc <4 x float> %math.result377 to <4 x half>, !dbg !48
  %math.packed378 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %586) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %587 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %572, <4 x float> %546) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %588 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %588, <4 x float> %550) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %590 = fsub <4 x float> %416, %494, !dbg !40
  %591 = extractelement <4 x float> %590, i64 0, !dbg !47
  %592 = tail call float @llvm.amdgcn.exp2.f32(float %591), !dbg !47
  %local.reduce = fadd float %556, %592, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1947 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %593 = insertelement <4 x float> poison, float %592, i64 0, !dbg !48
  %594 = extractelement <4 x float> %590, i64 1, !dbg !47
  %595 = tail call float @llvm.amdgcn.exp2.f32(float %594), !dbg !47
  %local.reduce62 = fadd float %559, %595, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1948 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %596 = insertelement <4 x float> %593, float %595, i64 1, !dbg !48
  %597 = extractelement <4 x float> %590, i64 2, !dbg !47
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %597), !dbg !47
  %local.reduce63 = fadd float %562, %598, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1949 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %599 = insertelement <4 x float> %596, float %598, i64 2, !dbg !48
  %600 = extractelement <4 x float> %590, i64 3, !dbg !47
  %601 = tail call float @llvm.amdgcn.exp2.f32(float %600), !dbg !47
  %local.reduce64 = fadd float %565, %601, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1950 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %602 = insertelement <4 x float> %599, float %601, i64 3, !dbg !48
  %math.result393 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %602) #8, !dbg !48, !frisk.register_lifetimes !5
  %603 = fptrunc <4 x float> %math.result393 to <4 x half>, !dbg !48
  %math.packed394 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %603) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %572, <4 x float> %568) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %588, <4 x float> %571) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %606 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %607 = fsub <4 x float> %369, %493, !dbg !40
  %608 = extractelement <4 x float> %607, i64 0, !dbg !47
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %608), !dbg !47
  %local.reduce96 = fadd float %609, %k0.sum.anchor, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1951 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %610 = insertelement <4 x float> poison, float %609, i64 0, !dbg !48
  %611 = extractelement <4 x float> %607, i64 1, !dbg !47
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %611), !dbg !47
  %local.reduce97 = fadd float %612, %k0.sum.anchor1944, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1952 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %613 = insertelement <4 x float> %610, float %612, i64 1, !dbg !48
  %614 = extractelement <4 x float> %607, i64 2, !dbg !47
  %615 = tail call float @llvm.amdgcn.exp2.f32(float %614), !dbg !47
  %local.reduce98 = fadd float %615, %k0.sum.anchor1945, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1953 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %616 = insertelement <4 x float> %613, float %615, i64 2, !dbg !48
  %617 = extractelement <4 x float> %607, i64 3, !dbg !47
  %618 = tail call float @llvm.amdgcn.exp2.f32(float %617), !dbg !47
  %local.reduce99 = fadd float %618, %k0.sum.anchor1946, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1954 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %619 = insertelement <4 x float> %616, float %618, i64 3, !dbg !48
  %math.result379 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %619) #8, !dbg !48, !frisk.register_lifetimes !5
  %620 = fptrunc <4 x float> %math.result379 to <4 x half>, !dbg !48
  %math.packed380 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %620) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %621 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %606, <4 x float> %587) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %622 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %623 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %622, <4 x float> %589) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %624 = fsub <4 x float> %422, %494, !dbg !40
  %625 = extractelement <4 x float> %624, i64 0, !dbg !47
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %625), !dbg !47
  %local.reduce65 = fadd float %626, %k0.sum.anchor1947, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1955 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %627 = insertelement <4 x float> poison, float %626, i64 0, !dbg !48
  %628 = extractelement <4 x float> %624, i64 1, !dbg !47
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !47
  %local.reduce66 = fadd float %629, %k0.sum.anchor1948, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1956 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %630 = insertelement <4 x float> %627, float %629, i64 1, !dbg !48
  %631 = extractelement <4 x float> %624, i64 2, !dbg !47
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %631), !dbg !47
  %local.reduce67 = fadd float %632, %k0.sum.anchor1949, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1957 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %630, float %632, i64 2, !dbg !48
  %634 = extractelement <4 x float> %624, i64 3, !dbg !47
  %635 = tail call float @llvm.amdgcn.exp2.f32(float %634), !dbg !47
  %local.reduce68 = fadd float %635, %k0.sum.anchor1950, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1958 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %636 = insertelement <4 x float> %633, float %635, i64 3, !dbg !48
  %math.result395 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %636) #8, !dbg !48, !frisk.register_lifetimes !5
  %637 = fptrunc <4 x float> %math.result395 to <4 x half>, !dbg !48
  %math.packed396 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %637) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %606, <4 x float> %604) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %639 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %622, <4 x float> %605) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %640 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %641 = fsub <4 x float> %379, %493, !dbg !40
  %642 = extractelement <4 x float> %641, i64 0, !dbg !47
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %642), !dbg !47
  %local.reduce100 = fadd float %643, %k0.sum.anchor1951, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1959 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %644 = insertelement <4 x float> poison, float %643, i64 0, !dbg !48
  %645 = extractelement <4 x float> %641, i64 1, !dbg !47
  %646 = tail call float @llvm.amdgcn.exp2.f32(float %645), !dbg !47
  %local.reduce101 = fadd float %646, %k0.sum.anchor1952, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1960 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %647 = insertelement <4 x float> %644, float %646, i64 1, !dbg !48
  %648 = extractelement <4 x float> %641, i64 2, !dbg !47
  %649 = tail call float @llvm.amdgcn.exp2.f32(float %648), !dbg !47
  %local.reduce102 = fadd float %649, %k0.sum.anchor1953, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1961 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %650 = insertelement <4 x float> %647, float %649, i64 2, !dbg !48
  %651 = extractelement <4 x float> %641, i64 3, !dbg !47
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !47
  %local.reduce103 = fadd float %652, %k0.sum.anchor1954, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1962 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %653 = insertelement <4 x float> %650, float %652, i64 3, !dbg !48
  %math.result381 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %653) #8, !dbg !48, !frisk.register_lifetimes !5
  %654 = fptrunc <4 x float> %math.result381 to <4 x half>, !dbg !48
  %math.packed382 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %654) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %655 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %640, <4 x float> %621) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %656 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %656, <4 x float> %623) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %658 = fsub <4 x float> %428, %494, !dbg !40
  %659 = extractelement <4 x float> %658, i64 0, !dbg !47
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %659), !dbg !47
  %local.reduce69 = fadd float %660, %k0.sum.anchor1955, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1963 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %661 = insertelement <4 x float> poison, float %660, i64 0, !dbg !48
  %662 = extractelement <4 x float> %658, i64 1, !dbg !47
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %662), !dbg !47
  %local.reduce70 = fadd float %663, %k0.sum.anchor1956, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1964 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %664 = insertelement <4 x float> %661, float %663, i64 1, !dbg !48
  %665 = extractelement <4 x float> %658, i64 2, !dbg !47
  %666 = tail call float @llvm.amdgcn.exp2.f32(float %665), !dbg !47
  %local.reduce71 = fadd float %666, %k0.sum.anchor1957, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1965 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %667 = insertelement <4 x float> %664, float %666, i64 2, !dbg !48
  %668 = extractelement <4 x float> %658, i64 3, !dbg !47
  %669 = tail call float @llvm.amdgcn.exp2.f32(float %668), !dbg !47
  %local.reduce72 = fadd float %669, %k0.sum.anchor1958, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1966 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %670 = insertelement <4 x float> %667, float %669, i64 3, !dbg !48
  %math.result397 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %670) #8, !dbg !48, !frisk.register_lifetimes !5
  %671 = fptrunc <4 x float> %math.result397 to <4 x half>, !dbg !48
  %math.packed398 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %671) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %672 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %640, <4 x float> %638) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %673 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %656, <4 x float> %639) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %674 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %675 = fsub <4 x float> %385, %493, !dbg !40
  %676 = extractelement <4 x float> %675, i64 0, !dbg !47
  %677 = tail call float @llvm.amdgcn.exp2.f32(float %676), !dbg !47
  %local.reduce104 = fadd float %677, %k0.sum.anchor1959, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1967 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %678 = insertelement <4 x float> poison, float %677, i64 0, !dbg !48
  %679 = extractelement <4 x float> %675, i64 1, !dbg !47
  %680 = tail call float @llvm.amdgcn.exp2.f32(float %679), !dbg !47
  %local.reduce105 = fadd float %680, %k0.sum.anchor1960, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1968 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %681 = insertelement <4 x float> %678, float %680, i64 1, !dbg !48
  %682 = extractelement <4 x float> %675, i64 2, !dbg !47
  %683 = tail call float @llvm.amdgcn.exp2.f32(float %682), !dbg !47
  %local.reduce106 = fadd float %683, %k0.sum.anchor1961, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1969 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %684 = insertelement <4 x float> %681, float %683, i64 2, !dbg !48
  %685 = extractelement <4 x float> %675, i64 3, !dbg !47
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %685), !dbg !47
  %local.reduce107 = fadd float %686, %k0.sum.anchor1962, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1970 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %687 = insertelement <4 x float> %684, float %686, i64 3, !dbg !48
  %math.result383 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %687) #8, !dbg !48, !frisk.register_lifetimes !5
  %688 = fptrunc <4 x float> %math.result383 to <4 x half>, !dbg !48
  %math.packed384 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %688) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %689 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %674, <4 x float> %655) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %690 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %691 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %690, <4 x float> %657) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %692 = fsub <4 x float> %434, %494, !dbg !40
  %693 = extractelement <4 x float> %692, i64 0, !dbg !47
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %693), !dbg !47
  %local.reduce73 = fadd float %694, %k0.sum.anchor1963, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1971 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %695 = insertelement <4 x float> poison, float %694, i64 0, !dbg !48
  %696 = extractelement <4 x float> %692, i64 1, !dbg !47
  %697 = tail call float @llvm.amdgcn.exp2.f32(float %696), !dbg !47
  %local.reduce74 = fadd float %697, %k0.sum.anchor1964, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1972 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %698 = insertelement <4 x float> %695, float %697, i64 1, !dbg !48
  %699 = extractelement <4 x float> %692, i64 2, !dbg !47
  %700 = tail call float @llvm.amdgcn.exp2.f32(float %699), !dbg !47
  %local.reduce75 = fadd float %700, %k0.sum.anchor1965, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1973 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %701 = insertelement <4 x float> %698, float %700, i64 2, !dbg !48
  %702 = extractelement <4 x float> %692, i64 3, !dbg !47
  %703 = tail call float @llvm.amdgcn.exp2.f32(float %702), !dbg !47
  %local.reduce76 = fadd float %703, %k0.sum.anchor1966, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1974 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %704 = insertelement <4 x float> %701, float %703, i64 3, !dbg !48
  %math.result399 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %704) #8, !dbg !48, !frisk.register_lifetimes !5
  %705 = fptrunc <4 x float> %math.result399 to <4 x half>, !dbg !48
  %math.packed400 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %705) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %706 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %674, <4 x float> %672) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %707 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %690, <4 x float> %673) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %708 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %709 = fsub <4 x float> %391, %493, !dbg !40
  %710 = extractelement <4 x float> %709, i64 0, !dbg !47
  %711 = tail call float @llvm.amdgcn.exp2.f32(float %710), !dbg !47
  %local.reduce108 = fadd float %711, %k0.sum.anchor1967, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1975 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %712 = insertelement <4 x float> poison, float %711, i64 0, !dbg !48
  %713 = extractelement <4 x float> %709, i64 1, !dbg !47
  %714 = tail call float @llvm.amdgcn.exp2.f32(float %713), !dbg !47
  %local.reduce109 = fadd float %714, %k0.sum.anchor1968, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1976 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %715 = insertelement <4 x float> %712, float %714, i64 1, !dbg !48
  %716 = extractelement <4 x float> %709, i64 2, !dbg !47
  %717 = tail call float @llvm.amdgcn.exp2.f32(float %716), !dbg !47
  %local.reduce110 = fadd float %717, %k0.sum.anchor1969, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1977 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %718 = insertelement <4 x float> %715, float %717, i64 2, !dbg !48
  %719 = extractelement <4 x float> %709, i64 3, !dbg !47
  %720 = tail call float @llvm.amdgcn.exp2.f32(float %719), !dbg !47
  %local.reduce111 = fadd float %720, %k0.sum.anchor1970, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1978 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %721 = insertelement <4 x float> %718, float %720, i64 3, !dbg !48
  %math.result385 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %721) #8, !dbg !48, !frisk.register_lifetimes !5
  %722 = fptrunc <4 x float> %math.result385 to <4 x half>, !dbg !48
  %math.packed386 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %722) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %723 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %708, <4 x float> %689) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %724 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %724, <4 x float> %691) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %726 = fsub <4 x float> %440, %494, !dbg !40
  %727 = extractelement <4 x float> %726, i64 0, !dbg !47
  %728 = tail call float @llvm.amdgcn.exp2.f32(float %727), !dbg !47
  %local.reduce77 = fadd float %728, %k0.sum.anchor1971, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1979 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %729 = insertelement <4 x float> poison, float %728, i64 0, !dbg !48
  %730 = extractelement <4 x float> %726, i64 1, !dbg !47
  %731 = tail call float @llvm.amdgcn.exp2.f32(float %730), !dbg !47
  %local.reduce78 = fadd float %731, %k0.sum.anchor1972, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1980 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %732 = insertelement <4 x float> %729, float %731, i64 1, !dbg !48
  %733 = extractelement <4 x float> %726, i64 2, !dbg !47
  %734 = tail call float @llvm.amdgcn.exp2.f32(float %733), !dbg !47
  %local.reduce79 = fadd float %734, %k0.sum.anchor1973, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1981 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %735 = insertelement <4 x float> %732, float %734, i64 2, !dbg !48
  %736 = extractelement <4 x float> %726, i64 3, !dbg !47
  %737 = tail call float @llvm.amdgcn.exp2.f32(float %736), !dbg !47
  %local.reduce80 = fadd float %737, %k0.sum.anchor1974, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1982 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %738 = insertelement <4 x float> %735, float %737, i64 3, !dbg !48
  %math.result401 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %738) #8, !dbg !48, !frisk.register_lifetimes !5
  %739 = fptrunc <4 x float> %math.result401 to <4 x half>, !dbg !48
  %math.packed402 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %739) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %740 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %708, <4 x float> %706) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %724, <4 x float> %707) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %742 = shufflevector <8 x half> %526, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %743 = fsub <4 x float> %397, %493, !dbg !40
  %744 = extractelement <4 x float> %743, i64 0, !dbg !47
  %745 = tail call float @llvm.amdgcn.exp2.f32(float %744), !dbg !47
  %local.reduce112 = fadd float %745, %k0.sum.anchor1975, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1983 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !48
  %747 = extractelement <4 x float> %743, i64 1, !dbg !47
  %748 = tail call float @llvm.amdgcn.exp2.f32(float %747), !dbg !47
  %local.reduce113 = fadd float %748, %k0.sum.anchor1976, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1984 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %749 = insertelement <4 x float> %746, float %748, i64 1, !dbg !48
  %750 = extractelement <4 x float> %743, i64 2, !dbg !47
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %750), !dbg !47
  %local.reduce114 = fadd float %751, %k0.sum.anchor1977, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1985 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %752 = insertelement <4 x float> %749, float %751, i64 2, !dbg !48
  %753 = extractelement <4 x float> %743, i64 3, !dbg !47
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !47
  %local.reduce115 = fadd float %754, %k0.sum.anchor1978, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1986 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %755 = insertelement <4 x float> %752, float %754, i64 3, !dbg !48
  %math.result387 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #8, !dbg !48, !frisk.register_lifetimes !5
  %756 = fptrunc <4 x float> %math.result387 to <4 x half>, !dbg !48
  %math.packed388 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %757 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %742, <4 x float> %723) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %758 = shufflevector <8 x half> %526, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %759 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %758, <4 x float> %725) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %760 = fsub <4 x float> %446, %494, !dbg !40
  %761 = extractelement <4 x float> %760, i64 0, !dbg !47
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %761), !dbg !47
  %local.reduce81 = fadd float %762, %k0.sum.anchor1979, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1987 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !48
  %764 = extractelement <4 x float> %760, i64 1, !dbg !47
  %765 = tail call float @llvm.amdgcn.exp2.f32(float %764), !dbg !47
  %local.reduce82 = fadd float %765, %k0.sum.anchor1980, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1988 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %766 = insertelement <4 x float> %763, float %765, i64 1, !dbg !48
  %767 = extractelement <4 x float> %760, i64 2, !dbg !47
  %768 = tail call float @llvm.amdgcn.exp2.f32(float %767), !dbg !47
  %local.reduce83 = fadd float %768, %k0.sum.anchor1981, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1989 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %769 = insertelement <4 x float> %766, float %768, i64 2, !dbg !48
  %770 = extractelement <4 x float> %760, i64 3, !dbg !47
  %771 = tail call float @llvm.amdgcn.exp2.f32(float %770), !dbg !47
  %local.reduce84 = fadd float %771, %k0.sum.anchor1982, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1990 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %772 = insertelement <4 x float> %769, float %771, i64 3, !dbg !48
  %math.result403 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %772) #8, !dbg !48, !frisk.register_lifetimes !5
  %773 = fptrunc <4 x float> %math.result403 to <4 x half>, !dbg !48
  %math.packed404 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %773) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %742, <4 x float> %740) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %758, <4 x float> %741) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %776 = bitcast <4 x i32> %527 to <8 x half>, !dbg !45
  %777 = shufflevector <8 x half> %776, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %778 = fsub <4 x float> %403, %493, !dbg !40
  %779 = extractelement <4 x float> %778, i64 0, !dbg !47
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %779), !dbg !47
  %local.reduce116 = fadd float %780, %k0.sum.anchor1983, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1991 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %781 = insertelement <4 x float> poison, float %780, i64 0, !dbg !48
  %782 = extractelement <4 x float> %778, i64 1, !dbg !47
  %783 = tail call float @llvm.amdgcn.exp2.f32(float %782), !dbg !47
  %local.reduce117 = fadd float %783, %k0.sum.anchor1984, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1992 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce120 = fadd float %k0.sum.anchor1991, %k0.sum.anchor1992, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1993 = tail call float asm sideeffect "", "=v,0"(float %local.reduce120) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %784 = insertelement <4 x float> %781, float %783, i64 1, !dbg !48
  %785 = extractelement <4 x float> %778, i64 2, !dbg !47
  %786 = tail call float @llvm.amdgcn.exp2.f32(float %785), !dbg !47
  %local.reduce118 = fadd float %786, %k0.sum.anchor1985, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1994 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %787 = insertelement <4 x float> %784, float %786, i64 2, !dbg !48
  %788 = extractelement <4 x float> %778, i64 3, !dbg !47
  %789 = tail call float @llvm.amdgcn.exp2.f32(float %788), !dbg !47
  %local.reduce119 = fadd float %789, %k0.sum.anchor1986, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1995 = tail call float asm sideeffect "", "=v,0"(float %local.reduce119) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce121 = fadd float %k0.sum.anchor1994, %k0.sum.anchor1995, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1996 = tail call float asm sideeffect "", "=v,0"(float %local.reduce121) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %790 = fadd float %k0.sum.anchor1993, %k0.sum.anchor1996, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1997 = tail call float asm sideeffect "", "=v,0"(float %790) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %791 = fadd float %502, %k0.sum.anchor1997, !dbg !49
  %k0.sum.anchor1998 = tail call float asm sideeffect "", "=v,0"(float %791) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %792 = insertelement <1 x float> poison, float %k0.sum.anchor1998, i64 0, !dbg !11
  %793 = insertvalue [2 x <1 x float>] poison, <1 x float> %792, 0, !dbg !11
  %794 = insertelement <4 x float> %787, float %789, i64 3, !dbg !48
  %math.result389 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #8, !dbg !48, !frisk.register_lifetimes !5
  %795 = fptrunc <4 x float> %math.result389 to <4 x half>, !dbg !48
  %math.packed390 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %777, <4 x float> %757) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %797 = shufflevector <8 x half> %776, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %797, <4 x float> %759) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %799 = fsub <4 x float> %452, %494, !dbg !40
  %800 = extractelement <4 x float> %799, i64 0, !dbg !47
  %801 = tail call float @llvm.amdgcn.exp2.f32(float %800), !dbg !47
  %local.reduce85 = fadd float %801, %k0.sum.anchor1987, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1999 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %802 = insertelement <4 x float> poison, float %801, i64 0, !dbg !48
  %803 = extractelement <4 x float> %799, i64 1, !dbg !47
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !47
  %local.reduce86 = fadd float %804, %k0.sum.anchor1988, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2000 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce89 = fadd float %k0.sum.anchor1999, %k0.sum.anchor2000, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2001 = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %805 = insertelement <4 x float> %802, float %804, i64 1, !dbg !48
  %806 = extractelement <4 x float> %799, i64 2, !dbg !47
  %807 = tail call float @llvm.amdgcn.exp2.f32(float %806), !dbg !47
  %local.reduce87 = fadd float %807, %k0.sum.anchor1989, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2002 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %808 = insertelement <4 x float> %805, float %807, i64 2, !dbg !48
  %809 = extractelement <4 x float> %799, i64 3, !dbg !47
  %810 = tail call float @llvm.amdgcn.exp2.f32(float %809), !dbg !47
  %local.reduce88 = fadd float %810, %k0.sum.anchor1990, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2003 = tail call float asm sideeffect "", "=v,0"(float %local.reduce88) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce90 = fadd float %k0.sum.anchor2002, %k0.sum.anchor2003, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2004 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %811 = fadd float %k0.sum.anchor2001, %k0.sum.anchor2004, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor2005 = tail call float asm sideeffect "", "=v,0"(float %811) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %812 = fadd float %510, %k0.sum.anchor2005, !dbg !49
  %k0.sum.anchor2006 = tail call float asm sideeffect "", "=v,0"(float %812) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %813 = insertelement <1 x float> poison, float %k0.sum.anchor2006, i64 0, !dbg !11
  %814 = insertvalue [2 x <1 x float>] %793, <1 x float> %813, 1, !dbg !11
  %815 = insertelement <4 x float> %808, float %810, i64 3, !dbg !48
  %math.result405 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %815) #8, !dbg !48, !frisk.register_lifetimes !5
  %816 = fptrunc <4 x float> %math.result405 to <4 x half>, !dbg !48
  %math.packed406 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %816) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %777, <4 x float> %774) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %797, <4 x float> %775) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1478 = or disjoint i64 %123, 64, !dbg !37
  %vdb.address1479 = or disjoint i64 %address.local924, %vdb.address1478, !dbg !37
  %vdb.address1480 = trunc nuw nsw i64 %vdb.address1479 to i32, !dbg !37
  %vdb.address1481 = shl nuw nsw i32 %vdb.address1480, 1, !dbg !37
  %vdb.address1489 = or disjoint i64 %address.local930, %vdb.address1478, !dbg !37
  %vdb.address1490 = trunc nuw nsw i64 %vdb.address1489 to i32, !dbg !37
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !37
  %819 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %vdb.address1481, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %820 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local938, i64 8192, !dbg !45
  %821 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %822 = bitcast <4 x i32> %821 to <8 x half>, !dbg !45
  %vdb.read1469 = getelementptr i8, ptr addrspace(3) %address.local947, i64 8448, !dbg !45
  %823 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1469, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !45
  %vdb.read1470 = getelementptr i8, ptr addrspace(3) %address.local947, i64 8704, !dbg !45
  %825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1470, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !45
  %vdb.read1471 = getelementptr i8, ptr addrspace(3) %address.local947, i64 8960, !dbg !45
  %827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1471, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %828 = bitcast <4 x i32> %827 to <8 x half>, !dbg !45
  %vdb.read1472 = getelementptr i8, ptr addrspace(3) %address.local938, i64 12288, !dbg !45
  %829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1472, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %830 = bitcast <4 x i32> %829 to <8 x half>, !dbg !45
  %vdb.read1473 = getelementptr i8, ptr addrspace(3) %address.local947, i64 12544, !dbg !45
  %831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1473, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %832 = bitcast <4 x i32> %831 to <8 x half>, !dbg !45
  %vdb.read1474 = getelementptr i8, ptr addrspace(3) %address.local947, i64 12800, !dbg !45
  %833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1474, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !45
  %vdb.read1475 = getelementptr i8, ptr addrspace(3) %address.local947, i64 13056, !dbg !45
  %835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1475, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %836 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %837 = extractvalue [2 x <8 x float>] %117, 0, !dbg !43
  %838 = shufflevector <8 x float> %837, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %839 = fmul <4 x float> %838, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %836, <4 x float> %839) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %841 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %842 = shufflevector <8 x float> %837, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %843 = fmul <4 x float> %842, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %841, <4 x float> %843) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %845 = extractvalue [2 x <8 x float>] %117, 1, !dbg !43
  %846 = shufflevector <8 x float> %845, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %847 = fmul <4 x float> %846, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %836, <4 x float> %847) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %849 = shufflevector <8 x float> %845, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %850 = fmul <4 x float> %849, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %841, <4 x float> %850) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %852 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %852, <4 x float> %840) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %854 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %854, <4 x float> %844) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %852, <4 x float> %848) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %854, <4 x float> %851) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %858 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %858, <4 x float> %853) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %860 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %860, <4 x float> %855) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %858, <4 x float> %856) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %860, <4 x float> %857) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %864 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %864, <4 x float> %859) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %866 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %866, <4 x float> %861) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %864, <4 x float> %862) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %866, <4 x float> %863) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %870 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %870, <4 x float> %865) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %872 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %872, <4 x float> %867) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %870, <4 x float> %868) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %872, <4 x float> %869) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %876 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %876, <4 x float> %871) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %878 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %878, <4 x float> %873) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %876, <4 x float> %874) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %878, <4 x float> %875) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %882 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %882, <4 x float> %877) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %884 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %884, <4 x float> %879) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %882, <4 x float> %880) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %884, <4 x float> %881) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %888 = bitcast <4 x i32> %835 to <8 x half>, !dbg !45
  %889 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %889, <4 x float> %883) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %891 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %891, <4 x float> %885) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %889, <4 x float> %886) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %891, <4 x float> %887) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1498 = or disjoint i64 %123, 96, !dbg !37
  %vdb.address1499 = or disjoint i64 %address.local924, %vdb.address1498, !dbg !37
  %vdb.address1500 = trunc nuw nsw i64 %vdb.address1499 to i32, !dbg !37
  %vdb.address1501 = shl nuw nsw i32 %vdb.address1500, 1, !dbg !37
  %vdb.address1509 = or disjoint i64 %address.local930, %vdb.address1498, !dbg !37
  %vdb.address1510 = trunc nuw nsw i64 %vdb.address1509 to i32, !dbg !37
  %vdb.address1511 = shl nuw nsw i32 %vdb.address1510, 1, !dbg !37
  %895 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1501, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %896 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1468, i32 %vdb.address1511, <4 x i32> %dma.buffer.rsrc1352) #9, !dbg !37
  %897 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local938, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %898 = bitcast <4 x i32> %897 to <8 x half>, !dbg !45
  %899 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local948, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %900 = bitcast <4 x i32> %899 to <8 x half>, !dbg !45
  %901 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local952, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %902 = bitcast <4 x i32> %901 to <8 x half>, !dbg !45
  %903 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local956, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %904 = bitcast <4 x i32> %903 to <8 x half>, !dbg !45
  %905 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local958, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %906 = bitcast <4 x i32> %905 to <8 x half>, !dbg !45
  %907 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local962, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %908 = bitcast <4 x i32> %907 to <8 x half>, !dbg !45
  %909 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local966, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %910 = bitcast <4 x i32> %909 to <8 x half>, !dbg !45
  %911 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local970, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %912 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %913 = extractvalue [2 x <8 x float>] %116, 0, !dbg !43
  %914 = shufflevector <8 x float> %913, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %915 = fmul <4 x float> %914, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %912, <4 x float> %915) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %917 = shufflevector <8 x half> %898, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %918 = shufflevector <8 x float> %913, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %919 = fmul <4 x float> %918, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %917, <4 x float> %919) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %921 = extractvalue [2 x <8 x float>] %116, 1, !dbg !43
  %922 = shufflevector <8 x float> %921, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %923 = fmul <4 x float> %922, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %912, <4 x float> %923) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %925 = shufflevector <8 x float> %921, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %926 = fmul <4 x float> %925, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %917, <4 x float> %926) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %928 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %928, <4 x float> %916) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %930 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %930, <4 x float> %920) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %928, <4 x float> %924) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %930, <4 x float> %927) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %934 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %934, <4 x float> %929) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %936 = shufflevector <8 x half> %902, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %936, <4 x float> %931) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %934, <4 x float> %932) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %936, <4 x float> %933) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %940 = shufflevector <8 x half> %904, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %941 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %940, <4 x float> %935) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %942 = shufflevector <8 x half> %904, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %942, <4 x float> %937) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %940, <4 x float> %938) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %945 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %942, <4 x float> %939) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %946 = shufflevector <8 x half> %906, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %947 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %946, <4 x float> %941) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %948 = shufflevector <8 x half> %906, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %948, <4 x float> %943) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %946, <4 x float> %944) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %948, <4 x float> %945) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %952 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %952, <4 x float> %947) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %954 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %954, <4 x float> %949) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %952, <4 x float> %950) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %954, <4 x float> %951) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %958 = shufflevector <8 x half> %910, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %958, <4 x float> %953) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %960 = shufflevector <8 x half> %910, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %960, <4 x float> %955) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %958, <4 x float> %956) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %960, <4 x float> %957) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %964 = bitcast <4 x i32> %911 to <8 x half>, !dbg !45
  %965 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %965, <4 x float> %959) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %967 = shufflevector <8 x half> %964, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %967, <4 x float> %961) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %965, <4 x float> %962) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %967, <4 x float> %963) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %971 = add nuw nsw i64 %121, 1, !dbg !11
  %k0.has.next.not = icmp eq i64 %121, 31, !dbg !45
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !45

k0.issue:                                         ; preds = %114
  %k0.next. = shl nuw nsw i64 %971, 14, !dbg !12
  %k0.next.2018 = add nuw nsw i64 %k0.next., %k0.prime.2007, !dbg !12
  %k0.next.kdma.source1536 = or disjoint i64 %k0.next.2018, %k0.prime.kdma.source1535, !dbg !12
  %k0.next.kdma.source1537 = trunc nuw nsw i64 %k0.next.kdma.source1536 to i32, !dbg !12
  %k0.next.kdma.source1538 = shl nuw nsw i32 %k0.next.kdma.source1537, 1, !dbg !12
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1538, %kdma.intra, !dbg !12
  %k0.next.2019 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1377, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1550 = or disjoint i64 %k0.next.2018, %k0.prime.kdma.source1549, !dbg !12
  %k0.next.kdma.source1551 = trunc nuw nsw i64 %k0.next.kdma.source1550 to i32, !dbg !12
  %k0.next.kdma.source1552 = shl nuw nsw i32 %k0.next.kdma.source1551, 1, !dbg !12
  %k0.next.kdma.offset1553 = or disjoint i32 %k0.next.kdma.source1552, %kdma.intra.hi, !dbg !12
  %k0.next.2020 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1554, i32 %k0.next.kdma.offset1553, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1567 = or disjoint i64 %k0.next.2018, %k0.prime.kdma.source1566, !dbg !12
  %k0.next.kdma.source1568 = trunc nuw nsw i64 %k0.next.kdma.source1567 to i32, !dbg !12
  %k0.next.kdma.source1569 = shl nuw nsw i32 %k0.next.kdma.source1568, 1, !dbg !12
  %k0.next.kdma.offset1570 = or disjoint i32 %k0.next.kdma.source1569, %kdma.intra, !dbg !12
  %k0.next.2021 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %k0.next.kdma.offset1570, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1584 = or disjoint i64 %k0.next.2018, %k0.prime.kdma.source1583, !dbg !12
  %k0.next.kdma.source1585 = trunc nuw nsw i64 %k0.next.kdma.source1584 to i32, !dbg !12
  %k0.next.kdma.source1586 = shl nuw nsw i32 %k0.next.kdma.source1585, 1, !dbg !12
  %k0.next.kdma.offset1587 = or disjoint i32 %k0.next.kdma.source1586, %kdma.intra.hi, !dbg !12
  %k0.next.2022 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1588, i32 %k0.next.kdma.offset1587, <4 x i32> %k0.prime.dma.buffer.rsrc1376) #9, !dbg !12, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %114
  %972 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %973 = bitcast <4 x i32> %972 to <8 x half>, !dbg !45
  %974 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1469, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %975 = bitcast <4 x i32> %974 to <8 x half>, !dbg !45
  %976 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1470, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %977 = bitcast <4 x i32> %976 to <8 x half>, !dbg !45
  %978 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1471, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %979 = bitcast <4 x i32> %978 to <8 x half>, !dbg !45
  %980 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1472, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %981 = bitcast <4 x i32> %980 to <8 x half>, !dbg !45
  %982 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1473, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %983 = bitcast <4 x i32> %982 to <8 x half>, !dbg !45
  %984 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1474, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %985 = bitcast <4 x i32> %984 to <8 x half>, !dbg !45
  %986 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1475, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %987 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %988 = extractvalue [2 x <8 x float>] %115, 0, !dbg !43
  %989 = shufflevector <8 x float> %988, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %990 = fmul <4 x float> %989, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %987, <4 x float> %990) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %992 = shufflevector <8 x half> %973, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %993 = shufflevector <8 x float> %988, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %994 = fmul <4 x float> %993, %499, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %992, <4 x float> %994) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %996 = extractvalue [2 x <8 x float>] %115, 1, !dbg !43
  %997 = shufflevector <8 x float> %996, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %998 = fmul <4 x float> %997, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %987, <4 x float> %998) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %1000 = shufflevector <8 x float> %996, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %1001 = fmul <4 x float> %1000, %507, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed392, <4 x half> %992, <4 x float> %1001) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1003 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %1003, <4 x float> %991) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1005 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed378, <4 x half> %1005, <4 x float> %995) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %1003, <4 x float> %999) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed394, <4 x half> %1005, <4 x float> %1002) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1009 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %1009, <4 x float> %1004) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1011 = shufflevector <8 x half> %977, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed380, <4 x half> %1011, <4 x float> %1006) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %1009, <4 x float> %1007) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1014 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed396, <4 x half> %1011, <4 x float> %1008) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1015 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1016 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %1015, <4 x float> %1010) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1017 = shufflevector <8 x half> %979, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1018 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed382, <4 x half> %1017, <4 x float> %1012) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1019 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1015, <4 x float> %1013) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1020 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed398, <4 x half> %1017, <4 x float> %1014) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1021 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1022 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %1021, <4 x float> %1016) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1023 = shufflevector <8 x half> %981, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1024 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed384, <4 x half> %1023, <4 x float> %1018) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1025 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1021, <4 x float> %1019) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1026 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed400, <4 x half> %1023, <4 x float> %1020) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1027 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %1027, <4 x float> %1022) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1029 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed386, <4 x half> %1029, <4 x float> %1024) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1031 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1027, <4 x float> %1025) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed402, <4 x half> %1029, <4 x float> %1026) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1033 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1034 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1033, <4 x float> %1028) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1035 = shufflevector <8 x half> %985, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed388, <4 x half> %1035, <4 x float> %1030) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1037 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1033, <4 x float> %1031) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1038 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed404, <4 x half> %1035, <4 x float> %1032) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1039 = bitcast <4 x i32> %986 to <8 x half>, !dbg !45
  %1040 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1041 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1040, <4 x float> %1034) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1042 = shufflevector <8 x half> %1039, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1043 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed390, <4 x half> %1042, <4 x float> %1036) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1044 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1040, <4 x float> %1037) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1045 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed406, <4 x half> %1042, <4 x float> %1038) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1046 = shufflevector <4 x float> %796, <4 x float> %798, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1047 = insertvalue [2 x <8 x float>] poison, <8 x float> %1046, 0, !dbg !11
  %1048 = shufflevector <4 x float> %817, <4 x float> %818, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1049 = insertvalue [2 x <8 x float>] %1047, <8 x float> %1048, 1, !dbg !11
  %1050 = insertvalue [2 x <1 x float>] poison, <1 x float> %489, 0, !dbg !11
  %1051 = insertvalue [2 x <1 x float>] %1050, <1 x float> %492, 1, !dbg !11
  %1052 = shufflevector <4 x float> %890, <4 x float> %892, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1053 = insertvalue [2 x <8 x float>] poison, <8 x float> %1052, 0, !dbg !11
  %1054 = shufflevector <4 x float> %893, <4 x float> %894, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1055 = insertvalue [2 x <8 x float>] %1053, <8 x float> %1054, 1, !dbg !11
  %1056 = shufflevector <4 x float> %966, <4 x float> %968, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1057 = insertvalue [2 x <8 x float>] poison, <8 x float> %1056, 0, !dbg !11
  %1058 = shufflevector <4 x float> %969, <4 x float> %970, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1059 = insertvalue [2 x <8 x float>] %1057, <8 x float> %1058, 1, !dbg !11
  %1060 = shufflevector <4 x float> %1041, <4 x float> %1043, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1061 = insertvalue [2 x <8 x float>] poison, <8 x float> %1060, 0, !dbg !11
  %1062 = shufflevector <4 x float> %1044, <4 x float> %1045, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1063 = insertvalue [2 x <8 x float>] %1061, <8 x float> %1062, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1064 = icmp samesign ult i64 %121, 31, !dbg !11
  br i1 %1064, label %114, label %1065, !dbg !11

1065:                                             ; preds = %k0.join
  %1066 = shl nuw nsw i32 %458, 2, !dbg !11
  %1067 = xor i32 %1066, 64, !dbg !11
  %1068 = bitcast float %k0.sum.anchor2006 to i32, !dbg !11
  %lane.sum.shuffle191 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1067, i32 %1068), !dbg !11
  %1069 = bitcast i32 %lane.sum.shuffle191 to float, !dbg !11
  %lane.sum.final192 = fadd float %k0.sum.anchor2006, %1069, !dbg !11
  %1070 = xor i32 %1066, 128, !dbg !11
  %1071 = bitcast float %lane.sum.final192 to i32, !dbg !11
  %lane.sum.shuffle193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1070, i32 %1071), !dbg !11
  %1072 = bitcast i32 %lane.sum.shuffle193 to float, !dbg !11
  %lane.sum.final194 = fadd float %lane.sum.final192, %1072, !dbg !11
  %1073 = insertelement <1 x float> poison, float %lane.sum.final194, i64 0, !dbg !11
  %1074 = bitcast float %k0.sum.anchor1998 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1067, i32 %1074), !dbg !11
  %1075 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %k0.sum.anchor1998, %1075, !dbg !11
  %1076 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle185 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1070, i32 %1076), !dbg !11
  %1077 = bitcast i32 %lane.sum.shuffle185 to float, !dbg !11
  %lane.sum.final186 = fadd float %lane.sum.final, %1077, !dbg !11
  %1078 = insertelement <1 x float> poison, float %lane.sum.final186, i64 0, !dbg !11
  %1079 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1079, 0, !dbg !11
  %1080 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1080, 0, !dbg !11
  %1081 = extractelement <4 x float> %1041, i64 0, !dbg !11
  %1082 = extractelement <4 x float> %1041, i64 1, !dbg !11
  %1083 = extractelement <4 x float> %1041, i64 2, !dbg !11
  %1084 = extractelement <4 x float> %1041, i64 3, !dbg !11
  %1085 = select i1 %.not, float %1082, float %1081, !dbg !11
  %1086 = bitcast float %1085 to i32, !dbg !11
  %1087 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1086), !dbg !11
  %1088 = bitcast i32 %1087 to float, !dbg !11
  %1089 = select i1 %.not, float %1081, float %1088, !dbg !11
  %1090 = select i1 %.not, float %1088, float %1082, !dbg !11
  %1091 = select i1 %.not, float %1084, float %1083, !dbg !11
  %1092 = bitcast float %1091 to i32, !dbg !11
  %1093 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1092), !dbg !11
  %1094 = bitcast i32 %1093 to float, !dbg !11
  %1095 = select i1 %.not, float %1083, float %1094, !dbg !11
  %1096 = select i1 %.not, float %1094, float %1084, !dbg !11
  %1097 = select i1 %.not36, float %1095, float %1089, !dbg !11
  %1098 = bitcast float %1097 to i32, !dbg !11
  %1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1098), !dbg !11
  %1100 = bitcast i32 %1099 to float, !dbg !11
  %1101 = select i1 %.not36, float %1096, float %1090, !dbg !11
  %1102 = bitcast float %1101 to i32, !dbg !11
  %1103 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1102), !dbg !11
  %1104 = bitcast i32 %1103 to float, !dbg !11
  %1105 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1106 = shufflevector <4 x i1> %1105, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1107 = insertelement <4 x float> poison, float %1089, i64 0, !dbg !11
  %1108 = insertelement <4 x float> %1107, float %1090, i64 1, !dbg !11
  %1109 = insertelement <4 x float> %1108, float %1100, i64 2, !dbg !11
  %1110 = insertelement <4 x float> %1109, float %1104, i64 3, !dbg !11
  %1111 = insertelement <4 x float> poison, float %1100, i64 0, !dbg !11
  %1112 = insertelement <4 x float> %1111, float %1104, i64 1, !dbg !11
  %1113 = insertelement <4 x float> %1112, float %1095, i64 2, !dbg !11
  %1114 = insertelement <4 x float> %1113, float %1096, i64 3, !dbg !11
  %1115 = select <4 x i1> %1106, <4 x float> %1110, <4 x float> %1114, !dbg !11
  %1116 = extractelement <4 x float> %1043, i64 0, !dbg !11
  %1117 = extractelement <4 x float> %1043, i64 1, !dbg !11
  %1118 = extractelement <4 x float> %1043, i64 2, !dbg !11
  %1119 = extractelement <4 x float> %1043, i64 3, !dbg !11
  %1120 = select i1 %.not, float %1117, float %1116, !dbg !11
  %1121 = bitcast float %1120 to i32, !dbg !11
  %1122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1121), !dbg !11
  %1123 = bitcast i32 %1122 to float, !dbg !11
  %1124 = select i1 %.not, float %1116, float %1123, !dbg !11
  %1125 = select i1 %.not, float %1123, float %1117, !dbg !11
  %1126 = select i1 %.not, float %1119, float %1118, !dbg !11
  %1127 = bitcast float %1126 to i32, !dbg !11
  %1128 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1127), !dbg !11
  %1129 = bitcast i32 %1128 to float, !dbg !11
  %1130 = select i1 %.not, float %1118, float %1129, !dbg !11
  %1131 = select i1 %.not, float %1129, float %1119, !dbg !11
  %1132 = select i1 %.not36, float %1130, float %1124, !dbg !11
  %1133 = bitcast float %1132 to i32, !dbg !11
  %1134 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1133), !dbg !11
  %1135 = bitcast i32 %1134 to float, !dbg !11
  %1136 = select i1 %.not36, float %1131, float %1125, !dbg !11
  %1137 = bitcast float %1136 to i32, !dbg !11
  %1138 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1137), !dbg !11
  %1139 = bitcast i32 %1138 to float, !dbg !11
  %1140 = insertelement <4 x float> poison, float %1124, i64 0, !dbg !11
  %1141 = insertelement <4 x float> %1140, float %1125, i64 1, !dbg !11
  %1142 = insertelement <4 x float> %1141, float %1135, i64 2, !dbg !11
  %1143 = insertelement <4 x float> %1142, float %1139, i64 3, !dbg !11
  %1144 = insertelement <4 x float> poison, float %1135, i64 0, !dbg !11
  %1145 = insertelement <4 x float> %1144, float %1139, i64 1, !dbg !11
  %1146 = insertelement <4 x float> %1145, float %1130, i64 2, !dbg !11
  %1147 = insertelement <4 x float> %1146, float %1131, i64 3, !dbg !11
  %1148 = select <4 x i1> %1106, <4 x float> %1143, <4 x float> %1147, !dbg !11
  %1149 = extractelement <4 x float> %1044, i64 0, !dbg !11
  %1150 = extractelement <4 x float> %1044, i64 1, !dbg !11
  %1151 = extractelement <4 x float> %1044, i64 2, !dbg !11
  %1152 = extractelement <4 x float> %1044, i64 3, !dbg !11
  %1153 = select i1 %.not, float %1150, float %1149, !dbg !11
  %1154 = bitcast float %1153 to i32, !dbg !11
  %1155 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1154), !dbg !11
  %1156 = bitcast i32 %1155 to float, !dbg !11
  %1157 = select i1 %.not, float %1149, float %1156, !dbg !11
  %1158 = select i1 %.not, float %1156, float %1150, !dbg !11
  %1159 = select i1 %.not, float %1152, float %1151, !dbg !11
  %1160 = bitcast float %1159 to i32, !dbg !11
  %1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1160), !dbg !11
  %1162 = bitcast i32 %1161 to float, !dbg !11
  %1163 = select i1 %.not, float %1151, float %1162, !dbg !11
  %1164 = select i1 %.not, float %1162, float %1152, !dbg !11
  %1165 = select i1 %.not36, float %1163, float %1157, !dbg !11
  %1166 = bitcast float %1165 to i32, !dbg !11
  %1167 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1166), !dbg !11
  %1168 = bitcast i32 %1167 to float, !dbg !11
  %1169 = select i1 %.not36, float %1164, float %1158, !dbg !11
  %1170 = bitcast float %1169 to i32, !dbg !11
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1170), !dbg !11
  %1172 = bitcast i32 %1171 to float, !dbg !11
  %1173 = insertelement <4 x float> poison, float %1157, i64 0, !dbg !11
  %1174 = insertelement <4 x float> %1173, float %1158, i64 1, !dbg !11
  %1175 = insertelement <4 x float> %1174, float %1168, i64 2, !dbg !11
  %1176 = insertelement <4 x float> %1175, float %1172, i64 3, !dbg !11
  %1177 = insertelement <4 x float> poison, float %1168, i64 0, !dbg !11
  %1178 = insertelement <4 x float> %1177, float %1172, i64 1, !dbg !11
  %1179 = insertelement <4 x float> %1178, float %1163, i64 2, !dbg !11
  %1180 = insertelement <4 x float> %1179, float %1164, i64 3, !dbg !11
  %1181 = select <4 x i1> %1106, <4 x float> %1176, <4 x float> %1180, !dbg !11
  %1182 = extractelement <4 x float> %1045, i64 0, !dbg !11
  %1183 = extractelement <4 x float> %1045, i64 1, !dbg !11
  %1184 = extractelement <4 x float> %1045, i64 2, !dbg !11
  %1185 = extractelement <4 x float> %1045, i64 3, !dbg !11
  %1186 = select i1 %.not, float %1183, float %1182, !dbg !11
  %1187 = bitcast float %1186 to i32, !dbg !11
  %1188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1187), !dbg !11
  %1189 = bitcast i32 %1188 to float, !dbg !11
  %1190 = select i1 %.not, float %1182, float %1189, !dbg !11
  %1191 = select i1 %.not, float %1189, float %1183, !dbg !11
  %1192 = select i1 %.not, float %1185, float %1184, !dbg !11
  %1193 = bitcast float %1192 to i32, !dbg !11
  %1194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1193), !dbg !11
  %1195 = bitcast i32 %1194 to float, !dbg !11
  %1196 = select i1 %.not, float %1184, float %1195, !dbg !11
  %1197 = select i1 %.not, float %1195, float %1185, !dbg !11
  %1198 = select i1 %.not36, float %1196, float %1190, !dbg !11
  %1199 = bitcast float %1198 to i32, !dbg !11
  %1200 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1199), !dbg !11
  %1201 = bitcast i32 %1200 to float, !dbg !11
  %1202 = select i1 %.not36, float %1197, float %1191, !dbg !11
  %1203 = bitcast float %1202 to i32, !dbg !11
  %1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1203), !dbg !11
  %1205 = bitcast i32 %1204 to float, !dbg !11
  %1206 = insertelement <4 x float> poison, float %1190, i64 0, !dbg !11
  %1207 = insertelement <4 x float> %1206, float %1191, i64 1, !dbg !11
  %1208 = insertelement <4 x float> %1207, float %1201, i64 2, !dbg !11
  %1209 = insertelement <4 x float> %1208, float %1205, i64 3, !dbg !11
  %1210 = insertelement <4 x float> poison, float %1201, i64 0, !dbg !11
  %1211 = insertelement <4 x float> %1210, float %1205, i64 1, !dbg !11
  %1212 = insertelement <4 x float> %1211, float %1196, i64 2, !dbg !11
  %1213 = insertelement <4 x float> %1212, float %1197, i64 3, !dbg !11
  %1214 = select <4 x i1> %1106, <4 x float> %1209, <4 x float> %1213, !dbg !11
  %1215 = extractelement <4 x float> %966, i64 0, !dbg !11
  %1216 = extractelement <4 x float> %966, i64 1, !dbg !11
  %1217 = extractelement <4 x float> %966, i64 2, !dbg !11
  %1218 = extractelement <4 x float> %966, i64 3, !dbg !11
  %1219 = select i1 %.not, float %1216, float %1215, !dbg !11
  %1220 = bitcast float %1219 to i32, !dbg !11
  %1221 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1220), !dbg !11
  %1222 = bitcast i32 %1221 to float, !dbg !11
  %1223 = select i1 %.not, float %1215, float %1222, !dbg !11
  %1224 = select i1 %.not, float %1222, float %1216, !dbg !11
  %1225 = select i1 %.not, float %1218, float %1217, !dbg !11
  %1226 = bitcast float %1225 to i32, !dbg !11
  %1227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1226), !dbg !11
  %1228 = bitcast i32 %1227 to float, !dbg !11
  %1229 = select i1 %.not, float %1217, float %1228, !dbg !11
  %1230 = select i1 %.not, float %1228, float %1218, !dbg !11
  %1231 = select i1 %.not36, float %1229, float %1223, !dbg !11
  %1232 = bitcast float %1231 to i32, !dbg !11
  %1233 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1232), !dbg !11
  %1234 = bitcast i32 %1233 to float, !dbg !11
  %1235 = select i1 %.not36, float %1230, float %1224, !dbg !11
  %1236 = bitcast float %1235 to i32, !dbg !11
  %1237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1236), !dbg !11
  %1238 = bitcast i32 %1237 to float, !dbg !11
  %1239 = insertelement <4 x float> poison, float %1223, i64 0, !dbg !11
  %1240 = insertelement <4 x float> %1239, float %1224, i64 1, !dbg !11
  %1241 = insertelement <4 x float> %1240, float %1234, i64 2, !dbg !11
  %1242 = insertelement <4 x float> %1241, float %1238, i64 3, !dbg !11
  %1243 = insertelement <4 x float> poison, float %1234, i64 0, !dbg !11
  %1244 = insertelement <4 x float> %1243, float %1238, i64 1, !dbg !11
  %1245 = insertelement <4 x float> %1244, float %1229, i64 2, !dbg !11
  %1246 = insertelement <4 x float> %1245, float %1230, i64 3, !dbg !11
  %1247 = select <4 x i1> %1106, <4 x float> %1242, <4 x float> %1246, !dbg !11
  %1248 = extractelement <4 x float> %968, i64 0, !dbg !11
  %1249 = extractelement <4 x float> %968, i64 1, !dbg !11
  %1250 = extractelement <4 x float> %968, i64 2, !dbg !11
  %1251 = extractelement <4 x float> %968, i64 3, !dbg !11
  %1252 = select i1 %.not, float %1249, float %1248, !dbg !11
  %1253 = bitcast float %1252 to i32, !dbg !11
  %1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1253), !dbg !11
  %1255 = bitcast i32 %1254 to float, !dbg !11
  %1256 = select i1 %.not, float %1248, float %1255, !dbg !11
  %1257 = select i1 %.not, float %1255, float %1249, !dbg !11
  %1258 = select i1 %.not, float %1251, float %1250, !dbg !11
  %1259 = bitcast float %1258 to i32, !dbg !11
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1259), !dbg !11
  %1261 = bitcast i32 %1260 to float, !dbg !11
  %1262 = select i1 %.not, float %1250, float %1261, !dbg !11
  %1263 = select i1 %.not, float %1261, float %1251, !dbg !11
  %1264 = select i1 %.not36, float %1262, float %1256, !dbg !11
  %1265 = bitcast float %1264 to i32, !dbg !11
  %1266 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1265), !dbg !11
  %1267 = bitcast i32 %1266 to float, !dbg !11
  %1268 = select i1 %.not36, float %1263, float %1257, !dbg !11
  %1269 = bitcast float %1268 to i32, !dbg !11
  %1270 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1269), !dbg !11
  %1271 = bitcast i32 %1270 to float, !dbg !11
  %1272 = insertelement <4 x float> poison, float %1256, i64 0, !dbg !11
  %1273 = insertelement <4 x float> %1272, float %1257, i64 1, !dbg !11
  %1274 = insertelement <4 x float> %1273, float %1267, i64 2, !dbg !11
  %1275 = insertelement <4 x float> %1274, float %1271, i64 3, !dbg !11
  %1276 = insertelement <4 x float> poison, float %1267, i64 0, !dbg !11
  %1277 = insertelement <4 x float> %1276, float %1271, i64 1, !dbg !11
  %1278 = insertelement <4 x float> %1277, float %1262, i64 2, !dbg !11
  %1279 = insertelement <4 x float> %1278, float %1263, i64 3, !dbg !11
  %1280 = select <4 x i1> %1106, <4 x float> %1275, <4 x float> %1279, !dbg !11
  %1281 = extractelement <4 x float> %969, i64 0, !dbg !11
  %1282 = extractelement <4 x float> %969, i64 1, !dbg !11
  %1283 = extractelement <4 x float> %969, i64 2, !dbg !11
  %1284 = extractelement <4 x float> %969, i64 3, !dbg !11
  %1285 = select i1 %.not, float %1282, float %1281, !dbg !11
  %1286 = bitcast float %1285 to i32, !dbg !11
  %1287 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1286), !dbg !11
  %1288 = bitcast i32 %1287 to float, !dbg !11
  %1289 = select i1 %.not, float %1281, float %1288, !dbg !11
  %1290 = select i1 %.not, float %1288, float %1282, !dbg !11
  %1291 = select i1 %.not, float %1284, float %1283, !dbg !11
  %1292 = bitcast float %1291 to i32, !dbg !11
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1292), !dbg !11
  %1294 = bitcast i32 %1293 to float, !dbg !11
  %1295 = select i1 %.not, float %1283, float %1294, !dbg !11
  %1296 = select i1 %.not, float %1294, float %1284, !dbg !11
  %1297 = select i1 %.not36, float %1295, float %1289, !dbg !11
  %1298 = bitcast float %1297 to i32, !dbg !11
  %1299 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1298), !dbg !11
  %1300 = bitcast i32 %1299 to float, !dbg !11
  %1301 = select i1 %.not36, float %1296, float %1290, !dbg !11
  %1302 = bitcast float %1301 to i32, !dbg !11
  %1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1302), !dbg !11
  %1304 = bitcast i32 %1303 to float, !dbg !11
  %1305 = insertelement <4 x float> poison, float %1289, i64 0, !dbg !11
  %1306 = insertelement <4 x float> %1305, float %1290, i64 1, !dbg !11
  %1307 = insertelement <4 x float> %1306, float %1300, i64 2, !dbg !11
  %1308 = insertelement <4 x float> %1307, float %1304, i64 3, !dbg !11
  %1309 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !11
  %1310 = insertelement <4 x float> %1309, float %1304, i64 1, !dbg !11
  %1311 = insertelement <4 x float> %1310, float %1295, i64 2, !dbg !11
  %1312 = insertelement <4 x float> %1311, float %1296, i64 3, !dbg !11
  %1313 = select <4 x i1> %1106, <4 x float> %1308, <4 x float> %1312, !dbg !11
  %1314 = extractelement <4 x float> %970, i64 0, !dbg !11
  %1315 = extractelement <4 x float> %970, i64 1, !dbg !11
  %1316 = extractelement <4 x float> %970, i64 2, !dbg !11
  %1317 = extractelement <4 x float> %970, i64 3, !dbg !11
  %1318 = select i1 %.not, float %1315, float %1314, !dbg !11
  %1319 = bitcast float %1318 to i32, !dbg !11
  %1320 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1319), !dbg !11
  %1321 = bitcast i32 %1320 to float, !dbg !11
  %1322 = select i1 %.not, float %1314, float %1321, !dbg !11
  %1323 = select i1 %.not, float %1321, float %1315, !dbg !11
  %1324 = select i1 %.not, float %1317, float %1316, !dbg !11
  %1325 = bitcast float %1324 to i32, !dbg !11
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1325), !dbg !11
  %1327 = bitcast i32 %1326 to float, !dbg !11
  %1328 = select i1 %.not, float %1316, float %1327, !dbg !11
  %1329 = select i1 %.not, float %1327, float %1317, !dbg !11
  %1330 = select i1 %.not36, float %1328, float %1322, !dbg !11
  %1331 = bitcast float %1330 to i32, !dbg !11
  %1332 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1331), !dbg !11
  %1333 = bitcast i32 %1332 to float, !dbg !11
  %1334 = select i1 %.not36, float %1329, float %1323, !dbg !11
  %1335 = bitcast float %1334 to i32, !dbg !11
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1335), !dbg !11
  %1337 = bitcast i32 %1336 to float, !dbg !11
  %1338 = insertelement <4 x float> poison, float %1322, i64 0, !dbg !11
  %1339 = insertelement <4 x float> %1338, float %1323, i64 1, !dbg !11
  %1340 = insertelement <4 x float> %1339, float %1333, i64 2, !dbg !11
  %1341 = insertelement <4 x float> %1340, float %1337, i64 3, !dbg !11
  %1342 = insertelement <4 x float> poison, float %1333, i64 0, !dbg !11
  %1343 = insertelement <4 x float> %1342, float %1337, i64 1, !dbg !11
  %1344 = insertelement <4 x float> %1343, float %1328, i64 2, !dbg !11
  %1345 = insertelement <4 x float> %1344, float %1329, i64 3, !dbg !11
  %1346 = select <4 x i1> %1106, <4 x float> %1341, <4 x float> %1345, !dbg !11
  %1347 = extractelement <4 x float> %890, i64 0, !dbg !11
  %1348 = extractelement <4 x float> %890, i64 1, !dbg !11
  %1349 = extractelement <4 x float> %890, i64 2, !dbg !11
  %1350 = extractelement <4 x float> %890, i64 3, !dbg !11
  %1351 = select i1 %.not, float %1348, float %1347, !dbg !11
  %1352 = bitcast float %1351 to i32, !dbg !11
  %1353 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1352), !dbg !11
  %1354 = bitcast i32 %1353 to float, !dbg !11
  %1355 = select i1 %.not, float %1347, float %1354, !dbg !11
  %1356 = select i1 %.not, float %1354, float %1348, !dbg !11
  %1357 = select i1 %.not, float %1350, float %1349, !dbg !11
  %1358 = bitcast float %1357 to i32, !dbg !11
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1358), !dbg !11
  %1360 = bitcast i32 %1359 to float, !dbg !11
  %1361 = select i1 %.not, float %1349, float %1360, !dbg !11
  %1362 = select i1 %.not, float %1360, float %1350, !dbg !11
  %1363 = select i1 %.not36, float %1361, float %1355, !dbg !11
  %1364 = bitcast float %1363 to i32, !dbg !11
  %1365 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1364), !dbg !11
  %1366 = bitcast i32 %1365 to float, !dbg !11
  %1367 = select i1 %.not36, float %1362, float %1356, !dbg !11
  %1368 = bitcast float %1367 to i32, !dbg !11
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1368), !dbg !11
  %1370 = bitcast i32 %1369 to float, !dbg !11
  %1371 = insertelement <4 x float> poison, float %1355, i64 0, !dbg !11
  %1372 = insertelement <4 x float> %1371, float %1356, i64 1, !dbg !11
  %1373 = insertelement <4 x float> %1372, float %1366, i64 2, !dbg !11
  %1374 = insertelement <4 x float> %1373, float %1370, i64 3, !dbg !11
  %1375 = insertelement <4 x float> poison, float %1366, i64 0, !dbg !11
  %1376 = insertelement <4 x float> %1375, float %1370, i64 1, !dbg !11
  %1377 = insertelement <4 x float> %1376, float %1361, i64 2, !dbg !11
  %1378 = insertelement <4 x float> %1377, float %1362, i64 3, !dbg !11
  %1379 = select <4 x i1> %1106, <4 x float> %1374, <4 x float> %1378, !dbg !11
  %1380 = extractelement <4 x float> %892, i64 0, !dbg !11
  %1381 = extractelement <4 x float> %892, i64 1, !dbg !11
  %1382 = extractelement <4 x float> %892, i64 2, !dbg !11
  %1383 = extractelement <4 x float> %892, i64 3, !dbg !11
  %1384 = select i1 %.not, float %1381, float %1380, !dbg !11
  %1385 = bitcast float %1384 to i32, !dbg !11
  %1386 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1385), !dbg !11
  %1387 = bitcast i32 %1386 to float, !dbg !11
  %1388 = select i1 %.not, float %1380, float %1387, !dbg !11
  %1389 = select i1 %.not, float %1387, float %1381, !dbg !11
  %1390 = select i1 %.not, float %1383, float %1382, !dbg !11
  %1391 = bitcast float %1390 to i32, !dbg !11
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1391), !dbg !11
  %1393 = bitcast i32 %1392 to float, !dbg !11
  %1394 = select i1 %.not, float %1382, float %1393, !dbg !11
  %1395 = select i1 %.not, float %1393, float %1383, !dbg !11
  %1396 = select i1 %.not36, float %1394, float %1388, !dbg !11
  %1397 = bitcast float %1396 to i32, !dbg !11
  %1398 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1397), !dbg !11
  %1399 = bitcast i32 %1398 to float, !dbg !11
  %1400 = select i1 %.not36, float %1395, float %1389, !dbg !11
  %1401 = bitcast float %1400 to i32, !dbg !11
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1401), !dbg !11
  %1403 = bitcast i32 %1402 to float, !dbg !11
  %1404 = insertelement <4 x float> poison, float %1388, i64 0, !dbg !11
  %1405 = insertelement <4 x float> %1404, float %1389, i64 1, !dbg !11
  %1406 = insertelement <4 x float> %1405, float %1399, i64 2, !dbg !11
  %1407 = insertelement <4 x float> %1406, float %1403, i64 3, !dbg !11
  %1408 = insertelement <4 x float> poison, float %1399, i64 0, !dbg !11
  %1409 = insertelement <4 x float> %1408, float %1403, i64 1, !dbg !11
  %1410 = insertelement <4 x float> %1409, float %1394, i64 2, !dbg !11
  %1411 = insertelement <4 x float> %1410, float %1395, i64 3, !dbg !11
  %1412 = select <4 x i1> %1106, <4 x float> %1407, <4 x float> %1411, !dbg !11
  %1413 = extractelement <4 x float> %893, i64 0, !dbg !11
  %1414 = extractelement <4 x float> %893, i64 1, !dbg !11
  %1415 = extractelement <4 x float> %893, i64 2, !dbg !11
  %1416 = extractelement <4 x float> %893, i64 3, !dbg !11
  %1417 = select i1 %.not, float %1414, float %1413, !dbg !11
  %1418 = bitcast float %1417 to i32, !dbg !11
  %1419 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1418), !dbg !11
  %1420 = bitcast i32 %1419 to float, !dbg !11
  %1421 = select i1 %.not, float %1413, float %1420, !dbg !11
  %1422 = select i1 %.not, float %1420, float %1414, !dbg !11
  %1423 = select i1 %.not, float %1416, float %1415, !dbg !11
  %1424 = bitcast float %1423 to i32, !dbg !11
  %1425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1424), !dbg !11
  %1426 = bitcast i32 %1425 to float, !dbg !11
  %1427 = select i1 %.not, float %1415, float %1426, !dbg !11
  %1428 = select i1 %.not, float %1426, float %1416, !dbg !11
  %1429 = select i1 %.not36, float %1427, float %1421, !dbg !11
  %1430 = bitcast float %1429 to i32, !dbg !11
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1430), !dbg !11
  %1432 = bitcast i32 %1431 to float, !dbg !11
  %1433 = select i1 %.not36, float %1428, float %1422, !dbg !11
  %1434 = bitcast float %1433 to i32, !dbg !11
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1434), !dbg !11
  %1436 = bitcast i32 %1435 to float, !dbg !11
  %1437 = insertelement <4 x float> poison, float %1421, i64 0, !dbg !11
  %1438 = insertelement <4 x float> %1437, float %1422, i64 1, !dbg !11
  %1439 = insertelement <4 x float> %1438, float %1432, i64 2, !dbg !11
  %1440 = insertelement <4 x float> %1439, float %1436, i64 3, !dbg !11
  %1441 = insertelement <4 x float> poison, float %1432, i64 0, !dbg !11
  %1442 = insertelement <4 x float> %1441, float %1436, i64 1, !dbg !11
  %1443 = insertelement <4 x float> %1442, float %1427, i64 2, !dbg !11
  %1444 = insertelement <4 x float> %1443, float %1428, i64 3, !dbg !11
  %1445 = select <4 x i1> %1106, <4 x float> %1440, <4 x float> %1444, !dbg !11
  %1446 = extractelement <4 x float> %894, i64 0, !dbg !11
  %1447 = extractelement <4 x float> %894, i64 1, !dbg !11
  %1448 = extractelement <4 x float> %894, i64 2, !dbg !11
  %1449 = extractelement <4 x float> %894, i64 3, !dbg !11
  %1450 = select i1 %.not, float %1447, float %1446, !dbg !11
  %1451 = bitcast float %1450 to i32, !dbg !11
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1451), !dbg !11
  %1453 = bitcast i32 %1452 to float, !dbg !11
  %1454 = select i1 %.not, float %1446, float %1453, !dbg !11
  %1455 = select i1 %.not, float %1453, float %1447, !dbg !11
  %1456 = select i1 %.not, float %1449, float %1448, !dbg !11
  %1457 = bitcast float %1456 to i32, !dbg !11
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1457), !dbg !11
  %1459 = bitcast i32 %1458 to float, !dbg !11
  %1460 = select i1 %.not, float %1448, float %1459, !dbg !11
  %1461 = select i1 %.not, float %1459, float %1449, !dbg !11
  %1462 = select i1 %.not36, float %1460, float %1454, !dbg !11
  %1463 = bitcast float %1462 to i32, !dbg !11
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1463), !dbg !11
  %1465 = bitcast i32 %1464 to float, !dbg !11
  %1466 = select i1 %.not36, float %1461, float %1455, !dbg !11
  %1467 = bitcast float %1466 to i32, !dbg !11
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1467), !dbg !11
  %1469 = bitcast i32 %1468 to float, !dbg !11
  %1470 = insertelement <4 x float> poison, float %1454, i64 0, !dbg !11
  %1471 = insertelement <4 x float> %1470, float %1455, i64 1, !dbg !11
  %1472 = insertelement <4 x float> %1471, float %1465, i64 2, !dbg !11
  %1473 = insertelement <4 x float> %1472, float %1469, i64 3, !dbg !11
  %1474 = insertelement <4 x float> poison, float %1465, i64 0, !dbg !11
  %1475 = insertelement <4 x float> %1474, float %1469, i64 1, !dbg !11
  %1476 = insertelement <4 x float> %1475, float %1460, i64 2, !dbg !11
  %1477 = insertelement <4 x float> %1476, float %1461, i64 3, !dbg !11
  %1478 = select <4 x i1> %1106, <4 x float> %1473, <4 x float> %1477, !dbg !11
  %1479 = extractelement <4 x float> %796, i64 0, !dbg !11
  %1480 = extractelement <4 x float> %796, i64 1, !dbg !11
  %1481 = extractelement <4 x float> %796, i64 2, !dbg !11
  %1482 = extractelement <4 x float> %796, i64 3, !dbg !11
  %1483 = select i1 %.not, float %1480, float %1479, !dbg !11
  %1484 = bitcast float %1483 to i32, !dbg !11
  %1485 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1484), !dbg !11
  %1486 = bitcast i32 %1485 to float, !dbg !11
  %1487 = select i1 %.not, float %1479, float %1486, !dbg !11
  %1488 = select i1 %.not, float %1486, float %1480, !dbg !11
  %1489 = select i1 %.not, float %1482, float %1481, !dbg !11
  %1490 = bitcast float %1489 to i32, !dbg !11
  %1491 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1490), !dbg !11
  %1492 = bitcast i32 %1491 to float, !dbg !11
  %1493 = select i1 %.not, float %1481, float %1492, !dbg !11
  %1494 = select i1 %.not, float %1492, float %1482, !dbg !11
  %1495 = select i1 %.not36, float %1493, float %1487, !dbg !11
  %1496 = bitcast float %1495 to i32, !dbg !11
  %1497 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1496), !dbg !11
  %1498 = select i1 %.not36, float %1494, float %1488, !dbg !11
  %1499 = bitcast float %1498 to i32, !dbg !11
  %1500 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1499), !dbg !11
  %1501 = extractelement <4 x float> %798, i64 0, !dbg !11
  %1502 = extractelement <4 x float> %798, i64 1, !dbg !11
  %1503 = extractelement <4 x float> %798, i64 2, !dbg !11
  %1504 = extractelement <4 x float> %798, i64 3, !dbg !11
  %1505 = select i1 %.not, float %1502, float %1501, !dbg !11
  %1506 = bitcast float %1505 to i32, !dbg !11
  %1507 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1506), !dbg !11
  %1508 = bitcast i32 %1507 to float, !dbg !11
  %1509 = select i1 %.not, float %1501, float %1508, !dbg !11
  %1510 = select i1 %.not, float %1508, float %1502, !dbg !11
  %1511 = select i1 %.not, float %1504, float %1503, !dbg !11
  %1512 = bitcast float %1511 to i32, !dbg !11
  %1513 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1512), !dbg !11
  %1514 = bitcast i32 %1513 to float, !dbg !11
  %1515 = select i1 %.not, float %1503, float %1514, !dbg !11
  %1516 = select i1 %.not, float %1514, float %1504, !dbg !11
  %1517 = select i1 %.not36, float %1515, float %1509, !dbg !11
  %1518 = bitcast float %1517 to i32, !dbg !11
  %1519 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1518), !dbg !11
  %1520 = select i1 %.not36, float %1516, float %1510, !dbg !11
  %1521 = bitcast float %1520 to i32, !dbg !11
  %1522 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1521), !dbg !11
  %1523 = extractelement <4 x float> %817, i64 0, !dbg !11
  %1524 = extractelement <4 x float> %817, i64 1, !dbg !11
  %1525 = extractelement <4 x float> %817, i64 2, !dbg !11
  %1526 = extractelement <4 x float> %817, i64 3, !dbg !11
  %1527 = select i1 %.not, float %1524, float %1523, !dbg !11
  %1528 = bitcast float %1527 to i32, !dbg !11
  %1529 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1528), !dbg !11
  %1530 = bitcast i32 %1529 to float, !dbg !11
  %1531 = select i1 %.not, float %1523, float %1530, !dbg !11
  %1532 = select i1 %.not, float %1530, float %1524, !dbg !11
  %1533 = select i1 %.not, float %1526, float %1525, !dbg !11
  %1534 = bitcast float %1533 to i32, !dbg !11
  %1535 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1534), !dbg !11
  %1536 = bitcast i32 %1535 to float, !dbg !11
  %1537 = select i1 %.not, float %1525, float %1536, !dbg !11
  %1538 = select i1 %.not, float %1536, float %1526, !dbg !11
  %1539 = select i1 %.not36, float %1537, float %1531, !dbg !11
  %1540 = bitcast float %1539 to i32, !dbg !11
  %1541 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1540), !dbg !11
  %1542 = select i1 %.not36, float %1538, float %1532, !dbg !11
  %1543 = bitcast float %1542 to i32, !dbg !11
  %1544 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1543), !dbg !11
  %1545 = extractelement <4 x float> %818, i64 0, !dbg !11
  %1546 = extractelement <4 x float> %818, i64 1, !dbg !11
  %1547 = extractelement <4 x float> %818, i64 2, !dbg !11
  %1548 = extractelement <4 x float> %818, i64 3, !dbg !11
  %1549 = select i1 %.not, float %1546, float %1545, !dbg !11
  %1550 = bitcast float %1549 to i32, !dbg !11
  %1551 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1550), !dbg !11
  %1552 = bitcast i32 %1551 to float, !dbg !11
  %1553 = select i1 %.not, float %1545, float %1552, !dbg !11
  %1554 = select i1 %.not, float %1552, float %1546, !dbg !11
  %1555 = select i1 %.not, float %1548, float %1547, !dbg !11
  %1556 = bitcast float %1555 to i32, !dbg !11
  %1557 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %464, i32 %1556), !dbg !11
  %1558 = bitcast i32 %1557 to float, !dbg !11
  %1559 = select i1 %.not, float %1547, float %1558, !dbg !11
  %1560 = select i1 %.not, float %1558, float %1548, !dbg !11
  %1561 = select i1 %.not36, float %1559, float %1553, !dbg !11
  %1562 = bitcast float %1561 to i32, !dbg !11
  %1563 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1562), !dbg !11
  %1564 = select i1 %.not36, float %1560, float %1554, !dbg !11
  %1565 = bitcast float %1564 to i32, !dbg !11
  %1566 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %472, i32 %1565), !dbg !11
  %1567 = shufflevector <1 x float> %1078, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1568 = shufflevector <1 x float> %1073, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1569 = bitcast i32 %1563 to float, !dbg !11
  %1570 = bitcast i32 %1566 to float, !dbg !11
  %1571 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !11
  %1572 = insertelement <4 x float> %1571, float %1554, i64 1, !dbg !11
  %1573 = insertelement <4 x float> %1572, float %1569, i64 2, !dbg !11
  %1574 = insertelement <4 x float> %1573, float %1570, i64 3, !dbg !11
  %1575 = insertelement <4 x float> poison, float %1569, i64 0, !dbg !11
  %1576 = insertelement <4 x float> %1575, float %1570, i64 1, !dbg !11
  %1577 = insertelement <4 x float> %1576, float %1559, i64 2, !dbg !11
  %1578 = insertelement <4 x float> %1577, float %1560, i64 3, !dbg !11
  %1579 = select <4 x i1> %1106, <4 x float> %1574, <4 x float> %1578, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1568, !dbg !50
  %normalized = fmul <4 x float> %row.reciprocal, %1579, !dbg !50
  %1580 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !51
  %1581 = extractelement <4 x half> %1580, i64 3, !dbg !51
  %1582 = extractelement <4 x half> %1580, i64 2, !dbg !51
  %1583 = extractelement <4 x half> %1580, i64 1, !dbg !51
  %1584 = extractelement <4 x half> %1580, i64 0, !dbg !51
  %1585 = bitcast i32 %1541 to float, !dbg !11
  %1586 = bitcast i32 %1544 to float, !dbg !11
  %1587 = insertelement <4 x float> poison, float %1531, i64 0, !dbg !11
  %1588 = insertelement <4 x float> %1587, float %1532, i64 1, !dbg !11
  %1589 = insertelement <4 x float> %1588, float %1585, i64 2, !dbg !11
  %1590 = insertelement <4 x float> %1589, float %1586, i64 3, !dbg !11
  %1591 = insertelement <4 x float> poison, float %1585, i64 0, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1586, i64 1, !dbg !11
  %1593 = insertelement <4 x float> %1592, float %1537, i64 2, !dbg !11
  %1594 = insertelement <4 x float> %1593, float %1538, i64 3, !dbg !11
  %1595 = select <4 x i1> %1106, <4 x float> %1590, <4 x float> %1594, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal, %1595, !dbg !50
  %1596 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !51
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !51
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !51
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !51
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !51
  %1601 = bitcast i32 %1519 to float, !dbg !11
  %1602 = bitcast i32 %1522 to float, !dbg !11
  %1603 = insertelement <4 x float> poison, float %1509, i64 0, !dbg !11
  %1604 = insertelement <4 x float> %1603, float %1510, i64 1, !dbg !11
  %1605 = insertelement <4 x float> %1604, float %1601, i64 2, !dbg !11
  %1606 = insertelement <4 x float> %1605, float %1602, i64 3, !dbg !11
  %1607 = insertelement <4 x float> poison, float %1601, i64 0, !dbg !11
  %1608 = insertelement <4 x float> %1607, float %1602, i64 1, !dbg !11
  %1609 = insertelement <4 x float> %1608, float %1515, i64 2, !dbg !11
  %1610 = insertelement <4 x float> %1609, float %1516, i64 3, !dbg !11
  %1611 = select <4 x i1> %1106, <4 x float> %1606, <4 x float> %1610, !dbg !11
  %row.reciprocal41 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1567, !dbg !50
  %normalized42 = fmul <4 x float> %row.reciprocal41, %1611, !dbg !50
  %1612 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !51
  %1613 = extractelement <4 x half> %1612, i64 3, !dbg !51
  %1614 = extractelement <4 x half> %1612, i64 2, !dbg !51
  %1615 = extractelement <4 x half> %1612, i64 1, !dbg !51
  %1616 = extractelement <4 x half> %1612, i64 0, !dbg !51
  %1617 = bitcast i32 %1497 to float, !dbg !11
  %1618 = bitcast i32 %1500 to float, !dbg !11
  %1619 = insertelement <4 x float> poison, float %1487, i64 0, !dbg !11
  %1620 = insertelement <4 x float> %1619, float %1488, i64 1, !dbg !11
  %1621 = insertelement <4 x float> %1620, float %1617, i64 2, !dbg !11
  %1622 = insertelement <4 x float> %1621, float %1618, i64 3, !dbg !11
  %1623 = insertelement <4 x float> poison, float %1617, i64 0, !dbg !11
  %1624 = insertelement <4 x float> %1623, float %1618, i64 1, !dbg !11
  %1625 = insertelement <4 x float> %1624, float %1493, i64 2, !dbg !11
  %1626 = insertelement <4 x float> %1625, float %1494, i64 3, !dbg !11
  %1627 = select <4 x i1> %1106, <4 x float> %1622, <4 x float> %1626, !dbg !11
  %normalized43 = fmul <4 x float> %row.reciprocal41, %1627, !dbg !50
  %1628 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !51
  %1629 = extractelement <4 x half> %1628, i64 3, !dbg !51
  %1630 = extractelement <4 x half> %1628, i64 2, !dbg !51
  %1631 = extractelement <4 x half> %1628, i64 1, !dbg !51
  %1632 = extractelement <4 x half> %1628, i64 0, !dbg !51
  %address.thread977 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !52, !range !9, !frisk.register_lifetimes !5
  %address.local978 = zext nneg i32 %address.thread977 to i64, !dbg !8
  %1633 = lshr i64 %address.local978, 2, !dbg !52
  %address.local981 = and i64 %1633, 48, !dbg !52
  %address.local983 = and i64 %address.local978, 240, !dbg !52
  %address.local982 = add i64 %25, %address.local978, !dbg !52
  %address.local984 = sub i64 %address.local982, %address.local983, !dbg !52
  %address.local985 = add i64 %address.local984, %address.local981, !dbg !52
  %1634 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %33, !dbg !52
  %address.local988.idx = shl nuw nsw i64 %address.local985, 8, !dbg !52
  %address.local988 = getelementptr inbounds nuw i8, ptr addrspace(1) %1634, i64 %address.local988.idx, !dbg !52
  %address.local990 = and i64 %1633, 12, !dbg !10
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local990, !dbg !52
  store half %1632, ptr addrspace(1) %address.local991, align 2, !dbg !52
  %address.local993 = or disjoint i64 %address.local990, 1, !dbg !52
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local993, !dbg !52
  store half %1631, ptr addrspace(1) %address.local994, align 2, !dbg !52
  %address.local996 = or disjoint i64 %address.local990, 2, !dbg !52
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local996, !dbg !52
  store half %1630, ptr addrspace(1) %address.local997, align 2, !dbg !52
  %address.local999 = or disjoint i64 %address.local990, 3, !dbg !52
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local999, !dbg !52
  store half %1629, ptr addrspace(1) %address.local1000, align 2, !dbg !52
  %address.local1002 = or disjoint i64 %address.local990, 16, !dbg !52
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1002, !dbg !52
  store half %1616, ptr addrspace(1) %address.local1003, align 2, !dbg !52
  %address.local1005 = or disjoint i64 %address.local990, 17, !dbg !52
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1005, !dbg !52
  store half %1615, ptr addrspace(1) %address.local1006, align 2, !dbg !52
  %address.local1008 = or disjoint i64 %address.local990, 18, !dbg !52
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1008, !dbg !52
  store half %1614, ptr addrspace(1) %address.local1009, align 2, !dbg !52
  %address.local1011 = or disjoint i64 %address.local990, 19, !dbg !52
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1011, !dbg !52
  store half %1613, ptr addrspace(1) %address.local1012, align 2, !dbg !52
  %address.local1016 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local988, i64 16384, !dbg !52
  %address.local1017 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local990, !dbg !52
  store half %1600, ptr addrspace(1) %address.local1017, align 2, !dbg !52
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local993, !dbg !52
  store half %1599, ptr addrspace(1) %address.local1019, align 2, !dbg !52
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local996, !dbg !52
  store half %1598, ptr addrspace(1) %address.local1021, align 2, !dbg !52
  %address.local1023 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local999, !dbg !52
  store half %1597, ptr addrspace(1) %address.local1023, align 2, !dbg !52
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1002, !dbg !52
  store half %1584, ptr addrspace(1) %address.local1025, align 2, !dbg !52
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1005, !dbg !52
  store half %1583, ptr addrspace(1) %address.local1027, align 2, !dbg !52
  %address.local1029 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1008, !dbg !52
  store half %1582, ptr addrspace(1) %address.local1029, align 2, !dbg !52
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1011, !dbg !52
  store half %1581, ptr addrspace(1) %address.local1031, align 2, !dbg !52
  %normalized45 = fmul <4 x float> %row.reciprocal, %1478, !dbg !50
  %1635 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !51
  %1636 = extractelement <4 x half> %1635, i64 3, !dbg !51
  %1637 = extractelement <4 x half> %1635, i64 2, !dbg !51
  %1638 = extractelement <4 x half> %1635, i64 1, !dbg !51
  %1639 = extractelement <4 x half> %1635, i64 0, !dbg !51
  %normalized46 = fmul <4 x float> %row.reciprocal, %1445, !dbg !50
  %1640 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !51
  %1641 = extractelement <4 x half> %1640, i64 3, !dbg !51
  %1642 = extractelement <4 x half> %1640, i64 2, !dbg !51
  %1643 = extractelement <4 x half> %1640, i64 1, !dbg !51
  %1644 = extractelement <4 x half> %1640, i64 0, !dbg !51
  %normalized48 = fmul <4 x float> %row.reciprocal41, %1412, !dbg !50
  %1645 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !51
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !51
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !51
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !51
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !51
  %normalized49 = fmul <4 x float> %row.reciprocal41, %1379, !dbg !50
  %1650 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !51
  %1651 = extractelement <4 x half> %1650, i64 3, !dbg !51
  %1652 = extractelement <4 x half> %1650, i64 2, !dbg !51
  %1653 = extractelement <4 x half> %1650, i64 1, !dbg !51
  %1654 = extractelement <4 x half> %1650, i64 0, !dbg !51
  %address.local1046 = or disjoint i64 %address.local990, 32, !dbg !52
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1046, !dbg !52
  store half %1654, ptr addrspace(1) %address.local1047, align 2, !dbg !52
  %address.local1049 = or disjoint i64 %address.local990, 33, !dbg !52
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1049, !dbg !52
  store half %1653, ptr addrspace(1) %address.local1050, align 2, !dbg !52
  %address.local1052 = or disjoint i64 %address.local990, 34, !dbg !52
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1052, !dbg !52
  store half %1652, ptr addrspace(1) %address.local1053, align 2, !dbg !52
  %address.local1055 = or disjoint i64 %address.local990, 35, !dbg !52
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1055, !dbg !52
  store half %1651, ptr addrspace(1) %address.local1056, align 2, !dbg !52
  %address.local1058 = or disjoint i64 %address.local990, 48, !dbg !52
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1058, !dbg !52
  store half %1649, ptr addrspace(1) %address.local1059, align 2, !dbg !52
  %address.local1061 = or disjoint i64 %address.local990, 49, !dbg !52
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1061, !dbg !52
  store half %1648, ptr addrspace(1) %address.local1062, align 2, !dbg !52
  %address.local1064 = or disjoint i64 %address.local990, 50, !dbg !52
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1064, !dbg !52
  store half %1647, ptr addrspace(1) %address.local1065, align 2, !dbg !52
  %address.local1067 = or i64 %1633, 51, !dbg !52
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1067, !dbg !52
  store half %1646, ptr addrspace(1) %address.local1068, align 2, !dbg !52
  %address.local1073 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1046, !dbg !52
  store half %1644, ptr addrspace(1) %address.local1073, align 2, !dbg !52
  %address.local1075 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1049, !dbg !52
  store half %1643, ptr addrspace(1) %address.local1075, align 2, !dbg !52
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1052, !dbg !52
  store half %1642, ptr addrspace(1) %address.local1077, align 2, !dbg !52
  %address.local1079 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1055, !dbg !52
  store half %1641, ptr addrspace(1) %address.local1079, align 2, !dbg !52
  %address.local1081 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1058, !dbg !52
  store half %1639, ptr addrspace(1) %address.local1081, align 2, !dbg !52
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1061, !dbg !52
  store half %1638, ptr addrspace(1) %address.local1083, align 2, !dbg !52
  %address.local1085 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1064, !dbg !52
  store half %1637, ptr addrspace(1) %address.local1085, align 2, !dbg !52
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1067, !dbg !52
  store half %1636, ptr addrspace(1) %address.local1087, align 2, !dbg !52
  %normalized51 = fmul <4 x float> %row.reciprocal, %1346, !dbg !50
  %1655 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !51
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !51
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !51
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !51
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !51
  %normalized52 = fmul <4 x float> %row.reciprocal, %1313, !dbg !50
  %1660 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !51
  %1661 = extractelement <4 x half> %1660, i64 3, !dbg !51
  %1662 = extractelement <4 x half> %1660, i64 2, !dbg !51
  %1663 = extractelement <4 x half> %1660, i64 1, !dbg !51
  %1664 = extractelement <4 x half> %1660, i64 0, !dbg !51
  %normalized54 = fmul <4 x float> %row.reciprocal41, %1280, !dbg !50
  %1665 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !51
  %1666 = extractelement <4 x half> %1665, i64 3, !dbg !51
  %1667 = extractelement <4 x half> %1665, i64 2, !dbg !51
  %1668 = extractelement <4 x half> %1665, i64 1, !dbg !51
  %1669 = extractelement <4 x half> %1665, i64 0, !dbg !51
  %normalized55 = fmul <4 x float> %row.reciprocal41, %1247, !dbg !50
  %1670 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !51
  %1671 = extractelement <4 x half> %1670, i64 3, !dbg !51
  %1672 = extractelement <4 x half> %1670, i64 2, !dbg !51
  %1673 = extractelement <4 x half> %1670, i64 1, !dbg !51
  %1674 = extractelement <4 x half> %1670, i64 0, !dbg !51
  %address.local1102 = or disjoint i64 %address.local990, 64, !dbg !52
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1102, !dbg !52
  store half %1674, ptr addrspace(1) %address.local1103, align 2, !dbg !52
  %address.local1105 = or disjoint i64 %address.local990, 65, !dbg !52
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1105, !dbg !52
  store half %1673, ptr addrspace(1) %address.local1106, align 2, !dbg !52
  %address.local1108 = or disjoint i64 %address.local990, 66, !dbg !52
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1108, !dbg !52
  store half %1672, ptr addrspace(1) %address.local1109, align 2, !dbg !52
  %address.local1111 = or disjoint i64 %address.local990, 67, !dbg !52
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1111, !dbg !52
  store half %1671, ptr addrspace(1) %address.local1112, align 2, !dbg !52
  %address.local1114 = or disjoint i64 %address.local990, 80, !dbg !52
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1114, !dbg !52
  store half %1669, ptr addrspace(1) %address.local1115, align 2, !dbg !52
  %address.local1117 = or disjoint i64 %address.local990, 81, !dbg !52
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1117, !dbg !52
  store half %1668, ptr addrspace(1) %address.local1118, align 2, !dbg !52
  %address.local1120 = or disjoint i64 %address.local990, 82, !dbg !52
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1120, !dbg !52
  store half %1667, ptr addrspace(1) %address.local1121, align 2, !dbg !52
  %address.local1123 = or disjoint i64 %address.local990, 83, !dbg !52
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1123, !dbg !52
  store half %1666, ptr addrspace(1) %address.local1124, align 2, !dbg !52
  %address.local1129 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1102, !dbg !52
  store half %1664, ptr addrspace(1) %address.local1129, align 2, !dbg !52
  %address.local1131 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1105, !dbg !52
  store half %1663, ptr addrspace(1) %address.local1131, align 2, !dbg !52
  %address.local1133 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1108, !dbg !52
  store half %1662, ptr addrspace(1) %address.local1133, align 2, !dbg !52
  %address.local1135 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1111, !dbg !52
  store half %1661, ptr addrspace(1) %address.local1135, align 2, !dbg !52
  %address.local1137 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1114, !dbg !52
  store half %1659, ptr addrspace(1) %address.local1137, align 2, !dbg !52
  %address.local1139 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1117, !dbg !52
  store half %1658, ptr addrspace(1) %address.local1139, align 2, !dbg !52
  %address.local1141 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1120, !dbg !52
  store half %1657, ptr addrspace(1) %address.local1141, align 2, !dbg !52
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1123, !dbg !52
  store half %1656, ptr addrspace(1) %address.local1143, align 2, !dbg !52
  %normalized57 = fmul <4 x float> %row.reciprocal, %1214, !dbg !50
  %1675 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !51
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !51
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !51
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !51
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !51
  %normalized58 = fmul <4 x float> %row.reciprocal, %1181, !dbg !50
  %1680 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !51
  %1681 = extractelement <4 x half> %1680, i64 3, !dbg !51
  %1682 = extractelement <4 x half> %1680, i64 2, !dbg !51
  %1683 = extractelement <4 x half> %1680, i64 1, !dbg !51
  %1684 = extractelement <4 x half> %1680, i64 0, !dbg !51
  %normalized60 = fmul <4 x float> %row.reciprocal41, %1148, !dbg !50
  %1685 = fptrunc <4 x float> %normalized60 to <4 x half>, !dbg !51
  %1686 = extractelement <4 x half> %1685, i64 3, !dbg !51
  %1687 = extractelement <4 x half> %1685, i64 2, !dbg !51
  %1688 = extractelement <4 x half> %1685, i64 1, !dbg !51
  %1689 = extractelement <4 x half> %1685, i64 0, !dbg !51
  %normalized61 = fmul <4 x float> %row.reciprocal41, %1115, !dbg !50
  %1690 = fptrunc <4 x float> %normalized61 to <4 x half>, !dbg !51
  %1691 = extractelement <4 x half> %1690, i64 3, !dbg !51
  %1692 = extractelement <4 x half> %1690, i64 2, !dbg !51
  %1693 = extractelement <4 x half> %1690, i64 1, !dbg !51
  %1694 = extractelement <4 x half> %1690, i64 0, !dbg !51
  %address.local1158 = or disjoint i64 %address.local990, 96, !dbg !52
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1158, !dbg !52
  store half %1694, ptr addrspace(1) %address.local1159, align 2, !dbg !52
  %address.local1161 = or disjoint i64 %address.local990, 97, !dbg !52
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1161, !dbg !52
  store half %1693, ptr addrspace(1) %address.local1162, align 2, !dbg !52
  %address.local1164 = or disjoint i64 %address.local990, 98, !dbg !52
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1164, !dbg !52
  store half %1692, ptr addrspace(1) %address.local1165, align 2, !dbg !52
  %address.local1167 = or disjoint i64 %address.local990, 99, !dbg !52
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1167, !dbg !52
  store half %1691, ptr addrspace(1) %address.local1168, align 2, !dbg !52
  %address.local1170 = or disjoint i64 %address.local990, 112, !dbg !52
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1170, !dbg !52
  store half %1689, ptr addrspace(1) %address.local1171, align 2, !dbg !52
  %address.local1173 = or disjoint i64 %address.local990, 113, !dbg !52
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1173, !dbg !52
  store half %1688, ptr addrspace(1) %address.local1174, align 2, !dbg !52
  %address.local1176 = or disjoint i64 %address.local990, 114, !dbg !52
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1176, !dbg !52
  store half %1687, ptr addrspace(1) %address.local1177, align 2, !dbg !52
  %address.local1179 = or i64 %1633, 115, !dbg !52
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local988, i64 %address.local1179, !dbg !52
  store half %1686, ptr addrspace(1) %address.local1180, align 2, !dbg !52
  %address.local1185 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1158, !dbg !52
  store half %1684, ptr addrspace(1) %address.local1185, align 2, !dbg !52
  %address.local1187 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1161, !dbg !52
  store half %1683, ptr addrspace(1) %address.local1187, align 2, !dbg !52
  %address.local1189 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1164, !dbg !52
  store half %1682, ptr addrspace(1) %address.local1189, align 2, !dbg !52
  %address.local1191 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1167, !dbg !52
  store half %1681, ptr addrspace(1) %address.local1191, align 2, !dbg !52
  %address.local1193 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1170, !dbg !52
  store half %1679, ptr addrspace(1) %address.local1193, align 2, !dbg !52
  %address.local1195 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1173, !dbg !52
  store half %1678, ptr addrspace(1) %address.local1195, align 2, !dbg !52
  %address.local1197 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1176, !dbg !52
  store half %1677, ptr addrspace(1) %address.local1197, align 2, !dbg !52
  %address.local1199 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1016, i64 %address.local1179, !dbg !52
  store half %1676, ptr addrspace(1) %address.local1199, align 2, !dbg !52
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

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
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal")
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
!12 = !DILocation(line: 68, column: 22, scope: !3)
!13 = !DILocation(line: 70, column: 26, scope: !3)
!14 = !{i32 0}
!15 = !{i32 1}
!16 = !{i32 2}
!17 = !{i32 3}
!18 = !{i32 4}
!19 = !{i32 5}
!20 = !{i32 6}
!21 = !{i32 7}
!22 = !{i32 8}
!23 = !{i32 9}
!24 = !{i32 10}
!25 = !{i32 11}
!26 = !{i32 12}
!27 = !{i32 13}
!28 = !{i32 14}
!29 = !{i32 15}
!30 = !{i32 16}
!31 = !{i32 17}
!32 = !{i32 18}
!33 = !{i32 19}
!34 = !{i32 20}
!35 = !{i32 21}
!36 = !{i32 22}
!37 = !DILocation(line: 86, column: 20, scope: !3)
!38 = !DILocation(line: 71, column: 23, scope: !3)
!39 = !DILocation(line: 74, column: 20, scope: !3)
!40 = !DILocation(line: 75, column: 22, scope: !3)
!41 = !DILocation(line: 76, column: 23, scope: !3)
!42 = !DILocation(line: 77, column: 20, scope: !3)
!43 = !DILocation(line: 88, column: 22, scope: !3)
!44 = !DILocation(line: 78, column: 22, scope: !3)
!45 = !DILocation(line: 87, column: 23, scope: !3)
!46 = !{i64 8192}
!47 = !DILocation(line: 79, column: 22, scope: !3)
!48 = !DILocation(line: 81, column: 16, scope: !3)
!49 = !DILocation(line: 80, column: 20, scope: !3)
!50 = !DILocation(line: 92, column: 23, scope: !3)
!51 = !DILocation(line: 93, column: 19, scope: !3)
!52 = !DILocation(line: 94, column: 9, scope: !3)
