; ModuleID = 'kernel.ll'
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
  %address.local = zext nneg i32 %address.thread to i64, !dbg !9
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local405 = and i64 %address.local404, 1536, !dbg !9
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !9
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local410 = and i64 %address.local409, 16256, !dbg !9
  %address.local411 = lshr i64 %address.local, 4, !dbg !9
  %address.local412 = and i64 %address.local411, 8, !dbg !9
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !9
  %36 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1360 = shl i64 %34, 8, !dbg !9
  %37 = add i64 %.idx1360, %.idx, !dbg !9
  %address.local414.idx = shl nuw nsw i64 %address.local413, 1, !dbg !9
  %38 = or disjoint i64 %address.local414.idx, %37, !dbg !9
  %dma.buffer.offset1358 = trunc i64 %38 to i32, !dbg !9
  %39 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !9
  %40 = bitcast <2 x i64> %39 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1359 = shufflevector <4 x i32> %40, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !9
  %address.local417 = and i64 %address.local416, 3584, !dbg !9
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !9
  %address.local423 = lshr i64 %address.local415, 4, !dbg !9
  %address.local424 = and i64 %address.local423, 24, !dbg !9
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !9
  %address.local426.idx = shl nuw nsw i64 %address.local425, 1, !dbg !9
  %41 = or disjoint i64 %address.local426.idx, %37, !dbg !9
  %dma.buffer.offset1356 = trunc i64 %41 to i32, !dbg !9
  %42 = shl nuw nsw i64 %31, 19, !dbg !9
  %43 = add i64 %35, %42, !dbg !9
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !9
  %44 = or disjoint i64 %43, 32, !dbg !9
  %45 = or disjoint i64 %address.local413, %44, !dbg !9
  %.tr = trunc i64 %45 to i32, !dbg !9
  %dma.buffer.offset1354 = shl i32 %.tr, 1, !dbg !9
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !9
  %46 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !9
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local418, i64 1, !dbg !9
  %48 = insertelement <4 x ptr addrspace(3)> %47, ptr addrspace(3) %address.local427, i64 2, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> %48, ptr addrspace(3) %address.local431, i64 3, !dbg !9
  %50 = ptrtoint <4 x ptr addrspace(3)> %49 to <4 x i64>, !dbg !9
  %51 = trunc <4 x i64> %50 to <4 x i32>, !dbg !9
  %52 = extractelement <4 x i32> %51, i64 0, !dbg !9
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !9
  %54 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1358, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %55 = extractelement <4 x i32> %51, i64 1, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1356, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %58 = extractelement <4 x i32> %51, i64 2, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1354, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %61 = extractelement <4 x i32> %51, i64 3, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = or disjoint i64 %address.local425, %44, !dbg !9
  %.tr1364 = trunc i64 %63 to i32, !dbg !9
  %dma.buffer.offset1352 = shl i32 %.tr1364, 1, !dbg !9
  %64 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1352, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %address.local435 = lshr i32 %address.thread, 2, !dbg !11
  %address.local436 = and i32 %address.local435, 4, !dbg !11
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local438 = and i32 %address.local437, 384, !dbg !11
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !11
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local441 = and i32 %address.local440, 120, !dbg !11
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !11
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local444 = and i32 %address.local443, 1024, !dbg !11
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !11
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !11
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !11
  %65 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !11
  %66 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %67 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1181 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !11
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1182 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !11
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !11
  %71 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %72 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %73 = or disjoint i64 %43, 64, !dbg !9
  %74 = or disjoint i64 %address.local413, %73, !dbg !9
  %.tr1366 = trunc i64 %74 to i32, !dbg !9
  %dma.buffer.offset1350 = shl i32 %.tr1366, 1, !dbg !9
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1350, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %76 = or disjoint i64 %address.local425, %73, !dbg !9
  %.tr1368 = trunc i64 %76 to i32, !dbg !9
  %dma.buffer.offset1348 = shl i32 %.tr1368, 1, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %lds.const.gep1183 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1183, align 8, !dbg !11
  %lds.const.gep1184 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1184, align 8, !dbg !11
  %lds.const.gep1185 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1185, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1186 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1186, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = or disjoint i64 %43, 96, !dbg !9
  %87 = or disjoint i64 %address.local413, %86, !dbg !9
  %.tr1370 = trunc i64 %87 to i32, !dbg !9
  %dma.buffer.offset1346 = shl i32 %.tr1370, 1, !dbg !9
  %88 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1346, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %89 = or disjoint i64 %address.local425, %86, !dbg !9
  %.tr1372 = trunc i64 %89 to i32, !dbg !9
  %dma.buffer.offset1344 = shl i32 %.tr1372, 1, !dbg !9
  %90 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1344, <4 x i32> %dma.buffer.rsrc1359) #9, !dbg !9
  %91 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep1181, align 8, !dbg !11
  %94 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !11
  %95 = load <4 x half>, ptr addrspace(3) %lds.const.gep1182, align 8, !dbg !11
  %96 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !11
  %98 = fmul <4 x half> %95, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep1183, align 8, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1184, align 8, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep1185, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !11
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep1186, align 8, !dbg !11
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %105 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !11
  %106 = fmul <4 x half> %103, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %107 = shl nuw nsw i32 %kdma.tid, 2, !dbg !13
  %108 = and i32 %107, 3084, !dbg !13
  %109 = and i32 %kdma.tid, 1, !dbg !13
  %110 = add nuw nsw i32 %109, 1, !dbg !13
  %narrow = or disjoint i32 %110, %108, !dbg !13
  %111 = lshr i32 %narrow, 1, !dbg !13
  %112 = zext nneg i32 %111 to i64, !dbg !13
  %k0.prime. = shl i64 %25, 19, !dbg !14
  %k0.prime.1923 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1466 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1467 = and i64 %k0.prime.kdma.source1466, 8, !dbg !14
  %k0.prime.kdma.source1468 = and i64 %k0.prime.kdma.source, 91, !dbg !14
  %k0.prime.kdma.source1469 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1470 = and i64 %k0.prime.kdma.source1469, 4, !dbg !14
  %k0.prime.kdma.source1471 = add nuw nsw i64 %k0.prime.kdma.source1470, %k0.prime.kdma.source, !dbg !14
  %k0.prime.kdma.source1472 = and i64 %k0.prime.kdma.source1471, 4, !dbg !14
  %k0.prime.kdma.source1473 = or disjoint i64 %k0.prime.kdma.source1472, %k0.prime.kdma.source1468, !dbg !14
  %k0.prime.kdma.source1474 = shl nuw nsw i64 %k0.prime.kdma.source1473, 7, !dbg !14
  %k0.prime.kdma.source1475 = or disjoint i64 %k0.prime.kdma.source1474, %k0.prime.kdma.source1467, !dbg !14
  %k0.prime.kdma.source1476 = or disjoint i64 %k0.prime.kdma.source1475, %k0.prime.1923, !dbg !14
  %k0.prime.kdma.source1477 = trunc nuw nsw i64 %k0.prime.kdma.source1476 to i32, !dbg !14
  %k0.prime.kdma.source1478 = shl nuw nsw i32 %k0.prime.kdma.source1477, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1478, %kdma.intra, !dbg !14
  %k0.prime.1926 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.1927 = insertelement <2 x i64> poison, i64 %k0.prime.1926, i64 0, !dbg !14
  %k0.prime.1928 = bitcast <2 x i64> %k0.prime.1927 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1343 = shufflevector <4 x i32> %k0.prime.1928, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.1929 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1494 = add i32 %53, 512, !dbg !14
  %k0.prime.kdma.source1479 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1480 = lshr i64 %k0.prime.kdma.source1479, 4, !dbg !14
  %k0.prime.kdma.source1481 = and i64 %k0.prime.kdma.source1480, 8, !dbg !14
  %k0.prime.kdma.source1482 = and i64 %k0.prime.kdma.source1479, 123, !dbg !14
  %k0.prime.kdma.source1483 = lshr i64 %k0.prime.kdma.source1479, 2, !dbg !14
  %k0.prime.kdma.source1484 = and i64 %k0.prime.kdma.source1483, 4, !dbg !14
  %k0.prime.kdma.source1485 = add nuw nsw i64 %k0.prime.kdma.source1484, %k0.prime.kdma.source1479, !dbg !14
  %k0.prime.kdma.source1486 = and i64 %k0.prime.kdma.source1485, 4, !dbg !14
  %k0.prime.kdma.source1487 = or disjoint i64 %k0.prime.kdma.source1486, %k0.prime.kdma.source1482, !dbg !14
  %k0.prime.kdma.source1488 = shl nuw nsw i64 %k0.prime.kdma.source1487, 7, !dbg !14
  %k0.prime.kdma.source1489 = or disjoint i64 %k0.prime.kdma.source1488, %k0.prime.kdma.source1481, !dbg !14
  %k0.prime.kdma.source1490 = or disjoint i64 %k0.prime.kdma.source1489, %k0.prime.1923, !dbg !14
  %k0.prime.kdma.source1491 = trunc nuw nsw i64 %k0.prime.kdma.source1490 to i32, !dbg !14
  %k0.prime.kdma.source1492 = shl nuw nsw i32 %k0.prime.kdma.source1491, 1, !dbg !14
  %k0.prime.kdma.offset1493 = or disjoint i32 %k0.prime.kdma.source1492, %kdma.intra.hi, !dbg !14
  %k0.prime.1930 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1494, i32 %k0.prime.kdma.offset1493, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1506 = or disjoint i64 %k0.prime.kdma.source1475, 16, !dbg !14
  %k0.prime.kdma.source1507 = or disjoint i64 %k0.prime.kdma.source1506, %k0.prime.1923, !dbg !14
  %k0.prime.kdma.source1508 = trunc nuw nsw i64 %k0.prime.kdma.source1507 to i32, !dbg !14
  %k0.prime.kdma.source1509 = shl nuw nsw i32 %k0.prime.kdma.source1508, 1, !dbg !14
  %k0.prime.kdma.offset1510 = or disjoint i32 %k0.prime.kdma.source1509, %kdma.intra, !dbg !14
  %k0.prime.1931 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset1510, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1528 = add i32 %56, 512, !dbg !14
  %k0.prime.kdma.source1523 = or disjoint i64 %k0.prime.kdma.source1489, 16, !dbg !14
  %k0.prime.kdma.source1524 = or disjoint i64 %k0.prime.kdma.source1523, %k0.prime.1923, !dbg !14
  %k0.prime.kdma.source1525 = trunc nuw nsw i64 %k0.prime.kdma.source1524 to i32, !dbg !14
  %k0.prime.kdma.source1526 = shl nuw nsw i32 %k0.prime.kdma.source1525, 1, !dbg !14
  %k0.prime.kdma.offset1527 = or disjoint i32 %k0.prime.kdma.source1526, %kdma.intra.hi, !dbg !14
  %k0.prime.1932 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1528, i32 %k0.prime.kdma.offset1527, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1560 = add i32 %59, 512
  %kdma.lds1594 = add i32 %62, 512
  %113 = ptrtoint ptr addrspace(1) %2 to i64
  %114 = insertelement <2 x i64> poison, i64 %113, i64 0
  %115 = bitcast <2 x i64> %114 to <4 x i32>
  %dma.buffer.rsrc1327 = shufflevector <4 x i32> %115, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %53, 8192
  %vdb.dma.base1420 = add i32 %56, 8192
  br label %116, !dbg !12

116:                                              ; preds = %4, %k0.join
  %117 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1020, %k0.join ]
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1016, %k0.join ]
  %119 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1012, %k0.join ]
  %120 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1008, %k0.join ]
  %121 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %771, %k0.join ]
  %122 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1006, %k0.join ]
  %123 = phi i64 [ 0, %4 ], [ %928, %k0.join ]
  %124 = shl nuw nsw i64 %123, 14, !dbg !14
  %125 = add nuw nsw i64 %124, %k0.prime.1923, !dbg !14
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %126 = or disjoint i64 %125, 32, !dbg !14
  %kdma.source1540 = or disjoint i64 %126, %k0.prime.kdma.source1475, !dbg !14
  %kdma.source1541 = trunc nuw nsw i64 %kdma.source1540 to i32, !dbg !14
  %kdma.source1542 = shl nuw nsw i32 %kdma.source1541, 1, !dbg !14
  %kdma.offset1543 = or disjoint i32 %kdma.source1542, %kdma.intra, !dbg !14
  %127 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1543, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1556 = or disjoint i64 %126, %k0.prime.kdma.source1489, !dbg !14
  %kdma.source1557 = trunc nuw nsw i64 %kdma.source1556 to i32, !dbg !14
  %kdma.source1558 = shl nuw nsw i32 %kdma.source1557, 1, !dbg !14
  %kdma.offset1559 = or disjoint i32 %kdma.source1558, %kdma.intra.hi, !dbg !14
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1560, i32 %kdma.offset1559, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1573 = or disjoint i64 %126, %k0.prime.kdma.source1506, !dbg !14
  %kdma.source1574 = trunc nuw nsw i64 %kdma.source1573 to i32, !dbg !14
  %kdma.source1575 = shl nuw nsw i32 %kdma.source1574, 1, !dbg !14
  %kdma.offset1576 = or disjoint i32 %kdma.source1575, %kdma.intra, !dbg !14
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1576, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1590 = or disjoint i64 %126, %k0.prime.kdma.source1523, !dbg !14
  %kdma.source1591 = trunc nuw nsw i64 %kdma.source1590 to i32, !dbg !14
  %kdma.source1592 = shl nuw nsw i32 %kdma.source1591, 1, !dbg !14
  %kdma.offset1593 = or disjoint i32 %kdma.source1592, %kdma.intra.hi, !dbg !14
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1593, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %address.local526 = lshr i64 %address.local499, 2, !dbg !13
  %address.local529 = shl nuw nsw i64 %address.local499, 1, !dbg !13
  %address.local530 = and i64 %address.local529, 24, !dbg !13
  %address.local532 = shl i64 %address.local499, 62, !dbg !13
  %address.local533 = ashr exact i64 %address.local532, 57, !dbg !13
  %address.local534 = and i64 %address.local533, -64, !dbg !13
  %address.local537 = and i64 %address.local526, 12, !dbg !11
  %address.local539 = and i64 %address.local499, 63, !dbg !11
  %address.local540 = lshr i64 %address.local539, 5, !dbg !11
  %address.local541 = mul nuw nsw i64 %address.local540, 1016, !dbg !11
  %.idx1381 = mul nuw nsw i64 %address.local499, 320, !dbg !13
  %131 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1381, !dbg !13
  %.idx1382 = mul nsw i64 %address.local526, -1280, !dbg !13
  %132 = getelementptr i8, ptr addrspace(3) %131, i64 %.idx1382, !dbg !13
  %133 = getelementptr [2 x i8], ptr addrspace(3) %132, i64 %address.local530, !dbg !13
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local534, !dbg !13
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local537, !dbg !13
  %address.local543 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local541, !dbg !13
  %136 = add nsw i64 %.idx1382, %.idx1381, !dbg !13
  %.idx1853 = shl nuw nsw i64 %address.local530, 1, !dbg !13
  %137 = or disjoint i64 %136, %.idx1853, !dbg !13
  %138 = or disjoint i64 %address.local534, %address.local537, !dbg !13
  %139 = shl nsw i64 %138, 1, !dbg !13
  %address.local543.idx = mul nuw nsw i64 %address.local540, 2032, !dbg !13
  %140 = add nsw i64 %137, %address.local543.idx, !dbg !13
  %141 = add nsw i64 %140, %139, !dbg !13
  %142 = lshr i64 %141, 6, !dbg !13
  %143 = and i64 %142, 8, !dbg !13
  %144 = xor i64 %143, %141, !dbg !13
  %145 = sub i64 %144, %141, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local543, i64 %145, !dbg !13
  %146 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local554 = shl nuw nsw i32 %address.thread498, 1, !dbg !13
  %address.local555 = and i32 %address.local554, 24, !dbg !13
  %address.local556 = shl nuw nsw i32 %address.thread498, 8, !dbg !13
  %address.local557 = and i32 %address.local556, 768, !dbg !13
  %address.local558 = or disjoint i32 %address.local555, %address.local557, !dbg !13
  %address.local559 = shl nuw nsw i32 %address.thread498, 5, !dbg !13
  %address.local560 = and i32 %address.local559, 96, !dbg !13
  %address.local561 = or disjoint i32 %address.local558, %address.local560, !dbg !13
  %address.local562 = zext nneg i32 %address.local561 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %112) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local563 = lshr i32 %address.thread498, 1, !dbg !13
  %address.local564 = and i32 %address.local563, 1, !dbg !13
  %address.local565 = zext nneg i32 %address.local564 to i64, !dbg !13
  %address.local566 = add nsw i64 %address.base, %address.local565, !dbg !13
  %address.local567 = shl i64 %address.local566, 6, !dbg !13
  %address.local568 = sub i64 %address.local562, %address.local567, !dbg !13
  %149 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local568, !dbg !13
  %150 = getelementptr [2 x i8], ptr addrspace(3) %149, i64 %address.local537, !dbg !13
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local541, !dbg !13
  %address.local572 = getelementptr i8, ptr addrspace(3) %151, i64 64, !dbg !13
  %.idx18561859 = add i64 %address.local568, %address.local537, !dbg !13
  %152 = shl i64 %.idx18561859, 1, !dbg !13
  %153 = add nuw nsw i64 %address.local543.idx, 64, !dbg !13
  %154 = add i64 %153, %152, !dbg !13
  %155 = lshr i64 %154, 6, !dbg !13
  %156 = and i64 %155, 8, !dbg !13
  %157 = xor i64 %156, %154, !dbg !13
  %158 = sub i64 %157, %154, !dbg !13
  %kdma.swizzled1727 = getelementptr i8, ptr addrspace(3) %address.local572, i64 %158, !dbg !13
  %159 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1727, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 128, !dbg !13
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 4096, !dbg !13
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 4224, !dbg !13
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %148, <4 x float> %147) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1735 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 1024, !dbg !13
  %168 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1739 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 1152, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1736 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 5120, !dbg !13
  %171 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 5248, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %161, <4 x float> %170) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1737 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %174 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1738 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !13, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %163, <4 x float> zeroinitializer) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %165, <4 x float> %176) #8, !dbg !13
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !13
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %162, <4 x float> %178) #8, !dbg !13
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %164, <4 x float> zeroinitializer) #8, !dbg !13
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %166, <4 x float> %180) #8, !dbg !13
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !13
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %171, <4 x float> %182) #8, !dbg !13
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %174, <4 x float> zeroinitializer) #8, !dbg !13
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %175, <4 x float> %184) #8, !dbg !13
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !13
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %172, <4 x float> %186) #8, !dbg !13
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %146, <4 x float> zeroinitializer) #8, !dbg !13
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %148, <4 x float> %188) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %161, <4 x float> %190) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %163, <4 x float> zeroinitializer) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %165, <4 x float> %192) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %194) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %164, <4 x float> zeroinitializer) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %166, <4 x float> %196) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %171, <4 x float> %198) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %174, <4 x float> zeroinitializer) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %175, <4 x float> %200) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %172, <4 x float> %202) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %204 = or disjoint i64 %125, 64, !dbg !14
  %kdma.source1606 = or disjoint i64 %204, %k0.prime.kdma.source1475, !dbg !14
  %kdma.source1607 = trunc nuw nsw i64 %kdma.source1606 to i32, !dbg !14
  %kdma.source1608 = shl nuw nsw i32 %kdma.source1607, 1, !dbg !14
  %kdma.offset1609 = or disjoint i32 %kdma.source1608, %kdma.intra, !dbg !14
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %kdma.offset1609, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1622 = or disjoint i64 %204, %k0.prime.kdma.source1489, !dbg !14
  %kdma.source1623 = trunc nuw nsw i64 %kdma.source1622 to i32, !dbg !14
  %kdma.source1624 = shl nuw nsw i32 %kdma.source1623, 1, !dbg !14
  %kdma.offset1625 = or disjoint i32 %kdma.source1624, %kdma.intra.hi, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1494, i32 %kdma.offset1625, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1639 = or disjoint i64 %204, %k0.prime.kdma.source1506, !dbg !14
  %kdma.source1640 = trunc nuw nsw i64 %kdma.source1639 to i32, !dbg !14
  %kdma.source1641 = shl nuw nsw i32 %kdma.source1640, 1, !dbg !14
  %kdma.offset1642 = or disjoint i32 %kdma.source1641, %kdma.intra, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1642, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1656 = or disjoint i64 %204, %k0.prime.kdma.source1523, !dbg !14
  %kdma.source1657 = trunc nuw nsw i64 %kdma.source1656 to i32, !dbg !14
  %kdma.source1658 = shl nuw nsw i32 %kdma.source1657, 1, !dbg !14
  %kdma.offset1659 = or disjoint i32 %kdma.source1658, %kdma.intra.hi, !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1528, i32 %kdma.offset1659, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.gep1757 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1761 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1758 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1762 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1759 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 8192, !dbg !13
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1763 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 8320, !dbg !13
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1760 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 12288, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1764 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 12416, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1765 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %209, <4 x float> %167) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1766 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %211, <4 x float> %219) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1767 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 9216, !dbg !13
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1767, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 9344, !dbg !13
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %213, <4 x float> %173) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 13312, !dbg !13
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1727, i64 13440, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !13, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %215, <4 x float> %225) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %210, <4 x float> %177) #8, !dbg !13
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %212, <4 x float> %229) #8, !dbg !13
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %214, <4 x float> %179) #8, !dbg !13
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %216, <4 x float> %231) #8, !dbg !13
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %217, <4 x float> %181) #8, !dbg !13
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %220, <4 x float> %233) #8, !dbg !13
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %223, <4 x float> %183) #8, !dbg !13
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %226, <4 x float> %235) #8, !dbg !13
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %218, <4 x float> %185) #8, !dbg !13
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %221, <4 x float> %237) #8, !dbg !13
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %224, <4 x float> %187) #8, !dbg !13
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %227, <4 x float> %239) #8, !dbg !13
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %209, <4 x float> %189) #8, !dbg !13
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %211, <4 x float> %241) #8, !dbg !13
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %213, <4 x float> %191) #8, !dbg !13
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %215, <4 x float> %243) #8, !dbg !13
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %210, <4 x float> %193) #8, !dbg !13
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %212, <4 x float> %245) #8, !dbg !13
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %214, <4 x float> %195) #8, !dbg !13
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %216, <4 x float> %247) #8, !dbg !13
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %217, <4 x float> %197) #8, !dbg !13
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %220, <4 x float> %249) #8, !dbg !13
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %223, <4 x float> %199) #8, !dbg !13
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %226, <4 x float> %251) #8, !dbg !13
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %218, <4 x float> %201) #8, !dbg !13
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %221, <4 x float> %253) #8, !dbg !13
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %224, <4 x float> %203) #8, !dbg !13
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %227, <4 x float> %255) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %257 = or disjoint i64 %125, 96, !dbg !14
  %kdma.source1672 = or disjoint i64 %257, %k0.prime.kdma.source1475, !dbg !14
  %kdma.source1673 = trunc nuw nsw i64 %kdma.source1672 to i32, !dbg !14
  %kdma.source1674 = shl nuw nsw i32 %kdma.source1673, 1, !dbg !14
  %kdma.offset1675 = or disjoint i32 %kdma.source1674, %kdma.intra, !dbg !14
  %258 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1675, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1688 = or disjoint i64 %257, %k0.prime.kdma.source1489, !dbg !14
  %kdma.source1689 = trunc nuw nsw i64 %kdma.source1688 to i32, !dbg !14
  %kdma.source1690 = shl nuw nsw i32 %kdma.source1689, 1, !dbg !14
  %kdma.offset1691 = or disjoint i32 %kdma.source1690, %kdma.intra.hi, !dbg !14
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1560, i32 %kdma.offset1691, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1705 = or disjoint i64 %257, %k0.prime.kdma.source1506, !dbg !14
  %kdma.source1706 = trunc nuw nsw i64 %kdma.source1705 to i32, !dbg !14
  %kdma.source1707 = shl nuw nsw i32 %kdma.source1706, 1, !dbg !14
  %kdma.offset1708 = or disjoint i32 %kdma.source1707, %kdma.intra, !dbg !14
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1708, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %kdma.source1722 = or disjoint i64 %257, %k0.prime.kdma.source1523, !dbg !14
  %kdma.source1723 = trunc nuw nsw i64 %kdma.source1722 to i32, !dbg !14
  %kdma.source1724 = shl nuw nsw i32 %kdma.source1723, 1, !dbg !14
  %kdma.offset1725 = or disjoint i32 %kdma.source1724, %kdma.intra.hi, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1594, i32 %kdma.offset1725, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14
  %262 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !13, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !13, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1727, align 8, !dbg !13, !frisk.k_lds_pair !32
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !13, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !13, !frisk.k_lds_pair !33
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !13, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !13, !frisk.k_lds_pair !34
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %262, <4 x float> %222) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !13, !frisk.k_lds_pair !35
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %264, <4 x float> %272) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !13, !frisk.k_lds_pair !36
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %266, <4 x float> %228) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !13, !frisk.k_lds_pair !37
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !13, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %268, <4 x float> %278) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %263, <4 x float> %230) #8, !dbg !13
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %265, <4 x float> %282) #8, !dbg !13
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %267, <4 x float> %232) #8, !dbg !13
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %269, <4 x float> %284) #8, !dbg !13
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %270, <4 x float> %234) #8, !dbg !13
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %273, <4 x float> %286) #8, !dbg !13
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %276, <4 x float> %236) #8, !dbg !13
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %279, <4 x float> %288) #8, !dbg !13
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %271, <4 x float> %238) #8, !dbg !13
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %274, <4 x float> %290) #8, !dbg !13
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %277, <4 x float> %240) #8, !dbg !13
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %280, <4 x float> %292) #8, !dbg !13
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %262, <4 x float> %242) #8, !dbg !13
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %264, <4 x float> %294) #8, !dbg !13
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %266, <4 x float> %244) #8, !dbg !13
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %268, <4 x float> %296) #8, !dbg !13
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %263, <4 x float> %246) #8, !dbg !13
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %265, <4 x float> %298) #8, !dbg !13
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %267, <4 x float> %248) #8, !dbg !13
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %269, <4 x float> %300) #8, !dbg !13
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %270, <4 x float> %250) #8, !dbg !13
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %273, <4 x float> %302) #8, !dbg !13
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %276, <4 x float> %252) #8, !dbg !13
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %279, <4 x float> %304) #8, !dbg !13
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %271, <4 x float> %254) #8, !dbg !13
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %274, <4 x float> %306) #8, !dbg !13
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %277, <4 x float> %256) #8, !dbg !13
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %280, <4 x float> %308) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %address.local901 = shl nuw nsw i64 %address.local499, 5, !dbg !38
  %address.local902 = and i64 %address.local901, 8064, !dbg !38
  %address.local903 = shl nuw nsw i64 %address.local499, 3, !dbg !38
  %address.local904 = and i64 %address.local903, 24, !dbg !38
  %address.local905 = or disjoint i64 %address.local902, %address.local904, !dbg !38
  %.idx13911392 = or disjoint i64 %address.local905, %125, !dbg !38
  %.idx13911392.tr = trunc nuw nsw i64 %.idx13911392 to i32, !dbg !38
  %dma.buffer.offset1326 = shl nuw nsw i32 %.idx13911392.tr, 1, !dbg !38
  %310 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1326, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %address.local911 = or disjoint i64 %address.local905, 8192, !dbg !38
  %.idx13931394 = or disjoint i64 %address.local911, %125, !dbg !38
  %.idx13931394.tr = trunc nuw nsw i64 %.idx13931394 to i32, !dbg !38
  %dma.buffer.offset1324 = shl nuw nsw i32 %.idx13931394.tr, 1, !dbg !38
  %311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1324, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !13, !frisk.k_tail_pair !15
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !13, !frisk.k_tail_pair !15
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !13, !frisk.k_tail_pair !16
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !13, !frisk.k_tail_pair !16
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !13, !frisk.k_tail_pair !17
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !13, !frisk.k_tail_pair !17
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !13, !frisk.k_tail_pair !18
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !13, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %312, <4 x float> %275) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %314, <4 x float> %320) #8, !dbg !13
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %316, <4 x float> %281) #8, !dbg !13
  %323 = extractelement <4 x float> %321, i64 0, !dbg !39
  %324 = tail call float @llvm.maxnum.f32(float %323, float -inf), !dbg !7, !frisk.local_reduction !5
  %325 = extractelement <4 x float> %321, i64 1, !dbg !39
  %326 = extractelement <4 x float> %321, i64 2, !dbg !39
  %327 = extractelement <4 x float> %321, i64 3, !dbg !39
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %318, <4 x float> %322) #8, !dbg !13
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !13, !frisk.k_tail_pair !19
  %330 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !13, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %313, <4 x float> %283) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39, !frisk.k_tail_prefetch !5
  %332 = extractelement <4 x float> %328, i64 0, !dbg !39
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %324, float %332), !dbg !7, !frisk.local_reduction !5
  %333 = extractelement <4 x float> %328, i64 1, !dbg !39
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %325, float %333), !dbg !7, !frisk.local_reduction !5
  %334 = extractelement <4 x float> %328, i64 2, !dbg !39
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %326, float %334), !dbg !7, !frisk.local_reduction !5
  %335 = extractelement <4 x float> %328, i64 3, !dbg !39
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %327, float %335), !dbg !7, !frisk.local_reduction !5
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !13, !frisk.k_tail_pair !20
  %337 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %338 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %315, <4 x float> %331) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1767, align 8, !dbg !13, !frisk.k_tail_pair !21
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %317, <4 x float> %285) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39, !frisk.k_tail_prefetch !5
  %342 = extractelement <4 x float> %338, i64 0, !dbg !39
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %342), !dbg !7, !frisk.local_reduction !5
  %343 = extractelement <4 x float> %338, i64 1, !dbg !39
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %343), !dbg !7, !frisk.local_reduction !5
  %344 = extractelement <4 x float> %338, i64 2, !dbg !39
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %344), !dbg !7, !frisk.local_reduction !5
  %345 = extractelement <4 x float> %338, i64 3, !dbg !39
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %345), !dbg !7, !frisk.local_reduction !5
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !13, !frisk.k_tail_pair !22
  %347 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %319, <4 x float> %341) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %329, <4 x float> %287) #8, !dbg !13
  %350 = extractelement <4 x float> %348, i64 0, !dbg !39
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %350), !dbg !7, !frisk.local_reduction !5
  %351 = extractelement <4 x float> %348, i64 1, !dbg !39
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %351), !dbg !7, !frisk.local_reduction !5
  %352 = extractelement <4 x float> %348, i64 2, !dbg !39
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %352), !dbg !7, !frisk.local_reduction !5
  %353 = extractelement <4 x float> %348, i64 3, !dbg !39
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %353), !dbg !7, !frisk.local_reduction !5
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %336, <4 x float> %349) #8, !dbg !13
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %339, <4 x float> %289) #8, !dbg !13
  %356 = extractelement <4 x float> %354, i64 0, !dbg !39
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %356), !dbg !7, !frisk.local_reduction !5
  %357 = extractelement <4 x float> %354, i64 1, !dbg !39
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %357), !dbg !7, !frisk.local_reduction !5
  %358 = extractelement <4 x float> %354, i64 2, !dbg !39
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %358), !dbg !7, !frisk.local_reduction !5
  %359 = extractelement <4 x float> %354, i64 3, !dbg !39
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %359), !dbg !7, !frisk.local_reduction !5
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %346, <4 x float> %355) #8, !dbg !13
  %361 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %330, <4 x float> %291) #8, !dbg !13
  %362 = extractelement <4 x float> %360, i64 0, !dbg !39
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %362), !dbg !7, !frisk.local_reduction !5
  %363 = extractelement <4 x float> %360, i64 1, !dbg !39
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %363), !dbg !7, !frisk.local_reduction !5
  %364 = extractelement <4 x float> %360, i64 2, !dbg !39
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %364), !dbg !7, !frisk.local_reduction !5
  %365 = extractelement <4 x float> %360, i64 3, !dbg !39
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %365), !dbg !7, !frisk.local_reduction !5
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %337, <4 x float> %361) #8, !dbg !13
  %367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %340, <4 x float> %293) #8, !dbg !13
  %368 = extractelement <4 x float> %366, i64 0, !dbg !39
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %368), !dbg !7, !frisk.local_reduction !5
  %369 = extractelement <4 x float> %366, i64 1, !dbg !39
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %369), !dbg !7, !frisk.local_reduction !5
  %370 = extractelement <4 x float> %366, i64 2, !dbg !39
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %370), !dbg !7, !frisk.local_reduction !5
  %371 = extractelement <4 x float> %366, i64 3, !dbg !39
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %371), !dbg !7, !frisk.local_reduction !5
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %347, <4 x float> %367) #8, !dbg !13
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %312, <4 x float> %295) #8, !dbg !13
  %374 = extractelement <4 x float> %372, i64 0, !dbg !39
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %374), !dbg !7, !frisk.local_reduction !5
  %375 = extractelement <4 x float> %372, i64 1, !dbg !39
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %375), !dbg !7, !frisk.local_reduction !5
  %376 = extractelement <4 x float> %372, i64 2, !dbg !39
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %376), !dbg !7, !frisk.local_reduction !5
  %377 = extractelement <4 x float> %372, i64 3, !dbg !39
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %377), !dbg !7, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !7, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !7, !frisk.local_reduction !5
  %378 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !7, !frisk.local_reduction !5
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %314, <4 x float> %373) #8, !dbg !13
  %380 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %316, <4 x float> %297) #8, !dbg !13
  %381 = extractelement <4 x float> %379, i64 0, !dbg !39
  %382 = extractelement <4 x float> %379, i64 1, !dbg !39
  %383 = extractelement <4 x float> %379, i64 2, !dbg !39
  %384 = extractelement <4 x float> %379, i64 3, !dbg !39
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %318, <4 x float> %380) #8, !dbg !13
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %313, <4 x float> %299) #8, !dbg !13
  %387 = extractelement <4 x float> %385, i64 0, !dbg !39
  %388 = extractelement <4 x float> %385, i64 1, !dbg !39
  %389 = extractelement <4 x float> %385, i64 2, !dbg !39
  %390 = extractelement <4 x float> %385, i64 3, !dbg !39
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %315, <4 x float> %386) #8, !dbg !13
  %392 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %317, <4 x float> %301) #8, !dbg !13
  %393 = extractelement <4 x float> %391, i64 0, !dbg !39
  %394 = extractelement <4 x float> %391, i64 1, !dbg !39
  %395 = extractelement <4 x float> %391, i64 2, !dbg !39
  %396 = extractelement <4 x float> %391, i64 3, !dbg !39
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %319, <4 x float> %392) #8, !dbg !13
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %329, <4 x float> %303) #8, !dbg !13
  %399 = extractelement <4 x float> %397, i64 0, !dbg !39
  %400 = extractelement <4 x float> %397, i64 1, !dbg !39
  %401 = extractelement <4 x float> %397, i64 2, !dbg !39
  %402 = extractelement <4 x float> %397, i64 3, !dbg !39
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %336, <4 x float> %398) #8, !dbg !13
  %404 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %339, <4 x float> %305) #8, !dbg !13
  %405 = extractelement <4 x float> %403, i64 0, !dbg !39
  %406 = extractelement <4 x float> %403, i64 1, !dbg !39
  %407 = extractelement <4 x float> %403, i64 2, !dbg !39
  %408 = extractelement <4 x float> %403, i64 3, !dbg !39
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %346, <4 x float> %404) #8, !dbg !13
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %330, <4 x float> %307) #8, !dbg !13
  %411 = extractelement <4 x float> %409, i64 0, !dbg !39
  %412 = extractelement <4 x float> %409, i64 1, !dbg !39
  %413 = extractelement <4 x float> %409, i64 2, !dbg !39
  %414 = extractelement <4 x float> %409, i64 3, !dbg !39
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %337, <4 x float> %410) #8, !dbg !13
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %340, <4 x float> %309) #8, !dbg !13
  %417 = extractelement <4 x float> %415, i64 0, !dbg !39
  %418 = extractelement <4 x float> %415, i64 1, !dbg !39
  %419 = extractelement <4 x float> %415, i64 2, !dbg !39
  %420 = extractelement <4 x float> %415, i64 3, !dbg !39
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %347, <4 x float> %416) #8, !dbg !13
  %422 = extractelement <4 x float> %421, i64 0, !dbg !39
  %423 = extractelement <4 x float> %421, i64 1, !dbg !39
  %424 = extractelement <4 x float> %421, i64 2, !dbg !39
  %425 = extractelement <4 x float> %421, i64 3, !dbg !39
  %426 = bitcast float %378 to i32, !dbg !7
  %427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %426), !dbg !7
  %428 = bitcast i32 %427 to float, !dbg !7
  %429 = tail call float @llvm.maxnum.f32(float %378, float %428), !dbg !7
  %430 = bitcast float %429 to i32, !dbg !7
  %431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %430), !dbg !7
  %432 = bitcast i32 %431 to float, !dbg !7
  %433 = tail call float @llvm.maxnum.f32(float %429, float %432), !dbg !7
  %434 = tail call float @llvm.maxnum.f32(float %381, float -inf), !dbg !7, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %434, float %387), !dbg !7, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %382, float %388), !dbg !7, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %383, float %389), !dbg !7, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %384, float %390), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %393), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %394), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %395), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %396), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %399), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %400), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %401), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %402), !dbg !7, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %405), !dbg !7, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %406), !dbg !7, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %407), !dbg !7, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %408), !dbg !7, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %411), !dbg !7, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %412), !dbg !7, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %413), !dbg !7, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %414), !dbg !7, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %417), !dbg !7, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %418), !dbg !7, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %419), !dbg !7, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %420), !dbg !7, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %422), !dbg !7, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %423), !dbg !7, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %424), !dbg !7, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %425), !dbg !7, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !7, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !7, !frisk.local_reduction !5
  %435 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !7, !frisk.local_reduction !5
  %436 = bitcast float %435 to i32, !dbg !7
  %437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %436), !dbg !7
  %438 = bitcast i32 %437 to float, !dbg !7
  %439 = tail call float @llvm.maxnum.f32(float %435, float %438), !dbg !7
  %440 = bitcast float %439 to i32, !dbg !7
  %441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %440), !dbg !7
  %442 = bitcast i32 %441 to float, !dbg !7
  %443 = tail call float @llvm.maxnum.f32(float %439, float %442), !dbg !7
  %444 = extractvalue [2 x <1 x float>] %120, 0, !dbg !7
  %445 = insertelement <1 x float> poison, float %433, i64 0, !dbg !7
  %446 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %445, <1 x float> %444), !dbg !7
  %447 = extractvalue [2 x <1 x float>] %120, 1, !dbg !7
  %448 = insertelement <1 x float> poison, float %443, i64 0, !dbg !7
  %449 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %448, <1 x float> %447), !dbg !7
  %450 = shufflevector <1 x float> %446, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %451 = shufflevector <1 x float> %449, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %452 = fsub <1 x float> %444, %446, !dbg !41
  %453 = extractelement <1 x float> %452, i64 0, !dbg !42
  %454 = tail call float @llvm.amdgcn.exp2.f32(float %453), !dbg !42
  %455 = insertelement <4 x float> poison, float %454, i64 0, !dbg !43
  %456 = shufflevector <4 x float> %455, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %457 = extractvalue [2 x <1 x float>] %121, 0, !dbg !44
  %458 = extractelement <1 x float> %457, i64 0, !dbg !44
  %459 = fmul float %458, %454, !dbg !44
  %460 = fsub <1 x float> %447, %449, !dbg !41
  %461 = extractelement <1 x float> %460, i64 0, !dbg !42
  %462 = tail call float @llvm.amdgcn.exp2.f32(float %461), !dbg !42
  %463 = insertelement <4 x float> poison, float %462, i64 0, !dbg !43
  %464 = shufflevector <4 x float> %463, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %465 = extractvalue [2 x <1 x float>] %121, 1, !dbg !44
  %466 = extractelement <1 x float> %465, i64 0, !dbg !44
  %467 = fmul float %466, %462, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = or disjoint i64 %address.local905, %126, !dbg !38
  %vdb.address1407 = trunc nuw nsw i64 %vdb.address to i32, !dbg !38
  %vdb.address1408 = shl nuw nsw i32 %vdb.address1407, 1, !dbg !38
  %vdb.address1413 = or disjoint i64 %address.local911, %126, !dbg !38
  %vdb.address1414 = trunc nuw nsw i64 %vdb.address1413 to i32, !dbg !38
  %vdb.address1415 = shl nuw nsw i32 %vdb.address1414, 1, !dbg !38
  %468 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1408, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %469 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1415, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %address.local913 = shl nuw nsw i32 %address.thread498, 3, !dbg !45
  %address.local914 = and i32 %address.local913, 120, !dbg !45
  %address.local916 = and i32 %address.local559, 1536, !dbg !45
  %address.local917 = or disjoint i32 %address.local914, %address.local916, !dbg !45
  %address.local918 = zext nneg i32 %address.local917 to i64, !dbg !45
  %address.local919 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local918, !dbg !45
  %470 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %471 = bitcast <4 x i32> %470 to <8 x half>, !dbg !45
  %address.local920 = and i64 %address.local499, 60, !dbg !45
  %address.local921 = add nuw nsw i64 %address.local920, %address.local499, !dbg !45
  %address.local923.idx = shl nuw nsw i64 %address.local921, 4, !dbg !45
  %address.local923 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local923.idx, !dbg !45
  %address.local924 = lshr i64 %address.local539, 4, !dbg !45
  %address.local926.idx = mul nuw nsw i64 %address.local924, 768, !dbg !45
  %address.local926 = getelementptr i8, ptr addrspace(3) %address.local923, i64 %address.local926.idx, !dbg !45
  %address.local928.idx = mul nsw i64 %address.local526, -64, !dbg !45
  %address.local928 = getelementptr i8, ptr addrspace(3) %address.local926, i64 %address.local928.idx, !dbg !45
  %address.local929 = getelementptr i8, ptr addrspace(3) %address.local928, i64 256, !dbg !45
  %472 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %473 = bitcast <4 x i32> %472 to <8 x half>, !dbg !45
  %address.local933 = getelementptr i8, ptr addrspace(3) %address.local928, i64 512, !dbg !45
  %474 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %475 = bitcast <4 x i32> %474 to <8 x half>, !dbg !45
  %address.local937 = getelementptr i8, ptr addrspace(3) %address.local928, i64 768, !dbg !45
  %476 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %477 = bitcast <4 x i32> %476 to <8 x half>, !dbg !45
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local919, i64 4096, !dbg !45
  %478 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %479 = bitcast <4 x i32> %478 to <8 x half>, !dbg !45
  %address.local943 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4352, !dbg !45
  %480 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %481 = bitcast <4 x i32> %480 to <8 x half>, !dbg !45
  %address.local947 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4608, !dbg !45
  %482 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %483 = bitcast <4 x i32> %482 to <8 x half>, !dbg !45
  %address.local951 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4864, !dbg !45
  %484 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %485 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %486 = extractvalue [2 x <8 x float>] %122, 0, !dbg !43
  %487 = shufflevector <8 x float> %486, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %488 = fmul <4 x float> %487, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %489 = fsub <4 x float> %321, %450, !dbg !40
  %490 = extractelement <4 x float> %489, i64 0, !dbg !47
  %491 = tail call float @llvm.amdgcn.exp2.f32(float %490), !dbg !47
  %492 = insertelement <4 x float> poison, float %491, i64 0, !dbg !48
  %493 = extractelement <4 x float> %489, i64 1, !dbg !47
  %494 = tail call float @llvm.amdgcn.exp2.f32(float %493), !dbg !47
  %495 = insertelement <4 x float> %492, float %494, i64 1, !dbg !48
  %496 = extractelement <4 x float> %489, i64 2, !dbg !47
  %497 = tail call float @llvm.amdgcn.exp2.f32(float %496), !dbg !47
  %498 = insertelement <4 x float> %495, float %497, i64 2, !dbg !48
  %499 = extractelement <4 x float> %489, i64 3, !dbg !47
  %500 = tail call float @llvm.amdgcn.exp2.f32(float %499), !dbg !47
  %501 = insertelement <4 x float> %498, float %500, i64 3, !dbg !48
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %501) #8, !dbg !48, !frisk.register_lifetimes !5
  %502 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !48
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %502) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %503 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %485, <4 x float> %488) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %504 = shufflevector <8 x half> %471, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %505 = shufflevector <8 x float> %486, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %506 = fmul <4 x float> %505, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %504, <4 x float> %506) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %508 = extractvalue [2 x <8 x float>] %122, 1, !dbg !43
  %509 = shufflevector <8 x float> %508, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %510 = fmul <4 x float> %509, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %511 = fsub <4 x float> %379, %451, !dbg !40
  %512 = extractelement <4 x float> %511, i64 0, !dbg !47
  %513 = tail call float @llvm.amdgcn.exp2.f32(float %512), !dbg !47
  %514 = insertelement <4 x float> poison, float %513, i64 0, !dbg !48
  %515 = extractelement <4 x float> %511, i64 1, !dbg !47
  %516 = tail call float @llvm.amdgcn.exp2.f32(float %515), !dbg !47
  %517 = insertelement <4 x float> %514, float %516, i64 1, !dbg !48
  %518 = extractelement <4 x float> %511, i64 2, !dbg !47
  %519 = tail call float @llvm.amdgcn.exp2.f32(float %518), !dbg !47
  %520 = insertelement <4 x float> %517, float %519, i64 2, !dbg !48
  %521 = extractelement <4 x float> %511, i64 3, !dbg !47
  %522 = tail call float @llvm.amdgcn.exp2.f32(float %521), !dbg !47
  %523 = insertelement <4 x float> %520, float %522, i64 3, !dbg !48
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %523) #8, !dbg !48, !frisk.register_lifetimes !5
  %524 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !48
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %524) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %525 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %485, <4 x float> %510) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %526 = shufflevector <8 x float> %508, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %527 = fmul <4 x float> %526, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %528 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %504, <4 x float> %527) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %529 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %530 = fsub <4 x float> %328, %450, !dbg !40
  %531 = extractelement <4 x float> %530, i64 0, !dbg !47
  %532 = tail call float @llvm.amdgcn.exp2.f32(float %531), !dbg !47
  %local.reduce89 = fadd float %491, %532, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %533 = insertelement <4 x float> poison, float %532, i64 0, !dbg !48
  %534 = extractelement <4 x float> %530, i64 1, !dbg !47
  %535 = tail call float @llvm.amdgcn.exp2.f32(float %534), !dbg !47
  %local.reduce90 = fadd float %494, %535, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1860 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %536 = insertelement <4 x float> %533, float %535, i64 1, !dbg !48
  %537 = extractelement <4 x float> %530, i64 2, !dbg !47
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %537), !dbg !47
  %local.reduce91 = fadd float %497, %538, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1861 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %539 = insertelement <4 x float> %536, float %538, i64 2, !dbg !48
  %540 = extractelement <4 x float> %530, i64 3, !dbg !47
  %541 = tail call float @llvm.amdgcn.exp2.f32(float %540), !dbg !47
  %local.reduce92 = fadd float %500, %541, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1862 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %542 = insertelement <4 x float> %539, float %541, i64 3, !dbg !48
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #8, !dbg !48, !frisk.register_lifetimes !5
  %543 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !48
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %529, <4 x float> %503) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %545 = shufflevector <8 x half> %473, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %545, <4 x float> %507) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %547 = fsub <4 x float> %385, %451, !dbg !40
  %548 = extractelement <4 x float> %547, i64 0, !dbg !47
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %548), !dbg !47
  %local.reduce = fadd float %513, %549, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1863 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %550 = insertelement <4 x float> poison, float %549, i64 0, !dbg !48
  %551 = extractelement <4 x float> %547, i64 1, !dbg !47
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %551), !dbg !47
  %local.reduce59 = fadd float %516, %552, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1864 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %553 = insertelement <4 x float> %550, float %552, i64 1, !dbg !48
  %554 = extractelement <4 x float> %547, i64 2, !dbg !47
  %555 = tail call float @llvm.amdgcn.exp2.f32(float %554), !dbg !47
  %local.reduce60 = fadd float %519, %555, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1865 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %556 = insertelement <4 x float> %553, float %555, i64 2, !dbg !48
  %557 = extractelement <4 x float> %547, i64 3, !dbg !47
  %558 = tail call float @llvm.amdgcn.exp2.f32(float %557), !dbg !47
  %local.reduce61 = fadd float %522, %558, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1866 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %559 = insertelement <4 x float> %556, float %558, i64 3, !dbg !48
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %559) #8, !dbg !48, !frisk.register_lifetimes !5
  %560 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !48
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %560) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %561 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %529, <4 x float> %525) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %562 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %545, <4 x float> %528) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %563 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %564 = fsub <4 x float> %338, %450, !dbg !40
  %565 = extractelement <4 x float> %564, i64 0, !dbg !47
  %566 = tail call float @llvm.amdgcn.exp2.f32(float %565), !dbg !47
  %local.reduce93 = fadd float %566, %k0.sum.anchor, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1867 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %567 = insertelement <4 x float> poison, float %566, i64 0, !dbg !48
  %568 = extractelement <4 x float> %564, i64 1, !dbg !47
  %569 = tail call float @llvm.amdgcn.exp2.f32(float %568), !dbg !47
  %local.reduce94 = fadd float %569, %k0.sum.anchor1860, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1868 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %570 = insertelement <4 x float> %567, float %569, i64 1, !dbg !48
  %571 = extractelement <4 x float> %564, i64 2, !dbg !47
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %571), !dbg !47
  %local.reduce95 = fadd float %572, %k0.sum.anchor1861, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1869 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %573 = insertelement <4 x float> %570, float %572, i64 2, !dbg !48
  %574 = extractelement <4 x float> %564, i64 3, !dbg !47
  %575 = tail call float @llvm.amdgcn.exp2.f32(float %574), !dbg !47
  %local.reduce96 = fadd float %575, %k0.sum.anchor1862, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1870 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %576 = insertelement <4 x float> %573, float %575, i64 3, !dbg !48
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %576) #8, !dbg !48, !frisk.register_lifetimes !5
  %577 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !48
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %577) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %578 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %563, <4 x float> %544) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %579 = shufflevector <8 x half> %475, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %580 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %579, <4 x float> %546) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %581 = fsub <4 x float> %391, %451, !dbg !40
  %582 = extractelement <4 x float> %581, i64 0, !dbg !47
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !47
  %local.reduce62 = fadd float %583, %k0.sum.anchor1863, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1871 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %584 = insertelement <4 x float> poison, float %583, i64 0, !dbg !48
  %585 = extractelement <4 x float> %581, i64 1, !dbg !47
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %585), !dbg !47
  %local.reduce63 = fadd float %586, %k0.sum.anchor1864, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1872 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %587 = insertelement <4 x float> %584, float %586, i64 1, !dbg !48
  %588 = extractelement <4 x float> %581, i64 2, !dbg !47
  %589 = tail call float @llvm.amdgcn.exp2.f32(float %588), !dbg !47
  %local.reduce64 = fadd float %589, %k0.sum.anchor1865, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1873 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %590 = insertelement <4 x float> %587, float %589, i64 2, !dbg !48
  %591 = extractelement <4 x float> %581, i64 3, !dbg !47
  %592 = tail call float @llvm.amdgcn.exp2.f32(float %591), !dbg !47
  %local.reduce65 = fadd float %592, %k0.sum.anchor1866, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1874 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %593 = insertelement <4 x float> %590, float %592, i64 3, !dbg !48
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %593) #8, !dbg !48, !frisk.register_lifetimes !5
  %594 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !48
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %594) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %595 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %563, <4 x float> %561) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %596 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %579, <4 x float> %562) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %597 = shufflevector <8 x half> %477, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %598 = fsub <4 x float> %348, %450, !dbg !40
  %599 = extractelement <4 x float> %598, i64 0, !dbg !47
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %599), !dbg !47
  %local.reduce97 = fadd float %600, %k0.sum.anchor1867, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1875 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> poison, float %600, i64 0, !dbg !48
  %602 = extractelement <4 x float> %598, i64 1, !dbg !47
  %603 = tail call float @llvm.amdgcn.exp2.f32(float %602), !dbg !47
  %local.reduce98 = fadd float %603, %k0.sum.anchor1868, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1876 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %604 = insertelement <4 x float> %601, float %603, i64 1, !dbg !48
  %605 = extractelement <4 x float> %598, i64 2, !dbg !47
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %605), !dbg !47
  %local.reduce99 = fadd float %606, %k0.sum.anchor1869, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1877 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %607 = insertelement <4 x float> %604, float %606, i64 2, !dbg !48
  %608 = extractelement <4 x float> %598, i64 3, !dbg !47
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %608), !dbg !47
  %local.reduce100 = fadd float %609, %k0.sum.anchor1870, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1878 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %610 = insertelement <4 x float> %607, float %609, i64 3, !dbg !48
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %610) #8, !dbg !48, !frisk.register_lifetimes !5
  %611 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !48
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %611) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %612 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %597, <4 x float> %578) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %613 = shufflevector <8 x half> %477, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %613, <4 x float> %580) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %615 = fsub <4 x float> %397, %451, !dbg !40
  %616 = extractelement <4 x float> %615, i64 0, !dbg !47
  %617 = tail call float @llvm.amdgcn.exp2.f32(float %616), !dbg !47
  %local.reduce66 = fadd float %617, %k0.sum.anchor1871, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1879 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %618 = insertelement <4 x float> poison, float %617, i64 0, !dbg !48
  %619 = extractelement <4 x float> %615, i64 1, !dbg !47
  %620 = tail call float @llvm.amdgcn.exp2.f32(float %619), !dbg !47
  %local.reduce67 = fadd float %620, %k0.sum.anchor1872, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1880 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %621 = insertelement <4 x float> %618, float %620, i64 1, !dbg !48
  %622 = extractelement <4 x float> %615, i64 2, !dbg !47
  %623 = tail call float @llvm.amdgcn.exp2.f32(float %622), !dbg !47
  %local.reduce68 = fadd float %623, %k0.sum.anchor1873, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1881 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %624 = insertelement <4 x float> %621, float %623, i64 2, !dbg !48
  %625 = extractelement <4 x float> %615, i64 3, !dbg !47
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %625), !dbg !47
  %local.reduce69 = fadd float %626, %k0.sum.anchor1874, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1882 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %627 = insertelement <4 x float> %624, float %626, i64 3, !dbg !48
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %627) #8, !dbg !48, !frisk.register_lifetimes !5
  %628 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !48
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %628) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %629 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %597, <4 x float> %595) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %630 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %613, <4 x float> %596) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %631 = shufflevector <8 x half> %479, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %632 = fsub <4 x float> %354, %450, !dbg !40
  %633 = extractelement <4 x float> %632, i64 0, !dbg !47
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %633), !dbg !47
  %local.reduce101 = fadd float %634, %k0.sum.anchor1875, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1883 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %635 = insertelement <4 x float> poison, float %634, i64 0, !dbg !48
  %636 = extractelement <4 x float> %632, i64 1, !dbg !47
  %637 = tail call float @llvm.amdgcn.exp2.f32(float %636), !dbg !47
  %local.reduce102 = fadd float %637, %k0.sum.anchor1876, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1884 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %638 = insertelement <4 x float> %635, float %637, i64 1, !dbg !48
  %639 = extractelement <4 x float> %632, i64 2, !dbg !47
  %640 = tail call float @llvm.amdgcn.exp2.f32(float %639), !dbg !47
  %local.reduce103 = fadd float %640, %k0.sum.anchor1877, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1885 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %641 = insertelement <4 x float> %638, float %640, i64 2, !dbg !48
  %642 = extractelement <4 x float> %632, i64 3, !dbg !47
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %642), !dbg !47
  %local.reduce104 = fadd float %643, %k0.sum.anchor1878, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1886 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %644 = insertelement <4 x float> %641, float %643, i64 3, !dbg !48
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %644) #8, !dbg !48, !frisk.register_lifetimes !5
  %645 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !48
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %645) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %646 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %631, <4 x float> %612) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %647 = shufflevector <8 x half> %479, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %648 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %647, <4 x float> %614) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %649 = fsub <4 x float> %403, %451, !dbg !40
  %650 = extractelement <4 x float> %649, i64 0, !dbg !47
  %651 = tail call float @llvm.amdgcn.exp2.f32(float %650), !dbg !47
  %local.reduce70 = fadd float %651, %k0.sum.anchor1879, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1887 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %652 = insertelement <4 x float> poison, float %651, i64 0, !dbg !48
  %653 = extractelement <4 x float> %649, i64 1, !dbg !47
  %654 = tail call float @llvm.amdgcn.exp2.f32(float %653), !dbg !47
  %local.reduce71 = fadd float %654, %k0.sum.anchor1880, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1888 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %655 = insertelement <4 x float> %652, float %654, i64 1, !dbg !48
  %656 = extractelement <4 x float> %649, i64 2, !dbg !47
  %657 = tail call float @llvm.amdgcn.exp2.f32(float %656), !dbg !47
  %local.reduce72 = fadd float %657, %k0.sum.anchor1881, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1889 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %658 = insertelement <4 x float> %655, float %657, i64 2, !dbg !48
  %659 = extractelement <4 x float> %649, i64 3, !dbg !47
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %659), !dbg !47
  %local.reduce73 = fadd float %660, %k0.sum.anchor1882, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1890 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %661 = insertelement <4 x float> %658, float %660, i64 3, !dbg !48
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %661) #8, !dbg !48, !frisk.register_lifetimes !5
  %662 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !48
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %662) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %663 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %631, <4 x float> %629) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %664 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %647, <4 x float> %630) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %665 = shufflevector <8 x half> %481, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %666 = fsub <4 x float> %360, %450, !dbg !40
  %667 = extractelement <4 x float> %666, i64 0, !dbg !47
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %667), !dbg !47
  %local.reduce105 = fadd float %668, %k0.sum.anchor1883, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1891 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %669 = insertelement <4 x float> poison, float %668, i64 0, !dbg !48
  %670 = extractelement <4 x float> %666, i64 1, !dbg !47
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !47
  %local.reduce106 = fadd float %671, %k0.sum.anchor1884, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1892 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %672 = insertelement <4 x float> %669, float %671, i64 1, !dbg !48
  %673 = extractelement <4 x float> %666, i64 2, !dbg !47
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %673), !dbg !47
  %local.reduce107 = fadd float %674, %k0.sum.anchor1885, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1893 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %675 = insertelement <4 x float> %672, float %674, i64 2, !dbg !48
  %676 = extractelement <4 x float> %666, i64 3, !dbg !47
  %677 = tail call float @llvm.amdgcn.exp2.f32(float %676), !dbg !47
  %local.reduce108 = fadd float %677, %k0.sum.anchor1886, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1894 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %678 = insertelement <4 x float> %675, float %677, i64 3, !dbg !48
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %678) #8, !dbg !48, !frisk.register_lifetimes !5
  %679 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !48
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %679) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %680 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %665, <4 x float> %646) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %681 = shufflevector <8 x half> %481, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %682 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %681, <4 x float> %648) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %683 = fsub <4 x float> %409, %451, !dbg !40
  %684 = extractelement <4 x float> %683, i64 0, !dbg !47
  %685 = tail call float @llvm.amdgcn.exp2.f32(float %684), !dbg !47
  %local.reduce74 = fadd float %685, %k0.sum.anchor1887, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1895 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %686 = insertelement <4 x float> poison, float %685, i64 0, !dbg !48
  %687 = extractelement <4 x float> %683, i64 1, !dbg !47
  %688 = tail call float @llvm.amdgcn.exp2.f32(float %687), !dbg !47
  %local.reduce75 = fadd float %688, %k0.sum.anchor1888, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1896 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %689 = insertelement <4 x float> %686, float %688, i64 1, !dbg !48
  %690 = extractelement <4 x float> %683, i64 2, !dbg !47
  %691 = tail call float @llvm.amdgcn.exp2.f32(float %690), !dbg !47
  %local.reduce76 = fadd float %691, %k0.sum.anchor1889, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1897 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %692 = insertelement <4 x float> %689, float %691, i64 2, !dbg !48
  %693 = extractelement <4 x float> %683, i64 3, !dbg !47
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %693), !dbg !47
  %local.reduce77 = fadd float %694, %k0.sum.anchor1890, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1898 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %695 = insertelement <4 x float> %692, float %694, i64 3, !dbg !48
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %695) #8, !dbg !48, !frisk.register_lifetimes !5
  %696 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !48
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %696) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %697 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %665, <4 x float> %663) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %681, <4 x float> %664) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %699 = shufflevector <8 x half> %483, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %700 = fsub <4 x float> %366, %450, !dbg !40
  %701 = extractelement <4 x float> %700, i64 0, !dbg !47
  %702 = tail call float @llvm.amdgcn.exp2.f32(float %701), !dbg !47
  %local.reduce109 = fadd float %702, %k0.sum.anchor1891, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1899 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %703 = insertelement <4 x float> poison, float %702, i64 0, !dbg !48
  %704 = extractelement <4 x float> %700, i64 1, !dbg !47
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %704), !dbg !47
  %local.reduce110 = fadd float %705, %k0.sum.anchor1892, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1900 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %706 = insertelement <4 x float> %703, float %705, i64 1, !dbg !48
  %707 = extractelement <4 x float> %700, i64 2, !dbg !47
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %707), !dbg !47
  %local.reduce111 = fadd float %708, %k0.sum.anchor1893, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1901 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %709 = insertelement <4 x float> %706, float %708, i64 2, !dbg !48
  %710 = extractelement <4 x float> %700, i64 3, !dbg !47
  %711 = tail call float @llvm.amdgcn.exp2.f32(float %710), !dbg !47
  %local.reduce112 = fadd float %711, %k0.sum.anchor1894, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1902 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %712 = insertelement <4 x float> %709, float %711, i64 3, !dbg !48
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %712) #8, !dbg !48, !frisk.register_lifetimes !5
  %713 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !48
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %713) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %714 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %699, <4 x float> %680) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %715 = shufflevector <8 x half> %483, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %715, <4 x float> %682) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %717 = fsub <4 x float> %415, %451, !dbg !40
  %718 = extractelement <4 x float> %717, i64 0, !dbg !47
  %719 = tail call float @llvm.amdgcn.exp2.f32(float %718), !dbg !47
  %local.reduce78 = fadd float %719, %k0.sum.anchor1895, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1903 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %720 = insertelement <4 x float> poison, float %719, i64 0, !dbg !48
  %721 = extractelement <4 x float> %717, i64 1, !dbg !47
  %722 = tail call float @llvm.amdgcn.exp2.f32(float %721), !dbg !47
  %local.reduce79 = fadd float %722, %k0.sum.anchor1896, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1904 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %723 = insertelement <4 x float> %720, float %722, i64 1, !dbg !48
  %724 = extractelement <4 x float> %717, i64 2, !dbg !47
  %725 = tail call float @llvm.amdgcn.exp2.f32(float %724), !dbg !47
  %local.reduce80 = fadd float %725, %k0.sum.anchor1897, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1905 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %726 = insertelement <4 x float> %723, float %725, i64 2, !dbg !48
  %727 = extractelement <4 x float> %717, i64 3, !dbg !47
  %728 = tail call float @llvm.amdgcn.exp2.f32(float %727), !dbg !47
  %local.reduce81 = fadd float %728, %k0.sum.anchor1898, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1906 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %729 = insertelement <4 x float> %726, float %728, i64 3, !dbg !48
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #8, !dbg !48, !frisk.register_lifetimes !5
  %730 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !48
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %699, <4 x float> %697) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %715, <4 x float> %698) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %733 = bitcast <4 x i32> %484 to <8 x half>, !dbg !45
  %734 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %735 = fsub <4 x float> %372, %450, !dbg !40
  %736 = extractelement <4 x float> %735, i64 0, !dbg !47
  %737 = tail call float @llvm.amdgcn.exp2.f32(float %736), !dbg !47
  %local.reduce113 = fadd float %737, %k0.sum.anchor1899, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1907 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %738 = insertelement <4 x float> poison, float %737, i64 0, !dbg !48
  %739 = extractelement <4 x float> %735, i64 1, !dbg !47
  %740 = tail call float @llvm.amdgcn.exp2.f32(float %739), !dbg !47
  %local.reduce114 = fadd float %740, %k0.sum.anchor1900, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1908 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce117 = fadd float %k0.sum.anchor1907, %k0.sum.anchor1908, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1909 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %741 = insertelement <4 x float> %738, float %740, i64 1, !dbg !48
  %742 = extractelement <4 x float> %735, i64 2, !dbg !47
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !47
  %local.reduce115 = fadd float %743, %k0.sum.anchor1901, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1910 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %744 = insertelement <4 x float> %741, float %743, i64 2, !dbg !48
  %745 = extractelement <4 x float> %735, i64 3, !dbg !47
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %745), !dbg !47
  %local.reduce116 = fadd float %746, %k0.sum.anchor1902, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1911 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce118 = fadd float %k0.sum.anchor1910, %k0.sum.anchor1911, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1912 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %747 = fadd float %k0.sum.anchor1909, %k0.sum.anchor1912, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1913 = tail call float asm sideeffect "", "=v,0"(float %747) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %748 = fadd float %459, %k0.sum.anchor1913, !dbg !49
  %k0.sum.anchor1914 = tail call float asm sideeffect "", "=v,0"(float %748) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %749 = insertelement <1 x float> poison, float %k0.sum.anchor1914, i64 0, !dbg !12
  %750 = insertvalue [2 x <1 x float>] poison, <1 x float> %749, 0, !dbg !12
  %751 = insertelement <4 x float> %744, float %746, i64 3, !dbg !48
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %751) #8, !dbg !48, !frisk.register_lifetimes !5
  %752 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !48
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %752) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %753 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %734, <4 x float> %714) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %754 = shufflevector <8 x half> %733, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %755 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %754, <4 x float> %716) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %756 = fsub <4 x float> %421, %451, !dbg !40
  %757 = extractelement <4 x float> %756, i64 0, !dbg !47
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %757), !dbg !47
  %local.reduce82 = fadd float %758, %k0.sum.anchor1903, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1915 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %759 = insertelement <4 x float> poison, float %758, i64 0, !dbg !48
  %760 = extractelement <4 x float> %756, i64 1, !dbg !47
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %760), !dbg !47
  %local.reduce83 = fadd float %761, %k0.sum.anchor1904, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1916 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce86 = fadd float %k0.sum.anchor1915, %k0.sum.anchor1916, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1917 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %762 = insertelement <4 x float> %759, float %761, i64 1, !dbg !48
  %763 = extractelement <4 x float> %756, i64 2, !dbg !47
  %764 = tail call float @llvm.amdgcn.exp2.f32(float %763), !dbg !47
  %local.reduce84 = fadd float %764, %k0.sum.anchor1905, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1918 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %765 = insertelement <4 x float> %762, float %764, i64 2, !dbg !48
  %766 = extractelement <4 x float> %756, i64 3, !dbg !47
  %767 = tail call float @llvm.amdgcn.exp2.f32(float %766), !dbg !47
  %local.reduce85 = fadd float %767, %k0.sum.anchor1906, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1919 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce87 = fadd float %k0.sum.anchor1918, %k0.sum.anchor1919, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1920 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %768 = fadd float %k0.sum.anchor1917, %k0.sum.anchor1920, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1921 = tail call float asm sideeffect "", "=v,0"(float %768) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %769 = fadd float %467, %k0.sum.anchor1921, !dbg !49
  %k0.sum.anchor1922 = tail call float asm sideeffect "", "=v,0"(float %769) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %770 = insertelement <1 x float> poison, float %k0.sum.anchor1922, i64 0, !dbg !12
  %771 = insertvalue [2 x <1 x float>] %750, <1 x float> %770, 1, !dbg !12
  %772 = insertelement <4 x float> %765, float %767, i64 3, !dbg !48
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %772) #8, !dbg !48, !frisk.register_lifetimes !5
  %773 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !48
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %773) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %734, <4 x float> %731) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %775 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %754, <4 x float> %732) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1428 = or disjoint i64 %address.local905, %204, !dbg !38
  %vdb.address1429 = trunc nuw nsw i64 %vdb.address1428 to i32, !dbg !38
  %vdb.address1430 = shl nuw nsw i32 %vdb.address1429, 1, !dbg !38
  %vdb.address1435 = or disjoint i64 %address.local911, %204, !dbg !38
  %vdb.address1436 = trunc nuw nsw i64 %vdb.address1435 to i32, !dbg !38
  %vdb.address1437 = shl nuw nsw i32 %vdb.address1436, 1, !dbg !38
  %776 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %vdb.address1430, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %777 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1437, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local919, i64 8192, !dbg !45
  %778 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %779 = bitcast <4 x i32> %778 to <8 x half>, !dbg !45
  %vdb.read1421 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8448, !dbg !45
  %780 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %781 = bitcast <4 x i32> %780 to <8 x half>, !dbg !45
  %vdb.read1422 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8704, !dbg !45
  %782 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %783 = bitcast <4 x i32> %782 to <8 x half>, !dbg !45
  %vdb.read1423 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8960, !dbg !45
  %784 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %785 = bitcast <4 x i32> %784 to <8 x half>, !dbg !45
  %vdb.read1424 = getelementptr i8, ptr addrspace(3) %address.local919, i64 12288, !dbg !45
  %786 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %787 = bitcast <4 x i32> %786 to <8 x half>, !dbg !45
  %vdb.read1425 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12544, !dbg !45
  %788 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %789 = bitcast <4 x i32> %788 to <8 x half>, !dbg !45
  %vdb.read1426 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12800, !dbg !45
  %790 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %791 = bitcast <4 x i32> %790 to <8 x half>, !dbg !45
  %vdb.read1427 = getelementptr i8, ptr addrspace(3) %address.local928, i64 13056, !dbg !45
  %792 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %793 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %794 = extractvalue [2 x <8 x float>] %119, 0, !dbg !43
  %795 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %796 = fmul <4 x float> %795, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %793, <4 x float> %796) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %798 = shufflevector <8 x half> %779, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %799 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %800 = fmul <4 x float> %799, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %798, <4 x float> %800) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %802 = extractvalue [2 x <8 x float>] %119, 1, !dbg !43
  %803 = shufflevector <8 x float> %802, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %804 = fmul <4 x float> %803, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %793, <4 x float> %804) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %806 = shufflevector <8 x float> %802, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %807 = fmul <4 x float> %806, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %798, <4 x float> %807) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %809 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %809, <4 x float> %797) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %811 = shufflevector <8 x half> %781, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %811, <4 x float> %801) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %809, <4 x float> %805) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %811, <4 x float> %808) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %815 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %815, <4 x float> %810) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %817 = shufflevector <8 x half> %783, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %817, <4 x float> %812) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %815, <4 x float> %813) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %817, <4 x float> %814) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %821 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %821, <4 x float> %816) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %823 = shufflevector <8 x half> %785, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %823, <4 x float> %818) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %821, <4 x float> %819) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %823, <4 x float> %820) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %827 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %827, <4 x float> %822) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %829 = shufflevector <8 x half> %787, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %829, <4 x float> %824) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %827, <4 x float> %825) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %829, <4 x float> %826) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %833 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %833, <4 x float> %828) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %835 = shufflevector <8 x half> %789, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %835, <4 x float> %830) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %833, <4 x float> %831) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %835, <4 x float> %832) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %839 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %839, <4 x float> %834) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %841 = shufflevector <8 x half> %791, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %841, <4 x float> %836) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %839, <4 x float> %837) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %841, <4 x float> %838) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %845 = bitcast <4 x i32> %792 to <8 x half>, !dbg !45
  %846 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %846, <4 x float> %840) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %848 = shufflevector <8 x half> %845, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %848, <4 x float> %842) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %846, <4 x float> %843) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %848, <4 x float> %844) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1442 = or disjoint i64 %address.local905, %257, !dbg !38
  %vdb.address1443 = trunc nuw nsw i64 %vdb.address1442 to i32, !dbg !38
  %vdb.address1444 = shl nuw nsw i32 %vdb.address1443, 1, !dbg !38
  %vdb.address1449 = or disjoint i64 %address.local911, %257, !dbg !38
  %vdb.address1450 = trunc nuw nsw i64 %vdb.address1449 to i32, !dbg !38
  %vdb.address1451 = shl nuw nsw i32 %vdb.address1450, 1, !dbg !38
  %852 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1444, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %853 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1451, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %854 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %855 = bitcast <4 x i32> %854 to <8 x half>, !dbg !45
  %856 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !45
  %858 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !45
  %860 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !45
  %862 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !45
  %864 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !45
  %866 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !45
  %868 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %869 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %870 = extractvalue [2 x <8 x float>] %118, 0, !dbg !43
  %871 = shufflevector <8 x float> %870, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %872 = fmul <4 x float> %871, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %869, <4 x float> %872) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %874 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %875 = shufflevector <8 x float> %870, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %876 = fmul <4 x float> %875, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %874, <4 x float> %876) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %878 = extractvalue [2 x <8 x float>] %118, 1, !dbg !43
  %879 = shufflevector <8 x float> %878, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %880 = fmul <4 x float> %879, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %869, <4 x float> %880) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %882 = shufflevector <8 x float> %878, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %883 = fmul <4 x float> %882, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %874, <4 x float> %883) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %885 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %885, <4 x float> %873) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %887 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %887, <4 x float> %877) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %885, <4 x float> %881) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %887, <4 x float> %884) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %891 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %891, <4 x float> %886) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %893 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %893, <4 x float> %888) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %891, <4 x float> %889) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %893, <4 x float> %890) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %897 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %897, <4 x float> %892) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %899 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %899, <4 x float> %894) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %897, <4 x float> %895) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %899, <4 x float> %896) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %903 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %903, <4 x float> %898) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %905 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %905, <4 x float> %900) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %903, <4 x float> %901) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %905, <4 x float> %902) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %909 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %909, <4 x float> %904) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %911 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %911, <4 x float> %906) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %909, <4 x float> %907) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %911, <4 x float> %908) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %915 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %915, <4 x float> %910) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %917 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %917, <4 x float> %912) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %915, <4 x float> %913) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %917, <4 x float> %914) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %921 = bitcast <4 x i32> %868 to <8 x half>, !dbg !45
  %922 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %922, <4 x float> %916) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %924 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %924, <4 x float> %918) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %922, <4 x float> %919) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %924, <4 x float> %920) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %928 = add nuw nsw i64 %123, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %123, 31, !dbg !45
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !45

k0.issue:                                         ; preds = %116
  %k0.next. = shl nuw nsw i64 %928, 14, !dbg !14
  %k0.next.1933 = add nuw nsw i64 %k0.next., %k0.prime.1923, !dbg !14
  %k0.next.kdma.source1476 = or disjoint i64 %k0.next.1933, %k0.prime.kdma.source1475, !dbg !14
  %k0.next.kdma.source1477 = trunc nuw nsw i64 %k0.next.kdma.source1476 to i32, !dbg !14
  %k0.next.kdma.source1478 = shl nuw nsw i32 %k0.next.kdma.source1477, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1478, %kdma.intra, !dbg !14
  %k0.next.1934 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1490 = or disjoint i64 %k0.next.1933, %k0.prime.kdma.source1489, !dbg !14
  %k0.next.kdma.source1491 = trunc nuw nsw i64 %k0.next.kdma.source1490 to i32, !dbg !14
  %k0.next.kdma.source1492 = shl nuw nsw i32 %k0.next.kdma.source1491, 1, !dbg !14
  %k0.next.kdma.offset1493 = or disjoint i32 %k0.next.kdma.source1492, %kdma.intra.hi, !dbg !14
  %k0.next.1935 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1494, i32 %k0.next.kdma.offset1493, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1507 = or disjoint i64 %k0.next.1933, %k0.prime.kdma.source1506, !dbg !14
  %k0.next.kdma.source1508 = trunc nuw nsw i64 %k0.next.kdma.source1507 to i32, !dbg !14
  %k0.next.kdma.source1509 = shl nuw nsw i32 %k0.next.kdma.source1508, 1, !dbg !14
  %k0.next.kdma.offset1510 = or disjoint i32 %k0.next.kdma.source1509, %kdma.intra, !dbg !14
  %k0.next.1936 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset1510, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1524 = or disjoint i64 %k0.next.1933, %k0.prime.kdma.source1523, !dbg !14
  %k0.next.kdma.source1525 = trunc nuw nsw i64 %k0.next.kdma.source1524 to i32, !dbg !14
  %k0.next.kdma.source1526 = shl nuw nsw i32 %k0.next.kdma.source1525, 1, !dbg !14
  %k0.next.kdma.offset1527 = or disjoint i32 %k0.next.kdma.source1526, %kdma.intra.hi, !dbg !14
  %k0.next.1937 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1528, i32 %k0.next.kdma.offset1527, <4 x i32> %k0.prime.dma.buffer.rsrc1343) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %116
  %929 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %930 = bitcast <4 x i32> %929 to <8 x half>, !dbg !45
  %931 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !45
  %933 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !45
  %935 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !45
  %937 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !45
  %939 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !45
  %941 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !45
  %943 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %944 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %945 = extractvalue [2 x <8 x float>] %117, 0, !dbg !43
  %946 = shufflevector <8 x float> %945, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %947 = fmul <4 x float> %946, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %944, <4 x float> %947) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %949 = shufflevector <8 x half> %930, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %950 = shufflevector <8 x float> %945, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %951 = fmul <4 x float> %950, %456, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %949, <4 x float> %951) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %953 = extractvalue [2 x <8 x float>] %117, 1, !dbg !43
  %954 = shufflevector <8 x float> %953, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %955 = fmul <4 x float> %954, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %944, <4 x float> %955) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %957 = shufflevector <8 x float> %953, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %958 = fmul <4 x float> %957, %464, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %949, <4 x float> %958) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %960 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %960, <4 x float> %948) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %962 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %962, <4 x float> %952) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %960, <4 x float> %956) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %962, <4 x float> %959) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %966 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %966, <4 x float> %961) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %968 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %968, <4 x float> %963) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %966, <4 x float> %964) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %968, <4 x float> %965) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %972 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %972, <4 x float> %967) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %974 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %974, <4 x float> %969) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %972, <4 x float> %970) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %974, <4 x float> %971) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %978 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %978, <4 x float> %973) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %980 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %980, <4 x float> %975) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %978, <4 x float> %976) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %980, <4 x float> %977) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %984 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %984, <4 x float> %979) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %986 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %986, <4 x float> %981) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %984, <4 x float> %982) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %986, <4 x float> %983) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %990 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %990, <4 x float> %985) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %992 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %992, <4 x float> %987) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %990, <4 x float> %988) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %992, <4 x float> %989) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %996 = bitcast <4 x i32> %943 to <8 x half>, !dbg !45
  %997 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %997, <4 x float> %991) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %999 = shufflevector <8 x half> %996, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %999, <4 x float> %993) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %997, <4 x float> %994) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %999, <4 x float> %995) #8, !dbg !45, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1003 = shufflevector <4 x float> %753, <4 x float> %755, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1004 = insertvalue [2 x <8 x float>] poison, <8 x float> %1003, 0, !dbg !12
  %1005 = shufflevector <4 x float> %774, <4 x float> %775, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1006 = insertvalue [2 x <8 x float>] %1004, <8 x float> %1005, 1, !dbg !12
  %1007 = insertvalue [2 x <1 x float>] poison, <1 x float> %446, 0, !dbg !12
  %1008 = insertvalue [2 x <1 x float>] %1007, <1 x float> %449, 1, !dbg !12
  %1009 = shufflevector <4 x float> %847, <4 x float> %849, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1010 = insertvalue [2 x <8 x float>] poison, <8 x float> %1009, 0, !dbg !12
  %1011 = shufflevector <4 x float> %850, <4 x float> %851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1012 = insertvalue [2 x <8 x float>] %1010, <8 x float> %1011, 1, !dbg !12
  %1013 = shufflevector <4 x float> %923, <4 x float> %925, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1014 = insertvalue [2 x <8 x float>] poison, <8 x float> %1013, 0, !dbg !12
  %1015 = shufflevector <4 x float> %926, <4 x float> %927, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1016 = insertvalue [2 x <8 x float>] %1014, <8 x float> %1015, 1, !dbg !12
  %1017 = shufflevector <4 x float> %998, <4 x float> %1000, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1018 = insertvalue [2 x <8 x float>] poison, <8 x float> %1017, 0, !dbg !12
  %1019 = shufflevector <4 x float> %1001, <4 x float> %1002, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1020 = insertvalue [2 x <8 x float>] %1018, <8 x float> %1019, 1, !dbg !12
  %1021 = icmp samesign ult i64 %123, 31, !dbg !12
  br i1 %1021, label %116, label %1022, !dbg !12

1022:                                             ; preds = %k0.join
  %1023 = shl nuw nsw i32 %8, 2, !dbg !12
  %1024 = xor i32 %1023, 64, !dbg !12
  %1025 = bitcast float %k0.sum.anchor1922 to i32, !dbg !12
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1024, i32 %1025), !dbg !12
  %1026 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !12
  %lane.sum.final189 = fadd float %k0.sum.anchor1922, %1026, !dbg !12
  %1027 = xor i32 %1023, 128, !dbg !12
  %1028 = bitcast float %lane.sum.final189 to i32, !dbg !12
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1027, i32 %1028), !dbg !12
  %1029 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !12
  %lane.sum.final191 = fadd float %lane.sum.final189, %1029, !dbg !12
  %1030 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !12
  %1031 = bitcast float %k0.sum.anchor1914 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1024, i32 %1031), !dbg !12
  %1032 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor1914, %1032, !dbg !12
  %1033 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1027, i32 %1033), !dbg !12
  %1034 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !12
  %lane.sum.final183 = fadd float %lane.sum.final, %1034, !dbg !12
  %1035 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !12
  %1036 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1036, 0, !dbg !12
  %1037 = and i32 %kdma.tid, 32, !dbg !12
  %.not36 = icmp eq i32 %1037, 0, !dbg !12
  %1038 = extractelement <4 x float> %998, i64 0, !dbg !12
  %1039 = extractelement <4 x float> %998, i64 1, !dbg !12
  %1040 = extractelement <4 x float> %998, i64 2, !dbg !12
  %1041 = extractelement <4 x float> %998, i64 3, !dbg !12
  %1042 = select i1 %.not, float %1039, float %1038, !dbg !12
  %1043 = bitcast float %1042 to i32, !dbg !12
  %1044 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1043), !dbg !12
  %1045 = bitcast i32 %1044 to float, !dbg !12
  %1046 = select i1 %.not, float %1038, float %1045, !dbg !12
  %1047 = select i1 %.not, float %1045, float %1039, !dbg !12
  %1048 = select i1 %.not, float %1041, float %1040, !dbg !12
  %1049 = bitcast float %1048 to i32, !dbg !12
  %1050 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1049), !dbg !12
  %1051 = bitcast i32 %1050 to float, !dbg !12
  %1052 = select i1 %.not, float %1040, float %1051, !dbg !12
  %1053 = select i1 %.not, float %1051, float %1041, !dbg !12
  %1054 = select i1 %.not36, float %1052, float %1046, !dbg !12
  %1055 = bitcast float %1054 to i32, !dbg !12
  %1056 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1055), !dbg !12
  %1057 = bitcast i32 %1056 to float, !dbg !12
  %1058 = select i1 %.not36, float %1053, float %1047, !dbg !12
  %1059 = bitcast float %1058 to i32, !dbg !12
  %1060 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1059), !dbg !12
  %1061 = bitcast i32 %1060 to float, !dbg !12
  %1062 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !12
  %1063 = shufflevector <4 x i1> %1062, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1064 = insertelement <4 x float> poison, float %1046, i64 0, !dbg !12
  %1065 = insertelement <4 x float> %1064, float %1047, i64 1, !dbg !12
  %1066 = insertelement <4 x float> %1065, float %1057, i64 2, !dbg !12
  %1067 = insertelement <4 x float> %1066, float %1061, i64 3, !dbg !12
  %1068 = insertelement <4 x float> poison, float %1057, i64 0, !dbg !12
  %1069 = insertelement <4 x float> %1068, float %1061, i64 1, !dbg !12
  %1070 = insertelement <4 x float> %1069, float %1052, i64 2, !dbg !12
  %1071 = insertelement <4 x float> %1070, float %1053, i64 3, !dbg !12
  %1072 = select <4 x i1> %1063, <4 x float> %1067, <4 x float> %1071, !dbg !12
  %1073 = extractelement <4 x float> %1000, i64 0, !dbg !12
  %1074 = extractelement <4 x float> %1000, i64 1, !dbg !12
  %1075 = extractelement <4 x float> %1000, i64 2, !dbg !12
  %1076 = extractelement <4 x float> %1000, i64 3, !dbg !12
  %1077 = select i1 %.not, float %1074, float %1073, !dbg !12
  %1078 = bitcast float %1077 to i32, !dbg !12
  %1079 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1078), !dbg !12
  %1080 = bitcast i32 %1079 to float, !dbg !12
  %1081 = select i1 %.not, float %1073, float %1080, !dbg !12
  %1082 = select i1 %.not, float %1080, float %1074, !dbg !12
  %1083 = select i1 %.not, float %1076, float %1075, !dbg !12
  %1084 = bitcast float %1083 to i32, !dbg !12
  %1085 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1084), !dbg !12
  %1086 = bitcast i32 %1085 to float, !dbg !12
  %1087 = select i1 %.not, float %1075, float %1086, !dbg !12
  %1088 = select i1 %.not, float %1086, float %1076, !dbg !12
  %1089 = select i1 %.not36, float %1087, float %1081, !dbg !12
  %1090 = bitcast float %1089 to i32, !dbg !12
  %1091 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1090), !dbg !12
  %1092 = bitcast i32 %1091 to float, !dbg !12
  %1093 = select i1 %.not36, float %1088, float %1082, !dbg !12
  %1094 = bitcast float %1093 to i32, !dbg !12
  %1095 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1094), !dbg !12
  %1096 = bitcast i32 %1095 to float, !dbg !12
  %1097 = insertelement <4 x float> poison, float %1081, i64 0, !dbg !12
  %1098 = insertelement <4 x float> %1097, float %1082, i64 1, !dbg !12
  %1099 = insertelement <4 x float> %1098, float %1092, i64 2, !dbg !12
  %1100 = insertelement <4 x float> %1099, float %1096, i64 3, !dbg !12
  %1101 = insertelement <4 x float> poison, float %1092, i64 0, !dbg !12
  %1102 = insertelement <4 x float> %1101, float %1096, i64 1, !dbg !12
  %1103 = insertelement <4 x float> %1102, float %1087, i64 2, !dbg !12
  %1104 = insertelement <4 x float> %1103, float %1088, i64 3, !dbg !12
  %1105 = select <4 x i1> %1063, <4 x float> %1100, <4 x float> %1104, !dbg !12
  %1106 = extractelement <4 x float> %1001, i64 0, !dbg !12
  %1107 = extractelement <4 x float> %1001, i64 1, !dbg !12
  %1108 = extractelement <4 x float> %1001, i64 2, !dbg !12
  %1109 = extractelement <4 x float> %1001, i64 3, !dbg !12
  %1110 = select i1 %.not, float %1107, float %1106, !dbg !12
  %1111 = bitcast float %1110 to i32, !dbg !12
  %1112 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1111), !dbg !12
  %1113 = bitcast i32 %1112 to float, !dbg !12
  %1114 = select i1 %.not, float %1106, float %1113, !dbg !12
  %1115 = select i1 %.not, float %1113, float %1107, !dbg !12
  %1116 = select i1 %.not, float %1109, float %1108, !dbg !12
  %1117 = bitcast float %1116 to i32, !dbg !12
  %1118 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1117), !dbg !12
  %1119 = bitcast i32 %1118 to float, !dbg !12
  %1120 = select i1 %.not, float %1108, float %1119, !dbg !12
  %1121 = select i1 %.not, float %1119, float %1109, !dbg !12
  %1122 = select i1 %.not36, float %1120, float %1114, !dbg !12
  %1123 = bitcast float %1122 to i32, !dbg !12
  %1124 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1123), !dbg !12
  %1125 = bitcast i32 %1124 to float, !dbg !12
  %1126 = select i1 %.not36, float %1121, float %1115, !dbg !12
  %1127 = bitcast float %1126 to i32, !dbg !12
  %1128 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1127), !dbg !12
  %1129 = bitcast i32 %1128 to float, !dbg !12
  %1130 = insertelement <4 x float> poison, float %1114, i64 0, !dbg !12
  %1131 = insertelement <4 x float> %1130, float %1115, i64 1, !dbg !12
  %1132 = insertelement <4 x float> %1131, float %1125, i64 2, !dbg !12
  %1133 = insertelement <4 x float> %1132, float %1129, i64 3, !dbg !12
  %1134 = insertelement <4 x float> poison, float %1125, i64 0, !dbg !12
  %1135 = insertelement <4 x float> %1134, float %1129, i64 1, !dbg !12
  %1136 = insertelement <4 x float> %1135, float %1120, i64 2, !dbg !12
  %1137 = insertelement <4 x float> %1136, float %1121, i64 3, !dbg !12
  %1138 = select <4 x i1> %1063, <4 x float> %1133, <4 x float> %1137, !dbg !12
  %1139 = extractelement <4 x float> %1002, i64 0, !dbg !12
  %1140 = extractelement <4 x float> %1002, i64 1, !dbg !12
  %1141 = extractelement <4 x float> %1002, i64 2, !dbg !12
  %1142 = extractelement <4 x float> %1002, i64 3, !dbg !12
  %1143 = select i1 %.not, float %1140, float %1139, !dbg !12
  %1144 = bitcast float %1143 to i32, !dbg !12
  %1145 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1144), !dbg !12
  %1146 = bitcast i32 %1145 to float, !dbg !12
  %1147 = select i1 %.not, float %1139, float %1146, !dbg !12
  %1148 = select i1 %.not, float %1146, float %1140, !dbg !12
  %1149 = select i1 %.not, float %1142, float %1141, !dbg !12
  %1150 = bitcast float %1149 to i32, !dbg !12
  %1151 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1150), !dbg !12
  %1152 = bitcast i32 %1151 to float, !dbg !12
  %1153 = select i1 %.not, float %1141, float %1152, !dbg !12
  %1154 = select i1 %.not, float %1152, float %1142, !dbg !12
  %1155 = select i1 %.not36, float %1153, float %1147, !dbg !12
  %1156 = bitcast float %1155 to i32, !dbg !12
  %1157 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1156), !dbg !12
  %1158 = bitcast i32 %1157 to float, !dbg !12
  %1159 = select i1 %.not36, float %1154, float %1148, !dbg !12
  %1160 = bitcast float %1159 to i32, !dbg !12
  %1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1160), !dbg !12
  %1162 = bitcast i32 %1161 to float, !dbg !12
  %1163 = insertelement <4 x float> poison, float %1147, i64 0, !dbg !12
  %1164 = insertelement <4 x float> %1163, float %1148, i64 1, !dbg !12
  %1165 = insertelement <4 x float> %1164, float %1158, i64 2, !dbg !12
  %1166 = insertelement <4 x float> %1165, float %1162, i64 3, !dbg !12
  %1167 = insertelement <4 x float> poison, float %1158, i64 0, !dbg !12
  %1168 = insertelement <4 x float> %1167, float %1162, i64 1, !dbg !12
  %1169 = insertelement <4 x float> %1168, float %1153, i64 2, !dbg !12
  %1170 = insertelement <4 x float> %1169, float %1154, i64 3, !dbg !12
  %1171 = select <4 x i1> %1063, <4 x float> %1166, <4 x float> %1170, !dbg !12
  %1172 = extractelement <4 x float> %923, i64 0, !dbg !12
  %1173 = extractelement <4 x float> %923, i64 1, !dbg !12
  %1174 = extractelement <4 x float> %923, i64 2, !dbg !12
  %1175 = extractelement <4 x float> %923, i64 3, !dbg !12
  %1176 = select i1 %.not, float %1173, float %1172, !dbg !12
  %1177 = bitcast float %1176 to i32, !dbg !12
  %1178 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1177), !dbg !12
  %1179 = bitcast i32 %1178 to float, !dbg !12
  %1180 = select i1 %.not, float %1172, float %1179, !dbg !12
  %1181 = select i1 %.not, float %1179, float %1173, !dbg !12
  %1182 = select i1 %.not, float %1175, float %1174, !dbg !12
  %1183 = bitcast float %1182 to i32, !dbg !12
  %1184 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1183), !dbg !12
  %1185 = bitcast i32 %1184 to float, !dbg !12
  %1186 = select i1 %.not, float %1174, float %1185, !dbg !12
  %1187 = select i1 %.not, float %1185, float %1175, !dbg !12
  %1188 = select i1 %.not36, float %1186, float %1180, !dbg !12
  %1189 = bitcast float %1188 to i32, !dbg !12
  %1190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1189), !dbg !12
  %1191 = bitcast i32 %1190 to float, !dbg !12
  %1192 = select i1 %.not36, float %1187, float %1181, !dbg !12
  %1193 = bitcast float %1192 to i32, !dbg !12
  %1194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1193), !dbg !12
  %1195 = bitcast i32 %1194 to float, !dbg !12
  %1196 = insertelement <4 x float> poison, float %1180, i64 0, !dbg !12
  %1197 = insertelement <4 x float> %1196, float %1181, i64 1, !dbg !12
  %1198 = insertelement <4 x float> %1197, float %1191, i64 2, !dbg !12
  %1199 = insertelement <4 x float> %1198, float %1195, i64 3, !dbg !12
  %1200 = insertelement <4 x float> poison, float %1191, i64 0, !dbg !12
  %1201 = insertelement <4 x float> %1200, float %1195, i64 1, !dbg !12
  %1202 = insertelement <4 x float> %1201, float %1186, i64 2, !dbg !12
  %1203 = insertelement <4 x float> %1202, float %1187, i64 3, !dbg !12
  %1204 = select <4 x i1> %1063, <4 x float> %1199, <4 x float> %1203, !dbg !12
  %1205 = extractelement <4 x float> %925, i64 0, !dbg !12
  %1206 = extractelement <4 x float> %925, i64 1, !dbg !12
  %1207 = extractelement <4 x float> %925, i64 2, !dbg !12
  %1208 = extractelement <4 x float> %925, i64 3, !dbg !12
  %1209 = select i1 %.not, float %1206, float %1205, !dbg !12
  %1210 = bitcast float %1209 to i32, !dbg !12
  %1211 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1210), !dbg !12
  %1212 = bitcast i32 %1211 to float, !dbg !12
  %1213 = select i1 %.not, float %1205, float %1212, !dbg !12
  %1214 = select i1 %.not, float %1212, float %1206, !dbg !12
  %1215 = select i1 %.not, float %1208, float %1207, !dbg !12
  %1216 = bitcast float %1215 to i32, !dbg !12
  %1217 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1216), !dbg !12
  %1218 = bitcast i32 %1217 to float, !dbg !12
  %1219 = select i1 %.not, float %1207, float %1218, !dbg !12
  %1220 = select i1 %.not, float %1218, float %1208, !dbg !12
  %1221 = select i1 %.not36, float %1219, float %1213, !dbg !12
  %1222 = bitcast float %1221 to i32, !dbg !12
  %1223 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1222), !dbg !12
  %1224 = bitcast i32 %1223 to float, !dbg !12
  %1225 = select i1 %.not36, float %1220, float %1214, !dbg !12
  %1226 = bitcast float %1225 to i32, !dbg !12
  %1227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1226), !dbg !12
  %1228 = bitcast i32 %1227 to float, !dbg !12
  %1229 = insertelement <4 x float> poison, float %1213, i64 0, !dbg !12
  %1230 = insertelement <4 x float> %1229, float %1214, i64 1, !dbg !12
  %1231 = insertelement <4 x float> %1230, float %1224, i64 2, !dbg !12
  %1232 = insertelement <4 x float> %1231, float %1228, i64 3, !dbg !12
  %1233 = insertelement <4 x float> poison, float %1224, i64 0, !dbg !12
  %1234 = insertelement <4 x float> %1233, float %1228, i64 1, !dbg !12
  %1235 = insertelement <4 x float> %1234, float %1219, i64 2, !dbg !12
  %1236 = insertelement <4 x float> %1235, float %1220, i64 3, !dbg !12
  %1237 = select <4 x i1> %1063, <4 x float> %1232, <4 x float> %1236, !dbg !12
  %1238 = extractelement <4 x float> %926, i64 0, !dbg !12
  %1239 = extractelement <4 x float> %926, i64 1, !dbg !12
  %1240 = extractelement <4 x float> %926, i64 2, !dbg !12
  %1241 = extractelement <4 x float> %926, i64 3, !dbg !12
  %1242 = select i1 %.not, float %1239, float %1238, !dbg !12
  %1243 = bitcast float %1242 to i32, !dbg !12
  %1244 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1243), !dbg !12
  %1245 = bitcast i32 %1244 to float, !dbg !12
  %1246 = select i1 %.not, float %1238, float %1245, !dbg !12
  %1247 = select i1 %.not, float %1245, float %1239, !dbg !12
  %1248 = select i1 %.not, float %1241, float %1240, !dbg !12
  %1249 = bitcast float %1248 to i32, !dbg !12
  %1250 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1249), !dbg !12
  %1251 = bitcast i32 %1250 to float, !dbg !12
  %1252 = select i1 %.not, float %1240, float %1251, !dbg !12
  %1253 = select i1 %.not, float %1251, float %1241, !dbg !12
  %1254 = select i1 %.not36, float %1252, float %1246, !dbg !12
  %1255 = bitcast float %1254 to i32, !dbg !12
  %1256 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1255), !dbg !12
  %1257 = bitcast i32 %1256 to float, !dbg !12
  %1258 = select i1 %.not36, float %1253, float %1247, !dbg !12
  %1259 = bitcast float %1258 to i32, !dbg !12
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1259), !dbg !12
  %1261 = bitcast i32 %1260 to float, !dbg !12
  %1262 = insertelement <4 x float> poison, float %1246, i64 0, !dbg !12
  %1263 = insertelement <4 x float> %1262, float %1247, i64 1, !dbg !12
  %1264 = insertelement <4 x float> %1263, float %1257, i64 2, !dbg !12
  %1265 = insertelement <4 x float> %1264, float %1261, i64 3, !dbg !12
  %1266 = insertelement <4 x float> poison, float %1257, i64 0, !dbg !12
  %1267 = insertelement <4 x float> %1266, float %1261, i64 1, !dbg !12
  %1268 = insertelement <4 x float> %1267, float %1252, i64 2, !dbg !12
  %1269 = insertelement <4 x float> %1268, float %1253, i64 3, !dbg !12
  %1270 = select <4 x i1> %1063, <4 x float> %1265, <4 x float> %1269, !dbg !12
  %1271 = extractelement <4 x float> %927, i64 0, !dbg !12
  %1272 = extractelement <4 x float> %927, i64 1, !dbg !12
  %1273 = extractelement <4 x float> %927, i64 2, !dbg !12
  %1274 = extractelement <4 x float> %927, i64 3, !dbg !12
  %1275 = select i1 %.not, float %1272, float %1271, !dbg !12
  %1276 = bitcast float %1275 to i32, !dbg !12
  %1277 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1276), !dbg !12
  %1278 = bitcast i32 %1277 to float, !dbg !12
  %1279 = select i1 %.not, float %1271, float %1278, !dbg !12
  %1280 = select i1 %.not, float %1278, float %1272, !dbg !12
  %1281 = select i1 %.not, float %1274, float %1273, !dbg !12
  %1282 = bitcast float %1281 to i32, !dbg !12
  %1283 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1282), !dbg !12
  %1284 = bitcast i32 %1283 to float, !dbg !12
  %1285 = select i1 %.not, float %1273, float %1284, !dbg !12
  %1286 = select i1 %.not, float %1284, float %1274, !dbg !12
  %1287 = select i1 %.not36, float %1285, float %1279, !dbg !12
  %1288 = bitcast float %1287 to i32, !dbg !12
  %1289 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1288), !dbg !12
  %1290 = bitcast i32 %1289 to float, !dbg !12
  %1291 = select i1 %.not36, float %1286, float %1280, !dbg !12
  %1292 = bitcast float %1291 to i32, !dbg !12
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1292), !dbg !12
  %1294 = bitcast i32 %1293 to float, !dbg !12
  %1295 = insertelement <4 x float> poison, float %1279, i64 0, !dbg !12
  %1296 = insertelement <4 x float> %1295, float %1280, i64 1, !dbg !12
  %1297 = insertelement <4 x float> %1296, float %1290, i64 2, !dbg !12
  %1298 = insertelement <4 x float> %1297, float %1294, i64 3, !dbg !12
  %1299 = insertelement <4 x float> poison, float %1290, i64 0, !dbg !12
  %1300 = insertelement <4 x float> %1299, float %1294, i64 1, !dbg !12
  %1301 = insertelement <4 x float> %1300, float %1285, i64 2, !dbg !12
  %1302 = insertelement <4 x float> %1301, float %1286, i64 3, !dbg !12
  %1303 = select <4 x i1> %1063, <4 x float> %1298, <4 x float> %1302, !dbg !12
  %1304 = extractelement <4 x float> %847, i64 0, !dbg !12
  %1305 = extractelement <4 x float> %847, i64 1, !dbg !12
  %1306 = extractelement <4 x float> %847, i64 2, !dbg !12
  %1307 = extractelement <4 x float> %847, i64 3, !dbg !12
  %1308 = select i1 %.not, float %1305, float %1304, !dbg !12
  %1309 = bitcast float %1308 to i32, !dbg !12
  %1310 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1309), !dbg !12
  %1311 = bitcast i32 %1310 to float, !dbg !12
  %1312 = select i1 %.not, float %1304, float %1311, !dbg !12
  %1313 = select i1 %.not, float %1311, float %1305, !dbg !12
  %1314 = select i1 %.not, float %1307, float %1306, !dbg !12
  %1315 = bitcast float %1314 to i32, !dbg !12
  %1316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1315), !dbg !12
  %1317 = bitcast i32 %1316 to float, !dbg !12
  %1318 = select i1 %.not, float %1306, float %1317, !dbg !12
  %1319 = select i1 %.not, float %1317, float %1307, !dbg !12
  %1320 = select i1 %.not36, float %1318, float %1312, !dbg !12
  %1321 = bitcast float %1320 to i32, !dbg !12
  %1322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1321), !dbg !12
  %1323 = bitcast i32 %1322 to float, !dbg !12
  %1324 = select i1 %.not36, float %1319, float %1313, !dbg !12
  %1325 = bitcast float %1324 to i32, !dbg !12
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1325), !dbg !12
  %1327 = bitcast i32 %1326 to float, !dbg !12
  %1328 = insertelement <4 x float> poison, float %1312, i64 0, !dbg !12
  %1329 = insertelement <4 x float> %1328, float %1313, i64 1, !dbg !12
  %1330 = insertelement <4 x float> %1329, float %1323, i64 2, !dbg !12
  %1331 = insertelement <4 x float> %1330, float %1327, i64 3, !dbg !12
  %1332 = insertelement <4 x float> poison, float %1323, i64 0, !dbg !12
  %1333 = insertelement <4 x float> %1332, float %1327, i64 1, !dbg !12
  %1334 = insertelement <4 x float> %1333, float %1318, i64 2, !dbg !12
  %1335 = insertelement <4 x float> %1334, float %1319, i64 3, !dbg !12
  %1336 = select <4 x i1> %1063, <4 x float> %1331, <4 x float> %1335, !dbg !12
  %1337 = extractelement <4 x float> %849, i64 0, !dbg !12
  %1338 = extractelement <4 x float> %849, i64 1, !dbg !12
  %1339 = extractelement <4 x float> %849, i64 2, !dbg !12
  %1340 = extractelement <4 x float> %849, i64 3, !dbg !12
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !12
  %1342 = bitcast float %1341 to i32, !dbg !12
  %1343 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1342), !dbg !12
  %1344 = bitcast i32 %1343 to float, !dbg !12
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !12
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !12
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !12
  %1348 = bitcast float %1347 to i32, !dbg !12
  %1349 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1348), !dbg !12
  %1350 = bitcast i32 %1349 to float, !dbg !12
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !12
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !12
  %1353 = select i1 %.not36, float %1351, float %1345, !dbg !12
  %1354 = bitcast float %1353 to i32, !dbg !12
  %1355 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1354), !dbg !12
  %1356 = bitcast i32 %1355 to float, !dbg !12
  %1357 = select i1 %.not36, float %1352, float %1346, !dbg !12
  %1358 = bitcast float %1357 to i32, !dbg !12
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1358), !dbg !12
  %1360 = bitcast i32 %1359 to float, !dbg !12
  %1361 = insertelement <4 x float> poison, float %1345, i64 0, !dbg !12
  %1362 = insertelement <4 x float> %1361, float %1346, i64 1, !dbg !12
  %1363 = insertelement <4 x float> %1362, float %1356, i64 2, !dbg !12
  %1364 = insertelement <4 x float> %1363, float %1360, i64 3, !dbg !12
  %1365 = insertelement <4 x float> poison, float %1356, i64 0, !dbg !12
  %1366 = insertelement <4 x float> %1365, float %1360, i64 1, !dbg !12
  %1367 = insertelement <4 x float> %1366, float %1351, i64 2, !dbg !12
  %1368 = insertelement <4 x float> %1367, float %1352, i64 3, !dbg !12
  %1369 = select <4 x i1> %1063, <4 x float> %1364, <4 x float> %1368, !dbg !12
  %1370 = extractelement <4 x float> %850, i64 0, !dbg !12
  %1371 = extractelement <4 x float> %850, i64 1, !dbg !12
  %1372 = extractelement <4 x float> %850, i64 2, !dbg !12
  %1373 = extractelement <4 x float> %850, i64 3, !dbg !12
  %1374 = select i1 %.not, float %1371, float %1370, !dbg !12
  %1375 = bitcast float %1374 to i32, !dbg !12
  %1376 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1375), !dbg !12
  %1377 = bitcast i32 %1376 to float, !dbg !12
  %1378 = select i1 %.not, float %1370, float %1377, !dbg !12
  %1379 = select i1 %.not, float %1377, float %1371, !dbg !12
  %1380 = select i1 %.not, float %1373, float %1372, !dbg !12
  %1381 = bitcast float %1380 to i32, !dbg !12
  %1382 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1381), !dbg !12
  %1383 = bitcast i32 %1382 to float, !dbg !12
  %1384 = select i1 %.not, float %1372, float %1383, !dbg !12
  %1385 = select i1 %.not, float %1383, float %1373, !dbg !12
  %1386 = select i1 %.not36, float %1384, float %1378, !dbg !12
  %1387 = bitcast float %1386 to i32, !dbg !12
  %1388 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1387), !dbg !12
  %1389 = bitcast i32 %1388 to float, !dbg !12
  %1390 = select i1 %.not36, float %1385, float %1379, !dbg !12
  %1391 = bitcast float %1390 to i32, !dbg !12
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1391), !dbg !12
  %1393 = bitcast i32 %1392 to float, !dbg !12
  %1394 = insertelement <4 x float> poison, float %1378, i64 0, !dbg !12
  %1395 = insertelement <4 x float> %1394, float %1379, i64 1, !dbg !12
  %1396 = insertelement <4 x float> %1395, float %1389, i64 2, !dbg !12
  %1397 = insertelement <4 x float> %1396, float %1393, i64 3, !dbg !12
  %1398 = insertelement <4 x float> poison, float %1389, i64 0, !dbg !12
  %1399 = insertelement <4 x float> %1398, float %1393, i64 1, !dbg !12
  %1400 = insertelement <4 x float> %1399, float %1384, i64 2, !dbg !12
  %1401 = insertelement <4 x float> %1400, float %1385, i64 3, !dbg !12
  %1402 = select <4 x i1> %1063, <4 x float> %1397, <4 x float> %1401, !dbg !12
  %1403 = extractelement <4 x float> %851, i64 0, !dbg !12
  %1404 = extractelement <4 x float> %851, i64 1, !dbg !12
  %1405 = extractelement <4 x float> %851, i64 2, !dbg !12
  %1406 = extractelement <4 x float> %851, i64 3, !dbg !12
  %1407 = select i1 %.not, float %1404, float %1403, !dbg !12
  %1408 = bitcast float %1407 to i32, !dbg !12
  %1409 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1408), !dbg !12
  %1410 = bitcast i32 %1409 to float, !dbg !12
  %1411 = select i1 %.not, float %1403, float %1410, !dbg !12
  %1412 = select i1 %.not, float %1410, float %1404, !dbg !12
  %1413 = select i1 %.not, float %1406, float %1405, !dbg !12
  %1414 = bitcast float %1413 to i32, !dbg !12
  %1415 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1414), !dbg !12
  %1416 = bitcast i32 %1415 to float, !dbg !12
  %1417 = select i1 %.not, float %1405, float %1416, !dbg !12
  %1418 = select i1 %.not, float %1416, float %1406, !dbg !12
  %1419 = select i1 %.not36, float %1417, float %1411, !dbg !12
  %1420 = bitcast float %1419 to i32, !dbg !12
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1420), !dbg !12
  %1422 = bitcast i32 %1421 to float, !dbg !12
  %1423 = select i1 %.not36, float %1418, float %1412, !dbg !12
  %1424 = bitcast float %1423 to i32, !dbg !12
  %1425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1424), !dbg !12
  %1426 = bitcast i32 %1425 to float, !dbg !12
  %1427 = insertelement <4 x float> poison, float %1411, i64 0, !dbg !12
  %1428 = insertelement <4 x float> %1427, float %1412, i64 1, !dbg !12
  %1429 = insertelement <4 x float> %1428, float %1422, i64 2, !dbg !12
  %1430 = insertelement <4 x float> %1429, float %1426, i64 3, !dbg !12
  %1431 = insertelement <4 x float> poison, float %1422, i64 0, !dbg !12
  %1432 = insertelement <4 x float> %1431, float %1426, i64 1, !dbg !12
  %1433 = insertelement <4 x float> %1432, float %1417, i64 2, !dbg !12
  %1434 = insertelement <4 x float> %1433, float %1418, i64 3, !dbg !12
  %1435 = select <4 x i1> %1063, <4 x float> %1430, <4 x float> %1434, !dbg !12
  %1436 = extractelement <4 x float> %753, i64 0, !dbg !12
  %1437 = extractelement <4 x float> %753, i64 1, !dbg !12
  %1438 = extractelement <4 x float> %753, i64 2, !dbg !12
  %1439 = extractelement <4 x float> %753, i64 3, !dbg !12
  %1440 = select i1 %.not, float %1437, float %1436, !dbg !12
  %1441 = bitcast float %1440 to i32, !dbg !12
  %1442 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1441), !dbg !12
  %1443 = bitcast i32 %1442 to float, !dbg !12
  %1444 = select i1 %.not, float %1436, float %1443, !dbg !12
  %1445 = select i1 %.not, float %1443, float %1437, !dbg !12
  %1446 = select i1 %.not, float %1439, float %1438, !dbg !12
  %1447 = bitcast float %1446 to i32, !dbg !12
  %1448 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1447), !dbg !12
  %1449 = bitcast i32 %1448 to float, !dbg !12
  %1450 = select i1 %.not, float %1438, float %1449, !dbg !12
  %1451 = select i1 %.not, float %1449, float %1439, !dbg !12
  %1452 = select i1 %.not36, float %1450, float %1444, !dbg !12
  %1453 = bitcast float %1452 to i32, !dbg !12
  %1454 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1453), !dbg !12
  %1455 = select i1 %.not36, float %1451, float %1445, !dbg !12
  %1456 = bitcast float %1455 to i32, !dbg !12
  %1457 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1456), !dbg !12
  %1458 = extractelement <4 x float> %755, i64 0, !dbg !12
  %1459 = extractelement <4 x float> %755, i64 1, !dbg !12
  %1460 = extractelement <4 x float> %755, i64 2, !dbg !12
  %1461 = extractelement <4 x float> %755, i64 3, !dbg !12
  %1462 = select i1 %.not, float %1459, float %1458, !dbg !12
  %1463 = bitcast float %1462 to i32, !dbg !12
  %1464 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1463), !dbg !12
  %1465 = bitcast i32 %1464 to float, !dbg !12
  %1466 = select i1 %.not, float %1458, float %1465, !dbg !12
  %1467 = select i1 %.not, float %1465, float %1459, !dbg !12
  %1468 = select i1 %.not, float %1461, float %1460, !dbg !12
  %1469 = bitcast float %1468 to i32, !dbg !12
  %1470 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1469), !dbg !12
  %1471 = bitcast i32 %1470 to float, !dbg !12
  %1472 = select i1 %.not, float %1460, float %1471, !dbg !12
  %1473 = select i1 %.not, float %1471, float %1461, !dbg !12
  %1474 = select i1 %.not36, float %1472, float %1466, !dbg !12
  %1475 = bitcast float %1474 to i32, !dbg !12
  %1476 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1475), !dbg !12
  %1477 = select i1 %.not36, float %1473, float %1467, !dbg !12
  %1478 = bitcast float %1477 to i32, !dbg !12
  %1479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1478), !dbg !12
  %1480 = extractelement <4 x float> %774, i64 0, !dbg !12
  %1481 = extractelement <4 x float> %774, i64 1, !dbg !12
  %1482 = extractelement <4 x float> %774, i64 2, !dbg !12
  %1483 = extractelement <4 x float> %774, i64 3, !dbg !12
  %1484 = select i1 %.not, float %1481, float %1480, !dbg !12
  %1485 = bitcast float %1484 to i32, !dbg !12
  %1486 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1485), !dbg !12
  %1487 = bitcast i32 %1486 to float, !dbg !12
  %1488 = select i1 %.not, float %1480, float %1487, !dbg !12
  %1489 = select i1 %.not, float %1487, float %1481, !dbg !12
  %1490 = select i1 %.not, float %1483, float %1482, !dbg !12
  %1491 = bitcast float %1490 to i32, !dbg !12
  %1492 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1491), !dbg !12
  %1493 = bitcast i32 %1492 to float, !dbg !12
  %1494 = select i1 %.not, float %1482, float %1493, !dbg !12
  %1495 = select i1 %.not, float %1493, float %1483, !dbg !12
  %1496 = select i1 %.not36, float %1494, float %1488, !dbg !12
  %1497 = bitcast float %1496 to i32, !dbg !12
  %1498 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1497), !dbg !12
  %1499 = select i1 %.not36, float %1495, float %1489, !dbg !12
  %1500 = bitcast float %1499 to i32, !dbg !12
  %1501 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1500), !dbg !12
  %1502 = extractelement <4 x float> %775, i64 0, !dbg !12
  %1503 = extractelement <4 x float> %775, i64 1, !dbg !12
  %1504 = extractelement <4 x float> %775, i64 2, !dbg !12
  %1505 = extractelement <4 x float> %775, i64 3, !dbg !12
  %1506 = select i1 %.not, float %1503, float %1502, !dbg !12
  %1507 = bitcast float %1506 to i32, !dbg !12
  %1508 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1507), !dbg !12
  %1509 = bitcast i32 %1508 to float, !dbg !12
  %1510 = select i1 %.not, float %1502, float %1509, !dbg !12
  %1511 = select i1 %.not, float %1509, float %1503, !dbg !12
  %1512 = select i1 %.not, float %1505, float %1504, !dbg !12
  %1513 = bitcast float %1512 to i32, !dbg !12
  %1514 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1513), !dbg !12
  %1515 = bitcast i32 %1514 to float, !dbg !12
  %1516 = select i1 %.not, float %1504, float %1515, !dbg !12
  %1517 = select i1 %.not, float %1515, float %1505, !dbg !12
  %1518 = select i1 %.not36, float %1516, float %1510, !dbg !12
  %1519 = bitcast float %1518 to i32, !dbg !12
  %1520 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1519), !dbg !12
  %1521 = select i1 %.not36, float %1517, float %1511, !dbg !12
  %1522 = bitcast float %1521 to i32, !dbg !12
  %1523 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1522), !dbg !12
  %1524 = shufflevector <1 x float> %1035, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1525 = shufflevector <1 x float> %1030, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1526 = bitcast i32 %1520 to float, !dbg !12
  %1527 = bitcast i32 %1523 to float, !dbg !12
  %1528 = insertelement <4 x float> poison, float %1510, i64 0, !dbg !12
  %1529 = insertelement <4 x float> %1528, float %1511, i64 1, !dbg !12
  %1530 = insertelement <4 x float> %1529, float %1526, i64 2, !dbg !12
  %1531 = insertelement <4 x float> %1530, float %1527, i64 3, !dbg !12
  %1532 = insertelement <4 x float> poison, float %1526, i64 0, !dbg !12
  %1533 = insertelement <4 x float> %1532, float %1527, i64 1, !dbg !12
  %1534 = insertelement <4 x float> %1533, float %1516, i64 2, !dbg !12
  %1535 = insertelement <4 x float> %1534, float %1517, i64 3, !dbg !12
  %1536 = select <4 x i1> %1063, <4 x float> %1531, <4 x float> %1535, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1525, !dbg !50
  %normalized = fmul <4 x float> %row.reciprocal, %1536, !dbg !50
  %1537 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !51
  %1538 = extractelement <4 x half> %1537, i64 3, !dbg !51
  %1539 = extractelement <4 x half> %1537, i64 2, !dbg !51
  %1540 = extractelement <4 x half> %1537, i64 1, !dbg !51
  %1541 = extractelement <4 x half> %1537, i64 0, !dbg !51
  %1542 = bitcast i32 %1498 to float, !dbg !12
  %1543 = bitcast i32 %1501 to float, !dbg !12
  %1544 = insertelement <4 x float> poison, float %1488, i64 0, !dbg !12
  %1545 = insertelement <4 x float> %1544, float %1489, i64 1, !dbg !12
  %1546 = insertelement <4 x float> %1545, float %1542, i64 2, !dbg !12
  %1547 = insertelement <4 x float> %1546, float %1543, i64 3, !dbg !12
  %1548 = insertelement <4 x float> poison, float %1542, i64 0, !dbg !12
  %1549 = insertelement <4 x float> %1548, float %1543, i64 1, !dbg !12
  %1550 = insertelement <4 x float> %1549, float %1494, i64 2, !dbg !12
  %1551 = insertelement <4 x float> %1550, float %1495, i64 3, !dbg !12
  %1552 = select <4 x i1> %1063, <4 x float> %1547, <4 x float> %1551, !dbg !12
  %normalized37 = fmul <4 x float> %row.reciprocal, %1552, !dbg !50
  %1553 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !51
  %1554 = extractelement <4 x half> %1553, i64 3, !dbg !51
  %1555 = extractelement <4 x half> %1553, i64 2, !dbg !51
  %1556 = extractelement <4 x half> %1553, i64 1, !dbg !51
  %1557 = extractelement <4 x half> %1553, i64 0, !dbg !51
  %1558 = bitcast i32 %1476 to float, !dbg !12
  %1559 = bitcast i32 %1479 to float, !dbg !12
  %1560 = insertelement <4 x float> poison, float %1466, i64 0, !dbg !12
  %1561 = insertelement <4 x float> %1560, float %1467, i64 1, !dbg !12
  %1562 = insertelement <4 x float> %1561, float %1558, i64 2, !dbg !12
  %1563 = insertelement <4 x float> %1562, float %1559, i64 3, !dbg !12
  %1564 = insertelement <4 x float> poison, float %1558, i64 0, !dbg !12
  %1565 = insertelement <4 x float> %1564, float %1559, i64 1, !dbg !12
  %1566 = insertelement <4 x float> %1565, float %1472, i64 2, !dbg !12
  %1567 = insertelement <4 x float> %1566, float %1473, i64 3, !dbg !12
  %1568 = select <4 x i1> %1063, <4 x float> %1563, <4 x float> %1567, !dbg !12
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1524, !dbg !50
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1568, !dbg !50
  %1569 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !51
  %1570 = extractelement <4 x half> %1569, i64 3, !dbg !51
  %1571 = extractelement <4 x half> %1569, i64 2, !dbg !51
  %1572 = extractelement <4 x half> %1569, i64 1, !dbg !51
  %1573 = extractelement <4 x half> %1569, i64 0, !dbg !51
  %1574 = bitcast i32 %1454 to float, !dbg !12
  %1575 = bitcast i32 %1457 to float, !dbg !12
  %1576 = insertelement <4 x float> poison, float %1444, i64 0, !dbg !12
  %1577 = insertelement <4 x float> %1576, float %1445, i64 1, !dbg !12
  %1578 = insertelement <4 x float> %1577, float %1574, i64 2, !dbg !12
  %1579 = insertelement <4 x float> %1578, float %1575, i64 3, !dbg !12
  %1580 = insertelement <4 x float> poison, float %1574, i64 0, !dbg !12
  %1581 = insertelement <4 x float> %1580, float %1575, i64 1, !dbg !12
  %1582 = insertelement <4 x float> %1581, float %1450, i64 2, !dbg !12
  %1583 = insertelement <4 x float> %1582, float %1451, i64 3, !dbg !12
  %1584 = select <4 x i1> %1063, <4 x float> %1579, <4 x float> %1583, !dbg !12
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1584, !dbg !50
  %1585 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !51
  %1586 = extractelement <4 x half> %1585, i64 3, !dbg !51
  %1587 = extractelement <4 x half> %1585, i64 2, !dbg !51
  %1588 = extractelement <4 x half> %1585, i64 1, !dbg !51
  %1589 = extractelement <4 x half> %1585, i64 0, !dbg !51
  %address.thread958 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !52, !range !10, !frisk.register_lifetimes !5
  %address.local959 = zext nneg i32 %address.thread958 to i64, !dbg !9
  %1590 = lshr i64 %address.local959, 2, !dbg !52
  %address.local962 = and i64 %1590, 48, !dbg !52
  %address.local964 = and i64 %address.local959, 240, !dbg !52
  %address.local963 = add i64 %34, %address.local959, !dbg !52
  %address.local965 = sub i64 %address.local963, %address.local964, !dbg !52
  %address.local966 = add i64 %address.local965, %address.local962, !dbg !52
  %1591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %42, !dbg !52
  %address.local969.idx = shl nuw nsw i64 %address.local966, 8, !dbg !52
  %address.local969 = getelementptr inbounds nuw i8, ptr addrspace(1) %1591, i64 %address.local969.idx, !dbg !52
  %address.local971 = and i64 %1590, 12, !dbg !11
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local971, !dbg !52
  store half %1589, ptr addrspace(1) %address.local972, align 2, !dbg !52
  %address.local974 = or disjoint i64 %address.local971, 1, !dbg !52
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local974, !dbg !52
  store half %1588, ptr addrspace(1) %address.local975, align 2, !dbg !52
  %address.local977 = or disjoint i64 %address.local971, 2, !dbg !52
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local977, !dbg !52
  store half %1587, ptr addrspace(1) %address.local978, align 2, !dbg !52
  %address.local980 = or disjoint i64 %address.local971, 3, !dbg !52
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local980, !dbg !52
  store half %1586, ptr addrspace(1) %address.local981, align 2, !dbg !52
  %address.local983 = or disjoint i64 %address.local971, 16, !dbg !52
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local983, !dbg !52
  store half %1573, ptr addrspace(1) %address.local984, align 2, !dbg !52
  %address.local986 = or disjoint i64 %address.local971, 17, !dbg !52
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local986, !dbg !52
  store half %1572, ptr addrspace(1) %address.local987, align 2, !dbg !52
  %address.local989 = or disjoint i64 %address.local971, 18, !dbg !52
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local989, !dbg !52
  store half %1571, ptr addrspace(1) %address.local990, align 2, !dbg !52
  %address.local992 = or disjoint i64 %address.local971, 19, !dbg !52
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local992, !dbg !52
  store half %1570, ptr addrspace(1) %address.local993, align 2, !dbg !52
  %address.local997 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local969, i64 16384, !dbg !52
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local971, !dbg !52
  store half %1557, ptr addrspace(1) %address.local998, align 2, !dbg !52
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local974, !dbg !52
  store half %1556, ptr addrspace(1) %address.local1000, align 2, !dbg !52
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local977, !dbg !52
  store half %1555, ptr addrspace(1) %address.local1002, align 2, !dbg !52
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local980, !dbg !52
  store half %1554, ptr addrspace(1) %address.local1004, align 2, !dbg !52
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local983, !dbg !52
  store half %1541, ptr addrspace(1) %address.local1006, align 2, !dbg !52
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local986, !dbg !52
  store half %1540, ptr addrspace(1) %address.local1008, align 2, !dbg !52
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local989, !dbg !52
  store half %1539, ptr addrspace(1) %address.local1010, align 2, !dbg !52
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local992, !dbg !52
  store half %1538, ptr addrspace(1) %address.local1012, align 2, !dbg !52
  %normalized42 = fmul <4 x float> %row.reciprocal, %1435, !dbg !50
  %1592 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !51
  %1593 = extractelement <4 x half> %1592, i64 3, !dbg !51
  %1594 = extractelement <4 x half> %1592, i64 2, !dbg !51
  %1595 = extractelement <4 x half> %1592, i64 1, !dbg !51
  %1596 = extractelement <4 x half> %1592, i64 0, !dbg !51
  %normalized43 = fmul <4 x float> %row.reciprocal, %1402, !dbg !50
  %1597 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !51
  %1598 = extractelement <4 x half> %1597, i64 3, !dbg !51
  %1599 = extractelement <4 x half> %1597, i64 2, !dbg !51
  %1600 = extractelement <4 x half> %1597, i64 1, !dbg !51
  %1601 = extractelement <4 x half> %1597, i64 0, !dbg !51
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1369, !dbg !50
  %1602 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !51
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !51
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !51
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !51
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !51
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1336, !dbg !50
  %1607 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !51
  %1608 = extractelement <4 x half> %1607, i64 3, !dbg !51
  %1609 = extractelement <4 x half> %1607, i64 2, !dbg !51
  %1610 = extractelement <4 x half> %1607, i64 1, !dbg !51
  %1611 = extractelement <4 x half> %1607, i64 0, !dbg !51
  %address.local1027 = or disjoint i64 %address.local971, 32, !dbg !52
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1027, !dbg !52
  store half %1611, ptr addrspace(1) %address.local1028, align 2, !dbg !52
  %address.local1030 = or disjoint i64 %address.local971, 33, !dbg !52
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1030, !dbg !52
  store half %1610, ptr addrspace(1) %address.local1031, align 2, !dbg !52
  %address.local1033 = or disjoint i64 %address.local971, 34, !dbg !52
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1033, !dbg !52
  store half %1609, ptr addrspace(1) %address.local1034, align 2, !dbg !52
  %address.local1036 = or disjoint i64 %address.local971, 35, !dbg !52
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1036, !dbg !52
  store half %1608, ptr addrspace(1) %address.local1037, align 2, !dbg !52
  %address.local1039 = or disjoint i64 %address.local971, 48, !dbg !52
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1039, !dbg !52
  store half %1606, ptr addrspace(1) %address.local1040, align 2, !dbg !52
  %address.local1042 = or disjoint i64 %address.local971, 49, !dbg !52
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1042, !dbg !52
  store half %1605, ptr addrspace(1) %address.local1043, align 2, !dbg !52
  %address.local1045 = or disjoint i64 %address.local971, 50, !dbg !52
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1045, !dbg !52
  store half %1604, ptr addrspace(1) %address.local1046, align 2, !dbg !52
  %address.local1048 = or i64 %1590, 51, !dbg !52
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1048, !dbg !52
  store half %1603, ptr addrspace(1) %address.local1049, align 2, !dbg !52
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1027, !dbg !52
  store half %1601, ptr addrspace(1) %address.local1054, align 2, !dbg !52
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1030, !dbg !52
  store half %1600, ptr addrspace(1) %address.local1056, align 2, !dbg !52
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1033, !dbg !52
  store half %1599, ptr addrspace(1) %address.local1058, align 2, !dbg !52
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1036, !dbg !52
  store half %1598, ptr addrspace(1) %address.local1060, align 2, !dbg !52
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1039, !dbg !52
  store half %1596, ptr addrspace(1) %address.local1062, align 2, !dbg !52
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1042, !dbg !52
  store half %1595, ptr addrspace(1) %address.local1064, align 2, !dbg !52
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1045, !dbg !52
  store half %1594, ptr addrspace(1) %address.local1066, align 2, !dbg !52
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1048, !dbg !52
  store half %1593, ptr addrspace(1) %address.local1068, align 2, !dbg !52
  %normalized48 = fmul <4 x float> %row.reciprocal, %1303, !dbg !50
  %1612 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !51
  %1613 = extractelement <4 x half> %1612, i64 3, !dbg !51
  %1614 = extractelement <4 x half> %1612, i64 2, !dbg !51
  %1615 = extractelement <4 x half> %1612, i64 1, !dbg !51
  %1616 = extractelement <4 x half> %1612, i64 0, !dbg !51
  %normalized49 = fmul <4 x float> %row.reciprocal, %1270, !dbg !50
  %1617 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !51
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !51
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !51
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !51
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !51
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1237, !dbg !50
  %1622 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !51
  %1623 = extractelement <4 x half> %1622, i64 3, !dbg !51
  %1624 = extractelement <4 x half> %1622, i64 2, !dbg !51
  %1625 = extractelement <4 x half> %1622, i64 1, !dbg !51
  %1626 = extractelement <4 x half> %1622, i64 0, !dbg !51
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1204, !dbg !50
  %1627 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !51
  %1628 = extractelement <4 x half> %1627, i64 3, !dbg !51
  %1629 = extractelement <4 x half> %1627, i64 2, !dbg !51
  %1630 = extractelement <4 x half> %1627, i64 1, !dbg !51
  %1631 = extractelement <4 x half> %1627, i64 0, !dbg !51
  %address.local1083 = or disjoint i64 %address.local971, 64, !dbg !52
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1083, !dbg !52
  store half %1631, ptr addrspace(1) %address.local1084, align 2, !dbg !52
  %address.local1086 = or disjoint i64 %address.local971, 65, !dbg !52
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1086, !dbg !52
  store half %1630, ptr addrspace(1) %address.local1087, align 2, !dbg !52
  %address.local1089 = or disjoint i64 %address.local971, 66, !dbg !52
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1089, !dbg !52
  store half %1629, ptr addrspace(1) %address.local1090, align 2, !dbg !52
  %address.local1092 = or disjoint i64 %address.local971, 67, !dbg !52
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1092, !dbg !52
  store half %1628, ptr addrspace(1) %address.local1093, align 2, !dbg !52
  %address.local1095 = or disjoint i64 %address.local971, 80, !dbg !52
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1095, !dbg !52
  store half %1626, ptr addrspace(1) %address.local1096, align 2, !dbg !52
  %address.local1098 = or disjoint i64 %address.local971, 81, !dbg !52
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1098, !dbg !52
  store half %1625, ptr addrspace(1) %address.local1099, align 2, !dbg !52
  %address.local1101 = or disjoint i64 %address.local971, 82, !dbg !52
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1101, !dbg !52
  store half %1624, ptr addrspace(1) %address.local1102, align 2, !dbg !52
  %address.local1104 = or disjoint i64 %address.local971, 83, !dbg !52
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1104, !dbg !52
  store half %1623, ptr addrspace(1) %address.local1105, align 2, !dbg !52
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1083, !dbg !52
  store half %1621, ptr addrspace(1) %address.local1110, align 2, !dbg !52
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1086, !dbg !52
  store half %1620, ptr addrspace(1) %address.local1112, align 2, !dbg !52
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1089, !dbg !52
  store half %1619, ptr addrspace(1) %address.local1114, align 2, !dbg !52
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1092, !dbg !52
  store half %1618, ptr addrspace(1) %address.local1116, align 2, !dbg !52
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1095, !dbg !52
  store half %1616, ptr addrspace(1) %address.local1118, align 2, !dbg !52
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1098, !dbg !52
  store half %1615, ptr addrspace(1) %address.local1120, align 2, !dbg !52
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1101, !dbg !52
  store half %1614, ptr addrspace(1) %address.local1122, align 2, !dbg !52
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1104, !dbg !52
  store half %1613, ptr addrspace(1) %address.local1124, align 2, !dbg !52
  %normalized54 = fmul <4 x float> %row.reciprocal, %1171, !dbg !50
  %1632 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !51
  %1633 = extractelement <4 x half> %1632, i64 3, !dbg !51
  %1634 = extractelement <4 x half> %1632, i64 2, !dbg !51
  %1635 = extractelement <4 x half> %1632, i64 1, !dbg !51
  %1636 = extractelement <4 x half> %1632, i64 0, !dbg !51
  %normalized55 = fmul <4 x float> %row.reciprocal, %1138, !dbg !50
  %1637 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !51
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !51
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !51
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !51
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !51
  %normalized57 = fmul <4 x float> %row.reciprocal38, %1105, !dbg !50
  %1642 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !51
  %1643 = extractelement <4 x half> %1642, i64 3, !dbg !51
  %1644 = extractelement <4 x half> %1642, i64 2, !dbg !51
  %1645 = extractelement <4 x half> %1642, i64 1, !dbg !51
  %1646 = extractelement <4 x half> %1642, i64 0, !dbg !51
  %normalized58 = fmul <4 x float> %row.reciprocal38, %1072, !dbg !50
  %1647 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !51
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !51
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !51
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !51
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !51
  %address.local1139 = or disjoint i64 %address.local971, 96, !dbg !52
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1139, !dbg !52
  store half %1651, ptr addrspace(1) %address.local1140, align 2, !dbg !52
  %address.local1142 = or disjoint i64 %address.local971, 97, !dbg !52
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1142, !dbg !52
  store half %1650, ptr addrspace(1) %address.local1143, align 2, !dbg !52
  %address.local1145 = or disjoint i64 %address.local971, 98, !dbg !52
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1145, !dbg !52
  store half %1649, ptr addrspace(1) %address.local1146, align 2, !dbg !52
  %address.local1148 = or disjoint i64 %address.local971, 99, !dbg !52
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1148, !dbg !52
  store half %1648, ptr addrspace(1) %address.local1149, align 2, !dbg !52
  %address.local1151 = or disjoint i64 %address.local971, 112, !dbg !52
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1151, !dbg !52
  store half %1646, ptr addrspace(1) %address.local1152, align 2, !dbg !52
  %address.local1154 = or disjoint i64 %address.local971, 113, !dbg !52
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1154, !dbg !52
  store half %1645, ptr addrspace(1) %address.local1155, align 2, !dbg !52
  %address.local1157 = or disjoint i64 %address.local971, 114, !dbg !52
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1157, !dbg !52
  store half %1644, ptr addrspace(1) %address.local1158, align 2, !dbg !52
  %address.local1160 = or i64 %1590, 115, !dbg !52
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1160, !dbg !52
  store half %1643, ptr addrspace(1) %address.local1161, align 2, !dbg !52
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1139, !dbg !52
  store half %1641, ptr addrspace(1) %address.local1166, align 2, !dbg !52
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1142, !dbg !52
  store half %1640, ptr addrspace(1) %address.local1168, align 2, !dbg !52
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1145, !dbg !52
  store half %1639, ptr addrspace(1) %address.local1170, align 2, !dbg !52
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1148, !dbg !52
  store half %1638, ptr addrspace(1) %address.local1172, align 2, !dbg !52
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1151, !dbg !52
  store half %1636, ptr addrspace(1) %address.local1174, align 2, !dbg !52
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1154, !dbg !52
  store half %1635, ptr addrspace(1) %address.local1176, align 2, !dbg !52
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1157, !dbg !52
  store half %1634, ptr addrspace(1) %address.local1178, align 2, !dbg !52
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1160, !dbg !52
  store half %1633, ptr addrspace(1) %address.local1180, align 2, !dbg !52
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
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #5

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
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

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/async_task_noncausal")
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
!13 = !DILocation(line: 70, column: 26, scope: !3)
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
!39 = !DILocation(line: 71, column: 23, scope: !3)
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
