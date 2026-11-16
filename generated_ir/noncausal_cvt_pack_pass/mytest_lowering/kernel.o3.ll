; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/noncausal_cvt_pack_pass/mytest_lowering/kernel.ll'
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
  %117 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %902, %k0.join ]
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %898, %k0.join ]
  %119 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %894, %k0.join ]
  %120 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %lazy.base.join, %k0.join ]
  %121 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %667, %k0.join ]
  %122 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %890, %k0.join ]
  %123 = phi i64 [ 0, %4 ], [ %816, %k0.join ]
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
  %lazy.base = extractvalue [2 x <1 x float>] %120, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base1938 = extractvalue [2 x <1 x float>] %120, 1, !dbg !13
  %lazy.seed1939 = shufflevector <1 x float> %lazy.base1938, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %146, <4 x float> %lazy.seed) #8, !dbg !13
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
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %159, <4 x float> %lazy.seed) #8, !dbg !13
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
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %163, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %165, <4 x float> %176) #8, !dbg !13
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %160, <4 x float> %lazy.seed) #8, !dbg !13
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %162, <4 x float> %178) #8, !dbg !13
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %164, <4 x float> %lazy.seed) #8, !dbg !13
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %166, <4 x float> %180) #8, !dbg !13
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %168, <4 x float> %lazy.seed) #8, !dbg !13
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %171, <4 x float> %182) #8, !dbg !13
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %174, <4 x float> %lazy.seed) #8, !dbg !13
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %175, <4 x float> %184) #8, !dbg !13
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %169, <4 x float> %lazy.seed) #8, !dbg !13
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %172, <4 x float> %186) #8, !dbg !13
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %146, <4 x float> %lazy.seed1939) #8, !dbg !13
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %148, <4 x float> %188) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %159, <4 x float> %lazy.seed1939) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %161, <4 x float> %190) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %163, <4 x float> %lazy.seed1939) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %165, <4 x float> %192) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %160, <4 x float> %lazy.seed1939) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %194) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %164, <4 x float> %lazy.seed1939) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %166, <4 x float> %196) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %168, <4 x float> %lazy.seed1939) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %171, <4 x float> %198) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %174, <4 x float> %lazy.seed1939) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %175, <4 x float> %200) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %169, <4 x float> %lazy.seed1939) #8, !dbg !13
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
  %lazy.peak = tail call float @llvm.maxnum.f32(float %433, float %443), !dbg !40
  %lazy.over = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !40
  %lazy.mask = tail call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !40
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !40
  %lazy.first = icmp eq i64 %123, 0, !dbg !40
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !40
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !40

lazy.slow:                                        ; preds = %116
  %lazy.positive = tail call float @llvm.maxnum.f32(float %433, float 0.000000e+00), !dbg !40
  %lazy.delta = select i1 %lazy.first, float %433, float %lazy.positive, !dbg !40
  %lazy.negated = fsub float 0.000000e+00, %lazy.delta, !dbg !40
  %lazy.exp = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !40
  %lazy.alpha = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp, !dbg !40
  %lazy.positive1940 = tail call float @llvm.maxnum.f32(float %443, float 0.000000e+00), !dbg !40
  %lazy.delta1941 = select i1 %lazy.first, float %443, float %lazy.positive1940, !dbg !40
  %lazy.negated1942 = fsub float 0.000000e+00, %lazy.delta1941, !dbg !40
  %lazy.exp1943 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated1942), !dbg !40
  %lazy.alpha1944 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp1943, !dbg !40
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !40
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %lazy.shifted = fsub <4 x float> %321, %lazy.delta.v, !dbg !40
  %lazy.shifted1947 = fsub <4 x float> %328, %lazy.delta.v, !dbg !40
  %lazy.shifted1950 = fsub <4 x float> %338, %lazy.delta.v, !dbg !40
  %lazy.shifted1953 = fsub <4 x float> %348, %lazy.delta.v, !dbg !40
  %lazy.shifted1956 = fsub <4 x float> %354, %lazy.delta.v, !dbg !40
  %lazy.shifted1959 = fsub <4 x float> %360, %lazy.delta.v, !dbg !40
  %lazy.shifted1962 = fsub <4 x float> %366, %lazy.delta.v, !dbg !40
  %lazy.shifted1965 = fsub <4 x float> %372, %lazy.delta.v, !dbg !40
  %lazy.delta.v.lane1966 = insertelement <1 x float> poison, float %lazy.delta1941, i64 0, !dbg !40
  %lazy.delta.v1967 = shufflevector <1 x float> %lazy.delta.v.lane1966, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %lazy.shifted1968 = fsub <4 x float> %379, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1971 = fsub <4 x float> %385, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1974 = fsub <4 x float> %391, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1977 = fsub <4 x float> %397, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1980 = fsub <4 x float> %403, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1983 = fsub <4 x float> %409, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1986 = fsub <4 x float> %415, %lazy.delta.v1967, !dbg !40
  %lazy.shifted1989 = fsub <4 x float> %421, %lazy.delta.v1967, !dbg !40
  %lazy.state = extractvalue [2 x <1 x float>] %121, 0, !dbg !40
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !40
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !40
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !40
  %lazy.state1990 = extractvalue [2 x <1 x float>] %121, 1, !dbg !40
  %lazy.alpha.v.lane1991 = insertelement <1 x float> poison, float %lazy.alpha1944, i64 0, !dbg !40
  %lazy.scaled1992 = fmul <1 x float> %lazy.state1990, %lazy.alpha.v.lane1991, !dbg !40
  %lazy.state.next1993 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1992, 1, !dbg !40
  %lazy.state1994 = extractvalue [2 x <8 x float>] %117, 0, !dbg !40
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !40
  %lazy.scaled1996 = fmul <8 x float> %lazy.state1994, %lazy.alpha.v, !dbg !40
  %lazy.state.next1997 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1996, 0, !dbg !40
  %lazy.state1998 = extractvalue [2 x <8 x float>] %117, 1, !dbg !40
  %lazy.alpha.v2000 = shufflevector <1 x float> %lazy.alpha.v.lane1991, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !40
  %lazy.scaled2001 = fmul <8 x float> %lazy.state1998, %lazy.alpha.v2000, !dbg !40
  %lazy.state.next2002 = insertvalue [2 x <8 x float>] %lazy.state.next1997, <8 x float> %lazy.scaled2001, 1, !dbg !40
  %lazy.state2003 = extractvalue [2 x <8 x float>] %118, 0, !dbg !40
  %lazy.scaled2006 = fmul <8 x float> %lazy.state2003, %lazy.alpha.v, !dbg !40
  %lazy.state.next2007 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2006, 0, !dbg !40
  %lazy.state2008 = extractvalue [2 x <8 x float>] %118, 1, !dbg !40
  %lazy.scaled2011 = fmul <8 x float> %lazy.state2008, %lazy.alpha.v2000, !dbg !40
  %lazy.state.next2012 = insertvalue [2 x <8 x float>] %lazy.state.next2007, <8 x float> %lazy.scaled2011, 1, !dbg !40
  %lazy.state2013 = extractvalue [2 x <8 x float>] %119, 0, !dbg !40
  %lazy.scaled2016 = fmul <8 x float> %lazy.state2013, %lazy.alpha.v, !dbg !40
  %lazy.state.next2017 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2016, 0, !dbg !40
  %lazy.state2018 = extractvalue [2 x <8 x float>] %119, 1, !dbg !40
  %lazy.scaled2021 = fmul <8 x float> %lazy.state2018, %lazy.alpha.v2000, !dbg !40
  %lazy.state.next2022 = insertvalue [2 x <8 x float>] %lazy.state.next2017, <8 x float> %lazy.scaled2021, 1, !dbg !40
  %lazy.state2023 = extractvalue [2 x <8 x float>] %122, 0, !dbg !40
  %lazy.scaled2026 = fmul <8 x float> %lazy.state2023, %lazy.alpha.v, !dbg !40
  %lazy.state.next2027 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2026, 0, !dbg !40
  %lazy.state2028 = extractvalue [2 x <8 x float>] %122, 1, !dbg !40
  %lazy.scaled2031 = fmul <8 x float> %lazy.state2028, %lazy.alpha.v2000, !dbg !40
  %lazy.state.next2032 = insertvalue [2 x <8 x float>] %lazy.state.next2027, <8 x float> %lazy.scaled2031, 1, !dbg !40
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !40
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !40
  %lazy.base.next2035 = fsub <1 x float> %lazy.base1938, %lazy.delta.v.lane1966, !dbg !40
  %lazy.base.state2036 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2035, 1, !dbg !40
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !41
  %.pre21 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !41
  %.pre23 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !41
  %.pre25 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !41
  %.pre27 = extractelement <4 x float> %lazy.shifted1968, i64 0, !dbg !41
  %.pre29 = extractelement <4 x float> %lazy.shifted1968, i64 1, !dbg !41
  %.pre31 = extractelement <4 x float> %lazy.shifted1968, i64 2, !dbg !41
  %.pre33 = extractelement <4 x float> %lazy.shifted1968, i64 3, !dbg !41
  %.pre35 = extractelement <4 x float> %lazy.shifted1947, i64 0, !dbg !41
  %.pre37 = extractelement <4 x float> %lazy.shifted1947, i64 1, !dbg !41
  %.pre39 = extractelement <4 x float> %lazy.shifted1947, i64 2, !dbg !41
  %.pre41 = extractelement <4 x float> %lazy.shifted1947, i64 3, !dbg !41
  %.pre43 = extractelement <4 x float> %lazy.shifted1971, i64 0, !dbg !41
  %.pre45 = extractelement <4 x float> %lazy.shifted1971, i64 1, !dbg !41
  %.pre47 = extractelement <4 x float> %lazy.shifted1971, i64 2, !dbg !41
  %.pre49 = extractelement <4 x float> %lazy.shifted1971, i64 3, !dbg !41
  %.pre51 = extractelement <4 x float> %lazy.shifted1950, i64 0, !dbg !41
  %.pre53 = extractelement <4 x float> %lazy.shifted1950, i64 1, !dbg !41
  %.pre55 = extractelement <4 x float> %lazy.shifted1950, i64 2, !dbg !41
  %.pre57 = extractelement <4 x float> %lazy.shifted1950, i64 3, !dbg !41
  %.pre59 = extractelement <4 x float> %lazy.shifted1974, i64 0, !dbg !41
  %.pre61 = extractelement <4 x float> %lazy.shifted1974, i64 1, !dbg !41
  %.pre63 = extractelement <4 x float> %lazy.shifted1974, i64 2, !dbg !41
  %.pre65 = extractelement <4 x float> %lazy.shifted1974, i64 3, !dbg !41
  %.pre67 = extractelement <4 x float> %lazy.shifted1953, i64 0, !dbg !41
  %.pre69 = extractelement <4 x float> %lazy.shifted1953, i64 1, !dbg !41
  %.pre71 = extractelement <4 x float> %lazy.shifted1953, i64 2, !dbg !41
  %.pre73 = extractelement <4 x float> %lazy.shifted1953, i64 3, !dbg !41
  %.pre75 = extractelement <4 x float> %lazy.shifted1977, i64 0, !dbg !41
  %.pre77 = extractelement <4 x float> %lazy.shifted1977, i64 1, !dbg !41
  %.pre79 = extractelement <4 x float> %lazy.shifted1977, i64 2, !dbg !41
  %.pre81 = extractelement <4 x float> %lazy.shifted1977, i64 3, !dbg !41
  %.pre83 = extractelement <4 x float> %lazy.shifted1956, i64 0, !dbg !41
  %.pre85 = extractelement <4 x float> %lazy.shifted1956, i64 1, !dbg !41
  %.pre87 = extractelement <4 x float> %lazy.shifted1956, i64 2, !dbg !41
  %.pre89 = extractelement <4 x float> %lazy.shifted1956, i64 3, !dbg !41
  %.pre91 = extractelement <4 x float> %lazy.shifted1980, i64 0, !dbg !41
  %.pre93 = extractelement <4 x float> %lazy.shifted1980, i64 1, !dbg !41
  %.pre95 = extractelement <4 x float> %lazy.shifted1980, i64 2, !dbg !41
  %.pre97 = extractelement <4 x float> %lazy.shifted1980, i64 3, !dbg !41
  %.pre99 = extractelement <4 x float> %lazy.shifted1959, i64 0, !dbg !41
  %.pre101 = extractelement <4 x float> %lazy.shifted1959, i64 1, !dbg !41
  %.pre103 = extractelement <4 x float> %lazy.shifted1959, i64 2, !dbg !41
  %.pre105 = extractelement <4 x float> %lazy.shifted1959, i64 3, !dbg !41
  %.pre107 = extractelement <4 x float> %lazy.shifted1983, i64 0, !dbg !41
  %.pre109 = extractelement <4 x float> %lazy.shifted1983, i64 1, !dbg !41
  %.pre111 = extractelement <4 x float> %lazy.shifted1983, i64 2, !dbg !41
  %.pre113 = extractelement <4 x float> %lazy.shifted1983, i64 3, !dbg !41
  %.pre115 = extractelement <4 x float> %lazy.shifted1962, i64 0, !dbg !41
  %.pre117 = extractelement <4 x float> %lazy.shifted1962, i64 1, !dbg !41
  %.pre119 = extractelement <4 x float> %lazy.shifted1962, i64 2, !dbg !41
  %.pre121 = extractelement <4 x float> %lazy.shifted1962, i64 3, !dbg !41
  %.pre123 = extractelement <4 x float> %lazy.shifted1986, i64 0, !dbg !41
  %.pre125 = extractelement <4 x float> %lazy.shifted1986, i64 1, !dbg !41
  %.pre127 = extractelement <4 x float> %lazy.shifted1986, i64 2, !dbg !41
  %.pre129 = extractelement <4 x float> %lazy.shifted1986, i64 3, !dbg !41
  %.pre131 = extractelement <4 x float> %lazy.shifted1965, i64 0, !dbg !41
  %.pre133 = extractelement <4 x float> %lazy.shifted1965, i64 1, !dbg !41
  %.pre135 = extractelement <4 x float> %lazy.shifted1965, i64 2, !dbg !41
  %.pre137 = extractelement <4 x float> %lazy.shifted1965, i64 3, !dbg !41
  %.pre139 = extractelement <4 x float> %lazy.shifted1989, i64 0, !dbg !41
  %.pre141 = extractelement <4 x float> %lazy.shifted1989, i64 1, !dbg !41
  %.pre143 = extractelement <4 x float> %lazy.shifted1989, i64 2, !dbg !41
  %.pre145 = extractelement <4 x float> %lazy.shifted1989, i64 3, !dbg !41
  br label %lazy.join, !dbg !40

lazy.join:                                        ; preds = %lazy.slow, %116
  %.pre-phi146 = phi float [ %.pre145, %lazy.slow ], [ %425, %116 ], !dbg !41
  %.pre-phi144 = phi float [ %.pre143, %lazy.slow ], [ %424, %116 ], !dbg !41
  %.pre-phi142 = phi float [ %.pre141, %lazy.slow ], [ %423, %116 ], !dbg !41
  %.pre-phi140 = phi float [ %.pre139, %lazy.slow ], [ %422, %116 ], !dbg !41
  %.pre-phi138 = phi float [ %.pre137, %lazy.slow ], [ %377, %116 ], !dbg !41
  %.pre-phi136 = phi float [ %.pre135, %lazy.slow ], [ %376, %116 ], !dbg !41
  %.pre-phi134 = phi float [ %.pre133, %lazy.slow ], [ %375, %116 ], !dbg !41
  %.pre-phi132 = phi float [ %.pre131, %lazy.slow ], [ %374, %116 ], !dbg !41
  %.pre-phi130 = phi float [ %.pre129, %lazy.slow ], [ %420, %116 ], !dbg !41
  %.pre-phi128 = phi float [ %.pre127, %lazy.slow ], [ %419, %116 ], !dbg !41
  %.pre-phi126 = phi float [ %.pre125, %lazy.slow ], [ %418, %116 ], !dbg !41
  %.pre-phi124 = phi float [ %.pre123, %lazy.slow ], [ %417, %116 ], !dbg !41
  %.pre-phi122 = phi float [ %.pre121, %lazy.slow ], [ %371, %116 ], !dbg !41
  %.pre-phi120 = phi float [ %.pre119, %lazy.slow ], [ %370, %116 ], !dbg !41
  %.pre-phi118 = phi float [ %.pre117, %lazy.slow ], [ %369, %116 ], !dbg !41
  %.pre-phi116 = phi float [ %.pre115, %lazy.slow ], [ %368, %116 ], !dbg !41
  %.pre-phi114 = phi float [ %.pre113, %lazy.slow ], [ %414, %116 ], !dbg !41
  %.pre-phi112 = phi float [ %.pre111, %lazy.slow ], [ %413, %116 ], !dbg !41
  %.pre-phi110 = phi float [ %.pre109, %lazy.slow ], [ %412, %116 ], !dbg !41
  %.pre-phi108 = phi float [ %.pre107, %lazy.slow ], [ %411, %116 ], !dbg !41
  %.pre-phi106 = phi float [ %.pre105, %lazy.slow ], [ %365, %116 ], !dbg !41
  %.pre-phi104 = phi float [ %.pre103, %lazy.slow ], [ %364, %116 ], !dbg !41
  %.pre-phi102 = phi float [ %.pre101, %lazy.slow ], [ %363, %116 ], !dbg !41
  %.pre-phi100 = phi float [ %.pre99, %lazy.slow ], [ %362, %116 ], !dbg !41
  %.pre-phi98 = phi float [ %.pre97, %lazy.slow ], [ %408, %116 ], !dbg !41
  %.pre-phi96 = phi float [ %.pre95, %lazy.slow ], [ %407, %116 ], !dbg !41
  %.pre-phi94 = phi float [ %.pre93, %lazy.slow ], [ %406, %116 ], !dbg !41
  %.pre-phi92 = phi float [ %.pre91, %lazy.slow ], [ %405, %116 ], !dbg !41
  %.pre-phi90 = phi float [ %.pre89, %lazy.slow ], [ %359, %116 ], !dbg !41
  %.pre-phi88 = phi float [ %.pre87, %lazy.slow ], [ %358, %116 ], !dbg !41
  %.pre-phi86 = phi float [ %.pre85, %lazy.slow ], [ %357, %116 ], !dbg !41
  %.pre-phi84 = phi float [ %.pre83, %lazy.slow ], [ %356, %116 ], !dbg !41
  %.pre-phi82 = phi float [ %.pre81, %lazy.slow ], [ %402, %116 ], !dbg !41
  %.pre-phi80 = phi float [ %.pre79, %lazy.slow ], [ %401, %116 ], !dbg !41
  %.pre-phi78 = phi float [ %.pre77, %lazy.slow ], [ %400, %116 ], !dbg !41
  %.pre-phi76 = phi float [ %.pre75, %lazy.slow ], [ %399, %116 ], !dbg !41
  %.pre-phi74 = phi float [ %.pre73, %lazy.slow ], [ %353, %116 ], !dbg !41
  %.pre-phi72 = phi float [ %.pre71, %lazy.slow ], [ %352, %116 ], !dbg !41
  %.pre-phi70 = phi float [ %.pre69, %lazy.slow ], [ %351, %116 ], !dbg !41
  %.pre-phi68 = phi float [ %.pre67, %lazy.slow ], [ %350, %116 ], !dbg !41
  %.pre-phi66 = phi float [ %.pre65, %lazy.slow ], [ %396, %116 ], !dbg !41
  %.pre-phi64 = phi float [ %.pre63, %lazy.slow ], [ %395, %116 ], !dbg !41
  %.pre-phi62 = phi float [ %.pre61, %lazy.slow ], [ %394, %116 ], !dbg !41
  %.pre-phi60 = phi float [ %.pre59, %lazy.slow ], [ %393, %116 ], !dbg !41
  %.pre-phi58 = phi float [ %.pre57, %lazy.slow ], [ %345, %116 ], !dbg !41
  %.pre-phi56 = phi float [ %.pre55, %lazy.slow ], [ %344, %116 ], !dbg !41
  %.pre-phi54 = phi float [ %.pre53, %lazy.slow ], [ %343, %116 ], !dbg !41
  %.pre-phi52 = phi float [ %.pre51, %lazy.slow ], [ %342, %116 ], !dbg !41
  %.pre-phi50 = phi float [ %.pre49, %lazy.slow ], [ %390, %116 ], !dbg !41
  %.pre-phi48 = phi float [ %.pre47, %lazy.slow ], [ %389, %116 ], !dbg !41
  %.pre-phi46 = phi float [ %.pre45, %lazy.slow ], [ %388, %116 ], !dbg !41
  %.pre-phi44 = phi float [ %.pre43, %lazy.slow ], [ %387, %116 ], !dbg !41
  %.pre-phi42 = phi float [ %.pre41, %lazy.slow ], [ %335, %116 ], !dbg !41
  %.pre-phi40 = phi float [ %.pre39, %lazy.slow ], [ %334, %116 ], !dbg !41
  %.pre-phi38 = phi float [ %.pre37, %lazy.slow ], [ %333, %116 ], !dbg !41
  %.pre-phi36 = phi float [ %.pre35, %lazy.slow ], [ %332, %116 ], !dbg !41
  %.pre-phi34 = phi float [ %.pre33, %lazy.slow ], [ %384, %116 ], !dbg !41
  %.pre-phi32 = phi float [ %.pre31, %lazy.slow ], [ %383, %116 ], !dbg !41
  %.pre-phi30 = phi float [ %.pre29, %lazy.slow ], [ %382, %116 ], !dbg !41
  %.pre-phi28 = phi float [ %.pre27, %lazy.slow ], [ %381, %116 ], !dbg !41
  %.pre-phi26 = phi float [ %.pre25, %lazy.slow ], [ %327, %116 ], !dbg !41
  %.pre-phi24 = phi float [ %.pre23, %lazy.slow ], [ %326, %116 ], !dbg !41
  %.pre-phi22 = phi float [ %.pre21, %lazy.slow ], [ %325, %116 ], !dbg !41
  %.pre-phi = phi float [ %.pre, %lazy.slow ], [ %323, %116 ], !dbg !41
  %lazy.state2052 = phi [2 x <1 x float>] [ %lazy.state.next1993, %lazy.slow ], [ %121, %116 ], !dbg !40
  %lazy.state2053 = phi [2 x <8 x float>] [ %lazy.state.next2002, %lazy.slow ], [ %117, %116 ], !dbg !40
  %lazy.state2054 = phi [2 x <8 x float>] [ %lazy.state.next2012, %lazy.slow ], [ %118, %116 ], !dbg !40
  %lazy.state2055 = phi [2 x <8 x float>] [ %lazy.state.next2022, %lazy.slow ], [ %119, %116 ], !dbg !40
  %lazy.state2056 = phi [2 x <8 x float>] [ %lazy.state.next2032, %lazy.slow ], [ %122, %116 ], !dbg !40
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2036, %lazy.slow ], [ %120, %116 ], !dbg !40
  %444 = extractvalue [2 x <1 x float>] %lazy.state2052, 0, !dbg !40
  %445 = extractelement <1 x float> %444, i64 0, !dbg !40
  %446 = extractvalue [2 x <1 x float>] %lazy.state2052, 1, !dbg !40
  %447 = extractelement <1 x float> %446, i64 0, !dbg !40
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
  %448 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1408, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %449 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1415, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %address.local913 = shl nuw nsw i32 %address.thread498, 3, !dbg !42
  %address.local914 = and i32 %address.local913, 120, !dbg !42
  %address.local916 = and i32 %address.local559, 1536, !dbg !42
  %address.local917 = or disjoint i32 %address.local914, %address.local916, !dbg !42
  %address.local918 = zext nneg i32 %address.local917 to i64, !dbg !42
  %address.local919 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local918, !dbg !42
  %450 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %451 = bitcast <4 x i32> %450 to <8 x half>, !dbg !42
  %address.local920 = and i64 %address.local499, 60, !dbg !42
  %address.local921 = add nuw nsw i64 %address.local920, %address.local499, !dbg !42
  %address.local923.idx = shl nuw nsw i64 %address.local921, 4, !dbg !42
  %address.local923 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local923.idx, !dbg !42
  %address.local924 = lshr i64 %address.local539, 4, !dbg !42
  %address.local926.idx = mul nuw nsw i64 %address.local924, 768, !dbg !42
  %address.local926 = getelementptr i8, ptr addrspace(3) %address.local923, i64 %address.local926.idx, !dbg !42
  %address.local928.idx = mul nsw i64 %address.local526, -64, !dbg !42
  %address.local928 = getelementptr i8, ptr addrspace(3) %address.local926, i64 %address.local928.idx, !dbg !42
  %address.local929 = getelementptr i8, ptr addrspace(3) %address.local928, i64 256, !dbg !42
  %452 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %453 = bitcast <4 x i32> %452 to <8 x half>, !dbg !42
  %address.local933 = getelementptr i8, ptr addrspace(3) %address.local928, i64 512, !dbg !42
  %454 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %455 = bitcast <4 x i32> %454 to <8 x half>, !dbg !42
  %address.local937 = getelementptr i8, ptr addrspace(3) %address.local928, i64 768, !dbg !42
  %456 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %457 = bitcast <4 x i32> %456 to <8 x half>, !dbg !42
  %address.local939 = getelementptr i8, ptr addrspace(3) %address.local919, i64 4096, !dbg !42
  %458 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %459 = bitcast <4 x i32> %458 to <8 x half>, !dbg !42
  %address.local943 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4352, !dbg !42
  %460 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %461 = bitcast <4 x i32> %460 to <8 x half>, !dbg !42
  %address.local947 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4608, !dbg !42
  %462 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %463 = bitcast <4 x i32> %462 to <8 x half>, !dbg !42
  %address.local951 = getelementptr i8, ptr addrspace(3) %address.local928, i64 4864, !dbg !42
  %464 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %465 = shufflevector <8 x half> %451, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %466 = extractvalue [2 x <8 x float>] %lazy.state2056, 0, !dbg !44
  %467 = shufflevector <8 x float> %466, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %468 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !41
  %469 = insertelement <4 x float> poison, float %468, i64 0, !dbg !45
  %470 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi22), !dbg !41
  %471 = insertelement <4 x float> %469, float %470, i64 1, !dbg !45
  %472 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi24), !dbg !41
  %473 = insertelement <4 x float> %471, float %472, i64 2, !dbg !45
  %474 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi26), !dbg !41
  %475 = insertelement <4 x float> %473, float %474, i64 3, !dbg !45
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %475) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !45
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !45
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !45
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !45
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !45
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !45
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !45
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !45
  %476 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !45
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %476) #8, !dbg !41, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %477 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %465, <4 x float> %467) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %478 = shufflevector <8 x half> %451, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %479 = shufflevector <8 x float> %466, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %480 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %478, <4 x float> %479) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %481 = extractvalue [2 x <8 x float>] %lazy.state2056, 1, !dbg !44
  %482 = shufflevector <8 x float> %481, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %483 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi28), !dbg !41
  %484 = insertelement <4 x float> poison, float %483, i64 0, !dbg !45
  %485 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi30), !dbg !41
  %486 = insertelement <4 x float> %484, float %485, i64 1, !dbg !45
  %487 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi32), !dbg !41
  %488 = insertelement <4 x float> %486, float %487, i64 2, !dbg !45
  %489 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi34), !dbg !41
  %490 = insertelement <4 x float> %488, float %489, i64 3, !dbg !45
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %490) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02057 = extractelement <4 x float> %math.result388, i64 0, !dbg !45
  %.cvtpack.f12058 = extractelement <4 x float> %math.result388, i64 1, !dbg !45
  %.cvtpack.f22059 = extractelement <4 x float> %math.result388, i64 2, !dbg !45
  %.cvtpack.f32060 = extractelement <4 x float> %math.result388, i64 3, !dbg !45
  %.cvtpack.p02061 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02057, float %.cvtpack.f12058) #12, !dbg !45
  %.cvtpack.p12062 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22059, float %.cvtpack.f32060) #12, !dbg !45
  %.cvtpack.v02063 = insertelement <2 x i32> poison, i32 %.cvtpack.p02061, i64 0, !dbg !45
  %.cvtpack.v12064 = insertelement <2 x i32> %.cvtpack.v02063, i32 %.cvtpack.p12062, i64 1, !dbg !45
  %491 = bitcast <2 x i32> %.cvtpack.v12064 to <4 x half>, !dbg !45
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %491) #8, !dbg !41, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %492 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %465, <4 x float> %482) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %493 = shufflevector <8 x float> %481, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %478, <4 x float> %493) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %495 = shufflevector <8 x half> %453, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %496 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi36), !dbg !41
  %local.reduce89 = fadd float %468, %496, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %497 = insertelement <4 x float> poison, float %496, i64 0, !dbg !45
  %498 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi38), !dbg !41
  %local.reduce90 = fadd float %470, %498, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1860 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %499 = insertelement <4 x float> %497, float %498, i64 1, !dbg !45
  %500 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi40), !dbg !41
  %local.reduce91 = fadd float %472, %500, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1861 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %501 = insertelement <4 x float> %499, float %500, i64 2, !dbg !45
  %502 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi42), !dbg !41
  %local.reduce92 = fadd float %474, %502, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1862 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %503 = insertelement <4 x float> %501, float %502, i64 3, !dbg !45
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %503) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02065 = extractelement <4 x float> %math.result374, i64 0, !dbg !45
  %.cvtpack.f12066 = extractelement <4 x float> %math.result374, i64 1, !dbg !45
  %.cvtpack.f22067 = extractelement <4 x float> %math.result374, i64 2, !dbg !45
  %.cvtpack.f32068 = extractelement <4 x float> %math.result374, i64 3, !dbg !45
  %.cvtpack.p02069 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02065, float %.cvtpack.f12066) #12, !dbg !45
  %.cvtpack.p12070 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22067, float %.cvtpack.f32068) #12, !dbg !45
  %.cvtpack.v02071 = insertelement <2 x i32> poison, i32 %.cvtpack.p02069, i64 0, !dbg !45
  %.cvtpack.v12072 = insertelement <2 x i32> %.cvtpack.v02071, i32 %.cvtpack.p12070, i64 1, !dbg !45
  %504 = bitcast <2 x i32> %.cvtpack.v12072 to <4 x half>, !dbg !45
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %504) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %505 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %495, <4 x float> %477) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %506 = shufflevector <8 x half> %453, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %506, <4 x float> %480) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi44), !dbg !41
  %local.reduce = fadd float %483, %508, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1863 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %509 = insertelement <4 x float> poison, float %508, i64 0, !dbg !45
  %510 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi46), !dbg !41
  %local.reduce59 = fadd float %485, %510, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1864 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %511 = insertelement <4 x float> %509, float %510, i64 1, !dbg !45
  %512 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi48), !dbg !41
  %local.reduce60 = fadd float %487, %512, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1865 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %513 = insertelement <4 x float> %511, float %512, i64 2, !dbg !45
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi50), !dbg !41
  %local.reduce61 = fadd float %489, %514, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1866 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %515 = insertelement <4 x float> %513, float %514, i64 3, !dbg !45
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %515) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02073 = extractelement <4 x float> %math.result390, i64 0, !dbg !45
  %.cvtpack.f12074 = extractelement <4 x float> %math.result390, i64 1, !dbg !45
  %.cvtpack.f22075 = extractelement <4 x float> %math.result390, i64 2, !dbg !45
  %.cvtpack.f32076 = extractelement <4 x float> %math.result390, i64 3, !dbg !45
  %.cvtpack.p02077 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02073, float %.cvtpack.f12074) #12, !dbg !45
  %.cvtpack.p12078 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22075, float %.cvtpack.f32076) #12, !dbg !45
  %.cvtpack.v02079 = insertelement <2 x i32> poison, i32 %.cvtpack.p02077, i64 0, !dbg !45
  %.cvtpack.v12080 = insertelement <2 x i32> %.cvtpack.v02079, i32 %.cvtpack.p12078, i64 1, !dbg !45
  %516 = bitcast <2 x i32> %.cvtpack.v12080 to <4 x half>, !dbg !45
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %516) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %517 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %495, <4 x float> %492) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %518 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %506, <4 x float> %494) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %519 = shufflevector <8 x half> %455, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %520 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi52), !dbg !41
  %local.reduce93 = fadd float %520, %k0.sum.anchor, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1867 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %521 = insertelement <4 x float> poison, float %520, i64 0, !dbg !45
  %522 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi54), !dbg !41
  %local.reduce94 = fadd float %522, %k0.sum.anchor1860, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1868 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %523 = insertelement <4 x float> %521, float %522, i64 1, !dbg !45
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi56), !dbg !41
  %local.reduce95 = fadd float %524, %k0.sum.anchor1861, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1869 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %525 = insertelement <4 x float> %523, float %524, i64 2, !dbg !45
  %526 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi58), !dbg !41
  %local.reduce96 = fadd float %526, %k0.sum.anchor1862, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1870 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %527 = insertelement <4 x float> %525, float %526, i64 3, !dbg !45
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %527) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02081 = extractelement <4 x float> %math.result376, i64 0, !dbg !45
  %.cvtpack.f12082 = extractelement <4 x float> %math.result376, i64 1, !dbg !45
  %.cvtpack.f22083 = extractelement <4 x float> %math.result376, i64 2, !dbg !45
  %.cvtpack.f32084 = extractelement <4 x float> %math.result376, i64 3, !dbg !45
  %.cvtpack.p02085 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02081, float %.cvtpack.f12082) #12, !dbg !45
  %.cvtpack.p12086 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22083, float %.cvtpack.f32084) #12, !dbg !45
  %.cvtpack.v02087 = insertelement <2 x i32> poison, i32 %.cvtpack.p02085, i64 0, !dbg !45
  %.cvtpack.v12088 = insertelement <2 x i32> %.cvtpack.v02087, i32 %.cvtpack.p12086, i64 1, !dbg !45
  %528 = bitcast <2 x i32> %.cvtpack.v12088 to <4 x half>, !dbg !45
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %528) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %529 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %519, <4 x float> %505) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %530 = shufflevector <8 x half> %455, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %530, <4 x float> %507) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %532 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi60), !dbg !41
  %local.reduce62 = fadd float %532, %k0.sum.anchor1863, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1871 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %533 = insertelement <4 x float> poison, float %532, i64 0, !dbg !45
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi62), !dbg !41
  %local.reduce63 = fadd float %534, %k0.sum.anchor1864, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1872 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %535 = insertelement <4 x float> %533, float %534, i64 1, !dbg !45
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi64), !dbg !41
  %local.reduce64 = fadd float %536, %k0.sum.anchor1865, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1873 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %537 = insertelement <4 x float> %535, float %536, i64 2, !dbg !45
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi66), !dbg !41
  %local.reduce65 = fadd float %538, %k0.sum.anchor1866, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1874 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %539 = insertelement <4 x float> %537, float %538, i64 3, !dbg !45
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %539) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02089 = extractelement <4 x float> %math.result392, i64 0, !dbg !45
  %.cvtpack.f12090 = extractelement <4 x float> %math.result392, i64 1, !dbg !45
  %.cvtpack.f22091 = extractelement <4 x float> %math.result392, i64 2, !dbg !45
  %.cvtpack.f32092 = extractelement <4 x float> %math.result392, i64 3, !dbg !45
  %.cvtpack.p02093 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02089, float %.cvtpack.f12090) #12, !dbg !45
  %.cvtpack.p12094 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22091, float %.cvtpack.f32092) #12, !dbg !45
  %.cvtpack.v02095 = insertelement <2 x i32> poison, i32 %.cvtpack.p02093, i64 0, !dbg !45
  %.cvtpack.v12096 = insertelement <2 x i32> %.cvtpack.v02095, i32 %.cvtpack.p12094, i64 1, !dbg !45
  %540 = bitcast <2 x i32> %.cvtpack.v12096 to <4 x half>, !dbg !45
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %540) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %519, <4 x float> %517) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %542 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %530, <4 x float> %518) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %543 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %544 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi68), !dbg !41
  %local.reduce97 = fadd float %544, %k0.sum.anchor1867, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1875 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !45
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi70), !dbg !41
  %local.reduce98 = fadd float %546, %k0.sum.anchor1868, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1876 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %547 = insertelement <4 x float> %545, float %546, i64 1, !dbg !45
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi72), !dbg !41
  %local.reduce99 = fadd float %548, %k0.sum.anchor1869, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1877 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %549 = insertelement <4 x float> %547, float %548, i64 2, !dbg !45
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi74), !dbg !41
  %local.reduce100 = fadd float %550, %k0.sum.anchor1870, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1878 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %551 = insertelement <4 x float> %549, float %550, i64 3, !dbg !45
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %551) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02097 = extractelement <4 x float> %math.result378, i64 0, !dbg !45
  %.cvtpack.f12098 = extractelement <4 x float> %math.result378, i64 1, !dbg !45
  %.cvtpack.f22099 = extractelement <4 x float> %math.result378, i64 2, !dbg !45
  %.cvtpack.f32100 = extractelement <4 x float> %math.result378, i64 3, !dbg !45
  %.cvtpack.p02101 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02097, float %.cvtpack.f12098) #12, !dbg !45
  %.cvtpack.p12102 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22099, float %.cvtpack.f32100) #12, !dbg !45
  %.cvtpack.v02103 = insertelement <2 x i32> poison, i32 %.cvtpack.p02101, i64 0, !dbg !45
  %.cvtpack.v12104 = insertelement <2 x i32> %.cvtpack.v02103, i32 %.cvtpack.p12102, i64 1, !dbg !45
  %552 = bitcast <2 x i32> %.cvtpack.v12104 to <4 x half>, !dbg !45
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %552) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %553 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %543, <4 x float> %529) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %554 = shufflevector <8 x half> %457, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %554, <4 x float> %531) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi76), !dbg !41
  %local.reduce66 = fadd float %556, %k0.sum.anchor1871, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1879 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %557 = insertelement <4 x float> poison, float %556, i64 0, !dbg !45
  %558 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi78), !dbg !41
  %local.reduce67 = fadd float %558, %k0.sum.anchor1872, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1880 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %559 = insertelement <4 x float> %557, float %558, i64 1, !dbg !45
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi80), !dbg !41
  %local.reduce68 = fadd float %560, %k0.sum.anchor1873, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1881 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %561 = insertelement <4 x float> %559, float %560, i64 2, !dbg !45
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi82), !dbg !41
  %local.reduce69 = fadd float %562, %k0.sum.anchor1874, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1882 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> %561, float %562, i64 3, !dbg !45
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %563) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02105 = extractelement <4 x float> %math.result394, i64 0, !dbg !45
  %.cvtpack.f12106 = extractelement <4 x float> %math.result394, i64 1, !dbg !45
  %.cvtpack.f22107 = extractelement <4 x float> %math.result394, i64 2, !dbg !45
  %.cvtpack.f32108 = extractelement <4 x float> %math.result394, i64 3, !dbg !45
  %.cvtpack.p02109 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02105, float %.cvtpack.f12106) #12, !dbg !45
  %.cvtpack.p12110 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22107, float %.cvtpack.f32108) #12, !dbg !45
  %.cvtpack.v02111 = insertelement <2 x i32> poison, i32 %.cvtpack.p02109, i64 0, !dbg !45
  %.cvtpack.v12112 = insertelement <2 x i32> %.cvtpack.v02111, i32 %.cvtpack.p12110, i64 1, !dbg !45
  %564 = bitcast <2 x i32> %.cvtpack.v12112 to <4 x half>, !dbg !45
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %564) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %543, <4 x float> %541) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %566 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %554, <4 x float> %542) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %567 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %568 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi84), !dbg !41
  %local.reduce101 = fadd float %568, %k0.sum.anchor1875, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1883 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %569 = insertelement <4 x float> poison, float %568, i64 0, !dbg !45
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi86), !dbg !41
  %local.reduce102 = fadd float %570, %k0.sum.anchor1876, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1884 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %571 = insertelement <4 x float> %569, float %570, i64 1, !dbg !45
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi88), !dbg !41
  %local.reduce103 = fadd float %572, %k0.sum.anchor1877, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1885 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %573 = insertelement <4 x float> %571, float %572, i64 2, !dbg !45
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi90), !dbg !41
  %local.reduce104 = fadd float %574, %k0.sum.anchor1878, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1886 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %575 = insertelement <4 x float> %573, float %574, i64 3, !dbg !45
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %575) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02113 = extractelement <4 x float> %math.result380, i64 0, !dbg !45
  %.cvtpack.f12114 = extractelement <4 x float> %math.result380, i64 1, !dbg !45
  %.cvtpack.f22115 = extractelement <4 x float> %math.result380, i64 2, !dbg !45
  %.cvtpack.f32116 = extractelement <4 x float> %math.result380, i64 3, !dbg !45
  %.cvtpack.p02117 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02113, float %.cvtpack.f12114) #12, !dbg !45
  %.cvtpack.p12118 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22115, float %.cvtpack.f32116) #12, !dbg !45
  %.cvtpack.v02119 = insertelement <2 x i32> poison, i32 %.cvtpack.p02117, i64 0, !dbg !45
  %.cvtpack.v12120 = insertelement <2 x i32> %.cvtpack.v02119, i32 %.cvtpack.p12118, i64 1, !dbg !45
  %576 = bitcast <2 x i32> %.cvtpack.v12120 to <4 x half>, !dbg !45
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %576) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %577 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %567, <4 x float> %553) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %578 = shufflevector <8 x half> %459, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %579 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %578, <4 x float> %555) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi92), !dbg !41
  %local.reduce70 = fadd float %580, %k0.sum.anchor1879, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1887 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %581 = insertelement <4 x float> poison, float %580, i64 0, !dbg !45
  %582 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi94), !dbg !41
  %local.reduce71 = fadd float %582, %k0.sum.anchor1880, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1888 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %583 = insertelement <4 x float> %581, float %582, i64 1, !dbg !45
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi96), !dbg !41
  %local.reduce72 = fadd float %584, %k0.sum.anchor1881, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1889 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %583, float %584, i64 2, !dbg !45
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi98), !dbg !41
  %local.reduce73 = fadd float %586, %k0.sum.anchor1882, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1890 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %587 = insertelement <4 x float> %585, float %586, i64 3, !dbg !45
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %587) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02121 = extractelement <4 x float> %math.result396, i64 0, !dbg !45
  %.cvtpack.f12122 = extractelement <4 x float> %math.result396, i64 1, !dbg !45
  %.cvtpack.f22123 = extractelement <4 x float> %math.result396, i64 2, !dbg !45
  %.cvtpack.f32124 = extractelement <4 x float> %math.result396, i64 3, !dbg !45
  %.cvtpack.p02125 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02121, float %.cvtpack.f12122) #12, !dbg !45
  %.cvtpack.p12126 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22123, float %.cvtpack.f32124) #12, !dbg !45
  %.cvtpack.v02127 = insertelement <2 x i32> poison, i32 %.cvtpack.p02125, i64 0, !dbg !45
  %.cvtpack.v12128 = insertelement <2 x i32> %.cvtpack.v02127, i32 %.cvtpack.p12126, i64 1, !dbg !45
  %588 = bitcast <2 x i32> %.cvtpack.v12128 to <4 x half>, !dbg !45
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %588) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %567, <4 x float> %565) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %590 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %578, <4 x float> %566) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %591 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %592 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi100), !dbg !41
  %local.reduce105 = fadd float %592, %k0.sum.anchor1883, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1891 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %593 = insertelement <4 x float> poison, float %592, i64 0, !dbg !45
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi102), !dbg !41
  %local.reduce106 = fadd float %594, %k0.sum.anchor1884, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1892 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> %593, float %594, i64 1, !dbg !45
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi104), !dbg !41
  %local.reduce107 = fadd float %596, %k0.sum.anchor1885, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1893 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %597 = insertelement <4 x float> %595, float %596, i64 2, !dbg !45
  %598 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi106), !dbg !41
  %local.reduce108 = fadd float %598, %k0.sum.anchor1886, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1894 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %599 = insertelement <4 x float> %597, float %598, i64 3, !dbg !45
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %599) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02129 = extractelement <4 x float> %math.result382, i64 0, !dbg !45
  %.cvtpack.f12130 = extractelement <4 x float> %math.result382, i64 1, !dbg !45
  %.cvtpack.f22131 = extractelement <4 x float> %math.result382, i64 2, !dbg !45
  %.cvtpack.f32132 = extractelement <4 x float> %math.result382, i64 3, !dbg !45
  %.cvtpack.p02133 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02129, float %.cvtpack.f12130) #12, !dbg !45
  %.cvtpack.p12134 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22131, float %.cvtpack.f32132) #12, !dbg !45
  %.cvtpack.v02135 = insertelement <2 x i32> poison, i32 %.cvtpack.p02133, i64 0, !dbg !45
  %.cvtpack.v12136 = insertelement <2 x i32> %.cvtpack.v02135, i32 %.cvtpack.p12134, i64 1, !dbg !45
  %600 = bitcast <2 x i32> %.cvtpack.v12136 to <4 x half>, !dbg !45
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %600) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %601 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %591, <4 x float> %577) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %602 = shufflevector <8 x half> %461, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %602, <4 x float> %579) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %604 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi108), !dbg !41
  %local.reduce74 = fadd float %604, %k0.sum.anchor1887, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1895 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %605 = insertelement <4 x float> poison, float %604, i64 0, !dbg !45
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi110), !dbg !41
  %local.reduce75 = fadd float %606, %k0.sum.anchor1888, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1896 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %607 = insertelement <4 x float> %605, float %606, i64 1, !dbg !45
  %608 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi112), !dbg !41
  %local.reduce76 = fadd float %608, %k0.sum.anchor1889, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1897 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %609 = insertelement <4 x float> %607, float %608, i64 2, !dbg !45
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi114), !dbg !41
  %local.reduce77 = fadd float %610, %k0.sum.anchor1890, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1898 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> %609, float %610, i64 3, !dbg !45
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %611) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02137 = extractelement <4 x float> %math.result398, i64 0, !dbg !45
  %.cvtpack.f12138 = extractelement <4 x float> %math.result398, i64 1, !dbg !45
  %.cvtpack.f22139 = extractelement <4 x float> %math.result398, i64 2, !dbg !45
  %.cvtpack.f32140 = extractelement <4 x float> %math.result398, i64 3, !dbg !45
  %.cvtpack.p02141 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02137, float %.cvtpack.f12138) #12, !dbg !45
  %.cvtpack.p12142 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22139, float %.cvtpack.f32140) #12, !dbg !45
  %.cvtpack.v02143 = insertelement <2 x i32> poison, i32 %.cvtpack.p02141, i64 0, !dbg !45
  %.cvtpack.v12144 = insertelement <2 x i32> %.cvtpack.v02143, i32 %.cvtpack.p12142, i64 1, !dbg !45
  %612 = bitcast <2 x i32> %.cvtpack.v12144 to <4 x half>, !dbg !45
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %612) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %613 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %591, <4 x float> %589) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %614 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %602, <4 x float> %590) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %615 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi116), !dbg !41
  %local.reduce109 = fadd float %616, %k0.sum.anchor1891, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1899 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %617 = insertelement <4 x float> poison, float %616, i64 0, !dbg !45
  %618 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi118), !dbg !41
  %local.reduce110 = fadd float %618, %k0.sum.anchor1892, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1900 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %619 = insertelement <4 x float> %617, float %618, i64 1, !dbg !45
  %620 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi120), !dbg !41
  %local.reduce111 = fadd float %620, %k0.sum.anchor1893, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1901 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %621 = insertelement <4 x float> %619, float %620, i64 2, !dbg !45
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi122), !dbg !41
  %local.reduce112 = fadd float %622, %k0.sum.anchor1894, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1902 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %623 = insertelement <4 x float> %621, float %622, i64 3, !dbg !45
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02145 = extractelement <4 x float> %math.result384, i64 0, !dbg !45
  %.cvtpack.f12146 = extractelement <4 x float> %math.result384, i64 1, !dbg !45
  %.cvtpack.f22147 = extractelement <4 x float> %math.result384, i64 2, !dbg !45
  %.cvtpack.f32148 = extractelement <4 x float> %math.result384, i64 3, !dbg !45
  %.cvtpack.p02149 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02145, float %.cvtpack.f12146) #12, !dbg !45
  %.cvtpack.p12150 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22147, float %.cvtpack.f32148) #12, !dbg !45
  %.cvtpack.v02151 = insertelement <2 x i32> poison, i32 %.cvtpack.p02149, i64 0, !dbg !45
  %.cvtpack.v12152 = insertelement <2 x i32> %.cvtpack.v02151, i32 %.cvtpack.p12150, i64 1, !dbg !45
  %624 = bitcast <2 x i32> %.cvtpack.v12152 to <4 x half>, !dbg !45
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %625 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %615, <4 x float> %601) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %626 = shufflevector <8 x half> %463, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %626, <4 x float> %603) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi124), !dbg !41
  %local.reduce78 = fadd float %628, %k0.sum.anchor1895, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1903 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !45
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi126), !dbg !41
  %local.reduce79 = fadd float %630, %k0.sum.anchor1896, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1904 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %631 = insertelement <4 x float> %629, float %630, i64 1, !dbg !45
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi128), !dbg !41
  %local.reduce80 = fadd float %632, %k0.sum.anchor1897, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1905 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %631, float %632, i64 2, !dbg !45
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi130), !dbg !41
  %local.reduce81 = fadd float %634, %k0.sum.anchor1898, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1906 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %635 = insertelement <4 x float> %633, float %634, i64 3, !dbg !45
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %635) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02153 = extractelement <4 x float> %math.result400, i64 0, !dbg !45
  %.cvtpack.f12154 = extractelement <4 x float> %math.result400, i64 1, !dbg !45
  %.cvtpack.f22155 = extractelement <4 x float> %math.result400, i64 2, !dbg !45
  %.cvtpack.f32156 = extractelement <4 x float> %math.result400, i64 3, !dbg !45
  %.cvtpack.p02157 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02153, float %.cvtpack.f12154) #12, !dbg !45
  %.cvtpack.p12158 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22155, float %.cvtpack.f32156) #12, !dbg !45
  %.cvtpack.v02159 = insertelement <2 x i32> poison, i32 %.cvtpack.p02157, i64 0, !dbg !45
  %.cvtpack.v12160 = insertelement <2 x i32> %.cvtpack.v02159, i32 %.cvtpack.p12158, i64 1, !dbg !45
  %636 = bitcast <2 x i32> %.cvtpack.v12160 to <4 x half>, !dbg !45
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %636) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %615, <4 x float> %613) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %626, <4 x float> %614) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %639 = bitcast <4 x i32> %464 to <8 x half>, !dbg !42
  %640 = shufflevector <8 x half> %639, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %641 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi132), !dbg !41
  %local.reduce113 = fadd float %641, %k0.sum.anchor1899, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1907 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %642 = insertelement <4 x float> poison, float %641, i64 0, !dbg !45
  %643 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi134), !dbg !41
  %local.reduce114 = fadd float %643, %k0.sum.anchor1900, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1908 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %local.reduce117 = fadd float %k0.sum.anchor1907, %k0.sum.anchor1908, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1909 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %644 = insertelement <4 x float> %642, float %643, i64 1, !dbg !45
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi136), !dbg !41
  %local.reduce115 = fadd float %645, %k0.sum.anchor1901, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1910 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> %644, float %645, i64 2, !dbg !45
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi138), !dbg !41
  %local.reduce116 = fadd float %647, %k0.sum.anchor1902, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1911 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %local.reduce118 = fadd float %k0.sum.anchor1910, %k0.sum.anchor1911, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1912 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %648 = fadd float %k0.sum.anchor1909, %k0.sum.anchor1912, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1913 = tail call float asm sideeffect "", "=v,0"(float %648) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %649 = fadd float %445, %k0.sum.anchor1913, !dbg !46
  %k0.sum.anchor1914 = tail call float asm sideeffect "", "=v,0"(float %649) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %650 = insertelement <1 x float> poison, float %k0.sum.anchor1914, i64 0, !dbg !12
  %651 = insertvalue [2 x <1 x float>] poison, <1 x float> %650, 0, !dbg !12
  %652 = insertelement <4 x float> %646, float %647, i64 3, !dbg !45
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %652) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02161 = extractelement <4 x float> %math.result386, i64 0, !dbg !45
  %.cvtpack.f12162 = extractelement <4 x float> %math.result386, i64 1, !dbg !45
  %.cvtpack.f22163 = extractelement <4 x float> %math.result386, i64 2, !dbg !45
  %.cvtpack.f32164 = extractelement <4 x float> %math.result386, i64 3, !dbg !45
  %.cvtpack.p02165 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02161, float %.cvtpack.f12162) #12, !dbg !45
  %.cvtpack.p12166 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22163, float %.cvtpack.f32164) #12, !dbg !45
  %.cvtpack.v02167 = insertelement <2 x i32> poison, i32 %.cvtpack.p02165, i64 0, !dbg !45
  %.cvtpack.v12168 = insertelement <2 x i32> %.cvtpack.v02167, i32 %.cvtpack.p12166, i64 1, !dbg !45
  %653 = bitcast <2 x i32> %.cvtpack.v12168 to <4 x half>, !dbg !45
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %653) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %654 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %640, <4 x float> %625) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %655 = shufflevector <8 x half> %639, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %656 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %655, <4 x float> %627) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %657 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi140), !dbg !41
  %local.reduce82 = fadd float %657, %k0.sum.anchor1903, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1915 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %658 = insertelement <4 x float> poison, float %657, i64 0, !dbg !45
  %659 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi142), !dbg !41
  %local.reduce83 = fadd float %659, %k0.sum.anchor1904, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1916 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %local.reduce86 = fadd float %k0.sum.anchor1915, %k0.sum.anchor1916, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1917 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %660 = insertelement <4 x float> %658, float %659, i64 1, !dbg !45
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi144), !dbg !41
  %local.reduce84 = fadd float %661, %k0.sum.anchor1905, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1918 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %660, float %661, i64 2, !dbg !45
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi146), !dbg !41
  %local.reduce85 = fadd float %663, %k0.sum.anchor1906, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1919 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %local.reduce87 = fadd float %k0.sum.anchor1918, %k0.sum.anchor1919, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1920 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %664 = fadd float %k0.sum.anchor1917, %k0.sum.anchor1920, !dbg !46, !frisk.local_reduction !5
  %k0.sum.anchor1921 = tail call float asm sideeffect "", "=v,0"(float %664) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %665 = fadd float %447, %k0.sum.anchor1921, !dbg !46
  %k0.sum.anchor1922 = tail call float asm sideeffect "", "=v,0"(float %665) #8, !dbg !46, !frisk.cross_bn_sum_anchor !5
  %666 = insertelement <1 x float> poison, float %k0.sum.anchor1922, i64 0, !dbg !12
  %667 = insertvalue [2 x <1 x float>] %651, <1 x float> %666, 1, !dbg !12
  %668 = insertelement <4 x float> %662, float %663, i64 3, !dbg !45
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %668) #8, !dbg !45, !frisk.register_lifetimes !5
  %.cvtpack.f02169 = extractelement <4 x float> %math.result402, i64 0, !dbg !45
  %.cvtpack.f12170 = extractelement <4 x float> %math.result402, i64 1, !dbg !45
  %.cvtpack.f22171 = extractelement <4 x float> %math.result402, i64 2, !dbg !45
  %.cvtpack.f32172 = extractelement <4 x float> %math.result402, i64 3, !dbg !45
  %.cvtpack.p02173 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02169, float %.cvtpack.f12170) #12, !dbg !45
  %.cvtpack.p12174 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22171, float %.cvtpack.f32172) #12, !dbg !45
  %.cvtpack.v02175 = insertelement <2 x i32> poison, i32 %.cvtpack.p02173, i64 0, !dbg !45
  %.cvtpack.v12176 = insertelement <2 x i32> %.cvtpack.v02175, i32 %.cvtpack.p12174, i64 1, !dbg !45
  %669 = bitcast <2 x i32> %.cvtpack.v12176 to <4 x half>, !dbg !45
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %669) #8, !dbg !46, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %640, <4 x float> %637) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %671 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %655, <4 x float> %638) #8, !dbg !42, !frisk.pv_interleave !19
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
  %672 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %vdb.address1430, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %673 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1437, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local919, i64 8192, !dbg !42
  %674 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %675 = bitcast <4 x i32> %674 to <8 x half>, !dbg !42
  %vdb.read1421 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8448, !dbg !42
  %676 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %677 = bitcast <4 x i32> %676 to <8 x half>, !dbg !42
  %vdb.read1422 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8704, !dbg !42
  %678 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %679 = bitcast <4 x i32> %678 to <8 x half>, !dbg !42
  %vdb.read1423 = getelementptr i8, ptr addrspace(3) %address.local928, i64 8960, !dbg !42
  %680 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %681 = bitcast <4 x i32> %680 to <8 x half>, !dbg !42
  %vdb.read1424 = getelementptr i8, ptr addrspace(3) %address.local919, i64 12288, !dbg !42
  %682 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %683 = bitcast <4 x i32> %682 to <8 x half>, !dbg !42
  %vdb.read1425 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12544, !dbg !42
  %684 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %685 = bitcast <4 x i32> %684 to <8 x half>, !dbg !42
  %vdb.read1426 = getelementptr i8, ptr addrspace(3) %address.local928, i64 12800, !dbg !42
  %686 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %687 = bitcast <4 x i32> %686 to <8 x half>, !dbg !42
  %vdb.read1427 = getelementptr i8, ptr addrspace(3) %address.local928, i64 13056, !dbg !42
  %688 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %689 = shufflevector <8 x half> %675, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %690 = extractvalue [2 x <8 x float>] %lazy.state2055, 0, !dbg !44
  %691 = shufflevector <8 x float> %690, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %692 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %689, <4 x float> %691) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %693 = shufflevector <8 x half> %675, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %694 = shufflevector <8 x float> %690, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %695 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %693, <4 x float> %694) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %696 = extractvalue [2 x <8 x float>] %lazy.state2055, 1, !dbg !44
  %697 = shufflevector <8 x float> %696, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %698 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %689, <4 x float> %697) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %699 = shufflevector <8 x float> %696, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %693, <4 x float> %699) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %701 = shufflevector <8 x half> %677, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %701, <4 x float> %692) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %703 = shufflevector <8 x half> %677, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %704 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %703, <4 x float> %695) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %705 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %701, <4 x float> %698) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %706 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %703, <4 x float> %700) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %707 = shufflevector <8 x half> %679, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %708 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %707, <4 x float> %702) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %709 = shufflevector <8 x half> %679, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %710 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %709, <4 x float> %704) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %711 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %707, <4 x float> %705) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %712 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %709, <4 x float> %706) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %713 = shufflevector <8 x half> %681, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %714 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %713, <4 x float> %708) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %715 = shufflevector <8 x half> %681, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %715, <4 x float> %710) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %717 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %713, <4 x float> %711) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %718 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %715, <4 x float> %712) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %719 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %720 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %719, <4 x float> %714) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %721 = shufflevector <8 x half> %683, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %722 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %721, <4 x float> %716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %723 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %719, <4 x float> %717) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %724 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %721, <4 x float> %718) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %725 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %726 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %725, <4 x float> %720) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %727 = shufflevector <8 x half> %685, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %728 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %727, <4 x float> %722) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %729 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %725, <4 x float> %723) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %730 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %727, <4 x float> %724) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %731 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %731, <4 x float> %726) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %733 = shufflevector <8 x half> %687, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %733, <4 x float> %728) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %735 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %731, <4 x float> %729) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %736 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %733, <4 x float> %730) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %737 = bitcast <4 x i32> %688 to <8 x half>, !dbg !42
  %738 = shufflevector <8 x half> %737, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %739 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %738, <4 x float> %732) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %740 = shufflevector <8 x half> %737, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %741 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %740, <4 x float> %734) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %738, <4 x float> %735) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %740, <4 x float> %736) #8, !dbg !42, !frisk.pv_interleave !19
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
  %744 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1444, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %745 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1420, i32 %vdb.address1451, <4 x i32> %dma.buffer.rsrc1327) #9, !dbg !38
  %746 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local919, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %747 = bitcast <4 x i32> %746 to <8 x half>, !dbg !42
  %748 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local929, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %749 = bitcast <4 x i32> %748 to <8 x half>, !dbg !42
  %750 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local933, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %751 = bitcast <4 x i32> %750 to <8 x half>, !dbg !42
  %752 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local937, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %753 = bitcast <4 x i32> %752 to <8 x half>, !dbg !42
  %754 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local939, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %755 = bitcast <4 x i32> %754 to <8 x half>, !dbg !42
  %756 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local943, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %757 = bitcast <4 x i32> %756 to <8 x half>, !dbg !42
  %758 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local947, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %759 = bitcast <4 x i32> %758 to <8 x half>, !dbg !42
  %760 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local951, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %761 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %762 = extractvalue [2 x <8 x float>] %lazy.state2054, 0, !dbg !44
  %763 = shufflevector <8 x float> %762, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %761, <4 x float> %763) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %765 = shufflevector <8 x half> %747, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %766 = shufflevector <8 x float> %762, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %767 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %765, <4 x float> %766) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %768 = extractvalue [2 x <8 x float>] %lazy.state2054, 1, !dbg !44
  %769 = shufflevector <8 x float> %768, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %761, <4 x float> %769) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %771 = shufflevector <8 x float> %768, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %765, <4 x float> %771) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %773 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %774 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %773, <4 x float> %764) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %775 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %775, <4 x float> %767) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %777 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %773, <4 x float> %770) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %778 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %775, <4 x float> %772) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %779 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %779, <4 x float> %774) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %781 = shufflevector <8 x half> %751, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %782 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %781, <4 x float> %776) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %783 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %779, <4 x float> %777) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %781, <4 x float> %778) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %785 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %785, <4 x float> %780) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %787 = shufflevector <8 x half> %753, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %788 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %787, <4 x float> %782) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %785, <4 x float> %783) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %790 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %787, <4 x float> %784) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %791 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %791, <4 x float> %786) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %793 = shufflevector <8 x half> %755, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %793, <4 x float> %788) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %791, <4 x float> %789) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %796 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %793, <4 x float> %790) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %797 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %797, <4 x float> %792) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %799 = shufflevector <8 x half> %757, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %799, <4 x float> %794) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %797, <4 x float> %795) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %802 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %799, <4 x float> %796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %803 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %803, <4 x float> %798) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %805 = shufflevector <8 x half> %759, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %805, <4 x float> %800) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %803, <4 x float> %801) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %805, <4 x float> %802) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %809 = bitcast <4 x i32> %760 to <8 x half>, !dbg !42
  %810 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %810, <4 x float> %804) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %812 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %812, <4 x float> %806) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %814 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %810, <4 x float> %807) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %812, <4 x float> %808) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %816 = add nuw nsw i64 %123, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %123, 31, !dbg !42
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !42

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %816, 14, !dbg !14
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

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %817 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %818 = bitcast <4 x i32> %817 to <8 x half>, !dbg !42
  %819 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1421, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %820 = bitcast <4 x i32> %819 to <8 x half>, !dbg !42
  %821 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1422, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %822 = bitcast <4 x i32> %821 to <8 x half>, !dbg !42
  %823 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1423, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %824 = bitcast <4 x i32> %823 to <8 x half>, !dbg !42
  %825 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1424, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %826 = bitcast <4 x i32> %825 to <8 x half>, !dbg !42
  %827 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1425, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %828 = bitcast <4 x i32> %827 to <8 x half>, !dbg !42
  %829 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1426, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %830 = bitcast <4 x i32> %829 to <8 x half>, !dbg !42
  %831 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1427, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %832 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  %833 = extractvalue [2 x <8 x float>] %lazy.state2053, 0, !dbg !44
  %834 = shufflevector <8 x float> %833, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %835 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %832, <4 x float> %834) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %836 = shufflevector <8 x half> %818, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  %837 = shufflevector <8 x float> %833, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %836, <4 x float> %837) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %839 = extractvalue [2 x <8 x float>] %lazy.state2053, 1, !dbg !44
  %840 = shufflevector <8 x float> %839, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %832, <4 x float> %840) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %842 = shufflevector <8 x float> %839, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %836, <4 x float> %842) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %844 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %844, <4 x float> %835) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %846 = shufflevector <8 x half> %820, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %846, <4 x float> %838) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %844, <4 x float> %841) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %846, <4 x float> %843) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %850 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %850, <4 x float> %845) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %852 = shufflevector <8 x half> %822, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %852, <4 x float> %847) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %850, <4 x float> %848) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %852, <4 x float> %849) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %856 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %856, <4 x float> %851) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %858 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %858, <4 x float> %853) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %856, <4 x float> %854) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %858, <4 x float> %855) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %862 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %862, <4 x float> %857) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %864 = shufflevector <8 x half> %826, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %864, <4 x float> %859) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %862, <4 x float> %860) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %864, <4 x float> %861) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %868 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %868, <4 x float> %863) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %870 = shufflevector <8 x half> %828, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %870, <4 x float> %865) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %872 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %868, <4 x float> %866) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %870, <4 x float> %867) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %874 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %874, <4 x float> %869) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %876 = shufflevector <8 x half> %830, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %876, <4 x float> %871) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %878 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %874, <4 x float> %872) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %876, <4 x float> %873) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %880 = bitcast <4 x i32> %831 to <8 x half>, !dbg !42
  %881 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %881, <4 x float> %875) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %883 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %883, <4 x float> %877) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %881, <4 x float> %878) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %883, <4 x float> %879) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %887 = shufflevector <4 x float> %654, <4 x float> %656, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %888 = insertvalue [2 x <8 x float>] poison, <8 x float> %887, 0, !dbg !12
  %889 = shufflevector <4 x float> %670, <4 x float> %671, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %890 = insertvalue [2 x <8 x float>] %888, <8 x float> %889, 1, !dbg !12
  %891 = shufflevector <4 x float> %739, <4 x float> %741, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %892 = insertvalue [2 x <8 x float>] poison, <8 x float> %891, 0, !dbg !12
  %893 = shufflevector <4 x float> %742, <4 x float> %743, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %894 = insertvalue [2 x <8 x float>] %892, <8 x float> %893, 1, !dbg !12
  %895 = shufflevector <4 x float> %811, <4 x float> %813, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %896 = insertvalue [2 x <8 x float>] poison, <8 x float> %895, 0, !dbg !12
  %897 = shufflevector <4 x float> %814, <4 x float> %815, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %898 = insertvalue [2 x <8 x float>] %896, <8 x float> %897, 1, !dbg !12
  %899 = shufflevector <4 x float> %882, <4 x float> %884, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %900 = insertvalue [2 x <8 x float>] poison, <8 x float> %899, 0, !dbg !12
  %901 = shufflevector <4 x float> %885, <4 x float> %886, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %902 = insertvalue [2 x <8 x float>] %900, <8 x float> %901, 1, !dbg !12
  %903 = icmp samesign ult i64 %123, 31, !dbg !12
  br i1 %903, label %116, label %904, !dbg !12

904:                                              ; preds = %k0.join
  %905 = shl nuw nsw i32 %8, 2, !dbg !12
  %906 = xor i32 %905, 64, !dbg !12
  %907 = bitcast float %k0.sum.anchor1922 to i32, !dbg !12
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %906, i32 %907), !dbg !12
  %908 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !12
  %lane.sum.final189 = fadd float %k0.sum.anchor1922, %908, !dbg !12
  %909 = xor i32 %905, 128, !dbg !12
  %910 = bitcast float %lane.sum.final189 to i32, !dbg !12
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %909, i32 %910), !dbg !12
  %911 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !12
  %lane.sum.final191 = fadd float %lane.sum.final189, %911, !dbg !12
  %912 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !12
  %913 = bitcast float %k0.sum.anchor1914 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %906, i32 %913), !dbg !12
  %914 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor1914, %914, !dbg !12
  %915 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %909, i32 %915), !dbg !12
  %916 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !12
  %lane.sum.final183 = fadd float %lane.sum.final, %916, !dbg !12
  %917 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !12
  %918 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %918, 0, !dbg !12
  %919 = and i32 %kdma.tid, 32, !dbg !12
  %.not36 = icmp eq i32 %919, 0, !dbg !12
  %920 = extractelement <4 x float> %882, i64 0, !dbg !12
  %921 = extractelement <4 x float> %882, i64 1, !dbg !12
  %922 = extractelement <4 x float> %882, i64 2, !dbg !12
  %923 = extractelement <4 x float> %882, i64 3, !dbg !12
  %924 = select i1 %.not, float %921, float %920, !dbg !12
  %925 = bitcast float %924 to i32, !dbg !12
  %926 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %925), !dbg !12
  %927 = bitcast i32 %926 to float, !dbg !12
  %928 = select i1 %.not, float %920, float %927, !dbg !12
  %929 = select i1 %.not, float %927, float %921, !dbg !12
  %930 = select i1 %.not, float %923, float %922, !dbg !12
  %931 = bitcast float %930 to i32, !dbg !12
  %932 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %931), !dbg !12
  %933 = bitcast i32 %932 to float, !dbg !12
  %934 = select i1 %.not, float %922, float %933, !dbg !12
  %935 = select i1 %.not, float %933, float %923, !dbg !12
  %936 = select i1 %.not36, float %934, float %928, !dbg !12
  %937 = bitcast float %936 to i32, !dbg !12
  %938 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %937), !dbg !12
  %939 = bitcast i32 %938 to float, !dbg !12
  %940 = select i1 %.not36, float %935, float %929, !dbg !12
  %941 = bitcast float %940 to i32, !dbg !12
  %942 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %941), !dbg !12
  %943 = bitcast i32 %942 to float, !dbg !12
  %944 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !12
  %945 = shufflevector <4 x i1> %944, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %946 = insertelement <4 x float> poison, float %928, i64 0, !dbg !12
  %947 = insertelement <4 x float> %946, float %929, i64 1, !dbg !12
  %948 = insertelement <4 x float> %947, float %939, i64 2, !dbg !12
  %949 = insertelement <4 x float> %948, float %943, i64 3, !dbg !12
  %950 = insertelement <4 x float> poison, float %939, i64 0, !dbg !12
  %951 = insertelement <4 x float> %950, float %943, i64 1, !dbg !12
  %952 = insertelement <4 x float> %951, float %934, i64 2, !dbg !12
  %953 = insertelement <4 x float> %952, float %935, i64 3, !dbg !12
  %954 = select <4 x i1> %945, <4 x float> %949, <4 x float> %953, !dbg !12
  %955 = extractelement <4 x float> %884, i64 0, !dbg !12
  %956 = extractelement <4 x float> %884, i64 1, !dbg !12
  %957 = extractelement <4 x float> %884, i64 2, !dbg !12
  %958 = extractelement <4 x float> %884, i64 3, !dbg !12
  %959 = select i1 %.not, float %956, float %955, !dbg !12
  %960 = bitcast float %959 to i32, !dbg !12
  %961 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %960), !dbg !12
  %962 = bitcast i32 %961 to float, !dbg !12
  %963 = select i1 %.not, float %955, float %962, !dbg !12
  %964 = select i1 %.not, float %962, float %956, !dbg !12
  %965 = select i1 %.not, float %958, float %957, !dbg !12
  %966 = bitcast float %965 to i32, !dbg !12
  %967 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %966), !dbg !12
  %968 = bitcast i32 %967 to float, !dbg !12
  %969 = select i1 %.not, float %957, float %968, !dbg !12
  %970 = select i1 %.not, float %968, float %958, !dbg !12
  %971 = select i1 %.not36, float %969, float %963, !dbg !12
  %972 = bitcast float %971 to i32, !dbg !12
  %973 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %972), !dbg !12
  %974 = bitcast i32 %973 to float, !dbg !12
  %975 = select i1 %.not36, float %970, float %964, !dbg !12
  %976 = bitcast float %975 to i32, !dbg !12
  %977 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %976), !dbg !12
  %978 = bitcast i32 %977 to float, !dbg !12
  %979 = insertelement <4 x float> poison, float %963, i64 0, !dbg !12
  %980 = insertelement <4 x float> %979, float %964, i64 1, !dbg !12
  %981 = insertelement <4 x float> %980, float %974, i64 2, !dbg !12
  %982 = insertelement <4 x float> %981, float %978, i64 3, !dbg !12
  %983 = insertelement <4 x float> poison, float %974, i64 0, !dbg !12
  %984 = insertelement <4 x float> %983, float %978, i64 1, !dbg !12
  %985 = insertelement <4 x float> %984, float %969, i64 2, !dbg !12
  %986 = insertelement <4 x float> %985, float %970, i64 3, !dbg !12
  %987 = select <4 x i1> %945, <4 x float> %982, <4 x float> %986, !dbg !12
  %988 = extractelement <4 x float> %885, i64 0, !dbg !12
  %989 = extractelement <4 x float> %885, i64 1, !dbg !12
  %990 = extractelement <4 x float> %885, i64 2, !dbg !12
  %991 = extractelement <4 x float> %885, i64 3, !dbg !12
  %992 = select i1 %.not, float %989, float %988, !dbg !12
  %993 = bitcast float %992 to i32, !dbg !12
  %994 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %993), !dbg !12
  %995 = bitcast i32 %994 to float, !dbg !12
  %996 = select i1 %.not, float %988, float %995, !dbg !12
  %997 = select i1 %.not, float %995, float %989, !dbg !12
  %998 = select i1 %.not, float %991, float %990, !dbg !12
  %999 = bitcast float %998 to i32, !dbg !12
  %1000 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %999), !dbg !12
  %1001 = bitcast i32 %1000 to float, !dbg !12
  %1002 = select i1 %.not, float %990, float %1001, !dbg !12
  %1003 = select i1 %.not, float %1001, float %991, !dbg !12
  %1004 = select i1 %.not36, float %1002, float %996, !dbg !12
  %1005 = bitcast float %1004 to i32, !dbg !12
  %1006 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1005), !dbg !12
  %1007 = bitcast i32 %1006 to float, !dbg !12
  %1008 = select i1 %.not36, float %1003, float %997, !dbg !12
  %1009 = bitcast float %1008 to i32, !dbg !12
  %1010 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1009), !dbg !12
  %1011 = bitcast i32 %1010 to float, !dbg !12
  %1012 = insertelement <4 x float> poison, float %996, i64 0, !dbg !12
  %1013 = insertelement <4 x float> %1012, float %997, i64 1, !dbg !12
  %1014 = insertelement <4 x float> %1013, float %1007, i64 2, !dbg !12
  %1015 = insertelement <4 x float> %1014, float %1011, i64 3, !dbg !12
  %1016 = insertelement <4 x float> poison, float %1007, i64 0, !dbg !12
  %1017 = insertelement <4 x float> %1016, float %1011, i64 1, !dbg !12
  %1018 = insertelement <4 x float> %1017, float %1002, i64 2, !dbg !12
  %1019 = insertelement <4 x float> %1018, float %1003, i64 3, !dbg !12
  %1020 = select <4 x i1> %945, <4 x float> %1015, <4 x float> %1019, !dbg !12
  %1021 = extractelement <4 x float> %886, i64 0, !dbg !12
  %1022 = extractelement <4 x float> %886, i64 1, !dbg !12
  %1023 = extractelement <4 x float> %886, i64 2, !dbg !12
  %1024 = extractelement <4 x float> %886, i64 3, !dbg !12
  %1025 = select i1 %.not, float %1022, float %1021, !dbg !12
  %1026 = bitcast float %1025 to i32, !dbg !12
  %1027 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1026), !dbg !12
  %1028 = bitcast i32 %1027 to float, !dbg !12
  %1029 = select i1 %.not, float %1021, float %1028, !dbg !12
  %1030 = select i1 %.not, float %1028, float %1022, !dbg !12
  %1031 = select i1 %.not, float %1024, float %1023, !dbg !12
  %1032 = bitcast float %1031 to i32, !dbg !12
  %1033 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1032), !dbg !12
  %1034 = bitcast i32 %1033 to float, !dbg !12
  %1035 = select i1 %.not, float %1023, float %1034, !dbg !12
  %1036 = select i1 %.not, float %1034, float %1024, !dbg !12
  %1037 = select i1 %.not36, float %1035, float %1029, !dbg !12
  %1038 = bitcast float %1037 to i32, !dbg !12
  %1039 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1038), !dbg !12
  %1040 = bitcast i32 %1039 to float, !dbg !12
  %1041 = select i1 %.not36, float %1036, float %1030, !dbg !12
  %1042 = bitcast float %1041 to i32, !dbg !12
  %1043 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1042), !dbg !12
  %1044 = bitcast i32 %1043 to float, !dbg !12
  %1045 = insertelement <4 x float> poison, float %1029, i64 0, !dbg !12
  %1046 = insertelement <4 x float> %1045, float %1030, i64 1, !dbg !12
  %1047 = insertelement <4 x float> %1046, float %1040, i64 2, !dbg !12
  %1048 = insertelement <4 x float> %1047, float %1044, i64 3, !dbg !12
  %1049 = insertelement <4 x float> poison, float %1040, i64 0, !dbg !12
  %1050 = insertelement <4 x float> %1049, float %1044, i64 1, !dbg !12
  %1051 = insertelement <4 x float> %1050, float %1035, i64 2, !dbg !12
  %1052 = insertelement <4 x float> %1051, float %1036, i64 3, !dbg !12
  %1053 = select <4 x i1> %945, <4 x float> %1048, <4 x float> %1052, !dbg !12
  %1054 = extractelement <4 x float> %811, i64 0, !dbg !12
  %1055 = extractelement <4 x float> %811, i64 1, !dbg !12
  %1056 = extractelement <4 x float> %811, i64 2, !dbg !12
  %1057 = extractelement <4 x float> %811, i64 3, !dbg !12
  %1058 = select i1 %.not, float %1055, float %1054, !dbg !12
  %1059 = bitcast float %1058 to i32, !dbg !12
  %1060 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1059), !dbg !12
  %1061 = bitcast i32 %1060 to float, !dbg !12
  %1062 = select i1 %.not, float %1054, float %1061, !dbg !12
  %1063 = select i1 %.not, float %1061, float %1055, !dbg !12
  %1064 = select i1 %.not, float %1057, float %1056, !dbg !12
  %1065 = bitcast float %1064 to i32, !dbg !12
  %1066 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1065), !dbg !12
  %1067 = bitcast i32 %1066 to float, !dbg !12
  %1068 = select i1 %.not, float %1056, float %1067, !dbg !12
  %1069 = select i1 %.not, float %1067, float %1057, !dbg !12
  %1070 = select i1 %.not36, float %1068, float %1062, !dbg !12
  %1071 = bitcast float %1070 to i32, !dbg !12
  %1072 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1071), !dbg !12
  %1073 = bitcast i32 %1072 to float, !dbg !12
  %1074 = select i1 %.not36, float %1069, float %1063, !dbg !12
  %1075 = bitcast float %1074 to i32, !dbg !12
  %1076 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1075), !dbg !12
  %1077 = bitcast i32 %1076 to float, !dbg !12
  %1078 = insertelement <4 x float> poison, float %1062, i64 0, !dbg !12
  %1079 = insertelement <4 x float> %1078, float %1063, i64 1, !dbg !12
  %1080 = insertelement <4 x float> %1079, float %1073, i64 2, !dbg !12
  %1081 = insertelement <4 x float> %1080, float %1077, i64 3, !dbg !12
  %1082 = insertelement <4 x float> poison, float %1073, i64 0, !dbg !12
  %1083 = insertelement <4 x float> %1082, float %1077, i64 1, !dbg !12
  %1084 = insertelement <4 x float> %1083, float %1068, i64 2, !dbg !12
  %1085 = insertelement <4 x float> %1084, float %1069, i64 3, !dbg !12
  %1086 = select <4 x i1> %945, <4 x float> %1081, <4 x float> %1085, !dbg !12
  %1087 = extractelement <4 x float> %813, i64 0, !dbg !12
  %1088 = extractelement <4 x float> %813, i64 1, !dbg !12
  %1089 = extractelement <4 x float> %813, i64 2, !dbg !12
  %1090 = extractelement <4 x float> %813, i64 3, !dbg !12
  %1091 = select i1 %.not, float %1088, float %1087, !dbg !12
  %1092 = bitcast float %1091 to i32, !dbg !12
  %1093 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1092), !dbg !12
  %1094 = bitcast i32 %1093 to float, !dbg !12
  %1095 = select i1 %.not, float %1087, float %1094, !dbg !12
  %1096 = select i1 %.not, float %1094, float %1088, !dbg !12
  %1097 = select i1 %.not, float %1090, float %1089, !dbg !12
  %1098 = bitcast float %1097 to i32, !dbg !12
  %1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1098), !dbg !12
  %1100 = bitcast i32 %1099 to float, !dbg !12
  %1101 = select i1 %.not, float %1089, float %1100, !dbg !12
  %1102 = select i1 %.not, float %1100, float %1090, !dbg !12
  %1103 = select i1 %.not36, float %1101, float %1095, !dbg !12
  %1104 = bitcast float %1103 to i32, !dbg !12
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1104), !dbg !12
  %1106 = bitcast i32 %1105 to float, !dbg !12
  %1107 = select i1 %.not36, float %1102, float %1096, !dbg !12
  %1108 = bitcast float %1107 to i32, !dbg !12
  %1109 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1108), !dbg !12
  %1110 = bitcast i32 %1109 to float, !dbg !12
  %1111 = insertelement <4 x float> poison, float %1095, i64 0, !dbg !12
  %1112 = insertelement <4 x float> %1111, float %1096, i64 1, !dbg !12
  %1113 = insertelement <4 x float> %1112, float %1106, i64 2, !dbg !12
  %1114 = insertelement <4 x float> %1113, float %1110, i64 3, !dbg !12
  %1115 = insertelement <4 x float> poison, float %1106, i64 0, !dbg !12
  %1116 = insertelement <4 x float> %1115, float %1110, i64 1, !dbg !12
  %1117 = insertelement <4 x float> %1116, float %1101, i64 2, !dbg !12
  %1118 = insertelement <4 x float> %1117, float %1102, i64 3, !dbg !12
  %1119 = select <4 x i1> %945, <4 x float> %1114, <4 x float> %1118, !dbg !12
  %1120 = extractelement <4 x float> %814, i64 0, !dbg !12
  %1121 = extractelement <4 x float> %814, i64 1, !dbg !12
  %1122 = extractelement <4 x float> %814, i64 2, !dbg !12
  %1123 = extractelement <4 x float> %814, i64 3, !dbg !12
  %1124 = select i1 %.not, float %1121, float %1120, !dbg !12
  %1125 = bitcast float %1124 to i32, !dbg !12
  %1126 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1125), !dbg !12
  %1127 = bitcast i32 %1126 to float, !dbg !12
  %1128 = select i1 %.not, float %1120, float %1127, !dbg !12
  %1129 = select i1 %.not, float %1127, float %1121, !dbg !12
  %1130 = select i1 %.not, float %1123, float %1122, !dbg !12
  %1131 = bitcast float %1130 to i32, !dbg !12
  %1132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1131), !dbg !12
  %1133 = bitcast i32 %1132 to float, !dbg !12
  %1134 = select i1 %.not, float %1122, float %1133, !dbg !12
  %1135 = select i1 %.not, float %1133, float %1123, !dbg !12
  %1136 = select i1 %.not36, float %1134, float %1128, !dbg !12
  %1137 = bitcast float %1136 to i32, !dbg !12
  %1138 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1137), !dbg !12
  %1139 = bitcast i32 %1138 to float, !dbg !12
  %1140 = select i1 %.not36, float %1135, float %1129, !dbg !12
  %1141 = bitcast float %1140 to i32, !dbg !12
  %1142 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1141), !dbg !12
  %1143 = bitcast i32 %1142 to float, !dbg !12
  %1144 = insertelement <4 x float> poison, float %1128, i64 0, !dbg !12
  %1145 = insertelement <4 x float> %1144, float %1129, i64 1, !dbg !12
  %1146 = insertelement <4 x float> %1145, float %1139, i64 2, !dbg !12
  %1147 = insertelement <4 x float> %1146, float %1143, i64 3, !dbg !12
  %1148 = insertelement <4 x float> poison, float %1139, i64 0, !dbg !12
  %1149 = insertelement <4 x float> %1148, float %1143, i64 1, !dbg !12
  %1150 = insertelement <4 x float> %1149, float %1134, i64 2, !dbg !12
  %1151 = insertelement <4 x float> %1150, float %1135, i64 3, !dbg !12
  %1152 = select <4 x i1> %945, <4 x float> %1147, <4 x float> %1151, !dbg !12
  %1153 = extractelement <4 x float> %815, i64 0, !dbg !12
  %1154 = extractelement <4 x float> %815, i64 1, !dbg !12
  %1155 = extractelement <4 x float> %815, i64 2, !dbg !12
  %1156 = extractelement <4 x float> %815, i64 3, !dbg !12
  %1157 = select i1 %.not, float %1154, float %1153, !dbg !12
  %1158 = bitcast float %1157 to i32, !dbg !12
  %1159 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1158), !dbg !12
  %1160 = bitcast i32 %1159 to float, !dbg !12
  %1161 = select i1 %.not, float %1153, float %1160, !dbg !12
  %1162 = select i1 %.not, float %1160, float %1154, !dbg !12
  %1163 = select i1 %.not, float %1156, float %1155, !dbg !12
  %1164 = bitcast float %1163 to i32, !dbg !12
  %1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1164), !dbg !12
  %1166 = bitcast i32 %1165 to float, !dbg !12
  %1167 = select i1 %.not, float %1155, float %1166, !dbg !12
  %1168 = select i1 %.not, float %1166, float %1156, !dbg !12
  %1169 = select i1 %.not36, float %1167, float %1161, !dbg !12
  %1170 = bitcast float %1169 to i32, !dbg !12
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1170), !dbg !12
  %1172 = bitcast i32 %1171 to float, !dbg !12
  %1173 = select i1 %.not36, float %1168, float %1162, !dbg !12
  %1174 = bitcast float %1173 to i32, !dbg !12
  %1175 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1174), !dbg !12
  %1176 = bitcast i32 %1175 to float, !dbg !12
  %1177 = insertelement <4 x float> poison, float %1161, i64 0, !dbg !12
  %1178 = insertelement <4 x float> %1177, float %1162, i64 1, !dbg !12
  %1179 = insertelement <4 x float> %1178, float %1172, i64 2, !dbg !12
  %1180 = insertelement <4 x float> %1179, float %1176, i64 3, !dbg !12
  %1181 = insertelement <4 x float> poison, float %1172, i64 0, !dbg !12
  %1182 = insertelement <4 x float> %1181, float %1176, i64 1, !dbg !12
  %1183 = insertelement <4 x float> %1182, float %1167, i64 2, !dbg !12
  %1184 = insertelement <4 x float> %1183, float %1168, i64 3, !dbg !12
  %1185 = select <4 x i1> %945, <4 x float> %1180, <4 x float> %1184, !dbg !12
  %1186 = extractelement <4 x float> %739, i64 0, !dbg !12
  %1187 = extractelement <4 x float> %739, i64 1, !dbg !12
  %1188 = extractelement <4 x float> %739, i64 2, !dbg !12
  %1189 = extractelement <4 x float> %739, i64 3, !dbg !12
  %1190 = select i1 %.not, float %1187, float %1186, !dbg !12
  %1191 = bitcast float %1190 to i32, !dbg !12
  %1192 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1191), !dbg !12
  %1193 = bitcast i32 %1192 to float, !dbg !12
  %1194 = select i1 %.not, float %1186, float %1193, !dbg !12
  %1195 = select i1 %.not, float %1193, float %1187, !dbg !12
  %1196 = select i1 %.not, float %1189, float %1188, !dbg !12
  %1197 = bitcast float %1196 to i32, !dbg !12
  %1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1197), !dbg !12
  %1199 = bitcast i32 %1198 to float, !dbg !12
  %1200 = select i1 %.not, float %1188, float %1199, !dbg !12
  %1201 = select i1 %.not, float %1199, float %1189, !dbg !12
  %1202 = select i1 %.not36, float %1200, float %1194, !dbg !12
  %1203 = bitcast float %1202 to i32, !dbg !12
  %1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1203), !dbg !12
  %1205 = bitcast i32 %1204 to float, !dbg !12
  %1206 = select i1 %.not36, float %1201, float %1195, !dbg !12
  %1207 = bitcast float %1206 to i32, !dbg !12
  %1208 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1207), !dbg !12
  %1209 = bitcast i32 %1208 to float, !dbg !12
  %1210 = insertelement <4 x float> poison, float %1194, i64 0, !dbg !12
  %1211 = insertelement <4 x float> %1210, float %1195, i64 1, !dbg !12
  %1212 = insertelement <4 x float> %1211, float %1205, i64 2, !dbg !12
  %1213 = insertelement <4 x float> %1212, float %1209, i64 3, !dbg !12
  %1214 = insertelement <4 x float> poison, float %1205, i64 0, !dbg !12
  %1215 = insertelement <4 x float> %1214, float %1209, i64 1, !dbg !12
  %1216 = insertelement <4 x float> %1215, float %1200, i64 2, !dbg !12
  %1217 = insertelement <4 x float> %1216, float %1201, i64 3, !dbg !12
  %1218 = select <4 x i1> %945, <4 x float> %1213, <4 x float> %1217, !dbg !12
  %1219 = extractelement <4 x float> %741, i64 0, !dbg !12
  %1220 = extractelement <4 x float> %741, i64 1, !dbg !12
  %1221 = extractelement <4 x float> %741, i64 2, !dbg !12
  %1222 = extractelement <4 x float> %741, i64 3, !dbg !12
  %1223 = select i1 %.not, float %1220, float %1219, !dbg !12
  %1224 = bitcast float %1223 to i32, !dbg !12
  %1225 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1224), !dbg !12
  %1226 = bitcast i32 %1225 to float, !dbg !12
  %1227 = select i1 %.not, float %1219, float %1226, !dbg !12
  %1228 = select i1 %.not, float %1226, float %1220, !dbg !12
  %1229 = select i1 %.not, float %1222, float %1221, !dbg !12
  %1230 = bitcast float %1229 to i32, !dbg !12
  %1231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1230), !dbg !12
  %1232 = bitcast i32 %1231 to float, !dbg !12
  %1233 = select i1 %.not, float %1221, float %1232, !dbg !12
  %1234 = select i1 %.not, float %1232, float %1222, !dbg !12
  %1235 = select i1 %.not36, float %1233, float %1227, !dbg !12
  %1236 = bitcast float %1235 to i32, !dbg !12
  %1237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1236), !dbg !12
  %1238 = bitcast i32 %1237 to float, !dbg !12
  %1239 = select i1 %.not36, float %1234, float %1228, !dbg !12
  %1240 = bitcast float %1239 to i32, !dbg !12
  %1241 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1240), !dbg !12
  %1242 = bitcast i32 %1241 to float, !dbg !12
  %1243 = insertelement <4 x float> poison, float %1227, i64 0, !dbg !12
  %1244 = insertelement <4 x float> %1243, float %1228, i64 1, !dbg !12
  %1245 = insertelement <4 x float> %1244, float %1238, i64 2, !dbg !12
  %1246 = insertelement <4 x float> %1245, float %1242, i64 3, !dbg !12
  %1247 = insertelement <4 x float> poison, float %1238, i64 0, !dbg !12
  %1248 = insertelement <4 x float> %1247, float %1242, i64 1, !dbg !12
  %1249 = insertelement <4 x float> %1248, float %1233, i64 2, !dbg !12
  %1250 = insertelement <4 x float> %1249, float %1234, i64 3, !dbg !12
  %1251 = select <4 x i1> %945, <4 x float> %1246, <4 x float> %1250, !dbg !12
  %1252 = extractelement <4 x float> %742, i64 0, !dbg !12
  %1253 = extractelement <4 x float> %742, i64 1, !dbg !12
  %1254 = extractelement <4 x float> %742, i64 2, !dbg !12
  %1255 = extractelement <4 x float> %742, i64 3, !dbg !12
  %1256 = select i1 %.not, float %1253, float %1252, !dbg !12
  %1257 = bitcast float %1256 to i32, !dbg !12
  %1258 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1257), !dbg !12
  %1259 = bitcast i32 %1258 to float, !dbg !12
  %1260 = select i1 %.not, float %1252, float %1259, !dbg !12
  %1261 = select i1 %.not, float %1259, float %1253, !dbg !12
  %1262 = select i1 %.not, float %1255, float %1254, !dbg !12
  %1263 = bitcast float %1262 to i32, !dbg !12
  %1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1263), !dbg !12
  %1265 = bitcast i32 %1264 to float, !dbg !12
  %1266 = select i1 %.not, float %1254, float %1265, !dbg !12
  %1267 = select i1 %.not, float %1265, float %1255, !dbg !12
  %1268 = select i1 %.not36, float %1266, float %1260, !dbg !12
  %1269 = bitcast float %1268 to i32, !dbg !12
  %1270 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1269), !dbg !12
  %1271 = bitcast i32 %1270 to float, !dbg !12
  %1272 = select i1 %.not36, float %1267, float %1261, !dbg !12
  %1273 = bitcast float %1272 to i32, !dbg !12
  %1274 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1273), !dbg !12
  %1275 = bitcast i32 %1274 to float, !dbg !12
  %1276 = insertelement <4 x float> poison, float %1260, i64 0, !dbg !12
  %1277 = insertelement <4 x float> %1276, float %1261, i64 1, !dbg !12
  %1278 = insertelement <4 x float> %1277, float %1271, i64 2, !dbg !12
  %1279 = insertelement <4 x float> %1278, float %1275, i64 3, !dbg !12
  %1280 = insertelement <4 x float> poison, float %1271, i64 0, !dbg !12
  %1281 = insertelement <4 x float> %1280, float %1275, i64 1, !dbg !12
  %1282 = insertelement <4 x float> %1281, float %1266, i64 2, !dbg !12
  %1283 = insertelement <4 x float> %1282, float %1267, i64 3, !dbg !12
  %1284 = select <4 x i1> %945, <4 x float> %1279, <4 x float> %1283, !dbg !12
  %1285 = extractelement <4 x float> %743, i64 0, !dbg !12
  %1286 = extractelement <4 x float> %743, i64 1, !dbg !12
  %1287 = extractelement <4 x float> %743, i64 2, !dbg !12
  %1288 = extractelement <4 x float> %743, i64 3, !dbg !12
  %1289 = select i1 %.not, float %1286, float %1285, !dbg !12
  %1290 = bitcast float %1289 to i32, !dbg !12
  %1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1290), !dbg !12
  %1292 = bitcast i32 %1291 to float, !dbg !12
  %1293 = select i1 %.not, float %1285, float %1292, !dbg !12
  %1294 = select i1 %.not, float %1292, float %1286, !dbg !12
  %1295 = select i1 %.not, float %1288, float %1287, !dbg !12
  %1296 = bitcast float %1295 to i32, !dbg !12
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1296), !dbg !12
  %1298 = bitcast i32 %1297 to float, !dbg !12
  %1299 = select i1 %.not, float %1287, float %1298, !dbg !12
  %1300 = select i1 %.not, float %1298, float %1288, !dbg !12
  %1301 = select i1 %.not36, float %1299, float %1293, !dbg !12
  %1302 = bitcast float %1301 to i32, !dbg !12
  %1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1302), !dbg !12
  %1304 = bitcast i32 %1303 to float, !dbg !12
  %1305 = select i1 %.not36, float %1300, float %1294, !dbg !12
  %1306 = bitcast float %1305 to i32, !dbg !12
  %1307 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1306), !dbg !12
  %1308 = bitcast i32 %1307 to float, !dbg !12
  %1309 = insertelement <4 x float> poison, float %1293, i64 0, !dbg !12
  %1310 = insertelement <4 x float> %1309, float %1294, i64 1, !dbg !12
  %1311 = insertelement <4 x float> %1310, float %1304, i64 2, !dbg !12
  %1312 = insertelement <4 x float> %1311, float %1308, i64 3, !dbg !12
  %1313 = insertelement <4 x float> poison, float %1304, i64 0, !dbg !12
  %1314 = insertelement <4 x float> %1313, float %1308, i64 1, !dbg !12
  %1315 = insertelement <4 x float> %1314, float %1299, i64 2, !dbg !12
  %1316 = insertelement <4 x float> %1315, float %1300, i64 3, !dbg !12
  %1317 = select <4 x i1> %945, <4 x float> %1312, <4 x float> %1316, !dbg !12
  %1318 = extractelement <4 x float> %654, i64 0, !dbg !12
  %1319 = extractelement <4 x float> %654, i64 1, !dbg !12
  %1320 = extractelement <4 x float> %654, i64 2, !dbg !12
  %1321 = extractelement <4 x float> %654, i64 3, !dbg !12
  %1322 = select i1 %.not, float %1319, float %1318, !dbg !12
  %1323 = bitcast float %1322 to i32, !dbg !12
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1323), !dbg !12
  %1325 = bitcast i32 %1324 to float, !dbg !12
  %1326 = select i1 %.not, float %1318, float %1325, !dbg !12
  %1327 = select i1 %.not, float %1325, float %1319, !dbg !12
  %1328 = select i1 %.not, float %1321, float %1320, !dbg !12
  %1329 = bitcast float %1328 to i32, !dbg !12
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1329), !dbg !12
  %1331 = bitcast i32 %1330 to float, !dbg !12
  %1332 = select i1 %.not, float %1320, float %1331, !dbg !12
  %1333 = select i1 %.not, float %1331, float %1321, !dbg !12
  %1334 = select i1 %.not36, float %1332, float %1326, !dbg !12
  %1335 = bitcast float %1334 to i32, !dbg !12
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1335), !dbg !12
  %1337 = select i1 %.not36, float %1333, float %1327, !dbg !12
  %1338 = bitcast float %1337 to i32, !dbg !12
  %1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1338), !dbg !12
  %1340 = extractelement <4 x float> %656, i64 0, !dbg !12
  %1341 = extractelement <4 x float> %656, i64 1, !dbg !12
  %1342 = extractelement <4 x float> %656, i64 2, !dbg !12
  %1343 = extractelement <4 x float> %656, i64 3, !dbg !12
  %1344 = select i1 %.not, float %1341, float %1340, !dbg !12
  %1345 = bitcast float %1344 to i32, !dbg !12
  %1346 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1345), !dbg !12
  %1347 = bitcast i32 %1346 to float, !dbg !12
  %1348 = select i1 %.not, float %1340, float %1347, !dbg !12
  %1349 = select i1 %.not, float %1347, float %1341, !dbg !12
  %1350 = select i1 %.not, float %1343, float %1342, !dbg !12
  %1351 = bitcast float %1350 to i32, !dbg !12
  %1352 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1351), !dbg !12
  %1353 = bitcast i32 %1352 to float, !dbg !12
  %1354 = select i1 %.not, float %1342, float %1353, !dbg !12
  %1355 = select i1 %.not, float %1353, float %1343, !dbg !12
  %1356 = select i1 %.not36, float %1354, float %1348, !dbg !12
  %1357 = bitcast float %1356 to i32, !dbg !12
  %1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1357), !dbg !12
  %1359 = select i1 %.not36, float %1355, float %1349, !dbg !12
  %1360 = bitcast float %1359 to i32, !dbg !12
  %1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1360), !dbg !12
  %1362 = extractelement <4 x float> %670, i64 0, !dbg !12
  %1363 = extractelement <4 x float> %670, i64 1, !dbg !12
  %1364 = extractelement <4 x float> %670, i64 2, !dbg !12
  %1365 = extractelement <4 x float> %670, i64 3, !dbg !12
  %1366 = select i1 %.not, float %1363, float %1362, !dbg !12
  %1367 = bitcast float %1366 to i32, !dbg !12
  %1368 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1367), !dbg !12
  %1369 = bitcast i32 %1368 to float, !dbg !12
  %1370 = select i1 %.not, float %1362, float %1369, !dbg !12
  %1371 = select i1 %.not, float %1369, float %1363, !dbg !12
  %1372 = select i1 %.not, float %1365, float %1364, !dbg !12
  %1373 = bitcast float %1372 to i32, !dbg !12
  %1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1373), !dbg !12
  %1375 = bitcast i32 %1374 to float, !dbg !12
  %1376 = select i1 %.not, float %1364, float %1375, !dbg !12
  %1377 = select i1 %.not, float %1375, float %1365, !dbg !12
  %1378 = select i1 %.not36, float %1376, float %1370, !dbg !12
  %1379 = bitcast float %1378 to i32, !dbg !12
  %1380 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1379), !dbg !12
  %1381 = select i1 %.not36, float %1377, float %1371, !dbg !12
  %1382 = bitcast float %1381 to i32, !dbg !12
  %1383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1382), !dbg !12
  %1384 = extractelement <4 x float> %671, i64 0, !dbg !12
  %1385 = extractelement <4 x float> %671, i64 1, !dbg !12
  %1386 = extractelement <4 x float> %671, i64 2, !dbg !12
  %1387 = extractelement <4 x float> %671, i64 3, !dbg !12
  %1388 = select i1 %.not, float %1385, float %1384, !dbg !12
  %1389 = bitcast float %1388 to i32, !dbg !12
  %1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1389), !dbg !12
  %1391 = bitcast i32 %1390 to float, !dbg !12
  %1392 = select i1 %.not, float %1384, float %1391, !dbg !12
  %1393 = select i1 %.not, float %1391, float %1385, !dbg !12
  %1394 = select i1 %.not, float %1387, float %1386, !dbg !12
  %1395 = bitcast float %1394 to i32, !dbg !12
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1395), !dbg !12
  %1397 = bitcast i32 %1396 to float, !dbg !12
  %1398 = select i1 %.not, float %1386, float %1397, !dbg !12
  %1399 = select i1 %.not, float %1397, float %1387, !dbg !12
  %1400 = select i1 %.not36, float %1398, float %1392, !dbg !12
  %1401 = bitcast float %1400 to i32, !dbg !12
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1401), !dbg !12
  %1403 = select i1 %.not36, float %1399, float %1393, !dbg !12
  %1404 = bitcast float %1403 to i32, !dbg !12
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1404), !dbg !12
  %1406 = shufflevector <1 x float> %917, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %1407 = shufflevector <1 x float> %912, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !47
  %1408 = bitcast i32 %1402 to float, !dbg !12
  %1409 = bitcast i32 %1405 to float, !dbg !12
  %1410 = insertelement <4 x float> poison, float %1392, i64 0, !dbg !12
  %1411 = insertelement <4 x float> %1410, float %1393, i64 1, !dbg !12
  %1412 = insertelement <4 x float> %1411, float %1408, i64 2, !dbg !12
  %1413 = insertelement <4 x float> %1412, float %1409, i64 3, !dbg !12
  %1414 = insertelement <4 x float> poison, float %1408, i64 0, !dbg !12
  %1415 = insertelement <4 x float> %1414, float %1409, i64 1, !dbg !12
  %1416 = insertelement <4 x float> %1415, float %1398, i64 2, !dbg !12
  %1417 = insertelement <4 x float> %1416, float %1399, i64 3, !dbg !12
  %1418 = select <4 x i1> %945, <4 x float> %1413, <4 x float> %1417, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1407, !dbg !47
  %normalized = fmul <4 x float> %row.reciprocal, %1418, !dbg !47
  %1419 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !48
  %1420 = extractelement <4 x half> %1419, i64 3, !dbg !48
  %1421 = extractelement <4 x half> %1419, i64 2, !dbg !48
  %1422 = extractelement <4 x half> %1419, i64 1, !dbg !48
  %1423 = extractelement <4 x half> %1419, i64 0, !dbg !48
  %1424 = bitcast i32 %1380 to float, !dbg !12
  %1425 = bitcast i32 %1383 to float, !dbg !12
  %1426 = insertelement <4 x float> poison, float %1370, i64 0, !dbg !12
  %1427 = insertelement <4 x float> %1426, float %1371, i64 1, !dbg !12
  %1428 = insertelement <4 x float> %1427, float %1424, i64 2, !dbg !12
  %1429 = insertelement <4 x float> %1428, float %1425, i64 3, !dbg !12
  %1430 = insertelement <4 x float> poison, float %1424, i64 0, !dbg !12
  %1431 = insertelement <4 x float> %1430, float %1425, i64 1, !dbg !12
  %1432 = insertelement <4 x float> %1431, float %1376, i64 2, !dbg !12
  %1433 = insertelement <4 x float> %1432, float %1377, i64 3, !dbg !12
  %1434 = select <4 x i1> %945, <4 x float> %1429, <4 x float> %1433, !dbg !12
  %normalized37 = fmul <4 x float> %row.reciprocal, %1434, !dbg !47
  %1435 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !48
  %1436 = extractelement <4 x half> %1435, i64 3, !dbg !48
  %1437 = extractelement <4 x half> %1435, i64 2, !dbg !48
  %1438 = extractelement <4 x half> %1435, i64 1, !dbg !48
  %1439 = extractelement <4 x half> %1435, i64 0, !dbg !48
  %1440 = bitcast i32 %1358 to float, !dbg !12
  %1441 = bitcast i32 %1361 to float, !dbg !12
  %1442 = insertelement <4 x float> poison, float %1348, i64 0, !dbg !12
  %1443 = insertelement <4 x float> %1442, float %1349, i64 1, !dbg !12
  %1444 = insertelement <4 x float> %1443, float %1440, i64 2, !dbg !12
  %1445 = insertelement <4 x float> %1444, float %1441, i64 3, !dbg !12
  %1446 = insertelement <4 x float> poison, float %1440, i64 0, !dbg !12
  %1447 = insertelement <4 x float> %1446, float %1441, i64 1, !dbg !12
  %1448 = insertelement <4 x float> %1447, float %1354, i64 2, !dbg !12
  %1449 = insertelement <4 x float> %1448, float %1355, i64 3, !dbg !12
  %1450 = select <4 x i1> %945, <4 x float> %1445, <4 x float> %1449, !dbg !12
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1406, !dbg !47
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1450, !dbg !47
  %1451 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !48
  %1452 = extractelement <4 x half> %1451, i64 3, !dbg !48
  %1453 = extractelement <4 x half> %1451, i64 2, !dbg !48
  %1454 = extractelement <4 x half> %1451, i64 1, !dbg !48
  %1455 = extractelement <4 x half> %1451, i64 0, !dbg !48
  %1456 = bitcast i32 %1336 to float, !dbg !12
  %1457 = bitcast i32 %1339 to float, !dbg !12
  %1458 = insertelement <4 x float> poison, float %1326, i64 0, !dbg !12
  %1459 = insertelement <4 x float> %1458, float %1327, i64 1, !dbg !12
  %1460 = insertelement <4 x float> %1459, float %1456, i64 2, !dbg !12
  %1461 = insertelement <4 x float> %1460, float %1457, i64 3, !dbg !12
  %1462 = insertelement <4 x float> poison, float %1456, i64 0, !dbg !12
  %1463 = insertelement <4 x float> %1462, float %1457, i64 1, !dbg !12
  %1464 = insertelement <4 x float> %1463, float %1332, i64 2, !dbg !12
  %1465 = insertelement <4 x float> %1464, float %1333, i64 3, !dbg !12
  %1466 = select <4 x i1> %945, <4 x float> %1461, <4 x float> %1465, !dbg !12
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1466, !dbg !47
  %1467 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !48
  %1468 = extractelement <4 x half> %1467, i64 3, !dbg !48
  %1469 = extractelement <4 x half> %1467, i64 2, !dbg !48
  %1470 = extractelement <4 x half> %1467, i64 1, !dbg !48
  %1471 = extractelement <4 x half> %1467, i64 0, !dbg !48
  %address.thread958 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !49, !range !10, !frisk.register_lifetimes !5
  %address.local959 = zext nneg i32 %address.thread958 to i64, !dbg !9
  %1472 = lshr i64 %address.local959, 2, !dbg !49
  %address.local962 = and i64 %1472, 48, !dbg !49
  %address.local964 = and i64 %address.local959, 240, !dbg !49
  %address.local963 = add i64 %34, %address.local959, !dbg !49
  %address.local965 = sub i64 %address.local963, %address.local964, !dbg !49
  %address.local966 = add i64 %address.local965, %address.local962, !dbg !49
  %1473 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %42, !dbg !49
  %address.local969.idx = shl nuw nsw i64 %address.local966, 8, !dbg !49
  %address.local969 = getelementptr inbounds nuw i8, ptr addrspace(1) %1473, i64 %address.local969.idx, !dbg !49
  %address.local971 = and i64 %1472, 12, !dbg !11
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local971, !dbg !49
  store half %1471, ptr addrspace(1) %address.local972, align 2, !dbg !49
  %address.local974 = or disjoint i64 %address.local971, 1, !dbg !49
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local974, !dbg !49
  store half %1470, ptr addrspace(1) %address.local975, align 2, !dbg !49
  %address.local977 = or disjoint i64 %address.local971, 2, !dbg !49
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local977, !dbg !49
  store half %1469, ptr addrspace(1) %address.local978, align 2, !dbg !49
  %address.local980 = or disjoint i64 %address.local971, 3, !dbg !49
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local980, !dbg !49
  store half %1468, ptr addrspace(1) %address.local981, align 2, !dbg !49
  %address.local983 = or disjoint i64 %address.local971, 16, !dbg !49
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local983, !dbg !49
  store half %1455, ptr addrspace(1) %address.local984, align 2, !dbg !49
  %address.local986 = or disjoint i64 %address.local971, 17, !dbg !49
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local986, !dbg !49
  store half %1454, ptr addrspace(1) %address.local987, align 2, !dbg !49
  %address.local989 = or disjoint i64 %address.local971, 18, !dbg !49
  %address.local990 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local989, !dbg !49
  store half %1453, ptr addrspace(1) %address.local990, align 2, !dbg !49
  %address.local992 = or disjoint i64 %address.local971, 19, !dbg !49
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local992, !dbg !49
  store half %1452, ptr addrspace(1) %address.local993, align 2, !dbg !49
  %address.local997 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local969, i64 16384, !dbg !49
  %address.local998 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local971, !dbg !49
  store half %1439, ptr addrspace(1) %address.local998, align 2, !dbg !49
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local974, !dbg !49
  store half %1438, ptr addrspace(1) %address.local1000, align 2, !dbg !49
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local977, !dbg !49
  store half %1437, ptr addrspace(1) %address.local1002, align 2, !dbg !49
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local980, !dbg !49
  store half %1436, ptr addrspace(1) %address.local1004, align 2, !dbg !49
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local983, !dbg !49
  store half %1423, ptr addrspace(1) %address.local1006, align 2, !dbg !49
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local986, !dbg !49
  store half %1422, ptr addrspace(1) %address.local1008, align 2, !dbg !49
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local989, !dbg !49
  store half %1421, ptr addrspace(1) %address.local1010, align 2, !dbg !49
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local992, !dbg !49
  store half %1420, ptr addrspace(1) %address.local1012, align 2, !dbg !49
  %normalized42 = fmul <4 x float> %row.reciprocal, %1317, !dbg !47
  %1474 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !48
  %1475 = extractelement <4 x half> %1474, i64 3, !dbg !48
  %1476 = extractelement <4 x half> %1474, i64 2, !dbg !48
  %1477 = extractelement <4 x half> %1474, i64 1, !dbg !48
  %1478 = extractelement <4 x half> %1474, i64 0, !dbg !48
  %normalized43 = fmul <4 x float> %row.reciprocal, %1284, !dbg !47
  %1479 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !48
  %1480 = extractelement <4 x half> %1479, i64 3, !dbg !48
  %1481 = extractelement <4 x half> %1479, i64 2, !dbg !48
  %1482 = extractelement <4 x half> %1479, i64 1, !dbg !48
  %1483 = extractelement <4 x half> %1479, i64 0, !dbg !48
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1251, !dbg !47
  %1484 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !48
  %1485 = extractelement <4 x half> %1484, i64 3, !dbg !48
  %1486 = extractelement <4 x half> %1484, i64 2, !dbg !48
  %1487 = extractelement <4 x half> %1484, i64 1, !dbg !48
  %1488 = extractelement <4 x half> %1484, i64 0, !dbg !48
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1218, !dbg !47
  %1489 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !48
  %1490 = extractelement <4 x half> %1489, i64 3, !dbg !48
  %1491 = extractelement <4 x half> %1489, i64 2, !dbg !48
  %1492 = extractelement <4 x half> %1489, i64 1, !dbg !48
  %1493 = extractelement <4 x half> %1489, i64 0, !dbg !48
  %address.local1027 = or disjoint i64 %address.local971, 32, !dbg !49
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1027, !dbg !49
  store half %1493, ptr addrspace(1) %address.local1028, align 2, !dbg !49
  %address.local1030 = or disjoint i64 %address.local971, 33, !dbg !49
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1030, !dbg !49
  store half %1492, ptr addrspace(1) %address.local1031, align 2, !dbg !49
  %address.local1033 = or disjoint i64 %address.local971, 34, !dbg !49
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1033, !dbg !49
  store half %1491, ptr addrspace(1) %address.local1034, align 2, !dbg !49
  %address.local1036 = or disjoint i64 %address.local971, 35, !dbg !49
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1036, !dbg !49
  store half %1490, ptr addrspace(1) %address.local1037, align 2, !dbg !49
  %address.local1039 = or disjoint i64 %address.local971, 48, !dbg !49
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1039, !dbg !49
  store half %1488, ptr addrspace(1) %address.local1040, align 2, !dbg !49
  %address.local1042 = or disjoint i64 %address.local971, 49, !dbg !49
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1042, !dbg !49
  store half %1487, ptr addrspace(1) %address.local1043, align 2, !dbg !49
  %address.local1045 = or disjoint i64 %address.local971, 50, !dbg !49
  %address.local1046 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1045, !dbg !49
  store half %1486, ptr addrspace(1) %address.local1046, align 2, !dbg !49
  %address.local1048 = or i64 %1472, 51, !dbg !49
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1048, !dbg !49
  store half %1485, ptr addrspace(1) %address.local1049, align 2, !dbg !49
  %address.local1054 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1027, !dbg !49
  store half %1483, ptr addrspace(1) %address.local1054, align 2, !dbg !49
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1030, !dbg !49
  store half %1482, ptr addrspace(1) %address.local1056, align 2, !dbg !49
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1033, !dbg !49
  store half %1481, ptr addrspace(1) %address.local1058, align 2, !dbg !49
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1036, !dbg !49
  store half %1480, ptr addrspace(1) %address.local1060, align 2, !dbg !49
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1039, !dbg !49
  store half %1478, ptr addrspace(1) %address.local1062, align 2, !dbg !49
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1042, !dbg !49
  store half %1477, ptr addrspace(1) %address.local1064, align 2, !dbg !49
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1045, !dbg !49
  store half %1476, ptr addrspace(1) %address.local1066, align 2, !dbg !49
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1048, !dbg !49
  store half %1475, ptr addrspace(1) %address.local1068, align 2, !dbg !49
  %normalized48 = fmul <4 x float> %row.reciprocal, %1185, !dbg !47
  %1494 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !48
  %1495 = extractelement <4 x half> %1494, i64 3, !dbg !48
  %1496 = extractelement <4 x half> %1494, i64 2, !dbg !48
  %1497 = extractelement <4 x half> %1494, i64 1, !dbg !48
  %1498 = extractelement <4 x half> %1494, i64 0, !dbg !48
  %normalized49 = fmul <4 x float> %row.reciprocal, %1152, !dbg !47
  %1499 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !48
  %1500 = extractelement <4 x half> %1499, i64 3, !dbg !48
  %1501 = extractelement <4 x half> %1499, i64 2, !dbg !48
  %1502 = extractelement <4 x half> %1499, i64 1, !dbg !48
  %1503 = extractelement <4 x half> %1499, i64 0, !dbg !48
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1119, !dbg !47
  %1504 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !48
  %1505 = extractelement <4 x half> %1504, i64 3, !dbg !48
  %1506 = extractelement <4 x half> %1504, i64 2, !dbg !48
  %1507 = extractelement <4 x half> %1504, i64 1, !dbg !48
  %1508 = extractelement <4 x half> %1504, i64 0, !dbg !48
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1086, !dbg !47
  %1509 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !48
  %1510 = extractelement <4 x half> %1509, i64 3, !dbg !48
  %1511 = extractelement <4 x half> %1509, i64 2, !dbg !48
  %1512 = extractelement <4 x half> %1509, i64 1, !dbg !48
  %1513 = extractelement <4 x half> %1509, i64 0, !dbg !48
  %address.local1083 = or disjoint i64 %address.local971, 64, !dbg !49
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1083, !dbg !49
  store half %1513, ptr addrspace(1) %address.local1084, align 2, !dbg !49
  %address.local1086 = or disjoint i64 %address.local971, 65, !dbg !49
  %address.local1087 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1086, !dbg !49
  store half %1512, ptr addrspace(1) %address.local1087, align 2, !dbg !49
  %address.local1089 = or disjoint i64 %address.local971, 66, !dbg !49
  %address.local1090 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1089, !dbg !49
  store half %1511, ptr addrspace(1) %address.local1090, align 2, !dbg !49
  %address.local1092 = or disjoint i64 %address.local971, 67, !dbg !49
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1092, !dbg !49
  store half %1510, ptr addrspace(1) %address.local1093, align 2, !dbg !49
  %address.local1095 = or disjoint i64 %address.local971, 80, !dbg !49
  %address.local1096 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1095, !dbg !49
  store half %1508, ptr addrspace(1) %address.local1096, align 2, !dbg !49
  %address.local1098 = or disjoint i64 %address.local971, 81, !dbg !49
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1098, !dbg !49
  store half %1507, ptr addrspace(1) %address.local1099, align 2, !dbg !49
  %address.local1101 = or disjoint i64 %address.local971, 82, !dbg !49
  %address.local1102 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1101, !dbg !49
  store half %1506, ptr addrspace(1) %address.local1102, align 2, !dbg !49
  %address.local1104 = or disjoint i64 %address.local971, 83, !dbg !49
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1104, !dbg !49
  store half %1505, ptr addrspace(1) %address.local1105, align 2, !dbg !49
  %address.local1110 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1083, !dbg !49
  store half %1503, ptr addrspace(1) %address.local1110, align 2, !dbg !49
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1086, !dbg !49
  store half %1502, ptr addrspace(1) %address.local1112, align 2, !dbg !49
  %address.local1114 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1089, !dbg !49
  store half %1501, ptr addrspace(1) %address.local1114, align 2, !dbg !49
  %address.local1116 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1092, !dbg !49
  store half %1500, ptr addrspace(1) %address.local1116, align 2, !dbg !49
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1095, !dbg !49
  store half %1498, ptr addrspace(1) %address.local1118, align 2, !dbg !49
  %address.local1120 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1098, !dbg !49
  store half %1497, ptr addrspace(1) %address.local1120, align 2, !dbg !49
  %address.local1122 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1101, !dbg !49
  store half %1496, ptr addrspace(1) %address.local1122, align 2, !dbg !49
  %address.local1124 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1104, !dbg !49
  store half %1495, ptr addrspace(1) %address.local1124, align 2, !dbg !49
  %normalized54 = fmul <4 x float> %row.reciprocal, %1053, !dbg !47
  %1514 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !48
  %1515 = extractelement <4 x half> %1514, i64 3, !dbg !48
  %1516 = extractelement <4 x half> %1514, i64 2, !dbg !48
  %1517 = extractelement <4 x half> %1514, i64 1, !dbg !48
  %1518 = extractelement <4 x half> %1514, i64 0, !dbg !48
  %normalized55 = fmul <4 x float> %row.reciprocal, %1020, !dbg !47
  %1519 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !48
  %1520 = extractelement <4 x half> %1519, i64 3, !dbg !48
  %1521 = extractelement <4 x half> %1519, i64 2, !dbg !48
  %1522 = extractelement <4 x half> %1519, i64 1, !dbg !48
  %1523 = extractelement <4 x half> %1519, i64 0, !dbg !48
  %normalized57 = fmul <4 x float> %row.reciprocal38, %987, !dbg !47
  %1524 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !48
  %1525 = extractelement <4 x half> %1524, i64 3, !dbg !48
  %1526 = extractelement <4 x half> %1524, i64 2, !dbg !48
  %1527 = extractelement <4 x half> %1524, i64 1, !dbg !48
  %1528 = extractelement <4 x half> %1524, i64 0, !dbg !48
  %normalized58 = fmul <4 x float> %row.reciprocal38, %954, !dbg !47
  %1529 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !48
  %1530 = extractelement <4 x half> %1529, i64 3, !dbg !48
  %1531 = extractelement <4 x half> %1529, i64 2, !dbg !48
  %1532 = extractelement <4 x half> %1529, i64 1, !dbg !48
  %1533 = extractelement <4 x half> %1529, i64 0, !dbg !48
  %address.local1139 = or disjoint i64 %address.local971, 96, !dbg !49
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1139, !dbg !49
  store half %1533, ptr addrspace(1) %address.local1140, align 2, !dbg !49
  %address.local1142 = or disjoint i64 %address.local971, 97, !dbg !49
  %address.local1143 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1142, !dbg !49
  store half %1532, ptr addrspace(1) %address.local1143, align 2, !dbg !49
  %address.local1145 = or disjoint i64 %address.local971, 98, !dbg !49
  %address.local1146 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1145, !dbg !49
  store half %1531, ptr addrspace(1) %address.local1146, align 2, !dbg !49
  %address.local1148 = or disjoint i64 %address.local971, 99, !dbg !49
  %address.local1149 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1148, !dbg !49
  store half %1530, ptr addrspace(1) %address.local1149, align 2, !dbg !49
  %address.local1151 = or disjoint i64 %address.local971, 112, !dbg !49
  %address.local1152 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1151, !dbg !49
  store half %1528, ptr addrspace(1) %address.local1152, align 2, !dbg !49
  %address.local1154 = or disjoint i64 %address.local971, 113, !dbg !49
  %address.local1155 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1154, !dbg !49
  store half %1527, ptr addrspace(1) %address.local1155, align 2, !dbg !49
  %address.local1157 = or disjoint i64 %address.local971, 114, !dbg !49
  %address.local1158 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1157, !dbg !49
  store half %1526, ptr addrspace(1) %address.local1158, align 2, !dbg !49
  %address.local1160 = or i64 %1472, 115, !dbg !49
  %address.local1161 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local969, i64 %address.local1160, !dbg !49
  store half %1525, ptr addrspace(1) %address.local1161, align 2, !dbg !49
  %address.local1166 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1139, !dbg !49
  store half %1523, ptr addrspace(1) %address.local1166, align 2, !dbg !49
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1142, !dbg !49
  store half %1522, ptr addrspace(1) %address.local1168, align 2, !dbg !49
  %address.local1170 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1145, !dbg !49
  store half %1521, ptr addrspace(1) %address.local1170, align 2, !dbg !49
  %address.local1172 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1148, !dbg !49
  store half %1520, ptr addrspace(1) %address.local1172, align 2, !dbg !49
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1151, !dbg !49
  store half %1518, ptr addrspace(1) %address.local1174, align 2, !dbg !49
  %address.local1176 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1154, !dbg !49
  store half %1517, ptr addrspace(1) %address.local1176, align 2, !dbg !49
  %address.local1178 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1157, !dbg !49
  store half %1516, ptr addrspace(1) %address.local1178, align 2, !dbg !49
  %address.local1180 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local997, i64 %address.local1160, !dbg !49
  store half %1515, ptr addrspace(1) %address.local1180, align 2, !dbg !49
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
!40 = !DILocation(line: 78, column: 22, scope: !3)
!41 = !DILocation(line: 79, column: 22, scope: !3)
!42 = !DILocation(line: 87, column: 23, scope: !3)
!43 = !{i64 8192}
!44 = !DILocation(line: 88, column: 22, scope: !3)
!45 = !DILocation(line: 81, column: 16, scope: !3)
!46 = !DILocation(line: 80, column: 20, scope: !3)
!47 = !DILocation(line: 92, column: 23, scope: !3)
!48 = !DILocation(line: 93, column: 19, scope: !3)
!49 = !DILocation(line: 94, column: 9, scope: !3)
