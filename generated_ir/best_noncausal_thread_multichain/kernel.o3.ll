; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_thread_multichain/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_thread_multichain/input.mlir"

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
  %address.local276 = zext nneg i32 %37 to i64, !dbg !9
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local276, !dbg !9
  %address.local280 = lshr i32 %address.thread, 4, !dbg !9
  %address.local281 = and i32 %address.local280, 8, !dbg !9
  %address.local282 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local283 = and i32 %address.local282, 16256, !dbg !9
  %address.local284 = or disjoint i32 %address.local281, %address.local283, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1250 = shl i64 %34, 8, !dbg !9
  %39 = add i64 %.idx1250, %.idx, !dbg !9
  %40 = shl nuw nsw i32 %address.local284, 1, !dbg !9
  %41 = trunc i64 %39 to i32, !dbg !9
  %dma.buffer.offset1248 = or disjoint i32 %40, %41, !dbg !9
  %42 = insertelement <2 x i64> poison, i64 %38, i64 0, !dbg !9
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1249 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local289 = or disjoint i64 %address.local276, 2048, !dbg !9
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local289, !dbg !9
  %address.local298 = or disjoint i32 %address.local284, 16, !dbg !9
  %44 = shl nuw nsw i32 %address.local298, 1, !dbg !9
  %dma.buffer.offset1246 = or disjoint i32 %44, %41, !dbg !9
  %45 = shl nuw nsw i64 %31, 19, !dbg !9
  %46 = add i64 %35, %45, !dbg !9
  %address.local301 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local276, !dbg !9
  %47 = trunc i64 %46 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local284, %47, !dbg !9
  %48 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1244 = or disjoint i32 %48, 64, !dbg !9
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local289, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local277, i64 0, !dbg !9
  %50 = insertelement <4 x ptr addrspace(3)> %49, ptr addrspace(3) %address.local290, i64 1, !dbg !9
  %51 = insertelement <4 x ptr addrspace(3)> %50, ptr addrspace(3) %address.local301, i64 2, !dbg !9
  %52 = insertelement <4 x ptr addrspace(3)> %51, ptr addrspace(3) %address.local305, i64 3, !dbg !9
  %53 = ptrtoint <4 x ptr addrspace(3)> %52 to <4 x i64>, !dbg !9
  %54 = trunc <4 x i64> %53 to <4 x i32>, !dbg !9
  %55 = extractelement <4 x i32> %54, i64 0, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1248, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %58 = extractelement <4 x i32> %54, i64 1, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1246, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %61 = extractelement <4 x i32> %54, i64 2, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1244, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %64 = extractelement <4 x i32> %54, i64 3, !dbg !9
  %65 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !9
  %.tr1254 = or disjoint i32 %address.local298, %47, !dbg !9
  %66 = shl i32 %.tr1254, 1, !dbg !9
  %dma.buffer.offset1242 = or disjoint i32 %66, 64, !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1242, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %address.local309 = lshr i32 %address.thread, 2, !dbg !11
  %address.local310 = and i32 %address.local309, 4, !dbg !11
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local312 = and i32 %address.local311, 384, !dbg !11
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !11
  %address.local315 = and i32 %36, 120, !dbg !11
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !11
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local318 = and i32 %address.local317, 1024, !dbg !11
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !11
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !11
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1071 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !11
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1071, align 8, !dbg !11
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1072 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1072, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !11
  %75 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1240 = or disjoint i32 %48, 128, !dbg !9
  %76 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1240, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %dma.buffer.offset1238 = or disjoint i32 %66, 128, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1238, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %lds.const.gep1073 = getelementptr i8, ptr addrspace(3) %address.local321, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1073, align 8, !dbg !11
  %lds.const.gep1074 = getelementptr i8, ptr addrspace(3) %address.local321, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1074, align 8, !dbg !11
  %lds.const.gep1075 = getelementptr i8, ptr addrspace(3) %address.local321, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1075, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1076 = getelementptr i8, ptr addrspace(3) %address.local321, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1076, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1236 = or disjoint i32 %48, 192, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1236, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %dma.buffer.offset1234 = or disjoint i32 %66, 192, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1234, <4 x i32> %dma.buffer.rsrc1249) #9, !dbg !9
  %88 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1071, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1072, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1073, align 8, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1074, align 8, !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1075, align 8, !dbg !11
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1076, align 8, !dbg !11
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
  %k0.prime.1917 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1356 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1357 = shl nuw nsw i64 %k0.prime.kdma.source1356, 9, !dbg !14
  %k0.prime.kdma.source1358 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1359 = add nuw nsw i64 %k0.prime.kdma.source1357, %k0.prime.kdma.source1358, !dbg !14
  %k0.prime.kdma.source1360 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1361 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1362 = and i64 %k0.prime.kdma.source1361, 1, !dbg !14
  %k0.prime.kdma.source1363 = and i64 %k0.prime.kdma.source1362, %k0.prime.kdma.source1356, !dbg !14
  %k0.prime.kdma.source1364 = or disjoint i64 %k0.prime.kdma.source1363, %k0.prime.kdma.source1360, !dbg !14
  %110 = shl nuw nsw i64 %k0.prime.kdma.source1364, 10, !dbg !14
  %k0.prime.kdma.source1366 = sub nsw i64 %k0.prime.kdma.source1359, %110, !dbg !14
  %k0.prime.kdma.source1367 = and i64 %k0.prime.kdma.source1366, 16256, !dbg !14
  %k0.prime.kdma.source1368 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1369 = and i32 %k0.prime.kdma.source1368, 8, !dbg !14
  %k0.prime.kdma.source1370 = zext nneg i32 %k0.prime.kdma.source1369 to i64, !dbg !14
  %k0.prime.kdma.source1371 = or disjoint i64 %k0.prime.kdma.source1367, %k0.prime.kdma.source1370, !dbg !14
  %k0.prime.kdma.source1372 = or disjoint i64 %k0.prime.kdma.source1371, %k0.prime.1917, !dbg !14
  %k0.prime.kdma.source1373 = trunc nuw nsw i64 %k0.prime.kdma.source1372 to i32, !dbg !14
  %k0.prime.kdma.source1374 = shl nuw nsw i32 %k0.prime.kdma.source1373, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1374, %kdma.intra, !dbg !14
  %k0.prime.1920 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.1921 = insertelement <2 x i64> poison, i64 %k0.prime.1920, i64 0, !dbg !14
  %k0.prime.1922 = bitcast <2 x i64> %k0.prime.1921 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1233 = shufflevector <4 x i32> %k0.prime.1922, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.1923 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1396 = add i32 %56, 512, !dbg !14
  %k0.prime.kdma.source1375 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1376 = lshr i64 %k0.prime.kdma.source1375, 4, !dbg !14
  %k0.prime.kdma.source1377 = shl nuw nsw i64 %k0.prime.kdma.source1376, 9, !dbg !14
  %k0.prime.kdma.source1378 = shl nuw nsw i64 %k0.prime.kdma.source1375, 7, !dbg !14
  %k0.prime.kdma.source1379 = add nuw nsw i64 %k0.prime.kdma.source1377, %k0.prime.kdma.source1378, !dbg !14
  %k0.prime.kdma.source1380 = lshr i64 %k0.prime.kdma.source1375, 5, !dbg !14
  %k0.prime.kdma.source1381 = lshr i64 %k0.prime.kdma.source1375, 2, !dbg !14
  %k0.prime.kdma.source1382 = and i64 %k0.prime.kdma.source1381, 1, !dbg !14
  %k0.prime.kdma.source1383 = and i64 %k0.prime.kdma.source1382, %k0.prime.kdma.source1376, !dbg !14
  %k0.prime.kdma.source1384 = add nuw nsw i64 %k0.prime.kdma.source1383, %k0.prime.kdma.source1380, !dbg !14
  %111 = shl nuw nsw i64 %k0.prime.kdma.source1384, 10, !dbg !14
  %k0.prime.kdma.source1386 = sub nsw i64 %k0.prime.kdma.source1379, %111, !dbg !14
  %k0.prime.kdma.source1387 = and i64 %k0.prime.kdma.source1386, 16256, !dbg !14
  %k0.prime.kdma.source1391 = or disjoint i64 %k0.prime.kdma.source1387, %k0.prime.kdma.source1370, !dbg !14
  %k0.prime.kdma.source1392 = or disjoint i64 %k0.prime.kdma.source1391, %k0.prime.1917, !dbg !14
  %k0.prime.kdma.source1393 = trunc nuw nsw i64 %k0.prime.kdma.source1392 to i32, !dbg !14
  %k0.prime.kdma.source1394 = shl nuw nsw i32 %k0.prime.kdma.source1393, 1, !dbg !14
  %k0.prime.kdma.offset1395 = or disjoint i32 %k0.prime.kdma.source1394, %kdma.intra.hi, !dbg !14
  %k0.prime.1924 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1396, i32 %k0.prime.kdma.offset1395, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1398 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1399 = lshr i64 %k0.prime.kdma.source1398, 4, !dbg !14
  %k0.prime.kdma.source1400 = shl nuw nsw i64 %k0.prime.kdma.source1399, 9, !dbg !14
  %k0.prime.kdma.source1402 = add nuw nsw i64 %k0.prime.kdma.source1400, %k0.prime.kdma.source1358, !dbg !14
  %k0.prime.kdma.source1403 = lshr i64 %k0.prime.kdma.source1398, 5, !dbg !14
  %k0.prime.kdma.source1406 = and i64 %k0.prime.kdma.source1399, %k0.prime.kdma.source1362, !dbg !14
  %k0.prime.kdma.source1407 = or disjoint i64 %k0.prime.kdma.source1406, %k0.prime.kdma.source1403, !dbg !14
  %112 = shl nuw nsw i64 %k0.prime.kdma.source1407, 10, !dbg !14
  %k0.prime.kdma.source1409 = sub nsw i64 %k0.prime.kdma.source1402, %112, !dbg !14
  %k0.prime.kdma.source1410 = and i64 %k0.prime.kdma.source1409, 16256, !dbg !14
  %k0.prime.kdma.source1413 = or disjoint i32 %k0.prime.kdma.source1369, 16, !dbg !14
  %k0.prime.kdma.source1414 = zext nneg i32 %k0.prime.kdma.source1413 to i64, !dbg !14
  %k0.prime.kdma.source1415 = or disjoint i64 %k0.prime.kdma.source1410, %k0.prime.kdma.source1414, !dbg !14
  %k0.prime.kdma.source1416 = or disjoint i64 %k0.prime.kdma.source1415, %k0.prime.1917, !dbg !14
  %k0.prime.kdma.source1417 = trunc nuw nsw i64 %k0.prime.kdma.source1416 to i32, !dbg !14
  %k0.prime.kdma.source1418 = shl nuw nsw i32 %k0.prime.kdma.source1417, 1, !dbg !14
  %k0.prime.kdma.offset1419 = add nuw nsw i32 %k0.prime.kdma.source1418, %kdma.intra, !dbg !14
  %k0.prime.1925 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.prime.kdma.offset1419, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1444 = add i32 %59, 512, !dbg !14
  %k0.prime.kdma.source1422 = or disjoint i64 %k0.prime.kdma.source1375, 256, !dbg !14
  %k0.prime.kdma.source1423 = lshr i64 %k0.prime.kdma.source1422, 4, !dbg !14
  %k0.prime.kdma.source1424 = shl nuw nsw i64 %k0.prime.kdma.source1423, 9, !dbg !14
  %k0.prime.kdma.source1426 = add nuw nsw i64 %k0.prime.kdma.source1424, %k0.prime.kdma.source1378, !dbg !14
  %k0.prime.kdma.source1427 = lshr i64 %k0.prime.kdma.source1422, 5, !dbg !14
  %k0.prime.kdma.source1430 = and i64 %k0.prime.kdma.source1423, %k0.prime.kdma.source1382, !dbg !14
  %k0.prime.kdma.source1431 = add nuw nsw i64 %k0.prime.kdma.source1430, %k0.prime.kdma.source1427, !dbg !14
  %113 = shl nuw nsw i64 %k0.prime.kdma.source1431, 10, !dbg !14
  %k0.prime.kdma.source1433 = sub nsw i64 %k0.prime.kdma.source1426, %113, !dbg !14
  %k0.prime.kdma.source1434 = and i64 %k0.prime.kdma.source1433, 16256, !dbg !14
  %k0.prime.kdma.source1439 = or disjoint i64 %k0.prime.kdma.source1434, %k0.prime.kdma.source1414, !dbg !14
  %k0.prime.kdma.source1440 = or disjoint i64 %k0.prime.kdma.source1439, %k0.prime.1917, !dbg !14
  %k0.prime.kdma.source1441 = trunc nuw nsw i64 %k0.prime.kdma.source1440 to i32, !dbg !14
  %k0.prime.kdma.source1442 = shl nuw nsw i32 %k0.prime.kdma.source1441, 1, !dbg !14
  %k0.prime.kdma.offset1443 = add nuw nsw i32 %k0.prime.kdma.source1442, %kdma.intra.hi, !dbg !14
  %k0.prime.1926 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1444, i32 %k0.prime.kdma.offset1443, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1488 = add i32 %62, 512
  %kdma.lds1536 = add i32 %65, 512
  %114 = ptrtoint ptr addrspace(1) %2 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1217 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %56, 8192
  %vdb.dma.base1310 = add i32 %59, 8192
  br label %117, !dbg !12

117:                                              ; preds = %4, %k0.join
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1024, %k0.join ]
  %119 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1020, %k0.join ]
  %120 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1016, %k0.join ]
  %121 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %lazy.base.join, %k0.join ]
  %122 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %789, %k0.join ]
  %123 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1012, %k0.join ]
  %124 = phi i64 [ 0, %4 ], [ %944, %k0.join ]
  %125 = shl nuw nsw i64 %124, 14, !dbg !14
  %126 = add nuw nsw i64 %125, %k0.prime.1917, !dbg !14
  %address.thread372 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local373 = zext nneg i32 %address.thread372 to i64, !dbg !9
  %address.local379 = lshr i64 %address.local373, 2, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %127 = or disjoint i64 %126, 32, !dbg !14
  %kdma.source1462 = or disjoint i64 %127, %k0.prime.kdma.source1371, !dbg !14
  %kdma.source1463 = trunc nuw nsw i64 %kdma.source1462 to i32, !dbg !14
  %kdma.source1464 = shl nuw nsw i32 %kdma.source1463, 1, !dbg !14
  %kdma.offset1465 = or disjoint i32 %kdma.source1464, %kdma.intra, !dbg !14
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1465, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1484 = or disjoint i64 %127, %k0.prime.kdma.source1391, !dbg !14
  %kdma.source1485 = trunc nuw nsw i64 %kdma.source1484 to i32, !dbg !14
  %kdma.source1486 = shl nuw nsw i32 %kdma.source1485, 1, !dbg !14
  %kdma.offset1487 = or disjoint i32 %kdma.source1486, %kdma.intra.hi, !dbg !14
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1488, i32 %kdma.offset1487, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1508 = or disjoint i64 %127, %k0.prime.kdma.source1415, !dbg !14
  %kdma.source1509 = trunc nuw nsw i64 %kdma.source1508 to i32, !dbg !14
  %kdma.source1510 = shl nuw nsw i32 %kdma.source1509, 1, !dbg !14
  %kdma.offset1511 = add nuw nsw i32 %kdma.source1510, %kdma.intra, !dbg !14
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1511, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1532 = or disjoint i64 %127, %k0.prime.kdma.source1439, !dbg !14
  %kdma.source1533 = trunc nuw nsw i64 %kdma.source1532 to i32, !dbg !14
  %kdma.source1534 = shl nuw nsw i32 %kdma.source1533, 1, !dbg !14
  %kdma.offset1535 = add nuw nsw i32 %kdma.source1534, %kdma.intra.hi, !dbg !14
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1536, i32 %kdma.offset1535, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %address.local416 = shl nuw nsw i64 %address.local373, 1, !dbg !13
  %address.local417 = and i64 %address.local416, 24, !dbg !13
  %address.local419 = shl i64 %address.local373, 62, !dbg !13
  %address.local420 = ashr exact i64 %address.local419, 57, !dbg !13
  %address.local421 = and i64 %address.local420, -64, !dbg !13
  %address.local424 = and i64 %address.local379, 12, !dbg !11
  %address.local426 = and i64 %address.local373, 63, !dbg !11
  %address.local427 = lshr i64 %address.local426, 5, !dbg !11
  %address.local428 = mul nuw nsw i64 %address.local427, 1016, !dbg !11
  %.idx1271 = mul nuw nsw i64 %address.local373, 320, !dbg !13
  %132 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1271, !dbg !13
  %.idx1272 = mul nsw i64 %address.local379, -1280, !dbg !13
  %133 = getelementptr i8, ptr addrspace(3) %132, i64 %.idx1272, !dbg !13
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local417, !dbg !13
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local421, !dbg !13
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local424, !dbg !13
  %address.local430 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local428, !dbg !13
  %137 = add nsw i64 %.idx1272, %.idx1271, !dbg !13
  %.idx1847 = shl nuw nsw i64 %address.local417, 1, !dbg !13
  %138 = or disjoint i64 %137, %.idx1847, !dbg !13
  %139 = or disjoint i64 %address.local421, %address.local424, !dbg !13
  %140 = shl nsw i64 %139, 1, !dbg !13
  %address.local430.idx = mul nuw nsw i64 %address.local427, 2032, !dbg !13
  %141 = add nsw i64 %138, %address.local430.idx, !dbg !13
  %142 = add nsw i64 %141, %140, !dbg !13
  %143 = lshr i64 %142, 6, !dbg !13
  %144 = and i64 %143, 8, !dbg !13
  %145 = xor i64 %144, %142, !dbg !13
  %146 = sub i64 %145, %142, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local430, i64 %146, !dbg !13
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %121, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base1932 = extractvalue [2 x <1 x float>] %121, 1, !dbg !13
  %lazy.seed1933 = shufflevector <1 x float> %lazy.base1932, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %147, <4 x float> %lazy.seed) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local441 = shl nuw nsw i32 %address.thread372, 1, !dbg !13
  %address.local442 = and i32 %address.local441, 24, !dbg !13
  %address.local443 = shl nuw nsw i32 %address.thread372, 8, !dbg !13
  %address.local444 = and i32 %address.local443, 768, !dbg !13
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !13
  %address.local446 = shl nuw nsw i32 %address.thread372, 5, !dbg !13
  %address.local447 = and i32 %address.local446, 96, !dbg !13
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !13
  %address.local449 = zext nneg i32 %address.local448 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %109) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local450 = lshr i32 %address.thread372, 1, !dbg !13
  %address.local451 = and i32 %address.local450, 1, !dbg !13
  %address.local452 = zext nneg i32 %address.local451 to i64, !dbg !13
  %address.local453 = add nsw i64 %address.base, %address.local452, !dbg !13
  %address.local454 = shl i64 %address.local453, 6, !dbg !13
  %address.local455 = sub i64 %address.local449, %address.local454, !dbg !13
  %150 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local455, !dbg !13
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local424, !dbg !13
  %152 = getelementptr [2 x i8], ptr addrspace(3) %151, i64 %address.local428, !dbg !13
  %address.local459 = getelementptr i8, ptr addrspace(3) %152, i64 64, !dbg !13
  %.idx18501853 = add i64 %address.local455, %address.local424, !dbg !13
  %153 = shl i64 %.idx18501853, 1, !dbg !13
  %154 = add nuw nsw i64 %address.local430.idx, 64, !dbg !13
  %155 = add i64 %154, %153, !dbg !13
  %156 = lshr i64 %155, 6, !dbg !13
  %157 = and i64 %156, 8, !dbg !13
  %158 = xor i64 %157, %155, !dbg !13
  %159 = sub i64 %158, %155, !dbg !13
  %kdma.swizzled1721 = getelementptr i8, ptr addrspace(3) %address.local459, i64 %159, !dbg !13
  %160 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1721, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1725 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 128, !dbg !13
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1725, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1722 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 4096, !dbg !13
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1726 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 4224, !dbg !13
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1726, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1723 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1727 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1724 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1724, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %167 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %148) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 1024, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 1152, !dbg !13
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %160, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 5120, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 5248, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %171) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %176 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !13, !frisk.k_lds_pair !21
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
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %147, <4 x float> %lazy.seed1933) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %149, <4 x float> %189) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %160, <4 x float> %lazy.seed1933) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %162, <4 x float> %191) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %164, <4 x float> %lazy.seed1933) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %166, <4 x float> %193) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %161, <4 x float> %lazy.seed1933) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %163, <4 x float> %195) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %165, <4 x float> %lazy.seed1933) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %167, <4 x float> %197) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %169, <4 x float> %lazy.seed1933) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %172, <4 x float> %199) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %175, <4 x float> %lazy.seed1933) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %176, <4 x float> %201) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %170, <4 x float> %lazy.seed1933) #8, !dbg !13
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %173, <4 x float> %203) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %205 = or disjoint i64 %126, 64, !dbg !14
  %kdma.source1554 = or disjoint i64 %205, %k0.prime.kdma.source1371, !dbg !14
  %kdma.source1555 = trunc nuw nsw i64 %kdma.source1554 to i32, !dbg !14
  %kdma.source1556 = shl nuw nsw i32 %kdma.source1555, 1, !dbg !14
  %kdma.offset1557 = or disjoint i32 %kdma.source1556, %kdma.intra, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1557, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1576 = or disjoint i64 %205, %k0.prime.kdma.source1391, !dbg !14
  %kdma.source1577 = trunc nuw nsw i64 %kdma.source1576 to i32, !dbg !14
  %kdma.source1578 = shl nuw nsw i32 %kdma.source1577, 1, !dbg !14
  %kdma.offset1579 = or disjoint i32 %kdma.source1578, %kdma.intra.hi, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1396, i32 %kdma.offset1579, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1600 = or disjoint i64 %205, %k0.prime.kdma.source1415, !dbg !14
  %kdma.source1601 = trunc nuw nsw i64 %kdma.source1600 to i32, !dbg !14
  %kdma.source1602 = shl nuw nsw i32 %kdma.source1601, 1, !dbg !14
  %kdma.offset1603 = add nuw nsw i32 %kdma.source1602, %kdma.intra, !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1603, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1624 = or disjoint i64 %205, %k0.prime.kdma.source1439, !dbg !14
  %kdma.source1625 = trunc nuw nsw i64 %kdma.source1624 to i32, !dbg !14
  %kdma.source1626 = shl nuw nsw i32 %kdma.source1625, 1, !dbg !14
  %kdma.offset1627 = add nuw nsw i32 %kdma.source1626, %kdma.intra.hi, !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1444, i32 %kdma.offset1627, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.gep1751 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1755 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1752 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1756 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1756, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1753 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 8192, !dbg !13
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1757 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 8320, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1754 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 12288, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1758 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 12416, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1759 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1763 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %210, <4 x float> %168) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1760 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1764 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %212, <4 x float> %220) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1761 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 9216, !dbg !13
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1765 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 9344, !dbg !13
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %214, <4 x float> %174) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1762 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 13312, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1766 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 13440, !dbg !13
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !13, !frisk.k_lds_pair !29
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
  %kdma.source1646 = or disjoint i64 %258, %k0.prime.kdma.source1371, !dbg !14
  %kdma.source1647 = trunc nuw nsw i64 %kdma.source1646 to i32, !dbg !14
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !14
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !14
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1649, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1668 = or disjoint i64 %258, %k0.prime.kdma.source1391, !dbg !14
  %kdma.source1669 = trunc nuw nsw i64 %kdma.source1668 to i32, !dbg !14
  %kdma.source1670 = shl nuw nsw i32 %kdma.source1669, 1, !dbg !14
  %kdma.offset1671 = or disjoint i32 %kdma.source1670, %kdma.intra.hi, !dbg !14
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1488, i32 %kdma.offset1671, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1692 = or disjoint i64 %258, %k0.prime.kdma.source1415, !dbg !14
  %kdma.source1693 = trunc nuw nsw i64 %kdma.source1692 to i32, !dbg !14
  %kdma.source1694 = shl nuw nsw i32 %kdma.source1693, 1, !dbg !14
  %kdma.offset1695 = add nuw nsw i32 %kdma.source1694, %kdma.intra, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1695, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %kdma.source1716 = or disjoint i64 %258, %k0.prime.kdma.source1439, !dbg !14
  %kdma.source1717 = trunc nuw nsw i64 %kdma.source1716 to i32, !dbg !14
  %kdma.source1718 = shl nuw nsw i32 %kdma.source1717, 1, !dbg !14
  %kdma.offset1719 = add nuw nsw i32 %kdma.source1718, %kdma.intra.hi, !dbg !14
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1536, i32 %kdma.offset1719, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !13, !frisk.k_lds_pair !30
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1724, align 8, !dbg !13, !frisk.k_lds_pair !31
  %267 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1721, align 8, !dbg !13, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1725, align 8, !dbg !13, !frisk.k_lds_pair !32
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !13, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1726, align 8, !dbg !13, !frisk.k_lds_pair !33
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !13, !frisk.k_lds_pair !34
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %263, <4 x float> %223) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !13, !frisk.k_lds_pair !35
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %265, <4 x float> %273) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !13, !frisk.k_lds_pair !36
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %267, <4 x float> %229) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !13, !frisk.k_lds_pair !37
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !13, !frisk.k_lds_pair !37
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
  %address.local788 = shl nuw nsw i64 %address.local373, 5, !dbg !38
  %address.local789 = and i64 %address.local788, 8064, !dbg !38
  %address.local790 = shl nuw nsw i64 %address.local373, 3, !dbg !9
  %address.local791 = shl nuw nsw i64 %address.local379, 5, !dbg !38
  %address.local792 = sub nsw i64 %address.local790, %address.local791, !dbg !38
  %address.local793 = add nsw i64 %address.local792, %address.local789, !dbg !38
  %.idx12811282 = add nsw i64 %address.local793, %126, !dbg !38
  %.idx12811282.tr = trunc nsw i64 %.idx12811282 to i32, !dbg !38
  %dma.buffer.offset1216 = shl nsw i32 %.idx12811282.tr, 1, !dbg !38
  %311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1216, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %address.local795 = shl nuw nsw i32 %address.thread372, 3, !dbg !38
  %address.local796 = and i32 %address.local795, 24, !dbg !38
  %address.local798 = and i32 %address.local446, 8064, !dbg !38
  %address.local799 = or disjoint i32 %address.local798, %address.local796, !dbg !38
  %address.local800 = or disjoint i32 %address.local799, 8192, !dbg !38
  %312 = trunc nuw nsw i64 %126 to i32, !dbg !38
  %.idx12831284.tr = or disjoint i32 %address.local800, %312, !dbg !38
  %dma.buffer.offset1214 = shl nuw nsw i32 %.idx12831284.tr, 1, !dbg !38
  %313 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1214, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !13, !frisk.k_tail_pair !15
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !13, !frisk.k_tail_pair !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !13, !frisk.k_tail_pair !16
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1756, align 8, !dbg !13, !frisk.k_tail_pair !16
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !13, !frisk.k_tail_pair !17
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !13, !frisk.k_tail_pair !17
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !13, !frisk.k_tail_pair !18
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !13, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %314, <4 x float> %276) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %323 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %316, <4 x float> %322) #8, !dbg !13
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %318, <4 x float> %282) #8, !dbg !13
  %325 = extractelement <4 x float> %323, i64 0, !dbg !13
  %326 = tail call float @llvm.maxnum.f32(float %325, float -inf), !dbg !7
  %327 = extractelement <4 x float> %323, i64 1, !dbg !13
  %328 = extractelement <4 x float> %323, i64 2, !dbg !13
  %329 = extractelement <4 x float> %323, i64 3, !dbg !13
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %320, <4 x float> %324) #8, !dbg !13
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !13, !frisk.k_tail_pair !19
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !13, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %315, <4 x float> %284) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %334 = extractelement <4 x float> %330, i64 0, !dbg !13
  %335 = tail call float @llvm.maxnum.f32(float %326, float %334), !dbg !7
  %336 = extractelement <4 x float> %330, i64 1, !dbg !13
  %337 = tail call float @llvm.maxnum.f32(float %327, float %336), !dbg !7
  %338 = extractelement <4 x float> %330, i64 2, !dbg !13
  %339 = tail call float @llvm.maxnum.f32(float %328, float %338), !dbg !7
  %340 = extractelement <4 x float> %330, i64 3, !dbg !13
  %341 = tail call float @llvm.maxnum.f32(float %329, float %340), !dbg !7
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !13, !frisk.k_tail_pair !20
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %317, <4 x float> %333) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !13, !frisk.k_tail_pair !21
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !13, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %319, <4 x float> %286) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %348 = extractelement <4 x float> %344, i64 0, !dbg !13
  %349 = tail call float @llvm.maxnum.f32(float %335, float %348), !dbg !7
  %350 = extractelement <4 x float> %344, i64 1, !dbg !13
  %351 = tail call float @llvm.maxnum.f32(float %337, float %350), !dbg !7
  %352 = extractelement <4 x float> %344, i64 2, !dbg !13
  %353 = tail call float @llvm.maxnum.f32(float %339, float %352), !dbg !7
  %354 = extractelement <4 x float> %344, i64 3, !dbg !13
  %355 = tail call float @llvm.maxnum.f32(float %341, float %354), !dbg !7
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !13, !frisk.k_tail_pair !22
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !13, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %321, <4 x float> %347) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %331, <4 x float> %288) #8, !dbg !13
  %360 = extractelement <4 x float> %358, i64 0, !dbg !13
  %361 = tail call float @llvm.maxnum.f32(float %349, float %360), !dbg !7
  %362 = extractelement <4 x float> %358, i64 1, !dbg !13
  %363 = tail call float @llvm.maxnum.f32(float %351, float %362), !dbg !7
  %364 = extractelement <4 x float> %358, i64 2, !dbg !13
  %365 = tail call float @llvm.maxnum.f32(float %353, float %364), !dbg !7
  %366 = extractelement <4 x float> %358, i64 3, !dbg !13
  %367 = tail call float @llvm.maxnum.f32(float %355, float %366), !dbg !7
  %368 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %342, <4 x float> %359) #8, !dbg !13
  %369 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %345, <4 x float> %290) #8, !dbg !13
  %370 = extractelement <4 x float> %368, i64 0, !dbg !13
  %371 = tail call float @llvm.maxnum.f32(float %361, float %370), !dbg !7
  %372 = extractelement <4 x float> %368, i64 1, !dbg !13
  %373 = tail call float @llvm.maxnum.f32(float %363, float %372), !dbg !7
  %374 = extractelement <4 x float> %368, i64 2, !dbg !13
  %375 = tail call float @llvm.maxnum.f32(float %365, float %374), !dbg !7
  %376 = extractelement <4 x float> %368, i64 3, !dbg !13
  %377 = tail call float @llvm.maxnum.f32(float %367, float %376), !dbg !7
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %356, <4 x float> %369) #8, !dbg !13
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %332, <4 x float> %292) #8, !dbg !13
  %380 = extractelement <4 x float> %378, i64 0, !dbg !13
  %381 = tail call float @llvm.maxnum.f32(float %371, float %380), !dbg !7
  %382 = extractelement <4 x float> %378, i64 1, !dbg !13
  %383 = tail call float @llvm.maxnum.f32(float %373, float %382), !dbg !7
  %384 = extractelement <4 x float> %378, i64 2, !dbg !13
  %385 = tail call float @llvm.maxnum.f32(float %375, float %384), !dbg !7
  %386 = extractelement <4 x float> %378, i64 3, !dbg !13
  %387 = tail call float @llvm.maxnum.f32(float %377, float %386), !dbg !7
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %343, <4 x float> %379) #8, !dbg !13
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %99, <4 x half> %346, <4 x float> %294) #8, !dbg !13
  %390 = extractelement <4 x float> %388, i64 0, !dbg !13
  %391 = tail call float @llvm.maxnum.f32(float %381, float %390), !dbg !7
  %392 = extractelement <4 x float> %388, i64 1, !dbg !13
  %393 = tail call float @llvm.maxnum.f32(float %383, float %392), !dbg !7
  %394 = extractelement <4 x float> %388, i64 2, !dbg !13
  %395 = tail call float @llvm.maxnum.f32(float %385, float %394), !dbg !7
  %396 = extractelement <4 x float> %388, i64 3, !dbg !13
  %397 = tail call float @llvm.maxnum.f32(float %387, float %396), !dbg !7
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %357, <4 x float> %389) #8, !dbg !13
  %399 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %314, <4 x float> %296) #8, !dbg !13
  %400 = extractelement <4 x float> %398, i64 0, !dbg !13
  %401 = tail call float @llvm.maxnum.f32(float %391, float %400), !dbg !7
  %402 = extractelement <4 x float> %398, i64 1, !dbg !13
  %403 = tail call float @llvm.maxnum.f32(float %393, float %402), !dbg !7
  %404 = extractelement <4 x float> %398, i64 2, !dbg !13
  %405 = tail call float @llvm.maxnum.f32(float %395, float %404), !dbg !7
  %406 = extractelement <4 x float> %398, i64 3, !dbg !13
  %407 = tail call float @llvm.maxnum.f32(float %397, float %406), !dbg !7
  %408 = tail call float @llvm.maxnum.f32(float %401, float %403), !dbg !7
  %409 = tail call float @llvm.maxnum.f32(float %405, float %407), !dbg !7
  %410 = tail call float @llvm.maxnum.f32(float %408, float %409), !dbg !7
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %316, <4 x float> %399) #8, !dbg !13
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %318, <4 x float> %298) #8, !dbg !13
  %413 = extractelement <4 x float> %411, i64 0, !dbg !13
  %414 = extractelement <4 x float> %411, i64 1, !dbg !13
  %415 = extractelement <4 x float> %411, i64 2, !dbg !13
  %416 = extractelement <4 x float> %411, i64 3, !dbg !13
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %320, <4 x float> %412) #8, !dbg !13
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %315, <4 x float> %300) #8, !dbg !13
  %419 = extractelement <4 x float> %417, i64 0, !dbg !13
  %420 = extractelement <4 x float> %417, i64 1, !dbg !13
  %421 = extractelement <4 x float> %417, i64 2, !dbg !13
  %422 = extractelement <4 x float> %417, i64 3, !dbg !13
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %317, <4 x float> %418) #8, !dbg !13
  %424 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %319, <4 x float> %302) #8, !dbg !13
  %425 = extractelement <4 x float> %423, i64 0, !dbg !13
  %426 = extractelement <4 x float> %423, i64 1, !dbg !13
  %427 = extractelement <4 x float> %423, i64 2, !dbg !13
  %428 = extractelement <4 x float> %423, i64 3, !dbg !13
  %429 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %321, <4 x float> %424) #8, !dbg !13
  %430 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %331, <4 x float> %304) #8, !dbg !13
  %431 = extractelement <4 x float> %429, i64 0, !dbg !13
  %432 = extractelement <4 x float> %429, i64 1, !dbg !13
  %433 = extractelement <4 x float> %429, i64 2, !dbg !13
  %434 = extractelement <4 x float> %429, i64 3, !dbg !13
  %435 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %342, <4 x float> %430) #8, !dbg !13
  %436 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %345, <4 x float> %306) #8, !dbg !13
  %437 = extractelement <4 x float> %435, i64 0, !dbg !13
  %438 = extractelement <4 x float> %435, i64 1, !dbg !13
  %439 = extractelement <4 x float> %435, i64 2, !dbg !13
  %440 = extractelement <4 x float> %435, i64 3, !dbg !13
  %441 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %356, <4 x float> %436) #8, !dbg !13
  %442 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %332, <4 x float> %308) #8, !dbg !13
  %443 = extractelement <4 x float> %441, i64 0, !dbg !13
  %444 = extractelement <4 x float> %441, i64 1, !dbg !13
  %445 = extractelement <4 x float> %441, i64 2, !dbg !13
  %446 = extractelement <4 x float> %441, i64 3, !dbg !13
  %447 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %343, <4 x float> %442) #8, !dbg !13
  %448 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %346, <4 x float> %310) #8, !dbg !13
  %449 = extractelement <4 x float> %447, i64 0, !dbg !13
  %450 = extractelement <4 x float> %447, i64 1, !dbg !13
  %451 = extractelement <4 x float> %447, i64 2, !dbg !13
  %452 = extractelement <4 x float> %447, i64 3, !dbg !13
  %453 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %103, <4 x half> %357, <4 x float> %448) #8, !dbg !13
  %454 = extractelement <4 x float> %453, i64 0, !dbg !13
  %455 = extractelement <4 x float> %453, i64 1, !dbg !13
  %456 = extractelement <4 x float> %453, i64 2, !dbg !13
  %457 = extractelement <4 x float> %453, i64 3, !dbg !13
  %458 = bitcast float %410 to i32, !dbg !7
  %459 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %458), !dbg !7
  %460 = bitcast i32 %459 to float, !dbg !7
  %461 = tail call float @llvm.maxnum.f32(float %410, float %460), !dbg !7
  %462 = bitcast float %461 to i32, !dbg !7
  %463 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %462), !dbg !7
  %464 = bitcast i32 %463 to float, !dbg !7
  %465 = tail call float @llvm.maxnum.f32(float %461, float %464), !dbg !7
  %466 = tail call float @llvm.maxnum.f32(float %413, float -inf), !dbg !7
  %467 = tail call float @llvm.maxnum.f32(float %466, float %419), !dbg !7
  %468 = tail call float @llvm.maxnum.f32(float %414, float %420), !dbg !7
  %469 = tail call float @llvm.maxnum.f32(float %415, float %421), !dbg !7
  %470 = tail call float @llvm.maxnum.f32(float %416, float %422), !dbg !7
  %471 = tail call float @llvm.maxnum.f32(float %467, float %425), !dbg !7
  %472 = tail call float @llvm.maxnum.f32(float %468, float %426), !dbg !7
  %473 = tail call float @llvm.maxnum.f32(float %469, float %427), !dbg !7
  %474 = tail call float @llvm.maxnum.f32(float %470, float %428), !dbg !7
  %475 = tail call float @llvm.maxnum.f32(float %471, float %431), !dbg !7
  %476 = tail call float @llvm.maxnum.f32(float %472, float %432), !dbg !7
  %477 = tail call float @llvm.maxnum.f32(float %473, float %433), !dbg !7
  %478 = tail call float @llvm.maxnum.f32(float %474, float %434), !dbg !7
  %479 = tail call float @llvm.maxnum.f32(float %475, float %437), !dbg !7
  %480 = tail call float @llvm.maxnum.f32(float %476, float %438), !dbg !7
  %481 = tail call float @llvm.maxnum.f32(float %477, float %439), !dbg !7
  %482 = tail call float @llvm.maxnum.f32(float %478, float %440), !dbg !7
  %483 = tail call float @llvm.maxnum.f32(float %479, float %443), !dbg !7
  %484 = tail call float @llvm.maxnum.f32(float %480, float %444), !dbg !7
  %485 = tail call float @llvm.maxnum.f32(float %481, float %445), !dbg !7
  %486 = tail call float @llvm.maxnum.f32(float %482, float %446), !dbg !7
  %487 = tail call float @llvm.maxnum.f32(float %483, float %449), !dbg !7
  %488 = tail call float @llvm.maxnum.f32(float %484, float %450), !dbg !7
  %489 = tail call float @llvm.maxnum.f32(float %485, float %451), !dbg !7
  %490 = tail call float @llvm.maxnum.f32(float %486, float %452), !dbg !7
  %491 = tail call float @llvm.maxnum.f32(float %487, float %454), !dbg !7
  %492 = tail call float @llvm.maxnum.f32(float %488, float %455), !dbg !7
  %493 = tail call float @llvm.maxnum.f32(float %489, float %456), !dbg !7
  %494 = tail call float @llvm.maxnum.f32(float %490, float %457), !dbg !7
  %495 = tail call float @llvm.maxnum.f32(float %491, float %492), !dbg !7
  %496 = tail call float @llvm.maxnum.f32(float %493, float %494), !dbg !7
  %497 = tail call float @llvm.maxnum.f32(float %495, float %496), !dbg !7
  %498 = bitcast float %497 to i32, !dbg !7
  %499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %498), !dbg !7
  %500 = bitcast i32 %499 to float, !dbg !7
  %501 = tail call float @llvm.maxnum.f32(float %497, float %500), !dbg !7
  %502 = bitcast float %501 to i32, !dbg !7
  %503 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %502), !dbg !7
  %504 = bitcast i32 %503 to float, !dbg !7
  %505 = tail call float @llvm.maxnum.f32(float %501, float %504), !dbg !7
  %lazy.peak = tail call float @llvm.maxnum.f32(float %465, float %505), !dbg !39
  %lazy.over = fcmp ogt float %lazy.peak, 8.000000e+00, !dbg !39
  %lazy.mask = tail call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !39
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !39
  %lazy.first = icmp eq i64 %124, 0, !dbg !39
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !39
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !39

lazy.slow:                                        ; preds = %117
  %lazy.positive = tail call float @llvm.maxnum.f32(float %465, float 0.000000e+00), !dbg !39
  %lazy.delta = select i1 %lazy.first, float %465, float %lazy.positive, !dbg !39
  %lazy.negated = fsub float 0.000000e+00, %lazy.delta, !dbg !39
  %lazy.exp = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !39
  %lazy.alpha = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp, !dbg !39
  %lazy.positive1934 = tail call float @llvm.maxnum.f32(float %505, float 0.000000e+00), !dbg !39
  %lazy.delta1935 = select i1 %lazy.first, float %505, float %lazy.positive1934, !dbg !39
  %lazy.negated1936 = fsub float 0.000000e+00, %lazy.delta1935, !dbg !39
  %lazy.exp1937 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated1936), !dbg !39
  %lazy.alpha1938 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp1937, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %323, %lazy.delta.v, !dbg !39
  %lazy.shifted1941 = fsub <4 x float> %330, %lazy.delta.v, !dbg !39
  %lazy.shifted1944 = fsub <4 x float> %344, %lazy.delta.v, !dbg !39
  %lazy.shifted1947 = fsub <4 x float> %358, %lazy.delta.v, !dbg !39
  %lazy.shifted1950 = fsub <4 x float> %368, %lazy.delta.v, !dbg !39
  %lazy.shifted1953 = fsub <4 x float> %378, %lazy.delta.v, !dbg !39
  %lazy.shifted1956 = fsub <4 x float> %388, %lazy.delta.v, !dbg !39
  %lazy.shifted1959 = fsub <4 x float> %398, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane1960 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !39
  %lazy.delta.v1961 = shufflevector <1 x float> %lazy.delta.v.lane1960, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted1962 = fsub <4 x float> %411, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1965 = fsub <4 x float> %417, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1968 = fsub <4 x float> %423, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1971 = fsub <4 x float> %429, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1974 = fsub <4 x float> %435, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1977 = fsub <4 x float> %441, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1980 = fsub <4 x float> %447, %lazy.delta.v1961, !dbg !39
  %lazy.shifted1983 = fsub <4 x float> %453, %lazy.delta.v1961, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %122, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state1984 = extractvalue [2 x <1 x float>] %122, 1, !dbg !39
  %lazy.alpha.v.lane1985 = insertelement <1 x float> poison, float %lazy.alpha1938, i64 0, !dbg !39
  %lazy.scaled1986 = fmul <1 x float> %lazy.state1984, %lazy.alpha.v.lane1985, !dbg !39
  %lazy.state.next1987 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1986, 1, !dbg !39
  %lazy.state1988 = extractvalue [2 x <8 x float>] %118, 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled1990 = fmul <8 x float> %lazy.state1988, %lazy.alpha.v, !dbg !39
  %lazy.state.next1991 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1990, 0, !dbg !39
  %lazy.state1992 = extractvalue [2 x <8 x float>] %118, 1, !dbg !39
  %lazy.alpha.v1994 = shufflevector <1 x float> %lazy.alpha.v.lane1985, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled1995 = fmul <8 x float> %lazy.state1992, %lazy.alpha.v1994, !dbg !39
  %lazy.state.next1996 = insertvalue [2 x <8 x float>] %lazy.state.next1991, <8 x float> %lazy.scaled1995, 1, !dbg !39
  %lazy.state1997 = extractvalue [2 x <8 x float>] %119, 0, !dbg !39
  %lazy.scaled2000 = fmul <8 x float> %lazy.state1997, %lazy.alpha.v, !dbg !39
  %lazy.state.next2001 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2000, 0, !dbg !39
  %lazy.state2002 = extractvalue [2 x <8 x float>] %119, 1, !dbg !39
  %lazy.scaled2005 = fmul <8 x float> %lazy.state2002, %lazy.alpha.v1994, !dbg !39
  %lazy.state.next2006 = insertvalue [2 x <8 x float>] %lazy.state.next2001, <8 x float> %lazy.scaled2005, 1, !dbg !39
  %lazy.state2007 = extractvalue [2 x <8 x float>] %120, 0, !dbg !39
  %lazy.scaled2010 = fmul <8 x float> %lazy.state2007, %lazy.alpha.v, !dbg !39
  %lazy.state.next2011 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2010, 0, !dbg !39
  %lazy.state2012 = extractvalue [2 x <8 x float>] %120, 1, !dbg !39
  %lazy.scaled2015 = fmul <8 x float> %lazy.state2012, %lazy.alpha.v1994, !dbg !39
  %lazy.state.next2016 = insertvalue [2 x <8 x float>] %lazy.state.next2011, <8 x float> %lazy.scaled2015, 1, !dbg !39
  %lazy.state2017 = extractvalue [2 x <8 x float>] %123, 0, !dbg !39
  %lazy.scaled2020 = fmul <8 x float> %lazy.state2017, %lazy.alpha.v, !dbg !39
  %lazy.state.next2021 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2020, 0, !dbg !39
  %lazy.state2022 = extractvalue [2 x <8 x float>] %123, 1, !dbg !39
  %lazy.scaled2025 = fmul <8 x float> %lazy.state2022, %lazy.alpha.v1994, !dbg !39
  %lazy.state.next2026 = insertvalue [2 x <8 x float>] %lazy.state.next2021, <8 x float> %lazy.scaled2025, 1, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.next2029 = fsub <1 x float> %lazy.base1932, %lazy.delta.v.lane1960, !dbg !39
  %lazy.base.state2030 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2029, 1, !dbg !39
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !40
  %.pre25 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !40
  %.pre27 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !40
  %.pre29 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !40
  %.pre31 = extractelement <4 x float> %lazy.shifted1962, i64 0, !dbg !40
  %.pre33 = extractelement <4 x float> %lazy.shifted1962, i64 1, !dbg !40
  %.pre35 = extractelement <4 x float> %lazy.shifted1962, i64 2, !dbg !40
  %.pre37 = extractelement <4 x float> %lazy.shifted1962, i64 3, !dbg !40
  %.pre39 = extractelement <4 x float> %lazy.shifted1941, i64 0, !dbg !40
  %.pre41 = extractelement <4 x float> %lazy.shifted1941, i64 1, !dbg !40
  %.pre43 = extractelement <4 x float> %lazy.shifted1941, i64 2, !dbg !40
  %.pre45 = extractelement <4 x float> %lazy.shifted1941, i64 3, !dbg !40
  %.pre47 = extractelement <4 x float> %lazy.shifted1965, i64 0, !dbg !40
  %.pre49 = extractelement <4 x float> %lazy.shifted1965, i64 1, !dbg !40
  %.pre51 = extractelement <4 x float> %lazy.shifted1965, i64 2, !dbg !40
  %.pre53 = extractelement <4 x float> %lazy.shifted1965, i64 3, !dbg !40
  %.pre55 = extractelement <4 x float> %lazy.shifted1944, i64 0, !dbg !40
  %.pre57 = extractelement <4 x float> %lazy.shifted1944, i64 1, !dbg !40
  %.pre59 = extractelement <4 x float> %lazy.shifted1944, i64 2, !dbg !40
  %.pre61 = extractelement <4 x float> %lazy.shifted1944, i64 3, !dbg !40
  %.pre63 = extractelement <4 x float> %lazy.shifted1968, i64 0, !dbg !40
  %.pre65 = extractelement <4 x float> %lazy.shifted1968, i64 1, !dbg !40
  %.pre67 = extractelement <4 x float> %lazy.shifted1968, i64 2, !dbg !40
  %.pre69 = extractelement <4 x float> %lazy.shifted1968, i64 3, !dbg !40
  %.pre71 = extractelement <4 x float> %lazy.shifted1947, i64 0, !dbg !40
  %.pre73 = extractelement <4 x float> %lazy.shifted1947, i64 1, !dbg !40
  %.pre75 = extractelement <4 x float> %lazy.shifted1947, i64 2, !dbg !40
  %.pre77 = extractelement <4 x float> %lazy.shifted1947, i64 3, !dbg !40
  %.pre79 = extractelement <4 x float> %lazy.shifted1971, i64 0, !dbg !40
  %.pre81 = extractelement <4 x float> %lazy.shifted1971, i64 1, !dbg !40
  %.pre83 = extractelement <4 x float> %lazy.shifted1971, i64 2, !dbg !40
  %.pre85 = extractelement <4 x float> %lazy.shifted1971, i64 3, !dbg !40
  %.pre87 = extractelement <4 x float> %lazy.shifted1950, i64 0, !dbg !40
  %.pre89 = extractelement <4 x float> %lazy.shifted1950, i64 1, !dbg !40
  %.pre91 = extractelement <4 x float> %lazy.shifted1950, i64 2, !dbg !40
  %.pre93 = extractelement <4 x float> %lazy.shifted1950, i64 3, !dbg !40
  %.pre95 = extractelement <4 x float> %lazy.shifted1974, i64 0, !dbg !40
  %.pre97 = extractelement <4 x float> %lazy.shifted1974, i64 1, !dbg !40
  %.pre99 = extractelement <4 x float> %lazy.shifted1974, i64 2, !dbg !40
  %.pre101 = extractelement <4 x float> %lazy.shifted1974, i64 3, !dbg !40
  %.pre103 = extractelement <4 x float> %lazy.shifted1953, i64 0, !dbg !40
  %.pre105 = extractelement <4 x float> %lazy.shifted1953, i64 1, !dbg !40
  %.pre107 = extractelement <4 x float> %lazy.shifted1953, i64 2, !dbg !40
  %.pre109 = extractelement <4 x float> %lazy.shifted1953, i64 3, !dbg !40
  %.pre111 = extractelement <4 x float> %lazy.shifted1977, i64 0, !dbg !40
  %.pre113 = extractelement <4 x float> %lazy.shifted1977, i64 1, !dbg !40
  %.pre115 = extractelement <4 x float> %lazy.shifted1977, i64 2, !dbg !40
  %.pre117 = extractelement <4 x float> %lazy.shifted1977, i64 3, !dbg !40
  %.pre119 = extractelement <4 x float> %lazy.shifted1956, i64 0, !dbg !40
  %.pre121 = extractelement <4 x float> %lazy.shifted1956, i64 1, !dbg !40
  %.pre123 = extractelement <4 x float> %lazy.shifted1956, i64 2, !dbg !40
  %.pre125 = extractelement <4 x float> %lazy.shifted1956, i64 3, !dbg !40
  %.pre127 = extractelement <4 x float> %lazy.shifted1980, i64 0, !dbg !40
  %.pre129 = extractelement <4 x float> %lazy.shifted1980, i64 1, !dbg !40
  %.pre131 = extractelement <4 x float> %lazy.shifted1980, i64 2, !dbg !40
  %.pre133 = extractelement <4 x float> %lazy.shifted1980, i64 3, !dbg !40
  %.pre135 = extractelement <4 x float> %lazy.shifted1959, i64 0, !dbg !40
  %.pre137 = extractelement <4 x float> %lazy.shifted1959, i64 1, !dbg !40
  %.pre139 = extractelement <4 x float> %lazy.shifted1959, i64 2, !dbg !40
  %.pre141 = extractelement <4 x float> %lazy.shifted1959, i64 3, !dbg !40
  %.pre143 = extractelement <4 x float> %lazy.shifted1983, i64 0, !dbg !40
  %.pre145 = extractelement <4 x float> %lazy.shifted1983, i64 1, !dbg !40
  %.pre147 = extractelement <4 x float> %lazy.shifted1983, i64 2, !dbg !40
  %.pre149 = extractelement <4 x float> %lazy.shifted1983, i64 3, !dbg !40
  br label %lazy.join, !dbg !39

lazy.join:                                        ; preds = %lazy.slow, %117
  %.pre-phi150 = phi float [ %.pre149, %lazy.slow ], [ %457, %117 ], !dbg !40
  %.pre-phi148 = phi float [ %.pre147, %lazy.slow ], [ %456, %117 ], !dbg !40
  %.pre-phi146 = phi float [ %.pre145, %lazy.slow ], [ %455, %117 ], !dbg !40
  %.pre-phi144 = phi float [ %.pre143, %lazy.slow ], [ %454, %117 ], !dbg !40
  %.pre-phi142 = phi float [ %.pre141, %lazy.slow ], [ %406, %117 ], !dbg !40
  %.pre-phi140 = phi float [ %.pre139, %lazy.slow ], [ %404, %117 ], !dbg !40
  %.pre-phi138 = phi float [ %.pre137, %lazy.slow ], [ %402, %117 ], !dbg !40
  %.pre-phi136 = phi float [ %.pre135, %lazy.slow ], [ %400, %117 ], !dbg !40
  %.pre-phi134 = phi float [ %.pre133, %lazy.slow ], [ %452, %117 ], !dbg !40
  %.pre-phi132 = phi float [ %.pre131, %lazy.slow ], [ %451, %117 ], !dbg !40
  %.pre-phi130 = phi float [ %.pre129, %lazy.slow ], [ %450, %117 ], !dbg !40
  %.pre-phi128 = phi float [ %.pre127, %lazy.slow ], [ %449, %117 ], !dbg !40
  %.pre-phi126 = phi float [ %.pre125, %lazy.slow ], [ %396, %117 ], !dbg !40
  %.pre-phi124 = phi float [ %.pre123, %lazy.slow ], [ %394, %117 ], !dbg !40
  %.pre-phi122 = phi float [ %.pre121, %lazy.slow ], [ %392, %117 ], !dbg !40
  %.pre-phi120 = phi float [ %.pre119, %lazy.slow ], [ %390, %117 ], !dbg !40
  %.pre-phi118 = phi float [ %.pre117, %lazy.slow ], [ %446, %117 ], !dbg !40
  %.pre-phi116 = phi float [ %.pre115, %lazy.slow ], [ %445, %117 ], !dbg !40
  %.pre-phi114 = phi float [ %.pre113, %lazy.slow ], [ %444, %117 ], !dbg !40
  %.pre-phi112 = phi float [ %.pre111, %lazy.slow ], [ %443, %117 ], !dbg !40
  %.pre-phi110 = phi float [ %.pre109, %lazy.slow ], [ %386, %117 ], !dbg !40
  %.pre-phi108 = phi float [ %.pre107, %lazy.slow ], [ %384, %117 ], !dbg !40
  %.pre-phi106 = phi float [ %.pre105, %lazy.slow ], [ %382, %117 ], !dbg !40
  %.pre-phi104 = phi float [ %.pre103, %lazy.slow ], [ %380, %117 ], !dbg !40
  %.pre-phi102 = phi float [ %.pre101, %lazy.slow ], [ %440, %117 ], !dbg !40
  %.pre-phi100 = phi float [ %.pre99, %lazy.slow ], [ %439, %117 ], !dbg !40
  %.pre-phi98 = phi float [ %.pre97, %lazy.slow ], [ %438, %117 ], !dbg !40
  %.pre-phi96 = phi float [ %.pre95, %lazy.slow ], [ %437, %117 ], !dbg !40
  %.pre-phi94 = phi float [ %.pre93, %lazy.slow ], [ %376, %117 ], !dbg !40
  %.pre-phi92 = phi float [ %.pre91, %lazy.slow ], [ %374, %117 ], !dbg !40
  %.pre-phi90 = phi float [ %.pre89, %lazy.slow ], [ %372, %117 ], !dbg !40
  %.pre-phi88 = phi float [ %.pre87, %lazy.slow ], [ %370, %117 ], !dbg !40
  %.pre-phi86 = phi float [ %.pre85, %lazy.slow ], [ %434, %117 ], !dbg !40
  %.pre-phi84 = phi float [ %.pre83, %lazy.slow ], [ %433, %117 ], !dbg !40
  %.pre-phi82 = phi float [ %.pre81, %lazy.slow ], [ %432, %117 ], !dbg !40
  %.pre-phi80 = phi float [ %.pre79, %lazy.slow ], [ %431, %117 ], !dbg !40
  %.pre-phi78 = phi float [ %.pre77, %lazy.slow ], [ %366, %117 ], !dbg !40
  %.pre-phi76 = phi float [ %.pre75, %lazy.slow ], [ %364, %117 ], !dbg !40
  %.pre-phi74 = phi float [ %.pre73, %lazy.slow ], [ %362, %117 ], !dbg !40
  %.pre-phi72 = phi float [ %.pre71, %lazy.slow ], [ %360, %117 ], !dbg !40
  %.pre-phi70 = phi float [ %.pre69, %lazy.slow ], [ %428, %117 ], !dbg !40
  %.pre-phi68 = phi float [ %.pre67, %lazy.slow ], [ %427, %117 ], !dbg !40
  %.pre-phi66 = phi float [ %.pre65, %lazy.slow ], [ %426, %117 ], !dbg !40
  %.pre-phi64 = phi float [ %.pre63, %lazy.slow ], [ %425, %117 ], !dbg !40
  %.pre-phi62 = phi float [ %.pre61, %lazy.slow ], [ %354, %117 ], !dbg !40
  %.pre-phi60 = phi float [ %.pre59, %lazy.slow ], [ %352, %117 ], !dbg !40
  %.pre-phi58 = phi float [ %.pre57, %lazy.slow ], [ %350, %117 ], !dbg !40
  %.pre-phi56 = phi float [ %.pre55, %lazy.slow ], [ %348, %117 ], !dbg !40
  %.pre-phi54 = phi float [ %.pre53, %lazy.slow ], [ %422, %117 ], !dbg !40
  %.pre-phi52 = phi float [ %.pre51, %lazy.slow ], [ %421, %117 ], !dbg !40
  %.pre-phi50 = phi float [ %.pre49, %lazy.slow ], [ %420, %117 ], !dbg !40
  %.pre-phi48 = phi float [ %.pre47, %lazy.slow ], [ %419, %117 ], !dbg !40
  %.pre-phi46 = phi float [ %.pre45, %lazy.slow ], [ %340, %117 ], !dbg !40
  %.pre-phi44 = phi float [ %.pre43, %lazy.slow ], [ %338, %117 ], !dbg !40
  %.pre-phi42 = phi float [ %.pre41, %lazy.slow ], [ %336, %117 ], !dbg !40
  %.pre-phi40 = phi float [ %.pre39, %lazy.slow ], [ %334, %117 ], !dbg !40
  %.pre-phi38 = phi float [ %.pre37, %lazy.slow ], [ %416, %117 ], !dbg !40
  %.pre-phi36 = phi float [ %.pre35, %lazy.slow ], [ %415, %117 ], !dbg !40
  %.pre-phi34 = phi float [ %.pre33, %lazy.slow ], [ %414, %117 ], !dbg !40
  %.pre-phi32 = phi float [ %.pre31, %lazy.slow ], [ %413, %117 ], !dbg !40
  %.pre-phi30 = phi float [ %.pre29, %lazy.slow ], [ %329, %117 ], !dbg !40
  %.pre-phi28 = phi float [ %.pre27, %lazy.slow ], [ %328, %117 ], !dbg !40
  %.pre-phi26 = phi float [ %.pre25, %lazy.slow ], [ %327, %117 ], !dbg !40
  %.pre-phi = phi float [ %.pre, %lazy.slow ], [ %325, %117 ], !dbg !40
  %lazy.state2046 = phi [2 x <1 x float>] [ %lazy.state.next1987, %lazy.slow ], [ %122, %117 ], !dbg !39
  %lazy.state2047 = phi [2 x <8 x float>] [ %lazy.state.next1996, %lazy.slow ], [ %118, %117 ], !dbg !39
  %lazy.state2048 = phi [2 x <8 x float>] [ %lazy.state.next2006, %lazy.slow ], [ %119, %117 ], !dbg !39
  %lazy.state2049 = phi [2 x <8 x float>] [ %lazy.state.next2016, %lazy.slow ], [ %120, %117 ], !dbg !39
  %lazy.state2050 = phi [2 x <8 x float>] [ %lazy.state.next2026, %lazy.slow ], [ %123, %117 ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2030, %lazy.slow ], [ %121, %117 ], !dbg !39
  %506 = extractvalue [2 x <1 x float>] %lazy.state2046, 0, !dbg !39
  %507 = extractelement <1 x float> %506, i64 0, !dbg !39
  %508 = extractvalue [2 x <1 x float>] %lazy.state2046, 1, !dbg !39
  %509 = extractelement <1 x float> %508, i64 0, !dbg !39
  %510 = extractvalue [2 x <8 x float>] %lazy.state2050, 0, !dbg !41
  %511 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %512 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %513 = extractvalue [2 x <8 x float>] %lazy.state2050, 1, !dbg !41
  %514 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %515 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = add nsw i64 %address.local793, %127, !dbg !38
  %vdb.address1297 = trunc nsw i64 %vdb.address to i32, !dbg !38
  %vdb.address1298 = shl nsw i32 %vdb.address1297, 1, !dbg !38
  %vdb.address1303 = trunc nuw nsw i64 %127 to i32, !dbg !38
  %vdb.address1304 = or disjoint i32 %address.local800, %vdb.address1303, !dbg !38
  %vdb.address1305 = shl nuw nsw i32 %vdb.address1304, 1, !dbg !38
  %516 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1298, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %517 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1310, i32 %vdb.address1305, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %address.local804 = and i32 %address.local795, 120, !dbg !42
  %address.local806 = and i32 %address.local446, 1536, !dbg !42
  %address.local807 = or disjoint i32 %address.local804, %address.local806, !dbg !42
  %address.local808 = zext nneg i32 %address.local807 to i64, !dbg !42
  %address.local809 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local808, !dbg !42
  %518 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local809, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %519 = bitcast <4 x i32> %518 to <8 x half>, !dbg !42
  %address.local810 = and i64 %address.local373, 60, !dbg !42
  %address.local811 = add nuw nsw i64 %address.local810, %address.local373, !dbg !42
  %address.local813.idx = shl nuw nsw i64 %address.local811, 4, !dbg !42
  %address.local813 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local813.idx, !dbg !42
  %address.local814 = lshr i64 %address.local426, 4, !dbg !42
  %address.local816.idx = mul nuw nsw i64 %address.local814, 768, !dbg !42
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local813, i64 %address.local816.idx, !dbg !42
  %address.local818.idx = mul nsw i64 %address.local379, -64, !dbg !42
  %address.local818 = getelementptr i8, ptr addrspace(3) %address.local816, i64 %address.local818.idx, !dbg !42
  %address.local819 = getelementptr i8, ptr addrspace(3) %address.local818, i64 256, !dbg !42
  %520 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %521 = bitcast <4 x i32> %520 to <8 x half>, !dbg !42
  %address.local823 = getelementptr i8, ptr addrspace(3) %address.local818, i64 512, !dbg !42
  %522 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %523 = bitcast <4 x i32> %522 to <8 x half>, !dbg !42
  %address.local827 = getelementptr i8, ptr addrspace(3) %address.local818, i64 768, !dbg !42
  %524 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local827, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %525 = bitcast <4 x i32> %524 to <8 x half>, !dbg !42
  %address.local829 = getelementptr i8, ptr addrspace(3) %address.local809, i64 4096, !dbg !42
  %526 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local829, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %527 = bitcast <4 x i32> %526 to <8 x half>, !dbg !42
  %address.local833 = getelementptr i8, ptr addrspace(3) %address.local818, i64 4352, !dbg !42
  %528 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local833, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %529 = bitcast <4 x i32> %528 to <8 x half>, !dbg !42
  %address.local837 = getelementptr i8, ptr addrspace(3) %address.local818, i64 4608, !dbg !42
  %530 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local837, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %531 = bitcast <4 x i32> %530 to <8 x half>, !dbg !42
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local818, i64 4864, !dbg !42
  %532 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %533 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !40
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !44
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi26), !dbg !40
  %537 = insertelement <4 x float> %535, float %536, i64 1, !dbg !44
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi28), !dbg !40
  %539 = insertelement <4 x float> %537, float %538, i64 2, !dbg !44
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi30), !dbg !40
  %541 = insertelement <4 x float> %539, float %540, i64 3, !dbg !44
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %541) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !44
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !44
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !44
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !44
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !44
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !44
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !44
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !44
  %542 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !44
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %542) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %543 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %533, <4 x float> %511) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %544 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %545 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %544, <4 x float> %512) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi32), !dbg !40
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !44
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi34), !dbg !40
  %549 = insertelement <4 x float> %547, float %548, i64 1, !dbg !44
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi36), !dbg !40
  %551 = insertelement <4 x float> %549, float %550, i64 2, !dbg !44
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi38), !dbg !40
  %553 = insertelement <4 x float> %551, float %552, i64 3, !dbg !44
  %math.result259 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %553) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02051 = extractelement <4 x float> %math.result259, i64 0, !dbg !44
  %.cvtpack.f12052 = extractelement <4 x float> %math.result259, i64 1, !dbg !44
  %.cvtpack.f22053 = extractelement <4 x float> %math.result259, i64 2, !dbg !44
  %.cvtpack.f32054 = extractelement <4 x float> %math.result259, i64 3, !dbg !44
  %.cvtpack.p02055 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02051, float %.cvtpack.f12052) #12, !dbg !44
  %.cvtpack.p12056 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22053, float %.cvtpack.f32054) #12, !dbg !44
  %.cvtpack.v02057 = insertelement <2 x i32> poison, i32 %.cvtpack.p02055, i64 0, !dbg !44
  %.cvtpack.v12058 = insertelement <2 x i32> %.cvtpack.v02057, i32 %.cvtpack.p12056, i64 1, !dbg !44
  %554 = bitcast <2 x i32> %.cvtpack.v12058 to <4 x half>, !dbg !44
  %math.packed260 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %554) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %533, <4 x float> %514) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %544, <4 x float> %515) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %557 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %558 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi40), !dbg !40
  %559 = fadd float %534, %558, !dbg !45
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %559) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %560 = insertelement <4 x float> poison, float %558, i64 0, !dbg !44
  %561 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi42), !dbg !40
  %562 = fadd float %536, %561, !dbg !45
  %k0.sum.anchor1854 = tail call float asm sideeffect "", "=v,0"(float %562) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> %560, float %561, i64 1, !dbg !44
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi44), !dbg !40
  %565 = fadd float %538, %564, !dbg !45
  %k0.sum.anchor1855 = tail call float asm sideeffect "", "=v,0"(float %565) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %566 = insertelement <4 x float> %563, float %564, i64 2, !dbg !44
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi46), !dbg !40
  %568 = fadd float %540, %567, !dbg !45
  %k0.sum.anchor1856 = tail call float asm sideeffect "", "=v,0"(float %568) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %569 = insertelement <4 x float> %566, float %567, i64 3, !dbg !44
  %math.result245 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %569) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02059 = extractelement <4 x float> %math.result245, i64 0, !dbg !44
  %.cvtpack.f12060 = extractelement <4 x float> %math.result245, i64 1, !dbg !44
  %.cvtpack.f22061 = extractelement <4 x float> %math.result245, i64 2, !dbg !44
  %.cvtpack.f32062 = extractelement <4 x float> %math.result245, i64 3, !dbg !44
  %.cvtpack.p02063 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02059, float %.cvtpack.f12060) #12, !dbg !44
  %.cvtpack.p12064 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22061, float %.cvtpack.f32062) #12, !dbg !44
  %.cvtpack.v02065 = insertelement <2 x i32> poison, i32 %.cvtpack.p02063, i64 0, !dbg !44
  %.cvtpack.v12066 = insertelement <2 x i32> %.cvtpack.v02065, i32 %.cvtpack.p12064, i64 1, !dbg !44
  %570 = bitcast <2 x i32> %.cvtpack.v12066 to <4 x half>, !dbg !44
  %math.packed246 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %570) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %571 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %557, <4 x float> %543) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %572 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %572, <4 x float> %545) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi48), !dbg !40
  %575 = fadd float %546, %574, !dbg !45
  %k0.sum.anchor1857 = tail call float asm sideeffect "", "=v,0"(float %575) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %576 = insertelement <4 x float> poison, float %574, i64 0, !dbg !44
  %577 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi50), !dbg !40
  %578 = fadd float %548, %577, !dbg !45
  %k0.sum.anchor1858 = tail call float asm sideeffect "", "=v,0"(float %578) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %579 = insertelement <4 x float> %576, float %577, i64 1, !dbg !44
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi52), !dbg !40
  %581 = fadd float %550, %580, !dbg !45
  %k0.sum.anchor1859 = tail call float asm sideeffect "", "=v,0"(float %581) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %582 = insertelement <4 x float> %579, float %580, i64 2, !dbg !44
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi54), !dbg !40
  %584 = fadd float %552, %583, !dbg !45
  %k0.sum.anchor1860 = tail call float asm sideeffect "", "=v,0"(float %584) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %582, float %583, i64 3, !dbg !44
  %math.result261 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %585) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02067 = extractelement <4 x float> %math.result261, i64 0, !dbg !44
  %.cvtpack.f12068 = extractelement <4 x float> %math.result261, i64 1, !dbg !44
  %.cvtpack.f22069 = extractelement <4 x float> %math.result261, i64 2, !dbg !44
  %.cvtpack.f32070 = extractelement <4 x float> %math.result261, i64 3, !dbg !44
  %.cvtpack.p02071 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02067, float %.cvtpack.f12068) #12, !dbg !44
  %.cvtpack.p12072 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22069, float %.cvtpack.f32070) #12, !dbg !44
  %.cvtpack.v02073 = insertelement <2 x i32> poison, i32 %.cvtpack.p02071, i64 0, !dbg !44
  %.cvtpack.v12074 = insertelement <2 x i32> %.cvtpack.v02073, i32 %.cvtpack.p12072, i64 1, !dbg !44
  %586 = bitcast <2 x i32> %.cvtpack.v12074 to <4 x half>, !dbg !44
  %math.packed262 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %586) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %587 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %557, <4 x float> %555) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %572, <4 x float> %556) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %589 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %590 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi56), !dbg !40
  %591 = fadd float %590, %k0.sum.anchor, !dbg !45
  %k0.sum.anchor1861 = tail call float asm sideeffect "", "=v,0"(float %591) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %592 = insertelement <4 x float> poison, float %590, i64 0, !dbg !44
  %593 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi58), !dbg !40
  %594 = fadd float %593, %k0.sum.anchor1854, !dbg !45
  %k0.sum.anchor1862 = tail call float asm sideeffect "", "=v,0"(float %594) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> %592, float %593, i64 1, !dbg !44
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi60), !dbg !40
  %597 = fadd float %596, %k0.sum.anchor1855, !dbg !45
  %k0.sum.anchor1863 = tail call float asm sideeffect "", "=v,0"(float %597) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %598 = insertelement <4 x float> %595, float %596, i64 2, !dbg !44
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi62), !dbg !40
  %600 = fadd float %599, %k0.sum.anchor1856, !dbg !45
  %k0.sum.anchor1864 = tail call float asm sideeffect "", "=v,0"(float %600) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> %598, float %599, i64 3, !dbg !44
  %math.result247 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %601) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02075 = extractelement <4 x float> %math.result247, i64 0, !dbg !44
  %.cvtpack.f12076 = extractelement <4 x float> %math.result247, i64 1, !dbg !44
  %.cvtpack.f22077 = extractelement <4 x float> %math.result247, i64 2, !dbg !44
  %.cvtpack.f32078 = extractelement <4 x float> %math.result247, i64 3, !dbg !44
  %.cvtpack.p02079 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02075, float %.cvtpack.f12076) #12, !dbg !44
  %.cvtpack.p12080 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22077, float %.cvtpack.f32078) #12, !dbg !44
  %.cvtpack.v02081 = insertelement <2 x i32> poison, i32 %.cvtpack.p02079, i64 0, !dbg !44
  %.cvtpack.v12082 = insertelement <2 x i32> %.cvtpack.v02081, i32 %.cvtpack.p12080, i64 1, !dbg !44
  %602 = bitcast <2 x i32> %.cvtpack.v12082 to <4 x half>, !dbg !44
  %math.packed248 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %602) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %589, <4 x float> %571) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %604 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %604, <4 x float> %573) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi64), !dbg !40
  %607 = fadd float %606, %k0.sum.anchor1857, !dbg !45
  %k0.sum.anchor1865 = tail call float asm sideeffect "", "=v,0"(float %607) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %608 = insertelement <4 x float> poison, float %606, i64 0, !dbg !44
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi66), !dbg !40
  %610 = fadd float %609, %k0.sum.anchor1858, !dbg !45
  %k0.sum.anchor1866 = tail call float asm sideeffect "", "=v,0"(float %610) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> %608, float %609, i64 1, !dbg !44
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi68), !dbg !40
  %613 = fadd float %612, %k0.sum.anchor1859, !dbg !45
  %k0.sum.anchor1867 = tail call float asm sideeffect "", "=v,0"(float %613) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %614 = insertelement <4 x float> %611, float %612, i64 2, !dbg !44
  %615 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi70), !dbg !40
  %616 = fadd float %615, %k0.sum.anchor1860, !dbg !45
  %k0.sum.anchor1868 = tail call float asm sideeffect "", "=v,0"(float %616) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %617 = insertelement <4 x float> %614, float %615, i64 3, !dbg !44
  %math.result263 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %617) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02083 = extractelement <4 x float> %math.result263, i64 0, !dbg !44
  %.cvtpack.f12084 = extractelement <4 x float> %math.result263, i64 1, !dbg !44
  %.cvtpack.f22085 = extractelement <4 x float> %math.result263, i64 2, !dbg !44
  %.cvtpack.f32086 = extractelement <4 x float> %math.result263, i64 3, !dbg !44
  %.cvtpack.p02087 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02083, float %.cvtpack.f12084) #12, !dbg !44
  %.cvtpack.p12088 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22085, float %.cvtpack.f32086) #12, !dbg !44
  %.cvtpack.v02089 = insertelement <2 x i32> poison, i32 %.cvtpack.p02087, i64 0, !dbg !44
  %.cvtpack.v12090 = insertelement <2 x i32> %.cvtpack.v02089, i32 %.cvtpack.p12088, i64 1, !dbg !44
  %618 = bitcast <2 x i32> %.cvtpack.v12090 to <4 x half>, !dbg !44
  %math.packed264 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %618) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %619 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %589, <4 x float> %587) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %604, <4 x float> %588) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %621 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi72), !dbg !40
  %623 = fadd float %622, %k0.sum.anchor1861, !dbg !45
  %k0.sum.anchor1869 = tail call float asm sideeffect "", "=v,0"(float %623) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %624 = insertelement <4 x float> poison, float %622, i64 0, !dbg !44
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi74), !dbg !40
  %626 = fadd float %625, %k0.sum.anchor1862, !dbg !45
  %k0.sum.anchor1870 = tail call float asm sideeffect "", "=v,0"(float %626) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %627 = insertelement <4 x float> %624, float %625, i64 1, !dbg !44
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi76), !dbg !40
  %629 = fadd float %628, %k0.sum.anchor1863, !dbg !45
  %k0.sum.anchor1871 = tail call float asm sideeffect "", "=v,0"(float %629) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %630 = insertelement <4 x float> %627, float %628, i64 2, !dbg !44
  %631 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi78), !dbg !40
  %632 = fadd float %631, %k0.sum.anchor1864, !dbg !45
  %k0.sum.anchor1872 = tail call float asm sideeffect "", "=v,0"(float %632) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %630, float %631, i64 3, !dbg !44
  %math.result249 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %633) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02091 = extractelement <4 x float> %math.result249, i64 0, !dbg !44
  %.cvtpack.f12092 = extractelement <4 x float> %math.result249, i64 1, !dbg !44
  %.cvtpack.f22093 = extractelement <4 x float> %math.result249, i64 2, !dbg !44
  %.cvtpack.f32094 = extractelement <4 x float> %math.result249, i64 3, !dbg !44
  %.cvtpack.p02095 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02091, float %.cvtpack.f12092) #12, !dbg !44
  %.cvtpack.p12096 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22093, float %.cvtpack.f32094) #12, !dbg !44
  %.cvtpack.v02097 = insertelement <2 x i32> poison, i32 %.cvtpack.p02095, i64 0, !dbg !44
  %.cvtpack.v12098 = insertelement <2 x i32> %.cvtpack.v02097, i32 %.cvtpack.p12096, i64 1, !dbg !44
  %634 = bitcast <2 x i32> %.cvtpack.v12098 to <4 x half>, !dbg !44
  %math.packed250 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %634) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %621, <4 x float> %603) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %636 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %636, <4 x float> %605) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %638 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi80), !dbg !40
  %639 = fadd float %638, %k0.sum.anchor1865, !dbg !45
  %k0.sum.anchor1873 = tail call float asm sideeffect "", "=v,0"(float %639) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %640 = insertelement <4 x float> poison, float %638, i64 0, !dbg !44
  %641 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi82), !dbg !40
  %642 = fadd float %641, %k0.sum.anchor1866, !dbg !45
  %k0.sum.anchor1874 = tail call float asm sideeffect "", "=v,0"(float %642) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %643 = insertelement <4 x float> %640, float %641, i64 1, !dbg !44
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi84), !dbg !40
  %645 = fadd float %644, %k0.sum.anchor1867, !dbg !45
  %k0.sum.anchor1875 = tail call float asm sideeffect "", "=v,0"(float %645) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> %643, float %644, i64 2, !dbg !44
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi86), !dbg !40
  %648 = fadd float %647, %k0.sum.anchor1868, !dbg !45
  %k0.sum.anchor1876 = tail call float asm sideeffect "", "=v,0"(float %648) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %649 = insertelement <4 x float> %646, float %647, i64 3, !dbg !44
  %math.result265 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02099 = extractelement <4 x float> %math.result265, i64 0, !dbg !44
  %.cvtpack.f12100 = extractelement <4 x float> %math.result265, i64 1, !dbg !44
  %.cvtpack.f22101 = extractelement <4 x float> %math.result265, i64 2, !dbg !44
  %.cvtpack.f32102 = extractelement <4 x float> %math.result265, i64 3, !dbg !44
  %.cvtpack.p02103 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02099, float %.cvtpack.f12100) #12, !dbg !44
  %.cvtpack.p12104 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22101, float %.cvtpack.f32102) #12, !dbg !44
  %.cvtpack.v02105 = insertelement <2 x i32> poison, i32 %.cvtpack.p02103, i64 0, !dbg !44
  %.cvtpack.v12106 = insertelement <2 x i32> %.cvtpack.v02105, i32 %.cvtpack.p12104, i64 1, !dbg !44
  %650 = bitcast <2 x i32> %.cvtpack.v12106 to <4 x half>, !dbg !44
  %math.packed266 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %621, <4 x float> %619) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %652 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %636, <4 x float> %620) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %653 = shufflevector <8 x half> %527, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %654 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi88), !dbg !40
  %655 = fadd float %654, %k0.sum.anchor1869, !dbg !45
  %k0.sum.anchor1877 = tail call float asm sideeffect "", "=v,0"(float %655) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %656 = insertelement <4 x float> poison, float %654, i64 0, !dbg !44
  %657 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi90), !dbg !40
  %658 = fadd float %657, %k0.sum.anchor1870, !dbg !45
  %k0.sum.anchor1878 = tail call float asm sideeffect "", "=v,0"(float %658) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %659 = insertelement <4 x float> %656, float %657, i64 1, !dbg !44
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi92), !dbg !40
  %661 = fadd float %660, %k0.sum.anchor1871, !dbg !45
  %k0.sum.anchor1879 = tail call float asm sideeffect "", "=v,0"(float %661) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %659, float %660, i64 2, !dbg !44
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi94), !dbg !40
  %664 = fadd float %663, %k0.sum.anchor1872, !dbg !45
  %k0.sum.anchor1880 = tail call float asm sideeffect "", "=v,0"(float %664) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %665 = insertelement <4 x float> %662, float %663, i64 3, !dbg !44
  %math.result251 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %665) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02107 = extractelement <4 x float> %math.result251, i64 0, !dbg !44
  %.cvtpack.f12108 = extractelement <4 x float> %math.result251, i64 1, !dbg !44
  %.cvtpack.f22109 = extractelement <4 x float> %math.result251, i64 2, !dbg !44
  %.cvtpack.f32110 = extractelement <4 x float> %math.result251, i64 3, !dbg !44
  %.cvtpack.p02111 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02107, float %.cvtpack.f12108) #12, !dbg !44
  %.cvtpack.p12112 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22109, float %.cvtpack.f32110) #12, !dbg !44
  %.cvtpack.v02113 = insertelement <2 x i32> poison, i32 %.cvtpack.p02111, i64 0, !dbg !44
  %.cvtpack.v12114 = insertelement <2 x i32> %.cvtpack.v02113, i32 %.cvtpack.p12112, i64 1, !dbg !44
  %666 = bitcast <2 x i32> %.cvtpack.v12114 to <4 x half>, !dbg !44
  %math.packed252 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %666) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %667 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %653, <4 x float> %635) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %668 = shufflevector <8 x half> %527, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %668, <4 x float> %637) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %670 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi96), !dbg !40
  %671 = fadd float %670, %k0.sum.anchor1873, !dbg !45
  %k0.sum.anchor1881 = tail call float asm sideeffect "", "=v,0"(float %671) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %672 = insertelement <4 x float> poison, float %670, i64 0, !dbg !44
  %673 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi98), !dbg !40
  %674 = fadd float %673, %k0.sum.anchor1874, !dbg !45
  %k0.sum.anchor1882 = tail call float asm sideeffect "", "=v,0"(float %674) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %675 = insertelement <4 x float> %672, float %673, i64 1, !dbg !44
  %676 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi100), !dbg !40
  %677 = fadd float %676, %k0.sum.anchor1875, !dbg !45
  %k0.sum.anchor1883 = tail call float asm sideeffect "", "=v,0"(float %677) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %678 = insertelement <4 x float> %675, float %676, i64 2, !dbg !44
  %679 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi102), !dbg !40
  %680 = fadd float %679, %k0.sum.anchor1876, !dbg !45
  %k0.sum.anchor1884 = tail call float asm sideeffect "", "=v,0"(float %680) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %681 = insertelement <4 x float> %678, float %679, i64 3, !dbg !44
  %math.result267 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %681) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02115 = extractelement <4 x float> %math.result267, i64 0, !dbg !44
  %.cvtpack.f12116 = extractelement <4 x float> %math.result267, i64 1, !dbg !44
  %.cvtpack.f22117 = extractelement <4 x float> %math.result267, i64 2, !dbg !44
  %.cvtpack.f32118 = extractelement <4 x float> %math.result267, i64 3, !dbg !44
  %.cvtpack.p02119 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02115, float %.cvtpack.f12116) #12, !dbg !44
  %.cvtpack.p12120 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22117, float %.cvtpack.f32118) #12, !dbg !44
  %.cvtpack.v02121 = insertelement <2 x i32> poison, i32 %.cvtpack.p02119, i64 0, !dbg !44
  %.cvtpack.v12122 = insertelement <2 x i32> %.cvtpack.v02121, i32 %.cvtpack.p12120, i64 1, !dbg !44
  %682 = bitcast <2 x i32> %.cvtpack.v12122 to <4 x half>, !dbg !44
  %math.packed268 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %682) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %653, <4 x float> %651) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %668, <4 x float> %652) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %685 = shufflevector <8 x half> %529, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi104), !dbg !40
  %687 = fadd float %686, %k0.sum.anchor1877, !dbg !45
  %k0.sum.anchor1885 = tail call float asm sideeffect "", "=v,0"(float %687) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %688 = insertelement <4 x float> poison, float %686, i64 0, !dbg !44
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi106), !dbg !40
  %690 = fadd float %689, %k0.sum.anchor1878, !dbg !45
  %k0.sum.anchor1886 = tail call float asm sideeffect "", "=v,0"(float %690) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %691 = insertelement <4 x float> %688, float %689, i64 1, !dbg !44
  %692 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi108), !dbg !40
  %693 = fadd float %692, %k0.sum.anchor1879, !dbg !45
  %k0.sum.anchor1887 = tail call float asm sideeffect "", "=v,0"(float %693) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %694 = insertelement <4 x float> %691, float %692, i64 2, !dbg !44
  %695 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi110), !dbg !40
  %696 = fadd float %695, %k0.sum.anchor1880, !dbg !45
  %k0.sum.anchor1888 = tail call float asm sideeffect "", "=v,0"(float %696) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %697 = insertelement <4 x float> %694, float %695, i64 3, !dbg !44
  %math.result253 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %697) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02123 = extractelement <4 x float> %math.result253, i64 0, !dbg !44
  %.cvtpack.f12124 = extractelement <4 x float> %math.result253, i64 1, !dbg !44
  %.cvtpack.f22125 = extractelement <4 x float> %math.result253, i64 2, !dbg !44
  %.cvtpack.f32126 = extractelement <4 x float> %math.result253, i64 3, !dbg !44
  %.cvtpack.p02127 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02123, float %.cvtpack.f12124) #12, !dbg !44
  %.cvtpack.p12128 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22125, float %.cvtpack.f32126) #12, !dbg !44
  %.cvtpack.v02129 = insertelement <2 x i32> poison, i32 %.cvtpack.p02127, i64 0, !dbg !44
  %.cvtpack.v12130 = insertelement <2 x i32> %.cvtpack.v02129, i32 %.cvtpack.p12128, i64 1, !dbg !44
  %698 = bitcast <2 x i32> %.cvtpack.v12130 to <4 x half>, !dbg !44
  %math.packed254 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %698) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %699 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %685, <4 x float> %667) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %700 = shufflevector <8 x half> %529, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %700, <4 x float> %669) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %702 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi112), !dbg !40
  %703 = fadd float %702, %k0.sum.anchor1881, !dbg !45
  %k0.sum.anchor1889 = tail call float asm sideeffect "", "=v,0"(float %703) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %704 = insertelement <4 x float> poison, float %702, i64 0, !dbg !44
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi114), !dbg !40
  %706 = fadd float %705, %k0.sum.anchor1882, !dbg !45
  %k0.sum.anchor1890 = tail call float asm sideeffect "", "=v,0"(float %706) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %707 = insertelement <4 x float> %704, float %705, i64 1, !dbg !44
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi116), !dbg !40
  %709 = fadd float %708, %k0.sum.anchor1883, !dbg !45
  %k0.sum.anchor1891 = tail call float asm sideeffect "", "=v,0"(float %709) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %710 = insertelement <4 x float> %707, float %708, i64 2, !dbg !44
  %711 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi118), !dbg !40
  %712 = fadd float %711, %k0.sum.anchor1884, !dbg !45
  %k0.sum.anchor1892 = tail call float asm sideeffect "", "=v,0"(float %712) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %713 = insertelement <4 x float> %710, float %711, i64 3, !dbg !44
  %math.result269 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %713) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02131 = extractelement <4 x float> %math.result269, i64 0, !dbg !44
  %.cvtpack.f12132 = extractelement <4 x float> %math.result269, i64 1, !dbg !44
  %.cvtpack.f22133 = extractelement <4 x float> %math.result269, i64 2, !dbg !44
  %.cvtpack.f32134 = extractelement <4 x float> %math.result269, i64 3, !dbg !44
  %.cvtpack.p02135 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02131, float %.cvtpack.f12132) #12, !dbg !44
  %.cvtpack.p12136 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22133, float %.cvtpack.f32134) #12, !dbg !44
  %.cvtpack.v02137 = insertelement <2 x i32> poison, i32 %.cvtpack.p02135, i64 0, !dbg !44
  %.cvtpack.v12138 = insertelement <2 x i32> %.cvtpack.v02137, i32 %.cvtpack.p12136, i64 1, !dbg !44
  %714 = bitcast <2 x i32> %.cvtpack.v12138 to <4 x half>, !dbg !44
  %math.packed270 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %714) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %715 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %685, <4 x float> %683) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %700, <4 x float> %684) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %717 = shufflevector <8 x half> %531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %718 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi120), !dbg !40
  %719 = fadd float %718, %k0.sum.anchor1885, !dbg !45
  %k0.sum.anchor1893 = tail call float asm sideeffect "", "=v,0"(float %719) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %720 = insertelement <4 x float> poison, float %718, i64 0, !dbg !44
  %721 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi122), !dbg !40
  %722 = fadd float %721, %k0.sum.anchor1886, !dbg !45
  %k0.sum.anchor1894 = tail call float asm sideeffect "", "=v,0"(float %722) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %723 = insertelement <4 x float> %720, float %721, i64 1, !dbg !44
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi124), !dbg !40
  %725 = fadd float %724, %k0.sum.anchor1887, !dbg !45
  %k0.sum.anchor1895 = tail call float asm sideeffect "", "=v,0"(float %725) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %726 = insertelement <4 x float> %723, float %724, i64 2, !dbg !44
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi126), !dbg !40
  %728 = fadd float %727, %k0.sum.anchor1888, !dbg !45
  %k0.sum.anchor1896 = tail call float asm sideeffect "", "=v,0"(float %728) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %729 = insertelement <4 x float> %726, float %727, i64 3, !dbg !44
  %math.result255 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02139 = extractelement <4 x float> %math.result255, i64 0, !dbg !44
  %.cvtpack.f12140 = extractelement <4 x float> %math.result255, i64 1, !dbg !44
  %.cvtpack.f22141 = extractelement <4 x float> %math.result255, i64 2, !dbg !44
  %.cvtpack.f32142 = extractelement <4 x float> %math.result255, i64 3, !dbg !44
  %.cvtpack.p02143 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02139, float %.cvtpack.f12140) #12, !dbg !44
  %.cvtpack.p12144 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22141, float %.cvtpack.f32142) #12, !dbg !44
  %.cvtpack.v02145 = insertelement <2 x i32> poison, i32 %.cvtpack.p02143, i64 0, !dbg !44
  %.cvtpack.v12146 = insertelement <2 x i32> %.cvtpack.v02145, i32 %.cvtpack.p12144, i64 1, !dbg !44
  %730 = bitcast <2 x i32> %.cvtpack.v12146 to <4 x half>, !dbg !44
  %math.packed256 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %717, <4 x float> %699) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %732 = shufflevector <8 x half> %531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %733 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %732, <4 x float> %701) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %734 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi128), !dbg !40
  %735 = fadd float %734, %k0.sum.anchor1889, !dbg !45
  %k0.sum.anchor1897 = tail call float asm sideeffect "", "=v,0"(float %735) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %736 = insertelement <4 x float> poison, float %734, i64 0, !dbg !44
  %737 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi130), !dbg !40
  %738 = fadd float %737, %k0.sum.anchor1890, !dbg !45
  %k0.sum.anchor1898 = tail call float asm sideeffect "", "=v,0"(float %738) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %739 = insertelement <4 x float> %736, float %737, i64 1, !dbg !44
  %740 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi132), !dbg !40
  %741 = fadd float %740, %k0.sum.anchor1891, !dbg !45
  %k0.sum.anchor1899 = tail call float asm sideeffect "", "=v,0"(float %741) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %742 = insertelement <4 x float> %739, float %740, i64 2, !dbg !44
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi134), !dbg !40
  %744 = fadd float %743, %k0.sum.anchor1892, !dbg !45
  %k0.sum.anchor1900 = tail call float asm sideeffect "", "=v,0"(float %744) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %745 = insertelement <4 x float> %742, float %743, i64 3, !dbg !44
  %math.result271 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %745) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02147 = extractelement <4 x float> %math.result271, i64 0, !dbg !44
  %.cvtpack.f12148 = extractelement <4 x float> %math.result271, i64 1, !dbg !44
  %.cvtpack.f22149 = extractelement <4 x float> %math.result271, i64 2, !dbg !44
  %.cvtpack.f32150 = extractelement <4 x float> %math.result271, i64 3, !dbg !44
  %.cvtpack.p02151 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02147, float %.cvtpack.f12148) #12, !dbg !44
  %.cvtpack.p12152 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22149, float %.cvtpack.f32150) #12, !dbg !44
  %.cvtpack.v02153 = insertelement <2 x i32> poison, i32 %.cvtpack.p02151, i64 0, !dbg !44
  %.cvtpack.v12154 = insertelement <2 x i32> %.cvtpack.v02153, i32 %.cvtpack.p12152, i64 1, !dbg !44
  %746 = bitcast <2 x i32> %.cvtpack.v12154 to <4 x half>, !dbg !44
  %math.packed272 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %746) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %717, <4 x float> %715) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %732, <4 x float> %716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %749 = bitcast <4 x i32> %532 to <8 x half>, !dbg !42
  %750 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi136), !dbg !40
  %752 = fadd float %751, %k0.sum.anchor1893, !dbg !45
  %k0.sum.anchor1901 = tail call float asm sideeffect "", "=v,0"(float %752) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %753 = insertelement <4 x float> poison, float %751, i64 0, !dbg !44
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi138), !dbg !40
  %755 = fadd float %754, %k0.sum.anchor1894, !dbg !45
  %k0.sum.anchor1902 = tail call float asm sideeffect "", "=v,0"(float %755) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %756 = fadd float %k0.sum.anchor1901, %k0.sum.anchor1902, !dbg !45
  %k0.sum.anchor1903 = tail call float asm sideeffect "", "=v,0"(float %756) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %757 = insertelement <4 x float> %753, float %754, i64 1, !dbg !44
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi140), !dbg !40
  %759 = fadd float %758, %k0.sum.anchor1895, !dbg !45
  %k0.sum.anchor1904 = tail call float asm sideeffect "", "=v,0"(float %759) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %760 = insertelement <4 x float> %757, float %758, i64 2, !dbg !44
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi142), !dbg !40
  %762 = fadd float %761, %k0.sum.anchor1896, !dbg !45
  %k0.sum.anchor1905 = tail call float asm sideeffect "", "=v,0"(float %762) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %763 = fadd float %k0.sum.anchor1904, %k0.sum.anchor1905, !dbg !45
  %k0.sum.anchor1906 = tail call float asm sideeffect "", "=v,0"(float %763) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %764 = fadd float %k0.sum.anchor1903, %k0.sum.anchor1906, !dbg !45
  %k0.sum.anchor1907 = tail call float asm sideeffect "", "=v,0"(float %764) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %765 = fadd float %507, %k0.sum.anchor1907, !dbg !45
  %k0.sum.anchor1908 = tail call float asm sideeffect "", "=v,0"(float %765) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %766 = insertelement <1 x float> poison, float %k0.sum.anchor1908, i64 0, !dbg !12
  %767 = insertvalue [2 x <1 x float>] poison, <1 x float> %766, 0, !dbg !12
  %768 = insertelement <4 x float> %760, float %761, i64 3, !dbg !44
  %math.result257 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %768) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02155 = extractelement <4 x float> %math.result257, i64 0, !dbg !44
  %.cvtpack.f12156 = extractelement <4 x float> %math.result257, i64 1, !dbg !44
  %.cvtpack.f22157 = extractelement <4 x float> %math.result257, i64 2, !dbg !44
  %.cvtpack.f32158 = extractelement <4 x float> %math.result257, i64 3, !dbg !44
  %.cvtpack.p02159 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02155, float %.cvtpack.f12156) #12, !dbg !44
  %.cvtpack.p12160 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22157, float %.cvtpack.f32158) #12, !dbg !44
  %.cvtpack.v02161 = insertelement <2 x i32> poison, i32 %.cvtpack.p02159, i64 0, !dbg !44
  %.cvtpack.v12162 = insertelement <2 x i32> %.cvtpack.v02161, i32 %.cvtpack.p12160, i64 1, !dbg !44
  %769 = bitcast <2 x i32> %.cvtpack.v12162 to <4 x half>, !dbg !44
  %math.packed258 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %769) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %750, <4 x float> %731) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %771 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %771, <4 x float> %733) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi144), !dbg !40
  %774 = fadd float %773, %k0.sum.anchor1897, !dbg !45
  %k0.sum.anchor1909 = tail call float asm sideeffect "", "=v,0"(float %774) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %775 = insertelement <4 x float> poison, float %773, i64 0, !dbg !44
  %776 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi146), !dbg !40
  %777 = fadd float %776, %k0.sum.anchor1898, !dbg !45
  %k0.sum.anchor1910 = tail call float asm sideeffect "", "=v,0"(float %777) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %778 = fadd float %k0.sum.anchor1909, %k0.sum.anchor1910, !dbg !45
  %k0.sum.anchor1911 = tail call float asm sideeffect "", "=v,0"(float %778) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %779 = insertelement <4 x float> %775, float %776, i64 1, !dbg !44
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi148), !dbg !40
  %781 = fadd float %780, %k0.sum.anchor1899, !dbg !45
  %k0.sum.anchor1912 = tail call float asm sideeffect "", "=v,0"(float %781) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %782 = insertelement <4 x float> %779, float %780, i64 2, !dbg !44
  %783 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi150), !dbg !40
  %784 = fadd float %783, %k0.sum.anchor1900, !dbg !45
  %k0.sum.anchor1913 = tail call float asm sideeffect "", "=v,0"(float %784) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %785 = fadd float %k0.sum.anchor1912, %k0.sum.anchor1913, !dbg !45
  %k0.sum.anchor1914 = tail call float asm sideeffect "", "=v,0"(float %785) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %786 = fadd float %k0.sum.anchor1911, %k0.sum.anchor1914, !dbg !45
  %k0.sum.anchor1915 = tail call float asm sideeffect "", "=v,0"(float %786) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %787 = fadd float %509, %k0.sum.anchor1915, !dbg !45
  %k0.sum.anchor1916 = tail call float asm sideeffect "", "=v,0"(float %787) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %788 = insertelement <1 x float> poison, float %k0.sum.anchor1916, i64 0, !dbg !12
  %789 = insertvalue [2 x <1 x float>] %767, <1 x float> %788, 1, !dbg !12
  %790 = insertelement <4 x float> %782, float %783, i64 3, !dbg !44
  %math.result273 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %790) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02163 = extractelement <4 x float> %math.result273, i64 0, !dbg !44
  %.cvtpack.f12164 = extractelement <4 x float> %math.result273, i64 1, !dbg !44
  %.cvtpack.f22165 = extractelement <4 x float> %math.result273, i64 2, !dbg !44
  %.cvtpack.f32166 = extractelement <4 x float> %math.result273, i64 3, !dbg !44
  %.cvtpack.p02167 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02163, float %.cvtpack.f12164) #12, !dbg !44
  %.cvtpack.p12168 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22165, float %.cvtpack.f32166) #12, !dbg !44
  %.cvtpack.v02169 = insertelement <2 x i32> poison, i32 %.cvtpack.p02167, i64 0, !dbg !44
  %.cvtpack.v12170 = insertelement <2 x i32> %.cvtpack.v02169, i32 %.cvtpack.p12168, i64 1, !dbg !44
  %791 = bitcast <2 x i32> %.cvtpack.v12170 to <4 x half>, !dbg !44
  %math.packed274 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %791) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %750, <4 x float> %747) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %771, <4 x float> %748) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %794 = extractvalue [2 x <8 x float>] %lazy.state2049, 0, !dbg !41
  %795 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %796 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %797 = extractvalue [2 x <8 x float>] %lazy.state2049, 1, !dbg !41
  %798 = shufflevector <8 x float> %797, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %799 = shufflevector <8 x float> %797, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1318 = add nsw i64 %address.local793, %205, !dbg !38
  %vdb.address1319 = trunc nsw i64 %vdb.address1318 to i32, !dbg !38
  %vdb.address1320 = shl nsw i32 %vdb.address1319, 1, !dbg !38
  %vdb.address1325 = trunc nuw nsw i64 %205 to i32, !dbg !38
  %vdb.address1326 = or disjoint i32 %address.local800, %vdb.address1325, !dbg !38
  %vdb.address1327 = shl nuw nsw i32 %vdb.address1326, 1, !dbg !38
  %800 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1320, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %801 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %vdb.address1327, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local809, i64 8192, !dbg !42
  %802 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %803 = bitcast <4 x i32> %802 to <8 x half>, !dbg !42
  %vdb.read1311 = getelementptr i8, ptr addrspace(3) %address.local818, i64 8448, !dbg !42
  %804 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1311, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %805 = bitcast <4 x i32> %804 to <8 x half>, !dbg !42
  %vdb.read1312 = getelementptr i8, ptr addrspace(3) %address.local818, i64 8704, !dbg !42
  %806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1312, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %807 = bitcast <4 x i32> %806 to <8 x half>, !dbg !42
  %vdb.read1313 = getelementptr i8, ptr addrspace(3) %address.local818, i64 8960, !dbg !42
  %808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1313, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !42
  %vdb.read1314 = getelementptr i8, ptr addrspace(3) %address.local809, i64 12288, !dbg !42
  %810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1314, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !42
  %vdb.read1315 = getelementptr i8, ptr addrspace(3) %address.local818, i64 12544, !dbg !42
  %812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1315, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !42
  %vdb.read1316 = getelementptr i8, ptr addrspace(3) %address.local818, i64 12800, !dbg !42
  %814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1316, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !42
  %vdb.read1317 = getelementptr i8, ptr addrspace(3) %address.local818, i64 13056, !dbg !42
  %816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1317, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %817 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %817, <4 x float> %795) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %819 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %819, <4 x float> %796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %817, <4 x float> %798) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %819, <4 x float> %799) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %823 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %823, <4 x float> %818) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %825 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %825, <4 x float> %820) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %827 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %823, <4 x float> %821) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %825, <4 x float> %822) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %829 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %829, <4 x float> %824) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %831 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %831, <4 x float> %826) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %833 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %829, <4 x float> %827) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %831, <4 x float> %828) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %835 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %835, <4 x float> %830) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %837 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %837, <4 x float> %832) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %835, <4 x float> %833) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %837, <4 x float> %834) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %841 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %841, <4 x float> %836) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %843 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %843, <4 x float> %838) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %841, <4 x float> %839) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %843, <4 x float> %840) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %847 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %847, <4 x float> %842) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %849 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %849, <4 x float> %844) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %847, <4 x float> %845) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %849, <4 x float> %846) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %853 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %853, <4 x float> %848) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %855 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %855, <4 x float> %850) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %853, <4 x float> %851) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %855, <4 x float> %852) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %859 = bitcast <4 x i32> %816 to <8 x half>, !dbg !42
  %860 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %860, <4 x float> %854) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %862 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %862, <4 x float> %856) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %860, <4 x float> %857) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %862, <4 x float> %858) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %866 = extractvalue [2 x <8 x float>] %lazy.state2048, 0, !dbg !41
  %867 = shufflevector <8 x float> %866, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %868 = shufflevector <8 x float> %866, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %869 = extractvalue [2 x <8 x float>] %lazy.state2048, 1, !dbg !41
  %870 = shufflevector <8 x float> %869, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %871 = shufflevector <8 x float> %869, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1332 = add nsw i64 %address.local793, %258, !dbg !38
  %vdb.address1333 = trunc nsw i64 %vdb.address1332 to i32, !dbg !38
  %vdb.address1334 = shl nsw i32 %vdb.address1333, 1, !dbg !38
  %vdb.address1339 = trunc nuw nsw i64 %258 to i32, !dbg !38
  %vdb.address1340 = or disjoint i32 %address.local800, %vdb.address1339, !dbg !38
  %vdb.address1341 = shl nuw nsw i32 %vdb.address1340, 1, !dbg !38
  %872 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1334, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %873 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1310, i32 %vdb.address1341, <4 x i32> %dma.buffer.rsrc1217) #9, !dbg !38
  %874 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local809, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !42
  %876 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !42
  %878 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !42
  %880 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local827, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !42
  %882 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local829, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !42
  %884 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local833, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !42
  %886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local837, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !42
  %888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %889 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %889, <4 x float> %867) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %891 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %891, <4 x float> %868) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %889, <4 x float> %870) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %891, <4 x float> %871) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %895 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %895, <4 x float> %890) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %897 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %897, <4 x float> %892) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %895, <4 x float> %893) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %897, <4 x float> %894) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %901 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %901, <4 x float> %896) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %903 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %903, <4 x float> %898) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %901, <4 x float> %899) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %903, <4 x float> %900) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %907 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %907, <4 x float> %902) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %909 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %909, <4 x float> %904) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %907, <4 x float> %905) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %909, <4 x float> %906) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %913 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %913, <4 x float> %908) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %915 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %915, <4 x float> %910) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %913, <4 x float> %911) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %915, <4 x float> %912) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %919 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %919, <4 x float> %914) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %921 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %921, <4 x float> %916) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %919, <4 x float> %917) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %921, <4 x float> %918) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %925 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %925, <4 x float> %920) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %927 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %927, <4 x float> %922) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %925, <4 x float> %923) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %927, <4 x float> %924) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %931 = bitcast <4 x i32> %888 to <8 x half>, !dbg !42
  %932 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %932, <4 x float> %926) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %934 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %934, <4 x float> %928) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %932, <4 x float> %929) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %934, <4 x float> %930) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %938 = extractvalue [2 x <8 x float>] %lazy.state2047, 0, !dbg !41
  %939 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %940 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %941 = extractvalue [2 x <8 x float>] %lazy.state2047, 1, !dbg !41
  %942 = shufflevector <8 x float> %941, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %943 = shufflevector <8 x float> %941, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %944 = add nuw nsw i64 %124, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %124, 31, !dbg !42
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !42

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %944, 14, !dbg !14
  %k0.next.1927 = add nuw nsw i64 %k0.next., %k0.prime.1917, !dbg !14
  %k0.next.kdma.source1372 = or disjoint i64 %k0.next.1927, %k0.prime.kdma.source1371, !dbg !14
  %k0.next.kdma.source1373 = trunc nuw nsw i64 %k0.next.kdma.source1372 to i32, !dbg !14
  %k0.next.kdma.source1374 = shl nuw nsw i32 %k0.next.kdma.source1373, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1374, %kdma.intra, !dbg !14
  %k0.next.1928 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1392 = or disjoint i64 %k0.next.1927, %k0.prime.kdma.source1391, !dbg !14
  %k0.next.kdma.source1393 = trunc nuw nsw i64 %k0.next.kdma.source1392 to i32, !dbg !14
  %k0.next.kdma.source1394 = shl nuw nsw i32 %k0.next.kdma.source1393, 1, !dbg !14
  %k0.next.kdma.offset1395 = or disjoint i32 %k0.next.kdma.source1394, %kdma.intra.hi, !dbg !14
  %k0.next.1929 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1396, i32 %k0.next.kdma.offset1395, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1416 = or disjoint i64 %k0.next.1927, %k0.prime.kdma.source1415, !dbg !14
  %k0.next.kdma.source1417 = trunc nuw nsw i64 %k0.next.kdma.source1416 to i32, !dbg !14
  %k0.next.kdma.source1418 = shl nuw nsw i32 %k0.next.kdma.source1417, 1, !dbg !14
  %k0.next.kdma.offset1419 = add nuw nsw i32 %k0.next.kdma.source1418, %kdma.intra, !dbg !14
  %k0.next.1930 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.next.kdma.offset1419, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1440 = or disjoint i64 %k0.next.1927, %k0.prime.kdma.source1439, !dbg !14
  %k0.next.kdma.source1441 = trunc nuw nsw i64 %k0.next.kdma.source1440 to i32, !dbg !14
  %k0.next.kdma.source1442 = shl nuw nsw i32 %k0.next.kdma.source1441, 1, !dbg !14
  %k0.next.kdma.offset1443 = add nuw nsw i32 %k0.next.kdma.source1442, %kdma.intra.hi, !dbg !14
  %k0.next.1931 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1444, i32 %k0.next.kdma.offset1443, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %945 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !42
  %947 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1311, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !42
  %949 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1312, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %950 = bitcast <4 x i32> %949 to <8 x half>, !dbg !42
  %951 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1313, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %952 = bitcast <4 x i32> %951 to <8 x half>, !dbg !42
  %953 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1314, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %954 = bitcast <4 x i32> %953 to <8 x half>, !dbg !42
  %955 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1315, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %956 = bitcast <4 x i32> %955 to <8 x half>, !dbg !42
  %957 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1316, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %958 = bitcast <4 x i32> %957 to <8 x half>, !dbg !42
  %959 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1317, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %960 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %960, <4 x float> %939) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %962 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %962, <4 x float> %940) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %960, <4 x float> %942) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %962, <4 x float> %943) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %966 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %966, <4 x float> %961) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %968 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %968, <4 x float> %963) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %966, <4 x float> %964) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %968, <4 x float> %965) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %972 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %972, <4 x float> %967) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %974 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %974, <4 x float> %969) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %972, <4 x float> %970) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %974, <4 x float> %971) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %978 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %978, <4 x float> %973) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %980 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %980, <4 x float> %975) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %978, <4 x float> %976) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %980, <4 x float> %977) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %984 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %984, <4 x float> %979) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %986 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %986, <4 x float> %981) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %984, <4 x float> %982) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %986, <4 x float> %983) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %990 = shufflevector <8 x half> %956, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %990, <4 x float> %985) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %992 = shufflevector <8 x half> %956, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %992, <4 x float> %987) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %990, <4 x float> %988) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %992, <4 x float> %989) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %996 = shufflevector <8 x half> %958, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %996, <4 x float> %991) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %998 = shufflevector <8 x half> %958, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %998, <4 x float> %993) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %996, <4 x float> %994) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %998, <4 x float> %995) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1002 = bitcast <4 x i32> %959 to <8 x half>, !dbg !42
  %1003 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1003, <4 x float> %997) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1005 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1005, <4 x float> %999) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1003, <4 x float> %1000) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1005, <4 x float> %1001) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1009 = shufflevector <4 x float> %770, <4 x float> %772, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1010 = insertvalue [2 x <8 x float>] poison, <8 x float> %1009, 0, !dbg !12
  %1011 = shufflevector <4 x float> %792, <4 x float> %793, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1012 = insertvalue [2 x <8 x float>] %1010, <8 x float> %1011, 1, !dbg !12
  %1013 = shufflevector <4 x float> %861, <4 x float> %863, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1014 = insertvalue [2 x <8 x float>] poison, <8 x float> %1013, 0, !dbg !12
  %1015 = shufflevector <4 x float> %864, <4 x float> %865, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1016 = insertvalue [2 x <8 x float>] %1014, <8 x float> %1015, 1, !dbg !12
  %1017 = shufflevector <4 x float> %933, <4 x float> %935, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1018 = insertvalue [2 x <8 x float>] poison, <8 x float> %1017, 0, !dbg !12
  %1019 = shufflevector <4 x float> %936, <4 x float> %937, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1020 = insertvalue [2 x <8 x float>] %1018, <8 x float> %1019, 1, !dbg !12
  %1021 = shufflevector <4 x float> %1004, <4 x float> %1006, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1022 = insertvalue [2 x <8 x float>] poison, <8 x float> %1021, 0, !dbg !12
  %1023 = shufflevector <4 x float> %1007, <4 x float> %1008, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1024 = insertvalue [2 x <8 x float>] %1022, <8 x float> %1023, 1, !dbg !12
  %1025 = icmp samesign ult i64 %124, 31, !dbg !12
  br i1 %1025, label %117, label %1026, !dbg !12

1026:                                             ; preds = %k0.join
  %1027 = shl nuw nsw i32 %8, 2, !dbg !12
  %1028 = xor i32 %1027, 64, !dbg !12
  %1029 = bitcast float %k0.sum.anchor1916 to i32, !dbg !12
  %lane.sum.shuffle47 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1029), !dbg !12
  %1030 = bitcast i32 %lane.sum.shuffle47 to float, !dbg !12
  %lane.sum.final48 = fadd float %k0.sum.anchor1916, %1030, !dbg !12
  %1031 = xor i32 %1027, 128, !dbg !12
  %1032 = bitcast float %lane.sum.final48 to i32, !dbg !12
  %lane.sum.shuffle49 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1031, i32 %1032), !dbg !12
  %1033 = bitcast i32 %lane.sum.shuffle49 to float, !dbg !12
  %lane.sum.final50 = fadd float %lane.sum.final48, %1033, !dbg !12
  %1034 = insertelement <1 x float> poison, float %lane.sum.final50, i64 0, !dbg !12
  %1035 = bitcast float %k0.sum.anchor1908 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1028, i32 %1035), !dbg !12
  %1036 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor1908, %1036, !dbg !12
  %1037 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle41 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1031, i32 %1037), !dbg !12
  %1038 = bitcast i32 %lane.sum.shuffle41 to float, !dbg !12
  %lane.sum.final42 = fadd float %lane.sum.final, %1038, !dbg !12
  %1039 = insertelement <1 x float> poison, float %lane.sum.final42, i64 0, !dbg !12
  %1040 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1040, 0, !dbg !12
  %1041 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %1041, 0, !dbg !12
  %1042 = extractelement <4 x float> %1004, i64 0, !dbg !12
  %1043 = extractelement <4 x float> %1004, i64 1, !dbg !12
  %1044 = extractelement <4 x float> %1004, i64 2, !dbg !12
  %1045 = extractelement <4 x float> %1004, i64 3, !dbg !12
  %1046 = select i1 %.not, float %1043, float %1042, !dbg !12
  %1047 = bitcast float %1046 to i32, !dbg !12
  %1048 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1047), !dbg !12
  %1049 = bitcast i32 %1048 to float, !dbg !12
  %1050 = select i1 %.not, float %1042, float %1049, !dbg !12
  %1051 = select i1 %.not, float %1049, float %1043, !dbg !12
  %1052 = select i1 %.not, float %1045, float %1044, !dbg !12
  %1053 = bitcast float %1052 to i32, !dbg !12
  %1054 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1053), !dbg !12
  %1055 = bitcast i32 %1054 to float, !dbg !12
  %1056 = select i1 %.not, float %1044, float %1055, !dbg !12
  %1057 = select i1 %.not, float %1055, float %1045, !dbg !12
  %1058 = select i1 %.not40, float %1056, float %1050, !dbg !12
  %1059 = bitcast float %1058 to i32, !dbg !12
  %1060 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1059), !dbg !12
  %1061 = bitcast i32 %1060 to float, !dbg !12
  %1062 = select i1 %.not40, float %1057, float %1051, !dbg !12
  %1063 = bitcast float %1062 to i32, !dbg !12
  %1064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1063), !dbg !12
  %1065 = bitcast i32 %1064 to float, !dbg !12
  %1066 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %1067 = shufflevector <4 x i1> %1066, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1068 = insertelement <4 x float> poison, float %1050, i64 0, !dbg !12
  %1069 = insertelement <4 x float> %1068, float %1051, i64 1, !dbg !12
  %1070 = insertelement <4 x float> %1069, float %1061, i64 2, !dbg !12
  %1071 = insertelement <4 x float> %1070, float %1065, i64 3, !dbg !12
  %1072 = insertelement <4 x float> poison, float %1061, i64 0, !dbg !12
  %1073 = insertelement <4 x float> %1072, float %1065, i64 1, !dbg !12
  %1074 = insertelement <4 x float> %1073, float %1056, i64 2, !dbg !12
  %1075 = insertelement <4 x float> %1074, float %1057, i64 3, !dbg !12
  %1076 = select <4 x i1> %1067, <4 x float> %1071, <4 x float> %1075, !dbg !12
  %1077 = extractelement <4 x float> %1006, i64 0, !dbg !12
  %1078 = extractelement <4 x float> %1006, i64 1, !dbg !12
  %1079 = extractelement <4 x float> %1006, i64 2, !dbg !12
  %1080 = extractelement <4 x float> %1006, i64 3, !dbg !12
  %1081 = select i1 %.not, float %1078, float %1077, !dbg !12
  %1082 = bitcast float %1081 to i32, !dbg !12
  %1083 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1082), !dbg !12
  %1084 = bitcast i32 %1083 to float, !dbg !12
  %1085 = select i1 %.not, float %1077, float %1084, !dbg !12
  %1086 = select i1 %.not, float %1084, float %1078, !dbg !12
  %1087 = select i1 %.not, float %1080, float %1079, !dbg !12
  %1088 = bitcast float %1087 to i32, !dbg !12
  %1089 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1088), !dbg !12
  %1090 = bitcast i32 %1089 to float, !dbg !12
  %1091 = select i1 %.not, float %1079, float %1090, !dbg !12
  %1092 = select i1 %.not, float %1090, float %1080, !dbg !12
  %1093 = select i1 %.not40, float %1091, float %1085, !dbg !12
  %1094 = bitcast float %1093 to i32, !dbg !12
  %1095 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1094), !dbg !12
  %1096 = bitcast i32 %1095 to float, !dbg !12
  %1097 = select i1 %.not40, float %1092, float %1086, !dbg !12
  %1098 = bitcast float %1097 to i32, !dbg !12
  %1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1098), !dbg !12
  %1100 = bitcast i32 %1099 to float, !dbg !12
  %1101 = insertelement <4 x float> poison, float %1085, i64 0, !dbg !12
  %1102 = insertelement <4 x float> %1101, float %1086, i64 1, !dbg !12
  %1103 = insertelement <4 x float> %1102, float %1096, i64 2, !dbg !12
  %1104 = insertelement <4 x float> %1103, float %1100, i64 3, !dbg !12
  %1105 = insertelement <4 x float> poison, float %1096, i64 0, !dbg !12
  %1106 = insertelement <4 x float> %1105, float %1100, i64 1, !dbg !12
  %1107 = insertelement <4 x float> %1106, float %1091, i64 2, !dbg !12
  %1108 = insertelement <4 x float> %1107, float %1092, i64 3, !dbg !12
  %1109 = select <4 x i1> %1067, <4 x float> %1104, <4 x float> %1108, !dbg !12
  %1110 = extractelement <4 x float> %1007, i64 0, !dbg !12
  %1111 = extractelement <4 x float> %1007, i64 1, !dbg !12
  %1112 = extractelement <4 x float> %1007, i64 2, !dbg !12
  %1113 = extractelement <4 x float> %1007, i64 3, !dbg !12
  %1114 = select i1 %.not, float %1111, float %1110, !dbg !12
  %1115 = bitcast float %1114 to i32, !dbg !12
  %1116 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1115), !dbg !12
  %1117 = bitcast i32 %1116 to float, !dbg !12
  %1118 = select i1 %.not, float %1110, float %1117, !dbg !12
  %1119 = select i1 %.not, float %1117, float %1111, !dbg !12
  %1120 = select i1 %.not, float %1113, float %1112, !dbg !12
  %1121 = bitcast float %1120 to i32, !dbg !12
  %1122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1121), !dbg !12
  %1123 = bitcast i32 %1122 to float, !dbg !12
  %1124 = select i1 %.not, float %1112, float %1123, !dbg !12
  %1125 = select i1 %.not, float %1123, float %1113, !dbg !12
  %1126 = select i1 %.not40, float %1124, float %1118, !dbg !12
  %1127 = bitcast float %1126 to i32, !dbg !12
  %1128 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1127), !dbg !12
  %1129 = bitcast i32 %1128 to float, !dbg !12
  %1130 = select i1 %.not40, float %1125, float %1119, !dbg !12
  %1131 = bitcast float %1130 to i32, !dbg !12
  %1132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1131), !dbg !12
  %1133 = bitcast i32 %1132 to float, !dbg !12
  %1134 = insertelement <4 x float> poison, float %1118, i64 0, !dbg !12
  %1135 = insertelement <4 x float> %1134, float %1119, i64 1, !dbg !12
  %1136 = insertelement <4 x float> %1135, float %1129, i64 2, !dbg !12
  %1137 = insertelement <4 x float> %1136, float %1133, i64 3, !dbg !12
  %1138 = insertelement <4 x float> poison, float %1129, i64 0, !dbg !12
  %1139 = insertelement <4 x float> %1138, float %1133, i64 1, !dbg !12
  %1140 = insertelement <4 x float> %1139, float %1124, i64 2, !dbg !12
  %1141 = insertelement <4 x float> %1140, float %1125, i64 3, !dbg !12
  %1142 = select <4 x i1> %1067, <4 x float> %1137, <4 x float> %1141, !dbg !12
  %1143 = extractelement <4 x float> %1008, i64 0, !dbg !12
  %1144 = extractelement <4 x float> %1008, i64 1, !dbg !12
  %1145 = extractelement <4 x float> %1008, i64 2, !dbg !12
  %1146 = extractelement <4 x float> %1008, i64 3, !dbg !12
  %1147 = select i1 %.not, float %1144, float %1143, !dbg !12
  %1148 = bitcast float %1147 to i32, !dbg !12
  %1149 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1148), !dbg !12
  %1150 = bitcast i32 %1149 to float, !dbg !12
  %1151 = select i1 %.not, float %1143, float %1150, !dbg !12
  %1152 = select i1 %.not, float %1150, float %1144, !dbg !12
  %1153 = select i1 %.not, float %1146, float %1145, !dbg !12
  %1154 = bitcast float %1153 to i32, !dbg !12
  %1155 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1154), !dbg !12
  %1156 = bitcast i32 %1155 to float, !dbg !12
  %1157 = select i1 %.not, float %1145, float %1156, !dbg !12
  %1158 = select i1 %.not, float %1156, float %1146, !dbg !12
  %1159 = select i1 %.not40, float %1157, float %1151, !dbg !12
  %1160 = bitcast float %1159 to i32, !dbg !12
  %1161 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1160), !dbg !12
  %1162 = bitcast i32 %1161 to float, !dbg !12
  %1163 = select i1 %.not40, float %1158, float %1152, !dbg !12
  %1164 = bitcast float %1163 to i32, !dbg !12
  %1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1164), !dbg !12
  %1166 = bitcast i32 %1165 to float, !dbg !12
  %1167 = insertelement <4 x float> poison, float %1151, i64 0, !dbg !12
  %1168 = insertelement <4 x float> %1167, float %1152, i64 1, !dbg !12
  %1169 = insertelement <4 x float> %1168, float %1162, i64 2, !dbg !12
  %1170 = insertelement <4 x float> %1169, float %1166, i64 3, !dbg !12
  %1171 = insertelement <4 x float> poison, float %1162, i64 0, !dbg !12
  %1172 = insertelement <4 x float> %1171, float %1166, i64 1, !dbg !12
  %1173 = insertelement <4 x float> %1172, float %1157, i64 2, !dbg !12
  %1174 = insertelement <4 x float> %1173, float %1158, i64 3, !dbg !12
  %1175 = select <4 x i1> %1067, <4 x float> %1170, <4 x float> %1174, !dbg !12
  %1176 = extractelement <4 x float> %933, i64 0, !dbg !12
  %1177 = extractelement <4 x float> %933, i64 1, !dbg !12
  %1178 = extractelement <4 x float> %933, i64 2, !dbg !12
  %1179 = extractelement <4 x float> %933, i64 3, !dbg !12
  %1180 = select i1 %.not, float %1177, float %1176, !dbg !12
  %1181 = bitcast float %1180 to i32, !dbg !12
  %1182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1181), !dbg !12
  %1183 = bitcast i32 %1182 to float, !dbg !12
  %1184 = select i1 %.not, float %1176, float %1183, !dbg !12
  %1185 = select i1 %.not, float %1183, float %1177, !dbg !12
  %1186 = select i1 %.not, float %1179, float %1178, !dbg !12
  %1187 = bitcast float %1186 to i32, !dbg !12
  %1188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1187), !dbg !12
  %1189 = bitcast i32 %1188 to float, !dbg !12
  %1190 = select i1 %.not, float %1178, float %1189, !dbg !12
  %1191 = select i1 %.not, float %1189, float %1179, !dbg !12
  %1192 = select i1 %.not40, float %1190, float %1184, !dbg !12
  %1193 = bitcast float %1192 to i32, !dbg !12
  %1194 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1193), !dbg !12
  %1195 = bitcast i32 %1194 to float, !dbg !12
  %1196 = select i1 %.not40, float %1191, float %1185, !dbg !12
  %1197 = bitcast float %1196 to i32, !dbg !12
  %1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1197), !dbg !12
  %1199 = bitcast i32 %1198 to float, !dbg !12
  %1200 = insertelement <4 x float> poison, float %1184, i64 0, !dbg !12
  %1201 = insertelement <4 x float> %1200, float %1185, i64 1, !dbg !12
  %1202 = insertelement <4 x float> %1201, float %1195, i64 2, !dbg !12
  %1203 = insertelement <4 x float> %1202, float %1199, i64 3, !dbg !12
  %1204 = insertelement <4 x float> poison, float %1195, i64 0, !dbg !12
  %1205 = insertelement <4 x float> %1204, float %1199, i64 1, !dbg !12
  %1206 = insertelement <4 x float> %1205, float %1190, i64 2, !dbg !12
  %1207 = insertelement <4 x float> %1206, float %1191, i64 3, !dbg !12
  %1208 = select <4 x i1> %1067, <4 x float> %1203, <4 x float> %1207, !dbg !12
  %1209 = extractelement <4 x float> %935, i64 0, !dbg !12
  %1210 = extractelement <4 x float> %935, i64 1, !dbg !12
  %1211 = extractelement <4 x float> %935, i64 2, !dbg !12
  %1212 = extractelement <4 x float> %935, i64 3, !dbg !12
  %1213 = select i1 %.not, float %1210, float %1209, !dbg !12
  %1214 = bitcast float %1213 to i32, !dbg !12
  %1215 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1214), !dbg !12
  %1216 = bitcast i32 %1215 to float, !dbg !12
  %1217 = select i1 %.not, float %1209, float %1216, !dbg !12
  %1218 = select i1 %.not, float %1216, float %1210, !dbg !12
  %1219 = select i1 %.not, float %1212, float %1211, !dbg !12
  %1220 = bitcast float %1219 to i32, !dbg !12
  %1221 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1220), !dbg !12
  %1222 = bitcast i32 %1221 to float, !dbg !12
  %1223 = select i1 %.not, float %1211, float %1222, !dbg !12
  %1224 = select i1 %.not, float %1222, float %1212, !dbg !12
  %1225 = select i1 %.not40, float %1223, float %1217, !dbg !12
  %1226 = bitcast float %1225 to i32, !dbg !12
  %1227 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1226), !dbg !12
  %1228 = bitcast i32 %1227 to float, !dbg !12
  %1229 = select i1 %.not40, float %1224, float %1218, !dbg !12
  %1230 = bitcast float %1229 to i32, !dbg !12
  %1231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1230), !dbg !12
  %1232 = bitcast i32 %1231 to float, !dbg !12
  %1233 = insertelement <4 x float> poison, float %1217, i64 0, !dbg !12
  %1234 = insertelement <4 x float> %1233, float %1218, i64 1, !dbg !12
  %1235 = insertelement <4 x float> %1234, float %1228, i64 2, !dbg !12
  %1236 = insertelement <4 x float> %1235, float %1232, i64 3, !dbg !12
  %1237 = insertelement <4 x float> poison, float %1228, i64 0, !dbg !12
  %1238 = insertelement <4 x float> %1237, float %1232, i64 1, !dbg !12
  %1239 = insertelement <4 x float> %1238, float %1223, i64 2, !dbg !12
  %1240 = insertelement <4 x float> %1239, float %1224, i64 3, !dbg !12
  %1241 = select <4 x i1> %1067, <4 x float> %1236, <4 x float> %1240, !dbg !12
  %1242 = extractelement <4 x float> %936, i64 0, !dbg !12
  %1243 = extractelement <4 x float> %936, i64 1, !dbg !12
  %1244 = extractelement <4 x float> %936, i64 2, !dbg !12
  %1245 = extractelement <4 x float> %936, i64 3, !dbg !12
  %1246 = select i1 %.not, float %1243, float %1242, !dbg !12
  %1247 = bitcast float %1246 to i32, !dbg !12
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1247), !dbg !12
  %1249 = bitcast i32 %1248 to float, !dbg !12
  %1250 = select i1 %.not, float %1242, float %1249, !dbg !12
  %1251 = select i1 %.not, float %1249, float %1243, !dbg !12
  %1252 = select i1 %.not, float %1245, float %1244, !dbg !12
  %1253 = bitcast float %1252 to i32, !dbg !12
  %1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1253), !dbg !12
  %1255 = bitcast i32 %1254 to float, !dbg !12
  %1256 = select i1 %.not, float %1244, float %1255, !dbg !12
  %1257 = select i1 %.not, float %1255, float %1245, !dbg !12
  %1258 = select i1 %.not40, float %1256, float %1250, !dbg !12
  %1259 = bitcast float %1258 to i32, !dbg !12
  %1260 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1259), !dbg !12
  %1261 = bitcast i32 %1260 to float, !dbg !12
  %1262 = select i1 %.not40, float %1257, float %1251, !dbg !12
  %1263 = bitcast float %1262 to i32, !dbg !12
  %1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1263), !dbg !12
  %1265 = bitcast i32 %1264 to float, !dbg !12
  %1266 = insertelement <4 x float> poison, float %1250, i64 0, !dbg !12
  %1267 = insertelement <4 x float> %1266, float %1251, i64 1, !dbg !12
  %1268 = insertelement <4 x float> %1267, float %1261, i64 2, !dbg !12
  %1269 = insertelement <4 x float> %1268, float %1265, i64 3, !dbg !12
  %1270 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !12
  %1271 = insertelement <4 x float> %1270, float %1265, i64 1, !dbg !12
  %1272 = insertelement <4 x float> %1271, float %1256, i64 2, !dbg !12
  %1273 = insertelement <4 x float> %1272, float %1257, i64 3, !dbg !12
  %1274 = select <4 x i1> %1067, <4 x float> %1269, <4 x float> %1273, !dbg !12
  %1275 = extractelement <4 x float> %937, i64 0, !dbg !12
  %1276 = extractelement <4 x float> %937, i64 1, !dbg !12
  %1277 = extractelement <4 x float> %937, i64 2, !dbg !12
  %1278 = extractelement <4 x float> %937, i64 3, !dbg !12
  %1279 = select i1 %.not, float %1276, float %1275, !dbg !12
  %1280 = bitcast float %1279 to i32, !dbg !12
  %1281 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1280), !dbg !12
  %1282 = bitcast i32 %1281 to float, !dbg !12
  %1283 = select i1 %.not, float %1275, float %1282, !dbg !12
  %1284 = select i1 %.not, float %1282, float %1276, !dbg !12
  %1285 = select i1 %.not, float %1278, float %1277, !dbg !12
  %1286 = bitcast float %1285 to i32, !dbg !12
  %1287 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1286), !dbg !12
  %1288 = bitcast i32 %1287 to float, !dbg !12
  %1289 = select i1 %.not, float %1277, float %1288, !dbg !12
  %1290 = select i1 %.not, float %1288, float %1278, !dbg !12
  %1291 = select i1 %.not40, float %1289, float %1283, !dbg !12
  %1292 = bitcast float %1291 to i32, !dbg !12
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1292), !dbg !12
  %1294 = bitcast i32 %1293 to float, !dbg !12
  %1295 = select i1 %.not40, float %1290, float %1284, !dbg !12
  %1296 = bitcast float %1295 to i32, !dbg !12
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1296), !dbg !12
  %1298 = bitcast i32 %1297 to float, !dbg !12
  %1299 = insertelement <4 x float> poison, float %1283, i64 0, !dbg !12
  %1300 = insertelement <4 x float> %1299, float %1284, i64 1, !dbg !12
  %1301 = insertelement <4 x float> %1300, float %1294, i64 2, !dbg !12
  %1302 = insertelement <4 x float> %1301, float %1298, i64 3, !dbg !12
  %1303 = insertelement <4 x float> poison, float %1294, i64 0, !dbg !12
  %1304 = insertelement <4 x float> %1303, float %1298, i64 1, !dbg !12
  %1305 = insertelement <4 x float> %1304, float %1289, i64 2, !dbg !12
  %1306 = insertelement <4 x float> %1305, float %1290, i64 3, !dbg !12
  %1307 = select <4 x i1> %1067, <4 x float> %1302, <4 x float> %1306, !dbg !12
  %1308 = extractelement <4 x float> %861, i64 0, !dbg !12
  %1309 = extractelement <4 x float> %861, i64 1, !dbg !12
  %1310 = extractelement <4 x float> %861, i64 2, !dbg !12
  %1311 = extractelement <4 x float> %861, i64 3, !dbg !12
  %1312 = select i1 %.not, float %1309, float %1308, !dbg !12
  %1313 = bitcast float %1312 to i32, !dbg !12
  %1314 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1313), !dbg !12
  %1315 = bitcast i32 %1314 to float, !dbg !12
  %1316 = select i1 %.not, float %1308, float %1315, !dbg !12
  %1317 = select i1 %.not, float %1315, float %1309, !dbg !12
  %1318 = select i1 %.not, float %1311, float %1310, !dbg !12
  %1319 = bitcast float %1318 to i32, !dbg !12
  %1320 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1319), !dbg !12
  %1321 = bitcast i32 %1320 to float, !dbg !12
  %1322 = select i1 %.not, float %1310, float %1321, !dbg !12
  %1323 = select i1 %.not, float %1321, float %1311, !dbg !12
  %1324 = select i1 %.not40, float %1322, float %1316, !dbg !12
  %1325 = bitcast float %1324 to i32, !dbg !12
  %1326 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1325), !dbg !12
  %1327 = bitcast i32 %1326 to float, !dbg !12
  %1328 = select i1 %.not40, float %1323, float %1317, !dbg !12
  %1329 = bitcast float %1328 to i32, !dbg !12
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1329), !dbg !12
  %1331 = bitcast i32 %1330 to float, !dbg !12
  %1332 = insertelement <4 x float> poison, float %1316, i64 0, !dbg !12
  %1333 = insertelement <4 x float> %1332, float %1317, i64 1, !dbg !12
  %1334 = insertelement <4 x float> %1333, float %1327, i64 2, !dbg !12
  %1335 = insertelement <4 x float> %1334, float %1331, i64 3, !dbg !12
  %1336 = insertelement <4 x float> poison, float %1327, i64 0, !dbg !12
  %1337 = insertelement <4 x float> %1336, float %1331, i64 1, !dbg !12
  %1338 = insertelement <4 x float> %1337, float %1322, i64 2, !dbg !12
  %1339 = insertelement <4 x float> %1338, float %1323, i64 3, !dbg !12
  %1340 = select <4 x i1> %1067, <4 x float> %1335, <4 x float> %1339, !dbg !12
  %1341 = extractelement <4 x float> %863, i64 0, !dbg !12
  %1342 = extractelement <4 x float> %863, i64 1, !dbg !12
  %1343 = extractelement <4 x float> %863, i64 2, !dbg !12
  %1344 = extractelement <4 x float> %863, i64 3, !dbg !12
  %1345 = select i1 %.not, float %1342, float %1341, !dbg !12
  %1346 = bitcast float %1345 to i32, !dbg !12
  %1347 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1346), !dbg !12
  %1348 = bitcast i32 %1347 to float, !dbg !12
  %1349 = select i1 %.not, float %1341, float %1348, !dbg !12
  %1350 = select i1 %.not, float %1348, float %1342, !dbg !12
  %1351 = select i1 %.not, float %1344, float %1343, !dbg !12
  %1352 = bitcast float %1351 to i32, !dbg !12
  %1353 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1352), !dbg !12
  %1354 = bitcast i32 %1353 to float, !dbg !12
  %1355 = select i1 %.not, float %1343, float %1354, !dbg !12
  %1356 = select i1 %.not, float %1354, float %1344, !dbg !12
  %1357 = select i1 %.not40, float %1355, float %1349, !dbg !12
  %1358 = bitcast float %1357 to i32, !dbg !12
  %1359 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1358), !dbg !12
  %1360 = bitcast i32 %1359 to float, !dbg !12
  %1361 = select i1 %.not40, float %1356, float %1350, !dbg !12
  %1362 = bitcast float %1361 to i32, !dbg !12
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1362), !dbg !12
  %1364 = bitcast i32 %1363 to float, !dbg !12
  %1365 = insertelement <4 x float> poison, float %1349, i64 0, !dbg !12
  %1366 = insertelement <4 x float> %1365, float %1350, i64 1, !dbg !12
  %1367 = insertelement <4 x float> %1366, float %1360, i64 2, !dbg !12
  %1368 = insertelement <4 x float> %1367, float %1364, i64 3, !dbg !12
  %1369 = insertelement <4 x float> poison, float %1360, i64 0, !dbg !12
  %1370 = insertelement <4 x float> %1369, float %1364, i64 1, !dbg !12
  %1371 = insertelement <4 x float> %1370, float %1355, i64 2, !dbg !12
  %1372 = insertelement <4 x float> %1371, float %1356, i64 3, !dbg !12
  %1373 = select <4 x i1> %1067, <4 x float> %1368, <4 x float> %1372, !dbg !12
  %1374 = extractelement <4 x float> %864, i64 0, !dbg !12
  %1375 = extractelement <4 x float> %864, i64 1, !dbg !12
  %1376 = extractelement <4 x float> %864, i64 2, !dbg !12
  %1377 = extractelement <4 x float> %864, i64 3, !dbg !12
  %1378 = select i1 %.not, float %1375, float %1374, !dbg !12
  %1379 = bitcast float %1378 to i32, !dbg !12
  %1380 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1379), !dbg !12
  %1381 = bitcast i32 %1380 to float, !dbg !12
  %1382 = select i1 %.not, float %1374, float %1381, !dbg !12
  %1383 = select i1 %.not, float %1381, float %1375, !dbg !12
  %1384 = select i1 %.not, float %1377, float %1376, !dbg !12
  %1385 = bitcast float %1384 to i32, !dbg !12
  %1386 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1385), !dbg !12
  %1387 = bitcast i32 %1386 to float, !dbg !12
  %1388 = select i1 %.not, float %1376, float %1387, !dbg !12
  %1389 = select i1 %.not, float %1387, float %1377, !dbg !12
  %1390 = select i1 %.not40, float %1388, float %1382, !dbg !12
  %1391 = bitcast float %1390 to i32, !dbg !12
  %1392 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1391), !dbg !12
  %1393 = bitcast i32 %1392 to float, !dbg !12
  %1394 = select i1 %.not40, float %1389, float %1383, !dbg !12
  %1395 = bitcast float %1394 to i32, !dbg !12
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1395), !dbg !12
  %1397 = bitcast i32 %1396 to float, !dbg !12
  %1398 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !12
  %1399 = insertelement <4 x float> %1398, float %1383, i64 1, !dbg !12
  %1400 = insertelement <4 x float> %1399, float %1393, i64 2, !dbg !12
  %1401 = insertelement <4 x float> %1400, float %1397, i64 3, !dbg !12
  %1402 = insertelement <4 x float> poison, float %1393, i64 0, !dbg !12
  %1403 = insertelement <4 x float> %1402, float %1397, i64 1, !dbg !12
  %1404 = insertelement <4 x float> %1403, float %1388, i64 2, !dbg !12
  %1405 = insertelement <4 x float> %1404, float %1389, i64 3, !dbg !12
  %1406 = select <4 x i1> %1067, <4 x float> %1401, <4 x float> %1405, !dbg !12
  %1407 = extractelement <4 x float> %865, i64 0, !dbg !12
  %1408 = extractelement <4 x float> %865, i64 1, !dbg !12
  %1409 = extractelement <4 x float> %865, i64 2, !dbg !12
  %1410 = extractelement <4 x float> %865, i64 3, !dbg !12
  %1411 = select i1 %.not, float %1408, float %1407, !dbg !12
  %1412 = bitcast float %1411 to i32, !dbg !12
  %1413 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1412), !dbg !12
  %1414 = bitcast i32 %1413 to float, !dbg !12
  %1415 = select i1 %.not, float %1407, float %1414, !dbg !12
  %1416 = select i1 %.not, float %1414, float %1408, !dbg !12
  %1417 = select i1 %.not, float %1410, float %1409, !dbg !12
  %1418 = bitcast float %1417 to i32, !dbg !12
  %1419 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1418), !dbg !12
  %1420 = bitcast i32 %1419 to float, !dbg !12
  %1421 = select i1 %.not, float %1409, float %1420, !dbg !12
  %1422 = select i1 %.not, float %1420, float %1410, !dbg !12
  %1423 = select i1 %.not40, float %1421, float %1415, !dbg !12
  %1424 = bitcast float %1423 to i32, !dbg !12
  %1425 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1424), !dbg !12
  %1426 = bitcast i32 %1425 to float, !dbg !12
  %1427 = select i1 %.not40, float %1422, float %1416, !dbg !12
  %1428 = bitcast float %1427 to i32, !dbg !12
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1428), !dbg !12
  %1430 = bitcast i32 %1429 to float, !dbg !12
  %1431 = insertelement <4 x float> poison, float %1415, i64 0, !dbg !12
  %1432 = insertelement <4 x float> %1431, float %1416, i64 1, !dbg !12
  %1433 = insertelement <4 x float> %1432, float %1426, i64 2, !dbg !12
  %1434 = insertelement <4 x float> %1433, float %1430, i64 3, !dbg !12
  %1435 = insertelement <4 x float> poison, float %1426, i64 0, !dbg !12
  %1436 = insertelement <4 x float> %1435, float %1430, i64 1, !dbg !12
  %1437 = insertelement <4 x float> %1436, float %1421, i64 2, !dbg !12
  %1438 = insertelement <4 x float> %1437, float %1422, i64 3, !dbg !12
  %1439 = select <4 x i1> %1067, <4 x float> %1434, <4 x float> %1438, !dbg !12
  %1440 = extractelement <4 x float> %770, i64 0, !dbg !12
  %1441 = extractelement <4 x float> %770, i64 1, !dbg !12
  %1442 = extractelement <4 x float> %770, i64 2, !dbg !12
  %1443 = extractelement <4 x float> %770, i64 3, !dbg !12
  %1444 = select i1 %.not, float %1441, float %1440, !dbg !12
  %1445 = bitcast float %1444 to i32, !dbg !12
  %1446 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1445), !dbg !12
  %1447 = bitcast i32 %1446 to float, !dbg !12
  %1448 = select i1 %.not, float %1440, float %1447, !dbg !12
  %1449 = select i1 %.not, float %1447, float %1441, !dbg !12
  %1450 = select i1 %.not, float %1443, float %1442, !dbg !12
  %1451 = bitcast float %1450 to i32, !dbg !12
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1451), !dbg !12
  %1453 = bitcast i32 %1452 to float, !dbg !12
  %1454 = select i1 %.not, float %1442, float %1453, !dbg !12
  %1455 = select i1 %.not, float %1453, float %1443, !dbg !12
  %1456 = select i1 %.not40, float %1454, float %1448, !dbg !12
  %1457 = bitcast float %1456 to i32, !dbg !12
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1457), !dbg !12
  %1459 = select i1 %.not40, float %1455, float %1449, !dbg !12
  %1460 = bitcast float %1459 to i32, !dbg !12
  %1461 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1460), !dbg !12
  %1462 = extractelement <4 x float> %772, i64 0, !dbg !12
  %1463 = extractelement <4 x float> %772, i64 1, !dbg !12
  %1464 = extractelement <4 x float> %772, i64 2, !dbg !12
  %1465 = extractelement <4 x float> %772, i64 3, !dbg !12
  %1466 = select i1 %.not, float %1463, float %1462, !dbg !12
  %1467 = bitcast float %1466 to i32, !dbg !12
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1467), !dbg !12
  %1469 = bitcast i32 %1468 to float, !dbg !12
  %1470 = select i1 %.not, float %1462, float %1469, !dbg !12
  %1471 = select i1 %.not, float %1469, float %1463, !dbg !12
  %1472 = select i1 %.not, float %1465, float %1464, !dbg !12
  %1473 = bitcast float %1472 to i32, !dbg !12
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1473), !dbg !12
  %1475 = bitcast i32 %1474 to float, !dbg !12
  %1476 = select i1 %.not, float %1464, float %1475, !dbg !12
  %1477 = select i1 %.not, float %1475, float %1465, !dbg !12
  %1478 = select i1 %.not40, float %1476, float %1470, !dbg !12
  %1479 = bitcast float %1478 to i32, !dbg !12
  %1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1479), !dbg !12
  %1481 = select i1 %.not40, float %1477, float %1471, !dbg !12
  %1482 = bitcast float %1481 to i32, !dbg !12
  %1483 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1482), !dbg !12
  %1484 = extractelement <4 x float> %792, i64 0, !dbg !12
  %1485 = extractelement <4 x float> %792, i64 1, !dbg !12
  %1486 = extractelement <4 x float> %792, i64 2, !dbg !12
  %1487 = extractelement <4 x float> %792, i64 3, !dbg !12
  %1488 = select i1 %.not, float %1485, float %1484, !dbg !12
  %1489 = bitcast float %1488 to i32, !dbg !12
  %1490 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1489), !dbg !12
  %1491 = bitcast i32 %1490 to float, !dbg !12
  %1492 = select i1 %.not, float %1484, float %1491, !dbg !12
  %1493 = select i1 %.not, float %1491, float %1485, !dbg !12
  %1494 = select i1 %.not, float %1487, float %1486, !dbg !12
  %1495 = bitcast float %1494 to i32, !dbg !12
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1495), !dbg !12
  %1497 = bitcast i32 %1496 to float, !dbg !12
  %1498 = select i1 %.not, float %1486, float %1497, !dbg !12
  %1499 = select i1 %.not, float %1497, float %1487, !dbg !12
  %1500 = select i1 %.not40, float %1498, float %1492, !dbg !12
  %1501 = bitcast float %1500 to i32, !dbg !12
  %1502 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1501), !dbg !12
  %1503 = select i1 %.not40, float %1499, float %1493, !dbg !12
  %1504 = bitcast float %1503 to i32, !dbg !12
  %1505 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1504), !dbg !12
  %1506 = extractelement <4 x float> %793, i64 0, !dbg !12
  %1507 = extractelement <4 x float> %793, i64 1, !dbg !12
  %1508 = extractelement <4 x float> %793, i64 2, !dbg !12
  %1509 = extractelement <4 x float> %793, i64 3, !dbg !12
  %1510 = select i1 %.not, float %1507, float %1506, !dbg !12
  %1511 = bitcast float %1510 to i32, !dbg !12
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1511), !dbg !12
  %1513 = bitcast i32 %1512 to float, !dbg !12
  %1514 = select i1 %.not, float %1506, float %1513, !dbg !12
  %1515 = select i1 %.not, float %1513, float %1507, !dbg !12
  %1516 = select i1 %.not, float %1509, float %1508, !dbg !12
  %1517 = bitcast float %1516 to i32, !dbg !12
  %1518 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1517), !dbg !12
  %1519 = bitcast i32 %1518 to float, !dbg !12
  %1520 = select i1 %.not, float %1508, float %1519, !dbg !12
  %1521 = select i1 %.not, float %1519, float %1509, !dbg !12
  %1522 = select i1 %.not40, float %1520, float %1514, !dbg !12
  %1523 = bitcast float %1522 to i32, !dbg !12
  %1524 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1523), !dbg !12
  %1525 = select i1 %.not40, float %1521, float %1515, !dbg !12
  %1526 = bitcast float %1525 to i32, !dbg !12
  %1527 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1526), !dbg !12
  %1528 = shufflevector <1 x float> %1039, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %1529 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1528, !dbg !46
  %1530 = shufflevector <1 x float> %1034, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %1531 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1530, !dbg !46
  %1532 = bitcast i32 %1524 to float, !dbg !12
  %1533 = bitcast i32 %1527 to float, !dbg !12
  %1534 = insertelement <4 x float> poison, float %1514, i64 0, !dbg !12
  %1535 = insertelement <4 x float> %1534, float %1515, i64 1, !dbg !12
  %1536 = insertelement <4 x float> %1535, float %1532, i64 2, !dbg !12
  %1537 = insertelement <4 x float> %1536, float %1533, i64 3, !dbg !12
  %1538 = insertelement <4 x float> poison, float %1532, i64 0, !dbg !12
  %1539 = insertelement <4 x float> %1538, float %1533, i64 1, !dbg !12
  %1540 = insertelement <4 x float> %1539, float %1520, i64 2, !dbg !12
  %1541 = insertelement <4 x float> %1540, float %1521, i64 3, !dbg !12
  %1542 = select <4 x i1> %1067, <4 x float> %1537, <4 x float> %1541, !dbg !12
  %1543 = fmul <4 x float> %1531, %1542, !dbg !46
  %1544 = fptrunc <4 x float> %1543 to <4 x half>, !dbg !47
  %1545 = extractelement <4 x half> %1544, i64 3, !dbg !47
  %1546 = extractelement <4 x half> %1544, i64 2, !dbg !47
  %1547 = extractelement <4 x half> %1544, i64 1, !dbg !47
  %1548 = extractelement <4 x half> %1544, i64 0, !dbg !47
  %1549 = bitcast i32 %1502 to float, !dbg !12
  %1550 = bitcast i32 %1505 to float, !dbg !12
  %1551 = insertelement <4 x float> poison, float %1492, i64 0, !dbg !12
  %1552 = insertelement <4 x float> %1551, float %1493, i64 1, !dbg !12
  %1553 = insertelement <4 x float> %1552, float %1549, i64 2, !dbg !12
  %1554 = insertelement <4 x float> %1553, float %1550, i64 3, !dbg !12
  %1555 = insertelement <4 x float> poison, float %1549, i64 0, !dbg !12
  %1556 = insertelement <4 x float> %1555, float %1550, i64 1, !dbg !12
  %1557 = insertelement <4 x float> %1556, float %1498, i64 2, !dbg !12
  %1558 = insertelement <4 x float> %1557, float %1499, i64 3, !dbg !12
  %1559 = select <4 x i1> %1067, <4 x float> %1554, <4 x float> %1558, !dbg !12
  %1560 = fmul <4 x float> %1531, %1559, !dbg !46
  %1561 = fptrunc <4 x float> %1560 to <4 x half>, !dbg !47
  %1562 = extractelement <4 x half> %1561, i64 3, !dbg !47
  %1563 = extractelement <4 x half> %1561, i64 2, !dbg !47
  %1564 = extractelement <4 x half> %1561, i64 1, !dbg !47
  %1565 = extractelement <4 x half> %1561, i64 0, !dbg !47
  %1566 = bitcast i32 %1480 to float, !dbg !12
  %1567 = bitcast i32 %1483 to float, !dbg !12
  %1568 = insertelement <4 x float> poison, float %1470, i64 0, !dbg !12
  %1569 = insertelement <4 x float> %1568, float %1471, i64 1, !dbg !12
  %1570 = insertelement <4 x float> %1569, float %1566, i64 2, !dbg !12
  %1571 = insertelement <4 x float> %1570, float %1567, i64 3, !dbg !12
  %1572 = insertelement <4 x float> poison, float %1566, i64 0, !dbg !12
  %1573 = insertelement <4 x float> %1572, float %1567, i64 1, !dbg !12
  %1574 = insertelement <4 x float> %1573, float %1476, i64 2, !dbg !12
  %1575 = insertelement <4 x float> %1574, float %1477, i64 3, !dbg !12
  %1576 = select <4 x i1> %1067, <4 x float> %1571, <4 x float> %1575, !dbg !12
  %1577 = fmul <4 x float> %1529, %1576, !dbg !46
  %1578 = fptrunc <4 x float> %1577 to <4 x half>, !dbg !47
  %1579 = extractelement <4 x half> %1578, i64 3, !dbg !47
  %1580 = extractelement <4 x half> %1578, i64 2, !dbg !47
  %1581 = extractelement <4 x half> %1578, i64 1, !dbg !47
  %1582 = extractelement <4 x half> %1578, i64 0, !dbg !47
  %1583 = bitcast i32 %1458 to float, !dbg !12
  %1584 = bitcast i32 %1461 to float, !dbg !12
  %1585 = insertelement <4 x float> poison, float %1448, i64 0, !dbg !12
  %1586 = insertelement <4 x float> %1585, float %1449, i64 1, !dbg !12
  %1587 = insertelement <4 x float> %1586, float %1583, i64 2, !dbg !12
  %1588 = insertelement <4 x float> %1587, float %1584, i64 3, !dbg !12
  %1589 = insertelement <4 x float> poison, float %1583, i64 0, !dbg !12
  %1590 = insertelement <4 x float> %1589, float %1584, i64 1, !dbg !12
  %1591 = insertelement <4 x float> %1590, float %1454, i64 2, !dbg !12
  %1592 = insertelement <4 x float> %1591, float %1455, i64 3, !dbg !12
  %1593 = select <4 x i1> %1067, <4 x float> %1588, <4 x float> %1592, !dbg !12
  %1594 = fmul <4 x float> %1529, %1593, !dbg !46
  %1595 = fptrunc <4 x float> %1594 to <4 x half>, !dbg !47
  %1596 = extractelement <4 x half> %1595, i64 3, !dbg !47
  %1597 = extractelement <4 x half> %1595, i64 2, !dbg !47
  %1598 = extractelement <4 x half> %1595, i64 1, !dbg !47
  %1599 = extractelement <4 x half> %1595, i64 0, !dbg !47
  %address.thread848 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local849 = zext nneg i32 %address.thread848 to i64, !dbg !9
  %1600 = lshr i64 %address.local849, 2, !dbg !48
  %address.local852 = and i64 %1600, 48, !dbg !48
  %address.local854 = and i64 %address.local849, 240, !dbg !48
  %address.local853 = add i64 %34, %address.local849, !dbg !48
  %address.local855 = sub i64 %address.local853, %address.local854, !dbg !48
  %address.local856 = add i64 %address.local855, %address.local852, !dbg !48
  %1601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !48
  %address.local859.idx = shl nuw nsw i64 %address.local856, 8, !dbg !48
  %address.local859 = getelementptr inbounds nuw i8, ptr addrspace(1) %1601, i64 %address.local859.idx, !dbg !48
  %address.local861 = and i64 %1600, 12, !dbg !11
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local861, !dbg !48
  store half %1599, ptr addrspace(1) %address.local862, align 2, !dbg !48
  %address.local864 = or disjoint i64 %address.local861, 1, !dbg !48
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local864, !dbg !48
  store half %1598, ptr addrspace(1) %address.local865, align 2, !dbg !48
  %address.local867 = or disjoint i64 %address.local861, 2, !dbg !48
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local867, !dbg !48
  store half %1597, ptr addrspace(1) %address.local868, align 2, !dbg !48
  %address.local870 = or disjoint i64 %address.local861, 3, !dbg !48
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local870, !dbg !48
  store half %1596, ptr addrspace(1) %address.local871, align 2, !dbg !48
  %address.local873 = or disjoint i64 %address.local861, 16, !dbg !48
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local873, !dbg !48
  store half %1582, ptr addrspace(1) %address.local874, align 2, !dbg !48
  %address.local876 = or disjoint i64 %address.local861, 17, !dbg !48
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local876, !dbg !48
  store half %1581, ptr addrspace(1) %address.local877, align 2, !dbg !48
  %address.local879 = or disjoint i64 %address.local861, 18, !dbg !48
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local879, !dbg !48
  store half %1580, ptr addrspace(1) %address.local880, align 2, !dbg !48
  %address.local882 = or disjoint i64 %address.local861, 19, !dbg !48
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local882, !dbg !48
  store half %1579, ptr addrspace(1) %address.local883, align 2, !dbg !48
  %address.local887 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local859, i64 16384, !dbg !48
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local861, !dbg !48
  store half %1565, ptr addrspace(1) %address.local888, align 2, !dbg !48
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local864, !dbg !48
  store half %1564, ptr addrspace(1) %address.local890, align 2, !dbg !48
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local867, !dbg !48
  store half %1563, ptr addrspace(1) %address.local892, align 2, !dbg !48
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local870, !dbg !48
  store half %1562, ptr addrspace(1) %address.local894, align 2, !dbg !48
  %address.local896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local873, !dbg !48
  store half %1548, ptr addrspace(1) %address.local896, align 2, !dbg !48
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local876, !dbg !48
  store half %1547, ptr addrspace(1) %address.local898, align 2, !dbg !48
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local879, !dbg !48
  store half %1546, ptr addrspace(1) %address.local900, align 2, !dbg !48
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local882, !dbg !48
  store half %1545, ptr addrspace(1) %address.local902, align 2, !dbg !48
  %1602 = fmul <4 x float> %1531, %1439, !dbg !46
  %1603 = fptrunc <4 x float> %1602 to <4 x half>, !dbg !47
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !47
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !47
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !47
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !47
  %1608 = fmul <4 x float> %1531, %1406, !dbg !46
  %1609 = fptrunc <4 x float> %1608 to <4 x half>, !dbg !47
  %1610 = extractelement <4 x half> %1609, i64 3, !dbg !47
  %1611 = extractelement <4 x half> %1609, i64 2, !dbg !47
  %1612 = extractelement <4 x half> %1609, i64 1, !dbg !47
  %1613 = extractelement <4 x half> %1609, i64 0, !dbg !47
  %1614 = fmul <4 x float> %1529, %1373, !dbg !46
  %1615 = fptrunc <4 x float> %1614 to <4 x half>, !dbg !47
  %1616 = extractelement <4 x half> %1615, i64 3, !dbg !47
  %1617 = extractelement <4 x half> %1615, i64 2, !dbg !47
  %1618 = extractelement <4 x half> %1615, i64 1, !dbg !47
  %1619 = extractelement <4 x half> %1615, i64 0, !dbg !47
  %1620 = fmul <4 x float> %1529, %1340, !dbg !46
  %1621 = fptrunc <4 x float> %1620 to <4 x half>, !dbg !47
  %1622 = extractelement <4 x half> %1621, i64 3, !dbg !47
  %1623 = extractelement <4 x half> %1621, i64 2, !dbg !47
  %1624 = extractelement <4 x half> %1621, i64 1, !dbg !47
  %1625 = extractelement <4 x half> %1621, i64 0, !dbg !47
  %address.local917 = or disjoint i64 %address.local861, 32, !dbg !48
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local917, !dbg !48
  store half %1625, ptr addrspace(1) %address.local918, align 2, !dbg !48
  %address.local920 = or disjoint i64 %address.local861, 33, !dbg !48
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local920, !dbg !48
  store half %1624, ptr addrspace(1) %address.local921, align 2, !dbg !48
  %address.local923 = or disjoint i64 %address.local861, 34, !dbg !48
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local923, !dbg !48
  store half %1623, ptr addrspace(1) %address.local924, align 2, !dbg !48
  %address.local926 = or disjoint i64 %address.local861, 35, !dbg !48
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local926, !dbg !48
  store half %1622, ptr addrspace(1) %address.local927, align 2, !dbg !48
  %address.local929 = or disjoint i64 %address.local861, 48, !dbg !48
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local929, !dbg !48
  store half %1619, ptr addrspace(1) %address.local930, align 2, !dbg !48
  %address.local932 = or disjoint i64 %address.local861, 49, !dbg !48
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local932, !dbg !48
  store half %1618, ptr addrspace(1) %address.local933, align 2, !dbg !48
  %address.local935 = or disjoint i64 %address.local861, 50, !dbg !48
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local935, !dbg !48
  store half %1617, ptr addrspace(1) %address.local936, align 2, !dbg !48
  %address.local938 = or i64 %1600, 51, !dbg !48
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local938, !dbg !48
  store half %1616, ptr addrspace(1) %address.local939, align 2, !dbg !48
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local917, !dbg !48
  store half %1613, ptr addrspace(1) %address.local944, align 2, !dbg !48
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local920, !dbg !48
  store half %1612, ptr addrspace(1) %address.local946, align 2, !dbg !48
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local923, !dbg !48
  store half %1611, ptr addrspace(1) %address.local948, align 2, !dbg !48
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local926, !dbg !48
  store half %1610, ptr addrspace(1) %address.local950, align 2, !dbg !48
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local929, !dbg !48
  store half %1607, ptr addrspace(1) %address.local952, align 2, !dbg !48
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local932, !dbg !48
  store half %1606, ptr addrspace(1) %address.local954, align 2, !dbg !48
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local935, !dbg !48
  store half %1605, ptr addrspace(1) %address.local956, align 2, !dbg !48
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local938, !dbg !48
  store half %1604, ptr addrspace(1) %address.local958, align 2, !dbg !48
  %1626 = fmul <4 x float> %1531, %1307, !dbg !46
  %1627 = fptrunc <4 x float> %1626 to <4 x half>, !dbg !47
  %1628 = extractelement <4 x half> %1627, i64 3, !dbg !47
  %1629 = extractelement <4 x half> %1627, i64 2, !dbg !47
  %1630 = extractelement <4 x half> %1627, i64 1, !dbg !47
  %1631 = extractelement <4 x half> %1627, i64 0, !dbg !47
  %1632 = fmul <4 x float> %1531, %1274, !dbg !46
  %1633 = fptrunc <4 x float> %1632 to <4 x half>, !dbg !47
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !47
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !47
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !47
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !47
  %1638 = fmul <4 x float> %1529, %1241, !dbg !46
  %1639 = fptrunc <4 x float> %1638 to <4 x half>, !dbg !47
  %1640 = extractelement <4 x half> %1639, i64 3, !dbg !47
  %1641 = extractelement <4 x half> %1639, i64 2, !dbg !47
  %1642 = extractelement <4 x half> %1639, i64 1, !dbg !47
  %1643 = extractelement <4 x half> %1639, i64 0, !dbg !47
  %1644 = fmul <4 x float> %1529, %1208, !dbg !46
  %1645 = fptrunc <4 x float> %1644 to <4 x half>, !dbg !47
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !47
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !47
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !47
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !47
  %address.local973 = or disjoint i64 %address.local861, 64, !dbg !48
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local973, !dbg !48
  store half %1649, ptr addrspace(1) %address.local974, align 2, !dbg !48
  %address.local976 = or disjoint i64 %address.local861, 65, !dbg !48
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local976, !dbg !48
  store half %1648, ptr addrspace(1) %address.local977, align 2, !dbg !48
  %address.local979 = or disjoint i64 %address.local861, 66, !dbg !48
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local979, !dbg !48
  store half %1647, ptr addrspace(1) %address.local980, align 2, !dbg !48
  %address.local982 = or disjoint i64 %address.local861, 67, !dbg !48
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local982, !dbg !48
  store half %1646, ptr addrspace(1) %address.local983, align 2, !dbg !48
  %address.local985 = or disjoint i64 %address.local861, 80, !dbg !48
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local985, !dbg !48
  store half %1643, ptr addrspace(1) %address.local986, align 2, !dbg !48
  %address.local988 = or disjoint i64 %address.local861, 81, !dbg !48
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local988, !dbg !48
  store half %1642, ptr addrspace(1) %address.local989, align 2, !dbg !48
  %address.local991 = or disjoint i64 %address.local861, 82, !dbg !48
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local991, !dbg !48
  store half %1641, ptr addrspace(1) %address.local992, align 2, !dbg !48
  %address.local994 = or disjoint i64 %address.local861, 83, !dbg !48
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local994, !dbg !48
  store half %1640, ptr addrspace(1) %address.local995, align 2, !dbg !48
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local973, !dbg !48
  store half %1637, ptr addrspace(1) %address.local1000, align 2, !dbg !48
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local976, !dbg !48
  store half %1636, ptr addrspace(1) %address.local1002, align 2, !dbg !48
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local979, !dbg !48
  store half %1635, ptr addrspace(1) %address.local1004, align 2, !dbg !48
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local982, !dbg !48
  store half %1634, ptr addrspace(1) %address.local1006, align 2, !dbg !48
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local985, !dbg !48
  store half %1631, ptr addrspace(1) %address.local1008, align 2, !dbg !48
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local988, !dbg !48
  store half %1630, ptr addrspace(1) %address.local1010, align 2, !dbg !48
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local991, !dbg !48
  store half %1629, ptr addrspace(1) %address.local1012, align 2, !dbg !48
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local994, !dbg !48
  store half %1628, ptr addrspace(1) %address.local1014, align 2, !dbg !48
  %1650 = fmul <4 x float> %1531, %1175, !dbg !46
  %1651 = fptrunc <4 x float> %1650 to <4 x half>, !dbg !47
  %1652 = extractelement <4 x half> %1651, i64 3, !dbg !47
  %1653 = extractelement <4 x half> %1651, i64 2, !dbg !47
  %1654 = extractelement <4 x half> %1651, i64 1, !dbg !47
  %1655 = extractelement <4 x half> %1651, i64 0, !dbg !47
  %1656 = fmul <4 x float> %1531, %1142, !dbg !46
  %1657 = fptrunc <4 x float> %1656 to <4 x half>, !dbg !47
  %1658 = extractelement <4 x half> %1657, i64 3, !dbg !47
  %1659 = extractelement <4 x half> %1657, i64 2, !dbg !47
  %1660 = extractelement <4 x half> %1657, i64 1, !dbg !47
  %1661 = extractelement <4 x half> %1657, i64 0, !dbg !47
  %1662 = fmul <4 x float> %1529, %1109, !dbg !46
  %1663 = fptrunc <4 x float> %1662 to <4 x half>, !dbg !47
  %1664 = extractelement <4 x half> %1663, i64 3, !dbg !47
  %1665 = extractelement <4 x half> %1663, i64 2, !dbg !47
  %1666 = extractelement <4 x half> %1663, i64 1, !dbg !47
  %1667 = extractelement <4 x half> %1663, i64 0, !dbg !47
  %1668 = fmul <4 x float> %1529, %1076, !dbg !46
  %1669 = fptrunc <4 x float> %1668 to <4 x half>, !dbg !47
  %1670 = extractelement <4 x half> %1669, i64 3, !dbg !47
  %1671 = extractelement <4 x half> %1669, i64 2, !dbg !47
  %1672 = extractelement <4 x half> %1669, i64 1, !dbg !47
  %1673 = extractelement <4 x half> %1669, i64 0, !dbg !47
  %address.local1029 = or disjoint i64 %address.local861, 96, !dbg !48
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1029, !dbg !48
  store half %1673, ptr addrspace(1) %address.local1030, align 2, !dbg !48
  %address.local1032 = or disjoint i64 %address.local861, 97, !dbg !48
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1032, !dbg !48
  store half %1672, ptr addrspace(1) %address.local1033, align 2, !dbg !48
  %address.local1035 = or disjoint i64 %address.local861, 98, !dbg !48
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1035, !dbg !48
  store half %1671, ptr addrspace(1) %address.local1036, align 2, !dbg !48
  %address.local1038 = or disjoint i64 %address.local861, 99, !dbg !48
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1038, !dbg !48
  store half %1670, ptr addrspace(1) %address.local1039, align 2, !dbg !48
  %address.local1041 = or disjoint i64 %address.local861, 112, !dbg !48
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1041, !dbg !48
  store half %1667, ptr addrspace(1) %address.local1042, align 2, !dbg !48
  %address.local1044 = or disjoint i64 %address.local861, 113, !dbg !48
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1044, !dbg !48
  store half %1666, ptr addrspace(1) %address.local1045, align 2, !dbg !48
  %address.local1047 = or disjoint i64 %address.local861, 114, !dbg !48
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1047, !dbg !48
  store half %1665, ptr addrspace(1) %address.local1048, align 2, !dbg !48
  %address.local1050 = or i64 %1600, 115, !dbg !48
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1050, !dbg !48
  store half %1664, ptr addrspace(1) %address.local1051, align 2, !dbg !48
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1029, !dbg !48
  store half %1661, ptr addrspace(1) %address.local1056, align 2, !dbg !48
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1032, !dbg !48
  store half %1660, ptr addrspace(1) %address.local1058, align 2, !dbg !48
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1035, !dbg !48
  store half %1659, ptr addrspace(1) %address.local1060, align 2, !dbg !48
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1038, !dbg !48
  store half %1658, ptr addrspace(1) %address.local1062, align 2, !dbg !48
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1041, !dbg !48
  store half %1655, ptr addrspace(1) %address.local1064, align 2, !dbg !48
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1044, !dbg !48
  store half %1654, ptr addrspace(1) %address.local1066, align 2, !dbg !48
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1047, !dbg !48
  store half %1653, ptr addrspace(1) %address.local1068, align 2, !dbg !48
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1050, !dbg !48
  store half %1652, ptr addrspace(1) %address.local1070, align 2, !dbg !48
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_thread_multichain")
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
