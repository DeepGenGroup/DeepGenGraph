; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/noncausal/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/noncausal/input.mlir"

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
  %address.local265 = zext nneg i32 %37 to i64, !dbg !9
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local265, !dbg !9
  %address.local269 = lshr i32 %address.thread, 4, !dbg !9
  %address.local270 = and i32 %address.local269, 8, !dbg !9
  %address.local271 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local272 = and i32 %address.local271, 16256, !dbg !9
  %address.local273 = or disjoint i32 %address.local270, %address.local272, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1239 = shl i64 %34, 8, !dbg !9
  %39 = add i64 %.idx1239, %.idx, !dbg !9
  %40 = shl nuw nsw i32 %address.local273, 1, !dbg !9
  %41 = trunc i64 %39 to i32, !dbg !9
  %dma.buffer.offset1237 = or disjoint i32 %40, %41, !dbg !9
  %42 = insertelement <2 x i64> poison, i64 %38, i64 0, !dbg !9
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1238 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local278 = or disjoint i64 %address.local265, 2048, !dbg !9
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !9
  %address.local287 = or disjoint i32 %address.local273, 16, !dbg !9
  %44 = shl nuw nsw i32 %address.local287, 1, !dbg !9
  %dma.buffer.offset1235 = or disjoint i32 %44, %41, !dbg !9
  %45 = shl nuw nsw i64 %31, 19, !dbg !9
  %46 = add i64 %35, %45, !dbg !9
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local265, !dbg !9
  %47 = trunc i64 %46 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local273, %47, !dbg !9
  %48 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1233 = or disjoint i32 %48, 64, !dbg !9
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local278, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local266, i64 0, !dbg !9
  %50 = insertelement <4 x ptr addrspace(3)> %49, ptr addrspace(3) %address.local279, i64 1, !dbg !9
  %51 = insertelement <4 x ptr addrspace(3)> %50, ptr addrspace(3) %address.local290, i64 2, !dbg !9
  %52 = insertelement <4 x ptr addrspace(3)> %51, ptr addrspace(3) %address.local294, i64 3, !dbg !9
  %53 = ptrtoint <4 x ptr addrspace(3)> %52 to <4 x i64>, !dbg !9
  %54 = trunc <4 x i64> %53 to <4 x i32>, !dbg !9
  %55 = extractelement <4 x i32> %54, i64 0, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1237, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %58 = extractelement <4 x i32> %54, i64 1, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1235, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %61 = extractelement <4 x i32> %54, i64 2, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1233, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %64 = extractelement <4 x i32> %54, i64 3, !dbg !9
  %65 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !9
  %.tr1243 = or disjoint i32 %address.local287, %47, !dbg !9
  %66 = shl i32 %.tr1243, 1, !dbg !9
  %dma.buffer.offset1231 = or disjoint i32 %66, 64, !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1231, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %address.local298 = lshr i32 %address.thread, 2, !dbg !11
  %address.local299 = and i32 %address.local298, 4, !dbg !11
  %address.local300 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local301 = and i32 %address.local300, 384, !dbg !11
  %address.local302 = or disjoint i32 %address.local299, %address.local301, !dbg !11
  %address.local304 = and i32 %36, 120, !dbg !11
  %address.local305 = or disjoint i32 %address.local302, %address.local304, !dbg !11
  %address.local306 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local307 = and i32 %address.local306, 1024, !dbg !11
  %address.local308 = or disjoint i32 %address.local305, %address.local307, !dbg !11
  %address.local309 = zext nneg i32 %address.local308 to i64, !dbg !11
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1060 = getelementptr i8, ptr addrspace(3) %address.local310, i64 1024, !dbg !11
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1060, align 8, !dbg !11
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1061 = getelementptr i8, ptr addrspace(3) %address.local310, i64 5120, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1061, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !11
  %75 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1229 = or disjoint i32 %48, 128, !dbg !9
  %76 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1229, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %dma.buffer.offset1227 = or disjoint i32 %66, 128, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1227, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %lds.const.gep1062 = getelementptr i8, ptr addrspace(3) %address.local310, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1062, align 8, !dbg !11
  %lds.const.gep1063 = getelementptr i8, ptr addrspace(3) %address.local310, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1063, align 8, !dbg !11
  %lds.const.gep1064 = getelementptr i8, ptr addrspace(3) %address.local310, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1065 = getelementptr i8, ptr addrspace(3) %address.local310, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1225 = or disjoint i32 %48, 192, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1225, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %dma.buffer.offset1223 = or disjoint i32 %66, 192, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1223, <4 x i32> %dma.buffer.rsrc1238) #9, !dbg !9
  %88 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1060, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1061, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1062, align 8, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1063, align 8, !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !11
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !11
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
  %k0.prime.1906 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1345 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1346 = shl nuw nsw i64 %k0.prime.kdma.source1345, 9, !dbg !14
  %k0.prime.kdma.source1347 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1348 = add nuw nsw i64 %k0.prime.kdma.source1346, %k0.prime.kdma.source1347, !dbg !14
  %k0.prime.kdma.source1349 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1350 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1351 = and i64 %k0.prime.kdma.source1350, 1, !dbg !14
  %k0.prime.kdma.source1352 = and i64 %k0.prime.kdma.source1351, %k0.prime.kdma.source1345, !dbg !14
  %k0.prime.kdma.source1353 = or disjoint i64 %k0.prime.kdma.source1352, %k0.prime.kdma.source1349, !dbg !14
  %110 = shl nuw nsw i64 %k0.prime.kdma.source1353, 10, !dbg !14
  %k0.prime.kdma.source1355 = sub nsw i64 %k0.prime.kdma.source1348, %110, !dbg !14
  %k0.prime.kdma.source1356 = and i64 %k0.prime.kdma.source1355, 16256, !dbg !14
  %k0.prime.kdma.source1357 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1358 = and i32 %k0.prime.kdma.source1357, 8, !dbg !14
  %k0.prime.kdma.source1359 = zext nneg i32 %k0.prime.kdma.source1358 to i64, !dbg !14
  %k0.prime.kdma.source1360 = or disjoint i64 %k0.prime.kdma.source1356, %k0.prime.kdma.source1359, !dbg !14
  %k0.prime.kdma.source1361 = or disjoint i64 %k0.prime.kdma.source1360, %k0.prime.1906, !dbg !14
  %k0.prime.kdma.source1362 = trunc nuw nsw i64 %k0.prime.kdma.source1361 to i32, !dbg !14
  %k0.prime.kdma.source1363 = shl nuw nsw i32 %k0.prime.kdma.source1362, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1363, %kdma.intra, !dbg !14
  %k0.prime.1909 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.1910 = insertelement <2 x i64> poison, i64 %k0.prime.1909, i64 0, !dbg !14
  %k0.prime.1911 = bitcast <2 x i64> %k0.prime.1910 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1222 = shufflevector <4 x i32> %k0.prime.1911, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.1912 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1385 = add i32 %56, 512, !dbg !14
  %k0.prime.kdma.source1364 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1365 = lshr i64 %k0.prime.kdma.source1364, 4, !dbg !14
  %k0.prime.kdma.source1366 = shl nuw nsw i64 %k0.prime.kdma.source1365, 9, !dbg !14
  %k0.prime.kdma.source1367 = shl nuw nsw i64 %k0.prime.kdma.source1364, 7, !dbg !14
  %k0.prime.kdma.source1368 = add nuw nsw i64 %k0.prime.kdma.source1366, %k0.prime.kdma.source1367, !dbg !14
  %k0.prime.kdma.source1369 = lshr i64 %k0.prime.kdma.source1364, 5, !dbg !14
  %k0.prime.kdma.source1370 = lshr i64 %k0.prime.kdma.source1364, 2, !dbg !14
  %k0.prime.kdma.source1371 = and i64 %k0.prime.kdma.source1370, 1, !dbg !14
  %k0.prime.kdma.source1372 = and i64 %k0.prime.kdma.source1371, %k0.prime.kdma.source1365, !dbg !14
  %k0.prime.kdma.source1373 = add nuw nsw i64 %k0.prime.kdma.source1372, %k0.prime.kdma.source1369, !dbg !14
  %111 = shl nuw nsw i64 %k0.prime.kdma.source1373, 10, !dbg !14
  %k0.prime.kdma.source1375 = sub nsw i64 %k0.prime.kdma.source1368, %111, !dbg !14
  %k0.prime.kdma.source1376 = and i64 %k0.prime.kdma.source1375, 16256, !dbg !14
  %k0.prime.kdma.source1380 = or disjoint i64 %k0.prime.kdma.source1376, %k0.prime.kdma.source1359, !dbg !14
  %k0.prime.kdma.source1381 = or disjoint i64 %k0.prime.kdma.source1380, %k0.prime.1906, !dbg !14
  %k0.prime.kdma.source1382 = trunc nuw nsw i64 %k0.prime.kdma.source1381 to i32, !dbg !14
  %k0.prime.kdma.source1383 = shl nuw nsw i32 %k0.prime.kdma.source1382, 1, !dbg !14
  %k0.prime.kdma.offset1384 = or disjoint i32 %k0.prime.kdma.source1383, %kdma.intra.hi, !dbg !14
  %k0.prime.1913 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1385, i32 %k0.prime.kdma.offset1384, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1387 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1388 = lshr i64 %k0.prime.kdma.source1387, 4, !dbg !14
  %k0.prime.kdma.source1389 = shl nuw nsw i64 %k0.prime.kdma.source1388, 9, !dbg !14
  %k0.prime.kdma.source1391 = add nuw nsw i64 %k0.prime.kdma.source1389, %k0.prime.kdma.source1347, !dbg !14
  %k0.prime.kdma.source1392 = lshr i64 %k0.prime.kdma.source1387, 5, !dbg !14
  %k0.prime.kdma.source1395 = and i64 %k0.prime.kdma.source1388, %k0.prime.kdma.source1351, !dbg !14
  %k0.prime.kdma.source1396 = or disjoint i64 %k0.prime.kdma.source1395, %k0.prime.kdma.source1392, !dbg !14
  %112 = shl nuw nsw i64 %k0.prime.kdma.source1396, 10, !dbg !14
  %k0.prime.kdma.source1398 = sub nsw i64 %k0.prime.kdma.source1391, %112, !dbg !14
  %k0.prime.kdma.source1399 = and i64 %k0.prime.kdma.source1398, 16256, !dbg !14
  %k0.prime.kdma.source1402 = or disjoint i32 %k0.prime.kdma.source1358, 16, !dbg !14
  %k0.prime.kdma.source1403 = zext nneg i32 %k0.prime.kdma.source1402 to i64, !dbg !14
  %k0.prime.kdma.source1404 = or disjoint i64 %k0.prime.kdma.source1399, %k0.prime.kdma.source1403, !dbg !14
  %k0.prime.kdma.source1405 = or disjoint i64 %k0.prime.kdma.source1404, %k0.prime.1906, !dbg !14
  %k0.prime.kdma.source1406 = trunc nuw nsw i64 %k0.prime.kdma.source1405 to i32, !dbg !14
  %k0.prime.kdma.source1407 = shl nuw nsw i32 %k0.prime.kdma.source1406, 1, !dbg !14
  %k0.prime.kdma.offset1408 = add nuw nsw i32 %k0.prime.kdma.source1407, %kdma.intra, !dbg !14
  %k0.prime.1914 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.prime.kdma.offset1408, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1433 = add i32 %59, 512, !dbg !14
  %k0.prime.kdma.source1411 = or disjoint i64 %k0.prime.kdma.source1364, 256, !dbg !14
  %k0.prime.kdma.source1412 = lshr i64 %k0.prime.kdma.source1411, 4, !dbg !14
  %k0.prime.kdma.source1413 = shl nuw nsw i64 %k0.prime.kdma.source1412, 9, !dbg !14
  %k0.prime.kdma.source1415 = add nuw nsw i64 %k0.prime.kdma.source1413, %k0.prime.kdma.source1367, !dbg !14
  %k0.prime.kdma.source1416 = lshr i64 %k0.prime.kdma.source1411, 5, !dbg !14
  %k0.prime.kdma.source1419 = and i64 %k0.prime.kdma.source1412, %k0.prime.kdma.source1371, !dbg !14
  %k0.prime.kdma.source1420 = add nuw nsw i64 %k0.prime.kdma.source1419, %k0.prime.kdma.source1416, !dbg !14
  %113 = shl nuw nsw i64 %k0.prime.kdma.source1420, 10, !dbg !14
  %k0.prime.kdma.source1422 = sub nsw i64 %k0.prime.kdma.source1415, %113, !dbg !14
  %k0.prime.kdma.source1423 = and i64 %k0.prime.kdma.source1422, 16256, !dbg !14
  %k0.prime.kdma.source1428 = or disjoint i64 %k0.prime.kdma.source1423, %k0.prime.kdma.source1403, !dbg !14
  %k0.prime.kdma.source1429 = or disjoint i64 %k0.prime.kdma.source1428, %k0.prime.1906, !dbg !14
  %k0.prime.kdma.source1430 = trunc nuw nsw i64 %k0.prime.kdma.source1429 to i32, !dbg !14
  %k0.prime.kdma.source1431 = shl nuw nsw i32 %k0.prime.kdma.source1430, 1, !dbg !14
  %k0.prime.kdma.offset1432 = add nuw nsw i32 %k0.prime.kdma.source1431, %kdma.intra.hi, !dbg !14
  %k0.prime.1915 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1433, i32 %k0.prime.kdma.offset1432, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1477 = add i32 %62, 512
  %kdma.lds1525 = add i32 %65, 512
  %114 = ptrtoint ptr addrspace(1) %2 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1206 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %56, 8192
  %vdb.dma.base1299 = add i32 %59, 8192
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
  %126 = add nuw nsw i64 %125, %k0.prime.1906, !dbg !14
  %address.thread361 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local362 = zext nneg i32 %address.thread361 to i64, !dbg !9
  %address.local368 = lshr i64 %address.local362, 2, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %127 = or disjoint i64 %126, 32, !dbg !14
  %kdma.source1451 = or disjoint i64 %127, %k0.prime.kdma.source1360, !dbg !14
  %kdma.source1452 = trunc nuw nsw i64 %kdma.source1451 to i32, !dbg !14
  %kdma.source1453 = shl nuw nsw i32 %kdma.source1452, 1, !dbg !14
  %kdma.offset1454 = or disjoint i32 %kdma.source1453, %kdma.intra, !dbg !14
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1454, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1473 = or disjoint i64 %127, %k0.prime.kdma.source1380, !dbg !14
  %kdma.source1474 = trunc nuw nsw i64 %kdma.source1473 to i32, !dbg !14
  %kdma.source1475 = shl nuw nsw i32 %kdma.source1474, 1, !dbg !14
  %kdma.offset1476 = or disjoint i32 %kdma.source1475, %kdma.intra.hi, !dbg !14
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1476, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1497 = or disjoint i64 %127, %k0.prime.kdma.source1404, !dbg !14
  %kdma.source1498 = trunc nuw nsw i64 %kdma.source1497 to i32, !dbg !14
  %kdma.source1499 = shl nuw nsw i32 %kdma.source1498, 1, !dbg !14
  %kdma.offset1500 = add nuw nsw i32 %kdma.source1499, %kdma.intra, !dbg !14
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1500, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1521 = or disjoint i64 %127, %k0.prime.kdma.source1428, !dbg !14
  %kdma.source1522 = trunc nuw nsw i64 %kdma.source1521 to i32, !dbg !14
  %kdma.source1523 = shl nuw nsw i32 %kdma.source1522, 1, !dbg !14
  %kdma.offset1524 = add nuw nsw i32 %kdma.source1523, %kdma.intra.hi, !dbg !14
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1524, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %address.local405 = shl nuw nsw i64 %address.local362, 1, !dbg !13
  %address.local406 = and i64 %address.local405, 24, !dbg !13
  %address.local408 = shl i64 %address.local362, 62, !dbg !13
  %address.local409 = ashr exact i64 %address.local408, 57, !dbg !13
  %address.local410 = and i64 %address.local409, -64, !dbg !13
  %address.local413 = and i64 %address.local368, 12, !dbg !11
  %address.local415 = and i64 %address.local362, 63, !dbg !11
  %address.local416 = lshr i64 %address.local415, 5, !dbg !11
  %address.local417 = mul nuw nsw i64 %address.local416, 1016, !dbg !11
  %.idx1260 = mul nuw nsw i64 %address.local362, 320, !dbg !13
  %132 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1260, !dbg !13
  %.idx1261 = mul nsw i64 %address.local368, -1280, !dbg !13
  %133 = getelementptr i8, ptr addrspace(3) %132, i64 %.idx1261, !dbg !13
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local406, !dbg !13
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local410, !dbg !13
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local413, !dbg !13
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local417, !dbg !13
  %137 = add nsw i64 %.idx1261, %.idx1260, !dbg !13
  %.idx1836 = shl nuw nsw i64 %address.local406, 1, !dbg !13
  %138 = or disjoint i64 %137, %.idx1836, !dbg !13
  %139 = or disjoint i64 %address.local410, %address.local413, !dbg !13
  %140 = shl nsw i64 %139, 1, !dbg !13
  %address.local419.idx = mul nuw nsw i64 %address.local416, 2032, !dbg !13
  %141 = add nsw i64 %138, %address.local419.idx, !dbg !13
  %142 = add nsw i64 %141, %140, !dbg !13
  %143 = lshr i64 %142, 6, !dbg !13
  %144 = and i64 %143, 8, !dbg !13
  %145 = xor i64 %144, %142, !dbg !13
  %146 = sub i64 %145, %142, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local419, i64 %146, !dbg !13
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %121, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base1921 = extractvalue [2 x <1 x float>] %121, 1, !dbg !13
  %lazy.seed1922 = shufflevector <1 x float> %lazy.base1921, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %147, <4 x float> %lazy.seed) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local430 = shl nuw nsw i32 %address.thread361, 1, !dbg !13
  %address.local431 = and i32 %address.local430, 24, !dbg !13
  %address.local432 = shl nuw nsw i32 %address.thread361, 8, !dbg !13
  %address.local433 = and i32 %address.local432, 768, !dbg !13
  %address.local434 = or disjoint i32 %address.local431, %address.local433, !dbg !13
  %address.local435 = shl nuw nsw i32 %address.thread361, 5, !dbg !13
  %address.local436 = and i32 %address.local435, 96, !dbg !13
  %address.local437 = or disjoint i32 %address.local434, %address.local436, !dbg !13
  %address.local438 = zext nneg i32 %address.local437 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %109) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local439 = lshr i32 %address.thread361, 1, !dbg !13
  %address.local440 = and i32 %address.local439, 1, !dbg !13
  %address.local441 = zext nneg i32 %address.local440 to i64, !dbg !13
  %address.local442 = add nsw i64 %address.base, %address.local441, !dbg !13
  %address.local443 = shl i64 %address.local442, 6, !dbg !13
  %address.local444 = sub i64 %address.local438, %address.local443, !dbg !13
  %150 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local444, !dbg !13
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local413, !dbg !13
  %152 = getelementptr [2 x i8], ptr addrspace(3) %151, i64 %address.local417, !dbg !13
  %address.local448 = getelementptr i8, ptr addrspace(3) %152, i64 64, !dbg !13
  %.idx18391842 = add i64 %address.local444, %address.local413, !dbg !13
  %153 = shl i64 %.idx18391842, 1, !dbg !13
  %154 = add nuw nsw i64 %address.local419.idx, 64, !dbg !13
  %155 = add i64 %154, %153, !dbg !13
  %156 = lshr i64 %155, 6, !dbg !13
  %157 = and i64 %156, 8, !dbg !13
  %158 = xor i64 %157, %155, !dbg !13
  %159 = sub i64 %158, %155, !dbg !13
  %kdma.swizzled1710 = getelementptr i8, ptr addrspace(3) %address.local448, i64 %159, !dbg !13
  %160 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1710, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1714 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 128, !dbg !13
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1714, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1711 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 4096, !dbg !13
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1711, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1715 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 4224, !dbg !13
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1715, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1712 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1712, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1716 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1716, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1713 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1713, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1717 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %167 = load <4 x half>, ptr addrspace(3) %kdma.gep1717, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %148) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1718 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 1024, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1718, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1722 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 1152, !dbg !13
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %160, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1719 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 5120, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1719, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1723 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 5248, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %171) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1720 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1720, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1721 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %176 = load <4 x half>, ptr addrspace(3) %kdma.gep1721, align 8, !dbg !13, !frisk.k_lds_pair !21
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
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %147, <4 x float> %lazy.seed1922) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %149, <4 x float> %189) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %160, <4 x float> %lazy.seed1922) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %162, <4 x float> %191) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %164, <4 x float> %lazy.seed1922) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %166, <4 x float> %193) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %161, <4 x float> %lazy.seed1922) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %163, <4 x float> %195) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %165, <4 x float> %lazy.seed1922) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %167, <4 x float> %197) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %169, <4 x float> %lazy.seed1922) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %172, <4 x float> %199) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %175, <4 x float> %lazy.seed1922) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %176, <4 x float> %201) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %170, <4 x float> %lazy.seed1922) #8, !dbg !13
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %173, <4 x float> %203) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %205 = or disjoint i64 %126, 64, !dbg !14
  %kdma.source1543 = or disjoint i64 %205, %k0.prime.kdma.source1360, !dbg !14
  %kdma.source1544 = trunc nuw nsw i64 %kdma.source1543 to i32, !dbg !14
  %kdma.source1545 = shl nuw nsw i32 %kdma.source1544, 1, !dbg !14
  %kdma.offset1546 = or disjoint i32 %kdma.source1545, %kdma.intra, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1546, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1565 = or disjoint i64 %205, %k0.prime.kdma.source1380, !dbg !14
  %kdma.source1566 = trunc nuw nsw i64 %kdma.source1565 to i32, !dbg !14
  %kdma.source1567 = shl nuw nsw i32 %kdma.source1566, 1, !dbg !14
  %kdma.offset1568 = or disjoint i32 %kdma.source1567, %kdma.intra.hi, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1385, i32 %kdma.offset1568, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1589 = or disjoint i64 %205, %k0.prime.kdma.source1404, !dbg !14
  %kdma.source1590 = trunc nuw nsw i64 %kdma.source1589 to i32, !dbg !14
  %kdma.source1591 = shl nuw nsw i32 %kdma.source1590, 1, !dbg !14
  %kdma.offset1592 = add nuw nsw i32 %kdma.source1591, %kdma.intra, !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1592, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1613 = or disjoint i64 %205, %k0.prime.kdma.source1428, !dbg !14
  %kdma.source1614 = trunc nuw nsw i64 %kdma.source1613 to i32, !dbg !14
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !14
  %kdma.offset1616 = add nuw nsw i32 %kdma.source1615, %kdma.intra.hi, !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1433, i32 %kdma.offset1616, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1744 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1744, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1741 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1745 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1745, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1742 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 8192, !dbg !13
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1746 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 8320, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1746, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1743 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 12288, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1743, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1747 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 12416, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1747, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1748 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1748, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1752 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %210, <4 x float> %168) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1749 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1749, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1753 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %212, <4 x float> %220) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1750 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 9216, !dbg !13
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1750, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1754 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 9344, !dbg !13
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %214, <4 x float> %174) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1751 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 13312, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1755 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 13440, !dbg !13
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !13, !frisk.k_lds_pair !29
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
  %kdma.source1635 = or disjoint i64 %258, %k0.prime.kdma.source1360, !dbg !14
  %kdma.source1636 = trunc nuw nsw i64 %kdma.source1635 to i32, !dbg !14
  %kdma.source1637 = shl nuw nsw i32 %kdma.source1636, 1, !dbg !14
  %kdma.offset1638 = or disjoint i32 %kdma.source1637, %kdma.intra, !dbg !14
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1638, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1657 = or disjoint i64 %258, %k0.prime.kdma.source1380, !dbg !14
  %kdma.source1658 = trunc nuw nsw i64 %kdma.source1657 to i32, !dbg !14
  %kdma.source1659 = shl nuw nsw i32 %kdma.source1658, 1, !dbg !14
  %kdma.offset1660 = or disjoint i32 %kdma.source1659, %kdma.intra.hi, !dbg !14
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1660, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1681 = or disjoint i64 %258, %k0.prime.kdma.source1404, !dbg !14
  %kdma.source1682 = trunc nuw nsw i64 %kdma.source1681 to i32, !dbg !14
  %kdma.source1683 = shl nuw nsw i32 %kdma.source1682, 1, !dbg !14
  %kdma.offset1684 = add nuw nsw i32 %kdma.source1683, %kdma.intra, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1684, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %kdma.source1705 = or disjoint i64 %258, %k0.prime.kdma.source1428, !dbg !14
  %kdma.source1706 = trunc nuw nsw i64 %kdma.source1705 to i32, !dbg !14
  %kdma.source1707 = shl nuw nsw i32 %kdma.source1706, 1, !dbg !14
  %kdma.offset1708 = add nuw nsw i32 %kdma.source1707, %kdma.intra.hi, !dbg !14
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1708, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1712, align 8, !dbg !13, !frisk.k_lds_pair !30
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1713, align 8, !dbg !13, !frisk.k_lds_pair !31
  %267 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1710, align 8, !dbg !13, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1714, align 8, !dbg !13, !frisk.k_lds_pair !32
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1711, align 8, !dbg !13, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1715, align 8, !dbg !13, !frisk.k_lds_pair !33
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1716, align 8, !dbg !13, !frisk.k_lds_pair !34
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1720, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %263, <4 x float> %223) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1717, align 8, !dbg !13, !frisk.k_lds_pair !35
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1721, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %265, <4 x float> %273) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1718, align 8, !dbg !13, !frisk.k_lds_pair !36
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %267, <4 x float> %229) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1719, align 8, !dbg !13, !frisk.k_lds_pair !37
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !13, !frisk.k_lds_pair !37
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
  %address.local777 = shl nuw nsw i64 %address.local362, 5, !dbg !38
  %address.local778 = and i64 %address.local777, 8064, !dbg !38
  %address.local779 = shl nuw nsw i64 %address.local362, 3, !dbg !9
  %address.local780 = shl nuw nsw i64 %address.local368, 5, !dbg !38
  %address.local781 = sub nsw i64 %address.local779, %address.local780, !dbg !38
  %address.local782 = add nsw i64 %address.local781, %address.local778, !dbg !38
  %.idx12701271 = add nsw i64 %address.local782, %126, !dbg !38
  %.idx12701271.tr = trunc nsw i64 %.idx12701271 to i32, !dbg !38
  %dma.buffer.offset1205 = shl nsw i32 %.idx12701271.tr, 1, !dbg !38
  %311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1205, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %address.local784 = shl nuw nsw i32 %address.thread361, 3, !dbg !38
  %address.local785 = and i32 %address.local784, 24, !dbg !38
  %address.local787 = and i32 %address.local435, 8064, !dbg !38
  %address.local788 = or disjoint i32 %address.local787, %address.local785, !dbg !38
  %address.local789 = or disjoint i32 %address.local788, 8192, !dbg !38
  %312 = trunc nuw nsw i64 %126 to i32, !dbg !38
  %.idx12721273.tr = or disjoint i32 %address.local789, %312, !dbg !38
  %dma.buffer.offset1203 = shl nuw nsw i32 %.idx12721273.tr, 1, !dbg !38
  %313 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1203, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !13, !frisk.k_tail_pair !15
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1744, align 8, !dbg !13, !frisk.k_tail_pair !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !13, !frisk.k_tail_pair !16
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1745, align 8, !dbg !13, !frisk.k_tail_pair !16
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !13, !frisk.k_tail_pair !17
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1746, align 8, !dbg !13, !frisk.k_tail_pair !17
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1743, align 8, !dbg !13, !frisk.k_tail_pair !18
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1747, align 8, !dbg !13, !frisk.k_tail_pair !18
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
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1748, align 8, !dbg !13, !frisk.k_tail_pair !19
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !13, !frisk.k_tail_pair !19
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
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1749, align 8, !dbg !13, !frisk.k_tail_pair !20
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %317, <4 x float> %333) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1750, align 8, !dbg !13, !frisk.k_tail_pair !21
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !13, !frisk.k_tail_pair !21
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
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !13, !frisk.k_tail_pair !22
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !13, !frisk.k_tail_pair !22
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
  %lazy.positive1923 = tail call float @llvm.maxnum.f32(float %505, float 0.000000e+00), !dbg !39
  %lazy.delta1924 = select i1 %lazy.first, float %505, float %lazy.positive1923, !dbg !39
  %lazy.negated1925 = fsub float 0.000000e+00, %lazy.delta1924, !dbg !39
  %lazy.exp1926 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated1925), !dbg !39
  %lazy.alpha1927 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp1926, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %323, %lazy.delta.v, !dbg !39
  %lazy.shifted1930 = fsub <4 x float> %330, %lazy.delta.v, !dbg !39
  %lazy.shifted1933 = fsub <4 x float> %344, %lazy.delta.v, !dbg !39
  %lazy.shifted1936 = fsub <4 x float> %358, %lazy.delta.v, !dbg !39
  %lazy.shifted1939 = fsub <4 x float> %368, %lazy.delta.v, !dbg !39
  %lazy.shifted1942 = fsub <4 x float> %378, %lazy.delta.v, !dbg !39
  %lazy.shifted1945 = fsub <4 x float> %388, %lazy.delta.v, !dbg !39
  %lazy.shifted1948 = fsub <4 x float> %398, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane1949 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !39
  %lazy.delta.v1950 = shufflevector <1 x float> %lazy.delta.v.lane1949, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted1951 = fsub <4 x float> %411, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1954 = fsub <4 x float> %417, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1957 = fsub <4 x float> %423, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1960 = fsub <4 x float> %429, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1963 = fsub <4 x float> %435, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1966 = fsub <4 x float> %441, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1969 = fsub <4 x float> %447, %lazy.delta.v1950, !dbg !39
  %lazy.shifted1972 = fsub <4 x float> %453, %lazy.delta.v1950, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %122, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state1973 = extractvalue [2 x <1 x float>] %122, 1, !dbg !39
  %lazy.alpha.v.lane1974 = insertelement <1 x float> poison, float %lazy.alpha1927, i64 0, !dbg !39
  %lazy.scaled1975 = fmul <1 x float> %lazy.state1973, %lazy.alpha.v.lane1974, !dbg !39
  %lazy.state.next1976 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1975, 1, !dbg !39
  %lazy.state1977 = extractvalue [2 x <8 x float>] %118, 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled1979 = fmul <8 x float> %lazy.state1977, %lazy.alpha.v, !dbg !39
  %lazy.state.next1980 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1979, 0, !dbg !39
  %lazy.state1981 = extractvalue [2 x <8 x float>] %118, 1, !dbg !39
  %lazy.alpha.v1983 = shufflevector <1 x float> %lazy.alpha.v.lane1974, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled1984 = fmul <8 x float> %lazy.state1981, %lazy.alpha.v1983, !dbg !39
  %lazy.state.next1985 = insertvalue [2 x <8 x float>] %lazy.state.next1980, <8 x float> %lazy.scaled1984, 1, !dbg !39
  %lazy.state1986 = extractvalue [2 x <8 x float>] %119, 0, !dbg !39
  %lazy.scaled1989 = fmul <8 x float> %lazy.state1986, %lazy.alpha.v, !dbg !39
  %lazy.state.next1990 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1989, 0, !dbg !39
  %lazy.state1991 = extractvalue [2 x <8 x float>] %119, 1, !dbg !39
  %lazy.scaled1994 = fmul <8 x float> %lazy.state1991, %lazy.alpha.v1983, !dbg !39
  %lazy.state.next1995 = insertvalue [2 x <8 x float>] %lazy.state.next1990, <8 x float> %lazy.scaled1994, 1, !dbg !39
  %lazy.state1996 = extractvalue [2 x <8 x float>] %120, 0, !dbg !39
  %lazy.scaled1999 = fmul <8 x float> %lazy.state1996, %lazy.alpha.v, !dbg !39
  %lazy.state.next2000 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1999, 0, !dbg !39
  %lazy.state2001 = extractvalue [2 x <8 x float>] %120, 1, !dbg !39
  %lazy.scaled2004 = fmul <8 x float> %lazy.state2001, %lazy.alpha.v1983, !dbg !39
  %lazy.state.next2005 = insertvalue [2 x <8 x float>] %lazy.state.next2000, <8 x float> %lazy.scaled2004, 1, !dbg !39
  %lazy.state2006 = extractvalue [2 x <8 x float>] %123, 0, !dbg !39
  %lazy.scaled2009 = fmul <8 x float> %lazy.state2006, %lazy.alpha.v, !dbg !39
  %lazy.state.next2010 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2009, 0, !dbg !39
  %lazy.state2011 = extractvalue [2 x <8 x float>] %123, 1, !dbg !39
  %lazy.scaled2014 = fmul <8 x float> %lazy.state2011, %lazy.alpha.v1983, !dbg !39
  %lazy.state.next2015 = insertvalue [2 x <8 x float>] %lazy.state.next2010, <8 x float> %lazy.scaled2014, 1, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.next2018 = fsub <1 x float> %lazy.base1921, %lazy.delta.v.lane1949, !dbg !39
  %lazy.base.state2019 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2018, 1, !dbg !39
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !40
  %.pre26 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !40
  %.pre28 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !40
  %.pre30 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !40
  %.pre32 = extractelement <4 x float> %lazy.shifted1951, i64 0, !dbg !40
  %.pre34 = extractelement <4 x float> %lazy.shifted1951, i64 1, !dbg !40
  %.pre36 = extractelement <4 x float> %lazy.shifted1951, i64 2, !dbg !40
  %.pre38 = extractelement <4 x float> %lazy.shifted1951, i64 3, !dbg !40
  %.pre40 = extractelement <4 x float> %lazy.shifted1930, i64 0, !dbg !40
  %.pre42 = extractelement <4 x float> %lazy.shifted1930, i64 1, !dbg !40
  %.pre44 = extractelement <4 x float> %lazy.shifted1930, i64 2, !dbg !40
  %.pre46 = extractelement <4 x float> %lazy.shifted1930, i64 3, !dbg !40
  %.pre48 = extractelement <4 x float> %lazy.shifted1954, i64 0, !dbg !40
  %.pre50 = extractelement <4 x float> %lazy.shifted1954, i64 1, !dbg !40
  %.pre52 = extractelement <4 x float> %lazy.shifted1954, i64 2, !dbg !40
  %.pre54 = extractelement <4 x float> %lazy.shifted1954, i64 3, !dbg !40
  %.pre56 = extractelement <4 x float> %lazy.shifted1933, i64 0, !dbg !40
  %.pre58 = extractelement <4 x float> %lazy.shifted1933, i64 1, !dbg !40
  %.pre60 = extractelement <4 x float> %lazy.shifted1933, i64 2, !dbg !40
  %.pre62 = extractelement <4 x float> %lazy.shifted1933, i64 3, !dbg !40
  %.pre64 = extractelement <4 x float> %lazy.shifted1957, i64 0, !dbg !40
  %.pre66 = extractelement <4 x float> %lazy.shifted1957, i64 1, !dbg !40
  %.pre68 = extractelement <4 x float> %lazy.shifted1957, i64 2, !dbg !40
  %.pre70 = extractelement <4 x float> %lazy.shifted1957, i64 3, !dbg !40
  %.pre72 = extractelement <4 x float> %lazy.shifted1936, i64 0, !dbg !40
  %.pre74 = extractelement <4 x float> %lazy.shifted1936, i64 1, !dbg !40
  %.pre76 = extractelement <4 x float> %lazy.shifted1936, i64 2, !dbg !40
  %.pre78 = extractelement <4 x float> %lazy.shifted1936, i64 3, !dbg !40
  %.pre80 = extractelement <4 x float> %lazy.shifted1960, i64 0, !dbg !40
  %.pre82 = extractelement <4 x float> %lazy.shifted1960, i64 1, !dbg !40
  %.pre84 = extractelement <4 x float> %lazy.shifted1960, i64 2, !dbg !40
  %.pre86 = extractelement <4 x float> %lazy.shifted1960, i64 3, !dbg !40
  %.pre88 = extractelement <4 x float> %lazy.shifted1939, i64 0, !dbg !40
  %.pre90 = extractelement <4 x float> %lazy.shifted1939, i64 1, !dbg !40
  %.pre92 = extractelement <4 x float> %lazy.shifted1939, i64 2, !dbg !40
  %.pre94 = extractelement <4 x float> %lazy.shifted1939, i64 3, !dbg !40
  %.pre96 = extractelement <4 x float> %lazy.shifted1963, i64 0, !dbg !40
  %.pre98 = extractelement <4 x float> %lazy.shifted1963, i64 1, !dbg !40
  %.pre100 = extractelement <4 x float> %lazy.shifted1963, i64 2, !dbg !40
  %.pre102 = extractelement <4 x float> %lazy.shifted1963, i64 3, !dbg !40
  %.pre104 = extractelement <4 x float> %lazy.shifted1942, i64 0, !dbg !40
  %.pre106 = extractelement <4 x float> %lazy.shifted1942, i64 1, !dbg !40
  %.pre108 = extractelement <4 x float> %lazy.shifted1942, i64 2, !dbg !40
  %.pre110 = extractelement <4 x float> %lazy.shifted1942, i64 3, !dbg !40
  %.pre112 = extractelement <4 x float> %lazy.shifted1966, i64 0, !dbg !40
  %.pre114 = extractelement <4 x float> %lazy.shifted1966, i64 1, !dbg !40
  %.pre116 = extractelement <4 x float> %lazy.shifted1966, i64 2, !dbg !40
  %.pre118 = extractelement <4 x float> %lazy.shifted1966, i64 3, !dbg !40
  %.pre120 = extractelement <4 x float> %lazy.shifted1945, i64 0, !dbg !40
  %.pre122 = extractelement <4 x float> %lazy.shifted1945, i64 1, !dbg !40
  %.pre124 = extractelement <4 x float> %lazy.shifted1945, i64 2, !dbg !40
  %.pre126 = extractelement <4 x float> %lazy.shifted1945, i64 3, !dbg !40
  %.pre128 = extractelement <4 x float> %lazy.shifted1969, i64 0, !dbg !40
  %.pre130 = extractelement <4 x float> %lazy.shifted1969, i64 1, !dbg !40
  %.pre132 = extractelement <4 x float> %lazy.shifted1969, i64 2, !dbg !40
  %.pre134 = extractelement <4 x float> %lazy.shifted1969, i64 3, !dbg !40
  %.pre136 = extractelement <4 x float> %lazy.shifted1948, i64 0, !dbg !40
  %.pre138 = extractelement <4 x float> %lazy.shifted1948, i64 1, !dbg !40
  %.pre140 = extractelement <4 x float> %lazy.shifted1948, i64 2, !dbg !40
  %.pre142 = extractelement <4 x float> %lazy.shifted1948, i64 3, !dbg !40
  %.pre144 = extractelement <4 x float> %lazy.shifted1972, i64 0, !dbg !40
  %.pre146 = extractelement <4 x float> %lazy.shifted1972, i64 1, !dbg !40
  %.pre148 = extractelement <4 x float> %lazy.shifted1972, i64 2, !dbg !40
  %.pre150 = extractelement <4 x float> %lazy.shifted1972, i64 3, !dbg !40
  br label %lazy.join, !dbg !39

lazy.join:                                        ; preds = %lazy.slow, %117
  %.pre-phi151 = phi float [ %.pre150, %lazy.slow ], [ %457, %117 ], !dbg !40
  %.pre-phi149 = phi float [ %.pre148, %lazy.slow ], [ %456, %117 ], !dbg !40
  %.pre-phi147 = phi float [ %.pre146, %lazy.slow ], [ %455, %117 ], !dbg !40
  %.pre-phi145 = phi float [ %.pre144, %lazy.slow ], [ %454, %117 ], !dbg !40
  %.pre-phi143 = phi float [ %.pre142, %lazy.slow ], [ %406, %117 ], !dbg !40
  %.pre-phi141 = phi float [ %.pre140, %lazy.slow ], [ %404, %117 ], !dbg !40
  %.pre-phi139 = phi float [ %.pre138, %lazy.slow ], [ %402, %117 ], !dbg !40
  %.pre-phi137 = phi float [ %.pre136, %lazy.slow ], [ %400, %117 ], !dbg !40
  %.pre-phi135 = phi float [ %.pre134, %lazy.slow ], [ %452, %117 ], !dbg !40
  %.pre-phi133 = phi float [ %.pre132, %lazy.slow ], [ %451, %117 ], !dbg !40
  %.pre-phi131 = phi float [ %.pre130, %lazy.slow ], [ %450, %117 ], !dbg !40
  %.pre-phi129 = phi float [ %.pre128, %lazy.slow ], [ %449, %117 ], !dbg !40
  %.pre-phi127 = phi float [ %.pre126, %lazy.slow ], [ %396, %117 ], !dbg !40
  %.pre-phi125 = phi float [ %.pre124, %lazy.slow ], [ %394, %117 ], !dbg !40
  %.pre-phi123 = phi float [ %.pre122, %lazy.slow ], [ %392, %117 ], !dbg !40
  %.pre-phi121 = phi float [ %.pre120, %lazy.slow ], [ %390, %117 ], !dbg !40
  %.pre-phi119 = phi float [ %.pre118, %lazy.slow ], [ %446, %117 ], !dbg !40
  %.pre-phi117 = phi float [ %.pre116, %lazy.slow ], [ %445, %117 ], !dbg !40
  %.pre-phi115 = phi float [ %.pre114, %lazy.slow ], [ %444, %117 ], !dbg !40
  %.pre-phi113 = phi float [ %.pre112, %lazy.slow ], [ %443, %117 ], !dbg !40
  %.pre-phi111 = phi float [ %.pre110, %lazy.slow ], [ %386, %117 ], !dbg !40
  %.pre-phi109 = phi float [ %.pre108, %lazy.slow ], [ %384, %117 ], !dbg !40
  %.pre-phi107 = phi float [ %.pre106, %lazy.slow ], [ %382, %117 ], !dbg !40
  %.pre-phi105 = phi float [ %.pre104, %lazy.slow ], [ %380, %117 ], !dbg !40
  %.pre-phi103 = phi float [ %.pre102, %lazy.slow ], [ %440, %117 ], !dbg !40
  %.pre-phi101 = phi float [ %.pre100, %lazy.slow ], [ %439, %117 ], !dbg !40
  %.pre-phi99 = phi float [ %.pre98, %lazy.slow ], [ %438, %117 ], !dbg !40
  %.pre-phi97 = phi float [ %.pre96, %lazy.slow ], [ %437, %117 ], !dbg !40
  %.pre-phi95 = phi float [ %.pre94, %lazy.slow ], [ %376, %117 ], !dbg !40
  %.pre-phi93 = phi float [ %.pre92, %lazy.slow ], [ %374, %117 ], !dbg !40
  %.pre-phi91 = phi float [ %.pre90, %lazy.slow ], [ %372, %117 ], !dbg !40
  %.pre-phi89 = phi float [ %.pre88, %lazy.slow ], [ %370, %117 ], !dbg !40
  %.pre-phi87 = phi float [ %.pre86, %lazy.slow ], [ %434, %117 ], !dbg !40
  %.pre-phi85 = phi float [ %.pre84, %lazy.slow ], [ %433, %117 ], !dbg !40
  %.pre-phi83 = phi float [ %.pre82, %lazy.slow ], [ %432, %117 ], !dbg !40
  %.pre-phi81 = phi float [ %.pre80, %lazy.slow ], [ %431, %117 ], !dbg !40
  %.pre-phi79 = phi float [ %.pre78, %lazy.slow ], [ %366, %117 ], !dbg !40
  %.pre-phi77 = phi float [ %.pre76, %lazy.slow ], [ %364, %117 ], !dbg !40
  %.pre-phi75 = phi float [ %.pre74, %lazy.slow ], [ %362, %117 ], !dbg !40
  %.pre-phi73 = phi float [ %.pre72, %lazy.slow ], [ %360, %117 ], !dbg !40
  %.pre-phi71 = phi float [ %.pre70, %lazy.slow ], [ %428, %117 ], !dbg !40
  %.pre-phi69 = phi float [ %.pre68, %lazy.slow ], [ %427, %117 ], !dbg !40
  %.pre-phi67 = phi float [ %.pre66, %lazy.slow ], [ %426, %117 ], !dbg !40
  %.pre-phi65 = phi float [ %.pre64, %lazy.slow ], [ %425, %117 ], !dbg !40
  %.pre-phi63 = phi float [ %.pre62, %lazy.slow ], [ %354, %117 ], !dbg !40
  %.pre-phi61 = phi float [ %.pre60, %lazy.slow ], [ %352, %117 ], !dbg !40
  %.pre-phi59 = phi float [ %.pre58, %lazy.slow ], [ %350, %117 ], !dbg !40
  %.pre-phi57 = phi float [ %.pre56, %lazy.slow ], [ %348, %117 ], !dbg !40
  %.pre-phi55 = phi float [ %.pre54, %lazy.slow ], [ %422, %117 ], !dbg !40
  %.pre-phi53 = phi float [ %.pre52, %lazy.slow ], [ %421, %117 ], !dbg !40
  %.pre-phi51 = phi float [ %.pre50, %lazy.slow ], [ %420, %117 ], !dbg !40
  %.pre-phi49 = phi float [ %.pre48, %lazy.slow ], [ %419, %117 ], !dbg !40
  %.pre-phi47 = phi float [ %.pre46, %lazy.slow ], [ %340, %117 ], !dbg !40
  %.pre-phi45 = phi float [ %.pre44, %lazy.slow ], [ %338, %117 ], !dbg !40
  %.pre-phi43 = phi float [ %.pre42, %lazy.slow ], [ %336, %117 ], !dbg !40
  %.pre-phi41 = phi float [ %.pre40, %lazy.slow ], [ %334, %117 ], !dbg !40
  %.pre-phi39 = phi float [ %.pre38, %lazy.slow ], [ %416, %117 ], !dbg !40
  %.pre-phi37 = phi float [ %.pre36, %lazy.slow ], [ %415, %117 ], !dbg !40
  %.pre-phi35 = phi float [ %.pre34, %lazy.slow ], [ %414, %117 ], !dbg !40
  %.pre-phi33 = phi float [ %.pre32, %lazy.slow ], [ %413, %117 ], !dbg !40
  %.pre-phi31 = phi float [ %.pre30, %lazy.slow ], [ %329, %117 ], !dbg !40
  %.pre-phi29 = phi float [ %.pre28, %lazy.slow ], [ %328, %117 ], !dbg !40
  %.pre-phi27 = phi float [ %.pre26, %lazy.slow ], [ %327, %117 ], !dbg !40
  %.pre-phi = phi float [ %.pre, %lazy.slow ], [ %325, %117 ], !dbg !40
  %lazy.state2035 = phi [2 x <1 x float>] [ %lazy.state.next1976, %lazy.slow ], [ %122, %117 ], !dbg !39
  %lazy.state2036 = phi [2 x <8 x float>] [ %lazy.state.next1985, %lazy.slow ], [ %118, %117 ], !dbg !39
  %lazy.state2037 = phi [2 x <8 x float>] [ %lazy.state.next1995, %lazy.slow ], [ %119, %117 ], !dbg !39
  %lazy.state2038 = phi [2 x <8 x float>] [ %lazy.state.next2005, %lazy.slow ], [ %120, %117 ], !dbg !39
  %lazy.state2039 = phi [2 x <8 x float>] [ %lazy.state.next2015, %lazy.slow ], [ %123, %117 ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2019, %lazy.slow ], [ %121, %117 ], !dbg !39
  %506 = extractvalue [2 x <1 x float>] %lazy.state2035, 0, !dbg !39
  %507 = extractelement <1 x float> %506, i64 0, !dbg !39
  %508 = extractvalue [2 x <1 x float>] %lazy.state2035, 1, !dbg !39
  %509 = extractelement <1 x float> %508, i64 0, !dbg !39
  %510 = extractvalue [2 x <8 x float>] %lazy.state2039, 0, !dbg !41
  %511 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %512 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %513 = extractvalue [2 x <8 x float>] %lazy.state2039, 1, !dbg !41
  %514 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %515 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address = add nsw i64 %address.local782, %127, !dbg !38
  %vdb.address1286 = trunc nsw i64 %vdb.address to i32, !dbg !38
  %vdb.address1287 = shl nsw i32 %vdb.address1286, 1, !dbg !38
  %vdb.address1292 = trunc nuw nsw i64 %127 to i32, !dbg !38
  %vdb.address1293 = or disjoint i32 %address.local789, %vdb.address1292, !dbg !38
  %vdb.address1294 = shl nuw nsw i32 %vdb.address1293, 1, !dbg !38
  %516 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1287, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %517 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1299, i32 %vdb.address1294, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %address.local793 = and i32 %address.local784, 120, !dbg !42
  %address.local795 = and i32 %address.local435, 1536, !dbg !42
  %address.local796 = or disjoint i32 %address.local793, %address.local795, !dbg !42
  %address.local797 = zext nneg i32 %address.local796 to i64, !dbg !42
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !42
  %518 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local798, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %519 = bitcast <4 x i32> %518 to <8 x half>, !dbg !42
  %address.local799 = and i64 %address.local362, 60, !dbg !42
  %address.local800 = add nuw nsw i64 %address.local799, %address.local362, !dbg !42
  %address.local802.idx = shl nuw nsw i64 %address.local800, 4, !dbg !42
  %address.local802 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local802.idx, !dbg !42
  %address.local803 = lshr i64 %address.local415, 4, !dbg !42
  %address.local805.idx = mul nuw nsw i64 %address.local803, 768, !dbg !42
  %address.local805 = getelementptr i8, ptr addrspace(3) %address.local802, i64 %address.local805.idx, !dbg !42
  %address.local807.idx = mul nsw i64 %address.local368, -64, !dbg !42
  %address.local807 = getelementptr i8, ptr addrspace(3) %address.local805, i64 %address.local807.idx, !dbg !42
  %address.local808 = getelementptr i8, ptr addrspace(3) %address.local807, i64 256, !dbg !42
  %520 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local808, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %521 = bitcast <4 x i32> %520 to <8 x half>, !dbg !42
  %address.local812 = getelementptr i8, ptr addrspace(3) %address.local807, i64 512, !dbg !42
  %522 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %523 = bitcast <4 x i32> %522 to <8 x half>, !dbg !42
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local807, i64 768, !dbg !42
  %524 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %525 = bitcast <4 x i32> %524 to <8 x half>, !dbg !42
  %address.local818 = getelementptr i8, ptr addrspace(3) %address.local798, i64 4096, !dbg !42
  %526 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local818, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %527 = bitcast <4 x i32> %526 to <8 x half>, !dbg !42
  %address.local822 = getelementptr i8, ptr addrspace(3) %address.local807, i64 4352, !dbg !42
  %528 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %529 = bitcast <4 x i32> %528 to <8 x half>, !dbg !42
  %address.local826 = getelementptr i8, ptr addrspace(3) %address.local807, i64 4608, !dbg !42
  %530 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %531 = bitcast <4 x i32> %530 to <8 x half>, !dbg !42
  %address.local830 = getelementptr i8, ptr addrspace(3) %address.local807, i64 4864, !dbg !42
  %532 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %533 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !40
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !44
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi27), !dbg !40
  %537 = insertelement <4 x float> %535, float %536, i64 1, !dbg !44
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi29), !dbg !40
  %539 = insertelement <4 x float> %537, float %538, i64 2, !dbg !44
  %540 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi31), !dbg !40
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
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi33), !dbg !40
  %547 = insertelement <4 x float> poison, float %546, i64 0, !dbg !44
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi35), !dbg !40
  %549 = insertelement <4 x float> %547, float %548, i64 1, !dbg !44
  %550 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi37), !dbg !40
  %551 = insertelement <4 x float> %549, float %550, i64 2, !dbg !44
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi39), !dbg !40
  %553 = insertelement <4 x float> %551, float %552, i64 3, !dbg !44
  %math.result248 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %553) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02040 = extractelement <4 x float> %math.result248, i64 0, !dbg !44
  %.cvtpack.f12041 = extractelement <4 x float> %math.result248, i64 1, !dbg !44
  %.cvtpack.f22042 = extractelement <4 x float> %math.result248, i64 2, !dbg !44
  %.cvtpack.f32043 = extractelement <4 x float> %math.result248, i64 3, !dbg !44
  %.cvtpack.p02044 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02040, float %.cvtpack.f12041) #12, !dbg !44
  %.cvtpack.p12045 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22042, float %.cvtpack.f32043) #12, !dbg !44
  %.cvtpack.v02046 = insertelement <2 x i32> poison, i32 %.cvtpack.p02044, i64 0, !dbg !44
  %.cvtpack.v12047 = insertelement <2 x i32> %.cvtpack.v02046, i32 %.cvtpack.p12045, i64 1, !dbg !44
  %554 = bitcast <2 x i32> %.cvtpack.v12047 to <4 x half>, !dbg !44
  %math.packed249 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %554) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %533, <4 x float> %514) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %544, <4 x float> %515) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %557 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %558 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi41), !dbg !40
  %559 = fadd float %534, %558, !dbg !45
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %559) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %560 = insertelement <4 x float> poison, float %558, i64 0, !dbg !44
  %561 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi43), !dbg !40
  %562 = fadd float %536, %561, !dbg !45
  %k0.sum.anchor1843 = tail call float asm sideeffect "", "=v,0"(float %562) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> %560, float %561, i64 1, !dbg !44
  %564 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi45), !dbg !40
  %565 = fadd float %538, %564, !dbg !45
  %k0.sum.anchor1844 = tail call float asm sideeffect "", "=v,0"(float %565) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %566 = insertelement <4 x float> %563, float %564, i64 2, !dbg !44
  %567 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi47), !dbg !40
  %568 = fadd float %540, %567, !dbg !45
  %k0.sum.anchor1845 = tail call float asm sideeffect "", "=v,0"(float %568) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %569 = insertelement <4 x float> %566, float %567, i64 3, !dbg !44
  %math.result234 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %569) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02048 = extractelement <4 x float> %math.result234, i64 0, !dbg !44
  %.cvtpack.f12049 = extractelement <4 x float> %math.result234, i64 1, !dbg !44
  %.cvtpack.f22050 = extractelement <4 x float> %math.result234, i64 2, !dbg !44
  %.cvtpack.f32051 = extractelement <4 x float> %math.result234, i64 3, !dbg !44
  %.cvtpack.p02052 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02048, float %.cvtpack.f12049) #12, !dbg !44
  %.cvtpack.p12053 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22050, float %.cvtpack.f32051) #12, !dbg !44
  %.cvtpack.v02054 = insertelement <2 x i32> poison, i32 %.cvtpack.p02052, i64 0, !dbg !44
  %.cvtpack.v12055 = insertelement <2 x i32> %.cvtpack.v02054, i32 %.cvtpack.p12053, i64 1, !dbg !44
  %570 = bitcast <2 x i32> %.cvtpack.v12055 to <4 x half>, !dbg !44
  %math.packed235 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %570) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %571 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %557, <4 x float> %543) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %572 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %572, <4 x float> %545) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %574 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi49), !dbg !40
  %575 = fadd float %546, %574, !dbg !45
  %k0.sum.anchor1846 = tail call float asm sideeffect "", "=v,0"(float %575) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %576 = insertelement <4 x float> poison, float %574, i64 0, !dbg !44
  %577 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi51), !dbg !40
  %578 = fadd float %548, %577, !dbg !45
  %k0.sum.anchor1847 = tail call float asm sideeffect "", "=v,0"(float %578) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %579 = insertelement <4 x float> %576, float %577, i64 1, !dbg !44
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi53), !dbg !40
  %581 = fadd float %550, %580, !dbg !45
  %k0.sum.anchor1848 = tail call float asm sideeffect "", "=v,0"(float %581) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %582 = insertelement <4 x float> %579, float %580, i64 2, !dbg !44
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi55), !dbg !40
  %584 = fadd float %552, %583, !dbg !45
  %k0.sum.anchor1849 = tail call float asm sideeffect "", "=v,0"(float %584) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %585 = insertelement <4 x float> %582, float %583, i64 3, !dbg !44
  %math.result250 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %585) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02056 = extractelement <4 x float> %math.result250, i64 0, !dbg !44
  %.cvtpack.f12057 = extractelement <4 x float> %math.result250, i64 1, !dbg !44
  %.cvtpack.f22058 = extractelement <4 x float> %math.result250, i64 2, !dbg !44
  %.cvtpack.f32059 = extractelement <4 x float> %math.result250, i64 3, !dbg !44
  %.cvtpack.p02060 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02056, float %.cvtpack.f12057) #12, !dbg !44
  %.cvtpack.p12061 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22058, float %.cvtpack.f32059) #12, !dbg !44
  %.cvtpack.v02062 = insertelement <2 x i32> poison, i32 %.cvtpack.p02060, i64 0, !dbg !44
  %.cvtpack.v12063 = insertelement <2 x i32> %.cvtpack.v02062, i32 %.cvtpack.p12061, i64 1, !dbg !44
  %586 = bitcast <2 x i32> %.cvtpack.v12063 to <4 x half>, !dbg !44
  %math.packed251 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %586) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %587 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %557, <4 x float> %555) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %572, <4 x float> %556) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %589 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %590 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi57), !dbg !40
  %591 = fadd float %590, %k0.sum.anchor, !dbg !45
  %k0.sum.anchor1850 = tail call float asm sideeffect "", "=v,0"(float %591) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %592 = insertelement <4 x float> poison, float %590, i64 0, !dbg !44
  %593 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi59), !dbg !40
  %594 = fadd float %593, %k0.sum.anchor1843, !dbg !45
  %k0.sum.anchor1851 = tail call float asm sideeffect "", "=v,0"(float %594) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> %592, float %593, i64 1, !dbg !44
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi61), !dbg !40
  %597 = fadd float %596, %k0.sum.anchor1844, !dbg !45
  %k0.sum.anchor1852 = tail call float asm sideeffect "", "=v,0"(float %597) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %598 = insertelement <4 x float> %595, float %596, i64 2, !dbg !44
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi63), !dbg !40
  %600 = fadd float %599, %k0.sum.anchor1845, !dbg !45
  %k0.sum.anchor1853 = tail call float asm sideeffect "", "=v,0"(float %600) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> %598, float %599, i64 3, !dbg !44
  %math.result236 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %601) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02064 = extractelement <4 x float> %math.result236, i64 0, !dbg !44
  %.cvtpack.f12065 = extractelement <4 x float> %math.result236, i64 1, !dbg !44
  %.cvtpack.f22066 = extractelement <4 x float> %math.result236, i64 2, !dbg !44
  %.cvtpack.f32067 = extractelement <4 x float> %math.result236, i64 3, !dbg !44
  %.cvtpack.p02068 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02064, float %.cvtpack.f12065) #12, !dbg !44
  %.cvtpack.p12069 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22066, float %.cvtpack.f32067) #12, !dbg !44
  %.cvtpack.v02070 = insertelement <2 x i32> poison, i32 %.cvtpack.p02068, i64 0, !dbg !44
  %.cvtpack.v12071 = insertelement <2 x i32> %.cvtpack.v02070, i32 %.cvtpack.p12069, i64 1, !dbg !44
  %602 = bitcast <2 x i32> %.cvtpack.v12071 to <4 x half>, !dbg !44
  %math.packed237 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %602) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %603 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %589, <4 x float> %571) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %604 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %605 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %604, <4 x float> %573) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %606 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi65), !dbg !40
  %607 = fadd float %606, %k0.sum.anchor1846, !dbg !45
  %k0.sum.anchor1854 = tail call float asm sideeffect "", "=v,0"(float %607) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %608 = insertelement <4 x float> poison, float %606, i64 0, !dbg !44
  %609 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi67), !dbg !40
  %610 = fadd float %609, %k0.sum.anchor1847, !dbg !45
  %k0.sum.anchor1855 = tail call float asm sideeffect "", "=v,0"(float %610) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> %608, float %609, i64 1, !dbg !44
  %612 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi69), !dbg !40
  %613 = fadd float %612, %k0.sum.anchor1848, !dbg !45
  %k0.sum.anchor1856 = tail call float asm sideeffect "", "=v,0"(float %613) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %614 = insertelement <4 x float> %611, float %612, i64 2, !dbg !44
  %615 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi71), !dbg !40
  %616 = fadd float %615, %k0.sum.anchor1849, !dbg !45
  %k0.sum.anchor1857 = tail call float asm sideeffect "", "=v,0"(float %616) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %617 = insertelement <4 x float> %614, float %615, i64 3, !dbg !44
  %math.result252 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %617) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02072 = extractelement <4 x float> %math.result252, i64 0, !dbg !44
  %.cvtpack.f12073 = extractelement <4 x float> %math.result252, i64 1, !dbg !44
  %.cvtpack.f22074 = extractelement <4 x float> %math.result252, i64 2, !dbg !44
  %.cvtpack.f32075 = extractelement <4 x float> %math.result252, i64 3, !dbg !44
  %.cvtpack.p02076 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02072, float %.cvtpack.f12073) #12, !dbg !44
  %.cvtpack.p12077 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22074, float %.cvtpack.f32075) #12, !dbg !44
  %.cvtpack.v02078 = insertelement <2 x i32> poison, i32 %.cvtpack.p02076, i64 0, !dbg !44
  %.cvtpack.v12079 = insertelement <2 x i32> %.cvtpack.v02078, i32 %.cvtpack.p12077, i64 1, !dbg !44
  %618 = bitcast <2 x i32> %.cvtpack.v12079 to <4 x half>, !dbg !44
  %math.packed253 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %618) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %619 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %589, <4 x float> %587) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %604, <4 x float> %588) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %621 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi73), !dbg !40
  %623 = fadd float %622, %k0.sum.anchor1850, !dbg !45
  %k0.sum.anchor1858 = tail call float asm sideeffect "", "=v,0"(float %623) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %624 = insertelement <4 x float> poison, float %622, i64 0, !dbg !44
  %625 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi75), !dbg !40
  %626 = fadd float %625, %k0.sum.anchor1851, !dbg !45
  %k0.sum.anchor1859 = tail call float asm sideeffect "", "=v,0"(float %626) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %627 = insertelement <4 x float> %624, float %625, i64 1, !dbg !44
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi77), !dbg !40
  %629 = fadd float %628, %k0.sum.anchor1852, !dbg !45
  %k0.sum.anchor1860 = tail call float asm sideeffect "", "=v,0"(float %629) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %630 = insertelement <4 x float> %627, float %628, i64 2, !dbg !44
  %631 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi79), !dbg !40
  %632 = fadd float %631, %k0.sum.anchor1853, !dbg !45
  %k0.sum.anchor1861 = tail call float asm sideeffect "", "=v,0"(float %632) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %633 = insertelement <4 x float> %630, float %631, i64 3, !dbg !44
  %math.result238 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %633) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02080 = extractelement <4 x float> %math.result238, i64 0, !dbg !44
  %.cvtpack.f12081 = extractelement <4 x float> %math.result238, i64 1, !dbg !44
  %.cvtpack.f22082 = extractelement <4 x float> %math.result238, i64 2, !dbg !44
  %.cvtpack.f32083 = extractelement <4 x float> %math.result238, i64 3, !dbg !44
  %.cvtpack.p02084 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02080, float %.cvtpack.f12081) #12, !dbg !44
  %.cvtpack.p12085 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22082, float %.cvtpack.f32083) #12, !dbg !44
  %.cvtpack.v02086 = insertelement <2 x i32> poison, i32 %.cvtpack.p02084, i64 0, !dbg !44
  %.cvtpack.v12087 = insertelement <2 x i32> %.cvtpack.v02086, i32 %.cvtpack.p12085, i64 1, !dbg !44
  %634 = bitcast <2 x i32> %.cvtpack.v12087 to <4 x half>, !dbg !44
  %math.packed239 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %634) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %621, <4 x float> %603) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %636 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %637 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %636, <4 x float> %605) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %638 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi81), !dbg !40
  %639 = fadd float %638, %k0.sum.anchor1854, !dbg !45
  %k0.sum.anchor1862 = tail call float asm sideeffect "", "=v,0"(float %639) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %640 = insertelement <4 x float> poison, float %638, i64 0, !dbg !44
  %641 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi83), !dbg !40
  %642 = fadd float %641, %k0.sum.anchor1855, !dbg !45
  %k0.sum.anchor1863 = tail call float asm sideeffect "", "=v,0"(float %642) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %643 = insertelement <4 x float> %640, float %641, i64 1, !dbg !44
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi85), !dbg !40
  %645 = fadd float %644, %k0.sum.anchor1856, !dbg !45
  %k0.sum.anchor1864 = tail call float asm sideeffect "", "=v,0"(float %645) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> %643, float %644, i64 2, !dbg !44
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi87), !dbg !40
  %648 = fadd float %647, %k0.sum.anchor1857, !dbg !45
  %k0.sum.anchor1865 = tail call float asm sideeffect "", "=v,0"(float %648) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %649 = insertelement <4 x float> %646, float %647, i64 3, !dbg !44
  %math.result254 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %649) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02088 = extractelement <4 x float> %math.result254, i64 0, !dbg !44
  %.cvtpack.f12089 = extractelement <4 x float> %math.result254, i64 1, !dbg !44
  %.cvtpack.f22090 = extractelement <4 x float> %math.result254, i64 2, !dbg !44
  %.cvtpack.f32091 = extractelement <4 x float> %math.result254, i64 3, !dbg !44
  %.cvtpack.p02092 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02088, float %.cvtpack.f12089) #12, !dbg !44
  %.cvtpack.p12093 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22090, float %.cvtpack.f32091) #12, !dbg !44
  %.cvtpack.v02094 = insertelement <2 x i32> poison, i32 %.cvtpack.p02092, i64 0, !dbg !44
  %.cvtpack.v12095 = insertelement <2 x i32> %.cvtpack.v02094, i32 %.cvtpack.p12093, i64 1, !dbg !44
  %650 = bitcast <2 x i32> %.cvtpack.v12095 to <4 x half>, !dbg !44
  %math.packed255 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %650) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %621, <4 x float> %619) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %652 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %636, <4 x float> %620) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %653 = shufflevector <8 x half> %527, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %654 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi89), !dbg !40
  %655 = fadd float %654, %k0.sum.anchor1858, !dbg !45
  %k0.sum.anchor1866 = tail call float asm sideeffect "", "=v,0"(float %655) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %656 = insertelement <4 x float> poison, float %654, i64 0, !dbg !44
  %657 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi91), !dbg !40
  %658 = fadd float %657, %k0.sum.anchor1859, !dbg !45
  %k0.sum.anchor1867 = tail call float asm sideeffect "", "=v,0"(float %658) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %659 = insertelement <4 x float> %656, float %657, i64 1, !dbg !44
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi93), !dbg !40
  %661 = fadd float %660, %k0.sum.anchor1860, !dbg !45
  %k0.sum.anchor1868 = tail call float asm sideeffect "", "=v,0"(float %661) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %659, float %660, i64 2, !dbg !44
  %663 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi95), !dbg !40
  %664 = fadd float %663, %k0.sum.anchor1861, !dbg !45
  %k0.sum.anchor1869 = tail call float asm sideeffect "", "=v,0"(float %664) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %665 = insertelement <4 x float> %662, float %663, i64 3, !dbg !44
  %math.result240 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %665) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02096 = extractelement <4 x float> %math.result240, i64 0, !dbg !44
  %.cvtpack.f12097 = extractelement <4 x float> %math.result240, i64 1, !dbg !44
  %.cvtpack.f22098 = extractelement <4 x float> %math.result240, i64 2, !dbg !44
  %.cvtpack.f32099 = extractelement <4 x float> %math.result240, i64 3, !dbg !44
  %.cvtpack.p02100 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02096, float %.cvtpack.f12097) #12, !dbg !44
  %.cvtpack.p12101 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22098, float %.cvtpack.f32099) #12, !dbg !44
  %.cvtpack.v02102 = insertelement <2 x i32> poison, i32 %.cvtpack.p02100, i64 0, !dbg !44
  %.cvtpack.v12103 = insertelement <2 x i32> %.cvtpack.v02102, i32 %.cvtpack.p12101, i64 1, !dbg !44
  %666 = bitcast <2 x i32> %.cvtpack.v12103 to <4 x half>, !dbg !44
  %math.packed241 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %666) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %667 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %653, <4 x float> %635) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %668 = shufflevector <8 x half> %527, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %668, <4 x float> %637) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %670 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi97), !dbg !40
  %671 = fadd float %670, %k0.sum.anchor1862, !dbg !45
  %k0.sum.anchor1870 = tail call float asm sideeffect "", "=v,0"(float %671) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %672 = insertelement <4 x float> poison, float %670, i64 0, !dbg !44
  %673 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi99), !dbg !40
  %674 = fadd float %673, %k0.sum.anchor1863, !dbg !45
  %k0.sum.anchor1871 = tail call float asm sideeffect "", "=v,0"(float %674) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %675 = insertelement <4 x float> %672, float %673, i64 1, !dbg !44
  %676 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi101), !dbg !40
  %677 = fadd float %676, %k0.sum.anchor1864, !dbg !45
  %k0.sum.anchor1872 = tail call float asm sideeffect "", "=v,0"(float %677) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %678 = insertelement <4 x float> %675, float %676, i64 2, !dbg !44
  %679 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi103), !dbg !40
  %680 = fadd float %679, %k0.sum.anchor1865, !dbg !45
  %k0.sum.anchor1873 = tail call float asm sideeffect "", "=v,0"(float %680) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %681 = insertelement <4 x float> %678, float %679, i64 3, !dbg !44
  %math.result256 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %681) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02104 = extractelement <4 x float> %math.result256, i64 0, !dbg !44
  %.cvtpack.f12105 = extractelement <4 x float> %math.result256, i64 1, !dbg !44
  %.cvtpack.f22106 = extractelement <4 x float> %math.result256, i64 2, !dbg !44
  %.cvtpack.f32107 = extractelement <4 x float> %math.result256, i64 3, !dbg !44
  %.cvtpack.p02108 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02104, float %.cvtpack.f12105) #12, !dbg !44
  %.cvtpack.p12109 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22106, float %.cvtpack.f32107) #12, !dbg !44
  %.cvtpack.v02110 = insertelement <2 x i32> poison, i32 %.cvtpack.p02108, i64 0, !dbg !44
  %.cvtpack.v12111 = insertelement <2 x i32> %.cvtpack.v02110, i32 %.cvtpack.p12109, i64 1, !dbg !44
  %682 = bitcast <2 x i32> %.cvtpack.v12111 to <4 x half>, !dbg !44
  %math.packed257 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %682) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %683 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %653, <4 x float> %651) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %668, <4 x float> %652) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %685 = shufflevector <8 x half> %529, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi105), !dbg !40
  %687 = fadd float %686, %k0.sum.anchor1866, !dbg !45
  %k0.sum.anchor1874 = tail call float asm sideeffect "", "=v,0"(float %687) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %688 = insertelement <4 x float> poison, float %686, i64 0, !dbg !44
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi107), !dbg !40
  %690 = fadd float %689, %k0.sum.anchor1867, !dbg !45
  %k0.sum.anchor1875 = tail call float asm sideeffect "", "=v,0"(float %690) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %691 = insertelement <4 x float> %688, float %689, i64 1, !dbg !44
  %692 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi109), !dbg !40
  %693 = fadd float %692, %k0.sum.anchor1868, !dbg !45
  %k0.sum.anchor1876 = tail call float asm sideeffect "", "=v,0"(float %693) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %694 = insertelement <4 x float> %691, float %692, i64 2, !dbg !44
  %695 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi111), !dbg !40
  %696 = fadd float %695, %k0.sum.anchor1869, !dbg !45
  %k0.sum.anchor1877 = tail call float asm sideeffect "", "=v,0"(float %696) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %697 = insertelement <4 x float> %694, float %695, i64 3, !dbg !44
  %math.result242 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %697) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02112 = extractelement <4 x float> %math.result242, i64 0, !dbg !44
  %.cvtpack.f12113 = extractelement <4 x float> %math.result242, i64 1, !dbg !44
  %.cvtpack.f22114 = extractelement <4 x float> %math.result242, i64 2, !dbg !44
  %.cvtpack.f32115 = extractelement <4 x float> %math.result242, i64 3, !dbg !44
  %.cvtpack.p02116 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02112, float %.cvtpack.f12113) #12, !dbg !44
  %.cvtpack.p12117 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22114, float %.cvtpack.f32115) #12, !dbg !44
  %.cvtpack.v02118 = insertelement <2 x i32> poison, i32 %.cvtpack.p02116, i64 0, !dbg !44
  %.cvtpack.v12119 = insertelement <2 x i32> %.cvtpack.v02118, i32 %.cvtpack.p12117, i64 1, !dbg !44
  %698 = bitcast <2 x i32> %.cvtpack.v12119 to <4 x half>, !dbg !44
  %math.packed243 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %698) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %699 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %685, <4 x float> %667) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %700 = shufflevector <8 x half> %529, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %700, <4 x float> %669) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %702 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi113), !dbg !40
  %703 = fadd float %702, %k0.sum.anchor1870, !dbg !45
  %k0.sum.anchor1878 = tail call float asm sideeffect "", "=v,0"(float %703) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %704 = insertelement <4 x float> poison, float %702, i64 0, !dbg !44
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi115), !dbg !40
  %706 = fadd float %705, %k0.sum.anchor1871, !dbg !45
  %k0.sum.anchor1879 = tail call float asm sideeffect "", "=v,0"(float %706) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %707 = insertelement <4 x float> %704, float %705, i64 1, !dbg !44
  %708 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi117), !dbg !40
  %709 = fadd float %708, %k0.sum.anchor1872, !dbg !45
  %k0.sum.anchor1880 = tail call float asm sideeffect "", "=v,0"(float %709) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %710 = insertelement <4 x float> %707, float %708, i64 2, !dbg !44
  %711 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi119), !dbg !40
  %712 = fadd float %711, %k0.sum.anchor1873, !dbg !45
  %k0.sum.anchor1881 = tail call float asm sideeffect "", "=v,0"(float %712) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %713 = insertelement <4 x float> %710, float %711, i64 3, !dbg !44
  %math.result258 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %713) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02120 = extractelement <4 x float> %math.result258, i64 0, !dbg !44
  %.cvtpack.f12121 = extractelement <4 x float> %math.result258, i64 1, !dbg !44
  %.cvtpack.f22122 = extractelement <4 x float> %math.result258, i64 2, !dbg !44
  %.cvtpack.f32123 = extractelement <4 x float> %math.result258, i64 3, !dbg !44
  %.cvtpack.p02124 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02120, float %.cvtpack.f12121) #12, !dbg !44
  %.cvtpack.p12125 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22122, float %.cvtpack.f32123) #12, !dbg !44
  %.cvtpack.v02126 = insertelement <2 x i32> poison, i32 %.cvtpack.p02124, i64 0, !dbg !44
  %.cvtpack.v12127 = insertelement <2 x i32> %.cvtpack.v02126, i32 %.cvtpack.p12125, i64 1, !dbg !44
  %714 = bitcast <2 x i32> %.cvtpack.v12127 to <4 x half>, !dbg !44
  %math.packed259 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %714) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %715 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %685, <4 x float> %683) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %700, <4 x float> %684) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %717 = shufflevector <8 x half> %531, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %718 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi121), !dbg !40
  %719 = fadd float %718, %k0.sum.anchor1874, !dbg !45
  %k0.sum.anchor1882 = tail call float asm sideeffect "", "=v,0"(float %719) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %720 = insertelement <4 x float> poison, float %718, i64 0, !dbg !44
  %721 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi123), !dbg !40
  %722 = fadd float %721, %k0.sum.anchor1875, !dbg !45
  %k0.sum.anchor1883 = tail call float asm sideeffect "", "=v,0"(float %722) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %723 = insertelement <4 x float> %720, float %721, i64 1, !dbg !44
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi125), !dbg !40
  %725 = fadd float %724, %k0.sum.anchor1876, !dbg !45
  %k0.sum.anchor1884 = tail call float asm sideeffect "", "=v,0"(float %725) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %726 = insertelement <4 x float> %723, float %724, i64 2, !dbg !44
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi127), !dbg !40
  %728 = fadd float %727, %k0.sum.anchor1877, !dbg !45
  %k0.sum.anchor1885 = tail call float asm sideeffect "", "=v,0"(float %728) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %729 = insertelement <4 x float> %726, float %727, i64 3, !dbg !44
  %math.result244 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %729) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02128 = extractelement <4 x float> %math.result244, i64 0, !dbg !44
  %.cvtpack.f12129 = extractelement <4 x float> %math.result244, i64 1, !dbg !44
  %.cvtpack.f22130 = extractelement <4 x float> %math.result244, i64 2, !dbg !44
  %.cvtpack.f32131 = extractelement <4 x float> %math.result244, i64 3, !dbg !44
  %.cvtpack.p02132 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02128, float %.cvtpack.f12129) #12, !dbg !44
  %.cvtpack.p12133 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22130, float %.cvtpack.f32131) #12, !dbg !44
  %.cvtpack.v02134 = insertelement <2 x i32> poison, i32 %.cvtpack.p02132, i64 0, !dbg !44
  %.cvtpack.v12135 = insertelement <2 x i32> %.cvtpack.v02134, i32 %.cvtpack.p12133, i64 1, !dbg !44
  %730 = bitcast <2 x i32> %.cvtpack.v12135 to <4 x half>, !dbg !44
  %math.packed245 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %730) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %731 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %717, <4 x float> %699) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %732 = shufflevector <8 x half> %531, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %733 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %732, <4 x float> %701) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %734 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi129), !dbg !40
  %735 = fadd float %734, %k0.sum.anchor1878, !dbg !45
  %k0.sum.anchor1886 = tail call float asm sideeffect "", "=v,0"(float %735) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %736 = insertelement <4 x float> poison, float %734, i64 0, !dbg !44
  %737 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi131), !dbg !40
  %738 = fadd float %737, %k0.sum.anchor1879, !dbg !45
  %k0.sum.anchor1887 = tail call float asm sideeffect "", "=v,0"(float %738) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %739 = insertelement <4 x float> %736, float %737, i64 1, !dbg !44
  %740 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi133), !dbg !40
  %741 = fadd float %740, %k0.sum.anchor1880, !dbg !45
  %k0.sum.anchor1888 = tail call float asm sideeffect "", "=v,0"(float %741) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %742 = insertelement <4 x float> %739, float %740, i64 2, !dbg !44
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi135), !dbg !40
  %744 = fadd float %743, %k0.sum.anchor1881, !dbg !45
  %k0.sum.anchor1889 = tail call float asm sideeffect "", "=v,0"(float %744) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %745 = insertelement <4 x float> %742, float %743, i64 3, !dbg !44
  %math.result260 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %745) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02136 = extractelement <4 x float> %math.result260, i64 0, !dbg !44
  %.cvtpack.f12137 = extractelement <4 x float> %math.result260, i64 1, !dbg !44
  %.cvtpack.f22138 = extractelement <4 x float> %math.result260, i64 2, !dbg !44
  %.cvtpack.f32139 = extractelement <4 x float> %math.result260, i64 3, !dbg !44
  %.cvtpack.p02140 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02136, float %.cvtpack.f12137) #12, !dbg !44
  %.cvtpack.p12141 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22138, float %.cvtpack.f32139) #12, !dbg !44
  %.cvtpack.v02142 = insertelement <2 x i32> poison, i32 %.cvtpack.p02140, i64 0, !dbg !44
  %.cvtpack.v12143 = insertelement <2 x i32> %.cvtpack.v02142, i32 %.cvtpack.p12141, i64 1, !dbg !44
  %746 = bitcast <2 x i32> %.cvtpack.v12143 to <4 x half>, !dbg !44
  %math.packed261 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %746) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %747 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %717, <4 x float> %715) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %732, <4 x float> %716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %749 = bitcast <4 x i32> %532 to <8 x half>, !dbg !42
  %750 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi137), !dbg !40
  %752 = fadd float %751, %k0.sum.anchor1882, !dbg !45
  %k0.sum.anchor1890 = tail call float asm sideeffect "", "=v,0"(float %752) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %753 = insertelement <4 x float> poison, float %751, i64 0, !dbg !44
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi139), !dbg !40
  %755 = fadd float %754, %k0.sum.anchor1883, !dbg !45
  %k0.sum.anchor1891 = tail call float asm sideeffect "", "=v,0"(float %755) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %756 = fadd float %k0.sum.anchor1890, %k0.sum.anchor1891, !dbg !45
  %k0.sum.anchor1892 = tail call float asm sideeffect "", "=v,0"(float %756) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %757 = insertelement <4 x float> %753, float %754, i64 1, !dbg !44
  %758 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi141), !dbg !40
  %759 = fadd float %758, %k0.sum.anchor1884, !dbg !45
  %k0.sum.anchor1893 = tail call float asm sideeffect "", "=v,0"(float %759) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %760 = insertelement <4 x float> %757, float %758, i64 2, !dbg !44
  %761 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi143), !dbg !40
  %762 = fadd float %761, %k0.sum.anchor1885, !dbg !45
  %k0.sum.anchor1894 = tail call float asm sideeffect "", "=v,0"(float %762) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %763 = fadd float %k0.sum.anchor1893, %k0.sum.anchor1894, !dbg !45
  %k0.sum.anchor1895 = tail call float asm sideeffect "", "=v,0"(float %763) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %764 = fadd float %k0.sum.anchor1892, %k0.sum.anchor1895, !dbg !45
  %k0.sum.anchor1896 = tail call float asm sideeffect "", "=v,0"(float %764) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %765 = fadd float %507, %k0.sum.anchor1896, !dbg !45
  %k0.sum.anchor1897 = tail call float asm sideeffect "", "=v,0"(float %765) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %766 = insertelement <1 x float> poison, float %k0.sum.anchor1897, i64 0, !dbg !12
  %767 = insertvalue [2 x <1 x float>] poison, <1 x float> %766, 0, !dbg !12
  %768 = insertelement <4 x float> %760, float %761, i64 3, !dbg !44
  %math.result246 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %768) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02144 = extractelement <4 x float> %math.result246, i64 0, !dbg !44
  %.cvtpack.f12145 = extractelement <4 x float> %math.result246, i64 1, !dbg !44
  %.cvtpack.f22146 = extractelement <4 x float> %math.result246, i64 2, !dbg !44
  %.cvtpack.f32147 = extractelement <4 x float> %math.result246, i64 3, !dbg !44
  %.cvtpack.p02148 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02144, float %.cvtpack.f12145) #12, !dbg !44
  %.cvtpack.p12149 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22146, float %.cvtpack.f32147) #12, !dbg !44
  %.cvtpack.v02150 = insertelement <2 x i32> poison, i32 %.cvtpack.p02148, i64 0, !dbg !44
  %.cvtpack.v12151 = insertelement <2 x i32> %.cvtpack.v02150, i32 %.cvtpack.p12149, i64 1, !dbg !44
  %769 = bitcast <2 x i32> %.cvtpack.v12151 to <4 x half>, !dbg !44
  %math.packed247 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %769) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %770 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %750, <4 x float> %731) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %771 = shufflevector <8 x half> %749, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %772 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %771, <4 x float> %733) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %773 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi145), !dbg !40
  %774 = fadd float %773, %k0.sum.anchor1886, !dbg !45
  %k0.sum.anchor1898 = tail call float asm sideeffect "", "=v,0"(float %774) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %775 = insertelement <4 x float> poison, float %773, i64 0, !dbg !44
  %776 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi147), !dbg !40
  %777 = fadd float %776, %k0.sum.anchor1887, !dbg !45
  %k0.sum.anchor1899 = tail call float asm sideeffect "", "=v,0"(float %777) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %778 = fadd float %k0.sum.anchor1898, %k0.sum.anchor1899, !dbg !45
  %k0.sum.anchor1900 = tail call float asm sideeffect "", "=v,0"(float %778) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %779 = insertelement <4 x float> %775, float %776, i64 1, !dbg !44
  %780 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi149), !dbg !40
  %781 = fadd float %780, %k0.sum.anchor1888, !dbg !45
  %k0.sum.anchor1901 = tail call float asm sideeffect "", "=v,0"(float %781) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %782 = insertelement <4 x float> %779, float %780, i64 2, !dbg !44
  %783 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi151), !dbg !40
  %784 = fadd float %783, %k0.sum.anchor1889, !dbg !45
  %k0.sum.anchor1902 = tail call float asm sideeffect "", "=v,0"(float %784) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %785 = fadd float %k0.sum.anchor1901, %k0.sum.anchor1902, !dbg !45
  %k0.sum.anchor1903 = tail call float asm sideeffect "", "=v,0"(float %785) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %786 = fadd float %k0.sum.anchor1900, %k0.sum.anchor1903, !dbg !45
  %k0.sum.anchor1904 = tail call float asm sideeffect "", "=v,0"(float %786) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %787 = fadd float %509, %k0.sum.anchor1904, !dbg !45
  %k0.sum.anchor1905 = tail call float asm sideeffect "", "=v,0"(float %787) #8, !dbg !45, !frisk.cross_bn_sum_anchor !5
  %788 = insertelement <1 x float> poison, float %k0.sum.anchor1905, i64 0, !dbg !12
  %789 = insertvalue [2 x <1 x float>] %767, <1 x float> %788, 1, !dbg !12
  %790 = insertelement <4 x float> %782, float %783, i64 3, !dbg !44
  %math.result262 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %790) #8, !dbg !44, !frisk.register_lifetimes !5
  %.cvtpack.f02152 = extractelement <4 x float> %math.result262, i64 0, !dbg !44
  %.cvtpack.f12153 = extractelement <4 x float> %math.result262, i64 1, !dbg !44
  %.cvtpack.f22154 = extractelement <4 x float> %math.result262, i64 2, !dbg !44
  %.cvtpack.f32155 = extractelement <4 x float> %math.result262, i64 3, !dbg !44
  %.cvtpack.p02156 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02152, float %.cvtpack.f12153) #12, !dbg !44
  %.cvtpack.p12157 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22154, float %.cvtpack.f32155) #12, !dbg !44
  %.cvtpack.v02158 = insertelement <2 x i32> poison, i32 %.cvtpack.p02156, i64 0, !dbg !44
  %.cvtpack.v12159 = insertelement <2 x i32> %.cvtpack.v02158, i32 %.cvtpack.p12157, i64 1, !dbg !44
  %791 = bitcast <2 x i32> %.cvtpack.v12159 to <4 x half>, !dbg !44
  %math.packed263 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %791) #8, !dbg !45, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %750, <4 x float> %747) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %771, <4 x float> %748) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %794 = extractvalue [2 x <8 x float>] %lazy.state2038, 0, !dbg !41
  %795 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %796 = shufflevector <8 x float> %794, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %797 = extractvalue [2 x <8 x float>] %lazy.state2038, 1, !dbg !41
  %798 = shufflevector <8 x float> %797, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %799 = shufflevector <8 x float> %797, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1307 = add nsw i64 %address.local782, %205, !dbg !38
  %vdb.address1308 = trunc nsw i64 %vdb.address1307 to i32, !dbg !38
  %vdb.address1309 = shl nsw i32 %vdb.address1308, 1, !dbg !38
  %vdb.address1314 = trunc nuw nsw i64 %205 to i32, !dbg !38
  %vdb.address1315 = or disjoint i32 %address.local789, %vdb.address1314, !dbg !38
  %vdb.address1316 = shl nuw nsw i32 %vdb.address1315, 1, !dbg !38
  %800 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1309, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %801 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %vdb.address1316, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local798, i64 8192, !dbg !42
  %802 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %803 = bitcast <4 x i32> %802 to <8 x half>, !dbg !42
  %vdb.read1300 = getelementptr i8, ptr addrspace(3) %address.local807, i64 8448, !dbg !42
  %804 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1300, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %805 = bitcast <4 x i32> %804 to <8 x half>, !dbg !42
  %vdb.read1301 = getelementptr i8, ptr addrspace(3) %address.local807, i64 8704, !dbg !42
  %806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1301, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %807 = bitcast <4 x i32> %806 to <8 x half>, !dbg !42
  %vdb.read1302 = getelementptr i8, ptr addrspace(3) %address.local807, i64 8960, !dbg !42
  %808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1302, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !42
  %vdb.read1303 = getelementptr i8, ptr addrspace(3) %address.local798, i64 12288, !dbg !42
  %810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1303, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !42
  %vdb.read1304 = getelementptr i8, ptr addrspace(3) %address.local807, i64 12544, !dbg !42
  %812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1304, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !42
  %vdb.read1305 = getelementptr i8, ptr addrspace(3) %address.local807, i64 12800, !dbg !42
  %814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1305, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !42
  %vdb.read1306 = getelementptr i8, ptr addrspace(3) %address.local807, i64 13056, !dbg !42
  %816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1306, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %817 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %817, <4 x float> %795) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %819 = shufflevector <8 x half> %803, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %819, <4 x float> %796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %817, <4 x float> %798) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %819, <4 x float> %799) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %823 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %823, <4 x float> %818) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %825 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %825, <4 x float> %820) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %827 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %823, <4 x float> %821) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %825, <4 x float> %822) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %829 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %829, <4 x float> %824) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %831 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %831, <4 x float> %826) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %833 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %829, <4 x float> %827) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %831, <4 x float> %828) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %835 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %835, <4 x float> %830) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %837 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %837, <4 x float> %832) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %835, <4 x float> %833) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %837, <4 x float> %834) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %841 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %841, <4 x float> %836) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %843 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %843, <4 x float> %838) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %841, <4 x float> %839) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %843, <4 x float> %840) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %847 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %847, <4 x float> %842) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %849 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %849, <4 x float> %844) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %847, <4 x float> %845) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %849, <4 x float> %846) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %853 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %853, <4 x float> %848) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %855 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %855, <4 x float> %850) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %857 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %853, <4 x float> %851) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %855, <4 x float> %852) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %859 = bitcast <4 x i32> %816 to <8 x half>, !dbg !42
  %860 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %860, <4 x float> %854) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %862 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %862, <4 x float> %856) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %860, <4 x float> %857) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %862, <4 x float> %858) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %866 = extractvalue [2 x <8 x float>] %lazy.state2037, 0, !dbg !41
  %867 = shufflevector <8 x float> %866, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %868 = shufflevector <8 x float> %866, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %869 = extractvalue [2 x <8 x float>] %lazy.state2037, 1, !dbg !41
  %870 = shufflevector <8 x float> %869, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %871 = shufflevector <8 x float> %869, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %vdb.address1321 = add nsw i64 %address.local782, %258, !dbg !38
  %vdb.address1322 = trunc nsw i64 %vdb.address1321 to i32, !dbg !38
  %vdb.address1323 = shl nsw i32 %vdb.address1322, 1, !dbg !38
  %vdb.address1328 = trunc nuw nsw i64 %258 to i32, !dbg !38
  %vdb.address1329 = or disjoint i32 %address.local789, %vdb.address1328, !dbg !38
  %vdb.address1330 = shl nuw nsw i32 %vdb.address1329, 1, !dbg !38
  %872 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1323, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %873 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1299, i32 %vdb.address1330, <4 x i32> %dma.buffer.rsrc1206) #9, !dbg !38
  %874 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local798, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %875 = bitcast <4 x i32> %874 to <8 x half>, !dbg !42
  %876 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local808, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %877 = bitcast <4 x i32> %876 to <8 x half>, !dbg !42
  %878 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %879 = bitcast <4 x i32> %878 to <8 x half>, !dbg !42
  %880 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %881 = bitcast <4 x i32> %880 to <8 x half>, !dbg !42
  %882 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local818, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %883 = bitcast <4 x i32> %882 to <8 x half>, !dbg !42
  %884 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %885 = bitcast <4 x i32> %884 to <8 x half>, !dbg !42
  %886 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %887 = bitcast <4 x i32> %886 to <8 x half>, !dbg !42
  %888 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %889 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %890 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %889, <4 x float> %867) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %891 = shufflevector <8 x half> %875, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %891, <4 x float> %868) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %889, <4 x float> %870) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %891, <4 x float> %871) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %895 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %895, <4 x float> %890) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %897 = shufflevector <8 x half> %877, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %897, <4 x float> %892) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %895, <4 x float> %893) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %897, <4 x float> %894) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %901 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %901, <4 x float> %896) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %903 = shufflevector <8 x half> %879, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %903, <4 x float> %898) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %901, <4 x float> %899) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %903, <4 x float> %900) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %907 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %907, <4 x float> %902) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %909 = shufflevector <8 x half> %881, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %909, <4 x float> %904) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %907, <4 x float> %905) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %909, <4 x float> %906) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %913 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %913, <4 x float> %908) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %915 = shufflevector <8 x half> %883, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %915, <4 x float> %910) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %913, <4 x float> %911) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %915, <4 x float> %912) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %919 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %919, <4 x float> %914) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %921 = shufflevector <8 x half> %885, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %921, <4 x float> %916) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %919, <4 x float> %917) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %921, <4 x float> %918) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %925 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %925, <4 x float> %920) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %927 = shufflevector <8 x half> %887, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %927, <4 x float> %922) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %925, <4 x float> %923) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %927, <4 x float> %924) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %931 = bitcast <4 x i32> %888 to <8 x half>, !dbg !42
  %932 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %932, <4 x float> %926) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %934 = shufflevector <8 x half> %931, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %934, <4 x float> %928) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %932, <4 x float> %929) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %934, <4 x float> %930) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %938 = extractvalue [2 x <8 x float>] %lazy.state2036, 0, !dbg !41
  %939 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %940 = shufflevector <8 x float> %938, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %941 = extractvalue [2 x <8 x float>] %lazy.state2036, 1, !dbg !41
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
  %k0.next.1916 = add nuw nsw i64 %k0.next., %k0.prime.1906, !dbg !14
  %k0.next.kdma.source1361 = or disjoint i64 %k0.next.1916, %k0.prime.kdma.source1360, !dbg !14
  %k0.next.kdma.source1362 = trunc nuw nsw i64 %k0.next.kdma.source1361 to i32, !dbg !14
  %k0.next.kdma.source1363 = shl nuw nsw i32 %k0.next.kdma.source1362, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1363, %kdma.intra, !dbg !14
  %k0.next.1917 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1381 = or disjoint i64 %k0.next.1916, %k0.prime.kdma.source1380, !dbg !14
  %k0.next.kdma.source1382 = trunc nuw nsw i64 %k0.next.kdma.source1381 to i32, !dbg !14
  %k0.next.kdma.source1383 = shl nuw nsw i32 %k0.next.kdma.source1382, 1, !dbg !14
  %k0.next.kdma.offset1384 = or disjoint i32 %k0.next.kdma.source1383, %kdma.intra.hi, !dbg !14
  %k0.next.1918 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1385, i32 %k0.next.kdma.offset1384, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1405 = or disjoint i64 %k0.next.1916, %k0.prime.kdma.source1404, !dbg !14
  %k0.next.kdma.source1406 = trunc nuw nsw i64 %k0.next.kdma.source1405 to i32, !dbg !14
  %k0.next.kdma.source1407 = shl nuw nsw i32 %k0.next.kdma.source1406, 1, !dbg !14
  %k0.next.kdma.offset1408 = add nuw nsw i32 %k0.next.kdma.source1407, %kdma.intra, !dbg !14
  %k0.next.1919 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.next.kdma.offset1408, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1429 = or disjoint i64 %k0.next.1916, %k0.prime.kdma.source1428, !dbg !14
  %k0.next.kdma.source1430 = trunc nuw nsw i64 %k0.next.kdma.source1429 to i32, !dbg !14
  %k0.next.kdma.source1431 = shl nuw nsw i32 %k0.next.kdma.source1430, 1, !dbg !14
  %k0.next.kdma.offset1432 = add nuw nsw i32 %k0.next.kdma.source1431, %kdma.intra.hi, !dbg !14
  %k0.next.1920 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1433, i32 %k0.next.kdma.offset1432, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %945 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %946 = bitcast <4 x i32> %945 to <8 x half>, !dbg !42
  %947 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1300, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %948 = bitcast <4 x i32> %947 to <8 x half>, !dbg !42
  %949 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1301, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %950 = bitcast <4 x i32> %949 to <8 x half>, !dbg !42
  %951 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1302, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %952 = bitcast <4 x i32> %951 to <8 x half>, !dbg !42
  %953 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1303, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %954 = bitcast <4 x i32> %953 to <8 x half>, !dbg !42
  %955 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1304, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %956 = bitcast <4 x i32> %955 to <8 x half>, !dbg !42
  %957 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1305, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %958 = bitcast <4 x i32> %957 to <8 x half>, !dbg !42
  %959 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1306, i32 0) #11, !dbg !42, !frisk.v_double_buffer !43
  %960 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %960, <4 x float> %939) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %962 = shufflevector <8 x half> %946, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %962, <4 x float> %940) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %960, <4 x float> %942) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %962, <4 x float> %943) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %966 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %966, <4 x float> %961) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %968 = shufflevector <8 x half> %948, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %968, <4 x float> %963) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %966, <4 x float> %964) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %971 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %968, <4 x float> %965) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %972 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %972, <4 x float> %967) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %974 = shufflevector <8 x half> %950, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %974, <4 x float> %969) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %972, <4 x float> %970) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %974, <4 x float> %971) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %978 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %978, <4 x float> %973) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %980 = shufflevector <8 x half> %952, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %980, <4 x float> %975) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %978, <4 x float> %976) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %983 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %980, <4 x float> %977) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %984 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %984, <4 x float> %979) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %986 = shufflevector <8 x half> %954, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %986, <4 x float> %981) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %984, <4 x float> %982) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %986, <4 x float> %983) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %990 = shufflevector <8 x half> %956, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %990, <4 x float> %985) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %992 = shufflevector <8 x half> %956, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %992, <4 x float> %987) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %990, <4 x float> %988) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %995 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %992, <4 x float> %989) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %996 = shufflevector <8 x half> %958, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %996, <4 x float> %991) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %998 = shufflevector <8 x half> %958, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %998, <4 x float> %993) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %996, <4 x float> %994) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %998, <4 x float> %995) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1002 = bitcast <4 x i32> %959 to <8 x half>, !dbg !42
  %1003 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1003, <4 x float> %997) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1005 = shufflevector <8 x half> %1002, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1005, <4 x float> %999) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1003, <4 x float> %1000) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1008 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1005, <4 x float> %1001) #8, !dbg !42, !frisk.pv_interleave !19
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
  %1027 = bitcast float %k0.sum.anchor1905 to i32, !dbg !12
  %1028 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1027), !dbg !12
  %1029 = bitcast float %k0.sum.anchor1897 to i32, !dbg !12
  %1030 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1029), !dbg !12
  %1031 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1031, 0, !dbg !12
  %1032 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %1032, 0, !dbg !12
  %1033 = extractelement <4 x float> %1004, i64 0, !dbg !12
  %1034 = extractelement <4 x float> %1004, i64 1, !dbg !12
  %1035 = extractelement <4 x float> %1004, i64 2, !dbg !12
  %1036 = extractelement <4 x float> %1004, i64 3, !dbg !12
  %1037 = select i1 %.not, float %1034, float %1033, !dbg !12
  %1038 = bitcast float %1037 to i32, !dbg !12
  %1039 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1038), !dbg !12
  %1040 = bitcast i32 %1039 to float, !dbg !12
  %1041 = select i1 %.not, float %1033, float %1040, !dbg !12
  %1042 = select i1 %.not, float %1040, float %1034, !dbg !12
  %1043 = select i1 %.not, float %1036, float %1035, !dbg !12
  %1044 = bitcast float %1043 to i32, !dbg !12
  %1045 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1044), !dbg !12
  %1046 = bitcast i32 %1045 to float, !dbg !12
  %1047 = select i1 %.not, float %1035, float %1046, !dbg !12
  %1048 = select i1 %.not, float %1046, float %1036, !dbg !12
  %1049 = select i1 %.not40, float %1047, float %1041, !dbg !12
  %1050 = bitcast float %1049 to i32, !dbg !12
  %1051 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1050), !dbg !12
  %1052 = bitcast i32 %1051 to float, !dbg !12
  %1053 = select i1 %.not40, float %1048, float %1042, !dbg !12
  %1054 = bitcast float %1053 to i32, !dbg !12
  %1055 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1054), !dbg !12
  %1056 = bitcast i32 %1055 to float, !dbg !12
  %1057 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %1058 = shufflevector <4 x i1> %1057, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1059 = insertelement <4 x float> poison, float %1041, i64 0, !dbg !12
  %1060 = insertelement <4 x float> %1059, float %1042, i64 1, !dbg !12
  %1061 = insertelement <4 x float> %1060, float %1052, i64 2, !dbg !12
  %1062 = insertelement <4 x float> %1061, float %1056, i64 3, !dbg !12
  %1063 = insertelement <4 x float> poison, float %1052, i64 0, !dbg !12
  %1064 = insertelement <4 x float> %1063, float %1056, i64 1, !dbg !12
  %1065 = insertelement <4 x float> %1064, float %1047, i64 2, !dbg !12
  %1066 = insertelement <4 x float> %1065, float %1048, i64 3, !dbg !12
  %1067 = select <4 x i1> %1058, <4 x float> %1062, <4 x float> %1066, !dbg !12
  %1068 = extractelement <4 x float> %1006, i64 0, !dbg !12
  %1069 = extractelement <4 x float> %1006, i64 1, !dbg !12
  %1070 = extractelement <4 x float> %1006, i64 2, !dbg !12
  %1071 = extractelement <4 x float> %1006, i64 3, !dbg !12
  %1072 = select i1 %.not, float %1069, float %1068, !dbg !12
  %1073 = bitcast float %1072 to i32, !dbg !12
  %1074 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1073), !dbg !12
  %1075 = bitcast i32 %1074 to float, !dbg !12
  %1076 = select i1 %.not, float %1068, float %1075, !dbg !12
  %1077 = select i1 %.not, float %1075, float %1069, !dbg !12
  %1078 = select i1 %.not, float %1071, float %1070, !dbg !12
  %1079 = bitcast float %1078 to i32, !dbg !12
  %1080 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1079), !dbg !12
  %1081 = bitcast i32 %1080 to float, !dbg !12
  %1082 = select i1 %.not, float %1070, float %1081, !dbg !12
  %1083 = select i1 %.not, float %1081, float %1071, !dbg !12
  %1084 = select i1 %.not40, float %1082, float %1076, !dbg !12
  %1085 = bitcast float %1084 to i32, !dbg !12
  %1086 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1085), !dbg !12
  %1087 = bitcast i32 %1086 to float, !dbg !12
  %1088 = select i1 %.not40, float %1083, float %1077, !dbg !12
  %1089 = bitcast float %1088 to i32, !dbg !12
  %1090 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1089), !dbg !12
  %1091 = bitcast i32 %1090 to float, !dbg !12
  %1092 = insertelement <4 x float> poison, float %1076, i64 0, !dbg !12
  %1093 = insertelement <4 x float> %1092, float %1077, i64 1, !dbg !12
  %1094 = insertelement <4 x float> %1093, float %1087, i64 2, !dbg !12
  %1095 = insertelement <4 x float> %1094, float %1091, i64 3, !dbg !12
  %1096 = insertelement <4 x float> poison, float %1087, i64 0, !dbg !12
  %1097 = insertelement <4 x float> %1096, float %1091, i64 1, !dbg !12
  %1098 = insertelement <4 x float> %1097, float %1082, i64 2, !dbg !12
  %1099 = insertelement <4 x float> %1098, float %1083, i64 3, !dbg !12
  %1100 = select <4 x i1> %1058, <4 x float> %1095, <4 x float> %1099, !dbg !12
  %1101 = extractelement <4 x float> %1007, i64 0, !dbg !12
  %1102 = extractelement <4 x float> %1007, i64 1, !dbg !12
  %1103 = extractelement <4 x float> %1007, i64 2, !dbg !12
  %1104 = extractelement <4 x float> %1007, i64 3, !dbg !12
  %1105 = select i1 %.not, float %1102, float %1101, !dbg !12
  %1106 = bitcast float %1105 to i32, !dbg !12
  %1107 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1106), !dbg !12
  %1108 = bitcast i32 %1107 to float, !dbg !12
  %1109 = select i1 %.not, float %1101, float %1108, !dbg !12
  %1110 = select i1 %.not, float %1108, float %1102, !dbg !12
  %1111 = select i1 %.not, float %1104, float %1103, !dbg !12
  %1112 = bitcast float %1111 to i32, !dbg !12
  %1113 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1112), !dbg !12
  %1114 = bitcast i32 %1113 to float, !dbg !12
  %1115 = select i1 %.not, float %1103, float %1114, !dbg !12
  %1116 = select i1 %.not, float %1114, float %1104, !dbg !12
  %1117 = select i1 %.not40, float %1115, float %1109, !dbg !12
  %1118 = bitcast float %1117 to i32, !dbg !12
  %1119 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1118), !dbg !12
  %1120 = bitcast i32 %1119 to float, !dbg !12
  %1121 = select i1 %.not40, float %1116, float %1110, !dbg !12
  %1122 = bitcast float %1121 to i32, !dbg !12
  %1123 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1122), !dbg !12
  %1124 = bitcast i32 %1123 to float, !dbg !12
  %1125 = insertelement <4 x float> poison, float %1109, i64 0, !dbg !12
  %1126 = insertelement <4 x float> %1125, float %1110, i64 1, !dbg !12
  %1127 = insertelement <4 x float> %1126, float %1120, i64 2, !dbg !12
  %1128 = insertelement <4 x float> %1127, float %1124, i64 3, !dbg !12
  %1129 = insertelement <4 x float> poison, float %1120, i64 0, !dbg !12
  %1130 = insertelement <4 x float> %1129, float %1124, i64 1, !dbg !12
  %1131 = insertelement <4 x float> %1130, float %1115, i64 2, !dbg !12
  %1132 = insertelement <4 x float> %1131, float %1116, i64 3, !dbg !12
  %1133 = select <4 x i1> %1058, <4 x float> %1128, <4 x float> %1132, !dbg !12
  %1134 = extractelement <4 x float> %1008, i64 0, !dbg !12
  %1135 = extractelement <4 x float> %1008, i64 1, !dbg !12
  %1136 = extractelement <4 x float> %1008, i64 2, !dbg !12
  %1137 = extractelement <4 x float> %1008, i64 3, !dbg !12
  %1138 = select i1 %.not, float %1135, float %1134, !dbg !12
  %1139 = bitcast float %1138 to i32, !dbg !12
  %1140 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1139), !dbg !12
  %1141 = bitcast i32 %1140 to float, !dbg !12
  %1142 = select i1 %.not, float %1134, float %1141, !dbg !12
  %1143 = select i1 %.not, float %1141, float %1135, !dbg !12
  %1144 = select i1 %.not, float %1137, float %1136, !dbg !12
  %1145 = bitcast float %1144 to i32, !dbg !12
  %1146 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1145), !dbg !12
  %1147 = bitcast i32 %1146 to float, !dbg !12
  %1148 = select i1 %.not, float %1136, float %1147, !dbg !12
  %1149 = select i1 %.not, float %1147, float %1137, !dbg !12
  %1150 = select i1 %.not40, float %1148, float %1142, !dbg !12
  %1151 = bitcast float %1150 to i32, !dbg !12
  %1152 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1151), !dbg !12
  %1153 = bitcast i32 %1152 to float, !dbg !12
  %1154 = select i1 %.not40, float %1149, float %1143, !dbg !12
  %1155 = bitcast float %1154 to i32, !dbg !12
  %1156 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1155), !dbg !12
  %1157 = bitcast i32 %1156 to float, !dbg !12
  %1158 = insertelement <4 x float> poison, float %1142, i64 0, !dbg !12
  %1159 = insertelement <4 x float> %1158, float %1143, i64 1, !dbg !12
  %1160 = insertelement <4 x float> %1159, float %1153, i64 2, !dbg !12
  %1161 = insertelement <4 x float> %1160, float %1157, i64 3, !dbg !12
  %1162 = insertelement <4 x float> poison, float %1153, i64 0, !dbg !12
  %1163 = insertelement <4 x float> %1162, float %1157, i64 1, !dbg !12
  %1164 = insertelement <4 x float> %1163, float %1148, i64 2, !dbg !12
  %1165 = insertelement <4 x float> %1164, float %1149, i64 3, !dbg !12
  %1166 = select <4 x i1> %1058, <4 x float> %1161, <4 x float> %1165, !dbg !12
  %1167 = extractelement <4 x float> %933, i64 0, !dbg !12
  %1168 = extractelement <4 x float> %933, i64 1, !dbg !12
  %1169 = extractelement <4 x float> %933, i64 2, !dbg !12
  %1170 = extractelement <4 x float> %933, i64 3, !dbg !12
  %1171 = select i1 %.not, float %1168, float %1167, !dbg !12
  %1172 = bitcast float %1171 to i32, !dbg !12
  %1173 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1172), !dbg !12
  %1174 = bitcast i32 %1173 to float, !dbg !12
  %1175 = select i1 %.not, float %1167, float %1174, !dbg !12
  %1176 = select i1 %.not, float %1174, float %1168, !dbg !12
  %1177 = select i1 %.not, float %1170, float %1169, !dbg !12
  %1178 = bitcast float %1177 to i32, !dbg !12
  %1179 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1178), !dbg !12
  %1180 = bitcast i32 %1179 to float, !dbg !12
  %1181 = select i1 %.not, float %1169, float %1180, !dbg !12
  %1182 = select i1 %.not, float %1180, float %1170, !dbg !12
  %1183 = select i1 %.not40, float %1181, float %1175, !dbg !12
  %1184 = bitcast float %1183 to i32, !dbg !12
  %1185 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1184), !dbg !12
  %1186 = bitcast i32 %1185 to float, !dbg !12
  %1187 = select i1 %.not40, float %1182, float %1176, !dbg !12
  %1188 = bitcast float %1187 to i32, !dbg !12
  %1189 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1188), !dbg !12
  %1190 = bitcast i32 %1189 to float, !dbg !12
  %1191 = insertelement <4 x float> poison, float %1175, i64 0, !dbg !12
  %1192 = insertelement <4 x float> %1191, float %1176, i64 1, !dbg !12
  %1193 = insertelement <4 x float> %1192, float %1186, i64 2, !dbg !12
  %1194 = insertelement <4 x float> %1193, float %1190, i64 3, !dbg !12
  %1195 = insertelement <4 x float> poison, float %1186, i64 0, !dbg !12
  %1196 = insertelement <4 x float> %1195, float %1190, i64 1, !dbg !12
  %1197 = insertelement <4 x float> %1196, float %1181, i64 2, !dbg !12
  %1198 = insertelement <4 x float> %1197, float %1182, i64 3, !dbg !12
  %1199 = select <4 x i1> %1058, <4 x float> %1194, <4 x float> %1198, !dbg !12
  %1200 = extractelement <4 x float> %935, i64 0, !dbg !12
  %1201 = extractelement <4 x float> %935, i64 1, !dbg !12
  %1202 = extractelement <4 x float> %935, i64 2, !dbg !12
  %1203 = extractelement <4 x float> %935, i64 3, !dbg !12
  %1204 = select i1 %.not, float %1201, float %1200, !dbg !12
  %1205 = bitcast float %1204 to i32, !dbg !12
  %1206 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1205), !dbg !12
  %1207 = bitcast i32 %1206 to float, !dbg !12
  %1208 = select i1 %.not, float %1200, float %1207, !dbg !12
  %1209 = select i1 %.not, float %1207, float %1201, !dbg !12
  %1210 = select i1 %.not, float %1203, float %1202, !dbg !12
  %1211 = bitcast float %1210 to i32, !dbg !12
  %1212 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1211), !dbg !12
  %1213 = bitcast i32 %1212 to float, !dbg !12
  %1214 = select i1 %.not, float %1202, float %1213, !dbg !12
  %1215 = select i1 %.not, float %1213, float %1203, !dbg !12
  %1216 = select i1 %.not40, float %1214, float %1208, !dbg !12
  %1217 = bitcast float %1216 to i32, !dbg !12
  %1218 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1217), !dbg !12
  %1219 = bitcast i32 %1218 to float, !dbg !12
  %1220 = select i1 %.not40, float %1215, float %1209, !dbg !12
  %1221 = bitcast float %1220 to i32, !dbg !12
  %1222 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1221), !dbg !12
  %1223 = bitcast i32 %1222 to float, !dbg !12
  %1224 = insertelement <4 x float> poison, float %1208, i64 0, !dbg !12
  %1225 = insertelement <4 x float> %1224, float %1209, i64 1, !dbg !12
  %1226 = insertelement <4 x float> %1225, float %1219, i64 2, !dbg !12
  %1227 = insertelement <4 x float> %1226, float %1223, i64 3, !dbg !12
  %1228 = insertelement <4 x float> poison, float %1219, i64 0, !dbg !12
  %1229 = insertelement <4 x float> %1228, float %1223, i64 1, !dbg !12
  %1230 = insertelement <4 x float> %1229, float %1214, i64 2, !dbg !12
  %1231 = insertelement <4 x float> %1230, float %1215, i64 3, !dbg !12
  %1232 = select <4 x i1> %1058, <4 x float> %1227, <4 x float> %1231, !dbg !12
  %1233 = extractelement <4 x float> %936, i64 0, !dbg !12
  %1234 = extractelement <4 x float> %936, i64 1, !dbg !12
  %1235 = extractelement <4 x float> %936, i64 2, !dbg !12
  %1236 = extractelement <4 x float> %936, i64 3, !dbg !12
  %1237 = select i1 %.not, float %1234, float %1233, !dbg !12
  %1238 = bitcast float %1237 to i32, !dbg !12
  %1239 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1238), !dbg !12
  %1240 = bitcast i32 %1239 to float, !dbg !12
  %1241 = select i1 %.not, float %1233, float %1240, !dbg !12
  %1242 = select i1 %.not, float %1240, float %1234, !dbg !12
  %1243 = select i1 %.not, float %1236, float %1235, !dbg !12
  %1244 = bitcast float %1243 to i32, !dbg !12
  %1245 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1244), !dbg !12
  %1246 = bitcast i32 %1245 to float, !dbg !12
  %1247 = select i1 %.not, float %1235, float %1246, !dbg !12
  %1248 = select i1 %.not, float %1246, float %1236, !dbg !12
  %1249 = select i1 %.not40, float %1247, float %1241, !dbg !12
  %1250 = bitcast float %1249 to i32, !dbg !12
  %1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1250), !dbg !12
  %1252 = bitcast i32 %1251 to float, !dbg !12
  %1253 = select i1 %.not40, float %1248, float %1242, !dbg !12
  %1254 = bitcast float %1253 to i32, !dbg !12
  %1255 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1254), !dbg !12
  %1256 = bitcast i32 %1255 to float, !dbg !12
  %1257 = insertelement <4 x float> poison, float %1241, i64 0, !dbg !12
  %1258 = insertelement <4 x float> %1257, float %1242, i64 1, !dbg !12
  %1259 = insertelement <4 x float> %1258, float %1252, i64 2, !dbg !12
  %1260 = insertelement <4 x float> %1259, float %1256, i64 3, !dbg !12
  %1261 = insertelement <4 x float> poison, float %1252, i64 0, !dbg !12
  %1262 = insertelement <4 x float> %1261, float %1256, i64 1, !dbg !12
  %1263 = insertelement <4 x float> %1262, float %1247, i64 2, !dbg !12
  %1264 = insertelement <4 x float> %1263, float %1248, i64 3, !dbg !12
  %1265 = select <4 x i1> %1058, <4 x float> %1260, <4 x float> %1264, !dbg !12
  %1266 = extractelement <4 x float> %937, i64 0, !dbg !12
  %1267 = extractelement <4 x float> %937, i64 1, !dbg !12
  %1268 = extractelement <4 x float> %937, i64 2, !dbg !12
  %1269 = extractelement <4 x float> %937, i64 3, !dbg !12
  %1270 = select i1 %.not, float %1267, float %1266, !dbg !12
  %1271 = bitcast float %1270 to i32, !dbg !12
  %1272 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1271), !dbg !12
  %1273 = bitcast i32 %1272 to float, !dbg !12
  %1274 = select i1 %.not, float %1266, float %1273, !dbg !12
  %1275 = select i1 %.not, float %1273, float %1267, !dbg !12
  %1276 = select i1 %.not, float %1269, float %1268, !dbg !12
  %1277 = bitcast float %1276 to i32, !dbg !12
  %1278 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1277), !dbg !12
  %1279 = bitcast i32 %1278 to float, !dbg !12
  %1280 = select i1 %.not, float %1268, float %1279, !dbg !12
  %1281 = select i1 %.not, float %1279, float %1269, !dbg !12
  %1282 = select i1 %.not40, float %1280, float %1274, !dbg !12
  %1283 = bitcast float %1282 to i32, !dbg !12
  %1284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1283), !dbg !12
  %1285 = bitcast i32 %1284 to float, !dbg !12
  %1286 = select i1 %.not40, float %1281, float %1275, !dbg !12
  %1287 = bitcast float %1286 to i32, !dbg !12
  %1288 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1287), !dbg !12
  %1289 = bitcast i32 %1288 to float, !dbg !12
  %1290 = insertelement <4 x float> poison, float %1274, i64 0, !dbg !12
  %1291 = insertelement <4 x float> %1290, float %1275, i64 1, !dbg !12
  %1292 = insertelement <4 x float> %1291, float %1285, i64 2, !dbg !12
  %1293 = insertelement <4 x float> %1292, float %1289, i64 3, !dbg !12
  %1294 = insertelement <4 x float> poison, float %1285, i64 0, !dbg !12
  %1295 = insertelement <4 x float> %1294, float %1289, i64 1, !dbg !12
  %1296 = insertelement <4 x float> %1295, float %1280, i64 2, !dbg !12
  %1297 = insertelement <4 x float> %1296, float %1281, i64 3, !dbg !12
  %1298 = select <4 x i1> %1058, <4 x float> %1293, <4 x float> %1297, !dbg !12
  %1299 = extractelement <4 x float> %861, i64 0, !dbg !12
  %1300 = extractelement <4 x float> %861, i64 1, !dbg !12
  %1301 = extractelement <4 x float> %861, i64 2, !dbg !12
  %1302 = extractelement <4 x float> %861, i64 3, !dbg !12
  %1303 = select i1 %.not, float %1300, float %1299, !dbg !12
  %1304 = bitcast float %1303 to i32, !dbg !12
  %1305 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1304), !dbg !12
  %1306 = bitcast i32 %1305 to float, !dbg !12
  %1307 = select i1 %.not, float %1299, float %1306, !dbg !12
  %1308 = select i1 %.not, float %1306, float %1300, !dbg !12
  %1309 = select i1 %.not, float %1302, float %1301, !dbg !12
  %1310 = bitcast float %1309 to i32, !dbg !12
  %1311 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1310), !dbg !12
  %1312 = bitcast i32 %1311 to float, !dbg !12
  %1313 = select i1 %.not, float %1301, float %1312, !dbg !12
  %1314 = select i1 %.not, float %1312, float %1302, !dbg !12
  %1315 = select i1 %.not40, float %1313, float %1307, !dbg !12
  %1316 = bitcast float %1315 to i32, !dbg !12
  %1317 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1316), !dbg !12
  %1318 = bitcast i32 %1317 to float, !dbg !12
  %1319 = select i1 %.not40, float %1314, float %1308, !dbg !12
  %1320 = bitcast float %1319 to i32, !dbg !12
  %1321 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1320), !dbg !12
  %1322 = bitcast i32 %1321 to float, !dbg !12
  %1323 = insertelement <4 x float> poison, float %1307, i64 0, !dbg !12
  %1324 = insertelement <4 x float> %1323, float %1308, i64 1, !dbg !12
  %1325 = insertelement <4 x float> %1324, float %1318, i64 2, !dbg !12
  %1326 = insertelement <4 x float> %1325, float %1322, i64 3, !dbg !12
  %1327 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !12
  %1328 = insertelement <4 x float> %1327, float %1322, i64 1, !dbg !12
  %1329 = insertelement <4 x float> %1328, float %1313, i64 2, !dbg !12
  %1330 = insertelement <4 x float> %1329, float %1314, i64 3, !dbg !12
  %1331 = select <4 x i1> %1058, <4 x float> %1326, <4 x float> %1330, !dbg !12
  %1332 = extractelement <4 x float> %863, i64 0, !dbg !12
  %1333 = extractelement <4 x float> %863, i64 1, !dbg !12
  %1334 = extractelement <4 x float> %863, i64 2, !dbg !12
  %1335 = extractelement <4 x float> %863, i64 3, !dbg !12
  %1336 = select i1 %.not, float %1333, float %1332, !dbg !12
  %1337 = bitcast float %1336 to i32, !dbg !12
  %1338 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1337), !dbg !12
  %1339 = bitcast i32 %1338 to float, !dbg !12
  %1340 = select i1 %.not, float %1332, float %1339, !dbg !12
  %1341 = select i1 %.not, float %1339, float %1333, !dbg !12
  %1342 = select i1 %.not, float %1335, float %1334, !dbg !12
  %1343 = bitcast float %1342 to i32, !dbg !12
  %1344 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1343), !dbg !12
  %1345 = bitcast i32 %1344 to float, !dbg !12
  %1346 = select i1 %.not, float %1334, float %1345, !dbg !12
  %1347 = select i1 %.not, float %1345, float %1335, !dbg !12
  %1348 = select i1 %.not40, float %1346, float %1340, !dbg !12
  %1349 = bitcast float %1348 to i32, !dbg !12
  %1350 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1349), !dbg !12
  %1351 = bitcast i32 %1350 to float, !dbg !12
  %1352 = select i1 %.not40, float %1347, float %1341, !dbg !12
  %1353 = bitcast float %1352 to i32, !dbg !12
  %1354 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1353), !dbg !12
  %1355 = bitcast i32 %1354 to float, !dbg !12
  %1356 = insertelement <4 x float> poison, float %1340, i64 0, !dbg !12
  %1357 = insertelement <4 x float> %1356, float %1341, i64 1, !dbg !12
  %1358 = insertelement <4 x float> %1357, float %1351, i64 2, !dbg !12
  %1359 = insertelement <4 x float> %1358, float %1355, i64 3, !dbg !12
  %1360 = insertelement <4 x float> poison, float %1351, i64 0, !dbg !12
  %1361 = insertelement <4 x float> %1360, float %1355, i64 1, !dbg !12
  %1362 = insertelement <4 x float> %1361, float %1346, i64 2, !dbg !12
  %1363 = insertelement <4 x float> %1362, float %1347, i64 3, !dbg !12
  %1364 = select <4 x i1> %1058, <4 x float> %1359, <4 x float> %1363, !dbg !12
  %1365 = extractelement <4 x float> %864, i64 0, !dbg !12
  %1366 = extractelement <4 x float> %864, i64 1, !dbg !12
  %1367 = extractelement <4 x float> %864, i64 2, !dbg !12
  %1368 = extractelement <4 x float> %864, i64 3, !dbg !12
  %1369 = select i1 %.not, float %1366, float %1365, !dbg !12
  %1370 = bitcast float %1369 to i32, !dbg !12
  %1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1370), !dbg !12
  %1372 = bitcast i32 %1371 to float, !dbg !12
  %1373 = select i1 %.not, float %1365, float %1372, !dbg !12
  %1374 = select i1 %.not, float %1372, float %1366, !dbg !12
  %1375 = select i1 %.not, float %1368, float %1367, !dbg !12
  %1376 = bitcast float %1375 to i32, !dbg !12
  %1377 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1376), !dbg !12
  %1378 = bitcast i32 %1377 to float, !dbg !12
  %1379 = select i1 %.not, float %1367, float %1378, !dbg !12
  %1380 = select i1 %.not, float %1378, float %1368, !dbg !12
  %1381 = select i1 %.not40, float %1379, float %1373, !dbg !12
  %1382 = bitcast float %1381 to i32, !dbg !12
  %1383 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1382), !dbg !12
  %1384 = bitcast i32 %1383 to float, !dbg !12
  %1385 = select i1 %.not40, float %1380, float %1374, !dbg !12
  %1386 = bitcast float %1385 to i32, !dbg !12
  %1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1386), !dbg !12
  %1388 = bitcast i32 %1387 to float, !dbg !12
  %1389 = insertelement <4 x float> poison, float %1373, i64 0, !dbg !12
  %1390 = insertelement <4 x float> %1389, float %1374, i64 1, !dbg !12
  %1391 = insertelement <4 x float> %1390, float %1384, i64 2, !dbg !12
  %1392 = insertelement <4 x float> %1391, float %1388, i64 3, !dbg !12
  %1393 = insertelement <4 x float> poison, float %1384, i64 0, !dbg !12
  %1394 = insertelement <4 x float> %1393, float %1388, i64 1, !dbg !12
  %1395 = insertelement <4 x float> %1394, float %1379, i64 2, !dbg !12
  %1396 = insertelement <4 x float> %1395, float %1380, i64 3, !dbg !12
  %1397 = select <4 x i1> %1058, <4 x float> %1392, <4 x float> %1396, !dbg !12
  %1398 = extractelement <4 x float> %865, i64 0, !dbg !12
  %1399 = extractelement <4 x float> %865, i64 1, !dbg !12
  %1400 = extractelement <4 x float> %865, i64 2, !dbg !12
  %1401 = extractelement <4 x float> %865, i64 3, !dbg !12
  %1402 = select i1 %.not, float %1399, float %1398, !dbg !12
  %1403 = bitcast float %1402 to i32, !dbg !12
  %1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1403), !dbg !12
  %1405 = bitcast i32 %1404 to float, !dbg !12
  %1406 = select i1 %.not, float %1398, float %1405, !dbg !12
  %1407 = select i1 %.not, float %1405, float %1399, !dbg !12
  %1408 = select i1 %.not, float %1401, float %1400, !dbg !12
  %1409 = bitcast float %1408 to i32, !dbg !12
  %1410 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1409), !dbg !12
  %1411 = bitcast i32 %1410 to float, !dbg !12
  %1412 = select i1 %.not, float %1400, float %1411, !dbg !12
  %1413 = select i1 %.not, float %1411, float %1401, !dbg !12
  %1414 = select i1 %.not40, float %1412, float %1406, !dbg !12
  %1415 = bitcast float %1414 to i32, !dbg !12
  %1416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1415), !dbg !12
  %1417 = bitcast i32 %1416 to float, !dbg !12
  %1418 = select i1 %.not40, float %1413, float %1407, !dbg !12
  %1419 = bitcast float %1418 to i32, !dbg !12
  %1420 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1419), !dbg !12
  %1421 = bitcast i32 %1420 to float, !dbg !12
  %1422 = insertelement <4 x float> poison, float %1406, i64 0, !dbg !12
  %1423 = insertelement <4 x float> %1422, float %1407, i64 1, !dbg !12
  %1424 = insertelement <4 x float> %1423, float %1417, i64 2, !dbg !12
  %1425 = insertelement <4 x float> %1424, float %1421, i64 3, !dbg !12
  %1426 = insertelement <4 x float> poison, float %1417, i64 0, !dbg !12
  %1427 = insertelement <4 x float> %1426, float %1421, i64 1, !dbg !12
  %1428 = insertelement <4 x float> %1427, float %1412, i64 2, !dbg !12
  %1429 = insertelement <4 x float> %1428, float %1413, i64 3, !dbg !12
  %1430 = select <4 x i1> %1058, <4 x float> %1425, <4 x float> %1429, !dbg !12
  %1431 = extractelement <4 x float> %770, i64 0, !dbg !12
  %1432 = extractelement <4 x float> %770, i64 1, !dbg !12
  %1433 = extractelement <4 x float> %770, i64 2, !dbg !12
  %1434 = extractelement <4 x float> %770, i64 3, !dbg !12
  %1435 = select i1 %.not, float %1432, float %1431, !dbg !12
  %1436 = bitcast float %1435 to i32, !dbg !12
  %1437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1436), !dbg !12
  %1438 = bitcast i32 %1437 to float, !dbg !12
  %1439 = select i1 %.not, float %1431, float %1438, !dbg !12
  %1440 = select i1 %.not, float %1438, float %1432, !dbg !12
  %1441 = select i1 %.not, float %1434, float %1433, !dbg !12
  %1442 = bitcast float %1441 to i32, !dbg !12
  %1443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1442), !dbg !12
  %1444 = bitcast i32 %1443 to float, !dbg !12
  %1445 = select i1 %.not, float %1433, float %1444, !dbg !12
  %1446 = select i1 %.not, float %1444, float %1434, !dbg !12
  %1447 = select i1 %.not40, float %1445, float %1439, !dbg !12
  %1448 = bitcast float %1447 to i32, !dbg !12
  %1449 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1448), !dbg !12
  %1450 = select i1 %.not40, float %1446, float %1440, !dbg !12
  %1451 = bitcast float %1450 to i32, !dbg !12
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1451), !dbg !12
  %1453 = extractelement <4 x float> %772, i64 0, !dbg !12
  %1454 = extractelement <4 x float> %772, i64 1, !dbg !12
  %1455 = extractelement <4 x float> %772, i64 2, !dbg !12
  %1456 = extractelement <4 x float> %772, i64 3, !dbg !12
  %1457 = select i1 %.not, float %1454, float %1453, !dbg !12
  %1458 = bitcast float %1457 to i32, !dbg !12
  %1459 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1458), !dbg !12
  %1460 = bitcast i32 %1459 to float, !dbg !12
  %1461 = select i1 %.not, float %1453, float %1460, !dbg !12
  %1462 = select i1 %.not, float %1460, float %1454, !dbg !12
  %1463 = select i1 %.not, float %1456, float %1455, !dbg !12
  %1464 = bitcast float %1463 to i32, !dbg !12
  %1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1464), !dbg !12
  %1466 = bitcast i32 %1465 to float, !dbg !12
  %1467 = select i1 %.not, float %1455, float %1466, !dbg !12
  %1468 = select i1 %.not, float %1466, float %1456, !dbg !12
  %1469 = select i1 %.not40, float %1467, float %1461, !dbg !12
  %1470 = bitcast float %1469 to i32, !dbg !12
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1470), !dbg !12
  %1472 = select i1 %.not40, float %1468, float %1462, !dbg !12
  %1473 = bitcast float %1472 to i32, !dbg !12
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1473), !dbg !12
  %1475 = extractelement <4 x float> %792, i64 0, !dbg !12
  %1476 = extractelement <4 x float> %792, i64 1, !dbg !12
  %1477 = extractelement <4 x float> %792, i64 2, !dbg !12
  %1478 = extractelement <4 x float> %792, i64 3, !dbg !12
  %1479 = select i1 %.not, float %1476, float %1475, !dbg !12
  %1480 = bitcast float %1479 to i32, !dbg !12
  %1481 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1480), !dbg !12
  %1482 = bitcast i32 %1481 to float, !dbg !12
  %1483 = select i1 %.not, float %1475, float %1482, !dbg !12
  %1484 = select i1 %.not, float %1482, float %1476, !dbg !12
  %1485 = select i1 %.not, float %1478, float %1477, !dbg !12
  %1486 = bitcast float %1485 to i32, !dbg !12
  %1487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1486), !dbg !12
  %1488 = bitcast i32 %1487 to float, !dbg !12
  %1489 = select i1 %.not, float %1477, float %1488, !dbg !12
  %1490 = select i1 %.not, float %1488, float %1478, !dbg !12
  %1491 = select i1 %.not40, float %1489, float %1483, !dbg !12
  %1492 = bitcast float %1491 to i32, !dbg !12
  %1493 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1492), !dbg !12
  %1494 = select i1 %.not40, float %1490, float %1484, !dbg !12
  %1495 = bitcast float %1494 to i32, !dbg !12
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1495), !dbg !12
  %1497 = extractelement <4 x float> %793, i64 0, !dbg !12
  %1498 = extractelement <4 x float> %793, i64 1, !dbg !12
  %1499 = extractelement <4 x float> %793, i64 2, !dbg !12
  %1500 = extractelement <4 x float> %793, i64 3, !dbg !12
  %1501 = select i1 %.not, float %1498, float %1497, !dbg !12
  %1502 = bitcast float %1501 to i32, !dbg !12
  %1503 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1502), !dbg !12
  %1504 = bitcast i32 %1503 to float, !dbg !12
  %1505 = select i1 %.not, float %1497, float %1504, !dbg !12
  %1506 = select i1 %.not, float %1504, float %1498, !dbg !12
  %1507 = select i1 %.not, float %1500, float %1499, !dbg !12
  %1508 = bitcast float %1507 to i32, !dbg !12
  %1509 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1508), !dbg !12
  %1510 = bitcast i32 %1509 to float, !dbg !12
  %1511 = select i1 %.not, float %1499, float %1510, !dbg !12
  %1512 = select i1 %.not, float %1510, float %1500, !dbg !12
  %1513 = select i1 %.not40, float %1511, float %1505, !dbg !12
  %1514 = bitcast float %1513 to i32, !dbg !12
  %1515 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1514), !dbg !12
  %1516 = select i1 %.not40, float %1512, float %1506, !dbg !12
  %1517 = bitcast float %1516 to i32, !dbg !12
  %1518 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1517), !dbg !12
  %1519 = insertelement <2 x i32> poison, i32 %1030, i64 0, !dbg !12
  %1520 = insertelement <2 x i32> %1519, i32 %1028, i64 1, !dbg !12
  %1521 = bitcast <2 x i32> %1520 to <2 x float>, !dbg !12
  %1522 = insertelement <2 x float> poison, float %k0.sum.anchor1897, i64 0, !dbg !12
  %1523 = insertelement <2 x float> %1522, float %k0.sum.anchor1905, i64 1, !dbg !12
  %1524 = fadd <2 x float> %1523, %1521, !dbg !12
  %bc = bitcast <2 x float> %1524 to <2 x i32>, !dbg !12
  %1525 = extractelement <2 x i32> %bc, i64 1, !dbg !12
  %1526 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1525), !dbg !12
  %bc182 = bitcast <2 x float> %1524 to <2 x i32>, !dbg !12
  %1527 = extractelement <2 x i32> %bc182, i64 0, !dbg !12
  %1528 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1527), !dbg !12
  %1529 = insertelement <2 x i32> poison, i32 %1528, i64 0, !dbg !12
  %1530 = insertelement <2 x i32> %1529, i32 %1526, i64 1, !dbg !12
  %1531 = bitcast <2 x i32> %1530 to <2 x float>, !dbg !12
  %1532 = fadd <2 x float> %1524, %1531, !dbg !12
  %1533 = fdiv arcp <2 x float> splat (float 1.000000e+00), %1532, !dbg !46
  %1534 = shufflevector <2 x float> %1533, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !46
  %1535 = shufflevector <2 x float> %1533, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !46
  %1536 = bitcast i32 %1515 to float, !dbg !12
  %1537 = bitcast i32 %1518 to float, !dbg !12
  %1538 = insertelement <4 x float> poison, float %1505, i64 0, !dbg !12
  %1539 = insertelement <4 x float> %1538, float %1506, i64 1, !dbg !12
  %1540 = insertelement <4 x float> %1539, float %1536, i64 2, !dbg !12
  %1541 = insertelement <4 x float> %1540, float %1537, i64 3, !dbg !12
  %1542 = insertelement <4 x float> poison, float %1536, i64 0, !dbg !12
  %1543 = insertelement <4 x float> %1542, float %1537, i64 1, !dbg !12
  %1544 = insertelement <4 x float> %1543, float %1511, i64 2, !dbg !12
  %1545 = insertelement <4 x float> %1544, float %1512, i64 3, !dbg !12
  %1546 = select <4 x i1> %1058, <4 x float> %1541, <4 x float> %1545, !dbg !12
  %1547 = fmul <4 x float> %1535, %1546, !dbg !46
  %1548 = fptrunc <4 x float> %1547 to <4 x half>, !dbg !47
  %1549 = extractelement <4 x half> %1548, i64 3, !dbg !47
  %1550 = extractelement <4 x half> %1548, i64 2, !dbg !47
  %1551 = extractelement <4 x half> %1548, i64 1, !dbg !47
  %1552 = extractelement <4 x half> %1548, i64 0, !dbg !47
  %1553 = bitcast i32 %1493 to float, !dbg !12
  %1554 = bitcast i32 %1496 to float, !dbg !12
  %1555 = insertelement <4 x float> poison, float %1483, i64 0, !dbg !12
  %1556 = insertelement <4 x float> %1555, float %1484, i64 1, !dbg !12
  %1557 = insertelement <4 x float> %1556, float %1553, i64 2, !dbg !12
  %1558 = insertelement <4 x float> %1557, float %1554, i64 3, !dbg !12
  %1559 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !12
  %1560 = insertelement <4 x float> %1559, float %1554, i64 1, !dbg !12
  %1561 = insertelement <4 x float> %1560, float %1489, i64 2, !dbg !12
  %1562 = insertelement <4 x float> %1561, float %1490, i64 3, !dbg !12
  %1563 = select <4 x i1> %1058, <4 x float> %1558, <4 x float> %1562, !dbg !12
  %1564 = fmul <4 x float> %1535, %1563, !dbg !46
  %1565 = fptrunc <4 x float> %1564 to <4 x half>, !dbg !47
  %1566 = extractelement <4 x half> %1565, i64 3, !dbg !47
  %1567 = extractelement <4 x half> %1565, i64 2, !dbg !47
  %1568 = extractelement <4 x half> %1565, i64 1, !dbg !47
  %1569 = extractelement <4 x half> %1565, i64 0, !dbg !47
  %1570 = bitcast i32 %1471 to float, !dbg !12
  %1571 = bitcast i32 %1474 to float, !dbg !12
  %1572 = insertelement <4 x float> poison, float %1461, i64 0, !dbg !12
  %1573 = insertelement <4 x float> %1572, float %1462, i64 1, !dbg !12
  %1574 = insertelement <4 x float> %1573, float %1570, i64 2, !dbg !12
  %1575 = insertelement <4 x float> %1574, float %1571, i64 3, !dbg !12
  %1576 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !12
  %1577 = insertelement <4 x float> %1576, float %1571, i64 1, !dbg !12
  %1578 = insertelement <4 x float> %1577, float %1467, i64 2, !dbg !12
  %1579 = insertelement <4 x float> %1578, float %1468, i64 3, !dbg !12
  %1580 = select <4 x i1> %1058, <4 x float> %1575, <4 x float> %1579, !dbg !12
  %1581 = fmul <4 x float> %1534, %1580, !dbg !46
  %1582 = fptrunc <4 x float> %1581 to <4 x half>, !dbg !47
  %1583 = extractelement <4 x half> %1582, i64 3, !dbg !47
  %1584 = extractelement <4 x half> %1582, i64 2, !dbg !47
  %1585 = extractelement <4 x half> %1582, i64 1, !dbg !47
  %1586 = extractelement <4 x half> %1582, i64 0, !dbg !47
  %1587 = bitcast i32 %1449 to float, !dbg !12
  %1588 = bitcast i32 %1452 to float, !dbg !12
  %1589 = insertelement <4 x float> poison, float %1439, i64 0, !dbg !12
  %1590 = insertelement <4 x float> %1589, float %1440, i64 1, !dbg !12
  %1591 = insertelement <4 x float> %1590, float %1587, i64 2, !dbg !12
  %1592 = insertelement <4 x float> %1591, float %1588, i64 3, !dbg !12
  %1593 = insertelement <4 x float> poison, float %1587, i64 0, !dbg !12
  %1594 = insertelement <4 x float> %1593, float %1588, i64 1, !dbg !12
  %1595 = insertelement <4 x float> %1594, float %1445, i64 2, !dbg !12
  %1596 = insertelement <4 x float> %1595, float %1446, i64 3, !dbg !12
  %1597 = select <4 x i1> %1058, <4 x float> %1592, <4 x float> %1596, !dbg !12
  %1598 = fmul <4 x float> %1534, %1597, !dbg !46
  %1599 = fptrunc <4 x float> %1598 to <4 x half>, !dbg !47
  %1600 = extractelement <4 x half> %1599, i64 3, !dbg !47
  %1601 = extractelement <4 x half> %1599, i64 2, !dbg !47
  %1602 = extractelement <4 x half> %1599, i64 1, !dbg !47
  %1603 = extractelement <4 x half> %1599, i64 0, !dbg !47
  %address.thread837 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !48, !range !10, !frisk.register_lifetimes !5
  %address.local838 = zext nneg i32 %address.thread837 to i64, !dbg !9
  %1604 = lshr i64 %address.local838, 2, !dbg !48
  %address.local841 = and i64 %1604, 48, !dbg !48
  %address.local843 = and i64 %address.local838, 240, !dbg !48
  %address.local842 = add i64 %34, %address.local838, !dbg !48
  %address.local844 = sub i64 %address.local842, %address.local843, !dbg !48
  %address.local845 = add i64 %address.local844, %address.local841, !dbg !48
  %1605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !48
  %address.local848.idx = shl nuw nsw i64 %address.local845, 8, !dbg !48
  %address.local848 = getelementptr inbounds nuw i8, ptr addrspace(1) %1605, i64 %address.local848.idx, !dbg !48
  %address.local850 = and i64 %1604, 12, !dbg !11
  %address.local851 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local850, !dbg !48
  store half %1603, ptr addrspace(1) %address.local851, align 2, !dbg !48
  %address.local853 = or disjoint i64 %address.local850, 1, !dbg !48
  %address.local854 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local853, !dbg !48
  store half %1602, ptr addrspace(1) %address.local854, align 2, !dbg !48
  %address.local856 = or disjoint i64 %address.local850, 2, !dbg !48
  %address.local857 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local856, !dbg !48
  store half %1601, ptr addrspace(1) %address.local857, align 2, !dbg !48
  %address.local859 = or disjoint i64 %address.local850, 3, !dbg !48
  %address.local860 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local859, !dbg !48
  store half %1600, ptr addrspace(1) %address.local860, align 2, !dbg !48
  %address.local862 = or disjoint i64 %address.local850, 16, !dbg !48
  %address.local863 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local862, !dbg !48
  store half %1586, ptr addrspace(1) %address.local863, align 2, !dbg !48
  %address.local865 = or disjoint i64 %address.local850, 17, !dbg !48
  %address.local866 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local865, !dbg !48
  store half %1585, ptr addrspace(1) %address.local866, align 2, !dbg !48
  %address.local868 = or disjoint i64 %address.local850, 18, !dbg !48
  %address.local869 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local868, !dbg !48
  store half %1584, ptr addrspace(1) %address.local869, align 2, !dbg !48
  %address.local871 = or disjoint i64 %address.local850, 19, !dbg !48
  %address.local872 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local871, !dbg !48
  store half %1583, ptr addrspace(1) %address.local872, align 2, !dbg !48
  %address.local876 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local848, i64 16384, !dbg !48
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local850, !dbg !48
  store half %1569, ptr addrspace(1) %address.local877, align 2, !dbg !48
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local853, !dbg !48
  store half %1568, ptr addrspace(1) %address.local879, align 2, !dbg !48
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local856, !dbg !48
  store half %1567, ptr addrspace(1) %address.local881, align 2, !dbg !48
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local859, !dbg !48
  store half %1566, ptr addrspace(1) %address.local883, align 2, !dbg !48
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local862, !dbg !48
  store half %1552, ptr addrspace(1) %address.local885, align 2, !dbg !48
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local865, !dbg !48
  store half %1551, ptr addrspace(1) %address.local887, align 2, !dbg !48
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local868, !dbg !48
  store half %1550, ptr addrspace(1) %address.local889, align 2, !dbg !48
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local871, !dbg !48
  store half %1549, ptr addrspace(1) %address.local891, align 2, !dbg !48
  %1606 = fmul <4 x float> %1535, %1430, !dbg !46
  %1607 = fptrunc <4 x float> %1606 to <4 x half>, !dbg !47
  %1608 = extractelement <4 x half> %1607, i64 3, !dbg !47
  %1609 = extractelement <4 x half> %1607, i64 2, !dbg !47
  %1610 = extractelement <4 x half> %1607, i64 1, !dbg !47
  %1611 = extractelement <4 x half> %1607, i64 0, !dbg !47
  %1612 = fmul <4 x float> %1535, %1397, !dbg !46
  %1613 = fptrunc <4 x float> %1612 to <4 x half>, !dbg !47
  %1614 = extractelement <4 x half> %1613, i64 3, !dbg !47
  %1615 = extractelement <4 x half> %1613, i64 2, !dbg !47
  %1616 = extractelement <4 x half> %1613, i64 1, !dbg !47
  %1617 = extractelement <4 x half> %1613, i64 0, !dbg !47
  %1618 = fmul <4 x float> %1534, %1364, !dbg !46
  %1619 = fptrunc <4 x float> %1618 to <4 x half>, !dbg !47
  %1620 = extractelement <4 x half> %1619, i64 3, !dbg !47
  %1621 = extractelement <4 x half> %1619, i64 2, !dbg !47
  %1622 = extractelement <4 x half> %1619, i64 1, !dbg !47
  %1623 = extractelement <4 x half> %1619, i64 0, !dbg !47
  %1624 = fmul <4 x float> %1534, %1331, !dbg !46
  %1625 = fptrunc <4 x float> %1624 to <4 x half>, !dbg !47
  %1626 = extractelement <4 x half> %1625, i64 3, !dbg !47
  %1627 = extractelement <4 x half> %1625, i64 2, !dbg !47
  %1628 = extractelement <4 x half> %1625, i64 1, !dbg !47
  %1629 = extractelement <4 x half> %1625, i64 0, !dbg !47
  %address.local906 = or disjoint i64 %address.local850, 32, !dbg !48
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local906, !dbg !48
  store half %1629, ptr addrspace(1) %address.local907, align 2, !dbg !48
  %address.local909 = or disjoint i64 %address.local850, 33, !dbg !48
  %address.local910 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local909, !dbg !48
  store half %1628, ptr addrspace(1) %address.local910, align 2, !dbg !48
  %address.local912 = or disjoint i64 %address.local850, 34, !dbg !48
  %address.local913 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local912, !dbg !48
  store half %1627, ptr addrspace(1) %address.local913, align 2, !dbg !48
  %address.local915 = or disjoint i64 %address.local850, 35, !dbg !48
  %address.local916 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local915, !dbg !48
  store half %1626, ptr addrspace(1) %address.local916, align 2, !dbg !48
  %address.local918 = or disjoint i64 %address.local850, 48, !dbg !48
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local918, !dbg !48
  store half %1623, ptr addrspace(1) %address.local919, align 2, !dbg !48
  %address.local921 = or disjoint i64 %address.local850, 49, !dbg !48
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local921, !dbg !48
  store half %1622, ptr addrspace(1) %address.local922, align 2, !dbg !48
  %address.local924 = or disjoint i64 %address.local850, 50, !dbg !48
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local924, !dbg !48
  store half %1621, ptr addrspace(1) %address.local925, align 2, !dbg !48
  %address.local927 = or i64 %1604, 51, !dbg !48
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local927, !dbg !48
  store half %1620, ptr addrspace(1) %address.local928, align 2, !dbg !48
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local906, !dbg !48
  store half %1617, ptr addrspace(1) %address.local933, align 2, !dbg !48
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local909, !dbg !48
  store half %1616, ptr addrspace(1) %address.local935, align 2, !dbg !48
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local912, !dbg !48
  store half %1615, ptr addrspace(1) %address.local937, align 2, !dbg !48
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local915, !dbg !48
  store half %1614, ptr addrspace(1) %address.local939, align 2, !dbg !48
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local918, !dbg !48
  store half %1611, ptr addrspace(1) %address.local941, align 2, !dbg !48
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local921, !dbg !48
  store half %1610, ptr addrspace(1) %address.local943, align 2, !dbg !48
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local924, !dbg !48
  store half %1609, ptr addrspace(1) %address.local945, align 2, !dbg !48
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local927, !dbg !48
  store half %1608, ptr addrspace(1) %address.local947, align 2, !dbg !48
  %1630 = fmul <4 x float> %1535, %1298, !dbg !46
  %1631 = fptrunc <4 x float> %1630 to <4 x half>, !dbg !47
  %1632 = extractelement <4 x half> %1631, i64 3, !dbg !47
  %1633 = extractelement <4 x half> %1631, i64 2, !dbg !47
  %1634 = extractelement <4 x half> %1631, i64 1, !dbg !47
  %1635 = extractelement <4 x half> %1631, i64 0, !dbg !47
  %1636 = fmul <4 x float> %1535, %1265, !dbg !46
  %1637 = fptrunc <4 x float> %1636 to <4 x half>, !dbg !47
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !47
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !47
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !47
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !47
  %1642 = fmul <4 x float> %1534, %1232, !dbg !46
  %1643 = fptrunc <4 x float> %1642 to <4 x half>, !dbg !47
  %1644 = extractelement <4 x half> %1643, i64 3, !dbg !47
  %1645 = extractelement <4 x half> %1643, i64 2, !dbg !47
  %1646 = extractelement <4 x half> %1643, i64 1, !dbg !47
  %1647 = extractelement <4 x half> %1643, i64 0, !dbg !47
  %1648 = fmul <4 x float> %1534, %1199, !dbg !46
  %1649 = fptrunc <4 x float> %1648 to <4 x half>, !dbg !47
  %1650 = extractelement <4 x half> %1649, i64 3, !dbg !47
  %1651 = extractelement <4 x half> %1649, i64 2, !dbg !47
  %1652 = extractelement <4 x half> %1649, i64 1, !dbg !47
  %1653 = extractelement <4 x half> %1649, i64 0, !dbg !47
  %address.local962 = or disjoint i64 %address.local850, 64, !dbg !48
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local962, !dbg !48
  store half %1653, ptr addrspace(1) %address.local963, align 2, !dbg !48
  %address.local965 = or disjoint i64 %address.local850, 65, !dbg !48
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local965, !dbg !48
  store half %1652, ptr addrspace(1) %address.local966, align 2, !dbg !48
  %address.local968 = or disjoint i64 %address.local850, 66, !dbg !48
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local968, !dbg !48
  store half %1651, ptr addrspace(1) %address.local969, align 2, !dbg !48
  %address.local971 = or disjoint i64 %address.local850, 67, !dbg !48
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local971, !dbg !48
  store half %1650, ptr addrspace(1) %address.local972, align 2, !dbg !48
  %address.local974 = or disjoint i64 %address.local850, 80, !dbg !48
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local974, !dbg !48
  store half %1647, ptr addrspace(1) %address.local975, align 2, !dbg !48
  %address.local977 = or disjoint i64 %address.local850, 81, !dbg !48
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local977, !dbg !48
  store half %1646, ptr addrspace(1) %address.local978, align 2, !dbg !48
  %address.local980 = or disjoint i64 %address.local850, 82, !dbg !48
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local980, !dbg !48
  store half %1645, ptr addrspace(1) %address.local981, align 2, !dbg !48
  %address.local983 = or disjoint i64 %address.local850, 83, !dbg !48
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local983, !dbg !48
  store half %1644, ptr addrspace(1) %address.local984, align 2, !dbg !48
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local962, !dbg !48
  store half %1641, ptr addrspace(1) %address.local989, align 2, !dbg !48
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local965, !dbg !48
  store half %1640, ptr addrspace(1) %address.local991, align 2, !dbg !48
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local968, !dbg !48
  store half %1639, ptr addrspace(1) %address.local993, align 2, !dbg !48
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local971, !dbg !48
  store half %1638, ptr addrspace(1) %address.local995, align 2, !dbg !48
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local974, !dbg !48
  store half %1635, ptr addrspace(1) %address.local997, align 2, !dbg !48
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local977, !dbg !48
  store half %1634, ptr addrspace(1) %address.local999, align 2, !dbg !48
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local980, !dbg !48
  store half %1633, ptr addrspace(1) %address.local1001, align 2, !dbg !48
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local983, !dbg !48
  store half %1632, ptr addrspace(1) %address.local1003, align 2, !dbg !48
  %1654 = fmul <4 x float> %1535, %1166, !dbg !46
  %1655 = fptrunc <4 x float> %1654 to <4 x half>, !dbg !47
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !47
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !47
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !47
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !47
  %1660 = fmul <4 x float> %1535, %1133, !dbg !46
  %1661 = fptrunc <4 x float> %1660 to <4 x half>, !dbg !47
  %1662 = extractelement <4 x half> %1661, i64 3, !dbg !47
  %1663 = extractelement <4 x half> %1661, i64 2, !dbg !47
  %1664 = extractelement <4 x half> %1661, i64 1, !dbg !47
  %1665 = extractelement <4 x half> %1661, i64 0, !dbg !47
  %1666 = fmul <4 x float> %1534, %1100, !dbg !46
  %1667 = fptrunc <4 x float> %1666 to <4 x half>, !dbg !47
  %1668 = extractelement <4 x half> %1667, i64 3, !dbg !47
  %1669 = extractelement <4 x half> %1667, i64 2, !dbg !47
  %1670 = extractelement <4 x half> %1667, i64 1, !dbg !47
  %1671 = extractelement <4 x half> %1667, i64 0, !dbg !47
  %1672 = fmul <4 x float> %1534, %1067, !dbg !46
  %1673 = fptrunc <4 x float> %1672 to <4 x half>, !dbg !47
  %1674 = extractelement <4 x half> %1673, i64 3, !dbg !47
  %1675 = extractelement <4 x half> %1673, i64 2, !dbg !47
  %1676 = extractelement <4 x half> %1673, i64 1, !dbg !47
  %1677 = extractelement <4 x half> %1673, i64 0, !dbg !47
  %address.local1018 = or disjoint i64 %address.local850, 96, !dbg !48
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1018, !dbg !48
  store half %1677, ptr addrspace(1) %address.local1019, align 2, !dbg !48
  %address.local1021 = or disjoint i64 %address.local850, 97, !dbg !48
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1021, !dbg !48
  store half %1676, ptr addrspace(1) %address.local1022, align 2, !dbg !48
  %address.local1024 = or disjoint i64 %address.local850, 98, !dbg !48
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1024, !dbg !48
  store half %1675, ptr addrspace(1) %address.local1025, align 2, !dbg !48
  %address.local1027 = or disjoint i64 %address.local850, 99, !dbg !48
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1027, !dbg !48
  store half %1674, ptr addrspace(1) %address.local1028, align 2, !dbg !48
  %address.local1030 = or disjoint i64 %address.local850, 112, !dbg !48
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1030, !dbg !48
  store half %1671, ptr addrspace(1) %address.local1031, align 2, !dbg !48
  %address.local1033 = or disjoint i64 %address.local850, 113, !dbg !48
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1033, !dbg !48
  store half %1670, ptr addrspace(1) %address.local1034, align 2, !dbg !48
  %address.local1036 = or disjoint i64 %address.local850, 114, !dbg !48
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1036, !dbg !48
  store half %1669, ptr addrspace(1) %address.local1037, align 2, !dbg !48
  %address.local1039 = or i64 %1604, 115, !dbg !48
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1039, !dbg !48
  store half %1668, ptr addrspace(1) %address.local1040, align 2, !dbg !48
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1018, !dbg !48
  store half %1665, ptr addrspace(1) %address.local1045, align 2, !dbg !48
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1021, !dbg !48
  store half %1664, ptr addrspace(1) %address.local1047, align 2, !dbg !48
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1024, !dbg !48
  store half %1663, ptr addrspace(1) %address.local1049, align 2, !dbg !48
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1027, !dbg !48
  store half %1662, ptr addrspace(1) %address.local1051, align 2, !dbg !48
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1030, !dbg !48
  store half %1659, ptr addrspace(1) %address.local1053, align 2, !dbg !48
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1033, !dbg !48
  store half %1658, ptr addrspace(1) %address.local1055, align 2, !dbg !48
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1036, !dbg !48
  store half %1657, ptr addrspace(1) %address.local1057, align 2, !dbg !48
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1039, !dbg !48
  store half %1656, ptr addrspace(1) %address.local1059, align 2, !dbg !48
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
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/noncausal")
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
