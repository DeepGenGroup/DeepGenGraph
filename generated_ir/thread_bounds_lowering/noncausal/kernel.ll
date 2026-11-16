; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/noncausal/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/noncausal/input.mlir"

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
  %address.local265 = zext nneg i32 %37 to i64, !dbg !9
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local265, !dbg !9
  %address.local267 = ptrtoint ptr addrspace(3) %address.local266 to i64, !dbg !9
  %address.local268 = trunc i64 %address.local267 to i32, !dbg !9
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local268), !dbg !9
  %address.local269 = lshr i32 %address.thread, 4, !dbg !9
  %address.local270 = and i32 %address.local269, 8, !dbg !9
  %address.local271 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local272 = and i32 %address.local271, 16256, !dbg !9
  %address.local273 = or disjoint i32 %address.local270, %address.local272, !dbg !9
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1239 = shl i64 %34, 8, !dbg !9
  %40 = add i64 %.idx, %.idx1239, !dbg !9
  %41 = shl nuw nsw i32 %address.local273, 1, !dbg !9
  %42 = trunc i64 %40 to i32, !dbg !9
  %dma.buffer.offset1237 = or disjoint i32 %41, %42, !dbg !9
  %43 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !9
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1238 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1237, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %address.local278 = or disjoint i64 %address.local265, 2048, !dbg !9
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local278, !dbg !9
  %address.local280 = ptrtoint ptr addrspace(3) %address.local279 to i64, !dbg !9
  %address.local281 = trunc i64 %address.local280 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local281), !dbg !9
  %address.local287 = or disjoint i32 %address.local273, 16, !dbg !9
  %47 = shl nuw nsw i32 %address.local287, 1, !dbg !9
  %dma.buffer.offset1235 = or disjoint i32 %47, %42, !dbg !9
  %48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1235, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %49 = shl nuw nsw i64 %31, 19, !dbg !9
  %50 = add i64 %49, %35, !dbg !9
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local265, !dbg !9
  %address.local291 = ptrtoint ptr addrspace(3) %address.local290 to i64, !dbg !9
  %address.local292 = trunc i64 %address.local291 to i32, !dbg !9
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local292), !dbg !9
  %52 = trunc i64 %50 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local273, %52, !dbg !9
  %53 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1233 = or disjoint i32 %53, 64, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1233, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %address.local294 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local278, !dbg !9
  %address.local295 = ptrtoint ptr addrspace(3) %address.local294 to i64, !dbg !9
  %address.local296 = trunc i64 %address.local295 to i32, !dbg !9
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local296), !dbg !9
  %.tr1243 = or disjoint i32 %address.local287, %52, !dbg !9
  %56 = shl i32 %.tr1243, 1, !dbg !9
  %dma.buffer.offset1231 = or disjoint i32 %56, 64, !dbg !9
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1231, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %address.local298 = lshr i32 %address.thread, 2, !dbg !12
  %address.local299 = and i32 %address.local298, 4, !dbg !12
  %address.local300 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local301 = and i32 %address.local300, 384, !dbg !12
  %address.local302 = or disjoint i32 %address.local299, %address.local301, !dbg !12
  %address.local304 = and i32 %36, 120, !dbg !12
  %address.local305 = or disjoint i32 %address.local302, %address.local304, !dbg !12
  %address.local306 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local307 = and i32 %address.local306, 1024, !dbg !12
  %address.local308 = or disjoint i32 %address.local305, %address.local307, !dbg !12
  %address.local309 = zext nneg i32 %address.local308 to i64, !dbg !12
  %address.local310 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local309, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local310, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep1060 = getelementptr i8, ptr addrspace(3) %address.local310, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1060, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep1061 = getelementptr i8, ptr addrspace(3) %address.local310, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1061, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1229 = or disjoint i32 %53, 128, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1229, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %dma.buffer.offset1227 = or disjoint i32 %56, 128, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1227, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %lds.const.gep1062 = getelementptr i8, ptr addrspace(3) %address.local310, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1062, align 8, !dbg !12
  %lds.const.gep1063 = getelementptr i8, ptr addrspace(3) %address.local310, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1063, align 8, !dbg !12
  %lds.const.gep1064 = getelementptr i8, ptr addrspace(3) %address.local310, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1065 = getelementptr i8, ptr addrspace(3) %address.local310, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1225 = or disjoint i32 %53, 192, !dbg !9
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1225, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %dma.buffer.offset1223 = or disjoint i32 %56, 192, !dbg !9
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1223, <4 x i32> %dma.buffer.rsrc1238) #10, !dbg !9
  %78 = load <4 x half>, ptr addrspace(3) %address.local310, align 8, !dbg !12
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1060, align 8, !dbg !12
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1061, align 8, !dbg !12
  %83 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !12
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !12
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1062, align 8, !dbg !12
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1063, align 8, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1064, align 8, !dbg !12
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1065, align 8, !dbg !12
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
  %k0.prime.1906 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.1907 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.1908 = or disjoint i64 %k0.prime.1906, %k0.prime.1907, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1345 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1346 = shl nuw nsw i64 %k0.prime.kdma.source1345, 9, !dbg !13
  %k0.prime.kdma.source1347 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !13
  %k0.prime.kdma.source1348 = add nuw nsw i64 %k0.prime.kdma.source1346, %k0.prime.kdma.source1347, !dbg !13
  %k0.prime.kdma.source1349 = lshr i64 %k0.prime.kdma.source, 5, !dbg !13
  %k0.prime.kdma.source1350 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1351 = and i64 %k0.prime.kdma.source1350, 1, !dbg !13
  %k0.prime.kdma.source1352 = and i64 %k0.prime.kdma.source1345, %k0.prime.kdma.source1351, !dbg !13
  %k0.prime.kdma.source1353 = or disjoint i64 %k0.prime.kdma.source1349, %k0.prime.kdma.source1352, !dbg !13
  %k0.prime.kdma.source1354 = shl nuw nsw i64 %k0.prime.kdma.source1353, 10, !dbg !13
  %k0.prime.kdma.source1355 = sub nsw i64 %k0.prime.kdma.source1348, %k0.prime.kdma.source1354, !dbg !13
  %k0.prime.kdma.source1356 = and i64 %k0.prime.kdma.source1355, 16256, !dbg !13
  %k0.prime.kdma.source1357 = lshr i32 %kdma.tid, 4, !dbg !13
  %k0.prime.kdma.source1358 = and i32 %k0.prime.kdma.source1357, 8, !dbg !13
  %k0.prime.kdma.source1359 = zext nneg i32 %k0.prime.kdma.source1358 to i64, !dbg !13
  %k0.prime.kdma.source1360 = or disjoint i64 %k0.prime.kdma.source1356, %k0.prime.kdma.source1359, !dbg !13
  %k0.prime.kdma.source1361 = or disjoint i64 %k0.prime.1908, %k0.prime.kdma.source1360, !dbg !13
  %k0.prime.kdma.source1362 = trunc nuw i64 %k0.prime.kdma.source1361 to i32, !dbg !13
  %k0.prime.kdma.source1363 = shl nuw nsw i32 %k0.prime.kdma.source1362, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1363, %kdma.intra, !dbg !13
  %k0.prime.1909 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1910 = insertelement <2 x i64> poison, i64 %k0.prime.1909, i64 0, !dbg !13
  %k0.prime.1911 = bitcast <2 x i64> %k0.prime.1910 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1222 = shufflevector <4 x i32> %k0.prime.1911, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1912 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1385 = add i32 %38, 512, !dbg !13
  %k0.prime.kdma.source1364 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1365 = lshr i64 %k0.prime.kdma.source1364, 4, !dbg !13
  %k0.prime.kdma.source1366 = shl nuw nsw i64 %k0.prime.kdma.source1365, 9, !dbg !13
  %k0.prime.kdma.source1367 = shl nuw nsw i64 %k0.prime.kdma.source1364, 7, !dbg !13
  %k0.prime.kdma.source1368 = add nuw nsw i64 %k0.prime.kdma.source1366, %k0.prime.kdma.source1367, !dbg !13
  %k0.prime.kdma.source1369 = lshr i64 %k0.prime.kdma.source1364, 5, !dbg !13
  %k0.prime.kdma.source1370 = lshr i64 %k0.prime.kdma.source1364, 2, !dbg !13
  %k0.prime.kdma.source1371 = and i64 %k0.prime.kdma.source1370, 1, !dbg !13
  %k0.prime.kdma.source1372 = and i64 %k0.prime.kdma.source1365, %k0.prime.kdma.source1371, !dbg !13
  %k0.prime.kdma.source1373 = add nuw nsw i64 %k0.prime.kdma.source1369, %k0.prime.kdma.source1372, !dbg !13
  %k0.prime.kdma.source1374 = shl nuw nsw i64 %k0.prime.kdma.source1373, 10, !dbg !13
  %k0.prime.kdma.source1375 = sub nsw i64 %k0.prime.kdma.source1368, %k0.prime.kdma.source1374, !dbg !13
  %k0.prime.kdma.source1376 = and i64 %k0.prime.kdma.source1375, 16256, !dbg !13
  %k0.prime.kdma.source1380 = or disjoint i64 %k0.prime.kdma.source1376, %k0.prime.kdma.source1359, !dbg !13
  %k0.prime.kdma.source1381 = or disjoint i64 %k0.prime.1908, %k0.prime.kdma.source1380, !dbg !13
  %k0.prime.kdma.source1382 = trunc nuw i64 %k0.prime.kdma.source1381 to i32, !dbg !13
  %k0.prime.kdma.source1383 = shl nuw nsw i32 %k0.prime.kdma.source1382, 1, !dbg !13
  %k0.prime.kdma.offset1384 = or disjoint i32 %k0.prime.kdma.source1383, %kdma.intra.hi, !dbg !13
  %k0.prime.1913 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1385, i32 %k0.prime.kdma.offset1384, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1387 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !13
  %k0.prime.kdma.source1388 = lshr i64 %k0.prime.kdma.source1387, 4, !dbg !13
  %k0.prime.kdma.source1389 = shl nuw nsw i64 %k0.prime.kdma.source1388, 9, !dbg !13
  %k0.prime.kdma.source1391 = add nuw nsw i64 %k0.prime.kdma.source1389, %k0.prime.kdma.source1347, !dbg !13
  %k0.prime.kdma.source1392 = lshr i64 %k0.prime.kdma.source1387, 5, !dbg !13
  %k0.prime.kdma.source1395 = and i64 %k0.prime.kdma.source1388, %k0.prime.kdma.source1351, !dbg !13
  %k0.prime.kdma.source1396 = or disjoint i64 %k0.prime.kdma.source1392, %k0.prime.kdma.source1395, !dbg !13
  %k0.prime.kdma.source1397 = shl nuw nsw i64 %k0.prime.kdma.source1396, 10, !dbg !13
  %k0.prime.kdma.source1398 = sub nsw i64 %k0.prime.kdma.source1391, %k0.prime.kdma.source1397, !dbg !13
  %k0.prime.kdma.source1399 = and i64 %k0.prime.kdma.source1398, 16256, !dbg !13
  %k0.prime.kdma.source1402 = or disjoint i32 %k0.prime.kdma.source1358, 16, !dbg !13
  %k0.prime.kdma.source1403 = zext nneg i32 %k0.prime.kdma.source1402 to i64, !dbg !13
  %k0.prime.kdma.source1404 = or disjoint i64 %k0.prime.kdma.source1399, %k0.prime.kdma.source1403, !dbg !13
  %k0.prime.kdma.source1405 = or disjoint i64 %k0.prime.1908, %k0.prime.kdma.source1404, !dbg !13
  %k0.prime.kdma.source1406 = trunc nuw i64 %k0.prime.kdma.source1405 to i32, !dbg !13
  %k0.prime.kdma.source1407 = shl nuw nsw i32 %k0.prime.kdma.source1406, 1, !dbg !13
  %k0.prime.kdma.offset1408 = add nuw i32 %k0.prime.kdma.source1407, %kdma.intra, !dbg !13
  %k0.prime.1914 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.prime.kdma.offset1408, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1433 = add i32 %46, 512, !dbg !13
  %k0.prime.kdma.source1411 = or disjoint i64 %k0.prime.kdma.source1364, 256, !dbg !13
  %k0.prime.kdma.source1412 = lshr i64 %k0.prime.kdma.source1411, 4, !dbg !13
  %k0.prime.kdma.source1413 = shl nuw nsw i64 %k0.prime.kdma.source1412, 9, !dbg !13
  %k0.prime.kdma.source1415 = add nuw nsw i64 %k0.prime.kdma.source1413, %k0.prime.kdma.source1367, !dbg !13
  %k0.prime.kdma.source1416 = lshr i64 %k0.prime.kdma.source1411, 5, !dbg !13
  %k0.prime.kdma.source1419 = and i64 %k0.prime.kdma.source1412, %k0.prime.kdma.source1371, !dbg !13
  %k0.prime.kdma.source1420 = add nuw nsw i64 %k0.prime.kdma.source1416, %k0.prime.kdma.source1419, !dbg !13
  %k0.prime.kdma.source1421 = shl nuw nsw i64 %k0.prime.kdma.source1420, 10, !dbg !13
  %k0.prime.kdma.source1422 = sub nsw i64 %k0.prime.kdma.source1415, %k0.prime.kdma.source1421, !dbg !13
  %k0.prime.kdma.source1423 = and i64 %k0.prime.kdma.source1422, 16256, !dbg !13
  %k0.prime.kdma.source1428 = or disjoint i64 %k0.prime.kdma.source1423, %k0.prime.kdma.source1403, !dbg !13
  %k0.prime.kdma.source1429 = or disjoint i64 %k0.prime.1908, %k0.prime.kdma.source1428, !dbg !13
  %k0.prime.kdma.source1430 = trunc nuw i64 %k0.prime.kdma.source1429 to i32, !dbg !13
  %k0.prime.kdma.source1431 = shl nuw nsw i32 %k0.prime.kdma.source1430, 1, !dbg !13
  %k0.prime.kdma.offset1432 = add nuw i32 %k0.prime.kdma.source1431, %kdma.intra.hi, !dbg !13
  %k0.prime.1915 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1433, i32 %k0.prime.kdma.offset1432, <4 x i32> %k0.prime.dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %100, !dbg !6

100:                                              ; preds = %k0.join, %4
  %101 = phi i64 [ %1001, %k0.join ], [ 0, %4 ], !dbg !6
  %102 = phi [2 x <8 x float>] [ %1069, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %103 = phi [2 x <1 x float>] [ %846, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %104 = phi [2 x <1 x float>] [ %lazy.base.join, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %105 = phi [2 x <8 x float>] [ %1073, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %106 = phi [2 x <8 x float>] [ %1077, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %107 = phi [2 x <8 x float>] [ %1081, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %108 = icmp samesign ult i64 %101, 32, !dbg !6
  br i1 %108, label %109, label %1082, !dbg !6

109:                                              ; preds = %100
  %110 = shl i64 %25, 19, !dbg !13
  %111 = and i64 %110, 16252928, !dbg !13
  %112 = shl nuw nsw i64 %101, 14, !dbg !13
  %113 = or disjoint i64 %111, %112, !dbg !13
  %address.thread361 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %address.local362 = zext nneg i32 %address.thread361 to i64, !dbg !9
  %address.local368 = lshr i64 %address.local362, 2, !dbg !13
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0, !dbg !13
  %116 = bitcast <2 x i64> %115 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1222 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1345 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1346 = shl nuw nsw i64 %kdma.source1345, 9, !dbg !13
  %kdma.source1347 = shl nuw nsw i64 %kdma.source, 7, !dbg !13
  %kdma.source1348 = add nuw nsw i64 %kdma.source1346, %kdma.source1347, !dbg !13
  %kdma.source1349 = lshr i64 %kdma.source, 5, !dbg !13
  %kdma.source1350 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1351 = and i64 %kdma.source1350, 1, !dbg !13
  %kdma.source1352 = and i64 %kdma.source1345, %kdma.source1351, !dbg !13
  %kdma.source1353 = or disjoint i64 %kdma.source1349, %kdma.source1352, !dbg !13
  %kdma.source1354 = shl nuw nsw i64 %kdma.source1353, 10, !dbg !13
  %kdma.source1355 = sub nsw i64 %kdma.source1348, %kdma.source1354, !dbg !13
  %kdma.source1356 = and i64 %kdma.source1355, 16256, !dbg !13
  %kdma.source1357 = lshr i32 %kdma.tid, 4, !dbg !13
  %kdma.source1358 = and i32 %kdma.source1357, 8, !dbg !13
  %kdma.source1359 = zext nneg i32 %kdma.source1358 to i64, !dbg !13
  %kdma.source1360 = or disjoint i64 %kdma.source1356, %kdma.source1359, !dbg !13
  %kdma.source1361 = or disjoint i64 %113, %kdma.source1360, !dbg !13
  %kdma.source1362 = trunc nuw i64 %kdma.source1361 to i32, !dbg !13
  %kdma.source1363 = shl nuw nsw i32 %kdma.source1362, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1363, %kdma.intra, !dbg !13
  %kdma.source1364 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1365 = lshr i64 %kdma.source1364, 4, !dbg !13
  %kdma.source1366 = shl nuw nsw i64 %kdma.source1365, 9, !dbg !13
  %kdma.source1367 = shl nuw nsw i64 %kdma.source1364, 7, !dbg !13
  %kdma.source1368 = add nuw nsw i64 %kdma.source1366, %kdma.source1367, !dbg !13
  %kdma.source1369 = lshr i64 %kdma.source1364, 5, !dbg !13
  %kdma.source1370 = lshr i64 %kdma.source1364, 2, !dbg !13
  %kdma.source1371 = and i64 %kdma.source1370, 1, !dbg !13
  %kdma.source1372 = and i64 %kdma.source1365, %kdma.source1371, !dbg !13
  %kdma.source1373 = add nuw nsw i64 %kdma.source1369, %kdma.source1372, !dbg !13
  %kdma.source1374 = shl nuw nsw i64 %kdma.source1373, 10, !dbg !13
  %kdma.source1375 = sub nsw i64 %kdma.source1368, %kdma.source1374, !dbg !13
  %kdma.source1376 = and i64 %kdma.source1375, 16256, !dbg !13
  %kdma.source1380 = or disjoint i64 %kdma.source1376, %kdma.source1359, !dbg !13
  %kdma.source1381 = or disjoint i64 %113, %kdma.source1380, !dbg !13
  %kdma.source1382 = trunc nuw i64 %kdma.source1381 to i32, !dbg !13
  %kdma.source1383 = shl nuw nsw i32 %kdma.source1382, 1, !dbg !13
  %kdma.offset1384 = or disjoint i32 %kdma.source1383, %kdma.intra.hi, !dbg !13
  %kdma.lds1385 = add i32 %38, 512, !dbg !13
  %kdma.source1387 = or disjoint i64 %kdma.source, 256, !dbg !13
  %kdma.source1388 = lshr i64 %kdma.source1387, 4, !dbg !13
  %kdma.source1389 = shl nuw nsw i64 %kdma.source1388, 9, !dbg !13
  %kdma.source1391 = add nuw nsw i64 %kdma.source1389, %kdma.source1347, !dbg !13
  %kdma.source1392 = lshr i64 %kdma.source1387, 5, !dbg !13
  %kdma.source1395 = and i64 %kdma.source1388, %kdma.source1351, !dbg !13
  %kdma.source1396 = or disjoint i64 %kdma.source1392, %kdma.source1395, !dbg !13
  %kdma.source1397 = shl nuw nsw i64 %kdma.source1396, 10, !dbg !13
  %kdma.source1398 = sub nsw i64 %kdma.source1391, %kdma.source1397, !dbg !13
  %kdma.source1399 = and i64 %kdma.source1398, 16256, !dbg !13
  %kdma.source1402 = or disjoint i32 %kdma.source1358, 16, !dbg !13
  %kdma.source1403 = zext nneg i32 %kdma.source1402 to i64, !dbg !13
  %kdma.source1404 = or disjoint i64 %kdma.source1399, %kdma.source1403, !dbg !13
  %kdma.source1405 = or disjoint i64 %113, %kdma.source1404, !dbg !13
  %kdma.source1406 = trunc nuw i64 %kdma.source1405 to i32, !dbg !13
  %kdma.source1407 = shl nuw nsw i32 %kdma.source1406, 1, !dbg !13
  %kdma.offset1408 = add nuw i32 %kdma.source1407, %kdma.intra, !dbg !13
  %kdma.source1411 = or disjoint i64 %kdma.source1364, 256, !dbg !13
  %kdma.source1412 = lshr i64 %kdma.source1411, 4, !dbg !13
  %kdma.source1413 = shl nuw nsw i64 %kdma.source1412, 9, !dbg !13
  %kdma.source1415 = add nuw nsw i64 %kdma.source1413, %kdma.source1367, !dbg !13
  %kdma.source1416 = lshr i64 %kdma.source1411, 5, !dbg !13
  %kdma.source1419 = and i64 %kdma.source1412, %kdma.source1371, !dbg !13
  %kdma.source1420 = add nuw nsw i64 %kdma.source1416, %kdma.source1419, !dbg !13
  %kdma.source1421 = shl nuw nsw i64 %kdma.source1420, 10, !dbg !13
  %kdma.source1422 = sub nsw i64 %kdma.source1415, %kdma.source1421, !dbg !13
  %kdma.source1423 = and i64 %kdma.source1422, 16256, !dbg !13
  %kdma.source1428 = or disjoint i64 %kdma.source1423, %kdma.source1403, !dbg !13
  %kdma.source1429 = or disjoint i64 %113, %kdma.source1428, !dbg !13
  %kdma.source1430 = trunc nuw i64 %kdma.source1429 to i32, !dbg !13
  %kdma.source1431 = shl nuw nsw i32 %kdma.source1430, 1, !dbg !13
  %kdma.offset1432 = add nuw i32 %kdma.source1431, %kdma.intra.hi, !dbg !13
  %kdma.lds1433 = add i32 %46, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %117 = or disjoint i64 %113, 32, !dbg !13
  %kdma.source1451 = or disjoint i64 %117, %kdma.source1360, !dbg !13
  %kdma.source1452 = trunc nuw i64 %kdma.source1451 to i32, !dbg !13
  %kdma.source1453 = shl nuw nsw i32 %kdma.source1452, 1, !dbg !13
  %kdma.offset1454 = or disjoint i32 %kdma.source1453, %kdma.intra, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1454, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1473 = or disjoint i64 %117, %kdma.source1380, !dbg !13
  %kdma.source1474 = trunc nuw i64 %kdma.source1473 to i32, !dbg !13
  %kdma.source1475 = shl nuw nsw i32 %kdma.source1474, 1, !dbg !13
  %kdma.offset1476 = or disjoint i32 %kdma.source1475, %kdma.intra.hi, !dbg !13
  %kdma.lds1477 = add i32 %51, 512, !dbg !13
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1476, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1497 = or disjoint i64 %117, %kdma.source1404, !dbg !13
  %kdma.source1498 = trunc nuw i64 %kdma.source1497 to i32, !dbg !13
  %kdma.source1499 = shl nuw nsw i32 %kdma.source1498, 1, !dbg !13
  %kdma.offset1500 = add nuw i32 %kdma.source1499, %kdma.intra, !dbg !13
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1500, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1521 = or disjoint i64 %117, %kdma.source1428, !dbg !13
  %kdma.source1522 = trunc nuw i64 %kdma.source1521 to i32, !dbg !13
  %kdma.source1523 = shl nuw nsw i32 %kdma.source1522, 1, !dbg !13
  %kdma.offset1524 = add nuw i32 %kdma.source1523, %kdma.intra.hi, !dbg !13
  %kdma.lds1525 = add i32 %55, 512, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1524, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %address.local405 = shl nuw nsw i64 %address.local362, 1, !dbg !15
  %address.local406 = and i64 %address.local405, 24, !dbg !15
  %address.local408 = shl i64 %address.local362, 62, !dbg !15
  %address.local409 = ashr exact i64 %address.local408, 57, !dbg !15
  %address.local410 = and i64 %address.local409, -64, !dbg !15
  %address.local413 = and i64 %address.local368, 12, !dbg !12
  %address.local415 = and i64 %address.local362, 63, !dbg !12
  %address.local416 = lshr i64 %address.local415, 5, !dbg !12
  %address.local417 = mul nuw nsw i64 %address.local416, 1016, !dbg !12
  %.idx1260 = mul nuw nsw i64 %address.local362, 320, !dbg !15
  %122 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1260, !dbg !15
  %.idx1261 = mul nsw i64 %address.local368, -1280, !dbg !15
  %123 = getelementptr i8, ptr addrspace(3) %122, i64 %.idx1261, !dbg !15
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local406, !dbg !15
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local410, !dbg !15
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local413, !dbg !15
  %address.local419 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local417, !dbg !15
  %127 = add nsw i64 %.idx1260, %.idx1261, !dbg !15
  %.idx1836 = shl nuw nsw i64 %address.local406, 1, !dbg !15
  %128 = or disjoint i64 %127, %.idx1836, !dbg !15
  %129 = or disjoint i64 %address.local410, %address.local413, !dbg !15
  %130 = shl nsw i64 %129, 1, !dbg !15
  %131 = add nsw i64 %130, %128, !dbg !15
  %address.local419.idx = mul nuw nsw i64 %address.local416, 2032, !dbg !15
  %132 = add nsw i64 %131, %address.local419.idx, !dbg !15
  %133 = lshr i64 %132, 6, !dbg !15
  %134 = and i64 %133, 8, !dbg !15
  %135 = xor i64 %132, %134, !dbg !15
  %136 = sub i64 %135, %132, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local419, i64 %136, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %lazy.base = extractvalue [2 x <1 x float>] %104, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base1921 = extractvalue [2 x <1 x float>] %104, 1, !dbg !15
  %lazy.seed1922 = shufflevector <1 x float> %lazy.base1921, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> %lazy.seed) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local430 = shl nuw nsw i32 %address.thread361, 1, !dbg !15
  %address.local431 = and i32 %address.local430, 24, !dbg !15
  %address.local432 = shl nuw nsw i32 %address.thread361, 8, !dbg !15
  %address.local433 = and i32 %address.local432, 768, !dbg !15
  %address.local434 = or disjoint i32 %address.local431, %address.local433, !dbg !15
  %address.local435 = shl nuw nsw i32 %address.thread361, 5, !dbg !15
  %address.local436 = and i32 %address.local435, 96, !dbg !15
  %address.local437 = or disjoint i32 %address.local434, %address.local436, !dbg !15
  %address.local438 = zext nneg i32 %address.local437 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %99) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local439 = lshr i32 %address.thread361, 1, !dbg !15
  %address.local440 = and i32 %address.local439, 1, !dbg !15
  %address.local441 = zext nneg i32 %address.local440 to i64, !dbg !15
  %address.local442 = add nsw i64 %address.base, %address.local441, !dbg !15
  %address.local443 = shl i64 %address.local442, 6, !dbg !15
  %address.local444 = sub i64 %address.local438, %address.local443, !dbg !15
  %140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local444, !dbg !15
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local413, !dbg !15
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local417, !dbg !15
  %address.local448 = getelementptr i8, ptr addrspace(3) %142, i64 64, !dbg !15
  %.idx18391842 = add i64 %address.local444, %address.local413, !dbg !15
  %143 = shl i64 %.idx18391842, 1, !dbg !15
  %144 = add i64 %143, %address.local419.idx, !dbg !15
  %145 = add i64 %144, 64, !dbg !15
  %146 = lshr i64 %145, 6, !dbg !15
  %147 = and i64 %146, 8, !dbg !15
  %148 = xor i64 %145, %147, !dbg !15
  %149 = sub i64 %148, %145, !dbg !15
  %kdma.swizzled1710 = getelementptr i8, ptr addrspace(3) %address.local448, i64 %149, !dbg !15
  %150 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1710, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1714 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 128, !dbg !15
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1714, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1711 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 4096, !dbg !15
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1711, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1715 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 4224, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1715, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1712 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1712, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1716 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1716, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1713 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1713, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1717 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1717, align 8, !dbg !15, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %139, <4 x float> %138) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1718 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 1024, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1718, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1722 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 1152, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !15, !frisk.k_lds_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1719 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 5120, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1719, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1723 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 5248, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !15, !frisk.k_lds_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1720 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1720, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1721 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1721, align 8, !dbg !15, !frisk.k_lds_pair !38
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
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> %lazy.seed1922) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> %lazy.seed1922) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> %lazy.seed1922) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> %lazy.seed1922) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> %lazy.seed1922) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> %lazy.seed1922) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> %lazy.seed1922) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> %lazy.seed1922) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %163, <4 x float> %193) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %195 = or disjoint i64 %113, 64, !dbg !13
  %kdma.source1543 = or disjoint i64 %195, %kdma.source1360, !dbg !13
  %kdma.source1544 = trunc nuw i64 %kdma.source1543 to i32, !dbg !13
  %kdma.source1545 = shl nuw nsw i32 %kdma.source1544, 1, !dbg !13
  %kdma.offset1546 = or disjoint i32 %kdma.source1545, %kdma.intra, !dbg !13
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %kdma.offset1546, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1565 = or disjoint i64 %195, %kdma.source1380, !dbg !13
  %kdma.source1566 = trunc nuw i64 %kdma.source1565 to i32, !dbg !13
  %kdma.source1567 = shl nuw nsw i32 %kdma.source1566, 1, !dbg !13
  %kdma.offset1568 = or disjoint i32 %kdma.source1567, %kdma.intra.hi, !dbg !13
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1385, i32 %kdma.offset1568, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1589 = or disjoint i64 %195, %kdma.source1404, !dbg !13
  %kdma.source1590 = trunc nuw i64 %kdma.source1589 to i32, !dbg !13
  %kdma.source1591 = shl nuw nsw i32 %kdma.source1590, 1, !dbg !13
  %kdma.offset1592 = add nuw i32 %kdma.source1591, %kdma.intra, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %kdma.offset1592, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1613 = or disjoint i64 %195, %kdma.source1428, !dbg !13
  %kdma.source1614 = trunc nuw i64 %kdma.source1613 to i32, !dbg !13
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !13
  %kdma.offset1616 = add nuw i32 %kdma.source1615, %kdma.intra.hi, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1433, i32 %kdma.offset1616, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1744 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1744, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1741 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1745 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1745, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1742 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 8192, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1746 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 8320, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1746, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1743 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 12288, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1743, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1747 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 12416, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1747, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1748 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1748, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1752 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !15, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %200, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1749 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1749, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1753 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %210) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1750 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 9216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1750, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1754 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 9344, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1751 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 13312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1755 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1710, i64 13440, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !15, !frisk.k_lds_pair !46
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
  %kdma.source1635 = or disjoint i64 %248, %kdma.source1360, !dbg !13
  %kdma.source1636 = trunc nuw i64 %kdma.source1635 to i32, !dbg !13
  %kdma.source1637 = shl nuw nsw i32 %kdma.source1636, 1, !dbg !13
  %kdma.offset1638 = or disjoint i32 %kdma.source1637, %kdma.intra, !dbg !13
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1638, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1657 = or disjoint i64 %248, %kdma.source1380, !dbg !13
  %kdma.source1658 = trunc nuw i64 %kdma.source1657 to i32, !dbg !13
  %kdma.source1659 = shl nuw nsw i32 %kdma.source1658, 1, !dbg !13
  %kdma.offset1660 = or disjoint i32 %kdma.source1659, %kdma.intra.hi, !dbg !13
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1477, i32 %kdma.offset1660, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1681 = or disjoint i64 %248, %kdma.source1404, !dbg !13
  %kdma.source1682 = trunc nuw i64 %kdma.source1681 to i32, !dbg !13
  %kdma.source1683 = shl nuw nsw i32 %kdma.source1682, 1, !dbg !13
  %kdma.offset1684 = add nuw i32 %kdma.source1683, %kdma.intra, !dbg !13
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1684, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %kdma.source1705 = or disjoint i64 %248, %kdma.source1428, !dbg !13
  %kdma.source1706 = trunc nuw i64 %kdma.source1705 to i32, !dbg !13
  %kdma.source1707 = shl nuw nsw i32 %kdma.source1706, 1, !dbg !13
  %kdma.offset1708 = add nuw i32 %kdma.source1707, %kdma.intra.hi, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1525, i32 %kdma.offset1708, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !47
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1712, align 8, !dbg !15, !frisk.k_lds_pair !47
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !48
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1713, align 8, !dbg !15, !frisk.k_lds_pair !48
  %257 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1710, align 8, !dbg !15, !frisk.k_lds_pair !49
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1714, align 8, !dbg !15, !frisk.k_lds_pair !49
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1711, align 8, !dbg !15, !frisk.k_lds_pair !50
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1715, align 8, !dbg !15, !frisk.k_lds_pair !50
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1716, align 8, !dbg !15, !frisk.k_lds_pair !51
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1720, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %253, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1717, align 8, !dbg !15, !frisk.k_lds_pair !52
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1721, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %255, <4 x float> %263) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1718, align 8, !dbg !15, !frisk.k_lds_pair !53
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %257, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1719, align 8, !dbg !15, !frisk.k_lds_pair !54
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !15, !frisk.k_lds_pair !54
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
  %address.local777 = shl nuw nsw i64 %address.local362, 5, !dbg !16
  %address.local778 = and i64 %address.local777, 8064, !dbg !16
  %address.local779 = shl nuw nsw i64 %address.local362, 3, !dbg !9
  %address.local780 = shl nuw nsw i64 %address.local368, 5, !dbg !16
  %address.local781 = sub nsw i64 %address.local779, %address.local780, !dbg !16
  %address.local782 = add nsw i64 %address.local778, %address.local781, !dbg !16
  %301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx12701271 = add nsw i64 %113, %address.local782, !dbg !16
  %.idx12701271.tr = trunc i64 %.idx12701271 to i32, !dbg !16
  %dma.buffer.offset1205 = shl i32 %.idx12701271.tr, 1, !dbg !16
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !16
  %303 = bitcast <2 x i64> %302 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1206 = shufflevector <4 x i32> %303, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1205, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %address.local784 = shl nuw nsw i32 %address.thread361, 3, !dbg !16
  %address.local785 = and i32 %address.local784, 24, !dbg !16
  %address.local787 = and i32 %address.local435, 8064, !dbg !16
  %address.local788 = or disjoint i32 %address.local785, %address.local787, !dbg !16
  %address.local789 = or disjoint i32 %address.local788, 8192, !dbg !16
  %305 = trunc nuw i64 %113 to i32, !dbg !16
  %.idx12721273.tr = or disjoint i32 %address.local789, %305, !dbg !16
  %dma.buffer.offset1203 = shl nuw nsw i32 %.idx12721273.tr, 1, !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1203, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !15, !frisk.k_tail_pair !33
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1744, align 8, !dbg !15, !frisk.k_tail_pair !33
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !15, !frisk.k_tail_pair !34
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1745, align 8, !dbg !15, !frisk.k_tail_pair !34
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !15, !frisk.k_tail_pair !35
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1746, align 8, !dbg !15, !frisk.k_tail_pair !35
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1743, align 8, !dbg !15, !frisk.k_tail_pair !36
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1747, align 8, !dbg !15, !frisk.k_tail_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %307, <4 x float> %266) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %309, <4 x float> %315) #9, !dbg !15
  %317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %311, <4 x float> %272) #9, !dbg !15
  %318 = extractelement <4 x float> %316, i64 0, !dbg !15
  %319 = call float @llvm.maxnum.f32(float %318, float 0xFFF0000000000000), !dbg !8
  %320 = extractelement <4 x float> %316, i64 1, !dbg !15
  %321 = extractelement <4 x float> %316, i64 2, !dbg !15
  %322 = extractelement <4 x float> %316, i64 3, !dbg !15
  %323 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %313, <4 x float> %317) #9, !dbg !15
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1748, align 8, !dbg !15, !frisk.k_tail_pair !31
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !15, !frisk.k_tail_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %308, <4 x float> %274) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %327 = extractelement <4 x float> %323, i64 0, !dbg !15
  %328 = call float @llvm.maxnum.f32(float %319, float %327), !dbg !8
  %329 = extractelement <4 x float> %323, i64 1, !dbg !15
  %330 = call float @llvm.maxnum.f32(float %320, float %329), !dbg !8
  %331 = extractelement <4 x float> %323, i64 2, !dbg !15
  %332 = call float @llvm.maxnum.f32(float %321, float %331), !dbg !8
  %333 = extractelement <4 x float> %323, i64 3, !dbg !15
  %334 = call float @llvm.maxnum.f32(float %322, float %333), !dbg !8
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1749, align 8, !dbg !15, !frisk.k_tail_pair !37
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !15, !frisk.k_tail_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %326) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1750, align 8, !dbg !15, !frisk.k_tail_pair !38
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !15, !frisk.k_tail_pair !38
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %312, <4 x float> %276) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %341 = extractelement <4 x float> %337, i64 0, !dbg !15
  %342 = call float @llvm.maxnum.f32(float %328, float %341), !dbg !8
  %343 = extractelement <4 x float> %337, i64 1, !dbg !15
  %344 = call float @llvm.maxnum.f32(float %330, float %343), !dbg !8
  %345 = extractelement <4 x float> %337, i64 2, !dbg !15
  %346 = call float @llvm.maxnum.f32(float %332, float %345), !dbg !8
  %347 = extractelement <4 x float> %337, i64 3, !dbg !15
  %348 = call float @llvm.maxnum.f32(float %334, float %347), !dbg !8
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !15, !frisk.k_tail_pair !39
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !15, !frisk.k_tail_pair !39
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %314, <4 x float> %340) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %324, <4 x float> %278) #9, !dbg !15
  %353 = extractelement <4 x float> %351, i64 0, !dbg !15
  %354 = call float @llvm.maxnum.f32(float %342, float %353), !dbg !8
  %355 = extractelement <4 x float> %351, i64 1, !dbg !15
  %356 = call float @llvm.maxnum.f32(float %344, float %355), !dbg !8
  %357 = extractelement <4 x float> %351, i64 2, !dbg !15
  %358 = call float @llvm.maxnum.f32(float %346, float %357), !dbg !8
  %359 = extractelement <4 x float> %351, i64 3, !dbg !15
  %360 = call float @llvm.maxnum.f32(float %348, float %359), !dbg !8
  %361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %335, <4 x float> %352) #9, !dbg !15
  %362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %338, <4 x float> %280) #9, !dbg !15
  %363 = extractelement <4 x float> %361, i64 0, !dbg !15
  %364 = call float @llvm.maxnum.f32(float %354, float %363), !dbg !8
  %365 = extractelement <4 x float> %361, i64 1, !dbg !15
  %366 = call float @llvm.maxnum.f32(float %356, float %365), !dbg !8
  %367 = extractelement <4 x float> %361, i64 2, !dbg !15
  %368 = call float @llvm.maxnum.f32(float %358, float %367), !dbg !8
  %369 = extractelement <4 x float> %361, i64 3, !dbg !15
  %370 = call float @llvm.maxnum.f32(float %360, float %369), !dbg !8
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %349, <4 x float> %362) #9, !dbg !15
  %372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %325, <4 x float> %282) #9, !dbg !15
  %373 = extractelement <4 x float> %371, i64 0, !dbg !15
  %374 = call float @llvm.maxnum.f32(float %364, float %373), !dbg !8
  %375 = extractelement <4 x float> %371, i64 1, !dbg !15
  %376 = call float @llvm.maxnum.f32(float %366, float %375), !dbg !8
  %377 = extractelement <4 x float> %371, i64 2, !dbg !15
  %378 = call float @llvm.maxnum.f32(float %368, float %377), !dbg !8
  %379 = extractelement <4 x float> %371, i64 3, !dbg !15
  %380 = call float @llvm.maxnum.f32(float %370, float %379), !dbg !8
  %381 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %336, <4 x float> %372) #9, !dbg !15
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %339, <4 x float> %284) #9, !dbg !15
  %383 = extractelement <4 x float> %381, i64 0, !dbg !15
  %384 = call float @llvm.maxnum.f32(float %374, float %383), !dbg !8
  %385 = extractelement <4 x float> %381, i64 1, !dbg !15
  %386 = call float @llvm.maxnum.f32(float %376, float %385), !dbg !8
  %387 = extractelement <4 x float> %381, i64 2, !dbg !15
  %388 = call float @llvm.maxnum.f32(float %378, float %387), !dbg !8
  %389 = extractelement <4 x float> %381, i64 3, !dbg !15
  %390 = call float @llvm.maxnum.f32(float %380, float %389), !dbg !8
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %350, <4 x float> %382) #9, !dbg !15
  %392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %307, <4 x float> %286) #9, !dbg !15
  %393 = extractelement <4 x float> %391, i64 0, !dbg !15
  %394 = call float @llvm.maxnum.f32(float %384, float %393), !dbg !8
  %395 = extractelement <4 x float> %391, i64 1, !dbg !15
  %396 = call float @llvm.maxnum.f32(float %386, float %395), !dbg !8
  %397 = extractelement <4 x float> %391, i64 2, !dbg !15
  %398 = call float @llvm.maxnum.f32(float %388, float %397), !dbg !8
  %399 = extractelement <4 x float> %391, i64 3, !dbg !15
  %400 = call float @llvm.maxnum.f32(float %390, float %399), !dbg !8
  %401 = call float @llvm.maxnum.f32(float %394, float %396), !dbg !8
  %402 = call float @llvm.maxnum.f32(float %398, float %400), !dbg !8
  %403 = call float @llvm.maxnum.f32(float %401, float %402), !dbg !8
  %404 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %309, <4 x float> %392) #9, !dbg !15
  %405 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %311, <4 x float> %288) #9, !dbg !15
  %406 = extractelement <4 x float> %404, i64 0, !dbg !15
  %407 = extractelement <4 x float> %404, i64 1, !dbg !15
  %408 = extractelement <4 x float> %404, i64 2, !dbg !15
  %409 = extractelement <4 x float> %404, i64 3, !dbg !15
  %410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %313, <4 x float> %405) #9, !dbg !15
  %411 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %308, <4 x float> %290) #9, !dbg !15
  %412 = extractelement <4 x float> %410, i64 0, !dbg !15
  %413 = extractelement <4 x float> %410, i64 1, !dbg !15
  %414 = extractelement <4 x float> %410, i64 2, !dbg !15
  %415 = extractelement <4 x float> %410, i64 3, !dbg !15
  %416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %310, <4 x float> %411) #9, !dbg !15
  %417 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %312, <4 x float> %292) #9, !dbg !15
  %418 = extractelement <4 x float> %416, i64 0, !dbg !15
  %419 = extractelement <4 x float> %416, i64 1, !dbg !15
  %420 = extractelement <4 x float> %416, i64 2, !dbg !15
  %421 = extractelement <4 x float> %416, i64 3, !dbg !15
  %422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %314, <4 x float> %417) #9, !dbg !15
  %423 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %324, <4 x float> %294) #9, !dbg !15
  %424 = extractelement <4 x float> %422, i64 0, !dbg !15
  %425 = extractelement <4 x float> %422, i64 1, !dbg !15
  %426 = extractelement <4 x float> %422, i64 2, !dbg !15
  %427 = extractelement <4 x float> %422, i64 3, !dbg !15
  %428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %335, <4 x float> %423) #9, !dbg !15
  %429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %338, <4 x float> %296) #9, !dbg !15
  %430 = extractelement <4 x float> %428, i64 0, !dbg !15
  %431 = extractelement <4 x float> %428, i64 1, !dbg !15
  %432 = extractelement <4 x float> %428, i64 2, !dbg !15
  %433 = extractelement <4 x float> %428, i64 3, !dbg !15
  %434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %349, <4 x float> %429) #9, !dbg !15
  %435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %325, <4 x float> %298) #9, !dbg !15
  %436 = extractelement <4 x float> %434, i64 0, !dbg !15
  %437 = extractelement <4 x float> %434, i64 1, !dbg !15
  %438 = extractelement <4 x float> %434, i64 2, !dbg !15
  %439 = extractelement <4 x float> %434, i64 3, !dbg !15
  %440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %336, <4 x float> %435) #9, !dbg !15
  %441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %92, <4 x half> %339, <4 x float> %300) #9, !dbg !15
  %442 = extractelement <4 x float> %440, i64 0, !dbg !15
  %443 = extractelement <4 x float> %440, i64 1, !dbg !15
  %444 = extractelement <4 x float> %440, i64 2, !dbg !15
  %445 = extractelement <4 x float> %440, i64 3, !dbg !15
  %446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %350, <4 x float> %441) #9, !dbg !15
  %447 = extractelement <4 x float> %446, i64 0, !dbg !15
  %448 = extractelement <4 x float> %446, i64 1, !dbg !15
  %449 = extractelement <4 x float> %446, i64 2, !dbg !15
  %450 = extractelement <4 x float> %446, i64 3, !dbg !15
  %451 = bitcast float %403 to i32, !dbg !8
  %452 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %451), !dbg !8
  %453 = bitcast i32 %452 to float, !dbg !8
  %454 = call float @llvm.maxnum.f32(float %403, float %453), !dbg !8
  %455 = bitcast float %454 to i32, !dbg !8
  %456 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %455), !dbg !8
  %457 = bitcast i32 %456 to float, !dbg !8
  %458 = call float @llvm.maxnum.f32(float %454, float %457), !dbg !8
  %459 = call float @llvm.maxnum.f32(float %406, float 0xFFF0000000000000), !dbg !8
  %460 = call float @llvm.maxnum.f32(float %459, float %412), !dbg !8
  %461 = call float @llvm.maxnum.f32(float %407, float %413), !dbg !8
  %462 = call float @llvm.maxnum.f32(float %408, float %414), !dbg !8
  %463 = call float @llvm.maxnum.f32(float %409, float %415), !dbg !8
  %464 = call float @llvm.maxnum.f32(float %460, float %418), !dbg !8
  %465 = call float @llvm.maxnum.f32(float %461, float %419), !dbg !8
  %466 = call float @llvm.maxnum.f32(float %462, float %420), !dbg !8
  %467 = call float @llvm.maxnum.f32(float %463, float %421), !dbg !8
  %468 = call float @llvm.maxnum.f32(float %464, float %424), !dbg !8
  %469 = call float @llvm.maxnum.f32(float %465, float %425), !dbg !8
  %470 = call float @llvm.maxnum.f32(float %466, float %426), !dbg !8
  %471 = call float @llvm.maxnum.f32(float %467, float %427), !dbg !8
  %472 = call float @llvm.maxnum.f32(float %468, float %430), !dbg !8
  %473 = call float @llvm.maxnum.f32(float %469, float %431), !dbg !8
  %474 = call float @llvm.maxnum.f32(float %470, float %432), !dbg !8
  %475 = call float @llvm.maxnum.f32(float %471, float %433), !dbg !8
  %476 = call float @llvm.maxnum.f32(float %472, float %436), !dbg !8
  %477 = call float @llvm.maxnum.f32(float %473, float %437), !dbg !8
  %478 = call float @llvm.maxnum.f32(float %474, float %438), !dbg !8
  %479 = call float @llvm.maxnum.f32(float %475, float %439), !dbg !8
  %480 = call float @llvm.maxnum.f32(float %476, float %442), !dbg !8
  %481 = call float @llvm.maxnum.f32(float %477, float %443), !dbg !8
  %482 = call float @llvm.maxnum.f32(float %478, float %444), !dbg !8
  %483 = call float @llvm.maxnum.f32(float %479, float %445), !dbg !8
  %484 = call float @llvm.maxnum.f32(float %480, float %447), !dbg !8
  %485 = call float @llvm.maxnum.f32(float %481, float %448), !dbg !8
  %486 = call float @llvm.maxnum.f32(float %482, float %449), !dbg !8
  %487 = call float @llvm.maxnum.f32(float %483, float %450), !dbg !8
  %488 = call float @llvm.maxnum.f32(float %484, float %485), !dbg !8
  %489 = call float @llvm.maxnum.f32(float %486, float %487), !dbg !8
  %490 = call float @llvm.maxnum.f32(float %488, float %489), !dbg !8
  %491 = bitcast float %490 to i32, !dbg !8
  %492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %491), !dbg !8
  %493 = bitcast i32 %492 to float, !dbg !8
  %494 = call float @llvm.maxnum.f32(float %490, float %493), !dbg !8
  %495 = bitcast float %494 to i32, !dbg !8
  %496 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %495), !dbg !8
  %497 = bitcast i32 %496 to float, !dbg !8
  %498 = call float @llvm.maxnum.f32(float %494, float %497), !dbg !8
  %lazy.peak = call float @llvm.maxnum.f32(float %458, float %498), !dbg !23
  %lazy.over = fcmp ogt float %lazy.peak, 0x4020000000000000, !dbg !23
  %lazy.mask = call i64 @llvm.amdgcn.ballot.i64(i1 %lazy.over), !dbg !23
  %lazy.any = icmp ne i64 %lazy.mask, 0, !dbg !23
  %lazy.first = icmp eq i64 %101, 0, !dbg !23
  %lazy.take = or i1 %lazy.first, %lazy.any, !dbg !23
  br i1 %lazy.take, label %lazy.slow, label %lazy.join, !dbg !23

lazy.slow:                                        ; preds = %109
  %lazy.positive = call float @llvm.maxnum.f32(float %458, float 0x0000000000000000), !dbg !23
  %lazy.delta = select i1 %lazy.first, float %458, float %lazy.positive, !dbg !23
  %lazy.negated = fsub float 0x0000000000000000, %lazy.delta, !dbg !23
  %lazy.exp = call float @llvm.amdgcn.exp2.f32(float %lazy.negated), !dbg !23
  %lazy.alpha = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp, !dbg !23
  %lazy.positive1923 = call float @llvm.maxnum.f32(float %498, float 0x0000000000000000), !dbg !23
  %lazy.delta1924 = select i1 %lazy.first, float %498, float %lazy.positive1923, !dbg !23
  %lazy.negated1925 = fsub float 0x0000000000000000, %lazy.delta1924, !dbg !23
  %lazy.exp1926 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated1925), !dbg !23
  %lazy.alpha1927 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp1926, !dbg !23
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted = fsub <4 x float> %316, %lazy.delta.v, !dbg !23
  %lazy.delta.v.lane1928 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1929 = shufflevector <1 x float> %lazy.delta.v.lane1928, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1930 = fsub <4 x float> %323, %lazy.delta.v1929, !dbg !23
  %lazy.delta.v.lane1931 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1932 = shufflevector <1 x float> %lazy.delta.v.lane1931, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1933 = fsub <4 x float> %337, %lazy.delta.v1932, !dbg !23
  %lazy.delta.v.lane1934 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1935 = shufflevector <1 x float> %lazy.delta.v.lane1934, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1936 = fsub <4 x float> %351, %lazy.delta.v1935, !dbg !23
  %lazy.delta.v.lane1937 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1938 = shufflevector <1 x float> %lazy.delta.v.lane1937, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1939 = fsub <4 x float> %361, %lazy.delta.v1938, !dbg !23
  %lazy.delta.v.lane1940 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1941 = shufflevector <1 x float> %lazy.delta.v.lane1940, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1942 = fsub <4 x float> %371, %lazy.delta.v1941, !dbg !23
  %lazy.delta.v.lane1943 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1944 = shufflevector <1 x float> %lazy.delta.v.lane1943, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1945 = fsub <4 x float> %381, %lazy.delta.v1944, !dbg !23
  %lazy.delta.v.lane1946 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1947 = shufflevector <1 x float> %lazy.delta.v.lane1946, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1948 = fsub <4 x float> %391, %lazy.delta.v1947, !dbg !23
  %lazy.delta.v.lane1949 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1950 = shufflevector <1 x float> %lazy.delta.v.lane1949, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1951 = fsub <4 x float> %404, %lazy.delta.v1950, !dbg !23
  %lazy.delta.v.lane1952 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1953 = shufflevector <1 x float> %lazy.delta.v.lane1952, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1954 = fsub <4 x float> %410, %lazy.delta.v1953, !dbg !23
  %lazy.delta.v.lane1955 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1956 = shufflevector <1 x float> %lazy.delta.v.lane1955, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1957 = fsub <4 x float> %416, %lazy.delta.v1956, !dbg !23
  %lazy.delta.v.lane1958 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1959 = shufflevector <1 x float> %lazy.delta.v.lane1958, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1960 = fsub <4 x float> %422, %lazy.delta.v1959, !dbg !23
  %lazy.delta.v.lane1961 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1962 = shufflevector <1 x float> %lazy.delta.v.lane1961, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1963 = fsub <4 x float> %428, %lazy.delta.v1962, !dbg !23
  %lazy.delta.v.lane1964 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1965 = shufflevector <1 x float> %lazy.delta.v.lane1964, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1966 = fsub <4 x float> %434, %lazy.delta.v1965, !dbg !23
  %lazy.delta.v.lane1967 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1968 = shufflevector <1 x float> %lazy.delta.v.lane1967, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1969 = fsub <4 x float> %440, %lazy.delta.v1968, !dbg !23
  %lazy.delta.v.lane1970 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.delta.v1971 = shufflevector <1 x float> %lazy.delta.v.lane1970, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1972 = fsub <4 x float> %446, %lazy.delta.v1971, !dbg !23
  %lazy.state = extractvalue [2 x <1 x float>] %103, 0, !dbg !23
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !23
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !23
  %lazy.state1973 = extractvalue [2 x <1 x float>] %103, 1, !dbg !23
  %lazy.alpha.v.lane1974 = insertelement <1 x float> poison, float %lazy.alpha1927, i64 0, !dbg !23
  %lazy.scaled1975 = fmul <1 x float> %lazy.state1973, %lazy.alpha.v.lane1974, !dbg !23
  %lazy.state.next1976 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1975, 1, !dbg !23
  %lazy.state1977 = extractvalue [2 x <8 x float>] %107, 0, !dbg !23
  %lazy.alpha.v.lane1978 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane1978, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1979 = fmul <8 x float> %lazy.state1977, %lazy.alpha.v, !dbg !23
  %lazy.state.next1980 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1979, 0, !dbg !23
  %lazy.state1981 = extractvalue [2 x <8 x float>] %107, 1, !dbg !23
  %lazy.alpha.v.lane1982 = insertelement <1 x float> poison, float %lazy.alpha1927, i64 0, !dbg !23
  %lazy.alpha.v1983 = shufflevector <1 x float> %lazy.alpha.v.lane1982, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1984 = fmul <8 x float> %lazy.state1981, %lazy.alpha.v1983, !dbg !23
  %lazy.state.next1985 = insertvalue [2 x <8 x float>] %lazy.state.next1980, <8 x float> %lazy.scaled1984, 1, !dbg !23
  %lazy.state1986 = extractvalue [2 x <8 x float>] %106, 0, !dbg !23
  %lazy.alpha.v.lane1987 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v1988 = shufflevector <1 x float> %lazy.alpha.v.lane1987, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1989 = fmul <8 x float> %lazy.state1986, %lazy.alpha.v1988, !dbg !23
  %lazy.state.next1990 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1989, 0, !dbg !23
  %lazy.state1991 = extractvalue [2 x <8 x float>] %106, 1, !dbg !23
  %lazy.alpha.v.lane1992 = insertelement <1 x float> poison, float %lazy.alpha1927, i64 0, !dbg !23
  %lazy.alpha.v1993 = shufflevector <1 x float> %lazy.alpha.v.lane1992, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1994 = fmul <8 x float> %lazy.state1991, %lazy.alpha.v1993, !dbg !23
  %lazy.state.next1995 = insertvalue [2 x <8 x float>] %lazy.state.next1990, <8 x float> %lazy.scaled1994, 1, !dbg !23
  %lazy.state1996 = extractvalue [2 x <8 x float>] %105, 0, !dbg !23
  %lazy.alpha.v.lane1997 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v1998 = shufflevector <1 x float> %lazy.alpha.v.lane1997, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1999 = fmul <8 x float> %lazy.state1996, %lazy.alpha.v1998, !dbg !23
  %lazy.state.next2000 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1999, 0, !dbg !23
  %lazy.state2001 = extractvalue [2 x <8 x float>] %105, 1, !dbg !23
  %lazy.alpha.v.lane2002 = insertelement <1 x float> poison, float %lazy.alpha1927, i64 0, !dbg !23
  %lazy.alpha.v2003 = shufflevector <1 x float> %lazy.alpha.v.lane2002, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2004 = fmul <8 x float> %lazy.state2001, %lazy.alpha.v2003, !dbg !23
  %lazy.state.next2005 = insertvalue [2 x <8 x float>] %lazy.state.next2000, <8 x float> %lazy.scaled2004, 1, !dbg !23
  %lazy.state2006 = extractvalue [2 x <8 x float>] %102, 0, !dbg !23
  %lazy.alpha.v.lane2007 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2008 = shufflevector <1 x float> %lazy.alpha.v.lane2007, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2009 = fmul <8 x float> %lazy.state2006, %lazy.alpha.v2008, !dbg !23
  %lazy.state.next2010 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2009, 0, !dbg !23
  %lazy.state2011 = extractvalue [2 x <8 x float>] %102, 1, !dbg !23
  %lazy.alpha.v.lane2012 = insertelement <1 x float> poison, float %lazy.alpha1927, i64 0, !dbg !23
  %lazy.alpha.v2013 = shufflevector <1 x float> %lazy.alpha.v.lane2012, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2014 = fmul <8 x float> %lazy.state2011, %lazy.alpha.v2013, !dbg !23
  %lazy.state.next2015 = insertvalue [2 x <8 x float>] %lazy.state.next2010, <8 x float> %lazy.scaled2014, 1, !dbg !23
  %lazy.base.old = extractvalue [2 x <1 x float>] %104, 0, !dbg !23
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !23
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !23
  %lazy.base.old2016 = extractvalue [2 x <1 x float>] %104, 1, !dbg !23
  %lazy.delta.b.lane2017 = insertelement <1 x float> poison, float %lazy.delta1924, i64 0, !dbg !23
  %lazy.base.next2018 = fsub <1 x float> %lazy.base.old2016, %lazy.delta.b.lane2017, !dbg !23
  %lazy.base.state2019 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2018, 1, !dbg !23
  br label %lazy.join, !dbg !23

lazy.join:                                        ; preds = %lazy.slow, %109
  %lazy.score = phi <4 x float> [ %316, %109 ], [ %lazy.shifted, %lazy.slow ], !dbg !23
  %lazy.score2020 = phi <4 x float> [ %323, %109 ], [ %lazy.shifted1930, %lazy.slow ], !dbg !23
  %lazy.score2021 = phi <4 x float> [ %337, %109 ], [ %lazy.shifted1933, %lazy.slow ], !dbg !23
  %lazy.score2022 = phi <4 x float> [ %351, %109 ], [ %lazy.shifted1936, %lazy.slow ], !dbg !23
  %lazy.score2023 = phi <4 x float> [ %361, %109 ], [ %lazy.shifted1939, %lazy.slow ], !dbg !23
  %lazy.score2024 = phi <4 x float> [ %371, %109 ], [ %lazy.shifted1942, %lazy.slow ], !dbg !23
  %lazy.score2025 = phi <4 x float> [ %381, %109 ], [ %lazy.shifted1945, %lazy.slow ], !dbg !23
  %lazy.score2026 = phi <4 x float> [ %391, %109 ], [ %lazy.shifted1948, %lazy.slow ], !dbg !23
  %lazy.score2027 = phi <4 x float> [ %404, %109 ], [ %lazy.shifted1951, %lazy.slow ], !dbg !23
  %lazy.score2028 = phi <4 x float> [ %410, %109 ], [ %lazy.shifted1954, %lazy.slow ], !dbg !23
  %lazy.score2029 = phi <4 x float> [ %416, %109 ], [ %lazy.shifted1957, %lazy.slow ], !dbg !23
  %lazy.score2030 = phi <4 x float> [ %422, %109 ], [ %lazy.shifted1960, %lazy.slow ], !dbg !23
  %lazy.score2031 = phi <4 x float> [ %428, %109 ], [ %lazy.shifted1963, %lazy.slow ], !dbg !23
  %lazy.score2032 = phi <4 x float> [ %434, %109 ], [ %lazy.shifted1966, %lazy.slow ], !dbg !23
  %lazy.score2033 = phi <4 x float> [ %440, %109 ], [ %lazy.shifted1969, %lazy.slow ], !dbg !23
  %lazy.score2034 = phi <4 x float> [ %446, %109 ], [ %lazy.shifted1972, %lazy.slow ], !dbg !23
  %lazy.state2035 = phi [2 x <1 x float>] [ %103, %109 ], [ %lazy.state.next1976, %lazy.slow ], !dbg !23
  %lazy.state2036 = phi [2 x <8 x float>] [ %107, %109 ], [ %lazy.state.next1985, %lazy.slow ], !dbg !23
  %lazy.state2037 = phi [2 x <8 x float>] [ %106, %109 ], [ %lazy.state.next1995, %lazy.slow ], !dbg !23
  %lazy.state2038 = phi [2 x <8 x float>] [ %105, %109 ], [ %lazy.state.next2005, %lazy.slow ], !dbg !23
  %lazy.state2039 = phi [2 x <8 x float>] [ %102, %109 ], [ %lazy.state.next2015, %lazy.slow ], !dbg !23
  %lazy.base.join = phi [2 x <1 x float>] [ %104, %109 ], [ %lazy.base.state2019, %lazy.slow ], !dbg !23
  %499 = extractvalue [2 x <1 x float>] %lazy.state2035, 0, !dbg !23
  %500 = extractelement <1 x float> %499, i64 0, !dbg !23
  %501 = extractvalue [2 x <1 x float>] %lazy.state2035, 1, !dbg !23
  %502 = extractelement <1 x float> %501, i64 0, !dbg !23
  %503 = extractvalue [2 x <8 x float>] %lazy.state2039, 0, !dbg !27
  %504 = shufflevector <8 x float> %503, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %505 = shufflevector <8 x float> %503, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %506 = extractvalue [2 x <8 x float>] %lazy.state2039, 1, !dbg !27
  %507 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %508 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = add nsw i64 %117, %address.local782, !dbg !16
  %vdb.address1286 = trunc i64 %vdb.address to i32, !dbg !16
  %vdb.address1287 = shl i32 %vdb.address1286, 1, !dbg !16
  %vdb.address1292 = trunc nuw i64 %117 to i32, !dbg !16
  %vdb.address1293 = or disjoint i32 %address.local789, %vdb.address1292, !dbg !16
  %vdb.address1294 = shl nuw nsw i32 %vdb.address1293, 1, !dbg !16
  %vdb.dma.base = add i32 %38, 8192, !dbg !14
  %509 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1287, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %vdb.dma.base1299 = add i32 %46, 8192, !dbg !14
  %510 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1299, i32 %vdb.address1294, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %address.local793 = and i32 %address.local784, 120, !dbg !14
  %address.local795 = and i32 %address.local435, 1536, !dbg !14
  %address.local796 = or disjoint i32 %address.local793, %address.local795, !dbg !14
  %address.local797 = zext nneg i32 %address.local796 to i64, !dbg !14
  %address.local798 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local797, !dbg !14
  %511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local798, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %512 = bitcast <4 x i32> %511 to <8 x half>, !dbg !14
  %address.local799 = and i64 %address.local362, 60, !dbg !14
  %address.local800 = add nuw nsw i64 %address.local799, %address.local362, !dbg !14
  %address.local802.idx = shl nuw nsw i64 %address.local800, 4, !dbg !14
  %address.local802 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local802.idx, !dbg !14
  %address.local803 = lshr i64 %address.local415, 4, !dbg !14
  %address.local805.idx = mul nuw nsw i64 %address.local803, 768, !dbg !14
  %address.local805 = getelementptr i8, ptr addrspace(3) %address.local802, i64 %address.local805.idx, !dbg !14
  %address.local807.idx = mul nsw i64 %address.local368, -64, !dbg !14
  %address.local807 = getelementptr i8, ptr addrspace(3) %address.local805, i64 %address.local807.idx, !dbg !14
  %address.local808 = getelementptr i8, ptr addrspace(3) %address.local807, i64 256, !dbg !14
  %513 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local808, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %514 = bitcast <4 x i32> %513 to <8 x half>, !dbg !14
  %address.local812 = getelementptr i8, ptr addrspace(3) %address.local807, i64 512, !dbg !14
  %515 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %516 = bitcast <4 x i32> %515 to <8 x half>, !dbg !14
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local807, i64 768, !dbg !14
  %517 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %518 = bitcast <4 x i32> %517 to <8 x half>, !dbg !14
  %address.local818 = getelementptr i8, ptr addrspace(3) %address.local798, i64 4096, !dbg !14
  %519 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local818, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %520 = bitcast <4 x i32> %519 to <8 x half>, !dbg !14
  %address.local822 = getelementptr i8, ptr addrspace(3) %address.local807, i64 4352, !dbg !14
  %521 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %522 = bitcast <4 x i32> %521 to <8 x half>, !dbg !14
  %address.local826 = getelementptr i8, ptr addrspace(3) %address.local807, i64 4608, !dbg !14
  %523 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %524 = bitcast <4 x i32> %523 to <8 x half>, !dbg !14
  %address.local830 = getelementptr i8, ptr addrspace(3) %address.local807, i64 4864, !dbg !14
  %525 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %526 = shufflevector <8 x half> %512, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %527 = extractelement <4 x float> %lazy.score, i64 0, !dbg !24
  %528 = call float @llvm.amdgcn.exp2.f32(float %527), !dbg !24
  %529 = insertelement <4 x float> poison, float %528, i64 0, !dbg !26
  %530 = extractelement <4 x float> %lazy.score, i64 1, !dbg !24
  %531 = call float @llvm.amdgcn.exp2.f32(float %530), !dbg !24
  %532 = insertelement <4 x float> %529, float %531, i64 1, !dbg !26
  %533 = extractelement <4 x float> %lazy.score, i64 2, !dbg !24
  %534 = call float @llvm.amdgcn.exp2.f32(float %533), !dbg !24
  %535 = insertelement <4 x float> %532, float %534, i64 2, !dbg !26
  %536 = extractelement <4 x float> %lazy.score, i64 3, !dbg !24
  %537 = call float @llvm.amdgcn.exp2.f32(float %536), !dbg !24
  %538 = insertelement <4 x float> %535, float %537, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %538) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !26
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !26
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !26
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !26
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !26
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !26
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !26
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !26
  %539 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !26
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %539) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %526, <4 x float> %504) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %541 = shufflevector <8 x half> %512, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %541, <4 x float> %505) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %543 = extractelement <4 x float> %lazy.score2027, i64 0, !dbg !24
  %544 = call float @llvm.amdgcn.exp2.f32(float %543), !dbg !24
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !26
  %546 = extractelement <4 x float> %lazy.score2027, i64 1, !dbg !24
  %547 = call float @llvm.amdgcn.exp2.f32(float %546), !dbg !24
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !26
  %549 = extractelement <4 x float> %lazy.score2027, i64 2, !dbg !24
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !24
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !26
  %552 = extractelement <4 x float> %lazy.score2027, i64 3, !dbg !24
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !24
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !26
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02040 = extractelement <4 x float> %math.result248, i32 0, !dbg !26
  %.cvtpack.f12041 = extractelement <4 x float> %math.result248, i32 1, !dbg !26
  %.cvtpack.f22042 = extractelement <4 x float> %math.result248, i32 2, !dbg !26
  %.cvtpack.f32043 = extractelement <4 x float> %math.result248, i32 3, !dbg !26
  %.cvtpack.p02044 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02040, float %.cvtpack.f12041), !dbg !26
  %.cvtpack.p12045 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22042, float %.cvtpack.f32043), !dbg !26
  %.cvtpack.v02046 = insertelement <2 x i32> poison, i32 %.cvtpack.p02044, i32 0, !dbg !26
  %.cvtpack.v12047 = insertelement <2 x i32> %.cvtpack.v02046, i32 %.cvtpack.p12045, i32 1, !dbg !26
  %555 = bitcast <2 x i32> %.cvtpack.v12047 to <4 x half>, !dbg !26
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %526, <4 x float> %507) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %541, <4 x float> %508) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %558 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %559 = extractelement <4 x float> %lazy.score2020, i64 0, !dbg !24
  %560 = call float @llvm.amdgcn.exp2.f32(float %559), !dbg !24
  %561 = fadd float %528, %560, !dbg !25
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %561) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %562 = insertelement <4 x float> poison, float %560, i64 0, !dbg !26
  %563 = extractelement <4 x float> %lazy.score2020, i64 1, !dbg !24
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !24
  %565 = fadd float %531, %564, !dbg !25
  %k0.sum.anchor1843 = call float asm sideeffect "", "=v,0"(float %565) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %566 = insertelement <4 x float> %562, float %564, i64 1, !dbg !26
  %567 = extractelement <4 x float> %lazy.score2020, i64 2, !dbg !24
  %568 = call float @llvm.amdgcn.exp2.f32(float %567), !dbg !24
  %569 = fadd float %534, %568, !dbg !25
  %k0.sum.anchor1844 = call float asm sideeffect "", "=v,0"(float %569) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %570 = insertelement <4 x float> %566, float %568, i64 2, !dbg !26
  %571 = extractelement <4 x float> %lazy.score2020, i64 3, !dbg !24
  %572 = call float @llvm.amdgcn.exp2.f32(float %571), !dbg !24
  %573 = fadd float %537, %572, !dbg !25
  %k0.sum.anchor1845 = call float asm sideeffect "", "=v,0"(float %573) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %570, float %572, i64 3, !dbg !26
  %math.result234 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02048 = extractelement <4 x float> %math.result234, i32 0, !dbg !26
  %.cvtpack.f12049 = extractelement <4 x float> %math.result234, i32 1, !dbg !26
  %.cvtpack.f22050 = extractelement <4 x float> %math.result234, i32 2, !dbg !26
  %.cvtpack.f32051 = extractelement <4 x float> %math.result234, i32 3, !dbg !26
  %.cvtpack.p02052 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02048, float %.cvtpack.f12049), !dbg !26
  %.cvtpack.p12053 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22050, float %.cvtpack.f32051), !dbg !26
  %.cvtpack.v02054 = insertelement <2 x i32> poison, i32 %.cvtpack.p02052, i32 0, !dbg !26
  %.cvtpack.v12055 = insertelement <2 x i32> %.cvtpack.v02054, i32 %.cvtpack.p12053, i32 1, !dbg !26
  %575 = bitcast <2 x i32> %.cvtpack.v12055 to <4 x half>, !dbg !26
  %math.packed235 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %558, <4 x float> %540) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %577 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %577, <4 x float> %542) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = extractelement <4 x float> %lazy.score2028, i64 0, !dbg !24
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !24
  %581 = fadd float %544, %580, !dbg !25
  %k0.sum.anchor1846 = call float asm sideeffect "", "=v,0"(float %581) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %582 = insertelement <4 x float> poison, float %580, i64 0, !dbg !26
  %583 = extractelement <4 x float> %lazy.score2028, i64 1, !dbg !24
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !24
  %585 = fadd float %547, %584, !dbg !25
  %k0.sum.anchor1847 = call float asm sideeffect "", "=v,0"(float %585) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %586 = insertelement <4 x float> %582, float %584, i64 1, !dbg !26
  %587 = extractelement <4 x float> %lazy.score2028, i64 2, !dbg !24
  %588 = call float @llvm.amdgcn.exp2.f32(float %587), !dbg !24
  %589 = fadd float %550, %588, !dbg !25
  %k0.sum.anchor1848 = call float asm sideeffect "", "=v,0"(float %589) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %590 = insertelement <4 x float> %586, float %588, i64 2, !dbg !26
  %591 = extractelement <4 x float> %lazy.score2028, i64 3, !dbg !24
  %592 = call float @llvm.amdgcn.exp2.f32(float %591), !dbg !24
  %593 = fadd float %553, %592, !dbg !25
  %k0.sum.anchor1849 = call float asm sideeffect "", "=v,0"(float %593) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %594 = insertelement <4 x float> %590, float %592, i64 3, !dbg !26
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %594) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02056 = extractelement <4 x float> %math.result250, i32 0, !dbg !26
  %.cvtpack.f12057 = extractelement <4 x float> %math.result250, i32 1, !dbg !26
  %.cvtpack.f22058 = extractelement <4 x float> %math.result250, i32 2, !dbg !26
  %.cvtpack.f32059 = extractelement <4 x float> %math.result250, i32 3, !dbg !26
  %.cvtpack.p02060 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02056, float %.cvtpack.f12057), !dbg !26
  %.cvtpack.p12061 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22058, float %.cvtpack.f32059), !dbg !26
  %.cvtpack.v02062 = insertelement <2 x i32> poison, i32 %.cvtpack.p02060, i32 0, !dbg !26
  %.cvtpack.v12063 = insertelement <2 x i32> %.cvtpack.v02062, i32 %.cvtpack.p12061, i32 1, !dbg !26
  %595 = bitcast <2 x i32> %.cvtpack.v12063 to <4 x half>, !dbg !26
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %595) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %558, <4 x float> %556) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %577, <4 x float> %557) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %598 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %599 = extractelement <4 x float> %lazy.score2021, i64 0, !dbg !24
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !24
  %601 = fadd float %k0.sum.anchor, %600, !dbg !25
  %k0.sum.anchor1850 = call float asm sideeffect "", "=v,0"(float %601) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> poison, float %600, i64 0, !dbg !26
  %603 = extractelement <4 x float> %lazy.score2021, i64 1, !dbg !24
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !24
  %605 = fadd float %k0.sum.anchor1843, %604, !dbg !25
  %k0.sum.anchor1851 = call float asm sideeffect "", "=v,0"(float %605) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %606 = insertelement <4 x float> %602, float %604, i64 1, !dbg !26
  %607 = extractelement <4 x float> %lazy.score2021, i64 2, !dbg !24
  %608 = call float @llvm.amdgcn.exp2.f32(float %607), !dbg !24
  %609 = fadd float %k0.sum.anchor1844, %608, !dbg !25
  %k0.sum.anchor1852 = call float asm sideeffect "", "=v,0"(float %609) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %610 = insertelement <4 x float> %606, float %608, i64 2, !dbg !26
  %611 = extractelement <4 x float> %lazy.score2021, i64 3, !dbg !24
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !24
  %613 = fadd float %k0.sum.anchor1845, %612, !dbg !25
  %k0.sum.anchor1853 = call float asm sideeffect "", "=v,0"(float %613) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %614 = insertelement <4 x float> %610, float %612, i64 3, !dbg !26
  %math.result236 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %614) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02064 = extractelement <4 x float> %math.result236, i32 0, !dbg !26
  %.cvtpack.f12065 = extractelement <4 x float> %math.result236, i32 1, !dbg !26
  %.cvtpack.f22066 = extractelement <4 x float> %math.result236, i32 2, !dbg !26
  %.cvtpack.f32067 = extractelement <4 x float> %math.result236, i32 3, !dbg !26
  %.cvtpack.p02068 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02064, float %.cvtpack.f12065), !dbg !26
  %.cvtpack.p12069 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22066, float %.cvtpack.f32067), !dbg !26
  %.cvtpack.v02070 = insertelement <2 x i32> poison, i32 %.cvtpack.p02068, i32 0, !dbg !26
  %.cvtpack.v12071 = insertelement <2 x i32> %.cvtpack.v02070, i32 %.cvtpack.p12069, i32 1, !dbg !26
  %615 = bitcast <2 x i32> %.cvtpack.v12071 to <4 x half>, !dbg !26
  %math.packed237 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %615) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %598, <4 x float> %576) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %617 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %617, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %619 = extractelement <4 x float> %lazy.score2029, i64 0, !dbg !24
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !24
  %621 = fadd float %k0.sum.anchor1846, %620, !dbg !25
  %k0.sum.anchor1854 = call float asm sideeffect "", "=v,0"(float %621) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> poison, float %620, i64 0, !dbg !26
  %623 = extractelement <4 x float> %lazy.score2029, i64 1, !dbg !24
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !24
  %625 = fadd float %k0.sum.anchor1847, %624, !dbg !25
  %k0.sum.anchor1855 = call float asm sideeffect "", "=v,0"(float %625) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %626 = insertelement <4 x float> %622, float %624, i64 1, !dbg !26
  %627 = extractelement <4 x float> %lazy.score2029, i64 2, !dbg !24
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !24
  %629 = fadd float %k0.sum.anchor1848, %628, !dbg !25
  %k0.sum.anchor1856 = call float asm sideeffect "", "=v,0"(float %629) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %630 = insertelement <4 x float> %626, float %628, i64 2, !dbg !26
  %631 = extractelement <4 x float> %lazy.score2029, i64 3, !dbg !24
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !24
  %633 = fadd float %k0.sum.anchor1849, %632, !dbg !25
  %k0.sum.anchor1857 = call float asm sideeffect "", "=v,0"(float %633) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %634 = insertelement <4 x float> %630, float %632, i64 3, !dbg !26
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02072 = extractelement <4 x float> %math.result252, i32 0, !dbg !26
  %.cvtpack.f12073 = extractelement <4 x float> %math.result252, i32 1, !dbg !26
  %.cvtpack.f22074 = extractelement <4 x float> %math.result252, i32 2, !dbg !26
  %.cvtpack.f32075 = extractelement <4 x float> %math.result252, i32 3, !dbg !26
  %.cvtpack.p02076 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02072, float %.cvtpack.f12073), !dbg !26
  %.cvtpack.p12077 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22074, float %.cvtpack.f32075), !dbg !26
  %.cvtpack.v02078 = insertelement <2 x i32> poison, i32 %.cvtpack.p02076, i32 0, !dbg !26
  %.cvtpack.v12079 = insertelement <2 x i32> %.cvtpack.v02078, i32 %.cvtpack.p12077, i32 1, !dbg !26
  %635 = bitcast <2 x i32> %.cvtpack.v12079 to <4 x half>, !dbg !26
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %598, <4 x float> %596) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %617, <4 x float> %597) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %638 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %639 = extractelement <4 x float> %lazy.score2022, i64 0, !dbg !24
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !24
  %641 = fadd float %k0.sum.anchor1850, %640, !dbg !25
  %k0.sum.anchor1858 = call float asm sideeffect "", "=v,0"(float %641) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %642 = insertelement <4 x float> poison, float %640, i64 0, !dbg !26
  %643 = extractelement <4 x float> %lazy.score2022, i64 1, !dbg !24
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !24
  %645 = fadd float %k0.sum.anchor1851, %644, !dbg !25
  %k0.sum.anchor1859 = call float asm sideeffect "", "=v,0"(float %645) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %646 = insertelement <4 x float> %642, float %644, i64 1, !dbg !26
  %647 = extractelement <4 x float> %lazy.score2022, i64 2, !dbg !24
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !24
  %649 = fadd float %k0.sum.anchor1852, %648, !dbg !25
  %k0.sum.anchor1860 = call float asm sideeffect "", "=v,0"(float %649) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %650 = insertelement <4 x float> %646, float %648, i64 2, !dbg !26
  %651 = extractelement <4 x float> %lazy.score2022, i64 3, !dbg !24
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !24
  %653 = fadd float %k0.sum.anchor1853, %652, !dbg !25
  %k0.sum.anchor1861 = call float asm sideeffect "", "=v,0"(float %653) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %654 = insertelement <4 x float> %650, float %652, i64 3, !dbg !26
  %math.result238 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02080 = extractelement <4 x float> %math.result238, i32 0, !dbg !26
  %.cvtpack.f12081 = extractelement <4 x float> %math.result238, i32 1, !dbg !26
  %.cvtpack.f22082 = extractelement <4 x float> %math.result238, i32 2, !dbg !26
  %.cvtpack.f32083 = extractelement <4 x float> %math.result238, i32 3, !dbg !26
  %.cvtpack.p02084 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02080, float %.cvtpack.f12081), !dbg !26
  %.cvtpack.p12085 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22082, float %.cvtpack.f32083), !dbg !26
  %.cvtpack.v02086 = insertelement <2 x i32> poison, i32 %.cvtpack.p02084, i32 0, !dbg !26
  %.cvtpack.v12087 = insertelement <2 x i32> %.cvtpack.v02086, i32 %.cvtpack.p12085, i32 1, !dbg !26
  %655 = bitcast <2 x i32> %.cvtpack.v12087 to <4 x half>, !dbg !26
  %math.packed239 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %638, <4 x float> %616) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %657 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %657, <4 x float> %618) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = extractelement <4 x float> %lazy.score2030, i64 0, !dbg !24
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !24
  %661 = fadd float %k0.sum.anchor1854, %660, !dbg !25
  %k0.sum.anchor1862 = call float asm sideeffect "", "=v,0"(float %661) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %662 = insertelement <4 x float> poison, float %660, i64 0, !dbg !26
  %663 = extractelement <4 x float> %lazy.score2030, i64 1, !dbg !24
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !24
  %665 = fadd float %k0.sum.anchor1855, %664, !dbg !25
  %k0.sum.anchor1863 = call float asm sideeffect "", "=v,0"(float %665) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %666 = insertelement <4 x float> %662, float %664, i64 1, !dbg !26
  %667 = extractelement <4 x float> %lazy.score2030, i64 2, !dbg !24
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !24
  %669 = fadd float %k0.sum.anchor1856, %668, !dbg !25
  %k0.sum.anchor1864 = call float asm sideeffect "", "=v,0"(float %669) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %666, float %668, i64 2, !dbg !26
  %671 = extractelement <4 x float> %lazy.score2030, i64 3, !dbg !24
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !24
  %673 = fadd float %k0.sum.anchor1857, %672, !dbg !25
  %k0.sum.anchor1865 = call float asm sideeffect "", "=v,0"(float %673) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %674 = insertelement <4 x float> %670, float %672, i64 3, !dbg !26
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %674) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02088 = extractelement <4 x float> %math.result254, i32 0, !dbg !26
  %.cvtpack.f12089 = extractelement <4 x float> %math.result254, i32 1, !dbg !26
  %.cvtpack.f22090 = extractelement <4 x float> %math.result254, i32 2, !dbg !26
  %.cvtpack.f32091 = extractelement <4 x float> %math.result254, i32 3, !dbg !26
  %.cvtpack.p02092 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02088, float %.cvtpack.f12089), !dbg !26
  %.cvtpack.p12093 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22090, float %.cvtpack.f32091), !dbg !26
  %.cvtpack.v02094 = insertelement <2 x i32> poison, i32 %.cvtpack.p02092, i32 0, !dbg !26
  %.cvtpack.v12095 = insertelement <2 x i32> %.cvtpack.v02094, i32 %.cvtpack.p12093, i32 1, !dbg !26
  %675 = bitcast <2 x i32> %.cvtpack.v12095 to <4 x half>, !dbg !26
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %675) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %638, <4 x float> %636) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %657, <4 x float> %637) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %678 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %679 = extractelement <4 x float> %lazy.score2023, i64 0, !dbg !24
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !24
  %681 = fadd float %k0.sum.anchor1858, %680, !dbg !25
  %k0.sum.anchor1866 = call float asm sideeffect "", "=v,0"(float %681) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %682 = insertelement <4 x float> poison, float %680, i64 0, !dbg !26
  %683 = extractelement <4 x float> %lazy.score2023, i64 1, !dbg !24
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !24
  %685 = fadd float %k0.sum.anchor1859, %684, !dbg !25
  %k0.sum.anchor1867 = call float asm sideeffect "", "=v,0"(float %685) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %686 = insertelement <4 x float> %682, float %684, i64 1, !dbg !26
  %687 = extractelement <4 x float> %lazy.score2023, i64 2, !dbg !24
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !24
  %689 = fadd float %k0.sum.anchor1860, %688, !dbg !25
  %k0.sum.anchor1868 = call float asm sideeffect "", "=v,0"(float %689) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %690 = insertelement <4 x float> %686, float %688, i64 2, !dbg !26
  %691 = extractelement <4 x float> %lazy.score2023, i64 3, !dbg !24
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !24
  %693 = fadd float %k0.sum.anchor1861, %692, !dbg !25
  %k0.sum.anchor1869 = call float asm sideeffect "", "=v,0"(float %693) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %694 = insertelement <4 x float> %690, float %692, i64 3, !dbg !26
  %math.result240 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02096 = extractelement <4 x float> %math.result240, i32 0, !dbg !26
  %.cvtpack.f12097 = extractelement <4 x float> %math.result240, i32 1, !dbg !26
  %.cvtpack.f22098 = extractelement <4 x float> %math.result240, i32 2, !dbg !26
  %.cvtpack.f32099 = extractelement <4 x float> %math.result240, i32 3, !dbg !26
  %.cvtpack.p02100 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02096, float %.cvtpack.f12097), !dbg !26
  %.cvtpack.p12101 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22098, float %.cvtpack.f32099), !dbg !26
  %.cvtpack.v02102 = insertelement <2 x i32> poison, i32 %.cvtpack.p02100, i32 0, !dbg !26
  %.cvtpack.v12103 = insertelement <2 x i32> %.cvtpack.v02102, i32 %.cvtpack.p12101, i32 1, !dbg !26
  %695 = bitcast <2 x i32> %.cvtpack.v12103 to <4 x half>, !dbg !26
  %math.packed241 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %678, <4 x float> %656) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %697 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %697, <4 x float> %658) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %699 = extractelement <4 x float> %lazy.score2031, i64 0, !dbg !24
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !24
  %701 = fadd float %k0.sum.anchor1862, %700, !dbg !25
  %k0.sum.anchor1870 = call float asm sideeffect "", "=v,0"(float %701) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %702 = insertelement <4 x float> poison, float %700, i64 0, !dbg !26
  %703 = extractelement <4 x float> %lazy.score2031, i64 1, !dbg !24
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !24
  %705 = fadd float %k0.sum.anchor1863, %704, !dbg !25
  %k0.sum.anchor1871 = call float asm sideeffect "", "=v,0"(float %705) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %706 = insertelement <4 x float> %702, float %704, i64 1, !dbg !26
  %707 = extractelement <4 x float> %lazy.score2031, i64 2, !dbg !24
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !24
  %709 = fadd float %k0.sum.anchor1864, %708, !dbg !25
  %k0.sum.anchor1872 = call float asm sideeffect "", "=v,0"(float %709) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %710 = insertelement <4 x float> %706, float %708, i64 2, !dbg !26
  %711 = extractelement <4 x float> %lazy.score2031, i64 3, !dbg !24
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !24
  %713 = fadd float %k0.sum.anchor1865, %712, !dbg !25
  %k0.sum.anchor1873 = call float asm sideeffect "", "=v,0"(float %713) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %714 = insertelement <4 x float> %710, float %712, i64 3, !dbg !26
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %714) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02104 = extractelement <4 x float> %math.result256, i32 0, !dbg !26
  %.cvtpack.f12105 = extractelement <4 x float> %math.result256, i32 1, !dbg !26
  %.cvtpack.f22106 = extractelement <4 x float> %math.result256, i32 2, !dbg !26
  %.cvtpack.f32107 = extractelement <4 x float> %math.result256, i32 3, !dbg !26
  %.cvtpack.p02108 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02104, float %.cvtpack.f12105), !dbg !26
  %.cvtpack.p12109 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22106, float %.cvtpack.f32107), !dbg !26
  %.cvtpack.v02110 = insertelement <2 x i32> poison, i32 %.cvtpack.p02108, i32 0, !dbg !26
  %.cvtpack.v12111 = insertelement <2 x i32> %.cvtpack.v02110, i32 %.cvtpack.p12109, i32 1, !dbg !26
  %715 = bitcast <2 x i32> %.cvtpack.v12111 to <4 x half>, !dbg !26
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %715) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %678, <4 x float> %676) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %697, <4 x float> %677) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %718 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %719 = extractelement <4 x float> %lazy.score2024, i64 0, !dbg !24
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !24
  %721 = fadd float %k0.sum.anchor1866, %720, !dbg !25
  %k0.sum.anchor1874 = call float asm sideeffect "", "=v,0"(float %721) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %722 = insertelement <4 x float> poison, float %720, i64 0, !dbg !26
  %723 = extractelement <4 x float> %lazy.score2024, i64 1, !dbg !24
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !24
  %725 = fadd float %k0.sum.anchor1867, %724, !dbg !25
  %k0.sum.anchor1875 = call float asm sideeffect "", "=v,0"(float %725) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %726 = insertelement <4 x float> %722, float %724, i64 1, !dbg !26
  %727 = extractelement <4 x float> %lazy.score2024, i64 2, !dbg !24
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !24
  %729 = fadd float %k0.sum.anchor1868, %728, !dbg !25
  %k0.sum.anchor1876 = call float asm sideeffect "", "=v,0"(float %729) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %730 = insertelement <4 x float> %726, float %728, i64 2, !dbg !26
  %731 = extractelement <4 x float> %lazy.score2024, i64 3, !dbg !24
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !24
  %733 = fadd float %k0.sum.anchor1869, %732, !dbg !25
  %k0.sum.anchor1877 = call float asm sideeffect "", "=v,0"(float %733) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %734 = insertelement <4 x float> %730, float %732, i64 3, !dbg !26
  %math.result242 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %734) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02112 = extractelement <4 x float> %math.result242, i32 0, !dbg !26
  %.cvtpack.f12113 = extractelement <4 x float> %math.result242, i32 1, !dbg !26
  %.cvtpack.f22114 = extractelement <4 x float> %math.result242, i32 2, !dbg !26
  %.cvtpack.f32115 = extractelement <4 x float> %math.result242, i32 3, !dbg !26
  %.cvtpack.p02116 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02112, float %.cvtpack.f12113), !dbg !26
  %.cvtpack.p12117 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22114, float %.cvtpack.f32115), !dbg !26
  %.cvtpack.v02118 = insertelement <2 x i32> poison, i32 %.cvtpack.p02116, i32 0, !dbg !26
  %.cvtpack.v12119 = insertelement <2 x i32> %.cvtpack.v02118, i32 %.cvtpack.p12117, i32 1, !dbg !26
  %735 = bitcast <2 x i32> %.cvtpack.v12119 to <4 x half>, !dbg !26
  %math.packed243 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %735) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %718, <4 x float> %696) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %737 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %737, <4 x float> %698) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %739 = extractelement <4 x float> %lazy.score2032, i64 0, !dbg !24
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !24
  %741 = fadd float %k0.sum.anchor1870, %740, !dbg !25
  %k0.sum.anchor1878 = call float asm sideeffect "", "=v,0"(float %741) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %742 = insertelement <4 x float> poison, float %740, i64 0, !dbg !26
  %743 = extractelement <4 x float> %lazy.score2032, i64 1, !dbg !24
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !24
  %745 = fadd float %k0.sum.anchor1871, %744, !dbg !25
  %k0.sum.anchor1879 = call float asm sideeffect "", "=v,0"(float %745) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %746 = insertelement <4 x float> %742, float %744, i64 1, !dbg !26
  %747 = extractelement <4 x float> %lazy.score2032, i64 2, !dbg !24
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !24
  %749 = fadd float %k0.sum.anchor1872, %748, !dbg !25
  %k0.sum.anchor1880 = call float asm sideeffect "", "=v,0"(float %749) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %750 = insertelement <4 x float> %746, float %748, i64 2, !dbg !26
  %751 = extractelement <4 x float> %lazy.score2032, i64 3, !dbg !24
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !24
  %753 = fadd float %k0.sum.anchor1873, %752, !dbg !25
  %k0.sum.anchor1881 = call float asm sideeffect "", "=v,0"(float %753) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %754 = insertelement <4 x float> %750, float %752, i64 3, !dbg !26
  %math.result258 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %754) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02120 = extractelement <4 x float> %math.result258, i32 0, !dbg !26
  %.cvtpack.f12121 = extractelement <4 x float> %math.result258, i32 1, !dbg !26
  %.cvtpack.f22122 = extractelement <4 x float> %math.result258, i32 2, !dbg !26
  %.cvtpack.f32123 = extractelement <4 x float> %math.result258, i32 3, !dbg !26
  %.cvtpack.p02124 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02120, float %.cvtpack.f12121), !dbg !26
  %.cvtpack.p12125 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22122, float %.cvtpack.f32123), !dbg !26
  %.cvtpack.v02126 = insertelement <2 x i32> poison, i32 %.cvtpack.p02124, i32 0, !dbg !26
  %.cvtpack.v12127 = insertelement <2 x i32> %.cvtpack.v02126, i32 %.cvtpack.p12125, i32 1, !dbg !26
  %755 = bitcast <2 x i32> %.cvtpack.v12127 to <4 x half>, !dbg !26
  %math.packed259 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %755) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %718, <4 x float> %716) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %737, <4 x float> %717) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %758 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = extractelement <4 x float> %lazy.score2025, i64 0, !dbg !24
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !24
  %761 = fadd float %k0.sum.anchor1874, %760, !dbg !25
  %k0.sum.anchor1882 = call float asm sideeffect "", "=v,0"(float %761) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %762 = insertelement <4 x float> poison, float %760, i64 0, !dbg !26
  %763 = extractelement <4 x float> %lazy.score2025, i64 1, !dbg !24
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !24
  %765 = fadd float %k0.sum.anchor1875, %764, !dbg !25
  %k0.sum.anchor1883 = call float asm sideeffect "", "=v,0"(float %765) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %766 = insertelement <4 x float> %762, float %764, i64 1, !dbg !26
  %767 = extractelement <4 x float> %lazy.score2025, i64 2, !dbg !24
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !24
  %769 = fadd float %k0.sum.anchor1876, %768, !dbg !25
  %k0.sum.anchor1884 = call float asm sideeffect "", "=v,0"(float %769) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %770 = insertelement <4 x float> %766, float %768, i64 2, !dbg !26
  %771 = extractelement <4 x float> %lazy.score2025, i64 3, !dbg !24
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !24
  %773 = fadd float %k0.sum.anchor1877, %772, !dbg !25
  %k0.sum.anchor1885 = call float asm sideeffect "", "=v,0"(float %773) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %774 = insertelement <4 x float> %770, float %772, i64 3, !dbg !26
  %math.result244 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %774) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02128 = extractelement <4 x float> %math.result244, i32 0, !dbg !26
  %.cvtpack.f12129 = extractelement <4 x float> %math.result244, i32 1, !dbg !26
  %.cvtpack.f22130 = extractelement <4 x float> %math.result244, i32 2, !dbg !26
  %.cvtpack.f32131 = extractelement <4 x float> %math.result244, i32 3, !dbg !26
  %.cvtpack.p02132 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02128, float %.cvtpack.f12129), !dbg !26
  %.cvtpack.p12133 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22130, float %.cvtpack.f32131), !dbg !26
  %.cvtpack.v02134 = insertelement <2 x i32> poison, i32 %.cvtpack.p02132, i32 0, !dbg !26
  %.cvtpack.v12135 = insertelement <2 x i32> %.cvtpack.v02134, i32 %.cvtpack.p12133, i32 1, !dbg !26
  %775 = bitcast <2 x i32> %.cvtpack.v12135 to <4 x half>, !dbg !26
  %math.packed245 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %775) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %758, <4 x float> %736) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %777, <4 x float> %738) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = extractelement <4 x float> %lazy.score2033, i64 0, !dbg !24
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !24
  %781 = fadd float %k0.sum.anchor1878, %780, !dbg !25
  %k0.sum.anchor1886 = call float asm sideeffect "", "=v,0"(float %781) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %782 = insertelement <4 x float> poison, float %780, i64 0, !dbg !26
  %783 = extractelement <4 x float> %lazy.score2033, i64 1, !dbg !24
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !24
  %785 = fadd float %k0.sum.anchor1879, %784, !dbg !25
  %k0.sum.anchor1887 = call float asm sideeffect "", "=v,0"(float %785) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %786 = insertelement <4 x float> %782, float %784, i64 1, !dbg !26
  %787 = extractelement <4 x float> %lazy.score2033, i64 2, !dbg !24
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !24
  %789 = fadd float %k0.sum.anchor1880, %788, !dbg !25
  %k0.sum.anchor1888 = call float asm sideeffect "", "=v,0"(float %789) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %790 = insertelement <4 x float> %786, float %788, i64 2, !dbg !26
  %791 = extractelement <4 x float> %lazy.score2033, i64 3, !dbg !24
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !24
  %793 = fadd float %k0.sum.anchor1881, %792, !dbg !25
  %k0.sum.anchor1889 = call float asm sideeffect "", "=v,0"(float %793) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %794 = insertelement <4 x float> %790, float %792, i64 3, !dbg !26
  %math.result260 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02136 = extractelement <4 x float> %math.result260, i32 0, !dbg !26
  %.cvtpack.f12137 = extractelement <4 x float> %math.result260, i32 1, !dbg !26
  %.cvtpack.f22138 = extractelement <4 x float> %math.result260, i32 2, !dbg !26
  %.cvtpack.f32139 = extractelement <4 x float> %math.result260, i32 3, !dbg !26
  %.cvtpack.p02140 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02136, float %.cvtpack.f12137), !dbg !26
  %.cvtpack.p12141 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22138, float %.cvtpack.f32139), !dbg !26
  %.cvtpack.v02142 = insertelement <2 x i32> poison, i32 %.cvtpack.p02140, i32 0, !dbg !26
  %.cvtpack.v12143 = insertelement <2 x i32> %.cvtpack.v02142, i32 %.cvtpack.p12141, i32 1, !dbg !26
  %795 = bitcast <2 x i32> %.cvtpack.v12143 to <4 x half>, !dbg !26
  %math.packed261 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %758, <4 x float> %756) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %777, <4 x float> %757) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = bitcast <4 x i32> %525 to <8 x half>, !dbg !14
  %799 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = extractelement <4 x float> %lazy.score2026, i64 0, !dbg !24
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !24
  %802 = fadd float %k0.sum.anchor1882, %801, !dbg !25
  %k0.sum.anchor1890 = call float asm sideeffect "", "=v,0"(float %802) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %803 = insertelement <4 x float> poison, float %801, i64 0, !dbg !26
  %804 = extractelement <4 x float> %lazy.score2026, i64 1, !dbg !24
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !24
  %806 = fadd float %k0.sum.anchor1883, %805, !dbg !25
  %k0.sum.anchor1891 = call float asm sideeffect "", "=v,0"(float %806) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %807 = fadd float %k0.sum.anchor1890, %k0.sum.anchor1891, !dbg !25
  %k0.sum.anchor1892 = call float asm sideeffect "", "=v,0"(float %807) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %808 = insertelement <4 x float> %803, float %805, i64 1, !dbg !26
  %809 = extractelement <4 x float> %lazy.score2026, i64 2, !dbg !24
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !24
  %811 = fadd float %k0.sum.anchor1884, %810, !dbg !25
  %k0.sum.anchor1893 = call float asm sideeffect "", "=v,0"(float %811) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %812 = insertelement <4 x float> %808, float %810, i64 2, !dbg !26
  %813 = extractelement <4 x float> %lazy.score2026, i64 3, !dbg !24
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !24
  %815 = fadd float %k0.sum.anchor1885, %814, !dbg !25
  %k0.sum.anchor1894 = call float asm sideeffect "", "=v,0"(float %815) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %816 = fadd float %k0.sum.anchor1893, %k0.sum.anchor1894, !dbg !25
  %k0.sum.anchor1895 = call float asm sideeffect "", "=v,0"(float %816) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %817 = fadd float %k0.sum.anchor1892, %k0.sum.anchor1895, !dbg !25
  %k0.sum.anchor1896 = call float asm sideeffect "", "=v,0"(float %817) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %818 = fadd float %k0.sum.anchor1896, %500, !dbg !25
  %k0.sum.anchor1897 = call float asm sideeffect "", "=v,0"(float %818) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %819 = insertelement <1 x float> poison, float %k0.sum.anchor1897, i64 0, !dbg !6
  %820 = insertvalue [2 x <1 x float>] poison, <1 x float> %819, 0, !dbg !6
  %821 = insertelement <4 x float> %812, float %814, i64 3, !dbg !26
  %math.result246 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %821) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02144 = extractelement <4 x float> %math.result246, i32 0, !dbg !26
  %.cvtpack.f12145 = extractelement <4 x float> %math.result246, i32 1, !dbg !26
  %.cvtpack.f22146 = extractelement <4 x float> %math.result246, i32 2, !dbg !26
  %.cvtpack.f32147 = extractelement <4 x float> %math.result246, i32 3, !dbg !26
  %.cvtpack.p02148 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02144, float %.cvtpack.f12145), !dbg !26
  %.cvtpack.p12149 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22146, float %.cvtpack.f32147), !dbg !26
  %.cvtpack.v02150 = insertelement <2 x i32> poison, i32 %.cvtpack.p02148, i32 0, !dbg !26
  %.cvtpack.v12151 = insertelement <2 x i32> %.cvtpack.v02150, i32 %.cvtpack.p12149, i32 1, !dbg !26
  %822 = bitcast <2 x i32> %.cvtpack.v12151 to <4 x half>, !dbg !26
  %math.packed247 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %822) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %799, <4 x float> %776) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %824 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %824, <4 x float> %778) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = extractelement <4 x float> %lazy.score2034, i64 0, !dbg !24
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !24
  %828 = fadd float %k0.sum.anchor1886, %827, !dbg !25
  %k0.sum.anchor1898 = call float asm sideeffect "", "=v,0"(float %828) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %829 = insertelement <4 x float> poison, float %827, i64 0, !dbg !26
  %830 = extractelement <4 x float> %lazy.score2034, i64 1, !dbg !24
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !24
  %832 = fadd float %k0.sum.anchor1887, %831, !dbg !25
  %k0.sum.anchor1899 = call float asm sideeffect "", "=v,0"(float %832) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %833 = fadd float %k0.sum.anchor1898, %k0.sum.anchor1899, !dbg !25
  %k0.sum.anchor1900 = call float asm sideeffect "", "=v,0"(float %833) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %834 = insertelement <4 x float> %829, float %831, i64 1, !dbg !26
  %835 = extractelement <4 x float> %lazy.score2034, i64 2, !dbg !24
  %836 = call float @llvm.amdgcn.exp2.f32(float %835), !dbg !24
  %837 = fadd float %k0.sum.anchor1888, %836, !dbg !25
  %k0.sum.anchor1901 = call float asm sideeffect "", "=v,0"(float %837) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %838 = insertelement <4 x float> %834, float %836, i64 2, !dbg !26
  %839 = extractelement <4 x float> %lazy.score2034, i64 3, !dbg !24
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !24
  %841 = fadd float %k0.sum.anchor1889, %840, !dbg !25
  %k0.sum.anchor1902 = call float asm sideeffect "", "=v,0"(float %841) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %842 = fadd float %k0.sum.anchor1901, %k0.sum.anchor1902, !dbg !25
  %k0.sum.anchor1903 = call float asm sideeffect "", "=v,0"(float %842) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %843 = fadd float %k0.sum.anchor1900, %k0.sum.anchor1903, !dbg !25
  %k0.sum.anchor1904 = call float asm sideeffect "", "=v,0"(float %843) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %844 = fadd float %k0.sum.anchor1904, %502, !dbg !25
  %k0.sum.anchor1905 = call float asm sideeffect "", "=v,0"(float %844) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %845 = insertelement <1 x float> poison, float %k0.sum.anchor1905, i64 0, !dbg !6
  %846 = insertvalue [2 x <1 x float>] %820, <1 x float> %845, 1, !dbg !6
  %847 = insertelement <4 x float> %838, float %840, i64 3, !dbg !26
  %math.result262 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %847) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02152 = extractelement <4 x float> %math.result262, i32 0, !dbg !26
  %.cvtpack.f12153 = extractelement <4 x float> %math.result262, i32 1, !dbg !26
  %.cvtpack.f22154 = extractelement <4 x float> %math.result262, i32 2, !dbg !26
  %.cvtpack.f32155 = extractelement <4 x float> %math.result262, i32 3, !dbg !26
  %.cvtpack.p02156 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02152, float %.cvtpack.f12153), !dbg !26
  %.cvtpack.p12157 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22154, float %.cvtpack.f32155), !dbg !26
  %.cvtpack.v02158 = insertelement <2 x i32> poison, i32 %.cvtpack.p02156, i32 0, !dbg !26
  %.cvtpack.v12159 = insertelement <2 x i32> %.cvtpack.v02158, i32 %.cvtpack.p12157, i32 1, !dbg !26
  %848 = bitcast <2 x i32> %.cvtpack.v12159 to <4 x half>, !dbg !26
  %math.packed263 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %848) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %799, <4 x float> %796) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %824, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %851 = extractvalue [2 x <8 x float>] %lazy.state2038, 0, !dbg !27
  %852 = shufflevector <8 x float> %851, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %853 = shufflevector <8 x float> %851, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %854 = extractvalue [2 x <8 x float>] %lazy.state2038, 1, !dbg !27
  %855 = shufflevector <8 x float> %854, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %856 = shufflevector <8 x float> %854, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1307 = add nsw i64 %195, %address.local782, !dbg !16
  %vdb.address1308 = trunc i64 %vdb.address1307 to i32, !dbg !16
  %vdb.address1309 = shl i32 %vdb.address1308, 1, !dbg !16
  %vdb.address1314 = trunc nuw i64 %195 to i32, !dbg !16
  %vdb.address1315 = or disjoint i32 %address.local789, %vdb.address1314, !dbg !16
  %vdb.address1316 = shl nuw nsw i32 %vdb.address1315, 1, !dbg !16
  %857 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %vdb.address1309, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %858 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %vdb.address1316, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local798, i64 8192, !dbg !14
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !14
  %vdb.read1300 = getelementptr i8, ptr addrspace(3) %address.local807, i64 8448, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1300, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %vdb.read1301 = getelementptr i8, ptr addrspace(3) %address.local807, i64 8704, !dbg !14
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1301, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !14
  %vdb.read1302 = getelementptr i8, ptr addrspace(3) %address.local807, i64 8960, !dbg !14
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1302, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !14
  %vdb.read1303 = getelementptr i8, ptr addrspace(3) %address.local798, i64 12288, !dbg !14
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1303, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !14
  %vdb.read1304 = getelementptr i8, ptr addrspace(3) %address.local807, i64 12544, !dbg !14
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1304, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !14
  %vdb.read1305 = getelementptr i8, ptr addrspace(3) %address.local807, i64 12800, !dbg !14
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1305, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !14
  %vdb.read1306 = getelementptr i8, ptr addrspace(3) %address.local807, i64 13056, !dbg !14
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1306, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %874 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %874, <4 x float> %852) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %876, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %874, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %876, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %880, <4 x float> %875) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %882, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %880, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %882, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %886, <4 x float> %881) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %888, <4 x float> %883) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %886, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %888, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %892, <4 x float> %887) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %894, <4 x float> %889) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %892, <4 x float> %890) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %894, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %898, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %900, <4 x float> %895) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %898, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %900, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %904, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %906, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %904, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %906, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %910, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %912, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %910, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %912, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = bitcast <4 x i32> %873 to <8 x half>, !dbg !14
  %917 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %917, <4 x float> %911) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %919, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %917, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %919, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %923 = extractvalue [2 x <8 x float>] %lazy.state2037, 0, !dbg !27
  %924 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %925 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %926 = extractvalue [2 x <8 x float>] %lazy.state2037, 1, !dbg !27
  %927 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %928 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1321 = add nsw i64 %248, %address.local782, !dbg !16
  %vdb.address1322 = trunc i64 %vdb.address1321 to i32, !dbg !16
  %vdb.address1323 = shl i32 %vdb.address1322, 1, !dbg !16
  %vdb.address1328 = trunc nuw i64 %248 to i32, !dbg !16
  %vdb.address1329 = or disjoint i32 %address.local789, %vdb.address1328, !dbg !16
  %vdb.address1330 = shl nuw nsw i32 %vdb.address1329, 1, !dbg !16
  %929 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1323, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %930 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1299, i32 %vdb.address1330, <4 x i32> %dma.buffer.rsrc1206) #10, !dbg !16
  %931 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local798, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !14
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local808, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !14
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local812, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !14
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local816, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !14
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local818, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !14
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !14
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local826, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !14
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %946 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %946, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %948 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %948, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %946, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %948, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %952, <4 x float> %947) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %954, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %952, <4 x float> %950) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %954, <4 x float> %951) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %958, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %960, <4 x float> %955) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %958, <4 x float> %956) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %960, <4 x float> %957) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %964, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %966, <4 x float> %961) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %964, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %966, <4 x float> %963) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %970, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %972, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %970, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %972, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %976, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %978, <4 x float> %973) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %976, <4 x float> %974) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %978, <4 x float> %975) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %982, <4 x float> %977) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %984, <4 x float> %979) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %982, <4 x float> %980) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %984, <4 x float> %981) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = bitcast <4 x i32> %945 to <8 x half>, !dbg !14
  %989 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %989, <4 x float> %983) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %991, <4 x float> %985) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %989, <4 x float> %986) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %991, <4 x float> %987) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %995 = extractvalue [2 x <8 x float>] %lazy.state2036, 0, !dbg !27
  %996 = shufflevector <8 x float> %995, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %997 = shufflevector <8 x float> %995, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %998 = extractvalue [2 x <8 x float>] %lazy.state2036, 1, !dbg !27
  %999 = shufflevector <8 x float> %998, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1000 = shufflevector <8 x float> %998, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1001 = add nuw nsw i64 %101, 1, !dbg !6
  %k0.has.next = icmp ult i64 %1001, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %1001, 14, !dbg !13
  %k0.next.1916 = or disjoint i64 %111, %k0.next., !dbg !13
  %k0.next.kdma.source1361 = or disjoint i64 %k0.next.1916, %kdma.source1360, !dbg !13
  %k0.next.kdma.source1362 = trunc nuw i64 %k0.next.kdma.source1361 to i32, !dbg !13
  %k0.next.kdma.source1363 = shl nuw nsw i32 %k0.next.kdma.source1362, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1363, %kdma.intra, !dbg !13
  %k0.next.1917 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1381 = or disjoint i64 %k0.next.1916, %kdma.source1380, !dbg !13
  %k0.next.kdma.source1382 = trunc nuw i64 %k0.next.kdma.source1381 to i32, !dbg !13
  %k0.next.kdma.source1383 = shl nuw nsw i32 %k0.next.kdma.source1382, 1, !dbg !13
  %k0.next.kdma.offset1384 = or disjoint i32 %k0.next.kdma.source1383, %kdma.intra.hi, !dbg !13
  %k0.next.1918 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1385, i32 %k0.next.kdma.offset1384, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1405 = or disjoint i64 %k0.next.1916, %kdma.source1404, !dbg !13
  %k0.next.kdma.source1406 = trunc nuw i64 %k0.next.kdma.source1405 to i32, !dbg !13
  %k0.next.kdma.source1407 = shl nuw nsw i32 %k0.next.kdma.source1406, 1, !dbg !13
  %k0.next.kdma.offset1408 = add nuw i32 %k0.next.kdma.source1407, %kdma.intra, !dbg !13
  %k0.next.1919 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.next.kdma.offset1408, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1429 = or disjoint i64 %k0.next.1916, %kdma.source1428, !dbg !13
  %k0.next.kdma.source1430 = trunc nuw i64 %k0.next.kdma.source1429 to i32, !dbg !13
  %k0.next.kdma.source1431 = shl nuw nsw i32 %k0.next.kdma.source1430, 1, !dbg !13
  %k0.next.kdma.offset1432 = add nuw i32 %k0.next.kdma.source1431, %kdma.intra.hi, !dbg !13
  %k0.next.1920 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1433, i32 %k0.next.kdma.offset1432, <4 x i32> %dma.buffer.rsrc1222) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %1002 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1003 = bitcast <4 x i32> %1002 to <8 x half>, !dbg !14
  %1004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1300, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !14
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1301, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !14
  %1008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1302, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !14
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1303, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !14
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1304, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !14
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1305, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !14
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1306, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1017 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1017, <4 x float> %996) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1019 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1019, <4 x float> %997) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1017, <4 x float> %999) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1019, <4 x float> %1000) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1023 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1023, <4 x float> %1018) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1025 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed235, <4 x half> %1025, <4 x float> %1020) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1023, <4 x float> %1021) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1025, <4 x float> %1022) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1029 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1029, <4 x float> %1024) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1031 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed237, <4 x half> %1031, <4 x float> %1026) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1029, <4 x float> %1027) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1031, <4 x float> %1028) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1035 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1035, <4 x float> %1030) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1037 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed239, <4 x half> %1037, <4 x float> %1032) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1035, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1037, <4 x float> %1034) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1041 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1041, <4 x float> %1036) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1043 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed241, <4 x half> %1043, <4 x float> %1038) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1041, <4 x float> %1039) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1043, <4 x float> %1040) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1047 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1047, <4 x float> %1042) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1049 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed243, <4 x half> %1049, <4 x float> %1044) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1047, <4 x float> %1045) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1049, <4 x float> %1046) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1053, <4 x float> %1048) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed245, <4 x half> %1055, <4 x float> %1050) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1053, <4 x float> %1051) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1055, <4 x float> %1052) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !14
  %1060 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1060, <4 x float> %1054) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1062 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed247, <4 x half> %1062, <4 x float> %1056) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1060, <4 x float> %1057) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1062, <4 x float> %1058) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1066 = shufflevector <4 x float> %823, <4 x float> %825, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1067 = insertvalue [2 x <8 x float>] poison, <8 x float> %1066, 0, !dbg !6
  %1068 = shufflevector <4 x float> %849, <4 x float> %850, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1069 = insertvalue [2 x <8 x float>] %1067, <8 x float> %1068, 1, !dbg !6
  %1070 = shufflevector <4 x float> %918, <4 x float> %920, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1071 = insertvalue [2 x <8 x float>] poison, <8 x float> %1070, 0, !dbg !6
  %1072 = shufflevector <4 x float> %921, <4 x float> %922, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1073 = insertvalue [2 x <8 x float>] %1071, <8 x float> %1072, 1, !dbg !6
  %1074 = shufflevector <4 x float> %990, <4 x float> %992, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1075 = insertvalue [2 x <8 x float>] poison, <8 x float> %1074, 0, !dbg !6
  %1076 = shufflevector <4 x float> %993, <4 x float> %994, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1077 = insertvalue [2 x <8 x float>] %1075, <8 x float> %1076, 1, !dbg !6
  %1078 = shufflevector <4 x float> %1061, <4 x float> %1063, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1079 = insertvalue [2 x <8 x float>] poison, <8 x float> %1078, 0, !dbg !6
  %1080 = shufflevector <4 x float> %1064, <4 x float> %1065, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1081 = insertvalue [2 x <8 x float>] %1079, <8 x float> %1080, 1, !dbg !6
  br label %100, !dbg !6

1082:                                             ; preds = %100
  %1083 = extractvalue [2 x <1 x float>] %103, 1, !dbg !6
  %1084 = extractelement <1 x float> %1083, i64 0, !dbg !6
  %1085 = bitcast float %1084 to i32, !dbg !6
  %1086 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1085), !dbg !6
  %1087 = bitcast i32 %1086 to float, !dbg !6
  %1088 = fadd float %1084, %1087, !dbg !6
  %1089 = bitcast float %1088 to i32, !dbg !6
  %1090 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1089), !dbg !6
  %1091 = bitcast i32 %1090 to float, !dbg !6
  %1092 = fadd float %1088, %1091, !dbg !6
  %1093 = extractvalue [2 x <1 x float>] %103, 0, !dbg !6
  %1094 = extractelement <1 x float> %1093, i64 0, !dbg !6
  %1095 = bitcast float %1094 to i32, !dbg !6
  %1096 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1095), !dbg !6
  %1097 = bitcast i32 %1096 to float, !dbg !6
  %1098 = fadd float %1094, %1097, !dbg !6
  %1099 = bitcast float %1098 to i32, !dbg !6
  %1100 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1099), !dbg !6
  %1101 = bitcast i32 %1100 to float, !dbg !6
  %1102 = fadd float %1098, %1101, !dbg !6
  %1103 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1103, 0, !dbg !6
  %1104 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1104, 0, !dbg !6
  %1105 = extractvalue [2 x <8 x float>] %107, 0, !dbg !6
  %1106 = extractelement <8 x float> %1105, i64 0, !dbg !6
  %1107 = extractelement <8 x float> %1105, i64 1, !dbg !6
  %1108 = extractelement <8 x float> %1105, i64 2, !dbg !6
  %1109 = extractelement <8 x float> %1105, i64 3, !dbg !6
  %1110 = select i1 %.not, float %1107, float %1106, !dbg !6
  %1111 = bitcast float %1110 to i32, !dbg !6
  %1112 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1111), !dbg !6
  %1113 = bitcast i32 %1112 to float, !dbg !6
  %1114 = select i1 %.not, float %1106, float %1113, !dbg !6
  %1115 = select i1 %.not, float %1113, float %1107, !dbg !6
  %1116 = select i1 %.not, float %1109, float %1108, !dbg !6
  %1117 = bitcast float %1116 to i32, !dbg !6
  %1118 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1117), !dbg !6
  %1119 = bitcast i32 %1118 to float, !dbg !6
  %1120 = select i1 %.not, float %1108, float %1119, !dbg !6
  %1121 = select i1 %.not, float %1119, float %1109, !dbg !6
  %1122 = select i1 %.not40, float %1120, float %1114, !dbg !6
  %1123 = bitcast float %1122 to i32, !dbg !6
  %1124 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1123), !dbg !6
  %1125 = bitcast i32 %1124 to float, !dbg !6
  %1126 = select i1 %.not40, float %1114, float %1125, !dbg !6
  %1127 = select i1 %.not40, float %1125, float %1120, !dbg !6
  %1128 = select i1 %.not40, float %1121, float %1115, !dbg !6
  %1129 = bitcast float %1128 to i32, !dbg !6
  %1130 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1129), !dbg !6
  %1131 = bitcast i32 %1130 to float, !dbg !6
  %1132 = select i1 %.not40, float %1115, float %1131, !dbg !6
  %1133 = select i1 %.not40, float %1131, float %1121, !dbg !6
  %1134 = extractelement <8 x float> %1105, i64 4, !dbg !6
  %1135 = extractelement <8 x float> %1105, i64 5, !dbg !6
  %1136 = extractelement <8 x float> %1105, i64 6, !dbg !6
  %1137 = extractelement <8 x float> %1105, i64 7, !dbg !6
  %1138 = select i1 %.not, float %1135, float %1134, !dbg !6
  %1139 = bitcast float %1138 to i32, !dbg !6
  %1140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1139), !dbg !6
  %1141 = bitcast i32 %1140 to float, !dbg !6
  %1142 = select i1 %.not, float %1134, float %1141, !dbg !6
  %1143 = select i1 %.not, float %1141, float %1135, !dbg !6
  %1144 = select i1 %.not, float %1137, float %1136, !dbg !6
  %1145 = bitcast float %1144 to i32, !dbg !6
  %1146 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1145), !dbg !6
  %1147 = bitcast i32 %1146 to float, !dbg !6
  %1148 = select i1 %.not, float %1136, float %1147, !dbg !6
  %1149 = select i1 %.not, float %1147, float %1137, !dbg !6
  %1150 = select i1 %.not40, float %1148, float %1142, !dbg !6
  %1151 = bitcast float %1150 to i32, !dbg !6
  %1152 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1151), !dbg !6
  %1153 = bitcast i32 %1152 to float, !dbg !6
  %1154 = select i1 %.not40, float %1142, float %1153, !dbg !6
  %1155 = select i1 %.not40, float %1153, float %1148, !dbg !6
  %1156 = select i1 %.not40, float %1149, float %1143, !dbg !6
  %1157 = bitcast float %1156 to i32, !dbg !6
  %1158 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1157), !dbg !6
  %1159 = bitcast i32 %1158 to float, !dbg !6
  %1160 = select i1 %.not40, float %1143, float %1159, !dbg !6
  %1161 = select i1 %.not40, float %1159, float %1149, !dbg !6
  %1162 = extractvalue [2 x <8 x float>] %107, 1, !dbg !6
  %1163 = extractelement <8 x float> %1162, i64 0, !dbg !6
  %1164 = extractelement <8 x float> %1162, i64 1, !dbg !6
  %1165 = extractelement <8 x float> %1162, i64 2, !dbg !6
  %1166 = extractelement <8 x float> %1162, i64 3, !dbg !6
  %1167 = select i1 %.not, float %1164, float %1163, !dbg !6
  %1168 = bitcast float %1167 to i32, !dbg !6
  %1169 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1168), !dbg !6
  %1170 = bitcast i32 %1169 to float, !dbg !6
  %1171 = select i1 %.not, float %1163, float %1170, !dbg !6
  %1172 = select i1 %.not, float %1170, float %1164, !dbg !6
  %1173 = select i1 %.not, float %1166, float %1165, !dbg !6
  %1174 = bitcast float %1173 to i32, !dbg !6
  %1175 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1174), !dbg !6
  %1176 = bitcast i32 %1175 to float, !dbg !6
  %1177 = select i1 %.not, float %1165, float %1176, !dbg !6
  %1178 = select i1 %.not, float %1176, float %1166, !dbg !6
  %1179 = select i1 %.not40, float %1177, float %1171, !dbg !6
  %1180 = bitcast float %1179 to i32, !dbg !6
  %1181 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1180), !dbg !6
  %1182 = bitcast i32 %1181 to float, !dbg !6
  %1183 = select i1 %.not40, float %1171, float %1182, !dbg !6
  %1184 = select i1 %.not40, float %1182, float %1177, !dbg !6
  %1185 = select i1 %.not40, float %1178, float %1172, !dbg !6
  %1186 = bitcast float %1185 to i32, !dbg !6
  %1187 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1186), !dbg !6
  %1188 = bitcast i32 %1187 to float, !dbg !6
  %1189 = select i1 %.not40, float %1172, float %1188, !dbg !6
  %1190 = select i1 %.not40, float %1188, float %1178, !dbg !6
  %1191 = extractelement <8 x float> %1162, i64 4, !dbg !6
  %1192 = extractelement <8 x float> %1162, i64 5, !dbg !6
  %1193 = extractelement <8 x float> %1162, i64 6, !dbg !6
  %1194 = extractelement <8 x float> %1162, i64 7, !dbg !6
  %1195 = select i1 %.not, float %1192, float %1191, !dbg !6
  %1196 = bitcast float %1195 to i32, !dbg !6
  %1197 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1196), !dbg !6
  %1198 = bitcast i32 %1197 to float, !dbg !6
  %1199 = select i1 %.not, float %1191, float %1198, !dbg !6
  %1200 = select i1 %.not, float %1198, float %1192, !dbg !6
  %1201 = select i1 %.not, float %1194, float %1193, !dbg !6
  %1202 = bitcast float %1201 to i32, !dbg !6
  %1203 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1202), !dbg !6
  %1204 = bitcast i32 %1203 to float, !dbg !6
  %1205 = select i1 %.not, float %1193, float %1204, !dbg !6
  %1206 = select i1 %.not, float %1204, float %1194, !dbg !6
  %1207 = select i1 %.not40, float %1205, float %1199, !dbg !6
  %1208 = bitcast float %1207 to i32, !dbg !6
  %1209 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1208), !dbg !6
  %1210 = bitcast i32 %1209 to float, !dbg !6
  %1211 = select i1 %.not40, float %1199, float %1210, !dbg !6
  %1212 = select i1 %.not40, float %1210, float %1205, !dbg !6
  %1213 = select i1 %.not40, float %1206, float %1200, !dbg !6
  %1214 = bitcast float %1213 to i32, !dbg !6
  %1215 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1214), !dbg !6
  %1216 = bitcast i32 %1215 to float, !dbg !6
  %1217 = select i1 %.not40, float %1200, float %1216, !dbg !6
  %1218 = select i1 %.not40, float %1216, float %1206, !dbg !6
  %1219 = extractvalue [2 x <8 x float>] %106, 0, !dbg !6
  %1220 = extractelement <8 x float> %1219, i64 0, !dbg !6
  %1221 = extractelement <8 x float> %1219, i64 1, !dbg !6
  %1222 = extractelement <8 x float> %1219, i64 2, !dbg !6
  %1223 = extractelement <8 x float> %1219, i64 3, !dbg !6
  %1224 = select i1 %.not, float %1221, float %1220, !dbg !6
  %1225 = bitcast float %1224 to i32, !dbg !6
  %1226 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1225), !dbg !6
  %1227 = bitcast i32 %1226 to float, !dbg !6
  %1228 = select i1 %.not, float %1220, float %1227, !dbg !6
  %1229 = select i1 %.not, float %1227, float %1221, !dbg !6
  %1230 = select i1 %.not, float %1223, float %1222, !dbg !6
  %1231 = bitcast float %1230 to i32, !dbg !6
  %1232 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1231), !dbg !6
  %1233 = bitcast i32 %1232 to float, !dbg !6
  %1234 = select i1 %.not, float %1222, float %1233, !dbg !6
  %1235 = select i1 %.not, float %1233, float %1223, !dbg !6
  %1236 = select i1 %.not40, float %1234, float %1228, !dbg !6
  %1237 = bitcast float %1236 to i32, !dbg !6
  %1238 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1237), !dbg !6
  %1239 = bitcast i32 %1238 to float, !dbg !6
  %1240 = select i1 %.not40, float %1228, float %1239, !dbg !6
  %1241 = select i1 %.not40, float %1239, float %1234, !dbg !6
  %1242 = select i1 %.not40, float %1235, float %1229, !dbg !6
  %1243 = bitcast float %1242 to i32, !dbg !6
  %1244 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1243), !dbg !6
  %1245 = bitcast i32 %1244 to float, !dbg !6
  %1246 = select i1 %.not40, float %1229, float %1245, !dbg !6
  %1247 = select i1 %.not40, float %1245, float %1235, !dbg !6
  %1248 = extractelement <8 x float> %1219, i64 4, !dbg !6
  %1249 = extractelement <8 x float> %1219, i64 5, !dbg !6
  %1250 = extractelement <8 x float> %1219, i64 6, !dbg !6
  %1251 = extractelement <8 x float> %1219, i64 7, !dbg !6
  %1252 = select i1 %.not, float %1249, float %1248, !dbg !6
  %1253 = bitcast float %1252 to i32, !dbg !6
  %1254 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1253), !dbg !6
  %1255 = bitcast i32 %1254 to float, !dbg !6
  %1256 = select i1 %.not, float %1248, float %1255, !dbg !6
  %1257 = select i1 %.not, float %1255, float %1249, !dbg !6
  %1258 = select i1 %.not, float %1251, float %1250, !dbg !6
  %1259 = bitcast float %1258 to i32, !dbg !6
  %1260 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1259), !dbg !6
  %1261 = bitcast i32 %1260 to float, !dbg !6
  %1262 = select i1 %.not, float %1250, float %1261, !dbg !6
  %1263 = select i1 %.not, float %1261, float %1251, !dbg !6
  %1264 = select i1 %.not40, float %1262, float %1256, !dbg !6
  %1265 = bitcast float %1264 to i32, !dbg !6
  %1266 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1265), !dbg !6
  %1267 = bitcast i32 %1266 to float, !dbg !6
  %1268 = select i1 %.not40, float %1256, float %1267, !dbg !6
  %1269 = select i1 %.not40, float %1267, float %1262, !dbg !6
  %1270 = select i1 %.not40, float %1263, float %1257, !dbg !6
  %1271 = bitcast float %1270 to i32, !dbg !6
  %1272 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1271), !dbg !6
  %1273 = bitcast i32 %1272 to float, !dbg !6
  %1274 = select i1 %.not40, float %1257, float %1273, !dbg !6
  %1275 = select i1 %.not40, float %1273, float %1263, !dbg !6
  %1276 = extractvalue [2 x <8 x float>] %106, 1, !dbg !6
  %1277 = extractelement <8 x float> %1276, i64 0, !dbg !6
  %1278 = extractelement <8 x float> %1276, i64 1, !dbg !6
  %1279 = extractelement <8 x float> %1276, i64 2, !dbg !6
  %1280 = extractelement <8 x float> %1276, i64 3, !dbg !6
  %1281 = select i1 %.not, float %1278, float %1277, !dbg !6
  %1282 = bitcast float %1281 to i32, !dbg !6
  %1283 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1282), !dbg !6
  %1284 = bitcast i32 %1283 to float, !dbg !6
  %1285 = select i1 %.not, float %1277, float %1284, !dbg !6
  %1286 = select i1 %.not, float %1284, float %1278, !dbg !6
  %1287 = select i1 %.not, float %1280, float %1279, !dbg !6
  %1288 = bitcast float %1287 to i32, !dbg !6
  %1289 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1288), !dbg !6
  %1290 = bitcast i32 %1289 to float, !dbg !6
  %1291 = select i1 %.not, float %1279, float %1290, !dbg !6
  %1292 = select i1 %.not, float %1290, float %1280, !dbg !6
  %1293 = select i1 %.not40, float %1291, float %1285, !dbg !6
  %1294 = bitcast float %1293 to i32, !dbg !6
  %1295 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1294), !dbg !6
  %1296 = bitcast i32 %1295 to float, !dbg !6
  %1297 = select i1 %.not40, float %1285, float %1296, !dbg !6
  %1298 = select i1 %.not40, float %1296, float %1291, !dbg !6
  %1299 = select i1 %.not40, float %1292, float %1286, !dbg !6
  %1300 = bitcast float %1299 to i32, !dbg !6
  %1301 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1300), !dbg !6
  %1302 = bitcast i32 %1301 to float, !dbg !6
  %1303 = select i1 %.not40, float %1286, float %1302, !dbg !6
  %1304 = select i1 %.not40, float %1302, float %1292, !dbg !6
  %1305 = extractelement <8 x float> %1276, i64 4, !dbg !6
  %1306 = extractelement <8 x float> %1276, i64 5, !dbg !6
  %1307 = extractelement <8 x float> %1276, i64 6, !dbg !6
  %1308 = extractelement <8 x float> %1276, i64 7, !dbg !6
  %1309 = select i1 %.not, float %1306, float %1305, !dbg !6
  %1310 = bitcast float %1309 to i32, !dbg !6
  %1311 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1310), !dbg !6
  %1312 = bitcast i32 %1311 to float, !dbg !6
  %1313 = select i1 %.not, float %1305, float %1312, !dbg !6
  %1314 = select i1 %.not, float %1312, float %1306, !dbg !6
  %1315 = select i1 %.not, float %1308, float %1307, !dbg !6
  %1316 = bitcast float %1315 to i32, !dbg !6
  %1317 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1316), !dbg !6
  %1318 = bitcast i32 %1317 to float, !dbg !6
  %1319 = select i1 %.not, float %1307, float %1318, !dbg !6
  %1320 = select i1 %.not, float %1318, float %1308, !dbg !6
  %1321 = select i1 %.not40, float %1319, float %1313, !dbg !6
  %1322 = bitcast float %1321 to i32, !dbg !6
  %1323 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1322), !dbg !6
  %1324 = bitcast i32 %1323 to float, !dbg !6
  %1325 = select i1 %.not40, float %1313, float %1324, !dbg !6
  %1326 = select i1 %.not40, float %1324, float %1319, !dbg !6
  %1327 = select i1 %.not40, float %1320, float %1314, !dbg !6
  %1328 = bitcast float %1327 to i32, !dbg !6
  %1329 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1328), !dbg !6
  %1330 = bitcast i32 %1329 to float, !dbg !6
  %1331 = select i1 %.not40, float %1314, float %1330, !dbg !6
  %1332 = select i1 %.not40, float %1330, float %1320, !dbg !6
  %1333 = extractvalue [2 x <8 x float>] %105, 0, !dbg !6
  %1334 = extractelement <8 x float> %1333, i64 0, !dbg !6
  %1335 = extractelement <8 x float> %1333, i64 1, !dbg !6
  %1336 = extractelement <8 x float> %1333, i64 2, !dbg !6
  %1337 = extractelement <8 x float> %1333, i64 3, !dbg !6
  %1338 = select i1 %.not, float %1335, float %1334, !dbg !6
  %1339 = bitcast float %1338 to i32, !dbg !6
  %1340 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1339), !dbg !6
  %1341 = bitcast i32 %1340 to float, !dbg !6
  %1342 = select i1 %.not, float %1334, float %1341, !dbg !6
  %1343 = select i1 %.not, float %1341, float %1335, !dbg !6
  %1344 = select i1 %.not, float %1337, float %1336, !dbg !6
  %1345 = bitcast float %1344 to i32, !dbg !6
  %1346 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1345), !dbg !6
  %1347 = bitcast i32 %1346 to float, !dbg !6
  %1348 = select i1 %.not, float %1336, float %1347, !dbg !6
  %1349 = select i1 %.not, float %1347, float %1337, !dbg !6
  %1350 = select i1 %.not40, float %1348, float %1342, !dbg !6
  %1351 = bitcast float %1350 to i32, !dbg !6
  %1352 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1351), !dbg !6
  %1353 = bitcast i32 %1352 to float, !dbg !6
  %1354 = select i1 %.not40, float %1342, float %1353, !dbg !6
  %1355 = select i1 %.not40, float %1353, float %1348, !dbg !6
  %1356 = select i1 %.not40, float %1349, float %1343, !dbg !6
  %1357 = bitcast float %1356 to i32, !dbg !6
  %1358 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1357), !dbg !6
  %1359 = bitcast i32 %1358 to float, !dbg !6
  %1360 = select i1 %.not40, float %1343, float %1359, !dbg !6
  %1361 = select i1 %.not40, float %1359, float %1349, !dbg !6
  %1362 = extractelement <8 x float> %1333, i64 4, !dbg !6
  %1363 = extractelement <8 x float> %1333, i64 5, !dbg !6
  %1364 = extractelement <8 x float> %1333, i64 6, !dbg !6
  %1365 = extractelement <8 x float> %1333, i64 7, !dbg !6
  %1366 = select i1 %.not, float %1363, float %1362, !dbg !6
  %1367 = bitcast float %1366 to i32, !dbg !6
  %1368 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1367), !dbg !6
  %1369 = bitcast i32 %1368 to float, !dbg !6
  %1370 = select i1 %.not, float %1362, float %1369, !dbg !6
  %1371 = select i1 %.not, float %1369, float %1363, !dbg !6
  %1372 = select i1 %.not, float %1365, float %1364, !dbg !6
  %1373 = bitcast float %1372 to i32, !dbg !6
  %1374 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1373), !dbg !6
  %1375 = bitcast i32 %1374 to float, !dbg !6
  %1376 = select i1 %.not, float %1364, float %1375, !dbg !6
  %1377 = select i1 %.not, float %1375, float %1365, !dbg !6
  %1378 = select i1 %.not40, float %1376, float %1370, !dbg !6
  %1379 = bitcast float %1378 to i32, !dbg !6
  %1380 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1379), !dbg !6
  %1381 = bitcast i32 %1380 to float, !dbg !6
  %1382 = select i1 %.not40, float %1370, float %1381, !dbg !6
  %1383 = select i1 %.not40, float %1381, float %1376, !dbg !6
  %1384 = select i1 %.not40, float %1377, float %1371, !dbg !6
  %1385 = bitcast float %1384 to i32, !dbg !6
  %1386 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1385), !dbg !6
  %1387 = bitcast i32 %1386 to float, !dbg !6
  %1388 = select i1 %.not40, float %1371, float %1387, !dbg !6
  %1389 = select i1 %.not40, float %1387, float %1377, !dbg !6
  %1390 = extractvalue [2 x <8 x float>] %105, 1, !dbg !6
  %1391 = extractelement <8 x float> %1390, i64 0, !dbg !6
  %1392 = extractelement <8 x float> %1390, i64 1, !dbg !6
  %1393 = extractelement <8 x float> %1390, i64 2, !dbg !6
  %1394 = extractelement <8 x float> %1390, i64 3, !dbg !6
  %1395 = select i1 %.not, float %1392, float %1391, !dbg !6
  %1396 = bitcast float %1395 to i32, !dbg !6
  %1397 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1396), !dbg !6
  %1398 = bitcast i32 %1397 to float, !dbg !6
  %1399 = select i1 %.not, float %1391, float %1398, !dbg !6
  %1400 = select i1 %.not, float %1398, float %1392, !dbg !6
  %1401 = select i1 %.not, float %1394, float %1393, !dbg !6
  %1402 = bitcast float %1401 to i32, !dbg !6
  %1403 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1402), !dbg !6
  %1404 = bitcast i32 %1403 to float, !dbg !6
  %1405 = select i1 %.not, float %1393, float %1404, !dbg !6
  %1406 = select i1 %.not, float %1404, float %1394, !dbg !6
  %1407 = select i1 %.not40, float %1405, float %1399, !dbg !6
  %1408 = bitcast float %1407 to i32, !dbg !6
  %1409 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1408), !dbg !6
  %1410 = bitcast i32 %1409 to float, !dbg !6
  %1411 = select i1 %.not40, float %1399, float %1410, !dbg !6
  %1412 = select i1 %.not40, float %1410, float %1405, !dbg !6
  %1413 = select i1 %.not40, float %1406, float %1400, !dbg !6
  %1414 = bitcast float %1413 to i32, !dbg !6
  %1415 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1414), !dbg !6
  %1416 = bitcast i32 %1415 to float, !dbg !6
  %1417 = select i1 %.not40, float %1400, float %1416, !dbg !6
  %1418 = select i1 %.not40, float %1416, float %1406, !dbg !6
  %1419 = extractelement <8 x float> %1390, i64 4, !dbg !6
  %1420 = extractelement <8 x float> %1390, i64 5, !dbg !6
  %1421 = extractelement <8 x float> %1390, i64 6, !dbg !6
  %1422 = extractelement <8 x float> %1390, i64 7, !dbg !6
  %1423 = select i1 %.not, float %1420, float %1419, !dbg !6
  %1424 = bitcast float %1423 to i32, !dbg !6
  %1425 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1424), !dbg !6
  %1426 = bitcast i32 %1425 to float, !dbg !6
  %1427 = select i1 %.not, float %1419, float %1426, !dbg !6
  %1428 = select i1 %.not, float %1426, float %1420, !dbg !6
  %1429 = select i1 %.not, float %1422, float %1421, !dbg !6
  %1430 = bitcast float %1429 to i32, !dbg !6
  %1431 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1430), !dbg !6
  %1432 = bitcast i32 %1431 to float, !dbg !6
  %1433 = select i1 %.not, float %1421, float %1432, !dbg !6
  %1434 = select i1 %.not, float %1432, float %1422, !dbg !6
  %1435 = select i1 %.not40, float %1433, float %1427, !dbg !6
  %1436 = bitcast float %1435 to i32, !dbg !6
  %1437 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1436), !dbg !6
  %1438 = bitcast i32 %1437 to float, !dbg !6
  %1439 = select i1 %.not40, float %1427, float %1438, !dbg !6
  %1440 = select i1 %.not40, float %1438, float %1433, !dbg !6
  %1441 = select i1 %.not40, float %1434, float %1428, !dbg !6
  %1442 = bitcast float %1441 to i32, !dbg !6
  %1443 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1442), !dbg !6
  %1444 = bitcast i32 %1443 to float, !dbg !6
  %1445 = select i1 %.not40, float %1428, float %1444, !dbg !6
  %1446 = select i1 %.not40, float %1444, float %1434, !dbg !6
  %1447 = extractvalue [2 x <8 x float>] %102, 0, !dbg !6
  %1448 = extractelement <8 x float> %1447, i64 0, !dbg !6
  %1449 = extractelement <8 x float> %1447, i64 1, !dbg !6
  %1450 = extractelement <8 x float> %1447, i64 2, !dbg !6
  %1451 = extractelement <8 x float> %1447, i64 3, !dbg !6
  %1452 = select i1 %.not, float %1449, float %1448, !dbg !6
  %1453 = bitcast float %1452 to i32, !dbg !6
  %1454 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1453), !dbg !6
  %1455 = bitcast i32 %1454 to float, !dbg !6
  %1456 = select i1 %.not, float %1448, float %1455, !dbg !6
  %1457 = select i1 %.not, float %1455, float %1449, !dbg !6
  %1458 = select i1 %.not, float %1451, float %1450, !dbg !6
  %1459 = bitcast float %1458 to i32, !dbg !6
  %1460 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1459), !dbg !6
  %1461 = bitcast i32 %1460 to float, !dbg !6
  %1462 = select i1 %.not, float %1450, float %1461, !dbg !6
  %1463 = select i1 %.not, float %1461, float %1451, !dbg !6
  %1464 = select i1 %.not40, float %1462, float %1456, !dbg !6
  %1465 = bitcast float %1464 to i32, !dbg !6
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1465), !dbg !6
  %1467 = select i1 %.not40, float %1463, float %1457, !dbg !6
  %1468 = bitcast float %1467 to i32, !dbg !6
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1468), !dbg !6
  %1470 = extractelement <8 x float> %1447, i64 4, !dbg !6
  %1471 = extractelement <8 x float> %1447, i64 5, !dbg !6
  %1472 = extractelement <8 x float> %1447, i64 6, !dbg !6
  %1473 = extractelement <8 x float> %1447, i64 7, !dbg !6
  %1474 = select i1 %.not, float %1471, float %1470, !dbg !6
  %1475 = bitcast float %1474 to i32, !dbg !6
  %1476 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1475), !dbg !6
  %1477 = bitcast i32 %1476 to float, !dbg !6
  %1478 = select i1 %.not, float %1470, float %1477, !dbg !6
  %1479 = select i1 %.not, float %1477, float %1471, !dbg !6
  %1480 = select i1 %.not, float %1473, float %1472, !dbg !6
  %1481 = bitcast float %1480 to i32, !dbg !6
  %1482 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1481), !dbg !6
  %1483 = bitcast i32 %1482 to float, !dbg !6
  %1484 = select i1 %.not, float %1472, float %1483, !dbg !6
  %1485 = select i1 %.not, float %1483, float %1473, !dbg !6
  %1486 = select i1 %.not40, float %1484, float %1478, !dbg !6
  %1487 = bitcast float %1486 to i32, !dbg !6
  %1488 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1487), !dbg !6
  %1489 = select i1 %.not40, float %1485, float %1479, !dbg !6
  %1490 = bitcast float %1489 to i32, !dbg !6
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1490), !dbg !6
  %1492 = extractvalue [2 x <8 x float>] %102, 1, !dbg !6
  %1493 = extractelement <8 x float> %1492, i64 0, !dbg !6
  %1494 = extractelement <8 x float> %1492, i64 1, !dbg !6
  %1495 = extractelement <8 x float> %1492, i64 2, !dbg !6
  %1496 = extractelement <8 x float> %1492, i64 3, !dbg !6
  %1497 = select i1 %.not, float %1494, float %1493, !dbg !6
  %1498 = bitcast float %1497 to i32, !dbg !6
  %1499 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1498), !dbg !6
  %1500 = bitcast i32 %1499 to float, !dbg !6
  %1501 = select i1 %.not, float %1493, float %1500, !dbg !6
  %1502 = select i1 %.not, float %1500, float %1494, !dbg !6
  %1503 = select i1 %.not, float %1496, float %1495, !dbg !6
  %1504 = bitcast float %1503 to i32, !dbg !6
  %1505 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1504), !dbg !6
  %1506 = bitcast i32 %1505 to float, !dbg !6
  %1507 = select i1 %.not, float %1495, float %1506, !dbg !6
  %1508 = select i1 %.not, float %1506, float %1496, !dbg !6
  %1509 = select i1 %.not40, float %1507, float %1501, !dbg !6
  %1510 = bitcast float %1509 to i32, !dbg !6
  %1511 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1510), !dbg !6
  %1512 = select i1 %.not40, float %1508, float %1502, !dbg !6
  %1513 = bitcast float %1512 to i32, !dbg !6
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1513), !dbg !6
  %1515 = extractelement <8 x float> %1492, i64 4, !dbg !6
  %1516 = extractelement <8 x float> %1492, i64 5, !dbg !6
  %1517 = extractelement <8 x float> %1492, i64 6, !dbg !6
  %1518 = extractelement <8 x float> %1492, i64 7, !dbg !6
  %1519 = select i1 %.not, float %1516, float %1515, !dbg !6
  %1520 = bitcast float %1519 to i32, !dbg !6
  %1521 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1520), !dbg !6
  %1522 = bitcast i32 %1521 to float, !dbg !6
  %1523 = select i1 %.not, float %1515, float %1522, !dbg !6
  %1524 = select i1 %.not, float %1522, float %1516, !dbg !6
  %1525 = select i1 %.not, float %1518, float %1517, !dbg !6
  %1526 = bitcast float %1525 to i32, !dbg !6
  %1527 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1526), !dbg !6
  %1528 = bitcast i32 %1527 to float, !dbg !6
  %1529 = select i1 %.not, float %1517, float %1528, !dbg !6
  %1530 = select i1 %.not, float %1528, float %1518, !dbg !6
  %1531 = select i1 %.not40, float %1529, float %1523, !dbg !6
  %1532 = bitcast float %1531 to i32, !dbg !6
  %1533 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1532), !dbg !6
  %1534 = select i1 %.not40, float %1530, float %1524, !dbg !6
  %1535 = bitcast float %1534 to i32, !dbg !6
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1535), !dbg !6
  %1537 = insertelement <4 x float> poison, float %1102, i64 0, !dbg !17
  %1538 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1537, !dbg !17
  %1539 = shufflevector <4 x float> %1538, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1540 = insertelement <4 x float> poison, float %1092, i64 0, !dbg !17
  %1541 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1540, !dbg !17
  %1542 = shufflevector <4 x float> %1541, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1543 = bitcast i32 %1533 to float, !dbg !6
  %1544 = select i1 %.not40, float %1523, float %1543, !dbg !6
  %1545 = insertelement <4 x float> poison, float %1544, i64 0, !dbg !17
  %1546 = bitcast i32 %1536 to float, !dbg !6
  %1547 = select i1 %.not40, float %1524, float %1546, !dbg !6
  %1548 = insertelement <4 x float> %1545, float %1547, i64 1, !dbg !17
  %1549 = select i1 %.not40, float %1543, float %1529, !dbg !6
  %1550 = insertelement <4 x float> %1548, float %1549, i64 2, !dbg !17
  %1551 = select i1 %.not40, float %1546, float %1530, !dbg !6
  %1552 = insertelement <4 x float> %1550, float %1551, i64 3, !dbg !17
  %1553 = fmul <4 x float> %1552, %1542, !dbg !17
  %1554 = fptrunc <4 x float> %1553 to <4 x half>, !dbg !18
  %1555 = extractelement <4 x half> %1554, i64 3, !dbg !18
  %1556 = extractelement <4 x half> %1554, i64 2, !dbg !18
  %1557 = extractelement <4 x half> %1554, i64 1, !dbg !18
  %1558 = extractelement <4 x half> %1554, i64 0, !dbg !18
  %1559 = bitcast i32 %1511 to float, !dbg !6
  %1560 = select i1 %.not40, float %1501, float %1559, !dbg !6
  %1561 = insertelement <4 x float> poison, float %1560, i64 0, !dbg !17
  %1562 = bitcast i32 %1514 to float, !dbg !6
  %1563 = select i1 %.not40, float %1502, float %1562, !dbg !6
  %1564 = insertelement <4 x float> %1561, float %1563, i64 1, !dbg !17
  %1565 = select i1 %.not40, float %1559, float %1507, !dbg !6
  %1566 = insertelement <4 x float> %1564, float %1565, i64 2, !dbg !17
  %1567 = select i1 %.not40, float %1562, float %1508, !dbg !6
  %1568 = insertelement <4 x float> %1566, float %1567, i64 3, !dbg !17
  %1569 = fmul <4 x float> %1568, %1542, !dbg !17
  %1570 = fptrunc <4 x float> %1569 to <4 x half>, !dbg !18
  %1571 = extractelement <4 x half> %1570, i64 3, !dbg !18
  %1572 = extractelement <4 x half> %1570, i64 2, !dbg !18
  %1573 = extractelement <4 x half> %1570, i64 1, !dbg !18
  %1574 = extractelement <4 x half> %1570, i64 0, !dbg !18
  %1575 = bitcast i32 %1488 to float, !dbg !6
  %1576 = select i1 %.not40, float %1478, float %1575, !dbg !6
  %1577 = insertelement <4 x float> poison, float %1576, i64 0, !dbg !17
  %1578 = bitcast i32 %1491 to float, !dbg !6
  %1579 = select i1 %.not40, float %1479, float %1578, !dbg !6
  %1580 = insertelement <4 x float> %1577, float %1579, i64 1, !dbg !17
  %1581 = select i1 %.not40, float %1575, float %1484, !dbg !6
  %1582 = insertelement <4 x float> %1580, float %1581, i64 2, !dbg !17
  %1583 = select i1 %.not40, float %1578, float %1485, !dbg !6
  %1584 = insertelement <4 x float> %1582, float %1583, i64 3, !dbg !17
  %1585 = fmul <4 x float> %1584, %1539, !dbg !17
  %1586 = fptrunc <4 x float> %1585 to <4 x half>, !dbg !18
  %1587 = extractelement <4 x half> %1586, i64 3, !dbg !18
  %1588 = extractelement <4 x half> %1586, i64 2, !dbg !18
  %1589 = extractelement <4 x half> %1586, i64 1, !dbg !18
  %1590 = extractelement <4 x half> %1586, i64 0, !dbg !18
  %1591 = bitcast i32 %1466 to float, !dbg !6
  %1592 = select i1 %.not40, float %1456, float %1591, !dbg !6
  %1593 = insertelement <4 x float> poison, float %1592, i64 0, !dbg !17
  %1594 = bitcast i32 %1469 to float, !dbg !6
  %1595 = select i1 %.not40, float %1457, float %1594, !dbg !6
  %1596 = insertelement <4 x float> %1593, float %1595, i64 1, !dbg !17
  %1597 = select i1 %.not40, float %1591, float %1462, !dbg !6
  %1598 = insertelement <4 x float> %1596, float %1597, i64 2, !dbg !17
  %1599 = select i1 %.not40, float %1594, float %1463, !dbg !6
  %1600 = insertelement <4 x float> %1598, float %1599, i64 3, !dbg !17
  %1601 = fmul <4 x float> %1600, %1539, !dbg !17
  %1602 = fptrunc <4 x float> %1601 to <4 x half>, !dbg !18
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !18
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !18
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !18
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !18
  %address.thread837 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local838 = zext nneg i32 %address.thread837 to i64, !dbg !9
  %address.local839 = add nuw i64 %32, %address.local838, !dbg !19
  %1607 = lshr i64 %address.local838, 2, !dbg !19
  %address.local841 = and i64 %1607, 48, !dbg !19
  %address.local842 = add i64 %address.local839, %address.local841, !dbg !19
  %address.local843 = and i64 %address.local838, 240, !dbg !19
  %address.local844 = sub i64 %address.local842, %address.local843, !dbg !19
  %address.local845 = add i64 %address.local844, %33, !dbg !19
  %address.local846 = shl nuw nsw i64 %address.local845, 7, !dbg !19
  %1608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local848 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1608, i64 %address.local846, !dbg !19
  %address.local850 = and i64 %1607, 12, !dbg !12
  %address.local851 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local850, !dbg !19
  store half %1606, ptr addrspace(1) %address.local851, align 2, !dbg !19
  %address.local853 = or disjoint i64 %address.local850, 1, !dbg !19
  %address.local854 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local853, !dbg !19
  store half %1605, ptr addrspace(1) %address.local854, align 2, !dbg !19
  %address.local856 = or disjoint i64 %address.local850, 2, !dbg !19
  %address.local857 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local856, !dbg !19
  store half %1604, ptr addrspace(1) %address.local857, align 2, !dbg !19
  %address.local859 = or disjoint i64 %address.local850, 3, !dbg !19
  %address.local860 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local859, !dbg !19
  store half %1603, ptr addrspace(1) %address.local860, align 2, !dbg !19
  %address.local862 = or disjoint i64 %address.local850, 16, !dbg !19
  %address.local863 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local862, !dbg !19
  store half %1590, ptr addrspace(1) %address.local863, align 2, !dbg !19
  %address.local865 = or disjoint i64 %address.local850, 17, !dbg !19
  %address.local866 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local865, !dbg !19
  store half %1589, ptr addrspace(1) %address.local866, align 2, !dbg !19
  %address.local868 = or disjoint i64 %address.local850, 18, !dbg !19
  %address.local869 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local868, !dbg !19
  store half %1588, ptr addrspace(1) %address.local869, align 2, !dbg !19
  %address.local871 = or disjoint i64 %address.local850, 19, !dbg !19
  %address.local872 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local871, !dbg !19
  store half %1587, ptr addrspace(1) %address.local872, align 2, !dbg !19
  %address.local876 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local848, i64 16384, !dbg !19
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local850, !dbg !19
  store half %1574, ptr addrspace(1) %address.local877, align 2, !dbg !19
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local853, !dbg !19
  store half %1573, ptr addrspace(1) %address.local879, align 2, !dbg !19
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local856, !dbg !19
  store half %1572, ptr addrspace(1) %address.local881, align 2, !dbg !19
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local859, !dbg !19
  store half %1571, ptr addrspace(1) %address.local883, align 2, !dbg !19
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local862, !dbg !19
  store half %1558, ptr addrspace(1) %address.local885, align 2, !dbg !19
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local865, !dbg !19
  store half %1557, ptr addrspace(1) %address.local887, align 2, !dbg !19
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local868, !dbg !19
  store half %1556, ptr addrspace(1) %address.local889, align 2, !dbg !19
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local871, !dbg !19
  store half %1555, ptr addrspace(1) %address.local891, align 2, !dbg !19
  %1609 = insertelement <4 x float> poison, float %1439, i64 0, !dbg !17
  %1610 = insertelement <4 x float> %1609, float %1445, i64 1, !dbg !17
  %1611 = insertelement <4 x float> %1610, float %1440, i64 2, !dbg !17
  %1612 = insertelement <4 x float> %1611, float %1446, i64 3, !dbg !17
  %1613 = fmul <4 x float> %1612, %1542, !dbg !17
  %1614 = fptrunc <4 x float> %1613 to <4 x half>, !dbg !18
  %1615 = extractelement <4 x half> %1614, i64 3, !dbg !18
  %1616 = extractelement <4 x half> %1614, i64 2, !dbg !18
  %1617 = extractelement <4 x half> %1614, i64 1, !dbg !18
  %1618 = extractelement <4 x half> %1614, i64 0, !dbg !18
  %1619 = insertelement <4 x float> poison, float %1411, i64 0, !dbg !17
  %1620 = insertelement <4 x float> %1619, float %1417, i64 1, !dbg !17
  %1621 = insertelement <4 x float> %1620, float %1412, i64 2, !dbg !17
  %1622 = insertelement <4 x float> %1621, float %1418, i64 3, !dbg !17
  %1623 = fmul <4 x float> %1622, %1542, !dbg !17
  %1624 = fptrunc <4 x float> %1623 to <4 x half>, !dbg !18
  %1625 = extractelement <4 x half> %1624, i64 3, !dbg !18
  %1626 = extractelement <4 x half> %1624, i64 2, !dbg !18
  %1627 = extractelement <4 x half> %1624, i64 1, !dbg !18
  %1628 = extractelement <4 x half> %1624, i64 0, !dbg !18
  %1629 = insertelement <4 x float> poison, float %1382, i64 0, !dbg !17
  %1630 = insertelement <4 x float> %1629, float %1388, i64 1, !dbg !17
  %1631 = insertelement <4 x float> %1630, float %1383, i64 2, !dbg !17
  %1632 = insertelement <4 x float> %1631, float %1389, i64 3, !dbg !17
  %1633 = fmul <4 x float> %1632, %1539, !dbg !17
  %1634 = fptrunc <4 x float> %1633 to <4 x half>, !dbg !18
  %1635 = extractelement <4 x half> %1634, i64 3, !dbg !18
  %1636 = extractelement <4 x half> %1634, i64 2, !dbg !18
  %1637 = extractelement <4 x half> %1634, i64 1, !dbg !18
  %1638 = extractelement <4 x half> %1634, i64 0, !dbg !18
  %1639 = insertelement <4 x float> poison, float %1354, i64 0, !dbg !17
  %1640 = insertelement <4 x float> %1639, float %1360, i64 1, !dbg !17
  %1641 = insertelement <4 x float> %1640, float %1355, i64 2, !dbg !17
  %1642 = insertelement <4 x float> %1641, float %1361, i64 3, !dbg !17
  %1643 = fmul <4 x float> %1642, %1539, !dbg !17
  %1644 = fptrunc <4 x float> %1643 to <4 x half>, !dbg !18
  %1645 = extractelement <4 x half> %1644, i64 3, !dbg !18
  %1646 = extractelement <4 x half> %1644, i64 2, !dbg !18
  %1647 = extractelement <4 x half> %1644, i64 1, !dbg !18
  %1648 = extractelement <4 x half> %1644, i64 0, !dbg !18
  %address.local906 = or disjoint i64 %address.local850, 32, !dbg !19
  %address.local907 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local906, !dbg !19
  store half %1648, ptr addrspace(1) %address.local907, align 2, !dbg !19
  %address.local909 = or disjoint i64 %address.local850, 33, !dbg !19
  %address.local910 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local909, !dbg !19
  store half %1647, ptr addrspace(1) %address.local910, align 2, !dbg !19
  %address.local912 = or disjoint i64 %address.local850, 34, !dbg !19
  %address.local913 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local912, !dbg !19
  store half %1646, ptr addrspace(1) %address.local913, align 2, !dbg !19
  %address.local915 = or disjoint i64 %address.local850, 35, !dbg !19
  %address.local916 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local915, !dbg !19
  store half %1645, ptr addrspace(1) %address.local916, align 2, !dbg !19
  %address.local918 = or disjoint i64 %address.local850, 48, !dbg !19
  %address.local919 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local918, !dbg !19
  store half %1638, ptr addrspace(1) %address.local919, align 2, !dbg !19
  %address.local921 = or disjoint i64 %address.local850, 49, !dbg !19
  %address.local922 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local921, !dbg !19
  store half %1637, ptr addrspace(1) %address.local922, align 2, !dbg !19
  %address.local924 = or disjoint i64 %address.local850, 50, !dbg !19
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local924, !dbg !19
  store half %1636, ptr addrspace(1) %address.local925, align 2, !dbg !19
  %address.local927 = or i64 %1607, 51, !dbg !19
  %address.local928 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local927, !dbg !19
  store half %1635, ptr addrspace(1) %address.local928, align 2, !dbg !19
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local906, !dbg !19
  store half %1628, ptr addrspace(1) %address.local933, align 2, !dbg !19
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local909, !dbg !19
  store half %1627, ptr addrspace(1) %address.local935, align 2, !dbg !19
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local912, !dbg !19
  store half %1626, ptr addrspace(1) %address.local937, align 2, !dbg !19
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local915, !dbg !19
  store half %1625, ptr addrspace(1) %address.local939, align 2, !dbg !19
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local918, !dbg !19
  store half %1618, ptr addrspace(1) %address.local941, align 2, !dbg !19
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local921, !dbg !19
  store half %1617, ptr addrspace(1) %address.local943, align 2, !dbg !19
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local924, !dbg !19
  store half %1616, ptr addrspace(1) %address.local945, align 2, !dbg !19
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local927, !dbg !19
  store half %1615, ptr addrspace(1) %address.local947, align 2, !dbg !19
  %1649 = insertelement <4 x float> poison, float %1325, i64 0, !dbg !17
  %1650 = insertelement <4 x float> %1649, float %1331, i64 1, !dbg !17
  %1651 = insertelement <4 x float> %1650, float %1326, i64 2, !dbg !17
  %1652 = insertelement <4 x float> %1651, float %1332, i64 3, !dbg !17
  %1653 = fmul <4 x float> %1652, %1542, !dbg !17
  %1654 = fptrunc <4 x float> %1653 to <4 x half>, !dbg !18
  %1655 = extractelement <4 x half> %1654, i64 3, !dbg !18
  %1656 = extractelement <4 x half> %1654, i64 2, !dbg !18
  %1657 = extractelement <4 x half> %1654, i64 1, !dbg !18
  %1658 = extractelement <4 x half> %1654, i64 0, !dbg !18
  %1659 = insertelement <4 x float> poison, float %1297, i64 0, !dbg !17
  %1660 = insertelement <4 x float> %1659, float %1303, i64 1, !dbg !17
  %1661 = insertelement <4 x float> %1660, float %1298, i64 2, !dbg !17
  %1662 = insertelement <4 x float> %1661, float %1304, i64 3, !dbg !17
  %1663 = fmul <4 x float> %1662, %1542, !dbg !17
  %1664 = fptrunc <4 x float> %1663 to <4 x half>, !dbg !18
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !18
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !18
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !18
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !18
  %1669 = insertelement <4 x float> poison, float %1268, i64 0, !dbg !17
  %1670 = insertelement <4 x float> %1669, float %1274, i64 1, !dbg !17
  %1671 = insertelement <4 x float> %1670, float %1269, i64 2, !dbg !17
  %1672 = insertelement <4 x float> %1671, float %1275, i64 3, !dbg !17
  %1673 = fmul <4 x float> %1672, %1539, !dbg !17
  %1674 = fptrunc <4 x float> %1673 to <4 x half>, !dbg !18
  %1675 = extractelement <4 x half> %1674, i64 3, !dbg !18
  %1676 = extractelement <4 x half> %1674, i64 2, !dbg !18
  %1677 = extractelement <4 x half> %1674, i64 1, !dbg !18
  %1678 = extractelement <4 x half> %1674, i64 0, !dbg !18
  %1679 = insertelement <4 x float> poison, float %1240, i64 0, !dbg !17
  %1680 = insertelement <4 x float> %1679, float %1246, i64 1, !dbg !17
  %1681 = insertelement <4 x float> %1680, float %1241, i64 2, !dbg !17
  %1682 = insertelement <4 x float> %1681, float %1247, i64 3, !dbg !17
  %1683 = fmul <4 x float> %1682, %1539, !dbg !17
  %1684 = fptrunc <4 x float> %1683 to <4 x half>, !dbg !18
  %1685 = extractelement <4 x half> %1684, i64 3, !dbg !18
  %1686 = extractelement <4 x half> %1684, i64 2, !dbg !18
  %1687 = extractelement <4 x half> %1684, i64 1, !dbg !18
  %1688 = extractelement <4 x half> %1684, i64 0, !dbg !18
  %address.local962 = or disjoint i64 %address.local850, 64, !dbg !19
  %address.local963 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local962, !dbg !19
  store half %1688, ptr addrspace(1) %address.local963, align 2, !dbg !19
  %address.local965 = or disjoint i64 %address.local850, 65, !dbg !19
  %address.local966 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local965, !dbg !19
  store half %1687, ptr addrspace(1) %address.local966, align 2, !dbg !19
  %address.local968 = or disjoint i64 %address.local850, 66, !dbg !19
  %address.local969 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local968, !dbg !19
  store half %1686, ptr addrspace(1) %address.local969, align 2, !dbg !19
  %address.local971 = or disjoint i64 %address.local850, 67, !dbg !19
  %address.local972 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local971, !dbg !19
  store half %1685, ptr addrspace(1) %address.local972, align 2, !dbg !19
  %address.local974 = or disjoint i64 %address.local850, 80, !dbg !19
  %address.local975 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local974, !dbg !19
  store half %1678, ptr addrspace(1) %address.local975, align 2, !dbg !19
  %address.local977 = or disjoint i64 %address.local850, 81, !dbg !19
  %address.local978 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local977, !dbg !19
  store half %1677, ptr addrspace(1) %address.local978, align 2, !dbg !19
  %address.local980 = or disjoint i64 %address.local850, 82, !dbg !19
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local980, !dbg !19
  store half %1676, ptr addrspace(1) %address.local981, align 2, !dbg !19
  %address.local983 = or disjoint i64 %address.local850, 83, !dbg !19
  %address.local984 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local983, !dbg !19
  store half %1675, ptr addrspace(1) %address.local984, align 2, !dbg !19
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local962, !dbg !19
  store half %1668, ptr addrspace(1) %address.local989, align 2, !dbg !19
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local965, !dbg !19
  store half %1667, ptr addrspace(1) %address.local991, align 2, !dbg !19
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local968, !dbg !19
  store half %1666, ptr addrspace(1) %address.local993, align 2, !dbg !19
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local971, !dbg !19
  store half %1665, ptr addrspace(1) %address.local995, align 2, !dbg !19
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local974, !dbg !19
  store half %1658, ptr addrspace(1) %address.local997, align 2, !dbg !19
  %address.local999 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local977, !dbg !19
  store half %1657, ptr addrspace(1) %address.local999, align 2, !dbg !19
  %address.local1001 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local980, !dbg !19
  store half %1656, ptr addrspace(1) %address.local1001, align 2, !dbg !19
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local983, !dbg !19
  store half %1655, ptr addrspace(1) %address.local1003, align 2, !dbg !19
  %1689 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !17
  %1690 = insertelement <4 x float> %1689, float %1217, i64 1, !dbg !17
  %1691 = insertelement <4 x float> %1690, float %1212, i64 2, !dbg !17
  %1692 = insertelement <4 x float> %1691, float %1218, i64 3, !dbg !17
  %1693 = fmul <4 x float> %1692, %1542, !dbg !17
  %1694 = fptrunc <4 x float> %1693 to <4 x half>, !dbg !18
  %1695 = extractelement <4 x half> %1694, i64 3, !dbg !18
  %1696 = extractelement <4 x half> %1694, i64 2, !dbg !18
  %1697 = extractelement <4 x half> %1694, i64 1, !dbg !18
  %1698 = extractelement <4 x half> %1694, i64 0, !dbg !18
  %1699 = insertelement <4 x float> poison, float %1183, i64 0, !dbg !17
  %1700 = insertelement <4 x float> %1699, float %1189, i64 1, !dbg !17
  %1701 = insertelement <4 x float> %1700, float %1184, i64 2, !dbg !17
  %1702 = insertelement <4 x float> %1701, float %1190, i64 3, !dbg !17
  %1703 = fmul <4 x float> %1702, %1542, !dbg !17
  %1704 = fptrunc <4 x float> %1703 to <4 x half>, !dbg !18
  %1705 = extractelement <4 x half> %1704, i64 3, !dbg !18
  %1706 = extractelement <4 x half> %1704, i64 2, !dbg !18
  %1707 = extractelement <4 x half> %1704, i64 1, !dbg !18
  %1708 = extractelement <4 x half> %1704, i64 0, !dbg !18
  %1709 = insertelement <4 x float> poison, float %1154, i64 0, !dbg !17
  %1710 = insertelement <4 x float> %1709, float %1160, i64 1, !dbg !17
  %1711 = insertelement <4 x float> %1710, float %1155, i64 2, !dbg !17
  %1712 = insertelement <4 x float> %1711, float %1161, i64 3, !dbg !17
  %1713 = fmul <4 x float> %1712, %1539, !dbg !17
  %1714 = fptrunc <4 x float> %1713 to <4 x half>, !dbg !18
  %1715 = extractelement <4 x half> %1714, i64 3, !dbg !18
  %1716 = extractelement <4 x half> %1714, i64 2, !dbg !18
  %1717 = extractelement <4 x half> %1714, i64 1, !dbg !18
  %1718 = extractelement <4 x half> %1714, i64 0, !dbg !18
  %1719 = insertelement <4 x float> poison, float %1126, i64 0, !dbg !17
  %1720 = insertelement <4 x float> %1719, float %1132, i64 1, !dbg !17
  %1721 = insertelement <4 x float> %1720, float %1127, i64 2, !dbg !17
  %1722 = insertelement <4 x float> %1721, float %1133, i64 3, !dbg !17
  %1723 = fmul <4 x float> %1722, %1539, !dbg !17
  %1724 = fptrunc <4 x float> %1723 to <4 x half>, !dbg !18
  %1725 = extractelement <4 x half> %1724, i64 3, !dbg !18
  %1726 = extractelement <4 x half> %1724, i64 2, !dbg !18
  %1727 = extractelement <4 x half> %1724, i64 1, !dbg !18
  %1728 = extractelement <4 x half> %1724, i64 0, !dbg !18
  %address.local1018 = or disjoint i64 %address.local850, 96, !dbg !19
  %address.local1019 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1018, !dbg !19
  store half %1728, ptr addrspace(1) %address.local1019, align 2, !dbg !19
  %address.local1021 = or disjoint i64 %address.local850, 97, !dbg !19
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1021, !dbg !19
  store half %1727, ptr addrspace(1) %address.local1022, align 2, !dbg !19
  %address.local1024 = or disjoint i64 %address.local850, 98, !dbg !19
  %address.local1025 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1024, !dbg !19
  store half %1726, ptr addrspace(1) %address.local1025, align 2, !dbg !19
  %address.local1027 = or disjoint i64 %address.local850, 99, !dbg !19
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1027, !dbg !19
  store half %1725, ptr addrspace(1) %address.local1028, align 2, !dbg !19
  %address.local1030 = or disjoint i64 %address.local850, 112, !dbg !19
  %address.local1031 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1030, !dbg !19
  store half %1718, ptr addrspace(1) %address.local1031, align 2, !dbg !19
  %address.local1033 = or disjoint i64 %address.local850, 113, !dbg !19
  %address.local1034 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1033, !dbg !19
  store half %1717, ptr addrspace(1) %address.local1034, align 2, !dbg !19
  %address.local1036 = or disjoint i64 %address.local850, 114, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1036, !dbg !19
  store half %1716, ptr addrspace(1) %address.local1037, align 2, !dbg !19
  %address.local1039 = or i64 %1607, 115, !dbg !19
  %address.local1040 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local848, i64 %address.local1039, !dbg !19
  store half %1715, ptr addrspace(1) %address.local1040, align 2, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1018, !dbg !19
  store half %1708, ptr addrspace(1) %address.local1045, align 2, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1021, !dbg !19
  store half %1707, ptr addrspace(1) %address.local1047, align 2, !dbg !19
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1024, !dbg !19
  store half %1706, ptr addrspace(1) %address.local1049, align 2, !dbg !19
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1027, !dbg !19
  store half %1705, ptr addrspace(1) %address.local1051, align 2, !dbg !19
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1030, !dbg !19
  store half %1698, ptr addrspace(1) %address.local1053, align 2, !dbg !19
  %address.local1055 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1033, !dbg !19
  store half %1697, ptr addrspace(1) %address.local1055, align 2, !dbg !19
  %address.local1057 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1036, !dbg !19
  store half %1696, ptr addrspace(1) %address.local1057, align 2, !dbg !19
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local876, i64 %address.local1039, !dbg !19
  store half %1695, ptr addrspace(1) %address.local1059, align 2, !dbg !19
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_bounds_lowering/noncausal")
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
