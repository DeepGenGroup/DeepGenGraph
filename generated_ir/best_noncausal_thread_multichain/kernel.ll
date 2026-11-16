; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_thread_multichain/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_thread_multichain/input.mlir"

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
  %address.local276 = zext nneg i32 %37 to i64, !dbg !9
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local276, !dbg !9
  %address.local278 = ptrtoint ptr addrspace(3) %address.local277 to i64, !dbg !9
  %address.local279 = trunc i64 %address.local278 to i32, !dbg !9
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local279), !dbg !9
  %address.local280 = lshr i32 %address.thread, 4, !dbg !9
  %address.local281 = and i32 %address.local280, 8, !dbg !9
  %address.local282 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local283 = and i32 %address.local282, 16256, !dbg !9
  %address.local284 = or disjoint i32 %address.local281, %address.local283, !dbg !9
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1250 = shl i64 %34, 8, !dbg !9
  %40 = add i64 %.idx, %.idx1250, !dbg !9
  %41 = shl nuw nsw i32 %address.local284, 1, !dbg !9
  %42 = trunc i64 %40 to i32, !dbg !9
  %dma.buffer.offset1248 = or disjoint i32 %41, %42, !dbg !9
  %43 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !9
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1249 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1248, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %address.local289 = or disjoint i64 %address.local276, 2048, !dbg !9
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local289, !dbg !9
  %address.local291 = ptrtoint ptr addrspace(3) %address.local290 to i64, !dbg !9
  %address.local292 = trunc i64 %address.local291 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local292), !dbg !9
  %address.local298 = or disjoint i32 %address.local284, 16, !dbg !9
  %47 = shl nuw nsw i32 %address.local298, 1, !dbg !9
  %dma.buffer.offset1246 = or disjoint i32 %47, %42, !dbg !9
  %48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1246, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %49 = shl nuw nsw i64 %31, 19, !dbg !9
  %50 = add i64 %49, %35, !dbg !9
  %address.local301 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local276, !dbg !9
  %address.local302 = ptrtoint ptr addrspace(3) %address.local301 to i64, !dbg !9
  %address.local303 = trunc i64 %address.local302 to i32, !dbg !9
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local303), !dbg !9
  %52 = trunc i64 %50 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local284, %52, !dbg !9
  %53 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1244 = or disjoint i32 %53, 64, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1244, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %address.local305 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local289, !dbg !9
  %address.local306 = ptrtoint ptr addrspace(3) %address.local305 to i64, !dbg !9
  %address.local307 = trunc i64 %address.local306 to i32, !dbg !9
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local307), !dbg !9
  %.tr1254 = or disjoint i32 %address.local298, %52, !dbg !9
  %56 = shl i32 %.tr1254, 1, !dbg !9
  %dma.buffer.offset1242 = or disjoint i32 %56, 64, !dbg !9
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1242, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %address.local309 = lshr i32 %address.thread, 2, !dbg !12
  %address.local310 = and i32 %address.local309, 4, !dbg !12
  %address.local311 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local312 = and i32 %address.local311, 384, !dbg !12
  %address.local313 = or disjoint i32 %address.local310, %address.local312, !dbg !12
  %address.local315 = and i32 %36, 120, !dbg !12
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !12
  %address.local317 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local318 = and i32 %address.local317, 1024, !dbg !12
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !12
  %address.local320 = zext nneg i32 %address.local319 to i64, !dbg !12
  %address.local321 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local320, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local321, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep1071 = getelementptr i8, ptr addrspace(3) %address.local321, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1071, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep1072 = getelementptr i8, ptr addrspace(3) %address.local321, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1072, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1240 = or disjoint i32 %53, 128, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1240, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %dma.buffer.offset1238 = or disjoint i32 %56, 128, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1238, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %lds.const.gep1073 = getelementptr i8, ptr addrspace(3) %address.local321, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1073, align 8, !dbg !12
  %lds.const.gep1074 = getelementptr i8, ptr addrspace(3) %address.local321, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1074, align 8, !dbg !12
  %lds.const.gep1075 = getelementptr i8, ptr addrspace(3) %address.local321, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1075, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1076 = getelementptr i8, ptr addrspace(3) %address.local321, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1076, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1236 = or disjoint i32 %53, 192, !dbg !9
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1236, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %dma.buffer.offset1234 = or disjoint i32 %56, 192, !dbg !9
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1234, <4 x i32> %dma.buffer.rsrc1249) #10, !dbg !9
  %78 = load <4 x half>, ptr addrspace(3) %address.local321, align 8, !dbg !12
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1071, align 8, !dbg !12
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1072, align 8, !dbg !12
  %83 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !12
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !12
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1073, align 8, !dbg !12
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1074, align 8, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1075, align 8, !dbg !12
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1076, align 8, !dbg !12
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
  %k0.prime.1917 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.1918 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.1919 = or disjoint i64 %k0.prime.1917, %k0.prime.1918, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1356 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1357 = shl nuw nsw i64 %k0.prime.kdma.source1356, 9, !dbg !13
  %k0.prime.kdma.source1358 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !13
  %k0.prime.kdma.source1359 = add nuw nsw i64 %k0.prime.kdma.source1357, %k0.prime.kdma.source1358, !dbg !13
  %k0.prime.kdma.source1360 = lshr i64 %k0.prime.kdma.source, 5, !dbg !13
  %k0.prime.kdma.source1361 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1362 = and i64 %k0.prime.kdma.source1361, 1, !dbg !13
  %k0.prime.kdma.source1363 = and i64 %k0.prime.kdma.source1356, %k0.prime.kdma.source1362, !dbg !13
  %k0.prime.kdma.source1364 = or disjoint i64 %k0.prime.kdma.source1360, %k0.prime.kdma.source1363, !dbg !13
  %k0.prime.kdma.source1365 = shl nuw nsw i64 %k0.prime.kdma.source1364, 10, !dbg !13
  %k0.prime.kdma.source1366 = sub nsw i64 %k0.prime.kdma.source1359, %k0.prime.kdma.source1365, !dbg !13
  %k0.prime.kdma.source1367 = and i64 %k0.prime.kdma.source1366, 16256, !dbg !13
  %k0.prime.kdma.source1368 = lshr i32 %kdma.tid, 4, !dbg !13
  %k0.prime.kdma.source1369 = and i32 %k0.prime.kdma.source1368, 8, !dbg !13
  %k0.prime.kdma.source1370 = zext nneg i32 %k0.prime.kdma.source1369 to i64, !dbg !13
  %k0.prime.kdma.source1371 = or disjoint i64 %k0.prime.kdma.source1367, %k0.prime.kdma.source1370, !dbg !13
  %k0.prime.kdma.source1372 = or disjoint i64 %k0.prime.1919, %k0.prime.kdma.source1371, !dbg !13
  %k0.prime.kdma.source1373 = trunc nuw i64 %k0.prime.kdma.source1372 to i32, !dbg !13
  %k0.prime.kdma.source1374 = shl nuw nsw i32 %k0.prime.kdma.source1373, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1374, %kdma.intra, !dbg !13
  %k0.prime.1920 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1921 = insertelement <2 x i64> poison, i64 %k0.prime.1920, i64 0, !dbg !13
  %k0.prime.1922 = bitcast <2 x i64> %k0.prime.1921 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1233 = shufflevector <4 x i32> %k0.prime.1922, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1923 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1396 = add i32 %38, 512, !dbg !13
  %k0.prime.kdma.source1375 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1376 = lshr i64 %k0.prime.kdma.source1375, 4, !dbg !13
  %k0.prime.kdma.source1377 = shl nuw nsw i64 %k0.prime.kdma.source1376, 9, !dbg !13
  %k0.prime.kdma.source1378 = shl nuw nsw i64 %k0.prime.kdma.source1375, 7, !dbg !13
  %k0.prime.kdma.source1379 = add nuw nsw i64 %k0.prime.kdma.source1377, %k0.prime.kdma.source1378, !dbg !13
  %k0.prime.kdma.source1380 = lshr i64 %k0.prime.kdma.source1375, 5, !dbg !13
  %k0.prime.kdma.source1381 = lshr i64 %k0.prime.kdma.source1375, 2, !dbg !13
  %k0.prime.kdma.source1382 = and i64 %k0.prime.kdma.source1381, 1, !dbg !13
  %k0.prime.kdma.source1383 = and i64 %k0.prime.kdma.source1376, %k0.prime.kdma.source1382, !dbg !13
  %k0.prime.kdma.source1384 = add nuw nsw i64 %k0.prime.kdma.source1380, %k0.prime.kdma.source1383, !dbg !13
  %k0.prime.kdma.source1385 = shl nuw nsw i64 %k0.prime.kdma.source1384, 10, !dbg !13
  %k0.prime.kdma.source1386 = sub nsw i64 %k0.prime.kdma.source1379, %k0.prime.kdma.source1385, !dbg !13
  %k0.prime.kdma.source1387 = and i64 %k0.prime.kdma.source1386, 16256, !dbg !13
  %k0.prime.kdma.source1391 = or disjoint i64 %k0.prime.kdma.source1387, %k0.prime.kdma.source1370, !dbg !13
  %k0.prime.kdma.source1392 = or disjoint i64 %k0.prime.1919, %k0.prime.kdma.source1391, !dbg !13
  %k0.prime.kdma.source1393 = trunc nuw i64 %k0.prime.kdma.source1392 to i32, !dbg !13
  %k0.prime.kdma.source1394 = shl nuw nsw i32 %k0.prime.kdma.source1393, 1, !dbg !13
  %k0.prime.kdma.offset1395 = or disjoint i32 %k0.prime.kdma.source1394, %kdma.intra.hi, !dbg !13
  %k0.prime.1924 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1396, i32 %k0.prime.kdma.offset1395, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1398 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !13
  %k0.prime.kdma.source1399 = lshr i64 %k0.prime.kdma.source1398, 4, !dbg !13
  %k0.prime.kdma.source1400 = shl nuw nsw i64 %k0.prime.kdma.source1399, 9, !dbg !13
  %k0.prime.kdma.source1402 = add nuw nsw i64 %k0.prime.kdma.source1400, %k0.prime.kdma.source1358, !dbg !13
  %k0.prime.kdma.source1403 = lshr i64 %k0.prime.kdma.source1398, 5, !dbg !13
  %k0.prime.kdma.source1406 = and i64 %k0.prime.kdma.source1399, %k0.prime.kdma.source1362, !dbg !13
  %k0.prime.kdma.source1407 = or disjoint i64 %k0.prime.kdma.source1403, %k0.prime.kdma.source1406, !dbg !13
  %k0.prime.kdma.source1408 = shl nuw nsw i64 %k0.prime.kdma.source1407, 10, !dbg !13
  %k0.prime.kdma.source1409 = sub nsw i64 %k0.prime.kdma.source1402, %k0.prime.kdma.source1408, !dbg !13
  %k0.prime.kdma.source1410 = and i64 %k0.prime.kdma.source1409, 16256, !dbg !13
  %k0.prime.kdma.source1413 = or disjoint i32 %k0.prime.kdma.source1369, 16, !dbg !13
  %k0.prime.kdma.source1414 = zext nneg i32 %k0.prime.kdma.source1413 to i64, !dbg !13
  %k0.prime.kdma.source1415 = or disjoint i64 %k0.prime.kdma.source1410, %k0.prime.kdma.source1414, !dbg !13
  %k0.prime.kdma.source1416 = or disjoint i64 %k0.prime.1919, %k0.prime.kdma.source1415, !dbg !13
  %k0.prime.kdma.source1417 = trunc nuw i64 %k0.prime.kdma.source1416 to i32, !dbg !13
  %k0.prime.kdma.source1418 = shl nuw nsw i32 %k0.prime.kdma.source1417, 1, !dbg !13
  %k0.prime.kdma.offset1419 = add nuw i32 %k0.prime.kdma.source1418, %kdma.intra, !dbg !13
  %k0.prime.1925 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.prime.kdma.offset1419, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1444 = add i32 %46, 512, !dbg !13
  %k0.prime.kdma.source1422 = or disjoint i64 %k0.prime.kdma.source1375, 256, !dbg !13
  %k0.prime.kdma.source1423 = lshr i64 %k0.prime.kdma.source1422, 4, !dbg !13
  %k0.prime.kdma.source1424 = shl nuw nsw i64 %k0.prime.kdma.source1423, 9, !dbg !13
  %k0.prime.kdma.source1426 = add nuw nsw i64 %k0.prime.kdma.source1424, %k0.prime.kdma.source1378, !dbg !13
  %k0.prime.kdma.source1427 = lshr i64 %k0.prime.kdma.source1422, 5, !dbg !13
  %k0.prime.kdma.source1430 = and i64 %k0.prime.kdma.source1423, %k0.prime.kdma.source1382, !dbg !13
  %k0.prime.kdma.source1431 = add nuw nsw i64 %k0.prime.kdma.source1427, %k0.prime.kdma.source1430, !dbg !13
  %k0.prime.kdma.source1432 = shl nuw nsw i64 %k0.prime.kdma.source1431, 10, !dbg !13
  %k0.prime.kdma.source1433 = sub nsw i64 %k0.prime.kdma.source1426, %k0.prime.kdma.source1432, !dbg !13
  %k0.prime.kdma.source1434 = and i64 %k0.prime.kdma.source1433, 16256, !dbg !13
  %k0.prime.kdma.source1439 = or disjoint i64 %k0.prime.kdma.source1434, %k0.prime.kdma.source1414, !dbg !13
  %k0.prime.kdma.source1440 = or disjoint i64 %k0.prime.1919, %k0.prime.kdma.source1439, !dbg !13
  %k0.prime.kdma.source1441 = trunc nuw i64 %k0.prime.kdma.source1440 to i32, !dbg !13
  %k0.prime.kdma.source1442 = shl nuw nsw i32 %k0.prime.kdma.source1441, 1, !dbg !13
  %k0.prime.kdma.offset1443 = add nuw i32 %k0.prime.kdma.source1442, %kdma.intra.hi, !dbg !13
  %k0.prime.1926 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1444, i32 %k0.prime.kdma.offset1443, <4 x i32> %k0.prime.dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
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
  %address.thread372 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %address.local373 = zext nneg i32 %address.thread372 to i64, !dbg !9
  %address.local379 = lshr i64 %address.local373, 2, !dbg !13
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0, !dbg !13
  %116 = bitcast <2 x i64> %115 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1233 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1356 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1357 = shl nuw nsw i64 %kdma.source1356, 9, !dbg !13
  %kdma.source1358 = shl nuw nsw i64 %kdma.source, 7, !dbg !13
  %kdma.source1359 = add nuw nsw i64 %kdma.source1357, %kdma.source1358, !dbg !13
  %kdma.source1360 = lshr i64 %kdma.source, 5, !dbg !13
  %kdma.source1361 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1362 = and i64 %kdma.source1361, 1, !dbg !13
  %kdma.source1363 = and i64 %kdma.source1356, %kdma.source1362, !dbg !13
  %kdma.source1364 = or disjoint i64 %kdma.source1360, %kdma.source1363, !dbg !13
  %kdma.source1365 = shl nuw nsw i64 %kdma.source1364, 10, !dbg !13
  %kdma.source1366 = sub nsw i64 %kdma.source1359, %kdma.source1365, !dbg !13
  %kdma.source1367 = and i64 %kdma.source1366, 16256, !dbg !13
  %kdma.source1368 = lshr i32 %kdma.tid, 4, !dbg !13
  %kdma.source1369 = and i32 %kdma.source1368, 8, !dbg !13
  %kdma.source1370 = zext nneg i32 %kdma.source1369 to i64, !dbg !13
  %kdma.source1371 = or disjoint i64 %kdma.source1367, %kdma.source1370, !dbg !13
  %kdma.source1372 = or disjoint i64 %113, %kdma.source1371, !dbg !13
  %kdma.source1373 = trunc nuw i64 %kdma.source1372 to i32, !dbg !13
  %kdma.source1374 = shl nuw nsw i32 %kdma.source1373, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1374, %kdma.intra, !dbg !13
  %kdma.source1375 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1376 = lshr i64 %kdma.source1375, 4, !dbg !13
  %kdma.source1377 = shl nuw nsw i64 %kdma.source1376, 9, !dbg !13
  %kdma.source1378 = shl nuw nsw i64 %kdma.source1375, 7, !dbg !13
  %kdma.source1379 = add nuw nsw i64 %kdma.source1377, %kdma.source1378, !dbg !13
  %kdma.source1380 = lshr i64 %kdma.source1375, 5, !dbg !13
  %kdma.source1381 = lshr i64 %kdma.source1375, 2, !dbg !13
  %kdma.source1382 = and i64 %kdma.source1381, 1, !dbg !13
  %kdma.source1383 = and i64 %kdma.source1376, %kdma.source1382, !dbg !13
  %kdma.source1384 = add nuw nsw i64 %kdma.source1380, %kdma.source1383, !dbg !13
  %kdma.source1385 = shl nuw nsw i64 %kdma.source1384, 10, !dbg !13
  %kdma.source1386 = sub nsw i64 %kdma.source1379, %kdma.source1385, !dbg !13
  %kdma.source1387 = and i64 %kdma.source1386, 16256, !dbg !13
  %kdma.source1391 = or disjoint i64 %kdma.source1387, %kdma.source1370, !dbg !13
  %kdma.source1392 = or disjoint i64 %113, %kdma.source1391, !dbg !13
  %kdma.source1393 = trunc nuw i64 %kdma.source1392 to i32, !dbg !13
  %kdma.source1394 = shl nuw nsw i32 %kdma.source1393, 1, !dbg !13
  %kdma.offset1395 = or disjoint i32 %kdma.source1394, %kdma.intra.hi, !dbg !13
  %kdma.lds1396 = add i32 %38, 512, !dbg !13
  %kdma.source1398 = or disjoint i64 %kdma.source, 256, !dbg !13
  %kdma.source1399 = lshr i64 %kdma.source1398, 4, !dbg !13
  %kdma.source1400 = shl nuw nsw i64 %kdma.source1399, 9, !dbg !13
  %kdma.source1402 = add nuw nsw i64 %kdma.source1400, %kdma.source1358, !dbg !13
  %kdma.source1403 = lshr i64 %kdma.source1398, 5, !dbg !13
  %kdma.source1406 = and i64 %kdma.source1399, %kdma.source1362, !dbg !13
  %kdma.source1407 = or disjoint i64 %kdma.source1403, %kdma.source1406, !dbg !13
  %kdma.source1408 = shl nuw nsw i64 %kdma.source1407, 10, !dbg !13
  %kdma.source1409 = sub nsw i64 %kdma.source1402, %kdma.source1408, !dbg !13
  %kdma.source1410 = and i64 %kdma.source1409, 16256, !dbg !13
  %kdma.source1413 = or disjoint i32 %kdma.source1369, 16, !dbg !13
  %kdma.source1414 = zext nneg i32 %kdma.source1413 to i64, !dbg !13
  %kdma.source1415 = or disjoint i64 %kdma.source1410, %kdma.source1414, !dbg !13
  %kdma.source1416 = or disjoint i64 %113, %kdma.source1415, !dbg !13
  %kdma.source1417 = trunc nuw i64 %kdma.source1416 to i32, !dbg !13
  %kdma.source1418 = shl nuw nsw i32 %kdma.source1417, 1, !dbg !13
  %kdma.offset1419 = add nuw i32 %kdma.source1418, %kdma.intra, !dbg !13
  %kdma.source1422 = or disjoint i64 %kdma.source1375, 256, !dbg !13
  %kdma.source1423 = lshr i64 %kdma.source1422, 4, !dbg !13
  %kdma.source1424 = shl nuw nsw i64 %kdma.source1423, 9, !dbg !13
  %kdma.source1426 = add nuw nsw i64 %kdma.source1424, %kdma.source1378, !dbg !13
  %kdma.source1427 = lshr i64 %kdma.source1422, 5, !dbg !13
  %kdma.source1430 = and i64 %kdma.source1423, %kdma.source1382, !dbg !13
  %kdma.source1431 = add nuw nsw i64 %kdma.source1427, %kdma.source1430, !dbg !13
  %kdma.source1432 = shl nuw nsw i64 %kdma.source1431, 10, !dbg !13
  %kdma.source1433 = sub nsw i64 %kdma.source1426, %kdma.source1432, !dbg !13
  %kdma.source1434 = and i64 %kdma.source1433, 16256, !dbg !13
  %kdma.source1439 = or disjoint i64 %kdma.source1434, %kdma.source1414, !dbg !13
  %kdma.source1440 = or disjoint i64 %113, %kdma.source1439, !dbg !13
  %kdma.source1441 = trunc nuw i64 %kdma.source1440 to i32, !dbg !13
  %kdma.source1442 = shl nuw nsw i32 %kdma.source1441, 1, !dbg !13
  %kdma.offset1443 = add nuw i32 %kdma.source1442, %kdma.intra.hi, !dbg !13
  %kdma.lds1444 = add i32 %46, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %117 = or disjoint i64 %113, 32, !dbg !13
  %kdma.source1462 = or disjoint i64 %117, %kdma.source1371, !dbg !13
  %kdma.source1463 = trunc nuw i64 %kdma.source1462 to i32, !dbg !13
  %kdma.source1464 = shl nuw nsw i32 %kdma.source1463, 1, !dbg !13
  %kdma.offset1465 = or disjoint i32 %kdma.source1464, %kdma.intra, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1465, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1484 = or disjoint i64 %117, %kdma.source1391, !dbg !13
  %kdma.source1485 = trunc nuw i64 %kdma.source1484 to i32, !dbg !13
  %kdma.source1486 = shl nuw nsw i32 %kdma.source1485, 1, !dbg !13
  %kdma.offset1487 = or disjoint i32 %kdma.source1486, %kdma.intra.hi, !dbg !13
  %kdma.lds1488 = add i32 %51, 512, !dbg !13
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1488, i32 %kdma.offset1487, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1508 = or disjoint i64 %117, %kdma.source1415, !dbg !13
  %kdma.source1509 = trunc nuw i64 %kdma.source1508 to i32, !dbg !13
  %kdma.source1510 = shl nuw nsw i32 %kdma.source1509, 1, !dbg !13
  %kdma.offset1511 = add nuw i32 %kdma.source1510, %kdma.intra, !dbg !13
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1511, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1532 = or disjoint i64 %117, %kdma.source1439, !dbg !13
  %kdma.source1533 = trunc nuw i64 %kdma.source1532 to i32, !dbg !13
  %kdma.source1534 = shl nuw nsw i32 %kdma.source1533, 1, !dbg !13
  %kdma.offset1535 = add nuw i32 %kdma.source1534, %kdma.intra.hi, !dbg !13
  %kdma.lds1536 = add i32 %55, 512, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1536, i32 %kdma.offset1535, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %address.local416 = shl nuw nsw i64 %address.local373, 1, !dbg !15
  %address.local417 = and i64 %address.local416, 24, !dbg !15
  %address.local419 = shl i64 %address.local373, 62, !dbg !15
  %address.local420 = ashr exact i64 %address.local419, 57, !dbg !15
  %address.local421 = and i64 %address.local420, -64, !dbg !15
  %address.local424 = and i64 %address.local379, 12, !dbg !12
  %address.local426 = and i64 %address.local373, 63, !dbg !12
  %address.local427 = lshr i64 %address.local426, 5, !dbg !12
  %address.local428 = mul nuw nsw i64 %address.local427, 1016, !dbg !12
  %.idx1271 = mul nuw nsw i64 %address.local373, 320, !dbg !15
  %122 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1271, !dbg !15
  %.idx1272 = mul nsw i64 %address.local379, -1280, !dbg !15
  %123 = getelementptr i8, ptr addrspace(3) %122, i64 %.idx1272, !dbg !15
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local417, !dbg !15
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local421, !dbg !15
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local424, !dbg !15
  %address.local430 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local428, !dbg !15
  %127 = add nsw i64 %.idx1271, %.idx1272, !dbg !15
  %.idx1847 = shl nuw nsw i64 %address.local417, 1, !dbg !15
  %128 = or disjoint i64 %127, %.idx1847, !dbg !15
  %129 = or disjoint i64 %address.local421, %address.local424, !dbg !15
  %130 = shl nsw i64 %129, 1, !dbg !15
  %131 = add nsw i64 %130, %128, !dbg !15
  %address.local430.idx = mul nuw nsw i64 %address.local427, 2032, !dbg !15
  %132 = add nsw i64 %131, %address.local430.idx, !dbg !15
  %133 = lshr i64 %132, 6, !dbg !15
  %134 = and i64 %133, 8, !dbg !15
  %135 = xor i64 %132, %134, !dbg !15
  %136 = sub i64 %135, %132, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local430, i64 %136, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %lazy.base = extractvalue [2 x <1 x float>] %104, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base1932 = extractvalue [2 x <1 x float>] %104, 1, !dbg !15
  %lazy.seed1933 = shufflevector <1 x float> %lazy.base1932, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> %lazy.seed) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local441 = shl nuw nsw i32 %address.thread372, 1, !dbg !15
  %address.local442 = and i32 %address.local441, 24, !dbg !15
  %address.local443 = shl nuw nsw i32 %address.thread372, 8, !dbg !15
  %address.local444 = and i32 %address.local443, 768, !dbg !15
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !15
  %address.local446 = shl nuw nsw i32 %address.thread372, 5, !dbg !15
  %address.local447 = and i32 %address.local446, 96, !dbg !15
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !15
  %address.local449 = zext nneg i32 %address.local448 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %99) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local450 = lshr i32 %address.thread372, 1, !dbg !15
  %address.local451 = and i32 %address.local450, 1, !dbg !15
  %address.local452 = zext nneg i32 %address.local451 to i64, !dbg !15
  %address.local453 = add nsw i64 %address.base, %address.local452, !dbg !15
  %address.local454 = shl i64 %address.local453, 6, !dbg !15
  %address.local455 = sub i64 %address.local449, %address.local454, !dbg !15
  %140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local455, !dbg !15
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local424, !dbg !15
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local428, !dbg !15
  %address.local459 = getelementptr i8, ptr addrspace(3) %142, i64 64, !dbg !15
  %.idx18501853 = add i64 %address.local455, %address.local424, !dbg !15
  %143 = shl i64 %.idx18501853, 1, !dbg !15
  %144 = add i64 %143, %address.local430.idx, !dbg !15
  %145 = add i64 %144, 64, !dbg !15
  %146 = lshr i64 %145, 6, !dbg !15
  %147 = and i64 %146, 8, !dbg !15
  %148 = xor i64 %145, %147, !dbg !15
  %149 = sub i64 %148, %145, !dbg !15
  %kdma.swizzled1721 = getelementptr i8, ptr addrspace(3) %address.local459, i64 %149, !dbg !15
  %150 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1721, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1725 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 128, !dbg !15
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1725, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1722 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 4096, !dbg !15
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1726 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 4224, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1726, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1723 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1727 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1724 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1724, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !15, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %139, <4 x float> %138) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 1024, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 1152, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !15, !frisk.k_lds_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 5120, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 5248, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !15, !frisk.k_lds_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !15, !frisk.k_lds_pair !38
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
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> %lazy.seed1933) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> %lazy.seed1933) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> %lazy.seed1933) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> %lazy.seed1933) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> %lazy.seed1933) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> %lazy.seed1933) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> %lazy.seed1933) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> %lazy.seed1933) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %163, <4 x float> %193) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %195 = or disjoint i64 %113, 64, !dbg !13
  %kdma.source1554 = or disjoint i64 %195, %kdma.source1371, !dbg !13
  %kdma.source1555 = trunc nuw i64 %kdma.source1554 to i32, !dbg !13
  %kdma.source1556 = shl nuw nsw i32 %kdma.source1555, 1, !dbg !13
  %kdma.offset1557 = or disjoint i32 %kdma.source1556, %kdma.intra, !dbg !13
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %kdma.offset1557, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1576 = or disjoint i64 %195, %kdma.source1391, !dbg !13
  %kdma.source1577 = trunc nuw i64 %kdma.source1576 to i32, !dbg !13
  %kdma.source1578 = shl nuw nsw i32 %kdma.source1577, 1, !dbg !13
  %kdma.offset1579 = or disjoint i32 %kdma.source1578, %kdma.intra.hi, !dbg !13
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1396, i32 %kdma.offset1579, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1600 = or disjoint i64 %195, %kdma.source1415, !dbg !13
  %kdma.source1601 = trunc nuw i64 %kdma.source1600 to i32, !dbg !13
  %kdma.source1602 = shl nuw nsw i32 %kdma.source1601, 1, !dbg !13
  %kdma.offset1603 = add nuw i32 %kdma.source1602, %kdma.intra, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %kdma.offset1603, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1624 = or disjoint i64 %195, %kdma.source1439, !dbg !13
  %kdma.source1625 = trunc nuw i64 %kdma.source1624 to i32, !dbg !13
  %kdma.source1626 = shl nuw nsw i32 %kdma.source1625, 1, !dbg !13
  %kdma.offset1627 = add nuw i32 %kdma.source1626, %kdma.intra.hi, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1444, i32 %kdma.offset1627, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.gep1751 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1755 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1752 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1756 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1756, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1753 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 8192, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1757 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 8320, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1754 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 12288, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1758 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 12416, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1759 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1763 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !15, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %200, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1760 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1764 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %210) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1761 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 9216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1765 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 9344, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !15, !frisk.k_lds_pair !45
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1762 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 13312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !15, !frisk.k_lds_pair !46
  %kdma.gep1766 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1721, i64 13440, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !15, !frisk.k_lds_pair !46
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
  %kdma.source1646 = or disjoint i64 %248, %kdma.source1371, !dbg !13
  %kdma.source1647 = trunc nuw i64 %kdma.source1646 to i32, !dbg !13
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !13
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !13
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1649, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1668 = or disjoint i64 %248, %kdma.source1391, !dbg !13
  %kdma.source1669 = trunc nuw i64 %kdma.source1668 to i32, !dbg !13
  %kdma.source1670 = shl nuw nsw i32 %kdma.source1669, 1, !dbg !13
  %kdma.offset1671 = or disjoint i32 %kdma.source1670, %kdma.intra.hi, !dbg !13
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1488, i32 %kdma.offset1671, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1692 = or disjoint i64 %248, %kdma.source1415, !dbg !13
  %kdma.source1693 = trunc nuw i64 %kdma.source1692 to i32, !dbg !13
  %kdma.source1694 = shl nuw nsw i32 %kdma.source1693, 1, !dbg !13
  %kdma.offset1695 = add nuw i32 %kdma.source1694, %kdma.intra, !dbg !13
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1695, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %kdma.source1716 = or disjoint i64 %248, %kdma.source1439, !dbg !13
  %kdma.source1717 = trunc nuw i64 %kdma.source1716 to i32, !dbg !13
  %kdma.source1718 = shl nuw nsw i32 %kdma.source1717, 1, !dbg !13
  %kdma.offset1719 = add nuw i32 %kdma.source1718, %kdma.intra.hi, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1536, i32 %kdma.offset1719, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !47
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1723, align 8, !dbg !15, !frisk.k_lds_pair !47
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !48
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1724, align 8, !dbg !15, !frisk.k_lds_pair !48
  %257 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1721, align 8, !dbg !15, !frisk.k_lds_pair !49
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1725, align 8, !dbg !15, !frisk.k_lds_pair !49
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1722, align 8, !dbg !15, !frisk.k_lds_pair !50
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1726, align 8, !dbg !15, !frisk.k_lds_pair !50
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !15, !frisk.k_lds_pair !51
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %253, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !15, !frisk.k_lds_pair !52
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %255, <4 x float> %263) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !15, !frisk.k_lds_pair !53
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !15, !frisk.k_lds_pair !53
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %257, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !15, !frisk.k_lds_pair !54
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !15, !frisk.k_lds_pair !54
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
  %address.local788 = shl nuw nsw i64 %address.local373, 5, !dbg !16
  %address.local789 = and i64 %address.local788, 8064, !dbg !16
  %address.local790 = shl nuw nsw i64 %address.local373, 3, !dbg !9
  %address.local791 = shl nuw nsw i64 %address.local379, 5, !dbg !16
  %address.local792 = sub nsw i64 %address.local790, %address.local791, !dbg !16
  %address.local793 = add nsw i64 %address.local789, %address.local792, !dbg !16
  %301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx12811282 = add nsw i64 %113, %address.local793, !dbg !16
  %.idx12811282.tr = trunc i64 %.idx12811282 to i32, !dbg !16
  %dma.buffer.offset1216 = shl i32 %.idx12811282.tr, 1, !dbg !16
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !16
  %303 = bitcast <2 x i64> %302 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1217 = shufflevector <4 x i32> %303, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1216, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %address.local795 = shl nuw nsw i32 %address.thread372, 3, !dbg !16
  %address.local796 = and i32 %address.local795, 24, !dbg !16
  %address.local798 = and i32 %address.local446, 8064, !dbg !16
  %address.local799 = or disjoint i32 %address.local796, %address.local798, !dbg !16
  %address.local800 = or disjoint i32 %address.local799, 8192, !dbg !16
  %305 = trunc nuw i64 %113 to i32, !dbg !16
  %.idx12831284.tr = or disjoint i32 %address.local800, %305, !dbg !16
  %dma.buffer.offset1214 = shl nuw nsw i32 %.idx12831284.tr, 1, !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1214, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1751, align 8, !dbg !15, !frisk.k_tail_pair !33
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1755, align 8, !dbg !15, !frisk.k_tail_pair !33
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1752, align 8, !dbg !15, !frisk.k_tail_pair !34
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1756, align 8, !dbg !15, !frisk.k_tail_pair !34
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1753, align 8, !dbg !15, !frisk.k_tail_pair !35
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1757, align 8, !dbg !15, !frisk.k_tail_pair !35
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1754, align 8, !dbg !15, !frisk.k_tail_pair !36
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1758, align 8, !dbg !15, !frisk.k_tail_pair !36
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
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1759, align 8, !dbg !15, !frisk.k_tail_pair !31
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1763, align 8, !dbg !15, !frisk.k_tail_pair !31
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
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1760, align 8, !dbg !15, !frisk.k_tail_pair !37
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1764, align 8, !dbg !15, !frisk.k_tail_pair !37
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %326) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1761, align 8, !dbg !15, !frisk.k_tail_pair !38
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1765, align 8, !dbg !15, !frisk.k_tail_pair !38
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
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1762, align 8, !dbg !15, !frisk.k_tail_pair !39
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1766, align 8, !dbg !15, !frisk.k_tail_pair !39
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
  %lazy.positive1934 = call float @llvm.maxnum.f32(float %498, float 0x0000000000000000), !dbg !23
  %lazy.delta1935 = select i1 %lazy.first, float %498, float %lazy.positive1934, !dbg !23
  %lazy.negated1936 = fsub float 0x0000000000000000, %lazy.delta1935, !dbg !23
  %lazy.exp1937 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated1936), !dbg !23
  %lazy.alpha1938 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp1937, !dbg !23
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted = fsub <4 x float> %316, %lazy.delta.v, !dbg !23
  %lazy.delta.v.lane1939 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1940 = shufflevector <1 x float> %lazy.delta.v.lane1939, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1941 = fsub <4 x float> %323, %lazy.delta.v1940, !dbg !23
  %lazy.delta.v.lane1942 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1943 = shufflevector <1 x float> %lazy.delta.v.lane1942, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1944 = fsub <4 x float> %337, %lazy.delta.v1943, !dbg !23
  %lazy.delta.v.lane1945 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1946 = shufflevector <1 x float> %lazy.delta.v.lane1945, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1947 = fsub <4 x float> %351, %lazy.delta.v1946, !dbg !23
  %lazy.delta.v.lane1948 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1949 = shufflevector <1 x float> %lazy.delta.v.lane1948, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1950 = fsub <4 x float> %361, %lazy.delta.v1949, !dbg !23
  %lazy.delta.v.lane1951 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1952 = shufflevector <1 x float> %lazy.delta.v.lane1951, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1953 = fsub <4 x float> %371, %lazy.delta.v1952, !dbg !23
  %lazy.delta.v.lane1954 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1955 = shufflevector <1 x float> %lazy.delta.v.lane1954, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1956 = fsub <4 x float> %381, %lazy.delta.v1955, !dbg !23
  %lazy.delta.v.lane1957 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1958 = shufflevector <1 x float> %lazy.delta.v.lane1957, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1959 = fsub <4 x float> %391, %lazy.delta.v1958, !dbg !23
  %lazy.delta.v.lane1960 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1961 = shufflevector <1 x float> %lazy.delta.v.lane1960, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1962 = fsub <4 x float> %404, %lazy.delta.v1961, !dbg !23
  %lazy.delta.v.lane1963 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1964 = shufflevector <1 x float> %lazy.delta.v.lane1963, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1965 = fsub <4 x float> %410, %lazy.delta.v1964, !dbg !23
  %lazy.delta.v.lane1966 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1967 = shufflevector <1 x float> %lazy.delta.v.lane1966, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1968 = fsub <4 x float> %416, %lazy.delta.v1967, !dbg !23
  %lazy.delta.v.lane1969 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1970 = shufflevector <1 x float> %lazy.delta.v.lane1969, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1971 = fsub <4 x float> %422, %lazy.delta.v1970, !dbg !23
  %lazy.delta.v.lane1972 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1973 = shufflevector <1 x float> %lazy.delta.v.lane1972, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1974 = fsub <4 x float> %428, %lazy.delta.v1973, !dbg !23
  %lazy.delta.v.lane1975 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1976 = shufflevector <1 x float> %lazy.delta.v.lane1975, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1977 = fsub <4 x float> %434, %lazy.delta.v1976, !dbg !23
  %lazy.delta.v.lane1978 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1979 = shufflevector <1 x float> %lazy.delta.v.lane1978, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1980 = fsub <4 x float> %440, %lazy.delta.v1979, !dbg !23
  %lazy.delta.v.lane1981 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.delta.v1982 = shufflevector <1 x float> %lazy.delta.v.lane1981, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1983 = fsub <4 x float> %446, %lazy.delta.v1982, !dbg !23
  %lazy.state = extractvalue [2 x <1 x float>] %103, 0, !dbg !23
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !23
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !23
  %lazy.state1984 = extractvalue [2 x <1 x float>] %103, 1, !dbg !23
  %lazy.alpha.v.lane1985 = insertelement <1 x float> poison, float %lazy.alpha1938, i64 0, !dbg !23
  %lazy.scaled1986 = fmul <1 x float> %lazy.state1984, %lazy.alpha.v.lane1985, !dbg !23
  %lazy.state.next1987 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1986, 1, !dbg !23
  %lazy.state1988 = extractvalue [2 x <8 x float>] %107, 0, !dbg !23
  %lazy.alpha.v.lane1989 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane1989, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1990 = fmul <8 x float> %lazy.state1988, %lazy.alpha.v, !dbg !23
  %lazy.state.next1991 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1990, 0, !dbg !23
  %lazy.state1992 = extractvalue [2 x <8 x float>] %107, 1, !dbg !23
  %lazy.alpha.v.lane1993 = insertelement <1 x float> poison, float %lazy.alpha1938, i64 0, !dbg !23
  %lazy.alpha.v1994 = shufflevector <1 x float> %lazy.alpha.v.lane1993, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1995 = fmul <8 x float> %lazy.state1992, %lazy.alpha.v1994, !dbg !23
  %lazy.state.next1996 = insertvalue [2 x <8 x float>] %lazy.state.next1991, <8 x float> %lazy.scaled1995, 1, !dbg !23
  %lazy.state1997 = extractvalue [2 x <8 x float>] %106, 0, !dbg !23
  %lazy.alpha.v.lane1998 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v1999 = shufflevector <1 x float> %lazy.alpha.v.lane1998, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2000 = fmul <8 x float> %lazy.state1997, %lazy.alpha.v1999, !dbg !23
  %lazy.state.next2001 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2000, 0, !dbg !23
  %lazy.state2002 = extractvalue [2 x <8 x float>] %106, 1, !dbg !23
  %lazy.alpha.v.lane2003 = insertelement <1 x float> poison, float %lazy.alpha1938, i64 0, !dbg !23
  %lazy.alpha.v2004 = shufflevector <1 x float> %lazy.alpha.v.lane2003, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2005 = fmul <8 x float> %lazy.state2002, %lazy.alpha.v2004, !dbg !23
  %lazy.state.next2006 = insertvalue [2 x <8 x float>] %lazy.state.next2001, <8 x float> %lazy.scaled2005, 1, !dbg !23
  %lazy.state2007 = extractvalue [2 x <8 x float>] %105, 0, !dbg !23
  %lazy.alpha.v.lane2008 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2009 = shufflevector <1 x float> %lazy.alpha.v.lane2008, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2010 = fmul <8 x float> %lazy.state2007, %lazy.alpha.v2009, !dbg !23
  %lazy.state.next2011 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2010, 0, !dbg !23
  %lazy.state2012 = extractvalue [2 x <8 x float>] %105, 1, !dbg !23
  %lazy.alpha.v.lane2013 = insertelement <1 x float> poison, float %lazy.alpha1938, i64 0, !dbg !23
  %lazy.alpha.v2014 = shufflevector <1 x float> %lazy.alpha.v.lane2013, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2015 = fmul <8 x float> %lazy.state2012, %lazy.alpha.v2014, !dbg !23
  %lazy.state.next2016 = insertvalue [2 x <8 x float>] %lazy.state.next2011, <8 x float> %lazy.scaled2015, 1, !dbg !23
  %lazy.state2017 = extractvalue [2 x <8 x float>] %102, 0, !dbg !23
  %lazy.alpha.v.lane2018 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v2019 = shufflevector <1 x float> %lazy.alpha.v.lane2018, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2020 = fmul <8 x float> %lazy.state2017, %lazy.alpha.v2019, !dbg !23
  %lazy.state.next2021 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled2020, 0, !dbg !23
  %lazy.state2022 = extractvalue [2 x <8 x float>] %102, 1, !dbg !23
  %lazy.alpha.v.lane2023 = insertelement <1 x float> poison, float %lazy.alpha1938, i64 0, !dbg !23
  %lazy.alpha.v2024 = shufflevector <1 x float> %lazy.alpha.v.lane2023, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2025 = fmul <8 x float> %lazy.state2022, %lazy.alpha.v2024, !dbg !23
  %lazy.state.next2026 = insertvalue [2 x <8 x float>] %lazy.state.next2021, <8 x float> %lazy.scaled2025, 1, !dbg !23
  %lazy.base.old = extractvalue [2 x <1 x float>] %104, 0, !dbg !23
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !23
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !23
  %lazy.base.old2027 = extractvalue [2 x <1 x float>] %104, 1, !dbg !23
  %lazy.delta.b.lane2028 = insertelement <1 x float> poison, float %lazy.delta1935, i64 0, !dbg !23
  %lazy.base.next2029 = fsub <1 x float> %lazy.base.old2027, %lazy.delta.b.lane2028, !dbg !23
  %lazy.base.state2030 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2029, 1, !dbg !23
  br label %lazy.join, !dbg !23

lazy.join:                                        ; preds = %lazy.slow, %109
  %lazy.score = phi <4 x float> [ %316, %109 ], [ %lazy.shifted, %lazy.slow ], !dbg !23
  %lazy.score2031 = phi <4 x float> [ %323, %109 ], [ %lazy.shifted1941, %lazy.slow ], !dbg !23
  %lazy.score2032 = phi <4 x float> [ %337, %109 ], [ %lazy.shifted1944, %lazy.slow ], !dbg !23
  %lazy.score2033 = phi <4 x float> [ %351, %109 ], [ %lazy.shifted1947, %lazy.slow ], !dbg !23
  %lazy.score2034 = phi <4 x float> [ %361, %109 ], [ %lazy.shifted1950, %lazy.slow ], !dbg !23
  %lazy.score2035 = phi <4 x float> [ %371, %109 ], [ %lazy.shifted1953, %lazy.slow ], !dbg !23
  %lazy.score2036 = phi <4 x float> [ %381, %109 ], [ %lazy.shifted1956, %lazy.slow ], !dbg !23
  %lazy.score2037 = phi <4 x float> [ %391, %109 ], [ %lazy.shifted1959, %lazy.slow ], !dbg !23
  %lazy.score2038 = phi <4 x float> [ %404, %109 ], [ %lazy.shifted1962, %lazy.slow ], !dbg !23
  %lazy.score2039 = phi <4 x float> [ %410, %109 ], [ %lazy.shifted1965, %lazy.slow ], !dbg !23
  %lazy.score2040 = phi <4 x float> [ %416, %109 ], [ %lazy.shifted1968, %lazy.slow ], !dbg !23
  %lazy.score2041 = phi <4 x float> [ %422, %109 ], [ %lazy.shifted1971, %lazy.slow ], !dbg !23
  %lazy.score2042 = phi <4 x float> [ %428, %109 ], [ %lazy.shifted1974, %lazy.slow ], !dbg !23
  %lazy.score2043 = phi <4 x float> [ %434, %109 ], [ %lazy.shifted1977, %lazy.slow ], !dbg !23
  %lazy.score2044 = phi <4 x float> [ %440, %109 ], [ %lazy.shifted1980, %lazy.slow ], !dbg !23
  %lazy.score2045 = phi <4 x float> [ %446, %109 ], [ %lazy.shifted1983, %lazy.slow ], !dbg !23
  %lazy.state2046 = phi [2 x <1 x float>] [ %103, %109 ], [ %lazy.state.next1987, %lazy.slow ], !dbg !23
  %lazy.state2047 = phi [2 x <8 x float>] [ %107, %109 ], [ %lazy.state.next1996, %lazy.slow ], !dbg !23
  %lazy.state2048 = phi [2 x <8 x float>] [ %106, %109 ], [ %lazy.state.next2006, %lazy.slow ], !dbg !23
  %lazy.state2049 = phi [2 x <8 x float>] [ %105, %109 ], [ %lazy.state.next2016, %lazy.slow ], !dbg !23
  %lazy.state2050 = phi [2 x <8 x float>] [ %102, %109 ], [ %lazy.state.next2026, %lazy.slow ], !dbg !23
  %lazy.base.join = phi [2 x <1 x float>] [ %104, %109 ], [ %lazy.base.state2030, %lazy.slow ], !dbg !23
  %499 = extractvalue [2 x <1 x float>] %lazy.state2046, 0, !dbg !23
  %500 = extractelement <1 x float> %499, i64 0, !dbg !23
  %501 = extractvalue [2 x <1 x float>] %lazy.state2046, 1, !dbg !23
  %502 = extractelement <1 x float> %501, i64 0, !dbg !23
  %503 = extractvalue [2 x <8 x float>] %lazy.state2050, 0, !dbg !27
  %504 = shufflevector <8 x float> %503, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %505 = shufflevector <8 x float> %503, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %506 = extractvalue [2 x <8 x float>] %lazy.state2050, 1, !dbg !27
  %507 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %508 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address = add nsw i64 %117, %address.local793, !dbg !16
  %vdb.address1297 = trunc i64 %vdb.address to i32, !dbg !16
  %vdb.address1298 = shl i32 %vdb.address1297, 1, !dbg !16
  %vdb.address1303 = trunc nuw i64 %117 to i32, !dbg !16
  %vdb.address1304 = or disjoint i32 %address.local800, %vdb.address1303, !dbg !16
  %vdb.address1305 = shl nuw nsw i32 %vdb.address1304, 1, !dbg !16
  %vdb.dma.base = add i32 %38, 8192, !dbg !14
  %509 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1298, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %vdb.dma.base1310 = add i32 %46, 8192, !dbg !14
  %510 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1310, i32 %vdb.address1305, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %address.local804 = and i32 %address.local795, 120, !dbg !14
  %address.local806 = and i32 %address.local446, 1536, !dbg !14
  %address.local807 = or disjoint i32 %address.local804, %address.local806, !dbg !14
  %address.local808 = zext nneg i32 %address.local807 to i64, !dbg !14
  %address.local809 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local808, !dbg !14
  %511 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local809, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %512 = bitcast <4 x i32> %511 to <8 x half>, !dbg !14
  %address.local810 = and i64 %address.local373, 60, !dbg !14
  %address.local811 = add nuw nsw i64 %address.local810, %address.local373, !dbg !14
  %address.local813.idx = shl nuw nsw i64 %address.local811, 4, !dbg !14
  %address.local813 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local813.idx, !dbg !14
  %address.local814 = lshr i64 %address.local426, 4, !dbg !14
  %address.local816.idx = mul nuw nsw i64 %address.local814, 768, !dbg !14
  %address.local816 = getelementptr i8, ptr addrspace(3) %address.local813, i64 %address.local816.idx, !dbg !14
  %address.local818.idx = mul nsw i64 %address.local379, -64, !dbg !14
  %address.local818 = getelementptr i8, ptr addrspace(3) %address.local816, i64 %address.local818.idx, !dbg !14
  %address.local819 = getelementptr i8, ptr addrspace(3) %address.local818, i64 256, !dbg !14
  %513 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %514 = bitcast <4 x i32> %513 to <8 x half>, !dbg !14
  %address.local823 = getelementptr i8, ptr addrspace(3) %address.local818, i64 512, !dbg !14
  %515 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %516 = bitcast <4 x i32> %515 to <8 x half>, !dbg !14
  %address.local827 = getelementptr i8, ptr addrspace(3) %address.local818, i64 768, !dbg !14
  %517 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local827, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %518 = bitcast <4 x i32> %517 to <8 x half>, !dbg !14
  %address.local829 = getelementptr i8, ptr addrspace(3) %address.local809, i64 4096, !dbg !14
  %519 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local829, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %520 = bitcast <4 x i32> %519 to <8 x half>, !dbg !14
  %address.local833 = getelementptr i8, ptr addrspace(3) %address.local818, i64 4352, !dbg !14
  %521 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local833, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %522 = bitcast <4 x i32> %521 to <8 x half>, !dbg !14
  %address.local837 = getelementptr i8, ptr addrspace(3) %address.local818, i64 4608, !dbg !14
  %523 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local837, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %524 = bitcast <4 x i32> %523 to <8 x half>, !dbg !14
  %address.local841 = getelementptr i8, ptr addrspace(3) %address.local818, i64 4864, !dbg !14
  %525 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
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
  %543 = extractelement <4 x float> %lazy.score2038, i64 0, !dbg !24
  %544 = call float @llvm.amdgcn.exp2.f32(float %543), !dbg !24
  %545 = insertelement <4 x float> poison, float %544, i64 0, !dbg !26
  %546 = extractelement <4 x float> %lazy.score2038, i64 1, !dbg !24
  %547 = call float @llvm.amdgcn.exp2.f32(float %546), !dbg !24
  %548 = insertelement <4 x float> %545, float %547, i64 1, !dbg !26
  %549 = extractelement <4 x float> %lazy.score2038, i64 2, !dbg !24
  %550 = call float @llvm.amdgcn.exp2.f32(float %549), !dbg !24
  %551 = insertelement <4 x float> %548, float %550, i64 2, !dbg !26
  %552 = extractelement <4 x float> %lazy.score2038, i64 3, !dbg !24
  %553 = call float @llvm.amdgcn.exp2.f32(float %552), !dbg !24
  %554 = insertelement <4 x float> %551, float %553, i64 3, !dbg !26
  %math.result259 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02051 = extractelement <4 x float> %math.result259, i32 0, !dbg !26
  %.cvtpack.f12052 = extractelement <4 x float> %math.result259, i32 1, !dbg !26
  %.cvtpack.f22053 = extractelement <4 x float> %math.result259, i32 2, !dbg !26
  %.cvtpack.f32054 = extractelement <4 x float> %math.result259, i32 3, !dbg !26
  %.cvtpack.p02055 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02051, float %.cvtpack.f12052), !dbg !26
  %.cvtpack.p12056 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22053, float %.cvtpack.f32054), !dbg !26
  %.cvtpack.v02057 = insertelement <2 x i32> poison, i32 %.cvtpack.p02055, i32 0, !dbg !26
  %.cvtpack.v12058 = insertelement <2 x i32> %.cvtpack.v02057, i32 %.cvtpack.p12056, i32 1, !dbg !26
  %555 = bitcast <2 x i32> %.cvtpack.v12058 to <4 x half>, !dbg !26
  %math.packed260 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %556 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %526, <4 x float> %507) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %541, <4 x float> %508) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %558 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %559 = extractelement <4 x float> %lazy.score2031, i64 0, !dbg !24
  %560 = call float @llvm.amdgcn.exp2.f32(float %559), !dbg !24
  %561 = fadd float %528, %560, !dbg !25
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %561) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %562 = insertelement <4 x float> poison, float %560, i64 0, !dbg !26
  %563 = extractelement <4 x float> %lazy.score2031, i64 1, !dbg !24
  %564 = call float @llvm.amdgcn.exp2.f32(float %563), !dbg !24
  %565 = fadd float %531, %564, !dbg !25
  %k0.sum.anchor1854 = call float asm sideeffect "", "=v,0"(float %565) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %566 = insertelement <4 x float> %562, float %564, i64 1, !dbg !26
  %567 = extractelement <4 x float> %lazy.score2031, i64 2, !dbg !24
  %568 = call float @llvm.amdgcn.exp2.f32(float %567), !dbg !24
  %569 = fadd float %534, %568, !dbg !25
  %k0.sum.anchor1855 = call float asm sideeffect "", "=v,0"(float %569) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %570 = insertelement <4 x float> %566, float %568, i64 2, !dbg !26
  %571 = extractelement <4 x float> %lazy.score2031, i64 3, !dbg !24
  %572 = call float @llvm.amdgcn.exp2.f32(float %571), !dbg !24
  %573 = fadd float %537, %572, !dbg !25
  %k0.sum.anchor1856 = call float asm sideeffect "", "=v,0"(float %573) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %574 = insertelement <4 x float> %570, float %572, i64 3, !dbg !26
  %math.result245 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %574) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02059 = extractelement <4 x float> %math.result245, i32 0, !dbg !26
  %.cvtpack.f12060 = extractelement <4 x float> %math.result245, i32 1, !dbg !26
  %.cvtpack.f22061 = extractelement <4 x float> %math.result245, i32 2, !dbg !26
  %.cvtpack.f32062 = extractelement <4 x float> %math.result245, i32 3, !dbg !26
  %.cvtpack.p02063 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02059, float %.cvtpack.f12060), !dbg !26
  %.cvtpack.p12064 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22061, float %.cvtpack.f32062), !dbg !26
  %.cvtpack.v02065 = insertelement <2 x i32> poison, i32 %.cvtpack.p02063, i32 0, !dbg !26
  %.cvtpack.v12066 = insertelement <2 x i32> %.cvtpack.v02065, i32 %.cvtpack.p12064, i32 1, !dbg !26
  %575 = bitcast <2 x i32> %.cvtpack.v12066 to <4 x half>, !dbg !26
  %math.packed246 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %575) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %576 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %558, <4 x float> %540) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %577 = shufflevector <8 x half> %514, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %577, <4 x float> %542) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = extractelement <4 x float> %lazy.score2039, i64 0, !dbg !24
  %580 = call float @llvm.amdgcn.exp2.f32(float %579), !dbg !24
  %581 = fadd float %544, %580, !dbg !25
  %k0.sum.anchor1857 = call float asm sideeffect "", "=v,0"(float %581) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %582 = insertelement <4 x float> poison, float %580, i64 0, !dbg !26
  %583 = extractelement <4 x float> %lazy.score2039, i64 1, !dbg !24
  %584 = call float @llvm.amdgcn.exp2.f32(float %583), !dbg !24
  %585 = fadd float %547, %584, !dbg !25
  %k0.sum.anchor1858 = call float asm sideeffect "", "=v,0"(float %585) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %586 = insertelement <4 x float> %582, float %584, i64 1, !dbg !26
  %587 = extractelement <4 x float> %lazy.score2039, i64 2, !dbg !24
  %588 = call float @llvm.amdgcn.exp2.f32(float %587), !dbg !24
  %589 = fadd float %550, %588, !dbg !25
  %k0.sum.anchor1859 = call float asm sideeffect "", "=v,0"(float %589) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %590 = insertelement <4 x float> %586, float %588, i64 2, !dbg !26
  %591 = extractelement <4 x float> %lazy.score2039, i64 3, !dbg !24
  %592 = call float @llvm.amdgcn.exp2.f32(float %591), !dbg !24
  %593 = fadd float %553, %592, !dbg !25
  %k0.sum.anchor1860 = call float asm sideeffect "", "=v,0"(float %593) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %594 = insertelement <4 x float> %590, float %592, i64 3, !dbg !26
  %math.result261 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %594) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02067 = extractelement <4 x float> %math.result261, i32 0, !dbg !26
  %.cvtpack.f12068 = extractelement <4 x float> %math.result261, i32 1, !dbg !26
  %.cvtpack.f22069 = extractelement <4 x float> %math.result261, i32 2, !dbg !26
  %.cvtpack.f32070 = extractelement <4 x float> %math.result261, i32 3, !dbg !26
  %.cvtpack.p02071 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02067, float %.cvtpack.f12068), !dbg !26
  %.cvtpack.p12072 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22069, float %.cvtpack.f32070), !dbg !26
  %.cvtpack.v02073 = insertelement <2 x i32> poison, i32 %.cvtpack.p02071, i32 0, !dbg !26
  %.cvtpack.v12074 = insertelement <2 x i32> %.cvtpack.v02073, i32 %.cvtpack.p12072, i32 1, !dbg !26
  %595 = bitcast <2 x i32> %.cvtpack.v12074 to <4 x half>, !dbg !26
  %math.packed262 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %595) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %558, <4 x float> %556) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %577, <4 x float> %557) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %598 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %599 = extractelement <4 x float> %lazy.score2032, i64 0, !dbg !24
  %600 = call float @llvm.amdgcn.exp2.f32(float %599), !dbg !24
  %601 = fadd float %k0.sum.anchor, %600, !dbg !25
  %k0.sum.anchor1861 = call float asm sideeffect "", "=v,0"(float %601) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %602 = insertelement <4 x float> poison, float %600, i64 0, !dbg !26
  %603 = extractelement <4 x float> %lazy.score2032, i64 1, !dbg !24
  %604 = call float @llvm.amdgcn.exp2.f32(float %603), !dbg !24
  %605 = fadd float %k0.sum.anchor1854, %604, !dbg !25
  %k0.sum.anchor1862 = call float asm sideeffect "", "=v,0"(float %605) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %606 = insertelement <4 x float> %602, float %604, i64 1, !dbg !26
  %607 = extractelement <4 x float> %lazy.score2032, i64 2, !dbg !24
  %608 = call float @llvm.amdgcn.exp2.f32(float %607), !dbg !24
  %609 = fadd float %k0.sum.anchor1855, %608, !dbg !25
  %k0.sum.anchor1863 = call float asm sideeffect "", "=v,0"(float %609) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %610 = insertelement <4 x float> %606, float %608, i64 2, !dbg !26
  %611 = extractelement <4 x float> %lazy.score2032, i64 3, !dbg !24
  %612 = call float @llvm.amdgcn.exp2.f32(float %611), !dbg !24
  %613 = fadd float %k0.sum.anchor1856, %612, !dbg !25
  %k0.sum.anchor1864 = call float asm sideeffect "", "=v,0"(float %613) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %614 = insertelement <4 x float> %610, float %612, i64 3, !dbg !26
  %math.result247 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %614) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02075 = extractelement <4 x float> %math.result247, i32 0, !dbg !26
  %.cvtpack.f12076 = extractelement <4 x float> %math.result247, i32 1, !dbg !26
  %.cvtpack.f22077 = extractelement <4 x float> %math.result247, i32 2, !dbg !26
  %.cvtpack.f32078 = extractelement <4 x float> %math.result247, i32 3, !dbg !26
  %.cvtpack.p02079 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02075, float %.cvtpack.f12076), !dbg !26
  %.cvtpack.p12080 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22077, float %.cvtpack.f32078), !dbg !26
  %.cvtpack.v02081 = insertelement <2 x i32> poison, i32 %.cvtpack.p02079, i32 0, !dbg !26
  %.cvtpack.v12082 = insertelement <2 x i32> %.cvtpack.v02081, i32 %.cvtpack.p12080, i32 1, !dbg !26
  %615 = bitcast <2 x i32> %.cvtpack.v12082 to <4 x half>, !dbg !26
  %math.packed248 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %615) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %598, <4 x float> %576) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %617 = shufflevector <8 x half> %516, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %618 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %617, <4 x float> %578) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %619 = extractelement <4 x float> %lazy.score2040, i64 0, !dbg !24
  %620 = call float @llvm.amdgcn.exp2.f32(float %619), !dbg !24
  %621 = fadd float %k0.sum.anchor1857, %620, !dbg !25
  %k0.sum.anchor1865 = call float asm sideeffect "", "=v,0"(float %621) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %622 = insertelement <4 x float> poison, float %620, i64 0, !dbg !26
  %623 = extractelement <4 x float> %lazy.score2040, i64 1, !dbg !24
  %624 = call float @llvm.amdgcn.exp2.f32(float %623), !dbg !24
  %625 = fadd float %k0.sum.anchor1858, %624, !dbg !25
  %k0.sum.anchor1866 = call float asm sideeffect "", "=v,0"(float %625) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %626 = insertelement <4 x float> %622, float %624, i64 1, !dbg !26
  %627 = extractelement <4 x float> %lazy.score2040, i64 2, !dbg !24
  %628 = call float @llvm.amdgcn.exp2.f32(float %627), !dbg !24
  %629 = fadd float %k0.sum.anchor1859, %628, !dbg !25
  %k0.sum.anchor1867 = call float asm sideeffect "", "=v,0"(float %629) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %630 = insertelement <4 x float> %626, float %628, i64 2, !dbg !26
  %631 = extractelement <4 x float> %lazy.score2040, i64 3, !dbg !24
  %632 = call float @llvm.amdgcn.exp2.f32(float %631), !dbg !24
  %633 = fadd float %k0.sum.anchor1860, %632, !dbg !25
  %k0.sum.anchor1868 = call float asm sideeffect "", "=v,0"(float %633) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %634 = insertelement <4 x float> %630, float %632, i64 3, !dbg !26
  %math.result263 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02083 = extractelement <4 x float> %math.result263, i32 0, !dbg !26
  %.cvtpack.f12084 = extractelement <4 x float> %math.result263, i32 1, !dbg !26
  %.cvtpack.f22085 = extractelement <4 x float> %math.result263, i32 2, !dbg !26
  %.cvtpack.f32086 = extractelement <4 x float> %math.result263, i32 3, !dbg !26
  %.cvtpack.p02087 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02083, float %.cvtpack.f12084), !dbg !26
  %.cvtpack.p12088 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22085, float %.cvtpack.f32086), !dbg !26
  %.cvtpack.v02089 = insertelement <2 x i32> poison, i32 %.cvtpack.p02087, i32 0, !dbg !26
  %.cvtpack.v12090 = insertelement <2 x i32> %.cvtpack.v02089, i32 %.cvtpack.p12088, i32 1, !dbg !26
  %635 = bitcast <2 x i32> %.cvtpack.v12090 to <4 x half>, !dbg !26
  %math.packed264 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %598, <4 x float> %596) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %617, <4 x float> %597) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %638 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %639 = extractelement <4 x float> %lazy.score2033, i64 0, !dbg !24
  %640 = call float @llvm.amdgcn.exp2.f32(float %639), !dbg !24
  %641 = fadd float %k0.sum.anchor1861, %640, !dbg !25
  %k0.sum.anchor1869 = call float asm sideeffect "", "=v,0"(float %641) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %642 = insertelement <4 x float> poison, float %640, i64 0, !dbg !26
  %643 = extractelement <4 x float> %lazy.score2033, i64 1, !dbg !24
  %644 = call float @llvm.amdgcn.exp2.f32(float %643), !dbg !24
  %645 = fadd float %k0.sum.anchor1862, %644, !dbg !25
  %k0.sum.anchor1870 = call float asm sideeffect "", "=v,0"(float %645) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %646 = insertelement <4 x float> %642, float %644, i64 1, !dbg !26
  %647 = extractelement <4 x float> %lazy.score2033, i64 2, !dbg !24
  %648 = call float @llvm.amdgcn.exp2.f32(float %647), !dbg !24
  %649 = fadd float %k0.sum.anchor1863, %648, !dbg !25
  %k0.sum.anchor1871 = call float asm sideeffect "", "=v,0"(float %649) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %650 = insertelement <4 x float> %646, float %648, i64 2, !dbg !26
  %651 = extractelement <4 x float> %lazy.score2033, i64 3, !dbg !24
  %652 = call float @llvm.amdgcn.exp2.f32(float %651), !dbg !24
  %653 = fadd float %k0.sum.anchor1864, %652, !dbg !25
  %k0.sum.anchor1872 = call float asm sideeffect "", "=v,0"(float %653) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %654 = insertelement <4 x float> %650, float %652, i64 3, !dbg !26
  %math.result249 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %654) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02091 = extractelement <4 x float> %math.result249, i32 0, !dbg !26
  %.cvtpack.f12092 = extractelement <4 x float> %math.result249, i32 1, !dbg !26
  %.cvtpack.f22093 = extractelement <4 x float> %math.result249, i32 2, !dbg !26
  %.cvtpack.f32094 = extractelement <4 x float> %math.result249, i32 3, !dbg !26
  %.cvtpack.p02095 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02091, float %.cvtpack.f12092), !dbg !26
  %.cvtpack.p12096 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22093, float %.cvtpack.f32094), !dbg !26
  %.cvtpack.v02097 = insertelement <2 x i32> poison, i32 %.cvtpack.p02095, i32 0, !dbg !26
  %.cvtpack.v12098 = insertelement <2 x i32> %.cvtpack.v02097, i32 %.cvtpack.p12096, i32 1, !dbg !26
  %655 = bitcast <2 x i32> %.cvtpack.v12098 to <4 x half>, !dbg !26
  %math.packed250 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %655) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %638, <4 x float> %616) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %657 = shufflevector <8 x half> %518, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %658 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %657, <4 x float> %618) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = extractelement <4 x float> %lazy.score2041, i64 0, !dbg !24
  %660 = call float @llvm.amdgcn.exp2.f32(float %659), !dbg !24
  %661 = fadd float %k0.sum.anchor1865, %660, !dbg !25
  %k0.sum.anchor1873 = call float asm sideeffect "", "=v,0"(float %661) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %662 = insertelement <4 x float> poison, float %660, i64 0, !dbg !26
  %663 = extractelement <4 x float> %lazy.score2041, i64 1, !dbg !24
  %664 = call float @llvm.amdgcn.exp2.f32(float %663), !dbg !24
  %665 = fadd float %k0.sum.anchor1866, %664, !dbg !25
  %k0.sum.anchor1874 = call float asm sideeffect "", "=v,0"(float %665) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %666 = insertelement <4 x float> %662, float %664, i64 1, !dbg !26
  %667 = extractelement <4 x float> %lazy.score2041, i64 2, !dbg !24
  %668 = call float @llvm.amdgcn.exp2.f32(float %667), !dbg !24
  %669 = fadd float %k0.sum.anchor1867, %668, !dbg !25
  %k0.sum.anchor1875 = call float asm sideeffect "", "=v,0"(float %669) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %670 = insertelement <4 x float> %666, float %668, i64 2, !dbg !26
  %671 = extractelement <4 x float> %lazy.score2041, i64 3, !dbg !24
  %672 = call float @llvm.amdgcn.exp2.f32(float %671), !dbg !24
  %673 = fadd float %k0.sum.anchor1868, %672, !dbg !25
  %k0.sum.anchor1876 = call float asm sideeffect "", "=v,0"(float %673) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %674 = insertelement <4 x float> %670, float %672, i64 3, !dbg !26
  %math.result265 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %674) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02099 = extractelement <4 x float> %math.result265, i32 0, !dbg !26
  %.cvtpack.f12100 = extractelement <4 x float> %math.result265, i32 1, !dbg !26
  %.cvtpack.f22101 = extractelement <4 x float> %math.result265, i32 2, !dbg !26
  %.cvtpack.f32102 = extractelement <4 x float> %math.result265, i32 3, !dbg !26
  %.cvtpack.p02103 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02099, float %.cvtpack.f12100), !dbg !26
  %.cvtpack.p12104 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22101, float %.cvtpack.f32102), !dbg !26
  %.cvtpack.v02105 = insertelement <2 x i32> poison, i32 %.cvtpack.p02103, i32 0, !dbg !26
  %.cvtpack.v12106 = insertelement <2 x i32> %.cvtpack.v02105, i32 %.cvtpack.p12104, i32 1, !dbg !26
  %675 = bitcast <2 x i32> %.cvtpack.v12106 to <4 x half>, !dbg !26
  %math.packed266 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %675) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %676 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %638, <4 x float> %636) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %657, <4 x float> %637) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %678 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %679 = extractelement <4 x float> %lazy.score2034, i64 0, !dbg !24
  %680 = call float @llvm.amdgcn.exp2.f32(float %679), !dbg !24
  %681 = fadd float %k0.sum.anchor1869, %680, !dbg !25
  %k0.sum.anchor1877 = call float asm sideeffect "", "=v,0"(float %681) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %682 = insertelement <4 x float> poison, float %680, i64 0, !dbg !26
  %683 = extractelement <4 x float> %lazy.score2034, i64 1, !dbg !24
  %684 = call float @llvm.amdgcn.exp2.f32(float %683), !dbg !24
  %685 = fadd float %k0.sum.anchor1870, %684, !dbg !25
  %k0.sum.anchor1878 = call float asm sideeffect "", "=v,0"(float %685) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %686 = insertelement <4 x float> %682, float %684, i64 1, !dbg !26
  %687 = extractelement <4 x float> %lazy.score2034, i64 2, !dbg !24
  %688 = call float @llvm.amdgcn.exp2.f32(float %687), !dbg !24
  %689 = fadd float %k0.sum.anchor1871, %688, !dbg !25
  %k0.sum.anchor1879 = call float asm sideeffect "", "=v,0"(float %689) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %690 = insertelement <4 x float> %686, float %688, i64 2, !dbg !26
  %691 = extractelement <4 x float> %lazy.score2034, i64 3, !dbg !24
  %692 = call float @llvm.amdgcn.exp2.f32(float %691), !dbg !24
  %693 = fadd float %k0.sum.anchor1872, %692, !dbg !25
  %k0.sum.anchor1880 = call float asm sideeffect "", "=v,0"(float %693) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %694 = insertelement <4 x float> %690, float %692, i64 3, !dbg !26
  %math.result251 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %694) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02107 = extractelement <4 x float> %math.result251, i32 0, !dbg !26
  %.cvtpack.f12108 = extractelement <4 x float> %math.result251, i32 1, !dbg !26
  %.cvtpack.f22109 = extractelement <4 x float> %math.result251, i32 2, !dbg !26
  %.cvtpack.f32110 = extractelement <4 x float> %math.result251, i32 3, !dbg !26
  %.cvtpack.p02111 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02107, float %.cvtpack.f12108), !dbg !26
  %.cvtpack.p12112 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22109, float %.cvtpack.f32110), !dbg !26
  %.cvtpack.v02113 = insertelement <2 x i32> poison, i32 %.cvtpack.p02111, i32 0, !dbg !26
  %.cvtpack.v12114 = insertelement <2 x i32> %.cvtpack.v02113, i32 %.cvtpack.p12112, i32 1, !dbg !26
  %695 = bitcast <2 x i32> %.cvtpack.v12114 to <4 x half>, !dbg !26
  %math.packed252 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %695) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %696 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %678, <4 x float> %656) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %697 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %698 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %697, <4 x float> %658) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %699 = extractelement <4 x float> %lazy.score2042, i64 0, !dbg !24
  %700 = call float @llvm.amdgcn.exp2.f32(float %699), !dbg !24
  %701 = fadd float %k0.sum.anchor1873, %700, !dbg !25
  %k0.sum.anchor1881 = call float asm sideeffect "", "=v,0"(float %701) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %702 = insertelement <4 x float> poison, float %700, i64 0, !dbg !26
  %703 = extractelement <4 x float> %lazy.score2042, i64 1, !dbg !24
  %704 = call float @llvm.amdgcn.exp2.f32(float %703), !dbg !24
  %705 = fadd float %k0.sum.anchor1874, %704, !dbg !25
  %k0.sum.anchor1882 = call float asm sideeffect "", "=v,0"(float %705) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %706 = insertelement <4 x float> %702, float %704, i64 1, !dbg !26
  %707 = extractelement <4 x float> %lazy.score2042, i64 2, !dbg !24
  %708 = call float @llvm.amdgcn.exp2.f32(float %707), !dbg !24
  %709 = fadd float %k0.sum.anchor1875, %708, !dbg !25
  %k0.sum.anchor1883 = call float asm sideeffect "", "=v,0"(float %709) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %710 = insertelement <4 x float> %706, float %708, i64 2, !dbg !26
  %711 = extractelement <4 x float> %lazy.score2042, i64 3, !dbg !24
  %712 = call float @llvm.amdgcn.exp2.f32(float %711), !dbg !24
  %713 = fadd float %k0.sum.anchor1876, %712, !dbg !25
  %k0.sum.anchor1884 = call float asm sideeffect "", "=v,0"(float %713) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %714 = insertelement <4 x float> %710, float %712, i64 3, !dbg !26
  %math.result267 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %714) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02115 = extractelement <4 x float> %math.result267, i32 0, !dbg !26
  %.cvtpack.f12116 = extractelement <4 x float> %math.result267, i32 1, !dbg !26
  %.cvtpack.f22117 = extractelement <4 x float> %math.result267, i32 2, !dbg !26
  %.cvtpack.f32118 = extractelement <4 x float> %math.result267, i32 3, !dbg !26
  %.cvtpack.p02119 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02115, float %.cvtpack.f12116), !dbg !26
  %.cvtpack.p12120 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22117, float %.cvtpack.f32118), !dbg !26
  %.cvtpack.v02121 = insertelement <2 x i32> poison, i32 %.cvtpack.p02119, i32 0, !dbg !26
  %.cvtpack.v12122 = insertelement <2 x i32> %.cvtpack.v02121, i32 %.cvtpack.p12120, i32 1, !dbg !26
  %715 = bitcast <2 x i32> %.cvtpack.v12122 to <4 x half>, !dbg !26
  %math.packed268 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %715) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %716 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %678, <4 x float> %676) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %697, <4 x float> %677) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %718 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %719 = extractelement <4 x float> %lazy.score2035, i64 0, !dbg !24
  %720 = call float @llvm.amdgcn.exp2.f32(float %719), !dbg !24
  %721 = fadd float %k0.sum.anchor1877, %720, !dbg !25
  %k0.sum.anchor1885 = call float asm sideeffect "", "=v,0"(float %721) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %722 = insertelement <4 x float> poison, float %720, i64 0, !dbg !26
  %723 = extractelement <4 x float> %lazy.score2035, i64 1, !dbg !24
  %724 = call float @llvm.amdgcn.exp2.f32(float %723), !dbg !24
  %725 = fadd float %k0.sum.anchor1878, %724, !dbg !25
  %k0.sum.anchor1886 = call float asm sideeffect "", "=v,0"(float %725) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %726 = insertelement <4 x float> %722, float %724, i64 1, !dbg !26
  %727 = extractelement <4 x float> %lazy.score2035, i64 2, !dbg !24
  %728 = call float @llvm.amdgcn.exp2.f32(float %727), !dbg !24
  %729 = fadd float %k0.sum.anchor1879, %728, !dbg !25
  %k0.sum.anchor1887 = call float asm sideeffect "", "=v,0"(float %729) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %730 = insertelement <4 x float> %726, float %728, i64 2, !dbg !26
  %731 = extractelement <4 x float> %lazy.score2035, i64 3, !dbg !24
  %732 = call float @llvm.amdgcn.exp2.f32(float %731), !dbg !24
  %733 = fadd float %k0.sum.anchor1880, %732, !dbg !25
  %k0.sum.anchor1888 = call float asm sideeffect "", "=v,0"(float %733) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %734 = insertelement <4 x float> %730, float %732, i64 3, !dbg !26
  %math.result253 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %734) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02123 = extractelement <4 x float> %math.result253, i32 0, !dbg !26
  %.cvtpack.f12124 = extractelement <4 x float> %math.result253, i32 1, !dbg !26
  %.cvtpack.f22125 = extractelement <4 x float> %math.result253, i32 2, !dbg !26
  %.cvtpack.f32126 = extractelement <4 x float> %math.result253, i32 3, !dbg !26
  %.cvtpack.p02127 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02123, float %.cvtpack.f12124), !dbg !26
  %.cvtpack.p12128 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22125, float %.cvtpack.f32126), !dbg !26
  %.cvtpack.v02129 = insertelement <2 x i32> poison, i32 %.cvtpack.p02127, i32 0, !dbg !26
  %.cvtpack.v12130 = insertelement <2 x i32> %.cvtpack.v02129, i32 %.cvtpack.p12128, i32 1, !dbg !26
  %735 = bitcast <2 x i32> %.cvtpack.v12130 to <4 x half>, !dbg !26
  %math.packed254 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %735) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %736 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %718, <4 x float> %696) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %737 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %738 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %737, <4 x float> %698) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %739 = extractelement <4 x float> %lazy.score2043, i64 0, !dbg !24
  %740 = call float @llvm.amdgcn.exp2.f32(float %739), !dbg !24
  %741 = fadd float %k0.sum.anchor1881, %740, !dbg !25
  %k0.sum.anchor1889 = call float asm sideeffect "", "=v,0"(float %741) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %742 = insertelement <4 x float> poison, float %740, i64 0, !dbg !26
  %743 = extractelement <4 x float> %lazy.score2043, i64 1, !dbg !24
  %744 = call float @llvm.amdgcn.exp2.f32(float %743), !dbg !24
  %745 = fadd float %k0.sum.anchor1882, %744, !dbg !25
  %k0.sum.anchor1890 = call float asm sideeffect "", "=v,0"(float %745) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %746 = insertelement <4 x float> %742, float %744, i64 1, !dbg !26
  %747 = extractelement <4 x float> %lazy.score2043, i64 2, !dbg !24
  %748 = call float @llvm.amdgcn.exp2.f32(float %747), !dbg !24
  %749 = fadd float %k0.sum.anchor1883, %748, !dbg !25
  %k0.sum.anchor1891 = call float asm sideeffect "", "=v,0"(float %749) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %750 = insertelement <4 x float> %746, float %748, i64 2, !dbg !26
  %751 = extractelement <4 x float> %lazy.score2043, i64 3, !dbg !24
  %752 = call float @llvm.amdgcn.exp2.f32(float %751), !dbg !24
  %753 = fadd float %k0.sum.anchor1884, %752, !dbg !25
  %k0.sum.anchor1892 = call float asm sideeffect "", "=v,0"(float %753) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %754 = insertelement <4 x float> %750, float %752, i64 3, !dbg !26
  %math.result269 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %754) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02131 = extractelement <4 x float> %math.result269, i32 0, !dbg !26
  %.cvtpack.f12132 = extractelement <4 x float> %math.result269, i32 1, !dbg !26
  %.cvtpack.f22133 = extractelement <4 x float> %math.result269, i32 2, !dbg !26
  %.cvtpack.f32134 = extractelement <4 x float> %math.result269, i32 3, !dbg !26
  %.cvtpack.p02135 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02131, float %.cvtpack.f12132), !dbg !26
  %.cvtpack.p12136 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22133, float %.cvtpack.f32134), !dbg !26
  %.cvtpack.v02137 = insertelement <2 x i32> poison, i32 %.cvtpack.p02135, i32 0, !dbg !26
  %.cvtpack.v12138 = insertelement <2 x i32> %.cvtpack.v02137, i32 %.cvtpack.p12136, i32 1, !dbg !26
  %755 = bitcast <2 x i32> %.cvtpack.v12138 to <4 x half>, !dbg !26
  %math.packed270 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %755) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %756 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %718, <4 x float> %716) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %737, <4 x float> %717) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %758 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = extractelement <4 x float> %lazy.score2036, i64 0, !dbg !24
  %760 = call float @llvm.amdgcn.exp2.f32(float %759), !dbg !24
  %761 = fadd float %k0.sum.anchor1885, %760, !dbg !25
  %k0.sum.anchor1893 = call float asm sideeffect "", "=v,0"(float %761) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %762 = insertelement <4 x float> poison, float %760, i64 0, !dbg !26
  %763 = extractelement <4 x float> %lazy.score2036, i64 1, !dbg !24
  %764 = call float @llvm.amdgcn.exp2.f32(float %763), !dbg !24
  %765 = fadd float %k0.sum.anchor1886, %764, !dbg !25
  %k0.sum.anchor1894 = call float asm sideeffect "", "=v,0"(float %765) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %766 = insertelement <4 x float> %762, float %764, i64 1, !dbg !26
  %767 = extractelement <4 x float> %lazy.score2036, i64 2, !dbg !24
  %768 = call float @llvm.amdgcn.exp2.f32(float %767), !dbg !24
  %769 = fadd float %k0.sum.anchor1887, %768, !dbg !25
  %k0.sum.anchor1895 = call float asm sideeffect "", "=v,0"(float %769) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %770 = insertelement <4 x float> %766, float %768, i64 2, !dbg !26
  %771 = extractelement <4 x float> %lazy.score2036, i64 3, !dbg !24
  %772 = call float @llvm.amdgcn.exp2.f32(float %771), !dbg !24
  %773 = fadd float %k0.sum.anchor1888, %772, !dbg !25
  %k0.sum.anchor1896 = call float asm sideeffect "", "=v,0"(float %773) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %774 = insertelement <4 x float> %770, float %772, i64 3, !dbg !26
  %math.result255 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %774) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02139 = extractelement <4 x float> %math.result255, i32 0, !dbg !26
  %.cvtpack.f12140 = extractelement <4 x float> %math.result255, i32 1, !dbg !26
  %.cvtpack.f22141 = extractelement <4 x float> %math.result255, i32 2, !dbg !26
  %.cvtpack.f32142 = extractelement <4 x float> %math.result255, i32 3, !dbg !26
  %.cvtpack.p02143 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02139, float %.cvtpack.f12140), !dbg !26
  %.cvtpack.p12144 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22141, float %.cvtpack.f32142), !dbg !26
  %.cvtpack.v02145 = insertelement <2 x i32> poison, i32 %.cvtpack.p02143, i32 0, !dbg !26
  %.cvtpack.v12146 = insertelement <2 x i32> %.cvtpack.v02145, i32 %.cvtpack.p12144, i32 1, !dbg !26
  %775 = bitcast <2 x i32> %.cvtpack.v12146 to <4 x half>, !dbg !26
  %math.packed256 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %775) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %776 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %758, <4 x float> %736) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %777 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %777, <4 x float> %738) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = extractelement <4 x float> %lazy.score2044, i64 0, !dbg !24
  %780 = call float @llvm.amdgcn.exp2.f32(float %779), !dbg !24
  %781 = fadd float %k0.sum.anchor1889, %780, !dbg !25
  %k0.sum.anchor1897 = call float asm sideeffect "", "=v,0"(float %781) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %782 = insertelement <4 x float> poison, float %780, i64 0, !dbg !26
  %783 = extractelement <4 x float> %lazy.score2044, i64 1, !dbg !24
  %784 = call float @llvm.amdgcn.exp2.f32(float %783), !dbg !24
  %785 = fadd float %k0.sum.anchor1890, %784, !dbg !25
  %k0.sum.anchor1898 = call float asm sideeffect "", "=v,0"(float %785) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %786 = insertelement <4 x float> %782, float %784, i64 1, !dbg !26
  %787 = extractelement <4 x float> %lazy.score2044, i64 2, !dbg !24
  %788 = call float @llvm.amdgcn.exp2.f32(float %787), !dbg !24
  %789 = fadd float %k0.sum.anchor1891, %788, !dbg !25
  %k0.sum.anchor1899 = call float asm sideeffect "", "=v,0"(float %789) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %790 = insertelement <4 x float> %786, float %788, i64 2, !dbg !26
  %791 = extractelement <4 x float> %lazy.score2044, i64 3, !dbg !24
  %792 = call float @llvm.amdgcn.exp2.f32(float %791), !dbg !24
  %793 = fadd float %k0.sum.anchor1892, %792, !dbg !25
  %k0.sum.anchor1900 = call float asm sideeffect "", "=v,0"(float %793) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %794 = insertelement <4 x float> %790, float %792, i64 3, !dbg !26
  %math.result271 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02147 = extractelement <4 x float> %math.result271, i32 0, !dbg !26
  %.cvtpack.f12148 = extractelement <4 x float> %math.result271, i32 1, !dbg !26
  %.cvtpack.f22149 = extractelement <4 x float> %math.result271, i32 2, !dbg !26
  %.cvtpack.f32150 = extractelement <4 x float> %math.result271, i32 3, !dbg !26
  %.cvtpack.p02151 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02147, float %.cvtpack.f12148), !dbg !26
  %.cvtpack.p12152 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22149, float %.cvtpack.f32150), !dbg !26
  %.cvtpack.v02153 = insertelement <2 x i32> poison, i32 %.cvtpack.p02151, i32 0, !dbg !26
  %.cvtpack.v12154 = insertelement <2 x i32> %.cvtpack.v02153, i32 %.cvtpack.p12152, i32 1, !dbg !26
  %795 = bitcast <2 x i32> %.cvtpack.v12154 to <4 x half>, !dbg !26
  %math.packed272 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %796 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %758, <4 x float> %756) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %777, <4 x float> %757) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = bitcast <4 x i32> %525 to <8 x half>, !dbg !14
  %799 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %800 = extractelement <4 x float> %lazy.score2037, i64 0, !dbg !24
  %801 = call float @llvm.amdgcn.exp2.f32(float %800), !dbg !24
  %802 = fadd float %k0.sum.anchor1893, %801, !dbg !25
  %k0.sum.anchor1901 = call float asm sideeffect "", "=v,0"(float %802) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %803 = insertelement <4 x float> poison, float %801, i64 0, !dbg !26
  %804 = extractelement <4 x float> %lazy.score2037, i64 1, !dbg !24
  %805 = call float @llvm.amdgcn.exp2.f32(float %804), !dbg !24
  %806 = fadd float %k0.sum.anchor1894, %805, !dbg !25
  %k0.sum.anchor1902 = call float asm sideeffect "", "=v,0"(float %806) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %807 = fadd float %k0.sum.anchor1901, %k0.sum.anchor1902, !dbg !25
  %k0.sum.anchor1903 = call float asm sideeffect "", "=v,0"(float %807) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %808 = insertelement <4 x float> %803, float %805, i64 1, !dbg !26
  %809 = extractelement <4 x float> %lazy.score2037, i64 2, !dbg !24
  %810 = call float @llvm.amdgcn.exp2.f32(float %809), !dbg !24
  %811 = fadd float %k0.sum.anchor1895, %810, !dbg !25
  %k0.sum.anchor1904 = call float asm sideeffect "", "=v,0"(float %811) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %812 = insertelement <4 x float> %808, float %810, i64 2, !dbg !26
  %813 = extractelement <4 x float> %lazy.score2037, i64 3, !dbg !24
  %814 = call float @llvm.amdgcn.exp2.f32(float %813), !dbg !24
  %815 = fadd float %k0.sum.anchor1896, %814, !dbg !25
  %k0.sum.anchor1905 = call float asm sideeffect "", "=v,0"(float %815) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %816 = fadd float %k0.sum.anchor1904, %k0.sum.anchor1905, !dbg !25
  %k0.sum.anchor1906 = call float asm sideeffect "", "=v,0"(float %816) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %817 = fadd float %k0.sum.anchor1903, %k0.sum.anchor1906, !dbg !25
  %k0.sum.anchor1907 = call float asm sideeffect "", "=v,0"(float %817) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %818 = fadd float %k0.sum.anchor1907, %500, !dbg !25
  %k0.sum.anchor1908 = call float asm sideeffect "", "=v,0"(float %818) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %819 = insertelement <1 x float> poison, float %k0.sum.anchor1908, i64 0, !dbg !6
  %820 = insertvalue [2 x <1 x float>] poison, <1 x float> %819, 0, !dbg !6
  %821 = insertelement <4 x float> %812, float %814, i64 3, !dbg !26
  %math.result257 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %821) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02155 = extractelement <4 x float> %math.result257, i32 0, !dbg !26
  %.cvtpack.f12156 = extractelement <4 x float> %math.result257, i32 1, !dbg !26
  %.cvtpack.f22157 = extractelement <4 x float> %math.result257, i32 2, !dbg !26
  %.cvtpack.f32158 = extractelement <4 x float> %math.result257, i32 3, !dbg !26
  %.cvtpack.p02159 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02155, float %.cvtpack.f12156), !dbg !26
  %.cvtpack.p12160 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22157, float %.cvtpack.f32158), !dbg !26
  %.cvtpack.v02161 = insertelement <2 x i32> poison, i32 %.cvtpack.p02159, i32 0, !dbg !26
  %.cvtpack.v12162 = insertelement <2 x i32> %.cvtpack.v02161, i32 %.cvtpack.p12160, i32 1, !dbg !26
  %822 = bitcast <2 x i32> %.cvtpack.v12162 to <4 x half>, !dbg !26
  %math.packed258 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %822) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %823 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %799, <4 x float> %776) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %824 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %824, <4 x float> %778) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = extractelement <4 x float> %lazy.score2045, i64 0, !dbg !24
  %827 = call float @llvm.amdgcn.exp2.f32(float %826), !dbg !24
  %828 = fadd float %k0.sum.anchor1897, %827, !dbg !25
  %k0.sum.anchor1909 = call float asm sideeffect "", "=v,0"(float %828) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %829 = insertelement <4 x float> poison, float %827, i64 0, !dbg !26
  %830 = extractelement <4 x float> %lazy.score2045, i64 1, !dbg !24
  %831 = call float @llvm.amdgcn.exp2.f32(float %830), !dbg !24
  %832 = fadd float %k0.sum.anchor1898, %831, !dbg !25
  %k0.sum.anchor1910 = call float asm sideeffect "", "=v,0"(float %832) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %833 = fadd float %k0.sum.anchor1909, %k0.sum.anchor1910, !dbg !25
  %k0.sum.anchor1911 = call float asm sideeffect "", "=v,0"(float %833) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %834 = insertelement <4 x float> %829, float %831, i64 1, !dbg !26
  %835 = extractelement <4 x float> %lazy.score2045, i64 2, !dbg !24
  %836 = call float @llvm.amdgcn.exp2.f32(float %835), !dbg !24
  %837 = fadd float %k0.sum.anchor1899, %836, !dbg !25
  %k0.sum.anchor1912 = call float asm sideeffect "", "=v,0"(float %837) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %838 = insertelement <4 x float> %834, float %836, i64 2, !dbg !26
  %839 = extractelement <4 x float> %lazy.score2045, i64 3, !dbg !24
  %840 = call float @llvm.amdgcn.exp2.f32(float %839), !dbg !24
  %841 = fadd float %k0.sum.anchor1900, %840, !dbg !25
  %k0.sum.anchor1913 = call float asm sideeffect "", "=v,0"(float %841) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %842 = fadd float %k0.sum.anchor1912, %k0.sum.anchor1913, !dbg !25
  %k0.sum.anchor1914 = call float asm sideeffect "", "=v,0"(float %842) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %843 = fadd float %k0.sum.anchor1911, %k0.sum.anchor1914, !dbg !25
  %k0.sum.anchor1915 = call float asm sideeffect "", "=v,0"(float %843) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %844 = fadd float %k0.sum.anchor1915, %502, !dbg !25
  %k0.sum.anchor1916 = call float asm sideeffect "", "=v,0"(float %844) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %845 = insertelement <1 x float> poison, float %k0.sum.anchor1916, i64 0, !dbg !6
  %846 = insertvalue [2 x <1 x float>] %820, <1 x float> %845, 1, !dbg !6
  %847 = insertelement <4 x float> %838, float %840, i64 3, !dbg !26
  %math.result273 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %847) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02163 = extractelement <4 x float> %math.result273, i32 0, !dbg !26
  %.cvtpack.f12164 = extractelement <4 x float> %math.result273, i32 1, !dbg !26
  %.cvtpack.f22165 = extractelement <4 x float> %math.result273, i32 2, !dbg !26
  %.cvtpack.f32166 = extractelement <4 x float> %math.result273, i32 3, !dbg !26
  %.cvtpack.p02167 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02163, float %.cvtpack.f12164), !dbg !26
  %.cvtpack.p12168 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22165, float %.cvtpack.f32166), !dbg !26
  %.cvtpack.v02169 = insertelement <2 x i32> poison, i32 %.cvtpack.p02167, i32 0, !dbg !26
  %.cvtpack.v12170 = insertelement <2 x i32> %.cvtpack.v02169, i32 %.cvtpack.p12168, i32 1, !dbg !26
  %848 = bitcast <2 x i32> %.cvtpack.v12170 to <4 x half>, !dbg !26
  %math.packed274 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %848) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %849 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %799, <4 x float> %796) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %824, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %851 = extractvalue [2 x <8 x float>] %lazy.state2049, 0, !dbg !27
  %852 = shufflevector <8 x float> %851, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %853 = shufflevector <8 x float> %851, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %854 = extractvalue [2 x <8 x float>] %lazy.state2049, 1, !dbg !27
  %855 = shufflevector <8 x float> %854, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %856 = shufflevector <8 x float> %854, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1318 = add nsw i64 %195, %address.local793, !dbg !16
  %vdb.address1319 = trunc i64 %vdb.address1318 to i32, !dbg !16
  %vdb.address1320 = shl i32 %vdb.address1319, 1, !dbg !16
  %vdb.address1325 = trunc nuw i64 %195 to i32, !dbg !16
  %vdb.address1326 = or disjoint i32 %address.local800, %vdb.address1325, !dbg !16
  %vdb.address1327 = shl nuw nsw i32 %vdb.address1326, 1, !dbg !16
  %857 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %vdb.address1320, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %858 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %vdb.address1327, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local809, i64 8192, !dbg !14
  %859 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %860 = bitcast <4 x i32> %859 to <8 x half>, !dbg !14
  %vdb.read1311 = getelementptr i8, ptr addrspace(3) %address.local818, i64 8448, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1311, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %vdb.read1312 = getelementptr i8, ptr addrspace(3) %address.local818, i64 8704, !dbg !14
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1312, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !14
  %vdb.read1313 = getelementptr i8, ptr addrspace(3) %address.local818, i64 8960, !dbg !14
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1313, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !14
  %vdb.read1314 = getelementptr i8, ptr addrspace(3) %address.local809, i64 12288, !dbg !14
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1314, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !14
  %vdb.read1315 = getelementptr i8, ptr addrspace(3) %address.local818, i64 12544, !dbg !14
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1315, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !14
  %vdb.read1316 = getelementptr i8, ptr addrspace(3) %address.local818, i64 12800, !dbg !14
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1316, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !14
  %vdb.read1317 = getelementptr i8, ptr addrspace(3) %address.local818, i64 13056, !dbg !14
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1317, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %874 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %875 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %874, <4 x float> %852) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %876 = shufflevector <8 x half> %860, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %876, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %874, <4 x float> %855) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %876, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %880, <4 x float> %875) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %882, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %880, <4 x float> %878) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %882, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %886, <4 x float> %881) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %888, <4 x float> %883) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %886, <4 x float> %884) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %888, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %892, <4 x float> %887) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %894, <4 x float> %889) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %892, <4 x float> %890) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %894, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %898, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %900, <4 x float> %895) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %898, <4 x float> %896) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %900, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %904, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %906, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %904, <4 x float> %902) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %906, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %910, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %912, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %910, <4 x float> %908) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %912, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = bitcast <4 x i32> %873 to <8 x half>, !dbg !14
  %917 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %917, <4 x float> %911) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %919 = shufflevector <8 x half> %916, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %919, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %917, <4 x float> %914) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %919, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %923 = extractvalue [2 x <8 x float>] %lazy.state2048, 0, !dbg !27
  %924 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %925 = shufflevector <8 x float> %923, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %926 = extractvalue [2 x <8 x float>] %lazy.state2048, 1, !dbg !27
  %927 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %928 = shufflevector <8 x float> %926, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %vdb.address1332 = add nsw i64 %248, %address.local793, !dbg !16
  %vdb.address1333 = trunc i64 %vdb.address1332 to i32, !dbg !16
  %vdb.address1334 = shl i32 %vdb.address1333, 1, !dbg !16
  %vdb.address1339 = trunc nuw i64 %248 to i32, !dbg !16
  %vdb.address1340 = or disjoint i32 %address.local800, %vdb.address1339, !dbg !16
  %vdb.address1341 = shl nuw nsw i32 %vdb.address1340, 1, !dbg !16
  %929 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1334, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %930 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1310, i32 %vdb.address1341, <4 x i32> %dma.buffer.rsrc1217) #10, !dbg !16
  %931 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local809, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %932 = bitcast <4 x i32> %931 to <8 x half>, !dbg !14
  %933 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local819, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %934 = bitcast <4 x i32> %933 to <8 x half>, !dbg !14
  %935 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local823, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %936 = bitcast <4 x i32> %935 to <8 x half>, !dbg !14
  %937 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local827, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %938 = bitcast <4 x i32> %937 to <8 x half>, !dbg !14
  %939 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local829, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %940 = bitcast <4 x i32> %939 to <8 x half>, !dbg !14
  %941 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local833, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %942 = bitcast <4 x i32> %941 to <8 x half>, !dbg !14
  %943 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local837, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %944 = bitcast <4 x i32> %943 to <8 x half>, !dbg !14
  %945 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local841, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %946 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %947 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %946, <4 x float> %924) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %948 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %949 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %948, <4 x float> %925) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %946, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %948, <4 x float> %928) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %952, <4 x float> %947) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %954, <4 x float> %949) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %952, <4 x float> %950) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %954, <4 x float> %951) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %958, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = shufflevector <8 x half> %936, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %960, <4 x float> %955) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %958, <4 x float> %956) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %960, <4 x float> %957) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %964, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = shufflevector <8 x half> %938, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %966, <4 x float> %961) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %964, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %966, <4 x float> %963) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %970, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = shufflevector <8 x half> %940, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %972, <4 x float> %967) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %970, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %972, <4 x float> %969) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %976, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = shufflevector <8 x half> %942, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %978, <4 x float> %973) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %976, <4 x float> %974) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %978, <4 x float> %975) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %982, <4 x float> %977) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = shufflevector <8 x half> %944, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %984, <4 x float> %979) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %982, <4 x float> %980) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %984, <4 x float> %981) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = bitcast <4 x i32> %945 to <8 x half>, !dbg !14
  %989 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %989, <4 x float> %983) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = shufflevector <8 x half> %988, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %992 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %991, <4 x float> %985) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %989, <4 x float> %986) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %991, <4 x float> %987) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %995 = extractvalue [2 x <8 x float>] %lazy.state2047, 0, !dbg !27
  %996 = shufflevector <8 x float> %995, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %997 = shufflevector <8 x float> %995, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %998 = extractvalue [2 x <8 x float>] %lazy.state2047, 1, !dbg !27
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
  %k0.next.1927 = or disjoint i64 %111, %k0.next., !dbg !13
  %k0.next.kdma.source1372 = or disjoint i64 %k0.next.1927, %kdma.source1371, !dbg !13
  %k0.next.kdma.source1373 = trunc nuw i64 %k0.next.kdma.source1372 to i32, !dbg !13
  %k0.next.kdma.source1374 = shl nuw nsw i32 %k0.next.kdma.source1373, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1374, %kdma.intra, !dbg !13
  %k0.next.1928 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1392 = or disjoint i64 %k0.next.1927, %kdma.source1391, !dbg !13
  %k0.next.kdma.source1393 = trunc nuw i64 %k0.next.kdma.source1392 to i32, !dbg !13
  %k0.next.kdma.source1394 = shl nuw nsw i32 %k0.next.kdma.source1393, 1, !dbg !13
  %k0.next.kdma.offset1395 = or disjoint i32 %k0.next.kdma.source1394, %kdma.intra.hi, !dbg !13
  %k0.next.1929 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1396, i32 %k0.next.kdma.offset1395, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1416 = or disjoint i64 %k0.next.1927, %kdma.source1415, !dbg !13
  %k0.next.kdma.source1417 = trunc nuw i64 %k0.next.kdma.source1416 to i32, !dbg !13
  %k0.next.kdma.source1418 = shl nuw nsw i32 %k0.next.kdma.source1417, 1, !dbg !13
  %k0.next.kdma.offset1419 = add nuw i32 %k0.next.kdma.source1418, %kdma.intra, !dbg !13
  %k0.next.1930 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.next.kdma.offset1419, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1440 = or disjoint i64 %k0.next.1927, %kdma.source1439, !dbg !13
  %k0.next.kdma.source1441 = trunc nuw i64 %k0.next.kdma.source1440 to i32, !dbg !13
  %k0.next.kdma.source1442 = shl nuw nsw i32 %k0.next.kdma.source1441, 1, !dbg !13
  %k0.next.kdma.offset1443 = add nuw i32 %k0.next.kdma.source1442, %kdma.intra.hi, !dbg !13
  %k0.next.1931 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1444, i32 %k0.next.kdma.offset1443, <4 x i32> %dma.buffer.rsrc1233) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %1002 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1003 = bitcast <4 x i32> %1002 to <8 x half>, !dbg !14
  %1004 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1311, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !14
  %1006 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1312, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !14
  %1008 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1313, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !14
  %1010 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1314, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !14
  %1012 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1315, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !14
  %1014 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1316, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !14
  %1016 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1317, i32 0) #0, !dbg !14, !frisk.v_double_buffer !32
  %1017 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1018 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1017, <4 x float> %996) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1019 = shufflevector <8 x half> %1003, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1020 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1019, <4 x float> %997) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1017, <4 x float> %999) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1022 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed260, <4 x half> %1019, <4 x float> %1000) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1023 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1023, <4 x float> %1018) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1025 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1026 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed246, <4 x half> %1025, <4 x float> %1020) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1023, <4 x float> %1021) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1028 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed262, <4 x half> %1025, <4 x float> %1022) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1029 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1029, <4 x float> %1024) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1031 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1032 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed248, <4 x half> %1031, <4 x float> %1026) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1029, <4 x float> %1027) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed264, <4 x half> %1031, <4 x float> %1028) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1035 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1035, <4 x float> %1030) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1037 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1038 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed250, <4 x half> %1037, <4 x float> %1032) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1035, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1040 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed266, <4 x half> %1037, <4 x float> %1034) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1041 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1041, <4 x float> %1036) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1043 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1044 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed252, <4 x half> %1043, <4 x float> %1038) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1041, <4 x float> %1039) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed268, <4 x half> %1043, <4 x float> %1040) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1047 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1047, <4 x float> %1042) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1049 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1050 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed254, <4 x half> %1049, <4 x float> %1044) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1047, <4 x float> %1045) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1052 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed270, <4 x half> %1049, <4 x float> %1046) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1053, <4 x float> %1048) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed256, <4 x half> %1055, <4 x float> %1050) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1053, <4 x float> %1051) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1058 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed272, <4 x half> %1055, <4 x float> %1052) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !14
  %1060 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1060, <4 x float> %1054) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1062 = shufflevector <8 x half> %1059, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1063 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed258, <4 x half> %1062, <4 x float> %1056) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1060, <4 x float> %1057) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1065 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed274, <4 x half> %1062, <4 x float> %1058) #9, !dbg !14, !frisk.pv_interleave !31
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
  %lane.sum.field43 = extractvalue [2 x <1 x float>] %103, 1, !dbg !6
  %lane.sum.local44 = extractelement <1 x float> %lane.sum.field43, i64 0, !dbg !6
  %1083 = shl nuw nsw i32 %8, 2, !dbg !6
  %1084 = xor i32 %1083, 64, !dbg !6
  %1085 = bitcast float %lane.sum.local44 to i32, !dbg !6
  %lane.sum.shuffle47 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1084, i32 %1085), !dbg !6
  %1086 = bitcast i32 %lane.sum.shuffle47 to float, !dbg !6
  %lane.sum.final48 = fadd float %lane.sum.local44, %1086, !dbg !6
  %1087 = xor i32 %1083, 128, !dbg !6
  %1088 = bitcast float %lane.sum.final48 to i32, !dbg !6
  %lane.sum.shuffle49 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1087, i32 %1088), !dbg !6
  %1089 = bitcast i32 %lane.sum.shuffle49 to float, !dbg !6
  %lane.sum.final50 = fadd float %lane.sum.final48, %1089, !dbg !6
  %1090 = insertelement <1 x float> poison, float %lane.sum.final50, i64 0, !dbg !6
  %lane.sum.field = extractvalue [2 x <1 x float>] %103, 0, !dbg !6
  %lane.sum.local = extractelement <1 x float> %lane.sum.field, i64 0, !dbg !6
  %1091 = bitcast float %lane.sum.local to i32, !dbg !6
  %lane.sum.shuffle = call i32 @llvm.amdgcn.ds.bpermute(i32 %1084, i32 %1091), !dbg !6
  %1092 = bitcast i32 %lane.sum.shuffle to float, !dbg !6
  %lane.sum.final = fadd float %lane.sum.local, %1092, !dbg !6
  %1093 = bitcast float %lane.sum.final to i32, !dbg !6
  %lane.sum.shuffle41 = call i32 @llvm.amdgcn.ds.bpermute(i32 %1087, i32 %1093), !dbg !6
  %1094 = bitcast i32 %lane.sum.shuffle41 to float, !dbg !6
  %lane.sum.final42 = fadd float %lane.sum.final, %1094, !dbg !6
  %1095 = insertelement <1 x float> poison, float %lane.sum.final42, i64 0, !dbg !6
  %1096 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1096, 0, !dbg !6
  %1097 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1097, 0, !dbg !6
  %1098 = extractvalue [2 x <8 x float>] %107, 0, !dbg !6
  %1099 = extractelement <8 x float> %1098, i64 0, !dbg !6
  %1100 = extractelement <8 x float> %1098, i64 1, !dbg !6
  %1101 = extractelement <8 x float> %1098, i64 2, !dbg !6
  %1102 = extractelement <8 x float> %1098, i64 3, !dbg !6
  %1103 = select i1 %.not, float %1100, float %1099, !dbg !6
  %1104 = bitcast float %1103 to i32, !dbg !6
  %1105 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1104), !dbg !6
  %1106 = bitcast i32 %1105 to float, !dbg !6
  %1107 = select i1 %.not, float %1099, float %1106, !dbg !6
  %1108 = select i1 %.not, float %1106, float %1100, !dbg !6
  %1109 = select i1 %.not, float %1102, float %1101, !dbg !6
  %1110 = bitcast float %1109 to i32, !dbg !6
  %1111 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1110), !dbg !6
  %1112 = bitcast i32 %1111 to float, !dbg !6
  %1113 = select i1 %.not, float %1101, float %1112, !dbg !6
  %1114 = select i1 %.not, float %1112, float %1102, !dbg !6
  %1115 = select i1 %.not40, float %1113, float %1107, !dbg !6
  %1116 = bitcast float %1115 to i32, !dbg !6
  %1117 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1116), !dbg !6
  %1118 = bitcast i32 %1117 to float, !dbg !6
  %1119 = select i1 %.not40, float %1107, float %1118, !dbg !6
  %1120 = select i1 %.not40, float %1118, float %1113, !dbg !6
  %1121 = select i1 %.not40, float %1114, float %1108, !dbg !6
  %1122 = bitcast float %1121 to i32, !dbg !6
  %1123 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1122), !dbg !6
  %1124 = bitcast i32 %1123 to float, !dbg !6
  %1125 = select i1 %.not40, float %1108, float %1124, !dbg !6
  %1126 = select i1 %.not40, float %1124, float %1114, !dbg !6
  %1127 = extractelement <8 x float> %1098, i64 4, !dbg !6
  %1128 = extractelement <8 x float> %1098, i64 5, !dbg !6
  %1129 = extractelement <8 x float> %1098, i64 6, !dbg !6
  %1130 = extractelement <8 x float> %1098, i64 7, !dbg !6
  %1131 = select i1 %.not, float %1128, float %1127, !dbg !6
  %1132 = bitcast float %1131 to i32, !dbg !6
  %1133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1132), !dbg !6
  %1134 = bitcast i32 %1133 to float, !dbg !6
  %1135 = select i1 %.not, float %1127, float %1134, !dbg !6
  %1136 = select i1 %.not, float %1134, float %1128, !dbg !6
  %1137 = select i1 %.not, float %1130, float %1129, !dbg !6
  %1138 = bitcast float %1137 to i32, !dbg !6
  %1139 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1138), !dbg !6
  %1140 = bitcast i32 %1139 to float, !dbg !6
  %1141 = select i1 %.not, float %1129, float %1140, !dbg !6
  %1142 = select i1 %.not, float %1140, float %1130, !dbg !6
  %1143 = select i1 %.not40, float %1141, float %1135, !dbg !6
  %1144 = bitcast float %1143 to i32, !dbg !6
  %1145 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1144), !dbg !6
  %1146 = bitcast i32 %1145 to float, !dbg !6
  %1147 = select i1 %.not40, float %1135, float %1146, !dbg !6
  %1148 = select i1 %.not40, float %1146, float %1141, !dbg !6
  %1149 = select i1 %.not40, float %1142, float %1136, !dbg !6
  %1150 = bitcast float %1149 to i32, !dbg !6
  %1151 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1150), !dbg !6
  %1152 = bitcast i32 %1151 to float, !dbg !6
  %1153 = select i1 %.not40, float %1136, float %1152, !dbg !6
  %1154 = select i1 %.not40, float %1152, float %1142, !dbg !6
  %1155 = extractvalue [2 x <8 x float>] %107, 1, !dbg !6
  %1156 = extractelement <8 x float> %1155, i64 0, !dbg !6
  %1157 = extractelement <8 x float> %1155, i64 1, !dbg !6
  %1158 = extractelement <8 x float> %1155, i64 2, !dbg !6
  %1159 = extractelement <8 x float> %1155, i64 3, !dbg !6
  %1160 = select i1 %.not, float %1157, float %1156, !dbg !6
  %1161 = bitcast float %1160 to i32, !dbg !6
  %1162 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1161), !dbg !6
  %1163 = bitcast i32 %1162 to float, !dbg !6
  %1164 = select i1 %.not, float %1156, float %1163, !dbg !6
  %1165 = select i1 %.not, float %1163, float %1157, !dbg !6
  %1166 = select i1 %.not, float %1159, float %1158, !dbg !6
  %1167 = bitcast float %1166 to i32, !dbg !6
  %1168 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1167), !dbg !6
  %1169 = bitcast i32 %1168 to float, !dbg !6
  %1170 = select i1 %.not, float %1158, float %1169, !dbg !6
  %1171 = select i1 %.not, float %1169, float %1159, !dbg !6
  %1172 = select i1 %.not40, float %1170, float %1164, !dbg !6
  %1173 = bitcast float %1172 to i32, !dbg !6
  %1174 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1173), !dbg !6
  %1175 = bitcast i32 %1174 to float, !dbg !6
  %1176 = select i1 %.not40, float %1164, float %1175, !dbg !6
  %1177 = select i1 %.not40, float %1175, float %1170, !dbg !6
  %1178 = select i1 %.not40, float %1171, float %1165, !dbg !6
  %1179 = bitcast float %1178 to i32, !dbg !6
  %1180 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1179), !dbg !6
  %1181 = bitcast i32 %1180 to float, !dbg !6
  %1182 = select i1 %.not40, float %1165, float %1181, !dbg !6
  %1183 = select i1 %.not40, float %1181, float %1171, !dbg !6
  %1184 = extractelement <8 x float> %1155, i64 4, !dbg !6
  %1185 = extractelement <8 x float> %1155, i64 5, !dbg !6
  %1186 = extractelement <8 x float> %1155, i64 6, !dbg !6
  %1187 = extractelement <8 x float> %1155, i64 7, !dbg !6
  %1188 = select i1 %.not, float %1185, float %1184, !dbg !6
  %1189 = bitcast float %1188 to i32, !dbg !6
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1189), !dbg !6
  %1191 = bitcast i32 %1190 to float, !dbg !6
  %1192 = select i1 %.not, float %1184, float %1191, !dbg !6
  %1193 = select i1 %.not, float %1191, float %1185, !dbg !6
  %1194 = select i1 %.not, float %1187, float %1186, !dbg !6
  %1195 = bitcast float %1194 to i32, !dbg !6
  %1196 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1195), !dbg !6
  %1197 = bitcast i32 %1196 to float, !dbg !6
  %1198 = select i1 %.not, float %1186, float %1197, !dbg !6
  %1199 = select i1 %.not, float %1197, float %1187, !dbg !6
  %1200 = select i1 %.not40, float %1198, float %1192, !dbg !6
  %1201 = bitcast float %1200 to i32, !dbg !6
  %1202 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1201), !dbg !6
  %1203 = bitcast i32 %1202 to float, !dbg !6
  %1204 = select i1 %.not40, float %1192, float %1203, !dbg !6
  %1205 = select i1 %.not40, float %1203, float %1198, !dbg !6
  %1206 = select i1 %.not40, float %1199, float %1193, !dbg !6
  %1207 = bitcast float %1206 to i32, !dbg !6
  %1208 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1207), !dbg !6
  %1209 = bitcast i32 %1208 to float, !dbg !6
  %1210 = select i1 %.not40, float %1193, float %1209, !dbg !6
  %1211 = select i1 %.not40, float %1209, float %1199, !dbg !6
  %1212 = extractvalue [2 x <8 x float>] %106, 0, !dbg !6
  %1213 = extractelement <8 x float> %1212, i64 0, !dbg !6
  %1214 = extractelement <8 x float> %1212, i64 1, !dbg !6
  %1215 = extractelement <8 x float> %1212, i64 2, !dbg !6
  %1216 = extractelement <8 x float> %1212, i64 3, !dbg !6
  %1217 = select i1 %.not, float %1214, float %1213, !dbg !6
  %1218 = bitcast float %1217 to i32, !dbg !6
  %1219 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1218), !dbg !6
  %1220 = bitcast i32 %1219 to float, !dbg !6
  %1221 = select i1 %.not, float %1213, float %1220, !dbg !6
  %1222 = select i1 %.not, float %1220, float %1214, !dbg !6
  %1223 = select i1 %.not, float %1216, float %1215, !dbg !6
  %1224 = bitcast float %1223 to i32, !dbg !6
  %1225 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1224), !dbg !6
  %1226 = bitcast i32 %1225 to float, !dbg !6
  %1227 = select i1 %.not, float %1215, float %1226, !dbg !6
  %1228 = select i1 %.not, float %1226, float %1216, !dbg !6
  %1229 = select i1 %.not40, float %1227, float %1221, !dbg !6
  %1230 = bitcast float %1229 to i32, !dbg !6
  %1231 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1230), !dbg !6
  %1232 = bitcast i32 %1231 to float, !dbg !6
  %1233 = select i1 %.not40, float %1221, float %1232, !dbg !6
  %1234 = select i1 %.not40, float %1232, float %1227, !dbg !6
  %1235 = select i1 %.not40, float %1228, float %1222, !dbg !6
  %1236 = bitcast float %1235 to i32, !dbg !6
  %1237 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1236), !dbg !6
  %1238 = bitcast i32 %1237 to float, !dbg !6
  %1239 = select i1 %.not40, float %1222, float %1238, !dbg !6
  %1240 = select i1 %.not40, float %1238, float %1228, !dbg !6
  %1241 = extractelement <8 x float> %1212, i64 4, !dbg !6
  %1242 = extractelement <8 x float> %1212, i64 5, !dbg !6
  %1243 = extractelement <8 x float> %1212, i64 6, !dbg !6
  %1244 = extractelement <8 x float> %1212, i64 7, !dbg !6
  %1245 = select i1 %.not, float %1242, float %1241, !dbg !6
  %1246 = bitcast float %1245 to i32, !dbg !6
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1246), !dbg !6
  %1248 = bitcast i32 %1247 to float, !dbg !6
  %1249 = select i1 %.not, float %1241, float %1248, !dbg !6
  %1250 = select i1 %.not, float %1248, float %1242, !dbg !6
  %1251 = select i1 %.not, float %1244, float %1243, !dbg !6
  %1252 = bitcast float %1251 to i32, !dbg !6
  %1253 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1252), !dbg !6
  %1254 = bitcast i32 %1253 to float, !dbg !6
  %1255 = select i1 %.not, float %1243, float %1254, !dbg !6
  %1256 = select i1 %.not, float %1254, float %1244, !dbg !6
  %1257 = select i1 %.not40, float %1255, float %1249, !dbg !6
  %1258 = bitcast float %1257 to i32, !dbg !6
  %1259 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1258), !dbg !6
  %1260 = bitcast i32 %1259 to float, !dbg !6
  %1261 = select i1 %.not40, float %1249, float %1260, !dbg !6
  %1262 = select i1 %.not40, float %1260, float %1255, !dbg !6
  %1263 = select i1 %.not40, float %1256, float %1250, !dbg !6
  %1264 = bitcast float %1263 to i32, !dbg !6
  %1265 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1264), !dbg !6
  %1266 = bitcast i32 %1265 to float, !dbg !6
  %1267 = select i1 %.not40, float %1250, float %1266, !dbg !6
  %1268 = select i1 %.not40, float %1266, float %1256, !dbg !6
  %1269 = extractvalue [2 x <8 x float>] %106, 1, !dbg !6
  %1270 = extractelement <8 x float> %1269, i64 0, !dbg !6
  %1271 = extractelement <8 x float> %1269, i64 1, !dbg !6
  %1272 = extractelement <8 x float> %1269, i64 2, !dbg !6
  %1273 = extractelement <8 x float> %1269, i64 3, !dbg !6
  %1274 = select i1 %.not, float %1271, float %1270, !dbg !6
  %1275 = bitcast float %1274 to i32, !dbg !6
  %1276 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1275), !dbg !6
  %1277 = bitcast i32 %1276 to float, !dbg !6
  %1278 = select i1 %.not, float %1270, float %1277, !dbg !6
  %1279 = select i1 %.not, float %1277, float %1271, !dbg !6
  %1280 = select i1 %.not, float %1273, float %1272, !dbg !6
  %1281 = bitcast float %1280 to i32, !dbg !6
  %1282 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1281), !dbg !6
  %1283 = bitcast i32 %1282 to float, !dbg !6
  %1284 = select i1 %.not, float %1272, float %1283, !dbg !6
  %1285 = select i1 %.not, float %1283, float %1273, !dbg !6
  %1286 = select i1 %.not40, float %1284, float %1278, !dbg !6
  %1287 = bitcast float %1286 to i32, !dbg !6
  %1288 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1287), !dbg !6
  %1289 = bitcast i32 %1288 to float, !dbg !6
  %1290 = select i1 %.not40, float %1278, float %1289, !dbg !6
  %1291 = select i1 %.not40, float %1289, float %1284, !dbg !6
  %1292 = select i1 %.not40, float %1285, float %1279, !dbg !6
  %1293 = bitcast float %1292 to i32, !dbg !6
  %1294 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1293), !dbg !6
  %1295 = bitcast i32 %1294 to float, !dbg !6
  %1296 = select i1 %.not40, float %1279, float %1295, !dbg !6
  %1297 = select i1 %.not40, float %1295, float %1285, !dbg !6
  %1298 = extractelement <8 x float> %1269, i64 4, !dbg !6
  %1299 = extractelement <8 x float> %1269, i64 5, !dbg !6
  %1300 = extractelement <8 x float> %1269, i64 6, !dbg !6
  %1301 = extractelement <8 x float> %1269, i64 7, !dbg !6
  %1302 = select i1 %.not, float %1299, float %1298, !dbg !6
  %1303 = bitcast float %1302 to i32, !dbg !6
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1303), !dbg !6
  %1305 = bitcast i32 %1304 to float, !dbg !6
  %1306 = select i1 %.not, float %1298, float %1305, !dbg !6
  %1307 = select i1 %.not, float %1305, float %1299, !dbg !6
  %1308 = select i1 %.not, float %1301, float %1300, !dbg !6
  %1309 = bitcast float %1308 to i32, !dbg !6
  %1310 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1309), !dbg !6
  %1311 = bitcast i32 %1310 to float, !dbg !6
  %1312 = select i1 %.not, float %1300, float %1311, !dbg !6
  %1313 = select i1 %.not, float %1311, float %1301, !dbg !6
  %1314 = select i1 %.not40, float %1312, float %1306, !dbg !6
  %1315 = bitcast float %1314 to i32, !dbg !6
  %1316 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1315), !dbg !6
  %1317 = bitcast i32 %1316 to float, !dbg !6
  %1318 = select i1 %.not40, float %1306, float %1317, !dbg !6
  %1319 = select i1 %.not40, float %1317, float %1312, !dbg !6
  %1320 = select i1 %.not40, float %1313, float %1307, !dbg !6
  %1321 = bitcast float %1320 to i32, !dbg !6
  %1322 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1321), !dbg !6
  %1323 = bitcast i32 %1322 to float, !dbg !6
  %1324 = select i1 %.not40, float %1307, float %1323, !dbg !6
  %1325 = select i1 %.not40, float %1323, float %1313, !dbg !6
  %1326 = extractvalue [2 x <8 x float>] %105, 0, !dbg !6
  %1327 = extractelement <8 x float> %1326, i64 0, !dbg !6
  %1328 = extractelement <8 x float> %1326, i64 1, !dbg !6
  %1329 = extractelement <8 x float> %1326, i64 2, !dbg !6
  %1330 = extractelement <8 x float> %1326, i64 3, !dbg !6
  %1331 = select i1 %.not, float %1328, float %1327, !dbg !6
  %1332 = bitcast float %1331 to i32, !dbg !6
  %1333 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1332), !dbg !6
  %1334 = bitcast i32 %1333 to float, !dbg !6
  %1335 = select i1 %.not, float %1327, float %1334, !dbg !6
  %1336 = select i1 %.not, float %1334, float %1328, !dbg !6
  %1337 = select i1 %.not, float %1330, float %1329, !dbg !6
  %1338 = bitcast float %1337 to i32, !dbg !6
  %1339 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1338), !dbg !6
  %1340 = bitcast i32 %1339 to float, !dbg !6
  %1341 = select i1 %.not, float %1329, float %1340, !dbg !6
  %1342 = select i1 %.not, float %1340, float %1330, !dbg !6
  %1343 = select i1 %.not40, float %1341, float %1335, !dbg !6
  %1344 = bitcast float %1343 to i32, !dbg !6
  %1345 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1344), !dbg !6
  %1346 = bitcast i32 %1345 to float, !dbg !6
  %1347 = select i1 %.not40, float %1335, float %1346, !dbg !6
  %1348 = select i1 %.not40, float %1346, float %1341, !dbg !6
  %1349 = select i1 %.not40, float %1342, float %1336, !dbg !6
  %1350 = bitcast float %1349 to i32, !dbg !6
  %1351 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1350), !dbg !6
  %1352 = bitcast i32 %1351 to float, !dbg !6
  %1353 = select i1 %.not40, float %1336, float %1352, !dbg !6
  %1354 = select i1 %.not40, float %1352, float %1342, !dbg !6
  %1355 = extractelement <8 x float> %1326, i64 4, !dbg !6
  %1356 = extractelement <8 x float> %1326, i64 5, !dbg !6
  %1357 = extractelement <8 x float> %1326, i64 6, !dbg !6
  %1358 = extractelement <8 x float> %1326, i64 7, !dbg !6
  %1359 = select i1 %.not, float %1356, float %1355, !dbg !6
  %1360 = bitcast float %1359 to i32, !dbg !6
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1360), !dbg !6
  %1362 = bitcast i32 %1361 to float, !dbg !6
  %1363 = select i1 %.not, float %1355, float %1362, !dbg !6
  %1364 = select i1 %.not, float %1362, float %1356, !dbg !6
  %1365 = select i1 %.not, float %1358, float %1357, !dbg !6
  %1366 = bitcast float %1365 to i32, !dbg !6
  %1367 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1366), !dbg !6
  %1368 = bitcast i32 %1367 to float, !dbg !6
  %1369 = select i1 %.not, float %1357, float %1368, !dbg !6
  %1370 = select i1 %.not, float %1368, float %1358, !dbg !6
  %1371 = select i1 %.not40, float %1369, float %1363, !dbg !6
  %1372 = bitcast float %1371 to i32, !dbg !6
  %1373 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1372), !dbg !6
  %1374 = bitcast i32 %1373 to float, !dbg !6
  %1375 = select i1 %.not40, float %1363, float %1374, !dbg !6
  %1376 = select i1 %.not40, float %1374, float %1369, !dbg !6
  %1377 = select i1 %.not40, float %1370, float %1364, !dbg !6
  %1378 = bitcast float %1377 to i32, !dbg !6
  %1379 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1378), !dbg !6
  %1380 = bitcast i32 %1379 to float, !dbg !6
  %1381 = select i1 %.not40, float %1364, float %1380, !dbg !6
  %1382 = select i1 %.not40, float %1380, float %1370, !dbg !6
  %1383 = extractvalue [2 x <8 x float>] %105, 1, !dbg !6
  %1384 = extractelement <8 x float> %1383, i64 0, !dbg !6
  %1385 = extractelement <8 x float> %1383, i64 1, !dbg !6
  %1386 = extractelement <8 x float> %1383, i64 2, !dbg !6
  %1387 = extractelement <8 x float> %1383, i64 3, !dbg !6
  %1388 = select i1 %.not, float %1385, float %1384, !dbg !6
  %1389 = bitcast float %1388 to i32, !dbg !6
  %1390 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1389), !dbg !6
  %1391 = bitcast i32 %1390 to float, !dbg !6
  %1392 = select i1 %.not, float %1384, float %1391, !dbg !6
  %1393 = select i1 %.not, float %1391, float %1385, !dbg !6
  %1394 = select i1 %.not, float %1387, float %1386, !dbg !6
  %1395 = bitcast float %1394 to i32, !dbg !6
  %1396 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1395), !dbg !6
  %1397 = bitcast i32 %1396 to float, !dbg !6
  %1398 = select i1 %.not, float %1386, float %1397, !dbg !6
  %1399 = select i1 %.not, float %1397, float %1387, !dbg !6
  %1400 = select i1 %.not40, float %1398, float %1392, !dbg !6
  %1401 = bitcast float %1400 to i32, !dbg !6
  %1402 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1401), !dbg !6
  %1403 = bitcast i32 %1402 to float, !dbg !6
  %1404 = select i1 %.not40, float %1392, float %1403, !dbg !6
  %1405 = select i1 %.not40, float %1403, float %1398, !dbg !6
  %1406 = select i1 %.not40, float %1399, float %1393, !dbg !6
  %1407 = bitcast float %1406 to i32, !dbg !6
  %1408 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1407), !dbg !6
  %1409 = bitcast i32 %1408 to float, !dbg !6
  %1410 = select i1 %.not40, float %1393, float %1409, !dbg !6
  %1411 = select i1 %.not40, float %1409, float %1399, !dbg !6
  %1412 = extractelement <8 x float> %1383, i64 4, !dbg !6
  %1413 = extractelement <8 x float> %1383, i64 5, !dbg !6
  %1414 = extractelement <8 x float> %1383, i64 6, !dbg !6
  %1415 = extractelement <8 x float> %1383, i64 7, !dbg !6
  %1416 = select i1 %.not, float %1413, float %1412, !dbg !6
  %1417 = bitcast float %1416 to i32, !dbg !6
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1417), !dbg !6
  %1419 = bitcast i32 %1418 to float, !dbg !6
  %1420 = select i1 %.not, float %1412, float %1419, !dbg !6
  %1421 = select i1 %.not, float %1419, float %1413, !dbg !6
  %1422 = select i1 %.not, float %1415, float %1414, !dbg !6
  %1423 = bitcast float %1422 to i32, !dbg !6
  %1424 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1423), !dbg !6
  %1425 = bitcast i32 %1424 to float, !dbg !6
  %1426 = select i1 %.not, float %1414, float %1425, !dbg !6
  %1427 = select i1 %.not, float %1425, float %1415, !dbg !6
  %1428 = select i1 %.not40, float %1426, float %1420, !dbg !6
  %1429 = bitcast float %1428 to i32, !dbg !6
  %1430 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1429), !dbg !6
  %1431 = bitcast i32 %1430 to float, !dbg !6
  %1432 = select i1 %.not40, float %1420, float %1431, !dbg !6
  %1433 = select i1 %.not40, float %1431, float %1426, !dbg !6
  %1434 = select i1 %.not40, float %1427, float %1421, !dbg !6
  %1435 = bitcast float %1434 to i32, !dbg !6
  %1436 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1435), !dbg !6
  %1437 = bitcast i32 %1436 to float, !dbg !6
  %1438 = select i1 %.not40, float %1421, float %1437, !dbg !6
  %1439 = select i1 %.not40, float %1437, float %1427, !dbg !6
  %1440 = extractvalue [2 x <8 x float>] %102, 0, !dbg !6
  %1441 = extractelement <8 x float> %1440, i64 0, !dbg !6
  %1442 = extractelement <8 x float> %1440, i64 1, !dbg !6
  %1443 = extractelement <8 x float> %1440, i64 2, !dbg !6
  %1444 = extractelement <8 x float> %1440, i64 3, !dbg !6
  %1445 = select i1 %.not, float %1442, float %1441, !dbg !6
  %1446 = bitcast float %1445 to i32, !dbg !6
  %1447 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1446), !dbg !6
  %1448 = bitcast i32 %1447 to float, !dbg !6
  %1449 = select i1 %.not, float %1441, float %1448, !dbg !6
  %1450 = select i1 %.not, float %1448, float %1442, !dbg !6
  %1451 = select i1 %.not, float %1444, float %1443, !dbg !6
  %1452 = bitcast float %1451 to i32, !dbg !6
  %1453 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1452), !dbg !6
  %1454 = bitcast i32 %1453 to float, !dbg !6
  %1455 = select i1 %.not, float %1443, float %1454, !dbg !6
  %1456 = select i1 %.not, float %1454, float %1444, !dbg !6
  %1457 = select i1 %.not40, float %1455, float %1449, !dbg !6
  %1458 = bitcast float %1457 to i32, !dbg !6
  %1459 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1458), !dbg !6
  %1460 = select i1 %.not40, float %1456, float %1450, !dbg !6
  %1461 = bitcast float %1460 to i32, !dbg !6
  %1462 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1461), !dbg !6
  %1463 = extractelement <8 x float> %1440, i64 4, !dbg !6
  %1464 = extractelement <8 x float> %1440, i64 5, !dbg !6
  %1465 = extractelement <8 x float> %1440, i64 6, !dbg !6
  %1466 = extractelement <8 x float> %1440, i64 7, !dbg !6
  %1467 = select i1 %.not, float %1464, float %1463, !dbg !6
  %1468 = bitcast float %1467 to i32, !dbg !6
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1468), !dbg !6
  %1470 = bitcast i32 %1469 to float, !dbg !6
  %1471 = select i1 %.not, float %1463, float %1470, !dbg !6
  %1472 = select i1 %.not, float %1470, float %1464, !dbg !6
  %1473 = select i1 %.not, float %1466, float %1465, !dbg !6
  %1474 = bitcast float %1473 to i32, !dbg !6
  %1475 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1474), !dbg !6
  %1476 = bitcast i32 %1475 to float, !dbg !6
  %1477 = select i1 %.not, float %1465, float %1476, !dbg !6
  %1478 = select i1 %.not, float %1476, float %1466, !dbg !6
  %1479 = select i1 %.not40, float %1477, float %1471, !dbg !6
  %1480 = bitcast float %1479 to i32, !dbg !6
  %1481 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1480), !dbg !6
  %1482 = select i1 %.not40, float %1478, float %1472, !dbg !6
  %1483 = bitcast float %1482 to i32, !dbg !6
  %1484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1483), !dbg !6
  %1485 = extractvalue [2 x <8 x float>] %102, 1, !dbg !6
  %1486 = extractelement <8 x float> %1485, i64 0, !dbg !6
  %1487 = extractelement <8 x float> %1485, i64 1, !dbg !6
  %1488 = extractelement <8 x float> %1485, i64 2, !dbg !6
  %1489 = extractelement <8 x float> %1485, i64 3, !dbg !6
  %1490 = select i1 %.not, float %1487, float %1486, !dbg !6
  %1491 = bitcast float %1490 to i32, !dbg !6
  %1492 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1491), !dbg !6
  %1493 = bitcast i32 %1492 to float, !dbg !6
  %1494 = select i1 %.not, float %1486, float %1493, !dbg !6
  %1495 = select i1 %.not, float %1493, float %1487, !dbg !6
  %1496 = select i1 %.not, float %1489, float %1488, !dbg !6
  %1497 = bitcast float %1496 to i32, !dbg !6
  %1498 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1497), !dbg !6
  %1499 = bitcast i32 %1498 to float, !dbg !6
  %1500 = select i1 %.not, float %1488, float %1499, !dbg !6
  %1501 = select i1 %.not, float %1499, float %1489, !dbg !6
  %1502 = select i1 %.not40, float %1500, float %1494, !dbg !6
  %1503 = bitcast float %1502 to i32, !dbg !6
  %1504 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1503), !dbg !6
  %1505 = select i1 %.not40, float %1501, float %1495, !dbg !6
  %1506 = bitcast float %1505 to i32, !dbg !6
  %1507 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1506), !dbg !6
  %1508 = extractelement <8 x float> %1485, i64 4, !dbg !6
  %1509 = extractelement <8 x float> %1485, i64 5, !dbg !6
  %1510 = extractelement <8 x float> %1485, i64 6, !dbg !6
  %1511 = extractelement <8 x float> %1485, i64 7, !dbg !6
  %1512 = select i1 %.not, float %1509, float %1508, !dbg !6
  %1513 = bitcast float %1512 to i32, !dbg !6
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1513), !dbg !6
  %1515 = bitcast i32 %1514 to float, !dbg !6
  %1516 = select i1 %.not, float %1508, float %1515, !dbg !6
  %1517 = select i1 %.not, float %1515, float %1509, !dbg !6
  %1518 = select i1 %.not, float %1511, float %1510, !dbg !6
  %1519 = bitcast float %1518 to i32, !dbg !6
  %1520 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1519), !dbg !6
  %1521 = bitcast i32 %1520 to float, !dbg !6
  %1522 = select i1 %.not, float %1510, float %1521, !dbg !6
  %1523 = select i1 %.not, float %1521, float %1511, !dbg !6
  %1524 = select i1 %.not40, float %1522, float %1516, !dbg !6
  %1525 = bitcast float %1524 to i32, !dbg !6
  %1526 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1525), !dbg !6
  %1527 = select i1 %.not40, float %1523, float %1517, !dbg !6
  %1528 = bitcast float %1527 to i32, !dbg !6
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1528), !dbg !6
  %1530 = shufflevector <1 x float> %1095, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1531 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1530, !dbg !17
  %1532 = shufflevector <1 x float> %1090, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1533 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %1532, !dbg !17
  %1534 = bitcast i32 %1526 to float, !dbg !6
  %1535 = select i1 %.not40, float %1516, float %1534, !dbg !6
  %1536 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !17
  %1537 = bitcast i32 %1529 to float, !dbg !6
  %1538 = select i1 %.not40, float %1517, float %1537, !dbg !6
  %1539 = insertelement <4 x float> %1536, float %1538, i64 1, !dbg !17
  %1540 = select i1 %.not40, float %1534, float %1522, !dbg !6
  %1541 = insertelement <4 x float> %1539, float %1540, i64 2, !dbg !17
  %1542 = select i1 %.not40, float %1537, float %1523, !dbg !6
  %1543 = insertelement <4 x float> %1541, float %1542, i64 3, !dbg !17
  %1544 = fmul <4 x float> %1543, %1533, !dbg !17
  %1545 = fptrunc <4 x float> %1544 to <4 x half>, !dbg !18
  %1546 = extractelement <4 x half> %1545, i64 3, !dbg !18
  %1547 = extractelement <4 x half> %1545, i64 2, !dbg !18
  %1548 = extractelement <4 x half> %1545, i64 1, !dbg !18
  %1549 = extractelement <4 x half> %1545, i64 0, !dbg !18
  %1550 = bitcast i32 %1504 to float, !dbg !6
  %1551 = select i1 %.not40, float %1494, float %1550, !dbg !6
  %1552 = insertelement <4 x float> poison, float %1551, i64 0, !dbg !17
  %1553 = bitcast i32 %1507 to float, !dbg !6
  %1554 = select i1 %.not40, float %1495, float %1553, !dbg !6
  %1555 = insertelement <4 x float> %1552, float %1554, i64 1, !dbg !17
  %1556 = select i1 %.not40, float %1550, float %1500, !dbg !6
  %1557 = insertelement <4 x float> %1555, float %1556, i64 2, !dbg !17
  %1558 = select i1 %.not40, float %1553, float %1501, !dbg !6
  %1559 = insertelement <4 x float> %1557, float %1558, i64 3, !dbg !17
  %1560 = fmul <4 x float> %1559, %1533, !dbg !17
  %1561 = fptrunc <4 x float> %1560 to <4 x half>, !dbg !18
  %1562 = extractelement <4 x half> %1561, i64 3, !dbg !18
  %1563 = extractelement <4 x half> %1561, i64 2, !dbg !18
  %1564 = extractelement <4 x half> %1561, i64 1, !dbg !18
  %1565 = extractelement <4 x half> %1561, i64 0, !dbg !18
  %1566 = bitcast i32 %1481 to float, !dbg !6
  %1567 = select i1 %.not40, float %1471, float %1566, !dbg !6
  %1568 = insertelement <4 x float> poison, float %1567, i64 0, !dbg !17
  %1569 = bitcast i32 %1484 to float, !dbg !6
  %1570 = select i1 %.not40, float %1472, float %1569, !dbg !6
  %1571 = insertelement <4 x float> %1568, float %1570, i64 1, !dbg !17
  %1572 = select i1 %.not40, float %1566, float %1477, !dbg !6
  %1573 = insertelement <4 x float> %1571, float %1572, i64 2, !dbg !17
  %1574 = select i1 %.not40, float %1569, float %1478, !dbg !6
  %1575 = insertelement <4 x float> %1573, float %1574, i64 3, !dbg !17
  %1576 = fmul <4 x float> %1575, %1531, !dbg !17
  %1577 = fptrunc <4 x float> %1576 to <4 x half>, !dbg !18
  %1578 = extractelement <4 x half> %1577, i64 3, !dbg !18
  %1579 = extractelement <4 x half> %1577, i64 2, !dbg !18
  %1580 = extractelement <4 x half> %1577, i64 1, !dbg !18
  %1581 = extractelement <4 x half> %1577, i64 0, !dbg !18
  %1582 = bitcast i32 %1459 to float, !dbg !6
  %1583 = select i1 %.not40, float %1449, float %1582, !dbg !6
  %1584 = insertelement <4 x float> poison, float %1583, i64 0, !dbg !17
  %1585 = bitcast i32 %1462 to float, !dbg !6
  %1586 = select i1 %.not40, float %1450, float %1585, !dbg !6
  %1587 = insertelement <4 x float> %1584, float %1586, i64 1, !dbg !17
  %1588 = select i1 %.not40, float %1582, float %1455, !dbg !6
  %1589 = insertelement <4 x float> %1587, float %1588, i64 2, !dbg !17
  %1590 = select i1 %.not40, float %1585, float %1456, !dbg !6
  %1591 = insertelement <4 x float> %1589, float %1590, i64 3, !dbg !17
  %1592 = fmul <4 x float> %1591, %1531, !dbg !17
  %1593 = fptrunc <4 x float> %1592 to <4 x half>, !dbg !18
  %1594 = extractelement <4 x half> %1593, i64 3, !dbg !18
  %1595 = extractelement <4 x half> %1593, i64 2, !dbg !18
  %1596 = extractelement <4 x half> %1593, i64 1, !dbg !18
  %1597 = extractelement <4 x half> %1593, i64 0, !dbg !18
  %address.thread848 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local849 = zext nneg i32 %address.thread848 to i64, !dbg !9
  %address.local850 = add nuw i64 %32, %address.local849, !dbg !19
  %1598 = lshr i64 %address.local849, 2, !dbg !19
  %address.local852 = and i64 %1598, 48, !dbg !19
  %address.local853 = add i64 %address.local850, %address.local852, !dbg !19
  %address.local854 = and i64 %address.local849, 240, !dbg !19
  %address.local855 = sub i64 %address.local853, %address.local854, !dbg !19
  %address.local856 = add i64 %address.local855, %33, !dbg !19
  %address.local857 = shl nuw nsw i64 %address.local856, 7, !dbg !19
  %1599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1599, i64 %address.local857, !dbg !19
  %address.local861 = and i64 %1598, 12, !dbg !12
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local861, !dbg !19
  store half %1597, ptr addrspace(1) %address.local862, align 2, !dbg !19
  %address.local864 = or disjoint i64 %address.local861, 1, !dbg !19
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local864, !dbg !19
  store half %1596, ptr addrspace(1) %address.local865, align 2, !dbg !19
  %address.local867 = or disjoint i64 %address.local861, 2, !dbg !19
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local867, !dbg !19
  store half %1595, ptr addrspace(1) %address.local868, align 2, !dbg !19
  %address.local870 = or disjoint i64 %address.local861, 3, !dbg !19
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local870, !dbg !19
  store half %1594, ptr addrspace(1) %address.local871, align 2, !dbg !19
  %address.local873 = or disjoint i64 %address.local861, 16, !dbg !19
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local873, !dbg !19
  store half %1581, ptr addrspace(1) %address.local874, align 2, !dbg !19
  %address.local876 = or disjoint i64 %address.local861, 17, !dbg !19
  %address.local877 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local876, !dbg !19
  store half %1580, ptr addrspace(1) %address.local877, align 2, !dbg !19
  %address.local879 = or disjoint i64 %address.local861, 18, !dbg !19
  %address.local880 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local879, !dbg !19
  store half %1579, ptr addrspace(1) %address.local880, align 2, !dbg !19
  %address.local882 = or disjoint i64 %address.local861, 19, !dbg !19
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local882, !dbg !19
  store half %1578, ptr addrspace(1) %address.local883, align 2, !dbg !19
  %address.local887 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local859, i64 16384, !dbg !19
  %address.local888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local861, !dbg !19
  store half %1565, ptr addrspace(1) %address.local888, align 2, !dbg !19
  %address.local890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local864, !dbg !19
  store half %1564, ptr addrspace(1) %address.local890, align 2, !dbg !19
  %address.local892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local867, !dbg !19
  store half %1563, ptr addrspace(1) %address.local892, align 2, !dbg !19
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local870, !dbg !19
  store half %1562, ptr addrspace(1) %address.local894, align 2, !dbg !19
  %address.local896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local873, !dbg !19
  store half %1549, ptr addrspace(1) %address.local896, align 2, !dbg !19
  %address.local898 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local876, !dbg !19
  store half %1548, ptr addrspace(1) %address.local898, align 2, !dbg !19
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local879, !dbg !19
  store half %1547, ptr addrspace(1) %address.local900, align 2, !dbg !19
  %address.local902 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local882, !dbg !19
  store half %1546, ptr addrspace(1) %address.local902, align 2, !dbg !19
  %1600 = insertelement <4 x float> poison, float %1432, i64 0, !dbg !17
  %1601 = insertelement <4 x float> %1600, float %1438, i64 1, !dbg !17
  %1602 = insertelement <4 x float> %1601, float %1433, i64 2, !dbg !17
  %1603 = insertelement <4 x float> %1602, float %1439, i64 3, !dbg !17
  %1604 = fmul <4 x float> %1603, %1533, !dbg !17
  %1605 = fptrunc <4 x float> %1604 to <4 x half>, !dbg !18
  %1606 = extractelement <4 x half> %1605, i64 3, !dbg !18
  %1607 = extractelement <4 x half> %1605, i64 2, !dbg !18
  %1608 = extractelement <4 x half> %1605, i64 1, !dbg !18
  %1609 = extractelement <4 x half> %1605, i64 0, !dbg !18
  %1610 = insertelement <4 x float> poison, float %1404, i64 0, !dbg !17
  %1611 = insertelement <4 x float> %1610, float %1410, i64 1, !dbg !17
  %1612 = insertelement <4 x float> %1611, float %1405, i64 2, !dbg !17
  %1613 = insertelement <4 x float> %1612, float %1411, i64 3, !dbg !17
  %1614 = fmul <4 x float> %1613, %1533, !dbg !17
  %1615 = fptrunc <4 x float> %1614 to <4 x half>, !dbg !18
  %1616 = extractelement <4 x half> %1615, i64 3, !dbg !18
  %1617 = extractelement <4 x half> %1615, i64 2, !dbg !18
  %1618 = extractelement <4 x half> %1615, i64 1, !dbg !18
  %1619 = extractelement <4 x half> %1615, i64 0, !dbg !18
  %1620 = insertelement <4 x float> poison, float %1375, i64 0, !dbg !17
  %1621 = insertelement <4 x float> %1620, float %1381, i64 1, !dbg !17
  %1622 = insertelement <4 x float> %1621, float %1376, i64 2, !dbg !17
  %1623 = insertelement <4 x float> %1622, float %1382, i64 3, !dbg !17
  %1624 = fmul <4 x float> %1623, %1531, !dbg !17
  %1625 = fptrunc <4 x float> %1624 to <4 x half>, !dbg !18
  %1626 = extractelement <4 x half> %1625, i64 3, !dbg !18
  %1627 = extractelement <4 x half> %1625, i64 2, !dbg !18
  %1628 = extractelement <4 x half> %1625, i64 1, !dbg !18
  %1629 = extractelement <4 x half> %1625, i64 0, !dbg !18
  %1630 = insertelement <4 x float> poison, float %1347, i64 0, !dbg !17
  %1631 = insertelement <4 x float> %1630, float %1353, i64 1, !dbg !17
  %1632 = insertelement <4 x float> %1631, float %1348, i64 2, !dbg !17
  %1633 = insertelement <4 x float> %1632, float %1354, i64 3, !dbg !17
  %1634 = fmul <4 x float> %1633, %1531, !dbg !17
  %1635 = fptrunc <4 x float> %1634 to <4 x half>, !dbg !18
  %1636 = extractelement <4 x half> %1635, i64 3, !dbg !18
  %1637 = extractelement <4 x half> %1635, i64 2, !dbg !18
  %1638 = extractelement <4 x half> %1635, i64 1, !dbg !18
  %1639 = extractelement <4 x half> %1635, i64 0, !dbg !18
  %address.local917 = or disjoint i64 %address.local861, 32, !dbg !19
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local917, !dbg !19
  store half %1639, ptr addrspace(1) %address.local918, align 2, !dbg !19
  %address.local920 = or disjoint i64 %address.local861, 33, !dbg !19
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local920, !dbg !19
  store half %1638, ptr addrspace(1) %address.local921, align 2, !dbg !19
  %address.local923 = or disjoint i64 %address.local861, 34, !dbg !19
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local923, !dbg !19
  store half %1637, ptr addrspace(1) %address.local924, align 2, !dbg !19
  %address.local926 = or disjoint i64 %address.local861, 35, !dbg !19
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local926, !dbg !19
  store half %1636, ptr addrspace(1) %address.local927, align 2, !dbg !19
  %address.local929 = or disjoint i64 %address.local861, 48, !dbg !19
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local929, !dbg !19
  store half %1629, ptr addrspace(1) %address.local930, align 2, !dbg !19
  %address.local932 = or disjoint i64 %address.local861, 49, !dbg !19
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local932, !dbg !19
  store half %1628, ptr addrspace(1) %address.local933, align 2, !dbg !19
  %address.local935 = or disjoint i64 %address.local861, 50, !dbg !19
  %address.local936 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local935, !dbg !19
  store half %1627, ptr addrspace(1) %address.local936, align 2, !dbg !19
  %address.local938 = or i64 %1598, 51, !dbg !19
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local938, !dbg !19
  store half %1626, ptr addrspace(1) %address.local939, align 2, !dbg !19
  %address.local944 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local917, !dbg !19
  store half %1619, ptr addrspace(1) %address.local944, align 2, !dbg !19
  %address.local946 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local920, !dbg !19
  store half %1618, ptr addrspace(1) %address.local946, align 2, !dbg !19
  %address.local948 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local923, !dbg !19
  store half %1617, ptr addrspace(1) %address.local948, align 2, !dbg !19
  %address.local950 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local926, !dbg !19
  store half %1616, ptr addrspace(1) %address.local950, align 2, !dbg !19
  %address.local952 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local929, !dbg !19
  store half %1609, ptr addrspace(1) %address.local952, align 2, !dbg !19
  %address.local954 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local932, !dbg !19
  store half %1608, ptr addrspace(1) %address.local954, align 2, !dbg !19
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local935, !dbg !19
  store half %1607, ptr addrspace(1) %address.local956, align 2, !dbg !19
  %address.local958 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local938, !dbg !19
  store half %1606, ptr addrspace(1) %address.local958, align 2, !dbg !19
  %1640 = insertelement <4 x float> poison, float %1318, i64 0, !dbg !17
  %1641 = insertelement <4 x float> %1640, float %1324, i64 1, !dbg !17
  %1642 = insertelement <4 x float> %1641, float %1319, i64 2, !dbg !17
  %1643 = insertelement <4 x float> %1642, float %1325, i64 3, !dbg !17
  %1644 = fmul <4 x float> %1643, %1533, !dbg !17
  %1645 = fptrunc <4 x float> %1644 to <4 x half>, !dbg !18
  %1646 = extractelement <4 x half> %1645, i64 3, !dbg !18
  %1647 = extractelement <4 x half> %1645, i64 2, !dbg !18
  %1648 = extractelement <4 x half> %1645, i64 1, !dbg !18
  %1649 = extractelement <4 x half> %1645, i64 0, !dbg !18
  %1650 = insertelement <4 x float> poison, float %1290, i64 0, !dbg !17
  %1651 = insertelement <4 x float> %1650, float %1296, i64 1, !dbg !17
  %1652 = insertelement <4 x float> %1651, float %1291, i64 2, !dbg !17
  %1653 = insertelement <4 x float> %1652, float %1297, i64 3, !dbg !17
  %1654 = fmul <4 x float> %1653, %1533, !dbg !17
  %1655 = fptrunc <4 x float> %1654 to <4 x half>, !dbg !18
  %1656 = extractelement <4 x half> %1655, i64 3, !dbg !18
  %1657 = extractelement <4 x half> %1655, i64 2, !dbg !18
  %1658 = extractelement <4 x half> %1655, i64 1, !dbg !18
  %1659 = extractelement <4 x half> %1655, i64 0, !dbg !18
  %1660 = insertelement <4 x float> poison, float %1261, i64 0, !dbg !17
  %1661 = insertelement <4 x float> %1660, float %1267, i64 1, !dbg !17
  %1662 = insertelement <4 x float> %1661, float %1262, i64 2, !dbg !17
  %1663 = insertelement <4 x float> %1662, float %1268, i64 3, !dbg !17
  %1664 = fmul <4 x float> %1663, %1531, !dbg !17
  %1665 = fptrunc <4 x float> %1664 to <4 x half>, !dbg !18
  %1666 = extractelement <4 x half> %1665, i64 3, !dbg !18
  %1667 = extractelement <4 x half> %1665, i64 2, !dbg !18
  %1668 = extractelement <4 x half> %1665, i64 1, !dbg !18
  %1669 = extractelement <4 x half> %1665, i64 0, !dbg !18
  %1670 = insertelement <4 x float> poison, float %1233, i64 0, !dbg !17
  %1671 = insertelement <4 x float> %1670, float %1239, i64 1, !dbg !17
  %1672 = insertelement <4 x float> %1671, float %1234, i64 2, !dbg !17
  %1673 = insertelement <4 x float> %1672, float %1240, i64 3, !dbg !17
  %1674 = fmul <4 x float> %1673, %1531, !dbg !17
  %1675 = fptrunc <4 x float> %1674 to <4 x half>, !dbg !18
  %1676 = extractelement <4 x half> %1675, i64 3, !dbg !18
  %1677 = extractelement <4 x half> %1675, i64 2, !dbg !18
  %1678 = extractelement <4 x half> %1675, i64 1, !dbg !18
  %1679 = extractelement <4 x half> %1675, i64 0, !dbg !18
  %address.local973 = or disjoint i64 %address.local861, 64, !dbg !19
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local973, !dbg !19
  store half %1679, ptr addrspace(1) %address.local974, align 2, !dbg !19
  %address.local976 = or disjoint i64 %address.local861, 65, !dbg !19
  %address.local977 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local976, !dbg !19
  store half %1678, ptr addrspace(1) %address.local977, align 2, !dbg !19
  %address.local979 = or disjoint i64 %address.local861, 66, !dbg !19
  %address.local980 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local979, !dbg !19
  store half %1677, ptr addrspace(1) %address.local980, align 2, !dbg !19
  %address.local982 = or disjoint i64 %address.local861, 67, !dbg !19
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local982, !dbg !19
  store half %1676, ptr addrspace(1) %address.local983, align 2, !dbg !19
  %address.local985 = or disjoint i64 %address.local861, 80, !dbg !19
  %address.local986 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local985, !dbg !19
  store half %1669, ptr addrspace(1) %address.local986, align 2, !dbg !19
  %address.local988 = or disjoint i64 %address.local861, 81, !dbg !19
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local988, !dbg !19
  store half %1668, ptr addrspace(1) %address.local989, align 2, !dbg !19
  %address.local991 = or disjoint i64 %address.local861, 82, !dbg !19
  %address.local992 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local991, !dbg !19
  store half %1667, ptr addrspace(1) %address.local992, align 2, !dbg !19
  %address.local994 = or disjoint i64 %address.local861, 83, !dbg !19
  %address.local995 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local994, !dbg !19
  store half %1666, ptr addrspace(1) %address.local995, align 2, !dbg !19
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local973, !dbg !19
  store half %1659, ptr addrspace(1) %address.local1000, align 2, !dbg !19
  %address.local1002 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local976, !dbg !19
  store half %1658, ptr addrspace(1) %address.local1002, align 2, !dbg !19
  %address.local1004 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local979, !dbg !19
  store half %1657, ptr addrspace(1) %address.local1004, align 2, !dbg !19
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local982, !dbg !19
  store half %1656, ptr addrspace(1) %address.local1006, align 2, !dbg !19
  %address.local1008 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local985, !dbg !19
  store half %1649, ptr addrspace(1) %address.local1008, align 2, !dbg !19
  %address.local1010 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local988, !dbg !19
  store half %1648, ptr addrspace(1) %address.local1010, align 2, !dbg !19
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local991, !dbg !19
  store half %1647, ptr addrspace(1) %address.local1012, align 2, !dbg !19
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local994, !dbg !19
  store half %1646, ptr addrspace(1) %address.local1014, align 2, !dbg !19
  %1680 = insertelement <4 x float> poison, float %1204, i64 0, !dbg !17
  %1681 = insertelement <4 x float> %1680, float %1210, i64 1, !dbg !17
  %1682 = insertelement <4 x float> %1681, float %1205, i64 2, !dbg !17
  %1683 = insertelement <4 x float> %1682, float %1211, i64 3, !dbg !17
  %1684 = fmul <4 x float> %1683, %1533, !dbg !17
  %1685 = fptrunc <4 x float> %1684 to <4 x half>, !dbg !18
  %1686 = extractelement <4 x half> %1685, i64 3, !dbg !18
  %1687 = extractelement <4 x half> %1685, i64 2, !dbg !18
  %1688 = extractelement <4 x half> %1685, i64 1, !dbg !18
  %1689 = extractelement <4 x half> %1685, i64 0, !dbg !18
  %1690 = insertelement <4 x float> poison, float %1176, i64 0, !dbg !17
  %1691 = insertelement <4 x float> %1690, float %1182, i64 1, !dbg !17
  %1692 = insertelement <4 x float> %1691, float %1177, i64 2, !dbg !17
  %1693 = insertelement <4 x float> %1692, float %1183, i64 3, !dbg !17
  %1694 = fmul <4 x float> %1693, %1533, !dbg !17
  %1695 = fptrunc <4 x float> %1694 to <4 x half>, !dbg !18
  %1696 = extractelement <4 x half> %1695, i64 3, !dbg !18
  %1697 = extractelement <4 x half> %1695, i64 2, !dbg !18
  %1698 = extractelement <4 x half> %1695, i64 1, !dbg !18
  %1699 = extractelement <4 x half> %1695, i64 0, !dbg !18
  %1700 = insertelement <4 x float> poison, float %1147, i64 0, !dbg !17
  %1701 = insertelement <4 x float> %1700, float %1153, i64 1, !dbg !17
  %1702 = insertelement <4 x float> %1701, float %1148, i64 2, !dbg !17
  %1703 = insertelement <4 x float> %1702, float %1154, i64 3, !dbg !17
  %1704 = fmul <4 x float> %1703, %1531, !dbg !17
  %1705 = fptrunc <4 x float> %1704 to <4 x half>, !dbg !18
  %1706 = extractelement <4 x half> %1705, i64 3, !dbg !18
  %1707 = extractelement <4 x half> %1705, i64 2, !dbg !18
  %1708 = extractelement <4 x half> %1705, i64 1, !dbg !18
  %1709 = extractelement <4 x half> %1705, i64 0, !dbg !18
  %1710 = insertelement <4 x float> poison, float %1119, i64 0, !dbg !17
  %1711 = insertelement <4 x float> %1710, float %1125, i64 1, !dbg !17
  %1712 = insertelement <4 x float> %1711, float %1120, i64 2, !dbg !17
  %1713 = insertelement <4 x float> %1712, float %1126, i64 3, !dbg !17
  %1714 = fmul <4 x float> %1713, %1531, !dbg !17
  %1715 = fptrunc <4 x float> %1714 to <4 x half>, !dbg !18
  %1716 = extractelement <4 x half> %1715, i64 3, !dbg !18
  %1717 = extractelement <4 x half> %1715, i64 2, !dbg !18
  %1718 = extractelement <4 x half> %1715, i64 1, !dbg !18
  %1719 = extractelement <4 x half> %1715, i64 0, !dbg !18
  %address.local1029 = or disjoint i64 %address.local861, 96, !dbg !19
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1029, !dbg !19
  store half %1719, ptr addrspace(1) %address.local1030, align 2, !dbg !19
  %address.local1032 = or disjoint i64 %address.local861, 97, !dbg !19
  %address.local1033 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1032, !dbg !19
  store half %1718, ptr addrspace(1) %address.local1033, align 2, !dbg !19
  %address.local1035 = or disjoint i64 %address.local861, 98, !dbg !19
  %address.local1036 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1035, !dbg !19
  store half %1717, ptr addrspace(1) %address.local1036, align 2, !dbg !19
  %address.local1038 = or disjoint i64 %address.local861, 99, !dbg !19
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1038, !dbg !19
  store half %1716, ptr addrspace(1) %address.local1039, align 2, !dbg !19
  %address.local1041 = or disjoint i64 %address.local861, 112, !dbg !19
  %address.local1042 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1041, !dbg !19
  store half %1709, ptr addrspace(1) %address.local1042, align 2, !dbg !19
  %address.local1044 = or disjoint i64 %address.local861, 113, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1044, !dbg !19
  store half %1708, ptr addrspace(1) %address.local1045, align 2, !dbg !19
  %address.local1047 = or disjoint i64 %address.local861, 114, !dbg !19
  %address.local1048 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1047, !dbg !19
  store half %1707, ptr addrspace(1) %address.local1048, align 2, !dbg !19
  %address.local1050 = or i64 %1598, 115, !dbg !19
  %address.local1051 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local859, i64 %address.local1050, !dbg !19
  store half %1706, ptr addrspace(1) %address.local1051, align 2, !dbg !19
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1029, !dbg !19
  store half %1699, ptr addrspace(1) %address.local1056, align 2, !dbg !19
  %address.local1058 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1032, !dbg !19
  store half %1698, ptr addrspace(1) %address.local1058, align 2, !dbg !19
  %address.local1060 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1035, !dbg !19
  store half %1697, ptr addrspace(1) %address.local1060, align 2, !dbg !19
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1038, !dbg !19
  store half %1696, ptr addrspace(1) %address.local1062, align 2, !dbg !19
  %address.local1064 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1041, !dbg !19
  store half %1689, ptr addrspace(1) %address.local1064, align 2, !dbg !19
  %address.local1066 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1044, !dbg !19
  store half %1688, ptr addrspace(1) %address.local1066, align 2, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1047, !dbg !19
  store half %1687, ptr addrspace(1) %address.local1068, align 2, !dbg !19
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local887, i64 %address.local1050, !dbg !19
  store half %1686, ptr addrspace(1) %address.local1070, align 2, !dbg !19
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

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/best_noncausal_thread_multichain")
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
