; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/noncausal.input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/noncausal.input.mlir"

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
  %address.local279 = zext nneg i32 %37 to i64, !dbg !9
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local279, !dbg !9
  %address.local281 = ptrtoint ptr addrspace(3) %address.local280 to i64, !dbg !9
  %address.local282 = trunc i64 %address.local281 to i32, !dbg !9
  %38 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local282), !dbg !9
  %address.local283 = lshr i32 %address.thread, 4, !dbg !9
  %address.local284 = and i32 %address.local283, 8, !dbg !9
  %address.local285 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local286 = and i32 %address.local285, 16256, !dbg !9
  %address.local287 = or disjoint i32 %address.local284, %address.local286, !dbg !9
  %39 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1285 = shl i64 %34, 8, !dbg !9
  %40 = add i64 %.idx, %.idx1285, !dbg !9
  %41 = shl nuw nsw i32 %address.local287, 1, !dbg !9
  %42 = trunc i64 %40 to i32, !dbg !9
  %dma.buffer.offset1283 = or disjoint i32 %41, %42, !dbg !9
  %43 = insertelement <2 x i64> poison, i64 %39, i64 0, !dbg !9
  %44 = bitcast <2 x i64> %43 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1284 = shufflevector <4 x i32> %44, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %45 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1283, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %address.local292 = or disjoint i64 %address.local279, 2048, !dbg !9
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local292, !dbg !9
  %address.local294 = ptrtoint ptr addrspace(3) %address.local293 to i64, !dbg !9
  %address.local295 = trunc i64 %address.local294 to i32, !dbg !9
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local295), !dbg !9
  %address.local301 = or disjoint i32 %address.local287, 16, !dbg !9
  %47 = shl nuw nsw i32 %address.local301, 1, !dbg !9
  %dma.buffer.offset1281 = or disjoint i32 %47, %42, !dbg !9
  %48 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1281, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %49 = shl nuw nsw i64 %31, 19, !dbg !9
  %50 = add i64 %49, %35, !dbg !9
  %address.local304 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local279, !dbg !9
  %address.local305 = ptrtoint ptr addrspace(3) %address.local304 to i64, !dbg !9
  %address.local306 = trunc i64 %address.local305 to i32, !dbg !9
  %51 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local306), !dbg !9
  %52 = trunc i64 %50 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local287, %52, !dbg !9
  %53 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1279 = or disjoint i32 %53, 64, !dbg !9
  %54 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1279, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %address.local308 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local292, !dbg !9
  %address.local309 = ptrtoint ptr addrspace(3) %address.local308 to i64, !dbg !9
  %address.local310 = trunc i64 %address.local309 to i32, !dbg !9
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local310), !dbg !9
  %.tr1289 = or disjoint i32 %address.local301, %52, !dbg !9
  %56 = shl i32 %.tr1289, 1, !dbg !9
  %dma.buffer.offset1277 = or disjoint i32 %56, 64, !dbg !9
  %57 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1277, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %address.local312 = lshr i32 %address.thread, 2, !dbg !12
  %address.local313 = and i32 %address.local312, 4, !dbg !12
  %address.local314 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local315 = and i32 %address.local314, 384, !dbg !12
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !12
  %address.local318 = and i32 %36, 120, !dbg !12
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !12
  %address.local320 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local321 = and i32 %address.local320, 1024, !dbg !12
  %address.local322 = or disjoint i32 %address.local319, %address.local321, !dbg !12
  %address.local323 = zext nneg i32 %address.local322 to i64, !dbg !12
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local323, !dbg !12
  %58 = load <4 x half>, ptr addrspace(3) %address.local324, align 8, !dbg !12
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local324, i64 4096, !dbg !12
  %59 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %60 = fmul <4 x half> %58, splat (half 0xH3015), !dbg !12
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local324, i64 1024, !dbg !12
  %61 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !12
  %62 = fmul <4 x half> %59, splat (half 0xH3015), !dbg !12
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %address.local324, i64 5120, !dbg !12
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !12
  %64 = fmul <4 x half> %61, splat (half 0xH3015), !dbg !12
  %65 = fmul <4 x half> %63, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1275 = or disjoint i32 %53, 128, !dbg !9
  %66 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1275, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %dma.buffer.offset1273 = or disjoint i32 %56, 128, !dbg !9
  %67 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1273, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %address.local324, i64 8192, !dbg !12
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !12
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local324, i64 12288, !dbg !12
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !12
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local324, i64 9216, !dbg !12
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !12
  %71 = fmul <4 x half> %68, splat (half 0xH3015), !dbg !12
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %address.local324, i64 13312, !dbg !12
  %72 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !12
  %73 = fmul <4 x half> %69, splat (half 0xH3015), !dbg !12
  %74 = fmul <4 x half> %70, splat (half 0xH3015), !dbg !12
  %75 = fmul <4 x half> %72, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1271 = or disjoint i32 %53, 192, !dbg !9
  %76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1271, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %dma.buffer.offset1269 = or disjoint i32 %56, 192, !dbg !9
  %77 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1269, <4 x i32> %dma.buffer.rsrc1284) #10, !dbg !9
  %78 = load <4 x half>, ptr addrspace(3) %address.local324, align 8, !dbg !12
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !12
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !12
  %81 = fmul <4 x half> %78, splat (half 0xH3015), !dbg !12
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !12
  %83 = fmul <4 x half> %79, splat (half 0xH3015), !dbg !12
  %84 = fmul <4 x half> %80, splat (half 0xH3015), !dbg !12
  %85 = fmul <4 x half> %82, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !12
  %87 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !12
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !12
  %89 = fmul <4 x half> %86, splat (half 0xH3015), !dbg !12
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !12
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
  %k0.prime.1893 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.1894 = shl nuw nsw i64 0, 14, !dbg !13
  %k0.prime.1895 = or disjoint i64 %k0.prime.1893, %k0.prime.1894, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1332 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1333 = shl nuw nsw i64 %k0.prime.kdma.source1332, 9, !dbg !13
  %k0.prime.kdma.source1334 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !13
  %k0.prime.kdma.source1335 = add nuw nsw i64 %k0.prime.kdma.source1333, %k0.prime.kdma.source1334, !dbg !13
  %k0.prime.kdma.source1336 = lshr i64 %k0.prime.kdma.source, 5, !dbg !13
  %k0.prime.kdma.source1337 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1338 = and i64 %k0.prime.kdma.source1337, 1, !dbg !13
  %k0.prime.kdma.source1339 = and i64 %k0.prime.kdma.source1332, %k0.prime.kdma.source1338, !dbg !13
  %k0.prime.kdma.source1340 = or disjoint i64 %k0.prime.kdma.source1336, %k0.prime.kdma.source1339, !dbg !13
  %k0.prime.kdma.source1341 = shl nuw nsw i64 %k0.prime.kdma.source1340, 10, !dbg !13
  %k0.prime.kdma.source1342 = sub nsw i64 %k0.prime.kdma.source1335, %k0.prime.kdma.source1341, !dbg !13
  %k0.prime.kdma.source1343 = and i64 %k0.prime.kdma.source1342, 16256, !dbg !13
  %k0.prime.kdma.source1344 = lshr i32 %kdma.tid, 4, !dbg !13
  %k0.prime.kdma.source1345 = and i32 %k0.prime.kdma.source1344, 8, !dbg !13
  %k0.prime.kdma.source1346 = zext nneg i32 %k0.prime.kdma.source1345 to i64, !dbg !13
  %k0.prime.kdma.source1347 = or disjoint i64 %k0.prime.kdma.source1343, %k0.prime.kdma.source1346, !dbg !13
  %k0.prime.kdma.source1348 = or disjoint i64 %k0.prime.1895, %k0.prime.kdma.source1347, !dbg !13
  %k0.prime.kdma.source1349 = trunc nuw i64 %k0.prime.kdma.source1348 to i32, !dbg !13
  %k0.prime.kdma.source1350 = shl nuw nsw i32 %k0.prime.kdma.source1349, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1350, %kdma.intra, !dbg !13
  %k0.prime.1896 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1897 = insertelement <2 x i64> poison, i64 %k0.prime.1896, i64 0, !dbg !13
  %k0.prime.1898 = bitcast <2 x i64> %k0.prime.1897 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1268 = shufflevector <4 x i32> %k0.prime.1898, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1899 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1372 = add i32 %38, 512, !dbg !13
  %k0.prime.kdma.source1351 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1352 = lshr i64 %k0.prime.kdma.source1351, 4, !dbg !13
  %k0.prime.kdma.source1353 = shl nuw nsw i64 %k0.prime.kdma.source1352, 9, !dbg !13
  %k0.prime.kdma.source1354 = shl nuw nsw i64 %k0.prime.kdma.source1351, 7, !dbg !13
  %k0.prime.kdma.source1355 = add nuw nsw i64 %k0.prime.kdma.source1353, %k0.prime.kdma.source1354, !dbg !13
  %k0.prime.kdma.source1356 = lshr i64 %k0.prime.kdma.source1351, 5, !dbg !13
  %k0.prime.kdma.source1357 = lshr i64 %k0.prime.kdma.source1351, 2, !dbg !13
  %k0.prime.kdma.source1358 = and i64 %k0.prime.kdma.source1357, 1, !dbg !13
  %k0.prime.kdma.source1359 = and i64 %k0.prime.kdma.source1352, %k0.prime.kdma.source1358, !dbg !13
  %k0.prime.kdma.source1360 = add nuw nsw i64 %k0.prime.kdma.source1356, %k0.prime.kdma.source1359, !dbg !13
  %k0.prime.kdma.source1361 = shl nuw nsw i64 %k0.prime.kdma.source1360, 10, !dbg !13
  %k0.prime.kdma.source1362 = sub nsw i64 %k0.prime.kdma.source1355, %k0.prime.kdma.source1361, !dbg !13
  %k0.prime.kdma.source1363 = and i64 %k0.prime.kdma.source1362, 16256, !dbg !13
  %k0.prime.kdma.source1367 = or disjoint i64 %k0.prime.kdma.source1363, %k0.prime.kdma.source1346, !dbg !13
  %k0.prime.kdma.source1368 = or disjoint i64 %k0.prime.1895, %k0.prime.kdma.source1367, !dbg !13
  %k0.prime.kdma.source1369 = trunc nuw i64 %k0.prime.kdma.source1368 to i32, !dbg !13
  %k0.prime.kdma.source1370 = shl nuw nsw i32 %k0.prime.kdma.source1369, 1, !dbg !13
  %k0.prime.kdma.offset1371 = or disjoint i32 %k0.prime.kdma.source1370, %kdma.intra.hi, !dbg !13
  %k0.prime.1900 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1372, i32 %k0.prime.kdma.offset1371, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.source1374 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !13
  %k0.prime.kdma.source1375 = lshr i64 %k0.prime.kdma.source1374, 4, !dbg !13
  %k0.prime.kdma.source1376 = shl nuw nsw i64 %k0.prime.kdma.source1375, 9, !dbg !13
  %k0.prime.kdma.source1378 = add nuw nsw i64 %k0.prime.kdma.source1376, %k0.prime.kdma.source1334, !dbg !13
  %k0.prime.kdma.source1379 = lshr i64 %k0.prime.kdma.source1374, 5, !dbg !13
  %k0.prime.kdma.source1382 = and i64 %k0.prime.kdma.source1375, %k0.prime.kdma.source1338, !dbg !13
  %k0.prime.kdma.source1383 = or disjoint i64 %k0.prime.kdma.source1379, %k0.prime.kdma.source1382, !dbg !13
  %k0.prime.kdma.source1384 = shl nuw nsw i64 %k0.prime.kdma.source1383, 10, !dbg !13
  %k0.prime.kdma.source1385 = sub nsw i64 %k0.prime.kdma.source1378, %k0.prime.kdma.source1384, !dbg !13
  %k0.prime.kdma.source1386 = and i64 %k0.prime.kdma.source1385, 16256, !dbg !13
  %k0.prime.kdma.source1389 = or disjoint i32 %k0.prime.kdma.source1345, 16, !dbg !13
  %k0.prime.kdma.source1390 = zext nneg i32 %k0.prime.kdma.source1389 to i64, !dbg !13
  %k0.prime.kdma.source1391 = or disjoint i64 %k0.prime.kdma.source1386, %k0.prime.kdma.source1390, !dbg !13
  %k0.prime.kdma.source1392 = or disjoint i64 %k0.prime.1895, %k0.prime.kdma.source1391, !dbg !13
  %k0.prime.kdma.source1393 = trunc nuw i64 %k0.prime.kdma.source1392 to i32, !dbg !13
  %k0.prime.kdma.source1394 = shl nuw nsw i32 %k0.prime.kdma.source1393, 1, !dbg !13
  %k0.prime.kdma.offset1395 = add nuw i32 %k0.prime.kdma.source1394, %kdma.intra, !dbg !13
  %k0.prime.1901 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.prime.kdma.offset1395, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.prime.kdma.lds1420 = add i32 %46, 512, !dbg !13
  %k0.prime.kdma.source1398 = or disjoint i64 %k0.prime.kdma.source1351, 256, !dbg !13
  %k0.prime.kdma.source1399 = lshr i64 %k0.prime.kdma.source1398, 4, !dbg !13
  %k0.prime.kdma.source1400 = shl nuw nsw i64 %k0.prime.kdma.source1399, 9, !dbg !13
  %k0.prime.kdma.source1402 = add nuw nsw i64 %k0.prime.kdma.source1400, %k0.prime.kdma.source1354, !dbg !13
  %k0.prime.kdma.source1403 = lshr i64 %k0.prime.kdma.source1398, 5, !dbg !13
  %k0.prime.kdma.source1406 = and i64 %k0.prime.kdma.source1399, %k0.prime.kdma.source1358, !dbg !13
  %k0.prime.kdma.source1407 = add nuw nsw i64 %k0.prime.kdma.source1403, %k0.prime.kdma.source1406, !dbg !13
  %k0.prime.kdma.source1408 = shl nuw nsw i64 %k0.prime.kdma.source1407, 10, !dbg !13
  %k0.prime.kdma.source1409 = sub nsw i64 %k0.prime.kdma.source1402, %k0.prime.kdma.source1408, !dbg !13
  %k0.prime.kdma.source1410 = and i64 %k0.prime.kdma.source1409, 16256, !dbg !13
  %k0.prime.kdma.source1415 = or disjoint i64 %k0.prime.kdma.source1410, %k0.prime.kdma.source1390, !dbg !13
  %k0.prime.kdma.source1416 = or disjoint i64 %k0.prime.1895, %k0.prime.kdma.source1415, !dbg !13
  %k0.prime.kdma.source1417 = trunc nuw i64 %k0.prime.kdma.source1416 to i32, !dbg !13
  %k0.prime.kdma.source1418 = shl nuw nsw i32 %k0.prime.kdma.source1417, 1, !dbg !13
  %k0.prime.kdma.offset1419 = add nuw i32 %k0.prime.kdma.source1418, %kdma.intra.hi, !dbg !13
  %k0.prime.1902 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1420, i32 %k0.prime.kdma.offset1419, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %100, !dbg !6

100:                                              ; preds = %k0.join, %4
  %101 = phi i64 [ %1004, %k0.join ], [ 0, %4 ], !dbg !6
  %102 = phi [2 x <8 x float>] [ %1072, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %103 = phi [2 x <1 x float>] [ %847, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %104 = phi [2 x <1 x float>] [ %lazy.base.join, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %105 = phi [2 x <8 x float>] [ %1076, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %106 = phi [2 x <8 x float>] [ %1080, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %107 = phi [2 x <8 x float>] [ %1084, %k0.join ], [ zeroinitializer, %4 ], !dbg !6
  %108 = icmp samesign ult i64 %101, 32, !dbg !6
  br i1 %108, label %109, label %1085, !dbg !6

109:                                              ; preds = %100
  %110 = shl i64 %25, 19, !dbg !13
  %111 = and i64 %110, 16252928, !dbg !13
  %112 = shl nuw nsw i64 %101, 14, !dbg !13
  %113 = or disjoint i64 %111, %112, !dbg !13
  %address.thread375 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !13, !range !30, !frisk.register_lifetimes !1
  %address.local376 = zext nneg i32 %address.thread375 to i64, !dbg !9
  %address.local382 = lshr i64 %address.local376, 2, !dbg !13
  %114 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0, !dbg !13
  %116 = bitcast <2 x i64> %115 to <4 x i32>, !dbg !13
  %dma.buffer.rsrc1268 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %kdma.source1332 = lshr i64 %kdma.source, 4, !dbg !13
  %kdma.source1333 = shl nuw nsw i64 %kdma.source1332, 9, !dbg !13
  %kdma.source1334 = shl nuw nsw i64 %kdma.source, 7, !dbg !13
  %kdma.source1335 = add nuw nsw i64 %kdma.source1333, %kdma.source1334, !dbg !13
  %kdma.source1336 = lshr i64 %kdma.source, 5, !dbg !13
  %kdma.source1337 = lshr i64 %kdma.source, 2, !dbg !13
  %kdma.source1338 = and i64 %kdma.source1337, 1, !dbg !13
  %kdma.source1339 = and i64 %kdma.source1332, %kdma.source1338, !dbg !13
  %kdma.source1340 = or disjoint i64 %kdma.source1336, %kdma.source1339, !dbg !13
  %kdma.source1341 = shl nuw nsw i64 %kdma.source1340, 10, !dbg !13
  %kdma.source1342 = sub nsw i64 %kdma.source1335, %kdma.source1341, !dbg !13
  %kdma.source1343 = and i64 %kdma.source1342, 16256, !dbg !13
  %kdma.source1344 = lshr i32 %kdma.tid, 4, !dbg !13
  %kdma.source1345 = and i32 %kdma.source1344, 8, !dbg !13
  %kdma.source1346 = zext nneg i32 %kdma.source1345 to i64, !dbg !13
  %kdma.source1347 = or disjoint i64 %kdma.source1343, %kdma.source1346, !dbg !13
  %kdma.source1348 = or disjoint i64 %113, %kdma.source1347, !dbg !13
  %kdma.source1349 = trunc nuw i64 %kdma.source1348 to i32, !dbg !13
  %kdma.source1350 = shl nuw nsw i32 %kdma.source1349, 1, !dbg !13
  %kdma.offset = or disjoint i32 %kdma.source1350, %kdma.intra, !dbg !13
  %kdma.source1351 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %kdma.source1352 = lshr i64 %kdma.source1351, 4, !dbg !13
  %kdma.source1353 = shl nuw nsw i64 %kdma.source1352, 9, !dbg !13
  %kdma.source1354 = shl nuw nsw i64 %kdma.source1351, 7, !dbg !13
  %kdma.source1355 = add nuw nsw i64 %kdma.source1353, %kdma.source1354, !dbg !13
  %kdma.source1356 = lshr i64 %kdma.source1351, 5, !dbg !13
  %kdma.source1357 = lshr i64 %kdma.source1351, 2, !dbg !13
  %kdma.source1358 = and i64 %kdma.source1357, 1, !dbg !13
  %kdma.source1359 = and i64 %kdma.source1352, %kdma.source1358, !dbg !13
  %kdma.source1360 = add nuw nsw i64 %kdma.source1356, %kdma.source1359, !dbg !13
  %kdma.source1361 = shl nuw nsw i64 %kdma.source1360, 10, !dbg !13
  %kdma.source1362 = sub nsw i64 %kdma.source1355, %kdma.source1361, !dbg !13
  %kdma.source1363 = and i64 %kdma.source1362, 16256, !dbg !13
  %kdma.source1367 = or disjoint i64 %kdma.source1363, %kdma.source1346, !dbg !13
  %kdma.source1368 = or disjoint i64 %113, %kdma.source1367, !dbg !13
  %kdma.source1369 = trunc nuw i64 %kdma.source1368 to i32, !dbg !13
  %kdma.source1370 = shl nuw nsw i32 %kdma.source1369, 1, !dbg !13
  %kdma.offset1371 = or disjoint i32 %kdma.source1370, %kdma.intra.hi, !dbg !13
  %kdma.lds1372 = add i32 %38, 512, !dbg !13
  %kdma.source1374 = or disjoint i64 %kdma.source, 256, !dbg !13
  %kdma.source1375 = lshr i64 %kdma.source1374, 4, !dbg !13
  %kdma.source1376 = shl nuw nsw i64 %kdma.source1375, 9, !dbg !13
  %kdma.source1378 = add nuw nsw i64 %kdma.source1376, %kdma.source1334, !dbg !13
  %kdma.source1379 = lshr i64 %kdma.source1374, 5, !dbg !13
  %kdma.source1382 = and i64 %kdma.source1375, %kdma.source1338, !dbg !13
  %kdma.source1383 = or disjoint i64 %kdma.source1379, %kdma.source1382, !dbg !13
  %kdma.source1384 = shl nuw nsw i64 %kdma.source1383, 10, !dbg !13
  %kdma.source1385 = sub nsw i64 %kdma.source1378, %kdma.source1384, !dbg !13
  %kdma.source1386 = and i64 %kdma.source1385, 16256, !dbg !13
  %kdma.source1389 = or disjoint i32 %kdma.source1345, 16, !dbg !13
  %kdma.source1390 = zext nneg i32 %kdma.source1389 to i64, !dbg !13
  %kdma.source1391 = or disjoint i64 %kdma.source1386, %kdma.source1390, !dbg !13
  %kdma.source1392 = or disjoint i64 %113, %kdma.source1391, !dbg !13
  %kdma.source1393 = trunc nuw i64 %kdma.source1392 to i32, !dbg !13
  %kdma.source1394 = shl nuw nsw i32 %kdma.source1393, 1, !dbg !13
  %kdma.offset1395 = add nuw i32 %kdma.source1394, %kdma.intra, !dbg !13
  %kdma.source1398 = or disjoint i64 %kdma.source1351, 256, !dbg !13
  %kdma.source1399 = lshr i64 %kdma.source1398, 4, !dbg !13
  %kdma.source1400 = shl nuw nsw i64 %kdma.source1399, 9, !dbg !13
  %kdma.source1402 = add nuw nsw i64 %kdma.source1400, %kdma.source1354, !dbg !13
  %kdma.source1403 = lshr i64 %kdma.source1398, 5, !dbg !13
  %kdma.source1406 = and i64 %kdma.source1399, %kdma.source1358, !dbg !13
  %kdma.source1407 = add nuw nsw i64 %kdma.source1403, %kdma.source1406, !dbg !13
  %kdma.source1408 = shl nuw nsw i64 %kdma.source1407, 10, !dbg !13
  %kdma.source1409 = sub nsw i64 %kdma.source1402, %kdma.source1408, !dbg !13
  %kdma.source1410 = and i64 %kdma.source1409, 16256, !dbg !13
  %kdma.source1415 = or disjoint i64 %kdma.source1410, %kdma.source1390, !dbg !13
  %kdma.source1416 = or disjoint i64 %113, %kdma.source1415, !dbg !13
  %kdma.source1417 = trunc nuw i64 %kdma.source1416 to i32, !dbg !13
  %kdma.source1418 = shl nuw nsw i32 %kdma.source1417, 1, !dbg !13
  %kdma.offset1419 = add nuw i32 %kdma.source1418, %kdma.intra.hi, !dbg !13
  %kdma.lds1420 = add i32 %46, 512, !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %117 = or disjoint i64 %113, 32, !dbg !13
  %kdma.source1438 = or disjoint i64 %117, %kdma.source1347, !dbg !13
  %kdma.source1439 = trunc nuw i64 %kdma.source1438 to i32, !dbg !13
  %kdma.source1440 = shl nuw nsw i32 %kdma.source1439, 1, !dbg !13
  %kdma.offset1441 = or disjoint i32 %kdma.source1440, %kdma.intra, !dbg !13
  %118 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1441, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1460 = or disjoint i64 %117, %kdma.source1367, !dbg !13
  %kdma.source1461 = trunc nuw i64 %kdma.source1460 to i32, !dbg !13
  %kdma.source1462 = shl nuw nsw i32 %kdma.source1461, 1, !dbg !13
  %kdma.offset1463 = or disjoint i32 %kdma.source1462, %kdma.intra.hi, !dbg !13
  %kdma.lds1464 = add i32 %51, 512, !dbg !13
  %119 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1464, i32 %kdma.offset1463, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1484 = or disjoint i64 %117, %kdma.source1391, !dbg !13
  %kdma.source1485 = trunc nuw i64 %kdma.source1484 to i32, !dbg !13
  %kdma.source1486 = shl nuw nsw i32 %kdma.source1485, 1, !dbg !13
  %kdma.offset1487 = add nuw i32 %kdma.source1486, %kdma.intra, !dbg !13
  %120 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1487, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1508 = or disjoint i64 %117, %kdma.source1415, !dbg !13
  %kdma.source1509 = trunc nuw i64 %kdma.source1508 to i32, !dbg !13
  %kdma.source1510 = shl nuw nsw i32 %kdma.source1509, 1, !dbg !13
  %kdma.offset1511 = add nuw i32 %kdma.source1510, %kdma.intra.hi, !dbg !13
  %kdma.lds1512 = add i32 %55, 512, !dbg !13
  %121 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1512, i32 %kdma.offset1511, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %address.local419 = shl nuw nsw i64 %address.local376, 1, !dbg !15
  %address.local420 = and i64 %address.local419, 24, !dbg !15
  %address.local422 = shl i64 %address.local376, 62, !dbg !15
  %address.local423 = ashr exact i64 %address.local422, 57, !dbg !15
  %address.local424 = and i64 %address.local423, -64, !dbg !15
  %address.local427 = and i64 %address.local382, 12, !dbg !12
  %address.local429 = and i64 %address.local376, 32, !dbg !12
  %address.local430 = icmp eq i64 %address.local429, 0, !dbg !12
  %address.local431 = select i1 %address.local430, i64 0, i64 1016, !dbg !12
  %.idx1306 = mul nuw nsw i64 %address.local376, 320, !dbg !15
  %122 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1306, !dbg !15
  %.idx1307 = mul nsw i64 %address.local382, -1280, !dbg !15
  %123 = getelementptr i8, ptr addrspace(3) %122, i64 %.idx1307, !dbg !15
  %124 = getelementptr [2 x i8], ptr addrspace(3) %123, i64 %address.local420, !dbg !15
  %125 = getelementptr [2 x i8], ptr addrspace(3) %124, i64 %address.local424, !dbg !15
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local427, !dbg !15
  %address.local433 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local431, !dbg !15
  %127 = add nsw i64 %.idx1306, %.idx1307, !dbg !15
  %.idx1823 = shl nuw nsw i64 %address.local420, 1, !dbg !15
  %128 = or disjoint i64 %127, %.idx1823, !dbg !15
  %129 = or disjoint i64 %address.local424, %address.local427, !dbg !15
  %130 = add nsw i64 %129, %address.local431, !dbg !15
  %131 = shl nsw i64 %130, 1, !dbg !15
  %132 = add nsw i64 %131, %128, !dbg !15
  %133 = lshr i64 %132, 6, !dbg !15
  %134 = and i64 %133, 8, !dbg !15
  %135 = xor i64 %132, %134, !dbg !15
  %136 = sub i64 %135, %132, !dbg !15
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local433, i64 %136, !dbg !15
  %137 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15
  %lazy.base = extractvalue [2 x <1 x float>] %104, 0, !dbg !15
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %lazy.base1908 = extractvalue [2 x <1 x float>] %104, 1, !dbg !15
  %lazy.seed1909 = shufflevector <1 x float> %lazy.base1908, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !15
  %138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %137, <4 x float> %lazy.seed) #9, !dbg !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !15
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15
  %address.local444 = shl nuw nsw i32 %address.thread375, 1, !dbg !15
  %address.local445 = and i32 %address.local444, 24, !dbg !15
  %address.local446 = shl nuw nsw i32 %address.thread375, 8, !dbg !15
  %address.local447 = and i32 %address.local446, 768, !dbg !15
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !15
  %address.local449 = shl nuw nsw i32 %address.thread375, 5, !dbg !15
  %address.local450 = and i32 %address.local449, 96, !dbg !15
  %address.local451 = or disjoint i32 %address.local448, %address.local450, !dbg !15
  %address.local452 = zext nneg i32 %address.local451 to i64, !dbg !15
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %99) #9, !dbg !15, !frisk.register_lifetimes !1
  %address.local453 = lshr i32 %address.thread375, 1, !dbg !15
  %address.local454 = and i32 %address.local453, 1, !dbg !15
  %address.local455 = zext nneg i32 %address.local454 to i64, !dbg !15
  %address.local456 = add nsw i64 %address.base, %address.local455, !dbg !15
  %address.local457 = shl i64 %address.local456, 6, !dbg !15
  %address.local458 = sub i64 %address.local452, %address.local457, !dbg !15
  %140 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local458, !dbg !15
  %141 = getelementptr [2 x i8], ptr addrspace(3) %140, i64 %address.local427, !dbg !15
  %142 = getelementptr [2 x i8], ptr addrspace(3) %141, i64 %address.local431, !dbg !15
  %address.local462 = getelementptr i8, ptr addrspace(3) %142, i64 64, !dbg !15
  %.idx18261829 = add i64 %address.local458, %address.local427, !dbg !15
  %143 = add i64 %.idx18261829, %address.local431, !dbg !15
  %144 = shl i64 %143, 1, !dbg !15
  %145 = add i64 %144, 64, !dbg !15
  %146 = lshr i64 %145, 6, !dbg !15
  %147 = and i64 %146, 8, !dbg !15
  %148 = xor i64 %145, %147, !dbg !15
  %149 = sub i64 %148, %145, !dbg !15
  %kdma.swizzled1697 = getelementptr i8, ptr addrspace(3) %address.local462, i64 %149, !dbg !15
  %150 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1697, align 8, !dbg !15, !frisk.k_lds_pair !32
  %kdma.gep1701 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 128, !dbg !15
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1701, align 8, !dbg !15, !frisk.k_lds_pair !32
  %kdma.gep1698 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 4096, !dbg !15
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1698, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1702 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 4224, !dbg !15
  %153 = load <4 x half>, ptr addrspace(3) %kdma.gep1702, align 8, !dbg !15, !frisk.k_lds_pair !33
  %kdma.gep1699 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !15
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1699, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1703 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !15
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1703, align 8, !dbg !15, !frisk.k_lds_pair !34
  %kdma.gep1700 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !15
  %156 = load <4 x half>, ptr addrspace(3) %kdma.gep1700, align 8, !dbg !15, !frisk.k_lds_pair !35
  %kdma.gep1704 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !15
  %157 = load <4 x half>, ptr addrspace(3) %kdma.gep1704, align 8, !dbg !15, !frisk.k_lds_pair !35
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %139, <4 x float> %138) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1705 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 1024, !dbg !15
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1705, align 8, !dbg !15, !frisk.k_lds_pair !31
  %kdma.gep1709 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 1152, !dbg !15
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1709, align 8, !dbg !15, !frisk.k_lds_pair !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %60, <4 x half> %150, <4 x float> %lazy.seed) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1706 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 5120, !dbg !15
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1706, align 8, !dbg !15, !frisk.k_lds_pair !36
  %kdma.gep1710 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 5248, !dbg !15
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1710, align 8, !dbg !15, !frisk.k_lds_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %164 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %62, <4 x half> %152, <4 x float> %161) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1707 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !15
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1707, align 8, !dbg !15, !frisk.k_lds_pair !37
  %kdma.gep1708 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !15
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1708, align 8, !dbg !15, !frisk.k_lds_pair !37
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
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %137, <4 x float> %lazy.seed1909) #9, !dbg !15
  %180 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %139, <4 x float> %179) #9, !dbg !15
  %181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %150, <4 x float> %lazy.seed1909) #9, !dbg !15
  %182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %152, <4 x float> %181) #9, !dbg !15
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %154, <4 x float> %lazy.seed1909) #9, !dbg !15
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %156, <4 x float> %183) #9, !dbg !15
  %185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %151, <4 x float> %lazy.seed1909) #9, !dbg !15
  %186 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %153, <4 x float> %185) #9, !dbg !15
  %187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %155, <4 x float> %lazy.seed1909) #9, !dbg !15
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %157, <4 x float> %187) #9, !dbg !15
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %159, <4 x float> %lazy.seed1909) #9, !dbg !15
  %190 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %162, <4 x float> %189) #9, !dbg !15
  %191 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %165, <4 x float> %lazy.seed1909) #9, !dbg !15
  %192 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %166, <4 x float> %191) #9, !dbg !15
  %193 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %64, <4 x half> %160, <4 x float> %lazy.seed1909) #9, !dbg !15
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %65, <4 x half> %163, <4 x float> %193) #9, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %195 = or disjoint i64 %113, 64, !dbg !13
  %kdma.source1530 = or disjoint i64 %195, %kdma.source1347, !dbg !13
  %kdma.source1531 = trunc nuw i64 %kdma.source1530 to i32, !dbg !13
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !13
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra, !dbg !13
  %196 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %kdma.offset1533, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1552 = or disjoint i64 %195, %kdma.source1367, !dbg !13
  %kdma.source1553 = trunc nuw i64 %kdma.source1552 to i32, !dbg !13
  %kdma.source1554 = shl nuw nsw i32 %kdma.source1553, 1, !dbg !13
  %kdma.offset1555 = or disjoint i32 %kdma.source1554, %kdma.intra.hi, !dbg !13
  %197 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1372, i32 %kdma.offset1555, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1576 = or disjoint i64 %195, %kdma.source1391, !dbg !13
  %kdma.source1577 = trunc nuw i64 %kdma.source1576 to i32, !dbg !13
  %kdma.source1578 = shl nuw nsw i32 %kdma.source1577, 1, !dbg !13
  %kdma.offset1579 = add nuw i32 %kdma.source1578, %kdma.intra, !dbg !13
  %198 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %kdma.offset1579, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1600 = or disjoint i64 %195, %kdma.source1415, !dbg !13
  %kdma.source1601 = trunc nuw i64 %kdma.source1600 to i32, !dbg !13
  %kdma.source1602 = shl nuw nsw i32 %kdma.source1601, 1, !dbg !13
  %kdma.offset1603 = add nuw i32 %kdma.source1602, %kdma.intra.hi, !dbg !13
  %199 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1420, i32 %kdma.offset1603, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.gep1727 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !15
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !15
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !15, !frisk.k_lds_pair !38
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !15
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !15
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !15, !frisk.k_lds_pair !39
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 8192, !dbg !15
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 8320, !dbg !15
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !15, !frisk.k_lds_pair !40
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 12288, !dbg !15
  %206 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 12416, !dbg !15
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !15, !frisk.k_lds_pair !41
  %kdma.gep1735 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !15
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !15, !frisk.k_lds_pair !42
  %kdma.gep1739 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !15
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !15, !frisk.k_lds_pair !42
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %210 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %200, <4 x float> %158) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1736 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !15
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !15, !frisk.k_lds_pair !43
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !15
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !15, !frisk.k_lds_pair !43
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %73, <4 x half> %202, <4 x float> %210) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1737 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 9216, !dbg !15
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !15, !frisk.k_lds_pair !44
  %kdma.gep1741 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 9344, !dbg !15
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !15, !frisk.k_lds_pair !44
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %216 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %204, <4 x float> %164) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %kdma.gep1738 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 13312, !dbg !15
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !15, !frisk.k_lds_pair !45
  %kdma.gep1742 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 13440, !dbg !15
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !15, !frisk.k_lds_pair !45
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
  %kdma.source1622 = or disjoint i64 %248, %kdma.source1347, !dbg !13
  %kdma.source1623 = trunc nuw i64 %kdma.source1622 to i32, !dbg !13
  %kdma.source1624 = shl nuw nsw i32 %kdma.source1623, 1, !dbg !13
  %kdma.offset1625 = or disjoint i32 %kdma.source1624, %kdma.intra, !dbg !13
  %249 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %kdma.offset1625, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1644 = or disjoint i64 %248, %kdma.source1367, !dbg !13
  %kdma.source1645 = trunc nuw i64 %kdma.source1644 to i32, !dbg !13
  %kdma.source1646 = shl nuw nsw i32 %kdma.source1645, 1, !dbg !13
  %kdma.offset1647 = or disjoint i32 %kdma.source1646, %kdma.intra.hi, !dbg !13
  %250 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1464, i32 %kdma.offset1647, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1668 = or disjoint i64 %248, %kdma.source1391, !dbg !13
  %kdma.source1669 = trunc nuw i64 %kdma.source1668 to i32, !dbg !13
  %kdma.source1670 = shl nuw nsw i32 %kdma.source1669, 1, !dbg !13
  %kdma.offset1671 = add nuw i32 %kdma.source1670, %kdma.intra, !dbg !13
  %251 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %kdma.offset1671, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %kdma.source1692 = or disjoint i64 %248, %kdma.source1415, !dbg !13
  %kdma.source1693 = trunc nuw i64 %kdma.source1692 to i32, !dbg !13
  %kdma.source1694 = shl nuw nsw i32 %kdma.source1693, 1, !dbg !13
  %kdma.offset1695 = add nuw i32 %kdma.source1694, %kdma.intra.hi, !dbg !13
  %252 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1512, i32 %kdma.offset1695, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !15, !frisk.k_lds_pair !46
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1699, align 8, !dbg !15, !frisk.k_lds_pair !46
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !15, !frisk.k_lds_pair !47
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1700, align 8, !dbg !15, !frisk.k_lds_pair !47
  %257 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1697, align 8, !dbg !15, !frisk.k_lds_pair !48
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1701, align 8, !dbg !15, !frisk.k_lds_pair !48
  %259 = load <4 x half>, ptr addrspace(3) %kdma.gep1698, align 8, !dbg !15, !frisk.k_lds_pair !49
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1702, align 8, !dbg !15, !frisk.k_lds_pair !49
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1703, align 8, !dbg !15, !frisk.k_lds_pair !50
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1707, align 8, !dbg !15, !frisk.k_lds_pair !50
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %253, <4 x float> %213) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1704, align 8, !dbg !15, !frisk.k_lds_pair !51
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1708, align 8, !dbg !15, !frisk.k_lds_pair !51
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %255, <4 x float> %263) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1705, align 8, !dbg !15, !frisk.k_lds_pair !52
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1709, align 8, !dbg !15, !frisk.k_lds_pair !52
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %257, <4 x float> %219) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_lds_prefetch !1
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1706, align 8, !dbg !15, !frisk.k_lds_pair !53
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1710, align 8, !dbg !15, !frisk.k_lds_pair !53
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
  %address.local791 = shl nuw nsw i64 %address.local376, 5, !dbg !16
  %address.local792 = and i64 %address.local791, 8064, !dbg !16
  %address.local793 = shl nuw nsw i64 %address.local376, 3, !dbg !9
  %address.local794 = shl nuw nsw i64 %address.local382, 5, !dbg !16
  %address.local795 = sub nsw i64 %address.local793, %address.local794, !dbg !16
  %address.local796 = add nsw i64 %address.local792, %address.local795, !dbg !16
  %301 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !16
  %.idx13161317 = add nsw i64 %113, %address.local796, !dbg !16
  %.idx13161317.tr = trunc i64 %.idx13161317 to i32, !dbg !16
  %dma.buffer.offset1251 = shl i32 %.idx13161317.tr, 1, !dbg !16
  %302 = insertelement <2 x i64> poison, i64 %301, i64 0, !dbg !16
  %303 = bitcast <2 x i64> %302 to <4 x i32>, !dbg !16
  %dma.buffer.rsrc1252 = shufflevector <4 x i32> %303, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !16
  %304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1251, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %address.local798 = shl nuw nsw i32 %address.thread375, 3, !dbg !16
  %address.local799 = and i32 %address.local798, 24, !dbg !16
  %address.local801 = and i32 %address.local449, 8064, !dbg !16
  %address.local802 = or disjoint i32 %address.local799, %address.local801, !dbg !16
  %address.local803 = or disjoint i32 %address.local802, 8192, !dbg !16
  %305 = trunc nuw i64 %113 to i32, !dbg !16
  %.idx13181319.tr = or disjoint i32 %address.local803, %305, !dbg !16
  %dma.buffer.offset1249 = shl nuw nsw i32 %.idx13181319.tr, 1, !dbg !16
  %306 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1249, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !15, !frisk.k_tail_pair !32
  %308 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !15, !frisk.k_tail_pair !32
  %309 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !15, !frisk.k_tail_pair !33
  %310 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !15, !frisk.k_tail_pair !33
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !15, !frisk.k_tail_pair !34
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !15, !frisk.k_tail_pair !34
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !15, !frisk.k_tail_pair !35
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !15, !frisk.k_tail_pair !35
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
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !15, !frisk.k_tail_pair !31
  %325 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !15, !frisk.k_tail_pair !31
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
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !15, !frisk.k_tail_pair !36
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !15, !frisk.k_tail_pair !36
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %310, <4 x float> %326) #9, !dbg !15
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !15, !frisk.k_tail_prefetch !1
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !15, !frisk.k_tail_pair !37
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !15, !frisk.k_tail_pair !37
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
  %349 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !15, !frisk.k_tail_pair !38
  %350 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !15, !frisk.k_tail_pair !38
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
  %lazy.positive1910 = call float @llvm.maxnum.f32(float %498, float 0x0000000000000000), !dbg !23
  %lazy.delta1911 = select i1 %lazy.first, float %498, float %lazy.positive1910, !dbg !23
  %lazy.negated1912 = fsub float 0x0000000000000000, %lazy.delta1911, !dbg !23
  %lazy.exp1913 = call float @llvm.amdgcn.exp2.f32(float %lazy.negated1912), !dbg !23
  %lazy.alpha1914 = select i1 %lazy.first, float 0x3FF0000000000000, float %lazy.exp1913, !dbg !23
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted = fsub <4 x float> %316, %lazy.delta.v, !dbg !23
  %lazy.delta.v.lane1915 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1916 = shufflevector <1 x float> %lazy.delta.v.lane1915, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1917 = fsub <4 x float> %323, %lazy.delta.v1916, !dbg !23
  %lazy.delta.v.lane1918 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1919 = shufflevector <1 x float> %lazy.delta.v.lane1918, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1920 = fsub <4 x float> %337, %lazy.delta.v1919, !dbg !23
  %lazy.delta.v.lane1921 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1922 = shufflevector <1 x float> %lazy.delta.v.lane1921, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1923 = fsub <4 x float> %351, %lazy.delta.v1922, !dbg !23
  %lazy.delta.v.lane1924 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1925 = shufflevector <1 x float> %lazy.delta.v.lane1924, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1926 = fsub <4 x float> %361, %lazy.delta.v1925, !dbg !23
  %lazy.delta.v.lane1927 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1928 = shufflevector <1 x float> %lazy.delta.v.lane1927, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1929 = fsub <4 x float> %371, %lazy.delta.v1928, !dbg !23
  %lazy.delta.v.lane1930 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1931 = shufflevector <1 x float> %lazy.delta.v.lane1930, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1932 = fsub <4 x float> %381, %lazy.delta.v1931, !dbg !23
  %lazy.delta.v.lane1933 = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.delta.v1934 = shufflevector <1 x float> %lazy.delta.v.lane1933, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1935 = fsub <4 x float> %391, %lazy.delta.v1934, !dbg !23
  %lazy.delta.v.lane1936 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1937 = shufflevector <1 x float> %lazy.delta.v.lane1936, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1938 = fsub <4 x float> %404, %lazy.delta.v1937, !dbg !23
  %lazy.delta.v.lane1939 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1940 = shufflevector <1 x float> %lazy.delta.v.lane1939, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1941 = fsub <4 x float> %410, %lazy.delta.v1940, !dbg !23
  %lazy.delta.v.lane1942 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1943 = shufflevector <1 x float> %lazy.delta.v.lane1942, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1944 = fsub <4 x float> %416, %lazy.delta.v1943, !dbg !23
  %lazy.delta.v.lane1945 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1946 = shufflevector <1 x float> %lazy.delta.v.lane1945, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1947 = fsub <4 x float> %422, %lazy.delta.v1946, !dbg !23
  %lazy.delta.v.lane1948 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1949 = shufflevector <1 x float> %lazy.delta.v.lane1948, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1950 = fsub <4 x float> %428, %lazy.delta.v1949, !dbg !23
  %lazy.delta.v.lane1951 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1952 = shufflevector <1 x float> %lazy.delta.v.lane1951, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1953 = fsub <4 x float> %434, %lazy.delta.v1952, !dbg !23
  %lazy.delta.v.lane1954 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1955 = shufflevector <1 x float> %lazy.delta.v.lane1954, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1956 = fsub <4 x float> %440, %lazy.delta.v1955, !dbg !23
  %lazy.delta.v.lane1957 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.delta.v1958 = shufflevector <1 x float> %lazy.delta.v.lane1957, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !23
  %lazy.shifted1959 = fsub <4 x float> %446, %lazy.delta.v1958, !dbg !23
  %lazy.state = extractvalue [2 x <1 x float>] %103, 0, !dbg !23
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !23
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !23
  %lazy.state1960 = extractvalue [2 x <1 x float>] %103, 1, !dbg !23
  %lazy.alpha.v.lane1961 = insertelement <1 x float> poison, float %lazy.alpha1914, i64 0, !dbg !23
  %lazy.scaled1962 = fmul <1 x float> %lazy.state1960, %lazy.alpha.v.lane1961, !dbg !23
  %lazy.state.next1963 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1962, 1, !dbg !23
  %lazy.state1964 = extractvalue [2 x <8 x float>] %107, 0, !dbg !23
  %lazy.alpha.v.lane1965 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane1965, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1966 = fmul <8 x float> %lazy.state1964, %lazy.alpha.v, !dbg !23
  %lazy.state.next1967 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1966, 0, !dbg !23
  %lazy.state1968 = extractvalue [2 x <8 x float>] %107, 1, !dbg !23
  %lazy.alpha.v.lane1969 = insertelement <1 x float> poison, float %lazy.alpha1914, i64 0, !dbg !23
  %lazy.alpha.v1970 = shufflevector <1 x float> %lazy.alpha.v.lane1969, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1971 = fmul <8 x float> %lazy.state1968, %lazy.alpha.v1970, !dbg !23
  %lazy.state.next1972 = insertvalue [2 x <8 x float>] %lazy.state.next1967, <8 x float> %lazy.scaled1971, 1, !dbg !23
  %lazy.state1973 = extractvalue [2 x <8 x float>] %106, 0, !dbg !23
  %lazy.alpha.v.lane1974 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v1975 = shufflevector <1 x float> %lazy.alpha.v.lane1974, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1976 = fmul <8 x float> %lazy.state1973, %lazy.alpha.v1975, !dbg !23
  %lazy.state.next1977 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1976, 0, !dbg !23
  %lazy.state1978 = extractvalue [2 x <8 x float>] %106, 1, !dbg !23
  %lazy.alpha.v.lane1979 = insertelement <1 x float> poison, float %lazy.alpha1914, i64 0, !dbg !23
  %lazy.alpha.v1980 = shufflevector <1 x float> %lazy.alpha.v.lane1979, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1981 = fmul <8 x float> %lazy.state1978, %lazy.alpha.v1980, !dbg !23
  %lazy.state.next1982 = insertvalue [2 x <8 x float>] %lazy.state.next1977, <8 x float> %lazy.scaled1981, 1, !dbg !23
  %lazy.state1983 = extractvalue [2 x <8 x float>] %105, 0, !dbg !23
  %lazy.alpha.v.lane1984 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v1985 = shufflevector <1 x float> %lazy.alpha.v.lane1984, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1986 = fmul <8 x float> %lazy.state1983, %lazy.alpha.v1985, !dbg !23
  %lazy.state.next1987 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1986, 0, !dbg !23
  %lazy.state1988 = extractvalue [2 x <8 x float>] %105, 1, !dbg !23
  %lazy.alpha.v.lane1989 = insertelement <1 x float> poison, float %lazy.alpha1914, i64 0, !dbg !23
  %lazy.alpha.v1990 = shufflevector <1 x float> %lazy.alpha.v.lane1989, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1991 = fmul <8 x float> %lazy.state1988, %lazy.alpha.v1990, !dbg !23
  %lazy.state.next1992 = insertvalue [2 x <8 x float>] %lazy.state.next1987, <8 x float> %lazy.scaled1991, 1, !dbg !23
  %lazy.state1993 = extractvalue [2 x <8 x float>] %102, 0, !dbg !23
  %lazy.alpha.v.lane1994 = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !23
  %lazy.alpha.v1995 = shufflevector <1 x float> %lazy.alpha.v.lane1994, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled1996 = fmul <8 x float> %lazy.state1993, %lazy.alpha.v1995, !dbg !23
  %lazy.state.next1997 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1996, 0, !dbg !23
  %lazy.state1998 = extractvalue [2 x <8 x float>] %102, 1, !dbg !23
  %lazy.alpha.v.lane1999 = insertelement <1 x float> poison, float %lazy.alpha1914, i64 0, !dbg !23
  %lazy.alpha.v2000 = shufflevector <1 x float> %lazy.alpha.v.lane1999, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !23
  %lazy.scaled2001 = fmul <8 x float> %lazy.state1998, %lazy.alpha.v2000, !dbg !23
  %lazy.state.next2002 = insertvalue [2 x <8 x float>] %lazy.state.next1997, <8 x float> %lazy.scaled2001, 1, !dbg !23
  %lazy.base.old = extractvalue [2 x <1 x float>] %104, 0, !dbg !23
  %lazy.delta.b.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !23
  %lazy.base.next = fsub <1 x float> %lazy.base.old, %lazy.delta.b.lane, !dbg !23
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !23
  %lazy.base.old2003 = extractvalue [2 x <1 x float>] %104, 1, !dbg !23
  %lazy.delta.b.lane2004 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !23
  %lazy.base.next2005 = fsub <1 x float> %lazy.base.old2003, %lazy.delta.b.lane2004, !dbg !23
  %lazy.base.state2006 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2005, 1, !dbg !23
  br label %lazy.join, !dbg !23

lazy.join:                                        ; preds = %lazy.slow, %109
  %lazy.score = phi <4 x float> [ %316, %109 ], [ %lazy.shifted, %lazy.slow ], !dbg !23
  %lazy.score2007 = phi <4 x float> [ %323, %109 ], [ %lazy.shifted1917, %lazy.slow ], !dbg !23
  %lazy.score2008 = phi <4 x float> [ %337, %109 ], [ %lazy.shifted1920, %lazy.slow ], !dbg !23
  %lazy.score2009 = phi <4 x float> [ %351, %109 ], [ %lazy.shifted1923, %lazy.slow ], !dbg !23
  %lazy.score2010 = phi <4 x float> [ %361, %109 ], [ %lazy.shifted1926, %lazy.slow ], !dbg !23
  %lazy.score2011 = phi <4 x float> [ %371, %109 ], [ %lazy.shifted1929, %lazy.slow ], !dbg !23
  %lazy.score2012 = phi <4 x float> [ %381, %109 ], [ %lazy.shifted1932, %lazy.slow ], !dbg !23
  %lazy.score2013 = phi <4 x float> [ %391, %109 ], [ %lazy.shifted1935, %lazy.slow ], !dbg !23
  %lazy.score2014 = phi <4 x float> [ %404, %109 ], [ %lazy.shifted1938, %lazy.slow ], !dbg !23
  %lazy.score2015 = phi <4 x float> [ %410, %109 ], [ %lazy.shifted1941, %lazy.slow ], !dbg !23
  %lazy.score2016 = phi <4 x float> [ %416, %109 ], [ %lazy.shifted1944, %lazy.slow ], !dbg !23
  %lazy.score2017 = phi <4 x float> [ %422, %109 ], [ %lazy.shifted1947, %lazy.slow ], !dbg !23
  %lazy.score2018 = phi <4 x float> [ %428, %109 ], [ %lazy.shifted1950, %lazy.slow ], !dbg !23
  %lazy.score2019 = phi <4 x float> [ %434, %109 ], [ %lazy.shifted1953, %lazy.slow ], !dbg !23
  %lazy.score2020 = phi <4 x float> [ %440, %109 ], [ %lazy.shifted1956, %lazy.slow ], !dbg !23
  %lazy.score2021 = phi <4 x float> [ %446, %109 ], [ %lazy.shifted1959, %lazy.slow ], !dbg !23
  %lazy.state2022 = phi [2 x <1 x float>] [ %103, %109 ], [ %lazy.state.next1963, %lazy.slow ], !dbg !23
  %lazy.state2023 = phi [2 x <8 x float>] [ %107, %109 ], [ %lazy.state.next1972, %lazy.slow ], !dbg !23
  %lazy.state2024 = phi [2 x <8 x float>] [ %106, %109 ], [ %lazy.state.next1982, %lazy.slow ], !dbg !23
  %lazy.state2025 = phi [2 x <8 x float>] [ %105, %109 ], [ %lazy.state.next1992, %lazy.slow ], !dbg !23
  %lazy.state2026 = phi [2 x <8 x float>] [ %102, %109 ], [ %lazy.state.next2002, %lazy.slow ], !dbg !23
  %lazy.base.join = phi [2 x <1 x float>] [ %104, %109 ], [ %lazy.base.state2006, %lazy.slow ], !dbg !23
  %499 = extractvalue [2 x <1 x float>] %lazy.state2022, 0, !dbg !23
  %500 = extractelement <1 x float> %499, i64 0, !dbg !23
  %501 = extractvalue [2 x <1 x float>] %lazy.state2022, 1, !dbg !23
  %502 = extractelement <1 x float> %501, i64 0, !dbg !23
  %503 = extractvalue [2 x <8 x float>] %lazy.state2026, 0, !dbg !27
  %504 = shufflevector <8 x float> %503, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %505 = shufflevector <8 x float> %503, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %506 = extractvalue [2 x <8 x float>] %lazy.state2026, 1, !dbg !27
  %507 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %508 = shufflevector <8 x float> %506, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %.idx13201321 = add nsw i64 %117, %address.local796, !dbg !16
  %.idx13201321.tr = trunc i64 %.idx13201321 to i32, !dbg !16
  %dma.buffer.offset1247 = shl i32 %.idx13201321.tr, 1, !dbg !16
  %509 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1247, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %510 = trunc nuw i64 %117 to i32, !dbg !16
  %.idx13221323.tr = or disjoint i32 %address.local803, %510, !dbg !16
  %dma.buffer.offset1245 = shl nuw nsw i32 %.idx13221323.tr, 1, !dbg !16
  %511 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset1245, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %address.local809 = and i32 %address.local798, 120, !dbg !14
  %address.local811 = and i32 %address.local449, 1536, !dbg !14
  %address.local812 = or disjoint i32 %address.local809, %address.local811, !dbg !14
  %address.local813 = zext nneg i32 %address.local812 to i64, !dbg !14
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local813, !dbg !14
  %512 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #0, !dbg !14
  %513 = bitcast <4 x i32> %512 to <8 x half>, !dbg !14
  %address.local820 = or disjoint i32 %address.local812, 128, !dbg !14
  %address.local821 = zext nneg i32 %address.local820 to i64, !dbg !14
  %address.local822 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local821, !dbg !14
  %514 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14
  %515 = bitcast <4 x i32> %514 to <8 x half>, !dbg !14
  %address.local828 = or disjoint i32 %address.local812, 256, !dbg !14
  %address.local829 = zext nneg i32 %address.local828 to i64, !dbg !14
  %address.local830 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local829, !dbg !14
  %516 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !14
  %517 = bitcast <4 x i32> %516 to <8 x half>, !dbg !14
  %address.local836 = or disjoint i32 %address.local812, 384, !dbg !14
  %address.local837 = zext nneg i32 %address.local836 to i64, !dbg !14
  %address.local838 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local837, !dbg !14
  %518 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local838, i32 0) #0, !dbg !14
  %519 = bitcast <4 x i32> %518 to <8 x half>, !dbg !14
  %address.local844 = or disjoint i32 %address.local812, 2048, !dbg !14
  %address.local845 = zext nneg i32 %address.local844 to i64, !dbg !14
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local845, !dbg !14
  %520 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local846, i32 0) #0, !dbg !14
  %521 = bitcast <4 x i32> %520 to <8 x half>, !dbg !14
  %address.local852 = or disjoint i32 %address.local812, 2176, !dbg !14
  %address.local853 = zext nneg i32 %address.local852 to i64, !dbg !14
  %address.local854 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local853, !dbg !14
  %522 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local854, i32 0) #0, !dbg !14
  %523 = bitcast <4 x i32> %522 to <8 x half>, !dbg !14
  %address.local860 = or disjoint i32 %address.local812, 2304, !dbg !14
  %address.local861 = zext nneg i32 %address.local860 to i64, !dbg !14
  %address.local862 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local861, !dbg !14
  %524 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local862, i32 0) #0, !dbg !14
  %525 = bitcast <4 x i32> %524 to <8 x half>, !dbg !14
  %address.local868 = or disjoint i32 %address.local812, 2432, !dbg !14
  %address.local869 = zext nneg i32 %address.local868 to i64, !dbg !14
  %address.local870 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local869, !dbg !14
  %526 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local870, i32 0) #0, !dbg !14
  %527 = shufflevector <8 x half> %513, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %528 = extractelement <4 x float> %lazy.score, i64 0, !dbg !24
  %529 = call float @llvm.amdgcn.exp2.f32(float %528), !dbg !24
  %530 = insertelement <4 x float> poison, float %529, i64 0, !dbg !26
  %531 = extractelement <4 x float> %lazy.score, i64 1, !dbg !24
  %532 = call float @llvm.amdgcn.exp2.f32(float %531), !dbg !24
  %533 = insertelement <4 x float> %530, float %532, i64 1, !dbg !26
  %534 = extractelement <4 x float> %lazy.score, i64 2, !dbg !24
  %535 = call float @llvm.amdgcn.exp2.f32(float %534), !dbg !24
  %536 = insertelement <4 x float> %533, float %535, i64 2, !dbg !26
  %537 = extractelement <4 x float> %lazy.score, i64 3, !dbg !24
  %538 = call float @llvm.amdgcn.exp2.f32(float %537), !dbg !24
  %539 = insertelement <4 x float> %536, float %538, i64 3, !dbg !26
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %539) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i32 0, !dbg !26
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i32 1, !dbg !26
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i32 2, !dbg !26
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i32 3, !dbg !26
  %.cvtpack.p0 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1), !dbg !26
  %.cvtpack.p1 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3), !dbg !26
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i32 0, !dbg !26
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i32 1, !dbg !26
  %540 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !26
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %540) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %527, <4 x float> %504) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %542 = shufflevector <8 x half> %513, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %543 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %542, <4 x float> %505) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %544 = extractelement <4 x float> %lazy.score2014, i64 0, !dbg !24
  %545 = call float @llvm.amdgcn.exp2.f32(float %544), !dbg !24
  %546 = insertelement <4 x float> poison, float %545, i64 0, !dbg !26
  %547 = extractelement <4 x float> %lazy.score2014, i64 1, !dbg !24
  %548 = call float @llvm.amdgcn.exp2.f32(float %547), !dbg !24
  %549 = insertelement <4 x float> %546, float %548, i64 1, !dbg !26
  %550 = extractelement <4 x float> %lazy.score2014, i64 2, !dbg !24
  %551 = call float @llvm.amdgcn.exp2.f32(float %550), !dbg !24
  %552 = insertelement <4 x float> %549, float %551, i64 2, !dbg !26
  %553 = extractelement <4 x float> %lazy.score2014, i64 3, !dbg !24
  %554 = call float @llvm.amdgcn.exp2.f32(float %553), !dbg !24
  %555 = insertelement <4 x float> %552, float %554, i64 3, !dbg !26
  %math.result262 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %555) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02027 = extractelement <4 x float> %math.result262, i32 0, !dbg !26
  %.cvtpack.f12028 = extractelement <4 x float> %math.result262, i32 1, !dbg !26
  %.cvtpack.f22029 = extractelement <4 x float> %math.result262, i32 2, !dbg !26
  %.cvtpack.f32030 = extractelement <4 x float> %math.result262, i32 3, !dbg !26
  %.cvtpack.p02031 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02027, float %.cvtpack.f12028), !dbg !26
  %.cvtpack.p12032 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22029, float %.cvtpack.f32030), !dbg !26
  %.cvtpack.v02033 = insertelement <2 x i32> poison, i32 %.cvtpack.p02031, i32 0, !dbg !26
  %.cvtpack.v12034 = insertelement <2 x i32> %.cvtpack.v02033, i32 %.cvtpack.p12032, i32 1, !dbg !26
  %556 = bitcast <2 x i32> %.cvtpack.v12034 to <4 x half>, !dbg !26
  %math.packed263 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %556) #9, !dbg !24, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %557 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %527, <4 x float> %507) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %558 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %542, <4 x float> %508) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %559 = shufflevector <8 x half> %515, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %560 = extractelement <4 x float> %lazy.score2007, i64 0, !dbg !24
  %561 = call float @llvm.amdgcn.exp2.f32(float %560), !dbg !24
  %562 = fadd float %529, %561, !dbg !25
  %k0.sum.anchor = call float asm sideeffect "", "=v,0"(float %562) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %563 = insertelement <4 x float> poison, float %561, i64 0, !dbg !26
  %564 = extractelement <4 x float> %lazy.score2007, i64 1, !dbg !24
  %565 = call float @llvm.amdgcn.exp2.f32(float %564), !dbg !24
  %566 = fadd float %532, %565, !dbg !25
  %k0.sum.anchor1830 = call float asm sideeffect "", "=v,0"(float %566) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %567 = insertelement <4 x float> %563, float %565, i64 1, !dbg !26
  %568 = extractelement <4 x float> %lazy.score2007, i64 2, !dbg !24
  %569 = call float @llvm.amdgcn.exp2.f32(float %568), !dbg !24
  %570 = fadd float %535, %569, !dbg !25
  %k0.sum.anchor1831 = call float asm sideeffect "", "=v,0"(float %570) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %571 = insertelement <4 x float> %567, float %569, i64 2, !dbg !26
  %572 = extractelement <4 x float> %lazy.score2007, i64 3, !dbg !24
  %573 = call float @llvm.amdgcn.exp2.f32(float %572), !dbg !24
  %574 = fadd float %538, %573, !dbg !25
  %k0.sum.anchor1832 = call float asm sideeffect "", "=v,0"(float %574) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %575 = insertelement <4 x float> %571, float %573, i64 3, !dbg !26
  %math.result248 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %575) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02035 = extractelement <4 x float> %math.result248, i32 0, !dbg !26
  %.cvtpack.f12036 = extractelement <4 x float> %math.result248, i32 1, !dbg !26
  %.cvtpack.f22037 = extractelement <4 x float> %math.result248, i32 2, !dbg !26
  %.cvtpack.f32038 = extractelement <4 x float> %math.result248, i32 3, !dbg !26
  %.cvtpack.p02039 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02035, float %.cvtpack.f12036), !dbg !26
  %.cvtpack.p12040 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22037, float %.cvtpack.f32038), !dbg !26
  %.cvtpack.v02041 = insertelement <2 x i32> poison, i32 %.cvtpack.p02039, i32 0, !dbg !26
  %.cvtpack.v12042 = insertelement <2 x i32> %.cvtpack.v02041, i32 %.cvtpack.p12040, i32 1, !dbg !26
  %576 = bitcast <2 x i32> %.cvtpack.v12042 to <4 x half>, !dbg !26
  %math.packed249 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %576) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %577 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %559, <4 x float> %541) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %578 = shufflevector <8 x half> %515, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %579 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %578, <4 x float> %543) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %580 = extractelement <4 x float> %lazy.score2015, i64 0, !dbg !24
  %581 = call float @llvm.amdgcn.exp2.f32(float %580), !dbg !24
  %582 = fadd float %545, %581, !dbg !25
  %k0.sum.anchor1833 = call float asm sideeffect "", "=v,0"(float %582) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %583 = insertelement <4 x float> poison, float %581, i64 0, !dbg !26
  %584 = extractelement <4 x float> %lazy.score2015, i64 1, !dbg !24
  %585 = call float @llvm.amdgcn.exp2.f32(float %584), !dbg !24
  %586 = fadd float %548, %585, !dbg !25
  %k0.sum.anchor1834 = call float asm sideeffect "", "=v,0"(float %586) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %587 = insertelement <4 x float> %583, float %585, i64 1, !dbg !26
  %588 = extractelement <4 x float> %lazy.score2015, i64 2, !dbg !24
  %589 = call float @llvm.amdgcn.exp2.f32(float %588), !dbg !24
  %590 = fadd float %551, %589, !dbg !25
  %k0.sum.anchor1835 = call float asm sideeffect "", "=v,0"(float %590) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %591 = insertelement <4 x float> %587, float %589, i64 2, !dbg !26
  %592 = extractelement <4 x float> %lazy.score2015, i64 3, !dbg !24
  %593 = call float @llvm.amdgcn.exp2.f32(float %592), !dbg !24
  %594 = fadd float %554, %593, !dbg !25
  %k0.sum.anchor1836 = call float asm sideeffect "", "=v,0"(float %594) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %595 = insertelement <4 x float> %591, float %593, i64 3, !dbg !26
  %math.result264 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %595) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02043 = extractelement <4 x float> %math.result264, i32 0, !dbg !26
  %.cvtpack.f12044 = extractelement <4 x float> %math.result264, i32 1, !dbg !26
  %.cvtpack.f22045 = extractelement <4 x float> %math.result264, i32 2, !dbg !26
  %.cvtpack.f32046 = extractelement <4 x float> %math.result264, i32 3, !dbg !26
  %.cvtpack.p02047 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02043, float %.cvtpack.f12044), !dbg !26
  %.cvtpack.p12048 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22045, float %.cvtpack.f32046), !dbg !26
  %.cvtpack.v02049 = insertelement <2 x i32> poison, i32 %.cvtpack.p02047, i32 0, !dbg !26
  %.cvtpack.v12050 = insertelement <2 x i32> %.cvtpack.v02049, i32 %.cvtpack.p12048, i32 1, !dbg !26
  %596 = bitcast <2 x i32> %.cvtpack.v12050 to <4 x half>, !dbg !26
  %math.packed265 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %596) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %597 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %559, <4 x float> %557) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %598 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %578, <4 x float> %558) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %599 = shufflevector <8 x half> %517, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %600 = extractelement <4 x float> %lazy.score2008, i64 0, !dbg !24
  %601 = call float @llvm.amdgcn.exp2.f32(float %600), !dbg !24
  %602 = fadd float %k0.sum.anchor, %601, !dbg !25
  %k0.sum.anchor1837 = call float asm sideeffect "", "=v,0"(float %602) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %603 = insertelement <4 x float> poison, float %601, i64 0, !dbg !26
  %604 = extractelement <4 x float> %lazy.score2008, i64 1, !dbg !24
  %605 = call float @llvm.amdgcn.exp2.f32(float %604), !dbg !24
  %606 = fadd float %k0.sum.anchor1830, %605, !dbg !25
  %k0.sum.anchor1838 = call float asm sideeffect "", "=v,0"(float %606) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %607 = insertelement <4 x float> %603, float %605, i64 1, !dbg !26
  %608 = extractelement <4 x float> %lazy.score2008, i64 2, !dbg !24
  %609 = call float @llvm.amdgcn.exp2.f32(float %608), !dbg !24
  %610 = fadd float %k0.sum.anchor1831, %609, !dbg !25
  %k0.sum.anchor1839 = call float asm sideeffect "", "=v,0"(float %610) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %611 = insertelement <4 x float> %607, float %609, i64 2, !dbg !26
  %612 = extractelement <4 x float> %lazy.score2008, i64 3, !dbg !24
  %613 = call float @llvm.amdgcn.exp2.f32(float %612), !dbg !24
  %614 = fadd float %k0.sum.anchor1832, %613, !dbg !25
  %k0.sum.anchor1840 = call float asm sideeffect "", "=v,0"(float %614) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %615 = insertelement <4 x float> %611, float %613, i64 3, !dbg !26
  %math.result250 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %615) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02051 = extractelement <4 x float> %math.result250, i32 0, !dbg !26
  %.cvtpack.f12052 = extractelement <4 x float> %math.result250, i32 1, !dbg !26
  %.cvtpack.f22053 = extractelement <4 x float> %math.result250, i32 2, !dbg !26
  %.cvtpack.f32054 = extractelement <4 x float> %math.result250, i32 3, !dbg !26
  %.cvtpack.p02055 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02051, float %.cvtpack.f12052), !dbg !26
  %.cvtpack.p12056 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22053, float %.cvtpack.f32054), !dbg !26
  %.cvtpack.v02057 = insertelement <2 x i32> poison, i32 %.cvtpack.p02055, i32 0, !dbg !26
  %.cvtpack.v12058 = insertelement <2 x i32> %.cvtpack.v02057, i32 %.cvtpack.p12056, i32 1, !dbg !26
  %616 = bitcast <2 x i32> %.cvtpack.v12058 to <4 x half>, !dbg !26
  %math.packed251 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %616) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %599, <4 x float> %577) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %618 = shufflevector <8 x half> %517, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %619 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %618, <4 x float> %579) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %620 = extractelement <4 x float> %lazy.score2016, i64 0, !dbg !24
  %621 = call float @llvm.amdgcn.exp2.f32(float %620), !dbg !24
  %622 = fadd float %k0.sum.anchor1833, %621, !dbg !25
  %k0.sum.anchor1841 = call float asm sideeffect "", "=v,0"(float %622) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %623 = insertelement <4 x float> poison, float %621, i64 0, !dbg !26
  %624 = extractelement <4 x float> %lazy.score2016, i64 1, !dbg !24
  %625 = call float @llvm.amdgcn.exp2.f32(float %624), !dbg !24
  %626 = fadd float %k0.sum.anchor1834, %625, !dbg !25
  %k0.sum.anchor1842 = call float asm sideeffect "", "=v,0"(float %626) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %627 = insertelement <4 x float> %623, float %625, i64 1, !dbg !26
  %628 = extractelement <4 x float> %lazy.score2016, i64 2, !dbg !24
  %629 = call float @llvm.amdgcn.exp2.f32(float %628), !dbg !24
  %630 = fadd float %k0.sum.anchor1835, %629, !dbg !25
  %k0.sum.anchor1843 = call float asm sideeffect "", "=v,0"(float %630) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %631 = insertelement <4 x float> %627, float %629, i64 2, !dbg !26
  %632 = extractelement <4 x float> %lazy.score2016, i64 3, !dbg !24
  %633 = call float @llvm.amdgcn.exp2.f32(float %632), !dbg !24
  %634 = fadd float %k0.sum.anchor1836, %633, !dbg !25
  %k0.sum.anchor1844 = call float asm sideeffect "", "=v,0"(float %634) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %635 = insertelement <4 x float> %631, float %633, i64 3, !dbg !26
  %math.result266 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %635) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02059 = extractelement <4 x float> %math.result266, i32 0, !dbg !26
  %.cvtpack.f12060 = extractelement <4 x float> %math.result266, i32 1, !dbg !26
  %.cvtpack.f22061 = extractelement <4 x float> %math.result266, i32 2, !dbg !26
  %.cvtpack.f32062 = extractelement <4 x float> %math.result266, i32 3, !dbg !26
  %.cvtpack.p02063 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02059, float %.cvtpack.f12060), !dbg !26
  %.cvtpack.p12064 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22061, float %.cvtpack.f32062), !dbg !26
  %.cvtpack.v02065 = insertelement <2 x i32> poison, i32 %.cvtpack.p02063, i32 0, !dbg !26
  %.cvtpack.v12066 = insertelement <2 x i32> %.cvtpack.v02065, i32 %.cvtpack.p12064, i32 1, !dbg !26
  %636 = bitcast <2 x i32> %.cvtpack.v12066 to <4 x half>, !dbg !26
  %math.packed267 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %636) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %599, <4 x float> %597) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %638 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %618, <4 x float> %598) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %639 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %640 = extractelement <4 x float> %lazy.score2009, i64 0, !dbg !24
  %641 = call float @llvm.amdgcn.exp2.f32(float %640), !dbg !24
  %642 = fadd float %k0.sum.anchor1837, %641, !dbg !25
  %k0.sum.anchor1845 = call float asm sideeffect "", "=v,0"(float %642) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %643 = insertelement <4 x float> poison, float %641, i64 0, !dbg !26
  %644 = extractelement <4 x float> %lazy.score2009, i64 1, !dbg !24
  %645 = call float @llvm.amdgcn.exp2.f32(float %644), !dbg !24
  %646 = fadd float %k0.sum.anchor1838, %645, !dbg !25
  %k0.sum.anchor1846 = call float asm sideeffect "", "=v,0"(float %646) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %647 = insertelement <4 x float> %643, float %645, i64 1, !dbg !26
  %648 = extractelement <4 x float> %lazy.score2009, i64 2, !dbg !24
  %649 = call float @llvm.amdgcn.exp2.f32(float %648), !dbg !24
  %650 = fadd float %k0.sum.anchor1839, %649, !dbg !25
  %k0.sum.anchor1847 = call float asm sideeffect "", "=v,0"(float %650) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %651 = insertelement <4 x float> %647, float %649, i64 2, !dbg !26
  %652 = extractelement <4 x float> %lazy.score2009, i64 3, !dbg !24
  %653 = call float @llvm.amdgcn.exp2.f32(float %652), !dbg !24
  %654 = fadd float %k0.sum.anchor1840, %653, !dbg !25
  %k0.sum.anchor1848 = call float asm sideeffect "", "=v,0"(float %654) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %655 = insertelement <4 x float> %651, float %653, i64 3, !dbg !26
  %math.result252 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %655) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02067 = extractelement <4 x float> %math.result252, i32 0, !dbg !26
  %.cvtpack.f12068 = extractelement <4 x float> %math.result252, i32 1, !dbg !26
  %.cvtpack.f22069 = extractelement <4 x float> %math.result252, i32 2, !dbg !26
  %.cvtpack.f32070 = extractelement <4 x float> %math.result252, i32 3, !dbg !26
  %.cvtpack.p02071 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02067, float %.cvtpack.f12068), !dbg !26
  %.cvtpack.p12072 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22069, float %.cvtpack.f32070), !dbg !26
  %.cvtpack.v02073 = insertelement <2 x i32> poison, i32 %.cvtpack.p02071, i32 0, !dbg !26
  %.cvtpack.v12074 = insertelement <2 x i32> %.cvtpack.v02073, i32 %.cvtpack.p12072, i32 1, !dbg !26
  %656 = bitcast <2 x i32> %.cvtpack.v12074 to <4 x half>, !dbg !26
  %math.packed253 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %656) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %639, <4 x float> %617) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %658 = shufflevector <8 x half> %519, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %658, <4 x float> %619) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %660 = extractelement <4 x float> %lazy.score2017, i64 0, !dbg !24
  %661 = call float @llvm.amdgcn.exp2.f32(float %660), !dbg !24
  %662 = fadd float %k0.sum.anchor1841, %661, !dbg !25
  %k0.sum.anchor1849 = call float asm sideeffect "", "=v,0"(float %662) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %663 = insertelement <4 x float> poison, float %661, i64 0, !dbg !26
  %664 = extractelement <4 x float> %lazy.score2017, i64 1, !dbg !24
  %665 = call float @llvm.amdgcn.exp2.f32(float %664), !dbg !24
  %666 = fadd float %k0.sum.anchor1842, %665, !dbg !25
  %k0.sum.anchor1850 = call float asm sideeffect "", "=v,0"(float %666) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %667 = insertelement <4 x float> %663, float %665, i64 1, !dbg !26
  %668 = extractelement <4 x float> %lazy.score2017, i64 2, !dbg !24
  %669 = call float @llvm.amdgcn.exp2.f32(float %668), !dbg !24
  %670 = fadd float %k0.sum.anchor1843, %669, !dbg !25
  %k0.sum.anchor1851 = call float asm sideeffect "", "=v,0"(float %670) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %671 = insertelement <4 x float> %667, float %669, i64 2, !dbg !26
  %672 = extractelement <4 x float> %lazy.score2017, i64 3, !dbg !24
  %673 = call float @llvm.amdgcn.exp2.f32(float %672), !dbg !24
  %674 = fadd float %k0.sum.anchor1844, %673, !dbg !25
  %k0.sum.anchor1852 = call float asm sideeffect "", "=v,0"(float %674) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %675 = insertelement <4 x float> %671, float %673, i64 3, !dbg !26
  %math.result268 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %675) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02075 = extractelement <4 x float> %math.result268, i32 0, !dbg !26
  %.cvtpack.f12076 = extractelement <4 x float> %math.result268, i32 1, !dbg !26
  %.cvtpack.f22077 = extractelement <4 x float> %math.result268, i32 2, !dbg !26
  %.cvtpack.f32078 = extractelement <4 x float> %math.result268, i32 3, !dbg !26
  %.cvtpack.p02079 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02075, float %.cvtpack.f12076), !dbg !26
  %.cvtpack.p12080 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22077, float %.cvtpack.f32078), !dbg !26
  %.cvtpack.v02081 = insertelement <2 x i32> poison, i32 %.cvtpack.p02079, i32 0, !dbg !26
  %.cvtpack.v12082 = insertelement <2 x i32> %.cvtpack.v02081, i32 %.cvtpack.p12080, i32 1, !dbg !26
  %676 = bitcast <2 x i32> %.cvtpack.v12082 to <4 x half>, !dbg !26
  %math.packed269 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %676) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %639, <4 x float> %637) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %678 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %658, <4 x float> %638) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %679 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %680 = extractelement <4 x float> %lazy.score2010, i64 0, !dbg !24
  %681 = call float @llvm.amdgcn.exp2.f32(float %680), !dbg !24
  %682 = fadd float %k0.sum.anchor1845, %681, !dbg !25
  %k0.sum.anchor1853 = call float asm sideeffect "", "=v,0"(float %682) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %683 = insertelement <4 x float> poison, float %681, i64 0, !dbg !26
  %684 = extractelement <4 x float> %lazy.score2010, i64 1, !dbg !24
  %685 = call float @llvm.amdgcn.exp2.f32(float %684), !dbg !24
  %686 = fadd float %k0.sum.anchor1846, %685, !dbg !25
  %k0.sum.anchor1854 = call float asm sideeffect "", "=v,0"(float %686) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %687 = insertelement <4 x float> %683, float %685, i64 1, !dbg !26
  %688 = extractelement <4 x float> %lazy.score2010, i64 2, !dbg !24
  %689 = call float @llvm.amdgcn.exp2.f32(float %688), !dbg !24
  %690 = fadd float %k0.sum.anchor1847, %689, !dbg !25
  %k0.sum.anchor1855 = call float asm sideeffect "", "=v,0"(float %690) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %691 = insertelement <4 x float> %687, float %689, i64 2, !dbg !26
  %692 = extractelement <4 x float> %lazy.score2010, i64 3, !dbg !24
  %693 = call float @llvm.amdgcn.exp2.f32(float %692), !dbg !24
  %694 = fadd float %k0.sum.anchor1848, %693, !dbg !25
  %k0.sum.anchor1856 = call float asm sideeffect "", "=v,0"(float %694) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %695 = insertelement <4 x float> %691, float %693, i64 3, !dbg !26
  %math.result254 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %695) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02083 = extractelement <4 x float> %math.result254, i32 0, !dbg !26
  %.cvtpack.f12084 = extractelement <4 x float> %math.result254, i32 1, !dbg !26
  %.cvtpack.f22085 = extractelement <4 x float> %math.result254, i32 2, !dbg !26
  %.cvtpack.f32086 = extractelement <4 x float> %math.result254, i32 3, !dbg !26
  %.cvtpack.p02087 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02083, float %.cvtpack.f12084), !dbg !26
  %.cvtpack.p12088 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22085, float %.cvtpack.f32086), !dbg !26
  %.cvtpack.v02089 = insertelement <2 x i32> poison, i32 %.cvtpack.p02087, i32 0, !dbg !26
  %.cvtpack.v12090 = insertelement <2 x i32> %.cvtpack.v02089, i32 %.cvtpack.p12088, i32 1, !dbg !26
  %696 = bitcast <2 x i32> %.cvtpack.v12090 to <4 x half>, !dbg !26
  %math.packed255 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %696) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %697 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %679, <4 x float> %657) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %698 = shufflevector <8 x half> %521, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %699 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %698, <4 x float> %659) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %700 = extractelement <4 x float> %lazy.score2018, i64 0, !dbg !24
  %701 = call float @llvm.amdgcn.exp2.f32(float %700), !dbg !24
  %702 = fadd float %k0.sum.anchor1849, %701, !dbg !25
  %k0.sum.anchor1857 = call float asm sideeffect "", "=v,0"(float %702) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %703 = insertelement <4 x float> poison, float %701, i64 0, !dbg !26
  %704 = extractelement <4 x float> %lazy.score2018, i64 1, !dbg !24
  %705 = call float @llvm.amdgcn.exp2.f32(float %704), !dbg !24
  %706 = fadd float %k0.sum.anchor1850, %705, !dbg !25
  %k0.sum.anchor1858 = call float asm sideeffect "", "=v,0"(float %706) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %707 = insertelement <4 x float> %703, float %705, i64 1, !dbg !26
  %708 = extractelement <4 x float> %lazy.score2018, i64 2, !dbg !24
  %709 = call float @llvm.amdgcn.exp2.f32(float %708), !dbg !24
  %710 = fadd float %k0.sum.anchor1851, %709, !dbg !25
  %k0.sum.anchor1859 = call float asm sideeffect "", "=v,0"(float %710) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %711 = insertelement <4 x float> %707, float %709, i64 2, !dbg !26
  %712 = extractelement <4 x float> %lazy.score2018, i64 3, !dbg !24
  %713 = call float @llvm.amdgcn.exp2.f32(float %712), !dbg !24
  %714 = fadd float %k0.sum.anchor1852, %713, !dbg !25
  %k0.sum.anchor1860 = call float asm sideeffect "", "=v,0"(float %714) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %715 = insertelement <4 x float> %711, float %713, i64 3, !dbg !26
  %math.result270 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %715) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02091 = extractelement <4 x float> %math.result270, i32 0, !dbg !26
  %.cvtpack.f12092 = extractelement <4 x float> %math.result270, i32 1, !dbg !26
  %.cvtpack.f22093 = extractelement <4 x float> %math.result270, i32 2, !dbg !26
  %.cvtpack.f32094 = extractelement <4 x float> %math.result270, i32 3, !dbg !26
  %.cvtpack.p02095 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02091, float %.cvtpack.f12092), !dbg !26
  %.cvtpack.p12096 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22093, float %.cvtpack.f32094), !dbg !26
  %.cvtpack.v02097 = insertelement <2 x i32> poison, i32 %.cvtpack.p02095, i32 0, !dbg !26
  %.cvtpack.v12098 = insertelement <2 x i32> %.cvtpack.v02097, i32 %.cvtpack.p12096, i32 1, !dbg !26
  %716 = bitcast <2 x i32> %.cvtpack.v12098 to <4 x half>, !dbg !26
  %math.packed271 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %716) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %717 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %679, <4 x float> %677) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %718 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %698, <4 x float> %678) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %719 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %720 = extractelement <4 x float> %lazy.score2011, i64 0, !dbg !24
  %721 = call float @llvm.amdgcn.exp2.f32(float %720), !dbg !24
  %722 = fadd float %k0.sum.anchor1853, %721, !dbg !25
  %k0.sum.anchor1861 = call float asm sideeffect "", "=v,0"(float %722) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %723 = insertelement <4 x float> poison, float %721, i64 0, !dbg !26
  %724 = extractelement <4 x float> %lazy.score2011, i64 1, !dbg !24
  %725 = call float @llvm.amdgcn.exp2.f32(float %724), !dbg !24
  %726 = fadd float %k0.sum.anchor1854, %725, !dbg !25
  %k0.sum.anchor1862 = call float asm sideeffect "", "=v,0"(float %726) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %727 = insertelement <4 x float> %723, float %725, i64 1, !dbg !26
  %728 = extractelement <4 x float> %lazy.score2011, i64 2, !dbg !24
  %729 = call float @llvm.amdgcn.exp2.f32(float %728), !dbg !24
  %730 = fadd float %k0.sum.anchor1855, %729, !dbg !25
  %k0.sum.anchor1863 = call float asm sideeffect "", "=v,0"(float %730) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %731 = insertelement <4 x float> %727, float %729, i64 2, !dbg !26
  %732 = extractelement <4 x float> %lazy.score2011, i64 3, !dbg !24
  %733 = call float @llvm.amdgcn.exp2.f32(float %732), !dbg !24
  %734 = fadd float %k0.sum.anchor1856, %733, !dbg !25
  %k0.sum.anchor1864 = call float asm sideeffect "", "=v,0"(float %734) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %735 = insertelement <4 x float> %731, float %733, i64 3, !dbg !26
  %math.result256 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %735) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02099 = extractelement <4 x float> %math.result256, i32 0, !dbg !26
  %.cvtpack.f12100 = extractelement <4 x float> %math.result256, i32 1, !dbg !26
  %.cvtpack.f22101 = extractelement <4 x float> %math.result256, i32 2, !dbg !26
  %.cvtpack.f32102 = extractelement <4 x float> %math.result256, i32 3, !dbg !26
  %.cvtpack.p02103 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02099, float %.cvtpack.f12100), !dbg !26
  %.cvtpack.p12104 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22101, float %.cvtpack.f32102), !dbg !26
  %.cvtpack.v02105 = insertelement <2 x i32> poison, i32 %.cvtpack.p02103, i32 0, !dbg !26
  %.cvtpack.v12106 = insertelement <2 x i32> %.cvtpack.v02105, i32 %.cvtpack.p12104, i32 1, !dbg !26
  %736 = bitcast <2 x i32> %.cvtpack.v12106 to <4 x half>, !dbg !26
  %math.packed257 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %736) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %737 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %719, <4 x float> %697) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %738 = shufflevector <8 x half> %523, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %739 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %738, <4 x float> %699) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %740 = extractelement <4 x float> %lazy.score2019, i64 0, !dbg !24
  %741 = call float @llvm.amdgcn.exp2.f32(float %740), !dbg !24
  %742 = fadd float %k0.sum.anchor1857, %741, !dbg !25
  %k0.sum.anchor1865 = call float asm sideeffect "", "=v,0"(float %742) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %743 = insertelement <4 x float> poison, float %741, i64 0, !dbg !26
  %744 = extractelement <4 x float> %lazy.score2019, i64 1, !dbg !24
  %745 = call float @llvm.amdgcn.exp2.f32(float %744), !dbg !24
  %746 = fadd float %k0.sum.anchor1858, %745, !dbg !25
  %k0.sum.anchor1866 = call float asm sideeffect "", "=v,0"(float %746) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %747 = insertelement <4 x float> %743, float %745, i64 1, !dbg !26
  %748 = extractelement <4 x float> %lazy.score2019, i64 2, !dbg !24
  %749 = call float @llvm.amdgcn.exp2.f32(float %748), !dbg !24
  %750 = fadd float %k0.sum.anchor1859, %749, !dbg !25
  %k0.sum.anchor1867 = call float asm sideeffect "", "=v,0"(float %750) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %751 = insertelement <4 x float> %747, float %749, i64 2, !dbg !26
  %752 = extractelement <4 x float> %lazy.score2019, i64 3, !dbg !24
  %753 = call float @llvm.amdgcn.exp2.f32(float %752), !dbg !24
  %754 = fadd float %k0.sum.anchor1860, %753, !dbg !25
  %k0.sum.anchor1868 = call float asm sideeffect "", "=v,0"(float %754) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %755 = insertelement <4 x float> %751, float %753, i64 3, !dbg !26
  %math.result272 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %755) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02107 = extractelement <4 x float> %math.result272, i32 0, !dbg !26
  %.cvtpack.f12108 = extractelement <4 x float> %math.result272, i32 1, !dbg !26
  %.cvtpack.f22109 = extractelement <4 x float> %math.result272, i32 2, !dbg !26
  %.cvtpack.f32110 = extractelement <4 x float> %math.result272, i32 3, !dbg !26
  %.cvtpack.p02111 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02107, float %.cvtpack.f12108), !dbg !26
  %.cvtpack.p12112 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22109, float %.cvtpack.f32110), !dbg !26
  %.cvtpack.v02113 = insertelement <2 x i32> poison, i32 %.cvtpack.p02111, i32 0, !dbg !26
  %.cvtpack.v12114 = insertelement <2 x i32> %.cvtpack.v02113, i32 %.cvtpack.p12112, i32 1, !dbg !26
  %756 = bitcast <2 x i32> %.cvtpack.v12114 to <4 x half>, !dbg !26
  %math.packed273 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %756) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %757 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %719, <4 x float> %717) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %758 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %738, <4 x float> %718) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %759 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %760 = extractelement <4 x float> %lazy.score2012, i64 0, !dbg !24
  %761 = call float @llvm.amdgcn.exp2.f32(float %760), !dbg !24
  %762 = fadd float %k0.sum.anchor1861, %761, !dbg !25
  %k0.sum.anchor1869 = call float asm sideeffect "", "=v,0"(float %762) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %763 = insertelement <4 x float> poison, float %761, i64 0, !dbg !26
  %764 = extractelement <4 x float> %lazy.score2012, i64 1, !dbg !24
  %765 = call float @llvm.amdgcn.exp2.f32(float %764), !dbg !24
  %766 = fadd float %k0.sum.anchor1862, %765, !dbg !25
  %k0.sum.anchor1870 = call float asm sideeffect "", "=v,0"(float %766) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %767 = insertelement <4 x float> %763, float %765, i64 1, !dbg !26
  %768 = extractelement <4 x float> %lazy.score2012, i64 2, !dbg !24
  %769 = call float @llvm.amdgcn.exp2.f32(float %768), !dbg !24
  %770 = fadd float %k0.sum.anchor1863, %769, !dbg !25
  %k0.sum.anchor1871 = call float asm sideeffect "", "=v,0"(float %770) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %771 = insertelement <4 x float> %767, float %769, i64 2, !dbg !26
  %772 = extractelement <4 x float> %lazy.score2012, i64 3, !dbg !24
  %773 = call float @llvm.amdgcn.exp2.f32(float %772), !dbg !24
  %774 = fadd float %k0.sum.anchor1864, %773, !dbg !25
  %k0.sum.anchor1872 = call float asm sideeffect "", "=v,0"(float %774) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %775 = insertelement <4 x float> %771, float %773, i64 3, !dbg !26
  %math.result258 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02115 = extractelement <4 x float> %math.result258, i32 0, !dbg !26
  %.cvtpack.f12116 = extractelement <4 x float> %math.result258, i32 1, !dbg !26
  %.cvtpack.f22117 = extractelement <4 x float> %math.result258, i32 2, !dbg !26
  %.cvtpack.f32118 = extractelement <4 x float> %math.result258, i32 3, !dbg !26
  %.cvtpack.p02119 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02115, float %.cvtpack.f12116), !dbg !26
  %.cvtpack.p12120 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22117, float %.cvtpack.f32118), !dbg !26
  %.cvtpack.v02121 = insertelement <2 x i32> poison, i32 %.cvtpack.p02119, i32 0, !dbg !26
  %.cvtpack.v12122 = insertelement <2 x i32> %.cvtpack.v02121, i32 %.cvtpack.p12120, i32 1, !dbg !26
  %776 = bitcast <2 x i32> %.cvtpack.v12122 to <4 x half>, !dbg !26
  %math.packed259 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %777 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %759, <4 x float> %737) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %778 = shufflevector <8 x half> %525, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %779 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %778, <4 x float> %739) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %780 = extractelement <4 x float> %lazy.score2020, i64 0, !dbg !24
  %781 = call float @llvm.amdgcn.exp2.f32(float %780), !dbg !24
  %782 = fadd float %k0.sum.anchor1865, %781, !dbg !25
  %k0.sum.anchor1873 = call float asm sideeffect "", "=v,0"(float %782) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %783 = insertelement <4 x float> poison, float %781, i64 0, !dbg !26
  %784 = extractelement <4 x float> %lazy.score2020, i64 1, !dbg !24
  %785 = call float @llvm.amdgcn.exp2.f32(float %784), !dbg !24
  %786 = fadd float %k0.sum.anchor1866, %785, !dbg !25
  %k0.sum.anchor1874 = call float asm sideeffect "", "=v,0"(float %786) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %787 = insertelement <4 x float> %783, float %785, i64 1, !dbg !26
  %788 = extractelement <4 x float> %lazy.score2020, i64 2, !dbg !24
  %789 = call float @llvm.amdgcn.exp2.f32(float %788), !dbg !24
  %790 = fadd float %k0.sum.anchor1867, %789, !dbg !25
  %k0.sum.anchor1875 = call float asm sideeffect "", "=v,0"(float %790) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %791 = insertelement <4 x float> %787, float %789, i64 2, !dbg !26
  %792 = extractelement <4 x float> %lazy.score2020, i64 3, !dbg !24
  %793 = call float @llvm.amdgcn.exp2.f32(float %792), !dbg !24
  %794 = fadd float %k0.sum.anchor1868, %793, !dbg !25
  %k0.sum.anchor1876 = call float asm sideeffect "", "=v,0"(float %794) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %795 = insertelement <4 x float> %791, float %793, i64 3, !dbg !26
  %math.result274 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %795) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02123 = extractelement <4 x float> %math.result274, i32 0, !dbg !26
  %.cvtpack.f12124 = extractelement <4 x float> %math.result274, i32 1, !dbg !26
  %.cvtpack.f22125 = extractelement <4 x float> %math.result274, i32 2, !dbg !26
  %.cvtpack.f32126 = extractelement <4 x float> %math.result274, i32 3, !dbg !26
  %.cvtpack.p02127 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02123, float %.cvtpack.f12124), !dbg !26
  %.cvtpack.p12128 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22125, float %.cvtpack.f32126), !dbg !26
  %.cvtpack.v02129 = insertelement <2 x i32> poison, i32 %.cvtpack.p02127, i32 0, !dbg !26
  %.cvtpack.v12130 = insertelement <2 x i32> %.cvtpack.v02129, i32 %.cvtpack.p12128, i32 1, !dbg !26
  %796 = bitcast <2 x i32> %.cvtpack.v12130 to <4 x half>, !dbg !26
  %math.packed275 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %796) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %797 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %759, <4 x float> %757) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %798 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %778, <4 x float> %758) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %799 = bitcast <4 x i32> %526 to <8 x half>, !dbg !14
  %800 = shufflevector <8 x half> %799, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %801 = extractelement <4 x float> %lazy.score2013, i64 0, !dbg !24
  %802 = call float @llvm.amdgcn.exp2.f32(float %801), !dbg !24
  %803 = fadd float %k0.sum.anchor1869, %802, !dbg !25
  %k0.sum.anchor1877 = call float asm sideeffect "", "=v,0"(float %803) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %804 = insertelement <4 x float> poison, float %802, i64 0, !dbg !26
  %805 = extractelement <4 x float> %lazy.score2013, i64 1, !dbg !24
  %806 = call float @llvm.amdgcn.exp2.f32(float %805), !dbg !24
  %807 = fadd float %k0.sum.anchor1870, %806, !dbg !25
  %k0.sum.anchor1878 = call float asm sideeffect "", "=v,0"(float %807) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %808 = fadd float %k0.sum.anchor1877, %k0.sum.anchor1878, !dbg !25
  %k0.sum.anchor1879 = call float asm sideeffect "", "=v,0"(float %808) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %809 = insertelement <4 x float> %804, float %806, i64 1, !dbg !26
  %810 = extractelement <4 x float> %lazy.score2013, i64 2, !dbg !24
  %811 = call float @llvm.amdgcn.exp2.f32(float %810), !dbg !24
  %812 = fadd float %k0.sum.anchor1871, %811, !dbg !25
  %k0.sum.anchor1880 = call float asm sideeffect "", "=v,0"(float %812) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %813 = insertelement <4 x float> %809, float %811, i64 2, !dbg !26
  %814 = extractelement <4 x float> %lazy.score2013, i64 3, !dbg !24
  %815 = call float @llvm.amdgcn.exp2.f32(float %814), !dbg !24
  %816 = fadd float %k0.sum.anchor1872, %815, !dbg !25
  %k0.sum.anchor1881 = call float asm sideeffect "", "=v,0"(float %816) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %817 = fadd float %k0.sum.anchor1880, %k0.sum.anchor1881, !dbg !25
  %k0.sum.anchor1882 = call float asm sideeffect "", "=v,0"(float %817) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %818 = fadd float %k0.sum.anchor1879, %k0.sum.anchor1882, !dbg !25
  %k0.sum.anchor1883 = call float asm sideeffect "", "=v,0"(float %818) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %819 = fadd float %k0.sum.anchor1883, %500, !dbg !25
  %k0.sum.anchor1884 = call float asm sideeffect "", "=v,0"(float %819) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %820 = insertelement <1 x float> poison, float %k0.sum.anchor1884, i64 0, !dbg !6
  %821 = insertvalue [2 x <1 x float>] poison, <1 x float> %820, 0, !dbg !6
  %822 = insertelement <4 x float> %813, float %815, i64 3, !dbg !26
  %math.result260 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %822) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02131 = extractelement <4 x float> %math.result260, i32 0, !dbg !26
  %.cvtpack.f12132 = extractelement <4 x float> %math.result260, i32 1, !dbg !26
  %.cvtpack.f22133 = extractelement <4 x float> %math.result260, i32 2, !dbg !26
  %.cvtpack.f32134 = extractelement <4 x float> %math.result260, i32 3, !dbg !26
  %.cvtpack.p02135 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02131, float %.cvtpack.f12132), !dbg !26
  %.cvtpack.p12136 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22133, float %.cvtpack.f32134), !dbg !26
  %.cvtpack.v02137 = insertelement <2 x i32> poison, i32 %.cvtpack.p02135, i32 0, !dbg !26
  %.cvtpack.v12138 = insertelement <2 x i32> %.cvtpack.v02137, i32 %.cvtpack.p12136, i32 1, !dbg !26
  %823 = bitcast <2 x i32> %.cvtpack.v12138 to <4 x half>, !dbg !26
  %math.packed261 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %823) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %824 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %800, <4 x float> %777) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %825 = shufflevector <8 x half> %799, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %826 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %825, <4 x float> %779) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %827 = extractelement <4 x float> %lazy.score2021, i64 0, !dbg !24
  %828 = call float @llvm.amdgcn.exp2.f32(float %827), !dbg !24
  %829 = fadd float %k0.sum.anchor1873, %828, !dbg !25
  %k0.sum.anchor1885 = call float asm sideeffect "", "=v,0"(float %829) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %830 = insertelement <4 x float> poison, float %828, i64 0, !dbg !26
  %831 = extractelement <4 x float> %lazy.score2021, i64 1, !dbg !24
  %832 = call float @llvm.amdgcn.exp2.f32(float %831), !dbg !24
  %833 = fadd float %k0.sum.anchor1874, %832, !dbg !25
  %k0.sum.anchor1886 = call float asm sideeffect "", "=v,0"(float %833) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %834 = fadd float %k0.sum.anchor1885, %k0.sum.anchor1886, !dbg !25
  %k0.sum.anchor1887 = call float asm sideeffect "", "=v,0"(float %834) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %835 = insertelement <4 x float> %830, float %832, i64 1, !dbg !26
  %836 = extractelement <4 x float> %lazy.score2021, i64 2, !dbg !24
  %837 = call float @llvm.amdgcn.exp2.f32(float %836), !dbg !24
  %838 = fadd float %k0.sum.anchor1875, %837, !dbg !25
  %k0.sum.anchor1888 = call float asm sideeffect "", "=v,0"(float %838) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %839 = insertelement <4 x float> %835, float %837, i64 2, !dbg !26
  %840 = extractelement <4 x float> %lazy.score2021, i64 3, !dbg !24
  %841 = call float @llvm.amdgcn.exp2.f32(float %840), !dbg !24
  %842 = fadd float %k0.sum.anchor1876, %841, !dbg !25
  %k0.sum.anchor1889 = call float asm sideeffect "", "=v,0"(float %842) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %843 = fadd float %k0.sum.anchor1888, %k0.sum.anchor1889, !dbg !25
  %k0.sum.anchor1890 = call float asm sideeffect "", "=v,0"(float %843) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %844 = fadd float %k0.sum.anchor1887, %k0.sum.anchor1890, !dbg !25
  %k0.sum.anchor1891 = call float asm sideeffect "", "=v,0"(float %844) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %845 = fadd float %k0.sum.anchor1891, %502, !dbg !25
  %k0.sum.anchor1892 = call float asm sideeffect "", "=v,0"(float %845) #9, !dbg !25, !frisk.cross_bn_sum_anchor !1
  %846 = insertelement <1 x float> poison, float %k0.sum.anchor1892, i64 0, !dbg !6
  %847 = insertvalue [2 x <1 x float>] %821, <1 x float> %846, 1, !dbg !6
  %848 = insertelement <4 x float> %839, float %841, i64 3, !dbg !26
  %math.result276 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %848) #9, !dbg !26, !frisk.register_lifetimes !1
  %.cvtpack.f02139 = extractelement <4 x float> %math.result276, i32 0, !dbg !26
  %.cvtpack.f12140 = extractelement <4 x float> %math.result276, i32 1, !dbg !26
  %.cvtpack.f22141 = extractelement <4 x float> %math.result276, i32 2, !dbg !26
  %.cvtpack.f32142 = extractelement <4 x float> %math.result276, i32 3, !dbg !26
  %.cvtpack.p02143 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02139, float %.cvtpack.f12140), !dbg !26
  %.cvtpack.p12144 = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22141, float %.cvtpack.f32142), !dbg !26
  %.cvtpack.v02145 = insertelement <2 x i32> poison, i32 %.cvtpack.p02143, i32 0, !dbg !26
  %.cvtpack.v12146 = insertelement <2 x i32> %.cvtpack.v02145, i32 %.cvtpack.p12144, i32 1, !dbg !26
  %849 = bitcast <2 x i32> %.cvtpack.v12146 to <4 x half>, !dbg !26
  %math.packed277 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %849) #9, !dbg !25, !frisk.register_lifetimes !1
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14, !frisk.softmax_fragments !1
  %850 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %800, <4 x float> %797) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %851 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %825, <4 x float> %798) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %852 = extractvalue [2 x <8 x float>] %lazy.state2025, 0, !dbg !27
  %853 = shufflevector <8 x float> %852, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %854 = shufflevector <8 x float> %852, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %855 = extractvalue [2 x <8 x float>] %lazy.state2025, 1, !dbg !27
  %856 = shufflevector <8 x float> %855, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %857 = shufflevector <8 x float> %855, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %.idx13241325 = add nsw i64 %195, %address.local796, !dbg !16
  %.idx13241325.tr = trunc i64 %.idx13241325 to i32, !dbg !16
  %dma.buffer.offset1243 = shl i32 %.idx13241325.tr, 1, !dbg !16
  %858 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %dma.buffer.offset1243, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %859 = trunc nuw i64 %195 to i32, !dbg !16
  %.idx13261327.tr = or disjoint i32 %address.local803, %859, !dbg !16
  %dma.buffer.offset1241 = shl nuw nsw i32 %.idx13261327.tr, 1, !dbg !16
  %860 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %dma.buffer.offset1241, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %address.local873 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local813, !dbg !14
  %861 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local873, i32 0) #0, !dbg !14
  %862 = bitcast <4 x i32> %861 to <8 x half>, !dbg !14
  %address.local874 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local821, !dbg !14
  %863 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #0, !dbg !14
  %864 = bitcast <4 x i32> %863 to <8 x half>, !dbg !14
  %address.local875 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local829, !dbg !14
  %865 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #0, !dbg !14
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !14
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local837, !dbg !14
  %867 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local876, i32 0) #0, !dbg !14
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !14
  %address.local877 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local845, !dbg !14
  %869 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local877, i32 0) #0, !dbg !14
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !14
  %address.local878 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local853, !dbg !14
  %871 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #0, !dbg !14
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !14
  %address.local879 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local861, !dbg !14
  %873 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #0, !dbg !14
  %874 = bitcast <4 x i32> %873 to <8 x half>, !dbg !14
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local869, !dbg !14
  %875 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local880, i32 0) #0, !dbg !14
  %876 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %877 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %876, <4 x float> %853) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %878 = shufflevector <8 x half> %862, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %879 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %878, <4 x float> %854) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %880 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %876, <4 x float> %856) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %881 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %878, <4 x float> %857) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %882 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %883 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %882, <4 x float> %877) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %884 = shufflevector <8 x half> %864, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %885 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %884, <4 x float> %879) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %886 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %882, <4 x float> %880) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %887 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %884, <4 x float> %881) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %888 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %889 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %888, <4 x float> %883) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %890 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %891 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %890, <4 x float> %885) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %892 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %888, <4 x float> %886) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %893 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %890, <4 x float> %887) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %894 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %895 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %894, <4 x float> %889) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %896 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %897 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %896, <4 x float> %891) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %898 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %894, <4 x float> %892) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %899 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %896, <4 x float> %893) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %900 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %901 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %900, <4 x float> %895) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %902 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %903 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %902, <4 x float> %897) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %904 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %900, <4 x float> %898) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %905 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %902, <4 x float> %899) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %906 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %907 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %906, <4 x float> %901) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %908 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %909 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %908, <4 x float> %903) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %910 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %906, <4 x float> %904) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %911 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %908, <4 x float> %905) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %912 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %913 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %912, <4 x float> %907) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %914 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %915 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %914, <4 x float> %909) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %916 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %912, <4 x float> %910) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %917 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %914, <4 x float> %911) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %918 = bitcast <4 x i32> %875 to <8 x half>, !dbg !14
  %919 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %920 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %919, <4 x float> %913) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %921 = shufflevector <8 x half> %918, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %922 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %921, <4 x float> %915) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %923 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %919, <4 x float> %916) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %924 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %921, <4 x float> %917) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %925 = extractvalue [2 x <8 x float>] %lazy.state2024, 0, !dbg !27
  %926 = shufflevector <8 x float> %925, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %927 = shufflevector <8 x float> %925, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %928 = extractvalue [2 x <8 x float>] %lazy.state2024, 1, !dbg !27
  %929 = shufflevector <8 x float> %928, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %930 = shufflevector <8 x float> %928, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %.idx13281329 = add nsw i64 %248, %address.local796, !dbg !16
  %.idx13281329.tr = trunc i64 %.idx13281329 to i32, !dbg !16
  %dma.buffer.offset1239 = shl i32 %.idx13281329.tr, 1, !dbg !16
  %931 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %51, i32 %dma.buffer.offset1239, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %.tr1331 = trunc nuw i64 %248 to i32, !dbg !16
  %932 = or disjoint i32 %address.local803, %.tr1331, !dbg !16
  %dma.buffer.offset = shl nuw nsw i32 %932, 1, !dbg !16
  %933 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %55, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1252) #10, !dbg !16
  %934 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #0, !dbg !14
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !14
  %936 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #0, !dbg !14
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !14
  %938 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #0, !dbg !14
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !14
  %940 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local838, i32 0) #0, !dbg !14
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !14
  %942 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local846, i32 0) #0, !dbg !14
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !14
  %944 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local854, i32 0) #0, !dbg !14
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !14
  %946 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local862, i32 0) #0, !dbg !14
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !14
  %948 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local870, i32 0) #0, !dbg !14
  %949 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %950 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %949, <4 x float> %926) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %951 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %952 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %951, <4 x float> %927) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %953 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %949, <4 x float> %929) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %954 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %951, <4 x float> %930) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %955 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %956 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %955, <4 x float> %950) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %957 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %958 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %957, <4 x float> %952) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %959 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %955, <4 x float> %953) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %960 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %957, <4 x float> %954) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %961 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %962 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %961, <4 x float> %956) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %963 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %964 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %963, <4 x float> %958) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %965 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %961, <4 x float> %959) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %966 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %963, <4 x float> %960) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %967 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %968 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %967, <4 x float> %962) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %969 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %970 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %969, <4 x float> %964) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %971 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %967, <4 x float> %965) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %972 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %969, <4 x float> %966) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %973 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %974 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %973, <4 x float> %968) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %975 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %976 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %975, <4 x float> %970) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %977 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %973, <4 x float> %971) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %978 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %975, <4 x float> %972) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %979 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %980 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %979, <4 x float> %974) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %981 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %982 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %981, <4 x float> %976) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %983 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %979, <4 x float> %977) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %984 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %981, <4 x float> %978) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %985 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %986 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %985, <4 x float> %980) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %987 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %988 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %987, <4 x float> %982) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %989 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %985, <4 x float> %983) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %990 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %987, <4 x float> %984) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %991 = bitcast <4 x i32> %948 to <8 x half>, !dbg !14
  %992 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %993 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %992, <4 x float> %986) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %994 = shufflevector <8 x half> %991, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %995 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %994, <4 x float> %988) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %996 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %992, <4 x float> %989) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %997 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %994, <4 x float> %990) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !27
  %998 = extractvalue [2 x <8 x float>] %lazy.state2023, 0, !dbg !27
  %999 = shufflevector <8 x float> %998, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1000 = shufflevector <8 x float> %998, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  %1001 = extractvalue [2 x <8 x float>] %lazy.state2023, 1, !dbg !27
  %1002 = shufflevector <8 x float> %1001, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !27
  %1003 = shufflevector <8 x float> %1001, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !27
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %1004 = add nuw nsw i64 %101, 1, !dbg !6
  %k0.has.next = icmp ult i64 %1004, 32, !dbg !14
  br i1 %k0.has.next, label %k0.issue, label %k0.join, !dbg !14

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %1004, 14, !dbg !13
  %k0.next.1903 = or disjoint i64 %111, %k0.next., !dbg !13
  %k0.next.kdma.source1348 = or disjoint i64 %k0.next.1903, %kdma.source1347, !dbg !13
  %k0.next.kdma.source1349 = trunc nuw i64 %k0.next.kdma.source1348 to i32, !dbg !13
  %k0.next.kdma.source1350 = shl nuw nsw i32 %k0.next.kdma.source1349, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1350, %kdma.intra, !dbg !13
  %k0.next.1904 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %38, i32 %k0.next.kdma.offset, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1368 = or disjoint i64 %k0.next.1903, %kdma.source1367, !dbg !13
  %k0.next.kdma.source1369 = trunc nuw i64 %k0.next.kdma.source1368 to i32, !dbg !13
  %k0.next.kdma.source1370 = shl nuw nsw i32 %k0.next.kdma.source1369, 1, !dbg !13
  %k0.next.kdma.offset1371 = or disjoint i32 %k0.next.kdma.source1370, %kdma.intra.hi, !dbg !13
  %k0.next.1905 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1372, i32 %k0.next.kdma.offset1371, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1392 = or disjoint i64 %k0.next.1903, %kdma.source1391, !dbg !13
  %k0.next.kdma.source1393 = trunc nuw i64 %k0.next.kdma.source1392 to i32, !dbg !13
  %k0.next.kdma.source1394 = shl nuw nsw i32 %k0.next.kdma.source1393, 1, !dbg !13
  %k0.next.kdma.offset1395 = add nuw i32 %k0.next.kdma.source1394, %kdma.intra, !dbg !13
  %k0.next.1906 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %46, i32 %k0.next.kdma.offset1395, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  %k0.next.kdma.source1416 = or disjoint i64 %k0.next.1903, %kdma.source1415, !dbg !13
  %k0.next.kdma.source1417 = trunc nuw i64 %k0.next.kdma.source1416 to i32, !dbg !13
  %k0.next.kdma.source1418 = shl nuw nsw i32 %k0.next.kdma.source1417, 1, !dbg !13
  %k0.next.kdma.offset1419 = add nuw i32 %k0.next.kdma.source1418, %kdma.intra.hi, !dbg !13
  %k0.next.1907 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1420, i32 %k0.next.kdma.offset1419, <4 x i32> %dma.buffer.rsrc1268) #10, !dbg !13, !frisk.cross_bn_k0 !1
  br label %k0.join

k0.join:                                          ; preds = %lazy.join, %k0.issue
  %1005 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local873, i32 0) #0, !dbg !14
  %1006 = bitcast <4 x i32> %1005 to <8 x half>, !dbg !14
  %1007 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #0, !dbg !14
  %1008 = bitcast <4 x i32> %1007 to <8 x half>, !dbg !14
  %1009 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #0, !dbg !14
  %1010 = bitcast <4 x i32> %1009 to <8 x half>, !dbg !14
  %1011 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local876, i32 0) #0, !dbg !14
  %1012 = bitcast <4 x i32> %1011 to <8 x half>, !dbg !14
  %1013 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local877, i32 0) #0, !dbg !14
  %1014 = bitcast <4 x i32> %1013 to <8 x half>, !dbg !14
  %1015 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #0, !dbg !14
  %1016 = bitcast <4 x i32> %1015 to <8 x half>, !dbg !14
  %1017 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #0, !dbg !14
  %1018 = bitcast <4 x i32> %1017 to <8 x half>, !dbg !14
  %1019 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local880, i32 0) #0, !dbg !14
  %1020 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1021 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1020, <4 x float> %999) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1022 = shufflevector <8 x half> %1006, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1023 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1022, <4 x float> %1000) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1024 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1020, <4 x float> %1002) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1025 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %1022, <4 x float> %1003) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1026 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1027 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1026, <4 x float> %1021) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1028 = shufflevector <8 x half> %1008, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1029 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %1028, <4 x float> %1023) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1030 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %1026, <4 x float> %1024) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1031 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %1028, <4 x float> %1025) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1032 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1033 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1032, <4 x float> %1027) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1034 = shufflevector <8 x half> %1010, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1035 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %1034, <4 x float> %1029) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1036 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %1032, <4 x float> %1030) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1037 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %1034, <4 x float> %1031) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1038 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1039 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1038, <4 x float> %1033) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1040 = shufflevector <8 x half> %1012, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1041 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %1040, <4 x float> %1035) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1042 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %1038, <4 x float> %1036) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1043 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %1040, <4 x float> %1037) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1044 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1045 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1044, <4 x float> %1039) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1046 = shufflevector <8 x half> %1014, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1047 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %1046, <4 x float> %1041) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1048 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %1044, <4 x float> %1042) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1049 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %1046, <4 x float> %1043) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1050 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1051 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1050, <4 x float> %1045) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1052 = shufflevector <8 x half> %1016, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1053 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %1052, <4 x float> %1047) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1054 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %1050, <4 x float> %1048) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1055 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %1052, <4 x float> %1049) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1056 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1057 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1056, <4 x float> %1051) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1058 = shufflevector <8 x half> %1018, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1059 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1058, <4 x float> %1053) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1060 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %1056, <4 x float> %1054) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1061 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %1058, <4 x float> %1055) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1062 = bitcast <4 x i32> %1019 to <8 x half>, !dbg !14
  %1063 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1064 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1063, <4 x float> %1057) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1065 = shufflevector <8 x half> %1062, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1066 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1065, <4 x float> %1059) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1067 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1063, <4 x float> %1060) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !14
  %1068 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1065, <4 x float> %1061) #9, !dbg !14, !frisk.pv_interleave !31
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !6
  %1069 = shufflevector <4 x float> %824, <4 x float> %826, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1070 = insertvalue [2 x <8 x float>] poison, <8 x float> %1069, 0, !dbg !6
  %1071 = shufflevector <4 x float> %850, <4 x float> %851, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1072 = insertvalue [2 x <8 x float>] %1070, <8 x float> %1071, 1, !dbg !6
  %1073 = shufflevector <4 x float> %920, <4 x float> %922, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1074 = insertvalue [2 x <8 x float>] poison, <8 x float> %1073, 0, !dbg !6
  %1075 = shufflevector <4 x float> %923, <4 x float> %924, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1076 = insertvalue [2 x <8 x float>] %1074, <8 x float> %1075, 1, !dbg !6
  %1077 = shufflevector <4 x float> %993, <4 x float> %995, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1078 = insertvalue [2 x <8 x float>] poison, <8 x float> %1077, 0, !dbg !6
  %1079 = shufflevector <4 x float> %996, <4 x float> %997, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1080 = insertvalue [2 x <8 x float>] %1078, <8 x float> %1079, 1, !dbg !6
  %1081 = shufflevector <4 x float> %1064, <4 x float> %1066, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1082 = insertvalue [2 x <8 x float>] poison, <8 x float> %1081, 0, !dbg !6
  %1083 = shufflevector <4 x float> %1067, <4 x float> %1068, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1084 = insertvalue [2 x <8 x float>] %1082, <8 x float> %1083, 1, !dbg !6
  br label %100, !dbg !6

1085:                                             ; preds = %100
  %1086 = extractvalue [2 x <1 x float>] %103, 1, !dbg !6
  %1087 = extractelement <1 x float> %1086, i64 0, !dbg !6
  %1088 = bitcast float %1087 to i32, !dbg !6
  %1089 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1088), !dbg !6
  %1090 = bitcast i32 %1089 to float, !dbg !6
  %1091 = fadd float %1087, %1090, !dbg !6
  %1092 = bitcast float %1091 to i32, !dbg !6
  %1093 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1092), !dbg !6
  %1094 = bitcast i32 %1093 to float, !dbg !6
  %1095 = fadd float %1091, %1094, !dbg !6
  %1096 = extractvalue [2 x <1 x float>] %103, 0, !dbg !6
  %1097 = extractelement <1 x float> %1096, i64 0, !dbg !6
  %1098 = bitcast float %1097 to i32, !dbg !6
  %1099 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1098), !dbg !6
  %1100 = bitcast i32 %1099 to float, !dbg !6
  %1101 = fadd float %1097, %1100, !dbg !6
  %1102 = bitcast float %1101 to i32, !dbg !6
  %1103 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1102), !dbg !6
  %1104 = bitcast i32 %1103 to float, !dbg !6
  %1105 = fadd float %1101, %1104, !dbg !6
  %1106 = and i32 %kdma.tid, 16, !dbg !6
  %.not = icmp eq i32 %1106, 0, !dbg !6
  %1107 = and i32 %kdma.tid, 32, !dbg !6
  %.not40 = icmp eq i32 %1107, 0, !dbg !6
  %1108 = extractvalue [2 x <8 x float>] %107, 0, !dbg !6
  %1109 = extractelement <8 x float> %1108, i64 0, !dbg !6
  %1110 = extractelement <8 x float> %1108, i64 1, !dbg !6
  %1111 = extractelement <8 x float> %1108, i64 2, !dbg !6
  %1112 = extractelement <8 x float> %1108, i64 3, !dbg !6
  %1113 = select i1 %.not, float %1110, float %1109, !dbg !6
  %1114 = bitcast float %1113 to i32, !dbg !6
  %1115 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1114), !dbg !6
  %1116 = bitcast i32 %1115 to float, !dbg !6
  %1117 = select i1 %.not, float %1109, float %1116, !dbg !6
  %1118 = select i1 %.not, float %1116, float %1110, !dbg !6
  %1119 = select i1 %.not, float %1112, float %1111, !dbg !6
  %1120 = bitcast float %1119 to i32, !dbg !6
  %1121 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1120), !dbg !6
  %1122 = bitcast i32 %1121 to float, !dbg !6
  %1123 = select i1 %.not, float %1111, float %1122, !dbg !6
  %1124 = select i1 %.not, float %1122, float %1112, !dbg !6
  %1125 = select i1 %.not40, float %1123, float %1117, !dbg !6
  %1126 = bitcast float %1125 to i32, !dbg !6
  %1127 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1126), !dbg !6
  %1128 = bitcast i32 %1127 to float, !dbg !6
  %1129 = select i1 %.not40, float %1117, float %1128, !dbg !6
  %1130 = select i1 %.not40, float %1128, float %1123, !dbg !6
  %1131 = select i1 %.not40, float %1124, float %1118, !dbg !6
  %1132 = bitcast float %1131 to i32, !dbg !6
  %1133 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1132), !dbg !6
  %1134 = bitcast i32 %1133 to float, !dbg !6
  %1135 = select i1 %.not40, float %1118, float %1134, !dbg !6
  %1136 = select i1 %.not40, float %1134, float %1124, !dbg !6
  %1137 = extractelement <8 x float> %1108, i64 4, !dbg !6
  %1138 = extractelement <8 x float> %1108, i64 5, !dbg !6
  %1139 = extractelement <8 x float> %1108, i64 6, !dbg !6
  %1140 = extractelement <8 x float> %1108, i64 7, !dbg !6
  %1141 = select i1 %.not, float %1138, float %1137, !dbg !6
  %1142 = bitcast float %1141 to i32, !dbg !6
  %1143 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1142), !dbg !6
  %1144 = bitcast i32 %1143 to float, !dbg !6
  %1145 = select i1 %.not, float %1137, float %1144, !dbg !6
  %1146 = select i1 %.not, float %1144, float %1138, !dbg !6
  %1147 = select i1 %.not, float %1140, float %1139, !dbg !6
  %1148 = bitcast float %1147 to i32, !dbg !6
  %1149 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1148), !dbg !6
  %1150 = bitcast i32 %1149 to float, !dbg !6
  %1151 = select i1 %.not, float %1139, float %1150, !dbg !6
  %1152 = select i1 %.not, float %1150, float %1140, !dbg !6
  %1153 = select i1 %.not40, float %1151, float %1145, !dbg !6
  %1154 = bitcast float %1153 to i32, !dbg !6
  %1155 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1154), !dbg !6
  %1156 = bitcast i32 %1155 to float, !dbg !6
  %1157 = select i1 %.not40, float %1145, float %1156, !dbg !6
  %1158 = select i1 %.not40, float %1156, float %1151, !dbg !6
  %1159 = select i1 %.not40, float %1152, float %1146, !dbg !6
  %1160 = bitcast float %1159 to i32, !dbg !6
  %1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1160), !dbg !6
  %1162 = bitcast i32 %1161 to float, !dbg !6
  %1163 = select i1 %.not40, float %1146, float %1162, !dbg !6
  %1164 = select i1 %.not40, float %1162, float %1152, !dbg !6
  %1165 = extractvalue [2 x <8 x float>] %107, 1, !dbg !6
  %1166 = extractelement <8 x float> %1165, i64 0, !dbg !6
  %1167 = extractelement <8 x float> %1165, i64 1, !dbg !6
  %1168 = extractelement <8 x float> %1165, i64 2, !dbg !6
  %1169 = extractelement <8 x float> %1165, i64 3, !dbg !6
  %1170 = select i1 %.not, float %1167, float %1166, !dbg !6
  %1171 = bitcast float %1170 to i32, !dbg !6
  %1172 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1171), !dbg !6
  %1173 = bitcast i32 %1172 to float, !dbg !6
  %1174 = select i1 %.not, float %1166, float %1173, !dbg !6
  %1175 = select i1 %.not, float %1173, float %1167, !dbg !6
  %1176 = select i1 %.not, float %1169, float %1168, !dbg !6
  %1177 = bitcast float %1176 to i32, !dbg !6
  %1178 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1177), !dbg !6
  %1179 = bitcast i32 %1178 to float, !dbg !6
  %1180 = select i1 %.not, float %1168, float %1179, !dbg !6
  %1181 = select i1 %.not, float %1179, float %1169, !dbg !6
  %1182 = select i1 %.not40, float %1180, float %1174, !dbg !6
  %1183 = bitcast float %1182 to i32, !dbg !6
  %1184 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1183), !dbg !6
  %1185 = bitcast i32 %1184 to float, !dbg !6
  %1186 = select i1 %.not40, float %1174, float %1185, !dbg !6
  %1187 = select i1 %.not40, float %1185, float %1180, !dbg !6
  %1188 = select i1 %.not40, float %1181, float %1175, !dbg !6
  %1189 = bitcast float %1188 to i32, !dbg !6
  %1190 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1189), !dbg !6
  %1191 = bitcast i32 %1190 to float, !dbg !6
  %1192 = select i1 %.not40, float %1175, float %1191, !dbg !6
  %1193 = select i1 %.not40, float %1191, float %1181, !dbg !6
  %1194 = extractelement <8 x float> %1165, i64 4, !dbg !6
  %1195 = extractelement <8 x float> %1165, i64 5, !dbg !6
  %1196 = extractelement <8 x float> %1165, i64 6, !dbg !6
  %1197 = extractelement <8 x float> %1165, i64 7, !dbg !6
  %1198 = select i1 %.not, float %1195, float %1194, !dbg !6
  %1199 = bitcast float %1198 to i32, !dbg !6
  %1200 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1199), !dbg !6
  %1201 = bitcast i32 %1200 to float, !dbg !6
  %1202 = select i1 %.not, float %1194, float %1201, !dbg !6
  %1203 = select i1 %.not, float %1201, float %1195, !dbg !6
  %1204 = select i1 %.not, float %1197, float %1196, !dbg !6
  %1205 = bitcast float %1204 to i32, !dbg !6
  %1206 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1205), !dbg !6
  %1207 = bitcast i32 %1206 to float, !dbg !6
  %1208 = select i1 %.not, float %1196, float %1207, !dbg !6
  %1209 = select i1 %.not, float %1207, float %1197, !dbg !6
  %1210 = select i1 %.not40, float %1208, float %1202, !dbg !6
  %1211 = bitcast float %1210 to i32, !dbg !6
  %1212 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1211), !dbg !6
  %1213 = bitcast i32 %1212 to float, !dbg !6
  %1214 = select i1 %.not40, float %1202, float %1213, !dbg !6
  %1215 = select i1 %.not40, float %1213, float %1208, !dbg !6
  %1216 = select i1 %.not40, float %1209, float %1203, !dbg !6
  %1217 = bitcast float %1216 to i32, !dbg !6
  %1218 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1217), !dbg !6
  %1219 = bitcast i32 %1218 to float, !dbg !6
  %1220 = select i1 %.not40, float %1203, float %1219, !dbg !6
  %1221 = select i1 %.not40, float %1219, float %1209, !dbg !6
  %1222 = extractvalue [2 x <8 x float>] %106, 0, !dbg !6
  %1223 = extractelement <8 x float> %1222, i64 0, !dbg !6
  %1224 = extractelement <8 x float> %1222, i64 1, !dbg !6
  %1225 = extractelement <8 x float> %1222, i64 2, !dbg !6
  %1226 = extractelement <8 x float> %1222, i64 3, !dbg !6
  %1227 = select i1 %.not, float %1224, float %1223, !dbg !6
  %1228 = bitcast float %1227 to i32, !dbg !6
  %1229 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1228), !dbg !6
  %1230 = bitcast i32 %1229 to float, !dbg !6
  %1231 = select i1 %.not, float %1223, float %1230, !dbg !6
  %1232 = select i1 %.not, float %1230, float %1224, !dbg !6
  %1233 = select i1 %.not, float %1226, float %1225, !dbg !6
  %1234 = bitcast float %1233 to i32, !dbg !6
  %1235 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1234), !dbg !6
  %1236 = bitcast i32 %1235 to float, !dbg !6
  %1237 = select i1 %.not, float %1225, float %1236, !dbg !6
  %1238 = select i1 %.not, float %1236, float %1226, !dbg !6
  %1239 = select i1 %.not40, float %1237, float %1231, !dbg !6
  %1240 = bitcast float %1239 to i32, !dbg !6
  %1241 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1240), !dbg !6
  %1242 = bitcast i32 %1241 to float, !dbg !6
  %1243 = select i1 %.not40, float %1231, float %1242, !dbg !6
  %1244 = select i1 %.not40, float %1242, float %1237, !dbg !6
  %1245 = select i1 %.not40, float %1238, float %1232, !dbg !6
  %1246 = bitcast float %1245 to i32, !dbg !6
  %1247 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1246), !dbg !6
  %1248 = bitcast i32 %1247 to float, !dbg !6
  %1249 = select i1 %.not40, float %1232, float %1248, !dbg !6
  %1250 = select i1 %.not40, float %1248, float %1238, !dbg !6
  %1251 = extractelement <8 x float> %1222, i64 4, !dbg !6
  %1252 = extractelement <8 x float> %1222, i64 5, !dbg !6
  %1253 = extractelement <8 x float> %1222, i64 6, !dbg !6
  %1254 = extractelement <8 x float> %1222, i64 7, !dbg !6
  %1255 = select i1 %.not, float %1252, float %1251, !dbg !6
  %1256 = bitcast float %1255 to i32, !dbg !6
  %1257 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1256), !dbg !6
  %1258 = bitcast i32 %1257 to float, !dbg !6
  %1259 = select i1 %.not, float %1251, float %1258, !dbg !6
  %1260 = select i1 %.not, float %1258, float %1252, !dbg !6
  %1261 = select i1 %.not, float %1254, float %1253, !dbg !6
  %1262 = bitcast float %1261 to i32, !dbg !6
  %1263 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1262), !dbg !6
  %1264 = bitcast i32 %1263 to float, !dbg !6
  %1265 = select i1 %.not, float %1253, float %1264, !dbg !6
  %1266 = select i1 %.not, float %1264, float %1254, !dbg !6
  %1267 = select i1 %.not40, float %1265, float %1259, !dbg !6
  %1268 = bitcast float %1267 to i32, !dbg !6
  %1269 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1268), !dbg !6
  %1270 = bitcast i32 %1269 to float, !dbg !6
  %1271 = select i1 %.not40, float %1259, float %1270, !dbg !6
  %1272 = select i1 %.not40, float %1270, float %1265, !dbg !6
  %1273 = select i1 %.not40, float %1266, float %1260, !dbg !6
  %1274 = bitcast float %1273 to i32, !dbg !6
  %1275 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1274), !dbg !6
  %1276 = bitcast i32 %1275 to float, !dbg !6
  %1277 = select i1 %.not40, float %1260, float %1276, !dbg !6
  %1278 = select i1 %.not40, float %1276, float %1266, !dbg !6
  %1279 = extractvalue [2 x <8 x float>] %106, 1, !dbg !6
  %1280 = extractelement <8 x float> %1279, i64 0, !dbg !6
  %1281 = extractelement <8 x float> %1279, i64 1, !dbg !6
  %1282 = extractelement <8 x float> %1279, i64 2, !dbg !6
  %1283 = extractelement <8 x float> %1279, i64 3, !dbg !6
  %1284 = select i1 %.not, float %1281, float %1280, !dbg !6
  %1285 = bitcast float %1284 to i32, !dbg !6
  %1286 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1285), !dbg !6
  %1287 = bitcast i32 %1286 to float, !dbg !6
  %1288 = select i1 %.not, float %1280, float %1287, !dbg !6
  %1289 = select i1 %.not, float %1287, float %1281, !dbg !6
  %1290 = select i1 %.not, float %1283, float %1282, !dbg !6
  %1291 = bitcast float %1290 to i32, !dbg !6
  %1292 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1291), !dbg !6
  %1293 = bitcast i32 %1292 to float, !dbg !6
  %1294 = select i1 %.not, float %1282, float %1293, !dbg !6
  %1295 = select i1 %.not, float %1293, float %1283, !dbg !6
  %1296 = select i1 %.not40, float %1294, float %1288, !dbg !6
  %1297 = bitcast float %1296 to i32, !dbg !6
  %1298 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1297), !dbg !6
  %1299 = bitcast i32 %1298 to float, !dbg !6
  %1300 = select i1 %.not40, float %1288, float %1299, !dbg !6
  %1301 = select i1 %.not40, float %1299, float %1294, !dbg !6
  %1302 = select i1 %.not40, float %1295, float %1289, !dbg !6
  %1303 = bitcast float %1302 to i32, !dbg !6
  %1304 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1303), !dbg !6
  %1305 = bitcast i32 %1304 to float, !dbg !6
  %1306 = select i1 %.not40, float %1289, float %1305, !dbg !6
  %1307 = select i1 %.not40, float %1305, float %1295, !dbg !6
  %1308 = extractelement <8 x float> %1279, i64 4, !dbg !6
  %1309 = extractelement <8 x float> %1279, i64 5, !dbg !6
  %1310 = extractelement <8 x float> %1279, i64 6, !dbg !6
  %1311 = extractelement <8 x float> %1279, i64 7, !dbg !6
  %1312 = select i1 %.not, float %1309, float %1308, !dbg !6
  %1313 = bitcast float %1312 to i32, !dbg !6
  %1314 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1313), !dbg !6
  %1315 = bitcast i32 %1314 to float, !dbg !6
  %1316 = select i1 %.not, float %1308, float %1315, !dbg !6
  %1317 = select i1 %.not, float %1315, float %1309, !dbg !6
  %1318 = select i1 %.not, float %1311, float %1310, !dbg !6
  %1319 = bitcast float %1318 to i32, !dbg !6
  %1320 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1319), !dbg !6
  %1321 = bitcast i32 %1320 to float, !dbg !6
  %1322 = select i1 %.not, float %1310, float %1321, !dbg !6
  %1323 = select i1 %.not, float %1321, float %1311, !dbg !6
  %1324 = select i1 %.not40, float %1322, float %1316, !dbg !6
  %1325 = bitcast float %1324 to i32, !dbg !6
  %1326 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1325), !dbg !6
  %1327 = bitcast i32 %1326 to float, !dbg !6
  %1328 = select i1 %.not40, float %1316, float %1327, !dbg !6
  %1329 = select i1 %.not40, float %1327, float %1322, !dbg !6
  %1330 = select i1 %.not40, float %1323, float %1317, !dbg !6
  %1331 = bitcast float %1330 to i32, !dbg !6
  %1332 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1331), !dbg !6
  %1333 = bitcast i32 %1332 to float, !dbg !6
  %1334 = select i1 %.not40, float %1317, float %1333, !dbg !6
  %1335 = select i1 %.not40, float %1333, float %1323, !dbg !6
  %1336 = extractvalue [2 x <8 x float>] %105, 0, !dbg !6
  %1337 = extractelement <8 x float> %1336, i64 0, !dbg !6
  %1338 = extractelement <8 x float> %1336, i64 1, !dbg !6
  %1339 = extractelement <8 x float> %1336, i64 2, !dbg !6
  %1340 = extractelement <8 x float> %1336, i64 3, !dbg !6
  %1341 = select i1 %.not, float %1338, float %1337, !dbg !6
  %1342 = bitcast float %1341 to i32, !dbg !6
  %1343 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1342), !dbg !6
  %1344 = bitcast i32 %1343 to float, !dbg !6
  %1345 = select i1 %.not, float %1337, float %1344, !dbg !6
  %1346 = select i1 %.not, float %1344, float %1338, !dbg !6
  %1347 = select i1 %.not, float %1340, float %1339, !dbg !6
  %1348 = bitcast float %1347 to i32, !dbg !6
  %1349 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1348), !dbg !6
  %1350 = bitcast i32 %1349 to float, !dbg !6
  %1351 = select i1 %.not, float %1339, float %1350, !dbg !6
  %1352 = select i1 %.not, float %1350, float %1340, !dbg !6
  %1353 = select i1 %.not40, float %1351, float %1345, !dbg !6
  %1354 = bitcast float %1353 to i32, !dbg !6
  %1355 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1354), !dbg !6
  %1356 = bitcast i32 %1355 to float, !dbg !6
  %1357 = select i1 %.not40, float %1345, float %1356, !dbg !6
  %1358 = select i1 %.not40, float %1356, float %1351, !dbg !6
  %1359 = select i1 %.not40, float %1352, float %1346, !dbg !6
  %1360 = bitcast float %1359 to i32, !dbg !6
  %1361 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1360), !dbg !6
  %1362 = bitcast i32 %1361 to float, !dbg !6
  %1363 = select i1 %.not40, float %1346, float %1362, !dbg !6
  %1364 = select i1 %.not40, float %1362, float %1352, !dbg !6
  %1365 = extractelement <8 x float> %1336, i64 4, !dbg !6
  %1366 = extractelement <8 x float> %1336, i64 5, !dbg !6
  %1367 = extractelement <8 x float> %1336, i64 6, !dbg !6
  %1368 = extractelement <8 x float> %1336, i64 7, !dbg !6
  %1369 = select i1 %.not, float %1366, float %1365, !dbg !6
  %1370 = bitcast float %1369 to i32, !dbg !6
  %1371 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1370), !dbg !6
  %1372 = bitcast i32 %1371 to float, !dbg !6
  %1373 = select i1 %.not, float %1365, float %1372, !dbg !6
  %1374 = select i1 %.not, float %1372, float %1366, !dbg !6
  %1375 = select i1 %.not, float %1368, float %1367, !dbg !6
  %1376 = bitcast float %1375 to i32, !dbg !6
  %1377 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1376), !dbg !6
  %1378 = bitcast i32 %1377 to float, !dbg !6
  %1379 = select i1 %.not, float %1367, float %1378, !dbg !6
  %1380 = select i1 %.not, float %1378, float %1368, !dbg !6
  %1381 = select i1 %.not40, float %1379, float %1373, !dbg !6
  %1382 = bitcast float %1381 to i32, !dbg !6
  %1383 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1382), !dbg !6
  %1384 = bitcast i32 %1383 to float, !dbg !6
  %1385 = select i1 %.not40, float %1373, float %1384, !dbg !6
  %1386 = select i1 %.not40, float %1384, float %1379, !dbg !6
  %1387 = select i1 %.not40, float %1380, float %1374, !dbg !6
  %1388 = bitcast float %1387 to i32, !dbg !6
  %1389 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1388), !dbg !6
  %1390 = bitcast i32 %1389 to float, !dbg !6
  %1391 = select i1 %.not40, float %1374, float %1390, !dbg !6
  %1392 = select i1 %.not40, float %1390, float %1380, !dbg !6
  %1393 = extractvalue [2 x <8 x float>] %105, 1, !dbg !6
  %1394 = extractelement <8 x float> %1393, i64 0, !dbg !6
  %1395 = extractelement <8 x float> %1393, i64 1, !dbg !6
  %1396 = extractelement <8 x float> %1393, i64 2, !dbg !6
  %1397 = extractelement <8 x float> %1393, i64 3, !dbg !6
  %1398 = select i1 %.not, float %1395, float %1394, !dbg !6
  %1399 = bitcast float %1398 to i32, !dbg !6
  %1400 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1399), !dbg !6
  %1401 = bitcast i32 %1400 to float, !dbg !6
  %1402 = select i1 %.not, float %1394, float %1401, !dbg !6
  %1403 = select i1 %.not, float %1401, float %1395, !dbg !6
  %1404 = select i1 %.not, float %1397, float %1396, !dbg !6
  %1405 = bitcast float %1404 to i32, !dbg !6
  %1406 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1405), !dbg !6
  %1407 = bitcast i32 %1406 to float, !dbg !6
  %1408 = select i1 %.not, float %1396, float %1407, !dbg !6
  %1409 = select i1 %.not, float %1407, float %1397, !dbg !6
  %1410 = select i1 %.not40, float %1408, float %1402, !dbg !6
  %1411 = bitcast float %1410 to i32, !dbg !6
  %1412 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1411), !dbg !6
  %1413 = bitcast i32 %1412 to float, !dbg !6
  %1414 = select i1 %.not40, float %1402, float %1413, !dbg !6
  %1415 = select i1 %.not40, float %1413, float %1408, !dbg !6
  %1416 = select i1 %.not40, float %1409, float %1403, !dbg !6
  %1417 = bitcast float %1416 to i32, !dbg !6
  %1418 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1417), !dbg !6
  %1419 = bitcast i32 %1418 to float, !dbg !6
  %1420 = select i1 %.not40, float %1403, float %1419, !dbg !6
  %1421 = select i1 %.not40, float %1419, float %1409, !dbg !6
  %1422 = extractelement <8 x float> %1393, i64 4, !dbg !6
  %1423 = extractelement <8 x float> %1393, i64 5, !dbg !6
  %1424 = extractelement <8 x float> %1393, i64 6, !dbg !6
  %1425 = extractelement <8 x float> %1393, i64 7, !dbg !6
  %1426 = select i1 %.not, float %1423, float %1422, !dbg !6
  %1427 = bitcast float %1426 to i32, !dbg !6
  %1428 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1427), !dbg !6
  %1429 = bitcast i32 %1428 to float, !dbg !6
  %1430 = select i1 %.not, float %1422, float %1429, !dbg !6
  %1431 = select i1 %.not, float %1429, float %1423, !dbg !6
  %1432 = select i1 %.not, float %1425, float %1424, !dbg !6
  %1433 = bitcast float %1432 to i32, !dbg !6
  %1434 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1433), !dbg !6
  %1435 = bitcast i32 %1434 to float, !dbg !6
  %1436 = select i1 %.not, float %1424, float %1435, !dbg !6
  %1437 = select i1 %.not, float %1435, float %1425, !dbg !6
  %1438 = select i1 %.not40, float %1436, float %1430, !dbg !6
  %1439 = bitcast float %1438 to i32, !dbg !6
  %1440 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1439), !dbg !6
  %1441 = bitcast i32 %1440 to float, !dbg !6
  %1442 = select i1 %.not40, float %1430, float %1441, !dbg !6
  %1443 = select i1 %.not40, float %1441, float %1436, !dbg !6
  %1444 = select i1 %.not40, float %1437, float %1431, !dbg !6
  %1445 = bitcast float %1444 to i32, !dbg !6
  %1446 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1445), !dbg !6
  %1447 = bitcast i32 %1446 to float, !dbg !6
  %1448 = select i1 %.not40, float %1431, float %1447, !dbg !6
  %1449 = select i1 %.not40, float %1447, float %1437, !dbg !6
  %1450 = extractvalue [2 x <8 x float>] %102, 0, !dbg !6
  %1451 = extractelement <8 x float> %1450, i64 0, !dbg !6
  %1452 = extractelement <8 x float> %1450, i64 1, !dbg !6
  %1453 = extractelement <8 x float> %1450, i64 2, !dbg !6
  %1454 = extractelement <8 x float> %1450, i64 3, !dbg !6
  %1455 = select i1 %.not, float %1452, float %1451, !dbg !6
  %1456 = bitcast float %1455 to i32, !dbg !6
  %1457 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1456), !dbg !6
  %1458 = bitcast i32 %1457 to float, !dbg !6
  %1459 = select i1 %.not, float %1451, float %1458, !dbg !6
  %1460 = select i1 %.not, float %1458, float %1452, !dbg !6
  %1461 = select i1 %.not, float %1454, float %1453, !dbg !6
  %1462 = bitcast float %1461 to i32, !dbg !6
  %1463 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1462), !dbg !6
  %1464 = bitcast i32 %1463 to float, !dbg !6
  %1465 = select i1 %.not, float %1453, float %1464, !dbg !6
  %1466 = select i1 %.not, float %1464, float %1454, !dbg !6
  %1467 = select i1 %.not40, float %1465, float %1459, !dbg !6
  %1468 = bitcast float %1467 to i32, !dbg !6
  %1469 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1468), !dbg !6
  %1470 = select i1 %.not40, float %1466, float %1460, !dbg !6
  %1471 = bitcast float %1470 to i32, !dbg !6
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1471), !dbg !6
  %1473 = extractelement <8 x float> %1450, i64 4, !dbg !6
  %1474 = extractelement <8 x float> %1450, i64 5, !dbg !6
  %1475 = extractelement <8 x float> %1450, i64 6, !dbg !6
  %1476 = extractelement <8 x float> %1450, i64 7, !dbg !6
  %1477 = select i1 %.not, float %1474, float %1473, !dbg !6
  %1478 = bitcast float %1477 to i32, !dbg !6
  %1479 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1478), !dbg !6
  %1480 = bitcast i32 %1479 to float, !dbg !6
  %1481 = select i1 %.not, float %1473, float %1480, !dbg !6
  %1482 = select i1 %.not, float %1480, float %1474, !dbg !6
  %1483 = select i1 %.not, float %1476, float %1475, !dbg !6
  %1484 = bitcast float %1483 to i32, !dbg !6
  %1485 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1484), !dbg !6
  %1486 = bitcast i32 %1485 to float, !dbg !6
  %1487 = select i1 %.not, float %1475, float %1486, !dbg !6
  %1488 = select i1 %.not, float %1486, float %1476, !dbg !6
  %1489 = select i1 %.not40, float %1487, float %1481, !dbg !6
  %1490 = bitcast float %1489 to i32, !dbg !6
  %1491 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1490), !dbg !6
  %1492 = select i1 %.not40, float %1488, float %1482, !dbg !6
  %1493 = bitcast float %1492 to i32, !dbg !6
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1493), !dbg !6
  %1495 = extractvalue [2 x <8 x float>] %102, 1, !dbg !6
  %1496 = extractelement <8 x float> %1495, i64 0, !dbg !6
  %1497 = extractelement <8 x float> %1495, i64 1, !dbg !6
  %1498 = extractelement <8 x float> %1495, i64 2, !dbg !6
  %1499 = extractelement <8 x float> %1495, i64 3, !dbg !6
  %1500 = select i1 %.not, float %1497, float %1496, !dbg !6
  %1501 = bitcast float %1500 to i32, !dbg !6
  %1502 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1501), !dbg !6
  %1503 = bitcast i32 %1502 to float, !dbg !6
  %1504 = select i1 %.not, float %1496, float %1503, !dbg !6
  %1505 = select i1 %.not, float %1503, float %1497, !dbg !6
  %1506 = select i1 %.not, float %1499, float %1498, !dbg !6
  %1507 = bitcast float %1506 to i32, !dbg !6
  %1508 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1507), !dbg !6
  %1509 = bitcast i32 %1508 to float, !dbg !6
  %1510 = select i1 %.not, float %1498, float %1509, !dbg !6
  %1511 = select i1 %.not, float %1509, float %1499, !dbg !6
  %1512 = select i1 %.not40, float %1510, float %1504, !dbg !6
  %1513 = bitcast float %1512 to i32, !dbg !6
  %1514 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1513), !dbg !6
  %1515 = select i1 %.not40, float %1511, float %1505, !dbg !6
  %1516 = bitcast float %1515 to i32, !dbg !6
  %1517 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1516), !dbg !6
  %1518 = extractelement <8 x float> %1495, i64 4, !dbg !6
  %1519 = extractelement <8 x float> %1495, i64 5, !dbg !6
  %1520 = extractelement <8 x float> %1495, i64 6, !dbg !6
  %1521 = extractelement <8 x float> %1495, i64 7, !dbg !6
  %1522 = select i1 %.not, float %1519, float %1518, !dbg !6
  %1523 = bitcast float %1522 to i32, !dbg !6
  %1524 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1523), !dbg !6
  %1525 = bitcast i32 %1524 to float, !dbg !6
  %1526 = select i1 %.not, float %1518, float %1525, !dbg !6
  %1527 = select i1 %.not, float %1525, float %1519, !dbg !6
  %1528 = select i1 %.not, float %1521, float %1520, !dbg !6
  %1529 = bitcast float %1528 to i32, !dbg !6
  %1530 = call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1529), !dbg !6
  %1531 = bitcast i32 %1530 to float, !dbg !6
  %1532 = select i1 %.not, float %1520, float %1531, !dbg !6
  %1533 = select i1 %.not, float %1531, float %1521, !dbg !6
  %1534 = select i1 %.not40, float %1532, float %1526, !dbg !6
  %1535 = bitcast float %1534 to i32, !dbg !6
  %1536 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1535), !dbg !6
  %1537 = select i1 %.not40, float %1533, float %1527, !dbg !6
  %1538 = bitcast float %1537 to i32, !dbg !6
  %1539 = call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1538), !dbg !6
  %1540 = insertelement <4 x float> poison, float %1105, i64 0, !dbg !17
  %1541 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1540, !dbg !17
  %1542 = shufflevector <4 x float> %1541, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1543 = insertelement <4 x float> poison, float %1095, i64 0, !dbg !17
  %1544 = fdiv arcp <4 x float> <float 0x3FF0000000000000, float poison, float poison, float poison>, %1543, !dbg !17
  %1545 = shufflevector <4 x float> %1544, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1546 = bitcast i32 %1536 to float, !dbg !6
  %1547 = select i1 %.not40, float %1526, float %1546, !dbg !6
  %1548 = insertelement <4 x float> poison, float %1547, i64 0, !dbg !17
  %1549 = bitcast i32 %1539 to float, !dbg !6
  %1550 = select i1 %.not40, float %1527, float %1549, !dbg !6
  %1551 = insertelement <4 x float> %1548, float %1550, i64 1, !dbg !17
  %1552 = select i1 %.not40, float %1546, float %1532, !dbg !6
  %1553 = insertelement <4 x float> %1551, float %1552, i64 2, !dbg !17
  %1554 = select i1 %.not40, float %1549, float %1533, !dbg !6
  %1555 = insertelement <4 x float> %1553, float %1554, i64 3, !dbg !17
  %1556 = fmul <4 x float> %1555, %1545, !dbg !17
  %1557 = fptrunc <4 x float> %1556 to <4 x half>, !dbg !18
  %1558 = extractelement <4 x half> %1557, i64 3, !dbg !18
  %1559 = extractelement <4 x half> %1557, i64 2, !dbg !18
  %1560 = extractelement <4 x half> %1557, i64 1, !dbg !18
  %1561 = extractelement <4 x half> %1557, i64 0, !dbg !18
  %1562 = bitcast i32 %1514 to float, !dbg !6
  %1563 = select i1 %.not40, float %1504, float %1562, !dbg !6
  %1564 = insertelement <4 x float> poison, float %1563, i64 0, !dbg !17
  %1565 = bitcast i32 %1517 to float, !dbg !6
  %1566 = select i1 %.not40, float %1505, float %1565, !dbg !6
  %1567 = insertelement <4 x float> %1564, float %1566, i64 1, !dbg !17
  %1568 = select i1 %.not40, float %1562, float %1510, !dbg !6
  %1569 = insertelement <4 x float> %1567, float %1568, i64 2, !dbg !17
  %1570 = select i1 %.not40, float %1565, float %1511, !dbg !6
  %1571 = insertelement <4 x float> %1569, float %1570, i64 3, !dbg !17
  %1572 = fmul <4 x float> %1571, %1545, !dbg !17
  %1573 = fptrunc <4 x float> %1572 to <4 x half>, !dbg !18
  %1574 = extractelement <4 x half> %1573, i64 3, !dbg !18
  %1575 = extractelement <4 x half> %1573, i64 2, !dbg !18
  %1576 = extractelement <4 x half> %1573, i64 1, !dbg !18
  %1577 = extractelement <4 x half> %1573, i64 0, !dbg !18
  %1578 = bitcast i32 %1491 to float, !dbg !6
  %1579 = select i1 %.not40, float %1481, float %1578, !dbg !6
  %1580 = insertelement <4 x float> poison, float %1579, i64 0, !dbg !17
  %1581 = bitcast i32 %1494 to float, !dbg !6
  %1582 = select i1 %.not40, float %1482, float %1581, !dbg !6
  %1583 = insertelement <4 x float> %1580, float %1582, i64 1, !dbg !17
  %1584 = select i1 %.not40, float %1578, float %1487, !dbg !6
  %1585 = insertelement <4 x float> %1583, float %1584, i64 2, !dbg !17
  %1586 = select i1 %.not40, float %1581, float %1488, !dbg !6
  %1587 = insertelement <4 x float> %1585, float %1586, i64 3, !dbg !17
  %1588 = fmul <4 x float> %1587, %1542, !dbg !17
  %1589 = fptrunc <4 x float> %1588 to <4 x half>, !dbg !18
  %1590 = extractelement <4 x half> %1589, i64 3, !dbg !18
  %1591 = extractelement <4 x half> %1589, i64 2, !dbg !18
  %1592 = extractelement <4 x half> %1589, i64 1, !dbg !18
  %1593 = extractelement <4 x half> %1589, i64 0, !dbg !18
  %1594 = bitcast i32 %1469 to float, !dbg !6
  %1595 = select i1 %.not40, float %1459, float %1594, !dbg !6
  %1596 = insertelement <4 x float> poison, float %1595, i64 0, !dbg !17
  %1597 = bitcast i32 %1472 to float, !dbg !6
  %1598 = select i1 %.not40, float %1460, float %1597, !dbg !6
  %1599 = insertelement <4 x float> %1596, float %1598, i64 1, !dbg !17
  %1600 = select i1 %.not40, float %1594, float %1465, !dbg !6
  %1601 = insertelement <4 x float> %1599, float %1600, i64 2, !dbg !17
  %1602 = select i1 %.not40, float %1597, float %1466, !dbg !6
  %1603 = insertelement <4 x float> %1601, float %1602, i64 3, !dbg !17
  %1604 = fmul <4 x float> %1603, %1542, !dbg !17
  %1605 = fptrunc <4 x float> %1604 to <4 x half>, !dbg !18
  %1606 = extractelement <4 x half> %1605, i64 3, !dbg !18
  %1607 = extractelement <4 x half> %1605, i64 2, !dbg !18
  %1608 = extractelement <4 x half> %1605, i64 1, !dbg !18
  %1609 = extractelement <4 x half> %1605, i64 0, !dbg !18
  %address.thread883 = call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #9, !dbg !19, !range !30, !frisk.register_lifetimes !1
  %address.local884 = zext nneg i32 %address.thread883 to i64, !dbg !9
  %address.local885 = add nuw i64 %32, %address.local884, !dbg !19
  %1610 = lshr i64 %address.local884, 2, !dbg !19
  %address.local887 = and i64 %1610, 48, !dbg !19
  %address.local888 = add i64 %address.local885, %address.local887, !dbg !19
  %address.local889 = and i64 %address.local884, 240, !dbg !19
  %address.local890 = sub i64 %address.local888, %address.local889, !dbg !19
  %address.local891 = add i64 %address.local890, %33, !dbg !19
  %address.local892 = shl nuw nsw i64 %address.local891, 7, !dbg !19
  %1611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %49, !dbg !19
  %address.local894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %1611, i64 %address.local892, !dbg !19
  %address.local896 = and i64 %1610, 12, !dbg !12
  %address.local897 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local896, !dbg !19
  store half %1609, ptr addrspace(1) %address.local897, align 2, !dbg !19
  %address.local899 = or disjoint i64 %address.local896, 1, !dbg !19
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local899, !dbg !19
  store half %1608, ptr addrspace(1) %address.local900, align 2, !dbg !19
  %address.local902 = or disjoint i64 %address.local896, 2, !dbg !19
  %address.local903 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local902, !dbg !19
  store half %1607, ptr addrspace(1) %address.local903, align 2, !dbg !19
  %address.local905 = or disjoint i64 %address.local896, 3, !dbg !19
  %address.local906 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local905, !dbg !19
  store half %1606, ptr addrspace(1) %address.local906, align 2, !dbg !19
  %address.local908 = or disjoint i64 %address.local896, 16, !dbg !19
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local908, !dbg !19
  store half %1593, ptr addrspace(1) %address.local909, align 2, !dbg !19
  %address.local911 = or disjoint i64 %address.local896, 17, !dbg !19
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local911, !dbg !19
  store half %1592, ptr addrspace(1) %address.local912, align 2, !dbg !19
  %address.local914 = or disjoint i64 %address.local896, 18, !dbg !19
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local914, !dbg !19
  store half %1591, ptr addrspace(1) %address.local915, align 2, !dbg !19
  %address.local917 = or disjoint i64 %address.local896, 19, !dbg !19
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local917, !dbg !19
  store half %1590, ptr addrspace(1) %address.local918, align 2, !dbg !19
  %address.local922 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local894, i64 16384, !dbg !19
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local896, !dbg !19
  store half %1577, ptr addrspace(1) %address.local923, align 2, !dbg !19
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local899, !dbg !19
  store half %1576, ptr addrspace(1) %address.local925, align 2, !dbg !19
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local902, !dbg !19
  store half %1575, ptr addrspace(1) %address.local927, align 2, !dbg !19
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local905, !dbg !19
  store half %1574, ptr addrspace(1) %address.local929, align 2, !dbg !19
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local908, !dbg !19
  store half %1561, ptr addrspace(1) %address.local931, align 2, !dbg !19
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local911, !dbg !19
  store half %1560, ptr addrspace(1) %address.local933, align 2, !dbg !19
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local914, !dbg !19
  store half %1559, ptr addrspace(1) %address.local935, align 2, !dbg !19
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local917, !dbg !19
  store half %1558, ptr addrspace(1) %address.local937, align 2, !dbg !19
  %1612 = insertelement <4 x float> poison, float %1442, i64 0, !dbg !17
  %1613 = insertelement <4 x float> %1612, float %1448, i64 1, !dbg !17
  %1614 = insertelement <4 x float> %1613, float %1443, i64 2, !dbg !17
  %1615 = insertelement <4 x float> %1614, float %1449, i64 3, !dbg !17
  %1616 = fmul <4 x float> %1615, %1545, !dbg !17
  %1617 = fptrunc <4 x float> %1616 to <4 x half>, !dbg !18
  %1618 = extractelement <4 x half> %1617, i64 3, !dbg !18
  %1619 = extractelement <4 x half> %1617, i64 2, !dbg !18
  %1620 = extractelement <4 x half> %1617, i64 1, !dbg !18
  %1621 = extractelement <4 x half> %1617, i64 0, !dbg !18
  %1622 = insertelement <4 x float> poison, float %1414, i64 0, !dbg !17
  %1623 = insertelement <4 x float> %1622, float %1420, i64 1, !dbg !17
  %1624 = insertelement <4 x float> %1623, float %1415, i64 2, !dbg !17
  %1625 = insertelement <4 x float> %1624, float %1421, i64 3, !dbg !17
  %1626 = fmul <4 x float> %1625, %1545, !dbg !17
  %1627 = fptrunc <4 x float> %1626 to <4 x half>, !dbg !18
  %1628 = extractelement <4 x half> %1627, i64 3, !dbg !18
  %1629 = extractelement <4 x half> %1627, i64 2, !dbg !18
  %1630 = extractelement <4 x half> %1627, i64 1, !dbg !18
  %1631 = extractelement <4 x half> %1627, i64 0, !dbg !18
  %1632 = insertelement <4 x float> poison, float %1385, i64 0, !dbg !17
  %1633 = insertelement <4 x float> %1632, float %1391, i64 1, !dbg !17
  %1634 = insertelement <4 x float> %1633, float %1386, i64 2, !dbg !17
  %1635 = insertelement <4 x float> %1634, float %1392, i64 3, !dbg !17
  %1636 = fmul <4 x float> %1635, %1542, !dbg !17
  %1637 = fptrunc <4 x float> %1636 to <4 x half>, !dbg !18
  %1638 = extractelement <4 x half> %1637, i64 3, !dbg !18
  %1639 = extractelement <4 x half> %1637, i64 2, !dbg !18
  %1640 = extractelement <4 x half> %1637, i64 1, !dbg !18
  %1641 = extractelement <4 x half> %1637, i64 0, !dbg !18
  %1642 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !17
  %1643 = insertelement <4 x float> %1642, float %1363, i64 1, !dbg !17
  %1644 = insertelement <4 x float> %1643, float %1358, i64 2, !dbg !17
  %1645 = insertelement <4 x float> %1644, float %1364, i64 3, !dbg !17
  %1646 = fmul <4 x float> %1645, %1542, !dbg !17
  %1647 = fptrunc <4 x float> %1646 to <4 x half>, !dbg !18
  %1648 = extractelement <4 x half> %1647, i64 3, !dbg !18
  %1649 = extractelement <4 x half> %1647, i64 2, !dbg !18
  %1650 = extractelement <4 x half> %1647, i64 1, !dbg !18
  %1651 = extractelement <4 x half> %1647, i64 0, !dbg !18
  %address.local952 = or disjoint i64 %address.local896, 32, !dbg !19
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local952, !dbg !19
  store half %1651, ptr addrspace(1) %address.local953, align 2, !dbg !19
  %address.local955 = or disjoint i64 %address.local896, 33, !dbg !19
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local955, !dbg !19
  store half %1650, ptr addrspace(1) %address.local956, align 2, !dbg !19
  %address.local958 = or disjoint i64 %address.local896, 34, !dbg !19
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local958, !dbg !19
  store half %1649, ptr addrspace(1) %address.local959, align 2, !dbg !19
  %address.local961 = or disjoint i64 %address.local896, 35, !dbg !19
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local961, !dbg !19
  store half %1648, ptr addrspace(1) %address.local962, align 2, !dbg !19
  %address.local964 = or disjoint i64 %address.local896, 48, !dbg !19
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local964, !dbg !19
  store half %1641, ptr addrspace(1) %address.local965, align 2, !dbg !19
  %address.local967 = or disjoint i64 %address.local896, 49, !dbg !19
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local967, !dbg !19
  store half %1640, ptr addrspace(1) %address.local968, align 2, !dbg !19
  %address.local970 = or disjoint i64 %address.local896, 50, !dbg !19
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local970, !dbg !19
  store half %1639, ptr addrspace(1) %address.local971, align 2, !dbg !19
  %address.local973 = or i64 %1610, 51, !dbg !19
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local973, !dbg !19
  store half %1638, ptr addrspace(1) %address.local974, align 2, !dbg !19
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local952, !dbg !19
  store half %1631, ptr addrspace(1) %address.local979, align 2, !dbg !19
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local955, !dbg !19
  store half %1630, ptr addrspace(1) %address.local981, align 2, !dbg !19
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local958, !dbg !19
  store half %1629, ptr addrspace(1) %address.local983, align 2, !dbg !19
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local961, !dbg !19
  store half %1628, ptr addrspace(1) %address.local985, align 2, !dbg !19
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local964, !dbg !19
  store half %1621, ptr addrspace(1) %address.local987, align 2, !dbg !19
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local967, !dbg !19
  store half %1620, ptr addrspace(1) %address.local989, align 2, !dbg !19
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local970, !dbg !19
  store half %1619, ptr addrspace(1) %address.local991, align 2, !dbg !19
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local973, !dbg !19
  store half %1618, ptr addrspace(1) %address.local993, align 2, !dbg !19
  %1652 = insertelement <4 x float> poison, float %1328, i64 0, !dbg !17
  %1653 = insertelement <4 x float> %1652, float %1334, i64 1, !dbg !17
  %1654 = insertelement <4 x float> %1653, float %1329, i64 2, !dbg !17
  %1655 = insertelement <4 x float> %1654, float %1335, i64 3, !dbg !17
  %1656 = fmul <4 x float> %1655, %1545, !dbg !17
  %1657 = fptrunc <4 x float> %1656 to <4 x half>, !dbg !18
  %1658 = extractelement <4 x half> %1657, i64 3, !dbg !18
  %1659 = extractelement <4 x half> %1657, i64 2, !dbg !18
  %1660 = extractelement <4 x half> %1657, i64 1, !dbg !18
  %1661 = extractelement <4 x half> %1657, i64 0, !dbg !18
  %1662 = insertelement <4 x float> poison, float %1300, i64 0, !dbg !17
  %1663 = insertelement <4 x float> %1662, float %1306, i64 1, !dbg !17
  %1664 = insertelement <4 x float> %1663, float %1301, i64 2, !dbg !17
  %1665 = insertelement <4 x float> %1664, float %1307, i64 3, !dbg !17
  %1666 = fmul <4 x float> %1665, %1545, !dbg !17
  %1667 = fptrunc <4 x float> %1666 to <4 x half>, !dbg !18
  %1668 = extractelement <4 x half> %1667, i64 3, !dbg !18
  %1669 = extractelement <4 x half> %1667, i64 2, !dbg !18
  %1670 = extractelement <4 x half> %1667, i64 1, !dbg !18
  %1671 = extractelement <4 x half> %1667, i64 0, !dbg !18
  %1672 = insertelement <4 x float> poison, float %1271, i64 0, !dbg !17
  %1673 = insertelement <4 x float> %1672, float %1277, i64 1, !dbg !17
  %1674 = insertelement <4 x float> %1673, float %1272, i64 2, !dbg !17
  %1675 = insertelement <4 x float> %1674, float %1278, i64 3, !dbg !17
  %1676 = fmul <4 x float> %1675, %1542, !dbg !17
  %1677 = fptrunc <4 x float> %1676 to <4 x half>, !dbg !18
  %1678 = extractelement <4 x half> %1677, i64 3, !dbg !18
  %1679 = extractelement <4 x half> %1677, i64 2, !dbg !18
  %1680 = extractelement <4 x half> %1677, i64 1, !dbg !18
  %1681 = extractelement <4 x half> %1677, i64 0, !dbg !18
  %1682 = insertelement <4 x float> poison, float %1243, i64 0, !dbg !17
  %1683 = insertelement <4 x float> %1682, float %1249, i64 1, !dbg !17
  %1684 = insertelement <4 x float> %1683, float %1244, i64 2, !dbg !17
  %1685 = insertelement <4 x float> %1684, float %1250, i64 3, !dbg !17
  %1686 = fmul <4 x float> %1685, %1542, !dbg !17
  %1687 = fptrunc <4 x float> %1686 to <4 x half>, !dbg !18
  %1688 = extractelement <4 x half> %1687, i64 3, !dbg !18
  %1689 = extractelement <4 x half> %1687, i64 2, !dbg !18
  %1690 = extractelement <4 x half> %1687, i64 1, !dbg !18
  %1691 = extractelement <4 x half> %1687, i64 0, !dbg !18
  %address.local1008 = or disjoint i64 %address.local896, 64, !dbg !19
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1008, !dbg !19
  store half %1691, ptr addrspace(1) %address.local1009, align 2, !dbg !19
  %address.local1011 = or disjoint i64 %address.local896, 65, !dbg !19
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1011, !dbg !19
  store half %1690, ptr addrspace(1) %address.local1012, align 2, !dbg !19
  %address.local1014 = or disjoint i64 %address.local896, 66, !dbg !19
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1014, !dbg !19
  store half %1689, ptr addrspace(1) %address.local1015, align 2, !dbg !19
  %address.local1017 = or disjoint i64 %address.local896, 67, !dbg !19
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1017, !dbg !19
  store half %1688, ptr addrspace(1) %address.local1018, align 2, !dbg !19
  %address.local1020 = or disjoint i64 %address.local896, 80, !dbg !19
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1020, !dbg !19
  store half %1681, ptr addrspace(1) %address.local1021, align 2, !dbg !19
  %address.local1023 = or disjoint i64 %address.local896, 81, !dbg !19
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1023, !dbg !19
  store half %1680, ptr addrspace(1) %address.local1024, align 2, !dbg !19
  %address.local1026 = or disjoint i64 %address.local896, 82, !dbg !19
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1026, !dbg !19
  store half %1679, ptr addrspace(1) %address.local1027, align 2, !dbg !19
  %address.local1029 = or disjoint i64 %address.local896, 83, !dbg !19
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1029, !dbg !19
  store half %1678, ptr addrspace(1) %address.local1030, align 2, !dbg !19
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1008, !dbg !19
  store half %1671, ptr addrspace(1) %address.local1035, align 2, !dbg !19
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1011, !dbg !19
  store half %1670, ptr addrspace(1) %address.local1037, align 2, !dbg !19
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1014, !dbg !19
  store half %1669, ptr addrspace(1) %address.local1039, align 2, !dbg !19
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1017, !dbg !19
  store half %1668, ptr addrspace(1) %address.local1041, align 2, !dbg !19
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1020, !dbg !19
  store half %1661, ptr addrspace(1) %address.local1043, align 2, !dbg !19
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1023, !dbg !19
  store half %1660, ptr addrspace(1) %address.local1045, align 2, !dbg !19
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1026, !dbg !19
  store half %1659, ptr addrspace(1) %address.local1047, align 2, !dbg !19
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1029, !dbg !19
  store half %1658, ptr addrspace(1) %address.local1049, align 2, !dbg !19
  %1692 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !17
  %1693 = insertelement <4 x float> %1692, float %1220, i64 1, !dbg !17
  %1694 = insertelement <4 x float> %1693, float %1215, i64 2, !dbg !17
  %1695 = insertelement <4 x float> %1694, float %1221, i64 3, !dbg !17
  %1696 = fmul <4 x float> %1695, %1545, !dbg !17
  %1697 = fptrunc <4 x float> %1696 to <4 x half>, !dbg !18
  %1698 = extractelement <4 x half> %1697, i64 3, !dbg !18
  %1699 = extractelement <4 x half> %1697, i64 2, !dbg !18
  %1700 = extractelement <4 x half> %1697, i64 1, !dbg !18
  %1701 = extractelement <4 x half> %1697, i64 0, !dbg !18
  %1702 = insertelement <4 x float> poison, float %1186, i64 0, !dbg !17
  %1703 = insertelement <4 x float> %1702, float %1192, i64 1, !dbg !17
  %1704 = insertelement <4 x float> %1703, float %1187, i64 2, !dbg !17
  %1705 = insertelement <4 x float> %1704, float %1193, i64 3, !dbg !17
  %1706 = fmul <4 x float> %1705, %1545, !dbg !17
  %1707 = fptrunc <4 x float> %1706 to <4 x half>, !dbg !18
  %1708 = extractelement <4 x half> %1707, i64 3, !dbg !18
  %1709 = extractelement <4 x half> %1707, i64 2, !dbg !18
  %1710 = extractelement <4 x half> %1707, i64 1, !dbg !18
  %1711 = extractelement <4 x half> %1707, i64 0, !dbg !18
  %1712 = insertelement <4 x float> poison, float %1157, i64 0, !dbg !17
  %1713 = insertelement <4 x float> %1712, float %1163, i64 1, !dbg !17
  %1714 = insertelement <4 x float> %1713, float %1158, i64 2, !dbg !17
  %1715 = insertelement <4 x float> %1714, float %1164, i64 3, !dbg !17
  %1716 = fmul <4 x float> %1715, %1542, !dbg !17
  %1717 = fptrunc <4 x float> %1716 to <4 x half>, !dbg !18
  %1718 = extractelement <4 x half> %1717, i64 3, !dbg !18
  %1719 = extractelement <4 x half> %1717, i64 2, !dbg !18
  %1720 = extractelement <4 x half> %1717, i64 1, !dbg !18
  %1721 = extractelement <4 x half> %1717, i64 0, !dbg !18
  %1722 = insertelement <4 x float> poison, float %1129, i64 0, !dbg !17
  %1723 = insertelement <4 x float> %1722, float %1135, i64 1, !dbg !17
  %1724 = insertelement <4 x float> %1723, float %1130, i64 2, !dbg !17
  %1725 = insertelement <4 x float> %1724, float %1136, i64 3, !dbg !17
  %1726 = fmul <4 x float> %1725, %1542, !dbg !17
  %1727 = fptrunc <4 x float> %1726 to <4 x half>, !dbg !18
  %1728 = extractelement <4 x half> %1727, i64 3, !dbg !18
  %1729 = extractelement <4 x half> %1727, i64 2, !dbg !18
  %1730 = extractelement <4 x half> %1727, i64 1, !dbg !18
  %1731 = extractelement <4 x half> %1727, i64 0, !dbg !18
  %address.local1064 = or disjoint i64 %address.local896, 96, !dbg !19
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1064, !dbg !19
  store half %1731, ptr addrspace(1) %address.local1065, align 2, !dbg !19
  %address.local1067 = or disjoint i64 %address.local896, 97, !dbg !19
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1067, !dbg !19
  store half %1730, ptr addrspace(1) %address.local1068, align 2, !dbg !19
  %address.local1070 = or disjoint i64 %address.local896, 98, !dbg !19
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1070, !dbg !19
  store half %1729, ptr addrspace(1) %address.local1071, align 2, !dbg !19
  %address.local1073 = or disjoint i64 %address.local896, 99, !dbg !19
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1073, !dbg !19
  store half %1728, ptr addrspace(1) %address.local1074, align 2, !dbg !19
  %address.local1076 = or disjoint i64 %address.local896, 112, !dbg !19
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1076, !dbg !19
  store half %1721, ptr addrspace(1) %address.local1077, align 2, !dbg !19
  %address.local1079 = or disjoint i64 %address.local896, 113, !dbg !19
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1079, !dbg !19
  store half %1720, ptr addrspace(1) %address.local1080, align 2, !dbg !19
  %address.local1082 = or disjoint i64 %address.local896, 114, !dbg !19
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1082, !dbg !19
  store half %1719, ptr addrspace(1) %address.local1083, align 2, !dbg !19
  %address.local1085 = or i64 %1610, 115, !dbg !19
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1085, !dbg !19
  store half %1718, ptr addrspace(1) %address.local1086, align 2, !dbg !19
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1064, !dbg !19
  store half %1711, ptr addrspace(1) %address.local1091, align 2, !dbg !19
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1067, !dbg !19
  store half %1710, ptr addrspace(1) %address.local1093, align 2, !dbg !19
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1070, !dbg !19
  store half %1709, ptr addrspace(1) %address.local1095, align 2, !dbg !19
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1073, !dbg !19
  store half %1708, ptr addrspace(1) %address.local1097, align 2, !dbg !19
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1076, !dbg !19
  store half %1701, ptr addrspace(1) %address.local1099, align 2, !dbg !19
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1079, !dbg !19
  store half %1700, ptr addrspace(1) %address.local1101, align 2, !dbg !19
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1082, !dbg !19
  store half %1699, ptr addrspace(1) %address.local1103, align 2, !dbg !19
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1085, !dbg !19
  store half %1698, ptr addrspace(1) %address.local1105, align 2, !dbg !19
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

!0 = !DIFile(filename: "noncausal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer")
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
!32 = !{i32 0}
!33 = !{i32 1}
!34 = !{i32 2}
!35 = !{i32 3}
!36 = !{i32 5}
!37 = !{i32 6}
!38 = !{i32 7}
!39 = !{i32 8}
!40 = !{i32 9}
!41 = !{i32 10}
!42 = !{i32 11}
!43 = !{i32 12}
!44 = !{i32 13}
!45 = !{i32 14}
!46 = !{i32 15}
!47 = !{i32 16}
!48 = !{i32 17}
!49 = !{i32 18}
!50 = !{i32 19}
!51 = !{i32 20}
!52 = !{i32 21}
!53 = !{i32 22}
