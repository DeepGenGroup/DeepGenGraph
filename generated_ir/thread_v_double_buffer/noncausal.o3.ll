; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/noncausal.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer/noncausal.input.mlir"

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
  %address.local279 = zext nneg i32 %37 to i64, !dbg !9
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local279, !dbg !9
  %address.local283 = lshr i32 %address.thread, 4, !dbg !9
  %address.local284 = and i32 %address.local283, 8, !dbg !9
  %address.local285 = shl nuw nsw i32 %address.thread, 7, !dbg !9
  %address.local286 = and i32 %address.local285, 16256, !dbg !9
  %address.local287 = or disjoint i32 %address.local284, %address.local286, !dbg !9
  %38 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1285 = shl i64 %34, 8, !dbg !9
  %39 = add i64 %.idx1285, %.idx, !dbg !9
  %40 = shl nuw nsw i32 %address.local287, 1, !dbg !9
  %41 = trunc i64 %39 to i32, !dbg !9
  %dma.buffer.offset1283 = or disjoint i32 %40, %41, !dbg !9
  %42 = insertelement <2 x i64> poison, i64 %38, i64 0, !dbg !9
  %43 = bitcast <2 x i64> %42 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1284 = shufflevector <4 x i32> %43, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local292 = or disjoint i64 %address.local279, 2048, !dbg !9
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local292, !dbg !9
  %address.local301 = or disjoint i32 %address.local287, 16, !dbg !9
  %44 = shl nuw nsw i32 %address.local301, 1, !dbg !9
  %dma.buffer.offset1281 = or disjoint i32 %44, %41, !dbg !9
  %45 = shl nuw nsw i64 %31, 19, !dbg !9
  %46 = add i64 %35, %45, !dbg !9
  %address.local304 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local279, !dbg !9
  %47 = trunc i64 %46 to i32, !dbg !9
  %.tr = or disjoint i32 %address.local287, %47, !dbg !9
  %48 = shl i32 %.tr, 1, !dbg !9
  %dma.buffer.offset1279 = or disjoint i32 %48, 64, !dbg !9
  %address.local308 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local292, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local280, i64 0, !dbg !9
  %50 = insertelement <4 x ptr addrspace(3)> %49, ptr addrspace(3) %address.local293, i64 1, !dbg !9
  %51 = insertelement <4 x ptr addrspace(3)> %50, ptr addrspace(3) %address.local304, i64 2, !dbg !9
  %52 = insertelement <4 x ptr addrspace(3)> %51, ptr addrspace(3) %address.local308, i64 3, !dbg !9
  %53 = ptrtoint <4 x ptr addrspace(3)> %52 to <4 x i64>, !dbg !9
  %54 = trunc <4 x i64> %53 to <4 x i32>, !dbg !9
  %55 = extractelement <4 x i32> %54, i64 0, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1283, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %58 = extractelement <4 x i32> %54, i64 1, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1281, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %61 = extractelement <4 x i32> %54, i64 2, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1279, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %64 = extractelement <4 x i32> %54, i64 3, !dbg !9
  %65 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %64), !dbg !9
  %.tr1289 = or disjoint i32 %address.local301, %47, !dbg !9
  %66 = shl i32 %.tr1289, 1, !dbg !9
  %dma.buffer.offset1277 = or disjoint i32 %66, 64, !dbg !9
  %67 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1277, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %address.local312 = lshr i32 %address.thread, 2, !dbg !11
  %address.local313 = and i32 %address.local312, 4, !dbg !11
  %address.local314 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local315 = and i32 %address.local314, 384, !dbg !11
  %address.local316 = or disjoint i32 %address.local313, %address.local315, !dbg !11
  %address.local318 = and i32 %36, 120, !dbg !11
  %address.local319 = or disjoint i32 %address.local316, %address.local318, !dbg !11
  %address.local320 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local321 = and i32 %address.local320, 1024, !dbg !11
  %address.local322 = or disjoint i32 %address.local319, %address.local321, !dbg !11
  %address.local323 = zext nneg i32 %address.local322 to i64, !dbg !11
  %address.local324 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local323, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %address.local324, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local324, i64 4096, !dbg !11
  %69 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %70 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1106 = getelementptr i8, ptr addrspace(3) %address.local324, i64 1024, !dbg !11
  %71 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !11
  %72 = fmul <4 x half> %69, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1107 = getelementptr i8, ptr addrspace(3) %address.local324, i64 5120, !dbg !11
  %73 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !11
  %74 = fmul <4 x half> %71, splat (half 1.275630e-01), !dbg !11
  %75 = fmul <4 x half> %73, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1275 = or disjoint i32 %48, 128, !dbg !9
  %76 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1275, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %dma.buffer.offset1273 = or disjoint i32 %66, 128, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1273, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %lds.const.gep1108 = getelementptr i8, ptr addrspace(3) %address.local324, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !11
  %lds.const.gep1109 = getelementptr i8, ptr addrspace(3) %address.local324, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !11
  %lds.const.gep1110 = getelementptr i8, ptr addrspace(3) %address.local324, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep1111 = getelementptr i8, ptr addrspace(3) %address.local324, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %dma.buffer.offset1271 = or disjoint i32 %48, 192, !dbg !9
  %86 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1271, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %dma.buffer.offset1269 = or disjoint i32 %66, 192, !dbg !9
  %87 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1269, <4 x i32> %dma.buffer.rsrc1284) #9, !dbg !9
  %88 = load <4 x half>, ptr addrspace(3) %address.local324, align 8, !dbg !11
  %89 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %90 = load <4 x half>, ptr addrspace(3) %lds.const.gep1106, align 8, !dbg !11
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep1107, align 8, !dbg !11
  %93 = fmul <4 x half> %89, splat (half 1.275630e-01), !dbg !11
  %94 = fmul <4 x half> %90, splat (half 1.275630e-01), !dbg !11
  %95 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %96 = load <4 x half>, ptr addrspace(3) %lds.const.gep1108, align 8, !dbg !11
  %97 = load <4 x half>, ptr addrspace(3) %lds.const.gep1109, align 8, !dbg !11
  %98 = load <4 x half>, ptr addrspace(3) %lds.const.gep1110, align 8, !dbg !11
  %99 = fmul <4 x half> %96, splat (half 1.275630e-01), !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep1111, align 8, !dbg !11
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
  %k0.prime.1893 = and i64 %k0.prime., 16252928, !dbg !14
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !14
  %k0.prime.kdma.source1332 = lshr i64 %k0.prime.kdma.source, 4, !dbg !14
  %k0.prime.kdma.source1333 = shl nuw nsw i64 %k0.prime.kdma.source1332, 9, !dbg !14
  %k0.prime.kdma.source1334 = shl nuw nsw i64 %k0.prime.kdma.source, 7, !dbg !14
  %k0.prime.kdma.source1335 = add nuw nsw i64 %k0.prime.kdma.source1333, %k0.prime.kdma.source1334, !dbg !14
  %k0.prime.kdma.source1336 = lshr i64 %k0.prime.kdma.source, 5, !dbg !14
  %k0.prime.kdma.source1337 = lshr i64 %k0.prime.kdma.source, 2, !dbg !14
  %k0.prime.kdma.source1338 = and i64 %k0.prime.kdma.source1337, 1, !dbg !14
  %k0.prime.kdma.source1339 = and i64 %k0.prime.kdma.source1338, %k0.prime.kdma.source1332, !dbg !14
  %k0.prime.kdma.source1340 = or disjoint i64 %k0.prime.kdma.source1339, %k0.prime.kdma.source1336, !dbg !14
  %110 = shl nuw nsw i64 %k0.prime.kdma.source1340, 10, !dbg !14
  %k0.prime.kdma.source1342 = sub nsw i64 %k0.prime.kdma.source1335, %110, !dbg !14
  %k0.prime.kdma.source1343 = and i64 %k0.prime.kdma.source1342, 16256, !dbg !14
  %k0.prime.kdma.source1344 = lshr i32 %kdma.tid, 4, !dbg !14
  %k0.prime.kdma.source1345 = and i32 %k0.prime.kdma.source1344, 8, !dbg !14
  %k0.prime.kdma.source1346 = zext nneg i32 %k0.prime.kdma.source1345 to i64, !dbg !14
  %k0.prime.kdma.source1347 = or disjoint i64 %k0.prime.kdma.source1343, %k0.prime.kdma.source1346, !dbg !14
  %k0.prime.kdma.source1348 = or disjoint i64 %k0.prime.kdma.source1347, %k0.prime.1893, !dbg !14
  %k0.prime.kdma.source1349 = trunc nuw nsw i64 %k0.prime.kdma.source1348 to i32, !dbg !14
  %k0.prime.kdma.source1350 = shl nuw nsw i32 %k0.prime.kdma.source1349, 1, !dbg !14
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1350, %kdma.intra, !dbg !14
  %k0.prime.1896 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !14
  %k0.prime.1897 = insertelement <2 x i64> poison, i64 %k0.prime.1896, i64 0, !dbg !14
  %k0.prime.1898 = bitcast <2 x i64> %k0.prime.1897 to <4 x i32>, !dbg !14
  %k0.prime.dma.buffer.rsrc1268 = shufflevector <4 x i32> %k0.prime.1898, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !14
  %k0.prime.1899 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1372 = add i32 %56, 512, !dbg !14
  %k0.prime.kdma.source1351 = zext nneg i32 %kdma.owner.hi to i64, !dbg !14
  %k0.prime.kdma.source1352 = lshr i64 %k0.prime.kdma.source1351, 4, !dbg !14
  %k0.prime.kdma.source1353 = shl nuw nsw i64 %k0.prime.kdma.source1352, 9, !dbg !14
  %k0.prime.kdma.source1354 = shl nuw nsw i64 %k0.prime.kdma.source1351, 7, !dbg !14
  %k0.prime.kdma.source1355 = add nuw nsw i64 %k0.prime.kdma.source1353, %k0.prime.kdma.source1354, !dbg !14
  %k0.prime.kdma.source1356 = lshr i64 %k0.prime.kdma.source1351, 5, !dbg !14
  %k0.prime.kdma.source1357 = lshr i64 %k0.prime.kdma.source1351, 2, !dbg !14
  %k0.prime.kdma.source1358 = and i64 %k0.prime.kdma.source1357, 1, !dbg !14
  %k0.prime.kdma.source1359 = and i64 %k0.prime.kdma.source1358, %k0.prime.kdma.source1352, !dbg !14
  %k0.prime.kdma.source1360 = add nuw nsw i64 %k0.prime.kdma.source1359, %k0.prime.kdma.source1356, !dbg !14
  %111 = shl nuw nsw i64 %k0.prime.kdma.source1360, 10, !dbg !14
  %k0.prime.kdma.source1362 = sub nsw i64 %k0.prime.kdma.source1355, %111, !dbg !14
  %k0.prime.kdma.source1363 = and i64 %k0.prime.kdma.source1362, 16256, !dbg !14
  %k0.prime.kdma.source1367 = or disjoint i64 %k0.prime.kdma.source1363, %k0.prime.kdma.source1346, !dbg !14
  %k0.prime.kdma.source1368 = or disjoint i64 %k0.prime.kdma.source1367, %k0.prime.1893, !dbg !14
  %k0.prime.kdma.source1369 = trunc nuw nsw i64 %k0.prime.kdma.source1368 to i32, !dbg !14
  %k0.prime.kdma.source1370 = shl nuw nsw i32 %k0.prime.kdma.source1369, 1, !dbg !14
  %k0.prime.kdma.offset1371 = or disjoint i32 %k0.prime.kdma.source1370, %kdma.intra.hi, !dbg !14
  %k0.prime.1900 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1372, i32 %k0.prime.kdma.offset1371, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1374 = or disjoint i64 %k0.prime.kdma.source, 256, !dbg !14
  %k0.prime.kdma.source1375 = lshr i64 %k0.prime.kdma.source1374, 4, !dbg !14
  %k0.prime.kdma.source1376 = shl nuw nsw i64 %k0.prime.kdma.source1375, 9, !dbg !14
  %k0.prime.kdma.source1378 = add nuw nsw i64 %k0.prime.kdma.source1376, %k0.prime.kdma.source1334, !dbg !14
  %k0.prime.kdma.source1379 = lshr i64 %k0.prime.kdma.source1374, 5, !dbg !14
  %k0.prime.kdma.source1382 = and i64 %k0.prime.kdma.source1375, %k0.prime.kdma.source1338, !dbg !14
  %k0.prime.kdma.source1383 = or disjoint i64 %k0.prime.kdma.source1382, %k0.prime.kdma.source1379, !dbg !14
  %112 = shl nuw nsw i64 %k0.prime.kdma.source1383, 10, !dbg !14
  %k0.prime.kdma.source1385 = sub nsw i64 %k0.prime.kdma.source1378, %112, !dbg !14
  %k0.prime.kdma.source1386 = and i64 %k0.prime.kdma.source1385, 16256, !dbg !14
  %k0.prime.kdma.source1389 = or disjoint i32 %k0.prime.kdma.source1345, 16, !dbg !14
  %k0.prime.kdma.source1390 = zext nneg i32 %k0.prime.kdma.source1389 to i64, !dbg !14
  %k0.prime.kdma.source1391 = or disjoint i64 %k0.prime.kdma.source1386, %k0.prime.kdma.source1390, !dbg !14
  %k0.prime.kdma.source1392 = or disjoint i64 %k0.prime.kdma.source1391, %k0.prime.1893, !dbg !14
  %k0.prime.kdma.source1393 = trunc nuw nsw i64 %k0.prime.kdma.source1392 to i32, !dbg !14
  %k0.prime.kdma.source1394 = shl nuw nsw i32 %k0.prime.kdma.source1393, 1, !dbg !14
  %k0.prime.kdma.offset1395 = add nuw nsw i32 %k0.prime.kdma.source1394, %kdma.intra, !dbg !14
  %k0.prime.1901 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.prime.kdma.offset1395, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1420 = add i32 %59, 512, !dbg !14
  %k0.prime.kdma.source1398 = or disjoint i64 %k0.prime.kdma.source1351, 256, !dbg !14
  %k0.prime.kdma.source1399 = lshr i64 %k0.prime.kdma.source1398, 4, !dbg !14
  %k0.prime.kdma.source1400 = shl nuw nsw i64 %k0.prime.kdma.source1399, 9, !dbg !14
  %k0.prime.kdma.source1402 = add nuw nsw i64 %k0.prime.kdma.source1400, %k0.prime.kdma.source1354, !dbg !14
  %k0.prime.kdma.source1403 = lshr i64 %k0.prime.kdma.source1398, 5, !dbg !14
  %k0.prime.kdma.source1406 = and i64 %k0.prime.kdma.source1399, %k0.prime.kdma.source1358, !dbg !14
  %k0.prime.kdma.source1407 = add nuw nsw i64 %k0.prime.kdma.source1406, %k0.prime.kdma.source1403, !dbg !14
  %113 = shl nuw nsw i64 %k0.prime.kdma.source1407, 10, !dbg !14
  %k0.prime.kdma.source1409 = sub nsw i64 %k0.prime.kdma.source1402, %113, !dbg !14
  %k0.prime.kdma.source1410 = and i64 %k0.prime.kdma.source1409, 16256, !dbg !14
  %k0.prime.kdma.source1415 = or disjoint i64 %k0.prime.kdma.source1410, %k0.prime.kdma.source1390, !dbg !14
  %k0.prime.kdma.source1416 = or disjoint i64 %k0.prime.kdma.source1415, %k0.prime.1893, !dbg !14
  %k0.prime.kdma.source1417 = trunc nuw nsw i64 %k0.prime.kdma.source1416 to i32, !dbg !14
  %k0.prime.kdma.source1418 = shl nuw nsw i32 %k0.prime.kdma.source1417, 1, !dbg !14
  %k0.prime.kdma.offset1419 = add nuw nsw i32 %k0.prime.kdma.source1418, %kdma.intra.hi, !dbg !14
  %k0.prime.1902 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1420, i32 %k0.prime.kdma.offset1419, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %kdma.lds1464 = add i32 %62, 512
  %kdma.lds1512 = add i32 %65, 512
  %114 = ptrtoint ptr addrspace(1) %2 to i64
  %115 = insertelement <2 x i64> poison, i64 %114, i64 0
  %116 = bitcast <2 x i64> %115 to <4 x i32>
  %dma.buffer.rsrc1252 = shufflevector <4 x i32> %116, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %117, !dbg !12

117:                                              ; preds = %4, %k0.join
  %118 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1027, %k0.join ]
  %119 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1023, %k0.join ]
  %120 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1019, %k0.join ]
  %121 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %lazy.base.join, %k0.join ]
  %122 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %790, %k0.join ]
  %123 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1015, %k0.join ]
  %124 = phi i64 [ 0, %4 ], [ %947, %k0.join ]
  %125 = shl nuw nsw i64 %124, 14, !dbg !14
  %126 = add nuw nsw i64 %125, %k0.prime.1893, !dbg !14
  %address.thread375 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !14, !range !10, !frisk.register_lifetimes !5
  %address.local376 = zext nneg i32 %address.thread375 to i64, !dbg !9
  %address.local382 = lshr i64 %address.local376, 2, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %127 = or disjoint i64 %126, 32, !dbg !14
  %kdma.source1438 = or disjoint i64 %127, %k0.prime.kdma.source1347, !dbg !14
  %kdma.source1439 = trunc nuw nsw i64 %kdma.source1438 to i32, !dbg !14
  %kdma.source1440 = shl nuw nsw i32 %kdma.source1439, 1, !dbg !14
  %kdma.offset1441 = or disjoint i32 %kdma.source1440, %kdma.intra, !dbg !14
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1441, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1460 = or disjoint i64 %127, %k0.prime.kdma.source1367, !dbg !14
  %kdma.source1461 = trunc nuw nsw i64 %kdma.source1460 to i32, !dbg !14
  %kdma.source1462 = shl nuw nsw i32 %kdma.source1461, 1, !dbg !14
  %kdma.offset1463 = or disjoint i32 %kdma.source1462, %kdma.intra.hi, !dbg !14
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1464, i32 %kdma.offset1463, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1484 = or disjoint i64 %127, %k0.prime.kdma.source1391, !dbg !14
  %kdma.source1485 = trunc nuw nsw i64 %kdma.source1484 to i32, !dbg !14
  %kdma.source1486 = shl nuw nsw i32 %kdma.source1485, 1, !dbg !14
  %kdma.offset1487 = add nuw nsw i32 %kdma.source1486, %kdma.intra, !dbg !14
  %130 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1487, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1508 = or disjoint i64 %127, %k0.prime.kdma.source1415, !dbg !14
  %kdma.source1509 = trunc nuw nsw i64 %kdma.source1508 to i32, !dbg !14
  %kdma.source1510 = shl nuw nsw i32 %kdma.source1509, 1, !dbg !14
  %kdma.offset1511 = add nuw nsw i32 %kdma.source1510, %kdma.intra.hi, !dbg !14
  %131 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1512, i32 %kdma.offset1511, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %address.local419 = shl nuw nsw i64 %address.local376, 1, !dbg !13
  %address.local420 = and i64 %address.local419, 24, !dbg !13
  %address.local422 = shl i64 %address.local376, 62, !dbg !13
  %address.local423 = ashr exact i64 %address.local422, 57, !dbg !13
  %address.local424 = and i64 %address.local423, -64, !dbg !13
  %address.local427 = and i64 %address.local382, 12, !dbg !11
  %address.local429 = and i64 %address.local376, 32, !dbg !11
  %address.local430 = icmp eq i64 %address.local429, 0, !dbg !11
  %address.local431 = select i1 %address.local430, i64 0, i64 1016, !dbg !11
  %.idx1306 = mul nuw nsw i64 %address.local376, 320, !dbg !13
  %132 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx1306, !dbg !13
  %.idx1307 = mul nsw i64 %address.local382, -1280, !dbg !13
  %133 = getelementptr i8, ptr addrspace(3) %132, i64 %.idx1307, !dbg !13
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local420, !dbg !13
  %135 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local424, !dbg !13
  %136 = getelementptr [2 x i8], ptr addrspace(3) %135, i64 %address.local427, !dbg !13
  %address.local433 = getelementptr [2 x i8], ptr addrspace(3) %136, i64 %address.local431, !dbg !13
  %137 = add nsw i64 %.idx1307, %.idx1306, !dbg !13
  %.idx1823 = shl nuw nsw i64 %address.local420, 1, !dbg !13
  %138 = or disjoint i64 %137, %.idx1823, !dbg !13
  %139 = or disjoint i64 %address.local424, %address.local427, !dbg !13
  %140 = add nsw i64 %139, %address.local431, !dbg !13
  %141 = shl nsw i64 %140, 1, !dbg !13
  %142 = add nsw i64 %138, %141, !dbg !13
  %143 = lshr i64 %142, 6, !dbg !13
  %144 = and i64 %143, 8, !dbg !13
  %145 = xor i64 %144, %142, !dbg !13
  %146 = sub i64 %145, %142, !dbg !13
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local433, i64 %146, !dbg !13
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13
  %lazy.base = extractvalue [2 x <1 x float>] %121, 0, !dbg !13
  %lazy.seed = shufflevector <1 x float> %lazy.base, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %lazy.base1908 = extractvalue [2 x <1 x float>] %121, 1, !dbg !13
  %lazy.seed1909 = shufflevector <1 x float> %lazy.base1908, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !13
  %148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %147, <4 x float> %lazy.seed) #8, !dbg !13
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !13
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13
  %address.local444 = shl nuw nsw i32 %address.thread375, 1, !dbg !13
  %address.local445 = and i32 %address.local444, 24, !dbg !13
  %address.local446 = shl nuw nsw i32 %address.thread375, 8, !dbg !13
  %address.local447 = and i32 %address.local446, 768, !dbg !13
  %address.local448 = or disjoint i32 %address.local445, %address.local447, !dbg !13
  %address.local449 = shl nuw nsw i32 %address.thread375, 5, !dbg !13
  %address.local450 = and i32 %address.local449, 96, !dbg !13
  %address.local451 = or disjoint i32 %address.local448, %address.local450, !dbg !13
  %address.local452 = zext nneg i32 %address.local451 to i64, !dbg !13
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %109) #8, !dbg !13, !frisk.register_lifetimes !5
  %address.local453 = lshr i32 %address.thread375, 1, !dbg !13
  %address.local454 = and i32 %address.local453, 1, !dbg !13
  %address.local455 = zext nneg i32 %address.local454 to i64, !dbg !13
  %address.local456 = add nsw i64 %address.base, %address.local455, !dbg !13
  %address.local457 = shl i64 %address.local456, 6, !dbg !13
  %address.local458 = sub i64 %address.local452, %address.local457, !dbg !13
  %150 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local458, !dbg !13
  %151 = getelementptr [2 x i8], ptr addrspace(3) %150, i64 %address.local427, !dbg !13
  %152 = getelementptr [2 x i8], ptr addrspace(3) %151, i64 %address.local431, !dbg !13
  %address.local462 = getelementptr i8, ptr addrspace(3) %152, i64 64, !dbg !13
  %.idx18261829 = add nuw nsw i64 %address.local431, %address.local427, !dbg !13
  %153 = add i64 %.idx18261829, %address.local458, !dbg !13
  %154 = shl i64 %153, 1, !dbg !13
  %155 = add i64 %154, 64, !dbg !13
  %156 = lshr i64 %155, 6, !dbg !13
  %157 = and i64 %156, 8, !dbg !13
  %158 = xor i64 %157, %155, !dbg !13
  %159 = sub i64 %158, %155, !dbg !13
  %kdma.swizzled1697 = getelementptr i8, ptr addrspace(3) %address.local462, i64 %159, !dbg !13
  %160 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1697, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1701 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 128, !dbg !13
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1701, align 8, !dbg !13, !frisk.k_lds_pair !15
  %kdma.gep1698 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 4096, !dbg !13
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1698, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1702 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 4224, !dbg !13
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1702, align 8, !dbg !13, !frisk.k_lds_pair !16
  %kdma.gep1699 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !13
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1699, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1703 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !13
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1703, align 8, !dbg !13, !frisk.k_lds_pair !17
  %kdma.gep1700 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !13
  %166 = load <4 x half>, ptr addrspace(3) %kdma.gep1700, align 8, !dbg !13, !frisk.k_lds_pair !18
  %kdma.gep1704 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !13
  %167 = load <4 x half>, ptr addrspace(3) %kdma.gep1704, align 8, !dbg !13, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %148) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1705 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 1024, !dbg !13
  %169 = load <4 x half>, ptr addrspace(3) %kdma.gep1705, align 8, !dbg !13, !frisk.k_lds_pair !19
  %kdma.gep1709 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 1152, !dbg !13
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1709, align 8, !dbg !13, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %160, <4 x float> %lazy.seed) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1706 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 5120, !dbg !13
  %172 = load <4 x half>, ptr addrspace(3) %kdma.gep1706, align 8, !dbg !13, !frisk.k_lds_pair !20
  %kdma.gep1710 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 5248, !dbg !13
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1710, align 8, !dbg !13, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %171) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1707 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !13
  %175 = load <4 x half>, ptr addrspace(3) %kdma.gep1707, align 8, !dbg !13, !frisk.k_lds_pair !21
  %kdma.gep1708 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !13
  %176 = load <4 x half>, ptr addrspace(3) %kdma.gep1708, align 8, !dbg !13, !frisk.k_lds_pair !21
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
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %147, <4 x float> %lazy.seed1909) #8, !dbg !13
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %149, <4 x float> %189) #8, !dbg !13
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %160, <4 x float> %lazy.seed1909) #8, !dbg !13
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %162, <4 x float> %191) #8, !dbg !13
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %164, <4 x float> %lazy.seed1909) #8, !dbg !13
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %166, <4 x float> %193) #8, !dbg !13
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %161, <4 x float> %lazy.seed1909) #8, !dbg !13
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %163, <4 x float> %195) #8, !dbg !13
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %165, <4 x float> %lazy.seed1909) #8, !dbg !13
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %167, <4 x float> %197) #8, !dbg !13
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %169, <4 x float> %lazy.seed1909) #8, !dbg !13
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %172, <4 x float> %199) #8, !dbg !13
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %175, <4 x float> %lazy.seed1909) #8, !dbg !13
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %176, <4 x float> %201) #8, !dbg !13
  %203 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %74, <4 x half> %170, <4 x float> %lazy.seed1909) #8, !dbg !13
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %75, <4 x half> %173, <4 x float> %203) #8, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !14
  fence syncscope("workgroup") release, !dbg !14
  tail call void @llvm.amdgcn.s.barrier(), !dbg !14
  fence syncscope("workgroup") acquire, !dbg !14
  %205 = or disjoint i64 %126, 64, !dbg !14
  %kdma.source1530 = or disjoint i64 %205, %k0.prime.kdma.source1347, !dbg !14
  %kdma.source1531 = trunc nuw nsw i64 %kdma.source1530 to i32, !dbg !14
  %kdma.source1532 = shl nuw nsw i32 %kdma.source1531, 1, !dbg !14
  %kdma.offset1533 = or disjoint i32 %kdma.source1532, %kdma.intra, !dbg !14
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1552 = or disjoint i64 %205, %k0.prime.kdma.source1367, !dbg !14
  %kdma.source1553 = trunc nuw nsw i64 %kdma.source1552 to i32, !dbg !14
  %kdma.source1554 = shl nuw nsw i32 %kdma.source1553, 1, !dbg !14
  %kdma.offset1555 = or disjoint i32 %kdma.source1554, %kdma.intra.hi, !dbg !14
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1372, i32 %kdma.offset1555, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1576 = or disjoint i64 %205, %k0.prime.kdma.source1391, !dbg !14
  %kdma.source1577 = trunc nuw nsw i64 %kdma.source1576 to i32, !dbg !14
  %kdma.source1578 = shl nuw nsw i32 %kdma.source1577, 1, !dbg !14
  %kdma.offset1579 = add nuw nsw i32 %kdma.source1578, %kdma.intra, !dbg !14
  %208 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1579, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1600 = or disjoint i64 %205, %k0.prime.kdma.source1415, !dbg !14
  %kdma.source1601 = trunc nuw nsw i64 %kdma.source1600 to i32, !dbg !14
  %kdma.source1602 = shl nuw nsw i32 %kdma.source1601, 1, !dbg !14
  %kdma.offset1603 = add nuw nsw i32 %kdma.source1602, %kdma.intra.hi, !dbg !14
  %209 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1420, i32 %kdma.offset1603, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.gep1727 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !13
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1731 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !13
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !13, !frisk.k_lds_pair !22
  %kdma.gep1728 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !13
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1732 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !13
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !13, !frisk.k_lds_pair !23
  %kdma.gep1729 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 8192, !dbg !13
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1733 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 8320, !dbg !13
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !13, !frisk.k_lds_pair !24
  %kdma.gep1730 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 12288, !dbg !13
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1734 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 12416, !dbg !13
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !13, !frisk.k_lds_pair !25
  %kdma.gep1735 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !13
  %218 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !13, !frisk.k_lds_pair !26
  %kdma.gep1739 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !13
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !13, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %210, <4 x float> %168) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1736 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !13
  %221 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !13, !frisk.k_lds_pair !27
  %kdma.gep1740 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !13
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !13, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %212, <4 x float> %220) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1737 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 9216, !dbg !13
  %224 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !13, !frisk.k_lds_pair !28
  %kdma.gep1741 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 9344, !dbg !13
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !13, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %214, <4 x float> %174) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %kdma.gep1738 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 13312, !dbg !13
  %227 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !13, !frisk.k_lds_pair !29
  %kdma.gep1742 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1697, i64 13440, !dbg !13
  %228 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !13, !frisk.k_lds_pair !29
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
  %kdma.source1622 = or disjoint i64 %258, %k0.prime.kdma.source1347, !dbg !14
  %kdma.source1623 = trunc nuw nsw i64 %kdma.source1622 to i32, !dbg !14
  %kdma.source1624 = shl nuw nsw i32 %kdma.source1623, 1, !dbg !14
  %kdma.offset1625 = or disjoint i32 %kdma.source1624, %kdma.intra, !dbg !14
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1625, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1644 = or disjoint i64 %258, %k0.prime.kdma.source1367, !dbg !14
  %kdma.source1645 = trunc nuw nsw i64 %kdma.source1644 to i32, !dbg !14
  %kdma.source1646 = shl nuw nsw i32 %kdma.source1645, 1, !dbg !14
  %kdma.offset1647 = or disjoint i32 %kdma.source1646, %kdma.intra.hi, !dbg !14
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1464, i32 %kdma.offset1647, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1668 = or disjoint i64 %258, %k0.prime.kdma.source1391, !dbg !14
  %kdma.source1669 = trunc nuw nsw i64 %kdma.source1668 to i32, !dbg !14
  %kdma.source1670 = shl nuw nsw i32 %kdma.source1669, 1, !dbg !14
  %kdma.offset1671 = add nuw nsw i32 %kdma.source1670, %kdma.intra, !dbg !14
  %261 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %kdma.offset1671, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %kdma.source1692 = or disjoint i64 %258, %k0.prime.kdma.source1415, !dbg !14
  %kdma.source1693 = trunc nuw nsw i64 %kdma.source1692 to i32, !dbg !14
  %kdma.source1694 = shl nuw nsw i32 %kdma.source1693, 1, !dbg !14
  %kdma.offset1695 = add nuw nsw i32 %kdma.source1694, %kdma.intra.hi, !dbg !14
  %262 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1512, i32 %kdma.offset1695, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14
  %263 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !13, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1699, align 8, !dbg !13, !frisk.k_lds_pair !30
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !13, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1700, align 8, !dbg !13, !frisk.k_lds_pair !31
  %267 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1697, align 8, !dbg !13, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1701, align 8, !dbg !13, !frisk.k_lds_pair !32
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1698, align 8, !dbg !13, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1702, align 8, !dbg !13, !frisk.k_lds_pair !33
  %271 = load <4 x half>, ptr addrspace(3) %kdma.gep1703, align 8, !dbg !13, !frisk.k_lds_pair !34
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1707, align 8, !dbg !13, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %263, <4 x float> %223) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %274 = load <4 x half>, ptr addrspace(3) %kdma.gep1704, align 8, !dbg !13, !frisk.k_lds_pair !35
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1708, align 8, !dbg !13, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %93, <4 x half> %265, <4 x float> %273) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %277 = load <4 x half>, ptr addrspace(3) %kdma.gep1705, align 8, !dbg !13, !frisk.k_lds_pair !36
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1709, align 8, !dbg !13, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %267, <4 x float> %229) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_lds_prefetch !5
  %280 = load <4 x half>, ptr addrspace(3) %kdma.gep1706, align 8, !dbg !13, !frisk.k_lds_pair !37
  %281 = load <4 x half>, ptr addrspace(3) %kdma.gep1710, align 8, !dbg !13, !frisk.k_lds_pair !37
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
  %address.local791 = shl nuw nsw i64 %address.local376, 5, !dbg !38
  %address.local792 = and i64 %address.local791, 8064, !dbg !38
  %address.local793 = shl nuw nsw i64 %address.local376, 3, !dbg !9
  %address.local794 = shl nuw nsw i64 %address.local382, 5, !dbg !38
  %address.local795 = sub nsw i64 %address.local793, %address.local794, !dbg !38
  %address.local796 = add nsw i64 %address.local795, %address.local792, !dbg !38
  %.idx13161317 = add nsw i64 %address.local796, %126, !dbg !38
  %.idx13161317.tr = trunc nsw i64 %.idx13161317 to i32, !dbg !38
  %dma.buffer.offset1251 = shl nsw i32 %.idx13161317.tr, 1, !dbg !38
  %311 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1251, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %address.local798 = shl nuw nsw i32 %address.thread375, 3, !dbg !38
  %address.local799 = and i32 %address.local798, 24, !dbg !38
  %address.local801 = and i32 %address.local449, 8064, !dbg !38
  %address.local802 = or disjoint i32 %address.local801, %address.local799, !dbg !38
  %address.local803 = or disjoint i32 %address.local802, 8192, !dbg !38
  %312 = trunc nuw nsw i64 %126 to i32, !dbg !38
  %.idx13181319.tr = or disjoint i32 %address.local803, %312, !dbg !38
  %dma.buffer.offset1249 = shl nuw nsw i32 %.idx13181319.tr, 1, !dbg !38
  %313 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1249, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1727, align 8, !dbg !13, !frisk.k_tail_pair !15
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1731, align 8, !dbg !13, !frisk.k_tail_pair !15
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1728, align 8, !dbg !13, !frisk.k_tail_pair !16
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1732, align 8, !dbg !13, !frisk.k_tail_pair !16
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1729, align 8, !dbg !13, !frisk.k_tail_pair !17
  %319 = load <4 x half>, ptr addrspace(3) %kdma.gep1733, align 8, !dbg !13, !frisk.k_tail_pair !17
  %320 = load <4 x half>, ptr addrspace(3) %kdma.gep1730, align 8, !dbg !13, !frisk.k_tail_pair !18
  %321 = load <4 x half>, ptr addrspace(3) %kdma.gep1734, align 8, !dbg !13, !frisk.k_tail_pair !18
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
  %331 = load <4 x half>, ptr addrspace(3) %kdma.gep1735, align 8, !dbg !13, !frisk.k_tail_pair !19
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1739, align 8, !dbg !13, !frisk.k_tail_pair !19
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
  %342 = load <4 x half>, ptr addrspace(3) %kdma.gep1736, align 8, !dbg !13, !frisk.k_tail_pair !20
  %343 = load <4 x half>, ptr addrspace(3) %kdma.gep1740, align 8, !dbg !13, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %101, <4 x half> %317, <4 x float> %333) #8, !dbg !13
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !13, !frisk.k_tail_prefetch !5
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1737, align 8, !dbg !13, !frisk.k_tail_pair !21
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1741, align 8, !dbg !13, !frisk.k_tail_pair !21
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
  %356 = load <4 x half>, ptr addrspace(3) %kdma.gep1738, align 8, !dbg !13, !frisk.k_tail_pair !22
  %357 = load <4 x half>, ptr addrspace(3) %kdma.gep1742, align 8, !dbg !13, !frisk.k_tail_pair !22
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
  %lazy.positive1910 = tail call float @llvm.maxnum.f32(float %505, float 0.000000e+00), !dbg !39
  %lazy.delta1911 = select i1 %lazy.first, float %505, float %lazy.positive1910, !dbg !39
  %lazy.negated1912 = fsub float 0.000000e+00, %lazy.delta1911, !dbg !39
  %lazy.exp1913 = tail call float @llvm.amdgcn.exp2.f32(float %lazy.negated1912), !dbg !39
  %lazy.alpha1914 = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp1913, !dbg !39
  %lazy.delta.v.lane = insertelement <1 x float> poison, float %lazy.delta, i64 0, !dbg !39
  %lazy.delta.v = shufflevector <1 x float> %lazy.delta.v.lane, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted = fsub <4 x float> %323, %lazy.delta.v, !dbg !39
  %lazy.shifted1917 = fsub <4 x float> %330, %lazy.delta.v, !dbg !39
  %lazy.shifted1920 = fsub <4 x float> %344, %lazy.delta.v, !dbg !39
  %lazy.shifted1923 = fsub <4 x float> %358, %lazy.delta.v, !dbg !39
  %lazy.shifted1926 = fsub <4 x float> %368, %lazy.delta.v, !dbg !39
  %lazy.shifted1929 = fsub <4 x float> %378, %lazy.delta.v, !dbg !39
  %lazy.shifted1932 = fsub <4 x float> %388, %lazy.delta.v, !dbg !39
  %lazy.shifted1935 = fsub <4 x float> %398, %lazy.delta.v, !dbg !39
  %lazy.delta.v.lane1936 = insertelement <1 x float> poison, float %lazy.delta1911, i64 0, !dbg !39
  %lazy.delta.v1937 = shufflevector <1 x float> %lazy.delta.v.lane1936, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !39
  %lazy.shifted1938 = fsub <4 x float> %411, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1941 = fsub <4 x float> %417, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1944 = fsub <4 x float> %423, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1947 = fsub <4 x float> %429, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1950 = fsub <4 x float> %435, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1953 = fsub <4 x float> %441, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1956 = fsub <4 x float> %447, %lazy.delta.v1937, !dbg !39
  %lazy.shifted1959 = fsub <4 x float> %453, %lazy.delta.v1937, !dbg !39
  %lazy.state = extractvalue [2 x <1 x float>] %122, 0, !dbg !39
  %lazy.alpha.v.lane = insertelement <1 x float> poison, float %lazy.alpha, i64 0, !dbg !39
  %lazy.scaled = fmul <1 x float> %lazy.state, %lazy.alpha.v.lane, !dbg !39
  %lazy.state.next = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.scaled, 0, !dbg !39
  %lazy.state1960 = extractvalue [2 x <1 x float>] %122, 1, !dbg !39
  %lazy.alpha.v.lane1961 = insertelement <1 x float> poison, float %lazy.alpha1914, i64 0, !dbg !39
  %lazy.scaled1962 = fmul <1 x float> %lazy.state1960, %lazy.alpha.v.lane1961, !dbg !39
  %lazy.state.next1963 = insertvalue [2 x <1 x float>] %lazy.state.next, <1 x float> %lazy.scaled1962, 1, !dbg !39
  %lazy.state1964 = extractvalue [2 x <8 x float>] %118, 0, !dbg !39
  %lazy.alpha.v = shufflevector <1 x float> %lazy.alpha.v.lane, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled1966 = fmul <8 x float> %lazy.state1964, %lazy.alpha.v, !dbg !39
  %lazy.state.next1967 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1966, 0, !dbg !39
  %lazy.state1968 = extractvalue [2 x <8 x float>] %118, 1, !dbg !39
  %lazy.alpha.v1970 = shufflevector <1 x float> %lazy.alpha.v.lane1961, <1 x float> poison, <8 x i32> zeroinitializer, !dbg !39
  %lazy.scaled1971 = fmul <8 x float> %lazy.state1968, %lazy.alpha.v1970, !dbg !39
  %lazy.state.next1972 = insertvalue [2 x <8 x float>] %lazy.state.next1967, <8 x float> %lazy.scaled1971, 1, !dbg !39
  %lazy.state1973 = extractvalue [2 x <8 x float>] %119, 0, !dbg !39
  %lazy.scaled1976 = fmul <8 x float> %lazy.state1973, %lazy.alpha.v, !dbg !39
  %lazy.state.next1977 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1976, 0, !dbg !39
  %lazy.state1978 = extractvalue [2 x <8 x float>] %119, 1, !dbg !39
  %lazy.scaled1981 = fmul <8 x float> %lazy.state1978, %lazy.alpha.v1970, !dbg !39
  %lazy.state.next1982 = insertvalue [2 x <8 x float>] %lazy.state.next1977, <8 x float> %lazy.scaled1981, 1, !dbg !39
  %lazy.state1983 = extractvalue [2 x <8 x float>] %120, 0, !dbg !39
  %lazy.scaled1986 = fmul <8 x float> %lazy.state1983, %lazy.alpha.v, !dbg !39
  %lazy.state.next1987 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1986, 0, !dbg !39
  %lazy.state1988 = extractvalue [2 x <8 x float>] %120, 1, !dbg !39
  %lazy.scaled1991 = fmul <8 x float> %lazy.state1988, %lazy.alpha.v1970, !dbg !39
  %lazy.state.next1992 = insertvalue [2 x <8 x float>] %lazy.state.next1987, <8 x float> %lazy.scaled1991, 1, !dbg !39
  %lazy.state1993 = extractvalue [2 x <8 x float>] %123, 0, !dbg !39
  %lazy.scaled1996 = fmul <8 x float> %lazy.state1993, %lazy.alpha.v, !dbg !39
  %lazy.state.next1997 = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.scaled1996, 0, !dbg !39
  %lazy.state1998 = extractvalue [2 x <8 x float>] %123, 1, !dbg !39
  %lazy.scaled2001 = fmul <8 x float> %lazy.state1998, %lazy.alpha.v1970, !dbg !39
  %lazy.state.next2002 = insertvalue [2 x <8 x float>] %lazy.state.next1997, <8 x float> %lazy.scaled2001, 1, !dbg !39
  %lazy.base.next = fsub <1 x float> %lazy.base, %lazy.delta.v.lane, !dbg !39
  %lazy.base.state = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.base.next, 0, !dbg !39
  %lazy.base.next2005 = fsub <1 x float> %lazy.base1908, %lazy.delta.v.lane1936, !dbg !39
  %lazy.base.state2006 = insertvalue [2 x <1 x float>] %lazy.base.state, <1 x float> %lazy.base.next2005, 1, !dbg !39
  %.pre = extractelement <4 x float> %lazy.shifted, i64 0, !dbg !40
  %.pre26 = extractelement <4 x float> %lazy.shifted, i64 1, !dbg !40
  %.pre28 = extractelement <4 x float> %lazy.shifted, i64 2, !dbg !40
  %.pre30 = extractelement <4 x float> %lazy.shifted, i64 3, !dbg !40
  %.pre32 = extractelement <4 x float> %lazy.shifted1938, i64 0, !dbg !40
  %.pre34 = extractelement <4 x float> %lazy.shifted1938, i64 1, !dbg !40
  %.pre36 = extractelement <4 x float> %lazy.shifted1938, i64 2, !dbg !40
  %.pre38 = extractelement <4 x float> %lazy.shifted1938, i64 3, !dbg !40
  %.pre40 = extractelement <4 x float> %lazy.shifted1917, i64 0, !dbg !40
  %.pre42 = extractelement <4 x float> %lazy.shifted1917, i64 1, !dbg !40
  %.pre44 = extractelement <4 x float> %lazy.shifted1917, i64 2, !dbg !40
  %.pre46 = extractelement <4 x float> %lazy.shifted1917, i64 3, !dbg !40
  %.pre48 = extractelement <4 x float> %lazy.shifted1941, i64 0, !dbg !40
  %.pre50 = extractelement <4 x float> %lazy.shifted1941, i64 1, !dbg !40
  %.pre52 = extractelement <4 x float> %lazy.shifted1941, i64 2, !dbg !40
  %.pre54 = extractelement <4 x float> %lazy.shifted1941, i64 3, !dbg !40
  %.pre56 = extractelement <4 x float> %lazy.shifted1920, i64 0, !dbg !40
  %.pre58 = extractelement <4 x float> %lazy.shifted1920, i64 1, !dbg !40
  %.pre60 = extractelement <4 x float> %lazy.shifted1920, i64 2, !dbg !40
  %.pre62 = extractelement <4 x float> %lazy.shifted1920, i64 3, !dbg !40
  %.pre64 = extractelement <4 x float> %lazy.shifted1944, i64 0, !dbg !40
  %.pre66 = extractelement <4 x float> %lazy.shifted1944, i64 1, !dbg !40
  %.pre68 = extractelement <4 x float> %lazy.shifted1944, i64 2, !dbg !40
  %.pre70 = extractelement <4 x float> %lazy.shifted1944, i64 3, !dbg !40
  %.pre72 = extractelement <4 x float> %lazy.shifted1923, i64 0, !dbg !40
  %.pre74 = extractelement <4 x float> %lazy.shifted1923, i64 1, !dbg !40
  %.pre76 = extractelement <4 x float> %lazy.shifted1923, i64 2, !dbg !40
  %.pre78 = extractelement <4 x float> %lazy.shifted1923, i64 3, !dbg !40
  %.pre80 = extractelement <4 x float> %lazy.shifted1947, i64 0, !dbg !40
  %.pre82 = extractelement <4 x float> %lazy.shifted1947, i64 1, !dbg !40
  %.pre84 = extractelement <4 x float> %lazy.shifted1947, i64 2, !dbg !40
  %.pre86 = extractelement <4 x float> %lazy.shifted1947, i64 3, !dbg !40
  %.pre88 = extractelement <4 x float> %lazy.shifted1926, i64 0, !dbg !40
  %.pre90 = extractelement <4 x float> %lazy.shifted1926, i64 1, !dbg !40
  %.pre92 = extractelement <4 x float> %lazy.shifted1926, i64 2, !dbg !40
  %.pre94 = extractelement <4 x float> %lazy.shifted1926, i64 3, !dbg !40
  %.pre96 = extractelement <4 x float> %lazy.shifted1950, i64 0, !dbg !40
  %.pre98 = extractelement <4 x float> %lazy.shifted1950, i64 1, !dbg !40
  %.pre100 = extractelement <4 x float> %lazy.shifted1950, i64 2, !dbg !40
  %.pre102 = extractelement <4 x float> %lazy.shifted1950, i64 3, !dbg !40
  %.pre104 = extractelement <4 x float> %lazy.shifted1929, i64 0, !dbg !40
  %.pre106 = extractelement <4 x float> %lazy.shifted1929, i64 1, !dbg !40
  %.pre108 = extractelement <4 x float> %lazy.shifted1929, i64 2, !dbg !40
  %.pre110 = extractelement <4 x float> %lazy.shifted1929, i64 3, !dbg !40
  %.pre112 = extractelement <4 x float> %lazy.shifted1953, i64 0, !dbg !40
  %.pre114 = extractelement <4 x float> %lazy.shifted1953, i64 1, !dbg !40
  %.pre116 = extractelement <4 x float> %lazy.shifted1953, i64 2, !dbg !40
  %.pre118 = extractelement <4 x float> %lazy.shifted1953, i64 3, !dbg !40
  %.pre120 = extractelement <4 x float> %lazy.shifted1932, i64 0, !dbg !40
  %.pre122 = extractelement <4 x float> %lazy.shifted1932, i64 1, !dbg !40
  %.pre124 = extractelement <4 x float> %lazy.shifted1932, i64 2, !dbg !40
  %.pre126 = extractelement <4 x float> %lazy.shifted1932, i64 3, !dbg !40
  %.pre128 = extractelement <4 x float> %lazy.shifted1956, i64 0, !dbg !40
  %.pre130 = extractelement <4 x float> %lazy.shifted1956, i64 1, !dbg !40
  %.pre132 = extractelement <4 x float> %lazy.shifted1956, i64 2, !dbg !40
  %.pre134 = extractelement <4 x float> %lazy.shifted1956, i64 3, !dbg !40
  %.pre136 = extractelement <4 x float> %lazy.shifted1935, i64 0, !dbg !40
  %.pre138 = extractelement <4 x float> %lazy.shifted1935, i64 1, !dbg !40
  %.pre140 = extractelement <4 x float> %lazy.shifted1935, i64 2, !dbg !40
  %.pre142 = extractelement <4 x float> %lazy.shifted1935, i64 3, !dbg !40
  %.pre144 = extractelement <4 x float> %lazy.shifted1959, i64 0, !dbg !40
  %.pre146 = extractelement <4 x float> %lazy.shifted1959, i64 1, !dbg !40
  %.pre148 = extractelement <4 x float> %lazy.shifted1959, i64 2, !dbg !40
  %.pre150 = extractelement <4 x float> %lazy.shifted1959, i64 3, !dbg !40
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
  %lazy.state2022 = phi [2 x <1 x float>] [ %lazy.state.next1963, %lazy.slow ], [ %122, %117 ], !dbg !39
  %lazy.state2023 = phi [2 x <8 x float>] [ %lazy.state.next1972, %lazy.slow ], [ %118, %117 ], !dbg !39
  %lazy.state2024 = phi [2 x <8 x float>] [ %lazy.state.next1982, %lazy.slow ], [ %119, %117 ], !dbg !39
  %lazy.state2025 = phi [2 x <8 x float>] [ %lazy.state.next1992, %lazy.slow ], [ %120, %117 ], !dbg !39
  %lazy.state2026 = phi [2 x <8 x float>] [ %lazy.state.next2002, %lazy.slow ], [ %123, %117 ], !dbg !39
  %lazy.base.join = phi [2 x <1 x float>] [ %lazy.base.state2006, %lazy.slow ], [ %121, %117 ], !dbg !39
  %506 = extractvalue [2 x <1 x float>] %lazy.state2022, 0, !dbg !39
  %507 = extractelement <1 x float> %506, i64 0, !dbg !39
  %508 = extractvalue [2 x <1 x float>] %lazy.state2022, 1, !dbg !39
  %509 = extractelement <1 x float> %508, i64 0, !dbg !39
  %510 = extractvalue [2 x <8 x float>] %lazy.state2026, 0, !dbg !41
  %511 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %512 = shufflevector <8 x float> %510, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %513 = extractvalue [2 x <8 x float>] %lazy.state2026, 1, !dbg !41
  %514 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %515 = shufflevector <8 x float> %513, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %.idx13201321 = add nsw i64 %address.local796, %127, !dbg !38
  %.idx13201321.tr = trunc nsw i64 %.idx13201321 to i32, !dbg !38
  %dma.buffer.offset1247 = shl nsw i32 %.idx13201321.tr, 1, !dbg !38
  %516 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1247, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %517 = trunc nuw nsw i64 %127 to i32, !dbg !38
  %.idx13221323.tr = or disjoint i32 %address.local803, %517, !dbg !38
  %dma.buffer.offset1245 = shl nuw nsw i32 %.idx13221323.tr, 1, !dbg !38
  %518 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset1245, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %address.local809 = and i32 %address.local798, 120, !dbg !42
  %address.local811 = and i32 %address.local449, 1536, !dbg !42
  %address.local812 = or disjoint i32 %address.local809, %address.local811, !dbg !42
  %address.local813 = zext nneg i32 %address.local812 to i64, !dbg !42
  %address.local814 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local813, !dbg !42
  %519 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #11, !dbg !42
  %520 = bitcast <4 x i32> %519 to <8 x half>, !dbg !42
  %address.local820 = or disjoint i32 %address.local812, 128, !dbg !42
  %address.local821 = zext nneg i32 %address.local820 to i64, !dbg !42
  %address.local822 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local821, !dbg !42
  %521 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #11, !dbg !42
  %522 = bitcast <4 x i32> %521 to <8 x half>, !dbg !42
  %address.local828 = or disjoint i32 %address.local812, 256, !dbg !42
  %address.local829 = zext nneg i32 %address.local828 to i64, !dbg !42
  %address.local830 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local829, !dbg !42
  %523 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #11, !dbg !42
  %524 = bitcast <4 x i32> %523 to <8 x half>, !dbg !42
  %address.local836 = or disjoint i32 %address.local812, 384, !dbg !42
  %address.local837 = zext nneg i32 %address.local836 to i64, !dbg !42
  %address.local838 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local837, !dbg !42
  %525 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local838, i32 0) #11, !dbg !42
  %526 = bitcast <4 x i32> %525 to <8 x half>, !dbg !42
  %address.local844 = or disjoint i32 %address.local812, 2048, !dbg !42
  %address.local845 = zext nneg i32 %address.local844 to i64, !dbg !42
  %address.local846 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local845, !dbg !42
  %527 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local846, i32 0) #11, !dbg !42
  %528 = bitcast <4 x i32> %527 to <8 x half>, !dbg !42
  %address.local852 = or disjoint i32 %address.local812, 2176, !dbg !42
  %address.local853 = zext nneg i32 %address.local852 to i64, !dbg !42
  %address.local854 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local853, !dbg !42
  %529 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local854, i32 0) #11, !dbg !42
  %530 = bitcast <4 x i32> %529 to <8 x half>, !dbg !42
  %address.local860 = or disjoint i32 %address.local812, 2304, !dbg !42
  %address.local861 = zext nneg i32 %address.local860 to i64, !dbg !42
  %address.local862 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local861, !dbg !42
  %531 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local862, i32 0) #11, !dbg !42
  %532 = bitcast <4 x i32> %531 to <8 x half>, !dbg !42
  %address.local868 = or disjoint i32 %address.local812, 2432, !dbg !42
  %address.local869 = zext nneg i32 %address.local868 to i64, !dbg !42
  %address.local870 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local869, !dbg !42
  %533 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local870, i32 0) #11, !dbg !42
  %534 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %535 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi), !dbg !40
  %536 = insertelement <4 x float> poison, float %535, i64 0, !dbg !43
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi27), !dbg !40
  %538 = insertelement <4 x float> %536, float %537, i64 1, !dbg !43
  %539 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi29), !dbg !40
  %540 = insertelement <4 x float> %538, float %539, i64 2, !dbg !43
  %541 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi31), !dbg !40
  %542 = insertelement <4 x float> %540, float %541, i64 3, !dbg !43
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %542) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f0 = extractelement <4 x float> %math.result, i64 0, !dbg !43
  %.cvtpack.f1 = extractelement <4 x float> %math.result, i64 1, !dbg !43
  %.cvtpack.f2 = extractelement <4 x float> %math.result, i64 2, !dbg !43
  %.cvtpack.f3 = extractelement <4 x float> %math.result, i64 3, !dbg !43
  %.cvtpack.p0 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f0, float %.cvtpack.f1) #12, !dbg !43
  %.cvtpack.p1 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f2, float %.cvtpack.f3) #12, !dbg !43
  %.cvtpack.v0 = insertelement <2 x i32> poison, i32 %.cvtpack.p0, i64 0, !dbg !43
  %.cvtpack.v1 = insertelement <2 x i32> %.cvtpack.v0, i32 %.cvtpack.p1, i64 1, !dbg !43
  %543 = bitcast <2 x i32> %.cvtpack.v1 to <4 x half>, !dbg !43
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %543) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %544 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %534, <4 x float> %511) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %545 = shufflevector <8 x half> %520, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %546 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %545, <4 x float> %512) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %547 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi33), !dbg !40
  %548 = insertelement <4 x float> poison, float %547, i64 0, !dbg !43
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi35), !dbg !40
  %550 = insertelement <4 x float> %548, float %549, i64 1, !dbg !43
  %551 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi37), !dbg !40
  %552 = insertelement <4 x float> %550, float %551, i64 2, !dbg !43
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi39), !dbg !40
  %554 = insertelement <4 x float> %552, float %553, i64 3, !dbg !43
  %math.result262 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %554) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02027 = extractelement <4 x float> %math.result262, i64 0, !dbg !43
  %.cvtpack.f12028 = extractelement <4 x float> %math.result262, i64 1, !dbg !43
  %.cvtpack.f22029 = extractelement <4 x float> %math.result262, i64 2, !dbg !43
  %.cvtpack.f32030 = extractelement <4 x float> %math.result262, i64 3, !dbg !43
  %.cvtpack.p02031 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02027, float %.cvtpack.f12028) #12, !dbg !43
  %.cvtpack.p12032 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22029, float %.cvtpack.f32030) #12, !dbg !43
  %.cvtpack.v02033 = insertelement <2 x i32> poison, i32 %.cvtpack.p02031, i64 0, !dbg !43
  %.cvtpack.v12034 = insertelement <2 x i32> %.cvtpack.v02033, i32 %.cvtpack.p12032, i64 1, !dbg !43
  %555 = bitcast <2 x i32> %.cvtpack.v12034 to <4 x half>, !dbg !43
  %math.packed263 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %555) #8, !dbg !40, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %556 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %534, <4 x float> %514) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %545, <4 x float> %515) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %558 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %559 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi41), !dbg !40
  %560 = fadd float %535, %559, !dbg !44
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %560) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %561 = insertelement <4 x float> poison, float %559, i64 0, !dbg !43
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi43), !dbg !40
  %563 = fadd float %537, %562, !dbg !44
  %k0.sum.anchor1830 = tail call float asm sideeffect "", "=v,0"(float %563) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %564 = insertelement <4 x float> %561, float %562, i64 1, !dbg !43
  %565 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi45), !dbg !40
  %566 = fadd float %539, %565, !dbg !44
  %k0.sum.anchor1831 = tail call float asm sideeffect "", "=v,0"(float %566) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %567 = insertelement <4 x float> %564, float %565, i64 2, !dbg !43
  %568 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi47), !dbg !40
  %569 = fadd float %541, %568, !dbg !44
  %k0.sum.anchor1832 = tail call float asm sideeffect "", "=v,0"(float %569) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %570 = insertelement <4 x float> %567, float %568, i64 3, !dbg !43
  %math.result248 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %570) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02035 = extractelement <4 x float> %math.result248, i64 0, !dbg !43
  %.cvtpack.f12036 = extractelement <4 x float> %math.result248, i64 1, !dbg !43
  %.cvtpack.f22037 = extractelement <4 x float> %math.result248, i64 2, !dbg !43
  %.cvtpack.f32038 = extractelement <4 x float> %math.result248, i64 3, !dbg !43
  %.cvtpack.p02039 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02035, float %.cvtpack.f12036) #12, !dbg !43
  %.cvtpack.p12040 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22037, float %.cvtpack.f32038) #12, !dbg !43
  %.cvtpack.v02041 = insertelement <2 x i32> poison, i32 %.cvtpack.p02039, i64 0, !dbg !43
  %.cvtpack.v12042 = insertelement <2 x i32> %.cvtpack.v02041, i32 %.cvtpack.p12040, i64 1, !dbg !43
  %571 = bitcast <2 x i32> %.cvtpack.v12042 to <4 x half>, !dbg !43
  %math.packed249 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %571) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %558, <4 x float> %544) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %573 = shufflevector <8 x half> %522, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %573, <4 x float> %546) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %575 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi49), !dbg !40
  %576 = fadd float %547, %575, !dbg !44
  %k0.sum.anchor1833 = tail call float asm sideeffect "", "=v,0"(float %576) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %577 = insertelement <4 x float> poison, float %575, i64 0, !dbg !43
  %578 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi51), !dbg !40
  %579 = fadd float %549, %578, !dbg !44
  %k0.sum.anchor1834 = tail call float asm sideeffect "", "=v,0"(float %579) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %580 = insertelement <4 x float> %577, float %578, i64 1, !dbg !43
  %581 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi53), !dbg !40
  %582 = fadd float %551, %581, !dbg !44
  %k0.sum.anchor1835 = tail call float asm sideeffect "", "=v,0"(float %582) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %583 = insertelement <4 x float> %580, float %581, i64 2, !dbg !43
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi55), !dbg !40
  %585 = fadd float %553, %584, !dbg !44
  %k0.sum.anchor1836 = tail call float asm sideeffect "", "=v,0"(float %585) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %586 = insertelement <4 x float> %583, float %584, i64 3, !dbg !43
  %math.result264 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %586) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02043 = extractelement <4 x float> %math.result264, i64 0, !dbg !43
  %.cvtpack.f12044 = extractelement <4 x float> %math.result264, i64 1, !dbg !43
  %.cvtpack.f22045 = extractelement <4 x float> %math.result264, i64 2, !dbg !43
  %.cvtpack.f32046 = extractelement <4 x float> %math.result264, i64 3, !dbg !43
  %.cvtpack.p02047 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02043, float %.cvtpack.f12044) #12, !dbg !43
  %.cvtpack.p12048 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22045, float %.cvtpack.f32046) #12, !dbg !43
  %.cvtpack.v02049 = insertelement <2 x i32> poison, i32 %.cvtpack.p02047, i64 0, !dbg !43
  %.cvtpack.v12050 = insertelement <2 x i32> %.cvtpack.v02049, i32 %.cvtpack.p12048, i64 1, !dbg !43
  %587 = bitcast <2 x i32> %.cvtpack.v12050 to <4 x half>, !dbg !43
  %math.packed265 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %587) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %588 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %558, <4 x float> %556) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %573, <4 x float> %557) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %590 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %591 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi57), !dbg !40
  %592 = fadd float %591, %k0.sum.anchor, !dbg !44
  %k0.sum.anchor1837 = tail call float asm sideeffect "", "=v,0"(float %592) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %593 = insertelement <4 x float> poison, float %591, i64 0, !dbg !43
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi59), !dbg !40
  %595 = fadd float %594, %k0.sum.anchor1830, !dbg !44
  %k0.sum.anchor1838 = tail call float asm sideeffect "", "=v,0"(float %595) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %596 = insertelement <4 x float> %593, float %594, i64 1, !dbg !43
  %597 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi61), !dbg !40
  %598 = fadd float %597, %k0.sum.anchor1831, !dbg !44
  %k0.sum.anchor1839 = tail call float asm sideeffect "", "=v,0"(float %598) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %599 = insertelement <4 x float> %596, float %597, i64 2, !dbg !43
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi63), !dbg !40
  %601 = fadd float %600, %k0.sum.anchor1832, !dbg !44
  %k0.sum.anchor1840 = tail call float asm sideeffect "", "=v,0"(float %601) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %602 = insertelement <4 x float> %599, float %600, i64 3, !dbg !43
  %math.result250 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %602) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02051 = extractelement <4 x float> %math.result250, i64 0, !dbg !43
  %.cvtpack.f12052 = extractelement <4 x float> %math.result250, i64 1, !dbg !43
  %.cvtpack.f22053 = extractelement <4 x float> %math.result250, i64 2, !dbg !43
  %.cvtpack.f32054 = extractelement <4 x float> %math.result250, i64 3, !dbg !43
  %.cvtpack.p02055 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02051, float %.cvtpack.f12052) #12, !dbg !43
  %.cvtpack.p12056 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22053, float %.cvtpack.f32054) #12, !dbg !43
  %.cvtpack.v02057 = insertelement <2 x i32> poison, i32 %.cvtpack.p02055, i64 0, !dbg !43
  %.cvtpack.v12058 = insertelement <2 x i32> %.cvtpack.v02057, i32 %.cvtpack.p12056, i64 1, !dbg !43
  %603 = bitcast <2 x i32> %.cvtpack.v12058 to <4 x half>, !dbg !43
  %math.packed251 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %603) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %604 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %590, <4 x float> %572) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %605 = shufflevector <8 x half> %524, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %606 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %605, <4 x float> %574) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %607 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi65), !dbg !40
  %608 = fadd float %607, %k0.sum.anchor1833, !dbg !44
  %k0.sum.anchor1841 = tail call float asm sideeffect "", "=v,0"(float %608) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %609 = insertelement <4 x float> poison, float %607, i64 0, !dbg !43
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi67), !dbg !40
  %611 = fadd float %610, %k0.sum.anchor1834, !dbg !44
  %k0.sum.anchor1842 = tail call float asm sideeffect "", "=v,0"(float %611) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %612 = insertelement <4 x float> %609, float %610, i64 1, !dbg !43
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi69), !dbg !40
  %614 = fadd float %613, %k0.sum.anchor1835, !dbg !44
  %k0.sum.anchor1843 = tail call float asm sideeffect "", "=v,0"(float %614) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %615 = insertelement <4 x float> %612, float %613, i64 2, !dbg !43
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi71), !dbg !40
  %617 = fadd float %616, %k0.sum.anchor1836, !dbg !44
  %k0.sum.anchor1844 = tail call float asm sideeffect "", "=v,0"(float %617) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %618 = insertelement <4 x float> %615, float %616, i64 3, !dbg !43
  %math.result266 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %618) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02059 = extractelement <4 x float> %math.result266, i64 0, !dbg !43
  %.cvtpack.f12060 = extractelement <4 x float> %math.result266, i64 1, !dbg !43
  %.cvtpack.f22061 = extractelement <4 x float> %math.result266, i64 2, !dbg !43
  %.cvtpack.f32062 = extractelement <4 x float> %math.result266, i64 3, !dbg !43
  %.cvtpack.p02063 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02059, float %.cvtpack.f12060) #12, !dbg !43
  %.cvtpack.p12064 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22061, float %.cvtpack.f32062) #12, !dbg !43
  %.cvtpack.v02065 = insertelement <2 x i32> poison, i32 %.cvtpack.p02063, i64 0, !dbg !43
  %.cvtpack.v12066 = insertelement <2 x i32> %.cvtpack.v02065, i32 %.cvtpack.p12064, i64 1, !dbg !43
  %619 = bitcast <2 x i32> %.cvtpack.v12066 to <4 x half>, !dbg !43
  %math.packed267 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %619) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %620 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %590, <4 x float> %588) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %621 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %605, <4 x float> %589) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %622 = shufflevector <8 x half> %526, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %623 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi73), !dbg !40
  %624 = fadd float %623, %k0.sum.anchor1837, !dbg !44
  %k0.sum.anchor1845 = tail call float asm sideeffect "", "=v,0"(float %624) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %625 = insertelement <4 x float> poison, float %623, i64 0, !dbg !43
  %626 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi75), !dbg !40
  %627 = fadd float %626, %k0.sum.anchor1838, !dbg !44
  %k0.sum.anchor1846 = tail call float asm sideeffect "", "=v,0"(float %627) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %628 = insertelement <4 x float> %625, float %626, i64 1, !dbg !43
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi77), !dbg !40
  %630 = fadd float %629, %k0.sum.anchor1839, !dbg !44
  %k0.sum.anchor1847 = tail call float asm sideeffect "", "=v,0"(float %630) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %631 = insertelement <4 x float> %628, float %629, i64 2, !dbg !43
  %632 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi79), !dbg !40
  %633 = fadd float %632, %k0.sum.anchor1840, !dbg !44
  %k0.sum.anchor1848 = tail call float asm sideeffect "", "=v,0"(float %633) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %634 = insertelement <4 x float> %631, float %632, i64 3, !dbg !43
  %math.result252 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %634) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02067 = extractelement <4 x float> %math.result252, i64 0, !dbg !43
  %.cvtpack.f12068 = extractelement <4 x float> %math.result252, i64 1, !dbg !43
  %.cvtpack.f22069 = extractelement <4 x float> %math.result252, i64 2, !dbg !43
  %.cvtpack.f32070 = extractelement <4 x float> %math.result252, i64 3, !dbg !43
  %.cvtpack.p02071 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02067, float %.cvtpack.f12068) #12, !dbg !43
  %.cvtpack.p12072 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22069, float %.cvtpack.f32070) #12, !dbg !43
  %.cvtpack.v02073 = insertelement <2 x i32> poison, i32 %.cvtpack.p02071, i64 0, !dbg !43
  %.cvtpack.v12074 = insertelement <2 x i32> %.cvtpack.v02073, i32 %.cvtpack.p12072, i64 1, !dbg !43
  %635 = bitcast <2 x i32> %.cvtpack.v12074 to <4 x half>, !dbg !43
  %math.packed253 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %635) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %636 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %622, <4 x float> %604) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %637 = shufflevector <8 x half> %526, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %638 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %637, <4 x float> %606) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %639 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi81), !dbg !40
  %640 = fadd float %639, %k0.sum.anchor1841, !dbg !44
  %k0.sum.anchor1849 = tail call float asm sideeffect "", "=v,0"(float %640) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %641 = insertelement <4 x float> poison, float %639, i64 0, !dbg !43
  %642 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi83), !dbg !40
  %643 = fadd float %642, %k0.sum.anchor1842, !dbg !44
  %k0.sum.anchor1850 = tail call float asm sideeffect "", "=v,0"(float %643) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %644 = insertelement <4 x float> %641, float %642, i64 1, !dbg !43
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi85), !dbg !40
  %646 = fadd float %645, %k0.sum.anchor1843, !dbg !44
  %k0.sum.anchor1851 = tail call float asm sideeffect "", "=v,0"(float %646) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %647 = insertelement <4 x float> %644, float %645, i64 2, !dbg !43
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi87), !dbg !40
  %649 = fadd float %648, %k0.sum.anchor1844, !dbg !44
  %k0.sum.anchor1852 = tail call float asm sideeffect "", "=v,0"(float %649) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %650 = insertelement <4 x float> %647, float %648, i64 3, !dbg !43
  %math.result268 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %650) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02075 = extractelement <4 x float> %math.result268, i64 0, !dbg !43
  %.cvtpack.f12076 = extractelement <4 x float> %math.result268, i64 1, !dbg !43
  %.cvtpack.f22077 = extractelement <4 x float> %math.result268, i64 2, !dbg !43
  %.cvtpack.f32078 = extractelement <4 x float> %math.result268, i64 3, !dbg !43
  %.cvtpack.p02079 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02075, float %.cvtpack.f12076) #12, !dbg !43
  %.cvtpack.p12080 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22077, float %.cvtpack.f32078) #12, !dbg !43
  %.cvtpack.v02081 = insertelement <2 x i32> poison, i32 %.cvtpack.p02079, i64 0, !dbg !43
  %.cvtpack.v12082 = insertelement <2 x i32> %.cvtpack.v02081, i32 %.cvtpack.p12080, i64 1, !dbg !43
  %651 = bitcast <2 x i32> %.cvtpack.v12082 to <4 x half>, !dbg !43
  %math.packed269 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %651) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %652 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %622, <4 x float> %620) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %653 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %637, <4 x float> %621) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %654 = shufflevector <8 x half> %528, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi89), !dbg !40
  %656 = fadd float %655, %k0.sum.anchor1845, !dbg !44
  %k0.sum.anchor1853 = tail call float asm sideeffect "", "=v,0"(float %656) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %657 = insertelement <4 x float> poison, float %655, i64 0, !dbg !43
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi91), !dbg !40
  %659 = fadd float %658, %k0.sum.anchor1846, !dbg !44
  %k0.sum.anchor1854 = tail call float asm sideeffect "", "=v,0"(float %659) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %660 = insertelement <4 x float> %657, float %658, i64 1, !dbg !43
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi93), !dbg !40
  %662 = fadd float %661, %k0.sum.anchor1847, !dbg !44
  %k0.sum.anchor1855 = tail call float asm sideeffect "", "=v,0"(float %662) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %663 = insertelement <4 x float> %660, float %661, i64 2, !dbg !43
  %664 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi95), !dbg !40
  %665 = fadd float %664, %k0.sum.anchor1848, !dbg !44
  %k0.sum.anchor1856 = tail call float asm sideeffect "", "=v,0"(float %665) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %666 = insertelement <4 x float> %663, float %664, i64 3, !dbg !43
  %math.result254 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %666) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02083 = extractelement <4 x float> %math.result254, i64 0, !dbg !43
  %.cvtpack.f12084 = extractelement <4 x float> %math.result254, i64 1, !dbg !43
  %.cvtpack.f22085 = extractelement <4 x float> %math.result254, i64 2, !dbg !43
  %.cvtpack.f32086 = extractelement <4 x float> %math.result254, i64 3, !dbg !43
  %.cvtpack.p02087 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02083, float %.cvtpack.f12084) #12, !dbg !43
  %.cvtpack.p12088 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22085, float %.cvtpack.f32086) #12, !dbg !43
  %.cvtpack.v02089 = insertelement <2 x i32> poison, i32 %.cvtpack.p02087, i64 0, !dbg !43
  %.cvtpack.v12090 = insertelement <2 x i32> %.cvtpack.v02089, i32 %.cvtpack.p12088, i64 1, !dbg !43
  %667 = bitcast <2 x i32> %.cvtpack.v12090 to <4 x half>, !dbg !43
  %math.packed255 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %667) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %668 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %654, <4 x float> %636) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %669 = shufflevector <8 x half> %528, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %670 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %669, <4 x float> %638) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi97), !dbg !40
  %672 = fadd float %671, %k0.sum.anchor1849, !dbg !44
  %k0.sum.anchor1857 = tail call float asm sideeffect "", "=v,0"(float %672) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %673 = insertelement <4 x float> poison, float %671, i64 0, !dbg !43
  %674 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi99), !dbg !40
  %675 = fadd float %674, %k0.sum.anchor1850, !dbg !44
  %k0.sum.anchor1858 = tail call float asm sideeffect "", "=v,0"(float %675) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %676 = insertelement <4 x float> %673, float %674, i64 1, !dbg !43
  %677 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi101), !dbg !40
  %678 = fadd float %677, %k0.sum.anchor1851, !dbg !44
  %k0.sum.anchor1859 = tail call float asm sideeffect "", "=v,0"(float %678) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %679 = insertelement <4 x float> %676, float %677, i64 2, !dbg !43
  %680 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi103), !dbg !40
  %681 = fadd float %680, %k0.sum.anchor1852, !dbg !44
  %k0.sum.anchor1860 = tail call float asm sideeffect "", "=v,0"(float %681) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %682 = insertelement <4 x float> %679, float %680, i64 3, !dbg !43
  %math.result270 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %682) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02091 = extractelement <4 x float> %math.result270, i64 0, !dbg !43
  %.cvtpack.f12092 = extractelement <4 x float> %math.result270, i64 1, !dbg !43
  %.cvtpack.f22093 = extractelement <4 x float> %math.result270, i64 2, !dbg !43
  %.cvtpack.f32094 = extractelement <4 x float> %math.result270, i64 3, !dbg !43
  %.cvtpack.p02095 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02091, float %.cvtpack.f12092) #12, !dbg !43
  %.cvtpack.p12096 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22093, float %.cvtpack.f32094) #12, !dbg !43
  %.cvtpack.v02097 = insertelement <2 x i32> poison, i32 %.cvtpack.p02095, i64 0, !dbg !43
  %.cvtpack.v12098 = insertelement <2 x i32> %.cvtpack.v02097, i32 %.cvtpack.p12096, i64 1, !dbg !43
  %683 = bitcast <2 x i32> %.cvtpack.v12098 to <4 x half>, !dbg !43
  %math.packed271 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %683) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %654, <4 x float> %652) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %685 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %669, <4 x float> %653) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %686 = shufflevector <8 x half> %530, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %687 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi105), !dbg !40
  %688 = fadd float %687, %k0.sum.anchor1853, !dbg !44
  %k0.sum.anchor1861 = tail call float asm sideeffect "", "=v,0"(float %688) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %689 = insertelement <4 x float> poison, float %687, i64 0, !dbg !43
  %690 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi107), !dbg !40
  %691 = fadd float %690, %k0.sum.anchor1854, !dbg !44
  %k0.sum.anchor1862 = tail call float asm sideeffect "", "=v,0"(float %691) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %692 = insertelement <4 x float> %689, float %690, i64 1, !dbg !43
  %693 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi109), !dbg !40
  %694 = fadd float %693, %k0.sum.anchor1855, !dbg !44
  %k0.sum.anchor1863 = tail call float asm sideeffect "", "=v,0"(float %694) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %695 = insertelement <4 x float> %692, float %693, i64 2, !dbg !43
  %696 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi111), !dbg !40
  %697 = fadd float %696, %k0.sum.anchor1856, !dbg !44
  %k0.sum.anchor1864 = tail call float asm sideeffect "", "=v,0"(float %697) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %698 = insertelement <4 x float> %695, float %696, i64 3, !dbg !43
  %math.result256 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %698) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02099 = extractelement <4 x float> %math.result256, i64 0, !dbg !43
  %.cvtpack.f12100 = extractelement <4 x float> %math.result256, i64 1, !dbg !43
  %.cvtpack.f22101 = extractelement <4 x float> %math.result256, i64 2, !dbg !43
  %.cvtpack.f32102 = extractelement <4 x float> %math.result256, i64 3, !dbg !43
  %.cvtpack.p02103 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02099, float %.cvtpack.f12100) #12, !dbg !43
  %.cvtpack.p12104 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22101, float %.cvtpack.f32102) #12, !dbg !43
  %.cvtpack.v02105 = insertelement <2 x i32> poison, i32 %.cvtpack.p02103, i64 0, !dbg !43
  %.cvtpack.v12106 = insertelement <2 x i32> %.cvtpack.v02105, i32 %.cvtpack.p12104, i64 1, !dbg !43
  %699 = bitcast <2 x i32> %.cvtpack.v12106 to <4 x half>, !dbg !43
  %math.packed257 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %699) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %700 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %686, <4 x float> %668) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %701 = shufflevector <8 x half> %530, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %702 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %701, <4 x float> %670) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %703 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi113), !dbg !40
  %704 = fadd float %703, %k0.sum.anchor1857, !dbg !44
  %k0.sum.anchor1865 = tail call float asm sideeffect "", "=v,0"(float %704) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %705 = insertelement <4 x float> poison, float %703, i64 0, !dbg !43
  %706 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi115), !dbg !40
  %707 = fadd float %706, %k0.sum.anchor1858, !dbg !44
  %k0.sum.anchor1866 = tail call float asm sideeffect "", "=v,0"(float %707) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %708 = insertelement <4 x float> %705, float %706, i64 1, !dbg !43
  %709 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi117), !dbg !40
  %710 = fadd float %709, %k0.sum.anchor1859, !dbg !44
  %k0.sum.anchor1867 = tail call float asm sideeffect "", "=v,0"(float %710) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %711 = insertelement <4 x float> %708, float %709, i64 2, !dbg !43
  %712 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi119), !dbg !40
  %713 = fadd float %712, %k0.sum.anchor1860, !dbg !44
  %k0.sum.anchor1868 = tail call float asm sideeffect "", "=v,0"(float %713) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %714 = insertelement <4 x float> %711, float %712, i64 3, !dbg !43
  %math.result272 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %714) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02107 = extractelement <4 x float> %math.result272, i64 0, !dbg !43
  %.cvtpack.f12108 = extractelement <4 x float> %math.result272, i64 1, !dbg !43
  %.cvtpack.f22109 = extractelement <4 x float> %math.result272, i64 2, !dbg !43
  %.cvtpack.f32110 = extractelement <4 x float> %math.result272, i64 3, !dbg !43
  %.cvtpack.p02111 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02107, float %.cvtpack.f12108) #12, !dbg !43
  %.cvtpack.p12112 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22109, float %.cvtpack.f32110) #12, !dbg !43
  %.cvtpack.v02113 = insertelement <2 x i32> poison, i32 %.cvtpack.p02111, i64 0, !dbg !43
  %.cvtpack.v12114 = insertelement <2 x i32> %.cvtpack.v02113, i32 %.cvtpack.p12112, i64 1, !dbg !43
  %715 = bitcast <2 x i32> %.cvtpack.v12114 to <4 x half>, !dbg !43
  %math.packed273 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %715) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %716 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %686, <4 x float> %684) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %717 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %701, <4 x float> %685) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %718 = shufflevector <8 x half> %532, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %719 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi121), !dbg !40
  %720 = fadd float %719, %k0.sum.anchor1861, !dbg !44
  %k0.sum.anchor1869 = tail call float asm sideeffect "", "=v,0"(float %720) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %721 = insertelement <4 x float> poison, float %719, i64 0, !dbg !43
  %722 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi123), !dbg !40
  %723 = fadd float %722, %k0.sum.anchor1862, !dbg !44
  %k0.sum.anchor1870 = tail call float asm sideeffect "", "=v,0"(float %723) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %724 = insertelement <4 x float> %721, float %722, i64 1, !dbg !43
  %725 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi125), !dbg !40
  %726 = fadd float %725, %k0.sum.anchor1863, !dbg !44
  %k0.sum.anchor1871 = tail call float asm sideeffect "", "=v,0"(float %726) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %727 = insertelement <4 x float> %724, float %725, i64 2, !dbg !43
  %728 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi127), !dbg !40
  %729 = fadd float %728, %k0.sum.anchor1864, !dbg !44
  %k0.sum.anchor1872 = tail call float asm sideeffect "", "=v,0"(float %729) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %730 = insertelement <4 x float> %727, float %728, i64 3, !dbg !43
  %math.result258 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %730) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02115 = extractelement <4 x float> %math.result258, i64 0, !dbg !43
  %.cvtpack.f12116 = extractelement <4 x float> %math.result258, i64 1, !dbg !43
  %.cvtpack.f22117 = extractelement <4 x float> %math.result258, i64 2, !dbg !43
  %.cvtpack.f32118 = extractelement <4 x float> %math.result258, i64 3, !dbg !43
  %.cvtpack.p02119 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02115, float %.cvtpack.f12116) #12, !dbg !43
  %.cvtpack.p12120 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22117, float %.cvtpack.f32118) #12, !dbg !43
  %.cvtpack.v02121 = insertelement <2 x i32> poison, i32 %.cvtpack.p02119, i64 0, !dbg !43
  %.cvtpack.v12122 = insertelement <2 x i32> %.cvtpack.v02121, i32 %.cvtpack.p12120, i64 1, !dbg !43
  %731 = bitcast <2 x i32> %.cvtpack.v12122 to <4 x half>, !dbg !43
  %math.packed259 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %731) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %718, <4 x float> %700) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %733 = shufflevector <8 x half> %532, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %733, <4 x float> %702) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %735 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi129), !dbg !40
  %736 = fadd float %735, %k0.sum.anchor1865, !dbg !44
  %k0.sum.anchor1873 = tail call float asm sideeffect "", "=v,0"(float %736) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %737 = insertelement <4 x float> poison, float %735, i64 0, !dbg !43
  %738 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi131), !dbg !40
  %739 = fadd float %738, %k0.sum.anchor1866, !dbg !44
  %k0.sum.anchor1874 = tail call float asm sideeffect "", "=v,0"(float %739) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %740 = insertelement <4 x float> %737, float %738, i64 1, !dbg !43
  %741 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi133), !dbg !40
  %742 = fadd float %741, %k0.sum.anchor1867, !dbg !44
  %k0.sum.anchor1875 = tail call float asm sideeffect "", "=v,0"(float %742) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %743 = insertelement <4 x float> %740, float %741, i64 2, !dbg !43
  %744 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi135), !dbg !40
  %745 = fadd float %744, %k0.sum.anchor1868, !dbg !44
  %k0.sum.anchor1876 = tail call float asm sideeffect "", "=v,0"(float %745) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %746 = insertelement <4 x float> %743, float %744, i64 3, !dbg !43
  %math.result274 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %746) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02123 = extractelement <4 x float> %math.result274, i64 0, !dbg !43
  %.cvtpack.f12124 = extractelement <4 x float> %math.result274, i64 1, !dbg !43
  %.cvtpack.f22125 = extractelement <4 x float> %math.result274, i64 2, !dbg !43
  %.cvtpack.f32126 = extractelement <4 x float> %math.result274, i64 3, !dbg !43
  %.cvtpack.p02127 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02123, float %.cvtpack.f12124) #12, !dbg !43
  %.cvtpack.p12128 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22125, float %.cvtpack.f32126) #12, !dbg !43
  %.cvtpack.v02129 = insertelement <2 x i32> poison, i32 %.cvtpack.p02127, i64 0, !dbg !43
  %.cvtpack.v12130 = insertelement <2 x i32> %.cvtpack.v02129, i32 %.cvtpack.p12128, i64 1, !dbg !43
  %747 = bitcast <2 x i32> %.cvtpack.v12130 to <4 x half>, !dbg !43
  %math.packed275 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %747) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %748 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %718, <4 x float> %716) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %749 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %733, <4 x float> %717) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %750 = bitcast <4 x i32> %533 to <8 x half>, !dbg !42
  %751 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %752 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi137), !dbg !40
  %753 = fadd float %752, %k0.sum.anchor1869, !dbg !44
  %k0.sum.anchor1877 = tail call float asm sideeffect "", "=v,0"(float %753) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %754 = insertelement <4 x float> poison, float %752, i64 0, !dbg !43
  %755 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi139), !dbg !40
  %756 = fadd float %755, %k0.sum.anchor1870, !dbg !44
  %k0.sum.anchor1878 = tail call float asm sideeffect "", "=v,0"(float %756) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %757 = fadd float %k0.sum.anchor1877, %k0.sum.anchor1878, !dbg !44
  %k0.sum.anchor1879 = tail call float asm sideeffect "", "=v,0"(float %757) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %758 = insertelement <4 x float> %754, float %755, i64 1, !dbg !43
  %759 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi141), !dbg !40
  %760 = fadd float %759, %k0.sum.anchor1871, !dbg !44
  %k0.sum.anchor1880 = tail call float asm sideeffect "", "=v,0"(float %760) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %761 = insertelement <4 x float> %758, float %759, i64 2, !dbg !43
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi143), !dbg !40
  %763 = fadd float %762, %k0.sum.anchor1872, !dbg !44
  %k0.sum.anchor1881 = tail call float asm sideeffect "", "=v,0"(float %763) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %764 = fadd float %k0.sum.anchor1880, %k0.sum.anchor1881, !dbg !44
  %k0.sum.anchor1882 = tail call float asm sideeffect "", "=v,0"(float %764) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %765 = fadd float %k0.sum.anchor1879, %k0.sum.anchor1882, !dbg !44
  %k0.sum.anchor1883 = tail call float asm sideeffect "", "=v,0"(float %765) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %766 = fadd float %507, %k0.sum.anchor1883, !dbg !44
  %k0.sum.anchor1884 = tail call float asm sideeffect "", "=v,0"(float %766) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %767 = insertelement <1 x float> poison, float %k0.sum.anchor1884, i64 0, !dbg !12
  %768 = insertvalue [2 x <1 x float>] poison, <1 x float> %767, 0, !dbg !12
  %769 = insertelement <4 x float> %761, float %762, i64 3, !dbg !43
  %math.result260 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %769) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02131 = extractelement <4 x float> %math.result260, i64 0, !dbg !43
  %.cvtpack.f12132 = extractelement <4 x float> %math.result260, i64 1, !dbg !43
  %.cvtpack.f22133 = extractelement <4 x float> %math.result260, i64 2, !dbg !43
  %.cvtpack.f32134 = extractelement <4 x float> %math.result260, i64 3, !dbg !43
  %.cvtpack.p02135 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02131, float %.cvtpack.f12132) #12, !dbg !43
  %.cvtpack.p12136 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22133, float %.cvtpack.f32134) #12, !dbg !43
  %.cvtpack.v02137 = insertelement <2 x i32> poison, i32 %.cvtpack.p02135, i64 0, !dbg !43
  %.cvtpack.v12138 = insertelement <2 x i32> %.cvtpack.v02137, i32 %.cvtpack.p12136, i64 1, !dbg !43
  %770 = bitcast <2 x i32> %.cvtpack.v12138 to <4 x half>, !dbg !43
  %math.packed261 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %770) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %771 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %751, <4 x float> %732) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %772 = shufflevector <8 x half> %750, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %773 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %772, <4 x float> %734) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi145), !dbg !40
  %775 = fadd float %774, %k0.sum.anchor1873, !dbg !44
  %k0.sum.anchor1885 = tail call float asm sideeffect "", "=v,0"(float %775) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %776 = insertelement <4 x float> poison, float %774, i64 0, !dbg !43
  %777 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi147), !dbg !40
  %778 = fadd float %777, %k0.sum.anchor1874, !dbg !44
  %k0.sum.anchor1886 = tail call float asm sideeffect "", "=v,0"(float %778) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %779 = fadd float %k0.sum.anchor1885, %k0.sum.anchor1886, !dbg !44
  %k0.sum.anchor1887 = tail call float asm sideeffect "", "=v,0"(float %779) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %780 = insertelement <4 x float> %776, float %777, i64 1, !dbg !43
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi149), !dbg !40
  %782 = fadd float %781, %k0.sum.anchor1875, !dbg !44
  %k0.sum.anchor1888 = tail call float asm sideeffect "", "=v,0"(float %782) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %783 = insertelement <4 x float> %780, float %781, i64 2, !dbg !43
  %784 = tail call float @llvm.amdgcn.exp2.f32(float %.pre-phi151), !dbg !40
  %785 = fadd float %784, %k0.sum.anchor1876, !dbg !44
  %k0.sum.anchor1889 = tail call float asm sideeffect "", "=v,0"(float %785) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %786 = fadd float %k0.sum.anchor1888, %k0.sum.anchor1889, !dbg !44
  %k0.sum.anchor1890 = tail call float asm sideeffect "", "=v,0"(float %786) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %787 = fadd float %k0.sum.anchor1887, %k0.sum.anchor1890, !dbg !44
  %k0.sum.anchor1891 = tail call float asm sideeffect "", "=v,0"(float %787) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %788 = fadd float %509, %k0.sum.anchor1891, !dbg !44
  %k0.sum.anchor1892 = tail call float asm sideeffect "", "=v,0"(float %788) #8, !dbg !44, !frisk.cross_bn_sum_anchor !5
  %789 = insertelement <1 x float> poison, float %k0.sum.anchor1892, i64 0, !dbg !12
  %790 = insertvalue [2 x <1 x float>] %768, <1 x float> %789, 1, !dbg !12
  %791 = insertelement <4 x float> %783, float %784, i64 3, !dbg !43
  %math.result276 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %791) #8, !dbg !43, !frisk.register_lifetimes !5
  %.cvtpack.f02139 = extractelement <4 x float> %math.result276, i64 0, !dbg !43
  %.cvtpack.f12140 = extractelement <4 x float> %math.result276, i64 1, !dbg !43
  %.cvtpack.f22141 = extractelement <4 x float> %math.result276, i64 2, !dbg !43
  %.cvtpack.f32142 = extractelement <4 x float> %math.result276, i64 3, !dbg !43
  %.cvtpack.p02143 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f02139, float %.cvtpack.f12140) #12, !dbg !43
  %.cvtpack.p12144 = tail call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %.cvtpack.f22141, float %.cvtpack.f32142) #12, !dbg !43
  %.cvtpack.v02145 = insertelement <2 x i32> poison, i32 %.cvtpack.p02143, i64 0, !dbg !43
  %.cvtpack.v12146 = insertelement <2 x i32> %.cvtpack.v02145, i32 %.cvtpack.p12144, i64 1, !dbg !43
  %792 = bitcast <2 x i32> %.cvtpack.v12146 to <4 x half>, !dbg !43
  %math.packed277 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %792) #8, !dbg !44, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42, !frisk.softmax_fragments !5
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %751, <4 x float> %748) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %794 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %772, <4 x float> %749) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %795 = extractvalue [2 x <8 x float>] %lazy.state2025, 0, !dbg !41
  %796 = shufflevector <8 x float> %795, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %797 = shufflevector <8 x float> %795, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %798 = extractvalue [2 x <8 x float>] %lazy.state2025, 1, !dbg !41
  %799 = shufflevector <8 x float> %798, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %800 = shufflevector <8 x float> %798, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %.idx13241325 = add nsw i64 %address.local796, %205, !dbg !38
  %.idx13241325.tr = trunc nsw i64 %.idx13241325 to i32, !dbg !38
  %dma.buffer.offset1243 = shl nsw i32 %.idx13241325.tr, 1, !dbg !38
  %801 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1243, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %802 = trunc nuw nsw i64 %205 to i32, !dbg !38
  %.idx13261327.tr = or disjoint i32 %address.local803, %802, !dbg !38
  %dma.buffer.offset1241 = shl nuw nsw i32 %.idx13261327.tr, 1, !dbg !38
  %803 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1241, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %address.local873 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local813, !dbg !42
  %804 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local873, i32 0) #11, !dbg !42
  %805 = bitcast <4 x i32> %804 to <8 x half>, !dbg !42
  %address.local874 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local821, !dbg !42
  %806 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !42
  %807 = bitcast <4 x i32> %806 to <8 x half>, !dbg !42
  %address.local875 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local829, !dbg !42
  %808 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #11, !dbg !42
  %809 = bitcast <4 x i32> %808 to <8 x half>, !dbg !42
  %address.local876 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local837, !dbg !42
  %810 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local876, i32 0) #11, !dbg !42
  %811 = bitcast <4 x i32> %810 to <8 x half>, !dbg !42
  %address.local877 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local845, !dbg !42
  %812 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local877, i32 0) #11, !dbg !42
  %813 = bitcast <4 x i32> %812 to <8 x half>, !dbg !42
  %address.local878 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local853, !dbg !42
  %814 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !42
  %815 = bitcast <4 x i32> %814 to <8 x half>, !dbg !42
  %address.local879 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local861, !dbg !42
  %816 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #11, !dbg !42
  %817 = bitcast <4 x i32> %816 to <8 x half>, !dbg !42
  %address.local880 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local869, !dbg !42
  %818 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local880, i32 0) #11, !dbg !42
  %819 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %820 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %819, <4 x float> %796) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %821 = shufflevector <8 x half> %805, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %821, <4 x float> %797) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %823 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %819, <4 x float> %799) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %821, <4 x float> %800) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %825 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %825, <4 x float> %820) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %827 = shufflevector <8 x half> %807, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %827, <4 x float> %822) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %829 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %825, <4 x float> %823) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %827, <4 x float> %824) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %831 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %832 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %831, <4 x float> %826) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %833 = shufflevector <8 x half> %809, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %833, <4 x float> %828) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %835 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %831, <4 x float> %829) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %833, <4 x float> %830) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %837 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %837, <4 x float> %832) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %839 = shufflevector <8 x half> %811, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %840 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %839, <4 x float> %834) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %837, <4 x float> %835) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %839, <4 x float> %836) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %843 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %844 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %843, <4 x float> %838) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %845 = shufflevector <8 x half> %813, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %846 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %845, <4 x float> %840) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %843, <4 x float> %841) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %845, <4 x float> %842) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %849 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %850 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %849, <4 x float> %844) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %851 = shufflevector <8 x half> %815, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %851, <4 x float> %846) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %849, <4 x float> %847) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %851, <4 x float> %848) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %855 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %855, <4 x float> %850) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %857 = shufflevector <8 x half> %817, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %857, <4 x float> %852) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %859 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %855, <4 x float> %853) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %857, <4 x float> %854) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %861 = bitcast <4 x i32> %818 to <8 x half>, !dbg !42
  %862 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %862, <4 x float> %856) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %864 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %864, <4 x float> %858) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %866 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %862, <4 x float> %859) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %864, <4 x float> %860) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %868 = extractvalue [2 x <8 x float>] %lazy.state2024, 0, !dbg !41
  %869 = shufflevector <8 x float> %868, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %870 = shufflevector <8 x float> %868, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %871 = extractvalue [2 x <8 x float>] %lazy.state2024, 1, !dbg !41
  %872 = shufflevector <8 x float> %871, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %873 = shufflevector <8 x float> %871, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %.idx13281329 = add nsw i64 %address.local796, %258, !dbg !38
  %.idx13281329.tr = trunc nsw i64 %.idx13281329 to i32, !dbg !38
  %dma.buffer.offset1239 = shl nsw i32 %.idx13281329.tr, 1, !dbg !38
  %874 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1239, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %.tr1331 = trunc nuw nsw i64 %258 to i32, !dbg !38
  %875 = or disjoint i32 %address.local803, %.tr1331, !dbg !38
  %dma.buffer.offset = shl nuw nsw i32 %875, 1, !dbg !38
  %876 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %65, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1252) #9, !dbg !38
  %877 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local814, i32 0) #11, !dbg !42
  %878 = bitcast <4 x i32> %877 to <8 x half>, !dbg !42
  %879 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local822, i32 0) #11, !dbg !42
  %880 = bitcast <4 x i32> %879 to <8 x half>, !dbg !42
  %881 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local830, i32 0) #11, !dbg !42
  %882 = bitcast <4 x i32> %881 to <8 x half>, !dbg !42
  %883 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local838, i32 0) #11, !dbg !42
  %884 = bitcast <4 x i32> %883 to <8 x half>, !dbg !42
  %885 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local846, i32 0) #11, !dbg !42
  %886 = bitcast <4 x i32> %885 to <8 x half>, !dbg !42
  %887 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local854, i32 0) #11, !dbg !42
  %888 = bitcast <4 x i32> %887 to <8 x half>, !dbg !42
  %889 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local862, i32 0) #11, !dbg !42
  %890 = bitcast <4 x i32> %889 to <8 x half>, !dbg !42
  %891 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local870, i32 0) #11, !dbg !42
  %892 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %892, <4 x float> %869) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %894 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %894, <4 x float> %870) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %896 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %892, <4 x float> %872) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %894, <4 x float> %873) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %898 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %898, <4 x float> %893) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %900 = shufflevector <8 x half> %880, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %900, <4 x float> %895) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %898, <4 x float> %896) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %900, <4 x float> %897) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %904 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %904, <4 x float> %899) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %906 = shufflevector <8 x half> %882, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %906, <4 x float> %901) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %904, <4 x float> %902) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %906, <4 x float> %903) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %910 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %910, <4 x float> %905) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %912 = shufflevector <8 x half> %884, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %912, <4 x float> %907) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %910, <4 x float> %908) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %912, <4 x float> %909) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %916 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %916, <4 x float> %911) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %918 = shufflevector <8 x half> %886, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %918, <4 x float> %913) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %916, <4 x float> %914) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %918, <4 x float> %915) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %922 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %922, <4 x float> %917) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %924 = shufflevector <8 x half> %888, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %924, <4 x float> %919) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %922, <4 x float> %920) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %924, <4 x float> %921) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %928 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %928, <4 x float> %923) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %930 = shufflevector <8 x half> %890, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %930, <4 x float> %925) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %932 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %928, <4 x float> %926) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %933 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %930, <4 x float> %927) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %934 = bitcast <4 x i32> %891 to <8 x half>, !dbg !42
  %935 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %935, <4 x float> %929) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %937 = shufflevector <8 x half> %934, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %937, <4 x float> %931) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %935, <4 x float> %932) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %937, <4 x float> %933) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !41
  %941 = extractvalue [2 x <8 x float>] %lazy.state2023, 0, !dbg !41
  %942 = shufflevector <8 x float> %941, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %943 = shufflevector <8 x float> %941, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  %944 = extractvalue [2 x <8 x float>] %lazy.state2023, 1, !dbg !41
  %945 = shufflevector <8 x float> %944, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !41
  %946 = shufflevector <8 x float> %944, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !41
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !38
  fence syncscope("workgroup") release, !dbg !38
  tail call void @llvm.amdgcn.s.barrier(), !dbg !38
  fence syncscope("workgroup") acquire, !dbg !38
  %947 = add nuw nsw i64 %124, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %124, 31, !dbg !42
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !42

k0.issue:                                         ; preds = %lazy.join
  %k0.next. = shl nuw nsw i64 %947, 14, !dbg !14
  %k0.next.1903 = add nuw nsw i64 %k0.next., %k0.prime.1893, !dbg !14
  %k0.next.kdma.source1348 = or disjoint i64 %k0.next.1903, %k0.prime.kdma.source1347, !dbg !14
  %k0.next.kdma.source1349 = trunc nuw nsw i64 %k0.next.kdma.source1348 to i32, !dbg !14
  %k0.next.kdma.source1350 = shl nuw nsw i32 %k0.next.kdma.source1349, 1, !dbg !14
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1350, %kdma.intra, !dbg !14
  %k0.next.1904 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1368 = or disjoint i64 %k0.next.1903, %k0.prime.kdma.source1367, !dbg !14
  %k0.next.kdma.source1369 = trunc nuw nsw i64 %k0.next.kdma.source1368 to i32, !dbg !14
  %k0.next.kdma.source1370 = shl nuw nsw i32 %k0.next.kdma.source1369, 1, !dbg !14
  %k0.next.kdma.offset1371 = or disjoint i32 %k0.next.kdma.source1370, %kdma.intra.hi, !dbg !14
  %k0.next.1905 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1372, i32 %k0.next.kdma.offset1371, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1392 = or disjoint i64 %k0.next.1903, %k0.prime.kdma.source1391, !dbg !14
  %k0.next.kdma.source1393 = trunc nuw nsw i64 %k0.next.kdma.source1392 to i32, !dbg !14
  %k0.next.kdma.source1394 = shl nuw nsw i32 %k0.next.kdma.source1393, 1, !dbg !14
  %k0.next.kdma.offset1395 = add nuw nsw i32 %k0.next.kdma.source1394, %kdma.intra, !dbg !14
  %k0.next.1906 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %k0.next.kdma.offset1395, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1416 = or disjoint i64 %k0.next.1903, %k0.prime.kdma.source1415, !dbg !14
  %k0.next.kdma.source1417 = trunc nuw nsw i64 %k0.next.kdma.source1416 to i32, !dbg !14
  %k0.next.kdma.source1418 = shl nuw nsw i32 %k0.next.kdma.source1417, 1, !dbg !14
  %k0.next.kdma.offset1419 = add nuw nsw i32 %k0.next.kdma.source1418, %kdma.intra.hi, !dbg !14
  %k0.next.1907 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1420, i32 %k0.next.kdma.offset1419, <4 x i32> %k0.prime.dma.buffer.rsrc1268) #9, !dbg !14, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %lazy.join
  %948 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local873, i32 0) #11, !dbg !42
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !42
  %950 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local874, i32 0) #11, !dbg !42
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !42
  %952 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local875, i32 0) #11, !dbg !42
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !42
  %954 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local876, i32 0) #11, !dbg !42
  %955 = bitcast <4 x i32> %954 to <8 x half>, !dbg !42
  %956 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local877, i32 0) #11, !dbg !42
  %957 = bitcast <4 x i32> %956 to <8 x half>, !dbg !42
  %958 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local878, i32 0) #11, !dbg !42
  %959 = bitcast <4 x i32> %958 to <8 x half>, !dbg !42
  %960 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local879, i32 0) #11, !dbg !42
  %961 = bitcast <4 x i32> %960 to <8 x half>, !dbg !42
  %962 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local880, i32 0) #11, !dbg !42
  %963 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %963, <4 x float> %942) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %965 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %965, <4 x float> %943) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %963, <4 x float> %945) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed263, <4 x half> %965, <4 x float> %946) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %969 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %969, <4 x float> %964) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %971 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed249, <4 x half> %971, <4 x float> %966) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %969, <4 x float> %967) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed265, <4 x half> %971, <4 x float> %968) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %975 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %975, <4 x float> %970) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %977 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed251, <4 x half> %977, <4 x float> %972) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %975, <4 x float> %973) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed267, <4 x half> %977, <4 x float> %974) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %981 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %981, <4 x float> %976) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %983 = shufflevector <8 x half> %955, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed253, <4 x half> %983, <4 x float> %978) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %981, <4 x float> %979) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed269, <4 x half> %983, <4 x float> %980) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %987 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %987, <4 x float> %982) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %989 = shufflevector <8 x half> %957, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed255, <4 x half> %989, <4 x float> %984) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %987, <4 x float> %985) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed271, <4 x half> %989, <4 x float> %986) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %993 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %993, <4 x float> %988) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %995 = shufflevector <8 x half> %959, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed257, <4 x half> %995, <4 x float> %990) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %993, <4 x float> %991) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed273, <4 x half> %995, <4 x float> %992) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %999 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %999, <4 x float> %994) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1001 = shufflevector <8 x half> %961, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed259, <4 x half> %1001, <4 x float> %996) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1003 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %999, <4 x float> %997) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed275, <4 x half> %1001, <4 x float> %998) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1005 = bitcast <4 x i32> %962 to <8 x half>, !dbg !42
  %1006 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1007 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1006, <4 x float> %1000) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1008 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1009 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed261, <4 x half> %1008, <4 x float> %1002) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1010 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1006, <4 x float> %1003) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !42
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed277, <4 x half> %1008, <4 x float> %1004) #8, !dbg !42, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %1012 = shufflevector <4 x float> %771, <4 x float> %773, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1013 = insertvalue [2 x <8 x float>] poison, <8 x float> %1012, 0, !dbg !12
  %1014 = shufflevector <4 x float> %793, <4 x float> %794, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1015 = insertvalue [2 x <8 x float>] %1013, <8 x float> %1014, 1, !dbg !12
  %1016 = shufflevector <4 x float> %863, <4 x float> %865, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1017 = insertvalue [2 x <8 x float>] poison, <8 x float> %1016, 0, !dbg !12
  %1018 = shufflevector <4 x float> %866, <4 x float> %867, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1019 = insertvalue [2 x <8 x float>] %1017, <8 x float> %1018, 1, !dbg !12
  %1020 = shufflevector <4 x float> %936, <4 x float> %938, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1021 = insertvalue [2 x <8 x float>] poison, <8 x float> %1020, 0, !dbg !12
  %1022 = shufflevector <4 x float> %939, <4 x float> %940, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1023 = insertvalue [2 x <8 x float>] %1021, <8 x float> %1022, 1, !dbg !12
  %1024 = shufflevector <4 x float> %1007, <4 x float> %1009, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1025 = insertvalue [2 x <8 x float>] poison, <8 x float> %1024, 0, !dbg !12
  %1026 = shufflevector <4 x float> %1010, <4 x float> %1011, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1027 = insertvalue [2 x <8 x float>] %1025, <8 x float> %1026, 1, !dbg !12
  %1028 = icmp samesign ult i64 %124, 31, !dbg !12
  br i1 %1028, label %117, label %1029, !dbg !12

1029:                                             ; preds = %k0.join
  %1030 = bitcast float %k0.sum.anchor1892 to i32, !dbg !12
  %1031 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1030), !dbg !12
  %1032 = bitcast float %k0.sum.anchor1884 to i32, !dbg !12
  %1033 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1032), !dbg !12
  %1034 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1034, 0, !dbg !12
  %1035 = and i32 %kdma.tid, 32, !dbg !12
  %.not40 = icmp eq i32 %1035, 0, !dbg !12
  %1036 = extractelement <4 x float> %1007, i64 0, !dbg !12
  %1037 = extractelement <4 x float> %1007, i64 1, !dbg !12
  %1038 = extractelement <4 x float> %1007, i64 2, !dbg !12
  %1039 = extractelement <4 x float> %1007, i64 3, !dbg !12
  %1040 = select i1 %.not, float %1037, float %1036, !dbg !12
  %1041 = bitcast float %1040 to i32, !dbg !12
  %1042 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1041), !dbg !12
  %1043 = bitcast i32 %1042 to float, !dbg !12
  %1044 = select i1 %.not, float %1036, float %1043, !dbg !12
  %1045 = select i1 %.not, float %1043, float %1037, !dbg !12
  %1046 = select i1 %.not, float %1039, float %1038, !dbg !12
  %1047 = bitcast float %1046 to i32, !dbg !12
  %1048 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1047), !dbg !12
  %1049 = bitcast i32 %1048 to float, !dbg !12
  %1050 = select i1 %.not, float %1038, float %1049, !dbg !12
  %1051 = select i1 %.not, float %1049, float %1039, !dbg !12
  %1052 = select i1 %.not40, float %1050, float %1044, !dbg !12
  %1053 = bitcast float %1052 to i32, !dbg !12
  %1054 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1053), !dbg !12
  %1055 = bitcast i32 %1054 to float, !dbg !12
  %1056 = select i1 %.not40, float %1051, float %1045, !dbg !12
  %1057 = bitcast float %1056 to i32, !dbg !12
  %1058 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1057), !dbg !12
  %1059 = bitcast i32 %1058 to float, !dbg !12
  %1060 = insertelement <4 x i1> poison, i1 %.not40, i64 0, !dbg !12
  %1061 = shufflevector <4 x i1> %1060, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1062 = insertelement <4 x float> poison, float %1044, i64 0, !dbg !12
  %1063 = insertelement <4 x float> %1062, float %1045, i64 1, !dbg !12
  %1064 = insertelement <4 x float> %1063, float %1055, i64 2, !dbg !12
  %1065 = insertelement <4 x float> %1064, float %1059, i64 3, !dbg !12
  %1066 = insertelement <4 x float> poison, float %1055, i64 0, !dbg !12
  %1067 = insertelement <4 x float> %1066, float %1059, i64 1, !dbg !12
  %1068 = insertelement <4 x float> %1067, float %1050, i64 2, !dbg !12
  %1069 = insertelement <4 x float> %1068, float %1051, i64 3, !dbg !12
  %1070 = select <4 x i1> %1061, <4 x float> %1065, <4 x float> %1069, !dbg !12
  %1071 = extractelement <4 x float> %1009, i64 0, !dbg !12
  %1072 = extractelement <4 x float> %1009, i64 1, !dbg !12
  %1073 = extractelement <4 x float> %1009, i64 2, !dbg !12
  %1074 = extractelement <4 x float> %1009, i64 3, !dbg !12
  %1075 = select i1 %.not, float %1072, float %1071, !dbg !12
  %1076 = bitcast float %1075 to i32, !dbg !12
  %1077 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1076), !dbg !12
  %1078 = bitcast i32 %1077 to float, !dbg !12
  %1079 = select i1 %.not, float %1071, float %1078, !dbg !12
  %1080 = select i1 %.not, float %1078, float %1072, !dbg !12
  %1081 = select i1 %.not, float %1074, float %1073, !dbg !12
  %1082 = bitcast float %1081 to i32, !dbg !12
  %1083 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1082), !dbg !12
  %1084 = bitcast i32 %1083 to float, !dbg !12
  %1085 = select i1 %.not, float %1073, float %1084, !dbg !12
  %1086 = select i1 %.not, float %1084, float %1074, !dbg !12
  %1087 = select i1 %.not40, float %1085, float %1079, !dbg !12
  %1088 = bitcast float %1087 to i32, !dbg !12
  %1089 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1088), !dbg !12
  %1090 = bitcast i32 %1089 to float, !dbg !12
  %1091 = select i1 %.not40, float %1086, float %1080, !dbg !12
  %1092 = bitcast float %1091 to i32, !dbg !12
  %1093 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1092), !dbg !12
  %1094 = bitcast i32 %1093 to float, !dbg !12
  %1095 = insertelement <4 x float> poison, float %1079, i64 0, !dbg !12
  %1096 = insertelement <4 x float> %1095, float %1080, i64 1, !dbg !12
  %1097 = insertelement <4 x float> %1096, float %1090, i64 2, !dbg !12
  %1098 = insertelement <4 x float> %1097, float %1094, i64 3, !dbg !12
  %1099 = insertelement <4 x float> poison, float %1090, i64 0, !dbg !12
  %1100 = insertelement <4 x float> %1099, float %1094, i64 1, !dbg !12
  %1101 = insertelement <4 x float> %1100, float %1085, i64 2, !dbg !12
  %1102 = insertelement <4 x float> %1101, float %1086, i64 3, !dbg !12
  %1103 = select <4 x i1> %1061, <4 x float> %1098, <4 x float> %1102, !dbg !12
  %1104 = extractelement <4 x float> %1010, i64 0, !dbg !12
  %1105 = extractelement <4 x float> %1010, i64 1, !dbg !12
  %1106 = extractelement <4 x float> %1010, i64 2, !dbg !12
  %1107 = extractelement <4 x float> %1010, i64 3, !dbg !12
  %1108 = select i1 %.not, float %1105, float %1104, !dbg !12
  %1109 = bitcast float %1108 to i32, !dbg !12
  %1110 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1109), !dbg !12
  %1111 = bitcast i32 %1110 to float, !dbg !12
  %1112 = select i1 %.not, float %1104, float %1111, !dbg !12
  %1113 = select i1 %.not, float %1111, float %1105, !dbg !12
  %1114 = select i1 %.not, float %1107, float %1106, !dbg !12
  %1115 = bitcast float %1114 to i32, !dbg !12
  %1116 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1115), !dbg !12
  %1117 = bitcast i32 %1116 to float, !dbg !12
  %1118 = select i1 %.not, float %1106, float %1117, !dbg !12
  %1119 = select i1 %.not, float %1117, float %1107, !dbg !12
  %1120 = select i1 %.not40, float %1118, float %1112, !dbg !12
  %1121 = bitcast float %1120 to i32, !dbg !12
  %1122 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1121), !dbg !12
  %1123 = bitcast i32 %1122 to float, !dbg !12
  %1124 = select i1 %.not40, float %1119, float %1113, !dbg !12
  %1125 = bitcast float %1124 to i32, !dbg !12
  %1126 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1125), !dbg !12
  %1127 = bitcast i32 %1126 to float, !dbg !12
  %1128 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !12
  %1129 = insertelement <4 x float> %1128, float %1113, i64 1, !dbg !12
  %1130 = insertelement <4 x float> %1129, float %1123, i64 2, !dbg !12
  %1131 = insertelement <4 x float> %1130, float %1127, i64 3, !dbg !12
  %1132 = insertelement <4 x float> poison, float %1123, i64 0, !dbg !12
  %1133 = insertelement <4 x float> %1132, float %1127, i64 1, !dbg !12
  %1134 = insertelement <4 x float> %1133, float %1118, i64 2, !dbg !12
  %1135 = insertelement <4 x float> %1134, float %1119, i64 3, !dbg !12
  %1136 = select <4 x i1> %1061, <4 x float> %1131, <4 x float> %1135, !dbg !12
  %1137 = extractelement <4 x float> %1011, i64 0, !dbg !12
  %1138 = extractelement <4 x float> %1011, i64 1, !dbg !12
  %1139 = extractelement <4 x float> %1011, i64 2, !dbg !12
  %1140 = extractelement <4 x float> %1011, i64 3, !dbg !12
  %1141 = select i1 %.not, float %1138, float %1137, !dbg !12
  %1142 = bitcast float %1141 to i32, !dbg !12
  %1143 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1142), !dbg !12
  %1144 = bitcast i32 %1143 to float, !dbg !12
  %1145 = select i1 %.not, float %1137, float %1144, !dbg !12
  %1146 = select i1 %.not, float %1144, float %1138, !dbg !12
  %1147 = select i1 %.not, float %1140, float %1139, !dbg !12
  %1148 = bitcast float %1147 to i32, !dbg !12
  %1149 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1148), !dbg !12
  %1150 = bitcast i32 %1149 to float, !dbg !12
  %1151 = select i1 %.not, float %1139, float %1150, !dbg !12
  %1152 = select i1 %.not, float %1150, float %1140, !dbg !12
  %1153 = select i1 %.not40, float %1151, float %1145, !dbg !12
  %1154 = bitcast float %1153 to i32, !dbg !12
  %1155 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1154), !dbg !12
  %1156 = bitcast i32 %1155 to float, !dbg !12
  %1157 = select i1 %.not40, float %1152, float %1146, !dbg !12
  %1158 = bitcast float %1157 to i32, !dbg !12
  %1159 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1158), !dbg !12
  %1160 = bitcast i32 %1159 to float, !dbg !12
  %1161 = insertelement <4 x float> poison, float %1145, i64 0, !dbg !12
  %1162 = insertelement <4 x float> %1161, float %1146, i64 1, !dbg !12
  %1163 = insertelement <4 x float> %1162, float %1156, i64 2, !dbg !12
  %1164 = insertelement <4 x float> %1163, float %1160, i64 3, !dbg !12
  %1165 = insertelement <4 x float> poison, float %1156, i64 0, !dbg !12
  %1166 = insertelement <4 x float> %1165, float %1160, i64 1, !dbg !12
  %1167 = insertelement <4 x float> %1166, float %1151, i64 2, !dbg !12
  %1168 = insertelement <4 x float> %1167, float %1152, i64 3, !dbg !12
  %1169 = select <4 x i1> %1061, <4 x float> %1164, <4 x float> %1168, !dbg !12
  %1170 = extractelement <4 x float> %936, i64 0, !dbg !12
  %1171 = extractelement <4 x float> %936, i64 1, !dbg !12
  %1172 = extractelement <4 x float> %936, i64 2, !dbg !12
  %1173 = extractelement <4 x float> %936, i64 3, !dbg !12
  %1174 = select i1 %.not, float %1171, float %1170, !dbg !12
  %1175 = bitcast float %1174 to i32, !dbg !12
  %1176 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1175), !dbg !12
  %1177 = bitcast i32 %1176 to float, !dbg !12
  %1178 = select i1 %.not, float %1170, float %1177, !dbg !12
  %1179 = select i1 %.not, float %1177, float %1171, !dbg !12
  %1180 = select i1 %.not, float %1173, float %1172, !dbg !12
  %1181 = bitcast float %1180 to i32, !dbg !12
  %1182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1181), !dbg !12
  %1183 = bitcast i32 %1182 to float, !dbg !12
  %1184 = select i1 %.not, float %1172, float %1183, !dbg !12
  %1185 = select i1 %.not, float %1183, float %1173, !dbg !12
  %1186 = select i1 %.not40, float %1184, float %1178, !dbg !12
  %1187 = bitcast float %1186 to i32, !dbg !12
  %1188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1187), !dbg !12
  %1189 = bitcast i32 %1188 to float, !dbg !12
  %1190 = select i1 %.not40, float %1185, float %1179, !dbg !12
  %1191 = bitcast float %1190 to i32, !dbg !12
  %1192 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1191), !dbg !12
  %1193 = bitcast i32 %1192 to float, !dbg !12
  %1194 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !12
  %1195 = insertelement <4 x float> %1194, float %1179, i64 1, !dbg !12
  %1196 = insertelement <4 x float> %1195, float %1189, i64 2, !dbg !12
  %1197 = insertelement <4 x float> %1196, float %1193, i64 3, !dbg !12
  %1198 = insertelement <4 x float> poison, float %1189, i64 0, !dbg !12
  %1199 = insertelement <4 x float> %1198, float %1193, i64 1, !dbg !12
  %1200 = insertelement <4 x float> %1199, float %1184, i64 2, !dbg !12
  %1201 = insertelement <4 x float> %1200, float %1185, i64 3, !dbg !12
  %1202 = select <4 x i1> %1061, <4 x float> %1197, <4 x float> %1201, !dbg !12
  %1203 = extractelement <4 x float> %938, i64 0, !dbg !12
  %1204 = extractelement <4 x float> %938, i64 1, !dbg !12
  %1205 = extractelement <4 x float> %938, i64 2, !dbg !12
  %1206 = extractelement <4 x float> %938, i64 3, !dbg !12
  %1207 = select i1 %.not, float %1204, float %1203, !dbg !12
  %1208 = bitcast float %1207 to i32, !dbg !12
  %1209 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1208), !dbg !12
  %1210 = bitcast i32 %1209 to float, !dbg !12
  %1211 = select i1 %.not, float %1203, float %1210, !dbg !12
  %1212 = select i1 %.not, float %1210, float %1204, !dbg !12
  %1213 = select i1 %.not, float %1206, float %1205, !dbg !12
  %1214 = bitcast float %1213 to i32, !dbg !12
  %1215 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1214), !dbg !12
  %1216 = bitcast i32 %1215 to float, !dbg !12
  %1217 = select i1 %.not, float %1205, float %1216, !dbg !12
  %1218 = select i1 %.not, float %1216, float %1206, !dbg !12
  %1219 = select i1 %.not40, float %1217, float %1211, !dbg !12
  %1220 = bitcast float %1219 to i32, !dbg !12
  %1221 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1220), !dbg !12
  %1222 = bitcast i32 %1221 to float, !dbg !12
  %1223 = select i1 %.not40, float %1218, float %1212, !dbg !12
  %1224 = bitcast float %1223 to i32, !dbg !12
  %1225 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1224), !dbg !12
  %1226 = bitcast i32 %1225 to float, !dbg !12
  %1227 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !12
  %1228 = insertelement <4 x float> %1227, float %1212, i64 1, !dbg !12
  %1229 = insertelement <4 x float> %1228, float %1222, i64 2, !dbg !12
  %1230 = insertelement <4 x float> %1229, float %1226, i64 3, !dbg !12
  %1231 = insertelement <4 x float> poison, float %1222, i64 0, !dbg !12
  %1232 = insertelement <4 x float> %1231, float %1226, i64 1, !dbg !12
  %1233 = insertelement <4 x float> %1232, float %1217, i64 2, !dbg !12
  %1234 = insertelement <4 x float> %1233, float %1218, i64 3, !dbg !12
  %1235 = select <4 x i1> %1061, <4 x float> %1230, <4 x float> %1234, !dbg !12
  %1236 = extractelement <4 x float> %939, i64 0, !dbg !12
  %1237 = extractelement <4 x float> %939, i64 1, !dbg !12
  %1238 = extractelement <4 x float> %939, i64 2, !dbg !12
  %1239 = extractelement <4 x float> %939, i64 3, !dbg !12
  %1240 = select i1 %.not, float %1237, float %1236, !dbg !12
  %1241 = bitcast float %1240 to i32, !dbg !12
  %1242 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1241), !dbg !12
  %1243 = bitcast i32 %1242 to float, !dbg !12
  %1244 = select i1 %.not, float %1236, float %1243, !dbg !12
  %1245 = select i1 %.not, float %1243, float %1237, !dbg !12
  %1246 = select i1 %.not, float %1239, float %1238, !dbg !12
  %1247 = bitcast float %1246 to i32, !dbg !12
  %1248 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1247), !dbg !12
  %1249 = bitcast i32 %1248 to float, !dbg !12
  %1250 = select i1 %.not, float %1238, float %1249, !dbg !12
  %1251 = select i1 %.not, float %1249, float %1239, !dbg !12
  %1252 = select i1 %.not40, float %1250, float %1244, !dbg !12
  %1253 = bitcast float %1252 to i32, !dbg !12
  %1254 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1253), !dbg !12
  %1255 = bitcast i32 %1254 to float, !dbg !12
  %1256 = select i1 %.not40, float %1251, float %1245, !dbg !12
  %1257 = bitcast float %1256 to i32, !dbg !12
  %1258 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1257), !dbg !12
  %1259 = bitcast i32 %1258 to float, !dbg !12
  %1260 = insertelement <4 x float> poison, float %1244, i64 0, !dbg !12
  %1261 = insertelement <4 x float> %1260, float %1245, i64 1, !dbg !12
  %1262 = insertelement <4 x float> %1261, float %1255, i64 2, !dbg !12
  %1263 = insertelement <4 x float> %1262, float %1259, i64 3, !dbg !12
  %1264 = insertelement <4 x float> poison, float %1255, i64 0, !dbg !12
  %1265 = insertelement <4 x float> %1264, float %1259, i64 1, !dbg !12
  %1266 = insertelement <4 x float> %1265, float %1250, i64 2, !dbg !12
  %1267 = insertelement <4 x float> %1266, float %1251, i64 3, !dbg !12
  %1268 = select <4 x i1> %1061, <4 x float> %1263, <4 x float> %1267, !dbg !12
  %1269 = extractelement <4 x float> %940, i64 0, !dbg !12
  %1270 = extractelement <4 x float> %940, i64 1, !dbg !12
  %1271 = extractelement <4 x float> %940, i64 2, !dbg !12
  %1272 = extractelement <4 x float> %940, i64 3, !dbg !12
  %1273 = select i1 %.not, float %1270, float %1269, !dbg !12
  %1274 = bitcast float %1273 to i32, !dbg !12
  %1275 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1274), !dbg !12
  %1276 = bitcast i32 %1275 to float, !dbg !12
  %1277 = select i1 %.not, float %1269, float %1276, !dbg !12
  %1278 = select i1 %.not, float %1276, float %1270, !dbg !12
  %1279 = select i1 %.not, float %1272, float %1271, !dbg !12
  %1280 = bitcast float %1279 to i32, !dbg !12
  %1281 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1280), !dbg !12
  %1282 = bitcast i32 %1281 to float, !dbg !12
  %1283 = select i1 %.not, float %1271, float %1282, !dbg !12
  %1284 = select i1 %.not, float %1282, float %1272, !dbg !12
  %1285 = select i1 %.not40, float %1283, float %1277, !dbg !12
  %1286 = bitcast float %1285 to i32, !dbg !12
  %1287 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1286), !dbg !12
  %1288 = bitcast i32 %1287 to float, !dbg !12
  %1289 = select i1 %.not40, float %1284, float %1278, !dbg !12
  %1290 = bitcast float %1289 to i32, !dbg !12
  %1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1290), !dbg !12
  %1292 = bitcast i32 %1291 to float, !dbg !12
  %1293 = insertelement <4 x float> poison, float %1277, i64 0, !dbg !12
  %1294 = insertelement <4 x float> %1293, float %1278, i64 1, !dbg !12
  %1295 = insertelement <4 x float> %1294, float %1288, i64 2, !dbg !12
  %1296 = insertelement <4 x float> %1295, float %1292, i64 3, !dbg !12
  %1297 = insertelement <4 x float> poison, float %1288, i64 0, !dbg !12
  %1298 = insertelement <4 x float> %1297, float %1292, i64 1, !dbg !12
  %1299 = insertelement <4 x float> %1298, float %1283, i64 2, !dbg !12
  %1300 = insertelement <4 x float> %1299, float %1284, i64 3, !dbg !12
  %1301 = select <4 x i1> %1061, <4 x float> %1296, <4 x float> %1300, !dbg !12
  %1302 = extractelement <4 x float> %863, i64 0, !dbg !12
  %1303 = extractelement <4 x float> %863, i64 1, !dbg !12
  %1304 = extractelement <4 x float> %863, i64 2, !dbg !12
  %1305 = extractelement <4 x float> %863, i64 3, !dbg !12
  %1306 = select i1 %.not, float %1303, float %1302, !dbg !12
  %1307 = bitcast float %1306 to i32, !dbg !12
  %1308 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1307), !dbg !12
  %1309 = bitcast i32 %1308 to float, !dbg !12
  %1310 = select i1 %.not, float %1302, float %1309, !dbg !12
  %1311 = select i1 %.not, float %1309, float %1303, !dbg !12
  %1312 = select i1 %.not, float %1305, float %1304, !dbg !12
  %1313 = bitcast float %1312 to i32, !dbg !12
  %1314 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1313), !dbg !12
  %1315 = bitcast i32 %1314 to float, !dbg !12
  %1316 = select i1 %.not, float %1304, float %1315, !dbg !12
  %1317 = select i1 %.not, float %1315, float %1305, !dbg !12
  %1318 = select i1 %.not40, float %1316, float %1310, !dbg !12
  %1319 = bitcast float %1318 to i32, !dbg !12
  %1320 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1319), !dbg !12
  %1321 = bitcast i32 %1320 to float, !dbg !12
  %1322 = select i1 %.not40, float %1317, float %1311, !dbg !12
  %1323 = bitcast float %1322 to i32, !dbg !12
  %1324 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1323), !dbg !12
  %1325 = bitcast i32 %1324 to float, !dbg !12
  %1326 = insertelement <4 x float> poison, float %1310, i64 0, !dbg !12
  %1327 = insertelement <4 x float> %1326, float %1311, i64 1, !dbg !12
  %1328 = insertelement <4 x float> %1327, float %1321, i64 2, !dbg !12
  %1329 = insertelement <4 x float> %1328, float %1325, i64 3, !dbg !12
  %1330 = insertelement <4 x float> poison, float %1321, i64 0, !dbg !12
  %1331 = insertelement <4 x float> %1330, float %1325, i64 1, !dbg !12
  %1332 = insertelement <4 x float> %1331, float %1316, i64 2, !dbg !12
  %1333 = insertelement <4 x float> %1332, float %1317, i64 3, !dbg !12
  %1334 = select <4 x i1> %1061, <4 x float> %1329, <4 x float> %1333, !dbg !12
  %1335 = extractelement <4 x float> %865, i64 0, !dbg !12
  %1336 = extractelement <4 x float> %865, i64 1, !dbg !12
  %1337 = extractelement <4 x float> %865, i64 2, !dbg !12
  %1338 = extractelement <4 x float> %865, i64 3, !dbg !12
  %1339 = select i1 %.not, float %1336, float %1335, !dbg !12
  %1340 = bitcast float %1339 to i32, !dbg !12
  %1341 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1340), !dbg !12
  %1342 = bitcast i32 %1341 to float, !dbg !12
  %1343 = select i1 %.not, float %1335, float %1342, !dbg !12
  %1344 = select i1 %.not, float %1342, float %1336, !dbg !12
  %1345 = select i1 %.not, float %1338, float %1337, !dbg !12
  %1346 = bitcast float %1345 to i32, !dbg !12
  %1347 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1346), !dbg !12
  %1348 = bitcast i32 %1347 to float, !dbg !12
  %1349 = select i1 %.not, float %1337, float %1348, !dbg !12
  %1350 = select i1 %.not, float %1348, float %1338, !dbg !12
  %1351 = select i1 %.not40, float %1349, float %1343, !dbg !12
  %1352 = bitcast float %1351 to i32, !dbg !12
  %1353 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1352), !dbg !12
  %1354 = bitcast i32 %1353 to float, !dbg !12
  %1355 = select i1 %.not40, float %1350, float %1344, !dbg !12
  %1356 = bitcast float %1355 to i32, !dbg !12
  %1357 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1356), !dbg !12
  %1358 = bitcast i32 %1357 to float, !dbg !12
  %1359 = insertelement <4 x float> poison, float %1343, i64 0, !dbg !12
  %1360 = insertelement <4 x float> %1359, float %1344, i64 1, !dbg !12
  %1361 = insertelement <4 x float> %1360, float %1354, i64 2, !dbg !12
  %1362 = insertelement <4 x float> %1361, float %1358, i64 3, !dbg !12
  %1363 = insertelement <4 x float> poison, float %1354, i64 0, !dbg !12
  %1364 = insertelement <4 x float> %1363, float %1358, i64 1, !dbg !12
  %1365 = insertelement <4 x float> %1364, float %1349, i64 2, !dbg !12
  %1366 = insertelement <4 x float> %1365, float %1350, i64 3, !dbg !12
  %1367 = select <4 x i1> %1061, <4 x float> %1362, <4 x float> %1366, !dbg !12
  %1368 = extractelement <4 x float> %866, i64 0, !dbg !12
  %1369 = extractelement <4 x float> %866, i64 1, !dbg !12
  %1370 = extractelement <4 x float> %866, i64 2, !dbg !12
  %1371 = extractelement <4 x float> %866, i64 3, !dbg !12
  %1372 = select i1 %.not, float %1369, float %1368, !dbg !12
  %1373 = bitcast float %1372 to i32, !dbg !12
  %1374 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1373), !dbg !12
  %1375 = bitcast i32 %1374 to float, !dbg !12
  %1376 = select i1 %.not, float %1368, float %1375, !dbg !12
  %1377 = select i1 %.not, float %1375, float %1369, !dbg !12
  %1378 = select i1 %.not, float %1371, float %1370, !dbg !12
  %1379 = bitcast float %1378 to i32, !dbg !12
  %1380 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1379), !dbg !12
  %1381 = bitcast i32 %1380 to float, !dbg !12
  %1382 = select i1 %.not, float %1370, float %1381, !dbg !12
  %1383 = select i1 %.not, float %1381, float %1371, !dbg !12
  %1384 = select i1 %.not40, float %1382, float %1376, !dbg !12
  %1385 = bitcast float %1384 to i32, !dbg !12
  %1386 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1385), !dbg !12
  %1387 = bitcast i32 %1386 to float, !dbg !12
  %1388 = select i1 %.not40, float %1383, float %1377, !dbg !12
  %1389 = bitcast float %1388 to i32, !dbg !12
  %1390 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1389), !dbg !12
  %1391 = bitcast i32 %1390 to float, !dbg !12
  %1392 = insertelement <4 x float> poison, float %1376, i64 0, !dbg !12
  %1393 = insertelement <4 x float> %1392, float %1377, i64 1, !dbg !12
  %1394 = insertelement <4 x float> %1393, float %1387, i64 2, !dbg !12
  %1395 = insertelement <4 x float> %1394, float %1391, i64 3, !dbg !12
  %1396 = insertelement <4 x float> poison, float %1387, i64 0, !dbg !12
  %1397 = insertelement <4 x float> %1396, float %1391, i64 1, !dbg !12
  %1398 = insertelement <4 x float> %1397, float %1382, i64 2, !dbg !12
  %1399 = insertelement <4 x float> %1398, float %1383, i64 3, !dbg !12
  %1400 = select <4 x i1> %1061, <4 x float> %1395, <4 x float> %1399, !dbg !12
  %1401 = extractelement <4 x float> %867, i64 0, !dbg !12
  %1402 = extractelement <4 x float> %867, i64 1, !dbg !12
  %1403 = extractelement <4 x float> %867, i64 2, !dbg !12
  %1404 = extractelement <4 x float> %867, i64 3, !dbg !12
  %1405 = select i1 %.not, float %1402, float %1401, !dbg !12
  %1406 = bitcast float %1405 to i32, !dbg !12
  %1407 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1406), !dbg !12
  %1408 = bitcast i32 %1407 to float, !dbg !12
  %1409 = select i1 %.not, float %1401, float %1408, !dbg !12
  %1410 = select i1 %.not, float %1408, float %1402, !dbg !12
  %1411 = select i1 %.not, float %1404, float %1403, !dbg !12
  %1412 = bitcast float %1411 to i32, !dbg !12
  %1413 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1412), !dbg !12
  %1414 = bitcast i32 %1413 to float, !dbg !12
  %1415 = select i1 %.not, float %1403, float %1414, !dbg !12
  %1416 = select i1 %.not, float %1414, float %1404, !dbg !12
  %1417 = select i1 %.not40, float %1415, float %1409, !dbg !12
  %1418 = bitcast float %1417 to i32, !dbg !12
  %1419 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1418), !dbg !12
  %1420 = bitcast i32 %1419 to float, !dbg !12
  %1421 = select i1 %.not40, float %1416, float %1410, !dbg !12
  %1422 = bitcast float %1421 to i32, !dbg !12
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1422), !dbg !12
  %1424 = bitcast i32 %1423 to float, !dbg !12
  %1425 = insertelement <4 x float> poison, float %1409, i64 0, !dbg !12
  %1426 = insertelement <4 x float> %1425, float %1410, i64 1, !dbg !12
  %1427 = insertelement <4 x float> %1426, float %1420, i64 2, !dbg !12
  %1428 = insertelement <4 x float> %1427, float %1424, i64 3, !dbg !12
  %1429 = insertelement <4 x float> poison, float %1420, i64 0, !dbg !12
  %1430 = insertelement <4 x float> %1429, float %1424, i64 1, !dbg !12
  %1431 = insertelement <4 x float> %1430, float %1415, i64 2, !dbg !12
  %1432 = insertelement <4 x float> %1431, float %1416, i64 3, !dbg !12
  %1433 = select <4 x i1> %1061, <4 x float> %1428, <4 x float> %1432, !dbg !12
  %1434 = extractelement <4 x float> %771, i64 0, !dbg !12
  %1435 = extractelement <4 x float> %771, i64 1, !dbg !12
  %1436 = extractelement <4 x float> %771, i64 2, !dbg !12
  %1437 = extractelement <4 x float> %771, i64 3, !dbg !12
  %1438 = select i1 %.not, float %1435, float %1434, !dbg !12
  %1439 = bitcast float %1438 to i32, !dbg !12
  %1440 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1439), !dbg !12
  %1441 = bitcast i32 %1440 to float, !dbg !12
  %1442 = select i1 %.not, float %1434, float %1441, !dbg !12
  %1443 = select i1 %.not, float %1441, float %1435, !dbg !12
  %1444 = select i1 %.not, float %1437, float %1436, !dbg !12
  %1445 = bitcast float %1444 to i32, !dbg !12
  %1446 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1445), !dbg !12
  %1447 = bitcast i32 %1446 to float, !dbg !12
  %1448 = select i1 %.not, float %1436, float %1447, !dbg !12
  %1449 = select i1 %.not, float %1447, float %1437, !dbg !12
  %1450 = select i1 %.not40, float %1448, float %1442, !dbg !12
  %1451 = bitcast float %1450 to i32, !dbg !12
  %1452 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1451), !dbg !12
  %1453 = select i1 %.not40, float %1449, float %1443, !dbg !12
  %1454 = bitcast float %1453 to i32, !dbg !12
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1454), !dbg !12
  %1456 = extractelement <4 x float> %773, i64 0, !dbg !12
  %1457 = extractelement <4 x float> %773, i64 1, !dbg !12
  %1458 = extractelement <4 x float> %773, i64 2, !dbg !12
  %1459 = extractelement <4 x float> %773, i64 3, !dbg !12
  %1460 = select i1 %.not, float %1457, float %1456, !dbg !12
  %1461 = bitcast float %1460 to i32, !dbg !12
  %1462 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1461), !dbg !12
  %1463 = bitcast i32 %1462 to float, !dbg !12
  %1464 = select i1 %.not, float %1456, float %1463, !dbg !12
  %1465 = select i1 %.not, float %1463, float %1457, !dbg !12
  %1466 = select i1 %.not, float %1459, float %1458, !dbg !12
  %1467 = bitcast float %1466 to i32, !dbg !12
  %1468 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1467), !dbg !12
  %1469 = bitcast i32 %1468 to float, !dbg !12
  %1470 = select i1 %.not, float %1458, float %1469, !dbg !12
  %1471 = select i1 %.not, float %1469, float %1459, !dbg !12
  %1472 = select i1 %.not40, float %1470, float %1464, !dbg !12
  %1473 = bitcast float %1472 to i32, !dbg !12
  %1474 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1473), !dbg !12
  %1475 = select i1 %.not40, float %1471, float %1465, !dbg !12
  %1476 = bitcast float %1475 to i32, !dbg !12
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1476), !dbg !12
  %1478 = extractelement <4 x float> %793, i64 0, !dbg !12
  %1479 = extractelement <4 x float> %793, i64 1, !dbg !12
  %1480 = extractelement <4 x float> %793, i64 2, !dbg !12
  %1481 = extractelement <4 x float> %793, i64 3, !dbg !12
  %1482 = select i1 %.not, float %1479, float %1478, !dbg !12
  %1483 = bitcast float %1482 to i32, !dbg !12
  %1484 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1483), !dbg !12
  %1485 = bitcast i32 %1484 to float, !dbg !12
  %1486 = select i1 %.not, float %1478, float %1485, !dbg !12
  %1487 = select i1 %.not, float %1485, float %1479, !dbg !12
  %1488 = select i1 %.not, float %1481, float %1480, !dbg !12
  %1489 = bitcast float %1488 to i32, !dbg !12
  %1490 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1489), !dbg !12
  %1491 = bitcast i32 %1490 to float, !dbg !12
  %1492 = select i1 %.not, float %1480, float %1491, !dbg !12
  %1493 = select i1 %.not, float %1491, float %1481, !dbg !12
  %1494 = select i1 %.not40, float %1492, float %1486, !dbg !12
  %1495 = bitcast float %1494 to i32, !dbg !12
  %1496 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1495), !dbg !12
  %1497 = select i1 %.not40, float %1493, float %1487, !dbg !12
  %1498 = bitcast float %1497 to i32, !dbg !12
  %1499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1498), !dbg !12
  %1500 = extractelement <4 x float> %794, i64 0, !dbg !12
  %1501 = extractelement <4 x float> %794, i64 1, !dbg !12
  %1502 = extractelement <4 x float> %794, i64 2, !dbg !12
  %1503 = extractelement <4 x float> %794, i64 3, !dbg !12
  %1504 = select i1 %.not, float %1501, float %1500, !dbg !12
  %1505 = bitcast float %1504 to i32, !dbg !12
  %1506 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1505), !dbg !12
  %1507 = bitcast i32 %1506 to float, !dbg !12
  %1508 = select i1 %.not, float %1500, float %1507, !dbg !12
  %1509 = select i1 %.not, float %1507, float %1501, !dbg !12
  %1510 = select i1 %.not, float %1503, float %1502, !dbg !12
  %1511 = bitcast float %1510 to i32, !dbg !12
  %1512 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1511), !dbg !12
  %1513 = bitcast i32 %1512 to float, !dbg !12
  %1514 = select i1 %.not, float %1502, float %1513, !dbg !12
  %1515 = select i1 %.not, float %1513, float %1503, !dbg !12
  %1516 = select i1 %.not40, float %1514, float %1508, !dbg !12
  %1517 = bitcast float %1516 to i32, !dbg !12
  %1518 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1517), !dbg !12
  %1519 = select i1 %.not40, float %1515, float %1509, !dbg !12
  %1520 = bitcast float %1519 to i32, !dbg !12
  %1521 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1520), !dbg !12
  %1522 = insertelement <2 x i32> poison, i32 %1033, i64 0, !dbg !12
  %1523 = insertelement <2 x i32> %1522, i32 %1031, i64 1, !dbg !12
  %1524 = bitcast <2 x i32> %1523 to <2 x float>, !dbg !12
  %1525 = insertelement <2 x float> poison, float %k0.sum.anchor1884, i64 0, !dbg !12
  %1526 = insertelement <2 x float> %1525, float %k0.sum.anchor1892, i64 1, !dbg !12
  %1527 = fadd <2 x float> %1526, %1524, !dbg !12
  %bc = bitcast <2 x float> %1527 to <2 x i32>, !dbg !12
  %1528 = extractelement <2 x i32> %bc, i64 1, !dbg !12
  %1529 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1528), !dbg !12
  %bc182 = bitcast <2 x float> %1527 to <2 x i32>, !dbg !12
  %1530 = extractelement <2 x i32> %bc182, i64 0, !dbg !12
  %1531 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1530), !dbg !12
  %1532 = insertelement <2 x i32> poison, i32 %1531, i64 0, !dbg !12
  %1533 = insertelement <2 x i32> %1532, i32 %1529, i64 1, !dbg !12
  %1534 = bitcast <2 x i32> %1533 to <2 x float>, !dbg !12
  %1535 = fadd <2 x float> %1527, %1534, !dbg !12
  %1536 = fdiv arcp <2 x float> splat (float 1.000000e+00), %1535, !dbg !45
  %1537 = shufflevector <2 x float> %1536, <2 x float> poison, <4 x i32> zeroinitializer, !dbg !45
  %1538 = shufflevector <2 x float> %1536, <2 x float> poison, <4 x i32> <i32 1, i32 1, i32 1, i32 1>, !dbg !45
  %1539 = bitcast i32 %1518 to float, !dbg !12
  %1540 = bitcast i32 %1521 to float, !dbg !12
  %1541 = insertelement <4 x float> poison, float %1508, i64 0, !dbg !12
  %1542 = insertelement <4 x float> %1541, float %1509, i64 1, !dbg !12
  %1543 = insertelement <4 x float> %1542, float %1539, i64 2, !dbg !12
  %1544 = insertelement <4 x float> %1543, float %1540, i64 3, !dbg !12
  %1545 = insertelement <4 x float> poison, float %1539, i64 0, !dbg !12
  %1546 = insertelement <4 x float> %1545, float %1540, i64 1, !dbg !12
  %1547 = insertelement <4 x float> %1546, float %1514, i64 2, !dbg !12
  %1548 = insertelement <4 x float> %1547, float %1515, i64 3, !dbg !12
  %1549 = select <4 x i1> %1061, <4 x float> %1544, <4 x float> %1548, !dbg !12
  %1550 = fmul <4 x float> %1538, %1549, !dbg !45
  %1551 = fptrunc <4 x float> %1550 to <4 x half>, !dbg !46
  %1552 = extractelement <4 x half> %1551, i64 3, !dbg !46
  %1553 = extractelement <4 x half> %1551, i64 2, !dbg !46
  %1554 = extractelement <4 x half> %1551, i64 1, !dbg !46
  %1555 = extractelement <4 x half> %1551, i64 0, !dbg !46
  %1556 = bitcast i32 %1496 to float, !dbg !12
  %1557 = bitcast i32 %1499 to float, !dbg !12
  %1558 = insertelement <4 x float> poison, float %1486, i64 0, !dbg !12
  %1559 = insertelement <4 x float> %1558, float %1487, i64 1, !dbg !12
  %1560 = insertelement <4 x float> %1559, float %1556, i64 2, !dbg !12
  %1561 = insertelement <4 x float> %1560, float %1557, i64 3, !dbg !12
  %1562 = insertelement <4 x float> poison, float %1556, i64 0, !dbg !12
  %1563 = insertelement <4 x float> %1562, float %1557, i64 1, !dbg !12
  %1564 = insertelement <4 x float> %1563, float %1492, i64 2, !dbg !12
  %1565 = insertelement <4 x float> %1564, float %1493, i64 3, !dbg !12
  %1566 = select <4 x i1> %1061, <4 x float> %1561, <4 x float> %1565, !dbg !12
  %1567 = fmul <4 x float> %1538, %1566, !dbg !45
  %1568 = fptrunc <4 x float> %1567 to <4 x half>, !dbg !46
  %1569 = extractelement <4 x half> %1568, i64 3, !dbg !46
  %1570 = extractelement <4 x half> %1568, i64 2, !dbg !46
  %1571 = extractelement <4 x half> %1568, i64 1, !dbg !46
  %1572 = extractelement <4 x half> %1568, i64 0, !dbg !46
  %1573 = bitcast i32 %1474 to float, !dbg !12
  %1574 = bitcast i32 %1477 to float, !dbg !12
  %1575 = insertelement <4 x float> poison, float %1464, i64 0, !dbg !12
  %1576 = insertelement <4 x float> %1575, float %1465, i64 1, !dbg !12
  %1577 = insertelement <4 x float> %1576, float %1573, i64 2, !dbg !12
  %1578 = insertelement <4 x float> %1577, float %1574, i64 3, !dbg !12
  %1579 = insertelement <4 x float> poison, float %1573, i64 0, !dbg !12
  %1580 = insertelement <4 x float> %1579, float %1574, i64 1, !dbg !12
  %1581 = insertelement <4 x float> %1580, float %1470, i64 2, !dbg !12
  %1582 = insertelement <4 x float> %1581, float %1471, i64 3, !dbg !12
  %1583 = select <4 x i1> %1061, <4 x float> %1578, <4 x float> %1582, !dbg !12
  %1584 = fmul <4 x float> %1537, %1583, !dbg !45
  %1585 = fptrunc <4 x float> %1584 to <4 x half>, !dbg !46
  %1586 = extractelement <4 x half> %1585, i64 3, !dbg !46
  %1587 = extractelement <4 x half> %1585, i64 2, !dbg !46
  %1588 = extractelement <4 x half> %1585, i64 1, !dbg !46
  %1589 = extractelement <4 x half> %1585, i64 0, !dbg !46
  %1590 = bitcast i32 %1452 to float, !dbg !12
  %1591 = bitcast i32 %1455 to float, !dbg !12
  %1592 = insertelement <4 x float> poison, float %1442, i64 0, !dbg !12
  %1593 = insertelement <4 x float> %1592, float %1443, i64 1, !dbg !12
  %1594 = insertelement <4 x float> %1593, float %1590, i64 2, !dbg !12
  %1595 = insertelement <4 x float> %1594, float %1591, i64 3, !dbg !12
  %1596 = insertelement <4 x float> poison, float %1590, i64 0, !dbg !12
  %1597 = insertelement <4 x float> %1596, float %1591, i64 1, !dbg !12
  %1598 = insertelement <4 x float> %1597, float %1448, i64 2, !dbg !12
  %1599 = insertelement <4 x float> %1598, float %1449, i64 3, !dbg !12
  %1600 = select <4 x i1> %1061, <4 x float> %1595, <4 x float> %1599, !dbg !12
  %1601 = fmul <4 x float> %1537, %1600, !dbg !45
  %1602 = fptrunc <4 x float> %1601 to <4 x half>, !dbg !46
  %1603 = extractelement <4 x half> %1602, i64 3, !dbg !46
  %1604 = extractelement <4 x half> %1602, i64 2, !dbg !46
  %1605 = extractelement <4 x half> %1602, i64 1, !dbg !46
  %1606 = extractelement <4 x half> %1602, i64 0, !dbg !46
  %address.thread883 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !47, !range !10, !frisk.register_lifetimes !5
  %address.local884 = zext nneg i32 %address.thread883 to i64, !dbg !9
  %1607 = lshr i64 %address.local884, 2, !dbg !47
  %address.local887 = and i64 %1607, 48, !dbg !47
  %address.local889 = and i64 %address.local884, 240, !dbg !47
  %address.local888 = add i64 %34, %address.local884, !dbg !47
  %address.local890 = sub i64 %address.local888, %address.local889, !dbg !47
  %address.local891 = add i64 %address.local890, %address.local887, !dbg !47
  %1608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %45, !dbg !47
  %address.local894.idx = shl nuw nsw i64 %address.local891, 8, !dbg !47
  %address.local894 = getelementptr inbounds nuw i8, ptr addrspace(1) %1608, i64 %address.local894.idx, !dbg !47
  %address.local896 = and i64 %1607, 12, !dbg !11
  %address.local897 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local896, !dbg !47
  store half %1606, ptr addrspace(1) %address.local897, align 2, !dbg !47
  %address.local899 = or disjoint i64 %address.local896, 1, !dbg !47
  %address.local900 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local899, !dbg !47
  store half %1605, ptr addrspace(1) %address.local900, align 2, !dbg !47
  %address.local902 = or disjoint i64 %address.local896, 2, !dbg !47
  %address.local903 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local902, !dbg !47
  store half %1604, ptr addrspace(1) %address.local903, align 2, !dbg !47
  %address.local905 = or disjoint i64 %address.local896, 3, !dbg !47
  %address.local906 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local905, !dbg !47
  store half %1603, ptr addrspace(1) %address.local906, align 2, !dbg !47
  %address.local908 = or disjoint i64 %address.local896, 16, !dbg !47
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local908, !dbg !47
  store half %1589, ptr addrspace(1) %address.local909, align 2, !dbg !47
  %address.local911 = or disjoint i64 %address.local896, 17, !dbg !47
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local911, !dbg !47
  store half %1588, ptr addrspace(1) %address.local912, align 2, !dbg !47
  %address.local914 = or disjoint i64 %address.local896, 18, !dbg !47
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local914, !dbg !47
  store half %1587, ptr addrspace(1) %address.local915, align 2, !dbg !47
  %address.local917 = or disjoint i64 %address.local896, 19, !dbg !47
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local917, !dbg !47
  store half %1586, ptr addrspace(1) %address.local918, align 2, !dbg !47
  %address.local922 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local894, i64 16384, !dbg !47
  %address.local923 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local896, !dbg !47
  store half %1572, ptr addrspace(1) %address.local923, align 2, !dbg !47
  %address.local925 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local899, !dbg !47
  store half %1571, ptr addrspace(1) %address.local925, align 2, !dbg !47
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local902, !dbg !47
  store half %1570, ptr addrspace(1) %address.local927, align 2, !dbg !47
  %address.local929 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local905, !dbg !47
  store half %1569, ptr addrspace(1) %address.local929, align 2, !dbg !47
  %address.local931 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local908, !dbg !47
  store half %1555, ptr addrspace(1) %address.local931, align 2, !dbg !47
  %address.local933 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local911, !dbg !47
  store half %1554, ptr addrspace(1) %address.local933, align 2, !dbg !47
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local914, !dbg !47
  store half %1553, ptr addrspace(1) %address.local935, align 2, !dbg !47
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local917, !dbg !47
  store half %1552, ptr addrspace(1) %address.local937, align 2, !dbg !47
  %1609 = fmul <4 x float> %1538, %1433, !dbg !45
  %1610 = fptrunc <4 x float> %1609 to <4 x half>, !dbg !46
  %1611 = extractelement <4 x half> %1610, i64 3, !dbg !46
  %1612 = extractelement <4 x half> %1610, i64 2, !dbg !46
  %1613 = extractelement <4 x half> %1610, i64 1, !dbg !46
  %1614 = extractelement <4 x half> %1610, i64 0, !dbg !46
  %1615 = fmul <4 x float> %1538, %1400, !dbg !45
  %1616 = fptrunc <4 x float> %1615 to <4 x half>, !dbg !46
  %1617 = extractelement <4 x half> %1616, i64 3, !dbg !46
  %1618 = extractelement <4 x half> %1616, i64 2, !dbg !46
  %1619 = extractelement <4 x half> %1616, i64 1, !dbg !46
  %1620 = extractelement <4 x half> %1616, i64 0, !dbg !46
  %1621 = fmul <4 x float> %1537, %1367, !dbg !45
  %1622 = fptrunc <4 x float> %1621 to <4 x half>, !dbg !46
  %1623 = extractelement <4 x half> %1622, i64 3, !dbg !46
  %1624 = extractelement <4 x half> %1622, i64 2, !dbg !46
  %1625 = extractelement <4 x half> %1622, i64 1, !dbg !46
  %1626 = extractelement <4 x half> %1622, i64 0, !dbg !46
  %1627 = fmul <4 x float> %1537, %1334, !dbg !45
  %1628 = fptrunc <4 x float> %1627 to <4 x half>, !dbg !46
  %1629 = extractelement <4 x half> %1628, i64 3, !dbg !46
  %1630 = extractelement <4 x half> %1628, i64 2, !dbg !46
  %1631 = extractelement <4 x half> %1628, i64 1, !dbg !46
  %1632 = extractelement <4 x half> %1628, i64 0, !dbg !46
  %address.local952 = or disjoint i64 %address.local896, 32, !dbg !47
  %address.local953 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local952, !dbg !47
  store half %1632, ptr addrspace(1) %address.local953, align 2, !dbg !47
  %address.local955 = or disjoint i64 %address.local896, 33, !dbg !47
  %address.local956 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local955, !dbg !47
  store half %1631, ptr addrspace(1) %address.local956, align 2, !dbg !47
  %address.local958 = or disjoint i64 %address.local896, 34, !dbg !47
  %address.local959 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local958, !dbg !47
  store half %1630, ptr addrspace(1) %address.local959, align 2, !dbg !47
  %address.local961 = or disjoint i64 %address.local896, 35, !dbg !47
  %address.local962 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local961, !dbg !47
  store half %1629, ptr addrspace(1) %address.local962, align 2, !dbg !47
  %address.local964 = or disjoint i64 %address.local896, 48, !dbg !47
  %address.local965 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local964, !dbg !47
  store half %1626, ptr addrspace(1) %address.local965, align 2, !dbg !47
  %address.local967 = or disjoint i64 %address.local896, 49, !dbg !47
  %address.local968 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local967, !dbg !47
  store half %1625, ptr addrspace(1) %address.local968, align 2, !dbg !47
  %address.local970 = or disjoint i64 %address.local896, 50, !dbg !47
  %address.local971 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local970, !dbg !47
  store half %1624, ptr addrspace(1) %address.local971, align 2, !dbg !47
  %address.local973 = or i64 %1607, 51, !dbg !47
  %address.local974 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local973, !dbg !47
  store half %1623, ptr addrspace(1) %address.local974, align 2, !dbg !47
  %address.local979 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local952, !dbg !47
  store half %1620, ptr addrspace(1) %address.local979, align 2, !dbg !47
  %address.local981 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local955, !dbg !47
  store half %1619, ptr addrspace(1) %address.local981, align 2, !dbg !47
  %address.local983 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local958, !dbg !47
  store half %1618, ptr addrspace(1) %address.local983, align 2, !dbg !47
  %address.local985 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local961, !dbg !47
  store half %1617, ptr addrspace(1) %address.local985, align 2, !dbg !47
  %address.local987 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local964, !dbg !47
  store half %1614, ptr addrspace(1) %address.local987, align 2, !dbg !47
  %address.local989 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local967, !dbg !47
  store half %1613, ptr addrspace(1) %address.local989, align 2, !dbg !47
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local970, !dbg !47
  store half %1612, ptr addrspace(1) %address.local991, align 2, !dbg !47
  %address.local993 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local973, !dbg !47
  store half %1611, ptr addrspace(1) %address.local993, align 2, !dbg !47
  %1633 = fmul <4 x float> %1538, %1301, !dbg !45
  %1634 = fptrunc <4 x float> %1633 to <4 x half>, !dbg !46
  %1635 = extractelement <4 x half> %1634, i64 3, !dbg !46
  %1636 = extractelement <4 x half> %1634, i64 2, !dbg !46
  %1637 = extractelement <4 x half> %1634, i64 1, !dbg !46
  %1638 = extractelement <4 x half> %1634, i64 0, !dbg !46
  %1639 = fmul <4 x float> %1538, %1268, !dbg !45
  %1640 = fptrunc <4 x float> %1639 to <4 x half>, !dbg !46
  %1641 = extractelement <4 x half> %1640, i64 3, !dbg !46
  %1642 = extractelement <4 x half> %1640, i64 2, !dbg !46
  %1643 = extractelement <4 x half> %1640, i64 1, !dbg !46
  %1644 = extractelement <4 x half> %1640, i64 0, !dbg !46
  %1645 = fmul <4 x float> %1537, %1235, !dbg !45
  %1646 = fptrunc <4 x float> %1645 to <4 x half>, !dbg !46
  %1647 = extractelement <4 x half> %1646, i64 3, !dbg !46
  %1648 = extractelement <4 x half> %1646, i64 2, !dbg !46
  %1649 = extractelement <4 x half> %1646, i64 1, !dbg !46
  %1650 = extractelement <4 x half> %1646, i64 0, !dbg !46
  %1651 = fmul <4 x float> %1537, %1202, !dbg !45
  %1652 = fptrunc <4 x float> %1651 to <4 x half>, !dbg !46
  %1653 = extractelement <4 x half> %1652, i64 3, !dbg !46
  %1654 = extractelement <4 x half> %1652, i64 2, !dbg !46
  %1655 = extractelement <4 x half> %1652, i64 1, !dbg !46
  %1656 = extractelement <4 x half> %1652, i64 0, !dbg !46
  %address.local1008 = or disjoint i64 %address.local896, 64, !dbg !47
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1008, !dbg !47
  store half %1656, ptr addrspace(1) %address.local1009, align 2, !dbg !47
  %address.local1011 = or disjoint i64 %address.local896, 65, !dbg !47
  %address.local1012 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1011, !dbg !47
  store half %1655, ptr addrspace(1) %address.local1012, align 2, !dbg !47
  %address.local1014 = or disjoint i64 %address.local896, 66, !dbg !47
  %address.local1015 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1014, !dbg !47
  store half %1654, ptr addrspace(1) %address.local1015, align 2, !dbg !47
  %address.local1017 = or disjoint i64 %address.local896, 67, !dbg !47
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1017, !dbg !47
  store half %1653, ptr addrspace(1) %address.local1018, align 2, !dbg !47
  %address.local1020 = or disjoint i64 %address.local896, 80, !dbg !47
  %address.local1021 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1020, !dbg !47
  store half %1650, ptr addrspace(1) %address.local1021, align 2, !dbg !47
  %address.local1023 = or disjoint i64 %address.local896, 81, !dbg !47
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1023, !dbg !47
  store half %1649, ptr addrspace(1) %address.local1024, align 2, !dbg !47
  %address.local1026 = or disjoint i64 %address.local896, 82, !dbg !47
  %address.local1027 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1026, !dbg !47
  store half %1648, ptr addrspace(1) %address.local1027, align 2, !dbg !47
  %address.local1029 = or disjoint i64 %address.local896, 83, !dbg !47
  %address.local1030 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1029, !dbg !47
  store half %1647, ptr addrspace(1) %address.local1030, align 2, !dbg !47
  %address.local1035 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1008, !dbg !47
  store half %1644, ptr addrspace(1) %address.local1035, align 2, !dbg !47
  %address.local1037 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1011, !dbg !47
  store half %1643, ptr addrspace(1) %address.local1037, align 2, !dbg !47
  %address.local1039 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1014, !dbg !47
  store half %1642, ptr addrspace(1) %address.local1039, align 2, !dbg !47
  %address.local1041 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1017, !dbg !47
  store half %1641, ptr addrspace(1) %address.local1041, align 2, !dbg !47
  %address.local1043 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1020, !dbg !47
  store half %1638, ptr addrspace(1) %address.local1043, align 2, !dbg !47
  %address.local1045 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1023, !dbg !47
  store half %1637, ptr addrspace(1) %address.local1045, align 2, !dbg !47
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1026, !dbg !47
  store half %1636, ptr addrspace(1) %address.local1047, align 2, !dbg !47
  %address.local1049 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1029, !dbg !47
  store half %1635, ptr addrspace(1) %address.local1049, align 2, !dbg !47
  %1657 = fmul <4 x float> %1538, %1169, !dbg !45
  %1658 = fptrunc <4 x float> %1657 to <4 x half>, !dbg !46
  %1659 = extractelement <4 x half> %1658, i64 3, !dbg !46
  %1660 = extractelement <4 x half> %1658, i64 2, !dbg !46
  %1661 = extractelement <4 x half> %1658, i64 1, !dbg !46
  %1662 = extractelement <4 x half> %1658, i64 0, !dbg !46
  %1663 = fmul <4 x float> %1538, %1136, !dbg !45
  %1664 = fptrunc <4 x float> %1663 to <4 x half>, !dbg !46
  %1665 = extractelement <4 x half> %1664, i64 3, !dbg !46
  %1666 = extractelement <4 x half> %1664, i64 2, !dbg !46
  %1667 = extractelement <4 x half> %1664, i64 1, !dbg !46
  %1668 = extractelement <4 x half> %1664, i64 0, !dbg !46
  %1669 = fmul <4 x float> %1537, %1103, !dbg !45
  %1670 = fptrunc <4 x float> %1669 to <4 x half>, !dbg !46
  %1671 = extractelement <4 x half> %1670, i64 3, !dbg !46
  %1672 = extractelement <4 x half> %1670, i64 2, !dbg !46
  %1673 = extractelement <4 x half> %1670, i64 1, !dbg !46
  %1674 = extractelement <4 x half> %1670, i64 0, !dbg !46
  %1675 = fmul <4 x float> %1537, %1070, !dbg !45
  %1676 = fptrunc <4 x float> %1675 to <4 x half>, !dbg !46
  %1677 = extractelement <4 x half> %1676, i64 3, !dbg !46
  %1678 = extractelement <4 x half> %1676, i64 2, !dbg !46
  %1679 = extractelement <4 x half> %1676, i64 1, !dbg !46
  %1680 = extractelement <4 x half> %1676, i64 0, !dbg !46
  %address.local1064 = or disjoint i64 %address.local896, 96, !dbg !47
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1064, !dbg !47
  store half %1680, ptr addrspace(1) %address.local1065, align 2, !dbg !47
  %address.local1067 = or disjoint i64 %address.local896, 97, !dbg !47
  %address.local1068 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1067, !dbg !47
  store half %1679, ptr addrspace(1) %address.local1068, align 2, !dbg !47
  %address.local1070 = or disjoint i64 %address.local896, 98, !dbg !47
  %address.local1071 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1070, !dbg !47
  store half %1678, ptr addrspace(1) %address.local1071, align 2, !dbg !47
  %address.local1073 = or disjoint i64 %address.local896, 99, !dbg !47
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1073, !dbg !47
  store half %1677, ptr addrspace(1) %address.local1074, align 2, !dbg !47
  %address.local1076 = or disjoint i64 %address.local896, 112, !dbg !47
  %address.local1077 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1076, !dbg !47
  store half %1674, ptr addrspace(1) %address.local1077, align 2, !dbg !47
  %address.local1079 = or disjoint i64 %address.local896, 113, !dbg !47
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1079, !dbg !47
  store half %1673, ptr addrspace(1) %address.local1080, align 2, !dbg !47
  %address.local1082 = or disjoint i64 %address.local896, 114, !dbg !47
  %address.local1083 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1082, !dbg !47
  store half %1672, ptr addrspace(1) %address.local1083, align 2, !dbg !47
  %address.local1085 = or i64 %1607, 115, !dbg !47
  %address.local1086 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local894, i64 %address.local1085, !dbg !47
  store half %1671, ptr addrspace(1) %address.local1086, align 2, !dbg !47
  %address.local1091 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1064, !dbg !47
  store half %1668, ptr addrspace(1) %address.local1091, align 2, !dbg !47
  %address.local1093 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1067, !dbg !47
  store half %1667, ptr addrspace(1) %address.local1093, align 2, !dbg !47
  %address.local1095 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1070, !dbg !47
  store half %1666, ptr addrspace(1) %address.local1095, align 2, !dbg !47
  %address.local1097 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1073, !dbg !47
  store half %1665, ptr addrspace(1) %address.local1097, align 2, !dbg !47
  %address.local1099 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1076, !dbg !47
  store half %1662, ptr addrspace(1) %address.local1099, align 2, !dbg !47
  %address.local1101 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1079, !dbg !47
  store half %1661, ptr addrspace(1) %address.local1101, align 2, !dbg !47
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1082, !dbg !47
  store half %1660, ptr addrspace(1) %address.local1103, align 2, !dbg !47
  %address.local1105 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local922, i64 %address.local1085, !dbg !47
  store half %1659, ptr addrspace(1) %address.local1105, align 2, !dbg !47
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
!1 = !DIFile(filename: "noncausal.input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_v_double_buffer")
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
!43 = !DILocation(line: 81, column: 16, scope: !3)
!44 = !DILocation(line: 80, column: 20, scope: !3)
!45 = !DILocation(line: 92, column: 23, scope: !3)
!46 = !DILocation(line: 93, column: 19, scope: !3)
!47 = !DILocation(line: 94, column: 9, scope: !3)
