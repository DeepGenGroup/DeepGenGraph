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
  %address.local351 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local352 = and i64 %address.local351, 1536, !dbg !9
  %address.local353 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local352, !dbg !9
  %address.local356 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local357 = and i64 %address.local356, 16256, !dbg !9
  %address.local358 = lshr i64 %address.local, 4, !dbg !9
  %address.local359 = and i64 %address.local358, 8, !dbg !9
  %address.local360 = or disjoint i64 %address.local357, %address.local359, !dbg !9
  %36 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1129 = shl i64 %34, 8, !dbg !9
  %37 = add i64 %.idx1129, %.idx, !dbg !9
  %address.local361.idx = shl nuw nsw i64 %address.local360, 1, !dbg !9
  %38 = or disjoint i64 %address.local361.idx, %37, !dbg !9
  %dma.buffer.offset1127 = trunc i64 %38 to i32, !dbg !9
  %39 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !9
  %40 = bitcast <2 x i64> %39 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1128 = shufflevector <4 x i32> %40, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local362 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local363 = shl nuw nsw i64 %address.local362, 3, !dbg !9
  %address.local364 = and i64 %address.local363, 3584, !dbg !9
  %address.local365 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local364, !dbg !9
  %address.local370 = lshr i64 %address.local362, 4, !dbg !9
  %address.local371 = and i64 %address.local370, 24, !dbg !9
  %address.local372 = or disjoint i64 %address.local371, %address.local357, !dbg !9
  %address.local373.idx = shl nuw nsw i64 %address.local372, 1, !dbg !9
  %41 = or disjoint i64 %address.local373.idx, %37, !dbg !9
  %dma.buffer.offset1125 = trunc i64 %41 to i32, !dbg !9
  %42 = shl nuw nsw i64 %31, 19, !dbg !9
  %43 = add i64 %35, %42, !dbg !9
  %address.local374 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local352, !dbg !9
  %44 = or disjoint i64 %43, 32, !dbg !9
  %45 = or disjoint i64 %address.local360, %44, !dbg !9
  %.tr = trunc i64 %45 to i32, !dbg !9
  %dma.buffer.offset1123 = shl i32 %.tr, 1, !dbg !9
  %address.local378 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local364, !dbg !9
  %46 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local353, i64 0, !dbg !9
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local365, i64 1, !dbg !9
  %48 = insertelement <4 x ptr addrspace(3)> %47, ptr addrspace(3) %address.local374, i64 2, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> %48, ptr addrspace(3) %address.local378, i64 3, !dbg !9
  %50 = ptrtoint <4 x ptr addrspace(3)> %49 to <4 x i64>, !dbg !9
  %51 = trunc <4 x i64> %50 to <4 x i32>, !dbg !9
  %52 = extractelement <4 x i32> %51, i64 0, !dbg !9
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !9
  %54 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1127, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %55 = extractelement <4 x i32> %51, i64 1, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1125, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %58 = extractelement <4 x i32> %51, i64 2, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1123, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %61 = extractelement <4 x i32> %51, i64 3, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = or disjoint i64 %address.local372, %44, !dbg !9
  %.tr1133 = trunc i64 %63 to i32, !dbg !9
  %dma.buffer.offset1121 = shl i32 %.tr1133, 1, !dbg !9
  %64 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1121, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %address.local382 = lshr i32 %address.thread, 2, !dbg !11
  %address.local383 = and i32 %address.local382, 4, !dbg !11
  %address.local384 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local385 = and i32 %address.local384, 384, !dbg !11
  %address.local386 = or disjoint i32 %address.local383, %address.local385, !dbg !11
  %address.local387 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local388 = and i32 %address.local387, 120, !dbg !11
  %address.local389 = or disjoint i32 %address.local386, %address.local388, !dbg !11
  %address.local390 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local391 = and i32 %address.local390, 1024, !dbg !11
  %address.local392 = or disjoint i32 %address.local389, %address.local391, !dbg !11
  %address.local393 = zext nneg i32 %address.local392 to i64, !dbg !11
  %address.local394 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local393, !dbg !11
  %65 = load <4 x half>, ptr addrspace(3) %address.local394, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local394, i64 4096, !dbg !11
  %66 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %67 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep950 = getelementptr i8, ptr addrspace(3) %address.local394, i64 1024, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !11
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep951 = getelementptr i8, ptr addrspace(3) %address.local394, i64 5120, !dbg !11
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !11
  %71 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %72 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %73 = or disjoint i64 %43, 64, !dbg !9
  %74 = or disjoint i64 %address.local360, %73, !dbg !9
  %.tr1135 = trunc i64 %74 to i32, !dbg !9
  %dma.buffer.offset1119 = shl i32 %.tr1135, 1, !dbg !9
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1119, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %76 = or disjoint i64 %address.local372, %73, !dbg !9
  %.tr1137 = trunc i64 %76 to i32, !dbg !9
  %dma.buffer.offset1117 = shl i32 %.tr1137, 1, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1117, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %lds.const.gep952 = getelementptr i8, ptr addrspace(3) %address.local394, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !11
  %lds.const.gep953 = getelementptr i8, ptr addrspace(3) %address.local394, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !11
  %lds.const.gep954 = getelementptr i8, ptr addrspace(3) %address.local394, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep955 = getelementptr i8, ptr addrspace(3) %address.local394, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = or disjoint i64 %43, 96, !dbg !9
  %87 = or disjoint i64 %address.local360, %86, !dbg !9
  %.tr1139 = trunc i64 %87 to i32, !dbg !9
  %dma.buffer.offset1115 = shl i32 %.tr1139, 1, !dbg !9
  %88 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1115, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %89 = or disjoint i64 %address.local372, %86, !dbg !9
  %.tr1141 = trunc i64 %89 to i32, !dbg !9
  %dma.buffer.offset1113 = shl i32 %.tr1141, 1, !dbg !9
  %90 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset1113, <4 x i32> %dma.buffer.rsrc1128) #9, !dbg !9
  %91 = load <4 x half>, ptr addrspace(3) %address.local394, align 8, !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep950, align 8, !dbg !11
  %94 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !11
  %95 = load <4 x half>, ptr addrspace(3) %lds.const.gep951, align 8, !dbg !11
  %96 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !11
  %98 = fmul <4 x half> %95, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep952, align 8, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep953, align 8, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep954, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !11
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep955, align 8, !dbg !11
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %105 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !11
  %106 = fmul <4 x half> %103, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %k0.prime. = shl i64 %25, 19, !dbg !13
  %k0.prime.1684 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1233 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1234 = and i64 %k0.prime.kdma.source1233, 8, !dbg !13
  %k0.prime.kdma.source1235 = and i64 %k0.prime.kdma.source, 91, !dbg !13
  %k0.prime.kdma.source1236 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1237 = and i64 %k0.prime.kdma.source1236, 4, !dbg !13
  %k0.prime.kdma.source1238 = add nuw nsw i64 %k0.prime.kdma.source1237, %k0.prime.kdma.source, !dbg !13
  %k0.prime.kdma.source1239 = and i64 %k0.prime.kdma.source1238, 4, !dbg !13
  %k0.prime.kdma.source1240 = or disjoint i64 %k0.prime.kdma.source1239, %k0.prime.kdma.source1235, !dbg !13
  %k0.prime.kdma.source1241 = shl nuw nsw i64 %k0.prime.kdma.source1240, 7, !dbg !13
  %k0.prime.kdma.source1242 = or disjoint i64 %k0.prime.kdma.source1241, %k0.prime.kdma.source1234, !dbg !13
  %k0.prime.kdma.source1243 = or disjoint i64 %k0.prime.kdma.source1242, %k0.prime.1684, !dbg !13
  %k0.prime.kdma.source1244 = trunc nuw nsw i64 %k0.prime.kdma.source1243 to i32, !dbg !13
  %k0.prime.kdma.source1245 = shl nuw nsw i32 %k0.prime.kdma.source1244, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1245, %kdma.intra, !dbg !13
  %k0.prime.1687 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1688 = insertelement <2 x i64> poison, i64 %k0.prime.1687, i64 0, !dbg !13
  %k0.prime.1689 = bitcast <2 x i64> %k0.prime.1688 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1112 = shufflevector <4 x i32> %k0.prime.1689, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1690 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1261 = add i32 %53, 512, !dbg !13
  %k0.prime.kdma.source1246 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1247 = lshr i64 %k0.prime.kdma.source1246, 4, !dbg !13
  %k0.prime.kdma.source1248 = and i64 %k0.prime.kdma.source1247, 8, !dbg !13
  %k0.prime.kdma.source1249 = and i64 %k0.prime.kdma.source1246, 123, !dbg !13
  %k0.prime.kdma.source1250 = lshr i64 %k0.prime.kdma.source1246, 2, !dbg !13
  %k0.prime.kdma.source1251 = and i64 %k0.prime.kdma.source1250, 4, !dbg !13
  %k0.prime.kdma.source1252 = add nuw nsw i64 %k0.prime.kdma.source1251, %k0.prime.kdma.source1246, !dbg !13
  %k0.prime.kdma.source1253 = and i64 %k0.prime.kdma.source1252, 4, !dbg !13
  %k0.prime.kdma.source1254 = or disjoint i64 %k0.prime.kdma.source1253, %k0.prime.kdma.source1249, !dbg !13
  %k0.prime.kdma.source1255 = shl nuw nsw i64 %k0.prime.kdma.source1254, 7, !dbg !13
  %k0.prime.kdma.source1256 = or disjoint i64 %k0.prime.kdma.source1255, %k0.prime.kdma.source1248, !dbg !13
  %k0.prime.kdma.source1257 = or disjoint i64 %k0.prime.kdma.source1256, %k0.prime.1684, !dbg !13
  %k0.prime.kdma.source1258 = trunc nuw nsw i64 %k0.prime.kdma.source1257 to i32, !dbg !13
  %k0.prime.kdma.source1259 = shl nuw nsw i32 %k0.prime.kdma.source1258, 1, !dbg !13
  %k0.prime.kdma.offset1260 = or disjoint i32 %k0.prime.kdma.source1259, %kdma.intra.hi, !dbg !13
  %k0.prime.1691 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1261, i32 %k0.prime.kdma.offset1260, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1273 = or disjoint i64 %k0.prime.kdma.source1242, 16, !dbg !13
  %k0.prime.kdma.source1274 = or disjoint i64 %k0.prime.kdma.source1273, %k0.prime.1684, !dbg !13
  %k0.prime.kdma.source1275 = trunc nuw nsw i64 %k0.prime.kdma.source1274 to i32, !dbg !13
  %k0.prime.kdma.source1276 = shl nuw nsw i32 %k0.prime.kdma.source1275, 1, !dbg !13
  %k0.prime.kdma.offset1277 = or disjoint i32 %k0.prime.kdma.source1276, %kdma.intra, !dbg !13
  %k0.prime.1692 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset1277, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1295 = add i32 %56, 512, !dbg !13
  %k0.prime.kdma.source1290 = or disjoint i64 %k0.prime.kdma.source1256, 16, !dbg !13
  %k0.prime.kdma.source1291 = or disjoint i64 %k0.prime.kdma.source1290, %k0.prime.1684, !dbg !13
  %k0.prime.kdma.source1292 = trunc nuw nsw i64 %k0.prime.kdma.source1291 to i32, !dbg !13
  %k0.prime.kdma.source1293 = shl nuw nsw i32 %k0.prime.kdma.source1292, 1, !dbg !13
  %k0.prime.kdma.offset1294 = or disjoint i32 %k0.prime.kdma.source1293, %kdma.intra.hi, !dbg !13
  %k0.prime.1693 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1295, i32 %k0.prime.kdma.offset1294, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %kdma.lds1327 = add i32 %59, 512
  %kdma.lds1361 = add i32 %62, 512
  %107 = ptrtoint ptr addrspace(1) %2 to i64
  %108 = insertelement <2 x i64> poison, i64 %107, i64 0
  %109 = bitcast <2 x i64> %108 to <4 x i32>
  %dma.buffer.rsrc1096 = shufflevector <4 x i32> %109, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %53, 8192
  %vdb.dma.base1187 = add i32 %56, 8192
  br label %110, !dbg !12

110:                                              ; preds = %4, %k0.join
  %111 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %999, %k0.join ]
  %112 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %995, %k0.join ]
  %113 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %991, %k0.join ]
  %114 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %987, %k0.join ]
  %115 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %750, %k0.join ]
  %116 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %985, %k0.join ]
  %117 = phi i64 [ 0, %4 ], [ %907, %k0.join ]
  %118 = shl nuw nsw i64 %117, 14, !dbg !13
  %119 = add nuw nsw i64 %118, %k0.prime.1684, !dbg !13
  %address.thread445 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !10, !frisk.register_lifetimes !5
  %address.local446 = zext nneg i32 %address.thread445 to i64, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %120 = or disjoint i64 %119, 32, !dbg !13
  %kdma.source1307 = or disjoint i64 %120, %k0.prime.kdma.source1242, !dbg !13
  %kdma.source1308 = trunc nuw nsw i64 %kdma.source1307 to i32, !dbg !13
  %kdma.source1309 = shl nuw nsw i32 %kdma.source1308, 1, !dbg !13
  %kdma.offset1310 = or disjoint i32 %kdma.source1309, %kdma.intra, !dbg !13
  %121 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1310, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1323 = or disjoint i64 %120, %k0.prime.kdma.source1256, !dbg !13
  %kdma.source1324 = trunc nuw nsw i64 %kdma.source1323 to i32, !dbg !13
  %kdma.source1325 = shl nuw nsw i32 %kdma.source1324, 1, !dbg !13
  %kdma.offset1326 = or disjoint i32 %kdma.source1325, %kdma.intra.hi, !dbg !13
  %122 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1327, i32 %kdma.offset1326, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1340 = or disjoint i64 %120, %k0.prime.kdma.source1273, !dbg !13
  %kdma.source1341 = trunc nuw nsw i64 %kdma.source1340 to i32, !dbg !13
  %kdma.source1342 = shl nuw nsw i32 %kdma.source1341, 1, !dbg !13
  %kdma.offset1343 = or disjoint i32 %kdma.source1342, %kdma.intra, !dbg !13
  %123 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1343, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1357 = or disjoint i64 %120, %k0.prime.kdma.source1290, !dbg !13
  %kdma.source1358 = trunc nuw nsw i64 %kdma.source1357 to i32, !dbg !13
  %kdma.source1359 = shl nuw nsw i32 %kdma.source1358, 1, !dbg !13
  %kdma.offset1360 = or disjoint i32 %kdma.source1359, %kdma.intra.hi, !dbg !13
  %124 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1361, i32 %kdma.offset1360, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %address.local472 = lshr i32 %address.thread445, 2, !dbg !14
  %address.local473 = and i32 %address.local472, 4, !dbg !14
  %address.local474 = shl nuw nsw i32 %address.thread445, 1, !dbg !14
  %address.local475 = and i32 %address.local474, 24, !dbg !14
  %address.local476 = or disjoint i32 %address.local473, %address.local475, !dbg !14
  %address.local477 = shl nuw nsw i32 %address.thread445, 5, !dbg !14
  %address.local478 = and i32 %address.local477, 1120, !dbg !14
  %address.local479 = or disjoint i32 %address.local476, %address.local478, !dbg !14
  %address.local480 = zext nneg i32 %address.local479 to i64, !dbg !14
  %address.local481 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local480, !dbg !14
  %125 = load <4 x half>, ptr addrspace(3) %address.local481, align 8, !dbg !14, !frisk.k_lds_pair !15
  %kdma.gep1500 = getelementptr i8, ptr addrspace(3) %address.local481, i64 1024, !dbg !14
  %126 = load <4 x half>, ptr addrspace(3) %kdma.gep1500, align 8, !dbg !14, !frisk.k_lds_pair !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %address.local481, i64 4096, !dbg !14
  %127 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !16
  %kdma.gep1501 = getelementptr i8, ptr addrspace(3) %address.local481, i64 5120, !dbg !14
  %128 = load <4 x half>, ptr addrspace(3) %kdma.gep1501, align 8, !dbg !14, !frisk.k_lds_pair !16
  %address.local499 = and i32 %address.local477, 1088, !dbg !14
  %address.local500 = or disjoint i32 %address.local499, %address.local476, !dbg !14
  %address.local503 = and i32 %address.local477, 32, !dbg !14
  %reass.sub = sub nsw i32 %address.local500, %address.local503, !dbg !14
  %address.local504 = add nsw i32 %reass.sub, 160, !dbg !14
  %address.local505 = zext nneg i32 %address.local504 to i64, !dbg !14
  %address.local506 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local505, !dbg !14
  %address.local506.idx = shl nuw nsw i64 %address.local505, 1, !dbg !14
  %129 = lshr i64 %address.local505, 5, !dbg !14
  %130 = and i64 %129, 8, !dbg !14
  %131 = xor i64 %130, %address.local506.idx, !dbg !14
  %132 = sub nsw i64 %131, %address.local506.idx, !dbg !14
  %kdma.swizzled1494 = getelementptr i8, ptr addrspace(3) %address.local506, i64 %132, !dbg !14
  %133 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1494, align 8, !dbg !14, !frisk.k_lds_pair !17
  %kdma.gep1502 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 1024, !dbg !14
  %134 = load <4 x half>, ptr addrspace(3) %kdma.gep1502, align 8, !dbg !14, !frisk.k_lds_pair !17
  %kdma.gep1495 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 4096, !dbg !14
  %135 = load <4 x half>, ptr addrspace(3) %kdma.gep1495, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep1503 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 5120, !dbg !14
  %136 = load <4 x half>, ptr addrspace(3) %kdma.gep1503, align 8, !dbg !14, !frisk.k_lds_pair !18
  %address.local481.idx = shl nuw nsw i64 %address.local480, 1, !dbg !14
  %137 = xor i64 %address.local481.idx, 520, !dbg !14
  %reass.sub6 = sub nsw i64 %137, %address.local481.idx, !dbg !14
  %kdma.swizzled1496 = getelementptr i8, ptr addrspace(3) %address.local481, i64 %reass.sub6, !dbg !14
  %138 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1496, align 8, !dbg !14, !frisk.k_lds_pair !19
  %kdma.gep1504 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 1024, !dbg !14
  %139 = load <4 x half>, ptr addrspace(3) %kdma.gep1504, align 8, !dbg !14, !frisk.k_lds_pair !19
  %kdma.gep1497 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 4096, !dbg !14
  %140 = load <4 x half>, ptr addrspace(3) %kdma.gep1497, align 8, !dbg !14, !frisk.k_lds_pair !20
  %kdma.gep1505 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 5120, !dbg !14
  %141 = load <4 x half>, ptr addrspace(3) %kdma.gep1505, align 8, !dbg !14, !frisk.k_lds_pair !20
  %lds.const.gep967 = getelementptr i8, ptr addrspace(3) %address.local506, i64 512, !dbg !14
  %142 = add nuw nsw i64 %address.local506.idx, 512, !dbg !14
  %143 = lshr i64 %142, 6, !dbg !14
  %144 = and i64 %143, 8, !dbg !14
  %145 = xor i64 %144, %142, !dbg !14
  %146 = sub nsw i64 %145, %142, !dbg !14
  %kdma.swizzled1498 = getelementptr i8, ptr addrspace(3) %lds.const.gep967, i64 %146, !dbg !14
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1498, align 8, !dbg !14, !frisk.k_lds_pair !21
  %kdma.gep1506 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 1024, !dbg !14
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep1506, align 8, !dbg !14, !frisk.k_lds_pair !21
  %kdma.gep1499 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 4096, !dbg !14
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep1499, align 8, !dbg !14, !frisk.k_lds_pair !22
  %kdma.gep1507 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 5120, !dbg !14
  %150 = load <4 x half>, ptr addrspace(3) %kdma.gep1507, align 8, !dbg !14, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %151 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %125, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %152 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %127, <4 x float> %151) #8, !dbg !14
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %133, <4 x float> zeroinitializer) #8, !dbg !14
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %135, <4 x float> %153) #8, !dbg !14
  %155 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %138, <4 x float> zeroinitializer) #8, !dbg !14
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %140, <4 x float> %155) #8, !dbg !14
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %147, <4 x float> zeroinitializer) #8, !dbg !14
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %149, <4 x float> %157) #8, !dbg !14
  %159 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %126, <4 x float> zeroinitializer) #8, !dbg !14
  %160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %128, <4 x float> %159) #8, !dbg !14
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %134, <4 x float> zeroinitializer) #8, !dbg !14
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %136, <4 x float> %161) #8, !dbg !14
  %163 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %139, <4 x float> zeroinitializer) #8, !dbg !14
  %164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %141, <4 x float> %163) #8, !dbg !14
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !14
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %150, <4 x float> %165) #8, !dbg !14
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %125, <4 x float> zeroinitializer) #8, !dbg !14
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %127, <4 x float> %167) #8, !dbg !14
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %133, <4 x float> zeroinitializer) #8, !dbg !14
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %135, <4 x float> %169) #8, !dbg !14
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %138, <4 x float> zeroinitializer) #8, !dbg !14
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %140, <4 x float> %171) #8, !dbg !14
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %147, <4 x float> zeroinitializer) #8, !dbg !14
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %173) #8, !dbg !14
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %126, <4 x float> zeroinitializer) #8, !dbg !14
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %128, <4 x float> %175) #8, !dbg !14
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %134, <4 x float> zeroinitializer) #8, !dbg !14
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %136, <4 x float> %177) #8, !dbg !14
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %139, <4 x float> zeroinitializer) #8, !dbg !14
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %141, <4 x float> %179) #8, !dbg !14
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !14
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %150, <4 x float> %181) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %183 = or disjoint i64 %119, 64, !dbg !13
  %kdma.source1373 = or disjoint i64 %183, %k0.prime.kdma.source1242, !dbg !13
  %kdma.source1374 = trunc nuw nsw i64 %kdma.source1373 to i32, !dbg !13
  %kdma.source1375 = shl nuw nsw i32 %kdma.source1374, 1, !dbg !13
  %kdma.offset1376 = or disjoint i32 %kdma.source1375, %kdma.intra, !dbg !13
  %184 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %kdma.offset1376, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1389 = or disjoint i64 %183, %k0.prime.kdma.source1256, !dbg !13
  %kdma.source1390 = trunc nuw nsw i64 %kdma.source1389 to i32, !dbg !13
  %kdma.source1391 = shl nuw nsw i32 %kdma.source1390, 1, !dbg !13
  %kdma.offset1392 = or disjoint i32 %kdma.source1391, %kdma.intra.hi, !dbg !13
  %185 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1261, i32 %kdma.offset1392, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1406 = or disjoint i64 %183, %k0.prime.kdma.source1273, !dbg !13
  %kdma.source1407 = trunc nuw nsw i64 %kdma.source1406 to i32, !dbg !13
  %kdma.source1408 = shl nuw nsw i32 %kdma.source1407, 1, !dbg !13
  %kdma.offset1409 = or disjoint i32 %kdma.source1408, %kdma.intra, !dbg !13
  %186 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1409, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1423 = or disjoint i64 %183, %k0.prime.kdma.source1290, !dbg !13
  %kdma.source1424 = trunc nuw nsw i64 %kdma.source1423 to i32, !dbg !13
  %kdma.source1425 = shl nuw nsw i32 %kdma.source1424, 1, !dbg !13
  %kdma.offset1426 = or disjoint i32 %kdma.source1425, %kdma.intra.hi, !dbg !13
  %187 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1295, i32 %kdma.offset1426, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.gep1524 = getelementptr i8, ptr addrspace(3) %address.local481, i64 8192, !dbg !14
  %188 = load <4 x half>, ptr addrspace(3) %kdma.gep1524, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep1532 = getelementptr i8, ptr addrspace(3) %address.local481, i64 9216, !dbg !14
  %189 = load <4 x half>, ptr addrspace(3) %kdma.gep1532, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep1525 = getelementptr i8, ptr addrspace(3) %address.local481, i64 12288, !dbg !14
  %190 = load <4 x half>, ptr addrspace(3) %kdma.gep1525, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep1533 = getelementptr i8, ptr addrspace(3) %address.local481, i64 13312, !dbg !14
  %191 = load <4 x half>, ptr addrspace(3) %kdma.gep1533, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep1526 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 8192, !dbg !14
  %192 = load <4 x half>, ptr addrspace(3) %kdma.gep1526, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep1534 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 9216, !dbg !14
  %193 = load <4 x half>, ptr addrspace(3) %kdma.gep1534, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep1527 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 12288, !dbg !14
  %194 = load <4 x half>, ptr addrspace(3) %kdma.gep1527, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep1535 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1494, i64 13312, !dbg !14
  %195 = load <4 x half>, ptr addrspace(3) %kdma.gep1535, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep1528 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 8192, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1528, align 8, !dbg !14, !frisk.k_lds_pair !27
  %kdma.gep1536 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 9216, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1536, align 8, !dbg !14, !frisk.k_lds_pair !27
  %kdma.gep1529 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 12288, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1529, align 8, !dbg !14, !frisk.k_lds_pair !28
  %kdma.gep1537 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1496, i64 13312, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1537, align 8, !dbg !14, !frisk.k_lds_pair !28
  %kdma.gep1530 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 8192, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1530, align 8, !dbg !14, !frisk.k_lds_pair !29
  %kdma.gep1538 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 9216, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1538, align 8, !dbg !14, !frisk.k_lds_pair !29
  %kdma.gep1531 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 12288, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1531, align 8, !dbg !14, !frisk.k_lds_pair !30
  %kdma.gep1539 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1498, i64 13312, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1539, align 8, !dbg !14, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %204 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %188, <4 x float> %152) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %205 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %190, <4 x float> %204) #8, !dbg !14
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %192, <4 x float> %154) #8, !dbg !14
  %207 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %194, <4 x float> %206) #8, !dbg !14
  %208 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %196, <4 x float> %156) #8, !dbg !14
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %198, <4 x float> %208) #8, !dbg !14
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %200, <4 x float> %158) #8, !dbg !14
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %202, <4 x float> %210) #8, !dbg !14
  %212 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %189, <4 x float> %160) #8, !dbg !14
  %213 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %191, <4 x float> %212) #8, !dbg !14
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %193, <4 x float> %162) #8, !dbg !14
  %215 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %195, <4 x float> %214) #8, !dbg !14
  %216 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %197, <4 x float> %164) #8, !dbg !14
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %199, <4 x float> %216) #8, !dbg !14
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %201, <4 x float> %166) #8, !dbg !14
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %203, <4 x float> %218) #8, !dbg !14
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %188, <4 x float> %168) #8, !dbg !14
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %190, <4 x float> %220) #8, !dbg !14
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %192, <4 x float> %170) #8, !dbg !14
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %194, <4 x float> %222) #8, !dbg !14
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %196, <4 x float> %172) #8, !dbg !14
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %198, <4 x float> %224) #8, !dbg !14
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %200, <4 x float> %174) #8, !dbg !14
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %202, <4 x float> %226) #8, !dbg !14
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %189, <4 x float> %176) #8, !dbg !14
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %191, <4 x float> %228) #8, !dbg !14
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %193, <4 x float> %178) #8, !dbg !14
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %195, <4 x float> %230) #8, !dbg !14
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %197, <4 x float> %180) #8, !dbg !14
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %199, <4 x float> %232) #8, !dbg !14
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %201, <4 x float> %182) #8, !dbg !14
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %203, <4 x float> %234) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %236 = or disjoint i64 %119, 96, !dbg !13
  %kdma.source1439 = or disjoint i64 %236, %k0.prime.kdma.source1242, !dbg !13
  %kdma.source1440 = trunc nuw nsw i64 %kdma.source1439 to i32, !dbg !13
  %kdma.source1441 = shl nuw nsw i32 %kdma.source1440, 1, !dbg !13
  %kdma.offset1442 = or disjoint i32 %kdma.source1441, %kdma.intra, !dbg !13
  %237 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1442, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1455 = or disjoint i64 %236, %k0.prime.kdma.source1256, !dbg !13
  %kdma.source1456 = trunc nuw nsw i64 %kdma.source1455 to i32, !dbg !13
  %kdma.source1457 = shl nuw nsw i32 %kdma.source1456, 1, !dbg !13
  %kdma.offset1458 = or disjoint i32 %kdma.source1457, %kdma.intra.hi, !dbg !13
  %238 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1327, i32 %kdma.offset1458, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1472 = or disjoint i64 %236, %k0.prime.kdma.source1273, !dbg !13
  %kdma.source1473 = trunc nuw nsw i64 %kdma.source1472 to i32, !dbg !13
  %kdma.source1474 = shl nuw nsw i32 %kdma.source1473, 1, !dbg !13
  %kdma.offset1475 = or disjoint i32 %kdma.source1474, %kdma.intra, !dbg !13
  %239 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1475, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %kdma.source1489 = or disjoint i64 %236, %k0.prime.kdma.source1290, !dbg !13
  %kdma.source1490 = trunc nuw nsw i64 %kdma.source1489 to i32, !dbg !13
  %kdma.source1491 = shl nuw nsw i32 %kdma.source1490, 1, !dbg !13
  %kdma.offset1492 = or disjoint i32 %kdma.source1491, %kdma.intra.hi, !dbg !13
  %240 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1361, i32 %kdma.offset1492, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13
  %241 = load <4 x half>, ptr addrspace(3) %address.local481, align 8, !dbg !14, !frisk.k_lds_pair !31
  %242 = load <4 x half>, ptr addrspace(3) %kdma.gep1500, align 8, !dbg !14, !frisk.k_lds_pair !31
  %243 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !32
  %244 = load <4 x half>, ptr addrspace(3) %kdma.gep1501, align 8, !dbg !14, !frisk.k_lds_pair !32
  %245 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1494, align 8, !dbg !14, !frisk.k_lds_pair !33
  %246 = load <4 x half>, ptr addrspace(3) %kdma.gep1502, align 8, !dbg !14, !frisk.k_lds_pair !33
  %247 = load <4 x half>, ptr addrspace(3) %kdma.gep1495, align 8, !dbg !14, !frisk.k_lds_pair !34
  %248 = load <4 x half>, ptr addrspace(3) %kdma.gep1503, align 8, !dbg !14, !frisk.k_lds_pair !34
  %249 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1496, align 8, !dbg !14, !frisk.k_lds_pair !35
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1504, align 8, !dbg !14, !frisk.k_lds_pair !35
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep1497, align 8, !dbg !14, !frisk.k_lds_pair !36
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1505, align 8, !dbg !14, !frisk.k_lds_pair !36
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1498, align 8, !dbg !14, !frisk.k_lds_pair !37
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1506, align 8, !dbg !14, !frisk.k_lds_pair !37
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep1499, align 8, !dbg !14, !frisk.k_lds_pair !38
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1507, align 8, !dbg !14, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %257 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %241, <4 x float> %205) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %258 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %243, <4 x float> %257) #8, !dbg !14
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %245, <4 x float> %207) #8, !dbg !14
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %247, <4 x float> %259) #8, !dbg !14
  %261 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %249, <4 x float> %209) #8, !dbg !14
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %251, <4 x float> %261) #8, !dbg !14
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %253, <4 x float> %211) #8, !dbg !14
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %255, <4 x float> %263) #8, !dbg !14
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %242, <4 x float> %213) #8, !dbg !14
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %244, <4 x float> %265) #8, !dbg !14
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %246, <4 x float> %215) #8, !dbg !14
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %248, <4 x float> %267) #8, !dbg !14
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %250, <4 x float> %217) #8, !dbg !14
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %252, <4 x float> %269) #8, !dbg !14
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %254, <4 x float> %219) #8, !dbg !14
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %256, <4 x float> %271) #8, !dbg !14
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %241, <4 x float> %221) #8, !dbg !14
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %243, <4 x float> %273) #8, !dbg !14
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %245, <4 x float> %223) #8, !dbg !14
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %247, <4 x float> %275) #8, !dbg !14
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %249, <4 x float> %225) #8, !dbg !14
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %251, <4 x float> %277) #8, !dbg !14
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %253, <4 x float> %227) #8, !dbg !14
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %255, <4 x float> %279) #8, !dbg !14
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %242, <4 x float> %229) #8, !dbg !14
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %244, <4 x float> %281) #8, !dbg !14
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %246, <4 x float> %231) #8, !dbg !14
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %248, <4 x float> %283) #8, !dbg !14
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %250, <4 x float> %233) #8, !dbg !14
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %252, <4 x float> %285) #8, !dbg !14
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %254, <4 x float> %235) #8, !dbg !14
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %256, <4 x float> %287) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local691 = shl nuw nsw i64 %address.local446, 5, !dbg !39
  %address.local692 = and i64 %address.local691, 8064, !dbg !39
  %address.local693 = shl nuw nsw i64 %address.local446, 3, !dbg !39
  %address.local694 = and i64 %address.local693, 24, !dbg !39
  %address.local695 = or disjoint i64 %address.local692, %address.local694, !dbg !39
  %.idx11581159 = or disjoint i64 %address.local695, %119, !dbg !39
  %.idx11581159.tr = trunc nuw nsw i64 %.idx11581159 to i32, !dbg !39
  %dma.buffer.offset1095 = shl nuw nsw i32 %.idx11581159.tr, 1, !dbg !39
  %289 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1095, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %address.local701 = or disjoint i64 %address.local695, 8192, !dbg !39
  %.idx11601161 = or disjoint i64 %address.local701, %119, !dbg !39
  %.idx11601161.tr = trunc nuw nsw i64 %.idx11601161 to i32, !dbg !39
  %dma.buffer.offset1093 = shl nuw nsw i32 %.idx11601161.tr, 1, !dbg !39
  %290 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1093, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %291 = load <4 x half>, ptr addrspace(3) %kdma.gep1524, align 8, !dbg !14, !frisk.k_tail_pair !15
  %292 = load <4 x half>, ptr addrspace(3) %kdma.gep1532, align 8, !dbg !14, !frisk.k_tail_pair !15
  %293 = load <4 x half>, ptr addrspace(3) %kdma.gep1525, align 8, !dbg !14, !frisk.k_tail_pair !16
  %294 = load <4 x half>, ptr addrspace(3) %kdma.gep1533, align 8, !dbg !14, !frisk.k_tail_pair !16
  %295 = load <4 x half>, ptr addrspace(3) %kdma.gep1526, align 8, !dbg !14, !frisk.k_tail_pair !17
  %296 = load <4 x half>, ptr addrspace(3) %kdma.gep1534, align 8, !dbg !14, !frisk.k_tail_pair !17
  %297 = load <4 x half>, ptr addrspace(3) %kdma.gep1527, align 8, !dbg !14, !frisk.k_tail_pair !18
  %298 = load <4 x half>, ptr addrspace(3) %kdma.gep1535, align 8, !dbg !14, !frisk.k_tail_pair !18
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1528, align 8, !dbg !14, !frisk.k_tail_pair !19
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1536, align 8, !dbg !14, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %291, <4 x float> %258) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1529, align 8, !dbg !14, !frisk.k_tail_pair !20
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1537, align 8, !dbg !14, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %293, <4 x float> %301) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %305 = load <4 x half>, ptr addrspace(3) %kdma.gep1530, align 8, !dbg !14, !frisk.k_tail_pair !21
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1538, align 8, !dbg !14, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %295, <4 x float> %260) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40, !frisk.k_tail_prefetch !5
  %308 = extractelement <4 x float> %304, i64 0, !dbg !40
  %309 = tail call float @llvm.maxnum.f32(float %308, float -inf), !dbg !7, !frisk.local_reduction !5
  %310 = extractelement <4 x float> %304, i64 1, !dbg !40
  %311 = extractelement <4 x float> %304, i64 2, !dbg !40
  %312 = extractelement <4 x float> %304, i64 3, !dbg !40
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1531, align 8, !dbg !14, !frisk.k_tail_pair !22
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1539, align 8, !dbg !14, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %297, <4 x float> %307) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %299, <4 x float> %262) #8, !dbg !14
  %317 = extractelement <4 x float> %315, i64 0, !dbg !40
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %309, float %317), !dbg !7, !frisk.local_reduction !5
  %318 = extractelement <4 x float> %315, i64 1, !dbg !40
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %310, float %318), !dbg !7, !frisk.local_reduction !5
  %319 = extractelement <4 x float> %315, i64 2, !dbg !40
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %311, float %319), !dbg !7, !frisk.local_reduction !5
  %320 = extractelement <4 x float> %315, i64 3, !dbg !40
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %312, float %320), !dbg !7, !frisk.local_reduction !5
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %302, <4 x float> %316) #8, !dbg !14
  %322 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %305, <4 x float> %264) #8, !dbg !14
  %323 = extractelement <4 x float> %321, i64 0, !dbg !40
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %323), !dbg !7, !frisk.local_reduction !5
  %324 = extractelement <4 x float> %321, i64 1, !dbg !40
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %324), !dbg !7, !frisk.local_reduction !5
  %325 = extractelement <4 x float> %321, i64 2, !dbg !40
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %325), !dbg !7, !frisk.local_reduction !5
  %326 = extractelement <4 x float> %321, i64 3, !dbg !40
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %326), !dbg !7, !frisk.local_reduction !5
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %313, <4 x float> %322) #8, !dbg !14
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %292, <4 x float> %266) #8, !dbg !14
  %329 = extractelement <4 x float> %327, i64 0, !dbg !40
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %329), !dbg !7, !frisk.local_reduction !5
  %330 = extractelement <4 x float> %327, i64 1, !dbg !40
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %330), !dbg !7, !frisk.local_reduction !5
  %331 = extractelement <4 x float> %327, i64 2, !dbg !40
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %331), !dbg !7, !frisk.local_reduction !5
  %332 = extractelement <4 x float> %327, i64 3, !dbg !40
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %332), !dbg !7, !frisk.local_reduction !5
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %294, <4 x float> %328) #8, !dbg !14
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %296, <4 x float> %268) #8, !dbg !14
  %335 = extractelement <4 x float> %333, i64 0, !dbg !40
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %335), !dbg !7, !frisk.local_reduction !5
  %336 = extractelement <4 x float> %333, i64 1, !dbg !40
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %336), !dbg !7, !frisk.local_reduction !5
  %337 = extractelement <4 x float> %333, i64 2, !dbg !40
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %337), !dbg !7, !frisk.local_reduction !5
  %338 = extractelement <4 x float> %333, i64 3, !dbg !40
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %338), !dbg !7, !frisk.local_reduction !5
  %339 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %298, <4 x float> %334) #8, !dbg !14
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %300, <4 x float> %270) #8, !dbg !14
  %341 = extractelement <4 x float> %339, i64 0, !dbg !40
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %341), !dbg !7, !frisk.local_reduction !5
  %342 = extractelement <4 x float> %339, i64 1, !dbg !40
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %342), !dbg !7, !frisk.local_reduction !5
  %343 = extractelement <4 x float> %339, i64 2, !dbg !40
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %343), !dbg !7, !frisk.local_reduction !5
  %344 = extractelement <4 x float> %339, i64 3, !dbg !40
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %344), !dbg !7, !frisk.local_reduction !5
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %303, <4 x float> %340) #8, !dbg !14
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %306, <4 x float> %272) #8, !dbg !14
  %347 = extractelement <4 x float> %345, i64 0, !dbg !40
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %local.reduce149, float %347), !dbg !7, !frisk.local_reduction !5
  %348 = extractelement <4 x float> %345, i64 1, !dbg !40
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %local.reduce150, float %348), !dbg !7, !frisk.local_reduction !5
  %349 = extractelement <4 x float> %345, i64 2, !dbg !40
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %349), !dbg !7, !frisk.local_reduction !5
  %350 = extractelement <4 x float> %345, i64 3, !dbg !40
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %350), !dbg !7, !frisk.local_reduction !5
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %314, <4 x float> %346) #8, !dbg !14
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %291, <4 x float> %274) #8, !dbg !14
  %353 = extractelement <4 x float> %351, i64 0, !dbg !40
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %353), !dbg !7, !frisk.local_reduction !5
  %354 = extractelement <4 x float> %351, i64 1, !dbg !40
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %354), !dbg !7, !frisk.local_reduction !5
  %355 = extractelement <4 x float> %351, i64 2, !dbg !40
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %355), !dbg !7, !frisk.local_reduction !5
  %356 = extractelement <4 x float> %351, i64 3, !dbg !40
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %356), !dbg !7, !frisk.local_reduction !5
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %local.reduce158), !dbg !7, !frisk.local_reduction !5
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %local.reduce160), !dbg !7, !frisk.local_reduction !5
  %357 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %local.reduce162), !dbg !7, !frisk.local_reduction !5
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %293, <4 x float> %352) #8, !dbg !14
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %295, <4 x float> %276) #8, !dbg !14
  %360 = extractelement <4 x float> %358, i64 0, !dbg !40
  %361 = extractelement <4 x float> %358, i64 1, !dbg !40
  %362 = extractelement <4 x float> %358, i64 2, !dbg !40
  %363 = extractelement <4 x float> %358, i64 3, !dbg !40
  %364 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %297, <4 x float> %359) #8, !dbg !14
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %299, <4 x float> %278) #8, !dbg !14
  %366 = extractelement <4 x float> %364, i64 0, !dbg !40
  %367 = extractelement <4 x float> %364, i64 1, !dbg !40
  %368 = extractelement <4 x float> %364, i64 2, !dbg !40
  %369 = extractelement <4 x float> %364, i64 3, !dbg !40
  %370 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %302, <4 x float> %365) #8, !dbg !14
  %371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %305, <4 x float> %280) #8, !dbg !14
  %372 = extractelement <4 x float> %370, i64 0, !dbg !40
  %373 = extractelement <4 x float> %370, i64 1, !dbg !40
  %374 = extractelement <4 x float> %370, i64 2, !dbg !40
  %375 = extractelement <4 x float> %370, i64 3, !dbg !40
  %376 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %313, <4 x float> %371) #8, !dbg !14
  %377 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %292, <4 x float> %282) #8, !dbg !14
  %378 = extractelement <4 x float> %376, i64 0, !dbg !40
  %379 = extractelement <4 x float> %376, i64 1, !dbg !40
  %380 = extractelement <4 x float> %376, i64 2, !dbg !40
  %381 = extractelement <4 x float> %376, i64 3, !dbg !40
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %294, <4 x float> %377) #8, !dbg !14
  %383 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %296, <4 x float> %284) #8, !dbg !14
  %384 = extractelement <4 x float> %382, i64 0, !dbg !40
  %385 = extractelement <4 x float> %382, i64 1, !dbg !40
  %386 = extractelement <4 x float> %382, i64 2, !dbg !40
  %387 = extractelement <4 x float> %382, i64 3, !dbg !40
  %388 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %298, <4 x float> %383) #8, !dbg !14
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %300, <4 x float> %286) #8, !dbg !14
  %390 = extractelement <4 x float> %388, i64 0, !dbg !40
  %391 = extractelement <4 x float> %388, i64 1, !dbg !40
  %392 = extractelement <4 x float> %388, i64 2, !dbg !40
  %393 = extractelement <4 x float> %388, i64 3, !dbg !40
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %303, <4 x float> %389) #8, !dbg !14
  %395 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %306, <4 x float> %288) #8, !dbg !14
  %396 = extractelement <4 x float> %394, i64 0, !dbg !40
  %397 = extractelement <4 x float> %394, i64 1, !dbg !40
  %398 = extractelement <4 x float> %394, i64 2, !dbg !40
  %399 = extractelement <4 x float> %394, i64 3, !dbg !40
  %400 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %314, <4 x float> %395) #8, !dbg !14
  %401 = extractelement <4 x float> %400, i64 0, !dbg !40
  %402 = extractelement <4 x float> %400, i64 1, !dbg !40
  %403 = extractelement <4 x float> %400, i64 2, !dbg !40
  %404 = extractelement <4 x float> %400, i64 3, !dbg !40
  %405 = bitcast float %357 to i32, !dbg !7
  %406 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %405), !dbg !7
  %407 = bitcast i32 %406 to float, !dbg !7
  %408 = tail call float @llvm.maxnum.f32(float %357, float %407), !dbg !7
  %409 = bitcast float %408 to i32, !dbg !7
  %410 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %409), !dbg !7
  %411 = bitcast i32 %410 to float, !dbg !7
  %412 = tail call float @llvm.maxnum.f32(float %408, float %411), !dbg !7
  %413 = tail call float @llvm.maxnum.f32(float %360, float -inf), !dbg !7, !frisk.local_reduction !5
  %local.reduce102 = tail call float @llvm.maxnum.f32(float %413, float %366), !dbg !7, !frisk.local_reduction !5
  %local.reduce103 = tail call float @llvm.maxnum.f32(float %361, float %367), !dbg !7, !frisk.local_reduction !5
  %local.reduce104 = tail call float @llvm.maxnum.f32(float %362, float %368), !dbg !7, !frisk.local_reduction !5
  %local.reduce105 = tail call float @llvm.maxnum.f32(float %363, float %369), !dbg !7, !frisk.local_reduction !5
  %local.reduce106 = tail call float @llvm.maxnum.f32(float %local.reduce102, float %372), !dbg !7, !frisk.local_reduction !5
  %local.reduce107 = tail call float @llvm.maxnum.f32(float %local.reduce103, float %373), !dbg !7, !frisk.local_reduction !5
  %local.reduce108 = tail call float @llvm.maxnum.f32(float %local.reduce104, float %374), !dbg !7, !frisk.local_reduction !5
  %local.reduce109 = tail call float @llvm.maxnum.f32(float %local.reduce105, float %375), !dbg !7, !frisk.local_reduction !5
  %local.reduce110 = tail call float @llvm.maxnum.f32(float %local.reduce106, float %378), !dbg !7, !frisk.local_reduction !5
  %local.reduce111 = tail call float @llvm.maxnum.f32(float %local.reduce107, float %379), !dbg !7, !frisk.local_reduction !5
  %local.reduce112 = tail call float @llvm.maxnum.f32(float %local.reduce108, float %380), !dbg !7, !frisk.local_reduction !5
  %local.reduce113 = tail call float @llvm.maxnum.f32(float %local.reduce109, float %381), !dbg !7, !frisk.local_reduction !5
  %local.reduce114 = tail call float @llvm.maxnum.f32(float %local.reduce110, float %384), !dbg !7, !frisk.local_reduction !5
  %local.reduce115 = tail call float @llvm.maxnum.f32(float %local.reduce111, float %385), !dbg !7, !frisk.local_reduction !5
  %local.reduce116 = tail call float @llvm.maxnum.f32(float %local.reduce112, float %386), !dbg !7, !frisk.local_reduction !5
  %local.reduce117 = tail call float @llvm.maxnum.f32(float %local.reduce113, float %387), !dbg !7, !frisk.local_reduction !5
  %local.reduce118 = tail call float @llvm.maxnum.f32(float %local.reduce114, float %390), !dbg !7, !frisk.local_reduction !5
  %local.reduce119 = tail call float @llvm.maxnum.f32(float %local.reduce115, float %391), !dbg !7, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %local.reduce116, float %392), !dbg !7, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %local.reduce117, float %393), !dbg !7, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %local.reduce118, float %396), !dbg !7, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %local.reduce119, float %397), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %398), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %399), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %401), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %402), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %403), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %404), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %local.reduce127), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %local.reduce129), !dbg !7, !frisk.local_reduction !5
  %414 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %local.reduce131), !dbg !7, !frisk.local_reduction !5
  %415 = bitcast float %414 to i32, !dbg !7
  %416 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %415), !dbg !7
  %417 = bitcast i32 %416 to float, !dbg !7
  %418 = tail call float @llvm.maxnum.f32(float %414, float %417), !dbg !7
  %419 = bitcast float %418 to i32, !dbg !7
  %420 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %419), !dbg !7
  %421 = bitcast i32 %420 to float, !dbg !7
  %422 = tail call float @llvm.maxnum.f32(float %418, float %421), !dbg !7
  %423 = extractvalue [2 x <1 x float>] %114, 0, !dbg !7
  %424 = insertelement <1 x float> poison, float %412, i64 0, !dbg !7
  %425 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %424, <1 x float> %423), !dbg !7
  %426 = extractvalue [2 x <1 x float>] %114, 1, !dbg !7
  %427 = insertelement <1 x float> poison, float %422, i64 0, !dbg !7
  %428 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %427, <1 x float> %426), !dbg !7
  %429 = shufflevector <1 x float> %425, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !41
  %430 = shufflevector <1 x float> %428, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !41
  %431 = fsub <1 x float> %423, %425, !dbg !42
  %432 = extractelement <1 x float> %431, i64 0, !dbg !43
  %433 = tail call float @llvm.amdgcn.exp2.f32(float %432), !dbg !43
  %434 = insertelement <4 x float> poison, float %433, i64 0, !dbg !44
  %435 = shufflevector <4 x float> %434, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %436 = extractvalue [2 x <1 x float>] %115, 0, !dbg !45
  %437 = extractelement <1 x float> %436, i64 0, !dbg !45
  %438 = fmul float %437, %433, !dbg !45
  %439 = fsub <1 x float> %426, %428, !dbg !42
  %440 = extractelement <1 x float> %439, i64 0, !dbg !43
  %441 = tail call float @llvm.amdgcn.exp2.f32(float %440), !dbg !43
  %442 = insertelement <4 x float> poison, float %441, i64 0, !dbg !44
  %443 = shufflevector <4 x float> %442, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %444 = extractvalue [2 x <1 x float>] %115, 1, !dbg !45
  %445 = extractelement <1 x float> %444, i64 0, !dbg !45
  %446 = fmul float %445, %441, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address = or disjoint i64 %address.local695, %120, !dbg !39
  %vdb.address1174 = trunc nuw nsw i64 %vdb.address to i32, !dbg !39
  %vdb.address1175 = shl nuw nsw i32 %vdb.address1174, 1, !dbg !39
  %vdb.address1180 = or disjoint i64 %address.local701, %120, !dbg !39
  %vdb.address1181 = trunc nuw nsw i64 %vdb.address1180 to i32, !dbg !39
  %vdb.address1182 = shl nuw nsw i32 %vdb.address1181, 1, !dbg !39
  %447 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1175, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %448 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1187, i32 %vdb.address1182, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %address.local703 = shl nuw nsw i32 %address.thread445, 3, !dbg !46
  %address.local704 = and i32 %address.local703, 504, !dbg !46
  %address.local705 = zext nneg i32 %address.local704 to i64, !dbg !46
  %address.local706 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local705, !dbg !46
  %449 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local706, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %450 = bitcast <4 x i32> %449 to <8 x half>, !dbg !46
  %address.local708 = getelementptr i8, ptr addrspace(3) %address.local706, i64 1024, !dbg !46
  %451 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local708, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %452 = bitcast <4 x i32> %451 to <8 x half>, !dbg !46
  %address.local710 = getelementptr i8, ptr addrspace(3) %address.local706, i64 2048, !dbg !46
  %453 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local710, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %454 = bitcast <4 x i32> %453 to <8 x half>, !dbg !46
  %address.local712 = getelementptr i8, ptr addrspace(3) %address.local706, i64 3072, !dbg !46
  %455 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local712, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %456 = bitcast <4 x i32> %455 to <8 x half>, !dbg !46
  %address.local714 = getelementptr i8, ptr addrspace(3) %address.local706, i64 4096, !dbg !46
  %457 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local714, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %458 = bitcast <4 x i32> %457 to <8 x half>, !dbg !46
  %address.local716 = getelementptr i8, ptr addrspace(3) %address.local706, i64 5120, !dbg !46
  %459 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local716, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %460 = bitcast <4 x i32> %459 to <8 x half>, !dbg !46
  %address.local718 = getelementptr i8, ptr addrspace(3) %address.local706, i64 6144, !dbg !46
  %461 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local718, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %462 = bitcast <4 x i32> %461 to <8 x half>, !dbg !46
  %address.local720 = getelementptr i8, ptr addrspace(3) %address.local706, i64 7168, !dbg !46
  %463 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local720, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %464 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %465 = extractvalue [2 x <8 x float>] %116, 0, !dbg !44
  %466 = shufflevector <8 x float> %465, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %467 = fmul <4 x float> %466, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %468 = fsub <4 x float> %304, %429, !dbg !41
  %469 = extractelement <4 x float> %468, i64 0, !dbg !48
  %470 = tail call float @llvm.amdgcn.exp2.f32(float %469), !dbg !48
  %471 = insertelement <4 x float> poison, float %470, i64 0, !dbg !49
  %472 = extractelement <4 x float> %468, i64 1, !dbg !48
  %473 = tail call float @llvm.amdgcn.exp2.f32(float %472), !dbg !48
  %474 = insertelement <4 x float> %471, float %473, i64 1, !dbg !49
  %475 = extractelement <4 x float> %468, i64 2, !dbg !48
  %476 = tail call float @llvm.amdgcn.exp2.f32(float %475), !dbg !48
  %477 = insertelement <4 x float> %474, float %476, i64 2, !dbg !49
  %478 = extractelement <4 x float> %468, i64 3, !dbg !48
  %479 = tail call float @llvm.amdgcn.exp2.f32(float %478), !dbg !48
  %480 = insertelement <4 x float> %477, float %479, i64 3, !dbg !49
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %480) #8, !dbg !49, !frisk.register_lifetimes !5
  %481 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !49
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %481) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %482 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %464, <4 x float> %467) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %483 = shufflevector <8 x half> %450, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %484 = shufflevector <8 x float> %465, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %485 = fmul <4 x float> %484, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %486 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %483, <4 x float> %485) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %487 = extractvalue [2 x <8 x float>] %116, 1, !dbg !44
  %488 = shufflevector <8 x float> %487, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %489 = fmul <4 x float> %488, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %490 = fsub <4 x float> %358, %430, !dbg !41
  %491 = extractelement <4 x float> %490, i64 0, !dbg !48
  %492 = tail call float @llvm.amdgcn.exp2.f32(float %491), !dbg !48
  %493 = insertelement <4 x float> poison, float %492, i64 0, !dbg !49
  %494 = extractelement <4 x float> %490, i64 1, !dbg !48
  %495 = tail call float @llvm.amdgcn.exp2.f32(float %494), !dbg !48
  %496 = insertelement <4 x float> %493, float %495, i64 1, !dbg !49
  %497 = extractelement <4 x float> %490, i64 2, !dbg !48
  %498 = tail call float @llvm.amdgcn.exp2.f32(float %497), !dbg !48
  %499 = insertelement <4 x float> %496, float %498, i64 2, !dbg !49
  %500 = extractelement <4 x float> %490, i64 3, !dbg !48
  %501 = tail call float @llvm.amdgcn.exp2.f32(float %500), !dbg !48
  %502 = insertelement <4 x float> %499, float %501, i64 3, !dbg !49
  %math.result335 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %502) #8, !dbg !49, !frisk.register_lifetimes !5
  %503 = fptrunc <4 x float> %math.result335 to <4 x half>, !dbg !49
  %math.packed336 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %503) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %504 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %464, <4 x float> %489) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %505 = shufflevector <8 x float> %487, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %506 = fmul <4 x float> %505, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %507 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %483, <4 x float> %506) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %508 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %509 = fsub <4 x float> %315, %429, !dbg !41
  %510 = extractelement <4 x float> %509, i64 0, !dbg !48
  %511 = tail call float @llvm.amdgcn.exp2.f32(float %510), !dbg !48
  %local.reduce71 = fadd float %470, %511, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %512 = insertelement <4 x float> poison, float %511, i64 0, !dbg !49
  %513 = extractelement <4 x float> %509, i64 1, !dbg !48
  %514 = tail call float @llvm.amdgcn.exp2.f32(float %513), !dbg !48
  %local.reduce72 = fadd float %473, %514, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1621 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %515 = insertelement <4 x float> %512, float %514, i64 1, !dbg !49
  %516 = extractelement <4 x float> %509, i64 2, !dbg !48
  %517 = tail call float @llvm.amdgcn.exp2.f32(float %516), !dbg !48
  %local.reduce73 = fadd float %476, %517, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1622 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %518 = insertelement <4 x float> %515, float %517, i64 2, !dbg !49
  %519 = extractelement <4 x float> %509, i64 3, !dbg !48
  %520 = tail call float @llvm.amdgcn.exp2.f32(float %519), !dbg !48
  %local.reduce74 = fadd float %479, %520, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1623 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %521 = insertelement <4 x float> %518, float %520, i64 3, !dbg !49
  %math.result321 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %521) #8, !dbg !49, !frisk.register_lifetimes !5
  %522 = fptrunc <4 x float> %math.result321 to <4 x half>, !dbg !49
  %math.packed322 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %522) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %523 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %508, <4 x float> %482) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %524 = shufflevector <8 x half> %452, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %525 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %524, <4 x float> %486) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %526 = fsub <4 x float> %364, %430, !dbg !41
  %527 = extractelement <4 x float> %526, i64 0, !dbg !48
  %528 = tail call float @llvm.amdgcn.exp2.f32(float %527), !dbg !48
  %local.reduce = fadd float %492, %528, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1624 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %529 = insertelement <4 x float> poison, float %528, i64 0, !dbg !49
  %530 = extractelement <4 x float> %526, i64 1, !dbg !48
  %531 = tail call float @llvm.amdgcn.exp2.f32(float %530), !dbg !48
  %local.reduce41 = fadd float %495, %531, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1625 = tail call float asm sideeffect "", "=v,0"(float %local.reduce41) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %532 = insertelement <4 x float> %529, float %531, i64 1, !dbg !49
  %533 = extractelement <4 x float> %526, i64 2, !dbg !48
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !48
  %local.reduce42 = fadd float %498, %534, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1626 = tail call float asm sideeffect "", "=v,0"(float %local.reduce42) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %535 = insertelement <4 x float> %532, float %534, i64 2, !dbg !49
  %536 = extractelement <4 x float> %526, i64 3, !dbg !48
  %537 = tail call float @llvm.amdgcn.exp2.f32(float %536), !dbg !48
  %local.reduce43 = fadd float %501, %537, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1627 = tail call float asm sideeffect "", "=v,0"(float %local.reduce43) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %538 = insertelement <4 x float> %535, float %537, i64 3, !dbg !49
  %math.result337 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %538) #8, !dbg !49, !frisk.register_lifetimes !5
  %539 = fptrunc <4 x float> %math.result337 to <4 x half>, !dbg !49
  %math.packed338 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %539) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %540 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %508, <4 x float> %504) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %541 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %524, <4 x float> %507) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %542 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %543 = fsub <4 x float> %321, %429, !dbg !41
  %544 = extractelement <4 x float> %543, i64 0, !dbg !48
  %545 = tail call float @llvm.amdgcn.exp2.f32(float %544), !dbg !48
  %local.reduce75 = fadd float %545, %k0.sum.anchor, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1628 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %546 = insertelement <4 x float> poison, float %545, i64 0, !dbg !49
  %547 = extractelement <4 x float> %543, i64 1, !dbg !48
  %548 = tail call float @llvm.amdgcn.exp2.f32(float %547), !dbg !48
  %local.reduce76 = fadd float %548, %k0.sum.anchor1621, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1629 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %549 = insertelement <4 x float> %546, float %548, i64 1, !dbg !49
  %550 = extractelement <4 x float> %543, i64 2, !dbg !48
  %551 = tail call float @llvm.amdgcn.exp2.f32(float %550), !dbg !48
  %local.reduce77 = fadd float %551, %k0.sum.anchor1622, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1630 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %552 = insertelement <4 x float> %549, float %551, i64 2, !dbg !49
  %553 = extractelement <4 x float> %543, i64 3, !dbg !48
  %554 = tail call float @llvm.amdgcn.exp2.f32(float %553), !dbg !48
  %local.reduce78 = fadd float %554, %k0.sum.anchor1623, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1631 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %555 = insertelement <4 x float> %552, float %554, i64 3, !dbg !49
  %math.result323 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %555) #8, !dbg !49, !frisk.register_lifetimes !5
  %556 = fptrunc <4 x float> %math.result323 to <4 x half>, !dbg !49
  %math.packed324 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %556) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %542, <4 x float> %523) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %558 = shufflevector <8 x half> %454, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %559 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %558, <4 x float> %525) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %560 = fsub <4 x float> %370, %430, !dbg !41
  %561 = extractelement <4 x float> %560, i64 0, !dbg !48
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %561), !dbg !48
  %local.reduce44 = fadd float %562, %k0.sum.anchor1624, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1632 = tail call float asm sideeffect "", "=v,0"(float %local.reduce44) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> poison, float %562, i64 0, !dbg !49
  %564 = extractelement <4 x float> %560, i64 1, !dbg !48
  %565 = tail call float @llvm.amdgcn.exp2.f32(float %564), !dbg !48
  %local.reduce45 = fadd float %565, %k0.sum.anchor1625, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1633 = tail call float asm sideeffect "", "=v,0"(float %local.reduce45) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %566 = insertelement <4 x float> %563, float %565, i64 1, !dbg !49
  %567 = extractelement <4 x float> %560, i64 2, !dbg !48
  %568 = tail call float @llvm.amdgcn.exp2.f32(float %567), !dbg !48
  %local.reduce46 = fadd float %568, %k0.sum.anchor1626, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1634 = tail call float asm sideeffect "", "=v,0"(float %local.reduce46) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %569 = insertelement <4 x float> %566, float %568, i64 2, !dbg !49
  %570 = extractelement <4 x float> %560, i64 3, !dbg !48
  %571 = tail call float @llvm.amdgcn.exp2.f32(float %570), !dbg !48
  %local.reduce47 = fadd float %571, %k0.sum.anchor1627, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1635 = tail call float asm sideeffect "", "=v,0"(float %local.reduce47) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %572 = insertelement <4 x float> %569, float %571, i64 3, !dbg !49
  %math.result339 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %572) #8, !dbg !49, !frisk.register_lifetimes !5
  %573 = fptrunc <4 x float> %math.result339 to <4 x half>, !dbg !49
  %math.packed340 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %573) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %574 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %542, <4 x float> %540) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %575 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %558, <4 x float> %541) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %576 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %577 = fsub <4 x float> %327, %429, !dbg !41
  %578 = extractelement <4 x float> %577, i64 0, !dbg !48
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !48
  %local.reduce79 = fadd float %579, %k0.sum.anchor1628, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1636 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %580 = insertelement <4 x float> poison, float %579, i64 0, !dbg !49
  %581 = extractelement <4 x float> %577, i64 1, !dbg !48
  %582 = tail call float @llvm.amdgcn.exp2.f32(float %581), !dbg !48
  %local.reduce80 = fadd float %582, %k0.sum.anchor1629, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1637 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %583 = insertelement <4 x float> %580, float %582, i64 1, !dbg !49
  %584 = extractelement <4 x float> %577, i64 2, !dbg !48
  %585 = tail call float @llvm.amdgcn.exp2.f32(float %584), !dbg !48
  %local.reduce81 = fadd float %585, %k0.sum.anchor1630, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1638 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %586 = insertelement <4 x float> %583, float %585, i64 2, !dbg !49
  %587 = extractelement <4 x float> %577, i64 3, !dbg !48
  %588 = tail call float @llvm.amdgcn.exp2.f32(float %587), !dbg !48
  %local.reduce82 = fadd float %588, %k0.sum.anchor1631, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1639 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %589 = insertelement <4 x float> %586, float %588, i64 3, !dbg !49
  %math.result325 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %589) #8, !dbg !49, !frisk.register_lifetimes !5
  %590 = fptrunc <4 x float> %math.result325 to <4 x half>, !dbg !49
  %math.packed326 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %590) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %576, <4 x float> %557) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %592 = shufflevector <8 x half> %456, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %593 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %592, <4 x float> %559) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %594 = fsub <4 x float> %376, %430, !dbg !41
  %595 = extractelement <4 x float> %594, i64 0, !dbg !48
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %595), !dbg !48
  %local.reduce48 = fadd float %596, %k0.sum.anchor1632, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1640 = tail call float asm sideeffect "", "=v,0"(float %local.reduce48) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %597 = insertelement <4 x float> poison, float %596, i64 0, !dbg !49
  %598 = extractelement <4 x float> %594, i64 1, !dbg !48
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %598), !dbg !48
  %local.reduce49 = fadd float %599, %k0.sum.anchor1633, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1641 = tail call float asm sideeffect "", "=v,0"(float %local.reduce49) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %600 = insertelement <4 x float> %597, float %599, i64 1, !dbg !49
  %601 = extractelement <4 x float> %594, i64 2, !dbg !48
  %602 = tail call float @llvm.amdgcn.exp2.f32(float %601), !dbg !48
  %local.reduce50 = fadd float %602, %k0.sum.anchor1634, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1642 = tail call float asm sideeffect "", "=v,0"(float %local.reduce50) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %603 = insertelement <4 x float> %600, float %602, i64 2, !dbg !49
  %604 = extractelement <4 x float> %594, i64 3, !dbg !48
  %605 = tail call float @llvm.amdgcn.exp2.f32(float %604), !dbg !48
  %local.reduce51 = fadd float %605, %k0.sum.anchor1635, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1643 = tail call float asm sideeffect "", "=v,0"(float %local.reduce51) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %606 = insertelement <4 x float> %603, float %605, i64 3, !dbg !49
  %math.result341 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %606) #8, !dbg !49, !frisk.register_lifetimes !5
  %607 = fptrunc <4 x float> %math.result341 to <4 x half>, !dbg !49
  %math.packed342 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %607) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %608 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %576, <4 x float> %574) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %609 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %592, <4 x float> %575) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %610 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %611 = fsub <4 x float> %333, %429, !dbg !41
  %612 = extractelement <4 x float> %611, i64 0, !dbg !48
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !48
  %local.reduce83 = fadd float %613, %k0.sum.anchor1636, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1644 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %614 = insertelement <4 x float> poison, float %613, i64 0, !dbg !49
  %615 = extractelement <4 x float> %611, i64 1, !dbg !48
  %616 = tail call float @llvm.amdgcn.exp2.f32(float %615), !dbg !48
  %local.reduce84 = fadd float %616, %k0.sum.anchor1637, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1645 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %617 = insertelement <4 x float> %614, float %616, i64 1, !dbg !49
  %618 = extractelement <4 x float> %611, i64 2, !dbg !48
  %619 = tail call float @llvm.amdgcn.exp2.f32(float %618), !dbg !48
  %local.reduce85 = fadd float %619, %k0.sum.anchor1638, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1646 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %620 = insertelement <4 x float> %617, float %619, i64 2, !dbg !49
  %621 = extractelement <4 x float> %611, i64 3, !dbg !48
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %621), !dbg !48
  %local.reduce86 = fadd float %622, %k0.sum.anchor1639, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1647 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %623 = insertelement <4 x float> %620, float %622, i64 3, !dbg !49
  %math.result327 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #8, !dbg !49, !frisk.register_lifetimes !5
  %624 = fptrunc <4 x float> %math.result327 to <4 x half>, !dbg !49
  %math.packed328 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %625 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %610, <4 x float> %591) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %626 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %627 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %626, <4 x float> %593) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %628 = fsub <4 x float> %382, %430, !dbg !41
  %629 = extractelement <4 x float> %628, i64 0, !dbg !48
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %629), !dbg !48
  %local.reduce52 = fadd float %630, %k0.sum.anchor1640, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1648 = tail call float asm sideeffect "", "=v,0"(float %local.reduce52) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %631 = insertelement <4 x float> poison, float %630, i64 0, !dbg !49
  %632 = extractelement <4 x float> %628, i64 1, !dbg !48
  %633 = tail call float @llvm.amdgcn.exp2.f32(float %632), !dbg !48
  %local.reduce53 = fadd float %633, %k0.sum.anchor1641, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1649 = tail call float asm sideeffect "", "=v,0"(float %local.reduce53) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %634 = insertelement <4 x float> %631, float %633, i64 1, !dbg !49
  %635 = extractelement <4 x float> %628, i64 2, !dbg !48
  %636 = tail call float @llvm.amdgcn.exp2.f32(float %635), !dbg !48
  %local.reduce54 = fadd float %636, %k0.sum.anchor1642, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1650 = tail call float asm sideeffect "", "=v,0"(float %local.reduce54) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %637 = insertelement <4 x float> %634, float %636, i64 2, !dbg !49
  %638 = extractelement <4 x float> %628, i64 3, !dbg !48
  %639 = tail call float @llvm.amdgcn.exp2.f32(float %638), !dbg !48
  %local.reduce55 = fadd float %639, %k0.sum.anchor1643, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1651 = tail call float asm sideeffect "", "=v,0"(float %local.reduce55) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %640 = insertelement <4 x float> %637, float %639, i64 3, !dbg !49
  %math.result343 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %640) #8, !dbg !49, !frisk.register_lifetimes !5
  %641 = fptrunc <4 x float> %math.result343 to <4 x half>, !dbg !49
  %math.packed344 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %641) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %642 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %610, <4 x float> %608) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %643 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %626, <4 x float> %609) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %644 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %645 = fsub <4 x float> %339, %429, !dbg !41
  %646 = extractelement <4 x float> %645, i64 0, !dbg !48
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %646), !dbg !48
  %local.reduce87 = fadd float %647, %k0.sum.anchor1644, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1652 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %648 = insertelement <4 x float> poison, float %647, i64 0, !dbg !49
  %649 = extractelement <4 x float> %645, i64 1, !dbg !48
  %650 = tail call float @llvm.amdgcn.exp2.f32(float %649), !dbg !48
  %local.reduce88 = fadd float %650, %k0.sum.anchor1645, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1653 = tail call float asm sideeffect "", "=v,0"(float %local.reduce88) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %651 = insertelement <4 x float> %648, float %650, i64 1, !dbg !49
  %652 = extractelement <4 x float> %645, i64 2, !dbg !48
  %653 = tail call float @llvm.amdgcn.exp2.f32(float %652), !dbg !48
  %local.reduce89 = fadd float %653, %k0.sum.anchor1646, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1654 = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %654 = insertelement <4 x float> %651, float %653, i64 2, !dbg !49
  %655 = extractelement <4 x float> %645, i64 3, !dbg !48
  %656 = tail call float @llvm.amdgcn.exp2.f32(float %655), !dbg !48
  %local.reduce90 = fadd float %656, %k0.sum.anchor1647, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1655 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %657 = insertelement <4 x float> %654, float %656, i64 3, !dbg !49
  %math.result329 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %657) #8, !dbg !49, !frisk.register_lifetimes !5
  %658 = fptrunc <4 x float> %math.result329 to <4 x half>, !dbg !49
  %math.packed330 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %658) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %659 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %644, <4 x float> %625) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %660 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %661 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %660, <4 x float> %627) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %662 = fsub <4 x float> %388, %430, !dbg !41
  %663 = extractelement <4 x float> %662, i64 0, !dbg !48
  %664 = tail call float @llvm.amdgcn.exp2.f32(float %663), !dbg !48
  %local.reduce56 = fadd float %664, %k0.sum.anchor1648, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1656 = tail call float asm sideeffect "", "=v,0"(float %local.reduce56) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %665 = insertelement <4 x float> poison, float %664, i64 0, !dbg !49
  %666 = extractelement <4 x float> %662, i64 1, !dbg !48
  %667 = tail call float @llvm.amdgcn.exp2.f32(float %666), !dbg !48
  %local.reduce57 = fadd float %667, %k0.sum.anchor1649, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1657 = tail call float asm sideeffect "", "=v,0"(float %local.reduce57) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %668 = insertelement <4 x float> %665, float %667, i64 1, !dbg !49
  %669 = extractelement <4 x float> %662, i64 2, !dbg !48
  %670 = tail call float @llvm.amdgcn.exp2.f32(float %669), !dbg !48
  %local.reduce58 = fadd float %670, %k0.sum.anchor1650, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1658 = tail call float asm sideeffect "", "=v,0"(float %local.reduce58) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %671 = insertelement <4 x float> %668, float %670, i64 2, !dbg !49
  %672 = extractelement <4 x float> %662, i64 3, !dbg !48
  %673 = tail call float @llvm.amdgcn.exp2.f32(float %672), !dbg !48
  %local.reduce59 = fadd float %673, %k0.sum.anchor1651, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1659 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %674 = insertelement <4 x float> %671, float %673, i64 3, !dbg !49
  %math.result345 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %674) #8, !dbg !49, !frisk.register_lifetimes !5
  %675 = fptrunc <4 x float> %math.result345 to <4 x half>, !dbg !49
  %math.packed346 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %675) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %676 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %644, <4 x float> %642) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %677 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %660, <4 x float> %643) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %678 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %679 = fsub <4 x float> %345, %429, !dbg !41
  %680 = extractelement <4 x float> %679, i64 0, !dbg !48
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %680), !dbg !48
  %local.reduce91 = fadd float %681, %k0.sum.anchor1652, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1660 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %682 = insertelement <4 x float> poison, float %681, i64 0, !dbg !49
  %683 = extractelement <4 x float> %679, i64 1, !dbg !48
  %684 = tail call float @llvm.amdgcn.exp2.f32(float %683), !dbg !48
  %local.reduce92 = fadd float %684, %k0.sum.anchor1653, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1661 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %685 = insertelement <4 x float> %682, float %684, i64 1, !dbg !49
  %686 = extractelement <4 x float> %679, i64 2, !dbg !48
  %687 = tail call float @llvm.amdgcn.exp2.f32(float %686), !dbg !48
  %local.reduce93 = fadd float %687, %k0.sum.anchor1654, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1662 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %688 = insertelement <4 x float> %685, float %687, i64 2, !dbg !49
  %689 = extractelement <4 x float> %679, i64 3, !dbg !48
  %690 = tail call float @llvm.amdgcn.exp2.f32(float %689), !dbg !48
  %local.reduce94 = fadd float %690, %k0.sum.anchor1655, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1663 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %691 = insertelement <4 x float> %688, float %690, i64 3, !dbg !49
  %math.result331 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %691) #8, !dbg !49, !frisk.register_lifetimes !5
  %692 = fptrunc <4 x float> %math.result331 to <4 x half>, !dbg !49
  %math.packed332 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %692) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %693 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %678, <4 x float> %659) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %694 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %695 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %694, <4 x float> %661) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %696 = fsub <4 x float> %394, %430, !dbg !41
  %697 = extractelement <4 x float> %696, i64 0, !dbg !48
  %698 = tail call float @llvm.amdgcn.exp2.f32(float %697), !dbg !48
  %local.reduce60 = fadd float %698, %k0.sum.anchor1656, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1664 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %699 = insertelement <4 x float> poison, float %698, i64 0, !dbg !49
  %700 = extractelement <4 x float> %696, i64 1, !dbg !48
  %701 = tail call float @llvm.amdgcn.exp2.f32(float %700), !dbg !48
  %local.reduce61 = fadd float %701, %k0.sum.anchor1657, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1665 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %702 = insertelement <4 x float> %699, float %701, i64 1, !dbg !49
  %703 = extractelement <4 x float> %696, i64 2, !dbg !48
  %704 = tail call float @llvm.amdgcn.exp2.f32(float %703), !dbg !48
  %local.reduce62 = fadd float %704, %k0.sum.anchor1658, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1666 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %705 = insertelement <4 x float> %702, float %704, i64 2, !dbg !49
  %706 = extractelement <4 x float> %696, i64 3, !dbg !48
  %707 = tail call float @llvm.amdgcn.exp2.f32(float %706), !dbg !48
  %local.reduce63 = fadd float %707, %k0.sum.anchor1659, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1667 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %708 = insertelement <4 x float> %705, float %707, i64 3, !dbg !49
  %math.result347 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %708) #8, !dbg !49, !frisk.register_lifetimes !5
  %709 = fptrunc <4 x float> %math.result347 to <4 x half>, !dbg !49
  %math.packed348 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %709) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %710 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %678, <4 x float> %676) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %711 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %694, <4 x float> %677) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %712 = bitcast <4 x i32> %463 to <8 x half>, !dbg !46
  %713 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %714 = fsub <4 x float> %351, %429, !dbg !41
  %715 = extractelement <4 x float> %714, i64 0, !dbg !48
  %716 = tail call float @llvm.amdgcn.exp2.f32(float %715), !dbg !48
  %local.reduce95 = fadd float %716, %k0.sum.anchor1660, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1668 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %717 = insertelement <4 x float> poison, float %716, i64 0, !dbg !49
  %718 = extractelement <4 x float> %714, i64 1, !dbg !48
  %719 = tail call float @llvm.amdgcn.exp2.f32(float %718), !dbg !48
  %local.reduce96 = fadd float %719, %k0.sum.anchor1661, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1669 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce99 = fadd float %k0.sum.anchor1668, %k0.sum.anchor1669, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1670 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %720 = insertelement <4 x float> %717, float %719, i64 1, !dbg !49
  %721 = extractelement <4 x float> %714, i64 2, !dbg !48
  %722 = tail call float @llvm.amdgcn.exp2.f32(float %721), !dbg !48
  %local.reduce97 = fadd float %722, %k0.sum.anchor1662, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1671 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %723 = insertelement <4 x float> %720, float %722, i64 2, !dbg !49
  %724 = extractelement <4 x float> %714, i64 3, !dbg !48
  %725 = tail call float @llvm.amdgcn.exp2.f32(float %724), !dbg !48
  %local.reduce98 = fadd float %725, %k0.sum.anchor1663, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1672 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce100 = fadd float %k0.sum.anchor1671, %k0.sum.anchor1672, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1673 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %726 = fadd float %k0.sum.anchor1670, %k0.sum.anchor1673, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1674 = tail call float asm sideeffect "", "=v,0"(float %726) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %727 = fadd float %438, %k0.sum.anchor1674, !dbg !50
  %k0.sum.anchor1675 = tail call float asm sideeffect "", "=v,0"(float %727) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %728 = insertelement <1 x float> poison, float %k0.sum.anchor1675, i64 0, !dbg !12
  %729 = insertvalue [2 x <1 x float>] poison, <1 x float> %728, 0, !dbg !12
  %730 = insertelement <4 x float> %723, float %725, i64 3, !dbg !49
  %math.result333 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %730) #8, !dbg !49, !frisk.register_lifetimes !5
  %731 = fptrunc <4 x float> %math.result333 to <4 x half>, !dbg !49
  %math.packed334 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %731) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %732 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %713, <4 x float> %693) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %733 = shufflevector <8 x half> %712, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %734 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %733, <4 x float> %695) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %735 = fsub <4 x float> %400, %430, !dbg !41
  %736 = extractelement <4 x float> %735, i64 0, !dbg !48
  %737 = tail call float @llvm.amdgcn.exp2.f32(float %736), !dbg !48
  %local.reduce64 = fadd float %737, %k0.sum.anchor1664, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1676 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %738 = insertelement <4 x float> poison, float %737, i64 0, !dbg !49
  %739 = extractelement <4 x float> %735, i64 1, !dbg !48
  %740 = tail call float @llvm.amdgcn.exp2.f32(float %739), !dbg !48
  %local.reduce65 = fadd float %740, %k0.sum.anchor1665, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1677 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce68 = fadd float %k0.sum.anchor1676, %k0.sum.anchor1677, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1678 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %741 = insertelement <4 x float> %738, float %740, i64 1, !dbg !49
  %742 = extractelement <4 x float> %735, i64 2, !dbg !48
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !48
  %local.reduce66 = fadd float %743, %k0.sum.anchor1666, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1679 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %744 = insertelement <4 x float> %741, float %743, i64 2, !dbg !49
  %745 = extractelement <4 x float> %735, i64 3, !dbg !48
  %746 = tail call float @llvm.amdgcn.exp2.f32(float %745), !dbg !48
  %local.reduce67 = fadd float %746, %k0.sum.anchor1667, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1680 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce69 = fadd float %k0.sum.anchor1679, %k0.sum.anchor1680, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1681 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %747 = fadd float %k0.sum.anchor1678, %k0.sum.anchor1681, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1682 = tail call float asm sideeffect "", "=v,0"(float %747) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %748 = fadd float %446, %k0.sum.anchor1682, !dbg !50
  %k0.sum.anchor1683 = tail call float asm sideeffect "", "=v,0"(float %748) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %749 = insertelement <1 x float> poison, float %k0.sum.anchor1683, i64 0, !dbg !12
  %750 = insertvalue [2 x <1 x float>] %729, <1 x float> %749, 1, !dbg !12
  %751 = insertelement <4 x float> %744, float %746, i64 3, !dbg !49
  %math.result349 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %751) #8, !dbg !49, !frisk.register_lifetimes !5
  %752 = fptrunc <4 x float> %math.result349 to <4 x half>, !dbg !49
  %math.packed350 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %752) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %753 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %713, <4 x float> %710) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %754 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %733, <4 x float> %711) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1195 = or disjoint i64 %address.local695, %183, !dbg !39
  %vdb.address1196 = trunc nuw nsw i64 %vdb.address1195 to i32, !dbg !39
  %vdb.address1197 = shl nuw nsw i32 %vdb.address1196, 1, !dbg !39
  %vdb.address1202 = or disjoint i64 %address.local701, %183, !dbg !39
  %vdb.address1203 = trunc nuw nsw i64 %vdb.address1202 to i32, !dbg !39
  %vdb.address1204 = shl nuw nsw i32 %vdb.address1203, 1, !dbg !39
  %755 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %vdb.address1197, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %756 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1204, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local706, i64 8192, !dbg !46
  %757 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %758 = bitcast <4 x i32> %757 to <8 x half>, !dbg !46
  %vdb.read1188 = getelementptr i8, ptr addrspace(3) %address.local706, i64 9216, !dbg !46
  %759 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1188, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %760 = bitcast <4 x i32> %759 to <8 x half>, !dbg !46
  %vdb.read1189 = getelementptr i8, ptr addrspace(3) %address.local706, i64 10240, !dbg !46
  %761 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1189, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %762 = bitcast <4 x i32> %761 to <8 x half>, !dbg !46
  %vdb.read1190 = getelementptr i8, ptr addrspace(3) %address.local706, i64 11264, !dbg !46
  %763 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1190, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %764 = bitcast <4 x i32> %763 to <8 x half>, !dbg !46
  %vdb.read1191 = getelementptr i8, ptr addrspace(3) %address.local706, i64 12288, !dbg !46
  %765 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1191, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %766 = bitcast <4 x i32> %765 to <8 x half>, !dbg !46
  %vdb.read1192 = getelementptr i8, ptr addrspace(3) %address.local706, i64 13312, !dbg !46
  %767 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1192, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %768 = bitcast <4 x i32> %767 to <8 x half>, !dbg !46
  %vdb.read1193 = getelementptr i8, ptr addrspace(3) %address.local706, i64 14336, !dbg !46
  %769 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1193, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %770 = bitcast <4 x i32> %769 to <8 x half>, !dbg !46
  %vdb.read1194 = getelementptr i8, ptr addrspace(3) %address.local706, i64 15360, !dbg !46
  %771 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1194, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %772 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %773 = extractvalue [2 x <8 x float>] %113, 0, !dbg !44
  %774 = shufflevector <8 x float> %773, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %775 = fmul <4 x float> %774, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %776 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %772, <4 x float> %775) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %777 = shufflevector <8 x half> %758, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %778 = shufflevector <8 x float> %773, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %779 = fmul <4 x float> %778, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %780 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %777, <4 x float> %779) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %781 = extractvalue [2 x <8 x float>] %113, 1, !dbg !44
  %782 = shufflevector <8 x float> %781, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %783 = fmul <4 x float> %782, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %772, <4 x float> %783) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %785 = shufflevector <8 x float> %781, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %786 = fmul <4 x float> %785, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %787 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %777, <4 x float> %786) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %788 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %789 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %788, <4 x float> %776) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %790 = shufflevector <8 x half> %760, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %791 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %790, <4 x float> %780) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %788, <4 x float> %784) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %793 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %790, <4 x float> %787) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %794 = shufflevector <8 x half> %762, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %794, <4 x float> %789) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %796 = shufflevector <8 x half> %762, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %796, <4 x float> %791) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %798 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %794, <4 x float> %792) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %799 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %796, <4 x float> %793) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %800 = shufflevector <8 x half> %764, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %800, <4 x float> %795) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %802 = shufflevector <8 x half> %764, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %802, <4 x float> %797) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %804 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %800, <4 x float> %798) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %802, <4 x float> %799) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %806 = shufflevector <8 x half> %766, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %806, <4 x float> %801) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %808 = shufflevector <8 x half> %766, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %809 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %808, <4 x float> %803) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %810 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %806, <4 x float> %804) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %808, <4 x float> %805) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %812 = shufflevector <8 x half> %768, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %812, <4 x float> %807) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %814 = shufflevector <8 x half> %768, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %814, <4 x float> %809) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %812, <4 x float> %810) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %814, <4 x float> %811) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %818 = shufflevector <8 x half> %770, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %818, <4 x float> %813) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %820 = shufflevector <8 x half> %770, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %820, <4 x float> %815) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %822 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %818, <4 x float> %816) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %823 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %820, <4 x float> %817) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %824 = bitcast <4 x i32> %771 to <8 x half>, !dbg !46
  %825 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %826 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %825, <4 x float> %819) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %827 = shufflevector <8 x half> %824, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %828 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %827, <4 x float> %821) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %829 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %825, <4 x float> %822) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %827, <4 x float> %823) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1209 = or disjoint i64 %address.local695, %236, !dbg !39
  %vdb.address1210 = trunc nuw nsw i64 %vdb.address1209 to i32, !dbg !39
  %vdb.address1211 = shl nuw nsw i32 %vdb.address1210, 1, !dbg !39
  %vdb.address1216 = or disjoint i64 %address.local701, %236, !dbg !39
  %vdb.address1217 = trunc nuw nsw i64 %vdb.address1216 to i32, !dbg !39
  %vdb.address1218 = shl nuw nsw i32 %vdb.address1217, 1, !dbg !39
  %831 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1211, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %832 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1187, i32 %vdb.address1218, <4 x i32> %dma.buffer.rsrc1096) #9, !dbg !39
  %833 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local706, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %834 = bitcast <4 x i32> %833 to <8 x half>, !dbg !46
  %835 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local708, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %836 = bitcast <4 x i32> %835 to <8 x half>, !dbg !46
  %837 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local710, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %838 = bitcast <4 x i32> %837 to <8 x half>, !dbg !46
  %839 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local712, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %840 = bitcast <4 x i32> %839 to <8 x half>, !dbg !46
  %841 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local714, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %842 = bitcast <4 x i32> %841 to <8 x half>, !dbg !46
  %843 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local716, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %844 = bitcast <4 x i32> %843 to <8 x half>, !dbg !46
  %845 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local718, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %846 = bitcast <4 x i32> %845 to <8 x half>, !dbg !46
  %847 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local720, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %848 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %849 = extractvalue [2 x <8 x float>] %112, 0, !dbg !44
  %850 = shufflevector <8 x float> %849, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %851 = fmul <4 x float> %850, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %852 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %848, <4 x float> %851) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %853 = shufflevector <8 x half> %834, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %854 = shufflevector <8 x float> %849, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %855 = fmul <4 x float> %854, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %856 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %853, <4 x float> %855) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %857 = extractvalue [2 x <8 x float>] %112, 1, !dbg !44
  %858 = shufflevector <8 x float> %857, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %859 = fmul <4 x float> %858, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %848, <4 x float> %859) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %861 = shufflevector <8 x float> %857, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %862 = fmul <4 x float> %861, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %863 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %853, <4 x float> %862) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %864 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %865 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %864, <4 x float> %852) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %866 = shufflevector <8 x half> %836, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %867 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %866, <4 x float> %856) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %864, <4 x float> %860) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %869 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %866, <4 x float> %863) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %870 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %870, <4 x float> %865) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %872 = shufflevector <8 x half> %838, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %872, <4 x float> %867) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %874 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %870, <4 x float> %868) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %872, <4 x float> %869) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %876 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %876, <4 x float> %871) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %878 = shufflevector <8 x half> %840, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %878, <4 x float> %873) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %880 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %876, <4 x float> %874) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %878, <4 x float> %875) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %882 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %882, <4 x float> %877) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %884 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %884, <4 x float> %879) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %886 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %882, <4 x float> %880) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %884, <4 x float> %881) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %888 = shufflevector <8 x half> %844, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %888, <4 x float> %883) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %890 = shufflevector <8 x half> %844, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %890, <4 x float> %885) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %888, <4 x float> %886) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %890, <4 x float> %887) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %894 = shufflevector <8 x half> %846, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %894, <4 x float> %889) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %896 = shufflevector <8 x half> %846, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %896, <4 x float> %891) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %894, <4 x float> %892) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %896, <4 x float> %893) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %900 = bitcast <4 x i32> %847 to <8 x half>, !dbg !46
  %901 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %901, <4 x float> %895) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %903 = shufflevector <8 x half> %900, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %903, <4 x float> %897) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %901, <4 x float> %898) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %903, <4 x float> %899) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %907 = add nuw nsw i64 %117, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %117, 31, !dbg !46
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !46

k0.issue:                                         ; preds = %110
  %k0.next. = shl nuw nsw i64 %907, 14, !dbg !13
  %k0.next.1694 = add nuw nsw i64 %k0.next., %k0.prime.1684, !dbg !13
  %k0.next.kdma.source1243 = or disjoint i64 %k0.next.1694, %k0.prime.kdma.source1242, !dbg !13
  %k0.next.kdma.source1244 = trunc nuw nsw i64 %k0.next.kdma.source1243 to i32, !dbg !13
  %k0.next.kdma.source1245 = shl nuw nsw i32 %k0.next.kdma.source1244, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1245, %kdma.intra, !dbg !13
  %k0.next.1695 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1257 = or disjoint i64 %k0.next.1694, %k0.prime.kdma.source1256, !dbg !13
  %k0.next.kdma.source1258 = trunc nuw nsw i64 %k0.next.kdma.source1257 to i32, !dbg !13
  %k0.next.kdma.source1259 = shl nuw nsw i32 %k0.next.kdma.source1258, 1, !dbg !13
  %k0.next.kdma.offset1260 = or disjoint i32 %k0.next.kdma.source1259, %kdma.intra.hi, !dbg !13
  %k0.next.1696 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1261, i32 %k0.next.kdma.offset1260, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1274 = or disjoint i64 %k0.next.1694, %k0.prime.kdma.source1273, !dbg !13
  %k0.next.kdma.source1275 = trunc nuw nsw i64 %k0.next.kdma.source1274 to i32, !dbg !13
  %k0.next.kdma.source1276 = shl nuw nsw i32 %k0.next.kdma.source1275, 1, !dbg !13
  %k0.next.kdma.offset1277 = or disjoint i32 %k0.next.kdma.source1276, %kdma.intra, !dbg !13
  %k0.next.1697 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset1277, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1291 = or disjoint i64 %k0.next.1694, %k0.prime.kdma.source1290, !dbg !13
  %k0.next.kdma.source1292 = trunc nuw nsw i64 %k0.next.kdma.source1291 to i32, !dbg !13
  %k0.next.kdma.source1293 = shl nuw nsw i32 %k0.next.kdma.source1292, 1, !dbg !13
  %k0.next.kdma.offset1294 = or disjoint i32 %k0.next.kdma.source1293, %kdma.intra.hi, !dbg !13
  %k0.next.1698 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1295, i32 %k0.next.kdma.offset1294, <4 x i32> %k0.prime.dma.buffer.rsrc1112) #9, !dbg !13, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %110
  %908 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %909 = bitcast <4 x i32> %908 to <8 x half>, !dbg !46
  %910 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1188, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %911 = bitcast <4 x i32> %910 to <8 x half>, !dbg !46
  %912 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1189, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %913 = bitcast <4 x i32> %912 to <8 x half>, !dbg !46
  %914 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1190, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %915 = bitcast <4 x i32> %914 to <8 x half>, !dbg !46
  %916 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1191, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %917 = bitcast <4 x i32> %916 to <8 x half>, !dbg !46
  %918 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1192, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %919 = bitcast <4 x i32> %918 to <8 x half>, !dbg !46
  %920 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1193, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %921 = bitcast <4 x i32> %920 to <8 x half>, !dbg !46
  %922 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1194, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %923 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %924 = extractvalue [2 x <8 x float>] %111, 0, !dbg !44
  %925 = shufflevector <8 x float> %924, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %926 = fmul <4 x float> %925, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %923, <4 x float> %926) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %928 = shufflevector <8 x half> %909, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %929 = shufflevector <8 x float> %924, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %930 = fmul <4 x float> %929, %435, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %928, <4 x float> %930) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %932 = extractvalue [2 x <8 x float>] %111, 1, !dbg !44
  %933 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %934 = fmul <4 x float> %933, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %923, <4 x float> %934) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %936 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %937 = fmul <4 x float> %936, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed336, <4 x half> %928, <4 x float> %937) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %939 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %940 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %939, <4 x float> %927) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %941 = shufflevector <8 x half> %911, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %942 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed322, <4 x half> %941, <4 x float> %931) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %939, <4 x float> %935) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %944 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed338, <4 x half> %941, <4 x float> %938) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %945 = shufflevector <8 x half> %913, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %945, <4 x float> %940) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %947 = shufflevector <8 x half> %913, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed324, <4 x half> %947, <4 x float> %942) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %949 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %945, <4 x float> %943) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed340, <4 x half> %947, <4 x float> %944) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %951 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %951, <4 x float> %946) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %953 = shufflevector <8 x half> %915, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed326, <4 x half> %953, <4 x float> %948) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %951, <4 x float> %949) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed342, <4 x half> %953, <4 x float> %950) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %957 = shufflevector <8 x half> %917, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %957, <4 x float> %952) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %959 = shufflevector <8 x half> %917, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed328, <4 x half> %959, <4 x float> %954) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %957, <4 x float> %955) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed344, <4 x half> %959, <4 x float> %956) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %963 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %963, <4 x float> %958) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %965 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed330, <4 x half> %965, <4 x float> %960) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %963, <4 x float> %961) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed346, <4 x half> %965, <4 x float> %962) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %969 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %969, <4 x float> %964) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %971 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed332, <4 x half> %971, <4 x float> %966) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %973 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %969, <4 x float> %967) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed348, <4 x half> %971, <4 x float> %968) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %975 = bitcast <4 x i32> %922 to <8 x half>, !dbg !46
  %976 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %977 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %976, <4 x float> %970) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %978 = shufflevector <8 x half> %975, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %979 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed334, <4 x half> %978, <4 x float> %972) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %976, <4 x float> %973) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed350, <4 x half> %978, <4 x float> %974) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %982 = shufflevector <4 x float> %732, <4 x float> %734, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %983 = insertvalue [2 x <8 x float>] poison, <8 x float> %982, 0, !dbg !12
  %984 = shufflevector <4 x float> %753, <4 x float> %754, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %985 = insertvalue [2 x <8 x float>] %983, <8 x float> %984, 1, !dbg !12
  %986 = insertvalue [2 x <1 x float>] poison, <1 x float> %425, 0, !dbg !12
  %987 = insertvalue [2 x <1 x float>] %986, <1 x float> %428, 1, !dbg !12
  %988 = shufflevector <4 x float> %826, <4 x float> %828, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %989 = insertvalue [2 x <8 x float>] poison, <8 x float> %988, 0, !dbg !12
  %990 = shufflevector <4 x float> %829, <4 x float> %830, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %991 = insertvalue [2 x <8 x float>] %989, <8 x float> %990, 1, !dbg !12
  %992 = shufflevector <4 x float> %902, <4 x float> %904, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %993 = insertvalue [2 x <8 x float>] poison, <8 x float> %992, 0, !dbg !12
  %994 = shufflevector <4 x float> %905, <4 x float> %906, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %995 = insertvalue [2 x <8 x float>] %993, <8 x float> %994, 1, !dbg !12
  %996 = shufflevector <4 x float> %977, <4 x float> %979, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %997 = insertvalue [2 x <8 x float>] poison, <8 x float> %996, 0, !dbg !12
  %998 = shufflevector <4 x float> %980, <4 x float> %981, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %999 = insertvalue [2 x <8 x float>] %997, <8 x float> %998, 1, !dbg !12
  %1000 = icmp samesign ult i64 %117, 31, !dbg !12
  br i1 %1000, label %110, label %1001, !dbg !12

1001:                                             ; preds = %k0.join
  %1002 = shl nuw nsw i32 %8, 2, !dbg !12
  %1003 = xor i32 %1002, 64, !dbg !12
  %1004 = bitcast float %k0.sum.anchor1683 to i32, !dbg !12
  %lane.sum.shuffle170 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1003, i32 %1004), !dbg !12
  %1005 = bitcast i32 %lane.sum.shuffle170 to float, !dbg !12
  %lane.sum.final171 = fadd float %k0.sum.anchor1683, %1005, !dbg !12
  %1006 = xor i32 %1002, 128, !dbg !12
  %1007 = bitcast float %lane.sum.final171 to i32, !dbg !12
  %lane.sum.shuffle172 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1006, i32 %1007), !dbg !12
  %1008 = bitcast i32 %lane.sum.shuffle172 to float, !dbg !12
  %lane.sum.final173 = fadd float %lane.sum.final171, %1008, !dbg !12
  %1009 = insertelement <1 x float> poison, float %lane.sum.final173, i64 0, !dbg !12
  %1010 = bitcast float %k0.sum.anchor1675 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1003, i32 %1010), !dbg !12
  %1011 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor1675, %1011, !dbg !12
  %1012 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle164 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1006, i32 %1012), !dbg !12
  %1013 = bitcast i32 %lane.sum.shuffle164 to float, !dbg !12
  %lane.sum.final165 = fadd float %lane.sum.final, %1013, !dbg !12
  %1014 = insertelement <1 x float> poison, float %lane.sum.final165, i64 0, !dbg !12
  %1015 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1015, 0, !dbg !12
  %1016 = and i32 %kdma.tid, 32, !dbg !12
  %.not18 = icmp eq i32 %1016, 0, !dbg !12
  %1017 = extractelement <4 x float> %977, i64 0, !dbg !12
  %1018 = extractelement <4 x float> %977, i64 1, !dbg !12
  %1019 = extractelement <4 x float> %977, i64 2, !dbg !12
  %1020 = extractelement <4 x float> %977, i64 3, !dbg !12
  %1021 = select i1 %.not, float %1018, float %1017, !dbg !12
  %1022 = bitcast float %1021 to i32, !dbg !12
  %1023 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1022), !dbg !12
  %1024 = bitcast i32 %1023 to float, !dbg !12
  %1025 = select i1 %.not, float %1017, float %1024, !dbg !12
  %1026 = select i1 %.not, float %1024, float %1018, !dbg !12
  %1027 = select i1 %.not, float %1020, float %1019, !dbg !12
  %1028 = bitcast float %1027 to i32, !dbg !12
  %1029 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1028), !dbg !12
  %1030 = bitcast i32 %1029 to float, !dbg !12
  %1031 = select i1 %.not, float %1019, float %1030, !dbg !12
  %1032 = select i1 %.not, float %1030, float %1020, !dbg !12
  %1033 = select i1 %.not18, float %1031, float %1025, !dbg !12
  %1034 = bitcast float %1033 to i32, !dbg !12
  %1035 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1034), !dbg !12
  %1036 = bitcast i32 %1035 to float, !dbg !12
  %1037 = select i1 %.not18, float %1032, float %1026, !dbg !12
  %1038 = bitcast float %1037 to i32, !dbg !12
  %1039 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1038), !dbg !12
  %1040 = bitcast i32 %1039 to float, !dbg !12
  %1041 = insertelement <4 x i1> poison, i1 %.not18, i64 0, !dbg !12
  %1042 = shufflevector <4 x i1> %1041, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1043 = insertelement <4 x float> poison, float %1025, i64 0, !dbg !12
  %1044 = insertelement <4 x float> %1043, float %1026, i64 1, !dbg !12
  %1045 = insertelement <4 x float> %1044, float %1036, i64 2, !dbg !12
  %1046 = insertelement <4 x float> %1045, float %1040, i64 3, !dbg !12
  %1047 = insertelement <4 x float> poison, float %1036, i64 0, !dbg !12
  %1048 = insertelement <4 x float> %1047, float %1040, i64 1, !dbg !12
  %1049 = insertelement <4 x float> %1048, float %1031, i64 2, !dbg !12
  %1050 = insertelement <4 x float> %1049, float %1032, i64 3, !dbg !12
  %1051 = select <4 x i1> %1042, <4 x float> %1046, <4 x float> %1050, !dbg !12
  %1052 = extractelement <4 x float> %979, i64 0, !dbg !12
  %1053 = extractelement <4 x float> %979, i64 1, !dbg !12
  %1054 = extractelement <4 x float> %979, i64 2, !dbg !12
  %1055 = extractelement <4 x float> %979, i64 3, !dbg !12
  %1056 = select i1 %.not, float %1053, float %1052, !dbg !12
  %1057 = bitcast float %1056 to i32, !dbg !12
  %1058 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1057), !dbg !12
  %1059 = bitcast i32 %1058 to float, !dbg !12
  %1060 = select i1 %.not, float %1052, float %1059, !dbg !12
  %1061 = select i1 %.not, float %1059, float %1053, !dbg !12
  %1062 = select i1 %.not, float %1055, float %1054, !dbg !12
  %1063 = bitcast float %1062 to i32, !dbg !12
  %1064 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1063), !dbg !12
  %1065 = bitcast i32 %1064 to float, !dbg !12
  %1066 = select i1 %.not, float %1054, float %1065, !dbg !12
  %1067 = select i1 %.not, float %1065, float %1055, !dbg !12
  %1068 = select i1 %.not18, float %1066, float %1060, !dbg !12
  %1069 = bitcast float %1068 to i32, !dbg !12
  %1070 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1069), !dbg !12
  %1071 = bitcast i32 %1070 to float, !dbg !12
  %1072 = select i1 %.not18, float %1067, float %1061, !dbg !12
  %1073 = bitcast float %1072 to i32, !dbg !12
  %1074 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1073), !dbg !12
  %1075 = bitcast i32 %1074 to float, !dbg !12
  %1076 = insertelement <4 x float> poison, float %1060, i64 0, !dbg !12
  %1077 = insertelement <4 x float> %1076, float %1061, i64 1, !dbg !12
  %1078 = insertelement <4 x float> %1077, float %1071, i64 2, !dbg !12
  %1079 = insertelement <4 x float> %1078, float %1075, i64 3, !dbg !12
  %1080 = insertelement <4 x float> poison, float %1071, i64 0, !dbg !12
  %1081 = insertelement <4 x float> %1080, float %1075, i64 1, !dbg !12
  %1082 = insertelement <4 x float> %1081, float %1066, i64 2, !dbg !12
  %1083 = insertelement <4 x float> %1082, float %1067, i64 3, !dbg !12
  %1084 = select <4 x i1> %1042, <4 x float> %1079, <4 x float> %1083, !dbg !12
  %1085 = extractelement <4 x float> %980, i64 0, !dbg !12
  %1086 = extractelement <4 x float> %980, i64 1, !dbg !12
  %1087 = extractelement <4 x float> %980, i64 2, !dbg !12
  %1088 = extractelement <4 x float> %980, i64 3, !dbg !12
  %1089 = select i1 %.not, float %1086, float %1085, !dbg !12
  %1090 = bitcast float %1089 to i32, !dbg !12
  %1091 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1090), !dbg !12
  %1092 = bitcast i32 %1091 to float, !dbg !12
  %1093 = select i1 %.not, float %1085, float %1092, !dbg !12
  %1094 = select i1 %.not, float %1092, float %1086, !dbg !12
  %1095 = select i1 %.not, float %1088, float %1087, !dbg !12
  %1096 = bitcast float %1095 to i32, !dbg !12
  %1097 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1096), !dbg !12
  %1098 = bitcast i32 %1097 to float, !dbg !12
  %1099 = select i1 %.not, float %1087, float %1098, !dbg !12
  %1100 = select i1 %.not, float %1098, float %1088, !dbg !12
  %1101 = select i1 %.not18, float %1099, float %1093, !dbg !12
  %1102 = bitcast float %1101 to i32, !dbg !12
  %1103 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1102), !dbg !12
  %1104 = bitcast i32 %1103 to float, !dbg !12
  %1105 = select i1 %.not18, float %1100, float %1094, !dbg !12
  %1106 = bitcast float %1105 to i32, !dbg !12
  %1107 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1106), !dbg !12
  %1108 = bitcast i32 %1107 to float, !dbg !12
  %1109 = insertelement <4 x float> poison, float %1093, i64 0, !dbg !12
  %1110 = insertelement <4 x float> %1109, float %1094, i64 1, !dbg !12
  %1111 = insertelement <4 x float> %1110, float %1104, i64 2, !dbg !12
  %1112 = insertelement <4 x float> %1111, float %1108, i64 3, !dbg !12
  %1113 = insertelement <4 x float> poison, float %1104, i64 0, !dbg !12
  %1114 = insertelement <4 x float> %1113, float %1108, i64 1, !dbg !12
  %1115 = insertelement <4 x float> %1114, float %1099, i64 2, !dbg !12
  %1116 = insertelement <4 x float> %1115, float %1100, i64 3, !dbg !12
  %1117 = select <4 x i1> %1042, <4 x float> %1112, <4 x float> %1116, !dbg !12
  %1118 = extractelement <4 x float> %981, i64 0, !dbg !12
  %1119 = extractelement <4 x float> %981, i64 1, !dbg !12
  %1120 = extractelement <4 x float> %981, i64 2, !dbg !12
  %1121 = extractelement <4 x float> %981, i64 3, !dbg !12
  %1122 = select i1 %.not, float %1119, float %1118, !dbg !12
  %1123 = bitcast float %1122 to i32, !dbg !12
  %1124 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1123), !dbg !12
  %1125 = bitcast i32 %1124 to float, !dbg !12
  %1126 = select i1 %.not, float %1118, float %1125, !dbg !12
  %1127 = select i1 %.not, float %1125, float %1119, !dbg !12
  %1128 = select i1 %.not, float %1121, float %1120, !dbg !12
  %1129 = bitcast float %1128 to i32, !dbg !12
  %1130 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1129), !dbg !12
  %1131 = bitcast i32 %1130 to float, !dbg !12
  %1132 = select i1 %.not, float %1120, float %1131, !dbg !12
  %1133 = select i1 %.not, float %1131, float %1121, !dbg !12
  %1134 = select i1 %.not18, float %1132, float %1126, !dbg !12
  %1135 = bitcast float %1134 to i32, !dbg !12
  %1136 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1135), !dbg !12
  %1137 = bitcast i32 %1136 to float, !dbg !12
  %1138 = select i1 %.not18, float %1133, float %1127, !dbg !12
  %1139 = bitcast float %1138 to i32, !dbg !12
  %1140 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1139), !dbg !12
  %1141 = bitcast i32 %1140 to float, !dbg !12
  %1142 = insertelement <4 x float> poison, float %1126, i64 0, !dbg !12
  %1143 = insertelement <4 x float> %1142, float %1127, i64 1, !dbg !12
  %1144 = insertelement <4 x float> %1143, float %1137, i64 2, !dbg !12
  %1145 = insertelement <4 x float> %1144, float %1141, i64 3, !dbg !12
  %1146 = insertelement <4 x float> poison, float %1137, i64 0, !dbg !12
  %1147 = insertelement <4 x float> %1146, float %1141, i64 1, !dbg !12
  %1148 = insertelement <4 x float> %1147, float %1132, i64 2, !dbg !12
  %1149 = insertelement <4 x float> %1148, float %1133, i64 3, !dbg !12
  %1150 = select <4 x i1> %1042, <4 x float> %1145, <4 x float> %1149, !dbg !12
  %1151 = extractelement <4 x float> %902, i64 0, !dbg !12
  %1152 = extractelement <4 x float> %902, i64 1, !dbg !12
  %1153 = extractelement <4 x float> %902, i64 2, !dbg !12
  %1154 = extractelement <4 x float> %902, i64 3, !dbg !12
  %1155 = select i1 %.not, float %1152, float %1151, !dbg !12
  %1156 = bitcast float %1155 to i32, !dbg !12
  %1157 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1156), !dbg !12
  %1158 = bitcast i32 %1157 to float, !dbg !12
  %1159 = select i1 %.not, float %1151, float %1158, !dbg !12
  %1160 = select i1 %.not, float %1158, float %1152, !dbg !12
  %1161 = select i1 %.not, float %1154, float %1153, !dbg !12
  %1162 = bitcast float %1161 to i32, !dbg !12
  %1163 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1162), !dbg !12
  %1164 = bitcast i32 %1163 to float, !dbg !12
  %1165 = select i1 %.not, float %1153, float %1164, !dbg !12
  %1166 = select i1 %.not, float %1164, float %1154, !dbg !12
  %1167 = select i1 %.not18, float %1165, float %1159, !dbg !12
  %1168 = bitcast float %1167 to i32, !dbg !12
  %1169 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1168), !dbg !12
  %1170 = bitcast i32 %1169 to float, !dbg !12
  %1171 = select i1 %.not18, float %1166, float %1160, !dbg !12
  %1172 = bitcast float %1171 to i32, !dbg !12
  %1173 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1172), !dbg !12
  %1174 = bitcast i32 %1173 to float, !dbg !12
  %1175 = insertelement <4 x float> poison, float %1159, i64 0, !dbg !12
  %1176 = insertelement <4 x float> %1175, float %1160, i64 1, !dbg !12
  %1177 = insertelement <4 x float> %1176, float %1170, i64 2, !dbg !12
  %1178 = insertelement <4 x float> %1177, float %1174, i64 3, !dbg !12
  %1179 = insertelement <4 x float> poison, float %1170, i64 0, !dbg !12
  %1180 = insertelement <4 x float> %1179, float %1174, i64 1, !dbg !12
  %1181 = insertelement <4 x float> %1180, float %1165, i64 2, !dbg !12
  %1182 = insertelement <4 x float> %1181, float %1166, i64 3, !dbg !12
  %1183 = select <4 x i1> %1042, <4 x float> %1178, <4 x float> %1182, !dbg !12
  %1184 = extractelement <4 x float> %904, i64 0, !dbg !12
  %1185 = extractelement <4 x float> %904, i64 1, !dbg !12
  %1186 = extractelement <4 x float> %904, i64 2, !dbg !12
  %1187 = extractelement <4 x float> %904, i64 3, !dbg !12
  %1188 = select i1 %.not, float %1185, float %1184, !dbg !12
  %1189 = bitcast float %1188 to i32, !dbg !12
  %1190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1189), !dbg !12
  %1191 = bitcast i32 %1190 to float, !dbg !12
  %1192 = select i1 %.not, float %1184, float %1191, !dbg !12
  %1193 = select i1 %.not, float %1191, float %1185, !dbg !12
  %1194 = select i1 %.not, float %1187, float %1186, !dbg !12
  %1195 = bitcast float %1194 to i32, !dbg !12
  %1196 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1195), !dbg !12
  %1197 = bitcast i32 %1196 to float, !dbg !12
  %1198 = select i1 %.not, float %1186, float %1197, !dbg !12
  %1199 = select i1 %.not, float %1197, float %1187, !dbg !12
  %1200 = select i1 %.not18, float %1198, float %1192, !dbg !12
  %1201 = bitcast float %1200 to i32, !dbg !12
  %1202 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1201), !dbg !12
  %1203 = bitcast i32 %1202 to float, !dbg !12
  %1204 = select i1 %.not18, float %1199, float %1193, !dbg !12
  %1205 = bitcast float %1204 to i32, !dbg !12
  %1206 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1205), !dbg !12
  %1207 = bitcast i32 %1206 to float, !dbg !12
  %1208 = insertelement <4 x float> poison, float %1192, i64 0, !dbg !12
  %1209 = insertelement <4 x float> %1208, float %1193, i64 1, !dbg !12
  %1210 = insertelement <4 x float> %1209, float %1203, i64 2, !dbg !12
  %1211 = insertelement <4 x float> %1210, float %1207, i64 3, !dbg !12
  %1212 = insertelement <4 x float> poison, float %1203, i64 0, !dbg !12
  %1213 = insertelement <4 x float> %1212, float %1207, i64 1, !dbg !12
  %1214 = insertelement <4 x float> %1213, float %1198, i64 2, !dbg !12
  %1215 = insertelement <4 x float> %1214, float %1199, i64 3, !dbg !12
  %1216 = select <4 x i1> %1042, <4 x float> %1211, <4 x float> %1215, !dbg !12
  %1217 = extractelement <4 x float> %905, i64 0, !dbg !12
  %1218 = extractelement <4 x float> %905, i64 1, !dbg !12
  %1219 = extractelement <4 x float> %905, i64 2, !dbg !12
  %1220 = extractelement <4 x float> %905, i64 3, !dbg !12
  %1221 = select i1 %.not, float %1218, float %1217, !dbg !12
  %1222 = bitcast float %1221 to i32, !dbg !12
  %1223 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1222), !dbg !12
  %1224 = bitcast i32 %1223 to float, !dbg !12
  %1225 = select i1 %.not, float %1217, float %1224, !dbg !12
  %1226 = select i1 %.not, float %1224, float %1218, !dbg !12
  %1227 = select i1 %.not, float %1220, float %1219, !dbg !12
  %1228 = bitcast float %1227 to i32, !dbg !12
  %1229 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1228), !dbg !12
  %1230 = bitcast i32 %1229 to float, !dbg !12
  %1231 = select i1 %.not, float %1219, float %1230, !dbg !12
  %1232 = select i1 %.not, float %1230, float %1220, !dbg !12
  %1233 = select i1 %.not18, float %1231, float %1225, !dbg !12
  %1234 = bitcast float %1233 to i32, !dbg !12
  %1235 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1234), !dbg !12
  %1236 = bitcast i32 %1235 to float, !dbg !12
  %1237 = select i1 %.not18, float %1232, float %1226, !dbg !12
  %1238 = bitcast float %1237 to i32, !dbg !12
  %1239 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1238), !dbg !12
  %1240 = bitcast i32 %1239 to float, !dbg !12
  %1241 = insertelement <4 x float> poison, float %1225, i64 0, !dbg !12
  %1242 = insertelement <4 x float> %1241, float %1226, i64 1, !dbg !12
  %1243 = insertelement <4 x float> %1242, float %1236, i64 2, !dbg !12
  %1244 = insertelement <4 x float> %1243, float %1240, i64 3, !dbg !12
  %1245 = insertelement <4 x float> poison, float %1236, i64 0, !dbg !12
  %1246 = insertelement <4 x float> %1245, float %1240, i64 1, !dbg !12
  %1247 = insertelement <4 x float> %1246, float %1231, i64 2, !dbg !12
  %1248 = insertelement <4 x float> %1247, float %1232, i64 3, !dbg !12
  %1249 = select <4 x i1> %1042, <4 x float> %1244, <4 x float> %1248, !dbg !12
  %1250 = extractelement <4 x float> %906, i64 0, !dbg !12
  %1251 = extractelement <4 x float> %906, i64 1, !dbg !12
  %1252 = extractelement <4 x float> %906, i64 2, !dbg !12
  %1253 = extractelement <4 x float> %906, i64 3, !dbg !12
  %1254 = select i1 %.not, float %1251, float %1250, !dbg !12
  %1255 = bitcast float %1254 to i32, !dbg !12
  %1256 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1255), !dbg !12
  %1257 = bitcast i32 %1256 to float, !dbg !12
  %1258 = select i1 %.not, float %1250, float %1257, !dbg !12
  %1259 = select i1 %.not, float %1257, float %1251, !dbg !12
  %1260 = select i1 %.not, float %1253, float %1252, !dbg !12
  %1261 = bitcast float %1260 to i32, !dbg !12
  %1262 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1261), !dbg !12
  %1263 = bitcast i32 %1262 to float, !dbg !12
  %1264 = select i1 %.not, float %1252, float %1263, !dbg !12
  %1265 = select i1 %.not, float %1263, float %1253, !dbg !12
  %1266 = select i1 %.not18, float %1264, float %1258, !dbg !12
  %1267 = bitcast float %1266 to i32, !dbg !12
  %1268 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1267), !dbg !12
  %1269 = bitcast i32 %1268 to float, !dbg !12
  %1270 = select i1 %.not18, float %1265, float %1259, !dbg !12
  %1271 = bitcast float %1270 to i32, !dbg !12
  %1272 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1271), !dbg !12
  %1273 = bitcast i32 %1272 to float, !dbg !12
  %1274 = insertelement <4 x float> poison, float %1258, i64 0, !dbg !12
  %1275 = insertelement <4 x float> %1274, float %1259, i64 1, !dbg !12
  %1276 = insertelement <4 x float> %1275, float %1269, i64 2, !dbg !12
  %1277 = insertelement <4 x float> %1276, float %1273, i64 3, !dbg !12
  %1278 = insertelement <4 x float> poison, float %1269, i64 0, !dbg !12
  %1279 = insertelement <4 x float> %1278, float %1273, i64 1, !dbg !12
  %1280 = insertelement <4 x float> %1279, float %1264, i64 2, !dbg !12
  %1281 = insertelement <4 x float> %1280, float %1265, i64 3, !dbg !12
  %1282 = select <4 x i1> %1042, <4 x float> %1277, <4 x float> %1281, !dbg !12
  %1283 = extractelement <4 x float> %826, i64 0, !dbg !12
  %1284 = extractelement <4 x float> %826, i64 1, !dbg !12
  %1285 = extractelement <4 x float> %826, i64 2, !dbg !12
  %1286 = extractelement <4 x float> %826, i64 3, !dbg !12
  %1287 = select i1 %.not, float %1284, float %1283, !dbg !12
  %1288 = bitcast float %1287 to i32, !dbg !12
  %1289 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1288), !dbg !12
  %1290 = bitcast i32 %1289 to float, !dbg !12
  %1291 = select i1 %.not, float %1283, float %1290, !dbg !12
  %1292 = select i1 %.not, float %1290, float %1284, !dbg !12
  %1293 = select i1 %.not, float %1286, float %1285, !dbg !12
  %1294 = bitcast float %1293 to i32, !dbg !12
  %1295 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1294), !dbg !12
  %1296 = bitcast i32 %1295 to float, !dbg !12
  %1297 = select i1 %.not, float %1285, float %1296, !dbg !12
  %1298 = select i1 %.not, float %1296, float %1286, !dbg !12
  %1299 = select i1 %.not18, float %1297, float %1291, !dbg !12
  %1300 = bitcast float %1299 to i32, !dbg !12
  %1301 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1300), !dbg !12
  %1302 = bitcast i32 %1301 to float, !dbg !12
  %1303 = select i1 %.not18, float %1298, float %1292, !dbg !12
  %1304 = bitcast float %1303 to i32, !dbg !12
  %1305 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1304), !dbg !12
  %1306 = bitcast i32 %1305 to float, !dbg !12
  %1307 = insertelement <4 x float> poison, float %1291, i64 0, !dbg !12
  %1308 = insertelement <4 x float> %1307, float %1292, i64 1, !dbg !12
  %1309 = insertelement <4 x float> %1308, float %1302, i64 2, !dbg !12
  %1310 = insertelement <4 x float> %1309, float %1306, i64 3, !dbg !12
  %1311 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !12
  %1312 = insertelement <4 x float> %1311, float %1306, i64 1, !dbg !12
  %1313 = insertelement <4 x float> %1312, float %1297, i64 2, !dbg !12
  %1314 = insertelement <4 x float> %1313, float %1298, i64 3, !dbg !12
  %1315 = select <4 x i1> %1042, <4 x float> %1310, <4 x float> %1314, !dbg !12
  %1316 = extractelement <4 x float> %828, i64 0, !dbg !12
  %1317 = extractelement <4 x float> %828, i64 1, !dbg !12
  %1318 = extractelement <4 x float> %828, i64 2, !dbg !12
  %1319 = extractelement <4 x float> %828, i64 3, !dbg !12
  %1320 = select i1 %.not, float %1317, float %1316, !dbg !12
  %1321 = bitcast float %1320 to i32, !dbg !12
  %1322 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1321), !dbg !12
  %1323 = bitcast i32 %1322 to float, !dbg !12
  %1324 = select i1 %.not, float %1316, float %1323, !dbg !12
  %1325 = select i1 %.not, float %1323, float %1317, !dbg !12
  %1326 = select i1 %.not, float %1319, float %1318, !dbg !12
  %1327 = bitcast float %1326 to i32, !dbg !12
  %1328 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1327), !dbg !12
  %1329 = bitcast i32 %1328 to float, !dbg !12
  %1330 = select i1 %.not, float %1318, float %1329, !dbg !12
  %1331 = select i1 %.not, float %1329, float %1319, !dbg !12
  %1332 = select i1 %.not18, float %1330, float %1324, !dbg !12
  %1333 = bitcast float %1332 to i32, !dbg !12
  %1334 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1333), !dbg !12
  %1335 = bitcast i32 %1334 to float, !dbg !12
  %1336 = select i1 %.not18, float %1331, float %1325, !dbg !12
  %1337 = bitcast float %1336 to i32, !dbg !12
  %1338 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1337), !dbg !12
  %1339 = bitcast i32 %1338 to float, !dbg !12
  %1340 = insertelement <4 x float> poison, float %1324, i64 0, !dbg !12
  %1341 = insertelement <4 x float> %1340, float %1325, i64 1, !dbg !12
  %1342 = insertelement <4 x float> %1341, float %1335, i64 2, !dbg !12
  %1343 = insertelement <4 x float> %1342, float %1339, i64 3, !dbg !12
  %1344 = insertelement <4 x float> poison, float %1335, i64 0, !dbg !12
  %1345 = insertelement <4 x float> %1344, float %1339, i64 1, !dbg !12
  %1346 = insertelement <4 x float> %1345, float %1330, i64 2, !dbg !12
  %1347 = insertelement <4 x float> %1346, float %1331, i64 3, !dbg !12
  %1348 = select <4 x i1> %1042, <4 x float> %1343, <4 x float> %1347, !dbg !12
  %1349 = extractelement <4 x float> %829, i64 0, !dbg !12
  %1350 = extractelement <4 x float> %829, i64 1, !dbg !12
  %1351 = extractelement <4 x float> %829, i64 2, !dbg !12
  %1352 = extractelement <4 x float> %829, i64 3, !dbg !12
  %1353 = select i1 %.not, float %1350, float %1349, !dbg !12
  %1354 = bitcast float %1353 to i32, !dbg !12
  %1355 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1354), !dbg !12
  %1356 = bitcast i32 %1355 to float, !dbg !12
  %1357 = select i1 %.not, float %1349, float %1356, !dbg !12
  %1358 = select i1 %.not, float %1356, float %1350, !dbg !12
  %1359 = select i1 %.not, float %1352, float %1351, !dbg !12
  %1360 = bitcast float %1359 to i32, !dbg !12
  %1361 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1360), !dbg !12
  %1362 = bitcast i32 %1361 to float, !dbg !12
  %1363 = select i1 %.not, float %1351, float %1362, !dbg !12
  %1364 = select i1 %.not, float %1362, float %1352, !dbg !12
  %1365 = select i1 %.not18, float %1363, float %1357, !dbg !12
  %1366 = bitcast float %1365 to i32, !dbg !12
  %1367 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1366), !dbg !12
  %1368 = bitcast i32 %1367 to float, !dbg !12
  %1369 = select i1 %.not18, float %1364, float %1358, !dbg !12
  %1370 = bitcast float %1369 to i32, !dbg !12
  %1371 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1370), !dbg !12
  %1372 = bitcast i32 %1371 to float, !dbg !12
  %1373 = insertelement <4 x float> poison, float %1357, i64 0, !dbg !12
  %1374 = insertelement <4 x float> %1373, float %1358, i64 1, !dbg !12
  %1375 = insertelement <4 x float> %1374, float %1368, i64 2, !dbg !12
  %1376 = insertelement <4 x float> %1375, float %1372, i64 3, !dbg !12
  %1377 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !12
  %1378 = insertelement <4 x float> %1377, float %1372, i64 1, !dbg !12
  %1379 = insertelement <4 x float> %1378, float %1363, i64 2, !dbg !12
  %1380 = insertelement <4 x float> %1379, float %1364, i64 3, !dbg !12
  %1381 = select <4 x i1> %1042, <4 x float> %1376, <4 x float> %1380, !dbg !12
  %1382 = extractelement <4 x float> %830, i64 0, !dbg !12
  %1383 = extractelement <4 x float> %830, i64 1, !dbg !12
  %1384 = extractelement <4 x float> %830, i64 2, !dbg !12
  %1385 = extractelement <4 x float> %830, i64 3, !dbg !12
  %1386 = select i1 %.not, float %1383, float %1382, !dbg !12
  %1387 = bitcast float %1386 to i32, !dbg !12
  %1388 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1387), !dbg !12
  %1389 = bitcast i32 %1388 to float, !dbg !12
  %1390 = select i1 %.not, float %1382, float %1389, !dbg !12
  %1391 = select i1 %.not, float %1389, float %1383, !dbg !12
  %1392 = select i1 %.not, float %1385, float %1384, !dbg !12
  %1393 = bitcast float %1392 to i32, !dbg !12
  %1394 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1393), !dbg !12
  %1395 = bitcast i32 %1394 to float, !dbg !12
  %1396 = select i1 %.not, float %1384, float %1395, !dbg !12
  %1397 = select i1 %.not, float %1395, float %1385, !dbg !12
  %1398 = select i1 %.not18, float %1396, float %1390, !dbg !12
  %1399 = bitcast float %1398 to i32, !dbg !12
  %1400 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1399), !dbg !12
  %1401 = bitcast i32 %1400 to float, !dbg !12
  %1402 = select i1 %.not18, float %1397, float %1391, !dbg !12
  %1403 = bitcast float %1402 to i32, !dbg !12
  %1404 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1403), !dbg !12
  %1405 = bitcast i32 %1404 to float, !dbg !12
  %1406 = insertelement <4 x float> poison, float %1390, i64 0, !dbg !12
  %1407 = insertelement <4 x float> %1406, float %1391, i64 1, !dbg !12
  %1408 = insertelement <4 x float> %1407, float %1401, i64 2, !dbg !12
  %1409 = insertelement <4 x float> %1408, float %1405, i64 3, !dbg !12
  %1410 = insertelement <4 x float> poison, float %1401, i64 0, !dbg !12
  %1411 = insertelement <4 x float> %1410, float %1405, i64 1, !dbg !12
  %1412 = insertelement <4 x float> %1411, float %1396, i64 2, !dbg !12
  %1413 = insertelement <4 x float> %1412, float %1397, i64 3, !dbg !12
  %1414 = select <4 x i1> %1042, <4 x float> %1409, <4 x float> %1413, !dbg !12
  %1415 = extractelement <4 x float> %732, i64 0, !dbg !12
  %1416 = extractelement <4 x float> %732, i64 1, !dbg !12
  %1417 = extractelement <4 x float> %732, i64 2, !dbg !12
  %1418 = extractelement <4 x float> %732, i64 3, !dbg !12
  %1419 = select i1 %.not, float %1416, float %1415, !dbg !12
  %1420 = bitcast float %1419 to i32, !dbg !12
  %1421 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1420), !dbg !12
  %1422 = bitcast i32 %1421 to float, !dbg !12
  %1423 = select i1 %.not, float %1415, float %1422, !dbg !12
  %1424 = select i1 %.not, float %1422, float %1416, !dbg !12
  %1425 = select i1 %.not, float %1418, float %1417, !dbg !12
  %1426 = bitcast float %1425 to i32, !dbg !12
  %1427 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1426), !dbg !12
  %1428 = bitcast i32 %1427 to float, !dbg !12
  %1429 = select i1 %.not, float %1417, float %1428, !dbg !12
  %1430 = select i1 %.not, float %1428, float %1418, !dbg !12
  %1431 = select i1 %.not18, float %1429, float %1423, !dbg !12
  %1432 = bitcast float %1431 to i32, !dbg !12
  %1433 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1432), !dbg !12
  %1434 = select i1 %.not18, float %1430, float %1424, !dbg !12
  %1435 = bitcast float %1434 to i32, !dbg !12
  %1436 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1435), !dbg !12
  %1437 = extractelement <4 x float> %734, i64 0, !dbg !12
  %1438 = extractelement <4 x float> %734, i64 1, !dbg !12
  %1439 = extractelement <4 x float> %734, i64 2, !dbg !12
  %1440 = extractelement <4 x float> %734, i64 3, !dbg !12
  %1441 = select i1 %.not, float %1438, float %1437, !dbg !12
  %1442 = bitcast float %1441 to i32, !dbg !12
  %1443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1442), !dbg !12
  %1444 = bitcast i32 %1443 to float, !dbg !12
  %1445 = select i1 %.not, float %1437, float %1444, !dbg !12
  %1446 = select i1 %.not, float %1444, float %1438, !dbg !12
  %1447 = select i1 %.not, float %1440, float %1439, !dbg !12
  %1448 = bitcast float %1447 to i32, !dbg !12
  %1449 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1448), !dbg !12
  %1450 = bitcast i32 %1449 to float, !dbg !12
  %1451 = select i1 %.not, float %1439, float %1450, !dbg !12
  %1452 = select i1 %.not, float %1450, float %1440, !dbg !12
  %1453 = select i1 %.not18, float %1451, float %1445, !dbg !12
  %1454 = bitcast float %1453 to i32, !dbg !12
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1454), !dbg !12
  %1456 = select i1 %.not18, float %1452, float %1446, !dbg !12
  %1457 = bitcast float %1456 to i32, !dbg !12
  %1458 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1457), !dbg !12
  %1459 = extractelement <4 x float> %753, i64 0, !dbg !12
  %1460 = extractelement <4 x float> %753, i64 1, !dbg !12
  %1461 = extractelement <4 x float> %753, i64 2, !dbg !12
  %1462 = extractelement <4 x float> %753, i64 3, !dbg !12
  %1463 = select i1 %.not, float %1460, float %1459, !dbg !12
  %1464 = bitcast float %1463 to i32, !dbg !12
  %1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1464), !dbg !12
  %1466 = bitcast i32 %1465 to float, !dbg !12
  %1467 = select i1 %.not, float %1459, float %1466, !dbg !12
  %1468 = select i1 %.not, float %1466, float %1460, !dbg !12
  %1469 = select i1 %.not, float %1462, float %1461, !dbg !12
  %1470 = bitcast float %1469 to i32, !dbg !12
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1470), !dbg !12
  %1472 = bitcast i32 %1471 to float, !dbg !12
  %1473 = select i1 %.not, float %1461, float %1472, !dbg !12
  %1474 = select i1 %.not, float %1472, float %1462, !dbg !12
  %1475 = select i1 %.not18, float %1473, float %1467, !dbg !12
  %1476 = bitcast float %1475 to i32, !dbg !12
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1476), !dbg !12
  %1478 = select i1 %.not18, float %1474, float %1468, !dbg !12
  %1479 = bitcast float %1478 to i32, !dbg !12
  %1480 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1479), !dbg !12
  %1481 = extractelement <4 x float> %754, i64 0, !dbg !12
  %1482 = extractelement <4 x float> %754, i64 1, !dbg !12
  %1483 = extractelement <4 x float> %754, i64 2, !dbg !12
  %1484 = extractelement <4 x float> %754, i64 3, !dbg !12
  %1485 = select i1 %.not, float %1482, float %1481, !dbg !12
  %1486 = bitcast float %1485 to i32, !dbg !12
  %1487 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1486), !dbg !12
  %1488 = bitcast i32 %1487 to float, !dbg !12
  %1489 = select i1 %.not, float %1481, float %1488, !dbg !12
  %1490 = select i1 %.not, float %1488, float %1482, !dbg !12
  %1491 = select i1 %.not, float %1484, float %1483, !dbg !12
  %1492 = bitcast float %1491 to i32, !dbg !12
  %1493 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1492), !dbg !12
  %1494 = bitcast i32 %1493 to float, !dbg !12
  %1495 = select i1 %.not, float %1483, float %1494, !dbg !12
  %1496 = select i1 %.not, float %1494, float %1484, !dbg !12
  %1497 = select i1 %.not18, float %1495, float %1489, !dbg !12
  %1498 = bitcast float %1497 to i32, !dbg !12
  %1499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1498), !dbg !12
  %1500 = select i1 %.not18, float %1496, float %1490, !dbg !12
  %1501 = bitcast float %1500 to i32, !dbg !12
  %1502 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1501), !dbg !12
  %1503 = shufflevector <1 x float> %1014, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1504 = shufflevector <1 x float> %1009, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1505 = bitcast i32 %1499 to float, !dbg !12
  %1506 = bitcast i32 %1502 to float, !dbg !12
  %1507 = insertelement <4 x float> poison, float %1489, i64 0, !dbg !12
  %1508 = insertelement <4 x float> %1507, float %1490, i64 1, !dbg !12
  %1509 = insertelement <4 x float> %1508, float %1505, i64 2, !dbg !12
  %1510 = insertelement <4 x float> %1509, float %1506, i64 3, !dbg !12
  %1511 = insertelement <4 x float> poison, float %1505, i64 0, !dbg !12
  %1512 = insertelement <4 x float> %1511, float %1506, i64 1, !dbg !12
  %1513 = insertelement <4 x float> %1512, float %1495, i64 2, !dbg !12
  %1514 = insertelement <4 x float> %1513, float %1496, i64 3, !dbg !12
  %1515 = select <4 x i1> %1042, <4 x float> %1510, <4 x float> %1514, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1504, !dbg !51
  %normalized = fmul <4 x float> %row.reciprocal, %1515, !dbg !51
  %1516 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !52
  %1517 = extractelement <4 x half> %1516, i64 3, !dbg !52
  %1518 = extractelement <4 x half> %1516, i64 2, !dbg !52
  %1519 = extractelement <4 x half> %1516, i64 1, !dbg !52
  %1520 = extractelement <4 x half> %1516, i64 0, !dbg !52
  %1521 = bitcast i32 %1477 to float, !dbg !12
  %1522 = bitcast i32 %1480 to float, !dbg !12
  %1523 = insertelement <4 x float> poison, float %1467, i64 0, !dbg !12
  %1524 = insertelement <4 x float> %1523, float %1468, i64 1, !dbg !12
  %1525 = insertelement <4 x float> %1524, float %1521, i64 2, !dbg !12
  %1526 = insertelement <4 x float> %1525, float %1522, i64 3, !dbg !12
  %1527 = insertelement <4 x float> poison, float %1521, i64 0, !dbg !12
  %1528 = insertelement <4 x float> %1527, float %1522, i64 1, !dbg !12
  %1529 = insertelement <4 x float> %1528, float %1473, i64 2, !dbg !12
  %1530 = insertelement <4 x float> %1529, float %1474, i64 3, !dbg !12
  %1531 = select <4 x i1> %1042, <4 x float> %1526, <4 x float> %1530, !dbg !12
  %normalized19 = fmul <4 x float> %row.reciprocal, %1531, !dbg !51
  %1532 = fptrunc <4 x float> %normalized19 to <4 x half>, !dbg !52
  %1533 = extractelement <4 x half> %1532, i64 3, !dbg !52
  %1534 = extractelement <4 x half> %1532, i64 2, !dbg !52
  %1535 = extractelement <4 x half> %1532, i64 1, !dbg !52
  %1536 = extractelement <4 x half> %1532, i64 0, !dbg !52
  %1537 = bitcast i32 %1455 to float, !dbg !12
  %1538 = bitcast i32 %1458 to float, !dbg !12
  %1539 = insertelement <4 x float> poison, float %1445, i64 0, !dbg !12
  %1540 = insertelement <4 x float> %1539, float %1446, i64 1, !dbg !12
  %1541 = insertelement <4 x float> %1540, float %1537, i64 2, !dbg !12
  %1542 = insertelement <4 x float> %1541, float %1538, i64 3, !dbg !12
  %1543 = insertelement <4 x float> poison, float %1537, i64 0, !dbg !12
  %1544 = insertelement <4 x float> %1543, float %1538, i64 1, !dbg !12
  %1545 = insertelement <4 x float> %1544, float %1451, i64 2, !dbg !12
  %1546 = insertelement <4 x float> %1545, float %1452, i64 3, !dbg !12
  %1547 = select <4 x i1> %1042, <4 x float> %1542, <4 x float> %1546, !dbg !12
  %row.reciprocal20 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1503, !dbg !51
  %normalized21 = fmul <4 x float> %row.reciprocal20, %1547, !dbg !51
  %1548 = fptrunc <4 x float> %normalized21 to <4 x half>, !dbg !52
  %1549 = extractelement <4 x half> %1548, i64 3, !dbg !52
  %1550 = extractelement <4 x half> %1548, i64 2, !dbg !52
  %1551 = extractelement <4 x half> %1548, i64 1, !dbg !52
  %1552 = extractelement <4 x half> %1548, i64 0, !dbg !52
  %1553 = bitcast i32 %1433 to float, !dbg !12
  %1554 = bitcast i32 %1436 to float, !dbg !12
  %1555 = insertelement <4 x float> poison, float %1423, i64 0, !dbg !12
  %1556 = insertelement <4 x float> %1555, float %1424, i64 1, !dbg !12
  %1557 = insertelement <4 x float> %1556, float %1553, i64 2, !dbg !12
  %1558 = insertelement <4 x float> %1557, float %1554, i64 3, !dbg !12
  %1559 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !12
  %1560 = insertelement <4 x float> %1559, float %1554, i64 1, !dbg !12
  %1561 = insertelement <4 x float> %1560, float %1429, i64 2, !dbg !12
  %1562 = insertelement <4 x float> %1561, float %1430, i64 3, !dbg !12
  %1563 = select <4 x i1> %1042, <4 x float> %1558, <4 x float> %1562, !dbg !12
  %normalized22 = fmul <4 x float> %row.reciprocal20, %1563, !dbg !51
  %1564 = fptrunc <4 x float> %normalized22 to <4 x half>, !dbg !52
  %1565 = extractelement <4 x half> %1564, i64 3, !dbg !52
  %1566 = extractelement <4 x half> %1564, i64 2, !dbg !52
  %1567 = extractelement <4 x half> %1564, i64 1, !dbg !52
  %1568 = extractelement <4 x half> %1564, i64 0, !dbg !52
  %address.thread727 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !53, !range !10, !frisk.register_lifetimes !5
  %address.local728 = zext nneg i32 %address.thread727 to i64, !dbg !9
  %1569 = lshr i64 %address.local728, 2, !dbg !53
  %address.local731 = and i64 %1569, 48, !dbg !53
  %address.local733 = and i64 %address.local728, 240, !dbg !53
  %address.local732 = add i64 %34, %address.local728, !dbg !53
  %address.local734 = sub i64 %address.local732, %address.local733, !dbg !53
  %address.local735 = add i64 %address.local734, %address.local731, !dbg !53
  %1570 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %42, !dbg !53
  %address.local738.idx = shl nuw nsw i64 %address.local735, 8, !dbg !53
  %address.local738 = getelementptr inbounds nuw i8, ptr addrspace(1) %1570, i64 %address.local738.idx, !dbg !53
  %address.local740 = and i64 %1569, 12, !dbg !11
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local740, !dbg !53
  store half %1568, ptr addrspace(1) %address.local741, align 2, !dbg !53
  %address.local743 = or disjoint i64 %address.local740, 1, !dbg !53
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local743, !dbg !53
  store half %1567, ptr addrspace(1) %address.local744, align 2, !dbg !53
  %address.local746 = or disjoint i64 %address.local740, 2, !dbg !53
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local746, !dbg !53
  store half %1566, ptr addrspace(1) %address.local747, align 2, !dbg !53
  %address.local749 = or disjoint i64 %address.local740, 3, !dbg !53
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local749, !dbg !53
  store half %1565, ptr addrspace(1) %address.local750, align 2, !dbg !53
  %address.local752 = or disjoint i64 %address.local740, 16, !dbg !53
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local752, !dbg !53
  store half %1552, ptr addrspace(1) %address.local753, align 2, !dbg !53
  %address.local755 = or disjoint i64 %address.local740, 17, !dbg !53
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local755, !dbg !53
  store half %1551, ptr addrspace(1) %address.local756, align 2, !dbg !53
  %address.local758 = or disjoint i64 %address.local740, 18, !dbg !53
  %address.local759 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local758, !dbg !53
  store half %1550, ptr addrspace(1) %address.local759, align 2, !dbg !53
  %address.local761 = or disjoint i64 %address.local740, 19, !dbg !53
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local761, !dbg !53
  store half %1549, ptr addrspace(1) %address.local762, align 2, !dbg !53
  %address.local766 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local738, i64 16384, !dbg !53
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local740, !dbg !53
  store half %1536, ptr addrspace(1) %address.local767, align 2, !dbg !53
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local743, !dbg !53
  store half %1535, ptr addrspace(1) %address.local769, align 2, !dbg !53
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local746, !dbg !53
  store half %1534, ptr addrspace(1) %address.local771, align 2, !dbg !53
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local749, !dbg !53
  store half %1533, ptr addrspace(1) %address.local773, align 2, !dbg !53
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local752, !dbg !53
  store half %1520, ptr addrspace(1) %address.local775, align 2, !dbg !53
  %address.local777 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local755, !dbg !53
  store half %1519, ptr addrspace(1) %address.local777, align 2, !dbg !53
  %address.local779 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local758, !dbg !53
  store half %1518, ptr addrspace(1) %address.local779, align 2, !dbg !53
  %address.local781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local761, !dbg !53
  store half %1517, ptr addrspace(1) %address.local781, align 2, !dbg !53
  %normalized24 = fmul <4 x float> %row.reciprocal, %1414, !dbg !51
  %1571 = fptrunc <4 x float> %normalized24 to <4 x half>, !dbg !52
  %1572 = extractelement <4 x half> %1571, i64 3, !dbg !52
  %1573 = extractelement <4 x half> %1571, i64 2, !dbg !52
  %1574 = extractelement <4 x half> %1571, i64 1, !dbg !52
  %1575 = extractelement <4 x half> %1571, i64 0, !dbg !52
  %normalized25 = fmul <4 x float> %row.reciprocal, %1381, !dbg !51
  %1576 = fptrunc <4 x float> %normalized25 to <4 x half>, !dbg !52
  %1577 = extractelement <4 x half> %1576, i64 3, !dbg !52
  %1578 = extractelement <4 x half> %1576, i64 2, !dbg !52
  %1579 = extractelement <4 x half> %1576, i64 1, !dbg !52
  %1580 = extractelement <4 x half> %1576, i64 0, !dbg !52
  %normalized27 = fmul <4 x float> %row.reciprocal20, %1348, !dbg !51
  %1581 = fptrunc <4 x float> %normalized27 to <4 x half>, !dbg !52
  %1582 = extractelement <4 x half> %1581, i64 3, !dbg !52
  %1583 = extractelement <4 x half> %1581, i64 2, !dbg !52
  %1584 = extractelement <4 x half> %1581, i64 1, !dbg !52
  %1585 = extractelement <4 x half> %1581, i64 0, !dbg !52
  %normalized28 = fmul <4 x float> %row.reciprocal20, %1315, !dbg !51
  %1586 = fptrunc <4 x float> %normalized28 to <4 x half>, !dbg !52
  %1587 = extractelement <4 x half> %1586, i64 3, !dbg !52
  %1588 = extractelement <4 x half> %1586, i64 2, !dbg !52
  %1589 = extractelement <4 x half> %1586, i64 1, !dbg !52
  %1590 = extractelement <4 x half> %1586, i64 0, !dbg !52
  %address.local796 = or disjoint i64 %address.local740, 32, !dbg !53
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local796, !dbg !53
  store half %1590, ptr addrspace(1) %address.local797, align 2, !dbg !53
  %address.local799 = or disjoint i64 %address.local740, 33, !dbg !53
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local799, !dbg !53
  store half %1589, ptr addrspace(1) %address.local800, align 2, !dbg !53
  %address.local802 = or disjoint i64 %address.local740, 34, !dbg !53
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local802, !dbg !53
  store half %1588, ptr addrspace(1) %address.local803, align 2, !dbg !53
  %address.local805 = or disjoint i64 %address.local740, 35, !dbg !53
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local805, !dbg !53
  store half %1587, ptr addrspace(1) %address.local806, align 2, !dbg !53
  %address.local808 = or disjoint i64 %address.local740, 48, !dbg !53
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local808, !dbg !53
  store half %1585, ptr addrspace(1) %address.local809, align 2, !dbg !53
  %address.local811 = or disjoint i64 %address.local740, 49, !dbg !53
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local811, !dbg !53
  store half %1584, ptr addrspace(1) %address.local812, align 2, !dbg !53
  %address.local814 = or disjoint i64 %address.local740, 50, !dbg !53
  %address.local815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local814, !dbg !53
  store half %1583, ptr addrspace(1) %address.local815, align 2, !dbg !53
  %address.local817 = or i64 %1569, 51, !dbg !53
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local817, !dbg !53
  store half %1582, ptr addrspace(1) %address.local818, align 2, !dbg !53
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local796, !dbg !53
  store half %1580, ptr addrspace(1) %address.local823, align 2, !dbg !53
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local799, !dbg !53
  store half %1579, ptr addrspace(1) %address.local825, align 2, !dbg !53
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local802, !dbg !53
  store half %1578, ptr addrspace(1) %address.local827, align 2, !dbg !53
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local805, !dbg !53
  store half %1577, ptr addrspace(1) %address.local829, align 2, !dbg !53
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local808, !dbg !53
  store half %1575, ptr addrspace(1) %address.local831, align 2, !dbg !53
  %address.local833 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local811, !dbg !53
  store half %1574, ptr addrspace(1) %address.local833, align 2, !dbg !53
  %address.local835 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local814, !dbg !53
  store half %1573, ptr addrspace(1) %address.local835, align 2, !dbg !53
  %address.local837 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local817, !dbg !53
  store half %1572, ptr addrspace(1) %address.local837, align 2, !dbg !53
  %normalized30 = fmul <4 x float> %row.reciprocal, %1282, !dbg !51
  %1591 = fptrunc <4 x float> %normalized30 to <4 x half>, !dbg !52
  %1592 = extractelement <4 x half> %1591, i64 3, !dbg !52
  %1593 = extractelement <4 x half> %1591, i64 2, !dbg !52
  %1594 = extractelement <4 x half> %1591, i64 1, !dbg !52
  %1595 = extractelement <4 x half> %1591, i64 0, !dbg !52
  %normalized31 = fmul <4 x float> %row.reciprocal, %1249, !dbg !51
  %1596 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !52
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !52
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !52
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !52
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !52
  %normalized33 = fmul <4 x float> %row.reciprocal20, %1216, !dbg !51
  %1601 = fptrunc <4 x float> %normalized33 to <4 x half>, !dbg !52
  %1602 = extractelement <4 x half> %1601, i64 3, !dbg !52
  %1603 = extractelement <4 x half> %1601, i64 2, !dbg !52
  %1604 = extractelement <4 x half> %1601, i64 1, !dbg !52
  %1605 = extractelement <4 x half> %1601, i64 0, !dbg !52
  %normalized34 = fmul <4 x float> %row.reciprocal20, %1183, !dbg !51
  %1606 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !52
  %1607 = extractelement <4 x half> %1606, i64 3, !dbg !52
  %1608 = extractelement <4 x half> %1606, i64 2, !dbg !52
  %1609 = extractelement <4 x half> %1606, i64 1, !dbg !52
  %1610 = extractelement <4 x half> %1606, i64 0, !dbg !52
  %address.local852 = or disjoint i64 %address.local740, 64, !dbg !53
  %address.local853 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local852, !dbg !53
  store half %1610, ptr addrspace(1) %address.local853, align 2, !dbg !53
  %address.local855 = or disjoint i64 %address.local740, 65, !dbg !53
  %address.local856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local855, !dbg !53
  store half %1609, ptr addrspace(1) %address.local856, align 2, !dbg !53
  %address.local858 = or disjoint i64 %address.local740, 66, !dbg !53
  %address.local859 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local858, !dbg !53
  store half %1608, ptr addrspace(1) %address.local859, align 2, !dbg !53
  %address.local861 = or disjoint i64 %address.local740, 67, !dbg !53
  %address.local862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local861, !dbg !53
  store half %1607, ptr addrspace(1) %address.local862, align 2, !dbg !53
  %address.local864 = or disjoint i64 %address.local740, 80, !dbg !53
  %address.local865 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local864, !dbg !53
  store half %1605, ptr addrspace(1) %address.local865, align 2, !dbg !53
  %address.local867 = or disjoint i64 %address.local740, 81, !dbg !53
  %address.local868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local867, !dbg !53
  store half %1604, ptr addrspace(1) %address.local868, align 2, !dbg !53
  %address.local870 = or disjoint i64 %address.local740, 82, !dbg !53
  %address.local871 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local870, !dbg !53
  store half %1603, ptr addrspace(1) %address.local871, align 2, !dbg !53
  %address.local873 = or disjoint i64 %address.local740, 83, !dbg !53
  %address.local874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local873, !dbg !53
  store half %1602, ptr addrspace(1) %address.local874, align 2, !dbg !53
  %address.local879 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local852, !dbg !53
  store half %1600, ptr addrspace(1) %address.local879, align 2, !dbg !53
  %address.local881 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local855, !dbg !53
  store half %1599, ptr addrspace(1) %address.local881, align 2, !dbg !53
  %address.local883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local858, !dbg !53
  store half %1598, ptr addrspace(1) %address.local883, align 2, !dbg !53
  %address.local885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local861, !dbg !53
  store half %1597, ptr addrspace(1) %address.local885, align 2, !dbg !53
  %address.local887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local864, !dbg !53
  store half %1595, ptr addrspace(1) %address.local887, align 2, !dbg !53
  %address.local889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local867, !dbg !53
  store half %1594, ptr addrspace(1) %address.local889, align 2, !dbg !53
  %address.local891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local870, !dbg !53
  store half %1593, ptr addrspace(1) %address.local891, align 2, !dbg !53
  %address.local893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local873, !dbg !53
  store half %1592, ptr addrspace(1) %address.local893, align 2, !dbg !53
  %normalized36 = fmul <4 x float> %row.reciprocal, %1150, !dbg !51
  %1611 = fptrunc <4 x float> %normalized36 to <4 x half>, !dbg !52
  %1612 = extractelement <4 x half> %1611, i64 3, !dbg !52
  %1613 = extractelement <4 x half> %1611, i64 2, !dbg !52
  %1614 = extractelement <4 x half> %1611, i64 1, !dbg !52
  %1615 = extractelement <4 x half> %1611, i64 0, !dbg !52
  %normalized37 = fmul <4 x float> %row.reciprocal, %1117, !dbg !51
  %1616 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !52
  %1617 = extractelement <4 x half> %1616, i64 3, !dbg !52
  %1618 = extractelement <4 x half> %1616, i64 2, !dbg !52
  %1619 = extractelement <4 x half> %1616, i64 1, !dbg !52
  %1620 = extractelement <4 x half> %1616, i64 0, !dbg !52
  %normalized39 = fmul <4 x float> %row.reciprocal20, %1084, !dbg !51
  %1621 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !52
  %1622 = extractelement <4 x half> %1621, i64 3, !dbg !52
  %1623 = extractelement <4 x half> %1621, i64 2, !dbg !52
  %1624 = extractelement <4 x half> %1621, i64 1, !dbg !52
  %1625 = extractelement <4 x half> %1621, i64 0, !dbg !52
  %normalized40 = fmul <4 x float> %row.reciprocal20, %1051, !dbg !51
  %1626 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !52
  %1627 = extractelement <4 x half> %1626, i64 3, !dbg !52
  %1628 = extractelement <4 x half> %1626, i64 2, !dbg !52
  %1629 = extractelement <4 x half> %1626, i64 1, !dbg !52
  %1630 = extractelement <4 x half> %1626, i64 0, !dbg !52
  %address.local908 = or disjoint i64 %address.local740, 96, !dbg !53
  %address.local909 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local908, !dbg !53
  store half %1630, ptr addrspace(1) %address.local909, align 2, !dbg !53
  %address.local911 = or disjoint i64 %address.local740, 97, !dbg !53
  %address.local912 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local911, !dbg !53
  store half %1629, ptr addrspace(1) %address.local912, align 2, !dbg !53
  %address.local914 = or disjoint i64 %address.local740, 98, !dbg !53
  %address.local915 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local914, !dbg !53
  store half %1628, ptr addrspace(1) %address.local915, align 2, !dbg !53
  %address.local917 = or disjoint i64 %address.local740, 99, !dbg !53
  %address.local918 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local917, !dbg !53
  store half %1627, ptr addrspace(1) %address.local918, align 2, !dbg !53
  %address.local920 = or disjoint i64 %address.local740, 112, !dbg !53
  %address.local921 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local920, !dbg !53
  store half %1625, ptr addrspace(1) %address.local921, align 2, !dbg !53
  %address.local923 = or disjoint i64 %address.local740, 113, !dbg !53
  %address.local924 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local923, !dbg !53
  store half %1624, ptr addrspace(1) %address.local924, align 2, !dbg !53
  %address.local926 = or disjoint i64 %address.local740, 114, !dbg !53
  %address.local927 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local926, !dbg !53
  store half %1623, ptr addrspace(1) %address.local927, align 2, !dbg !53
  %address.local929 = or i64 %1569, 115, !dbg !53
  %address.local930 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local738, i64 %address.local929, !dbg !53
  store half %1622, ptr addrspace(1) %address.local930, align 2, !dbg !53
  %address.local935 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local908, !dbg !53
  store half %1620, ptr addrspace(1) %address.local935, align 2, !dbg !53
  %address.local937 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local911, !dbg !53
  store half %1619, ptr addrspace(1) %address.local937, align 2, !dbg !53
  %address.local939 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local914, !dbg !53
  store half %1618, ptr addrspace(1) %address.local939, align 2, !dbg !53
  %address.local941 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local917, !dbg !53
  store half %1617, ptr addrspace(1) %address.local941, align 2, !dbg !53
  %address.local943 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local920, !dbg !53
  store half %1615, ptr addrspace(1) %address.local943, align 2, !dbg !53
  %address.local945 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local923, !dbg !53
  store half %1614, ptr addrspace(1) %address.local945, align 2, !dbg !53
  %address.local947 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local926, !dbg !53
  store half %1613, ptr addrspace(1) %address.local947, align 2, !dbg !53
  %address.local949 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local766, i64 %address.local929, !dbg !53
  store half %1612, ptr addrspace(1) %address.local949, align 2, !dbg !53
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
!13 = !DILocation(line: 68, column: 22, scope: !3)
!14 = !DILocation(line: 70, column: 26, scope: !3)
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
!38 = !{i32 23}
!39 = !DILocation(line: 86, column: 20, scope: !3)
!40 = !DILocation(line: 71, column: 23, scope: !3)
!41 = !DILocation(line: 75, column: 22, scope: !3)
!42 = !DILocation(line: 76, column: 23, scope: !3)
!43 = !DILocation(line: 77, column: 20, scope: !3)
!44 = !DILocation(line: 88, column: 22, scope: !3)
!45 = !DILocation(line: 78, column: 22, scope: !3)
!46 = !DILocation(line: 87, column: 23, scope: !3)
!47 = !{i64 8192}
!48 = !DILocation(line: 79, column: 22, scope: !3)
!49 = !DILocation(line: 81, column: 16, scope: !3)
!50 = !DILocation(line: 80, column: 20, scope: !3)
!51 = !DILocation(line: 92, column: 23, scope: !3)
!52 = !DILocation(line: 93, column: 19, scope: !3)
!53 = !DILocation(line: 94, column: 9, scope: !3)
