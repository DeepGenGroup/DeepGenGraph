; ModuleID = 'generated_ir/inner_inst_b_permute_swizzle_noncausal/kernel.ll'
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
  %address.local316 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local317 = and i64 %address.local316, 1536, !dbg !9
  %address.local318 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local317, !dbg !9
  %address.local321 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local322 = and i64 %address.local321, 16256, !dbg !9
  %address.local323 = lshr i64 %address.local, 4, !dbg !9
  %address.local324 = and i64 %address.local323, 8, !dbg !9
  %address.local325 = or disjoint i64 %address.local322, %address.local324, !dbg !9
  %36 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !9
  %.idx1006 = shl i64 %34, 8, !dbg !9
  %37 = add i64 %.idx1006, %.idx, !dbg !9
  %address.local326.idx = shl nuw nsw i64 %address.local325, 1, !dbg !9
  %38 = or disjoint i64 %address.local326.idx, %37, !dbg !9
  %dma.buffer.offset1004 = trunc i64 %38 to i32, !dbg !9
  %39 = insertelement <2 x i64> poison, i64 %36, i64 0, !dbg !9
  %40 = bitcast <2 x i64> %39 to <4 x i32>, !dbg !9
  %dma.buffer.rsrc1005 = shufflevector <4 x i32> %40, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !9
  %address.local327 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local328 = shl nuw nsw i64 %address.local327, 3, !dbg !9
  %address.local329 = and i64 %address.local328, 3584, !dbg !9
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local329, !dbg !9
  %address.local335 = lshr i64 %address.local327, 4, !dbg !9
  %address.local336 = and i64 %address.local335, 24, !dbg !9
  %address.local337 = or disjoint i64 %address.local336, %address.local322, !dbg !9
  %address.local338.idx = shl nuw nsw i64 %address.local337, 1, !dbg !9
  %41 = or disjoint i64 %address.local338.idx, %37, !dbg !9
  %dma.buffer.offset1002 = trunc i64 %41 to i32, !dbg !9
  %42 = shl nuw nsw i64 %31, 19, !dbg !9
  %43 = add i64 %35, %42, !dbg !9
  %address.local339 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local317, !dbg !9
  %44 = or disjoint i64 %43, 32, !dbg !9
  %45 = or disjoint i64 %address.local325, %44, !dbg !9
  %.tr = trunc i64 %45 to i32, !dbg !9
  %dma.buffer.offset1000 = shl i32 %.tr, 1, !dbg !9
  %address.local343 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local329, !dbg !9
  %46 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local318, i64 0, !dbg !9
  %47 = insertelement <4 x ptr addrspace(3)> %46, ptr addrspace(3) %address.local330, i64 1, !dbg !9
  %48 = insertelement <4 x ptr addrspace(3)> %47, ptr addrspace(3) %address.local339, i64 2, !dbg !9
  %49 = insertelement <4 x ptr addrspace(3)> %48, ptr addrspace(3) %address.local343, i64 3, !dbg !9
  %50 = ptrtoint <4 x ptr addrspace(3)> %49 to <4 x i64>, !dbg !9
  %51 = trunc <4 x i64> %50 to <4 x i32>, !dbg !9
  %52 = extractelement <4 x i32> %51, i64 0, !dbg !9
  %53 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %52), !dbg !9
  %54 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset1004, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %55 = extractelement <4 x i32> %51, i64 1, !dbg !9
  %56 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %55), !dbg !9
  %57 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset1002, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %58 = extractelement <4 x i32> %51, i64 2, !dbg !9
  %59 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %58), !dbg !9
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset1000, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %61 = extractelement <4 x i32> %51, i64 3, !dbg !9
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61), !dbg !9
  %63 = or disjoint i64 %address.local337, %44, !dbg !9
  %.tr1010 = trunc i64 %63 to i32, !dbg !9
  %dma.buffer.offset998 = shl i32 %.tr1010, 1, !dbg !9
  %64 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset998, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %address.local347 = lshr i32 %address.thread, 2, !dbg !11
  %address.local348 = and i32 %address.local347, 4, !dbg !11
  %address.local349 = shl nuw nsw i32 %address.thread, 1, !dbg !11
  %address.local350 = and i32 %address.local349, 384, !dbg !11
  %address.local351 = or disjoint i32 %address.local348, %address.local350, !dbg !11
  %address.local352 = shl nuw nsw i32 %address.thread, 3, !dbg !11
  %address.local353 = and i32 %address.local352, 120, !dbg !11
  %address.local354 = or disjoint i32 %address.local351, %address.local353, !dbg !11
  %address.local355 = shl nuw nsw i32 %address.thread, 5, !dbg !11
  %address.local356 = and i32 %address.local355, 1024, !dbg !11
  %address.local357 = or disjoint i32 %address.local354, %address.local356, !dbg !11
  %address.local358 = zext nneg i32 %address.local357 to i64, !dbg !11
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local358, !dbg !11
  %65 = load <4 x half>, ptr addrspace(3) %address.local359, align 8, !dbg !11
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local359, i64 4096, !dbg !11
  %66 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %67 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep827 = getelementptr i8, ptr addrspace(3) %address.local359, i64 1024, !dbg !11
  %68 = load <4 x half>, ptr addrspace(3) %lds.const.gep827, align 8, !dbg !11
  %69 = fmul <4 x half> %66, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep828 = getelementptr i8, ptr addrspace(3) %address.local359, i64 5120, !dbg !11
  %70 = load <4 x half>, ptr addrspace(3) %lds.const.gep828, align 8, !dbg !11
  %71 = fmul <4 x half> %68, splat (half 1.275630e-01), !dbg !11
  %72 = fmul <4 x half> %70, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %73 = or disjoint i64 %43, 64, !dbg !9
  %74 = or disjoint i64 %address.local325, %73, !dbg !9
  %.tr1012 = trunc i64 %74 to i32, !dbg !9
  %dma.buffer.offset996 = shl i32 %.tr1012, 1, !dbg !9
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset996, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %76 = or disjoint i64 %address.local337, %73, !dbg !9
  %.tr1014 = trunc i64 %76 to i32, !dbg !9
  %dma.buffer.offset994 = shl i32 %.tr1014, 1, !dbg !9
  %77 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset994, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %lds.const.gep829 = getelementptr i8, ptr addrspace(3) %address.local359, i64 8192, !dbg !11
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep829, align 8, !dbg !11
  %lds.const.gep830 = getelementptr i8, ptr addrspace(3) %address.local359, i64 12288, !dbg !11
  %79 = load <4 x half>, ptr addrspace(3) %lds.const.gep830, align 8, !dbg !11
  %lds.const.gep831 = getelementptr i8, ptr addrspace(3) %address.local359, i64 9216, !dbg !11
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep831, align 8, !dbg !11
  %81 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !11
  %lds.const.gep832 = getelementptr i8, ptr addrspace(3) %address.local359, i64 13312, !dbg !11
  %82 = load <4 x half>, ptr addrspace(3) %lds.const.gep832, align 8, !dbg !11
  %83 = fmul <4 x half> %79, splat (half 1.275630e-01), !dbg !11
  %84 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !11
  %85 = fmul <4 x half> %82, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %86 = or disjoint i64 %43, 96, !dbg !9
  %87 = or disjoint i64 %address.local325, %86, !dbg !9
  %.tr1016 = trunc i64 %87 to i32, !dbg !9
  %dma.buffer.offset992 = shl i32 %.tr1016, 1, !dbg !9
  %88 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %dma.buffer.offset992, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %89 = or disjoint i64 %address.local337, %86, !dbg !9
  %.tr1018 = trunc i64 %89 to i32, !dbg !9
  %dma.buffer.offset990 = shl i32 %.tr1018, 1, !dbg !9
  %90 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %dma.buffer.offset990, <4 x i32> %dma.buffer.rsrc1005) #9, !dbg !9
  %91 = load <4 x half>, ptr addrspace(3) %address.local359, align 8, !dbg !11
  %92 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !11
  %93 = load <4 x half>, ptr addrspace(3) %lds.const.gep827, align 8, !dbg !11
  %94 = fmul <4 x half> %91, splat (half 1.275630e-01), !dbg !11
  %95 = load <4 x half>, ptr addrspace(3) %lds.const.gep828, align 8, !dbg !11
  %96 = fmul <4 x half> %92, splat (half 1.275630e-01), !dbg !11
  %97 = fmul <4 x half> %93, splat (half 1.275630e-01), !dbg !11
  %98 = fmul <4 x half> %95, splat (half 1.275630e-01), !dbg !11
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !9
  fence syncscope("workgroup") release, !dbg !9
  tail call void @llvm.amdgcn.s.barrier(), !dbg !9
  fence syncscope("workgroup") acquire, !dbg !9
  %99 = load <4 x half>, ptr addrspace(3) %lds.const.gep829, align 8, !dbg !11
  %100 = load <4 x half>, ptr addrspace(3) %lds.const.gep830, align 8, !dbg !11
  %101 = load <4 x half>, ptr addrspace(3) %lds.const.gep831, align 8, !dbg !11
  %102 = fmul <4 x half> %99, splat (half 1.275630e-01), !dbg !11
  %103 = load <4 x half>, ptr addrspace(3) %lds.const.gep832, align 8, !dbg !11
  %104 = fmul <4 x half> %100, splat (half 1.275630e-01), !dbg !11
  %105 = fmul <4 x half> %101, splat (half 1.275630e-01), !dbg !11
  %106 = fmul <4 x half> %103, splat (half 1.275630e-01), !dbg !11
  fence syncscope("workgroup") release, !dbg !12
  tail call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %k0.prime. = shl i64 %25, 19, !dbg !13
  %k0.prime.1567 = and i64 %k0.prime., 16252928, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1110 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1111 = and i64 %k0.prime.kdma.source1110, 8, !dbg !13
  %k0.prime.kdma.source1112 = and i64 %k0.prime.kdma.source, 95, !dbg !13
  %k0.prime.kdma.source1113 = shl nuw nsw i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1114 = and i64 %k0.prime.kdma.source1113, 32, !dbg !13
  %k0.prime.kdma.source1117 = or disjoint i64 %k0.prime.kdma.source1114, %k0.prime.kdma.source1112, !dbg !13
  %k0.prime.kdma.source1118 = shl nuw nsw i64 %k0.prime.kdma.source1117, 7, !dbg !13
  %k0.prime.kdma.source1119 = or disjoint i64 %k0.prime.kdma.source1118, %k0.prime.kdma.source1111, !dbg !13
  %k0.prime.kdma.source1120 = or disjoint i64 %k0.prime.kdma.source1119, %k0.prime.1567, !dbg !13
  %k0.prime.kdma.source1121 = trunc nuw nsw i64 %k0.prime.kdma.source1120 to i32, !dbg !13
  %k0.prime.kdma.source1122 = shl nuw nsw i32 %k0.prime.kdma.source1121, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1122, %kdma.intra, !dbg !13
  %k0.prime.1570 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1571 = insertelement <2 x i64> poison, i64 %k0.prime.1570, i64 0, !dbg !13
  %k0.prime.1572 = bitcast <2 x i64> %k0.prime.1571 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc989 = shufflevector <4 x i32> %k0.prime.1572, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1573 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1138 = add i32 %53, 512, !dbg !13
  %k0.prime.kdma.source1123 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1124 = lshr i64 %k0.prime.kdma.source1123, 4, !dbg !13
  %k0.prime.kdma.source1125 = and i64 %k0.prime.kdma.source1124, 8, !dbg !13
  %k0.prime.kdma.source1126 = and i64 %k0.prime.kdma.source1123, 95, !dbg !13
  %k0.prime.kdma.source1127 = shl nuw nsw i64 %k0.prime.kdma.source1123, 2, !dbg !13
  %k0.prime.kdma.source1128 = and i64 %k0.prime.kdma.source1127, 32, !dbg !13
  %k0.prime.kdma.source1129 = add nuw nsw i64 %k0.prime.kdma.source1128, %k0.prime.kdma.source1123, !dbg !13
  %k0.prime.kdma.source1130 = and i64 %k0.prime.kdma.source1129, 32, !dbg !13
  %k0.prime.kdma.source1131 = or disjoint i64 %k0.prime.kdma.source1130, %k0.prime.kdma.source1126, !dbg !13
  %k0.prime.kdma.source1132 = shl nuw nsw i64 %k0.prime.kdma.source1131, 7, !dbg !13
  %k0.prime.kdma.source1133 = or disjoint i64 %k0.prime.kdma.source1132, %k0.prime.kdma.source1125, !dbg !13
  %k0.prime.kdma.source1134 = or disjoint i64 %k0.prime.kdma.source1133, %k0.prime.1567, !dbg !13
  %k0.prime.kdma.source1135 = trunc nuw nsw i64 %k0.prime.kdma.source1134 to i32, !dbg !13
  %k0.prime.kdma.source1136 = shl nuw nsw i32 %k0.prime.kdma.source1135, 1, !dbg !13
  %k0.prime.kdma.offset1137 = or disjoint i32 %k0.prime.kdma.source1136, %kdma.intra.hi, !dbg !13
  %k0.prime.1574 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1138, i32 %k0.prime.kdma.offset1137, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1150 = or disjoint i64 %k0.prime.kdma.source1119, 16, !dbg !13
  %k0.prime.kdma.source1151 = or disjoint i64 %k0.prime.kdma.source1150, %k0.prime.1567, !dbg !13
  %k0.prime.kdma.source1152 = trunc nuw nsw i64 %k0.prime.kdma.source1151 to i32, !dbg !13
  %k0.prime.kdma.source1153 = shl nuw nsw i32 %k0.prime.kdma.source1152, 1, !dbg !13
  %k0.prime.kdma.offset1154 = or disjoint i32 %k0.prime.kdma.source1153, %kdma.intra, !dbg !13
  %k0.prime.1575 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.prime.kdma.offset1154, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1172 = add i32 %56, 512, !dbg !13
  %k0.prime.kdma.source1167 = or disjoint i64 %k0.prime.kdma.source1133, 16, !dbg !13
  %k0.prime.kdma.source1168 = or disjoint i64 %k0.prime.kdma.source1167, %k0.prime.1567, !dbg !13
  %k0.prime.kdma.source1169 = trunc nuw nsw i64 %k0.prime.kdma.source1168 to i32, !dbg !13
  %k0.prime.kdma.source1170 = shl nuw nsw i32 %k0.prime.kdma.source1169, 1, !dbg !13
  %k0.prime.kdma.offset1171 = or disjoint i32 %k0.prime.kdma.source1170, %kdma.intra.hi, !dbg !13
  %k0.prime.1576 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1172, i32 %k0.prime.kdma.offset1171, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %kdma.lds1204 = add i32 %59, 512
  %kdma.lds1238 = add i32 %62, 512
  %107 = ptrtoint ptr addrspace(1) %2 to i64
  %108 = insertelement <2 x i64> poison, i64 %107, i64 0
  %109 = bitcast <2 x i64> %108 to <4 x i32>
  %dma.buffer.rsrc973 = shufflevector <4 x i32> %109, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %vdb.dma.base = add i32 %53, 8192
  %vdb.dma.base1064 = add i32 %56, 8192
  br label %110, !dbg !12

110:                                              ; preds = %4, %k0.join
  %111 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1007, %k0.join ]
  %112 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1003, %k0.join ]
  %113 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %999, %k0.join ]
  %114 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %995, %k0.join ]
  %115 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %758, %k0.join ]
  %116 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %993, %k0.join ]
  %117 = phi i64 [ 0, %4 ], [ %915, %k0.join ]
  %118 = shl nuw nsw i64 %117, 14, !dbg !13
  %119 = add nuw nsw i64 %118, %k0.prime.1567, !dbg !13
  %address.thread410 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !10, !frisk.register_lifetimes !5
  %address.local411 = zext nneg i32 %address.thread410 to i64, !dbg !9
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %120 = or disjoint i64 %119, 32, !dbg !13
  %kdma.source1184 = or disjoint i64 %120, %k0.prime.kdma.source1119, !dbg !13
  %kdma.source1185 = trunc nuw nsw i64 %kdma.source1184 to i32, !dbg !13
  %kdma.source1186 = shl nuw nsw i32 %kdma.source1185, 1, !dbg !13
  %kdma.offset1187 = or disjoint i32 %kdma.source1186, %kdma.intra, !dbg !13
  %121 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1187, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1200 = or disjoint i64 %120, %k0.prime.kdma.source1133, !dbg !13
  %kdma.source1201 = trunc nuw nsw i64 %kdma.source1200 to i32, !dbg !13
  %kdma.source1202 = shl nuw nsw i32 %kdma.source1201, 1, !dbg !13
  %kdma.offset1203 = or disjoint i32 %kdma.source1202, %kdma.intra.hi, !dbg !13
  %122 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1204, i32 %kdma.offset1203, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1217 = or disjoint i64 %120, %k0.prime.kdma.source1150, !dbg !13
  %kdma.source1218 = trunc nuw nsw i64 %kdma.source1217 to i32, !dbg !13
  %kdma.source1219 = shl nuw nsw i32 %kdma.source1218, 1, !dbg !13
  %kdma.offset1220 = or disjoint i32 %kdma.source1219, %kdma.intra, !dbg !13
  %123 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1220, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1234 = or disjoint i64 %120, %k0.prime.kdma.source1167, !dbg !13
  %kdma.source1235 = trunc nuw nsw i64 %kdma.source1234 to i32, !dbg !13
  %kdma.source1236 = shl nuw nsw i32 %kdma.source1235, 1, !dbg !13
  %kdma.offset1237 = or disjoint i32 %kdma.source1236, %kdma.intra.hi, !dbg !13
  %124 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1238, i32 %kdma.offset1237, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %address.local437 = lshr i64 %address.local411, 2, !dbg !11
  %address.local438 = and i64 %address.local437, 12, !dbg !11
  %address.local439 = shl nuw nsw i32 %address.thread410, 1, !dbg !14
  %address.local440 = and i32 %address.local439, 24, !dbg !14
  %address.local441 = shl nuw nsw i32 %address.thread410, 5, !dbg !14
  %address.local442 = and i32 %address.local441, 96, !dbg !14
  %address.local443 = or disjoint i32 %address.local440, %address.local442, !dbg !14
  %address.local444 = shl nuw nsw i32 %address.thread410, 7, !dbg !14
  %address.local445 = and i32 %address.local444, 256, !dbg !14
  %address.local446 = or disjoint i32 %address.local443, %address.local445, !dbg !14
  %address.local447 = zext nneg i32 %address.local446 to i64, !dbg !14
  %address.local449 = and i64 %address.local411, 32, !dbg !11
  %address.local450 = icmp eq i64 %address.local449, 0, !dbg !11
  %address.local451 = select i1 %address.local450, i64 0, i64 1016, !dbg !11
  %125 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local438, !dbg !14
  %126 = getelementptr [2 x i8], ptr addrspace(3) %125, i64 %address.local447, !dbg !14
  %address.local453 = getelementptr [2 x i8], ptr addrspace(3) %126, i64 %address.local451, !dbg !14
  %.idx14971499 = add nuw nsw i64 %address.local451, %address.local438, !dbg !14
  %127 = add nuw nsw i64 %.idx14971499, %address.local447, !dbg !14
  %128 = shl nuw nsw i64 %127, 1, !dbg !14
  %129 = lshr i64 %127, 5, !dbg !14
  %130 = and i64 %129, 8, !dbg !14
  %131 = xor i64 %130, %128, !dbg !14
  %132 = sub nsw i64 %131, %128, !dbg !14
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local453, i64 %132, !dbg !14
  %133 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !15
  %kdma.gep1371 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 256, !dbg !14
  %134 = load <4 x half>, ptr addrspace(3) %kdma.gep1371, align 8, !dbg !14, !frisk.k_lds_pair !15
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !14
  %135 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !16
  %kdma.gep1372 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4352, !dbg !14
  %136 = load <4 x half>, ptr addrspace(3) %kdma.gep1372, align 8, !dbg !14, !frisk.k_lds_pair !16
  %address.local478 = shl nuw nsw i32 %address.thread410, 8, !dbg !14
  %address.local479 = and i32 %address.local478, 512, !dbg !14
  %address.local480 = zext nneg i32 %address.local479 to i64, !dbg !14
  %address.local481 = sub nsw i64 %address.local447, %address.local480, !dbg !14
  %137 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local481, !dbg !14
  %138 = getelementptr [2 x i8], ptr addrspace(3) %137, i64 %address.local438, !dbg !14
  %139 = getelementptr [2 x i8], ptr addrspace(3) %138, i64 %address.local451, !dbg !14
  %address.local485 = getelementptr i8, ptr addrspace(3) %139, i64 512, !dbg !14
  %140 = add nsw i64 %.idx14971499, %address.local481, !dbg !14
  %141 = shl nsw i64 %140, 1, !dbg !14
  %142 = add nsw i64 %141, 512, !dbg !14
  %143 = lshr i64 %142, 6, !dbg !14
  %144 = and i64 %143, 8, !dbg !14
  %145 = xor i64 %144, %142, !dbg !14
  %146 = sub nsw i64 %145, %142, !dbg !14
  %kdma.swizzled1373 = getelementptr i8, ptr addrspace(3) %address.local485, i64 %146, !dbg !14
  %147 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1373, align 8, !dbg !14, !frisk.k_lds_pair !17
  %kdma.gep1375 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 256, !dbg !14
  %148 = load <4 x half>, ptr addrspace(3) %kdma.gep1375, align 8, !dbg !14, !frisk.k_lds_pair !17
  %kdma.gep1374 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 4096, !dbg !14
  %149 = load <4 x half>, ptr addrspace(3) %kdma.gep1374, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep1376 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 4352, !dbg !14
  %150 = load <4 x half>, ptr addrspace(3) %kdma.gep1376, align 8, !dbg !14, !frisk.k_lds_pair !18
  %kdma.gep1377 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !14
  %151 = load <4 x half>, ptr addrspace(3) %kdma.gep1377, align 8, !dbg !14, !frisk.k_lds_pair !19
  %kdma.gep1379 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1280, !dbg !14
  %152 = load <4 x half>, ptr addrspace(3) %kdma.gep1379, align 8, !dbg !14, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %133, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1378 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !14
  %154 = load <4 x half>, ptr addrspace(3) %kdma.gep1378, align 8, !dbg !14, !frisk.k_lds_pair !20
  %kdma.gep1380 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5376, !dbg !14
  %155 = load <4 x half>, ptr addrspace(3) %kdma.gep1380, align 8, !dbg !14, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %135, <4 x float> %153) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %134, <4 x float> zeroinitializer) #8, !dbg !14
  %158 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %136, <4 x float> %157) #8, !dbg !14
  %kdma.gep1381 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 1024, !dbg !14
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1381, align 8, !dbg !14, !frisk.k_lds_pair !21
  %kdma.gep1383 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 1280, !dbg !14
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1383, align 8, !dbg !14, !frisk.k_lds_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %161 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %147, <4 x float> zeroinitializer) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1382 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 5120, !dbg !14
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1382, align 8, !dbg !14, !frisk.k_lds_pair !22
  %kdma.gep1384 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 5376, !dbg !14
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1384, align 8, !dbg !14, !frisk.k_lds_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %149, <4 x float> %161) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !14
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %150, <4 x float> %165) #8, !dbg !14
  %167 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %151, <4 x float> zeroinitializer) #8, !dbg !14
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %154, <4 x float> %167) #8, !dbg !14
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !14
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %155, <4 x float> %169) #8, !dbg !14
  %171 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !14
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %162, <4 x float> %171) #8, !dbg !14
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %67, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !14
  %174 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %163, <4 x float> %173) #8, !dbg !14
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %133, <4 x float> zeroinitializer) #8, !dbg !14
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %135, <4 x float> %175) #8, !dbg !14
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %134, <4 x float> zeroinitializer) #8, !dbg !14
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %136, <4 x float> %177) #8, !dbg !14
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %147, <4 x float> zeroinitializer) #8, !dbg !14
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %149, <4 x float> %179) #8, !dbg !14
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !14
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %150, <4 x float> %181) #8, !dbg !14
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %151, <4 x float> zeroinitializer) #8, !dbg !14
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %154, <4 x float> %183) #8, !dbg !14
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !14
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %155, <4 x float> %185) #8, !dbg !14
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !14
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %162, <4 x float> %187) #8, !dbg !14
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %71, <4 x half> %160, <4 x float> zeroinitializer) #8, !dbg !14
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %72, <4 x half> %163, <4 x float> %189) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %191 = or disjoint i64 %119, 64, !dbg !13
  %kdma.source1250 = or disjoint i64 %191, %k0.prime.kdma.source1119, !dbg !13
  %kdma.source1251 = trunc nuw nsw i64 %kdma.source1250 to i32, !dbg !13
  %kdma.source1252 = shl nuw nsw i32 %kdma.source1251, 1, !dbg !13
  %kdma.offset1253 = or disjoint i32 %kdma.source1252, %kdma.intra, !dbg !13
  %192 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %kdma.offset1253, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1266 = or disjoint i64 %191, %k0.prime.kdma.source1133, !dbg !13
  %kdma.source1267 = trunc nuw nsw i64 %kdma.source1266 to i32, !dbg !13
  %kdma.source1268 = shl nuw nsw i32 %kdma.source1267, 1, !dbg !13
  %kdma.offset1269 = or disjoint i32 %kdma.source1268, %kdma.intra.hi, !dbg !13
  %193 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1138, i32 %kdma.offset1269, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1283 = or disjoint i64 %191, %k0.prime.kdma.source1150, !dbg !13
  %kdma.source1284 = trunc nuw nsw i64 %kdma.source1283 to i32, !dbg !13
  %kdma.source1285 = shl nuw nsw i32 %kdma.source1284, 1, !dbg !13
  %kdma.offset1286 = or disjoint i32 %kdma.source1285, %kdma.intra, !dbg !13
  %194 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %kdma.offset1286, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1300 = or disjoint i64 %191, %k0.prime.kdma.source1167, !dbg !13
  %kdma.source1301 = trunc nuw nsw i64 %kdma.source1300 to i32, !dbg !13
  %kdma.source1302 = shl nuw nsw i32 %kdma.source1301, 1, !dbg !13
  %kdma.offset1303 = or disjoint i32 %kdma.source1302, %kdma.intra.hi, !dbg !13
  %195 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1172, i32 %kdma.offset1303, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.gep1401 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !14
  %196 = load <4 x half>, ptr addrspace(3) %kdma.gep1401, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep1403 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8448, !dbg !14
  %197 = load <4 x half>, ptr addrspace(3) %kdma.gep1403, align 8, !dbg !14, !frisk.k_lds_pair !23
  %kdma.gep1402 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !14
  %198 = load <4 x half>, ptr addrspace(3) %kdma.gep1402, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep1404 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12544, !dbg !14
  %199 = load <4 x half>, ptr addrspace(3) %kdma.gep1404, align 8, !dbg !14, !frisk.k_lds_pair !24
  %kdma.gep1405 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 8192, !dbg !14
  %200 = load <4 x half>, ptr addrspace(3) %kdma.gep1405, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep1407 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 8448, !dbg !14
  %201 = load <4 x half>, ptr addrspace(3) %kdma.gep1407, align 8, !dbg !14, !frisk.k_lds_pair !25
  %kdma.gep1406 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 12288, !dbg !14
  %202 = load <4 x half>, ptr addrspace(3) %kdma.gep1406, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep1408 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 12544, !dbg !14
  %203 = load <4 x half>, ptr addrspace(3) %kdma.gep1408, align 8, !dbg !14, !frisk.k_lds_pair !26
  %kdma.gep1409 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !14
  %204 = load <4 x half>, ptr addrspace(3) %kdma.gep1409, align 8, !dbg !14, !frisk.k_lds_pair !27
  %kdma.gep1411 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9472, !dbg !14
  %205 = load <4 x half>, ptr addrspace(3) %kdma.gep1411, align 8, !dbg !14, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %206 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %196, <4 x float> %156) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1410 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !14
  %207 = load <4 x half>, ptr addrspace(3) %kdma.gep1410, align 8, !dbg !14, !frisk.k_lds_pair !28
  %kdma.gep1412 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13568, !dbg !14
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1412, align 8, !dbg !14, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %209 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %198, <4 x float> %206) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %210 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %197, <4 x float> %158) #8, !dbg !14
  %211 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %199, <4 x float> %210) #8, !dbg !14
  %kdma.gep1413 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 9216, !dbg !14
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1413, align 8, !dbg !14, !frisk.k_lds_pair !29
  %kdma.gep1415 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 9472, !dbg !14
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1415, align 8, !dbg !14, !frisk.k_lds_pair !29
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %214 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %200, <4 x float> %164) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %kdma.gep1414 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 13312, !dbg !14
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1414, align 8, !dbg !14, !frisk.k_lds_pair !30
  %kdma.gep1416 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1373, i64 13568, !dbg !14
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1416, align 8, !dbg !14, !frisk.k_lds_pair !30
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %217 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %202, <4 x float> %214) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %201, <4 x float> %166) #8, !dbg !14
  %219 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %203, <4 x float> %218) #8, !dbg !14
  %220 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %204, <4 x float> %168) #8, !dbg !14
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %207, <4 x float> %220) #8, !dbg !14
  %222 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %205, <4 x float> %170) #8, !dbg !14
  %223 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %208, <4 x float> %222) #8, !dbg !14
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %212, <4 x float> %172) #8, !dbg !14
  %225 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %215, <4 x float> %224) #8, !dbg !14
  %226 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %213, <4 x float> %174) #8, !dbg !14
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %216, <4 x float> %226) #8, !dbg !14
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %196, <4 x float> %176) #8, !dbg !14
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %198, <4 x float> %228) #8, !dbg !14
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %197, <4 x float> %178) #8, !dbg !14
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %199, <4 x float> %230) #8, !dbg !14
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %200, <4 x float> %180) #8, !dbg !14
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %202, <4 x float> %232) #8, !dbg !14
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %201, <4 x float> %182) #8, !dbg !14
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %203, <4 x float> %234) #8, !dbg !14
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %204, <4 x float> %184) #8, !dbg !14
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %207, <4 x float> %236) #8, !dbg !14
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %205, <4 x float> %186) #8, !dbg !14
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %208, <4 x float> %238) #8, !dbg !14
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %212, <4 x float> %188) #8, !dbg !14
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %215, <4 x float> %240) #8, !dbg !14
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %84, <4 x half> %213, <4 x float> %190) #8, !dbg !14
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %85, <4 x half> %216, <4 x float> %242) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %244 = or disjoint i64 %119, 96, !dbg !13
  %kdma.source1316 = or disjoint i64 %244, %k0.prime.kdma.source1119, !dbg !13
  %kdma.source1317 = trunc nuw nsw i64 %kdma.source1316 to i32, !dbg !13
  %kdma.source1318 = shl nuw nsw i32 %kdma.source1317, 1, !dbg !13
  %kdma.offset1319 = or disjoint i32 %kdma.source1318, %kdma.intra, !dbg !13
  %245 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %59, i32 %kdma.offset1319, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1332 = or disjoint i64 %244, %k0.prime.kdma.source1133, !dbg !13
  %kdma.source1333 = trunc nuw nsw i64 %kdma.source1332 to i32, !dbg !13
  %kdma.source1334 = shl nuw nsw i32 %kdma.source1333, 1, !dbg !13
  %kdma.offset1335 = or disjoint i32 %kdma.source1334, %kdma.intra.hi, !dbg !13
  %246 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1204, i32 %kdma.offset1335, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1349 = or disjoint i64 %244, %k0.prime.kdma.source1150, !dbg !13
  %kdma.source1350 = trunc nuw nsw i64 %kdma.source1349 to i32, !dbg !13
  %kdma.source1351 = shl nuw nsw i32 %kdma.source1350, 1, !dbg !13
  %kdma.offset1352 = or disjoint i32 %kdma.source1351, %kdma.intra, !dbg !13
  %247 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %62, i32 %kdma.offset1352, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %kdma.source1366 = or disjoint i64 %244, %k0.prime.kdma.source1167, !dbg !13
  %kdma.source1367 = trunc nuw nsw i64 %kdma.source1366 to i32, !dbg !13
  %kdma.source1368 = shl nuw nsw i32 %kdma.source1367, 1, !dbg !13
  %kdma.offset1369 = or disjoint i32 %kdma.source1368, %kdma.intra.hi, !dbg !13
  %248 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1238, i32 %kdma.offset1369, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13
  %249 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !14, !frisk.k_lds_pair !31
  %250 = load <4 x half>, ptr addrspace(3) %kdma.gep1371, align 8, !dbg !14, !frisk.k_lds_pair !31
  %251 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !14, !frisk.k_lds_pair !32
  %252 = load <4 x half>, ptr addrspace(3) %kdma.gep1372, align 8, !dbg !14, !frisk.k_lds_pair !32
  %253 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1373, align 8, !dbg !14, !frisk.k_lds_pair !33
  %254 = load <4 x half>, ptr addrspace(3) %kdma.gep1375, align 8, !dbg !14, !frisk.k_lds_pair !33
  %255 = load <4 x half>, ptr addrspace(3) %kdma.gep1374, align 8, !dbg !14, !frisk.k_lds_pair !34
  %256 = load <4 x half>, ptr addrspace(3) %kdma.gep1376, align 8, !dbg !14, !frisk.k_lds_pair !34
  %257 = load <4 x half>, ptr addrspace(3) %kdma.gep1377, align 8, !dbg !14, !frisk.k_lds_pair !35
  %258 = load <4 x half>, ptr addrspace(3) %kdma.gep1379, align 8, !dbg !14, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %249, <4 x float> %209) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %260 = load <4 x half>, ptr addrspace(3) %kdma.gep1378, align 8, !dbg !14, !frisk.k_lds_pair !36
  %261 = load <4 x half>, ptr addrspace(3) %kdma.gep1380, align 8, !dbg !14, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %251, <4 x float> %259) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %250, <4 x float> %211) #8, !dbg !14
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %252, <4 x float> %263) #8, !dbg !14
  %265 = load <4 x half>, ptr addrspace(3) %kdma.gep1381, align 8, !dbg !14, !frisk.k_lds_pair !37
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1383, align 8, !dbg !14, !frisk.k_lds_pair !37
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %253, <4 x float> %217) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1382, align 8, !dbg !14, !frisk.k_lds_pair !38
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1384, align 8, !dbg !14, !frisk.k_lds_pair !38
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %255, <4 x float> %267) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_lds_prefetch !5
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %254, <4 x float> %219) #8, !dbg !14
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %256, <4 x float> %271) #8, !dbg !14
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %257, <4 x float> %221) #8, !dbg !14
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %260, <4 x float> %273) #8, !dbg !14
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %258, <4 x float> %223) #8, !dbg !14
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %261, <4 x float> %275) #8, !dbg !14
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %265, <4 x float> %225) #8, !dbg !14
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %268, <4 x float> %277) #8, !dbg !14
  %279 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %94, <4 x half> %266, <4 x float> %227) #8, !dbg !14
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %96, <4 x half> %269, <4 x float> %279) #8, !dbg !14
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %249, <4 x float> %229) #8, !dbg !14
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %251, <4 x float> %281) #8, !dbg !14
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %250, <4 x float> %231) #8, !dbg !14
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %252, <4 x float> %283) #8, !dbg !14
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %253, <4 x float> %233) #8, !dbg !14
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %255, <4 x float> %285) #8, !dbg !14
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %254, <4 x float> %235) #8, !dbg !14
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %256, <4 x float> %287) #8, !dbg !14
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %257, <4 x float> %237) #8, !dbg !14
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %260, <4 x float> %289) #8, !dbg !14
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %258, <4 x float> %239) #8, !dbg !14
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %261, <4 x float> %291) #8, !dbg !14
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %265, <4 x float> %241) #8, !dbg !14
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %268, <4 x float> %293) #8, !dbg !14
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %97, <4 x half> %266, <4 x float> %243) #8, !dbg !14
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %98, <4 x half> %269, <4 x float> %295) #8, !dbg !14
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local568 = shl nuw nsw i64 %address.local411, 5, !dbg !39
  %address.local569 = and i64 %address.local568, 8064, !dbg !39
  %address.local570 = shl nuw nsw i64 %address.local411, 3, !dbg !39
  %address.local571 = and i64 %address.local570, 24, !dbg !39
  %address.local572 = or disjoint i64 %address.local569, %address.local571, !dbg !39
  %.idx10351036 = or disjoint i64 %address.local572, %119, !dbg !39
  %.idx10351036.tr = trunc nuw nsw i64 %.idx10351036 to i32, !dbg !39
  %dma.buffer.offset972 = shl nuw nsw i32 %.idx10351036.tr, 1, !dbg !39
  %297 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %dma.buffer.offset972, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %address.local578 = or disjoint i64 %address.local572, 8192, !dbg !39
  %.idx10371038 = or disjoint i64 %address.local578, %119, !dbg !39
  %.idx10371038.tr = trunc nuw nsw i64 %.idx10371038 to i32, !dbg !39
  %dma.buffer.offset970 = shl nuw nsw i32 %.idx10371038.tr, 1, !dbg !39
  %298 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %dma.buffer.offset970, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %299 = load <4 x half>, ptr addrspace(3) %kdma.gep1401, align 8, !dbg !14, !frisk.k_tail_pair !15
  %300 = load <4 x half>, ptr addrspace(3) %kdma.gep1403, align 8, !dbg !14, !frisk.k_tail_pair !15
  %301 = load <4 x half>, ptr addrspace(3) %kdma.gep1402, align 8, !dbg !14, !frisk.k_tail_pair !16
  %302 = load <4 x half>, ptr addrspace(3) %kdma.gep1404, align 8, !dbg !14, !frisk.k_tail_pair !16
  %303 = load <4 x half>, ptr addrspace(3) %kdma.gep1405, align 8, !dbg !14, !frisk.k_tail_pair !17
  %304 = load <4 x half>, ptr addrspace(3) %kdma.gep1407, align 8, !dbg !14, !frisk.k_tail_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %299, <4 x float> %262) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %306 = load <4 x half>, ptr addrspace(3) %kdma.gep1406, align 8, !dbg !14, !frisk.k_tail_pair !18
  %307 = load <4 x half>, ptr addrspace(3) %kdma.gep1408, align 8, !dbg !14, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %301, <4 x float> %305) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %309 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %300, <4 x float> %264) #8, !dbg !14
  %310 = extractelement <4 x float> %308, i64 0, !dbg !40
  %311 = tail call float @llvm.maxnum.f32(float %310, float -inf), !dbg !7, !frisk.local_reduction !5
  %312 = extractelement <4 x float> %308, i64 1, !dbg !40
  %313 = extractelement <4 x float> %308, i64 2, !dbg !40
  %314 = extractelement <4 x float> %308, i64 3, !dbg !40
  %315 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %302, <4 x float> %309) #8, !dbg !14
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1409, align 8, !dbg !14, !frisk.k_tail_pair !19
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1411, align 8, !dbg !14, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %318 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %303, <4 x float> %270) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40, !frisk.k_tail_prefetch !5
  %319 = extractelement <4 x float> %315, i64 0, !dbg !40
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %311, float %319), !dbg !7, !frisk.local_reduction !5
  %320 = extractelement <4 x float> %315, i64 1, !dbg !40
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %312, float %320), !dbg !7, !frisk.local_reduction !5
  %321 = extractelement <4 x float> %315, i64 2, !dbg !40
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %313, float %321), !dbg !7, !frisk.local_reduction !5
  %322 = extractelement <4 x float> %315, i64 3, !dbg !40
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %314, float %322), !dbg !7, !frisk.local_reduction !5
  %323 = load <4 x half>, ptr addrspace(3) %kdma.gep1410, align 8, !dbg !14, !frisk.k_tail_pair !20
  %324 = load <4 x half>, ptr addrspace(3) %kdma.gep1412, align 8, !dbg !14, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %306, <4 x float> %318) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %326 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %304, <4 x float> %272) #8, !dbg !14
  %327 = extractelement <4 x float> %325, i64 0, !dbg !40
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %327), !dbg !7, !frisk.local_reduction !5
  %328 = extractelement <4 x float> %325, i64 1, !dbg !40
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %328), !dbg !7, !frisk.local_reduction !5
  %329 = extractelement <4 x float> %325, i64 2, !dbg !40
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce145, float %329), !dbg !7, !frisk.local_reduction !5
  %330 = extractelement <4 x float> %325, i64 3, !dbg !40
  %local.reduce150 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %330), !dbg !7, !frisk.local_reduction !5
  %331 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %307, <4 x float> %326) #8, !dbg !14
  %332 = load <4 x half>, ptr addrspace(3) %kdma.gep1413, align 8, !dbg !14, !frisk.k_tail_pair !21
  %333 = load <4 x half>, ptr addrspace(3) %kdma.gep1415, align 8, !dbg !14, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %334 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %316, <4 x float> %274) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !40, !frisk.k_tail_prefetch !5
  %335 = extractelement <4 x float> %331, i64 0, !dbg !40
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %local.reduce147, float %335), !dbg !7, !frisk.local_reduction !5
  %336 = extractelement <4 x float> %331, i64 1, !dbg !40
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %336), !dbg !7, !frisk.local_reduction !5
  %337 = extractelement <4 x float> %331, i64 2, !dbg !40
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %local.reduce149, float %337), !dbg !7, !frisk.local_reduction !5
  %338 = extractelement <4 x float> %331, i64 3, !dbg !40
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %local.reduce150, float %338), !dbg !7, !frisk.local_reduction !5
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1414, align 8, !dbg !14, !frisk.k_tail_pair !22
  %340 = load <4 x half>, ptr addrspace(3) %kdma.gep1416, align 8, !dbg !14, !frisk.k_tail_pair !22
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %323, <4 x float> %334) #8, !dbg !14
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !14, !frisk.k_tail_prefetch !5
  %342 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %317, <4 x float> %276) #8, !dbg !14
  %343 = extractelement <4 x float> %341, i64 0, !dbg !40
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %343), !dbg !7, !frisk.local_reduction !5
  %344 = extractelement <4 x float> %341, i64 1, !dbg !40
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %344), !dbg !7, !frisk.local_reduction !5
  %345 = extractelement <4 x float> %341, i64 2, !dbg !40
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %345), !dbg !7, !frisk.local_reduction !5
  %346 = extractelement <4 x float> %341, i64 3, !dbg !40
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %346), !dbg !7, !frisk.local_reduction !5
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %324, <4 x float> %342) #8, !dbg !14
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %332, <4 x float> %278) #8, !dbg !14
  %349 = extractelement <4 x float> %347, i64 0, !dbg !40
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %349), !dbg !7, !frisk.local_reduction !5
  %350 = extractelement <4 x float> %347, i64 1, !dbg !40
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %350), !dbg !7, !frisk.local_reduction !5
  %351 = extractelement <4 x float> %347, i64 2, !dbg !40
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %351), !dbg !7, !frisk.local_reduction !5
  %352 = extractelement <4 x float> %347, i64 3, !dbg !40
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %352), !dbg !7, !frisk.local_reduction !5
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %339, <4 x float> %348) #8, !dbg !14
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %102, <4 x half> %333, <4 x float> %280) #8, !dbg !14
  %355 = extractelement <4 x float> %353, i64 0, !dbg !40
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %355), !dbg !7, !frisk.local_reduction !5
  %356 = extractelement <4 x float> %353, i64 1, !dbg !40
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %356), !dbg !7, !frisk.local_reduction !5
  %357 = extractelement <4 x float> %353, i64 2, !dbg !40
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %357), !dbg !7, !frisk.local_reduction !5
  %358 = extractelement <4 x float> %353, i64 3, !dbg !40
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %358), !dbg !7, !frisk.local_reduction !5
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %104, <4 x half> %340, <4 x float> %354) #8, !dbg !14
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %299, <4 x float> %282) #8, !dbg !14
  %361 = extractelement <4 x float> %359, i64 0, !dbg !40
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %361), !dbg !7, !frisk.local_reduction !5
  %362 = extractelement <4 x float> %359, i64 1, !dbg !40
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %362), !dbg !7, !frisk.local_reduction !5
  %363 = extractelement <4 x float> %359, i64 2, !dbg !40
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %363), !dbg !7, !frisk.local_reduction !5
  %364 = extractelement <4 x float> %359, i64 3, !dbg !40
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %364), !dbg !7, !frisk.local_reduction !5
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %local.reduce168), !dbg !7, !frisk.local_reduction !5
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %local.reduce170), !dbg !7, !frisk.local_reduction !5
  %365 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %local.reduce172), !dbg !7, !frisk.local_reduction !5
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %301, <4 x float> %360) #8, !dbg !14
  %367 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %300, <4 x float> %284) #8, !dbg !14
  %368 = extractelement <4 x float> %366, i64 0, !dbg !40
  %369 = extractelement <4 x float> %366, i64 1, !dbg !40
  %370 = extractelement <4 x float> %366, i64 2, !dbg !40
  %371 = extractelement <4 x float> %366, i64 3, !dbg !40
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %302, <4 x float> %367) #8, !dbg !14
  %373 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %303, <4 x float> %286) #8, !dbg !14
  %374 = extractelement <4 x float> %372, i64 0, !dbg !40
  %375 = extractelement <4 x float> %372, i64 1, !dbg !40
  %376 = extractelement <4 x float> %372, i64 2, !dbg !40
  %377 = extractelement <4 x float> %372, i64 3, !dbg !40
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %306, <4 x float> %373) #8, !dbg !14
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %304, <4 x float> %288) #8, !dbg !14
  %380 = extractelement <4 x float> %378, i64 0, !dbg !40
  %381 = extractelement <4 x float> %378, i64 1, !dbg !40
  %382 = extractelement <4 x float> %378, i64 2, !dbg !40
  %383 = extractelement <4 x float> %378, i64 3, !dbg !40
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %307, <4 x float> %379) #8, !dbg !14
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %316, <4 x float> %290) #8, !dbg !14
  %386 = extractelement <4 x float> %384, i64 0, !dbg !40
  %387 = extractelement <4 x float> %384, i64 1, !dbg !40
  %388 = extractelement <4 x float> %384, i64 2, !dbg !40
  %389 = extractelement <4 x float> %384, i64 3, !dbg !40
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %323, <4 x float> %385) #8, !dbg !14
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %317, <4 x float> %292) #8, !dbg !14
  %392 = extractelement <4 x float> %390, i64 0, !dbg !40
  %393 = extractelement <4 x float> %390, i64 1, !dbg !40
  %394 = extractelement <4 x float> %390, i64 2, !dbg !40
  %395 = extractelement <4 x float> %390, i64 3, !dbg !40
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %324, <4 x float> %391) #8, !dbg !14
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %332, <4 x float> %294) #8, !dbg !14
  %398 = extractelement <4 x float> %396, i64 0, !dbg !40
  %399 = extractelement <4 x float> %396, i64 1, !dbg !40
  %400 = extractelement <4 x float> %396, i64 2, !dbg !40
  %401 = extractelement <4 x float> %396, i64 3, !dbg !40
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %339, <4 x float> %397) #8, !dbg !14
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %105, <4 x half> %333, <4 x float> %296) #8, !dbg !14
  %404 = extractelement <4 x float> %402, i64 0, !dbg !40
  %405 = extractelement <4 x float> %402, i64 1, !dbg !40
  %406 = extractelement <4 x float> %402, i64 2, !dbg !40
  %407 = extractelement <4 x float> %402, i64 3, !dbg !40
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %106, <4 x half> %340, <4 x float> %403) #8, !dbg !14
  %409 = extractelement <4 x float> %408, i64 0, !dbg !40
  %410 = extractelement <4 x float> %408, i64 1, !dbg !40
  %411 = extractelement <4 x float> %408, i64 2, !dbg !40
  %412 = extractelement <4 x float> %408, i64 3, !dbg !40
  %413 = bitcast float %365 to i32, !dbg !7
  %414 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %413), !dbg !7
  %415 = bitcast i32 %414 to float, !dbg !7
  %416 = tail call float @llvm.maxnum.f32(float %365, float %415), !dbg !7
  %417 = bitcast float %416 to i32, !dbg !7
  %418 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %417), !dbg !7
  %419 = bitcast i32 %418 to float, !dbg !7
  %420 = tail call float @llvm.maxnum.f32(float %416, float %419), !dbg !7
  %421 = tail call float @llvm.maxnum.f32(float %368, float -inf), !dbg !7, !frisk.local_reduction !5
  %local.reduce112 = tail call float @llvm.maxnum.f32(float %421, float %374), !dbg !7, !frisk.local_reduction !5
  %local.reduce113 = tail call float @llvm.maxnum.f32(float %369, float %375), !dbg !7, !frisk.local_reduction !5
  %local.reduce114 = tail call float @llvm.maxnum.f32(float %370, float %376), !dbg !7, !frisk.local_reduction !5
  %local.reduce115 = tail call float @llvm.maxnum.f32(float %371, float %377), !dbg !7, !frisk.local_reduction !5
  %local.reduce116 = tail call float @llvm.maxnum.f32(float %local.reduce112, float %380), !dbg !7, !frisk.local_reduction !5
  %local.reduce117 = tail call float @llvm.maxnum.f32(float %local.reduce113, float %381), !dbg !7, !frisk.local_reduction !5
  %local.reduce118 = tail call float @llvm.maxnum.f32(float %local.reduce114, float %382), !dbg !7, !frisk.local_reduction !5
  %local.reduce119 = tail call float @llvm.maxnum.f32(float %local.reduce115, float %383), !dbg !7, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %local.reduce116, float %386), !dbg !7, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %local.reduce117, float %387), !dbg !7, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %local.reduce118, float %388), !dbg !7, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %local.reduce119, float %389), !dbg !7, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %392), !dbg !7, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %393), !dbg !7, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %394), !dbg !7, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %395), !dbg !7, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %398), !dbg !7, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %399), !dbg !7, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %400), !dbg !7, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %401), !dbg !7, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %404), !dbg !7, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %405), !dbg !7, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %406), !dbg !7, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %407), !dbg !7, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %409), !dbg !7, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %410), !dbg !7, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %411), !dbg !7, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %412), !dbg !7, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %local.reduce137), !dbg !7, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %local.reduce139), !dbg !7, !frisk.local_reduction !5
  %422 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %local.reduce141), !dbg !7, !frisk.local_reduction !5
  %423 = bitcast float %422 to i32, !dbg !7
  %424 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %423), !dbg !7
  %425 = bitcast i32 %424 to float, !dbg !7
  %426 = tail call float @llvm.maxnum.f32(float %422, float %425), !dbg !7
  %427 = bitcast float %426 to i32, !dbg !7
  %428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %427), !dbg !7
  %429 = bitcast i32 %428 to float, !dbg !7
  %430 = tail call float @llvm.maxnum.f32(float %426, float %429), !dbg !7
  %431 = extractvalue [2 x <1 x float>] %114, 0, !dbg !7
  %432 = insertelement <1 x float> poison, float %420, i64 0, !dbg !7
  %433 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %432, <1 x float> %431), !dbg !7
  %434 = extractvalue [2 x <1 x float>] %114, 1, !dbg !7
  %435 = insertelement <1 x float> poison, float %430, i64 0, !dbg !7
  %436 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %435, <1 x float> %434), !dbg !7
  %437 = shufflevector <1 x float> %433, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !41
  %438 = shufflevector <1 x float> %436, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !41
  %439 = fsub <1 x float> %431, %433, !dbg !42
  %440 = extractelement <1 x float> %439, i64 0, !dbg !43
  %441 = tail call float @llvm.amdgcn.exp2.f32(float %440), !dbg !43
  %442 = insertelement <4 x float> poison, float %441, i64 0, !dbg !44
  %443 = shufflevector <4 x float> %442, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %444 = extractvalue [2 x <1 x float>] %115, 0, !dbg !45
  %445 = extractelement <1 x float> %444, i64 0, !dbg !45
  %446 = fmul float %445, %441, !dbg !45
  %447 = fsub <1 x float> %434, %436, !dbg !42
  %448 = extractelement <1 x float> %447, i64 0, !dbg !43
  %449 = tail call float @llvm.amdgcn.exp2.f32(float %448), !dbg !43
  %450 = insertelement <4 x float> poison, float %449, i64 0, !dbg !44
  %451 = shufflevector <4 x float> %450, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !44
  %452 = extractvalue [2 x <1 x float>] %115, 1, !dbg !45
  %453 = extractelement <1 x float> %452, i64 0, !dbg !45
  %454 = fmul float %453, %449, !dbg !45
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address = or disjoint i64 %address.local572, %120, !dbg !39
  %vdb.address1051 = trunc nuw nsw i64 %vdb.address to i32, !dbg !39
  %vdb.address1052 = shl nuw nsw i32 %vdb.address1051, 1, !dbg !39
  %vdb.address1057 = or disjoint i64 %address.local578, %120, !dbg !39
  %vdb.address1058 = trunc nuw nsw i64 %vdb.address1057 to i32, !dbg !39
  %vdb.address1059 = shl nuw nsw i32 %vdb.address1058, 1, !dbg !39
  %455 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1052, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %456 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1064, i32 %vdb.address1059, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %address.local580 = shl nuw nsw i32 %address.thread410, 3, !dbg !46
  %address.local581 = and i32 %address.local580, 504, !dbg !46
  %address.local582 = zext nneg i32 %address.local581 to i64, !dbg !46
  %address.local583 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local582, !dbg !46
  %457 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local583, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %458 = bitcast <4 x i32> %457 to <8 x half>, !dbg !46
  %address.local585 = getelementptr i8, ptr addrspace(3) %address.local583, i64 1024, !dbg !46
  %459 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local585, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %460 = bitcast <4 x i32> %459 to <8 x half>, !dbg !46
  %address.local587 = getelementptr i8, ptr addrspace(3) %address.local583, i64 2048, !dbg !46
  %461 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local587, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %462 = bitcast <4 x i32> %461 to <8 x half>, !dbg !46
  %address.local589 = getelementptr i8, ptr addrspace(3) %address.local583, i64 3072, !dbg !46
  %463 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local589, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %464 = bitcast <4 x i32> %463 to <8 x half>, !dbg !46
  %address.local591 = getelementptr i8, ptr addrspace(3) %address.local583, i64 4096, !dbg !46
  %465 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local591, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %466 = bitcast <4 x i32> %465 to <8 x half>, !dbg !46
  %address.local593 = getelementptr i8, ptr addrspace(3) %address.local583, i64 5120, !dbg !46
  %467 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local593, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %468 = bitcast <4 x i32> %467 to <8 x half>, !dbg !46
  %address.local595 = getelementptr i8, ptr addrspace(3) %address.local583, i64 6144, !dbg !46
  %469 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local595, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %470 = bitcast <4 x i32> %469 to <8 x half>, !dbg !46
  %address.local597 = getelementptr i8, ptr addrspace(3) %address.local583, i64 7168, !dbg !46
  %471 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local597, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %472 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %473 = extractvalue [2 x <8 x float>] %116, 0, !dbg !44
  %474 = shufflevector <8 x float> %473, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %475 = fmul <4 x float> %474, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %476 = fsub <4 x float> %308, %437, !dbg !41
  %477 = extractelement <4 x float> %476, i64 0, !dbg !48
  %478 = tail call float @llvm.amdgcn.exp2.f32(float %477), !dbg !48
  %479 = insertelement <4 x float> poison, float %478, i64 0, !dbg !49
  %480 = extractelement <4 x float> %476, i64 1, !dbg !48
  %481 = tail call float @llvm.amdgcn.exp2.f32(float %480), !dbg !48
  %482 = insertelement <4 x float> %479, float %481, i64 1, !dbg !49
  %483 = extractelement <4 x float> %476, i64 2, !dbg !48
  %484 = tail call float @llvm.amdgcn.exp2.f32(float %483), !dbg !48
  %485 = insertelement <4 x float> %482, float %484, i64 2, !dbg !49
  %486 = extractelement <4 x float> %476, i64 3, !dbg !48
  %487 = tail call float @llvm.amdgcn.exp2.f32(float %486), !dbg !48
  %488 = insertelement <4 x float> %485, float %487, i64 3, !dbg !49
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %488) #8, !dbg !49, !frisk.register_lifetimes !5
  %489 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !49
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %489) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %490 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %472, <4 x float> %475) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %491 = shufflevector <8 x half> %458, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %492 = shufflevector <8 x float> %473, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %493 = fmul <4 x float> %492, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %494 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %491, <4 x float> %493) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %495 = extractvalue [2 x <8 x float>] %116, 1, !dbg !44
  %496 = shufflevector <8 x float> %495, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %497 = fmul <4 x float> %496, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %498 = fsub <4 x float> %366, %438, !dbg !41
  %499 = extractelement <4 x float> %498, i64 0, !dbg !48
  %500 = tail call float @llvm.amdgcn.exp2.f32(float %499), !dbg !48
  %501 = insertelement <4 x float> poison, float %500, i64 0, !dbg !49
  %502 = extractelement <4 x float> %498, i64 1, !dbg !48
  %503 = tail call float @llvm.amdgcn.exp2.f32(float %502), !dbg !48
  %504 = insertelement <4 x float> %501, float %503, i64 1, !dbg !49
  %505 = extractelement <4 x float> %498, i64 2, !dbg !48
  %506 = tail call float @llvm.amdgcn.exp2.f32(float %505), !dbg !48
  %507 = insertelement <4 x float> %504, float %506, i64 2, !dbg !49
  %508 = extractelement <4 x float> %498, i64 3, !dbg !48
  %509 = tail call float @llvm.amdgcn.exp2.f32(float %508), !dbg !48
  %510 = insertelement <4 x float> %507, float %509, i64 3, !dbg !49
  %math.result300 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %510) #8, !dbg !49, !frisk.register_lifetimes !5
  %511 = fptrunc <4 x float> %math.result300 to <4 x half>, !dbg !49
  %math.packed301 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %511) #8, !dbg !48, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %512 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %472, <4 x float> %497) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %513 = shufflevector <8 x float> %495, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %514 = fmul <4 x float> %513, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %515 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %491, <4 x float> %514) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %516 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %517 = fsub <4 x float> %315, %437, !dbg !41
  %518 = extractelement <4 x float> %517, i64 0, !dbg !48
  %519 = tail call float @llvm.amdgcn.exp2.f32(float %518), !dbg !48
  %local.reduce81 = fadd float %478, %519, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %520 = insertelement <4 x float> poison, float %519, i64 0, !dbg !49
  %521 = extractelement <4 x float> %517, i64 1, !dbg !48
  %522 = tail call float @llvm.amdgcn.exp2.f32(float %521), !dbg !48
  %local.reduce82 = fadd float %481, %522, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1504 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %523 = insertelement <4 x float> %520, float %522, i64 1, !dbg !49
  %524 = extractelement <4 x float> %517, i64 2, !dbg !48
  %525 = tail call float @llvm.amdgcn.exp2.f32(float %524), !dbg !48
  %local.reduce83 = fadd float %484, %525, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1505 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %526 = insertelement <4 x float> %523, float %525, i64 2, !dbg !49
  %527 = extractelement <4 x float> %517, i64 3, !dbg !48
  %528 = tail call float @llvm.amdgcn.exp2.f32(float %527), !dbg !48
  %local.reduce84 = fadd float %487, %528, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1506 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %529 = insertelement <4 x float> %526, float %528, i64 3, !dbg !49
  %math.result286 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %529) #8, !dbg !49, !frisk.register_lifetimes !5
  %530 = fptrunc <4 x float> %math.result286 to <4 x half>, !dbg !49
  %math.packed287 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %530) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %531 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %516, <4 x float> %490) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %532 = shufflevector <8 x half> %460, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %533 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %532, <4 x float> %494) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %534 = fsub <4 x float> %372, %438, !dbg !41
  %535 = extractelement <4 x float> %534, i64 0, !dbg !48
  %536 = tail call float @llvm.amdgcn.exp2.f32(float %535), !dbg !48
  %local.reduce = fadd float %500, %536, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1507 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %537 = insertelement <4 x float> poison, float %536, i64 0, !dbg !49
  %538 = extractelement <4 x float> %534, i64 1, !dbg !48
  %539 = tail call float @llvm.amdgcn.exp2.f32(float %538), !dbg !48
  %local.reduce51 = fadd float %503, %539, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1508 = tail call float asm sideeffect "", "=v,0"(float %local.reduce51) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %540 = insertelement <4 x float> %537, float %539, i64 1, !dbg !49
  %541 = extractelement <4 x float> %534, i64 2, !dbg !48
  %542 = tail call float @llvm.amdgcn.exp2.f32(float %541), !dbg !48
  %local.reduce52 = fadd float %506, %542, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1509 = tail call float asm sideeffect "", "=v,0"(float %local.reduce52) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %543 = insertelement <4 x float> %540, float %542, i64 2, !dbg !49
  %544 = extractelement <4 x float> %534, i64 3, !dbg !48
  %545 = tail call float @llvm.amdgcn.exp2.f32(float %544), !dbg !48
  %local.reduce53 = fadd float %509, %545, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1510 = tail call float asm sideeffect "", "=v,0"(float %local.reduce53) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %546 = insertelement <4 x float> %543, float %545, i64 3, !dbg !49
  %math.result302 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %546) #8, !dbg !49, !frisk.register_lifetimes !5
  %547 = fptrunc <4 x float> %math.result302 to <4 x half>, !dbg !49
  %math.packed303 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %547) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %548 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %516, <4 x float> %512) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %549 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %532, <4 x float> %515) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %550 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %551 = fsub <4 x float> %325, %437, !dbg !41
  %552 = extractelement <4 x float> %551, i64 0, !dbg !48
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !48
  %local.reduce85 = fadd float %553, %k0.sum.anchor, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1511 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %554 = insertelement <4 x float> poison, float %553, i64 0, !dbg !49
  %555 = extractelement <4 x float> %551, i64 1, !dbg !48
  %556 = tail call float @llvm.amdgcn.exp2.f32(float %555), !dbg !48
  %local.reduce86 = fadd float %556, %k0.sum.anchor1504, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1512 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %557 = insertelement <4 x float> %554, float %556, i64 1, !dbg !49
  %558 = extractelement <4 x float> %551, i64 2, !dbg !48
  %559 = tail call float @llvm.amdgcn.exp2.f32(float %558), !dbg !48
  %local.reduce87 = fadd float %559, %k0.sum.anchor1505, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1513 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %560 = insertelement <4 x float> %557, float %559, i64 2, !dbg !49
  %561 = extractelement <4 x float> %551, i64 3, !dbg !48
  %562 = tail call float @llvm.amdgcn.exp2.f32(float %561), !dbg !48
  %local.reduce88 = fadd float %562, %k0.sum.anchor1506, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1514 = tail call float asm sideeffect "", "=v,0"(float %local.reduce88) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %563 = insertelement <4 x float> %560, float %562, i64 3, !dbg !49
  %math.result288 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %563) #8, !dbg !49, !frisk.register_lifetimes !5
  %564 = fptrunc <4 x float> %math.result288 to <4 x half>, !dbg !49
  %math.packed289 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %564) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %565 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %550, <4 x float> %531) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %566 = shufflevector <8 x half> %462, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %567 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %566, <4 x float> %533) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %568 = fsub <4 x float> %378, %438, !dbg !41
  %569 = extractelement <4 x float> %568, i64 0, !dbg !48
  %570 = tail call float @llvm.amdgcn.exp2.f32(float %569), !dbg !48
  %local.reduce54 = fadd float %570, %k0.sum.anchor1507, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1515 = tail call float asm sideeffect "", "=v,0"(float %local.reduce54) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %571 = insertelement <4 x float> poison, float %570, i64 0, !dbg !49
  %572 = extractelement <4 x float> %568, i64 1, !dbg !48
  %573 = tail call float @llvm.amdgcn.exp2.f32(float %572), !dbg !48
  %local.reduce55 = fadd float %573, %k0.sum.anchor1508, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1516 = tail call float asm sideeffect "", "=v,0"(float %local.reduce55) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %574 = insertelement <4 x float> %571, float %573, i64 1, !dbg !49
  %575 = extractelement <4 x float> %568, i64 2, !dbg !48
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %575), !dbg !48
  %local.reduce56 = fadd float %576, %k0.sum.anchor1509, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1517 = tail call float asm sideeffect "", "=v,0"(float %local.reduce56) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %577 = insertelement <4 x float> %574, float %576, i64 2, !dbg !49
  %578 = extractelement <4 x float> %568, i64 3, !dbg !48
  %579 = tail call float @llvm.amdgcn.exp2.f32(float %578), !dbg !48
  %local.reduce57 = fadd float %579, %k0.sum.anchor1510, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1518 = tail call float asm sideeffect "", "=v,0"(float %local.reduce57) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %580 = insertelement <4 x float> %577, float %579, i64 3, !dbg !49
  %math.result304 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %580) #8, !dbg !49, !frisk.register_lifetimes !5
  %581 = fptrunc <4 x float> %math.result304 to <4 x half>, !dbg !49
  %math.packed305 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %581) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %582 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %550, <4 x float> %548) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %583 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %566, <4 x float> %549) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %584 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %585 = fsub <4 x float> %331, %437, !dbg !41
  %586 = extractelement <4 x float> %585, i64 0, !dbg !48
  %587 = tail call float @llvm.amdgcn.exp2.f32(float %586), !dbg !48
  %local.reduce89 = fadd float %587, %k0.sum.anchor1511, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1519 = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %588 = insertelement <4 x float> poison, float %587, i64 0, !dbg !49
  %589 = extractelement <4 x float> %585, i64 1, !dbg !48
  %590 = tail call float @llvm.amdgcn.exp2.f32(float %589), !dbg !48
  %local.reduce90 = fadd float %590, %k0.sum.anchor1512, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1520 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %591 = insertelement <4 x float> %588, float %590, i64 1, !dbg !49
  %592 = extractelement <4 x float> %585, i64 2, !dbg !48
  %593 = tail call float @llvm.amdgcn.exp2.f32(float %592), !dbg !48
  %local.reduce91 = fadd float %593, %k0.sum.anchor1513, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1521 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %594 = insertelement <4 x float> %591, float %593, i64 2, !dbg !49
  %595 = extractelement <4 x float> %585, i64 3, !dbg !48
  %596 = tail call float @llvm.amdgcn.exp2.f32(float %595), !dbg !48
  %local.reduce92 = fadd float %596, %k0.sum.anchor1514, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1522 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %597 = insertelement <4 x float> %594, float %596, i64 3, !dbg !49
  %math.result290 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %597) #8, !dbg !49, !frisk.register_lifetimes !5
  %598 = fptrunc <4 x float> %math.result290 to <4 x half>, !dbg !49
  %math.packed291 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %598) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %599 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %584, <4 x float> %565) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %600 = shufflevector <8 x half> %464, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %601 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %600, <4 x float> %567) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %602 = fsub <4 x float> %384, %438, !dbg !41
  %603 = extractelement <4 x float> %602, i64 0, !dbg !48
  %604 = tail call float @llvm.amdgcn.exp2.f32(float %603), !dbg !48
  %local.reduce58 = fadd float %604, %k0.sum.anchor1515, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1523 = tail call float asm sideeffect "", "=v,0"(float %local.reduce58) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %605 = insertelement <4 x float> poison, float %604, i64 0, !dbg !49
  %606 = extractelement <4 x float> %602, i64 1, !dbg !48
  %607 = tail call float @llvm.amdgcn.exp2.f32(float %606), !dbg !48
  %local.reduce59 = fadd float %607, %k0.sum.anchor1516, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1524 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %608 = insertelement <4 x float> %605, float %607, i64 1, !dbg !49
  %609 = extractelement <4 x float> %602, i64 2, !dbg !48
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %609), !dbg !48
  %local.reduce60 = fadd float %610, %k0.sum.anchor1517, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1525 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %611 = insertelement <4 x float> %608, float %610, i64 2, !dbg !49
  %612 = extractelement <4 x float> %602, i64 3, !dbg !48
  %613 = tail call float @llvm.amdgcn.exp2.f32(float %612), !dbg !48
  %local.reduce61 = fadd float %613, %k0.sum.anchor1518, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1526 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %614 = insertelement <4 x float> %611, float %613, i64 3, !dbg !49
  %math.result306 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %614) #8, !dbg !49, !frisk.register_lifetimes !5
  %615 = fptrunc <4 x float> %math.result306 to <4 x half>, !dbg !49
  %math.packed307 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %615) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %616 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %584, <4 x float> %582) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %617 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %600, <4 x float> %583) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %618 = shufflevector <8 x half> %466, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %619 = fsub <4 x float> %341, %437, !dbg !41
  %620 = extractelement <4 x float> %619, i64 0, !dbg !48
  %621 = tail call float @llvm.amdgcn.exp2.f32(float %620), !dbg !48
  %local.reduce93 = fadd float %621, %k0.sum.anchor1519, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1527 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %622 = insertelement <4 x float> poison, float %621, i64 0, !dbg !49
  %623 = extractelement <4 x float> %619, i64 1, !dbg !48
  %624 = tail call float @llvm.amdgcn.exp2.f32(float %623), !dbg !48
  %local.reduce94 = fadd float %624, %k0.sum.anchor1520, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1528 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %625 = insertelement <4 x float> %622, float %624, i64 1, !dbg !49
  %626 = extractelement <4 x float> %619, i64 2, !dbg !48
  %627 = tail call float @llvm.amdgcn.exp2.f32(float %626), !dbg !48
  %local.reduce95 = fadd float %627, %k0.sum.anchor1521, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1529 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %628 = insertelement <4 x float> %625, float %627, i64 2, !dbg !49
  %629 = extractelement <4 x float> %619, i64 3, !dbg !48
  %630 = tail call float @llvm.amdgcn.exp2.f32(float %629), !dbg !48
  %local.reduce96 = fadd float %630, %k0.sum.anchor1522, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1530 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %631 = insertelement <4 x float> %628, float %630, i64 3, !dbg !49
  %math.result292 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %631) #8, !dbg !49, !frisk.register_lifetimes !5
  %632 = fptrunc <4 x float> %math.result292 to <4 x half>, !dbg !49
  %math.packed293 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %632) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %633 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %618, <4 x float> %599) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %634 = shufflevector <8 x half> %466, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %635 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %634, <4 x float> %601) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %636 = fsub <4 x float> %390, %438, !dbg !41
  %637 = extractelement <4 x float> %636, i64 0, !dbg !48
  %638 = tail call float @llvm.amdgcn.exp2.f32(float %637), !dbg !48
  %local.reduce62 = fadd float %638, %k0.sum.anchor1523, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1531 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %639 = insertelement <4 x float> poison, float %638, i64 0, !dbg !49
  %640 = extractelement <4 x float> %636, i64 1, !dbg !48
  %641 = tail call float @llvm.amdgcn.exp2.f32(float %640), !dbg !48
  %local.reduce63 = fadd float %641, %k0.sum.anchor1524, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1532 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %642 = insertelement <4 x float> %639, float %641, i64 1, !dbg !49
  %643 = extractelement <4 x float> %636, i64 2, !dbg !48
  %644 = tail call float @llvm.amdgcn.exp2.f32(float %643), !dbg !48
  %local.reduce64 = fadd float %644, %k0.sum.anchor1525, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1533 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %645 = insertelement <4 x float> %642, float %644, i64 2, !dbg !49
  %646 = extractelement <4 x float> %636, i64 3, !dbg !48
  %647 = tail call float @llvm.amdgcn.exp2.f32(float %646), !dbg !48
  %local.reduce65 = fadd float %647, %k0.sum.anchor1526, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1534 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %648 = insertelement <4 x float> %645, float %647, i64 3, !dbg !49
  %math.result308 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %648) #8, !dbg !49, !frisk.register_lifetimes !5
  %649 = fptrunc <4 x float> %math.result308 to <4 x half>, !dbg !49
  %math.packed309 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %649) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %650 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %618, <4 x float> %616) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %651 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %634, <4 x float> %617) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %652 = shufflevector <8 x half> %468, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %653 = fsub <4 x float> %347, %437, !dbg !41
  %654 = extractelement <4 x float> %653, i64 0, !dbg !48
  %655 = tail call float @llvm.amdgcn.exp2.f32(float %654), !dbg !48
  %local.reduce97 = fadd float %655, %k0.sum.anchor1527, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1535 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %656 = insertelement <4 x float> poison, float %655, i64 0, !dbg !49
  %657 = extractelement <4 x float> %653, i64 1, !dbg !48
  %658 = tail call float @llvm.amdgcn.exp2.f32(float %657), !dbg !48
  %local.reduce98 = fadd float %658, %k0.sum.anchor1528, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1536 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %659 = insertelement <4 x float> %656, float %658, i64 1, !dbg !49
  %660 = extractelement <4 x float> %653, i64 2, !dbg !48
  %661 = tail call float @llvm.amdgcn.exp2.f32(float %660), !dbg !48
  %local.reduce99 = fadd float %661, %k0.sum.anchor1529, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1537 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %662 = insertelement <4 x float> %659, float %661, i64 2, !dbg !49
  %663 = extractelement <4 x float> %653, i64 3, !dbg !48
  %664 = tail call float @llvm.amdgcn.exp2.f32(float %663), !dbg !48
  %local.reduce100 = fadd float %664, %k0.sum.anchor1530, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1538 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %665 = insertelement <4 x float> %662, float %664, i64 3, !dbg !49
  %math.result294 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %665) #8, !dbg !49, !frisk.register_lifetimes !5
  %666 = fptrunc <4 x float> %math.result294 to <4 x half>, !dbg !49
  %math.packed295 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %666) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %667 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %652, <4 x float> %633) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %668 = shufflevector <8 x half> %468, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %669 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %668, <4 x float> %635) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %670 = fsub <4 x float> %396, %438, !dbg !41
  %671 = extractelement <4 x float> %670, i64 0, !dbg !48
  %672 = tail call float @llvm.amdgcn.exp2.f32(float %671), !dbg !48
  %local.reduce66 = fadd float %672, %k0.sum.anchor1531, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1539 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %673 = insertelement <4 x float> poison, float %672, i64 0, !dbg !49
  %674 = extractelement <4 x float> %670, i64 1, !dbg !48
  %675 = tail call float @llvm.amdgcn.exp2.f32(float %674), !dbg !48
  %local.reduce67 = fadd float %675, %k0.sum.anchor1532, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1540 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %676 = insertelement <4 x float> %673, float %675, i64 1, !dbg !49
  %677 = extractelement <4 x float> %670, i64 2, !dbg !48
  %678 = tail call float @llvm.amdgcn.exp2.f32(float %677), !dbg !48
  %local.reduce68 = fadd float %678, %k0.sum.anchor1533, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1541 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %679 = insertelement <4 x float> %676, float %678, i64 2, !dbg !49
  %680 = extractelement <4 x float> %670, i64 3, !dbg !48
  %681 = tail call float @llvm.amdgcn.exp2.f32(float %680), !dbg !48
  %local.reduce69 = fadd float %681, %k0.sum.anchor1534, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1542 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %682 = insertelement <4 x float> %679, float %681, i64 3, !dbg !49
  %math.result310 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %682) #8, !dbg !49, !frisk.register_lifetimes !5
  %683 = fptrunc <4 x float> %math.result310 to <4 x half>, !dbg !49
  %math.packed311 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %683) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %684 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %652, <4 x float> %650) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %685 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %668, <4 x float> %651) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %686 = shufflevector <8 x half> %470, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %687 = fsub <4 x float> %353, %437, !dbg !41
  %688 = extractelement <4 x float> %687, i64 0, !dbg !48
  %689 = tail call float @llvm.amdgcn.exp2.f32(float %688), !dbg !48
  %local.reduce101 = fadd float %689, %k0.sum.anchor1535, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1543 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %690 = insertelement <4 x float> poison, float %689, i64 0, !dbg !49
  %691 = extractelement <4 x float> %687, i64 1, !dbg !48
  %692 = tail call float @llvm.amdgcn.exp2.f32(float %691), !dbg !48
  %local.reduce102 = fadd float %692, %k0.sum.anchor1536, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1544 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %693 = insertelement <4 x float> %690, float %692, i64 1, !dbg !49
  %694 = extractelement <4 x float> %687, i64 2, !dbg !48
  %695 = tail call float @llvm.amdgcn.exp2.f32(float %694), !dbg !48
  %local.reduce103 = fadd float %695, %k0.sum.anchor1537, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1545 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %696 = insertelement <4 x float> %693, float %695, i64 2, !dbg !49
  %697 = extractelement <4 x float> %687, i64 3, !dbg !48
  %698 = tail call float @llvm.amdgcn.exp2.f32(float %697), !dbg !48
  %local.reduce104 = fadd float %698, %k0.sum.anchor1538, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1546 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %699 = insertelement <4 x float> %696, float %698, i64 3, !dbg !49
  %math.result296 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %699) #8, !dbg !49, !frisk.register_lifetimes !5
  %700 = fptrunc <4 x float> %math.result296 to <4 x half>, !dbg !49
  %math.packed297 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %700) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %701 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %686, <4 x float> %667) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %702 = shufflevector <8 x half> %470, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %703 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %702, <4 x float> %669) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %704 = fsub <4 x float> %402, %438, !dbg !41
  %705 = extractelement <4 x float> %704, i64 0, !dbg !48
  %706 = tail call float @llvm.amdgcn.exp2.f32(float %705), !dbg !48
  %local.reduce70 = fadd float %706, %k0.sum.anchor1539, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1547 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %707 = insertelement <4 x float> poison, float %706, i64 0, !dbg !49
  %708 = extractelement <4 x float> %704, i64 1, !dbg !48
  %709 = tail call float @llvm.amdgcn.exp2.f32(float %708), !dbg !48
  %local.reduce71 = fadd float %709, %k0.sum.anchor1540, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1548 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %710 = insertelement <4 x float> %707, float %709, i64 1, !dbg !49
  %711 = extractelement <4 x float> %704, i64 2, !dbg !48
  %712 = tail call float @llvm.amdgcn.exp2.f32(float %711), !dbg !48
  %local.reduce72 = fadd float %712, %k0.sum.anchor1541, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1549 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %713 = insertelement <4 x float> %710, float %712, i64 2, !dbg !49
  %714 = extractelement <4 x float> %704, i64 3, !dbg !48
  %715 = tail call float @llvm.amdgcn.exp2.f32(float %714), !dbg !48
  %local.reduce73 = fadd float %715, %k0.sum.anchor1542, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1550 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %716 = insertelement <4 x float> %713, float %715, i64 3, !dbg !49
  %math.result312 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %716) #8, !dbg !49, !frisk.register_lifetimes !5
  %717 = fptrunc <4 x float> %math.result312 to <4 x half>, !dbg !49
  %math.packed313 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %717) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %718 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %686, <4 x float> %684) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %719 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %702, <4 x float> %685) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %720 = bitcast <4 x i32> %471 to <8 x half>, !dbg !46
  %721 = shufflevector <8 x half> %720, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %722 = fsub <4 x float> %359, %437, !dbg !41
  %723 = extractelement <4 x float> %722, i64 0, !dbg !48
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %723), !dbg !48
  %local.reduce105 = fadd float %724, %k0.sum.anchor1543, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1551 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %725 = insertelement <4 x float> poison, float %724, i64 0, !dbg !49
  %726 = extractelement <4 x float> %722, i64 1, !dbg !48
  %727 = tail call float @llvm.amdgcn.exp2.f32(float %726), !dbg !48
  %local.reduce106 = fadd float %727, %k0.sum.anchor1544, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1552 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce109 = fadd float %k0.sum.anchor1551, %k0.sum.anchor1552, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1553 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %728 = insertelement <4 x float> %725, float %727, i64 1, !dbg !49
  %729 = extractelement <4 x float> %722, i64 2, !dbg !48
  %730 = tail call float @llvm.amdgcn.exp2.f32(float %729), !dbg !48
  %local.reduce107 = fadd float %730, %k0.sum.anchor1545, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1554 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %731 = insertelement <4 x float> %728, float %730, i64 2, !dbg !49
  %732 = extractelement <4 x float> %722, i64 3, !dbg !48
  %733 = tail call float @llvm.amdgcn.exp2.f32(float %732), !dbg !48
  %local.reduce108 = fadd float %733, %k0.sum.anchor1546, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1555 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce110 = fadd float %k0.sum.anchor1554, %k0.sum.anchor1555, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1556 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %734 = fadd float %k0.sum.anchor1553, %k0.sum.anchor1556, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1557 = tail call float asm sideeffect "", "=v,0"(float %734) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %735 = fadd float %446, %k0.sum.anchor1557, !dbg !50
  %k0.sum.anchor1558 = tail call float asm sideeffect "", "=v,0"(float %735) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %736 = insertelement <1 x float> poison, float %k0.sum.anchor1558, i64 0, !dbg !12
  %737 = insertvalue [2 x <1 x float>] poison, <1 x float> %736, 0, !dbg !12
  %738 = insertelement <4 x float> %731, float %733, i64 3, !dbg !49
  %math.result298 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %738) #8, !dbg !49, !frisk.register_lifetimes !5
  %739 = fptrunc <4 x float> %math.result298 to <4 x half>, !dbg !49
  %math.packed299 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %739) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %740 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %721, <4 x float> %701) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %741 = shufflevector <8 x half> %720, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %741, <4 x float> %703) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %743 = fsub <4 x float> %408, %438, !dbg !41
  %744 = extractelement <4 x float> %743, i64 0, !dbg !48
  %745 = tail call float @llvm.amdgcn.exp2.f32(float %744), !dbg !48
  %local.reduce74 = fadd float %745, %k0.sum.anchor1547, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1559 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %746 = insertelement <4 x float> poison, float %745, i64 0, !dbg !49
  %747 = extractelement <4 x float> %743, i64 1, !dbg !48
  %748 = tail call float @llvm.amdgcn.exp2.f32(float %747), !dbg !48
  %local.reduce75 = fadd float %748, %k0.sum.anchor1548, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1560 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce78 = fadd float %k0.sum.anchor1559, %k0.sum.anchor1560, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1561 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %749 = insertelement <4 x float> %746, float %748, i64 1, !dbg !49
  %750 = extractelement <4 x float> %743, i64 2, !dbg !48
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %750), !dbg !48
  %local.reduce76 = fadd float %751, %k0.sum.anchor1549, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1562 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %752 = insertelement <4 x float> %749, float %751, i64 2, !dbg !49
  %753 = extractelement <4 x float> %743, i64 3, !dbg !48
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !48
  %local.reduce77 = fadd float %754, %k0.sum.anchor1550, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1563 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %local.reduce79 = fadd float %k0.sum.anchor1562, %k0.sum.anchor1563, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1564 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %755 = fadd float %k0.sum.anchor1561, %k0.sum.anchor1564, !dbg !50, !frisk.local_reduction !5
  %k0.sum.anchor1565 = tail call float asm sideeffect "", "=v,0"(float %755) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %756 = fadd float %454, %k0.sum.anchor1565, !dbg !50
  %k0.sum.anchor1566 = tail call float asm sideeffect "", "=v,0"(float %756) #8, !dbg !50, !frisk.cross_bn_sum_anchor !5
  %757 = insertelement <1 x float> poison, float %k0.sum.anchor1566, i64 0, !dbg !12
  %758 = insertvalue [2 x <1 x float>] %737, <1 x float> %757, 1, !dbg !12
  %759 = insertelement <4 x float> %752, float %754, i64 3, !dbg !49
  %math.result314 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %759) #8, !dbg !49, !frisk.register_lifetimes !5
  %760 = fptrunc <4 x float> %math.result314 to <4 x half>, !dbg !49
  %math.packed315 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %760) #8, !dbg !50, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46, !frisk.softmax_fragments !5
  %761 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %721, <4 x float> %718) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %762 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %741, <4 x float> %719) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1072 = or disjoint i64 %address.local572, %191, !dbg !39
  %vdb.address1073 = trunc nuw nsw i64 %vdb.address1072 to i32, !dbg !39
  %vdb.address1074 = shl nuw nsw i32 %vdb.address1073, 1, !dbg !39
  %vdb.address1079 = or disjoint i64 %address.local578, %191, !dbg !39
  %vdb.address1080 = trunc nuw nsw i64 %vdb.address1079 to i32, !dbg !39
  %vdb.address1081 = shl nuw nsw i32 %vdb.address1080, 1, !dbg !39
  %763 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %vdb.address1074, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %764 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %vdb.address1081, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local583, i64 8192, !dbg !46
  %765 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %766 = bitcast <4 x i32> %765 to <8 x half>, !dbg !46
  %vdb.read1065 = getelementptr i8, ptr addrspace(3) %address.local583, i64 9216, !dbg !46
  %767 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1065, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %768 = bitcast <4 x i32> %767 to <8 x half>, !dbg !46
  %vdb.read1066 = getelementptr i8, ptr addrspace(3) %address.local583, i64 10240, !dbg !46
  %769 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1066, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %770 = bitcast <4 x i32> %769 to <8 x half>, !dbg !46
  %vdb.read1067 = getelementptr i8, ptr addrspace(3) %address.local583, i64 11264, !dbg !46
  %771 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1067, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %772 = bitcast <4 x i32> %771 to <8 x half>, !dbg !46
  %vdb.read1068 = getelementptr i8, ptr addrspace(3) %address.local583, i64 12288, !dbg !46
  %773 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1068, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %774 = bitcast <4 x i32> %773 to <8 x half>, !dbg !46
  %vdb.read1069 = getelementptr i8, ptr addrspace(3) %address.local583, i64 13312, !dbg !46
  %775 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1069, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %776 = bitcast <4 x i32> %775 to <8 x half>, !dbg !46
  %vdb.read1070 = getelementptr i8, ptr addrspace(3) %address.local583, i64 14336, !dbg !46
  %777 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1070, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %778 = bitcast <4 x i32> %777 to <8 x half>, !dbg !46
  %vdb.read1071 = getelementptr i8, ptr addrspace(3) %address.local583, i64 15360, !dbg !46
  %779 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1071, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %780 = shufflevector <8 x half> %766, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %781 = extractvalue [2 x <8 x float>] %113, 0, !dbg !44
  %782 = shufflevector <8 x float> %781, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %783 = fmul <4 x float> %782, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %784 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %780, <4 x float> %783) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %785 = shufflevector <8 x half> %766, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %786 = shufflevector <8 x float> %781, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %787 = fmul <4 x float> %786, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %788 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %785, <4 x float> %787) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %789 = extractvalue [2 x <8 x float>] %113, 1, !dbg !44
  %790 = shufflevector <8 x float> %789, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %791 = fmul <4 x float> %790, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %792 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %780, <4 x float> %791) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %793 = shufflevector <8 x float> %789, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %794 = fmul <4 x float> %793, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %795 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %785, <4 x float> %794) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %796 = shufflevector <8 x half> %768, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %797 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %796, <4 x float> %784) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %798 = shufflevector <8 x half> %768, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %799 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %798, <4 x float> %788) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %800 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %796, <4 x float> %792) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %801 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %798, <4 x float> %795) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %802 = shufflevector <8 x half> %770, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %803 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %802, <4 x float> %797) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %804 = shufflevector <8 x half> %770, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %805 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %804, <4 x float> %799) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %806 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %802, <4 x float> %800) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %807 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %804, <4 x float> %801) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %808 = shufflevector <8 x half> %772, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %809 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %808, <4 x float> %803) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %810 = shufflevector <8 x half> %772, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %811 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %810, <4 x float> %805) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %808, <4 x float> %806) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %813 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %810, <4 x float> %807) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %814 = shufflevector <8 x half> %774, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %815 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %814, <4 x float> %809) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %816 = shufflevector <8 x half> %774, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %817 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %816, <4 x float> %811) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %818 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %814, <4 x float> %812) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %816, <4 x float> %813) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %820 = shufflevector <8 x half> %776, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %820, <4 x float> %815) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %822 = shufflevector <8 x half> %776, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %823 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %822, <4 x float> %817) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %820, <4 x float> %818) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %822, <4 x float> %819) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %826 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %827 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %826, <4 x float> %821) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %828 = shufflevector <8 x half> %778, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %829 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %828, <4 x float> %823) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %826, <4 x float> %824) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %828, <4 x float> %825) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %832 = bitcast <4 x i32> %779 to <8 x half>, !dbg !46
  %833 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %834 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %833, <4 x float> %827) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %835 = shufflevector <8 x half> %832, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %835, <4 x float> %829) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %833, <4 x float> %830) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %838 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %835, <4 x float> %831) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %vdb.address1086 = or disjoint i64 %address.local572, %244, !dbg !39
  %vdb.address1087 = trunc nuw nsw i64 %vdb.address1086 to i32, !dbg !39
  %vdb.address1088 = shl nuw nsw i32 %vdb.address1087, 1, !dbg !39
  %vdb.address1093 = or disjoint i64 %address.local578, %244, !dbg !39
  %vdb.address1094 = trunc nuw nsw i64 %vdb.address1093 to i32, !dbg !39
  %vdb.address1095 = shl nuw nsw i32 %vdb.address1094, 1, !dbg !39
  %839 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1088, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %840 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1064, i32 %vdb.address1095, <4 x i32> %dma.buffer.rsrc973) #9, !dbg !39
  %841 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local583, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %842 = bitcast <4 x i32> %841 to <8 x half>, !dbg !46
  %843 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local585, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %844 = bitcast <4 x i32> %843 to <8 x half>, !dbg !46
  %845 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local587, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %846 = bitcast <4 x i32> %845 to <8 x half>, !dbg !46
  %847 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local589, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %848 = bitcast <4 x i32> %847 to <8 x half>, !dbg !46
  %849 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local591, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %850 = bitcast <4 x i32> %849 to <8 x half>, !dbg !46
  %851 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local593, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %852 = bitcast <4 x i32> %851 to <8 x half>, !dbg !46
  %853 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local595, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %854 = bitcast <4 x i32> %853 to <8 x half>, !dbg !46
  %855 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local597, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %856 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %857 = extractvalue [2 x <8 x float>] %112, 0, !dbg !44
  %858 = shufflevector <8 x float> %857, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %859 = fmul <4 x float> %858, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %856, <4 x float> %859) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %861 = shufflevector <8 x half> %842, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %862 = shufflevector <8 x float> %857, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %863 = fmul <4 x float> %862, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %864 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %861, <4 x float> %863) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %865 = extractvalue [2 x <8 x float>] %112, 1, !dbg !44
  %866 = shufflevector <8 x float> %865, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %867 = fmul <4 x float> %866, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %868 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %856, <4 x float> %867) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %869 = shufflevector <8 x float> %865, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %870 = fmul <4 x float> %869, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %871 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %861, <4 x float> %870) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %872 = shufflevector <8 x half> %844, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %873 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %872, <4 x float> %860) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %874 = shufflevector <8 x half> %844, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %875 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %874, <4 x float> %864) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %876 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %872, <4 x float> %868) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %877 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %874, <4 x float> %871) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %878 = shufflevector <8 x half> %846, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %879 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %878, <4 x float> %873) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %880 = shufflevector <8 x half> %846, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %880, <4 x float> %875) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %882 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %878, <4 x float> %876) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %880, <4 x float> %877) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %884 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %884, <4 x float> %879) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %886 = shufflevector <8 x half> %848, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %886, <4 x float> %881) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %884, <4 x float> %882) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %886, <4 x float> %883) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %890 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %890, <4 x float> %885) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %892 = shufflevector <8 x half> %850, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %892, <4 x float> %887) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %894 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %890, <4 x float> %888) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %892, <4 x float> %889) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %896 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %896, <4 x float> %891) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %898 = shufflevector <8 x half> %852, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %898, <4 x float> %893) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %896, <4 x float> %894) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %898, <4 x float> %895) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %902 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %902, <4 x float> %897) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %904 = shufflevector <8 x half> %854, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %904, <4 x float> %899) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %902, <4 x float> %900) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %904, <4 x float> %901) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %908 = bitcast <4 x i32> %855 to <8 x half>, !dbg !46
  %909 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %909, <4 x float> %903) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %911 = shufflevector <8 x half> %908, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %911, <4 x float> %905) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %909, <4 x float> %906) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %914 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %911, <4 x float> %907) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !39
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !39
  fence syncscope("workgroup") release, !dbg !39
  tail call void @llvm.amdgcn.s.barrier(), !dbg !39
  fence syncscope("workgroup") acquire, !dbg !39
  %915 = add nuw nsw i64 %117, 1, !dbg !12
  %k0.has.next.not = icmp eq i64 %117, 31, !dbg !46
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !46

k0.issue:                                         ; preds = %110
  %k0.next. = shl nuw nsw i64 %915, 14, !dbg !13
  %k0.next.1577 = add nuw nsw i64 %k0.next., %k0.prime.1567, !dbg !13
  %k0.next.kdma.source1120 = or disjoint i64 %k0.next.1577, %k0.prime.kdma.source1119, !dbg !13
  %k0.next.kdma.source1121 = trunc nuw nsw i64 %k0.next.kdma.source1120 to i32, !dbg !13
  %k0.next.kdma.source1122 = shl nuw nsw i32 %k0.next.kdma.source1121, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1122, %kdma.intra, !dbg !13
  %k0.next.1578 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %53, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1134 = or disjoint i64 %k0.next.1577, %k0.prime.kdma.source1133, !dbg !13
  %k0.next.kdma.source1135 = trunc nuw nsw i64 %k0.next.kdma.source1134 to i32, !dbg !13
  %k0.next.kdma.source1136 = shl nuw nsw i32 %k0.next.kdma.source1135, 1, !dbg !13
  %k0.next.kdma.offset1137 = or disjoint i32 %k0.next.kdma.source1136, %kdma.intra.hi, !dbg !13
  %k0.next.1579 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1138, i32 %k0.next.kdma.offset1137, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1151 = or disjoint i64 %k0.next.1577, %k0.prime.kdma.source1150, !dbg !13
  %k0.next.kdma.source1152 = trunc nuw nsw i64 %k0.next.kdma.source1151 to i32, !dbg !13
  %k0.next.kdma.source1153 = shl nuw nsw i32 %k0.next.kdma.source1152, 1, !dbg !13
  %k0.next.kdma.offset1154 = or disjoint i32 %k0.next.kdma.source1153, %kdma.intra, !dbg !13
  %k0.next.1580 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %56, i32 %k0.next.kdma.offset1154, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1168 = or disjoint i64 %k0.next.1577, %k0.prime.kdma.source1167, !dbg !13
  %k0.next.kdma.source1169 = trunc nuw nsw i64 %k0.next.kdma.source1168 to i32, !dbg !13
  %k0.next.kdma.source1170 = shl nuw nsw i32 %k0.next.kdma.source1169, 1, !dbg !13
  %k0.next.kdma.offset1171 = or disjoint i32 %k0.next.kdma.source1170, %kdma.intra.hi, !dbg !13
  %k0.next.1581 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1172, i32 %k0.next.kdma.offset1171, <4 x i32> %k0.prime.dma.buffer.rsrc989) #9, !dbg !13, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %110
  %916 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %917 = bitcast <4 x i32> %916 to <8 x half>, !dbg !46
  %918 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1065, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %919 = bitcast <4 x i32> %918 to <8 x half>, !dbg !46
  %920 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1066, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %921 = bitcast <4 x i32> %920 to <8 x half>, !dbg !46
  %922 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1067, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %923 = bitcast <4 x i32> %922 to <8 x half>, !dbg !46
  %924 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1068, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %925 = bitcast <4 x i32> %924 to <8 x half>, !dbg !46
  %926 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1069, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %927 = bitcast <4 x i32> %926 to <8 x half>, !dbg !46
  %928 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1070, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %929 = bitcast <4 x i32> %928 to <8 x half>, !dbg !46
  %930 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1071, i32 0) #11, !dbg !46, !frisk.v_double_buffer !47
  %931 = shufflevector <8 x half> %917, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  %932 = extractvalue [2 x <8 x float>] %111, 0, !dbg !44
  %933 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %934 = fmul <4 x float> %933, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %935 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %931, <4 x float> %934) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %936 = shufflevector <8 x half> %917, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  %937 = shufflevector <8 x float> %932, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %938 = fmul <4 x float> %937, %443, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %939 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %936, <4 x float> %938) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %940 = extractvalue [2 x <8 x float>] %111, 1, !dbg !44
  %941 = shufflevector <8 x float> %940, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !44
  %942 = fmul <4 x float> %941, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %943 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %931, <4 x float> %942) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !44
  %944 = shufflevector <8 x float> %940, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !44
  %945 = fmul <4 x float> %944, %451, !dbg !44
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %946 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed301, <4 x half> %936, <4 x float> %945) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %947 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %948 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %947, <4 x float> %935) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %949 = shufflevector <8 x half> %919, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %950 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed287, <4 x half> %949, <4 x float> %939) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %951 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %947, <4 x float> %943) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %952 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed303, <4 x half> %949, <4 x float> %946) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %953 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %954 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %953, <4 x float> %948) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %955 = shufflevector <8 x half> %921, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %956 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed289, <4 x half> %955, <4 x float> %950) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %953, <4 x float> %951) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %958 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed305, <4 x half> %955, <4 x float> %952) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %959 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %960 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %959, <4 x float> %954) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %961 = shufflevector <8 x half> %923, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %962 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed291, <4 x half> %961, <4 x float> %956) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %959, <4 x float> %957) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %964 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed307, <4 x half> %961, <4 x float> %958) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %965 = shufflevector <8 x half> %925, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %966 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %965, <4 x float> %960) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %967 = shufflevector <8 x half> %925, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %968 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed293, <4 x half> %967, <4 x float> %962) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %969 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %965, <4 x float> %963) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed309, <4 x half> %967, <4 x float> %964) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %971 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %971, <4 x float> %966) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %973 = shufflevector <8 x half> %927, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed295, <4 x half> %973, <4 x float> %968) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %971, <4 x float> %969) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed311, <4 x half> %973, <4 x float> %970) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %977 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %977, <4 x float> %972) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %979 = shufflevector <8 x half> %929, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed297, <4 x half> %979, <4 x float> %974) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %977, <4 x float> %975) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed313, <4 x half> %979, <4 x float> %976) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %983 = bitcast <4 x i32> %930 to <8 x half>, !dbg !46
  %984 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %985 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %984, <4 x float> %978) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %986 = shufflevector <8 x half> %983, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed299, <4 x half> %986, <4 x float> %980) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %984, <4 x float> %981) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !46
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed315, <4 x half> %986, <4 x float> %982) #8, !dbg !46, !frisk.pv_interleave !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12
  %990 = shufflevector <4 x float> %740, <4 x float> %742, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %991 = insertvalue [2 x <8 x float>] poison, <8 x float> %990, 0, !dbg !12
  %992 = shufflevector <4 x float> %761, <4 x float> %762, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %993 = insertvalue [2 x <8 x float>] %991, <8 x float> %992, 1, !dbg !12
  %994 = insertvalue [2 x <1 x float>] poison, <1 x float> %433, 0, !dbg !12
  %995 = insertvalue [2 x <1 x float>] %994, <1 x float> %436, 1, !dbg !12
  %996 = shufflevector <4 x float> %834, <4 x float> %836, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %997 = insertvalue [2 x <8 x float>] poison, <8 x float> %996, 0, !dbg !12
  %998 = shufflevector <4 x float> %837, <4 x float> %838, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %999 = insertvalue [2 x <8 x float>] %997, <8 x float> %998, 1, !dbg !12
  %1000 = shufflevector <4 x float> %910, <4 x float> %912, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1001 = insertvalue [2 x <8 x float>] poison, <8 x float> %1000, 0, !dbg !12
  %1002 = shufflevector <4 x float> %913, <4 x float> %914, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1003 = insertvalue [2 x <8 x float>] %1001, <8 x float> %1002, 1, !dbg !12
  %1004 = shufflevector <4 x float> %985, <4 x float> %987, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1005 = insertvalue [2 x <8 x float>] poison, <8 x float> %1004, 0, !dbg !12
  %1006 = shufflevector <4 x float> %988, <4 x float> %989, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !12
  %1007 = insertvalue [2 x <8 x float>] %1005, <8 x float> %1006, 1, !dbg !12
  %1008 = icmp samesign ult i64 %117, 31, !dbg !12
  br i1 %1008, label %110, label %1009, !dbg !12

1009:                                             ; preds = %k0.join
  %1010 = shl nuw nsw i32 %8, 2, !dbg !12
  %1011 = xor i32 %1010, 64, !dbg !12
  %1012 = bitcast float %k0.sum.anchor1566 to i32, !dbg !12
  %lane.sum.shuffle180 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1011, i32 %1012), !dbg !12
  %1013 = bitcast i32 %lane.sum.shuffle180 to float, !dbg !12
  %lane.sum.final181 = fadd float %k0.sum.anchor1566, %1013, !dbg !12
  %1014 = xor i32 %1010, 128, !dbg !12
  %1015 = bitcast float %lane.sum.final181 to i32, !dbg !12
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1015), !dbg !12
  %1016 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !12
  %lane.sum.final183 = fadd float %lane.sum.final181, %1016, !dbg !12
  %1017 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !12
  %1018 = bitcast float %k0.sum.anchor1558 to i32, !dbg !12
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1011, i32 %1018), !dbg !12
  %1019 = bitcast i32 %lane.sum.shuffle to float, !dbg !12
  %lane.sum.final = fadd float %k0.sum.anchor1558, %1019, !dbg !12
  %1020 = bitcast float %lane.sum.final to i32, !dbg !12
  %lane.sum.shuffle174 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1014, i32 %1020), !dbg !12
  %1021 = bitcast i32 %lane.sum.shuffle174 to float, !dbg !12
  %lane.sum.final175 = fadd float %lane.sum.final, %1021, !dbg !12
  %1022 = insertelement <1 x float> poison, float %lane.sum.final175, i64 0, !dbg !12
  %1023 = and i32 %kdma.tid, 16, !dbg !12
  %.not = icmp eq i32 %1023, 0, !dbg !12
  %1024 = and i32 %kdma.tid, 32, !dbg !12
  %.not28 = icmp eq i32 %1024, 0, !dbg !12
  %1025 = extractelement <4 x float> %985, i64 0, !dbg !12
  %1026 = extractelement <4 x float> %985, i64 1, !dbg !12
  %1027 = extractelement <4 x float> %985, i64 2, !dbg !12
  %1028 = extractelement <4 x float> %985, i64 3, !dbg !12
  %1029 = select i1 %.not, float %1026, float %1025, !dbg !12
  %1030 = bitcast float %1029 to i32, !dbg !12
  %1031 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1030), !dbg !12
  %1032 = bitcast i32 %1031 to float, !dbg !12
  %1033 = select i1 %.not, float %1025, float %1032, !dbg !12
  %1034 = select i1 %.not, float %1032, float %1026, !dbg !12
  %1035 = select i1 %.not, float %1028, float %1027, !dbg !12
  %1036 = bitcast float %1035 to i32, !dbg !12
  %1037 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1036), !dbg !12
  %1038 = bitcast i32 %1037 to float, !dbg !12
  %1039 = select i1 %.not, float %1027, float %1038, !dbg !12
  %1040 = select i1 %.not, float %1038, float %1028, !dbg !12
  %1041 = select i1 %.not28, float %1039, float %1033, !dbg !12
  %1042 = bitcast float %1041 to i32, !dbg !12
  %1043 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1042), !dbg !12
  %1044 = bitcast i32 %1043 to float, !dbg !12
  %1045 = select i1 %.not28, float %1040, float %1034, !dbg !12
  %1046 = bitcast float %1045 to i32, !dbg !12
  %1047 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1046), !dbg !12
  %1048 = bitcast i32 %1047 to float, !dbg !12
  %1049 = insertelement <4 x i1> poison, i1 %.not28, i64 0, !dbg !12
  %1050 = shufflevector <4 x i1> %1049, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !12
  %1051 = insertelement <4 x float> poison, float %1033, i64 0, !dbg !12
  %1052 = insertelement <4 x float> %1051, float %1034, i64 1, !dbg !12
  %1053 = insertelement <4 x float> %1052, float %1044, i64 2, !dbg !12
  %1054 = insertelement <4 x float> %1053, float %1048, i64 3, !dbg !12
  %1055 = insertelement <4 x float> poison, float %1044, i64 0, !dbg !12
  %1056 = insertelement <4 x float> %1055, float %1048, i64 1, !dbg !12
  %1057 = insertelement <4 x float> %1056, float %1039, i64 2, !dbg !12
  %1058 = insertelement <4 x float> %1057, float %1040, i64 3, !dbg !12
  %1059 = select <4 x i1> %1050, <4 x float> %1054, <4 x float> %1058, !dbg !12
  %1060 = extractelement <4 x float> %987, i64 0, !dbg !12
  %1061 = extractelement <4 x float> %987, i64 1, !dbg !12
  %1062 = extractelement <4 x float> %987, i64 2, !dbg !12
  %1063 = extractelement <4 x float> %987, i64 3, !dbg !12
  %1064 = select i1 %.not, float %1061, float %1060, !dbg !12
  %1065 = bitcast float %1064 to i32, !dbg !12
  %1066 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1065), !dbg !12
  %1067 = bitcast i32 %1066 to float, !dbg !12
  %1068 = select i1 %.not, float %1060, float %1067, !dbg !12
  %1069 = select i1 %.not, float %1067, float %1061, !dbg !12
  %1070 = select i1 %.not, float %1063, float %1062, !dbg !12
  %1071 = bitcast float %1070 to i32, !dbg !12
  %1072 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1071), !dbg !12
  %1073 = bitcast i32 %1072 to float, !dbg !12
  %1074 = select i1 %.not, float %1062, float %1073, !dbg !12
  %1075 = select i1 %.not, float %1073, float %1063, !dbg !12
  %1076 = select i1 %.not28, float %1074, float %1068, !dbg !12
  %1077 = bitcast float %1076 to i32, !dbg !12
  %1078 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1077), !dbg !12
  %1079 = bitcast i32 %1078 to float, !dbg !12
  %1080 = select i1 %.not28, float %1075, float %1069, !dbg !12
  %1081 = bitcast float %1080 to i32, !dbg !12
  %1082 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1081), !dbg !12
  %1083 = bitcast i32 %1082 to float, !dbg !12
  %1084 = insertelement <4 x float> poison, float %1068, i64 0, !dbg !12
  %1085 = insertelement <4 x float> %1084, float %1069, i64 1, !dbg !12
  %1086 = insertelement <4 x float> %1085, float %1079, i64 2, !dbg !12
  %1087 = insertelement <4 x float> %1086, float %1083, i64 3, !dbg !12
  %1088 = insertelement <4 x float> poison, float %1079, i64 0, !dbg !12
  %1089 = insertelement <4 x float> %1088, float %1083, i64 1, !dbg !12
  %1090 = insertelement <4 x float> %1089, float %1074, i64 2, !dbg !12
  %1091 = insertelement <4 x float> %1090, float %1075, i64 3, !dbg !12
  %1092 = select <4 x i1> %1050, <4 x float> %1087, <4 x float> %1091, !dbg !12
  %1093 = extractelement <4 x float> %988, i64 0, !dbg !12
  %1094 = extractelement <4 x float> %988, i64 1, !dbg !12
  %1095 = extractelement <4 x float> %988, i64 2, !dbg !12
  %1096 = extractelement <4 x float> %988, i64 3, !dbg !12
  %1097 = select i1 %.not, float %1094, float %1093, !dbg !12
  %1098 = bitcast float %1097 to i32, !dbg !12
  %1099 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1098), !dbg !12
  %1100 = bitcast i32 %1099 to float, !dbg !12
  %1101 = select i1 %.not, float %1093, float %1100, !dbg !12
  %1102 = select i1 %.not, float %1100, float %1094, !dbg !12
  %1103 = select i1 %.not, float %1096, float %1095, !dbg !12
  %1104 = bitcast float %1103 to i32, !dbg !12
  %1105 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1104), !dbg !12
  %1106 = bitcast i32 %1105 to float, !dbg !12
  %1107 = select i1 %.not, float %1095, float %1106, !dbg !12
  %1108 = select i1 %.not, float %1106, float %1096, !dbg !12
  %1109 = select i1 %.not28, float %1107, float %1101, !dbg !12
  %1110 = bitcast float %1109 to i32, !dbg !12
  %1111 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1110), !dbg !12
  %1112 = bitcast i32 %1111 to float, !dbg !12
  %1113 = select i1 %.not28, float %1108, float %1102, !dbg !12
  %1114 = bitcast float %1113 to i32, !dbg !12
  %1115 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1114), !dbg !12
  %1116 = bitcast i32 %1115 to float, !dbg !12
  %1117 = insertelement <4 x float> poison, float %1101, i64 0, !dbg !12
  %1118 = insertelement <4 x float> %1117, float %1102, i64 1, !dbg !12
  %1119 = insertelement <4 x float> %1118, float %1112, i64 2, !dbg !12
  %1120 = insertelement <4 x float> %1119, float %1116, i64 3, !dbg !12
  %1121 = insertelement <4 x float> poison, float %1112, i64 0, !dbg !12
  %1122 = insertelement <4 x float> %1121, float %1116, i64 1, !dbg !12
  %1123 = insertelement <4 x float> %1122, float %1107, i64 2, !dbg !12
  %1124 = insertelement <4 x float> %1123, float %1108, i64 3, !dbg !12
  %1125 = select <4 x i1> %1050, <4 x float> %1120, <4 x float> %1124, !dbg !12
  %1126 = extractelement <4 x float> %989, i64 0, !dbg !12
  %1127 = extractelement <4 x float> %989, i64 1, !dbg !12
  %1128 = extractelement <4 x float> %989, i64 2, !dbg !12
  %1129 = extractelement <4 x float> %989, i64 3, !dbg !12
  %1130 = select i1 %.not, float %1127, float %1126, !dbg !12
  %1131 = bitcast float %1130 to i32, !dbg !12
  %1132 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1131), !dbg !12
  %1133 = bitcast i32 %1132 to float, !dbg !12
  %1134 = select i1 %.not, float %1126, float %1133, !dbg !12
  %1135 = select i1 %.not, float %1133, float %1127, !dbg !12
  %1136 = select i1 %.not, float %1129, float %1128, !dbg !12
  %1137 = bitcast float %1136 to i32, !dbg !12
  %1138 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1137), !dbg !12
  %1139 = bitcast i32 %1138 to float, !dbg !12
  %1140 = select i1 %.not, float %1128, float %1139, !dbg !12
  %1141 = select i1 %.not, float %1139, float %1129, !dbg !12
  %1142 = select i1 %.not28, float %1140, float %1134, !dbg !12
  %1143 = bitcast float %1142 to i32, !dbg !12
  %1144 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1143), !dbg !12
  %1145 = bitcast i32 %1144 to float, !dbg !12
  %1146 = select i1 %.not28, float %1141, float %1135, !dbg !12
  %1147 = bitcast float %1146 to i32, !dbg !12
  %1148 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1147), !dbg !12
  %1149 = bitcast i32 %1148 to float, !dbg !12
  %1150 = insertelement <4 x float> poison, float %1134, i64 0, !dbg !12
  %1151 = insertelement <4 x float> %1150, float %1135, i64 1, !dbg !12
  %1152 = insertelement <4 x float> %1151, float %1145, i64 2, !dbg !12
  %1153 = insertelement <4 x float> %1152, float %1149, i64 3, !dbg !12
  %1154 = insertelement <4 x float> poison, float %1145, i64 0, !dbg !12
  %1155 = insertelement <4 x float> %1154, float %1149, i64 1, !dbg !12
  %1156 = insertelement <4 x float> %1155, float %1140, i64 2, !dbg !12
  %1157 = insertelement <4 x float> %1156, float %1141, i64 3, !dbg !12
  %1158 = select <4 x i1> %1050, <4 x float> %1153, <4 x float> %1157, !dbg !12
  %1159 = extractelement <4 x float> %910, i64 0, !dbg !12
  %1160 = extractelement <4 x float> %910, i64 1, !dbg !12
  %1161 = extractelement <4 x float> %910, i64 2, !dbg !12
  %1162 = extractelement <4 x float> %910, i64 3, !dbg !12
  %1163 = select i1 %.not, float %1160, float %1159, !dbg !12
  %1164 = bitcast float %1163 to i32, !dbg !12
  %1165 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1164), !dbg !12
  %1166 = bitcast i32 %1165 to float, !dbg !12
  %1167 = select i1 %.not, float %1159, float %1166, !dbg !12
  %1168 = select i1 %.not, float %1166, float %1160, !dbg !12
  %1169 = select i1 %.not, float %1162, float %1161, !dbg !12
  %1170 = bitcast float %1169 to i32, !dbg !12
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1170), !dbg !12
  %1172 = bitcast i32 %1171 to float, !dbg !12
  %1173 = select i1 %.not, float %1161, float %1172, !dbg !12
  %1174 = select i1 %.not, float %1172, float %1162, !dbg !12
  %1175 = select i1 %.not28, float %1173, float %1167, !dbg !12
  %1176 = bitcast float %1175 to i32, !dbg !12
  %1177 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1176), !dbg !12
  %1178 = bitcast i32 %1177 to float, !dbg !12
  %1179 = select i1 %.not28, float %1174, float %1168, !dbg !12
  %1180 = bitcast float %1179 to i32, !dbg !12
  %1181 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1180), !dbg !12
  %1182 = bitcast i32 %1181 to float, !dbg !12
  %1183 = insertelement <4 x float> poison, float %1167, i64 0, !dbg !12
  %1184 = insertelement <4 x float> %1183, float %1168, i64 1, !dbg !12
  %1185 = insertelement <4 x float> %1184, float %1178, i64 2, !dbg !12
  %1186 = insertelement <4 x float> %1185, float %1182, i64 3, !dbg !12
  %1187 = insertelement <4 x float> poison, float %1178, i64 0, !dbg !12
  %1188 = insertelement <4 x float> %1187, float %1182, i64 1, !dbg !12
  %1189 = insertelement <4 x float> %1188, float %1173, i64 2, !dbg !12
  %1190 = insertelement <4 x float> %1189, float %1174, i64 3, !dbg !12
  %1191 = select <4 x i1> %1050, <4 x float> %1186, <4 x float> %1190, !dbg !12
  %1192 = extractelement <4 x float> %912, i64 0, !dbg !12
  %1193 = extractelement <4 x float> %912, i64 1, !dbg !12
  %1194 = extractelement <4 x float> %912, i64 2, !dbg !12
  %1195 = extractelement <4 x float> %912, i64 3, !dbg !12
  %1196 = select i1 %.not, float %1193, float %1192, !dbg !12
  %1197 = bitcast float %1196 to i32, !dbg !12
  %1198 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1197), !dbg !12
  %1199 = bitcast i32 %1198 to float, !dbg !12
  %1200 = select i1 %.not, float %1192, float %1199, !dbg !12
  %1201 = select i1 %.not, float %1199, float %1193, !dbg !12
  %1202 = select i1 %.not, float %1195, float %1194, !dbg !12
  %1203 = bitcast float %1202 to i32, !dbg !12
  %1204 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1203), !dbg !12
  %1205 = bitcast i32 %1204 to float, !dbg !12
  %1206 = select i1 %.not, float %1194, float %1205, !dbg !12
  %1207 = select i1 %.not, float %1205, float %1195, !dbg !12
  %1208 = select i1 %.not28, float %1206, float %1200, !dbg !12
  %1209 = bitcast float %1208 to i32, !dbg !12
  %1210 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1209), !dbg !12
  %1211 = bitcast i32 %1210 to float, !dbg !12
  %1212 = select i1 %.not28, float %1207, float %1201, !dbg !12
  %1213 = bitcast float %1212 to i32, !dbg !12
  %1214 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1213), !dbg !12
  %1215 = bitcast i32 %1214 to float, !dbg !12
  %1216 = insertelement <4 x float> poison, float %1200, i64 0, !dbg !12
  %1217 = insertelement <4 x float> %1216, float %1201, i64 1, !dbg !12
  %1218 = insertelement <4 x float> %1217, float %1211, i64 2, !dbg !12
  %1219 = insertelement <4 x float> %1218, float %1215, i64 3, !dbg !12
  %1220 = insertelement <4 x float> poison, float %1211, i64 0, !dbg !12
  %1221 = insertelement <4 x float> %1220, float %1215, i64 1, !dbg !12
  %1222 = insertelement <4 x float> %1221, float %1206, i64 2, !dbg !12
  %1223 = insertelement <4 x float> %1222, float %1207, i64 3, !dbg !12
  %1224 = select <4 x i1> %1050, <4 x float> %1219, <4 x float> %1223, !dbg !12
  %1225 = extractelement <4 x float> %913, i64 0, !dbg !12
  %1226 = extractelement <4 x float> %913, i64 1, !dbg !12
  %1227 = extractelement <4 x float> %913, i64 2, !dbg !12
  %1228 = extractelement <4 x float> %913, i64 3, !dbg !12
  %1229 = select i1 %.not, float %1226, float %1225, !dbg !12
  %1230 = bitcast float %1229 to i32, !dbg !12
  %1231 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1230), !dbg !12
  %1232 = bitcast i32 %1231 to float, !dbg !12
  %1233 = select i1 %.not, float %1225, float %1232, !dbg !12
  %1234 = select i1 %.not, float %1232, float %1226, !dbg !12
  %1235 = select i1 %.not, float %1228, float %1227, !dbg !12
  %1236 = bitcast float %1235 to i32, !dbg !12
  %1237 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1236), !dbg !12
  %1238 = bitcast i32 %1237 to float, !dbg !12
  %1239 = select i1 %.not, float %1227, float %1238, !dbg !12
  %1240 = select i1 %.not, float %1238, float %1228, !dbg !12
  %1241 = select i1 %.not28, float %1239, float %1233, !dbg !12
  %1242 = bitcast float %1241 to i32, !dbg !12
  %1243 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1242), !dbg !12
  %1244 = bitcast i32 %1243 to float, !dbg !12
  %1245 = select i1 %.not28, float %1240, float %1234, !dbg !12
  %1246 = bitcast float %1245 to i32, !dbg !12
  %1247 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1246), !dbg !12
  %1248 = bitcast i32 %1247 to float, !dbg !12
  %1249 = insertelement <4 x float> poison, float %1233, i64 0, !dbg !12
  %1250 = insertelement <4 x float> %1249, float %1234, i64 1, !dbg !12
  %1251 = insertelement <4 x float> %1250, float %1244, i64 2, !dbg !12
  %1252 = insertelement <4 x float> %1251, float %1248, i64 3, !dbg !12
  %1253 = insertelement <4 x float> poison, float %1244, i64 0, !dbg !12
  %1254 = insertelement <4 x float> %1253, float %1248, i64 1, !dbg !12
  %1255 = insertelement <4 x float> %1254, float %1239, i64 2, !dbg !12
  %1256 = insertelement <4 x float> %1255, float %1240, i64 3, !dbg !12
  %1257 = select <4 x i1> %1050, <4 x float> %1252, <4 x float> %1256, !dbg !12
  %1258 = extractelement <4 x float> %914, i64 0, !dbg !12
  %1259 = extractelement <4 x float> %914, i64 1, !dbg !12
  %1260 = extractelement <4 x float> %914, i64 2, !dbg !12
  %1261 = extractelement <4 x float> %914, i64 3, !dbg !12
  %1262 = select i1 %.not, float %1259, float %1258, !dbg !12
  %1263 = bitcast float %1262 to i32, !dbg !12
  %1264 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1263), !dbg !12
  %1265 = bitcast i32 %1264 to float, !dbg !12
  %1266 = select i1 %.not, float %1258, float %1265, !dbg !12
  %1267 = select i1 %.not, float %1265, float %1259, !dbg !12
  %1268 = select i1 %.not, float %1261, float %1260, !dbg !12
  %1269 = bitcast float %1268 to i32, !dbg !12
  %1270 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1269), !dbg !12
  %1271 = bitcast i32 %1270 to float, !dbg !12
  %1272 = select i1 %.not, float %1260, float %1271, !dbg !12
  %1273 = select i1 %.not, float %1271, float %1261, !dbg !12
  %1274 = select i1 %.not28, float %1272, float %1266, !dbg !12
  %1275 = bitcast float %1274 to i32, !dbg !12
  %1276 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1275), !dbg !12
  %1277 = bitcast i32 %1276 to float, !dbg !12
  %1278 = select i1 %.not28, float %1273, float %1267, !dbg !12
  %1279 = bitcast float %1278 to i32, !dbg !12
  %1280 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1279), !dbg !12
  %1281 = bitcast i32 %1280 to float, !dbg !12
  %1282 = insertelement <4 x float> poison, float %1266, i64 0, !dbg !12
  %1283 = insertelement <4 x float> %1282, float %1267, i64 1, !dbg !12
  %1284 = insertelement <4 x float> %1283, float %1277, i64 2, !dbg !12
  %1285 = insertelement <4 x float> %1284, float %1281, i64 3, !dbg !12
  %1286 = insertelement <4 x float> poison, float %1277, i64 0, !dbg !12
  %1287 = insertelement <4 x float> %1286, float %1281, i64 1, !dbg !12
  %1288 = insertelement <4 x float> %1287, float %1272, i64 2, !dbg !12
  %1289 = insertelement <4 x float> %1288, float %1273, i64 3, !dbg !12
  %1290 = select <4 x i1> %1050, <4 x float> %1285, <4 x float> %1289, !dbg !12
  %1291 = extractelement <4 x float> %834, i64 0, !dbg !12
  %1292 = extractelement <4 x float> %834, i64 1, !dbg !12
  %1293 = extractelement <4 x float> %834, i64 2, !dbg !12
  %1294 = extractelement <4 x float> %834, i64 3, !dbg !12
  %1295 = select i1 %.not, float %1292, float %1291, !dbg !12
  %1296 = bitcast float %1295 to i32, !dbg !12
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1296), !dbg !12
  %1298 = bitcast i32 %1297 to float, !dbg !12
  %1299 = select i1 %.not, float %1291, float %1298, !dbg !12
  %1300 = select i1 %.not, float %1298, float %1292, !dbg !12
  %1301 = select i1 %.not, float %1294, float %1293, !dbg !12
  %1302 = bitcast float %1301 to i32, !dbg !12
  %1303 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1302), !dbg !12
  %1304 = bitcast i32 %1303 to float, !dbg !12
  %1305 = select i1 %.not, float %1293, float %1304, !dbg !12
  %1306 = select i1 %.not, float %1304, float %1294, !dbg !12
  %1307 = select i1 %.not28, float %1305, float %1299, !dbg !12
  %1308 = bitcast float %1307 to i32, !dbg !12
  %1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1308), !dbg !12
  %1310 = bitcast i32 %1309 to float, !dbg !12
  %1311 = select i1 %.not28, float %1306, float %1300, !dbg !12
  %1312 = bitcast float %1311 to i32, !dbg !12
  %1313 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1312), !dbg !12
  %1314 = bitcast i32 %1313 to float, !dbg !12
  %1315 = insertelement <4 x float> poison, float %1299, i64 0, !dbg !12
  %1316 = insertelement <4 x float> %1315, float %1300, i64 1, !dbg !12
  %1317 = insertelement <4 x float> %1316, float %1310, i64 2, !dbg !12
  %1318 = insertelement <4 x float> %1317, float %1314, i64 3, !dbg !12
  %1319 = insertelement <4 x float> poison, float %1310, i64 0, !dbg !12
  %1320 = insertelement <4 x float> %1319, float %1314, i64 1, !dbg !12
  %1321 = insertelement <4 x float> %1320, float %1305, i64 2, !dbg !12
  %1322 = insertelement <4 x float> %1321, float %1306, i64 3, !dbg !12
  %1323 = select <4 x i1> %1050, <4 x float> %1318, <4 x float> %1322, !dbg !12
  %1324 = extractelement <4 x float> %836, i64 0, !dbg !12
  %1325 = extractelement <4 x float> %836, i64 1, !dbg !12
  %1326 = extractelement <4 x float> %836, i64 2, !dbg !12
  %1327 = extractelement <4 x float> %836, i64 3, !dbg !12
  %1328 = select i1 %.not, float %1325, float %1324, !dbg !12
  %1329 = bitcast float %1328 to i32, !dbg !12
  %1330 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1329), !dbg !12
  %1331 = bitcast i32 %1330 to float, !dbg !12
  %1332 = select i1 %.not, float %1324, float %1331, !dbg !12
  %1333 = select i1 %.not, float %1331, float %1325, !dbg !12
  %1334 = select i1 %.not, float %1327, float %1326, !dbg !12
  %1335 = bitcast float %1334 to i32, !dbg !12
  %1336 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1335), !dbg !12
  %1337 = bitcast i32 %1336 to float, !dbg !12
  %1338 = select i1 %.not, float %1326, float %1337, !dbg !12
  %1339 = select i1 %.not, float %1337, float %1327, !dbg !12
  %1340 = select i1 %.not28, float %1338, float %1332, !dbg !12
  %1341 = bitcast float %1340 to i32, !dbg !12
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1341), !dbg !12
  %1343 = bitcast i32 %1342 to float, !dbg !12
  %1344 = select i1 %.not28, float %1339, float %1333, !dbg !12
  %1345 = bitcast float %1344 to i32, !dbg !12
  %1346 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1345), !dbg !12
  %1347 = bitcast i32 %1346 to float, !dbg !12
  %1348 = insertelement <4 x float> poison, float %1332, i64 0, !dbg !12
  %1349 = insertelement <4 x float> %1348, float %1333, i64 1, !dbg !12
  %1350 = insertelement <4 x float> %1349, float %1343, i64 2, !dbg !12
  %1351 = insertelement <4 x float> %1350, float %1347, i64 3, !dbg !12
  %1352 = insertelement <4 x float> poison, float %1343, i64 0, !dbg !12
  %1353 = insertelement <4 x float> %1352, float %1347, i64 1, !dbg !12
  %1354 = insertelement <4 x float> %1353, float %1338, i64 2, !dbg !12
  %1355 = insertelement <4 x float> %1354, float %1339, i64 3, !dbg !12
  %1356 = select <4 x i1> %1050, <4 x float> %1351, <4 x float> %1355, !dbg !12
  %1357 = extractelement <4 x float> %837, i64 0, !dbg !12
  %1358 = extractelement <4 x float> %837, i64 1, !dbg !12
  %1359 = extractelement <4 x float> %837, i64 2, !dbg !12
  %1360 = extractelement <4 x float> %837, i64 3, !dbg !12
  %1361 = select i1 %.not, float %1358, float %1357, !dbg !12
  %1362 = bitcast float %1361 to i32, !dbg !12
  %1363 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1362), !dbg !12
  %1364 = bitcast i32 %1363 to float, !dbg !12
  %1365 = select i1 %.not, float %1357, float %1364, !dbg !12
  %1366 = select i1 %.not, float %1364, float %1358, !dbg !12
  %1367 = select i1 %.not, float %1360, float %1359, !dbg !12
  %1368 = bitcast float %1367 to i32, !dbg !12
  %1369 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1368), !dbg !12
  %1370 = bitcast i32 %1369 to float, !dbg !12
  %1371 = select i1 %.not, float %1359, float %1370, !dbg !12
  %1372 = select i1 %.not, float %1370, float %1360, !dbg !12
  %1373 = select i1 %.not28, float %1371, float %1365, !dbg !12
  %1374 = bitcast float %1373 to i32, !dbg !12
  %1375 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1374), !dbg !12
  %1376 = bitcast i32 %1375 to float, !dbg !12
  %1377 = select i1 %.not28, float %1372, float %1366, !dbg !12
  %1378 = bitcast float %1377 to i32, !dbg !12
  %1379 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1378), !dbg !12
  %1380 = bitcast i32 %1379 to float, !dbg !12
  %1381 = insertelement <4 x float> poison, float %1365, i64 0, !dbg !12
  %1382 = insertelement <4 x float> %1381, float %1366, i64 1, !dbg !12
  %1383 = insertelement <4 x float> %1382, float %1376, i64 2, !dbg !12
  %1384 = insertelement <4 x float> %1383, float %1380, i64 3, !dbg !12
  %1385 = insertelement <4 x float> poison, float %1376, i64 0, !dbg !12
  %1386 = insertelement <4 x float> %1385, float %1380, i64 1, !dbg !12
  %1387 = insertelement <4 x float> %1386, float %1371, i64 2, !dbg !12
  %1388 = insertelement <4 x float> %1387, float %1372, i64 3, !dbg !12
  %1389 = select <4 x i1> %1050, <4 x float> %1384, <4 x float> %1388, !dbg !12
  %1390 = extractelement <4 x float> %838, i64 0, !dbg !12
  %1391 = extractelement <4 x float> %838, i64 1, !dbg !12
  %1392 = extractelement <4 x float> %838, i64 2, !dbg !12
  %1393 = extractelement <4 x float> %838, i64 3, !dbg !12
  %1394 = select i1 %.not, float %1391, float %1390, !dbg !12
  %1395 = bitcast float %1394 to i32, !dbg !12
  %1396 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1395), !dbg !12
  %1397 = bitcast i32 %1396 to float, !dbg !12
  %1398 = select i1 %.not, float %1390, float %1397, !dbg !12
  %1399 = select i1 %.not, float %1397, float %1391, !dbg !12
  %1400 = select i1 %.not, float %1393, float %1392, !dbg !12
  %1401 = bitcast float %1400 to i32, !dbg !12
  %1402 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1401), !dbg !12
  %1403 = bitcast i32 %1402 to float, !dbg !12
  %1404 = select i1 %.not, float %1392, float %1403, !dbg !12
  %1405 = select i1 %.not, float %1403, float %1393, !dbg !12
  %1406 = select i1 %.not28, float %1404, float %1398, !dbg !12
  %1407 = bitcast float %1406 to i32, !dbg !12
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1407), !dbg !12
  %1409 = bitcast i32 %1408 to float, !dbg !12
  %1410 = select i1 %.not28, float %1405, float %1399, !dbg !12
  %1411 = bitcast float %1410 to i32, !dbg !12
  %1412 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1411), !dbg !12
  %1413 = bitcast i32 %1412 to float, !dbg !12
  %1414 = insertelement <4 x float> poison, float %1398, i64 0, !dbg !12
  %1415 = insertelement <4 x float> %1414, float %1399, i64 1, !dbg !12
  %1416 = insertelement <4 x float> %1415, float %1409, i64 2, !dbg !12
  %1417 = insertelement <4 x float> %1416, float %1413, i64 3, !dbg !12
  %1418 = insertelement <4 x float> poison, float %1409, i64 0, !dbg !12
  %1419 = insertelement <4 x float> %1418, float %1413, i64 1, !dbg !12
  %1420 = insertelement <4 x float> %1419, float %1404, i64 2, !dbg !12
  %1421 = insertelement <4 x float> %1420, float %1405, i64 3, !dbg !12
  %1422 = select <4 x i1> %1050, <4 x float> %1417, <4 x float> %1421, !dbg !12
  %1423 = extractelement <4 x float> %740, i64 0, !dbg !12
  %1424 = extractelement <4 x float> %740, i64 1, !dbg !12
  %1425 = extractelement <4 x float> %740, i64 2, !dbg !12
  %1426 = extractelement <4 x float> %740, i64 3, !dbg !12
  %1427 = select i1 %.not, float %1424, float %1423, !dbg !12
  %1428 = bitcast float %1427 to i32, !dbg !12
  %1429 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1428), !dbg !12
  %1430 = bitcast i32 %1429 to float, !dbg !12
  %1431 = select i1 %.not, float %1423, float %1430, !dbg !12
  %1432 = select i1 %.not, float %1430, float %1424, !dbg !12
  %1433 = select i1 %.not, float %1426, float %1425, !dbg !12
  %1434 = bitcast float %1433 to i32, !dbg !12
  %1435 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1434), !dbg !12
  %1436 = bitcast i32 %1435 to float, !dbg !12
  %1437 = select i1 %.not, float %1425, float %1436, !dbg !12
  %1438 = select i1 %.not, float %1436, float %1426, !dbg !12
  %1439 = select i1 %.not28, float %1437, float %1431, !dbg !12
  %1440 = bitcast float %1439 to i32, !dbg !12
  %1441 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1440), !dbg !12
  %1442 = select i1 %.not28, float %1438, float %1432, !dbg !12
  %1443 = bitcast float %1442 to i32, !dbg !12
  %1444 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1443), !dbg !12
  %1445 = extractelement <4 x float> %742, i64 0, !dbg !12
  %1446 = extractelement <4 x float> %742, i64 1, !dbg !12
  %1447 = extractelement <4 x float> %742, i64 2, !dbg !12
  %1448 = extractelement <4 x float> %742, i64 3, !dbg !12
  %1449 = select i1 %.not, float %1446, float %1445, !dbg !12
  %1450 = bitcast float %1449 to i32, !dbg !12
  %1451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1450), !dbg !12
  %1452 = bitcast i32 %1451 to float, !dbg !12
  %1453 = select i1 %.not, float %1445, float %1452, !dbg !12
  %1454 = select i1 %.not, float %1452, float %1446, !dbg !12
  %1455 = select i1 %.not, float %1448, float %1447, !dbg !12
  %1456 = bitcast float %1455 to i32, !dbg !12
  %1457 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1456), !dbg !12
  %1458 = bitcast i32 %1457 to float, !dbg !12
  %1459 = select i1 %.not, float %1447, float %1458, !dbg !12
  %1460 = select i1 %.not, float %1458, float %1448, !dbg !12
  %1461 = select i1 %.not28, float %1459, float %1453, !dbg !12
  %1462 = bitcast float %1461 to i32, !dbg !12
  %1463 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1462), !dbg !12
  %1464 = select i1 %.not28, float %1460, float %1454, !dbg !12
  %1465 = bitcast float %1464 to i32, !dbg !12
  %1466 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1465), !dbg !12
  %1467 = extractelement <4 x float> %761, i64 0, !dbg !12
  %1468 = extractelement <4 x float> %761, i64 1, !dbg !12
  %1469 = extractelement <4 x float> %761, i64 2, !dbg !12
  %1470 = extractelement <4 x float> %761, i64 3, !dbg !12
  %1471 = select i1 %.not, float %1468, float %1467, !dbg !12
  %1472 = bitcast float %1471 to i32, !dbg !12
  %1473 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1472), !dbg !12
  %1474 = bitcast i32 %1473 to float, !dbg !12
  %1475 = select i1 %.not, float %1467, float %1474, !dbg !12
  %1476 = select i1 %.not, float %1474, float %1468, !dbg !12
  %1477 = select i1 %.not, float %1470, float %1469, !dbg !12
  %1478 = bitcast float %1477 to i32, !dbg !12
  %1479 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1478), !dbg !12
  %1480 = bitcast i32 %1479 to float, !dbg !12
  %1481 = select i1 %.not, float %1469, float %1480, !dbg !12
  %1482 = select i1 %.not, float %1480, float %1470, !dbg !12
  %1483 = select i1 %.not28, float %1481, float %1475, !dbg !12
  %1484 = bitcast float %1483 to i32, !dbg !12
  %1485 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1484), !dbg !12
  %1486 = select i1 %.not28, float %1482, float %1476, !dbg !12
  %1487 = bitcast float %1486 to i32, !dbg !12
  %1488 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1487), !dbg !12
  %1489 = extractelement <4 x float> %762, i64 0, !dbg !12
  %1490 = extractelement <4 x float> %762, i64 1, !dbg !12
  %1491 = extractelement <4 x float> %762, i64 2, !dbg !12
  %1492 = extractelement <4 x float> %762, i64 3, !dbg !12
  %1493 = select i1 %.not, float %1490, float %1489, !dbg !12
  %1494 = bitcast float %1493 to i32, !dbg !12
  %1495 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1494), !dbg !12
  %1496 = bitcast i32 %1495 to float, !dbg !12
  %1497 = select i1 %.not, float %1489, float %1496, !dbg !12
  %1498 = select i1 %.not, float %1496, float %1490, !dbg !12
  %1499 = select i1 %.not, float %1492, float %1491, !dbg !12
  %1500 = bitcast float %1499 to i32, !dbg !12
  %1501 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %14, i32 %1500), !dbg !12
  %1502 = bitcast i32 %1501 to float, !dbg !12
  %1503 = select i1 %.not, float %1491, float %1502, !dbg !12
  %1504 = select i1 %.not, float %1502, float %1492, !dbg !12
  %1505 = select i1 %.not28, float %1503, float %1497, !dbg !12
  %1506 = bitcast float %1505 to i32, !dbg !12
  %1507 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1506), !dbg !12
  %1508 = select i1 %.not28, float %1504, float %1498, !dbg !12
  %1509 = bitcast float %1508 to i32, !dbg !12
  %1510 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %18, i32 %1509), !dbg !12
  %1511 = shufflevector <1 x float> %1022, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1512 = shufflevector <1 x float> %1017, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !51
  %1513 = bitcast i32 %1507 to float, !dbg !12
  %1514 = bitcast i32 %1510 to float, !dbg !12
  %1515 = insertelement <4 x float> poison, float %1497, i64 0, !dbg !12
  %1516 = insertelement <4 x float> %1515, float %1498, i64 1, !dbg !12
  %1517 = insertelement <4 x float> %1516, float %1513, i64 2, !dbg !12
  %1518 = insertelement <4 x float> %1517, float %1514, i64 3, !dbg !12
  %1519 = insertelement <4 x float> poison, float %1513, i64 0, !dbg !12
  %1520 = insertelement <4 x float> %1519, float %1514, i64 1, !dbg !12
  %1521 = insertelement <4 x float> %1520, float %1503, i64 2, !dbg !12
  %1522 = insertelement <4 x float> %1521, float %1504, i64 3, !dbg !12
  %1523 = select <4 x i1> %1050, <4 x float> %1518, <4 x float> %1522, !dbg !12
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1512, !dbg !51
  %normalized = fmul <4 x float> %row.reciprocal, %1523, !dbg !51
  %1524 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !52
  %1525 = extractelement <4 x half> %1524, i64 3, !dbg !52
  %1526 = extractelement <4 x half> %1524, i64 2, !dbg !52
  %1527 = extractelement <4 x half> %1524, i64 1, !dbg !52
  %1528 = extractelement <4 x half> %1524, i64 0, !dbg !52
  %1529 = bitcast i32 %1485 to float, !dbg !12
  %1530 = bitcast i32 %1488 to float, !dbg !12
  %1531 = insertelement <4 x float> poison, float %1475, i64 0, !dbg !12
  %1532 = insertelement <4 x float> %1531, float %1476, i64 1, !dbg !12
  %1533 = insertelement <4 x float> %1532, float %1529, i64 2, !dbg !12
  %1534 = insertelement <4 x float> %1533, float %1530, i64 3, !dbg !12
  %1535 = insertelement <4 x float> poison, float %1529, i64 0, !dbg !12
  %1536 = insertelement <4 x float> %1535, float %1530, i64 1, !dbg !12
  %1537 = insertelement <4 x float> %1536, float %1481, i64 2, !dbg !12
  %1538 = insertelement <4 x float> %1537, float %1482, i64 3, !dbg !12
  %1539 = select <4 x i1> %1050, <4 x float> %1534, <4 x float> %1538, !dbg !12
  %normalized29 = fmul <4 x float> %row.reciprocal, %1539, !dbg !51
  %1540 = fptrunc <4 x float> %normalized29 to <4 x half>, !dbg !52
  %1541 = extractelement <4 x half> %1540, i64 3, !dbg !52
  %1542 = extractelement <4 x half> %1540, i64 2, !dbg !52
  %1543 = extractelement <4 x half> %1540, i64 1, !dbg !52
  %1544 = extractelement <4 x half> %1540, i64 0, !dbg !52
  %1545 = bitcast i32 %1463 to float, !dbg !12
  %1546 = bitcast i32 %1466 to float, !dbg !12
  %1547 = insertelement <4 x float> poison, float %1453, i64 0, !dbg !12
  %1548 = insertelement <4 x float> %1547, float %1454, i64 1, !dbg !12
  %1549 = insertelement <4 x float> %1548, float %1545, i64 2, !dbg !12
  %1550 = insertelement <4 x float> %1549, float %1546, i64 3, !dbg !12
  %1551 = insertelement <4 x float> poison, float %1545, i64 0, !dbg !12
  %1552 = insertelement <4 x float> %1551, float %1546, i64 1, !dbg !12
  %1553 = insertelement <4 x float> %1552, float %1459, i64 2, !dbg !12
  %1554 = insertelement <4 x float> %1553, float %1460, i64 3, !dbg !12
  %1555 = select <4 x i1> %1050, <4 x float> %1550, <4 x float> %1554, !dbg !12
  %row.reciprocal30 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1511, !dbg !51
  %normalized31 = fmul <4 x float> %row.reciprocal30, %1555, !dbg !51
  %1556 = fptrunc <4 x float> %normalized31 to <4 x half>, !dbg !52
  %1557 = extractelement <4 x half> %1556, i64 3, !dbg !52
  %1558 = extractelement <4 x half> %1556, i64 2, !dbg !52
  %1559 = extractelement <4 x half> %1556, i64 1, !dbg !52
  %1560 = extractelement <4 x half> %1556, i64 0, !dbg !52
  %1561 = bitcast i32 %1441 to float, !dbg !12
  %1562 = bitcast i32 %1444 to float, !dbg !12
  %1563 = insertelement <4 x float> poison, float %1431, i64 0, !dbg !12
  %1564 = insertelement <4 x float> %1563, float %1432, i64 1, !dbg !12
  %1565 = insertelement <4 x float> %1564, float %1561, i64 2, !dbg !12
  %1566 = insertelement <4 x float> %1565, float %1562, i64 3, !dbg !12
  %1567 = insertelement <4 x float> poison, float %1561, i64 0, !dbg !12
  %1568 = insertelement <4 x float> %1567, float %1562, i64 1, !dbg !12
  %1569 = insertelement <4 x float> %1568, float %1437, i64 2, !dbg !12
  %1570 = insertelement <4 x float> %1569, float %1438, i64 3, !dbg !12
  %1571 = select <4 x i1> %1050, <4 x float> %1566, <4 x float> %1570, !dbg !12
  %normalized32 = fmul <4 x float> %row.reciprocal30, %1571, !dbg !51
  %1572 = fptrunc <4 x float> %normalized32 to <4 x half>, !dbg !52
  %1573 = extractelement <4 x half> %1572, i64 3, !dbg !52
  %1574 = extractelement <4 x half> %1572, i64 2, !dbg !52
  %1575 = extractelement <4 x half> %1572, i64 1, !dbg !52
  %1576 = extractelement <4 x half> %1572, i64 0, !dbg !52
  %address.thread604 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !53, !range !10, !frisk.register_lifetimes !5
  %address.local605 = zext nneg i32 %address.thread604 to i64, !dbg !9
  %1577 = lshr i64 %address.local605, 2, !dbg !53
  %address.local608 = and i64 %1577, 48, !dbg !53
  %address.local610 = and i64 %address.local605, 240, !dbg !53
  %address.local609 = add i64 %34, %address.local605, !dbg !53
  %address.local611 = sub i64 %address.local609, %address.local610, !dbg !53
  %address.local612 = add i64 %address.local611, %address.local608, !dbg !53
  %1578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %42, !dbg !53
  %address.local615.idx = shl nuw nsw i64 %address.local612, 8, !dbg !53
  %address.local615 = getelementptr inbounds nuw i8, ptr addrspace(1) %1578, i64 %address.local615.idx, !dbg !53
  %address.local617 = and i64 %1577, 12, !dbg !11
  %address.local618 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local617, !dbg !53
  store half %1576, ptr addrspace(1) %address.local618, align 2, !dbg !53
  %address.local620 = or disjoint i64 %address.local617, 1, !dbg !53
  %address.local621 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local620, !dbg !53
  store half %1575, ptr addrspace(1) %address.local621, align 2, !dbg !53
  %address.local623 = or disjoint i64 %address.local617, 2, !dbg !53
  %address.local624 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local623, !dbg !53
  store half %1574, ptr addrspace(1) %address.local624, align 2, !dbg !53
  %address.local626 = or disjoint i64 %address.local617, 3, !dbg !53
  %address.local627 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local626, !dbg !53
  store half %1573, ptr addrspace(1) %address.local627, align 2, !dbg !53
  %address.local629 = or disjoint i64 %address.local617, 16, !dbg !53
  %address.local630 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local629, !dbg !53
  store half %1560, ptr addrspace(1) %address.local630, align 2, !dbg !53
  %address.local632 = or disjoint i64 %address.local617, 17, !dbg !53
  %address.local633 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local632, !dbg !53
  store half %1559, ptr addrspace(1) %address.local633, align 2, !dbg !53
  %address.local635 = or disjoint i64 %address.local617, 18, !dbg !53
  %address.local636 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local635, !dbg !53
  store half %1558, ptr addrspace(1) %address.local636, align 2, !dbg !53
  %address.local638 = or disjoint i64 %address.local617, 19, !dbg !53
  %address.local639 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local638, !dbg !53
  store half %1557, ptr addrspace(1) %address.local639, align 2, !dbg !53
  %address.local643 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local615, i64 16384, !dbg !53
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local617, !dbg !53
  store half %1544, ptr addrspace(1) %address.local644, align 2, !dbg !53
  %address.local646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local620, !dbg !53
  store half %1543, ptr addrspace(1) %address.local646, align 2, !dbg !53
  %address.local648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local623, !dbg !53
  store half %1542, ptr addrspace(1) %address.local648, align 2, !dbg !53
  %address.local650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local626, !dbg !53
  store half %1541, ptr addrspace(1) %address.local650, align 2, !dbg !53
  %address.local652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local629, !dbg !53
  store half %1528, ptr addrspace(1) %address.local652, align 2, !dbg !53
  %address.local654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local632, !dbg !53
  store half %1527, ptr addrspace(1) %address.local654, align 2, !dbg !53
  %address.local656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local635, !dbg !53
  store half %1526, ptr addrspace(1) %address.local656, align 2, !dbg !53
  %address.local658 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local638, !dbg !53
  store half %1525, ptr addrspace(1) %address.local658, align 2, !dbg !53
  %normalized34 = fmul <4 x float> %row.reciprocal, %1422, !dbg !51
  %1579 = fptrunc <4 x float> %normalized34 to <4 x half>, !dbg !52
  %1580 = extractelement <4 x half> %1579, i64 3, !dbg !52
  %1581 = extractelement <4 x half> %1579, i64 2, !dbg !52
  %1582 = extractelement <4 x half> %1579, i64 1, !dbg !52
  %1583 = extractelement <4 x half> %1579, i64 0, !dbg !52
  %normalized35 = fmul <4 x float> %row.reciprocal, %1389, !dbg !51
  %1584 = fptrunc <4 x float> %normalized35 to <4 x half>, !dbg !52
  %1585 = extractelement <4 x half> %1584, i64 3, !dbg !52
  %1586 = extractelement <4 x half> %1584, i64 2, !dbg !52
  %1587 = extractelement <4 x half> %1584, i64 1, !dbg !52
  %1588 = extractelement <4 x half> %1584, i64 0, !dbg !52
  %normalized37 = fmul <4 x float> %row.reciprocal30, %1356, !dbg !51
  %1589 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !52
  %1590 = extractelement <4 x half> %1589, i64 3, !dbg !52
  %1591 = extractelement <4 x half> %1589, i64 2, !dbg !52
  %1592 = extractelement <4 x half> %1589, i64 1, !dbg !52
  %1593 = extractelement <4 x half> %1589, i64 0, !dbg !52
  %normalized38 = fmul <4 x float> %row.reciprocal30, %1323, !dbg !51
  %1594 = fptrunc <4 x float> %normalized38 to <4 x half>, !dbg !52
  %1595 = extractelement <4 x half> %1594, i64 3, !dbg !52
  %1596 = extractelement <4 x half> %1594, i64 2, !dbg !52
  %1597 = extractelement <4 x half> %1594, i64 1, !dbg !52
  %1598 = extractelement <4 x half> %1594, i64 0, !dbg !52
  %address.local673 = or disjoint i64 %address.local617, 32, !dbg !53
  %address.local674 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local673, !dbg !53
  store half %1598, ptr addrspace(1) %address.local674, align 2, !dbg !53
  %address.local676 = or disjoint i64 %address.local617, 33, !dbg !53
  %address.local677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local676, !dbg !53
  store half %1597, ptr addrspace(1) %address.local677, align 2, !dbg !53
  %address.local679 = or disjoint i64 %address.local617, 34, !dbg !53
  %address.local680 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local679, !dbg !53
  store half %1596, ptr addrspace(1) %address.local680, align 2, !dbg !53
  %address.local682 = or disjoint i64 %address.local617, 35, !dbg !53
  %address.local683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local682, !dbg !53
  store half %1595, ptr addrspace(1) %address.local683, align 2, !dbg !53
  %address.local685 = or disjoint i64 %address.local617, 48, !dbg !53
  %address.local686 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local685, !dbg !53
  store half %1593, ptr addrspace(1) %address.local686, align 2, !dbg !53
  %address.local688 = or disjoint i64 %address.local617, 49, !dbg !53
  %address.local689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local688, !dbg !53
  store half %1592, ptr addrspace(1) %address.local689, align 2, !dbg !53
  %address.local691 = or disjoint i64 %address.local617, 50, !dbg !53
  %address.local692 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local691, !dbg !53
  store half %1591, ptr addrspace(1) %address.local692, align 2, !dbg !53
  %address.local694 = or i64 %1577, 51, !dbg !53
  %address.local695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local694, !dbg !53
  store half %1590, ptr addrspace(1) %address.local695, align 2, !dbg !53
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local673, !dbg !53
  store half %1588, ptr addrspace(1) %address.local700, align 2, !dbg !53
  %address.local702 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local676, !dbg !53
  store half %1587, ptr addrspace(1) %address.local702, align 2, !dbg !53
  %address.local704 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local679, !dbg !53
  store half %1586, ptr addrspace(1) %address.local704, align 2, !dbg !53
  %address.local706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local682, !dbg !53
  store half %1585, ptr addrspace(1) %address.local706, align 2, !dbg !53
  %address.local708 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local685, !dbg !53
  store half %1583, ptr addrspace(1) %address.local708, align 2, !dbg !53
  %address.local710 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local688, !dbg !53
  store half %1582, ptr addrspace(1) %address.local710, align 2, !dbg !53
  %address.local712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local691, !dbg !53
  store half %1581, ptr addrspace(1) %address.local712, align 2, !dbg !53
  %address.local714 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local694, !dbg !53
  store half %1580, ptr addrspace(1) %address.local714, align 2, !dbg !53
  %normalized40 = fmul <4 x float> %row.reciprocal, %1290, !dbg !51
  %1599 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !52
  %1600 = extractelement <4 x half> %1599, i64 3, !dbg !52
  %1601 = extractelement <4 x half> %1599, i64 2, !dbg !52
  %1602 = extractelement <4 x half> %1599, i64 1, !dbg !52
  %1603 = extractelement <4 x half> %1599, i64 0, !dbg !52
  %normalized41 = fmul <4 x float> %row.reciprocal, %1257, !dbg !51
  %1604 = fptrunc <4 x float> %normalized41 to <4 x half>, !dbg !52
  %1605 = extractelement <4 x half> %1604, i64 3, !dbg !52
  %1606 = extractelement <4 x half> %1604, i64 2, !dbg !52
  %1607 = extractelement <4 x half> %1604, i64 1, !dbg !52
  %1608 = extractelement <4 x half> %1604, i64 0, !dbg !52
  %normalized43 = fmul <4 x float> %row.reciprocal30, %1224, !dbg !51
  %1609 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !52
  %1610 = extractelement <4 x half> %1609, i64 3, !dbg !52
  %1611 = extractelement <4 x half> %1609, i64 2, !dbg !52
  %1612 = extractelement <4 x half> %1609, i64 1, !dbg !52
  %1613 = extractelement <4 x half> %1609, i64 0, !dbg !52
  %normalized44 = fmul <4 x float> %row.reciprocal30, %1191, !dbg !51
  %1614 = fptrunc <4 x float> %normalized44 to <4 x half>, !dbg !52
  %1615 = extractelement <4 x half> %1614, i64 3, !dbg !52
  %1616 = extractelement <4 x half> %1614, i64 2, !dbg !52
  %1617 = extractelement <4 x half> %1614, i64 1, !dbg !52
  %1618 = extractelement <4 x half> %1614, i64 0, !dbg !52
  %address.local729 = or disjoint i64 %address.local617, 64, !dbg !53
  %address.local730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local729, !dbg !53
  store half %1618, ptr addrspace(1) %address.local730, align 2, !dbg !53
  %address.local732 = or disjoint i64 %address.local617, 65, !dbg !53
  %address.local733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local732, !dbg !53
  store half %1617, ptr addrspace(1) %address.local733, align 2, !dbg !53
  %address.local735 = or disjoint i64 %address.local617, 66, !dbg !53
  %address.local736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local735, !dbg !53
  store half %1616, ptr addrspace(1) %address.local736, align 2, !dbg !53
  %address.local738 = or disjoint i64 %address.local617, 67, !dbg !53
  %address.local739 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local738, !dbg !53
  store half %1615, ptr addrspace(1) %address.local739, align 2, !dbg !53
  %address.local741 = or disjoint i64 %address.local617, 80, !dbg !53
  %address.local742 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local741, !dbg !53
  store half %1613, ptr addrspace(1) %address.local742, align 2, !dbg !53
  %address.local744 = or disjoint i64 %address.local617, 81, !dbg !53
  %address.local745 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local744, !dbg !53
  store half %1612, ptr addrspace(1) %address.local745, align 2, !dbg !53
  %address.local747 = or disjoint i64 %address.local617, 82, !dbg !53
  %address.local748 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local747, !dbg !53
  store half %1611, ptr addrspace(1) %address.local748, align 2, !dbg !53
  %address.local750 = or disjoint i64 %address.local617, 83, !dbg !53
  %address.local751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local750, !dbg !53
  store half %1610, ptr addrspace(1) %address.local751, align 2, !dbg !53
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local729, !dbg !53
  store half %1608, ptr addrspace(1) %address.local756, align 2, !dbg !53
  %address.local758 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local732, !dbg !53
  store half %1607, ptr addrspace(1) %address.local758, align 2, !dbg !53
  %address.local760 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local735, !dbg !53
  store half %1606, ptr addrspace(1) %address.local760, align 2, !dbg !53
  %address.local762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local738, !dbg !53
  store half %1605, ptr addrspace(1) %address.local762, align 2, !dbg !53
  %address.local764 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local741, !dbg !53
  store half %1603, ptr addrspace(1) %address.local764, align 2, !dbg !53
  %address.local766 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local744, !dbg !53
  store half %1602, ptr addrspace(1) %address.local766, align 2, !dbg !53
  %address.local768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local747, !dbg !53
  store half %1601, ptr addrspace(1) %address.local768, align 2, !dbg !53
  %address.local770 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local750, !dbg !53
  store half %1600, ptr addrspace(1) %address.local770, align 2, !dbg !53
  %normalized46 = fmul <4 x float> %row.reciprocal, %1158, !dbg !51
  %1619 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !52
  %1620 = extractelement <4 x half> %1619, i64 3, !dbg !52
  %1621 = extractelement <4 x half> %1619, i64 2, !dbg !52
  %1622 = extractelement <4 x half> %1619, i64 1, !dbg !52
  %1623 = extractelement <4 x half> %1619, i64 0, !dbg !52
  %normalized47 = fmul <4 x float> %row.reciprocal, %1125, !dbg !51
  %1624 = fptrunc <4 x float> %normalized47 to <4 x half>, !dbg !52
  %1625 = extractelement <4 x half> %1624, i64 3, !dbg !52
  %1626 = extractelement <4 x half> %1624, i64 2, !dbg !52
  %1627 = extractelement <4 x half> %1624, i64 1, !dbg !52
  %1628 = extractelement <4 x half> %1624, i64 0, !dbg !52
  %normalized49 = fmul <4 x float> %row.reciprocal30, %1092, !dbg !51
  %1629 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !52
  %1630 = extractelement <4 x half> %1629, i64 3, !dbg !52
  %1631 = extractelement <4 x half> %1629, i64 2, !dbg !52
  %1632 = extractelement <4 x half> %1629, i64 1, !dbg !52
  %1633 = extractelement <4 x half> %1629, i64 0, !dbg !52
  %normalized50 = fmul <4 x float> %row.reciprocal30, %1059, !dbg !51
  %1634 = fptrunc <4 x float> %normalized50 to <4 x half>, !dbg !52
  %1635 = extractelement <4 x half> %1634, i64 3, !dbg !52
  %1636 = extractelement <4 x half> %1634, i64 2, !dbg !52
  %1637 = extractelement <4 x half> %1634, i64 1, !dbg !52
  %1638 = extractelement <4 x half> %1634, i64 0, !dbg !52
  %address.local785 = or disjoint i64 %address.local617, 96, !dbg !53
  %address.local786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local785, !dbg !53
  store half %1638, ptr addrspace(1) %address.local786, align 2, !dbg !53
  %address.local788 = or disjoint i64 %address.local617, 97, !dbg !53
  %address.local789 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local788, !dbg !53
  store half %1637, ptr addrspace(1) %address.local789, align 2, !dbg !53
  %address.local791 = or disjoint i64 %address.local617, 98, !dbg !53
  %address.local792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local791, !dbg !53
  store half %1636, ptr addrspace(1) %address.local792, align 2, !dbg !53
  %address.local794 = or disjoint i64 %address.local617, 99, !dbg !53
  %address.local795 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local794, !dbg !53
  store half %1635, ptr addrspace(1) %address.local795, align 2, !dbg !53
  %address.local797 = or disjoint i64 %address.local617, 112, !dbg !53
  %address.local798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local797, !dbg !53
  store half %1633, ptr addrspace(1) %address.local798, align 2, !dbg !53
  %address.local800 = or disjoint i64 %address.local617, 113, !dbg !53
  %address.local801 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local800, !dbg !53
  store half %1632, ptr addrspace(1) %address.local801, align 2, !dbg !53
  %address.local803 = or disjoint i64 %address.local617, 114, !dbg !53
  %address.local804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local803, !dbg !53
  store half %1631, ptr addrspace(1) %address.local804, align 2, !dbg !53
  %address.local806 = or i64 %1577, 115, !dbg !53
  %address.local807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local615, i64 %address.local806, !dbg !53
  store half %1630, ptr addrspace(1) %address.local807, align 2, !dbg !53
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local785, !dbg !53
  store half %1628, ptr addrspace(1) %address.local812, align 2, !dbg !53
  %address.local814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local788, !dbg !53
  store half %1627, ptr addrspace(1) %address.local814, align 2, !dbg !53
  %address.local816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local791, !dbg !53
  store half %1626, ptr addrspace(1) %address.local816, align 2, !dbg !53
  %address.local818 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local794, !dbg !53
  store half %1625, ptr addrspace(1) %address.local818, align 2, !dbg !53
  %address.local820 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local797, !dbg !53
  store half %1623, ptr addrspace(1) %address.local820, align 2, !dbg !53
  %address.local822 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local800, !dbg !53
  store half %1622, ptr addrspace(1) %address.local822, align 2, !dbg !53
  %address.local824 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local803, !dbg !53
  store half %1621, ptr addrspace(1) %address.local824, align 2, !dbg !53
  %address.local826 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local643, i64 %address.local806, !dbg !53
  store half %1620, ptr addrspace(1) %address.local826, align 2, !dbg !53
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
